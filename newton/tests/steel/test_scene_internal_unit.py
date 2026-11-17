# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The internal unit of SolverSteel's scene solve (``solve_mode="scene"``), and joint friction.

An articulation's *internal unit* holds its rows of Steel's per-articulation row families, in Steel's
slot order (drives, mimic, connect, joint limits, joint friction): each family's COUNT pass counts the
articulation's rows into ``_scene_family_rows``, the unit carries their sum (``_scene_internal_rows``),
and the families' WRITE passes write them from the unit's first row, one family after another. The
limit family is the scene's limits unit of before, unchanged: its rows, their order and their counts.

Joint friction (``enable_joint_friction``): a DOF of a PRISMATIC, REVOLUTE, BALL or D6 joint with
positive :attr:`~newton.Model.joint_friction` ``f`` gets one row of ``J = 1`` on the DOF whose impulse
the sweeps clamp to the box ``[-f dt, f dt]``, Steel's law. The bound is ``SceneSolve.row_bound``.
Steel builds these rows on its dense response storage (its sparse path refuses joint friction), so the
scene is compared with Steel's matrix-free solve within a tolerance, and the bounds, which hold no
``J``, bit for bit.

These tests sit beside ``test_scene_features.py`` (which holds the other per-row features) to keep that
module near its size; the drive family's are in ``test_scene_drives.py``.
"""

import difflib
import hashlib
import itertools
import re
import unittest
from functools import partial

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel import scene_rows, scene_solve, sparse_contact
from newton._src.solvers.steel.kernels import PGS_CONSTRAINT_TYPE_JOINT_FRICTION
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import DT
from newton.tests.steel._scene_fixtures import (
    DRIVE_ROWS_SNIPPET_SHA256,
    FRICTION_ROWS_SNIPPET_SHA256,
    FUSED_DRIVE_ROWS_SNIPPET_SHA256,
    LIMIT_ROWS_SNIPPET_SHA256,
    MIMIC_ROWS_SNIPPET_SHA256,
    SCENE,
    VELOCITY_LIMIT_ROWS_SNIPPET_SHA256,
    _humanoids,
    _Run,
)
from newton.tests.steel.test_scene_features import (
    CROWD_ROWS,
    HUMANOID_ROWS,
    TRACK_STEPS,
    _assert_phases_match_serial_oracle,
    _track_steel,
    _within_track_tolerance,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

FRICTION = 0.5
"""``joint_friction`` [N·m] of every hinge of the tracking tests."""

HUMANOID_HINGES = 21
"""Hinges of one ``nv_humanoid.xml`` humanoid (27 DOFs less its free root's 6), each one DOF of a REVOLUTE
joint or an axis of a D6 joint (the importer merges a body's hinges into one D6 joint)."""

REGULARIZATION = 0.5
"""``contact_regularization`` of the oracle test's combined feature set."""

CAPTURE_STEPS = 60
"""Steps of the capture test: four phases of 15 steps with friction on every hinge, every other hinge,
no hinge and every hinge again."""


def _hinge_dofs(model) -> np.ndarray:
    """The DOFs of the model's REVOLUTE and D6 joints, the humanoids' hinges, in DOF order."""
    joint_type = model.joint_type.numpy()
    starts = model.joint_qd_start.numpy()
    hinges = np.flatnonzero(np.isin(joint_type, (int(newton.JointType.REVOLUTE), int(newton.JointType.D6))))
    return np.concatenate([np.arange(starts[joint], starts[joint + 1]) for joint in hinges])


def _set_friction(model, dofs, values) -> np.ndarray:
    """Give ``dofs`` the friction ``values`` and every other DOF none, in place; return the coefficients."""
    friction = np.zeros(model.joint_dof_count, dtype=np.float32)
    friction[dofs] = values
    model.joint_friction.assign(friction)
    return friction


def _at_rest(model, state):
    """Start from the model's pose at rest: gravity alone moves the humanoids."""


def _friction_rows(solver) -> tuple[np.ndarray, np.ndarray]:
    """The impulses and bounds of the scene's joint-friction rows of the last step, in row order."""
    rows = int(solver._constraint_count.numpy()[0])
    friction = solver._row_type.numpy()[0, :rows] == PGS_CONSTRAINT_TYPE_JOINT_FRICTION
    return solver._impulses.numpy()[0, :rows][friction], solver._scene.row_bound.numpy()[0, :rows][friction]


def _steel_friction_bounds(solver) -> np.ndarray:
    """Steel's joint-friction bounds (its rows' ``mu``), world by world, each world's rows in row order."""
    row_type = solver._row_type.numpy()
    row_mu = solver._row_mu.numpy()
    bounds = []
    for world, rows in enumerate(solver._constraint_count.numpy()):
        friction = row_type[world, :rows] == PGS_CONSTRAINT_TYPE_JOINT_FRICTION
        bounds.append(row_mu[world, :rows][friction])
    return np.concatenate(bounds)


def test_joint_friction_tracks_steel(test, device):
    """Hold the hinges of two lying humanoids with dry friction, as Steel's matrix-free solve does.

    Every hinge of ``_humanoids(worlds=2)`` has ``joint_friction`` 0.5 N·m. From rest under gravity, the
    velocities after the first step agree with Steel's matrix-free solve within 1e-5 (and 1e-6), and the
    joint coordinates after 30 steps within 1e-3 RMS. Every hinge has its row, in the scene as in Steel,
    and the box acts both ways: some rows end a step on their bound and some inside it. The same scene
    without joint friction falls outside that tolerance of Steel's frictional solve, so the test sees the
    friction rows.
    """
    model = _humanoids(device, worlds=2)
    _set_friction(model, _hinge_dofs(model), FRICTION)
    scene, steel = _track_steel(test, model, init=_at_rest, rows=HUMANOID_ROWS, enable_joint_friction=True)
    solve = scene.solver._scene
    test.assertTrue(solve.features.joint_friction)
    test.assertEqual(solve.row_bound.shape, (1, solve.row_max))
    impulses, bounds = _friction_rows(scene.solver)
    test.assertEqual(impulses.size, 2 * HUMANOID_HINGES)
    test.assertEqual(_steel_friction_bounds(steel.solver).size, 2 * HUMANOID_HINGES)
    np.testing.assert_array_equal(bounds, np.float32(FRICTION) * np.float32(DT))
    test.assertTrue(np.all(np.abs(impulses) <= bounds))
    test.assertTrue(np.any(np.abs(impulses) == bounds), "no friction row reached its bound")
    test.assertTrue(np.any(np.abs(impulses) < bounds), "every friction row is on its bound")

    frictionless = _Run(model, HUMANOID_ROWS, {})
    test.assertFalse(frictionless.solver._scene.features.joint_friction)
    first = None
    for step in range(TRACK_STEPS):
        frictionless.step()
        if step == 0:
            first = frictionless.state_0.joint_qd.numpy()
    test.assertFalse(
        _within_track_tolerance(
            first, steel.first_qd, frictionless.state_0.joint_q.numpy(), steel.state_0.joint_q.numpy()
        ),
        "the scene without joint friction tracks Steel's frictional solve",
    )


def test_joint_friction_bounds_match_steel_bitwise(test, device):
    """Bound each joint-friction row by ``f dt``, Steel's dense rows' ``mu``, bit for bit and in Steel's order.

    Two lying humanoids, each hinge with its own coefficient (one hinge per world without friction, which
    builds no row), step once from the same state and contacts. Over the joint-friction rows, the scene's
    ``row_bound`` equals Steel's dense rows' ``mu`` world by world (``np.array_equal``), and both equal
    ``float32(f) * float32(dt)`` of each hinge with friction, in DOF order. The scene's ``row_mu`` of those
    rows holds the bound too, as Steel's. The joint-friction rows count in Steel's articulation family per
    model world: the row high-water marks equal Steel's. The internal unit's rows are its families' sum.
    """
    model = _humanoids(device, worlds=2)
    hinges = _hinge_dofs(model)
    values = (0.25 + 0.01 * np.arange(hinges.size)).astype(np.float32)
    # The fourth hinge of each world builds no row.
    values[[3, 3 + HUMANOID_HINGES]] = 0.0
    friction = _set_friction(model, hinges, values)

    def initial_state():
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        return state

    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    pipeline.collide(initial_state(), contacts)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)

    solvers = {}
    for name, options in (("scene", {}), ("steel", {"solve_mode": "serial"})):
        solver = SolverSteel(
            model,
            **{**SCENE, **options},
            articulation_constraint_max=HUMANOID_ROWS,
            enable_joint_friction=True,
            row_watermark=True,
        )
        test.assertIs(solver.solve_mode == "scene", name == "scene")
        solver.step(initial_state(), model.state(), model.control(), contacts, DT)
        test.assertFalse(np.any(solver.constraint_overflow.numpy()), f"{name} overflowed")
        solvers[name] = solver

    scene = solvers["scene"]
    rows = int(scene._constraint_count.numpy()[0])
    is_friction = scene._row_type.numpy()[0, :rows] == PGS_CONSTRAINT_TYPE_JOINT_FRICTION
    scene_bounds = scene._scene.row_bound.numpy()[0, :rows][is_friction]
    steel_bounds = _steel_friction_bounds(solvers["steel"])
    expected = friction[hinges][friction[hinges] > 0.0] * np.float32(DT)
    test.assertEqual(scene_bounds.size, 2 * (HUMANOID_HINGES - 1))
    test.assertTrue(np.array_equal(scene_bounds, steel_bounds), f"{scene_bounds} != {steel_bounds}")
    test.assertTrue(np.array_equal(scene_bounds, expected))
    test.assertTrue(np.array_equal(scene._row_mu.numpy()[0, :rows][is_friction], scene_bounds))

    marks = scene._row_watermarks.numpy()
    test.assertTrue(
        np.array_equal(marks, solvers["steel"]._row_watermarks.numpy()),
        f"{scene.constraint_row_watermarks()} != {solvers['steel'].constraint_row_watermarks()}",
    )
    family_rows = scene._scene_family_rows.numpy()
    internal = scene._scene_internal_rows.numpy()
    np.testing.assert_array_equal(family_rows[scene_rows.FAMILY_JOINT_FRICTION], HUMANOID_HINGES - 1)
    np.testing.assert_array_equal(internal, family_rows.sum(axis=0))
    np.testing.assert_array_equal(scene._scene_limit_rows.numpy(), family_rows[scene_rows.FAMILY_LIMIT])


class _Stepper:
    """A scene solver stepping fixed state buffers, so that one captured step replays every later step."""

    def __init__(self, model, **options):
        self.solver = SolverSteel(model, articulation_constraint_max=HUMANOID_ROWS, **{**SCENE, **options})
        self.pipeline = newton.CollisionPipeline(model, deterministic=True)
        self.contacts = self.pipeline.contacts()
        self.state_0, self.state_1 = model.state(), model.state()
        self.control = model.control()
        newton.eval_fk(model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def step(self):
        self.pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, DT)
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            wp.copy(getattr(self.state_0, name), getattr(self.state_1, name))


def test_internal_unit_rows_change_under_capture(test, device):
    """Replay one captured step exactly as eager steps while the internal units' rows appear and disappear.

    Two standing humanoids fall with joint limits and joint friction: their limit rows change with the
    pose, and the hinges' friction is edited in place between steps (every hinge, every other hinge, none,
    every hinge again), so the friction rows disappear and come back while one captured graph replays the
    step. The knees and the hips' x axes start within the activation gap of their upper bound, so their
    limit rows and their friction rows share the internal unit from the first step. ``joint_q`` and
    ``joint_qd`` equal the eager steps' at every step, the joint-friction family's rows follow the
    edits, and the eager row count takes at least three values.
    """
    model = _humanoids(device, worlds=2, lying=False)
    hinges = _hinge_dofs(model)
    schedule = (hinges, hinges[::2], hinges[:0], hinges)
    phase_steps = CAPTURE_STEPS // len(schedule)
    _set_friction(model, hinges, FRICTION)
    eager = _Stepper(model, enable_joint_friction=True)
    captured = _Stepper(model, enable_joint_friction=True)

    def assert_equal(step):
        for name in ("joint_q", "joint_qd"):
            a, b = getattr(eager.state_0, name).numpy(), getattr(captured.state_0, name).numpy()
            test.assertTrue(
                np.array_equal(a, b), f"{name} differs after step {step}: max |diff| {np.max(np.abs(a - b)):.3e}"
            )

    row_counts, friction_counts = set(), []
    eager.step()
    captured.step()
    assert_equal(1)
    # Capturing records the step without running it.
    with wp.ScopedCapture(device) as recorded:
        captured.step()
    for index in range(1, CAPTURE_STEPS):
        if index % phase_steps == 0:
            _set_friction(model, schedule[index // phase_steps], FRICTION)
        eager.step()
        wp.capture_launch(recorded.graph)
        assert_equal(index + 1)
        row_counts.add(eager.solver.scene_statistics()["rows"])
        family_rows = eager.solver._scene_family_rows.numpy()
        friction_counts.append(int(family_rows[scene_rows.FAMILY_JOINT_FRICTION].sum()))
        np.testing.assert_array_equal(eager.solver._scene_internal_rows.numpy(), family_rows.sum(axis=0))
        test.assertGreater(int(family_rows[scene_rows.FAMILY_LIMIT].sum()), 0)
    expected = [schedule[index // phase_steps].size for index in range(1, CAPTURE_STEPS)]
    test.assertEqual(friction_counts, expected)
    test.assertGreaterEqual(len(row_counts), 3, msg=f"row counts {sorted(row_counts)}")
    test.assertTrue(np.all(np.isfinite(captured.state_0.joint_q.numpy())))
    test.assertFalse(np.any(eager.solver.constraint_overflow.numpy()))


def test_joint_friction_matches_the_serial_oracle(test, device):
    """Clamp the joint-friction rows in the phased sweep exactly as one thread walking the rows does.

    Four lying humanoids 0.4 m apart overlap, so units join pairs of them; every hinge has friction
    0.5 N·m, and some friction rows end on their bound. The phases (the default color loop, and the while
    loop, whose tail one thread walks) match the serial oracle (``scene_sweep="serial"``) bit for bit
    over 30 steps in each phase order, so the lane-group loop's box clamp is the one-thread loop's. They
    do with joint friction alone (the ``_jf`` kernels) and with every implemented feature on (``_fpi_reg_jf``:
    friction position iterations, ``contact_regularization=0.5`` and joint friction), whose featured
    arguments all carry data.
    """
    model = _humanoids(device, per_world=4, spacing=0.4)
    _set_friction(model, _hinge_dofs(model), FRICTION)

    def saturated(solver):
        impulses, bounds = _friction_rows(solver)
        return bool(np.any(np.abs(impulses) == bounds))

    feature_sets = {
        "_jf": {"enable_joint_friction": True},
        "_fpi_reg_jf": {
            "enable_joint_friction": True,
            "contact_regularization": REGULARIZATION,
            "iterations": 6,
            "contact_friction_position_iterations": 2,
        },
    }
    for suffix, options in feature_sets.items():
        with test.subTest(features=suffix):
            probe = _Run(model, CROWD_ROWS, {}, collide=False, **options)
            test.assertTrue(probe.solver._scene.sides_kernel.key.endswith(suffix))
            _assert_phases_match_serial_oracle(
                test, model, options, witness=saturated, what="no friction row reached its bound"
            )


def test_internal_units_count_toward_the_unit_capacity(test, device):
    """Size the units with one internal unit per articulation that can have internal rows, whatever its rows.

    Two humanoids with joint friction or drive rows and no joint limits: each has hinges, so each can have an
    internal unit, and ``unit_max`` is a third of the rows plus 2; with joint limits, drive rows and joint
    friction together it is still 2 (one unit holds every family). Without joint friction and drive rows the
    internal units are the limits units. With joint velocity limits each humanoid can also have a late unit (its
    hinges' velocity limits are positive), which adds 2 more, whether or not it has internal units; its bit
    (``FAMILY_VELOCITY_LIMIT``) is not an internal family's. The solver's one per-articulation flag records which
    families each articulation can bear (``_scene_family_bearing``), which the units' capacity, the factor groups'
    families and their late rows read. A unit build takes the internal rows under either name, not both, and late
    rows only when an articulation can bear them.
    """
    model = _humanoids(device, worlds=2)
    rows = 2 * HUMANOID_ROWS
    drive, limit, friction = scene_rows.FAMILY_DRIVE, scene_rows.FAMILY_LIMIT, scene_rows.FAMILY_JOINT_FRICTION
    late = scene_rows.FAMILY_VELOCITY_LIMIT
    velocity_limits = {"enable_joint_velocity_limits": True}
    cases = (
        ({"enable_joint_limits": False, "enable_joint_friction": True}, 0, 2, 0, (friction,)),
        ({"enable_joint_friction": True}, 2, 2, 0, (limit, friction)),
        ({}, 2, 2, 0, (limit,)),
        ({"enable_joint_limits": False, "drive_mode": "rows"}, 0, 2, 0, (drive,)),
        ({"drive_mode": "rows"}, 2, 2, 0, (drive, limit)),
        ({"drive_mode": "rows", "enable_joint_friction": True}, 2, 2, 0, (drive, limit, friction)),
        ({"enable_joint_limits": False, **velocity_limits}, 0, 0, 2, ()),
        ({**velocity_limits}, 2, 2, 2, (limit,)),
        ({"drive_mode": "rows", "enable_joint_friction": True, **velocity_limits}, 2, 2, 2, (drive, limit, friction)),
    )
    for options, limit_units, internal_units, late_units, families in cases:
        if device.is_cpu and options.get("enable_joint_friction"):
            # Steel refuses joint friction on the CPU, and so does the scene there (Steel's checks run first).
            continue
        with test.subTest(**options):
            solver = SolverSteel(model, articulation_constraint_max=HUMANOID_ROWS, **{**SCENE, **options})
            units = solver._scene_units
            test.assertEqual(units.limit_unit_count, limit_units)
            test.assertEqual(units.internal_unit_count, internal_units)
            test.assertEqual(units.late_unit_count, late_units)
            test.assertEqual(units.unit_max, rows // 3 + internal_units + late_units)
            test.assertEqual(solver._scene.internal_families, families)
            test.assertEqual(solver._scene.late_rows_in_use, late_units > 0)
            bits = sum(1 << family for family in families) | ((1 << late) if late_units else 0)
            np.testing.assert_array_equal(solver._scene_family_bearing, bits)
            np.testing.assert_array_equal(units.internal_bearing_articulations(solver), int(internal_units > 0))
            np.testing.assert_array_equal(units.late_bearing_articulations(solver), int(late_units > 0))
    with test.assertRaisesRegex(ValueError, "late_rows needs an articulation"):
        solver = SolverSteel(model, articulation_constraint_max=HUMANOID_ROWS, **SCENE)
        solver._scene_units.build(solver, None, late_rows=solver._scene_late_rows)
    # Internal rows beside the limit rows: the drive rows' family (joint friction's on the CPU is refused).
    solver = SolverSteel(model, articulation_constraint_max=HUMANOID_ROWS, **{**SCENE, "drive_mode": "rows"})
    with test.assertRaisesRegex(ValueError, "internal_rows or limit_rows"):
        solver._scene_units.build(
            solver, None, internal_rows=solver._scene_internal_rows, limit_rows=solver._scene_limit_rows
        )
    with test.assertRaisesRegex(ValueError, r"^limit_rows must have shape"):
        solver._scene_units.build(solver, None, limit_rows=solver._scene_family_rows)
    with test.assertRaisesRegex(ValueError, r"^internal_rows must have shape"):
        solver._scene_units.build(solver, None, internal_rows=solver._scene_family_rows)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _site_lines(sites: dict[str, str]) -> set[str]:
    """The lines a row family's sites own in its expanded builder.

    Each line of the template that holds a family site (``$FAMILY_*``), expanded with ``sites``. A block site (its
    text empty or ending a line) begins its template line: it owns its own lines, not the template text that
    follows it on that line (``$FAMILY_CANDIDATE_HEAD`` does not own ``const int dof = start + local_dof;``). An
    in-line site owns the template line that holds it with that site substituted (so an in-line row type ``3``
    owns ``row_type.data[row_offset] = 3;``, not every line holding a ``3``). No such template line holds a mask
    site or a ``ROW_SLOTS`` token in an in-line site's line, so neither expansion changes them.
    """
    owned = set()
    for template_line in scene_rows._JOINT_DOF_ROWS_TEMPLATE.splitlines(keepends=True):
        if "$FAMILY_" not in template_line:
            continue
        assert "$MASK_" not in template_line, template_line
        block = re.match(r"\$FAMILY_[A-Z_]+", template_line)
        if block is not None and (sites[block.group()] == "" or sites[block.group()].endswith("\n")):
            assert "$FAMILY_" not in template_line[block.end() :], template_line
            owned.update(sites[block.group()].splitlines())
            continue
        assert "$ROW_SLOTS" not in template_line, template_line
        expanded = template_line
        for placeholder in sorted(sites, key=len, reverse=True):
            expanded = expanded.replace(placeholder, sites[placeholder])
        owned.update(expanded.splitlines())
    return owned


class TestSteelSceneInternalUnit(unittest.TestCase):
    def test_row_families_expand_one_builder_template(self):
        """Build every per-articulation row family from one builder template, the joint limits' text unchanged.

        The drive (with and without the fused velocity clamp), joint-limit, joint-friction, velocity-limit and mimic
        row builders are expansions of one template (``_JOINT_DOF_ROWS_TEMPLATE``), each family filling every family
        site, and the mimic family its own support mask's load (the OR of its two DOFs' masks) too. Each family's
        snippet for one and two mask words hashes to its pinned text (``_scene_fixtures``); the joint-limit
        builder's one-word snippet is Steel's builder's, character for character. The joint-limit family's
        kernel for one word is ``build_sparse_joint_limit_rows`` itself. Every line in which two expansions differ
        is a line its family's sites own: a line of a site's text, or the template line of an in-line site with
        that site filled in (the mimic family's mask load among its own). The driven-DOF marker's candidate lines
        are the drive family's own sites. The mimic family's builder is the dispatcher's; the connect family has
        none there (its rows go through the contact response).
        """
        families = {
            "limit": (scene_rows.sparse_joint_limit_rows_snippet, scene_rows._LIMIT_FAMILY_SITES),
            "friction": (scene_rows.sparse_joint_friction_rows_snippet, scene_rows._FRICTION_FAMILY_SITES),
            "drive": (scene_rows.sparse_drive_rows_snippet, scene_rows._DRIVE_FAMILY_SITES),
            "fused_drive": (
                partial(scene_rows.sparse_drive_rows_snippet, fused=True),
                scene_rows._FUSED_DRIVE_FAMILY_SITES,
            ),
            "velocity_limit": (
                scene_rows.sparse_joint_velocity_limit_rows_snippet,
                scene_rows._VELOCITY_LIMIT_FAMILY_SITES,
            ),
            "mimic": (scene_rows.sparse_mimic_rows_snippet, scene_rows._MIMIC_FAMILY_SITES),
        }
        hashes = {
            "limit": LIMIT_ROWS_SNIPPET_SHA256,
            "friction": FRICTION_ROWS_SNIPPET_SHA256,
            "drive": DRIVE_ROWS_SNIPPET_SHA256,
            "fused_drive": FUSED_DRIVE_ROWS_SNIPPET_SHA256,
            "velocity_limit": VELOCITY_LIMIT_ROWS_SNIPPET_SHA256,
            "mimic": MIMIC_ROWS_SNIPPET_SHA256,
        }
        owned = {name: _site_lines(sites) for name, (_snippet, sites) in families.items()}
        # Each family owns its support mask's load: the mimic family's is the OR of its two DOFs' masks.
        for name in families:
            mask_sites = scene_rows._MIMIC_MASK_SITES if name == "mimic" else scene_rows._JOINT_DOF_ROWS_MASK_SITES
            for text in mask_sites["$MASK_LOAD"]:
                owned[name].update(text.splitlines())
        for words in (1, 2):
            texts = {name: snippet(words) for name, (snippet, _sites) in families.items()}
            for name, text in texts.items():
                with self.subTest(mask_words=words, family=name):
                    self.assertEqual(_sha256(text), hashes[name][words])
            for first, second in itertools.combinations(families, 2):
                with self.subTest(mask_words=words, pair=(first, second)):
                    first_lines, second_lines = texts[first].splitlines(), texts[second].splitlines()
                    matcher = difflib.SequenceMatcher(None, first_lines, second_lines, autojunk=False)
                    shared = 0
                    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                        if tag == "equal":
                            shared += i2 - i1
                            continue
                        for line in first_lines[i1:i2]:
                            self.assertIn(line, owned[first])
                        for line in second_lines[j1:j2]:
                            self.assertIn(line, owned[second])
                    self.assertGreater(shared, 0.75 * len(first_lines))
        # An in-line site owns its own line only: the row type 3 does not own a line merely holding a 3; a block
        # site owns its own lines, not the template text after it on its line.
        self.assertIn("                row_type.data[row_offset] = 3;", owned["limit"])
        self.assertNotIn("    const int lane = tid & 31;", owned["limit"])
        for name in families:
            self.assertNotIn("        const int dof = start + local_dof;", owned[name], name)
        marker = scene_rows.sparse_drive_marker_snippet()
        drive_sites = scene_rows._DRIVE_FAMILY_SITES
        self.assertIn(
            drive_sites["$FAMILY_CANDIDATE_VALUES"] + "        int active = 0;\n" + drive_sites["$FAMILY_LAW"], marker
        )
        template = scene_rows._JOINT_DOF_ROWS_TEMPLATE
        for _snippet, sites in families.values():
            missing = dict(sites)
            missing.pop("$FAMILY_LAW")
            with self.assertRaisesRegex(AssertionError, "family sites"):
                scene_rows._expand_family(template, missing)
        self.assertIs(
            scene_rows.get_sparse_joint_dof_rows_kernel(scene_rows.FAMILY_LIMIT),
            sparse_contact.build_sparse_joint_limit_rows,
        )
        self.assertIs(
            scene_rows.get_sparse_joint_dof_rows_kernel(scene_rows.FAMILY_LIMIT, 2),
            scene_rows.get_sparse_joint_limit_rows_kernel(2),
        )
        self.assertEqual(
            scene_rows.get_sparse_joint_dof_rows_kernel(scene_rows.FAMILY_JOINT_FRICTION, 2).key,
            "build_sparse_joint_friction_rows_w2",
        )
        self.assertIs(
            scene_rows.get_sparse_joint_dof_rows_kernel(scene_rows.FAMILY_DRIVE, 2),
            scene_rows.get_sparse_drive_rows_kernel(2),
        )
        self.assertEqual(
            scene_rows.get_sparse_joint_dof_rows_kernel(scene_rows.FAMILY_DRIVE).key, "build_sparse_drive_rows_w1"
        )
        self.assertEqual(
            scene_rows.get_sparse_joint_dof_rows_kernel(scene_rows.FAMILY_DRIVE, 2, fused=True).key,
            "build_sparse_drive_rows_fused_w2",
        )
        self.assertIs(
            scene_rows.get_sparse_joint_dof_rows_kernel(scene_rows.FAMILY_VELOCITY_LIMIT, 2),
            scene_rows.get_sparse_joint_velocity_limit_rows_kernel(2),
        )
        self.assertEqual(
            scene_rows.get_sparse_joint_dof_rows_kernel(scene_rows.FAMILY_VELOCITY_LIMIT).key,
            "build_sparse_joint_velocity_limit_rows_w1",
        )
        with self.assertRaisesRegex(ValueError, "drive family's build"):
            scene_rows.get_sparse_joint_dof_rows_kernel(scene_rows.FAMILY_LIMIT, fused=True)
        self.assertEqual(scene_solve.SCENE_ROW_FAMILIES, tuple(range(scene_rows.FAMILY_COUNT)))
        self.assertIs(
            scene_rows.get_sparse_joint_dof_rows_kernel(scene_rows.FAMILY_MIMIC, 2),
            scene_rows.get_sparse_mimic_rows_kernel(2),
        )
        self.assertEqual(
            scene_rows.get_sparse_joint_dof_rows_kernel(scene_rows.FAMILY_MIMIC).key, "build_sparse_mimic_rows_w1"
        )
        # The connect family's rows are built through the contact response, not by a factor group's builder.
        with self.assertRaisesRegex(ValueError, "get_scene_connect_rows_kernel"):
            scene_rows.get_sparse_joint_dof_rows_kernel(scene_rows.FAMILY_CONNECT)


add_function_test(
    TestSteelSceneInternalUnit,
    test_internal_units_count_toward_the_unit_capacity.__name__,
    test_internal_units_count_toward_the_unit_capacity,
    devices=get_test_devices(),
)

# CUDA only, each for the reason beside it: Steel's joint friction is its matrix-free solve's (its CPU split solve
# rejects it); test_scene_cpu steps the joint friction rows on the CPU against CUDA's.
for _fn in (
    test_joint_friction_tracks_steel,  # Steel's joint friction
    test_joint_friction_bounds_match_steel_bitwise,  # Steel's joint friction
    test_internal_unit_rows_change_under_capture,  # graph capture
    test_joint_friction_matches_the_serial_oracle,  # the phases; the CPU sweeps the serial oracle alone
):
    add_function_test(TestSteelSceneInternalUnit, _fn.__name__, _fn, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
