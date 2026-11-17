# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scene rows of SolverSteel (``solve_mode="scene"``): row templates, units before rows, rows by scans, status."""

import inspect
import json
import re
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel import scene_rows, scene_solve, sparse_contact
from newton._src.solvers.steel.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    _finish_connect_row,
    allocate_world_contact_slots,
    populate_connect_J_for_size,
    prepare_world_contact_rows,
)
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import CONTACT_MAX, DT, _build_contact_scene, _heap
from newton.tests.steel._scene_fixtures import SCENE, _global_box_model, _humanoids, _steel_on
from newton.tests.steel.test_scene_tripwire import _native_snippet
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

SCENE_ROW_MAX = 3 * CONTACT_MAX
"""``articulation_constraint_max`` of the scene-mode fixtures: three rows per contact of the buffer, per world."""

POISON = -7

# A lying humanoid's finite joint-limit bounds: 21 hinges, a lower and an upper bound each.
HUMANOID_LIMIT_ROWS = 42


def _initial_state(model):
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state


def _collide(model, state):
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return contacts


def _step_rows(solver, model, state, contacts):
    """Step once, through the scene rows."""
    solver.step(state, model.state(), model.control(), contacts, DT)


def _scene_solver(model, **options):
    return SolverSteel(model, **{**SCENE, "articulation_constraint_max": SCENE_ROW_MAX, **options})


def _rows(model, **options):
    """A scene-mode solver stepped once through its rows from the initial state, with the state and contacts."""
    solver = _scene_solver(model, **options)
    state = _initial_state(model)
    contacts = _collide(model, state)
    _step_rows(solver, model, state, contacts)
    return solver, state, contacts


def _unit_view(solver):
    """The step's units in sorted order, read back."""
    units = solver._scene_units
    count = int(units.unit_count.numpy()[0])
    order = units.unit_order.numpy()[:count]
    return {
        "count": count,
        "order": order,
        "contact": units.unit_contact.numpy()[order],
        "side_a": units.unit_side_a.numpy()[order],
        "side_b": units.unit_side_b.numpy()[order],
        "rows": units.unit_rows.numpy()[order],
        "row_start": solver._scene.unit_row_start.numpy()[order],
    }


def _assert_row_order(test, solver):
    """Check the row layout: one-sided units side-major, then the two-sided units by color; return the two-sided count."""
    units, scene = solver._scene_units, solver._scene
    sides = units.side_count
    view = _unit_view(solver)
    count = view["count"]
    rows, contact, row_start = view["rows"], view["contact"], view["row_start"]
    row_count = int(solver._constraint_count.numpy()[0])
    test.assertEqual(row_count, int(scene.rows_requested.numpy()[0]))
    test.assertEqual(int(rows.sum()), row_count)

    side_start = units.side_start.numpy()
    one_count = units.one_sided_count.numpy()
    one_start = scene.one_sided_start.numpy()
    seq_row_start = scene.seq_row_start.numpy()
    np.testing.assert_array_equal(one_start, np.concatenate(([0], np.cumsum(one_count))))
    one_rows = int(seq_row_start[one_start[sides]])
    # The sides' one-sided rows tile [0, one_rows) in side order.
    begin, end = scene.side_row_begin.numpy(), scene.side_row_end.numpy()
    test.assertEqual(int(begin[0]), 0)
    np.testing.assert_array_equal(begin[1:], end[:-1])
    test.assertEqual(int(end[-1]), one_rows)

    sequence = []
    for side in range(sides):
        positions = np.arange(side_start[side], side_start[side] + one_count[side])
        sequence.extend(positions.tolist())
        firsts = begin[side] + np.concatenate(([0], np.cumsum(rows[positions])[:-1])) if positions.size else []
        np.testing.assert_array_equal(row_start[positions], firsts, err_msg=f"side {side}")
        test.assertEqual(int(begin[side] + rows[positions].sum()), int(end[side]))
    two_sided = int(units.color_offsets.numpy()[-1])
    by_color = units.order_by_color.numpy()[:two_sided]
    sequence.extend(by_color.tolist())
    test.assertEqual(len(sequence), count)
    np.testing.assert_array_equal(scene.unit_sequence.numpy()[:count], sequence)
    # The two-sided units follow in color order.
    firsts = one_rows + np.concatenate(([0], np.cumsum(rows[by_color])[:-1])) if two_sided else []
    np.testing.assert_array_equal(row_start[by_color], firsts)

    # Every unit's rows are [unit_row_start, +unit_rows); a contact has three, the normal first.
    row_type = solver._row_type.numpy()[0]
    row_parent = solver._row_parent.numpy()[0]
    contact_slot = solver._contact_slot.numpy()
    is_contact = contact >= 0
    np.testing.assert_array_equal(rows[is_contact], 3)
    np.testing.assert_array_equal(contact_slot[contact[is_contact]], row_start[is_contact])
    normal = row_start[is_contact]
    np.testing.assert_array_equal(row_type[normal], PGS_CONSTRAINT_TYPE_CONTACT)
    np.testing.assert_array_equal(row_type[normal + 1], PGS_CONSTRAINT_TYPE_FRICTION)
    np.testing.assert_array_equal(row_type[normal + 2], PGS_CONSTRAINT_TYPE_FRICTION)
    np.testing.assert_array_equal(row_parent[normal], -1)
    np.testing.assert_array_equal(row_parent[normal + 1], normal)
    np.testing.assert_array_equal(row_parent[normal + 2], normal)
    for first, n in zip(row_start[~is_contact], rows[~is_contact], strict=True):
        np.testing.assert_array_equal(row_type[first : first + n], PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
    return two_sided


def _assert_rows_match_steel(test, model, state, *, gap, min_contacts):
    """Step Steel's deterministic per-world build and the scene rows on the same state and contacts, and compare."""
    contacts = _collide(model, state)
    test.assertGreaterEqual(int(contacts.rigid_contact_count.numpy()[0]), min_contacts)
    options = {**SCENE, "joint_limit_activation_gap": gap}
    # Steel's reference stays Steel's under a forced scene mode.
    steel_options = {**options, "solve_mode": "serial"}
    steel = SolverSteel(
        model,
        deterministic=wp.DeterministicMode.RUN_TO_RUN,
        articulation_constraint_max=128,
        **_steel_on(model, steel_options),
    )
    steel.step(state, model.state(), model.control(), contacts, DT)
    scene = SolverSteel(model, articulation_constraint_max=128, **options)
    _step_rows(scene, model, state, contacts)

    test.assertEqual(steel._sparse_mass_matrix_size, 27)
    test.assertEqual(steel._sparse_row_dof.shape, scene._sparse_row_dof.shape)
    test.assertFalse(np.any(steel.constraint_overflow.numpy()))
    test.assertFalse(np.any(scene.constraint_overflow.numpy()))
    row_count = int(scene._constraint_count.numpy()[0])
    test.assertEqual(int(steel._constraint_count.numpy()[0]), row_count)

    limit_rows = int(scene._scene_limit_rows.numpy()[0])
    test.assertGreater(limit_rows, 0)
    count = int(contacts.rigid_contact_count.numpy()[0])
    steel_slot = steel._contact_slot.numpy()[:count]
    scene_slot = scene._contact_slot.numpy()[:count]
    np.testing.assert_array_equal(steel_slot >= 0, scene_slot >= 0)
    with_rows = np.flatnonzero(scene_slot >= 0)
    test.assertGreaterEqual(with_rows.size, min_contacts)
    test.assertEqual(limit_rows + 3 * with_rows.size, row_count)

    steel_rows = (steel_slot[with_rows, None] + np.arange(3)).ravel()
    scene_rows = (scene_slot[with_rows, None] + np.arange(3)).ravel()
    limits = np.arange(limit_rows)
    fields = ("_sparse_row_dof", "_sparse_row_factor", "_diag", "_rhs", "_sparse_row_incident")
    fields += ("_row_type", "_row_mu", "_phi", "_target_velocity")
    for name in fields:
        expected = getattr(steel, name).numpy()[0]
        actual = getattr(scene, name).numpy()[0]
        test.assertTrue(np.array_equal(expected[steel_rows], actual[scene_rows]), msg=f"{name} of the contact rows")
        test.assertTrue(np.array_equal(expected[limits], actual[limits]), msg=f"{name} of the limit rows")
    np.testing.assert_array_equal(steel._row_type.numpy()[0][limits], PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
    steel_parent = steel._row_parent.numpy()[0]
    scene_parent = scene._row_parent.numpy()[0]
    np.testing.assert_array_equal(
        steel_parent[steel_rows] - np.repeat(steel_slot[with_rows], 3),
        scene_parent[scene_rows] - np.repeat(scene_slot[with_rows], 3),
    )
    np.testing.assert_array_equal(steel_parent[limits], scene_parent[limits])
    return limit_rows, with_rows.size


def test_rows_match_steel_rows_bitwise(test, device):
    """Build the same rows as Steel's deterministic per-world build, bit for bit, on one lying humanoid."""
    model = _humanoids(device)
    with test.subTest(pose="initial"):
        # The lying humanoid rests on its right lower arm: two contacts and the bounds within 0.1 rad.
        _assert_rows_match_steel(test, model, _initial_state(model), gap=0.1, min_contacts=1)
    with test.subTest(pose="lowered"):
        # Lowered 0.25 m into the ground the torso and legs touch too, and a 10 rad gap activates every bound.
        state = model.state()
        joint_q = model.joint_q.numpy().copy()
        joint_q[2] -= 0.25
        state.joint_q.assign(joint_q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        limit_rows, _contacts = _assert_rows_match_steel(test, model, state, gap=10.0, min_contacts=10)
        test.assertEqual(limit_rows, HUMANOID_LIMIT_ROWS)


def test_row_order_is_side_major_then_colors(test, device):
    """Lay the rows out as all one-sided units side-major in sorted order, then the two-sided units by color."""
    for name in ("heap", "humanoids"):
        with test.subTest(scene=name):
            if name == "heap":
                model = _heap(device)
            else:
                # 0.4 m apart the lying humanoids overlap by 4 cm, so units join pairs of them.
                model = _humanoids(device, per_world=4, spacing=0.4)
            solver, _state, _contacts = _rows(model)
            test.assertFalse(np.any(solver.constraint_overflow.numpy()))
            two_sided = _assert_row_order(test, solver)
            test.assertGreater(two_sided, 0)


def test_free_free_rows_have_prefix_12(test, device):
    """Give rows between two free boxes two physical 6-entry blocks, rows against the ground one, and use no MF rows."""
    solver, _state, _contacts = _rows(_heap(device))
    view = _unit_view(solver)
    prefix = solver._scene.row_free_prefix.numpy()
    row_dof = solver._sparse_row_dof.numpy()[0]
    two_sided = view["side_b"] >= 0
    test.assertTrue(np.any(two_sided))
    test.assertTrue(np.any(~two_sided))
    for first, both in zip(view["row_start"], two_sided, strict=True):
        expected = 12 if both else 6
        np.testing.assert_array_equal(prefix[first : first + 3], expected)
        np.testing.assert_array_equal(row_dof[first : first + 3, :expected] >= 0, True)
        np.testing.assert_array_equal(row_dof[first : first + 3, expected:], -1)
    test.assertFalse(np.any(solver._mf_constraint_count.numpy()))


def test_rows_overflow_in_scan_order(test, device):
    """Drop the units past the row capacity in row order, route their contacts nowhere and latch the overflow."""
    # Units for every contact, so the row capacity, not the unit capacity, drops them. The heap is one
    # world of free bodies: its 300 scene rows are the two per-world budgets, summed.
    solver, _state, contacts = _rows(
        _heap(device),
        articulation_constraint_max=100,
        free_body_constraint_max=200,
        scene_unit_max=CONTACT_MAX,
        warn_constraint_overflow=False,
    )
    test.assertEqual(solver._scene_plan.row_max, 300)
    units, scene = solver._scene_units, solver._scene
    test.assertEqual(int(units.unit_overflow.numpy()[0]), 0)
    count = int(units.unit_count.numpy()[0])
    row_count = int(solver._constraint_count.numpy()[0])
    requested = int(scene.rows_requested.numpy()[0])
    test.assertLessEqual(row_count, 300)
    test.assertGreater(requested, 300)

    seq_row_start = scene.seq_row_start.numpy()
    kept = seq_row_start[1 : count + 1] <= 300
    kept_count = int(np.count_nonzero(kept))
    # The kept units are a prefix of the row order, so the dropped ones are its suffix.
    test.assertTrue(np.all(kept[:kept_count]))
    test.assertGreater(count, kept_count)
    test.assertEqual(row_count, int(seq_row_start[kept_count]))
    order = units.unit_order.numpy()
    sequence = scene.unit_sequence.numpy()[:count]
    unit_contact = units.unit_contact.numpy()
    row_start = scene.unit_row_start.numpy()
    dropped = order[sequence[kept_count:]]
    test.assertTrue(np.all(row_start[dropped] == -1))
    test.assertTrue(np.all(row_start[order[sequence[:kept_count]]] >= 0))
    dropped_contacts = unit_contact[dropped]
    test.assertTrue(np.all(dropped_contacts >= 0))
    np.testing.assert_array_equal(solver._contact_path.numpy()[dropped_contacts], -1)
    np.testing.assert_array_equal(solver._contact_slot.numpy()[dropped_contacts], -1)
    test.assertEqual(int(scene.side_dropped.numpy().sum()), requested - row_count)
    # The model world of each side of a dropped unit is latched, the last entry for world -1. The
    # heap's boxes are global bodies, so that is the last entry alone.
    model_world = solver.model.articulation_world.numpy()
    sides = np.concatenate((units.unit_side_a.numpy()[dropped], units.unit_side_b.numpy()[dropped]))
    worlds = model_world[sides[sides >= 0]]
    overflow = solver.constraint_overflow.numpy()
    expected = np.zeros_like(overflow)
    expected[np.where(worlds < 0, overflow.size - 1, worlds)] = True
    np.testing.assert_array_equal(overflow, expected)
    np.testing.assert_array_equal(overflow, [False, True])
    test.assertEqual(int(scene.overflow.numpy()[0]), 1)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_unit_overflow_latches_every_world(test, device):
    """Latch every status entry when units pass the unit capacity; contacts without a unit get no rows."""
    model = _build_contact_scene(device, worlds=2)
    model.rigid_contact_max = CONTACT_MAX
    solver, _state, contacts = _rows(model, scene_unit_max=16, warn_constraint_overflow=False)
    units = solver._scene_units
    test.assertGreater(int(units.unit_overflow.numpy()[0]), 0)
    test.assertEqual(int(units.unit_count.numpy()[0]), 16)
    test.assertEqual(solver.constraint_overflow.shape, (3,))
    test.assertTrue(np.all(solver.constraint_overflow.numpy()))
    count = int(contacts.rigid_contact_count.numpy()[0])
    with_unit = np.zeros(count, dtype=bool)
    unit_contact = _unit_view(solver)["contact"]
    with_unit[unit_contact[unit_contact >= 0]] = True
    test.assertFalse(np.all(with_unit))
    contact_slot = solver._contact_slot.numpy()[:count]
    np.testing.assert_array_equal(contact_slot[~with_unit], -1)
    test.assertTrue(np.all(contact_slot[with_unit] >= 0))


def test_global_body_couples_model_worlds(test, device):
    """Solve a global body against the bodies of every model world in the one solve world; Steel drops and flags them."""
    model = _global_box_model(device)
    test.assertEqual(model.articulation_world.numpy().tolist(), [0, 1, -1])
    solver, _state, _contacts = _rows(model)
    view = _unit_view(solver)
    two_sided = view["side_b"] >= 0
    pairs = {(int(a), int(b)) for a, b in zip(view["side_a"][two_sided], view["side_b"][two_sided], strict=True)}
    test.assertEqual(pairs, {(2, 0), (2, 1)})
    test.assertFalse(np.any(solver._cross_world_contacts.numpy()))
    test.assertFalse(np.any(solver.constraint_overflow.numpy()))
    test.assertEqual(int(solver._scene.overflow.numpy()[0]), 0)
    test.assertTrue(np.all(view["row_start"] >= 0))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        steel = SolverSteel(
            model, **_steel_on(model, {**SCENE, "solve_mode": "serial"}), warn_constraint_overflow=False
        )
    state = _initial_state(model)
    steel.step(state, model.state(), model.control(), _collide(model, state), DT)
    test.assertGreater(int(steel._cross_world_contacts.numpy()[1]), 0)
    np.testing.assert_array_equal(steel.constraint_overflow.numpy(), [False, True, True])


def test_limits_unit_rows_follow_the_count_pass(test, device):
    """Give the limits unit the rows of the limits COUNT pass, ahead of its articulation's contacts."""
    solver, _state, _contacts = _rows(_humanoids(device), joint_limit_activation_gap=10.0)
    test.assertEqual(int(solver._scene_limit_rows.numpy()[0]), HUMANOID_LIMIT_ROWS)
    view = _unit_view(solver)
    limits = view["contact"] == -1
    test.assertEqual(int(np.count_nonzero(limits)), 1)
    test.assertEqual(int(view["rows"][limits][0]), HUMANOID_LIMIT_ROWS)
    test.assertEqual(int(view["row_start"][limits][0]), 0)
    row_type = solver._row_type.numpy()[0]
    np.testing.assert_array_equal(row_type[:HUMANOID_LIMIT_ROWS], PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
    test.assertNotEqual(int(row_type[HUMANOID_LIMIT_ROWS]), PGS_CONSTRAINT_TYPE_JOINT_LIMIT)

    # No bound is within a zero gap of the initial pose.
    solver, _state, _contacts = _rows(_humanoids(device), joint_limit_activation_gap=0.0)
    test.assertEqual(int(solver._scene_limit_rows.numpy()[0]), 0)
    test.assertEqual(solver.color_statistics()["limit_units"], 0)


def test_scene_row_build_replays_under_capture(test, device):
    """Rebuild the scene rows from a captured graph: fixed shapes, no host readback, the eager result."""
    for name in ("heap", "humanoids"):
        with test.subTest(scene=name):
            model = _heap(device) if name == "heap" else _humanoids(device, per_world=4, spacing=0.4)
            solver, state, contacts = _rows(model)
            scene = solver._scene

            def snapshot(solver=solver, scene=scene):
                rows = int(solver._constraint_count.numpy()[0])
                view = _unit_view(solver)
                return {
                    "_constraint_count": rows,
                    "_sparse_row_dof": solver._sparse_row_dof.numpy()[0, :rows].copy(),
                    "_sparse_row_factor": solver._sparse_row_factor.numpy()[0, :rows].copy(),
                    "_rhs": solver._rhs.numpy()[0, :rows].copy(),
                    "_diag": solver._diag.numpy()[0, :rows].copy(),
                    "unit_row_start": view["row_start"].copy(),
                    "seq_row_start": scene.seq_row_start.numpy().copy(),
                }

            eager = snapshot()
            test.assertGreater(eager["_constraint_count"], 0)
            with wp.ScopedCapture(model.device) as capture:
                scene_solve.build_scene_rows(solver, state, solver, contacts, DT)
            for array in (
                solver._sparse_row_dof,
                solver._sparse_row_factor,
                solver._rhs,
                solver._diag,
                scene.unit_row_start,
                scene.seq_row_start,
                solver._constraint_count,
            ):
                array.fill_(POISON)
            wp.capture_launch(capture.graph)
            replay = snapshot()
            for key, value in eager.items():
                np.testing.assert_array_equal(replay[key], value, err_msg=key)


def test_scene_statistics_reads_back(test, device):
    """Read the scene's rows and units back to the host as plain ints, after a step and only in scene mode."""
    model = _heap(device)
    solver = _scene_solver(model)
    with test.assertRaisesRegex(RuntimeError, "needs a step first"):
        solver.scene_statistics()
    with test.assertRaisesRegex(RuntimeError, 'needs solve_mode="scene"'):
        SolverSteel(model, **_steel_on(model, {"solve_mode": "serial"})).scene_statistics()
    state = _initial_state(model)
    _step_rows(solver, model, state, _collide(model, state))
    stats = solver.scene_statistics()
    keys = ["rows", "rows_requested", "row_max", "dropped_rows", "units", "unit_overflow"]
    keys += ["one_sided_units", "two_sided_units", "late_units", "coupled_sides", "colors", "tail"]
    keys += ["compliant_contacts", "compliant_contacts_skipped", "compliant_contacts_dropped"]
    keys += ["compliant_contacts_invalid", "compliance_error", "torsion_rows", "torsion_status", "scene_stepped"]
    test.assertEqual(list(stats), keys)
    test.assertIs(stats.pop("scene_stepped"), True)
    for key, value in stats.items():
        test.assertIs(type(value), int, msg=key)
    test.assertEqual(stats["rows"], int(solver._constraint_count.numpy()[0]))
    test.assertEqual(stats["units"], int(solver._scene_units.unit_count.numpy()[0]))
    # One world of free bodies: both per-world budgets.
    test.assertEqual(stats["row_max"], SCENE_ROW_MAX + solver.free_body_constraint_max)
    test.assertEqual(stats["dropped_rows"], 0)
    test.assertEqual(stats["one_sided_units"] + stats["two_sided_units"], stats["units"])
    test.assertGreater(stats["two_sided_units"], 0)
    # Late units (velocity limits) are one-sided units; the heap has none.
    test.assertEqual(stats["late_units"], 0)
    # Without contact_compliance and contact torsion their counts are zero.
    for key in keys[-8:-1]:
        test.assertEqual(stats[key], 0, key)
    test.assertEqual(json.loads(json.dumps(stats)), stats)


class _StandIn:
    """A stand-in for a solver, a state or contacts: every attribute and item is another stand-in."""

    def __getattr__(self, name):
        return _StandIn()

    def __getitem__(self, key):
        return _StandIn()


class TestSteelSceneRows(unittest.TestCase):
    def test_row_templates_expand_to_steel_text(self):
        """Expand the scene's two row templates at one mask word to Steel's own native text, character for character.

        The contact response's template with the specialised factor branch at 27 DOFs is the snippet of Steel's
        ``_get_sparse_contact_response_kernel(27)``, and the joint-limit family's expansion of the internal-row
        template is the source Steel's ``_build_sparse_joint_limit_rows`` is built from.
        """
        self.assertEqual(
            scene_rows._response_snippet(27, mask_words=1, specialized=True),
            _native_snippet(sparse_contact._get_sparse_contact_response_kernel(27)),
        )
        self.assertEqual(
            scene_rows.sparse_joint_limit_rows_snippet(1), sparse_contact._build_sparse_joint_limit_rows.native_snippet
        )

    def test_connect_rows_need_the_closure_row_count(self):
        """Build the connect rows for the closures' most rows, which is required: 3, 5 or 6.

        ``max_closure_rows`` has no default, so a caller that omits it fails at build time instead of building a
        kernel without the rotation rows for a REVOLUTE (5 rows) or FIXED (6 rows) closure. BALL closures alone (3)
        leave the rotation rows out of the kernel; 5 and 6 share the kernel that builds them, and a count that is no
        closure's raises ``ValueError``.
        """
        self.assertEqual(scene_rows.CONNECT_ROWS_MAX, 6)
        with self.assertRaises(TypeError):
            scene_rows.get_scene_connect_rows_kernel(27)
        with self.assertRaisesRegex(ValueError, "max_closure_rows must be 3, 5 or 6"):
            scene_rows.get_scene_connect_rows_kernel(27, max_closure_rows=4)
        kernel = scene_rows.get_scene_connect_rows_kernel(27, 2, max_closure_rows=3)
        self.assertEqual(kernel.key, "populate_scene_connect_rows_27_8_4_w2")
        rotation = scene_rows.get_scene_connect_rows_kernel(27, 2, max_closure_rows=6)
        self.assertEqual(rotation.key, "populate_scene_connect_rows_27_8_4_rotation_w2")
        self.assertIs(scene_rows.get_scene_connect_rows_kernel(27, 2, max_closure_rows=5), rotation)

    def test_contact_builders_launch_by_name(self):
        """Give Steel's contact row builders every argument by its name, and refuse a missing or unknown one.

        The scene's row build launches ``allocate_world_contact_slots`` and ``prepare_world_contact_rows`` by name
        (``scene_solve._launch_by_name``), so a parameter upstream adds, renames or removes fails at the launch, named,
        instead of shifting the arguments after it. The argument dicts the row build passes, read on stand-ins (no
        solver), hold exactly each kernel's parameters; a dict without one of them, or with one more, raises
        ``TypeError`` naming it, before anything launches.
        """
        stand_in = _StandIn()
        geometry = scene_solve._contact_geometry(stand_in, 1)
        cases = {
            allocate_world_contact_slots: scene_solve._contact_slot_arguments(
                stand_in, stand_in, geometry, stand_in, stand_in
            ),
            prepare_world_contact_rows: scene_solve._contact_row_arguments(
                stand_in, stand_in, stand_in, stand_in, geometry
            ),
        }
        for kernel, arguments in cases.items():
            with self.subTest(kernel=kernel.key):
                self.assertEqual(sorted(arguments), sorted(arg.label for arg in kernel.adj.args))
                dropped = next(iter(arguments))
                missing = {name: value for name, value in arguments.items() if name != dropped}
                with self.assertRaisesRegex(TypeError, rf"missing arguments \['{dropped}'\]"):
                    scene_solve._launch_by_name(kernel, 1, missing, None)
                with self.assertRaisesRegex(TypeError, r"unknown arguments \['not_an_argument'\]"):
                    scene_solve._launch_by_name(kernel, 1, {**arguments, "not_an_argument": 0}, None)

    def test_connect_rotation_rows_restate_steel_text(self):
        """Keep the scene's rotation rows of a REVOLUTE or FIXED loop closure equal to Steel's text.

        ``scene_rows.scene_connect_rotation_rows`` restates the lines of Steel's ``populate_connect_J_for_size`` that
        read the rows' directions and errors (Steel's own ``_connect_rotation_rows``) and give each row its target under
        a prescribed parent; ``scene_rows.connect_rotation_mirror`` extracts both, which must be equal. A changed line
        inside Steel's ranges no longer matches, and a changed first or last line is no longer found. Steel builds each
        row's Jacobian as ``e . omega_parent - e . omega_child``, the couple the scene gives the response's two sides.
        """
        scene, steel = scene_rows.connect_rotation_mirror()
        self.assertGreaterEqual(len(steel), 8, steel)
        self.assertEqual(scene, steel)
        source = inspect.getsource(populate_connect_J_for_size.func)
        inside, last = "if prescribed and body_p >= 0:", "target = -wp.dot(e, omega_a)"
        self.assertEqual((source.count(inside), source.count(last)), (1, 1))
        scene, steel = scene_rows.connect_rotation_mirror(steel=source.replace(inside, "if body_p >= 0:"))
        self.assertNotEqual(scene, steel)
        with self.assertRaisesRegex(LookupError, "populate_connect_J_for_size has no line"):
            scene_rows.connect_rotation_mirror(steel=source.replace(last, "target = wp.dot(e, omega_a)"))
        compact = "".join(source.split())
        for call in scene_rows._SOURCE_CONNECT_ROTATION_JACOBIAN:
            with self.subTest(call=call):
                self.assertEqual(compact.count(call), 1)

    def test_connect_norm_min_is_steels(self):
        """Make a connect row inert below the squared norm Steel's ``_finish_connect_row`` tests, its literal."""
        (literal,) = re.findall(r"if norm_sq > ([0-9.eE+-]+):", inspect.getsource(_finish_connect_row.func))
        self.assertEqual(scene_rows.CONNECT_NORM_MIN, float(literal))


# The rows build on every device; a Steel reference is Steel's split solve on the CPU (_steel_on).
for _fn in (
    test_row_order_is_side_major_then_colors,
    test_free_free_rows_have_prefix_12,
    test_rows_overflow_in_scan_order,
    test_unit_overflow_latches_every_world,
    test_global_body_couples_model_worlds,
    test_limits_unit_rows_follow_the_count_pass,
    test_scene_statistics_reads_back,
):
    add_function_test(TestSteelSceneRows, _fn.__name__, _fn, devices=get_test_devices())

# CUDA only, each for the reason beside it (test_scene_cpu compares the CPU's rows with CUDA's).
for _fn in (
    test_rows_match_steel_rows_bitwise,  # Steel's sparse rows are its matrix-free solve's; its CPU split's are dense
    test_scene_row_build_replays_under_capture,  # graph capture
):
    add_function_test(TestSteelSceneRows, _fn.__name__, _fn, devices=get_cuda_test_devices())

if __name__ == "__main__":
    unittest.main()
