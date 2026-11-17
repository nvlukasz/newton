# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scene mode of SolverSteel (``solve_mode="scene"``): one solve world, its row plan and its rejections.

Steel's own checks run first, so a combination Steel rejects raises Steel's error under the mode too; what the mode
then rejects on its own raises :class:`ValueError` naming the mode.
"""

import re
import unittest
import warnings
from functools import partial
from typing import NamedTuple

import numpy as np
import warp as wp

import newton
from newton import NewtonWarning
from newton._src.solvers.steel._testing import kernel_overrides
from newton._src.solvers.steel.scene_rows import FAMILY_CONNECT, FAMILY_MIMIC
from newton._src.solvers.steel.scene_solve import SCENE_FREE_SUPPORT
from newton._src.solvers.steel.scene_units import SceneUnits
from newton.solvers import SolverBase, SolverSteel
from newton.solvers.experimental.coupled import CouplingInterface
from newton.tests.steel import test_contacts
from newton.tests.steel._fixtures import (
    DT,
    _build_contact_scene,
    _build_four_bar,
    _heap,
    _parent_wrench_scene,
    _set_box_velocities,
    set_torsion_radius,
)
from newton.tests.steel._scene_fixtures import (
    REBOUND_APEX_ATOL,
    SCENE,
    _drop_sphere,
    _global_box_model,
    _humanoids,
    _scene_heap,
    _steel_on,
    assert_rebound_tracks_steel,
)
from newton.tests.steel.test_articulation import _build_square_loop, _build_world_hinged_swing
from newton.tests.steel.test_link_break import AXIS_Y, _four_bar, _rod_inertia
from newton.tests.steel.test_parent_wrench import _revolute_wrench_torques
from newton.tests.steel.test_scene_aliases import _carry
from newton.tests.steel.test_scene_compliance import _CompliantStepper
from newton.tests.steel.test_scene_features import TRACK_ROWS
from newton.tests.steel.test_scene_torsion import FIXTURE_DT, _pad_model, _PadRun
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

STEEL = {**SCENE, "solve_mode": "serial"}
"""The scene tests' options on Steel's serial solve, the reference of the scene's checks."""


def _assert_steel_message(test, raised: BaseException, steel: BaseException) -> None:
    """Assert that the scene mode raised Steel's message, word for word.

    A message that formats the requested mode names ``'scene'`` where Steel's serial solve names ``'serial'``; one that
    states a requirement (``"requires solve_mode='serial'"``) is the same text under both.
    """
    message = str(steel)
    test.assertIn(str(raised), (message, message.replace("solve_mode='serial'", "solve_mode='scene'")))


def test_scene_mode_is_off_by_default(test, device):
    """Plan, allocate and fold nothing unless asked."""
    solver = SolverSteel(_humanoids(device))
    test.assertEqual(solver.solve_mode, "serial")
    test.assertIs(solver._scene_mode, False)
    test.assertIsNone(solver._scene_plan)
    test.assertIsNone(solver._scene_units)
    test.assertIsNone(solver._scene)


def test_scene_mode_folds_model_worlds(test, device):
    """Solve every model world in one solve world, keeping the model's worlds for the status flags."""
    solver = SolverSteel(_humanoids(device, worlds=4), articulation_constraint_max=128, **SCENE)
    test.assertEqual(solver.solve_mode, "scene")
    test.assertIs(solver._scene_mode, True)
    test.assertIsInstance(solver._scene_units, SceneUnits)
    test.assertEqual(solver._world_count, 1)
    test.assertTrue(np.all(solver._art_to_world.numpy() == 0))
    test.assertEqual(solver.constraint_overflow.shape, (5,))
    plan = solver._scene_plan
    test.assertEqual(
        (plan.support, plan.row_max, plan.coordinate_count, plan.articulated_size, plan.free_body_count),
        (30, 512, 108, 27, 0),
    )
    test.assertEqual(solver._sparse_row_dof.shape, (1, 512, 30))
    test.assertEqual(solver._sparse_row_factor.shape, (1, 512, 30))
    test.assertEqual(solver._sparse_row_incident.shape, (1, 512))
    test.assertEqual(solver._sparse_factor_velocity_delta.shape, (1, 108))
    test.assertEqual(solver._sparse_row_free_response.shape, (1, 1, 1))
    test.assertEqual(solver._scene_limit_rows.shape, (4,))
    test.assertEqual(solver._scene_row_cursor.shape, (4,))
    test.assertFalse(np.any(solver._scene_limit_rows.numpy()))
    test.assertFalse(np.any(solver._scene_row_cursor.numpy()))


def test_scene_mode_selects_the_plan_past_the_shared_memory_gate(test, device):
    """Select the sparse factor for 64 humanoids in one world, where Steel's per-world gate keeps dense factors."""
    model = _humanoids(device, worlds=1, per_world=64)
    solver = SolverSteel(model, articulation_constraint_max=8192, **SCENE)
    test.assertEqual(solver._sparse_mass_matrix_size, 27)
    test.assertIsNone(solver._pgs_solve_sparse_kernel)
    test.assertIsNone(solver._pgs_solve_mf_gs_kernel)
    for size in solver._size_groups:
        with test.subTest(size=size):
            test.assertEqual(solver._J_by_size[size].shape, (1, 1, 1))
            test.assertEqual(solver._Y_by_size[size].shape, (1, 1, 1))
    test.assertEqual(solver._J_world.shape, (1, 1, 1))
    test.assertIsNone(solver._paired_response_primary_size)
    test.assertIs(solver._local_internal_fast_path, False)
    steel = SolverSteel(model, articulation_constraint_max=8192, **STEEL)
    test.assertIsNone(steel._sparse_mass_matrix_size)


def test_scene_mode_free_bodies_only_plan(test, device):
    """Plan a scene of free bodies without a factor: two physical 6-entry blocks per row."""
    solver = SolverSteel(_heap(device), **SCENE)
    plan = solver._scene_plan
    rows = solver._articulation_constraint_max
    test.assertIsNone(plan.articulated_size)
    test.assertEqual(plan.support, SCENE_FREE_SUPPORT)
    test.assertEqual(plan.support, 12)
    test.assertEqual(plan.coordinate_count, 648)
    test.assertEqual(plan.free_body_count, 108)
    test.assertIsNone(solver._sparse_mass_matrix_size)
    test.assertEqual(solver._sparse_row_dof.shape, (1, rows, 12))
    test.assertEqual(solver._sparse_row_factor.shape, (1, rows, 12))
    test.assertEqual(solver._sparse_row_free_response.shape[2], 12)
    test.assertEqual(solver._sparse_factor_velocity_delta.shape, (1, 648))
    test.assertEqual(solver._Linv_by_size[6].shape, (108, 6, 6))
    test.assertTrue(np.all(solver._sparse_world_free_dof_mask.numpy() == 1))
    test.assertEqual(solver._sparse_body_dof_mask.shape, (solver.model.body_count,))
    test.assertFalse(np.any(solver._sparse_body_dof_mask.numpy()))
    test.assertIsNone(solver._pgs_solve_sparse_kernel)
    test.assertIsNone(solver._pgs_solve_mf_gs_kernel)


def test_scene_mode_with_props_widens_the_free_response(test, device):
    """Give rows of humanoids and free props a 12-entry free-body response."""
    solver = SolverSteel(_humanoids(device, worlds=2, props=True), **SCENE)
    rows = solver._articulation_constraint_max
    test.assertEqual(solver._scene_plan.support, 30)
    test.assertEqual(solver._scene_plan.free_body_count, 2)
    test.assertEqual(solver._scene_plan.articulated_size, 27)
    test.assertEqual(solver._sparse_row_free_response.shape, (1, rows, 12))
    test.assertIn(6, solver._Linv_by_size)


def test_scene_mode_accepts_steel_options(test, device):
    """Build a scene solver with each of Steel's options the scene runs.

    The propagation response solves with the immediate response and reads back as requested; the row schedules read
    back as requested, ``"phased"`` running the coupled sides in the order ``"AB"``; sleeping, the warm start, contact
    compliance, friction patches, contact torsion and Steel's other friction modes build their scene features. On the
    CPU Steel's own check refuses contact torsion, contact compliance and the other friction modes first, under the
    mode too, with Steel's message.
    """
    model = _humanoids(device)
    cuda = wp.get_device(device).is_cuda
    with test.subTest(option="articulated_contact_response='propagation'"):
        solver = SolverSteel(model, **{**SCENE, "articulated_contact_response": "propagation"})
        test.assertEqual(solver.articulated_contact_response, "propagation")
        test.assertEqual(solver._scene_contact_response, "immediate")
        test.assertIsNotNone(solver._scene)
    for schedule, order in (("phased", "AB"), ("contact_then_internal", "BA")):
        with test.subTest(option=f"row_schedule={schedule!r}"):
            solver = SolverSteel(model, **{**SCENE, "row_schedule": schedule})
            test.assertEqual((solver.row_schedule, solver._scene.phase_order), (schedule, order))
            test.assertEqual(solver.scene_phase_order, "BA")
    with test.subTest(option="enable_sleeping=True"):
        solver = SolverSteel(model, **{**SCENE, "enable_sleeping": True})
        test.assertIsNotNone(solver._sleeping)
        test.assertIsNotNone(solver._scene)
    with test.subTest(option="contact_warmstart=True"):
        solver = SolverSteel(model, **{**SCENE, "contact_warmstart": True})
        test.assertTrue(solver._contact_warmstart)
        test.assertIsNotNone(solver._scene)
    with test.subTest(option="friction_anchor_beta > 0"):
        test.assertTrue(SolverSteel(model, **{**SCENE, "friction_anchor_beta": 0.2})._scene.features.patches)
    steel_first = {
        "contact_compliance=True": ({"contact_compliance": True}, 0.0),
        "contact torsion": ({}, 0.01),
        **{
            f"friction_mode={mode!r}": ({"friction_mode": mode}, 0.0)
            for mode in ("bisection", "bisection_desaxce", "coulomb_newton")
        },
    }
    for option, (options, radius) in steel_first.items():
        with test.subTest(option=option):
            set_torsion_radius(model, radius)
            try:
                if cuda:
                    solver = SolverSteel(model, **{**SCENE, **options})
                    features = solver._scene.features
                    test.assertTrue(
                        {
                            "contact_compliance=True": features.compliance,
                            "contact torsion": features.torsion,
                        }.get(option, features.friction_mode == options.get("friction_mode"))
                    )
                else:
                    with test.assertRaises((NotImplementedError, ValueError)) as steel:
                        SolverSteel(model, **{**STEEL, **options})
                    with test.assertRaises(type(steel.exception)) as raised:
                        SolverSteel(model, **{**SCENE, **options})
                    _assert_steel_message(test, raised.exception, steel.exception)
            finally:
                set_torsion_radius(model, 0.0)
    if cuda:
        point = {name: value for name, value in SCENE.items() if name != "friction_anchor_beta"}
        with test.subTest(option="friction_mode without friction_anchor_beta"):
            # Steel's constructor selects point friction with its warning, under the mode too.
            with test.assertWarnsRegex(NewtonWarning, re.escape("friction_mode='bisection' uses point friction")):
                solver = SolverSteel(model, **point, friction_mode="bisection")
            test.assertEqual((solver.friction_anchor_beta, solver._scene.features.friction_mode), (0.0, "bisection"))


def test_scene_mode_sums_the_per_world_row_budgets(test, device):
    """Size the scene's rows as Steel's per-world budgets summed over the model's worlds."""
    budgets = {"articulation_constraint_max": 128, "free_body_constraint_max": 48}
    # Four worlds of a humanoid and a free prop box each: both budgets in every world.
    solver = SolverSteel(_humanoids(device, worlds=4, props=True), **budgets, **SCENE)
    rows = 4 * (128 + 48)
    test.assertEqual(solver.articulation_constraint_max, 128)
    test.assertEqual(solver.free_body_constraint_max, 48)
    test.assertEqual(solver._scene_plan.row_max, rows)
    test.assertEqual(solver._sparse_row_dof.shape[:2], (1, rows))
    test.assertEqual(solver._rhs.shape, (1, rows))
    test.assertEqual(solver._scene.row_max, rows)
    # A third of the rows for contact units, and one limits unit per humanoid.
    test.assertEqual(solver._scene_units.unit_max, rows // 3 + 4)
    # Without free bodies the free-body budget adds nothing.
    solver = SolverSteel(_humanoids(device, worlds=4), **budgets, **SCENE)
    test.assertEqual(solver._scene_plan.row_max, 4 * 128)
    # A global box on the boxes of two worlds: Steel solves global bodies within world 0's budgets.
    model = _global_box_model(device)
    test.assertEqual(model.world_count, 2)
    solver = SolverSteel(model, **budgets, **SCENE)
    test.assertEqual(solver._scene_plan.row_max, 2 * (128 + 48))


def test_scene_capacity_warning_reads_the_pooled_budget(test, device):
    """Warn at construction only when the scene's pooled rows cannot hold the rows every step needs.

    With ``joint_limit_activation_gap=inf`` every finite limit of a humanoid is a row in every step, ``R`` per
    humanoid. Four worlds of a humanoid with ``articulation_constraint_max=R``: each world's rows fit its budget, so
    neither Steel nor the scene warns, though the scene's one solve world holds all ``4 R`` rows (it compares them with
    its pooled capacity, ``4 R``). With a budget of ``R - 1`` both warn: Steel for each world, the scene for its one
    solve world against its pooled ``4 (R - 1)`` rows.
    """
    model = _humanoids(device, worlds=4)
    persistent = {"joint_limit_activation_gap": float("inf")}
    probe = SolverSteel(model, **{**STEEL, **persistent}, articulation_constraint_max=4096)
    limit_rows, friction_rows = probe._persistent_dense_rows_per_world(model)
    rows = limit_rows + friction_rows
    test.assertEqual(rows.tolist(), [int(rows[0])] * 4)
    per_world = int(rows[0])
    test.assertGreater(per_world, 1)

    def capacity_warnings(options: dict, budget: int) -> list[str]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SolverSteel(model, **{**options, **persistent}, articulation_constraint_max=budget)
        return [str(w.message) for w in caught if "dense rows in every step" in str(w.message)]

    for options in (STEEL, SCENE):
        with test.subTest(solve_mode=options["solve_mode"], budget="fits"):
            test.assertEqual(capacity_warnings(options, per_world), [])
    with test.subTest(solve_mode="serial", budget="short"):
        (message,) = capacity_warnings(STEEL, per_world - 1)
        test.assertIn(f"worlds [0, 1, 2, 3] need at least {per_world} dense rows", message)
        test.assertIn(f"more than articulation_constraint_max={per_world - 1}.", message)
    with test.subTest(solve_mode="scene", budget="short"):
        (message,) = capacity_warnings(SCENE, per_world - 1)
        test.assertIn(f"SolverSteel: the scene needs at least {4 * per_world} dense rows", message)
        pooled = f"the scene's {4 * (per_world - 1)} rows (the per-world row budgets, summed over the model's worlds)"
        test.assertIn(f"more than {pooled}.", message)
        test.assertNotIn("worlds [", message)


def _box_on_ground(device, *, requires_grad=False, particle=False):
    """A free box on the ground, finalized with ``requires_grad`` or beside one particle: models Steel rejects."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    if particle:
        builder.add_particle((1.0, 0.0, 0.1), (0.0, 0.0, 0.0), 1.0)
    return builder.finalize(device=device, requires_grad=requires_grad)


def test_scene_mode_lets_steel_reject_first(test, device):
    """Raise Steel's own error where Steel rejects the options or the model, whatever the scene mode allows.

    Joint friction keeps Steel's rejection of ``contact_compliance``, the warm start keeps Steel's rejections of
    friction position iterations and of contact torsion, compliance keeps Steel's rejections of the options its
    compliant update does not define (``velocity_iterations``, ``contact_regularization``, warm start, friction
    position iterations), a friction mode other than ``"fixed_normal"`` keeps Steel's rejections of friction patches,
    contact torsion, contact compliance and the propagation response, friction patches keep Steel's rejection of
    contact compliance, sleeping keeps Steel's rejections of the propagation response, of the warm start and velocity
    iterations and of thresholds that are not finite and positive, the row schedules keep Steel's rejections of an
    unknown schedule and of contact torsion and compliance outside ``"interleaved"``, Nesterov momentum keeps Steel's
    rejection of the propagation response, and the propagation response keeps Steel's rejections of an unknown
    response, of contact torsion without device torsion, of compliance, of loop-closing joints and of its own options'
    values: each raises Steel's message, word for word, where a message that names the requested mode names
    ``'scene'``. Models with gradients or particles raise Steel's model check's message. Friction patches with
    ``contact_shared_anchor`` or ``contact_friction_shared_anchor`` warn as Steel does.
    """
    model = _humanoids(device)
    four_bar = _build_four_bar().finalize(device=device)
    # Each row: the check, the model, the options, the torsion radius and a fragment of Steel's message.
    rejections = (
        (
            "joint friction with compliance",
            model,
            {"enable_joint_friction": True, "contact_compliance": True},
            0.0,
            "enable_joint_friction=True does not support contact_compliance=True",
        ),
        (
            "warm start with friction position iterations",
            model,
            {"contact_warmstart": True, "contact_friction_position_iterations": 1},
            0.0,
            "contact_friction_position_iterations >= 0 requires contact_warmstart=False",
        ),
        *(
            (f"compliance with {name}", model, {**options, "contact_compliance": True}, 0.0, fragment)
            for name, options, fragment in (
                ("velocity iterations", {"velocity_iterations": 1}, "requires velocity_iterations=0"),
                ("regularization", {"contact_regularization": 0.5}, "requires contact_regularization=0.0"),
                ("warm start", {"contact_warmstart": True}, "requires contact_warmstart=False"),
                (
                    "friction position iterations",
                    {"contact_friction_position_iterations": 1},
                    "requires contact_friction_position_iterations=-1",
                ),
            )
        ),
        *(
            (f"{check} with friction_mode={mode!r}", model, {**options, "friction_mode": mode}, radius, repr(mode))
            for mode in ("bisection", "bisection_desaxce", "coulomb_newton")
            for check, options, radius in (
                ("patches", {"friction_anchor_beta": 0.2}, 0.0),
                ("compliance", {"contact_compliance": True}, 0.0),
                ("propagation", {"articulated_contact_response": "propagation"}, 0.0),
                ("torsion", {}, 0.01),
            )
        ),
        (
            "patches with compliance",
            model,
            {"friction_anchor_beta": 0.2, "contact_compliance": True},
            0.0,
            "friction_anchor_beta > 0",
        ),
        ("warm start with torsion", model, {"contact_warmstart": True}, 0.01, "contact_warmstart=False"),
        *(
            (f"sleeping with {name}", model, {**options, "enable_sleeping": True}, 0.0, fragment)
            for name, options, fragment in (
                (
                    "propagation",
                    {"articulated_contact_response": "propagation"},
                    "articulated_contact_response='propagation'",
                ),
                ("warm start", {"contact_warmstart": True}, "without warm start"),
                ("velocity iterations", {"velocity_iterations": 1}, "velocity-only iterations"),
                ("a zero quiet time", {"sleep_quiet_time": 0.0}, "finite and positive"),
                ("a NaN threshold", {"sleep_linear_threshold": float("nan")}, "finite and positive"),
            )
        ),
        ("unknown row schedule", model, {"row_schedule": "bad"}, 0.0, "row_schedule must be"),
        (
            "compliance with phased",
            model,
            {"row_schedule": "phased", "contact_compliance": True},
            0.0,
            "requires row_schedule='interleaved', got 'phased'",
        ),
        (
            "torsion with contact_then_internal",
            model,
            {"row_schedule": "contact_then_internal"},
            0.01,
            "requires row_schedule='interleaved', got 'contact_then_internal'",
        ),
        (
            "nesterov with propagation",
            model,
            {"nesterov_momentum": True, "articulated_contact_response": "propagation"},
            0.0,
            "nesterov_momentum requires",
        ),
        (
            "unknown contact response",
            model,
            {"articulated_contact_response": "nearest"},
            0.0,
            "articulated_contact_response must be",
        ),
        (
            "torsion with propagation",
            model,
            {"articulated_contact_response": "propagation"},
            0.01,
            "requires contact_torsion_device=True",
        ),
        (
            "compliance with propagation",
            model,
            {"articulated_contact_response": "propagation", "contact_compliance": True},
            0.0,
            "contact_compliance=True is not supported with articulated_contact_response='propagation'",
        ),
        (
            "same-articulation rows without propagation",
            model,
            {"propagation_same_articulation_rows": True},
            0.0,
            "propagation_same_articulation_rows requires articulated_contact_response='propagation'",
        ),
        (
            "propagation capacity without propagation",
            model,
            {"propagation_constraint_max": 64},
            0.0,
            "propagation_constraint_max requires articulated_contact_response='propagation'",
        ),
        (
            "no cached response bodies",
            model,
            {"articulated_contact_response": "propagation", "propagation_cached_response_body_max": 0},
            0.0,
            "propagation_cached_response_body_max must be at least 1",
        ),
        (
            "loop joints with propagation",
            four_bar,
            {"articulated_contact_response": "propagation"},
            0.0,
            "Loop-closing joints are not supported",
        ),
        ("gradients", _box_on_ground(device, requires_grad=True), {}, 0.0, "does not support gradients"),
        ("particles", _box_on_ground(device, particle=True), {}, 0.0, "does not simulate particles"),
    )
    for check, rejected, options, radius, fragment in rejections:
        with test.subTest(check=check):
            set_torsion_radius(rejected, radius)
            try:
                with test.assertRaises((NotImplementedError, ValueError)) as steel:
                    SolverSteel(rejected, **{**STEEL, **options})
                with test.assertRaises(type(steel.exception)) as raised:
                    SolverSteel(rejected, **{**SCENE, **options})
            finally:
                set_torsion_radius(rejected, 0.0)
            _assert_steel_message(test, raised.exception, steel.exception)
            test.assertIn(fragment, str(steel.exception))
    for flag in ("contact_shared_anchor", "contact_friction_shared_anchor"):
        with test.subTest(check=f"{flag} with patches"):
            patched = {"friction_anchor_beta": 0.2, flag: True}
            pattern = re.escape("Patch friction selects its own friction locations")
            with test.assertWarnsRegex(NewtonWarning, pattern) as steel:
                SolverSteel(model, **{**STEEL, **patched})
            with test.assertWarnsRegex(NewtonWarning, pattern) as scene:
                solver = SolverSteel(model, **{**SCENE, **patched})
            test.assertEqual(str(scene.warning), str(steel.warning))
            test.assertTrue(solver._scene.features.patches)


def test_scene_mode_steps_without_iterations(test, device):
    """With ``iterations=0`` build the rows and sweep none, eagerly and in a captured graph, as Steel does."""
    # Four lying humanoids 0.4 m apart: ground contacts, limits and units joining neighbors.
    model = _humanoids(device, per_world=4, spacing=0.4)
    options = {**SCENE, "iterations": 0, "articulation_constraint_max": 512}
    control = model.control()

    def initial_state():
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        return state

    state = initial_state()
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)

    eager = SolverSteel(model, **options)
    eager_out = model.state()
    eager.step(initial_state(), eager_out, control, contacts, DT)
    stats = eager.scene_statistics()
    test.assertGreater(stats["rows"], 0)
    test.assertGreater(stats["two_sided_units"], 0)
    test.assertEqual(stats["dropped_rows"], 0)
    test.assertFalse(np.any(eager._impulses.numpy()[0, : stats["rows"]]))

    # Steel without iterations: the same free motion, through dense factors instead of the sparse one.
    steel = SolverSteel(model, **{**options, "solve_mode": "serial"})
    steel_out = model.state()
    steel.step(initial_state(), steel_out, control, contacts, DT)
    np.testing.assert_allclose(eager_out.joint_qd.numpy(), steel_out.joint_qd.numpy(), rtol=1.0e-4, atol=1.0e-5)

    captured = SolverSteel(model, **options)
    captured_in, captured_out = initial_state(), model.state()
    captured.step(captured_in, captured_out, control, contacts, DT)
    with wp.ScopedCapture(device=device) as capture:
        captured.step(captured_in, captured_out, control, contacts, DT)
    reset = initial_state()
    for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
        getattr(captured_in, name).assign(getattr(reset, name))
    captured_out.joint_q.fill_(np.nan)
    captured_out.joint_qd.fill_(np.nan)
    wp.capture_launch(capture.graph)
    for name in ("joint_q", "joint_qd"):
        np.testing.assert_array_equal(getattr(captured_out, name).numpy(), getattr(eager_out, name).numpy(), name)
    test.assertEqual(captured.scene_statistics(), stats)


def test_scene_mode_reads_notified_restitution(test, device):
    """Rebound as Steel does once a ``SHAPE_PROPERTIES`` notify brings restitution to a model built without it.

    The sphere of ``_build_plane_model`` falls 5 cm under gravity, as a free body or as the root of a
    two-link articulation, which the scene factors as a six-DOF group (Steel refuses a notified
    restitution on its own sparse factors, not on the scene's). Both solvers step without
    restitution; before the fifth step the coefficients become 0.5 and are notified, so the impacts
    rebound, and the vertical velocity agrees with Steel's within 1e-4 of Steel's at every step, the
    articulated sphere's within ``REBOUND_APEX_ATOL`` where it crosses zero.
    """
    for scene, atol in (("free", 0.0), ("articulated", REBOUND_APEX_ATOL)):
        with test.subTest(scene=scene):
            runs, velocities = _drop_sphere(device, restitution=0.5, scene=scene, notify_after=4)
            if scene == "articulated":
                # A six-DOF factor group: Steel's notify check of its own sparse factors would refuse it.
                test.assertEqual(runs[0].solver._sparse_mass_matrix_size, 6)
            assert_rebound_tracks_steel(test, runs, velocities, scene, atol=atol)


def test_scene_mode_builds_its_model_features(test, device):
    """Build the scene's rows of the model features it implements, and refuse a test-only override it cannot run.

    Free-body velocity limits and the depenetration cap build the scene's late rows and its cap; mimic relationships
    and BALL loop-closing joints build the internal units' mimic and connect families. The kernel override
    ``sparse_mass_matrix=False``, which selects Steel's dense factors, raises :class:`NotImplementedError`.
    """
    model = _scene_heap(device, register_attributes=True)
    for name in ("rigid_body_max_linear_velocity", "rigid_body_max_angular_velocity"):
        getattr(model.steel, name).fill_(10.0)
    model.steel.rigid_body_max_depenetration_velocity.fill_(1.0)
    solver = SolverSteel(model, **SCENE)
    test.assertTrue(solver._scene_mode)
    test.assertTrue(solver._scene.features.late_rows and solver._scene.features.depenetration_cap)
    with test.subTest(model="mimic and loop-closing joints"):
        builder = _build_four_bar()
        builder.set_joint_mimic(1, 0, coeffs=(0.0, -1.0))
        solver = SolverSteel(builder.finalize(device=device), **SCENE)
        test.assertTrue(solver._scene_mode)
        families = solver._scene.internal_families
        test.assertEqual(families, tuple(sorted(families)))
        test.assertIn(FAMILY_MIMIC, families)
        test.assertIn(FAMILY_CONNECT, families)
    with test.subTest(model="kernel override sparse_mass_matrix=False"):
        model = _humanoids(device)
        with kernel_overrides({"sparse_mass_matrix": False}):
            with test.assertRaisesRegex(NotImplementedError, re.escape("kernel override sparse_mass_matrix=False")):
                SolverSteel(model, **SCENE)


def _hinge_closed_by(device, closure: str) -> newton.Model:
    """A link hinged to the world about y and closed back to the world at its tip by a ``closure`` loop joint.

    ``closure`` is ``"ball"``, ``"revolute"`` (about y, the hinge's axis) or ``"fixed"``: the loop-closing joint's
    type, whose connect rows are 3, 5 or 6.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    link = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_box(link, hx=0.02, hy=0.02, hz=0.1)
    hinge = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.6), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
    )
    builder.add_articulation([hinge])
    add_closure = {
        "ball": builder.add_joint_ball,
        "revolute": partial(builder.add_joint_revolute, axis=wp.vec3(0.0, 1.0, 0.0)),
        "fixed": builder.add_joint_fixed,
    }[closure]
    add_closure(
        parent=-1,
        child=link,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.4), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
    )
    return builder.finalize(device=device)


CLOSURE_STEPS = 60
"""Steps of the loop-closure tests against Steel's serial solve."""

CLOSURE_RMS = 1.0e-4
"""Agreement of the scene with Steel's serial solve on the loop-closure tests: RMS [m] of the bodies' center-of-mass
positions over every step, and RMS [N m] of the undriven hinges' axis torques in ``BODY_PARENT_F`` over the sampled
steps."""

CLOSURE_OPTIONS = {"iterations": 32, "baumgarte": 0.1}
"""Options of the loop-closure tests under both solves: Steel's closure tests' Baumgarte factor, at 32 iterations."""

CLOSURE_WRENCH_EVERY = 20
"""Steps between the samples of ``BODY_PARENT_F`` in :func:`test_parent_wrench_includes_closure_reactions`."""


def _hinge_pair(device, closure: str = "fixed", *, at_hinge: bool = False, reverse: bool = False) -> newton.Model:
    """Two 0.2 m links on hinges about y, with a ``closure`` loop joint between them beside the hinge that joins them.

    The first link hangs from the world at its end and the second from the first's tip, both along x. ``closure`` is
    ``"fixed"`` (a weld) or ``"revolute"`` (a hinge about x, across the tree hinges' axis). By default the loop joint's
    anchors are at the second link's center, 0.1 m past the first link's tip: the welded model of ``test_link_break``'s
    ``test_weld_loop_holds_a_hinge``, whose anchor rows with the tree hinge already lock the bend, so its rotation rows
    are redundant. With ``at_hinge`` both anchors are on the hinge between the links, where the anchor rows are inert
    and only the closure's rotation rows hold the bend and carry the second link's weight moment. ``reverse`` makes the
    second link the loop joint's parent and the first its child.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -9.81))
    links = [
        builder.add_link(
            xform=wp.transform(wp.vec3(x, 0.0, 1.0), wp.quat_identity()),
            mass=1.0,
            inertia=_rod_inertia(1.0, 0.2, (1, 0, 0)),
        )
        for x in (0.1, 0.3)
    ]
    root = builder.add_joint_revolute(
        parent=-1,
        child=links[0],
        axis=AXIS_Y,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.1, 0.0, 0.0), wp.quat_identity()),
    )
    hinge = builder.add_joint_revolute(
        parent=links[0],
        child=links[1],
        axis=AXIS_Y,
        parent_xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.1, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([root, hinge])
    # The loop joint's anchor in each link's frame: on the hinge, or at the second link's center.
    anchors = [
        wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()) for x in ((0.1, -0.1) if at_hinge else (0.2, 0.0))
    ]
    ends = list(zip(links, anchors, strict=True))
    (parent, parent_xform), (child, child_xform) = ends[::-1] if reverse else ends
    add_closure = {
        "fixed": builder.add_joint_fixed,
        "revolute": partial(builder.add_joint_revolute, axis=wp.vec3(1.0, 0.0, 0.0)),
    }[closure]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*another joint already connects these bodies", NewtonWarning)
        add_closure(parent=parent, child=child, parent_xform=parent_xform, child_xform=child_xform)
    return builder.finalize(device=device)


def _welded_carried_load(device) -> tuple[newton.Model, int]:
    """A kinematic carrier holding a free box by one FIXED loop joint 5 cm off the box's center; returns the carrier.

    The carrier of ``_fixtures._build_carried_load``, 1 m up, with the box 0.2 m below it: the weld is a prescribed
    parent whose angular velocity enters the weld rows' target. Prescribed by ``test_scene_aliases._carry`` it screws
    about z, and the box, held by its weld's rotation rows, turns with it about the anchor.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    carrier = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(carrier, hx=0.03, hy=0.03, hz=0.03)
    load = builder.add_body(xform=wp.transform(wp.vec3(0.05, 0.0, 0.8), wp.quat_identity()))
    builder.add_shape_box(load, hx=0.04, hy=0.04, hz=0.04, cfg=newton.ModelBuilder.ShapeConfig(density=500.0))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*another joint already connects these bodies", NewtonWarning)
        builder.add_joint_fixed(
            parent=carrier,
            child=load,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.2), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.05, 0.0, 0.0), wp.quat_identity()),
        )
    return builder.finalize(device=device), carrier


def _revolute_four_bar(device) -> newton.Model:
    """``test_link_break``'s joint-break crank-rocker: three tree hinges, the loop closed by a REVOLUTE joint."""
    builder, *_ = _four_bar("joint")
    return builder.finalize(device=device)


def _undriven_tree_hinges(model) -> list[int]:
    """The REVOLUTE joints of ``model``'s articulation trees whose DOF has no drive stiffness."""
    tree = (model.joint_type.numpy() == int(newton.JointType.REVOLUTE)) & (model.joint_articulation.numpy() >= 0)
    stiffness, dof = model.joint_target_ke.numpy(), model.joint_qd_start.numpy()
    return [int(joint) for joint in np.flatnonzero(tree) if stiffness[dof[joint]] == 0.0]


def _com_positions(model, body_q: np.ndarray) -> np.ndarray:
    """World positions [m] of ``model``'s body centers of mass at the poses ``body_q`` (``[bodies, 7]``)."""
    com = model.body_com.numpy().astype(np.float64)
    position, xyz, w = body_q[:, :3], body_q[:, 3:6], body_q[:, 6:7]
    twice_cross = 2.0 * np.cross(xyz, com)
    return position + com + w * twice_cross + np.cross(xyz, twice_cross)


class _ClosureRun(NamedTuple):
    """One run of :func:`_closure_run`."""

    solver: SolverSteel
    positions: np.ndarray
    """The positions [m] of the bodies' centers of mass after every step, ``[steps, bodies, 3]``."""
    torques: np.ndarray
    """The undriven tree hinges' axis torques [N m] in each sampled ``BODY_PARENT_F``, ``[samples, hinges]``."""
    wrenches: np.ndarray
    """Each sampled ``BODY_PARENT_F``, ``[samples, bodies, 6]``."""


def _closure_run(
    model, solve_mode: str, *, crank_target: float | None = None, carrier: int | None = None, wrench_every: int = 0
) -> _ClosureRun:
    """Step ``model`` :data:`CLOSURE_STEPS` times from its forward kinematics, without contacts, under ``solve_mode``.

    ``crank_target`` [rad] is DOF 0's drive target, and ``carrier`` a kinematic carrier body prescribed before each
    step (``test_scene_aliases._carry``). ``BODY_PARENT_F`` is sampled every ``wrench_every`` steps (never for 0), with
    the axis torques of the undriven tree hinges (:func:`_undriven_tree_hinges`) about each hinge's anchor at the
    step's start.
    """
    solver = SolverSteel(model, solve_mode=solve_mode, **CLOSURE_OPTIONS)
    observables = solver.observables({SolverBase.ObservableKind.BODY_PARENT_F}) if wrench_every else None
    states = [model.state(), model.state()]
    newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])
    control = model.control()
    if crank_target is not None:
        targets = model.joint_target_q.numpy().copy()
        targets[0] = crank_target
        control.joint_target_q.assign(targets)
    hinges = _undriven_tree_hinges(model)
    positions, torques, wrenches = [], [], []
    for step in range(CLOSURE_STEPS):
        if carrier is not None:
            _carry(model, states[0], carrier, step)
        states[0].clear_forces()
        solver.step(states[0], states[1], control, None, DT, observables=observables)
        if wrench_every and step % wrench_every == wrench_every - 1:
            wrench = observables.body_parent_f.numpy().astype(np.float64)
            hinge_torques = _revolute_wrench_torques(model, states[0], wrench)
            torques.append([hinge_torques[joint] for joint in hinges])
            wrenches.append(wrench)
        states.reverse()
        positions.append(_com_positions(model, states[0].body_q.numpy().astype(np.float64)))
    return _ClosureRun(
        solver,
        np.asarray(positions, dtype=np.float64),
        np.asarray(torques, dtype=np.float64).reshape(len(torques), len(hinges)),
        np.asarray(wrenches, dtype=np.float64),
    )


def _connect_family_rows(solver) -> int:
    """The scene's connect rows of its last step: the internal units' connect family, summed."""
    return int(solver._scene_family_rows.numpy()[FAMILY_CONNECT].sum())


def test_loop_closures_match_steel_under_scene(test, device):
    """Solve REVOLUTE and FIXED loop closures under the mode as Steel's serial solve does.

    Steel closes a BALL loop with three anchor rows, a REVOLUTE one with two more hinge rows and a FIXED one with
    three more rotation rows; the scene's connect family builds the same rows. The hinge closed back to the world by
    each closure type (a prescribed parent) builds its 3, 5 or 6 rows under the mode. Then each model below is stepped
    :data:`CLOSURE_STEPS` times from rest under both solves:

    - the joint-break crank-rocker of ``test_link_break``, a REVOLUTE closure between two links of one articulation,
      whose planar tree leaves its hinge rows inert;
    - the spatial square loop of ``test_articulation``, a REVOLUTE closure whose hinge rows act;
    - the free body of ``test_articulation`` hung from the world by a REVOLUTE closure, the rows' free-body branch;
    - :func:`_hinge_pair` braced across its hinge by a REVOLUTE closure from the second link to the first, and welded
      at its hinge by a FIXED closure, where the closure's rotation rows alone hold the bend;
    - :func:`_hinge_pair`'s welded model of ``test_link_break``, a FIXED closure whose rotation rows are redundant;
    - :func:`_welded_carried_load`, a FIXED closure to a screwing kinematic carrier, whose angular velocity is the
      rotation rows' target.

    Every model moves, the scene's connect family holds the closure's rows after the last step, the closure's first
    row and its rows' norms (``_connect_slot``, ``_connect_row_inv_norm``) are Steel's, and its bodies' centers of mass
    agree with Steel's within :data:`CLOSURE_RMS` RMS over every step. So a closure the scene dropped, solved with
    fewer rows or left its rotation rows inert fails: on the braced and the welded hinge pair, and on the carried load,
    when the scene's rotation rows are inert.
    """
    for closure, rows in (("ball", 3), ("revolute", 5), ("fixed", 6)):
        with test.subTest(closure=closure, model="hinge closed to the world"):
            model = _hinge_closed_by(device, closure)
            steel = SolverSteel(model, **STEEL)
            test.assertEqual(steel._connect_rows_np.tolist(), [rows])
            solver = SolverSteel(model, **SCENE)
            test.assertIn(FAMILY_CONNECT, solver._scene.internal_families)
            state_0, state_1 = model.state(), model.state()
            solver.step(state_0, state_1, model.control(), None, DT)
            test.assertEqual(_connect_family_rows(solver), rows)
    swing, _ = _build_world_hinged_swing(test, "revolute")
    square, _ = _build_square_loop("revolute")
    carried_load, carrier = _welded_carried_load(device)
    models = {
        "revolute four-bar": (_revolute_four_bar(device), 5, None),
        "revolute square loop": (square.finalize(device=device), 5, None),
        "revolute swing": (swing.finalize(device=device), 5, None),
        "hinge braced across its axis": (_hinge_pair(device, "revolute", at_hinge=True, reverse=True), 5, None),
        "hinge welded at its axis": (_hinge_pair(device, "fixed", at_hinge=True), 6, None),
        "welded hinge": (_hinge_pair(device), 6, None),
        "welded carried load": (carried_load, 6, carrier),
    }
    for name, (model, rows, carried_by) in models.items():
        with test.subTest(model=name):
            scene = _closure_run(model, "scene", carrier=carried_by)
            steel = _closure_run(model, "serial", carrier=carried_by)
            test.assertEqual(steel.solver._connect_rows_np.tolist(), [rows])
            test.assertEqual(_connect_family_rows(scene.solver), rows)
            test.assertEqual(scene.solver._connect_slot.numpy().tolist(), steel.solver._connect_slot.numpy().tolist())
            np.testing.assert_allclose(
                scene.solver._connect_row_inv_norm.numpy()[:, :rows],
                steel.solver._connect_row_inv_norm.numpy()[:, :rows],
                rtol=1.0e-4,
            )
            test.assertGreater(float(np.abs(steel.positions[-1] - steel.positions[0]).max()), 1.0e-3)
            test.assertTrue(np.all(np.isfinite(scene.positions)))
            rms = float(np.sqrt(np.mean((scene.positions - steel.positions) ** 2)))
            test.assertLessEqual(rms, CLOSURE_RMS)


def test_scene_rejects_new_upstream_features(test, device):
    """Refuse the Steel features the scene does not implement yet with :class:`ValueError` naming the mode.

    Nesterov momentum, ``reuse_kinematics`` and coupling (the coupling hooks a coupled solver calls) each raise under
    the mode, where Steel's serial solve accepts them. Steel's own checks run first: on the CPU, where Steel refuses
    Nesterov momentum, the mode raises Steel's error.
    """
    model = _humanoids(device)
    cuda = wp.get_device(device).is_cuda
    with test.subTest(feature="nesterov_momentum=True"):
        if cuda:
            SolverSteel(model, **STEEL, nesterov_momentum=True)
            with test.assertRaisesRegex(
                ValueError, re.escape("solve_mode='scene' does not support nesterov_momentum=True")
            ):
                SolverSteel(model, **SCENE, nesterov_momentum=True)
        else:
            with test.assertRaises(NotImplementedError) as steel:
                SolverSteel(model, **STEEL, nesterov_momentum=True)
            with test.assertRaises(NotImplementedError) as raised:
                SolverSteel(model, **SCENE, nesterov_momentum=True)
            test.assertEqual(str(raised.exception), str(steel.exception))
    with test.subTest(feature="reuse_kinematics=True"):
        SolverSteel(model, **STEEL, reuse_kinematics=True)
        with test.assertRaisesRegex(ValueError, re.escape("solve_mode='scene' does not support reuse_kinematics=True")):
            SolverSteel(model, **SCENE, reuse_kinematics=True)
    with test.subTest(feature="coupling"):
        steel = SolverSteel(model, **STEEL)
        scene = SolverSteel(model, **SCENE)
        steel.coupling_notify_iteration_restarts(False)
        kind = wp.array([int(CouplingInterface.EndpointKind.BODY)], dtype=int, device=device)
        index = wp.array([0], dtype=int, device=device)
        local_pos = wp.zeros(1, dtype=wp.vec3, device=device)
        mass = wp.zeros(1, dtype=float, device=device)
        steel.coupling_eval_effective_mass(kind, index, local_pos, mass)
        message = re.escape("solve_mode='scene' does not support coupling")
        with test.assertRaisesRegex(ValueError, message):
            scene.coupling_notify_iteration_restarts(False)
        with test.assertRaisesRegex(ValueError, message):
            scene.coupling_eval_effective_mass(kind, index, local_pos, mass)
        with test.assertRaisesRegex(ValueError, message):
            scene.coupling_notify_input_state_update(model.state(), newton.StateFlags.JOINT_Q)


def test_scene_options_validate_their_values(test, device):
    """Refuse a ``scene_unit_max`` below 1 and an unknown ``scene_phase_order`` under the mode, after Steel's checks.

    Each raises :class:`ValueError` naming the mode and the value; Steel's own rejection of the same construction
    comes first. A valid value reads back as requested.
    """
    model = _humanoids(device)
    for option, value, hint in (
        ("scene_unit_max", 0, "at least 1"),
        ("scene_phase_order", "BB", "'BA', 'AB' or 'ABA'"),
    ):
        with test.subTest(option=option):
            message = f"solve_mode='scene' does not support {option}={value!r}; it must be {hint}"
            with test.assertRaisesRegex(ValueError, re.escape(message)):
                SolverSteel(model, **SCENE, **{option: value})
            # Steel's check of an unknown row schedule raises first.
            with test.assertRaisesRegex(ValueError, "row_schedule must be"):
                SolverSteel(model, **SCENE, row_schedule="bad", **{option: value})
    solver = SolverSteel(model, **SCENE, scene_unit_max=7, scene_phase_order="ABA")
    test.assertEqual((solver.scene_unit_max, solver.scene_phase_order), (7, "ABA"))
    test.assertEqual(solver._scene_units.unit_max, 7)
    test.assertEqual(solver._scene.phase_order, "ABA")


def test_propagation_options_are_accepted_and_ignored(test, device):
    """Accept the propagation response's own options under the mode and build nothing of them.

    With ``articulated_contact_response="propagation"`` the scene solves with the immediate response: the propagation
    options (``propagation_constraint_max``, ``propagation_color_order``, ``propagation_same_articulation_rows``,
    ``propagation_cached_response``, ``propagation_cached_response_body_max``) read back as requested, no propagation
    row family is active, and the scene's plan and features are those of the immediate request without them.
    """
    model = _humanoids(device, worlds=2)
    options = {
        "propagation_constraint_max": 64,
        "propagation_color_order": "longest_first",
        "propagation_same_articulation_rows": True,
        "propagation_cached_response": False,
        "propagation_cached_response_body_max": 3,
    }
    solver = SolverSteel(model, **SCENE, articulated_contact_response="propagation", **options)
    immediate = SolverSteel(model, **SCENE)
    for name, value in options.items():
        with test.subTest(option=name):
            test.assertEqual(getattr(solver, name), value)
    test.assertEqual(solver.articulated_contact_response, "propagation")
    test.assertEqual(solver._scene_contact_response, "immediate")
    test.assertFalse(solver._propagation_active)
    test.assertFalse(solver._propagation_torsion_rows)
    test.assertEqual(solver._propagation_max_constraints, solver._requested_dense_max_constraints)
    for plan in (solver._scene_plan, immediate._scene_plan):
        test.assertEqual([group.kind for group in plan.groups], ["factor"])
    plan, reference = solver._scene_plan, immediate._scene_plan
    test.assertEqual(
        (plan.support, plan.row_max, plan.coordinate_count, plan.free_body_count, plan.mask_words),
        (reference.support, reference.row_max, reference.coordinate_count, reference.free_body_count, 1),
    )
    test.assertEqual((plan.groups[0].key, plan.groups[0].arts), (reference.groups[0].key, reference.groups[0].arts))
    test.assertEqual(solver._scene.features, immediate._scene.features)
    test.assertEqual(solver._scene_units.unit_max, immediate._scene_units.unit_max)
    # Steel's serial solve builds the propagation rows these options configure.
    steel = SolverSteel(model, **STEEL, articulated_contact_response="propagation", **options)
    test.assertTrue(steel._propagation_active)


def _models_without_responding_bodies(device):
    """Yield ``(name, model)`` for models with no responding articulation and no free body.

    A ground plane alone; static shapes alone (a box and a sphere on the world body beside the ground); and a box
    link welded to the world (a fixed joint: an articulation with no response DOF), resting 1 mm deep on the ground,
    which its joint's collision filter lets it touch (``collision_filter_parent=False``): contacts with no responding
    side.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    yield "ground", builder.finalize(device=device)

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), hx=0.2, hy=0.2, hz=0.2)
    builder.add_shape_sphere(-1, xform=wp.transform(wp.vec3(1.0, 0.0, 0.1), wp.quat_identity()), radius=0.1)
    yield "static shapes", builder.finalize(device=device)

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    pose = wp.transform(wp.vec3(0.0, 0.0, 0.099), wp.quat_identity())
    link = builder.add_link(xform=pose)
    builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
    builder.add_articulation([builder.add_joint_fixed(-1, link, parent_xform=pose, collision_filter_parent=False)])
    yield "welded box", builder.finalize(device=device)


EMPTY_STEPS = 4
"""Steps of :func:`test_scene_mode_steps_models_without_responding_bodies` after the first, eager and captured."""


def _state_arrays(state) -> dict[str, np.ndarray]:
    """The state's joint and body arrays on the host, those the model has."""
    names = ("joint_q", "joint_qd", "body_q", "body_qd")
    return {name: getattr(state, name).numpy() for name in names if getattr(state, name) is not None}


def _check_no_rows(test, solver, contacts):
    """Assert that ``solver``'s last step built no row and no unit: every contact routed nowhere, no row mark."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    np.testing.assert_array_equal(solver._contact_path.numpy()[:count], -1)
    test.assertFalse(np.any(solver.constraint_overflow.numpy()))
    if solver._scene_mode:
        stats = solver.scene_statistics()
        test.assertEqual((stats["rows"], stats["rows_requested"], stats["units"], stats["dropped_rows"]), (0, 0, 0, 0))
        # A model without articulations returns from every step before the scene runs: zeros, no step of the scene.
        test.assertIs(stats["scene_stepped"], bool(solver.model.articulation_count))
        if not solver.model.articulation_count:
            # The capacity is the scene's; every count of a step is zero.
            test.assertEqual(stats.pop("row_max"), solver._scene.row_max)
            test.assertEqual(set(stats.values()), {0})
            colors = solver.color_statistics()
            test.assertIs(colors.pop("scene_stepped"), False)
            test.assertEqual(colors.pop("per_color"), [0] * solver._scene_units.color_cap)
            test.assertEqual(set(colors.values()), {0})
            timing = solver.time_scene_units(n=2)
            test.assertEqual(timing, {"build_ms": 0.0, "color_ms": 0.0, "n": 2, "mode": "none", "scene_stepped": False})
    return count


def test_scene_mode_constructs_models_without_responding_bodies(test, device):
    """Construct the scene for a model without a responding articulation or free body: no factor group, no unit.

    Sleeping, which needs an articulation, raises Steel's ``_SleepState`` error on the models without one, under the
    mode too.
    """
    for name, model in _models_without_responding_bodies(device):
        with test.subTest(model=name):
            solver = SolverSteel(model, **SCENE)
            test.assertTrue(solver._scene_mode)
            test.assertIsNone(solver._sparse_mass_matrix_size)
            test.assertFalse(any(group.kind == "factor" for group in solver._scene_plan.groups))
            if not model.articulation_count:
                with test.assertRaises(ValueError) as raised:
                    SolverSteel(model, enable_sleeping=True, **SCENE)
                test.assertIn("at least one articulation", str(raised.exception))
                with test.assertRaises(ValueError) as steel:
                    SolverSteel(model, enable_sleeping=True, **STEEL)
                test.assertEqual(str(raised.exception), str(steel.exception))


def test_scene_mode_steps_models_without_responding_bodies(test, device):
    """Step a model without a responding articulation or free body, as Steel does.

    A ground plane alone, static shapes alone and a box welded to the world, resting on the ground, give the scene no
    factor group, no free body and no unit (:func:`_models_without_responding_bodies`). The scene solver and Steel's
    serial solver, both with ``row_watermark``, step ``1 + EMPTY_STEPS`` times with a collision pipeline. The
    welded box has contacts with the ground, the other models none; every contact is routed nowhere (no responding
    side), and no row or unit is built (``scene_statistics()``: rows, rows requested, units and dropped rows 0; on a
    model without articulations every read-back is zeros with ``scene_stepped`` false); every row mark is zero, the
    contact mark the contact count; no overflow (``check_constraint_capacity`` passes, after a reset too); and the
    state after each step is Steel's bit for bit, without NaN. A step of a model without articulations returns before
    any stage, in Steel and in the scene. On CUDA a
    graph of :data:`EMPTY_STEPS` steps, the states swapped between them, captured after an eager first step, ends at
    Steel's state, building no row.
    """
    for name, model in _models_without_responding_bodies(device):
        with test.subTest(model=name):
            runs = {}
            for label, options in (("scene", SCENE), ("steel", _steel_on(model, STEEL))):
                solver = SolverSteel(model, row_watermark=True, **options)
                test.assertIs(solver._scene_mode, label == "scene")
                pipeline = newton.CollisionPipeline(model)
                contacts = pipeline.contacts()
                state_0, state_1, control = model.state(), model.state(), model.control()
                states = []
                for _ in range(1 + EMPTY_STEPS):
                    pipeline.collide(state_0, contacts)
                    solver.step(state_0, state_1, control, contacts, DT)
                    state_0, state_1 = state_1, state_0
                    states.append(_state_arrays(state_0))
                count = _check_no_rows(test, solver, contacts)
                test.assertEqual(count > 0, name == "welded box")
                marks = solver.constraint_row_watermarks()
                test.assertEqual(marks.pop("contact_high_water"), count)
                test.assertEqual(set(marks.values()), {0})
                solver.check_constraint_capacity()
                solver.reset(state_0)
                solver.check_constraint_capacity()
                runs[label] = states
            for step, (scene, steel) in enumerate(zip(runs["scene"], runs["steel"], strict=True)):
                test.assertEqual(scene.keys(), steel.keys())
                for key, value in scene.items():
                    test.assertTrue(np.all(np.isfinite(value)), f"{key} step {step}")
                    np.testing.assert_array_equal(value, steel[key], err_msg=f"{key} step {step}")
            if wp.get_device(device).is_cuda:
                captured = SolverSteel(model, **SCENE)
                pipeline = newton.CollisionPipeline(model)
                contacts = pipeline.contacts()
                state_0, state_1, control = model.state(), model.state(), model.control()
                pipeline.collide(state_0, contacts)
                captured.step(state_0, state_1, control, contacts, DT)
                state_0, state_1 = state_1, state_0
                with wp.ScopedCapture(device=device) as capture:
                    for _ in range(EMPTY_STEPS):
                        pipeline.collide(state_0, contacts)
                        captured.step(state_0, state_1, control, contacts, DT)
                        state_0, state_1 = state_1, state_0
                wp.capture_launch(capture.graph)
                for key, value in _state_arrays(state_0).items():
                    np.testing.assert_array_equal(value, runs["steel"][-1][key], err_msg=f"captured {key}")
                test.assertEqual(_check_no_rows(test, captured, contacts) > 0, name == "welded box")
                captured.check_constraint_capacity()


def test_scene_mode_accepts_none_anchor_beta(test, device):
    """Resolve the default friction_anchor_beta as Steel does: friction patches (0.2), which the scene runs."""
    solver = SolverSteel(_humanoids(device), solve_mode="scene")
    test.assertIs(solver._friction_anchors_enabled, True)
    test.assertEqual(solver.friction_anchor_beta, 0.2)
    test.assertTrue(solver._scene.features.patches)


def test_articulation_constraint_max_reads_the_requested_budget(test, device):
    """Read back the requested per-world budget under the mode, whose rows pool it, and reject assigning it as Steel.

    The scene's row capacity is the budget summed over the model's worlds; ``articulation_constraint_max`` still reads
    the requested value, and assigning it raises Steel's :class:`AttributeError` with Steel's message.
    """
    model = _humanoids(device, worlds=2)
    steel = SolverSteel(model, articulation_constraint_max=64, **STEEL)
    scene = SolverSteel(model, articulation_constraint_max=64, **SCENE)
    test.assertEqual((steel.articulation_constraint_max, scene.articulation_constraint_max), (64, 64))
    test.assertEqual((steel._articulation_constraint_max, scene._articulation_constraint_max), (64, 128))
    with test.assertRaises(AttributeError) as expected:
        steel.articulation_constraint_max = 128
    with test.assertRaises(AttributeError) as raised:
        scene.articulation_constraint_max = 128
    test.assertEqual(str(raised.exception), str(expected.exception))
    test.assertEqual(scene.articulation_constraint_max, 64)


def _assert_refused_in_capture(test, device, reads: dict) -> None:
    """Assert that each of ``reads`` (name: callable) raises inside a CUDA graph capture, naming the capture."""
    with wp.ScopedCapture(device=device):
        for name, read in reads.items():
            with test.subTest(read=name), test.assertRaisesRegex(RuntimeError, "outside CUDA graph capture"):
                read()


SLIDER_HEIGHT = 0.2
"""Height [m] of :func:`_compliant_slider`'s sphere above its contact: no contact at the start."""


def _compliant_slider(device):
    """``run_fixture``'s 0.3 kg sphere on a vertical slider, above the ground, with a one-row budget per world.

    Returns ``(solver, step)``: the scene solver with ``contact_compliance`` and both per-world budgets one row (two
    rows pooled, short of a contact's three), and a step that collides, fills the contacts' material (stiffness
    3000 N/m, damping 20 N s/m) and steps fixed state buffers, so that one captured step replays every later one.
    ``step.state`` is the state the step reads.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.rigid_gap = 0.005
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    pose = wp.transform(wp.vec3(0.0, 0.0, SLIDER_HEIGHT), wp.quat_identity())
    body = builder.add_link(xform=pose)
    builder.add_articulation(
        [builder.add_joint_prismatic(parent=-1, child=body, axis=newton.Axis.Z, parent_xform=pose)]
    )
    density = 0.3 / (4.0 / 3.0 * np.pi * 0.05**3)
    builder.add_shape_sphere(body, radius=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=density, mu=0.5))
    model = builder.finalize(device=device)
    model.rigid_contact_max = 32
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=32)
    contacts = pipeline.contacts()
    for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
        setattr(contacts, name, wp.zeros(32, dtype=float, device=device))
    solver = SolverSteel(
        model,
        solve_mode="scene",
        contact_compliance=True,
        friction_anchor_beta=0.0,
        velocity_iterations=0,
        articulation_constraint_max=1,
        free_body_constraint_max=1,
        warn_constraint_overflow=False,
    )
    state, next_state, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    def step():
        pipeline.collide(state, contacts)
        contacts.rigid_contact_stiffness.fill_(3000.0)
        contacts.rigid_contact_damping.fill_(20.0)
        contacts.rigid_contact_friction.fill_(1.0)
        solver.step(state, next_state, control, contacts, DT)
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            wp.copy(getattr(state, name), getattr(next_state, name))

    step.state = state
    return solver, step


def test_scene_status_reports_through_steel_api(test, device):
    """Report the scene's torsion and compliance status through Steel's status API, and refuse every read in capture.

    Torsion: the pads' contacts made hydroelastic (a positive stiffness), which the scene's spin rows refuse; the eager
    step latches the cause instead of raising. Steel's ``_status_sources`` carries the scene's torsion status word, so
    ``check_constraint_capacity`` raises it with Steel's prefix, live and from a ``snapshot_constraint_status`` copy,
    and passes once a reset cleared it. Compliance: the compliant contact scene's captured step, replayed after its
    contacts' stiffness turned NaN, latches into Steel's ``ComplianceBuffers.status``, which the check raises with
    Steel's message and ``scene_statistics()`` reports (``compliant_contacts_invalid`` and ``compliance_error``); a
    reset clears both. A captured replay whose compliant contact the pooled one-row budgets drop
    (:func:`_compliant_slider`) reports ``compliant_contacts_dropped`` and ``compliance_error`` and raises Steel's row
    overflow, which its eager step raises too (``test_scene_compliance.test_compliance_counts_match_steel``); a reset
    clears both. Inside a capture the scene's read-back (``scene_statistics``, ``color_statistics``,
    ``time_scene_units``, ``_torsion_stats``) and Steel's status reads raise, as Steel's do.
    """
    model, state = _pad_model(device)
    run = _PadRun(model, state, solve_mode="scene")
    run.pipeline.collide(run.state_0, run.contacts)
    run.contacts.rigid_contact_stiffness = wp.ones(run.contacts.rigid_contact_max, device=model.device)
    solver = run.solver
    solver.step(run.state_0, run.state_1, run.control, run.contacts, FIXTURE_DT)
    test.assertEqual(solver.scene_statistics()["torsion_status"], 2)
    test.assertIn("scene_torsion", solver._status_sources())
    solver.snapshot_constraint_status()
    hydroelastic = "Steel contact torsion rejected its input: Contact torsion does not support hydroelastic"
    for snapshot in (True, False):
        with test.subTest(snapshot=snapshot), test.assertRaisesRegex(RuntimeError, hydroelastic):
            solver.check_constraint_capacity(snapshot=snapshot)
    solver.reset(run.state_0)
    solver.snapshot_constraint_status()
    solver.check_constraint_capacity(snapshot=True)
    solver.check_constraint_capacity()
    _assert_refused_in_capture(
        test,
        device,
        {
            "scene_statistics": solver.scene_statistics,
            "color_statistics": solver.color_statistics,
            "time_scene_units": solver.time_scene_units,
            "_torsion_stats": lambda: solver._torsion_stats,
            "snapshot_constraint_status": solver.snapshot_constraint_status,
            "check_constraint_capacity": solver.check_constraint_capacity,
        },
    )

    stepper = _CompliantStepper(_build_contact_scene(device), TRACK_ROWS, {})
    _set_box_velocities(stepper.solver.model, stepper.state_0)
    stepper.step()
    stepper.collide()
    solver = stepper.solver
    with wp.ScopedCapture(device=device) as capture:
        solver.step(stepper.state_0, stepper.state_1, stepper.control, stepper.contacts, DT)
    wp.capture_launch(capture.graph)
    solver.check_constraint_capacity()
    test.assertEqual(solver.scene_statistics()["compliance_error"], 0)
    stepper.contacts.rigid_contact_stiffness.fill_(float("nan"))
    wp.capture_launch(capture.graph)
    stats = solver.scene_statistics()
    test.assertEqual((stats["compliant_contacts_invalid"], stats["compliance_error"]), (1, 1))
    test.assertEqual(stats["compliant_contacts_dropped"], 0)
    solver.snapshot_constraint_status()
    invalid = r"Steel contact compliance rejected its input: Invalid contact stiffness \(contact 0\)"
    for snapshot in (True, False):
        with test.subTest(snapshot=snapshot), test.assertRaisesRegex(RuntimeError, invalid):
            solver.check_constraint_capacity(snapshot=snapshot)
    solver.reset(stepper.state_0)
    solver.check_constraint_capacity()
    stats = solver.scene_statistics()
    test.assertEqual((stats["compliant_contacts_invalid"], stats["compliance_error"]), (0, 0))
    _assert_refused_in_capture(
        test,
        device,
        {
            "compliance_contact_count": lambda: solver.compliance_contact_count,
            "compliance_skipped_contact_count": lambda: solver.compliance_skipped_contact_count,
        },
    )

    # A compliant contact that the pooled budget drops: the sphere above the ground steps eagerly without contacts,
    # then the captured step replays with the sphere 1 mm into the ground, whose contact's three rows pass the two.
    solver, step = _compliant_slider(device)
    step()
    with wp.ScopedCapture(device=device) as capture:
        step()
    wp.capture_launch(capture.graph)
    solver.check_constraint_capacity()
    state = step.state
    state.joint_q.fill_(0.049 - SLIDER_HEIGHT)
    state.joint_qd.zero_()
    newton.eval_fk(solver.model, state.joint_q, state.joint_qd, state)
    wp.capture_launch(capture.graph)
    stats = solver.scene_statistics()
    test.assertEqual((stats["rows"], stats["dropped_rows"]), (0, 3))
    test.assertEqual(
        (stats["compliant_contacts_dropped"], stats["compliant_contacts_invalid"], stats["compliance_error"]), (1, 0, 1)
    )
    solver.snapshot_constraint_status()
    # Steel latches the row loss first (a global error), as its eager step raises it.
    dropped = "Steel contact compliance rejected its input: contact_compliance rejects overflowing solver rows"
    for snapshot in (True, False):
        with test.subTest(snapshot=snapshot, case="dropped"), test.assertRaisesRegex(RuntimeError, dropped):
            solver.check_constraint_capacity(snapshot=snapshot)
    solver.reset(state)
    solver.check_constraint_capacity()
    stats = solver.scene_statistics()
    test.assertEqual((stats["compliant_contacts_dropped"], stats["compliance_error"]), (0, 0))


PARENT_WRENCH_STEPS = 60
"""Steps of :func:`test_parent_wrench_matches_steel_under_scene`, enough for the boxes to settle."""

PARENT_WRENCH_RMS = 1.0e-5
"""Agreement [RMS, N and N m] of the scene's ``BODY_PARENT_F`` with Steel's serial solve on the welded boxes."""


def test_parent_wrench_matches_steel_under_scene(test, device):
    """Report ``BODY_PARENT_F`` and ``CONTACT_F`` from the scene's one solve world as Steel's serial solve does.

    Two free boxes resting on the ground, each carrying a smaller box welded on top
    (``_fixtures._parent_wrench_scene``), stepped :data:`PARENT_WRENCH_STEPS` times under the scene solve and under
    Steel's serial solve with both observables requested. Each weld carries its top box's weight, so Steel's parent
    wrenches are far from zero; each base box's free joint transmits nothing once the ground contacts are read from the
    step's rows, so a contact read from the wrong row or world would leave the stack's weight in it. The scene's parent
    wrenches agree with Steel's within :data:`PARENT_WRENCH_RMS` RMS, and its per-contact forces carry each stack's
    weight within 3 %, as Steel's do.
    """
    model, _ = _parent_wrench_scene(device)
    kinds = {SolverBase.ObservableKind.BODY_PARENT_F, SolverBase.ObservableKind.CONTACT_F}
    weights = np.linalg.norm(model.gravity.numpy()[0]) * model.body_mass.numpy()
    stacks = weights.reshape(-1, 2).sum(axis=1)
    results = {}
    for mode in ("scene", "serial"):
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
        contacts = pipeline.contacts()
        solver = SolverSteel(
            model, solve_mode=mode, iterations=32, friction_anchor_beta=0.0, articulation_constraint_max=128
        )
        observables = solver.observables(kinds, contacts=contacts)
        states = [model.state(), model.state()]
        control = model.control()
        for _ in range(PARENT_WRENCH_STEPS):
            pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], control, contacts, DT, observables=observables)
            states.reverse()
        net = test_contacts._net_body_forces(model, contacts, observables.contact_f.numpy())
        lifted = net[:, 2].reshape(-1, 2).sum(axis=1)
        np.testing.assert_allclose(lifted, stacks, rtol=0.03, err_msg=f"{mode}: contact forces")
        results[mode] = observables.body_parent_f.numpy()
    # The welds carry their top boxes: the reference is far from the bound.
    test.assertGreater(float(np.max(np.abs(results["serial"]))), 0.5 * float(np.min(weights)))
    test.assertTrue(np.all(np.isfinite(results["scene"])))
    rms = float(np.sqrt(np.mean((results["scene"] - results["serial"]) ** 2)))
    test.assertLessEqual(rms, PARENT_WRENCH_RMS)


class TestSteelSceneMode(unittest.TestCase):
    pass


for _fn in (
    test_scene_mode_is_off_by_default,
    test_scene_mode_folds_model_worlds,
    test_scene_mode_free_bodies_only_plan,
    test_scene_mode_with_props_widens_the_free_response,
    test_scene_mode_accepts_steel_options,
    test_scene_mode_builds_its_model_features,
    test_scene_mode_constructs_models_without_responding_bodies,
    test_scene_mode_steps_models_without_responding_bodies,
    test_scene_mode_accepts_none_anchor_beta,
    test_scene_mode_sums_the_per_world_row_budgets,
    test_scene_capacity_warning_reads_the_pooled_budget,
    test_articulation_constraint_max_reads_the_requested_budget,
    test_loop_closures_match_steel_under_scene,
    test_scene_rejects_new_upstream_features,
    test_scene_options_validate_their_values,
    test_propagation_options_are_accepted_and_ignored,
    test_parent_wrench_matches_steel_under_scene,
):
    add_function_test(TestSteelSceneMode, _fn.__name__, _fn, devices=get_test_devices())

# CUDA only, each for the reason beside it.
for _fn in (
    test_scene_mode_selects_the_plan_past_the_shared_memory_gate,  # the shared-memory gate
    test_scene_mode_lets_steel_reject_first,  # on the CPU Steel refuses its CUDA-only features before these checks
    test_scene_mode_steps_without_iterations,  # graph capture
    test_scene_mode_reads_notified_restitution,  # the device list of the stepping tests is re-decided with the step
    test_scene_status_reports_through_steel_api,  # graph capture
):
    add_function_test(TestSteelSceneMode, _fn.__name__, _fn, devices=get_cuda_test_devices())


def test_parent_wrench_includes_closure_reactions(test, device):
    """Report the loop closures' reactions in ``BODY_PARENT_F`` under the mode as Steel's serial solve does.

    ``BODY_PARENT_F`` maps each closure row's impulse back to a force or couple through the closure's first row and the
    norm its row builder applied (``SolverSteel._connect_slot`` and ``_connect_row_inv_norm``), which the scene's
    connect rows record as Steel's do. Each linkage below is stepped :data:`CLOSURE_STEPS` times under both solves with
    the observable requested:

    - the four-bar of Steel's own closure test, a BALL closure, its crank driven to 0.6 rad;
    - the joint-break crank-rocker, a REVOLUTE closure whose anchor forces carry the loop;
    - :func:`_hinge_pair` braced across its hinge by a REVOLUTE closure and welded at its hinge by a FIXED one, whose
      rotation rows' couples alone carry the second link's weight moment past the undriven hinge between the links;
    - :func:`_hinge_pair`'s welded model of ``test_link_break``, a FIXED closure.

    Steel's report carries a load (its largest entry is above half the lightest body's weight), and the axis torques of
    the undriven tree hinges, sampled every :data:`CLOSURE_WRENCH_EVERY` steps, agree with Steel's within
    :data:`CLOSURE_RMS` RMS. An undriven hinge transmits no axis torque, so they are close to zero, but only with every
    closure reaction counted: without the anchor forces or the rotation rows' couples, the closure's load lands on
    the hinges.
    """
    models = {
        "ball four-bar": (_build_four_bar().finalize(device=device), 0.6),
        "revolute four-bar": (_revolute_four_bar(device), None),
        "hinge braced across its axis": (_hinge_pair(device, "revolute", at_hinge=True, reverse=True), None),
        "hinge welded at its axis": (_hinge_pair(device, "fixed", at_hinge=True), None),
        "welded hinge": (_hinge_pair(device), None),
    }
    for name, (model, crank_target) in models.items():
        with test.subTest(model=name):
            runs = {
                mode: _closure_run(model, mode, crank_target=crank_target, wrench_every=CLOSURE_WRENCH_EVERY)
                for mode in ("scene", "serial")
            }
            weights = np.linalg.norm(model.gravity.numpy()[0]) * model.body_mass.numpy()
            test.assertGreater(float(np.abs(runs["serial"].wrenches).max()), 0.5 * float(weights.min()))
            hinges = len(_undriven_tree_hinges(model))
            test.assertEqual(runs["scene"].torques.shape, (CLOSURE_STEPS // CLOSURE_WRENCH_EVERY, hinges))
            test.assertTrue(np.all(np.isfinite(runs["scene"].wrenches)))
            rms = float(np.sqrt(np.mean((runs["scene"].torques - runs["serial"].torques) ** 2)))
            test.assertLessEqual(rms, CLOSURE_RMS)


add_function_test(
    TestSteelSceneMode,
    test_parent_wrench_includes_closure_reactions.__name__,
    test_parent_wrench_includes_closure_reactions,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
