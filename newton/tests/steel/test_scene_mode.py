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

import numpy as np
import warp as wp

import newton
from newton import NewtonWarning
from newton._src.solvers.steel._testing import kernel_overrides
from newton._src.solvers.steel.scene_rows import FAMILY_CONNECT, FAMILY_MIMIC
from newton._src.solvers.steel.scene_solve import SCENE_FREE_SUPPORT
from newton._src.solvers.steel.scene_units import SceneUnits
from newton.solvers import SolverSteel
from newton.solvers.experimental.coupled import CouplingInterface
from newton.tests.steel._fixtures import DT, _build_four_bar, _heap, set_torsion_radius
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
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

needs_read_back = unittest.skipUnless(hasattr(SolverSteel, "scene_statistics"), "needs the read-back API")

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


@needs_read_back
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


def test_scene_rejects_revolute_and_fixed_closures(test, device):
    """Refuse REVOLUTE and FIXED loop-closing joints at construction with the mode's error, not a kernel failure.

    Steel closes a REVOLUTE loop with five rows and a FIXED one with six; the scene's connect rows are the three of a
    BALL closure. A BALL closure builds the scene's connect family; a REVOLUTE or a FIXED closure raises
    :class:`ValueError` naming the mode and the joint type before any scene kernel is built, where Steel's serial solve
    builds all three.
    """
    for closure, rows in (("ball", 3), ("revolute", 5), ("fixed", 6)):
        with test.subTest(closure=closure):
            model = _hinge_closed_by(device, closure)
            steel = SolverSteel(model, **STEEL)
            test.assertEqual(steel._connect_rows_np.tolist(), [rows])
            if closure == "ball":
                solver = SolverSteel(model, **SCENE)
                test.assertIn(FAMILY_CONNECT, solver._scene.internal_families)
                continue
            message = f"solve_mode='scene' does not support loop-closing {closure.upper()} joints (joints [1])"
            with test.assertRaisesRegex(ValueError, re.escape(message)):
                SolverSteel(model, **SCENE)


def _surface_belt(device) -> newton.Model:
    """A free box resting on a static mesh belt whose surface moves (:attr:`newton.Mesh.enable_surface_velocity`)."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    belt = newton.Mesh.create_box(2.0, 2.0, 0.05, compute_inertia=False)
    belt.enable_surface_velocity = True
    builder.add_shape_mesh(-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity()), mesh=belt)
    box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.05)
    return builder.finalize(device=device)


def test_scene_rejects_new_upstream_features(test, device):
    """Refuse the Steel features the scene does not implement yet with :class:`ValueError` naming the mode.

    Nesterov momentum, ``reuse_kinematics``, a mesh whose surface moves and coupling (the coupling hooks a coupled
    solver calls) each raise under the mode, where Steel's serial solve accepts them. Steel's own checks run first:
    on the CPU, where Steel refuses Nesterov momentum, the mode raises Steel's error.
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
    with test.subTest(feature="surface velocity"):
        belt = _surface_belt(device)
        test.assertTrue(SolverSteel(belt, **STEEL)._has_surface_velocity)
        with test.assertRaisesRegex(ValueError, re.escape("solve_mode='scene' does not support mesh surface velocity")):
            SolverSteel(belt, **SCENE)
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
    if solver._scene_mode and solver.model.joint_count:
        stats = solver.scene_statistics()
        test.assertEqual((stats["rows"], stats["rows_requested"], stats["units"], stats["dropped_rows"]), (0, 0, 0, 0))
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


@needs_read_back
def test_scene_mode_steps_models_without_responding_bodies(test, device):
    """Step a model without a responding articulation or free body, as Steel does.

    A ground plane alone, static shapes alone and a box welded to the world, resting on the ground, give the scene no
    factor group, no free body and no unit (:func:`_models_without_responding_bodies`). The scene solver and Steel's
    serial solver, both with ``row_watermark``, step ``1 + EMPTY_STEPS`` times with a collision pipeline. The
    welded box has contacts with the ground, the other models none; every contact is routed nowhere (no responding
    side), and no row or unit is built (``scene_statistics()``: rows, rows requested, units and dropped rows 0, where
    the scene's step runs); every row mark is zero, the contact mark the contact count; no overflow
    (``check_constraint_capacity`` passes, after a reset too); and the state after each step is Steel's bit for bit,
    without NaN. A step of a model without joints returns before any stage, in Steel and in the scene. On CUDA a
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
    test_scene_rejects_revolute_and_fixed_closures,
    test_scene_rejects_new_upstream_features,
    test_scene_options_validate_their_values,
    test_propagation_options_are_accepted_and_ignored,
):
    add_function_test(TestSteelSceneMode, _fn.__name__, _fn, devices=get_test_devices())

# CUDA only, each for the reason beside it.
for _fn in (
    test_scene_mode_selects_the_plan_past_the_shared_memory_gate,  # the shared-memory gate
    test_scene_mode_lets_steel_reject_first,  # on the CPU Steel refuses its CUDA-only features before these checks
    test_scene_mode_steps_without_iterations,  # graph capture
    test_scene_mode_reads_notified_restitution,  # the device list of the stepping tests is re-decided with the step
):
    add_function_test(TestSteelSceneMode, _fn.__name__, _fn, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
