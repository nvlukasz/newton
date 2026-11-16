# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Options that the scene mode of SolverSteel (``solve_mode="scene"``) runs as aliases of the scene's own solve.

``row_schedule="phased"`` runs the scene with its coupled sides in the phase order ``"AB"``, bit for bit
``scene_phase_order="AB"``, and ``row_schedule="contact_then_internal"`` runs the scene's own order; both read back
as requested, after Steel's checks of the schedules, and so does ``scene_phase_order``.

``articulated_contact_response="propagation"`` solves with the immediate response, bit for bit the scene of
``"immediate"``, after Steel's checks of the propagation response; it reads back as requested, and none of its rows,
tree factors or caches is built.

Mimic relationships and loop-closing joints are rows of the scene's internal units (the mimic and connect families),
which track Steel's; ``enable_bilateral_preelimination=True`` runs those plain rows, bit for bit the scene without it,
and reads back as requested: Steel's pre-elimination (its direct closure solve) is not built in the scene mode.
"""

import re
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel import scene_rows
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import (
    DT,
    _build_carried_load,
    _build_four_bar,
    _build_scissor_model,
    _build_two_revolute_chain,
    _loop_anchor_gap,
)
from newton.tests.steel._scene_fixtures import SCENE, _humanoids, _rms
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

needs_scene_step = unittest.skipUnless(hasattr(SolverSteel, "_seed_contact_impulses"), "needs the scene step")
needs_read_back = unittest.skipUnless(hasattr(SolverSteel, "scene_statistics"), "needs the read-back API")

HUMANOID_ROWS = {"articulation_constraint_max": 1024}
"""The row budget of :func:`_humanoid_rows`' one world, which none of their 30 steps overflows."""

ROW_SPACINGS = {"apart": 0.5, "touching": 0.4}
"""Spacings [m] of :func:`_humanoid_rows`: 6 cm between neighbours, or 4 cm of overlap (as the crowd of
``test_scene_sweep``)."""

TRACK_ITERATIONS = 32
"""Iterations of the schedule test that tracks Steel."""

TRACK_RMS = 1.0e-3
"""RMS bound of the joint coordinates after 30 steps against Steel's."""

MIN_FINITE_STEPS = 10
"""Fewest leading steps of a propagation-alias comparison that both runs must keep finite (the scissor keeps 12)."""

BILATERAL = {"iterations": 32, "baumgarte": 0.1}
"""Options of the mimic and connect tests: Steel's closure tests' Baumgarte factor, at 32 iterations."""

LOOP_GAP_ATOL = 1.0e-5
"""Bound [m] on the scene's loop anchor gap less Steel's at every step."""

BILATERAL_Q_ATOL = {1: 1.0e-6, 30: 1.0e-5}
"""Bound on the joint coordinates' largest difference from Steel's after each compared step (measured on the L40:
at most 1.5e-8 after step 1 and 6.0e-8 after step 30, over the four-bar, the carried load and the mimic chain)."""

ALIAS_GAP_MAX = 1.0e-4
"""Bound [m] on the four-bar's loop anchor gap after 720 steps under the pre-elimination alias."""

CARRIER_VELOCITY = (np.array([0.3, 0.0, 0.1]), np.array([0.0, 0.0, 1.5]))
"""The kinematic carrier's linear [m/s] and angular [rad/s] velocity, as in Steel's carried-load test."""

PREELIMINATION_ONLY = ("_preelim_art_to_idx", "_preelim_slots", "_preelim_S", "_preelim_LS")
"""Storage of Steel's pre-elimination plan (``_build_preelimination_plan``)."""

PROPAGATION_ONLY = (
    "_propagation_joint_parent_slot",
    "_propagation_body_response",
    "_propagation_tree_Ia",
    "_propagation_tree_D_chol",
    "_propagation_cache_R",
)
"""Storage of the propagation response: the tree factors, the bodies' responses and the cached responses."""


def _finite_steps(states):
    """The number of leading ``(joint_q, joint_qd)`` states of ``states`` that are finite."""
    for step, state in enumerate(states):
        if not all(np.all(np.isfinite(field)) for field in state):
            return step
    return len(states)


def _trajectory(model, *, steps=30, init=None, statistics=None, states=None, **options):
    """Step ``model`` with a solver of ``options`` and its own deterministic collision pipeline.

    A ``statistics`` list receives the scene's :meth:`~newton.solvers.SolverSteel.scene_statistics` after each
    step, and a ``states`` list each step's ``(joint_q, joint_qd)``.
    """
    solver = SolverSteel(model, **options)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    control = model.control()
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    if init is not None:
        init(model, state_0)
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        if statistics is not None:
            statistics.append(solver.scene_statistics())
        if states is not None:
            # On the CPU numpy() is a view of the state, which a later step overwrites: keep a copy.
            states.append((state_0.joint_q.numpy().copy(), state_0.joint_qd.numpy().copy()))
    return solver, state_0.joint_q.numpy(), state_0.joint_qd.numpy()


def _humanoid_rows(device, spacing):
    """Four lying humanoids ``spacing`` [m] apart in one world (:data:`ROW_SPACINGS`).

    Apart, each humanoid's sides are isolated (no two-sided unit); touching, the neighbours' contacts make two-sided
    units and coupled sides.
    """
    return _humanoids(device, per_world=4, spacing=spacing)


def _assert_no_overflow(test, *solvers):
    for solver in solvers:
        test.assertFalse(np.any(solver.constraint_overflow.numpy()), "a row budget overflowed")


def test_aliases_read_back_as_requested(test, device):
    """Read every alias back as requested while the scene runs its own solve.

    ``row_schedule="phased"`` sets the effective phase order to ``"AB"`` (``_scene_phase_order``), whatever
    ``scene_phase_order`` says, and ``scene_phase_order`` still reads the requested order;
    ``"contact_then_internal"`` keeps the requested order. ``articulated_contact_response="propagation"`` solves with
    the immediate response (``_scene_contact_response``), building none of the propagation response's storage, which
    Steel's serial solve builds on the same model; the deprecated ``"propagation_colored"`` keeps the scene mode and
    reads back ``"propagation"``. ``enable_bilateral_preelimination=True`` builds none of Steel's
    pre-elimination storage, which Steel's serial solve builds on the four-bar.
    """
    model = _humanoids(device)
    for schedule, order, effective in (
        ("interleaved", "BA", "BA"),
        ("phased", "BA", "AB"),
        ("phased", "ABA", "AB"),
        ("contact_then_internal", "ABA", "ABA"),
    ):
        with test.subTest(row_schedule=schedule, scene_phase_order=order):
            solver = SolverSteel(model, **SCENE, row_schedule=schedule, scene_phase_order=order)
            test.assertEqual((solver.row_schedule, solver.scene_phase_order), (schedule, order))
            test.assertEqual((solver._requested_scene_phase_order, solver._scene_phase_order), (order, effective))
            test.assertEqual(solver._scene.phase_order, effective)
    with test.subTest(articulated_contact_response="propagation"):
        alias = SolverSteel(model, **SCENE, articulated_contact_response="propagation")
        steel = SolverSteel(model, **{**SCENE, "solve_mode": "serial"}, articulated_contact_response="propagation")
        test.assertEqual(alias.articulated_contact_response, "propagation")
        test.assertEqual(alias._scene_contact_response, "immediate")
        test.assertFalse(alias._propagation_active)
        test.assertTrue(steel._propagation_active)
        for attribute in PROPAGATION_ONLY:
            test.assertTrue(hasattr(steel, attribute), attribute)
            test.assertFalse(hasattr(alias, attribute), attribute)
    with test.subTest(articulated_contact_response="propagation_colored"):
        # The deprecated spelling keeps an explicit scene request: it names the propagation response, which the scene
        # solves with the immediate response, where the other modes take it for solve_mode="colored".
        message = "articulated_contact_response='propagation_colored' is deprecated; with solve_mode='scene'"
        with test.assertWarnsRegex(newton.NewtonDeprecationWarning, re.escape(message)):
            alias = SolverSteel(model, **SCENE, articulated_contact_response="propagation_colored")
        test.assertEqual((alias.solve_mode, alias.articulated_contact_response), ("scene", "propagation"))
        test.assertEqual(alias._scene_contact_response, "immediate")
        test.assertFalse(alias._propagation_active)
        test.assertIsNotNone(alias._scene)
    with test.subTest(enable_bilateral_preelimination=True):
        four_bar = _build_four_bar().finalize(device=device)
        alias = SolverSteel(four_bar, **BILATERAL, solve_mode="scene", enable_bilateral_preelimination=True)
        steel = SolverSteel(four_bar, **BILATERAL, solve_mode="serial", enable_bilateral_preelimination=True)
        test.assertTrue(alias.enable_bilateral_preelimination)
        test.assertFalse(alias._preelim_active)
        test.assertTrue(steel._preelim_active)
        for attribute in PREELIMINATION_ONLY:
            test.assertTrue(hasattr(steel, attribute), attribute)
            test.assertFalse(hasattr(alias, attribute), attribute)


@needs_scene_step
@needs_read_back
def test_phased_alias_runs_ab_order(test, device):
    """Step ``row_schedule="phased"`` as the scene in the phase order ``"AB"``, bit for bit, reading back ``"phased"``.

    Steel's ``"phased"`` sweeps each iteration's internal rows, then its contacts, then its velocity limits. The
    scene's isolated sides already run their internal unit, their contacts and their late rows in that order, and
    its coupled sides do under ``"AB"`` (one-sided rows, the internal unit first, then the two-sided units, then
    phase C), so the alias runs ``"AB"`` whatever ``scene_phase_order`` says. Four lying humanoids in one world, over
    30 steps: 0.5 m apart, where every side is isolated and the orders agree, and touching (0.4 m), where coupled
    sides make the default ``"BA"`` step differently, so the equality is the order's.
    """
    for name, spacing in ROW_SPACINGS.items():
        with test.subTest(humanoids=name):
            model = _humanoid_rows(device, spacing)
            statistics = []
            phased, phased_q, phased_qd = _trajectory(model, **SCENE, **HUMANOID_ROWS, row_schedule="phased")
            ab, ab_q, ab_qd = _trajectory(
                model, **SCENE, **HUMANOID_ROWS, scene_phase_order="AB", statistics=statistics
            )
            ba, ba_q, _ = _trajectory(model, **SCENE, **HUMANOID_ROWS)
            _assert_no_overflow(test, phased, ab, ba)
            test.assertEqual(phased.solve_mode, "scene")
            test.assertEqual(phased.row_schedule, "phased")
            test.assertEqual((phased.scene_phase_order, phased._scene_phase_order), ("BA", "AB"))
            test.assertEqual(phased._scene.phase_order, "AB")
            test.assertEqual((ab.row_schedule, ab._scene_phase_order), ("interleaved", "AB"))
            test.assertEqual(ba._scene_phase_order, "BA")
            test.assertTrue(np.all(np.isfinite(phased_q)))
            test.assertTrue(np.array_equal(phased_q, ab_q))
            test.assertTrue(np.array_equal(phased_qd, ab_qd))
            coupled = max(step["coupled_sides"] for step in statistics)
            if name == "apart":
                test.assertEqual(coupled, 0)
                test.assertTrue(np.array_equal(ba_q, ab_q))
            else:
                test.assertGreater(coupled, 0)
                test.assertFalse(np.array_equal(ba_q, ab_q))


@needs_scene_step
def test_contact_then_internal_alias_runs(test, device):
    """Step ``row_schedule="contact_then_internal"`` in the scene's own order, tracking Steel's schedule.

    The scene has no sweep of every contact iteration before every internal one: the alias runs the scene's
    order (bit for bit the scene of ``"interleaved"``) and reads back ``"contact_then_internal"``. On the humanoids
    of :func:`test_phased_alias_runs_ab_order`, apart and touching, at 32 iterations it stays finite, and its joint
    coordinates after 30 steps are within 1e-3 RMS of Steel's ``"contact_then_internal"`` (``solve_mode="serial"``).
    """
    for name, spacing in ROW_SPACINGS.items():
        with test.subTest(humanoids=name):
            model = _humanoid_rows(device, spacing)
            options = {**SCENE, **HUMANOID_ROWS, "iterations": TRACK_ITERATIONS}
            alias, alias_q, alias_qd = _trajectory(model, **options, row_schedule="contact_then_internal")
            scene, scene_q, scene_qd = _trajectory(model, **options)
            steel, steel_q, _ = _trajectory(
                model, **{**options, "solve_mode": "serial"}, row_schedule="contact_then_internal"
            )
            _assert_no_overflow(test, alias, scene, steel)
            test.assertEqual(alias.solve_mode, "scene")
            test.assertIsNone(steel._scene)
            test.assertEqual(alias.row_schedule, "contact_then_internal")
            test.assertEqual(alias._scene.phase_order, "BA")
            test.assertTrue(np.all(np.isfinite(alias_q)))
            test.assertTrue(np.all(np.isfinite(alias_qd)))
            test.assertTrue(np.array_equal(alias_q, scene_q))
            test.assertTrue(np.array_equal(alias_qd, scene_qd))
            test.assertLess(_rms(alias_q, steel_q), TRACK_RMS)


@needs_scene_step
@needs_read_back
def test_propagation_alias_is_bitwise_immediate_scene(test, device):
    """Step the propagation response as the scene's immediate response, bit for bit, reading back as requested.

    The scene's factor coordinates give the immediate response exactly and its coloring parallelizes the
    contacts, which is what the propagation response buys Steel, so ``"propagation"`` solves with the immediate
    response (``_scene_contact_response``). On the scissor's contacts between two links of one articulation and on
    two worlds of a lying humanoid, each step's state equals the immediate request's byte for byte, and none of the
    propagation response's storage is built, which Steel's propagation solve builds on the same model. The humanoids
    are compared over all 30 steps, all finite. The scissor's links start interpenetrating without gravity: its first
    step has contact rows, the next ones push the links apart at about 2,000 rad/s, and its state is no longer finite
    from step 13, in Steel's solve as in the scene's, so it is compared over the steps both runs keep finite, at least
    :data:`MIN_FINITE_STEPS`.
    """
    scenes = {
        "scissor": (_build_scissor_model(device), {}),
        "humanoids": (_humanoids(device, worlds=2), {"articulation_constraint_max": 128}),
    }
    for name, (model, budgets) in scenes.items():
        statistics, immediate_states = [], []
        immediate, _, _ = _trajectory(model, **SCENE, **budgets, statistics=statistics, states=immediate_states)
        test.assertEqual(immediate._scene_contact_response, "immediate")
        test.assertGreater(statistics[0]["rows"], 0)
        test.assertTrue(all(np.all(np.isfinite(field)) for field in immediate_states[0]))
        test.assertGreater(float(np.max(np.abs(immediate_states[0][1]))), 0.0)
        if name == "humanoids":
            test.assertTrue(all(np.all(np.isfinite(field)) for state in immediate_states for field in state))
        with test.subTest(scene=name):
            states = []
            alias, _, _ = _trajectory(
                model, **SCENE, **budgets, articulated_contact_response="propagation", states=states
            )
            steel = SolverSteel(
                model, **{**SCENE, **budgets, "solve_mode": "serial"}, articulated_contact_response="propagation"
            )
            test.assertEqual(alias.solve_mode, "scene")
            test.assertEqual(alias.articulated_contact_response, "propagation")
            test.assertEqual(alias._scene_contact_response, "immediate")
            test.assertFalse(alias._propagation_active)
            test.assertTrue(steel._propagation_active)
            for attribute in PROPAGATION_ONLY:
                test.assertTrue(hasattr(steel, attribute), attribute)
                test.assertFalse(hasattr(alias, attribute), attribute)
            _assert_no_overflow(test, immediate, alias)
            test.assertEqual(len(states), len(immediate_states))
            finite = _finite_steps(states)
            test.assertEqual(finite, _finite_steps(immediate_states), f"{name}: finite steps differ")
            test.assertGreaterEqual(finite, MIN_FINITE_STEPS, f"{name}: {finite} finite steps")
            if name == "humanoids":
                test.assertEqual(finite, len(states), f"{name}: {finite} finite steps")
            for step, (got, want) in enumerate(zip(states[:finite], immediate_states[:finite], strict=True)):
                for label, field, reference in zip(("joint_q", "joint_qd"), got, want, strict=True):
                    message = f"{name}: {label} at step {step + 1} of {finite} finite steps"
                    test.assertEqual(field.tobytes(), reference.tobytes(), message)


def _carry(model, state, carrier, step):
    """Prescribe the kinematic carrier of :func:`_build_carried_load` at step ``step``: a screw at
    :data:`CARRIER_VELOCITY` from its start at (0, 0, 1) m, its free joint's coordinates and velocities, then its
    articulation's forward kinematics (Steel's carried-load test, ``test_articulation._prescribe_carrier``)."""
    joint = int(np.flatnonzero(model.joint_child.numpy() == carrier)[0])
    q_start, qd_start = int(model.joint_q_start.numpy()[joint]), int(model.joint_qd_start.numpy()[joint])
    linear, angular = CARRIER_VELOCITY
    t = step * DT
    speed = float(np.linalg.norm(angular))
    rotation = wp.quat_from_axis_angle(wp.vec3(*(angular / speed)), speed * t)
    q, qd = state.joint_q.numpy(), state.joint_qd.numpy()
    q[q_start : q_start + 3] = np.array([0.0, 0.0, 1.0]) + linear * t
    q[q_start + 3 : q_start + 7] = [rotation[0], rotation[1], rotation[2], rotation[3]]
    qd[qd_start : qd_start + 3] = linear
    qd[qd_start + 3 : qd_start + 6] = angular
    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    articulation = wp.array([int(model.joint_articulation.numpy()[joint])], dtype=wp.int32, device=model.device)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state, indices=articulation)


def _bilateral_run(model, *, steps, leader_target=None, carrier=None, **options):
    """Step ``model`` without contacts, as Steel's closure and mimic tests do, with a solver of ``options``.

    ``leader_target`` drives DOF 0 (the four-bar's crank, the mimic chain's leader) to that position, and
    ``carrier`` is the kinematic carrier of :func:`_build_carried_load`, prescribed before each step
    (:func:`_carry`). Returns the solver and each step's ``joint_q`` and loop anchor gap
    (:func:`_loop_anchor_gap`; ``None`` without a loop joint).
    """
    solver = SolverSteel(model, **options)
    control = model.control()
    if leader_target is not None:
        targets = model.joint_target_q.numpy().copy()
        targets[0] = leader_target
        control.joint_target_q.assign(targets)
    loops = np.any(model.joint_type.numpy() == int(newton.JointType.BALL))
    state_0, state_1 = model.state(), model.state()
    coordinates, gaps = [], []
    for step in range(steps):
        if carrier is not None:
            _carry(model, state_0, carrier, step)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
        # A copy: on the CPU numpy() is a view of the state, which later steps overwrite.
        coordinates.append(state_0.joint_q.numpy().copy())
        gaps.append(_loop_anchor_gap(model, state_0) if loops else None)
    return solver, coordinates, gaps


def _assert_tracks_steel(test, label, scene, steel, reference="Steel's"):
    """Assert the scene's ``joint_q`` within :data:`BILATERAL_Q_ATOL` of Steel's after each compared step, all
    finite, the coordinates of the two runs (``_bilateral_run``'s second result) being ``scene`` and ``steel``;
    ``reference`` names the second run in the messages."""
    for step, atol in BILATERAL_Q_ATOL.items():
        got, want = scene[step - 1], steel[step - 1]
        test.assertTrue(np.all(np.isfinite(got)), f"{label}: joint_q not finite after step {step}")
        error = float(np.max(np.abs(got.astype(np.float64) - want)))
        test.assertLess(error, atol, f"{label}: joint_q off {reference} by {error:.3e} after step {step}")


@needs_scene_step
def test_connect_rows_track_steel(test, device):
    """Track Steel's loop closures with the scene's connect rows, at 32 iterations.

    The four-bar's BALL closure joins two links of its one articulation (the contact response's self-contact path,
    ``art_a == art_b``); the carried load's three closures pin a free body to a kinematic carrier that screws along
    :data:`CARRIER_VELOCITY` (a prescribed parent, whose anchor velocity is the rows' target). The joint coordinates
    after steps 1 and 30 are within :data:`BILATERAL_Q_ATOL` of Steel's (``solve_mode="serial"``), and the loop anchor
    gap within :data:`LOOP_GAP_ATOL` of Steel's after every one of the 30 steps. The rows are the internal units'
    connect family: three per enabled closure.
    """
    builder, carrier, _, _ = _build_carried_load()
    scenes = {
        "four-bar": (_build_four_bar().finalize(device=device), {"leader_target": 0.6}, 3),
        "carried load": (builder.finalize(device=device), {"carrier": carrier}, 9),
    }
    for name, (model, run, rows) in scenes.items():
        with test.subTest(scene=name):
            scene, scene_q, scene_gaps = _bilateral_run(model, steps=30, **run, **BILATERAL, solve_mode="scene")
            steel, steel_q, steel_gaps = _bilateral_run(model, steps=30, **run, **BILATERAL, solve_mode="serial")
            test.assertEqual(scene.solve_mode, "scene")
            test.assertIsNone(steel._scene)
            _assert_no_overflow(test, scene, steel)
            family_rows = scene._scene_family_rows.numpy()
            test.assertEqual(int(family_rows[scene_rows.FAMILY_CONNECT].sum()), rows)
            _assert_tracks_steel(test, name, scene_q, steel_q)
            differences = np.abs(np.asarray(scene_gaps) - np.asarray(steel_gaps))
            worst = int(np.argmax(differences))
            test.assertLess(
                float(differences[worst]),
                LOOP_GAP_ATOL,
                f"{name}: loop gap off Steel's by {differences[worst]:.3e} m after step {worst + 1}",
            )


@needs_scene_step
def test_mimic_rows_track_steel(test, device):
    """Track Steel's mimic row with the scene's, at 32 iterations.

    The two-revolute chain's follower mimics its driven leader, ``q_f = 1 - 0.5 q_l``, from a start that violates it
    by 1 rad; the leader is driven to 0.6 rad. The joint coordinates after steps 1 and 30 are within
    :data:`BILATERAL_Q_ATOL` of Steel's, and the row is the internal unit's mimic family.
    """
    builder, _, _ = _build_two_revolute_chain(1.0, -0.5)
    model = builder.finalize(device=device)
    scene, scene_q, _ = _bilateral_run(model, steps=30, leader_target=0.6, **BILATERAL, solve_mode="scene")
    steel, steel_q, _ = _bilateral_run(model, steps=30, leader_target=0.6, **BILATERAL, solve_mode="serial")
    test.assertEqual(scene.solve_mode, "scene")
    _assert_no_overflow(test, scene, steel)
    test.assertEqual(int(scene._scene_family_rows.numpy()[scene_rows.FAMILY_MIMIC].sum()), 1)
    _assert_tracks_steel(test, "mimic chain", scene_q, steel_q)
    # Non-vacuity: the follower moved toward its mimic from the violated start.
    test.assertGreater(abs(float(scene_q[-1][1])), 0.1)


class _CapturedStepper:
    """A solver stepping fixed state buffers without contacts, so that one captured step replays later ones; DOF 0
    (the four-bar's crank) is driven to 0.6 rad."""

    def __init__(self, model, **options):
        self.model = model
        self.solver = SolverSteel(model, **options)
        self.state_0, self.state_1 = model.state(), model.state()
        self.control = model.control()
        targets = model.joint_target_q.numpy().copy()
        targets[0] = 0.6
        self.control.joint_target_q.assign(targets)

    def step(self):
        self.solver.step(self.state_0, self.state_1, self.control, None, DT)
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            wp.copy(getattr(self.state_0, name), getattr(self.state_1, name))


@needs_scene_step
def test_connect_release_under_capture(test, device):
    """Drop the four-bar's connect rows and bring them back between replays of one captured step.

    The four-bar's closure is enabled for steps 1-20 and 41-60 and released (``Model.joint_enabled`` false, in
    place) for steps 21-40, with one graph captured while it was enabled. The connect family has three rows while
    the closure is enabled and none while it is released, in the eager and the replayed steps, whose ``joint_q``
    and ``joint_qd`` are equal at every step. The release lets the loop open, and the restored rows pull it back.
    """
    model = _build_four_bar().finalize(device=device)
    closure = int(np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.BALL))[-1])

    def set_enabled(enabled):
        flags = model.joint_enabled.numpy()
        flags[closure] = enabled
        model.joint_enabled.assign(flags)

    options = {**BILATERAL, "solve_mode": "scene"}
    eager, captured = _CapturedStepper(model, **options), _CapturedStepper(model, **options)
    eager.step()
    captured.step()
    with wp.ScopedCapture(device) as recorded:
        captured.step()
    gaps = [_loop_anchor_gap(model, eager.state_0)]
    for index in range(1, 60):
        set_enabled(not 20 <= index < 40)
        eager.step()
        wp.capture_launch(recorded.graph)
        for name in ("joint_q", "joint_qd"):
            a, b = getattr(eager.state_0, name).numpy(), getattr(captured.state_0, name).numpy()
            test.assertTrue(np.array_equal(a, b), f"{name} differs after step {index + 1}")
        expected = 0 if 20 <= index < 40 else 3
        for solver in (eager.solver, captured.solver):
            connect_rows = int(solver._scene_family_rows.numpy()[scene_rows.FAMILY_CONNECT].sum())
            test.assertEqual(connect_rows, expected, f"connect rows after step {index + 1}")
        gaps.append(_loop_anchor_gap(model, eager.state_0))
    set_enabled(True)
    test.assertTrue(np.all(np.isfinite(captured.state_0.joint_q.numpy())))
    # Released, the loop opens; restored, it closes again.
    test.assertGreater(gaps[39], 10.0 * max(gaps[19], 1.0e-6))
    test.assertLess(gaps[59], 0.5 * gaps[39])


@needs_scene_step
def test_preelimination_alias_converges(test, device):
    """Run ``enable_bilateral_preelimination=True`` as the scene's plain rows, reading back as requested.

    Steel's pre-elimination (a regularised direct solve of each articulation's closure rows) is not built in the scene
    mode: the alias steps the four-bar bit for bit as the scene without it, none of Steel's pre-elimination storage
    exists, and at 32 iterations the loop anchor gap after 720 driven steps is below :data:`ALIAS_GAP_MAX`.
    """
    model = _build_four_bar().finalize(device=device)
    alias, alias_q, alias_gaps = _bilateral_run(
        model, steps=720, leader_target=0.6, **BILATERAL, solve_mode="scene", enable_bilateral_preelimination=True
    )
    plain, plain_q, _ = _bilateral_run(model, steps=720, leader_target=0.6, **BILATERAL, solve_mode="scene")
    steel = SolverSteel(model, **BILATERAL, solve_mode="serial", enable_bilateral_preelimination=True)
    test.assertEqual(alias.solve_mode, "scene")
    test.assertTrue(alias.enable_bilateral_preelimination)
    test.assertFalse(plain.enable_bilateral_preelimination)
    test.assertFalse(alias._preelim_active)
    test.assertTrue(steel._preelim_active)
    for attribute in PREELIMINATION_ONLY:
        test.assertTrue(hasattr(steel, attribute), attribute)
        test.assertFalse(hasattr(alias, attribute), attribute)
    for step, (got, want) in enumerate(zip(alias_q, plain_q, strict=True)):
        test.assertEqual(got.tobytes(), want.tobytes(), f"joint_q differs after step {step + 1}")
    test.assertTrue(np.all(np.isfinite(alias_q[-1])))
    test.assertGreater(abs(float(alias_q[-1][0])), 0.5)
    test.assertLess(alias_gaps[-1], ALIAS_GAP_MAX, f"loop gap {alias_gaps[-1]:.3e} m after 720 steps")


class TestSteelSceneAliases(unittest.TestCase):
    pass


for _fn in (
    test_aliases_read_back_as_requested,
    test_phased_alias_runs_ab_order,
):
    add_function_test(TestSteelSceneAliases, _fn.__name__, _fn, devices=get_test_devices())

# CUDA only, each for the reason beside it (the device lists of the stepping tests are re-decided with the step).
for _fn in (
    test_contact_then_internal_alias_runs,  # Steel's contact_then_internal reference was measured on CUDA
    test_propagation_alias_is_bitwise_immediate_scene,  # Steel's propagation reference was measured on CUDA
    test_connect_rows_track_steel,  # the closure tolerances were measured on CUDA
    test_mimic_rows_track_steel,  # the mimic tolerances were measured on CUDA
    test_connect_release_under_capture,  # graph capture
    test_preelimination_alias_converges,  # Steel's pre-elimination reference was measured on CUDA
):
    add_function_test(TestSteelSceneAliases, _fn.__name__, _fn, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
