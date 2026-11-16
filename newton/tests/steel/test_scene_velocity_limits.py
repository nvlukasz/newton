# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint velocity limits (``enable_joint_velocity_limits``) in Steel's scene solve: the late rows and phase C.

A DOF of a PRISMATIC, REVOLUTE or D6 joint whose :attr:`~newton.Model.joint_velocity_limit` ``qdot_max`` is positive
gets two unilateral rows, the lower bound's (``J = +1`` on the DOF) and the upper's (``J = -1``), both with the target
velocity ``-qdot_max``: Steel's velocity-limit rows, whose stateless update applies only the impulse of the current
overshoot (and stores it as the row's impulse). Steel sweeps them after every other row of the world in each
iteration, the last word. The scene holds an articulation's velocity-limit rows in its *late unit*, which the scans
place after the tail, side-major; each iteration ends with phase C, which sweeps the coupled sides' late rows after the
whole coupled pass (B, A, C under ``"BA"``), and the isolated sides run A then C in each of their sweeps. With drive
rows and ``fuse_joint_velocity_limits`` (Steel's default) a driven DOF gets no velocity-limit rows: phase C clamps its
drive row's velocity statelessly after its side's velocity limits, the drive impulse unchanged, and the scene marks
the driven DOFs before stage 1 (``_drive_slot``), where Steel reserves its drive rows.

Free bodies: with ``rigid_body_max_linear_velocity`` and ``rigid_body_max_angular_velocity`` registered (whatever
``enable_joint_velocity_limits``), a free body whose limit is finite and positive gets Steel's two rows per root DOF
``[lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]``, ``J = +e_axis`` and ``-e_axis`` on the world axis (in the FREE joint's
parent-anchor frame of ``joint_qd``), target ``-limit``: its side's late unit, on its six physical entries, swept in
phase C like an articulation's.

Steel refuses velocity limits on its sparse path, so the scene is compared with Steel's dense matrix-free solve
within the rounding that separates the two factorizations (``test_scene_drives.DRIVE_TRACK_RTOL``), the rows' targets
(which hold no ``J``) bit for bit, and the phases with the serial oracle bit for bit. Free bodies have no factor, so
their rows and steps are compared with Steel's free-body solve within ``TRACK_RTOL`` (1e-5) and their Jacobians bit
for bit.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel import scene_pgs, scene_rows
from newton._src.solvers.steel.kernels import (
    PGS_CONSTRAINT_TYPE_JOINT_TARGET,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
)
from newton._src.solvers.steel.scene_units import UNIT_CONTACT_LATE
from newton._src.solvers.steel.solver_steel import _SOLVE_MODES
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import DT
from newton.tests.steel._scene_fixtures import SCENE, _humanoids, _Run, _scene_heap, _steel_on
from newton.tests.steel.test_scene_drives import CAPTURE_STEPS, DIAGONAL_RTOL, DRIVE_TRACK_RTOL, _set_drives
from newton.tests.steel.test_scene_features import (
    CROWD_ROWS,
    HUMANOID_ROWS,
    TRACK_ATOL,
    TRACK_RTOL,
    TRACK_STEPS,
    _assert_phases_match_serial_oracle,
    _track_steel,
    _within_track_tolerance,
)
from newton.tests.steel.test_scene_internal_unit import HUMANOID_HINGES, _hinge_dofs, _Stepper
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

needs_scene_mode = unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
needs_read_back = unittest.skipUnless(hasattr(SolverSteel, "scene_statistics"), "needs the read-back API")

VELOCITY_LIMIT = 2.0
"""``joint_velocity_limit`` [rad/s] of every hinge of the tracking tests."""

SPIN = 3.0
"""Initial speed [rad/s] of the tracking tests' hinges, alternating in sign: past :data:`VELOCITY_LIMIT`, so Steel's
pre-solve scaling brings them to it and the velocity-limit rows (or the fused clamp) hold them there."""

CROWD_LIMIT = 0.05
"""``joint_velocity_limit`` [rad/s] of the crowd's hinges: below the speeds at which the lying humanoids settle, so the
velocity-limit rows act."""

OVERFLOW_ROWS = 4 * 256
"""Rows of the overflow test's probe: four lying humanoids' contact, limit and velocity-limit rows, with room."""

ACTIVATION_FRACTION = 0.5
"""``velocity_limit_activation_fraction`` of the gated tests: a pair of rows exists while its speed is at least half its
limit."""

FREE_LINEAR_LIMIT = 0.5
"""``rigid_body_max_linear_velocity`` [m/s] of the free-body tests' boxes."""

FREE_ANGULAR_LIMIT = 2.0
"""``rigid_body_max_angular_velocity`` [rad/s] of the free-body tests' boxes."""

FREE_THROW = np.array((1.0, -0.8, 1.2, 3.0, -2.5, 4.0), dtype=np.float32)
"""A box's initial velocity (linear [m/s], angular [rad/s]), past both limits on every axis. The boxes alternate the
signs of all but the vertical component, so every box rises: no box meets the ground in the first steps."""

FREE_PITCH = 0.25
"""Pitch [m] of the free tracking test's 3 x 3 boxes, so that no two boxes meet: clamped to 0.5 m/s, two boxes close
at most 12.5 cm in the test's 30 steps, the gap between them is 21 cm."""

FREE_ROWS = 256
"""``articulation_constraint_max`` of the free-body runs (Steel's free-body budget adds 512 rows per world)."""

FREE_ANCHORS = (
    wp.quat_identity(),
    wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), 1.1),
    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * np.pi),
)
"""The FREE-joint parent-anchor rotations of the rows test's boxes: world-aligned, oblique, and a quarter turn."""

FREE_TWIST = np.array((0.4, 0.1, -0.3, 1.5, -0.2, 0.9), dtype=np.float32)
"""The rows test's world twist: under :data:`ACTIVATION_FRACTION` of the free limits the world axes lin_x, lin_z
and ang_x pass the gate (0.25 m/s, 1 rad/s) and the others do not."""


def _limit_hinges(model, limits) -> np.ndarray:
    """Give the model's hinges the velocity limits ``limits`` (a scalar or one per hinge) and return the hinges."""
    hinges = _hinge_dofs(model)
    values = model.joint_velocity_limit.numpy()
    values[hinges] = limits
    model.joint_velocity_limit.assign(values)
    return hinges


def _spin_hinges(model, state):
    """Spin every hinge at :data:`SPIN` rad/s, alternating in sign, from the model's pose."""
    hinges = _hinge_dofs(model)
    qd = state.joint_qd.numpy()
    qd[hinges] = SPIN * np.where(np.arange(hinges.size) % 2 == 0, 1.0, -1.0)
    state.joint_qd.assign(qd)


def _kept_rows(solver) -> int:
    return int(solver._constraint_count.numpy()[0])


def _velocity_limit_rows(solver) -> np.ndarray:
    """Whether each of the scene's kept rows of the last step is a velocity-limit row."""
    return solver._row_type.numpy()[0, : _kept_rows(solver)] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT


def _steel_rows(solver, array, row_type) -> np.ndarray:
    """``array``'s values over Steel's rows of ``row_type``, world by world, each world's rows in row order."""
    types = solver._row_type.numpy()
    values = array.numpy()
    counts = solver._constraint_count.numpy()
    return np.concatenate([values[world, :rows][types[world, :rows] == row_type] for world, rows in enumerate(counts)])


def _free_body_dofs(model) -> np.ndarray:
    """The first DOF of each free rigid body (a FREE joint alone in its articulation), in articulation order."""
    starts = model.articulation_start.numpy()
    joint_type = model.joint_type.numpy()
    first = [
        starts[art]
        for art in range(model.articulation_count)
        if starts[art + 1] - starts[art] == 1 and joint_type[starts[art]] == int(newton.JointType.FREE)
    ]
    return model.joint_qd_start.numpy()[np.asarray(first, dtype=np.int64)]


def _limit_free_bodies(model, linear=FREE_LINEAR_LIMIT, angular=FREE_ANGULAR_LIMIT):
    """Give every body the rigid-body velocity limits (Steel reads the free bodies')."""
    model.steel.rigid_body_max_linear_velocity.fill_(linear)
    model.steel.rigid_body_max_angular_velocity.fill_(angular)


def _throw_free_bodies(model, state):
    """Throw every free body at :data:`FREE_THROW`, the signs of all but its vertical speed alternating by body."""
    qd = state.joint_qd.numpy()
    flip = np.array((-1.0, -1.0, 1.0, -1.0, -1.0, -1.0), dtype=np.float32)
    for index, dof in enumerate(_free_body_dofs(model)):
        qd[dof : dof + 6] = FREE_THROW * (flip if index % 2 else 1.0)
    state.joint_qd.assign(qd)


def _free_late_arts(solver) -> np.ndarray:
    """The free rigid bodies' articulations."""
    return np.flatnonzero(solver._model_plan.is_free_rigid != 0)


def _velocity_limit_applied(solver) -> bool:
    """Whether a velocity-limit row of the last step applied an impulse (its stored impulse is the last visit's)."""
    rows = _velocity_limit_rows(solver)
    return bool(np.any(solver._impulses.numpy()[0, : rows.size][rows] != 0.0))


@needs_scene_mode
def test_joint_velocity_limits_track_steel(test, device):
    """Hold two standing humanoids' spun hinges at 2 rad/s with velocity-limit rows, as Steel's dense solve does.

    Every hinge of ``_humanoids(worlds=2, lying=False)`` has the velocity limit 2 rad/s and starts spun at 3 rad/s,
    alternating in sign. With velocity limits alone, and with drive rows on the even hinges under the fused clamp
    (Steel's default ``fuse_joint_velocity_limits``), the velocities after the first step agree with Steel's dense
    matrix-free solve within the factorizations' rounding (:data:`DRIVE_TRACK_RTOL`, and 1e-6) and the joint
    coordinates after 30 steps within 1e-3 RMS. Every limited DOF without a fused drive has its two rows in the scene
    as in Steel, and they are the scene's last rows, the late range; with velocity limits alone a velocity-limit row
    applies an impulse at some step. The same scene without velocity limits falls outside that tolerance of Steel's
    limited solve.
    """
    cases = {
        "limits": {},
        "fused": {"drive_mode": "rows"},
    }
    for name, options in cases.items():
        with test.subTest(case=name):
            model = _humanoids(device, worlds=2, lying=False)
            hinges = _limit_hinges(model, VELOCITY_LIMIT)
            driven = hinges[::2] if options else hinges[:0]
            _set_drives(model, driven)
            scene, steel = _track_steel(
                test,
                model,
                init=_spin_hinges,
                rows=HUMANOID_ROWS,
                rtol=DRIVE_TRACK_RTOL,
                witness=None if options else _velocity_limit_applied,
                enable_joint_velocity_limits=True,
                **options,
            )
            solver = scene.solver
            features = solver._scene.features
            test.assertTrue(features.late_rows)
            test.assertEqual(features.fused_clamp, bool(options))
            test.assertEqual(solver.velocity_limits_fused, bool(options))
            test.assertTrue(solver._scene.late_rows_in_use)
            vl_rows = _velocity_limit_rows(solver)
            test.assertEqual(int(np.count_nonzero(vl_rows)), 2 * (hinges.size - driven.size))
            steel_targets = _steel_rows(
                steel.solver, steel.solver._target_velocity, PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
            )
            test.assertEqual(steel_targets.size, int(np.count_nonzero(vl_rows)))
            # The velocity-limit rows are the scene's last rows: the late range, after every other row.
            late_begin, late_end = solver._scene.row_ranges.numpy()[scene_pgs.RANGE_LATE]
            test.assertEqual(late_end, vl_rows.size)
            test.assertTrue(np.all(vl_rows[late_begin:]) and not np.any(vl_rows[:late_begin]))
            drive_rows = solver._row_type.numpy()[0, : vl_rows.size] == PGS_CONSTRAINT_TYPE_JOINT_TARGET
            test.assertEqual(int(np.count_nonzero(drive_rows)), driven.size)

            free = _Run(model, HUMANOID_ROWS, {}, **options)
            _spin_hinges(model, free.state_0)
            first = None
            for step in range(TRACK_STEPS):
                free.step()
                if step == 0:
                    first = free.state_0.joint_qd.numpy()
            test.assertFalse(
                _within_track_tolerance(
                    first,
                    steel.first_qd,
                    free.state_0.joint_q.numpy(),
                    steel.state_0.joint_q.numpy(),
                    rtol=DRIVE_TRACK_RTOL,
                    atol=TRACK_ATOL,
                ),
                "the scene without velocity limits tracks Steel's limited solve",
            )


@needs_scene_mode
def test_velocity_limit_rows_match_steel_rows(test, device):
    """Build Steel's velocity-limit rows, in Steel's order, with Steel's targets bit for bit and its watermarks.

    Two lying humanoids, each hinge with its own velocity limit (one hinge per world unlimited, ``0``, and one
    infinite, whose rows never act but are built, as Steel builds them), step once from the same state and contacts.
    Over the velocity-limit rows, world by world in Steel's slot order (by DOF, the lower bound before the upper):
    the targets ``-qdot_max`` equal Steel's dense rows' bit for bit, ``mu`` and ``phi`` are zero, and the diagonals
    agree within the factorizations' rounding (:data:`DIAGONAL_RTOL`). Each articulation's late unit holds its two
    rows per limited hinge (``_scene_late_rows``), and the rows count in Steel's articulation family per model world:
    the row high-water marks equal Steel's. With the activation fraction 0.5 and the even hinges spun at 3 rad/s, the
    others at 0.3 rad/s, the gate on the predicted velocity builds the pairs of some limited hinges and not of
    others, exactly Steel's rows (an infinite limit builds none under a positive fraction).
    """
    model = _humanoids(device, worlds=2)
    hinges = _hinge_dofs(model)
    index = np.arange(hinges.size) % HUMANOID_HINGES
    limits = (VELOCITY_LIMIT + 0.01 * index).astype(np.float32)
    limits[index == 3] = 0.0
    limits[index == 4] = np.inf
    _limit_hinges(model, limits)
    speeds = np.where(index % 2 == 0, SPIN, 0.1 * SPIN).astype(np.float32)

    def initial_state(spun):
        state = model.state()
        if spun:
            qd = state.joint_qd.numpy()
            qd[hinges] = speeds
            state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        return state

    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    pipeline.collide(initial_state(False), contacts)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    limited = limits > 0.0
    vl = PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
    for fraction in (0.0, ACTIVATION_FRACTION):
        with test.subTest(velocity_limit_activation_fraction=fraction):
            solvers = {}
            for name, options in (("scene", {}), ("steel", {"solve_mode": "serial"})):
                solver = SolverSteel(
                    model,
                    **{**SCENE, **options},
                    articulation_constraint_max=HUMANOID_ROWS,
                    enable_joint_velocity_limits=True,
                    velocity_limit_activation_fraction=fraction,
                    row_watermark=True,
                )
                test.assertIs(solver.solve_mode == "scene", name == "scene")
                solver.step(initial_state(fraction > 0.0), model.state(), model.control(), contacts, DT)
                test.assertFalse(np.any(solver.constraint_overflow.numpy()), f"{name} overflowed")
                solvers[name] = solver
            scene, steel = solvers["scene"], solvers["steel"]
            rows = _velocity_limit_rows(scene)

            def scene_rows(array, rows=rows):
                return array.numpy()[0, : rows.size][rows]

            targets = _steel_rows(steel, steel._target_velocity, vl)
            if fraction == 0.0:
                test.assertEqual(int(np.count_nonzero(rows)), 2 * int(np.count_nonzero(limited)))
                np.testing.assert_array_equal(targets, np.repeat(-limits[limited], 2))
                np.testing.assert_array_equal(scene._scene_late_rows.numpy(), 2 * (HUMANOID_HINGES - 1))
                units = scene._scene_units
                contact = units.unit_contact.numpy()[: int(units.unit_count.numpy()[0])]
                test.assertEqual(int(np.count_nonzero(contact == UNIT_CONTACT_LATE)), 2)
            else:
                # The gate held back some limited hinges' pairs and kept others'.
                test.assertGreater(int(np.count_nonzero(rows)), 0)
                test.assertLess(int(np.count_nonzero(rows)), 2 * int(np.count_nonzero(limited & np.isfinite(limits))))
            for name, values in (("_target_velocity", targets), ("_row_mu", 0.0), ("_phi", 0.0)):
                with test.subTest(array=name):
                    ours, theirs = scene_rows(getattr(scene, name)), _steel_rows(steel, getattr(steel, name), vl)
                    test.assertTrue(np.array_equal(ours, theirs), f"{ours} != {theirs}")
                    test.assertTrue(np.array_equal(ours, np.broadcast_to(np.float32(values), ours.shape)))
            np.testing.assert_allclose(scene_rows(scene._diag), _steel_rows(steel, steel._diag, vl), rtol=DIAGONAL_RTOL)
            marks = scene._row_watermarks.numpy()
            test.assertTrue(
                np.array_equal(marks, steel._row_watermarks.numpy()),
                f"{scene.constraint_row_watermarks()} != {steel.constraint_row_watermarks()}",
            )


@needs_scene_mode
def test_late_rows_follow_the_coupled_pass(test, device):
    """Sweep the late rows in phase C exactly as one thread walking the rows does, in every phase order.

    Four lying humanoids 0.4 m apart overlap, so units join pairs of them; every hinge has the velocity limit
    0.05 rad/s, which the settling humanoids pass. The phases (the default color loop, and the while loop, whose tail
    one thread walks) match the serial oracle (``scene_sweep="serial"``) bit for bit over 30 steps in each phase
    order (B, A, C; A, B, C; A, B, A, C), with velocity limits alone (``_late``) and with drive rows on the even
    hinges under the fused clamp (``_drv_late_fvc``), so the isolated sides' A then C and the coupled sides' C after
    the whole coupled pass are the oracle's row order; some velocity-limit row applies an impulse. The fused clamp
    changes the velocities: the same crowd without it (``fuse_joint_velocity_limits=False``), whose driven hinges have
    no velocity limit and so no velocity-limit rows either, moves its hinges more than 1e-3 rad/s otherwise at some
    step. The free bodies' late rows follow the coupled pass too: a 3 x 2 x 2 box heap with the free velocity limits,
    its boxes thrown into one another past them (``_late`` without ``enable_joint_velocity_limits``), matches the
    oracle in every order.
    """
    model = _humanoids(device, per_world=4, spacing=0.4)
    hinges = _limit_hinges(model, CROWD_LIMIT)
    driven = hinges[::2]
    feature_sets = {
        "_late": {"enable_joint_velocity_limits": True},
        "_drv_late_fvc": {"enable_joint_velocity_limits": True, "drive_mode": "rows"},
    }
    for suffix, options in feature_sets.items():
        with test.subTest(features=suffix):
            _set_drives(model, driven if "drive_mode" in options else hinges[:0])
            probe = _Run(model, CROWD_ROWS, {}, collide=False, **options)
            test.assertTrue(probe.solver._scene.sides_kernel.key.endswith(suffix))
            _assert_phases_match_serial_oracle(
                test, model, options, witness=_velocity_limit_applied, what="no velocity-limit row acted"
            )
    with test.subTest(witness="fused clamp"):
        options = feature_sets["_drv_late_fvc"]
        fused = _Run(model, CROWD_ROWS, {}, **options)
        history = []
        for _ in range(TRACK_STEPS):
            fused.step()
            history.append(fused.state_0.joint_qd.numpy()[driven])
        limits = model.joint_velocity_limit.numpy()
        unlimited = limits.copy()
        unlimited[driven] = 0.0
        model.joint_velocity_limit.assign(unlimited)
        try:
            unfused = _Run(model, CROWD_ROWS, {}, fuse_joint_velocity_limits=False, **options)
            test.assertFalse(unfused.solver._scene.features.fused_clamp)
            change = 0.0
            for expected in history:
                unfused.step()
                change = max(change, float(np.max(np.abs(unfused.state_0.joint_qd.numpy()[driven] - expected))))
            test.assertEqual(
                int(np.count_nonzero(_velocity_limit_rows(unfused.solver))), 2 * (hinges.size - driven.size)
            )
        finally:
            model.joint_velocity_limit.assign(limits)
        test.assertGreater(change, 1.0e-3, "the fused clamp changed no driven hinge's velocity")
    with test.subTest(features="_late", sides="free bodies"):
        heap = _scene_heap(device, nx=3, ny=2, nz=2, register_attributes=True)
        _limit_free_bodies(heap)
        # The runs' states copy the model's velocities: the boxes start thrown into one another.
        _throw_free_bodies(heap, heap)
        _assert_phases_match_serial_oracle(
            test, heap, {}, witness=_velocity_limit_applied, what="no free-body velocity-limit row acted"
        )


@needs_scene_mode
@needs_read_back
def test_late_rows_overflow_in_row_order(test, device):
    """Drop whole late units past the row capacity in row order; no sweep reads a dropped late row.

    Four lying humanoids 0.4 m apart with velocity limits on every hinge: a probe with room for every row gives the
    first late row ``L`` and the late units' 42 rows each. A capacity of ``L + 100`` rows ends inside the third late
    unit, so the first two late units are kept and the last two dropped whole: ``scene_statistics()["dropped_rows"]``
    counts their 84 rows and ``constraint_overflow`` is latched. The kept rows are the probe's first rows bit for bit
    (types, supports, targets and diagonals), the late ranges of the dropped sides are empty, so the kept late rows are
    a row-order prefix. In every phase order, a run whose rows from ``L`` on (impulses, diagonals, right-hand sides,
    incident velocities and responses) are poisoned with NaN before the step keeps finite coordinates, velocities and
    kept impulses: no sweep reads a row past the kept ones. The runs do not print the overflow warning
    (``warn_constraint_overflow=False``); ``test_scene_features`` tests it.
    """
    model = _humanoids(device, per_world=4, spacing=0.4)
    _limit_hinges(model, VELOCITY_LIMIT)
    options = {"enable_joint_velocity_limits": True}
    probe = _Run(model, OVERFLOW_ROWS, {}, **options)
    probe.step()
    test.assertFalse(np.any(probe.solver.constraint_overflow.numpy()))
    test.assertEqual(probe.solver.scene_statistics()["dropped_rows"], 0)
    first_late, rows = (int(v) for v in probe.solver._scene.row_ranges.numpy()[scene_pgs.RANGE_LATE])
    late_rows = 2 * HUMANOID_HINGES
    test.assertEqual(rows - first_late, 4 * late_rows)
    capacity = first_late + 100
    kept = first_late + 2 * late_rows

    for order in ("BA", "AB", "ABA"):
        with test.subTest(scene_phase_order=order):
            run = _Run(model, capacity, {}, scene_phase_order=order, warn_constraint_overflow=False, **options)
            solver = run.solver
            for name in ("_impulses", "_diag", "_rhs", "_sparse_row_incident"):
                values = getattr(solver, name).numpy()
                values[0, first_late:] = np.nan
                getattr(solver, name).assign(values)
            factor = solver._sparse_row_factor.numpy()
            factor[0, first_late:] = np.nan
            solver._sparse_row_factor.assign(factor)
            run.step()
            stats = solver.scene_statistics()
            test.assertEqual(stats["rows"], kept)
            test.assertEqual(stats["dropped_rows"], 2 * late_rows)
            test.assertTrue(bool(solver.constraint_overflow.numpy()[0]))
            for name in ("_row_type", "_target_velocity", "_diag"):
                ours = getattr(solver, name).numpy()[0, :kept]
                theirs = getattr(probe.solver, name).numpy()[0, :kept]
                test.assertTrue(np.array_equal(ours, theirs), name)
            test.assertTrue(
                np.array_equal(solver._sparse_row_dof.numpy()[0, :kept], probe.solver._sparse_row_dof.numpy()[0, :kept])
            )
            begin = solver._scene.side_late_begin.numpy()
            end = solver._scene.side_late_end.numpy()
            np.testing.assert_array_equal(end - begin, [late_rows, late_rows, 0, 0])
            np.testing.assert_array_equal(begin[:2], [first_late, first_late + late_rows])
            test.assertTrue(np.all(np.isfinite(run.state_0.joint_q.numpy())))
            test.assertTrue(np.all(np.isfinite(run.state_0.joint_qd.numpy())))
            test.assertTrue(np.all(np.isfinite(solver._impulses.numpy()[0, :kept])))


@needs_scene_mode
def test_driven_dof_marker_matches_the_drive_rows(test, device):
    """Mark exactly the DOFs the drive family builds a row for, and give the other limited DOFs velocity-limit rows.

    Two lying humanoids with drive rows of mixed gains (per world one hinge undriven, one driven by its damping alone
    and one by its stiffness alone) and a distinct velocity limit and target velocity per hinge, so a drive row's
    target velocity and a velocity-limit row's target name their DOF. Under the fused clamp the scene marks
    ``_drive_slot`` before stage 1: ``0`` (a marker, not a row index) at exactly the DOFs that own a drive row after
    the step, ``-1`` everywhere else, and the velocity-limit rows are those of the limited hinges without a drive row,
    two each. Without the fused clamp (``fuse_joint_velocity_limits=False``) nothing is marked and every limited hinge
    has its velocity-limit rows.
    """
    model = _humanoids(device, worlds=2)
    hinges = _hinge_dofs(model)
    index = np.arange(hinges.size) % HUMANOID_HINGES
    ke = np.full(hinges.size, 20.0, dtype=np.float32)
    kd = np.full(hinges.size, 1.0, dtype=np.float32)
    ke[index == 3], kd[index == 3] = 0.0, 0.0
    ke[index == 4] = 0.0
    kd[index == 5] = 0.0
    _set_drives(model, hinges, ke=ke, kd=kd)
    target_qd = model.joint_target_qd.numpy()
    target_qd[hinges] = (0.001 * (np.arange(hinges.size) + 1)).astype(np.float32)
    model.joint_target_qd.assign(target_qd)
    limits = (VELOCITY_LIMIT + 0.01 * np.arange(hinges.size)).astype(np.float32)
    _limit_hinges(model, limits)
    driven = hinges[(ke > 0.0) | (kd > 0.0)]
    test.assertEqual(driven.size, 2 * (HUMANOID_HINGES - 1))

    for fused in (True, False):
        with test.subTest(fuse_joint_velocity_limits=fused):
            run = _Run(
                model,
                HUMANOID_ROWS,
                {},
                drive_mode="rows",
                enable_joint_velocity_limits=True,
                fuse_joint_velocity_limits=fused,
            )
            solver = run.solver
            test.assertEqual(solver.velocity_limits_fused, fused)
            run.step()
            test.assertFalse(np.any(solver.constraint_overflow.numpy()))
            kept = _kept_rows(solver)
            row_type = solver._row_type.numpy()[0, :kept]
            targets = solver._target_velocity.numpy()[0, :kept]
            owners = np.flatnonzero(np.isin(target_qd, targets[row_type == PGS_CONSTRAINT_TYPE_JOINT_TARGET]))
            test.assertTrue(np.array_equal(owners, driven), f"{owners} != {driven}")
            marker = solver._drive_slot.numpy()
            if fused:
                np.testing.assert_array_equal(np.flatnonzero(marker == 0), driven)
                np.testing.assert_array_equal(marker[marker != 0], -1)
            else:
                np.testing.assert_array_equal(marker, -1)
            limited = hinges if not fused else np.setdiff1d(hinges, driven)
            vl_targets = targets[row_type == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT]
            test.assertEqual(vl_targets.size, 2 * limited.size)
            np.testing.assert_array_equal(vl_targets, np.repeat(-limits[np.searchsorted(hinges, limited)], 2))


@needs_scene_mode
def test_velocity_limits_with_wide_masks_are_bitwise_one_word(test, device):
    """Build velocity-limit and fused drive rows and step bit for bit with two-word support masks.

    The test-only override ``scene_mask_words=2`` gives the humanoids' factor group two-word masks, although they
    fit one, so the multi-word builders (``build_sparse_joint_velocity_limit_rows_w2`` and
    ``build_sparse_drive_rows_fused_w2``) run where the one-word builders are the reference: two standing humanoids
    with spun hinges, velocity limits on every hinge and drive rows on the even ones step to the same states, bit for
    bit, at each of 20 steps.
    """
    model = _humanoids(device, worlds=2, lying=False)
    hinges = _limit_hinges(model, VELOCITY_LIMIT)
    _set_drives(model, hinges[::2])
    options = {"drive_mode": "rows", "enable_joint_velocity_limits": True}
    one = _Run(model, HUMANOID_ROWS, {}, **options)
    two = _Run(model, HUMANOID_ROWS, {"scene_mask_words": 2}, **options)
    for arrays in two.solver._scene.group_arrays:
        test.assertEqual(arrays.late_kernel.key, "build_sparse_joint_velocity_limit_rows_w2")
        test.assertEqual(arrays.internal_kernels[scene_rows.FAMILY_DRIVE].key, "build_sparse_drive_rows_fused_w2")
    for run in (one, two):
        _spin_hinges(model, run.state_0)
    for step in range(20):
        one.step()
        two.step()
        test.assertEqual(int(np.count_nonzero(_velocity_limit_rows(two.solver))), hinges.size)
        for name in ("joint_q", "joint_qd"):
            a, b = getattr(one.state_0, name).numpy(), getattr(two.state_0, name).numpy()
            test.assertTrue(np.array_equal(a, b), f"{name} differs after step {step + 1}")


def _anchored_boxes(device) -> newton.Model:
    """One free box per :data:`FREE_ANCHORS` rotation of its FREE joint's parent anchor, 1 m up, at :data:`FREE_TWIST`.

    Each box (half extents 5, 7 and 9 cm, so its inertia is anisotropic in every anchor frame) is a link with a FREE
    joint whose parent anchor sits off the origin with its rotation; its ``joint_q`` keeps the box world-aligned and
    its ``joint_qd`` holds the world twist :data:`FREE_TWIST` in the anchor frame (``test_contacts._set_free_anchor``'s
    convention). Steel's custom attributes are registered and every box has the free velocity limits.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverSteel.register_custom_attributes(builder)
    builder.add_ground_plane()
    anchors = []
    for index, rotation in enumerate(FREE_ANCHORS):
        pose = wp.transform(wp.vec3(0.4 * index, 0.0, 1.0), wp.quat_identity())
        body = builder.add_link(xform=pose)
        builder.add_shape_box(body, hx=0.05, hy=0.07, hz=0.09)
        anchor = wp.transform(wp.vec3(0.1 * index, -0.2, 0.3), rotation)
        builder.add_articulation([builder.add_joint_free(body, parent_xform=anchor)])
        anchors.append((anchor, pose))
    model = builder.finalize(device=device)
    _limit_free_bodies(model)
    joint_q, joint_qd = model.joint_q.numpy(), model.joint_qd.numpy()
    q_start, qd_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    for joint, (anchor, pose) in enumerate(anchors):
        local = wp.transform_inverse(anchor) * pose
        joint_q[q_start[joint] : q_start[joint] + 7] = [*local.p, *local.q]
        to_anchor = np.array(wp.quat_to_matrix(anchor.q)).reshape(3, 3).T
        joint_qd[qd_start[joint] : qd_start[joint] + 3] = to_anchor @ FREE_TWIST[:3]
        joint_qd[qd_start[joint] + 3 : qd_start[joint] + 6] = to_anchor @ FREE_TWIST[3:]
    model.joint_q.assign(joint_q)
    model.joint_qd.assign(joint_qd)
    return model


@needs_scene_mode
@needs_read_back
def test_free_body_velocity_limits_track_steel(test, device):
    """Hold thrown boxes at 0.5 m/s and 2 rad/s with their velocity-limit rows, as Steel's free-body solve does.

    A 3 x 3 layer of the box heap (``_scene_heap(register_attributes=True)``, 25 cm apart so that no two boxes meet) has
    the rigid-body velocity limits 0.5 m/s and 2 rad/s on every box, without ``enable_joint_velocity_limits``, and every
    box is thrown upward past both limits on every axis (:data:`FREE_THROW`). Each box's side has a late unit of Steel's
    twelve rows (two per root DOF) on its six physical entries, the scene's last rows, as many as Steel's free-body
    velocity-limit rows; a velocity-limit row applies an impulse, and the velocities after the first step are at the
    limits (the angular ones then damped by the boxes' ``rigid_body_angular_damping``, ``w *= 1 - d dt``). They agree
    with Steel's matrix-free solve within 1e-5 (and 1e-6), as tight as without velocity limits (free bodies have no
    factor), and the coordinates after 30 steps within 1e-3 RMS. The same boxes without limits fall outside that
    tolerance of Steel's limited solve.
    """
    model = _scene_heap(device, nx=3, ny=3, nz=1, pitch=FREE_PITCH, register_attributes=True)
    _limit_free_bodies(model)
    scene, steel = _track_steel(
        test, model, init=_throw_free_bodies, rows=FREE_ROWS, rtol=TRACK_RTOL, witness=_velocity_limit_applied
    )
    solver = scene.solver
    test.assertFalse(solver._enable_joint_velocity_limits)
    test.assertTrue(solver._scene.features.late_rows)
    test.assertIs(solver._scene.free_late_kernel, scene_rows.build_sparse_free_velocity_limit_rows)
    test.assertEqual(solver._scene.group_arrays, ())
    boxes = model.articulation_count
    vl_rows = _velocity_limit_rows(solver)
    test.assertEqual(int(np.count_nonzero(vl_rows)), 12 * boxes)
    mf_count = int(steel.solver._mf_constraint_count.numpy()[0])
    mf_types = steel.solver._mf_row_type.numpy()[0, :mf_count]
    test.assertEqual(int(np.count_nonzero(mf_types == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)), 12 * boxes)
    late_begin, late_end = solver._scene.row_ranges.numpy()[scene_pgs.RANGE_LATE]
    test.assertEqual(late_end, vl_rows.size)
    test.assertTrue(np.all(vl_rows[late_begin:]) and not np.any(vl_rows[:late_begin]))
    np.testing.assert_array_equal(solver._scene_late_rows.numpy(), 12)
    test.assertEqual(solver.scene_statistics()["late_units"], boxes)
    np.testing.assert_array_equal(solver._scene.row_free_prefix.numpy()[late_begin:late_end], 6)
    first = scene.first_qd.reshape(boxes, 6)
    damped = FREE_ANGULAR_LIMIT * (1.0 - model.steel.rigid_body_angular_damping.numpy()[:, None] * DT)
    np.testing.assert_allclose(np.abs(first[:, :3]), FREE_LINEAR_LIMIT, rtol=TRACK_RTOL)
    np.testing.assert_allclose(np.abs(first[:, 3:]), np.broadcast_to(damped, (boxes, 3)), rtol=TRACK_RTOL)

    _limit_free_bodies(model, np.inf, np.inf)
    free = _Run(model, FREE_ROWS, {})
    test.assertFalse(free.solver._scene.features.late_rows)
    _throw_free_bodies(model, free.state_0)
    first = None
    for step in range(TRACK_STEPS):
        free.step()
        if step == 0:
            first = free.state_0.joint_qd.numpy().copy()
    test.assertFalse(
        _within_track_tolerance(first, steel.first_qd, free.state_0.joint_q.numpy(), steel.state_0.joint_q.numpy()),
        "the boxes without velocity limits track Steel's limited solve",
    )


@needs_scene_mode
@needs_read_back
def test_free_body_late_units_overflow_in_row_order(test, device):
    """Drop whole free-body late units past the row capacity in row order; no sweep reads a dropped late row.

    The tracking test's nine thrown boxes: a probe with room for every row gives the first late row ``L`` and twelve
    late rows per box. A capacity of ``L + 54`` rows ends inside the fifth box's late unit, so four late units are kept
    and five dropped whole: ``scene_statistics()["dropped_rows"]`` counts their 60 rows, ``constraint_overflow`` is
    latched, the kept rows are the probe's (types, supports, Jacobians, targets, diagonals) bit for bit, and the late
    ranges of the dropped boxes are empty. Rows from ``L + 48`` on poisoned with NaN before the step (impulses,
    diagonals, right-hand sides, incident velocities and responses) leave the state finite.
    """
    model = _scene_heap(device, nx=3, ny=3, nz=1, pitch=FREE_PITCH, register_attributes=True)
    _limit_free_bodies(model)
    probe = _Run(model, FREE_ROWS, {})
    _throw_free_bodies(model, probe.state_0)
    probe.step()
    first_late, rows = (int(v) for v in probe.solver._scene.row_ranges.numpy()[scene_pgs.RANGE_LATE])
    boxes = model.articulation_count
    test.assertEqual(rows - first_late, 12 * boxes)
    kept = first_late + 4 * 12
    # One articulation row and the rest of the budget in the free-body rows, summed into the scene's capacity.
    run = _Run(model, 1, {}, free_body_constraint_max=kept + 5, warn_constraint_overflow=False)
    _throw_free_bodies(model, run.state_0)
    solver = run.solver
    test.assertEqual(solver._scene.row_max, kept + 6)
    for name in ("_impulses", "_diag", "_rhs", "_sparse_row_incident"):
        values = getattr(solver, name).numpy()
        values[0, kept:] = np.nan
        getattr(solver, name).assign(values)
    for name in ("_sparse_row_factor", "_sparse_row_free_response"):
        values = getattr(solver, name).numpy()
        values[0, kept:] = np.nan
        getattr(solver, name).assign(values)
    run.step()
    stats = solver.scene_statistics()
    test.assertEqual(stats["rows"], kept)
    test.assertEqual(stats["dropped_rows"], 12 * (boxes - 4))
    # The heap's boxes are global bodies (world -1): the overflow latches the global entry.
    test.assertTrue(bool(solver.constraint_overflow.numpy()[-1]))
    for name in ("_row_type", "_target_velocity", "_diag", "_sparse_row_dof", "_sparse_row_factor"):
        ours, theirs = getattr(solver, name).numpy()[0, :kept], getattr(probe.solver, name).numpy()[0, :kept]
        test.assertTrue(np.array_equal(ours, theirs), name)
    begin, end = solver._scene.side_late_begin.numpy(), solver._scene.side_late_end.numpy()
    np.testing.assert_array_equal(end - begin, [12] * 4 + [0] * (boxes - 4))
    for name in ("joint_q", "joint_qd"):
        test.assertTrue(np.all(np.isfinite(getattr(run.state_0, name).numpy())), name)
    test.assertTrue(np.all(np.isfinite(solver._impulses.numpy()[0, :kept])))


@needs_scene_mode
def test_free_body_velocity_limit_rows_match_steel_rows(test, device):
    """Build Steel's free-body velocity-limit rows on rotated anchors under the activation gate, Jacobians bit for bit.

    Three boxes whose FREE joints' parent anchors are world-aligned, oblique and a quarter turn about z move at the
    world twist :data:`FREE_TWIST` (in each anchor frame) 1 m above the ground; the last box has no angular limit
    (``inf``), a limit of one kind only. With the activation fraction 0.5 the gate reads each box's world-axis speeds,
    so lin_x, lin_z and ang_x build their pairs and the other axes none, on every anchor: six rows per box (four, the
    linear ones, for the last), in Steel's slot order (root DOF, then lower before upper; Steel's reservation
    interleaves the boxes' rows, the scene keeps each box's together). After one step from the same state, box by box,
    the rows' Jacobians on the box's six physical entries equal Steel's free-body rows' (world axes turned into the
    anchor frame), bit for bit on the world-aligned anchor and within 1e-7 on the rotated ones (the two kernels' fused
    multiply-adds in ``quat_rotate_inv`` may round one ulp apart), ``-target`` equals the limit Steel keeps in
    ``mf_phi`` bit for bit, ``mu`` and ``phi`` are zero, and the diagonals agree with Steel's ``1 / mf_eff_mass_inv``
    within 1e-5. The rows count in Steel's free-body family: the row high-water marks equal Steel's.
    """
    model = _anchored_boxes(device)
    boxes = len(FREE_ANCHORS)
    angular = model.steel.rigid_body_max_angular_velocity.numpy()
    angular[boxes - 1] = np.inf
    model.steel.rigid_body_max_angular_velocity.assign(angular)
    # lin_x, lin_z and ang_x of each box, the lower bound before the upper; the last box's linear rows only.
    world_axes = [np.repeat([0, 2, 3], 2)] * (boxes - 1) + [np.repeat([0, 2], 2)]
    per_box = np.array([axes.size for axes in world_axes])
    box_of_row = np.repeat(np.arange(boxes), per_box)
    world_axes = np.concatenate(world_axes)
    limits = np.where(world_axes < 3, np.float32(FREE_LINEAR_LIMIT), np.float32(FREE_ANGULAR_LIMIT))
    solvers = {}
    for name, options in (("scene", {}), ("steel", {"solve_mode": "serial"})):
        solver = SolverSteel(
            model,
            **_steel_on(model, {**SCENE, **options}),
            articulation_constraint_max=FREE_ROWS,
            velocity_limit_activation_fraction=ACTIVATION_FRACTION,
            row_watermark=True,
        )
        test.assertIs(solver.solve_mode == "scene", name == "scene")
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        solver.step(state, model.state(), model.control(), None, DT)
        test.assertFalse(np.any(solver.constraint_overflow.numpy()), f"{name} overflowed")
        solvers[name] = solver
    scene, steel = solvers["scene"], solvers["steel"]
    rows = np.flatnonzero(_velocity_limit_rows(scene))
    test.assertEqual(rows.size, world_axes.size)
    np.testing.assert_array_equal(scene._scene_late_rows.numpy(), per_box)
    mf_count = int(steel._mf_constraint_count.numpy()[0])
    mf_rows = np.flatnonzero(steel._mf_row_type.numpy()[0, :mf_count] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
    test.assertEqual(mf_rows.size, rows.size)
    # Steel's rows box by box, each box's in its slot order.
    mf_rows = mf_rows[np.argsort(steel._mf_body_a.numpy()[0, mf_rows], kind="stable")]
    np.testing.assert_array_equal(steel._mf_body_a.numpy()[0, mf_rows], box_of_row)
    ours = scene._sparse_row_factor.numpy()[0, rows, :6]
    theirs = steel._mf_J_a.numpy()[0, mf_rows]
    aligned = box_of_row == 0
    test.assertTrue(np.array_equal(ours[aligned], theirs[aligned]), f"{ours[aligned]} != {theirs[aligned]}")
    np.testing.assert_allclose(ours, theirs, rtol=0.0, atol=1.0e-7)
    offsets = scene._articulation_world_dof_offset.numpy()
    expected_dofs = offsets[box_of_row][:, None] + np.arange(6)
    np.testing.assert_array_equal(scene._sparse_row_dof.numpy()[0, rows, :6], expected_dofs)
    np.testing.assert_array_equal(scene._sparse_row_dof.numpy()[0, rows, 6:], -1)
    np.testing.assert_array_equal(-scene._target_velocity.numpy()[0, rows], limits)
    np.testing.assert_array_equal(steel._mf_phi.numpy()[0, mf_rows], limits)
    np.testing.assert_array_equal(scene._row_mu.numpy()[0, rows], 0.0)
    np.testing.assert_array_equal(scene._phi.numpy()[0, rows], 0.0)
    signs = np.tile([1.0, -1.0], rows.size // 2)
    for index in range(rows.size):
        # J is +-e_axis on the world axis, expressed in the box's anchor frame.
        rotation = np.array(wp.quat_to_matrix(FREE_ANCHORS[box_of_row[index]])).reshape(3, 3)
        axis = world_axes[index]
        block = slice(0, 3) if axis < 3 else slice(3, 6)
        np.testing.assert_allclose(ours[index, block], signs[index] * rotation[axis % 3], atol=1.0e-6)
    np.testing.assert_allclose(
        scene._diag.numpy()[0, rows], 1.0 / steel._mf_eff_mass_inv.numpy()[0, mf_rows], rtol=TRACK_RTOL
    )
    marks = scene._row_watermarks.numpy()
    test.assertTrue(
        np.array_equal(marks, steel._row_watermarks.numpy()),
        f"{scene.constraint_row_watermarks()} != {steel.constraint_row_watermarks()}",
    )


@needs_scene_mode
@needs_read_back
def test_late_rows_change_under_capture(test, device):
    """Replay one captured step exactly as eager steps while late units come and go under the activation gate.

    Two standing humanoids with a free box each (``_humanoids(props=True)``) have joint velocity limits of 2 rad/s on
    every hinge and the box velocity limits 0.5 m/s and 2 rad/s, with the activation fraction 0.5: a pair of rows
    exists while its speed is at least half its limit. The hinges start spun at 3 rad/s and the boxes thrown past
    their limits, and both again after 30 steps, so the late units of the humanoids and the boxes appear as they
    speed up and disappear as they slow down while one captured graph replays the step. ``joint_q``, ``joint_qd``,
    ``body_q`` and ``body_qd`` equal the eager steps' at every step, the eager steps' late units take at least two
    counts, and each box's late unit is present at some step and absent at another.
    """
    model = _humanoids(device, worlds=2, lying=False, props=True)
    _limit_hinges(model, VELOCITY_LIMIT)
    _limit_free_bodies(model)
    options = {"enable_joint_velocity_limits": True, "velocity_limit_activation_fraction": ACTIVATION_FRACTION}
    eager = _Stepper(model, **options)
    captured = _Stepper(model, **options)
    boxes = _free_late_arts(eager.solver)
    test.assertEqual(boxes.size, 2)
    test.assertIsNotNone(eager.solver._scene.free_late_kernel)

    def kick():
        for stepper in (eager, captured):
            _spin_hinges(model, stepper.state_0)
            _throw_free_bodies(model, stepper.state_0)

    def assert_equal(step):
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            a, b = getattr(eager.state_0, name).numpy(), getattr(captured.state_0, name).numpy()
            test.assertTrue(
                np.array_equal(a, b), f"{name} differs after step {step}: max |diff| {np.max(np.abs(a - b)):.3e}"
            )

    kick()
    eager.step()
    captured.step()
    assert_equal(1)
    with wp.ScopedCapture(device) as recorded:
        captured.step()
    late_units, box_late = set(), []
    for index in range(1, CAPTURE_STEPS):
        if index == CAPTURE_STEPS // 2:
            kick()
        eager.step()
        wp.capture_launch(recorded.graph)
        assert_equal(index + 1)
        late_units.add(eager.solver.scene_statistics()["late_units"])
        box_late.append(eager.solver._scene_late_rows.numpy()[boxes] > 0)
    box_late = np.asarray(box_late)
    test.assertGreaterEqual(len(late_units), 2, msg=f"late unit counts {sorted(late_units)}")
    test.assertTrue(np.all(np.any(box_late, axis=0)), "a box never had a late unit")
    test.assertTrue(np.all(np.any(~box_late, axis=0)), "a box always had a late unit")
    test.assertTrue(np.all(np.isfinite(captured.state_0.joint_q.numpy())))
    test.assertFalse(np.any(eager.solver.constraint_overflow.numpy()))


class TestSteelSceneVelocityLimits(unittest.TestCase):
    pass


# On every device; Steel's free-body velocity limits are its serial solve's too (_steel_on).
for _fn in (
    test_late_rows_overflow_in_row_order,
    test_driven_dof_marker_matches_the_drive_rows,
    test_velocity_limits_with_wide_masks_are_bitwise_one_word,
    test_free_body_velocity_limits_track_steel,
    test_free_body_late_units_overflow_in_row_order,
    test_free_body_velocity_limit_rows_match_steel_rows,
):
    add_function_test(TestSteelSceneVelocityLimits, _fn.__name__, _fn, devices=get_test_devices())


# CUDA only, each for the reason beside it: the references are Steel's CUDA joint velocity limits; test_scene_cpu steps
# the scene's on the CPU against CUDA's.
for _fn in (
    test_joint_velocity_limits_track_steel,  # Steel's joint velocity limits
    test_velocity_limit_rows_match_steel_rows,  # Steel's joint velocity limits
    test_late_rows_follow_the_coupled_pass,  # the phases; the CPU sweeps the serial oracle alone
    test_late_rows_change_under_capture,  # graph capture
):
    add_function_test(TestSteelSceneVelocityLimits, _fn.__name__, _fn, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
