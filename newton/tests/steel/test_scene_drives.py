# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drive rows (``drive_mode="rows"``) in Steel's scene solve (``solve_mode="scene"``).

A DOF of a PRISMATIC, REVOLUTE or D6 joint with positive :attr:`~newton.Model.joint_target_ke` or
:attr:`~newton.Model.joint_target_kd` gets one row of ``J = 1`` on the DOF, the first rows of its
articulation's internal unit (Steel's slot order: drives before mimic, connect, joint limits and joint
friction). Steel's drive descriptors (``kernels.compute_physx_pgs_drive_desc``) turn each drive row's parameters and
diagonal into four descriptors, which the scene computes row by row once the diagonal is final
(``scene_solve.scene_drive_descriptors``); the sweeps then replace a drive row's impulse with the force-drive update
of Steel's drive descriptors ``old * impulse_multiplier + (J v) * vel_multiplier + target_vel_bias``, clamped to
the row's maximum impulse ``joint_effort_limit * dt`` and not relaxed.

Steel builds drive rows on its dense response storage (its sparse path refuses them), so the scene is
compared with Steel's matrix-free solve within a tolerance, the drive parameters (which hold no ``J``)
bit for bit, the descriptors' arithmetic bit for bit (Steel's own kernel on the scene's rows), and the
descriptors, which read the row's diagonal, within the rounding that separates the two factorizations
(:data:`DIAGONAL_RTOL`). A drive row's velocity is its response times its impulse, so that rounding
reaches the driven joints' velocities: the step-1 velocities track Steel within :data:`DRIVE_TRACK_RTOL`.
``velocity_drive_mode``, which acts in the velocity-only iterations, is tested in
``test_scene_velocity_iterations.py``; joint velocity limits beside drive rows, with or without the fused clamp, in
``test_scene_velocity_limits.py``.

These tests sit beside ``test_scene_internal_unit.py``, whose internal unit the drive rows join.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel import scene_rows
from newton._src.solvers.steel.kernels import (
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_TARGET,
    compute_physx_pgs_drive_desc,
)
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import DT
from newton.tests.steel._scene_fixtures import SCENE, _humanoids, _Run
from newton.tests.steel.test_scene_features import (
    CROWD_ROWS,
    HUMANOID_ROWS,
    TRACK_ATOL,
    TRACK_STEPS,
    _assert_phases_match_serial_oracle,
    _track_steel,
    _within_track_tolerance,
)
from newton.tests.steel.test_scene_internal_unit import (
    HUMANOID_HINGES,
    _at_rest,
    _hinge_dofs,
    _set_friction,
    _Stepper,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

DRIVE_KE = 20.0
DRIVE_KD = 1.0
"""Target stiffness [N·m/rad] and damping [N·m·s/rad] of every driven hinge."""

DRIVE_TARGET = 0.2
"""Position target [rad] of every driven hinge."""

CLAMP_EFFORT = 0.5
"""``joint_effort_limit`` [N·m] of the oracle test's even hinges: far below the drive's 4 N·m at the target's 0.2 rad,
so their rows end on the maximum impulse ``0.5 dt``."""

REGULARIZATION = 0.5
"""``contact_regularization`` of the oracle test's combined feature set."""

FRICTION = 0.5
"""``joint_friction`` [N·m] of every hinge of the oracle test's combined feature set."""

DIAGONAL_RTOL = 3.0e-5
"""Relative tolerance of the drive rows' diagonals (and the impulse multiplier ``1 - x``, which follows them) against
Steel's. Steel's drive rows take their response from its dense Cholesky factor, the scene's from the sparse factor;
on the humanoid (``cond(H)`` about 1.1e4 in float32) the two diagonals differ by up to 1.5e-5. Against a float64
inverse of the same mass matrix (``H`` plus armature) the scene's err by at most 5.3e-6 and Steel's by 1.0e-5."""

DRIVE_TRACK_RTOL = 5.0e-5
"""Relative tolerance of the velocities after the first step against Steel's, with drive rows: the response rounding of
:data:`DIAGONAL_RTOL` reaches a driven joint's velocity, response times impulse (2.6e-5 measured, at DOF 7 of each
humanoid, whose diagonals differ by 1.5e-5); without drive rows the same two factors put the hinges of the standing
humanoids, spun at 0.5 rad/s, 3.4e-5 apart after one step."""

CAPTURE_STEPS = 60
"""Steps of the capture test: four phases of 15 steps with drives on every hinge, every other hinge, no hinge and
every hinge again, the targets changing sign at every phase."""


def _hinge_coordinates(model, dofs) -> np.ndarray:
    """The joint coordinates (``joint_q`` and ``joint_target_q`` index) of hinge DOFs ``dofs`` (REVOLUTE and D6)."""
    qd_start = model.joint_qd_start.numpy()
    q_start = model.joint_q_start.numpy()
    joint = np.searchsorted(qd_start, dofs, side="right") - 1
    return q_start[joint] + (dofs - qd_start[joint])


def _set_drives(model, dofs, *, ke=DRIVE_KE, kd=DRIVE_KD, target=DRIVE_TARGET, effort=None):
    """Drive ``dofs`` with gains ``ke``, ``kd`` to ``target`` (scalars or per DOF) and every other DOF not at all.

    Writes the model's gains and its position targets in place (a :meth:`~newton.Model.control` made after
    this copies the targets); ``effort`` (per DOF) replaces the effort limits of ``dofs`` when given.
    """
    ke_all = np.zeros(model.joint_dof_count, dtype=np.float32)
    kd_all = np.zeros(model.joint_dof_count, dtype=np.float32)
    ke_all[dofs] = ke
    kd_all[dofs] = kd
    model.joint_target_ke.assign(ke_all)
    model.joint_target_kd.assign(kd_all)
    targets = model.joint_target_q.numpy()
    targets[_hinge_coordinates(model, dofs)] = target
    model.joint_target_q.assign(targets)
    if effort is not None:
        limits = model.joint_effort_limit.numpy()
        limits[dofs] = effort
        model.joint_effort_limit.assign(limits)


def _drive_rows(solver) -> np.ndarray:
    """Whether each of the scene's rows of the last step is a drive row."""
    rows = int(solver._constraint_count.numpy()[0])
    return solver._row_type.numpy()[0, :rows] == PGS_CONSTRAINT_TYPE_JOINT_TARGET


def _steel_drive_rows(solver, array) -> np.ndarray:
    """``array``'s values over Steel's drive rows, world by world, each world's rows in row order."""
    row_type = solver._row_type.numpy()
    values = array.numpy()
    return np.concatenate(
        [
            values[world, :rows][row_type[world, :rows] == PGS_CONSTRAINT_TYPE_JOINT_TARGET]
            for world, rows in enumerate(solver._constraint_count.numpy())
        ]
    )


def test_drive_rows_track_steel(test, device):
    """Drive the hinges of two standing humanoids to 0.2 rad with drive rows, as Steel's matrix-free solve does.

    Every hinge of ``_humanoids(worlds=2, lying=False)`` has the target stiffness 20 N·m/rad, the damping
    1 N·m·s/rad and the position target 0.2 rad (past the upper limit of some hinges, whose limit rows then
    hold their drive). From rest under gravity, the velocities after the first step agree with Steel's
    matrix-free solve within 5e-5 (and 1e-6, :data:`DRIVE_TRACK_RTOL`), and the joint coordinates after
    30 steps within 1e-3 RMS. Every hinge has its drive row, in the scene as in Steel, the first rows of its
    articulation's internal unit. The same scene without drive rows falls outside that tolerance of Steel's
    driven solve.
    """
    model = _humanoids(device, worlds=2, lying=False)
    hinges = _hinge_dofs(model)
    _set_drives(model, hinges)
    scene, steel = _track_steel(
        test, model, init=_at_rest, rows=HUMANOID_ROWS, rtol=DRIVE_TRACK_RTOL, drive_mode="rows"
    )
    solver = scene.solver
    test.assertTrue(solver._scene.features.drives)
    test.assertEqual(solver._scene.internal_families[0], scene_rows.FAMILY_DRIVE)
    test.assertEqual(int(np.count_nonzero(_drive_rows(solver))), 2 * HUMANOID_HINGES)
    test.assertEqual(_steel_drive_rows(steel.solver, steel.solver._diag).size, 2 * HUMANOID_HINGES)
    row_type = solver._row_type.numpy()[0, : int(solver._constraint_count.numpy()[0])]
    for begin in solver._scene.side_row_begin.numpy():
        np.testing.assert_array_equal(row_type[begin : begin + HUMANOID_HINGES], PGS_CONSTRAINT_TYPE_JOINT_TARGET)
    test.assertTrue(np.any(row_type == PGS_CONSTRAINT_TYPE_JOINT_LIMIT), "no limit row holds a drive")

    _set_drives(model, hinges, ke=0.0, kd=0.0)
    undriven = _Run(model, HUMANOID_ROWS, {}, drive_mode="rows")
    first = None
    for step in range(TRACK_STEPS):
        undriven.step()
        if step == 0:
            first = undriven.state_0.joint_qd.numpy()
            test.assertFalse(np.any(_drive_rows(undriven.solver)))
    test.assertFalse(
        _within_track_tolerance(
            first,
            steel.first_qd,
            undriven.state_0.joint_q.numpy(),
            steel.state_0.joint_q.numpy(),
            rtol=DRIVE_TRACK_RTOL,
            atol=TRACK_ATOL,
        ),
        "the scene without drive rows tracks Steel's driven solve",
    )


def test_drive_descriptors_track_steel(test, device):
    """Give each drive row Steel's parameters bit for bit and Steel's force-drive descriptors to its rounding.

    Two lying humanoids, each hinge with its own gains, targets, target velocity and effort limit (a zero and an
    infinite one, which leave the impulse unbounded at 1e20, and a finite one), one hinge per world undriven (no row)
    and one each driven by its stiffness or its damping alone, step once from the same state and contacts. Over the
    drive rows, in Steel's order: the target velocities, stiffnesses, dampings, position errors ``target - q``, effort
    limits and maximum impulses equal Steel's dense rows' bit for bit and their float32 values. The descriptors are
    Steel's arithmetic bit for bit: Steel's drive descriptors (``compute_physx_pgs_drive_desc``) on the scene's rows (its one
    solve world) writes the scene's four descriptors, zeros on every other row. The velocity multipliers and the
    target-velocity biases, which barely move with the row's diagonal, equal Steel's within ``rtol=1e-5``; the diagonals
    (which carry no ``cfm``) and the impulse multipliers ``1 - x``, which follow them, within the factorizations'
    rounding (:data:`DIAGONAL_RTOL`). The drive rows count in Steel's articulation family per model world: the row
    high-water marks equal Steel's. Each articulation's internal unit holds its drive rows; its rows are its families'
    sum.
    """
    model = _humanoids(device, worlds=2)
    hinges = _hinge_dofs(model)
    index = np.arange(hinges.size) % HUMANOID_HINGES
    ke = (10.0 + index).astype(np.float32)
    kd = (0.5 + 0.1 * index).astype(np.float32)
    # Per world: the fourth hinge undriven, the fifth by damping alone, the sixth by stiffness alone.
    ke[index == 3], kd[index == 3] = 0.0, 0.0
    ke[index == 4] = 0.0
    kd[index == 5] = 0.0
    effort = np.full(hinges.size, 1.0e6, dtype=np.float32)
    effort[index == 7], effort[index == 8], effort[index == 9] = 0.0, np.inf, 2.0
    targets = (0.2 + 0.01 * index).astype(np.float32)
    _set_drives(model, hinges, ke=ke, kd=kd, target=targets, effort=effort)
    target_qd = model.joint_target_qd.numpy()
    target_qd[hinges] = (0.1 * (index % 3 - 1)).astype(np.float32)
    model.joint_target_qd.assign(target_qd)

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
            drive_mode="rows",
            cfm=1.0e-2,
            row_watermark=True,
        )
        test.assertIs(solver.solve_mode == "scene", name == "scene")
        solver.step(initial_state(), model.state(), model.control(), contacts, DT)
        test.assertFalse(np.any(solver.constraint_overflow.numpy()), f"{name} overflowed")
        solvers[name] = solver

    scene, steel = solvers["scene"], solvers["steel"]
    drive = _drive_rows(scene)
    rows = drive.size
    driven = (ke > 0.0) | (kd > 0.0)
    test.assertEqual(int(np.count_nonzero(drive)), 2 * (HUMANOID_HINGES - 1))

    def drive_values(array):
        return array.numpy()[0, :rows][drive]

    q = model.joint_q.numpy()[_hinge_coordinates(model, hinges)]
    max_impulse = np.where((effort > 0.0) & np.isfinite(effort), effort * np.float32(DT), np.float32(1.0e20))
    exact = {
        "_target_velocity": target_qd[hinges],
        "_drive_stiffness": ke,
        "_drive_damping": kd,
        "_drive_geom_error": targets - q,
        "_drive_max_force": effort,
        "_drive_max_impulse": max_impulse.astype(np.float32),
        "_row_mu": np.zeros(hinges.size, dtype=np.float32),
        "_phi": np.zeros(hinges.size, dtype=np.float32),
    }
    for name, expected in exact.items():
        with test.subTest(array=name):
            values = drive_values(getattr(scene, name))
            steel_values = _steel_drive_rows(steel, getattr(steel, name))
            test.assertTrue(np.array_equal(values, steel_values), f"{values} != {steel_values}")
            test.assertTrue(np.array_equal(values, expected[driven]), f"{values} != {expected[driven]}")
    tolerances = {
        "_drive_target_vel_bias": 1.0e-5,
        "_drive_vel_multiplier": 1.0e-5,
        "_drive_impulse_multiplier": DIAGONAL_RTOL,
        "_diag": DIAGONAL_RTOL,
    }
    for name, rtol in tolerances.items():
        with test.subTest(array=name):
            values = drive_values(getattr(scene, name))
            test.assertTrue(np.all(np.isfinite(values)))
            np.testing.assert_allclose(values, _steel_drive_rows(steel, getattr(steel, name)), rtol=rtol)
    descriptors = scene._scene.drive_descriptors
    test.assertEqual([array.shape for array in descriptors], [(1, scene._scene.row_max)] * 4)
    test.assertIs(descriptors[0], scene._drive_target_vel_bias)
    test.assertTrue(np.any(drive_values(scene._drive_target_vel_bias) != 0.0))
    # Steel's own descriptor kernel on the scene's rows reproduces the scene's descriptors bit for bit.
    steel_descriptors = [wp.full((1, scene._scene.row_max), -1.0, dtype=wp.float32, device=device) for _ in range(4)]
    wp.launch(
        compute_physx_pgs_drive_desc,
        dim=1,
        inputs=[
            scene._constraint_count,
            scene._row_type,
            scene._diag,
            scene._target_velocity,
            scene._drive_stiffness,
            scene._drive_damping,
            scene._drive_geom_error,
            scene._drive_max_force,
            DT,
        ],
        outputs=steel_descriptors,
        device=device,
    )
    for ours, steels in zip(descriptors, steel_descriptors, strict=True):
        a, b = ours.numpy()[0, :rows], steels.numpy()[0, :rows]
        test.assertTrue(np.array_equal(a, b), f"max |diff| {np.max(np.abs(a - b)):.3e}")
        test.assertTrue(np.all(a[~drive] == 0.0))

    marks = scene._row_watermarks.numpy()
    test.assertTrue(
        np.array_equal(marks, steel._row_watermarks.numpy()),
        f"{scene.constraint_row_watermarks()} != {steel.constraint_row_watermarks()}",
    )
    family_rows = scene._scene_family_rows.numpy()
    np.testing.assert_array_equal(family_rows[scene_rows.FAMILY_DRIVE], HUMANOID_HINGES - 1)
    np.testing.assert_array_equal(scene._scene_internal_rows.numpy(), family_rows.sum(axis=0))


def test_drive_rows_change_under_capture(test, device):
    """Replay one captured step exactly as eager steps while drive rows appear and disappear and targets move.

    Two standing humanoids fall with joint limits and drive rows. Between steps the hinges' gains are edited in
    place (every hinge, every other hinge, none, every hinge again) and their targets change sign, so drive rows
    disappear and come back, and their descriptors change, while one captured graph replays the step. ``joint_q``
    and ``joint_qd`` equal the eager steps' at every step, the drive family's rows follow the edits, the internal
    units hold the drives and the limits together at some step, and the eager row count takes at least three
    values.
    """
    model = _humanoids(device, worlds=2, lying=False)
    hinges = _hinge_dofs(model)
    schedule = (hinges, hinges[::2], hinges[:0], hinges)
    phase_steps = CAPTURE_STEPS // len(schedule)
    _set_drives(model, hinges)
    eager = _Stepper(model, drive_mode="rows")
    captured = _Stepper(model, drive_mode="rows")
    coordinates = _hinge_coordinates(model, hinges)

    def edit(phase):
        _set_drives(model, schedule[phase])
        sign = -1.0 if phase % 2 else 1.0
        for stepper in (eager, captured):
            targets = stepper.control.joint_target_q.numpy()
            targets[coordinates] = sign * DRIVE_TARGET
            stepper.control.joint_target_q.assign(targets)

    def assert_equal(step):
        for name in ("joint_q", "joint_qd"):
            a, b = getattr(eager.state_0, name).numpy(), getattr(captured.state_0, name).numpy()
            test.assertTrue(
                np.array_equal(a, b), f"{name} differs after step {step}: max |diff| {np.max(np.abs(a - b)):.3e}"
            )

    row_counts, drive_counts, shared = set(), [], False
    eager.step()
    captured.step()
    assert_equal(1)
    with wp.ScopedCapture(device) as recorded:
        captured.step()
    for index in range(1, CAPTURE_STEPS):
        if index % phase_steps == 0:
            edit(index // phase_steps)
        eager.step()
        wp.capture_launch(recorded.graph)
        assert_equal(index + 1)
        row_counts.add(eager.solver.scene_statistics()["rows"])
        family_rows = eager.solver._scene_family_rows.numpy()
        drive_counts.append(int(family_rows[scene_rows.FAMILY_DRIVE].sum()))
        np.testing.assert_array_equal(eager.solver._scene_internal_rows.numpy(), family_rows.sum(axis=0))
        both = (family_rows[scene_rows.FAMILY_DRIVE] > 0) & (family_rows[scene_rows.FAMILY_LIMIT] > 0)
        shared = shared or bool(np.any(both))
    expected = [schedule[index // phase_steps].size for index in range(1, CAPTURE_STEPS)]
    test.assertEqual(drive_counts, expected)
    test.assertTrue(shared, "no internal unit held drive and limit rows together")
    test.assertGreaterEqual(len(row_counts), 3, msg=f"row counts {sorted(row_counts)}")
    test.assertTrue(np.all(np.isfinite(captured.state_0.joint_q.numpy())))
    test.assertFalse(np.any(eager.solver.constraint_overflow.numpy()))


def test_drive_rows_match_the_serial_oracle(test, device):
    """Update the drive rows in the phased sweep exactly as one thread walking the rows does.

    Four lying humanoids 0.4 m apart overlap, so units join pairs of them; every hinge is driven to 0.2 rad, and
    the even hinges' effort limit of 0.5 N·m holds their rows on the maximum impulse. The phases (the default color
    loop, and the while loop, whose tail one thread walks) match the serial oracle (``scene_sweep="serial"``) bit
    for bit over 30 steps in each phase order, so the lane-group loop's force-drive update and clamp are the
    one-thread loop's. They do with drive rows alone (the ``_drv`` kernels) and with every implemented feature on
    (``_fpi_reg_jf_drv``: friction position iterations, ``contact_regularization=0.5``, joint friction and drive
    rows, whose featured arguments all carry data).
    """
    model = _humanoids(device, per_world=4, spacing=0.4)
    hinges = _hinge_dofs(model)
    effort = np.where(np.arange(hinges.size) % 2 == 0, CLAMP_EFFORT, 1.0e6).astype(np.float32)
    _set_drives(model, hinges, effort=effort)
    _set_friction(model, hinges, FRICTION)

    def clamped(solver):
        drive = _drive_rows(solver)
        rows = drive.size
        impulses = solver._impulses.numpy()[0, :rows][drive]
        bounds = solver._scene.drive_descriptors[3].numpy()[0, :rows][drive]
        return bool(np.any(np.abs(impulses) == bounds))

    feature_sets = {
        "_drv": {"drive_mode": "rows"},
        "_fpi_reg_jf_drv": {
            "drive_mode": "rows",
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
                test, model, options, witness=clamped, what="no drive row reached its maximum impulse"
            )


def test_drive_rows_with_wide_masks_are_bitwise_one_word(test, device):
    """Build drive rows and step bit for bit with support masks widened to two words (``scene_mask_words=2``).

    The test-only override gives the humanoids' factor group two-word masks, although they fit one, so the drive
    family's multi-word builder (``build_sparse_drive_rows_w2``) runs where the one-word builder is the reference: it
    visits the same coordinates in the same order, so two standing humanoids driven to 0.2 rad step to the same
    states, bit for bit, at each of 20 steps.
    """
    model = _humanoids(device, worlds=2, lying=False)
    _set_drives(model, _hinge_dofs(model))
    one = _Run(model, HUMANOID_ROWS, {}, drive_mode="rows")
    two = _Run(model, HUMANOID_ROWS, {"scene_mask_words": 2}, drive_mode="rows")
    for arrays in two.solver._scene.group_arrays:
        test.assertEqual(arrays.internal_kernels[scene_rows.FAMILY_DRIVE].key, "build_sparse_drive_rows_w2")
    for step in range(20):
        one.step()
        two.step()
        test.assertEqual(int(np.count_nonzero(_drive_rows(two.solver))), 2 * HUMANOID_HINGES)
        for name in ("joint_q", "joint_qd"):
            a, b = getattr(one.state_0, name).numpy(), getattr(two.state_0, name).numpy()
            test.assertTrue(np.array_equal(a, b), f"{name} differs after step {step + 1}")


def test_drive_rows_build_beside_velocity_limits(test, device):
    """Build joint velocity limits beside drive rows, fused or not, and store the velocity drive mode as Steel does.

    Joint velocity limits beside drive rows construct: with ``fuse_joint_velocity_limits`` (the default) the fused
    clamp is in effect (the ``_drv_late_fvc`` kernels), without it it is not (``_drv_late``). Without velocity limits
    the velocity drive mode reads back as stored and the fused clamp is inactive (the drive mode's effect, in the
    velocity-only iterations, is tested in ``test_scene_velocity_iterations.py``). The internal unit's drive rows read
    the step's control: counting them without it raises :class:`ValueError`.
    """
    model = _humanoids(device)
    for fused in (True, False):
        with test.subTest(fuse_joint_velocity_limits=fused):
            solver = SolverSteel(
                model,
                **{**SCENE, "drive_mode": "rows"},
                enable_joint_velocity_limits=True,
                fuse_joint_velocity_limits=fused,
            )
            test.assertEqual(solver.solve_mode, "scene")
            test.assertEqual(solver.velocity_limits_fused, fused)
            test.assertTrue(solver._scene.sides_kernel.key.endswith("_drv_late_fvc" if fused else "_drv_late"))
    for velocity_drive_mode in ("freeze", "active"):
        with test.subTest(velocity_drive_mode=velocity_drive_mode):
            solver = SolverSteel(model, **{**SCENE, "drive_mode": "rows", "velocity_drive_mode": velocity_drive_mode})
            test.assertEqual(solver.solve_mode, "scene")
            test.assertEqual(solver.drive_mode, "rows")
            test.assertEqual(solver.velocity_drive_mode, velocity_drive_mode)
            test.assertFalse(solver.velocity_limits_fused)
            test.assertTrue(solver.fuse_joint_velocity_limits)
    state = model.state()
    with test.assertRaisesRegex(ValueError, "drive rows read the step's control"):
        solver._scene.count_internal_rows(solver, state, DT)


class TestSteelSceneDrives(unittest.TestCase):
    pass


for _fn in (test_drive_rows_with_wide_masks_are_bitwise_one_word, test_drive_rows_build_beside_velocity_limits):
    add_function_test(TestSteelSceneDrives, _fn.__name__, _fn, devices=get_test_devices())

# CUDA only, each for the reason beside it. Steel builds drive rows on the CPU too, but the comparisons with Steel run
# on CUDA alone: test_scene_cpu steps the scene's drive rows on the CPU against CUDA's.
for _fn in (
    test_drive_rows_track_steel,  # compared with Steel on CUDA
    test_drive_descriptors_track_steel,  # compared with Steel on CUDA
    test_drive_rows_change_under_capture,  # graph capture
    test_drive_rows_match_the_serial_oracle,  # the phases; the CPU sweeps the serial oracle alone
):
    add_function_test(TestSteelSceneDrives, _fn.__name__, _fn, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
