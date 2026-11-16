# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Velocity-only iterations (``velocity_iterations``) in Steel's scene solve (``solve_mode="scene"``).

After the position solve Steel keeps its velocity, from which positions integrate, and launches the same solve again
on the same rows with the right-hand side of its velocity pass (``compute_world_contact_velocity_bias`` on its dense
rows, ``compute_mf_velocity_rhs`` on its free-body rows): no position bias, but a speculative contact keeps its
allowance ``phi / dt`` while its end gap after the position solve, ``phi + dt (J v_pos - target)``, stays above the
slop, an impacting contact keeps its rebound target and a joint limit its allowance. The pass continues the position
solve's impulses and velocity for ``velocity_iterations`` sweeps, weighs no row, solves friction from its first
sweep, and skips the drive rows under ``velocity_drive_mode="freeze"`` (the default) or solves them under
``"active"``; the step publishes the refined velocity.

The scene writes that right-hand side with ``prepare_scene_velocity_rows``, forming ``J v_pos`` in factor coordinates
as ``J v_hat + Z delta`` from the position solve's ``delta``, and runs ``SceneSolve.solve(rhs=..., reset=False)``,
which sweeps the same rows again from the position solve's ``delta`` and impulses and decodes again. Steel's sparse
path refuses velocity iterations, so the scene is compared with Steel's matrix-free solve within the tolerances of
``test_scene_features.py`` (each row's velocity-pass right-hand side within 1e-6), and with its serial oracle, its
eager steps and its own position solve bit for bit.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel.kernels import (
    PGS_CONSTRAINT_TYPE_CONNECT,
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
)
from newton._src.solvers.steel.scene_pgs import RANGE_ALL, get_sweep_rows_serial_kernel
from newton._src.solvers.steel.scene_rows import _scene_row_dot
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import DT, _build_contact_scene, _build_plane_model, _set_box_velocities
from newton.tests.steel._scene_fixtures import (
    REBOUND_APEX_ATOL,
    REBOUND_DROP,
    REBOUND_GRAVITY,
    REBOUND_ROWS,
    REBOUND_STEPS,
    SCENE,
    _humanoids,
    _Run,
    assert_rebound_tracks_steel,
)
from newton.tests.steel.test_scene_drives import _drive_rows, _set_drives
from newton.tests.steel.test_scene_features import (
    CROWD_ROWS,
    HUMANOID_ROWS,
    REGULARIZATION,
    TRACK_ROWS,
    TRACK_STEPS,
    _assert_phases_match_serial_oracle,
    _set_drop_velocities,
    _track_steel,
    _within_track_tolerance,
)
from newton.tests.steel.test_scene_internal_unit import HUMANOID_HINGES, _hinge_dofs, _set_friction
from newton.tests.steel.test_scene_velocity_limits import CROWD_LIMIT, _limit_hinges
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

needs_scene_step = unittest.skipUnless(hasattr(SolverSteel, "_seed_contact_impulses"), "needs the scene step")
needs_read_back = unittest.skipUnless(hasattr(SolverSteel, "scene_statistics"), "needs the read-back API")

VELOCITY_ITERATIONS = 2
"""``velocity_iterations`` of the tests."""

VELOCITY_RHS_ATOL = 1.0e-6
"""Absolute tolerance of a row's velocity-pass right-hand side against Steel's: a free-body row's speculative allowance
``phi / dt``, which Steel's ``compute_mf_velocity_rhs`` divides and the scene multiplies by ``1 / dt``, 1 ulp apart. A
rebound target (``-target - rebound``, from the rebound velocity both solvers' row builders write) is Steel's bit for
bit, as is a dense row's allowance."""

FREEZE_ITERATIONS = 4
FREEZE_STEPS = 10
"""``velocity_iterations`` and steps of the drive-mode test: enough for the contacts to load and an active drive row to
move."""

SPLIT = (3, 2)
"""Position and velocity iterations of the continuation test, which equal one solve of their sum."""

CAPTURE_STEPS = 60
"""Steps of the capture test."""

JOINT_FRICTION = 0.5
"""``joint_friction`` [N·m] of every hinge in the oracle test's featured set."""


def _contact_rows(test, solver, count) -> dict:
    """Every row of each of the first ``count`` contacts with rows, its normal row first, in contact order.

    Returns, per row: the contact, the row's place in its contact (0 for the normal row), Steel's row family (0 the
    scene's rows or Steel's dense rows, 1 Steel's free-body rows), ``phi``, the position solve's ``rhs`` and the
    velocity pass's (``_rhs_unbiased`` or ``_mf_rhs_unbiased``). The rows' types are asserted.
    """
    path = solver._contact_path.numpy()[:count]
    slot = solver._contact_slot.numpy()[:count]
    world = solver._contact_world.numpy()[:count]
    needed = solver._contact_slots_needed.numpy()[:count]
    families = {
        0: [a.numpy() for a in (solver._row_type, solver._phi, solver._rhs, solver._rhs_unbiased)],
        1: [a.numpy() for a in (solver._mf_row_type, solver._mf_phi, solver._mf_rhs, solver._mf_rhs_unbiased)],
    }
    rows = {name: [] for name in ("contact", "place", "family", "phi", "rhs", "rhs_unbiased")}
    for c in np.flatnonzero(slot >= 0):
        row_type, phi, rhs, rhs_unbiased = families[int(path[c])]
        for place in range(int(needed[c])):
            index = (world[c], slot[c] + place)
            expected = PGS_CONSTRAINT_TYPE_CONTACT if place == 0 else PGS_CONSTRAINT_TYPE_FRICTION
            test.assertEqual(row_type[index], expected)
            for name, value in zip(rows, (c, place, path[c], phi[index], rhs[index], rhs_unbiased[index]), strict=True):
                rows[name].append(value)
    return {name: np.asarray(values) for name, values in rows.items()}


@needs_scene_step
def test_velocity_iterations_track_steel(test, device):
    """Refine the position solve's velocity with two velocity-only iterations, as Steel's matrix-free solve does.

    On the contact scene with the boxes sliding at 1.5 m/s (their contacts 1 mm deep, so the position solve pushes
    them out and the velocity pass drops that push, on Steel's dense rows and on its free-body rows alike), the
    velocities after one step agree with Steel's within 1e-5 (and 1e-6) and the joint coordinates after 30 steps
    within 1e-3 RMS, with the default 4 position iterations and with none (the velocity pass alone solves the rows).
    The scene's ``rhs_unbiased`` is ``[1, row_max]``. The same scene without velocity iterations falls outside that
    tolerance of Steel's solve with them, so the test would see a pass that changes nothing.
    """
    model = _build_contact_scene(device)
    for iterations in (SCENE["iterations"], 0):
        with test.subTest(iterations=iterations):
            scene, steel = _track_steel(test, model, iterations=iterations, velocity_iterations=VELOCITY_ITERATIONS)
            solve = scene.solver._scene
            test.assertEqual(solve.velocity_iterations, VELOCITY_ITERATIONS)
            test.assertEqual(scene.solver._rhs_unbiased.shape, (1, solve.row_max))
            position_only = _Run(model, TRACK_ROWS, {}, iterations=iterations)
            _set_box_velocities(model, position_only.state_0)
            first = None
            for step in range(TRACK_STEPS):
                position_only.step()
                if step == 0:
                    first = position_only.state_0.joint_qd.numpy()
            test.assertFalse(
                _within_track_tolerance(
                    first, steel.first_qd, position_only.state_0.joint_q.numpy(), steel.state_0.joint_q.numpy()
                ),
                "the scene without velocity iterations tracks Steel's velocity pass",
            )


@needs_scene_step
def test_velocity_rows_rhs_is_unbiased(test, device):
    """Give each contact row the velocity-pass right-hand side Steel's rows get, on the same contacts.

    On the contact scene with restitution 0.4, some bodies are pushed into the ground at 1 m/s and the others rest
    1 mm deep; the arm's contacts are speculative, 16 cm and more above the ground. After one step of each solver
    with two velocity iterations, every row of every contact (its normal row and its friction rows) has the
    velocity-pass right-hand side of Steel's row for that contact (its dense ``_rhs_unbiased`` or its free-body
    ``_mf_rhs_unbiased``) within 1e-6, ``phi`` bit for bit, and the rows whose right-hand side is zero are Steel's.
    The law shows on both of Steel's row families: an impacting contact keeps its rebound target (the position
    solve's right-hand side and Steel's, bit for bit), a resting contact loses its Baumgarte bias (zero where the position solve's
    was negative), and on Steel's dense rows a speculative contact that stays open keeps its allowance ``phi / dt``.
    The scene's rows past the kept rows are zero.
    """
    model = _build_contact_scene(device, restitution=0.4)

    def initial_state():
        state = model.state()
        _set_drop_velocities(model, state)
        return state

    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    pipeline.collide(initial_state(), contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)

    rows = {}
    for name, options in (("scene", {}), ("steel", {"solve_mode": "serial"})):
        solver = SolverSteel(
            model,
            **{**SCENE, **options},
            articulation_constraint_max=TRACK_ROWS,
            velocity_iterations=VELOCITY_ITERATIONS,
        )
        test.assertIs(solver.solve_mode == "scene", name == "scene")
        solver.step(initial_state(), model.state(), model.control(), contacts, DT)
        test.assertFalse(np.any(solver.constraint_overflow.numpy()), f"{name} overflowed")
        rows[name] = _contact_rows(test, solver, count)
        if name == "scene":
            kept = int(solver._constraint_count.numpy()[0])
            test.assertGreater(kept, 0)
            np.testing.assert_array_equal(solver._rhs_unbiased.numpy()[0, kept:], 0.0)

    scene, steel = rows["scene"], rows["steel"]
    for name in ("contact", "place"):
        np.testing.assert_array_equal(scene[name], steel[name], err_msg=name)
    np.testing.assert_array_equal(scene["family"], 0)
    np.testing.assert_array_equal(scene["phi"], steel["phi"])
    np.testing.assert_allclose(scene["rhs_unbiased"], steel["rhs_unbiased"], rtol=0.0, atol=VELOCITY_RHS_ATOL)
    np.testing.assert_array_equal(scene["rhs_unbiased"] == 0.0, steel["rhs_unbiased"] == 0.0)

    normal = steel["place"] == 0
    # A rebound target, kept by the velocity pass: the position solve's right-hand side, -rebound < 0 (the ground is
    # static, so no target), the only negative one.
    rebound = normal & (steel["rhs_unbiased"] < 0.0)
    # A Baumgarte bias, dropped: the position solve's right-hand side was negative, the velocity pass's is zero.
    dropped = normal & (steel["phi"] < 0.0) & (steel["rhs"] < 0.0) & (steel["rhs_unbiased"] == 0.0)
    # A speculative allowance, kept.
    speculative = normal & (steel["phi"] > 0.0) & (steel["rhs_unbiased"] > 0.0)
    for family in (0, 1):
        in_family = steel["family"] == family
        test.assertTrue(np.any(rebound & in_family), f"no rebound target on Steel's family {family}")
        test.assertTrue(np.any(dropped & in_family), f"no Baumgarte bias dropped on Steel's family {family}")
    test.assertTrue(np.any(speculative & (steel["family"] == 0)), "no speculative allowance kept")
    np.testing.assert_array_equal(scene["rhs_unbiased"][rebound], scene["rhs"][rebound])
    np.testing.assert_array_equal(steel["rhs_unbiased"][rebound], steel["rhs"][rebound])
    # Both solvers take the rebound target from the same rebound velocity: bit for bit.
    np.testing.assert_array_equal(scene["rhs_unbiased"][rebound], steel["rhs_unbiased"][rebound])
    np.testing.assert_array_equal(scene["rhs_unbiased"][dropped], 0.0)
    test.assertTrue(np.all(scene["rhs"][dropped] < 0.0))


@needs_scene_step
def test_velocity_iterations_rebound_tracks_steel(test, device):
    """Land and rebound a falling sphere through the velocity pass as Steel does, as a free body and articulated.

    The sphere of ``_build_plane_model`` falls 5 cm under gravity onto the plane, restitution 0.5, its contact
    speculative within 10 cm: before the impact the velocity pass keeps the allowance of a contact that stays open and
    drops it where the position solve closed the gap, at the impact it keeps the rebound target. With four velocity
    iterations the vertical velocity agrees with Steel's matrix-free solve (its free-body rows for the free sphere, its
    dense rows for the articulated one) within 1e-4 of Steel's at each of 60 steps, plus 1e-6 m/s
    (``REBOUND_APEX_ATOL``) where it crosses zero, and the sphere rebounds. The free sphere is a scene of free bodies
    only. Its velocity pass sums the row's ``J v`` from the scene's incident velocity and ``delta``, where Steel's
    free-body rows dot their Jacobian with the velocity: the two round 1 ulp apart (6e-8 m/s, measured at an apex,
    where Steel's velocity is 4e-8 m/s), so the free sphere, bitwise Steel's without velocity iterations, takes the
    articulated sphere's floor.
    """
    for scene in ("free", "articulated"):
        with test.subTest(scene=scene):
            model, body = _build_plane_model(
                device, separation=REBOUND_DROP, restitution=0.5, gravity=REBOUND_GRAVITY, scene=scene
            )
            runs = tuple(
                _Run(model, REBOUND_ROWS, {}, velocity_iterations=4, **options)
                for options in ({}, {"solve_mode": "serial"})
            )
            velocities = np.zeros((len(runs), REBOUND_STEPS), dtype=np.float32)
            for step in range(REBOUND_STEPS):
                for k, run in enumerate(runs):
                    run.step()
                    velocities[k, step] = run.state_0.body_qd.numpy()[body, 2]
            if scene == "free":
                test.assertEqual(runs[0].solver._scene.group_arrays, ())
            assert_rebound_tracks_steel(test, runs, velocities, scene, atol=REBOUND_APEX_ATOL)


@needs_scene_step
def test_velocity_iterations_freeze_or_solve_drive_rows(test, device):
    """Leave the drive rows as the position solve left them under ``"freeze"``, and solve them under ``"active"``.

    Two lying humanoids drive every hinge to 0.2 rad with drive rows and settle on the ground. With four velocity
    iterations both velocity drive modes construct (the ``_drv`` kernels) and step 10 times. Under ``"freeze"`` (the
    default) the drive rows' impulses after the velocity pass are the position solve's, bit for bit, at every step;
    under ``"active"`` they move at some step. In both modes the velocity pass moves some contact's impulse.
    """
    model = _humanoids(device, worlds=2)
    _set_drives(model, _hinge_dofs(model))
    for mode in ("freeze", "active"):
        with test.subTest(velocity_drive_mode=mode):
            run = _Run(
                model,
                HUMANOID_ROWS,
                {},
                drive_mode="rows",
                velocity_iterations=FREEZE_ITERATIONS,
                velocity_drive_mode=mode,
            )
            test.assertEqual(run.solver.solve_mode, "scene")
            test.assertEqual(run.solver.velocity_drive_mode, mode)
            test.assertTrue(run.solver._scene.sides_kernel.key.endswith("_drv"))
            records = _record_solves(run.solver)
            drive_moved = contact_moved = False
            for step in range(FREEZE_STEPS):
                run.step()
                (*_, position), (*_, velocity) = records[-2:]
                drive = _drive_rows(run.solver)
                test.assertEqual(int(np.count_nonzero(drive)), 2 * HUMANOID_HINGES)
                contact = run.solver._row_type.numpy()[0, : drive.size] == PGS_CONSTRAINT_TYPE_CONTACT
                contact_moved = contact_moved or not np.array_equal(velocity[contact], position[contact])
                moved = not np.array_equal(velocity[drive], position[drive])
                if mode == "freeze":
                    test.assertFalse(moved, f"a frozen drive row moved at step {step + 1}")
                drive_moved = drive_moved or moved
            test.assertEqual(len(records), 2 * FREEZE_STEPS)
            test.assertTrue(contact_moved, "the velocity pass moved no contact")
            if mode == "active":
                test.assertTrue(drive_moved, "no active drive row moved")


def _record_solves(solver, *, velocity_rhs=None) -> list:
    """Record ``v_out``, ``delta`` and the kept rows' impulses after each of the scene's solves of a step.

    Wraps ``solver``'s ``SceneSolve.solve``; ``velocity_rhs(solver)``, when given, replaces the right-hand side of the
    velocity pass (the call with ``reset=False``). Returns the list the records are appended to.
    """
    scene = solver._scene
    solve = scene.solve
    records = []

    def recording_solve(steel, *, rhs=None, reset=True, **kwargs):
        if not reset and velocity_rhs is not None:
            rhs = velocity_rhs(steel)
        solve(steel, rhs=rhs, reset=reset, **kwargs)
        rows = int(steel._constraint_count.numpy()[0])
        # Copies: on the CPU numpy() is a view of the solver's arrays, which the next solve overwrites.
        records.append(
            tuple(
                array.copy()
                for array in (
                    steel._v_out.numpy(),
                    steel._sparse_factor_velocity_delta.numpy(),
                    steel._impulses.numpy()[0, :rows],
                )
            )
        )

    scene.solve = recording_solve
    return records


@needs_scene_step
@needs_read_back
def test_velocity_pass_continues_the_position_solve(test, device):
    """Continue the position solve in the second solve: 3 sweeps, then 2 with ``reset=False``, are 5 sweeps.

    Four lying humanoids 0.4 m apart overlap, so units join pairs of them and phase B runs. A solve of 3 position
    iterations followed by ``solve(rhs=<its biased rhs>, reset=False)`` with 2 velocity iterations leaves ``v_out``,
    ``delta`` and the impulses bit for bit as one solve of 5 iterations: the second call keeps the position solve's
    ``delta``, impulses, ``v_out``, row ranges and colors and sweeps the count it is given, its isolated sides with
    their own stationary exit (a sweep that changes nothing is a bitwise fixed point, so the exit cannot tell the two
    calls from one). So it does with the default kernels and with the featured ones of drive rows solved in the
    velocity pass (``velocity_drive_mode="active"``) beside joint velocity limits under the fused clamp
    (``_drv_late_fvc``): phase C and the clamp run in the second call as in the first.
    """
    model = _humanoids(device, per_world=4, spacing=0.4)
    hinges = _limit_hinges(model, CROWD_LIMIT)
    feature_sets = {
        "": {},
        "_drv_late_fvc": {"drive_mode": "rows", "velocity_drive_mode": "active", "enable_joint_velocity_limits": True},
    }
    position_iterations, velocity_iterations = SPLIT
    for suffix, options in feature_sets.items():
        with test.subTest(features=suffix or "none"):
            if suffix:
                _set_drives(model, hinges[::2])
            split = _Run(
                model,
                CROWD_ROWS,
                {},
                iterations=position_iterations,
                velocity_iterations=velocity_iterations,
                **options,
            )
            whole = _Run(model, CROWD_ROWS, {}, iterations=position_iterations + velocity_iterations, **options)
            if suffix:
                test.assertTrue(split.solver._scene.sides_kernel.key.endswith(suffix))
            pieces = _record_solves(split.solver, velocity_rhs=lambda steel: steel._rhs)
            one = _record_solves(whole.solver)
            split.step()
            whole.step()
            test.assertEqual((len(pieces), len(one)), (2, 1))
            test.assertGreater(split.solver.scene_statistics()["two_sided_units"], 0)
            for name, a, b in zip(("v_out", "delta", "impulses"), pieces[1], one[0], strict=True):
                test.assertTrue(np.array_equal(a, b), f"{name}: max |diff| {np.max(np.abs(a - b)):.3e}")
            test.assertFalse(np.array_equal(pieces[0][2], one[0][2]), "the second solve swept nothing")


def _speculative_rhs_dropped(solver) -> bool:
    """Whether a kept speculative contact row (``phi > 0``) lost its allowance in the velocity pass: its velocity-pass
    right-hand side differs from the position solve's, which ``J v_pos`` (``J v_hat + Z delta``) decided."""
    rows = int(solver._constraint_count.numpy()[0])
    speculative = (solver._row_type.numpy()[0, :rows] == PGS_CONSTRAINT_TYPE_CONTACT) & (
        solver._phi.numpy()[0, :rows] > 0.0
    )
    rhs = solver._rhs.numpy()[0, :rows]
    rhs_unbiased = solver._rhs_unbiased.numpy()[0, :rows]
    return bool(np.any(rhs_unbiased[speculative] != rhs[speculative]))


@needs_scene_step
def test_velocity_iterations_match_the_serial_oracle(test, device):
    """Run both passes of the step in the phased sweep exactly as one thread walking the rows does.

    Four lying humanoids 0.4 m apart overlap, so units join pairs of them. With two velocity iterations the phases (the
    default color loop, and the while loop, whose tail one thread walks) match the serial oracle
    (``scene_sweep="serial"``) bit for bit over 30 steps in each phase order, ``joint_q``, ``joint_qd`` and the
    impulses after the velocity pass: the velocity pass's right-hand side, written from the position solve's
    ``delta``, and its sweeps are the oracle's. Some speculative contact loses its allowance in the velocity pass, so
    its right-hand side differs from the position solve's. They match with the default kernels and with the featured
    ones of drive rows frozen in the velocity pass, joint friction, ``contact_regularization=0.5`` (weight 1 in the
    velocity pass) and friction in the last 2 of 6 position iterations (from the velocity pass's first sweep).
    """
    model = _humanoids(device, per_world=4, spacing=0.4)
    hinges = _hinge_dofs(model)
    feature_sets = {
        "": {"velocity_iterations": VELOCITY_ITERATIONS},
        "_fpi_reg_jf_drv": {
            "velocity_iterations": VELOCITY_ITERATIONS,
            "drive_mode": "rows",
            "enable_joint_friction": True,
            "contact_regularization": REGULARIZATION,
            "iterations": 6,
            "contact_friction_position_iterations": 2,
        },
    }
    for suffix, options in feature_sets.items():
        with test.subTest(features=suffix or "none"):
            if suffix:
                _set_drives(model, hinges)
                _set_friction(model, hinges, JOINT_FRICTION)
                probe = _Run(model, CROWD_ROWS, {}, collide=False, **options)
                test.assertTrue(probe.solver._scene.sides_kernel.key.endswith(suffix))
            _assert_phases_match_serial_oracle(
                test,
                model,
                options,
                witness=_speculative_rhs_dropped,
                what="no speculative contact lost its allowance in the velocity pass",
            )


class _FixedRun(_Run):
    """A :class:`_Run` stepping fixed state buffers, so that one captured step replays every later step."""

    def step(self):
        self.pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, DT)
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            wp.copy(getattr(self.state_0, name), getattr(self.state_1, name))


@needs_scene_step
@needs_read_back
def test_velocity_iterations_replay_under_capture(test, device):
    """Replay one captured step, its velocity pass included, exactly as eager steps.

    Four lying humanoids 0.4 m apart overlap, so units join pairs of them and the velocity pass runs its own
    conditional nodes (the coupled sides' branch and phase B's color loop) after the position solve's, with
    ``contact_regularization=0.5``, so one graph holds the position solve's weights and the velocity pass's ones. With
    two velocity iterations ``joint_q``, ``joint_qd``, ``body_q`` and ``body_qd`` equal the eager steps' after each
    of 60 steps, and the velocity pass changes some hinge's velocity (the published velocity, against the position
    solve's).
    """
    model = _humanoids(device, per_world=4, spacing=0.4)
    hinges = _hinge_dofs(model)
    options = {"velocity_iterations": VELOCITY_ITERATIONS, "contact_regularization": REGULARIZATION}
    eager = _FixedRun(model, CROWD_ROWS, {}, **options)
    captured = _FixedRun(model, CROWD_ROWS, {}, **options)

    def assert_equal(step):
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            a, b = getattr(eager.state_0, name).numpy(), getattr(captured.state_0, name).numpy()
            test.assertTrue(np.array_equal(a, b), f"{name} differs after step {step}")

    eager.step()
    captured.step()
    assert_equal(1)
    with wp.ScopedCapture(device) as recorded:
        captured.step()
    refined = two_sided = 0
    for index in range(1, CAPTURE_STEPS):
        eager.step()
        wp.capture_launch(recorded.graph)
        assert_equal(index + 1)
        solver = eager.solver
        two_sided = max(two_sided, solver.scene_statistics()["two_sided_units"])
        refined += int(np.any(solver._v_out.numpy()[hinges] != solver._v_out_snap.numpy()[hinges]))
    test.assertGreater(two_sided, 0)
    test.assertGreater(refined, 0, "the velocity pass changed no hinge's velocity")
    test.assertTrue(np.all(np.isfinite(captured.state_0.joint_q.numpy())))
    test.assertFalse(np.any(eager.solver.constraint_overflow.numpy()))


@needs_scene_step
def test_regularized_velocity_pass_tracks_steel(test, device):
    """Weigh no row in the velocity pass, as Steel's, beside the regularized position solve.

    On the contact scene with the boxes sliding at 1.5 m/s and ``contact_regularization=0.5``, the boxes' contacts,
    1 mm deep, weigh ``1 / 1.5`` in the position solve; the velocity pass's regularized sweeps read a ``[1, row_max]``
    array of ones in place of ``row_w``. With two velocity iterations the velocities after one step agree with
    Steel's matrix-free solve within 1e-5 (and 1e-6), and the joint coordinates after 30 steps within 1e-3 RMS. The
    same scene whose velocity pass weighs its rows with the position solve's ``row_w`` falls outside that tolerance
    of Steel's solve, so the test would see a weight that changes nothing.
    """
    model = _build_contact_scene(device)
    options = {"contact_regularization": REGULARIZATION, "velocity_iterations": VELOCITY_ITERATIONS}
    _scene, steel = _track_steel(test, model, **options)
    solve = _scene.solver._scene
    test.assertTrue(solve.features.regularize)
    ones = solve._velocity_row_w_arg
    test.assertIsNot(ones, solve.row_w)
    test.assertEqual(ones.shape, (1, solve.row_max))
    np.testing.assert_array_equal(ones.numpy(), 1.0)
    weighed = _Run(model, TRACK_ROWS, {}, **options)
    weighed.solver._scene._velocity_row_w_arg = weighed.solver._scene.row_w
    _set_box_velocities(model, weighed.state_0)
    first = None
    for step in range(TRACK_STEPS):
        weighed.step()
        if step == 0:
            first = weighed.state_0.joint_qd.numpy()
    test.assertFalse(
        _within_track_tolerance(first, steel.first_qd, weighed.state_0.joint_q.numpy(), steel.state_0.joint_q.numpy()),
        "a velocity pass weighing its rows tracks Steel's unweighed one",
    )


@wp.kernel
def _row_dots(
    row_dof: wp.array3d[int],
    row_factor: wp.array3d[float],
    delta: wp.array2d[float],
    steel_sums: int,
    dots: wp.array[float],
):
    i = wp.tid()
    dots[i] = _scene_row_dot(row_dof, row_factor, delta, i, steel_sums)


@needs_scene_step
def test_scene_row_dot_mirrors_the_thread_loop(test, device):
    """Sum each row's ``Z delta`` in ``_scene_row_dot`` as the one-thread row loop sums its ``dot``, bit for bit.

    ``prepare_scene_velocity_rows`` forms a row's ``J v`` after the position solve with ``_scene_row_dot``, the Python
    mirror of the native loop's dot (``SCENE_ROWS_CUDA``). On the rows of two lying humanoids beside a free box after
    one step (factor rows and rows that start with free-body entries, 30 entries wide), the serial walk visits each row
    alone as a bilateral row of right-hand side 0 and diagonal 1 from the step's ``delta``, so its impulse is
    ``-dot`` exactly; for Steel's sums and for serial sums, every row's mirror equals that ``dot`` bit for bit.
    """
    model = _humanoids(device, per_world=2, spacing=0.4, props=True)
    run = _Run(model, CROWD_ROWS, {})
    run.step()
    solver = run.solver
    rows = int(solver._constraint_count.numpy()[0])
    test.assertGreater(rows, 0)
    support = solver._scene_plan.support
    test.assertGreater(support, 16)
    prefix = solver._scene.row_free_prefix.numpy()[:rows]
    test.assertTrue(np.any(prefix > 0) and np.any(prefix == 0))
    delta = solver._sparse_factor_velocity_delta
    test.assertTrue(np.any(delta.numpy() != 0.0))
    shape = solver._rhs.shape
    zeros = wp.zeros(shape, dtype=wp.float32, device=device)
    ones = wp.ones(shape, dtype=wp.float32, device=device)
    bilateral = wp.full(shape, PGS_CONSTRAINT_TYPE_CONNECT, dtype=wp.int32, device=device)
    parents = wp.full(shape, -1, dtype=wp.int32, device=device)
    for sums in ("steel", "serial"):
        with test.subTest(sums=sums):
            mirror = wp.zeros(rows, dtype=wp.float32, device=device)
            wp.launch(
                _row_dots,
                dim=rows,
                inputs=[solver._sparse_row_dof, solver._sparse_row_factor, delta, int(sums == "steel")],
                outputs=[mirror],
                device=device,
            )
            kernel = get_sweep_rows_serial_kernel(sums, support, free_response=True)
            loop = np.empty(rows, dtype=np.float32)
            for row in range(rows):
                ranges = np.zeros((4, 2), dtype=np.int32)
                ranges[RANGE_ALL] = (row, row + 1)
                impulses = wp.zeros(shape, dtype=wp.float32, device=device)
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[
                        RANGE_ALL,
                        1,
                        0,
                        wp.array(ranges, dtype=wp.int32, device=device),
                        solver._sparse_row_dof,
                        solver._sparse_row_factor,
                        solver._sparse_row_free_response,
                        solver._scene.row_free_prefix,
                        zeros,
                        zeros,
                        ones,
                        bilateral,
                        parents,
                        zeros,
                        1.0,
                        wp.clone(delta),
                        impulses,
                    ],
                    device=device,
                )
                loop[row] = -impulses.numpy()[0, row]
            test.assertTrue(np.any(loop != 0.0))
            test.assertTrue(np.array_equal(mirror.numpy(), loop), f"max |diff| {np.max(np.abs(mirror.numpy() - loop))}")


class TestSteelSceneVelocityIterations(unittest.TestCase):
    pass


# On every device: the velocity pass against the scene's own position solve and the loop's Python mirror.
for _fn in (
    test_velocity_iterations_freeze_or_solve_drive_rows,
    test_velocity_pass_continues_the_position_solve,
    test_scene_row_dot_mirrors_the_thread_loop,
):
    add_function_test(TestSteelSceneVelocityIterations, _fn.__name__, _fn, devices=get_test_devices())


# CUDA only, each for the reason beside it: the references are Steel's CUDA velocity pass; test_scene_cpu steps the
# scene's velocity pass on the CPU against CUDA's.
for _fn in (
    test_velocity_iterations_track_steel,  # Steel's velocity pass
    test_velocity_rows_rhs_is_unbiased,  # Steel's velocity pass
    test_velocity_iterations_rebound_tracks_steel,  # Steel's velocity pass
    test_velocity_iterations_match_the_serial_oracle,  # the phases; the CPU sweeps the serial oracle alone
    test_velocity_iterations_replay_under_capture,  # graph capture
    test_regularized_velocity_pass_tracks_steel,  # Steel's velocity pass
):
    add_function_test(TestSteelSceneVelocityIterations, _fn.__name__, _fn, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
