# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The scene solve of SolverSteel (``solve_mode="scene"``) on the CPU against the same scene on CUDA, feature by
feature.

On the CPU the scene sweeps serially (the phases' oracle) through the host shim, and every launch of its row build,
its row preparation and its decode runs as a host kernel; on CUDA the phases run. Each feature's CPU run is checked
against its CUDA run, with the bounds within which the CUDA scene tracks Steel's solve in the feature's own test: the
first step on the same contacts (CUDA's, copied to the CPU), the coordinates after 30 steps each on its own device's
contacts. Every case asserts that the feature acts (its witness, or its rows), that the CPU sweeps serially and CUDA in
phases, with the same build constants, and that no budget overflows. The features Steel refuses on the CPU (contact
compliance, joint friction, the friction modes other than ``"fixed_normal"``), which the scene refuses there too because
Steel's checks run first, are not stepped on the CPU.
"""

import unittest
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel.scene_rows import FAMILY_CONNECT, FAMILY_MIMIC
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import (
    DT,
    _build_carried_load,
    _build_contact_scene,
    _build_four_bar,
    _build_plane_model,
    _build_two_revolute_chain,
    _heap,
    _set_box_velocities,
)
from newton.tests.steel._scene_fixtures import (
    REBOUND_DROP,
    REBOUND_GRAVITY,
    REBOUND_ROWS,
    REBOUND_RTOL,
    SCENE,
    _chain_worlds,
    _humanoids,
    _rms,
    _Run,
    _scene_heap,
    _scene_worlds,
)
from newton.tests.steel.test_scene_aliases import BILATERAL, _assert_tracks_steel, _bilateral_run
from newton.tests.steel.test_scene_depenetration import CAP_ROWS, DEPENETRATION_CAP, _capped_boxes
from newton.tests.steel.test_scene_drives import DRIVE_TRACK_RTOL, _drive_rows, _set_drives
from newton.tests.steel.test_scene_features import (
    CROWD_ROWS,
    FRICTION_ITERATIONS,
    FRICTION_POSITION_ITERATIONS,
    HUMANOID_ROWS,
    ONE_ROW_GAP,
    REGULARIZATION,
    TRACK_ATOL,
    TRACK_RMS,
    TRACK_ROWS,
    TRACK_RTOL,
    TRACK_STEPS,
)
from newton.tests.steel.test_scene_friction_modes import (
    FREE_MODE_ATOL,
    LAYER_PITCH,
    MODE_TRACK_Q_RMS,
    _mixed_scene,
    _slide_and_drop,
)
from newton.tests.steel.test_scene_internal_unit import _at_rest, _hinge_dofs
from newton.tests.steel.test_scene_state import (
    HEAP_FREE_ROWS,
    HEAP_ITERATIONS,
    PATCHES,
    REST_ATOL,
    SLEEP_STEPS,
    SLIDE_SPEED,
    _asleep,
    _islands,
    _patch_state,
    _SeedRun,
    _sleep_run,
)
from newton.tests.steel.test_scene_velocity_iterations import VELOCITY_ITERATIONS
from newton.tests.steel.test_scene_velocity_limits import (
    FREE_PITCH,
    FREE_ROWS,
    VELOCITY_LIMIT,
    _free_body_dofs,
    _limit_free_bodies,
    _limit_hinges,
    _spin_hinges,
    _throw_free_bodies,
    _velocity_limit_applied,
)

CUDA = "cuda:0"


@dataclass
class _Case:
    """One feature's scene on both devices: ``build(device)`` returns the model, which a ``runner`` steps from
    ``init``'s state with ``rows`` rows per world and ``options``. The velocities after the first step (on the same
    contacts) agree within ``rtol`` and ``atol`` (``rtol=None``: a scene whose tests bound its coordinates alone), the
    free bodies' within ``free`` (``(rtol, atol)``) where the feature's test bounds them apart, and ``same(run)``'s
    arrays are equal on both devices after the first step; the joint coordinates after :data:`TRACK_STEPS` steps agree
    within ``rms`` RMS (the feature's bounds against Steel), and ``witness(run)`` holds after some step on each
    device."""

    build: Callable
    options: dict = field(default_factory=dict)
    runner: type = _Run
    rows: int = TRACK_ROWS
    init: Callable | None = _set_box_velocities
    rtol: float | None = TRACK_RTOL
    atol: float = TRACK_ATOL
    free: tuple[float, float] | None = None
    rms: float = TRACK_RMS
    witness: Callable | None = None
    same: Callable | None = None


def _two_sided(run) -> bool:
    return run.solver._scene_units.statistics()["two_sided"] > 0


def _rebounds(run) -> bool:
    """Whether the dropped sphere moves up after the last step: its contact fired with restitution."""
    return bool(np.any(run.state_0.body_qd.numpy()[:, 2] > 0.0))


def _seeded(run) -> bool:
    """Whether the warm start seeded some impulse of the last step (``_SeedRun`` keeps the seeds)."""
    return bool(np.any(run.solver.seeds[0] != 0.0))


def _limited_velocity(run) -> bool:
    return _velocity_limit_applied(run.solver)


def _drives(run) -> bool:
    return bool(np.any(_drive_rows(run.solver)))


def _capped(run) -> bool:
    """Whether some row's right-hand side of the last step is the depenetration cap's."""
    rows = int(run.solver._constraint_count.numpy()[0])
    return bool(np.any(run.solver._rhs.numpy()[0, :rows] == np.float32(-DEPENETRATION_CAP)))


def _unbiased(run) -> bool:
    """Whether the velocity pass's right-hand side of the last step differs from the position solve's on some row."""
    rows = int(run.solver._constraint_count.numpy()[0])
    return bool(np.any(run.solver._rhs_unbiased.numpy()[0, :rows] != run.solver._rhs.numpy()[0, :rows]))


def _patch_regions(run) -> tuple:
    """The step's contacts' patch anchors (``FrictionPatches.weight > 0``) and row counts (a region's member past its
    anchors has its normal row alone)."""
    count = int(run.contacts.rigid_contact_count.numpy()[0])
    anchors, _history = _patch_state(run.solver, count)
    return anchors, run.solver._contact_slots_needed.numpy()[:count]


def _flooded(run) -> bool:
    """Whether the patch flood grew a region past its anchors in the last step: some anchor, and some member with one
    row (on the CPU Steel's serial flood, ``friction_patches.flooded``; on CUDA its device flood)."""
    anchors, needed = _patch_regions(run)
    return bool(np.any(anchors)) and bool(np.any(needed == 1))


def _driven(device):
    model = _humanoids(device, worlds=2, lying=False)
    _set_drives(model, _hinge_dofs(model))
    return model


def _limited(device, *, drives):
    model = _humanoids(device, worlds=2, lying=False)
    hinges = _limit_hinges(model, VELOCITY_LIMIT)
    _set_drives(model, hinges[::2] if drives else hinges[:0])
    return model


def _thrown(device):
    model = _scene_heap(device, nx=3, ny=3, nz=1, pitch=FREE_PITCH, register_attributes=True)
    _limit_free_bodies(model)
    return model


def _layer(device):
    return _heap(device, nx=3, ny=3, nz=1, pitch=LAYER_PITCH)


def _crowd(device):
    return _humanoids(device, per_world=4, spacing=0.4)


def _two_chains(device):
    return _scene_worlds(device, ["chain:2@0.03", "chain:3@0.03"])


def _hanging_chains(device):
    return _chain_worlds([2, 3], device)


def _capped_pair(device):
    return _capped_boxes(device, (DEPENETRATION_CAP,) * 2)


def _dropped_sphere(device):
    return _build_plane_model(device, separation=REBOUND_DROP, restitution=0.5, gravity=REBOUND_GRAVITY)[0]


def _limited_alone(device):
    return _limited(device, drives=False)


def _limited_driven(device):
    return _limited(device, drives=True)


def _slide(model, state):
    _set_box_velocities(model, state, SLIDE_SPEED)


_LAYER = {"build": _layer, "init": _slide_and_drop, "atol": FREE_MODE_ATOL, "rms": MODE_TRACK_Q_RMS}
_HUMANOID_BOX = {"build": _mixed_scene, "rtol": DRIVE_TRACK_RTOL, "rms": MODE_TRACK_Q_RMS}
_PATCHES = {"options": PATCHES, "witness": _flooded, "same": _patch_regions}
# test_warmstart_tracks_steel's bounds: the articulation's DOFs within the factorizations' rounding, the free boxes'
# within TRACK_RTOL and FREE_MODE_ATOL.
_WARM = {"runner": _SeedRun, "init": _slide, "rtol": DRIVE_TRACK_RTOL, "free": (TRACK_RTOL, FREE_MODE_ATOL)}

CASES: dict[str, _Case] = {
    # The baseline: point friction and joint limits on the contact scene, then a crowd whose units join pairs of
    # humanoids (phase B on CUDA, the walk of the two-sided rows on the CPU), and two topologies (the generic response).
    "point friction": _Case(_build_contact_scene),
    # The crowd's first step is not compared: on the same contacts its velocities differ by 6.5e-6 m/s (the host's
    # rounding against CUDA's FMA contraction, through four Gauss-Seidel iterations of coupled humanoids), while its
    # coordinates after 30 steps differ by 1.6e-6 RMS, 0.0016 of the bound. Its own tests bound its coordinates alone.
    "crowd": _Case(_crowd, rows=CROWD_ROWS, rtol=None, witness=_two_sided),
    "two topologies": _Case(_two_chains, rows=64, init=None),
    "hanging chains": _Case(_hanging_chains, rows=64, init=None),
    # Row features of the contact scene.
    "regularization": _Case(_build_contact_scene, {"contact_regularization": REGULARIZATION}),
    "velocity iterations": _Case(_build_contact_scene, {"velocity_iterations": VELOCITY_ITERATIONS}, witness=_unbiased),
    "friction position iterations": _Case(
        _build_contact_scene,
        {"iterations": FRICTION_ITERATIONS, "contact_friction_position_iterations": FRICTION_POSITION_ITERATIONS},
    ),
    "one-row contacts": _Case(_build_contact_scene, {"contact_friction_gap_max": ONE_ROW_GAP}),
    "depenetration cap": _Case(_capped_pair, rows=CAP_ROWS, init=None, witness=_capped),
    "restitution": _Case(_dropped_sphere, rows=REBOUND_ROWS, init=None, rtol=REBOUND_RTOL, witness=_rebounds),
    # Internal and late rows of the humanoids.
    "drive rows": _Case(
        _driven, {"drive_mode": "rows"}, rows=HUMANOID_ROWS, init=_at_rest, rtol=DRIVE_TRACK_RTOL, witness=_drives
    ),
    "joint velocity limits": _Case(
        _limited_alone,
        {"enable_joint_velocity_limits": True},
        rows=HUMANOID_ROWS,
        init=_spin_hinges,
        rtol=DRIVE_TRACK_RTOL,
        witness=_limited_velocity,
    ),
    "fused clamp": _Case(
        _limited_driven,
        {"enable_joint_velocity_limits": True, "drive_mode": "rows"},
        rows=HUMANOID_ROWS,
        init=_spin_hinges,
        rtol=DRIVE_TRACK_RTOL,
        witness=_limited_velocity,
    ),
    "free-body velocity limits": _Case(_thrown, rows=FREE_ROWS, init=_throw_free_bodies, witness=_limited_velocity),
    # Friction patches (Steel's default), on the layer of boxes and the humanoid with a box: the patches' anchors and
    # row counts after the first step are CUDA's (the CPU's serial flood, CUDA's device flood).
    "patches layer": _Case(**_LAYER, **_PATCHES),
    "patches humanoid": _Case(**_HUMANOID_BOX, **_PATCHES),
    # The contact warm start, under point friction and under patches.
    "warm start": _Case(_build_contact_scene, **_WARM, rms=MODE_TRACK_Q_RMS, witness=_seeded),
    "warm start patches": _Case(_build_contact_scene, PATCHES, **_WARM, rms=TRACK_RMS, witness=_seeded),
}
"""The features the CPU runs: each one's scene and bounds, those of its test against Steel."""


def _step_on(run, source) -> None:
    """One step of ``run`` on the contacts of ``source``, the other device's run, which has just collided.

    ``run`` collides as it steps (its own buffers, a runner's material fill), then every contact array is overwritten
    with ``source``'s: the CPU's collision rounds otherwise than CUDA's, so the first step's comparison is the solve's
    alone.
    """
    if hasattr(run, "collide"):
        run.collide()
    else:
        run.pipeline.collide(run.state_0, run.contacts)
    for name, array in vars(run.contacts).items():
        if isinstance(array, wp.array):
            wp.copy(array, getattr(source.contacts, name))
    run.solver.step(run.state_0, run.state_1, run.control, run.contacts, DT)
    run.state_0, run.state_1 = run.state_1, run.state_0


def _assert_case(test, name: str, case: _Case) -> dict:
    """Step ``case`` on the CPU and on CUDA side by side and compare; return the measured differences.

    The first step solves CUDA's contacts on both devices (:func:`_step_on`): its velocities agree within the case's
    ``rtol`` and ``atol``. Every later step collides on its own device, so the coordinates after :data:`TRACK_STEPS`
    steps, within the case's RMS bound, include the two collision pipelines' rounding.
    """
    runs = {}
    for device in ("cpu", CUDA):
        model = case.build(device)
        run = case.runner(model, case.rows, {}, **case.options)
        if case.init is not None:
            case.init(model, run.state_0)
        runs[device] = run
    cpu, cuda = runs["cpu"], runs[CUDA]
    test.assertEqual(cpu.solver._scene.sweep, "serial")
    test.assertEqual(cuda.solver._scene.sweep, "phases")
    test.assertEqual(cpu.solver._scene.features, cuda.solver._scene.features)
    seen = dict.fromkeys(runs, case.witness is None)
    first = None
    for step in range(TRACK_STEPS):
        cuda.step()
        if step == 0:
            _step_on(cpu, cuda)
            first = cpu.state_0.joint_qd.numpy().copy(), cuda.state_0.joint_qd.numpy().copy()
            test.assertGreater(float(np.max(np.abs(first[1]))), 0.0, f"{name}: nothing moves")
            if case.same is not None:
                for index, (a, b) in enumerate(zip(case.same(cpu), case.same(cuda), strict=True)):
                    np.testing.assert_array_equal(a, b, err_msg=f"{name}: same({index}) after the first step")
        else:
            cpu.step()
        if case.witness is not None:
            for device, run in runs.items():
                seen[device] = case.witness(run) or seen[device]
    test.assertTrue(all(seen.values()), f"{name}: the witness held on {seen}")
    for device, run in runs.items():
        test.assertFalse(np.any(run.solver.constraint_overflow.numpy()), f"{name} on {device}: a budget overflowed")
    q_cpu, q_cuda = cpu.state_0.joint_q.numpy(), cuda.state_0.joint_q.numpy()
    test.assertTrue(np.all(np.isfinite(q_cpu)), name)
    if case.rtol is not None:
        free = np.zeros(first[1].size, dtype=bool)
        if case.free is not None:
            for start in _free_body_dofs(cuda.solver.model):
                free[start : start + 6] = True
            rtol, atol = case.free
            np.testing.assert_allclose(first[0][free], first[1][free], rtol=rtol, atol=atol, err_msg=f"{name}: free")
        np.testing.assert_allclose(
            first[0][~free], first[1][~free], rtol=case.rtol, atol=case.atol, err_msg=f"{name}: first step"
        )
    rms = _rms(q_cpu, q_cuda)
    test.assertLess(rms, case.rms, f"{name}: joint_q after {TRACK_STEPS} steps")
    return {"first_max_abs": float(np.max(np.abs(first[0].astype(np.float64) - first[1]))), "q_rms": rms}


ROW_INTEGERS = ("_sparse_row_dof", "_row_type", "_row_parent")
"""Row fields the CPU builds as CUDA does, entry for entry: the supports, the row types, the friction rows' parents."""

ROW_FLOATS = ("_sparse_row_factor", "_diag", "_rhs", "_sparse_row_incident", "_row_mu", "_phi", "_target_velocity")
"""Row fields the CPU computes with its own rounding: its factor sums one lane's entries in another order than CUDA's
warp, and the host contracts no product into an FMA. Compared by :func:`_row_excess`."""

ROW_RTOL = 1.0e-5
ROW_ATOL = 1.0e-6
"""Bounds of :data:`ROW_FLOATS` between the CPU's rows and CUDA's, :func:`_row_excess`'s: a 27-DOF factor's rounding,
27 x float32's epsilon (1.19e-7), is 3.2e-6 of a row's scale, about a third of ``ROW_RTOL``. Measured on an L40: the
worst difference is 0.55 of the bound (the lying humanoid's ``_rhs``), the
Jacobian rows' 0.30 (entrywise up to 1.1e-4 relative on small entries, hence the per-row scale)."""


def _row_excess(actual: np.ndarray, expected: np.ndarray) -> float:
    """The largest difference of two row fields over its bound, ``ROW_RTOL`` times the row's scale plus ``ROW_ATOL``.

    A row's Jacobian entries (``[rows, support]``) are dot products with the factor's columns, whose rounding scales
    with the row's largest entry, so their scale is that entry; a per-row value's scale is its own magnitude.
    """
    actual, expected = actual.astype(np.float64), expected.astype(np.float64)
    scale = np.abs(expected)
    if expected.ndim == 2:
        scale = np.broadcast_to(np.max(scale, axis=1, keepdims=True), expected.shape)
    excess = np.abs(actual - expected) / (ROW_RTOL * scale + ROW_ATOL)
    return float(np.max(excess)) if excess.size else 0.0


def _rows(device, state_of, gap):
    """One lying humanoid's scene rows after one step from ``state_of(model)``, with the joint limits within ``gap``."""
    model = _humanoids(device)
    state = state_of(model)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    solver = SolverSteel(model, articulation_constraint_max=128, **{**SCENE, "joint_limit_activation_gap": gap})
    solver.step(state, model.state(), model.control(), contacts, DT)
    rows = int(solver._constraint_count.numpy()[0])
    count = int(contacts.rigid_contact_count.numpy()[0])
    view = {name: getattr(solver, name).numpy()[0, :rows] for name in ROW_INTEGERS + ROW_FLOATS}
    view["_contact_slot"] = solver._contact_slot.numpy()[:count]
    view["limit_rows"] = int(solver._scene_limit_rows.numpy()[0])
    return solver, view


def _at_rest_state(model):
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state


def _lowered_state(model):
    """The humanoid lowered 0.25 m into the ground: the torso and legs touch too."""
    state = model.state()
    joint_q = model.joint_q.numpy().copy()
    joint_q[2] -= 0.25
    state.joint_q.assign(joint_q)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state


@unittest.skipUnless(wp.is_cuda_available(), "compares the CPU scene with the CUDA scene")
class TestSteelSceneCpu(unittest.TestCase):
    def test_cpu_scene_tracks_the_cuda_scene(self):
        """Step every feature's scene on the CPU as on CUDA, within the bounds of its test against Steel.

        Per :data:`CASES`: the velocities after the first step within the feature's ``rtol`` and ``atol``, the joint
        coordinates after :data:`TRACK_STEPS` steps within its RMS bound, the feature acting on both devices (its
        witness), the CPU sweeping serially and CUDA in phases, with the same build constants (``SceneFeatures``).
        """
        for name, case in CASES.items():
            with self.subTest(case=name):
                _assert_case(self, name, case)

    def test_cpu_rows_match_cuda_rows(self):
        """Build the scene's rows on the CPU as on CUDA: the same rows in the same places, their values to rounding.

        ``test_scene_rows.test_rows_match_steel_rows_bitwise``'s humanoid, lying (two contacts, the bounds within
        0.1 rad) and lowered 0.25 m into the ground (more contacts, every bound within a 10 rad gap): after one step of
        each device the rows' count, the contacts' slots, the limit rows, the supports, the row types and the parents
        are equal entry for entry, and the Jacobians, diagonals, right-hand sides, incident velocities, friction
        coefficients, gaps and targets within :func:`_row_excess`'s bound.
        """
        for pose, state_of, gap in (("lying", _at_rest_state, 0.1), ("lowered", _lowered_state, 10.0)):
            with self.subTest(pose=pose):
                (cpu, rows_cpu), (cuda, rows_cuda) = (_rows(device, state_of, gap) for device in ("cpu", CUDA))
                self.assertEqual(cpu._scene.sweep, "serial")
                for solver in (cpu, cuda):
                    self.assertFalse(np.any(solver.constraint_overflow.numpy()))
                self.assertGreater(rows_cpu["limit_rows"], 0)
                self.assertGreater(int(np.count_nonzero(rows_cpu["_contact_slot"] >= 0)), 0)
                for name in ("limit_rows", "_contact_slot", *ROW_INTEGERS):
                    np.testing.assert_array_equal(rows_cpu[name], rows_cuda[name], err_msg=f"{pose}: {name}")
                excess = {name: _row_excess(rows_cpu[name], rows_cuda[name]) for name in ROW_FLOATS}
                self.assertLessEqual(max(excess.values()), 1.0, f"{pose}: difference over bound {excess}")

    def test_cpu_sleeping_rests_like_cuda(self):
        """Sleep the heap's islands on the CPU as on CUDA: every box asleep, the same islands, the same heights.

        ``test_sleeping_tracks_steel``'s outcome on both devices: the 108 boxes of ``_heap`` settle with sleeping at
        :data:`HEAP_ITERATIONS` and all sleep within :data:`SLEEP_STEPS` steps, their partitions into islands are the
        same sets, and their heights agree within :data:`REST_ATOL`.
        """
        runs = {
            device: _sleep_run(_heap(device), free_body_constraint_max=HEAP_FREE_ROWS, iterations=HEAP_ITERATIONS)
            for device in ("cpu", CUDA)
        }
        asleep_at = {}
        for step in range(1, SLEEP_STEPS + 1):
            for device, run in runs.items():
                run.step()
                if device not in asleep_at and _asleep(run):
                    asleep_at[device] = step
        self.assertEqual(sorted(asleep_at), sorted(runs), f"asleep within {SLEEP_STEPS} steps: {asleep_at}")
        for device, run in runs.items():
            self.assertTrue(_asleep(run), f"{device}: a box woke again by step {SLEEP_STEPS}")
            self.assertFalse(np.any(run.solver.constraint_overflow.numpy()), device)
        self.assertEqual(runs["cpu"].solver._scene.sweep, "serial")
        self.assertEqual(_islands(runs["cpu"].solver), _islands(runs[CUDA].solver))
        heights = {device: run.state_0.body_q.numpy()[:, 2] for device, run in runs.items()}
        np.testing.assert_allclose(heights["cpu"], heights[CUDA], rtol=0.0, atol=REST_ATOL)

    def test_cpu_bilateral_rows_track_cuda(self):
        """Step the mimic and connect rows on the CPU as on CUDA, within the bounds of their tests against Steel.

        The four-bar (a closure inside one articulation), the carried load (closures to a kinematic carrier) and the
        mimic chain, at 32 iterations without contacts: the joint coordinates after steps 1 and 30 within
        ``test_scene_aliases.BILATERAL_Q_ATOL`` of the CUDA scene's, and the rows in their families on both devices.
        """
        builder, carrier, _, _ = _build_carried_load()
        scenes = {
            "four-bar": (lambda device: _build_four_bar().finalize(device=device), {"leader_target": 0.6}, "connect"),
            "carried load": (lambda device: builder.finalize(device=device), {"carrier": carrier}, "connect"),
            "mimic chain": (
                lambda device: _build_two_revolute_chain(1.0, -0.5)[0].finalize(device=device),
                {"leader_target": 0.6},
                "mimic",
            ),
        }
        families = {"connect": FAMILY_CONNECT, "mimic": FAMILY_MIMIC}
        for name, (build, run, family) in scenes.items():
            with self.subTest(scene=name):
                results = {
                    device: _bilateral_run(build(device), steps=30, **run, **BILATERAL, solve_mode="scene")
                    for device in ("cpu", CUDA)
                }
                for device, (solver, _, _) in results.items():
                    self.assertGreater(int(solver._scene_family_rows.numpy()[families[family]].sum()), 0, device)
                    self.assertFalse(np.any(solver.constraint_overflow.numpy()), device)
                self.assertEqual(results["cpu"][0]._scene.sweep, "serial")
                _assert_tracks_steel(self, name, results["cpu"][1], results[CUDA][1], reference="CUDA's")


if __name__ == "__main__":
    unittest.main()
