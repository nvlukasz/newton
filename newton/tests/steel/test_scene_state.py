# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Steel's state-carrying features in its scene solve (``solve_mode="scene"``): persistent friction patches and
the contact warm start.

Steel's matrix-free default is persistent friction patches (``friction_anchor_beta=None`` resolves to 0.2): each body
pair's contacts form regions, at most two of each region's contacts are its anchors, which get the region's friction
rows at the anchor points (the other members keep their normal row alone), and the anchors share the region's pooled
normal load (``row_mu`` divided by the anchors that got rows, ``link_patch_rows``; the friction radius is ``row_mu``
times the sum of the region's normal impulses, a ring through the normal rows' parents). The scene builds the patches
before it routes the contacts, writes the anchors' friction rows at the anchor points (a ``patches`` variant of its
contact response kernel), links the rings on its rows, and walks them in its sweeps (``SceneFeatures.patches``).

Steel's contact warm start (``contact_warmstart=True``) seeds each matched contact's rows with its impulses of the
previous step (``gather_contact_warmstart`` on the scene's one row family, and ``seed_patch_impulses`` for the patch
anchors), and the scene applies the seeds to its velocity change ``delta`` side by side before its sweeps
(``scene_seed_sides``), where Steel installs them into its velocities through its dense responses.

Steel's experimental sleeping (``enable_sleeping=True``) runs around the scene's step as around Steel's: ``begin``
builds the islands and their wake events before the step, ``finish`` freezes the quiet ones after it, and the scene's
row build reads Steel's masks, so a sleeping island's contacts get no unit (``contact_response_mask``), its joint
limits and drives no row (the masked ``limit_q_index``) and its joint friction none (``_constraint_art_active``), and
friction patches carry its frozen regions. Islands join over contacts within Steel's solve worlds, which the scene's
fold does not change.

The scene is checked against Steel's matrix-free solve within the tracking tolerances of ``test_scene_features.py``,
its terms without ``J`` (the coefficients, the row counts, the patch state, the gather's law) bit for bit, against its
serial oracle bit for bit, and against its eager steps under capture; its sleeping by its outcome against Steel's (the
islands asleep, the resting poses) and against an oracle of the rows an articulation must have.
"""

import itertools
import unittest
import warnings
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton import NewtonWarning
from newton._src.solvers.steel.kernels import CONTACT_GENERATION_NONE, PGS_CONSTRAINT_TYPE_CONTACT, reset_row_warmstart
from newton._src.solvers.steel.kernels import gather_contact_warmstart as steel_gather_contact_warmstart
from newton._src.solvers.steel.scene_pgs import RANGE_LATE, RANGE_TWO_SIDED
from newton._src.solvers.steel.scene_rows import FAMILY_JOINT_FRICTION, FAMILY_LIMIT
from newton._src.solvers.steel.scene_warmstart import (
    _reset_selects_world,
    scene_reset_row_warmstart,
    scene_reset_world_selection,
    scene_seed_sides,
)
from newton._src.solvers.steel.solver_steel import _SOLVE_MODES
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import CONTACT_MAX, DT, _build_contact_scene, _heap, _set_box_velocities
from newton.tests.steel._scene_fixtures import _global_box_model, _humanoids, _rms, _Run, _scene_heap
from newton.tests.steel.test_capacity import _box_on_ground
from newton.tests.steel.test_scene_drives import DRIVE_TRACK_RTOL
from newton.tests.steel.test_scene_features import (
    CROWD_ROWS,
    TRACK_ATOL,
    TRACK_RMS,
    TRACK_ROWS,
    TRACK_RTOL,
    TRACK_STEPS,
    _assert_phases_match_serial_oracle,
)
from newton.tests.steel.test_scene_velocity_iterations import VELOCITY_ITERATIONS, _FixedRun
from newton.tests.steel.test_scene_velocity_limits import _free_body_dofs, _throw_free_bodies
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

# The tolerances, scenes and runs shared with the other scene test modules (test_scene_features, test_scene_drives,
# test_scene_friction_modes, test_scene_velocity_iterations, test_scene_velocity_limits) are imported by the functions
# that use them, all behind the scene-mode gate, so that this module's own checks run without those modules.

PATCHES = {"friction_anchor_beta": None}
"""Steel's default friction over the :data:`~newton.tests.steel._scene_fixtures.SCENE` options' point friction: an
omitted ``friction_anchor_beta``, which Steel's matrix-free solve resolves to persistent friction patches (0.2)."""

STEEL_BETA = 0.2
"""Steel's ``friction_anchor_beta`` for an omitted one in its matrix-free solve (``SolverSteel``'s constructor)."""

POINT_FRICTION = "point friction"
"""The fragment of Steel's warnings where an omitted ``friction_anchor_beta`` selects point friction (contact
compliance and the friction modes other than ``"fixed_normal"``)."""

RADIUS_ITERATIONS = 64
"""Iterations of the friction radius test. A tangent pair is clamped to its region's pooled load at its visit; the
normal rows of the region visited after it move the pooled load on, so the bound holds after a step to the tolerance
of :data:`RADIUS_RTOL` once the sweeps have converged: the sliding free box's anchors pass their radius by 6.8 % after
4 iterations and 1.0e-4 after 16, and stay within it after 64."""

RADIUS_RTOL = 1.0e-6
"""Relative slack of an anchor's tangent impulse over its friction radius ``row_mu * sum of the ring's normal
impulses`` after a step: rounding (5.0e-8 at most measured at :data:`RADIUS_ITERATIONS`, the articulated box)."""

CAPTURE_STEPS = 30
"""Steps of the capture tests."""

WARMSTART = {"contact_warmstart": True}
"""Steel's contact warm start over the :data:`~newton.tests.steel._scene_fixtures.SCENE` options."""

SETTLE_STEPS = 10
"""Steps before a seeded step: the lying humanoids, 4 mm above the ground, land in about seven steps, so their ground
contacts carry impulses, as do the units that join them."""

SLIDE_SPEED = 0.1
"""Sliding speed [m/s] of the warm-start tests' boxes: 0.42 mm per step at 240 Hz, within the collision pipeline's
contact matching threshold (``contact_matching_pos_threshold``, 0.5 mm of a contact's midpoint per pass), so the
sliding contacts stay matched and carry their impulses; at the 1.5 m/s of the other tests a contact moves 6 mm per
step and every sliding contact starts cold."""

SEED_ATOL = 5.0e-5
"""Absolute agreement [N s] of the scene's seeded impulses with Steel's, contact for contact, over thirty steps: the
previous step's impulses of two solves whose states drift apart by rounding (1.1e-5 at most measured, against seeds of
0.05 to 0.25 N s); a wrong seed (unscaled, untransported, another clamp) is off by 1e-2 or more. The relative
agreement is ``test_scene_drives.DRIVE_TRACK_RTOL``: the previous step's impulses, which the factorizations' rounding
moves apart on the articulation."""


SWEEP_RULE_RTOL = 1.0e-4
"""Agreement of ``delta`` after a position solve with :func:`_seed_reference` on the final impulses, relative to the
largest entry: the sweep's sum of every change against one sum of the final impulses, rounding apart (2.4e-7 at most
measured on the humanoids, 1.6e-5 on the heap over steps 10 to 15); a rule that
drifted (a free entry taking its factor value, an entry dropped) is off by the entries' own size."""


class _Matching:
    """A run with Steel's contact warm start (:data:`WARMSTART`, unless ``options`` turn it off) and a deterministic
    collision pipeline that matches each step's contacts to the last pass's (``contact_matching="latest"``), which the
    warm start requires."""

    def __init__(self, model, rows, overrides, **options):
        super().__init__(model, rows, overrides, collide=False, **{**WARMSTART, **options})
        self.pipeline = newton.CollisionPipeline(model, deterministic=True, contact_matching="latest")
        self.contacts = self.pipeline.contacts()


class _WarmRun(_Matching, _Run):
    """A :class:`~newton.tests.steel._scene_fixtures._Run` with the warm start (:class:`_Matching`)."""


def _warm_fixed_run(model, rows, overrides, **options):
    """A ``_FixedRun`` with the warm start (:class:`_Matching`): one captured step replays every later step."""

    class _WarmFixedRun(_Matching, _FixedRun):
        pass

    return _WarmFixedRun(model, rows, overrides, **options)


def _record_scene_seeds(solver):
    """Keep each step's seeded impulses in ``solver.seeds`` (``{path: impulses}``), eagerly: the scene's impulses when
    its position solve starts (``SceneSolve.solve`` with ``reset``), after the gather and the patch seed."""
    scene = solver._scene
    solve = scene.solve

    def seeded_solve(solver, *, rhs=None, reset=True, **kwargs):
        if reset:
            # A copy: on the CPU numpy() is a view of the impulses, which the solve overwrites.
            solver.seeds = {0: solver._impulses.numpy().copy()}
        return solve(solver, rhs=rhs, reset=reset, **kwargs)

    scene.solve = seeded_solve


class _SeedRun(_WarmRun):
    """A :class:`_WarmRun` that keeps each step's seeded impulses in ``solver.seeds`` (:func:`_record_scene_seeds`)."""

    def __init__(self, model, rows, overrides, **options):
        super().__init__(model, rows, overrides, **options)
        _record_scene_seeds(self.solver)


def _record_steel_seeds(solver):
    """Keep each step's seeded impulses of Steel's matrix-free solve in ``solver.seeds`` (``{path: impulses}``),
    eagerly: its articulated (dense) and free-body rows when it installs them, after both gathers and the patch seed."""
    install = solver._apply_mf_warmstart_velocity

    def seeded_install():
        # Copies: on the CPU numpy() is a view of the impulses, which the solve overwrites.
        solver.seeds = {0: solver._impulses.numpy().copy(), 1: solver._mf_impulses.numpy().copy()}
        install()

    solver._apply_mf_warmstart_velocity = seeded_install


def _contact_seeds(solver, count) -> dict:
    """The seeds of the last step's kept contacts, ``{contact: [normal, tangent, tangent]}``, NaN past their rows."""
    slot = solver._contact_slot.numpy()[:count]
    path = solver._contact_path.numpy()[:count]
    world = solver._contact_world.numpy()[:count]
    needed = solver._contact_slots_needed.numpy()[:count]
    seeds = {}
    for c in range(count):
        if slot[c] < 0 or path[c] < 0:
            continue
        values = np.full(3, np.nan, np.float32)
        values[: needed[c]] = solver.seeds[int(path[c])][world[c], slot[c] : slot[c] + needed[c]]
        seeds[c] = values
    return seeds


def _relation(solver, contacts) -> int:
    """Steel's ``warmstart_history_relation`` of ``contacts`` to the solved history: 0 unrelated, 1 the same set, 2
    matched to it."""
    stream = solver._contact_stream(contacts)
    previous = int(solver._ws_history_generation.numpy()[0])
    if stream == 0 or int(solver._ws_history_stream.numpy()[0]) != stream or previous == CONTACT_GENERATION_NONE:
        return 0
    if int(contacts.contact_generation.numpy()[0]) == previous:
        return 1
    return 2 if int(contacts.rigid_contact_match_generation.numpy()[0]) == previous else 0


def _check_gather(test, solver) -> dict:
    """Check every gather of ``solver``'s scene against Steel's law; returns the counts ``{"gathers", "seeded"}``.

    Each gather starts from the whole capacity at zero, and its impulses equal Steel's ``gather_contact_warmstart``
    (Steel's own kernel) launched on the scene's own snapshot and rows bit for bit (with the coefficients Steel's
    gather reads, ``SceneSolve.gather_mu`` under friction patches with free bodies). A matched contact's normal row
    holds, independently in NumPy, ``decay * (dt / dt_previous)`` times its carried impulse (0 where the normal
    reversed), every other contact's normal row zero.
    """
    gather = solver._gather_warmstart
    counts = {"gathers": 0, "seeded": 0}

    def checked(contacts, route, dt):
        test.assertEqual(route, 0)
        test.assertFalse(np.any(solver._impulses.numpy()), "the impulses do not start the gather at zero")
        relation = _relation(solver, contacts)
        gather(contacts, route, dt)
        expected = wp.zeros_like(solver._impulses)
        wp.launch(
            steel_gather_contact_warmstart,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                route,
                solver._contact_path,
                solver._contact_slot,
                solver._contact_world,
                contacts.rigid_contact_match_index,
                contacts.rigid_contact_match_generation,
                solver._ws_prev_dense_slot,
                solver._ws_prev_impulses,
                solver._ws_prev_row_type,
                solver._ws_prev_row_parent,
                solver._constraint_count,
                solver._row_type,
                solver._row_parent,
                contacts.rigid_contact_normal,
                solver._ws_prev_contact_normal,
                solver._row_mu if solver._scene.gather_mu is None else solver._scene.gather_mu,
                solver._contact_warmstart_decay,
                dt,
                solver._ws_history_dt,
                contacts.contact_generation,
                solver._contact_stream(contacts),
                solver._ws_history_generation,
                solver._ws_history_stream,
                solver._articulation_constraint_max,
            ],
            outputs=[expected],
            device=solver.model.device,
        )
        got = solver._impulses.numpy()[0]
        test.assertTrue(np.array_equal(got, expected.numpy()[0]), "the gather differs from Steel's law")
        count = int(contacts.rigid_contact_count.numpy()[0])
        rows = int(solver._constraint_count.numpy()[0])
        path, slot = solver._contact_path.numpy()[:count], solver._contact_slot.numpy()[:count]
        match = contacts.rigid_contact_match_index.numpy()[:count]
        normal, previous_normal = contacts.rigid_contact_normal.numpy(), solver._ws_prev_contact_normal.numpy()
        prev_slot, prev_impulses = solver._ws_prev_dense_slot.numpy(), solver._ws_prev_impulses.numpy()[0]
        row_type, prev_type = solver._row_type.numpy()[0], solver._ws_prev_row_type.numpy()[0]
        history = np.float32(solver._ws_history_dt.numpy()[0])
        ratio = np.float32(dt) / history if history > 0.0 else np.float32(1.0)
        scale = np.float32(np.float32(solver._contact_warmstart_decay) * ratio)
        for c in range(count):
            row = int(slot[c])
            if path[c] != 0 or row < 0 or row >= rows or row_type[row] != PGS_CONSTRAINT_TYPE_CONTACT:
                continue
            expected_normal = np.float32(0.0)
            source = c if relation == 1 else int(match[c])
            if relation and source >= 0 and 0 <= prev_slot[source] < prev_impulses.shape[0]:
                if prev_type[prev_slot[source]] == PGS_CONSTRAINT_TYPE_CONTACT:
                    turn = float(np.dot(normal[c].astype(np.float64), previous_normal[source].astype(np.float64)))
                    test.assertGreater(abs(turn), 1.0e-3)
                    if turn > 0.0:
                        expected_normal = max(np.float32(scale * prev_impulses[prev_slot[source]]), np.float32(0.0))
            test.assertEqual(got[row], expected_normal, f"contact {c}")
        counts["gathers"] += 1
        counts["seeded"] += int(np.count_nonzero(got))

    solver._gather_warmstart = checked
    return counts


def _seed_reference(test, solver, impulses=None) -> tuple[np.ndarray, dict]:
    """The scene's seeded ``delta`` of the last step in NumPy float32, side by side in ``scene_seed_sides``' order
    (:func:`_seed_walk` on the solver's rows). ``impulses`` replaces the solver's (the seeds). Returns
    ``(delta, visited)``, ``visited`` mapping each visited ``(row, entry)`` to its side.
    """
    scene, units = solver._scene, solver._scene_units
    return _seed_walk(
        test,
        side_count=scene.side_count,
        side_row_begin=scene.side_row_begin.numpy(),
        side_row_end=scene.side_row_end.numpy(),
        incidence_start=units.incidence_start.numpy(),
        incidence_unit=units.incidence_unit.numpy(),
        unit_order=units.unit_order.numpy(),
        unit_rows=units.unit_rows.numpy(),
        unit_row_start=scene.unit_row_start.numpy(),
        offset=solver._articulation_world_dof_offset.numpy(),
        block=solver._articulation_response_dof_count.numpy(),
        dofs=solver._sparse_row_dof.numpy()[0],
        factor=solver._sparse_row_factor.numpy()[0],
        free=solver._sparse_row_free_response.numpy()[0],
        prefix=scene.row_free_prefix.numpy() if scene.free_response else np.zeros(scene.row_max, np.int32),
        impulses=solver._impulses.numpy()[0] if impulses is None else impulses,
        size=solver._sparse_factor_velocity_delta.shape[1],
    )


def _seed_walk(
    test,
    *,
    side_count,
    side_row_begin,
    side_row_end,
    incidence_start,
    incidence_unit,
    unit_order,
    unit_rows,
    unit_row_start,
    offset,
    block,
    dofs,
    factor,
    free,
    prefix,
    impulses,
    size,
) -> tuple[np.ndarray, dict]:
    """Seeded rows applied to a zeroed ``delta`` of ``size`` entries in NumPy float32, in ``scene_seed_sides``' order.

    Side ``s`` visits its one-sided rows ``side_row_begin[s]`` to ``side_row_end[s]``, then the rows of each two-sided
    unit of its incidence list in order (``unit_order``, skipping a dropped unit), and adds ``resp * lambda`` to each
    entry of the row in its own coordinate block (``offset``, ``block``: the articulations' world DOF offsets and
    response DOF counts), ``resp`` the free response over the row's free ``prefix`` and its factor value past it.
    Asserts that every entry of a one-sided row is in its side's block and that no entry is visited twice. Returns
    ``(delta, visited)``, ``visited`` mapping each visited ``(row, entry)`` to its side.
    """
    delta = np.zeros(size, np.float32)
    visited = {}
    for side in range(side_count):
        first, last = offset[side], offset[side] + block[side]

        def visit(row, one_sided, side=side, first=first, last=last):
            impulse = impulses[row]
            for k in range(dofs.shape[1]):
                index = dofs[row, k]
                if index < 0:
                    continue
                if one_sided:
                    test.assertTrue(first <= index < last, f"row {row} entry {k} is outside side {side}'s block")
                elif not first <= index < last:
                    continue
                test.assertNotIn((row, k), visited, f"row {row} entry {k}")
                visited[row, k] = side
                if impulse != 0.0:
                    response = free[row, k] if k < prefix[row] else factor[row, k]
                    delta[index] = np.float32(delta[index] + np.float32(response * impulse))

        for row in range(side_row_begin[side], side_row_end[side]):
            visit(row, True)
        for entry in range(incidence_start[side], incidence_start[side + 1]):
            slot = unit_order[incidence_unit[entry]]
            if unit_row_start[slot] >= 0:
                for row in range(unit_row_start[slot], unit_row_start[slot] + unit_rows[slot]):
                    visit(row, False)
    return delta, visited


def _ring(parents: np.ndarray, normal_row: int) -> list[int]:
    """The normal rows of a patch region from ``normal_row``, walking the normal rows' parents (Steel's ring, -1 ends
    a point contact's walk, a self-loop a lone member's)."""
    rows = [normal_row]
    row = int(parents[normal_row])
    while row >= 0 and row != normal_row:
        rows.append(row)
        row = int(parents[row])
    return rows


def _contact_rows(solver, count):
    """The step's kept contacts' ``(contact, normal row, rows)`` on the scene's rows (``_contact_slot``)."""
    slot = solver._contact_slot.numpy()[:count]
    needed = solver._contact_slots_needed.numpy()[:count]
    return [(c, int(slot[c]), int(needed[c])) for c in range(count) if slot[c] >= 0]


def _patch_state(solver, count):
    """The anchors of the step's contacts (``FrictionPatches.weight > 0``) and their carried history
    (``previous.valid``) after a step."""
    patches = solver._friction_patches
    return patches.view.weight.numpy()[:count] > 0.0, patches.previous.valid.numpy()[:count]


def _steel_mu(steel, c):
    """Steel's ``row_mu`` of contact ``c``'s rows on its own path: its articulated (dense) rows ``_row_mu`` or its
    free-body rows ``_mf_row_mu``."""
    world, slot = int(steel._contact_world.numpy()[c]), int(steel._contact_slot.numpy()[c])
    path = int(steel._contact_path.numpy()[c])
    rows = steel._row_mu if path == 0 else steel._mf_row_mu
    return rows.numpy()[world, slot : slot + 3]


@unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
def test_anchor_beta_none_resolves_as_steel(test, device):
    """Resolve an omitted ``friction_anchor_beta`` as Steel's constructor does, under the scene solve too.

    On the contact scene: Steel's default takes persistent friction patches (0.2), and contact compliance and the
    friction mode ``"bisection"`` point friction with Steel's point-friction warning. Each case resolves as with
    ``solve_mode="serial"``.
    """
    model = _build_contact_scene(device)
    cases = (
        ("default", {}, STEEL_BETA, False),
        ("contact_compliance", {"contact_compliance": True}, 0.0, True),
        ("friction_mode", {"friction_mode": "bisection"}, 0.0, True),
    )
    for name, options, beta, warns in cases:
        with test.subTest(case=name):
            resolved = {}
            for solve_mode in ("scene", "serial"):
                if warns:
                    with test.assertWarnsRegex(NewtonWarning, POINT_FRICTION):
                        solver = SolverSteel(model, solve_mode=solve_mode, **options)
                else:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        solver = SolverSteel(model, solve_mode=solve_mode, **options)
                    test.assertEqual([str(w.message) for w in caught if issubclass(w.category, NewtonWarning)], [])
                test.assertEqual(solver.solve_mode, solve_mode)
                resolved[solve_mode] = (solver.friction_anchor_beta, solver._friction_anchors_enabled)
            test.assertEqual(resolved["scene"], (beta, beta > 0.0))
            test.assertEqual(resolved["scene"], resolved["serial"])


def _track_patches(test, model, *, init, rtol, atol):
    """Step the scene and Steel's matrix-free solve with Steel's default patches side by side from ``init``'s state.

    The patch state after every step is Steel's bit for bit: the same anchors, the same carried history; the
    velocities after the first step agree within ``rtol`` and ``atol``, the joint coordinates after :data:`TRACK_STEPS`
    steps within ``test_scene_friction_modes.MODE_TRACK_Q_RMS``. Returns the two runs, scene first.

    The anchors are a discrete function of a state that differs from Steel's by rounding: a region's anchors are its
    farthest pair of contacts, so a tie between two pairs' distances at the last ULP can pick other anchors on other
    hardware or another compiler. The fixtures here stay far from such ties (30 of 30 steps equal);
    should one flip, the fallback is to compare the patch state after the first step alone, not a scene fault.
    """
    from newton.tests.steel.test_scene_friction_modes import MODE_TRACK_Q_RMS  # noqa: PLC0415

    scene = _Run(model, TRACK_ROWS, {}, **PATCHES)
    steel = _Run(model, TRACK_ROWS, {}, solve_mode="serial", **PATCHES)
    test.assertEqual(scene.solver.solve_mode, "scene")
    test.assertIsNone(steel.solver._scene)
    for run in (scene, steel):
        test.assertTrue(run.solver._friction_anchors_enabled)
        init(model, run.state_0)
    test.assertTrue(scene.solver._scene.features.patches)
    for step in range(1, TRACK_STEPS + 1):
        scene.step()
        steel.step()
        count = int(scene.contacts.rigid_contact_count.numpy()[0])
        test.assertEqual(count, int(steel.contacts.rigid_contact_count.numpy()[0]), f"step {step}")
        anchors, valid = _patch_state(scene.solver, count)
        steel_anchors, steel_valid = _patch_state(steel.solver, count)
        test.assertGreater(np.count_nonzero(anchors), 0, f"step {step}")
        test.assertEqual(np.count_nonzero(anchors), np.count_nonzero(steel_anchors), f"step {step}")
        test.assertTrue(np.array_equal(anchors, steel_anchors), f"the anchors differ from Steel's after step {step}")
        test.assertTrue(
            np.array_equal(valid, steel_valid), f"the anchor history differs from Steel's after step {step}"
        )
        if step == 1:
            qd_scene, qd_steel = scene.state_0.joint_qd.numpy(), steel.state_0.joint_qd.numpy()
            test.assertGreater(np.max(np.abs(qd_steel)), 0.0)
            np.testing.assert_allclose(qd_scene, qd_steel, rtol=rtol, atol=atol)
    q_scene, q_steel = scene.state_0.joint_q.numpy(), steel.state_0.joint_q.numpy()
    test.assertTrue(np.all(np.isfinite(q_scene)))
    test.assertLess(_rms(q_scene, q_steel), MODE_TRACK_Q_RMS)
    for run in (scene, steel):
        test.assertFalse(np.any(run.solver.constraint_overflow.numpy()))
    return scene, steel


@unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
def test_patch_friction_tracks_steel(test, device):
    """Step Steel's default friction patches as Steel's matrix-free solve does: one step and thirty.

    A layer of nine free boxes apart on the ground (``_heap(nx=3, ny=3, nz=1)``, 25 cm apart, every contact on Steel's
    free-body path), sliding at 1.5 m/s and dropped onto the ground at 0.5 m/s, and a lying humanoid beside a free box
    (articulated contacts beside free-body ones), both sliding at 1.5 m/s, with ``friction_anchor_beta`` omitted. After
    every step the anchors and the carried anchor history equal Steel's contact for contact; the velocities after the
    first step agree with Steel's within 1e-5 and ``test_scene_friction_modes.FREE_MODE_ATOL`` on the boxes (2.7e-6 at
    most measured), within the factorizations' rounding on the humanoid
    (``test_scene_drives.DRIVE_TRACK_RTOL`` and 1e-6: 2.4e-5 relative needed), and the joint coordinates after thirty
    steps within ``test_scene_friction_modes.MODE_TRACK_Q_RMS`` (1.1e-6 and 3.9e-7 measured).
    """
    from newton.tests.steel.test_scene_friction_modes import (  # noqa: PLC0415
        FREE_MODE_ATOL,
        LAYER_PITCH,
        _mixed_scene,
        _slide_and_drop,
    )

    scenes = {
        "layer": (
            lambda: _heap(device, nx=3, ny=3, nz=1, pitch=LAYER_PITCH),
            {"init": _slide_and_drop, "rtol": TRACK_RTOL, "atol": FREE_MODE_ATOL},
        ),
        "humanoid_box": (
            lambda: _mixed_scene(device),
            {"init": _set_box_velocities, "rtol": DRIVE_TRACK_RTOL, "atol": TRACK_ATOL},
        ),
    }
    for name, (build, options) in scenes.items():
        with test.subTest(scene=name):
            _track_patches(test, build(), **options)


@unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
def test_patch_radius_uses_divided_row_mu(test, device):
    """Share a region's friction budget among its anchors and bound their pairs by the region's pooled load.

    A free box on the ground (four contacts: one region, two anchors) and the contact scene's articulated box (a D6
    root on the ground beside two free boxes), sliding at 1.5 m/s, step once with ``friction_anchor_beta`` omitted.
    Each anchor's tangent rows hold the material coefficient (its normal row's ``row_mu``) divided by its region's
    anchors that got rows, the box's ``mu / 2``, and equal Steel's rows for the same contact (its free-body
    ``_mf_row_mu`` for a free box, its articulated ``_row_mu`` for the articulated one) bit for bit; every contact that
    is no anchor has its normal row alone, as in Steel. After the step each anchor's tangent impulse is within its
    radius, ``row_mu`` times the sum of its region's normal impulses (the ring of normal-row parents), and some
    anchor's passes ``row_mu`` times its own normal impulse alone: the radius is the region's.
    """
    boxes = {
        "free box": (lambda: _box_on_ground(device), 1),
        "articulated box": (lambda: _build_contact_scene(device, worlds=1), 0),
    }
    for name, (build, box_path) in boxes.items():
        with test.subTest(box=name):
            model = build()
            scene = _Run(model, TRACK_ROWS, {}, iterations=RADIUS_ITERATIONS, **PATCHES)
            steel = _Run(model, TRACK_ROWS, {}, solve_mode="serial", iterations=RADIUS_ITERATIONS, **PATCHES)
            for run in (scene, steel):
                _set_box_velocities(model, run.state_0)
                run.step()
            solver = scene.solver
            count = int(scene.contacts.rigid_contact_count.numpy()[0])
            test.assertEqual(count, int(steel.contacts.rigid_contact_count.numpy()[0]))
            weight = solver._friction_patches.view.weight.numpy()[:count]
            np.testing.assert_array_equal(weight, steel.solver._friction_patches.view.weight.numpy()[:count])
            needed = solver._contact_slots_needed.numpy()[:count]
            # Under the infinite friction gap an anchor has three rows, every other contact its normal row alone.
            np.testing.assert_array_equal(needed, np.where(weight > 0.0, 3, 1))
            np.testing.assert_array_equal(needed, steel.solver._contact_slots_needed.numpy()[:count])
            mu = solver._row_mu.numpy()[0]
            parents = solver._row_parent.numpy()[0]
            impulses = solver._impulses.numpy()[0]
            steel_path = steel.solver._contact_path.numpy()[:count]
            contact_rows = _contact_rows(solver, count)
            rows_of = {normal_row: rows for _c, normal_row, rows in contact_rows}
            halves = ring_loaded = 0
            for c, normal_row, rows in contact_rows:
                if rows != 3:
                    continue
                ring = _ring(parents, normal_row)
                # link_patch_rows divides by the region's anchors that got their rows.
                anchors = sum(int(rows_of[row] == 3) for row in ring)
                expected = np.float32(mu[normal_row] / np.float32(anchors))
                np.testing.assert_array_equal(mu[normal_row + 1 : normal_row + 3], expected)
                np.testing.assert_array_equal(mu[normal_row : normal_row + 3], _steel_mu(steel.solver, c))
                if len(ring) == 4 and steel_path[c] == box_path:
                    test.assertEqual(anchors, 2)
                    np.testing.assert_array_equal(mu[normal_row + 1], np.float32(mu[normal_row] / 2))
                    halves += 1
                load = float(np.sum(impulses[ring], dtype=np.float64))
                tangent = float(np.hypot(impulses[normal_row + 1], impulses[normal_row + 2]))
                radius = float(mu[normal_row + 1]) * load
                test.assertLessEqual(tangent, radius * (1.0 + RADIUS_RTOL), f"contact {c}")
                if tangent > float(mu[normal_row + 1]) * float(impulses[normal_row]) * (1.0 + RADIUS_RTOL):
                    ring_loaded += 1
            test.assertGreaterEqual(halves, 2)
            test.assertGreater(ring_loaded, 0, "no anchor's pair passed its own normal's disk")


@unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
@unittest.skipUnless(hasattr(SolverSteel, "scene_statistics"), "needs the read-back API")
def test_patch_rows_follow_the_anchors(test, device):
    """Give the anchors three rows and every other contact one, under the infinite friction gap.

    A small heap of free boxes (``_heap(nx=3, ny=2, nz=2)``, box-box and ground contacts) steps once with Steel's
    default patches: every kept contact's rows are three for an anchor and one otherwise, Steel's row counts contact for
    contact, and the scene's rows are their sum; the contacts reach the generic response kernel's one-row variant
    (``one_row_contacts``) without a ``contact_friction_gap_max``, and the units are sized for one row each.
    """
    model = _heap(device, nx=3, ny=2, nz=2)
    scene = _Run(model, CROWD_ROWS, {}, **PATCHES)
    steel = _Run(model, CROWD_ROWS, {}, solve_mode="serial", **PATCHES)
    for run in (scene, steel):
        run.step()
    solver = scene.solver
    test.assertTrue(solver._friction_every_contact)
    test.assertTrue(solver._scene.one_row_contacts)
    test.assertTrue(solver._scene.features.patches)
    stats = solver.scene_statistics()
    test.assertEqual(solver._scene_units.unit_max, stats["row_max"] + solver._scene_units.limit_unit_count)
    count = int(scene.contacts.rigid_contact_count.numpy()[0])
    weight = solver._friction_patches.view.weight.numpy()[:count]
    needed = solver._contact_slots_needed.numpy()[:count]
    kept = solver._contact_slot.numpy()[:count] >= 0
    test.assertTrue(np.all(kept))
    np.testing.assert_array_equal(needed, np.where(weight > 0.0, 3, 1))
    np.testing.assert_array_equal(needed, steel.solver._contact_slots_needed.numpy()[:count])
    test.assertEqual(set(needed.tolist()), {1, 3})
    test.assertEqual(stats["rows"], int(needed.sum()))
    test.assertEqual(stats["dropped_rows"], 0)


@unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
def test_patch_anchors_take_precedence_over_shared_anchors(test, device):
    """Put the anchors' tangent rows at the anchor points whatever the shared-anchor flags, as Steel does.

    The layer of free boxes 1 mm above the ground (the witness points of a contact 1 mm apart) steps once under
    ``contact_shared_anchor`` and ``contact_friction_shared_anchor`` and without either. A row is its diagonal, its
    factor values, its free response and its incident velocity. With Steel's default patches
    the tangent rows of every anchor are the same bit for bit under every flag (Steel's ``contact_row_points``: patch
    rows act at the anchors; both flags warn as in Steel), and differ from the point-friction rows of the same contact;
    the normal rows are the point-friction ones bit for bit (the anchors move the tangent rows alone). Under point
    friction the friction flag moves the tangent rows to the witness midpoint.
    """
    from newton.tests.steel.test_scene_friction_modes import LAYER_PITCH  # noqa: PLC0415

    model = _heap(device, nx=3, ny=3, nz=1, pitch=LAYER_PITCH)
    flags = {
        "none": {},
        "contact_shared_anchor": {"contact_shared_anchor": True},
        "contact_friction_shared_anchor": {"contact_friction_shared_anchor": True},
    }
    rows = {}
    for beta in (None, 0.0):
        for name, options in flags.items():
            if beta is None and options:
                with test.assertWarnsRegex(NewtonWarning, "Patch friction selects its own friction locations"):
                    run = _Run(model, TRACK_ROWS, {}, friction_anchor_beta=beta, **options)
            else:
                run = _Run(model, TRACK_ROWS, {}, friction_anchor_beta=beta, **options)
            run.step()
            solver = run.solver
            count = int(run.contacts.rigid_contact_count.numpy()[0])
            arrays = [
                solver._diag.numpy()[0],
                solver._sparse_row_factor.numpy()[0],
                solver._sparse_row_free_response.numpy()[0],
                solver._sparse_row_incident.numpy()[0],
            ]
            rows[beta, name] = {
                c: tuple(np.array(array[slot : slot + needed]) for array in arrays)
                for c, slot, needed in _contact_rows(solver, count)
            }

    def same(a, b, part):
        return all(np.array_equal(x[part], y[part]) for x, y in zip(a, b, strict=True))

    tangents, normal = slice(1, 3), slice(0, 1)
    patch, point = rows[None, "none"], rows[0.0, "none"]
    anchors = [c for c, values in patch.items() if len(values[0]) == 3]
    test.assertGreater(len(anchors), 0)
    for name in flags:
        for c in anchors:
            test.assertTrue(same(rows[None, name][c], patch[c], tangents), f"{name}: contact {c}'s tangent rows moved")
    for c, values in patch.items():
        test.assertTrue(same(values, point[c], normal), f"contact {c}'s normal row differs from point friction's")
    test.assertTrue(any(not same(patch[c], point[c], tangents) for c in anchors))
    friction_shared = rows[0.0, "contact_friction_shared_anchor"]
    test.assertTrue(any(not same(friction_shared[c], point[c], tangents) for c in point))


def _two_sided_ring(solver) -> bool:
    """Whether a two-sided unit's normal row links another normal row of its region (a ring of at least two)."""
    rows = int(solver._constraint_count.numpy()[0])
    begin, end = solver._scene.row_ranges.numpy()[RANGE_TWO_SIDED]
    kinds = solver._row_type.numpy()[0, :rows]
    parents = solver._row_parent.numpy()[0, :rows]
    normal = np.flatnonzero(kinds == PGS_CONSTRAINT_TYPE_CONTACT)
    linked = normal[(parents[normal] >= 0) & (parents[normal] != normal)]
    return bool(np.any((linked >= begin) & (linked < end)))


@unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
def test_patches_match_the_serial_oracle(test, device):
    """Walk the regions' rings in the phased sweep exactly as one thread walking the rows does, in every order.

    A small heap of free boxes thrown into one another (two-sided free-body units in phase B, box on box) and four
    lying humanoids 0.4 m apart with a free box inside the first (two-sided articulated units), with Steel's default
    patches: the phases (the default color loop, and the while loop) match the serial oracle (``scene_sweep="serial"``)
    bit for bit over 30 steps in each phase order, with a region of two-sided rows whose ring holds two normal rows or
    more.
    """
    heap = _scene_heap(device, nx=3, ny=2, nz=2, register_attributes=True)
    _throw_free_bodies(heap, heap)
    crowds = {"heap": heap, "humanoids": _humanoids(device, per_world=4, spacing=0.4, props=True)}
    for name, model in crowds.items():
        with test.subTest(scene=name):
            _assert_phases_match_serial_oracle(
                test, model, PATCHES, witness=_two_sided_ring, what="no two-sided region linked two normal rows"
            )


@unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
def test_patches_replay_under_capture(test, device):
    """Replay one captured step with Steel's default patches exactly as eager steps, as regions come and go.

    Four lying humanoids 0.4 m apart with a free box inside the first, every body sliding at 1.5 m/s: contacts and their
    regions appear and disappear from step to step, and the anchors move. After each of :data:`CAPTURE_STEPS` steps
    ``joint_q``, ``joint_qd``, the impulses, the rows' parents and coefficients and the patch state (the anchors, their
    weights, points and displacements, the regions' rings and the carried history) equal the eager steps' bit for bit,
    and the anchors and the number of regions change between replays.
    """
    model = _humanoids(device, per_world=4, spacing=0.4, props=True)
    eager, captured = runs = [_FixedRun(model, CROWD_ROWS, {}, **PATCHES) for _ in range(2)]
    for run in runs:
        _set_box_velocities(model, run.state_0)
        run.step()
    with wp.ScopedCapture(device) as recorded:
        captured.step()
    anchor_sets, region_counts = set(), set()
    for step in range(2, CAPTURE_STEPS + 1):
        eager.step()
        wp.capture_launch(recorded.graph)
        for name in ("joint_q", "joint_qd"):
            a, b = getattr(eager.state_0, name).numpy(), getattr(captured.state_0, name).numpy()
            test.assertTrue(np.array_equal(a, b), f"{name} differs after step {step}")
        for name in ("_impulses", "_row_type", "_row_parent", "_row_mu"):
            a, b = getattr(eager.solver, name).numpy(), getattr(captured.solver, name).numpy()
            test.assertTrue(np.array_equal(a, b), f"{name} differs after step {step}")
        patches = [run.solver._friction_patches for run in runs]
        for name in ("weight", "next_contact", "point_a", "phi"):
            a, b = (getattr(p.view, name).numpy() for p in patches)
            test.assertTrue(np.array_equal(a, b), f"patch {name} differs after step {step}")
        for name in ("owner", "valid"):
            a, b = (getattr(p.current, name).numpy() for p in patches)
            test.assertTrue(np.array_equal(a, b), f"patch {name} differs after step {step}")
        test.assertTrue(np.array_equal(patches[0].previous.valid.numpy(), patches[1].previous.valid.numpy()))
        count = int(eager.contacts.rigid_contact_count.numpy()[0])
        weight = patches[0].view.weight.numpy()[:count]
        anchor_sets.add(tuple(np.flatnonzero(weight > 0.0).tolist()))
        owners = patches[0].current.owner.numpy()[:count]
        region_counts.add(int(np.unique(owners[owners >= 0]).size))
    test.assertGreater(len(anchor_sets), 1, "the anchors never changed between replays")
    test.assertGreater(len(region_counts), 1, "the number of regions never changed between replays")
    test.assertTrue(np.all(np.isfinite(captured.state_0.joint_q.numpy())))


@unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
def test_seed_sides_matches_host_reference(test, device):
    """Apply the gathered seeds to ``delta`` side by side, each side its own block, as a NumPy walk does bit for bit.

    Two scenes with Steel's warm start, :data:`SETTLE_STEPS` steps, then a step that builds and gathers but sweeps
    nothing (``iterations = 0``), so the impulses are the seeds and ``delta`` the seeds applied (``scene_seed_sides``):

    - four lying humanoids 0.4 m apart, landed on the ground: articulated rows in factor coordinates, with two-sided
      units between the humanoids (their contacts move too fast to stay matched, so their rows start cold);
    - a small heap of free boxes at rest (``_heap(nx=3, ny=2, nz=2)``): seeded two-sided rows between two free bodies,
      two six-entry physical blocks each (free prefix 12).

    ``delta`` equals the NumPy float32 walk of :func:`_seed_reference` bit for bit (the kernel compiles without FMA
    contraction, ``fuse_fp=False``, so each ``resp * lambda`` rounds before its sum, as in NumPy; compiled with it,
    11 of the humanoids' 108 entries differ by up to 5 ulp and 59 of the heap's 72, which equal a fused-multiply-add
    walk instead): every one-sided row's entries lie in its side's block, and each
    entry of a seeded row is applied once, by the side whose block holds it. Seeded one-sided rows occur in both
    scenes, seeded two-sided rows in the heap. A cold step on a new contact buffer (contacts with no relation to the
    solved ones) seeds nothing: every impulse and every ``delta`` entry stays zero.
    """
    scenes = {
        "humanoids": lambda: _humanoids(device, per_world=4, spacing=0.4),
        "heap": lambda: _heap(device, nx=3, ny=2, nz=2),
    }
    for name, build in scenes.items():
        with test.subTest(scene=name):
            run = _WarmRun(build(), CROWD_ROWS, {})
            for _ in range(SETTLE_STEPS):
                run.step()
            solver = run.solver
            solver._iterations = 0
            run.step()
            test.assertEqual(int(solver._scene_units.has_two_sided.numpy()[0]), 1)
            units = solver._scene_units
            test.assertGreater(int(units.incidence_start.numpy()[units.side_count]), 0)
            rows = int(solver._constraint_count.numpy()[0])
            impulses = solver._impulses.numpy()[0]
            seeded = np.flatnonzero(impulses[:rows])
            test.assertFalse(np.any(impulses[rows:]))
            reference, visited = _seed_reference(test, solver)
            delta = solver._sparse_factor_velocity_delta.numpy()[0]
            differ = np.flatnonzero(delta != reference)
            test.assertTrue(np.array_equal(delta, reference), f"delta differs at {differ[:8]}")
            dofs = solver._sparse_row_dof.numpy()[0]
            for row in seeded:
                for k in np.flatnonzero(dofs[row] >= 0):
                    test.assertIn((int(row), int(k)), visited, f"seeded row {row} entry {k} was not applied")
            begin, end = solver._scene.row_ranges.numpy()[RANGE_TWO_SIDED]
            test.assertTrue(np.any(seeded < begin), "no one-sided row was seeded")
            if name == "heap":
                test.assertTrue(np.any((seeded >= begin) & (seeded < end)), "no two-sided row was seeded")
            test.assertTrue(np.any(delta))
            # A contact buffer the solver never solved: no relation to the history, so the step starts cold.
            cold = run.pipeline.contacts()
            run.pipeline.collide(run.state_0, cold)
            solver.step(run.state_0, run.state_1, run.control, cold, DT)
            test.assertGreater(int(cold.rigid_contact_count.numpy()[0]), 0)
            test.assertFalse(np.any(solver._impulses.numpy()))
            test.assertFalse(np.any(solver._sparse_factor_velocity_delta.numpy()))


@unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
def test_sweep_and_seed_share_the_response_rule(test, device):
    """Tie the seed's response rule to the sweep's: after a position solve, ``delta`` is ``resp_k * lambda`` summed
    over every row's final impulse.

    The rule exists twice: ``scene_warmstart._scene_entry_response``, which ``scene_seed_sides`` calls, and the
    sweeps' CUDA text (``_thread_loop`` and ``_group_loop``: ``value += (k < prefix ? rows_free_response :
    rows_factor) * change``), which the tripwire pins and so cannot call a ``wp.func``. The
    seed applies ``resp_k * lambda`` to a zeroed ``delta`` and each sweep visit adds ``resp_k * (new - old)``, so after
    the position solve ``delta`` is ``resp_k * lambda`` over the final impulses, which :func:`_seed_reference` (the
    seed's walk in NumPy, a third copy) recomputes: the two agree within :data:`SWEEP_RULE_RTOL` of the largest entry,
    so neither copy can drift from the other unseen. Scenes without late rows (a velocity-limit or spin row applies
    its new impulse, not the change) and without velocity iterations: the landed humanoids 0.4 m apart (factor
    coordinates, joint limit rows, two-sided articulated units) and the heap of free boxes (two-sided free units, free
    prefix 12), warm-started.
    """
    scenes = {
        "humanoids": lambda: _humanoids(device, per_world=4, spacing=0.4),
        "heap": lambda: _heap(device, nx=3, ny=2, nz=2),
    }
    for name, build in scenes.items():
        with test.subTest(scene=name):
            run = _WarmRun(build(), CROWD_ROWS, {})
            for _ in range(SETTLE_STEPS):
                run.step()
            solver = run.solver
            test.assertEqual(solver._scene.velocity_iterations, 0)
            ranges = solver._scene.row_ranges.numpy()
            test.assertEqual(int(ranges[RANGE_LATE][0]), int(ranges[RANGE_LATE][1]), "late rows")
            test.assertGreater(int(ranges[RANGE_TWO_SIDED][1]), int(ranges[RANGE_TWO_SIDED][0]), "no two-sided row")
            reference, _visited = _seed_reference(test, solver)
            delta = solver._sparse_factor_velocity_delta.numpy()[0]
            scale = float(np.max(np.abs(reference)))
            test.assertGreater(scale, 0.0)
            np.testing.assert_allclose(delta, reference, rtol=0.0, atol=SWEEP_RULE_RTOL * scale)


def test_seed_sides_applies_the_response_rule(test, device):
    """Apply seeded rows to ``delta`` side by side as :func:`_seed_walk` does bit for bit, on rows built by hand.

    The kernel runs alone (no solver): three sides with blocks of 6, 3 and 4 coordinates; one one-sided row each (a
    free-body row whose six entries all take the free response, a factor row with a zero impulse, a row whose free
    prefix covers two of its four entries); and three two-sided units listed through a permuted ``unit_order``: two
    rows between sides 0 and 2, a unit dropped past the row capacity (``unit_row_start < 0``) between sides 1 and 2,
    whose rows hold impulses but are not visited, and one row between sides 1 and 0. With the sweeps' free-response
    build constant on, each entry takes ``rows_free_response`` below its row's free prefix and ``rows_factor`` past
    it; with it off, every entry takes its factor value. Both equal the NumPy walk that the solver-level response-rule
    guard (:func:`test_sweep_and_seed_share_the_response_rule`) compares the sweep with.
    """
    entries = 12
    dofs = np.full((8, entries), -1, np.int32)
    dofs[0, :6] = np.arange(0, 6)  # side 0's one-sided free-body row
    dofs[1, :3] = np.arange(6, 9)  # side 1's one-sided row, zero impulse
    dofs[2, :4] = np.arange(9, 13)  # side 2's one-sided row
    for row in (3, 4):  # unit slot 0: sides 0 and 2
        dofs[row, :10] = np.concatenate((np.arange(0, 6), np.arange(9, 13)))
    dofs[5, :7] = np.concatenate((np.arange(6, 9), np.arange(9, 13)))  # unit slot 1, dropped: sides 1 and 2
    dofs[6, :9] = np.concatenate((np.arange(6, 9), np.arange(0, 6)))  # unit slot 2: sides 1 and 0
    rng = np.random.default_rng(7)
    factor = rng.uniform(-2.0, 2.0, (8, entries)).astype(np.float32)
    free = rng.uniform(-2.0, 2.0, (8, entries)).astype(np.float32)
    impulses = rng.uniform(0.05, 0.5, 8).astype(np.float32)
    impulses[1] = 0.0
    impulses[7] = 0.0  # past every range
    rows = {
        "side_count": 3,
        "side_row_begin": np.array([0, 1, 2], np.int32),
        "side_row_end": np.array([1, 2, 3], np.int32),
        # Sorted positions 0, 1, 2 hold slots 2, 0, 1.
        "unit_order": np.array([2, 0, 1], np.int32),
        "incidence_start": np.array([0, 2, 4, 6], np.int32),
        "incidence_unit": np.array([1, 0, 2, 0, 1, 2], np.int32),
        "unit_row_start": np.array([3, -1, 6], np.int32),
        "unit_rows": np.array([2, 2, 1], np.int32),
        "offset": np.array([0, 6, 9], np.int32),
        "block": np.array([6, 3, 4], np.int32),
        "dofs": dofs,
        "factor": factor,
        "free": free,
        "impulses": impulses,
        "size": 13,
    }
    row_free_prefix = np.array([6, 0, 2, 6, 6, 3, 3, 0], np.int32)
    for free_response in (1, 0):
        with test.subTest(free_response=free_response):
            prefix = row_free_prefix if free_response else np.zeros_like(row_free_prefix)
            expected, visited = _seed_walk(test, prefix=prefix, **rows)
            test.assertNotIn(5, {row for row, _k in visited})
            test.assertEqual(len(visited), 6 + 3 + 4 + 2 * 10 + 9)
            delta = wp.zeros((1, rows["size"]), dtype=float, device=device)
            wp.launch(
                scene_seed_sides,
                dim=rows["side_count"],
                inputs=[
                    rows["side_count"],
                    free_response,
                    wp.array(rows["side_row_begin"], dtype=int, device=device),
                    wp.array(rows["side_row_end"], dtype=int, device=device),
                    wp.array(rows["incidence_start"], dtype=int, device=device),
                    wp.array(rows["incidence_unit"], dtype=int, device=device),
                    wp.array(rows["unit_order"], dtype=int, device=device),
                    wp.array(rows["unit_row_start"], dtype=int, device=device),
                    wp.array(rows["unit_rows"], dtype=int, device=device),
                    wp.array(rows["offset"], dtype=int, device=device),
                    wp.array(rows["block"], dtype=int, device=device),
                    wp.array(dofs[None], dtype=int, device=device),
                    wp.array(factor[None], dtype=float, device=device),
                    wp.array(free[None], dtype=float, device=device),
                    wp.array(row_free_prefix, dtype=int, device=device),
                    wp.array(impulses[None], dtype=float, device=device),
                ],
                outputs=[delta],
                device=device,
            )
            got = delta.numpy()[0]
            test.assertTrue(np.all(got != 0.0))
            test.assertTrue(np.array_equal(got, expected), f"delta differs at {np.flatnonzero(got != expected)}")


@wp.kernel
def _reset_selected_worlds(
    world_mask: wp.array[wp.bool], world_count: int, global_shares: int, selected: wp.array[int]
):
    world = wp.tid()
    selected[world] = wp.where(_reset_selects_world(world_mask, world, world_count, global_shares), 1, 0)


def test_reset_selection_is_steels(test, device):
    """Select the worlds of a reset mask as Steel's ``reset_row_warmstart`` does, and clear the rows only when every
    world is selected.

    The scene states Steel's rule (``scene_warmstart._reset_selects_world``) beside Steel's kernel, which keeps it
    inline. For one and three worlds, every ``(world_count + 1,)`` mask, with and without a responding global
    articulation sharing world 0's rows: the worlds the scene's rule selects are those whose carried rows Steel's
    kernel clears, and ``scene_reset_world_selection`` and ``scene_reset_row_warmstart`` clear the scene's rows exactly
    when Steel's clears every world's.
    """
    for world_count in (1, 3):
        for global_shares in (0, 1):
            for bits in itertools.product((False, True), repeat=world_count + 1):
                with test.subTest(world_count=world_count, global_shares=global_shares, mask=bits):
                    mask = wp.array(bits, dtype=wp.bool, device=device)
                    carried = [
                        wp.full((world_count, 2), value, dtype=dtype, device=device)
                        for value, dtype in ((1.0, float), (3, int), (0, int))
                    ]
                    wp.launch(
                        reset_row_warmstart,
                        dim=(world_count, 2),
                        inputs=[mask, world_count, global_shares, *carried],
                        device=device,
                    )
                    steel = carried[0].numpy()[:, 0] == 0.0
                    selected = wp.zeros(world_count, dtype=int, device=device)
                    wp.launch(
                        _reset_selected_worlds,
                        dim=world_count,
                        inputs=[mask, world_count, global_shares, selected],
                        device=device,
                    )
                    np.testing.assert_array_equal(selected.numpy() == 1, steel)
                    all_selected = wp.full(1, 1, dtype=int, device=device)
                    wp.launch(
                        scene_reset_world_selection,
                        dim=world_count,
                        inputs=[mask, world_count, global_shares],
                        outputs=[all_selected],
                        device=device,
                    )
                    rows = [
                        wp.full((1, 4), value, dtype=dtype, device=device)
                        for value, dtype in ((1.0, float), (3, int), (0, int))
                    ]
                    wp.launch(scene_reset_row_warmstart, dim=(1, 4), inputs=[all_selected], outputs=rows, device=device)
                    test.assertEqual(int(all_selected.numpy()[0]), int(np.all(steel)))
                    cleared = rows[0].numpy()[0] == 0.0
                    test.assertTrue(np.all(cleared) if np.all(steel) else not np.any(cleared))
                    test.assertTrue(np.all(rows[1].numpy() == (-1 if np.all(steel) else 3)))


def _check_gather_mu(test, solver, count) -> int:
    """Check ``SceneSolve.gather_mu`` after a step under friction patches with free bodies; returns the anchors on
    Steel's free-body path whose rows' coefficient the patch link divided.

    Each kept contact with its tangent pair holds there the coefficient Steel's gather reads: on Steel's dense path (a
    responding articulated side) its rows' own, divided by the link; on Steel's free-body path the one before the
    link, its normal row's, as Steel gathers those rows before linking them.
    """
    mu, gather = solver._row_mu.numpy()[0], solver._scene.gather_mu.numpy()[0]
    slot, needed = solver._contact_slot.numpy()[:count], solver._contact_slots_needed.numpy()[:count]
    art_a, art_b = solver._contact_art_a.numpy()[:count], solver._contact_art_b.numpy()[:count]
    dofs, free = solver._articulation_response_dof_count.numpy(), solver._is_free_rigid.numpy()
    divided = 0
    for c in range(count):
        row = int(slot[c])
        if row < 0 or needed[c] != 3:
            continue
        if any(art >= 0 and dofs[art] > 0 and free[art] == 0 for art in (art_a[c], art_b[c])):
            np.testing.assert_array_equal(gather[row + 1 : row + 3], mu[row + 1 : row + 3], f"contact {c}")
        else:
            np.testing.assert_array_equal(gather[row + 1 : row + 3], mu[row], f"contact {c}")
            divided += int(np.any(mu[row + 1 : row + 3] != mu[row]))
    return divided


def _compare_seeds(test, model, scene, steel, step, diverged, totals, *, patches):
    """Compare the last step's seeds of the scene with Steel's, contact for contact, within ``DRIVE_TRACK_RTOL`` and
    :data:`SEED_ATOL`, adding to ``totals``.

    Under ``patches`` a region (a body pair) whose anchors differ from Steel's at a step, a resting box's two tied
    diagonals picked apart by states one rounding apart, joins ``diverged`` and is no longer compared:
    its impulses then follow other anchors; ``gather_mu`` is checked (:func:`_check_gather_mu`).
    """
    count = int(scene.contacts.rigid_contact_count.numpy()[0])
    test.assertEqual(count, int(steel.contacts.rigid_contact_count.numpy()[0]), f"step {step}")
    ours, theirs = _contact_seeds(scene.solver, count), _contact_seeds(steel.solver, count)
    test.assertEqual(sorted(ours), sorted(theirs), f"step {step}")
    shape_body = model.shape_body.numpy()
    bodies = zip(
        shape_body[scene.contacts.rigid_contact_shape0.numpy()[:count]],
        shape_body[scene.contacts.rigid_contact_shape1.numpy()[:count]],
        strict=True,
    )
    regions = [tuple(sorted((int(a), int(b)))) for a, b in bodies]
    if patches:
        diverged.update(regions[c] for c, values in ours.items() if np.isnan(values[1]) != np.isnan(theirs[c][1]))
        totals["divided"] += _check_gather_mu(test, scene.solver, count)
    for c, values in ours.items():
        if regions[c] in diverged:
            continue
        np.testing.assert_allclose(
            values, theirs[c], rtol=DRIVE_TRACK_RTOL, atol=SEED_ATOL, err_msg=f"step {step}, contact {c}"
        )
        totals["seeded"] += int(np.count_nonzero(np.nan_to_num(values)))
        totals["tangents"] += int(np.count_nonzero(np.nan_to_num(values[1:])))
        totals["compared"] += 1


@unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
def test_warmstart_tracks_steel(test, device):
    """Step Steel's contact warm start as Steel's matrix-free solve does: one step and thirty, and the seeds.

    The contact scene (two worlds of a ground, two free boxes and a D6 box with a driven arm), every box sliding at
    :data:`SLIDE_SPEED`, with Steel's warm start under point friction, under Steel's default patches
    (``seed_patch_impulses`` joins), under the friction mode ``"bisection"`` and with :data:`VELOCITY_ITERATIONS`
    velocity iterations (which continue from the seeded position solve), each a combination Steel allows. The
    fixture's boxes never meet each other or the articulation: they are 19 cm apart, within the collision gap, so the
    pipeline reports speculative contacts between the two boxes of each world (two-sided free-body units, swept in
    another order than Steel's world), and their rows carry no impulse after any step. The velocities after the first
    step agree with Steel's within 1e-5 and ``test_scene_friction_modes.FREE_MODE_ATOL`` on the free boxes, within
    ``test_scene_drives.DRIVE_TRACK_RTOL`` and 1e-6 on the articulation (Steel's warm start takes its dense rows), the
    joint coordinates after :data:`TRACK_STEPS` steps within ``test_scene_friction_modes.MODE_TRACK_Q_RMS``. The seeds:

    - the law (:func:`_check_gather`): every gather of the scene equals Steel's ``gather_contact_warmstart`` on the
      scene's own snapshot bit for bit, and its normal rows ``decay * (dt / dt_previous) * lambda`` in NumPy;
    - the values: each kept contact's seeds of every step (after the patch seed) agree with Steel's seeds of the same
      contact within ``DRIVE_TRACK_RTOL`` and :data:`SEED_ATOL` (the previous step's impulses already differ by
      rounding), its row count Steel's, and some contact is seeded. Under patches a free box's anchor without patch
      history keeps its gathered tangent, which Steel clamps with the coefficient before the patch link (its free-body
      gather runs before ``link_patch_rows``): the scene's ``gather_mu`` gives its gather the same coefficient.

    Under patches a resting box's anchors are one of its two diagonals, which tie: states one rounding apart pick
    either from the third step on, so a region whose anchors differ from Steel's is compared up to
    that step (:func:`_compare_seeds`), ``gather_mu`` holds Steel's coefficients for every anchor and differs from the
    rows' on some free-body anchor (:func:`_check_gather_mu`), and the coordinates after thirty steps agree within
    ``test_scene_features.TRACK_RMS`` (1.0e-5 measured, past ``MODE_TRACK_Q_RMS``, the other cases' bound).
    """
    from newton.tests.steel.test_scene_friction_modes import FREE_MODE_ATOL, MODE_TRACK_Q_RMS  # noqa: PLC0415

    cases = {
        "point friction": {},
        "patches": PATCHES,
        "bisection": {"friction_mode": "bisection"},
        "velocity iterations": {"velocity_iterations": VELOCITY_ITERATIONS},
    }
    for name, options in cases.items():
        with test.subTest(case=name):
            model = _build_contact_scene(device)
            free = np.zeros(model.joint_dof_count, dtype=bool)
            for start in _free_body_dofs(model):
                free[start : start + 6] = True
            scene = _WarmRun(model, TRACK_ROWS, {}, **options)
            steel = _WarmRun(model, TRACK_ROWS, {}, solve_mode="serial", **options)
            test.assertEqual(scene.solver.solve_mode, "scene")
            test.assertIsNone(steel.solver._scene)
            test.assertEqual(scene.solver._scene.gather_mu is not None, options is PATCHES)
            _record_scene_seeds(scene.solver)
            _record_steel_seeds(steel.solver)
            gathers = _check_gather(test, scene.solver)
            for run in (scene, steel):
                _set_box_velocities(model, run.state_0, speed=SLIDE_SPEED)
            totals = {"seeded": 0, "tangents": 0, "compared": 0, "divided": 0}
            diverged = set()
            for step in range(1, TRACK_STEPS + 1):
                scene.step()
                steel.step()
                begin, end = scene.solver._scene.row_ranges.numpy()[RANGE_TWO_SIDED]
                test.assertFalse(np.any(scene.solver._impulses.numpy()[0, begin:end]), f"boxes met at step {step}")
                _compare_seeds(test, model, scene, steel, step, diverged, totals, patches=options is PATCHES)
                if step == 1:
                    qd_scene, qd_steel = scene.state_0.joint_qd.numpy(), steel.state_0.joint_qd.numpy()
                    test.assertGreater(np.max(np.abs(qd_steel)), 0.0)
                    np.testing.assert_allclose(qd_scene[free], qd_steel[free], rtol=TRACK_RTOL, atol=FREE_MODE_ATOL)
                    np.testing.assert_allclose(qd_scene[~free], qd_steel[~free], rtol=DRIVE_TRACK_RTOL, atol=TRACK_ATOL)
            test.assertEqual(gathers["gathers"], TRACK_STEPS)
            test.assertGreater(gathers["seeded"], 0)
            test.assertGreater(totals["seeded"], 0)
            test.assertGreater(totals["tangents"], 0)
            if options is PATCHES:
                test.assertGreater(totals["divided"], 0)
            q_scene, q_steel = scene.state_0.joint_q.numpy(), steel.state_0.joint_q.numpy()
            test.assertTrue(np.all(np.isfinite(q_scene)))
            test.assertLess(_rms(q_scene, q_steel), TRACK_RMS if options is PATCHES else MODE_TRACK_Q_RMS)
            for run in (scene, steel):
                test.assertFalse(np.any(run.solver.constraint_overflow.numpy()))


def _two_sided_seeded(solver) -> bool:
    """Whether the last step seeded a row of a two-sided unit (``solver.seeds``, :class:`_SeedRun`)."""
    begin, end = solver._scene.row_ranges.numpy()[RANGE_TWO_SIDED]
    return bool(np.any(solver.seeds[0][0, begin:end]))


@unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
def test_warmstart_matches_the_serial_oracle(test, device):
    """Sweep from the seeds in the phased sweep exactly as one thread walking the rows does, in every order.

    A small heap of free boxes at rest (``_heap(nx=3, ny=2, nz=2)``: box-on-box contacts, two-sided units in phase B)
    with Steel's warm start: each step starts from its seeds, the gathered impulses and their ``delta``, and the phases
    (the default color loop, and the while loop) match the serial oracle (``scene_sweep="serial"``) bit for bit over 30
    steps in each phase order, so each step also seeds both from the same impulses. Some step seeds a two-sided unit's
    row (from the sixth step on, when the resting contacts stay matched).
    """
    model = _heap(device, nx=3, ny=2, nz=2)
    _assert_phases_match_serial_oracle(
        test, model, {}, witness=_two_sided_seeded, what="no two-sided row was seeded", runner=_SeedRun
    )


@unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
def test_warmstart_replays_under_capture(test, device):
    """Replay one captured warm-started step exactly as eager steps, as contacts come and go, and keep its history.

    Four lying humanoids 0.4 m apart with a free box inside the first, every body sliding at 1.5 m/s, with Steel's warm
    start: the collision pass, the gather, the seed and the snapshot are in the graph. After each of
    :data:`CAPTURE_STEPS` steps ``joint_q``, ``joint_qd``, the impulses, ``delta`` and the carried history (the rows'
    impulses, types and parents, each contact's first row and normal, the step, generation and stream) equal the eager
    steps' bit for bit; the contact count changes between replays, and the eager steps seed contacts.

    Then, as Steel's ``test_interleaved_contact_buffers_start_cold_under_graph_replay``: on the contact scene, three
    graphs (a collision pass into the solved buffer, one into another buffer, and a step with no sweep) replay. A pass
    straight into the solved buffer then the step carries the impulses (some seed); a pass into the other buffer before
    it starts the step cold (no seed), the generations of the device history deciding.
    """
    model = _humanoids(device, per_world=4, spacing=0.4, props=True)
    eager, captured = runs = [_warm_fixed_run(model, CROWD_ROWS, {}) for _ in range(2)]
    _record_scene_seeds(eager.solver)
    for run in runs:
        _set_box_velocities(model, run.state_0)
        run.step()
    with wp.ScopedCapture(device) as recorded:
        captured.step()
    history = (
        "_impulses",
        "_sparse_factor_velocity_delta",
        "_ws_prev_impulses",
        "_ws_prev_row_type",
        "_ws_prev_row_parent",
        "_ws_prev_dense_slot",
        "_ws_prev_contact_normal",
        "_ws_history_dt",
        "_ws_history_generation",
        "_ws_history_stream",
    )
    counts, seeded = set(), 0
    for step in range(2, CAPTURE_STEPS + 1):
        eager.step()
        wp.capture_launch(recorded.graph)
        for name in ("joint_q", "joint_qd"):
            a, b = getattr(eager.state_0, name).numpy(), getattr(captured.state_0, name).numpy()
            test.assertTrue(np.array_equal(a, b), f"{name} differs after step {step}")
        for name in history:
            a, b = getattr(eager.solver, name).numpy(), getattr(captured.solver, name).numpy()
            test.assertTrue(np.array_equal(a, b), f"{name} differs after step {step}")
        counts.add(int(eager.contacts.rigid_contact_count.numpy()[0]))
        seeded += int(np.count_nonzero(eager.solver.seeds[0]))
    test.assertGreater(len(counts), 1, "the contact count never changed between replays")
    test.assertGreater(seeded, 0)
    test.assertTrue(np.all(np.isfinite(captured.state_0.joint_q.numpy())))

    model = _build_contact_scene(device)
    run = _WarmRun(model, TRACK_ROWS, {})
    for _ in range(SETTLE_STEPS):
        run.step()
    solver, contacts, other = run.solver, run.contacts, run.pipeline.contacts()
    solver._iterations = 0
    graphs = {}
    for name, buffer in (("collide", contacts), ("collide_other", other)):
        with wp.ScopedCapture(device) as capture:
            run.pipeline.collide(run.state_0, buffer)
        graphs[name] = capture.graph
    with wp.ScopedCapture(device) as capture:
        solver.step(run.state_0, run.state_1, run.control, contacts, DT)
    graphs["step"] = capture.graph
    for sequence, warm in ((("collide", "step"), True), (("collide_other", "collide", "step"), False)):
        for name in sequence:
            wp.capture_launch(graphs[name])
        impulses = solver._impulses.numpy()
        test.assertEqual(bool(np.any(impulses)), warm, sequence)


@unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
def test_warmstart_carries_into_the_velocity_pass(test, device):
    """Start the velocity pass from the position solve's impulses and ``delta``, the seeds included, as Steel does.

    Four lying humanoids 0.5 m apart land with Steel's warm start and :data:`VELOCITY_ITERATIONS` velocity iterations;
    the next step sweeps no position iteration (``iterations = 0``), so the position solve ends at its seeds. When the
    velocity pass starts (``SceneSolve.prepare_velocity_rows``) the impulses are the seeds (some nonzero) and ``delta``
    is their application (:func:`_seed_reference`) bit for bit: nothing resets them between the passes. The velocity
    pass then moves the impulses on from there.
    """
    model = _humanoids(device, per_world=4, spacing=0.5)
    run = _WarmRun(model, CROWD_ROWS, {}, velocity_iterations=VELOCITY_ITERATIONS)
    for _ in range(SETTLE_STEPS):
        run.step()
    solver = run.solver
    prepare = solver._scene.prepare_velocity_rows
    entry = {}

    def recorded(solver, dt):
        # Copies: on the CPU numpy() is a view of the solver's arrays, which the velocity pass overwrites.
        entry["impulses"] = solver._impulses.numpy()[0].copy()
        entry["delta"] = solver._sparse_factor_velocity_delta.numpy()[0].copy()
        prepare(solver, dt)

    solver._scene.prepare_velocity_rows = recorded
    solver._iterations = 0
    run.step()
    test.assertTrue(np.any(entry["impulses"]))
    reference, _visited = _seed_reference(test, solver, impulses=entry["impulses"])
    test.assertTrue(np.any(reference))
    test.assertTrue(np.array_equal(entry["delta"], reference))
    test.assertFalse(np.array_equal(solver._impulses.numpy()[0], entry["impulses"]), "the velocity pass did not act")


@unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
def test_warmstart_rejections_and_cold_start(test, device):
    """Start cold exactly as without the warm start, clear the whole impulse capacity, and refuse unmatched contacts.

    - Four lying humanoids 0.4 m apart with a free box inside the first: with no contact matched (the first step, then
      a step on a buffer the solver never solved) the warm-started scene steps exactly as the cold one, ``joint_q``,
      ``joint_qd`` and the kept rows' impulses bit for bit.
    - With the warm start every impulse past the kept rows is zero after a step, as Steel's
      ``prepare_world_impulses(warmstart=1)`` clears the whole capacity, and so is its snapshot; a cold step clears the
      kept rows only.
    - Contacts without matching raise Steel's ``ValueError`` at the step, under the scene too.
    """
    model = _humanoids(device, per_world=4, spacing=0.4, props=True)
    warm = _WarmRun(model, CROWD_ROWS, {})
    cold = _WarmRun(model, CROWD_ROWS, {}, contact_warmstart=False)

    def assert_equal(label):
        rows = int(warm.solver._constraint_count.numpy()[0])
        test.assertEqual(rows, int(cold.solver._constraint_count.numpy()[0]), label)
        for name in ("joint_q", "joint_qd"):
            a, b = getattr(warm.state_0, name).numpy(), getattr(cold.state_0, name).numpy()
            test.assertTrue(np.array_equal(a, b), f"{name} differs: {label}")
        a, b = warm.solver._impulses.numpy()[0, :rows], cold.solver._impulses.numpy()[0, :rows]
        test.assertTrue(np.array_equal(a, b), f"impulses differ: {label}")

    for run in (warm, cold):
        run.step()
    assert_equal("first step")
    fresh = warm.pipeline.contacts()
    warm.contacts = fresh
    for run in (warm, cold):
        run.step()
    assert_equal("a buffer the solver never solved")

    sentinel = 7.0
    for run in (warm, cold):
        run.solver._impulses.fill_(sentinel)
        run.step()
    rows = int(warm.solver._constraint_count.numpy()[0])
    test.assertLess(rows, warm.solver._scene.row_max)
    test.assertFalse(np.any(warm.solver._impulses.numpy()[0, rows:]))
    test.assertFalse(np.any(warm.solver._ws_prev_impulses.numpy()[0, rows:]))
    cold_rows = int(cold.solver._constraint_count.numpy()[0])
    test.assertTrue(np.all(cold.solver._impulses.numpy()[0, cold_rows:] == sentinel))

    unmatched = newton.CollisionPipeline(model, deterministic=True)
    contacts = unmatched.contacts()
    unmatched.collide(warm.state_0, contacts)
    with test.assertRaisesRegex(ValueError, "contact matching"):
        warm.solver.step(warm.state_0, warm.state_1, warm.control, contacts, DT)


def _contact_worlds(model, contacts) -> np.ndarray:
    """The model world of each of the step's contacts: its shapes' world (a ground shape is global)."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape_world = model.shape_world.numpy()
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    return np.maximum(shape_world[shape0], shape_world[shape1])


@unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
def test_warmstart_resets_by_world_mask(test, device):
    """Discard the carried impulses of the masked worlds alone, contact by contact, as Steel's per-world reset does.

    The contact scene in three worlds, warm-started by :data:`SETTLE_STEPS` steps, in the scene and in Steel. The scene
    folds the worlds into one row family, so its reset clears each contact's carried first row whose model world is
    masked (``_ws_prev_dense_slot`` to -1), and the rows only when every world is: after a reset of world 1 the carried
    rows stay, and the next step (no sweep) seeds no contact of world 1 in either solver and the contacts of worlds 0
    and 2 as Steel seeds them (within ``DRIVE_TRACK_RTOL`` and :data:`SEED_ATOL`, some seeded); a reset of every world
    clears the rows too, and the next step seeds nothing. The velocities after the step without a sweep (``v_hat``
    plus the seeds: the scene's decode of the seeded ``delta`` against Steel's install) agree with Steel's within the
    tracking tolerances (``1e-5`` and ``test_scene_friction_modes.FREE_MODE_ATOL`` on the free boxes,
    ``test_scene_drives.DRIVE_TRACK_RTOL`` and ``TRACK_ATOL`` on the articulations). The reset is captured with a live
    mask and replayed with world 0 then world 2 selected.
    """
    from newton.tests.steel.test_scene_friction_modes import FREE_MODE_ATOL  # noqa: PLC0415

    model = _build_contact_scene(device, worlds=3)
    runs = {"scene": _WarmRun(model, TRACK_ROWS, {}), "steel": _WarmRun(model, TRACK_ROWS, {}, solve_mode="serial")}
    for run in runs.values():
        _set_box_velocities(model, run.state_0, speed=SLIDE_SPEED)
        for _ in range(SETTLE_STEPS):
            run.step()
    scene = runs["scene"].solver
    worlds = _contact_worlds(model, runs["scene"].contacts)
    test.assertEqual(set(worlds.tolist()), {0, 1, 2})
    carried = scene._ws_prev_impulses.numpy()
    slots = scene._ws_prev_dense_slot.numpy()
    mask = wp.array([False, True, False, False], dtype=wp.bool, device=device)
    for run in runs.values():
        run.solver.reset(run.state_0, mask)
    count = worlds.size
    after = scene._ws_prev_dense_slot.numpy()
    np.testing.assert_array_equal(after[:count][worlds == 1], -1)
    np.testing.assert_array_equal(after[:count][worlds != 1], slots[:count][worlds != 1])
    test.assertTrue(np.array_equal(scene._ws_prev_impulses.numpy(), carried), "a partial reset cleared the rows")
    seeds = {}
    for name, run in runs.items():
        run.solver._iterations = 0
        if name == "steel":
            _record_steel_seeds(run.solver)
        else:
            _record_scene_seeds(run.solver)
        run.step()
        seeds[name] = _contact_seeds(run.solver, int(run.contacts.rigid_contact_count.numpy()[0]))
    worlds = _contact_worlds(model, runs["scene"].contacts)
    test.assertEqual(worlds.size, int(runs["steel"].contacts.rigid_contact_count.numpy()[0]))
    test.assertEqual(sorted(seeds["scene"]), sorted(seeds["steel"]))
    for name, values in seeds.items():
        test.assertFalse(any(np.any(np.nan_to_num(values[c])) for c in values if worlds[c] == 1), name)
        test.assertTrue(any(np.any(np.nan_to_num(values[c])) for c in values if worlds[c] != 1), name)
    for c, values in seeds["scene"].items():
        np.testing.assert_allclose(
            values, seeds["steel"][c], rtol=DRIVE_TRACK_RTOL, atol=SEED_ATOL, err_msg=f"contact {c}"
        )
    free = np.zeros(model.joint_dof_count, dtype=bool)
    for start in _free_body_dofs(model):
        free[start : start + 6] = True
    qd_scene, qd_steel = runs["scene"].state_0.joint_qd.numpy(), runs["steel"].state_0.joint_qd.numpy()
    test.assertGreater(np.max(np.abs(qd_steel)), 0.0)
    np.testing.assert_allclose(qd_scene[free], qd_steel[free], rtol=TRACK_RTOL, atol=FREE_MODE_ATOL)
    np.testing.assert_allclose(qd_scene[~free], qd_steel[~free], rtol=DRIVE_TRACK_RTOL, atol=TRACK_ATOL)

    with wp.ScopedCapture(device) as capture:
        scene.reset(runs["scene"].state_0, mask)
    for selected in (0, 2):
        runs["scene"].step()
        worlds = _contact_worlds(model, runs["scene"].contacts)
        slots = scene._ws_prev_dense_slot.numpy()[: worlds.size]
        values = np.zeros(4, dtype=bool)
        values[selected] = True
        mask.assign(values)
        wp.capture_launch(capture.graph)
        after = scene._ws_prev_dense_slot.numpy()[: worlds.size]
        np.testing.assert_array_equal(after[worlds == selected], -1)
        np.testing.assert_array_equal(after[worlds != selected], slots[worlds != selected])
        test.assertTrue(np.any(after[worlds != selected] >= 0))

    mask.assign(np.array([True, True, True, False]))
    scene.reset(runs["scene"].state_0, mask)
    test.assertFalse(np.any(scene._ws_prev_impulses.numpy()))
    test.assertTrue(np.all(scene._ws_prev_row_type.numpy() == -1))
    test.assertTrue(np.all(scene._ws_prev_dense_slot.numpy() == -1))
    runs["scene"].step()
    test.assertFalse(np.any(scene.seeds[0]))


# --- Sleeping ---------------------------------------------------------------------------------------------------------

SLEEP = {"enable_sleeping": True}
"""Steel's experimental sleeping at its default thresholds (0.05 m/s, 0.15 rad/s and a quiet time of 0.5 s) with its
rows skipped (``sleep_skip_constraints``), over the :data:`~newton.tests.steel._scene_fixtures.SCENE` options."""

SLEEP_STEPS = 300
"""Steps at ``DT`` within which every box of a sleeping test sleeps, in the scene and in Steel."""

HEAP_ITERATIONS = 16
"""Iterations of the heap's sleeping tests, in both solvers (``test_sleeping``'s boxes take 16 too). At the
:data:`~newton.tests.steel._scene_fixtures.SCENE` options' 4 the heap never sleeps in either solver, so neither more
steps nor a shorter quiet time helps: Gauss-Seidel leaves its boxes rocking above Steel's angular threshold (0.15 rad/s;
Steel's fastest box turns at 0.26 to 0.45 rad/s at every check over 600 steps). At 16 the scene puts every box to
sleep at step 127 and Steel at step 128."""

HEAP_FREE_ROWS = 8192
"""``free_body_constraint_max`` of the heap's sleeping tests: Steel's free-body rows of every heap contact. Its default
of 512 overflows, and Steel's sleeping never lets a world whose rows overflowed sleep."""

REST_ATOL = 1.0e-3
"""Agreement [m] of resting positions: the scene's heights with Steel's, and a captured run's with its eager steps'."""

DROP_HEIGHT = 0.2
"""Height [m] of the spare box's centre above the ground when it is dropped onto the sleeping heap."""

DROP_REPLAYS = 30
"""Replays after the drop: the spare box falls the 60 mm onto the heap's top layer, which its contacts reach at the
27th, and rests there (40.0 mm above the top box's centre, within 0.1 mm, from the 28th)."""

ANT_REPLAYS = 1200
"""Replays of two steps each within which the undriven ant comes to rest and sleeps (``test_sleeping``'s bound)."""

ANT_JOINT_FRICTION = 0.05
"""Joint friction [N m] of the ant's hinges in :func:`test_sleeping_articulation_gets_no_internal_rows`."""


def _sleep_run(model, *, scene=True, fixed=False, rows=None, **options):
    """A run with :data:`SLEEP`: the scene's (``scene``) or Steel's, stepping its states (``fixed``: one pair of
    buffers, so that one captured step replays every later step)."""
    run = _FixedRun if fixed else _Run
    rows = TRACK_ROWS if rows is None else rows
    return run(model, rows, {}, **SLEEP, solve_mode="scene" if scene else "serial", **options)


def _asleep(run, bodies=slice(None)) -> bool:
    """Whether every one of ``bodies`` sleeps after the run's last step."""
    return not np.any(run.solver._sleeping.body_awake.numpy()[bodies])


def _islands(solver) -> set[frozenset[int]]:
    """The partition of the bodies into islands (``body_island``) as a set of sets: an island's id is its root, which
    two solvers need not share."""
    islands: dict[int, set[int]] = {}
    for body, island in enumerate(solver._sleeping.body_island.numpy().tolist()):
        islands.setdefault(island, set()).add(body)
    return {frozenset(members) for members in islands.values()}


def _body_pairs(model, contacts) -> set[tuple[int, int]]:
    """The pairs of bodies (lower first) that the step's contacts join; a ground contact names body -1."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape_body = model.shape_body.numpy()
    a = shape_body[contacts.rigid_contact_shape0.numpy()[:count]]
    b = shape_body[contacts.rigid_contact_shape1.numpy()[:count]]
    return {(int(min(i, j)), int(max(i, j))) for i, j in zip(a, b, strict=True)}


def _set_free_body(model, state, body, *, position=None, velocity=None):
    """Set a free body's position or linear velocity in its joint coordinates and its body pose or twist alike, leaving
    every other body's state untouched (an FK pass could round the sleeping bodies' poses, which wakes them)."""
    joint = int(np.flatnonzero(model.joint_child.numpy() == body)[0])
    q_start, qd_start = int(model.joint_q_start.numpy()[joint]), int(model.joint_qd_start.numpy()[joint])
    if position is not None:
        joint_q, body_q = state.joint_q.numpy(), state.body_q.numpy()
        joint_q[q_start : q_start + 3] = position
        body_q[body, :3] = position
        state.joint_q.assign(joint_q)
        state.body_q.assign(body_q)
    if velocity is not None:
        joint_qd, body_qd = state.joint_qd.numpy(), state.body_qd.numpy()
        joint_qd[qd_start : qd_start + 3] = velocity
        body_qd[body, :3] = velocity
        state.joint_qd.assign(joint_qd)
        state.body_qd.assign(body_qd)


@unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
def test_sleeping_tracks_steel(test, device):
    """Sleep the heap's islands as Steel does: every box asleep within :data:`SLEEP_STEPS` steps in both solvers, the
    same islands, and each box resting within :data:`REST_ATOL` of Steel's height.

    The 108 boxes of ``_heap`` settle onto the ground with :data:`SLEEP` at :data:`HEAP_ITERATIONS`, in the scene and
    in Steel. The heap does not track Steel step by step (the Gauss-Seidel orders differ),
    so the criteria are the outcome: both report every box asleep after :data:`SLEEP_STEPS` steps, their partitions of
    the boxes into islands (``body_island``) are the same sets, and the boxes' heights agree within :data:`REST_ATOL`.
    """
    model = _heap(device)
    runs = {
        name: _sleep_run(
            model, scene=name == "scene", free_body_constraint_max=HEAP_FREE_ROWS, iterations=HEAP_ITERATIONS
        )
        for name in ("scene", "steel")
    }
    asleep_at = {}
    for step in range(1, SLEEP_STEPS + 1):
        for name, run in runs.items():
            run.step()
            if name not in asleep_at and _asleep(run):
                asleep_at[name] = step
    test.assertEqual(sorted(asleep_at), ["scene", "steel"], f"asleep within {SLEEP_STEPS} steps: {asleep_at}")
    for name, run in runs.items():
        test.assertTrue(_asleep(run), f"{name}: a box woke again by step {SLEEP_STEPS}")
        test.assertFalse(np.any(run.solver.constraint_overflow.numpy()), name)
    test.assertEqual(_islands(runs["scene"].solver), _islands(runs["steel"].solver))
    heights = {name: run.state_0.body_q.numpy()[:, 2] for name, run in runs.items()}
    np.testing.assert_allclose(heights["scene"], heights["steel"], rtol=0.0, atol=REST_ATOL)


@unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
@unittest.skipUnless(hasattr(SolverSteel, "scene_statistics"), "needs the read-back API")
def test_sleeping_island_gets_no_units(test, device):
    """Give a sleeping island's contacts no units: once every box of the heap sleeps, a step has no unit, row or color.

    ``begin`` clears a sleeping body's entry of ``contact_response_mask``, which the scene's contact classification
    reads as ``body_has_response_dofs`` as Steel's allocation does, so a contact without an awake responding side
    gets path -1, the outcome of Steel's allocation, and :class:`SceneUnits` makes no unit of it. The collision
    pipeline still reports the heap's contacts.
    """
    model = _heap(device)
    run = _sleep_run(model, free_body_constraint_max=HEAP_FREE_ROWS, iterations=HEAP_ITERATIONS)
    run.step()
    awake = run.solver.scene_statistics()
    test.assertGreater(awake["two_sided_units"], 0)
    for _ in range(SLEEP_STEPS):
        if _asleep(run):
            break
        run.step()
    test.assertTrue(_asleep(run), f"the heap did not sleep within {SLEEP_STEPS} steps")
    run.step()
    count = int(run.contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    np.testing.assert_array_equal(run.solver._contact_path.numpy()[:count], -1)
    stats = run.solver.scene_statistics()
    test.assertEqual({key: stats[key] for key in ("units", "rows", "colors")}, {"units": 0, "rows": 0, "colors": 0})


@unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
@unittest.skipUnless(hasattr(SolverSteel, "scene_statistics"), "needs the read-back API")
def test_all_asleep_scene_replays_under_capture(test, device):
    """Replay a step captured while the heap was awake through a scene without units, and wake the heap on contact.

    Review Focus 1. ``_heap`` with its spare box beside it at :data:`HEAP_ITERATIONS`, eager and captured (one step of
    fixed buffers each, captured at the second step, when units exist). The replays follow the eager steps, awake flags
    equal, until every box sleeps. A replay then builds no unit, row or color (``scene_statistics``): the graph's
    launches keep their grids, ``SceneUnits.color`` and the sweeps' coupled branch run under ``wp.capture_if`` on
    ``has_two_sided`` (0 here), and nothing is read back; it decodes ``v_hat`` into ``joint_qd`` unchanged (zero) and
    writes no NaN. Then the spare box is moved over the top box of the heap's centre column, its centre
    :data:`DROP_HEIGHT` above the ground: its pose changed, so it alone wakes and falls (in free flight it has no unit
    either), and the first replay whose contacts join it to that box wakes the box's island (Steel's ``begin`` unions
    a contact's islands, and an awake member wakes the island); before that the heap sleeps. After
    :data:`DROP_REPLAYS` replays every pose is within :data:`REST_ATOL` of the eager run's and the spare box rests on
    the top box.
    """
    model = _scene_heap(device, spare=True)
    spare = model.body_count - 1
    eager, captured = runs = [
        _sleep_run(model, fixed=True, free_body_constraint_max=HEAP_FREE_ROWS, iterations=HEAP_ITERATIONS)
        for _ in range(2)
    ]
    for run in runs:
        run.step()
    with wp.ScopedCapture(device) as recorded:
        captured.step()

    def replay():
        eager.step()
        wp.capture_launch(recorded.graph)
        awake = [run.solver._sleeping.body_awake.numpy() for run in runs]
        np.testing.assert_array_equal(awake[1], awake[0])

    replay()
    test.assertGreater(captured.solver.scene_statistics()["two_sided_units"], 0)
    for _ in range(SLEEP_STEPS):
        if _asleep(captured):
            break
        replay()
    test.assertTrue(_asleep(captured), f"the heap did not sleep within {SLEEP_STEPS} replays")
    joint_qd = captured.state_0.joint_qd.numpy()
    replay()
    stats = captured.solver.scene_statistics()
    test.assertEqual({key: stats[key] for key in ("units", "rows", "colors")}, {"units": 0, "rows": 0, "colors": 0})
    np.testing.assert_array_equal(captured.state_0.joint_qd.numpy(), joint_qd)
    for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
        test.assertTrue(np.all(np.isfinite(getattr(captured.state_0, name).numpy())), name)

    # The top box of the heap's centre column, under the drop, and its island.
    body_q = captured.state_0.body_q.numpy()
    column = np.flatnonzero(np.linalg.norm(body_q[:spare, :2], axis=1) < 0.01)
    under = int(column[np.argmax(body_q[column, 2])])
    island = next(members for members in _islands(captured.solver) if under in members)
    test.assertNotIn(spare, island)
    for run in runs:
        _set_free_body(model, run.state_0, spare, position=(*body_q[under, :2], DROP_HEIGHT))
    touched = None
    for index in range(1, DROP_REPLAYS + 1):
        replay()
        joined = (under, spare) in _body_pairs(model, captured.contacts)
        awake = captured.solver._sleeping.body_awake.numpy()
        test.assertEqual(int(awake[spare]), 1, f"replay {index}")
        if touched is None and joined:
            touched = index
        if touched is None:
            test.assertFalse(np.any(awake[:spare]), f"the heap woke before the contact, replay {index}")
            test.assertEqual(captured.solver.scene_statistics()["units"], 0, f"replay {index}")
        else:
            woken = awake[sorted(island)]
            test.assertTrue(np.all(woken), f"the contact did not wake the top box's island, replay {index}")
    test.assertIsNotNone(touched, f"the spare box did not reach the heap within {DROP_REPLAYS} replays")
    poses = [run.state_0.body_q.numpy() for run in runs]
    np.testing.assert_allclose(poses[1][:, :3], poses[0][:, :3], rtol=0.0, atol=REST_ATOL)
    rest = poses[1][spare, :3] - poses[1][under, :3]
    np.testing.assert_allclose(rest, (0.0, 0.0, 0.04), rtol=0.0, atol=REST_ATOL)


def _ant(device):
    """``nv_ant.xml`` on the ground, undriven: a free torso and eight limited hinges (``test_sleeping``'s ant)."""
    builder = newton.ModelBuilder()
    builder.add_mjcf(
        str(Path(newton.__file__).parent / "examples/assets/nv_ant.xml"),
        ignore_names=["floor", "ground"],
        collapse_fixed_joints=False,
        parse_sites=False,
    )
    builder.joint_target_ke[:] = [0.0] * len(builder.joint_target_ke)
    builder.joint_target_kd[:] = [0.0] * len(builder.joint_target_kd)
    builder.joint_act[:] = [0.0] * len(builder.joint_act)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    hinges = model.joint_type.numpy() == int(newton.JointType.REVOLUTE)
    starts = model.joint_qd_start.numpy()
    friction = np.zeros(model.joint_dof_count, dtype=np.float32)
    for joint in np.flatnonzero(hinges):
        friction[starts[joint] : starts[joint + 1]] = ANT_JOINT_FRICTION
    model.joint_friction.assign(friction)
    return model


class _AntRun:
    """The ant's scene solver with :data:`SLEEP`, limits on every bound and joint friction, stepping two steps at a
    time between its two states (``test_sleeping``'s ant; the graph of two steps replays every later pair)."""

    def __init__(self, model, **options):
        self.solver = SolverSteel(
            model,
            solve_mode="scene",
            **SLEEP,
            enable_joint_limits=True,
            enable_joint_friction=True,
            iterations=8,
            friction_anchor_beta=0.0,
            articulation_constraint_max=256,
            **options,
        )
        self.pipeline = newton.CollisionPipeline(model, rigid_contact_max=128, deterministic=True)
        self.contacts = self.pipeline.contacts()
        self.states = [model.state(), model.state()]
        self.control = model.control()

    def steps(self):
        for _ in range(2):
            self.pipeline.collide(self.states[0], self.contacts)
            self.solver.step(self.states[0], self.states[1], self.control, self.contacts, DT)
            self.states.reverse()


def _internal_rows(solver) -> tuple[int, int, int]:
    """The one articulation's internal rows: its joint-limit and joint-friction family counts, and their sum."""
    families = solver._scene_family_rows.numpy()[:, 0]
    internal = int(solver._scene_internal_rows.numpy()[0])
    return int(families[FAMILY_LIMIT]), int(families[FAMILY_JOINT_FRICTION]), internal


@unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
@unittest.skipUnless(hasattr(SolverSteel, "scene_statistics"), "needs the read-back API")
def test_sleeping_articulation_gets_no_internal_rows(test, device):
    """Give a sleeping articulation no internal rows, and restore them when a force wakes it, under capture.

    The undriven ant (:func:`_ant`) with every bound of its eight hinges a joint-limit row
    (``joint_limit_activation_gap=inf``: 16 rows, one per finite bound, counted from the model) and joint friction on
    every hinge (8 rows). Asleep with its rows skipped, its drive and limit families read the masked
    ``limit_q_index`` (-1 on its DOFs) and its joint friction ``_constraint_art_active`` (0), so it has no internal
    rows, no unit and no row; a force wakes it within one replay and its 24 rows return, with Steel's coordinates in
    the masked index. With ``sleep_skip_constraints=False`` it keeps its 24 rows while it sleeps. Two steps per
    captured graph, as in ``test_sleeping.test_ant_joint_limits_skip_and_restore``.
    """
    model = _ant(device)
    hinges = np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.REVOLUTE))
    starts = model.joint_qd_start.numpy()
    dofs = np.concatenate([np.arange(starts[j], starts[j + 1]) for j in hinges])
    lower, upper = model.joint_limit_lower.numpy()[dofs], model.joint_limit_upper.numpy()[dofs]
    limit_rows = int(np.count_nonzero(np.isfinite(lower)) + np.count_nonzero(np.isfinite(upper)))
    test.assertEqual((limit_rows, dofs.size), (16, 8))
    awake_rows = (limit_rows, dofs.size, limit_rows + dofs.size)
    for skip in (True, False):
        with test.subTest(sleep_skip_constraints=skip):
            run = _AntRun(model, sleep_skip_constraints=skip)
            solver = run.solver
            run.steps()
            test.assertEqual(_internal_rows(solver), awake_rows)
            with wp.ScopedCapture(device) as capture:
                run.steps()
            for replay in range(ANT_REPLAYS):
                wp.capture_launch(capture.graph)
                if replay % 100 == 99 and not solver._sleeping.art_awake.numpy()[0]:
                    break
            np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [0])
            stats = solver.scene_statistics()
            if skip:
                np.testing.assert_array_equal(solver._constraint_art_active.numpy(), [0])
                test.assertTrue(np.all(solver._sleeping.limit_q_index.numpy() < 0))
                test.assertEqual(_internal_rows(solver), (0, 0, 0))
                test.assertEqual((stats["units"], stats["rows"]), (0, 0))
            else:
                np.testing.assert_array_equal(solver._constraint_art_active.numpy(), [1])
                test.assertEqual(_internal_rows(solver), awake_rows)
                test.assertGreaterEqual(stats["rows"], awake_rows[2])
            force = np.zeros((model.body_count, 6), dtype=np.float32)
            force[0, 0] = 100.0
            for state in run.states:
                state.body_f.assign(force)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [1])
            test.assertEqual(_internal_rows(solver), awake_rows)
            np.testing.assert_array_equal(solver._sleeping.limit_q_index.numpy(), solver._joint_limit_q_index.numpy())
            for name in ("body_q", "body_qd"):
                test.assertTrue(np.all(np.isfinite(getattr(run.states[0], name).numpy())), name)
            test.assertFalse(np.any(solver.constraint_overflow.numpy()))


def _two_world_box_model(device):
    """One box on the ground replicated into two model worlds at the same pose (``test_sleeping``'s two worlds)."""
    box = newton.ModelBuilder()
    body = box.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
    box.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.replicate(box, 2)
    return builder.finalize(device=device)


@unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
def test_masked_reset_wakes_model_worlds(test, device):
    """Wake only the model worlds a masked reset selects, though the scene folds them into one solve world.

    ``reset(world_mask)`` wakes through ``_SleepState.wake``, which selects articulations by their model world
    (``_articulation_model_world``); the final mask entry selects global articulations. Two worlds' boxes asleep, in
    the scene and in Steel: an empty mask and a global-only mask wake neither, world 0's mask wakes world 0's box, and
    after one step it alone is awake (``test_sleeping.test_masked_reset_wakes_only_selected_worlds``).
    """
    model = _two_world_box_model(device)
    for name, scene in (("scene", True), ("steel", False)):
        with test.subTest(solver=name):
            run = _sleep_run(model, scene=scene)
            for _ in range(SLEEP_STEPS):
                run.step()
                if _asleep(run):
                    break
            test.assertTrue(_asleep(run))
            for mask, expected in (
                ([False, False, False], [0, 0]),
                ([False, False, True], [0, 0]),
                ([True, False, False], [1, 0]),
            ):
                run.solver.reset(run.state_0, world_mask=wp.array(mask, dtype=wp.bool, device=device))
                np.testing.assert_array_equal(run.solver._sleeping.body_awake.numpy(), expected, err_msg=str(mask))
            run.step()
            np.testing.assert_array_equal(run.solver._sleeping.art_awake.numpy(), [1, 0])


def _two_boxes(device):
    """Two boxes on the ground 0.3 m apart (0.5 m between centres), on low friction (0.2)."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.2
    builder.add_ground_plane()
    for x in (0.0, 0.5):
        body = builder.add_body(xform=wp.transform((x, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    return builder.finalize(device=device)


@unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
def test_contact_with_an_awake_box_wakes_its_island(test, device):
    """Wake a sleeping box on the step an awake box first touches it, as Steel's island union does.

    :func:`_two_boxes` asleep, in the scene and in Steel. Box 0 is set sliding at 2 m/s toward box 1: its velocity
    changed, so it alone wakes, and it slides awake without any further wake event. Until a step's contacts join the two
    boxes, box 1 sleeps with its pose frozen; on that step ``begin`` unions their islands, the island has an awake
    member, and box 1 wakes, both boxes reporting one island.
    """
    model = _two_boxes(device)
    for name, scene in (("scene", True), ("steel", False)):
        with test.subTest(solver=name):
            run = _sleep_run(model, scene=scene)
            for _ in range(SLEEP_STEPS):
                run.step()
                if _asleep(run):
                    break
            test.assertTrue(_asleep(run))
            frozen = run.state_0.body_q.numpy()[1].copy()
            _set_free_body(model, run.state_0, 0, velocity=(2.0, 0.0, 0.0))
            for step in range(1, SLEEP_STEPS + 1):
                run.step()
                awake = run.solver._sleeping.body_awake.numpy()
                if (0, 1) in _body_pairs(model, run.contacts):
                    np.testing.assert_array_equal(awake, [1, 1], err_msg=f"step {step}")
                    islands = run.solver._sleeping.body_island.numpy()
                    test.assertEqual(int(islands[0]), int(islands[1]))
                    break
                np.testing.assert_array_equal(awake, [1, 0], err_msg=f"step {step}")
                np.testing.assert_array_equal(run.state_0.body_q.numpy()[1], frozen, err_msg=f"step {step}")
            else:
                test.fail(f"box 0 did not reach box 1 within {SLEEP_STEPS} steps")


def _global_box_on_world_zero(device):
    """A box on the ground in each of two model worlds 1 m apart, and a global box resting on world 0's box alone."""
    scene = newton.ModelBuilder()
    body = scene.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
    scene.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder()
    builder.replicate(scene, 2, xforms=[wp.transform((x, 0.0, 0.0), wp.quat_identity()) for x in (0.0, 1.0)])
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.3), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.rigid_contact_max = CONTACT_MAX
    return model


@unittest.skipUnless("scene" in _SOLVE_MODES, "needs the scene mode")
def test_islands_join_within_steel_solve_worlds(test, device):
    """Join islands over a contact only within Steel's solve worlds, which the scene's fold leaves Steel's.

    Steel solves a global articulation in world 0, so its island joins world 0's over a contact and a contact with
    another world's articulation makes the step's islands incomplete, which keeps every island awake. The scene solves
    every articulation in one solve world but keeps Steel's join: a global box resting on world 0's box sleeps with it,
    and a global box resting on the boxes of worlds 0 and 1 (``_global_box_model``) keeps everything awake, in both
    solvers (Steel warns of the other worlds' contacts at construction and drops them; the scene warns of nothing).
    """
    for label, build, islands in (
        ("on world 0", _global_box_on_world_zero, {frozenset({0, 2}), frozenset({1})}),
        ("on worlds 0 and 1", _global_box_model, None),
    ):
        model = build(device)
        test.assertEqual(model.articulation_world.numpy().tolist(), [0, 1, -1])
        runs = {}
        # Steel warns that the global box can touch another world's body; the scene solves such contacts and does not.
        with test.assertWarnsRegex(NewtonWarning, r"global \(world -1\) articulations \[2\]"):
            runs["steel"] = _sleep_run(model, scene=False, warn_constraint_overflow=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runs["scene"] = _sleep_run(model, scene=True, warn_constraint_overflow=False)
        test.assertEqual([str(warning.message) for warning in caught], [], label)
        for _ in range(SLEEP_STEPS):
            for run in runs.values():
                run.step()
        for name, run in runs.items():
            with test.subTest(model=label, solver=name):
                awake = run.solver._sleeping.body_awake.numpy()
                if islands is None:
                    np.testing.assert_array_equal(awake, [1, 1, 1])
                    test.assertGreater(int(run.solver._sleeping.incomplete.numpy()[0]), 0)
                else:
                    np.testing.assert_array_equal(awake, [0, 0, 0])
                    test.assertEqual(_islands(run.solver), islands)


class TestSteelSceneState(unittest.TestCase):
    pass


# On every device: the scene's own invariants and host references (patches on the CPU too, Steel's default).
for _fn in (
    test_seed_sides_applies_the_response_rule,
    test_reset_selection_is_steels,
    test_patch_anchors_take_precedence_over_shared_anchors,
    test_seed_sides_matches_host_reference,
    test_sweep_and_seed_share_the_response_rule,
    test_warmstart_carries_into_the_velocity_pass,
    test_warmstart_rejections_and_cold_start,
    test_sleeping_island_gets_no_units,
    test_sleeping_articulation_gets_no_internal_rows,
):
    add_function_test(TestSteelSceneState, _fn.__name__, _fn, devices=get_test_devices())


# CUDA only, each for the reason beside it; test_scene_cpu steps them on the CPU against CUDA's.
for _fn in (
    test_anchor_beta_none_resolves_as_steel,  # Steel's omitted beta
    test_patch_friction_tracks_steel,  # Steel's patches
    test_patch_radius_uses_divided_row_mu,  # Steel's patches
    test_patch_rows_follow_the_anchors,  # Steel's patches
    test_patches_match_the_serial_oracle,  # the phases; the CPU sweeps the serial oracle alone
    test_patches_replay_under_capture,  # graph capture
    test_warmstart_tracks_steel,  # Steel's warm start
    test_warmstart_matches_the_serial_oracle,  # the phases
    test_warmstart_replays_under_capture,  # graph capture
    test_warmstart_resets_by_world_mask,  # Steel's warm start
    test_sleeping_tracks_steel,  # Steel's sleeping
    test_all_asleep_scene_replays_under_capture,  # graph capture
    test_masked_reset_wakes_model_worlds,  # Steel's sleeping
    test_contact_with_an_awake_box_wakes_its_island,  # Steel's sleeping
    test_islands_join_within_steel_solve_worlds,  # Steel's sleeping
):
    add_function_test(TestSteelSceneState, _fn.__name__, _fn, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
