# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact torsion under ``solve_mode="scene"``: the spin rows as late rows, swept by Steel's coupled budget.

Steel appends one angular row per contact group after a world's rows and sweeps it after the world's dense rows and
before the velocity limits (``torque_sweep_source``): the group's anchor tangent pairs are rescaled into ``mu`` times
the pooled normal load, and the spin row is bounded by ``radius * max(mu * load - sliding, 0)``. The scene groups the
contacts on the device (``group_scene_torsion``), writes the spin rows after every unit's rows (one-sided ones per
side, two-sided ones by the color of their group's representative contact unit) and sweeps them in phase C, before
the velocity limits. The scene is compared with Steel within the factor's rounding, row by row where the terms do not
hold ``J``, and with its serial oracle bit for bit.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel import contact_torsion, scene_torsion_device
from newton._src.solvers.steel.kernels import PGS_CONSTRAINT_TYPE_TORSION
from newton._src.solvers.steel.scene_pgs import RANGE_LATE
from newton.solvers import SolverSteel
from newton.tests.steel import test_scene_features
from newton.tests.steel._fixtures import DT, set_torsion_radius
from newton.tests.steel._scene_fixtures import _rms, _Run
from newton.tests.steel.test_scene_torsion_groups import PAD_WORLDS, _stacked_pads
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

PAD_HALF = (0.02, 0.015, 0.025)
"""Half extents [m] of ``test_torsion.fixture``'s pads, and of the stacked boxes."""

PAD_INERTIA = (0.00012, 0.00012, 0.00008)
"""Principal inertia [kg m^2] of the pads and boxes, mass 0.3 kg."""

RADIUS = 0.01
"""Torsion radius [m] of every shape."""

FIXTURE_DT = 0.0025
"""``test_torsion.fixture``'s step [s]."""

FIXTURE_OPTIONS = {
    "friction_anchor_beta": 0.0,
    "iterations": 128,
    "velocity_iterations": 0,
    "baumgarte": 0.05,
    "cfm": 0,
    "contact_regularization": 0,
    "enable_joint_velocity_limits": True,
    "contact_warmstart": False,
    "articulation_constraint_max": 64,
    "free_body_constraint_max": 16,
}
"""``test_torsion.fixture``'s solver options (point friction)."""

TRACK_QD_RTOL = 1.0e-5
TRACK_QD_ATOL = 1.0e-6
"""Agreement of the pads' joint velocities with Steel's after one step, the scene's tracking tolerance: Steel's dense
factor and the scene's sparse factor round the articulated pads' response differently (2.4e-7 at most on the L40, the
spin stopped)."""

TRACK_Q_RMS = 1.0e-6
"""Agreement [RMS] of the joint coordinates with Steel's after ``test_scene_features.TRACK_STEPS`` steps. Tighter
than the scene's 1e-3 convention (``test_scene_features.TRACK_RMS``): a spin stopped one or two steps late moves a
pad's angle by about 1e-3 [rad], so 1e-3 would barely constrain torsion (2.2e-9 measured on the pads)."""

DIAG_RTOL = 1.0e-5
"""Agreement of a spin row's diagonal and incident velocity (``J H^-1 J^T + cfm`` and ``J v_hat``, ``J`` through the
factor) with Steel's dense row's."""

BUDGET_RTOL = 1.0e-5
"""The slack of ``sum |t| <= mu * sum lambda_n``."""

STACK_WORLDS = 3
"""Worlds of the stacked boxes: a box on the ground and a box on it, spinning against each other."""

STACK_ROWS = 4 * 128
"""``articulation_constraint_max`` of the stacked boxes' runs."""

SEPARATION = 5.0e-6
"""Gap [m] of the retirement test's pads: touching for the grouping (at most ``_TOUCH_TOLERANCE``, 1e-5 m) and
apart for the velocity pass (more than ``_CONTACT_END_GAP_SLOP``, 1e-6 m)."""

CAPTURE_STEPS = 40
"""Steps of the capture test."""

GROUND_SPIN = 2.0
"""Spin [rad/s] of the box on the ground: its spin row, at its bound ``mu m g radius``, slows it about 0.46 rad/s per
step."""


def _pad_model(device, *, spin=1.0, sliding=0.0, closing=0.1, separation=0.0, restitution=0.0, radius=RADIUS):
    """``test_torsion.fixture``'s two articulated pads touching face to face, without gravity, and their state.

    The pads close at ``closing`` [m/s], slide at ``sliding`` [m/s] and spin at ``spin`` [rad/s] against each other;
    ``separation`` [m] opens a gap between them. Returns ``(model, state)``.
    """
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    axes = [newton.ModelBuilder.JointDofConfig(axis=wp.vec3(*a)) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    for side in (-1, 1):
        pose = wp.transform(wp.vec3(0, 0, side * (PAD_HALF[2] + separation / 2)), wp.quat_identity())
        body = builder.add_link(xform=pose, mass=0.3, inertia=wp.mat33(np.diag(PAD_INERTIA).astype(np.float32)))
        joint = builder.add_joint_d6(-1, body, parent_xform=pose, linear_axes=axes, angular_axes=axes)
        builder.add_articulation([joint])
        builder.add_shape_box(
            body,
            hx=PAD_HALF[0],
            hy=PAD_HALF[1],
            hz=PAD_HALF[2],
            cfg=newton.ModelBuilder.ShapeConfig(density=0, mu=0.5, restitution=restitution),
        )
    model = builder.finalize(device=device)
    set_torsion_radius(model, radius)
    state = model.state()
    state.joint_qd.assign(np.array([sliding, 0, closing, 0, 0, spin, -sliding, 0, -closing, 0, 0, -spin], np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state


class _PadRun:
    """A solver of the pads stepping its own states with ``test_torsion.fixture``'s collision pipeline."""

    def __init__(self, model, state, **options):
        self.solver = SolverSteel(model, **{**FIXTURE_OPTIONS, **options})
        self.solver._rigid_body_angular_damping.zero_()
        self.pipeline = newton.CollisionPipeline(
            model, contact_matching="latest", reduce_contacts=False, broad_phase="nxn", rigid_contact_max=64
        )
        self.contacts = self.pipeline.contacts()
        self.state_0 = model.state()
        self.state_1 = model.state()
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            wp.copy(getattr(self.state_0, name), getattr(state, name))
        self.control = model.control()

    def step(self, dt=FIXTURE_DT):
        self.pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)
        self.state_0, self.state_1 = self.state_1, self.state_0


def _stacked_boxes(device, worlds=STACK_WORLDS):
    """``worlds`` worlds of two articulated boxes stacked on the ground under gravity, spinning against each other.

    The bottom box rests on the ground (one-sided torsion groups), the top box on it (two-sided ones); each box is a
    D6 link, so Steel solves their contacts on its articulated rows. The model's joint velocities spin the boxes.
    """
    stack = newton.ModelBuilder()
    axes = [newton.ModelBuilder.JointDofConfig(axis=wp.vec3(*a)) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    inertia = wp.mat33(np.diag(PAD_INERTIA).astype(np.float32))
    for level in (0, 1):
        pose = wp.transform(wp.vec3(0.0, 0.0, PAD_HALF[2] * (1 + 2 * level)), wp.quat_identity())
        body = stack.add_link(xform=pose, mass=0.3, inertia=inertia)
        joint = stack.add_joint_d6(-1, body, parent_xform=pose, linear_axes=axes, angular_axes=axes)
        stack.add_articulation([joint])
        stack.add_shape_box(
            body, hx=PAD_HALF[0], hy=PAD_HALF[1], hz=PAD_HALF[2], cfg=newton.ModelBuilder.ShapeConfig(density=0, mu=0.5)
        )
    builder = newton.ModelBuilder()
    builder.replicate(stack, worlds, spacing=(1.0, 0.0, 0.0))
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    model = builder.finalize(device=device)
    set_torsion_radius(model, RADIUS)
    qd = np.zeros((worlds, 2, 6), np.float32)
    qd[:, 0, 5], qd[:, 1, 5] = 2.0, -3.0
    model.joint_qd.assign(qd.ravel())
    return model


def _spin_rows(solver) -> np.ndarray:
    """The scene's spin rows (``PGS_CONSTRAINT_TYPE_TORSION``) among its kept rows."""
    rows = int(solver._constraint_count.numpy()[0])
    return np.flatnonzero(solver._row_type.numpy()[0, :rows] == PGS_CONSTRAINT_TYPE_TORSION)


def _two_sided_spin_rows(solver) -> int:
    """The scene's two-sided spin rows: those its color pass sweeps."""
    start = solver._scene.torsion_color_start.numpy()
    return int(start[-1] - start[0])


def _one_sided_spin_rows(solver) -> int:
    """The scene's one-sided spin rows: those its sides sweep."""
    scene = solver._scene
    return int(np.sum(scene.side_torsion_end.numpy() - scene.side_torsion_begin.numpy()))


def _groups_by_contacts(solver, stats, count) -> dict:
    """``stats``' groups (Steel's ``_torsion_stats`` format) keyed by their members' contact indices (of ``count``).

    A group's normal rows are rows of its solve world: Steel's, or the scene's one world.
    """
    slots = solver._contact_slot.numpy()[:count]
    worlds = solver._contact_world.numpy()[:count] if solver.solve_mode != "scene" else np.zeros_like(slots)
    keyed = {}
    for group in stats["groups"]:
        rows = np.isin(slots, group["normal_rows"]) & (worlds == group["world"]) & (slots >= 0)
        keyed[frozenset(int(c) for c in np.flatnonzero(rows))] = group
    return keyed


def _pad_on_body(device, *, kinematic=False):
    """An articulated pad (a D6 link) under a box on a free joint, face to face, without gravity, and their state.

    The pad closes on the box at 0.1 m/s. A dynamic box rests and the pad spins at 10 rad/s (``test_torsion``'s
    free-body fixture without its limit); a ``kinematic`` box (a prescribed free body, without a response) spins at
    5 rad/s and the pad rests. Every shape has the torsion radius. Returns ``(model, state)``.
    """
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    inertia = wp.mat33(np.diag(PAD_INERTIA).astype(np.float32))
    shape = newton.ModelBuilder.ShapeConfig(density=0, mu=0.5, restitution=0.0)
    pose = wp.transform(wp.vec3(0, 0, -PAD_HALF[2]), wp.quat_identity())
    pad = builder.add_link(xform=pose, mass=0.3, inertia=inertia)
    axes = [newton.ModelBuilder.JointDofConfig(axis=wp.vec3(*a)) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    builder.add_articulation([builder.add_joint_d6(-1, pad, parent_xform=pose, linear_axes=axes, angular_axes=axes)])
    builder.add_shape_box(pad, hx=PAD_HALF[0], hy=PAD_HALF[1], hz=PAD_HALF[2], cfg=shape)
    box = builder.add_body(
        xform=wp.transform(wp.vec3(0, 0, PAD_HALF[2]), wp.quat_identity()), mass=0.3, inertia=inertia
    )
    builder.add_shape_box(box, hx=PAD_HALF[0], hy=PAD_HALF[1], hz=PAD_HALF[2], cfg=shape)
    model = builder.finalize(device=device)
    if kinematic:
        flags = model.body_flags.numpy()
        flags[box] |= int(newton.BodyFlags.KINEMATIC)
        model.body_flags.assign(flags)
    set_torsion_radius(model, RADIUS)
    state = model.state()
    qd = np.zeros(12, np.float32)
    qd[2] = 0.1
    if kinematic:
        qd[11] = 5.0
    else:
        qd[5] = 10.0
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state


def test_torsion_tracks_steel(test, device):
    """Step ``test_torsion``'s spinning pads with torsion as Steel does: one step and thirty.

    The two articulated pads close at 0.1 m/s and spin at 1 rad/s against each other, every shape with a 1 cm torsion
    radius: one two-sided spin row, which the scene sweeps in its color pass. The joint velocities after one step
    agree with Steel's within the factor's rounding (:data:`TRACK_QD_RTOL`), and the joint coordinates after
    ``TRACK_STEPS`` steps within :data:`TRACK_Q_RMS` RMS; Steel's and the scene's spin rows stop the spin.
    """
    track_steps = test_scene_features.TRACK_STEPS
    model, state = _pad_model(device)
    scene = _PadRun(model, state, solve_mode="scene")
    steel = _PadRun(model, state, solve_mode="serial")
    test.assertTrue(scene.solver._scene.features.torsion)
    for step in range(track_steps):
        scene.step()
        steel.step()
        if step == 0:
            test.assertEqual(len(_spin_rows(scene.solver)), 1)
            test.assertEqual(_two_sided_spin_rows(scene.solver), 1)
            test.assertEqual(steel.solver._torsion_stats["rows"], 1)
            qd_scene, qd_steel = scene.state_0.joint_qd.numpy(), steel.state_0.joint_qd.numpy()
            np.testing.assert_allclose(qd_scene, qd_steel, rtol=TRACK_QD_RTOL, atol=TRACK_QD_ATOL)
            # The spin stops below the disk's Coulomb torque capacity, as test_spin_stops_below_bound requires.
            test.assertLess(np.max(np.abs(qd_scene.reshape(2, 6)[:, 5])), 1.0e-4)
    q_scene, q_steel = scene.state_0.joint_q.numpy(), steel.state_0.joint_q.numpy()
    test.assertTrue(np.all(np.isfinite(q_scene)))
    test.assertLess(_rms(q_scene, q_steel), TRACK_Q_RMS)
    test.assertFalse(np.any(scene.solver.constraint_overflow.numpy()))


def _pair_response(solver, row) -> dict:
    """Row ``row``'s velocity response by coordinate: its free-body response over its free prefix, else its factor."""
    dof = solver._sparse_row_dof.numpy()[0, row]
    factor = solver._sparse_row_factor.numpy()[0, row]
    prefix = int(solver._scene.row_free_prefix.numpy()[row])
    # Without free bodies the free-body response is a stand-in, and every prefix is 0.
    free = solver._sparse_row_free_response.numpy()[0, row] if prefix else None
    return {int(index): float(free[k] if k < prefix else factor[k]) for k, index in enumerate(dof) if index >= 0}


def test_torsion_rescales_member_tangents(test, device):
    """Bound the members' sliding and the spin by the group's budget, rescaling tangent pairs that pass it.

    The pads spin at 100 rad/s and slide at 10 m/s against each other. After one step every group's anchor sliding
    ``sum |t|`` is at most ``mu * sum lambda_n * (1 + 1e-5)`` and the spin impulse at most its bound
    ``radius * (mu * sum lambda_n - sum |t|)``, with the same slack of ``1e-5 * mu * sum lambda_n`` (the bound is the
    difference of two near-equal float32 sums). The scene's own pair projection keeps each tangent pair inside its
    contact's cone, so the sum passes the budget only when it is pushed: the members' tangent impulses are then tripled
    and the spin row alone is swept once (the serial walk over that row). Its budget rescales every pair by one factor
    to ``mu * sum lambda_n``, applies the pairs' and the spin row's impulse changes through their rows' responses, and
    clamps the spin impulse to its bound, now about zero.
    """
    model, state = _pad_model(device, spin=100.0, sliding=10.0)
    run = _PadRun(model, state, solve_mode="scene")
    run.step()
    solver = run.solver
    scene = solver._scene
    stats = solver._torsion_stats
    test.assertEqual(stats["rows"], 1)
    group = stats["groups"][0]
    spin, normals, mu, radius = group["row"], np.asarray(group["normal_rows"]), group["mu"], group["effective_radius_m"]
    test.assertEqual(list(_spin_rows(solver)), [spin])
    test.assertEqual(sorted(group["anchor_rows"]), sorted(normals.tolist()))

    def budget(impulses):
        load = float(np.sum(np.maximum(impulses[normals].astype(np.float64), 0.0)))
        sliding = float(sum(np.linalg.norm(impulses[n + 1 : n + 3].astype(np.float64)) for n in normals))
        return load, sliding

    impulses = solver._impulses.numpy()[0]
    load, sliding = budget(impulses)
    test.assertGreater(load, 0.0)
    test.assertGreater(sliding, 0.0)
    test.assertLessEqual(sliding, mu * load * (1.0 + BUDGET_RTOL))
    # The bound is a difference of near-equal float32 sums: the slack is the budget's, 1e-5 of mu * load.
    test.assertLessEqual(abs(float(impulses[spin])), radius * (max(mu * load - sliding, 0.0) + BUDGET_RTOL * mu * load))

    # Push the members' sliding past the budget, then sweep the spin row alone.
    pushed = impulses.copy()
    for n in normals:
        pushed[n + 1 : n + 3] *= 3.0
    solver._impulses.assign(pushed[None, :])
    _, sliding_pushed = budget(pushed)
    test.assertGreater(sliding_pushed, mu * load)
    ranges = scene.row_ranges.numpy()
    ranges[RANGE_LATE] = (spin, spin + 1)
    scene.row_ranges.assign(ranges)
    delta_before = solver._sparse_factor_velocity_delta.numpy()[0].astype(np.float64)
    scene._walk(solver, RANGE_LATE, 1, stationary_exit=False, iteration=0)
    after = solver._impulses.numpy()[0]
    load_after, sliding_after = budget(after)
    test.assertEqual(load_after, load)
    np.testing.assert_allclose(sliding_after, mu * load, rtol=BUDGET_RTOL)
    scale = (mu * load) / sliding_pushed
    expected_change = np.zeros_like(delta_before)
    for n in normals:
        np.testing.assert_allclose(after[n + 1 : n + 3], pushed[n + 1 : n + 3] * scale, rtol=1.0e-6)
        for row in (n + 1, n + 2):
            for index, value in _pair_response(solver, row).items():
                expected_change[index] += value * (float(after[row]) - float(pushed[row]))
    bound = radius * (max(mu * load - sliding_after, 0.0) + BUDGET_RTOL * mu * load)
    test.assertLessEqual(abs(float(after[spin])), bound)
    for index, value in _pair_response(solver, spin).items():
        expected_change[index] += value * (float(after[spin]) - float(pushed[spin]))
    change = solver._sparse_factor_velocity_delta.numpy()[0].astype(np.float64) - delta_before
    np.testing.assert_allclose(change, expected_change, rtol=1.0e-4, atol=1.0e-6 * np.max(np.abs(expected_change)))


def test_torsion_late_rows_match_oracle(test, device):
    """Sweep the spin rows in phase C exactly as one thread walking every sweep does, in every phase order.

    Three worlds of two articulated boxes stacked on the ground and spinning against each other: the bottom box's
    ground contacts give one-sided spin rows (its side's), the box-on-box contacts two-sided ones, swept by the color
    of their group's representative contact unit. The phases (the default color loop, and the while loop) match the
    serial oracle (``scene_sweep="serial"``) bit for bit over 30 steps in each phase order.
    """
    model = _stacked_boxes(device)

    def both(solver):
        return _two_sided_spin_rows(solver) > 0 and _one_sided_spin_rows(solver) > 0

    test_scene_features._assert_phases_match_serial_oracle(
        test, model, {}, witness=both, what="no one-sided and two-sided spin rows", runner=_StackRun
    )


class _StackRun(_Run):
    """A :class:`_Run` of the stacked boxes with :data:`STACK_ROWS` rows per world."""

    def __init__(self, model, rows, overrides, **options):
        super().__init__(model, STACK_ROWS, overrides, **options)


def test_spin_rows_match_steel_rows(test, device):
    """Write each group's spin row as Steel's ``_apply_rows`` and ``_jacobians`` do, on the stacked pads' worlds.

    Steel (``solve_mode="serial"``, its device preparation ``contact_torsion_device=True``, whose rows its host
    oracle's equal) and the scene step the stacked pads (four worlds: five two-sided groups) once from one state and its
    contacts. Every contact with rows has Steel's solve world as its model world (the scene's ``torsion_world``, which
    keys a patch region). Per group, matched by its members' contacts: the members' group ids (Steel's
    ``_contact_torsion_group``) form the same partition, and the spin row's type, lead contact (its ``row_parent``),
    ``row_mu`` (the group's coefficient), radius, ``phi``, target and restitution are Steel's bit for bit. Its ``J``,
    stored through the factor of each articulated pad, agrees through its diagonal ``J H^-1 J^T + cfm`` and its
    incident velocity ``J v_hat`` with Steel's dense row within :data:`DIAG_RTOL`.
    """
    solvers, contacts = {}, {}
    for name, mode in (("scene", "scene"), ("steel", "serial")):
        model, state = _stacked_pads(device)
        pipeline = newton.CollisionPipeline(
            model,
            contact_matching="latest",
            reduce_contacts=False,
            broad_phase="nxn",
            rigid_contact_max=64 * PAD_WORLDS,
        )
        contacts[name] = pipeline.contacts()
        solver = SolverSteel(
            model,
            solve_mode=mode,
            iterations=16,
            velocity_iterations=0,
            contact_warmstart=False,
            friction_anchor_beta=0.0,
            articulation_constraint_max=128,
            contact_torsion_device=mode != "scene",
        )
        pipeline.collide(state, contacts[name])
        solver.step(state, model.state(), model.control(), contacts[name], DT)
        solvers[name] = solver
    scene, steel = solvers["scene"], solvers["steel"]
    np.testing.assert_array_equal(
        contacts["scene"].rigid_contact_count.numpy(), contacts["steel"].rigid_contact_count.numpy()
    )
    count = int(contacts["scene"].rigid_contact_count.numpy()[0])
    # Each contact's model world (scene_torsion_routing's torsion_world, which keys a patch region with its owner) is
    # Steel's solve world of its rows, on every contact with rows in both solvers, across the four worlds.
    with_rows = (scene._contact_slot.numpy()[:count] >= 0) & (steel._contact_slot.numpy()[:count] >= 0)
    steel_world = steel._contact_world.numpy()[:count][with_rows]
    test.assertEqual(set(steel_world.tolist()), set(range(PAD_WORLDS)))
    np.testing.assert_array_equal(scene._scene.torsion_world.numpy()[:count][with_rows], steel_world)
    stats = {"scene": scene._torsion_stats, "steel": steel._device_torsion.read_stats()}
    groups = {name: _groups_by_contacts(solvers[name], stats[name], count) for name in solvers}
    test.assertEqual(len(groups["steel"]), 5)
    test.assertEqual(set(groups["scene"]), set(groups["steel"]))
    for name, solver in solvers.items():
        # The members' group ids: every member's normal row names its group's spin row.
        membership = solver._contact_torsion_group.numpy()
        slots, worlds = solver._contact_slot.numpy()[:count], solver._contact_world.numpy()[:count]
        for members, group in groups[name].items():
            for c in members:
                world = 0 if solver.solve_mode == "scene" else worlds[c]
                test.assertEqual(int(membership[world, slots[c]]), group["row"], name)
    arrays = {
        name: {
            key: getattr(solver, f"_{key}").numpy()
            for key in ("row_type", "row_parent", "row_mu", "phi", "target_velocity", "row_rebound_velocity", "diag")
        }
        for name, solver in solvers.items()
    }
    radius = {name: solver._torsion_row_radius.numpy() for name, solver in solvers.items()}
    steel_v_hat = steel._v_hat.numpy()
    steel_j = steel._J_world.numpy()
    steel_dofs = steel._world_dof_indices.numpy()
    for members, steel_group in groups["steel"].items():
        scene_group = groups["scene"][members]
        world, steel_row, scene_row = steel_group["world"], steel_group["row"], scene_group["row"]
        values = {"scene": (0, scene_row), "steel": (world, steel_row)}
        for key in ("row_type", "row_mu", "phi", "target_velocity", "row_rebound_velocity"):
            test.assertEqual(arrays["scene"][key][values["scene"]], arrays["steel"][key][values["steel"]], key)
        test.assertEqual(arrays["scene"]["row_type"][0, scene_row], PGS_CONSTRAINT_TYPE_TORSION)
        test.assertEqual(radius["scene"][0, scene_row], radius["steel"][world, steel_row])
        test.assertEqual(scene_group["mu"], steel_group["mu"])
        test.assertEqual(scene_group["effective_radius_m"], steel_group["effective_radius_m"])
        # The lead's shapes, gathered at the step (torsion_shape_pair), are Steel's.
        test.assertEqual(scene_group["shape_pair"], steel_group["shape_pair"])
        # The lead: the parent rows name one contact.
        scene_lead = int(
            np.flatnonzero(scene._contact_slot.numpy()[:count] == arrays["scene"]["row_parent"][0, scene_row])[0]
        )
        steel_slots = steel._contact_slot.numpy()[:count]
        steel_worlds = steel._contact_world.numpy()[:count]
        steel_parent = arrays["steel"]["row_parent"][world, steel_row]
        steel_lead = int(np.flatnonzero((steel_slots == steel_parent) & (steel_worlds == world))[0])
        test.assertEqual(scene_lead, steel_lead)
        np.testing.assert_allclose(
            arrays["scene"]["diag"][0, scene_row], arrays["steel"]["diag"][world, steel_row], rtol=DIAG_RTOL
        )
        valid = steel_dofs[world] >= 0
        steel_incident = float(np.dot(steel_j[world, steel_row][valid], steel_v_hat[steel_dofs[world][valid]]))
        scene_incident = float(scene._sparse_row_incident.numpy()[0, scene_row])
        np.testing.assert_allclose(scene_incident, steel_incident, rtol=DIAG_RTOL, atol=1.0e-6)


def _box_on_ground(device, spin=GROUND_SPIN):
    """One articulated box (a D6 link) resting on the ground under gravity, spinning at ``spin`` [rad/s] about z.

    Every shape has the torsion radius. Returns ``(model, state)``.
    """
    builder = newton.ModelBuilder()
    pose = wp.transform(wp.vec3(0.0, 0.0, PAD_HALF[2]), wp.quat_identity())
    body = builder.add_link(xform=pose, mass=0.3, inertia=wp.mat33(np.diag(PAD_INERTIA).astype(np.float32)))
    axes = [newton.ModelBuilder.JointDofConfig(axis=wp.vec3(*a)) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    builder.add_articulation([builder.add_joint_d6(-1, body, parent_xform=pose, linear_axes=axes, angular_axes=axes)])
    shape = newton.ModelBuilder.ShapeConfig(density=0, mu=0.5)
    builder.add_shape_box(body, hx=PAD_HALF[0], hy=PAD_HALF[1], hz=PAD_HALF[2], cfg=shape)
    builder.add_ground_plane(cfg=shape)
    model = builder.finalize(device=device)
    set_torsion_radius(model, RADIUS)
    state = model.state()
    qd = np.zeros(6, np.float32)
    qd[5] = spin
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state


def _rotate(quaternion, vector, *, inverse=False):
    """``vector`` rotated by the unit quaternion ``(x, y, z, w)``, or by its inverse."""
    u = np.asarray(quaternion[:3], np.float64) * (-1.0 if inverse else 1.0)
    w = float(quaternion[3])
    t = 2.0 * np.cross(u, vector)
    return vector + w * t + np.cross(u, t)


def _central_witness(model, state, contacts):
    """Collapse the collided manifold of one shape pair to one witness at its center, as ``fixture(center_only=True)``.

    The witness keeps contact 0's shapes, normal and margins; its two points are the means of the manifold's world
    points on each shape, taken back to each body's frame. A central witness has no lever arm about the normal, so
    point friction cannot resist the spin: only the spin row does (``test_torsion``'s fixtures do the same).
    """
    count = int(contacts.rigid_contact_count.numpy()[0])
    if count == 0:
        return
    shapes = np.stack([contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count]])
    assert np.all(shapes == shapes[:, :1]), "one shape pair"
    shape_body = model.shape_body.numpy()
    poses = state.body_q.numpy().astype(np.float64)
    for side, name in enumerate(("rigid_contact_point0", "rigid_contact_point1")):
        points = getattr(contacts, name).numpy()
        body = int(shape_body[shapes[side, 0]])
        world = points[:count].astype(np.float64)
        if body >= 0:
            world = np.stack([_rotate(poses[body, 3:], p) for p in world]) + poses[body, :3]
        center = world.mean(axis=0)
        if body >= 0:
            center = _rotate(poses[body, 3:], center - poses[body, :3], inverse=True)
        points[0] = center
        getattr(contacts, name).assign(points)
    contacts.rigid_contact_count.assign(np.array([1], np.int32))


class _CentralRun(_PadRun):
    """A :class:`_PadRun` whose collided manifold becomes one central witness before each step."""

    def __init__(self, model, state, **options):
        super().__init__(model, state, **options)
        self.model = model

    def step(self, dt=FIXTURE_DT):
        self.pipeline.collide(self.state_0, self.contacts)
        _central_witness(self.model, self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)
        self.state_0, self.state_1 = self.state_1, self.state_0


def _spin_impulse(solver) -> float:
    """The impulse of the solver's one spin row after its step (Steel's ``_torsion_stats`` row)."""
    row = solver._torsion_stats["groups"][0]["row"]
    return float(solver._impulses.numpy()[0, row])


def test_one_sided_and_free_body_spin_rows_track_steel(test, device):
    """Stop a spin by a one-sided spin row and by a spin row with a free side as Steel does: one step and thirty.

    Each case keeps one central witness per step (as ``fixture(center_only=True)``), so point friction has no lever arm
    and the spin row alone resists the spin. A box spinning at 2 rad/s on the ground under gravity: a one-sided spin
    row (its side's, swept by its side in phase C), saturated at its bound in the first step. A pad spinning at 10
    rad/s under a box on a free joint, closing at 0.1 m/s: a two-sided spin row whose free side responds through its
    six physical entries (free prefix 6, the budget's free-body responses). Steel solves both on its articulated rows
    with the same spin row: after one step its impulse is nonzero in both solvers and agrees within
    :data:`TRACK_QD_RTOL`, the spin has slowed, the joint velocities agree within :data:`TRACK_QD_RTOL`, and the joint
    coordinates after ``TRACK_STEPS`` steps within :data:`TRACK_Q_RMS` RMS.
    """
    track_steps = test_scene_features.TRACK_STEPS
    for name, (model, state), spin_dof in (
        ("ground", _box_on_ground(device), 5),
        ("free body", _pad_on_body(device), 5),
    ):
        with test.subTest(case=name):
            scene = _CentralRun(model, state, solve_mode="scene")
            steel = _CentralRun(model, state, solve_mode="serial")
            spin = float(state.joint_qd.numpy()[spin_dof])
            for step in range(track_steps):
                scene.step()
                steel.step()
                if step == 0:
                    rows = _spin_rows(scene.solver)
                    test.assertEqual(len(rows), 1)
                    test.assertEqual(steel.solver._torsion_stats["rows"], 1)
                    if name == "ground":
                        test.assertEqual(_one_sided_spin_rows(scene.solver), 1)
                    else:
                        test.assertEqual(_two_sided_spin_rows(scene.solver), 1)
                        test.assertEqual(int(scene.solver._scene.row_free_prefix.numpy()[rows[0]]), 6)
                    # The spin row carries the load: its impulse, nonzero in both solvers, slowed the spin.
                    impulses = _spin_impulse(scene.solver), _spin_impulse(steel.solver)
                    test.assertNotEqual(impulses[1], 0.0)
                    np.testing.assert_allclose(impulses[0], impulses[1], rtol=TRACK_QD_RTOL)
                    qd_scene, qd_steel = scene.state_0.joint_qd.numpy(), steel.state_0.joint_qd.numpy()
                    test.assertLess(abs(float(qd_steel[spin_dof])), abs(spin) - 0.1)
                    np.testing.assert_allclose(qd_scene, qd_steel, rtol=TRACK_QD_RTOL, atol=TRACK_QD_ATOL)
            q_scene, q_steel = scene.state_0.joint_q.numpy(), steel.state_0.joint_q.numpy()
            test.assertTrue(np.all(np.isfinite(q_scene)))
            test.assertLess(_rms(q_scene, q_steel), TRACK_Q_RMS)
            test.assertFalse(np.any(scene.solver.constraint_overflow.numpy()))


def test_prescribed_side_sets_the_spin_target(test, device):
    """Give a spin row the target of a prescribed side, as Steel's ``_apply_rows`` does.

    A kinematic box on a free joint (a prescribed body, without a response) spins at 5 rad/s on an articulated pad:
    its contacts are one-sided, on the pad's side, and the spin row's target is ``-sign * dot(n, omega)`` of the
    kinematic body (Steel's ``_host_dot3``). Steel's device preparation and the scene, stepped once from one state,
    write the same nonzero target, ``row_mu``, radius and ``phi`` bit for bit.
    """
    rows = {}
    for name, mode in (("scene", "scene"), ("steel", "serial")):
        model, state = _pad_on_body(device, kinematic=True)
        run = _PadRun(model, state, solve_mode=mode, contact_torsion_device=mode != "scene")
        run.step()
        solver = run.solver
        stats = solver._torsion_stats if mode == "scene" else solver._device_torsion.read_stats()
        test.assertEqual(stats["rows"], 1, name)
        if mode == "scene":
            # The kinematic box has no response: the contacts and the spin row are the pad's side's alone.
            test.assertEqual(_one_sided_spin_rows(solver), 1)
        row = stats["groups"][0]["row"]
        rows[name] = [
            getattr(solver, f"_{key}").numpy()[0, row]
            for key in ("row_type", "target_velocity", "row_mu", "torsion_row_radius", "phi")
        ]
    test.assertEqual(int(rows["scene"][0]), PGS_CONSTRAINT_TYPE_TORSION)
    test.assertNotEqual(float(rows["steel"][1]), 0.0)
    for key, a, b in zip(("type", "target", "mu", "radius", "phi"), rows["scene"], rows["steel"], strict=True):
        test.assertEqual(a, b, key)


def test_retired_spin_row_refunds_under_velocity_iterations(test, device):
    """Retire a spin row whose members part after the position solve, and refund its spin in the velocity pass.

    The pads, 5 um apart (touching for the grouping), close at 1 m/s with restitution 1 and spin at 1 rad/s: the
    position solve loads the contact and its spin row, and the rebound parts the pads, so every member's end gap
    passes the slop. Steel's velocity pass marks the spin row retired (``_TORSION_SPIN_RETIRED``, its bound 0) and the
    sweep refunds its impulse: the scene marks its spin row in ``_contact_torsion_group`` as Steel does, and the spin
    impulse is 0 after the velocity iterations where the position solve alone left it nonzero, as Steel's.
    """
    impulses = {}
    for velocity_iterations in (0, 2):
        for name, mode in (("scene", "scene"), ("steel", "serial")):
            model, state = _pad_model(device, closing=1.0, separation=SEPARATION, restitution=1.0)
            run = _PadRun(model, state, solve_mode=mode, velocity_iterations=velocity_iterations)
            run.step()
            stats = run.solver._torsion_stats
            test.assertEqual(stats["rows"], 1, name)
            row = stats["groups"][0]["row"]
            impulses[name, velocity_iterations] = float(run.solver._impulses.numpy()[0, row])
            marker = int(run.solver._contact_torsion_group.numpy()[0, row])
            expected = contact_torsion._TORSION_SPIN_RETIRED if velocity_iterations else -1
            test.assertEqual(marker, expected, (name, velocity_iterations))
    for name in ("scene", "steel"):
        test.assertNotEqual(impulses[name, 0], 0.0, name)
        test.assertEqual(impulses[name, 2], 0.0, name)


def test_spin_rows_come_and_go_under_capture(test, device):
    """Replay one captured step while spin rows appear and disappear, as eager steps do.

    The stacked boxes fall 2 mm onto the ground and settle, so the first steps have no touching group and the later
    ones have both kinds; every fifth step the collision's contacts are dropped (count 0), so every group goes and
    then comes back. One captured solver step (the collision runs before it, outside the graph) replays every step
    with ``joint_q``, ``joint_qd`` and the impulses equal to the eager twin's bit for bit, and the spin rows' count
    takes several values, zero among them.
    """
    model = _stacked_boxes(device, worlds=2)
    q = model.joint_q.numpy()
    q[2::6] += 0.002
    model.joint_q.assign(q)
    eager, captured = runs = [_Stepper(model) for _ in range(2)]
    # One eager step builds every kernel and buffer; the capture then records the solver's step alone.
    for run in runs:
        run.collide()
        run.step()
        run.advance()
    with wp.ScopedCapture(device) as recorded:
        captured.step()
    counts = set()
    for step in range(1, CAPTURE_STEPS):
        for run in runs:
            run.collide(drop=step % 5 == 4)
        eager.step()
        wp.capture_launch(recorded.graph)
        for run in runs:
            run.advance()
        counts.add(len(_spin_rows(eager.solver)))
        for name in ("joint_q", "joint_qd"):
            a, b = getattr(eager.state_0, name).numpy(), getattr(captured.state_0, name).numpy()
            test.assertTrue(np.array_equal(a, b), f"{name} differs after step {step}")
        np.testing.assert_array_equal(captured.solver._impulses.numpy(), eager.solver._impulses.numpy())
        np.testing.assert_array_equal(captured.solver._row_type.numpy(), eager.solver._row_type.numpy())
    test.assertIn(0, counts)
    test.assertGreater(len(counts), 2)


class _Stepper:
    """The stacked boxes' scene solver stepping fixed state buffers, so that one captured step serves every step."""

    def __init__(self, model):
        self.solver = SolverSteel(
            model, articulation_constraint_max=STACK_ROWS, solve_mode="scene", friction_anchor_beta=0.0
        )
        self.pipeline = newton.CollisionPipeline(model, deterministic=True)
        self.contacts = self.pipeline.contacts()
        self.state_0, self.state_1 = model.state(), model.state()
        self.control = model.control()
        newton.eval_fk(model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def collide(self, *, drop=False):
        self.pipeline.collide(self.state_0, self.contacts)
        if drop:
            self.contacts.rigid_contact_count.zero_()

    def step(self):
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, DT)

    def advance(self):
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            wp.copy(getattr(self.state_0, name), getattr(self.state_1, name))


def test_torsion_errors_are_latched(test, device):
    """Refuse torsion where Steel refuses it, latching the cause instead of raising, and drop rows past capacity.

    Steel raises from its preparation; the scene, which reads nothing back inside a step, latches the cause in
    ``scene_statistics()["torsion_status"]`` (Steel's bits) and builds no spin row that step: hydroelastic contacts (a
    positive stiffness, bit 2; Steel's "does not support hydroelastic contacts"), contact rows lost to the row budget
    (bit 4, where Steel gives the world no dense torsion rows) and a contact count past the buffer (bit 1; Steel's
    "Contact input overflow"); contact compliance's compliant contacts are hydroelastic ones, so with contact torsion
    they get their compliant rows and no spin row. Spin rows come after every unit's rows, so at a budget one row short
    of them the spin row alone is dropped (bit 8): the rows that fit are kept, as Steel keeps a world's torsion rows
    that fit; the step reports one dropped row, latches ``constraint_overflow``, and sweeps no spin row. With the budget
    exact the row is kept. ``check_constraint_capacity`` raises the latched cause with its message
    (``SCENE_TORSION_MESSAGES``), as Steel's deferred (captured) validation does, and a reset clears the latch.
    """
    model, state = _pad_model(device)
    run = _PadRun(model, state, solve_mode="scene")
    run.step()
    stats = run.solver.scene_statistics()
    test.assertEqual((stats["torsion_rows"], stats["torsion_status"], stats["dropped_rows"]), (1, 0, 0))
    rows = stats["rows"]
    # A step without contacts has no group and no spin row.
    run.solver.step(run.state_0, run.state_1, run.control, None, FIXTURE_DT)
    stats = run.solver.scene_statistics()
    test.assertEqual((stats["torsion_rows"], stats["torsion_status"]), (0, 0))
    test.assertEqual(run.solver._torsion_stats, {"rows": 0, "groups": []})
    # Hydroelastic contacts: a stiffness on the collided contacts, as Steel's test gives them.
    run = _PadRun(model, state, solve_mode="scene")
    run.pipeline.collide(run.state_0, run.contacts)
    run.contacts.rigid_contact_stiffness = wp.ones(run.contacts.rigid_contact_max, device=model.device)
    run.solver.step(run.state_0, run.state_1, run.control, run.contacts, FIXTURE_DT)
    stats = run.solver.scene_statistics()
    test.assertEqual((stats["torsion_rows"], stats["torsion_status"]), (0, 2))
    test.assertEqual(len(_spin_rows(run.solver)), 0)
    # Steel's deferred validation raises the latched cause at the capacity check, with Steel's message.
    with test.assertRaisesRegex(RuntimeError, "rejected its input: Contact torsion does not support hydroelastic"):
        run.solver.check_constraint_capacity()
    run.solver.reset(run.state_0)
    test.assertEqual(run.solver.scene_statistics()["torsion_status"], 0)
    run.solver.check_constraint_capacity()
    # Contact compliance with torsion: Steel's torsion refuses the compliant (hydroelastic) contacts; the scene
    # assembles them and builds no spin row.
    run = _PadRun(model, state, solve_mode="scene", contact_compliance=True)
    for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
        setattr(run.contacts, name, wp.zeros(run.contacts.rigid_contact_max, dtype=float, device=model.device))
    run.pipeline.collide(run.state_0, run.contacts)
    for name, value in (("rigid_contact_stiffness", 2.0e4), ("rigid_contact_damping", 50.0)):
        getattr(run.contacts, name).fill_(value)
    run.solver.step(run.state_0, run.state_1, run.control, run.contacts, FIXTURE_DT)
    stats = run.solver.scene_statistics()
    test.assertEqual((stats["torsion_rows"], stats["torsion_status"]), (0, 2))
    test.assertGreater(stats["compliant_contacts"], 0)
    # An overflowing contact input: the count past the buffer.
    run = _PadRun(model, state, solve_mode="scene", warn_constraint_overflow=False)
    run.pipeline.collide(run.state_0, run.contacts)
    run.contacts.rigid_contact_count.fill_(run.contacts.rigid_contact_max + 1)
    run.solver.step(run.state_0, run.state_1, run.control, run.contacts, FIXTURE_DT)
    stats = run.solver.scene_statistics()
    test.assertEqual((stats["torsion_rows"], stats["torsion_status"] & 1), (0, 1))
    test.assertEqual(len(_spin_rows(run.solver)), 0)
    with test.assertRaisesRegex(RuntimeError, "rejected its input: Contact input overflow"):
        run.solver.check_constraint_capacity()
    # The budget (one world, so the scene's rows are its articulation budget): exact, one row short, and short of a
    # contact's rows.
    for budget, kept, status in ((rows, 1, 0), (rows - 1, 0, 8), (rows - 4, 0, 4)):
        with test.subTest(row_max=budget):
            run = _PadRun(
                model, state, solve_mode="scene", articulation_constraint_max=budget, warn_constraint_overflow=False
            )
            test.assertEqual(run.solver._scene.row_max, budget)
            run.step()
            stats = run.solver.scene_statistics()
            test.assertEqual((stats["torsion_rows"], stats["torsion_status"] & 12), (kept, status))
            test.assertEqual(len(_spin_rows(run.solver)), kept)
            test.assertEqual(bool(np.any(run.solver.constraint_overflow.numpy())), status != 0)
            if status == 8:
                test.assertEqual(stats["dropped_rows"], 1)
                test.assertEqual(stats["rows"], rows - 1)
            if status:
                with test.assertRaisesRegex(
                    RuntimeError, "rejected its input: (Contact torsion rows dropped|Contact rows lost)"
                ):
                    run.solver.check_constraint_capacity()


def test_spin_rows_count_in_the_row_watermarks(test, device):
    """Count each spin row in its model world's row marks, as Steel counts its torsion rows with its dense rows.

    Steel appends a world's torsion rows to its dense rows and counts them in its marks (``prepare_torsion_rows`` sets
    the world's row and slot counts past them). The stacked boxes' three worlds, each with a one-sided and a two-sided
    spin row, step ``WATERMARK_STEPS`` times with ``row_watermark`` under the scene solve and under Steel (its host
    preparation): every mark equals Steel's, the articulation family's kept and requested rows among them, which hold
    each world's spin rows. On the pads, at a budget one row short of the spin row, the dropped spin
    row is requested and not kept (Steel raises there): the requested mark is the rows with it, the kept mark one
    less, an excess of one row in one overflowing world-step and no dropped contact row.
    """
    watermark_steps = test_scene_features.WATERMARK_STEPS
    model = _stacked_boxes(device)
    scene = _StackRun(model, STACK_ROWS, {}, row_watermark=True)
    steel = _StackRun(model, STACK_ROWS, {}, solve_mode="serial", row_watermark=True)
    for _ in range(watermark_steps):
        scene.step()
        steel.step()
        test.assertGreater(_one_sided_spin_rows(scene.solver), 0)
        test.assertGreater(_two_sided_spin_rows(scene.solver), 0)
    marks = scene.solver.constraint_row_watermarks()
    test.assertEqual(marks, steel.solver.constraint_row_watermarks())
    test.assertGreater(marks["articulation_high_water"], 0)
    model, state = _pad_model(device)
    rows = 0
    for budget in (None, "short"):
        options = {} if budget is None else {"articulation_constraint_max": rows - 1}
        run = _PadRun(model, state, solve_mode="scene", row_watermark=True, warn_constraint_overflow=False, **options)
        run.step()
        stats = run.solver.scene_statistics()
        marks = run.solver.constraint_row_watermarks()
        if budget is None:
            rows = stats["rows"]
            test.assertEqual(stats["torsion_rows"], 1)
            test.assertEqual((marks["articulation_high_water"], marks["articulation_raw_high_water"]), (rows, rows))
            continue
        test.assertEqual((stats["torsion_rows"], stats["dropped_rows"]), (0, 1))
        test.assertEqual(
            (
                marks["articulation_high_water"],
                marks["articulation_raw_high_water"],
                marks["articulation_dropped_contact_rows_high_water"],
                marks["articulation_overflow_excess_high_water"],
                marks["articulation_overflow_world_steps"],
            ),
            (rows - 1, rows, 0, 1, 1),
        )


class TestSceneTorsion(unittest.TestCase):
    def test_torsion_helpers_mirror_steels_gather(self):
        """Keep the scene's copies of Steel's frame, compatibility and budget expressions equal to Steel's text.

        ``scene_torsion_device`` copies three expressions of Steel's device torsion preparation as ``wp.func``s:
        the contact frame of ``_gather``, the cluster compatibility walk and the member budget of ``_group_world``.
        Each copy's lines, and Steel's lines they mirror with the workspace's names substituted, are extracted by
        ``scene_torsion_device._mirrored_lines``; they must be equal, so that a change of Steel's text fails here.
        """
        for name in scene_torsion_device.SCENE_MIRRORED_HELPERS:
            with self.subTest(helper=name):
                scene, steel = scene_torsion_device._mirrored_lines(name)
                self.assertGreater(len(scene), 4)
                self.assertEqual(scene, steel)


# CUDA only: contact torsion requires a CUDA model (Steel's own check, which the scene keeps on the CPU).
for _fn in (
    test_torsion_tracks_steel,
    test_torsion_rescales_member_tangents,
    test_torsion_late_rows_match_oracle,
    test_spin_rows_match_steel_rows,
    test_one_sided_and_free_body_spin_rows_track_steel,
    test_prescribed_side_sets_the_spin_target,
    test_retired_spin_row_refunds_under_velocity_iterations,
    test_spin_rows_come_and_go_under_capture,
    test_torsion_errors_are_latched,
    test_spin_rows_count_in_the_row_watermarks,
):
    add_function_test(TestSceneTorsion, _fn.__name__, _fn, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
