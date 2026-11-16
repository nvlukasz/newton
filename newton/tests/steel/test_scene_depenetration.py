# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The depenetration cap of Steel's scene solve (``solve_mode="scene"``): ``rigid_body_max_depenetration_velocity``.

Steel bounds the Baumgarte bias ``pgs_beta * phi / dt`` of a penetrating contact it solves on its free-body path (every
side a free body or without response DOFs) below by ``-max_depen``, the bodies' cap in Steel's order: the first body's
as it is, lowered to the second's when that is finite, positive and smaller, and applied when finite and positive
(``compute_mf_effective_mass_and_rhs``). An impact whose rebound fires keeps its rebound target. A contact with an
articulated side is solved on Steel's dense path, which has no cap. The scene applies the cap after its row
preparation (``cap_scene_depenetration``, ``SceneFeatures.depenetration_cap``), so a capped row's right-hand side is
Steel's bit for bit, and the scene is compared with Steel's matrix-free solve within ``TRACK_RTOL`` (free bodies have no
factor).
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel.kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_LOCAL_SOLVE_OWNER_GENERAL
from newton._src.solvers.steel.scene_solve import cap_scene_depenetration
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import DT
from newton.tests.steel._scene_fixtures import SCENE, _steel_on
from newton.tests.steel.test_scene_features import TRACK_ATOL, TRACK_RTOL
from newton.tests.unittest_utils import add_function_test, get_test_devices

DEPENETRATION_CAP = 0.1
"""``rigid_body_max_depenetration_velocity`` [m/s] of the capped boxes: below the 0.24 m/s Baumgarte bias of their
5 mm overlap at the default ``baumgarte`` 0.2 and ``DT`` 1/240 s, so the cap binds."""

OVERLAP = 0.005
"""Interpenetration [m] of the boxes' faces at the start."""

HALF = 0.05
"""Half extent [m] of the boxes."""

CAP_ROWS = 64
"""``articulation_constraint_max`` of the cap tests' runs."""

LOOSE_CAP = 1.0
"""A cap [m/s] above the boxes' 0.24 m/s Baumgarte bias: finite and positive, so the feature is on; it never binds."""

POISON = -123.25
"""Fill of the right-hand sides before the cap kernel runs alone: a row it does not write keeps it."""

SLIDER_SPEED = 0.5
"""Initial speed [m/s] of the lone sunk slider."""

SLIDER_STEPS = 30
"""Steps of the sunk slider test."""

ORDER_GAP_BAND = (0.05, 0.2)
"""Bounds of the pushed slider fixture's largest joint velocity difference [m/s or rad/s] from Steel at step
:data:`SLIDER_STEPS` under the default scene phase order ``"BA"``: 0.115 measured on an L40."""


def _capped_boxes(device, caps, *, plate=False) -> newton.Model:
    """Two boxes side by side on x, their facing sides :data:`OVERLAP` deep in each other, without gravity.

    ``caps`` are the boxes' ``rigid_body_max_depenetration_velocity`` (``inf`` for no cap). With ``plate`` the second
    body is instead the link of a one-link articulation, a slider along x, so the first box's contacts with it are
    Steel's dense rows; the first box then also rests :data:`OVERLAP` deep in the ground, whose contacts are Steel's
    free-body rows, and the link hovers 1 cm above the ground, out of the 2 mm contact gap.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, 0.0))
    SolverSteel.register_custom_attributes(builder)
    builder.rigid_gap = 0.002
    height = HALF - OVERLAP if plate else 1.0
    if plate:
        builder.add_ground_plane()
    first = builder.add_body(
        xform=wp.transform(wp.vec3(-HALF + 0.5 * OVERLAP, 0.0, height), wp.quat_identity()),
        custom_attributes={"steel:rigid_body_max_depenetration_velocity": caps[0]},
    )
    builder.add_shape_box(first, hx=HALF, hy=HALF, hz=HALF)
    second_pose = wp.transform(wp.vec3(HALF - 0.5 * OVERLAP, 0.0, HALF + 0.01 if plate else height), wp.quat_identity())
    if plate:
        link = builder.add_link(
            xform=second_pose, custom_attributes={"steel:rigid_body_max_depenetration_velocity": caps[1]}
        )
        builder.add_shape_box(link, hx=HALF, hy=HALF, hz=HALF)
        slider = builder.add_joint_prismatic(parent=-1, child=link, axis=(1.0, 0.0, 0.0), parent_xform=second_pose)
        builder.add_articulation([slider])
    else:
        second = builder.add_body(
            xform=second_pose, custom_attributes={"steel:rigid_body_max_depenetration_velocity": caps[1]}
        )
        builder.add_shape_box(second, hx=HALF, hy=HALF, hz=HALF)
    return builder.finalize(device=device)


def _sunk_slider(device, *, box=True) -> newton.Model:
    """A box on a slider along x, its link :data:`OVERLAP` deep in the ground, without gravity.

    The link's ground contacts have normal rows with ``J = 0`` on the slider's one DOF. With ``box`` a free box rests
    as deep in the ground on the link's ``-x`` side, :data:`OVERLAP` into it, and pushes it (the
    :func:`_capped_boxes` plate fixture with the link sunk and no cap).
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, 0.0))
    SolverSteel.register_custom_attributes(builder)
    builder.rigid_gap = 0.002
    builder.add_ground_plane()
    if box:
        first = builder.add_body(
            xform=wp.transform(wp.vec3(-HALF + 0.5 * OVERLAP, 0.0, HALF - OVERLAP), wp.quat_identity())
        )
        builder.add_shape_box(first, hx=HALF, hy=HALF, hz=HALF)
    pose = wp.transform(wp.vec3(HALF - 0.5 * OVERLAP, 0.0, HALF - OVERLAP), wp.quat_identity())
    link = builder.add_link(xform=pose)
    builder.add_shape_box(link, hx=HALF, hy=HALF, hz=HALF)
    slider = builder.add_joint_prismatic(parent=-1, child=link, axis=(1.0, 0.0, 0.0), parent_xform=pose)
    builder.add_articulation([slider])
    return builder.finalize(device=device)


def _slider_speeds(model, steps, *, speed=0.0, **options):
    """Step the sunk slider ``steps`` times.

    Returns the solver, the slider's speed after each step, the contacts, the first step's impulses and diagonals,
    and the joint velocities after the last step.
    """
    solver = SolverSteel(model, **_steel_on(model, {**SCENE, **options}), articulation_constraint_max=CAP_ROWS)
    state, out = model.state(), model.state()
    if speed:
        joint_qd = state.joint_qd.numpy()
        joint_qd[-1] = speed
        state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    control = model.control()
    speeds = []
    for step in range(steps):
        pipeline.collide(state, contacts)
        solver.step(state, out, control, contacts, DT)
        state, out = out, state
        speeds.append(float(state.joint_qd.numpy()[-1]))
        if step == 0:
            first = (solver._impulses.numpy().copy(), solver._diag.numpy().copy())
    return solver, np.asarray(speeds), contacts, first, state.joint_qd.numpy()


def _step_once(model, **options):
    """Scene and Steel solvers stepped once from the model's state with the same contacts.

    Returns ``(solvers, states, contacts)``, the first two by ``"scene"`` and ``"steel"``.
    """
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    solvers, states = {}, {}
    for name, solve in (("scene", {}), ("steel", {"solve_mode": "serial"})):
        solver = SolverSteel(
            model, **_steel_on(model, {**SCENE, **solve, **options}), articulation_constraint_max=CAP_ROWS
        )
        out = model.state()
        solver.step(state, out, model.control(), contacts, DT)
        solvers[name], states[name] = solver, out
    return solvers, states, contacts


def _separating_speed(state) -> float:
    """The two free boxes' relative speed along x after the step (their FREE joints' world-aligned ``joint_qd``)."""
    joint_qd = state.joint_qd.numpy()
    return float(joint_qd[6] - joint_qd[0])


def _normal_rows(solvers, contacts):
    """The scene's and Steel's normal rows of every contact Steel solves on its free-body path, and their rhs."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    scene, steel = solvers["scene"], solvers["steel"]
    path = steel._contact_path.numpy()[:count]
    free = np.flatnonzero(path == 1)
    scene_rows = scene._contact_slot.numpy()[:count][free]
    steel_rows = steel._contact_slot.numpy()[:count][free]
    return free, scene._rhs.numpy()[0, scene_rows], steel._mf_rhs.numpy()[0, steel_rows], scene_rows


def test_depenetration_cap_tracks_steel(test, device):
    """Separate two boxes 5 mm deep in each other at their bodies' depenetration cap, as Steel's free-body solve does.

    Two free boxes interpenetrating 5 mm, without gravity, with the cap 0.1 m/s on both: the scene's cap feature is on,
    every penetrating normal row's right-hand side is ``-0.1`` and Steel's bit for bit, and the separating speed after
    the first step agrees with Steel's within 1e-5, less than half the 0.24 m/s separation of the uncapped boxes (the
    cap binds). Steel's order of the two caps holds too, in the scene as in Steel: a cap on one box only (the other
    ``inf``, in either order) bounds the contact, and of the two cases with one cap ``0``, only the one whose ``0``
    is not on the contacts' first body bounds them (Steel takes the first body's cap as it is). In every case the
    right-hand sides of the bounded rows equal Steel's bit for bit, the others agree within 1e-6, and the separating
    speed after the first step agrees with Steel's within 1e-5.
    """
    uncapped_solvers, uncapped, _ = _step_once(_capped_boxes(device, (np.inf, np.inf)))
    test.assertFalse(uncapped_solvers["scene"]._scene.features.depenetration_cap)
    uncapped_speed = _separating_speed(uncapped["scene"])
    test.assertGreater(uncapped_speed, 0.2)
    cases = {
        "both": (DEPENETRATION_CAP, DEPENETRATION_CAP),
        "first": (DEPENETRATION_CAP, np.inf),
        "second": (np.inf, DEPENETRATION_CAP),
        "zero first": (0.0, DEPENETRATION_CAP),
        "zero second": (DEPENETRATION_CAP, 0.0),
    }
    zero_bound = {}
    for name, caps in cases.items():
        with test.subTest(caps=name):
            model = _capped_boxes(device, caps)
            solvers, states, contacts = _step_once(model)
            test.assertTrue(solvers["scene"]._scene.features.depenetration_cap)
            test.assertTrue(solvers["scene"]._scene.sides_kernel.key.endswith("_dpc"))
            free, ours, theirs, _rows = _normal_rows(solvers, contacts)
            test.assertGreater(free.size, 0)
            bound = theirs == np.float32(-DEPENETRATION_CAP)
            test.assertTrue(np.array_equal(ours[bound], theirs[bound]), f"{ours} != {theirs}")
            np.testing.assert_allclose(ours, theirs, rtol=1.0e-6)
            speed, steel_speed = _separating_speed(states["scene"]), _separating_speed(states["steel"])
            np.testing.assert_allclose(speed, steel_speed, rtol=TRACK_RTOL)
            np.testing.assert_allclose(
                states["scene"].joint_qd.numpy(), states["steel"].joint_qd.numpy(), rtol=TRACK_RTOL, atol=TRACK_ATOL
            )
            if name == "both":
                test.assertLess(speed, 0.5 * uncapped_speed)
            if name.startswith("zero"):
                zero_bound[name] = bool(np.any(bound))
            else:
                test.assertTrue(np.all(bound), f"{name}: unbounded rows {theirs}")
    # Steel's order: the first body's cap is taken as it is, so a non-positive one turns the cap off.
    test.assertEqual(sorted(zero_bound.values()), [False, True], zero_bound)


def test_depenetration_cap_leaves_articulated_contacts(test, device):
    """Bound a capped box's contacts with the ground and not those with an articulated link, as Steel does.

    A free box with the cap 0.1 m/s rests 5 mm deep in the ground and 5 mm into the side of a box on a slider, a
    one-link articulation, without gravity. Steel solves the box's ground contacts on its free-body path, which bounds
    their bias, and its contacts with the link on its dense path, which does not. The scene's ground rows are bounded
    and equal Steel's bit for bit; its rows against the link equal the rows of the same scene without the cap bit for
    bit; and the velocities after the first step agree with Steel's within 1e-5.
    """
    model = _capped_boxes(device, (DEPENETRATION_CAP, DEPENETRATION_CAP), plate=True)
    solvers, states, contacts = _step_once(model)
    scene = solvers["scene"]
    test.assertTrue(scene._scene.features.depenetration_cap)
    free, ours, theirs, _rows = _normal_rows(solvers, contacts)
    test.assertGreater(free.size, 0)
    np.testing.assert_array_equal(theirs, np.float32(-DEPENETRATION_CAP))
    test.assertTrue(np.array_equal(ours, theirs), f"{ours} != {theirs}")
    count = int(contacts.rigid_contact_count.numpy()[0])
    dense = np.flatnonzero(solvers["steel"]._contact_path.numpy()[:count] == 0)
    test.assertGreater(dense.size, 0)
    rows = scene._contact_slot.numpy()[:count][dense]
    np.testing.assert_array_equal(scene._row_type.numpy()[0, rows], PGS_CONSTRAINT_TYPE_CONTACT)
    uncapped_solvers, _, _ = _step_once(_capped_boxes(device, (np.inf, np.inf), plate=True))
    uncapped = uncapped_solvers["scene"]
    test.assertFalse(uncapped._scene.features.depenetration_cap)
    uncapped_rows = uncapped._contact_slot.numpy()[:count][dense]
    test.assertTrue(np.array_equal(scene._rhs.numpy()[0, rows], uncapped._rhs.numpy()[0, uncapped_rows]))
    test.assertTrue(np.all(scene._rhs.numpy()[0, rows] < -DEPENETRATION_CAP), "the link's contacts are not deep")
    np.testing.assert_allclose(
        states["scene"].joint_qd.numpy(), states["steel"].joint_qd.numpy(), rtol=TRACK_RTOL, atol=TRACK_ATOL
    )


def _launch_cap(solver, contacts):
    """Launch :func:`cap_scene_depenetration` alone on ``solver``'s rows of its last step, as ``prepare_rows`` does."""
    wp.launch(
        cap_scene_depenetration,
        dim=contacts.rigid_contact_max,
        inputs=[
            contacts.rigid_contact_count,
            solver._contact_slot,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            solver._contact_art_a,
            solver._contact_art_b,
            solver.model.shape_body,
            solver._articulation_response_dof_count,
            solver._is_free_rigid,
            solver._rigid_body_max_depenetration_velocity,
            solver._row_type,
            solver._phi,
            solver._target_velocity,
            solver._row_rebound_velocity,
            solver._sparse_row_incident,
            solver._baumgarte,
            DT,
        ],
        outputs=[solver._rhs],
        device=solver.model.device,
    )


def test_depenetration_cap_writes_only_where_it_binds(test, device):
    """Write a free-body contact's right-hand side only where the cap binds; every other row keeps its own.

    Two free boxes 5 mm deep in each other. With a cap of 1 m/s on both, above their 0.24 m/s Baumgarte bias, the cap
    feature is on and never binds: every row's right-hand side after the step equals the uncapped scene's bit for bit,
    and the cap kernel launched alone on right-hand sides filled with a poison writes none. With 0.1 m/s on both it
    binds on every penetrating normal row: launched alone it writes exactly those rows, ``-target - max_depen``,
    Steel's value bit for bit, and leaves the poison on every other row.
    """
    uncapped, _, _ = _step_once(_capped_boxes(device, (np.inf, np.inf)))
    rows = int(uncapped["scene"]._constraint_count.numpy()[0])
    expected = uncapped["scene"]._rhs.numpy()[0, :rows]
    for caps, binds in (((LOOSE_CAP, LOOSE_CAP), False), ((DEPENETRATION_CAP, DEPENETRATION_CAP), True)):
        with test.subTest(caps=caps):
            solvers, _, contacts = _step_once(_capped_boxes(device, caps))
            scene = solvers["scene"]
            test.assertTrue(scene._scene.features.depenetration_cap)
            test.assertEqual(int(scene._constraint_count.numpy()[0]), rows)
            free, _, theirs, normal_rows = _normal_rows(solvers, contacts)
            test.assertGreater(free.size, 0)
            if not binds:
                test.assertTrue(np.array_equal(scene._rhs.numpy()[0, :rows], expected))
            scene._rhs.fill_(POISON)
            _launch_cap(scene, contacts)
            rhs = scene._rhs.numpy()[0]
            written = np.flatnonzero(rhs != POISON)
            if binds:
                test.assertEqual(sorted(written.tolist()), sorted(normal_rows.tolist()))
                test.assertTrue(np.array_equal(rhs[normal_rows], theirs), f"{rhs[normal_rows]} != {theirs}")
                np.testing.assert_array_equal(theirs, np.float32(-DEPENETRATION_CAP))
            else:
                test.assertEqual(written.size, 0, f"the loose cap wrote rows {written}")


def test_zero_jacobian_contact_rows_solve_as_steel(test, device):
    """Solve a slider link's ground contacts, whose normal rows have ``J = 0``, as Steel's general sweep does.

    A box on a slider along x sits 5 mm in the ground, a free box beside it pushing it (fixture A). The slider's
    ground contacts' normal rows cannot move its one DOF. Steel's local-owned kernels (the single, pair and residual
    pair owners) skip such rows through ``active``, but the fixture's world is not theirs: Steel solves it on its
    general matrix-free sweep, asserted here (``_local_solve_owner``). Its 12 free-body rows exclude the 20-row single
    and pair owners, which need ``mf_count == 0``, and the 40-row residual-pair owner needs the articulation's DOFs
    first, while the free box is added before the link; this is owner selection by structure, not the 24 dense rows.
    The general sweep gives each such row the diagonal ``cfm`` and the impulse ``0.24 m/s / cfm`` per iteration,
    which makes the friction box of its tangent pair unbounded; the scene's rows are the same, ``cfm`` bit for bit and
    the impulse within 1e-6 of Steel's.

    That impulse, about ``rhs * iterations / cfm``, makes the fixture's trajectory sensitive to which row has the last
    word, under every scene phase order. Steel sweeps the slider's ground rows, then the box's rows on the slider, then
    the box's ground rows (its free-body path), and the slider slides at about 0.11 m/s after the first step. The
    default ``"BA"`` sweeps the box's two-sided unit before the slider's ground rows and holds the slider; ``"AB"``
    sweeps both ground units before the box's rows on the slider, which is not Steel's order either: it matches the
    slider's speed within 1.3 % but scrambles the box. At step 30 Steel's slider is still again, as is the scene's,
    and the largest joint velocity difference from Steel is about 0.115 under ``"BA"`` and 0.44 under ``"AB"``.
    Alone (fixture C, no free body, Steel's general sweep), the slider started at 0.5 m/s is held still at steps 1 and
    30 by Steel and by the scene.
    """
    model = _sunk_slider(device)
    steel, steel_speeds, contacts, (steel_impulses, steel_diag), _ = _slider_speeds(model, 1, solve_mode="serial")
    test.assertEqual(int(steel._local_solve_owner.numpy()[0]), PGS_LOCAL_SOLVE_OWNER_GENERAL)
    scene, scene_speeds, _, (scene_impulses, scene_diag), _ = _slider_speeds(model, 1)
    count = int(contacts.rigid_contact_count.numpy()[0])
    link_shape = model.shape_count - 1
    shapes = np.stack([contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count]])
    ground = np.flatnonzero(np.any(shapes == link_shape, axis=0) & np.any(shapes == 0, axis=0))
    test.assertGreater(ground.size, 0)
    test.assertTrue(np.all(steel._contact_path.numpy()[ground] == 0))
    cfm = np.float32(scene._cfm)
    for contact in ground:
        ours, theirs = int(scene._contact_slot.numpy()[contact]), int(steel._contact_slot.numpy()[contact])
        world = int(steel._contact_world.numpy()[contact])
        test.assertTrue(np.all(scene._sparse_row_factor.numpy()[0, ours] == 0.0), "the normal row moves the slider")
        test.assertEqual(scene_diag[0, ours], cfm)
        test.assertEqual(steel_diag[world, theirs], cfm)
        test.assertGreater(steel_impulses[world, theirs], 1.0e5)
        np.testing.assert_allclose(scene_impulses[0, ours], steel_impulses[world, theirs], rtol=1.0e-6)
    test.assertGreater(steel_speeds[0], 0.05)
    test.assertLess(abs(scene_speeds[0]), 1.0e-6)
    _, ab_speeds, _, _, _ = _slider_speeds(model, 1, scene_phase_order="AB")
    np.testing.assert_allclose(ab_speeds[0], steel_speeds[0], rtol=0.05)
    # Step 30 of fixture A: the slider is still again in Steel and in the scene; the box keeps the order's mark.
    _, steel_speeds, _, _, steel_qd = _slider_speeds(model, SLIDER_STEPS, solve_mode="serial")
    _, scene_speeds, _, _, scene_qd = _slider_speeds(model, SLIDER_STEPS)
    _, _, _, _, ab_qd = _slider_speeds(model, SLIDER_STEPS, scene_phase_order="AB")
    test.assertLess(abs(steel_speeds[-1]), 1.0e-6)
    test.assertLess(abs(scene_speeds[-1]), 1.0e-6)
    default_gap = float(np.max(np.abs(scene_qd - steel_qd)))
    ab_gap = float(np.max(np.abs(ab_qd - steel_qd)))
    test.assertTrue(ORDER_GAP_BAND[0] < default_gap < ORDER_GAP_BAND[1], f'"BA" step-30 gap {default_gap}')
    test.assertGreater(ab_gap, 2.0 * default_gap, f'"AB" step-30 gap {ab_gap} vs "BA" {default_gap}')
    alone = _sunk_slider(device, box=False)
    _, steel_speeds, _, _, _ = _slider_speeds(alone, SLIDER_STEPS, speed=SLIDER_SPEED, solve_mode="serial")
    _, scene_speeds, _, _, _ = _slider_speeds(alone, SLIDER_STEPS, speed=SLIDER_SPEED)
    for step in (0, SLIDER_STEPS - 1):
        test.assertLess(abs(steel_speeds[step]), 1.0e-6, step + 1)
        test.assertLess(abs(scene_speeds[step]), 1.0e-6, step + 1)


class TestSteelSceneDepenetration(unittest.TestCase):
    pass


# On every device: Steel's reference is its serial solve (_steel_on), whose free-body rows cap alike.
for _fn in (
    test_depenetration_cap_tracks_steel,
    test_depenetration_cap_leaves_articulated_contacts,
    test_depenetration_cap_writes_only_where_it_binds,
    test_zero_jacobian_contact_rows_solve_as_steel,
):
    add_function_test(TestSteelSceneDepenetration, _fn.__name__, _fn, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
