# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Steel's friction modes (``friction_mode``) in the scene solve of SolverSteel (``solve_mode="scene"``).

Beside its default ``"fixed_normal"`` update (a contact's tangent pair on the disk of its current normal impulse),
Steel's matrix-free solve has three experimental contact updates that solve a contact's normal and tangent impulses
together at its first tangent row's visit: ``"bisection"`` bisects the normal impulse and re-solves the tangent pair at
each probe, on every contact; ``"bisection_desaxce"`` adds de Saxce's bias ``mu |c_T|`` to the normal target of its
free-body contacts; ``"coulomb_newton"`` solves a free-body contact's Coulomb cone exactly, its articulated contacts
keeping the default update. The scene flags each contact's update on its first tangent row at the row preparation
(``prepare_scene_friction_modes``: Steel's free-body path for a contact without an articulated side, its dense rows
otherwise) and its featured sweeps solve a flagged contact's three rows together (``SceneFeatures.friction_mode``).

The scene is checked against Steel's matrix-free solve within the tracking tolerances of ``test_scene_features.py``,
against its serial oracle bit for bit, against its eager steps under capture, and its flags against the expected set.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel import scene_pgs
from newton._src.solvers.steel.kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION
from newton.tests.steel._fixtures import _heap, _set_box_velocities
from newton.tests.steel._scene_fixtures import _humanoids, _rms, _Run, _scene_heap
from newton.tests.steel.test_scene_drives import DRIVE_TRACK_RTOL
from newton.tests.steel.test_scene_features import (
    CROWD_ROWS,
    FRICTION_ITERATIONS,
    FRICTION_POSITION_ITERATIONS,
    ONE_ROW_GAP,
    TRACK_ATOL,
    TRACK_ROWS,
    _assert_phases_match_serial_oracle,
    _track_steel,
)
from newton.tests.steel.test_scene_velocity_iterations import VELOCITY_ITERATIONS, _FixedRun
from newton.tests.steel.test_scene_velocity_limits import _free_body_dofs, _throw_free_bodies
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

MODES = ("bisection", "bisection_desaxce", "coulomb_newton")
"""Steel's friction modes other than its default ``"fixed_normal"``."""

ARTICULATED_MODES = ("bisection", "bisection_desaxce")
"""The modes whose update Steel's articulated (dense) contacts take too, through its dense bisection text."""

LAYER_PITCH = 0.25
"""Spacing [m] of the tracking test's layer of boxes (``_heap(nx=3, ny=3, nz=1)``): no two boxes meet, so each box's
contacts are with the ground alone."""

DROP_SPEED = 0.5
"""Downward speed [m/s] of the layer's boxes at the start: they meet the ground 1 mm below them in the first step."""

FREE_MODE_ATOL = 1.0e-5
"""Absolute tolerance of the free boxes' velocities after the first step against Steel's. The scene's and Steel's
free-body sums round apart (1.8e-6 at most under ``"fixed_normal"`` on the layer, measured), and
the modes' laws stop at absolute tolerances (Coulomb-Newton's ``1e-6 (1 + phi0)`` on its cone residual, the
bisection's last bracket), so differently rounded inputs stop them at different points: 7.2e-6 at most under
``"coulomb_newton"``, 1.8e-6 under ``"bisection"``."""

MODE_TRACK_Q_RMS = 1.0e-5
"""Agreement [RMS] of the joint coordinates with Steel's after :data:`TRACK_STEPS` steps. Tighter than the scene's
1e-3 convention (``test_scene_features.TRACK_RMS``), so that a mode's law is constrained: 1.1e-6 at most measured."""

FRICTIONLESS_RTOL = 1.0e-6
"""Bound of a frictionless contact's tangent impulses under ``"coulomb_newton"``, relative to the largest normal
impulse: Steel's Coulomb-Newton iteration stops within its tolerance of the zero cone, not at it (2.8e-9 against
normal impulses of 1.7 N s in both solvers, measured); the bisection projects them to 0."""

BIAS_WITNESS = 1.0
"""Least difference [m/s] of the layer's first velocities between ``"bisection_desaxce"`` and ``"bisection"``: the bias
``mu |c_T|`` of a box sliding at 1.5 m/s raises its normal target by about 0.75 m/s (11.9 m/s at most against 1.2
measured)."""

FLAG_STEPS = 10
"""Steps of the flag test: the humanoid falls 4 mm onto the ground, so its contacts go from one row to three."""

CAPTURE_STEPS = 30
"""Steps of the capture test."""

SLIDING_RTOL = 1.0e-4
"""A contact slides when its tangent impulse is within this fraction of its cone's radius ``mu * lambda_n``."""

PAIR_RADIUS = 0.05
"""Radius [m] of the two spheres of :func:`_sphere_pair`."""

PAIR_MU = 0.2
"""Friction coefficient of :func:`_sphere_pair`'s spheres: their tangential speed (2 m/s apart) passes the cone of
their normal impact (1 m/s), so the contact slides."""


def _expected_modes(solver, count, mode) -> np.ndarray:
    """The flags ``row_friction_mode`` should hold after a step under ``mode``, from the step's ``count`` contacts.

    Each contact with its tangent pair (``contact_slots_needed`` 3) flags its first tangent row: Steel's free-body
    update when every side is a free body or has no response DOFs, its dense one otherwise (the tangent pair's own
    update, ``SCENE_FRICTION_FIXED_NORMAL``, under ``"coulomb_newton"``). Every other row is not flagged.
    """
    slots = solver._contact_slot.numpy()[:count]
    needed = solver._contact_slots_needed.numpy()[:count]
    dofs = solver._articulation_response_dof_count.numpy()
    free = solver._is_free_rigid.numpy()

    def articulated(art):
        return art >= 0 and dofs[art] > 0 and free[art] == 0

    dense = scene_pgs.SCENE_FRICTION_DENSE if mode in ARTICULATED_MODES else scene_pgs.SCENE_FRICTION_FIXED_NORMAL
    expected = np.zeros(solver._scene.row_max, dtype=np.int32)
    for c, art_a, art_b in zip(
        range(count), solver._contact_art_a.numpy()[:count], solver._contact_art_b.numpy()[:count], strict=True
    ):
        if slots[c] < 0 or needed[c] != 3:
            continue
        on_free_path = not articulated(art_a) and not articulated(art_b)
        expected[slots[c] + 1] = scene_pgs.SCENE_FRICTION_FREE if on_free_path else dense
    return expected


def _flagged_rows(solver, kind=None) -> np.ndarray:
    """The first tangent rows the last step flagged (of ``kind`` when given) among its kept rows."""
    rows = int(solver._constraint_count.numpy()[0])
    flags = solver._scene.row_friction_mode.numpy()[0, :rows]
    return np.flatnonzero(flags != 0 if kind is None else flags == kind)


def _sliding_rows(solver, rows) -> np.ndarray:
    """Of the first tangent rows ``rows``, those whose contact slides: a loaded normal and a tangent impulse on the
    cone's rim, ``|t| >= (1 - SLIDING_RTOL) * mu * lambda_n``."""
    impulses = solver._impulses.numpy()[0]
    mu = solver._row_mu.numpy()[0]
    normal = impulses[rows - 1].astype(np.float64)
    tangent = np.hypot(impulses[rows].astype(np.float64), impulses[rows + 1].astype(np.float64))
    radius = mu[rows].astype(np.float64) * normal
    return rows[(normal > 0.0) & (radius > 0.0) & (tangent >= (1.0 - SLIDING_RTOL) * radius)]


def _two_sided(solver, rows) -> np.ndarray:
    """Of ``rows``, those in the two-sided units' range (phase B)."""
    begin, end = solver._scene.row_ranges.numpy()[scene_pgs.RANGE_TWO_SIDED]
    return rows[(rows >= begin) & (rows < end)]


def _mixed_scene(device):
    """One lying humanoid and a free 0.3 m box beside it on the ground, clear of it: articulated and free-body
    contacts in one world."""
    return _humanoids(device, props=True)


def _sphere_pair(device, *, articulated):
    """Two spheres 1 mm apart in the air, without gravity or ground: one contact, so one two-sided unit.

    The first is a free body, or with ``articulated`` a link on a six-axis D6 joint (Steel's dense rows, a hand on a
    free body); the second is a free body. Two spheres meet at one point, where two boxes meet at several contacts of
    one pair, several units whose Gauss-Seidel order would differ from Steel's.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.rigid_gap = 0.005
    shape = newton.ModelBuilder.ShapeConfig(mu=PAIR_MU)
    offset = PAIR_RADIUS + 0.0005
    first = wp.transform(wp.vec3(-offset, 0.0, 0.5), wp.quat_identity())
    if articulated:
        body = builder.add_link(xform=first)
        axes = [newton.ModelBuilder.JointDofConfig(axis=axis) for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)]
        joint = builder.add_joint_d6(-1, body, parent_xform=first, linear_axes=axes, angular_axes=axes)
        builder.add_articulation([joint])
    else:
        body = builder.add_body(xform=first)
    builder.add_shape_sphere(body, radius=PAIR_RADIUS, cfg=shape)
    other = builder.add_body(xform=wp.transform(wp.vec3(offset, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_sphere(other, radius=PAIR_RADIUS, cfg=shape)
    return builder.finalize(device=device)


def _collide_and_slide(model, state):
    """The spheres of :func:`_sphere_pair` approach at 0.5 m/s each and pass each other at 1 m/s each, tangentially."""
    qd = state.joint_qd.numpy()
    starts = model.joint_qd_start.numpy()
    for joint, sign in zip(range(model.joint_count), (1.0, -1.0), strict=True):
        qd[starts[joint] : starts[joint] + 2] = (0.5 * sign, 1.0 * sign)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def _frictionless(model):
    """``model`` with every shape's friction coefficient zero."""
    model.shape_material_mu.zero_()
    return model


def _slide_and_drop(model, state):
    """Every free box and articulated root slides at 1.5 m/s (``_set_box_velocities``), the free boxes also falling
    at :data:`DROP_SPEED`."""
    _set_box_velocities(model, state)
    qd = state.joint_qd.numpy()
    for start in _free_body_dofs(model):
        qd[start + 2] = -DROP_SPEED
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def test_friction_modes_track_steel(test, device):
    """Step each friction mode as Steel's matrix-free solve does: one step and thirty.

    Per mode: a layer of nine free boxes apart on the ground (``_heap(nx=3, ny=3, nz=1)``, 25 cm apart, every contact on
    Steel's free-body path), sliding at 1.5 m/s and dropped onto the ground at 0.5 m/s, so the contacts act in the first
    step; a lying humanoid beside a free box (articulated contacts beside free-body ones), both sliding at 1.5 m/s; and
    the same humanoid and box without friction. The velocities after the first step agree with Steel's within 1e-5 and
    :data:`FREE_MODE_ATOL` on the boxes, and within the factorizations' rounding on the humanoid
    (``test_scene_drives.DRIVE_TRACK_RTOL`` and 1e-6: under every mode Steel solves the humanoid on its dense rows, as
    under ``"fixed_normal"`` with dense factors, which differs from the scene's sparse factor as much); the joint
    coordinates after :data:`TRACK_STEPS` steps agree within :data:`MODE_TRACK_Q_RMS`. The mode acts: the box's
    contacts are flagged with Steel's free-body update after some step, the humanoid's with its dense one under the
    bisection modes (with the pair's update under ``"coulomb_newton"``, where Steel's dense rows keep it); without
    friction every tangent impulse is zero in both solvers (within :data:`FRICTIONLESS_RTOL` of the normal impulses
    under ``"coulomb_newton"``). The flags seen are gathered over every step.

    One more subtest steps the humanoid and the box under ``"bisection"`` with friction position iterations (the last
    2 of 6) and two velocity iterations, a combination Steel allows, at the same bounds (Steel refuses contact
    regularization beside a friction mode).

    The box heap of ``_heap(device)`` itself does not track Steel under any friction mode, ``"fixed_normal"`` included
    (1.5 m/s and more apart after one step, measured): its stacked boxes couple their contacts, and
    the scene sweeps them in another order than Steel's world (one-sided units side by side, two-sided units by color).
    The serial-oracle test sweeps a heap's two-sided units under each mode instead. De Saxce's bias acts on the layer:
    its first velocities differ from the bisection's by more than 1 m/s (the boxes rebound off the ground).
    """
    layer = {"init": _slide_and_drop, "atol": FREE_MODE_ATOL}
    layer_first = {}
    humanoid = {"rtol": DRIVE_TRACK_RTOL}
    scenes = {
        "layer": (lambda: _heap(device, nx=3, ny=3, nz=1, pitch=LAYER_PITCH), layer),
        "humanoid_box": (lambda: _mixed_scene(device), humanoid),
        "frictionless": (lambda: _frictionless(_mixed_scene(device)), humanoid),
    }
    for mode in MODES:
        for name, (build, options) in scenes.items():
            with test.subTest(friction_mode=mode, scene=name):
                seen = set()

                def flagged(solver, seen=seen):
                    rows = int(solver._constraint_count.numpy()[0])
                    seen.update(np.unique(solver._scene.row_friction_mode.numpy()[0, :rows]).tolist())
                    return len(seen) > 1

                scene, steel = _track_steel(test, build(), friction_mode=mode, witness=flagged, **options)
                test.assertEqual(scene.solver._scene.features.friction_mode, mode)
                test.assertLess(_rms(scene.state_0.joint_q.numpy(), steel.state_0.joint_q.numpy()), MODE_TRACK_Q_RMS)
                free, dense = scene_pgs.SCENE_FRICTION_FREE, scene_pgs.SCENE_FRICTION_DENSE
                if name == "layer":
                    test.assertEqual(seen, {0, free})
                    layer_first[mode] = scene.first_qd
                else:
                    test.assertEqual(seen, {0, free, dense} if mode in ARTICULATED_MODES else {0, free})
                if name == "frictionless":
                    for run in (scene, steel):
                        tangents, normals = _contact_values(run.solver)
                        if mode == "coulomb_newton":
                            # Coulomb-Newton stops at its tolerance of the zero cone (Steel's law, in both solvers).
                            bound = FRICTIONLESS_RTOL * float(np.max(normals))
                            test.assertLessEqual(float(np.max(np.abs(tangents))), bound)
                        else:
                            test.assertFalse(np.any(tangents))
    bias = np.max(np.abs(layer_first["bisection_desaxce"] - layer_first["bisection"]))
    test.assertGreater(bias, BIAS_WITNESS, "de Saxce's bias did not act on the free boxes")
    # Steel restricts the modes from patches, torsion, compliance, contact regularization and the propagation responses:
    # a mode beside the velocity pass and friction position iterations tracks Steel too.
    combined = {
        "iterations": FRICTION_ITERATIONS,
        "contact_friction_position_iterations": FRICTION_POSITION_ITERATIONS,
        "velocity_iterations": VELOCITY_ITERATIONS,
    }
    with test.subTest(friction_mode="bisection", scene="humanoid_box", **combined):
        seen = set()

        def flagged_combined(solver, seen=seen):
            rows = int(solver._constraint_count.numpy()[0])
            seen.update(np.unique(solver._scene.row_friction_mode.numpy()[0, :rows]).tolist())
            return len(seen) > 1

        scene, steel = _track_steel(
            test, _mixed_scene(device), friction_mode="bisection", witness=flagged_combined, **humanoid, **combined
        )
        features = scene.solver._scene.features
        test.assertEqual(features.friction_mode, "bisection")
        test.assertTrue(features.friction_position_iterations)
        test.assertEqual(scene.solver._scene.velocity_iterations, VELOCITY_ITERATIONS)
        test.assertEqual(seen, {0, scene_pgs.SCENE_FRICTION_FREE, scene_pgs.SCENE_FRICTION_DENSE})
        test.assertLess(_rms(scene.state_0.joint_q.numpy(), steel.state_0.joint_q.numpy()), MODE_TRACK_Q_RMS)


def test_friction_modes_track_steel_on_one_two_sided_unit(test, device):
    """Step each friction mode on one two-sided unit as Steel's matrix-free solve does: one step and thirty.

    Two spheres meeting in the air (:func:`_sphere_pair`): two free bodies (Steel's free-body path), and an articulated
    sphere on a free one (Steel's dense rows). One contact makes one two-sided unit, swept in phase B, so the scene's
    unit order cannot differ from Steel's rows. The spheres approach at 1 m/s and pass at 2 m/s, so the contact slides
    (its first tangent row on the cone's rim at some step, :func:`_sliding_rows`).
    The velocities after the first step agree with Steel's within 1e-5 and :data:`FREE_MODE_ATOL` on the free pair and
    within ``test_scene_drives.DRIVE_TRACK_RTOL`` and 1e-6 on the articulated one (Steel's dense rows against the
    scene's sparse factor), the coordinates after :data:`TRACK_STEPS` steps within :data:`MODE_TRACK_Q_RMS`. The
    contact is flagged with Steel's free-body update on the free pair, its dense one on the articulated pair under the
    bisection modes, and none under ``"coulomb_newton"`` (where Steel's dense rows keep the pair's update).
    """
    kinds = {"free": (False, {"atol": FREE_MODE_ATOL}), "articulated": (True, {"rtol": DRIVE_TRACK_RTOL})}
    for mode in MODES:
        for kind, (articulated, options) in kinds.items():
            with test.subTest(friction_mode=mode, pair=kind):
                seen, units, slid = set(), [], []

                def one_unit(solver, seen=seen, units=units, slid=slid):
                    rows = int(solver._constraint_count.numpy()[0])
                    seen.update(np.unique(solver._scene.row_friction_mode.numpy()[0, :rows]).tolist())
                    units.append(solver._scene_units.statistics()["two_sided"])
                    # The first tangent rows: a friction row right after its contact row.
                    kinds = solver._row_type.numpy()[0, :rows]
                    first = np.flatnonzero(
                        (kinds[1:] == PGS_CONSTRAINT_TYPE_FRICTION) & (kinds[:-1] == PGS_CONSTRAINT_TYPE_CONTACT)
                    )
                    slid.append(_sliding_rows(solver, _two_sided(solver, first + 1)).size)
                    return units[-1] > 0

                model = _sphere_pair(device, articulated=articulated)
                scene, steel = _track_steel(
                    test, model, init=_collide_and_slide, friction_mode=mode, witness=one_unit, **options
                )
                # Without a ground every contact unit is two-sided: one at most, the one contact.
                test.assertEqual(max(units), 1, units)
                test.assertGreater(max(slid), 0, "the contact never slid")
                if not articulated:
                    expected = {0, scene_pgs.SCENE_FRICTION_FREE}
                elif mode in ARTICULATED_MODES:
                    expected = {0, scene_pgs.SCENE_FRICTION_DENSE}
                else:
                    expected = {0}
                test.assertEqual(seen, expected)
                test.assertLess(_rms(scene.state_0.joint_q.numpy(), steel.state_0.joint_q.numpy()), MODE_TRACK_Q_RMS)


def _contact_values(solver) -> tuple[np.ndarray, np.ndarray]:
    """Every friction row's and every contact row's impulse of the solver's dense (or scene) and free-body rows."""
    tangents, normals = [], []
    for types, impulses in ((solver._row_type, solver._impulses), (solver._mf_row_type, solver._mf_impulses)):
        if types is None or impulses is None or types.shape != impulses.shape:
            continue
        kinds, values = types.numpy(), impulses.numpy()
        tangents.append(values[kinds == PGS_CONSTRAINT_TYPE_FRICTION])
        normals.append(values[kinds == PGS_CONSTRAINT_TYPE_CONTACT])
    return np.concatenate(tangents), np.concatenate(normals)


def test_friction_modes_flag_their_contacts(test, device):
    """Flag each contact's update on its first tangent row as Steel selects it, and no other row.

    The lying humanoid beside the box steps :data:`FLAG_STEPS` times under each mode with
    ``contact_friction_gap_max`` 2 mm: the humanoid's contacts start 4 mm above the ground with their normal row alone
    and get their tangent pair once it lands; the box rests on the ground. After every step ``row_friction_mode``
    equals the expected flags bit for bit (:func:`_expected_modes`, over every row): Steel's free-body update on the
    box's contacts, its dense bisection on the humanoid's under the bisection modes and the pair's update under
    ``"coulomb_newton"``, and nothing on a one-row contact or any other row. The steps see one-row contacts and both
    kinds of contact with their tangent pair. A one-row contact keeps the normal update under every mode, as Steel's
    lone normal row does: the steps track Steel's (its velocities after the first step within the factorizations'
    rounding, its coordinates after the last within :data:`MODE_TRACK_Q_RMS`).
    """
    for mode in MODES:
        with test.subTest(friction_mode=mode):
            model = _mixed_scene(device)
            options = {"friction_mode": mode, "contact_friction_gap_max": ONE_ROW_GAP}
            run = _Run(model, TRACK_ROWS, {}, **options)
            steel = _Run(model, TRACK_ROWS, {}, solve_mode="serial", **options)
            solver = run.solver
            needed_seen, kinds_seen = set(), set()
            for step in range(FLAG_STEPS):
                run.step()
                steel.step()
                if step == 0:
                    np.testing.assert_allclose(
                        run.state_0.joint_qd.numpy(),
                        steel.state_0.joint_qd.numpy(),
                        rtol=DRIVE_TRACK_RTOL,
                        atol=TRACK_ATOL,
                    )
                count = int(run.contacts.rigid_contact_count.numpy()[0])
                expected = _expected_modes(solver, count, mode)
                flags = solver._scene.row_friction_mode.numpy()[0]
                test.assertTrue(np.array_equal(flags, expected), f"step {step}: {np.flatnonzero(flags != expected)}")
                kept = solver._contact_slot.numpy()[:count] >= 0
                needed_seen.update(solver._contact_slots_needed.numpy()[:count][kept].tolist())
                kinds_seen.update(np.unique(expected).tolist())
            test.assertEqual(needed_seen, {1, 3})
            test.assertLess(_rms(run.state_0.joint_q.numpy(), steel.state_0.joint_q.numpy()), MODE_TRACK_Q_RMS)
            articulated = scene_pgs.SCENE_FRICTION_DENSE if mode in ARTICULATED_MODES else 0
            test.assertEqual(kinds_seen, {0, scene_pgs.SCENE_FRICTION_FREE, articulated})


def test_friction_modes_match_the_serial_oracle(test, device):
    """Solve the flagged contacts in the phased sweep exactly as one thread walking the rows does, in every order.

    Per mode: a small heap of free boxes thrown into one another (two-sided free-body units in phase B, sliding), and
    under the bisection modes also four lying humanoids 0.4 m apart with a free box inside the first (two-sided
    articulated units in phase B). The phases (the default color loop, and the while loop) match the serial oracle
    (``scene_sweep="serial"``) bit for bit over 30 steps in each phase order, with a flagged contact that slides in
    phase B (the heap) or a flagged two-sided articulated contact that slides (the humanoids).
    """
    heap = _scene_heap(device, nx=3, ny=2, nz=2, register_attributes=True)
    _throw_free_bodies(heap, heap)

    def sliding_free_contact(solver):
        flagged = _flagged_rows(solver, scene_pgs.SCENE_FRICTION_FREE)
        return bool(_sliding_rows(solver, _two_sided(solver, flagged)).size)

    def articulated_two_sided(solver):
        flagged = _flagged_rows(solver, scene_pgs.SCENE_FRICTION_DENSE)
        return bool(_sliding_rows(solver, _two_sided(solver, flagged)).size)

    for mode in MODES:
        with test.subTest(friction_mode=mode, scene="heap"):
            _assert_phases_match_serial_oracle(
                test,
                heap,
                {"friction_mode": mode},
                witness=sliding_free_contact,
                what="no flagged two-sided free-body contact slid",
            )
        if mode in ARTICULATED_MODES:
            with test.subTest(friction_mode=mode, scene="humanoids"):
                _assert_phases_match_serial_oracle(
                    test,
                    _humanoids(device, per_world=4, spacing=0.4, props=True),
                    {"friction_mode": mode},
                    witness=articulated_two_sided,
                    what="no flagged two-sided articulated contact slid",
                )


def test_friction_modes_replay_under_capture(test, device):
    """Replay one captured step under each mode exactly as eager steps.

    Four lying humanoids 0.4 m apart with a free box inside the first, every body sliding at 1.5 m/s: units join pairs
    of humanoids (phase B's conditional nodes) and the flags change with the contacts from step to step. After each of
    :data:`CAPTURE_STEPS` steps ``joint_q``, ``joint_qd``, the impulses and the flags equal the eager steps' bit for
    bit, some contact is flagged, and the flagged rows differ between replays.
    """
    for mode in MODES:
        with test.subTest(friction_mode=mode):
            model = _humanoids(device, per_world=4, spacing=0.4, props=True)
            eager, captured = runs = [_FixedRun(model, CROWD_ROWS, {}, friction_mode=mode) for _ in range(2)]
            for run in runs:
                _set_box_velocities(model, run.state_0)
                run.step()
            with wp.ScopedCapture(device) as recorded:
                captured.step()
            flagged, flag_sets = 0, set()
            for step in range(2, CAPTURE_STEPS + 1):
                eager.step()
                wp.capture_launch(recorded.graph)
                for name in ("joint_q", "joint_qd"):
                    a, b = getattr(eager.state_0, name).numpy(), getattr(captured.state_0, name).numpy()
                    test.assertTrue(np.array_equal(a, b), f"{name} differs after step {step}")
                for name in ("_impulses", "_row_type"):
                    a, b = getattr(eager.solver, name).numpy(), getattr(captured.solver, name).numpy()
                    test.assertTrue(np.array_equal(a, b), f"{name} differs after step {step}")
                test.assertTrue(
                    np.array_equal(
                        eager.solver._scene.row_friction_mode.numpy(), captured.solver._scene.row_friction_mode.numpy()
                    )
                )
                flagged = max(flagged, _flagged_rows(eager.solver).size)
                rows = _flagged_rows(captured.solver)
                flags = captured.solver._scene.row_friction_mode.numpy()[0, rows]
                flag_sets.add((tuple(rows.tolist()), tuple(flags.tolist())))
            test.assertGreater(flagged, 0)
            test.assertGreater(len(flag_sets), 1, "the flags never changed between replays")
            test.assertTrue(np.all(np.isfinite(captured.state_0.joint_q.numpy())))


class TestSteelSceneFrictionModes(unittest.TestCase):
    def test_unknown_friction_mode_is_rejected(self):
        """Refuse a featured kernel for a friction mode the sweep texts do not implement (``_feature_suffix``)."""
        with self.assertRaisesRegex(ValueError, "friction_mode must be 'fixed_normal' or one of .*got 'sticky'"):
            scene_pgs._feature_suffix(scene_pgs.SceneFeatures(friction_mode="sticky"))
        for mode in MODES:
            self.assertTrue(scene_pgs._feature_suffix(scene_pgs.SceneFeatures(friction_mode=mode)))


# CUDA only, each for the reason beside it: Steel's friction modes are CUDA's (its CPU solve rejects them, and so does
# the scene's on the CPU, where Steel's checks run first).
for _fn in (
    test_friction_modes_track_steel,  # Steel's friction modes
    test_friction_modes_track_steel_on_one_two_sided_unit,  # Steel's friction modes
    test_friction_modes_flag_their_contacts,  # Steel's friction modes
    test_friction_modes_match_the_serial_oracle,  # the phases; the CPU sweeps the serial oracle alone
    test_friction_modes_replay_under_capture,  # graph capture
):
    add_function_test(TestSteelSceneFrictionModes, _fn.__name__, _fn, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
