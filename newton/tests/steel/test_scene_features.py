# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-row features of the scene mode (``solve_mode="scene"``) against Steel's matrix-free solve.

Steel's sparse path excludes these features, so the scene is not compared with it bit for bit:
a stepped scene is compared with Steel's matrix-free solve (its dense and free-body rows) within a
tolerance, and per row the terms without ``J`` or ``Z`` (the coefficients, the targets, which rows
fire) bit for bit, while every term holding ``J v`` agrees within a tolerance.

The features are build constants of the scene's kernels (``SceneFeatures``, fixed at construction):
every feature off compiles the sources the scene tripwire pins, character for character.

Restitution: an impacting contact's rebound target ``rhs = -target - rebound``, Steel's law: the rebound velocity
``-e * u_start`` comes from the contact's step-start closing speed (``kernels.contact_rebound_velocity``, written by
``prepare_world_contact_rows`` into ``_row_rebound_velocity``, the same function Steel's dense and free-body row
builders call), and ``prepare_scene_rows`` decides whether it fires (``kernels.contact_restitution_fires``) from the
incident velocity ``J v_hat`` the response kernel stores (``_sparse_row_incident``), where Steel dots its dense or
free-body Jacobians with ``v_hat``.

One-row contacts: a contact beyond ``contact_friction_gap_max`` gets its normal row only (Steel's
gap gates, ``contact_slots_needed``), its unit holds that one row, and the response kernel writes it
alone.

Friction position iterations: with ``contact_friction_position_iterations`` the sweeps before
Steel's ``_friction_start_iteration`` skip the tangent pairs, and the stationary exit waits for it.

Contact regularization: with ``contact_regularization`` a penetrating contact's row takes the weight
``1 / (1 + g)``, written by ``prepare_scene_rows`` into ``SceneSolve.row_w`` as Steel's bias kernels write
theirs (1 for a contact whose rebound fires), and each sweep relaxes its impulse toward zero with the rest.

The feature kernels: every sweep factory has one featured definition beside its default, whose
arguments are the default's and every feature's (``friction_start``, ``row_w``, ``row_bound`` and the
four drive descriptors); a feature that is off gets a placeholder argument and adds no line to the
snippet. The phased sweep reproduces the serial oracle bit for bit with each feature's rows in phase B
as well as phase A.

Joint friction and the internal unit it joins are tested in ``test_scene_internal_unit.py``, drive rows
in ``test_scene_drives.py``.

Row watermarks: ``row_watermark`` counts the scene's rows per model world, as Steel counts each
world's rows, into Steel's high-water marks.

Pooled row budgets: the scene's row capacity is Steel's per-world budgets summed over the model's
worlds, so Steel's per-world overflow assertions cannot hold under the scene solve. Their scene-level
variants here overflow the pooled capacity and assert what the scene reports: the model worlds whose
units were dropped, ``scene_statistics()["dropped_rows"]``, its one overflow warning, the zero contact
forces of the dropped rows, and the watermarks' excess of the worlds that lost rows.
"""

import dataclasses
import hashlib
import re
import unittest
from functools import partial

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
)
from newton._src.solvers.steel.scene_pgs import (
    _HOST_ROUNDING_SHIM_CPP,
    _HOST_SHIM_CPP,
    RANGE_TWO_SIDED,
    SceneFeatures,
    get_sweep_block_kernel,
    get_sweep_color_kernel,
    get_sweep_order_serial_kernel,
    get_sweep_rows_serial_kernel,
    get_sweep_sides_kernel,
    get_sweep_torsion_kernel,
    scene_kernel_snippets,
)
from newton._src.solvers.steel.scene_rows import _get_scene_contact_response_kernel
from newton.solvers import SolverBase, SolverSteel
from newton.tests.steel._fixtures import DT, _build_contact_scene, _friction_impulses, _set_box_velocities
from newton.tests.steel._scene_fixtures import (
    DEFAULT_SNIPPET_SHA256,
    SCENE,
    _drop_sphere,
    _humanoids,
    _rms,
    _Run,
    assert_rebound_tracks_steel,
)
from newton.tests.steel.test_capacity import _box_on_ground
from newton.tests.steel.test_contacts import FILL, LOWER_MASS, _check_rows, _net_body_forces, _stacked_boxes
from newton.tests.unittest_utils import StdOutCapture, add_function_test, get_cuda_test_devices, get_test_devices

needs_read_back = unittest.skipUnless(hasattr(SolverSteel, "scene_statistics"), "needs the read-back API")
"""The mark of the tests that step the scene mode: they run once the solver has the scene mode and its read-back."""

RESTITUTION_RHS_ATOL = 1.0e-6
"""Absolute tolerance of a contact row's ``rhs`` against Steel's free-body row that does not fire: its Baumgarte or
speculative bias ``phi / dt``, which Steel's ``compute_mf_effective_mass_and_rhs`` divides and the scene, as Steel's
dense ``compute_world_contact_bias``, multiplies by ``1 / dt``, 1 ulp apart. A row whose rebound fires is Steel's bit
for bit (``-target - rebound`` from the same rebound velocity), as is a dense row that does not."""

TRACK_ROWS = 64
"""``articulation_constraint_max`` of the contact scene's runs: a world's base and arm need about 21 rows."""

TRACK_STEPS = 30
"""Steps of the tests that track Steel."""

TRACK_RTOL = 1.0e-5
TRACK_ATOL = 1.0e-6
"""Tolerances of the velocities after the first step against Steel's matrix-free solve."""

TRACK_RMS = 1.0e-3
"""RMS bound of the joint coordinates after :data:`TRACK_STEPS` steps against Steel's."""

ONE_ROW_GAP = 0.002
"""``contact_friction_gap_max`` [m] of the one-row test.

The contact scene's boxes rest 1 mm deep, so their contacts keep friction rows; the arm's speculative
contacts, 16 cm and more above the ground, get a normal row only.
"""

FRICTION_ITERATIONS = 6
FRICTION_POSITION_ITERATIONS = 2
"""The friction gate of the tests: friction in the last 2 of 6 position iterations, from the fifth sweep."""

CROWD_ROWS = 4 * 128
"""Rows per world of four lying humanoids, Steel's 128 each."""

WATERMARK_STEPS = 10
WATERMARK_WORLDS = 3
HUMANOID_ROWS = 128
"""Steel's rows per world of one humanoid."""

TINY_BUDGET = {"articulation_constraint_max": 1, "free_body_constraint_max": 1}
"""Per-world row budgets whose sum over the worlds is still too small for one contact's three rows per world."""

UNIT_MAX = 64
"""``scene_unit_max`` of the overflow tests: a unit for every contact, so every loss is a row drop.

The default sizes the units from the pooled row capacity (a third of it), which these budgets make so
small that most units would pass ``scene_unit_max`` instead of being dropped for rows.
"""

OVERFLOW_STEPS = 5
"""Steps of the overflow warning and watermark tests."""

OVERFLOW_WARNING = re.compile(
    r"^Warning: Steel scene rows overflow: (\d+) units requested (\d+) rows, limit (\d+); kept (\d+) rows\. "
    r"(\d+) further units passed scene_unit_max\."
)
"""The scene solve's overflow ``printf``, with its counts: units, rows requested, row capacity, rows kept, unit
overflow."""

CONTACT_F_STEPS = 4
"""Steps of the scene-level CONTACT_F overflow test: the boxes, without rows, fall about 1.4 mm and keep their
contacts."""

SENTINEL = -5.0
"""Fill of the response kernel's row outputs past the scene's rows."""

REGULARIZATION = 0.5
"""``contact_regularization`` of the regularization tests: a penetrating contact's row weighs ``1 / 1.5``."""

STATIONARY_ITERATIONS = 16
"""Iterations of the stationary-exit test: friction waits for the fifteenth sweep, with
:data:`FRICTION_POSITION_ITERATIONS` friction position iterations."""

SPHERE_RADIUS = 0.05
SPHERE_MU = 0.5
"""The sliding sphere of the stationary-exit test [m], and the friction of its shape and of the ground."""

FEATURE_SETS = {
    "_fpi": SceneFeatures(friction_position_iterations=True),
    "_reg": SceneFeatures(regularize=True),
    "_fpi_reg": SceneFeatures(friction_position_iterations=True, regularize=True),
    "_jf": SceneFeatures(joint_friction=True),
    "_reg_jf": SceneFeatures(regularize=True, joint_friction=True),
    "_fpi_reg_jf": SceneFeatures(friction_position_iterations=True, regularize=True, joint_friction=True),
    "_drv": SceneFeatures(drives=True),
    "_jf_drv": SceneFeatures(joint_friction=True, drives=True),
    "_fpi_reg_jf_drv": SceneFeatures(
        friction_position_iterations=True, regularize=True, joint_friction=True, drives=True
    ),
    "_late": SceneFeatures(late_rows=True),
    "_drv_late": SceneFeatures(drives=True, late_rows=True),
    "_drv_late_fvc": SceneFeatures(drives=True, late_rows=True, fused_clamp=True),
    "_fpi_reg_jf_drv_late_fvc": SceneFeatures(
        friction_position_iterations=True,
        regularize=True,
        joint_friction=True,
        drives=True,
        late_rows=True,
        fused_clamp=True,
    ),
    "_dpc": SceneFeatures(depenetration_cap=True),
    "_fpi_reg_jf_drv_late_fvc_dpc": SceneFeatures(
        friction_position_iterations=True,
        regularize=True,
        joint_friction=True,
        drives=True,
        late_rows=True,
        fused_clamp=True,
        depenetration_cap=True,
    ),
    "_cmp": SceneFeatures(compliance=True),
    "_fpi_reg_jf_drv_late_fvc_dpc_cmp": SceneFeatures(
        friction_position_iterations=True,
        regularize=True,
        joint_friction=True,
        drives=True,
        late_rows=True,
        fused_clamp=True,
        depenetration_cap=True,
        compliance=True,
    ),
    "_tor": SceneFeatures(torsion=True),
    "_fpi_reg_jf_drv_late_fvc_dpc_tor_cmp": SceneFeatures(
        friction_position_iterations=True,
        regularize=True,
        joint_friction=True,
        drives=True,
        late_rows=True,
        fused_clamp=True,
        depenetration_cap=True,
        torsion=True,
        compliance=True,
    ),
    "_bis": SceneFeatures(friction_mode="bisection"),
    "_bisx": SceneFeatures(friction_mode="bisection_desaxce"),
    "_cn": SceneFeatures(friction_mode="coulomb_newton"),
    "_fpi_reg_jf_drv_late_fvc_dpc_tor_cmp_bisx": SceneFeatures(
        friction_position_iterations=True,
        regularize=True,
        joint_friction=True,
        drives=True,
        late_rows=True,
        fused_clamp=True,
        depenetration_cap=True,
        torsion=True,
        compliance=True,
        friction_mode="bisection_desaxce",
    ),
    "_pat": SceneFeatures(patches=True),
    "_fpi_reg_jf_drv_late_fvc_dpc_tor_cmp_bisx_pat": SceneFeatures(
        friction_position_iterations=True,
        regularize=True,
        joint_friction=True,
        drives=True,
        late_rows=True,
        fused_clamp=True,
        depenetration_cap=True,
        torsion=True,
        compliance=True,
        friction_mode="bisection_desaxce",
        patches=True,
    ),
}
"""Sets of the implemented per-row features, each feature alone and together, by the name suffix of their kernels.

The fused velocity clamp (``fused_clamp``) runs on drive rows in phase C, so it comes with ``drives`` and
``late_rows``. The depenetration cap (``depenetration_cap``) is the row preparation's: it adds no argument and no
line to the sweeps. Contact compliance (``compliance``) adds ``row_gamma`` and rewrites one line of each row loop,
its right-hand side. Contact torsion (``torsion``) adds C1 to every row loop and runs phase C without late rows too;
its token comes before ``_cmp`` (field order), its arguments after ``row_gamma``. A friction mode (``friction_mode``,
``_bis``, ``_bisx`` or ``_cn`` by its value) adds ``row_friction_mode`` after torsion's arguments, its device functions
before the row loops and the combined update of a flagged contact to every row loop; Steel rejects it beside torsion
and compliance, which the all-on set composes anyway. Friction patches (``patches``, ``_pat``) add no argument and
turn each row loop's normal load of a tangent pair into its region's ring sum; Steel rejects them beside a friction
mode and compliance, which the all-on set composes anyway."""

FEATURE_ARGS = [
    "friction_start",
    "row_w",
    "row_bound",
    "drive_target_vel_bias",
    "drive_vel_multiplier",
    "drive_impulse_multiplier",
    "drive_max_impulse",
    "side_late_begin",
    "side_late_end",
    "drive_vel_limit",
    "freeze_drives",
    "row_gamma",
    "torsion_radius",
    "torsion_group",
    "torsion_member_range",
    "torsion_member_rows",
    "side_torsion_begin",
    "side_torsion_end",
    "torsion_color_start",
    "row_friction_mode",
]
"""The featured kernels' arguments after the default's, every feature's in order."""

RIGID_RHS = "const float rhs = rows_rhs.data[row] + rows_incident.data[row];"
COMPLIANT_RHS = (
    "const float rhs = rows_rhs.data[row] + row_gamma.data[row] * impulses.data[row] + rows_incident.data[row];"
)
"""A row's right-hand side in its update, and the compliant one: ``gamma * lambda`` with the row's impulse joins it,
Steel's ``update_compliant_rhs`` (``gamma`` is 0 on every other row)."""

FREEZE_LINE = "if (freeze_drives && row_type == JOINT_TARGET) continue;"
"""The drive rows' branch's first line: the velocity-only iterations under ``velocity_drive_mode="freeze"`` skip the
row (Steel's ``freeze_drive_rows``)."""

ROW_ARGS = [
    "rows_dof",
    "rows_factor",
    "rows_free_response",
    "rows_free_prefix",
    "rows_rhs",
    "rows_incident",
    "rows_diag",
    "rows_type",
    "rows_parent",
    "rows_mu",
    "omega",
    "delta",
    "impulses",
]
"""The last arguments of every default sweep kernel: the rows, the relaxation, delta and the impulses."""


@needs_read_back
def test_restitution_rebound_tracks_steel(test, device):
    """Rebound a falling sphere as Steel's matrix-free solve does, through the impact and the flight after it.

    The sphere falls 5 cm under gravity onto the plane, restitution 0.5 on both shapes: it strikes at
    about 1 m/s, above the 0.5 m/s threshold, and leaves at half that speed, then strikes again at
    0.57 m/s and leaves at 0.29 m/s. Steel solves the contact on its free-body rows, the scene on a
    row of its free group, in physical coordinates. The vertical velocity agrees with Steel's within
    1e-4 of Steel's at each of 60 steps, and the rebound is positive.
    """
    runs, velocities = _drop_sphere(device, restitution=0.5)
    assert_rebound_tracks_steel(test, runs, velocities)


def _set_drop_velocities(model, state, speed=1.0):
    """Push some bodies of the contact scene into the ground at ``speed``; the others rest.

    World 0: the first free box and the articulated base (its d6 joint's Z axis); world 1: the second
    free box. So both of Steel's row families, and both worlds, have contacts that fire and contacts that
    do not.
    """
    pushed = {(0, 0), (0, 2), (1, 1)}
    joints_per_world = model.joint_count // model.world_count
    joint_type = model.joint_type.numpy()
    starts = model.joint_qd_start.numpy()
    joint_qd = state.joint_qd.numpy()
    for joint in range(model.joint_count):
        if (joint // joints_per_world, joint % joints_per_world) not in pushed:
            continue
        if joint_type[joint] == int(newton.JointType.FREE):
            joint_qd[starts[joint] + 2] = -speed
        elif joint_type[joint] == int(newton.JointType.D6):
            joint_qd[starts[joint] + 1] = -speed
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def _normal_rows(test, solver, count):
    """Each contact's normal row: its family (0 dense or scene, 1 Steel's free-body rows), ``rhs``, rebound velocity and
    target.

    Contacts without rows get family ``-1``. Steel keeps free-body targets only for prescribed motion
    (``_mf_target_velocity`` is a 1 x 1 stand-in otherwise), so a free-body row without one reads ``0``.
    """
    path = solver._contact_path.numpy()[:count]
    slot = solver._contact_slot.numpy()[:count]
    world = solver._contact_world.numpy()[:count]
    mf_target = solver._mf_target_velocity.numpy()
    if mf_target.shape != solver._mf_rhs.shape:
        mf_target = np.zeros(solver._mf_rhs.shape, dtype=np.float32)
    families = {
        0: (
            solver._row_type.numpy(),
            solver._rhs.numpy(),
            solver._row_rebound_velocity.numpy(),
            solver._target_velocity.numpy(),
        ),
        1: (solver._mf_row_type.numpy(), solver._mf_rhs.numpy(), solver._mf_row_rebound_velocity.numpy(), mf_target),
    }
    family = np.where(slot >= 0, path, -1)
    rhs = np.zeros(count, dtype=np.float32)
    rebound = np.zeros(count, dtype=np.float32)
    target = np.zeros(count, dtype=np.float32)
    for c in np.flatnonzero(family >= 0):
        row_type, family_rhs, family_rebound, family_target = families[int(family[c])]
        test.assertEqual(row_type[world[c], slot[c]], PGS_CONSTRAINT_TYPE_CONTACT)
        rhs[c] = family_rhs[world[c], slot[c]]
        rebound[c] = family_rebound[world[c], slot[c]]
        target[c] = family_target[world[c], slot[c]]
    return family, rhs, rebound, target


@needs_read_back
def test_restitution_rows_fire_where_steel_fires(test, device):
    """Give the scene's contacts the rebound targets Steel's matrix-free rows get, on the same contacts.

    On the contact scene with restitution 0.4, some bodies are pushed into the ground at 1 m/s (above
    the 0.5 m/s threshold) and the others rest. Each solver steps once with restitution and once with
    ``enable_restitution=False`` from the same state and contacts: the contacts whose normal row's
    ``rhs`` differs between the two carry a rebound target. That set is Steel's, on its dense rows and
    on its free-body rows alike, and contains contacts of both, while the resting contacts are in
    neither. The rebound velocities (``-e * u_start`` from the step-start closing speed, positive on the
    firing contacts) and the targets match Steel's bit for bit (the targets are zero: the ground is
    static and nothing is prescribed), and so does the ``rhs`` of every firing row, Steel's
    ``-target - rebound``. On the dense rows that do not fire it is the bias alone, Steel's bit for bit.
    On Steel's free-body rows that do not fire it is the bias alone too, but not bitwise: Steel's
    ``compute_mf_effective_mass_and_rhs`` divides ``beta * phi`` by ``dt`` where the scene, as Steel's
    dense ``compute_world_contact_bias``, multiplies it by ``1 / dt``, which may round 1 ulp apart
    (``RESTITUTION_RHS_ATOL``).
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
        for enabled in (True, False):
            solver = SolverSteel(model, **{**SCENE, **options}, enable_restitution=enabled)
            test.assertEqual(solver.solve_mode == "scene", name == "scene")
            solver.step(initial_state(), model.state(), model.control(), contacts, DT)
            test.assertFalse(np.any(solver.constraint_overflow.numpy()), f"{name} overflowed")
            rows[name, enabled] = _normal_rows(test, solver, count)

    scene_family, scene_rhs, scene_rebound, scene_target = rows["scene", True]
    steel_family, steel_rhs, steel_rebound, steel_target = rows["steel", True]
    with_rows = steel_family >= 0
    np.testing.assert_array_equal(scene_family >= 0, with_rows)
    np.testing.assert_array_equal(scene_family[with_rows], 0)
    # Steel routes the free boxes' contacts to its free-body rows and the base's to its dense rows.
    test.assertEqual(set(steel_family[with_rows].tolist()), {0, 1})
    np.testing.assert_array_equal(scene_rebound[with_rows], steel_rebound[with_rows])
    np.testing.assert_array_equal(scene_target[with_rows], steel_target[with_rows])
    np.testing.assert_array_equal(steel_target[with_rows], np.float32(0.0))

    fires = {name: with_rows & (rows[name, True][1] != rows[name, False][1]) for name in ("scene", "steel")}
    np.testing.assert_array_equal(fires["scene"], fires["steel"])
    for family in (0, 1):
        in_family = steel_family == family
        test.assertTrue(np.any(fires["steel"] & in_family), f"no contact fires on Steel's family {family}")
        test.assertTrue(np.any(~fires["steel"] & in_family), f"every contact fires on Steel's family {family}")
    np.testing.assert_allclose(scene_rhs[with_rows], steel_rhs[with_rows], rtol=0.0, atol=RESTITUTION_RHS_ATOL)
    # The firing rows take Steel's rebound target from the same rebound velocity: bit for bit, on both families.
    test.assertTrue(np.all(steel_rebound[fires["steel"]] > 0.0))
    np.testing.assert_array_equal(scene_rhs[fires["steel"]], steel_rhs[fires["steel"]])
    np.testing.assert_array_equal(
        steel_rhs[fires["steel"]], -steel_target[fires["steel"]] - steel_rebound[fires["steel"]]
    )
    # The dense rows that do not fire keep the bias alone, from phi: Steel's bit for bit.
    resting = with_rows & ~fires["steel"] & (steel_family == 0)
    np.testing.assert_array_equal(scene_rhs[resting], steel_rhs[resting])


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_default_features_compile_todays_source(test, device):
    """Compile every feature off to the sources the scene tripwire pins, character for character.

    ``scene_kernel_snippets`` with ``SceneFeatures()`` equals the call without ``features`` and hashes
    to ``DEFAULT_SNIPPET_SHA256``. A feature that is on changes the sweep kernels' sources: friction
    position iterations gate them on ``friction_start``, and leave the contact response's alone.
    """
    default = scene_kernel_snippets(30, free_response=False, dof_max=27)
    featured = scene_kernel_snippets(30, free_response=False, dof_max=27, features=SceneFeatures())
    test.assertEqual(featured, default)
    for key, snippet in featured.items():
        test.assertEqual(_sha256(snippet), DEFAULT_SNIPPET_SHA256[f"30/False/{key}"], key)
    gated = scene_kernel_snippets(
        30, free_response=False, dof_max=27, features=SceneFeatures(friction_position_iterations=True)
    )
    test.assertEqual(gated["response"], default["response"])
    for key in ("sides", "serial", "order_serial", "color", "block"):
        test.assertNotEqual(gated[key], default[key], key)
        test.assertIn("friction_start", gated[key], key)
        test.assertNotIn("friction_start", default[key], key)


def _within_track_tolerance(first_a, first_b, last_a, last_b, *, rtol=TRACK_RTOL, atol=TRACK_ATOL) -> bool:
    """Whether two runs agree as :func:`_track_steel` requires of the scene and Steel.

    ``first_*`` are the velocities after the first step, within ``rtol`` and ``atol`` (:data:`TRACK_RTOL` and
    :data:`TRACK_ATOL` by default) of ``first_b``, and ``last_*`` the joint coordinates after
    :data:`TRACK_STEPS` steps, within :data:`TRACK_RMS` RMS.
    """
    first = np.allclose(first_a, first_b, rtol=rtol, atol=atol)
    return bool(first) and _rms(last_a, last_b) < TRACK_RMS


def _track_steel(
    test,
    model,
    *,
    init=_set_box_velocities,
    rows=TRACK_ROWS,
    rtol=TRACK_RTOL,
    atol=TRACK_ATOL,
    witness=None,
    runner=_Run,
    **options,
):
    """Step the scene solve and Steel's matrix-free solve side by side from ``init``'s state.

    Both take :data:`SCENE` with ``options`` and ``rows`` rows per world (:data:`TRACK_ROWS` by default), each a
    ``runner`` (:class:`_Run` by default, or a subclass that prepares the contacts).
    The velocities after the first step agree within ``rtol`` and ``atol`` (:data:`TRACK_RTOL` and
    :data:`TRACK_ATOL` by default), the joint coordinates after :data:`TRACK_STEPS` steps within
    :data:`TRACK_RMS` RMS, and neither solver overflows. ``witness(solver)``, when given, is called with the
    scene's solver after every step and must hold after some step (a witness may accumulate what it sees).
    Returns the two runs, scene first; each keeps its velocities after the first step in ``first_qd``.
    """
    scene = runner(model, rows, {}, **options)
    steel = runner(model, rows, {}, solve_mode="serial", **options)
    test.assertEqual(scene.solver.solve_mode, "scene")
    test.assertIsNone(steel.solver._scene)
    for run in (scene, steel):
        init(model, run.state_0)
    seen = witness is None
    for step in range(TRACK_STEPS):
        scene.step()
        steel.step()
        if witness is not None:
            seen = witness(scene.solver) or seen
        if step == 0:
            # Copies: on the CPU numpy() is a view of the state, which later steps overwrite.
            qd_scene, qd_steel = scene.state_0.joint_qd.numpy().copy(), steel.state_0.joint_qd.numpy().copy()
            scene.first_qd, steel.first_qd = qd_scene, qd_steel
            test.assertGreater(np.max(np.abs(qd_steel)), 0.0)
            np.testing.assert_allclose(qd_scene, qd_steel, rtol=rtol, atol=atol)
    test.assertTrue(seen, "the witness never held")
    q_scene, q_steel = scene.state_0.joint_q.numpy(), steel.state_0.joint_q.numpy()
    test.assertTrue(np.all(np.isfinite(q_scene)))
    test.assertLess(_rms(q_scene, q_steel), TRACK_RMS)
    for run in (scene, steel):
        test.assertFalse(np.any(run.solver.constraint_overflow.numpy()))
    return scene, steel


@needs_read_back
def test_one_row_contacts_track_steel(test, device):
    """Solve the contacts beyond ``contact_friction_gap_max`` with their normal row alone, as Steel does.

    On the contact scene with the boxes sliding at 1.5 m/s, a 2 mm gap keeps the friction rows of the
    boxes' contacts, 1 mm deep, and gives the arm's speculative contacts a normal row only (Steel's gap
    gates write ``contact_slots_needed``). Their units hold that one row, so the scene keeps fewer than
    three rows per unit, and the response kernel writes it alone, leaving the next unit's rows intact.
    The velocities after one step agree with Steel's matrix-free solve within 1e-5 (and 1e-6), and the
    joint coordinates after 30 steps within 1e-3 RMS. A contact unit may have one row, so the scene
    sizes its units as one per row of the budget.
    """
    model = _build_contact_scene(device)
    scene, _steel = _track_steel(test, model, contact_friction_gap_max=ONE_ROW_GAP)
    solver = scene.solver
    test.assertTrue(solver._scene.one_row_contacts)
    stats = solver.scene_statistics()
    test.assertLess(stats["rows"], 3 * stats["units"])
    test.assertEqual(stats["dropped_rows"], 0)
    count = int(scene.contacts.rigid_contact_count.numpy()[0])
    kept = solver._contact_slot.numpy()[:count] >= 0
    slots = solver._contact_slots_needed.numpy()[:count][kept]
    test.assertEqual(set(slots.tolist()), {1, 3})
    test.assertEqual(solver._scene_units.unit_max, stats["row_max"] + solver._scene_units.limit_unit_count)


@needs_read_back
def test_friction_position_iterations_track_steel(test, device):
    """Solve friction in the last position iterations only, as Steel's ``contact_friction_position_iterations``.

    With 6 iterations and 2 friction position iterations the tangent pairs wait for the fifth sweep
    (Steel's ``_friction_start_iteration``), and the stationary exit with them. On the contact scene with
    the boxes sliding at 1.5 m/s, the velocities after one step agree with Steel's matrix-free solve
    within 1e-5 (and 1e-6), and the joint coordinates after 30 steps within 1e-3 RMS.
    """
    model = _build_contact_scene(device)
    scene, _steel = _track_steel(
        test, model, iterations=FRICTION_ITERATIONS, contact_friction_position_iterations=FRICTION_POSITION_ITERATIONS
    )
    test.assertTrue(scene.solver._scene.features.friction_position_iterations)
    test.assertEqual(scene.solver._scene.friction_start, FRICTION_ITERATIONS - FRICTION_POSITION_ITERATIONS)


@needs_read_back
def test_friction_position_iterations_gate_the_scene_sweep(test, device):
    """Gate the tangent pairs on the sweep: from the first sweep as without the gate, never with ``0``.

    On the contact scene with the boxes sliding, friction position iterations of at least the 6
    iterations start friction at the first sweep, so the gated kernels step bit for bit as the default
    ones, whose source has no gate. ``0`` never solves friction: every tangent pair's impulse stays zero
    while the scene has friction rows. ``3`` solves it in the last three sweeps and differs from both.
    Solvers with equal features share their sweep kernels; the gate selects other ones.
    """
    model = _build_contact_scene(device)

    def run(**options):
        result = _Run(model, TRACK_ROWS, {}, iterations=FRICTION_ITERATIONS, **options)
        _set_box_velocities(model, result.state_0)
        friction = []
        for _ in range(12):
            result.step()
            friction.append(_friction_impulses(result.solver))
        return result, friction

    default, _ = run()
    test.assertFalse(default.solver._scene.features.friction_position_iterations)
    for k in (FRICTION_ITERATIONS, FRICTION_ITERATIONS + 5):
        with test.subTest(contact_friction_position_iterations=k):
            gated, _ = run(contact_friction_position_iterations=k)
            test.assertEqual(gated.solver._scene.friction_start, 0)
            test.assertIsNot(gated.solver._scene.sides_kernel, default.solver._scene.sides_kernel)
            for name in ("joint_q", "joint_qd"):
                test.assertTrue(
                    np.array_equal(getattr(gated.state_0, name).numpy(), getattr(default.state_0, name).numpy()), name
                )
    none, friction = run(contact_friction_position_iterations=0)
    test.assertEqual(none.solver._scene.friction_start, FRICTION_ITERATIONS)
    test.assertTrue(any(f.size for f in friction), "the scene has no friction rows")
    for step, impulses in enumerate(friction):
        np.testing.assert_array_equal(impulses, 0.0, err_msg=f"step {step}")
    late, friction = run(contact_friction_position_iterations=3)
    test.assertTrue(any(np.any(f != 0.0) for f in friction), "three friction iterations solved no friction")
    for other in (none, default):
        test.assertGreater(float(np.abs(late.state_0.joint_qd.numpy() - other.state_0.joint_qd.numpy()).max()), 0.0)
    # Equal features share the cached kernels; the gate is part of the cache key.
    test.assertIs(late.solver._scene.sides_kernel, none.solver._scene.sides_kernel)
    again = _Run(model, TRACK_ROWS, {}, iterations=FRICTION_ITERATIONS, collide=False)
    test.assertIs(again.solver._scene.sides_kernel, default.solver._scene.sides_kernel)


def _assert_phases_match_serial_oracle(test, model, options, *, witness=None, what="", runner=_Run):
    """Step the phased sweep as one thread walking the rows does, bit for bit, in every phase order.

    ``model`` is a crowd whose units join pairs of articulations (asserted): the coupled sides' launches
    and phase B run. In each phase order the serial oracle (``scene_sweep="serial"``) steps
    :data:`TRACK_STEPS` times with ``options``; the phases (the default color loop, and the while loop,
    whose tail one thread walks) then match its ``joint_q``, ``joint_qd`` and impulses after every step.
    ``witness(solver)``, when given, must hold after some step of each oracle run (``what`` names it
    otherwise), so that the feature under test acts. Every run is a ``runner`` (:class:`_Run` by default, or a
    subclass that prepares the contacts).
    """

    def state(run):
        rows = int(run.solver._constraint_count.numpy()[0])
        return run.state_0.joint_q.numpy(), run.state_0.joint_qd.numpy(), run.solver._impulses.numpy()[0, :rows]

    for order in ("BA", "AB", "ABA"):
        oracle = runner(model, CROWD_ROWS, {"scene_sweep": "serial"}, scene_phase_order=order, **options)
        history = []
        two_sided = 0
        seen = witness is None
        for _ in range(TRACK_STEPS):
            oracle.step()
            history.append(state(oracle))
            two_sided = max(two_sided, oracle.solver.scene_statistics()["two_sided_units"])
            seen = seen or witness(oracle.solver)
        test.assertGreater(two_sided, 0, order)
        test.assertTrue(seen, f"{order}: {what}")
        test.assertTrue(np.all(np.isfinite(history[-1][0])), order)
        for overrides in ({}, {"scene_color_loop": "while"}):
            with test.subTest(scene_phase_order=order, **overrides):
                run = runner(model, CROWD_ROWS, overrides, scene_phase_order=order, **options)
                for step, expected in enumerate(history, start=1):
                    run.step()
                    for field, a, b in zip(("joint_q", "joint_qd", "impulses"), expected, state(run), strict=True):
                        test.assertTrue(np.array_equal(a, b), msg=f"{field} at step {step}")


@needs_read_back
def test_friction_position_iterations_match_the_serial_oracle(test, device):
    """Gate friction in the phased sweep exactly as one thread walking the rows does, in every phase order.

    Four lying humanoids 0.4 m apart overlap, so units join pairs of them: the coupled sides' launches
    each take their iteration's first friction sweep, and the isolated sides' launch gates its own loop.
    With 6 iterations and 2 friction position iterations, the phases (the default color loop, and the
    while loop, whose tail one thread walks) match the serial oracle (``scene_sweep="serial"``) bit for
    bit over 30 steps in each phase order.
    """
    model = _humanoids(device, per_world=4, spacing=0.4)
    options = {"iterations": FRICTION_ITERATIONS, "contact_friction_position_iterations": FRICTION_POSITION_ITERATIONS}
    _assert_phases_match_serial_oracle(test, model, options)


@needs_read_back
def test_regularization_matches_the_serial_oracle(test, device):
    """Weigh the penetrating contacts in the phased sweep exactly as one thread walking the rows does.

    Four lying humanoids 0.4 m apart overlap by 4 cm, so units join pairs of them and their contacts
    penetrate: with ``contact_regularization=0.5`` rows of weight ``1 / 1.5`` run through phase B (the
    colors, the block and the tail) and the coupled sides' sweeps, not only through phase A as on the
    contact scene. The phases (the default color loop, and the while loop, whose tail one thread walks)
    match the serial oracle (``scene_sweep="serial"``) bit for bit over 30 steps in each phase order.
    """
    model = _humanoids(device, per_world=4, spacing=0.4)

    def two_sided_weighed(solver):
        begin, end = solver._scene.row_ranges.numpy()[RANGE_TWO_SIDED]
        return bool(np.any(solver._scene.row_w.numpy()[0, begin:end] < 1.0))

    _assert_phases_match_serial_oracle(
        test,
        model,
        {"contact_regularization": REGULARIZATION},
        witness=two_sided_weighed,
        what="no two-sided row was regularized",
    )


@needs_read_back
def test_row_watermark_counts_rows_per_model_world(test, device):
    """Accumulate the row high-water marks per model world, as Steel counts each world's rows.

    Three worlds of a lying humanoid fold into the scene's one solve world, whose row count is their
    sum; the scene counts each unit's rows in its side's model world instead. After 10 steps every
    mark (the rows kept and requested, the dropped contact rows, the excess and the overflowing
    world-steps of each row family, and the contact count) equals Steel's, which counts per world.
    The articulation marks are positive and below the scene's row count. On the contact scene the
    free boxes' contacts count in Steel's free-body family, as Steel routes them to its free-body
    rows, and the articulated base's and arm's in its articulation family: both marks equal Steel's.
    """
    scenes = {
        "humanoids": (_humanoids(device, worlds=WATERMARK_WORLDS), HUMANOID_ROWS),
        "contact_scene": (_build_contact_scene(device), TRACK_ROWS),
    }
    for name, (model, rows) in scenes.items():
        with test.subTest(scene=name):
            scene = _Run(model, rows, {}, row_watermark=True)
            steel = _Run(model, rows, {}, solve_mode="serial", row_watermark=True)
            total = 0
            for _ in range(WATERMARK_STEPS):
                scene.step()
                steel.step()
                total = max(total, int(scene.solver._constraint_count.numpy()[0]))
            marks = scene.solver._row_watermarks.numpy()
            test.assertTrue(
                np.array_equal(marks, steel.solver._row_watermarks.numpy()),
                f"{scene.solver.constraint_row_watermarks()} != {steel.solver.constraint_row_watermarks()}",
            )
            named = scene.solver.constraint_row_watermarks()
            test.assertGreater(named["articulation_high_water"], 0)
            test.assertLess(named["articulation_high_water"], total)
            test.assertGreater(named["contact_high_water"], 0)
            if name == "contact_scene":
                test.assertGreater(named["free_body_high_water"], 0)
            else:
                test.assertEqual(named["free_body_high_water"], 0)


def _contact_model_worlds(model, contacts) -> np.ndarray:
    """Each contact's model world: its non-global shape's (the ground is world -1)."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape_world = model.shape_world.numpy()
    return np.maximum(
        shape_world[contacts.rigid_contact_shape0.numpy()[:count]],
        shape_world[contacts.rigid_contact_shape1.numpy()[:count]],
    )


def _world_rows(model, solver, contacts):
    """Per model world, the contact rows of the last step kept and dropped: ``contact_slots_needed`` per contact."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    worlds = _contact_model_worlds(model, contacts)
    slots = solver._contact_slots_needed.numpy()[:count]
    dropped = solver._contact_slot.numpy()[:count] < 0
    kept_rows = np.zeros(model.world_count, dtype=np.int64)
    dropped_rows = np.zeros(model.world_count, dtype=np.int64)
    np.add.at(kept_rows, worlds[~dropped], slots[~dropped])
    np.add.at(dropped_rows, worlds[dropped], slots[dropped])
    return kept_rows, dropped_rows


@needs_read_back
def test_scene_capacity_failure_is_observable(test, device):
    """Flag the model worlds whose units the pooled row capacity dropped, until reset.

    Steel's ``test_capacity_failure_is_observable`` on its box on the ground, under the scene solve with point
    friction (``friction_anchor_beta=0``: three rows per contact) and one articulation row and one free-body row per
    world, a pooled capacity of two rows per world.
    With the default unit capacity, a third of the rows (one unit), the box's one stored unit is dropped
    for rows and the others pass ``scene_unit_max``: every status entry, the global one too, is set.
    With :data:`UNIT_MAX` units every loss is a row drop, in one world and in two (where world 0 keeps
    one contact's three rows, the first unit in row order, and world 1 none): ``constraint_overflow`` is
    set exactly for the model worlds with a dropped contact, the global entry stays clear, and
    ``scene_statistics()["dropped_rows"]`` equals the dropped contacts' rows. In each case after one
    step ``check_constraint_capacity`` raises, ``dropped_rows`` is positive, and a reset clears the
    status.
    """
    for worlds, unit_max in ((1, None), (1, UNIT_MAX), (2, UNIT_MAX)):
        with test.subTest(worlds=worlds, scene_unit_max=unit_max):
            model = _box_on_ground(device, worlds=worlds)
            # Point friction: the kept rows count three per contact.
            solver = SolverSteel(
                model,
                solve_mode="scene",
                scene_unit_max=unit_max,
                warn_constraint_overflow=False,
                friction_anchor_beta=0.0,
                **TINY_BUDGET,
            )
            test.assertEqual(solver.solve_mode, "scene")
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            state_in, state_out = model.state(), model.state()
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, model.control(), contacts, DT)
            with test.assertRaisesRegex(RuntimeError, "capacity"):
                solver.check_constraint_capacity()
            stats = solver.scene_statistics()
            test.assertEqual(stats["row_max"], 2 * worlds)
            test.assertGreater(stats["dropped_rows"], 0)
            if unit_max is None:
                test.assertGreater(stats["unit_overflow"], 0)
                test.assertTrue(np.all(solver.constraint_overflow.numpy()))
            else:
                test.assertEqual(stats["unit_overflow"], 0)
                kept, dropped = _world_rows(model, solver, contacts)
                test.assertEqual(int(dropped.sum()), stats["dropped_rows"])
                test.assertEqual(int(kept.sum()), stats["rows"])
                test.assertTrue(np.all(dropped > 0), dropped)
                np.testing.assert_array_equal(kept, [0] if worlds == 1 else [3, 0])
                np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [*(dropped > 0), False])
            solver.reset(state_out)
            test.assertFalse(np.any(solver.constraint_overflow.numpy()))
            solver.check_constraint_capacity()


@needs_read_back
def test_scene_overflow_warning_is_printed_once(test, device):
    """Print the scene's one device-side overflow warning in the first overflowing step, and none after.

    Steel's ``test_overflow_warning_is_printed_once`` on its two boxes on the ground, under the scene
    solve with one articulation row and one free-body row per world and :data:`UNIT_MAX` units: the
    pooled capacity of four rows drops units for rows in each of 5 steps. Each step's output is
    captured on its own; the overflow ``printf`` appears exactly once over the 5 steps, alone in the
    first step's output, with that step's units, rows requested, row capacity, rows kept and unit
    overflow (``scene_statistics()``).
    """
    model = _box_on_ground(device, worlds=2)
    solver = SolverSteel(
        model, solve_mode="scene", scene_unit_max=UNIT_MAX, warn_constraint_overflow=True, **TINY_BUDGET
    )
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    step_output = []
    first = None
    for step in range(OVERFLOW_STEPS):
        capture = StdOutCapture()
        capture.begin()
        try:
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, model.control(), contacts, DT)
        finally:
            # Flush the device printf into this capture, also when the step raises.
            try:
                wp.synchronize()
            finally:
                output = capture.end()
        step_output.append(
            [
                line
                for line in output.splitlines()
                if line.strip() and not (line.startswith("Module") and "load on device" in line)
            ]
        )
        stats = solver.scene_statistics()
        test.assertGreater(stats["dropped_rows"], 0, f"step {step}")
        test.assertEqual(stats["unit_overflow"], 0, f"step {step}")
        if first is None:
            first = stats
        state_in, state_out = state_out, state_in
    warnings_printed = [line for lines in step_output for line in lines if OVERFLOW_WARNING.match(line)]
    test.assertEqual(len(warnings_printed), 1, step_output)
    test.assertEqual(len(step_output[0]), 1, step_output[0])
    match = OVERFLOW_WARNING.match(step_output[0][0])
    test.assertIsNotNone(match, step_output[0])
    expected = (first["units"], first["rows_requested"], first["row_max"], first["rows"], first["unit_overflow"])
    test.assertEqual(tuple(int(value) for value in match.groups()), expected)
    for later in step_output[1:]:
        test.assertEqual(later, [])


@needs_read_back
def test_scene_contact_f_reports_each_box_weight_row_overflow(test, device):
    """Report each kept contact's force and zero on every contact the pooled row capacity dropped.

    Steel's ``test_contact_f_reports_each_box_weight`` row-overflow variant on its two stacked boxes, under the scene
    solve with point friction (``friction_anchor_beta=0``: three rows per contact), one articulation row and
    :data:`UNIT_MAX` units, so every loss is a row drop. The observable is filled with a sentinel and the boxes step 4
    times. In both subtests the live rows hold a finite linear force and no torque, the rows past the contact count are
    zero, every dropped contact of each box reads exactly zero, ``dropped_rows`` is ``rows_requested - rows > 0``, no
    unit passes ``scene_unit_max``, and the overflow flag is set on the global entry only, as the stacked boxes are
    world -1 bodies.

    - ``all dropped``: one free-body row (two rows, the one world's pooled capacity) holds no contact's three rows, so
      the scene keeps no row and every contact, box-box ones included, reads zero.
    - ``one kept``: two free-body rows (three rows), the smallest budget that keeps a contact unit, keep exactly one:
      the first unit in row order, a ground contact of the lower box (one-sided units come before the two-sided box-box
      units, so no budget keeps a box-box contact while dropping a ground contact). ``rows`` equals its three slots.
      Its force is its rows' impulses over ``dt``, Steel's observable: the force on shape 0's body pushes it away from
      shape 1, against the contact normal (which points from shape 0 to shape 1), by the normal row's impulse, and the
      tangential magnitude is the two friction rows', both within 1e-5 relative. It is the only force on the lower box,
      upward, and the upper box gets none. Steel's original checks each box's weight after 150 steps at rest; one kept
      ground contact does not hold the stack at rest (it tips about that corner and the upper box falls through), so
      the kept contact carries less than the lower box's weight, which is asserted instead.
    """
    for label, free in (("all dropped", 1), ("one kept", 2)):
        with test.subTest(label):
            model, pipeline = _stacked_boxes(device)
            # Point friction: a contact unit is three rows, so the budgets keep none or exactly one.
            solver = SolverSteel(
                model,
                solve_mode="scene",
                scene_unit_max=UNIT_MAX,
                iterations=32,
                warn_constraint_overflow=False,
                articulation_constraint_max=1,
                free_body_constraint_max=free,
                friction_anchor_beta=0.0,
            )
            test.assertEqual(solver.solve_mode, "scene")
            contacts = pipeline.contacts()
            observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
            observables.contact_f.fill_(wp.spatial_vector(FILL, FILL, FILL, FILL, FILL, FILL))
            states = [model.state(), model.state()]
            control = model.control()
            for _ in range(CONTACT_F_STEPS):
                pipeline.collide(states[0], contacts)
                solver.step(states[0], states[1], control, contacts, DT, observables=observables)
                states.reverse()
            contact_f = observables.contact_f.numpy()
            _check_rows(test, contacts, contact_f)
            count = int(contacts.rigid_contact_count.numpy()[0])
            slot = solver._contact_slot.numpy()[:count]
            dropped = slot < 0
            shape_body = model.shape_body.numpy()
            bodies = np.stack(
                [
                    shape_body[contacts.rigid_contact_shape0.numpy()[:count]],
                    shape_body[contacts.rigid_contact_shape1.numpy()[:count]],
                ]
            )
            for body in range(model.body_count):
                touches = np.any(bodies == body, axis=0)
                test.assertTrue(np.any(touches & dropped), f"no dropped contact of box {body}")
                np.testing.assert_array_equal(contact_f[:count][touches & dropped], 0.0, err_msg=f"box {body}")
            # The box-box contacts, between two free bodies, are dropped too.
            test.assertTrue(np.any(dropped & np.all(bodies >= 0, axis=0)))
            stats = solver.scene_statistics()
            test.assertEqual(stats["row_max"], 1 + free)
            test.assertEqual(stats["unit_overflow"], 0)
            test.assertEqual(stats["dropped_rows"], stats["rows_requested"] - stats["rows"])
            test.assertGreater(stats["dropped_rows"], 0)
            # The stacked boxes are global (world -1) bodies: their dropped units latch the last, global entry.
            np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [False, True])
            if label == "all dropped":
                test.assertTrue(np.all(dropped))
                test.assertEqual(stats["rows"], 0)
                continue
            kept = np.flatnonzero(~dropped)
            test.assertEqual(len(kept), 1)
            c = int(kept[0])
            test.assertEqual(int(slot[c]), 0)
            test.assertEqual(
                sorted(bodies[:, c].tolist()), [-1, 0], "the kept contact is not the lower box's ground contact"
            )
            test.assertEqual(stats["rows"], int(solver._contact_slots_needed.numpy()[c]))
            test.assertEqual(stats["rows"], 3)
            impulses = solver._impulses.numpy()[0, :3] / DT
            force = contact_f[c, :3].astype(np.float64)
            normal = contacts.rigid_contact_normal.numpy()[c].astype(np.float64)
            normal_force = float(np.dot(force, normal))
            tangential_force = float(np.linalg.norm(force - normal_force * normal))
            test.assertLess(normal_force, 0.0)
            np.testing.assert_allclose(-normal_force, impulses[0], rtol=1.0e-5)
            np.testing.assert_allclose(tangential_force, np.linalg.norm(impulses[1:]), rtol=1.0e-5)
            net = _net_body_forces(model, contacts, contact_f)
            np.testing.assert_array_equal(net[1], 0.0)
            weight = float(np.linalg.norm(model.gravity.numpy()[0]))
            test.assertGreater(net[0, 2], 0.0)
            test.assertLess(net[0, 2], LOWER_MASS * weight)


@needs_read_back
def test_scene_watermark_excess_counts_dropped_rows(test, device):
    """Record as a world's excess the rows its dropped units asked for, and none for a world that lost none.

    Two boxes on the ground in two worlds under the scene solve with point friction (``friction_anchor_beta=0``: three
    rows per contact), ``row_watermark`` and :data:`UNIT_MAX` units: the pooled capacity holds world 0's contact rows
    (at most one row more) and none of world 1's, which come after them in row order. Steel's excess is over each
    world's own budget; the scene pools the budgets, so a world's excess is the rows of its dropped units. Over 5 steps,
    from each step's kept and dropped contacts per model world (``contact_slots_needed`` each): the free-body marks
    (both boxes' contacts are free-body rows) hold the most rows a world kept and requested, the most a world dropped as
    its excess and dropped contact rows, and one overflowing world-step per world and step with a dropped contact, so
    world 0, which lost none, adds none. The dropped rows per step sum to ``scene_statistics()["dropped_rows"]``, and
    the articulation family records nothing.
    """
    model = _box_on_ground(device, worlds=2)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    worlds = _contact_model_worlds(model, contacts)
    world_rows = 3 * np.bincount(worlds, minlength=model.world_count)
    test.assertGreater(world_rows[0], 0)
    # Two worlds of (1 + free) rows each hold world 0's rows and at most one more.
    free = (int(world_rows[0]) + 1) // 2 - 1
    # Point friction: three rows per contact, as world_rows counts them.
    solver = SolverSteel(
        model,
        solve_mode="scene",
        row_watermark=True,
        scene_unit_max=UNIT_MAX,
        warn_constraint_overflow=False,
        articulation_constraint_max=1,
        free_body_constraint_max=free,
        friction_anchor_beta=0.0,
    )
    # The scene's row capacity, the budgets summed over the two worlds.
    test.assertGreaterEqual(solver._articulation_constraint_max, int(world_rows[0]))
    test.assertLess(solver._articulation_constraint_max, int(world_rows.sum()))
    state_in, state_out = model.state(), model.state()
    kept_marks = requested_marks = excess_marks = world_steps = 0
    lost_one_world_only = False
    for step in range(OVERFLOW_STEPS):
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, model.control(), contacts, DT)
        state_in, state_out = state_out, state_in
        kept, dropped = _world_rows(model, solver, contacts)
        stats = solver.scene_statistics()
        test.assertEqual(stats["unit_overflow"], 0, f"step {step}")
        test.assertEqual(int(dropped.sum()), stats["dropped_rows"], f"step {step}")
        kept_marks = max(kept_marks, int(kept.max()))
        requested_marks = max(requested_marks, int((kept + dropped).max()))
        excess_marks = max(excess_marks, int(dropped.max()))
        world_steps += int(np.count_nonzero(dropped))
        lost_one_world_only |= bool(dropped[0] == 0 and dropped[1] > 0)
    test.assertTrue(lost_one_world_only, "no step lost rows in world 1 alone")
    test.assertGreater(excess_marks, 0)
    marks = solver.constraint_row_watermarks()
    test.assertEqual(marks["free_body_high_water"], kept_marks)
    test.assertEqual(marks["free_body_raw_high_water"], requested_marks)
    test.assertEqual(marks["free_body_overflow_excess_high_water"], excess_marks)
    test.assertEqual(marks["free_body_dropped_contact_rows_high_water"], excess_marks)
    test.assertEqual(marks["free_body_overflow_world_steps"], world_steps)
    test.assertLess(world_steps, model.world_count * OVERFLOW_STEPS)
    for name in ("high_water", "raw_high_water", "overflow_excess_high_water", "overflow_world_steps"):
        test.assertEqual(marks[f"articulation_{name}"], 0, name)


def _row_owners(test, solver, contacts, rows):
    """Each of the scene's ``rows`` rows' contact (``-1`` for the others), from the kept contacts' slots and row
    counts."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    slot = solver._contact_slot.numpy()[:count]
    slots_needed = solver._contact_slots_needed.numpy()[:count]
    owner = np.full(rows, -1, dtype=np.int64)
    for c in np.flatnonzero(slot >= 0):
        span = slice(int(slot[c]), int(slot[c] + slots_needed[c]))
        test.assertTrue(np.all(owner[span] == -1), f"contact {c} shares its rows")
        owner[span] = c
    return owner


@needs_read_back
def test_one_row_contacts_use_the_slots_response(test, device):
    """Give a contact past ``contact_friction_gap_max`` one row, which the response kernel writes alone.

    On the contact scene with a 2 mm gap, after one step the kept contacts' rows (``contact_slots_needed``
    from each one's slot, 1 or 3) and the joint-limit rows partition the scene's rows exactly: a contact's
    first row is its normal row, its other two its friction rows, and every other row a joint-limit row.
    On two lying humanoids every contact is 4 mm or more above the ground, past the gap, so every unit is
    one normal row and, without joint limits, the last row is a contact's. The response kernel's outputs
    are filled with a sentinel past the rows; a kernel writing three rows per contact would write two
    rows past the last contact's, so the sentinel must survive.
    """
    model = _build_contact_scene(device)
    run = _Run(model, TRACK_ROWS, {}, contact_friction_gap_max=ONE_ROW_GAP)
    run.step()
    solver = run.solver
    test.assertTrue(solver._scene.one_row_contacts)
    test.assertEqual(solver._scene.response_kind, "generic")
    test.assertTrue(solver._scene.response_kernel.key.endswith("_slots"), solver._scene.response_kernel.key)
    rows = solver.scene_statistics()["rows"]
    test.assertEqual(solver.scene_statistics()["dropped_rows"], 0)
    owner = _row_owners(test, solver, run.contacts, rows)
    row_type = solver._row_type.numpy()[0, :rows]
    slot = solver._contact_slot.numpy()
    slots_needed = solver._contact_slots_needed.numpy()
    kept = np.unique(owner[owner >= 0])
    test.assertEqual(set(slots_needed[kept].tolist()), {1, 3})
    test.assertEqual(rows, int(slots_needed[kept].sum()) + int(np.count_nonzero(owner < 0)))
    np.testing.assert_array_equal(row_type[owner < 0], PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
    for c in kept:
        start = int(slot[c])
        test.assertEqual(row_type[start], PGS_CONSTRAINT_TYPE_CONTACT, f"contact {c}")
        np.testing.assert_array_equal(row_type[start + 1 : start + slots_needed[c]], PGS_CONSTRAINT_TYPE_FRICTION)

    humanoids = _humanoids(device, worlds=2)
    run = _Run(humanoids, HUMANOID_ROWS, {}, contact_friction_gap_max=ONE_ROW_GAP, enable_joint_limits=False)
    solver = run.solver
    test.assertTrue(solver._scene.one_row_contacts)
    outputs = (solver._sparse_row_incident, solver._diag, solver._sparse_row_factor)
    for array in outputs:
        array.fill_(SENTINEL)
    solver._sparse_row_dof.fill_(int(SENTINEL))
    run.step()
    stats = solver.scene_statistics()
    rows = stats["rows"]
    test.assertGreater(rows, 1)
    test.assertEqual(stats["dropped_rows"], 0)
    test.assertLessEqual(rows + 2, stats["row_max"])
    owner = _row_owners(test, solver, run.contacts, rows)
    test.assertTrue(np.all(owner >= 0), "a row that no contact holds")
    test.assertEqual(set(solver._contact_slots_needed.numpy()[np.unique(owner)].tolist()), {1})
    np.testing.assert_array_equal(solver._row_type.numpy()[0, :rows], PGS_CONSTRAINT_TYPE_CONTACT)
    for array in outputs:
        np.testing.assert_array_equal(array.numpy()[0, rows:], SENTINEL)
    for array in outputs[:2]:
        # Every row's incident velocity and diagonal were written.
        test.assertFalse(np.any(array.numpy()[0, :rows] == SENTINEL))
    np.testing.assert_array_equal(solver._sparse_row_dof.numpy()[0, rows:], int(SENTINEL))


@needs_read_back
def test_specialized_response_rejects_one_row_contacts(test, device):
    """Refuse the specialised response kernel, which writes three rows per contact, with one-row contacts.

    Lying humanoids (one factor group) take ``scene_response_kernel="specialized"``, but with
    ``contact_friction_gap_max`` a contact may have one row, and with friction patches (an omitted
    ``friction_anchor_beta``) every contact that is no anchor has one: construction raises ``ValueError``
    naming the cause and telling the caller to use ``'generic'``.
    """
    model = _humanoids(device, worlds=2)
    accepted = _Run(model, HUMANOID_ROWS, {"scene_response_kernel": "specialized"}, collide=False)
    test.assertEqual(accepted.solver._scene.response_kind, "specialized")
    test.assertFalse(accepted.solver._scene.one_row_contacts)
    with test.assertRaisesRegex(ValueError, r"contact_friction_gap_max\); use 'generic'"):
        _Run(model, HUMANOID_ROWS, {"scene_response_kernel": "specialized"}, contact_friction_gap_max=ONE_ROW_GAP)
    with test.assertRaisesRegex(ValueError, r"friction patches\); use 'generic'"):
        _Run(model, HUMANOID_ROWS, {"scene_response_kernel": "specialized"}, friction_anchor_beta=None)


@needs_read_back
def test_regularization_tracks_steel(test, device):
    """Regularize the penetrating contacts as Steel's matrix-free solve does, ``contact_regularization=0.5``.

    On the contact scene with the boxes sliding at 1.5 m/s, the boxes' contacts, 1 mm deep, weigh
    ``1 / (1 + g)``: each sweep moves their impulse toward the rigid solution with that weight and toward
    zero with the rest. The velocities after one step agree with Steel's matrix-free solve within 1e-5
    (and 1e-6), and the joint coordinates after 30 steps within 1e-3 RMS. The scene's ``row_w`` is
    ``[1, row_max]``. The same scene without regularization falls outside that tolerance of Steel's
    regularized solve, so the test would see a weight that changes nothing.
    """
    model = _build_contact_scene(device)
    scene, steel = _track_steel(test, model, contact_regularization=REGULARIZATION)
    solve = scene.solver._scene
    test.assertTrue(solve.features.regularize)
    test.assertEqual(solve.row_w.shape, (1, solve.row_max))
    rigid = _Run(model, TRACK_ROWS, {})
    test.assertIsNone(rigid.solver._scene.row_w)
    _set_box_velocities(model, rigid.state_0)
    first = None
    for step in range(TRACK_STEPS):
        rigid.step()
        if step == 0:
            first = rigid.state_0.joint_qd.numpy()
    test.assertFalse(
        _within_track_tolerance(first, steel.first_qd, rigid.state_0.joint_q.numpy(), steel.state_0.joint_q.numpy()),
        "the rigid scene tracks Steel's regularized solve",
    )


def _contact_weights(test, solver, count):
    """Each contact's normal-row family (as :func:`_normal_rows`), ``phi`` and regularization weight.

    The scene's weight is its ``row_w``; Steel's is its dense ``_row_w`` or its free-body ``_mf_row_w``.
    Contacts without rows get family ``-1``.
    """
    path = solver._contact_path.numpy()[:count]
    slot = solver._contact_slot.numpy()[:count]
    world = solver._contact_world.numpy()[:count]
    dense_w = solver._scene.row_w.numpy() if solver.solve_mode == "scene" else solver._row_w.numpy()
    families = {
        0: (solver._row_type.numpy(), solver._phi.numpy(), dense_w),
        1: (solver._mf_row_type.numpy(), solver._mf_phi.numpy(), solver._mf_row_w.numpy()),
    }
    family = np.where(slot >= 0, path, -1)
    phi = np.zeros(count, dtype=np.float32)
    weight = np.zeros(count, dtype=np.float32)
    for c in np.flatnonzero(family >= 0):
        row_type, family_phi, family_w = families[int(family[c])]
        test.assertEqual(row_type[world[c], slot[c]], PGS_CONSTRAINT_TYPE_CONTACT)
        phi[c] = family_phi[world[c], slot[c]]
        weight[c] = family_w[world[c], slot[c]]
    return family, phi, weight


ARM_LIMIT = 0.05
"""Half-width [rad] of the finite limits the weights test gives the contact scene's hinge, about its rest angle."""


def _limit_hinges(model, half_width):
    """Limit every REVOLUTE DOF of ``model`` to ``half_width`` either side of its rest coordinate, in place."""
    lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
    joint_q = model.joint_q.numpy()
    q_starts, qd_starts = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    for joint in np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.REVOLUTE)):
        dof, coordinate = qd_starts[joint], q_starts[joint]
        lower[dof] = joint_q[coordinate] - half_width
        upper[dof] = joint_q[coordinate] + half_width
    model.joint_limit_lower.assign(lower)
    model.joint_limit_upper.assign(upper)


def _limit_weights(solver):
    """The regularization weights of the joint-limit rows of the last step, world by world in row order."""
    dense_w = solver._scene.row_w.numpy() if solver.solve_mode == "scene" else solver._row_w.numpy()
    row_type = solver._row_type.numpy()
    weights = []
    for world, rows in enumerate(solver._constraint_count.numpy()):
        weights.append(dense_w[world, :rows][row_type[world, :rows] == PGS_CONSTRAINT_TYPE_JOINT_LIMIT])
    return np.concatenate(weights)


@needs_read_back
def test_row_weights_match_steel_bitwise(test, device):
    """Weigh each contact's row as Steel's dense and free-body rows do, bit for bit.

    On the contact scene with restitution 0.4 and ``contact_regularization=0.5``, some bodies are pushed
    into the ground at 1 m/s and the others rest (as in :func:`test_restitution_rows_fire_where_steel_fires`).
    After one step from the same state and contacts, the scene's ``row_w`` at each contact's normal row
    equals the weight Steel gives its dense row (``compute_world_contact_bias`` and
    ``apply_world_contact_restitution``) or its free-body row (``compute_mf_effective_mass_and_rhs``):
    ``1 / (1 + g)`` for a penetrating contact, 1 for a speculative one and for one whose rebound fires, so
    an impact's rebound stays exact. Penetrating contacts that do not fire, and penetrating contacts that
    fire, occur on both of Steel's families. The arm's hinge gets finite limits 0.05 rad either side of its
    rest angle, within the activation gap, so every world has joint-limit rows: every other row of the
    scene, friction and joint limit, weighs 1, as Steel's limit rows do. The weight holds no ``J``, so the
    comparison is ``np.array_equal``.
    """
    model = _build_contact_scene(device, restitution=0.4)
    _limit_hinges(model, ARM_LIMIT)

    def initial_state():
        state = model.state()
        _set_drop_velocities(model, state)
        return state

    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    pipeline.collide(initial_state(), contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)

    weights, rhs, limit_w = {}, {}, {}
    for name, options in (("scene", {}), ("steel", {"solve_mode": "serial"})):
        for enabled in (True, False):
            solver = SolverSteel(
                model, **{**SCENE, **options}, enable_restitution=enabled, contact_regularization=REGULARIZATION
            )
            test.assertEqual(solver.solve_mode == "scene", name == "scene")
            solver.step(initial_state(), model.state(), model.control(), contacts, DT)
            test.assertFalse(np.any(solver.constraint_overflow.numpy()), f"{name} overflowed")
            rhs[name, enabled] = _normal_rows(test, solver, count)[1]
            if enabled:
                weights[name] = _contact_weights(test, solver, count)
                limit_w[name] = _limit_weights(solver)
                if name == "scene":
                    rows = int(solver._constraint_count.numpy()[0])
                    owner = _row_owners(test, solver, contacts, rows)
                    normal = np.zeros(rows, dtype=bool)
                    normal[solver._contact_slot.numpy()[np.unique(owner[owner >= 0])]] = True
                    np.testing.assert_array_equal(solver._scene.row_w.numpy()[0, :rows][~normal], 1.0)

    scene_family, scene_phi, scene_weight = weights["scene"]
    steel_family, steel_phi, steel_weight = weights["steel"]
    with_rows = steel_family >= 0
    np.testing.assert_array_equal(scene_family >= 0, with_rows)
    test.assertTrue(np.array_equal(scene_weight[with_rows], steel_weight[with_rows]))
    contact_w = np.float32(1.0 / (1.0 + REGULARIZATION))
    test.assertEqual(set(steel_weight[with_rows].tolist()), {float(contact_w), 1.0})
    fires = with_rows & (rhs["steel", True] != rhs["steel", False])
    np.testing.assert_array_equal(fires, with_rows & (rhs["scene", True] != rhs["scene", False]))
    penetrating = with_rows & (steel_phi <= 0.0)
    np.testing.assert_array_equal(scene_phi[penetrating] <= 0.0, True)
    np.testing.assert_array_equal(steel_weight[penetrating & ~fires], contact_w)
    np.testing.assert_array_equal(steel_weight[fires | (with_rows & ~penetrating)], 1.0)
    for family in (0, 1):
        in_family = steel_family == family
        test.assertTrue(np.any(penetrating & ~fires & in_family), f"no regularized contact on Steel's family {family}")
        test.assertTrue(np.any(penetrating & fires & in_family), f"no firing penetrating contact on family {family}")
    # Each world's arm has a lower and an upper limit row, which weigh 1 in the scene as in Steel.
    test.assertEqual(limit_w["scene"].size, 2 * model.world_count)
    test.assertTrue(np.array_equal(limit_w["scene"], limit_w["steel"]))
    np.testing.assert_array_equal(limit_w["scene"], 1.0)


def _sliding_sphere(device):
    """A free sphere resting on the ground plane, both with friction :data:`SPHERE_MU`: one contact."""
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(mu=SPHERE_MU)
    builder.add_ground_plane(cfg=cfg)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, SPHERE_RADIUS), wp.quat_identity()))
    builder.add_shape_sphere(body, radius=SPHERE_RADIUS, cfg=cfg)
    return builder.finalize(device=device)


@needs_read_back
def test_friction_start_holds_the_stationary_exit(test, device):
    """Keep sweeping past a stationary sweep until friction starts, as Steel's exit waits for it.

    A sphere slides at 1.5 m/s on the ground with friction: one contact, whose normal row is the only row
    the sweeps before friction solve, and which reaches a fixed point within a few of them. With 16
    iterations and 2 friction position iterations the tangent pair waits for the fifteenth sweep: a
    stationary sweep before it must not end the sphere's loop (``_gated_sweeps``, ``!friction_skipped``),
    or the tangent pair would never be solved. Steel's sweeps hold the same exit (``global_iter >=
    friction_start_iteration``). After the first step the tangent pair carries an impulse and the
    velocities agree with Steel's matrix-free solve within 1e-5 (and 1e-6); after 30 steps the joint
    coordinates agree within 1e-3 RMS and the sphere, rolling, has lost a fifth of its speed.
    """
    model = _sliding_sphere(device)
    options = {
        "iterations": STATIONARY_ITERATIONS,
        "contact_friction_position_iterations": FRICTION_POSITION_ITERATIONS,
    }
    scene = _Run(model, TRACK_ROWS, {}, **options)
    steel = _Run(model, TRACK_ROWS, {}, solve_mode="serial", **options)
    test.assertEqual(scene.solver.solve_mode, "scene")
    test.assertIsNone(steel.solver._scene)
    test.assertEqual(scene.solver._scene.friction_start, STATIONARY_ITERATIONS - FRICTION_POSITION_ITERATIONS)
    speed = 1.5
    for run in (scene, steel):
        _set_box_velocities(model, run.state_0, speed=speed)
    for step in range(TRACK_STEPS):
        scene.step()
        steel.step()
        if step == 0:
            test.assertEqual(int(scene.contacts.rigid_contact_count.numpy()[0]), 1)
            friction = _friction_impulses(scene.solver)
            test.assertEqual(friction.size, 2)
            test.assertTrue(np.any(friction != 0.0), "the tangent pair was never solved")
            qd_scene, qd_steel = scene.state_0.joint_qd.numpy(), steel.state_0.joint_qd.numpy()
            np.testing.assert_allclose(qd_scene, qd_steel, rtol=TRACK_RTOL, atol=TRACK_ATOL)
    q_scene, q_steel = scene.state_0.joint_q.numpy(), steel.state_0.joint_q.numpy()
    test.assertLess(_rms(q_scene, q_steel), TRACK_RMS)
    test.assertLess(float(scene.state_0.joint_qd.numpy()[0]), 0.8 * speed)
    for run in (scene, steel):
        test.assertFalse(np.any(run.solver.constraint_overflow.numpy()))


def _sweep_factories() -> dict:
    """Each sweep factory with its build parameters bound, by its key in ``scene_kernel_snippets``."""
    return {
        "sides": partial(get_sweep_sides_kernel, "lanes8", "steel", 30),
        "serial": partial(get_sweep_rows_serial_kernel, "steel", 30),
        "order_serial": partial(get_sweep_order_serial_kernel, "BA", "steel", 30),
        "color": partial(get_sweep_color_kernel, "lanes8", "steel", 30),
        "block": partial(get_sweep_block_kernel, "lanes8", "steel", 30),
    }


def _arg_names(kernel) -> list[str]:
    return [arg.label for arg in kernel.adj.args]


def _without_box_clamp(snippet: str) -> str:
    """A sweep snippet without the lines joint friction adds: its row type, and each row loop's box clamp."""
    lines = snippet.splitlines(keepends=True)
    return "".join(line for line in lines if "JOINT_FRICTION" not in line and "row_bound" not in line)


def _without_branch(lines, first: str):
    """``lines`` without the update branch from the line ``first`` to its ``} else``, which joins the next branch."""
    kept, in_branch = [], False
    for line in lines:
        if line.strip() == first:
            in_branch = True
        if not in_branch:
            kept.append(line)
        elif line.strip() == "} else":
            in_branch = False
    return kept


def _without_late_rows(snippet: str) -> str:
    """A row-loop snippet without the lines late rows add, and with its change line as without them.

    Late rows name the row type ``VELOCITY_LIMIT``, declare the stateless rows' change, add the velocity-limit
    branch and apply a stateless row's change instead of its impulse's: the inverse removes the first three and
    restores the change line.
    """
    lines = [
        line
        for line in snippet.splitlines(keepends=True)
        if "constexpr int VELOCITY_LIMIT" not in line
        and line.strip() not in ("bool stateless = false;", "float stateless_change = 0.0f;")
        and "A stateless row of phase C" not in line
    ]
    restored = [
        line.replace("stateless ? stateless_change : new_impulse - old_impulse", "new_impulse - old_impulse")
        for line in _without_branch(lines, "if (row_type == VELOCITY_LIMIT) {")
    ]
    return "".join(restored)


def _without_compliance(snippet: str) -> str:
    """A row-loop snippet without the lines contact compliance adds: its comment, and its right-hand side restored."""
    lines = [
        line
        for line in snippet.splitlines(keepends=True)
        if not line.strip().startswith("// Compliance:") and "other row and already in the row's diagonal" not in line
    ]
    return "".join(line.replace(COMPLIANT_RHS, RIGID_RHS) for line in lines)


def _without_fused_clamp(snippet: str) -> str:
    """A row-loop snippet without the lines the fused clamp adds: its pass flag, its gate and its update branch."""
    lines = [
        line
        for line in snippet.splitlines(keepends=True)
        if line.strip() not in ("bool clamp_pass = false;", "if (clamp_pass && row_type != JOINT_TARGET) continue;")
    ]
    return "".join(_without_branch(lines, "if (clamp_pass) {"))


def _undo(snippet: str, *, cuts=(), restore=None) -> str:
    """``snippet`` with blocks of a feature's lines cut and its rewritten lines restored, line by line.

    Each cut ``(first, last, replacement)`` removes the lines from the first one whose stripped text is ``first``
    through the next one whose stripped text is ``last`` (with a blank line after it, when ``last`` ends in a
    newline), putting ``replacement`` (a string of whole lines) in their place. ``restore`` maps a line's stripped
    text to the text it replaces, keeping the line's indentation.
    """
    lines = snippet.splitlines(keepends=True)
    for first, last, replacement in cuts:
        start = next(i for i, line in enumerate(lines) if line.strip() == first)
        end = next(i for i in range(start, len(lines)) if lines[i].strip() == last.strip())
        if last.endswith("\n") and lines[end + 1].strip() == "":
            end += 1
        lines[start : end + 1] = replacement.splitlines(keepends=True)
    for old, new in (restore or {}).items():
        lines = [line.replace(old, new) if line.strip() == old else line for line in lines]
    return "".join(lines)


def _without_late_phase(key: str, snippet: str) -> str:
    """A late-rows sides (``"lanes8"``) or order-oracle (``"BA"``) snippet without its phase C, and its row loop's
    late lines (:func:`_without_late_rows`): the snippet of the same features without ``late_rows``.

    The sides kernel's late rows add the side's phase C (its late rows' sweep), the coupled late pass and the sweep of
    a side over both phases; the order oracle's add the phase C of a side and of the isolated or coupled sides, and walk
    each sweep's passes from a list.
    """
    snippet = _without_late_rows(snippet)
    phase_c = ("// Phase C of one side: its late rows.", "};\n", "")
    coupled_comment = "// The coupled sides take passes 1 (phase A) and 2 (phase C, the late rows), one sweep each."
    if key == "sides":
        return _undo(
            snippet,
            cuts=(
                phase_c,
                (coupled_comment, coupled_comment, ""),
                (
                    "const int late_begin = side_late_begin.data[side];",
                    "const int late_end = side_late_end.data[side];",
                    "",
                ),
                (
                    "// A sweep of the side: phase A (but in the late pass), then phase C (but in the coupled pass).",
                    "};",
                    "",
                ),
            ),
            restore={
                "if (side_flag != (coupled_pass != 0 ? 1 : 0)) return;": "if (side_flag != coupled_pass) return;",
                "if (__any_sync(mask, sweep_side()) == 0) break;": (
                    "if (__any_sync(mask, sweep_rows_group(begin, end, lane, mask)) == 0) break;"
                ),
            },
        )
    return _undo(
        snippet,
        cuts=(
            phase_c,
            ("const auto sweep_late_sides = [&](int coupled) -> int {", "};", ""),
            (
                "const int passes[5] = {0, 3, 2, 1, 4};",
                "}",
                "        int changed = sweep_sides(0);\n"
                "        changed |= sweep_rows(two_sided_begin, two_sided_end);\n"
                "        changed |= sweep_sides(1);\n",
            ),
        ),
    )


def _without_fused_walks(key: str, snippet: str) -> str:
    """A fused-clamp snippet without the clamp's lines (:func:`_without_fused_clamp`) and without its clamp passes:
    the snippet of the same features without ``fused_clamp``.

    The sides and order snippets' phase C ends with a clamp pass over the side's one-sided rows; the serial walk of
    every row ends each walk with one.
    """
    snippet = _without_fused_clamp(snippet)
    if key == "serial":
        return _undo(
            snippet,
            cuts=(
                (
                    "// Each walk of every row ends with the fused clamp of its drive rows (Steel's clamp after its "
                    "velocity limits);",
                    "};",
                    "",
                ),
            ),
            restore={
                "if (!sweep_walk() && stationary_exit != 0) break;": (
                    "if (!sweep_rows(begin, end) && stationary_exit != 0) break;"
                )
            },
        )
    return _undo(
        snippet,
        cuts=(("clamp_pass = true;", "clamp_pass = false;", ""),),
        restore={
            "// Phase C of one side: its late rows (velocity limits), then the fused clamp of its drive rows.": (
                "// Phase C of one side: its late rows."
            )
        },
    )


# The phase-C passes of the order oracle under late rows in the default phase order, which contact torsion's replace.
_LATE_PASSES_BA = """        const int passes[5] = {0, 3, 2, 1, 4};
        int changed = 0;
        #pragma unroll 1
        for (int pass = 0; pass < 5; ++pass) {
            const int kind = passes[pass];
            if (kind == 2) changed |= sweep_rows(two_sided_begin, two_sided_end);
            else if (kind < 2) changed |= sweep_sides(kind);
            else changed |= sweep_late_sides(kind - 3);
        }
"""


def _without_torsion(key: str, snippet: str) -> str:
    """A sweep snippet without the lines contact torsion adds: the snippet of the same features without ``torsion``,
    with phase C kept (as the snippet without torsion has it under late rows).

    Every row loop loses C1: the row types ``TORSION`` and ``TORSION_RETIRED``, the spin row's heads (each the budget,
    ending with the bound's ``return``), the head's call in the row visit and the spin row's clamp. The sides snippet
    also loses its sides' spin-row ranges and the torsion pass of a side's sweep; the order oracle its sides' spin rows,
    the two-sided spin rows' range and its torsion passes.
    """
    lines = []
    skipping = False
    for line in snippet.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("// C1's head of spin row"):
            if lines and not lines[-1].strip():
                lines.pop()
            skipping = True
        if skipping:
            if stripped == "return bound;":
                skipping = "end"
            elif skipping == "end" and stripped == "};":
                skipping = False
            continue
        if stripped in (
            "constexpr int TORSION = 7;",
            "constexpr int TORSION_RETIRED = -2;",
            "float torsion_bound = 0.0f;",
        ):
            continue
        if stripped.startswith("// C1: a spin row first rescales") or stripped.startswith(
            "if (row_type == TORSION) torsion_bound"
        ):
            continue
        lines.append(line)
    snippet = "".join(_without_branch(lines, "if (row_type == TORSION) {"))
    if key == "sides":
        lane = "sweep_rows_group(torsion_begin, torsion_end, lane, mask)"
        snippet = _undo(
            snippet,
            cuts=(
                (
                    "const int torsion_begin = side_torsion_begin.data[side];",
                    "const int torsion_end = side_torsion_end.data[side];",
                    "",
                ),
                (
                    "// A sweep of the side: phase A (passes 0 and 1), its one-sided spin rows (C1, passes 0 and 3),"
                    " then phase C",
                    "// (passes 0 and 2).",
                    "    // A sweep of the side: phase A (but in the late pass), then phase C (but in the coupled"
                    " pass).\n",
                ),
                (
                    f"if (coupled_pass == 0 || coupled_pass == 3) changed |= {lane};",
                    f"if (coupled_pass == 0 || coupled_pass == 3) changed |= {lane};",
                    "",
                ),
            ),
        )
        return "".join(
            line.replace("if (coupled_pass < 2) changed |=", "if (coupled_pass != 2) changed |=").replace(
                "if (coupled_pass == 0 || coupled_pass == 2) changed |=", "if (coupled_pass != 1) changed |="
            )
            for line in snippet.splitlines(keepends=True)
        )
    if key == "order_serial":
        return _undo(
            snippet,
            cuts=(
                (
                    "// The one-sided spin rows of the isolated (0) or the coupled (1) sides, side by side; then the "
                    "two-sided ones.",
                    "const int torsion_two_end = torsion_color_start.data[torsion_color_start.shape[0] - 1];",
                    "",
                ),
                ("const int passes[8] = {0, 5, 3, 2, 1, 6, 7, 4};", "}", _LATE_PASSES_BA),
            ),
        )
    return snippet


_MODE_LINES = (
    "constexpr int FIXED_NORMAL = 0;",
    "constexpr int FREE_BODY = 2;",
    "const int contact_mode = friction ? row_friction_mode.data[row] : FIXED_NORMAL;",
    "if (friction && denominator <= 0.0f) continue;",
    "float new_normal = 0.0f;",
    "float normal_change = 0.0f;",
    "if (contact_mode != FIXED_NORMAL) impulses.data[parent] = new_normal;",
)
"""The single lines a friction mode adds: its update names, each row visit's head (the contact's update and the pairs'
diagonal guard), the combined update's declarations and the tail's write of its normal impulse."""


def _without_friction_mode(snippet: str) -> str:
    """A sweep snippet without the lines a friction mode adds: the snippet of the same features with the mode off.

    The device functions before the row loops (``contact_block`` and the mode's law, through the law's closing
    ``};``), the single lines of :data:`_MODE_LINES` and their comments, the combined update's branch (from its
    opening line to its ``} else``), the normal impulse's response in the tail (from ``if (normal_change != 0.0f) {``
    to its closing ``}``) and its change in the changed-row test.
    """
    start = snippet.find("\n    // A contact's rows [normal, t1, t2]")
    if start >= 0:
        law = max(snippet.find("const auto contact_bisection"), snippet.find("const auto contact_coulomb_newton"))
        end = snippet.index("\n    };\n", law) + len("\n    };\n")
        snippet = snippet[:start] + snippet[end:]
    lines = [
        line
        for line in snippet.splitlines(keepends=True)
        if line.strip() not in _MODE_LINES
        and not line.strip().startswith("// Friction modes:")
        and "as Steel's modes drop the pairs' diagonal guard" not in line
    ]
    lines = _without_branch(lines, "if (friction && contact_mode != FIXED_NORMAL) {")
    kept, in_response = [], False
    for line in lines:
        if line.strip() == "if (normal_change != 0.0f) {":
            in_response = True
        elif in_response and line.strip() == "}":
            in_response = False
        elif not in_response:
            kept.append(line.replace(" || normal_change != 0.0f", ""))
    return "".join(kept)


_PATCH_LOAD_LINES = (
    "for (int next = rows_parent.data[parent]; next >= 0 && next != parent; next = rows_parent.data[next])",
    "normal_load += impulses.data[next];",
)
"""The lines friction patches add to a tangent pair's normal load: the walk of its region's ring."""

PATCH_LOAD = "float normal_load = impulses.data[parent];"
POINT_LOAD = "const float normal_load = impulses.data[parent];"
"""A tangent pair's normal load: the start of its region's ring sum under friction patches, its normal row's impulse
otherwise."""


def _without_patches(snippet: str) -> str:
    """A sweep snippet without the lines friction patches add: the ring's walk and its comment, the load restored."""
    lines = [
        line
        for line in snippet.splitlines(keepends=True)
        if line.strip() not in _PATCH_LOAD_LINES
        and not line.strip().startswith("// Friction patches:")
        and "a lone member's itself" not in line
    ]
    return "".join(line.replace(PATCH_LOAD, POINT_LOAD) if line.strip() == PATCH_LOAD else line for line in lines)


def _without_drive_update(snippet: str) -> str:
    """A sweep snippet without the lines drive rows add: its row type, and each row loop's force-drive branch.

    The branch runs from ``if (row_type == JOINT_TARGET) {`` to its ``} else``, which joins the clamps after it; it
    holds the frozen drives' skip (:data:`FREEZE_LINE`) before the update.
    """
    kept, in_branch = [], False
    for line in snippet.splitlines(keepends=True):
        if "constexpr int JOINT_TARGET" in line:
            continue
        if line.strip() == "if (row_type == JOINT_TARGET) {":
            in_branch = True
        if not in_branch:
            kept.append(line)
        elif line.strip() == "} else":
            in_branch = False
    return "".join(kept)


class TestSteelSceneFeatures(unittest.TestCase):
    def test_featured_kernels_share_one_definition(self):
        """Build every feature off as today's kernels, and any feature on from one featured definition.

        Every feature off (``SceneFeatures()``), each sweep factory returns its default kernel: today's name, arguments
        ending in the row arguments and no feature argument, and the native sources the scene tripwire pins (19 hashes).
        With any feature on it returns a kernel of its one featured definition: the same Python stub for every set,
        named the default's name and one token per feature that is on (``_fpi``, ``_reg``, ``_jf``, ``_drv``, ``_late``,
        ``_fvc``, ``_dpc``, ``_cmp``: ``_fpi_reg``, ``_reg_jf``, ``_jf_drv``, ``_drv_late_fvc`` and so on), and taking
        the default's arguments and then every feature's, ``friction_start``, ``row_w``, ``row_bound``, the four drive
        descriptors, the late ranges, the drive rows' velocity limits, the frozen drives' flag and the rows' compliance
        ``row_gamma``, whichever are on (the depenetration cap has none). Equal feature sets share the cached kernel;
        different sets do not. Joint friction adds its box clamp to every sweep's snippet and drive rows their
        force-drive branch, which first skips the row while ``freeze_drives`` (the velocity-only iterations under
        ``velocity_drive_mode="freeze"``), in every row loop, and neither any other line: a set without its lines is the
        set without the feature, character for character (``_drv`` the default, ``_jf_drv`` ``_jf``, ``_fpi_reg_jf_drv``
        ``_fpi_reg_jf``); only the drive branch reads ``freeze_drives``. Late rows give every row loop the stateless
        velocity-limit update and change nothing else in the color, block and serial snippets (a phase-B kernel never
        reaches a late row; the serial walk reaches them in the row order); the sides and order snippets also walk phase
        C, the late ranges, and change nothing else (without their phase C lines they are the snippets without late
        rows). The fused clamp adds its clamp pass to every row loop and changes nothing else in the color and block
        snippets; the sides, order and serial snippets run its pass after their late rows (the serial walk after each
        walk of every row), and change nothing else. The depenetration cap changes no sweep snippet. Contact compliance
        rewrites the right-hand side of every row loop's update, ``gamma * lambda`` added, and changes nothing else;
        only that line reads ``row_gamma``. Contact torsion adds C1 to every row loop and its spin rows' passes to phase
        C. A friction mode (``_bis``, ``_bisx``, ``_cn``) adds its device functions before the row loops, each row
        visit's head (only it reads ``row_friction_mode``), the combined update of a flagged contact with the mode's law
        (de Saxce's bias under ``_bisx`` alone) and the tail's normal impulse, and nothing else: without them a set is
        the set without the mode, the all-on set and its two-sided spin rows' pass included. Friction patches (``_pat``)
        turn each row loop's normal load of a tangent pair into the walk of its region's ring, and nothing else: no
        argument, and without the walk a set is the set without patches, the all-on set and its spin rows' pass
        included. A fused clamp without drive rows and late rows is refused.
        """
        hashes = {}
        for support, free_response, dof_max in ((21, False, 27), (30, False, 27), (12, True, 6)):
            for key, snippet in scene_kernel_snippets(support, free_response=free_response, dof_max=dof_max).items():
                hashes[f"{support}/{free_response}/{key}"] = _sha256(snippet)
        self.assertEqual(hashes, {k: v for k, v in DEFAULT_SNIPPET_SHA256.items() if k in hashes})
        self.assertEqual(len(hashes), 18)
        for key, factory in _sweep_factories().items():
            with self.subTest(kernel=key):
                default = factory()
                self.assertIs(factory(features=SceneFeatures()), default)
                args = _arg_names(default)
                self.assertEqual(args[-len(ROW_ARGS) :], ROW_ARGS)
                self.assertFalse(set(FEATURE_ARGS) & set(args), args)
                self.assertNotRegex(default.key, r"_(fpi|reg|jf|drv|late|fvc|dpc|tor|cmp|bis|bisx|cn|pat)")
                kernels = {}
                for suffix, features in FEATURE_SETS.items():
                    kernel = factory(features=features)
                    kernels[suffix] = kernel
                    self.assertEqual(kernel.key, default.key + suffix)
                    self.assertEqual(_arg_names(kernel), args + FEATURE_ARGS, suffix)
                    self.assertIsNot(kernel.func.__code__, default.func.__code__, suffix)
                    self.assertIs(kernel.func.__code__, kernels["_fpi"].func.__code__, suffix)
                    self.assertIs(factory(features=SceneFeatures(**dataclasses.asdict(features))), kernel, suffix)
                self.assertEqual(len({id(kernel) for kernel in (default, *kernels.values())}), 1 + len(FEATURE_SETS))
        snippets = {
            suffix: scene_kernel_snippets(30, free_response=False, dof_max=27, features=features)
            for suffix, features in FEATURE_SETS.items()
        }
        snippets[""] = scene_kernel_snippets(30, free_response=False, dof_max=27)
        # The rounding shim (CPU-only text) heads every sweep snippet of a friction mode or contact torsion, after the
        # host shim, and no other; the comparisons below are of the snippets without it.
        for suffix, by_key in snippets.items():
            features = FEATURE_SETS.get(suffix, SceneFeatures())
            rounding = features.friction_mode != "fixed_normal" or features.torsion
            for key, snippet in by_key.items():
                if key == "response":
                    self.assertNotIn(_HOST_ROUNDING_SHIM_CPP, snippet, suffix)
                    continue
                self.assertEqual(snippet.count(_HOST_ROUNDING_SHIM_CPP), int(rounding), (suffix, key))
                self.assertEqual(snippet.startswith(_HOST_SHIM_CPP + _HOST_ROUNDING_SHIM_CPP), rounding, (suffix, key))
                by_key[key] = snippet.replace(_HOST_ROUNDING_SHIM_CPP, "")
        for key in ("sides", "serial", "order_serial", "color", "block"):
            with self.subTest(snippet=key):
                self.assertNotIn("row_bound", snippets["_reg"][key])
                self.assertIn("row_bound", snippets["_reg_jf"][key])
                self.assertEqual(_without_box_clamp(snippets["_reg_jf"][key]), snippets["_reg"][key])
                for with_drives, without in (("_drv", ""), ("_jf_drv", "_jf"), ("_fpi_reg_jf_drv", "_fpi_reg_jf")):
                    self.assertNotIn("drive_max_impulse", snippets[without][key])
                    self.assertIn("drive_max_impulse", snippets[with_drives][key])
                    self.assertEqual(_without_drive_update(snippets[with_drives][key]), snippets[without][key])
                    # The frozen drives' skip opens every row loop's drive branch; nothing else reads the flag.
                    lines = [line.strip() for line in snippets[with_drives][key].splitlines()]
                    branches = [i for i, line in enumerate(lines) if line == "if (row_type == JOINT_TARGET) {"]
                    self.assertTrue(branches)
                    self.assertEqual([lines[i + 2] for i in branches], [FREEZE_LINE] * len(branches))
                    self.assertEqual(sum("freeze_drives" in line for line in lines), len(branches))
                    self.assertNotIn("freeze_drives", snippets[without][key])
                for with_late, without in (("_late", ""), ("_drv_late", "_drv")):
                    self.assertNotIn("VELOCITY_LIMIT", snippets[without][key])
                    self.assertIn("if (row_type == VELOCITY_LIMIT) {", snippets[with_late][key])
                    if key in ("color", "block", "serial"):
                        self.assertEqual(_without_late_rows(snippets[with_late][key]), snippets[without][key])
                    else:
                        self.assertIn("side_late_begin.data[side]", snippets[with_late][key])
                        self.assertEqual(_without_late_phase(key, snippets[with_late][key]), snippets[without][key])
                self.assertNotIn("clamp_pass", snippets["_drv_late"][key])
                self.assertIn("if (clamp_pass) {", snippets["_drv_late_fvc"][key])
                if key in ("color", "block"):
                    self.assertEqual(_without_fused_clamp(snippets["_drv_late_fvc"][key]), snippets["_drv_late"][key])
                else:
                    self.assertIn("clamp_pass = true;", snippets["_drv_late_fvc"][key])
                    self.assertEqual(
                        _without_fused_walks(key, snippets["_drv_late_fvc"][key]), snippets["_drv_late"][key]
                    )
                # The depenetration cap adds nothing to the sweeps.
                self.assertEqual(snippets["_dpc"][key], snippets[""][key])
                self.assertEqual(
                    snippets["_fpi_reg_jf_drv_late_fvc_dpc"][key], snippets["_fpi_reg_jf_drv_late_fvc"][key]
                )
                # Compliance rewrites each row loop's right-hand side, the only line that reads row_gamma.
                all_on = "_fpi_reg_jf_drv_late_fvc_dpc"
                for with_compliance, without in (("_cmp", ""), (all_on + "_cmp", all_on)):
                    lines = [line.strip() for line in snippets[with_compliance][key].splitlines()]
                    rhs_lines = [line for line in lines if line.startswith("const float rhs =")]
                    self.assertTrue(rhs_lines)
                    self.assertEqual(rhs_lines, [COMPLIANT_RHS] * len(rhs_lines))
                    self.assertEqual(sum("row_gamma" in line for line in lines), len(rhs_lines))
                    self.assertNotIn("row_gamma", snippets[without][key])
                    self.assertEqual(_without_compliance(snippets[with_compliance][key]), snippets[without][key])
                # Contact torsion adds C1 to every row loop and, in the sides and order snippets, its spin rows' passes
                # to phase C, which it runs without late rows too; nothing else.
                all_on_tor = "_fpi_reg_jf_drv_late_fvc_dpc_tor_cmp"
                self.assertNotIn("TORSION", snippets[all_on + "_cmp"][key])
                self.assertIn("if (row_type == TORSION) {", snippets["_tor"][key])
                self.assertEqual(_without_torsion(key, snippets[all_on_tor][key]), snippets[all_on + "_cmp"][key])
                if key in ("sides", "order_serial"):
                    self.assertIn("side_torsion_begin.data[side]", snippets["_tor"][key])
                    self.assertEqual(
                        _without_torsion(key, snippets["_tor"][key]), _without_late_rows(snippets["_late"][key])
                    )
                else:
                    self.assertEqual(_without_torsion(key, snippets["_tor"][key]), snippets[""][key])
                # A friction mode adds its device functions, each row visit's head, the combined update and its tail
                # lines to every row loop, and nothing else; only the head reads row_friction_mode. Its law: the
                # bisection under "bisection" and "bisection_desaxce", de Saxce's bias under the latter alone, and
                # Coulomb-Newton under "coulomb_newton".
                for with_mode, law, bias in (
                    ("_bis", "bisection", False),
                    ("_bisx", "bisection", True),
                    ("_cn", "coulomb_newton", False),
                ):
                    snippet = snippets[with_mode][key]
                    lines = [line.strip() for line in snippet.splitlines()]
                    heads = [line for line in lines if line.startswith("const int contact_mode =")]
                    self.assertTrue(heads)
                    self.assertEqual(sum("row_friction_mode.data" in line for line in lines), len(heads))
                    self.assertEqual(lines.count("if (friction && contact_mode != FIXED_NORMAL) {"), len(heads))
                    self.assertEqual(lines.count(f"contact_{law}(u, g, target, mu, lambda);"), len(heads))
                    self.assertEqual(lines.count("if (contact_mode == FREE_BODY) {"), len(heads) if bias else 0)
                    self.assertNotIn("contact_mode", snippets[""][key])
                    self.assertEqual(_without_friction_mode(snippet), snippets[""][key])
                all_on_mode = all_on_tor + "_bisx"
                self.assertEqual(_without_friction_mode(snippets[all_on_mode][key]), snippets[all_on_tor][key])
                # Friction patches turn each row loop's tangent-pair load into its region's ring sum, and nothing else.
                for with_patches, without in (("_pat", ""), (all_on_mode + "_pat", all_on_mode)):
                    lines = [line.strip() for line in snippets[with_patches][key].splitlines()]
                    loads = [line for line in lines if line == PATCH_LOAD]
                    self.assertTrue(loads)
                    self.assertNotIn(POINT_LOAD, lines)
                    for ring_line in _PATCH_LOAD_LINES:
                        self.assertEqual(lines.count(ring_line), len(loads), ring_line)
                    self.assertNotIn("rows_parent.data[next]", snippets[without][key])
                    self.assertEqual(_without_patches(snippets[with_patches][key]), snippets[without][key])
        for suffix in FEATURE_SETS:
            self.assertEqual(snippets[suffix]["response"], snippets[""]["response"], suffix)
            # The two-sided spin rows' color pass exists with contact torsion alone.
            self.assertEqual("torsion" in snippets[suffix], "_tor" in suffix, suffix)
        # The two-sided spin rows' color pass takes the friction mode's lines too.
        all_on_tor = "_fpi_reg_jf_drv_late_fvc_dpc_tor_cmp"
        self.assertEqual(
            _without_friction_mode(snippets[all_on_tor + "_bisx"]["torsion"]), snippets[all_on_tor]["torsion"]
        )
        # And the patches' ring.
        self.assertEqual(
            _without_patches(snippets[all_on_tor + "_bisx_pat"]["torsion"]), snippets[all_on_tor + "_bisx"]["torsion"]
        )
        for suffix in ("_tor", all_on_tor, all_on_tor + "_bisx", all_on_tor + "_bisx_pat"):
            kernel = get_sweep_torsion_kernel("lanes8", "steel", 30, features=FEATURE_SETS[suffix])
            self.assertEqual(kernel.key, "steel_scene_torsion_lanes8_steel_30_0" + suffix)
            self.assertEqual(_arg_names(kernel), ROW_ARGS + FEATURE_ARGS)
        with self.assertRaisesRegex(ValueError, "contact torsion"):
            get_sweep_torsion_kernel("lanes8", "steel", 30, features=SceneFeatures(late_rows=True))
        for features in (SceneFeatures(late_rows=True, fused_clamp=True), SceneFeatures(drives=True, fused_clamp=True)):
            with self.assertRaisesRegex(ValueError, "fused_clamp"):
                get_sweep_sides_kernel("lanes8", "steel", 30, features=features)

    def test_patch_response_variant_reads_the_anchor_points(self):
        """Build friction patches' contact response from the one-row definition: the row counts and the anchors.

        The ``patches`` variant of the generic response kernel is the ``slots_needed`` variant (each contact's rows from
        ``contact_slots_needed``, which patches imply), one definition (one code object) under its own key, whose
        tangent rows take the anchor points ``anchor_point0`` and ``anchor_point1`` as their lever arms; both variants
        take the anchor points after the shared-anchor flags, which the ``slots_needed`` variant does not read. The
        native snippet is the default's (the scene tripwire's ``response``), and the default definition is unchanged.
        ``test_scene_state.test_patch_anchors_take_precedence_over_shared_anchors`` checks the rows it writes: the
        tangent rows at the anchors whatever the flags, the normal rows at the witness points.
        """
        default = _get_scene_contact_response_kernel(27)
        slots = _get_scene_contact_response_kernel(27, slots_needed=True)
        patched = _get_scene_contact_response_kernel(27, slots_needed=True, patches=True)
        self.assertIs(_get_scene_contact_response_kernel(27, patches=True), patched)
        self.assertEqual(slots.key, default.key + "_slots")
        self.assertEqual(patched.key, slots.key + "_patches")
        self.assertIs(patched.func.__code__, slots.func.__code__)
        self.assertIsNot(slots.func.__code__, default.func.__code__)
        args = _arg_names(slots)
        anchors = args.index("contact_friction_shared_anchor") + 1
        self.assertEqual(args[anchors : anchors + 2], ["anchor_point0", "anchor_point1"])
        self.assertEqual(_arg_names(patched), args)
        self.assertNotIn("anchor_point0", _arg_names(default))


# On every device; a Steel reference is Steel's serial solve (its CPU fallback on the CPU, _steel_on).
for _fn in (
    test_one_row_contacts_track_steel,
    test_friction_position_iterations_track_steel,
    test_friction_position_iterations_gate_the_scene_sweep,
    test_row_watermark_counts_rows_per_model_world,
    test_scene_capacity_failure_is_observable,
    test_scene_contact_f_reports_each_box_weight_row_overflow,
    test_scene_watermark_excess_counts_dropped_rows,
    test_one_row_contacts_use_the_slots_response,
    test_specialized_response_rejects_one_row_contacts,
    test_friction_start_holds_the_stationary_exit,
):
    add_function_test(TestSteelSceneFeatures, _fn.__name__, _fn, devices=get_test_devices())

# CUDA only, each for the reason beside it: the Steel references of restitution and regularization are Steel's CUDA
# solve; the scene's phases run on CUDA alone; test_scene_cpu steps the regularized scene on the CPU against CUDA's.
for _fn in (
    test_restitution_rebound_tracks_steel,  # Steel's restitution
    test_restitution_rows_fire_where_steel_fires,  # Steel's restitution
    test_friction_position_iterations_match_the_serial_oracle,  # the phases; the CPU sweeps the serial oracle alone
    test_regularization_tracks_steel,  # Steel's regularization
    test_regularization_matches_the_serial_oracle,  # the phases
    test_row_weights_match_steel_bitwise,  # Steel's regularization
):
    add_function_test(TestSteelSceneFeatures, _fn.__name__, _fn, devices=get_cuda_test_devices())

# The warning test checks its own output: CheckOutput's capture would receive the device printf.
add_function_test(
    TestSteelSceneFeatures,
    "test_scene_overflow_warning_is_printed_once",
    test_scene_overflow_warning_is_printed_once,
    devices=get_test_devices(),
    check_output=False,
)

add_function_test(
    TestSteelSceneFeatures,
    "test_default_features_compile_todays_source",
    test_default_features_compile_todays_source,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
