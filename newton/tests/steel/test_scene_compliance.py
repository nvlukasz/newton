# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact compliance of the scene mode (``solve_mode="scene"``, ``contact_compliance=True``).

Steel solves a contact with a positive per-contact stiffness ``k`` by its implicit spring-damper law
(``contact_compliance.py``): with damping ``c``, separation ``phi`` and step ``dt``, the row's compliance
``gamma = 1 / (dt (dt k + c))`` and bias ``bias = k phi / (dt k + c)`` (the dashpot off across an open gap) give
the residual ``J v + bias + gamma lambda``. Steel prepares the rows on the device with that law in float64
(``contact_compliance.prepare_rows``) and latches its errors into status words. The scene prepares the rows of its
one row pool with the same law and the same status codes (``scene_compliance.prepare_scene_compliance``, after the
row preparation and the depenetration cap): ``row_gamma``, ``rhs = bias - target``, ``diag + (gamma - cfm)`` and
the friction rows' coefficients weighted by the contact's friction scale; its sweeps add ``gamma lambda`` to a
row's right-hand side at each visit.

The scene's law is checked against Steel's device law on the same inputs bit for bit, and the scene against the
reference law (``_compliance_reference.py``), against Steel's solve within the tracking tolerances, against the
serial oracle bit for bit, and against itself under capture.
"""

import math
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel import contact_compliance, scene_compliance
from newton._src.solvers.steel.contact_compliance import normal_coefficients, start_step
from newton._src.solvers.steel.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
)
from newton._src.solvers.steel.scene_pgs import RANGE_TWO_SIDED
from newton.solvers import SolverSteel
from newton.tests.steel._compliance_reference import solve_reference
from newton.tests.steel._fixtures import DT, _build_contact_scene, _set_box_velocities, run_fixture
from newton.tests.steel._scene_fixtures import _humanoids, _Run
from newton.tests.steel.test_scene_features import TRACK_ROWS, _assert_phases_match_serial_oracle, _track_steel
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

MATERIAL_ARRAYS = ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction")
"""The contacts' per-contact material arrays that the compliant law reads."""

STIFFNESS = 2.0e4
DAMPING = 50.0
FRICTION_SCALE = 0.5
"""The material of every contact of the compliant runs: stiffness [N/m], damping [N s/m] and friction scale."""

REFERENCE_DEPTH = 0.001
"""Penetration [m] of the reference-law sphere: ``run_fixture``'s sphere (radius 0.05 m) placed 1 mm low."""

REFERENCE_STIFFNESS = 3000.0
REFERENCE_DAMPING = 20.0
REFERENCE_DT = 0.005
REFERENCE_ITERATIONS = 8
"""``run_fixture``'s material, step and iterations, which the reference law takes too."""

DIAG_RTOL = 1.0e-6
"""Tolerance of a compliant row's diagonal against its Delassus entry plus ``gamma``: the entry is recomputed on the
host from the row's factor values in float64, the solver sums it in float32."""

IMPULSE_RTOL = 1.0e-5
"""Tolerance of the impulse after the sweep against the reference law's PGS on the row's own Delassus entry."""

CAPTURE_STEPS = 30
"""Steps of the capture test."""

CONTACT_CAPACITY = 32
"""Contacts of the step-check test's hand-made buffers (``run_fixture``'s capacity)."""

LAW_ROWS = 40
"""Rows of the law guard's one-world row storage."""

LAW_ACTIVE_ROWS = 36
"""Active rows of the law guard: a contact mapped to a row past them is mapped beyond the active rows."""

LAW_CFM = 1.0e-6
"""Constraint force mixing of the law guard's rows, which a compliant row's ``gamma`` replaces."""

LAW_DTS = (0.005, 1.0e-3, 1.0e4)
"""Steps [s] of the law guard: the fixtures' step, a short one, and one long enough that a stiff contact's
``gamma`` underflows float32 (Steel keeps it the smallest normal float32, so its friction rows stay weighted)."""

_F32_MAX = float(np.finfo(np.float32).max)

LAW_CONTACTS = (
    # (stiffness, damping, friction scale, separation, path, slot, slots needed); slot s is a contact row with its
    # two friction rows s + 1 and s + 2 unless _LAW_ROW_TYPES says otherwise.
    (3000.0, 20.0, 0.5, -1.0e-3, 0, 0, 3),  # penetrating
    (3000.0, 20.0, 0.0, 2.0e-3, 0, 3, 3),  # an open gap (no dashpot), a zero friction scale (weight 1)
    (2.0e4, 50.0, 0.25, 0.0, 0, 6, 3),  # touching: the dashpot acts at zero separation
    (0.0, 20.0, 0.5, -1.0e-3, 0, 9, 3),  # zero stiffness: rigid, uncounted
    (3000.0, 20.0, 0.5, -1.0e-3, -1, -1, 0),  # excluded on purpose by the allocator: skipped
    (3000.0, 20.0, 0.5, -1.0e-3, -1, -1, 3),  # requested rows and lost them
    (-1.0, 20.0, 0.5, -1.0e-3, 0, 12, 3),  # negative stiffness
    (math.nan, 20.0, 0.5, -1.0e-3, 0, 12, 3),  # non-finite stiffness
    (3000.0, -1.0, 0.5, -1.0e-3, 0, 12, 3),  # negative damping
    (3000.0, 20.0, math.inf, -1.0e-3, 0, 12, 3),  # non-finite friction scale
    (3000.0, 20.0, 0.5, math.nan, 0, 15, 3),  # non-finite separation
    (1.0e-36, 0.0, 0.5, -1.0e-3, 0, 18, 3),  # gamma past the float32 range at the short steps
    (3.0e38, 0.0, 0.25, -1.0e-3, 0, 21, 3),  # gamma below the float32 range at the long step
    (3000.0, 20.0, 0.5, -1.0e-3, 0, 24, 3),  # mapped to a joint-limit row
    (3000.0, 20.0, 0.5, -1.0e-3, 0, 37, 3),  # mapped beyond the active rows
    (3000.0, 20.0, 0.5, -1.0e-3, 0, 27, 3),  # two contacts on one normal row (one latches)
    (3000.0, 20.0, 0.5, -1.0e-3, 0, 27, 3),
    (5.0e3, 0.0, 2.0, -2.0e-3, 0, 30, 3),  # no damping, a friction scale above one
    (5.0e3, 10.0, 0.5, -1.0e-3, 0, 33, 3),
    (3000.0, 20.0, 0.5, -1.0e-3, 0, 33, 3),  # past the contact count: ignored
)
"""The law guard's contacts, one per case of Steel's per-contact checks and law."""

LAW_CONTACT_COUNT = len(LAW_CONTACTS) - 1
"""Contacts of the law guard's step; the last of :data:`LAW_CONTACTS` is past the count."""

_LAW_ROW_TYPES = {24: PGS_CONSTRAINT_TYPE_JOINT_LIMIT}
"""Rows of the law guard that are not contact rows."""


def _law_rows() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The law guard's row storage on the host: type, parent, separation, target, diagonal and friction coefficient.

    Every third row from 0 is a contact row whose two friction rows follow it (a joint-limit row at 24, two free
    rows past the last contact row), with distinct diagonals, targets and coefficients, and separations that the
    contacts of :data:`LAW_CONTACTS` mapped to them read.
    """
    row_type = np.full((1, LAW_ROWS), PGS_CONSTRAINT_TYPE_FRICTION, dtype=np.int32)
    row_parent = np.full((1, LAW_ROWS), -1, dtype=np.int32)
    phi = np.zeros((1, LAW_ROWS), dtype=np.float32)
    for slot in range(0, LAW_ROWS - 2, 3):
        row_type[0, slot] = _LAW_ROW_TYPES.get(slot, PGS_CONSTRAINT_TYPE_CONTACT)
        if row_type[0, slot] == PGS_CONSTRAINT_TYPE_CONTACT:
            row_parent[0, slot + 1 : slot + 3] = slot
    row_type[0, LAW_ROWS - 1] = PGS_CONSTRAINT_TYPE_JOINT_LIMIT
    for _, _, _, separation, _, slot, _ in LAW_CONTACTS:
        if slot >= 0:
            phi[0, slot] = separation
    rows = np.arange(LAW_ROWS, dtype=np.float32)
    target = (0.01 * (rows % 5)).astype(np.float32)[None, :]
    target[0, 0] = 0.0
    diag = (1.0 + 0.125 * rows).astype(np.float32)[None, :]
    row_mu = (0.3 + 0.01 * rows).astype(np.float32)[None, :]
    return row_type, row_parent, phi, target, diag, row_mu


def _law_inputs(device):
    """The law guard's contacts and contact mapping as device arrays (one world, the dense route)."""
    columns = list(zip(*LAW_CONTACTS, strict=True))
    contacts = SimpleNamespace(
        rigid_contact_count=wp.array([LAW_CONTACT_COUNT], dtype=int, device=device),
        rigid_contact_max=len(LAW_CONTACTS),
        **{
            name: wp.array(np.asarray(values, dtype=np.float32), dtype=float, device=device)
            for name, values in zip(MATERIAL_ARRAYS, columns[:3], strict=True)
        },
    )
    path, slot, needed = (wp.array(np.asarray(values), dtype=int, device=device) for values in columns[4:])
    world = wp.zeros(len(LAW_CONTACTS), dtype=int, device=device)
    return contacts, path, slot, needed, world


def _steel_law(device, dt: float) -> dict[str, np.ndarray]:
    """Steel's device preparation of the law guard's rows (``contact_compliance``'s kernels on the dense route)."""
    contacts, path, slot, needed, world = _law_inputs(device)
    row_type, row_parent, phi, target, diag, row_mu = (wp.array(a, device=device) for a in _law_rows())
    count = wp.array([LAW_ACTIVE_ROWS], dtype=int, device=device)
    rhs = wp.zeros((1, LAW_ROWS), dtype=float, device=device)
    gamma = wp.zeros((1, LAW_ROWS), dtype=float, device=device)
    weight = wp.zeros((1, LAW_ROWS), dtype=float, device=device)
    counts = wp.zeros(2, dtype=wp.int32, device=device)
    status = wp.empty(contact_compliance._STATUS_CODES + 1, dtype=wp.int32, device=device)
    wp.launch(contact_compliance._reset_status, dim=1, inputs=[status], device=device)
    unused = wp.zeros((1, 1), dtype=float, device=device)
    unused_count = wp.zeros(1, dtype=int, device=device)
    unused_type = wp.zeros((1, 1), dtype=int, device=device)
    wp.launch(
        contact_compliance._prepare_contacts,
        dim=contacts.rigid_contact_max,
        inputs=[
            contacts.rigid_contact_count,
            *(getattr(contacts, name) for name in MATERIAL_ARRAYS),
            path,
            slot,
            world,
            needed,
            wp.float64(dt),
            count,
            row_type,
            phi,
            target,
            unused_count,
            unused_type,
            unused,
            unused,
            0,
            unused_count,
            unused_type,
            unused,
            unused,
            LAW_CFM,
            wp.float64(LAW_CFM),
        ],
        outputs=[
            diag,
            rhs,
            unused,
            unused,
            unused,
            unused,
            gamma,
            wp.zeros((1, 1), dtype=float, device=device),
            wp.zeros((1, 1), dtype=float, device=device),
            weight,
            unused,
            unused,
            counts,
            status,
        ],
        device=device,
    )
    wp.launch(
        contact_compliance._weight_friction_rows,
        dim=row_mu.shape,
        inputs=[count, row_type, row_parent, gamma, weight],
        outputs=[row_mu],
        device=device,
    )
    return {
        "gamma": gamma.numpy(),
        "rhs": rhs.numpy(),
        "diag": diag.numpy(),
        "row_mu": row_mu.numpy(),
        "counts": counts.numpy(),
        "status": status.numpy(),
    }


def _scene_law(
    device,
    dt: float,
    *,
    requested: int = LAW_ACTIVE_ROWS,
    unit_overflow: int = 0,
    restitution: float = 0.0,
    **contacts_edit,
):
    """The scene's preparation of the law guard's rows (``prepare_scene_compliance`` on a stand-in solver)."""
    contacts, path, slot, needed, _ = _law_inputs(device)
    for name, value in contacts_edit.items():
        setattr(contacts, name, value)
    row_type, row_parent, phi, target, diag, row_mu = (wp.array(a, device=device) for a in _law_rows())
    rows = scene_compliance.SceneComplianceRows(
        count=wp.array([LAW_ACTIVE_ROWS], dtype=int, device=device),
        requested=wp.array([requested], dtype=int, device=device),
        unit_overflow=wp.array([unit_overflow], dtype=int, device=device),
        row_type=row_type,
        row_parent=row_parent,
        phi=phi,
        target_velocity=target,
        diag=diag,
        rhs=wp.zeros((1, LAW_ROWS), dtype=float, device=device),
        row_mu=row_mu,
    )
    # Stale step-local values that the preparation clears.
    gamma = wp.full((1, LAW_ROWS), 9.0, dtype=float, device=device)
    solver = SimpleNamespace(
        model=SimpleNamespace(device=device),
        _compliance=SimpleNamespace(
            counts=wp.full(2, 7, dtype=wp.int32, device=device),
            dense_weight=wp.zeros((1, LAW_ROWS), dtype=float, device=device),
        ),
        _contact_path=path,
        _contact_slot=slot,
        _contact_slots_needed=needed,
        _cfm=LAW_CFM,
        _enable_restitution=True,
        _shape_restitution=wp.array([0.0, restitution, 0.0], dtype=float, device=device),
    )
    status = wp.empty(contact_compliance._STATUS_CODES + 1, dtype=wp.int32, device=device)
    wp.launch(contact_compliance._reset_status, dim=1, inputs=[status], device=device)
    scene_compliance.prepare_scene_compliance(solver, contacts, rows, gamma, status, dt=dt)
    return {
        "gamma": gamma.numpy(),
        "rhs": rows.rhs.numpy(),
        "diag": rows.diag.numpy(),
        "row_mu": rows.row_mu.numpy(),
        "counts": solver._compliance.counts.numpy(),
        "status": status.numpy(),
        "status_array": status,
    }


def test_scene_compliance_law_is_steels(test, device):
    """Prepare the scene's compliant rows as Steel's device preparation does, bit for bit.

    The scene restates the law of ``contact_compliance._prepare_contacts`` (Steel exposes it inside that kernel
    only): ``gamma`` and the bias in float64 under ``fuse_fp: False``, rounded once to float32, the diagonal's
    ``gamma - cfm`` rounded once, and Steel's per-contact checks in Steel's order. On one world of rows with every
    case of :data:`LAW_CONTACTS` (penetrating, open, touching, rigid, skipped, lost, each invalid input, both ends
    of the float32 range, a non-contact row, a row beyond the active rows, two contacts on one row, a contact past
    the count) and each step of :data:`LAW_DTS`, the scene's ``row_gamma``, right-hand sides, diagonals, friction
    coefficients, counts and status word equal Steel's kernels' on the dense route.
    """
    for dt in LAW_DTS:
        with test.subTest(dt=dt):
            steel = _steel_law(device, dt)
            scene = _scene_law(device, dt)
            for key, theirs in steel.items():
                np.testing.assert_array_equal(scene[key], theirs, err_msg=key)
            compliant = np.flatnonzero(steel["gamma"][0] > 0.0)
            test.assertGreaterEqual(compliant.size, 5)
            test.assertEqual(int(steel["counts"][0]), compliant.size)  # the shared row counts once
            test.assertEqual(int(steel["counts"][1]), 1)
            latched = set(np.flatnonzero(steel["status"][: contact_compliance._STATUS_CODES]).tolist())
            expected = {
                contact_compliance._UNMAPPED,
                contact_compliance._INVALID_STIFFNESS,
                contact_compliance._INVALID_MATERIAL,
                contact_compliance._INVALID_SEPARATION,
                contact_compliance._DENSE_TYPE,
                contact_compliance._BEYOND_ROWS,
                contact_compliance._DUPLICATE,
            }
            if dt < 1.0:
                expected.add(contact_compliance._FLOAT32_RANGE)
            test.assertEqual(latched, expected)
            # The first error is the lost contact's (contact 5), which Steel's validation raises.
            with test.assertRaisesRegex(RuntimeError, r"dropped or routed to an unsupported path \(contact 5\)"):
                contact_compliance.validate(scene["status_array"], clear=False)
    # The underflowing gamma keeps its row compliant and its friction rows weighted.
    underflow = _scene_law(device, LAW_DTS[-1])
    test.assertEqual(float(underflow["gamma"][0, 21]), float(np.finfo(np.float32).tiny))
    test.assertNotEqual(float(underflow["row_mu"][0, 22]), float(_law_rows()[5][0, 22]))


def test_scene_compliance_latches_step_errors(test, device):
    """Latch the step's global errors into the status word with Steel's codes and messages.

    A positive shape restitution latches Steel's restitution error, rows requested past the rows kept and units
    dropped past ``scene_unit_max`` (whose rows the requested rows do not hold) latch the overflowing-rows error,
    and a contact count past the buffer latches the overflowing-input error (each a global
    error, ahead of every contact's): ``contact_compliance.validate`` raises Steel's message from the scene's word.
    Without them the guard's first error is a contact's.
    """
    dt = LAW_DTS[0]
    cases = (
        ({"restitution": 0.5}, ValueError, r"^contact_compliance requires zero shape restitution$"),
        ({"requested": LAW_ACTIVE_ROWS + 1}, RuntimeError, r"^contact_compliance rejects overflowing solver rows$"),
        ({"unit_overflow": 1}, RuntimeError, r"^contact_compliance rejects overflowing solver rows$"),
        (
            {"rigid_contact_count": wp.array([len(LAW_CONTACTS) + 1], dtype=int, device=device)},
            RuntimeError,
            r"^contact_compliance rejects overflowing contact input$",
        ),
    )
    for options, error, message in cases:
        with test.subTest(case=next(iter(options))):
            scene = _scene_law(device, dt, **options)
            with test.assertRaisesRegex(error, message):
                contact_compliance.validate(scene["status_array"], clear=False)


class _CompliantRun(_Run):
    """A :class:`_Run` with ``contact_compliance`` whose contacts carry one material for every contact.

    The material arrays are filled after every collision, so that the step reads them as the collision left the
    contacts (``material``: stiffness, damping and friction scale; :data:`STIFFNESS` and the rest by default).
    """

    def __init__(self, model, rows, overrides, *, material=(STIFFNESS, DAMPING, FRICTION_SCALE), **options):
        super().__init__(model, rows, overrides, **{"contact_compliance": True, **options})
        self.material = material
        if self.contacts is not None:
            for name in MATERIAL_ARRAYS:
                setattr(
                    self.contacts, name, wp.zeros(self.contacts.rigid_contact_max, dtype=float, device=model.device)
                )

    def collide(self):
        self.pipeline.collide(self.state_0, self.contacts)
        for name, value in zip(MATERIAL_ARRAYS, self.material, strict=True):
            getattr(self.contacts, name).fill_(value)

    def step(self):
        if self.pipeline is not None:
            self.collide()
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, DT)
        self.state_0, self.state_1 = self.state_1, self.state_0


class _CompliantStepper(_CompliantRun):
    """A :class:`_CompliantRun` stepping fixed state buffers, so that one captured step replays every later step."""

    def step(self):
        self.collide()
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, DT)
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            wp.copy(getattr(self.state_0, name), getattr(self.state_1, name))


def _compliant_rows(solver) -> np.ndarray:
    """The scene's rows with a positive ``gamma``: the compliant contacts' normal rows."""
    rows = int(solver._constraint_count.numpy()[0])
    return np.flatnonzero(solver._scene.row_gamma.numpy()[0, :rows] > 0.0)


def test_compliance_matches_reference_law(test, device):
    """Assemble and solve one compliant contact as the reference law does, closed and open.

    ``run_fixture``'s 0.3 kg sphere on a vertical slider rests 1 mm deep in the ground with the stiffness 3000 N/m
    and the damping 20 N s/m, one contact. After one step of the scene, its normal row's ``gamma`` and bias equal
    ``normal_coefficients``' rounded once to float32 (Steel's float64 law), the row's diagonal is its Delassus entry
    plus ``gamma`` (the constraint force mixing replaced), and its impulse equals the reference's PGS
    (``solve_reference``) on the row's own Delassus entry and incident velocity within 1e-5. The same sphere 1 mm
    above the ground has a speculative contact: its row takes the spring-only law (the dashpot off across the open
    gap), and neither the scene nor the reference pushes, the gap staying open through the step.
    """
    for case, offset in (("penetrating", -REFERENCE_DEPTH), ("speculative", REFERENCE_DEPTH)):
        with test.subTest(contact=case):
            _, paths, solver, _ = run_fixture(
                device=device,
                articulated=True,
                enabled=True,
                steps=1,
                height=0.05 + offset,
                dt=REFERENCE_DT,
                iterations=REFERENCE_ITERATIONS,
                stiffness=REFERENCE_STIFFNESS,
                solver_options={"solve_mode": "scene"},
            )
            scene = solver._scene
            test.assertTrue(scene.features.compliance)
            test.assertTrue(scene.sides_kernel.key.endswith("_cmp"))
            test.assertEqual(paths, {0})
            test.assertEqual(solver.compliance_contact_count, 1)
            rows = _compliant_rows(solver)
            test.assertEqual(rows.size, 1)
            row = int(rows[0])
            test.assertEqual(int(solver._row_type.numpy()[0, row]), PGS_CONSTRAINT_TYPE_CONTACT)
            phi = float(solver._phi.numpy()[0, row])
            test.assertEqual(phi > 0.0, offset > 0.0)
            test.assertEqual(float(solver._target_velocity.numpy()[0, row]), 0.0)
            gamma, bias = normal_coefficients(REFERENCE_STIFFNESS, REFERENCE_DAMPING, phi, dt=REFERENCE_DT)
            if phi > 0.0:
                # The spring-only law: no damping in the denominator.
                np.testing.assert_allclose(gamma, 1.0 / (REFERENCE_DT * REFERENCE_DT * REFERENCE_STIFFNESS))
            ours_gamma = float(scene.row_gamma.numpy()[0, row])
            ours_bias = float(solver._rhs.numpy()[0, row])
            test.assertEqual(ours_gamma, float(np.float32(gamma)))
            test.assertEqual(ours_bias, float(np.float32(bias)))
            # The row's Delassus entry from its factor values, and its incident velocity J v_hat.
            support = solver._sparse_row_dof.numpy()[0, row] >= 0
            delassus = float(np.sum(solver._sparse_row_factor.numpy()[0, row][support].astype(np.float64) ** 2))
            test.assertGreater(delassus, 0.0)
            np.testing.assert_allclose(float(solver._diag.numpy()[0, row]), delassus + ours_gamma, rtol=DIAG_RTOL)
            incident = float(solver._sparse_row_incident.numpy()[0, row])
            impulse, _ = solve_reference(
                [[delassus]],
                [incident],
                [phi],
                [REFERENCE_STIFFNESS],
                [REFERENCE_DAMPING],
                dt=REFERENCE_DT,
                iterations=REFERENCE_ITERATIONS,
                device=device,
            )
            ours = float(solver._impulses.numpy()[0, row])
            if phi > 0.0:
                test.assertEqual((ours, float(impulse[0])), (0.0, 0.0))
            else:
                test.assertGreater(float(impulse[0]), 0.0)
                np.testing.assert_allclose(ours, float(impulse[0]), rtol=IMPULSE_RTOL)


@unittest.skipUnless(hasattr(SolverSteel, "scene_statistics"), "needs the read-back API")
def test_compliance_tracks_steel(test, device):
    """Step the compliant contact scene as Steel's solve does.

    The contact scene (an articulated box with a driven arm and two free boxes per world, all sliding at 1.5 m/s)
    with ``contact_compliance`` and point friction, every contact of stiffness 2e4 N/m, damping 50 N s/m and friction
    scale 0.5: the velocities after the first step agree with Steel's within 1e-5 (and 1e-6), and the joint
    coordinates after 30 steps within 1e-3 RMS, with compliant rows in every step. Each kept contact has a normal row
    of its own (the scene's placement gives each contact rows of its own; Steel's duplicate check would latch).
    """
    model = _build_contact_scene(device)

    def compliant(solver):
        return solver.scene_statistics()["compliant_contacts"] > 0

    scene, steel = _track_steel(
        test, model, runner=_CompliantRun, witness=compliant, contact_compliance=True, friction_anchor_beta=0.0
    )
    test.assertTrue(scene.solver._scene.features.compliance)
    count = int(scene.contacts.rigid_contact_count.numpy()[0])
    slots = scene.solver._contact_slot.numpy()[:count]
    kept = slots[slots >= 0]
    test.assertGreater(kept.size, 1)
    test.assertEqual(np.unique(kept).size, kept.size)
    test.assertEqual(scene.solver.compliance_contact_count, steel.solver.compliance_contact_count)
    test.assertEqual(scene.solver.compliance_skipped_contact_count, steel.solver.compliance_skipped_contact_count)


@unittest.skipUnless(hasattr(SolverSteel, "scene_statistics"), "needs the read-back API")
def test_compliance_replays_under_capture(test, device):
    """Replay one captured compliant step exactly as eager steps.

    The compliant contact scene, sliding: the scene prepares its compliant rows on the device, so one captured step
    (the collision, the material fill and the solver step) replays the next 29 steps with ``joint_q`` and
    ``joint_qd`` equal to the eager steps' at every step, and the compliant contacts and their friction rows'
    coefficients equal the eager step's.
    """
    model = _build_contact_scene(device)
    eager = _CompliantStepper(model, TRACK_ROWS, {})
    captured = _CompliantStepper(model, TRACK_ROWS, {})
    for stepper in (eager, captured):
        _set_box_velocities(model, stepper.state_0)

    def assert_equal(step):
        for name in ("joint_q", "joint_qd"):
            a, b = getattr(eager.state_0, name).numpy(), getattr(captured.state_0, name).numpy()
            test.assertTrue(np.array_equal(a, b), f"{name} differs after step {step}")
        np.testing.assert_array_equal(captured.solver._scene.row_gamma.numpy(), eager.solver._scene.row_gamma.numpy())
        np.testing.assert_array_equal(captured.solver._row_mu.numpy(), eager.solver._row_mu.numpy())

    eager.step()
    captured.step()
    assert_equal(1)
    with wp.ScopedCapture(device) as recorded:
        captured.step()
    for step in range(2, CAPTURE_STEPS + 1):
        eager.step()
        wp.capture_launch(recorded.graph)
        assert_equal(step)
        test.assertGreater(eager.solver.scene_statistics()["compliant_contacts"], 0)
    test.assertEqual(captured.solver.compliance_contact_count, eager.solver.compliance_contact_count)


def test_compliant_friction_matches_steel_bitwise(test, device):
    """Weigh each compliant contact's friction rows' coefficient by its friction scale, as Steel's arrays.

    One step of the contact scene with the friction scale 0.25 (and a zero scale, which Steel takes as 1): every
    compliant contact's two friction rows hold Steel's coefficient bit for bit, from Steel's dense rows for the
    articulated box's contacts and from its free-body rows for the free boxes'.
    """
    model = _build_contact_scene(device)
    for scale in (0.25, 0.0):
        with test.subTest(friction_scale=scale):
            runs = {}
            for name, options in (("scene", {}), ("steel", {"solve_mode": "serial"})):
                run = _CompliantRun(model, TRACK_ROWS, {}, material=(STIFFNESS, DAMPING, scale), **options)
                _set_box_velocities(model, run.state_0)
                run.step()
                runs[name] = run
            scene, steel = runs["scene"].solver, runs["steel"].solver
            count = int(runs["scene"].contacts.rigid_contact_count.numpy()[0])
            test.assertEqual(count, int(runs["steel"].contacts.rigid_contact_count.numpy()[0]))
            test.assertGreater(scene.compliance_contact_count, 0)
            test.assertEqual(scene.compliance_contact_count, steel.compliance_contact_count)
            paths = steel._contact_path.numpy()[:count]
            kept = np.flatnonzero(steel._contact_slot.numpy()[:count] >= 0)
            test.assertEqual(kept.size, steel.compliance_contact_count)
            test.assertEqual(set(paths[kept].tolist()), {0, 1})
            ours = scene._row_mu.numpy()[0]
            for contact in kept:
                slot = int(scene._contact_slot.numpy()[contact])
                theirs_slot = int(steel._contact_slot.numpy()[contact])
                world = int(steel._contact_world.numpy()[contact])
                theirs = (steel._row_mu if paths[contact] == 0 else steel._mf_row_mu).numpy()[world]
                for child in (1, 2):
                    test.assertEqual(int(scene._row_type.numpy()[0, slot + child]), PGS_CONSTRAINT_TYPE_FRICTION)
                    test.assertEqual(int(scene._row_parent.numpy()[0, slot + child]), slot)
                    test.assertEqual(ours[slot + child], theirs[theirs_slot + child], f"contact {contact}")
            unweighted = _Run(model, TRACK_ROWS, {})
            _set_box_velocities(model, unweighted.state_0)
            unweighted.step()
            slots = scene._contact_slot.numpy()[kept]
            expected = unweighted.solver._row_mu.numpy()[0, slots + 1] * np.float32(scale if scale else 1.0)
            np.testing.assert_array_equal(ours[slots + 1], expected)


@unittest.skipUnless(hasattr(SolverSteel, "scene_statistics"), "needs the read-back API")
def test_compliance_counts_match_steel(test, device):
    """Count the compliant and the excluded contacts as Steel does, and raise where Steel raises.

    On ``run_fixture``'s sphere, free and on a slider: a contact past the gap gate is excluded on purpose, counted as
    skipped and not compliant, as in Steel; a zero stiffness counts in neither (a kinematic sphere, Steel's other
    exclusion, leaves no responding body: ``test_compliance.test_nonresponding_contact_is_skipped`` checks it). An
    eager step whose compliant contact a one-row budget drops, or whose stiffness is negative or not finite, raises
    the error Steel's eager step raises, word for word: the scene latches Steel's status codes.
    """
    cases = {
        "gap gate": {"height": 0.052, "solver_options": {"contact_gap_gate": 0.001}},
        "zero stiffness": {"stiffness": 0.0},
        "compliant": {},
    }
    for name, options in cases.items():
        for articulated in (False, True):
            with test.subTest(case=name, articulated=articulated):
                solvers = {}
                for solve, mode in (("scene", "scene"), ("steel", "serial")):
                    solver_options = {**options.get("solver_options", {}), "solve_mode": mode}
                    fixture = {key: value for key, value in options.items() if key != "solver_options"}
                    _, _, solvers[solve], _ = run_fixture(
                        device=device,
                        articulated=articulated,
                        enabled=True,
                        steps=1,
                        solver_options=solver_options,
                        **fixture,
                    )
                scene, steel = solvers["scene"], solvers["steel"]
                stats = scene.scene_statistics()
                test.assertEqual(stats["compliant_contacts"], steel.compliance_contact_count)
                test.assertEqual(stats["compliant_contacts_skipped"], steel.compliance_skipped_contact_count)
                test.assertEqual(scene.compliance_contact_count, steel.compliance_contact_count)
                test.assertEqual(scene.compliance_skipped_contact_count, steel.compliance_skipped_contact_count)
                expected = {"gap gate": (0, 1), "zero stiffness": (0, 0), "compliant": (1, 0)}
                test.assertEqual((stats["compliant_contacts"], stats["compliant_contacts_skipped"]), expected[name])
    budget = {"articulation_constraint_max": 1, "free_body_constraint_max": 1, "warn_constraint_overflow": False}
    failing = {"dropped": ({}, budget), "negative": ({"stiffness": -1.0}, {}), "nan": ({"stiffness": math.nan}, {})}
    for name, (fixture, solver_options) in failing.items():
        for articulated in (False, True):
            with test.subTest(case=name, articulated=articulated):
                raised = {}
                for mode in ("scene", "serial"):
                    with test.assertRaises((RuntimeError, ValueError)) as context:
                        run_fixture(
                            device=device,
                            articulated=articulated,
                            enabled=True,
                            steps=1,
                            solver_options={**solver_options, "solve_mode": mode},
                            **fixture,
                        )
                    raised[mode] = (type(context.exception), str(context.exception))
                test.assertEqual(raised["scene"], raised["serial"])


def test_scene_step_checks_keep_steel_messages(test, device):
    """Raise ``start_step``'s step checks in scene mode with Steel's messages, before any work is launched.

    ``start_scene_step`` runs ``start_step``'s checks, which read no device memory. A compliant scene solver
    (``run_fixture``'s articulated sphere) stepped with a zero, negative or non-finite ``dt``, without contacts, with
    contacts lacking each per-contact material array, and with one of them shorter than the contact buffer raises
    the ``ValueError`` that Steel's ``start_step`` raises for the same contacts and ``dt``, word for word.
    """
    _, _, scene, _ = run_fixture(
        device=device, articulated=True, enabled=True, steps=1, solver_options={"solve_mode": "scene"}
    )
    test.assertEqual(scene.solve_mode, "scene")
    model = scene.model
    state_0, state_1, control = model.state(), model.state(), model.control()
    steel = SimpleNamespace(model=SimpleNamespace(device=model.device))

    def contacts_with(missing=None, short=None):
        contacts = newton.Contacts(CONTACT_CAPACITY, 0, device=model.device)
        for name in MATERIAL_ARRAYS:
            size = CONTACT_CAPACITY - 1 if name == short else CONTACT_CAPACITY
            setattr(
                contacts,
                name,
                None if name == missing else wp.zeros(size, dtype=float, device=model.device),
            )
        return contacts

    def assert_same_message(contacts, dt):
        with test.assertRaises(ValueError) as expected:
            start_step(steel, contacts, dt)
        with test.assertRaises(ValueError) as raised:
            scene.step(state_0, state_1, control, contacts, dt)
        test.assertEqual(str(raised.exception), str(expected.exception))

    for dt in (0.0, -DT, float("nan"), float("inf")):
        with test.subTest(dt=dt):
            assert_same_message(contacts_with(), dt)
    with test.subTest(contacts=None):
        assert_same_message(None, DT)
    for name in MATERIAL_ARRAYS:
        with test.subTest(missing=name):
            assert_same_message(contacts_with(missing=name), DT)
        with test.subTest(short=name):
            assert_same_message(contacts_with(short=name), DT)


def test_compliance_matches_the_serial_oracle(test, device):
    """Solve compliant rows in the phased sweep exactly as one thread walking the rows does, in every phase order.

    Four lying humanoids 0.4 m apart overlap, so units join pairs of them, and every contact is compliant: compliant
    normal rows run through phase B (the colors, the block and the tail) and the coupled sides' sweeps. The phases
    (the default color loop, and the while loop, whose tail one thread walks) match the serial oracle
    (``scene_sweep="serial"``) bit for bit over 30 steps in each phase order.
    """
    model = _humanoids(device, per_world=4, spacing=0.4)

    def two_sided_compliant(solver):
        begin, end = solver._scene.row_ranges.numpy()[RANGE_TWO_SIDED]
        return bool(np.any(solver._scene.row_gamma.numpy()[0, begin:end] > 0.0))

    _assert_phases_match_serial_oracle(
        test,
        model,
        {"contact_compliance": True},
        witness=two_sided_compliant,
        what="no two-sided row was compliant",
        runner=_CompliantRun,
    )


class TestSteelSceneCompliance(unittest.TestCase):
    pass


for _fn in (test_scene_compliance_law_is_steels, test_scene_compliance_latches_step_errors):
    add_function_test(TestSteelSceneCompliance, _fn.__name__, _fn, devices=get_test_devices())

# CUDA only, each for the reason beside it: Steel refuses contact compliance on the CPU, and so does the scene there,
# whose construction runs Steel's checks first.
for _fn in (
    test_compliance_matches_reference_law,  # Steel's compliance
    test_compliance_tracks_steel,  # Steel's compliance
    test_compliance_replays_under_capture,  # graph capture
    test_compliant_friction_matches_steel_bitwise,  # Steel's compliance
    test_compliance_counts_match_steel,  # Steel's compliance
    test_scene_step_checks_keep_steel_messages,  # Steel's compliance
    test_compliance_matches_the_serial_oracle,  # the phases; the CPU sweeps the serial oracle alone
):
    add_function_test(TestSteelSceneCompliance, _fn.__name__, _fn, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
