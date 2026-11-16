# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact compliance of the scene mode (``solve_mode="scene"`` with ``contact_compliance=True``).

The scene prepares its compliant rows on the device as Steel does (:func:`contact_compliance.prepare_rows
<newton._src.solvers.steel.contact_compliance.prepare_rows>`), with Steel's float64 law, Steel's per-contact checks in
Steel's order and Steel's status codes, on its one row pool instead of Steel's per-world row families. A compliant
contact's normal row gets ``row_gamma = gamma``, ``rhs = bias - target`` and ``diag + (gamma - cfm)``, and its
friction rows' coefficients are weighted by the contact's friction scale (Steel's ``_weight_friction_rows``); the
scene's sweeps add ``gamma * lambda`` to the row's residual. Errors latch into a status word of
:class:`~newton._src.solvers.steel.contact_compliance.ComplianceBuffers`'s layout, which
:func:`~newton._src.solvers.steel.contact_compliance.validate` reads with Steel's messages. Nothing is read back on the
host, so a step can be captured.

Steel exposes its law only inside ``contact_compliance._prepare_contacts``, so :func:`scene_compliant_law` restates it
under the same module options (``fuse_fp: False``), and ``test_scene_compliance.test_scene_compliance_law_is_steels``
checks the scene's preparation against Steel's kernels on the same inputs, bit for bit.
"""

from typing import NamedTuple

import warp as wp

from . import contact_compliance
from .contact_compliance import (
    _BEYOND_ROWS,
    _DENSE_TYPE,
    _DUPLICATE,
    _FLOAT32_RANGE,
    _INPUT_OVERFLOW,
    _INVALID_MATERIAL,
    _INVALID_SEPARATION,
    _INVALID_STIFFNESS,
    _ROW_OVERFLOW,
    _STATUS_CODES,
    _UNMAPPED,
    _latch,
)
from .kernels import PGS_CONSTRAINT_TYPE_CONTACT, WARP_DETERMINISM_OFF

# SolverSteel rejects models with ``requires_grad``, so no kernel needs an adjoint.
# Unfused arithmetic keeps the float64 coefficient law equal to Steel's (contact_compliance's options).
wp.set_module_options({"enable_backward": False, "fuse_fp": False, **WARP_DETERMINISM_OFF})


class SceneComplianceRows(NamedTuple):
    """The scene's row storage that :func:`prepare_scene_compliance` reads and writes, one solve world ``[1, rows]``."""

    count: wp.array
    """Rows kept by the step, ``[1]``."""
    requested: wp.array
    """Rows the step's kept units requested, ``[1]``; more than :attr:`count` when rows were dropped at the capacity."""
    unit_overflow: wp.array
    """Units the step dropped past ``scene_unit_max``, ``[1]``; their rows never enter :attr:`requested`."""
    row_type: wp.array
    row_parent: wp.array
    phi: wp.array
    target_velocity: wp.array
    diag: wp.array
    """Each row's diagonal; a compliant row's gains ``gamma - cfm``."""
    rhs: wp.array
    """Each row's right-hand side; a compliant row's becomes ``bias - target``."""
    row_mu: wp.array
    """Each row's friction coefficient; a compliant contact's friction rows' are weighted by its friction scale."""


@wp.func
def scene_compliant_law(stiffness: float, damping: float, separation: float, dt: wp.float64) -> wp.vec2d:
    """``(gamma, bias)`` of one compliant row in float64, the law of ``contact_compliance._prepare_contacts``.

    ``denominator = dt * k + (c if separation <= 0 else 0)`` (a dashpot must not act across an open gap),
    ``gamma = 1 / (dt * denominator)`` [1/kg] and ``bias = k * separation / denominator`` [m/s], with Steel's
    expressions in Steel's order. The caller checks the inputs and the results' float32 range as Steel does.
    """
    k64 = wp.float64(stiffness)
    c64 = wp.float64(0.0)
    if separation <= 0.0:
        c64 = wp.float64(damping)
    denominator = dt * k64 + c64
    gamma = wp.float64(1.0) / (dt * denominator)
    bias = k64 * wp.float64(separation) / denominator
    return wp.vec2d(gamma, bias)


@wp.kernel
def _check_scene_inputs(
    contact_count: wp.array[int],
    contact_capacity: int,
    row_count: wp.array[int],
    rows_requested: wp.array[int],
    unit_overflow: wp.array[int],
    status: wp.array[wp.int32],
):
    """Latch Steel's global input errors (``contact_compliance._check_inputs``) for the scene's one row pool.

    Rows are lost when the kept units requested more than the pool kept or when units were dropped past
    ``scene_unit_max``; Steel latches its row-overflow code on any lost reservation.
    """
    if contact_count[0] < 0 or contact_count[0] > contact_capacity:
        _latch(status, _INPUT_OVERFLOW, -1)
    if rows_requested[0] > row_count[0] or unit_overflow[0] > 0:
        _latch(status, _ROW_OVERFLOW, -1)


@wp.kernel
def _prepare_scene_contacts(
    contact_count: wp.array[int],
    stiffness: wp.array[float],
    damping: wp.array[float],
    friction_scale: wp.array[float],
    paths: wp.array[int],
    slots: wp.array[int],
    slots_needed: wp.array[int],
    dt: wp.float64,
    row_count: wp.array[int],
    row_type: wp.array2d[int],
    phi: wp.array2d[float],
    target: wp.array2d[float],
    pgs_cfm64: wp.float64,
    diag: wp.array2d[float],
    rhs: wp.array2d[float],
    gamma_out: wp.array2d[float],
    weight_out: wp.array2d[float],
    counts: wp.array[wp.int32],
    status: wp.array[wp.int32],
):
    """Prepare each compliant contact's normal row (grid: contacts), ``contact_compliance._prepare_contacts``.

    Steel's dense route on the scene's one row pool: the scene places every kept contact's rows in the pool,
    whichever path the allocator recorded, so a contact's slot is its row and every route is the dense one.
    """
    contact = wp.tid()
    if contact >= contact_count[0]:
        return
    k = stiffness[contact]
    if not wp.isfinite(k) or k < 0.0:
        _latch(status, _INVALID_STIFFNESS, contact)
        return
    if k == 0.0:
        return
    path = paths[contact]
    slot = slots[contact]
    # No capacity request means the allocator intentionally excluded this pair
    # (non-responding bodies, world filtering or a positive-gap gate).
    if path == -1 and slot == -1 and slots_needed[contact] == 0:
        wp.atomic_add(counts, 1, 1)
        return
    if path < 0 or path > 2 or slot < 0 or slot >= gamma_out.shape[1]:
        _latch(status, _UNMAPPED, contact)
        return
    if slot >= row_count[0]:
        _latch(status, _BEYOND_ROWS, contact)
        return
    c = damping[contact]
    scale = friction_scale[contact]
    if not wp.isfinite(c) or c < 0.0 or not wp.isfinite(scale) or scale < 0.0:
        _latch(status, _INVALID_MATERIAL, contact)
        return
    weight = scale
    if scale == 0.0:
        weight = 1.0
    row_phi = phi[0, slot]
    if not wp.isfinite(row_phi):
        _latch(status, _INVALID_SEPARATION, contact)
        return
    law = scene_compliant_law(k, c, row_phi, dt)
    gamma = law[0]
    bias = law[1]
    limit = wp.float64(3.4028234663852886e38)
    if wp.abs(gamma) > limit or wp.abs(bias) > limit:
        _latch(status, _FLOAT32_RANGE, contact)
        return
    # A positive gamma stays nonzero in float32: zero marks a row without compliance.
    gamma32 = wp.max(float(gamma), 1.17549435e-38)
    bias32 = float(bias)
    if row_type[0, slot] != PGS_CONSTRAINT_TYPE_CONTACT:
        _latch(status, _DENSE_TYPE, contact)
        return
    if wp.atomic_add(gamma_out, 0, slot, gamma32) != 0.0:
        _latch(status, _DUPLICATE, contact)
        return
    weight_out[0, slot] = weight
    rhs[0, slot] = bias32 - target[0, slot]
    # The compliance replaces the row's constraint force mixing, rounded once from float64.
    diag[0, slot] = diag[0, slot] + float(gamma - pgs_cfm64)
    wp.atomic_add(counts, 0, 1)


def start_scene_step(solver, contacts, dt):
    """Check the step's host-side inputs with Steel's :func:`~newton._src.solvers.steel.contact_compliance.start_step`.

    Steel's checks read no device data (friction anchors, a finite positive ``dt``, the per-contact material arrays
    and their shapes), so a scene step stays capturable; the checks of the step's data run on the device in
    :func:`prepare_scene_compliance` and latch into its status word.
    """
    contact_compliance.start_step(solver, contacts, dt)


def prepare_scene_compliance(solver, contacts, rows: SceneComplianceRows, gamma_out, status, *, dt: float) -> None:
    """Prepare the scene's compliant rows on the device, as ``contact_compliance.prepare_rows`` does Steel's.

    Run after the scene's rows are prepared and their depenetration is capped. Clears ``gamma_out`` (``[1, rows]``,
    the scene's ``row_gamma``) and the step's counts, latches the step's global errors (contact input past the buffer,
    rows requested past the rows kept or units dropped past ``scene_unit_max``, a positive shape restitution with
    restitution enabled), then writes each
    compliant contact's ``gamma``, right-hand side and diagonal into ``rows`` and weights its friction rows.

    Args:
        solver: The scene's Steel solver: its contact mapping (``_contact_path``, ``_contact_slot``,
            ``_contact_slots_needed``), ``_cfm``, the restitution inputs, and its
            :class:`~newton._src.solvers.steel.contact_compliance.ComplianceBuffers` (``_compliance``), whose
            ``counts`` receive ``[compliant, skipped]`` and whose ``dense_weight`` (shaped like ``gamma_out``)
            receives the compliant rows' friction weights.
        contacts: The step's contacts with their per-contact material arrays.
        rows: The scene's row storage.
        gamma_out: Each row's compliance, ``0`` but on the compliant normal rows.
        status: A status word of ``ComplianceBuffers``'s layout (``status`` for captured steps, ``eager_status`` for
            eager ones) into which errors latch with Steel's codes.
        dt: The step [s].
    """
    buffers = solver._compliance
    device = solver.model.device
    if status.shape != (_STATUS_CODES + 1,):
        raise ValueError(f"the compliance status word must have shape ({_STATUS_CODES + 1},), got {status.shape}")
    if gamma_out.shape != rows.diag.shape or buffers.dense_weight.shape != gamma_out.shape:
        raise ValueError(
            f"the scene's compliance arrays must share the rows' shape {rows.diag.shape}: gamma {gamma_out.shape}, "
            f"weight {buffers.dense_weight.shape}"
        )
    gamma_out.zero_()
    buffers.counts.zero_()
    wp.launch(
        _check_scene_inputs,
        dim=1,
        inputs=[
            contacts.rigid_contact_count,
            contacts.rigid_contact_max,
            rows.count,
            rows.requested,
            rows.unit_overflow,
            status,
        ],
        device=device,
    )
    if solver._enable_restitution:
        wp.launch(
            contact_compliance._check_restitution,
            dim=solver._shape_restitution.shape[0],
            inputs=[solver._shape_restitution, status],
            device=device,
        )
    wp.launch(
        _prepare_scene_contacts,
        dim=contacts.rigid_contact_max,
        inputs=[
            contacts.rigid_contact_count,
            contacts.rigid_contact_stiffness,
            contacts.rigid_contact_damping,
            contacts.rigid_contact_friction,
            solver._contact_path,
            solver._contact_slot,
            solver._contact_slots_needed,
            wp.float64(dt),
            rows.count,
            rows.row_type,
            rows.phi,
            rows.target_velocity,
            wp.float64(solver._cfm),
        ],
        outputs=[rows.diag, rows.rhs, gamma_out, buffers.dense_weight, buffers.counts, status],
        device=device,
    )
    wp.launch(
        contact_compliance._weight_friction_rows,
        dim=rows.row_mu.shape,
        inputs=[rows.count, rows.row_type, rows.row_parent, gamma_out, buffers.dense_weight],
        outputs=[rows.row_mu],
        device=device,
    )


def scene_compliance_counts(solver) -> tuple[int, int]:
    """The ``(compliant, skipped)`` contact counts of the scene's latest prepared step.

    The scene counts into Steel's ``ComplianceBuffers.counts``, so this is Steel's
    :func:`~newton._src.solvers.steel.contact_compliance.counts`: it reads the device on the host and raises
    :class:`RuntimeError` during CUDA graph capture.
    """
    return contact_compliance.counts(solver)
