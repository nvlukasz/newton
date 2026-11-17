# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The scene solve of SolverSteel (``solve_mode="scene"``): the whole model as one solve world.

The scene mode folds every model world into one solve world, so nothing in the constraint solve is per world. The
scene's rows are Steel's factor-coordinate row arrays with one world: ``_sparse_row_dof`` and ``_sparse_row_factor``
of shape ``(1, row_max, support)``, the free-body response ``_sparse_row_free_response`` ``(1, row_max,
SCENE_FREE_SUPPORT)`` when the scene has free bodies, ``_sparse_row_incident`` ``(1, row_max)`` and the velocity delta
``_sparse_factor_velocity_delta`` ``(1, coordinate_count)``. Steel's per-world shared-memory sweep, and the gate that
sizes it, do not apply.

This module ties the scene's parts together: the groups and the row plan (:func:`plan_scene_groups`,
:class:`ScenePlan`), the placement of the units' rows (:class:`SceneSolve`), the row finishing kernels, the planning
functions the solver calls at construction (:func:`setup_scene_plan` and its helpers) and the scene's stage 4
(:func:`build_scene_rows`). The units are :mod:`~newton._src.solvers.steel.scene_units`', the row builders
:mod:`~newton._src.solvers.steel.scene_rows`', the sweeps :mod:`~newton._src.solvers.steel.scene_pgs`', the spin rows
:mod:`~newton._src.solvers.steel.scene_torsion_device`', the compliant rows
:mod:`~newton._src.solvers.steel.scene_compliance`' and the warm start
:mod:`~newton._src.solvers.steel.scene_warmstart`'.

Each articulated (non-free-body) joint topology of a model needs a factor that fits one warp's static shared memory
and contact rows that fit the response kernel's (see **Groups**); mimic joints and CONNECT loop closures are rows of
the internal unit. A topology's support masks take ``ceil(DOFs / 64)`` ``uint64`` words (:attr:`SceneGroup.mask_words`),
so it may have more than 64 DOFs. Free bodies are 6-DOF blocks of the velocity delta in physical coordinates, so a row
between two of them has two 6-entry blocks; a model of free bodies only has no factor at all. :class:`ScenePlan`
records the plan selected at construction.

**Groups** (:class:`SceneGroup`, :func:`plan_scene_groups`). The plan sorts the responding articulations into scene
groups that share one response representation: a factor group per joint topology (:func:`topology_key`, the input of
the sparse factor plan), in order of first articulation, then one free group of the free bodies (the test-only kernel
overrides ``scene_group_split`` and ``scene_mask_words`` split each factor group by articulation index into several and
widen its support masks). Each factor group has its own factor storage and launches (:class:`SceneGroupArrays`): its
factors are one block of the scene's flat ``L^-1`` (:attr:`SceneSolve.Linv`), and the factor, the unconstrained solve,
the armature copy, the internal rows (one launch per row family) and the decode run once per group. The group-table
contact response kernel (``scene_response_kernel="generic"``, the default) reads each endpoint's plan and factor through
its group (:func:`scene_response_table`); its static shared scratch grows with the largest group's DOFs, and a scene
past what a block holds raises :class:`NotImplementedError` at construction. The kernel specialised on one factor plan
(``"specialized"``) remains selectable on a scene of one factor group. The row support ``S`` covers the widest row of
any group, and of two articulations together (:func:`scene_support`), so one row store serves every group.

**Rows** (:class:`SceneSolve`). The row build (:func:`build_scene_rows`, the scene's ``_stage4_build_rows``) builds the
units first (:class:`~newton._src.solvers.steel.scene_units.SceneUnits`) and then places their rows by two exclusive
scans in unit order (:func:`~newton._src.geometry.utils.scan_with_total` on preallocated scratch): every one-sided unit,
side-major in sorted order, so a side's one-sided rows are contiguous; then the two-sided units by color, the tail last.
Units whose rows would pass ``row_max`` are dropped in that order. Steel's own builders then write the rows at the
places the scans gave them, and one row-parallel kernel adds the constraint force mixing, writes the bias, or an
impacting contact's rebound target (Steel's restitution law: the rebound velocity ``-e * u_start`` the row build takes
from the step-start closing speed, ``kernels.contact_rebound_velocity``), and clears the impulses, where Steel runs a
serial loop per world.

**Sweep** (:meth:`SceneSolve.solve`). Steel's sparse row loop runs on the scene's rows in global memory: phase A sweeps
each side's one-sided rows, one launch running every iteration of the sides without two-sided units with Steel's
stationary exit per side. When the step has two-sided units, each iteration then gives the coupled sides one sweep and
runs phase B inside a conditional graph node: the two-sided units color by color, all units of a color at once (the big
colors in a conditional loop, the small ones in one block), then the tail on one thread. The solver's
``scene_phase_order`` sets the order of the two in each iteration: phase B first (``"BA"``, the default), the coupled
sides first (``"AB"``), or the coupled sides again after phase B (``"ABA"``). With Steel's summation order (the
default) one world of one humanoid steps bitwise as Steel's own solve, and the phased sweep reproduces one thread
walking the rows in the order's sequence (every row in row order for ``"AB"``) bit for bit. Steel's decode kernels then
write ``v_out``. The sweep kernels are :mod:`~newton._src.solvers.steel.scene_pgs`'.

**Velocity-only iterations** (``velocity_iterations``). After the position solve the step keeps its velocity (``v_out``,
which positions integrate) and runs Steel's velocity pass on the same rows: :func:`prepare_scene_velocity_rows` writes
their right-hand side without the position bias, Steel's ``compute_world_contact_velocity_bias`` law with ``J v`` of the
position solve formed in factor coordinates as ``J v_hat + Z delta``, and :meth:`SceneSolve.solve` with ``reset=False``
sweeps them ``velocity_iterations`` times from the position solve's ``delta`` and impulses, then decodes the refined
``v_out``, which the step publishes. As Steel's pass it solves friction from its first sweep, weighs no row
(regularization off) and, under ``velocity_drive_mode="freeze"``, skips the drive rows.

**Warm start** (``contact_warmstart``). Steel's warm start gathers each matched contact's impulses of the previous step
into its rows and seeds the sweeps from them; the scene applies the seeds to ``delta`` side by side and discards a
world's carry contact by contact on a reset (:mod:`~newton._src.solvers.steel.scene_warmstart`, whose launchers
:class:`SceneSolve` calls).

**The internal unit.** An articulation's rows of Steel's per-articulation row families (drives, mimic, connect, joint
limits, joint friction: ``FAMILY_DRIVE`` to ``FAMILY_JOINT_FRICTION``, Steel's slot order) form one one-sided unit, its
*internal unit*, ahead of its contacts. Each family's COUNT pass counts the articulation's rows into its row of the
solver's ``_scene_family_rows``, the unit carries their sum (``_scene_internal_rows``), and the families' WRITE passes
write them one family after another from the unit's first row (:meth:`SceneSolve.launch_internal_rows`). The
joint-limit family is Steel's sparse joint-limit builder, unchanged; the drive family (``drive_mode="rows"``) and the
joint-friction family build Steel's drive and joint-friction rows in the factor's coordinates. The mimic family builds
Steel's mimic rows, ``J = e_follower - c1 e_leader``, over the OR of their two DOFs' support masks, and the connect
family Steel's CONNECT loop-closure rows, three per enabled closure, through the contact response's self-contact path
(both anchors in the owner's articulation, or a prescribed parent's anchor velocity in the target; a free-body child
takes its free-body branch), one launch for every closure owner, free bodies included
(:func:`~newton._src.solvers.steel.scene_rows.get_scene_connect_rows_kernel`). Both are unclamped two-sided rows: the
sweeps' rigid update clamps contact and limit rows alone, and :func:`prepare_scene_rows` gives them the Baumgarte term
for either sign of ``phi``, as Steel does. Steel's bilateral pre-elimination is not built in the scene mode:
``enable_bilateral_preelimination=True`` runs these rows.

**Late rows and phase C.** Steel sweeps a world's joint velocity-limit rows after every other row of each iteration, so
that they have the last word, and its free bodies' velocity-limit rows last of all. An articulation's velocity-limit
rows (``FAMILY_VELOCITY_LIMIT``, ``enable_joint_velocity_limits``) form its *late unit*, and so do a free body's
(``rigid_body_max_linear_velocity`` and ``rigid_body_max_angular_velocity``, its six physical entries,
:func:`~newton._src.solvers.steel.scene_rows.build_sparse_free_velocity_limit_rows`); the scans place them after the
tail, side-major; each iteration ends with phase C, which sweeps them with Steel's stateless update after the
iteration's whole coupled pass (B, A, C under ``"BA"``), and the isolated sides run A then C in each of their sweeps.
Sides share no late row, so side-major phase C is Steel's order. With drive rows and ``fuse_joint_velocity_limits`` a
driven DOF gets no velocity-limit rows; phase C clamps its drive row's velocity statelessly after its side's velocity
limits instead (the fused clamp), the drive impulse unchanged.

**Contact torsion.** Steel's spin rows (one angular row per contact group, ``contact_torsion.py``) are late rows: the
row preparation groups the contacts on the device
(:func:`~newton._src.solvers.steel.scene_torsion_device.group_scene_torsion`) and appends one spin row per group after
every unit's rows (:meth:`SceneSolve.prepare_torsion`), and phase C starts with Steel's C1 on them, before the velocity
limits: each side's one-sided spin rows, then the two-sided ones by the color of their group's representative contact
unit (:func:`~newton._src.solvers.steel.scene_pgs.get_sweep_torsion_kernel`). C1 rescales the group's anchor tangent
pairs into its budget and bounds the spin row by the rest (``SceneFeatures.torsion``).

**The depenetration cap.** A free body's ``rigid_body_max_depenetration_velocity`` bounds the Baumgarte bias of the
penetrating contacts Steel solves on its free-body path (every side a free body or without response DOFs), as Steel's
``compute_mf_effective_mass_and_rhs`` does; the row preparation applies it after :func:`prepare_scene_rows`
(:func:`cap_scene_depenetration`, ``SceneFeatures.depenetration_cap``).

**Per-row features.** The sweep kernels' per-row features are build constants fixed at construction
(:class:`~newton._src.solvers.steel.scene_pgs.SceneFeatures`): every feature off compiles the sources the parity tests
pin, and any feature on compiles each sweep factory's one featured definition, which takes every feature's argument.
Friction position iterations gate the tangent pairs on the sweep index. Contact regularization gives a penetrating
contact's row the weight ``1 / (1 + g)`` (``SceneSolve.row_w``, written with the bias), and the sweeps relax its impulse
toward zero by the rest, as Steel's matrix-free update. Joint friction clamps a joint-friction row's impulse to its box
``[-f dt, f dt]`` (``SceneSolve.row_bound``, written by the row's builder), as Steel's projection. Drive rows replace
their impulse with the force-drive update of Steel's matrix-free solve, clamped to the row's maximum impulse, from four
descriptors per row (:func:`scene_drive_descriptors`, computed once the row's diagonal is final). A contact past
``contact_friction_gap_max`` is a unit of one row, its normal row. ``row_watermark`` counts the rows per model world
into Steel's high-water marks (:meth:`SceneSolve.accumulate_row_watermarks`). A ``friction_mode`` other than
``"fixed_normal"`` solves a flagged contact's normal and tangent rows together at its first tangent row's visit, with
Steel's bisection or Coulomb-Newton law (``SceneFeatures.friction_mode``, :func:`prepare_scene_friction_modes`).
Contact compliance prepares the compliant rows on Steel's device law
(:func:`~newton._src.solvers.steel.scene_compliance.prepare_scene_compliance`).

**The planning functions.** The solver calls :func:`setup_scene_plan` at construction, after its sparse factor
selection planned the groups (:func:`plan_scene_groups` on :func:`scene_topology_keys`, with :func:`factor_body_masks`
and :func:`scene_support`, or :func:`setup_scene_free_only` for a model of free bodies only), and sizes the row pool
with :func:`scene_row_capacity`. Each takes the solver as its first argument and reads the solver attributes its
docstring names; none modifies Steel's state beyond the ``_scene_*`` attributes and the row arrays it documents.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Literal

import numpy as np
import warp as wp

from ...geometry.utils import _scan_scratch_size, scan_with_total
from . import contact_compliance
from .contact_torsion import _TOUCH_TOLERANCE
from .friction_patches import WARP_DETERMINISM_OFF, link_patch_rows
from .kernels import (
    _CONTACT_END_GAP_SLOP,
    PGS_CONSTRAINT_TYPE_CONNECT,
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_TARGET,
    PGS_CONSTRAINT_TYPE_MIMIC,
    ROW_SLOTS_COUNT,
    ROW_SLOTS_WRITE,
    ROW_WATERMARK_FAMILY_STRIDE,
    accumulate_contact_watermark,
    allocate_world_contact_slots,
    contact_restitution_fires,
    gather_group_armature,
    prepare_world_contact_rows,
    scatter_augmented_drive_dof_K,
)
from .scene_compliance import SceneComplianceRows, prepare_scene_compliance
from .scene_pgs import (
    RANGE_ALL,
    RANGE_TAIL,
    RANGE_TWO_SIDED,
    SCENE_BLOCK_COLOR_MAX,
    SCENE_BLOCK_THREADS,
    SCENE_COLOR_LOOPS,
    SCENE_COLOR_THREADS_MAX,
    SCENE_FRICTION_DENSE,
    SCENE_FRICTION_FIXED_NORMAL,
    SCENE_FRICTION_FREE,
    SCENE_PASS_COUPLED,
    SCENE_PASS_ISOLATED,
    SCENE_PASS_LATE,
    SCENE_PASS_TORSION,
    SCENE_RESPONSE_KERNELS,
    SCENE_SIDE_LANES,
    SCENE_SWEEP_BLOCK_DIM,
    SCENE_SWEEPS,
    SCENE_UNIT_LANES,
    SceneFeatures,
    _check_sweep,
    _default_sum_order,
    _feature_suffix,
    _has_phase_c,
    _ScenePass,
    advance_color,
    get_sweep_block_kernel,
    get_sweep_color_kernel,
    get_sweep_order_serial_kernel,
    get_sweep_rows_serial_kernel,
    get_sweep_sides_kernel,
    get_sweep_torsion_kernel,
    reset_color_loop,
    scene_row_ranges,
)
from .scene_rows import (
    FAMILY_CONNECT,
    FAMILY_COUNT,
    FAMILY_DRIVE,
    FAMILY_JOINT_FRICTION,
    FAMILY_LIMIT,
    FAMILY_MIMIC,
    FAMILY_VELOCITY_LIMIT,
    WARP_SHARED_BYTES_PER_THREAD,
    _finish_scene_row,
    _get_scene_contact_response_kernel,
    _scene_row_dot,
    get_scene_connect_rows_kernel,
    get_sparse_drive_marker_kernel,
    get_sparse_free_velocity_limit_rows_kernel,
    get_sparse_joint_dof_rows_kernel,
    scene_contact_response_dof_limit,
    scene_contact_response_shared_bytes,
)
from .scene_torsion_device import (
    SCENE_TORSION_ROW_WORKERS,
    SceneTorsionGroups,
    SceneTorsionShapes,
    get_scene_torsion_rows_kernel,
    group_scene_torsion,
    prepare_scene_velocity_torsion,
    scene_torsion_counts,
    scene_torsion_eligibility,
    scene_torsion_finish,
    scene_torsion_keys,
    scene_torsion_place,
    scene_torsion_routing,
)
from .scene_units import (
    _SIDE_LATE,
    _SIDE_ONE_SIDED,
    UNIT_CONTACT_LATE,
    _contact_unit_sides,
    _latch_side_world,
    _scene_contact_on_free_path,
    _unit_model_world,
)
from .scene_warmstart import clear_warmstart, finish_gather_mu, keep_gather_mu, seed
from .sparse_contact import (
    _get_sparse_contact_response_kernel,
    apply_sparse_factor_velocity,
    apply_sparse_free_velocity,
)
from .sparse_mass_matrix import (
    _get_crba_sparse_factor_kernel,
    _SparseMassMatrixIndices,
    _SparseMassMatrixPlan,
    distinct_mask_values,
    mask_value,
    solve_sparse_mass_matrix,
)

wp.set_module_options({"enable_backward": False, **WARP_DETERMINISM_OFF})


SCENE_FREE_SUPPORT = 12
"""Entries of a row between two free bodies, two 6-DOF blocks in physical coordinates.

It is the least row support of a scene with free bodies and the width of a row's free-body
response ``H^-1 J^T``.
"""


def _mask_words(size: int) -> int:
    """``uint64`` words of a support mask over ``size`` coordinates, ``ceil(size / 64)``."""
    return -(-size // 64)


@dataclass(frozen=True, eq=False)
class SceneGroup:
    """Articulations of the scene that share one response representation, fixed at construction.

    A ``"factor"`` group holds the articulations of one joint topology (:func:`topology_key`). They
    share one sparse factor plan, and their rows are in its factor coordinates. Their factors
    ``L^-1`` are one block of the scene's factor storage: ``nonzero_count`` entries per
    articulation in ``arts`` order, from ``linv_offset`` on. The ``"free"`` group holds the free
    rigid bodies with a response, which keep their physical 6-DOF blocks and Steel's dense
    6 x 6 factors, so it has no key, no plan and no factor entries. Groups compare by identity.
    """

    kind: Literal["free", "factor"]
    """``"factor"`` for the articulations of one topology, ``"free"`` for the free rigid bodies."""
    key: tuple[tuple[int, ...], tuple[int, ...]] | None
    """The topology key of a factor group's articulations, ``None`` for the free group."""
    size: int
    """Response DOFs of each member: the plan's ``dof_count``, or 6 for the free group."""
    plan: _SparseMassMatrixPlan | None
    """The sparse factor plan of the group's topology, ``None`` for the free group."""
    arts: tuple[int, ...]
    """The group's articulations in increasing order; a member's position is its slot in the group."""
    nonzero_count: int
    """Factor entries per member, the plan's ``nonzero_count`` (0 for the free group)."""
    linv_offset: int
    """The first factor entry of the group's block: the entries of the groups before it."""
    mask_words: int
    """``uint64`` words of a support mask over ``size`` coordinates: the plan's ``ceil(size / 64)``, or
    more under the test-only ``scene_mask_words``; 1 for the free group."""


def topology_key(joint_parent: np.ndarray, joint_dof_count: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The topology key of an articulation: its local joint parents and DOFs per joint, as tuples of ints.

    ``joint_parent`` holds each tree joint's parent as a local joint index, ``-1`` for a root, and
    ``joint_dof_count`` each joint's DOFs: exactly the input of :meth:`_SparseMassMatrixPlan.build`,
    which fixes the factor's sparsity, masks and permutation. Articulations with equal keys share
    one plan. The key is hashable and compares by value.
    """
    return tuple(np.asarray(joint_parent).tolist()), tuple(np.asarray(joint_dof_count).tolist())


def plan_scene_groups(
    keys: Sequence[tuple | None], free: Sequence[bool], *, split: int = 1, mask_words: int = 1
) -> tuple[SceneGroup, ...]:
    """Group the scene's articulations by response representation (:class:`SceneGroup`).

    ``keys[art]`` is articulation ``art``'s topology key (:func:`topology_key`), or ``None`` for an
    articulation without a factor (one without response DOFs, or a free rigid body); ``free[art]``
    marks the free rigid bodies with a response, whose keys are not read. The factor groups come
    in order of their first articulation, one per distinct key, each with the plan its key
    builds; then one free group when any ``free[art]``. Group ``g``'s factor block starts at
    ``linv_offset_g``, the sum of ``len(arts_h) * nonzero_count_h`` over the groups ``h < g``.

    Each key's plan has support masks of ``ceil(DOFs / 64)`` ``uint64`` words, the group's
    ``mask_words``, so a key may have more than 64 DOFs. ``mask_words`` (test only: the solver's
    kernel override ``scene_mask_words``) is the least number of words: a larger value widens
    every factor group's masks, whose kernels must then give the rows of the narrower ones bit for
    bit. ``mask_words=1`` is the plan above.

    ``split`` (test only: the solver's kernel override ``scene_group_split``) splits the
    articulations of each key by ``art % split`` into up to ``split`` factor groups of that key,
    still in order of their first articulation, which share the key's plan; a residue without
    articulations makes no group. ``split=1`` is the plan above.

    Raises:
        ValueError: For a key :meth:`_SparseMassMatrixPlan.build` cannot plan, when ``keys`` and
            ``free`` differ in length, or for ``split < 1`` or ``mask_words < 1``.
    """
    if len(keys) != len(free):
        raise ValueError(f"plan_scene_groups got {len(keys)} keys for {len(free)} free flags")
    if split < 1:
        raise ValueError(f"plan_scene_groups needs a split of at least 1, got {split}")
    if mask_words < 1:
        raise ValueError(f"plan_scene_groups needs masks of at least one word, got {mask_words}")
    members: dict[tuple, list[int]] = {}
    free_arts: list[int] = []
    for art, (key, is_free) in enumerate(zip(keys, free, strict=True)):
        if is_free:
            free_arts.append(art)
        elif key is not None:
            members.setdefault((key, art % split), []).append(art)
    groups = []
    plans: dict[tuple, _SparseMassMatrixPlan] = {}
    offset = 0
    # Dictionaries keep insertion order: the factor groups come in order of their first articulation.
    for (key, _residue), arts in members.items():
        plan = plans.get(key)
        if plan is None:
            # The key's DOFs fix its masks' words; a key without DOFs is the plan's error to raise.
            words = max(_mask_words(sum(key[1])), mask_words)
            plan = plans[key] = _SparseMassMatrixPlan.build(
                *(np.asarray(part, dtype=np.int32) for part in key), mask_words=words
            )
        size = plan.dof_count
        groups.append(
            SceneGroup(
                kind="factor",
                key=key,
                size=size,
                plan=plan,
                arts=tuple(arts),
                nonzero_count=plan.nonzero_count,
                linv_offset=offset,
                mask_words=plan.mask_words,
            )
        )
        offset += len(arts) * plan.nonzero_count
    if free_arts:
        groups.append(
            SceneGroup(
                kind="free",
                key=None,
                size=6,
                plan=None,
                arts=tuple(free_arts),
                nonzero_count=0,
                linv_offset=offset,
                mask_words=_mask_words(6),
            )
        )
    return tuple(groups)


@dataclass(frozen=True)
class ScenePlan:
    """The row plan of scene mode, fixed at construction."""

    support: int
    """Entries per row (``S``): the most response coordinates one row touches."""
    row_max: int
    """Row capacity of the scene (``R_max``): Steel's per-world budgets summed over the model's worlds.

    ``articulation_constraint_max``, plus ``free_body_constraint_max`` when the model has free bodies,
    times the model's worlds (``scene_row_capacity``).
    """
    coordinate_count: int
    """Response coordinates of the scene (``C``), the length of the velocity delta."""
    free_body_count: int
    """Free rigid bodies with a response, each a 6-entry block of the velocity delta."""
    groups: tuple[SceneGroup, ...]
    """The scene groups (:func:`plan_scene_groups`): the factor groups, then the free group if any."""
    mask_words: int
    """``uint64`` words of the widest support mask, the maximum of the groups' ``mask_words``."""

    @property
    def articulated_size(self) -> int | None:
        """DOFs of the single factor group, or ``None`` for a scene of free bodies only.

        Raises:
            ValueError: When the plan has more than one factor group, and so no single articulated size.
        """
        sizes = [group.size for group in self.groups if group.kind == "factor"]
        if len(sizes) > 1:
            raise ValueError(f"the scene plan has {len(sizes)} factor groups and no single articulated size")
        return sizes[0] if sizes else None


def _check_response_shared_memory(dof_max: int, lanes: int, device) -> None:
    """Refuse a scene whose largest group overflows the group-table response kernel's static shared memory.

    The kernel's static shared memory, its scratch and Warp's per-thread reservation
    (:func:`~newton._src.solvers.steel.scene_rows.scene_contact_response_shared_bytes`), grows
    with the largest group's DOFs, ``dof_max``, and a CUDA block holds at most 48 KiB of it (less on
    a device that allows less), so a scene past that would fail when the kernel compiles, at its
    first step or inside a graph capture. On the CPU the scratch is per thread and nothing is
    checked.

    Raises:
        NotImplementedError: Naming the DOFs, the bytes needed and allowed, and the most DOFs that fit.
    """
    device = wp.get_device(device)
    if not device.is_cuda:
        return
    shared_limit = static_shared_limit(device)
    needed = scene_contact_response_shared_bytes(dof_max, lanes_per_contact=lanes)
    if needed > shared_limit:
        fits = scene_contact_response_dof_limit(shared_limit, lanes_per_contact=lanes)
        raise NotImplementedError(
            f'solve_mode="scene" cannot build contact rows for a {dof_max}-DOF topology: the contact response '
            f"kernel needs {needed} B of static shared memory per block (its scratch and Warp's per-thread "
            f"reservation), the device allows {shared_limit} B, enough for topologies of at most {fits} DOFs"
        )


@dataclass(eq=False)
class SceneGroupArrays:
    """The device storage and kernels of one factor group (:class:`SceneGroup`), fixed at construction.

    Row ``i`` of every per-member array is the group's ``i``-th articulation, ``group_to_art[i]``,
    which is the launch row of Steel's group kernels: each launch runs them unchanged on the group.
    """

    group: SceneGroup
    """The factor group."""
    group_to_art: wp.array
    """``int32`` ``[members]``: the group's articulations, ``group.arts``."""
    indices: _SparseMassMatrixIndices
    """The group plan's index arrays on the device (:meth:`_SparseMassMatrixPlan.to_device`)."""
    armature: wp.array2d
    """``float32`` ``[members, size]``: each member's effective joint armature (zero past its DOFs)."""
    status: wp.array
    """``int32`` ``[members]``: the factor's status, ``1`` after a nonpositive or nonfinite pivot."""
    scratch: wp.array2d
    """``float32`` ``[members, size]``: the unconstrained solve's intermediate ``L^-1 P tau``."""
    factor_kernel: wp.Kernel
    """The sparse factor kernel of the group's plan, ``warps_per_block`` warps per block."""
    internal_kernels: dict[int, wp.Kernel]
    """The row builders of the internal-unit families the group launches, by family in Steel's slot
    order, each for the group's ``mask_words``
    (:func:`~newton._src.solvers.steel.scene_rows.get_sparse_joint_dof_rows_kernel`):
    ``FAMILY_DRIVE`` when ``drive_mode="rows"`` and a member has a DOF of a drive's joint type,
    ``FAMILY_MIMIC`` when a member owns a mimic row,
    ``FAMILY_LIMIT`` (Steel's own joint-limit builder for one word) when joint limits are on and the
    group's size has limited joints, ``FAMILY_JOINT_FRICTION`` when joint friction is on and a member
    has a DOF that supports it (``scene_group_families``). The drive family's builder is its
    fused velocity clamp's build when the clamp is in effect (``SolverSteel.velocity_limits_fused``)."""
    warps_per_block: int
    """Warps per block of the factor: as many as its static shared memory allows, at most four."""
    Linv: wp.array2d
    """``float32`` ``[members, nonzero_count]``: the members' factors ``L^-1``, a view of the group's
    block of :attr:`SceneSolve.Linv`."""
    late_kernel: wp.Kernel | None = None
    """The velocity-limit family's row builder for the group's ``mask_words``, the builder of its members' late
    units (:func:`~newton._src.solvers.steel.scene_rows.get_sparse_joint_velocity_limit_rows_kernel`), when
    joint velocity limits are on and a member can bear their rows (``scene_group_late``, which picks
    the free-body builder for the free group, :attr:`SceneSolve.free_late_kernel`); ``None`` otherwise."""


def scene_response_table(
    groups: Sequence[SceneGroup], articulation_count: int, linv: wp.array, device, *, articulated: np.ndarray
) -> tuple[wp.array, wp.array, tuple[wp.array, ...]]:
    """The group table of the generic contact response kernel over the factor groups of ``groups``.

    Factor group ``g`` is entry ``g`` of ``groups`` (:func:`plan_scene_groups` puts the factor
    groups first). ``articulated`` (``bool``, ``[articulation_count]``) marks the responding
    articulations that are not free rigid bodies: each must be in a factor group, since the kernel
    takes an endpoint that is not a free body through its group. Returns ``(art_group, art_slot,
    table)``, ``int32`` arrays unless noted:

    - ``art_group`` [``articulation_count``]: the articulation's factor group, ``-1`` for a free
      body or an articulation without a response; ``art_slot``: its position in the group's
      ``arts``, ``-1`` likewise.
    - ``table``: ``(group_dof_count, group_nonzero_count, group_linv_offset, group_permutation_base,
      group_row_offset_base, permutation, row_offsets, Linv)``. The first five have one entry per
      factor group: its ``size``, ``nonzero_count`` and ``linv_offset``, and the first entries of
      its plan's ``permutation`` and ``row_offsets`` in the concatenated ``permutation``
      [sum of sizes] and ``row_offsets`` [sum of sizes + 1]. ``Linv`` is ``linv``, the flat
      ``float32`` factor storage that holds every group's block.

    Raises:
        ValueError: When ``linv`` does not hold exactly the factor groups' entries, or when an
            ``articulated`` articulation has no factor group.
    """
    factor = [group for group in groups if group.kind == "factor"]
    if any(group.kind != "factor" for group in groups[: len(factor)]):
        raise ValueError("the factor groups must come before the free group")
    entries = sum(len(group.arts) * group.nonzero_count for group in factor)
    if linv.ndim != 1 or linv.size != entries:
        raise ValueError(f"Linv must be flat with the factor groups' {entries} entries, got shape {linv.shape}")
    art_group = np.full(articulation_count, -1, dtype=np.int32)
    art_slot = np.full(articulation_count, -1, dtype=np.int32)
    for index, group in enumerate(factor):
        arts = np.asarray(group.arts, dtype=np.int32)
        art_group[arts] = index
        art_slot[arts] = np.arange(len(arts), dtype=np.int32)
    ungrouped = np.flatnonzero(np.asarray(articulated, dtype=bool) & (art_group < 0))
    if ungrouped.size:
        raise ValueError(
            f"responding articulations {ungrouped[:8].tolist()} are not free bodies and have no factor group"
        )
    sizes = np.asarray([group.size for group in factor], dtype=np.int32)
    permutation = [group.plan.permutation for group in factor]
    row_offsets = [group.plan.row_offsets for group in factor]
    columns = {
        "group_dof_count": sizes,
        "group_nonzero_count": [group.nonzero_count for group in factor],
        "group_linv_offset": [group.linv_offset for group in factor],
        "group_permutation_base": np.cumsum(sizes) - sizes,
        "group_row_offset_base": np.cumsum(sizes + 1) - (sizes + 1),
        "permutation": np.concatenate(permutation) if factor else [],
        "row_offsets": np.concatenate(row_offsets) if factor else [],
    }
    table = tuple(
        wp.array(np.asarray(value, dtype=np.int32), dtype=wp.int32, device=device) for value in columns.values()
    )
    return (
        wp.array(art_group, dtype=wp.int32, device=device),
        wp.array(art_slot, dtype=wp.int32, device=device),
        (*table, linv),
    )


@wp.kernel
def scene_row_sequence(
    side_count: int,
    unit_max: int,
    color_total_slot: int,
    unit_count: wp.array[int],
    unit_order: wp.array[int],
    unit_side_a: wp.array[int],
    unit_rows: wp.array[int],
    side_start: wp.array[int],
    one_sided_start: wp.array[int],
    color_offsets: wp.array[int],
    order_by_color: wp.array[int],
    # outputs
    unit_sequence: wp.array[int],
    seq_rows: wp.array[int],
):
    """List the units in row order with their row counts (grid: ``unit_max + 1``).

    Thread ``t`` places the unit at sorted position ``t`` when it is one-sided: entry
    ``one_sided_start[s] + t - side_start[s]`` of side ``s``'s run, so the one-sided units come
    side-major in sorted order. It also places entry ``t`` of the order by color after every
    one-sided unit, at ``one_sided_start[side_count] + t``. ``unit_sequence`` receives the sorted
    position and ``seq_rows`` the unit's rows. Entries from ``unit_count`` on get ``-1`` and ``0``
    rows, so the exclusive scan of ``seq_rows`` (``unit_max + 1`` entries) is fully defined and its
    entry ``unit_count`` is the rows requested.
    """
    t = wp.tid()
    count = unit_count[0]
    if t < count:
        slot = unit_order[t]
        side = unit_side_a[slot]
        local = t - side_start[side]
        first = one_sided_start[side]
        if local < one_sided_start[side + 1] - first:
            unit_sequence[first + local] = t
            seq_rows[first + local] = unit_rows[slot]
    if t < color_offsets[color_total_slot]:
        position = order_by_color[t]
        entry = one_sided_start[side_count] + t
        unit_sequence[entry] = position
        seq_rows[entry] = unit_rows[unit_order[position]]
    if t >= count:
        seq_rows[t] = 0
        if t < unit_max:
            unit_sequence[t] = -1


@wp.kernel
def scene_late_sequence(
    side_count: int,
    color_total_slot: int,
    unit_order: wp.array[int],
    unit_rows: wp.array[int],
    side_start: wp.array[int],
    unit_one_sided_count: wp.array[int],
    late_count: wp.array[int],
    one_sided_start: wp.array[int],
    color_offsets: wp.array[int],
    late_start: wp.array[int],
    # outputs
    unit_sequence: wp.array[int],
    seq_rows: wp.array[int],
):
    """List the late units after the tail, side-major (grid: ``side_count``).

    A late unit is the last of its side's one-sided units in the units' sort (``unit_one_sided_count``, late units
    included), and the phase-A one-sided units (``one_sided_start``, late units excluded) leave it to this kernel:
    side ``s``'s late unit, at sorted position ``side_start[s] + unit_one_sided_count[s] - 1``, takes entry
    ``one_sided_start[side_count] + color_offsets[color_total_slot] + late_start[s]`` of the row order, after
    every one-sided and two-sided unit, the late units of the sides before it first (``late_start``, the exclusive
    scan of ``late_count``).
    """
    s = wp.tid()
    if late_count[s] == 0:
        return
    position = side_start[s] + unit_one_sided_count[s] - 1
    entry = one_sided_start[side_count] + color_offsets[color_total_slot] + late_start[s]
    unit_sequence[entry] = position
    seq_rows[entry] = unit_rows[unit_order[position]]


@wp.kernel
def scene_phase_a_counts(
    unit_one_sided_count: wp.array[int],
    late_count: wp.array[int],
    # outputs
    one_sided_count: wp.array[int],
):
    """Count each side's phase-A one-sided units: its one-sided units less its late unit (grid: ``side_count + 1``)."""
    s = wp.tid()
    one_sided_count[s] = unit_one_sided_count[s] - late_count[s]


@wp.kernel
def scene_assign_rows(
    row_max: int,
    unit_count: wp.array[int],
    unit_order: wp.array[int],
    unit_contact: wp.array[int],
    unit_side_a: wp.array[int],
    unit_sequence: wp.array[int],
    seq_row_start: wp.array[int],
    # outputs
    unit_row_start: wp.array[int],
    contact_slot: wp.array[int],
    contact_path: wp.array[int],
    limit_row_cursor: wp.array[int],
    late_row_cursor: wp.array[int],
    side_dropped: wp.array[int],
    rows_requested: wp.array[int],
    constraint_count: wp.array[int],
):
    """Give each unit the rows the scan placed it at, or drop it when they pass ``row_max`` (grid: ``unit_max``).

    Thread ``j`` takes entry ``j`` of the row order, rows ``[seq_row_start[j], seq_row_start[j + 1])``.
    A unit that fits records its first row in ``unit_row_start`` (per slot) and in its contact's
    ``contact_slot`` or, for an internal unit, its articulation's row cursor (``limit_row_cursor``,
    which every family's WRITE pass advances), for a late unit its articulation's late cursor
    (``late_row_cursor``, which the velocity-limit family's WRITE pass advances). One that does not gets
    ``unit_row_start == -1``, its contact ``contact_slot == contact_path == -1`` (an internal or late unit
    keeps its cursor at ``row_max``, set before the launch, so no WRITE pass writes a row), and
    its rows are counted in ``side_dropped[side_a]``. The kept units are a prefix of the order, so
    the row count is the end of the last kept unit, which only that unit's thread writes (thread 0
    when none is kept); thread 0 also writes the rows requested.
    """
    j = wp.tid()
    count = unit_count[0]
    if j == 0:
        rows_requested[0] = seq_row_start[count]
        if count == 0 or seq_row_start[1] > row_max:
            constraint_count[0] = 0
    if j >= count:
        return
    start = seq_row_start[j]
    end = seq_row_start[j + 1]
    slot = unit_order[unit_sequence[j]]
    c = unit_contact[slot]
    if end <= row_max:
        unit_row_start[slot] = start
        if c >= 0:
            contact_slot[c] = start
        elif c == UNIT_CONTACT_LATE:
            late_row_cursor[unit_side_a[slot]] = start
        else:
            limit_row_cursor[unit_side_a[slot]] = start
        if j + 1 == count:
            constraint_count[0] = end
        elif seq_row_start[j + 2] > row_max:
            constraint_count[0] = end
    else:
        unit_row_start[slot] = -1
        if c >= 0:
            contact_slot[c] = -1
            contact_path[c] = -1
        wp.atomic_add(side_dropped, unit_side_a[slot], end - start)


@wp.kernel
def scene_side_rows(
    one_sided_start: wp.array[int],
    seq_row_start: wp.array[int],
    constraint_count: wp.array[int],
    # outputs
    side_row_begin: wp.array[int],
    side_row_end: wp.array[int],
):
    """Write each side's contiguous one-sided rows ``[side_row_begin, side_row_end)`` (grid: sides).

    Both ends are clamped to the row count, so a side whose units were dropped lists only its kept
    rows.
    """
    s = wp.tid()
    rows = constraint_count[0]
    side_row_begin[s] = wp.min(seq_row_start[one_sided_start[s]], rows)
    side_row_end[s] = wp.min(seq_row_start[one_sided_start[s + 1]], rows)


@wp.kernel
def scene_side_late_rows(
    side_count: int,
    color_total_slot: int,
    one_sided_start: wp.array[int],
    color_offsets: wp.array[int],
    late_start: wp.array[int],
    seq_row_start: wp.array[int],
    constraint_count: wp.array[int],
    # outputs
    side_late_begin: wp.array[int],
    side_late_end: wp.array[int],
):
    """Write each side's late rows ``[side_late_begin, side_late_end)``, its late unit's (grid: sides).

    The late units follow every one-sided and two-sided unit in the row order, side-major (:func:`scene_late_sequence`).
    Both ends are clamped to the row count, so a side without a late unit, or whose late unit was dropped past
    ``row_max``, lists none.
    """
    s = wp.tid()
    rows = constraint_count[0]
    first = one_sided_start[side_count] + color_offsets[color_total_slot]
    side_late_begin[s] = wp.min(seq_row_start[first + late_start[s]], rows)
    side_late_end[s] = wp.min(seq_row_start[first + late_start[s + 1]], rows)


@wp.kernel
def prepare_scene_rows(
    constraint_count: wp.array[int],
    row_type: wp.array2d[int],
    phi: wp.array2d[float],
    target_velocity: wp.array2d[float],
    row_rebound_velocity: wp.array2d[float],
    row_incident: wp.array2d[float],
    row_dof: wp.array3d[int],
    free_dof_mask: wp.array2d[int],
    pgs_beta: float,
    friction_anchor_beta: float,
    contact_speculative_scale: float,
    contact_w: float,
    pgs_cfm: float,
    dt: float,
    # outputs
    rhs: wp.array2d[float],
    diag: wp.array2d[float],
    impulses: wp.array2d[float],
    row_free_prefix: wp.array[int],
    row_w: wp.array2d[float],
):
    """Finish every row in parallel: diagonal, bias, impulse, free prefix and weight (grid: ``row_max``).

    Replaces Steel's serial per-world ``finalize_world_diag_cfm``, ``compute_world_contact_bias``
    and ``prepare_world_impulses`` with their per-row arithmetic: ``cfm`` is added to the
    diagonal of every row except drive rows, ``rhs`` receives the bias alone (the incident
    velocity stays in ``_sparse_row_incident``) and the impulse starts at zero.
    ``row_free_prefix`` receives the row's leading entries in physical free-body coordinates: 0,
    6 or 12 (one or two free endpoints, which the response kernel puts first).

    A contact row with a positive ``row_rebound_velocity`` whose rebound fires
    (:func:`~newton._src.solvers.steel.kernels.contact_restitution_fires` on ``J v_hat - target``,
    with ``J v_hat`` the row's ``row_incident``) gets its rebound target instead of the bias, as
    Steel's ``apply_world_contact_restitution`` gives it: ``rhs = -target - rebound``. Every other
    row's ``rhs`` is the bias, unchanged. ``row_rebound_velocity`` is the separating speed
    ``-e * u_start`` that ``prepare_world_contact_rows`` writes every step from the contact's
    step-start closing speed (``kernels.contact_rebound_velocity``); it is zero for a contact that
    does not close faster than the restitution threshold, and for every contact with
    ``enable_restitution=False``.

    With contact regularization (``contact_w = 1 / (1 + g) < 1``) ``row_w`` receives each row's
    weight, as Steel's ``compute_world_contact_bias`` writes its dense rows' and
    ``compute_mf_effective_mass_and_rhs`` its free-body rows': ``contact_w`` for a penetrating
    contact (``phi <= 0``), 1 for every other row and for a contact whose rebound fires (Steel's
    ``apply_world_contact_restitution`` keeps an impact's rebound exact). Without it nothing is
    written, and ``row_w`` may be empty.
    """
    i = wp.tid()
    if i >= constraint_count[0]:
        return
    kind = row_type[0, i]
    if kind != PGS_CONSTRAINT_TYPE_JOINT_TARGET:
        diag[0, i] += pgs_cfm

    # compute_world_contact_bias, one row.
    inv_dt = 1.0 / dt
    row_phi = phi[0, i]
    target = target_velocity[0, i]
    bias = -target
    weight = float(1.0)
    if kind == PGS_CONSTRAINT_TYPE_CONTACT:
        if row_phi <= 0.0:
            # A free-body contact's Baumgarte term is bounded after this kernel where the depenetration cap binds
            # (cap_scene_depenetration).
            bias += pgs_beta * row_phi * inv_dt
            weight = contact_w
        else:
            bias += contact_speculative_scale * row_phi * inv_dt
        # apply_world_contact_restitution, one row: an impact replaces the bias by its rebound target.
        rebound_velocity = row_rebound_velocity[0, i]
        if rebound_velocity > 0.0:
            relative_incident = row_incident[0, i] - target
            if contact_restitution_fires(row_phi, relative_incident, dt):
                bias = -target - rebound_velocity
                # An impact is impulsive, not a spring: its rebound stays exact (weight 1).
                weight = 1.0
    elif kind == PGS_CONSTRAINT_TYPE_FRICTION:
        bias += friction_anchor_beta * row_phi * inv_dt
    elif kind == PGS_CONSTRAINT_TYPE_JOINT_LIMIT:
        if row_phi < 0.0:
            bias += pgs_beta * row_phi * inv_dt
        else:
            bias += row_phi * inv_dt
    elif kind == PGS_CONSTRAINT_TYPE_MIMIC or kind == PGS_CONSTRAINT_TYPE_CONNECT:
        bias += pgs_beta * row_phi * inv_dt
    _finish_scene_row(i, bias, weight, contact_w, row_dof, free_dof_mask, rhs, row_w, impulses, row_free_prefix)


@wp.kernel
def cap_scene_depenetration(
    contact_count: wp.array[int],
    contact_slot: wp.array[int],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    shape_body: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    is_free_rigid: wp.array[int],
    max_depenetration_velocity: wp.array[float],
    row_type: wp.array2d[int],
    phi: wp.array2d[float],
    target_velocity: wp.array2d[float],
    row_rebound_velocity: wp.array2d[float],
    row_incident: wp.array2d[float],
    pgs_beta: float,
    dt: float,
    # outputs
    rhs: wp.array2d[float],
):
    """Bound a penetrating free-body contact's Baumgarte bias by its bodies' depenetration velocity (grid: contacts).

    Steel's depenetration cap (``kernels.compute_mf_effective_mass_and_rhs``), on the contacts Steel solves on its
    free-body path: each side without response DOFs or a free rigid body (:func:`_scene_contact_on_free_path`,
    ``_allocate_world_contact_slot``'s ``is_mf``), whose rows carry the responding sides' bodies ``body_a`` and
    ``body_b`` (``_build_mf_contact_row``, ``-1`` for a side without response DOFs). For the kept normal row of such a
    contact (``contact_slot``) with ``phi < 0``, Steel's ``max_depen`` is formed in Steel's order: ``1e20``, replaced by
    ``body_a``'s ``max_depenetration_velocity`` as it is, then lowered to ``body_b``'s when that is finite, positive and
    smaller; a finite positive ``max_depen`` bounds the Baumgarte term, ``max(bias, -max_depen)``. A contact whose
    rebound fires keeps the rebound target that replaces its bias after the cap, as in
    ``compute_mf_effective_mass_and_rhs``. The kernel writes a row only where the cap binds, ``bias < -max_depen`` with
    ``bias`` the Baumgarte term of :func:`prepare_scene_rows`' contact branch (``pgs_beta * phi / dt`` of a penetrating
    contact, recomputed here in the same order): the row's ``rhs`` becomes ``-target - max_depen``, Steel's value bit
    for bit. A row where the cap does not bind keeps the ``rhs`` :func:`prepare_scene_rows` wrote, and so does a contact
    with an articulated side (Steel's dense path, which has no cap), a speculative contact and a friction row. Contact
    compliance, which runs after the cap, replaces a compliant row's ``rhs`` whether or not the cap bound
    (:func:`~newton._src.solvers.steel.scene_compliance.prepare_scene_compliance`).
    """
    c = wp.tid()
    if c >= wp.min(contact_count[0], contact_slot.shape[0]):
        return
    row = contact_slot[c]
    if row < 0:
        return
    art_a = contact_art_a[c]
    art_b = contact_art_b[c]
    if not _scene_contact_on_free_path(art_a, art_b, articulation_response_dof_count, is_free_rigid):
        return
    a_has_dofs = art_a >= 0 and articulation_response_dof_count[art_a] > 0
    b_has_dofs = art_b >= 0 and articulation_response_dof_count[art_b] > 0
    if row_type[0, row] != PGS_CONSTRAINT_TYPE_CONTACT:
        return
    row_phi = phi[0, row]
    if row_phi >= 0.0:
        return
    ba = int(-1)
    bb = int(-1)
    if a_has_dofs:
        ba = shape_body[contact_shape0[c]]
    if b_has_dofs:
        bb = shape_body[contact_shape1[c]]
    # compute_mf_effective_mass_and_rhs, its order.
    max_depen = float(1.0e20)
    if ba >= 0:
        max_depen = max_depenetration_velocity[ba]
    if bb >= 0:
        max_depen_b = max_depenetration_velocity[bb]
        if max_depen_b > 0.0 and wp.isfinite(max_depen_b):
            if max_depen_b < max_depen:
                max_depen = max_depen_b
    if not (max_depen > 0.0 and wp.isfinite(max_depen)):
        return
    target = target_velocity[0, row]
    # An impact's rebound target replaces the bias after the cap, as prepare_scene_rows gave it.
    if row_rebound_velocity[0, row] > 0.0:
        if contact_restitution_fires(row_phi, row_incident[0, row] - target, dt):
            return
    # The Baumgarte term of prepare_scene_rows' contact branch (a penetrating contact), bounded where the cap binds
    # (compute_mf_effective_mass_and_rhs' max): -target + max(bias, -max_depen) is -target - max_depen then, and the rhs
    # prepare_scene_rows wrote otherwise, which the row keeps.
    inv_dt = 1.0 / dt
    bias = pgs_beta * row_phi * inv_dt
    if bias < -max_depen:
        rhs[0, row] = -target - max_depen


@wp.kernel
def prepare_scene_friction_modes(
    contact_count: wp.array[int],
    contact_slot: wp.array[int],
    contact_slots_needed: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    is_free_rigid: wp.array[int],
    articulated_mode: int,
    # outputs
    row_friction_mode: wp.array2d[int],
):
    """Flag the update each contact's tangent pair takes under a friction mode (grid: contacts; ``friction_mode``).

    Steel builds its contact update per row family (``solver_steel._get_pgs_solve_mf_gs_kernel``): its free-body
    contacts take the mode's free-body text, its dense contacts the dense bisection under ``"bisection"`` and
    ``"bisection_desaxce"`` (whose de Saxce bias the dense text leaves out) and the tangent pair's update under
    ``"coulomb_newton"``. The scene sweeps both kinds in one pass, so each contact with its tangent pair writes its
    update on its first tangent row (``contact_slot + 1``): :data:`SCENE_FRICTION_FREE` on Steel's free-body path
    (``scene_units._scene_contact_on_free_path``, the test of :func:`cap_scene_depenetration`), ``articulated_mode``
    (:data:`SCENE_FRICTION_DENSE`, or :data:`SCENE_FRICTION_FIXED_NORMAL` under ``"coulomb_newton"``) otherwise. A
    contact with its normal row alone (``contact_slots_needed`` 1, past ``contact_friction_gap_max``) keeps the normal
    update, as Steel's lone normal row has no tangent problem, and a contact without rows writes nothing.
    ``row_friction_mode`` is cleared before the launch.
    """
    c = wp.tid()
    if c >= wp.min(contact_count[0], contact_slot.shape[0]):
        return
    row = contact_slot[c]
    if row < 0 or contact_slots_needed[c] != 3:
        return
    mode = articulated_mode
    if _scene_contact_on_free_path(contact_art_a[c], contact_art_b[c], articulation_response_dof_count, is_free_rigid):
        mode = SCENE_FRICTION_FREE
    row_friction_mode[0, row + 1] = mode


@wp.kernel
def prepare_scene_velocity_rows(
    constraint_count: wp.array[int],
    row_type: wp.array2d[int],
    phi: wp.array2d[float],
    target_velocity: wp.array2d[float],
    row_rebound_velocity: wp.array2d[float],
    row_incident: wp.array2d[float],
    row_dof: wp.array3d[int],
    row_factor: wp.array3d[float],
    delta: wp.array2d[float],
    steel_sums: int,
    dt: float,
    # outputs
    rhs_unbiased: wp.array2d[float],
):
    """Write the velocity-only iterations' right-hand side of every row (grid: ``row_max``; zeros past the rows).

    Steel's ``kernels.compute_world_contact_velocity_bias`` term by term, on the scene's rows, after the position solve:
    ``rhs = -target``; a contact row with a positive ``row_rebound_velocity`` whose rebound fires
    (:func:`~newton._src.solvers.steel.kernels.contact_restitution_fires`, on ``J v_hat - target`` with ``J v_hat`` the
    row's ``row_incident``) keeps its rebound target, ``rhs = -target - rebound``; otherwise a contact row with
    ``phi > 0`` whose end gap after the position solve, ``phi + dt * (J v_pos - target)``, stays above
    ``_CONTACT_END_GAP_SLOP`` keeps its speculative allowance ``phi / dt``; a joint-limit row with ``phi >= 0`` keeps
    ``phi / dt``; every other row and term is dropped (the Baumgarte bias, a penetrating contact's or limit's, and the
    friction anchors'). Steel's free-body rows follow the same law (``compute_mf_velocity_rhs``), the scene's free rows
    this one, as :func:`prepare_scene_rows` gives them the dense law.

    ``J v_pos``, the row's velocity after the position solve (Steel's ``J v_out``), is formed in factor coordinates as
    ``J v_hat + Z delta``: the row's ``row_incident`` plus its factor values ``row_factor`` dotted with the position
    solve's ``delta``, summed as the one-thread row loop sums its ``dot`` (``scene_rows._scene_row_dot``, Steel's order
    when ``steel_sums``), so it is the ``J v`` a next visit of the row would form. Steel's free-body law divides ``phi``
    by ``dt`` where this one, as the dense law, multiplies it by ``1 / dt``, as the position solve's rows do. Like
    ``rhs`` in :func:`prepare_scene_rows` it excludes the incident velocity, which the sweeps add at load.
    """
    i = wp.tid()
    if i >= constraint_count[0]:
        rhs_unbiased[0, i] = 0.0
        return
    inv_dt = 1.0 / dt
    row_phi = phi[0, i]
    kind = row_type[0, i]
    target = target_velocity[0, i]
    rhs = -target
    if kind == PGS_CONSTRAINT_TYPE_CONTACT:
        rebound_velocity = row_rebound_velocity[0, i]
        relative_incident = float(0.0)
        bounce = False
        if rebound_velocity > 0.0:
            relative_incident = row_incident[0, i] - target
            bounce = contact_restitution_fires(row_phi, relative_incident, dt)
        if bounce:
            rhs -= rebound_velocity
        elif row_phi > 0.0:
            jv_position = row_incident[0, i] + _scene_row_dot(row_dof, row_factor, delta, i, steel_sums)
            end_gap = row_phi + dt * (jv_position - target)
            if end_gap > _CONTACT_END_GAP_SLOP:
                rhs += row_phi * inv_dt
    elif kind == PGS_CONSTRAINT_TYPE_JOINT_LIMIT:
        if row_phi >= 0.0:
            rhs += row_phi * inv_dt
    rhs_unbiased[0, i] = rhs


@wp.kernel
def scene_drive_descriptors(
    constraint_count: wp.array[int],
    row_type: wp.array2d[int],
    diag: wp.array2d[float],
    target_velocity: wp.array2d[float],
    drive_stiffness: wp.array2d[float],
    drive_damping: wp.array2d[float],
    drive_geom_error: wp.array2d[float],
    drive_max_force: wp.array2d[float],
    dt: float,
    # outputs
    drive_target_vel_bias: wp.array2d[float],
    drive_vel_multiplier: wp.array2d[float],
    drive_impulse_multiplier: wp.array2d[float],
    drive_max_impulse: wp.array2d[float],
):
    """Write each drive row's force-drive descriptors from its final diagonal (grid: ``row_max``).

    Steel's ``kernels.compute_physx_pgs_drive_desc``, one row per thread instead of one world per thread, with its
    arithmetic in its order: with stiffness ``ke``, damping ``kd``, the row's unit response ``r`` (its diagonal,
    which carries no ``cfm``: :func:`prepare_scene_rows`),
    ``a = dt (dt ke + kd)``, ``b = dt (kd v_target)`` and ``x = 1 / (1 + a r)`` (``0`` when ``r <= 0``), a
    drive row gets ``target_vel_bias = x b + ke x dt (q_target - q)``, ``vel_multiplier = -x a``,
    ``impulse_multiplier = 1 - x`` and ``max_impulse = f dt`` for a positive finite effort limit ``f``,
    ``1e20`` otherwise. Every other row below ``constraint_count`` gets zeros, as in Steel. The drive
    parameters are the ones the drive family's builder wrote
    (:func:`~newton._src.solvers.steel.scene_rows.get_sparse_drive_rows_kernel`).
    """
    i = wp.tid()
    if i >= constraint_count[0]:
        return
    if row_type[0, i] != PGS_CONSTRAINT_TYPE_JOINT_TARGET:
        drive_target_vel_bias[0, i] = 0.0
        drive_vel_multiplier[0, i] = 0.0
        drive_impulse_multiplier[0, i] = 0.0
        drive_max_impulse[0, i] = 0.0
        return

    stiffness = drive_stiffness[0, i]
    damping = drive_damping[0, i]
    unit_response = diag[0, i]

    a = dt * (dt * stiffness + damping)
    b = dt * (damping * target_velocity[0, i])
    x = float(0.0)
    if unit_response > 0.0:
        x = 1.0 / (1.0 + a * unit_response)

    drive_target_vel_bias[0, i] = x * b + stiffness * x * dt * drive_geom_error[0, i]
    drive_vel_multiplier[0, i] = -x * a
    drive_impulse_multiplier[0, i] = 1.0 - x

    max_force = drive_max_force[0, i]
    max_impulse = float(1.0e20)
    if max_force > 0.0 and wp.isfinite(max_force):
        max_impulse = max_force * dt
    drive_max_impulse[0, i] = max_impulse


@wp.kernel
def scene_row_status(
    row_max: int,
    unit_count: wp.array[int],
    unit_overflow: wp.array[int],
    unit_order: wp.array[int],
    unit_side_a: wp.array[int],
    unit_side_b: wp.array[int],
    unit_sequence: wp.array[int],
    seq_row_start: wp.array[int],
    rows_requested: wp.array[int],
    constraint_count: wp.array[int],
    contact_count: wp.array[int],
    reduction_overflow: wp.array[int],
    contact_capacity: int,
    art_model_world: wp.array[int],
    warn: int,
    # in/out
    warning_emitted: wp.array[int],
    # outputs
    overflow: wp.array[int],
    constraint_overflow: wp.array[wp.bool],
):
    """Latch the model worlds that lost rows and report the step's losses (grid: ``max(unit_max, entries)``).

    ``constraint_overflow`` has one entry per model world and a last one for global
    (world ``-1``) bodies. Every entry is latched when units passed ``scene_unit_max``, the
    contact buffer overflowed or contact reduction dropped candidates. A unit dropped for rows
    latches the model world of each of its sides, so every side with ``side_dropped > 0`` (and the
    other side of a dropped two-sided unit) is reported. Thread 0 writes ``overflow[0]``, ``1``
    when this step lost rows, units or contacts, and with ``warn`` prints one warning for the
    solver's lifetime.
    """
    t = wp.tid()
    entries = constraint_overflow.shape[0]
    contacts_lost = contact_count[0] > contact_capacity or reduction_overflow[0] != 0
    units_lost = unit_overflow[0] > 0
    if t < entries:
        if contacts_lost or units_lost:
            constraint_overflow[t] = True
    if t < unit_count[0]:
        if seq_row_start[t + 1] > row_max:
            slot = unit_order[unit_sequence[t]]
            _latch_side_world(unit_side_a[slot], art_model_world, constraint_overflow)
            side_b = unit_side_b[slot]
            if side_b >= 0:
                _latch_side_world(side_b, art_model_world, constraint_overflow)
    if t == 0:
        rows_lost = rows_requested[0] > constraint_count[0]
        lost = int(0)
        if rows_lost or units_lost or contacts_lost:
            lost = 1
        overflow[0] = lost
        if warn != 0 and (rows_lost or units_lost):
            if wp.atomic_exch(warning_emitted, 0, 1) == 0:
                wp.printf(
                    "Warning: Steel scene rows overflow: %d units requested %d rows, limit %d; kept %d rows. "
                    "%d further units passed scene_unit_max. Units are dropped in row order, the last first: the "
                    "late units (velocity limits) of the last sides, then the tail and the last colors, then the "
                    "one-sided units of the last sides. Increase articulation_constraint_max or "
                    "free_body_constraint_max (rows per world) or scene_unit_max.\n",
                    unit_count[0],
                    rows_requested[0],
                    row_max,
                    constraint_count[0],
                    unit_overflow[0],
                )


# --- The internal unit (an articulation's row families) ------------------------------------------

SCENE_ROW_FAMILIES = (
    FAMILY_DRIVE,
    FAMILY_MIMIC,
    FAMILY_CONNECT,
    FAMILY_LIMIT,
    FAMILY_JOINT_FRICTION,
    FAMILY_VELOCITY_LIMIT,
)
"""The per-articulation row families (``FAMILY_COUNT`` of them), in Steel's slot order.

Steel lays out a world's dense rows as ``[drives][mimic][connect][joint limits][joint friction][joint velocity
limits]`` (``_stage4_build_rows``); an internal unit holds one articulation's rows of the first five families in that
order, and its late unit the velocity-limit family's, which Steel sweeps last. The ids are defined beside their
builders, :func:`~newton._src.solvers.steel.scene_rows.get_sparse_joint_dof_rows_kernel`.
"""


def allocate_scene_internal_rows(articulation_count: int, families: Sequence[int], device) -> tuple:
    """Allocate the internal unit's row counts for ``families`` in use (a sorted sequence of internal family ids).

    Returns ``(family_rows, internal_rows)``: ``family_rows`` (``int32``, ``[FAMILY_COUNT,
    max(articulation_count, 1)]``, zero) receives each family's COUNT pass, one row per family (the velocity-limit
    family's row holds the late units' rows, which are not internal), and
    ``internal_rows`` (``int32``, ``[articulation_count]``) their sum, the internal unit's rows. With
    more than one family in use ``internal_rows`` is an array of its own, which
    :func:`sum_scene_family_rows` writes; with one it views that family's row, which is then the sum,
    and with none the joint limits' row, which stays zero. Without articulations and with at most one
    family in use, ``internal_rows`` is that one-element view of ``family_rows`` (the one column kept),
    which no launch reads.
    """
    # A model without articulations keeps one column, so that its family rows are views (no launch reads them).
    family_rows = wp.zeros((FAMILY_COUNT, max(articulation_count, 1)), dtype=wp.int32, device=device)
    if len(families) > 1:
        internal_rows = wp.zeros(articulation_count, dtype=wp.int32, device=device)
    else:
        internal_rows = family_rows[families[0] if families else FAMILY_LIMIT]
    return family_rows, internal_rows


@wp.kernel
def sum_scene_family_rows(
    first_family: int,
    last_family: int,
    family_rows: wp.array2d[int],
    # outputs
    internal_rows: wp.array[int],
):
    """Write each articulation's internal-unit rows, the sum of its families' (grid: articulations).

    ``family_rows[f, art]`` holds family ``f``'s COUNT pass for articulation ``art``; the families
    ``first_family`` to ``last_family`` (inclusive) are summed in family order.
    """
    art = wp.tid()
    rows = int(0)
    for family in range(first_family, last_family + 1):
        rows += family_rows[family, art]
    internal_rows[art] = rows


# --- Row watermarks (row_watermark) -------------------------------------------------------------

SCENE_WATERMARK_FAMILIES = 2
"""Steel's row families the scene's rows count in: 0 the articulation (dense) rows, 1 the free-body rows."""

SCENE_WORLD_ROWS_KEPT = 0
SCENE_WORLD_ROWS_REQUESTED = 1
SCENE_WORLD_ROWS_DROPPED_CONTACT = 2
"""Counts of ``SceneSolve.world_rows[family, count, world]``: kept, requested and dropped contact rows."""


@wp.kernel
def count_scene_world_rows(
    row_max: int,
    has_free_rigid: int,
    unit_count: wp.array[int],
    unit_order: wp.array[int],
    unit_contact: wp.array[int],
    unit_side_a: wp.array[int],
    unit_side_b: wp.array[int],
    unit_sequence: wp.array[int],
    seq_row_start: wp.array[int],
    art_model_world: wp.array[int],
    is_free_rigid: wp.array[int],
    # outputs
    world_rows: wp.array3d[int],
):
    """Add each unit's rows to its model world's counts in Steel's row family (grid: ``unit_max``).

    Entry ``j`` of the row order adds its rows to ``world_rows[family, SCENE_WORLD_ROWS_REQUESTED, world]``, and to
    ``SCENE_WORLD_ROWS_KEPT`` when it was kept or ``SCENE_WORLD_ROWS_DROPPED_CONTACT`` when a contact unit was dropped
    past ``row_max``. ``world`` is the model world of the unit's ``side_a`` (``scene_units._unit_model_world``).
    ``family`` is Steel's: 1 (free-body rows) for a contact whose sides are all free bodies, as Steel routes such
    contacts when the model has free bodies (``is_mf`` of ``kernels._allocate_world_contact_slot``: each side without
    response DOFs, which has no unit side, or a free body), and for a free body's late unit (its velocity-limit rows,
    which Steel keeps after its free-body contact rows), else 0. Free bodies have no internal unit. Integer atomics, so
    the counts do not depend on the order of the threads.
    """
    j = wp.tid()
    if j >= unit_count[0]:
        return
    start = seq_row_start[j]
    end = seq_row_start[j + 1]
    slot = unit_order[unit_sequence[j]]
    side_a = unit_side_a[slot]
    side_b = unit_side_b[slot]
    contact = unit_contact[slot]
    world = _unit_model_world(side_a, side_b, art_model_world, world_rows.shape[2])
    family = int(0)
    if has_free_rigid != 0 and is_free_rigid[side_a] != 0:
        if side_b < 0 or is_free_rigid[side_b] != 0:
            family = 1
    rows = end - start
    wp.atomic_add(world_rows, family, SCENE_WORLD_ROWS_REQUESTED, world, rows)
    if end <= row_max:
        wp.atomic_add(world_rows, family, SCENE_WORLD_ROWS_KEPT, world, rows)
    elif contact >= 0:
        wp.atomic_add(world_rows, family, SCENE_WORLD_ROWS_DROPPED_CONTACT, world, rows)


@wp.kernel
def count_scene_torsion_rows(
    torsion_counts: wp.array[int],
    order: wp.array[int],
    member_start: wp.array[int],
    members: wp.array[int],
    contact_path: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    art_model_world: wp.array[int],
    # outputs
    world_rows: wp.array3d[int],
):
    """Add each spin row of the step to its model world's counts in Steel's articulation family (grid: the groups'
    capacity; contact torsion).

    Steel appends a world's torsion rows to its dense rows, so its marks count them: ``prepare_torsion_rows`` sets the
    world's row and slot counts past them. Sorted position ``k`` of the step's spin rows (:func:`scene_torsion_counts`)
    adds one row to ``world_rows[0, SCENE_WORLD_ROWS_REQUESTED, world]`` and, when kept, to ``SCENE_WORLD_ROWS_KEPT``,
    with ``world`` the model world of its representative contact's unit's ``side_a``
    (``scene_units._unit_model_world``), where :func:`count_scene_world_rows` counts that unit's rows. Spin rows are
    dense rows (Steel's free-body contacts get none), so their family is 0. A spin row dropped past the capacity, where
    Steel raises, is requested and not kept; it is no contact row, so no dropped contact row either.
    """
    k = wp.tid()
    if k >= torsion_counts[1]:
        return
    rep = members[member_start[order[k]]]
    side_a, side_b = _contact_unit_sides(
        rep, contact_path, contact_art_a, contact_art_b, articulation_response_dof_count
    )
    world = _unit_model_world(side_a, side_b, art_model_world, world_rows.shape[2])
    wp.atomic_add(world_rows, 0, SCENE_WORLD_ROWS_REQUESTED, world, 1)
    if k < torsion_counts[2]:
        wp.atomic_add(world_rows, 0, SCENE_WORLD_ROWS_KEPT, world, 1)


@wp.kernel
def accumulate_scene_row_watermarks(
    # in/out
    world_rows: wp.array3d[int],
    # outputs
    watermarks: wp.array[int],
):
    """Fold each model world's rows of the step into Steel's row high-water marks, then clear them.

    Grid: families x worlds.

    The marks of family ``f`` from ``f * ROW_WATERMARK_FAMILY_STRIDE``, as Steel's
    ``accumulate_row_watermarks`` per world: the most rows kept and requested, the most dropped
    contact rows and, for a world that lost rows, the most rows lost and one overflowing world-step.
    Steel's excess is over the world's own budget; the scene pools the budgets, so a world's excess
    is the rows of its dropped units. Integer ``atomic_max`` and ``atomic_add``, so the marks do not
    depend on the order of the threads.
    """
    family, world = wp.tid()
    kept = world_rows[family, SCENE_WORLD_ROWS_KEPT, world]
    requested = world_rows[family, SCENE_WORLD_ROWS_REQUESTED, world]
    dropped = world_rows[family, SCENE_WORLD_ROWS_DROPPED_CONTACT, world]
    base = family * ROW_WATERMARK_FAMILY_STRIDE
    wp.atomic_max(watermarks, base, kept)
    wp.atomic_max(watermarks, base + 1, requested)
    wp.atomic_max(watermarks, base + 2, dropped)
    if requested > kept:
        wp.atomic_max(watermarks, base + 3, requested - kept)
        wp.atomic_add(watermarks, base + 4, 1)
    world_rows[family, SCENE_WORLD_ROWS_KEPT, world] = 0
    world_rows[family, SCENE_WORLD_ROWS_REQUESTED, world] = 0
    world_rows[family, SCENE_WORLD_ROWS_DROPPED_CONTACT, world] = 0


class SceneSolve:
    """The scene's rows, placed by scans in unit order (``solve_mode="scene"``).

    Built by :func:`build_scene_rows` after :class:`SceneUnits` has built,
    sorted and colored the step's units. ``side_count`` is the plan's articulations, ``unit_max``
    the units' capacity and ``row_max`` (:attr:`ScenePlan.row_max`) the rows' capacity.

    **Row order.** First every one-sided unit but the late ones (internal units, ground or static
    contacts, self-contacts), side-major and in sorted order within a side, so side ``s``'s one-sided rows are
    the contiguous ``[side_row_begin[s], side_row_end[s])``; then the two-sided units in
    ``order_by_color`` order: color 0 to the last color, then the tail; then the late units (each
    articulation's joint velocity-limit rows, each free body's velocity-limit rows), side-major, so side ``s``'s late
    rows are the contiguous ``[side_late_begin[s], side_late_end[s])``. A contact unit's rows are
    its normal row and its two friction rows, in that order; an internal unit's rows are its row
    families' in Steel's slot order (``FAMILY_DRIVE`` to ``FAMILY_JOINT_FRICTION``), each family's in
    Steel's ballot order: the joint limits' lower then upper bound by DOF, the joint friction's by DOF; a late
    unit's rows are the velocity limits' lower then upper bound by DOF (by root DOF for a free body).
    With contact torsion the spin rows follow every unit's rows (:meth:`prepare_torsion`, after the rows are written):
    the one-sided ones side-major, then the two-sided ones by the color of their group's representative contact unit.
    Every array below is a function of the sorted units and their colors.

    - ``one_sided_start`` (``int32``, ``[side_count + 1]``): exclusive scan of the phase-A one-sided units per
      side, the units' ``one_sided_count`` less their ``late_count``; entry ``side_count`` is their number.
    - ``late_start`` (``int32``, ``[side_count + 1]``, with late rows): exclusive scan of the units'
      ``late_count``; side ``s``'s late unit is entry ``one_sided_start[side_count] + two-sided units +
      late_start[s]`` of the row order (:func:`scene_late_sequence`).
    - ``side_late_begin``, ``side_late_end`` (``int32``, ``[side_count]``, with late rows): each side's late
      rows, clamped to the kept rows (:func:`scene_side_late_rows`).
    - ``unit_sequence`` (``int32``, ``[unit_max]``): the sorted positions of the units in row order;
      ``-1`` from ``unit_count`` on.
    - ``seq_rows``, ``seq_row_start`` (``int32``, ``[unit_max + 1]``): rows of each entry of the
      row order, ``0`` from ``unit_count`` on, and their exclusive scan, so entry ``j``'s rows are
      ``[seq_row_start[j], seq_row_start[j + 1])`` when kept and ``seq_row_start[unit_count]`` is the
      rows requested.
    - ``unit_row_start`` (``int32``, ``[unit_max]``, per slot): the unit's first row, ``-1`` when it
      was dropped.
    - ``side_row_begin``, ``side_row_end`` (``int32``, ``[side_count]``): each side's one-sided rows,
      clamped to the kept rows.
    - ``side_dropped`` (``int32``, ``[side_count]``): rows of dropped units by ``side_a``, so their
      sum is the rows dropped.
    - ``rows_requested``, ``overflow`` (``int32``, ``[1]``): the rows the units asked for, and ``1``
      when the step lost rows, units or contacts.
    - ``row_free_prefix`` (``int32``, ``[row_max]``): leading entries of a row in physical free-body
      coordinates, 0, 6 or 12.

    The solver's ``_constraint_count[0]`` receives the kept rows. Units whose rows would pass
    ``row_max`` are dropped in row order: the kept units are a prefix of ``unit_sequence``, so the late
    units go first, and the kept late rows are a prefix of the late rows in row order. A
    dropped contact gets ``contact_path == contact_slot == -1``; a contact past ``scene_unit_max``
    has no unit and keeps ``contact_slot == -1``.

    **The factor groups.** Each factor group of the plan (:class:`SceneGroup`) has its own device
    storage and kernels (:class:`SceneGroupArrays`), allocated here, and every pass over the
    articulated factors launches once per group with Steel's kernels: :meth:`factor` (stage 1),
    :meth:`solve_mass` (the unconstrained step), :meth:`gather_armature` (construction and
    notifications), :meth:`launch_internal_rows` (the internal unit's COUNT and WRITE passes, one
    launch per row family in use) and the decode.

    - ``Linv`` (``float32``, ``[sum over the groups of members x nonzeros]``): the factors ``L^-1``
      of every factor group, group ``g``'s block from its ``linv_offset``; each group's ``Linv``
      is a 2-D view of its block.
    - ``group_arrays``: the factor groups' :class:`SceneGroupArrays`, in plan order.
    - ``free_group_to_art`` (``int32``, ``[free bodies]``, ``None`` without free bodies): the free
      group's articulations, the rows of the decode's free-body launch. That is Steel's size-6
      ``group_to_art`` when the free bodies are its only members, and an array of their own when
      a six-DOF topology shares Steel's size-6 group with them.

    **The contact responses.** The kernel override ``scene_response_kernel``, read at construction, selects the kernel
    that writes the contact rows' responses (:data:`SCENE_RESPONSE_KERNELS`): ``"generic"`` (the default), the
    group-table kernel (:func:`~newton._src.solvers.steel.scene_rows._get_scene_contact_response_kernel`), which reads
    each endpoint's plan and factor through its group, or ``"specialized"``, the solver's own kernel specialised on its
    one factor plan (or, without a factor, on the free bodies). The generic kernel's arithmetic per endpoint is the
    specialised kernel's, so on a scene of one factor group the two write the same rows bit for bit; it is the default
    because it costs within 1 % of the specialised kernel on one topology. The specialised kernel, Steel's own
    (``sparse_contact._get_sparse_contact_response_kernel``), is the generic kernel's reference on three-row contacts,
    and is refused with one-row contacts (:attr:`one_row_contacts`). It indexes the factors by Steel's group index of
    the articulation's size, so it serves only a scene of at most one factor group whose members are the whole of
    Steel's group of their size: ``"specialized"`` raises :class:`ValueError` on a scene of several factor groups and
    :class:`NotImplementedError` when a six-DOF topology shares Steel's size-6 group with the free bodies. The table
    (:func:`scene_response_table`) is built in either case:

    - ``art_group``, ``art_slot`` (``int32``, ``[articulations]``): each articulation's factor
      group and its position in the group, ``-1`` for free bodies and articulations without a
      response.
    - ``response_table``: the per-group DOFs, nonzeros, first factor entries and plan bases, the
      groups' concatenated permutations and row offsets, and ``Linv``, the groups' flat factors.
    - ``response_kind``, ``response_kernel``, ``factor_inputs``: the selected kernel and the
      arguments that take the plan arrays' place in it: the group table, the one group's
      permutation, row offsets and factors, or, for free bodies only, one-element stand-ins.

    Every launch has a fixed shape and nothing is read back, so the rows can be captured in a CUDA
    graph. The placement's exclusive scans, two, or three with late units, are Newton's single-launch
    :func:`~newton._src.geometry.utils.scan_with_total` on one preallocated scratch, which they share one at a time,
    outside any conditional node.

    **The sweep** (:meth:`solve`). Phase A sweeps each side's one-sided rows
    (:func:`get_sweep_sides_kernel`). A side without two-sided units this step (``side_coupled ==
    0``) depends on no other side, so one launch runs all its iterations with Steel's stationary
    exit per side; with late rows each of its sweeps then runs its phase C, its late rows (and the fused
    clamp of its drive rows). When the step has two-sided units (``any_coupled``), each iteration then gives
    every coupled side one phase-A sweep and runs phase B (in the solver's ``scene_phase_order``:
    ``"BA"`` by default, ``"AB"``, or ``"ABA"`` with a second phase-A sweep after phase B, see
    :data:`SCENE_PHASE_ORDERS`), and with late rows ends with phase C, one sweep of the coupled sides' late
    rows (:meth:`_sweep_late`): the velocity limits have the last word in each iteration, as in Steel. Phase B is one
    sweep of the two-sided units: the colors in order, all units of a color at once (they share no side), then the tail,
    whose units may share sides, on one thread. The default schedule (``"while_block"``) runs the leading colors with
    more than :data:`SCENE_BLOCK_COLOR_MAX` units through a conditional loop of the color kernel
    (:func:`get_sweep_color_kernel`, :func:`reset_color_loop`, :func:`advance_color`), then the remaining colors and the
    tail in one block (:func:`get_sweep_block_kernel`). Steel's decode kernels then write ``v_out``. With
    ``velocity_iterations`` the step then runs the velocity pass on the same rows: :meth:`prepare_velocity_rows` writes
    their right-hand side without the position bias and :meth:`solve` with ``reset=False`` sweeps them again from the
    position solve's ``delta`` and impulses (``_ScenePass``: the pass's right-hand side, sweeps, friction start, weights
    and frozen drives, which every sweep launch reads). The kernel overrides
    (:func:`~newton._src.solvers.steel._testing.kernel_overrides`), read at construction:

    - ``scene_side_sweep``: phase A's lanes per side, ``"lanes8"`` (default) or ``"thread"``.
    - ``scene_unit_sweep``: phase B's lanes per unit, ``"lanes8"`` (default) or ``"thread"``.
    - ``scene_sum_order``: ``"steel"`` (default) sums a row's products in Steel's 32-lane order, so
      the sweep is bitwise Steel's on the same rows; ``"serial"`` sums them in entry order. Rows wider
      than 32 entries default to ``"serial"`` (and reject ``"steel"``).
    - ``scene_sweep``: ``"phases"`` (default) or ``"serial"``, one thread walking every row in row
      order with a stationary exit over the scene: the oracle of the phases. In the phase orders
      ``"BA"`` and ``"ABA"``, and in every order under contact torsion, the oracle walks each sweep in the order's
      sequence instead (:func:`get_sweep_order_serial_kernel`). On the CPU the sweep is ``"serial"`` whatever this
      says: the one-thread kernels compile for the host through the shim (:data:`_HOST_SHIM_CPP`), the lane groups
      and the block kernel do not.
    - ``scene_color_loop``: phase B's schedule. ``"while_block"`` (default) as above; ``"while"``,
      the conditional loop for every color, then a one-thread walk of the tail
      (:func:`get_sweep_rows_serial_kernel` over ``RANGE_TAIL``); ``"block"``, the block for every
      color and the tail; ``"unroll"``, the debugging fallback, ``color_cap`` launches of the color
      kernel with a constant color (empty colors return at once), then the tail's walk;
      ``"serial"``, one thread walking the two-sided rows ``RANGE_TWO_SIDED`` in row order, the
      oracle of phase B.
    - ``scene_block_color_max``: the default schedule's threshold, :data:`SCENE_BLOCK_COLOR_MAX`
      units unless set (a test and tuning seam): the leading colors with more units run through the
      while loop. Colors of first-fit are only roughly non-increasing in size, so a big color after
      a small one runs in the block, correctly but more slowly.

    The phases and the colors reproduce their oracles bit for bit when their arithmetic is the
    one-thread loop's: Steel's sums at either granularity, or ``"thread"`` sides and units with
    serial sums. 8 lanes with serial sums add in another order, so no oracle is bitwise for them.

    A round of the conditional loop was measured at about 10 µs on an L40 and an ``"unroll"`` launch
    at about 1.6 µs on the color kernel's grid, so the unroll's fixed 64 launches become the cheaper
    loop from about 17 colors through the while loop: re-time ``"unroll"`` against ``"while_block"``
    on a scene that runs more than about 17 colors through it.

    Arrays of the sweep, allocated here:

    - ``row_ranges`` (``int32``, ``[4, 2]``): ``[begin, end)`` of every kept row (``RANGE_ALL``), of
      the two-sided units' rows (``RANGE_TWO_SIDED``), of the tail's (``RANGE_TAIL``) and of the late rows
      (``RANGE_LATE``); the two-sided and tail ranges end at the first late row.
    - ``used_colors``, ``big_colors`` (``int32``, ``[1]``): the colors below the cap in use (the
      highest plus one) and the leading ones the while loop runs (:func:`scene_row_ranges`).
    - ``color_cursor``, ``color_keep_going`` (``int32``, ``[1]``): the while loop's color and
      condition.

    ``any_coupled`` is :class:`SceneUnits`' ``has_two_sided`` (``int32``, ``[1]``): ``1`` when the
    step has a two-sided unit, kept or dropped for rows.

    **Per-row features.** ``features`` (:class:`SceneFeatures`) are the sweep kernels' build
    constants, from the solver's options; every factory call passes them, so solvers with equal
    features share their kernels. With any feature on, every sweep launch passes every feature's
    argument (:meth:`_feature_inputs`), a placeholder for a feature that is off. ``friction_start`` is
    Steel's ``_friction_start_iteration`` of ``iterations``: with
    ``contact_friction_position_iterations`` every sweep launch takes it, less the launch's first
    iteration, and the sweeps before it skip the tangent pairs. With ``contact_regularization``,
    ``row_w`` (``float32``, ``[1, row_max]``) is Steel's dense ``_row_w``, which
    :func:`prepare_scene_rows` fills with each row's weight (``contact_w = 1 / (1 + g)`` for a
    penetrating contact that does not rebound, 1 otherwise) and the sweeps read; it is ``None``
    otherwise, and the row build and the featured kernels take an empty ``[1, 0]`` array. With
    ``enable_joint_friction``, ``row_bound`` (``float32``, ``[1, row_max]``, allocated here) holds each
    joint-friction row's bound ``f dt`` (``f`` with its load term under Steel's load-proportional friction,
    ``SolverSteel._joint_friction_total``), which the row's builder writes and the sweeps clamp to; it is
    ``None`` otherwise, and the featured kernels take an empty ``[1, 0]`` array. With ``drive_mode="rows"``,
    ``drive_descriptors`` are Steel's four ``[1, row_max]`` drive descriptor arrays (``_drive_target_vel_bias``,
    ``_drive_vel_multiplier``, ``_drive_impulse_multiplier``, ``_drive_max_impulse``), which
    :func:`scene_drive_descriptors` writes after the bias and the sweeps read; they are ``None``
    otherwise, and the featured kernels take empty ``[1, 0]`` arrays. With ``enable_joint_velocity_limits`` or
    free-body velocity limits (``features.late_rows``) the sweeps take the late ranges ``side_late_begin`` and
    ``side_late_end``; when no articulation can bear velocity-limit rows (``late_rows_in_use`` false) they are
    allocated empty of rows (every side's range ``[0, 0)``) and nothing late is built. ``free_late_kernel`` is the
    free bodies' late-row builder when one of them can bear velocity-limit rows
    (:func:`~newton._src.solvers.steel.scene_rows.build_sparse_free_velocity_limit_rows`), ``None`` otherwise. With
    the depenetration cap (``features.depenetration_cap``) the row preparation bounds the penetrating free-body
    contacts' bias (:func:`cap_scene_depenetration`). With contact compliance (``features.compliance``) ``row_gamma``
    (``float32``, ``[1, row_max]``) holds each compliant normal row's ``gamma``, which the row preparation writes
    (:meth:`prepare_compliance`) and the sweeps read; it is ``None`` otherwise. The step's compliant and skipped
    contacts and its latched errors are Steel's (the solver's ``ComplianceBuffers``: ``counts``, ``status`` and
    ``eager_status``), which Steel's reset clears. With the fused velocity clamp
    (``features.fused_clamp``) the sweeps take Steel's ``_drive_vel_limit`` ``[1, row_max]``, which the drive rows'
    builder writes, and the step marks the driven DOFs before stage 1 (:meth:`mark_driven_dofs`). ``internal_families``
    lists the row families some factor group builds, in family order. A contact past ``contact_friction_gap_max`` has
    its normal row alone, as has a friction patch region's member past its anchors (``one_row_contacts``): its unit
    holds one row, and the generic response kernel, built to read ``contact_slots_needed``, writes it alone (the
    specialised one, which writes three, is refused); with friction patches (``features.patches``) it is the kernel's
    ``patches`` variant, whose tangent rows act at the anchors. With
    ``row_watermark``, ``world_rows`` (``int32``, ``[2, 3, model worlds]``) holds the step's kept, requested and dropped
    contact rows per Steel row family and model world, folded into Steel's marks by :meth:`accumulate_row_watermarks`;
    it is ``None`` otherwise (the spin rows count in the articulation family, as Steel's torsion rows count with its
    dense rows). With contact torsion (``features.torsion``,
    :meth:`_allocate_torsion`) the step groups the contacts into ``torsion_groups`` (:class:`SceneTorsionGroups`) and
    appends their spin rows (:meth:`prepare_torsion`); ``torsion_radius`` and ``torsion_group`` (``[1, row_max]``, the
    solver's ``_torsion_row_radius`` and ``_contact_torsion_group``), ``torsion_member_range`` and
    ``torsion_member_rows``, ``side_torsion_begin`` and ``side_torsion_end``, and ``torsion_color_start`` are the
    sweeps' torsion arguments, ``torsion_counts`` the step's first, requested and kept spin rows and
    ``torsion_status`` the latched refusals, ``torsion_shapes`` the shapes' torsion support as Steel marks it;
    ``torsion_kernel`` sweeps the two-sided spin rows (:func:`get_sweep_torsion_kernel`). Without it
    ``torsion_groups``, ``torsion_shapes`` and ``torsion_kernel`` are ``None``. With a
    friction mode (``features.friction_mode``) ``row_friction_mode`` (``int32``, ``[1, row_max]``) holds each
    contact's update on its first tangent row (:data:`SCENE_FRICTION_FIXED_NORMAL` and after), which
    :meth:`prepare_friction_modes` writes every step and the sweeps read; it is ``None`` otherwise.
    """

    def __init__(self, solver):
        units = solver._scene_units
        plan = solver._scene_plan
        device = solver.model.device
        self.device = device
        self.side_count = int(units.side_count)
        self.unit_max = int(units.unit_max)
        self.row_max = int(plan.row_max)
        self.color_total_slot = int(units.color_cap) + 1
        self.built = False
        sides, unit_max = self.side_count, self.unit_max
        self.one_sided_start = wp.zeros(sides + 1, dtype=wp.int32, device=device)
        self.unit_sequence = wp.full(unit_max, -1, dtype=wp.int32, device=device)
        self.seq_rows = wp.zeros(unit_max + 1, dtype=wp.int32, device=device)
        self.seq_row_start = wp.zeros(unit_max + 1, dtype=wp.int32, device=device)
        self.unit_row_start = wp.full(unit_max, -1, dtype=wp.int32, device=device)
        self.side_row_begin = wp.zeros(sides, dtype=wp.int32, device=device)
        self.side_row_end = wp.zeros(sides, dtype=wp.int32, device=device)
        self.side_dropped = wp.zeros(sides, dtype=wp.int32, device=device)
        self.rows_requested = wp.zeros(1, dtype=wp.int32, device=device)
        self.overflow = wp.zeros(1, dtype=wp.int32, device=device)
        self.row_free_prefix = wp.zeros(self.row_max, dtype=wp.int32, device=device)
        # The units' one-sided counts with their trailing zero (side_a < side_count), whose
        # exclusive scan ends in the one-sided total.
        self._one_sided_counts = units._side_counts[int(_SIDE_ONE_SIDED)]
        # The placement's exclusive scans (assign_rows): one scratch, which they share one at a time, and the length
        # of each scan, every entry of its counts.
        self._scan_scratch = wp.zeros(
            _scan_scratch_size(max(sides, unit_max) + 1, device), dtype=wp.int64, device=device
        )
        self._scan_total = wp.zeros(1, dtype=wp.int32, device=device)
        self._side_scan_length = wp.full(1, sides + 1, dtype=wp.int32, device=device)
        self._unit_scan_length = wp.full(1, unit_max + 1, dtype=wp.int32, device=device)
        overrides = solver._kernel_overrides
        # The factor groups' storage: one flat Linv holding each group's block, and per group its
        # articulations, plan indices, armature, status, scratch and factor kernel.
        factor = [group for group in plan.groups if group.kind == "factor"]
        self.Linv = wp.zeros(
            sum(len(group.arts) * group.nonzero_count for group in factor), dtype=wp.float32, device=device
        )
        self.group_arrays = tuple(self._group_arrays(solver, group) for group in factor)
        # The free bodies' decode rows: Steel's size-6 group unless a six-DOF topology shares it.
        free_arts = np.asarray([art for group in plan.groups if group.kind == "free" for art in group.arts], np.int32)
        self.free_group_to_art = None
        if free_arts.size:
            steel_six = np.flatnonzero(solver._model_plan.response_dof_count == 6)
            self.free_group_to_art = (
                solver._group_to_art[6]
                if np.array_equal(steel_six, free_arts)
                else wp.array(free_arts, dtype=wp.int32, device=device)
            )
        # The table of the generic response kernel over the same groups and the same Linv.
        response_dof_count = solver._model_plan.response_dof_count
        self.art_group, self.art_slot, self.response_table = scene_response_table(
            plan.groups,
            solver.model.articulation_count,
            self.Linv,
            device,
            articulated=(response_dof_count > 0) & (solver._model_plan.is_free_rigid == 0),
        )
        # The generic kernel is the default. The specialised kernel reads one plan and indexes the
        # factors by Steel's group index of the articulation's size, so it serves a scene of at most
        # one factor group, in Steel's size order (a six-DOF topology beside free bodies shares
        # Steel's size group with them, so it is not).
        one_group = len(factor) <= 1
        steel_order = not factor or np.array_equal(np.flatnonzero(response_dof_count == factor[0].size), factor[0].arts)
        self.response_kind = str(overrides.get("scene_response_kernel", SCENE_RESPONSE_KERNELS[0]))
        if self.response_kind not in SCENE_RESPONSE_KERNELS:
            raise ValueError(
                f"scene_response_kernel must be one of {list(SCENE_RESPONSE_KERNELS)}, got {self.response_kind!r}"
            )
        if self.response_kind == "specialized" and not one_group:
            raise ValueError(
                f"scene_response_kernel='specialized' reads one factor group, the scene has {len(factor)}; "
                "use 'generic'"
            )
        if self.response_kind == "specialized" and not steel_order:
            raise NotImplementedError("the specialised response kernel reads a factor group in Steel's size order")
        if self.response_kind == "specialized" and plan.mask_words > 1:
            raise ValueError(
                f"scene_response_kernel='specialized' reads one-word support masks, the scene's have "
                f"{plan.mask_words} words; use 'generic'"
            )
        # Contacts past contact_friction_gap_max get a normal row only (contact_slots_needed == 1), and so do a
        # friction patch region's members past its anchors (Steel's allocation, kernels._allocate_world_contact_slot),
        # under any gap: the generic kernel then reads each contact's row count, where it writes three otherwise.
        patches = bool(solver._friction_anchors_enabled)
        one_row_causes = [
            cause
            for cause, on in (
                ("contact_friction_gap_max", not solver._friction_every_contact),
                ("friction patches", patches),
            )
            if on
        ]
        self.one_row_contacts = bool(one_row_causes)
        if self.response_kind == "specialized" and self.one_row_contacts:
            raise ValueError(
                "scene_response_kernel='specialized' is the generic kernel's reference on three-row contacts, the "
                f"scene's contacts may have one ({', '.join(one_row_causes)}); use 'generic'"
            )
        # The kernel and its factor inputs, the arguments between the anchors and the free
        # factors. A plan of free bodies only has no factor, and so no specialised kernel of the
        # solver's own: every endpoint takes the free-body branch, which reads none of the factor
        # inputs, so one-element stand-ins fill them.
        lanes = solver._sparse_contact_lanes
        if self.response_kind == "generic":
            # A model without responding bodies has no group: the free-body size, whose kernel no contact reaches.
            dof_max = max((group.size for group in plan.groups), default=6)
            _check_response_shared_memory(dof_max, lanes, device)
            # Friction patches: the variant whose tangent rows act at the anchor points (it reads the row counts too).
            self.response_kernel = _get_scene_contact_response_kernel(
                dof_max, plan.mask_words, lanes_per_contact=lanes, slots_needed=self.one_row_contacts, patches=patches
            )
            self.factor_inputs = (self.art_group, self.art_slot, *self.response_table)
        elif factor:
            # Steel's own kernel of the one group's plan (its sparse path's _sparse_contact_response_kernel).
            arrays = self.group_arrays[0]
            self.response_kernel = _get_sparse_contact_response_kernel(factor[0].size, lanes_per_contact=lanes)
            self.factor_inputs = (arrays.indices.permutation, arrays.indices.row_offsets, arrays.Linv)
        else:
            self.response_kernel = _get_sparse_contact_response_kernel(6, lanes_per_contact=lanes)
            self.factor_inputs = (
                wp.zeros(1, dtype=wp.int32, device=device),
                wp.zeros(1, dtype=wp.int32, device=device),
                wp.zeros((1, 1), dtype=wp.float32, device=device),
            )
        # The one-row variant's anchor points without patches (anchor_points): placeholders it does not read.
        self._no_anchor_points = None
        if self.one_row_contacts and not patches:
            self._no_anchor_points = wp.zeros(1, dtype=wp.vec3, device=device)
        # The sweep's build constants, each from its option (every later per-row feature sets its own),
        # and the first sweep that solves the tangent pairs under friction position iterations.
        self.features = SceneFeatures(
            friction_position_iterations=solver._contact_friction_position_iterations >= 0,
            regularize=solver._regularization_enabled,
            joint_friction=bool(solver._enable_joint_friction),
            drives=bool(solver._has_drive_rows),
            late_rows=bool(solver._scene_late_rows_enabled),
            fused_clamp=bool(solver._velocity_limits_fused),
            depenetration_cap=bool(solver._scene_depenetration_cap),
            torsion=bool(solver._contact_torsion_enabled),
            compliance=bool(solver._contact_compliance),
            friction_mode=str(solver._friction_mode),
            patches=patches,
        )
        self.friction_start = int(solver._friction_start_iteration(solver._iterations))
        features = self.features
        # Whether the sweep kernels are the featured definition, which takes every feature's argument.
        self._featured = bool(_feature_suffix(features))
        # contact_regularization: each row's weight is Steel's dense _row_w, [1, row_max] with
        # regularization only, which prepare_scene_rows writes every step. Without it the row build and
        # the featured kernels take an empty [1, 0] stand-in, which holds no device memory.
        self.contact_w = float(solver._contact_w)
        self.row_w = solver._row_w if features.regularize else None
        self._row_w_arg = self.row_w if self.row_w is not None else wp.empty((1, 0), dtype=wp.float32, device=device)
        # velocity_iterations: Steel's velocity pass runs without regularization, weight 1 on every row, so the
        # regularized sweeps take a ones [1, row_max] in it (the rigid update at every relaxation).
        self.velocity_iterations = int(solver._velocity_iterations)
        self._velocity_row_w_arg = self._row_w_arg
        if self.velocity_iterations > 0 and features.regularize:
            self._velocity_row_w_arg = wp.ones((1, self.row_max), dtype=wp.float32, device=device)
        # The pass solve() runs, which the sweep launches read (_row_inputs, _feature_inputs).
        self._pass = None
        # enable_joint_friction: each joint-friction row's bound, [1, row_max], which its builder writes; the
        # featured kernels take an empty [1, 0] stand-in without it.
        self.row_bound = (
            wp.zeros((1, self.row_max), dtype=wp.float32, device=device) if features.joint_friction else None
        )
        self._row_bound_arg = (
            self.row_bound if self.row_bound is not None else wp.empty((1, 0), dtype=wp.float32, device=device)
        )
        # drive_mode="rows": Steel's drive descriptors, [1, row_max] in the scene's one solve world, which
        # scene_drive_descriptors writes; the featured kernels take empty [1, 0] stand-ins without them.
        self.drive_descriptors = (
            (
                solver._drive_target_vel_bias,
                solver._drive_vel_multiplier,
                solver._drive_impulse_multiplier,
                solver._drive_max_impulse,
            )
            if features.drives
            else None
        )
        self._drive_args = (
            self.drive_descriptors
            if self.drive_descriptors is not None
            else (wp.empty((1, 0), dtype=wp.float32, device=device),) * 4
        )
        # The internal unit's row families some factor group builds, in Steel's slot order.
        self.internal_families = tuple(solver._scene_internal_families)
        # The connect family (loop closures): one launch over the closure owners, a free body among them, each
        # owner's closures in the connect table's order, through the contact response's group table.
        self.connect_kernel = None
        if FAMILY_CONNECT in self.internal_families:
            closure_art = np.asarray(solver._connect_art_np, dtype=np.int64)
            owners, closures = np.unique(closure_art, return_counts=True)
            self.connect_owner_art = wp.array(owners.astype(np.int32), dtype=wp.int32, device=device)
            self.connect_owner_start = wp.array(
                np.concatenate(([0], np.cumsum(closures))).astype(np.int32), dtype=wp.int32, device=device
            )
            self.connect_owner_closures = wp.array(
                np.argsort(closure_art, kind="stable").astype(np.int32), dtype=wp.int32, device=device
            )
            connect_dof_max = max(group.size for group in plan.groups)
            _check_response_shared_memory(connect_dof_max, solver._sparse_contact_lanes, device)
            # The closures' most rows: 3 for CONNECT closures, which the builder supports alone.
            self.connect_kernel = get_scene_connect_rows_kernel(
                connect_dof_max,
                plan.mask_words,
                max_closure_rows=int(np.max(solver._connect_rows_np)),
                lanes_per_contact=solver._sparse_contact_lanes,
            )
        # Joint and free-body velocity limits: the late units, one per articulation (free body) that bears
        # velocity-limit rows, and each side's late rows, which the sweeps read ([0, 0) every step when no group builds
        # late rows). Their placement needs the phase-A one-sided counts (the units' less their late units) and the
        # late units' scan.
        free_group = next((group for group in plan.groups if group.kind == "free"), None)
        self.free_late_kernel = scene_group_late(solver, free_group) if free_group is not None else None
        self.late_rows_in_use = self.free_late_kernel is not None or any(
            arrays.late_kernel is not None for arrays in self.group_arrays
        )
        self.side_late_begin = self.side_late_end = self.late_start = None
        if _has_phase_c(features):
            # Contact torsion runs phase C too: without velocity limits its late ranges stay empty.
            self.side_late_begin = wp.zeros(sides, dtype=wp.int32, device=device)
            self.side_late_end = wp.zeros(sides, dtype=wp.int32, device=device)
        if self.late_rows_in_use:
            self.late_start = wp.zeros(sides + 1, dtype=wp.int32, device=device)
            self._unit_one_sided_counts = self._one_sided_counts
            self._late_counts = units._side_counts[int(_SIDE_LATE)]
            self._one_sided_counts = wp.zeros(sides + 1, dtype=wp.int32, device=device)
        empty_sides = wp.empty(0, dtype=wp.int32, device=device)
        self._late_args = (
            (self.side_late_begin, self.side_late_end) if _has_phase_c(features) else (empty_sides, empty_sides)
        )
        # The fused velocity clamp: each drive row's DOF velocity limit, [1, row_max], which the drive rows' builder
        # writes; the featured kernels take an empty [1, 0] stand-in without it.
        self._drive_vel_limit_arg = (
            solver._drive_vel_limit if features.fused_clamp else wp.empty((1, 0), dtype=wp.float32, device=device)
        )
        # contact_compliance: each row's gamma, [1, row_max] (0 but on the compliant normal rows), which
        # prepare_scene_compliance writes every step (its counts and status are Steel's ComplianceBuffers'); the
        # featured kernels take an empty [1, 0] stand-in without it.
        self.row_gamma = None
        if features.compliance:
            self.row_gamma = wp.zeros((1, self.row_max), dtype=wp.float32, device=device)
        self._row_gamma_arg = (
            self.row_gamma if self.row_gamma is not None else wp.empty((1, 0), dtype=wp.float32, device=device)
        )
        self._allocate_torsion(solver, plan)
        # A friction mode: each contact's update on its first tangent row, [1, row_max] (SCENE_FRICTION_FIXED_NORMAL
        # elsewhere), which prepare_scene_friction_modes writes every step; the featured kernels take an empty [1, 0]
        # stand-in without it.
        self.row_friction_mode = (
            wp.zeros((1, self.row_max), dtype=wp.int32, device=device)
            if features.friction_mode != "fixed_normal"
            else None
        )
        self._friction_mode_arg = (
            self.row_friction_mode
            if self.row_friction_mode is not None
            else wp.empty((1, 0), dtype=wp.int32, device=device)
        )
        # The sweep.
        self.color_cap = int(units.color_cap)
        self.side_sweep = str(overrides.get("scene_side_sweep", "lanes8"))
        self.sum_order = str(overrides.get("scene_sum_order", _default_sum_order(plan.support)))
        self.sweep = str(overrides.get("scene_sweep", "phases"))
        if self.sweep not in SCENE_SWEEPS:
            raise ValueError(f"scene_sweep must be one of {list(SCENE_SWEEPS)}, got {self.sweep!r}")
        if not device.is_cuda:
            # The CPU sweeps serially, the phases' oracle: the lane groups' shuffles and the block kernel's barriers
            # are CUDA's alone (_sweep_snippet), so no phase-B schedule runs there.
            self.sweep = "serial"
        free_response = plan.free_body_count > 0
        # Whether rows may start with free-body entries: the sweeps' build constant, which the warm start's seed reads.
        self.free_response = bool(free_response)
        # The warm start's reset: whether a reset mask selects every world (scene_reset_world_selection).
        self._reset_all_selected = wp.zeros(1, dtype=wp.int32, device=device) if solver._contact_warmstart else None
        # The warm start's gather coefficients under friction patches with free bodies (scene_warmstart_gather_mu):
        # Steel's free-body rows are gathered before the patch link divides the anchors' coefficients.
        self.gather_mu = None
        if solver._contact_warmstart and solver._friction_anchors_enabled and plan.free_body_count > 0:
            self.gather_mu = wp.zeros((1, self.row_max), dtype=wp.float32, device=device)
        self.sides_kernel = get_sweep_sides_kernel(
            self.side_sweep, self.sum_order, plan.support, free_response=free_response, features=features
        )
        self.serial_kernel = get_sweep_rows_serial_kernel(
            self.sum_order, plan.support, free_response=free_response, features=features
        )
        # The phases of each iteration of the coupled sides, and their oracle's walk ("AB" walks RANGE_ALL).
        self.phase_order = str(solver._scene_phase_order)
        self.order_kernel = None
        # Under contact torsion the spin rows follow the late units in row order but run before them in phase C, so
        # the order oracle walks every phase order, "AB" too.
        if self.sweep == "serial" and (self.phase_order != "AB" or features.torsion):
            self.order_kernel = get_sweep_order_serial_kernel(
                self.phase_order, self.sum_order, plan.support, free_response=free_response, features=features
            )
        self.row_ranges = wp.zeros((4, 2), dtype=wp.int32, device=device)
        # Phase B, the two-sided units.
        self.unit_sweep = str(overrides.get("scene_unit_sweep", "lanes8"))
        self.color_loop = str(overrides.get("scene_color_loop", SCENE_COLOR_LOOPS[0]))
        if self.color_loop not in SCENE_COLOR_LOOPS:
            raise ValueError(f"scene_color_loop must be one of {list(SCENE_COLOR_LOOPS)}, got {self.color_loop!r}")
        _check_sweep(self.unit_sweep, self.sum_order, plan.support, key="scene_unit_sweep")
        self.color_kernel = None
        self.block_kernel = None
        if self.color_loop in ("while_block", "while", "unroll"):
            self.color_kernel = get_sweep_color_kernel(
                self.unit_sweep, self.sum_order, plan.support, free_response=free_response, features=features
            )
        if self.color_loop in ("while_block", "block"):
            self.block_kernel = get_sweep_block_kernel(
                self.unit_sweep, self.sum_order, plan.support, free_response=free_response, features=features
            )
        # Contact torsion's two-sided spin rows, swept by color in phase C.
        self.torsion_kernel = None
        if features.torsion:
            self.torsion_kernel = get_sweep_torsion_kernel(
                self.unit_sweep, self.sum_order, plan.support, free_response=free_response, features=features
            )
        lanes = SCENE_UNIT_LANES[self.unit_sweep]
        self.color_workers = max(1, min(unit_max, SCENE_COLOR_THREADS_MAX // lanes))
        # The while loop runs the leading colors with more units than this: every color ("while"),
        # those past the block's crossover (the default, or scene_block_color_max), or none ("block").
        block_color_max = int(overrides.get("scene_block_color_max", SCENE_BLOCK_COLOR_MAX))
        if block_color_max < 0:
            raise ValueError(f"scene_block_color_max must be at least 0, got {block_color_max}")
        self.block_color_max = {"while": -1, "block": unit_max}.get(self.color_loop, block_color_max)
        self.used_colors = wp.zeros(1, dtype=wp.int32, device=device)
        self.big_colors = wp.zeros(1, dtype=wp.int32, device=device)
        self.color_cursor = wp.zeros(1, dtype=wp.int32, device=device)
        self.color_keep_going = wp.zeros(1, dtype=wp.int32, device=device)
        # Whether the step has a two-sided unit: the units' own flag (incidence_start[side_count] > 0).
        self.any_coupled = units.has_two_sided
        # scene_statistics() warns once per solver when units took the tail color.
        self.tail_warned = False
        # row_watermark: each model world's rows of the step by Steel's row family, cleared as they fold
        # into the marks (accumulate_row_watermarks).
        self.world_rows = (
            wp.zeros(
                (SCENE_WATERMARK_FAMILIES, SCENE_WORLD_ROWS_DROPPED_CONTACT + 1, max(int(solver.model.world_count), 1)),
                dtype=wp.int32,
                device=device,
            )
            if solver._row_watermark
            else None
        )

    def _group_arrays(self, solver, group: SceneGroup) -> SceneGroupArrays:
        """Allocate factor group ``group``'s storage, with its ``Linv`` a view of its block of :attr:`Linv`."""
        device = self.device
        members, size, nonzeros = len(group.arts), group.size, group.nonzero_count
        warps = scene_factor_warps(solver, size, nonzeros)
        if warps < 1:
            raise NotImplementedError(
                f'solve_mode="scene" cannot factor a {size}-DOF topology of {nonzeros} factor entries '
                "in one warp's shared memory"
            )
        block = self.Linv[group.linv_offset : group.linv_offset + members * nonzeros]
        return SceneGroupArrays(
            group=group,
            group_to_art=wp.array(np.asarray(group.arts, dtype=np.int32), dtype=wp.int32, device=device),
            indices=group.plan.to_device(device),
            armature=wp.zeros((members, size), dtype=wp.float32, device=device),
            status=wp.zeros(members, dtype=wp.int32, device=device),
            scratch=wp.empty((members, size), dtype=wp.float32, device=device),
            factor_kernel=_get_crba_sparse_factor_kernel(
                size, nonzeros, warps_per_block=warps, mask_words=group.mask_words
            ),
            internal_kernels={
                family: get_sparse_joint_dof_rows_kernel(
                    family, group.mask_words, fused=family == FAMILY_DRIVE and solver._velocity_limits_fused
                )
                for family in scene_group_families(solver, group)
            },
            warps_per_block=warps,
            Linv=block.reshape((members, nonzeros)),
            late_kernel=scene_group_late(solver, group),
        )

    def _allocate_torsion(self, solver, plan: ScenePlan) -> None:
        """Allocate contact torsion's arrays (``features.torsion``), or the sweeps' placeholders without it.

        Per contact (the solver's contact capacity): the groups (:class:`SceneTorsionGroups`), the routing, the
        eligibility and the friction coefficients the grouping reads. The keys and their order (twice the capacity: the
        radix sort's scratch is the back half), and the members' normal rows. Per row (``[1, row_max]``): the spin rows'
        radius and the members' group ids, Steel's ``_torsion_row_radius`` and ``_contact_torsion_group`` (the solver
        views these), and each spin row's members' range and its lead's shape pair (``[row_max, 2]`` each, the latter
        for :meth:`torsion_statistics`). Per side the one-sided spin rows' range, per color slot (the cap, the tail, the
        end) the two-sided ones' first row, the step's counts (first spin row, requested, kept), its status and the
        latched status. Per shape, whether its type supports torsion (``torsion_shapes``,
        :class:`~newton._src.solvers.steel.scene_torsion_device.SceneTorsionShapes`), marked here and again by the
        solver's :meth:`~newton._src.solvers.steel.scene_torsion_device.SceneTorsionShapes.update` on a
        ``SHAPE_TYPE_AND_SOURCE`` notification, where Steel marks its own.
        """
        device = self.device
        self.torsion_groups = None
        self.torsion_shapes = None
        if not self.features.torsion:
            self._torsion_args = (
                wp.empty((1, 0), dtype=wp.float32, device=device),
                wp.empty((1, 0), dtype=wp.int32, device=device),
                wp.empty((0, 2), dtype=wp.int32, device=device),
                *(wp.empty(0, dtype=wp.int32, device=device) for _ in range(4)),
            )
            return
        contacts = int(solver._max_contacts_alloc)
        rows = self.row_max
        self.torsion_groups = SceneTorsionGroups.allocate(contacts, device)
        self.torsion_shapes = SceneTorsionShapes.allocate(solver.model)
        self.torsion_path = wp.zeros(contacts, dtype=wp.int32, device=device)
        self.torsion_eligible = wp.zeros(contacts, dtype=wp.int32, device=device)
        self.torsion_friction = wp.zeros(contacts, dtype=wp.float32, device=device)
        # Friction patches: each contact's model world, which keys its patch region with the owner (Steel's).
        self.torsion_world = wp.zeros(contacts, dtype=wp.int32, device=device)
        self.torsion_keys = wp.zeros(2 * contacts, dtype=wp.int64, device=device)
        self.torsion_order = wp.zeros(2 * contacts, dtype=wp.int32, device=device)
        self.torsion_member_rows = wp.zeros(contacts, dtype=wp.int32, device=device)
        self.torsion_radius = wp.zeros((1, rows), dtype=wp.float32, device=device)
        self.torsion_group = wp.full((1, rows), -1, dtype=wp.int32, device=device)
        self.torsion_member_range = wp.zeros((rows, 2), dtype=wp.int32, device=device)
        self.torsion_shape_pair = wp.zeros((rows, 2), dtype=wp.int32, device=device)
        self.side_torsion_begin = wp.zeros(self.side_count, dtype=wp.int32, device=device)
        self.side_torsion_end = wp.zeros(self.side_count, dtype=wp.int32, device=device)
        self.torsion_color_start = wp.zeros(int(solver._scene_units.color_cap) + 2, dtype=wp.int32, device=device)
        self.torsion_counts = wp.zeros(3, dtype=wp.int32, device=device)
        self.torsion_step_status = wp.zeros(1, dtype=wp.int32, device=device)
        self.torsion_status = wp.zeros(1, dtype=wp.int32, device=device)
        self._no_stiffness = wp.zeros(1, dtype=wp.float32, device=device)
        lanes = int(solver._sparse_contact_lanes)
        dof_max = max(group.size for group in plan.groups)
        _check_response_shared_memory(dof_max, lanes, device)
        self.torsion_rows_kernel = get_scene_torsion_rows_kernel(dof_max, plan.mask_words, lanes=lanes)
        self.torsion_lanes = lanes
        self._torsion_args = (
            self.torsion_radius,
            self.torsion_group,
            self.torsion_member_range,
            self.torsion_member_rows,
            self.side_torsion_begin,
            self.side_torsion_end,
            self.torsion_color_start,
        )

    def gather_armature(self, solver) -> None:
        """Copy the effective joint armature into each factor group's ``armature``, one launch per group.

        Steel's :func:`gather_group_armature` over the group's articulations; the solver calls it
        after each refresh of the effective armature (at construction and on a notification).
        """
        for arrays in self.group_arrays:
            wp.launch(
                gather_group_armature,
                (len(arrays.group.arts), arrays.group.size),
                inputs=[
                    arrays.group_to_art,
                    solver._articulation_dof_start,
                    solver._articulation_H_rows,
                    solver._joint_armature_device,
                ],
                outputs=[arrays.armature],
                device=self.device,
            )

    def factor(self, solver, state_aug) -> None:
        """Assemble and factor the mass matrices of the articulations due for a refresh, group by group.

        Per factor group, Steel's ``_stage1_sparse_factor``: the implicit drive terms of its
        articulations (:func:`scatter_augmented_drive_dof_K`, with augmented drive rows) and its
        factor kernel into its block of :attr:`Linv`, one launch each.
        """
        model = solver.model
        for arrays in self.group_arrays:
            members = len(arrays.group.arts)
            if solver._articulation_max_dofs > 0:
                wp.launch(
                    scatter_augmented_drive_dof_K,
                    dim=members,
                    inputs=[
                        arrays.group_to_art,
                        solver._articulation_dof_start,
                        arrays.group.size,
                        solver._articulation_max_dofs,
                        solver._mass_update_mask,
                        solver._aug_row_counts,
                        solver._aug_row_dof_index,
                        solver._aug_row_K,
                    ],
                    outputs=[solver._sparse_drive_dof_K],
                    device=self.device,
                )
            wp.launch(
                arrays.factor_kernel,
                dim=members * 32,
                inputs=[
                    arrays.group_to_art,
                    solver._mass_update_mask,
                    model.articulation_start,
                    solver._articulation_dof_start,
                    model.joint_child,
                    state_aug._joint_S_s,
                    solver._body_I_c,
                    arrays.armature,
                    solver._sparse_drive_dof_K,
                    arrays.indices,
                ],
                outputs=[arrays.Linv, arrays.status],
                block_dim=32 * arrays.warps_per_block,
                device=self.device,
            )

    def solve_mass(self, solver, rhs: wp.array, out: wp.array) -> None:
        """Write ``out = H^-1 rhs`` on each active articulation of the factor groups, one launch per group.

        Steel's :func:`solve_sparse_mass_matrix` with the group's factors; ``rhs`` and ``out`` are
        DOF arrays, the joint torques and accelerations of the unconstrained step.
        """
        for arrays in self.group_arrays:
            wp.launch(
                solve_sparse_mass_matrix,
                dim=len(arrays.group.arts) * 32,
                inputs=[
                    arrays.group_to_art,
                    solver._articulation_dof_start,
                    arrays.indices,
                    arrays.Linv,
                    rhs,
                    solver._dynamics_art_active,
                ],
                outputs=[arrays.scratch, out],
                block_dim=128,
                device=self.device,
            )

    def launch_internal_rows(
        self,
        solver,
        state_in,
        mode: int,
        cursor: wp.array,
        *,
        dt: float,
        control=None,
        limit_q_index: wp.array | None = None,
        state_aug=None,
    ) -> None:
        """Run the internal unit's row builders in ``mode``, one launch per family in use per factor group.

        The families run in Steel's slot order (:attr:`internal_families`), each over the groups that
        build it (:attr:`SceneGroupArrays.internal_kernels`): the drives with Steel's drive law on the
        step's ``control`` (its joint targets; required with the drive family), the mimic rows with Steel's
        mimic law on ``state_in``'s coordinates, the joint limits with
        Steel's sparse joint-limit builder of the group's mask words, exactly as before the internal
        unit, and the joint friction with its law at the step's ``dt`` (on Steel's load-proportional totals when
        they are active). The connect family is one launch over
        every closure owner (:meth:`_launch_connect_rows`), free bodies included, on ``state_in``'s poses and
        ``state_aug``'s motion subspaces (required with the connect family). Every builder works per
        articulation, so the order of the groups' launches does not change the rows.

        ``cursor`` is a 1-D ``[articulations]`` array that every family advances in turn, or, for
        ``ROW_SLOTS_COUNT``, the 2-D ``[FAMILY_COUNT, articulations]`` family counts, each family adding
        to its own row. ``ROW_SLOTS_COUNT`` adds each articulation's rows to the cursor and writes no
        row; ``ROW_SLOTS_WRITE`` writes them from ``cursor[art]``, the unit's first row, family after
        family (a family's rows in Steel's ballot order: the drives' by DOF, the mimic rows and the closures'
        three rows in their tables' order, a limit's lower then upper bound by DOF, the joint friction's by
        DOF), and none at or past the row capacity. The mimic and connect families read their enabled flags
        (:attr:`~newton.Model.constraint_mimic_enabled`, :attr:`~newton.Model.joint_enabled`) in both modes, so a
        released closure has no rows from the step it is released on, under capture too.

        ``limit_q_index`` (``[DOFs]``, default ``SolverSteel._joint_limit_q_index``) is each DOF's coordinate that
        the drive and joint-limit families read, ``-1`` for none; with sleeping it is Steel's masked copy
        (``_SleepState.limit_q_index``), ``-1`` on every DOF of an articulation that sleeps with its rows skipped,
        whose joint friction then skips it through ``_constraint_art_active`` (``0``): such an articulation gets no
        internal rows in COUNT or WRITE, as in Steel, where its limit rows vanish through the same index, its joint
        friction through the same flag, and its drive rows cannot exist (a positive gain keeps it awake). The mimic
        and connect families read neither: Steel's sleeping never lets an articulation that owns a mimic row or a
        loop closure sleep (``sleeping._pinned_articulations``), so their owners are awake and get their rows every
        step, as in Steel.
        """
        model = solver.model
        if FAMILY_DRIVE in self.internal_families and control is None:
            raise ValueError("the internal unit's drive rows read the step's control (its joint targets)")
        if FAMILY_CONNECT in self.internal_families and state_aug is None:
            raise ValueError("the internal unit's connect rows read the step's motion subspaces (state_aug)")
        if limit_q_index is None:
            limit_q_index = solver._joint_limit_q_index
        for family in self.internal_families:
            family_cursor = cursor[family] if cursor.ndim == 2 else cursor
            if family == FAMILY_CONNECT:
                self._launch_connect_rows(solver, state_in, state_aug, mode, family_cursor)
                continue
            for arrays in self.group_arrays:
                kernel = arrays.internal_kernels.get(family)
                if kernel is None:
                    continue
                indices = arrays.indices
                factor = [
                    indices.ancestor_mask,
                    indices.inverse_permutation,
                    indices.lookup,
                    arrays.Linv,
                    solver._v_hat,
                ]
                rows = [
                    solver._slot_counter,
                    solver._row_type,
                    solver._row_parent,
                    solver._row_mu,
                    solver._phi,
                    solver._target_velocity,
                    solver._sparse_row_dof,
                    solver._sparse_row_factor,
                    solver._sparse_row_incident,
                    solver._diag,
                ]
                if family == FAMILY_DRIVE:
                    law = [
                        # Each DOF's coordinate, -1 outside PRISMATIC, REVOLUTE and D6 joints: the drives'
                        # joint types, and their targets' index (joint_target_q_start is joint_q_start).
                        limit_q_index,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        model.joint_effort_limit,
                        state_in.joint_q,
                        control.joint_target_q,
                        control.joint_target_qd,
                        # Read and written by the fused velocity clamp's build alone.
                        model.joint_velocity_limit,
                    ]
                    rows += [
                        solver._drive_stiffness,
                        solver._drive_damping,
                        solver._drive_geom_error,
                        solver._drive_max_force,
                        solver._drive_vel_limit,
                    ]
                elif family == FAMILY_MIMIC:
                    law = [
                        solver._mimic_art_start,
                        solver._mimic_art_list,
                        solver._mimic_dof0,
                        solver._mimic_dof1,
                        solver._mimic_q0,
                        solver._mimic_q1,
                        solver._mimic_legacy,
                        solver._mimic_owner,
                        solver._mimic_enabled,
                        solver._mimic_coef0,
                        solver._mimic_coef1,
                        solver._joint_mimic_coeffs,
                        state_in.joint_q,
                    ]
                elif family == FAMILY_LIMIT:
                    law = [
                        limit_q_index,
                        model.joint_limit_lower,
                        model.joint_limit_upper,
                        state_in.joint_q,
                        solver._joint_limit_activation_gap,
                    ]
                elif family == FAMILY_JOINT_FRICTION:
                    # Steel's rule: with load-proportional friction, each DOF's friction plus its load term, which
                    # SolverSteel._stage3_joint_friction_load writes before the rows are built.
                    joint_friction = (
                        solver._joint_friction_total if solver._joint_friction_load_active else model.joint_friction
                    )
                    law = [
                        solver._constraint_art_active,
                        solver._friction_dof_supported,
                        solver._kinematic_dof_mask,
                        joint_friction,
                        dt,
                    ]
                    rows.append(self.row_bound)
                else:
                    raise NotImplementedError(f"the scene's internal unit has no launch for row family {family}")
                wp.launch(
                    kernel,
                    dim=len(arrays.group.arts) * 32,
                    inputs=[
                        arrays.group_to_art,
                        solver._art_to_world,
                        solver._articulation_world_dof_offset,
                        solver._articulation_dof_start,
                        *law,
                        *factor,
                    ],
                    outputs=[*rows, mode, family_cursor],
                    block_dim=128,
                    device=self.device,
                )

    def _launch_connect_rows(self, solver, state_in, state_aug, mode: int, cursor: wp.array) -> None:
        """Run the connect family's builder in ``mode``: one lane group per closure owner (``connect_owner_art``).

        :func:`~newton._src.solvers.steel.scene_rows.get_scene_connect_rows_kernel` on the solver's connect table
        and the contact response's group table (:attr:`art_group`, :attr:`art_slot`, :attr:`response_table`), with
        ``state_in``'s poses and twists and ``state_aug``'s motion subspaces; ``cursor`` (``[articulations]``) as in
        :meth:`launch_internal_rows`. As Steel's connect rows, a closure between two moving sides acts at the anchors'
        midpoint, which the builder records in the solver's ``_connect_at_midpoint`` for the force report, and the
        solver's ``_kinematic_dof_mask`` marks the DOFs of a kinematic chain.
        """
        model = solver.model
        lanes = solver._sparse_contact_lanes
        is_free_rigid = solver._is_free_rigid if solver._is_free_rigid is not None else solver._dummy_is_free_rigid
        free_factor = solver._Linv_by_size[6] if solver._has_free_rigid_bodies else solver._sparse_free_factor_dummy
        wp.launch(
            self.connect_kernel,
            dim=self.connect_owner_art.shape[0] * lanes,
            inputs=[
                self.connect_owner_art,
                self.connect_owner_start,
                self.connect_owner_closures,
                solver._connect_joint,
                solver._connect_body_p,
                solver._connect_body_c,
                solver._connect_parent_prescribed,
                solver._kinematic_dof_mask,
                model.joint_enabled,
                model.joint_X_p,
                model.joint_X_c,
                state_in.body_q,
                state_in.body_qd,
                model.body_com,
                solver._body_to_joint,
                model.joint_ancestor,
                model.joint_qd_start,
                solver._art_group_idx,
                solver._articulation_response_dof_count,
                is_free_rigid,
                solver._articulation_world_dof_offset,
                solver._articulation_dof_start,
                solver._articulation_origin,
                solver._sparse_body_dof_mask,
                state_aug._joint_S_s,
                self.art_group,
                self.art_slot,
                *self.response_table,
                free_factor,
                solver._v_hat,
            ],
            outputs=[
                solver._row_type,
                solver._row_parent,
                solver._row_mu,
                solver._phi,
                solver._target_velocity,
                solver._sparse_row_dof,
                solver._sparse_row_factor,
                solver._sparse_row_free_response,
                solver._sparse_row_incident,
                solver._diag,
                solver._connect_at_midpoint,
                mode,
                cursor,
            ],
            block_dim=128,
            device=self.device,
        )

    def count_internal_rows(
        self,
        solver,
        state_in,
        dt: float,
        *,
        control=None,
        limit_q_index: wp.array | None = None,
        state_aug=None,
    ) -> None:
        """Count each articulation's internal-unit rows: per family, then their sum (the COUNT passes).

        Clears the family counts in use (one span of ``SolverSteel._scene_family_rows``), runs the
        families' COUNT passes into their rows (:meth:`launch_internal_rows`) and, with more than one
        family in use, writes their sum into ``SolverSteel._scene_internal_rows``
        (:func:`sum_scene_family_rows`). With one family the sum is that family's row itself, which
        ``_scene_internal_rows`` then views, so the joint limits alone launch what they did before the
        internal unit. ``control`` is the step's, which the drive family reads, ``limit_q_index`` the
        coordinates the drive and joint-limit families read and ``state_aug`` the step's augmented state, which the
        connect family reads (:meth:`launch_internal_rows`). Fixed grids and no host read.
        """
        families = self.internal_families
        if not families:
            return
        family_rows = solver._scene_family_rows
        family_rows[families[0] : families[-1] + 1].zero_()
        self.launch_internal_rows(
            solver,
            state_in,
            ROW_SLOTS_COUNT,
            family_rows,
            dt=dt,
            control=control,
            limit_q_index=limit_q_index,
            state_aug=state_aug,
        )
        if len(families) > 1:
            wp.launch(
                sum_scene_family_rows,
                dim=self.side_count,
                inputs=[families[0], families[-1], family_rows],
                outputs=[solver._scene_internal_rows],
                device=self.device,
            )

    def count_late_rows(self, solver) -> None:
        """Count each side's late rows, an articulation's joint velocity-limit rows or a free body's velocity-limit
        rows (the velocity-limit family's COUNT pass).

        Clears the family's row of ``SolverSteel._scene_family_rows``, which ``_scene_late_rows`` views, and runs
        the family's COUNT pass into it (:meth:`launch_late_rows`). Fixed grids and no host read.
        """
        solver._scene_late_rows.zero_()
        self.launch_late_rows(solver, ROW_SLOTS_COUNT, solver._scene_late_rows)

    def launch_late_rows(self, solver, mode: int, cursor: wp.array) -> None:
        """Run the velocity-limit family's row builders in ``mode``, one launch per group that builds them.

        Steel's velocity-limit law
        (:func:`~newton._src.solvers.steel.scene_rows.get_sparse_joint_velocity_limit_rows_kernel`)
        on the step's predicted velocity ``v_hat`` (the activation gate), the sleeping articulations
        (``_constraint_art_active``) and, with the fused velocity clamp, the driven DOFs' marks in ``_drive_slot``
        (:meth:`mark_driven_dofs`), per factor group; then, with free-body velocity limits, Steel's free-body law
        (:attr:`free_late_kernel`,
        :func:`~newton._src.solvers.steel.scene_rows.build_sparse_free_velocity_limit_rows`) over the free
        bodies, on the model's ``rigid_body_max_linear_velocity`` and ``rigid_body_max_angular_velocity`` of this
        step. ``cursor`` (``[articulations]``): ``ROW_SLOTS_COUNT`` adds each articulation's rows to it and writes
        none; ``ROW_SLOTS_WRITE`` writes them from ``cursor[art]``, the late unit's first row, in Steel's slot order
        (lower then upper bound by DOF, by root DOF for a free body), and none at or past the row capacity.
        """
        model = solver.model
        for arrays in self.group_arrays:
            if arrays.late_kernel is None:
                continue
            indices = arrays.indices
            wp.launch(
                arrays.late_kernel,
                dim=len(arrays.group.arts) * 32,
                inputs=[
                    arrays.group_to_art,
                    solver._art_to_world,
                    solver._articulation_world_dof_offset,
                    solver._articulation_dof_start,
                    solver._constraint_art_active,
                    solver._joint_limit_q_index,
                    model.joint_velocity_limit,
                    float(solver._velocity_limit_activation_fraction),
                    solver._drive_slot,
                    int(solver._velocity_limits_fused),
                    indices.ancestor_mask,
                    indices.inverse_permutation,
                    indices.lookup,
                    arrays.Linv,
                    solver._v_hat,
                ],
                outputs=[
                    solver._slot_counter,
                    solver._row_type,
                    solver._row_parent,
                    solver._row_mu,
                    solver._phi,
                    solver._target_velocity,
                    solver._sparse_row_dof,
                    solver._sparse_row_factor,
                    solver._sparse_row_incident,
                    solver._diag,
                    mode,
                    cursor,
                ],
                block_dim=128,
                device=self.device,
            )
        if self.free_late_kernel is not None:
            wp.launch(
                self.free_late_kernel,
                dim=solver._free_rigid_body_count,
                inputs=[
                    solver._free_rigid_body_indices,
                    solver._body_to_articulation,
                    solver._art_to_world,
                    solver._is_free_rigid,
                    model.body_flags,
                    solver._constraint_art_active,
                    solver._articulation_world_dof_offset,
                    solver._articulation_dof_start,
                    solver._articulation_root_dof_start,
                    solver._art_group_idx,
                    solver._rigid_body_max_linear_velocity,
                    solver._rigid_body_max_angular_velocity,
                    solver._free_root_anchor,
                    float(solver._velocity_limit_activation_fraction),
                    solver._Linv_by_size[6],
                    solver._v_hat,
                ],
                outputs=[
                    solver._slot_counter,
                    solver._row_type,
                    solver._row_parent,
                    solver._row_mu,
                    solver._phi,
                    solver._target_velocity,
                    solver._sparse_row_dof,
                    solver._sparse_row_factor,
                    solver._sparse_row_free_response,
                    solver._sparse_row_incident,
                    solver._diag,
                    mode,
                    cursor,
                ],
                device=self.device,
            )

    def mark_driven_dofs(self, solver) -> None:
        """Mark the DOFs the drive family builds a row for in ``_drive_slot`` (the fused velocity clamp).

        Steel reserves its drive rows before stage 1 (``_begin_dense_rows``), so that its pre-solve velocity
        scaling and its velocity-limit rows skip the driven DOFs through ``drive_slot[dof] >= 0``; the scene places
        its rows later, by scans, so the step calls this where Steel reserves them instead. ``_drive_slot`` is filled
        with ``-1``, then each factor group that builds drive rows writes ``0`` at every DOF the drive family's
        candidate test passes for (:func:`~newton._src.solvers.steel.scene_rows.get_sparse_drive_marker_kernel`,
        the drive family's own candidate text): a marker, not a row index. One fill and one launch per such group.
        The mark precedes the row placement, so a DOF whose internal unit is then dropped past ``row_max`` keeps its
        ``0`` and gets neither a drive row nor velocity-limit rows, where Steel leaves a drive row past its world's
        budget at ``-1`` and so prescales the DOF and builds its velocity-limit rows; under overflow the scene already
        differs from Steel by its pooled budget.
        """
        model = solver.model
        solver._drive_slot.fill_(-1)
        kernel = get_sparse_drive_marker_kernel()
        for arrays in self.group_arrays:
            if FAMILY_DRIVE not in arrays.internal_kernels:
                continue
            size = arrays.group.size
            wp.launch(
                kernel,
                dim=len(arrays.group.arts) * size,
                inputs=[
                    arrays.group_to_art,
                    solver._articulation_dof_start,
                    size,
                    solver._joint_limit_q_index,
                    model.joint_target_ke,
                    model.joint_target_kd,
                ],
                outputs=[solver._drive_slot],
                device=self.device,
            )

    def anchor_points(self, solver) -> tuple:
        """The one-row response kernel's ``anchor_point0`` and ``anchor_point1``: friction patches' anchors
        (``FrictionPatches.point_a`` and ``point_b``, the tangent rows' lever arms), one-element placeholders that
        the kernel does not read without patches."""
        if self.features.patches:
            view = solver._friction_patches.view
            return view.point_a, view.point_b
        return self._no_anchor_points, self._no_anchor_points

    def _scan(self, counts: wp.array, prefix_sums: wp.array, length: wp.array) -> None:
        """Write the exclusive scan of ``counts`` into ``prefix_sums``, every one of ``length[0]`` entries.

        :func:`~newton._src.geometry.utils.scan_with_total` on the placement's one scratch: one launch, no allocation,
        so the placement can be captured; the scans run one at a time, each leaving the scratch zeroed. The scan of
        integer counts is exact, so it equals ``wp.utils.array_scan(counts, prefix_sums, inclusive=False)``.
        """
        scan_with_total(counts, prefix_sums, length, self._scan_total, self._scan_scratch)

    def assign_rows(self, solver) -> None:
        """Place the step's units' rows in row order; drop the units past ``row_max``.

        Scans the one-sided counts, lists the units in row order (:func:`scene_row_sequence`),
        scans their rows, assigns them (:func:`scene_assign_rows`), which also points each kept
        contact's ``contact_slot`` and each kept internal unit's cursor at its first row, and writes
        each side's one-sided rows (:func:`scene_side_rows`). With late units (:attr:`late_rows_in_use`) the
        one-sided counts are the phase-A ones (the units' less their late units, :func:`scene_phase_a_counts`),
        the late units are listed after the tail (:func:`scene_late_sequence`, from the scan of the late counts
        into :attr:`late_start`), each kept late unit's cursor (``SolverSteel._scene_late_cursor``) points at
        its first row, and each side's late rows are written (:func:`scene_side_late_rows`).
        """
        units = solver._scene_units
        device = self.device
        late = self.late_rows_in_use
        if late:
            wp.launch(
                scene_phase_a_counts,
                dim=self.side_count + 1,
                inputs=[self._unit_one_sided_counts, self._late_counts],
                outputs=[self._one_sided_counts],
                device=device,
            )
            self._scan(self._late_counts, self.late_start, self._side_scan_length)
        self._scan(self._one_sided_counts, self.one_sided_start, self._side_scan_length)
        wp.launch(
            scene_row_sequence,
            dim=self.unit_max + 1,
            inputs=[
                self.side_count,
                self.unit_max,
                self.color_total_slot,
                units.unit_count,
                units.unit_order,
                units.unit_side_a,
                units.unit_rows,
                units.side_start,
                self.one_sided_start,
                units.color_offsets,
                units.order_by_color,
            ],
            outputs=[self.unit_sequence, self.seq_rows],
            device=device,
        )
        if late:
            wp.launch(
                scene_late_sequence,
                dim=self.side_count,
                inputs=[
                    self.side_count,
                    self.color_total_slot,
                    units.unit_order,
                    units.unit_rows,
                    units.side_start,
                    self._unit_one_sided_counts,
                    self._late_counts,
                    self.one_sided_start,
                    units.color_offsets,
                    self.late_start,
                ],
                outputs=[self.unit_sequence, self.seq_rows],
                device=device,
            )
        self._scan(self.seq_rows, self.seq_row_start, self._unit_scan_length)
        self.side_dropped.zero_()
        # Internal and late units that are dropped, or that never got a unit, write no rows.
        solver._scene_row_cursor.fill_(self.row_max)
        if late:
            solver._scene_late_cursor.fill_(self.row_max)
        wp.launch(
            scene_assign_rows,
            dim=self.unit_max,
            inputs=[
                self.row_max,
                units.unit_count,
                units.unit_order,
                units.unit_contact,
                units.unit_side_a,
                self.unit_sequence,
                self.seq_row_start,
            ],
            outputs=[
                self.unit_row_start,
                solver._contact_slot,
                solver._contact_path,
                solver._scene_row_cursor,
                # Without late units no unit takes the late cursor, which the internal one then stands in for.
                solver._scene_late_cursor if late else solver._scene_row_cursor,
                self.side_dropped,
                self.rows_requested,
                solver._constraint_count,
            ],
            device=device,
        )
        wp.launch(
            scene_side_rows,
            dim=self.side_count,
            inputs=[self.one_sided_start, self.seq_row_start, solver._constraint_count],
            outputs=[self.side_row_begin, self.side_row_end],
            device=device,
        )
        if late:
            wp.launch(
                scene_side_late_rows,
                dim=self.side_count,
                inputs=[
                    self.side_count,
                    self.color_total_slot,
                    self.one_sided_start,
                    units.color_offsets,
                    self.late_start,
                    self.seq_row_start,
                    solver._constraint_count,
                ],
                outputs=[self.side_late_begin, self.side_late_end],
                device=device,
            )

    def prepare_rows(self, solver, dt: float, contacts=None) -> None:
        """Add the constraint force mixing, write the bias or rebound target, clear impulses, record free prefixes.

        With contact regularization it also writes each row's weight into :attr:`row_w`
        (:func:`prepare_scene_rows`). With drive rows it then writes their force-drive descriptors
        (:func:`scene_drive_descriptors`, one more launch of grid ``row_max``) from their diagonals,
        to which :func:`prepare_scene_rows` adds no ``cfm``. With the depenetration cap
        (``features.depenetration_cap``) and the step's ``contacts`` it then bounds the bias of the penetrating
        free-body contacts by the model's ``rigid_body_max_depenetration_velocity`` of this step
        (:func:`cap_scene_depenetration`, one more launch of grid ``rigid_contact_max``). With contact compliance
        (``features.compliance``) it then assembles the compliant contacts' rows on Steel's law
        (:meth:`prepare_compliance`). With a friction mode (``features.friction_mode``)
        it then flags each contact's update (:meth:`prepare_friction_modes`, one clear and one more launch of grid
        ``rigid_contact_max``).
        """
        wp.launch(
            prepare_scene_rows,
            dim=self.row_max,
            inputs=[
                solver._constraint_count,
                solver._row_type,
                solver._phi,
                solver._target_velocity,
                solver._row_rebound_velocity,
                solver._sparse_row_incident,
                solver._sparse_row_dof,
                solver._sparse_world_free_dof_mask,
                solver._baumgarte,
                solver._friction_anchor_beta,
                solver._contact_speculative_scale,
                self.contact_w,
                solver._cfm,
                dt,
            ],
            outputs=[solver._rhs, solver._diag, solver._impulses, self.row_free_prefix, self._row_w_arg],
            device=self.device,
        )
        if self.features.drives:
            wp.launch(
                scene_drive_descriptors,
                dim=self.row_max,
                inputs=[
                    solver._constraint_count,
                    solver._row_type,
                    solver._diag,
                    solver._target_velocity,
                    solver._drive_stiffness,
                    solver._drive_damping,
                    solver._drive_geom_error,
                    solver._drive_max_force,
                    dt,
                ],
                outputs=list(self.drive_descriptors),
                device=self.device,
            )
        if self.features.depenetration_cap and contacts is not None:
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
                    dt,
                ],
                outputs=[solver._rhs],
                device=self.device,
            )
        if self.features.compliance:
            self.prepare_compliance(solver, dt, contacts)
        if self.row_friction_mode is not None:
            self.prepare_friction_modes(solver, contacts)

    def prepare_friction_modes(self, solver, contacts) -> None:
        """Flag the update each contact's tangent pair takes under a friction mode (``features.friction_mode``).

        Clears :attr:`row_friction_mode`, then one launch of :func:`prepare_scene_friction_modes` over
        ``rigid_contact_max`` contacts writes each contact's update on its first tangent row:
        :data:`SCENE_FRICTION_FREE` on Steel's free-body path, :data:`SCENE_FRICTION_DENSE` for an articulated
        contact (:data:`SCENE_FRICTION_FIXED_NORMAL` under ``"coulomb_newton"``, which Steel's dense rows do not take).
        Fixed grid, no host read, so the step can be captured. Without contacts only the clear runs.
        """
        self.row_friction_mode.zero_()
        if contacts is None:
            return
        articulated = (
            SCENE_FRICTION_FIXED_NORMAL if self.features.friction_mode == "coulomb_newton" else SCENE_FRICTION_DENSE
        )
        is_free_rigid = solver._is_free_rigid if solver._is_free_rigid is not None else solver._dummy_is_free_rigid
        wp.launch(
            prepare_scene_friction_modes,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                solver._contact_slot,
                solver._contact_slots_needed,
                solver._contact_art_a,
                solver._contact_art_b,
                solver._articulation_response_dof_count,
                is_free_rigid,
                articulated,
            ],
            outputs=[self.row_friction_mode],
            device=self.device,
        )

    def prepare_compliance(self, solver, dt: float, contacts) -> None:
        """Assemble the compliant contacts' rows on Steel's device law (``contact_compliance``), after the cap.

        :func:`~newton._src.solvers.steel.scene_compliance.prepare_scene_compliance` on the scene's row store: it
        clears :attr:`row_gamma` and the step's counts (the solver's ``ComplianceBuffers.counts``), latches the step's
        errors with Steel's codes (rows lost past ``row_max`` or units past ``scene_unit_max`` latch the row overflow),
        writes each compliant contact's ``gamma``, ``rhs`` and diagonal and weights its friction rows' coefficients.
        As Steel's ``contact_compliance.prepare_rows``, a captured step latches into ``ComplianceBuffers.status``, which
        the solver's status reports read, and an eager step into ``eager_status``, which is checked at once: the first
        latched error raises Steel's exception and the word is cleared. Fixed grids, no host read under capture.
        Without contacts only the clears run.
        """
        buffers = solver._compliance
        if contacts is None:
            self.row_gamma.zero_()
            buffers.counts.zero_()
            return
        capturing = self.device.is_capturing
        status = buffers.status if capturing else buffers.eager_status
        rows = SceneComplianceRows(
            count=solver._constraint_count,
            requested=self.rows_requested,
            unit_overflow=solver._scene_units.unit_overflow,
            row_type=solver._row_type,
            row_parent=solver._row_parent,
            phi=solver._phi,
            target_velocity=solver._target_velocity,
            diag=solver._diag,
            rhs=solver._rhs,
            row_mu=solver._row_mu,
        )
        prepare_scene_compliance(solver, contacts, rows, self.row_gamma, status, dt=dt)
        if not capturing:
            # An eager step raises at once; the next step starts from a clear status.
            contact_compliance.validate(status, clear=True)

    def prepare_torsion(self, solver, state_in, state_aug, contacts) -> None:
        """Write the step's spin rows after every unit's rows (contact torsion; Steel's ``prepare_torsion_rows``).

        Steel's preparation runs after its rows are built (``contact_torsion.prepare_torsion_rows``, by its device
        path ``contact_torsion_device.DeviceTorsionPreparation.prepare``); the scene's runs after :meth:`prepare_rows`,
        so the friction coefficients it reads are the rows' after compliance:

        1. :func:`~newton._src.solvers.steel.scene_torsion_device.scene_torsion_routing` routes the contacts as Steel's
           grouping sees them (its free-body contacts off path 0, which Steel's grouping does not see), gathers the
           anchors' friction coefficients from the rows and latches Steel's refusals of the step in
           ``torsion_step_status``: an overflowing contact input, a hydroelastic contact, rows or units lost before the
           spin rows.
        2. :func:`~newton._src.solvers.steel.scene_torsion_device.scene_torsion_eligibility` (on the shapes' torsion
           support as Steel marks it, ``torsion_shapes``) and
           :func:`~newton._src.solvers.steel.scene_torsion_device.group_scene_torsion` group the contacts as Steel
           does: under friction patches by region, keyed by the contact's model world and its patch owner as Steel
           keys them (``patches`` and ``patch_owner`` derive from one flag, ``_friction_anchors_enabled``). The
           anchors' coefficients are read after ``link_patch_rows`` divided them, as Steel's grouping reads them, and
           the C1 rescale bounds a group by its members' pooled load, which a region's normal-only members share.
        3. ``scene_torsion_keys`` and a radix sort order the groups' spin rows: the one-sided ones side-major, the
           two-sided ones by the color of their representative contact unit, the order phase C sweeps them in.
        4. ``scene_torsion_counts`` appends them after the kept rows, those that fit in ``row_max``;
           ``scene_torsion_place`` gives each its row and writes the sides' and colors' ranges.
        5. ``get_scene_torsion_rows_kernel`` writes each kept spin row (``J = (0, n)`` through the contact
           response, no rebound) and ``scene_torsion_finish`` finishes it as :func:`prepare_scene_rows` would and lists
           its members (``torsion_member_rows``, ``torsion_group``).

        A step that Steel would refuse builds no spin row and latches its cause in :attr:`torsion_status` (the bits of
        ``scene_torsion_device.SCENE_TORSION_MESSAGES``), where Steel raises; spin rows dropped past the capacity keep
        the prefix that fits and latch ``SCENE_TORSION_CAPACITY`` and their worlds' ``constraint_overflow``. Fixed
        grids, no host read and no allocation: capturable.
        """
        device = self.device
        model = solver.model
        groups = self.torsion_groups
        capacity = groups.capacity
        units = solver._scene_units
        is_free_rigid = solver._is_free_rigid if solver._is_free_rigid is not None else solver._dummy_is_free_rigid
        self.torsion_group.fill_(-1)
        self.side_torsion_begin.zero_()
        self.side_torsion_end.zero_()
        self.torsion_step_status.zero_()
        # Friction patches: the eligibility's patches and the grouping's patch_owner derive from this one flag.
        patches = bool(solver._friction_anchors_enabled)
        patch_owner = solver._friction_patches.current.owner if patches else None
        if contacts is not None and contacts.rigid_contact_max > 0:
            stiffness = contacts.rigid_contact_stiffness
            wp.launch(
                scene_torsion_routing,
                dim=contacts.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_max,
                    solver._contact_path,
                    solver._contact_slot,
                    solver._contact_slots_needed,
                    solver._contact_art_a,
                    solver._contact_art_b,
                    solver._articulation_response_dof_count,
                    is_free_rigid,
                    stiffness if stiffness is not None else self._no_stiffness,
                    int(stiffness is not None),
                    solver._row_mu,
                    self.rows_requested,
                    solver._constraint_count,
                    units.unit_overflow,
                    solver._articulation_model_world,
                    max(int(model.world_count), 1),
                ],
                outputs=[self.torsion_path, self.torsion_friction, self.torsion_world, self.torsion_step_status],
                device=device,
            )
            scene_torsion_eligibility(
                contacts,
                model.shape_body,
                body_q=state_in.body_q,
                shape_supported=self.torsion_shapes.supported,
                shape_radius=solver._contact_torsion_shape_radius,
                contact_path=self.torsion_path,
                contact_slot=solver._contact_slot,
                contact_slots_needed=solver._contact_slots_needed,
                patches=patches,
                out=self.torsion_eligible,
                device=device,
            )
            group_scene_torsion(
                contacts,
                model.shape_body,
                self.torsion_eligible,
                body_q=state_in.body_q,
                shape_radius=solver._contact_torsion_shape_radius,
                contact_slots_needed=solver._contact_slots_needed,
                friction=self.torsion_friction,
                patch_owner=patch_owner,
                contact_world=self.torsion_world if patches else None,
                out=groups,
                device=device,
            )
        else:
            groups.group_count.zero_()
        wp.launch(
            scene_torsion_keys,
            dim=capacity,
            inputs=[
                groups.group_count,
                groups.member_start,
                groups.members,
                solver._contact_path,
                solver._contact_art_a,
                solver._contact_art_b,
                solver._articulation_response_dof_count,
                units.contact_entry,
                units._unit_slot,
                units.unit_color,
                self.torsion_step_status,
            ],
            outputs=[self.torsion_keys, self.torsion_order],
            device=device,
        )
        if capacity > 0:
            wp.utils.radix_sort_pairs(self.torsion_keys, self.torsion_order, capacity)
        wp.launch(
            scene_torsion_counts,
            dim=1,
            inputs=[self.row_max, groups.group_count],
            outputs=[
                self.torsion_step_status,
                self.rows_requested,
                solver._constraint_count,
                self.torsion_status,
                self.torsion_counts,
            ],
            device=device,
        )
        color_slots = self.torsion_color_start.shape[0]
        wp.launch(
            scene_torsion_place,
            dim=max(capacity, color_slots),
            inputs=[
                color_slots,
                self.torsion_keys,
                self.torsion_order,
                self.torsion_counts,
                groups.member_start,
                groups.members,
                solver._contact_path,
                solver._contact_art_a,
                solver._contact_art_b,
                solver._articulation_response_dof_count,
                solver._articulation_model_world,
            ],
            outputs=[
                self.side_torsion_begin,
                self.side_torsion_end,
                self.torsion_color_start,
                solver.constraint_overflow,
            ],
            device=device,
        )
        if capacity == 0 or contacts is None:
            # Without contacts no group, so no spin row to write.
            return
        lanes = self.torsion_lanes
        workers = max(1, min(capacity, SCENE_TORSION_ROW_WORKERS))
        wp.launch(
            self.torsion_rows_kernel,
            dim=workers * lanes,
            inputs=[
                self.torsion_counts,
                workers,
                self.torsion_order,
                groups.member_start,
                groups.members,
                groups.lead,
                groups.coefficient,
                groups.radius,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                solver._contact_slot,
                solver._contact_art_a,
                solver._contact_art_b,
                model.shape_body,
                solver._body_to_articulation,
                solver._prescribed_articulation,
                state_aug._body_v_s,
                solver._art_group_idx,
                solver._articulation_response_dof_count,
                is_free_rigid,
                solver._articulation_world_dof_offset,
                solver._articulation_dof_start,
                solver._sparse_body_dof_mask,
                state_aug._joint_S_s,
                self.art_group,
                self.art_slot,
                *self.response_table,
                solver._Linv_by_size[6] if solver._has_free_rigid_bodies else solver._sparse_free_factor_dummy,
                solver._v_hat,
            ],
            outputs=[
                solver._row_type,
                solver._row_parent,
                solver._row_mu,
                solver._phi,
                solver._target_velocity,
                solver._row_rebound_velocity,
                self.torsion_radius,
                self.torsion_member_range,
                self.torsion_shape_pair,
                solver._sparse_row_dof,
                solver._sparse_row_factor,
                solver._sparse_row_free_response,
                solver._sparse_row_incident,
                solver._diag,
            ],
            block_dim=128,
            device=device,
        )
        wp.launch(
            scene_torsion_finish,
            dim=capacity,
            inputs=[
                self.torsion_counts,
                self.torsion_order,
                groups.member_start,
                groups.members,
                solver._contact_slot,
                solver._contact_slots_needed,
                solver._sparse_row_dof,
                solver._sparse_world_free_dof_mask,
                solver._target_velocity,
                solver._cfm,
                self.contact_w,
            ],
            outputs=[
                solver._rhs,
                solver._diag,
                solver._impulses,
                self.row_free_prefix,
                self._row_w_arg,
                self.torsion_group,
                self.torsion_member_rows,
            ],
            device=device,
        )

    def clear_torsion(self) -> None:
        """Clear the latched torsion status (a solver reset; contact torsion only)."""
        if self.torsion_groups is not None:
            self.torsion_status.zero_()

    def torsion_statistics(self, solver) -> dict:
        """The last step's spin rows in Steel's ``_torsion_stats`` form (contact torsion; reads the device).

        ``{"rows": n, "groups": [...]}`` with one entry per kept spin row in row order: ``world`` (0, the scene's one
        solve world), ``row``, ``normal_rows`` and ``anchor_rows`` (its members' normal rows, and those of the members
        with their tangent pair), ``shape_pair`` (its lead's shapes), ``effective_radius_m``, ``mu`` and
        ``gap_max_m`` (the largest gap of its touching members), Steel's keys and values, from the step's device arrays
        (a snapshot of the last step until the next one, the contacts not read). Call it outside capture.
        """
        counts = self.torsion_counts.numpy()
        base, kept = int(counts[0]), int(counts[2])
        groups = self.torsion_groups
        order = self.torsion_order.numpy()
        member_start = groups.member_start.numpy()
        members = groups.members.numpy()
        shape_pair = self.torsion_shape_pair.numpy()
        member_rows = self.torsion_member_rows.numpy()
        gap = groups._work.gap.numpy()
        radius = self.torsion_radius.numpy()[0]
        mu = solver._row_mu.numpy()[0]
        stats = []
        for k in range(kept):
            g = int(order[k])
            row = base + k
            span = slice(int(member_start[g]), int(member_start[g + 1]))
            group_members = members[span]
            encoded = member_rows[span]
            gaps = gap[group_members]
            touching = gaps[gaps <= _TOUCH_TOLERANCE]
            stats.append(
                {
                    "world": 0,
                    "row": row,
                    "normal_rows": (encoded >> 1).tolist(),
                    "anchor_rows": (encoded[(encoded & 1) != 0] >> 1).tolist(),
                    "shape_pair": shape_pair[row].tolist(),
                    "effective_radius_m": float(radius[row]),
                    "mu": float(mu[row]),
                    "gap_max_m": float(np.max(touching)) if touching.size else 0.0,
                }
            )
        return {"rows": len(stats), "groups": stats}

    def report_status(self, solver, contacts) -> None:
        """Latch :attr:`SolverSteel.constraint_overflow` for this step's losses and warn once."""
        units = solver._scene_units
        has_contacts = contacts is not None and contacts.rigid_contact_max > 0
        entries = solver.constraint_overflow.shape[0]
        wp.launch(
            scene_row_status,
            dim=max(self.unit_max, entries),
            inputs=[
                self.row_max,
                units.unit_count,
                units.unit_overflow,
                units.unit_order,
                units.unit_side_a,
                units.unit_side_b,
                self.unit_sequence,
                self.seq_row_start,
                self.rows_requested,
                solver._constraint_count,
                contacts.rigid_contact_count if has_contacts else solver._dummy_contact_count,
                contacts._reduction_overflow if has_contacts else solver._dummy_contact_count,
                contacts.rigid_contact_max if has_contacts else 0,
                solver._articulation_model_world,
                int(solver._warn_constraint_overflow),
            ],
            outputs=[solver._row_overflow_warning_emitted, self.overflow, solver.constraint_overflow],
            device=self.device,
        )

    def accumulate_row_watermarks(self, solver, contacts) -> None:
        """Fold the step's rows, per model world, into Steel's row high-water marks (``row_watermark``).

        Steel counts each solve world's rows (``accumulate_row_watermarks``). The scene's one solve
        world holds every model world's rows, so :func:`count_scene_world_rows` counts each unit's
        rows in the model world of its ``side_a`` (a global side takes its other side's) and Steel's
        row family, and :func:`accumulate_scene_row_watermarks` folds the counts into Steel's marks;
        the contact count's mark is Steel's ``accumulate_contact_watermark``. Units past
        ``scene_unit_max`` have no rows and are not counted. With contact torsion
        :func:`count_scene_torsion_rows` counts each spin row too, in its representative contact unit's model world,
        as Steel counts its torsion rows with its dense rows. Fixed grids, integer atomics and no host
        read, so a captured step accumulates as an eager one.
        """
        units = solver._scene_units
        device = self.device
        is_free_rigid = solver._is_free_rigid if solver._is_free_rigid is not None else solver._dummy_is_free_rigid
        wp.launch(
            count_scene_world_rows,
            dim=self.unit_max,
            inputs=[
                self.row_max,
                int(solver._has_free_rigid_bodies),
                units.unit_count,
                units.unit_order,
                units.unit_contact,
                units.unit_side_a,
                units.unit_side_b,
                self.unit_sequence,
                self.seq_row_start,
                solver._articulation_model_world,
                is_free_rigid,
            ],
            outputs=[self.world_rows],
            device=device,
        )
        if self.features.torsion:
            groups = self.torsion_groups
            wp.launch(
                count_scene_torsion_rows,
                dim=groups.capacity,
                inputs=[
                    self.torsion_counts,
                    self.torsion_order,
                    groups.member_start,
                    groups.members,
                    solver._contact_path,
                    solver._contact_art_a,
                    solver._contact_art_b,
                    solver._articulation_response_dof_count,
                    solver._articulation_model_world,
                ],
                outputs=[self.world_rows],
                device=device,
            )
        wp.launch(
            accumulate_scene_row_watermarks,
            dim=(self.world_rows.shape[0], self.world_rows.shape[2]),
            outputs=[self.world_rows, solver._row_watermarks],
            device=device,
        )
        if contacts is not None:
            wp.launch(
                accumulate_contact_watermark,
                dim=1,
                inputs=[contacts.rigid_contact_count],
                outputs=[solver._row_watermarks],
                device=device,
            )

    def prepare_velocity_rows(self, solver, dt: float) -> None:
        """Write the velocity-only iterations' right-hand side into ``SolverSteel._rhs_unbiased`` (Steel's
        ``_compute_velocity_pass_rhs``), after the position solve.

        One launch of :func:`prepare_scene_velocity_rows` over ``row_max`` rows: Steel's velocity-pass law on each
        row, with the row's ``J v`` of the position solve formed in factor coordinates from the position solve's
        ``delta``, summed in the one-thread row loop's order (``scene_rows._scene_row_dot``, by ``scene_sum_order``;
        8 lanes with serial sums add in another order). With contact torsion one more launch concludes each spin row's
        admission (``scene_torsion_device.prepare_scene_velocity_torsion``, Steel's ``prepare_torsion_velocity_pass``):
        a spin row whose members all part after the position solve is retired, its bound 0 in the velocity
        iterations. Fixed grids, no host read.
        """
        wp.launch(
            prepare_scene_velocity_rows,
            dim=self.row_max,
            inputs=[
                solver._constraint_count,
                solver._row_type,
                solver._phi,
                solver._target_velocity,
                solver._row_rebound_velocity,
                solver._sparse_row_incident,
                solver._sparse_row_dof,
                solver._sparse_row_factor,
                solver._sparse_factor_velocity_delta,
                int(self.sum_order == "steel"),
                dt,
            ],
            outputs=[solver._rhs_unbiased],
            device=self.device,
        )
        if self.features.torsion:
            wp.launch(
                prepare_scene_velocity_torsion,
                dim=self.torsion_groups.capacity,
                inputs=[
                    self.torsion_counts,
                    self.torsion_member_range,
                    self.torsion_member_rows,
                    solver._phi,
                    solver._target_velocity,
                    solver._sparse_row_incident,
                    solver._sparse_row_dof,
                    solver._sparse_row_factor,
                    solver._sparse_factor_velocity_delta,
                    int(self.sum_order == "steel"),
                    dt,
                ],
                outputs=[self.torsion_group],
                device=self.device,
            )

    def solve(self, solver, *, rhs: wp.array | None = None, reset: bool = True, seed: bool = False) -> None:
        """Sweep the scene's rows and decode the velocity change into ``v_out`` (``_solve_matrix_free``).

        1. With ``reset`` (the position solve): ``v_out = v_hat``, ``delta = 0``, with ``seed`` (Steel's contact warm
           start) the seeded impulses applied to ``delta`` side by side (``scene_warmstart.seed``, Steel's install),
           and the step's row ranges and colors (:func:`scene_row_ranges`).
        2. ``scene_sweep="serial"``: one thread walks every row, ``iterations`` sweeps with the
           stationary exit, each sweep in the sequence of ``phase_order``. ``"phases"``: the
           isolated sides' launch, then under ``wp.capture_if(any_coupled)`` each iteration's
           coupled-side sweep and phase B (``scene_color_loop``) in ``phase_order``, whose while
           loop is a ``wp.capture_while`` nested in that branch.
        3. Steel's decode: ``apply_sparse_factor_velocity`` for the articulated group,
           ``apply_sparse_free_velocity`` for free bodies.

        ``rhs`` replaces the rows' right-hand side ``SolverSteel._rhs`` (``None``) in the sweeps. ``reset=False`` is
        Steel's velocity pass, its second launch of the same solve (``SolverSteel._solve_matrix_free``): it starts from
        the position solve's ``delta``, impulses (which only the row preparation clears), row ranges and colors, sweeps
        its rows again, ``velocity_iterations`` times with ``rhs``, then decodes again, which rewrites ``v_out`` from
        ``v_hat`` and the refined ``delta``. As in Steel's launch the pass weighs no row (a regularized sweep reads
        weight 1 on every row), solves friction from its first sweep (``friction_start = 0``, iteration 0 first), leaves
        the drive rows as they are under ``velocity_drive_mode="freeze"`` (read at the call) and runs each sweep's phase
        C and the fused clamp; its isolated sides take the stationary exit in their own launch. With ``reset`` and no
        ``rhs`` the default kernels' launches are the ones before the velocity pass existed, argument for argument; the
        featured kernels take ``freeze_drives = 0``.

        With no sweep (``iterations=0``, or ``velocity_iterations=0`` and ``reset=False``) nothing sweeps, as in
        Steel: with ``reset`` the rows stay built and their impulses zero (the seeds with ``seed``), and the decode of
        the zero (seeded) ``delta`` writes ``v_hat`` (plus the seeds' velocity).

        Fixed grids, no host read inside a capture and no allocation, so a step can be captured.
        Run eagerly, ``wp.capture_if`` reads ``any_coupled`` back once per pass and
        ``wp.capture_while`` reads its condition back once per color round.
        """
        units = solver._scene_units
        device = self.device
        rhs = solver._rhs if rhs is None else rhs
        if reset:
            self._pass = _ScenePass(rhs, int(solver._iterations), self.friction_start, self._row_w_arg, 0)
            delta = solver._sparse_factor_velocity_delta
            wp.copy(solver._v_out, solver._v_hat)
            delta.zero_()
            if seed:
                self._seed(solver)
            wp.launch(
                scene_row_ranges,
                dim=1,
                inputs=[
                    self.side_count,
                    self.color_cap,
                    self.block_color_max,
                    solver._constraint_count,
                    self.one_sided_start,
                    self.seq_row_start,
                    units.color_count_per,
                    units.color_offsets,
                ],
                outputs=[self.row_ranges, self.used_colors, self.big_colors],
                device=device,
            )
        else:
            freeze = int(self.features.drives and solver._velocity_drive_mode == "freeze")
            self._pass = _ScenePass(rhs, self.velocity_iterations, 0, self._velocity_row_w_arg, freeze)
        if self._pass.iterations > 0:
            self._sweep(solver)
        self._decode(solver)

    def _seed(self, solver) -> None:
        """Apply the seeded impulses to ``delta``, one thread per side (``scene_warmstart.seed``)."""
        seed(self, solver)

    def keep_gather_mu(self, solver) -> None:
        """Keep the rows' friction coefficients before the patch link for the warm start's gather (``gather_mu``;
        :func:`~newton._src.solvers.steel.scene_warmstart.keep_gather_mu`)."""
        keep_gather_mu(self, solver)

    def finish_gather_mu(self, solver, contacts) -> None:
        """After the patch link, give the contacts on Steel's dense path their divided coefficients in ``gather_mu``
        (:func:`~newton._src.solvers.steel.scene_warmstart.finish_gather_mu`)."""
        finish_gather_mu(self, solver, contacts)

    def clear_warmstart(self, solver, world_mask: wp.array | None) -> None:
        """Discard the warm start's carried impulses of the worlds ``world_mask`` selects (all for ``None``).

        :func:`~newton._src.solvers.steel.scene_warmstart.clear_warmstart`: contact by contact by its unit's model
        world, the rows themselves only when every world is selected. No host read, so a reset with a live mask can be
        captured.
        """
        clear_warmstart(self, solver, world_mask)

    def _sweep(self, solver) -> None:
        """Every iteration of the sweep that ``scene_sweep`` and ``phase_order`` select (step 2 of :meth:`solve`)."""
        if self.sweep == "serial" and self.order_kernel is not None:
            self._walk_in_order(solver)
        elif self.sweep == "serial":
            self._walk(solver, RANGE_ALL, self._pass.iterations, stationary_exit=True, iteration=0)
        else:
            self._sweep_sides(solver, coupled_pass=SCENE_PASS_ISOLATED, iteration=0)
            wp.capture_if(self.any_coupled, on_true=partial(self._sweep_coupled, solver))

    def _row_inputs(self, solver) -> list:
        """The sweep kernels' row arguments (the pass's ``rhs``), relaxation, delta and impulses."""
        return [
            solver._sparse_row_dof,
            solver._sparse_row_factor,
            solver._sparse_row_free_response,
            self.row_free_prefix,
            self._pass.rhs,
            solver._sparse_row_incident,
            solver._diag,
            solver._row_type,
            solver._row_parent,
            solver._row_mu,
            solver._relaxation,
            solver._sparse_factor_velocity_delta,
            solver._impulses,
        ]

    def _feature_inputs(self, iteration: int) -> list:
        """The sweep kernels' feature arguments of a launch that starts at ``iteration``, after the row inputs.

        With every feature off, none: the default kernels take no feature argument. With any feature on,
        every feature's argument of the featured definition, in order, a feature that is off taking its
        placeholder: the pass's ``friction_start`` less ``iteration``, the launch's first sweep that solves the
        tangent pairs (``0`` without friction position iterations), then the pass's weights, :attr:`row_w` in
        the position solve and ones in the velocity pass (an empty ``[1, 0]`` array without regularization),
        :attr:`row_bound` (an empty ``[1, 0]`` array without joint friction), the four :attr:`drive_descriptors`
        (empty ``[1, 0]`` arrays without drive rows), :attr:`side_late_begin` and :attr:`side_late_end` (empty
        ``[0]`` arrays without joint velocity limits), Steel's ``_drive_vel_limit`` (an empty ``[1, 0]`` array
        without the fused velocity clamp), the pass's ``freeze_drives`` (``0`` without drive rows),
        :attr:`row_gamma` (an empty ``[1, 0]`` array without contact compliance), contact torsion's seven arrays
        (:meth:`_allocate_torsion`; empty arrays without it) and :attr:`row_friction_mode` (an empty ``[1, 0]`` array
        without a friction mode).
        """
        if not self._featured:
            return []
        features = self.features
        sweep = self._pass
        return [
            sweep.friction_start - iteration if features.friction_position_iterations else 0,
            sweep.row_w,
            self._row_bound_arg,
            *self._drive_args,
            *self._late_args,
            self._drive_vel_limit_arg,
            sweep.freeze_drives,
            self._row_gamma_arg,
            *self._torsion_args,
            self._friction_mode_arg,
        ]

    def _sweep_sides(self, solver, coupled_pass: int, iteration: int) -> None:
        """Phase A: all iterations of the isolated sides (``0``), or one sweep of the coupled sides (``1``).

        With late rows the isolated sides run phase C after phase A in each sweep, and ``2`` (:data:`SCENE_PASS_LATE`)
        gives the coupled sides one phase-C sweep. ``iteration`` is the first iteration of the launch: 0 for the
        isolated sides, the coupled pass's iteration for one sweep of the coupled sides.
        """
        wp.launch(
            self.sides_kernel,
            dim=self.side_count * SCENE_SIDE_LANES[self.side_sweep],
            inputs=[
                self.side_count,
                coupled_pass,
                self._pass.iterations,
                self.side_row_begin,
                self.side_row_end,
                solver._scene_units.side_coupled,
                *self._row_inputs(solver),
                *self._feature_inputs(iteration),
            ],
            block_dim=SCENE_SWEEP_BLOCK_DIM,
            device=self.device,
        )

    def _walk(self, solver, range_index: int, sweeps: int, *, stationary_exit: bool, iteration: int) -> None:
        """One thread sweeps ``row_ranges[range_index]`` in row order ``sweeps`` times, from ``iteration``."""
        wp.launch(
            self.serial_kernel,
            dim=1,
            inputs=[
                range_index,
                sweeps,
                int(stationary_exit),
                self.row_ranges,
                *self._row_inputs(solver),
                *self._feature_inputs(iteration),
            ],
            block_dim=SCENE_SWEEP_BLOCK_DIM,
            device=self.device,
        )

    def _walk_in_order(self, solver) -> None:
        """The oracle of a phase order other than ``"AB"``: one thread walks every sweep in the order's sequence."""
        wp.launch(
            self.order_kernel,
            dim=1,
            inputs=[
                self.side_count,
                self._pass.iterations,
                self.side_row_begin,
                self.side_row_end,
                solver._scene_units.side_coupled,
                self.row_ranges,
                *self._row_inputs(solver),
                *self._feature_inputs(0),
            ],
            block_dim=SCENE_SWEEP_BLOCK_DIM,
            device=self.device,
        )

    def _sweep_coupled(self, solver) -> None:
        """Each iteration of the coupled sides: phase A, their one-sided sweep, and phase B, the two-sided units.

        The phases run in ``phase_order``: ``"BA"`` (the default), ``"AB"``, or ``"ABA"``, which
        gives the coupled sides a second sweep after phase B. With late rows or contact torsion each iteration then
        ends with phase C (:meth:`_sweep_late`): B, A, C under ``"BA"``, A, B, C under ``"AB"``, A, B, A, C under
        ``"ABA"``.
        """
        for iteration in range(self._pass.iterations):
            for phase in self.phase_order:
                if phase == "A":
                    self._sweep_sides(solver, coupled_pass=SCENE_PASS_COUPLED, iteration=iteration)
                else:
                    self._sweep_two_sided(solver, iteration)
            if _has_phase_c(self.features):
                self._sweep_late(solver)

    def _sweep_late(self, solver) -> None:
        """Phase C of the coupled sides: one sweep of their late rows, after the iteration's whole coupled pass.

        The sides kernel's late pass (:data:`SCENE_PASS_LATE`): each coupled side sweeps its late rows
        ``[side_late_begin, side_late_end)``, the velocity limits, and with the fused clamp then clamps its drive rows,
        so the velocity limits have the last word of the iteration (Steel's order in its matrix-free sweep,
        ``solver_steel._get_pgs_solve_mf_gs_kernel``). Phase C holds no tangent pair, so the launch's first iteration
        does not matter to it: it passes ``0``, as the isolated sides' launch does.

        With contact torsion phase C starts with C1, the spin rows, as Steel sweeps its torsion rows right after its
        dense rows and before its velocity limits (``solver_steel._get_pgs_solve_mf_gs_kernel``): the torsion pass
        (:data:`SCENE_PASS_TORSION`) sweeps each coupled side's one-sided spin rows, then one block sweeps the
        two-sided ones color by color (:func:`get_sweep_torsion_kernel`), and the late pass follows (C2).
        """
        if self.features.torsion:
            self._sweep_sides(solver, coupled_pass=SCENE_PASS_TORSION, iteration=0)
            wp.launch(
                self.torsion_kernel,
                dim=SCENE_BLOCK_THREADS,
                inputs=[*self._row_inputs(solver), *self._feature_inputs(0)],
                block_dim=SCENE_BLOCK_THREADS,
                device=self.device,
            )
        self._sweep_sides(solver, coupled_pass=SCENE_PASS_LATE, iteration=0)

    def _sweep_two_sided(self, solver, iteration: int) -> None:
        """One sweep of the two-sided units in ``iteration``: the colors in order, then the tail.

        The schedule is ``scene_color_loop``'s.
        """
        loop = self.color_loop
        if loop == "serial":
            self._walk(solver, RANGE_TWO_SIDED, 1, stationary_exit=False, iteration=iteration)
            return
        if loop == "unroll":
            for color in range(self.color_cap):
                self._sweep_color(solver, color, iteration)
        elif loop != "block":
            wp.launch(
                reset_color_loop,
                dim=1,
                inputs=[self.big_colors],
                outputs=[self.color_cursor, self.color_keep_going],
                device=self.device,
            )
            wp.capture_while(self.color_keep_going, partial(self._color_round, solver, iteration))
        if loop in ("while", "unroll"):
            self._walk(solver, RANGE_TAIL, 1, stationary_exit=False, iteration=iteration)
        else:
            self._sweep_block(solver, iteration)

    def _sweep_color(self, solver, color: int, iteration: int) -> None:
        """One color's units at once in ``iteration``: ``color``, or with ``-1`` the while loop's ``color_cursor``."""
        wp.launch(
            self.color_kernel,
            dim=self.color_workers * SCENE_UNIT_LANES[self.unit_sweep],
            inputs=[
                color,
                self.color_workers,
                self.side_count,
                self.color_cursor,
                solver._scene_units.color_offsets,
                self.one_sided_start,
                self.seq_row_start,
                solver._constraint_count,
                *self._row_inputs(solver),
                *self._feature_inputs(iteration),
            ],
            block_dim=SCENE_SWEEP_BLOCK_DIM,
            device=self.device,
        )

    def _color_round(self, solver, iteration: int) -> None:
        """The while loop's body: the cursor's color, then the next color."""
        self._sweep_color(solver, -1, iteration)
        wp.launch(
            advance_color,
            dim=1,
            inputs=[self.big_colors],
            outputs=[self.color_cursor, self.color_keep_going],
            device=self.device,
        )

    def _sweep_block(self, solver, iteration: int) -> None:
        """The colors after the while loop's, then the tail, in one block, in ``iteration``."""
        wp.launch(
            self.block_kernel,
            dim=SCENE_BLOCK_THREADS,
            inputs=[
                self.side_count,
                self.big_colors,
                self.used_colors,
                self.row_ranges,
                solver._scene_units.color_offsets,
                self.one_sided_start,
                self.seq_row_start,
                solver._constraint_count,
                *self._row_inputs(solver),
                *self._feature_inputs(iteration),
            ],
            block_dim=SCENE_BLOCK_THREADS,
            device=self.device,
        )

    def _decode(self, solver) -> None:
        """Steel's decode, scene-wide: ``v_out = v_hat + P^T L^-T delta`` per articulation, ``+ delta`` per free body.

        One launch per factor group, then one for the free group (``free_group_to_art``).
        """
        device = self.device
        delta = solver._sparse_factor_velocity_delta
        for arrays in self.group_arrays:
            wp.launch(
                apply_sparse_factor_velocity,
                dim=(len(arrays.group.arts), arrays.group.size),
                inputs=[
                    arrays.group_to_art,
                    solver._art_to_world,
                    solver._articulation_world_dof_offset,
                    solver._articulation_dof_start,
                    arrays.indices.permutation,
                    arrays.indices.lookup,
                    arrays.Linv,
                    delta,
                    solver._v_hat,
                ],
                outputs=[solver._v_out],
                device=device,
            )
        if solver._has_free_rigid_bodies:
            wp.launch(
                apply_sparse_free_velocity,
                dim=(self.free_group_to_art.shape[0], 6),
                inputs=[
                    self.free_group_to_art,
                    solver._art_to_world,
                    solver._articulation_world_dof_offset,
                    solver._articulation_dof_start,
                    delta,
                    solver._v_hat,
                ],
                outputs=[solver._v_out],
                device=device,
            )


def scene_topology_keys(solver, model) -> list[tuple | None]:
    """The topology key (:func:`topology_key`) of each articulation, ``None`` where it has no factor.

    An articulation without response DOFs and a free rigid body have no key. Any other
    articulation's key holds the local parent of each tree joint (``-1`` for a parent outside
    the articulation) and each joint's DOF count, the input of its sparse factor plan. Reads the solver's
    ``_model_plan`` and ``_articulation_joint_end``.
    """
    plan = solver._model_plan
    keys: list[tuple | None] = [None] * len(plan.response_dof_count)
    articulated = np.flatnonzero((plan.response_dof_count > 0) & (plan.is_free_rigid == 0))
    if not articulated.size:
        return keys
    articulation_start = model.articulation_start.numpy()
    joint_end = solver._articulation_joint_end.numpy()
    joint_ancestor = model.joint_ancestor.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    for art in articulated:
        start, end = int(articulation_start[art]), int(joint_end[art])
        parents = joint_ancestor[start:end]
        parents = np.where(parents >= start, parents - start, -1)
        keys[art] = topology_key(parents, np.diff(joint_qd_start[start : end + 1]))
    return keys


def factor_body_masks(solver, model, groups: tuple[SceneGroup, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Each body's support mask in its articulation's factor coordinates, and its factor group.

    The mask of a factor group's link is its plan's ``joint_ancestor_mask`` entry of the joint
    whose child it is: the factor coordinates of its ancestor joints, which a contact on it
    touches. Returns ``(body_masks, body_group)`` over the model's bodies: ``uint64`` masks,
    zero for a body outside the factor groups (a free body, a static or kinematic body), and
    the index of its group in ``groups``, ``-1`` outside them. The masks have the words of the
    widest group (:attr:`SceneGroup.mask_words`): ``[bodies]`` for one word, as Steel's sparse
    path has, and ``[bodies, mask_words]`` for more, a narrower group's masks zero in the
    words past its own. Reads the model's ``articulation_start`` and ``joint_child``; ``solver`` is the calling
    convention of the planning functions and is not read.
    """
    words = max((group.mask_words for group in groups if group.kind == "factor"), default=1)
    body_masks = np.zeros((model.body_count, words) if words > 1 else model.body_count, dtype=np.uint64)
    body_group = np.full(model.body_count, -1, dtype=np.int32)
    articulation_start = model.articulation_start.numpy()
    joint_child = model.joint_child.numpy()
    for index, group in enumerate(groups):
        if group.kind != "factor":
            continue
        # The members share the key, so each has the key's joints from its articulation_start on.
        joints = articulation_start[np.asarray(group.arts)][:, None] + np.arange(len(group.key[0]))
        children = joint_child[joints]
        valid = children >= 0
        masks = group.plan.joint_ancestor_mask
        if words > 1:
            # Word w of the group's masks is word w of the scene's; a one-word group fills word 0.
            masks = np.pad(masks.reshape(len(masks), -1), ((0, 0), (0, words - group.plan.mask_words)))
        body_masks[children[valid]] = np.broadcast_to(masks, children.shape + masks.shape[1:])[valid]
        body_group[children[valid]] = index
    return body_masks, body_group


def scene_support(solver, model, groups: tuple[SceneGroup, ...], body_masks: np.ndarray, body_group: np.ndarray) -> int:
    """The scene rows' support ``S``: the most response coordinates one row touches.

    ``body_masks`` and ``body_group`` are :func:`factor_body_masks` of ``groups``. For factor
    group ``g`` let ``bits_g`` be the largest popcount of its plan's ancestor masks (a contact
    on its deepest link, or a joint-limit row) and ``self_g`` the largest popcount of the OR
    of two of its contact bodies' masks (a contact within one articulation; the contact bodies
    are its links with a shape). The rules:

    - ``S`` is the largest ``bits_g`` or ``self_g`` over the groups.
    - A row between two articulations holds their disjoint coordinate blocks, so ``S`` is at
      least ``b1 + b2``, the two largest ``bits`` over the articulations: the scene folds
      every world into one, so any two may meet, and one articulation alone adds nothing.
    - With free bodies, a row between a link and a free body has up to ``bits_g + 6``
      entries and one between two free bodies :data:`SCENE_FREE_SUPPORT`.
    - A closure's rows hold the OR of its two endpoints' masks (:func:`scene_closure_support`): a loop closure's parent
      and child bodies (the child's alone under a prescribed parent), a mimic row's follower and leader DOFs. Neither
      need carry a shape nor share a chain, so neither ``self_g`` nor ``bits_g`` bounds them.
    - No row has more entries than the scene's response coordinates, which cap ``S``.

    With one factor group and no closure this is the rule Steel's sparse path applies to one world
    (``SolverSteel._setup_sparse_mass_matrix``), on the folded world. Reads the solver's ``_has_free_rigid_bodies``
    and ``_max_world_dofs``, and the closures' (:func:`scene_closure_support`).
    """
    shape_body = model.shape_body.numpy() if model.shape_count else np.zeros(0, dtype=np.int32)
    contact_bodies = shape_body[shape_body >= 0]
    support = 0
    bits = []
    for index, group in enumerate(groups):
        if group.kind != "factor":
            continue
        group_bits = max((mask_value(mask).bit_count() for mask in group.plan.joint_ancestor_mask), default=0)
        # Each distinct contact mask once, as an int of all its words.
        contact_masks = distinct_mask_values(body_masks[contact_bodies[body_group[contact_bodies] == index]])
        support = max(support, group_bits)
        for first in contact_masks:
            for second in contact_masks:
                support = max(support, (first | second).bit_count())
        bits.extend([group_bits] * min(len(group.arts), 2))
    widest = sorted(bits, reverse=True)[:2]
    if len(widest) == 2:
        support = max(support, widest[0] + widest[1])
    if solver._has_free_rigid_bodies:
        support = max(support, max(bits, default=0) + 6, SCENE_FREE_SUPPORT)
    support = max(support, scene_closure_support(solver, groups, body_masks, body_group))
    return min(support, solver._max_world_dofs)


def scene_closure_support(
    solver, groups: tuple[SceneGroup, ...], body_masks: np.ndarray, body_group: np.ndarray
) -> int:
    """The most entries of a closure's row in factor coordinates: :func:`scene_support`'s closure rule.

    A loop closure's connect rows (Steel's ``SolverSteel._build_connect_plan``) whose child is a link of a factor group
    hold the OR of its parent's and its child's body masks, the child's alone under a prescribed parent (the world or a
    kinematic body of another articulation); a free-body child's rows are its 6 physical entries, which the free bodies'
    rule bounds. A mimic row (Steel's ``SolverSteel._build_mimic_plan``) holds the OR of its follower's and its leader's
    DOF masks (the plan's ``ancestor_mask`` of each DOF, the support of its factor column). ``0`` without closures.
    Reads Steel's connect and mimic tables: ``_connect_count``, ``_connect_body_p_np``, ``_connect_body_c``,
    ``_connect_parent_prescribed_np``, ``_mimic_count``, ``_mimic_dof0``, ``_mimic_dof1``, ``_mimic_art_list``,
    ``_mimic_art_start_np`` and ``_model_plan``.
    """
    support = 0
    if solver._connect_count:
        for parent, child, prescribed in zip(
            solver._connect_body_p_np.tolist(),
            solver._connect_body_c.numpy().tolist(),
            solver._connect_parent_prescribed_np.tolist(),
            strict=True,
        ):
            if body_group[child] < 0:
                continue
            mask = mask_value(body_masks[child])
            if not prescribed:
                mask |= mask_value(body_masks[parent])
            support = max(support, mask.bit_count())
    if solver._mimic_count:
        dof_start = np.asarray(solver._model_plan.articulation_dof_start, dtype=np.int64)
        art_group = {art: group for group in groups if group.kind == "factor" for art in group.arts}
        followers, leaders = solver._mimic_dof0.numpy(), solver._mimic_dof1.numpy()
        rows = solver._mimic_art_list.numpy()
        starts = solver._mimic_art_start_np
        for art in np.flatnonzero(np.diff(starts)).tolist():
            masks = art_group[art].plan.ancestor_mask
            for row in rows[starts[art] : starts[art + 1]].tolist():
                follower, leader = int(followers[row] - dof_start[art]), int(leaders[row] - dof_start[art])
                support = max(support, (mask_value(masks[follower]) | mask_value(masks[leader])).bit_count())
    return support


def factor_bytes_per_warp(size: int, nonzero_count: int) -> int:
    """Static shared memory one warp of the sparse factor kernel stages: its factor, inverse and forces.

    Steel's per-warp bytes of ``SolverSteel._setup_sparse_mass_matrix``, ``4 * (2 * nonzero_count + 6 * size)``.
    """
    return 4 * (2 * nonzero_count + 6 * size)


def static_shared_limit(device) -> int:
    """Static shared memory a block may hold on ``device``: 48 KiB, or less if the device allows less (Steel's)."""
    return min(48 * 1024, int(getattr(wp.get_device(device), "max_shared_memory_per_block", 0)))


def scene_factor_warps(solver, size: int, nonzero_count: int) -> int:
    """Warps per block of a scene factor group's sparse factor kernel, counting the static shared memory Warp itself
    reserves per thread of the block (``scene_rows.WARP_SHARED_BYTES_PER_THREAD``).

    Steel's sparse path sizes its factor's blocks by the factor's bytes alone
    (``SolverSteel._setup_sparse_mass_matrix``, :func:`factor_bytes_per_warp`). Past 64 DOFs one warp can come within
    the reservation of a whole block, and the kernel would then fail when it compiles instead of the scene solve
    refusing the topology at construction. On the CPU the kernel's single-lane path keeps its factor in local arrays,
    with no shared memory to bound it: a block holds the most warps, Steel's ``_SPARSE_FACTOR_WARPS_PER_BLOCK``.
    ``0`` when not even one warp fits. Reads the solver's ``model.device``.
    """
    from . import solver_steel  # noqa: PLC0415 (the solver module builds on the scene modules)

    warps_max = solver_steel._SPARSE_FACTOR_WARPS_PER_BLOCK
    device = solver.model.device
    if not device.is_cuda:
        return warps_max
    per_warp = factor_bytes_per_warp(size, nonzero_count) + 32 * WARP_SHARED_BYTES_PER_THREAD
    return min(warps_max, static_shared_limit(device) // per_warp)


def setup_scene_free_only(solver) -> None:
    """Plan the scene rows of a model of free bodies only (``solve_mode="scene"``).

    Without an articulated group there is no factor, so ``_sparse_mass_matrix_size`` stays
    ``None``, the scene's one group is its free group (``_sparse_mass_matrix_groups``) and no
    body has factor coordinates (``_sparse_body_dof_mask`` is zero). Every response
    coordinate belongs to a free body, so a row holds at most two physical 6-entry blocks
    (:data:`SCENE_FREE_SUPPORT`), with ``H^-1 J^T`` of the same width in its free response.

    Called from the solver's sparse factor selection in place of its factor plan. Reads the solver's ``model``,
    ``_model_plan``, ``_world_count``, ``_articulation_constraint_max`` and ``_max_world_dofs``; writes Steel's sparse
    row store (``_sparse_row_dof``, ``_sparse_row_factor``, ``_sparse_row_free_response``, ``_sparse_row_incident``,
    ``_sparse_factor_velocity_delta``, ``_sparse_world_free_dof_mask``, ``_sparse_free_factor_dummy``), its
    ``_sparse_body_dof_mask``, ``_sparse_mass_matrix_groups``, ``_sparse_contact_lanes`` and no sweep kernel
    (``_pgs_solve_sparse_kernel = None``).
    """
    model = solver.model
    device = model.device
    plan = solver._model_plan
    solver._sparse_mass_matrix_groups = plan_scene_groups(
        scene_topology_keys(solver, model), (plan.is_free_rigid != 0) & (plan.response_dof_count > 0)
    )
    shape = (solver._world_count, solver._articulation_constraint_max, SCENE_FREE_SUPPORT)
    solver._sparse_body_dof_mask = wp.zeros(model.body_count, dtype=wp.uint64, device=device)
    solver._sparse_row_dof = wp.empty(shape, dtype=wp.int32, device=device)
    solver._sparse_row_factor = wp.empty(shape, dtype=wp.float32, device=device)
    solver._sparse_row_free_response = wp.empty(shape, dtype=wp.float32, device=device)
    solver._sparse_row_incident = wp.empty(shape[:2], dtype=wp.float32, device=device)
    coordinates = (solver._world_count, solver._max_world_dofs)
    solver._sparse_factor_velocity_delta = wp.empty(coordinates, dtype=wp.float32, device=device)
    solver._sparse_world_free_dof_mask = wp.ones(coordinates, dtype=wp.int32, device=device)
    solver._sparse_free_factor_dummy = wp.zeros((1, 6, 6), dtype=wp.float32, device=device)
    solver._sparse_contact_lanes = 8
    solver._pgs_solve_sparse_kernel = None


def scene_row_capacity(solver, model) -> int:
    """The scene solve's row capacity: Steel's per-world row budgets, summed over the model's worlds.

    Each model world brings ``articulation_constraint_max`` rows and, when the model has free
    bodies, ``free_body_constraint_max`` more. Global entities (world ``-1``) add no world: Steel
    solves them in world 0, within that world's budgets. Reads the solver's ``_requested_dense_max_constraints``
    (the requested ``articulation_constraint_max``), ``_has_free_rigid_bodies`` and ``_free_body_constraint_max``.
    """
    per_world = solver._requested_dense_max_constraints
    if solver._has_free_rigid_bodies:
        per_world += solver._free_body_constraint_max
    return max(int(model.world_count), 1) * per_world


def setup_scene_plan(solver, model) -> None:
    """Record the scene solve's row plan and allocate its per-articulation internal-row counters.

    ``_scene_family_rows`` (``int32``, ``[FAMILY_COUNT, articulations]``) receives each
    articulation's rows of each internal-unit family from its COUNT pass, and
    ``_scene_internal_rows`` (``[articulations]``) their sum, the internal unit's rows: an array of
    its own when more than one family is in use (:meth:`SceneSolve.count_internal_rows` sums them),
    otherwise a view of the one family's row (of the joint limits' with none). ``_scene_limit_rows``
    views the joint-limit family's row. ``_scene_internal_families`` lists the families in use, in
    Steel's slot order (the factor groups' and the connect family, which every closure owner shares,
    free bodies included), and ``_scene_row_cursor`` is the WRITE passes' cursor. ``_scene_late_rows`` views
    the velocity-limit family's row, each articulation's late rows, and ``_scene_late_cursor``
    (``[articulations]``, with late rows) is that family's WRITE cursor. ``_scene_late_rows_enabled`` is
    whether the scene builds late rows: with ``enable_joint_velocity_limits``, or when a free body can bear
    velocity-limit rows (its bit of :attr:`_scene_family_bearing`; Steel builds the free bodies' whatever
    ``enable_joint_velocity_limits``). ``_scene_depenetration_cap`` is whether a free body has a finite positive
    ``rigid_body_max_depenetration_velocity`` (read once, here), the scene's ``SceneFeatures.depenetration_cap``.
    ``_scene_family_bearing`` records which articulations can bear rows of which family
    (:func:`scene_internal_bearing`). ``_scene_plan`` is the :class:`ScenePlan`. Without the scene mode
    (``solver._scene_mode`` false) nothing is planned or allocated: every attribute is ``None``, empty or ``False``.

    Called at construction after the solver's sparse factor selection, which under the scene mode planned the groups
    (``_sparse_mass_matrix_groups``, :func:`plan_scene_groups`) and allocated the row store (``_sparse_row_dof`` of
    the scene's support); raises :class:`NotImplementedError` when an articulated model got no factor. Reads the
    solver's ``_scene_mode``, ``_model_plan``, ``_sparse_mass_matrix_size``, ``_sparse_mass_matrix_groups``,
    ``_sparse_row_dof``, ``_articulation_constraint_max`` (the scene's row capacity, :func:`scene_row_capacity`),
    ``_max_world_dofs``, ``_free_rigid_body_count``, ``_enable_joint_velocity_limits`` and
    ``_namespaced_attribute``, and what :func:`scene_internal_bearing` and :func:`scene_group_families` read.
    """
    solver._scene_plan = None
    solver._scene_family_rows = None
    solver._scene_internal_rows = None
    solver._scene_internal_families = ()
    solver._scene_limit_rows = None
    solver._scene_row_cursor = None
    solver._scene_late_rows = None
    solver._scene_late_cursor = None
    solver._scene_late_rows_enabled = False
    solver._scene_depenetration_cap = False
    solver._scene_family_bearing = None
    if not solver._scene_mode:
        return
    solver._scene_family_bearing = scene_internal_bearing(solver, model)
    plan = solver._model_plan
    if np.any((plan.response_dof_count > 0) & (plan.is_free_rigid == 0)) and solver._sparse_mass_matrix_size is None:
        raise NotImplementedError('solve_mode="scene" found no sparse factor for this articulated topology')
    groups = solver._sparse_mass_matrix_groups
    solver._scene_plan = ScenePlan(
        support=int(solver._sparse_row_dof.shape[2]),
        row_max=int(solver._articulation_constraint_max),
        coordinate_count=int(solver._max_world_dofs),
        free_body_count=int(solver._free_rigid_body_count),
        groups=groups,
        mask_words=max((group.mask_words for group in groups), default=1),
    )
    families = {family for group in groups if group.kind == "factor" for family in scene_group_families(solver, group)}
    # The connect family is the scene's, not a factor group's: a loop closure's child may be a free body.
    if np.any((solver._scene_family_bearing >> FAMILY_CONNECT) & 1):
        families.add(FAMILY_CONNECT)
    families = sorted(families)
    solver._scene_internal_families = tuple(families)
    solver._scene_family_rows, solver._scene_internal_rows = allocate_scene_internal_rows(
        model.articulation_count, families, model.device
    )
    solver._scene_limit_rows = solver._scene_family_rows[FAMILY_LIMIT]
    solver._scene_row_cursor = wp.zeros(model.articulation_count, dtype=wp.int32, device=model.device)
    solver._scene_late_rows = solver._scene_family_rows[FAMILY_VELOCITY_LIMIT]
    free_late = bool(np.any((solver._scene_family_bearing >> FAMILY_VELOCITY_LIMIT) & 1))
    solver._scene_late_rows_enabled = bool(solver._enable_joint_velocity_limits) or free_late
    if solver._scene_late_rows_enabled:
        solver._scene_late_cursor = wp.zeros(model.articulation_count, dtype=wp.int32, device=model.device)
    caps = solver._namespaced_attribute("rigid_body_max_depenetration_velocity")
    free_bodies = np.asarray(plan.response_free_rigid_body_indices, dtype=np.int64)
    if caps is not None and free_bodies.size:
        values = caps.numpy()[free_bodies]
        solver._scene_depenetration_cap = bool(np.any(np.isfinite(values) & (values > 0.0)))


def scene_group_late(solver, group) -> wp.Kernel | None:
    """The builder of scene group ``group``'s late units (velocity-limit rows), ``None`` when it builds none.

    A group builds late rows when the scene does (``_scene_late_rows_enabled``) and the velocity-limit bit of
    :attr:`_scene_family_bearing` (:func:`scene_internal_bearing`) is set on some member. The builder is the
    free-body velocity limits' for the free group
    (:func:`~newton._src.solvers.steel.scene_rows.build_sparse_free_velocity_limit_rows`) and the joint
    velocity limits' for a factor group, of its ``mask_words``
    (:func:`~newton._src.solvers.steel.scene_rows.get_sparse_joint_velocity_limit_rows_kernel`). Reads the solver's
    ``_scene_late_rows_enabled`` and ``_scene_family_bearing`` (:func:`setup_scene_plan`).
    """
    if not solver._scene_late_rows_enabled:
        return None
    bearing = solver._scene_family_bearing[np.asarray(group.arts, dtype=np.int64)]
    if not np.any((bearing >> FAMILY_VELOCITY_LIMIT) & 1):
        return None
    if group.kind == "free":
        return get_sparse_free_velocity_limit_rows_kernel()
    return get_sparse_joint_dof_rows_kernel(FAMILY_VELOCITY_LIMIT, group.mask_words)


def scene_group_families(solver, group) -> tuple[int, ...]:
    """The internal-unit row families factor group ``group`` builds, in Steel's slot order.

    The drives, the mimic rows and the joint friction when a member can bear their rows
    (:attr:`_scene_family_bearing`, :func:`scene_internal_bearing`); the joint limits when
    ``enable_joint_limits`` is on and the group's size has limited joints (Steel's
    ``_joint_limit_sizes``, the test of the limits pass before the internal unit, which keeps its
    launches). Free bodies build none. The connect family is not a group's: one launch builds every
    closure's rows (:func:`setup_scene_plan`). Reads the solver's ``_scene_family_bearing``,
    ``_enable_joint_limits`` and ``_joint_limit_sizes``.
    """
    if group.kind != "factor":
        return ()
    bearing = int(np.bitwise_or.reduce(solver._scene_family_bearing[np.asarray(group.arts, dtype=np.int64)]))
    families = []
    if bearing & (1 << FAMILY_DRIVE):
        families.append(FAMILY_DRIVE)
    if bearing & (1 << FAMILY_MIMIC):
        families.append(FAMILY_MIMIC)
    if solver._enable_joint_limits and group.size in solver._joint_limit_sizes:
        families.append(FAMILY_LIMIT)
    if bearing & (1 << FAMILY_JOINT_FRICTION):
        families.append(FAMILY_JOINT_FRICTION)
    return tuple(families)


def scene_internal_bearing(solver, model) -> np.ndarray:
    """Flag (``int32``, one per articulation) the internal-unit row families each articulation can bear.

    Bit ``1 << family`` is set when one of the articulation's response DOFs can have a row of that
    family at some step, whatever the step's poses, coefficients and targets:

    - ``FAMILY_DRIVE``, with ``drive_mode="rows"``: a DOF of a PRISMATIC, REVOLUTE or D6 joint
      (``_joint_limit_q_index >= 0``), the joint types of Steel's drive rows, whatever its gains,
      which are read on the device at every step and may be edited in place;
    - ``FAMILY_LIMIT``, with ``enable_joint_limits``: such a DOF with a finite lower or upper bound
      (the bounds are read once, here), the per-DOF test of Steel's joint-limit row builders;
    - ``FAMILY_JOINT_FRICTION``, with ``enable_joint_friction``: a DOF of a joint type that supports
      friction (Steel's ``_friction_dof_supported``: PRISMATIC, REVOLUTE, BALL and D6 joints outside
      loop closures), whatever its coefficient.
    - ``FAMILY_MIMIC`` and ``FAMILY_CONNECT``, under the scene mode (``solver._scene_mode``): an articulation that
      owns a mimic row (Steel's ``SolverSteel._build_mimic_plan``) or a loop closure
      (``SolverSteel._build_connect_plan``, a free body among them), whatever its enabled flags, which are read on
      the device at every step.
    - ``FAMILY_VELOCITY_LIMIT``, with ``enable_joint_velocity_limits``: a DOF of a PRISMATIC, REVOLUTE or
      D6 joint whose :attr:`~newton.Model.joint_velocity_limit` is not non-positive (read once, here; an
      infinite limit builds rows in Steel too, which never act), the static part of Steel's per-DOF test of
      its velocity-limit rows (``allocate_joint_velocity_limit_slots``). Its rows are the late unit's. A free
      rigid body with a response bears it when both rigid-body velocity limits are registered and its
      ``rigid_body_max_linear_velocity`` or ``rigid_body_max_angular_velocity`` is finite and positive (read
      once, here), the static part of Steel's per-axis test (``allocate_rigid_velocity_limit_slots``, which
      skips non-positive and non-finite limits, with ``enable_joint_velocity_limits`` or not).

    Free rigid bodies (a FREE joint) bear no other family but the connect family (a loop closure's child).
    Each joint family's test is one prefix sum over the
    DOFs, read at each articulation's response DOFs ``[articulation_dof_start, + response_dof_count)``.
    The scene's units (``scene_units.SceneUnits``, whose capacity counts an internal unit per articulation with
    an internal family's bit and a late unit per articulation with the velocity-limit bit),
    :func:`scene_group_families` and :func:`scene_group_late` read it.

    Reads the solver's ``_model_plan``, ``_joint_limit_q_index_host``, ``_has_drive_rows``, ``_enable_joint_limits``,
    ``_enable_joint_friction``, ``_friction_dof_supported_host``, ``_enable_joint_velocity_limits``, ``_scene_mode``,
    ``_mimic_count``, ``_mimic_art_start_np``, ``_connect_count``, ``_connect_art_np``,
    ``_has_rigid_body_velocity_limits``, ``_namespaced_attribute`` and ``_body_to_articulation``.
    """
    bearing = np.zeros(model.articulation_count, dtype=np.int32)
    if not model.articulation_count or not model.joint_dof_count:
        return bearing
    plan = solver._model_plan
    start = np.asarray(plan.articulation_dof_start, dtype=np.int64)
    end = start + np.asarray(plan.response_dof_count, dtype=np.int64)
    q_index = np.asarray(solver._joint_limit_q_index_host)
    per_family = {}
    if solver._has_drive_rows:
        per_family[FAMILY_DRIVE] = q_index >= 0
    if solver._enable_joint_limits:
        finite = np.isfinite(model.joint_limit_lower.numpy()) | np.isfinite(model.joint_limit_upper.numpy())
        per_family[FAMILY_LIMIT] = (q_index >= 0) & finite
    if solver._enable_joint_friction:
        per_family[FAMILY_JOINT_FRICTION] = np.asarray(solver._friction_dof_supported_host, dtype=bool)
    if solver._enable_joint_velocity_limits:
        # Steel skips a DOF whose limit is non-positive (``qdot_max <= 0``); any other limit builds its rows.
        per_family[FAMILY_VELOCITY_LIMIT] = (q_index >= 0) & ~(model.joint_velocity_limit.numpy() <= 0.0)
    for family, dofs in per_family.items():
        counts = np.concatenate(([0], np.cumsum(dofs, dtype=np.int64)))
        bearing |= (counts[end] > counts[start]).astype(np.int32) << family
    if solver._scene_mode and solver._mimic_count:
        bearing[np.diff(solver._mimic_art_start_np) > 0] |= 1 << FAMILY_MIMIC
    if solver._scene_mode and solver._connect_count:
        bearing[solver._connect_art_np] |= 1 << FAMILY_CONNECT
    free_bodies = np.asarray(plan.response_free_rigid_body_indices, dtype=np.int64)
    if solver._has_rigid_body_velocity_limits and free_bodies.size:
        bears = np.zeros(free_bodies.size, dtype=bool)
        for name in ("rigid_body_max_linear_velocity", "rigid_body_max_angular_velocity"):
            limits = solver._namespaced_attribute(name).numpy()[free_bodies]
            bears |= np.isfinite(limits) & (limits > 0.0)
        arts = solver._body_to_articulation.numpy()[free_bodies[bears]]
        bearing[arts] |= 1 << FAMILY_VELOCITY_LIMIT
    return bearing


def _launch_by_name(kernel: wp.Kernel, dim: int, arguments: dict, device) -> None:
    """Launch ``kernel`` with every argument of its signature by name, in its order, from ``arguments``.

    Steel's contact row builders gain and rename parameters upstream; launching them by name makes a new, renamed or
    removed parameter fail here, naming it, instead of shifting every later argument by one.

    Raises:
        TypeError: When ``arguments`` misses a parameter of ``kernel`` or names one it does not have.
    """
    names = [arg.label for arg in kernel.adj.args]
    missing = [name for name in names if name not in arguments]
    unknown = sorted(set(arguments) - set(names))
    if missing or unknown:
        raise TypeError(f"{kernel.key} launched with missing arguments {missing} and unknown arguments {unknown}")
    wp.launch(kernel, dim=dim, inputs=[arguments[name] for name in names], device=device)


def _contact_geometry(contacts, threads: int) -> dict:
    """Each contact's geometry, by the contact row builders' argument names (``contact_count`` to
    ``contact_thickness1``), with the builders' grid-stride thread count ``threads``."""
    return {
        "contact_count": contacts.rigid_contact_count,
        "total_num_threads": threads,
        "contact_point0": contacts.rigid_contact_point0,
        "contact_point1": contacts.rigid_contact_point1,
        "contact_normal": contacts.rigid_contact_normal,
        "contact_shape0": contacts.rigid_contact_shape0,
        "contact_shape1": contacts.rigid_contact_shape1,
        "contact_thickness0": contacts.rigid_contact_margin0,
        "contact_thickness1": contacts.rigid_contact_margin1,
    }


def _contact_slot_arguments(solver, state_in, contact_geometry: dict, contact_response, is_free_rigid) -> dict:
    """The arguments of Steel's ``allocate_world_contact_slots`` in the scene's row build, by name.

    The contacts are routed without reserving rows (``classify_only``): every contact with a responding side goes to
    the factor-coordinate rows (path 0), free-body pairs included, and none to the propagation rows; the spare torsion
    rows are the propagation rows', and the scene's spin rows are its own late rows. ``contact_response`` is the
    bodies' response mask (Steel's, or the sleeping one), ``is_free_rigid`` the free-body flags (or Steel's dummy).
    """
    return {
        **contact_geometry,
        "body_q": state_in.body_q,
        "shape_body": solver.model.shape_body,
        "body_to_articulation": solver._body_to_articulation,
        "art_to_world": solver._art_to_world,
        "articulation_response_dof_count": solver._articulation_response_dof_count,
        "body_flags": solver.model.body_flags,
        "body_has_response_dofs": contact_response,
        "is_free_rigid": is_free_rigid,
        # Free-body pairs take the factor-coordinate rows too, and no contact the propagation rows.
        "has_free_rigid": 0,
        "propagation_articulated": 0,
        "propagation_same_articulation": 0,
        "propagation_free_free": 0,
        "max_constraints": solver._articulation_constraint_max,
        "mf_max_constraints": solver._free_body_constraint_max,
        "propagation_max_constraints": solver._propagation_max_constraints,
        "art_model_world": solver._articulation_model_world,
        "contact_gap_limit": solver._contact_gap_limit,
        "contact_friction_gap_limit": solver._friction_gap,
        "friction_patches": solver._friction_patches.view,
        # SceneSolve places the rows.
        "classify_only": 1,
        # Spare torsion rows are the propagation rows'; the spin rows are the scene's own late rows.
        "shape_torsion_radius": solver._no_torsion_radius,
        "torsion_spares": 0,
        "contact_world": solver._contact_world,
        "contact_slot": solver._contact_slot,
        "contact_art_a": solver._contact_art_a,
        "contact_art_b": solver._contact_art_b,
        "contact_slots_needed": solver._contact_slots_needed,
        "world_slot_counter": solver._slot_counter,
        "contact_path": solver._contact_path,
        "mf_slot_counter": solver._dummy_mf_slot_counter,
        "propagation_slot_counter": solver._propagation_slot_counter,
        "dense_contact_world_flag": solver._dense_contact_world_flag,
        "dense_dropped_contact_rows": solver._row_dropped_dense,
        "mf_dropped_contact_rows": solver._row_dropped_mf,
        "propagation_dropped_contact_rows": solver._row_dropped_propagation,
        "dense_first_rejected_slot": solver._dense_first_rejected_slot,
        "mf_first_rejected_slot": solver._dummy_mf_slot_counter,
        "propagation_first_rejected_slot": solver._propagation_first_rejected_slot,
        "cross_world_contacts": solver._cross_world_contacts,
    }


def _contact_row_arguments(solver, state_in, state_aug, contacts, contact_geometry: dict) -> dict:
    """The arguments of Steel's ``prepare_world_contact_rows`` in the scene's row build, by name: the contacts'
    row metadata in the scene's one row store, each contact row's rebound velocity included (``_row_rebound_velocity``,
    from the step-start closing speed)."""
    return {
        **contact_geometry,
        "contact_world": solver._contact_world,
        "contact_slot": solver._contact_slot,
        "contact_art_a": solver._contact_art_a,
        "contact_art_b": solver._contact_art_b,
        "contact_path": solver._contact_path,
        "contact_slots_needed": solver._contact_slots_needed,
        "shape_body": solver.model.shape_body,
        "body_q": state_in.body_q,
        "body_v_s": state_aug._body_v_s,
        "contact_surface_velocity": contacts.rigid_contact_surface_velocity,
        "prescribed_articulation": solver._prescribed_articulation,
        "articulation_origin": solver._articulation_origin,
        "shape_material_mu": solver._shape_mu,
        "shape_material_restitution": solver._shape_restitution,
        "friction_patches": solver._friction_patches.view,
        "contact_shared_anchor": solver._contact_anchor_args[0],
        "contact_friction_shared_anchor": solver._contact_anchor_args[1],
        "restitution_velocity_threshold": solver._rebound_velocity_threshold,
        "world_row_type": solver._row_type,
        "world_row_parent": solver._row_parent,
        "world_row_mu": solver._row_mu,
        "world_phi": solver._phi,
        "world_target_velocity": solver._target_velocity,
        "world_row_rebound_velocity": solver._row_rebound_velocity,
    }


def build_scene_rows(solver, state_in, state_aug, contacts, dt: float, control=None) -> None:
    """Build the scene's rows (``solve_mode="scene"``; Steel's stage 4): units first, then rows placed by scans in
    unit order.

    1. The internal unit's COUNT passes write each articulation's rows of each row family in use
       (drives, mimic, connect, joint limits, joint friction) into ``_scene_family_rows`` and their sum into
       ``_scene_internal_rows`` (:meth:`SceneSolve.count_internal_rows`). The drives read
       ``control``'s joint targets, so a scene with drive rows needs the step's ``control``. With late
       rows (joint or free-body velocity limits) the velocity-limit family's COUNT pass writes each
       articulation's late rows into ``_scene_late_rows`` (:meth:`SceneSolve.count_late_rows`). With sleeping,
       Steel's mask selection applies: the drive and joint-limit families read ``_SleepState.limit_q_index``,
       ``-1`` on the DOFs of an articulation that sleeps with its rows skipped, and the joint friction and the
       late rows ``_constraint_art_active``, ``0`` for it, so it has no internal or late rows.
    2. With friction patches (``friction_anchor_beta > 0``) Steel's patches are built first, as Steel builds
       them before its allocation reads their anchors, carrying the regions of sleeping bodies
       (``_SleepState.patch_frozen_bodies``) and their anchors' displacement over the step's ``dt``. With contact
       compliance the contacts' row requests are cleared first, as Steel's ``contact_compliance.begin_rows`` clears
       them. The contacts are routed without reserving rows
       (``classify_only``): every contact with a responding side goes to the factor-coordinate rows (path 0),
       free-body pairs included, with three rows, or its normal row alone past ``contact_friction_gap_max`` or
       as a patch region's member past its anchors. With sleeping a body responds through
       ``_SleepState.contact_response_mask``, as in Steel's allocation, so a contact without an awake responding
       side gets path -1 and no unit. Steel's ``allocate_world_contact_slots`` is launched by its arguments' names
       (no propagation rows, no spare torsion rows: the scene's spin rows are its own).
    3. :class:`SceneUnits` builds the units, an internal and a late unit with the COUNT passes' rows, and
       colors the two-sided ones.
    4. :meth:`SceneSolve.assign_rows` places their rows: one-sided units side-major, then the
       two-sided units by color, then the late units side-major; units past
       ``articulation_constraint_max`` are dropped.
    5. Steel's builders write the rows at those places: the internal unit's WRITE passes from
       ``_scene_row_cursor``, family after family, the late unit's from ``_scene_late_cursor``, then
       ``prepare_world_contact_rows`` (by its arguments' names; it writes each contact row's rebound velocity,
       ``_row_rebound_velocity``, from the step-start closing speed) and the sparse contact response kernel from
       ``_contact_slot``, each contact's ``contact_slots_needed`` rows (three, or its normal row alone), an anchor's
       tangent rows at its anchor point (the kernel's ``patches`` variant). With friction patches
       ``link_patch_rows`` then rings each region's normal rows through their ``row_parent`` and divides the
       anchors' ``row_mu`` by the region's anchors with rows, on the scene's one route (route 0), as Steel links them
       after its row builders.
    6. :meth:`SceneSolve.prepare_rows` adds ``cfm``, writes the bias and the rebound targets, clears
       the impulses, with drive rows writes their force-drive descriptors and with the depenetration cap
       bounds the free bodies' contact bias; with contact torsion :meth:`SceneSolve.prepare_torsion` then groups
       the contacts and appends the spin rows after every unit's rows; :meth:`SceneSolve.report_status` latches
       the losses.

    Launch shapes are fixed by the capacities and nothing is read back to the host, so the
    build can be captured in a CUDA graph. Steel's per-world reservation and finishers, its
    dense and free-body (MF) row families and its per-world status are not used. ``state_in`` is the step's input
    state, ``state_aug`` its augmented state (the body twists and the motion subspaces), ``control`` its control
    (required with drive rows). Reads the solver's ``_scene``, ``_scene_units`` and the attributes Steel's own row
    build reads; writes Steel's contact routing and its row store with one world.
    """
    model = solver.model
    scene = solver._scene
    units = solver._scene_units
    has_contacts = contacts is not None and contacts.rigid_contact_max > 0
    # The internal unit's rows, one launch per row family in use per factor group.
    internal = bool(scene.internal_families)
    is_free_rigid = solver._is_free_rigid if solver._is_free_rigid is not None else solver._dummy_is_free_rigid
    # Steel's mask selection (_stage4_build_rows): sleeping islands that skip constraints get no contact, drive or
    # joint-limit rows (their contact response and limit coordinates are masked), and friction patches carry their
    # frozen regions. Joint friction and the late rows read _constraint_art_active.
    sleeping = solver._sleeping
    limit_q_index = solver._joint_limit_q_index
    contact_response = solver._body_has_response_dofs
    frozen_bodies = None
    if sleeping is not None:
        if sleeping.limit_q_index is not None:
            limit_q_index = sleeping.limit_q_index
        contact_response = sleeping.contact_response_mask
        frozen_bodies = sleeping.patch_frozen_bodies

    if internal:
        scene.count_internal_rows(
            solver, state_in, dt, control=control, limit_q_index=limit_q_index, state_aug=state_aug
        )
    # The late units' rows (joint and free-body velocity limits), one launch per group that builds them.
    late = scene.late_rows_in_use
    if late:
        scene.count_late_rows(solver)

    solver._cross_world_contacts.zero_()
    if has_contacts:
        from . import solver_steel  # noqa: PLC0415 (the solver module builds on the scene modules)

        contact_build_threads = min(contacts.rigid_contact_max, solver_steel._CONTACT_BUILD_THREAD_CAP)
        contact_geometry = _contact_geometry(contacts, contact_build_threads)
        if scene.features.compliance:
            # The allocator writes slots_needed only after its skip gates, so a stale request from an earlier step
            # could make an excluded contact look lost to the compliance checks (Steel's contact_compliance.begin_rows).
            solver._contact_slots_needed.zero_()
        if solver._friction_anchors_enabled:
            # Steel's patches first (its _stage4_build_rows' order): the allocation reads their anchors, three rows for
            # an anchor and the normal row alone for the region's other members; a sleeping island's regions are
            # carried.
            solver._friction_patches.build(
                model,
                state_in,
                contacts,
                shape_material_mu=solver._shape_mu,
                contact_gap_limit=solver._contact_gap_limit,
                friction_gap_limit=solver._friction_gap,
                frozen_bodies=frozen_bodies,
                dt=dt,
            )
        _launch_by_name(
            allocate_world_contact_slots,
            contact_build_threads,
            _contact_slot_arguments(solver, state_in, contact_geometry, contact_response, is_free_rigid),
            model.device,
        )

    units.build(
        solver, contacts, internal_rows=solver._scene_internal_rows, late_rows=solver._scene_late_rows if late else None
    )
    units.color()
    scene.assign_rows(solver)

    if internal:
        scene.launch_internal_rows(
            solver,
            state_in,
            ROW_SLOTS_WRITE,
            solver._scene_row_cursor,
            dt=dt,
            control=control,
            limit_q_index=limit_q_index,
            state_aug=state_aug,
        )
    if late:
        scene.launch_late_rows(solver, ROW_SLOTS_WRITE, solver._scene_late_cursor)
    if has_contacts:
        _launch_by_name(
            prepare_world_contact_rows,
            contact_build_threads,
            _contact_row_arguments(solver, state_in, state_aug, contacts, contact_geometry),
            model.device,
        )
        lanes = solver._sparse_contact_lanes
        workers = min(contact_build_threads, solver_steel._CONTACT_JACOBIAN_WORKER_CAP * 32 // lanes)
        # The response kernel's per-contact row counts and anchor inputs: Steel's own kernel (specialized) reads every
        # contact's row count and the friction patches; the generic one reads the row counts and the anchor points
        # only when a contact may have one row.
        if scene.response_kind == "specialized":
            row_counts = [solver._contact_slots_needed]
            anchors = [solver._friction_patches.view]
        elif scene.one_row_contacts:
            row_counts = [solver._contact_slots_needed]
            # Friction patches' anchors, the tangent rows' lever arms (Steel's contact_row_points), or placeholders.
            anchors = list(scene.anchor_points(solver))
        else:
            row_counts = []
            anchors = []
        wp.launch(
            scene.response_kernel,
            dim=workers * lanes,
            inputs=[
                contacts.rigid_contact_count,
                workers,
                *list(contact_geometry.values())[2:],
                solver._contact_world,
                solver._contact_slot,
                solver._contact_art_a,
                solver._contact_art_b,
                solver._contact_path,
                *row_counts,
                solver._art_group_idx,
                solver._articulation_response_dof_count,
                is_free_rigid,
                solver._articulation_world_dof_offset,
                solver._articulation_dof_start,
                solver._articulation_origin,
                solver._sparse_body_dof_mask,
                state_aug._joint_S_s,
                model.shape_body,
                state_in.body_q,
                *solver._contact_anchor_args,
                *anchors,
                *scene.factor_inputs,
                solver._Linv_by_size[6] if solver._has_free_rigid_bodies else solver._sparse_free_factor_dummy,
                solver._v_hat,
            ],
            outputs=[
                solver._sparse_row_dof,
                solver._sparse_row_factor,
                solver._sparse_row_free_response,
                solver._sparse_row_incident,
                solver._diag,
            ],
            block_dim=128,
            device=model.device,
        )
        if solver._friction_anchors_enabled:
            if scene.gather_mu is not None:
                # Steel gathers its free-body rows' warm start before this link (its _stage4_build_rows' order).
                scene.keep_gather_mu(solver)
            # Chain each region's normal rows into a ring through their parents and share its friction budget
            # among the anchors that got rows (Steel's link after its row builders), on the scene's one route (route
            # 0, the factor-coordinate rows), before anything reads a tangent row's row_mu or a normal row's parent
            # (the row preparation, the torsion grouping, the sweeps).
            wp.launch(
                link_patch_rows,
                dim=contacts.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    solver._friction_patches.view,
                    solver._contact_world,
                    solver._contact_slot,
                    solver._contact_path,
                    solver._contact_slots_needed,
                    0,
                    solver._row_parent,
                    solver._row_mu,
                ],
                device=model.device,
            )
            if scene.gather_mu is not None:
                scene.finish_gather_mu(solver, contacts)

    scene.prepare_rows(solver, dt, contacts if has_contacts else None)
    if scene.features.torsion:
        scene.prepare_torsion(solver, state_in, state_aug, contacts if has_contacts else None)
    scene.report_status(solver, contacts)
    scene.built = True
