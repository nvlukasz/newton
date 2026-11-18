# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The scene mode's sweep: Steel's sparse row loop on the scene's one solve world, and its kernel factories.

The scene mode (``solve_mode="scene"``) folds every model world into one solve world. Its rows are Steel's
factor-coordinate row arrays with one world, and its velocity delta and impulses live in global memory, so Steel's
per-world shared-memory sweep (``sparse_pgs._get_pgs_solve_sparse_kernel``) does not apply. This module holds the
native text of the scene's sweep, Steel's sparse row loop restated for those arrays, and the factories of its kernels.

**Sweep.** Phase A sweeps each side's one-sided rows, one launch running every iteration of the sides without
two-sided units with Steel's stationary exit per side (:func:`get_sweep_sides_kernel`). When the step has two-sided
units, each iteration then gives the coupled sides one sweep and runs phase B: the two-sided units color by color,
all units of a color at once (:func:`get_sweep_color_kernel` for the big colors, :func:`get_sweep_block_kernel` for
the small ones and the tail). ``scene_phase_order`` (:data:`SCENE_PHASE_ORDERS`) sets the order of the two in each
iteration. With Steel's summation order (:data:`SCENE_SUM_ORDERS`, the default up to 32 entries per row) one world of
one humanoid steps bitwise as Steel's own solve, and the phased sweep reproduces one thread walking the rows in the
order's sequence bit for bit (the serial oracles :func:`get_sweep_rows_serial_kernel` and
:func:`get_sweep_order_serial_kernel`, which also run on the CPU).

**Late rows and phase C.** Steel sweeps a world's velocity-limit rows after every other row of each iteration, and
its contact torsion's spin rows before them. Each iteration of the scene ends with phase C, which sweeps them in that
order: each side's one-sided ones, then the two-sided spin rows color by color (:func:`get_sweep_torsion_kernel`),
then the coupled sides' velocity limits.

**Per-row features** (:class:`SceneFeatures`) are build constants fixed at construction: every feature off compiles
the default kernels, whose native text the scene tripwire pins (:func:`scene_kernel_snippets`), and any feature on
compiles each factory's one featured definition, which takes every feature's argument.

**Restated laws.** Where Steel exposes a law as text the scene can paste unchanged, the scene imports it
(``friction.FRICTION_PAIR_CUDA``). Its friction modes' laws, the friction patches' pooled normal load and contact
torsion's budget are restated here (the friction modes and the budget round every operation on its own); each
restated law mirrors Steel's text statement for statement, which :func:`_mirrored_lines` extracts from both for the
guard test, so a change to Steel's law fails a test instead of leaving the scene's copy behind.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from functools import cache, partial
from typing import Literal

import warp as wp

from .contact_torsion import _TORSION_SPIN_RETIRED, torque_sweep_source
from .friction import FRICTION_PAIR_CUDA
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_TARGET,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    PGS_CONSTRAINT_TYPE_TORSION,
    WARP_DETERMINISM_OFF,
)
from .scene_rows import _response_snippet
from .sparse_pgs import _get_pgs_solve_sparse_kernel

wp.set_module_options({"enable_backward": False, **WARP_DETERMINISM_OFF})


@dataclass(frozen=True)
class SceneFeatures:
    """The per-row features of the scene's sweep kernels: build constants, fixed at construction.

    ``scene_solve.SceneSolve`` builds its ``features`` from the solver's options, and every sweep factory
    (:func:`get_sweep_sides_kernel` and the rest) and :func:`scene_kernel_snippets` takes them. With
    every feature off, ``SceneFeatures()``, a factory returns its default kernel, whose stubs and
    native sources are the ones the scene tripwire pins character for character. With any feature on it
    returns a kernel of its one featured definition: the default's arguments followed by every
    feature's, in field order (``friction_start: int``, ``row_w: wp.array2d[float32]``,
    ``row_bound: wp.array2d[float32]``, then the four drive descriptors ``drive_target_vel_bias``,
    ``drive_vel_multiplier``, ``drive_impulse_multiplier`` and ``drive_max_impulse``, each
    ``wp.array2d[float32]``, then ``side_late_begin`` and ``side_late_end``, each ``wp.array[int32]``, then
    ``drive_vel_limit: wp.array2d[float32]`` and ``freeze_drives: int``, then ``row_gamma: wp.array2d[float32]``, then
    contact torsion's ``torsion_radius: wp.array2d[float32]``, ``torsion_group: wp.array2d[int32]``,
    ``torsion_member_range: wp.array2d[int32]``, ``torsion_member_rows``, ``side_torsion_begin``, ``side_torsion_end``
    and ``torsion_color_start``, each ``wp.array[int32]``, then the friction modes' ``row_friction_mode:
    wp.array2d[int32]``; the depenetration cap has none), and a name with one token per feature that is on (``_fpi``,
    ``_reg``, ``_jf``, ``_drv``, ``_late``, ``_fvc``, ``_dpc``, ``_tor``, ``_cmp``, the friction mode's ``_bis``,
    ``_bisx`` or ``_cn``, and ``_pat``; a feature's arguments follow the earlier features', compliance's before
    torsion's, which came later, while the tokens are in field order; friction patches have none). A feature that is off
    adds no line to the snippet, and its arguments get placeholders at launch (``SceneSolve._feature_inputs``). The
    features are part of the factories' cache keys, so solvers with equal features share their kernels. A field the
    sweep kernels do not implement (:data:`_FEATURE_TOKENS` does not name it) keeps its default: the scene mode rejects
    its options, and a factory refuses it.

    A feature of the row preparation alone, which adds no line to the sweeps (the depenetration cap), still makes
    the sweep factories return their featured definitions: a model whose only feature it is compiles the featured
    kernels in place of the default ones, the default's text with every feature's placeholder argument, under cache
    entries of their own. The cost is compile time and kernel-cache churn; the sweeps' arithmetic is the default's.

    - ``friction_position_iterations``: ``contact_friction_position_iterations >= 0``.
      ``friction_start`` is the first of the launch's sweeps that solves the tangent pairs: Steel's
      ``_friction_start_iteration`` less the iteration the launch starts at. Earlier sweeps skip every
      tangent pair, whose impulses stay zero, and the stationary exit waits for it. Placeholder ``0``.
    - ``regularize``: ``contact_regularization > 0``. ``row_w`` ``[1, row_max]`` holds each row's
      weight ``w`` (``scene_solve.prepare_scene_rows``): ``1 / (1 + g)`` for a penetrating contact whose
      rebound does not fire, 1 for every other row. A row with ``w < 1`` takes Steel's dense
      matrix-free update, ``old + omega * (-residual / diag * w - (1 - w) * old)``, which also relaxes
      its impulse toward zero; a row with ``w == 1`` keeps the rigid update ``old - omega * residual /
      diag``. At the default relaxation ``omega = 1`` the two agree on a row of weight 1 to the bit,
      but for the sign of a zero (Steel's ``old + 1 * (-r / d)`` is ``old - (1 * r) / d``), so the
      branch matters only at ``omega != 1``, where it keeps the rows of weight 1 stepping exactly as
      without regularization (Steel's formula would round ``omega * (-r / d)`` instead).
      Placeholder an empty ``[1, 0]`` array.
    - ``joint_friction``: ``enable_joint_friction``. ``row_bound`` ``[1, row_max]`` holds each
      joint-friction row's bound ``f dt`` (written by its row builder,
      ``scene_rows.get_sparse_joint_friction_rows_kernel``), and the sweeps clamp the row's impulse to the box
      ``[-bound, bound]`` after its update, Steel's projection (``joint_friction_projection`` of
      ``_get_pgs_solve_mf_gs_kernel``). Rows of every other type ignore it. Placeholder an empty
      ``[1, 0]`` array.
    - ``drives``: ``drive_mode="rows"``. The four drive descriptors ``[1, row_max]`` (Steel's
      ``_drive_target_vel_bias``, ``_drive_vel_multiplier``, ``_drive_impulse_multiplier`` and
      ``_drive_max_impulse``, written by ``scene_solve.scene_drive_descriptors``) give a drive row's impulse
      update, which replaces the row's rigid or regularized update: Steel's force-drive update
      ``old * impulse_multiplier + (J v) * vel_multiplier + target_vel_bias``, clamped to
      ``[-max_impulse, max_impulse]``, not relaxed by ``omega`` (``drive_update`` of
      ``_get_pgs_solve_mf_gs_kernel``), with ``J v`` the row's dot product plus its incident velocity. Rows of every
      other type ignore them. ``freeze_drives`` is ``1`` in the velocity-only iterations under
      ``velocity_drive_mode="freeze"``, where a drive row is left as it is (Steel's ``freeze_drive_rows``), and ``0``
      otherwise;
      only the drive rows' branch reads it. Placeholders four empty ``[1, 0]`` arrays and ``0``.
    - ``late_rows``: ``enable_joint_velocity_limits``, or free-body velocity limits (the solver's scene late rows). The
      late rows (every articulation's joint velocity-limit rows and every free body's velocity-limit rows, its side's
      late unit) follow the tail in the row order; ``side_late_begin`` and ``side_late_end`` (``[side_count]``,
      ``SceneSolve.side_late_begin``) hold each side's. They run in phase C, after every other row of their side in each
      iteration (Steel's "last word": its velocity limits last in each sweep of ``_get_pgs_solve_mf_gs_kernel``): the
      sides kernel's isolated pass sweeps a side's one-sided rows and then its late rows in each sweep, its third pass
      (``coupled_pass == 2``) sweeps the coupled sides' late rows once (after phase B and A), the order oracle walks
      them last in each sweep, and the serial walk reaches them last in the row order. A velocity-limit row takes
      Steel's stateless update (``dense_velocity_limit_pass``, and the free-body rows' pass, which multiplies by the
      stored inverse diagonal): the impulse of the current overshoot, ``-residual / diag`` when the residual is negative
      and ``0`` otherwise, not relaxed by ``omega``; the row's impulse is that impulse, and it is the velocity change
      applied, through the row's free-body response for a free body's rows. Placeholders two empty ``[0]`` arrays.
    - ``fused_clamp``: ``fuse_joint_velocity_limits`` in effect (the solver's fused velocity limits: drive rows and
      joint velocity limits). ``drive_vel_limit`` ``[1, row_max]`` (Steel's ``_drive_vel_limit``, written by the drive
      rows' builder) holds each drive row's DOF velocity limit. After a side's late rows, phase C clamps each of its
      drive rows statelessly, as Steel's ``fused_velocity_clamp_pass``: with ``J v`` the row's dot product plus its
      incident velocity, an overshoot ``|J v| > qdot_max`` applies ``(clamp(J v, -qdot_max, qdot_max) - J v) / diag`` to
      the velocities, the drive impulse unchanged. The serial walk of every row (``RANGE_ALL``) clamps its drive rows
      after each walk; its walks of the two-sided ranges, which hold no drive row, do not. Needs ``late_rows`` and
      ``drives``. Placeholder an empty ``[1, 0]`` array.
    - ``depenetration_cap``: a free body with a finite positive ``rigid_body_max_depenetration_velocity``
      (the solver's scene depenetration cap). The row preparation bounds the Baumgarte bias of the penetrating
      contacts of Steel's free-body path by their bodies' cap (``scene_solve.cap_scene_depenetration``, after
      ``scene_solve.prepare_scene_rows``, whose text is unchanged), which takes the per-body caps. The sweeps read no
      cap: the feature adds no argument and no line to them, and their featured definition (``_dpc``) has the
      default's text.
    - ``compliance``: ``contact_compliance``. The row preparation assembles each compliant contact's normal row on
      the device after the depenetration cap (``scene_compliance.prepare_scene_compliance``):
      ``row_gamma`` ``[1, row_max]`` holds its ``gamma`` (``0`` on every other row), its ``rhs`` becomes ``bias -
      target`` and its diagonal ``diag + (gamma - cfm)``. The sweeps add ``gamma * lambda`` with the row's impulse
      to its right-hand side at each visit, before the update (Steel's compliant residual in its sweeps,
      ``contact_compliance`` of ``_get_pgs_solve_mf_gs_kernel``), so the normal update is ``delta lambda = -(w +
      gamma lambda) / (diag + gamma)`` with ``gamma`` folded into the stored diagonal once; the division is the
      default's. Under ``scene_phase_order="ABA"`` a coupled side's one-sided rows are visited twice per iteration,
      and the second visit reads the impulse of the first, the exact Gauss-Seidel step of the compliant system.
      Placeholder an empty ``[1, 0]`` array.
    - ``torsion``: contact torsion (a shape with a positive ``steel:min_torsional_patch_radius``). The row preparation
      groups the contacts on the device (``group_scene_torsion``) and writes one spin row per group (row type
      ``TORSION``, ``J = (0, n)`` with the group's lead normal) after every unit's rows
      (``SceneSolve.prepare_torsion``). The sweeps run Steel's C1 on a spin row (``contact_torsion``'s
      ``torque_sweep_source``): its group's anchor tangent pairs, whose normal rows
      ``torsion_member_rows`` lists (``[torsion_member_range[row, 0], torsion_member_range[row, 1])``, each a normal
      row shifted left by one with bit 0 set for an anchor), are rescaled into ``mu * sum max(lambda_n, 0)`` when
      their magnitudes pass it, each rescaled pair's change applied through its rows' responses, and the spin row's
      update is clamped to ``torsion_radius * max(mu * sum lambda_n - sum |t|, 0)``, 0 for a row the velocity pass
      retired (``torsion_group[row] == -2``). The spin rows are late rows: in phase C each side sweeps its one-sided
      spin rows (``[side_torsion_begin, side_torsion_end)``), then the two-sided ones run by the color of their
      group's representative contact unit (``torsion_color_start``, :func:`get_sweep_torsion_kernel`), then the
      velocity limits, Steel's order (its torsion sweep, then its velocity limits, in ``_get_pgs_solve_mf_gs_kernel``).
      Without ``late_rows`` the sweeps still run phase C, on empty late ranges. The rescale is Steel's guard for
      pairs projected into their cone before later rows lowered the group's load, as a friction patch region's
      normal-only members do when swept after its anchors. Placeholders empty arrays.
    - ``friction_mode``: Steel's ``friction_mode`` when it is not ``"fixed_normal"`` (the field's default, off):
      ``"bisection"``, ``"bisection_desaxce"`` or ``"coulomb_newton"`` (tokens ``_bis``, ``_bisx``, ``_cn``), Steel's
      experimental contact updates of its matrix-free solve (its ``dense_friction_update`` and
      ``mf_friction_update`` under these modes). ``row_friction_mode`` ``[1, row_max]`` holds, on each contact's first
      tangent row, the update its contact takes, which ``scene_solve.prepare_scene_friction_modes`` decides as Steel
      does (0 on every other row):
      :data:`SCENE_FRICTION_FIXED_NORMAL` keeps the tangent pair's update, :data:`SCENE_FRICTION_DENSE` takes the
      mode's update through Steel's dense text (a side is an articulation with response DOFs that is not a free body)
      and :data:`SCENE_FRICTION_FREE` through its free-body text (every side a free body or without response DOFs).
      Under ``"bisection"`` and ``"bisection_desaxce"`` every contact with its tangent pair takes Steel's bisection
      (the de Saxce bias ``mu |c_T|`` on the free-body ones' normal target only); under ``"coulomb_newton"`` the
      free-body ones take Steel's Coulomb-Newton solve and the articulated ones keep the pair's update. As in Steel the
      contact's three rows are solved together at its first tangent row's visit (the normal row's visit before it
      keeps its own update): the update writes the normal and both tangent impulses and applies their changes through
      the three rows' responses in Steel's order (normal, second tangent, first tangent), and the stationary exit sees
      each change. Under every mode a tangent pair whose first row has no diagonal is skipped, as Steel's other modes
      drop the pairs' diagonal guard. The laws are Steel's statement for statement (the guard test compares them with
      Steel's sources, :func:`_mirrored_lines`) but round every operation on its own, where Steel's texts compile
      with FMA fusion (``fuse_fp=True``): bitwise agreement with Steel is not intended. Placeholder an empty ``[1, 0]``
      array.
    - ``patches``: persistent friction patches, ``friction_anchor_beta > 0`` (Steel's matrix-free default). The row
      build (``scene_solve.build_scene_rows``) builds Steel's patches before it routes the contacts, so a region's
      anchors get three rows each and its other members their normal row alone (Steel's allocation), writes the anchors'
      tangent rows at the anchor points (the response kernel's ``patches`` variant) and links each region's normal rows
      into a ring through their ``row_parent`` (``link_patch_rows``, which also divides the anchors' ``row_mu`` by the
      region's anchors with rows). A tangent pair's normal load is then its region's: the sum of the ring's normal
      impulses from its own normal row in ring order, the pooled load of Steel's sparse sweep
      (``sparse_pgs._get_pgs_solve_sparse_kernel``), ``mf_friction_radius`` and ``dense_friction_update``, whose radius
      is ``max(row_mu * load, 0)`` (``_PATCH_LOAD_CUDA``); no other line changes and no argument is added, the ring
      being in ``rows_parent`` and ``impulses``. Steel's free-body text skips a pair whose disk is zero and whose
      impulses are zero (``mf_friction_radius``); the scene's one text, which also serves Steel's dense rows (no such
      skip), does not, as by then the pair's dot products are formed and the rest of such a visit changes no velocity:
      both changes are zero, so no response is applied and the stationary exit sees no change (only a stored ``-0``
      impulse becomes ``+0``). Steel's constructor keeps patches from the friction modes and compliance. Race-freedom: a
      tangent pair reads its region's other normal impulses, which other units hold. A region's contacts are of one body
      pair, so their units have the same sides: in a side's one-sided pass they are the side's own units, which its
      lanes sweep one row at a time; in phase B they are two-sided units joining the same two sides, which the coloring
      never puts in one color, so each is written in another color's launch or before the block kernel's
      ``__syncthreads`` between colors; the tail and the oracles walk serially. No ring member's impulse is written
      while a pair reads it.
    - ``bilateral``: reserved; it stays off. The mimic and connect rows need no build flag: the rigid update
      clamps the contact and limit rows alone (``_UNILATERAL_CLAMP_CUDA``), and the featured branches act on
      their own row types, so a mimic or connect row takes Steel's unclamped update in the default kernels, and
      ``scene_solve.prepare_scene_rows`` already gives it the two-sided Baumgarte term.
    """

    friction_position_iterations: bool = False
    regularize: bool = False
    joint_friction: bool = False
    drives: bool = False
    late_rows: bool = False
    fused_clamp: bool = False
    depenetration_cap: bool = False
    torsion: bool = False
    compliance: bool = False
    friction_mode: str = "fixed_normal"
    patches: bool = False
    bilateral: bool = False


_FEATURE_TOKENS = {
    "friction_position_iterations": "fpi",
    "regularize": "reg",
    "joint_friction": "jf",
    "drives": "drv",
    "late_rows": "late",
    "fused_clamp": "fvc",
    "depenetration_cap": "dpc",
    "torsion": "tor",
    "compliance": "cmp",
    "friction_mode": {"bisection": "bis", "bisection_desaxce": "bisx", "coulomb_newton": "cn"},
    "patches": "pat",
}
"""The per-row features the sweep kernels implement, in :class:`SceneFeatures`' field order, with each
one's token in a featured kernel's name (a string-valued feature's by its value)."""


def _feature_suffix(features: SceneFeatures) -> str:
    """A featured kernel's name suffix: ``_`` and the token of each feature that is on, in field order.

    ``""`` with every feature off, the default kernels. Raises :class:`NotImplementedError` for a
    feature that is on and that the sweep kernels do not implement yet, and :class:`ValueError` for
    ``fused_clamp`` without ``late_rows`` and ``drives`` and for an unknown ``friction_mode``.
    """
    default = SceneFeatures()
    on = [field.name for field in fields(features) if getattr(features, field.name) != getattr(default, field.name)]
    missing = [name for name in on if name not in _FEATURE_TOKENS]
    if missing:
        raise NotImplementedError(f"the scene sweep kernels do not implement {', '.join(missing)} yet")
    if features.fused_clamp and not (features.late_rows and features.drives):
        raise ValueError(
            "the fused velocity clamp (fused_clamp) runs in phase C on drive rows: it needs late_rows and drives"
        )
    modes = _FEATURE_TOKENS["friction_mode"]
    if features.friction_mode != default.friction_mode and features.friction_mode not in modes:
        raise ValueError(
            f"friction_mode must be 'fixed_normal' or one of {list(modes)}, got {features.friction_mode!r}"
        )
    tokens = [_FEATURE_TOKENS[name] for name in on]
    return "".join(f"_{modes[features.friction_mode] if token is modes else token}" for token in tokens)


# --- The friction modes' updates, flagged per contact -------------------------------------------

SCENE_FRICTION_FIXED_NORMAL = 0
"""``SceneSolve.row_friction_mode`` of a row whose contact keeps the tangent pair's update: every row but a contact's
first tangent row, and that row of an articulated contact under ``friction_mode="coulomb_newton"``."""
SCENE_FRICTION_DENSE = 1
"""The friction mode's update through Steel's dense text: an articulated contact (a side is an articulation with
response DOFs that is not a free body) under ``"bisection"`` or ``"bisection_desaxce"``."""
SCENE_FRICTION_FREE = 2
"""The friction mode's update through Steel's free-body text: a contact on Steel's free-body path (every side a free
body or without response DOFs, ``scene_units._scene_contact_on_free_path``)."""


# --- The sweep ---------------------------------------------------------------------------------

RANGE_ALL = 0
"""Index of ``SceneSolve.row_ranges``' entry for every kept row, ``[0, rows)``."""
RANGE_TWO_SIDED = 1
"""Index of the entry for the two-sided units' rows, the colors and the tail:
``[first two-sided row, first late row)``."""
RANGE_TAIL = 2
"""Index of the entry for the tail's rows, the two-sided units past the color cap:
``[first tail row, first late row)``."""
RANGE_LATE = 3
"""Index of the entry for the late rows, every late unit's (joint velocity limits): ``[first late row, rows)``.

Without late rows the first late row is ``rows``, so the two-sided and tail ranges end at ``rows`` and this one is
empty."""

SCENE_PASS_ISOLATED = 0
SCENE_PASS_COUPLED = 1
SCENE_PASS_LATE = 2
SCENE_PASS_TORSION = 3
"""The passes of the sides kernel (its ``coupled_pass``): every iteration of the isolated sides (phase A, and phase C
with late rows, in each sweep), one phase-A sweep of the coupled sides, and, with late rows, one phase-C sweep of the
coupled sides (``SceneSolve._sweep_late``). With contact torsion the isolated sides' sweeps also run their
one-sided spin rows between phases A and C, and the coupled sides' phase C splits in two around the two-sided spin rows:
the fourth pass sweeps their one-sided spin rows (C1), the third their late rows (C2)."""

SCENE_SIDE_LANES = {"thread": 1, "lanes8": 8, "lanes16": 16, "lanes32": 32}
"""Lanes per side or unit of each sweep granularity (``scene_side_sweep``, ``scene_unit_sweep``). A row's entries are
spread over the lanes (lane g holds entries g, g + lanes, ...), so wider groups load a row with fewer passes; with
Steel's sums every granularity reduces in Steel's 32-lane order."""
SCENE_SUM_ORDERS = ("serial", "steel")
"""Summation orders of a row's dot products (``scene_sum_order``)."""
SCENE_SWEEPS = ("phases", "serial")
"""Schedules of the scene sweep (``scene_sweep``)."""
SCENE_PHASE_ORDERS = ("BA", "AB", "ABA")
"""Orders of the phases in each iteration of the coupled sides (``scene_phase_order``), the default first.

``"A"`` is one sweep of the coupled sides' one-sided rows, ``"B"`` one sweep of the two-sided units. With late rows
each iteration ends with phase C, one sweep of the coupled sides' late rows (B, A, C under ``"BA"``).
"""
SCENE_UNIT_LANES = SCENE_SIDE_LANES
"""Lanes per two-sided unit of each phase-B granularity (``scene_unit_sweep``)."""
SCENE_RESPONSE_KERNELS = ("generic", "specialized")
"""Contact response kernels of the row build (``scene_response_kernel``), the default first."""
SCENE_COLOR_LOOPS = ("while_block", "while", "unroll", "block", "serial")
"""Schedules of the two-sided colors in each iteration (``scene_color_loop``), the default first."""
SCENE_COLOR_THREADS_MAX = 65536
"""Threads of the color kernel's fixed grid; its work items stride over a color's units."""
SCENE_BLOCK_COLOR_MAX = 64
"""Units of the largest color the default schedule leaves to the single block: up to this size one block's pass
over a color costs less than a conditional round of the full grid (measured on one GPU)."""
SCENE_BLOCK_THREADS = 256
"""Threads of the single-block kernel, one block."""
SUM_TREE_SUPPORT_MAX = 32
"""Widest row whose Steel-order sums are reproduced: one entry per lane of Steel's warp."""
SCENE_SWEEP_BLOCK_DIM = 64
"""Threads per block of the sweep kernels, as Steel's sparse sweep (lane groups stay inside a warp)."""

# The impulse update, shared by the thread and the lane-group loops (the impulse update of Steel's sparse sweep,
# ``sparse_pgs._get_pgs_solve_sparse_kernel``), with the dot
# products already reduced: the tangent pair's, then a normal or limit row's (``_RIGID_UPDATE_CUDA``).
# Its named parts, in the order :func:`_impulse_update` joins them: the declarations (the row's
# right-hand side ``_IMPULSE_RHS_CUDA``, then its residual and impulses ``_IMPULSE_DECLS_CUDA``), the
# tangent pair's branch and the opening of the row branch
# (``_PAIR_BRANCH_CUDA``), the row's update, the clamps that close the row branch
# (``_UNILATERAL_CLAMP_CUDA``), then the velocity change of the row's response (``_IMPULSE_CHANGE_CUDA``).
_IMPULSE_RHS_CUDA = """
            const float rhs = rows_rhs.data[row] + rows_incident.data[row];
"""
_IMPULSE_DECLS_CUDA = """            const float residual = dot + rhs;
            const float old_impulse = impulses.data[row];
            float new_impulse = old_impulse;
            float sibling_change = 0.0f;
            float new_sibling = 0.0f;
"""
# Contact compliance (``SceneFeatures.compliance``): a compliant normal row's right-hand side gains ``gamma * lambda``
# with the row's impulse at each visit, before its update, as Steel's sweeps add it to a compliant contact's residual
# (``contact_compliance`` in ``_get_pgs_solve_mf_gs_kernel``): its residual is ``J v + bias + gamma lambda``, here
# ``(rhs + gamma lambda) + J v_hat`` and then ``Z delta``; ``row_gamma`` is 0 on every other row. It takes
# ``_IMPULSE_RHS_CUDA``'s place. The update divides by the stored diagonal, into which the row preparation folded gamma.
_COMPLIANT_RHS_CUDA = """
            // Compliance: gamma * lambda_old joins the row's rhs (Steel's compliant residual); gamma is 0 on every
            // other row and already in the row's diagonal.
            const float rhs = rows_rhs.data[row] + row_gamma.data[row] * impulses.data[row] + rows_incident.data[row];
"""
# A tangent pair's normal load, the second line of the pair branch: its normal row's impulse. Friction patches put
# their region's ring sum in its place (``_PATCH_LOAD_CUDA``).
_NORMAL_LOAD_CUDA = """                const float normal_load = impulses.data[parent];
"""
_PAIR_BRANCH_CUDA = (
    """            if (friction) {
"""
    + _NORMAL_LOAD_CUDA
    + """                const float radius = fmaxf(rows_mu.data[row] * normal_load, 0.0f);
                const float old_sibling = impulses.data[sibling];
                float2 pair = make_float2(0.0f, 0.0f);
                if (radius > 0.0f) {
                    const float sibling_rhs = rows_rhs.data[sibling] + rows_incident.data[sibling];
                    const float sibling_residual = sibling_dot + sibling_rhs;
                    pair = friction_pair_candidate(denominator, cross, rows_diag.data[sibling], residual,
                        sibling_residual, old_impulse, old_sibling, radius, omega);
                }
                const float magnitude = sqrtf(pair.x * pair.x + pair.y * pair.y);
                const float scale = magnitude > radius ? radius / magnitude : 1.0f;
                new_impulse = pair.x * scale;
                new_sibling = pair.y * scale;
                sibling_change = new_sibling - old_sibling;
            } else {
"""
)
# Friction patches (``SceneFeatures.patches``): a tangent pair's normal load is its region's, the sum of the impulses
# of the region's normal rows, which link_patch_rows rings through their parents (a point contact's normal row has
# parent -1, a lone member's itself): the pooled load of Steel's sparse sweep (its articulated rows' walk in
# ``sparse_pgs._get_pgs_solve_sparse_kernel``, which the guard test compares with this text), the walk from the pair's
# own normal row in ring order. ``rows_mu`` is already divided by the region's anchors with rows (link_patch_rows). It
# takes ``_NORMAL_LOAD_CUDA``'s place.
_PATCH_LOAD_CUDA = """\
                // Friction patches: the region's normal rows ring through their parents (a point contact's parent is
                // -1, a lone member's itself); its anchors share the pooled load.
                float normal_load = impulses.data[parent];
                for (int next = rows_parent.data[parent]; next >= 0 && next != parent; next = rows_parent.data[next])
                    normal_load += impulses.data[next];
"""
_RIGID_UPDATE_CUDA = """                new_impulse = old_impulse - omega * residual / denominator;
"""
# The unilateral clamp of the contact and limit rows, which closes the row branch.
_UNILATERAL_CLAMP_CUDA = """                if (row_type == CONTACT || row_type == LIMIT)
                    new_impulse = fmaxf(new_impulse, 0.0f);
            }
"""
# The velocity change of the row's response: its impulse's change (``_IMPULSE_CHANGE_CUDA``).
_IMPULSE_CHANGE_CUDA = """            const float change = new_impulse - old_impulse;
"""

# Contact regularization (``SceneFeatures.regularize``): Steel's dense matrix-free update of a weighted
# row (``_get_pgs_solve_mf_gs_kernel``; its free-body rows weigh the same way but multiply by
# the stored inverse diagonal mf_diag instead of dividing), where ``row_w`` holds the row's weight:
# ``1 / (1 + g)`` for a penetrating contact, 1 for every other row, which keeps the rigid update. At
# omega = 1 Steel's update of a weight-1 row is the rigid one (but for a zero's sign), so the branch
# matters only at omega != 1.
_REGULARIZED_UPDATE_CUDA = """\
                // A row of weight w < 1 also relaxes its impulse toward zero (Steel's matrix-free update).
                const float w = row_w.data[row];
                if (w < 1.0f) {
                    const float relaxed = -residual / denominator * w - (1.0f - w) * old_impulse;
                    new_impulse = old_impulse + omega * relaxed;
                } else {
                    new_impulse = old_impulse - omega * residual / denominator;
                }
"""


# Joint friction (``SceneFeatures.joint_friction``): after the update, a joint-friction row's impulse is
# clamped to its box [-f dt, f dt], Steel's projection (``joint_friction_projection`` in
# ``_get_pgs_solve_mf_gs_kernel``, after the unilateral clamp); ``row_bound`` holds the bound, where Steel's dense rows
# keep it in mu. It takes the unilateral clamp's place, closing the row branch.
_BOX_CLAMP_CUDA = """                if (row_type == CONTACT || row_type == LIMIT)
                    new_impulse = fmaxf(new_impulse, 0.0f);
                else if (row_type == JOINT_FRICTION)
                    // The joint friction box [-f dt, f dt]: row_bound holds the bound.
                    new_impulse = fminf(fmaxf(new_impulse, -row_bound.data[row]), row_bound.data[row]);
            }
"""


# Drive rows (``SceneFeatures.drives``): Steel's dense matrix-free update of a drive row (``drive_update`` in
# ``_get_pgs_solve_mf_gs_kernel``), its first branch after the rigid or regularized update, which it replaces: the
# force-drive update from the row's descriptors (scene_drive_descriptors), clamped to its maximum impulse
# and not relaxed by omega. Steel's J v (jv) is here the row's dot product plus its incident velocity J v_hat. The
# contact/limit clamp and the joint friction box follow as Steel's later branches, through the "else". Before the
# update, the velocity-only iterations under velocity_drive_mode="freeze" (``freeze_drives``) leave the row as it
# is, Steel's freeze_drive_rows skip; a clamp pass never reaches this branch.
_DRIVE_UPDATE_CUDA = """\
                if (row_type == JOINT_TARGET) {
                    // velocity_drive_mode="freeze": the velocity-only iterations skip the drive rows (Steel's skip).
                    if (freeze_drives && row_type == JOINT_TARGET) continue;
                    // The force-drive update (Steel's drive row); not relaxed by omega.
                    const float jv = dot + rows_incident.data[row];
                    const float max_impulse = drive_max_impulse.data[row];
                    new_impulse = old_impulse * drive_impulse_multiplier.data[row] + jv * drive_vel_multiplier.data[row]
                        + drive_target_vel_bias.data[row];
                    new_impulse = fminf(fmaxf(new_impulse, -max_impulse), max_impulse);
                } else
"""


# Late rows (``SceneFeatures.late_rows``): phase C's rows are stateless. A velocity-limit row (Steel's
# dense_velocity_limit_pass) applies the impulse of the current overshoot,
# -residual / diag when the residual is negative, unrelaxed, and stores that impulse; the fused clamp of a drive row
# (``SceneFeatures.fused_clamp``) applies a velocity change and leaves the drive impulse alone. Both apply
# ``stateless_change`` rather than new_impulse - old_impulse (``_STATELESS_CHANGE_CUDA``).
_STATELESS_DECLS_CUDA = """\
            // A stateless row of phase C applies a velocity change of its own, not its impulse's change.
            bool stateless = false;
            float stateless_change = 0.0f;
"""
_VELOCITY_LIMIT_UPDATE_CUDA = """\
                if (row_type == VELOCITY_LIMIT) {
                    // Steel's stateless velocity limit: only the current overshoot's impulse, not relaxed by omega.
                    new_impulse = residual < 0.0f ? -residual / denominator : 0.0f;
                    stateless = true;
                    stateless_change = new_impulse;
                } else
"""
_STATELESS_CHANGE_CUDA = """            const float change = stateless ? stateless_change : new_impulse - old_impulse;
"""

# The fused velocity clamp (``SceneFeatures.fused_clamp``): in a clamp pass (``clamp_pass``, which visits drive rows
# alone) a drive row whose J v passes its DOF's velocity limit gets the velocity change that returns it to the limit,
# Steel's fused_velocity_clamp_pass; the drive impulse is left unchanged. It is the
# first branch after the rigid or regularized update, before the force-drive update it stands in for.
_FUSED_CLAMP_UPDATE_CUDA = """\
                if (clamp_pass) {
                    // Steel's fused clamp of a driven DOF: the velocity returns to its limit, the drive impulse stays.
                    const float jv = dot + rows_incident.data[row];
                    const float qdot_max = drive_vel_limit.data[row];
                    new_impulse = old_impulse;
                    stateless = true;
                    if (fabsf(jv) > qdot_max)
                        stateless_change = (fminf(fmaxf(jv, -qdot_max), qdot_max) - jv) / denominator;
                } else
"""
# A clamp pass visits the drive rows alone (after the row type, as the friction gate).
_CLAMP_PASS_GATE_CUDA = """            if (clamp_pass && row_type != JOINT_TARGET) continue;
"""

# Contact torsion (``SceneFeatures.torsion``): C1, Steel's torque sweep (``contact_torsion.torque_sweep_source``), on
# a spin row. Its head (``_TORSION_HEAD_CUDA``, before the row's bases) rescales the group's tangent pairs and forms
# the spin row's bound, ``torsion_bound`` (``_torsion_budget``); the row's update is then Steel's ``old - omega *
# residual / diag`` (the rigid update, the regularized one at weight 1), clamped to the bound. A spin row without a
# diagonal skips its update, as every row does: Steel's ``trial = 0`` there keeps its impulse at the 0 it starts each
# step from, after the same rescale.
_TORSION_HEAD_CUDA = """\
            // C1: a spin row first rescales its group's tangent pairs into the budget and forms its bound.
            float torsion_bound = 0.0f;
            if (row_type == TORSION) torsion_bound = {budget};
"""
_TORSION_UPDATE_CUDA = """\
                if (row_type == TORSION) {
                    // C1: the spin row's update, clamped to its bound (Steel's torque_sweep_source).
                    const float trial = new_impulse;
                    new_impulse = fminf(fmaxf(trial, -torsion_bound), torsion_bound);
                } else
"""

# The rounding shim, the host shim's companion: on the CPU the round-to-nearest intrinsics that the friction modes'
# laws and contact torsion's budget name are the plain float operations, which round to nearest as CUDA's do. Each is
# its own lambda, so no two fuse into an FMA (the host compiler contracts within one expression alone), as CUDA's
# intrinsics never fuse. It follows :data:`_HOST_SHIM_CPP` at the head of a sweep snippet with a friction mode or
# contact torsion (``_sweep_snippet``), and of no other; on CUDA it expands to nothing.
_HOST_ROUNDING_SHIM_CPP = """#if !defined(__CUDA_ARCH__)
    static constexpr auto __fadd_rn = [](float a, float b) { return a + b; };
    static constexpr auto __fsub_rn = [](float a, float b) { return a - b; };
    static constexpr auto __fmul_rn = [](float a, float b) { return a * b; };
    static constexpr auto __fdiv_rn = [](float a, float b) { return a / b; };
    static constexpr auto __fsqrt_rn = [](float a) { return sqrtf(a); };
#endif
"""

# Friction modes (``SceneFeatures.friction_mode``): Steel's other contact updates of its matrix-free solve, which
# solve a contact's normal and tangent impulses together at its first tangent row's visit (``dense_friction_update``
# and ``mf_friction_update`` of ``_get_pgs_solve_mf_gs_kernel`` under these modes). Their device functions precede the
# row loops (``_friction_mode_funcs``): ``contact_block`` gathers the inputs of both of Steel's texts,
# ``contact_bisection`` is Steel's bisection (one law in ``_DENSE_BISECTION_FRICTION_SOURCE`` and
# ``_MF_BISECTION_FRICTION_SOURCE``) and ``contact_coulomb_newton`` its Coulomb-Newton solve of a free-body contact
# (``_MF_COULOMB_NEWTON_FRICTION_SOURCE``). Each law is Steel's statement for statement, with Steel's names, from
# Steel's ``if (G_nn >= 1.0e-20f)`` on (the guard test compares the two, :func:`_mirrored_lines`), except that every
# product, sum, quotient and root rounds on its own (``__fmul_rn`` and the rest, no contraction), so each inlined copy
# (the sides, the colors, the block, the oracles) rounds alike. Steel compiles its texts with Warp's FMA fusion on
# (``fuse_fp=True``), while these explicit round-to-nearest intrinsics never fuse, so the two round differently on
# equal inputs: the expression order is Steel's, bitwise agreement with Steel is not intended.
_FRICTION_MODE_BLOCK_CUDA = """
    // A contact's rows [normal, t1, t2] (rows normal to normal + 2, which share their contact's support): u = J v
    // of each (J v_hat + Z delta, a tangent's with its right-hand side) and g = J Y of each pair (nn, nt1, nt2, t1t1,
    // t1t2, t2t2; Y a row's response: its free-body response over its free prefix, its factor values elsewhere),
    // summed entry by entry. Steel's dense text (an articulated contact) starts a tangent's sum at its right-hand
    // side; its free-body text (free_order) adds it after J v and sums g in a second loop, whose sums run in the same
    // order.
    const auto contact_block = [&](int normal, bool free_order, float* u, float* g) {
        const int block_base = normal * SUPPORT;
        const int block_free_base = normal * free_width;
        const int block_prefix = free_response ? rows_free_prefix.data[normal] : 0;
        u[0] = 0.0f;
        u[1] = 0.0f;
        u[2] = 0.0f;
        if (!free_order) {
            u[0] = rows_incident.data[normal];
            u[1] = __fadd_rn(rows_rhs.data[normal + 1], rows_incident.data[normal + 1]);
            u[2] = __fadd_rn(rows_rhs.data[normal + 2], rows_incident.data[normal + 2]);
        }
        for (int m = 0; m < 6; ++m) g[m] = 0.0f;
        for (int k = 0; k < SUPPORT; ++k) {
            const int index = rows_dof.data[block_base + k];
            if (index < 0) continue;
            const float velocity = delta.data[index];
            float j[3], y[3];
            for (int r = 0; r < 3; ++r) {
                j[r] = rows_factor.data[block_base + r * SUPPORT + k];
                y[r] = k < block_prefix ? rows_free_response.data[block_free_base + r * free_width + k] : j[r];
                u[r] = __fadd_rn(u[r], __fmul_rn(j[r], velocity));
            }
            g[0] = __fadd_rn(g[0], __fmul_rn(j[0], y[0]));
            g[1] = __fadd_rn(g[1], __fmul_rn(j[0], y[1]));
            g[2] = __fadd_rn(g[2], __fmul_rn(j[0], y[2]));
            g[3] = __fadd_rn(g[3], __fmul_rn(j[1], y[1]));
            g[4] = __fadd_rn(g[4], __fmul_rn(j[1], y[2]));
            g[5] = __fadd_rn(g[5], __fmul_rn(j[2], y[2]));
        }
        if (free_order) {
            u[0] = __fadd_rn(u[0], rows_incident.data[normal]);
            u[1] = __fadd_rn(__fadd_rn(u[1], rows_incident.data[normal + 1]), rows_rhs.data[normal + 1]);
            u[2] = __fadd_rn(__fadd_rn(u[2], rows_incident.data[normal + 2]), rows_rhs.data[normal + 2]);
        }
    };
"""
# The head of a friction mode's law: Steel's names for the contact's inputs (u, g) and its old impulses (lambda),
# which receives the new ones; they stay the old ones without a normal coupling (G_nn < 1e-20).
_FRICTION_LAW_HEAD_CUDA = """\
        const float u_n = u[0], u_t1 = u[1], u_t2 = u[2];
        const float G_nn = g[0], G_nt1 = g[1], G_nt2 = g[2], G_t1t1 = g[3], G_t1t2 = g[4], G_t2t2 = g[5];
        const float old_lambda_n = lambda[0], old_lambda_t1 = lambda[1], old_lambda_t2 = lambda[2];
        float new_lambda_n = old_lambda_n, new_lambda_t1 = old_lambda_t1, new_lambda_t2 = old_lambda_t2;
"""
_FRICTION_LAW_TAIL_CUDA = """\
        lambda[0] = new_lambda_n;
        lambda[1] = new_lambda_t1;
        lambda[2] = new_lambda_t2;
    };
"""
_FRICTION_MODE_BISECTION_CUDA = (
    """
    // Steel's bisection: the normal impulse bisected in [0, hi] over 20 probes, the 2x2 tangent problem re-solved and
    // projected on the probe's cone at each, the root interpolated in the last bracket, and the tangents solved and
    // projected once more for it.
    const auto contact_bisection = [&](const float* u, const float* g, float target_vel_n, float mu, float* lambda) {
"""
    + _FRICTION_LAW_HEAD_CUDA
    + """        if (G_nn >= 1.0e-20f) {
            // At zero load the cone also zeroes the sliding pair.
            const float u_n_at_zero = __fsub_rn(__fsub_rn(__fsub_rn(u_n, __fmul_rn(G_nn, old_lambda_n)),
                __fmul_rn(G_nt1, old_lambda_t1)), __fmul_rn(G_nt2, old_lambda_t2));
            if (u_n_at_zero >= target_vel_n) {
                new_lambda_n = 0.0f;
                new_lambda_t1 = 0.0f;
                new_lambda_t2 = 0.0f;
            } else {
                float lo = 0.0f;
                float hi = fmaxf(__fmul_rn(old_lambda_n, 2.0f),
                    __fadd_rn(__fdiv_rn(__fsub_rn(target_vel_n, u_n), G_nn), old_lambda_n));
                hi = fmaxf(hi, 1.0f);
                float f_lo = __fsub_rn(u_n_at_zero, target_vel_n), f_hi = -1.0f;
                for (int _bi = 0; _bi < 20; _bi++) {
                    const float mid = __fmul_rn(0.5f, __fadd_rn(lo, hi));
                    const float d_n = __fsub_rn(mid, old_lambda_n);
                    const float ut1_eff = __fadd_rn(u_t1, __fmul_rn(G_nt1, d_n));
                    const float ut2_eff = __fadd_rn(u_t2, __fmul_rn(G_nt2, d_n));
                    const float det = __fsub_rn(__fmul_rn(G_t1t1, G_t2t2), __fmul_rn(G_t1t2, G_t1t2));
                    float d_t1 = 0.0f, d_t2 = 0.0f;
                    if (fabsf(det) > 1.0e-20f) {
                        d_t1 = __fdiv_rn(__fadd_rn(__fmul_rn(-ut1_eff, G_t2t2), __fmul_rn(ut2_eff, G_t1t2)), det);
                        d_t2 = __fdiv_rn(__fsub_rn(__fmul_rn(ut1_eff, G_t1t2), __fmul_rn(ut2_eff, G_t1t1)), det);
                    }
                    float trial_t1 = __fadd_rn(old_lambda_t1, d_t1);
                    float trial_t2 = __fadd_rn(old_lambda_t2, d_t2);
                    const float flimit = __fmul_rn(mu, mid);
                    const float tmag =
                        __fsqrt_rn(__fadd_rn(__fmul_rn(trial_t1, trial_t1), __fmul_rn(trial_t2, trial_t2)));
                    if (tmag > flimit && tmag > 1.0e-20f) {
                        const float sc = __fdiv_rn(flimit, tmag);
                        trial_t1 = __fmul_rn(trial_t1, sc);
                        trial_t2 = __fmul_rn(trial_t2, sc);
                    }
                    const float d_t1_actual = __fsub_rn(trial_t1, old_lambda_t1);
                    const float d_t2_actual = __fsub_rn(trial_t2, old_lambda_t2);
                    const float u_n_trial = __fadd_rn(__fadd_rn(__fadd_rn(u_n, __fmul_rn(G_nn, d_n)),
                        __fmul_rn(G_nt1, d_t1_actual)), __fmul_rn(G_nt2, d_t2_actual));
                    if (u_n_trial < target_vel_n) {
                        lo = mid;
                        f_lo = __fsub_rn(u_n_trial, target_vel_n);
                    } else {
                        hi = mid;
                        f_hi = __fsub_rn(u_n_trial, target_vel_n);
                    }
                }
                // Interpolate the root in the final bracket, whose width scales with the initial one.
                new_lambda_n = f_hi >= 0.0f
                    ? __fsub_rn(lo, __fdiv_rn(__fmul_rn(f_lo, __fsub_rn(hi, lo)), __fsub_rn(f_hi, f_lo)))
                    : __fmul_rn(0.5f, __fadd_rn(lo, hi));
                const float d_n_final = __fsub_rn(new_lambda_n, old_lambda_n);
                const float ut1_f = __fadd_rn(u_t1, __fmul_rn(G_nt1, d_n_final));
                const float ut2_f = __fadd_rn(u_t2, __fmul_rn(G_nt2, d_n_final));
                const float det_f = __fsub_rn(__fmul_rn(G_t1t1, G_t2t2), __fmul_rn(G_t1t2, G_t1t2));
                float d_t1_f = 0.0f, d_t2_f = 0.0f;
                if (fabsf(det_f) > 1.0e-20f) {
                    d_t1_f = __fdiv_rn(__fadd_rn(__fmul_rn(-ut1_f, G_t2t2), __fmul_rn(ut2_f, G_t1t2)), det_f);
                    d_t2_f = __fdiv_rn(__fsub_rn(__fmul_rn(ut1_f, G_t1t2), __fmul_rn(ut2_f, G_t1t1)), det_f);
                }
                new_lambda_t1 = __fadd_rn(old_lambda_t1, d_t1_f);
                new_lambda_t2 = __fadd_rn(old_lambda_t2, d_t2_f);
                const float flimit_f = __fmul_rn(mu, new_lambda_n);
                const float tmag_f = __fsqrt_rn(
                    __fadd_rn(__fmul_rn(new_lambda_t1, new_lambda_t1), __fmul_rn(new_lambda_t2, new_lambda_t2)));
                if (tmag_f > flimit_f && tmag_f > 1.0e-20f) {
                    const float sc_f = __fdiv_rn(flimit_f, tmag_f);
                    new_lambda_t1 = __fmul_rn(new_lambda_t1, sc_f);
                    new_lambda_t2 = __fmul_rn(new_lambda_t2, sc_f);
                }
            }
        }
"""
    + _FRICTION_LAW_TAIL_CUDA
)
_FRICTION_MODE_COULOMB_NEWTON_CUDA = (
    """
    // Steel's Coulomb-Newton solve of a free-body contact: the velocities free of the old impulses; an approaching
    // contact solves its Coulomb cone exactly for the tangential force s(x) of its tangent system shifted by x, x
    // bracketed by doubling and refined by Newton steps kept inside the bracket (bisection otherwise), and its normal
    // impulse follows, at least zero, its tangents projected on its cone; a receding one takes zero impulses.
    const auto contact_coulomb_newton = [&](const float* u, const float* g, float target_vel_n, float mu,
                                            float* lambda) {
"""
    + _FRICTION_LAW_HEAD_CUDA
    + """        if (G_nn >= 1.0e-20f) {
            const float u_free_n = __fsub_rn(u_n, __fadd_rn(__fadd_rn(__fmul_rn(G_nn, old_lambda_n),
                __fmul_rn(G_nt1, old_lambda_t1)), __fmul_rn(G_nt2, old_lambda_t2)));
            const float u_free_t1 = __fsub_rn(u_t1, __fadd_rn(__fadd_rn(__fmul_rn(G_nt1, old_lambda_n),
                __fmul_rn(G_t1t1, old_lambda_t1)), __fmul_rn(G_t1t2, old_lambda_t2)));
            const float u_free_t2 = __fsub_rn(u_t2, __fadd_rn(__fadd_rn(__fmul_rn(G_nt2, old_lambda_n),
                __fmul_rn(G_t1t2, old_lambda_t1)), __fmul_rn(G_t2t2, old_lambda_t2)));
            if (u_free_n < target_vel_n) {
                const float bN = __fsub_rn(u_free_n, target_vel_n);
                const float bT0 = u_free_t1;
                const float bT1 = u_free_t2;
                const float WN = G_nn;
                const float wNT0 = G_nt1;
                const float wNT1 = G_nt2;
                const float AT00 = __fsub_rn(G_t1t1, __fdiv_rn(__fmul_rn(wNT0, wNT0), WN));
                const float AT01 = __fsub_rn(G_t1t2, __fdiv_rn(__fmul_rn(wNT0, wNT1), WN));
                const float AT10 = __fsub_rn(G_t1t2, __fdiv_rn(__fmul_rn(wNT1, wNT0), WN));
                const float AT11 = __fsub_rn(G_t2t2, __fdiv_rn(__fmul_rn(wNT1, wNT1), WN));
                const float cT0 = __fsub_rn(bT0, __fmul_rn(__fdiv_rn(bN, WN), wNT0));
                const float cT1 = __fsub_rn(bT1, __fmul_rn(__fdiv_rn(bN, WN), wNT1));
                const float inv_det0 = __fdiv_rn(1.0f, __fsub_rn(__fmul_rn(AT00, AT11), __fmul_rn(AT01, AT10)));
                const float s0_0 = __fmul_rn(__fsub_rn(__fmul_rn(AT11, cT0), __fmul_rn(AT01, cT1)), inv_det0);
                const float s0_1 = __fmul_rn(__fsub_rn(__fmul_rn(AT00, cT1), __fmul_rn(AT10, cT0)), inv_det0);
                const float norm_s0 = __fsqrt_rn(__fadd_rn(__fmul_rn(s0_0, s0_0), __fmul_rn(s0_1, s0_1)));
                const float phi0 = __fsub_rn(norm_s0, __fdiv_rn(__fmul_rn(mu,
                    __fsub_rn(__fadd_rn(__fmul_rn(wNT0, s0_0), __fmul_rn(wNT1, s0_1)), bN)), WN));
                float last_s0 = 0.0f, last_s1 = 0.0f;
                if (phi0 <= 0.0f) {
                    last_s0 = s0_0;
                    last_s1 = s0_1;
                } else {
                    float hi = 1.0f;
                    for (int _e = 0; _e < 30; _e++) {
                        const float a_hi = __fadd_rn(AT00, hi);
                        const float d_hi = __fadd_rn(AT11, hi);
                        const float idet_hi =
                            __fdiv_rn(1.0f, __fsub_rn(__fmul_rn(a_hi, d_hi), __fmul_rn(AT01, AT10)));
                        const float sh0 = __fmul_rn(__fsub_rn(__fmul_rn(d_hi, cT0), __fmul_rn(AT01, cT1)), idet_hi);
                        const float sh1 = __fmul_rn(__fsub_rn(__fmul_rn(a_hi, cT1), __fmul_rn(AT10, cT0)), idet_hi);
                        const float ns_hi = __fsqrt_rn(__fadd_rn(__fmul_rn(sh0, sh0), __fmul_rn(sh1, sh1)));
                        const float phi_hi = __fsub_rn(ns_hi, __fdiv_rn(__fmul_rn(mu,
                            __fsub_rn(__fadd_rn(__fmul_rn(wNT0, sh0), __fmul_rn(wNT1, sh1)), bN)), WN));
                        if (phi_hi < 0.0f) break;
                        hi = __fmul_rn(hi, 2.0f);
                    }
                    float lo = 0.0f;
                    float x = __fmul_rn(0.5f, __fadd_rn(lo, hi));
                    const float tol = __fadd_rn(1.0e-6f, __fmul_rn(1.0e-6f, phi0));
                    for (int _it = 0; _it < 50; _it++) {
                        const float ax = __fadd_rn(AT00, x);
                        const float dx = __fadd_rn(AT11, x);
                        const float idet = __fdiv_rn(1.0f, __fsub_rn(__fmul_rn(ax, dx), __fmul_rn(AT01, AT10)));
                        const float s0 = __fmul_rn(__fsub_rn(__fmul_rn(dx, cT0), __fmul_rn(AT01, cT1)), idet);
                        const float s1 = __fmul_rn(__fsub_rn(__fmul_rn(ax, cT1), __fmul_rn(AT10, cT0)), idet);
                        const float t0 = __fmul_rn(__fsub_rn(__fmul_rn(dx, s0), __fmul_rn(AT01, s1)), idet);
                        const float t1 = __fmul_rn(__fsub_rn(__fmul_rn(ax, s1), __fmul_rn(AT10, s0)), idet);
                        const float ns = __fsqrt_rn(__fadd_rn(__fmul_rn(s0, s0), __fmul_rn(s1, s1)));
                        const float fx = __fsub_rn(ns, __fdiv_rn(__fmul_rn(mu,
                            __fsub_rn(__fadd_rn(__fmul_rn(wNT0, s0), __fmul_rn(wNT1, s1)), bN)), WN));
                        const float dfx = __fadd_rn(__fdiv_rn(-__fadd_rn(__fmul_rn(s0, t0), __fmul_rn(s1, t1)), ns),
                            __fdiv_rn(__fmul_rn(mu, __fadd_rn(__fmul_rn(wNT0, t0), __fmul_rn(wNT1, t1))), WN));
                        last_s0 = s0;
                        last_s1 = s1;
                        if (fabsf(fx) < tol || fabsf(__fsub_rn(hi, lo)) < __fmul_rn(1.0e-6f, __fadd_rn(1.0f, hi)))
                            break;
                        if (fx > 0.0f) lo = x;
                        else hi = x;
                        float x_new = __fmul_rn(0.5f, __fadd_rn(lo, hi));
                        if (dfx != 0.0f) {
                            const float x_newton = __fsub_rn(x, __fdiv_rn(fx, dfx));
                            if (x_newton > lo && x_newton < hi) x_new = x_newton;
                        }
                        x = x_new;
                    }
                }
                const float rT0 = -last_s0;
                const float rT1 = -last_s1;
                // The finite search can end just below zero load; the exact root is non-negative.
                new_lambda_n = fmaxf(
                    __fdiv_rn(-__fadd_rn(__fadd_rn(__fmul_rn(wNT0, rT0), __fmul_rn(wNT1, rT1)), bN), WN), 0.0f);
                new_lambda_t1 = rT0;
                new_lambda_t2 = rT1;
                // The root search stops within a tolerance and leaves a residual tangent at mu = 0.
                const float flimit = fmaxf(__fmul_rn(mu, new_lambda_n), 0.0f);
                const float tmag = __fsqrt_rn(
                    __fadd_rn(__fmul_rn(new_lambda_t1, new_lambda_t1), __fmul_rn(new_lambda_t2, new_lambda_t2)));
                if (tmag > flimit) {
                    const float sc = __fdiv_rn(flimit, tmag);
                    new_lambda_t1 = __fmul_rn(new_lambda_t1, sc);
                    new_lambda_t2 = __fmul_rn(new_lambda_t2, sc);
                }
            } else {
                new_lambda_n = 0.0f;
                new_lambda_t1 = 0.0f;
                new_lambda_t2 = 0.0f;
            }
        }
"""
    + _FRICTION_LAW_TAIL_CUDA
)
# The head of a row visit under a friction mode, after its bases: the update its contact takes, which
# prepare_scene_friction_modes writes on the contact's first tangent row, and Steel's modes' diagonal guard of a pair:
# Steel's modes leave the friction rows out of its diagonal guard, so a pair without a diagonal is skipped here.
_FRICTION_MODE_HEAD_CUDA = """\
            // Friction modes: the contact's update (row_friction_mode); a tangent pair without a diagonal is skipped,
            // as Steel's modes drop the pairs' diagonal guard.
            const int contact_mode = friction ? row_friction_mode.data[row] : FIXED_NORMAL;
            if (friction && denominator <= 0.0f) continue;
"""
_FRICTION_MODE_DECLS_CUDA = """\
            // Friction modes: the normal impulse of a contact's combined update, and its change.
            float new_normal = 0.0f;
            float normal_change = 0.0f;
"""
# The combined update of a flagged contact (contact_mode DENSE or FREE_BODY), the first branch, before the tangent
# pair's: the contact's three rows together, Steel's dense text for an articulated contact, its free-body text (with
# de Saxce's bias under "bisection_desaxce") otherwise.
_FRICTION_MODE_UPDATE_CUDA = """\
            if (friction && contact_mode != FIXED_NORMAL) {
                // Steel's friction mode: the contact's normal and tangent impulses together, at its first tangent row.
                float u[3], g[6];
                contact_block(parent, contact_mode == FREE_BODY, u, g);
                const float mu = rows_mu.data[row];
                float target = -rows_rhs.data[parent];
$BIAS                const float old_normal = impulses.data[parent];
                const float old_sibling = impulses.data[sibling];
                float lambda[3] = {old_normal, old_impulse, old_sibling};
                $LAW(u, g, target, mu, lambda);
                new_normal = lambda[0];
                normal_change = new_normal - old_normal;
                new_impulse = lambda[1];
                new_sibling = lambda[2];
                sibling_change = new_sibling - old_sibling;
            } else
"""
# de Saxce's bias mu |c_T| on a free-body contact's normal target (``_MF_DESAXCE_BIAS_SOURCE``, which Steel splices
# between its velocities and its coupling; Steel's dense text splices nothing).
_FRICTION_MODE_DESAXCE_CUDA = """\
                if (contact_mode == FREE_BODY) {
                    const float c_T_mag = __fsqrt_rn(__fadd_rn(__fmul_rn(u[1], u[1]), __fmul_rn(u[2], u[2])));
                    target = __fadd_rn(target, __fmul_rn(mu, c_T_mag));
                }
"""


def _friction_mode_funcs(features: SceneFeatures) -> str:
    """The device functions of ``features.friction_mode`` that precede the row loops: the contact's inputs and the
    mode's law (``contact_bisection`` under ``"bisection"`` and ``"bisection_desaxce"``, ``contact_coulomb_newton``
    under ``"coulomb_newton"``); nothing with the mode off. The rounding intrinsics they name have their host lambdas
    at the snippet's head (:data:`_HOST_ROUNDING_SHIM_CPP`)."""
    if features.friction_mode == "fixed_normal":
        return ""
    law = (
        _FRICTION_MODE_COULOMB_NEWTON_CUDA
        if features.friction_mode == "coulomb_newton"
        else _FRICTION_MODE_BISECTION_CUDA
    )
    return _FRICTION_MODE_BLOCK_CUDA + law


_FRICTION_MODE_RESPOND_CUDA = """\
                        if (normal_change != 0.0f) {
                            // Steel's friction mode: the normal's change first, then the pair's.
$ENTRY                            const float normal_response = $K < prefix
                                ? rows_free_response.data[parent * free_width + $K]
                                : rows_factor.data[parent * SUPPORT + $K];
                            value = __fadd_rn(value, __fmul_rn(normal_response, normal_change));
                        }
"""


def _friction_mode_tail(features: SceneFeatures, part: str, *, group: bool = False) -> str:
    """A row loop's tail lines of ``features.friction_mode`` (nothing with the mode off), by ``part``: ``"write"`` the
    combined update's normal impulse (lane 0's, in the lane-group loop ``group``), ``"changed"`` its change in the
    test of a changed row, and ``"respond"`` its velocity change through the normal row's response, before the pair's
    (Steel's order: normal, second tangent, first tangent)."""
    if features.friction_mode == "fixed_normal":
        return ""
    if part == "write":
        indent = " " * (16 if group else 12)
        return f"{indent}if (contact_mode != FIXED_NORMAL) impulses.data[parent] = new_normal;\n"
    if part == "changed":
        return " || normal_change != 0.0f"
    entry = "                            const int entry = lane + q * GROUP;\n" if group else ""
    return _FRICTION_MODE_RESPOND_CUDA.replace("$ENTRY", entry).replace("$K", "entry" if group else "k")


def _friction_mode_update(features: SceneFeatures) -> str:
    """The combined update's branch of ``features.friction_mode`` (``_FRICTION_MODE_UPDATE_CUDA``), with its law and
    de Saxce's bias under ``"bisection_desaxce"``; nothing with the mode off."""
    if features.friction_mode == "fixed_normal":
        return ""
    law = "contact_coulomb_newton" if features.friction_mode == "coulomb_newton" else "contact_bisection"
    bias = _FRICTION_MODE_DESAXCE_CUDA if features.friction_mode == "bisection_desaxce" else ""
    return _FRICTION_MODE_UPDATE_CUDA.replace("$BIAS", bias).replace("$LAW", law)


def _impulse_update(features: SceneFeatures) -> str:
    """The impulse update of a row visit, with ``features``' lines, joined from its named parts.

    Contact regularization weighs the normal and limit rows' update; the fused clamp and drive rows replace it
    (the clamp in a clamp pass, the force-drive update otherwise), in Steel's branch order before the clamps;
    late rows give a velocity-limit row its stateless update, and the stateless rows their declarations and their
    own velocity change; contact torsion clamps a spin row's update to its bound; joint friction clamps a
    joint-friction row's impulse to its box. Every feature off joins
    the declarations, the pair branch, the rigid update, the unilateral clamp and the impulse's change.
    Contact compliance gives the declarations a compliant row's right-hand side, ``gamma * lambda`` added: the
    residual of the normal update ``delta lambda = -(w + gamma lambda) / (diag + gamma)``, whose ``diag + gamma`` is
    the stored diagonal (``scene_compliance.prepare_scene_compliance`` folds gamma in once), so the division is the
    default's. A friction mode declares the combined update's normal impulse
    and puts its branch first, before the tangent pair's (:func:`_friction_mode_update`). Friction patches walk the
    tangent pair's region for its normal load (``_PATCH_LOAD_CUDA``).
    """
    rhs = _COMPLIANT_RHS_CUDA if features.compliance else _IMPULSE_RHS_CUDA
    update = _REGULARIZED_UPDATE_CUDA if features.regularize else _RIGID_UPDATE_CUDA
    fused = _FUSED_CLAMP_UPDATE_CUDA if features.fused_clamp else ""
    drive = _DRIVE_UPDATE_CUDA if features.drives else ""
    late = _VELOCITY_LIMIT_UPDATE_CUDA if features.late_rows else ""
    torsion = _TORSION_UPDATE_CUDA if features.torsion else ""
    clamps = _BOX_CLAMP_CUDA if features.joint_friction else _UNILATERAL_CLAMP_CUDA
    stateless = _STATELESS_DECLS_CUDA if features.late_rows else ""
    change = _STATELESS_CHANGE_CUDA if features.late_rows else _IMPULSE_CHANGE_CUDA
    modes = _FRICTION_MODE_DECLS_CUDA if features.friction_mode != "fixed_normal" else ""
    pair = _PAIR_BRANCH_CUDA.replace(_NORMAL_LOAD_CUDA, _PATCH_LOAD_CUDA) if features.patches else _PAIR_BRANCH_CUDA
    return (
        rhs
        + _IMPULSE_DECLS_CUDA
        + stateless
        + modes
        + _friction_mode_update(features)
        + pair
        + update
        + fused
        + drive
        + late
        + torsion
        + clamps
        + change
    )


# The host shim, the head of every sweep snippet: on the CPU it declares the CUDA vector
# types and float helpers the thread loop names. Warp's host runtime declares ``sqrtf`` and ``fabsf`` (crt.h) but
# neither ``fmaxf`` nor ``fminf``, and no vector type. A snippet is pasted inside a function body, where no function
# can be defined, so the shim declares local structs and ``static constexpr`` lambdas, which the snippet's lambdas
# name without a capture. ``fmaxf`` and ``fminf`` return the other operand when one is NaN, as C's do (glibc's
# expressions). On CUDA it expands to nothing.
_HOST_SHIM_CPP = """#if !defined(__CUDA_ARCH__)
    struct int2 { int x, y; };
    struct float2 { float x, y; };
    static constexpr auto make_int2 = [](int x, int y) { return int2{x, y}; };
    static constexpr auto make_float2 = [](float x, float y) { return float2{x, y}; };
    static constexpr auto fmaxf = [](float a, float b) { return (a >= b || b != b) ? a : b; };
    static constexpr auto fminf = [](float a, float b) { return (a <= b || b != b) ? a : b; };
#endif
"""

# The row types, the friction pair and the names both row loops share; needs ``SUPPORT`` and the arrays.
_ROWS_PRELUDE_CUDA = (
    f"""
    constexpr int CONTACT = {int(PGS_CONSTRAINT_TYPE_CONTACT)};
    constexpr int FRICTION = {int(PGS_CONSTRAINT_TYPE_FRICTION)};
    constexpr int LIMIT = {int(PGS_CONSTRAINT_TYPE_JOINT_LIMIT)};
"""
    + FRICTION_PAIR_CUDA
    + """
    // Entries of a row loaded before the wide ones (supports are a valid prefix padded with -1).
    constexpr int NARROW = SUPPORT < 16 ? SUPPORT : 16;
    const int free_width = rows_free_response.shape[2];

"""
)
# The head of a row visit, shared by the thread and the lane-group loops (Steel's sparse sweep's): it skips a
# tangent pair's second row and a normal or limit row without a diagonal, and names the row's bases.
_ROW_TYPE_CUDA = """
            const int row_type = rows_type.data[row];
            const bool friction = row_type == FRICTION;
"""
_ROW_BASES_CUDA = """            const int parent = friction ? rows_parent.data[row] : -1;
            // The tangent pair is solved at its first row; the second is its sibling.
            if (friction && row != parent + 1) continue;
            const int sibling = friction ? parent + 2 : -1;
            const float denominator = rows_diag.data[row];
            // Normal and limit rows without a diagonal are skipped; a tangent pair never is.
            if (!friction && denominator <= 0.0f) continue;
            const int base = row * SUPPORT;
            const int sibling_base = sibling * SUPPORT;
            // The leading entries in physical free-body coordinates respond through the row's own
            // free-body response; every other entry through its factor value.
            const int prefix = free_response ? rows_free_prefix.data[row] : 0;
            const int free_base = row * free_width;
            const int sibling_free_base = sibling * free_width;
"""
# Friction position iterations (``SceneFeatures.friction_position_iterations``): a sweep before
# ``friction_start`` skips both rows of every tangent pair (Steel's gate on its ``_friction_start_iteration``),
# so their impulses stay at the zero each step starts from. The sweep sets ``friction_skipped``.
_FRICTION_GATE_CUDA = """            if (friction && friction_skipped) continue;
"""


def _row_header(features: SceneFeatures, *, group: bool = False) -> str:
    """The head of a row visit; friction position iterations add the tangent pairs' gate after the row type, the
    fused clamp its clamp pass's gate, which visits the drive rows alone, and contact torsion C1's head of a spin row
    (``torsion_budget``, or ``torsion_budget_group`` in the lane-group loop ``group``); a friction mode reads the
    contact's update and skips a tangent pair without a diagonal after the bases."""
    gate = _FRICTION_GATE_CUDA if features.friction_position_iterations else ""
    clamp_gate = _CLAMP_PASS_GATE_CUDA if features.fused_clamp else ""
    budget = "torsion_budget_group(row, lane, mask, changed)" if group else "torsion_budget(row, changed)"
    torsion = _TORSION_HEAD_CUDA.replace("{budget}", budget) if features.torsion else ""
    modes = _FRICTION_MODE_HEAD_CUDA if features.friction_mode != "fixed_normal" else ""
    return _ROW_TYPE_CUDA + gate + clamp_gate + torsion + _ROW_BASES_CUDA + modes


def _thread_loop(features: SceneFeatures) -> str:
    """Steel's row loop for one thread, ``sweep_rows(begin, end)``, with ``features``' row header and update.

    It follows the prelude.
    """
    return (
        """
    // Steel's sparse row loop (sparse_pgs.py) for one thread over rows [begin, end) of the
    // scene's W = 1 arrays: delta and the impulses live in global memory, and a row's right-hand
    // side is its bias plus its incident velocity, formed at load as Steel's kernel does. Returns 1
    // when any impulse changed.
    const auto sweep_rows = [&](int begin, int end) -> int {
        int changed = 0;
        for (int row = begin; row < end; ++row) {"""
        + _row_header(features)
        + """
            int index[SUPPORT];
            if (SUPPORT % 2 == 0) {
                // Even rows are 8-byte aligned: read the indices in pairs, the wide ones only when
                // the narrow ones are all valid.
                const int2* pairs = reinterpret_cast<const int2*>(rows_dof.data + base);
                #pragma unroll
                for (int pair = 0; pair < NARROW / 2; ++pair) {
                    const int2 two = pairs[pair];
                    index[2 * pair] = two.x;
                    index[2 * pair + 1] = two.y;
                }
                const bool wide = SUPPORT > NARROW && index[NARROW - 1] >= 0;
                #pragma unroll
                for (int pair = NARROW / 2; pair < SUPPORT / 2; ++pair) {
                    const int2 two = wide ? pairs[pair] : make_int2(-1, -1);
                    index[2 * pair] = two.x;
                    index[2 * pair + 1] = two.y;
                }
            } else {
                #pragma unroll
                for (int k = 0; k < NARROW; ++k) index[k] = rows_dof.data[base + k];
                const bool wide = SUPPORT > NARROW && index[NARROW - 1] >= 0;
                #pragma unroll
                for (int k = NARROW; k < SUPPORT; ++k) index[k] = wide ? rows_dof.data[base + k] : -1;
            }

"""
        # The row's dot products. _scene_row_dot (prepare_scene_velocity_rows) mirrors ``dot`` statement by statement,
        # Steel's tree and the serial chain; test_scene_row_dot_mirrors_the_thread_loop pins the two bit for bit.
        + """            float dot = 0.0f;
            float sibling_dot = 0.0f;
            float cross = 0.0f;
            if (steel_sums) {
                // Steel's lane l holds entry l, zero past the support; its first shuffle level adds
                // lanes l and l + 16, then 8, 4, 2 and 1, every add plain.
                float dots[16];
                float sibling_dots[16];
                float crosses[16];
                #pragma unroll
                for (int lane = 0; lane < 16; ++lane) {
                    float low = 0.0f, sibling_low = 0.0f, cross_low = 0.0f;
                    float high = 0.0f, sibling_high = 0.0f, cross_high = 0.0f;
                    if (lane < SUPPORT && index[lane] >= 0) {
                        const int k = lane;
                        const float value = rows_factor.data[base + k];
                        const float sibling_value = friction ? rows_factor.data[sibling_base + k] : 0.0f;
                        const float sibling_response = friction
                            ? (k < prefix ? rows_free_response.data[sibling_free_base + k] : sibling_value) : 0.0f;
                        const float velocity = delta.data[index[k]];
                        low += value * velocity;
                        sibling_low += sibling_value * velocity;
                        cross_low += value * sibling_response;
                    }
                    if (lane + 16 < SUPPORT && index[lane + 16] >= 0) {
                        const int k = lane + 16;
                        const float value = rows_factor.data[base + k];
                        const float sibling_value = friction ? rows_factor.data[sibling_base + k] : 0.0f;
                        const float sibling_response = friction
                            ? (k < prefix ? rows_free_response.data[sibling_free_base + k] : sibling_value) : 0.0f;
                        const float velocity = delta.data[index[k]];
                        high += value * velocity;
                        sibling_high += sibling_value * velocity;
                        cross_high += value * sibling_response;
                    }
                    dots[lane] = low + high;
                    sibling_dots[lane] = sibling_low + sibling_high;
                    crosses[lane] = cross_low + cross_high;
                }
                // Levels 8, 4, 2, 1, each with a constant trip count so the arrays stay in registers.
                #define SCENE_TREE_LEVEL(SHIFT) \\
                    _Pragma("unroll") \\
                    for (int lane = 0; lane < SHIFT; ++lane) { \\
                        dots[lane] += dots[lane + SHIFT]; \\
                        sibling_dots[lane] += sibling_dots[lane + SHIFT]; \\
                        crosses[lane] += crosses[lane + SHIFT]; \\
                    }
                SCENE_TREE_LEVEL(8)
                SCENE_TREE_LEVEL(4)
                SCENE_TREE_LEVEL(2)
                SCENE_TREE_LEVEL(1)
                #undef SCENE_TREE_LEVEL
                dot = dots[0];
                sibling_dot = sibling_dots[0];
                cross = crosses[0];
            } else {
                // One multiply-add chain in entry order.
                #pragma unroll
                for (int k = 0; k < SUPPORT; ++k) {
                    if (index[k] >= 0) {
                        const float value = rows_factor.data[base + k];
                        const float velocity = delta.data[index[k]];
                        dot += value * velocity;
                        if (friction) {
                            const float sibling_value = rows_factor.data[sibling_base + k];
                            const float sibling_response =
                                k < prefix ? rows_free_response.data[sibling_free_base + k] : sibling_value;
                            sibling_dot += sibling_value * velocity;
                            cross += value * sibling_response;
                        }
                    }
                }
            }
"""
        + _impulse_update(features)
        + """
            if (friction) impulses.data[sibling] = new_sibling;
            impulses.data[row] = new_impulse;
"""
        + _friction_mode_tail(features, "write")
        + "            if (change != 0.0f || sibling_change != 0.0f"
        + _friction_mode_tail(features, "changed")
        + """) {
                changed = 1;
                // A support's coordinates are distinct: read every entry before writing any.
                float current[SUPPORT];
                #pragma unroll
                for (int k = 0; k < SUPPORT; ++k)
                    if (index[k] >= 0) current[k] = delta.data[index[k]];
                #pragma unroll
                for (int k = 0; k < SUPPORT; ++k) {
                    if (index[k] >= 0) {
                        // Steel's paired-tangent response order: the sibling's change, then the row's.
                        float value = current[k];
"""
        + _friction_mode_tail(features, "respond")
        + """                        if (sibling_change != 0.0f)
                            value += (k < prefix ? rows_free_response.data[sibling_free_base + k]
                                                 : rows_factor.data[sibling_base + k]) * sibling_change;
                        if (change != 0.0f)
                            value += (k < prefix ? rows_free_response.data[free_base + k]
                                                 : rows_factor.data[base + k]) * change;
                        delta.data[index[k]] = value;
                    }
                }
            }
        }
        return changed;
    };
"""
    )


# Steel's row loop for one thread, ``sweep_rows(begin, end)``; it follows the prelude.
_THREAD_LOOP_CUDA = _thread_loop(SceneFeatures())
SCENE_ROWS_CUDA = _ROWS_PRELUDE_CUDA + _THREAD_LOOP_CUDA
"""Steel's sparse row loop for one thread, ``sweep_rows(begin, end)``, which returns ``changed``.

It needs, in scope: ``constexpr int SUPPORT`` (entries per row), ``constexpr bool steel_sums`` and
``constexpr bool free_response``; the W = 1 row arrays ``rows_dof``, ``rows_factor``,
``rows_free_response``, ``rows_free_prefix``, ``rows_rhs`` (the bias), ``rows_incident``,
``rows_diag``, ``rows_type``, ``rows_parent`` and ``rows_mu``; ``delta``, ``impulses`` and
``float omega``. The sweep kernels' snippets declare them.

The arithmetic is Steel's expression for expression (``sparse_pgs._get_pgs_solve_sparse_kernel``): the tangent pair
is solved at its first row with ``FRICTION_PAIR_CUDA``, a normal or limit row with ``diag <= 0`` is
skipped, the impulse update divides by the diagonal, and the delta update adds the sibling's term
before the row's own. With ``steel_sums`` the dot products follow Steel's 32-lane shuffle tree, so
the sweep is bitwise Steel's on the same rows; otherwise they are one chain in entry order. The
lane-group loop of the side and unit kernels shares the prelude and the head of a row visit.
"""


# The same loop for a group of GROUP lanes of one warp, lane g holding entries g + q·GROUP:
# ``sweep_rows_group(begin, end, lane, mask)``, whose return value every lane of the group shares.
# With Steel's sums the 8 lanes reproduce Steel's 32-lane tree: Steel's levels 16 and 8 pair entries
# a lane holds itself (q with q + 2, then 0 with 1), levels 4, 2 and 1 are shuffles in the group.
def _group_loop(features: SceneFeatures) -> str:
    """The lane-group row loop, ``sweep_rows_group(begin, end, lane, mask)``, with ``features``' head and update."""
    return (
        """
    const auto sweep_rows_group = [&](int begin, int end, int lane, unsigned mask) -> int {
        int changed = 0;
        for (int row = begin; row < end; ++row) {"""
        + _row_header(features, group=True)
        + """
            int indices[PER_LANE];
            float responses[PER_LANE];
            float sibling_responses[PER_LANE];
            float dots[PER_LANE];
            float sibling_dots[PER_LANE];
            float crosses[PER_LANE];
            float dot = 0.0f;
            float sibling_dot = 0.0f;
            float cross = 0.0f;
            // Entries 16 and up are read only when entry 15 is valid (a valid prefix).
            #pragma unroll
            for (int q = 0; q < PER_LANE; ++q) {
                const int entry = lane + q * GROUP;
                indices[q] = q * GROUP < 16 && entry < SUPPORT ? rows_dof.data[base + entry] : -1;
            }
            const bool wide = SUPPORT > 16 && __shfl_sync(mask, indices[15 / GROUP], 15 % GROUP, GROUP) >= 0;
            #pragma unroll
            for (int q = 0; q < PER_LANE; ++q) {
                const int entry = lane + q * GROUP;
                if (q * GROUP >= 16) indices[q] = wide && entry < SUPPORT ? rows_dof.data[base + entry] : -1;
            }
            #pragma unroll
            for (int q = 0; q < PER_LANE; ++q) {
                const int entry = lane + q * GROUP;
                const int index = indices[q];
                const float value = index >= 0 ? rows_factor.data[base + entry] : 0.0f;
                const float sibling_value = friction && index >= 0 ? rows_factor.data[sibling_base + entry] : 0.0f;
                float response = value;
                float sibling_response = sibling_value;
                if (index >= 0 && entry < prefix) {
                    response = rows_free_response.data[free_base + entry];
                    if (friction) sibling_response = rows_free_response.data[sibling_free_base + entry];
                }
                responses[q] = response;
                sibling_responses[q] = sibling_response;
                if (steel_sums) {
                    float own = 0.0f, sibling_own = 0.0f, cross_own = 0.0f;
                    if (index >= 0) {
                        const float velocity = delta.data[index];
                        own += value * velocity;
                        sibling_own += sibling_value * velocity;
                        cross_own += value * sibling_response;
                    }
                    dots[q] = own;
                    sibling_dots[q] = sibling_own;
                    crosses[q] = cross_own;
                } else if (index >= 0) {
                    const float velocity = delta.data[index];
                    dot += value * velocity;
                    sibling_dot += sibling_value * velocity;
                    cross += value * sibling_response;
                }
            }
            if (steel_sums) {
                #pragma unroll
                for (int half = PER_LANE / 2; half > 0; half >>= 1) {
                    #pragma unroll
                    for (int q = 0; q < half; ++q) {
                        dots[q] += dots[q + half];
                        sibling_dots[q] += sibling_dots[q + half];
                        crosses[q] += crosses[q + half];
                    }
                }
                dot = dots[0];
                sibling_dot = sibling_dots[0];
                cross = crosses[0];
            }
            for (int shift = GROUP / 2; shift > 0; shift >>= 1) {
                dot += __shfl_down_sync(mask, dot, shift, GROUP);
                sibling_dot += __shfl_down_sync(mask, sibling_dot, shift, GROUP);
                cross += __shfl_down_sync(mask, cross, shift, GROUP);
            }
            dot = __shfl_sync(mask, dot, 0, GROUP);
            sibling_dot = __shfl_sync(mask, sibling_dot, 0, GROUP);
            cross = __shfl_sync(mask, cross, 0, GROUP);"""
        + _impulse_update(features)
        + """
            // Every lane has read the old impulses before lane 0 overwrites them.
            __syncwarp(mask);
            if (lane == 0) {
                if (friction) impulses.data[sibling] = new_sibling;
                impulses.data[row] = new_impulse;
"""
        + _friction_mode_tail(features, "write", group=True)
        + """            }
            if (change != 0.0f || sibling_change != 0.0f"""
        + _friction_mode_tail(features, "changed")
        + """) {
                changed = 1;
                float current[PER_LANE];
                #pragma unroll
                for (int q = 0; q < PER_LANE; ++q)
                    if (indices[q] >= 0) current[q] = delta.data[indices[q]];
                #pragma unroll
                for (int q = 0; q < PER_LANE; ++q) {
                    if (indices[q] >= 0) {
                        // Steel's paired-tangent response order: the sibling's change, then the row's.
                        float value = current[q];
"""
        + _friction_mode_tail(features, "respond", group=True)
        + """                        if (sibling_change != 0.0f) value += sibling_responses[q] * sibling_change;
                        if (change != 0.0f) value += responses[q] * change;
                        delta.data[indices[q]] = value;
                    }
                }
            }
            // The group's delta writes are visible before the next row reads them.
            __syncwarp(mask);
        }
        return changed;
    };
"""
    )


def _torsion_budget(*, group: bool) -> str:
    """C1's head of a spin row under contact torsion: ``torsion_budget(spin, changed)`` (with ``group``,
    ``torsion_budget_group(spin, lane, mask, changed)``), which returns the spin row's bound.

    Steel's ``contact_torsion.torque_sweep_source`` statement for statement, with Steel's names (the guard test
    compares the two, :func:`_mirrored_lines`): the members' pooled normal load ``sum max(lambda_n, 0)`` and the
    anchors' sliding ``sum sqrt(t1^2 + t2^2)``, the budget ``mu * load`` with ``mu = max(row_mu, 0)`` the group's
    coefficient, the anchors' tangent pairs rescaled by ``budget / sliding`` when the sliding passes it, each change
    applied through the pair's rows' responses ``Y1 * delta1 + Y2 * delta2`` (their free-body responses over their
    free prefix, their factor values elsewhere) and the sliding summed again, and the bound ``radius * max(budget -
    sliding, 0)``, 0 for a retired spin row. The members are the spin row's ``torsion_member_rows`` range, each a
    normal row shifted left by one with bit 0 set when the row is followed by its own tangent pair (an anchor), where
    Steel walks its rows for them; they belong to other units. Every product, sum and quotient is rounded on its own
    (``__fmul_rn`` and the rest, no contraction), so each inlined copy rounds alike. A lane group's lanes all form the
    sums; lane 0 writes a rescaled pair after every lane has read it, and each lane applies the change to its entries.
    """
    if group:
        name, params = "torsion_budget_group", "int spin, int lane, unsigned mask"
        first, step, sync = "lane", "GROUP", "                __syncwarp(mask);\n"
        write = """                    if (lane == 0) {
                        impulses.data[n + 1] = t1;
                        impulses.data[n + 2] = t2;
                    }
"""
    else:
        name, params = "torsion_budget", "int spin"
        first, step, sync = "0", "1", ""
        write = """                    impulses.data[n + 1] = t1;
                    impulses.data[n + 2] = t2;
"""
    return (
        f"""
    // C1's head of spin row `spin` (Steel's torque_sweep_source): the budget, the rescale, the bound.
    const auto {name} = [&]({params}, int& changed) -> float {{
        const float radius = torsion_radius.data[spin];
        // The spin row carries its group's undivided coefficient.
        const float mu = fmaxf(rows_mu.data[spin], 0.0f);
        const int spin_group = torsion_group.data[spin];
        const int first = torsion_member_range.data[2 * spin];
        const int last = torsion_member_range.data[2 * spin + 1];
        float normal_load = 0.0f;
        float sliding_used = 0.0f;
        for (int k = first; k < last; ++k) {{
            const int member = torsion_member_rows.data[k];
            const int n = member >> 1;
            normal_load = __fadd_rn(normal_load, fmaxf(impulses.data[n], 0.0f));
            // Only a normal followed by its own tangent pair (an anchor) consumes sliding budget.
            if (member & 1) {{
                const float a = impulses.data[n + 1], b = impulses.data[n + 2];
                sliding_used = __fadd_rn(sliding_used, __fsqrt_rn(__fadd_rn(__fmul_rn(a, a), __fmul_rn(b, b))));
            }}
        }}
        const float normal_budget = __fmul_rn(mu, normal_load);
        if (sliding_used > normal_budget) {{
            const float scale = __fdiv_rn(normal_budget, sliding_used);
            sliding_used = 0.0f;
            for (int k = first; k < last; ++k) {{
                const int member = torsion_member_rows.data[k];
                if (!(member & 1)) continue;
                const int n = member >> 1;
                const float t1 = __fmul_rn(impulses.data[n + 1], scale);
                const float t2 = __fmul_rn(impulses.data[n + 2], scale);
                const float delta1 = __fsub_rn(t1, impulses.data[n + 1]);
                const float delta2 = __fsub_rn(t2, impulses.data[n + 2]);
"""
        + sync
        + """                if (delta1 != 0.0f || delta2 != 0.0f) {
                    changed = 1;
"""
        + write
        + f"""                    // The pair's velocity change through each row's response (its
                    // free-body response over its free prefix, its factor values elsewhere; the pair shares its
                    // contact's support), in Steel's order Y1 * delta1 + Y2 * delta2.
                    const int prefix = free_response ? rows_free_prefix.data[n + 1] : 0;
                    for (int entry = {first}; entry < SUPPORT; entry += {step}) {{
                        const int index = rows_dof.data[(n + 1) * SUPPORT + entry];
                        if (index < 0) continue;
                        const float y1 = entry < prefix ? rows_free_response.data[(n + 1) * free_width + entry]
                                                        : rows_factor.data[(n + 1) * SUPPORT + entry];
                        const float y2 = entry < prefix ? rows_free_response.data[(n + 2) * free_width + entry]
                                                        : rows_factor.data[(n + 2) * SUPPORT + entry];
                        delta.data[index] =
                            __fadd_rn(delta.data[index], __fadd_rn(__fmul_rn(y1, delta1), __fmul_rn(y2, delta2)));
                    }}
                }}
                sliding_used = __fadd_rn(sliding_used, __fsqrt_rn(__fadd_rn(__fmul_rn(t1, t1), __fmul_rn(t2, t2))));
"""
        + sync
        + """            }
        }
        float bound = __fmul_rn(radius, fmaxf(__fsub_rn(normal_budget, sliding_used), 0.0f));
        if (spin_group == TORSION_RETIRED) bound = 0.0f;
        return bound;
    };
"""
    )


def _check_sweep(granularity: str, sums: str, support: int, *, key: str = "scene_side_sweep") -> int:
    """Validate a sweep kernel's build parameters and return its lanes per work item.

    ``key`` names the granularity's override in the error (``scene_side_sweep`` or ``scene_unit_sweep``).
    """
    if granularity not in SCENE_SIDE_LANES:
        raise ValueError(f"{key} must be one of {sorted(SCENE_SIDE_LANES)}, got {granularity!r}")
    if sums not in SCENE_SUM_ORDERS:
        raise ValueError(f"scene_sum_order must be one of {list(SCENE_SUM_ORDERS)}, got {sums!r}")
    if support <= 0:
        raise ValueError(f"the row support must be positive, got {support}")
    if sums == "steel" and support > SUM_TREE_SUPPORT_MAX:
        raise ValueError(
            f"scene_sum_order='steel' reproduces Steel's {SUM_TREE_SUPPORT_MAX}-lane sums, but the rows hold "
            f"{support} entries; use scene_sum_order='serial'"
        )
    return SCENE_SIDE_LANES[granularity]


def _sweep_snippet(
    granularity: str,
    sums: str,
    support: int,
    free_response: bool,
    body: str,
    *,
    thread_loop: bool = False,
    block: bool = False,
    phase_c: bool = False,
    features: SceneFeatures,
) -> str:
    """A sweep kernel's native snippet: the host shim, the build constants, the row loop for the granularity and
    ``body``.

    The snippet starts with :data:`_HOST_SHIM_CPP` (and, with a friction mode or contact torsion, the rounding
    intrinsics' :data:`_HOST_ROUNDING_SHIM_CPP`), so the one-thread loop compiles for the CPU as well. A lane-group
    granularity's shuffles and a block kernel's barriers (``block``, whose ``body`` syncs the block between colors)
    exist on CUDA alone: such a snippet keeps the rest under ``#if defined(__CUDA_ARCH__)`` and is empty on the CPU,
    where ``scene_solve.SceneSolve`` sweeps serially. On CUDA the shim and the guards preprocess to the snippet from
    before the shim, character for character.

    A lane-group granularity declares ``sweep_rows_group``; ``thread_loop`` adds the one-thread
    ``sweep_rows`` beside it (``"thread"`` has only that one). ``phase_c`` declares phase C of one side
    beside the row loop under late rows (``sweep_late``, ``sweep_late_group`` for a lane group). With
    friction position iterations (``features``) the row loops skip the tangent pairs while
    ``friction_skipped``, which the head declares and ``body`` sets before each sweep; with
    regularization they weigh the normal and limit
    rows' update by ``row_w``; with joint friction the head names the row type ``JOINT_FRICTION`` and
    the row loops clamp those rows' impulses to ``row_bound``; with drives the head names the row type
    ``JOINT_TARGET`` and the row loops give those rows the force-drive update of their descriptors; with
    late rows the head names the row type ``VELOCITY_LIMIT`` and the row loops give those rows Steel's
    stateless update; with the fused clamp the head declares ``clamp_pass``, which phase C sets while it
    walks a side's drive rows with the clamp; with contact torsion the head names ``TORSION`` and
    ``TORSION_RETIRED``, C1's heads of a spin row precede the row loops, and ``phase_c`` declares phase C without late
    rows too; with a friction mode the head names the contacts' updates ``FIXED_NORMAL`` and ``FREE_BODY`` and the
    mode's device functions precede the row loops (:func:`_friction_mode_funcs`).
    """
    group = SCENE_SIDE_LANES[granularity]
    head = f"""
    constexpr int SUPPORT = {int(support)};
    constexpr bool steel_sums = {"true" if sums == "steel" else "false"};
    constexpr bool free_response = {"true" if free_response else "false"};
"""
    # The device functions the row loops call: contact torsion's C1 heads of a spin row, for the row loops the snippet
    # declares, then a friction mode's.
    funcs = ""
    if features.torsion:
        funcs = _torsion_budget(group=False) if group == 1 or thread_loop else ""
        funcs = (_torsion_budget(group=True) if group > 1 else "") + funcs
    funcs += _friction_mode_funcs(features)
    if group == 1:
        loop = _ROWS_PRELUDE_CUDA + funcs + _thread_loop(features)
    else:
        # Steel's sums need one entry per Steel lane, so 4 per lane of 8; serial sums cover any row.
        per_lane = max(SUM_TREE_SUPPORT_MAX, int(support) + group - 1) // group
        head += f"    constexpr int GROUP = {group};\n    constexpr int PER_LANE = {per_lane};\n"
        loop = _ROWS_PRELUDE_CUDA + funcs + _group_loop(features) + (_thread_loop(features) if thread_loop else "")
    if features.friction_position_iterations:
        head += "    bool friction_skipped = false;\n"
    if features.joint_friction:
        head += f"    constexpr int JOINT_FRICTION = {int(PGS_CONSTRAINT_TYPE_JOINT_FRICTION)};\n"
    if features.drives:
        head += f"    constexpr int JOINT_TARGET = {int(PGS_CONSTRAINT_TYPE_JOINT_TARGET)};\n"
    if features.late_rows:
        head += f"    constexpr int VELOCITY_LIMIT = {int(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)};\n"
    # Phase C runs with late rows and with contact torsion, whose spin rows are late rows too.
    if phase_c and _has_phase_c(features):
        loop += _late_sweep(features, group=group > 1)
    if features.fused_clamp:
        head += "    bool clamp_pass = false;\n"
    if features.torsion:
        head += f"    constexpr int TORSION = {int(PGS_CONSTRAINT_TYPE_TORSION)};\n"
        head += f"    constexpr int TORSION_RETIRED = {int(_TORSION_SPIN_RETIRED)};\n"
    if features.friction_mode != "fixed_normal":
        head += f"    constexpr int FIXED_NORMAL = {int(SCENE_FRICTION_FIXED_NORMAL)};\n"
        head += f"    constexpr int FREE_BODY = {int(SCENE_FRICTION_FREE)};\n"
    shim = _HOST_SHIM_CPP
    if features.friction_mode != "fixed_normal" or features.torsion:
        # The friction modes' laws and contact torsion's budget name CUDA's rounding intrinsics.
        shim += _HOST_ROUNDING_SHIM_CPP
    if group > 1 or block:
        return shim + "#if defined(__CUDA_ARCH__)\n" + head + loop + body + "#endif\n"
    return shim + head + loop + body


def _has_phase_c(features: SceneFeatures) -> bool:
    """Whether the sweeps run phase C: with late rows (velocity limits) or contact torsion (spin rows)."""
    return features.late_rows or features.torsion


def _late_sweep(features: SceneFeatures, *, group: bool) -> str:
    """Phase C of one side under late rows: ``sweep_late(begin, end, late_begin, late_end)`` (or ``sweep_late_group``
    with ``lane`` and ``mask`` beside a lane-group loop), which returns ``changed``.

    It sweeps the side's late rows ``[late_begin, late_end)`` and, with the fused clamp, then walks its one-sided
    rows ``[begin, end)`` in a clamp pass, which clamps its drive rows (the velocity limits, then the fused clamp, in
    ``_get_pgs_solve_mf_gs_kernel``).
    """
    if group:
        name, params, sweep = (
            "sweep_late_group",
            "int begin, int end, int late_begin, int late_end, int lane, unsigned mask",
            "sweep_rows_group({}, lane, mask)",
        )
    else:
        name, params, sweep = "sweep_late", "int begin, int end, int late_begin, int late_end", "sweep_rows({})"
    clamp = (
        f"""        clamp_pass = true;
        changed |= {sweep.format("begin, end")};
        clamp_pass = false;
"""
        if features.fused_clamp
        else ""
    )
    what = "its late rows (velocity limits), then the fused clamp of its drive rows" if clamp else "its late rows"
    return (
        f"""
    // Phase C of one side: {what}.
    const auto {name} = [&]({params}) -> int {{
        int changed = {sweep.format("late_begin, late_end")};
"""
        + clamp
        + """        return changed;
    };
"""
    )


def _gated_sweeps(count: str, stationary: str) -> str:
    """A sweep loop under friction position iterations: sweep ``sweep`` skips the tangent pairs before
    ``friction_start``, and the stationary exit (``stationary``, evaluated after the sweep) waits for it.
    """
    return f"""
    for (int sweep = 0; sweep < {count}; ++sweep) {{
        friction_skipped = sweep < friction_start;
        if ({stationary} && !friction_skipped) break;
    }}
"""


# A launch of one sweep under friction position iterations: the host passes ``friction_start`` less
# the launch's iteration, so the sweep solves the tangent pairs from ``friction_start <= 0`` on.
_ONE_SWEEP_GATE_CUDA = """    friction_skipped = friction_start > 0;
"""


# How a thread of a lane-group launch finds its lane, its group's shuffle mask and its work item.
_GROUP_INDEX_CUDA = """
    const int item = tid / GROUP;
    const int lane = tid % GROUP;
    const unsigned mask = (0xffffffffu >> (32 - GROUP)) << ((tid & 31) & ~(GROUP - 1));
"""


def _sweep_sides_snippet(
    granularity: str, sums: str, support: int, free_response: bool, features: SceneFeatures
) -> str:
    """The native snippet of :func:`get_sweep_sides_kernel`; it raises the factory's :class:`ValueError`."""
    group = _check_sweep(granularity, sums, support)
    if _has_phase_c(features):
        return _sweep_sides_late_snippet(granularity, sums, support, free_response, features)
    gated = features.friction_position_iterations
    walk = """
    const int side_flag = side_coupled.data[side] != 0 ? 1 : 0;
    if (side_flag != coupled_pass) return;
    const int begin = side_row_begin.data[side];
    const int end = side_row_end.data[side];
    const int sweeps = coupled_pass != 0 ? 1 : iterations;
"""
    if group == 1:
        body = (
            """
    const int side = tid;
    if (side >= side_count) return;
"""
            + walk
            + (
                _gated_sweeps("sweeps", "!sweep_rows(begin, end)")
                if gated
                else """
    for (int sweep = 0; sweep < sweeps; ++sweep)
        if (!sweep_rows(begin, end)) break;
"""
            )
        )
    else:
        body = (
            _GROUP_INDEX_CUDA
            + """
    const int side = item;
    if (side >= side_count) return;
"""
            + walk
            + (
                _gated_sweeps("sweeps", "__any_sync(mask, sweep_rows_group(begin, end, lane, mask)) == 0")
                if gated
                else """
    for (int sweep = 0; sweep < sweeps; ++sweep)
        if (__any_sync(mask, sweep_rows_group(begin, end, lane, mask)) == 0) break;
"""
            )
        )

    return _sweep_snippet(granularity, sums, support, free_response, body, features=features)


def _sweep_sides_late_snippet(
    granularity: str, sums: str, support: int, free_response: bool, features: SceneFeatures
) -> str:
    """:func:`_sweep_sides_snippet` under late rows: each sweep of a side runs phase A, then phase C.

    The isolated pass (``coupled_pass == 0``) runs both in each of its sweeps, the coupled pass (``1``) phase A
    alone and the late pass (``2``) phase C alone, one sweep each, on the coupled sides. With contact torsion a sweep
    of a side runs its one-sided spin rows ``[side_torsion_begin, side_torsion_end)`` between phase A and phase C:
    in each sweep of the isolated pass, and alone in the torsion pass (``3``, :data:`SCENE_PASS_TORSION`), which
    precedes the coupled sides' two-sided spin rows and their late pass.
    """
    group = SCENE_SIDE_LANES[granularity]
    if group == 1:
        index = """
    const int side = tid;
    if (side >= side_count) return;
"""
        phase_a = "sweep_rows(begin, end)"
        phase_c = "sweep_late(begin, end, late_begin, late_end)"
        spin_rows = "sweep_rows(torsion_begin, torsion_end)"
        stationary = "!sweep_side()"
    else:
        index = (
            _GROUP_INDEX_CUDA
            + """
    const int side = item;
    if (side >= side_count) return;
"""
        )
        phase_a = "sweep_rows_group(begin, end, lane, mask)"
        phase_c = "sweep_late_group(begin, end, late_begin, late_end, lane, mask)"
        spin_rows = "sweep_rows_group(torsion_begin, torsion_end, lane, mask)"
        stationary = "__any_sync(mask, sweep_side()) == 0"
    if features.torsion:
        ranges = """    const int torsion_begin = side_torsion_begin.data[side];
    const int torsion_end = side_torsion_end.data[side];
"""
        side_sweep = f"""\
    // A sweep of the side: phase A (passes 0 and 1), its one-sided spin rows (C1, passes 0 and 3), then phase C
    // (passes 0 and 2).
    const auto sweep_side = [&]() -> int {{
        int changed = 0;
        if (coupled_pass < 2) changed |= {phase_a};
        if (coupled_pass == 0 || coupled_pass == 3) changed |= {spin_rows};
        if (coupled_pass == 0 || coupled_pass == 2) changed |= {phase_c};
        return changed;
    }};
"""
    else:
        ranges = ""
        side_sweep = f"""\
    // A sweep of the side: phase A (but in the late pass), then phase C (but in the coupled pass).
    const auto sweep_side = [&]() -> int {{
        int changed = 0;
        if (coupled_pass != 2) changed |= {phase_a};
        if (coupled_pass != 1) changed |= {phase_c};
        return changed;
    }};
"""
    body = (
        index
        + """
    // The coupled sides take passes 1 (phase A) and 2 (phase C, the late rows), one sweep each.
    const int side_flag = side_coupled.data[side] != 0 ? 1 : 0;
    if (side_flag != (coupled_pass != 0 ? 1 : 0)) return;
    const int begin = side_row_begin.data[side];
    const int end = side_row_end.data[side];
    const int late_begin = side_late_begin.data[side];
    const int late_end = side_late_end.data[side];
"""
        + ranges
        + """    const int sweeps = coupled_pass != 0 ? 1 : iterations;
"""
        + side_sweep
        + (
            _gated_sweeps("sweeps", stationary)
            if features.friction_position_iterations
            else f"""
    for (int sweep = 0; sweep < sweeps; ++sweep)
        if ({stationary}) break;
"""
        )
    )
    return _sweep_snippet(granularity, sums, support, free_response, body, phase_c=True, features=features)


@cache
def get_sweep_sides_kernel(
    granularity: Literal["thread", "lanes8", "lanes16", "lanes32"],
    sums: Literal["serial", "steel"],
    support: int,
    *,
    free_response: bool = False,
    features: SceneFeatures = SceneFeatures(),  # noqa: B008 (frozen)
) -> wp.Kernel:
    """Phases A and C: sweep each side's one-sided rows ``[side_row_begin[s], side_row_end[s])``, then its late rows.

    ``coupled_pass == 0`` takes the sides with ``side_coupled[s] == 0``: they share no unit with
    another side this step, so each runs all ``iterations`` sweeps in this one launch and stops
    after a sweep that changes no impulse (Steel's stationary exit, per side). Such a
    sweep is a bitwise fixed point, so the exit changes no result. ``coupled_pass == 1`` gives each
    coupled side one sweep. ``granularity`` is ``"thread"`` (one thread per side) or ``"lanes8"``,
    ``"lanes16"`` or ``"lanes32"`` (that many lanes per side, lane g holding entries g, g + lanes, ...).
    ``support`` is the rows' entries (``S``) and ``free_response`` whether rows may start with free-body entries.
    ``features`` are the build constants (:class:`SceneFeatures`): with any feature on, the kernel
    is the featured definition, which takes ``friction_start: int``, ``row_w``, ``row_bound``, the
    four drive descriptors, ``side_late_begin``, ``side_late_end``, ``drive_vel_limit`` and ``freeze_drives: int``
    after ``impulses``. With friction position iterations sweep ``k`` of the
    launch skips the tangent pairs while ``k < friction_start``, and the stationary exit waits for
    ``friction_start`` (the coupled pass passes it less its iteration). With regularization a normal or
    limit row's update reads its weight ``row_w[0, row]`` (Steel's matrix-free update where it is below
    1). With joint friction a joint-friction row's impulse is clamped to ``[-row_bound[0, row],
    row_bound[0, row]]``. With drives a drive row's impulse takes the force-drive update of its
    descriptors (:class:`SceneFeatures`), unless ``freeze_drives`` (the velocity-only iterations under
    ``velocity_drive_mode="freeze"``) leaves the row as it is. With late rows each sweep of a side then runs its
    phase C, its late rows ``[side_late_begin[s], side_late_end[s])`` (with the fused clamp, then its drive rows'
    clamp), and ``coupled_pass == 2`` gives each coupled side one phase-C sweep (``SceneSolve._sweep_late``).

    Launch with ``dim = side_count * SCENE_SIDE_LANES[granularity]`` and
    ``block_dim = SCENE_SWEEP_BLOCK_DIM``. CUDA only. Raises :class:`ValueError` for Steel-order
    sums over more than 32 entries.
    """

    suffix = _feature_suffix(features)
    if suffix:
        return _featured_sides_kernel(
            _sweep_sides_snippet(granularity, sums, support, free_response, features),
            f"steel_scene_sides_{granularity}_{sums}_{int(support)}_{int(free_response)}{suffix}",
        )

    @wp.func_native(_sweep_sides_snippet(granularity, sums, support, free_response, features))
    def sweep_sides_native(
        tid: int,
        side_count: int,
        coupled_pass: int,
        iterations: int,
        side_row_begin: wp.array[wp.int32],
        side_row_end: wp.array[wp.int32],
        side_coupled: wp.array[wp.int32],
        rows_dof: wp.array3d[wp.int32],
        rows_factor: wp.array3d[wp.float32],
        rows_free_response: wp.array3d[wp.float32],
        rows_free_prefix: wp.array[wp.int32],
        rows_rhs: wp.array2d[wp.float32],
        rows_incident: wp.array2d[wp.float32],
        rows_diag: wp.array2d[wp.float32],
        rows_type: wp.array2d[wp.int32],
        rows_parent: wp.array2d[wp.int32],
        rows_mu: wp.array2d[wp.float32],
        omega: float,
        delta: wp.array2d[wp.float32],
        impulses: wp.array2d[wp.float32],
    ): ...

    def sweep_sides(
        side_count: int,
        coupled_pass: int,
        iterations: int,
        side_row_begin: wp.array[wp.int32],
        side_row_end: wp.array[wp.int32],
        side_coupled: wp.array[wp.int32],
        rows_dof: wp.array3d[wp.int32],
        rows_factor: wp.array3d[wp.float32],
        rows_free_response: wp.array3d[wp.float32],
        rows_free_prefix: wp.array[wp.int32],
        rows_rhs: wp.array2d[wp.float32],
        rows_incident: wp.array2d[wp.float32],
        rows_diag: wp.array2d[wp.float32],
        rows_type: wp.array2d[wp.int32],
        rows_parent: wp.array2d[wp.int32],
        rows_mu: wp.array2d[wp.float32],
        omega: float,
        delta: wp.array2d[wp.float32],
        impulses: wp.array2d[wp.float32],
    ):
        tid = wp.tid()
        sweep_sides_native(
            tid,
            side_count,
            coupled_pass,
            iterations,
            side_row_begin,
            side_row_end,
            side_coupled,
            rows_dof,
            rows_factor,
            rows_free_response,
            rows_free_prefix,
            rows_rhs,
            rows_incident,
            rows_diag,
            rows_type,
            rows_parent,
            rows_mu,
            omega,
            delta,
            impulses,
        )

    sweep_sides.__name__ = f"steel_scene_sides_{granularity}_{sums}_{int(support)}_{int(free_response)}"
    sweep_sides.__qualname__ = sweep_sides.__name__
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(sweep_sides)


def _featured_sides_kernel(snippet: str, name: str) -> wp.Kernel:
    """The featured definition of :func:`get_sweep_sides_kernel`: its arguments, then every feature's."""

    @wp.func_native(snippet)
    def sweep_sides_native(
        tid: int,
        side_count: int,
        coupled_pass: int,
        iterations: int,
        side_row_begin: wp.array[wp.int32],
        side_row_end: wp.array[wp.int32],
        side_coupled: wp.array[wp.int32],
        rows_dof: wp.array3d[wp.int32],
        rows_factor: wp.array3d[wp.float32],
        rows_free_response: wp.array3d[wp.float32],
        rows_free_prefix: wp.array[wp.int32],
        rows_rhs: wp.array2d[wp.float32],
        rows_incident: wp.array2d[wp.float32],
        rows_diag: wp.array2d[wp.float32],
        rows_type: wp.array2d[wp.int32],
        rows_parent: wp.array2d[wp.int32],
        rows_mu: wp.array2d[wp.float32],
        omega: float,
        delta: wp.array2d[wp.float32],
        impulses: wp.array2d[wp.float32],
        friction_start: int,
        row_w: wp.array2d[wp.float32],
        row_bound: wp.array2d[wp.float32],
        drive_target_vel_bias: wp.array2d[wp.float32],
        drive_vel_multiplier: wp.array2d[wp.float32],
        drive_impulse_multiplier: wp.array2d[wp.float32],
        drive_max_impulse: wp.array2d[wp.float32],
        side_late_begin: wp.array[wp.int32],
        side_late_end: wp.array[wp.int32],
        drive_vel_limit: wp.array2d[wp.float32],
        freeze_drives: int,
        row_gamma: wp.array2d[wp.float32],
        torsion_radius: wp.array2d[wp.float32],
        torsion_group: wp.array2d[wp.int32],
        torsion_member_range: wp.array2d[wp.int32],
        torsion_member_rows: wp.array[wp.int32],
        side_torsion_begin: wp.array[wp.int32],
        side_torsion_end: wp.array[wp.int32],
        torsion_color_start: wp.array[wp.int32],
        row_friction_mode: wp.array2d[wp.int32],
    ): ...

    def sweep_sides(
        side_count: int,
        coupled_pass: int,
        iterations: int,
        side_row_begin: wp.array[wp.int32],
        side_row_end: wp.array[wp.int32],
        side_coupled: wp.array[wp.int32],
        rows_dof: wp.array3d[wp.int32],
        rows_factor: wp.array3d[wp.float32],
        rows_free_response: wp.array3d[wp.float32],
        rows_free_prefix: wp.array[wp.int32],
        rows_rhs: wp.array2d[wp.float32],
        rows_incident: wp.array2d[wp.float32],
        rows_diag: wp.array2d[wp.float32],
        rows_type: wp.array2d[wp.int32],
        rows_parent: wp.array2d[wp.int32],
        rows_mu: wp.array2d[wp.float32],
        omega: float,
        delta: wp.array2d[wp.float32],
        impulses: wp.array2d[wp.float32],
        friction_start: int,
        row_w: wp.array2d[wp.float32],
        row_bound: wp.array2d[wp.float32],
        drive_target_vel_bias: wp.array2d[wp.float32],
        drive_vel_multiplier: wp.array2d[wp.float32],
        drive_impulse_multiplier: wp.array2d[wp.float32],
        drive_max_impulse: wp.array2d[wp.float32],
        side_late_begin: wp.array[wp.int32],
        side_late_end: wp.array[wp.int32],
        drive_vel_limit: wp.array2d[wp.float32],
        freeze_drives: int,
        row_gamma: wp.array2d[wp.float32],
        torsion_radius: wp.array2d[wp.float32],
        torsion_group: wp.array2d[wp.int32],
        torsion_member_range: wp.array2d[wp.int32],
        torsion_member_rows: wp.array[wp.int32],
        side_torsion_begin: wp.array[wp.int32],
        side_torsion_end: wp.array[wp.int32],
        torsion_color_start: wp.array[wp.int32],
        row_friction_mode: wp.array2d[wp.int32],
    ):
        tid = wp.tid()
        sweep_sides_native(
            tid,
            side_count,
            coupled_pass,
            iterations,
            side_row_begin,
            side_row_end,
            side_coupled,
            rows_dof,
            rows_factor,
            rows_free_response,
            rows_free_prefix,
            rows_rhs,
            rows_incident,
            rows_diag,
            rows_type,
            rows_parent,
            rows_mu,
            omega,
            delta,
            impulses,
            friction_start,
            row_w,
            row_bound,
            drive_target_vel_bias,
            drive_vel_multiplier,
            drive_impulse_multiplier,
            drive_max_impulse,
            side_late_begin,
            side_late_end,
            drive_vel_limit,
            freeze_drives,
            row_gamma,
            torsion_radius,
            torsion_group,
            torsion_member_range,
            torsion_member_rows,
            side_torsion_begin,
            side_torsion_end,
            torsion_color_start,
            row_friction_mode,
        )

    sweep_sides.__name__ = name
    sweep_sides.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(sweep_sides)


def _sweep_rows_serial_snippet(sums: str, support: int, free_response: bool, features: SceneFeatures) -> str:
    """The native snippet of :func:`get_sweep_rows_serial_kernel`; it raises the factory's :class:`ValueError`."""
    _check_sweep("thread", sums, support)
    if features.fused_clamp:
        # The fused clamp after each walk of every row: its drive rows, once every row (the late rows last) is swept.
        # The two-sided ranges, which phase B's schedules walk, hold no drive row, so their walks skip the clamp.
        walk = (
            """
    if (tid != 0) return;
    const int begin = row_ranges.data[2 * range_index];
    const int end = row_ranges.data[2 * range_index + 1];
    // Each walk of every row ends with the fused clamp of its drive rows (Steel's clamp after its velocity limits);
    // the other ranges (the two-sided units, the tail) hold no drive row.
    const auto sweep_walk = [&]() -> int {
        int changed = sweep_rows(begin, end);
        if (range_index == """
            + str(int(RANGE_ALL))
            + """) {
            clamp_pass = true;
            changed |= sweep_rows(begin, end);
            clamp_pass = false;
        }
        return changed;
    };
"""
        )
        stationary = "!sweep_walk() && stationary_exit != 0"
        body = walk + (
            _gated_sweeps("sweeps", stationary).lstrip("\n")
            if features.friction_position_iterations
            else f"""    for (int sweep = 0; sweep < sweeps; ++sweep)
        if ({stationary}) break;
"""
        )
        return _sweep_snippet("thread", sums, support, free_response, body, features=features)
    if features.friction_position_iterations:
        body = """
    if (tid != 0) return;
    const int begin = row_ranges.data[2 * range_index];
    const int end = row_ranges.data[2 * range_index + 1];
""" + _gated_sweeps("sweeps", "!sweep_rows(begin, end) && stationary_exit != 0").lstrip("\n")
    else:
        body = """
    if (tid != 0) return;
    const int begin = row_ranges.data[2 * range_index];
    const int end = row_ranges.data[2 * range_index + 1];
    for (int sweep = 0; sweep < sweeps; ++sweep)
        if (!sweep_rows(begin, end) && stationary_exit != 0) break;
"""

    return _sweep_snippet("thread", sums, support, free_response, body, features=features)


@cache
def get_sweep_rows_serial_kernel(
    sums: Literal["serial", "steel"],
    support: int,
    *,
    free_response: bool = False,
    features: SceneFeatures = SceneFeatures(),  # noqa: B008 (frozen)
) -> wp.Kernel:
    """One thread walks the rows of ``row_ranges[range_index]`` in row order, ``sweeps`` times.

    With ``stationary_exit`` it stops after a sweep that changes no impulse. Over ``RANGE_ALL``
    with ``iterations`` sweeps and the exit this is the scene sweep's serial oracle
    (``scene_sweep="serial"``) in the phase order ``"AB"`` without contact torsion. With one sweep and no
    exit it walks the two-sided units over
    ``RANGE_TWO_SIDED`` (``scene_color_loop="serial"``, phase B's oracle) or the tail over
    ``RANGE_TAIL`` (``scene_color_loop="while"`` and ``"unroll"``). The phased sweep matches these
    oracles bit for bit when its arithmetic is this loop's: Steel's sums at either granularity, or
    one thread per side and per unit with serial sums. With 8 lanes and serial sums (the default for
    rows over 32 entries) the lanes add in another order, and no oracle is bitwise.
    ``support``, ``free_response`` and ``features`` are as :func:`get_sweep_sides_kernel`'s, with the
    same feature arguments: with friction position iterations sweep ``k`` gates the tangent pairs on
    ``friction_start`` and the exit waits for it; regularization, joint friction and drives act per
    row as there.

    Launch with ``dim = 1``, on CUDA or on the CPU (the host shim, :data:`_HOST_SHIM_CPP`). Raises
    :class:`ValueError` for Steel-order sums over more than 32 entries.
    """

    suffix = _feature_suffix(features)
    if suffix:
        return _featured_rows_serial_kernel(
            _sweep_rows_serial_snippet(sums, support, free_response, features),
            f"steel_scene_serial_{sums}_{int(support)}_{int(free_response)}{suffix}",
        )

    @wp.func_native(_sweep_rows_serial_snippet(sums, support, free_response, features))
    def sweep_rows_native(
        tid: int,
        range_index: int,
        sweeps: int,
        stationary_exit: int,
        row_ranges: wp.array2d[wp.int32],
        rows_dof: wp.array3d[wp.int32],
        rows_factor: wp.array3d[wp.float32],
        rows_free_response: wp.array3d[wp.float32],
        rows_free_prefix: wp.array[wp.int32],
        rows_rhs: wp.array2d[wp.float32],
        rows_incident: wp.array2d[wp.float32],
        rows_diag: wp.array2d[wp.float32],
        rows_type: wp.array2d[wp.int32],
        rows_parent: wp.array2d[wp.int32],
        rows_mu: wp.array2d[wp.float32],
        omega: float,
        delta: wp.array2d[wp.float32],
        impulses: wp.array2d[wp.float32],
    ): ...

    def sweep_rows_serial(
        range_index: int,
        sweeps: int,
        stationary_exit: int,
        row_ranges: wp.array2d[wp.int32],
        rows_dof: wp.array3d[wp.int32],
        rows_factor: wp.array3d[wp.float32],
        rows_free_response: wp.array3d[wp.float32],
        rows_free_prefix: wp.array[wp.int32],
        rows_rhs: wp.array2d[wp.float32],
        rows_incident: wp.array2d[wp.float32],
        rows_diag: wp.array2d[wp.float32],
        rows_type: wp.array2d[wp.int32],
        rows_parent: wp.array2d[wp.int32],
        rows_mu: wp.array2d[wp.float32],
        omega: float,
        delta: wp.array2d[wp.float32],
        impulses: wp.array2d[wp.float32],
    ):
        tid = wp.tid()
        sweep_rows_native(
            tid,
            range_index,
            sweeps,
            stationary_exit,
            row_ranges,
            rows_dof,
            rows_factor,
            rows_free_response,
            rows_free_prefix,
            rows_rhs,
            rows_incident,
            rows_diag,
            rows_type,
            rows_parent,
            rows_mu,
            omega,
            delta,
            impulses,
        )

    sweep_rows_serial.__name__ = f"steel_scene_serial_{sums}_{int(support)}_{int(free_response)}"
    sweep_rows_serial.__qualname__ = sweep_rows_serial.__name__
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(sweep_rows_serial)


def _featured_rows_serial_kernel(snippet: str, name: str) -> wp.Kernel:
    """The featured definition of :func:`get_sweep_rows_serial_kernel`: its arguments, then every feature's."""

    @wp.func_native(snippet)
    def sweep_rows_native(
        tid: int,
        range_index: int,
        sweeps: int,
        stationary_exit: int,
        row_ranges: wp.array2d[wp.int32],
        rows_dof: wp.array3d[wp.int32],
        rows_factor: wp.array3d[wp.float32],
        rows_free_response: wp.array3d[wp.float32],
        rows_free_prefix: wp.array[wp.int32],
        rows_rhs: wp.array2d[wp.float32],
        rows_incident: wp.array2d[wp.float32],
        rows_diag: wp.array2d[wp.float32],
        rows_type: wp.array2d[wp.int32],
        rows_parent: wp.array2d[wp.int32],
        rows_mu: wp.array2d[wp.float32],
        omega: float,
        delta: wp.array2d[wp.float32],
        impulses: wp.array2d[wp.float32],
        friction_start: int,
        row_w: wp.array2d[wp.float32],
        row_bound: wp.array2d[wp.float32],
        drive_target_vel_bias: wp.array2d[wp.float32],
        drive_vel_multiplier: wp.array2d[wp.float32],
        drive_impulse_multiplier: wp.array2d[wp.float32],
        drive_max_impulse: wp.array2d[wp.float32],
        side_late_begin: wp.array[wp.int32],
        side_late_end: wp.array[wp.int32],
        drive_vel_limit: wp.array2d[wp.float32],
        freeze_drives: int,
        row_gamma: wp.array2d[wp.float32],
        torsion_radius: wp.array2d[wp.float32],
        torsion_group: wp.array2d[wp.int32],
        torsion_member_range: wp.array2d[wp.int32],
        torsion_member_rows: wp.array[wp.int32],
        side_torsion_begin: wp.array[wp.int32],
        side_torsion_end: wp.array[wp.int32],
        torsion_color_start: wp.array[wp.int32],
        row_friction_mode: wp.array2d[wp.int32],
    ): ...

    def sweep_rows_serial(
        range_index: int,
        sweeps: int,
        stationary_exit: int,
        row_ranges: wp.array2d[wp.int32],
        rows_dof: wp.array3d[wp.int32],
        rows_factor: wp.array3d[wp.float32],
        rows_free_response: wp.array3d[wp.float32],
        rows_free_prefix: wp.array[wp.int32],
        rows_rhs: wp.array2d[wp.float32],
        rows_incident: wp.array2d[wp.float32],
        rows_diag: wp.array2d[wp.float32],
        rows_type: wp.array2d[wp.int32],
        rows_parent: wp.array2d[wp.int32],
        rows_mu: wp.array2d[wp.float32],
        omega: float,
        delta: wp.array2d[wp.float32],
        impulses: wp.array2d[wp.float32],
        friction_start: int,
        row_w: wp.array2d[wp.float32],
        row_bound: wp.array2d[wp.float32],
        drive_target_vel_bias: wp.array2d[wp.float32],
        drive_vel_multiplier: wp.array2d[wp.float32],
        drive_impulse_multiplier: wp.array2d[wp.float32],
        drive_max_impulse: wp.array2d[wp.float32],
        side_late_begin: wp.array[wp.int32],
        side_late_end: wp.array[wp.int32],
        drive_vel_limit: wp.array2d[wp.float32],
        freeze_drives: int,
        row_gamma: wp.array2d[wp.float32],
        torsion_radius: wp.array2d[wp.float32],
        torsion_group: wp.array2d[wp.int32],
        torsion_member_range: wp.array2d[wp.int32],
        torsion_member_rows: wp.array[wp.int32],
        side_torsion_begin: wp.array[wp.int32],
        side_torsion_end: wp.array[wp.int32],
        torsion_color_start: wp.array[wp.int32],
        row_friction_mode: wp.array2d[wp.int32],
    ):
        tid = wp.tid()
        sweep_rows_native(
            tid,
            range_index,
            sweeps,
            stationary_exit,
            row_ranges,
            rows_dof,
            rows_factor,
            rows_free_response,
            rows_free_prefix,
            rows_rhs,
            rows_incident,
            rows_diag,
            rows_type,
            rows_parent,
            rows_mu,
            omega,
            delta,
            impulses,
            friction_start,
            row_w,
            row_bound,
            drive_target_vel_bias,
            drive_vel_multiplier,
            drive_impulse_multiplier,
            drive_max_impulse,
            side_late_begin,
            side_late_end,
            drive_vel_limit,
            freeze_drives,
            row_gamma,
            torsion_radius,
            torsion_group,
            torsion_member_range,
            torsion_member_rows,
            side_torsion_begin,
            side_torsion_end,
            torsion_color_start,
            row_friction_mode,
        )

    sweep_rows_serial.__name__ = name
    sweep_rows_serial.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(sweep_rows_serial)


def _sweep_order_serial_snippet(
    phase_order: str, sums: str, support: int, free_response: bool, features: SceneFeatures
) -> str:
    """The native snippet of :func:`get_sweep_order_serial_kernel`; it raises the factory's :class:`ValueError`."""
    if phase_order not in SCENE_PHASE_ORDERS:
        raise ValueError(f"scene_phase_order must be one of {list(SCENE_PHASE_ORDERS)}, got {phase_order!r}")
    _check_sweep("thread", sums, support)
    phases = {
        "A": "        changed |= sweep_sides(1);\n",
        "B": "        changed |= sweep_rows(two_sided_begin, two_sided_end);\n",
    }
    # Phase C (late rows): the late rows of the isolated (0) or the coupled (1) sides, side by side, each side's after
    # its phase A in a sweep of the isolated sides and after the whole coupled pass of the coupled ones.
    late = _has_phase_c(features)
    sequence = "        int changed = sweep_sides(0);\n" + "".join(phases[phase] for phase in phase_order)
    if features.torsion:
        # Contact torsion's spin rows lead phase C: 5 and 6 the one-sided ones of the isolated and the coupled sides
        # (each side's between its phases A and C), 7 the two-sided ones in row order, the color pass's oracle; the
        # two-sided units' and the two-sided spin rows' walks share one call site.
        passes = [0, 5, 3, *({"A": 1, "B": 2}[phase] for phase in phase_order), 6, 7, 4]
        sequence = f"""        const int passes[{len(passes)}] = {{{", ".join(str(kind) for kind in passes)}}};
        int changed = 0;
        #pragma unroll 1
        for (int pass = 0; pass < {len(passes)}; ++pass) {{
            const int kind = passes[pass];
            if (kind == 2 || kind == 7)
                changed |= sweep_rows(kind == 2 ? two_sided_begin : torsion_two_begin,
                                      kind == 2 ? two_sided_end : torsion_two_end);
            else if (kind < 2) changed |= sweep_sides(kind);
            else if (kind < 5) changed |= sweep_late_sides(kind - 3);
            else changed |= sweep_torsion_sides(kind - 5);
        }}
"""
    elif late:
        # Each phase's lambda has one call site, so the isolated and the coupled sides run one compiled copy of the
        # row loop: under FMA contraction two inlined copies of one loop may round differently (with the fused clamp
        # the isolated sides' copies parted from the coupled sides' and from the phases). The sweep
        # walks a pass list in the order's sequence: 0 and 1 phase A of the isolated and the coupled sides, 2
        # phase B, 3 and 4 phase C of the isolated and the coupled sides.
        passes = [0, 3, *({"A": 1, "B": 2}[phase] for phase in phase_order), 4]
        sequence = f"""        const int passes[{len(passes)}] = {{{", ".join(str(kind) for kind in passes)}}};
        int changed = 0;
        #pragma unroll 1
        for (int pass = 0; pass < {len(passes)}; ++pass) {{
            const int kind = passes[pass];
            if (kind == 2) changed |= sweep_rows(two_sided_begin, two_sided_end);
            else if (kind < 2) changed |= sweep_sides(kind);
            else changed |= sweep_late_sides(kind - 3);
        }}
"""
    late_sides = (
        """    const auto sweep_late_sides = [&](int coupled) -> int {
        int changed = 0;
        for (int side = 0; side < side_count; ++side)
            if ((side_coupled.data[side] != 0 ? 1 : 0) == coupled)
                changed |= sweep_late(side_row_begin.data[side], side_row_end.data[side],
                                      side_late_begin.data[side], side_late_end.data[side]);
        return changed;
    };
"""
        if late
        else ""
    )
    torsion_sides = (
        """\
    // The one-sided spin rows of the isolated (0) or the coupled (1) sides, side by side; then the two-sided ones.
    const auto sweep_torsion_sides = [&](int coupled) -> int {
        int changed = 0;
        for (int side = 0; side < side_count; ++side)
            if ((side_coupled.data[side] != 0 ? 1 : 0) == coupled)
                changed |= sweep_rows(side_torsion_begin.data[side], side_torsion_end.data[side]);
        return changed;
    };
    const int torsion_two_begin = torsion_color_start.data[0];
    const int torsion_two_end = torsion_color_start.data[torsion_color_start.shape[0] - 1];
"""
        if features.torsion
        else ""
    )
    body = (
        f"""
    if (tid != 0) return;
    const int two_sided_begin = row_ranges.data[2 * {RANGE_TWO_SIDED}];
    const int two_sided_end = row_ranges.data[2 * {RANGE_TWO_SIDED} + 1];
    // The one-sided rows of the isolated (0) or the coupled (1) sides, side by side.
    const auto sweep_sides = [&](int coupled) -> int {{
        int changed = 0;
        for (int side = 0; side < side_count; ++side)
            if ((side_coupled.data[side] != 0 ? 1 : 0) == coupled)
                changed |= sweep_rows(side_row_begin.data[side], side_row_end.data[side]);
        return changed;
    }};
"""
        + late_sides
        + torsion_sides
        + """    for (int sweep = 0; sweep < iterations; ++sweep) {
"""
        + ("        friction_skipped = sweep < friction_start;\n" if features.friction_position_iterations else "")
        + sequence
        + (
            "        if (!changed && !friction_skipped) break;\n"
            if features.friction_position_iterations
            else "        if (!changed) break;\n"
        )
        + """    }
"""
    )

    return _sweep_snippet("thread", sums, support, free_response, body, phase_c=late, features=features)


@cache
def get_sweep_order_serial_kernel(
    phase_order: Literal["AB", "BA", "ABA"],
    sums: Literal["serial", "steel"],
    support: int,
    *,
    free_response: bool = False,
    features: SceneFeatures = SceneFeatures(),  # noqa: B008 (frozen)
) -> wp.Kernel:
    """The serial oracle of the scene sweep in a phase order (``scene_phase_order``) other than ``"AB"``.

    One thread runs ``iterations`` sweeps and stops after a sweep that changes no impulse. Each
    sweep walks the one-sided rows of the sides without two-sided units (``side_coupled[s] == 0``)
    side by side, then the order's phases: ``"A"`` walks the coupled sides' one-sided rows side by
    side, ``[side_row_begin[s], side_row_end[s])`` for each side with ``side_coupled[s] == 1``, and
    ``"B"`` the two-sided units' rows ``row_ranges[RANGE_TWO_SIDED]`` in row order (the colors, then
    the tail). The coupled sides' one-sided rows are not contiguous, since the isolated sides' rows
    sit between them. The isolated sides share no row or coordinate with any other side, so where
    their rows fall in a sweep cannot matter, and a sweep that changes no impulse is a bitwise fixed
    point; the phased sweep therefore matches this walk bit for bit under the same arithmetic as
    :func:`get_sweep_rows_serial_kernel`. For ``"AB"`` this walk differs from that kernel's over
    ``RANGE_ALL`` only in where the isolated sides' rows fall, and that kernel remains ``"AB"``'s
    oracle. ``sums``, ``support``, ``free_response`` and ``features`` are as
    :func:`get_sweep_sides_kernel`'s, with the same feature arguments: with friction position
    iterations sweep ``k`` gates the tangent pairs on ``friction_start`` and the exit waits for it;
    regularization, joint friction and drives act per row as there. With late rows each sweep ends with phase C: the
    isolated sides' late rows after their phase A, and the coupled sides' after the order's phases. With contact
    torsion phase C starts with the spin rows, as the phased sweep runs them: each isolated side's one-sided spin rows
    between its phases A and C, and after the order's phases the coupled sides' one-sided spin rows side by side, the
    two-sided ones in row order (``[torsion_color_start[0], torsion_color_start[-1])``, colors then the tail), then
    the coupled sides' late rows. The spin rows follow the late rows in row order, so under contact torsion this walk
    is the oracle of ``"AB"`` too.

    Launch with ``dim = 1``, on CUDA or on the CPU (the host shim, :data:`_HOST_SHIM_CPP`). Raises
    :class:`ValueError` for an unknown order or for Steel-order sums over more than 32 entries.
    """

    suffix = _feature_suffix(features)
    if suffix:
        return _featured_order_serial_kernel(
            _sweep_order_serial_snippet(phase_order, sums, support, free_response, features),
            f"steel_scene_serial_{phase_order}_{sums}_{int(support)}_{int(free_response)}{suffix}",
        )

    @wp.func_native(_sweep_order_serial_snippet(phase_order, sums, support, free_response, features))
    def sweep_order_native(
        tid: int,
        side_count: int,
        iterations: int,
        side_row_begin: wp.array[wp.int32],
        side_row_end: wp.array[wp.int32],
        side_coupled: wp.array[wp.int32],
        row_ranges: wp.array2d[wp.int32],
        rows_dof: wp.array3d[wp.int32],
        rows_factor: wp.array3d[wp.float32],
        rows_free_response: wp.array3d[wp.float32],
        rows_free_prefix: wp.array[wp.int32],
        rows_rhs: wp.array2d[wp.float32],
        rows_incident: wp.array2d[wp.float32],
        rows_diag: wp.array2d[wp.float32],
        rows_type: wp.array2d[wp.int32],
        rows_parent: wp.array2d[wp.int32],
        rows_mu: wp.array2d[wp.float32],
        omega: float,
        delta: wp.array2d[wp.float32],
        impulses: wp.array2d[wp.float32],
    ): ...

    def sweep_order_serial(
        side_count: int,
        iterations: int,
        side_row_begin: wp.array[wp.int32],
        side_row_end: wp.array[wp.int32],
        side_coupled: wp.array[wp.int32],
        row_ranges: wp.array2d[wp.int32],
        rows_dof: wp.array3d[wp.int32],
        rows_factor: wp.array3d[wp.float32],
        rows_free_response: wp.array3d[wp.float32],
        rows_free_prefix: wp.array[wp.int32],
        rows_rhs: wp.array2d[wp.float32],
        rows_incident: wp.array2d[wp.float32],
        rows_diag: wp.array2d[wp.float32],
        rows_type: wp.array2d[wp.int32],
        rows_parent: wp.array2d[wp.int32],
        rows_mu: wp.array2d[wp.float32],
        omega: float,
        delta: wp.array2d[wp.float32],
        impulses: wp.array2d[wp.float32],
    ):
        tid = wp.tid()
        sweep_order_native(
            tid,
            side_count,
            iterations,
            side_row_begin,
            side_row_end,
            side_coupled,
            row_ranges,
            rows_dof,
            rows_factor,
            rows_free_response,
            rows_free_prefix,
            rows_rhs,
            rows_incident,
            rows_diag,
            rows_type,
            rows_parent,
            rows_mu,
            omega,
            delta,
            impulses,
        )

    sweep_order_serial.__name__ = f"steel_scene_serial_{phase_order}_{sums}_{int(support)}_{int(free_response)}"
    sweep_order_serial.__qualname__ = sweep_order_serial.__name__
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(sweep_order_serial)


def _featured_order_serial_kernel(snippet: str, name: str) -> wp.Kernel:
    """The featured definition of :func:`get_sweep_order_serial_kernel`: its arguments, then every feature's."""

    @wp.func_native(snippet)
    def sweep_order_native(
        tid: int,
        side_count: int,
        iterations: int,
        side_row_begin: wp.array[wp.int32],
        side_row_end: wp.array[wp.int32],
        side_coupled: wp.array[wp.int32],
        row_ranges: wp.array2d[wp.int32],
        rows_dof: wp.array3d[wp.int32],
        rows_factor: wp.array3d[wp.float32],
        rows_free_response: wp.array3d[wp.float32],
        rows_free_prefix: wp.array[wp.int32],
        rows_rhs: wp.array2d[wp.float32],
        rows_incident: wp.array2d[wp.float32],
        rows_diag: wp.array2d[wp.float32],
        rows_type: wp.array2d[wp.int32],
        rows_parent: wp.array2d[wp.int32],
        rows_mu: wp.array2d[wp.float32],
        omega: float,
        delta: wp.array2d[wp.float32],
        impulses: wp.array2d[wp.float32],
        friction_start: int,
        row_w: wp.array2d[wp.float32],
        row_bound: wp.array2d[wp.float32],
        drive_target_vel_bias: wp.array2d[wp.float32],
        drive_vel_multiplier: wp.array2d[wp.float32],
        drive_impulse_multiplier: wp.array2d[wp.float32],
        drive_max_impulse: wp.array2d[wp.float32],
        side_late_begin: wp.array[wp.int32],
        side_late_end: wp.array[wp.int32],
        drive_vel_limit: wp.array2d[wp.float32],
        freeze_drives: int,
        row_gamma: wp.array2d[wp.float32],
        torsion_radius: wp.array2d[wp.float32],
        torsion_group: wp.array2d[wp.int32],
        torsion_member_range: wp.array2d[wp.int32],
        torsion_member_rows: wp.array[wp.int32],
        side_torsion_begin: wp.array[wp.int32],
        side_torsion_end: wp.array[wp.int32],
        torsion_color_start: wp.array[wp.int32],
        row_friction_mode: wp.array2d[wp.int32],
    ): ...

    def sweep_order_serial(
        side_count: int,
        iterations: int,
        side_row_begin: wp.array[wp.int32],
        side_row_end: wp.array[wp.int32],
        side_coupled: wp.array[wp.int32],
        row_ranges: wp.array2d[wp.int32],
        rows_dof: wp.array3d[wp.int32],
        rows_factor: wp.array3d[wp.float32],
        rows_free_response: wp.array3d[wp.float32],
        rows_free_prefix: wp.array[wp.int32],
        rows_rhs: wp.array2d[wp.float32],
        rows_incident: wp.array2d[wp.float32],
        rows_diag: wp.array2d[wp.float32],
        rows_type: wp.array2d[wp.int32],
        rows_parent: wp.array2d[wp.int32],
        rows_mu: wp.array2d[wp.float32],
        omega: float,
        delta: wp.array2d[wp.float32],
        impulses: wp.array2d[wp.float32],
        friction_start: int,
        row_w: wp.array2d[wp.float32],
        row_bound: wp.array2d[wp.float32],
        drive_target_vel_bias: wp.array2d[wp.float32],
        drive_vel_multiplier: wp.array2d[wp.float32],
        drive_impulse_multiplier: wp.array2d[wp.float32],
        drive_max_impulse: wp.array2d[wp.float32],
        side_late_begin: wp.array[wp.int32],
        side_late_end: wp.array[wp.int32],
        drive_vel_limit: wp.array2d[wp.float32],
        freeze_drives: int,
        row_gamma: wp.array2d[wp.float32],
        torsion_radius: wp.array2d[wp.float32],
        torsion_group: wp.array2d[wp.int32],
        torsion_member_range: wp.array2d[wp.int32],
        torsion_member_rows: wp.array[wp.int32],
        side_torsion_begin: wp.array[wp.int32],
        side_torsion_end: wp.array[wp.int32],
        torsion_color_start: wp.array[wp.int32],
        row_friction_mode: wp.array2d[wp.int32],
    ):
        tid = wp.tid()
        sweep_order_native(
            tid,
            side_count,
            iterations,
            side_row_begin,
            side_row_end,
            side_coupled,
            row_ranges,
            rows_dof,
            rows_factor,
            rows_free_response,
            rows_free_prefix,
            rows_rhs,
            rows_incident,
            rows_diag,
            rows_type,
            rows_parent,
            rows_mu,
            omega,
            delta,
            impulses,
            friction_start,
            row_w,
            row_bound,
            drive_target_vel_bias,
            drive_vel_multiplier,
            drive_impulse_multiplier,
            drive_max_impulse,
            side_late_begin,
            side_late_end,
            drive_vel_limit,
            freeze_drives,
            row_gamma,
            torsion_radius,
            torsion_group,
            torsion_member_range,
            torsion_member_rows,
            side_torsion_begin,
            side_torsion_end,
            torsion_color_start,
            row_friction_mode,
        )

    sweep_order_serial.__name__ = name
    sweep_order_serial.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(sweep_order_serial)


# What a phase-B kernel's body needs before its units: the kept rows and the two-sided units' first entry.
_UNIT_ENTRY_CUDA = """
    // Entry e of the row order owns rows [seq_row_start[e], seq_row_start[e + 1]); a unit whose rows
    // end past the kept rows was dropped past row_max (the kept units are a prefix of the order).
    const int rows = constraint_count.data[0];
    const int two_sided_entry = one_sided_start.data[side_count];
"""


def _sweep_color_snippet(
    granularity: str, sums: str, support: int, free_response: bool, features: SceneFeatures
) -> str:
    """The native snippet of :func:`get_sweep_color_kernel`; it raises the factory's :class:`ValueError`."""
    group = _check_sweep(granularity, sums, support, key="scene_unit_sweep")
    head = (
        """
    const int item = tid;
"""
        if group == 1
        else _GROUP_INDEX_CUDA
    )
    sweep = "sweep_rows(begin, end)" if group == 1 else "sweep_rows_group(begin, end, lane, mask)"
    body = (
        head
        + _UNIT_ENTRY_CUDA
        + (_ONE_SWEEP_GATE_CUDA if features.friction_position_iterations else "")
        + f"""
    const int color = color_param >= 0 ? color_param : color_cursor.data[0];
    const int first = two_sided_entry + color_offsets.data[color];
    const int count = color_offsets.data[color + 1] - color_offsets.data[color];
    for (int position = item; position < count; position += workers) {{
        const int begin = seq_row_start.data[first + position];
        const int end = seq_row_start.data[first + position + 1];
        if (end <= rows) {sweep};
    }}
"""
    )

    return _sweep_snippet(granularity, sums, support, free_response, body, features=features)


@cache
def get_sweep_color_kernel(
    granularity: Literal["thread", "lanes8", "lanes16", "lanes32"],
    sums: Literal["serial", "steel"],
    support: int,
    *,
    free_response: bool = False,
    features: SceneFeatures = SceneFeatures(),  # noqa: B008 (frozen)
) -> wp.Kernel:
    """Phase B: one sweep of one color's two-sided units, every unit of the color at once.

    The color is ``color`` when it is non-negative (``scene_color_loop="unroll"``, one launch per
    color) and ``color_cursor[0]`` otherwise (the while loop, which :func:`advance_color` moves).
    Unit ``i`` of color ``c`` is entry ``one_sided_start[side_count] + color_offsets[c] + i`` of the
    row order, and it sweeps its rows ``[seq_row_start[entry], seq_row_start[entry + 1])`` once with
    the row loop of :data:`SCENE_ROWS_CUDA`; a unit dropped past ``row_max`` is skipped, and an empty
    color returns at once. Units of one color share no side, so they touch disjoint coordinates and
    impulses and may run in any order: ``workers`` work items (one thread, or 8 lanes as
    :func:`get_sweep_sides_kernel`) stride over the color's units. With any feature on
    (``features``) the kernel takes ``friction_start: int``, ``row_w``, ``row_bound`` and the drive
    descriptors last, as :func:`get_sweep_sides_kernel`: ``friction_start`` has the launch's iteration
    already subtracted, and its one sweep skips the tangent pairs while ``friction_start > 0``.

    Launch with ``dim = workers * SCENE_UNIT_LANES[granularity]``, at most
    :data:`SCENE_COLOR_THREADS_MAX`, and ``block_dim = SCENE_SWEEP_BLOCK_DIM``. CUDA only. Raises
    :class:`ValueError` for Steel-order sums over more than 32 entries.
    """

    suffix = _feature_suffix(features)
    if suffix:
        return _featured_color_kernel(
            _sweep_color_snippet(granularity, sums, support, free_response, features),
            f"steel_scene_color_{granularity}_{sums}_{int(support)}_{int(free_response)}{suffix}",
        )

    @wp.func_native(_sweep_color_snippet(granularity, sums, support, free_response, features))
    def sweep_color_native(
        tid: int,
        color_param: int,
        workers: int,
        side_count: int,
        color_cursor: wp.array[wp.int32],
        color_offsets: wp.array[wp.int32],
        one_sided_start: wp.array[wp.int32],
        seq_row_start: wp.array[wp.int32],
        constraint_count: wp.array[wp.int32],
        rows_dof: wp.array3d[wp.int32],
        rows_factor: wp.array3d[wp.float32],
        rows_free_response: wp.array3d[wp.float32],
        rows_free_prefix: wp.array[wp.int32],
        rows_rhs: wp.array2d[wp.float32],
        rows_incident: wp.array2d[wp.float32],
        rows_diag: wp.array2d[wp.float32],
        rows_type: wp.array2d[wp.int32],
        rows_parent: wp.array2d[wp.int32],
        rows_mu: wp.array2d[wp.float32],
        omega: float,
        delta: wp.array2d[wp.float32],
        impulses: wp.array2d[wp.float32],
    ): ...

    def sweep_color(
        color_param: int,
        workers: int,
        side_count: int,
        color_cursor: wp.array[wp.int32],
        color_offsets: wp.array[wp.int32],
        one_sided_start: wp.array[wp.int32],
        seq_row_start: wp.array[wp.int32],
        constraint_count: wp.array[wp.int32],
        rows_dof: wp.array3d[wp.int32],
        rows_factor: wp.array3d[wp.float32],
        rows_free_response: wp.array3d[wp.float32],
        rows_free_prefix: wp.array[wp.int32],
        rows_rhs: wp.array2d[wp.float32],
        rows_incident: wp.array2d[wp.float32],
        rows_diag: wp.array2d[wp.float32],
        rows_type: wp.array2d[wp.int32],
        rows_parent: wp.array2d[wp.int32],
        rows_mu: wp.array2d[wp.float32],
        omega: float,
        delta: wp.array2d[wp.float32],
        impulses: wp.array2d[wp.float32],
    ):
        tid = wp.tid()
        sweep_color_native(
            tid,
            color_param,
            workers,
            side_count,
            color_cursor,
            color_offsets,
            one_sided_start,
            seq_row_start,
            constraint_count,
            rows_dof,
            rows_factor,
            rows_free_response,
            rows_free_prefix,
            rows_rhs,
            rows_incident,
            rows_diag,
            rows_type,
            rows_parent,
            rows_mu,
            omega,
            delta,
            impulses,
        )

    sweep_color.__name__ = f"steel_scene_color_{granularity}_{sums}_{int(support)}_{int(free_response)}"
    sweep_color.__qualname__ = sweep_color.__name__
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(sweep_color)


def _featured_color_kernel(snippet: str, name: str) -> wp.Kernel:
    """The featured definition of :func:`get_sweep_color_kernel`: its arguments, then every feature's."""

    @wp.func_native(snippet)
    def sweep_color_native(
        tid: int,
        color_param: int,
        workers: int,
        side_count: int,
        color_cursor: wp.array[wp.int32],
        color_offsets: wp.array[wp.int32],
        one_sided_start: wp.array[wp.int32],
        seq_row_start: wp.array[wp.int32],
        constraint_count: wp.array[wp.int32],
        rows_dof: wp.array3d[wp.int32],
        rows_factor: wp.array3d[wp.float32],
        rows_free_response: wp.array3d[wp.float32],
        rows_free_prefix: wp.array[wp.int32],
        rows_rhs: wp.array2d[wp.float32],
        rows_incident: wp.array2d[wp.float32],
        rows_diag: wp.array2d[wp.float32],
        rows_type: wp.array2d[wp.int32],
        rows_parent: wp.array2d[wp.int32],
        rows_mu: wp.array2d[wp.float32],
        omega: float,
        delta: wp.array2d[wp.float32],
        impulses: wp.array2d[wp.float32],
        friction_start: int,
        row_w: wp.array2d[wp.float32],
        row_bound: wp.array2d[wp.float32],
        drive_target_vel_bias: wp.array2d[wp.float32],
        drive_vel_multiplier: wp.array2d[wp.float32],
        drive_impulse_multiplier: wp.array2d[wp.float32],
        drive_max_impulse: wp.array2d[wp.float32],
        side_late_begin: wp.array[wp.int32],
        side_late_end: wp.array[wp.int32],
        drive_vel_limit: wp.array2d[wp.float32],
        freeze_drives: int,
        row_gamma: wp.array2d[wp.float32],
        torsion_radius: wp.array2d[wp.float32],
        torsion_group: wp.array2d[wp.int32],
        torsion_member_range: wp.array2d[wp.int32],
        torsion_member_rows: wp.array[wp.int32],
        side_torsion_begin: wp.array[wp.int32],
        side_torsion_end: wp.array[wp.int32],
        torsion_color_start: wp.array[wp.int32],
        row_friction_mode: wp.array2d[wp.int32],
    ): ...

    def sweep_color(
        color_param: int,
        workers: int,
        side_count: int,
        color_cursor: wp.array[wp.int32],
        color_offsets: wp.array[wp.int32],
        one_sided_start: wp.array[wp.int32],
        seq_row_start: wp.array[wp.int32],
        constraint_count: wp.array[wp.int32],
        rows_dof: wp.array3d[wp.int32],
        rows_factor: wp.array3d[wp.float32],
        rows_free_response: wp.array3d[wp.float32],
        rows_free_prefix: wp.array[wp.int32],
        rows_rhs: wp.array2d[wp.float32],
        rows_incident: wp.array2d[wp.float32],
        rows_diag: wp.array2d[wp.float32],
        rows_type: wp.array2d[wp.int32],
        rows_parent: wp.array2d[wp.int32],
        rows_mu: wp.array2d[wp.float32],
        omega: float,
        delta: wp.array2d[wp.float32],
        impulses: wp.array2d[wp.float32],
        friction_start: int,
        row_w: wp.array2d[wp.float32],
        row_bound: wp.array2d[wp.float32],
        drive_target_vel_bias: wp.array2d[wp.float32],
        drive_vel_multiplier: wp.array2d[wp.float32],
        drive_impulse_multiplier: wp.array2d[wp.float32],
        drive_max_impulse: wp.array2d[wp.float32],
        side_late_begin: wp.array[wp.int32],
        side_late_end: wp.array[wp.int32],
        drive_vel_limit: wp.array2d[wp.float32],
        freeze_drives: int,
        row_gamma: wp.array2d[wp.float32],
        torsion_radius: wp.array2d[wp.float32],
        torsion_group: wp.array2d[wp.int32],
        torsion_member_range: wp.array2d[wp.int32],
        torsion_member_rows: wp.array[wp.int32],
        side_torsion_begin: wp.array[wp.int32],
        side_torsion_end: wp.array[wp.int32],
        torsion_color_start: wp.array[wp.int32],
        row_friction_mode: wp.array2d[wp.int32],
    ):
        tid = wp.tid()
        sweep_color_native(
            tid,
            color_param,
            workers,
            side_count,
            color_cursor,
            color_offsets,
            one_sided_start,
            seq_row_start,
            constraint_count,
            rows_dof,
            rows_factor,
            rows_free_response,
            rows_free_prefix,
            rows_rhs,
            rows_incident,
            rows_diag,
            rows_type,
            rows_parent,
            rows_mu,
            omega,
            delta,
            impulses,
            friction_start,
            row_w,
            row_bound,
            drive_target_vel_bias,
            drive_vel_multiplier,
            drive_impulse_multiplier,
            drive_max_impulse,
            side_late_begin,
            side_late_end,
            drive_vel_limit,
            freeze_drives,
            row_gamma,
            torsion_radius,
            torsion_group,
            torsion_member_range,
            torsion_member_rows,
            side_torsion_begin,
            side_torsion_end,
            torsion_color_start,
            row_friction_mode,
        )

    sweep_color.__name__ = name
    sweep_color.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(sweep_color)


def _sweep_block_snippet(
    granularity: str, sums: str, support: int, free_response: bool, features: SceneFeatures
) -> str:
    """The native snippet of :func:`get_sweep_block_kernel`; it raises the factory's :class:`ValueError`."""
    group = _check_sweep(granularity, sums, support, key="scene_unit_sweep")
    head = (
        """
    const int item = tid;
"""
        if group == 1
        else _GROUP_INDEX_CUDA
    )
    sweep = "sweep_rows(begin, end)" if group == 1 else "sweep_rows_group(begin, end, lane, mask)"
    body = (
        head
        + _UNIT_ENTRY_CUDA
        + (_ONE_SWEEP_GATE_CUDA if features.friction_position_iterations else "")
        + f"""
    constexpr int WORKERS = {SCENE_BLOCK_THREADS // group};
    const int used = used_colors.data[0];
    for (int color = big_colors.data[0]; color < used; ++color) {{
        const int first = two_sided_entry + color_offsets.data[color];
        const int count = color_offsets.data[color + 1] - color_offsets.data[color];
        for (int position = item; position < count; position += WORKERS) {{
            const int begin = seq_row_start.data[first + position];
            const int end = seq_row_start.data[first + position + 1];
            if (end <= rows) {sweep};
        }}
        // The color's writes are visible to the whole block before the next color reads them.
        __syncthreads();
    }}
    // The tail: units past the color cap may share sides, so one thread walks their rows in order.
    if (tid == 0) sweep_rows(row_ranges.data[2 * {RANGE_TAIL}], row_ranges.data[2 * {RANGE_TAIL} + 1]);
"""
    )

    return _sweep_snippet(
        granularity, sums, support, free_response, body, thread_loop=True, block=True, features=features
    )


@cache
def get_sweep_block_kernel(
    granularity: Literal["thread", "lanes8", "lanes16", "lanes32"],
    sums: Literal["serial", "steel"],
    support: int,
    *,
    free_response: bool = False,
    features: SceneFeatures = SceneFeatures(),  # noqa: B008 (frozen)
) -> wp.Kernel:
    """Phase B in one block: colors ``[big_colors[0], used_colors[0])`` in order, then the tail.

    The block's :data:`SCENE_BLOCK_THREADS` threads (work items of one thread or 8 lanes, as
    :func:`get_sweep_color_kernel`'s) stride over each color's units, and ``__syncthreads()``
    after each color makes its writes visible to the next. Thread 0 then walks the tail's rows
    ``row_ranges[RANGE_TAIL]`` once in row order with the one-thread loop: tail units may share
    sides. A small color costs this block one pass of its work items instead of a conditional round
    of a full grid (up to :data:`SCENE_BLOCK_COLOR_MAX` units); a big color still runs here,
    correctly but on one block, when it follows a small one. With any feature on (``features``) the
    kernel takes ``friction_start: int``, ``row_w``, ``row_bound`` and the drive descriptors last, as
    :func:`get_sweep_color_kernel`, for the colors and the tail.

    Launch with ``dim = block_dim = SCENE_BLOCK_THREADS``. CUDA only. Raises :class:`ValueError` for
    Steel-order sums over more than 32 entries.
    """

    suffix = _feature_suffix(features)
    if suffix:
        return _featured_block_kernel(
            _sweep_block_snippet(granularity, sums, support, free_response, features),
            f"steel_scene_block_{granularity}_{sums}_{int(support)}_{int(free_response)}{suffix}",
        )

    @wp.func_native(_sweep_block_snippet(granularity, sums, support, free_response, features))
    def sweep_block_native(
        tid: int,
        side_count: int,
        big_colors: wp.array[wp.int32],
        used_colors: wp.array[wp.int32],
        row_ranges: wp.array2d[wp.int32],
        color_offsets: wp.array[wp.int32],
        one_sided_start: wp.array[wp.int32],
        seq_row_start: wp.array[wp.int32],
        constraint_count: wp.array[wp.int32],
        rows_dof: wp.array3d[wp.int32],
        rows_factor: wp.array3d[wp.float32],
        rows_free_response: wp.array3d[wp.float32],
        rows_free_prefix: wp.array[wp.int32],
        rows_rhs: wp.array2d[wp.float32],
        rows_incident: wp.array2d[wp.float32],
        rows_diag: wp.array2d[wp.float32],
        rows_type: wp.array2d[wp.int32],
        rows_parent: wp.array2d[wp.int32],
        rows_mu: wp.array2d[wp.float32],
        omega: float,
        delta: wp.array2d[wp.float32],
        impulses: wp.array2d[wp.float32],
    ): ...

    def sweep_block(
        side_count: int,
        big_colors: wp.array[wp.int32],
        used_colors: wp.array[wp.int32],
        row_ranges: wp.array2d[wp.int32],
        color_offsets: wp.array[wp.int32],
        one_sided_start: wp.array[wp.int32],
        seq_row_start: wp.array[wp.int32],
        constraint_count: wp.array[wp.int32],
        rows_dof: wp.array3d[wp.int32],
        rows_factor: wp.array3d[wp.float32],
        rows_free_response: wp.array3d[wp.float32],
        rows_free_prefix: wp.array[wp.int32],
        rows_rhs: wp.array2d[wp.float32],
        rows_incident: wp.array2d[wp.float32],
        rows_diag: wp.array2d[wp.float32],
        rows_type: wp.array2d[wp.int32],
        rows_parent: wp.array2d[wp.int32],
        rows_mu: wp.array2d[wp.float32],
        omega: float,
        delta: wp.array2d[wp.float32],
        impulses: wp.array2d[wp.float32],
    ):
        tid = wp.tid()
        sweep_block_native(
            tid,
            side_count,
            big_colors,
            used_colors,
            row_ranges,
            color_offsets,
            one_sided_start,
            seq_row_start,
            constraint_count,
            rows_dof,
            rows_factor,
            rows_free_response,
            rows_free_prefix,
            rows_rhs,
            rows_incident,
            rows_diag,
            rows_type,
            rows_parent,
            rows_mu,
            omega,
            delta,
            impulses,
        )

    sweep_block.__name__ = f"steel_scene_block_{granularity}_{sums}_{int(support)}_{int(free_response)}"
    sweep_block.__qualname__ = sweep_block.__name__
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(sweep_block)


def _featured_block_kernel(snippet: str, name: str) -> wp.Kernel:
    """The featured definition of :func:`get_sweep_block_kernel`: its arguments, then every feature's."""

    @wp.func_native(snippet)
    def sweep_block_native(
        tid: int,
        side_count: int,
        big_colors: wp.array[wp.int32],
        used_colors: wp.array[wp.int32],
        row_ranges: wp.array2d[wp.int32],
        color_offsets: wp.array[wp.int32],
        one_sided_start: wp.array[wp.int32],
        seq_row_start: wp.array[wp.int32],
        constraint_count: wp.array[wp.int32],
        rows_dof: wp.array3d[wp.int32],
        rows_factor: wp.array3d[wp.float32],
        rows_free_response: wp.array3d[wp.float32],
        rows_free_prefix: wp.array[wp.int32],
        rows_rhs: wp.array2d[wp.float32],
        rows_incident: wp.array2d[wp.float32],
        rows_diag: wp.array2d[wp.float32],
        rows_type: wp.array2d[wp.int32],
        rows_parent: wp.array2d[wp.int32],
        rows_mu: wp.array2d[wp.float32],
        omega: float,
        delta: wp.array2d[wp.float32],
        impulses: wp.array2d[wp.float32],
        friction_start: int,
        row_w: wp.array2d[wp.float32],
        row_bound: wp.array2d[wp.float32],
        drive_target_vel_bias: wp.array2d[wp.float32],
        drive_vel_multiplier: wp.array2d[wp.float32],
        drive_impulse_multiplier: wp.array2d[wp.float32],
        drive_max_impulse: wp.array2d[wp.float32],
        side_late_begin: wp.array[wp.int32],
        side_late_end: wp.array[wp.int32],
        drive_vel_limit: wp.array2d[wp.float32],
        freeze_drives: int,
        row_gamma: wp.array2d[wp.float32],
        torsion_radius: wp.array2d[wp.float32],
        torsion_group: wp.array2d[wp.int32],
        torsion_member_range: wp.array2d[wp.int32],
        torsion_member_rows: wp.array[wp.int32],
        side_torsion_begin: wp.array[wp.int32],
        side_torsion_end: wp.array[wp.int32],
        torsion_color_start: wp.array[wp.int32],
        row_friction_mode: wp.array2d[wp.int32],
    ): ...

    def sweep_block(
        side_count: int,
        big_colors: wp.array[wp.int32],
        used_colors: wp.array[wp.int32],
        row_ranges: wp.array2d[wp.int32],
        color_offsets: wp.array[wp.int32],
        one_sided_start: wp.array[wp.int32],
        seq_row_start: wp.array[wp.int32],
        constraint_count: wp.array[wp.int32],
        rows_dof: wp.array3d[wp.int32],
        rows_factor: wp.array3d[wp.float32],
        rows_free_response: wp.array3d[wp.float32],
        rows_free_prefix: wp.array[wp.int32],
        rows_rhs: wp.array2d[wp.float32],
        rows_incident: wp.array2d[wp.float32],
        rows_diag: wp.array2d[wp.float32],
        rows_type: wp.array2d[wp.int32],
        rows_parent: wp.array2d[wp.int32],
        rows_mu: wp.array2d[wp.float32],
        omega: float,
        delta: wp.array2d[wp.float32],
        impulses: wp.array2d[wp.float32],
        friction_start: int,
        row_w: wp.array2d[wp.float32],
        row_bound: wp.array2d[wp.float32],
        drive_target_vel_bias: wp.array2d[wp.float32],
        drive_vel_multiplier: wp.array2d[wp.float32],
        drive_impulse_multiplier: wp.array2d[wp.float32],
        drive_max_impulse: wp.array2d[wp.float32],
        side_late_begin: wp.array[wp.int32],
        side_late_end: wp.array[wp.int32],
        drive_vel_limit: wp.array2d[wp.float32],
        freeze_drives: int,
        row_gamma: wp.array2d[wp.float32],
        torsion_radius: wp.array2d[wp.float32],
        torsion_group: wp.array2d[wp.int32],
        torsion_member_range: wp.array2d[wp.int32],
        torsion_member_rows: wp.array[wp.int32],
        side_torsion_begin: wp.array[wp.int32],
        side_torsion_end: wp.array[wp.int32],
        torsion_color_start: wp.array[wp.int32],
        row_friction_mode: wp.array2d[wp.int32],
    ):
        tid = wp.tid()
        sweep_block_native(
            tid,
            side_count,
            big_colors,
            used_colors,
            row_ranges,
            color_offsets,
            one_sided_start,
            seq_row_start,
            constraint_count,
            rows_dof,
            rows_factor,
            rows_free_response,
            rows_free_prefix,
            rows_rhs,
            rows_incident,
            rows_diag,
            rows_type,
            rows_parent,
            rows_mu,
            omega,
            delta,
            impulses,
            friction_start,
            row_w,
            row_bound,
            drive_target_vel_bias,
            drive_vel_multiplier,
            drive_impulse_multiplier,
            drive_max_impulse,
            side_late_begin,
            side_late_end,
            drive_vel_limit,
            freeze_drives,
            row_gamma,
            torsion_radius,
            torsion_group,
            torsion_member_range,
            torsion_member_rows,
            side_torsion_begin,
            side_torsion_end,
            torsion_color_start,
            row_friction_mode,
        )

    sweep_block.__name__ = name
    sweep_block.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(sweep_block)


def _sweep_torsion_snippet(
    granularity: str, sums: str, support: int, free_response: bool, features: SceneFeatures
) -> str:
    """The native snippet of :func:`get_sweep_torsion_kernel`; it raises the factory's :class:`ValueError`."""
    group = _check_sweep(granularity, sums, support, key="scene_unit_sweep")
    if not features.torsion:
        raise ValueError("the two-sided spin rows' sweep needs contact torsion (SceneFeatures.torsion)")
    head = (
        """
    const int item = tid;
"""
        if group == 1
        else _GROUP_INDEX_CUDA
    )
    sweep = "sweep_rows(row, row + 1)" if group == 1 else "sweep_rows_group(row, row + 1, lane, mask)"
    body = (
        head
        + f"""
    // The two-sided spin rows (C1), color by color: those of a color at once (their groups' representative contact
    // units share no side, and a spin row's sides are its representative's), then the tail's on one thread.
    constexpr int WORKERS = {SCENE_BLOCK_THREADS // group};
    const int colors = torsion_color_start.shape[0] - 2;
    for (int color = 0; color < colors; ++color) {{
        const int first = torsion_color_start.data[color];
        const int last = torsion_color_start.data[color + 1];
        for (int row = first + item; row < last; row += WORKERS) {sweep};
        // The color's writes are visible to the whole block before the next color reads them.
        __syncthreads();
    }}
    // The tail's: units past the color cap may share sides, so one thread walks their spin rows in order.
    if (tid == 0) sweep_rows(torsion_color_start.data[colors], torsion_color_start.data[colors + 1]);
"""
    )
    return _sweep_snippet(
        granularity, sums, support, free_response, body, thread_loop=True, block=True, features=features
    )


@cache
def get_sweep_torsion_kernel(
    granularity: Literal["thread", "lanes8", "lanes16", "lanes32"],
    sums: Literal["serial", "steel"],
    support: int,
    *,
    free_response: bool = False,
    features: SceneFeatures,
) -> wp.Kernel:
    """Contact torsion's two-sided spin rows in phase C, in one block: C1 color by color, then the tail's.

    A two-sided spin row (its group's members join two articulations) is swept with the color of its group's
    representative contact unit: the spin rows of color ``c`` are ``[torsion_color_start[c], torsion_color_start[c +
    1])``, the tail's (the color cap's) the last range (``SceneSolve.prepare_torsion``). Units of a color share no
    side, a spin row's sides are its representative unit's, and a spin row writes only its own members' tangent pairs
    and the velocities of its two sides, so the rows of a color run at once: the block's :data:`SCENE_BLOCK_THREADS`
    threads (work items of one thread or 8 lanes, the phase-B granularity ``scene_unit_sweep``) stride over them,
    ``__syncthreads()`` orders the colors, and thread 0 then walks the tail's in row order. The serial walk of the
    two-sided spin rows in row order is its oracle (the order oracle's). The kernel exists only featured, with contact
    torsion on (``features``): it takes the rows and every feature's arguments, as the other featured sweep kernels.

    Launch with ``dim = block_dim = SCENE_BLOCK_THREADS`` after the coupled sides' one-sided spin rows and before their
    late rows (``SceneSolve._sweep_late``). CUDA only. Raises :class:`ValueError` without contact torsion and for
    Steel-order sums over more than 32 entries.
    """
    suffix = _feature_suffix(features)
    snippet = _sweep_torsion_snippet(granularity, sums, support, free_response, features)
    name = f"steel_scene_torsion_{granularity}_{sums}_{int(support)}_{int(free_response)}{suffix}"

    @wp.func_native(snippet)
    def sweep_torsion_native(
        tid: int,
        rows_dof: wp.array3d[wp.int32],
        rows_factor: wp.array3d[wp.float32],
        rows_free_response: wp.array3d[wp.float32],
        rows_free_prefix: wp.array[wp.int32],
        rows_rhs: wp.array2d[wp.float32],
        rows_incident: wp.array2d[wp.float32],
        rows_diag: wp.array2d[wp.float32],
        rows_type: wp.array2d[wp.int32],
        rows_parent: wp.array2d[wp.int32],
        rows_mu: wp.array2d[wp.float32],
        omega: float,
        delta: wp.array2d[wp.float32],
        impulses: wp.array2d[wp.float32],
        friction_start: int,
        row_w: wp.array2d[wp.float32],
        row_bound: wp.array2d[wp.float32],
        drive_target_vel_bias: wp.array2d[wp.float32],
        drive_vel_multiplier: wp.array2d[wp.float32],
        drive_impulse_multiplier: wp.array2d[wp.float32],
        drive_max_impulse: wp.array2d[wp.float32],
        side_late_begin: wp.array[wp.int32],
        side_late_end: wp.array[wp.int32],
        drive_vel_limit: wp.array2d[wp.float32],
        freeze_drives: int,
        row_gamma: wp.array2d[wp.float32],
        torsion_radius: wp.array2d[wp.float32],
        torsion_group: wp.array2d[wp.int32],
        torsion_member_range: wp.array2d[wp.int32],
        torsion_member_rows: wp.array[wp.int32],
        side_torsion_begin: wp.array[wp.int32],
        side_torsion_end: wp.array[wp.int32],
        torsion_color_start: wp.array[wp.int32],
        row_friction_mode: wp.array2d[wp.int32],
    ): ...

    def sweep_torsion(
        rows_dof: wp.array3d[wp.int32],
        rows_factor: wp.array3d[wp.float32],
        rows_free_response: wp.array3d[wp.float32],
        rows_free_prefix: wp.array[wp.int32],
        rows_rhs: wp.array2d[wp.float32],
        rows_incident: wp.array2d[wp.float32],
        rows_diag: wp.array2d[wp.float32],
        rows_type: wp.array2d[wp.int32],
        rows_parent: wp.array2d[wp.int32],
        rows_mu: wp.array2d[wp.float32],
        omega: float,
        delta: wp.array2d[wp.float32],
        impulses: wp.array2d[wp.float32],
        friction_start: int,
        row_w: wp.array2d[wp.float32],
        row_bound: wp.array2d[wp.float32],
        drive_target_vel_bias: wp.array2d[wp.float32],
        drive_vel_multiplier: wp.array2d[wp.float32],
        drive_impulse_multiplier: wp.array2d[wp.float32],
        drive_max_impulse: wp.array2d[wp.float32],
        side_late_begin: wp.array[wp.int32],
        side_late_end: wp.array[wp.int32],
        drive_vel_limit: wp.array2d[wp.float32],
        freeze_drives: int,
        row_gamma: wp.array2d[wp.float32],
        torsion_radius: wp.array2d[wp.float32],
        torsion_group: wp.array2d[wp.int32],
        torsion_member_range: wp.array2d[wp.int32],
        torsion_member_rows: wp.array[wp.int32],
        side_torsion_begin: wp.array[wp.int32],
        side_torsion_end: wp.array[wp.int32],
        torsion_color_start: wp.array[wp.int32],
        row_friction_mode: wp.array2d[wp.int32],
    ):
        tid = wp.tid()
        sweep_torsion_native(
            tid,
            rows_dof,
            rows_factor,
            rows_free_response,
            rows_free_prefix,
            rows_rhs,
            rows_incident,
            rows_diag,
            rows_type,
            rows_parent,
            rows_mu,
            omega,
            delta,
            impulses,
            friction_start,
            row_w,
            row_bound,
            drive_target_vel_bias,
            drive_vel_multiplier,
            drive_impulse_multiplier,
            drive_max_impulse,
            side_late_begin,
            side_late_end,
            drive_vel_limit,
            freeze_drives,
            row_gamma,
            torsion_radius,
            torsion_group,
            torsion_member_range,
            torsion_member_rows,
            side_torsion_begin,
            side_torsion_end,
            torsion_color_start,
            row_friction_mode,
        )

    sweep_torsion.__name__ = name
    sweep_torsion.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(sweep_torsion)


def _default_sum_order(support: int) -> Literal["serial", "steel"]:
    """The scene sweep's sums without a ``scene_sum_order`` override: Steel's for rows of at most 32 entries."""
    return "steel" if support <= SUM_TREE_SUPPORT_MAX else "serial"


def scene_kernel_snippets(
    support: int,
    *,
    free_response: bool,
    dof_max: int,
    features: SceneFeatures = SceneFeatures(),  # noqa: B008 (frozen)
) -> dict[str, str]:
    """The native snippets of the scene's row-build and sweep kernels in the default configuration, by kernel.

    The default configuration is ``scene_solve.SceneSolve``'s without overrides: the group-table contact
    response kernel for groups of at most ``dof_max`` DOFs with one-word support masks and 8 lanes
    per contact, ``"lanes32"`` sides, ``"lanes8"`` units, :func:`_default_sum_order` and the ``"BA"`` order
    oracle. Keys: ``response`` (the group-table contact response, ``scene_rows._response_snippet`` with
    ``specialized=False``), ``sides`` (:func:`get_sweep_sides_kernel`), ``serial``
    (:func:`get_sweep_rows_serial_kernel`), ``order_serial`` (:func:`get_sweep_order_serial_kernel`), ``color``
    (:func:`get_sweep_color_kernel`) and ``block`` (:func:`get_sweep_block_kernel`), and with contact torsion
    ``torsion`` (:func:`get_sweep_torsion_kernel`). The scene tripwire pins their hashes with every feature off, so
    that a change to a default kernel's source is a deliberate one. ``features`` are the sweep kernels' build
    constants; every feature off (``SceneFeatures()``, the default) gives the pinned sources.
    """
    sums = _default_sum_order(support)
    return {
        "response": _response_snippet(dof_max, specialized=False),
        "sides": _sweep_sides_snippet("lanes32", sums, support, free_response, features),
        "serial": _sweep_rows_serial_snippet(sums, support, free_response, features),
        "order_serial": _sweep_order_serial_snippet(SCENE_PHASE_ORDERS[0], sums, support, free_response, features),
        "color": _sweep_color_snippet("lanes8", sums, support, free_response, features),
        "block": _sweep_block_snippet("lanes8", sums, support, free_response, features),
        **(
            {"torsion": _sweep_torsion_snippet("lanes8", sums, support, free_response, features)}
            if features.torsion
            else {}
        ),
    }


@wp.kernel
def reset_color_loop(
    loop_colors: wp.array[int],
    # outputs
    color_cursor: wp.array[int],
    color_keep_going: wp.array[int],
):
    """Start the color loop at color 0, and keep going when it has a color to run (grid: 1).

    ``loop_colors[0]`` is the number of leading colors the while loop runs (``big_colors``).
    """
    color_cursor[0] = 0
    keep = int(0)
    if loop_colors[0] > 0:
        keep = 1
    color_keep_going[0] = keep


@wp.kernel
def advance_color(
    loop_colors: wp.array[int],
    # in/out
    color_cursor: wp.array[int],
    # outputs
    color_keep_going: wp.array[int],
):
    """Move the color loop to the next color, and stop it after the last one it runs (grid: 1)."""
    color = color_cursor[0] + 1
    color_cursor[0] = color
    keep = int(0)
    if color < loop_colors[0]:
        keep = 1
    color_keep_going[0] = keep


@wp.kernel
def scene_row_ranges(
    side_count: int,
    color_cap: int,
    block_color_max: int,
    constraint_count: wp.array[int],
    one_sided_start: wp.array[int],
    seq_row_start: wp.array[int],
    color_count_per: wp.array[int],
    color_offsets: wp.array[int],
    # outputs
    row_ranges: wp.array2d[int],
    used_colors: wp.array[int],
    big_colors: wp.array[int],
):
    """Write the step's row ranges and the colors phase B runs (grid: 1).

    ``row_ranges[RANGE_ALL]`` is every kept row ``[0, rows)``; ``row_ranges[RANGE_TWO_SIDED]``
    starts at the first two-sided unit's row, after every one-sided row; ``row_ranges[RANGE_TAIL]``
    at the first tail unit's row, after the colors; ``row_ranges[RANGE_LATE]`` at the first late unit's
    row, after the tail. The two-sided and tail ranges end at the first late row and the late range at
    ``rows``, each clipped to ``rows``, so a phase-B sweep never reads a late row and dropped units' rows
    are never swept. Without late units the first late row is ``rows``. ``used_colors[0]`` is the highest color below
    ``color_cap`` with a unit, plus one (``0`` without colored units). ``big_colors[0]`` counts the
    leading colors with more than ``block_color_max`` units, the ones the while loop runs. Colors of
    first-fit are only roughly non-increasing in size, so a big color can follow a small one; it is
    then left to the single block, which sweeps it correctly, only more slowly.
    """
    rows = constraint_count[0]
    first_two_sided = one_sided_start[side_count]
    two_sided_begin = wp.min(seq_row_start[first_two_sided], rows)
    tail_begin = wp.min(seq_row_start[first_two_sided + color_offsets[color_cap]], rows)
    # The late units follow every two-sided one (color_offsets[color_cap + 1] of them).
    late_begin = wp.min(seq_row_start[first_two_sided + color_offsets[color_cap + 1]], rows)
    row_ranges[RANGE_ALL, 0] = 0
    row_ranges[RANGE_ALL, 1] = rows
    row_ranges[RANGE_TWO_SIDED, 0] = two_sided_begin
    row_ranges[RANGE_TWO_SIDED, 1] = late_begin
    row_ranges[RANGE_TAIL, 0] = tail_begin
    row_ranges[RANGE_TAIL, 1] = late_begin
    row_ranges[RANGE_LATE, 0] = late_begin
    row_ranges[RANGE_LATE, 1] = rows
    used = int(0)
    for color in range(color_cap):
        if color_count_per[color] > 0:
            used = color + 1
    big = int(0)
    for color in range(used):
        if big == color and color_count_per[color] > block_color_max:
            big = color + 1
    used_colors[0] = used
    big_colors[0] = big


@dataclass(frozen=True)
class _ScenePass:
    """One pass of ``SceneSolve.solve`` over the step's rows: the position solve or the velocity-only iterations.

    ``rhs`` is the rows' right-hand side, ``iterations`` the sweeps, ``friction_start`` the first sweep that solves
    the tangent pairs (before the launch's iteration is taken off), ``row_w`` the regularized sweeps' weights and
    ``freeze_drives`` whether the drive rows are left as they are.
    """

    rhs: wp.array
    iterations: int
    friction_start: int
    row_w: wp.array
    freeze_drives: int


# --- Guards of the restated laws ----------------------------------------------------------------
#
# The scene restates some of Steel's laws instead of pasting them: its friction modes and contact torsion's budget
# round every operation on its own, and the friction patches' pooled load walks the scene's own arrays. Each restated
# law mirrors Steel's current text statement for statement. _mirrored_lines reduces the scene's text and Steel's to
# comparable lines, which the guard test (test_scene_pgs) asserts equal, so a change to one of Steel's laws fails that
# test instead of leaving the scene's copy behind: each text is cut to the law's region, its statements are parsed
# (comments, preprocessor lines, layout, types and ``const`` dropped, one line per declarator, compound assignments as
# plain ones), the rounding intrinsics become the operators they round, every expression is printed fully
# parenthesized, and each side's names are rewritten onto common ones.

_C_TOKEN = re.compile(
    r"\s*(?:(?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fFuUlL]*)|(?P<name>[$A-Za-z_]\w*)"
    r"|(?P<op>\+\+|--|->|<<|>>|<=|>=|==|!=|&&|\|\||[-+*/%&|^]=|[-+*/%<>=!&|^~?:.,;(){}\[\]]|\S))"
)
"""A token of C: a number, a name (``$`` starts a pattern's wildcard), an operator or any other character."""

_C_BINARY = {
    "||": 1,
    "&&": 2,
    "|": 3,
    "^": 4,
    "&": 5,
    "==": 6,
    "!=": 6,
    "<": 7,
    ">": 7,
    "<=": 7,
    ">=": 7,
    "<<": 8,
    ">>": 8,
    "+": 9,
    "-": 9,
    "*": 10,
    "/": 10,
    "%": 10,
}
"""C's binary operators by precedence, the loosest first."""

_C_ROUNDED = {"__fadd_rn": "+", "__fsub_rn": "-", "__fmul_rn": "*", "__fdiv_rn": "/"}
"""The round-to-nearest intrinsics of the restated laws and the operator each rounds."""

_C_TYPES = frozenset({"const", "auto", "bool", "int", "unsigned", "float", "float2", "int2"})
"""The words that start a declaration in the mirrored texts."""


def _c_tokens(text: str) -> list[str]:
    """The tokens of C ``text`` without its comments and preprocessor lines."""
    text = re.sub(r"(?m)^\s*#.*$", "", re.sub(r"//[^\n]*", "", text))
    tokens, at = [], 0
    while text[at:].strip():
        match = _C_TOKEN.match(text, at)
        tokens.append(match.group(match.lastgroup))
        at = match.end()
    return tokens


def _c_number(text: str) -> str:
    """A literal's value: ``1.0e-20f`` and ``1e-20`` print alike."""
    digits = text.rstrip("fFuUlL")
    return repr(float(digits)) if any(mark in digits for mark in ".eE") else str(int(digits))


class _CStatements:
    """Statements of C tokens as comparable lines; ``rules`` rewrite each expression (:func:`_c_rewrite`).

    An expression is a tuple tree: ``("n", name)``, ``("k", value)``, ``("u", op, operand)``, ``("b", op, left,
    right)``, ``("?", condition, then, otherwise)``, ``("c", function, ("a", *arguments))``, ``("i", base, index)`` and
    ``(".", base, field)``. A statement is a tuple of its own: ``("=", target, value)``, ``("if", condition)``,
    ``("else",)``, ``("for", header)``, ``("{",)``, ``("}",)``, ``("break",)``, ``("continue",)``, ``("return",
    value)``, ``("decl", name)``, ``("lambda", name)`` or ``("do", expression)``. A block's statements sit between its
    braces, a body without braces gets them, and a block the tokens leave open is closed at their end.
    """

    def __init__(self, tokens: list[str], rules: list[tuple[tuple, tuple]]):
        self.tokens, self.at, self.rules = tokens, 0, rules

    def peek(self) -> str | None:
        return self.tokens[self.at] if self.at < len(self.tokens) else None

    def take(self, expected: str | None = None) -> str:
        token = self.peek()
        if token is None or (expected is not None and token != expected):
            raise ValueError(f"expected {expected!r} at token {self.at}, got {token!r}")
        self.at += 1
        return token

    def expression(self) -> tuple:
        """A whole expression, rewritten by the rules."""
        return _c_rewrite(self.ternary(), self.rules)

    def ternary(self) -> tuple:
        condition = self.binary(1)
        if self.peek() != "?":
            return condition
        self.take("?")
        then = self.ternary()
        self.take(":")
        return ("?", condition, then, self.ternary())

    def binary(self, floor: int) -> tuple:
        left = self.unary()
        while self.peek() in _C_BINARY and _C_BINARY[self.peek()] >= floor:
            op = self.take()
            left = ("b", op, left, self.binary(_C_BINARY[op] + 1))
        return left

    def unary(self) -> tuple:
        if self.peek() in ("-", "+", "!", "~"):
            op = self.take()
            return ("u", op, self.unary())
        return self.postfix(self.primary())

    def primary(self) -> tuple:
        token = self.take()
        if token == "(":
            inner = self.ternary()
            self.take(")")
            return inner
        if token[0].isdigit() or token[0] == ".":
            return ("k", _c_number(token))
        return ("n", token)

    def postfix(self, node: tuple) -> tuple:
        while self.peek() in ("(", "[", ".", "->"):
            token = self.take()
            if token == "(":
                arguments = []
                while self.peek() != ")":
                    arguments.append(self.ternary())
                    if self.peek() == ",":
                        self.take(",")
                self.take(")")
                if node[0] == "n" and node[1] in _C_ROUNDED and len(arguments) == 2:
                    node = ("b", _C_ROUNDED[node[1]], *arguments)
                elif node == ("n", "__fsqrt_rn"):
                    node = ("c", ("n", "sqrtf"), ("a", *arguments))
                else:
                    node = ("c", node, ("a", *arguments))
            elif token == "[":
                node = ("i", node, self.ternary())
                self.take("]")
            else:
                node = (".", node, self.take())
        return node

    def statements(self) -> list[tuple]:
        """The statements up to the end of the tokens or a brace that closes a block they did not open."""
        found = []
        while self.peek() not in (None, "}"):
            found += self.statement()
        return found

    def body(self) -> list[tuple]:
        """The body of an ``if``, an ``else`` or a ``for``, in braces; nothing when the tokens end first."""
        if self.peek() is None:
            return []
        inner = self.statement()
        return inner if inner[0] == ("{",) else [("{",), *inner, ("}",)]

    def statement(self) -> list[tuple]:
        token = self.peek()
        if token == "{":
            self.take("{")
            inner = self.statements()
            if self.peek() == "}":
                self.take("}")
            return [("{",), *inner, ("}",)]
        if token == "if":
            self.take("if")
            self.take("(")
            condition = self.expression()
            self.take(")")
            found = [("if", condition), *self.body()]
            if self.peek() == "else":
                self.take("else")
                found += [("else",), *self.body()]
            return found
        if token == "for":
            self.take("for")
            self.take("(")
            start = self.simple()
            self.take(";")
            condition = self.expression()
            self.take(";")
            step = self.simple()
            self.take(")")
            header = "; ".join((*map(_c_line, start), _c_show(condition), *map(_c_line, step)))
            return [("for", header), *self.body()]
        if token in ("break", "continue"):
            self.take()
            self.take(";")
            return [(token,)]
        if token == "return":
            self.take("return")
            value = self.expression()
            self.take(";")
            return [("return", value)]
        found = self.simple()
        self.take(";")
        return found

    def simple(self) -> list[tuple]:
        """A declaration (a statement per declarator) or an expression statement, without its semicolon."""
        if self.peek() in _C_TYPES:
            while self.peek() in _C_TYPES:
                self.take()
            found = []
            while True:
                while self.peek() in ("*", "&"):
                    self.take()
                name = self.take()
                if self.peek() == "[":
                    self.take("[")
                    name = f"{name}[{_c_show(self.expression())}]"
                    self.take("]")
                if self.peek() != "=":
                    found.append(("decl", name))
                elif self.tokens[self.at + 1] == "[":
                    self.take("=")
                    return [*found, *self.lambda_(name)]
                elif self.tokens[self.at + 1] == "{":
                    self.take("=")
                    self.take("{")
                    values = []
                    while self.peek() != "}":
                        values.append(self.expression())
                        if self.peek() == ",":
                            self.take(",")
                    self.take("}")
                    found.append(("=", ("n", name), ("a", *values)))
                else:
                    self.take("=")
                    found.append(("=", _c_rewrite(("n", name), self.rules), self.expression()))
                if self.peek() != ",":
                    return found
                self.take(",")
        if self.peek() in ("++", "--"):
            op = self.take()
            target = self.expression()
            return [("=", target, ("b", op[0], target, ("k", "1")))]
        target = self.expression()
        op = self.peek()
        if op == "=":
            self.take("=")
            return [("=", target, self.expression())]
        if op in ("+=", "-=", "*=", "/="):
            self.take()
            return [("=", target, ("b", op[0], target, self.expression()))]
        if op in ("++", "--"):
            self.take()
            return [("=", target, ("b", op[0], target, ("k", "1")))]
        return [("do", target)]

    def lambda_(self, name: str) -> list[tuple]:
        """A lambda's statements: its captures, parameters and return type skipped, its body in braces."""
        for opening, closing in (("[", "]"), ("(", ")")):
            depth = 0
            while True:
                token = self.take()
                depth += token == opening
                depth -= token == closing
                if depth == 0:
                    break
        if self.peek() == "->":
            self.take("->")
            self.take()
        return [("lambda", name), *self.statement()]


def _c_show(node) -> str:
    """An expression tree printed fully parenthesized."""
    kind = node[0]
    if kind in ("n", "k"):
        return node[1]
    if kind == "u":
        return f"{node[1]}{_c_show(node[2])}"
    if kind == "b":
        return f"({_c_show(node[2])} {node[1]} {_c_show(node[3])})"
    if kind == "?":
        return f"({_c_show(node[1])} ? {_c_show(node[2])} : {_c_show(node[3])})"
    if kind == "c":
        return f"{_c_show(node[1])}{_c_show(node[2])}"
    if kind == "a":
        return f"({', '.join(_c_show(argument) for argument in node[1:])})"
    if kind == "i":
        return f"{_c_show(node[1])}[{_c_show(node[2])}]"
    return f"{_c_show(node[1])}.{node[2]}"


def _c_line(statement: tuple) -> str:
    """A statement printed as one line."""
    kind = statement[0]
    if kind == "=":
        return f"{_c_show(statement[1])} = {_c_show(statement[2])}"
    if kind in ("if", "return", "do"):
        return f"{kind} {_c_show(statement[1])}"
    if kind in ("for", "decl", "lambda"):
        return f"{kind} {statement[1]}"
    return kind


def _c_pattern(text: str) -> tuple:
    """The expression tree of ``text``, a pattern whose ``$`` names match any subtree."""
    return _CStatements(_c_tokens(text), []).ternary()


def _c_match(pattern, node, bound: dict) -> bool:
    if isinstance(pattern, tuple) and pattern[0] == "n" and pattern[1].startswith("$"):
        if pattern[1] in bound:
            return bound[pattern[1]] == node
        bound[pattern[1]] = node
        return True
    if not isinstance(pattern, tuple) or not isinstance(node, tuple) or len(pattern) != len(node):
        return pattern == node
    return all(_c_match(part, other, bound) for part, other in zip(pattern, node, strict=True))


def _c_substitute(node, bound: dict):
    if isinstance(node, tuple) and node[0] == "n" and node[1] in bound:
        return bound[node[1]]
    return tuple(_c_substitute(part, bound) for part in node) if isinstance(node, tuple) else node


def _c_rewrite(node, rules: list[tuple[tuple, tuple]]):
    """``node`` with every subtree a rule's pattern matches replaced by its replacement, the innermost first."""
    if not isinstance(node, tuple):
        return node
    node = tuple(_c_rewrite(part, rules) for part in node)
    for pattern, replacement in rules:
        bound = {}
        if _c_match(pattern, node, bound):
            return _c_substitute(replacement, bound)
    return node


def _c_names(node) -> set[str]:
    """The variables an expression tree reads (not the functions it calls)."""
    if not isinstance(node, tuple):
        return set()
    if node[0] == "n":
        return {node[1]}
    if node[0] == "c":
        return _c_names(node[2])
    return set().union(*(_c_names(part) for part in node[1:]))


@dataclass(frozen=True)
class _Mirror:
    """One restated law: the scene's text and Steel's, each one's region that holds the law, and how they compare.

    A region starts at the first token of its ``begin`` (the text's start without one) and ends before the first
    ``end`` after it (the text's end without one). ``scene_names`` and ``steel_names`` map patterns of each side's
    expressions to the common names (:func:`_c_rewrite`). With ``keep`` only the assignments to these variables and
    the conditions that read nothing else are compared, the statements of the law the two texts share where they
    differ in how they walk their rows; ``drop`` names variables Steel writes and never reads.
    """

    scene: Callable[[], str]
    steel: Callable[[], str]
    scene_begin: str | None = None
    scene_end: str | None = None
    steel_begin: str | None = None
    steel_end: str | None = None
    scene_names: dict[str, str] = field(default_factory=dict)
    steel_names: dict[str, str] = field(default_factory=dict)
    keep: frozenset[str] | None = None
    drop: frozenset[str] = frozenset()


def _steel_text(name: str) -> Callable[[], str]:
    """Steel's module-level source text ``name`` of ``solver_steel``, read when the guard runs."""

    def text() -> str:
        from . import solver_steel  # noqa: PLC0415 (the solver module builds on the scene modules)

        return getattr(solver_steel, name)

    return text


def _steel_sparse_sweep() -> str:
    """The native snippet of Steel's sparse sweep (``sparse_pgs._get_pgs_solve_sparse_kernel``), whose articulated
    rows' pooled patch load the scene's friction patches restate."""
    kernel = _get_pgs_solve_sparse_kernel(_MIRROR_ROWS, _MIRROR_DOFS, _MIRROR_DOFS)
    cells = (cell.cell_contents for cell in kernel.func.__closure__ or ())
    (snippet,) = (cell.native_snippet for cell in cells if isinstance(cell, wp.Function) and cell.native_snippet)
    return snippet


_MIRROR_ROWS = 64
_MIRROR_DOFS = 27
"""Row capacity and DOFs of the Steel kernels whose text the guards read: their laws do not depend on them."""

_SCENE_FRICTION_LAW = {"scene_begin": "if (G_nn >= 1.0e-20f)", "scene_end": "lambda[0] = new_lambda_n;"}
"""The region of the scene's friction laws: their ``if (G_nn >= 1.0e-20f)``, before the new impulses are returned."""

_SOURCE_FRICTION_LAW = {"steel_begin": "if (G_nn >= 1.0e-20f)", "steel_end": "d_n_total = new_lambda_n - old_lambda_n;"}
"""The region of Steel's friction laws: their ``if (G_nn >= 1.0e-20f)``, before the changes are applied."""

_TORSION_LAW = frozenset(
    {
        "radius",
        "mu",
        "spin_group",
        "normal_load",
        "sliding_used",
        "a",
        "b",
        "normal_budget",
        "scale",
        "t1",
        "t2",
        "delta1",
        "delta2",
        "velocity",
        "bound",
        "next",
    }
)
"""The variables of contact torsion's budget, the rescale, the bound and the clamped update: the law the scene's C1
shares with Steel's torsion sweep, which walk their members and apply their changes in their own ways."""

_SCENE_TORSION_NAMES = {
    "impulses.data[$row]": "impulse[$row]",
    "torsion_radius.data[$row]": "spin_radius",
    "rows_mu.data[$row]": "spin_mu",
    "torsion_group.data[spin]": "stored_spin_group",
    "delta.data[index]": "velocity",
    "TORSION_RETIRED": str(int(_TORSION_SPIN_RETIRED)),
    "torsion_bound": "bound",
    "new_impulse": "next",
}
_SOURCE_TORSION_NAMES = {
    "s_lam_dense[$row]": "impulse[$row]",
    "torsion_row_radius.data[$row]": "spin_radius",
    "world_row_mu.data[$row]": "spin_mu",
    "world_torsion_group.data[off_dense + spin]": "stored_spin_group",
    "s_v[d]": "velocity",
    "Y_world.data[$base + (n + 1) * $dofs + $entry]": "y1",
    "Y_world.data[$base + (n + 2) * $dofs + $entry]": "y2",
}
"""Each side's names of contact torsion's arrays, mapped onto common ones: the impulses, the spin row's radius,
coefficient and group, a velocity entry the rescale changes, and the tangent rows' responses there."""

_MIRRORS = {
    "coulomb_newton": _Mirror(
        lambda: _FRICTION_MODE_COULOMB_NEWTON_CUDA,
        _steel_text("_MF_COULOMB_NEWTON_FRICTION_SOURCE"),
        **_SCENE_FRICTION_LAW,
        **_SOURCE_FRICTION_LAW,
        drop=frozenset({"alpha"}),
    ),
    "bisection_free": _Mirror(
        lambda: _FRICTION_MODE_BISECTION_CUDA,
        _steel_text("_MF_BISECTION_FRICTION_SOURCE"),
        **_SCENE_FRICTION_LAW,
        **_SOURCE_FRICTION_LAW,
    ),
    "bisection_dense": _Mirror(
        lambda: _FRICTION_MODE_BISECTION_CUDA,
        _steel_text("_DENSE_BISECTION_FRICTION_SOURCE"),
        **_SCENE_FRICTION_LAW,
        **_SOURCE_FRICTION_LAW,
    ),
    "desaxce": _Mirror(
        lambda: _FRICTION_MODE_DESAXCE_CUDA,
        _steel_text("_MF_DESAXCE_BIAS_SOURCE"),
        scene_begin="const float c_T_mag",
        steel_begin="float c_T_mag",
        scene_names={"u[1]": "u_t1", "u[2]": "u_t2", "target": "target_vel_n"},
    ),
    "patch_load": _Mirror(
        lambda: _PATCH_LOAD_CUDA,
        _steel_sparse_sweep,
        steel_begin="float normal_load = impulse[parent];",
        steel_end="const float radius",
        scene_names={
            "impulses.data[$row]": "impulse[$row]",
            "rows_parent.data[$row]": "world_row_parent.data[offset + $row]",
        },
    ),
    **{
        f"torsion_budget{'_group' if group else ''}{'_linked' if linked else ''}": _Mirror(
            partial(lambda group: _torsion_budget(group=group) + _TORSION_UPDATE_CUDA, group),
            partial(torque_sweep_source, _MIRROR_DOFS, linked=linked),
            scene_names=_SCENE_TORSION_NAMES,
            steel_names=_SOURCE_TORSION_NAMES,
            keep=_TORSION_LAW,
        )
        for group in (False, True)
        for linked in (False, True)
    },
}
"""The laws the scene restates, by name: the friction modes' Coulomb-Newton solve (Steel's free-body text, the only one
it has), bisection (Steel's free-body and dense texts, one law) and de Saxce's bias, the friction patches' pooled
normal load (Steel's sparse sweep's articulated rows), and contact torsion's budget in each of the scene's row loops
(one thread, a lane group) against each form of Steel's torsion sweep (its rows scanned, its rows linked)."""


def _law_lines(text: str, begin: str | None, end: str | None, names: dict[str, str], mirror: _Mirror) -> list[str]:
    """The lines of the law in ``text``'s region between ``begin`` and ``end``, rewritten by ``names``."""
    tokens = _c_tokens(text)
    start = 0 if begin is None else _token_index(tokens, _c_tokens(begin), 0)
    stop = len(tokens) if end is None else _token_index(tokens, _c_tokens(end), start + 1)
    rules = [(_c_pattern(pattern), _c_pattern(common)) for pattern, common in names.items()]
    lines = []
    for statement in _CStatements(tokens[start:stop], rules).statements():
        target = statement[1] if statement[0] == "=" and statement[1][0] == "n" else None
        if target is not None and target[1] in mirror.drop:
            continue
        if mirror.keep is not None:
            assigned = target is not None and target[1] in mirror.keep
            tested = statement[0] == "if" and _c_names(statement[1]) <= mirror.keep
            if not (assigned or tested):
                continue
        lines.append(_c_line(statement))
    return lines


def _token_index(tokens: list[str], marker: list[str], start: int) -> int:
    """The first index from ``start`` where ``marker``'s tokens appear in ``tokens``."""
    for at in range(start, len(tokens) - len(marker) + 1):
        if tokens[at : at + len(marker)] == marker:
            return at
    raise ValueError(f"the law's marker {' '.join(marker)!r} is not in its text")


def _mirrored_lines(name: str, *, steel: str | None = None) -> tuple[list[str], list[str]]:
    """The mirrored lines of the restated law ``name`` (a key of :data:`_MIRRORS`): the scene's, then Steel's.

    Equal lists mean that the scene's text states Steel's current law statement for statement. ``steel`` replaces
    Steel's text, so that the guard can be shown to see a change.
    """
    mirror = _MIRRORS[name]
    return (
        _law_lines(mirror.scene(), mirror.scene_begin, mirror.scene_end, mirror.scene_names, mirror),
        _law_lines(
            mirror.steel() if steel is None else steel,
            mirror.steel_begin,
            mirror.steel_end,
            mirror.steel_names,
            mirror,
        ),
    )
