# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-level aggregate broad phase.

Groups the colliding shapes of each articulation into *aggregates* (one per
articulation, or chunks of a long one; see :func:`infer_shape_aggregates`), bins
one union box per aggregate in a uniform hashed grid (or sweeps them with
:class:`BroadPhaseSAP`: with ``main="sap"``, or where the default of
:meth:`BroadPhaseAggregate.from_model` picks it), and expands the overlapping
aggregate pairs, each aggregate's own member pairs, its infinite planes and
heightfields, and the finite static shapes of a hashed static grid near it into
shape pairs.
Each pair reserves its output slot with one 64-bit atomic (each block of lanes
reserves its slots with one under ``_KNOBS["emission"] = "tile"``), and the
published count is clamped to the pair buffer. On request (``ordered=True``)
the pairs are instead counted per work item, scanned and written in a
documented, deterministic order.

See Also:
    :class:`BroadPhaseSAP` in ``broad_phase_sap.py`` for the shape-level sweep,
    :class:`BroadPhaseAllPairs` in ``broad_phase_nxn.py`` for the reference rules.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import warp as wp

from ..core._warnings import _external_warning_stacklevel
from ..core.types import Devicelike
from ..exceptions import NewtonWarning
from .broad_phase_common import (
    check_aabb_overlap,
    check_aabb_overlap_moving,
    is_pair_excluded,
    is_shape_pair_immovable_filtered,
    is_shape_pair_same_body_filtered,
    test_world_and_group_pair,
    write_pair,
)
from .broad_phase_sap import BroadPhaseSAP
from .flags import ShapeFlags
from .types import GeoType

if TYPE_CHECKING:
    from ..sim.model import Model

wp.set_module_options({"enable_backward": False})


AGGREGATE_PAIRS_PER_AGGREGATE = 16
"""Default capacity of the aggregate-pair list, per aggregate.

Sized from measured crowds of 65,536 humanoids: settled and moving piles reach
up to 4.14 overlapping aggregate pairs per humanoid, and twice that, rounded up to a power of two, gives 16. Single aggregates reach 17-19
pairs where piles meet; the list is shared, so only the mean matters.
"""

SHAPE_PAIRS_PER_SHAPE = 12
"""Default shape-pair capacity per colliding shape for the aggregate broad phase."""

UNBOUNDED_PER_WORLD_WARNING = 8
"""Construction warns when one world sees more unbounded shapes than this."""

STATIC_UNBOUNDED_CELLS = 64
"""A finite static shape whose box has an edge longer than this many static cells joins its world's unbounded list."""

_STATIC_CELL_SIZE_FLOOR = 0.5
"""Smallest derived static cell size [m]."""

_STATIC_ENTRY_SLACK = 64
"""Static grid capacity beyond twice the first build's entries."""

_STATIC_BIN_PAD = 1.0e-5
"""Relative pad of a static box binned on the host, ``1e-5 * max(1 m, |coordinate|)``.

The host bins in float64 and the device finds cells in float32; the pad, far
above float32 rounding, keeps every cell the device can compute for a point of
the box inside the box's binned cells.
"""

_STATIC_CELL_LIMIT = 1 << 29
"""Static cell coordinates are clamped to +-2**29, so that offset by 2**29 they hash as non-negative int32."""

_MIXED_GROUP_BASE = wp.constant(-(1 << 30))
"""Group summary base of an aggregate whose members are in different groups.

Aggregate ``a`` gets the unique negative summary ``_MIXED_GROUP_BASE - a``, which
:func:`test_group_pair` pairs with every summary except 0 and itself. User
collision groups must stay above this value.
"""

_KNOBS = {
    "main": "grid",
    "union": "thread",
    "self_filter": "mask",
    "emission": "atomic",
    "ordered": False,
    "statics": "grid",
    "max_aggregate_size": 32,
    "grid_cell_ratio_max": 3.0,
    "grid_min_aggregates": 32768,
    "grid_min_worlds": 2048,
    "grid_occupancy_max": 90.0,
}
"""Kernel and membership defaults, read at construction.

``main`` is ``"grid"`` (a uniform grid of the union boxes' lower corners, see
:class:`BroadPhaseAggregate`; it needs ``shape_reach``, which
:meth:`BroadPhaseAggregate.from_model` computes, and a direct construction
without one gets ``"sap"``; 22.8 % faster per collide than ``"sap"`` on 65,536
piled humanoids and 5.5 % slower on a sparse crowd of them, measured on an NVIDIA L40)
or ``"sap"`` (:class:`BroadPhaseSAP` over the union boxes);
``union`` is ``"thread"`` (one thread per aggregate) or ``"warp"`` (one block per
aggregate with tile reductions); ``self_filter`` is ``"mask"`` (a 32-bit
exclusion mask per member, for aggregates of at most 32 members) or ``"search"``
(the same-body test and a binary search of the filter pairs); ``emission`` is
``"atomic"`` (one atomic per pair, 26-31 % faster than ``"tile"`` on 65,536
humanoids on an L40) or ``"tile"`` (one atomic per block
and work item, a count pass then a write pass: the fallback should per-pair
atomics misbehave on another GPU); ``ordered`` ``True`` replaces the emission
by the ordered count, scan and write passes (see :class:`BroadPhaseAggregate`);
``statics`` is ``"grid"`` (finite static shapes in a hashed static grid, tested
by the ``"static"`` stage) or ``"aggregates"`` (each finite static shape becomes
a one-shape aggregate of the main level); ``max_aggregate_size`` 32 splits each
inferred aggregate into chunks of about 32 members over whole bodies, so a
chunk can exceed 32 by up to one body's shapes less one (76.6 % faster on
128-link chains under ``main="sap"`` on an L40, and no humanoid is split), and
0 keeps each articulation whole; ``grid_cell_ratio_max`` (a positive
finite number, 3.0) is the cell guard of :meth:`BroadPhaseAggregate.from_model`: the
longest aggregate's reach sets the grid's cell for every aggregate, and when the
cell would exceed this ratio times the median aggregate extent (2 x reach), the
default main level falls back to ``"sap"``.

``grid_min_aggregates`` (an integer >= 0, 32,768), ``grid_min_worlds`` (an
integer >= 1, 2,048) and ``grid_occupancy_max`` (a positive number, 90.0;
``inf`` turns the guard off) are the size, worlds and occupancy rules of the
default main level (see
:meth:`BroadPhaseAggregate.from_model`). Each is a judgment bracketed by
measurements, not a measured optimum, taken on an L40 at each scene's initial
pose:

- ``grid_min_aggregates``: the grid's main stage costs about 0.1-0.25 ms at
  every size while the SAP's grows with the aggregates, so in few worlds the
  SAP wins below a size. On piled humanoids in one world (S2) the grid wins
  from about 28,400 aggregates; on the sparse crowd (S1) it loses at every
  measured size, so this threshold costs S1 24 % at 32,768 and 8 % at 65,536
  aggregates (the latter within the 10 % the grid is allowed to lose on a
  sparse scene for its gain on piles).
- ``grid_min_worlds``: from this many worlds holding aggregates on, the size
  rule is skipped, since the SAP sorts every world's segment (0.289 ms of main
  stage at 4,096 worlds of one humanoid and 4.29 ms at 65,536, against the
  grid's fixed 0.15 ms). The threshold is derived from that cost; nothing was
  measured between 1 and 4,096 worlds.
- ``grid_occupancy_max``: the mean over aggregates of how many aggregates share
  its grid cell, itself included. The grid-side scenes read at most 18.9 (S2),
  the chains of 128 capsules at least 428, and nothing was measured between:
  where the grid starts to lose as the occupancy rises is unknown, and 90
  leaves 4.8x on either side. The chains are below the size threshold too, so
  this guard decides only for crowded scenes above it or in many worlds.
"""

_KNOB_VALUES = {
    "main": ("sap", "grid"),
    "union": ("thread", "warp"),
    "self_filter": ("mask", "search"),
    "emission": ("tile", "atomic"),
    "ordered": (False, True),
    "statics": ("grid", "aggregates"),
}

_MASK_WIDTH = wp.constant(32)
_BOX_INFINITY = wp.constant(1.0e30)
_UNION_PAD = wp.constant(1.0e-6)
"""Outward pad of each union box: ``1e-6 * max(1 m, largest |coordinate|)`` on every side.

:class:`BroadPhaseSAP`'s sweep is half open: a box whose projected lower bound
equals another's projected upper bound is not swept, so union boxes that only
touch on its axis, two coincident zero-extent boxes among them, would never
pair. The pad, at least eight float32 ulps of the box's largest coordinate,
survives the rounding of the projection; an absolute pad would vanish in it far
from the origin. Padding only adds aggregate pairs, whose members are then
tested exactly.
"""

# Emission modes of the member-test visitors.
_COUNT = wp.constant(0)
_WRITE = wp.constant(1)
_ATOMIC = wp.constant(2)
_ORDERED_WRITE = wp.constant(3)

_STATIC_CELL_LIMIT_FLOAT = wp.constant(float(_STATIC_CELL_LIMIT))
_STATIC_CELL_OFFSET = wp.constant(_STATIC_CELL_LIMIT)
_HASH_WORLD = wp.constant(wp.uint32(0x9E3779B1))
_HASH_X = wp.constant(wp.uint32(0x85EBCA77))
_HASH_Y = wp.constant(wp.uint32(0xC2B2AE3D))
_HASH_Z = wp.constant(wp.uint32(0x27D4EB2F))
_MIX_1 = wp.constant(wp.uint32(0x85EBCA6B))
_MIX_2 = wp.constant(wp.uint32(0xC2B2AE35))

_ORDER_SENTINEL = wp.constant(wp.int64(0x7FFFFFFFFFFFFFFF))
"""Sort key of the unused aggregate-pair slots of the ordered list; above every ``(a << 32) | b``."""

_ORDERED_SLOTS_MAX = 2**31 - 1
"""Largest pair count the int32 item offsets of the ordered list can address."""

_GRID_CELL_SLACK = 2.0**-8
"""Relative slack of the grid's cell over twice the largest member reach (see :class:`BroadPhaseAggregate`)."""

_RADIUS_ROUNDING = 1.0e-6
"""Relative tolerance within which a primitive's scale bound counts as its collision radius.

The builder stores the radius as a float32 of the float64 sum or norm of the
scale, and the host recomputes it from the float32 scale: the two differ by a
few float32 ulps (2**-23 relative each). Keeping the radius within this
tolerance keeps builder-authored primitives' reach bit for bit; the at most
1e-6 relative it can under-bound is far inside the cell's ``_GRID_CELL_SLACK``.
"""

_GRID_OWN_CELL = wp.constant(13)
"""Index of the offset (0, 0, 0) among the 27 grid neighbour offsets, numbered x fastest.

Offsets 14..26 are the half shell's forward offsets: dz > 0, or dz == 0 and
dy > 0, or dz == dy == 0 and dx > 0; their opposites are 0..12.
"""

AggregateStage = Literal["union", "main", "self", "static", "expand", "write", "publish"]
_STAGES = ("union", "main", "self", "static", "expand", "write", "publish")


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


@wp.func
def _swept_box(
    shape: int,
    shape_lower: wp.array[wp.vec3],
    shape_upper: wp.array[wp.vec3],
    shape_displacement: wp.array[wp.vec3],
    displacement_limit: float,
):
    """Return a shape's box swept by its displacement, clamped per axis when the limit is non-negative."""
    lower = shape_lower[shape]
    upper = shape_upper[shape]
    if shape_displacement.shape[0] > 0:
        d = shape_displacement[shape]
        if displacement_limit >= 0.0:
            d = wp.vec3(
                wp.clamp(d[0], -displacement_limit, displacement_limit),
                wp.clamp(d[1], -displacement_limit, displacement_limit),
                wp.clamp(d[2], -displacement_limit, displacement_limit),
            )
        lower = lower + wp.min(d, wp.vec3(0.0))
        upper = upper + wp.max(d, wp.vec3(0.0))
    return lower, upper


@wp.func
def _store_union(
    a: int,
    lower: wp.vec3,
    upper: wp.vec3,
    group: int,
    aggregate_lower: wp.array[wp.vec3],
    aggregate_upper: wp.array[wp.vec3],
    aggregate_group: wp.array[int],
    extent_max: wp.array[float],
):
    reach = wp.max(
        wp.max(wp.max(wp.abs(lower[0]), wp.abs(lower[1])), wp.abs(lower[2])),
        wp.max(wp.max(wp.abs(upper[0]), wp.abs(upper[1])), wp.abs(upper[2])),
    )
    pad = wp.vec3(_UNION_PAD * wp.max(reach, 1.0))
    aggregate_lower[a] = lower - pad
    aggregate_upper[a] = upper + pad
    aggregate_group[a] = group
    # The extent statistic is the unpadded union's.
    edge = upper - lower
    extent = wp.max(edge[0], wp.max(edge[1], edge[2]))
    # Read first so that a settled latch costs no atomic.
    if extent > extent_max[0]:
        wp.atomic_max(extent_max, 0, extent)


@wp.func
def _pair_passes(
    lo: int,
    hi: int,
    shape_lower: wp.array[wp.vec3],
    shape_upper: wp.array[wp.vec3],
    shape_group: wp.array[int],
    shape_world: wp.array[int],
    shape_body: wp.array[int],
    body_flags: wp.array[int],
    include_static_kinematic_pairs: bool,
    shape_displacement: wp.array[wp.vec3],
) -> bool:
    """Apply the world/group, (swept) overlap and immovable tests to a canonical pair."""
    if not test_world_and_group_pair(shape_world[lo], shape_world[hi], shape_group[lo], shape_group[hi]):
        return False
    if not check_aabb_overlap_moving(lo, hi, shape_lower, shape_upper, 0.0, 0.0, shape_displacement):
        return False
    return not is_shape_pair_immovable_filtered(lo, hi, shape_body, body_flags, include_static_kinematic_pairs)


@wp.func
def _emit(
    pair: wp.vec2i,
    mode: int,
    slot: wp.int64,
    candidate_pair: wp.array[wp.vec2i],
    pair_demand: wp.array[wp.int64],
):
    """Write a pair at its reserved slot, or reserve one with an atomic, dropping slots past the buffer."""
    target = slot
    if mode == _ATOMIC:
        target = wp.atomic_add(pair_demand, 0, wp.int64(1))
    if target < wp.int64(candidate_pair.shape[0]):
        candidate_pair[wp.int32(target)] = pair


@wp.func
def _reserve_block_slots(hits: int, lane: int, capacity: int, pair_demand: wp.array[wp.int64]) -> wp.int64:
    """Reserve a block's slots with one 64-bit atomic and return this lane's first slot.

    Every lane of the block must call this. The base is broadcast as an int32
    clamped to ``capacity``, so a block past the buffer gets slots that are all
    dropped.
    """
    offsets = wp.tile_scan_exclusive(wp.tile(hits))
    offset = wp.untile(offsets)
    last = wp.block_dim() - 1
    base = int(0)
    if lane == last:
        total = offset + hits
        if total > 0:
            old = wp.atomic_add(pair_demand, 0, wp.int64(total))
            base = wp.int32(wp.min(old, wp.int64(capacity)))
    block_base = wp.tile_extract(wp.tile(base), last)
    return wp.int64(block_base) + wp.int64(offset)


@wp.func
def _self_visit(
    a: int,
    lane: int,
    stride: int,
    mode: int,
    slot: wp.int64,
    use_mask: bool,
    aggregate_start: wp.array[int],
    aggregate_shapes: wp.array[int],
    aggregate_world: wp.array[int],
    aggregate_mask: wp.array2d[wp.uint32],
    aggregate_lower: wp.array[wp.vec3],
    aggregate_upper: wp.array[wp.vec3],
    unbounded_start: wp.array[int],
    unbounded_shapes: wp.array[int],
    world_count: int,
    shape_lower: wp.array[wp.vec3],
    shape_upper: wp.array[wp.vec3],
    shape_group: wp.array[int],
    shape_world: wp.array[int],
    shape_body: wp.array[int],
    body_flags: wp.array[int],
    include_static_kinematic_pairs: bool,
    shape_displacement: wp.array[wp.vec3],
    displacement_limit: float,
    filter_pairs: wp.array[wp.vec2i],
    filter_pair_count: int,
    cross_filter_pairs: wp.array[wp.vec2i],
    cross_filter_pair_count: int,
    candidate_pair: wp.array[wp.vec2i],
    pair_demand: wp.array[wp.int64],
) -> int:
    """Test one aggregate's member pairs and its members against its unbounded shapes.

    Returns the lane's hit count. Modes other than ``_COUNT`` also emit the hits,
    in the same order on every call.
    """
    start = aggregate_start[a]
    n = aggregate_start[a + 1] - start
    hits = int(0)

    # Member pairs (i, j), i < j, in row-major order, ``stride`` apart per lane.
    i = int(0)
    j = lane + 1
    while i < n - 1 and j >= n:
        j = j - n + i + 2
        i = i + 1
    while i < n - 1:
        s = aggregate_shapes[start + i]
        t = aggregate_shapes[start + j]
        excluded = bool(False)
        if use_mask:
            excluded = ((aggregate_mask[a, i] >> wp.uint32(j)) & wp.uint32(1)) != wp.uint32(0)
        else:
            excluded = is_shape_pair_same_body_filtered(s, t, shape_body) or is_pair_excluded(
                wp.vec2i(s, t), filter_pairs, filter_pair_count
            )
        if not excluded:
            if _pair_passes(
                s,
                t,
                shape_lower,
                shape_upper,
                shape_group,
                shape_world,
                shape_body,
                body_flags,
                include_static_kinematic_pairs,
                shape_displacement,
            ):
                if mode != _COUNT:
                    _emit(wp.vec2i(s, t), mode, slot + wp.int64(hits), candidate_pair, pair_demand)
                hits += 1
        j += stride
        while i < n - 1 and j >= n:
            j = j - n + i + 2
            i = i + 1

    # Unbounded shapes: the world -1 segment and the aggregate's own world, or
    # every segment for a world -1 aggregate.
    world = aggregate_world[a]
    union_lower = aggregate_lower[a]
    union_upper = aggregate_upper[a]
    for segment in range(2):
        begin = int(0)
        end = int(0)
        if segment == 0:
            begin = unbounded_start[0]
            end = unbounded_start[1]
            if world < 0:
                end = unbounded_start[world_count + 1]
        elif world >= 0:
            begin = unbounded_start[world + 1]
            end = unbounded_start[world + 2]
        for k in range(begin, end):
            u = unbounded_shapes[k]
            u_lower, u_upper = _swept_box(u, shape_lower, shape_upper, shape_displacement, displacement_limit)
            if check_aabb_overlap(union_lower, union_upper, 0.0, u_lower, u_upper, 0.0):
                m = lane
                while m < n:
                    s = aggregate_shapes[start + m]
                    lo = wp.min(s, u)
                    hi = wp.max(s, u)
                    if _pair_passes(
                        lo,
                        hi,
                        shape_lower,
                        shape_upper,
                        shape_group,
                        shape_world,
                        shape_body,
                        body_flags,
                        include_static_kinematic_pairs,
                        shape_displacement,
                    ):
                        if not is_pair_excluded(wp.vec2i(lo, hi), cross_filter_pairs, cross_filter_pair_count):
                            if mode != _COUNT:
                                _emit(wp.vec2i(lo, hi), mode, slot + wp.int64(hits), candidate_pair, pair_demand)
                            hits += 1
                    m += stride
    return hits


@wp.func
def _expand_visit(
    pair: wp.vec2i,
    lane: int,
    stride: int,
    mode: int,
    slot: wp.int64,
    aggregate_start: wp.array[int],
    aggregate_shapes: wp.array[int],
    shape_lower: wp.array[wp.vec3],
    shape_upper: wp.array[wp.vec3],
    shape_group: wp.array[int],
    shape_world: wp.array[int],
    shape_body: wp.array[int],
    body_flags: wp.array[int],
    include_static_kinematic_pairs: bool,
    shape_displacement: wp.array[wp.vec3],
    cross_filter_pairs: wp.array[wp.vec2i],
    cross_filter_pair_count: int,
    candidate_pair: wp.array[wp.vec2i],
    pair_demand: wp.array[wp.int64],
) -> int:
    """Test the member pairs of one aggregate pair: lanes stride over the first, each loops over the second.

    Returns the lane's hit count. Modes other than ``_COUNT`` also emit the hits,
    in the same order on every call.
    """
    a_end = aggregate_start[pair[0] + 1]
    b_start = aggregate_start[pair[1]]
    b_end = aggregate_start[pair[1] + 1]
    moving = shape_displacement.shape[0] > 0
    hits = int(0)
    m = aggregate_start[pair[0]] + lane
    while m < a_end:
        s = aggregate_shapes[m]
        s_lower = shape_lower[s]
        s_upper = shape_upper[s]
        s_world = shape_world[s]
        s_group = shape_group[s]
        for k in range(b_start, b_end):
            t = aggregate_shapes[k]
            if test_world_and_group_pair(s_world, shape_world[t], s_group, shape_group[t]):
                lo = wp.min(s, t)
                hi = wp.max(s, t)
                overlap = bool(False)
                if moving:
                    overlap = check_aabb_overlap_moving(lo, hi, shape_lower, shape_upper, 0.0, 0.0, shape_displacement)
                else:
                    overlap = check_aabb_overlap(s_lower, s_upper, 0.0, shape_lower[t], shape_upper[t], 0.0)
                if overlap:
                    if not is_shape_pair_immovable_filtered(
                        lo, hi, shape_body, body_flags, include_static_kinematic_pairs
                    ):
                        if not is_pair_excluded(wp.vec2i(lo, hi), cross_filter_pairs, cross_filter_pair_count):
                            if mode != _COUNT:
                                _emit(wp.vec2i(lo, hi), mode, slot + wp.int64(hits), candidate_pair, pair_demand)
                            hits += 1
        m += stride
    return hits


@wp.func
def _emit_ordered(hit: int, pair: wp.vec2i, base: int, emitted: int, candidate_pair: wp.array[wp.vec2i]) -> int:
    """Write one round's hits in lane order at ``base + emitted`` onward and return the new ``emitted``.

    Every lane of the block must call this, with the same ``base`` and
    ``emitted``. Slots past the buffer are dropped.
    """
    ranks = wp.tile_scan_inclusive(wp.tile(hit))
    rank = wp.untile(ranks)
    total = wp.tile_extract(ranks, wp.block_dim() - 1)
    if hit != 0:
        slot = base + emitted + rank - 1
        if slot < candidate_pair.shape[0]:
            candidate_pair[slot] = pair
    return emitted + total


@wp.func
def _self_visit_ordered(
    a: int,
    lane: int,
    stride: int,
    write: bool,
    base: int,
    use_mask: bool,
    aggregate_start: wp.array[int],
    aggregate_shapes: wp.array[int],
    aggregate_world: wp.array[int],
    aggregate_mask: wp.array2d[wp.uint32],
    aggregate_lower: wp.array[wp.vec3],
    aggregate_upper: wp.array[wp.vec3],
    unbounded_start: wp.array[int],
    unbounded_shapes: wp.array[int],
    world_count: int,
    shape_lower: wp.array[wp.vec3],
    shape_upper: wp.array[wp.vec3],
    shape_group: wp.array[int],
    shape_world: wp.array[int],
    shape_body: wp.array[int],
    body_flags: wp.array[int],
    include_static_kinematic_pairs: bool,
    shape_displacement: wp.array[wp.vec3],
    displacement_limit: float,
    filter_pairs: wp.array[wp.vec2i],
    filter_pair_count: int,
    cross_filter_pairs: wp.array[wp.vec2i],
    cross_filter_pair_count: int,
    candidate_pair: wp.array[wp.vec2i],
) -> int:
    """Test one aggregate's member pairs, then its members against its unbounded shapes, in test order.

    The tests run in rounds of ``stride``: round r gives lane l the test
    ``r * stride + l`` of each sequence, with the same rules as
    :func:`_self_visit`. Counting (``write`` false) returns the lane's hits;
    writing emits the block's hits in test order from ``base`` and returns the
    block's total. Every lane of the block must call this.
    """
    start = aggregate_start[a]
    n = aggregate_start[a + 1] - start
    hits = int(0)

    # Member pairs (i, j), i < j, in row-major order.
    i = int(0)
    j = lane + 1
    while i < n - 1 and j >= n:
        j = j - n + i + 2
        i = i + 1
    for _round in range((n * (n - 1) // 2 + stride - 1) // stride):
        hit = int(0)
        pair = wp.vec2i(0, 0)
        if i < n - 1:
            s = aggregate_shapes[start + i]
            t = aggregate_shapes[start + j]
            excluded = bool(False)
            if use_mask:
                excluded = ((aggregate_mask[a, i] >> wp.uint32(j)) & wp.uint32(1)) != wp.uint32(0)
            else:
                excluded = is_shape_pair_same_body_filtered(s, t, shape_body) or is_pair_excluded(
                    wp.vec2i(s, t), filter_pairs, filter_pair_count
                )
            if not excluded:
                if _pair_passes(
                    s,
                    t,
                    shape_lower,
                    shape_upper,
                    shape_group,
                    shape_world,
                    shape_body,
                    body_flags,
                    include_static_kinematic_pairs,
                    shape_displacement,
                ):
                    hit = 1
                    pair = wp.vec2i(s, t)
            j += stride
            while i < n - 1 and j >= n:
                j = j - n + i + 2
                i = i + 1
        if write:
            hits = _emit_ordered(hit, pair, base, hits, candidate_pair)
        else:
            hits += hit

    # Unbounded shapes in CSR order: the world -1 segment and the aggregate's
    # own world, or every segment for a world -1 aggregate.
    world = aggregate_world[a]
    union_lower = aggregate_lower[a]
    union_upper = aggregate_upper[a]
    for segment in range(2):
        begin = int(0)
        end = int(0)
        if segment == 0:
            begin = unbounded_start[0]
            end = unbounded_start[1]
            if world < 0:
                end = unbounded_start[world_count + 1]
        elif world >= 0:
            begin = unbounded_start[world + 1]
            end = unbounded_start[world + 2]
        for k in range(begin, end):
            u = unbounded_shapes[k]
            u_lower, u_upper = _swept_box(u, shape_lower, shape_upper, shape_displacement, displacement_limit)
            if check_aabb_overlap(union_lower, union_upper, 0.0, u_lower, u_upper, 0.0):
                for round_index in range((n + stride - 1) // stride):
                    m = round_index * stride + lane
                    hit = int(0)
                    pair = wp.vec2i(0, 0)
                    if m < n:
                        s = aggregate_shapes[start + m]
                        lo = wp.min(s, u)
                        hi = wp.max(s, u)
                        if _pair_passes(
                            lo,
                            hi,
                            shape_lower,
                            shape_upper,
                            shape_group,
                            shape_world,
                            shape_body,
                            body_flags,
                            include_static_kinematic_pairs,
                            shape_displacement,
                        ):
                            if not is_pair_excluded(wp.vec2i(lo, hi), cross_filter_pairs, cross_filter_pair_count):
                                hit = 1
                                pair = wp.vec2i(lo, hi)
                    if write:
                        hits = _emit_ordered(hit, pair, base, hits, candidate_pair)
                    else:
                        hits += hit
    return hits


@wp.func
def _expand_visit_ordered(
    pair: wp.vec2i,
    lane: int,
    stride: int,
    write: bool,
    base: int,
    aggregate_start: wp.array[int],
    aggregate_shapes: wp.array[int],
    shape_lower: wp.array[wp.vec3],
    shape_upper: wp.array[wp.vec3],
    shape_group: wp.array[int],
    shape_world: wp.array[int],
    shape_body: wp.array[int],
    body_flags: wp.array[int],
    include_static_kinematic_pairs: bool,
    shape_displacement: wp.array[wp.vec3],
    cross_filter_pairs: wp.array[wp.vec2i],
    cross_filter_pair_count: int,
    candidate_pair: wp.array[wp.vec2i],
) -> int:
    """Test the member pairs of one aggregate pair in (member of a) x (member of b) order.

    The tests run in rounds of ``stride``, with the same rules as
    :func:`_expand_visit`. Counting (``write`` false) returns the lane's hits;
    writing emits the block's hits in test order from ``base`` and returns the
    block's total. Every lane of the block must call this.
    """
    a_start = aggregate_start[pair[0]]
    b_start = aggregate_start[pair[1]]
    b_count = aggregate_start[pair[1] + 1] - b_start
    tests = (aggregate_start[pair[0] + 1] - a_start) * b_count
    moving = shape_displacement.shape[0] > 0
    hits = int(0)
    for round_index in range((tests + stride - 1) // stride):
        index = round_index * stride + lane
        hit = int(0)
        emitted_pair = wp.vec2i(0, 0)
        if index < tests:
            row = index // b_count
            s = aggregate_shapes[a_start + row]
            t = aggregate_shapes[b_start + index - row * b_count]
            if test_world_and_group_pair(shape_world[s], shape_world[t], shape_group[s], shape_group[t]):
                lo = wp.min(s, t)
                hi = wp.max(s, t)
                overlap = bool(False)
                if moving:
                    overlap = check_aabb_overlap_moving(lo, hi, shape_lower, shape_upper, 0.0, 0.0, shape_displacement)
                else:
                    overlap = check_aabb_overlap(
                        shape_lower[s], shape_upper[s], 0.0, shape_lower[t], shape_upper[t], 0.0
                    )
                if overlap:
                    if not is_shape_pair_immovable_filtered(
                        lo, hi, shape_body, body_flags, include_static_kinematic_pairs
                    ):
                        if not is_pair_excluded(wp.vec2i(lo, hi), cross_filter_pairs, cross_filter_pair_count):
                            hit = 1
                            emitted_pair = wp.vec2i(lo, hi)
        if write:
            hits = _emit_ordered(hit, emitted_pair, base, hits, candidate_pair)
        else:
            hits += hit
    return hits


@wp.func
def _static_bucket(world_key: int, x: int, y: int, z: int, mask: int) -> int:
    """Hash a world key (world + 1) and a static cell to a bucket of the static table; ``mask`` is its size - 1."""
    h = wp.uint32(world_key) * _HASH_WORLD
    h = h ^ (wp.uint32(x + _STATIC_CELL_OFFSET) * _HASH_X)
    h = h ^ (wp.uint32(y + _STATIC_CELL_OFFSET) * _HASH_Y)
    h = h ^ (wp.uint32(z + _STATIC_CELL_OFFSET) * _HASH_Z)
    h = h ^ (h >> wp.uint32(16))
    h = h * _MIX_1
    h = h ^ (h >> wp.uint32(13))
    h = h * _MIX_2
    h = h ^ (h >> wp.uint32(16))
    return wp.int32(h & wp.uint32(mask))


@wp.func
def _static_cell(floored: float) -> int:
    """Clamp a floored cell coordinate to +-2**29 and convert it; NaN gives 0."""
    if wp.isnan(floored):
        return 0
    return int(wp.clamp(floored, -_STATIC_CELL_LIMIT_FLOAT, _STATIC_CELL_LIMIT_FLOAT))


@wp.func
def _static_member_hits(
    g: int,
    lane: int,
    stride: int,
    mode: int,
    slot: wp.int64,
    base: int,
    hits: int,
    start: int,
    n: int,
    union_lower: wp.vec3,
    union_upper: wp.vec3,
    aggregate_shapes: wp.array[int],
    static_shapes: wp.array[int],
    shape_lower: wp.array[wp.vec3],
    shape_upper: wp.array[wp.vec3],
    shape_group: wp.array[int],
    shape_world: wp.array[int],
    shape_body: wp.array[int],
    body_flags: wp.array[int],
    include_static_kinematic_pairs: bool,
    shape_displacement: wp.array[wp.vec3],
    displacement_limit: float,
    cross_filter_pairs: wp.array[wp.vec2i],
    cross_filter_pair_count: int,
    candidate_pair: wp.array[wp.vec2i],
    pair_demand: wp.array[wp.int64],
) -> int:
    """Test an aggregate's members against the grid static ``g`` and return the updated ``hits``.

    The lanes take the members in rounds of ``stride``, with the rules of the
    unbounded tests in :func:`_self_visit`: the union box against the static's
    (swept) box, then per member world/group, (swept) overlap, immovable pair
    and cross exclusion. ``_ORDERED_WRITE`` writes the block's hits in member
    order from ``base + hits`` and returns the block's running total (every
    lane must call it); the other modes count, or emit, the lane's hits.
    """
    u = static_shapes[g]
    u_lower, u_upper = _swept_box(u, shape_lower, shape_upper, shape_displacement, displacement_limit)
    if check_aabb_overlap(union_lower, union_upper, 0.0, u_lower, u_upper, 0.0):
        for round_index in range((n + stride - 1) // stride):
            m = round_index * stride + lane
            hit = int(0)
            pair = wp.vec2i(0, 0)
            if m < n:
                s = aggregate_shapes[start + m]
                lo = wp.min(s, u)
                hi = wp.max(s, u)
                if _pair_passes(
                    lo,
                    hi,
                    shape_lower,
                    shape_upper,
                    shape_group,
                    shape_world,
                    shape_body,
                    body_flags,
                    include_static_kinematic_pairs,
                    shape_displacement,
                ):
                    if not is_pair_excluded(wp.vec2i(lo, hi), cross_filter_pairs, cross_filter_pair_count):
                        hit = 1
                        pair = wp.vec2i(lo, hi)
            if mode == _ORDERED_WRITE:
                hits = _emit_ordered(hit, pair, base, hits, candidate_pair)
            elif hit != 0:
                if mode != _COUNT:
                    _emit(pair, mode, slot + wp.int64(hits), candidate_pair, pair_demand)
                hits += 1
    return hits


@wp.func
def _static_visit(
    a: int,
    lane: int,
    stride: int,
    mode: int,
    slot: wp.int64,
    base: int,
    aggregate_start: wp.array[int],
    aggregate_shapes: wp.array[int],
    aggregate_world: wp.array[int],
    aggregate_lower: wp.array[wp.vec3],
    aggregate_upper: wp.array[wp.vec3],
    static_start: wp.array[int],
    static_shapes: wp.array[int],
    static_key: wp.array[int],
    static_cell_lower: wp.array[wp.vec3i],
    static_cell_upper: wp.array[wp.vec3i],
    static_entries: wp.array[wp.vec4i],
    static_bucket_start: wp.array[int],
    static_bucket_mask: int,
    inv_cell_size: float,
    shape_lower: wp.array[wp.vec3],
    shape_upper: wp.array[wp.vec3],
    shape_group: wp.array[int],
    shape_world: wp.array[int],
    shape_body: wp.array[int],
    body_flags: wp.array[int],
    include_static_kinematic_pairs: bool,
    shape_displacement: wp.array[wp.vec3],
    displacement_limit: float,
    cross_filter_pairs: wp.array[wp.vec2i],
    cross_filter_pair_count: int,
    candidate_pair: wp.array[wp.vec2i],
    pair_demand: wp.array[wp.int64],
) -> int:
    """Test one aggregate's members against the grid statics whose binned cells meet its cells, each once.

    The statics in view are world -1's and the aggregate's own world's. When the
    cells under the aggregate's box, times its world keys (one for a world -1
    aggregate, else two), are at most the statics in view, the aggregate walks
    those cells (z, then y, then x), looks each up under world -1's key and
    then its own world's, and tests a static only in the cell holding the lower
    corner of the intersection of its cells with the static's binned cells.
    Otherwise (also for a non-finite box, which fails every overlap test) it
    lists the statics in view, world -1's then its world's, and tests those
    whose binned cells meet its cells. Either way a static is seen where it was
    binned. Returns ``hits`` as :func:`_static_member_hits` does.
    """
    start = aggregate_start[a]
    n = aggregate_start[a + 1] - start
    world = aggregate_world[a]
    union_lower = aggregate_lower[a]
    union_upper = aggregate_upper[a]
    keys = int(1)
    own_begin = int(0)
    own_end = int(0)
    if world >= 0:
        keys = 2
        own_begin = static_start[world + 1]
        own_end = static_start[world + 2]
    in_view = static_start[1] - static_start[0] + own_end - own_begin
    low = wp.vec3(
        wp.floor(union_lower[0] * inv_cell_size),
        wp.floor(union_lower[1] * inv_cell_size),
        wp.floor(union_lower[2] * inv_cell_size),
    )
    high = wp.vec3(
        wp.floor(union_upper[0] * inv_cell_size),
        wp.floor(union_upper[1] * inv_cell_size),
        wp.floor(union_upper[2] * inv_cell_size),
    )
    span = high - low + wp.vec3(1.0)
    cell_low = wp.vec3i(_static_cell(low[0]), _static_cell(low[1]), _static_cell(low[2]))
    cell_high = wp.vec3i(_static_cell(high[0]), _static_cell(high[1]), _static_cell(high[2]))
    hits = int(0)
    # A non-finite box gives a NaN or infinite cell count and lists the statics.
    if span[0] * span[1] * span[2] * float(keys) <= float(in_view):
        for z in range(cell_low[2], cell_high[2] + 1):
            for y in range(cell_low[1], cell_high[1] + 1):
                for x in range(cell_low[0], cell_high[0] + 1):
                    for key_index in range(keys):
                        world_key = key_index * (world + 1)
                        bucket = _static_bucket(world_key, x, y, z, static_bucket_mask)
                        for e in range(static_bucket_start[bucket], static_bucket_start[bucket + 1]):
                            entry = static_entries[e]
                            g = entry[0]
                            if entry[1] == x and entry[2] == y and entry[3] == z and static_key[g] == world_key:
                                corner = static_cell_lower[g]
                                if (
                                    x == wp.max(cell_low[0], corner[0])
                                    and y == wp.max(cell_low[1], corner[1])
                                    and z == wp.max(cell_low[2], corner[2])
                                ):
                                    hits = _static_member_hits(
                                        g,
                                        lane,
                                        stride,
                                        mode,
                                        slot,
                                        base,
                                        hits,
                                        start,
                                        n,
                                        union_lower,
                                        union_upper,
                                        aggregate_shapes,
                                        static_shapes,
                                        shape_lower,
                                        shape_upper,
                                        shape_group,
                                        shape_world,
                                        shape_body,
                                        body_flags,
                                        include_static_kinematic_pairs,
                                        shape_displacement,
                                        displacement_limit,
                                        cross_filter_pairs,
                                        cross_filter_pair_count,
                                        candidate_pair,
                                        pair_demand,
                                    )
    else:
        for segment in range(keys):
            begin = static_start[0]
            end = static_start[1]
            if segment == 1:
                begin = own_begin
                end = own_end
            for g in range(begin, end):
                corner = static_cell_lower[g]
                top = static_cell_upper[g]
                if (
                    cell_low[0] <= top[0]
                    and corner[0] <= cell_high[0]
                    and cell_low[1] <= top[1]
                    and corner[1] <= cell_high[1]
                    and cell_low[2] <= top[2]
                    and corner[2] <= cell_high[2]
                ):
                    hits = _static_member_hits(
                        g,
                        lane,
                        stride,
                        mode,
                        slot,
                        base,
                        hits,
                        start,
                        n,
                        union_lower,
                        union_upper,
                        aggregate_shapes,
                        static_shapes,
                        shape_lower,
                        shape_upper,
                        shape_group,
                        shape_world,
                        shape_body,
                        body_flags,
                        include_static_kinematic_pairs,
                        shape_displacement,
                        displacement_limit,
                        cross_filter_pairs,
                        cross_filter_pair_count,
                        candidate_pair,
                        pair_demand,
                    )
    return hits


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@functools.cache
def _create_union_kernel(union: str):
    """Create the union-box kernel: one thread per aggregate, or one block with tile reductions."""
    if union == "thread":

        @wp.kernel(enable_backward=False, module="unique")
        def union_thread_kernel(
            aggregate_count: int,
            aggregate_start: wp.array[int],
            aggregate_shapes: wp.array[int],
            shape_lower: wp.array[wp.vec3],
            shape_upper: wp.array[wp.vec3],
            shape_group: wp.array[int],
            shape_displacement: wp.array[wp.vec3],
            displacement_limit: float,
            # outputs
            aggregate_lower: wp.array[wp.vec3],
            aggregate_upper: wp.array[wp.vec3],
            aggregate_group: wp.array[int],
            extent_max: wp.array[float],
            aggregate_pair_count: wp.array[int],
            pair_demand: wp.array[wp.int64],
        ):
            a = wp.tid()
            if a == 0:
                aggregate_pair_count[0] = 0
                pair_demand[0] = wp.int64(0)
            if a >= aggregate_count:
                return
            start = aggregate_start[a]
            end = aggregate_start[a + 1]
            lower = wp.vec3(_BOX_INFINITY)
            upper = wp.vec3(-_BOX_INFINITY)
            first_group = shape_group[aggregate_shapes[start]]
            mixed = int(0)
            for m in range(start, end):
                s = aggregate_shapes[m]
                box_lower, box_upper = _swept_box(s, shape_lower, shape_upper, shape_displacement, displacement_limit)
                lower = wp.min(lower, box_lower)
                upper = wp.max(upper, box_upper)
                if shape_group[s] != first_group:
                    mixed = 1
            group = first_group
            if mixed != 0:
                group = _MIXED_GROUP_BASE - a
            _store_union(a, lower, upper, group, aggregate_lower, aggregate_upper, aggregate_group, extent_max)

        return union_thread_kernel

    @wp.kernel(enable_backward=False, module="unique")
    def union_warp_kernel(
        aggregate_count: int,
        aggregate_start: wp.array[int],
        aggregate_shapes: wp.array[int],
        shape_lower: wp.array[wp.vec3],
        shape_upper: wp.array[wp.vec3],
        shape_group: wp.array[int],
        shape_displacement: wp.array[wp.vec3],
        displacement_limit: float,
        # outputs
        aggregate_lower: wp.array[wp.vec3],
        aggregate_upper: wp.array[wp.vec3],
        aggregate_group: wp.array[int],
        extent_max: wp.array[float],
        aggregate_pair_count: wp.array[int],
        pair_demand: wp.array[wp.int64],
    ):
        a, lane = wp.tid()
        if a == 0 and lane == 0:
            aggregate_pair_count[0] = 0
            pair_demand[0] = wp.int64(0)
        if a >= aggregate_count:
            return
        end = aggregate_start[a + 1]
        lower = wp.vec3(_BOX_INFINITY)
        upper = wp.vec3(-_BOX_INFINITY)
        group_min = int(2147483647)
        group_max = int(-2147483647)
        m = aggregate_start[a] + lane
        while m < end:
            s = aggregate_shapes[m]
            box_lower, box_upper = _swept_box(s, shape_lower, shape_upper, shape_displacement, displacement_limit)
            lower = wp.min(lower, box_lower)
            upper = wp.max(upper, box_upper)
            group_min = wp.min(group_min, shape_group[s])
            group_max = wp.max(group_max, shape_group[s])
            m += wp.block_dim()
        union_lower = wp.vec3(
            wp.tile_extract(wp.tile_min(wp.tile(lower[0])), 0),
            wp.tile_extract(wp.tile_min(wp.tile(lower[1])), 0),
            wp.tile_extract(wp.tile_min(wp.tile(lower[2])), 0),
        )
        union_upper = wp.vec3(
            wp.tile_extract(wp.tile_max(wp.tile(upper[0])), 0),
            wp.tile_extract(wp.tile_max(wp.tile(upper[1])), 0),
            wp.tile_extract(wp.tile_max(wp.tile(upper[2])), 0),
        )
        first_group = wp.tile_extract(wp.tile_min(wp.tile(group_min)), 0)
        last_group = wp.tile_extract(wp.tile_max(wp.tile(group_max)), 0)
        if lane == 0:
            group = first_group
            if first_group != last_group:
                group = _MIXED_GROUP_BASE - a
            _store_union(
                a, union_lower, union_upper, group, aggregate_lower, aggregate_upper, aggregate_group, extent_max
            )

    return union_warp_kernel


@wp.func
def _verifying(verify: wp.array[int]) -> bool:
    """Whether the device prints its buffer warnings: ``verify`` holds a nonzero flag (an empty array never does)."""
    if verify.shape[0] == 0:
        return False
    return verify[0] != 0


@wp.func
def _grid_cell(point: wp.vec3, inv_cell_size: float) -> wp.vec3i:
    """Return the main-level grid cell of a point, clamped as :func:`_static_cell` clamps (NaN gives 0)."""
    return wp.vec3i(
        _static_cell(wp.floor(point[0] * inv_cell_size)),
        _static_cell(wp.floor(point[1] * inv_cell_size)),
        _static_cell(wp.floor(point[2] * inv_cell_size)),
    )


@wp.func
def _grid_offset(n: int) -> wp.vec3i:
    """Return the n-th of the 27 neighbour offsets, x fastest; ``_GRID_OWN_CELL`` is (0, 0, 0)."""
    return wp.vec3i(n % 3 - 1, (n // 3) % 3 - 1, n // 9 - 1)


@wp.func
def _grid_visit(
    a: int,
    world: int,
    lower: wp.vec3,
    upper: wp.vec3,
    group: int,
    target: wp.vec3i,
    world_key: int,
    after_a: bool,
    grid_cell: wp.array[wp.vec3i],
    grid_order: wp.array[int],
    grid_bucket_start: wp.array[int],
    grid_bucket_count: wp.array[int],
    bucket_mask: int,
    aggregate_lower: wp.array[wp.vec3],
    aggregate_upper: wp.array[wp.vec3],
    aggregate_world: wp.array[int],
    aggregate_group: wp.array[int],
    aggregate_pairs: wp.array[wp.vec2i],
    aggregate_pair_count: wp.array[int],
    aggregate_pairs_max: int,
):
    """Pair aggregate ``a`` with the aggregates binned in cell ``target`` under ``world_key`` (world + 1).

    The bucket can hold other cells and world keys; only candidates whose own
    cell and world key match are tested, so each one is seen once per query
    cell. ``after_a`` keeps only candidates b > a (the own cell under the own
    key). A candidate passing the world/group rule and the box overlap is
    appended as ``(min, max)`` with one atomic, past ``aggregate_pairs_max``
    only counted.
    """
    bucket = _static_bucket(world_key, target[0], target[1], target[2], bucket_mask)
    begin = grid_bucket_start[bucket]
    for e in range(begin, begin + grid_bucket_count[bucket]):
        b = grid_order[e]
        if after_a and b <= a:
            continue
        cell = grid_cell[b]
        if cell[0] != target[0] or cell[1] != target[1] or cell[2] != target[2]:
            continue
        if aggregate_world[b] + 1 != world_key:
            continue
        if test_world_and_group_pair(world, aggregate_world[b], group, aggregate_group[b]) and check_aabb_overlap(
            lower, upper, 0.0, aggregate_lower[b], aggregate_upper[b], 0.0
        ):
            write_pair(wp.vec2i(wp.min(a, b), wp.max(a, b)), aggregate_pairs, aggregate_pair_count, aggregate_pairs_max)


@wp.kernel(enable_backward=False, module="unique")
def _grid_bin_kernel(
    aggregate_lower: wp.array[wp.vec3],
    aggregate_upper: wp.array[wp.vec3],
    aggregate_world: wp.array[int],
    cell_size: float,
    inv_cell_size: float,
    bucket_mask: int,
    verify: wp.array[int],
    # outputs
    grid_cell: wp.array[wp.vec3i],
    grid_bucket: wp.array[int],
    grid_order: wp.array[int],
    grid_bucket_count: wp.array[int],
    aggregate_extent_overflow: wp.array[int],
):
    """Bin each union box by the cell of its lower corner, count the buckets and latch boxes too large for the grid.

    The bucket is ``hash(world + 1, cell) & bucket_mask``. The latch is set
    when a box edge exceeds the cell size by more than twice its union pads
    (the pads, and as much again for float32 rounding), or when the box's
    corners are more than one cell apart on an axis (a pad or rounding at a
    cell boundary): either can hide an aggregate pair from the query, whose
    neighbours are one cell away. Under ``verify`` (see :func:`_verifying`)
    the thread that sets the latch prints a warning: once per latch, until
    ``reset_statistics()`` clears it.
    """
    a = wp.tid()
    lower = aggregate_lower[a]
    upper = aggregate_upper[a]
    cell = _grid_cell(lower, inv_cell_size)
    bucket = _static_bucket(aggregate_world[a] + 1, cell[0], cell[1], cell[2], bucket_mask)
    grid_cell[a] = cell
    grid_bucket[a] = bucket
    grid_order[a] = a
    wp.atomic_add(grid_bucket_count, bucket, 1)

    top = _grid_cell(upper, inv_cell_size)
    edge = upper - lower
    reach = wp.max(
        wp.max(wp.max(wp.abs(lower[0]), wp.abs(lower[1])), wp.abs(lower[2])),
        wp.max(wp.max(wp.abs(upper[0]), wp.abs(upper[1])), wp.abs(upper[2])),
    )
    allowance = 4.0 * _UNION_PAD * wp.max(reach, 1.0)
    wide = wp.max(edge[0], wp.max(edge[1], edge[2])) > cell_size + allowance
    spread = top[0] - cell[0] > 1 or top[1] - cell[1] > 1 or top[2] - cell[2] > 1
    # Read first so that a settled latch costs no store.
    if (wide or spread) and aggregate_extent_overflow[0] == 0:
        if _verifying(verify):
            # One atomic per box that finds the latch clear; only the one that sets it prints.
            if wp.atomic_max(aggregate_extent_overflow, 0, 1) == 0:
                wp.printf(
                    "Warning: Aggregate broad phase: the union box of aggregate %d (longest edge %f m) does not fit "
                    "the grid cell (%f m); aggregate pairs may have been missed. "
                    "See CollisionPipeline.broad_phase_statistics().\n",
                    a,
                    wp.max(edge[0], wp.max(edge[1], edge[2])),
                    cell_size,
                )
        else:
            aggregate_extent_overflow[0] = 1


@wp.kernel(enable_backward=False, module="unique")
def _grid_query_kernel(
    aggregate_lower: wp.array[wp.vec3],
    aggregate_upper: wp.array[wp.vec3],
    aggregate_world: wp.array[int],
    aggregate_group: wp.array[int],
    grid_cell: wp.array[wp.vec3i],
    grid_order: wp.array[int],
    grid_bucket_start: wp.array[int],
    grid_bucket_count: wp.array[int],
    bucket_mask: int,
    global_aggregates: int,
    aggregate_pairs_max: int,
    # outputs
    aggregate_pairs: wp.array[wp.vec2i],
    aggregate_pair_count: wp.array[int],
):
    """Query the grid for each aggregate, in bucket order: its half shell under its world key, then world -1's full shell.

    The half shell is the own cell (candidates b > a) and the 13 forward
    offsets, so a pair of one world key is found from exactly one side. World
    -1 aggregates are binned under key 0, which only aggregates of worlds 0 and
    up query with all 27 cells.
    """
    i = wp.tid()
    a = grid_order[i]
    cell = grid_cell[a]
    world = aggregate_world[a]
    lower = aggregate_lower[a]
    upper = aggregate_upper[a]
    group = aggregate_group[a]
    for n in range(_GRID_OWN_CELL, 27):
        _grid_visit(
            a,
            world,
            lower,
            upper,
            group,
            cell + _grid_offset(n),
            world + 1,
            n == _GRID_OWN_CELL,
            grid_cell,
            grid_order,
            grid_bucket_start,
            grid_bucket_count,
            bucket_mask,
            aggregate_lower,
            aggregate_upper,
            aggregate_world,
            aggregate_group,
            aggregate_pairs,
            aggregate_pair_count,
            aggregate_pairs_max,
        )
    if world >= 0 and global_aggregates > 0:
        for n in range(27):
            _grid_visit(
                a,
                world,
                lower,
                upper,
                group,
                cell + _grid_offset(n),
                0,
                False,
                grid_cell,
                grid_order,
                grid_bucket_start,
                grid_bucket_count,
                bucket_mask,
                aggregate_lower,
                aggregate_upper,
                aggregate_world,
                aggregate_group,
                aggregate_pairs,
                aggregate_pair_count,
                aggregate_pairs_max,
            )


@functools.cache
def _create_self_kernel(mask: bool, tile_emission: bool):
    """Create the self-pass kernel: one block per aggregate, member pairs and unbounded shapes."""

    @wp.kernel(enable_backward=False, module="unique")
    def self_kernel(
        aggregate_start: wp.array[int],
        aggregate_shapes: wp.array[int],
        aggregate_world: wp.array[int],
        aggregate_mask: wp.array2d[wp.uint32],
        aggregate_lower: wp.array[wp.vec3],
        aggregate_upper: wp.array[wp.vec3],
        unbounded_start: wp.array[int],
        unbounded_shapes: wp.array[int],
        world_count: int,
        shape_lower: wp.array[wp.vec3],
        shape_upper: wp.array[wp.vec3],
        shape_group: wp.array[int],
        shape_world: wp.array[int],
        shape_body: wp.array[int],
        body_flags: wp.array[int],
        include_static_kinematic_pairs: bool,
        shape_displacement: wp.array[wp.vec3],
        displacement_limit: float,
        filter_pairs: wp.array[wp.vec2i],
        filter_pair_count: int,
        cross_filter_pairs: wp.array[wp.vec2i],
        cross_filter_pair_count: int,
        # outputs
        candidate_pair: wp.array[wp.vec2i],
        pair_demand: wp.array[wp.int64],
    ):
        a, lane = wp.tid()
        stride = wp.block_dim()
        use_mask = bool(False)
        if wp.static(mask):
            use_mask = aggregate_start[a + 1] - aggregate_start[a] <= _MASK_WIDTH
        if wp.static(tile_emission):
            hits = _self_visit(
                a,
                lane,
                stride,
                _COUNT,
                wp.int64(0),
                use_mask,
                aggregate_start,
                aggregate_shapes,
                aggregate_world,
                aggregate_mask,
                aggregate_lower,
                aggregate_upper,
                unbounded_start,
                unbounded_shapes,
                world_count,
                shape_lower,
                shape_upper,
                shape_group,
                shape_world,
                shape_body,
                body_flags,
                include_static_kinematic_pairs,
                shape_displacement,
                displacement_limit,
                filter_pairs,
                filter_pair_count,
                cross_filter_pairs,
                cross_filter_pair_count,
                candidate_pair,
                pair_demand,
            )
            slot = _reserve_block_slots(hits, lane, candidate_pair.shape[0], pair_demand)
            if hits > 0 and slot < wp.int64(candidate_pair.shape[0]):
                _self_visit(
                    a,
                    lane,
                    stride,
                    _WRITE,
                    slot,
                    use_mask,
                    aggregate_start,
                    aggregate_shapes,
                    aggregate_world,
                    aggregate_mask,
                    aggregate_lower,
                    aggregate_upper,
                    unbounded_start,
                    unbounded_shapes,
                    world_count,
                    shape_lower,
                    shape_upper,
                    shape_group,
                    shape_world,
                    shape_body,
                    body_flags,
                    include_static_kinematic_pairs,
                    shape_displacement,
                    displacement_limit,
                    filter_pairs,
                    filter_pair_count,
                    cross_filter_pairs,
                    cross_filter_pair_count,
                    candidate_pair,
                    pair_demand,
                )
        else:
            _self_visit(
                a,
                lane,
                stride,
                _ATOMIC,
                wp.int64(0),
                use_mask,
                aggregate_start,
                aggregate_shapes,
                aggregate_world,
                aggregate_mask,
                aggregate_lower,
                aggregate_upper,
                unbounded_start,
                unbounded_shapes,
                world_count,
                shape_lower,
                shape_upper,
                shape_group,
                shape_world,
                shape_body,
                body_flags,
                include_static_kinematic_pairs,
                shape_displacement,
                displacement_limit,
                filter_pairs,
                filter_pair_count,
                cross_filter_pairs,
                cross_filter_pair_count,
                candidate_pair,
                pair_demand,
            )

    return self_kernel


@functools.cache
def _create_expand_kernel(tile_emission: bool):
    """Create the expansion kernel: a fixed grid of blocks striding over the aggregate pairs."""

    @wp.kernel(enable_backward=False, module="unique")
    def expand_kernel(
        grid: int,
        aggregate_pairs: wp.array[wp.vec2i],
        aggregate_pair_count: wp.array[int],
        aggregate_pairs_max: int,
        aggregate_start: wp.array[int],
        aggregate_shapes: wp.array[int],
        shape_lower: wp.array[wp.vec3],
        shape_upper: wp.array[wp.vec3],
        shape_group: wp.array[int],
        shape_world: wp.array[int],
        shape_body: wp.array[int],
        body_flags: wp.array[int],
        include_static_kinematic_pairs: bool,
        shape_displacement: wp.array[wp.vec3],
        cross_filter_pairs: wp.array[wp.vec2i],
        cross_filter_pair_count: int,
        # outputs
        candidate_pair: wp.array[wp.vec2i],
        pair_demand: wp.array[wp.int64],
    ):
        block, lane = wp.tid()
        stride = wp.block_dim()
        count = wp.min(aggregate_pair_count[0], aggregate_pairs_max)
        p = block
        while p < count:
            pair = aggregate_pairs[p]
            if wp.static(tile_emission):
                hits = _expand_visit(
                    pair,
                    lane,
                    stride,
                    _COUNT,
                    wp.int64(0),
                    aggregate_start,
                    aggregate_shapes,
                    shape_lower,
                    shape_upper,
                    shape_group,
                    shape_world,
                    shape_body,
                    body_flags,
                    include_static_kinematic_pairs,
                    shape_displacement,
                    cross_filter_pairs,
                    cross_filter_pair_count,
                    candidate_pair,
                    pair_demand,
                )
                slot = _reserve_block_slots(hits, lane, candidate_pair.shape[0], pair_demand)
                if hits > 0 and slot < wp.int64(candidate_pair.shape[0]):
                    _expand_visit(
                        pair,
                        lane,
                        stride,
                        _WRITE,
                        slot,
                        aggregate_start,
                        aggregate_shapes,
                        shape_lower,
                        shape_upper,
                        shape_group,
                        shape_world,
                        shape_body,
                        body_flags,
                        include_static_kinematic_pairs,
                        shape_displacement,
                        cross_filter_pairs,
                        cross_filter_pair_count,
                        candidate_pair,
                        pair_demand,
                    )
            else:
                _expand_visit(
                    pair,
                    lane,
                    stride,
                    _ATOMIC,
                    wp.int64(0),
                    aggregate_start,
                    aggregate_shapes,
                    shape_lower,
                    shape_upper,
                    shape_group,
                    shape_world,
                    shape_body,
                    body_flags,
                    include_static_kinematic_pairs,
                    shape_displacement,
                    cross_filter_pairs,
                    cross_filter_pair_count,
                    candidate_pair,
                    pair_demand,
                )
            p += grid

    return expand_kernel


@functools.cache
def _create_static_kernel(tile_emission: bool):
    """Create the static-pass kernel: one block per aggregate tests its members against the grid statics near it."""

    @wp.kernel(enable_backward=False, module="unique")
    def static_kernel(
        aggregate_start: wp.array[int],
        aggregate_shapes: wp.array[int],
        aggregate_world: wp.array[int],
        aggregate_lower: wp.array[wp.vec3],
        aggregate_upper: wp.array[wp.vec3],
        static_start: wp.array[int],
        static_shapes: wp.array[int],
        static_key: wp.array[int],
        static_cell_lower: wp.array[wp.vec3i],
        static_cell_upper: wp.array[wp.vec3i],
        static_entries: wp.array[wp.vec4i],
        static_bucket_start: wp.array[int],
        static_bucket_mask: int,
        inv_cell_size: float,
        shape_lower: wp.array[wp.vec3],
        shape_upper: wp.array[wp.vec3],
        shape_group: wp.array[int],
        shape_world: wp.array[int],
        shape_body: wp.array[int],
        body_flags: wp.array[int],
        include_static_kinematic_pairs: bool,
        shape_displacement: wp.array[wp.vec3],
        displacement_limit: float,
        cross_filter_pairs: wp.array[wp.vec2i],
        cross_filter_pair_count: int,
        # outputs
        candidate_pair: wp.array[wp.vec2i],
        pair_demand: wp.array[wp.int64],
    ):
        a, lane = wp.tid()
        stride = wp.block_dim()
        if wp.static(tile_emission):
            hits = _static_visit(
                a,
                lane,
                stride,
                _COUNT,
                wp.int64(0),
                0,
                aggregate_start,
                aggregate_shapes,
                aggregate_world,
                aggregate_lower,
                aggregate_upper,
                static_start,
                static_shapes,
                static_key,
                static_cell_lower,
                static_cell_upper,
                static_entries,
                static_bucket_start,
                static_bucket_mask,
                inv_cell_size,
                shape_lower,
                shape_upper,
                shape_group,
                shape_world,
                shape_body,
                body_flags,
                include_static_kinematic_pairs,
                shape_displacement,
                displacement_limit,
                cross_filter_pairs,
                cross_filter_pair_count,
                candidate_pair,
                pair_demand,
            )
            slot = _reserve_block_slots(hits, lane, candidate_pair.shape[0], pair_demand)
            if hits > 0 and slot < wp.int64(candidate_pair.shape[0]):
                _static_visit(
                    a,
                    lane,
                    stride,
                    _WRITE,
                    slot,
                    0,
                    aggregate_start,
                    aggregate_shapes,
                    aggregate_world,
                    aggregate_lower,
                    aggregate_upper,
                    static_start,
                    static_shapes,
                    static_key,
                    static_cell_lower,
                    static_cell_upper,
                    static_entries,
                    static_bucket_start,
                    static_bucket_mask,
                    inv_cell_size,
                    shape_lower,
                    shape_upper,
                    shape_group,
                    shape_world,
                    shape_body,
                    body_flags,
                    include_static_kinematic_pairs,
                    shape_displacement,
                    displacement_limit,
                    cross_filter_pairs,
                    cross_filter_pair_count,
                    candidate_pair,
                    pair_demand,
                )
        else:
            _static_visit(
                a,
                lane,
                stride,
                _ATOMIC,
                wp.int64(0),
                0,
                aggregate_start,
                aggregate_shapes,
                aggregate_world,
                aggregate_lower,
                aggregate_upper,
                static_start,
                static_shapes,
                static_key,
                static_cell_lower,
                static_cell_upper,
                static_entries,
                static_bucket_start,
                static_bucket_mask,
                inv_cell_size,
                shape_lower,
                shape_upper,
                shape_group,
                shape_world,
                shape_body,
                body_flags,
                include_static_kinematic_pairs,
                shape_displacement,
                displacement_limit,
                cross_filter_pairs,
                cross_filter_pair_count,
                candidate_pair,
                pair_demand,
            )

    return static_kernel


@functools.cache
def _create_ordered_static_kernel(write: bool):
    """Create an ordered static-pass kernel: one block per aggregate counts its static hits, or writes them in order."""

    @wp.kernel(enable_backward=False, module="unique")
    def ordered_static_kernel(
        item_base: int,
        aggregate_start: wp.array[int],
        aggregate_shapes: wp.array[int],
        aggregate_world: wp.array[int],
        aggregate_lower: wp.array[wp.vec3],
        aggregate_upper: wp.array[wp.vec3],
        static_start: wp.array[int],
        static_shapes: wp.array[int],
        static_key: wp.array[int],
        static_cell_lower: wp.array[wp.vec3i],
        static_cell_upper: wp.array[wp.vec3i],
        static_entries: wp.array[wp.vec4i],
        static_bucket_start: wp.array[int],
        static_bucket_mask: int,
        inv_cell_size: float,
        shape_lower: wp.array[wp.vec3],
        shape_upper: wp.array[wp.vec3],
        shape_group: wp.array[int],
        shape_world: wp.array[int],
        shape_body: wp.array[int],
        body_flags: wp.array[int],
        include_static_kinematic_pairs: bool,
        shape_displacement: wp.array[wp.vec3],
        displacement_limit: float,
        cross_filter_pairs: wp.array[wp.vec2i],
        cross_filter_pair_count: int,
        pair_demand: wp.array[wp.int64],
        item_offset: wp.array[int],
        # outputs: item_count (count pass), candidate_pair (write pass)
        item_count: wp.array[int],
        candidate_pair: wp.array[wp.vec2i],
    ):
        a, lane = wp.tid()
        item = item_base + a
        mode = _COUNT
        base = int(0)
        if wp.static(write):
            mode = _ORDERED_WRITE
            base = item_offset[item]
            # Uniform over the block: an empty item, or one past the buffer, writes nothing.
            if item_count[item] == 0 or base >= candidate_pair.shape[0]:
                return
        hits = _static_visit(
            a,
            lane,
            wp.block_dim(),
            mode,
            wp.int64(0),
            base,
            aggregate_start,
            aggregate_shapes,
            aggregate_world,
            aggregate_lower,
            aggregate_upper,
            static_start,
            static_shapes,
            static_key,
            static_cell_lower,
            static_cell_upper,
            static_entries,
            static_bucket_start,
            static_bucket_mask,
            inv_cell_size,
            shape_lower,
            shape_upper,
            shape_group,
            shape_world,
            shape_body,
            body_flags,
            include_static_kinematic_pairs,
            shape_displacement,
            displacement_limit,
            cross_filter_pairs,
            cross_filter_pair_count,
            candidate_pair,
            pair_demand,
        )
        if wp.static(not write):
            total = wp.tile_extract(wp.tile_sum(wp.tile(hits)), 0)
            if lane == 0:
                item_count[item] = total

    return ordered_static_kernel


@wp.kernel(enable_backward=False, module="unique")
def _static_entry_bucket_kernel(
    entries: wp.array[wp.vec4i],
    static_key: wp.array[int],
    mask: int,
    # outputs
    buckets: wp.array[int],
):
    """Hash each static grid entry ``(g, x, y, z)`` with the static stage's hash, for the host's sort."""
    e = wp.tid()
    entry = entries[e]
    buckets[e] = _static_bucket(static_key[entry[0]], entry[1], entry[2], entry[3], mask)


@wp.func
def _publish_demand(
    demand: wp.int64,
    aggregate_pair_count: wp.array[int],
    aggregate_pairs_max: int,
    capacity: int,
    verify: wp.array[int],
    candidate_pair_count: wp.array[int],
    pair_overflow: wp.array[int],
    pair_demand_max: wp.array[wp.int64],
    aggregate_pair_overflow: wp.array[int],
    aggregate_pair_demand_max: wp.array[int],
):
    """Publish the clamped pair count, set the per-collide overflow flags and latch the demands.

    Under ``verify`` (see :func:`_verifying`) each overflow also prints a
    warning, once per collide: the publish runs on one thread, after either
    main level has counted its aggregate pairs.
    """
    limit = wp.int64(capacity)
    candidate_pair_count[0] = wp.int32(wp.min(demand, limit))
    pair_overflow[0] = 0
    if demand > limit:
        pair_overflow[0] = 1
        if _verifying(verify):
            wp.printf(
                "Warning: Aggregate broad phase candidate pair buffer overflowed %lld > %d; the pairs past it were "
                "dropped. Raise shape_pairs_max.\n",
                demand,
                capacity,
            )
    pair_demand_max[0] = wp.max(pair_demand_max[0], demand)
    count = aggregate_pair_count[0]
    aggregate_pair_overflow[0] = 0
    if count > aggregate_pairs_max:
        aggregate_pair_overflow[0] = 1
        if _verifying(verify):
            wp.printf(
                "Warning: Aggregate broad phase aggregate-pair list overflowed %d > %d; the aggregate pairs past it "
                "were dropped. Raise aggregate_pairs_max.\n",
                count,
                aggregate_pairs_max,
            )
    aggregate_pair_demand_max[0] = wp.max(aggregate_pair_demand_max[0], count)


@wp.kernel(enable_backward=False, module="unique")
def _publish_kernel(
    pair_demand: wp.array[wp.int64],
    aggregate_pair_count: wp.array[int],
    aggregate_pairs_max: int,
    capacity: int,
    verify: wp.array[int],
    # outputs
    candidate_pair_count: wp.array[int],
    pair_overflow: wp.array[int],
    pair_demand_max: wp.array[wp.int64],
    aggregate_pair_overflow: wp.array[int],
    aggregate_pair_demand_max: wp.array[int],
):
    """Publish the clamped pair count, set the per-collide overflow flags and latch the demands."""
    _publish_demand(
        pair_demand[0],
        aggregate_pair_count,
        aggregate_pairs_max,
        capacity,
        verify,
        candidate_pair_count,
        pair_overflow,
        pair_demand_max,
        aggregate_pair_overflow,
        aggregate_pair_demand_max,
    )


@wp.kernel(enable_backward=False, module="unique")
def _order_keys_kernel(
    aggregate_pairs: wp.array[wp.vec2i],
    aggregate_pair_count: wp.array[int],
    aggregate_pairs_max: int,
    # outputs
    aggregate_pair_keys: wp.array[wp.int64],
    aggregate_pair_order: wp.array[int],
):
    """Key each stored aggregate pair ``(a << 32) | b``, a < b, and every unused slot with the sentinel."""
    p = wp.tid()
    key = _ORDER_SENTINEL
    if p < wp.min(aggregate_pair_count[0], aggregate_pairs_max):
        pair = aggregate_pairs[p]
        key = (wp.int64(wp.min(pair[0], pair[1])) << wp.int64(32)) | wp.int64(wp.max(pair[0], pair[1]))
    aggregate_pair_keys[p] = key
    aggregate_pair_order[p] = p


@functools.cache
def _create_ordered_self_kernel(mask: bool, write: bool):
    """Create an ordered self-pass kernel: one block per aggregate counts its hits, or writes them in test order."""

    @wp.kernel(enable_backward=False, module="unique")
    def ordered_self_kernel(
        aggregate_start: wp.array[int],
        aggregate_shapes: wp.array[int],
        aggregate_world: wp.array[int],
        aggregate_mask: wp.array2d[wp.uint32],
        aggregate_lower: wp.array[wp.vec3],
        aggregate_upper: wp.array[wp.vec3],
        unbounded_start: wp.array[int],
        unbounded_shapes: wp.array[int],
        world_count: int,
        shape_lower: wp.array[wp.vec3],
        shape_upper: wp.array[wp.vec3],
        shape_group: wp.array[int],
        shape_world: wp.array[int],
        shape_body: wp.array[int],
        body_flags: wp.array[int],
        include_static_kinematic_pairs: bool,
        shape_displacement: wp.array[wp.vec3],
        displacement_limit: float,
        filter_pairs: wp.array[wp.vec2i],
        filter_pair_count: int,
        cross_filter_pairs: wp.array[wp.vec2i],
        cross_filter_pair_count: int,
        item_offset: wp.array[int],
        # outputs: item_count (count pass), candidate_pair (write pass)
        item_count: wp.array[int],
        candidate_pair: wp.array[wp.vec2i],
    ):
        a, lane = wp.tid()
        use_mask = bool(False)
        if wp.static(mask):
            use_mask = aggregate_start[a + 1] - aggregate_start[a] <= _MASK_WIDTH
        base = int(0)
        if wp.static(write):
            base = item_offset[a]
            # Uniform over the block: an empty item, or one past the buffer, writes nothing.
            if item_count[a] == 0 or base >= candidate_pair.shape[0]:
                return
        hits = _self_visit_ordered(
            a,
            lane,
            wp.block_dim(),
            wp.static(write),
            base,
            use_mask,
            aggregate_start,
            aggregate_shapes,
            aggregate_world,
            aggregate_mask,
            aggregate_lower,
            aggregate_upper,
            unbounded_start,
            unbounded_shapes,
            world_count,
            shape_lower,
            shape_upper,
            shape_group,
            shape_world,
            shape_body,
            body_flags,
            include_static_kinematic_pairs,
            shape_displacement,
            displacement_limit,
            filter_pairs,
            filter_pair_count,
            cross_filter_pairs,
            cross_filter_pair_count,
            candidate_pair,
        )
        if wp.static(not write):
            total = wp.tile_extract(wp.tile_sum(wp.tile(hits)), 0)
            if lane == 0:
                item_count[a] = total

    return ordered_self_kernel


@functools.cache
def _create_ordered_expand_kernel(write: bool):
    """Create an ordered expansion kernel: a fixed grid of blocks strides over the sorted aggregate pairs."""

    @wp.kernel(enable_backward=False, module="unique")
    def ordered_expand_kernel(
        grid: int,
        item_base: int,
        aggregate_pair_keys: wp.array[wp.int64],
        aggregate_pair_count: wp.array[int],
        aggregate_pairs_max: int,
        aggregate_start: wp.array[int],
        aggregate_shapes: wp.array[int],
        shape_lower: wp.array[wp.vec3],
        shape_upper: wp.array[wp.vec3],
        shape_group: wp.array[int],
        shape_world: wp.array[int],
        shape_body: wp.array[int],
        body_flags: wp.array[int],
        include_static_kinematic_pairs: bool,
        shape_displacement: wp.array[wp.vec3],
        cross_filter_pairs: wp.array[wp.vec2i],
        cross_filter_pair_count: int,
        item_offset: wp.array[int],
        # outputs: item_count (count pass), candidate_pair (write pass)
        item_count: wp.array[int],
        candidate_pair: wp.array[wp.vec2i],
    ):
        block, lane = wp.tid()
        stride = wp.block_dim()
        count = wp.min(aggregate_pair_count[0], aggregate_pairs_max)
        p = block
        while p < count:
            item = item_base + p
            key = aggregate_pair_keys[p]
            first = key >> wp.int64(32)
            pair = wp.vec2i(wp.int32(first), wp.int32(key - (first << wp.int64(32))))
            if wp.static(write):
                base = item_offset[item]
                if item_count[item] > 0 and base < candidate_pair.shape[0]:
                    _expand_visit_ordered(
                        pair,
                        lane,
                        stride,
                        True,
                        base,
                        aggregate_start,
                        aggregate_shapes,
                        shape_lower,
                        shape_upper,
                        shape_group,
                        shape_world,
                        shape_body,
                        body_flags,
                        include_static_kinematic_pairs,
                        shape_displacement,
                        cross_filter_pairs,
                        cross_filter_pair_count,
                        candidate_pair,
                    )
            else:
                hits = _expand_visit_ordered(
                    pair,
                    lane,
                    stride,
                    False,
                    0,
                    aggregate_start,
                    aggregate_shapes,
                    shape_lower,
                    shape_upper,
                    shape_group,
                    shape_world,
                    shape_body,
                    body_flags,
                    include_static_kinematic_pairs,
                    shape_displacement,
                    cross_filter_pairs,
                    cross_filter_pair_count,
                    candidate_pair,
                )
                total = wp.tile_extract(wp.tile_sum(wp.tile(hits)), 0)
                if lane == 0:
                    item_count[item] = total
            p += grid
        if wp.static(not write):
            # The unused aggregate-pair slots count nothing.
            q = count + block * stride + lane
            while q < aggregate_pairs_max:
                item_count[item_base + q] = 0
                q += grid * stride

    return ordered_expand_kernel


@wp.kernel(enable_backward=False, module="unique")
def _ordered_publish_kernel(
    item_offset: wp.array[int],
    item_count: wp.array[int],
    aggregate_pair_count: wp.array[int],
    aggregate_pairs_max: int,
    capacity: int,
    verify: wp.array[int],
    # outputs
    candidate_pair_count: wp.array[int],
    pair_demand: wp.array[wp.int64],
    pair_overflow: wp.array[int],
    pair_demand_max: wp.array[wp.int64],
    aggregate_pair_overflow: wp.array[int],
    aggregate_pair_demand_max: wp.array[int],
):
    """Take the demand from the last item's offset and count, then publish as :func:`_publish_kernel` does."""
    last = item_count.shape[0] - 1
    demand = wp.int64(0)
    if last >= 0:
        demand = wp.int64(item_offset[last]) + wp.int64(item_count[last])
    pair_demand[0] = demand
    _publish_demand(
        demand,
        aggregate_pair_count,
        aggregate_pairs_max,
        capacity,
        verify,
        candidate_pair_count,
        pair_overflow,
        pair_demand_max,
        aggregate_pair_overflow,
        aggregate_pair_demand_max,
    )


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------


def _host(array, dtype=None) -> np.ndarray:
    """Return a NumPy copy of a NumPy or Warp array."""
    if isinstance(array, wp.array):
        array = array.numpy()
    return np.asarray(array) if dtype is None else np.asarray(array, dtype=dtype)


def _static_table_size(capacity: int) -> int:
    """Bucket count of the static grid's hash table: the smallest power of two at least ``capacity``."""
    return 1 << max(0, (int(capacity) - 1).bit_length())


def _scan_kwargs(count: int, device) -> dict[str, wp.array]:
    """Preallocated ``array_scan`` storage when Warp takes it, so that a scan captured in a conditional body allocates
    nothing; older Warp versions allocate internally."""
    if not hasattr(wp.utils, "array_scan_temp_storage_size"):
        return {}
    size = wp.utils.array_scan_temp_storage_size(count, wp.int32, device)
    return {"temp_storage": wp.empty(size, dtype=wp.uint8, device=device)}


def _colliding_mask(shape_flags: np.ndarray | None, shape_count: int) -> np.ndarray:
    if shape_flags is None:
        return np.ones(shape_count, dtype=bool)
    return (np.asarray(shape_flags) & int(ShapeFlags.COLLIDE_SHAPES)) != 0


def unbounded_shape_mask(
    shape_type: np.ndarray | wp.array,
    shape_scale: np.ndarray | wp.array,
    shape_body: np.ndarray | wp.array,
    shape_flags: np.ndarray | wp.array | None,
) -> np.ndarray:
    """Return which shapes the aggregate broad phase keeps out of the sweep as unbounded.

    Unbounded shapes are colliding static (body -1) infinite planes (a plane with
    zero x and y scale) and heightfields.

    Args:
        shape_type: Geometry type of each shape, shape [shape_count].
        shape_scale: Scale of each shape, shape [shape_count, 3].
        shape_body: Body of each shape (-1 for static shapes), shape [shape_count].
        shape_flags: Shape flags, shape [shape_count]; ``None`` treats every shape as colliding.

    Returns:
        Boolean mask, shape [shape_count].
    """
    shape_type = _host(shape_type)
    shape_scale = _host(shape_scale).reshape(-1, 3)
    shape_body = _host(shape_body)
    flags = None if shape_flags is None else _host(shape_flags)
    plane = (shape_type == int(GeoType.PLANE)) & (shape_scale[:, 0] == 0.0) & (shape_scale[:, 1] == 0.0)
    heightfield = shape_type == int(GeoType.HFIELD)
    return _colliding_mask(flags, len(shape_type)) & (shape_body < 0) & (plane | heightfield)


def _number_by_lowest_shape(keys: np.ndarray, shapes: np.ndarray) -> np.ndarray:
    """Number groups of ``shapes`` (ascending) by key, in the order of each group's lowest shape."""
    _, inverse = np.unique(keys, return_inverse=True)
    inverse = inverse.reshape(-1)
    lowest = np.full(int(inverse.max()) + 1 if len(inverse) else 0, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(lowest, inverse, shapes)
    rank = np.empty(len(lowest), dtype=np.int32)
    rank[np.argsort(lowest, kind="stable")] = np.arange(len(lowest), dtype=np.int32)
    return rank[inverse]


def _body_articulation(model: Model) -> np.ndarray:
    """Return each body's articulation, -1 for a body no articulated joint moves, int64 [body_count]."""
    body_articulation = np.full(int(model.body_count), -1, dtype=np.int64)
    if int(model.joint_count) > 0 and model.joint_child is not None and model.joint_articulation is not None:
        joint_child = _host(model.joint_child, np.int64)
        joint_articulation = _host(model.joint_articulation, np.int64)
        tree = (joint_articulation >= 0) & (joint_child >= 0)
        body_articulation[joint_child[tree]] = joint_articulation[tree]
    return body_articulation


def infer_shape_aggregates(model: Model, *, max_aggregate_size: int | None = None) -> np.ndarray:
    """Infer the aggregate of each shape from the model's articulations.

    Each articulation's colliding body shapes form one aggregate per world,
    split into chunks of about ``max_aggregate_size`` members (32 by default)
    that never split a body, and each body outside any articulation forms its
    own. Aggregates are numbered by their lowest member shape index. Static
    (body -1), unbounded and non-colliding shapes get -1.

    Args:
        model: The model; reads ``shape_body``, ``shape_world``, ``shape_flags``,
            ``joint_child`` and ``joint_articulation``.
        max_aggregate_size: A positive size splits each aggregate into chunks
            over whole bodies: its members are ordered by their body's lowest
            shape index, then by shape index, and a chunk is closed at the first
            body boundary once it holds that many members. A body's shapes thus
            stay in one chunk even when they are not consecutive in shape order,
            and a chunk can exceed the size by up to one body's shapes less one.
            0 keeps aggregates whole, and ``None`` reads the module default (32).

    Returns:
        Aggregate id of each shape, shape [shape_count], int32.
    """
    size = _KNOBS["max_aggregate_size"] if max_aggregate_size is None else max_aggregate_size
    if int(size) != size or size < 0:
        raise ValueError(f"max_aggregate_size must be a non-negative integer, got {max_aggregate_size!r}")
    size = int(size)

    shape_count = int(model.shape_count)
    aggregate = np.full(shape_count, -1, dtype=np.int32)
    if shape_count == 0:
        return aggregate
    shape_body = _host(model.shape_body, np.int64)
    shape_world = _host(model.shape_world, np.int64)
    members = np.flatnonzero(_colliding_mask(_host(model.shape_flags), shape_count) & (shape_body >= 0))
    if len(members) == 0:
        return aggregate

    body_articulation = _body_articulation(model)

    body = shape_body[members]
    world = shape_world[members]
    articulation = body_articulation[body]
    world_span = int(max(shape_world.max(), -1)) + 2
    articulated_keys = (int(articulation.max()) + 1) * world_span if len(articulation) else 0
    keys = np.where(articulation >= 0, articulation * world_span + world + 1, articulated_keys + body)
    ids = _number_by_lowest_shape(keys, members)

    if size > 0:
        chunk_keys = np.empty(len(members), dtype=np.int64)
        # Whole bodies: each aggregate's members by their body's lowest shape, then by shape.
        body_first = np.full(int(body.max()) + 1, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(body_first, body, members)
        order = np.lexsort((members, body_first[body], ids))
        chunk = -1
        previous_id = -1
        previous_body = -1
        filled = 0
        for position in order:
            if ids[position] != previous_id:
                chunk += 1
                filled = 0
                previous_id = ids[position]
            elif filled >= size and body[position] != previous_body:
                chunk += 1
                filled = 0
            chunk_keys[position] = chunk
            previous_body = body[position]
            filled += 1
        ids = _number_by_lowest_shape(chunk_keys, members)

    aggregate[members] = ids
    return aggregate


def _linear_joint_articulations(model: Model) -> np.ndarray:
    """Return whether each articulation has a non-root joint with a linear degree of freedom.

    Such a joint (prismatic, free, D6 with a linear axis: ``joint_dof_dim[j, 0] > 0``
    with a parent body) lets its child travel arbitrarily far from the root, so
    the articulation has no finite reach bound. Bool [articulation_count].
    """
    split = np.zeros(int(model.articulation_count), dtype=bool)
    if int(model.joint_count) == 0 or model.joint_articulation is None or model.joint_dof_dim is None:
        return split
    parent = _host(model.joint_parent, np.int64)
    articulation = _host(model.joint_articulation, np.int64)
    linear = _host(model.joint_dof_dim, np.int64).reshape(-1, 2)[:, 0] > 0
    split[articulation[(parent >= 0) & (articulation >= 0) & linear]] = True
    return split


def _split_articulations(model: Model, shape_aggregate: np.ndarray, split: np.ndarray) -> tuple[np.ndarray, int]:
    """Give each body of the ``split`` articulations its own aggregate.

    Each such body's colliding shapes form one aggregate (one shape per body
    gives one-shape aggregates); the ids are renumbered by lowest member shape.

    Returns:
        The aggregate of each shape (int32) and the number of aggregates the
        split articulations became.
    """
    members = np.flatnonzero(shape_aggregate >= 0)
    body = _host(model.shape_body, np.int64)[members]
    articulation = np.where(body >= 0, _body_articulation(model)[np.maximum(body, 0)], -1)
    moved = articulation >= 0
    moved[moved] = split[articulation[moved]]
    if not np.any(moved):
        return shape_aggregate, 0
    keys = shape_aggregate[members].astype(np.int64)
    keys[moved] = int(keys.max()) + 1 + body[moved]
    aggregate = np.array(shape_aggregate, dtype=np.int32)
    aggregate[members] = _number_by_lowest_shape(keys, members)
    return aggregate, len(np.unique(body[moved]))


def _common_ancestor(a: np.ndarray, b: np.ndarray, parent: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Return, elementwise, the deepest body that is an ancestor of (or is) both ``a`` and ``b``, or -1 if none."""
    a = np.array(a, dtype=np.int64)
    b = np.array(b, dtype=np.int64)
    valid = (a >= 0) & (b >= 0)
    while True:
        depth_a = np.where(valid, depth[np.maximum(a, 0)], 0)
        depth_b = np.where(valid, depth[np.maximum(b, 0)], 0)
        lift_a = valid & (depth_a > depth_b)
        lift_b = valid & (depth_b > depth_a)
        if not (np.any(lift_a) or np.any(lift_b)):
            break
        a[lift_a] = parent[a[lift_a]]
        b[lift_b] = parent[b[lift_b]]
    while True:
        differ = valid & (a != b)
        if not np.any(differ):
            break
        a[differ] = parent[a[differ]]
        b[differ] = parent[b[differ]]
        valid &= (a >= 0) & (b >= 0)
    return np.where(valid, a, -1)


def _model_shape_bound(model: Model) -> np.ndarray:
    """Return how far each shape's box reaches from its origin on every axis, before margin and gap [m], float64.

    Spheres, capsules and boxes get the larger of their collision radius and
    the bound of the box ``compute_shape_aabbs`` builds from their scale:
    ``|s0|``, ``|s0| + |s1|`` and the norm of ``s`` (by Cauchy-Schwarz on the
    rows of ``|R|``). For shapes the builder authored the two agree; a scale bound
    within ``_RADIUS_ROUNDING`` of the radius (float32 rounding of the same sum)
    keeps the radius, so their reach is unchanged. Primitives that
    :meth:`~newton.ModelBuilder.approximate_meshes` made from a mesh keep the
    mesh's collision radius, which need not bound them, so they get their
    scale's. Every other shape gets the bound ``compute_shape_velocities`` uses
    for a shape about its origin: the larger of its collision radius and the
    norm of the farthest corner of its local box (``shape_collision_aabb_*``,
    scale included). A mesh's or convex hull's collision radius bounds its
    size, not its offset from the origin, and the pipeline places its box by
    the local box, so geometry authored away from the origin needs the corner.
    """
    radius = _host(model.shape_collision_radius, np.float64)
    if len(radius) == 0:
        return radius
    shape_type = _host(model.shape_type, np.int64)
    scale = np.abs(_host(model.shape_scale, np.float64).reshape(-1, 3))
    sphere = shape_type == int(GeoType.SPHERE)
    capsule = shape_type == int(GeoType.CAPSULE)
    box = shape_type == int(GeoType.BOX)
    primitive = sphere | capsule | box
    scale_bound = np.zeros_like(radius)
    scale_bound[sphere] = scale[sphere, 0]
    scale_bound[capsule] = scale[capsule, 0] + scale[capsule, 1]
    scale_bound[box] = np.linalg.norm(scale[box], axis=1)
    bound = radius.copy()
    grown = primitive & (scale_bound > radius * (1.0 + _RADIUS_ROUNDING))
    bound[grown] = scale_bound[grown]
    local_lower = getattr(model, "shape_collision_aabb_lower", None)
    local_upper = getattr(model, "shape_collision_aabb_upper", None)
    if local_lower is not None and local_upper is not None:
        corner = np.maximum(
            np.abs(_host(local_lower, np.float64).reshape(-1, 3)),
            np.abs(_host(local_upper, np.float64).reshape(-1, 3)),
        )
        bound[~primitive] = np.maximum(radius, np.linalg.norm(corner, axis=1))[~primitive]
    return bound


def _model_shape_reach(model: Model, split: np.ndarray, shape_aggregate: np.ndarray, growth: float = 0.0) -> np.ndarray:
    """Return how far each shape reaches from its aggregate's root [m], float32 [shape_count].

    An aggregate's root is the deepest body that is an ancestor of (or is) each
    of its member bodies along the joints of unsplit articulations: the
    articulation's root for a whole articulation, the first body of a chunk
    that is one piece of the tree. Each joint between the root and the shape's
    body adds ``|X_p.p| + |X_c.p|`` (its offsets in parent and child), and the
    shape adds ``|shape_transform.p|`` and its bound, :func:`_model_shape_bound`
    plus margin and gap. Bodies of ``split`` articulations and bodies outside
    articulations are their own roots, and a static shape reaches only its
    bound (its transform is in world space). While the joints keep their
    anchors together, a body's origin lies within the joint offsets along its
    path of its root's origin, so each member's box lies within its reach of
    the root's origin on every axis, and twice an aggregate's largest reach
    bounds every edge of its union box. Members with no common root (a given
    grouping across articulations) reach from their own articulation roots,
    which bounds nothing; the latch reports such a box. ``growth`` [m] is added
    to every body shape's reach for the speculative growth of its box and
    sweep (see :meth:`BroadPhaseAggregate.from_model`); static shapes neither
    grow nor move.
    """
    shape_body = _host(model.shape_body, np.int64)
    reach = _model_shape_bound(model) + _host(model.shape_margin, np.float64) + _host(model.shape_gap, np.float64)
    body_count = int(model.body_count)
    length = np.zeros(body_count, dtype=np.float64)
    depth = np.zeros(body_count, dtype=np.int64)
    parent_body = np.full(body_count, -1, dtype=np.int64)
    if int(model.joint_count) > 0 and model.joint_articulation is not None:
        parent = _host(model.joint_parent, np.int64)
        child = _host(model.joint_child, np.int64)
        articulation = _host(model.joint_articulation, np.int64)
        offset_parent = np.linalg.norm(_host(model.joint_X_p, np.float64).reshape(-1, 7)[:, :3], axis=1)
        offset_child = np.linalg.norm(_host(model.joint_X_c, np.float64).reshape(-1, 7)[:, :3], axis=1)
        chained = (parent >= 0) & (child >= 0) & (articulation >= 0)
        chained[chained] = ~split[articulation[chained]]
        parent, child = parent[chained], child[chained]
        segment = (offset_parent + offset_child)[chained]
        parent_body[child] = parent
        for _ in range(body_count + 1):  # one pass per tree level
            updated = length.copy()
            updated[child] = length[parent] + segment
            updated_depth = depth.copy()
            updated_depth[child] = depth[parent] + 1
            if np.array_equal(updated, length) and np.array_equal(updated_depth, depth):
                break
            length = updated
            depth = updated_depth
    on_body = shape_body >= 0
    shape_offset = np.linalg.norm(_host(model.shape_transform, np.float64).reshape(-1, 7)[:, :3], axis=1)
    reach[on_body] += shape_offset[on_body] + length[shape_body[on_body]] + growth

    # Each aggregate's root, folded over its member bodies one member at a time.
    members = np.flatnonzero(on_body & (shape_aggregate >= 0))
    if len(members):
        ids = shape_aggregate[members].astype(np.int64)
        order = np.argsort(ids, kind="stable")
        ids, bodies = ids[order], shape_body[members][order]
        first = np.r_[True, ids[1:] != ids[:-1]]
        rank = np.arange(len(ids)) - np.maximum.accumulate(np.where(first, np.arange(len(ids)), 0))
        root = np.full(int(ids.max()) + 1, -1, dtype=np.int64)
        root[ids[first]] = bodies[first]
        for k in range(1, int(rank.max()) + 1):
            at = rank == k
            root[ids[at]] = _common_ancestor(root[ids[at]], bodies[at], parent_body, depth)
        root_length = np.where(root >= 0, length[np.maximum(root, 0)], 0.0)
        reach[members[order]] -= root_length[ids]
    return reach.astype(np.float32)


def _model_shape_position(model: Model) -> np.ndarray:
    """Return each shape's world position in the model's initial state (``body_q``) [m], float64 [shape_count, 3]."""
    shape_body = _host(model.shape_body, np.int64)
    position = _host(model.shape_transform, np.float64).reshape(-1, 7)[:, :3].copy()
    on_body = shape_body >= 0
    if np.any(on_body):
        body_q = _host(model.body_q, np.float64).reshape(-1, 7)[shape_body[on_body]]
        # Rotate by the body's quaternion (x, y, z, w), then translate.
        axis, w = body_q[:, 3:6], body_q[:, 6:7]
        local = position[on_body]
        twice_cross = 2.0 * np.cross(axis, local)
        position[on_body] = body_q[:, :3] + local + w * twice_cross + np.cross(axis, twice_cross)
    return position


def _model_shape_extent(model: Model) -> float:
    """Return the largest |coordinate| of any shape's world position in the model's initial state [m]."""
    if int(model.shape_count) == 0:
        return 0.0
    return float(np.abs(_model_shape_position(model)).max())


def _model_shape_box(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Return each shape's box in the model's initial state [m], float64 lower and upper [shape_count, 3].

    Spheres, capsules, boxes, and meshes, convex hulls and heightfields (by
    their rotated local box, ``shape_collision_aabb_*``) get the box that
    ``compute_shape_aabbs`` gives them in :meth:`CollisionPipeline.collide`,
    computed in float64, so the two agree to float32 rounding (a corner on a
    cell boundary can bin to the neighbouring cell). Other shapes get an
    approximation, the cube of their :func:`_model_shape_bound` about their
    origin. Every box grows by the shape's margin and gap, as there.
    """
    position = _model_shape_position(model)
    shape_body = _host(model.shape_body, np.int64)
    shape_type = _host(model.shape_type, np.int64)
    scale = _host(model.shape_scale, np.float64).reshape(-1, 3)
    half = np.repeat(_model_shape_bound(model)[:, None], 3, axis=1)
    sphere = shape_type == int(GeoType.SPHERE)
    half[sphere] = scale[sphere, :1]
    local_box = (
        (shape_type == int(GeoType.MESH))
        | (shape_type == int(GeoType.CONVEX_MESH))
        | (shape_type == int(GeoType.HFIELD))
    )
    if getattr(model, "shape_collision_aabb_lower", None) is None or model.shape_collision_aabb_upper is None:
        local_box[:] = False
    oriented = np.flatnonzero((shape_type == int(GeoType.CAPSULE)) | (shape_type == int(GeoType.BOX)) | local_box)
    if len(oriented):
        # The shape's world rotation q_body * q_shape (x, y, z, w), then its matrix's columns: its axes in world space.
        sx, sy, sz, sw = _host(model.shape_transform, np.float64).reshape(-1, 7)[oriented, 3:].T
        body = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(oriented), 1))
        on_body = shape_body[oriented] >= 0
        body[on_body] = _host(model.body_q, np.float64).reshape(-1, 7)[shape_body[oriented][on_body], 3:]
        bx, by, bz, bw = body.T
        x = bw * sx + sw * bx + by * sz - bz * sy
        y = bw * sy + sw * by + bz * sx - bx * sz
        z = bw * sz + sw * bz + bx * sy - by * sx
        w = bw * sw - bx * sx - by * sy - bz * sz
        columns = np.stack(
            (
                np.stack((1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)), axis=1),
                np.stack((2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w)), axis=1),
                np.stack((2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)), axis=1),
            ),
            axis=2,
        )
        capsule = shape_type[oriented] == int(GeoType.CAPSULE)
        rows = oriented[capsule]
        # A capsule: radius scale[0] around its local z axis, half height scale[1].
        half[rows] = scale[rows, :1] + np.abs(columns[capsule, :, 2]) * scale[rows, 1:2]
        box = shape_type[oriented] == int(GeoType.BOX)
        rows = oriented[box]
        half[rows] = np.einsum("nij,nj->ni", np.abs(columns[box]), scale[rows])
        local = local_box[oriented]
        if np.any(local):
            # A local box (scale baked in): its centre rotated off the origin, its half extents through |R|.
            rows = oriented[local]
            local_lower = _host(model.shape_collision_aabb_lower, np.float64).reshape(-1, 3)[rows]
            local_upper = _host(model.shape_collision_aabb_upper, np.float64).reshape(-1, 3)[rows]
            position[rows] += np.einsum("nij,nj->ni", columns[local], 0.5 * (local_lower + local_upper))
            half[rows] = np.einsum("nij,nj->ni", np.abs(columns[local]), 0.5 * (local_upper - local_lower))
    half += (_host(model.shape_margin, np.float64) + _host(model.shape_gap, np.float64))[:, None]
    return position - half, position + half


def _aggregate_worlds(aggregate_world: np.ndarray) -> int:
    """Return how many distinct worlds >= 0 hold aggregates; world -1 is not counted."""
    return len(np.unique(aggregate_world[aggregate_world >= 0]))


def _grid_occupancy(lower: np.ndarray, upper: np.ndarray, world: np.ndarray, cell_size: float) -> float:
    """Return the mean over aggregates of how many aggregates share its grid cell, itself included.

    ``lower`` and ``upper`` are the union boxes before their pad, float32 [A, 3].
    Each is padded as ``_store_union`` pads it and binned as
    ``_grid_bin_kernel`` bins it: by the cell of its padded lower corner, in
    float32, under the world key ``world + 1``. Aggregates are counted per
    exact (world key, cell), not per hash bucket.
    """
    if len(lower) == 0:
        return 0.0
    reach = np.maximum(np.abs(lower).max(axis=1), np.abs(upper).max(axis=1))
    corner = lower - (np.float32(_UNION_PAD) * np.maximum(reach, np.float32(1.0)))[:, None]
    cell = np.floor(corner * (np.float32(1.0) / np.float32(cell_size)))
    limit = float(_STATIC_CELL_LIMIT)
    cell = np.clip(np.nan_to_num(cell, nan=0.0), -limit, limit).astype(np.int64)
    keys = np.column_stack((world.astype(np.int64) + 1, cell))
    _, counts = np.unique(keys, axis=0, return_counts=True)
    # Each aggregate of a key holding n counts n, so the mean is sum(n^2) / A.
    return float(np.sum(counts.astype(np.float64) ** 2) / len(lower))


@dataclass
class _LaunchArgs:
    """The arrays of the last :meth:`BroadPhaseAggregate.launch`, replayed by ``_launch_stage``."""

    shape_lower: wp.array
    shape_upper: wp.array
    shape_group: wp.array
    shape_world: wp.array
    body_flags: wp.array
    include_static_kinematic_pairs: bool
    shape_displacement: wp.array
    displacement_limit: float
    candidate_pair: wp.array
    candidate_pair_count: wp.array
    device: wp.Device


class BroadPhaseAggregate:
    """Two-level broad phase over aggregates of shapes.

    .. experimental::

    The colliding body shapes are grouped into aggregates, one per articulation
    and world, split into chunks of about ``max_aggregate_size`` members
    over whole bodies (and, under the grid, per body for articulations with linear
    joints; see :meth:`from_model`), and every collide runs up to six stages,
    all with launch sizes fixed at construction and counts kept on the device,
    so :meth:`launch` can be captured in a CUDA graph:

    1. ``"union"``: one box per aggregate covering its member boxes, padded
       outward by about 1e-6 of its largest coordinate (at least 1e-6 m) so that
       boxes which only touch still pair at the main level, and a group
       summary: the members' group when they share one, otherwise the unique
       negative id ``_MIXED_GROUP_BASE - a``. Summaries are recomputed every
       collide, so runtime group changes are honoured.
    2. ``"main"``: the overlapping aggregate pairs, under the aggregate worlds
       and group summaries, into ``aggregate_pairs``: with ``main="grid"`` (the
       default, given a reach and where the rules of :meth:`from_model` keep
       it) a uniform grid of the union boxes (below), with ``main="sap"``
       :class:`~newton.geometry.BroadPhaseSAP` over them.
    3. ``"self"``: one block per aggregate tests its member pairs and its
       members against the unbounded shapes (infinite planes, heightfields and
       very large statics) of its world and of world -1.
    4. ``"static"`` (with finite static shapes): one block per aggregate tests
       its members against the finite statics of the static grid near it.
    5. ``"expand"``: a fixed grid of blocks tests the member pairs of each
       overlapping aggregate pair.
    6. ``"publish"``: ``candidate_pair_count[0] = min(pair_demand, capacity)``.

    Member pairs follow the rules of :class:`~newton.geometry.BroadPhaseAllPairs`: worlds and
    collision groups, same-body and filter-pair exclusions, the immovable-pair
    filter and the (swept) box overlap. The pair set is deterministic; the slot
    order follows the order of the atomics.

    Each pair reserves its slot with one 64-bit atomic on ``pair_demand`` (each
    block reserves its slots with one under ``_KNOBS["emission"] = "tile"``).
    Pairs past the buffer are dropped, ``pair_overflow`` is set for the collide
    and ``pair_demand_max`` latches the true need. Aggregate pairs past
    ``aggregate_pairs_max`` are dropped the same way and reported by
    ``aggregate_pair_overflow`` and ``aggregate_pair_demand_max``. With
    ``verify`` (which :class:`~newton.CollisionPipeline` sets from its
    ``verify_buffers``) the device also prints a warning for each, and when
    the grid's latch below is set, with no host read.

    With ``main="grid"`` the cell size h (``cell_size``) is
    ``2 R (1 + 2**-8) + 2 p``: R, the largest ``shape_reach`` over the members
    (in :meth:`from_model`, speculative growth included), bounds every
    union-box edge by 2R, and p = ``1e-6 * max(1 m, C)`` is the
    union pad at C = ``shape_extent + R``, the scene's initial extent (the
    largest |coordinate| of a shape's world position, which :meth:`from_model`
    reads from the model's initial state). So a box exactly 2R wide, with its
    pad and float32 rounding out to that extent, never spans three cells;
    bodies that travel far beyond it can still fire the latch, which
    :meth:`statistics` reports and
    :meth:`CollisionPipeline.broad_phase_statistics` warns about once (again
    after :meth:`reset_statistics`). The
    longest aggregate's reach sets the cell for every aggregate, so one long
    chunk among small aggregates makes the cells crowded: :meth:`from_model`
    measures each chunk of a long articulation from the chunk's own root, and
    by default falls back to the SAP when the cell would exceed
    ``_KNOBS["grid_cell_ratio_max"]`` times the median aggregate extent.

    By default :meth:`from_model` keeps the grid only where it should pay
    (measured on an L40 at each scene's initial pose). In fewer than
    ``_KNOBS["grid_min_worlds"]`` (2,048) worlds holding aggregates, it builds
    the SAP below ``_KNOBS["grid_min_aggregates"]`` (32,768) aggregates, where
    the grid's fixed main stage costs more than the sweep. In any number of
    worlds it builds the SAP when the cell guard trips, or when the mean over
    aggregates of how many aggregates share its cell (under its world's key,
    by the lower corner at the initial pose, itself included) exceeds
    ``_KNOBS["grid_occupancy_max"]`` (90), since each query then scans crowded
    cells. The thresholds are judgments bracketed by measured scenes, not
    measured optima (see ``_KNOBS``).

    Each collide bins every union box by the cell of its lower corner into
    bucket ``hash(world + 1, cell) & (T - 1)`` of T buckets (the next power of
    two at least 2A), counts the buckets, radix sorts the aggregates by bucket
    (A keys of log2(T) bits) and scans the counts into bucket starts. Each
    aggregate then tests, under its world's key, the aggregates of its own cell
    with a larger id and those of the 13 forward neighbour cells, and an
    aggregate of world 0 and up also those of all 27 cells under world -1's key
    0. Every candidate's cell and world are checked, so each pair is found
    once. A union box with an edge over h, or (by its pad or float32 rounding
    at a cell boundary) more than one cell between its corners, can hide
    pairs, and ``aggregate_extent_overflow`` latches it. :meth:`from_model`
    splits articulations with linear joints, which have no reach bound, into
    one aggregate per body. A given ``shape_aggregate`` that keeps such an
    articulation whole, or an articulation with several root joints, can get a
    cell too small for its union box; the latch reports it.

    With ``ordered=True`` the list itself is deterministic, provided
    ``aggregate_pairs_max`` is not exceeded (past it, which aggregate pairs are
    kept depends on the main level's arrival order). The work items are the
    aggregates 0..A-1, then (with finite statics in the static grid) each
    aggregate's static pairs, then the stored aggregate pairs sorted on
    ``(a << 32) | b`` (a < b), and the list holds, in this order:

    - for each aggregate, its member pairs (i < j) in member order, then each
      of its unbounded shapes in CSR order (world -1 first, then by world)
      against its members in member order;
    - for each aggregate, its static pairs (see below);
    - for each aggregate pair, the (member of a) x (member of b) pairs, rows
      over a's members;

    with the hits of each item in test order. Members are in shape order
    within their aggregate. The pair set and ``pair_demand`` are those of the
    unordered emission. The stages change to: "main" also keys the stored
    aggregate pairs (unused slots get a sentinel above every key) and radix
    sorts all ``aggregate_pairs_max`` of them, so the launch stays fixed;
    "self", "static" and "expand" count each item's hits into ``item_count``;
    a "write" stage scans the counts exclusively into ``item_offset`` and
    writes every item's hits from its offset, round by round in test order;
    "publish" takes ``pair_demand = item_offset[-1] + item_count[-1]``. On
    overflow the list is the uncapped list's first ``capacity`` entries.
    Construction raises :class:`ValueError` when the int32 offsets could
    overflow (see ``ordered`` in :meth:`__init__`).

    The static items are A..2A-1, by aggregate. Each holds the statics in the
    static stage's visiting order, each against the members in member order:
    when the aggregate walks the grid, by the cell where the pair is tested
    (z, then y, then x), world -1's statics before the aggregate's world's,
    then by shape; when it lists them, world -1's before its world's, by
    shape.

    Finite static shapes (colliding body -1 shapes other than infinite planes
    and heightfields) live in a static grid that :meth:`build_static` bins on
    the host, outside the collide step; :meth:`launch` raises
    :class:`RuntimeError` until the first build, which
    :meth:`CollisionPipeline.refresh_static_shapes` makes from the model. The
    grid hashes ``(world + 1, cell)`` keys into a power-of-two table over
    entries sorted by bucket. A static whose box has an edge over
    ``STATIC_UNBOUNDED_CELLS`` cells joins its world's unbounded list instead.
    The "static" stage tests a static once per aggregate, with the member-pair
    rules and the current boxes, but finds it through the cells it was binned
    into: a static moved since the last build is seen at its new place only
    after the next one. World -1 aggregates beside finite statics of worlds 0
    and up raise :class:`NotImplementedError`. ``_KNOBS["statics"] =
    "aggregates"`` instead makes each finite static a one-shape aggregate of
    the main level, which needs no build and follows moving statics.

    With speculative contacts, each member box is swept by its displacement
    clamped per axis to ``sort_axis_displacement_limit`` before the union. A
    member pair farther apart than the clamp allows can be missed. This differs
    from :class:`~newton.geometry.BroadPhaseSAP`, which clamps only the displacement projected on
    its sort axis, so the two modes can miss different pairs.

    Membership, worlds, bodies and filter pairs are fixed at construction.
    """

    def __init__(
        self,
        shape_world: wp.array[wp.int32] | np.ndarray,
        shape_body: wp.array[wp.int32] | np.ndarray,
        shape_aggregate: wp.array[wp.int32] | np.ndarray,
        shape_type: wp.array[wp.int32] | np.ndarray,
        shape_scale: wp.array[wp.vec3] | np.ndarray,
        shape_flags: wp.array[wp.int32] | np.ndarray | None = None,
        filter_pairs: np.ndarray | None = None,
        *,
        main: str | None = None,
        shape_reach: np.ndarray | None = None,
        shape_extent: float | None = None,
        aggregate_pairs_max: int | None = None,
        ordered: bool | None = None,
        static_cell_size: float | None = None,
        verify: bool = False,
        device: Devicelike | None = None,
    ) -> None:
        """Build the aggregates and allocate the per-collide buffers.

        Args:
            shape_world: World index of each shape (-1 for shapes shared by every world).
            shape_body: Body of each shape (-1 for static shapes).
            shape_aggregate: Aggregate of each shape, as :meth:`from_model` infers it. The colliding body shapes must carry the ids
                0..aggregate_count-1, each aggregate in one world and each body's
                colliding shapes in one aggregate; entries of other shapes are ignored.
            shape_type: Geometry type of each shape.
            shape_scale: Scale of each shape, shape [shape_count, 3].
            shape_flags: Shape flags; only shapes with ``COLLIDE_SHAPES`` take part.
                ``None`` treats every shape as colliding.
            filter_pairs: Excluded shape pairs, canonical (i < j) and sorted
                lexicographically, int32 shape [pair_count, 2], as
                :meth:`~newton.Model.shape_collision_filter_pairs_array` returns them.
            main: Main-level algorithm, ``"grid"`` or ``"sap"`` (see the class). ``None``
                reads the module default (``"grid"``), except that without
                ``shape_reach`` it uses ``"sap"``: the grid needs a reach;
                :meth:`from_model` computes one; a direct construction without one
                gets the SAP. An explicit ``"grid"`` without ``shape_reach`` raises.
                The rules by which :meth:`from_model` picks the default main level
                do not apply to a direct construction.
            shape_reach: How far each shape reaches from a point shared by its
                aggregate on every axis [m], float32 [shape_count], as
                :meth:`from_model` computes it from the joint and shape offsets and
                bounds. ``main="grid"`` needs it and sizes its cell from the largest
                value over the members (see the class); other main levels ignore it.
                It must cover the boxes :meth:`launch` is given, speculative growth
                and sweep included: under a speculative cap c, 2c per body shape.
            shape_extent: How far from the origin the shapes start [m]: the largest
                |coordinate| of a shape's world position, as :meth:`from_model` reads
                it from the model's initial state. The grid's cell leaves room for
                the union pad out to this extent. ``None`` counts 0. Only
                ``main="grid"`` uses it.
            aggregate_pairs_max: Capacity of the aggregate-pair list. ``None`` gives
                ``AGGREGATE_PAIRS_PER_AGGREGATE`` per aggregate.
            ordered: Write the pairs in the deterministic order described on the
                class, through a count pass, a scan and a write pass. ``None`` reads
                the module default (``False``: unordered emission). The item offsets
                are int32, so the worst case, sum over aggregates of
                ``n(n-1)/2 + n*U`` (n members, U unbounded and finite static shapes
                in view) plus ``aggregate_pairs_max * largest_aggregate**2`` when
                there are two aggregates or more, must stay below 2**31.
            static_cell_size: Cell size of the static grid [m]. ``None`` takes the
                longest union-box edge at the first :meth:`build_static`, at least
                0.5 m. A given value is used as is: a small one multiplies the
                entries, up to about 66**3 per static below the
                ``STATIC_UNBOUNDED_CELLS`` edge, and so the grid's memory and the
                host sort of each build.
            verify: Have the device print a warning, without a host read, when a
                collide drops candidate pairs past the buffer or aggregate pairs
                past ``aggregate_pairs_max`` (once per collide each), and when the
                grid's ``aggregate_extent_overflow`` latch is set (once, until
                :meth:`reset_statistics`). The flag lives in ``verify_flag``, an
                int32 device array of one entry allocated only when ``verify`` is
                set (``None`` otherwise); zeroing it silences the warnings, in
                captured graphs too. :class:`~newton.CollisionPipeline` passes its
                ``verify_buffers``.
            device: Device for the arrays. If None, the device of the first Warp
                array argument, else the CPU.

        Raises:
            ValueError: If the arrays disagree in length, an aggregate spans worlds,
                one body's colliding shapes fall in two aggregates, a colliding body
                shape has no aggregate, the ids are not exactly 0..aggregate_count-1,
                the filter pairs are not canonical and sorted, ``static_cell_size``
                is not positive, an ordered list could pass the int32 offsets, or
                ``main="grid"`` lacks ``shape_reach``, or a member's reach is not
                finite and non-negative, or none is positive, or ``shape_extent`` is
                not finite and non-negative.
            NotImplementedError: If the static grid would hold finite static
                shapes of worlds 0 and up while an aggregate is in world -1.
        """
        if device is None:
            for array in (shape_world, shape_body, shape_aggregate, shape_type, shape_scale, shape_flags):
                if isinstance(array, wp.array):
                    device = array.device
                    break
            else:
                device = "cpu"
        self.device = wp.get_device(device)

        knobs = dict(_KNOBS)
        self.main_fallback = ""
        if main is not None:
            knobs["main"] = main
        elif knobs["main"] == "grid" and shape_reach is None:
            # The grid needs a reach; from_model computes one, and a direct construction without one gets the SAP.
            knobs["main"] = "sap"
            self.main_fallback = "reach"
        if ordered is not None:
            knobs["ordered"] = ordered
        for key, values in _KNOB_VALUES.items():
            if knobs[key] not in values:
                raise ValueError(f"Unsupported aggregate broad phase {key} {knobs[key]!r}; expected one of {values}")
        self.main = knobs["main"]
        self.ordered = bool(knobs["ordered"])
        if self.main == "grid" and shape_reach is None:
            raise ValueError(
                "BroadPhaseAggregate(main='grid') needs shape_reach; BroadPhaseAggregate.from_model computes it"
            )
        if shape_extent is not None and not (np.isfinite(shape_extent) and shape_extent >= 0.0):
            raise ValueError(f"shape_extent must be a non-negative finite number or None, got {shape_extent!r}")
        if static_cell_size is not None and not (np.isfinite(static_cell_size) and static_cell_size > 0.0):
            raise ValueError(f"static_cell_size must be a positive finite number or None, got {static_cell_size!r}")

        world = _host(shape_world, np.int64)
        body = _host(shape_body, np.int64)
        aggregate = _host(shape_aggregate, np.int64)
        shape_type_np = _host(shape_type, np.int64)
        scale = _host(shape_scale, np.float32).reshape(-1, 3)
        flags = None if shape_flags is None else _host(shape_flags, np.int64)
        reach = None if shape_reach is None else _host(shape_reach, np.float64).reshape(-1)
        shape_count = len(world)
        lengths = [len(body), len(aggregate), len(shape_type_np), len(scale)]
        if flags is not None:
            lengths.append(len(flags))
        if reach is not None:
            lengths.append(len(reach))
        if any(length != shape_count for length in lengths):
            raise ValueError(f"Per-shape arrays must all have {shape_count} entries, got {lengths}")
        if shape_count and world.min() < -1:
            raise ValueError("World indices must be -1 or non-negative")
        self.shape_count = shape_count

        colliding = _colliding_mask(flags, shape_count)
        unbounded = unbounded_shape_mask(shape_type_np, scale, body, flags)
        finite_static_mask = colliding & (body < 0) & ~unbounded
        finite_statics = np.flatnonzero(finite_static_mask)
        static_aggregates = knobs["statics"] == "aggregates"

        # Members: colliding body shapes, in shape-index order.
        members = np.flatnonzero(colliding & (body >= 0))
        member_ids = aggregate[members]
        if np.any(member_ids < 0):
            raise ValueError(
                f"{int(np.count_nonzero(member_ids < 0))} colliding body shapes have no aggregate "
                f"(first: shape {int(members[np.argmax(member_ids < 0)])})"
            )
        if static_aggregates and len(finite_statics):
            # Each finite static shape becomes a one-shape aggregate after the given ones.
            aggregate = aggregate.copy()
            aggregate[finite_statics] = (int(member_ids.max()) + 1 if len(members) else 0) + np.arange(
                len(finite_statics)
            )
            members = np.flatnonzero(colliding & ((body >= 0) | finite_static_mask))
            member_ids = aggregate[members]
        aggregate_count = int(member_ids.max()) + 1 if len(members) else 0
        sizes = np.bincount(member_ids, minlength=aggregate_count) if len(members) else np.zeros(0, dtype=np.int64)
        if np.any(sizes == 0):
            raise ValueError(
                f"Aggregate ids must be exactly 0..{aggregate_count - 1}; "
                f"{int(np.count_nonzero(sizes == 0))} ids in that range have no member"
            )
        member_worlds = world[members]
        world_low = np.full(aggregate_count, np.iinfo(np.int64).max, dtype=np.int64)
        world_high = np.full(aggregate_count, np.iinfo(np.int64).min, dtype=np.int64)
        np.minimum.at(world_low, member_ids, member_worlds)
        np.maximum.at(world_high, member_ids, member_worlds)
        if np.any(world_low != world_high):
            raise ValueError(f"Aggregate {int(np.argmax(world_low != world_high))} spans worlds")
        member_bodies = body[members]
        on_body = member_bodies >= 0
        body_span = int(member_bodies.max()) + 1 if np.any(on_body) else 0
        body_low = np.full(body_span, np.iinfo(np.int64).max, dtype=np.int64)
        body_high = np.full(body_span, np.iinfo(np.int64).min, dtype=np.int64)
        np.minimum.at(body_low, member_bodies[on_body], member_ids[on_body])
        np.maximum.at(body_high, member_bodies[on_body], member_ids[on_body])
        split_bodies = (body_low != body_high) & (body_high >= 0)
        if np.any(split_bodies):
            raise ValueError(f"The colliding shapes of body {int(np.argmax(split_bodies))} fall in two aggregates")

        # Aggregate CSR, each aggregate's members in shape-index order.
        order = np.argsort(member_ids, kind="stable")
        aggregate_shapes_np = members[order].astype(np.int32)
        aggregate_start_np = np.zeros(aggregate_count + 1, dtype=np.int32)
        np.cumsum(sizes, out=aggregate_start_np[1:])
        aggregate_world_np = world_low.astype(np.int32)
        self.aggregate_count = aggregate_count
        self.largest_aggregate = int(sizes.max()) if aggregate_count else 0
        self.aggregate_worlds = _aggregate_worlds(aggregate_world_np)
        self._aggregate_world_np = aggregate_world_np
        # The occupancy from_model's rules computed, else 0.
        self.grid_occupancy = 0.0
        # The grid's cell: twice the largest member reach, with room for the union pad and rounding.
        self.cell_size: float | None = None
        if self.main == "grid" and len(members):
            member_reach = reach[members]
            if not np.all(np.isfinite(member_reach)) or member_reach.min() < 0.0 or member_reach.max() <= 0.0:
                raise ValueError(
                    "BroadPhaseAggregate(main='grid') needs a shape_reach that is finite and non-negative for every "
                    f"aggregate member and positive for one; got {member_reach.min():g}..{member_reach.max():g}"
                )
            reach_max = float(member_reach.max())
            pad = _UNION_PAD * max(1.0, (0.0 if shape_extent is None else float(shape_extent)) + reach_max)
            self.cell_size = float(np.float32(2.0 * reach_max * (1.0 + _GRID_CELL_SLACK) + 2.0 * pad))
        self._aggregate_start_np = aggregate_start_np
        self._aggregate_shapes_np = aggregate_shapes_np

        # Finite statics: the static grid, unless they are aggregates. A world -1
        # aggregate would have to look up every world's cells.
        grid_statics = finite_statics if not static_aggregates else finite_statics[:0]
        if len(grid_statics):
            world_statics = int(np.count_nonzero(world[grid_statics] >= 0))
            global_aggregates = int(np.count_nonzero(aggregate_world_np < 0))
            if world_statics and global_aggregates:
                raise NotImplementedError(
                    "BroadPhaseAggregate's static grid does not support world -1 aggregates beside finite static "
                    f"shapes of worlds 0 and up ({global_aggregates} world -1 aggregates, {world_statics} finite static "
                    "shapes in worlds >= 0); set broad_phase_aggregate._KNOBS['statics'] = 'aggregates' to sweep "
                    "the statics as one-shape aggregates instead"
                )

        member_aggregate = np.full(shape_count, -1, dtype=np.int64)
        member_aggregate[members] = member_ids
        member_local = np.full(shape_count, -1, dtype=np.int64)
        member_local[aggregate_shapes_np] = np.arange(len(members)) - np.repeat(aggregate_start_np[:-1], sizes)

        # Filter pairs.
        pairs = np.zeros((0, 2), dtype=np.int64) if filter_pairs is None else _host(filter_pairs, np.int64)
        pairs = pairs.reshape(-1, 2)
        if len(pairs):
            if pairs.min() < 0 or pairs.max() >= shape_count or np.any(pairs[:, 0] >= pairs[:, 1]):
                raise ValueError("filter_pairs must hold canonical (i < j) shape index pairs")
            keys = pairs[:, 0] * shape_count + pairs[:, 1]
            if np.any(np.diff(keys) <= 0):
                raise ValueError("filter_pairs must be sorted lexicographically without duplicates")
        self._filter_pair_count = len(pairs)
        pair_aggregate = member_aggregate[pairs]
        within = (pair_aggregate[:, 0] >= 0) & (pair_aggregate[:, 0] == pair_aggregate[:, 1])
        relevant = (member_aggregate >= 0) | unbounded | finite_static_mask
        cross = pairs[~within & relevant[pairs[:, 0]] & relevant[pairs[:, 1]]]

        # Static exclusion masks of aggregates of at most 32 members: same body or filter pair.
        mask_np = np.zeros((aggregate_count, 32), dtype=np.uint32)
        local_pairs = [pairs[within]]
        body_members = aggregate_shapes_np[body[aggregate_shapes_np] >= 0]
        by_body = body_members[np.lexsort((body_members, body[body_members]))].astype(np.int64)
        offset = 1
        while offset < len(by_body):
            same = body[by_body[offset:]] == body[by_body[:-offset]]
            if not np.any(same):
                break
            local_pairs.append(np.stack((by_body[:-offset][same], by_body[offset:][same]), axis=1))
            offset += 1
        local_pairs = np.concatenate(local_pairs).reshape(-1, 2)
        small = sizes[member_aggregate[local_pairs[:, 0]]] <= 32 if len(local_pairs) else np.zeros(0, dtype=bool)
        local_pairs = local_pairs[small]
        if len(local_pairs):
            owner = member_aggregate[local_pairs[:, 0]]
            first = member_local[local_pairs[:, 0]]
            second = member_local[local_pairs[:, 1]]
            np.bitwise_or.at(mask_np, (owner, first), np.left_shift(np.uint32(1), second.astype(np.uint32)))
            np.bitwise_or.at(mask_np, (owner, second), np.left_shift(np.uint32(1), first.astype(np.uint32)))

        # Unbounded shapes: a CSR over worlds, segment 0 = world -1, segment w + 1 = world w.
        world_count = int(world.max()) + 1 if shape_count and world.max() >= 0 else 0
        self.world_count = world_count
        self._shape_world_np = world
        self._unbounded_mask = unbounded
        unbounded_shapes_np, unbounded_start_np, segment_counts = self._unbounded_csr(
            np.flatnonzero(unbounded), stacklevel=3
        )
        self.unbounded_count = len(unbounded_shapes_np)

        if aggregate_pairs_max is None:
            aggregate_pairs_max = AGGREGATE_PAIRS_PER_AGGREGATE * aggregate_count
        if int(aggregate_pairs_max) != aggregate_pairs_max or aggregate_pairs_max < 0:
            raise ValueError(f"aggregate_pairs_max must be a non-negative integer, got {aggregate_pairs_max!r}")
        self.aggregate_pairs_max = int(aggregate_pairs_max)
        if self.ordered:
            # Grid statics may join the unbounded lists at the first build_static,
            # so each aggregate's view counts both kinds.
            static_counts = np.bincount(world[grid_statics] + 1, minlength=world_count + 1)
            own_segment = np.maximum(aggregate_world_np, -1) + 1
            in_view = np.where(
                aggregate_world_np >= 0,
                segment_counts[0] + segment_counts[own_segment] + static_counts[0] + static_counts[own_segment],
                len(unbounded_shapes_np) + len(grid_statics),
            ).astype(np.int64)
            sizes_np = sizes.astype(np.int64)
            slots = int(np.sum(sizes_np * (sizes_np - 1) // 2 + sizes_np * in_view))
            if aggregate_count >= 2:
                slots += self.aggregate_pairs_max * self.largest_aggregate**2
            self._ordered_slots = slots
            if slots > _ORDERED_SLOTS_MAX:
                raise ValueError(
                    f"BroadPhaseAggregate(ordered=True) could need {slots:,} pair slots per collide, more than "
                    f"its int32 item offsets address ({_ORDERED_SLOTS_MAX:,}); lower aggregate_pairs_max "
                    f"({self.aggregate_pairs_max:,}) or split the aggregates (largest: {self.largest_aggregate} "
                    "members)"
                )

        # The search path needs the full filter list; skip the copy when every self pass uses masks.
        self._mask = knobs["self_filter"] == "mask"
        needs_search = not self._mask or self.largest_aggregate > 32

        device = self.device
        self.aggregate_start = wp.array(aggregate_start_np, dtype=wp.int32, device=device)
        self.aggregate_shapes = wp.array(aggregate_shapes_np, dtype=wp.int32, device=device)
        self.aggregate_world = wp.array(aggregate_world_np, dtype=wp.int32, device=device)
        self.aggregate_mask = wp.array(mask_np, dtype=wp.uint32, device=device)
        self.unbounded_start = wp.array(unbounded_start_np, dtype=wp.int32, device=device)
        self.unbounded_shapes = wp.array(unbounded_shapes_np, dtype=wp.int32, device=device)
        self.cross_filter_pairs = wp.array(cross.astype(np.int32).reshape(-1, 2), dtype=wp.vec2i, device=device)
        self._search_filter_pairs = wp.array(
            (pairs if needs_search else pairs[:0]).astype(np.int32).reshape(-1, 2), dtype=wp.vec2i, device=device
        )
        self._shape_body = wp.array(body.astype(np.int32), dtype=wp.int32, device=device)
        self._no_displacement = wp.empty(0, dtype=wp.vec3, device=device)
        self._no_body_flags = wp.empty(0, dtype=wp.int32, device=device)

        self.aggregate_lower = wp.zeros(aggregate_count, dtype=wp.vec3, device=device)
        self.aggregate_upper = wp.zeros(aggregate_count, dtype=wp.vec3, device=device)
        self.aggregate_group = wp.zeros(aggregate_count, dtype=wp.int32, device=device)
        self.aggregate_pairs = wp.zeros(self.aggregate_pairs_max, dtype=wp.vec2i, device=device)
        self.aggregate_pair_count = wp.zeros(1, dtype=wp.int32, device=device)
        self.pair_demand = wp.zeros(1, dtype=wp.int64, device=device)
        self.pair_demand_max = wp.zeros(1, dtype=wp.int64, device=device)
        self.pair_overflow = wp.zeros(1, dtype=wp.int32, device=device)
        self.aggregate_pair_overflow = wp.zeros(1, dtype=wp.int32, device=device)
        self.aggregate_pair_demand_max = wp.zeros(1, dtype=wp.int32, device=device)
        self.extent_max = wp.zeros(1, dtype=wp.float32, device=device)
        # The device-side warnings' flag (see verify); None leaves them off.
        self.verify_flag = wp.ones(1, dtype=wp.int32, device=device) if verify else None
        # How many times reset_statistics() ran, so that a caller can re-arm its own warning.
        self._statistics_resets = 0

        # Static grid: its arrays are allocated by the first build_static, when the
        # cell size and the statics too large for it are known.
        self.finite_static_count = len(finite_statics)
        self.static_grid = len(grid_statics) > 0
        self._grid_candidates = grid_statics
        self._static_cell_size_request = None if static_cell_size is None else float(static_cell_size)
        self.static_cell_size: float | None = None
        self.static_shape_count = 0
        self.static_entry_count = 0
        self.static_entry_capacity = 0
        self.static_entry_overflow = 0
        self.static_start: wp.array | None = None
        self.static_shapes: wp.array | None = None
        self.static_key: wp.array | None = None
        self.static_cell_lower: wp.array | None = None
        self.static_cell_upper: wp.array | None = None
        self.static_entries: wp.array | None = None
        self.static_bucket_start: wp.array | None = None
        static_stage = self.static_grid and aggregate_count >= 1

        # Ordered emission: sort scratch (twice the count, as the radix sort needs) and the per-item scan.
        # The items are the aggregates, then (with the static stage) each aggregate's static pairs, then the
        # aggregate-pair slots.
        self._static_item_base = aggregate_count
        self._expand_item_base = aggregate_count * (2 if static_stage else 1)
        self.aggregate_pair_keys: wp.array | None = None
        self.aggregate_pair_order: wp.array | None = None
        self.item_count: wp.array | None = None
        self.item_offset: wp.array | None = None
        if self.ordered:
            items = self._expand_item_base + self.aggregate_pairs_max
            self.aggregate_pair_keys = wp.zeros(2 * self.aggregate_pairs_max, dtype=wp.int64, device=device)
            self.aggregate_pair_order = wp.zeros(2 * self.aggregate_pairs_max, dtype=wp.int32, device=device)
            self.item_count = wp.zeros(items, dtype=wp.int32, device=device)
            self.item_offset = wp.zeros(items, dtype=wp.int32, device=device)
            self._item_scan_kwargs = _scan_kwargs(items, device)
        # The keys hold a < A in bits 32 and up and b < A below; the sort covers just those bits.
        self._order_end_bit = 32 + max(1, (aggregate_count - 1).bit_length())

        self._main = (
            BroadPhaseSAP(aggregate_world_np, sort_type="auto", device=device)
            if aggregate_count >= 2 and self.main == "sap"
            else None
        )
        self.split_aggregates = 0
        self.aggregate_extent_overflow: wp.array | None = None
        self.grid_cell: wp.array | None = None
        self.grid_bucket: wp.array | None = None
        self.grid_order: wp.array | None = None
        self.grid_bucket_count: wp.array | None = None
        self.grid_bucket_start: wp.array | None = None
        if self.main == "grid":
            self.aggregate_extent_overflow = wp.zeros(1, dtype=wp.int32, device=device)
            if aggregate_count >= 2:
                self._allocate_grid(aggregate_world_np)
        self._union_mode = knobs["union"]
        self._union_kernel = _create_union_kernel(knobs["union"])
        if self.ordered:
            self._self_kernel = _create_ordered_self_kernel(self._mask, False)
            self._self_write_kernel = _create_ordered_self_kernel(self._mask, True)
            self._expand_kernel = _create_ordered_expand_kernel(False)
            self._expand_write_kernel = _create_ordered_expand_kernel(True)
            self._static_kernel = _create_ordered_static_kernel(False) if static_stage else None
            self._static_write_kernel = _create_ordered_static_kernel(True) if static_stage else None
        else:
            self._self_kernel = _create_self_kernel(self._mask, knobs["emission"] == "tile")
            self._expand_kernel = _create_expand_kernel(knobs["emission"] == "tile")
            self._static_kernel = _create_static_kernel(knobs["emission"] == "tile") if static_stage else None
        # CPU tile launches run one lane per block (Warp GH-1413).
        self._block_dim = 1 if device.is_cpu else 32
        lanes = 64 * device.sm_count if device.is_cuda else 256
        self._expand_grid = min(self.aggregate_pairs_max, lanes)
        stages = ["union"]
        if aggregate_count >= 2:
            stages.append("main")
        if aggregate_count >= 1:
            stages.append("self")
        if static_stage:
            stages.append("static")
        if aggregate_count >= 2 and self._expand_grid > 0:
            stages.append("expand")
        if self.ordered:
            stages.append("write")
        stages.append("publish")
        self._stages = tuple(stages)
        self._args: _LaunchArgs | None = None

    @classmethod
    def from_model(
        cls,
        model: Model,
        *,
        shape_aggregate: np.ndarray | None = None,
        max_aggregate_size: int | None = None,
        main: str | None = None,
        aggregate_pairs_max: int | None = None,
        ordered: bool | None = None,
        static_cell_size: float | None = None,
        speculative_contact_gap_max: float | None = None,
        verify: bool = False,
    ) -> BroadPhaseAggregate:
        """Build the broad phase for a model, inferring the aggregates from its articulations.

        Args:
            model: The model whose shapes, bodies, joints and filter pairs are read.
            shape_aggregate: Aggregate of each shape, replacing the inferred one.
            max_aggregate_size: A positive size splits each inferred aggregate into
                chunks of about that many members over whole bodies (see
                :func:`infer_shape_aggregates`), so a chunk can exceed it by up to
                one body's shapes less one. ``None`` reads the module default
                (32). Not allowed together with ``shape_aggregate``.
            main: Main-level algorithm, see :class:`BroadPhaseAggregate`. ``None`` reads
                the module default (``"grid"``). Under ``"grid"`` every articulation with a non-root
                joint that has a linear degree of freedom (prismatic, free, D6 with a
                linear axis) is split into one aggregate per body, unless
                ``shape_aggregate`` is given, and ``shape_reach`` is computed from
                the model: from each aggregate's root (the deepest body above all
                its member bodies: the articulation's root, or a chunk's first
                body), every joint adds its offsets in parent and child, and every
                shape its offset in its body, the reach of its box about its origin
                (the larger of its collision radius and, for spheres, capsules and
                boxes, the bound of their scale, otherwise the norm of the
                farthest corner of its local box, ``shape_collision_aabb_*``),
                margin and gap.
                A split articulation's bodies, bodies outside articulations and
                static shapes are their own roots. ``shape_extent`` is the largest
                |coordinate| of a shape's world position in the model's initial
                state (``body_q``). A given ``shape_aggregate`` is used as is with
                this reach, and ``aggregate_extent_overflow`` reports a union box
                the cell cannot hold. With ``main=None`` the grid is built, then
                rules pick the main level, in this order; the thresholds are
                judgments bracketed by scenes measured on an L40 at their
                initial pose (see ``_KNOBS``). *Worlds:* from
                ``_KNOBS["grid_min_worlds"]`` (2,048) worlds >= 0 holding aggregates
                on, the size rule is skipped, since the SAP sorts every world's
                segment. *Size:* otherwise, below ``_KNOBS["grid_min_aggregates"]``
                (32,768) aggregates the SAP is built (``main_fallback == "size"``),
                since the grid's main stage costs about 0.1-0.25 ms at every size;
                a model without aggregates gets it too, harmlessly, since neither
                level has a main stage below two aggregates. *Cell
                guard:* when the grid's cell would exceed
                ``_KNOBS["grid_cell_ratio_max"]`` times the median over aggregates
                of twice their reach, the SAP is built (``"cell"``), with a
                :class:`NewtonWarning`. *Occupancy:* when the mean over aggregates
                of how many aggregates share its grid cell, itself included,
                exceeds ``_KNOBS["grid_occupancy_max"]`` (90; ``inf`` turns the
                guard off), the SAP is built (``"occupancy"``), without a warning.
                The occupancy bins the union
                boxes of the initial pose (``body_q``; spheres, capsules, boxes,
                meshes, convex hulls and heightfields bounded as
                :meth:`CollisionPipeline.collide` bounds them, other shapes by the
                cube of their reach about their origin) as the grid bins them: by the
                cell of the padded lower corner, under the world key; the
                statistics report it as ``grid_occupancy``. A SAP picked by a rule
                takes the unsplit aggregates. An explicit ``"grid"`` skips the
                rules.
            aggregate_pairs_max: Capacity of the aggregate-pair list.
            ordered: Write the pairs in the deterministic order, see
                :class:`BroadPhaseAggregate`. ``None`` reads the module default.
            static_cell_size: Cell size of the static grid [m], see :meth:`__init__`.
            speculative_contact_gap_max: The speculative cap the boxes will be grown
                and swept by [m], as :class:`~newton.CollisionPipeline` takes it, or
                ``None`` for none. Each body shape's box can grow by up to the cap
                on every side, and its sweep before the union can add up to the
                cap more, so under the grid every body shape's reach grows by twice
                the cap (the cell by four times it), and the cell still holds every
                union box. Adding to the reach rather than to the cell keeps the
                cell's slack and pad, which are sized from the reach, sized for the
                grown boxes. :class:`~newton.CollisionPipeline` passes its own.
            verify: Device-side warnings for dropped pairs and the grid's latch,
                see :meth:`__init__`. :class:`~newton.CollisionPipeline` passes its
                ``verify_buffers``.

        Returns:
            The broad phase on the model's device. With finite static shapes, call
            :meth:`build_static` before the first :meth:`launch`.

        Raises:
            NotImplementedError: If the model has body variants
                (:meth:`ModelBuilder.add_body_variants <newton.ModelBuilder.add_body_variants>`).
            ValueError: As :meth:`__init__` does; also if ``max_aggregate_size`` is
                given with ``shape_aggregate``, if ``speculative_contact_gap_max`` is
                not a non-negative finite number or ``None``, if ``main=None``
                resolves to the grid and ``_KNOBS["grid_cell_ratio_max"]`` is not a positive finite
                number, ``_KNOBS["grid_occupancy_max"]`` not a positive number,
                ``_KNOBS["grid_min_aggregates"]`` not an integer >= 0 or
                ``_KNOBS["grid_min_worlds"]`` not an integer >= 1, or, for the
                grid, the model's initial pose puts a shape at a non-finite
                position.

        Warns:
            NewtonWarning: When the cell guard builds the SAP instead of the grid.
        """
        if model._has_variant_rows:
            # Selecting a variant rewrites a row's geometry and flags on the device, after the aggregates, their
            # members and the grid's reach were fixed from the rows as built.
            raise NotImplementedError(
                "BroadPhaseAggregate does not support models with body variants (ModelBuilder.add_body_variants); "
                "use the 'sap' or 'nxn' broad phase"
            )
        inferred = shape_aggregate is None
        if inferred:
            shape_aggregate = infer_shape_aggregates(model, max_aggregate_size=max_aggregate_size)
        elif max_aggregate_size is not None:
            raise ValueError("max_aggregate_size applies to inferred aggregates only; it cannot split shape_aggregate")
        growth = 0.0
        if speculative_contact_gap_max is not None:
            cap = speculative_contact_gap_max
            number = isinstance(cap, (int, float, np.integer, np.floating)) and not isinstance(cap, bool)
            if not (number and np.isfinite(cap) and cap >= 0.0):
                raise ValueError(
                    f"speculative_contact_gap_max must be a non-negative finite number or None, got {cap!r}"
                )
            # Each body shape's box grows by up to the cap on every side, and its sweep adds up to the cap more.
            growth = 2.0 * float(cap)
        guarded = main is None
        main = _KNOBS["main"] if main is None else main
        ratio_max = None
        if guarded and main == "grid":
            ratio_max = _KNOBS["grid_cell_ratio_max"]
            occupancy_max = _KNOBS["grid_occupancy_max"]
            for name, value, finite in (
                ("grid_cell_ratio_max", ratio_max, True),
                ("grid_occupancy_max", occupancy_max, False),
            ):
                number = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
                if not (number and value > 0.0 and (np.isfinite(value) or not finite)):
                    kind = "positive finite number" if finite else "positive number"
                    raise ValueError(f"{name} must be a {kind}, got {value!r}")
            min_aggregates = _KNOBS["grid_min_aggregates"]
            min_worlds = _KNOBS["grid_min_worlds"]
            for name, value, least in (("grid_min_aggregates", min_aggregates, 0), ("grid_min_worlds", min_worlds, 1)):
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < least:
                    raise ValueError(f"{name} must be an integer of at least {least}, got {value!r}")

        def build(level: str, aggregate: np.ndarray, reach: np.ndarray | None, extent: float | None):
            return cls(
                model.shape_world,
                model.shape_body,
                aggregate,
                model.shape_type,
                model.shape_scale,
                model.shape_flags,
                model.shape_collision_filter_pairs_array(),
                main=level,
                shape_reach=reach,
                shape_extent=extent,
                aggregate_pairs_max=aggregate_pairs_max,
                ordered=ordered,
                static_cell_size=static_cell_size,
                verify=verify,
                device=model.device,
            )

        if main != "grid":
            return build(main, shape_aggregate, None, None)
        split = _linear_joint_articulations(model)
        grid_aggregate, split_aggregates = shape_aggregate, 0
        if inferred:
            grid_aggregate, split_aggregates = _split_articulations(model, shape_aggregate, split)
        shape_reach = _model_shape_reach(model, split, grid_aggregate, growth)
        shape_extent = _model_shape_extent(model)
        if not np.isfinite(shape_extent):
            raise ValueError(
                "BroadPhaseAggregate.from_model(main='grid'): the model's initial pose (body_q and shape_transform) "
                "puts a shape at a non-finite position"
            )
        broad_phase = build("grid", grid_aggregate, shape_reach, shape_extent)
        broad_phase.split_aggregates = split_aggregates
        if ratio_max is None:
            return broad_phase
        # The default main level's rules, in order; the first that trips builds the sweep.
        fallback, occupancy = "", 0.0
        if broad_phase.aggregate_worlds < min_worlds and broad_phase.aggregate_count < min_aggregates:
            # Few worlds and few aggregates: the grid's fixed main stage costs more than the sweep.
            fallback = "size"
        elif broad_phase.aggregate_count:
            # The longest aggregate's reach sets the cell for all; far above the median, the sweep is cheaper.
            shapes, starts = broad_phase._aggregate_shapes_np, broad_phase._aggregate_start_np[:-1]
            extent = 2.0 * np.maximum.reduceat(shape_reach[shapes].astype(np.float64), starts)
            median = float(np.median(extent))
            if broad_phase.cell_size > ratio_max * median:
                warnings.warn(
                    f"BroadPhaseAggregate.from_model: the grid's cell would be {broad_phase.cell_size:.3g} m, "
                    f"{broad_phase.cell_size / median:.1f} times the median aggregate extent (2 x reach, {median:.3g} "
                    f"m), above the cell guard's ratio {ratio_max:g}: the longest aggregate sets the cell for all of "
                    "them, so the exact sweep is built instead, which costs time, not pairs. Likely causes: a long "
                    "articulation, or a given shape_aggregate that keeps one whole, or a shape whose geometry sits "
                    "far from its frame origin. To choose yourself, build "
                    "BroadPhaseAggregate.from_model(model, main='sap', verify=True) (no warning) or main='grid' "
                    "(keeps the grid), with a smaller max_aggregate_size for shorter chunks and the pipeline's "
                    "speculative_contact_gap_max, and pass it to CollisionPipeline with a NarrowPhase (the expert "
                    "path)",
                    NewtonWarning,
                    stacklevel=_external_warning_stacklevel(),
                )
                fallback = "cell"
            else:
                # Crowded cells: each query scans its cell's aggregates. Binned as the grid bins the initial pose.
                shape_lower, shape_upper = _model_shape_box(model)
                lower = np.minimum.reduceat(shape_lower[shapes], starts).astype(np.float32)
                upper = np.maximum.reduceat(shape_upper[shapes], starts).astype(np.float32)
                occupancy = _grid_occupancy(lower, upper, broad_phase._aggregate_world_np, broad_phase.cell_size)
                if occupancy > occupancy_max:
                    fallback = "occupancy"
        if fallback:
            broad_phase = None  # free the grid's device arrays before the sweep allocates its own
            broad_phase = build("sap", shape_aggregate, None, None)
            broad_phase.main_fallback = fallback
        broad_phase.grid_occupancy = occupancy
        return broad_phase

    def _allocate_grid(self, aggregate_world_np: np.ndarray) -> None:
        """Allocate the grid main level's arrays, for two aggregates or more.

        T = the next power of two at least 2A buckets. Per aggregate: its cell,
        and its bucket and id as the radix sort's keys and values (2A each, the
        second half the sort's scratch); per bucket: its count and its start.
        """
        device = self.device
        count = self.aggregate_count
        table = _static_table_size(2 * count)
        self._grid_bucket_mask = table - 1
        self._grid_end_bit = table.bit_length() - 1
        self._grid_inv_cell_size = float(np.float32(1.0) / np.float32(self.cell_size))
        self._grid_global_aggregates = int(np.count_nonzero(aggregate_world_np < 0))
        self.grid_cell = wp.zeros(count, dtype=wp.vec3i, device=device)
        self.grid_bucket = wp.zeros(2 * count, dtype=wp.int32, device=device)
        self.grid_order = wp.zeros(2 * count, dtype=wp.int32, device=device)
        self.grid_bucket_count = wp.zeros(table, dtype=wp.int32, device=device)
        self.grid_bucket_start = wp.zeros(table, dtype=wp.int32, device=device)
        self._grid_scan_kwargs = _scan_kwargs(table, device)

    def _unbounded_csr(self, shapes: np.ndarray, stacklevel: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Order unbounded shapes into a CSR over worlds, warning when a world sees many of them.

        Segment 0 holds world -1 and segment w + 1 world w, each in shape order.
        ``stacklevel`` is the warning's, counted from this method.

        Returns:
            The shapes in CSR order (int32), the segment starts (int32, world_count + 2)
            and the segment counts.
        """
        segment = self._shape_world_np[shapes] + 1
        shapes = shapes[np.argsort(segment, kind="stable")].astype(np.int32)
        counts = np.bincount(segment, minlength=self.world_count + 1)
        start = np.zeros(self.world_count + 2, dtype=np.int32)
        np.cumsum(counts, out=start[1:])
        seen = counts[0] + (counts[1:] if self.world_count else np.zeros(1, dtype=np.int64))
        if seen.max() > UNBOUNDED_PER_WORLD_WARNING:
            warnings.warn(
                f"BroadPhaseAggregate: a world sees {int(seen.max())} unbounded shapes (infinite planes, "
                f"heightfields and static shapes larger than {STATIC_UNBOUNDED_CELLS} static cells), more than "
                f"{UNBOUNDED_PER_WORLD_WARNING}; every aggregate tests its box against each of them every collide",
                NewtonWarning,
                stacklevel=stacklevel,
            )
        return shapes, start, counts

    def build_static(self, shape_lower: wp.array[wp.vec3], shape_upper: wp.array[wp.vec3]) -> None:
        """Bin the finite static shapes into the static grid, from the current boxes of every shape.

        The ``"static"`` stage sees each static only through the cells it was
        binned into, so a static shape that moves is seen at its new place only
        after the next call (:meth:`CollisionPipeline.refresh_static_shapes`
        makes it from the model): until then an aggregate tests it only where its
        cells meet the old ones, against the current boxes, so pairs at the new
        place are missed and no false pair appears. A no-op without finite
        static shapes, or when ``_KNOBS["statics"]`` made them aggregates.

        The first call sets the cell size h (``static_cell_size``, or the longest
        union-box edge, at least 0.5 m), moves every static with a box edge over
        ``STATIC_UNBOUNDED_CELLS * h`` to its world's unbounded list, and
        allocates the grid: entries for ``2 * (first entries) + 64`` static
        cells, and a hash table of the next power of two of buckets. Later calls
        rebin into the same arrays, so captured graphs keep working; statics
        whose entries do not fit are left out whole, in grid order, and not seen,
        with ``static_entry_overflow`` set and a :class:`NewtonWarning`.

        Args:
            shape_lower: Lower bound of each shape's box, as passed to :meth:`launch` [m].
            shape_upper: Upper bound of each shape's box [m].

        Raises:
            RuntimeError: While the device is capturing a graph, or when a static
                shape would move between the grid and the unbounded lists (rebuild
                the broad phase then).
            ValueError: If the arrays do not have one box per shape, or a finite
                static shape's box is not finite.
        """
        if self.device.is_capturing:
            raise RuntimeError(
                "BroadPhaseAggregate.build_static() reads and writes the device from the host; call it outside "
                "graph capture"
            )
        lower = _host(shape_lower, np.float64).reshape(-1, 3)
        upper = _host(shape_upper, np.float64).reshape(-1, 3)
        for name, array in (("shape_lower", lower), ("shape_upper", upper)):
            if len(array) != self.shape_count:
                raise ValueError(f"{name} must have {self.shape_count} entries, got {len(array)}")
        if not self.static_grid:
            return
        candidates = self._grid_candidates
        finite = np.all(np.isfinite(lower[candidates]), axis=1) & np.all(np.isfinite(upper[candidates]), axis=1)
        if not np.all(finite):
            raise ValueError(
                f"The boxes of finite static shapes must be finite; shape {int(candidates[np.argmin(finite)])} "
                "has a non-finite bound"
            )

        first = self.static_cell_size is None
        if first:
            cell_size = self._static_cell_size_request
            if cell_size is None:
                cell_size = max(_STATIC_CELL_SIZE_FLOOR, self._longest_union_edge(lower, upper))
            cell_size = float(np.float32(cell_size))
        else:
            cell_size = self.static_cell_size
        edge = np.max(upper[candidates] - lower[candidates], axis=1)
        large = edge > STATIC_UNBOUNDED_CELLS * cell_size
        if first:
            # The grid statics, by world (world -1 first) and shape; the large ones join the unbounded lists.
            grid = candidates[~large]
            grid = grid[np.argsort(self._shape_world_np[grid], kind="stable")]
        else:
            if not np.array_equal(large, self._static_large):
                changed = candidates[large != self._static_large]
                raise RuntimeError(
                    f"BroadPhaseAggregate.build_static(): {len(changed)} finite static shapes (first: shape "
                    f"{int(changed[0])}) crossed the {STATIC_UNBOUNDED_CELLS}-cell edge "
                    f"({STATIC_UNBOUNDED_CELLS * cell_size:g} m) between the static grid and the unbounded lists; "
                    "rebuild the pipeline"
                )
            grid = self._grid_shapes_np

        # The cells of each grid static: its box padded on the host, so that the
        # device's float32 cells of any of its points lie inside.
        inv_cell_size = np.float32(1.0) / np.float32(cell_size)
        limit = float(_STATIC_CELL_LIMIT)
        grid_lower = lower[grid]
        grid_upper = upper[grid]
        pad_lower = _STATIC_BIN_PAD * np.maximum(1.0, np.abs(grid_lower))
        pad_upper = _STATIC_BIN_PAD * np.maximum(1.0, np.abs(grid_upper))
        cell_low = np.clip(np.floor((grid_lower - pad_lower) * float(inv_cell_size)), -limit, limit).astype(np.int64)
        cell_high = np.clip(np.floor((grid_upper + pad_upper) * float(inv_cell_size)), -limit, limit).astype(np.int64)
        span = cell_high - cell_low + 1
        counts = np.prod(span, axis=1)

        if first:
            self._allocate_static(grid, large, cell_size, inv_cell_size, 2 * int(counts.sum()) + _STATIC_ENTRY_SLACK)

        # Whole statics in grid order while their entries fit.
        kept = int(np.searchsorted(np.cumsum(counts), self.static_entry_capacity, side="right"))
        counts = counts[:kept]
        entry_count = int(counts.sum())
        owner = np.repeat(np.arange(kept, dtype=np.int64), counts)
        local = np.arange(entry_count, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
        span_x = span[owner, 0]
        span_y = span[owner, 1]
        entries = np.empty((entry_count, 4), dtype=np.int64)
        entries[:, 0] = owner
        entries[:, 1] = cell_low[owner, 0] + local % span_x
        entries[:, 2] = cell_low[owner, 1] + (local // span_x) % span_y
        entries[:, 3] = cell_low[owner, 2] + local // (span_x * span_y)
        entries = entries.astype(np.int32)
        buckets = np.zeros(0, dtype=np.int64)
        if entry_count:
            entries_device = wp.array(entries, dtype=wp.vec4i, device=self.device)
            buckets_device = wp.empty(entry_count, dtype=wp.int32, device=self.device)
            wp.launch(
                _static_entry_bucket_kernel,
                dim=entry_count,
                inputs=[entries_device, self.static_key, self._static_bucket_mask],
                outputs=[buckets_device],
                device=self.device,
                record_tape=False,
            )
            buckets = buckets_device.numpy().astype(np.int64)
        # By bucket, then world key, cell (z, y, x) and grid index: one world key's
        # statics of a cell are adjacent and in shape order.
        world_key = self._shape_world_np[grid][entries[:, 0]] + 1
        order = np.lexsort((entries[:, 0], entries[:, 1], entries[:, 2], entries[:, 3], world_key, buckets))
        table = self.static_bucket_start.shape[0] - 1
        bucket_start = np.searchsorted(buckets[order], np.arange(table + 1), side="left").astype(np.int32)
        padded = np.full((self.static_entry_capacity, 4), -1, dtype=np.int32)
        padded[:entry_count] = entries[order]
        self.static_entries.assign(padded)
        self.static_bucket_start.assign(bucket_start)
        if len(grid):
            # A static that did not fit has no entries, so no walk finds it. Its range lies beyond the
            # clamped cells on both sides, so the list path's interval test rejects it from every
            # aggregate too. (An empty range such as [c, c - 1] would still pass that test.)
            cell_low[kept:] = _STATIC_CELL_LIMIT + 1
            cell_high[kept:] = -_STATIC_CELL_LIMIT - 1
            self.static_cell_lower.assign(cell_low.astype(np.int32))
            self.static_cell_upper.assign(cell_high.astype(np.int32))
        self.static_entry_count = entry_count
        self.static_entry_overflow = int(kept < len(grid))
        if kept < len(grid):
            warnings.warn(
                f"BroadPhaseAggregate: the static grid holds {self.static_entry_capacity:,} cell entries; "
                f"{len(grid) - kept} of its {len(grid)} static shapes (first: shape {int(grid[kept])}) do not fit "
                "and are not binned, so their pairs can be missed; rebuild the pipeline to size the grid anew",
                NewtonWarning,
                stacklevel=2,
            )

    def _longest_union_edge(self, lower: np.ndarray, upper: np.ndarray) -> float:
        """Return the longest finite edge of the aggregates' union boxes over the given shape boxes [m], or 0."""
        if self.aggregate_count == 0:
            return 0.0
        starts = self._aggregate_start_np[:-1]
        union_lower = np.minimum.reduceat(lower[self._aggregate_shapes_np], starts, axis=0)
        union_upper = np.maximum.reduceat(upper[self._aggregate_shapes_np], starts, axis=0)
        edges = (union_upper - union_lower).reshape(-1)
        edges = edges[np.isfinite(edges)]
        return float(edges.max()) if len(edges) else 0.0

    def _allocate_static(
        self, grid: np.ndarray, large: np.ndarray, cell_size: float, inv_cell_size: np.float32, capacity: int
    ) -> None:
        """Allocate the static grid at the first build and move the large statics to the unbounded lists."""
        device = self.device
        if np.any(large):
            shapes = np.concatenate((np.flatnonzero(self._unbounded_mask), self._grid_candidates[large]))
            unbounded_shapes_np, unbounded_start_np, _ = self._unbounded_csr(np.sort(shapes), stacklevel=4)
            self.unbounded_start = wp.array(unbounded_start_np, dtype=wp.int32, device=device)
            self.unbounded_shapes = wp.array(unbounded_shapes_np, dtype=wp.int32, device=device)
            self.unbounded_count = len(unbounded_shapes_np)
        segment = self._shape_world_np[grid] + 1
        static_start_np = np.zeros(self.world_count + 2, dtype=np.int32)
        np.cumsum(np.bincount(segment, minlength=self.world_count + 1), out=static_start_np[1:])
        table = _static_table_size(capacity)
        self.static_start = wp.array(static_start_np, dtype=wp.int32, device=device)
        self.static_shapes = wp.array(grid.astype(np.int32), dtype=wp.int32, device=device)
        self.static_key = wp.array(segment.astype(np.int32), dtype=wp.int32, device=device)
        self.static_cell_lower = wp.zeros(len(grid), dtype=wp.vec3i, device=device)
        self.static_cell_upper = wp.zeros(len(grid), dtype=wp.vec3i, device=device)
        self.static_entries = wp.zeros(capacity, dtype=wp.vec4i, device=device)
        self.static_bucket_start = wp.zeros(table + 1, dtype=wp.int32, device=device)
        self._static_bucket_mask = table - 1
        self._static_inv_cell_size = float(inv_cell_size)
        self._static_large = large
        self._grid_shapes_np = grid
        self.static_cell_size = cell_size
        self.static_shape_count = len(grid)
        self.static_entry_capacity = capacity

    def launch(
        self,
        shape_lower: wp.array[wp.vec3],
        shape_upper: wp.array[wp.vec3],
        shape_gap: wp.array[float] | None,
        shape_collision_group: wp.array[int],
        shape_world: wp.array[int],
        shape_count: int,
        # Outputs
        candidate_pair: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[int],
        device: Devicelike | None = None,
        filter_pairs: wp.array[wp.vec2i] | None = None,
        num_filter_pairs: int | None = None,
        skip_count_zero: bool = False,
        *,
        shape_body: wp.array[int] | None = None,
        body_flags: wp.array[int] | None = None,
        include_static_kinematic_pairs: bool = True,
        shape_displacement: wp.array[wp.vec3] | None = None,
        sort_axis_displacement_limit: float | None = None,
    ) -> None:
        """Find the candidate shape pairs; the signature is that of :meth:`~newton.geometry.BroadPhaseSAP.launch`.

        Args:
            shape_lower: Lower bound of each shape's box, already expanded by its margin and gap [m].
            shape_upper: Upper bound of each shape's box [m].
            shape_gap: Must be ``None``: the boxes are pre-expanded.
            shape_collision_group: Collision group of each shape, read every collide.
            shape_world: World index of each shape.
            shape_count: Number of shapes; must equal the construction's.
            candidate_pair: Output canonical pairs ``(min, max)``; its length is the capacity.
            candidate_pair_count: Output count, ``min(pair_demand, capacity)``.
            device: Device to launch on. If None, the construction's.
            filter_pairs: ``None`` or the construction's filter pairs; the exclusions
                were built at construction.
            num_filter_pairs: Count of ``filter_pairs``; must match the construction's.
            skip_count_zero: Accepted for compatibility; the publish writes the count.
            shape_body: Accepted for compatibility; the bodies are the construction's.
            body_flags: Body flags identifying kinematic bodies.
            include_static_kinematic_pairs: Whether to keep static-kinematic and
                kinematic-kinematic pairs.
            shape_displacement: Optional displacement of each shape over the collision
                interval, for swept tests [m].
            sort_axis_displacement_limit: Optional non-negative per-axis clamp of the
                displacement used to sweep the union boxes [m]. Member pairs are tested
                with the unclamped displacement.

        Raises:
            ValueError: If an argument disagrees with the construction.
            RuntimeError: If there are finite static shapes and :meth:`build_static`
                has not run yet.
        """
        del skip_count_zero
        if shape_gap is not None:
            raise ValueError("BroadPhaseAggregate expects pre-expanded boxes; shape_gap must be None")
        if self.static_grid and self.static_cell_size is None:
            raise RuntimeError(
                f"BroadPhaseAggregate has {self.finite_static_count} finite static shapes; call build_static() "
                "(CollisionPipeline.refresh_static_shapes()) before the first launch()"
            )
        if shape_count != self.shape_count:
            raise ValueError(f"shape_count must be {self.shape_count}, got {shape_count}")
        for name, array in (("shape_lower", shape_lower), ("shape_upper", shape_upper)):
            if array.shape[0] != self.shape_count:
                raise ValueError(f"{name} must have {self.shape_count} entries, got {array.shape[0]}")
        if filter_pairs is not None:
            count = num_filter_pairs if num_filter_pairs is not None else filter_pairs.shape[0]
            if count != self._filter_pair_count:
                raise ValueError(
                    f"filter_pairs must be the construction's {self._filter_pair_count} pairs, got {count}"
                )
        if shape_body is not None and shape_body.shape[0] != self.shape_count:
            raise ValueError(f"shape_body must have {self.shape_count} entries, got {shape_body.shape[0]}")
        if shape_displacement is not None and shape_displacement.shape[0] != self.shape_count:
            raise ValueError(
                f"shape_displacement length must match the shape bounds ({self.shape_count}), "
                f"got {shape_displacement.shape[0]}"
            )
        if sort_axis_displacement_limit is None:
            limit = -1.0
        else:
            if not np.isfinite(sort_axis_displacement_limit) or sort_axis_displacement_limit < 0.0:
                raise ValueError(
                    "sort_axis_displacement_limit must be a non-negative finite number, "
                    f"got {sort_axis_displacement_limit!r}"
                )
            limit = float(sort_axis_displacement_limit)

        self._args = _LaunchArgs(
            shape_lower=shape_lower,
            shape_upper=shape_upper,
            shape_group=shape_collision_group,
            shape_world=shape_world,
            body_flags=self._no_body_flags if body_flags is None else body_flags,
            include_static_kinematic_pairs=bool(include_static_kinematic_pairs),
            shape_displacement=self._no_displacement if shape_displacement is None else shape_displacement,
            displacement_limit=limit,
            candidate_pair=candidate_pair,
            candidate_pair_count=candidate_pair_count,
            device=self.device if device is None else wp.get_device(device),
        )
        for stage in self._stages:
            self._run_stage(stage)

    def _launch_stage(self, stage: AggregateStage) -> None:
        """Replay one stage with the last :meth:`launch`'s arrays, for timing.

        Zeroes the counter the stage appends to first: ``aggregate_pair_count``
        for ``"main"``, ``pair_demand`` for ``"self"``, ``"static"`` and
        ``"expand"``. A stage skipped at construction does nothing; ``"write"`` runs
        only when ordered. When ordered, ``"self"``, ``"static"`` and
        ``"expand"`` are the count passes and ``"write"`` is the scan and the
        write passes.
        """
        if stage not in _STAGES:
            raise ValueError(f"Unknown aggregate broad phase stage {stage!r}; expected one of {_STAGES}")
        if self._args is None:
            raise RuntimeError("_launch_stage replays the last launch(); call launch() first")
        if stage == "main":
            self.aggregate_pair_count.zero_()
        elif stage in ("self", "static", "expand"):
            self.pair_demand.zero_()
        if stage in self._stages:
            self._run_stage(stage)

    def _run_stage(self, stage: str) -> None:
        args = self._args
        device = args.device
        if stage == "union":
            inputs = [
                self.aggregate_count,
                self.aggregate_start,
                self.aggregate_shapes,
                args.shape_lower,
                args.shape_upper,
                args.shape_group,
                args.shape_displacement,
                args.displacement_limit,
            ]
            outputs = [
                self.aggregate_lower,
                self.aggregate_upper,
                self.aggregate_group,
                self.extent_max,
                self.aggregate_pair_count,
                self.pair_demand,
            ]
            if self._union_mode == "thread":
                wp.launch(
                    self._union_kernel,
                    dim=max(self.aggregate_count, 1),
                    inputs=inputs,
                    outputs=outputs,
                    device=device,
                    record_tape=False,
                )
            else:
                wp.launch_tiled(
                    self._union_kernel,
                    dim=max(self.aggregate_count, 1),
                    inputs=inputs,
                    outputs=outputs,
                    block_dim=self._block_dim,
                    device=device,
                    record_tape=False,
                )
        elif stage == "main":
            if self.main == "grid":
                self._run_grid(device)
            else:
                self._main.launch(
                    self.aggregate_lower,
                    self.aggregate_upper,
                    None,
                    self.aggregate_group,
                    self.aggregate_world,
                    self.aggregate_count,
                    self.aggregate_pairs,
                    self.aggregate_pair_count,
                    device,
                    skip_count_zero=True,
                )
            if self.ordered:
                wp.launch(
                    _order_keys_kernel,
                    dim=self.aggregate_pairs_max,
                    inputs=[self.aggregate_pairs, self.aggregate_pair_count, self.aggregate_pairs_max],
                    outputs=[self.aggregate_pair_keys, self.aggregate_pair_order],
                    device=device,
                    record_tape=False,
                )
                wp.utils.radix_sort_pairs(
                    self.aggregate_pair_keys,
                    self.aggregate_pair_order,
                    self.aggregate_pairs_max,
                    end_bit=self._order_end_bit,
                )
        elif self.ordered and stage in ("self", "static", "expand", "write"):
            self._run_ordered_stage(stage)
        elif stage == "static":
            wp.launch_tiled(
                self._static_kernel,
                dim=self.aggregate_count,
                inputs=self._static_inputs(args),
                outputs=[args.candidate_pair, self.pair_demand],
                block_dim=self._block_dim,
                device=device,
                record_tape=False,
            )
        elif stage == "self":
            wp.launch_tiled(
                self._self_kernel,
                dim=self.aggregate_count,
                inputs=[
                    self.aggregate_start,
                    self.aggregate_shapes,
                    self.aggregate_world,
                    self.aggregate_mask,
                    self.aggregate_lower,
                    self.aggregate_upper,
                    self.unbounded_start,
                    self.unbounded_shapes,
                    self.world_count,
                    args.shape_lower,
                    args.shape_upper,
                    args.shape_group,
                    args.shape_world,
                    self._shape_body,
                    args.body_flags,
                    args.include_static_kinematic_pairs,
                    args.shape_displacement,
                    args.displacement_limit,
                    self._search_filter_pairs,
                    self._search_filter_pairs.shape[0],
                    self.cross_filter_pairs,
                    self.cross_filter_pairs.shape[0],
                ],
                outputs=[args.candidate_pair, self.pair_demand],
                block_dim=self._block_dim,
                device=device,
                record_tape=False,
            )
        elif stage == "expand":
            wp.launch_tiled(
                self._expand_kernel,
                dim=self._expand_grid,
                inputs=[
                    self._expand_grid,
                    self.aggregate_pairs,
                    self.aggregate_pair_count,
                    self.aggregate_pairs_max,
                    self.aggregate_start,
                    self.aggregate_shapes,
                    args.shape_lower,
                    args.shape_upper,
                    args.shape_group,
                    args.shape_world,
                    self._shape_body,
                    args.body_flags,
                    args.include_static_kinematic_pairs,
                    args.shape_displacement,
                    self.cross_filter_pairs,
                    self.cross_filter_pairs.shape[0],
                ],
                outputs=[args.candidate_pair, self.pair_demand],
                block_dim=self._block_dim,
                device=device,
                record_tape=False,
            )
        elif self.ordered:
            wp.launch(
                _ordered_publish_kernel,
                dim=1,
                inputs=[
                    self.item_offset,
                    self.item_count,
                    self.aggregate_pair_count,
                    self.aggregate_pairs_max,
                    args.candidate_pair.shape[0],
                    self.verify_flag,
                ],
                outputs=[
                    args.candidate_pair_count,
                    self.pair_demand,
                    self.pair_overflow,
                    self.pair_demand_max,
                    self.aggregate_pair_overflow,
                    self.aggregate_pair_demand_max,
                ],
                device=device,
                record_tape=False,
            )
        else:
            wp.launch(
                _publish_kernel,
                dim=1,
                inputs=[
                    self.pair_demand,
                    self.aggregate_pair_count,
                    self.aggregate_pairs_max,
                    args.candidate_pair.shape[0],
                    self.verify_flag,
                ],
                outputs=[
                    args.candidate_pair_count,
                    self.pair_overflow,
                    self.pair_demand_max,
                    self.aggregate_pair_overflow,
                    self.aggregate_pair_demand_max,
                ],
                device=device,
                record_tape=False,
            )

    def _run_grid(self, device: wp.Device) -> None:
        """Run the grid main level: zero the bucket counts, bin, sort by bucket, scan the starts and query."""
        count = self.aggregate_count
        self.grid_bucket_count.zero_()
        wp.launch(
            _grid_bin_kernel,
            dim=count,
            inputs=[
                self.aggregate_lower,
                self.aggregate_upper,
                self.aggregate_world,
                self.cell_size,
                self._grid_inv_cell_size,
                self._grid_bucket_mask,
                self.verify_flag,
            ],
            outputs=[
                self.grid_cell,
                self.grid_bucket,
                self.grid_order,
                self.grid_bucket_count,
                self.aggregate_extent_overflow,
            ],
            device=device,
            record_tape=False,
        )
        wp.utils.radix_sort_pairs(self.grid_bucket, self.grid_order, count, end_bit=self._grid_end_bit)
        wp.utils.array_scan(self.grid_bucket_count, self.grid_bucket_start, inclusive=False, **self._grid_scan_kwargs)
        wp.launch(
            _grid_query_kernel,
            dim=count,
            inputs=[
                self.aggregate_lower,
                self.aggregate_upper,
                self.aggregate_world,
                self.aggregate_group,
                self.grid_cell,
                self.grid_order,
                self.grid_bucket_start,
                self.grid_bucket_count,
                self._grid_bucket_mask,
                self._grid_global_aggregates,
                self.aggregate_pairs_max,
            ],
            outputs=[self.aggregate_pairs, self.aggregate_pair_count],
            device=device,
            record_tape=False,
        )

    def _static_inputs(self, args: _LaunchArgs) -> list:
        """The static-stage kernel inputs shared by the unordered and ordered kernels."""
        return [
            self.aggregate_start,
            self.aggregate_shapes,
            self.aggregate_world,
            self.aggregate_lower,
            self.aggregate_upper,
            self.static_start,
            self.static_shapes,
            self.static_key,
            self.static_cell_lower,
            self.static_cell_upper,
            self.static_entries,
            self.static_bucket_start,
            self._static_bucket_mask,
            self._static_inv_cell_size,
            args.shape_lower,
            args.shape_upper,
            args.shape_group,
            args.shape_world,
            self._shape_body,
            args.body_flags,
            args.include_static_kinematic_pairs,
            args.shape_displacement,
            args.displacement_limit,
            self.cross_filter_pairs,
            self.cross_filter_pairs.shape[0],
        ]

    def _run_ordered_stage(self, stage: str) -> None:
        """Run an ordered count pass (``"self"``, ``"static"``, ``"expand"``) or the scan and write passes (``"write"``)."""
        args = self._args
        device = args.device
        self_inputs = [
            self.aggregate_start,
            self.aggregate_shapes,
            self.aggregate_world,
            self.aggregate_mask,
            self.aggregate_lower,
            self.aggregate_upper,
            self.unbounded_start,
            self.unbounded_shapes,
            self.world_count,
            args.shape_lower,
            args.shape_upper,
            args.shape_group,
            args.shape_world,
            self._shape_body,
            args.body_flags,
            args.include_static_kinematic_pairs,
            args.shape_displacement,
            args.displacement_limit,
            self._search_filter_pairs,
            self._search_filter_pairs.shape[0],
            self.cross_filter_pairs,
            self.cross_filter_pairs.shape[0],
            self.item_offset,
        ]
        expand_inputs = [
            self._expand_grid,
            self._expand_item_base,
            self.aggregate_pair_keys,
            self.aggregate_pair_count,
            self.aggregate_pairs_max,
            self.aggregate_start,
            self.aggregate_shapes,
            args.shape_lower,
            args.shape_upper,
            args.shape_group,
            args.shape_world,
            self._shape_body,
            args.body_flags,
            args.include_static_kinematic_pairs,
            args.shape_displacement,
            self.cross_filter_pairs,
            self.cross_filter_pairs.shape[0],
            self.item_offset,
        ]
        static_inputs = None
        if "static" in self._stages:
            static_inputs = [
                self._static_item_base,
                *self._static_inputs(args),
                self.pair_demand,
                self.item_offset,
            ]
        outputs = [self.item_count, args.candidate_pair]
        launches = []
        if stage == "self":
            launches.append((self._self_kernel, self.aggregate_count, self_inputs))
        elif stage == "static":
            launches.append((self._static_kernel, self.aggregate_count, static_inputs))
        elif stage == "expand":
            launches.append((self._expand_kernel, self._expand_grid, expand_inputs))
        else:
            wp.utils.array_scan(self.item_count, self.item_offset, inclusive=False, **self._item_scan_kwargs)
            if "self" in self._stages:
                launches.append((self._self_write_kernel, self.aggregate_count, self_inputs))
            if "static" in self._stages:
                launches.append((self._static_write_kernel, self.aggregate_count, static_inputs))
            if "expand" in self._stages:
                launches.append((self._expand_write_kernel, self._expand_grid, expand_inputs))
        for kernel, dim, inputs in launches:
            wp.launch_tiled(
                kernel,
                dim=dim,
                inputs=inputs,
                outputs=outputs,
                block_dim=self._block_dim,
                device=device,
                record_tape=False,
            )

    def statistics(self) -> dict[str, int | float | str]:
        """Read the sizes, the last collide's counts and flags, and the latched maxima.

        Returns:
            ``main_level``, the main level in use (``"grid"`` or ``"sap"``), and
            ``main_fallback``, why the default grid was not built: ``"reach"`` (a
            direct construction without ``shape_reach``), ``"size"``, ``"cell"``
            or ``"occupancy"`` (the rules of :meth:`from_model`), or ``""``;
            ``aggregate_worlds``, how many worlds >= 0 hold aggregates, and
            ``grid_occupancy``, the mean over aggregates of how many share its
            grid cell at the initial pose, as the occupancy rule of
            :meth:`from_model` computed it (0.0 where the rule did not run);
            ``aggregates``, ``largest_aggregate``, ``unbounded_shapes`` and
            ``aggregate_pairs_max`` (construction; ``unbounded_shapes`` counts the
            statics too large for the grid once it is built); ``split_aggregates``
            (the aggregates the articulations :meth:`from_model` split for
            ``main="grid"`` became, else 0); ``static_shapes`` (the finite
            statics the static grid holds) and
            ``static_entry_overflow`` (1 when the last :meth:`build_static`
            dropped statics past the grid's capacity); ``aggregate_pair_count``
            (the main level's demand), ``aggregate_pair_overflow``,
            ``pair_demand`` and ``pair_overflow`` (last collide);
            ``aggregate_pair_demand_max``, ``pair_demand_max``, ``extent_max``
            (the longest union-box edge [m]) and, with ``main="grid"``,
            ``aggregate_extent_overflow`` (1 once a union box did not fit the
            grid's cell, so aggregate pairs may have been missed; else 0) latched
            since construction or :meth:`reset_statistics`; ``cell_size``, the
            grid's cell [m] (construction; 0 for other main levels), which holds a
            box of twice the largest member reach with its union pad out to the
            initial extent: bodies that travel far beyond it can fire the latch.

        Raises:
            RuntimeError: While the device is capturing a graph.
        """
        if self.device.is_capturing:
            raise RuntimeError("BroadPhaseAggregate.statistics() reads the device; call it outside graph capture")
        return {
            "main_level": self.main,
            "main_fallback": self.main_fallback,
            "aggregate_worlds": self.aggregate_worlds,
            "grid_occupancy": self.grid_occupancy,
            "aggregates": self.aggregate_count,
            "largest_aggregate": self.largest_aggregate,
            "split_aggregates": self.split_aggregates,
            "unbounded_shapes": self.unbounded_count,
            "static_shapes": self.static_shape_count,
            "static_entry_overflow": self.static_entry_overflow,
            "aggregate_pairs_max": self.aggregate_pairs_max,
            "aggregate_pair_count": int(self.aggregate_pair_count.numpy()[0]),
            "aggregate_pair_demand_max": int(self.aggregate_pair_demand_max.numpy()[0]),
            "aggregate_pair_overflow": int(self.aggregate_pair_overflow.numpy()[0]),
            "pair_demand": int(self.pair_demand.numpy()[0]),
            "pair_demand_max": int(self.pair_demand_max.numpy()[0]),
            "pair_overflow": int(self.pair_overflow.numpy()[0]),
            "extent_max": float(self.extent_max.numpy()[0]),
            "aggregate_extent_overflow": (
                0 if self.aggregate_extent_overflow is None else int(self.aggregate_extent_overflow.numpy()[0])
            ),
            "cell_size": 0.0 if self.cell_size is None else self.cell_size,
        }

    def reset_statistics(self) -> None:
        """Zero the latched values.

        These are ``pair_demand_max``, ``aggregate_pair_demand_max``, ``extent_max``
        and ``aggregate_extent_overflow``. Clearing the latch re-arms its
        device-side warning (see ``verify``) and
        :meth:`CollisionPipeline.broad_phase_statistics`'s.
        """
        self._statistics_resets += 1
        self.pair_demand_max.zero_()
        self.aggregate_pair_demand_max.zero_()
        self.extent_max.zero_()
        if self.aggregate_extent_overflow is not None:
            self.aggregate_extent_overflow.zero_()
