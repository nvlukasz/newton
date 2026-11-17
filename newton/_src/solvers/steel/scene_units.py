# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scene-wide constraint units of SolverSteel's scene mode: collection, counting sort by side, side CSR, incidence
and coloring.

A *unit* is the work item of the scene solve: one contact (its normal row and, when it has
friction, the two tangent rows), one articulation's internal unit, its block of internal rows
(drives, mimic and connect rows, joint limits and joint friction, by row family; the *limits unit* before joint
friction), or one articulation's late unit, its velocity-limit rows (a free body's too), which the scene sweeps
last. Units are
collected over the whole scene, not per world, in a deterministic slot order (a scan over flags),
sorted by ``(side_a, f, contact)`` with a counting sort by side, where ``f`` puts each side's
one-sided units first, indexed by side, and the units between two articulations are colored so
that units sharing a side get different colors and listed by color. Every launch has a fixed
shape and nothing is read back to the host, so :meth:`SceneUnits.build` and
:meth:`SceneUnits.color` can be captured in a CUDA graph (the coloring is a conditional node on
whether any unit joins two articulations, and its round loop a nested one).
:meth:`SceneUnits.statistics` reads the result back, outside capture.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .friction_patches import WARP_DETERMINISM_OFF
from .scene_rows import FAMILY_CONNECT, FAMILY_LIMIT, FAMILY_MIMIC, FAMILY_VELOCITY_LIMIT

wp.set_module_options({"enable_backward": False, **WARP_DETERMINISM_OFF})

UNIT_KEY_BITS = 63
"""Bits of a unit's sort key, a non-negative ``int64``."""

COLOR_CAP_MAX = 64
"""Largest ``color_cap``: a side's used colors are the bits of one ``uint64`` mask."""

COLOR_ROUND_MAX = 64
"""Most pair rounds of one coloring; units still uncolored then take the tail color."""

PAIR_RULES = ("speculative", "local_min")
"""Rules of the pair rounds: speculative first-fit (the default) or local minima."""

COLOR_ORDER_BLOCK = 64
"""Sorted positions per block of the counting scatter that lists the units by color."""

SCAN_TILE = 1024
"""Entries per tile of the allocation-free exclusive scan that runs inside the coloring's conditional node."""

SCAN_BLOCK_DIM = 256
"""Threads per block of that scan's tile kernels."""

_SCAN_TILE = wp.constant(SCAN_TILE)

UNIT_CONTACT_INTERNAL = -1
"""``unit_contact`` of an internal unit."""

UNIT_CONTACT_LATE = -2
"""``unit_contact`` of a late unit (an articulation's joint velocity-limit rows, a free body's velocity-limit rows)."""

_UNIT_CONTACT_LATE = wp.constant(UNIT_CONTACT_LATE)

# Rows of SceneUnits._side_counts, all cleared by one memset per sort. The first three rows are
# scanned together into the side, incidence and incoming starts.
_SIDE_UNITS = wp.constant(0)
_SIDE_INCIDENCE = wp.constant(1)
_SIDE_INCOMING = wp.constant(2)
_SIDE_ONE_SIDED = wp.constant(3)
_SIDE_CURSOR = wp.constant(4)
_SIDE_INCOMING_CURSOR = wp.constant(5)
_SIDE_LATE = wp.constant(6)
_SIDE_ROWS = 7
_SIDE_SCANNED_ROWS = 3


@wp.func
def _unit_side(art: int, articulation_response_dof_count: wp.array[int]) -> int:
    """The articulation itself when it has response DOFs, else ``-1`` (ground, static or elided kinematic)."""
    if art >= 0:
        if articulation_response_dof_count[art] > 0:
            return art
    return -1


@wp.func
def _contact_unit_sides(
    c: int,
    contact_path: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    articulation_response_dof_count: wp.array[int],
):
    """The sides ``(side_a, side_b)``, ``side_a >= side_b``, of contact ``c``'s unit; ``side_a == -1`` without one."""
    # Dropped contacts (path -1) keep stale articulation indices; they get no unit.
    if contact_path[c] < 0:
        return -1, -1
    side_0 = _unit_side(contact_art_a[c], articulation_response_dof_count)
    side_1 = _unit_side(contact_art_b[c], articulation_response_dof_count)
    return wp.max(side_0, side_1), wp.min(side_0, side_1)


@wp.func
def _is_two_sided(side_a: int, side_b: int) -> bool:
    """Whether a unit joins two articulations (not a limits unit, a ground or static contact, or a self-contact)."""
    return side_b >= 0 and side_b != side_a


@wp.func
def _unit_model_world(side_a: int, side_b: int, art_model_world: wp.array[int], world_count: int) -> int:
    """The model world a unit's rows count in: ``side_a``'s, or ``side_b``'s when ``side_a`` is global.

    Unit sides are ordered ``side_a >= side_b``, and a global (world ``-1``) side takes its other
    side's world. A unit of global sides only counts in world 0, where Steel solves global bodies.
    This is Steel's routing of a contact to its rows' world, ``_allocate_world_contact_slot``
    (``newton/_src/solvers/steel/kernels.py``): the world of its responding sides, through
    ``art_to_world``, which puts global bodies in world 0.
    """
    world = art_model_world[side_a]
    if (world < 0 or world >= world_count) and side_b >= 0:
        world = art_model_world[side_b]
    if world < 0 or world >= world_count:
        world = 0
    return world


@wp.func
def _latch_side_world(side: int, art_model_world: wp.array[int], constraint_overflow: wp.array[wp.bool]):
    """Latch the status entry of a side's model world, the last entry for a global (world ``-1``) side."""
    global_slot = constraint_overflow.shape[0] - 1
    world = art_model_world[side]
    if world < 0 or world >= global_slot:
        world = global_slot
    constraint_overflow[world] = True


@wp.func
def _scene_contact_on_free_path(
    art_a: int, art_b: int, articulation_response_dof_count: wp.array[int], is_free_rigid: wp.array[int]
) -> bool:
    """Whether Steel solves the contact between articulations ``art_a`` and ``art_b`` (``-1`` for none) on its free-body
    path: each side without response DOFs or a free rigid body (``kernels._allocate_world_contact_slot``'s ``is_mf``).
    A model without free bodies, whose ``is_mf`` is false, has no contact with rows that passes: one of its sides
    responds, so it has response DOFs and is no free body.
    """
    if art_a >= 0:
        if articulation_response_dof_count[art_a] > 0 and is_free_rigid[art_a] == 0:
            return False
    if art_b >= 0:
        if articulation_response_dof_count[art_b] > 0 and is_free_rigid[art_b] == 0:
            return False
    return True


@wp.kernel
def flag_units(
    side_count: int,
    limit_source: wp.array[int],
    limit_sides: int,
    late_source: wp.array[int],
    late_sides: int,
    contact_count: wp.array[int],
    contact_capacity: int,
    contact_path: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    # outputs
    unit_flag: wp.array[int],
):
    """Flag every entry that yields a unit (grid: ``side_count + late_sides`` + the contact buffer's capacity).

    Entry ``i < side_count`` is articulation ``i``'s internal unit, flagged when
    ``limit_source[i] > 0``; ``limit_sides`` is ``side_count`` when there is a source and ``0``
    otherwise. Entry ``side_count + i`` with ``i < late_sides`` is articulation ``i``'s late unit,
    flagged when ``late_source[i] > 0``; ``late_sides`` is ``side_count`` when there is a late source and
    ``0`` otherwise. Entry ``side_count + late_sides + c`` is contact ``c``, flagged when it is routed and
    has a responding side. Entries past the contact count are not written: an exclusive scan reads only
    the entries before its own, so they never reach a slot. A frame whose contact count exceeds
    the buffer routes no contact, so it flags none.
    """
    i = wp.tid()
    if i >= side_count + late_sides + wp.min(contact_count[0], contact_capacity):
        return
    flag = int(0)
    if i < side_count:
        if i < limit_sides:
            if limit_source[i] > 0:
                flag = 1
    elif i < side_count + late_sides:
        if late_source[i - side_count] > 0:
            flag = 1
    else:
        side_a, _side_b = _contact_unit_sides(
            i - side_count - late_sides, contact_path, contact_art_a, contact_art_b, articulation_response_dof_count
        )
        if side_a >= 0:
            flag = 1
    unit_flag[i] = flag


@wp.kernel
def collect_units(
    side_count: int,
    limit_source: wp.array[int],
    limit_sides: int,
    use_limit_rows: int,
    late_source: wp.array[int],
    late_sides: int,
    contact_count: wp.array[int],
    contact_capacity: int,
    contact_path: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    contact_slots_needed: wp.array[int],
    unit_flag: wp.array[int],
    unit_slot: wp.array[int],
    unit_max: int,
    # outputs
    unit_count: wp.array[int],
    unit_overflow: wp.array[int],
    unit_contact: wp.array[int],
    unit_side_a: wp.array[int],
    unit_side_b: wp.array[int],
    unit_rows: wp.array[int],
):
    """Write each flagged entry's unit at the slot the flags' exclusive scan gives it (grid: as :func:`flag_units`).

    The slots are therefore the internal units by articulation, the late units by articulation, then the
    contacts by index. Units whose slot is ``unit_max`` or more are counted in ``unit_overflow`` and not
    written, so the kept units are the first ``unit_max`` of that order. An internal unit is one-sided
    (``side_b == -1``), has ``unit_contact == -1`` (:data:`UNIT_CONTACT_INTERNAL`) and rows
    ``limit_source[art]`` when ``use_limit_rows`` (the internal rows of :meth:`SceneUnits.build`), else ``0``.
    A late unit is one-sided, has ``unit_contact == -2`` (:data:`UNIT_CONTACT_LATE`) and rows
    ``late_source[art]``. A contact unit has ``contact_slots_needed`` rows. Thread 0 publishes the count and
    the overflow.
    """
    i = wp.tid()
    count = side_count + late_sides + wp.min(contact_count[0], contact_capacity)
    if i == 0:
        total = int(0)
        if count > 0:
            total = unit_slot[count - 1] + unit_flag[count - 1]
        kept = wp.min(total, unit_max)
        unit_count[0] = kept
        unit_overflow[0] = total - kept
    if i >= count:
        return
    if unit_flag[i] == 0:
        return
    slot = unit_slot[i]
    if slot >= unit_max:
        return
    if i < side_count:
        rows = int(0)
        if use_limit_rows != 0:
            rows = limit_source[i]
        unit_contact[slot] = -1
        unit_side_a[slot] = i
        unit_side_b[slot] = -1
        unit_rows[slot] = rows
    elif i < side_count + late_sides:
        unit_contact[slot] = _UNIT_CONTACT_LATE
        unit_side_a[slot] = i - side_count
        unit_side_b[slot] = -1
        unit_rows[slot] = late_source[i - side_count]
    else:
        c = i - side_count - late_sides
        side_a, side_b = _contact_unit_sides(
            c, contact_path, contact_art_a, contact_art_b, articulation_response_dof_count
        )
        unit_contact[slot] = c
        unit_side_a[slot] = side_a
        unit_side_b[slot] = side_b
        unit_rows[slot] = contact_slots_needed[c]


@wp.kernel
def count_unit_sides(
    unit_count: wp.array[int],
    side_bits: int,
    contact_bits: int,
    unit_contact: wp.array[int],
    unit_side_a: wp.array[int],
    unit_side_b: wp.array[int],
    # outputs
    unit_key: wp.array[wp.int64],
    side_counts: wp.array2d[int],
    incoming_bucket: wp.array[int],
):
    """Key every unit and count it per side (grid: ``unit_max``, thread = slot).

    ``key = (side_a + 1) << (side_bits + contact_bits) | f << contact_bits | (contact + 1)`` with
    ``f = side_b + 1`` for a two-sided unit and ``f = 0`` for a one-sided one (self-contacts
    included), so a side's one-sided units sort first, the limits unit (contact ``-1``) before
    its contacts. A late unit (contact ``-2``) takes ``2**contact_bits - 1`` in place of
    ``contact + 1``, more than any contact's, so it is the last of its side's one-sided units.
    ``side_counts`` rows: units by ``side_a``, incidence entries per side (both sides of each
    two-sided unit), incoming entries by ``side_b``, one-sided units (late units included) by
    ``side_a`` and late units by ``side_a``. ``incoming_bucket`` is ``side_b`` for a two-sided unit
    and ``-1`` otherwise.
    """
    u = wp.tid()
    if u >= unit_count[0]:
        return
    side_a = unit_side_a[u]
    side_b = unit_side_b[u]
    field = int(0)
    bucket = int(-1)
    contact = unit_contact[u]
    contact_key = wp.int64(contact + 1)
    if contact == _UNIT_CONTACT_LATE:
        contact_key = (wp.int64(1) << wp.int64(contact_bits)) - wp.int64(1)
        wp.atomic_add(side_counts, _SIDE_LATE, side_a, 1)
    wp.atomic_add(side_counts, _SIDE_UNITS, side_a, 1)
    if _is_two_sided(side_a, side_b):
        field = side_b + 1
        bucket = side_b
        wp.atomic_add(side_counts, _SIDE_INCIDENCE, side_a, 1)
        wp.atomic_add(side_counts, _SIDE_INCIDENCE, side_b, 1)
        wp.atomic_add(side_counts, _SIDE_INCOMING, side_b, 1)
    else:
        wp.atomic_add(side_counts, _SIDE_ONE_SIDED, side_a, 1)
    unit_key[u] = (
        (wp.int64(side_a + 1) << wp.int64(side_bits + contact_bits))
        | (wp.int64(field) << wp.int64(contact_bits))
        | contact_key
    )
    incoming_bucket[u] = bucket


@wp.kernel
def publish_side_starts(
    side_count: int,
    side_counts: wp.array2d[int],
    side_scan: wp.array[int],
    # outputs
    incidence_start: wp.array[int],
    incoming_start: wp.array[int],
    side_coupled: wp.array[int],
):
    """Split the joint exclusive scan of the first three count rows into starts (grid: ``side_count + 1``).

    Row ``r``'s entries are scanned after every entry of the rows before it, so subtracting the
    row's first scanned entry gives its own exclusive scan. The units' row needs no correction:
    ``side_start`` is a view of the scan's first row. ``side_coupled[s]`` is ``1`` when side
    ``s`` has an incidence entry.
    """
    s = wp.tid()
    sides = side_count + 1
    incidence_start[s] = side_scan[sides + s] - side_scan[sides]
    incoming_start[s] = side_scan[2 * sides + s] - side_scan[2 * sides]
    if s < side_count:
        coupled = int(0)
        if side_counts[_SIDE_INCIDENCE, s] > 0:
            coupled = 1
        side_coupled[s] = coupled


@wp.kernel
def flag_two_sided(
    side_count: int,
    incidence_start: wp.array[int],
    # outputs
    has_two_sided: wp.array[int],
):
    """Set ``has_two_sided[0]`` to ``1`` when any unit joins two articulations, else ``0`` (grid: 1)."""
    flag = int(0)
    if incidence_start[side_count] > 0:
        flag = 1
    has_two_sided[0] = flag


@wp.kernel
def bucket_fill(
    element_count: wp.array[int],
    bucket_of: wp.array[int],
    bucket_start: wp.array[int],
    # in/out
    bucket_cursor: wp.array[int],
    # outputs
    bucket_items: wp.array[int],
):
    """Drop each element into its bucket's range, in arrival order (grid: ``unit_max``).

    Element ``e < element_count[0]`` with ``bucket_of[e] >= 0`` is written at
    ``bucket_start[b] + j``, ``j`` from an atomic cursor that starts at zero. The order within a
    bucket is therefore arbitrary; :func:`bucket_rank` replaces it with the key order.
    """
    e = wp.tid()
    if e >= element_count[0]:
        return
    b = bucket_of[e]
    if b < 0:
        return
    bucket_items[bucket_start[b] + wp.atomic_add(bucket_cursor, b, 1)] = e


@wp.kernel
def bucket_rank(
    bucket_count: int,
    bucket_start: wp.array[int],
    bucket_of: wp.array[int],
    bucket_items: wp.array[int],
    key: wp.array[wp.int64],
    write_position: int,
    # outputs
    order: wp.array[int],
    position: wp.array[int],
):
    """Place each bucketed element at its bucket's start plus its rank by key (grid: ``unit_max``).

    Thread ``i < bucket_start[bucket_count]`` takes the element at ``bucket_items[i]``; its rank
    is the number of elements of its bucket with a smaller key, counted over the bucket's range
    (``O(bucket size)`` per element). Keys are distinct, so the ranks of a bucket are a
    permutation and the result does not depend on :func:`bucket_fill`'s arrival order. With
    ``write_position``, ``position[e]`` receives the element's place in ``order``.
    """
    i = wp.tid()
    if i >= bucket_start[bucket_count]:
        return
    e = bucket_items[i]
    b = bucket_of[e]
    k = key[e]
    start = bucket_start[b]
    rank = int(0)
    for j in range(start, bucket_start[b + 1]):
        if key[bucket_items[j]] < k:
            rank += 1
    order[start + rank] = e
    if write_position != 0:
        position[e] = start + rank


@wp.kernel
def scatter_incidence(
    side_count: int,
    unit_count: wp.array[int],
    unit_order: wp.array[int],
    unit_position: wp.array[int],
    unit_side_a: wp.array[int],
    unit_side_b: wp.array[int],
    side_counts: wp.array2d[int],
    side_start: wp.array[int],
    incidence_start: wp.array[int],
    incoming_start: wp.array[int],
    incoming_slot: wp.array[int],
    # outputs
    incidence_unit: wp.array[int],
):
    """Write each two-sided unit's sorted position under both of its sides (grid: ``unit_max``).

    A side's entries are its own two-sided units (``side_a`` == side), which follow its one-sided
    prefix contiguously in its sorted segment, then the units that reach it as ``side_b``, which
    the second counting pass listed by ``side_b`` in key order (``incoming_slot``). Both groups are
    in position order and every unit of the first precedes every unit of the second (its
    ``side_a`` is larger), so each side's list is sorted by position.
    """
    i = wp.tid()
    if i < unit_count[0]:
        u = unit_order[i]
        side_a = unit_side_a[u]
        side_b = unit_side_b[u]
        if _is_two_sided(side_a, side_b):
            run_start = side_start[side_a] + side_counts[_SIDE_ONE_SIDED, side_a]
            incidence_unit[incidence_start[side_a] + i - run_start] = i
    if i < incoming_start[side_count]:
        v = incoming_slot[i]
        side = unit_side_b[v]
        outgoing = side_counts[_SIDE_INCIDENCE, side] - side_counts[_SIDE_INCOMING, side]
        incidence_unit[incidence_start[side] + outgoing + i - incoming_start[side]] = unit_position[v]


@wp.func
def _lowest_clear_bit(used: wp.uint64) -> int:
    """Index of the lowest clear bit of ``used``; ``64`` when every bit is set."""
    free = ~used
    if free == wp.uint64(0):
        return 64
    # Binary search for the lowest set bit of ``free``; no shift by 64 can occur.
    bit = int(0)
    if (free & wp.uint64(0xFFFFFFFF)) == wp.uint64(0):
        free = free >> wp.uint64(32)
        bit += 32
    if (free & wp.uint64(0xFFFF)) == wp.uint64(0):
        free = free >> wp.uint64(16)
        bit += 16
    if (free & wp.uint64(0xFF)) == wp.uint64(0):
        free = free >> wp.uint64(8)
        bit += 8
    if (free & wp.uint64(0xF)) == wp.uint64(0):
        free = free >> wp.uint64(4)
        bit += 4
    if (free & wp.uint64(0x3)) == wp.uint64(0):
        free = free >> wp.uint64(2)
        bit += 2
    if (free & wp.uint64(0x1)) == wp.uint64(0):
        bit += 1
    return bit


@wp.func
def _first_uncolored_on_side(
    p: int,
    side: int,
    incidence_start: wp.array[int],
    incidence_unit: wp.array[int],
    unit_order: wp.array[int],
    unit_color: wp.array[int],
) -> bool:
    """Whether no two-sided unit of ``side`` before position ``p`` was uncolored when the round began."""
    first = bool(True)
    for i in range(incidence_start[side], incidence_start[side + 1]):
        q = incidence_unit[i]
        # A side's incidence list is sorted by position, so the walk stops at ``p`` itself.
        if q >= p:
            break
        # Negative: uncolored, or colored in this round (still pending until count_uncolored).
        if unit_color[unit_order[q]] < 0:
            first = False
            break
    return first


@wp.kernel
def color_pair_round(
    unit_count: wp.array[int],
    unit_order: wp.array[int],
    unit_side_a: wp.array[int],
    unit_side_b: wp.array[int],
    incidence_start: wp.array[int],
    incidence_unit: wp.array[int],
    color_cap: int,
    # in/out
    unit_color: wp.array[int],
    side_mask: wp.array[wp.uint64],
    rounds: wp.array[int],
    uncolored: wp.array[int],
    keep_going: wp.array[int],
):
    """Color the two-sided units that are local minima of the uncolored ones (grid: ``unit_max``).

    Thread ``p`` is the unit at sorted position ``p``; one-sided units are never colored. An
    uncolored two-sided unit is a local minimum when no uncolored two-sided unit at a smaller
    position shares either of its sides; it takes the lowest color free in both sides' masks, or
    the tail color ``color_cap`` when no color below the cap is free, and sets that bit in both
    masks. The new color is written pending (``-2 - color``) and made final by
    :func:`count_uncolored`, so every thread's minimum test sees the round's starting state
    whatever the other threads have written.

    The units of one side are all neighbors, so a side has at most one local minimum per round:
    a side's mask is read and written by one thread per round, and plain stores suffice.
    Thread 0 also counts the round and clears ``uncolored`` and ``keep_going`` for
    :func:`count_uncolored`.
    """
    p = wp.tid()
    if p == 0:
        rounds[0] = rounds[0] + 1
        uncolored[0] = 0
        keep_going[0] = 0
    if p >= unit_count[0]:
        return
    u = unit_order[p]
    if unit_color[u] != -1:
        return
    side_a = unit_side_a[u]
    side_b = unit_side_b[u]
    if not _is_two_sided(side_a, side_b):
        return
    if not _first_uncolored_on_side(p, side_a, incidence_start, incidence_unit, unit_order, unit_color):
        return
    if not _first_uncolored_on_side(p, side_b, incidence_start, incidence_unit, unit_order, unit_color):
        return
    bit = _lowest_clear_bit(side_mask[side_a] | side_mask[side_b])
    color = color_cap
    if bit < color_cap:
        color = bit
        flag = wp.uint64(1) << wp.uint64(bit)
        side_mask[side_a] = side_mask[side_a] | flag
        side_mask[side_b] = side_mask[side_b] | flag
    unit_color[u] = -2 - color


@wp.kernel
def color_propose(
    side_start: wp.array[int],
    one_sided_count: wp.array[int],
    unit_order: wp.array[int],
    unit_side_b: wp.array[int],
    unit_color: wp.array[int],
    side_mask: wp.array[wp.uint64],
    color_cap: int,
    # outputs
    proposal: wp.array[int],
    rounds: wp.array[int],
    uncolored: wp.array[int],
    keep_going: wp.array[int],
):
    """Propose a color for every uncolored two-sided unit, first-fit per owning side (grid: sides).

    The thread of side ``s`` owns the two-sided units with ``side_a == s``, the positions after the
    side's one-sided prefix, and walks them in sorted order, as one block of a speculative
    (Gebremedhin-Manne) coloring: each takes the lowest color free in ``side_mask[s]``, in the
    colors this thread proposed before it and in ``side_mask[side_b]``, so the units of one owner
    never propose the same color. The proposal goes to ``proposal[p]``; every other two-sided
    position of the side gets ``-1``, and the one-sided positions keep the ``-1`` they were given
    before the first round. When no color below ``color_cap`` is free, the unit proposes the tail
    color ``color_cap`` only if the committed masks alone (``side_mask[s] | side_mask[side_b]``)
    have none free either; otherwise this round's proposals of its owner are what block it, some
    of which may lose, so it proposes nothing (``-1``) and retries next round. Units of different
    owners may clash on a shared side; :func:`color_resolve` settles that by position. Thread 0
    also counts the round and clears ``uncolored`` and ``keep_going`` for :func:`count_uncolored`.

    Progress, at every ``color_cap``: the uncolored unit at the lowest sorted position is the
    first uncolored unit of its owner, so ``own`` is still the committed mask when it is
    reached. It therefore proposes either a free color or, when its committed masks are full,
    the tail; it never defers. No unit at a smaller position proposes anything (every one is
    colored), so :func:`color_resolve` accepts its proposal. Each round thus colors or tails
    at least one unit, and the loop ends when none is left, or after
    :data:`COLOR_ROUND_MAX` rounds, when :func:`color_tail_fallback` gives the rest the tail.
    """
    s = wp.tid()
    if s == 0:
        rounds[0] = rounds[0] + 1
        uncolored[0] = 0
        keep_going[0] = 0
    committed = side_mask[s]
    own = committed
    for p in range(side_start[s] + one_sided_count[s], side_start[s + 1]):
        u = unit_order[p]
        color = int(-1)
        if unit_color[u] == -1:
            other = side_mask[unit_side_b[u]]
            bit = _lowest_clear_bit(own | other)
            if bit < color_cap:
                color = bit
                own = own | (wp.uint64(1) << wp.uint64(bit))
            elif _lowest_clear_bit(committed | other) >= color_cap:
                color = color_cap
        proposal[p] = color


@wp.func
def _first_proposal_on_side(
    p: int,
    color: int,
    side: int,
    incidence_start: wp.array[int],
    incidence_unit: wp.array[int],
    proposal: wp.array[int],
) -> bool:
    """Whether no two-sided unit of ``side`` before position ``p`` proposed ``color`` this round."""
    first = bool(True)
    for i in range(incidence_start[side], incidence_start[side + 1]):
        q = incidence_unit[i]
        # A side's incidence list is sorted by position, so the walk stops at ``p`` itself.
        if q >= p:
            break
        if proposal[q] == color:
            first = False
            break
    return first


@wp.kernel
def color_resolve(
    unit_count: wp.array[int],
    unit_order: wp.array[int],
    unit_side_a: wp.array[int],
    unit_side_b: wp.array[int],
    incidence_start: wp.array[int],
    incidence_unit: wp.array[int],
    proposal: wp.array[int],
    color_cap: int,
    # in/out
    unit_color: wp.array[int],
    side_mask: wp.array[wp.uint64],
):
    """Accept the proposals that no earlier unit on either side made too (grid: ``unit_max``).

    Thread ``p`` is the unit at sorted position ``p``; one-sided positions hold no proposal. Its
    proposal wins unless a unit at a smaller position that shares a side proposed the same
    color; a loser stays uncolored for the next round (losing to a unit that itself loses only
    costs a round). A winner takes its color and sets that bit in both sides' masks; several
    winners may share a side with different colors, hence ``atomic_or``. A proposal of
    ``color_cap`` takes the tail color at once and sets no bit. The masks are read only by
    :func:`color_propose`, a launch earlier.
    """
    p = wp.tid()
    if p >= unit_count[0]:
        return
    color = proposal[p]
    if color < 0:
        return
    u = unit_order[p]
    if color >= color_cap:
        unit_color[u] = color_cap
        return
    side_a = unit_side_a[u]
    side_b = unit_side_b[u]
    if not _first_proposal_on_side(p, color, side_a, incidence_start, incidence_unit, proposal):
        return
    if not _first_proposal_on_side(p, color, side_b, incidence_start, incidence_unit, proposal):
        return
    unit_color[u] = color
    flag = wp.uint64(1) << wp.uint64(color)
    wp.atomic_or(side_mask, side_a, flag)
    wp.atomic_or(side_mask, side_b, flag)


@wp.kernel
def count_uncolored(
    unit_count: wp.array[int],
    unit_side_a: wp.array[int],
    unit_side_b: wp.array[int],
    rounds: wp.array[int],
    round_max: int,
    # in/out
    unit_color: wp.array[int],
    uncolored: wp.array[int],
    keep_going: wp.array[int],
):
    """Make the round's pending colors final and count the two-sided units still uncolored (grid: ``unit_max``).

    Thread ``u`` is slot ``u``. One-sided units stay at ``-1`` and are not counted. ``uncolored``
    and ``keep_going`` start at 0 (cleared by :meth:`SceneUnits.color` before the first count and
    by the round's first kernel, :func:`color_propose` or :func:`color_pair_round`);
    ``keep_going`` becomes 1 when some two-sided unit is still uncolored and fewer than
    ``round_max`` rounds have run. Pending colors exist only under the ``local_min`` rule.
    """
    u = wp.tid()
    if u >= unit_count[0]:
        return
    color = unit_color[u]
    if color <= -2:
        unit_color[u] = -2 - color
    elif color == -1:
        if _is_two_sided(unit_side_a[u], unit_side_b[u]):
            wp.atomic_add(uncolored, 0, 1)
            if rounds[0] < round_max:
                keep_going[0] = 1


@wp.kernel
def color_tail_fallback(
    unit_count: wp.array[int],
    unit_side_a: wp.array[int],
    unit_side_b: wp.array[int],
    color_cap: int,
    # in/out
    unit_color: wp.array[int],
):
    """Give the tail color ``color_cap`` to every two-sided unit the round cap left uncolored (grid: ``unit_max``)."""
    u = wp.tid()
    if u < unit_count[0] and unit_color[u] == -1:
        if _is_two_sided(unit_side_a[u], unit_side_b[u]):
            unit_color[u] = color_cap


@wp.kernel
def count_colors(
    unit_count: wp.array[int],
    unit_order: wp.array[int],
    unit_color: wp.array[int],
    block: int,
    block_count: int,
    # in/out
    color_block_count: wp.array[int],
):
    """Count the colored units per color and block of ``block`` sorted positions (grid: ``unit_max``).

    ``color_block_count[c * block_count + k]`` counts the units of color ``c`` at positions
    ``[k * block, (k + 1) * block)``: color-major, so its exclusive scan gives every (color,
    block) pair the first position of its run in the order by color. Uncolored (one-sided) units
    are not counted.
    """
    p = wp.tid()
    if p >= unit_count[0]:
        return
    color = unit_color[unit_order[p]]
    if color < 0:
        return
    wp.atomic_add(color_block_count, color * block_count + p // block, 1)


@wp.kernel
def scatter_by_color(
    unit_count: wp.array[int],
    unit_order: wp.array[int],
    unit_color: wp.array[int],
    block: int,
    block_count: int,
    color_slots: int,
    # in/out
    color_block_cursor: wp.array[int],
    # outputs
    color_count_per: wp.array[int],
    color_offsets: wp.array[int],
    order_by_color: wp.array[int],
):
    """List the colored units' sorted positions by color, in position order within a color (grid: blocks).

    ``color_block_cursor`` starts as the exclusive scan of :func:`count_colors`' counts. The
    thread of block ``k`` walks the block's positions in order, skips the uncolored (one-sided)
    units and writes each other at its color's cursor for the block, which only this thread
    advances. The destination of position ``p`` is therefore the units of lower colors, plus
    those of its color in earlier blocks, plus those of its color earlier in its block: its
    stable rank, with no atomics. Thread 0 first reads the per-color counts and offsets from the
    scan's block-0 entries, which only it advances.
    """
    k = wp.tid()
    if k == 0:
        for color in range(color_slots):
            start = color_block_cursor[color * block_count]
            color_offsets[color] = start
            if color + 1 < color_slots:
                color_count_per[color] = color_block_cursor[(color + 1) * block_count] - start
            else:
                color_count_per[color] = 0
    end = wp.min((k + 1) * block, unit_count[0])
    for p in range(k * block, end):
        color = unit_color[unit_order[p]]
        if color >= 0:
            cursor = color * block_count + k
            dest = color_block_cursor[cursor]
            color_block_cursor[cursor] = dest + 1
            order_by_color[dest] = p


@wp.kernel
def scan_tiles(
    values: wp.array[int],
    # outputs
    scanned: wp.array[int],
    tile_totals: wp.array[int],
):
    """Scan each tile of :data:`SCAN_TILE` entries exclusively and record its total (tiled grid: tiles).

    The first of the three launches of :meth:`SceneUnits._scan_exclusive`, an exclusive scan that
    allocates nothing, so it can run inside a conditional graph node where ``wp.utils.array_scan``
    (whose scratch is a graph allocation) cannot. The last tile is padded with zeros.
    """
    k = wp.tid()
    tile = wp.tile_load(values, shape=_SCAN_TILE, offset=k * _SCAN_TILE)
    wp.tile_store(scanned, wp.tile_scan_exclusive(tile), offset=k * _SCAN_TILE)
    wp.tile_store(tile_totals, wp.tile_sum(tile), offset=k)


@wp.kernel
def scan_tile_totals(
    tile_totals: wp.array[int],
    tile_count: int,
    # outputs
    tile_offsets: wp.array[int],
):
    """Scan the tile totals exclusively into each tile's offset, in one block (tiled grid: 1)."""
    carry = int(0)
    for start in range(0, tile_count, _SCAN_TILE):
        tile = wp.tile_load(tile_totals, shape=_SCAN_TILE, offset=start)
        ones = wp.tile_ones(shape=_SCAN_TILE, dtype=int)
        wp.tile_store(tile_offsets, wp.tile_scan_exclusive(tile) + ones * carry, offset=start)
        carry = carry + wp.tile_extract(wp.tile_sum(tile), 0)


@wp.kernel
def add_tile_offsets(
    count: int,
    tile_offsets: wp.array[int],
    # in/out
    scanned: wp.array[int],
):
    """Add each entry's tile offset to its scan within the tile (grid: entries)."""
    i = wp.tid()
    if i < count:
        scanned[i] = scanned[i] + tile_offsets[i // _SCAN_TILE]


class SceneUnits:
    """The scene's constraint units, sorted and indexed by side, with the two-sided units colored.

    A *side* is an articulation index of Steel's plan (free rigid bodies are 6-DOF articulations)
    with response DOFs; a ground, static or elided kinematic endpoint is side ``-1``. A unit's
    sides are ordered ``side_a >= side_b``: a one-sided unit has ``side_b == -1`` or, for a
    contact between two links of one articulation, ``side_b == side_a``; a *two-sided* unit has
    ``0 <= side_b < side_a``. Every unit has ``side_a >= 0``.

    **Slots.** Units fill slots ``[0, unit_count)`` in a deterministic order: :func:`flag_units`
    flags each internal unit, each late unit and each contact that yields a unit, an exclusive
    ``array_scan`` of the flags gives each its slot, and :func:`collect_units` writes it there. The slots
    are therefore the internal units by articulation, the late units by articulation, then the contacts by
    index: after a build, contact ``c``'s unit, when it has one, is slot ``_unit_slot[contact_entry + c]``
    (``contact_entry``, an ``int``, the internal and late entries before the contacts'). Per slot (``int32``,
    ``[unit_max]``):

    - ``unit_contact``: the contact index, ``-1`` for an internal unit (:data:`UNIT_CONTACT_INTERNAL`) or
      ``-2`` for a late unit (:data:`UNIT_CONTACT_LATE`).
    - ``unit_side_a``, ``unit_side_b``: the unit's sides.
    - ``unit_rows``: rows of the unit: ``contact_slots_needed`` of a contact (1 or 3); for an
      internal unit, its articulation's ``internal_rows`` when :meth:`build` is given them, else ``0``; for a
      late unit, its articulation's ``late_rows``.
    - ``unit_key`` (``int64``): the sort key
      ``(side_a + 1) << (side_bits + contact_bits) | f << contact_bits | (contact + 1)``, with
      ``f = side_b + 1`` for a two-sided unit and ``f = 0`` for a one-sided one; a late unit takes
      ``2**contact_bits - 1`` in place of ``contact + 1``. The field widths
      are fixed at construction (:meth:`key_bits`); construction raises :class:`ValueError` when
      the key would need more than :data:`UNIT_KEY_BITS` bits. Keys are distinct.
    - ``unit_color``: see the coloring below.

    Counters (``int32``, ``[1]``): ``unit_count = min(total, unit_max)`` and
    ``unit_overflow = max(total - unit_max, 0)``. On overflow the kept units are the first
    ``unit_max`` slots of that order, the same on every run; no slot past ``unit_count`` is written.

    **Sorted positions.** ``unit_order`` (``int32``, ``[unit_max]``): the unit at sorted position
    ``p < unit_count`` is slot ``unit_order[p]``, so the sorted keys ``unit_key[unit_order[:count]]``
    increase strictly. Sorting is a counting sort by ``side_a``: :func:`count_unit_sides` counts
    per side, one scan gives the side starts, :func:`bucket_fill` drops each slot into its side's
    range and :func:`bucket_rank` moves it to its rank by key within the side. The sorted order,
    and everything indexed by position below, is a function of the unit set alone, whatever slots
    the units occupy. Consumers that hold a position read a unit's fields through ``unit_order``;
    ``_unit_position`` (``int32``, ``[unit_max]``, per slot) is its inverse. The rank costs
    ``O(n)`` per unit of a side with ``n`` units: cheap for the tens of units of a humanoid or a
    drum, quadratic for a side with thousands.

    **Per side** (``int32``, ``side_count`` = articulations of the plan):

    - ``side_start`` (``[side_count + 1]``): CSR over sorted positions by ``side_a``. Positions
      ``[side_start[s], side_start[s + 1])`` hold the units with ``side_a == s`` in key order.
      ``side_start[side_count] == unit_count``.
    - ``one_sided_count`` (``[side_count]``): the side's one-sided units, self-contacts and its late unit
      included. They are the *prefix* ``[side_start[s], side_start[s] + one_sided_count[s])`` of the side's
      segment, the internal unit first, then the contacts by index, then the late unit; the side's two-sided
      units (``side_b < s``) follow, ordered by ``(side_b, contact)``.
    - ``late_count`` (``[side_count]``): ``1`` when the side has a late unit, the last of its one-sided
      prefix, else ``0``. The scene places late units after every other unit (they hold the rows Steel sweeps
      last, :class:`~newton._src.solvers.steel.scene_solve.SceneSolve`); to the coloring and the incidence
      they are one-sided units like any other.
    - ``incidence_start`` (``[side_count + 1]``) with ``incidence_unit`` (``int32``,
      ``[2 * unit_max]``): every two-sided unit listed once under each of its sides, as its sorted
      position, ordered by side and then position. Entries
      ``[incidence_start[s], incidence_start[s + 1])`` belong to side ``s``; entries past
      ``incidence_start[side_count]`` (twice the two-sided units) are stale. One-sided units
      appear only in ``side_start``.
    - ``side_coupled`` (``[side_count]``): ``1`` when the side has a two-sided unit
      (``incidence_start[s + 1] > incidence_start[s]``), else ``0``.
    - ``has_two_sided`` (``int32``, ``[1]``): ``1`` when any unit is two-sided, the condition of
      the coloring.

    Internal units come from :meth:`build`'s ``internal_rows`` (one per articulation with
    ``internal_rows[art] > 0``) or, without them, exist only with ``enable_joint_limits``, one per
    articulation flagged in ``art_has_limits`` (``int32``, ``[side_count]``, see
    :meth:`limit_bearing_articulations`): free rigid bodies and articulations without a finite
    limit get none. ``limit_unit_count`` is the number of flagged articulations, and
    ``internal_unit_count`` the number of articulations that can have an internal unit
    (:meth:`internal_bearing_articulations`): the limit-bearing ones, with ``drive_mode="rows"``
    those with a DOF of a drive's joint type, and with ``enable_joint_friction`` those with a DOF whose
    joint supports friction. Late units come from :meth:`build`'s ``late_rows`` (one per articulation with
    ``late_rows[art] > 0``); ``late_unit_count`` is the number of articulations that can have one
    (:meth:`late_bearing_articulations`, when the scene solve builds late rows, ``_scene_late_rows_enabled``).

    **Coloring** (:meth:`color`, after :meth:`build`). Only two-sided units are colored; one-sided
    units are solved per side in order and keep ``unit_color == -1``. ``unit_color`` (``int32``,
    ``[unit_max]``) is per slot, so the color of the unit at position ``p`` is
    ``unit_color[unit_order[p]]``. It is ``-1`` before coloring, for one-sided units and in
    unused slots; after it every two-sided unit has a color in ``[0, color_cap)`` or the *tail*
    color ``color_cap`` (the solver's ``scene_color_cap`` kernel override, at most and by default
    :data:`COLOR_CAP_MAX`). Two two-sided units
    that share a side never share a color below ``color_cap``; tail units may share a side and are
    meant to run one at a time after the colors. ``side_mask``, the counters, ``color_count_per``
    and ``color_offsets`` are cleared first; everything else runs under
    ``wp.capture_if(has_two_sided)``, so a step without two-sided units colors nothing.

    - Pair rounds color the two-sided units, starting from empty ``side_mask`` words, in a
      ``wp.capture_while`` loop on ``keep_going``, each round ending with :func:`count_uncolored`,
      until no two-sided unit is uncolored or :data:`COLOR_ROUND_MAX` rounds have run.
      ``pair_rule`` (one of :data:`PAIR_RULES`, the solver's ``scene_pair_rule`` kernel override) picks the
      round:

      - ``"speculative"`` (default): speculative first-fit. :func:`color_propose` lets each side
        propose colors for the uncolored two-sided units it owns (``side_a``), first-fit in sorted
        order over both sides' masks, and :func:`color_resolve` accepts every proposal that no
        unit at a smaller position on either side made too. Many units of one side are colored
        per round, so the rounds grow with the clashes between owners, not with a side's units.
      - ``"local_min"``: :func:`color_pair_round` colors every two-sided unit that is the first
        uncolored one, by position, on both of its sides, with the lowest color free in both
        sides' masks. A side gets at most one color per round, so a side with ``n`` two-sided
        units needs at least ``n`` rounds.
    - Tail: a unit takes ``color_cap`` and sets no mask bit when the colors already committed on
      its sides leave none below the cap free: a full ``side_mask[side_a] | side_mask[side_b]`` at
      the round's start. Under ``"speculative"``, a unit that only its owner's proposals of the
      same round keep from a free color proposes nothing and retries next round. After the loop,
      :func:`color_tail_fallback` gives ``color_cap`` to any two-sided unit the round cap left
      uncolored.

    ``side_mask`` (``uint64``, ``[side_count]``): bit ``c`` is set when a two-sided unit of the
    side has color ``c < color_cap``. Counters (``int32``, ``[1]``): ``rounds`` (pair rounds run),
    ``keep_going`` (the loop's condition) and ``uncolored``: after :meth:`color`, the number of
    two-sided units the round cap left uncolored for the tail fallback, ``0`` when the rounds
    finished. The speculative rule's proposals are in ``_color_proposal`` (``int32``,
    ``[unit_max]``, by sorted position, ``-1`` at one-sided positions). Rounds, colors and masks
    depend only on the sorted order.

    Order by color (the end of :meth:`color`), over ``color_cap + 2`` color slots: slots
    ``0..color_cap - 1`` are the colors, slot ``color_cap`` the tail and slot ``color_cap + 1``
    counts nothing, so its offset is the total.

    - ``color_count_per`` (``int32``, ``[color_cap + 2]``): two-sided units per color slot.
    - ``color_offsets`` (``int32``, ``[color_cap + 2]``): its exclusive scan; the units of color
      ``c`` are ``order_by_color[color_offsets[c]:color_offsets[c + 1]]``, the tail units start
      at ``color_offsets[color_cap]`` and ``color_offsets[color_cap + 1]`` is the number of
      two-sided units. Both are all zero when no unit is two-sided.
    - ``order_by_color`` (``int32``, ``[unit_max]``): the two-sided units' *sorted positions*
      (indices into ``unit_order``, the convention of ``incidence_unit``), grouped by color and in
      increasing position within a color (a stable order); the unit at ``order_by_color[i]`` is
      slot ``unit_order[order_by_color[i]]``. Entries from the two-sided count on are not written.

    The order is a counting scatter in blocks of :data:`COLOR_ORDER_BLOCK` positions:
    :func:`count_colors` counts per (color, block), one exclusive scan over those counts,
    color-major, gives each (color, block) the start of its run (an allocation-free tile scan,
    :meth:`_scan_exclusive`, since it runs inside the conditional node), and
    :func:`scatter_by_color` walks each block in position order on one thread. It uses no
    atomic to place a unit, so the order is a function of the sorted units and colors alone.

    **Capacity.** ``unit_max`` is, in order of precedence: ``unit_max`` (``scene_unit_max``);
    ``row_max // contact_rows + internal_unit_count + late_unit_count`` when a row budget ``row_max`` is
    given (a contact unit has at least ``contact_rows`` rows in the scene solve, three, or one when contacts
    past ``contact_friction_gap_max`` keep their normal row alone, so no further unit could receive
    rows); or, when both are ``None``, the first build's ``contacts.rigid_contact_max`` plus its
    possible internal and late units, allocated then. A build under graph capture therefore needs the
    capacity allocated at construction or by an earlier eager build.

    The internal units count toward ``unit_max`` by articulation, not by row: an articulation has at
    most one internal unit, holding every row of its row families (its drives, its joint limits and
    its joint friction), so the units never outnumber ``internal_unit_count`` whatever rows the step's
    poses, gains, targets and coefficients give them. ``internal_unit_count`` counts every articulation
    that could have one: drives and joint friction count by joint type (a gain or a coefficient edited
    in place may turn a row on at any step), joint limits by finite bound
    (the solver's ``_scene_family_bearing``). Their rows count toward the row budget like a contact's:
    the scene places them in row order and drops whole units past ``row_max``
    (:class:`~newton._src.solvers.steel.scene_solve.SceneSolve`), since Steel's per-world budget
    ``articulation_constraint_max``, which the scene pools, holds them too. The late units count the same way,
    one per articulation that can bear joint or free-body velocity-limit rows (``late_unit_count``), whatever its rows.
    """

    def __init__(self, solver, unit_max: int | None = None, *, row_max: int | None = None, contact_rows: int = 3):
        model = solver.model
        self.device = model.device
        self.side_count = int(model.articulation_count)
        # The solver's contact capacity bounds every contact buffer its step accepts.
        self.contact_max = int(solver._max_contacts_alloc)
        self.side_bits, self.contact_bits = self.key_bits(self.side_count, self.contact_max)
        self.built = False
        self._last_contacts = None
        self.limit_units = bool(solver._enable_joint_limits)
        self.art_has_limits = None
        self.limit_unit_count = 0
        if self.limit_units and self.side_count:
            has_limits = self.limit_bearing_articulations(solver)
            self.art_has_limits = wp.array(has_limits, dtype=wp.int32, device=self.device)
            self.limit_unit_count = int(np.count_nonzero(has_limits))
        # Without joint friction, drive rows, mimic rows or loop closures the internal units are the limits units.
        self.internal_unit_count = self.limit_unit_count
        bearing = solver._scene_family_bearing
        bilateral = False
        if bearing is not None:
            # The mimic and connect families' bits (the scene solve's alone).
            bilateral = bool(np.any(bearing & ((1 << FAMILY_MIMIC) | (1 << FAMILY_CONNECT))))
        if (solver._enable_joint_friction or solver._has_drive_rows or bilateral) and self.side_count:
            self.internal_unit_count = int(np.count_nonzero(self.internal_bearing_articulations(solver)))
        # Joint and free-body velocity limits: at most one late unit per articulation that can bear their rows (the
        # scene solve's; units built without the scene solve build none).
        self.late_unit_count = 0
        if solver._scene_late_rows_enabled and self.side_count:
            self.late_unit_count = int(np.count_nonzero(self.late_bearing_articulations(solver)))
        # A build's late entries, one per articulation, exist only when some articulation can bear late rows.
        self._late_sides = self.side_count if self.late_unit_count else 0

        device = self.device
        sides = self.side_count + 1
        self.unit_count = wp.zeros(1, dtype=wp.int32, device=device)
        self.unit_overflow = wp.zeros(1, dtype=wp.int32, device=device)
        self._no_contacts = wp.zeros(1, dtype=wp.int32, device=device)
        # Per side, one memset per sort: the counts (rows _SIDE_*) and the two bucket cursors.
        self._side_counts = wp.zeros((_SIDE_ROWS, sides), dtype=wp.int32, device=device)
        self._side_scan_input = self._side_counts.flatten()[: _SIDE_SCANNED_ROWS * sides]
        self._side_scan = wp.zeros(_SIDE_SCANNED_ROWS * sides, dtype=wp.int32, device=device)
        self.side_start = self._side_scan[:sides]
        self.one_sided_count = self._side_counts[_SIDE_ONE_SIDED][: self.side_count]
        self.late_count = self._side_counts[_SIDE_LATE][: self.side_count]
        self.incidence_start = wp.zeros(sides, dtype=wp.int32, device=device)
        self._incoming_start = wp.zeros(sides, dtype=wp.int32, device=device)
        self.side_coupled = wp.zeros(self.side_count, dtype=wp.int32, device=device)
        self.has_two_sided = wp.zeros(1, dtype=wp.int32, device=device)

        overrides = solver._kernel_overrides
        self.color_cap = int(overrides.get("scene_color_cap", COLOR_CAP_MAX))
        if not 1 <= self.color_cap <= COLOR_CAP_MAX:
            raise ValueError(f"scene_color_cap must be between 1 and {COLOR_CAP_MAX}, got {self.color_cap}")
        self.pair_rule = overrides.get("scene_pair_rule", "speculative")
        if self.pair_rule not in PAIR_RULES:
            raise ValueError(f"scene_pair_rule must be one of {PAIR_RULES}, got {self.pair_rule!r}")
        self.side_mask = wp.zeros(self.side_count, dtype=wp.uint64, device=device)
        # One buffer, cleared by one memset per coloring, viewed as the three counters.
        self._color_counters = wp.zeros(3, dtype=wp.int32, device=device)
        self.rounds = self._color_counters[0:1]
        self.uncolored = self._color_counters[1:2]
        self.keep_going = self._color_counters[2:3]
        self.color_slots = self.color_cap + 2
        # One buffer, cleared by one memset per coloring, viewed as the counts and the offsets.
        self._color_order_counts = wp.zeros(2 * self.color_slots, dtype=wp.int32, device=device)
        self.color_count_per = self._color_order_counts[: self.color_slots]
        self.color_offsets = self._color_order_counts[self.color_slots :]

        self.unit_max = None
        if unit_max is None and row_max is not None:
            if int(row_max) < 0:
                raise ValueError(f"row_max must be at least 0, got {row_max}")
            unit_max = max(int(row_max) // int(contact_rows) + self.internal_unit_count + self.late_unit_count, 1)
        if unit_max is not None:
            self._allocate(int(unit_max))

    @staticmethod
    def key_bits(side_count: int, contact_max: int) -> tuple[int, int]:
        """Return the widths ``(side_bits, contact_bits)`` of the sort key's fields.

        ``side_bits`` holds ``side_count + 1`` (``side_a + 1`` and ``f <= side_b + 1``) and
        ``contact_bits`` holds ``contact_max + 1`` (``contact + 1``). Raises
        :class:`ValueError` when ``contact_bits + 2 * side_bits`` exceeds :data:`UNIT_KEY_BITS`.
        """
        side_bits = (int(side_count) + 1).bit_length()
        contact_bits = (int(contact_max) + 1).bit_length()
        if contact_bits + 2 * side_bits > UNIT_KEY_BITS:
            raise ValueError(
                f"scene_units sort keys need {contact_bits} + 2 x {side_bits} bits for {contact_max} contacts "
                f"and {side_count} articulations, more than {UNIT_KEY_BITS} bits"
            )
        return side_bits, contact_bits

    @staticmethod
    def limit_bearing_articulations(solver) -> np.ndarray:
        """Flag (``int32``, one per articulation) the articulations that can emit joint-limit rows.

        An articulation is flagged when one of its response DOFs has a limit coordinate in Steel's
        ``_joint_limit_q_index`` (PRISMATIC, REVOLUTE and D6 DOFs) and a finite lower or upper
        bound: the per-DOF test of Steel's joint-limit row builders, so free rigid bodies and
        articulations whose limits are all infinite are not flagged. The bounds are read once,
        at construction: this is the joint-limit bit of the solver's per-articulation flags,
        ``_scene_family_bearing`` (:func:`~newton._src.solvers.steel.scene_solve.scene_internal_bearing`).
        """
        return ((solver._scene_family_bearing >> FAMILY_LIMIT) & 1).astype(np.int32)

    @staticmethod
    def internal_bearing_articulations(solver) -> np.ndarray:
        """Flag (``int32``, one per articulation) the articulations that can have an internal unit.

        Those that can bear a row of some internal family, any internal family's bit of the solver's
        per-articulation flags ``_scene_family_bearing``
        (:func:`~newton._src.solvers.steel.scene_solve.scene_internal_bearing`; the velocity-limit bit is the late
        unit's, :meth:`late_bearing_articulations`): with ``enable_joint_limits``
        the limit-bearing ones (:meth:`limit_bearing_articulations`); with ``drive_mode="rows"`` every
        articulation with a response DOF of a PRISMATIC, REVOLUTE or D6 joint, whatever its gains; with
        ``enable_joint_friction`` every articulation with a response DOF whose joint type supports
        friction (Steel's ``_friction_dof_supported``: PRISMATIC, REVOLUTE, BALL and D6 joints outside
        loop closures), whatever its coefficient, since gains and coefficients are read on the device
        at every step; under the scene mode every articulation that owns a mimic row or a loop closure,
        whatever its enabled flags. Free rigid bodies (a FREE joint) get none but a loop closure's.
        """
        # The internal unit's families are those before the velocity limits, whose rows are late rows (Steel's slot
        # order).
        return ((solver._scene_family_bearing & ((1 << FAMILY_VELOCITY_LIMIT) - 1)) != 0).astype(np.int32)

    @staticmethod
    def late_bearing_articulations(solver) -> np.ndarray:
        """Flag (``int32``, one per articulation) the articulations that can have a late unit.

        The velocity-limit bit of the solver's per-articulation flags ``_scene_family_bearing``
        (:func:`~newton._src.solvers.steel.scene_solve.scene_internal_bearing`): with
        ``enable_joint_velocity_limits`` an articulation with a response DOF of a PRISMATIC, REVOLUTE or D6 joint,
        whatever its :attr:`~newton.Model.joint_velocity_limit` (Steel's per-DOF test of its velocity-limit rows
        reads the limit at every step); with both rigid-body velocity limits registered a free rigid body whose
        linear or angular limit is finite and positive, Steel's per-axis test of its free-body velocity-limit rows.
        """
        return ((solver._scene_family_bearing >> FAMILY_VELOCITY_LIMIT) & 1).astype(np.int32)

    def _allocate(self, unit_max: int) -> None:
        if unit_max < 1:
            raise ValueError(f"scene_unit_max must be at least 1, got {unit_max}")
        device = self.device
        self.unit_max = unit_max
        self.unit_contact = wp.zeros(unit_max, dtype=wp.int32, device=device)
        self.unit_side_a = wp.zeros(unit_max, dtype=wp.int32, device=device)
        self.unit_side_b = wp.zeros(unit_max, dtype=wp.int32, device=device)
        self.unit_rows = wp.zeros(unit_max, dtype=wp.int32, device=device)
        self.unit_key = wp.zeros(unit_max, dtype=wp.int64, device=device)
        self.unit_order = wp.zeros(unit_max, dtype=wp.int32, device=device)
        self.unit_color = wp.full(unit_max, -1, dtype=wp.int32, device=device)
        self.incidence_unit = wp.zeros(2 * unit_max, dtype=wp.int32, device=device)
        self.order_by_color = wp.zeros(unit_max, dtype=wp.int32, device=device)
        # Sorting scratch, per slot: the sorted position, the incoming bucket (side_b of a two-sided
        # unit, else -1); per bucket entry: the bucket fill and the incoming units by side_b.
        self._unit_position = wp.zeros(unit_max, dtype=wp.int32, device=device)
        self._incoming_bucket = wp.zeros(unit_max, dtype=wp.int32, device=device)
        self._bucket_items = wp.zeros(unit_max, dtype=wp.int32, device=device)
        self._incoming_slot = wp.zeros(unit_max, dtype=wp.int32, device=device)
        self._color_proposal = wp.full(unit_max, -1, dtype=wp.int32, device=device)
        self._color_blocks = (unit_max + COLOR_ORDER_BLOCK - 1) // COLOR_ORDER_BLOCK
        # Per (color slot, block), color-major: counts, then their exclusive scan used as cursors.
        self._color_block_count = wp.zeros(self.color_slots * self._color_blocks, dtype=wp.int32, device=device)
        self._color_block_cursor = wp.zeros(self.color_slots * self._color_blocks, dtype=wp.int32, device=device)
        # Tile totals and offsets of _scan_exclusive over those counts.
        scan_tiles_max = (self.color_slots * self._color_blocks + SCAN_TILE - 1) // SCAN_TILE
        self._scan_tile_totals = wp.zeros(scan_tiles_max, dtype=wp.int32, device=device)
        self._scan_tile_offsets = wp.zeros(scan_tiles_max, dtype=wp.int32, device=device)
        # Collection: a flag and its scanned slot per internal unit, per late unit and per contact of the capacity.
        entries = max(self.side_count + self._late_sides + self.contact_max, 1)
        self._unit_flag = wp.zeros(entries, dtype=wp.int32, device=device)
        self._unit_slot = wp.zeros(entries, dtype=wp.int32, device=device)

    def build(
        self,
        solver,
        contacts,
        stream=None,
        *,
        internal_rows: wp.array | None = None,
        limit_rows: wp.array | None = None,
        late_rows: wp.array | None = None,
    ) -> None:
        """Collect, sort and index this step's units from the solver's contact routing.

        Runs after the solver's contacts are routed (``contact_path``, ``contact_art_a/b``,
        ``contact_slots_needed``), on ``stream`` or else the current stream, with launch shapes
        fixed by the capacities. With ``internal_rows`` (``int32``, ``[side_count]``, read on the
        device), an internal unit exists for each articulation with ``internal_rows[art] > 0`` and
        carries that many rows; without it, :attr:`art_has_limits` decides and the rows are ``0``.
        ``limit_rows`` is its name from before joint friction, for an internal unit of joint limits
        alone; passing both raises :class:`ValueError`. With ``late_rows`` (``int32``, ``[side_count]``),
        a late unit exists for each articulation with ``late_rows[art] > 0`` and carries that many rows; it
        needs a :attr:`late_unit_count` above zero (:class:`ValueError` otherwise). Remembers ``contacts`` for
        the solver's ``time_scene_units``.
        """
        if internal_rows is not None and limit_rows is not None:
            raise ValueError("pass internal_rows or limit_rows, not both")
        name, rows = ("limit_rows", limit_rows) if limit_rows is not None else ("internal_rows", internal_rows)
        for label, array in ((name, rows), ("late_rows", late_rows)):
            if array is not None and array.shape != (self.side_count,):
                raise ValueError(f"{label} must have shape ({self.side_count},), got {array.shape}")
        if late_rows is not None and not self._late_sides:
            raise ValueError("late_rows needs an articulation that can bear late rows (late_unit_count > 0)")
        if stream is None:
            self._build(solver, contacts, rows, late_rows)
        else:
            with wp.ScopedStream(stream):
                self._build(solver, contacts, rows, late_rows)

    def _build(self, solver, contacts, internal_rows, late_rows=None) -> None:
        device = self.device
        has_contacts = contacts is not None and contacts.rigid_contact_max > 0
        if has_contacts and contacts.rigid_contact_max > self.contact_max:
            # The solver's step rejects such a buffer first; this guards direct calls.
            raise ValueError(
                f"scene_units sort keys were sized for {self.contact_max} contacts, got a buffer of "
                f"{contacts.rigid_contact_max}"
            )
        if self.unit_max is None:
            if device.is_capturing:
                raise RuntimeError(
                    "SolverSteel scene units are allocated by the first step; run one step before "
                    "capturing a graph, or pass scene_unit_max"
                )
            contact_capacity = contacts.rigid_contact_max if has_contacts else solver._max_contacts_alloc
            limits = self.side_count if internal_rows is not None else self.limit_unit_count
            self._allocate(contact_capacity + limits + self.late_unit_count)
        self._collect(solver, contacts if has_contacts else None, internal_rows, late_rows)
        self._sort()
        self.built = True
        self._last_contacts = contacts

    def _collect(self, solver, contacts, internal_rows, late_rows) -> None:
        """Flag, scan and compact the units into slots: internal and late units by articulation, then contacts."""
        device = self.device
        side_count = self.side_count
        if internal_rows is not None:
            limit_source, limit_sides, use_limit_rows = internal_rows, side_count, 1
        elif self.limit_unit_count:
            limit_source, limit_sides, use_limit_rows = self.art_has_limits, side_count, 0
        else:
            limit_source, limit_sides, use_limit_rows = None, 0, 0
        late_sides = side_count if late_rows is not None else 0
        # Contact c's flag entry is contact_entry + c: its unit, when it has one, is slot _unit_slot[contact_entry + c].
        self.contact_entry = side_count + late_sides
        if contacts is not None:
            contact_count, contact_capacity = contacts.rigid_contact_count, contacts.rigid_contact_max
            routing = [solver._contact_path, solver._contact_art_a, solver._contact_art_b]
            slots_needed = solver._contact_slots_needed
        else:
            contact_count, contact_capacity = self._no_contacts, 0
            routing = [None, None, None]
            slots_needed = None
        response = solver._articulation_response_dof_count
        entries = side_count + late_sides + contact_capacity
        if entries:
            wp.launch(
                flag_units,
                dim=entries,
                inputs=[
                    side_count,
                    limit_source,
                    limit_sides,
                    late_rows,
                    late_sides,
                    contact_count,
                    contact_capacity,
                    *routing,
                    response,
                ],
                outputs=[self._unit_flag],
                device=device,
            )
            # The build runs at the top level of a captured step, where array_scan's scratch comes from the mempool;
            # only the colors' scan runs inside a conditional graph node, which cannot allocate (_scan_exclusive).
            wp.utils.array_scan(self._unit_flag[:entries], self._unit_slot[:entries], inclusive=False)
        wp.launch(
            collect_units,
            dim=max(entries, 1),
            inputs=[
                side_count,
                limit_source,
                limit_sides,
                use_limit_rows,
                late_rows,
                late_sides,
                contact_count,
                contact_capacity,
                *routing,
                response,
                slots_needed,
                self._unit_flag,
                self._unit_slot,
                self.unit_max,
            ],
            outputs=[
                self.unit_count,
                self.unit_overflow,
                self.unit_contact,
                self.unit_side_a,
                self.unit_side_b,
                self.unit_rows,
            ],
            device=device,
        )

    def _sort(self) -> None:
        """Key, count and sort the collected units by side, then index them: CSR, one-sided prefix and incidence."""
        device = self.device
        unit_max = self.unit_max
        side_count = self.side_count
        self._side_counts.zero_()
        wp.launch(
            count_unit_sides,
            dim=unit_max,
            inputs=[
                self.unit_count,
                self.side_bits,
                self.contact_bits,
                self.unit_contact,
                self.unit_side_a,
                self.unit_side_b,
            ],
            outputs=[self.unit_key, self._side_counts, self._incoming_bucket],
            device=device,
        )
        # Outside any conditional graph node, as the flags' scan in _collect.
        wp.utils.array_scan(self._side_scan_input, self._side_scan, inclusive=False)
        wp.launch(
            publish_side_starts,
            dim=side_count + 1,
            inputs=[side_count, self._side_counts, self._side_scan],
            outputs=[self.incidence_start, self._incoming_start, self.side_coupled],
            device=device,
        )
        wp.launch(
            flag_two_sided,
            dim=1,
            inputs=[side_count, self.incidence_start],
            outputs=[self.has_two_sided],
            device=device,
        )
        # Counting sort by side_a: every slot into unit_order, in key order within its side.
        wp.launch(
            bucket_fill,
            dim=unit_max,
            inputs=[self.unit_count, self.unit_side_a, self.side_start, self._side_counts[_SIDE_CURSOR]],
            outputs=[self._bucket_items],
            device=device,
        )
        wp.launch(
            bucket_rank,
            dim=unit_max,
            inputs=[side_count, self.side_start, self.unit_side_a, self._bucket_items, self.unit_key, 1],
            outputs=[self.unit_order, self._unit_position],
            device=device,
        )
        # Counting sort by side_b: the two-sided slots into _incoming_slot, in key (= position) order.
        wp.launch(
            bucket_fill,
            dim=unit_max,
            inputs=[
                self.unit_count,
                self._incoming_bucket,
                self._incoming_start,
                self._side_counts[_SIDE_INCOMING_CURSOR],
            ],
            outputs=[self._bucket_items],
            device=device,
        )
        wp.launch(
            bucket_rank,
            dim=unit_max,
            inputs=[side_count, self._incoming_start, self._incoming_bucket, self._bucket_items, self.unit_key, 0],
            outputs=[self._incoming_slot, self._unit_position],
            device=device,
        )
        wp.launch(
            scatter_incidence,
            dim=unit_max,
            inputs=[
                side_count,
                self.unit_count,
                self.unit_order,
                self._unit_position,
                self.unit_side_a,
                self.unit_side_b,
                self._side_counts,
                self.side_start,
                self.incidence_start,
                self._incoming_start,
                self._incoming_slot,
            ],
            outputs=[self.incidence_unit],
            device=device,
        )

    def color(self, stream=None) -> None:
        """Color this step's two-sided units and list them by color.

        Clears ``unit_color`` (one-sided units stay ``-1``), ``side_mask``, the counters,
        ``color_count_per`` and ``color_offsets``; then, under ``wp.capture_if(has_two_sided)``,
        the pair rounds (``pair_rule``) until done or capped, the tail fallback and the order by
        color (``color_count_per``, ``color_offsets``, ``order_by_color``). Runs after
        :meth:`build`, on ``stream`` or else the current stream, with launch shapes fixed by the
        capacities. Under graph capture the branch and the round loop are conditional nodes and
        nothing is read back; run eagerly, ``wp.capture_if`` reads ``has_two_sided`` back once and
        ``wp.capture_while`` reads ``keep_going`` back once per round.
        """
        if self.unit_max is None:
            raise RuntimeError("SceneUnits.color() needs the units of a build() first")
        if stream is None:
            self._color()
        else:
            with wp.ScopedStream(stream):
                self._color()

    def _color(self) -> None:
        self.unit_color.fill_(-1)
        self.side_mask.zero_()
        self._color_counters.zero_()
        self._color_order_counts.zero_()
        wp.capture_if(self.has_two_sided, self._color_two_sided)

    def _color_two_sided(self) -> None:
        if self.pair_rule == "speculative":
            # One-sided positions never get a proposal; the rounds rewrite every two-sided one.
            self._color_proposal.fill_(-1)
        # The first count sets keep_going for the loop: some two-sided unit is uncolored.
        self._count_uncolored()
        wp.capture_while(self.keep_going, self._color_round)
        wp.launch(
            color_tail_fallback,
            dim=self.unit_max,
            inputs=[self.unit_count, self.unit_side_a, self.unit_side_b, self.color_cap],
            outputs=[self.unit_color],
            device=self.device,
        )
        self._order_by_color()

    def _order_by_color(self) -> None:
        device = self.device
        blocks = self._color_blocks
        self._color_block_count.zero_()
        wp.launch(
            count_colors,
            dim=self.unit_max,
            inputs=[self.unit_count, self.unit_order, self.unit_color, COLOR_ORDER_BLOCK, blocks],
            outputs=[self._color_block_count],
            device=device,
        )
        self._scan_exclusive(self._color_block_count, self._color_block_cursor)
        wp.launch(
            scatter_by_color,
            dim=blocks,
            inputs=[self.unit_count, self.unit_order, self.unit_color, COLOR_ORDER_BLOCK, blocks, self.color_slots],
            outputs=[self._color_block_cursor, self.color_count_per, self.color_offsets, self.order_by_color],
            device=device,
        )

    def _scan_exclusive(self, values: wp.array, scanned: wp.array) -> None:
        """Exclusive scan of ``values`` into ``scanned`` with no allocation: three launches of fixed shape.

        :func:`scan_tiles` scans each tile and records its total, :func:`scan_tile_totals` scans
        the totals in one block, and :func:`add_tile_offsets` adds them. ``wp.utils.array_scan``
        allocates CUB scratch, which a CUDA graph's conditional body cannot hold.
        """
        device = self.device
        count = values.shape[0]
        tiles = (count + SCAN_TILE - 1) // SCAN_TILE
        # A CPU tile launch runs one lane per block (Warp GH-1413): the tile operations take the whole tile there.
        block_dim = 1 if device.is_cpu else SCAN_BLOCK_DIM
        wp.launch_tiled(
            scan_tiles,
            dim=[tiles],
            inputs=[values],
            outputs=[scanned, self._scan_tile_totals],
            block_dim=block_dim,
            device=device,
        )
        wp.launch_tiled(
            scan_tile_totals,
            dim=[1],
            inputs=[self._scan_tile_totals, tiles],
            outputs=[self._scan_tile_offsets],
            block_dim=block_dim,
            device=device,
        )
        wp.launch(
            add_tile_offsets, dim=count, inputs=[count, self._scan_tile_offsets], outputs=[scanned], device=device
        )

    def statistics(self) -> dict:
        """Return the last :meth:`color`'s units and colors, read back to the host.

        Keys (all plain ``int``, ``per_color`` a ``list[int]``):

        - ``units``: units of the step (``unit_count``); ``overflow``: units past the capacity,
          counted but not stored.
        - ``two_sided``: units between two articulations; ``internal_units``: internal units (an
          articulation's own rows: drives, mimic, loop closures, joint limits and joint friction).
        - ``rounds``: pair rounds run, at most :data:`COLOR_ROUND_MAX`; ``0`` when no unit is
          two-sided.
        - ``colors``: the highest color below ``color_cap`` that a two-sided unit has, plus one
          (``0`` when none is colored below the cap).
        - ``per_color``: two-sided units of each color ``0..color_cap - 1`` (``color_cap`` entries).
        - ``tail``: two-sided units of the tail color ``color_cap``, whether mask capped or left by
          the round cap, so ``sum(per_color) + tail == two_sided``.
        - ``tail_fallback``: of those, the units the round cap left for the tail fallback
          (``uncolored``).

        Only two-sided units are colored, so the color keys describe them alone; one-sided units
        count only in ``units`` and ``internal_units`` (a late unit only in ``units``). Reads the device, so call it
        outside graph capture.
        """
        if not self.built:
            raise RuntimeError("SceneUnits.statistics() needs the units of a build() and color() first")
        cap = self.color_cap
        units = int(self.unit_count.numpy()[0])
        counts = self.color_count_per.numpy()
        used = np.flatnonzero(counts[:cap])
        internal = int(np.count_nonzero(self.unit_contact[:units].numpy() == UNIT_CONTACT_INTERNAL)) if units else 0
        return {
            "units": units,
            "overflow": int(self.unit_overflow.numpy()[0]),
            "two_sided": int(self.incidence_start.numpy()[self.side_count]) // 2,
            "internal_units": internal,
            "rounds": int(self.rounds.numpy()[0]),
            "colors": int(used[-1]) + 1 if used.size else 0,
            "per_color": [int(n) for n in counts[:cap]],
            "tail": int(counts[cap]),
            "tail_fallback": int(self.uncolored.numpy()[0]),
        }

    def _color_round(self) -> None:
        if self.pair_rule == "speculative":
            self._speculative_round()
        else:
            self._local_min_round()
        self._count_uncolored()

    def _speculative_round(self) -> None:
        wp.launch(
            color_propose,
            dim=self.side_count,
            inputs=[
                self.side_start,
                self.one_sided_count,
                self.unit_order,
                self.unit_side_b,
                self.unit_color,
                self.side_mask,
                self.color_cap,
            ],
            outputs=[self._color_proposal, self.rounds, self.uncolored, self.keep_going],
            device=self.device,
        )
        wp.launch(
            color_resolve,
            dim=self.unit_max,
            inputs=[
                self.unit_count,
                self.unit_order,
                self.unit_side_a,
                self.unit_side_b,
                self.incidence_start,
                self.incidence_unit,
                self._color_proposal,
                self.color_cap,
            ],
            outputs=[self.unit_color, self.side_mask],
            device=self.device,
        )

    def _local_min_round(self) -> None:
        wp.launch(
            color_pair_round,
            dim=self.unit_max,
            inputs=[
                self.unit_count,
                self.unit_order,
                self.unit_side_a,
                self.unit_side_b,
                self.incidence_start,
                self.incidence_unit,
                self.color_cap,
            ],
            outputs=[self.unit_color, self.side_mask, self.rounds, self.uncolored, self.keep_going],
            device=self.device,
        )

    def _count_uncolored(self) -> None:
        wp.launch(
            count_uncolored,
            dim=self.unit_max,
            inputs=[self.unit_count, self.unit_side_a, self.unit_side_b, self.rounds, COLOR_ROUND_MAX],
            outputs=[self.unit_color, self.uncolored, self.keep_going],
            device=self.device,
        )
