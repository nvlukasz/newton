# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Contact torsion of the scene solve (``solve_mode="scene"``): Steel's torsion groups on the folded scene and the
spin rows.

Steel prepares its torsion rows per solve world on the device (``contact_torsion_device``): ``_gather`` admits the
witnesses and ``_group_world`` clusters one world per thread. The scene folds every world into one, so
:func:`group_scene_torsion` sorts the contacts by cluster key instead and runs ``_group_world``'s clustering and
row selection with one thread per key segment; :func:`scene_torsion_eligibility` computes ``_gather``'s eligibility,
with the shapes' torsion support marked by Steel's own rule (:class:`SceneTorsionShapes`). Steel's row emission is
replaced by the scene's spin rows, late rows after every unit's rows (:func:`scene_torsion_routing` to
:func:`scene_torsion_finish`, :func:`get_scene_torsion_rows_kernel`, :func:`prepare_scene_velocity_torsion`): the
spin rows that fit are kept, a prefix in row order, and a dropped one flags its worlds in ``constraint_overflow``, as
Steel keeps the torsion rows of a world that fit.

The grouping shares three expressions with Steel's ``_gather`` and ``_group_world``: the contact frame, the cluster
compatibility walk and the member budget. They are copies here (:func:`scene_contact_frame`,
:func:`scene_cluster_compatible`, :func:`scene_member_budget`), compiled with Steel's rounding (``fuse_fp: False``)
so that they compute Steel's values bit for bit; :func:`_mirrored_lines` extracts each copy's lines and the Steel
lines it mirrors, and a test keeps them equal.
"""

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from functools import cache

import warp as wp

from ...geometry.utils import _scan_scratch_size, scan_with_total
from ...sim import Contacts
from . import contact_torsion_device
from .contact_torsion import _NORMAL_COSINE, _SUPPORTED_TYPES, _TORSION_SPIN_RETIRED, _TOUCH_TOLERANCE
from .contact_torsion_device import _host_dot3, _host_transform_point, _lower_bound, _mark_supported_types
from .kernels import _CONTACT_END_GAP_SLOP, PGS_CONSTRAINT_TYPE_TORSION, WARP_DETERMINISM_OFF, contact_base_rows
from .scene_rows import _finish_scene_row, _scene_row_dot, scene_contact_response_snippet
from .scene_units import _contact_unit_sides, _latch_side_world, _scene_contact_on_free_path, _unit_model_world

# The rounding of Steel's device torsion preparation: its frames and thresholds, bit for bit.
wp.set_module_options({"fuse_fp": False, **WARP_DETERMINISM_OFF})

_ROW_KERNEL_OPTIONS = {"enable_backward": False, "module": "unique", "module_options": WARP_DETERMINISM_OFF}
"""The spin rows' kernels keep the rounding of the scene's row kernels (fused multiply-adds allowed), not this
module's: the velocity-pass admission forms a member's end gap as the scene's velocity rows form it."""


# --- Copies of Steel's torsion preparation expressions, kept equal to Steel's text by _mirrored_lines -----------------


@wp.func
def scene_contact_frame(
    c: int,
    a: int,
    b: int,
    shape_body: wp.array[int],
    normals: wp.array[wp.vec3],
    point_a: wp.array[wp.vec3],
    point_b: wp.array[wp.vec3],
    margin_a: wp.array[float],
    margin_b: wp.array[float],
    poses: wp.array[wp.transform],
):
    """Contact ``c``'s row normal (B to A), midpoint and gap [m] between shapes ``a`` and ``b``.

    ``_gather``'s frame: the negated contact normal, each point in world space through its body's pose, the margins
    along the normal, the gap along the normal, and the midpoint, under this module's ``fuse_fp: False``.
    """
    normal = -normals[c]
    ba = shape_body[a]
    bb = shape_body[b]
    pa = point_a[c]
    pb = point_b[c]
    if ba >= 0:
        pa = _host_transform_point(poses[ba], pa)
    if bb >= 0:
        pb = _host_transform_point(poses[bb], pb)
    pa -= margin_a[c] * normal
    pb += margin_b[c] * normal
    gap = _host_dot3(normal, pa - pb)
    point = 0.5 * (pa + pb)
    return normal, point, gap


@wp.func
def scene_cluster_compatible(
    normal: wp.array[wp.vec3], point: wp.array[wp.vec3], next_member: wp.array[int], c: int, rep: int
) -> bool:
    """Whether contact ``c`` may join representative ``rep``'s cluster: compatible with every member.

    ``_group_world``'s clustering rule: along the member links from ``rep``, each member's normal within
    ``_NORMAL_COSINE`` of ``c``'s and ``c``'s contact plane within ``_TOUCH_TOLERANCE`` of the member's point. Every
    member is visited.
    """
    compatible = bool(True)
    member = rep
    while member >= 0:
        if _host_dot3(normal[c], normal[member]) < _NORMAL_COSINE:
            compatible = False
        if wp.abs(_host_dot3(normal[c], point[c] - point[member])) > _TOUCH_TOLERANCE:
            compatible = False
        member = next_member[member]
    return compatible


@wp.func
def scene_member_budget(
    c: int,
    radius_a: float,
    radius_b: float,
    anchor: int,
    gap: float,
    radius: float,
    anchor_count: int,
    first_anchor: int,
    touching: int,
):
    """Fold member ``c`` into its cluster's spin budget, as ``_group_world``'s row selection does.

    Returns ``radius`` raised to the larger of the member's shape radii [m], ``anchor_count`` and ``first_anchor``
    with an anchor (a member with its tangent pair) counted, and ``touching``, the first member whose ``gap`` is at
    most ``_TOUCH_TOLERANCE``, in the order the members are folded.
    """
    radius = wp.max(radius, wp.max(radius_a, radius_b))
    if anchor != 0:
        anchor_count += 1
        if first_anchor < 0:
            first_anchor = c
    if touching < 0 and gap <= _TOUCH_TOLERANCE:
        touching = c
    return radius, anchor_count, first_anchor, touching


SCENE_MIRRORED_HELPERS = ("scene_contact_frame", "scene_cluster_compatible", "scene_member_budget")
"""The copies of Steel's expressions that :func:`_mirrored_lines` compares with Steel's text."""

_MIRRORS = {
    "scene_contact_frame": (
        "_gather",
        (
            ("normal = -normals[c]", "gap = _host_dot3(normal, pa - pb)"),
            ("work.point[c] = 0.5 * (pa + pb)", "work.point[c] = 0.5 * (pa + pb)"),
        ),
        (("work.point[c] = ", "point = "),),
    ),
    "scene_cluster_compatible": (
        "_group_world",
        (
            ("compatible = bool(True)", "compatible = bool(True)"),
            ("member = other", "member = work.next_member[member]"),
        ),
        (("work.", ""), ("member = other", "member = rep")),
    ),
    "scene_member_budget": (
        "_group_world",
        (("radius = wp.max(radius, ", "touching = c"),),
        (
            ("shape_radius[work.shape_a[c]]", "radius_a"),
            ("shape_radius[work.shape_b[c]]", "radius_b"),
            ("work.anchor[c]", "anchor"),
            ("work.gap[c]", "gap"),
        ),
    ),
}
"""Per copy: Steel's function in ``contact_torsion_device``, the ranges of its lines the copy mirrors (first and last
line, stripped, matched as prefixes in order), and the substitutions from Steel's names to the copy's."""


def _mirrored_lines(name: str) -> tuple[list[str], list[str]]:
    """The lines of the copy ``name`` (one of :data:`SCENE_MIRRORED_HELPERS`) and the Steel lines it mirrors.

    The copy's lines are its body after the docstring, without the final ``return``; Steel's are the ranges of
    :data:`_MIRRORS` in Steel's ``_gather`` or ``_group_world`` source, with the workspace's names substituted by
    the copy's. Each range is dedented on its own. Equal lists mean the copy computes Steel's expressions; a test
    asserts it, so a change of Steel's text fails there. Raises ``LookupError`` when a range is not in Steel's text.
    """
    steel_name, ranges, substitutions = _MIRRORS[name]
    copy = textwrap.dedent(inspect.getsource(globals()[name].func))
    body = ast.parse(copy).body[0].body
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    copy_lines = copy.splitlines()[body[0].lineno - 1 : body[-1].lineno - 1]
    scene = textwrap.dedent("\n".join(copy_lines)).splitlines()
    source = inspect.getsource(getattr(contact_torsion_device, steel_name).func).splitlines()
    steel = []
    position = 0
    for first, last in ranges:
        start = next((i for i in range(position, len(source)) if source[i].strip().startswith(first)), None)
        if start is None:
            raise LookupError(f"{steel_name} has no line {first!r} for {name}")
        end = next((i for i in range(start, len(source)) if source[i].strip().startswith(last)), None)
        if end is None:
            raise LookupError(f"{steel_name} has no line {last!r} after {first!r} for {name}")
        block = textwrap.dedent("\n".join(source[start : end + 1])).splitlines()
        for old, new in substitutions:
            block = [line.replace(old, new) for line in block]
        steel.extend(block)
        position = end + 1
    return scene, steel


# --- Steel's torsion grouping on the folded scene ------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _scene_torsion_eligibility(
    count: wp.array[int],
    input_capacity: int,
    paths: wp.array[int],
    slots: wp.array[int],
    slots_needed: wp.array[int],
    shape_a: wp.array[int],
    shape_b: wp.array[int],
    normals: wp.array[wp.vec3],
    point_a: wp.array[wp.vec3],
    point_b: wp.array[wp.vec3],
    margin_a: wp.array[float],
    margin_b: wp.array[float],
    shape_body: wp.array[int],
    shape_supported: wp.array[int],
    shape_radius: wp.array[float],
    poses: wp.array[wp.transform],
    patch_mode: int,
    # outputs
    eligible: wp.array[int],
):
    """``_gather``'s eligibility of contact ``c`` on Steel's dense rows (grid: ``eligible``'s length).

    The gates in ``_gather``'s order: past the contact count, off Steel's articulated rows (``paths[c] != 0``) or
    without rows (``slots[c] < 0``), 0; from there -1 (a rejected witness): a shape index out of range, both radii
    zero, a shape whose type does not support torsion (``shape_supported``, :class:`SceneTorsionShapes`), and without
    patches a contact without its friction pair (base rows ``contact_base_rows(slots_needed[c]) != 3``); then without
    patches a contact more than ``_TOUCH_TOLERANCE`` apart is 0; every other one is 1. ``_gather``'s checks of the
    solve world and of the dense rows at ``slots[c]`` validate Steel's per-world allocation and are not repeated: a
    contact the allocation routed and kept has its rows by construction. Its propagation family (contacts on the
    propagation response) has no counterpart: the scene solves every responding contact on its own rows.
    """
    c = wp.tid()
    eligible[c] = 0
    if c >= wp.min(count[0], input_capacity):
        return
    if paths[c] != 0 or slots[c] < 0:
        return
    eligible[c] = -1
    a = shape_a[c]
    b = shape_b[c]
    if a < 0 or b < 0 or a >= shape_body.shape[0] or b >= shape_body.shape[0]:
        return
    if wp.max(shape_radius[a], shape_radius[b]) <= 0.0:
        return
    if shape_supported[a] == 0 or shape_supported[b] == 0:
        return
    if contact_base_rows(slots_needed[c]) != 3 and patch_mode == 0:
        return
    _normal, _point, gap = scene_contact_frame(
        c, a, b, shape_body, normals, point_a, point_b, margin_a, margin_b, poses
    )
    if gap > _TOUCH_TOLERANCE and patch_mode == 0:
        eligible[c] = 0
        return
    eligible[c] = 1


@wp.struct
class _SceneTorsionWork:
    """Scratch of :func:`group_scene_torsion`, one entry per contact unless noted."""

    keys: wp.array[wp.int64]
    """``[2 * capacity]``: the cluster key, sorted with :attr:`indices` (the radix sort's scratch is the back half)."""
    indices: wp.array[int]
    """``[2 * capacity]``: the contacts in key order, contact order within a key."""
    eligible: wp.array[int]
    """The contact's eligibility as the clustering reads it: 0 past the count or for an unkeyed contact."""
    normal: wp.array[wp.vec3]
    point: wp.array[wp.vec3]
    gap: wp.array[float]
    group: wp.array[int]
    """The contact's cluster representative, -1 when not eligible (Steel's ``work.group``)."""
    next_member: wp.array[int]
    tail: wp.array[int]
    size: wp.array[int]
    """A representative's member count when its cluster gets a row, else 0."""
    row: wp.array[int]
    """1 for a representative whose cluster gets a row, else 0."""
    member_end: wp.array[int]
    """The inclusive scan of :attr:`size`: one past a cluster's last member in ``members``."""
    rank_end: wp.array[int]
    """The inclusive scan of :attr:`row`: one past a cluster's place in ``member_start``."""


@wp.kernel(enable_backward=False)
def _scene_torsion_keys(
    count: wp.array[int],
    input_capacity: int,
    eligible: wp.array[int],
    shape_a: wp.array[int],
    shape_b: wp.array[int],
    normals: wp.array[wp.vec3],
    point_a: wp.array[wp.vec3],
    point_b: wp.array[wp.vec3],
    margin_a: wp.array[float],
    margin_b: wp.array[float],
    shape_body: wp.array[int],
    poses: wp.array[wp.transform],
    patch_mode: int,
    owner: wp.array[int],
    world: wp.array[int],
    has_world: int,
    work: _SceneTorsionWork,
    # outputs
    contact_group: wp.array[int],
    members: wp.array[int],
    lead: wp.array[int],
    coefficient: wp.array[float],
    radius: wp.array[float],
):
    """Clear contact ``c``'s outputs and scratch, and key it by its cluster (grid: the capacity).

    A keyed contact is an eligible one (1) and, under patches, a rejected witness (-1). Its key is, under patches, its
    region's: its solve world ``world[c]`` (0 without ``has_world``) and its patch owner shifted to be non-negative,
    ``(world << 32) | (owner + 2^31)``, as Steel's regions are per solve world (``_patch_keys``), else
    the stored shape pair ``(shape_a, shape_b)``, not canonicalised: ``_group_world``'s pair test. Every
    other contact keeps the largest key and sorts last. An eligible contact also records its frame
    (:func:`scene_contact_frame`); one whose shape index is out of range is not keyed.
    """
    c = wp.tid()
    work.keys[c] = wp.int64(9223372036854775807)
    work.indices[c] = c
    work.eligible[c] = 0
    work.group[c] = -1
    work.next_member[c] = -1
    work.tail[c] = c
    work.size[c] = 0
    work.row[c] = 0
    contact_group[c] = -1
    members[c] = -1
    lead[c] = -1
    coefficient[c] = 0.0
    radius[c] = 0.0
    if c >= wp.min(count[0], input_capacity):
        return
    state = eligible[c]
    if state != 1 and (state != -1 or patch_mode == 0):
        return
    a = shape_a[c]
    b = shape_b[c]
    if state == 1:
        if a < 0 or b < 0 or a >= shape_body.shape[0] or b >= shape_body.shape[0]:
            return
        normal, point, gap = scene_contact_frame(
            c, a, b, shape_body, normals, point_a, point_b, margin_a, margin_b, poses
        )
        work.normal[c] = normal
        work.point[c] = point
        work.gap[c] = gap
    work.eligible[c] = state
    if patch_mode != 0:
        region_world = int(0)
        if has_world != 0:
            region_world = world[c]
        work.keys[c] = (wp.int64(region_world) << wp.int64(32)) | (wp.int64(owner[c]) + wp.int64(2147483648))
    else:
        work.keys[c] = (wp.int64(a) << wp.int64(32)) | wp.int64(b)


@wp.kernel(enable_backward=False)
def _scene_torsion_segments(
    capacity: int,
    patch_mode: int,
    shape_a: wp.array[int],
    shape_b: wp.array[int],
    slots_needed: wp.array[int],
    friction: wp.array[float],
    shape_radius: wp.array[float],
    work: _SceneTorsionWork,
    # outputs
    lead: wp.array[int],
    coefficient: wp.array[float],
    radius: wp.array[float],
):
    """Run ``_group_world``'s passes 1 and 2 on one key segment, from its first sorted position (grid: the capacity).

    Pass 1 is Steel's ordered greedy clustering in contact order: an eligible contact joins the first earlier
    representative of its segment whose every member it is compatible with (without patches
    :func:`scene_cluster_compatible`: normal cosine at least ``_NORMAL_COSINE`` and contact-plane offset at most
    ``_TOUCH_TOLERANCE``; under patches every member), else it seeds its own cluster. The segment is Steel's pair (or
    owner) test; ``first_pair`` orders only Steel's row emission and is dropped, and with it Steel's scan past the
    chosen representative, which writes nothing else. Pass 2 per representative: under patches a rejected witness
    (-1) in the segment, which shares the representative's owner, rejects the region; radius, anchors and the first
    touching member over the members in contact order (:func:`scene_member_budget`); no row without an anchor or a
    touching member, and none for a coefficient ``friction[first anchor]`` (times the anchors under patches) that is
    not positive. The load-ring check of Steel's patch rows validates Steel's row parents, not the partition, and is
    not run. A representative whose cluster gets a row records ``lead`` (the first touching member), ``coefficient``,
    ``radius``, its member count and its row flag.
    """
    i = wp.tid()
    key = work.keys[i]
    if key == wp.int64(9223372036854775807):
        return
    if i > 0 and work.keys[i - 1] == key:
        return
    start = i
    end = i + 1
    while end < capacity and work.keys[end] == key:
        end += 1
    # Pass 1: the ordered greedy clustering.
    for k in range(start, end):
        c = work.indices[k]
        if work.eligible[c] != 1:
            continue
        chosen = int(-1)
        for j in range(start, k):
            other = work.indices[j]
            if work.eligible[other] != 1:
                continue
            if work.group[other] != other:
                continue
            compatible = bool(True)
            if patch_mode == 0:
                compatible = scene_cluster_compatible(work.normal, work.point, work.next_member, c, other)
            if compatible:
                chosen = other
                break
        if chosen < 0:
            chosen = c
        else:
            work.next_member[work.tail[chosen]] = c
            work.tail[chosen] = c
        work.group[c] = chosen
    # Pass 2: the clusters that get a row.
    for k in range(start, end):
        rep = work.indices[k]
        if work.group[rep] != rep:
            continue
        rejected = bool(False)
        anchor_count = int(0)
        first_anchor = int(-1)
        touching = int(-1)
        member_count = int(0)
        spin_radius = float(0.0)
        for j in range(start, end):
            c = work.indices[j]
            if patch_mode != 0 and work.eligible[c] == -1:
                rejected = True
            if work.group[c] != rep:
                continue
            member_count += 1
            spin_radius, anchor_count, first_anchor, touching = scene_member_budget(
                c,
                shape_radius[shape_a[c]],
                shape_radius[shape_b[c]],
                int(contact_base_rows(slots_needed[c]) == 3),
                work.gap[c],
                spin_radius,
                anchor_count,
                first_anchor,
                touching,
            )
        if rejected:
            continue
        if first_anchor < 0 or touching < 0:
            continue
        value = friction[first_anchor]
        if patch_mode != 0:
            value *= float(anchor_count)
        if value <= 0.0:
            continue
        coefficient[rep] = value
        radius[rep] = spin_radius
        lead[rep] = touching
        work.size[rep] = member_count
        work.row[rep] = 1


@wp.kernel(enable_backward=False)
def _scene_torsion_emit(
    capacity: int,
    work: _SceneTorsionWork,
    # outputs
    contact_group: wp.array[int],
    group_count: wp.array[int],
    member_start: wp.array[int],
    members: wp.array[int],
):
    """Compact the clusters that get a row in representative order (grid: the capacity + 1).

    Thread ``t`` writes ``member_start[t]`` past the last cluster, and for a representative ``t`` its cluster's start
    and members, in contact order along the member links, and their ``contact_group``.
    """
    t = wp.tid()
    groups = work.rank_end[capacity - 1]
    total = work.member_end[capacity - 1]
    if t == 0:
        group_count[0] = groups
    if t >= groups:
        member_start[t] = total
    if t >= capacity:
        return
    if work.row[t] == 0:
        return
    k = work.member_end[t] - work.size[t]
    member_start[work.rank_end[t] - 1] = k
    member = t
    while member >= 0:
        members[k] = member
        contact_group[member] = t
        k += 1
        member = work.next_member[member]


@wp.kernel(enable_backward=False)
def _add_counts(counts: wp.array[int], prefix_sums: wp.array[int]):
    """Turn an exclusive scan into the inclusive one: add each entry's own count (grid: entries)."""
    i = wp.tid()
    prefix_sums[i] = prefix_sums[i] + counts[i]


@dataclass(eq=False)
class SceneTorsionGroups:
    """The contact torsion groups of a folded scene (:func:`group_scene_torsion`), ``int32`` arrays unless noted.

    A group is a cluster of Steel's torsion clustering that would get a spin row. Allocate one with :meth:`allocate`
    and pass it back as ``out`` to group every step into the same arrays (graph capture).
    """

    contact_group: wp.array
    """``[capacity]``: the representative (lowest) contact of the contact's group, -1 when its cluster gets no row or
    it is in none: the partition of Steel's ``_contact_torsion_group``."""
    group_count: wp.array
    """``[1]``: the number of groups."""
    member_start: wp.array
    """``[capacity + 1]``: group ``g``'s members are ``members[member_start[g]:member_start[g + 1]]``; every entry from
    ``group_count`` on is the member total. Groups are in representative order."""
    members: wp.array
    """``[capacity]``: the groups' contacts, each group's in contact order (its representative first), -1 past the
    member total."""
    lead: wp.array
    """``[capacity]``: at a representative, its group's first touching member (Steel's ``lead``, whose normal and
    rows a spin row takes), -1 elsewhere."""
    coefficient: wp.array
    """``float32`` ``[capacity]``: at a representative, its group's friction coefficient (Steel's ``coefficient``),
    0 elsewhere."""
    radius: wp.array
    """``float32`` ``[capacity]``: at a representative, its group's torsion radius [m] (Steel's ``radius``), 0
    elsewhere."""
    _work: _SceneTorsionWork = field(repr=False)
    _no_owner: wp.array = field(repr=False)
    _scan_length: wp.array = field(repr=False)
    """``[1]``: the capacity, the entries every scan of the grouping covers."""
    _scan_total: wp.array = field(repr=False)
    _scan_scratch: wp.array = field(repr=False)
    """``int64``: the scratch of the grouping's scans (``scan_with_total``), which they share one at a time."""

    @property
    def capacity(self) -> int:
        """The contacts the arrays hold."""
        return self.members.shape[0]

    def _inclusive_scan(self, counts: wp.array, prefix_sums: wp.array) -> None:
        """Write the inclusive scan of ``counts`` (``[capacity]``) into ``prefix_sums``: one launch of the exclusive
        :func:`~newton._src.geometry.utils.scan_with_total` on the preallocated scratch, then one adding each count.

        No allocation, so the grouping can be captured; the scans run one at a time and leave the scratch zeroed. Exact
        on integers, so it equals ``wp.utils.array_scan(counts, prefix_sums, inclusive=True)``.
        """
        scan_with_total(counts, prefix_sums, self._scan_length, self._scan_total, self._scan_scratch)
        wp.launch(_add_counts, dim=counts.shape[0], inputs=[counts], outputs=[prefix_sums], device=counts.device)

    @classmethod
    def allocate(cls, capacity: int, device=None) -> "SceneTorsionGroups":
        """Allocate the groups and the scratch of :func:`group_scene_torsion` for ``capacity`` contacts."""
        work = _SceneTorsionWork()
        work.keys = wp.empty(2 * capacity, dtype=wp.int64, device=device)
        work.indices = wp.empty(2 * capacity, dtype=int, device=device)
        for name in ("eligible", "group", "next_member", "tail", "size", "row", "member_end", "rank_end"):
            setattr(work, name, wp.empty(capacity, dtype=int, device=device))
        work.normal = wp.empty(capacity, dtype=wp.vec3, device=device)
        work.point = wp.empty(capacity, dtype=wp.vec3, device=device)
        work.gap = wp.empty(capacity, dtype=float, device=device)
        return cls(
            contact_group=wp.full(capacity, -1, dtype=int, device=device),
            group_count=wp.zeros(1, dtype=int, device=device),
            member_start=wp.zeros(capacity + 1, dtype=int, device=device),
            members=wp.full(capacity, -1, dtype=int, device=device),
            lead=wp.full(capacity, -1, dtype=int, device=device),
            coefficient=wp.zeros(capacity, dtype=float, device=device),
            radius=wp.zeros(capacity, dtype=float, device=device),
            _work=work,
            _no_owner=wp.full(1, -1, dtype=int, device=device),
            _scan_length=wp.full(1, capacity, dtype=int, device=device),
            _scan_total=wp.zeros(1, dtype=int, device=device),
            _scan_scratch=wp.zeros(_scan_scratch_size(capacity, device), dtype=wp.int64, device=device),
        )


@dataclass(eq=False)
class SceneTorsionShapes:
    """Which shapes' geometry types support contact torsion, marked on the device by Steel's rule.

    Steel's device preparation marks each shape whose current :attr:`~newton.Model.shape_type` is one of
    ``_SUPPORTED_TYPES`` (``DeviceTorsionPreparation.update_shape_types``) at construction and on a
    ``SHAPE_TYPE_AND_SOURCE`` notification, as made by :meth:`~newton.Model.select_body_variants`, and its gather reads
    the marks. :meth:`update` launches the same kernel (``_mark_supported_types``) into :attr:`supported`, so the
    scene's eligibility reads the types Steel's reads: call it where Steel calls ``update_shape_types``. No host read
    and no allocation: capturable.
    """

    supported: wp.array
    """``int32`` ``[shape_count]``: 1 for a shape whose type supports torsion as last marked, else 0."""
    _supported_types: wp.array = field(repr=False)

    @classmethod
    def allocate(cls, model) -> "SceneTorsionShapes":
        """Allocate the marks for ``model``'s shapes and mark them (:meth:`update`)."""
        device = model.device
        shapes = cls(
            supported=wp.zeros(model.shape_count, dtype=int, device=device),
            _supported_types=wp.array([int(t) for t in _SUPPORTED_TYPES], dtype=int, device=device),
        )
        shapes.update(model)
        return shapes

    def update(self, model) -> None:
        """Mark the shapes whose current type supports torsion (Steel's ``update_shape_types``)."""
        if model.shape_count == 0:
            return
        wp.launch(
            _mark_supported_types,
            dim=model.shape_count,
            inputs=[model.shape_type, self._supported_types],
            outputs=[self.supported],
            device=model.device,
        )


def scene_torsion_eligibility(
    contacts: Contacts,
    shape_body: wp.array,
    *,
    body_q: wp.array,
    shape_supported: wp.array,
    shape_radius: wp.array,
    contact_path: wp.array,
    contact_slot: wp.array,
    contact_slots_needed: wp.array,
    patches: bool = False,
    out: wp.array | None = None,
    device=None,
) -> wp.array:
    """Each contact's torsion eligibility, as Steel's ``_gather`` computes it (``int32`` ``[rigid_contact_max]``).

    1 for an eligible contact, -1 for a rejected witness (which rejects its region under patches), 0 otherwise; see
    :func:`_scene_torsion_eligibility` for the gates. The inputs are the model's and the contact routing's, no
    solver's: ``shape_body`` the model's, ``shape_supported`` the shapes' torsion support as Steel marks it
    (:attr:`SceneTorsionShapes.supported`), ``shape_radius`` [m] the model's ``steel:min_torsional_patch_radius``,
    ``body_q`` the step's input body poses, ``contact_path``, ``contact_slot`` and ``contact_slots_needed`` the contact
    routing's (Steel's ``_contact_path``, ``_contact_slot`` and ``_contact_slots_needed``: path 0 for an articulated
    contact, its normal row, -1 when it has none, and 3 base rows with its friction pair), ``patches`` whether
    persistent friction patches are on. One launch over the contact capacity, no host read: capturable. ``out`` takes
    the result when given (at least ``rigid_contact_max`` long).

    Preconditions: the routing must be Steel's. ``contact_path`` is 0 only for a contact Steel solves on its
    articulated (dense) rows: a contact Steel's ``is_mf`` rule sends to its free-body path (every side a free body or
    without response DOFs) must not be 0 here. The scene solve's own ``_contact_path`` routes those to 0 too, so its
    caller (``SceneSolve.prepare_torsion``) excludes them first. ``patches`` must agree with the ``patch_owner`` that
    :func:`group_scene_torsion` takes: derive both from one flag. Steel's latched preparation errors (hydroelastic
    contacts, an overflowing contact input, dropped dense rows) are not detected here; the scene's preparation
    latches them (``SceneSolve.prepare_torsion``).
    """
    device = wp.get_device(device if device is not None else shape_body.device)
    capacity = contacts.rigid_contact_max
    if out is None:
        out = wp.empty(capacity, dtype=int, device=device)
    elif out.shape[0] < capacity:
        raise ValueError(f"out holds {out.shape[0]} contacts, fewer than the contacts' {capacity}")
    if out.shape[0] == 0:
        return out
    wp.launch(
        _scene_torsion_eligibility,
        dim=out.shape[0],
        inputs=[
            contacts.rigid_contact_count,
            capacity,
            contact_path,
            contact_slot,
            contact_slots_needed,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_normal,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            contacts.rigid_contact_margin0,
            contacts.rigid_contact_margin1,
            shape_body,
            shape_supported,
            shape_radius,
            body_q,
            int(patches),
        ],
        outputs=[out],
        device=device,
    )
    return out


def group_scene_torsion(
    contacts: Contacts,
    shape_body: wp.array,
    eligible: wp.array,
    *,
    body_q: wp.array,
    shape_radius: wp.array,
    contact_slots_needed: wp.array,
    friction: wp.array,
    patch_owner: wp.array | None = None,
    contact_world: wp.array | None = None,
    out: SceneTorsionGroups | None = None,
    device=None,
) -> SceneTorsionGroups:
    """Group the scene's contacts into torsion groups: Steel's partition, one thread per cluster key.

    Steel's ``_group_world`` clusters one solve world per thread; after the fold one world holds every
    contact of the scene. Here the contacts are sorted by cluster key, the patch owner when ``patch_owner`` is given
    (persistent friction patches) and the stored shape pair ``(shape_a, shape_b)`` otherwise, with the contact index
    as the tie-break (a stable radix sort of the contacts in index order). One thread per key segment then runs
    ``_group_world``'s passes 1 and 2 on the segment in contact order (:func:`_scene_torsion_segments`): the same
    clusters, and the same clusters with a row, as Steel's, which are listed in :class:`SceneTorsionGroups`. Pass 3,
    Steel's row emission, is not reproduced: the scene places its spin rows itself.

    Under patches the key is Steel's ``(world, owner)`` (``_patch_keys``), the world from
    ``contact_world``: the witnesses without an owner (a negative ``patch_owner``) pool into one region per world,
    as in Steel, whose regions are per solve world. An owner is a region's seed contact, so the world changes nothing
    else.

    Preconditions: ``eligible`` is computed from Steel's routing (:func:`scene_torsion_eligibility`'s): the caller
    excludes the contacts Steel solves on its free-body path. ``patch_owner`` must be given exactly when
    ``eligible`` was computed with ``patches=True``: derive both from one flag. Nothing Steel's preparation latches
    is validated here (an overflowing input, dropped dense rows, the spin row capacity, the patch load ring): the
    scene's preparation latches those (``SceneSolve.prepare_torsion``).

    Work: pass 1 visits, for each eligible contact, the earlier contacts of its segment and walks the cluster of each
    candidate representative, and pass 2 scans the segment once per representative, so a segment of ``s`` contacts
    costs ``O(s^2)`` on its thread (Steel's world costs ``O(n_world^2)``) and the call ``O(s_max n)`` for ``n``
    contacts: linear in the scene's contacts while a shape pair's manifold or a patch region stays bounded.
    Deterministic (a key sort with the index tie-break and a sequential scan per segment, no atomics) and capturable:
    fixed launch grids over the capacity, no host read and, with ``out``, no allocation.

    Args:
        contacts: The step's contacts; ``rigid_contact_max`` is the capacity, ``rigid_contact_count`` the count.
        shape_body: The model's body of each shape (-1 for a static shape).
        eligible: ``int32`` ``[rigid_contact_max]``: each contact's eligibility with ``_gather``'s semantics (1, -1, 0),
            :func:`scene_torsion_eligibility`.
        body_q: The step's input body poses, from which the contact frames are computed as ``_gather`` does.
        shape_radius: ``float32`` per shape: the model's ``steel:min_torsional_patch_radius`` [m].
        contact_slots_needed: ``int32`` per contact: an anchor holds its friction pair, 3 base rows
            (``contact_base_rows``; Steel's ``_contact_slots_needed``).
        friction: ``float32`` per contact: the coefficient of an anchor's friction rows, Steel's
            ``_row_mu[world, slot + 1]`` (after ``link_patch_rows`` under patches); read at each group's first anchor.
        patch_owner: ``int32`` per contact: the persistent friction patch owner (the region seed), ``None`` without
            patches.
        contact_world: ``int32`` per contact: the solve world of the contact's rows (Steel's ``_contact_world``; the
            scene solve's model world of the contact's unit), which keys a patch region with its owner. Read under
            patches only; ``None`` puts every contact in world 0.
        out: Groups from :meth:`SceneTorsionGroups.allocate` to write into, of at least ``rigid_contact_max``
            contacts; new ones when ``None``.
        device: The device to launch on, ``eligible``'s by default.
    """
    device = wp.get_device(device if device is not None else eligible.device)
    input_capacity = contacts.rigid_contact_max
    if out is None:
        out = SceneTorsionGroups.allocate(input_capacity, device)
    capacity = out.capacity
    if capacity < input_capacity:
        raise ValueError(f"out holds {capacity} contacts, fewer than the contacts' {input_capacity}")
    if capacity == 0:
        out.group_count.zero_()
        out.member_start.zero_()
        return out
    patch_mode = int(patch_owner is not None)
    work = out._work
    wp.launch(
        _scene_torsion_keys,
        dim=capacity,
        inputs=[
            contacts.rigid_contact_count,
            input_capacity,
            eligible,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_normal,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            contacts.rigid_contact_margin0,
            contacts.rigid_contact_margin1,
            shape_body,
            body_q,
            patch_mode,
            patch_owner if patch_owner is not None else out._no_owner,
            contact_world if contact_world is not None else out._no_owner,
            int(contact_world is not None),
            work,
        ],
        outputs=[out.contact_group, out.members, out.lead, out.coefficient, out.radius],
        device=device,
    )
    wp.utils.radix_sort_pairs(work.keys, work.indices, capacity)
    wp.launch(
        _scene_torsion_segments,
        dim=capacity,
        inputs=[
            capacity,
            patch_mode,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contact_slots_needed,
            friction,
            shape_radius,
            work,
        ],
        outputs=[out.lead, out.coefficient, out.radius],
        device=device,
    )
    out._inclusive_scan(work.size, work.member_end)
    out._inclusive_scan(work.row, work.rank_end)
    wp.launch(
        _scene_torsion_emit,
        dim=capacity + 1,
        inputs=[capacity, work],
        outputs=[out.contact_group, out.group_count, out.member_start, out.members],
        device=device,
    )
    return out


# --- Contact torsion (SceneFeatures.torsion): the spin rows, late rows after every unit's ---------------------------

SCENE_TORSION_INPUT_OVERFLOW = 1
"""Bit of ``SceneSolve.torsion_status``: the contact count passed the contact buffer (Steel's "Contact input overflow
before contact torsion")."""
SCENE_TORSION_HYDROELASTIC = 2
"""Bit of ``SceneSolve.torsion_status``: a contact with a positive stiffness (Steel's "Contact torsion does not support
hydroelastic contacts")."""
SCENE_TORSION_ROWS_LOST = 4
"""Bit of ``SceneSolve.torsion_status``: the step lost rows or units before the spin rows. Steel gives a world that
dropped dense rows no dense torsion rows; the scene, whose losses are scene-wide, builds no spin row that step."""
SCENE_TORSION_CAPACITY = 8
"""Bit of ``SceneSolve.torsion_status``: spin rows dropped past the row capacity. The rows that fit are kept, a
prefix in row order, and each dropped row flags its worlds in ``constraint_overflow``, as Steel keeps the torsion rows
of a world that fit and flags the world."""
SCENE_TORSION_REFUSED = SCENE_TORSION_INPUT_OVERFLOW | SCENE_TORSION_HYDROELASTIC | SCENE_TORSION_ROWS_LOST
"""The bits on which a step builds no spin row, as Steel refuses its preparation."""
SCENE_TORSION_MESSAGES = {
    SCENE_TORSION_INPUT_OVERFLOW: "Contact input overflow before contact torsion",
    SCENE_TORSION_HYDROELASTIC: "Contact torsion does not support hydroelastic contacts",
    SCENE_TORSION_ROWS_LOST: "Contact rows lost before contact torsion; no spin rows built",
    SCENE_TORSION_CAPACITY: "Contact torsion rows dropped past the row capacity",
}
"""A message for each bit of ``SceneSolve.torsion_status``, in bit order. Steel's ``DeviceTorsionPreparation.validate``
raises the first two (with these messages); it raises neither of the last two, whose losses Steel records in
``constraint_overflow`` alone, so their messages name the scene's latched cause."""

SCENE_TORSION_ROW_WORKERS = 1024
"""Lane groups of the spin rows' writer (:func:`get_scene_torsion_rows_kernel`), at most one per group."""

_TORSION_REFUSED = wp.constant(SCENE_TORSION_REFUSED)
_TORSION_CAPACITY = wp.constant(SCENE_TORSION_CAPACITY)
_TWO_SIDED_KEY = wp.constant(wp.int64(1 << 62))
_NO_KEY = wp.constant(wp.int64(9223372036854775807))


@wp.kernel(**_ROW_KERNEL_OPTIONS)
def scene_torsion_routing(
    contact_count: wp.array[int],
    contact_capacity: int,
    contact_path: wp.array[int],
    contact_slot: wp.array[int],
    contact_slots_needed: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    is_free_rigid: wp.array[int],
    contact_stiffness: wp.array[float],
    has_stiffness: int,
    row_mu: wp.array2d[float],
    rows_requested: wp.array[int],
    constraint_count: wp.array[int],
    unit_overflow: wp.array[int],
    art_model_world: wp.array[int],
    world_count: int,
    # outputs
    torsion_path: wp.array[int],
    friction: wp.array[float],
    torsion_world: wp.array[int],
    step_status: wp.array[int],
):
    """Route the step's contacts as Steel's torsion preparation sees them, and latch its refusals (grid: contacts).

    ``torsion_path`` is the scene's ``contact_path`` with Steel's free-body (MF) contacts moved off path 0: the scene
    gives every contact with a responding side its rows on path 0, where Steel solves a contact whose every side is a
    free body or has no response DOFs (``_allocate_world_contact_slot``'s ``is_mf``,
    :func:`~newton._src.solvers.steel.scene_units._scene_contact_on_free_path`) on its free-body path, which runs no
    torsion. ``friction`` is the coefficient Steel's grouping reads (``_row_mu[world, slot + 1]`` in
    ``_group_world``): the scene's first friction row of a contact with its tangent pair (3 base rows,
    ``contact_base_rows``), read after every pass that rewrites friction rows (contact compliance), 0 otherwise; under
    friction patches the coefficient after ``link_patch_rows`` divided it by the region's anchors, as Steel's grouping
    reads it. ``torsion_world`` is the model world of the contact's
    unit (:func:`_unit_model_world`, where the row watermarks count its rows): Steel's solve world of its rows, which
    keys a patch region with its owner (``group_scene_torsion``'s ``contact_world``), 0 for a contact without rows.
    ``step_status`` (cleared before the launch) gathers Steel's refusals of the step:
    :data:`SCENE_TORSION_INPUT_OVERFLOW` for a contact count past the buffer, :data:`SCENE_TORSION_HYDROELASTIC` for a
    counted contact with a positive stiffness (``_gather``) and :data:`SCENE_TORSION_ROWS_LOST` when the
    step's units asked for more rows than were kept, or passed ``scene_unit_max``.
    """
    c = wp.tid()
    if c == 0:
        status = int(0)
        if contact_count[0] > contact_capacity or contact_count[0] < 0:
            status = status | SCENE_TORSION_INPUT_OVERFLOW
        if rows_requested[0] > constraint_count[0] or unit_overflow[0] > 0:
            status = status | SCENE_TORSION_ROWS_LOST
        if status != 0:
            wp.atomic_or(step_status, 0, status)
    if c >= wp.min(contact_count[0], contact_capacity):
        torsion_path[c] = -1
        friction[c] = 0.0
        torsion_world[c] = 0
        return
    if has_stiffness != 0:
        if contact_stiffness[c] > 0.0:
            wp.atomic_or(step_status, 0, SCENE_TORSION_HYDROELASTIC)
    world = int(0)
    side_a, side_b = _contact_unit_sides(c, contact_path, contact_art_a, contact_art_b, articulation_response_dof_count)
    if side_a >= 0:
        world = _unit_model_world(side_a, side_b, art_model_world, world_count)
    torsion_world[c] = world
    path = contact_path[c]
    if path == 0:
        if _scene_contact_on_free_path(
            contact_art_a[c], contact_art_b[c], articulation_response_dof_count, is_free_rigid
        ):
            path = 1
    torsion_path[c] = path
    value = float(0.0)
    slot = contact_slot[c]
    if slot >= 0 and contact_base_rows(contact_slots_needed[c]) == 3 and slot + 1 < row_mu.shape[1]:
        value = row_mu[0, slot + 1]
    friction[c] = value


@wp.kernel(**_ROW_KERNEL_OPTIONS)
def scene_torsion_keys(
    group_count: wp.array[int],
    member_start: wp.array[int],
    members: wp.array[int],
    contact_path: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    contact_entry: int,
    unit_slot: wp.array[int],
    unit_color: wp.array[int],
    step_status: wp.array[int],
    # outputs
    keys: wp.array[wp.int64],
    order: wp.array[int],
):
    """Key each torsion group by where its spin row goes (grid: the groups' capacity).

    A group whose members are one-sided contacts (its sides ``side_b < 0`` or ``side_b == side_a``, the units' rule)
    keys ``side_a << 32 | g``: its spin row is a one-sided late row of that side. A two-sided group keys ``1 << 62 |
    color << 32 | g`` with ``color`` its representative contact's unit's (the tail's is the color cap): its spin row is
    swept with that color. The representative's unit is the units' slot of its flag entry ``contact_entry + rep``
    (:class:`~newton._src.solvers.steel.scene_units.SceneUnits`). Every other entry, and every group of a step whose
    ``step_status`` refuses torsion, keys the largest ``int64`` and sorts last; ``order`` is the identity, the
    sort's values.
    """
    g = wp.tid()
    keys[g] = _NO_KEY
    order[g] = g
    if g >= group_count[0] or (step_status[0] & _TORSION_REFUSED) != 0:
        return
    rep = members[member_start[g]]
    side_a, side_b = _contact_unit_sides(
        rep, contact_path, contact_art_a, contact_art_b, articulation_response_dof_count
    )
    if side_b < 0 or side_b == side_a:
        keys[g] = (wp.int64(side_a) << wp.int64(32)) | wp.int64(g)
    else:
        color = unit_color[unit_slot[contact_entry + rep]]
        keys[g] = _TWO_SIDED_KEY | (wp.int64(color) << wp.int64(32)) | wp.int64(g)


@wp.kernel(**_ROW_KERNEL_OPTIONS)
def scene_torsion_counts(
    row_max: int,
    group_count: wp.array[int],
    # in/out
    step_status: wp.array[int],
    rows_requested: wp.array[int],
    constraint_count: wp.array[int],
    torsion_status: wp.array[int],
    # outputs
    torsion_counts: wp.array[int],
):
    """Append the step's spin rows after its kept rows, as many as fit in ``row_max`` (grid: 1).

    ``torsion_counts`` receives the first spin row (the kept rows so far), the spin rows requested (every group's,
    none in a step that refuses torsion) and those kept: the spin rows are the last rows of the order, so they drop
    first, in row order, and the kept ones are a prefix. The rows requested and the row count grow by them, so
    ``scene_statistics()["dropped_rows"]`` counts the dropped ones. A drop adds :data:`SCENE_TORSION_CAPACITY` to the
    step's status, which ``torsion_status`` latches until a reset.
    """
    base = constraint_count[0]
    status = step_status[0]
    requested = group_count[0]
    if (status & _TORSION_REFUSED) != 0:
        requested = 0
    kept = wp.max(wp.min(requested, row_max - base), 0)
    if kept < requested:
        status = status | _TORSION_CAPACITY
    torsion_counts[0] = base
    torsion_counts[1] = requested
    torsion_counts[2] = kept
    rows_requested[0] = rows_requested[0] + requested
    constraint_count[0] = base + kept
    step_status[0] = status
    torsion_status[0] = torsion_status[0] | status


@wp.kernel(**_ROW_KERNEL_OPTIONS)
def scene_torsion_place(
    color_slots: int,
    keys: wp.array[wp.int64],
    order: wp.array[int],
    torsion_counts: wp.array[int],
    member_start: wp.array[int],
    members: wp.array[int],
    contact_path: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    art_model_world: wp.array[int],
    # outputs
    side_torsion_begin: wp.array[int],
    side_torsion_end: wp.array[int],
    torsion_color_start: wp.array[int],
    constraint_overflow: wp.array[wp.bool],
):
    """Place the spin rows in key order from the first spin row (grid: ``max(groups' capacity, color_slots)``).

    Sorted position ``k`` takes row ``base + k``: the one-sided spin rows first, side-major, then the two-sided ones
    by color, the tail's last. A spin row dropped past the capacity latches the model world of each of its sides in
    ``constraint_overflow``. Each side with one-sided spin rows gets their kept range ``[side_torsion_begin,
    side_torsion_end)`` (both cleared before the launch, so every other side's range is empty). Thread ``t <
    color_slots`` (the color cap plus two) writes ``torsion_color_start[t]``, the first kept row of color ``t``'s
    two-sided spin rows (a binary search of the keys); entry ``color_slots - 1`` is the last kept spin row's end.
    """
    k = wp.tid()
    base = torsion_counts[0]
    requested = torsion_counts[1]
    end = base + torsion_counts[2]
    if k < color_slots:
        first = _TWO_SIDED_KEY | (wp.int64(k) << wp.int64(32))
        torsion_color_start[k] = wp.min(base + _lower_bound(keys, first, requested), end)
    if k >= requested:
        return
    key = keys[k]
    row = base + k
    if row >= end:
        rep = members[member_start[order[k]]]
        side_a, side_b = _contact_unit_sides(
            rep, contact_path, contact_art_a, contact_art_b, articulation_response_dof_count
        )
        _latch_side_world(side_a, art_model_world, constraint_overflow)
        if side_b >= 0:
            _latch_side_world(side_b, art_model_world, constraint_overflow)
    if key >= _TWO_SIDED_KEY:
        return
    side = int(key >> wp.int64(32))
    if k == 0 or (keys[k - 1] >> wp.int64(32)) != (key >> wp.int64(32)):
        side_torsion_begin[side] = wp.min(row, end)
    if k == requested - 1 or (keys[k + 1] >> wp.int64(32)) != (key >> wp.int64(32)):
        side_torsion_end[side] = wp.min(row + 1, end)


@wp.kernel(**_ROW_KERNEL_OPTIONS)
def scene_torsion_finish(
    torsion_counts: wp.array[int],
    order: wp.array[int],
    member_start: wp.array[int],
    members: wp.array[int],
    contact_slot: wp.array[int],
    contact_slots_needed: wp.array[int],
    row_dof: wp.array3d[int],
    free_dof_mask: wp.array2d[int],
    target_velocity: wp.array2d[float],
    pgs_cfm: float,
    contact_w: float,
    # outputs
    rhs: wp.array2d[float],
    diag: wp.array2d[float],
    impulses: wp.array2d[float],
    row_free_prefix: wp.array[int],
    row_w: wp.array2d[float],
    torsion_group: wp.array2d[int],
    torsion_member_rows: wp.array[int],
):
    """Finish each kept spin row as :func:`prepare_scene_rows` finishes a row, and list its members (grid: the groups'
    capacity).

    A spin row, written after :func:`prepare_scene_rows` ran, gets that kernel's arithmetic for its type: ``cfm`` on its
    diagonal, then its tail (:func:`_finish_scene_row`): the bias ``-target`` (Steel's angular row has no positional
    term), weight 1 under contact regularization, a zero impulse and its free prefix. Its members, the normal rows of
    its group's contacts (Steel's ``membership[world, member.slot] = row`` in ``prepare_torsion_rows``), get its row in
    ``torsion_group`` (cleared to ``-1`` before the launch, so the spin row's own entry stays ``-1``: admitted) and are
    listed in ``torsion_member_rows`` in the group's member order, each normal row shifted left by one with bit 0 set
    when the contact holds its tangent pair (3 base rows, ``contact_base_rows``, an anchor): the members the spin
    rows' coupled friction budget walks.
    """
    k = wp.tid()
    if k >= torsion_counts[2]:
        return
    row = torsion_counts[0] + k
    g = order[k]
    # prepare_scene_rows' diagonal and tail for a row of its type: cfm (a spin row is not a drive row), the bias
    # -target (no positional term) and weight 1.
    diag[0, row] += pgs_cfm
    _finish_scene_row(
        row, -target_velocity[0, row], 1.0, contact_w, row_dof, free_dof_mask, rhs, row_w, impulses, row_free_prefix
    )
    for m in range(member_start[g], member_start[g + 1]):
        c = members[m]
        normal = contact_slot[c]
        torsion_group[0, normal] = row
        anchor = int(0)
        if contact_base_rows(contact_slots_needed[c]) == 3:
            anchor = 1
        torsion_member_rows[m] = (normal << 1) | anchor


@cache
def get_scene_torsion_rows_kernel(dof_max: int, mask_words: int = 1, *, lanes: int = 8, warps: int = 4) -> wp.Kernel:
    """The spin rows' writer: Steel's ``_apply_rows`` and ``_jacobians`` through the contacts' response.

    One lane group per kept spin row (sorted position ``k``, row ``base + k``), striding over them. Lane 0 writes the
    row's metadata as ``_apply_rows`` does: type ``TORSION``, ``row_parent`` the lead's normal row, ``row_mu`` the
    group's coefficient, ``phi`` and the rebound velocity 0, the target ``-sign * dot(n, omega)`` of each prescribed
    side's body (``body_a`` sign +1, ``body_b`` sign -1, Steel's ``_host_dot3``), its radius (``torsion_radius``), its
    members' range (``torsion_member_range``) and its lead's shapes (``torsion_shape_pair``). The group's lead is its
    first touching member (``SceneTorsionGroups.lead``), whose normal ``n`` (B to A) gives the row ``J = (0, n)``: the
    contact response's projection (:func:`~newton._src.solvers.steel.scene_rows.scene_contact_response_snippet`, the
    group-table kernel's arithmetic) of a zero linear direction and the angular part ``n`` on side A and ``-n`` on side
    B, so a row of ``J (H^-1 J^T)`` in the factor's coordinates of each articulated side and the six physical entries of
    a free one, with its incident velocity ``J v_hat`` and diagonal ``J H^-1 J^T``. Steel's ``_jacobians`` sums
    ``sign * dot(n, S[3:])`` over each side's ancestor DOFs (its body's support mask here). Launch with
    ``block_dim = 32 * warps``.
    """
    snippet = scene_contact_response_snippet(dof_max, mask_words, lanes_per_contact=lanes, warps_per_block=warps)

    @wp.func_native(snippet)
    def project_native(
        tid: int,
        body_a: int,
        body_b: int,
        art_a: int,
        art_b: int,
        world: int,
        slot: int,
        row_count: int,
        angular_a: wp.mat33,
        angular_b: wp.mat33,
        directions: wp.mat33,
        art_group_idx: wp.array[int],
        articulation_response_dof_count: wp.array[int],
        is_free_rigid: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        articulation_dof_start: wp.array[int],
        body_dof_mask: wp.array[wp.uint64],
        joint_S_s: wp.array[wp.spatial_vector],
        art_group: wp.array[int],
        art_slot: wp.array[int],
        group_dof_count: wp.array[int],
        group_nonzero_count: wp.array[int],
        group_linv_offset: wp.array[int],
        group_permutation_base: wp.array[int],
        group_row_offset_base: wp.array[int],
        permutation: wp.array[int],
        row_offsets: wp.array[int],
        Linv: wp.array[float],
        free_inverse_factor: wp.array3d[float],
        v_hat: wp.array[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_free_response: wp.array3d[float],
        row_incident: wp.array2d[float],
        diagonal: wp.array2d[float],
    ): ...

    def torsion_rows_kernel(  # pragma: no cover - compiled by wp.kernel
        torsion_counts: wp.array[int],
        workers: int,
        order: wp.array[int],
        member_start: wp.array[int],
        members: wp.array[int],
        lead: wp.array[int],
        coefficient: wp.array[float],
        radius: wp.array[float],
        contact_normal: wp.array[wp.vec3],
        contact_shape0: wp.array[int],
        contact_shape1: wp.array[int],
        contact_slot: wp.array[int],
        contact_art_a: wp.array[int],
        contact_art_b: wp.array[int],
        shape_body: wp.array[int],
        body_articulation: wp.array[int],
        prescribed_articulation: wp.array[int],
        body_v_s: wp.array[wp.spatial_vector],
        art_group_idx: wp.array[int],
        articulation_response_dof_count: wp.array[int],
        is_free_rigid: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        articulation_dof_start: wp.array[int],
        body_dof_mask: wp.array[wp.uint64],
        joint_S_s: wp.array[wp.spatial_vector],
        art_group: wp.array[int],
        art_slot: wp.array[int],
        group_dof_count: wp.array[int],
        group_nonzero_count: wp.array[int],
        group_linv_offset: wp.array[int],
        group_permutation_base: wp.array[int],
        group_row_offset_base: wp.array[int],
        permutation: wp.array[int],
        row_offsets: wp.array[int],
        Linv: wp.array[float],
        free_inverse_factor: wp.array3d[float],
        v_hat: wp.array[float],
        # outputs
        row_type: wp.array2d[int],
        row_parent: wp.array2d[int],
        row_mu: wp.array2d[float],
        phi: wp.array2d[float],
        target_velocity: wp.array2d[float],
        row_rebound_velocity: wp.array2d[float],
        torsion_radius: wp.array2d[float],
        torsion_member_range: wp.array2d[int],
        torsion_shape_pair: wp.array2d[int],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_free_response: wp.array3d[float],
        row_incident: wp.array2d[float],
        diagonal: wp.array2d[float],
    ):
        tid = wp.tid()
        worker = tid // lanes
        base = torsion_counts[0]
        for k in range(worker, torsion_counts[2], workers):
            g = order[k]
            row = base + k
            rep = members[member_start[g]]
            c = lead[rep]
            body_a = int(-1)
            body_b = int(-1)
            if contact_shape0[c] >= 0:
                body_a = shape_body[contact_shape0[c]]
            if contact_shape1[c] >= 0:
                body_b = shape_body[contact_shape1[c]]
            # The contact normal is stored A to B; Steel's spin row takes the row normal, B to A.
            normal = -contact_normal[c]
            angular = wp.mat33(0.0)
            for axis in range(3):
                angular[0, axis] = normal[axis]
            if tid % lanes == 0:
                row_type[0, row] = PGS_CONSTRAINT_TYPE_TORSION
                row_parent[0, row] = contact_slot[c]
                row_mu[0, row] = coefficient[rep]
                phi[0, row] = 0.0
                row_rebound_velocity[0, row] = 0.0
                torsion_radius[0, row] = radius[rep]
                torsion_member_range[row, 0] = member_start[g]
                torsion_member_range[row, 1] = member_start[g + 1]
                # The lead's shapes, Steel's statistics' shape pair, kept with the step's rows.
                torsion_shape_pair[row, 0] = contact_shape0[c]
                torsion_shape_pair[row, 1] = contact_shape1[c]
                # A prescribed side moves the row's target, as Steel's _apply_rows gives it (_prescribed_spin).
                velocity = float(0.0)
                for side in range(2):
                    body = body_a
                    sign = float(1.0)
                    if side == 1:
                        body = body_b
                        sign = -1.0
                    if body >= 0:
                        art = body_articulation[body]
                        if art >= 0:
                            if prescribed_articulation[art] != 0:
                                velocity -= sign * _host_dot3(normal, wp.spatial_bottom(body_v_s[body]))
                target_velocity[0, row] = velocity
            project_native(
                tid,
                body_a,
                body_b,
                contact_art_a[c],
                contact_art_b[c],
                0,
                row,
                1,
                angular,
                angular,
                wp.mat33(0.0),
                art_group_idx,
                articulation_response_dof_count,
                is_free_rigid,
                articulation_world_dof_offset,
                articulation_dof_start,
                body_dof_mask,
                joint_S_s,
                art_group,
                art_slot,
                group_dof_count,
                group_nonzero_count,
                group_linv_offset,
                group_permutation_base,
                group_row_offset_base,
                permutation,
                row_offsets,
                Linv,
                free_inverse_factor,
                v_hat,
                row_dof,
                row_factor,
                row_free_response,
                row_incident,
                diagonal,
            )

    name = f"steel_scene_torsion_rows_{int(dof_max)}_{int(lanes)}_{int(warps)}" + (
        f"_w{int(mask_words)}" if mask_words > 1 else ""
    )
    torsion_rows_kernel.__name__ = name
    torsion_rows_kernel.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(torsion_rows_kernel)


@wp.kernel(**_ROW_KERNEL_OPTIONS)
def prepare_scene_velocity_torsion(
    torsion_counts: wp.array[int],
    torsion_member_range: wp.array2d[int],
    torsion_member_rows: wp.array[int],
    phi: wp.array2d[float],
    target_velocity: wp.array2d[float],
    row_incident: wp.array2d[float],
    row_dof: wp.array3d[int],
    row_factor: wp.array3d[float],
    delta: wp.array2d[float],
    steel_sums: int,
    dt: float,
    # outputs
    torsion_group: wp.array2d[int],
):
    """Conclude each kept spin row's admission for the velocity-only iterations (grid: the groups' capacity).

    Steel's ``_prepare_velocity_torsion_rows`` on the scene's rows: a spin row stays admitted (``-1``) when a member's
    normal row touches, ``phi <= _CONTACT_END_GAP_SLOP`` or its end gap after the position solve
    ``phi + dt * (J v_pos - target) <= _CONTACT_END_GAP_SLOP``, with ``J v_pos`` formed as
    :func:`prepare_scene_velocity_rows` forms it (``J v_hat + Z delta``, :func:`_scene_row_dot`); otherwise it is
    retired (``_TORSION_SPIN_RETIRED``), and the coupled friction budget bounds it by 0, which refunds its spin
    impulse. Steel's ``rhs =
    -target`` of the spin row is :func:`prepare_scene_velocity_rows`' for a row of its type; the impulses are kept.
    """
    k = wp.tid()
    if k >= torsion_counts[2]:
        return
    spin = torsion_counts[0] + k
    touching = bool(False)
    for m in range(torsion_member_range[spin, 0], torsion_member_range[spin, 1]):
        n = torsion_member_rows[m] >> 1
        speed = row_incident[0, n] + _scene_row_dot(row_dof, row_factor, delta, n, steel_sums)
        end_gap = phi[0, n] + dt * (speed - target_velocity[0, n])
        # Retain touching support through rebound, with the end-gap slop as the touching tolerance (Steel's).
        if phi[0, n] <= _CONTACT_END_GAP_SLOP or end_gap <= _CONTACT_END_GAP_SLOP:
            touching = True
    if touching:
        torsion_group[0, spin] = -1
    else:
        torsion_group[0, spin] = _TORSION_SPIN_RETIRED
