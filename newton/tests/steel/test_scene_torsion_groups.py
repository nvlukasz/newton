# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The contact torsion groups of the scene solve: Steel's partition with one thread per cluster key.

Steel groups the contact witnesses of contact torsion per solve world (``_group_world``, one thread per world, the
world's contacts in contact order): an ordered greedy clustering of each shape pair's eligible contacts by normal and
contact plane, or one region per persistent friction patch, then the clusters that get a spin row. After the scene's
fold one world holds every contact of the scene, so ``group_scene_torsion`` sorts the contacts by cluster key (the
stored shape pair, or the patch owner) and runs the same two passes with one thread per key segment;
``scene_torsion_eligibility`` computes ``_gather``'s eligibility from the contacts and the routing arrays, with the
shapes' torsion support marked by Steel's rule (``SceneTorsionShapes``, also across body variants).

The scene's partition is compared with Steel's on a stepped multi-world Steel solver, with Steel's device preparation
on ``test_torsion``'s multi-world synthetic witnesses (segments of one contact and of many), and on hand-built
contacts rule by rule; it is checked to be deterministic and capturable.
"""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel.contact_torsion import _SUPPORTED_TYPES
from newton._src.solvers.steel.contact_torsion_device import DeviceTorsionPreparation
from newton._src.solvers.steel.scene_torsion_device import (
    SceneTorsionGroups,
    SceneTorsionShapes,
    group_scene_torsion,
    scene_torsion_eligibility,
)
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import DT, PATCH_OPTIONS, set_torsion_radius
from newton.tests.steel.test_torsion import synthetic_fixture
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

ANCHOR = 3
"""``contact_slots_needed`` of a contact with its friction pair: an anchor."""

NORMAL_ONLY = 1
"""``contact_slots_needed`` of a contact with its normal row alone."""

SHAPE_RADIUS = 0.01
"""Torsion radius [m] of the hand-built contacts' shapes."""

FRICTION = 0.5
"""Friction coefficient of the hand-built contacts' anchors."""

APART = 1.0e-3
"""Gap [m] of a hand-built contact that is not touching: above ``_TOUCH_TOLERANCE`` (1e-5 m)."""

PAD_HALF = (0.02, 0.015, 0.025)
"""Half extents [m] of the stacked pads."""

PAD_WORLDS = 4
"""Worlds of the stacked pads: as built, the top pad lifted, the middle pad turned, and no torsion radius."""

SEGMENT_WITNESSES = 256
"""Witnesses of the one-world synthetic fixture: two shape pairs of 128 contacts each."""

RANDOM_CONTACTS = 400
"""Contacts of the random fixture of the determinism and capture tests."""

OUTPUTS = ("contact_group", "group_count", "member_start", "members", "lead", "coefficient", "radius")
"""The arrays of :class:`SceneTorsionGroups`."""


def _contacts(device, pairs, normals, points, *, gaps=None, capacity=None):
    """Hand-built contacts between static shapes: contact ``i`` joins ``pairs[i]`` at world point ``points[i]``.

    ``normals[i]`` is the row normal (B to A, Steel's ``-rigid_contact_normal``), normalized here; ``gaps[i]`` [m]
    separates the two contact points along it (0, touching, by default). ``capacity`` (the count by default) is the
    buffer's ``rigid_contact_max``.
    """
    count = len(pairs)
    capacity = count if capacity is None else capacity
    contacts = newton.Contacts(capacity, 0, device=device)
    contacts.rigid_contact_count.assign(np.array([count], np.int32))
    if count == 0:
        return contacts
    pairs = np.asarray(pairs, np.int32).reshape(count, 2)
    normals = np.asarray(normals, np.float32).reshape(count, 3)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    points = np.asarray(points, np.float32).reshape(count, 3)
    gaps = np.zeros(count, np.float32) if gaps is None else np.asarray(gaps, np.float32)
    values = {
        "rigid_contact_shape0": pairs[:, 0],
        "rigid_contact_shape1": pairs[:, 1],
        "rigid_contact_normal": -normals,
        "rigid_contact_point0": points + gaps[:, None] * normals,
        "rigid_contact_point1": points,
    }
    for name, value in values.items():
        array = getattr(contacts, name)
        full = array.numpy()
        full[:count] = value
        array.assign(full)
    return contacts


def _int_array(values, device):
    return wp.array(np.asarray(values, np.int32), dtype=int, device=device)


def _float_array(values, device):
    return wp.array(np.asarray(values, np.float32), dtype=float, device=device)


def _group(contacts, shape_count, eligible, slots_needed, friction, *, radius=None, owner=None, out=None):
    """Group hand-built contacts between ``shape_count`` static shapes (no body poses are read)."""
    device = contacts.rigid_contact_count.device
    radius = np.full(shape_count, SHAPE_RADIUS) if radius is None else radius
    return group_scene_torsion(
        contacts,
        wp.full(shape_count, -1, dtype=int, device=device),
        _int_array(eligible, device),
        body_q=wp.zeros(1, dtype=wp.transform, device=device),
        shape_radius=_float_array(radius, device),
        contact_slots_needed=_int_array(slots_needed, device),
        friction=_float_array(friction, device),
        patch_owner=None if owner is None else _int_array(owner, device),
        out=out,
    )


def _clusters(test, groups):
    """The groups as ``{representative: members}``, after checking the compacted lists against ``contact_group``.

    Groups are in representative order, members in contact order with the representative first, every
    ``member_start`` entry from ``group_count`` on is the member total and ``members`` is -1 past it.
    """
    contact_group = groups.contact_group.numpy()
    count = int(groups.group_count.numpy()[0])
    start = groups.member_start.numpy()
    members = groups.members.numpy()
    test.assertEqual(start.shape[0], contact_group.shape[0] + 1)
    test.assertEqual(int(start[0]), 0)
    total = int(start[count])
    np.testing.assert_array_equal(start[count:], total)
    np.testing.assert_array_equal(members[total:], -1)
    clusters = {}
    for g in range(count):
        cluster = members[start[g] : start[g + 1]]
        test.assertGreater(cluster.size, 0)
        test.assertTrue(np.all(np.diff(cluster) > 0), f"group {g} is not in contact order: {cluster}")
        clusters[int(cluster[0])] = tuple(int(c) for c in cluster)
    test.assertEqual(list(clusters), sorted(clusters))
    test.assertEqual(len(clusters), count)
    expected = np.full_like(contact_group, -1)
    for rep, cluster in clusters.items():
        expected[list(cluster)] = rep
    np.testing.assert_array_equal(contact_group, expected)
    return clusters


def _partition(test, groups):
    """The groups as a set of frozensets of contacts."""
    return {frozenset(cluster) for cluster in _clusters(test, groups).values()}


def _steel_partition(solver, contacts):
    """Steel's torsion partition: contact ``c``'s spin row is ``_contact_torsion_group[world_c, _contact_slot[c]]``."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    path = solver._contact_path.numpy()[:count]
    slot = solver._contact_slot.numpy()[:count]
    world = solver._contact_world.numpy()[:count]
    membership = solver._contact_torsion_group.numpy()
    rows = {}
    for c in range(count):
        if path[c] == 0 and slot[c] >= 0 and membership[world[c], slot[c]] >= 0:
            rows.setdefault((int(world[c]), int(membership[world[c], slot[c]])), set()).add(c)
    return {frozenset(cluster) for cluster in rows.values()}


def _anchor_friction(solver, contacts):
    """Each contact's friction row coefficient ``_row_mu[world, slot + 1]`` as Steel's preparation reads it (0 for a
    contact without its friction pair)."""
    capacity = contacts.rigid_contact_max
    slot = solver._contact_slot.numpy()[:capacity]
    world = solver._contact_world.numpy()[:capacity]
    needed = solver._contact_slots_needed.numpy()[:capacity]
    mu = solver._row_mu.numpy()
    kept = (slot >= 0) & (needed == ANCHOR) & (world >= 0) & (world < mu.shape[0]) & (slot + 1 < mu.shape[1])
    friction = np.zeros(capacity, np.float32)
    friction[kept] = mu[world[kept], slot[kept] + 1]
    return wp.array(friction, dtype=float, device=solver._row_mu.device)


def _scene_groups(solver, state, contacts, *, out=None):
    """The scene's groups of a Steel solver's contacts, with the inputs built from the model and the contact routing.

    Returns ``(groups, eligible)``.
    """
    model = solver.model
    patches = bool(solver._friction_anchors_enabled)
    common = {
        "body_q": state.body_q,
        "shape_radius": solver._contact_torsion_shape_radius,
        "contact_slots_needed": solver._contact_slots_needed,
    }
    eligible = scene_torsion_eligibility(
        contacts,
        model.shape_body,
        shape_supported=SceneTorsionShapes.allocate(model).supported,
        contact_path=solver._contact_path,
        contact_slot=solver._contact_slot,
        patches=patches,
        **common,
    )
    groups = group_scene_torsion(
        contacts,
        model.shape_body,
        eligible,
        friction=_anchor_friction(solver, contacts),
        patch_owner=solver._friction_patches.current.owner if patches else None,
        contact_world=solver._contact_world,
        out=out,
        **common,
    )
    return groups, eligible


def _assert_rows_match(test, groups, work):
    """Each Steel cluster with a spin row is a scene group with Steel's lead, coefficient and radius, bit for bit."""
    reps = np.flatnonzero(work.lead.numpy() >= 0)
    contact_group = groups.contact_group.numpy()
    np.testing.assert_array_equal(contact_group[reps], reps)
    test.assertEqual(int(groups.group_count.numpy()[0]), reps.size)
    for name in ("lead", "coefficient", "radius"):
        np.testing.assert_array_equal(getattr(groups, name).numpy()[reps], getattr(work, name).numpy()[reps], name)


def _stacked_pads(device):
    """:data:`PAD_WORLDS` worlds of three equal articulated pads stacked face to face along z, without gravity.

    Each pad is a free-moving D6 link (``fixture``'s pads); the bottom and top pads close on the middle one at
    0.1 m/s and spin against it, so each world has two shape pairs in contact, and the bottom and top pads, 5 cm
    apart, have speculative witnesses: not touching, so ineligible without patches and a region without a touching
    member under them. World 1 lifts its top pad 4 mm (its pair's witnesses are gapped too), world 2 turns its middle
    pad 0.3 rad about z, and world 3's shapes get no torsion radius (rejected witnesses). With patches, half of the
    witnesses are anchors. Steel gives five spin rows: two in worlds 0 and 2, one in world 1.

    Returns ``(model, state)``.
    """
    pads = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    axes = [newton.ModelBuilder.JointDofConfig(axis=wp.vec3(*a)) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    inertia = wp.mat33(np.diag([0.00012, 0.00012, 0.00008]).astype(np.float32))
    for level in (-1, 0, 1):
        pose = wp.transform(wp.vec3(0.0, 0.0, 2.0 * PAD_HALF[2] * level), wp.quat_identity())
        body = pads.add_link(xform=pose, mass=0.3, inertia=inertia)
        joint = pads.add_joint_d6(-1, body, parent_xform=pose, linear_axes=axes, angular_axes=axes)
        pads.add_articulation([joint])
        pads.add_shape_box(
            body, hx=PAD_HALF[0], hy=PAD_HALF[1], hz=PAD_HALF[2], cfg=newton.ModelBuilder.ShapeConfig(density=0, mu=0.5)
        )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(pads, PAD_WORLDS, spacing=(1.0, 0.0, 0.0))
    model = builder.finalize(device=device)
    set_torsion_radius(model, SHAPE_RADIUS, np.flatnonzero(model.shape_world.numpy() != PAD_WORLDS - 1))
    # Six coordinates and six DOFs per pad: x, y, z, then the three angles.
    joint_q = model.joint_q.numpy().reshape(PAD_WORLDS, 3, 6)
    joint_q[1, 2, 2] += 0.004
    joint_q[2, 1, 5] += 0.3
    joint_qd = np.zeros((PAD_WORLDS, 3, 6), np.float32)
    joint_qd[:, 0, 2], joint_qd[:, 2, 2] = 0.1, -0.1
    joint_qd[:, 0, 5], joint_qd[:, 2, 5] = 1.0, -1.0
    state = model.state()
    state.joint_q.assign(joint_q.ravel())
    state.joint_qd.assign(joint_qd.ravel())
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state


def test_scene_torsion_groups_match_steel_grouping(test, device):
    """Group a stepped Steel solver's contacts as Steel's device preparation did, with and without patches.

    Steel (``contact_torsion_device=True``) steps the stacked pads once; its partition is read from
    ``_contact_torsion_group`` (-1 unless a contact's cluster got a spin row). The scene's groups of the same
    contacts, its eligibility built from the model and Steel's contact routing, are the same sets of contacts, and its
    lead, coefficient and radius are Steel's bit for bit. ``test_torsion``'s ``synthetic_fixture`` is a mock model and
    solver that cannot be stepped, so this test steps the stacked pads instead and
    ``test_scene_torsion_groups_match_device_preparation`` matches the synthetic fixture.
    """
    for patches in (False, True):
        with test.subTest(patches=patches):
            model, state = _stacked_pads(device)
            pipeline = newton.CollisionPipeline(
                model,
                contact_matching="latest",
                reduce_contacts=False,
                broad_phase="nxn",
                rigid_contact_max=64 * PAD_WORLDS,
            )
            contacts = pipeline.contacts()
            options = PATCH_OPTIONS if patches else {"friction_anchor_beta": 0.0}
            solver = SolverSteel(
                model,
                solve_mode="serial",
                contact_torsion_device=True,
                iterations=16,
                velocity_iterations=0,
                contact_warmstart=False,
                articulation_constraint_max=128,
                free_body_constraint_max=16,
                **options,
            )
            pipeline.collide(state, contacts)
            solver.step(state, model.state(), model.control(), contacts, DT)
            solver.check_constraint_capacity()
            expected = _steel_partition(solver, contacts)
            groups, eligible = _scene_groups(solver, state, contacts)
            test.assertEqual(_partition(test, groups), expected)
            _assert_rows_match(test, groups, solver._device_torsion.work)
            capacity = contacts.rigid_contact_max
            np.testing.assert_array_equal(eligible.numpy(), solver._device_torsion.work.eligible.numpy()[:capacity])
            # Two shape pairs touch in worlds 0 and 2, one in world 1; world 3 has contacts and no radius.
            test.assertEqual(len(expected), 5)
            count = int(contacts.rigid_contact_count.numpy()[0])
            shape_world = model.shape_world.numpy()[contacts.rigid_contact_shape0.numpy()[:count]]
            test.assertTrue(np.any(shape_world == PAD_WORLDS - 1))
            test.assertTrue(np.all(groups.contact_group.numpy()[:count][shape_world == PAD_WORLDS - 1] == -1))


def _modify_synthetic(scenario, solver, contacts, seed):
    """Edit ``synthetic_fixture``'s witnesses like ``test_torsion``'s device preparation tests do."""
    device = contacts.rigid_contact_count.device
    if scenario == "nonanchor":
        anchors = solver._contact_slots_needed.numpy()
        anchors[::2] = NORMAL_ONLY
        solver._contact_slots_needed.assign(anchors)
    elif scenario == "positive_gap":
        points = contacts.rigid_contact_point0.numpy()
        points[:, 2] = 0.002
        contacts.rigid_contact_point0.assign(points)
    elif scenario == "selector":
        radii = np.zeros(solver.model.shape_count, dtype=np.float32)
        radii[0] = SHAPE_RADIUS
        solver._contact_torsion_shape_radius = wp.array(radii, dtype=float, device=device)
    elif scenario == "nontransitive":
        # Adjacent normals match, but the endpoints do not: the greedy clustering tests every member.
        angles = (np.arange(contacts.rigid_contact_max) % 3) * 0.035
        normals = np.column_stack((np.sin(angles), np.zeros_like(angles), -np.cos(angles)))
        contacts.rigid_contact_normal.assign(normals.astype(np.float32))
        contacts.rigid_contact_point0.zero_()
        contacts.rigid_contact_point1.zero_()
    elif scenario == "rejected":
        types = solver.model.shape_type.numpy()
        types[0] = int(newton.GeoType.MESH)
        solver.model.shape_type.assign(types)
        normals = contacts.rigid_contact_normal.numpy()
        normals[3] = (1, 0, 0)
        contacts.rigid_contact_normal.assign(normals)
        points = contacts.rigid_contact_point0.numpy()
        points[6, 2] = 0.001
        contacts.rigid_contact_point0.assign(points)
    elif scenario == "randomized":
        rng = np.random.default_rng(seed)
        positive = rng.choice((0.01, 0.03), solver.model.shape_count)
        radii = np.where(rng.random(solver.model.shape_count) < 0.6, positive, 0.0).astype(np.float32)
        solver._contact_torsion_shape_radius = wp.array(radii, dtype=float, device=device)
        anchors = solver._contact_slots_needed.numpy()
        anchors[rng.random(anchors.size) < 0.3] = NORMAL_ONLY
        solver._contact_slots_needed.assign(anchors)
        points = contacts.rigid_contact_point0.numpy()
        points[rng.random(len(points)) < 0.3, 2] = 0.002
        contacts.rigid_contact_point0.assign(points)


def _match_preparation(test, solver, state, augmented, contacts):
    """Run Steel's device preparation (the per-world scan, ``_group_world``) and the scene's grouping on the same
    witnesses; check the eligibility, the frames, the partition and the rows. Returns the scene's groups."""
    preparer = DeviceTorsionPreparation(solver, patch_segments=False)
    preparer.prepare(state, augmented, contacts)
    groups, eligible = _scene_groups(solver, state, contacts)
    work = preparer.work
    np.testing.assert_array_equal(eligible.numpy(), work.eligible.numpy())
    kept = np.flatnonzero(eligible.numpy() == 1)
    for name in ("normal", "point", "gap"):
        np.testing.assert_array_equal(getattr(groups._work, name).numpy()[kept], getattr(work, name).numpy()[kept])
    test.assertEqual(_partition(test, groups), _steel_partition(solver, contacts))
    _assert_rows_match(test, groups, work)
    return groups


def test_scene_torsion_groups_match_device_preparation(test, device):
    """Match Steel's device preparation on the multi-world synthetic witnesses of ``test_torsion``.

    The scenarios of its device preparation tests (anchor-free contacts, gapped contacts, a radius on one shape,
    non-transitive normals, a rejected region and a turned normal, randomized radii, anchors, gaps and owners), each
    with and without patches, over four shuffled worlds: the eligibility and the frames bit for bit, the partition,
    and the lead, coefficient and radius of every cluster with a row.
    """
    scenarios = ("plain", "nonanchor", "positive_gap", "selector", "nontransitive", "rejected")
    cases = [(scenario, 0) for scenario in scenarios] + [("randomized", seed) for seed in range(4)]
    for patches in (False, True):
        for scenario, seed in cases:
            with test.subTest(patches=patches, scenario=scenario, seed=seed):
                solver, state, augmented, contacts = synthetic_fixture(
                    worlds=4, witnesses=12, patches=patches, seed=seed, device=device
                )
                _modify_synthetic(scenario, solver, contacts, seed)
                if scenario == "randomized" and patches and seed % 2:
                    # A negative owner: one region of one world, whose witnesses Steel pools within the world.
                    owner = solver._friction_patches.current.owner.numpy()
                    owner[owner == owner.max()] = -1
                    solver._friction_patches.current.owner.assign(owner)
                _match_preparation(test, solver, state, augmented, contacts)


def test_scene_torsion_groups_segment_sizes(test, device):
    """Match Steel's device preparation on segments of one contact and of many.

    512 worlds of one witness each give 512 segments of one contact; one world of :data:`SEGMENT_WITNESSES`
    witnesses with non-transitive normals gives two segments (its two shape pairs, or patch regions) of 128 contacts.
    """
    for patches in (False, True):
        with test.subTest(patches=patches, segments="one contact"):
            solver, state, augmented, contacts = synthetic_fixture(
                worlds=512, witnesses=1, patches=patches, device=device
            )
            groups = _match_preparation(test, solver, state, augmented, contacts)
            test.assertEqual(int(groups.group_count.numpy()[0]), 512)
        with test.subTest(patches=patches, segments="many contacts"):
            solver, state, augmented, contacts = synthetic_fixture(
                worlds=1, witnesses=SEGMENT_WITNESSES, patches=patches, device=device
            )
            _modify_synthetic("nontransitive", solver, contacts, 0)
            groups = _match_preparation(test, solver, state, augmented, contacts)
            sizes = np.diff(groups.member_start.numpy()[: int(groups.group_count.numpy()[0]) + 1])
            if patches:
                np.testing.assert_array_equal(sizes, [SEGMENT_WITNESSES // 2] * 2)
            else:
                test.assertGreater(sizes.size, 2)
                test.assertGreater(int(sizes.max()), 1)
            test.assertEqual(int(sizes.sum()), SEGMENT_WITNESSES)


def test_scene_torsion_groups_cluster_pairs_and_normals(test, device):
    """Cluster each stored shape pair's contacts by normal and contact plane, greedily in contact order.

    Pair (0, 1) has three coplanar contacts with one normal, the third after contacts of other pairs: one group. Pair
    (2, 3) has a contact, one with a normal 90 degrees off and one 1 cm off its plane: three groups. Pair (1, 0) is
    pair (0, 1) stored the other way: its own group. Pair (4, 5) has normals 0.035, 0, 0.07 and 0.035 rad from z: the
    second joins the first; the third is close enough to the first but not to the second, so it seeds its own
    cluster (every member is tested, not the representative alone); the fourth joins the first cluster. The group's
    radius is the largest of its shapes' radii.
    """
    tilt = [(np.sin(a), 0.0, np.cos(a)) for a in (0.035, 0.0, 0.07, 0.035)]
    pairs = [(0, 1), (0, 1), (2, 3), (2, 3), (2, 3), (0, 1), (1, 0), (4, 5), (4, 5), (4, 5), (4, 5)]
    normals = [(0, 0, 1)] * 3 + [(1, 0, 0), (0, 0, 1), (0, 0, 1), (0, 0, 1), *tilt]
    points = [(0, 0, 0), (0.1, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0.01), (0, 0.1, 0), (0.2, 0, 0)] + [(0, 0, 0)] * 4
    radius = np.full(6, SHAPE_RADIUS, np.float32)
    radius[1] = 2.0 * SHAPE_RADIUS
    n = len(pairs)
    contacts = _contacts(device, pairs, normals, points)
    groups = _group(contacts, 6, [1] * n, [ANCHOR] * n, [FRICTION] * n, radius=radius)
    clusters = _clusters(test, groups)
    test.assertEqual(clusters, {0: (0, 1, 5), 2: (2,), 3: (3,), 4: (4,), 6: (6,), 7: (7, 8, 10), 9: (9,)})
    test.assertEqual(groups.lead.numpy()[list(clusters)].tolist(), list(clusters))
    np.testing.assert_array_equal(groups.coefficient.numpy()[list(clusters)], np.float32(FRICTION))
    np.testing.assert_array_equal(groups.radius.numpy()[[0, 6]], np.float32(2.0 * SHAPE_RADIUS))
    np.testing.assert_array_equal(groups.radius.numpy()[[2, 3, 4, 7, 9]], np.float32(SHAPE_RADIUS))
    np.testing.assert_array_equal(groups.member_start.numpy()[:8], [0, 3, 4, 5, 6, 7, 10, 11])


def test_scene_torsion_groups_drop_clusters_without_a_row(test, device):
    """Give no group to a cluster without an anchor, one with a non-positive coefficient, or an ineligible pair.

    Pair (0, 1) has no anchor; pair (2, 3) a zero coefficient and pair (4, 5) a negative one; pair (6, 7) is not
    eligible (a world without eligible contacts); pair (8, 9) gets its group, beside a rejected witness (-1) of its
    pair that does not reject it without patches and is no member; pair (10, 11) is eligible but not touching (only a
    caller can mark it so without patches) and has no touching member.
    """
    pairs = [(0, 1), (0, 1), (2, 3), (4, 5), (6, 7), (6, 7), (8, 9), (8, 9), (10, 11)]
    n = len(pairs)
    contacts = _contacts(device, pairs, [(0, 0, 1)] * n, [(0, 0, 0)] * n, gaps=[0.0] * 8 + [APART])
    eligible = [1, 1, 1, 1, 0, 0, -1, 1, 1]
    slots_needed = [NORMAL_ONLY, NORMAL_ONLY] + [ANCHOR] * 7
    friction = [FRICTION, FRICTION, 0.0, -FRICTION, FRICTION, FRICTION, FRICTION, FRICTION, FRICTION]
    groups = _group(contacts, 12, eligible, slots_needed, friction)
    test.assertEqual(_clusters(test, groups), {7: (7,)})
    test.assertEqual(int(groups.lead.numpy()[7]), 7)


def test_scene_torsion_groups_with_patches(test, device):
    """Group by patch owner: one region per owner across shape pairs and normals, rejected whole by a witness.

    Owner 0 holds contacts of pairs (0, 1) and (2, 1) with different normals, two anchors (one gapped) and a
    non-anchor: one group whose coefficient is the first anchor's times its two anchors, led by its first touching
    member. Owner 3's region has a rejected witness (-1), before its eligible contact: no group. Owner 5's only anchor
    is gapped and a non-anchor touches: a group led by the non-anchor. Owner 7 has no anchor, owner 9 no touching
    member, owner 10 a zero coefficient: no group. The radius is the largest over the members' shapes. On
    ``test_torsion``'s synthetic witnesses of four worlds, with one region of worlds 1 and 2 without an owner (-1),
    the partition is Steel's device preparation's: each world's owner-less witnesses form a region of their own, as
    Steel keys a region by its solve world and its owner (``contact_world``).
    """
    pairs = [(0, 1), (2, 1), (0, 1), (4, 5), (4, 5), (6, 7), (6, 7), (8, 9), (8, 9), (10, 11), (12, 13)]
    normals = [(0, 0, 1), (1, 0, 0), (0, 0, 1)] + [(0, 0, 1)] * 8
    gaps = [0.0, 0.0, APART, 0.0, 0.0, APART, 0.0, 0.0, 0.0, APART, 0.0]
    n = len(pairs)
    contacts = _contacts(device, pairs, normals, [(0, 0, 0)] * n, gaps=gaps)
    owner = [0, 0, 0, 3, 3, 5, 5, 7, 7, 9, 10]
    eligible = [1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1]
    slots_needed = [ANCHOR, NORMAL_ONLY, ANCHOR, ANCHOR, ANCHOR, ANCHOR, NORMAL_ONLY, NORMAL_ONLY, NORMAL_ONLY]
    slots_needed += [ANCHOR, ANCHOR]
    friction = [0.25, 0.0, 0.75] + [FRICTION] * 7 + [0.0]
    radius = np.full(14, SHAPE_RADIUS, np.float32)
    radius[2] = 3.0 * SHAPE_RADIUS
    groups = _group(contacts, 14, eligible, slots_needed, friction, radius=radius, owner=owner)
    test.assertEqual(_clusters(test, groups), {0: (0, 1, 2), 5: (5, 6)})
    test.assertEqual(groups.lead.numpy()[[0, 5]].tolist(), [0, 6])
    np.testing.assert_array_equal(groups.coefficient.numpy()[[0, 5]], np.float32([0.5, FRICTION]))
    np.testing.assert_array_equal(groups.radius.numpy()[[0, 5]], np.float32([3.0 * SHAPE_RADIUS, SHAPE_RADIUS]))
    # Owner-less witnesses in two worlds: Steel keys a region by its solve world and its owner, so each world's pool
    # of witnesses without an owner is a region of its own, as in Steel's device preparation.
    solver, state, augmented, contacts = synthetic_fixture(worlds=4, witnesses=12, patches=True, device=device)
    count = int(contacts.rigid_contact_count.numpy()[0])
    owner = solver._friction_patches.current.owner.numpy()
    world = solver._contact_world.numpy()
    pools = {}
    for w in (1, 2):
        members = np.flatnonzero((world[:count] == w) & (owner[:count] >= 0))
        pool = members[owner[members] == owner[members].min()]
        owner[pool] = -1
        pools[w] = frozenset(pool.tolist())
    solver._friction_patches.current.owner.assign(owner)
    groups = _match_preparation(test, solver, state, augmented, contacts)
    partition = _partition(test, groups)
    for w, pool in pools.items():
        test.assertGreater(len(pool), 1)
        test.assertIn(pool, partition, f"world {w}'s owner-less witnesses are not one region of their own")


def test_scene_torsion_groups_empty_contacts(test, device):
    """Group no contact: a buffer with a zero count, and a buffer of no capacity."""
    for capacity in (6, 0):
        with test.subTest(capacity=capacity):
            contacts = _contacts(device, [], [], [], capacity=capacity)
            eligible = scene_torsion_eligibility(
                contacts,
                wp.full(2, -1, dtype=int, device=device),
                body_q=wp.zeros(1, dtype=wp.transform, device=device),
                shape_supported=wp.full(2, 1, dtype=int, device=device),
                shape_radius=wp.full(2, SHAPE_RADIUS, dtype=float, device=device),
                contact_path=wp.zeros(capacity, dtype=int, device=device),
                contact_slot=wp.zeros(capacity, dtype=int, device=device),
                contact_slots_needed=wp.full(capacity, ANCHOR, dtype=int, device=device),
            )
            np.testing.assert_array_equal(eligible.numpy(), np.zeros(capacity, np.int32))
            groups = _group(contacts, 2, [1] * capacity, [ANCHOR] * capacity, [FRICTION] * capacity)
            test.assertEqual(_clusters(test, groups), {})
            np.testing.assert_array_equal(groups.member_start.numpy(), np.zeros(capacity + 1, np.int32))


def test_scene_torsion_shapes_follow_body_variants(test, device):
    """Mark torsion support across body variants as Steel's device preparation marks it.

    A free body declared with two variants, a box (a type contact torsion supports) and a triangle mesh (one it does
    not), and a ground plane. :class:`SceneTorsionShapes` marks the current types like Steel's
    ``update_shape_types``: the box at construction; after :meth:`~newton.Model.select_body_variants` picks the mesh,
    the marks are stale until :meth:`SceneTorsionShapes.update`, as Steel's are until its notification, then the
    mesh row is unsupported. On CUDA, where Steel builds its device preparation, the marks equal Steel's after each
    ``notify_model_changed``; and the scene's eligibility of a contact with the variant body rejects it then.
    """
    variants = []
    for kind in ("box", "mesh"):
        variant = newton.ModelBuilder()
        body = variant.add_body()
        if kind == "box":
            variant.add_shape_box(body, hx=0.05, hy=0.05, hz=0.025)
        else:
            mesh = newton.Mesh.create_box(0.05, 0.05, 0.025, compute_inertia=False)
            variant.add_shape_mesh(body, mesh=mesh)
        variants.append(variant)
    builder = newton.ModelBuilder()
    SolverSteel.register_custom_attributes(builder)
    box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.025), wp.quat_identity()))
    builder.add_body_variants(box, variants)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    set_torsion_radius(model, SHAPE_RADIUS)
    shapes = SceneTorsionShapes.allocate(model)
    steel = None
    if wp.get_device(device).is_cuda:
        steel = SolverSteel(
            model,
            solve_mode="serial",
            contact_torsion_device=True,
            contact_warmstart=False,
            velocity_iterations=0,
            friction_anchor_beta=0.0,
        )

    def expected():
        return np.isin(model.shape_type.numpy(), [int(t) for t in _SUPPORTED_TYPES]).astype(np.int32)

    initial = expected()
    test.assertTrue(np.all(initial == 1))
    np.testing.assert_array_equal(shapes.supported.numpy(), initial)
    if steel is not None:
        np.testing.assert_array_equal(shapes.supported.numpy(), steel._device_torsion.supported.numpy())
    flags = model.select_body_variants(
        wp.array([1], dtype=wp.int32, device=device), bodies=wp.array([box], dtype=wp.int32, device=device)
    )
    selected = expected()
    test.assertEqual(int(selected.sum()), int(initial.sum()) - 1)
    # Stale until the update, as Steel's marks are until its notification.
    np.testing.assert_array_equal(shapes.supported.numpy(), initial)
    shapes.update(model)
    np.testing.assert_array_equal(shapes.supported.numpy(), selected)
    if steel is not None:
        steel.notify_model_changed(flags)
        np.testing.assert_array_equal(shapes.supported.numpy(), steel._device_torsion.supported.numpy())
    # A touching contact between the variant's mesh row and the ground: a rejected witness (-1) now, eligible before.
    mesh_shape = int(np.flatnonzero(selected == 0)[0])
    ground = int(np.flatnonzero(model.shape_body.numpy() == -1)[0])
    contacts = _contacts(device, [(mesh_shape, ground)], [(0, 0, 1)], [(0, 0, 0)])
    routing = {
        "contact_path": wp.zeros(1, dtype=int, device=device),
        "contact_slot": wp.zeros(1, dtype=int, device=device),
        "contact_slots_needed": wp.full(1, ANCHOR, dtype=int, device=device),
    }
    # Zero poses keep the contact points as given (the hand-built contacts' convention): touching.
    poses = wp.zeros(model.body_count, dtype=wp.transform, device=device)
    common = {"body_q": poses, "shape_radius": model.steel.min_torsional_patch_radius, **routing}
    for marks, value in ((wp.array(initial, dtype=int, device=device), 1), (shapes.supported, -1)):
        eligible = scene_torsion_eligibility(contacts, model.shape_body, shape_supported=marks, **common)
        test.assertEqual(int(eligible.numpy()[0]), value)


def _random_inputs(device, seed, count=RANDOM_CONTACTS):
    """Random contacts among 12 static shapes and their grouping inputs, with patch owners.

    Normals near three directions and points on a few planes give several clusters per shape pair. Returns
    ``(contacts, inputs)``, ``inputs`` the keyword arguments of :func:`group_scene_torsion` but ``patch_owner``,
    which is ``inputs["owner"]``.
    """
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, 12, size=(count, 2))
    directions = np.array([(0, 0, 1), (0, 0.02, 1), (1, 0, 0)], np.float32)
    normals = directions[rng.integers(0, 3, count)]
    points = np.column_stack((rng.uniform(-0.1, 0.1, (count, 2)), rng.integers(0, 3, count) * 0.004))
    gaps = np.where(rng.random(count) < 0.2, APART, 0.0)
    contacts = _contacts(device, pairs, normals, points, gaps=gaps)
    inputs = {
        "shape_body": wp.full(12, -1, dtype=int, device=device),
        "eligible": _int_array(rng.choice([1, 1, 1, 1, 1, 1, 1, -1, 0, 0], count), device),
        "body_q": wp.zeros(1, dtype=wp.transform, device=device),
        "shape_radius": _float_array(rng.choice([0.0, 0.01, 0.02], 12), device),
        "contact_slots_needed": _int_array(rng.choice([ANCHOR, ANCHOR, NORMAL_ONLY], count), device),
        "friction": _float_array(rng.choice([0.0, 0.3, 0.5, 0.5], count), device),
        "owner": _int_array(rng.integers(-1, 40, count), device),
    }
    return contacts, inputs


def _call(contacts, inputs, patches, out=None):
    """:func:`group_scene_torsion` on :func:`_random_inputs`' contacts, with or without their patch owners."""
    arguments = {name: value for name, value in inputs.items() if name not in ("shape_body", "eligible", "owner")}
    owner = inputs["owner"] if patches else None
    return group_scene_torsion(
        contacts, inputs["shape_body"], inputs["eligible"], patch_owner=owner, out=out, **arguments
    )


def _snapshot(groups):
    return {name: getattr(groups, name).numpy().copy() for name in OUTPUTS}


def _assert_same(test, actual, expected, label):
    for name in OUTPUTS:
        np.testing.assert_array_equal(actual[name], expected[name], f"{label}: {name}")


def test_scene_torsion_groups_are_deterministic(test, device):
    """Group the same contacts into the same arrays, bit for bit: fresh twice and into reused groups.

    The reused call reads nothing back to the host and assigns nothing from it.
    """
    for patches in (False, True):
        with test.subTest(patches=patches):
            contacts, inputs = _random_inputs(device, seed=7)
            first = _call(contacts, inputs, patches)
            expected = _snapshot(first)
            test.assertGreater(int(expected["group_count"][0]), 3)
            _clusters(test, first)
            _assert_same(test, _snapshot(_call(contacts, inputs, patches)), expected, "fresh")
            with (
                patch.object(wp.array, "numpy", side_effect=AssertionError("Unexpected device-to-host transfer")),
                patch.object(wp.array, "assign", side_effect=AssertionError("Unexpected host assignment")),
            ):
                _call(contacts, inputs, patches, out=first)
            _assert_same(test, _snapshot(first), expected, "reused")


def test_scene_torsion_groups_capture(test, device):
    """Capture the grouping into allocated groups and replay it: the eager result, also after the inputs change.

    The graph replays the eager groups of its inputs bit for bit; with half the contacts and a new eligibility it
    replays the eager groups of those.
    """
    for patches in (False, True):
        with test.subTest(patches=patches):
            contacts, inputs = _random_inputs(device, seed=11)
            groups = SceneTorsionGroups.allocate(contacts.rigid_contact_max, device)
            _call(contacts, inputs, patches, out=groups)
            expected = _snapshot(groups)
            with wp.ScopedCapture(device) as capture:
                _call(contacts, inputs, patches, out=groups)
            for name in OUTPUTS:
                getattr(groups, name).fill_(7)
            wp.capture_launch(capture.graph)
            _assert_same(test, _snapshot(groups), expected, "replay")
            contacts.rigid_contact_count.fill_(RANDOM_CONTACTS // 2)
            eligible = inputs["eligible"].numpy()
            inputs["eligible"].assign(np.roll(eligible, 3))
            wp.capture_launch(capture.graph)
            changed = _snapshot(_call(contacts, inputs, patches))
            test.assertFalse(np.array_equal(changed["contact_group"], expected["contact_group"]))
            _assert_same(test, _snapshot(groups), changed, "changed")


class TestSceneTorsionGroups(unittest.TestCase):
    pass


# The grouping kernels run on any device, as Steel's preparation kernels do.
for _fn in (
    test_scene_torsion_groups_match_device_preparation,
    test_scene_torsion_groups_segment_sizes,
    test_scene_torsion_groups_cluster_pairs_and_normals,
    test_scene_torsion_groups_drop_clusters_without_a_row,
    test_scene_torsion_groups_with_patches,
    test_scene_torsion_groups_empty_contacts,
    test_scene_torsion_shapes_follow_body_variants,
    test_scene_torsion_groups_are_deterministic,
):
    add_function_test(TestSceneTorsionGroups, _fn.__name__, _fn, devices=get_test_devices())

# CUDA only, each for the reason beside it.
for _fn in (
    test_scene_torsion_groups_match_steel_grouping,  # contact torsion, which Steel's own check refuses on the CPU
    test_scene_torsion_groups_capture,  # graph capture
):
    add_function_test(TestSceneTorsionGroups, _fn.__name__, _fn, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
