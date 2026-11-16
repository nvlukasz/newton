# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scene groups of SolverSteel (``solve_mode="scene"``).

The topology key and the plan of factor and free groups, the group-table contact response kernel
against the specialised one, and the per-group factor storage and launches: a scene split into
several groups of one topology (the test-only ``scene_group_split``) against its one group, and
the armature that a notification brings to every group. Then scenes of several topologies
(:func:`_scene_worlds`): each world against the same world alone, the scene against Steel, rows
wider than 32 entries against the serial oracle, and a world ``-1`` box that couples two
topologies. Then topologies with a diagonal mass matrix: diagonal factor plans, and their factor
groups against Steel's diagonal-mass kernels. Then six-DOF chains beside free boxes and on them: a
factor group of the free bodies' size against Steel's dense solve. Last, topologies of more than 64
DOFs: the native helpers of multi-word support masks against Python ints, a 65-link chain and a
70-link tree, whose masks take two words, against Steel's dense solve, and scenes whose masks fit
one word stepped with two-word masks against their one-word selves, bit for bit.
"""

import hashlib
import re
import unittest
from functools import cache

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel._testing import kernel_overrides
from newton._src.solvers.steel.kernels import PGS_CONSTRAINT_TYPE_JOINT_LIMIT
from newton._src.solvers.steel.scene_rows import (
    FAMILY_LIMIT,
    get_sparse_joint_limit_rows_kernel,
    mask_helpers_snippet,
    scene_contact_response_dof_limit,
    scene_contact_response_shared_bytes,
    sparse_joint_limit_rows_snippet,
)
from newton._src.solvers.steel.scene_solve import (
    SceneGroup,
    ScenePlan,
    factor_body_masks,
    plan_scene_groups,
    scene_response_table,
    scene_support,
    scene_topology_keys,
    topology_key,
)
from newton._src.solvers.steel.sparse_contact import build_sparse_joint_limit_rows
from newton._src.solvers.steel.sparse_mass_matrix import _SparseMassMatrixPlan, distinct_mask_values, mask_value
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import CONTACT_MAX, _heap, _tree
from newton.tests.steel._scene_fixtures import (
    LIMIT_ROWS_SNIPPET_SHA256,
    SCENE,
    _chain_worlds,
    _humanoid,
    _humanoids,
    _make_chain_world,
    _rms,
    _Run,
    _scene_builder,
    _scene_worlds,
    _without_passive_springs,
    _world_parts,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _factor_plan(key) -> _SparseMassMatrixPlan:
    return _SparseMassMatrixPlan.build(*map(np.asarray, key))


def test_topology_key_is_the_factor_plan_input(test, device):
    """Key each articulation by its local joint parents and DOFs per joint, the input of its sparse factor plan."""
    model = _humanoids(device, worlds=3)
    keys = scene_topology_keys(SolverSteel(model, solve_mode="serial"), model)
    test.assertEqual(len(keys), 3)
    test.assertEqual(keys[1], keys[0])
    test.assertEqual(keys[2], keys[0])
    test.assertEqual(_factor_plan(keys[0]).dof_count, 27)
    # Plain integers, so a key is hashable and equal across devices.
    test.assertTrue(all(type(value) is int for part in keys[0] for value in part))

    chains = _chain_worlds([2, 3], device)
    keys = scene_topology_keys(SolverSteel(chains, solve_mode="serial"), chains)
    test.assertEqual(len(keys), 2)
    test.assertNotEqual(keys[0], keys[1])
    test.assertEqual(keys[0], ((-1, 0), (1, 1)))
    test.assertEqual(keys[1], ((-1, 0, 1), (1, 1, 1)))
    test.assertEqual(keys[0][1], (1, 1))
    test.assertEqual(keys[1][1], (1, 1, 1))


def test_plan_scene_groups_orders_factor_groups_then_free(test, device):
    """Number factor groups by their first articulation, the free group last, each owning its block of ``Linv``."""
    k1 = topology_key(np.array([-1, 0], dtype=np.int32), np.array([1, 1], dtype=np.int32))
    k2 = topology_key(np.array([-1, 0, 1], dtype=np.int32), np.array([1, 1, 1], dtype=np.int32))
    test.assertEqual(k1, ((-1, 0), (1, 1)))
    test.assertEqual(k2, ((-1, 0, 1), (1, 1, 1)))
    nnz1, nnz2 = _factor_plan(k1).nonzero_count, _factor_plan(k2).nonzero_count

    groups = plan_scene_groups([k2, k1, None, k2, k1], [False, False, True, False, False])
    test.assertTrue(all(isinstance(group, SceneGroup) for group in groups))
    test.assertEqual(tuple(group.kind for group in groups), ("factor", "factor", "free"))
    test.assertEqual(tuple(group.arts for group in groups), ((0, 3), (1, 4), (2,)))
    test.assertEqual(groups[0].linv_offset, 0)
    test.assertEqual(groups[1].linv_offset, 2 * nnz2)
    test.assertEqual(groups[2].linv_offset, 2 * nnz2 + 2 * nnz1)

    test.assertEqual(tuple(group.key for group in groups), (k2, k1, None))
    test.assertEqual(tuple(group.size for group in groups), (3, 2, 6))
    test.assertEqual(tuple(group.nonzero_count for group in groups), (nnz2, nnz1, 0))
    test.assertEqual(tuple(group.mask_words for group in groups), (1, 1, 1))
    test.assertEqual(groups[0].plan.nonzero_count, nnz2)
    test.assertEqual(groups[1].plan.dof_count, 2)
    test.assertIsNone(groups[2].plan)

    # No free articulation, no free group; no articulation, no group.
    test.assertEqual(tuple(group.kind for group in plan_scene_groups([k1, None], [False, False])), ("factor",))
    test.assertEqual(plan_scene_groups([], []), ())

    # The plan's articulated size is its single factor group's; with two factor groups there is none.
    plan = ScenePlan(support=6, row_max=8, coordinate_count=16, free_body_count=1, groups=groups[1:], mask_words=1)
    test.assertEqual(plan.articulated_size, 2)
    plan = ScenePlan(support=6, row_max=8, coordinate_count=16, free_body_count=1, groups=groups, mask_words=1)
    with test.assertRaises(ValueError):
        _ = plan.articulated_size


def test_plan_scene_groups_splits_by_articulation(test, device):
    """Split each topology's articulations by index mod ``split`` into groups that share its plan.

    The solver's test-only kernel override ``scene_group_split`` passes ``split``.
    """
    k1 = topology_key(np.array([-1, 0], dtype=np.int32), np.array([1, 1], dtype=np.int32))
    k2 = topology_key(np.array([-1, 0, 1], dtype=np.int32), np.array([1, 1, 1], dtype=np.int32))
    nnz1, nnz2 = _factor_plan(k1).nonzero_count, _factor_plan(k2).nonzero_count
    keys = [k2, k1, None, k2, k1, k2]
    free = [False, False, True, False, False, False]

    # (k2, even), (k1, odd), (k2, odd), (k1, even), in order of their first articulation; the free group last.
    groups = plan_scene_groups(keys, free, split=2)
    test.assertEqual(tuple(group.kind for group in groups), ("factor", "factor", "factor", "factor", "free"))
    test.assertEqual(tuple(group.arts for group in groups), ((0,), (1,), (3, 5), (4,), (2,)))
    test.assertEqual(tuple(group.key for group in groups), (k2, k1, k2, k1, None))
    test.assertEqual(
        tuple(group.linv_offset for group in groups),
        (0, nnz2, nnz2 + nnz1, 3 * nnz2 + nnz1, 3 * nnz2 + 2 * nnz1),
    )
    # The groups of one key share its plan.
    test.assertIs(groups[2].plan, groups[0].plan)
    test.assertIs(groups[3].plan, groups[1].plan)

    # A split past the articulations of a key leaves no empty group; a split of 1 is the plain plan.
    test.assertEqual(tuple(group.arts for group in plan_scene_groups([k1, k1], [False, False], split=3)), ((0,), (1,)))
    plain = plan_scene_groups(keys, free)
    test.assertEqual(
        tuple((group.arts, group.linv_offset) for group in plan_scene_groups(keys, free, split=1)),
        tuple((group.arts, group.linv_offset) for group in plain),
    )
    with test.assertRaisesRegex(ValueError, "split"):
        plan_scene_groups(keys, free, split=0)


def test_plan_scene_groups_sizes_masks_by_dofs(test, device):
    """Give each factor group's plan support masks of ``ceil(DOFs / 64)`` words, or at least ``mask_words``.

    A 65-link chain's plan has two-word masks, ``[coordinates, 2]`` with word 0 holding coordinates
    0 to 63: its tip's mask covers all 65, its root's one. The device indices hold a coordinate's
    words consecutively. Steel's one-word plan refuses the chain, and ``mask_words`` (the test-only
    ``scene_mask_words``) widens a two-link chain's masks without changing their coordinates.
    """
    k2 = topology_key(np.array([-1, 0], dtype=np.int32), np.array([1, 1], dtype=np.int32))
    k65 = topology_key(np.arange(-1, 64, dtype=np.int32), np.ones(65, dtype=np.int32))
    groups = plan_scene_groups([k2, k65, None], [False, False, True])
    test.assertEqual(tuple((group.size, group.mask_words) for group in groups), ((2, 1), (65, 2), (6, 1)))
    chain = groups[1].plan
    test.assertEqual(chain.mask_words, 2)
    test.assertEqual(chain.ancestor_mask.shape, (65, 2))
    test.assertEqual(chain.joint_ancestor_mask.shape, (65, 2))
    # Joint j's support is the chain's first j + 1 joints: the tip all 65 coordinates, the root one.
    test.assertEqual(mask_value(chain.joint_ancestor_mask[64]), (1 << 65) - 1)
    test.assertEqual(mask_value(chain.joint_ancestor_mask[0]).bit_count(), 1)
    test.assertEqual([chain.endpoint_support(joint).size for joint in range(65)], list(range(1, 66)))
    indices = chain.to_device(device)
    test.assertTrue(np.array_equal(indices.ancestor_mask.numpy(), chain.ancestor_mask.reshape(-1)))
    with test.assertRaisesRegex(ValueError, "1 to 64 DOFs"):
        _SparseMassMatrixPlan.build(*map(np.asarray, k65))
    # One word stays the one-dimensional masks of Steel's plans; a wider plan keeps their coordinates.
    test.assertEqual(groups[0].plan.ancestor_mask.shape, (2,))
    wide = plan_scene_groups([k2, k65], [False, False], mask_words=3)
    test.assertEqual(tuple(group.mask_words for group in wide), (3, 3))
    test.assertEqual(wide[0].plan.ancestor_mask.shape, (2, 3))
    test.assertEqual(
        [mask_value(mask) for mask in wide[0].plan.ancestor_mask],
        [int(mask) for mask in groups[0].plan.ancestor_mask],
    )
    with test.assertRaisesRegex(ValueError, "at least one word"):
        plan_scene_groups([k2], [False], mask_words=0)


def test_response_table_requires_a_group_per_articulation(test, device):
    """Refuse a group table that leaves a responding non-free articulation without a factor group."""
    k1 = topology_key(np.array([-1, 0], dtype=np.int32), np.array([1, 1], dtype=np.int32))
    nnz1 = _factor_plan(k1).nonzero_count
    groups = plan_scene_groups([k1, None, k1, None], [False, False, False, True])
    linv = wp.zeros(2 * nnz1, dtype=wp.float32, device=device)

    articulated = np.array([True, False, True, False])
    art_group, art_slot, table = scene_response_table(groups, 4, linv, device, articulated=articulated)
    test.assertEqual(art_group.numpy().tolist(), [0, -1, 0, -1])
    test.assertEqual(art_slot.numpy().tolist(), [0, -1, 1, -1])
    test.assertIs(table[-1], linv)

    # Articulation 1 responds without a factor group: its endpoints would reach the factor branch with group -1.
    with test.assertRaisesRegex(ValueError, r"articulations \[1\]"):
        scene_response_table(groups, 4, linv, device, articulated=np.array([True, True, True, False]))


def test_one_topology_scene_has_one_factor_group(test, device):
    """Plan one factor group, today's sparse factor, for a scene of one topology, and a free group for its props."""
    solver = SolverSteel(_humanoids(device, worlds=4), articulation_constraint_max=128, **SCENE)
    plan = solver._scene_plan
    test.assertEqual(len(plan.groups), 1)
    test.assertEqual(plan.groups[0].kind, "factor")
    test.assertEqual(plan.groups[0].arts, (0, 1, 2, 3))
    test.assertEqual(plan.articulated_size, 27)
    test.assertEqual(plan.mask_words, 1)
    test.assertIs(plan.groups[0].plan, solver._sparse_mass_matrix_plan)
    test.assertEqual(plan.groups[0].linv_offset, 0)

    solver = SolverSteel(_humanoids(device, worlds=4, props=True), articulation_constraint_max=128, **SCENE)
    plan = solver._scene_plan
    test.assertEqual(tuple(group.kind for group in plan.groups), ("factor", "free"))
    test.assertEqual(plan.articulated_size, 27)
    test.assertEqual(plan.mask_words, 1)
    factor, free = plan.groups
    test.assertEqual(len(factor.arts), 4)
    test.assertEqual(len(free.arts), 4)
    test.assertEqual(free.linv_offset, 4 * factor.nonzero_count)
    test.assertEqual(len(free.arts), plan.free_body_count)
    # The free bodies are Steel's whole size-6 group, so the decode's free rows are that very array.
    test.assertIs(solver._scene.free_group_to_art, solver._group_to_art[6])


def test_free_only_scene_has_one_free_group(test, device):
    """Plan a scene of free bodies only as one free group, without an articulated size."""
    solver = SolverSteel(_heap(device), **SCENE)
    plan = solver._scene_plan
    test.assertEqual([group.kind for group in plan.groups], ["free"])
    test.assertIsNone(plan.articulated_size)
    test.assertEqual(len(plan.groups[0].arts), plan.free_body_count)
    test.assertEqual(plan.mask_words, 1)
    test.assertIs(solver._scene.free_group_to_art, solver._group_to_art[6])


def test_generic_response_kernel_is_bitwise_specialized(test, device):
    """Build the rows through the group table (the default) bit for bit as the specialised kernel.

    On a scene of one factor group the generic kernel's arithmetic, and its order, is the
    specialised kernel's (``scene_response_kernel="specialized"``) for every endpoint, so the rows
    after the first step and the states after each of 30 steps are bitwise equal: on a crowd of
    four humanoids per world 0.5 m apart, on humanoids beside free boxes (a factor group and the
    free group), on the boxes moved to touch the humanoids (contacts with a free and a factor
    endpoint), on humanoids 0.4 m apart, whose 4 cm overlaps give contacts with two factor
    endpoints, and on a heap of free bodies only (an empty table: every endpoint takes the free
    branch).
    """
    scenes = {
        "crowd": (_humanoids(device, worlds=2, per_world=4, spacing=0.5), 4 * 128, 27),
        "props": (_humanoids(device, worlds=2, props=True), 128, 27),
        # Each box 0.05 m behind its humanoid's back at x = 0 (spacing 0.4 puts it at x = -0.2).
        "touch": (_humanoids(device, worlds=2, props=True, spacing=0.4), 128, 27),
        "pairs": (_humanoids(device, worlds=2, per_world=4, spacing=0.4), 4 * 128, 27),
        "heap": (_heap(device), 3 * CONTACT_MAX, 6),
    }
    rows_fields = (
        "_sparse_row_dof",
        "_sparse_row_factor",
        "_sparse_row_free_response",
        "_sparse_row_incident",
        "_diag",
    )
    for name, (model, rows, dof_max) in scenes.items():
        with test.subTest(scene=name):
            specialized = _Run(model, rows, {"scene_response_kernel": "specialized"})
            generic = _Run(model, rows, {})
            test.assertEqual(specialized.solver._scene.response_kind, "specialized")
            test.assertEqual(generic.solver._scene.response_kind, "generic")
            kernel = generic.solver._scene.response_kernel
            test.assertTrue(kernel.key.startswith(f"populate_scene_contact_response_{dof_max}_"))
            for step in range(30):
                specialized.step()
                generic.step()
                if step == 0:
                    row_count = int(specialized.solver._constraint_count.numpy()[0])
                    test.assertGreater(row_count, 0)
                    test.assertEqual(int(generic.solver._constraint_count.numpy()[0]), row_count)
                    test.assertFalse(np.any(specialized.solver.constraint_overflow.numpy()))
                    test.assertFalse(np.any(generic.solver.constraint_overflow.numpy()))
                    for field in rows_fields:
                        expected = getattr(specialized.solver, field).numpy()[0, :row_count]
                        actual = getattr(generic.solver, field).numpy()[0, :row_count]
                        test.assertTrue(np.array_equal(expected, actual), msg=f"{field} after step 1")
                    if name == "props":
                        # The boxes' ground contacts take the free branch.
                        test.assertTrue(np.any(specialized.solver._scene.row_free_prefix.numpy()[:row_count] > 0))
                    if name in ("touch", "pairs"):
                        # Contacts with rows (the factor-coordinate path 0) between two articulations.
                        solver = specialized.solver
                        count = int(specialized.contacts.rigid_contact_count.numpy()[0])
                        art_a = solver._contact_art_a.numpy()[:count]
                        art_b = solver._contact_art_b.numpy()[:count]
                        kept = (solver._contact_slot.numpy()[:count] >= 0) & (solver._contact_path.numpy()[:count] == 0)
                        between = kept & (art_a >= 0) & (art_b >= 0) & (art_a != art_b)
                        if name == "touch":
                            # Some joins a box and a humanoid: one endpoint takes the free branch, the other the
                            # factor branch.
                            is_free = solver._is_free_rigid.numpy()
                            mixed = is_free[np.maximum(art_a, 0)] != is_free[np.maximum(art_b, 0)]
                            test.assertTrue(np.any(between & mixed))
                        else:
                            # Some joins two humanoids: both endpoints take the factor branch.
                            test.assertTrue(np.any(between))
                for field in ("joint_q", "joint_qd"):
                    expected = getattr(specialized.state_0, field).numpy()
                    actual = getattr(generic.state_0, field).numpy()
                    test.assertTrue(np.array_equal(expected, actual), msg=f"{field} after step {step + 1}")


def _contacts_between_groups(run) -> int:
    """Kept contacts of ``run``'s last step (rows on the factor-coordinate path 0) between two factor groups'
    articulations."""
    solver = run.solver
    count = int(run.contacts.rigid_contact_count.numpy()[0])
    art_a = solver._contact_art_a.numpy()[:count]
    art_b = solver._contact_art_b.numpy()[:count]
    kept = (solver._contact_slot.numpy()[:count] >= 0) & (solver._contact_path.numpy()[:count] == 0)
    art_group = solver._scene.art_group.numpy()
    group_a = np.where(art_a >= 0, art_group[np.maximum(art_a, 0)], -1)
    group_b = np.where(art_b >= 0, art_group[np.maximum(art_b, 0)], -1)
    return int(np.count_nonzero(kept & (group_a >= 0) & (group_b >= 0) & (group_a != group_b)))


def test_split_groups_are_bitwise_one_group(test, device):
    """Step a scene split into k factor groups of one topology bit for bit as its one group.

    The test-only override ``scene_group_split=k`` splits the factor group by articulation index
    mod k, so each of the k groups of the one plan has its own factor storage and its own launches
    (factor, unconstrained solve, armature, limit rows, decode), and the rows read every endpoint
    through the group table (``scene_response_kernel="generic"``). For k = 1, 2 and 3 the row
    count, the impulses and the states after each of 30 steps are bitwise those of one group
    with the specialised response kernel, on a crowd of eight humanoids in one
    world 0.5 m apart, on 64 worlds of one humanoid, and on two worlds of four humanoids 0.4 m
    apart, whose 4 cm overlaps give contacts between articulations of two groups. Every scene
    builds joint-limit rows. An explicit ``"specialized"`` kernel on several groups and a split
    below 1 raise.
    """
    scenes = {
        "crowd": (_humanoids(device, worlds=1, per_world=8, spacing=0.5), 8 * 128),
        "worlds": (_humanoids(device, worlds=64), 128),
        "pairs": (_humanoids(device, worlds=2, per_world=4, spacing=0.4), 4 * 128),
    }
    for name, (model, rows) in scenes.items():
        with test.subTest(scene=name):
            reference = _Run(model, rows, {"scene_response_kernel": "specialized"})
            test.assertEqual(reference.solver._scene.response_kind, "specialized")
            test.assertEqual(len(reference.solver._scene.group_arrays), 1)
            runs = {
                split: _Run(model, rows, {"scene_group_split": split, "scene_response_kernel": "generic"})
                for split in (1, 2, 3)
            }
            for split, run in runs.items():
                test.assertEqual(run.solver._scene.response_kind, "generic")
                group_arrays = run.solver._scene.group_arrays
                test.assertEqual(len(group_arrays), split)
                for residue, arrays in enumerate(group_arrays):
                    test.assertEqual(arrays.group.arts, tuple(range(residue, model.articulation_count, split)))
                # Steel's single-plan factor views the one group's; with several groups there is none.
                if split == 1:
                    test.assertIs(run.solver._sparse_Linv, group_arrays[0].Linv)
                    test.assertIs(run.solver._sparse_mass_matrix_status, group_arrays[0].status)
                else:
                    test.assertIsNone(run.solver._sparse_Linv)
                    test.assertIsNone(run.solver._sparse_mass_matrix_status)
            rows_seen = limit_rows = 0
            between_groups = dict.fromkeys(runs, 0)
            for step in range(30):
                reference.step()
                count = int(reference.solver._constraint_count.numpy()[0])
                rows_seen = max(rows_seen, count)
                row_type = reference.solver._row_type.numpy()[0, :count]
                limit_rows += int(np.count_nonzero(row_type == PGS_CONSTRAINT_TYPE_JOINT_LIMIT))
                impulses = reference.solver._impulses.numpy()[0, :count]
                for split, run in runs.items():
                    run.step()
                    between_groups[split] += _contacts_between_groups(run)
                    where = f"k = {split} after step {step + 1}"
                    test.assertEqual(int(run.solver._constraint_count.numpy()[0]), count, msg=where)
                    test.assertTrue(np.array_equal(run.solver._impulses.numpy()[0, :count], impulses), msg=where)
                    for field in ("joint_q", "joint_qd"):
                        expected = getattr(reference.state_0, field).numpy()
                        actual = getattr(run.state_0, field).numpy()
                        test.assertTrue(np.array_equal(expected, actual), msg=f"{field}, {where}")
            test.assertGreater(rows_seen, 0)
            # The limits COUNT and WRITE passes ran per group on rows the reference has too.
            test.assertGreater(limit_rows, 0)
            test.assertEqual(between_groups[1], 0)
            if name == "pairs":
                # Contacts whose two endpoints read their factors through two groups' entries of the table.
                test.assertGreater(between_groups[2], 0)
                test.assertGreater(between_groups[3], 0)
            for run in (reference, *runs.values()):
                test.assertFalse(np.any(run.solver.constraint_overflow.numpy()))

    model, rows = scenes["worlds"]
    with kernel_overrides({"scene_group_split": 2, "scene_response_kernel": "specialized"}):
        with test.assertRaisesRegex(ValueError, "use 'generic'"):
            SolverSteel(model, articulation_constraint_max=rows, **SCENE)
    with kernel_overrides({"scene_group_split": 0}):
        with test.assertRaisesRegex(ValueError, "scene_group_split must be at least 1, got 0"):
            SolverSteel(model, articulation_constraint_max=rows, **SCENE)


def test_notify_armature_reaches_every_group(test, device):
    """Bring a notified armature to every scene group's factor, a group the ``world_mask`` leaves out included.

    Under ``scene_group_split=2`` the humanoids of four worlds form group 0 (worlds 0 and 2) and
    group 1 (worlds 1 and 3). The armature of every DOF of worlds 1 and 3 is set to 0.05 and a
    ``JOINT_DOF_PROPERTIES`` notification masked to those worlds, group 1's members, follows. After
    one step group 1's armature is a fresh solver's at 0.05 and group 0's is as it was, and 10
    steps are bitwise those of a fresh solver built with these per-world armatures. The mask
    selects warm-start histories only: like Steel's, the notification re-reads the armature of
    every world, so setting 0.05 everywhere under the same mask also reaches group 0, whose worlds
    the mask leaves out, and Steel's size-group copy.
    """
    split = {"scene_group_split": 2}
    model = _humanoids(device, worlds=4)
    run = _Run(model, 128, split)
    group_arrays = run.solver._scene.group_arrays
    test.assertEqual([arrays.group.arts for arrays in group_arrays], [(0, 2), (1, 3)])
    # Several factor groups default to the group-table response kernel.
    test.assertEqual(run.solver._scene.response_kind, "generic")
    # Copies: on the CPU numpy() is a view of the armature, which the notification rewrites.
    before = [arrays.armature.numpy().copy() for arrays in group_arrays]
    stale = _Run(_humanoids(device, worlds=4), 128, split)

    dof_world = np.repeat(model.joint_world.numpy(), np.diff(model.joint_qd_start.numpy()))
    armature = model.joint_armature.numpy()
    test.assertFalse(np.any(armature == 0.05))
    armature[np.isin(dof_world, (1, 3))] = 0.05
    model.joint_armature.assign(armature)
    world_mask = wp.array([False, True, False, True, False], dtype=wp.bool, device=device)
    run.solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES, world_mask=world_mask)
    run.step()

    uniform = _humanoids(device, worlds=4)
    uniform.joint_armature.fill_(0.05)
    with kernel_overrides(split):
        fresh = SolverSteel(uniform, articulation_constraint_max=128, **SCENE)
    after = [arrays.armature.numpy() for arrays in group_arrays]
    test.assertTrue(np.array_equal(after[1], fresh._scene.group_arrays[1].armature.numpy()))
    test.assertFalse(np.array_equal(after[1], before[1]))
    test.assertTrue(np.array_equal(after[0], before[0]))

    # The model now holds the per-world armatures, so a solver built from it starts with them.
    reference = _Run(model, 128, split)
    reference.step()
    stale.step()
    for _ in range(9):
        run.step()
        reference.step()
        stale.step()
    for field in ("joint_q", "joint_qd"):
        expected = getattr(reference.state_0, field).numpy()
        test.assertTrue(np.array_equal(getattr(run.state_0, field).numpy(), expected), msg=field)
    # Without the notification the armature of worlds 1 and 3 stays the old one, and so do their states.
    test.assertFalse(np.array_equal(stale.state_0.joint_q.numpy(), run.state_0.joint_q.numpy()))
    for solver in (run.solver, reference.solver, stale.solver):
        test.assertFalse(np.any(solver.constraint_overflow.numpy()))

    model.joint_armature.fill_(0.05)
    run.solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES, world_mask=world_mask)
    for group, arrays in enumerate(group_arrays):
        expected = fresh._scene.group_arrays[group].armature.numpy()
        test.assertTrue(np.array_equal(arrays.armature.numpy(), expected), msg=f"group {group}")
    np.testing.assert_array_equal(run.solver._R_by_size[27].numpy(), fresh._R_by_size[27].numpy())


def _factor_groups(solver) -> list:
    return [group for group in solver._scene_plan.groups if group.kind == "factor"]


def test_heterogeneous_worlds_match_isolated_worlds_bitwise(test, device):
    """Step each world of a scene of two topologies bit for bit as the same world alone in the scene solve.

    A two-link and a three-link chain are two factor groups, each with its own factor and its own
    launches, and the rows read every endpoint through the group table. Each world's joint
    coordinates and free bodies after each of 30 steps are bitwise those of the world alone:
    with the chains hanging 1 m up (no rows), with a free box beside the three-link chain (rows
    on its free branch), and with both chains lying on the ground (rows on the factor branch of
    two groups). The scene's support is wider than a world's alone, by -1/0 padding that adds
    exact zeros; every support stays within 32 entries, where Steel's summation order applies
    on both sides. The specialised response kernel, which reads one factor plan, is refused.
    """
    # Each scene, and whether it builds rows.
    scenes = {("chain:2", "chain:3"): False, ("chain:2", "chain:3+box"): True, ("chain:2@0.03", "chain:3@0.03"): True}
    rows = 128
    for kinds, has_rows in scenes.items():
        with test.subTest(kinds=kinds):
            model = _scene_worlds(device, kinds)
            scene = _Run(model, rows, {})
            alone = [_Run(_scene_worlds(device, [kind]), rows, {}) for kind in kinds]
            test.assertEqual([len(group.arts) for group in _factor_groups(scene.solver)], [1, 1])
            test.assertEqual(scene.solver._scene.response_kind, "generic")
            test.assertLessEqual(scene.solver._scene_plan.support, 32)
            for run in alone:
                test.assertLessEqual(run.solver._scene_plan.support, 32)
            parts = [(_world_parts(model, world), _world_parts(run.solver.model, 0)) for world, run in enumerate(alone)]
            for kind, ((coordinates, bodies), (alone_coordinates, alone_bodies)) in zip(kinds, parts, strict=True):
                test.assertEqual(len(coordinates), len(alone_coordinates))
                test.assertEqual(len(bodies), len(alone_bodies))
                # The world's free box is compared through body_q.
                test.assertEqual(len(bodies), 1 if kind.endswith("+box") else 0)
            rows_seen = 0
            for step in range(30):
                scene.step()
                rows_seen = max(rows_seen, int(scene.solver._constraint_count.numpy()[0]))
                for world, run in enumerate(alone):
                    run.step()
                    (coordinates, bodies), (alone_coordinates, alone_bodies) = parts[world]
                    where = f"world {world} after step {step + 1}"
                    expected = run.state_0.joint_q.numpy()[alone_coordinates]
                    test.assertTrue(np.array_equal(scene.state_0.joint_q.numpy()[coordinates], expected), msg=where)
                    expected = run.state_0.body_q.numpy()[alone_bodies]
                    test.assertTrue(np.array_equal(scene.state_0.body_q.numpy()[bodies], expected), msg=where)
            if has_rows:
                test.assertGreater(rows_seen, 0)
            if kinds[1] == "chain:3@0.03":
                # The lying chains' rows are in factor coordinates, wider in the scene than alone.
                test.assertGreater(scene.solver._scene_plan.support, alone[1].solver._scene_plan.support)
            for run in (scene, *alone):
                test.assertFalse(np.any(run.solver.constraint_overflow.numpy()))
            with kernel_overrides({"scene_response_kernel": "specialized"}):
                with test.assertRaisesRegex(ValueError, "use 'generic'"):
                    SolverSteel(model, articulation_constraint_max=rows, **SCENE)


def test_heterogeneous_scene_tracks_steel(test, device):
    """Follow Steel's dense solve on two chain topologies, one factor group each.

    Steel factors a two-link and a three-link chain densely, the scene solve in two sparse factor
    groups: the velocities after one step agree to rounding, and the joint coordinates over 20
    steps stay within 1 mrad RMS. With the chains hanging 1 m up only the factors and the
    unconstrained step meet Steel's; with the chains lying on the ground the contact rows of both
    topologies do too.
    """
    scenes = {
        "hanging": _chain_worlds([2, 3], device),
        "lying": _scene_worlds(device, ["chain:2@0.03", "chain:3@0.03"]),
    }
    for name, model in scenes.items():
        with test.subTest(chains=name):
            scene = _Run(model, 64, {})
            steel = _Run(model, 64, {}, solve_mode="serial")
            test.assertEqual([group.size for group in _factor_groups(scene.solver)], [2, 3])
            test.assertEqual(scene.solver._scene.response_kind, "generic")
            test.assertIsNone(steel.solver._scene)
            test.assertIsNone(steel.solver._sparse_mass_matrix_size)
            rows_seen = 0
            for step in range(20):
                scene.step()
                steel.step()
                rows_seen = max(rows_seen, int(scene.solver._constraint_count.numpy()[0]))
                if step == 0:
                    qd_scene, qd_steel = scene.state_0.joint_qd.numpy(), steel.state_0.joint_qd.numpy()
                    test.assertGreater(np.max(np.abs(qd_steel)), 0.0)
                    np.testing.assert_allclose(qd_scene, qd_steel, rtol=1e-5, atol=1e-6)
            q_scene, q_steel = scene.state_0.joint_q.numpy(), steel.state_0.joint_q.numpy()
            test.assertTrue(np.all(np.isfinite(q_scene)))
            test.assertLess(_rms(q_scene, q_steel), 1e-3)
            if name == "lying":
                test.assertGreater(rows_seen, 0)
            for run in (scene, steel):
                test.assertFalse(np.any(run.solver.constraint_overflow.numpy()))


def test_wide_rows_match_the_thread_oracle_bitwise(test, device):
    """Sweep rows wider than 32 entries (serial sums) as the serial oracle: bitwise with one thread per side and unit.

    A 40-link chain beside two humanoids widens the scene's support past 32 (a contact on the
    chain's last link alone has 40 entries), so the rows sum in entry order. With one thread per
    side and per unit (``scene_side_sweep`` and ``scene_unit_sweep`` ``"thread"``) the phases are
    the one-thread walk's arithmetic, and 30 steps are bitwise the serial oracle
    (``scene_sweep="serial"``). The default eight lanes sum in another order: their velocities
    after one step agree with the oracle to rounding, and the joint coordinates over 30 steps
    stay within 1 mrad RMS. With the chain hanging 1 m up only the humanoids touch the ground;
    with the chain lying on the ground its rows reach more than 32 entries.
    """
    thread = {"scene_side_sweep": "thread", "scene_unit_sweep": "thread"}
    rows = 512
    for chain in ("chain:40", "chain:40@0.03"):
        with test.subTest(chain=chain):
            model = _scene_worlds(device, ["humanoid", chain, "humanoid"])
            runs = {
                "thread": _Run(model, rows, thread),
                "lanes8": _Run(model, rows, {}),
                "oracle": _Run(model, rows, {**thread, "scene_sweep": "serial"}),
            }
            plan = runs["lanes8"].solver._scene_plan
            test.assertGreater(plan.support, 32)
            test.assertEqual([group.size for group in _factor_groups(runs["lanes8"].solver)], [27, 40])
            for run in runs.values():
                test.assertEqual(run.solver._scene.sum_order, "serial")
                test.assertEqual(run.solver._scene.response_kind, "generic")
            widest = 0
            for step in range(30):
                for run in runs.values():
                    run.step()
                solver = runs["lanes8"].solver
                count = int(solver._constraint_count.numpy()[0])
                entries = np.count_nonzero(solver._sparse_row_dof.numpy()[0, :count] >= 0, axis=1)
                widest = max(widest, int(entries.max(initial=0)))
                where = f"after step {step + 1}"
                expected = runs["oracle"].state_0.joint_q.numpy()
                test.assertTrue(np.array_equal(runs["thread"].state_0.joint_q.numpy(), expected), msg=where)
                if step == 0:
                    np.testing.assert_allclose(
                        runs["lanes8"].state_0.joint_qd.numpy(),
                        runs["oracle"].state_0.joint_qd.numpy(),
                        rtol=1e-4,
                        atol=1e-5,
                    )
            q, q_oracle = runs["lanes8"].state_0.joint_q.numpy(), runs["oracle"].state_0.joint_q.numpy()
            test.assertTrue(np.all(np.isfinite(q)))
            test.assertLess(_rms(q, q_oracle), 1e-3)
            if chain == "chain:40@0.03":
                test.assertGreater(widest, 32)
            for run in runs.values():
                test.assertFalse(np.any(run.solver.constraint_overflow.numpy()))


def test_global_body_between_two_topologies(test, device):
    """Couple two topologies through a free world -1 box that rests on both chains' tips.

    A two-link and a three-link chain lie on the ground in two worlds, their tips 0.2 m apart,
    and a 0.3 m by 0.3 m by 0.1 m box of the global world -1 drops 1 cm onto both. Its contacts
    with the two chains are units between the box and an articulation of each factor group. After
    60 steps (a quarter second) the box rests on the tips, touching both groups' chains, finite,
    without lost rows, at the height of the serial oracle's run (``scene_sweep="serial"``) within
    1 mm.
    """
    builder = _scene_builder(
        ["chain:2@0.03", "chain:3@0.03"],
        xforms=[
            wp.transform(wp.vec3(0.0, -0.1, 0.0), wp.quat_identity()),
            wp.transform(wp.vec3(-0.3, 0.1, 0.0), wp.quat_identity()),
        ],
    )
    test.assertEqual(builder.current_world, -1)
    # Both tips end at x = 0.6, 0.2 m apart on y; the box covers both tip links.
    box = builder.add_body(xform=wp.transform(wp.vec3(0.45, 0.0, 0.12), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.15, hy=0.15, hz=0.05)
    model = builder.finalize(device=device)
    test.assertEqual(int(model.body_world.numpy()[box]), -1)
    runs = {sweep: _Run(model, 128, {"scene_sweep": sweep}) for sweep in ("phases", "serial")}
    test.assertEqual([group.size for group in _factor_groups(runs["phases"].solver)], [2, 3])
    test.assertEqual(runs["phases"].solver._scene.response_kind, "generic")
    for _ in range(60):
        for run in runs.values():
            run.step()
    test.assertGreaterEqual(runs["phases"].solver._scene_units.statistics()["two_sided"], 2)
    # The box's kept contacts (rows on path 0) reach an articulation of each factor group.
    run = runs["phases"]
    box_art = int(np.flatnonzero(model.joint_child.numpy()[model.articulation_start.numpy()[:-1]] == box)[0])
    count = int(run.contacts.rigid_contact_count.numpy()[0])
    art_a = run.solver._contact_art_a.numpy()[:count]
    art_b = run.solver._contact_art_b.numpy()[:count]
    kept = (run.solver._contact_slot.numpy()[:count] >= 0) & (run.solver._contact_path.numpy()[:count] == 0)
    other = np.where(art_a == box_art, art_b, art_a)
    with_box = kept & ((art_a == box_art) | (art_b == box_art)) & (other >= 0) & (other != box_art)
    art_group = run.solver._scene.art_group.numpy()
    test.assertEqual(set(art_group[other[with_box]].tolist()), {0, 1})
    for run in runs.values():
        for field in ("joint_q", "joint_qd", "body_q"):
            test.assertTrue(np.all(np.isfinite(getattr(run.state_0, field).numpy())), msg=field)
        test.assertFalse(np.any(run.solver.constraint_overflow.numpy()))
    height = runs["phases"].state_0.body_q.numpy()[box, 2]
    # On the 6 cm tip links (resting on the ground the box would be 5 cm up).
    test.assertGreater(height, 0.09)
    test.assertLess(abs(height - runs["serial"].state_0.body_q.numpy()[box, 2]), 1e-3)


def test_diagonal_topologies_plan_diagonal_factors(test, device):
    """Plan a topology without coupled DOFs as a sparse factor of one entry per DOF: a diagonal factor.

    A pendulum (one revolute joint) and a fixed base with three revolute children have a diagonal
    mass matrix. Their plans hold the diagonal only, each DOF its own support and every pivot in
    one dependency level, so the scene solve factors them as ordinary factor groups.
    """
    pendulum = _SparseMassMatrixPlan.build(np.array([-1]), np.array([1]))
    test.assertEqual(pendulum.dof_count, 1)
    test.assertEqual(pendulum.nonzero_count, 1)

    model = _scene_worlds(device, ["siblings"])
    keys = scene_topology_keys(SolverSteel(model, solve_mode="serial"), model)
    test.assertEqual(len(keys), 1)
    siblings = _factor_plan(keys[0])
    test.assertEqual(siblings.dof_count, 3)
    test.assertEqual(siblings.nonzero_count, 3)
    test.assertEqual(sorted(int(mask) for mask in siblings.ancestor_mask), [1, 2, 4])
    test.assertEqual(siblings.factor_level_count, 1)


def _revolute_dofs(model, world: int) -> np.ndarray:
    """The DOFs (indices into ``joint_qd``) of ``world``'s revolute joints, also their indices into ``joint_q``."""
    joints = np.flatnonzero(
        (model.joint_world.numpy() == world) & (model.joint_type.numpy() == int(newton.JointType.REVOLUTE))
    )
    dofs = model.joint_qd_start.numpy()[joints]
    if not np.array_equal(dofs, model.joint_q_start.numpy()[joints]):
        raise ValueError(f"world {world}'s revolute coordinates and DOFs have different indices")
    return dofs


def _kept_contacts_of(run, art: int) -> int:
    """Kept contacts of ``run``'s last step (rows on the factor-coordinate path 0) with ``art`` as an endpoint."""
    solver = run.solver
    count = int(run.contacts.rigid_contact_count.numpy()[0])
    kept = (solver._contact_slot.numpy()[:count] >= 0) & (solver._contact_path.numpy()[:count] == 0)
    touches = (solver._contact_art_a.numpy()[:count] == art) | (solver._contact_art_b.numpy()[:count] == art)
    return int(np.count_nonzero(kept & touches))


def test_diagonal_scene_tracks_steel(test, device):
    """Follow Steel's diagonal-mass path on a pendulum and three revolute siblings, each a factor group with a
    diagonal plan.

    Steel solves the pendulum and the siblings with its diagonal-mass kernels (a division per DOF)
    and the humanoid beside them densely; the scene solve factors the three topologies in three
    factor groups, the first two with one entry per DOF. The velocities after one step agree to
    rounding, and the joint coordinates over 20 steps stay within 1 mrad RMS. Left alone the
    siblings would hang still: they start swinging toward joint limits of 0.2 rad, which hold
    them, so limit rows run on a diagonal factor. In the second scene the pendulum lies on the
    ground, so contact rows run on a one-DOF factor.
    """
    limit, q0, qd0 = 0.2, np.array([0.1, -0.1, 0.1]), np.array([2.0, -2.0, 2.5])
    scenes = {
        "hanging": ["pendulum", "siblings", "humanoid"],
        "lying": ["chain:1@0.03", "siblings", "humanoid"],
    }
    for name, kinds in scenes.items():
        with test.subTest(scene=name):
            model = _scene_worlds(device, kinds)
            pendulum, siblings = _revolute_dofs(model, 0), _revolute_dofs(model, 1)
            test.assertEqual((len(pendulum), len(siblings)), (1, 3))
            for array, values in (
                (model.joint_limit_lower, -limit),
                (model.joint_limit_upper, limit),
                (model.joint_q, q0),
                (model.joint_qd, qd0),
            ):
                host = array.numpy()
                host[siblings] = values
                array.assign(host)
            scene = _Run(model, 128, {})
            steel = _Run(model, 128, {}, solve_mode="serial")

            # Steel's diagonal-mass kernels for the pendulum and the siblings, its dense ones for the humanoid.
            test.assertEqual(steel.solver._diagonal_mass_sizes, frozenset((1, 3)))
            test.assertIsNone(steel.solver._sparse_mass_matrix_size)
            test.assertIsNone(steel.solver._scene)
            # The scene solve selects no diagonal kernel: three factor groups, the first two diagonal plans.
            test.assertEqual(scene.solver._diagonal_mass_sizes, frozenset())
            test.assertEqual(len(scene.solver._scene_plan.groups), 3)
            groups = _factor_groups(scene.solver)
            test.assertEqual([group.size for group in groups], [1, 3, 27])
            test.assertEqual([group.nonzero_count == group.size for group in groups], [True, True, False])
            test.assertEqual(scene.solver._scene.response_kind, "generic")

            reach = 0.0
            pendulum_contacts = 0
            for step in range(20):
                scene.step()
                steel.step()
                reach = max(reach, float(np.max(np.abs(scene.state_0.joint_q.numpy()[siblings]))))
                pendulum_contacts = max(pendulum_contacts, _kept_contacts_of(scene, 0))
                if step == 0:
                    qd_scene, qd_steel = scene.state_0.joint_qd.numpy(), steel.state_0.joint_qd.numpy()
                    if name == "hanging":
                        # The pendulum falls from horizontal: about 0.2 rad/s after one step.
                        test.assertGreater(abs(float(qd_steel[pendulum[0]])), 0.1)
                    np.testing.assert_allclose(qd_scene, qd_steel, rtol=1e-5, atol=1e-6)
            q_scene, q_steel = scene.state_0.joint_q.numpy(), steel.state_0.joint_q.numpy()
            test.assertTrue(np.all(np.isfinite(q_scene)))
            test.assertLess(_rms(q_scene, q_steel), 1e-3)
            # Unlimited, the siblings would swing to 0.28 rad within the 20 steps; the limits stop them at 0.2.
            test.assertGreater(reach, limit - 0.02)
            test.assertLess(reach, limit + 0.02)
            if name == "lying":
                test.assertGreater(pendulum_contacts, 0)
            for run in (scene, steel):
                test.assertFalse(np.any(run.solver.constraint_overflow.numpy()))


def test_six_dof_chain_beside_free_boxes_tracks_steel(test, device):
    """Follow Steel's dense solve on six-DOF chains beside free boxes, the chains a factor group of their own.

    The chains and the boxes have six response DOFs each. Steel factors all four in one dense
    size-6 group (its sparse path refuses a six-DOF articulation beside free bodies). The scene
    solve factors the chains in their own factor group and the boxes are its free group, while
    Steel's dense size-6 pipeline keeps running for all four: the rows read the boxes' dense
    factors and ignore the chains'. The boxes rest on the ground, so their contact rows run in
    both scenes; in the second the chains lie on the ground too, so contact rows run on the six-DOF
    factor group. The velocities, the boxes' poses and the chains' joint coordinates after one
    step agree to rounding (the chains' velocities to 1e-4 of the fastest chain speed, at least
    5e-7: a six-link chain's sparse inverse factor and Steel's dense Cholesky round apart within
    cond(H) eps_f32, which the boxes do not change), and over 30 steps the poses and coordinates stay within 1e-3
    RMS. Steel's single-plan factor attributes stay unset, and the boxes' rows run in both scenes.
    The specialised response kernel, which indexes the factors by Steel's size group, is refused.
    """
    scenes = {
        "hanging": ["chain:6+box", "chain:6+box"],
        "lying": ["chain:6@0.03+box", "chain:6@0.03+box"],
    }
    for name, kinds in scenes.items():
        with test.subTest(scene=name):
            model = _scene_worlds(device, kinds)
            revolute = np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.REVOLUTE))
            chain_q = model.joint_q_start.numpy()[revolute]
            chain_dofs = model.joint_qd_start.numpy()[revolute]
            boxes = np.concatenate([_world_parts(model, world)[1] for world in range(len(kinds))])
            free = np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.FREE))
            box_dofs = np.concatenate([np.arange(start, start + 6) for start in model.joint_qd_start.numpy()[free]])
            test.assertEqual((len(chain_q), len(boxes), len(box_dofs)), (12, 2, 12))
            scene = _Run(model, 128, {})
            steel = _Run(model, 128, {}, solve_mode="serial")

            # Steel: one dense size-6 group of the two chains and the two boxes, no sparse factor.
            test.assertEqual(steel.solver._n_arts_by_size, {6: 4})
            test.assertIsNone(steel.solver._sparse_mass_matrix_size)
            test.assertIsNone(steel.solver._scene)
            # The scene: the chains' factor group, then the boxes' free group; Steel's dense size-6
            # group still holds all four, the boxes' CRBA reading the composite inertias.
            groups = [(group.kind, group.size, group.arts) for group in scene.solver._scene_plan.groups]
            test.assertEqual(groups, [("factor", 6, (0, 2)), ("free", 6, (1, 3))])
            test.assertEqual(scene.solver._n_arts_by_size, {6: 4})
            test.assertNotIn(6, scene.solver._scene_factor_only_sizes)
            test.assertNotIn(6, scene.solver._free_body_inertia_sizes)
            # The decode's free rows are the boxes, not Steel's size-6 group.
            test.assertEqual(scene.solver._scene.free_group_to_art.numpy().tolist(), [1, 3])
            # Steel's single-plan factor attributes view a factor group only when it is Steel's whole size group.
            for name_ in ("_sparse_mass_matrix_indices", "_sparse_Linv", "_crba_sparse_factor_kernel"):
                test.assertIsNone(getattr(scene.solver, name_), msg=name_)
            # The specialised response kernel indexes factors by Steel's size-6 group, which the boxes share.
            test.assertEqual(scene.solver._scene.response_kind, "generic")
            if name == "hanging":
                with kernel_overrides({"scene_response_kernel": "specialized"}):
                    with test.assertRaisesRegex(NotImplementedError, "in Steel's size order"):
                        SolverSteel(model, articulation_constraint_max=128, **SCENE)

            chain_contacts = 0
            box_contacts = 0
            for step in range(30):
                scene.step()
                steel.step()
                chain_contacts = max(chain_contacts, _kept_contacts_of(scene, 0) + _kept_contacts_of(scene, 2))
                box_contacts = max(box_contacts, _kept_contacts_of(scene, 1) + _kept_contacts_of(scene, 3))
                if step == 0:
                    qd_scene, qd_steel = scene.state_0.joint_qd.numpy(), steel.state_0.joint_qd.numpy()
                    if name == "hanging":
                        # The chains fall from horizontal: about 0.2 rad/s after one step.
                        test.assertGreater(np.max(np.abs(qd_steel[chain_dofs])), 0.1)
                    np.testing.assert_allclose(qd_scene[box_dofs], qd_steel[box_dofs], rtol=1e-5, atol=1e-6)
                    # The scene's explicit inverse factor and Steel's dense Cholesky each solve the
                    # six-link chain's H (cond 5.5e3) to about 0.05 cond(H) eps_f32 of |qdd| against a
                    # float64 reference, and they round apart by 1.3e-5 of the fastest speed, with or
                    # without the boxes (measured): 2.7e-6 rad/s at 0.22 rad/s hanging. At rest the
                    # chains move at 1.19e-6 rad/s and differ by 1.76e-7, so the floor sits at 5e-7.
                    chain_atol = max(5e-7, 1e-4 * float(np.max(np.abs(qd_steel[chain_dofs]))))
                    np.testing.assert_allclose(qd_scene[chain_dofs], qd_steel[chain_dofs], rtol=1e-5, atol=chain_atol)
                    np.testing.assert_allclose(
                        scene.state_0.body_q.numpy()[boxes], steel.state_0.body_q.numpy()[boxes], rtol=1e-5, atol=1e-6
                    )
                    np.testing.assert_allclose(
                        scene.state_0.joint_q.numpy()[chain_q],
                        steel.state_0.joint_q.numpy()[chain_q],
                        rtol=1e-5,
                        atol=1e-6,
                    )
            q_scene, q_steel = scene.state_0.joint_q.numpy(), steel.state_0.joint_q.numpy()
            box_scene, box_steel = scene.state_0.body_q.numpy()[boxes], steel.state_0.body_q.numpy()[boxes]
            test.assertTrue(np.all(np.isfinite(q_scene)))
            test.assertLess(_rms(q_scene[chain_q], q_steel[chain_q]), 1e-3)
            test.assertLess(_rms(box_scene, box_steel), 1e-3)
            # The boxes rest on the ground in both scenes: their rows read Steel's size-6 factors.
            test.assertGreater(box_contacts, 0)
            if name == "lying":
                test.assertGreater(chain_contacts, 0)
            for run in (scene, steel):
                test.assertFalse(np.any(run.solver.constraint_overflow.numpy()))


def _chains_over_boxes(device) -> newton.Model:
    """Two worlds of a six-link chain 0.25 m up, 2 cm above a free 0.2 m box under its third link, on the ground."""
    world = _make_chain_world(6, root_height=0.25)
    box = world.add_body(xform=wp.transform(wp.vec3(0.75, 0.0, 0.1), wp.quat_identity()))
    world.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverSteel.register_custom_attributes(builder)
    builder.rigid_gap = 0.005
    for _ in range(2):
        builder.add_world(world)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def test_six_dof_chain_on_a_free_box_tracks_steel(test, device):
    """Follow Steel's dense solve on six-DOF chains falling onto free boxes, with rows joining a chain link and a box.

    Each chain lies 2 cm above its world's box, inside the chains' 0.1 m contact gap, so contact
    rows join a chain link (the six-DOF factor group) and the box (the free group): one endpoint
    reads the chain's factor through the group table, the other the box's dense factor by its index
    in Steel's size-6 group, which holds the chain and the box side by side. The chains fall onto
    the boxes and swing toward the ground, and over 30 steps their joint coordinates stay within
    1e-4 RMS of Steel's and the boxes' poses within 1e-5. The coupled rows run in another order
    than Steel's, so the paths are compared over the run rather than to rounding after one step.
    """
    model = _chains_over_boxes(device)
    revolute = np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.REVOLUTE))
    chain_q = model.joint_q_start.numpy()[revolute]
    boxes = np.concatenate([_world_parts(model, world)[1] for world in range(2)])
    test.assertEqual((len(chain_q), len(boxes)), (12, 2))
    scene = _Run(model, 128, {})
    steel = _Run(model, 128, {}, solve_mode="serial")
    groups = [(group.kind, group.size, group.arts) for group in scene.solver._scene_plan.groups]
    test.assertEqual(groups, [("factor", 6, (0, 2)), ("free", 6, (1, 3))])
    test.assertEqual(steel.solver._n_arts_by_size, {6: 4})
    test.assertEqual(scene.solver._scene.response_kind, "generic")
    joined = 0
    for _ in range(30):
        scene.step()
        steel.step()
        solver = scene.solver
        count = int(scene.contacts.rigid_contact_count.numpy()[0])
        art_a, art_b = solver._contact_art_a.numpy()[:count], solver._contact_art_b.numpy()[:count]
        kept = (solver._contact_slot.numpy()[:count] >= 0) & (solver._contact_path.numpy()[:count] == 0)
        chain_box = (np.isin(art_a, (0, 2)) & np.isin(art_b, (1, 3))) | (
            np.isin(art_a, (1, 3)) & np.isin(art_b, (0, 2))
        )
        joined = max(joined, int(np.count_nonzero(kept & chain_box)))
    test.assertGreater(joined, 0)
    q_scene, q_steel = scene.state_0.joint_q.numpy(), steel.state_0.joint_q.numpy()
    test.assertTrue(np.all(np.isfinite(q_scene)))
    # Measured: 1.17e-6 rad for the chains and 2.5e-8 for the boxes after the 30 steps.
    test.assertLess(_rms(q_scene[chain_q], q_steel[chain_q]), 1e-4)
    test.assertLess(_rms(scene.state_0.body_q.numpy()[boxes], steel.state_0.body_q.numpy()[boxes]), 1e-5)
    for run in (scene, steel):
        test.assertFalse(np.any(run.solver.constraint_overflow.numpy()))


# The probe of the mask helpers: one thread per mask, which tests and ranks every coordinate (and
# ranks 64 * MASK_WORDS, the mask's popcount), then walks the mask with mask_next until it is empty.
_MASK_PROBE = """
    unsigned long long mask[MASK_WORDS];
    for (int word = 0; word < MASK_WORDS; ++word) mask[word] = masks.data[item * MASK_WORDS + word];
    const int bits = 64 * MASK_WORDS;
    for (int bit = 0; bit < bits; ++bit) {
        tested.data[item * bits + bit] = mask_test(mask, bit) ? 1 : 0;
        ranked.data[item * (bits + 1) + bit] = mask_rank(mask, bit);
    }
    ranked.data[item * (bits + 1) + bits] = mask_rank(mask, bits);
    int count = 0;
    for (int bit = mask_next(mask); bit >= 0; bit = mask_next(mask)) walked.data[item * (bits + 1) + count++] = bit;
    walked.data[item * (bits + 1) + count] = -1;
"""


@cache
def _mask_probe_kernel(mask_words: int) -> wp.Kernel:
    @wp.func_native(mask_helpers_snippet(mask_words) + _MASK_PROBE)
    def probe_native(
        item: int,
        masks: wp.array[wp.uint64],
        tested: wp.array[int],
        ranked: wp.array[int],
        walked: wp.array[int],
    ): ...

    def probe_kernel(
        masks: wp.array[wp.uint64],
        tested: wp.array[int],
        ranked: wp.array[int],
        walked: wp.array[int],
    ):
        probe_native(wp.tid(), masks, tested, ranked, walked)

    probe_kernel.__name__ = f"mask_probe_{mask_words}"
    probe_kernel.__qualname__ = probe_kernel.__name__
    return wp.kernel(enable_backward=False, module="unique")(probe_kernel)


def test_mask_helpers_match_single_word(test, device):
    """Test, rank and walk support masks of one and two ``uint64`` words in native code, as Python ints do.

    The helpers of the kernels' multi-word masks (``_MASK_HELPERS_CUDA`` with ``MASK_WORDS`` 1 and
    2), on random 64- and 128-bit masks with the empty and the full one: ``mask_test`` of every
    coordinate, ``mask_rank`` of every coordinate and of ``64 * MASK_WORDS`` (the popcount), and
    ``mask_next`` until the mask is empty, which lists its coordinates in increasing order. Word 0
    holds coordinates 0 to 63. A kernel whose masks fit one word keeps the single-word
    expressions instead, character for character: the tripwire (``test_scene_tripwire``) pins the
    contact response's source, and this test the joint-limit row builder's, whose one-word text is
    Steel's own builder's (``LIMIT_ROWS_SNIPPET_SHA256[1]``).
    """
    limit_rows = sparse_joint_limit_rows_snippet(1)
    test.assertEqual(hashlib.sha256(limit_rows.encode()).hexdigest(), LIMIT_ROWS_SNIPPET_SHA256[1])
    test.assertNotEqual(sparse_joint_limit_rows_snippet(2), limit_rows)
    rng = np.random.default_rng(1707)
    for mask_words in (1, 2):
        with test.subTest(mask_words=mask_words):
            bits = 64 * mask_words
            values = [0, (1 << bits) - 1, 1 << (bits - 1), 1 | (1 << 63)]
            values += [int.from_bytes(rng.bytes(8 * mask_words), "little") for _ in range(12)]
            # Sparse masks too, whose walk crosses empty words.
            values += [int(sum(1 << int(bit) for bit in rng.choice(bits, size=3, replace=False))) for _ in range(8)]
            words = np.asarray(
                [[(value >> (64 * word)) & ((1 << 64) - 1) for word in range(mask_words)] for value in values],
                dtype=np.uint64,
            )
            masks = wp.array(words.reshape(-1), dtype=wp.uint64, device=device)
            tested = wp.full(len(values) * bits, -2, dtype=int, device=device)
            ranked = wp.full(len(values) * (bits + 1), -2, dtype=int, device=device)
            walked = wp.full(len(values) * (bits + 1), -2, dtype=int, device=device)
            wp.launch(
                _mask_probe_kernel(mask_words), dim=len(values), inputs=[masks, tested, ranked, walked], device=device
            )
            tested = tested.numpy().reshape(len(values), bits)
            ranked = ranked.numpy().reshape(len(values), bits + 1)
            walked = walked.numpy().reshape(len(values), bits + 1)
            for index, value in enumerate(values):
                where = f"mask {value:#x}"
                expected = [(value >> bit) & 1 for bit in range(bits)]
                test.assertEqual(tested[index].tolist(), expected, msg=where)
                expected = [(value & ((1 << bit) - 1)).bit_count() for bit in range(bits + 1)]
                test.assertEqual(ranked[index].tolist(), expected, msg=where)
                members = [bit for bit in range(bits) if (value >> bit) & 1]
                test.assertEqual(walked[index, : len(members) + 1].tolist(), [*members, -1], msg=where)


def _limited(model, limit: float = 0.01):
    """``model`` with every DOF limited to ``[-limit, limit]``: from rest, inside the activation gap of both bounds."""
    model.joint_limit_lower.fill_(-limit)
    model.joint_limit_upper.fill_(limit)
    return model


def _wide_tree_tracks_steel(test, model, sizes, rows, *, collide=True, step_one=True, atol_speed=0.0, limit=None):
    """Step ``model`` 20 times in the scene solve and in Steel's dense solve, and compare them.

    ``sizes`` are the scene's factor groups, the first wider than 64 DOFs. ``step_one`` compares the
    velocities after one step at ``rtol=1e-4`` and ``atol = max(1e-5, atol_speed * max|qd|)`` of
    Steel's fastest DOF; the joint coordinates after 20 steps stay within 1 mrad RMS. ``collide``
    steps with the collision pipeline's contacts (else none), and ``limit`` is the bound of a
    :func:`_limited` model: both solvers then hold every DOF within 2 mrad of it, and rows ran.
    """
    scene = _Run(model, rows, {}, collide=collide)
    steel = _Run(model, rows, {}, collide=collide, solve_mode="serial")
    plan = scene.solver._scene_plan
    groups = _factor_groups(scene.solver)
    test.assertEqual(plan.mask_words, 2)
    test.assertEqual([group.size for group in groups], sizes)
    test.assertEqual([group.mask_words for group in groups], [-(-size // 64) for size in sizes])
    test.assertEqual(scene.solver._scene.response_kind, "generic")
    test.assertTrue(
        scene.solver._scene.response_kernel.key.startswith(f"populate_scene_contact_response_{sizes[0]}_8_4_w2")
    )
    test.assertEqual(scene.solver._sparse_body_dof_mask.shape, (2 * model.body_count,))
    # Past 64 DOFs Steel keeps its dense factors.
    test.assertIsNone(steel.solver._sparse_mass_matrix_size)
    test.assertIsNone(steel.solver._scene)
    rows_seen = 0
    for step in range(20):
        scene.step()
        steel.step()
        rows_seen = max(rows_seen, int(scene.solver._constraint_count.numpy()[0]))
        if step == 0 and step_one:
            qd_scene, qd_steel = scene.state_0.joint_qd.numpy(), steel.state_0.joint_qd.numpy()
            speed = float(np.max(np.abs(qd_steel)))
            test.assertGreater(speed, 0.1)
            np.testing.assert_allclose(qd_scene, qd_steel, rtol=1e-4, atol=max(1e-5, atol_speed * speed))
    q_scene, q_steel = scene.state_0.joint_q.numpy(), steel.state_0.joint_q.numpy()
    test.assertTrue(np.all(np.isfinite(q_scene)))
    test.assertLess(_rms(q_scene, q_steel), 1e-3)
    if limit is not None:
        test.assertGreater(rows_seen, 0)
        for run in (scene, steel):
            reach = float(np.max(np.abs(run.state_0.joint_q.numpy())))
            test.assertLess(abs(reach - limit), 2e-3, msg=f"reach {reach}")
    for run in (scene, steel):
        test.assertFalse(np.any(run.solver.constraint_overflow.numpy()))
    return rows_seen


def test_tree_65_tracks_steel(test, device):
    """Follow Steel's dense solve on a 65-link chain, whose support masks take two words.

    The 65 revolute DOFs of :func:`_tree`'s chain are one factor group with two-word masks
    (``mask_words == 2``); Steel factors the chain densely. Hanging, the chain swings under its
    drives (the factor, the unconstrained step and the decode); with every DOF limited to
    0.01 rad around its start, the joint-limit rows of both bounds run from the first step, their
    supports spanning both words, and hold the drives within the bound; lying on the ground beside
    a humanoid, contact rows run on the chain's two-word masks and on the humanoid's one-word ones,
    widened to the scene's two.

    The chain's mass matrix has cond(H) = 1.8e7. After one step the scene's explicit inverse factor
    is 1.8e-5 rad/s from a float64 solve at 0.42 rad/s, Steel's dense Cholesky 1.1e-7 (measured),
    as with 64 links and one-word masks (1.2e-5 at 0.39 rad/s), so the velocities agree to 1e-4 of
    the fastest speed rather than the 1e-5 of a better-conditioned tree. The lying chain's contact
    rows run in another order than Steel's (the 64-link chain's step-1 velocities differ by 6.6e-4
    at 0.041 rad/s), so that scene is compared over the run only.
    """
    scenes = {
        "hanging": (lambda: _tree(device, 65), [65], {}),
        "limited": (lambda: _limited(_tree(device, 65)), [65], {"limit": 0.01}),
        "lying": (lambda: _scene_worlds(device, ["chain:65@0.03", "humanoid"]), [65, 27], {"step_one": False}),
    }
    for name, (build, sizes, options) in scenes.items():
        with test.subTest(scene=name):
            rows = 1024 if name == "lying" else 256
            rows_seen = _wide_tree_tracks_steel(test, build(), sizes, rows, atol_speed=1e-4, **options)
            if name == "lying":
                test.assertGreater(rows_seen, 0)


def test_branched_tree_70_tracks_steel(test, device):
    """Follow Steel's dense solve on a 70-link binary tree, whose support masks take two words.

    :func:`_tree`'s branched tree has 70 revolute DOFs in seven levels, a factor group with
    two-word masks whose deep links' supports reach past coordinate 63 (the reverse depth-first
    order puts the root last). Hanging, its drives swing it; with every DOF limited to 0.01 rad
    around its start, the joint-limit rows run from the first step and hold it within the bound.
    The tree steps without contacts: its sibling subtrees' capsules coincide, and their thousands
    of self-contacts would pass both solvers' row capacity.
    """
    scenes = {
        "hanging": (lambda: _tree(device, 70, branched=True), {}),
        "limited": (lambda: _limited(_tree(device, 70, branched=True)), {"limit": 0.01}),
    }
    for name, (build, options) in scenes.items():
        with test.subTest(scene=name):
            _wide_tree_tracks_steel(test, build(), [70], 512, collide=False, **options)


def test_two_word_masks_are_bitwise_one_word(test, device):
    """Build rows and step bit for bit with support masks widened to two words (``scene_mask_words=2``).

    The test-only override gives every factor group two-word masks, although each fits one, so the
    multi-word kernels (the contact response and the joint-limit rows) run where the one-word
    kernels are the reference: they visit the same coordinates in the same order, so the rows
    after the first step and the states after each of 30 steps are bitwise equal. On a crowd of
    humanoids 0.4 m apart (contacts with two factor endpoints), on humanoids beside free boxes
    that touch them (a factor and a free endpoint), and on a three-link and a 40-link chain lying
    on the ground with every DOF limited to 0.01 rad (two factor groups, contact and limit rows).
    Last, :func:`_crossing_worlds` limited to 0.01 rad, whose 65-link chain takes two words, against
    three: rows with coordinates past 63, and contacts between groups of two and one word. The
    specialised response kernel, which reads one-word masks, is refused.
    """
    # Each scene: its builder, rows per world, and the words of the reference and of the widened run.
    scenes = {
        "pairs": (lambda: _humanoids(device, worlds=2, per_world=4, spacing=0.4), 4 * 128, 1, 2),
        "touch": (lambda: _humanoids(device, worlds=2, props=True, spacing=0.4), 128, 1, 2),
        "chains": (lambda: _limited(_scene_worlds(device, ["chain:3@0.03", "chain:40@0.03"])), 1024, 1, 2),
        "crossing": (lambda: _limited(_crossing_worlds(device)), CROSSING_ROWS, 2, 3),
    }
    rows_fields = (
        "_sparse_row_dof",
        "_sparse_row_factor",
        "_sparse_row_free_response",
        "_sparse_row_incident",
        "_diag",
    )
    for name, (build, rows, words, wide) in scenes.items():
        with test.subTest(scene=name):
            model = build()
            one = _Run(model, rows, {})
            two = _Run(model, rows, {"scene_mask_words": wide})
            test.assertEqual(one.solver._scene_plan.mask_words, words)
            test.assertEqual(two.solver._scene_plan.mask_words, wide)
            test.assertEqual(
                [group.mask_words for group in _factor_groups(two.solver)], [wide] * len(_factor_groups(one.solver))
            )
            test.assertEqual(two.solver._sparse_body_dof_mask.shape, (wide * model.body_count,))
            test.assertIn(f"_w{wide}", two.solver._scene.response_kernel.key)
            for arrays in two.solver._scene.group_arrays:
                limit_kernel = arrays.internal_kernels[FAMILY_LIMIT]
                test.assertTrue(limit_kernel.key.startswith(f"build_sparse_joint_limit_rows_w{wide}"))
                test.assertEqual(arrays.indices.ancestor_mask.shape, (wide * arrays.group.size,))
            for step in range(30):
                one.step()
                two.step()
                if step == 0:
                    row_count = int(one.solver._constraint_count.numpy()[0])
                    test.assertGreater(row_count, 0)
                    test.assertEqual(int(two.solver._constraint_count.numpy()[0]), row_count)
                    for field in rows_fields:
                        expected = getattr(one.solver, field).numpy()[0, :row_count]
                        actual = getattr(two.solver, field).numpy()[0, :row_count]
                        test.assertTrue(np.array_equal(expected, actual), msg=f"{field} after step 1")
                    if name in ("chains", "crossing"):
                        row_type = one.solver._row_type.numpy()[0, :row_count]
                        limits = int(np.count_nonzero(row_type == PGS_CONSTRAINT_TYPE_JOINT_LIMIT))
                        test.assertGreater(limits, 0)
                        test.assertGreater(row_count, limits)
                for field in ("joint_q", "joint_qd", "body_q"):
                    expected = getattr(one.state_0, field).numpy()
                    actual = getattr(two.state_0, field).numpy()
                    test.assertTrue(np.array_equal(expected, actual), msg=f"{field} after step {step + 1}")
            for run in (one, two):
                test.assertFalse(np.any(run.solver.constraint_overflow.numpy()))
    with kernel_overrides({"scene_response_kernel": "specialized", "scene_mask_words": 2}):
        with test.assertRaisesRegex(ValueError, "one-word support masks"):
            SolverSteel(_humanoids(device, worlds=2), articulation_constraint_max=128, **SCENE)


def test_scene_refuses_groups_past_shared_memory(test, device):
    """Refuse at construction a topology whose factor or contact rows overflow a block's static shared memory.

    The group-table response kernel's static shared memory is its scratch, which grows with the
    largest group's DOFs, plus the 4 B per thread that Warp reserves in every kernel
    (:func:`scene_contact_response_shared_bytes`): with 8 lanes per contact and 128 threads,
    ``192 * (dof_max + 6) + 512`` bytes, past 48 KiB at 248 DOFs. A 248-link binary tree has a small
    factor that fits a warp, so the scene solve refuses it for its contact rows, naming the bytes
    needed and allowed and the most DOFs that fit (247), before the kernel compiles; Steel
    (``solve_mode="serial"``) builds it as before, densely. A 247-link tree builds, its response
    kernel (four mask words) compiles at the launch's 128 threads (248 to 250 DOFs were found to fail
    there, 0xc080 to 0xc200 bytes), and it steps. A 108-link chain's factor does not fit one warp:
    refused for its factor, naming the bytes.
    """
    limit = min(48 * 1024, int(wp.get_device(device).max_shared_memory_per_block))
    fits = scene_contact_response_dof_limit(limit)
    test.assertEqual(fits, 247)
    test.assertLessEqual(scene_contact_response_shared_bytes(fits), limit)
    test.assertGreater(scene_contact_response_shared_bytes(fits + 1), limit)

    wide = _tree(device, fits + 1, branched=True)
    needed = scene_contact_response_shared_bytes(fits + 1)
    message = (
        f"cannot build contact rows for a {fits + 1}-DOF topology: the contact response kernel needs {needed} B "
        f"of static shared memory per block (its scratch and Warp's per-thread reservation), the device allows "
        f"{limit} B, enough for topologies of at most {fits} DOFs"
    )
    with test.assertRaisesRegex(NotImplementedError, re.escape(message)):
        SolverSteel(wide, articulation_constraint_max=64, **SCENE)
    steel = SolverSteel(wide, articulation_constraint_max=64, **{**SCENE, "solve_mode": "serial"})
    test.assertIsNone(steel._scene)

    run = _Run(_tree(device, fits, branched=True), 64, {}, collide=False)
    plan = run.solver._scene_plan
    test.assertEqual([(group.size, group.mask_words) for group in _factor_groups(run.solver)], [(fits, 4)])
    test.assertEqual(plan.mask_words, 4)
    kernel = run.solver._scene.response_kernel
    test.assertTrue(kernel.key.startswith(f"populate_scene_contact_response_{fits}_8_4_w4"))
    # The scene launches the response kernel in blocks of 128 threads.
    wp.load_module(kernel.module, device=device, block_dim=128)
    run.step()
    test.assertTrue(np.all(np.isfinite(run.state_0.joint_q.numpy())))

    with test.assertRaisesRegex(
        NotImplementedError,
        r"cannot factor a 108-DOF topology of 5886 factor entries: one warp stages its factor in 49680 B",
    ):
        SolverSteel(_tree(device, 108), articulation_constraint_max=64, **SCENE)


def test_scene_support_reads_distinct_masks(test, device):
    """Read each distinct contact mask once (numpy first), as the per-shape ints would give it.

    :func:`distinct_mask_values` on random one-, two- and three-word masks with repeats equals the
    set of every row's int. On a 65-link chain beside a humanoid, whose body masks take two words
    with the humanoid's widened from one, it gives each factor group the same masks as the
    per-shape ints, and the scene's support (:func:`scene_support`) equals the
    support computed from every pair of contact shapes' int masks.
    """
    rng = np.random.default_rng(1707)
    for words in (1, 2, 3):
        with test.subTest(words=words):
            pool = rng.integers(0, 2**63, size=(12, words), dtype=np.uint64) * np.uint64(2) + np.uint64(1)
            pool[0] = 0
            pool[1] = np.iinfo(np.uint64).max
            masks = pool[rng.integers(0, len(pool), size=200)]
            masks = masks[:, 0] if words == 1 else masks
            test.assertEqual(distinct_mask_values(masks), sorted({mask_value(row) for row in masks}))

    model = _scene_worlds(device, ["chain:65@0.03", "humanoid"])
    solver = SolverSteel(model, solve_mode="serial")
    plan = solver._model_plan
    groups = plan_scene_groups(
        scene_topology_keys(solver, model), (plan.is_free_rigid != 0) & (plan.response_dof_count > 0)
    )
    test.assertEqual([(group.size, group.mask_words) for group in groups], [(65, 2), (27, 1)])
    body_masks, body_group = factor_body_masks(solver, model, groups)
    test.assertEqual(body_masks.shape, (model.body_count, 2))
    shape_body = model.shape_body.numpy()
    contact_bodies = shape_body[shape_body >= 0]
    bits, support = [], 0
    for index, group in enumerate(groups):
        shapes = body_masks[contact_bodies[body_group[contact_bodies] == index]]
        per_shape = [mask_value(row) for row in shapes]
        test.assertEqual(distinct_mask_values(shapes), sorted(set(per_shape)))
        group_bits = max(mask_value(mask).bit_count() for mask in group.plan.joint_ancestor_mask)
        support = max([support, group_bits, *((a | b).bit_count() for a in per_shape for b in per_shape)])
        bits.extend([group_bits] * min(len(group.arts), 2))
    widest = sorted(bits, reverse=True)[:2]
    support = min(max(support, widest[0] + widest[1]), solver._max_world_dofs)
    test.assertEqual(scene_support(solver, model, groups, body_masks, body_group), support)


def test_limit_rows_kernels_share_steels_signature(test, device):
    """Give the joint-limit row builder of every mask width Steel's builder's arguments.

    One word is Steel's builder itself (:func:`build_sparse_joint_limit_rows`); the multi-word
    builders repeat its argument list, which this pins name for name and type for type.
    """
    test.assertIs(get_sparse_joint_limit_rows_kernel(1), build_sparse_joint_limit_rows)
    expected = [(arg.label, str(arg.type)) for arg in build_sparse_joint_limit_rows.adj.args]
    for words in (2, 3):
        kernel = get_sparse_joint_limit_rows_kernel(words)
        test.assertIsNot(kernel, build_sparse_joint_limit_rows)
        test.assertEqual([(arg.label, str(arg.type)) for arg in kernel.adj.args], expected, msg=f"{words} words")


CROSSING_ROWS = 2048
"""Rows per world of :func:`_crossing_worlds`: about 1,100 contact and limit rows, with room."""


def _crossing_worlds(device, worlds: int = 2) -> newton.Model:
    """Worlds of a 65-link chain lying on the ground, a six-link chain across its tip, and a humanoid.

    The 65-link chain (:func:`_make_chain_world` at ``root_height=0.03``) runs along x; the
    six-link chain hangs 0.09 m up along y, its third link lying on the long chain's last link
    (x = 19.35 m), whose support spans the long chain's every coordinate, both mask words. The
    lying humanoid sits 2 m off on y, clear of both. So the factor groups are the 65-link chain's
    (two words), the six-link chain's and the humanoid's (one word each), and contacts join the
    first two. The humanoid's passive joint stiffness is zeroed (:func:`_without_passive_springs`).
    """
    world = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverSteel.register_custom_attributes(world)
    world.rigid_gap = 0.005
    world.add_builder(_make_chain_world(65, root_height=0.03))
    across = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * np.pi)
    world.add_builder(_make_chain_world(6, root_height=0.09), xform=wp.transform(wp.vec3(19.35, -0.75, 0.0), across))
    world.add_builder(_humanoid(), xform=wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()))
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverSteel.register_custom_attributes(builder)
    builder.rigid_gap = 0.005
    for _ in range(worlds):
        builder.add_world(world)
    builder.add_ground_plane()
    return _without_passive_springs(builder.finalize(device=device))


def test_two_word_rows_match_the_thread_oracle_bitwise(test, device):
    """Sweep rows with coordinates past 63, and units between groups of two and one mask word, as the serial oracle.

    :func:`_crossing_worlds` limited to 0.01 rad: the 65-link chain's contact and limit rows hold
    its coordinate 64 (the root, last in reverse depth-first order), in a scene support of 130
    entries, and the six-link chain lying across its tip makes two-sided units between a two-word
    and a one-word group. With one thread per side and per unit and serial sums, 30 steps are
    bitwise the serial oracle (``scene_sweep="serial"``): joint coordinates, velocities and the
    rows' impulses.
    """
    model = _limited(_crossing_worlds(device))
    thread = {"scene_side_sweep": "thread", "scene_unit_sweep": "thread"}
    runs = {
        "thread": _Run(model, CROSSING_ROWS, thread),
        "oracle": _Run(model, CROSSING_ROWS, {**thread, "scene_sweep": "serial"}),
    }
    solver = runs["thread"].solver
    plan = solver._scene_plan
    test.assertEqual([(group.size, group.mask_words) for group in _factor_groups(solver)], [(65, 2), (6, 1), (27, 1)])
    test.assertEqual(plan.mask_words, 2)
    test.assertEqual(plan.support, 130)
    test.assertEqual(solver._scene.sum_order, "serial")
    chains = np.asarray(_factor_groups(solver)[0].arts)
    root = solver._articulation_world_dof_offset.numpy()[chains] + 64
    across, past_63 = 0, 0
    for step in range(30):
        for run in runs.values():
            run.step()
        count = int(solver._constraint_count.numpy()[0])
        past_63 = max(past_63, int(np.count_nonzero(np.isin(solver._sparse_row_dof.numpy()[0, :count], root))))
        across = max(across, _contacts_between_groups(runs["thread"]))
        for field in ("joint_q", "joint_qd"):
            expected = getattr(runs["oracle"].state_0, field).numpy()
            test.assertTrue(
                np.array_equal(getattr(runs["thread"].state_0, field).numpy(), expected),
                msg=f"{field} at step {step + 1}",
            )
        oracle = runs["oracle"].solver
        test.assertEqual(int(oracle._constraint_count.numpy()[0]), count)
        test.assertTrue(
            np.array_equal(solver._impulses.numpy()[0, :count], oracle._impulses.numpy()[0, :count]),
            msg=f"impulses at step {step + 1}",
        )
    test.assertGreater(past_63, 0)
    test.assertGreater(across, 0)
    for run in runs.values():
        test.assertTrue(np.all(np.isfinite(run.state_0.joint_q.numpy())))
        test.assertFalse(np.any(run.solver.constraint_overflow.numpy()))


class TestSteelSceneGroups(unittest.TestCase):
    pass


# Keys, groups and the group table are host-side plans; the serial solver that computes the keys
# constructs on either device.
for _fn in (
    test_topology_key_is_the_factor_plan_input,
    test_plan_scene_groups_orders_factor_groups_then_free,
    test_plan_scene_groups_splits_by_articulation,
    test_response_table_requires_a_group_per_articulation,
    test_diagonal_topologies_plan_diagonal_factors,
    test_mask_helpers_match_single_word,
    test_plan_scene_groups_sizes_masks_by_dofs,
    test_scene_support_reads_distinct_masks,
    test_limit_rows_kernels_share_steels_signature,
):
    add_function_test(TestSteelSceneGroups, _fn.__name__, _fn, devices=get_test_devices())

# The articulated scene plan builds on every device; a Steel reference is Steel's serial solve (_steel_on).
for _fn in (
    test_one_topology_scene_has_one_factor_group,
    test_free_only_scene_has_one_free_group,
    test_generic_response_kernel_is_bitwise_specialized,
    test_split_groups_are_bitwise_one_group,
    test_notify_armature_reaches_every_group,
    test_heterogeneous_worlds_match_isolated_worlds_bitwise,
    test_heterogeneous_scene_tracks_steel,
    test_global_body_between_two_topologies,
    test_six_dof_chain_beside_free_boxes_tracks_steel,
    test_six_dof_chain_on_a_free_box_tracks_steel,
    test_tree_65_tracks_steel,
    test_branched_tree_70_tracks_steel,
    test_two_word_masks_are_bitwise_one_word,
):
    add_function_test(TestSteelSceneGroups, _fn.__name__, _fn, devices=get_test_devices())

# CUDA only, each for the reason beside it.
for _fn in (
    test_wide_rows_match_the_thread_oracle_bitwise,  # the phases; the CPU sweeps the serial oracle alone
    test_diagonal_scene_tracks_steel,  # Steel's diagonal-mass kernels against the scene's diagonal factor groups
    test_scene_refuses_groups_past_shared_memory,  # the shared-memory gate (the CPU's factor uses no shared memory)
    test_two_word_rows_match_the_thread_oracle_bitwise,  # the phases
):
    add_function_test(TestSteelSceneGroups, _fn.__name__, _fn, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
