# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import warnings
from math import sqrt
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton import NewtonWarning
from newton._src.geometry import broad_phase_aggregate, broad_phase_nxn
from newton._src.geometry.broad_phase_sap import _advance_sap_chunk_base
from newton._src.geometry.flags import ShapeFlags
from newton.geometry import BroadPhaseAggregate, BroadPhaseAllPairs, BroadPhaseExplicit, BroadPhaseSAP
from newton.tests.test_collision_pipeline import _aggregate_humanoids
from newton.tests.unittest_utils import StdOutCapture, add_function_test, get_cuda_test_devices, get_test_devices

# NOTE: The test_group_pair and test_world_and_group_pair functions below are copied
# from newton._src.geometry.broad_phase_common because they need to be available as
# host-side Python functions for testing/verification. The original functions are
# decorated with @wp.func for use in GPU kernels and can be called from host code,
# but the overhead is huge. This duplication allows us to verify that the GPU collision
# filtering logic matches the expected behavior efficiently without the overhead of
# calling @wp.func decorated functions from the host.


def test_group_pair(group_a: int, group_b: int) -> bool:
    """Test if two collision groups should interact.

    Args:
        group_a: First collision group ID. Positive values indicate groups that only collide with themselves (and with negative groups).
                Negative values indicate groups that collide with everything except their negative counterpart.
                Zero indicates no collisions.
        group_b: Second collision group ID. Same meaning as group_a.

    Returns:
        bool: True if the groups should collide, False if they should not.
    """
    if group_a == 0 or group_b == 0:
        return False
    if group_a > 0:
        return group_a == group_b or group_b < 0
    if group_a < 0:
        return group_a != group_b
    return False


def test_world_and_group_pair(world_a: int, world_b: int, collision_group_a: int, collision_group_b: int) -> bool:
    """Test if two entities should collide based on world indices and collision groups.

    World indices define which simulation world an entity belongs to:
    - Index -1: Global entities that collide with all worlds
    - Indices 0, 1, 2, ...: World-specific entities

    Collision rules:
    1. Entities from different worlds (except -1) do not collide
    2. Global entities (index -1) collide with all worlds
    3. Within the same world, collision groups determine interactions

    Args:
        world_a: World index of first entity
        world_b: World index of second entity
        collision_group_a: Collision group of first entity
        collision_group_b: Collision group of second entity

    Returns:
        bool: True if the entities should collide, False otherwise
    """
    # Check world indices first
    if world_a != -1 and world_b != -1 and world_a != world_b:
        return False

    # If same world or at least one is global (-1), check collision groups
    return test_group_pair(collision_group_a, collision_group_b)


def check_aabb_overlap_host(
    box1_lower: wp.vec3,
    box1_upper: wp.vec3,
    box1_cutoff: float,
    box2_lower: wp.vec3,
    box2_upper: wp.vec3,
    box2_cutoff: float,
) -> bool:
    cutoff_combined = max(box1_cutoff, box2_cutoff)
    return (
        box1_lower[0] <= box2_upper[0] + cutoff_combined
        and box1_upper[0] >= box2_lower[0] - cutoff_combined
        and box1_lower[1] <= box2_upper[1] + cutoff_combined
        and box1_upper[1] >= box2_lower[1] - cutoff_combined
        and box1_lower[2] <= box2_upper[2] + cutoff_combined
        and box1_upper[2] >= box2_lower[2] - cutoff_combined
    )


def find_overlapping_pairs_np(
    box_lower: np.ndarray,
    box_upper: np.ndarray,
    cutoff: np.ndarray,
    collision_group: np.ndarray,
    shape_world: np.ndarray | None = None,
    shape_flags: np.ndarray | None = None,
):
    """
    Brute-force n^2 algorithm to find all overlapping bounding box pairs.
    Each box is axis-aligned, defined by min (lower) and max (upper) corners.
    Returns a list of (i, j) pairs with i < j, where boxes i and j overlap.

    Args:
        shape_flags: Optional array of shape flags. If provided, only geometries with
            COLLIDE_SHAPES flag set will participate in collision detection.
    """
    n = box_lower.shape[0]
    pairs = []
    for i in range(n):
        # Skip if shape_flags is provided and this geometry doesn't have COLLIDE_SHAPES flag
        if shape_flags is not None:
            if (shape_flags[i] & ShapeFlags.COLLIDE_SHAPES) == 0:
                continue

        for j in range(i + 1, n):
            # Skip if shape_flags is provided and this geometry doesn't have COLLIDE_SHAPES flag
            if shape_flags is not None:
                if (shape_flags[j] & ShapeFlags.COLLIDE_SHAPES) == 0:
                    continue

            # Check world and collision group compatibility
            if shape_world is not None:
                world_i = int(shape_world[i])
                world_j = int(shape_world[j])
                group_i = int(collision_group[i])
                group_j = int(collision_group[j])
                # Use the combined test function
                if not test_world_and_group_pair(world_i, world_j, group_i, group_j):
                    continue
            else:
                # No world information, just check collision groups
                if not test_group_pair(int(collision_group[i]), int(collision_group[j])):
                    continue

            # Check for overlap in all three axes
            cutoff_combined = max(cutoff[i], cutoff[j])
            if (
                box_lower[i, 0] <= box_upper[j, 0] + cutoff_combined
                and box_upper[i, 0] >= box_lower[j, 0] - cutoff_combined
                and box_lower[i, 1] <= box_upper[j, 1] + cutoff_combined
                and box_upper[i, 1] >= box_lower[j, 1] - cutoff_combined
                and box_lower[i, 2] <= box_upper[j, 2] + cutoff_combined
                and box_upper[i, 2] >= box_lower[j, 2] - cutoff_combined
            ):
                pairs.append((i, j))
    return pairs


@wp.kernel
def _test_advance_sap_chunk_base(
    chunk_base: int,
    chunk_stride: int,
    total_work_packages: int,
    result: wp.array[int],
):
    result[0] = _advance_sap_chunk_base(chunk_base, chunk_stride, total_work_packages)


@wp.kernel
def _test_nxn_pair_indices(indices: wp.array[int], sizes: wp.array[int], pairs: wp.array[wp.vec2i]):
    """Evaluate implicit pair indices without allocating a quadratic pair buffer."""
    tid = wp.tid()
    row, col = broad_phase_nxn._get_lower_triangular_indices(indices[tid], sizes[tid])
    pairs[tid] = wp.vec2i(row, col)


class TestBroadPhase(unittest.TestCase):
    def test_nxn_pair_index_boundaries(self):
        """Map row boundaries exactly through the largest int32 single-world pair count."""
        rng = np.random.default_rng(42)
        indices, sizes, expected = [], [], []
        for size in (0, 1, 2, 3, 8, 32, 257, 46341, 46342, 65536):
            rows = np.arange(size, dtype=np.int64)
            starts = rows * (2 * size - rows - 1) // 2
            total = size * (size - 1) // 2
            if total:
                selected = rows[np.unique([0, min(1, size - 1), size // 2, max(0, size - 2), size - 1])]
                boundaries = starts[selected]
                samples = np.unique(
                    np.clip(
                        np.concatenate((boundaries - 1, boundaries, boundaries + 1, rng.integers(total, size=100))),
                        0,
                        total - 1,
                    )
                )
                row = np.searchsorted(starts, samples, side="right") - 1
                indices.extend(samples)
                sizes.extend([size] * len(samples))
                expected.extend(zip(row, samples - starts[row] + row + 1, strict=True))
            indices.append(total)
            sizes.append(size)
            expected.append((-1, -1))
            indices.append(-1)
            sizes.append(size)
            expected.append((-1, size - 1))
        for device in get_test_devices():
            with self.subTest(device=device):
                broad_phase = BroadPhaseAllPairs(np.zeros(65536, dtype=np.int32), device=device)
                self.assertEqual(broad_phase.num_kernel_threads, 65536 * 65535 // 2)
                np.testing.assert_array_equal(broad_phase.world_cumsum_lower_tri.numpy(), [65536 * 65535 // 2] * 2)
                pairs = wp.empty(len(indices), dtype=wp.vec2i, device=device)
                wp.launch(
                    _test_nxn_pair_indices,
                    dim=len(indices),
                    inputs=[wp.array(indices, dtype=int, device=device), wp.array(sizes, dtype=int, device=device)],
                    outputs=[pairs],
                    device=device,
                )
                np.testing.assert_array_equal(pairs.numpy(), expected)

    def test_explicit_large_pair_warning_thresholds(self):
        """Warn only when total and busiest-world pair thresholds are met."""
        cases = (
            ("total boundary", [0, 3, 1], 3, False, 0),
            ("below world threshold", [0, 2, 2], 4, False, 1),
            ("multiworld prefix", [0, 2, 3], 4, False, 1),
            ("world boundary", [0, 3, 1], 4, True, 1),
            ("shared globals", [1, 2, 1], 4, True, 1),
            ("many small worlds", [0, *([1] * 8)], 8, False, 1),
            ("global-only worlds", [4, 0, 0, 0], 4, True, 1),
            ("no regular worlds", [4], 4, True, 0),
            ("single world", [0, 4], 4, True, 0),
        )
        with (
            mock.patch.object(broad_phase_nxn, "_EXPLICIT_LARGE_PAIR_COUNT_TOTAL", 3),
            mock.patch.object(broad_phase_nxn, "_EXPLICIT_LARGE_PAIR_COUNT_PER_WORLD", 3),
            mock.patch.object(wp, "launch", side_effect=AssertionError("Launched warning kernel")),
        ):
            for device in get_test_devices():
                for label, counts, count, expected_warning, expected_reads in cases:
                    worlds = wp.array(np.repeat(np.arange(len(counts)) - 1, 2), dtype=wp.int32, device=device)
                    pairs = np.repeat(np.arange(0, 2 * len(counts), 2)[:, None] + [0, 1], counts, axis=0)
                    pairs = wp.array(pairs, dtype=wp.vec2i, device=device)[:count]
                    with (
                        self.subTest(device=device, case=label),
                        mock.patch.object(pairs, "numpy", wraps=pairs.numpy) as read_pairs,
                        mock.patch.object(worlds, "numpy", wraps=worlds.numpy) as read_worlds,
                        warnings.catch_warnings(record=True) as caught,
                    ):
                        warnings.filterwarnings("always", message="Explicit broad phase tests", category=NewtonWarning)
                        BroadPhaseExplicit()._warn_large_pair_count(pairs, worlds, len(counts) - 1)
                    self.assertEqual(len(caught), int(expected_warning))
                    self.assertEqual(read_pairs.call_count, expected_reads)
                    self.assertEqual(read_worlds.call_count, expected_reads)
                    if caught:
                        self.assertIn("broad_phase='sap'", str(caught[0].message))

    def test_sap_auto_sort_selection(self):
        """Select bounded tile sizes for small and medium-sized SAP worlds."""
        cases = (
            (0, "segmented", 0),
            (1, "tile", 32),
            (32, "tile", 32),
            (33, "tile", 64),
            (63, "tile", 64),
            (64, "tile", 128),
            (129, "tile", 256),
            (257, "tile", 512),
            (513, "segmented", 513),
        )
        for shape_count, expected_sort, expected_stride in cases:
            with self.subTest(shape_count=shape_count):
                shape_world = np.zeros(shape_count, dtype=np.int32)
                broad_phase = BroadPhaseSAP(shape_world, sort_type="auto", device="cpu")
                self.assertEqual(broad_phase.sort_type, expected_sort)
                self.assertEqual(broad_phase.max_shapes_per_world, expected_stride)

    def test_sap_single_segment_uses_radix_sort(self):
        """Use ordinary radix sorting when SAP has exactly one segment."""
        shape_count = 4
        device = wp.get_device()
        shape_lower = wp.full(shape_count, wp.vec3(-1.0), dtype=wp.vec3, device=device)
        shape_upper = wp.full(shape_count, wp.vec3(1.0), dtype=wp.vec3, device=device)
        # Exercise the all-global fast path's group filtering as well as sorting.
        shape_group = wp.array([0, 1, 1, -1], dtype=wp.int32, device=device)
        shape_world = wp.full(shape_count, -1, dtype=wp.int32, device=device)
        candidate_pair = wp.zeros(6, dtype=wp.vec2i, device=device)
        candidate_pair_count = wp.zeros(1, dtype=wp.int32, device=device)

        broad_phase = BroadPhaseSAP(shape_world, sort_type="segmented", device=device)
        self.assertEqual(broad_phase.world_count, 1)
        self.assertTrue(broad_phase._single_segment_identity_map)

        filtered_flags = np.full(shape_count, ShapeFlags.COLLIDE_SHAPES, dtype=np.int32)
        filtered_flags[1] = 0
        filtered_broad_phase = BroadPhaseSAP(
            shape_world,
            shape_flags=filtered_flags,
            sort_type="segmented",
            device=device,
        )
        self.assertFalse(filtered_broad_phase._single_segment_identity_map)

        with mock.patch.object(wp.utils, "radix_sort_pairs", wraps=wp.utils.radix_sort_pairs) as radix_sort:
            broad_phase.launch(
                shape_lower,
                shape_upper,
                None,
                shape_group,
                shape_world,
                shape_count,
                candidate_pair,
                candidate_pair_count,
                device=device,
            )

        radix_sort.assert_called_once_with(
            keys=broad_phase.sap_projection_lower,
            values=broad_phase.sap_sort_index,
            count=broad_phase.max_shapes_per_world,
        )
        self.assertEqual(int(candidate_pair_count.numpy()[0]), 3)

    def test_sap_broadphase_auto(self):
        """Generate correct SAP pairs with automatic sort selection."""
        for shape_count in (2, 31, 33, 63, 70, 150, 300):
            with self.subTest(shape_count=shape_count):
                self._test_sap_broadphase_impl("auto", shape_count)

    def test_sap_broadphase_multiple_worlds_auto(self):
        """Generate correct automatic-SAP pairs across multiple worlds."""
        self._test_sap_broadphase_multiple_worlds_impl("auto")

    def test_sap_broadphase_with_shape_flags_auto(self):
        """Respect shape flags when automatic SAP sorting is enabled."""
        self._test_sap_broadphase_with_shape_flags_impl("auto")

    def test_sap_edge_cases_auto(self):
        """Handle SAP edge cases with automatic sort selection."""
        self._test_sap_edge_cases_impl("auto")

    def test_sap_chunk_advance_avoids_int32_overflow(self):
        """Stop dense SAP scheduling before the next chunk would overflow."""
        result = wp.empty(1, dtype=wp.int32)
        wp.launch(
            _test_advance_sap_chunk_base,
            dim=1,
            inputs=[2_147_418_112, 262_140, 2_147_483_647, result],
        )
        self.assertEqual(int(result.numpy()[0]), 2_147_483_647)

        wp.launch(_test_advance_sap_chunk_base, dim=1, inputs=[100, 16, 1_000, result])
        self.assertEqual(int(result.numpy()[0]), 116)

    def test_public_launch_previous_positional_layout(self):
        """Preserve the previous positional layout of public launch methods."""
        device = wp.get_device()
        shape_lower = wp.array([wp.vec3(-1.0), wp.vec3(-0.5)], dtype=wp.vec3, device=device)
        shape_upper = wp.array([wp.vec3(1.0), wp.vec3(0.5)], dtype=wp.vec3, device=device)
        shape_group = wp.ones(2, dtype=wp.int32, device=device)
        shape_world = wp.zeros(2, dtype=wp.int32, device=device)
        candidate_pair = wp.zeros(1, dtype=wp.vec2i, device=device)
        candidate_pair_count = wp.zeros(1, dtype=wp.int32, device=device)

        BroadPhaseAllPairs(shape_world, device=device).launch(
            shape_lower,
            shape_upper,
            None,
            shape_group,
            shape_world,
            2,
            candidate_pair,
            candidate_pair_count,
            device,
            None,
            None,
            False,
        )
        self.assertEqual(int(candidate_pair_count.numpy()[0]), 1)

        explicit_pairs = wp.array([(0, 1)], dtype=wp.vec2i, device=device)
        BroadPhaseExplicit().launch(
            shape_lower,
            shape_upper,
            None,
            explicit_pairs,
            1,
            candidate_pair,
            candidate_pair_count,
            device,
            False,
        )
        self.assertEqual(int(candidate_pair_count.numpy()[0]), 1)

        BroadPhaseSAP(shape_world, device=device).launch(
            shape_lower,
            shape_upper,
            None,
            shape_group,
            shape_world,
            2,
            candidate_pair,
            candidate_pair_count,
            device,
            None,
            None,
            False,
        )
        self.assertEqual(int(candidate_pair_count.numpy()[0]), 1)

    def test_sap_adaptive_chunk_enumerates_dense_pairs(self):
        """Verify adaptive SAP chunks enumerate every overlapping pair exactly once."""
        device = wp.get_device()
        shape_count = 32
        expected_count = shape_count * (shape_count - 1) // 2
        shape_lower = wp.full(shape_count, wp.vec3(-1.0), dtype=wp.vec3, device=device)
        shape_upper = wp.full(shape_count, wp.vec3(1.0), dtype=wp.vec3, device=device)
        shape_group = wp.ones(shape_count, dtype=wp.int32, device=device)
        shape_world = wp.zeros(shape_count, dtype=wp.int32, device=device)
        candidate_pair = wp.zeros(expected_count, dtype=wp.vec2i, device=device)
        candidate_pair_count = wp.zeros(1, dtype=wp.int32, device=device)

        BroadPhaseSAP(shape_world, device=device).launch(
            shape_lower,
            shape_upper,
            None,
            shape_group,
            shape_world,
            shape_count,
            candidate_pair,
            candidate_pair_count,
            device=device,
        )

        count = int(candidate_pair_count.numpy()[0])
        self.assertEqual(count, expected_count)
        actual_pairs = {tuple(pair) for pair in candidate_pair.numpy()[:count]}
        expected_pairs = {(i, j) for i in range(shape_count) for j in range(i + 1, shape_count)}
        self.assertEqual(actual_pairs, expected_pairs)

    def test_swept_aabb_requires_simultaneous_overlap(self):
        """Reject swept-union overlap unless both AABBs overlap at one time."""
        device = wp.get_device()
        shape_world = wp.zeros(2, dtype=wp.int32, device=device)
        shape_group = wp.ones(2, dtype=wp.int32, device=device)
        explicit_pairs = wp.array([(0, 1)], dtype=wp.vec2i, device=device)
        candidate_pair = wp.zeros(1, dtype=wp.vec2i, device=device)
        candidate_pair_count = wp.zeros(1, dtype=wp.int32, device=device)

        def launch(broad_phase, lower, upper, displacement):
            if isinstance(broad_phase, BroadPhaseExplicit):
                broad_phase.launch(
                    lower,
                    upper,
                    None,
                    explicit_pairs,
                    1,
                    candidate_pair,
                    candidate_pair_count,
                    device,
                    shape_displacement=displacement,
                )
            else:
                broad_phase.launch(
                    lower,
                    upper,
                    None,
                    shape_group,
                    shape_world,
                    2,
                    candidate_pair,
                    candidate_pair_count,
                    device,
                    shape_displacement=displacement,
                )
            return int(candidate_pair_count.numpy()[0])

        half_extent = wp.vec3(0.1)
        lower = wp.array([-half_extent, wp.vec3(0.5, -0.5, 0.0) - half_extent], dtype=wp.vec3, device=device)
        upper = wp.array([half_extent, wp.vec3(0.5, -0.5, 0.0) + half_extent], dtype=wp.vec3, device=device)
        same_displacement = wp.array([wp.vec3(1.0, 1.0, 0.0), wp.vec3(1.0, 1.0, 0.0)], dtype=wp.vec3, device=device)
        union_lower = wp.array([-half_extent, wp.vec3(0.5, -0.5, 0.0) - half_extent], dtype=wp.vec3, device=device)
        union_upper = wp.array(
            [wp.vec3(1.0, 1.0, 0.0) + half_extent, wp.vec3(1.5, 0.5, 0.0) + half_extent],
            dtype=wp.vec3,
            device=device,
        )

        colliding_lower = wp.array([-half_extent, wp.vec3(1.0, 1.0, 0.0) - half_extent], dtype=wp.vec3, device=device)
        colliding_upper = wp.array([half_extent, wp.vec3(1.0, 1.0, 0.0) + half_extent], dtype=wp.vec3, device=device)
        colliding_displacement = wp.array([wp.vec3(1.0, 1.0, 0.0), wp.vec3(0.0)], dtype=wp.vec3, device=device)

        for broad_phase in (
            BroadPhaseAllPairs(shape_world, device=device),
            BroadPhaseSAP(shape_world, device=device),
            BroadPhaseExplicit(),
        ):
            with self.subTest(broad_phase=type(broad_phase).__name__):
                self.assertEqual(launch(broad_phase, union_lower, union_upper, None), 1)
                self.assertEqual(launch(broad_phase, lower, upper, same_displacement), 0)
                self.assertEqual(launch(broad_phase, colliding_lower, colliding_upper, colliding_displacement), 1)

        fast_lower = wp.array([-half_extent, wp.vec3(0.4, 0.0, 0.0) - half_extent], dtype=wp.vec3, device=device)
        fast_upper = wp.array([half_extent, wp.vec3(0.4, 0.0, 0.0) + half_extent], dtype=wp.vec3, device=device)
        fast_displacement = wp.array([wp.vec3(2.0, 0.0, 0.0), wp.vec3(1.7, 0.0, 0.0)], dtype=wp.vec3, device=device)
        BroadPhaseSAP(shape_world, device=device).launch(
            fast_lower,
            fast_upper,
            None,
            shape_group,
            shape_world,
            2,
            candidate_pair,
            candidate_pair_count,
            device,
            shape_displacement=fast_displacement,
            sort_axis_displacement_limit=0.25,
        )
        self.assertEqual(int(candidate_pair_count.numpy()[0]), 1)

    def test_shape_displacement_length_validation(self):
        """Reject shape displacements whose length differs from the shape bounds."""
        device = wp.get_device()
        shape_lower = wp.zeros(2, dtype=wp.vec3, device=device)
        shape_upper = wp.ones(2, dtype=wp.vec3, device=device)
        shape_group = wp.ones(2, dtype=wp.int32, device=device)
        shape_world = wp.zeros(2, dtype=wp.int32, device=device)
        shape_displacement = wp.zeros(1, dtype=wp.vec3, device=device)
        candidate_pair = wp.zeros(1, dtype=wp.vec2i, device=device)
        candidate_pair_count = wp.zeros(1, dtype=wp.int32, device=device)

        for broad_phase in (
            BroadPhaseAllPairs(shape_world, device=device),
            BroadPhaseSAP(shape_world, device=device),
        ):
            with (
                self.subTest(broad_phase=type(broad_phase).__name__),
                self.assertRaisesRegex(ValueError, "shape_displacement length must match"),
            ):
                broad_phase.launch(
                    shape_lower,
                    shape_upper,
                    None,
                    shape_group,
                    shape_world,
                    2,
                    candidate_pair,
                    candidate_pair_count,
                    device,
                    shape_displacement=shape_displacement,
                )

        with self.assertRaisesRegex(ValueError, "shape_displacement length must match"):
            BroadPhaseExplicit().launch(
                shape_lower,
                shape_upper,
                None,
                wp.array([(0, 1)], dtype=wp.vec2i, device=device),
                1,
                candidate_pair,
                candidate_pair_count,
                device,
                shape_displacement=shape_displacement,
            )

        BroadPhaseSAP(shape_world, device=device).launch(
            shape_lower,
            shape_upper,
            None,
            shape_group,
            shape_world,
            1,
            candidate_pair,
            candidate_pair_count,
            device,
        )
        self.assertEqual(int(candidate_pair_count.numpy()[0]), 0)

    def test_nxn_broadphase(self):
        verbose = False

        # Create random bounding boxes in min-max format
        ngeom = 30

        # Generate random centers and sizes using the new Generator API
        rng = np.random.Generator(np.random.PCG64(42))

        centers = rng.random((ngeom, 3)) * 3.0
        sizes = rng.random((ngeom, 3)) * 2.0  # box half-extent up to 1.0 in each direction
        geom_bounding_box_lower = centers - sizes
        geom_bounding_box_upper = centers + sizes

        np_geom_cutoff = np.zeros(ngeom, dtype=np.float32)
        num_groups = 5  # The zero group does not need to be counted
        np_collision_group = rng.integers(1, num_groups + 1, size=ngeom, dtype=np.int32)

        # Overwrite n random elements with -1
        minus_one_count = int(sqrt(ngeom))  # Number of elements to overwrite with -1
        random_indices = rng.choice(ngeom, size=minus_one_count, replace=False)
        np_collision_group[random_indices] = -1

        pairs_np = find_overlapping_pairs_np(
            geom_bounding_box_lower, geom_bounding_box_upper, np_geom_cutoff, np_collision_group
        )

        if verbose:
            print("Numpy contact pairs:")
            for i, pair in enumerate(pairs_np):
                body_a, body_b = pair
                group_a = np_collision_group[body_a]
                group_b = np_collision_group[body_b]
                print(f"  Pair {i}: bodies ({body_a}, {body_b}) with collision groups ({group_a}, {group_b})")

        # The number of elements in the lower triangular part of an n x n matrix (excluding the diagonal)
        # is given by n * (n - 1) // 2
        num_lower_tri_elements = ngeom * (ngeom - 1) // 2

        geom_lower = wp.array(geom_bounding_box_lower, dtype=wp.vec3)
        geom_upper = wp.array(geom_bounding_box_upper, dtype=wp.vec3)
        geom_cutoff = wp.array(np_geom_cutoff)
        collision_group = wp.array(np_collision_group)
        candidate_pair_count = wp.array(
            [
                0,
            ],
            dtype=wp.int32,
        )
        max_candidate_pair = num_lower_tri_elements
        candidate_pair = wp.array(np.zeros((max_candidate_pair, 2), dtype=wp.int32), dtype=wp.vec2i)

        # Create shape world array with all shapes in global world (-1)
        shape_world = wp.array(np.full(ngeom, 0, dtype=np.int32), dtype=wp.int32)

        # Initialize BroadPhaseAllPairs with shape_world (which represents world grouping)
        nxn_broadphase = BroadPhaseAllPairs(shape_world)

        nxn_broadphase.launch(
            geom_lower,
            geom_upper,
            geom_cutoff,
            collision_group,
            shape_world,
            ngeom,
            candidate_pair,
            candidate_pair_count,
        )

        pairs_wp = candidate_pair.numpy()
        candidate_pair_count = candidate_pair_count.numpy()[0]

        if verbose:
            print("Warp contact pairs:")
            for i in range(candidate_pair_count):
                pair = pairs_wp[i]
                body_a, body_b = pair[0], pair[1]
                group_a = np_collision_group[body_a]
                group_b = np_collision_group[body_b]
                print(f"  Pair {i}: bodies ({body_a}, {body_b}) with collision groups ({group_a}, {group_b})")

            print("Checking if bounding boxes actually overlap:")
            for i in range(candidate_pair_count):
                pair = pairs_wp[i]
                body_a, body_b = pair[0], pair[1]

                # Get bounding boxes for both bodies
                box_a_lower = geom_bounding_box_lower[body_a]
                box_a_upper = geom_bounding_box_upper[body_a]
                box_b_lower = geom_bounding_box_lower[body_b]
                box_b_upper = geom_bounding_box_upper[body_b]

                # Get cutoffs for both bodies
                cutoff_a = np_geom_cutoff[body_a]
                cutoff_b = np_geom_cutoff[body_b]

                # Check overlap using the function
                overlap = check_aabb_overlap_host(
                    wp.vec3(box_a_lower[0], box_a_lower[1], box_a_lower[2]),
                    wp.vec3(box_a_upper[0], box_a_upper[1], box_a_upper[2]),
                    cutoff_a,
                    wp.vec3(box_b_lower[0], box_b_lower[1], box_b_lower[2]),
                    wp.vec3(box_b_upper[0], box_b_upper[1], box_b_upper[2]),
                    cutoff_b,
                )

                print(f"  Pair {i}: bodies ({body_a}, {body_b}) - overlap: {overlap}")

        if len(pairs_np) != candidate_pair_count:
            print(f"len(pairs_np)={len(pairs_np)}, candidate_pair_count={candidate_pair_count}")
            assert len(pairs_np) == candidate_pair_count

        # Ensure every element in pairs_wp is also present in pairs_np
        pairs_np_set = {tuple(pair) for pair in pairs_np}
        for pair in pairs_wp[:candidate_pair_count]:
            assert tuple(pair) in pairs_np_set, f"Pair {tuple(pair)} from Warp not found in numpy pairs"

        if verbose:
            print(len(pairs_np))

    def test_nxn_broadphase_multiple_worlds(self):
        """Test NxN broad phase with objects in different worlds and mixed collision groups."""
        verbose = False

        # Create random bounding boxes in min-max format
        ngeom = 50
        world_count = 4  # We'll distribute objects across 4 different worlds

        # Generate random centers and sizes using the new Generator API
        rng = np.random.Generator(np.random.PCG64(123))

        centers = rng.random((ngeom, 3)) * 5.0
        sizes = rng.random((ngeom, 3)) * 1.5  # box half-extent up to 1.5 in each direction
        geom_bounding_box_lower = centers - sizes
        geom_bounding_box_upper = centers + sizes

        np_geom_cutoff = np.zeros(ngeom, dtype=np.float32)

        # Randomly assign collision groups (including some negative ones for shared entities)
        num_groups = 5
        np_collision_group = rng.integers(1, num_groups + 1, size=ngeom, dtype=np.int32)

        # Make some entities shared (negative collision group)
        num_shared = int(sqrt(ngeom))
        shared_indices = rng.choice(ngeom, size=num_shared, replace=False)
        np_collision_group[shared_indices] = -1

        # Randomly distribute objects across worlds
        # Some objects in specific worlds (0, 1, 2, 3), some global (-1)
        np_shape_world = rng.integers(0, world_count, size=ngeom, dtype=np.int32)

        # Make some entities global (world -1) - they should collide with all worlds
        num_global = max(3, ngeom // 10)
        global_indices = rng.choice(ngeom, size=num_global, replace=False)
        np_shape_world[global_indices] = -1

        if verbose:
            print("\nTest setup:")
            print(f"  Total geometries: {ngeom}")
            print(f"  Number of worlds: {world_count}")
            print(f"  Global entities (world=-1): {num_global}")
            print(f"  Shared entities (group=-1): {num_shared}")
            print("\nWorld distribution:")
            for world_id in range(-1, world_count):
                count = np.sum(np_shape_world == world_id)
                print(f"  World {world_id}: {count} objects")
            print("\nCollision group distribution:")
            unique_groups = np.unique(np_collision_group)
            for group in unique_groups:
                count = np.sum(np_collision_group == group)
                print(f"  Group {group}: {count} objects")

        # Compute expected pairs using numpy
        pairs_np = find_overlapping_pairs_np(
            geom_bounding_box_lower, geom_bounding_box_upper, np_geom_cutoff, np_collision_group, np_shape_world
        )

        if verbose:
            print(f"\nExpected number of pairs: {len(pairs_np)}")
            if len(pairs_np) <= 20:
                print("Numpy contact pairs:")
                for i, pair in enumerate(pairs_np):
                    body_a, body_b = pair
                    world_a, world_b = np_shape_world[body_a], np_shape_world[body_b]
                    group_a, group_b = np_collision_group[body_a], np_collision_group[body_b]
                    print(
                        f"  Pair {i}: bodies ({body_a}, {body_b}) "
                        f"worlds ({world_a}, {world_b}) groups ({group_a}, {group_b})"
                    )

        # Setup Warp arrays
        num_lower_tri_elements = ngeom * (ngeom - 1) // 2
        max_candidate_pair = num_lower_tri_elements

        geom_lower = wp.array(geom_bounding_box_lower, dtype=wp.vec3)
        geom_upper = wp.array(geom_bounding_box_upper, dtype=wp.vec3)
        geom_cutoff = wp.array(np_geom_cutoff)
        collision_group = wp.array(np_collision_group)
        shape_world = wp.array(np_shape_world, dtype=wp.int32)
        candidate_pair_count = wp.array([0], dtype=wp.int32)
        candidate_pair = wp.array(np.zeros((max_candidate_pair, 2), dtype=wp.int32), dtype=wp.vec2i)

        # Initialize BroadPhaseAllPairs with shape_world for precomputation
        nxn_broadphase = BroadPhaseAllPairs(shape_world)

        if verbose:
            print("\nPrecomputed world map info:")
            print(f"  Number of kernel threads: {nxn_broadphase.num_kernel_threads}")
            print(f"  World slice ends: {nxn_broadphase.world_slice_ends.numpy()}")
            print(f"  World cumsum lower tri: {nxn_broadphase.world_cumsum_lower_tri.numpy()}")

        # Launch broad phase
        nxn_broadphase.launch(
            geom_lower,
            geom_upper,
            geom_cutoff,
            collision_group,
            shape_world,
            ngeom,
            candidate_pair,
            candidate_pair_count,
        )

        # Get results
        pairs_wp = candidate_pair.numpy()
        num_candidate_pair_result = candidate_pair_count.numpy()[0]

        if verbose:
            print(f"\nWarp found {num_candidate_pair_result} pairs")
            if num_candidate_pair_result <= 20:
                print("Warp contact pairs:")
                for i in range(num_candidate_pair_result):
                    pair = pairs_wp[i]
                    body_a, body_b = pair[0], pair[1]
                    world_a, world_b = np_shape_world[body_a], np_shape_world[body_b]
                    group_a, group_b = np_collision_group[body_a], np_collision_group[body_b]
                    print(
                        f"  Pair {i}: bodies ({body_a}, {body_b}) "
                        f"worlds ({world_a}, {world_b}) groups ({group_a}, {group_b})"
                    )

        # Verify results
        if len(pairs_np) != num_candidate_pair_result:
            print(f"\nMismatch: Expected {len(pairs_np)} pairs, got {num_candidate_pair_result}")

            # Show missing or extra pairs for debugging
            pairs_np_set = {tuple(pair) for pair in pairs_np}
            pairs_wp_set = {tuple(pairs_wp[i]) for i in range(num_candidate_pair_result)}

            missing = pairs_np_set - pairs_wp_set
            extra = pairs_wp_set - pairs_np_set

            if missing:
                print(f"Missing pairs ({len(missing)}):")
                for pair in list(missing)[:10]:
                    body_a, body_b = pair
                    world_a, world_b = np_shape_world[body_a], np_shape_world[body_b]
                    group_a, group_b = np_collision_group[body_a], np_collision_group[body_b]
                    print(f"  {pair}: worlds ({world_a}, {world_b}) groups ({group_a}, {group_b})")

            if extra:
                print(f"Extra pairs ({len(extra)}):")
                for pair in list(extra)[:10]:
                    body_a, body_b = pair
                    world_a, world_b = np_shape_world[body_a], np_shape_world[body_b]
                    group_a, group_b = np_collision_group[body_a], np_collision_group[body_b]
                    print(f"  {pair}: worlds ({world_a}, {world_b}) groups ({group_a}, {group_b})")

            assert len(pairs_np) == num_candidate_pair_result

        # Ensure every element in pairs_wp is also present in pairs_np
        pairs_np_set = {tuple(pair) for pair in pairs_np}
        for pair in pairs_wp[:num_candidate_pair_result]:
            assert tuple(pair) in pairs_np_set, f"Pair {tuple(pair)} from Warp not found in numpy pairs"

        if verbose:
            print(f"\nTest passed! All {len(pairs_np)} pairs matched.")

    def test_nxn_broadphase_with_shape_flags(self):
        """Test NxN broad phase with ShapeFlags filtering.

        This test verifies that:
        - Shapes without COLLIDE_SHAPES flag are correctly filtered out
        - Filtering works correctly with multiple worlds
        - num_regular_worlds is correctly computed after filtering (tests bug fix)
        - Edge case: filtering out all positive-world geometries but keeping -1 (tests critical bug)
        """
        verbose = False

        # Create random bounding boxes in min-max format
        ngeom = 50
        world_count = 4  # We'll distribute objects across 4 different worlds

        # Generate random centers and sizes using the new Generator API
        rng = np.random.Generator(np.random.PCG64(456))

        centers = rng.random((ngeom, 3)) * 5.0
        sizes = rng.random((ngeom, 3)) * 1.5  # box half-extent up to 1.5 in each direction
        geom_bounding_box_lower = centers - sizes
        geom_bounding_box_upper = centers + sizes

        np_geom_cutoff = np.zeros(ngeom, dtype=np.float32)

        # Randomly assign collision groups
        num_groups = 5
        np_collision_group = rng.integers(1, num_groups + 1, size=ngeom, dtype=np.int32)

        # Make some entities shared (negative collision group)
        num_shared = int(sqrt(ngeom))
        shared_indices = rng.choice(ngeom, size=num_shared, replace=False)
        np_collision_group[shared_indices] = -1

        # Randomly distribute objects across worlds
        # Some objects in specific worlds (0, 1, 2, 3), some global (-1)
        np_shape_world = rng.integers(0, world_count, size=ngeom, dtype=np.int32)

        # Make some entities global (world -1) - they should collide with all worlds
        num_global = max(3, ngeom // 10)
        global_indices = rng.choice(ngeom, size=num_global, replace=False)
        np_shape_world[global_indices] = -1

        # Create shape flags: some geometries will be visual-only (no COLLIDE_SHAPES flag)
        # Critical test case: filter out all positive-world geometries, keep only -1
        np_shape_flags = np.zeros(ngeom, dtype=np.int32)

        # Assign flags: ~70% will have COLLIDE_SHAPES flag, 30% will be visual-only
        colliding_indices = rng.choice(ngeom, size=int(0.7 * ngeom), replace=False)
        np_shape_flags[colliding_indices] = ShapeFlags.COLLIDE_SHAPES

        # Also set VISIBLE flag on some for completeness
        visible_indices = rng.choice(ngeom, size=int(0.8 * ngeom), replace=False)
        np_shape_flags[visible_indices] |= ShapeFlags.VISIBLE

        # CRITICAL TEST CASE: Filter out all geometries from world 0 (but keep world -1)
        # This tests the bug where num_regular_worlds was computed incorrectly after filtering
        world_0_mask = np_shape_world == 0
        np_shape_flags[world_0_mask] = 0  # Remove COLLIDE_SHAPES from all world 0 geometries

        # Count how many colliding geometries remain after filtering
        colliding_mask = (np_shape_flags & ShapeFlags.COLLIDE_SHAPES) != 0
        num_colliding = np.sum(colliding_mask)

        if verbose:
            print("\nTest setup with ShapeFlags:")
            print(f"  Total geometries: {ngeom}")
            print(f"  Geometries with COLLIDE_SHAPES flag: {num_colliding}")
            print(f"  Geometries filtered out: {ngeom - num_colliding}")
            print(f"  Number of worlds: {world_count}")
            print(f"  Global entities (world=-1): {num_global}")
            print(f"  World 0 geometries filtered: {np.sum(world_0_mask)}")
            print("\nWorld distribution (after filtering):")
            for world_id in range(-1, world_count):
                count_total = np.sum(np_shape_world == world_id)
                count_colliding = np.sum((np_shape_world == world_id) & colliding_mask)
                print(f"  World {world_id}: {count_total} total, {count_colliding} colliding")

        # Compute expected pairs using numpy (with shape flags filtering)
        pairs_np = find_overlapping_pairs_np(
            geom_bounding_box_lower,
            geom_bounding_box_upper,
            np_geom_cutoff,
            np_collision_group,
            np_shape_world,
            np_shape_flags,
        )

        if verbose:
            print(f"\nExpected number of pairs (after flag filtering): {len(pairs_np)}")
            if len(pairs_np) <= 20:
                print("Numpy contact pairs:")
                for i, pair in enumerate(pairs_np):
                    body_a, body_b = pair
                    world_a, world_b = np_shape_world[body_a], np_shape_world[body_b]
                    group_a, group_b = np_collision_group[body_a], np_collision_group[body_b]
                    flag_a, flag_b = np_shape_flags[body_a], np_shape_flags[body_b]
                    print(
                        f"  Pair {i}: bodies ({body_a}, {body_b}) "
                        f"worlds ({world_a}, {world_b}) groups ({group_a}, {group_b}) "
                        f"flags ({flag_a}, {flag_b})"
                    )

        # Setup Warp arrays
        num_lower_tri_elements = ngeom * (ngeom - 1) // 2
        max_candidate_pair = num_lower_tri_elements

        geom_lower = wp.array(geom_bounding_box_lower, dtype=wp.vec3)
        geom_upper = wp.array(geom_bounding_box_upper, dtype=wp.vec3)
        geom_cutoff = wp.array(np_geom_cutoff)
        collision_group = wp.array(np_collision_group)
        shape_world = wp.array(np_shape_world, dtype=wp.int32)
        shape_flags = wp.array(np_shape_flags, dtype=wp.int32)
        candidate_pair_count = wp.array([0], dtype=wp.int32)
        candidate_pair = wp.array(np.zeros((max_candidate_pair, 2), dtype=wp.int32), dtype=wp.vec2i)

        # Initialize BroadPhaseAllPairs with shape_world AND shape_flags
        nxn_broadphase = BroadPhaseAllPairs(shape_world, shape_flags=shape_flags)

        if verbose:
            print("\nPrecomputed world map info (with flags):")
            print(f"  Number of kernel threads: {nxn_broadphase.num_kernel_threads}")
            print(f"  Number of regular worlds: {nxn_broadphase.num_regular_worlds}")
            print(f"  World slice ends: {nxn_broadphase.world_slice_ends.numpy()}")
            print(f"  World cumsum lower tri: {nxn_broadphase.world_cumsum_lower_tri.numpy()}")

        # Verify num_regular_worlds is correct after filtering
        # It should be the number of worlds that have at least one colliding geometry
        colliding_worlds = np.unique(np_shape_world[colliding_mask])
        colliding_worlds = colliding_worlds[colliding_worlds >= 0]  # Exclude -1
        expected_num_regular_worlds = len(colliding_worlds)

        self.assertEqual(
            nxn_broadphase.num_regular_worlds,
            expected_num_regular_worlds,
            f"num_regular_worlds mismatch: expected {expected_num_regular_worlds} "
            f"(after filtering), got {nxn_broadphase.num_regular_worlds}",
        )

        # Launch broad phase
        nxn_broadphase.launch(
            geom_lower,
            geom_upper,
            geom_cutoff,
            collision_group,
            shape_world,
            ngeom,
            candidate_pair,
            candidate_pair_count,
        )

        # Get results
        pairs_wp = candidate_pair.numpy()
        num_candidate_pair_result = candidate_pair_count.numpy()[0]

        if verbose:
            print(f"\nWarp found {num_candidate_pair_result} pairs")
            if num_candidate_pair_result <= 20:
                print("Warp contact pairs:")
                for i in range(num_candidate_pair_result):
                    pair = pairs_wp[i]
                    body_a, body_b = pair[0], pair[1]
                    world_a, world_b = np_shape_world[body_a], np_shape_world[body_b]
                    group_a, group_b = np_collision_group[body_a], np_collision_group[body_b]
                    flag_a, flag_b = np_shape_flags[body_a], np_shape_flags[body_b]
                    print(
                        f"  Pair {i}: bodies ({body_a}, {body_b}) "
                        f"worlds ({world_a}, {world_b}) groups ({group_a}, {group_b}) "
                        f"flags ({flag_a}, {flag_b})"
                    )

        # Verify results
        if len(pairs_np) != num_candidate_pair_result:
            print(f"\nMismatch: Expected {len(pairs_np)} pairs, got {num_candidate_pair_result}")

            # Show missing or extra pairs for debugging
            pairs_np_set = {tuple(pair) for pair in pairs_np}
            pairs_wp_set = {tuple(pairs_wp[i]) for i in range(num_candidate_pair_result)}

            missing = pairs_np_set - pairs_wp_set
            extra = pairs_wp_set - pairs_np_set

            if missing:
                print(f"Missing pairs ({len(missing)}):")
                for pair in list(missing)[:10]:
                    body_a, body_b = pair
                    world_a, world_b = np_shape_world[body_a], np_shape_world[body_b]
                    group_a, group_b = np_collision_group[body_a], np_collision_group[body_b]
                    flag_a, flag_b = np_shape_flags[body_a], np_shape_flags[body_b]
                    print(
                        f"  {pair}: worlds ({world_a}, {world_b}) groups ({group_a}, {group_b}) "
                        f"flags ({flag_a}, {flag_b})"
                    )

            if extra:
                print(f"Extra pairs ({len(extra)}):")
                for pair in list(extra)[:10]:
                    body_a, body_b = pair
                    world_a, world_b = np_shape_world[body_a], np_shape_world[body_b]
                    group_a, group_b = np_collision_group[body_a], np_collision_group[body_b]
                    flag_a, flag_b = np_shape_flags[body_a], np_shape_flags[body_b]
                    print(
                        f"  {pair}: worlds ({world_a}, {world_b}) groups ({group_a}, {group_b}) "
                        f"flags ({flag_a}, {flag_b})"
                    )

            assert len(pairs_np) == num_candidate_pair_result

        # Ensure every element in pairs_wp is also present in pairs_np
        pairs_np_set = {tuple(pair) for pair in pairs_np}
        for pair in pairs_wp[:num_candidate_pair_result]:
            assert tuple(pair) in pairs_np_set, f"Pair {tuple(pair)} from Warp not found in numpy pairs"

        # Verify that no pairs contain filtered-out geometries
        for pair in pairs_wp[:num_candidate_pair_result]:
            body_a, body_b = pair[0], pair[1]
            flag_a = np_shape_flags[body_a]
            flag_b = np_shape_flags[body_b]
            assert (flag_a & ShapeFlags.COLLIDE_SHAPES) != 0, (
                f"Pair contains filtered geometry {body_a} with flag {flag_a}"
            )
            assert (flag_b & ShapeFlags.COLLIDE_SHAPES) != 0, (
                f"Pair contains filtered geometry {body_b} with flag {flag_b}"
            )

        if verbose:
            print(f"\nTest passed! All {len(pairs_np)} pairs matched, no filtered geometries included.")

    def test_explicit_pairs_broadphase(self):
        verbose = False

        # Create random bounding boxes in min-max format
        ngeom = 30

        # Generate random centers and sizes using the new Generator API
        rng = np.random.Generator(np.random.PCG64(42))

        centers = rng.random((ngeom, 3)) * 3.0
        sizes = rng.random((ngeom, 3)) * 2.0  # box half-extent up to 1.0 in each direction
        geom_bounding_box_lower = centers - sizes
        geom_bounding_box_upper = centers + sizes

        np_geom_cutoff = np.zeros(ngeom, dtype=np.float32)

        # Create explicit pairs to check - we'll take a subset of all possible pairs
        # For example, check pairs (0,1), (1,2), (2,3), etc.
        num_pairs_to_check = ngeom - 1
        explicit_pairs = np.array([(i, i + 1) for i in range(num_pairs_to_check)], dtype=np.int32)

        # Get ground truth overlaps for these explicit pairs
        pairs_np = []
        for pair in explicit_pairs:
            body_a, body_b = pair[0], pair[1]

            # Get bounding boxes for both bodies
            box_a_lower = geom_bounding_box_lower[body_a]
            box_a_upper = geom_bounding_box_upper[body_a]
            box_b_lower = geom_bounding_box_lower[body_b]
            box_b_upper = geom_bounding_box_upper[body_b]

            # Get cutoffs for both bodies
            cutoff_a = np_geom_cutoff[body_a]
            cutoff_b = np_geom_cutoff[body_b]

            # Check overlap using the function
            if check_aabb_overlap_host(
                wp.vec3(box_a_lower[0], box_a_lower[1], box_a_lower[2]),
                wp.vec3(box_a_upper[0], box_a_upper[1], box_a_upper[2]),
                cutoff_a,
                wp.vec3(box_b_lower[0], box_b_lower[1], box_b_lower[2]),
                wp.vec3(box_b_upper[0], box_b_upper[1], box_b_upper[2]),
                cutoff_b,
            ):
                pairs_np.append(tuple(pair))

        if verbose:
            print("Numpy contact pairs:")
            for i, pair in enumerate(pairs_np):
                print(f"  Pair {i}: bodies {pair}")

        # Convert data to Warp arrays
        geom_lower = wp.array(geom_bounding_box_lower, dtype=wp.vec3)
        geom_upper = wp.array(geom_bounding_box_upper, dtype=wp.vec3)
        geom_cutoff = wp.array(np_geom_cutoff)
        explicit_pairs_wp = wp.array(explicit_pairs, dtype=wp.vec2i)
        candidate_pair_count = wp.array(
            [
                0,
            ],
            dtype=wp.int32,
        )
        max_candidate_pair = num_pairs_to_check
        candidate_pair = wp.array(np.zeros((max_candidate_pair, 2), dtype=np.int32), dtype=wp.vec2i)

        explicit_broadphase = BroadPhaseExplicit()

        explicit_broadphase.launch(
            geom_lower,
            geom_upper,
            geom_cutoff,
            explicit_pairs_wp,
            num_pairs_to_check,
            candidate_pair,
            candidate_pair_count,
        )

        pairs_wp = candidate_pair.numpy()
        candidate_pair_count = candidate_pair_count.numpy()[0]

        if verbose:
            print("Warp contact pairs:")
            for i in range(candidate_pair_count):
                pair = pairs_wp[i]
                print(f"  Pair {i}: bodies ({pair[0]}, {pair[1]})")

            print("Checking if bounding boxes actually overlap:")
            for i in range(candidate_pair_count):
                pair = pairs_wp[i]
                body_a, body_b = pair[0], pair[1]

                # Get bounding boxes for both bodies
                box_a_lower = geom_bounding_box_lower[body_a]
                box_a_upper = geom_bounding_box_upper[body_a]
                box_b_lower = geom_bounding_box_lower[body_b]
                box_b_upper = geom_bounding_box_upper[body_b]

                # Get cutoffs for both bodies
                cutoff_a = np_geom_cutoff[body_a]
                cutoff_b = np_geom_cutoff[body_b]

                # Check overlap using the function
                overlap = check_aabb_overlap_host(
                    wp.vec3(box_a_lower[0], box_a_lower[1], box_a_lower[2]),
                    wp.vec3(box_a_upper[0], box_a_upper[1], box_a_upper[2]),
                    cutoff_a,
                    wp.vec3(box_b_lower[0], box_b_lower[1], box_b_lower[2]),
                    wp.vec3(box_b_upper[0], box_b_upper[1], box_b_upper[2]),
                    cutoff_b,
                )

                print(f"  Pair {i}: bodies ({body_a}, {body_b}) - overlap: {overlap}")

        if len(pairs_np) != candidate_pair_count:
            print(f"len(pairs_np)={len(pairs_np)}, candidate_pair_count={candidate_pair_count}")
            assert len(pairs_np) == candidate_pair_count

        # Ensure every element in pairs_wp is also present in pairs_np
        pairs_np_set = {tuple(pair) for pair in pairs_np}
        for pair in pairs_wp[:candidate_pair_count]:
            assert tuple(pair) in pairs_np_set, f"Pair {tuple(pair)} from Warp not found in numpy pairs"

        if verbose:
            print(len(pairs_np))

    def _test_sap_broadphase_impl(self, sort_type, ngeom=30):
        verbose = False

        # Create random bounding boxes in min-max format

        # Generate random centers and sizes using the new Generator API
        rng = np.random.Generator(np.random.PCG64(42))

        centers = rng.random((ngeom, 3)) * 3.0
        sizes = rng.random((ngeom, 3)) * 2.0  # box half-extent up to 1.0 in each direction
        geom_bounding_box_lower = centers - sizes
        geom_bounding_box_upper = centers + sizes

        np_geom_cutoff = np.zeros(ngeom, dtype=np.float32)
        num_groups = 5  # The zero group does not need to be counted
        np_collision_group = rng.integers(1, num_groups + 1, size=ngeom, dtype=np.int32)

        # Overwrite n random elements with -1
        minus_one_count = int(sqrt(ngeom))  # Number of elements to overwrite with -1
        random_indices = rng.choice(ngeom, size=minus_one_count, replace=False)
        np_collision_group[random_indices] = -1

        pairs_np = find_overlapping_pairs_np(
            geom_bounding_box_lower, geom_bounding_box_upper, np_geom_cutoff, np_collision_group
        )

        if verbose:
            print("Numpy contact pairs:")
            for i, pair in enumerate(pairs_np):
                body_a, body_b = pair
                group_a = np_collision_group[body_a]
                group_b = np_collision_group[body_b]
                print(f"  Pair {i}: bodies ({body_a}, {body_b}) with collision groups ({group_a}, {group_b})")

        # The number of elements in the lower triangular part of an n x n matrix (excluding the diagonal)
        # is given by n * (n - 1) // 2
        num_lower_tri_elements = ngeom * (ngeom - 1) // 2

        geom_lower = wp.array(geom_bounding_box_lower, dtype=wp.vec3)
        geom_upper = wp.array(geom_bounding_box_upper, dtype=wp.vec3)
        geom_cutoff = wp.array(np_geom_cutoff)
        collision_group = wp.array(np_collision_group)
        candidate_pair_count = wp.array(
            [
                0,
            ],
            dtype=wp.int32,
        )
        max_candidate_pair = num_lower_tri_elements
        candidate_pair = wp.array(np.zeros((max_candidate_pair, 2), dtype=wp.int32), dtype=wp.vec2i)

        # Create shape world array with all shapes in world 0
        shape_world = wp.array(np.full(ngeom, 0, dtype=np.int32), dtype=wp.int32)

        # Initialize BroadPhaseSAP with shape_world for precomputation
        sap_broadphase = BroadPhaseSAP(shape_world, sort_type=sort_type)

        sap_broadphase.launch(
            geom_lower,
            geom_upper,
            geom_cutoff,
            collision_group,
            shape_world,
            ngeom,
            candidate_pair,
            candidate_pair_count,
        )

        pairs_wp = candidate_pair.numpy()
        candidate_pair_count = candidate_pair_count.numpy()[0]

        if verbose:
            print("Warp contact pairs:")
            for i in range(candidate_pair_count):
                pair = pairs_wp[i]
                body_a, body_b = pair[0], pair[1]
                group_a = np_collision_group[body_a]
                group_b = np_collision_group[body_b]
                print(f"  Pair {i}: bodies ({body_a}, {body_b}) with collision groups ({group_a}, {group_b})")

            print("Checking if bounding boxes actually overlap:")
            for i in range(candidate_pair_count):
                pair = pairs_wp[i]
                body_a, body_b = pair[0], pair[1]

                # Get bounding boxes for both bodies
                box_a_lower = geom_bounding_box_lower[body_a]
                box_a_upper = geom_bounding_box_upper[body_a]
                box_b_lower = geom_bounding_box_lower[body_b]
                box_b_upper = geom_bounding_box_upper[body_b]

                # Get cutoffs for both bodies
                cutoff_a = np_geom_cutoff[body_a]
                cutoff_b = np_geom_cutoff[body_b]

                # Check overlap using the function
                overlap = check_aabb_overlap_host(
                    wp.vec3(box_a_lower[0], box_a_lower[1], box_a_lower[2]),
                    wp.vec3(box_a_upper[0], box_a_upper[1], box_a_upper[2]),
                    cutoff_a,
                    wp.vec3(box_b_lower[0], box_b_lower[1], box_b_lower[2]),
                    wp.vec3(box_b_upper[0], box_b_upper[1], box_b_upper[2]),
                    cutoff_b,
                )

                print(f"  Pair {i}: bodies ({body_a}, {body_b}) - overlap: {overlap}")

        if len(pairs_np) != candidate_pair_count:
            print(f"len(pairs_np)={len(pairs_np)}, candidate_pair_count={candidate_pair_count}")
            # print("pairs_np:", pairs_np)
            # print("pairs_wp[:candidate_pair_count]:", pairs_wp[:candidate_pair_count])
            assert len(pairs_np) == candidate_pair_count

        # Ensure every element in pairs_wp is also present in pairs_np
        pairs_np_set = {tuple(pair) for pair in pairs_np}
        for pair in pairs_wp[:candidate_pair_count]:
            assert tuple(pair) in pairs_np_set, f"Pair {tuple(pair)} from Warp not found in numpy pairs"

        if verbose:
            print(len(pairs_np))

    def test_sap_broadphase_segmented(self):
        """Test SAP broad phase with segmented sort."""
        self._test_sap_broadphase_impl("segmented")

    def test_sap_broadphase_tile(self):
        """Test SAP broad phase with tile sort."""
        self._test_sap_broadphase_impl("tile")

    def _test_sap_broadphase_multiple_worlds_impl(self, sort_type):
        """Test SAP broad phase with objects in different worlds and mixed collision groups."""
        verbose = False

        # Create a scenario with multiple worlds
        ngeom = 40
        rng = np.random.Generator(np.random.PCG64(123))

        # Generate random centers and sizes
        centers = rng.random((ngeom, 3)) * 5.0
        sizes = rng.random((ngeom, 3)) * 1.5
        geom_bounding_box_lower = centers - sizes
        geom_bounding_box_upper = centers + sizes

        np_geom_cutoff = np.zeros(ngeom, dtype=np.float32)

        # Create a mix of world IDs: some in world 0, some in world 1, some in world 2, some shared (-1)
        np_shape_world = np.zeros(ngeom, dtype=np.int32)

        # Distribute geometries across worlds
        world_0_count = ngeom // 3
        world_1_count = ngeom // 3

        np_shape_world[:world_0_count] = 0
        np_shape_world[world_0_count : world_0_count + world_1_count] = 1
        np_shape_world[world_0_count + world_1_count :] = -1  # Shared entities

        # Shuffle to make it more realistic
        rng.shuffle(np_shape_world)

        # Create collision groups (positive values for filtering within worlds)
        num_groups = 4
        np_collision_group = rng.integers(1, num_groups + 1, size=ngeom, dtype=np.int32)

        # Make some collision groups negative (shared across all worlds)
        minus_one_count = int(sqrt(ngeom))
        random_indices = rng.choice(ngeom, size=minus_one_count, replace=False)
        np_collision_group[random_indices] = -1

        if verbose:
            print("\nGeometry world assignments:")
            for i in range(ngeom):
                print(f"  Geom {i}: world={np_shape_world[i]}, collision_group={np_collision_group[i]}")

        # Find expected pairs using numpy
        pairs_np = find_overlapping_pairs_np(
            geom_bounding_box_lower, geom_bounding_box_upper, np_geom_cutoff, np_collision_group, np_shape_world
        )

        if verbose:
            print(f"\nNumpy found {len(pairs_np)} pairs:")
            for i, pair in enumerate(pairs_np):
                body_a, body_b = pair
                world_a = np_shape_world[body_a]
                world_b = np_shape_world[body_b]
                group_a = np_collision_group[body_a]
                group_b = np_collision_group[body_b]
                print(
                    f"  Pair {i}: bodies ({body_a}, {body_b}) worlds ({world_a}, {world_b}) groups ({group_a}, {group_b})"
                )

        # Setup Warp arrays
        num_lower_tri_elements = ngeom * (ngeom - 1) // 2
        geom_lower = wp.array(geom_bounding_box_lower, dtype=wp.vec3)
        geom_upper = wp.array(geom_bounding_box_upper, dtype=wp.vec3)
        geom_cutoff = wp.array(np_geom_cutoff)
        collision_group = wp.array(np_collision_group)
        shape_world = wp.array(np_shape_world, dtype=wp.int32)
        candidate_pair_count = wp.array([0], dtype=wp.int32)
        max_candidate_pair = num_lower_tri_elements
        candidate_pair = wp.array(np.zeros((max_candidate_pair, 2), dtype=wp.int32), dtype=wp.vec2i)

        # Initialize and launch SAP broad phase
        sap_broadphase = BroadPhaseSAP(shape_world, sort_type=sort_type)
        sap_broadphase.launch(
            geom_lower,
            geom_upper,
            geom_cutoff,
            collision_group,
            shape_world,
            ngeom,
            candidate_pair,
            candidate_pair_count,
        )

        pairs_wp = candidate_pair.numpy()
        num_candidate_pair_val = candidate_pair_count.numpy()[0]

        if verbose:
            print(f"\nWarp found {num_candidate_pair_val} pairs:")
            for i in range(num_candidate_pair_val):
                pair = pairs_wp[i]
                body_a, body_b = pair[0], pair[1]
                world_a = np_shape_world[body_a]
                world_b = np_shape_world[body_b]
                group_a = np_collision_group[body_a]
                group_b = np_collision_group[body_b]
                print(
                    f"  Pair {i}: bodies ({body_a}, {body_b}) worlds ({world_a}, {world_b}) groups ({group_a}, {group_b})"
                )

        # Verify results
        if len(pairs_np) != num_candidate_pair_val:
            print(f"\nMismatch: numpy found {len(pairs_np)} pairs, Warp found {num_candidate_pair_val} pairs")

            # Show missing pairs
            pairs_np_set = {tuple(pair) for pair in pairs_np}
            pairs_wp_set = {tuple(pairs_wp[i]) for i in range(num_candidate_pair_val)}

            missing_in_warp = pairs_np_set - pairs_wp_set
            extra_in_warp = pairs_wp_set - pairs_np_set

            if missing_in_warp:
                print(f"\nPairs in numpy but not in Warp ({len(missing_in_warp)}):")
                for pair in list(missing_in_warp)[:10]:  # Show first 10
                    a, b = pair
                    print(
                        f"  ({a}, {b}): worlds ({np_shape_world[a]}, {np_shape_world[b]}) groups ({np_collision_group[a]}, {np_collision_group[b]})"
                    )

            if extra_in_warp:
                print(f"\nPairs in Warp but not in numpy ({len(extra_in_warp)}):")
                for pair in list(extra_in_warp)[:10]:  # Show first 10
                    a, b = pair
                    print(
                        f"  ({a}, {b}): worlds ({np_shape_world[a]}, {np_shape_world[b]}) groups ({np_collision_group[a]}, {np_collision_group[b]})"
                    )

            assert len(pairs_np) == num_candidate_pair_val

        # Ensure every element in pairs_wp is also present in pairs_np
        pairs_np_set = {tuple(pair) for pair in pairs_np}
        for pair in pairs_wp[:num_candidate_pair_val]:
            pair_tuple = tuple(pair)
            assert pair_tuple in pairs_np_set, f"Pair {pair_tuple} from Warp not found in numpy pairs"

        if verbose:
            print(f"\nTest passed! Found {len(pairs_np)} valid collision pairs across multiple worlds.")

    def test_sap_broadphase_multiple_worlds_segmented(self):
        """Test SAP broad phase with multiple worlds using segmented sort."""
        self._test_sap_broadphase_multiple_worlds_impl("segmented")

    def test_sap_broadphase_multiple_worlds_tile(self):
        """Test SAP broad phase with multiple worlds using tile sort."""
        self._test_sap_broadphase_multiple_worlds_impl("tile")

    def _test_sap_broadphase_with_shape_flags_impl(self, sort_type):
        """Test SAP broad phase with ShapeFlags filtering.

        This test verifies that:
        - Shapes without COLLIDE_SHAPES flag are correctly filtered out
        - Filtering works correctly with multiple worlds
        - num_regular_worlds is correctly computed after filtering (tests bug fix)
        """
        verbose = False

        # Create random bounding boxes in min-max format
        ngeom = 40
        world_count = 4

        # Generate random centers and sizes
        rng = np.random.Generator(np.random.PCG64(789))

        centers = rng.random((ngeom, 3)) * 5.0
        sizes = rng.random((ngeom, 3)) * 1.5
        geom_bounding_box_lower = centers - sizes
        geom_bounding_box_upper = centers + sizes

        np_geom_cutoff = np.zeros(ngeom, dtype=np.float32)

        # Randomly assign collision groups
        num_groups = 5
        np_collision_group = rng.integers(1, num_groups + 1, size=ngeom, dtype=np.int32)

        # Make some entities shared (negative collision group)
        num_shared = int(sqrt(ngeom))
        shared_indices = rng.choice(ngeom, size=num_shared, replace=False)
        np_collision_group[shared_indices] = -1

        # Randomly distribute objects across worlds
        np_shape_world = rng.integers(0, world_count, size=ngeom, dtype=np.int32)

        # Make some entities global (world -1)
        num_global = max(3, ngeom // 10)
        global_indices = rng.choice(ngeom, size=num_global, replace=False)
        np_shape_world[global_indices] = -1

        # Create shape flags: some geometries will be visual-only
        np_shape_flags = np.zeros(ngeom, dtype=np.int32)

        # Assign flags: ~60% will have COLLIDE_SHAPES flag
        colliding_indices = rng.choice(ngeom, size=int(0.6 * ngeom), replace=False)
        np_shape_flags[colliding_indices] = ShapeFlags.COLLIDE_SHAPES

        # CRITICAL TEST CASE: Filter out all geometries from world 1
        world_1_mask = np_shape_world == 1
        np_shape_flags[world_1_mask] = 0  # Remove COLLIDE_SHAPES from all world 1 geometries

        # Count how many colliding geometries remain after filtering
        colliding_mask = (np_shape_flags & ShapeFlags.COLLIDE_SHAPES) != 0
        num_colliding = np.sum(colliding_mask)

        if verbose:
            print("\nSAP test setup with ShapeFlags:")
            print(f"  Total geometries: {ngeom}")
            print(f"  Geometries with COLLIDE_SHAPES flag: {num_colliding}")
            print(f"  Geometries filtered out: {ngeom - num_colliding}")

        # Compute expected pairs using numpy (with shape flags filtering)
        pairs_np = find_overlapping_pairs_np(
            geom_bounding_box_lower,
            geom_bounding_box_upper,
            np_geom_cutoff,
            np_collision_group,
            np_shape_world,
            np_shape_flags,
        )

        # Setup Warp arrays
        num_lower_tri_elements = ngeom * (ngeom - 1) // 2
        geom_lower = wp.array(geom_bounding_box_lower, dtype=wp.vec3)
        geom_upper = wp.array(geom_bounding_box_upper, dtype=wp.vec3)
        geom_cutoff = wp.array(np_geom_cutoff)
        collision_group = wp.array(np_collision_group)
        shape_world = wp.array(np_shape_world, dtype=wp.int32)
        shape_flags = wp.array(np_shape_flags, dtype=wp.int32)
        candidate_pair_count = wp.array([0], dtype=wp.int32)
        candidate_pair = wp.array(np.zeros((num_lower_tri_elements, 2), dtype=wp.int32), dtype=wp.vec2i)

        # Initialize SAP broad phase with shape_flags
        sap_broadphase = BroadPhaseSAP(shape_world, shape_flags=shape_flags, sort_type=sort_type)

        # Verify num_regular_worlds is correct after filtering
        colliding_worlds = np.unique(np_shape_world[colliding_mask])
        colliding_worlds = colliding_worlds[colliding_worlds >= 0]  # Exclude -1
        expected_num_regular_worlds = len(colliding_worlds)

        self.assertEqual(
            sap_broadphase.num_regular_worlds,
            expected_num_regular_worlds,
            f"num_regular_worlds mismatch: expected {expected_num_regular_worlds} "
            f"(after filtering), got {sap_broadphase.num_regular_worlds}",
        )

        # Launch SAP broad phase
        sap_broadphase.launch(
            geom_lower,
            geom_upper,
            geom_cutoff,
            collision_group,
            shape_world,
            ngeom,
            candidate_pair,
            candidate_pair_count,
        )

        # Get results
        pairs_wp = candidate_pair.numpy()
        num_candidate_pair_result = candidate_pair_count.numpy()[0]

        # Verify results
        if len(pairs_np) != num_candidate_pair_result:
            pairs_np_set = {tuple(pair) for pair in pairs_np}
            pairs_wp_set = {tuple(pairs_wp[i]) for i in range(num_candidate_pair_result)}

            missing = pairs_np_set - pairs_wp_set
            extra = pairs_wp_set - pairs_np_set

            if missing:
                print(f"\nMissing pairs ({len(missing)}):")
                for pair in list(missing)[:10]:
                    body_a, body_b = pair
                    print(f"  {pair}: worlds ({np_shape_world[body_a]}, {np_shape_world[body_b]})")

            if extra:
                print(f"\nExtra pairs ({len(extra)}):")
                for pair in list(extra)[:10]:
                    body_a, body_b = pair
                    print(f"  {pair}: worlds ({np_shape_world[body_a]}, {np_shape_world[body_b]})")

            assert len(pairs_np) == num_candidate_pair_result

        # Ensure every element in pairs_wp is also present in pairs_np
        pairs_np_set = {tuple(pair) for pair in pairs_np}
        for pair in pairs_wp[:num_candidate_pair_result]:
            assert tuple(pair) in pairs_np_set, f"Pair {tuple(pair)} from Warp not found in numpy pairs"

        # Verify that no pairs contain filtered-out geometries
        for pair in pairs_wp[:num_candidate_pair_result]:
            body_a, body_b = pair[0], pair[1]
            flag_a = np_shape_flags[body_a]
            flag_b = np_shape_flags[body_b]
            assert (flag_a & ShapeFlags.COLLIDE_SHAPES) != 0, (
                f"Pair contains filtered geometry {body_a} with flag {flag_a}"
            )
            assert (flag_b & ShapeFlags.COLLIDE_SHAPES) != 0, (
                f"Pair contains filtered geometry {body_b} with flag {flag_b}"
            )

    def test_sap_broadphase_with_shape_flags_segmented(self):
        """Test SAP broad phase with ShapeFlags using segmented sort."""
        self._test_sap_broadphase_with_shape_flags_impl("segmented")

    def test_sap_broadphase_with_shape_flags_tile(self):
        """Test SAP broad phase with ShapeFlags using tile sort."""
        self._test_sap_broadphase_with_shape_flags_impl("tile")

    def test_nxn_edge_cases(self):
        """Test NxN broad phase with tricky edge cases to verify GPU code correctness.

        This test includes:
        - Boundary conditions (AABBs exactly touching)
        - Various cutoff distances that create/prevent overlaps
        - Complex world/group interactions
        - Duplicate pair prevention
        - Mixed global (-1) and world-specific entities
        - Large number of geometries to stress-test GPU code
        """
        verbose = False

        # Create a carefully crafted scenario with edge cases
        # Use larger number to really stress test the GPU code
        base_cases = 24  # Base edge cases
        num_clusters = 10  # Number of overlapping clusters
        cluster_size = 8  # Geometries per cluster
        num_isolated = 20  # Isolated geometries
        ngeom = base_cases + (num_clusters * cluster_size) + num_isolated  # Total: 124 geometries

        # Case 1: Two boxes exactly touching (boundary case) - should overlap with cutoff=0
        box1_lower = np.array([0.0, 0.0, 0.0])
        box1_upper = np.array([1.0, 1.0, 1.0])
        box2_lower = np.array([1.0, 0.0, 0.0])  # Exactly touching box1 on x-axis
        box2_upper = np.array([2.0, 1.0, 1.0])

        # Case 2: Two boxes very close but not touching - should overlap with cutoff > 0
        box3_lower = np.array([2.1, 0.0, 0.0])
        box3_upper = np.array([3.0, 1.0, 1.0])
        box4_lower = np.array([3.2, 0.0, 0.0])  # Gap of 0.2
        box4_upper = np.array([4.0, 1.0, 1.0])

        # Case 3: Three boxes in a row, all overlapping
        box5_lower = np.array([5.0, 0.0, 0.0])
        box5_upper = np.array([6.5, 1.0, 1.0])
        box6_lower = np.array([6.0, 0.0, 0.0])
        box6_upper = np.array([7.5, 1.0, 1.0])
        box7_lower = np.array([7.0, 0.0, 0.0])
        box7_upper = np.array([8.5, 1.0, 1.0])

        # Case 4: Multiple boxes in same location (stress test for duplicates)
        # All these boxes should collide with each other
        box8_lower = np.array([10.0, 0.0, 0.0])
        box8_upper = np.array([11.0, 1.0, 1.0])
        box9_lower = np.array([10.1, 0.1, 0.1])
        box9_upper = np.array([10.9, 0.9, 0.9])
        box10_lower = np.array([10.2, 0.2, 0.2])
        box10_upper = np.array([10.8, 0.8, 0.8])
        box11_lower = np.array([10.3, 0.3, 0.3])
        box11_upper = np.array([10.7, 0.7, 0.7])

        # Case 5: Global entities (-1 world) that should collide with multiple worlds
        box12_lower = np.array([15.0, 0.0, 0.0])
        box12_upper = np.array([16.0, 1.0, 1.0])
        box13_lower = np.array([15.2, 0.2, 0.2])
        box13_upper = np.array([15.8, 0.8, 0.8])

        # Case 6: Collision group filtering edge cases
        # Same location but different groups (some should collide, some shouldn't)
        box14_lower = np.array([20.0, 0.0, 0.0])
        box14_upper = np.array([21.0, 1.0, 1.0])
        box15_lower = np.array([20.1, 0.1, 0.1])
        box15_upper = np.array([20.9, 0.9, 0.9])
        box16_lower = np.array([20.2, 0.2, 0.2])
        box16_upper = np.array([20.8, 0.8, 0.8])
        box17_lower = np.array([20.3, 0.3, 0.3])
        box17_upper = np.array([20.7, 0.7, 0.7])

        # Case 7: Different worlds (should NOT collide even if overlapping)
        box18_lower = np.array([25.0, 0.0, 0.0])
        box18_upper = np.array([26.0, 1.0, 1.0])
        box19_lower = np.array([25.2, 0.2, 0.2])
        box19_upper = np.array([25.8, 0.8, 0.8])

        # Case 8: Isolated boxes (no collisions)
        box20_lower = np.array([30.0, 0.0, 0.0])
        box20_upper = np.array([31.0, 1.0, 1.0])
        box21_lower = np.array([35.0, 0.0, 0.0])
        box21_upper = np.array([36.0, 1.0, 1.0])
        box22_lower = np.array([40.0, 0.0, 0.0])
        box22_upper = np.array([41.0, 1.0, 1.0])

        # Case 9: Zero collision group (should never collide)
        box23_lower = np.array([45.0, 0.0, 0.0])
        box23_upper = np.array([46.0, 1.0, 1.0])
        box24_lower = np.array([45.2, 0.2, 0.2])
        box24_upper = np.array([45.8, 0.8, 0.8])

        # Build the base cases list
        base_lowers = [
            box1_lower,
            box2_lower,
            box3_lower,
            box4_lower,
            box5_lower,
            box6_lower,
            box7_lower,
            box8_lower,
            box9_lower,
            box10_lower,
            box11_lower,
            box12_lower,
            box13_lower,
            box14_lower,
            box15_lower,
            box16_lower,
            box17_lower,
            box18_lower,
            box19_lower,
            box20_lower,
            box21_lower,
            box22_lower,
            box23_lower,
            box24_lower,
        ]
        base_uppers = [
            box1_upper,
            box2_upper,
            box3_upper,
            box4_upper,
            box5_upper,
            box6_upper,
            box7_upper,
            box8_upper,
            box9_upper,
            box10_upper,
            box11_upper,
            box12_upper,
            box13_upper,
            box14_upper,
            box15_upper,
            box16_upper,
            box17_upper,
            box18_upper,
            box19_upper,
            box20_upper,
            box21_upper,
            box22_upper,
            box23_upper,
            box24_upper,
        ]

        # Case 10+: Add overlapping clusters (stress test for many-to-many collisions)
        # Each cluster has multiple overlapping geometries in the same location
        rng = np.random.Generator(np.random.PCG64(999))
        cluster_lowers = []
        cluster_uppers = []
        cluster_cutoffs = []
        cluster_groups = []
        cluster_worlds = []

        for cluster_id in range(num_clusters):
            # Random center for this cluster
            cluster_center = np.array([50.0 + cluster_id * 10.0, rng.random() * 5.0, rng.random() * 5.0])
            # Randomly assign world (mix of specific worlds and global)
            if cluster_id < 3:
                cluster_world = -1  # First 3 clusters are global
            else:
                cluster_world = cluster_id % 4  # Distribute across 4 worlds

            # Random collision group for this cluster
            cluster_group = (cluster_id % 3) + 1  # Groups 1, 2, 3

            for _ in range(cluster_size):
                # Create slightly offset overlapping boxes
                offset = rng.random(3) * 0.5
                lower = cluster_center - 0.5 + offset
                upper = cluster_center + 0.5 + offset
                cluster_lowers.append(lower)
                cluster_uppers.append(upper)
                cluster_cutoffs.append(0.0)
                cluster_groups.append(cluster_group)
                cluster_worlds.append(cluster_world)

        # Case 11+: Add isolated geometries (should have no collisions)
        isolated_lowers = []
        isolated_uppers = []
        isolated_cutoffs = []
        isolated_groups = []
        isolated_worlds = []

        for i in range(num_isolated):
            # Place far apart
            center = np.array([200.0 + i * 5.0, 0.0, 0.0])
            isolated_lowers.append(center - 0.3)
            isolated_uppers.append(center + 0.3)
            isolated_cutoffs.append(0.0)
            isolated_groups.append((i % 5) + 1)  # Varied groups
            isolated_worlds.append(i % 3)  # Distribute across worlds

        # Combine all geometries
        all_lowers = base_lowers + cluster_lowers + isolated_lowers
        all_uppers = base_uppers + cluster_uppers + isolated_uppers

        geom_bounding_box_lower = np.array(all_lowers)
        geom_bounding_box_upper = np.array(all_uppers)

        # Combine cutoffs
        base_cutoffs = [
            0.0,
            0.0,  # box1, box2: exactly touching
            0.0,
            0.15,  # box3, box4: gap of 0.2, cutoff 0.15 makes them overlap
            0.0,
            0.0,
            0.0,  # box5-7: overlapping chain
            0.0,
            0.0,
            0.0,
            0.0,  # box8-11: nested boxes
            0.0,
            0.0,  # box12-13: global entities
            0.0,
            0.0,
            0.0,
            0.0,  # box14-17: group filtering
            0.0,
            0.0,  # box18-19: different worlds
            0.0,
            0.0,
            0.0,  # box20-22: isolated
            0.0,
            0.0,  # box23-24: zero group
        ]
        np_geom_cutoff = np.array(base_cutoffs + cluster_cutoffs + isolated_cutoffs, dtype=np.float32)

        # Combine collision groups
        base_groups = [
            1,
            1,  # box1-2: same group, should collide
            2,
            2,  # box3-4: same group, should collide (with cutoff)
            1,
            1,
            1,  # box5-7: same group, chain collision
            1,
            1,
            1,
            1,  # box8-11: same group, all collide
            -1,
            -1,  # box12-13: negative group, collide with each other
            1,
            2,
            -1,
            -2,  # box14-17: mixed groups (1 w/ -1, 2 w/ -1, not 1 w/ 2, not -1 w/ -2)
            1,
            1,  # box18-19: same group but different worlds
            1,
            2,
            3,  # box20-22: different groups, no collision
            0,
            0,  # box23-24: zero group, never collide
        ]
        np_collision_group = np.array(base_groups + cluster_groups + isolated_groups, dtype=np.int32)

        # Combine worlds
        base_worlds = [
            0,
            0,  # box1-2: world 0
            0,
            0,  # box3-4: world 0
            0,
            0,
            0,  # box5-7: world 0
            0,
            0,
            0,
            0,  # box8-11: world 0
            -1,
            -1,  # box12-13: global, collide with all worlds
            0,
            0,
            0,
            0,  # box14-17: world 0
            0,
            1,  # box18-19: different worlds (should NOT collide)
            0,
            0,
            0,  # box20-22: world 0 (but different groups)
            0,
            0,  # box23-24: world 0 (but zero group)
        ]
        np_shape_world = np.array(base_worlds + cluster_worlds + isolated_worlds, dtype=np.int32)

        if verbose:
            print("\n=== NxN Edge Case Test Setup ===")
            print(f"Total geometries: {ngeom}")
            for i in range(ngeom):
                print(
                    f"  Geom {i}: world={np_shape_world[i]}, group={np_collision_group[i]}, "
                    f"cutoff={np_geom_cutoff[i]:.2f}"
                )

        # Compute expected pairs using numpy
        pairs_np = find_overlapping_pairs_np(
            geom_bounding_box_lower, geom_bounding_box_upper, np_geom_cutoff, np_collision_group, np_shape_world
        )

        if verbose:
            print(f"\nExpected {len(pairs_np)} pairs from numpy verification")
            for pair in pairs_np:
                a, b = pair
                print(
                    f"  Pair ({a}, {b}): worlds ({np_shape_world[a]}, {np_shape_world[b]}) "
                    f"groups ({np_collision_group[a]}, {np_collision_group[b]})"
                )

        # Setup Warp arrays
        num_lower_tri_elements = ngeom * (ngeom - 1) // 2
        geom_lower = wp.array(geom_bounding_box_lower, dtype=wp.vec3)
        geom_upper = wp.array(geom_bounding_box_upper, dtype=wp.vec3)
        geom_cutoff = wp.array(np_geom_cutoff)
        collision_group = wp.array(np_collision_group)
        shape_world = wp.array(np_shape_world, dtype=wp.int32)
        candidate_pair_count = wp.array([0], dtype=wp.int32)
        candidate_pair = wp.array(np.zeros((num_lower_tri_elements, 2), dtype=wp.int32), dtype=wp.vec2i)

        # Initialize and launch NxN broad phase
        nxn_broadphase = BroadPhaseAllPairs(shape_world)
        nxn_broadphase.launch(
            geom_lower,
            geom_upper,
            geom_cutoff,
            collision_group,
            shape_world,
            ngeom,
            candidate_pair,
            candidate_pair_count,
        )

        # Get results
        pairs_wp = candidate_pair.numpy()
        num_candidate_pair_result = candidate_pair_count.numpy()[0]

        if verbose:
            print(f"\nWarp found {num_candidate_pair_result} pairs")
            for i in range(num_candidate_pair_result):
                pair = pairs_wp[i]
                a, b = pair[0], pair[1]
                print(
                    f"  Pair ({a}, {b}): worlds ({np_shape_world[a]}, {np_shape_world[b]}) "
                    f"groups ({np_collision_group[a]}, {np_collision_group[b]})"
                )

        # Verify: check for duplicate pairs
        pairs_wp_set = {tuple(pairs_wp[i]) for i in range(num_candidate_pair_result)}
        self.assertEqual(
            len(pairs_wp_set), num_candidate_pair_result, "Duplicate pairs detected in NxN broad phase results"
        )

        # Verify: check count matches
        if len(pairs_np) != num_candidate_pair_result:
            pairs_np_set = {tuple(pair) for pair in pairs_np}
            missing = pairs_np_set - pairs_wp_set
            extra = pairs_wp_set - pairs_np_set

            if missing:
                print(f"\nMissing pairs ({len(missing)}):")
                for pair in list(missing):
                    a, b = pair
                    print(
                        f"  ({a}, {b}): worlds ({np_shape_world[a]}, {np_shape_world[b]}) "
                        f"groups ({np_collision_group[a]}, {np_collision_group[b]})"
                    )

            if extra:
                print(f"\nExtra pairs ({len(extra)}):")
                for pair in list(extra):
                    a, b = pair
                    print(
                        f"  ({a}, {b}): worlds ({np_shape_world[a]}, {np_shape_world[b]}) "
                        f"groups ({np_collision_group[a]}, {np_collision_group[b]})"
                    )

        self.assertEqual(
            len(pairs_np), num_candidate_pair_result, f"Expected {len(pairs_np)} pairs, got {num_candidate_pair_result}"
        )

        # Verify: all Warp pairs are in numpy pairs
        pairs_np_set = {tuple(pair) for pair in pairs_np}
        for pair in pairs_wp[:num_candidate_pair_result]:
            pair_tuple = tuple(pair)
            self.assertIn(pair_tuple, pairs_np_set, f"Pair {pair_tuple} from Warp not found in numpy pairs")

        if verbose:
            print(f"\n✓ Test passed! All {len(pairs_np)} pairs matched, no duplicates.")

    def _test_sap_edge_cases_impl(self, sort_type):
        """Test SAP broad phase with tricky edge cases to verify GPU code correctness.

        This test includes:
        - Boundary conditions (AABBs exactly touching)
        - Various cutoff distances that create/prevent overlaps
        - Complex world/group interactions
        - Duplicate pair prevention (especially for shared geometries)
        - Mixed global (-1) and world-specific entities
        - Large number of geometries to stress-test GPU sorting and sweep
        """
        verbose = False

        # Create a carefully crafted scenario with edge cases
        # Use larger number to really stress test the SAP algorithm
        base_cases = 26  # Base edge cases
        num_clusters = 12  # Number of overlapping clusters (SAP stress test)
        cluster_size = 10  # Geometries per cluster (larger for SAP)
        num_isolated = 25  # Isolated geometries
        ngeom = base_cases + (num_clusters * cluster_size) + num_isolated  # Total: 171 geometries

        # Case 1: Two boxes exactly touching along sweep axis (x-axis) - should overlap
        box1_lower = np.array([0.0, 0.0, 0.0])
        box1_upper = np.array([1.0, 1.0, 1.0])
        box2_lower = np.array([1.0, 0.0, 0.0])  # Exactly touching box1 on x-axis
        box2_upper = np.array([2.0, 1.0, 1.0])

        # Case 2: Boxes with gap, but cutoff makes them overlap
        box3_lower = np.array([2.15, 0.0, 0.0])
        box3_upper = np.array([3.0, 1.0, 1.0])
        box4_lower = np.array([3.25, 0.0, 0.0])  # Gap of 0.25
        box4_upper = np.array([4.0, 1.0, 1.0])

        # Case 3: Overlapping chain (stress test for SAP sorting)
        box5_lower = np.array([5.0, 0.0, 0.0])
        box5_upper = np.array([6.5, 1.0, 1.0])
        box6_lower = np.array([6.0, 0.0, 0.0])
        box6_upper = np.array([7.5, 1.0, 1.0])
        box7_lower = np.array([7.0, 0.0, 0.0])
        box7_upper = np.array([8.5, 1.0, 1.0])
        box8_lower = np.array([8.0, 0.0, 0.0])
        box8_upper = np.array([9.5, 1.0, 1.0])

        # Case 4: Multiple boxes in same location (duplicate prevention test)
        box9_lower = np.array([10.0, 0.0, 0.0])
        box9_upper = np.array([11.0, 1.0, 1.0])
        box10_lower = np.array([10.1, 0.1, 0.1])
        box10_upper = np.array([10.9, 0.9, 0.9])
        box11_lower = np.array([10.2, 0.2, 0.2])
        box11_upper = np.array([10.8, 0.8, 0.8])

        # Case 5: Global entities that should appear in multiple worlds
        # Critical: these should only generate ONE pair total, not one per world
        box12_lower = np.array([15.0, 0.0, 0.0])
        box12_upper = np.array([16.0, 1.0, 1.0])
        box13_lower = np.array([15.2, 0.2, 0.2])
        box13_upper = np.array([15.8, 0.8, 0.8])

        # Case 6: Mixed global and world-specific entities
        box14_lower = np.array([18.0, 0.0, 0.0])
        box14_upper = np.array([19.0, 1.0, 1.0])
        box15_lower = np.array([18.2, 0.2, 0.2])
        box15_upper = np.array([18.8, 0.8, 0.8])
        box16_lower = np.array([18.4, 0.4, 0.4])
        box16_upper = np.array([18.6, 0.6, 0.6])

        # Case 7: Collision group edge cases at same location
        box17_lower = np.array([22.0, 0.0, 0.0])
        box17_upper = np.array([23.0, 1.0, 1.0])
        box18_lower = np.array([22.1, 0.1, 0.1])
        box18_upper = np.array([22.9, 0.9, 0.9])
        box19_lower = np.array([22.2, 0.2, 0.2])
        box19_upper = np.array([22.8, 0.8, 0.8])
        box20_lower = np.array([22.3, 0.3, 0.3])
        box20_upper = np.array([22.7, 0.7, 0.7])

        # Case 8: Different worlds, overlapping (should NOT collide)
        box21_lower = np.array([26.0, 0.0, 0.0])
        box21_upper = np.array([27.0, 1.0, 1.0])
        box22_lower = np.array([26.2, 0.2, 0.2])
        box22_upper = np.array([26.8, 0.8, 0.8])

        # Case 9: Reverse order in space (tests SAP sorting correctness)
        box23_lower = np.array([30.0, 0.0, 0.0])
        box23_upper = np.array([31.0, 1.0, 1.0])
        box24_lower = np.array([29.0, 0.0, 0.0])  # Lower x than box23
        box24_upper = np.array([30.5, 1.0, 1.0])

        # Case 10: Zero collision group (never collides)
        box25_lower = np.array([33.0, 0.0, 0.0])
        box25_upper = np.array([34.0, 1.0, 1.0])
        box26_lower = np.array([33.2, 0.2, 0.2])
        box26_upper = np.array([33.8, 0.8, 0.8])

        # Build the base cases list
        base_lowers = [
            box1_lower,
            box2_lower,
            box3_lower,
            box4_lower,
            box5_lower,
            box6_lower,
            box7_lower,
            box8_lower,
            box9_lower,
            box10_lower,
            box11_lower,
            box12_lower,
            box13_lower,
            box14_lower,
            box15_lower,
            box16_lower,
            box17_lower,
            box18_lower,
            box19_lower,
            box20_lower,
            box21_lower,
            box22_lower,
            box23_lower,
            box24_lower,
            box25_lower,
            box26_lower,
        ]
        base_uppers = [
            box1_upper,
            box2_upper,
            box3_upper,
            box4_upper,
            box5_upper,
            box6_upper,
            box7_upper,
            box8_upper,
            box9_upper,
            box10_upper,
            box11_upper,
            box12_upper,
            box13_upper,
            box14_upper,
            box15_upper,
            box16_upper,
            box17_upper,
            box18_upper,
            box19_upper,
            box20_upper,
            box21_upper,
            box22_upper,
            box23_upper,
            box24_upper,
            box25_upper,
            box26_upper,
        ]

        # Case 11+: Add overlapping clusters along the sweep axis (SAP stress test)
        # These will all have similar projections on the sweep axis, stressing the sorting
        rng = np.random.Generator(np.random.PCG64(888))
        cluster_lowers = []
        cluster_uppers = []
        cluster_cutoffs = []
        cluster_groups = []
        cluster_worlds = []

        for cluster_id in range(num_clusters):
            # Random center for this cluster, but align them along x-axis for SAP stress
            # This creates a challenging scenario where many boxes overlap in the sweep direction
            x_base = 100.0 + cluster_id * 8.0
            cluster_center = np.array([x_base, rng.random() * 10.0, rng.random() * 10.0])

            # Randomly assign world (mix of specific worlds and global)
            if cluster_id < 4:
                cluster_world = -1  # First 4 clusters are global (tests duplicate prevention)
            else:
                cluster_world = cluster_id % 5  # Distribute across 5 worlds

            # Random collision group for this cluster
            cluster_group = (cluster_id % 4) + 1  # Groups 1, 2, 3, 4

            for i in range(cluster_size):
                # Create overlapping boxes with variation along sweep axis
                offset = rng.random(3) * 0.6
                # Extend along x-axis to create overlaps in sweep direction
                lower = cluster_center - 0.6 + offset
                upper = cluster_center + 0.6 + offset
                cluster_lowers.append(lower)
                cluster_uppers.append(upper)
                cluster_cutoffs.append(0.0 if i % 3 != 0 else 0.1)  # Mix of cutoffs
                cluster_groups.append(cluster_group if i % 5 != 0 else -1)  # Some shared
                cluster_worlds.append(cluster_world)

        # Case 12+: Add isolated geometries along x-axis (tests SAP correctly skips far objects)
        isolated_lowers = []
        isolated_uppers = []
        isolated_cutoffs = []
        isolated_groups = []
        isolated_worlds = []

        for i in range(num_isolated):
            # Place far apart along x-axis
            center = np.array([300.0 + i * 6.0, rng.random() * 5.0, rng.random() * 5.0])
            isolated_lowers.append(center - 0.25)
            isolated_uppers.append(center + 0.25)
            isolated_cutoffs.append(0.0)
            isolated_groups.append((i % 6) + 1)  # Varied groups
            isolated_worlds.append(i % 4)  # Distribute across worlds

        # Combine all geometries
        all_lowers = base_lowers + cluster_lowers + isolated_lowers
        all_uppers = base_uppers + cluster_uppers + isolated_uppers

        geom_bounding_box_lower = np.array(all_lowers)
        geom_bounding_box_upper = np.array(all_uppers)

        # Combine cutoffs
        base_cutoffs = [
            0.0,
            0.0,  # box1-2: exactly touching
            0.0,
            0.15,  # box3-4: gap of 0.25, cutoff 0.15 makes them overlap (combined 0.3)
            0.0,
            0.0,
            0.0,
            0.0,  # box5-8: overlapping chain
            0.0,
            0.0,
            0.0,  # box9-11: nested boxes
            0.0,
            0.0,  # box12-13: global entities (duplicate prevention critical)
            0.0,
            0.0,
            0.0,  # box14-16: mixed global/world
            0.0,
            0.0,
            0.0,
            0.0,  # box17-20: group filtering
            0.0,
            0.0,  # box21-22: different worlds
            0.0,
            0.0,  # box23-24: reverse order
            0.0,
            0.0,  # box25-26: zero group
        ]
        np_geom_cutoff = np.array(base_cutoffs + cluster_cutoffs + isolated_cutoffs, dtype=np.float32)

        # Combine collision groups
        base_groups = [
            1,
            1,  # box1-2: same group
            2,
            2,  # box3-4: same group
            1,
            1,
            1,
            1,  # box5-8: same group, chain
            1,
            1,
            1,  # box9-11: same group, nested
            -1,
            -2,  # box12-13: both negative (SHOULD collide, different negative values)
            -1,
            1,
            2,  # box14-16: global collides with both groups
            1,
            2,
            -1,
            -2,  # box17-20: 1 w/ -1, 2 w/ -1, not 1 w/ 2, not -1 w/ -2
            1,
            1,  # box21-22: same group but different worlds
            1,
            1,  # box23-24: reverse order
            0,
            0,  # box25-26: zero group
        ]
        np_collision_group = np.array(base_groups + cluster_groups + isolated_groups, dtype=np.int32)

        # Combine worlds
        base_worlds = [
            0,
            0,  # box1-2
            0,
            0,  # box3-4
            0,
            0,
            0,
            0,  # box5-8
            0,
            0,
            0,  # box9-11
            -1,
            -1,  # box12-13: BOTH global (critical for duplicate prevention)
            -1,
            1,
            2,  # box14-16: global with world-specific
            0,
            0,
            0,
            0,  # box17-20
            0,
            1,  # box21-22: different worlds
            0,
            0,  # box23-24
            0,
            0,  # box25-26
        ]
        np_shape_world = np.array(base_worlds + cluster_worlds + isolated_worlds, dtype=np.int32)

        if verbose:
            print("\n=== SAP Edge Case Test Setup ===")
            print(f"Total geometries: {ngeom}")
            for i in range(ngeom):
                print(
                    f"  Geom {i}: world={np_shape_world[i]}, group={np_collision_group[i]}, "
                    f"cutoff={np_geom_cutoff[i]:.2f}"
                )

        # Compute expected pairs using numpy
        pairs_np = find_overlapping_pairs_np(
            geom_bounding_box_lower, geom_bounding_box_upper, np_geom_cutoff, np_collision_group, np_shape_world
        )

        if verbose:
            print(f"\nExpected {len(pairs_np)} pairs from numpy verification")
            for pair in pairs_np:
                a, b = pair
                print(
                    f"  Pair ({a}, {b}): worlds ({np_shape_world[a]}, {np_shape_world[b]}) "
                    f"groups ({np_collision_group[a]}, {np_collision_group[b]})"
                )

        # Setup Warp arrays
        num_lower_tri_elements = ngeom * (ngeom - 1) // 2
        geom_lower = wp.array(geom_bounding_box_lower, dtype=wp.vec3)
        geom_upper = wp.array(geom_bounding_box_upper, dtype=wp.vec3)
        geom_cutoff = wp.array(np_geom_cutoff)
        collision_group = wp.array(np_collision_group)
        shape_world = wp.array(np_shape_world, dtype=wp.int32)
        candidate_pair_count = wp.array([0], dtype=wp.int32)
        candidate_pair = wp.array(np.zeros((num_lower_tri_elements, 2), dtype=wp.int32), dtype=wp.vec2i)

        # Initialize and launch SAP broad phase
        sap_broadphase = BroadPhaseSAP(shape_world, sort_type=sort_type)
        sap_broadphase.launch(
            geom_lower,
            geom_upper,
            geom_cutoff,
            collision_group,
            shape_world,
            ngeom,
            candidate_pair,
            candidate_pair_count,
        )

        # Get results
        pairs_wp = candidate_pair.numpy()
        num_candidate_pair_result = candidate_pair_count.numpy()[0]

        if verbose:
            print(f"\nWarp found {num_candidate_pair_result} pairs")
            for i in range(num_candidate_pair_result):
                pair = pairs_wp[i]
                a, b = pair[0], pair[1]
                print(
                    f"  Pair ({a}, {b}): worlds ({np_shape_world[a]}, {np_shape_world[b]}) "
                    f"groups ({np_collision_group[a]}, {np_collision_group[b]})"
                )

        # Verify: check for duplicate pairs
        pairs_wp_set = {tuple(pairs_wp[i]) for i in range(num_candidate_pair_result)}
        self.assertEqual(
            len(pairs_wp_set), num_candidate_pair_result, "Duplicate pairs detected in SAP broad phase results"
        )

        # Verify: check count matches
        if len(pairs_np) != num_candidate_pair_result:
            pairs_np_set = {tuple(pair) for pair in pairs_np}
            missing = pairs_np_set - pairs_wp_set
            extra = pairs_wp_set - pairs_np_set

            if missing:
                print(f"\nMissing pairs ({len(missing)}):")
                for pair in list(missing):
                    a, b = pair
                    print(
                        f"  ({a}, {b}): worlds ({np_shape_world[a]}, {np_shape_world[b]}) "
                        f"groups ({np_collision_group[a]}, {np_collision_group[b]})"
                    )

            if extra:
                print(f"\nExtra pairs ({len(extra)}):")
                for pair in list(extra):
                    a, b = pair
                    print(
                        f"  ({a}, {b}): worlds ({np_shape_world[a]}, {np_shape_world[b]}) "
                        f"groups ({np_collision_group[a]}, {np_collision_group[b]})"
                    )

        self.assertEqual(
            len(pairs_np), num_candidate_pair_result, f"Expected {len(pairs_np)} pairs, got {num_candidate_pair_result}"
        )

        # Verify: all Warp pairs are in numpy pairs
        pairs_np_set = {tuple(pair) for pair in pairs_np}
        for pair in pairs_wp[:num_candidate_pair_result]:
            pair_tuple = tuple(pair)
            self.assertIn(pair_tuple, pairs_np_set, f"Pair {pair_tuple} from Warp not found in numpy pairs")

        if verbose:
            print(f"\n✓ Test passed! All {len(pairs_np)} pairs matched, no duplicates.")

    def test_sap_edge_cases_segmented(self):
        """Test SAP edge cases with segmented sort."""
        self._test_sap_edge_cases_impl("segmented")

    def test_sap_edge_cases_tile(self):
        """Test SAP edge cases with tile sort."""
        self._test_sap_edge_cases_impl("tile")

    def test_per_shape_gap_broad_phase(self):
        """
        Test that all broad phase modes correctly handle per-shape contact gaps
        by applying them during AABB overlap checks (not pre-expanded).

        Setup two spheres (A and B) at different separations from a ground plane:
        - Sphere A: small margin, should NOT be detected by broad phase when far
        - Sphere B: large margin, SHOULD be detected by broad phase when at same distance

        This tests that the broad phase kernels correctly expand AABBs by the provided
        margins during overlap testing, not requiring pre-expanded AABBs.
        """
        # Create UNEXPANDED AABBs for: ground plane + 2 spheres
        # The margins will be passed separately to test that broad phase applies them correctly

        # Ground plane AABB (infinite in XY, at z=0) WITHOUT margin
        ground_aabb_lower = wp.vec3(-1000.0, -1000.0, 0.0)
        ground_aabb_upper = wp.vec3(1000.0, 1000.0, 0.0)

        # Sphere A (radius=0.2, center at z=0.24) WITHOUT margin
        # AABB: z range = [0.24-0.2, 0.24+0.2] = [0.04, 0.44]
        sphere_a_aabb_lower = wp.vec3(-0.2, -0.2, 0.04)
        sphere_a_aabb_upper = wp.vec3(0.2, 0.2, 0.44)

        # Sphere B (radius=0.2, center at z=0.24) WITHOUT margin
        # AABB: z range = [0.04, 0.44]
        sphere_b_aabb_lower = wp.vec3(10.0 - 0.2, -0.2, 0.04)
        sphere_b_aabb_upper = wp.vec3(10.0 + 0.2, 0.2, 0.44)

        aabb_lower = wp.array([ground_aabb_lower, sphere_a_aabb_lower, sphere_b_aabb_lower], dtype=wp.vec3)
        aabb_upper = wp.array([ground_aabb_upper, sphere_a_aabb_upper, sphere_b_aabb_upper], dtype=wp.vec3)

        # Pass per-shape margins to broad phase - it will apply them during overlap checks
        # ground=0.01, sphereA=0.02, sphereB=0.06
        # With margins applied:
        # - Ground AABB becomes [-0.01, 0.01] in z
        # - Sphere A AABB becomes [0.04-0.02, 0.44+0.02] = [0.02, 0.46] - does NOT overlap ground
        # - Sphere B AABB becomes [0.04-0.06, 0.44+0.06] = [-0.02, 0.50] - DOES overlap ground
        shape_gap = wp.array([0.01, 0.02, 0.06], dtype=wp.float32)

        # Use collision group 1 for all shapes (group -1 collides with everything, group 0 means no collision)
        collision_group = wp.array([1, 1, 1], dtype=wp.int32)
        shape_world = wp.array([0, 0, 0], dtype=wp.int32)

        # Test NXN broad phase
        nxn_bp = BroadPhaseAllPairs(shape_world)
        pairs_nxn = wp.zeros(100, dtype=wp.vec2i)
        pair_count_nxn = wp.zeros(1, dtype=wp.int32)

        nxn_bp.launch(
            aabb_lower,
            aabb_upper,
            shape_gap,
            collision_group,
            shape_world,
            3,
            pairs_nxn,
            pair_count_nxn,
        )

        pairs_np = pairs_nxn.numpy()
        count_nxn = pair_count_nxn.numpy()[0]

        # Check that sphere B-ground pair is detected, but sphere A-ground is not
        has_sphere_b_ground = any((p[0] == 0 and p[1] == 2) or (p[0] == 2 and p[1] == 0) for p in pairs_np[:count_nxn])
        has_sphere_a_ground = any((p[0] == 0 and p[1] == 1) or (p[0] == 1 and p[1] == 0) for p in pairs_np[:count_nxn])

        self.assertTrue(has_sphere_b_ground, "NXN: Sphere B (large margin) should overlap ground")
        self.assertFalse(has_sphere_a_ground, "NXN: Sphere A (small margin) should NOT overlap ground")

        # Test SAP broad phase
        sap_bp = BroadPhaseSAP(shape_world)
        pairs_sap = wp.zeros(100, dtype=wp.vec2i)
        pair_count_sap = wp.zeros(1, dtype=wp.int32)

        sap_bp.launch(
            aabb_lower,
            aabb_upper,
            shape_gap,
            collision_group,
            shape_world,
            3,
            pairs_sap,
            pair_count_sap,
        )

        pairs_np = pairs_sap.numpy()
        count_sap = pair_count_sap.numpy()[0]

        has_sphere_b_ground = any((p[0] == 0 and p[1] == 2) or (p[0] == 2 and p[1] == 0) for p in pairs_np[:count_sap])
        has_sphere_a_ground = any((p[0] == 0 and p[1] == 1) or (p[0] == 1 and p[1] == 0) for p in pairs_np[:count_sap])

        self.assertTrue(has_sphere_b_ground, "SAP: Sphere B (large margin) should overlap ground")
        self.assertFalse(has_sphere_a_ground, "SAP: Sphere A (small margin) should NOT overlap ground")

    def test_sap_conditional_capture_retains_overlapping_pairs(self):
        """Capture SAP without allocating scan storage inside a conditional body."""
        device = wp.get_device()
        if not device.is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("Conditional graph capture requires a supported CUDA device")
        if not hasattr(wp.utils, "array_scan_temp_storage_size"):
            self.skipTest("Conditional SAP capture requires Warp scan storage support")

        count = 128
        centers = np.zeros((count, 3), dtype=np.float32)
        centers[:, 0] = np.arange(count) // 2
        lower = wp.array(centers - 0.1, dtype=wp.vec3, device=device)
        upper = wp.array(centers + 0.1, dtype=wp.vec3, device=device)
        groups = wp.ones(count, dtype=wp.int32, device=device)
        worlds = wp.zeros(count, dtype=wp.int32, device=device)
        pairs = wp.empty(count // 2, dtype=wp.vec2i, device=device)
        pair_count = wp.zeros(1, dtype=wp.int32, device=device)
        enabled = wp.ones(1, dtype=wp.int32, device=device)
        broad_phase = BroadPhaseSAP(worlds, device=device)

        def launch():
            broad_phase.launch(lower, upper, None, groups, worlds, count, pairs, pair_count, device=device)

        launch()
        pair_count.zero_()
        with wp.ScopedCapture(device=device) as capture:
            wp.capture_if(enabled, on_true=launch)
        for _ in range(2):
            wp.capture_launch(capture.graph)
            actual_count = int(pair_count.numpy()[0])
            self.assertEqual(actual_count, count // 2)
            actual_pairs = {tuple(pair) for pair in pairs.numpy()[:actual_count]}
            self.assertEqual(actual_pairs, {(i, i + 1) for i in range(0, count, 2)})
        enabled.zero_()
        pair_count.zero_()
        wp.capture_launch(capture.graph)
        self.assertEqual(int(pair_count.numpy()[0]), 0)


class TestBroadPhaseAggregate(unittest.TestCase):
    pass


_AGGREGATE_COUNT = 40
_AGGREGATE_SIZES = (1, 19, 32, 40)
_AGGREGATE_BASE_GROUPS = (1, 2, -1, -2, 0)
_AGGREGATE_DRAWN_GROUPS = (0, 1, 2, 3, -1, -2)
_KINEMATIC = 1 << 1


def _aggregate_scene(
    seed: int, *, worlds: int = 3, static_boxes: int = 0, global_statics: bool = False
) -> dict[str, np.ndarray]:
    """Build a seeded broad-phase scene around 40 aggregates.

    Shapes 0 and 1 are infinite planes, then come the 920 aggregate members,
    20 visual-only shapes on the bodies of aggregates 0-19 and ``static_boxes``
    finite static boxes. The static boxes are in world -1 when
    ``global_statics``; otherwise they cycle through world -1 and every world,
    and the aggregates, which cycle through world -1 and every world without
    static boxes, keep out of world -1 (the static grid does not support world
    -1 aggregates beside per-world statics). With static boxes, five more
    filter pairs join members and overlapping static boxes.

    Returns:
        The per-shape boxes ``lower``/``upper`` [m], ``world``, ``group``,
        ``body``, ``flags``, ``type``, ``scale`` and true ``aggregate`` ids, the
        per-body ``body_flags``, and the canonical, sorted ``filter_pairs``.
    """
    rng = np.random.default_rng(seed)
    collide = int(ShapeFlags.COLLIDE_SHAPES)
    columns = ("lower", "upper", "world", "group", "body", "flags", "type", "scale", "aggregate")
    shapes = {key: [] for key in columns}

    def add(lower, upper, world, group, body, flags, shape_type, scale, aggregate):
        values = (lower, upper, world, group, body, flags, int(shape_type), scale, aggregate)
        for key, value in zip(columns, values, strict=True):
            shapes[key].append(value)
        return len(shapes["world"]) - 1

    plane_worlds = (-1, 0) if worlds > 0 else (-1, -1)
    for world, top in zip(plane_worlds, (0.5, 1.0), strict=True):
        add((-1.0e6, -1.0e6, -1.0e6), (1.0e6, 1.0e6, top), world, -1, -1, collide, newton.GeoType.PLANE, (0.0,) * 3, -1)

    members: list[list[int]] = []
    bodies: list[list[int]] = []
    aggregate_worlds: list[int] = []
    body_count = 0
    for a in range(_AGGREGATE_COUNT):
        size = _AGGREGATE_SIZES[a % len(_AGGREGATE_SIZES)]
        if static_boxes == 0 or global_statics:
            world = a % (worlds + 1) - 1
        else:
            world = a % worlds if worlds > 0 else -1
        aggregate_worlds.append(world)
        base_group = _AGGREGATE_BASE_GROUPS[a % len(_AGGREGATE_BASE_GROUPS)]
        centre = rng.uniform(0.0, 4.0, 3)
        members.append([])
        bodies.append([])
        for m in range(size):
            # Consecutive members share a body two by two; bodies never span aggregates.
            if m % 2 == 0:
                bodies[a].append(body_count)
                body_count += 1
            group = base_group if rng.random() < 0.7 else int(rng.choice(_AGGREGATE_DRAWN_GROUPS))
            member_centre = centre + rng.uniform(-0.4, 0.4, 3)
            half = rng.uniform(0.05, 0.25, 3)
            shape = add(
                member_centre - half,
                member_centre + half,
                world,
                group,
                bodies[a][-1],
                collide,
                newton.GeoType.CAPSULE,
                (0.1, 0.2, 0.0),
                a,
            )
            members[a].append(shape)

    for a in range(20):
        centre = rng.uniform(0.0, 4.0, 3)
        half = rng.uniform(0.05, 0.25, 3)
        add(centre - half, centre + half, aggregate_worlds[a], 1, bodies[a][0], 0, newton.GeoType.SPHERE, half, -1)

    statics = []
    for k in range(static_boxes):
        centre = rng.uniform(0.0, 4.0, 3)
        half = rng.uniform(0.05, 0.25, 3)
        world = -1 if global_statics else k % (worlds + 1) - 1
        statics.append(add(centre - half, centre + half, world, -1, -1, collide, newton.GeoType.BOX, half, -1))

    # Within-aggregate filter pairs are drawn uniformly (members overlap often).
    # Across and member-plane pairs are drawn among overlapping, world-compatible
    # pairs, so that their exclusions decide pairs nxn would otherwise emit.
    filter_pairs: set[tuple[int, int]] = set()
    multi = [a for a in range(_AGGREGATE_COUNT) if len(members[a]) > 1]
    while len(filter_pairs) < 30:
        i, j = rng.choice(members[multi[rng.integers(len(multi))]], 2, replace=False)
        filter_pairs.add((int(min(i, j)), int(max(i, j))))

    box_lower = np.asarray(shapes["lower"], dtype=np.float32)
    box_upper = np.asarray(shapes["upper"], dtype=np.float32)
    shape_world = np.asarray(shapes["world"])
    shape_aggregate = np.asarray(shapes["aggregate"])
    flat = np.concatenate([np.asarray(m) for m in members])
    overlap = np.all(
        (box_lower[flat][:, None, :] <= box_upper[flat][None, :, :])
        & (box_upper[flat][:, None, :] >= box_lower[flat][None, :, :]),
        axis=2,
    )

    def draw(candidates_i, candidates_j, count):
        world_i = shape_world[candidates_i]
        world_j = shape_world[candidates_j]
        keep = (world_i == world_j) | (world_i == -1) | (world_j == -1)
        candidates = np.stack((candidates_i[keep], candidates_j[keep]), axis=1)
        added = 0
        for k in rng.permutation(len(candidates)):
            pair = (int(candidates[k].min()), int(candidates[k].max()))
            if pair not in filter_pairs:
                filter_pairs.add(pair)
                added += 1
                if added == count:
                    return
        raise AssertionError(f"scene {seed} has only {added} of {count} candidate filter pairs")

    first, second = np.nonzero(np.triu(overlap, 1))
    across = shape_aggregate[flat[first]] != shape_aggregate[flat[second]]
    draw(flat[first][across], flat[second][across], 30)
    plane_tops = box_upper[:2, 2]
    plane, member = np.nonzero(box_lower[flat][None, :, 2] <= plane_tops[:, None])
    draw(plane, flat[member], 5)
    if statics:
        statics = np.asarray(statics)
        touch = np.all(
            (box_lower[flat][:, None, :] <= box_upper[statics][None, :, :])
            & (box_upper[flat][:, None, :] >= box_lower[statics][None, :, :]),
            axis=2,
        )
        member, static = np.nonzero(touch)
        draw(flat[member], statics[static], 5)

    body_flags = np.zeros(body_count, dtype=np.int32)
    body_flags[:2] = _KINEMATIC
    return {
        "lower": np.asarray(shapes["lower"], dtype=np.float32),
        "upper": np.asarray(shapes["upper"], dtype=np.float32),
        "world": np.asarray(shapes["world"], dtype=np.int32),
        "group": np.asarray(shapes["group"], dtype=np.int32),
        "body": np.asarray(shapes["body"], dtype=np.int32),
        "flags": np.asarray(shapes["flags"], dtype=np.int32),
        "type": np.asarray(shapes["type"], dtype=np.int32),
        "scale": np.asarray(shapes["scale"], dtype=np.float32),
        "aggregate": np.asarray(shapes["aggregate"], dtype=np.int32),
        "body_flags": body_flags,
        "filter_pairs": np.asarray(sorted(filter_pairs), dtype=np.int32).reshape(-1, 2),
    }


def _aggregate_arrays(scene: dict[str, np.ndarray], device) -> dict[str, wp.array]:
    """Copy a scene's launch inputs to ``device``."""
    pairs = scene["filter_pairs"]
    return {
        "lower": wp.array(scene["lower"], dtype=wp.vec3, device=device),
        "upper": wp.array(scene["upper"], dtype=wp.vec3, device=device),
        "world": wp.array(scene["world"], dtype=wp.int32, device=device),
        "group": wp.array(scene["group"], dtype=wp.int32, device=device),
        "body": wp.array(scene["body"], dtype=wp.int32, device=device),
        "body_flags": wp.array(scene["body_flags"], dtype=wp.int32, device=device),
        "filter_pairs": wp.array(pairs, dtype=wp.vec2i, device=device) if len(pairs) else None,
    }


def _make_aggregate(scene: dict[str, np.ndarray], device, **kwargs) -> BroadPhaseAggregate:
    """Construct a :class:`BroadPhaseAggregate` from a scene's true aggregate ids."""
    return BroadPhaseAggregate(
        scene["world"],
        scene["body"],
        scene["aggregate"],
        scene["type"],
        scene["scale"],
        scene["flags"],
        scene["filter_pairs"],
        device=device,
        **kwargs,
    )


def _pair_set(pairs: np.ndarray, count: int) -> set[tuple[int, int]]:
    return {(int(a), int(b)) for a, b in pairs[:count]}


def _sorted_pairs(pairs: np.ndarray, count: int) -> np.ndarray:
    pairs = pairs[:count]
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]


def _displacement_array(displacement: np.ndarray | None, device) -> wp.array | None:
    return None if displacement is None else wp.array(displacement, dtype=wp.vec3, device=device)


def _nxn_pair_set(
    scene: dict[str, np.ndarray],
    device,
    *,
    include_static_kinematic_pairs: bool = True,
    displacement: np.ndarray | None = None,
) -> set[tuple[int, int]]:
    """Run :class:`BroadPhaseAllPairs` on a scene at the all-pairs capacity."""
    arrays = _aggregate_arrays(scene, device)
    shape_count = len(scene["world"])
    capacity = shape_count * (shape_count - 1) // 2
    pairs = wp.zeros(capacity, dtype=wp.vec2i, device=device)
    count = wp.zeros(1, dtype=wp.int32, device=device)
    nxn = BroadPhaseAllPairs(arrays["world"], shape_flags=scene["flags"], device=device)
    nxn.launch(
        arrays["lower"],
        arrays["upper"],
        None,
        arrays["group"],
        arrays["world"],
        shape_count,
        pairs,
        count,
        device=device,
        filter_pairs=arrays["filter_pairs"],
        shape_body=arrays["body"],
        body_flags=arrays["body_flags"],
        include_static_kinematic_pairs=include_static_kinematic_pairs,
        shape_displacement=_displacement_array(displacement, device),
    )
    n = int(count.numpy()[0])
    assert n <= capacity
    return _pair_set(pairs.numpy(), n)


def _launch_aggregate(
    bp: BroadPhaseAggregate,
    arrays: dict[str, wp.array],
    pairs: wp.array,
    count: wp.array,
    device,
    *,
    include_static_kinematic_pairs: bool = True,
    displacement: wp.array | None = None,
    limit: float | None = None,
) -> None:
    """Launch ``bp`` the way :class:`CollisionPipeline` launches ``sap``."""
    bp.launch(
        arrays["lower"],
        arrays["upper"],
        None,
        arrays["group"],
        arrays["world"],
        arrays["world"].shape[0],
        pairs,
        count,
        device=device,
        filter_pairs=arrays["filter_pairs"],
        skip_count_zero=True,
        shape_body=arrays["body"],
        body_flags=arrays["body_flags"],
        include_static_kinematic_pairs=include_static_kinematic_pairs,
        shape_displacement=displacement,
        sort_axis_displacement_limit=limit,
    )


def _aggregate_pair_set(
    bp: BroadPhaseAggregate,
    scene: dict[str, np.ndarray],
    device,
    *,
    capacity: int | None = None,
    include_static_kinematic_pairs: bool = True,
    displacement: np.ndarray | None = None,
    limit: float | None = None,
    arrays: dict[str, wp.array] | None = None,
) -> tuple[set[tuple[int, int]], int, np.ndarray]:
    """Launch ``bp`` on a scene and return the pair set, the published count and the raw buffer."""
    if arrays is None:
        arrays = _aggregate_arrays(scene, device)
    shape_count = len(scene["world"])
    if capacity is None:
        capacity = shape_count * (shape_count - 1) // 2
    pairs = wp.zeros(capacity, dtype=wp.vec2i, device=device)
    count = wp.zeros(1, dtype=wp.int32, device=device)
    _launch_aggregate(
        bp,
        arrays,
        pairs,
        count,
        device,
        include_static_kinematic_pairs=include_static_kinematic_pairs,
        displacement=_displacement_array(displacement, device),
        limit=limit,
    )
    n = int(count.numpy()[0])
    raw = pairs.numpy()
    return _pair_set(raw, n), n, raw


def _model_aabbs_and_nxn(model: newton.Model, state: newton.State, device):
    """Collide a model with an ``nxn`` pipeline; return its AABBs and its pair set."""
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    count = int(pipeline.broad_phase_pair_count.numpy()[0])
    pairs = _pair_set(pipeline.broad_phase_shape_pairs.numpy(), count)
    return pipeline.narrow_phase.shape_aabb_lower, pipeline.narrow_phase.shape_aabb_upper, pairs


def _model_aggregate_pair_set(bp: BroadPhaseAggregate, model: newton.Model, lower, upper, device):
    """Launch ``bp`` on a model's pipeline AABBs, as :class:`CollisionPipeline` launches ``sap``."""
    filter_pairs = model.shape_collision_filter_pairs_array()
    arrays = {
        "lower": lower,
        "upper": upper,
        "world": model.shape_world,
        "group": model.shape_collision_group,
        "body": model.shape_body,
        "body_flags": model.body_flags,
        "filter_pairs": wp.array(filter_pairs, dtype=wp.vec2i, device=device) if len(filter_pairs) else None,
    }
    capacity = model.shape_count * (model.shape_count - 1) // 2
    pairs = wp.zeros(capacity, dtype=wp.vec2i, device=device)
    count = wp.zeros(1, dtype=wp.int32, device=device)
    _launch_aggregate(bp, arrays, pairs, count, device)
    return _pair_set(pairs.numpy(), int(count.numpy()[0]))


def test_aggregate_matches_nxn_random(test, device):
    """E1: the aggregate pair set equals nxn's on random scenes, with and without static-kinematic pairs."""
    for seed in range(4):
        scene = _aggregate_scene(seed)
        bp = _make_aggregate(scene, device)
        for include in (True, False):
            with test.subTest(seed=seed, include_static_kinematic_pairs=include):
                nxn_set = _nxn_pair_set(scene, device, include_static_kinematic_pairs=include)
                agg_set, count, raw = _aggregate_pair_set(bp, scene, device, include_static_kinematic_pairs=include)
                test.assertEqual(agg_set, nxn_set)
                test.assertEqual(count, len(agg_set))
                test.assertTrue(np.all(raw[:count, 0] < raw[:count, 1]))
                test.assertEqual(bp.statistics()["pair_overflow"], 0)
                test.assertGreater(len(nxn_set), 1000)


def test_aggregate_kinematic_pairs_match_nxn(test, device):
    """Kinematic pairs within aggregates, across them and against planes follow nxn's immovable filter."""
    scene = dict(_aggregate_scene(0))
    members = scene["aggregate"] >= 0
    kinematic = np.unique(scene["body"][members & (scene["aggregate"] % 2 == 0)])
    scene["body_flags"] = np.zeros_like(scene["body_flags"])
    scene["body_flags"][kinematic] = _KINEMATIC
    bp = _make_aggregate(scene, device)
    nxn_sets = {}
    for include in (True, False):
        with test.subTest(include_static_kinematic_pairs=include):
            nxn_sets[include] = _nxn_pair_set(scene, device, include_static_kinematic_pairs=include)
            agg_set, count, _ = _aggregate_pair_set(bp, scene, device, include_static_kinematic_pairs=include)
            test.assertEqual(agg_set, nxn_sets[include])
            test.assertEqual(count, len(agg_set))
    dropped = nxn_sets[True] - nxn_sets[False]
    aggregate = scene["aggregate"]
    test.assertTrue(any(aggregate[i] >= 0 and aggregate[i] != aggregate[j] for i, j in dropped))
    test.assertTrue(any(aggregate[i] >= 0 and aggregate[i] == aggregate[j] for i, j in dropped))
    test.assertTrue(any(i < 2 for i, _ in dropped))


def test_aggregate_group_rules_match_host_reference(test, device):
    """Group summaries keep every member pair the host group rules allow, and only those."""
    groups = (-1, -1, -1, -2, 1, 0, 0, 1, 2, -1, 3)
    aggregates = (0, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5)
    aggregate_worlds = (-1, -1, 0, 0, 0, -1)
    shape_count = len(groups)
    collide = int(ShapeFlags.COLLIDE_SHAPES)
    scene = {
        "lower": np.zeros((shape_count, 3), dtype=np.float32),
        "upper": np.ones((shape_count, 3), dtype=np.float32),
        "world": np.asarray([aggregate_worlds[a] for a in aggregates], dtype=np.int32),
        "group": np.asarray(groups, dtype=np.int32),
        "body": np.arange(shape_count, dtype=np.int32),
        "flags": np.full(shape_count, collide, dtype=np.int32),
        "type": np.full(shape_count, int(newton.GeoType.BOX), dtype=np.int32),
        "scale": np.full((shape_count, 3), 0.5, dtype=np.float32),
        "aggregate": np.asarray(aggregates, dtype=np.int32),
        "body_flags": np.zeros(shape_count, dtype=np.int32),
        "filter_pairs": np.zeros((0, 2), dtype=np.int32),
    }
    expected = {
        (i, j)
        for i in range(shape_count)
        for j in range(i + 1, shape_count)
        if test_world_and_group_pair(
            int(scene["world"][i]), int(scene["world"][j]), int(scene["group"][i]), int(scene["group"][j])
        )
    }
    agg_set, count, _ = _aggregate_pair_set(_make_aggregate(scene, device), scene, device)
    test.assertEqual(agg_set, expected)
    test.assertEqual(count, len(expected))
    for absent in ((0, 1), (0, 2), (1, 2)):
        test.assertNotIn(absent, agg_set)
    for present in ((0, 3), (4, 7)):
        test.assertIn(present, agg_set)

    # Coincident zero-extent boxes only touch; the padded union boxes keep SAP's
    # half-open sweep from dropping the aggregate pairs between them.
    point = np.full((shape_count, 3), 0.5, dtype=np.float32)
    touching = dict(scene, lower=point, upper=point)
    test.assertEqual(_aggregate_pair_set(_make_aggregate(touching, device), touching, device)[0], expected)


def test_aggregate_runtime_group_change_matches_nxn(test, device):
    """Groups reassigned after construction are honoured at both levels without a rebuild."""
    scene = _aggregate_scene(2)
    bp = _make_aggregate(scene, device)
    arrays = _aggregate_arrays(scene, device)
    agg_set, _, _ = _aggregate_pair_set(bp, scene, device, arrays=arrays)
    test.assertEqual(agg_set, _nxn_pair_set(scene, device))

    changed = dict(scene)
    changed["group"] = scene["group"].copy()
    members = scene["aggregate"] >= 0
    changed["group"][members] = np.where(scene["aggregate"][members] % 2 == 0, 5, -3)
    arrays["group"].assign(changed["group"])
    agg_set, count, _ = _aggregate_pair_set(bp, changed, device, arrays=arrays)
    test.assertEqual(agg_set, _nxn_pair_set(changed, device))
    test.assertEqual(count, len(agg_set))


def _two_world_chain_model(device) -> newton.Model:
    """A ground plane, two worlds of a 3-link revolute chain and a box body, and a static heightfield."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for _ in range(2):
        builder.begin_world()
        joints = []
        parent = -1
        for k in range(3):
            link = builder.add_link(xform=wp.transform(wp.vec3(0.4 * k, 0.0, 0.15), wp.quat_identity()))
            builder.add_shape_capsule(link, radius=0.1, half_height=0.15)
            if k == 1:
                builder.add_shape_sphere(link, radius=0.12)
            if parent >= 0:
                parent_xform = wp.transform(wp.vec3(0.4, 0.0, 0.0), wp.quat_identity())
            else:
                parent_xform = wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity())
            joints.append(builder.add_joint_revolute(parent, link, parent_xform=parent_xform, axis=newton.Axis.Y))
            parent = link
        builder.add_articulation(joints)
        box = builder.add_body(xform=wp.transform(wp.vec3(0.4, 0.25, 0.15), wp.quat_identity()))
        builder.add_shape_box(box, hx=0.15, hy=0.15, hz=0.15)
        builder.add_shape_sphere(box, radius=0.1, cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False))
        builder.end_world()
    heightfield = newton.Heightfield(data=np.zeros((4, 4), dtype=np.float32), nrow=4, ncol=4, hx=1.0, hy=1.0)
    builder.add_shape_heightfield(
        xform=wp.transform(wp.vec3(0.4, 0.0, 0.0), wp.quat_identity()), heightfield=heightfield
    )
    return builder.finalize(device=device)


def test_aggregate_membership_from_model(test, device):
    """Aggregates follow articulations and bodies; planes, heightfields and visual shapes stay out."""
    model = _two_world_chain_model(device)
    shape_type = model.shape_type.numpy()
    shape_flags = model.shape_flags.numpy()
    aggregate = broad_phase_aggregate.infer_shape_aggregates(model)
    test.assertEqual(int(aggregate.max()) + 1, 4)

    plane = np.flatnonzero(shape_type == int(newton.GeoType.PLANE))
    heightfield = np.flatnonzero(shape_type == int(newton.GeoType.HFIELD))
    visual = np.flatnonzero((shape_flags & int(ShapeFlags.COLLIDE_SHAPES)) == 0)
    test.assertEqual((len(plane), len(heightfield), len(visual)), (1, 1, 2))
    for shape in (*plane, *heightfield, *visual):
        test.assertEqual(aggregate[shape], -1)

    shape_body = model.shape_body.numpy()
    body_articulation = np.full(model.body_count, -1, dtype=np.int32)
    body_articulation[model.joint_child.numpy()] = model.joint_articulation.numpy()
    first_shape = {}
    for shape in np.flatnonzero(aggregate >= 0):
        first_shape.setdefault(int(aggregate[shape]), int(shape))
    test.assertEqual(sorted(first_shape.values()), [first_shape[a] for a in range(4)])
    for articulation in range(model.articulation_count):
        chain = np.flatnonzero(
            (shape_body >= 0)
            & (body_articulation[np.maximum(shape_body, 0)] == articulation)
            & ((shape_flags & int(ShapeFlags.COLLIDE_SHAPES)) != 0)
        )
        if len(chain) == 4:
            test.assertEqual(len(set(aggregate[chain].tolist())), 1)

    split = broad_phase_aggregate.infer_shape_aggregates(model, max_aggregate_size=2)
    test.assertEqual(int(split.max()) + 1, 6)

    bp = BroadPhaseAggregate.from_model(model)
    stats = bp.statistics()
    test.assertEqual(stats["aggregates"], 4)
    test.assertEqual(stats["largest_aggregate"], 4)
    test.assertEqual(stats["unbounded_shapes"], 2)

    state = model.state()
    lower, upper, nxn_set = _model_aabbs_and_nxn(model, state, device)
    test.assertGreater(len(nxn_set), 0)
    test.assertEqual(_model_aggregate_pair_set(bp, model, lower, upper, device), nxn_set)


def test_aggregate_finite_statics_from_model_match_nxn(test, device):
    """A model's finite static box goes to the static grid and pairs with the body it overlaps, as under nxn."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    box = builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(0.3, 0.0, 0.5), wp.quat_identity()), hx=0.2, hy=0.2, hz=0.2
    )
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()))
    sphere = builder.add_shape_sphere(body, radius=0.2)
    model = builder.finalize(device=device)
    bp = BroadPhaseAggregate.from_model(model)
    lower, upper, nxn_set = _model_aabbs_and_nxn(model, model.state(), device)
    bp.build_static(lower, upper)
    test.assertEqual(bp.statistics()["static_shapes"], 1)
    test.assertIn((min(box, sphere), max(box, sphere)), nxn_set)
    test.assertEqual(_model_aggregate_pair_set(bp, model, lower, upper, device), nxn_set)


def _prismatic_model(device) -> tuple[newton.Model, int, int, int]:
    """Build a ground plane, a free root capsule with a prismatic (x) child built at x = 3.0, and a body at x = 3.1.

    Returns:
        The model and the root, child and other capsule's shape indices.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    root = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
    root_shape = builder.add_shape_capsule(root, radius=0.1, half_height=0.3)
    child = builder.add_link(xform=wp.transform(wp.vec3(3.0, 0.0, 0.0), wp.quat_identity()))
    child_shape = builder.add_shape_capsule(child, radius=0.1, half_height=0.3)
    root_joint = builder.add_joint_free(root)
    slide = builder.add_joint_prismatic(
        root, child, parent_xform=wp.transform(wp.vec3(3.0, 0.0, 0.0), wp.quat_identity()), axis=newton.Axis.X
    )
    builder.add_articulation([root_joint, slide])
    other = builder.add_body(xform=wp.transform(wp.vec3(3.1, 0.0, 0.0), wp.quat_identity()))
    other_shape = builder.add_shape_capsule(other, radius=0.1, half_height=0.3)
    return builder.finalize(device=device), root_shape, child_shape, other_shape


def test_aggregate_prismatic_articulation_matches_nxn(test, device):
    """Under the sweep, a prismatic articulation slid 3 m stays one aggregate whose box spans both links."""
    model, root_shape, child_shape, other_shape = _prismatic_model(device)

    aggregate = broad_phase_aggregate.infer_shape_aggregates(model)
    test.assertEqual(aggregate[root_shape], aggregate[child_shape])
    test.assertNotEqual(aggregate[child_shape], aggregate[other_shape])

    bp = BroadPhaseAggregate.from_model(model, main="sap")
    lower, upper, nxn_set = _model_aabbs_and_nxn(model, model.state(), device)
    agg_set = _model_aggregate_pair_set(bp, model, lower, upper, device)
    test.assertIn((child_shape, other_shape), agg_set)
    test.assertEqual(agg_set, nxn_set)


def _host_swept_union(scene: dict[str, np.ndarray], displacement: np.ndarray, limit: float):
    """Union of each aggregate's member boxes swept by the per-axis clamped displacement."""
    clamped = np.clip(displacement, -limit, limit).astype(np.float32)
    lower = scene["lower"] + np.minimum(clamped, np.float32(0.0))
    upper = scene["upper"] + np.maximum(clamped, np.float32(0.0))
    members = scene["aggregate"] >= 0
    count = int(scene["aggregate"].max()) + 1
    union_lower = np.full((count, 3), np.inf, dtype=np.float32)
    union_upper = np.full((count, 3), -np.inf, dtype=np.float32)
    np.minimum.at(union_lower, scene["aggregate"][members], lower[members])
    np.maximum.at(union_upper, scene["aggregate"][members], upper[members])
    return union_lower, union_upper


def test_aggregate_swept_matches_nxn_within_clamp(test, device):
    """Swept union boxes keep every swept pair while the displacements stay inside the clamp."""
    scene = _aggregate_scene(1)
    rng = np.random.default_rng(101)
    displacement = rng.uniform(-0.05, 0.05, (len(scene["world"]), 3)).astype(np.float32)
    displacement[scene["body"] < 0] = 0.0
    bp = _make_aggregate(scene, device)
    agg_set, count, _ = _aggregate_pair_set(bp, scene, device, displacement=displacement, limit=0.1)
    nxn_set = _nxn_pair_set(scene, device, displacement=displacement)
    test.assertGreater(len(nxn_set - _nxn_pair_set(scene, device)), 0)
    test.assertEqual(agg_set, nxn_set)
    test.assertEqual(count, len(agg_set))

    # The union boxes are the members' boxes swept by the clamped displacement,
    # also when the clamp binds, padded outward by about 1e-6 of their largest
    # coordinate.
    for limit in (0.1, 0.02):
        _aggregate_pair_set(bp, scene, device, displacement=displacement, limit=limit)
        union_lower, union_upper = _host_swept_union(scene, displacement, limit)
        lower, upper = bp.aggregate_lower.numpy(), bp.aggregate_upper.numpy()
        np.testing.assert_array_less(lower, union_lower)
        np.testing.assert_array_less(union_upper, upper)
        tolerance = 2.0e-6 * max(1.0, float(np.abs(np.concatenate((union_lower, union_upper))).max()))
        np.testing.assert_allclose(lower, union_lower, rtol=0.0, atol=tolerance)
        np.testing.assert_allclose(upper, union_upper, rtol=0.0, atol=tolerance)


def test_aggregate_two_instances_agree(test, device):
    """E4: two instances give the same sorted pairs, counts and demand over 50 moving frames."""
    scene = _aggregate_scene(7)
    first = _make_aggregate(scene, device)
    second = _make_aggregate(scene, device)
    arrays = _aggregate_arrays(scene, device)
    shape_count = len(scene["world"])
    capacity = shape_count * (shape_count - 1) // 2
    buffers = [
        (wp.zeros(capacity, dtype=wp.vec2i, device=device), wp.zeros(1, dtype=wp.int32, device=device))
        for _ in range(2)
    ]
    rng = np.random.default_rng(7)
    lower = scene["lower"].copy()
    upper = scene["upper"].copy()
    for frame in range(50):
        step = rng.uniform(-0.05, 0.05, (shape_count, 3)).astype(np.float32)
        lower += step
        upper += step
        arrays["lower"].assign(lower)
        arrays["upper"].assign(upper)
        results = []
        for bp, (pairs, count) in zip((first, second), buffers, strict=True):
            _launch_aggregate(bp, arrays, pairs, count, device)
            n = int(count.numpy()[0])
            results.append((_sorted_pairs(pairs.numpy(), n), n, bp.statistics()["pair_demand"]))
        with test.subTest(frame=frame):
            test.assertTrue(np.array_equal(results[0][0], results[1][0]))
            test.assertEqual(results[0][1], results[1][1])
            test.assertEqual(results[0][2], results[1][2])


def test_aggregate_pair_overflow_is_reported(test, device):
    """E6: overflow clamps the count, keeps the true demand and writes nothing past the buffer."""
    scene = _aggregate_scene(0)
    nxn_set = _nxn_pair_set(scene, device)
    n = len(nxn_set)
    bp = _make_aggregate(scene, device)
    arrays = _aggregate_arrays(scene, device)
    buffer = wp.full(n, wp.vec2i(-7, -7), dtype=wp.vec2i, device=device)
    count = wp.zeros(1, dtype=wp.int32, device=device)
    _launch_aggregate(bp, arrays, buffer[: n // 2], count, device)
    stats = bp.statistics()
    test.assertEqual(int(count.numpy()[0]), n // 2)
    test.assertEqual(stats["pair_overflow"], 1)
    test.assertEqual(stats["pair_demand"], n)
    test.assertEqual(stats["pair_demand_max"], n)
    raw = buffer.numpy()
    test.assertTrue(np.all(raw[n // 2 :] == -7))
    written = _pair_set(raw, n // 2)
    test.assertEqual(len(written), n // 2)
    test.assertTrue(written <= nxn_set)


def test_aggregate_level_overflow_is_reported(test, device):
    """An aggregate-pair overflow is flagged and latched, and drops pairs rather than inventing them."""
    scene = _aggregate_scene(0)
    nxn_set = _nxn_pair_set(scene, device)
    small = _make_aggregate(scene, device, aggregate_pairs_max=4)
    large = _make_aggregate(scene, device, aggregate_pairs_max=100000)
    small_set, _, _ = _aggregate_pair_set(small, scene, device)
    large_set, _, _ = _aggregate_pair_set(large, scene, device)
    small_stats = small.statistics()
    large_stats = large.statistics()
    test.assertGreater(large_stats["aggregate_pair_count"], 4)
    test.assertEqual(small_stats["aggregate_pair_overflow"], 1)
    test.assertEqual(small_stats["aggregate_pair_demand_max"], large_stats["aggregate_pair_count"])
    test.assertTrue(small_set <= nxn_set)
    test.assertEqual(large_set, nxn_set)
    test.assertEqual(small_stats["pair_overflow"], 0)


def test_aggregate_knobs_agree(test, device):
    """Every kernel variant B2 and B3 measure, on the default grid and on the sweep, gives the same pair set.

    The reference is a direct construction without a reach, which takes the sweep.
    """
    scene = _aggregate_scene(3)
    reference_bp = _make_aggregate(scene, device)
    test.assertEqual(reference_bp.statistics()["main_level"], "sap")
    reference, _, _ = _aggregate_pair_set(reference_bp, scene, device)
    for knobs in (
        {},
        {"union": "warp"},
        {"self_filter": "search"},
        {"emission": "tile"},
        {"ordered": True},
        {"ordered": True, "self_filter": "search"},
        {"main": "sap"},
        {"main": "sap", "union": "warp", "self_filter": "search"},
        {"main": "sap", "emission": "tile"},
        {"main": "sap", "ordered": True},
    ):
        with test.subTest(knobs=knobs):
            with mock.patch.dict(broad_phase_aggregate._KNOBS, knobs):
                bp = _make_aggregate(scene, device, shape_reach=_grid_reach(scene))
            test.assertEqual(bp.statistics()["main_level"], knobs.get("main", "grid"))
            agg_set, count, _ = _aggregate_pair_set(bp, scene, device)
            test.assertEqual(agg_set, reference)
            test.assertEqual(count, len(reference))


def test_aggregate_capture_replay_matches_eager(test, device):
    """A captured launch replays to the eager pair set and count; an ordered one to the eager list."""
    scene = _aggregate_scene(4)
    arrays = _aggregate_arrays(scene, device)
    shape_count = len(scene["world"])
    for main, ordered in (("sap", False), ("sap", True), ("grid", False), ("grid", True)):
        with test.subTest(main=main, ordered=ordered):
            bp = _make_aggregate(scene, device, main=main, shape_reach=_grid_reach(scene), ordered=ordered)
            pairs = wp.zeros(shape_count * (shape_count - 1) // 2, dtype=wp.vec2i, device=device)
            count = wp.zeros(1, dtype=wp.int32, device=device)
            _launch_aggregate(bp, arrays, pairs, count, device)
            eager_count = int(count.numpy()[0])
            eager = pairs.numpy()[:eager_count].copy()

            with wp.ScopedCapture(device=device) as capture:
                _launch_aggregate(bp, arrays, pairs, count, device)
            pairs.fill_(wp.vec2i(-7, -7))
            count.fill_(12345)
            wp.capture_launch(capture.graph)
            replay_count = int(count.numpy()[0])
            replay = pairs.numpy()[:replay_count]
            test.assertEqual(replay_count, eager_count)
            test.assertEqual(_pair_set(replay, replay_count), _pair_set(eager, eager_count))
            if ordered:
                test.assertTrue(np.array_equal(replay, eager))
            test.assertEqual(bp.statistics()["pair_demand"], replay_count)


def test_aggregate_verify_capture_replay(test, device):
    """The device-side warnings replay from a captured graph, and the device flag silences them without a recapture.

    A grid launch with half the pair buffer is captured with ``verify=True``:
    each replay prints the overflow once; after ``verify_flag.zero_()`` the
    same graph prints nothing.
    """
    scene = _aggregate_scene(4)
    arrays = _aggregate_arrays(scene, device)
    n = len(_nxn_pair_set(scene, device))
    bp = _make_aggregate(scene, device, main="grid", shape_reach=_grid_reach(scene), verify=True)
    pairs = wp.zeros(n // 2, dtype=wp.vec2i, device=device)
    count = wp.zeros(1, dtype=wp.int32, device=device)
    output = _device_output(lambda: _launch_aggregate(bp, arrays, pairs, count, device))
    test.assertEqual(output.count("candidate pair buffer overflowed"), 1, output)
    with wp.ScopedCapture(device=device) as capture:
        _launch_aggregate(bp, arrays, pairs, count, device)

    def replay():
        wp.capture_launch(capture.graph)
        wp.capture_launch(capture.graph)

    output = _device_output(replay)
    test.assertEqual(output.count(f"candidate pair buffer overflowed {n} > {n // 2}"), 2, output)
    test.assertEqual(int(count.numpy()[0]), n // 2)
    bp.verify_flag.zero_()
    test.assertEqual(_device_output(replay), "")
    test.assertEqual(bp.statistics()["pair_overflow"], 1)


def _static_plan(bp: BroadPhaseAggregate) -> tuple[np.ndarray, np.ndarray]:
    """Replay the static stage's choice per aggregate on the host, after a launch.

    Returns:
        Whether each aggregate walks its grid cells (else it lists the statics in
        view), and its lowest cell, shape [aggregate_count, 3].
    """
    inv = np.float32(1.0) / np.float32(bp.static_cell_size)
    low = np.floor(bp.aggregate_lower.numpy() * inv)
    high = np.floor(bp.aggregate_upper.numpy() * inv)
    cells = np.prod(high - low + np.float32(1.0), axis=1)
    world = bp.aggregate_world.numpy()
    start = bp.static_start.numpy().astype(np.int64)
    segment = np.maximum(world, -1) + 1
    in_view = (start[1] - start[0]) + np.where(world >= 0, start[segment + 1] - start[segment], 0)
    walk = cells * np.where(world >= 0, 2, 1) <= in_view
    limit = float(1 << 29)
    return walk, np.clip(low, -limit, limit).astype(np.int64)


def _documented_order(
    scene: dict[str, np.ndarray], pairs: set[tuple[int, int]], bp: BroadPhaseAggregate | None = None
) -> np.ndarray:
    """Sort a pair set into the documented order of ``BroadPhaseAggregate(ordered=True)``.

    The aggregates come first, by id: their member pairs (i < j) in member order,
    then each unbounded shape in CSR order (by world, world -1 first) against the
    members in member order. With ``bp``, its CSR (which holds the statics too
    large for its grid) and its static grid are used, and the aggregates' static
    pairs follow, by aggregate: the statics in the static stage's visiting order
    (by test cell, z then y then x, when the aggregate walks the grid; world -1
    before the aggregate's world; then by shape), each against the members in
    member order. The aggregate pairs (a, b), a < b, follow in key order, each
    in (member of a) x (member of b) order. Members are in shape order within
    their aggregate.
    """
    aggregate = scene["aggregate"]
    local = np.full(len(aggregate), -1, dtype=np.int64)
    for a in range(int(aggregate.max()) + 1):
        shapes = np.flatnonzero(aggregate == a)
        local[shapes] = np.arange(len(shapes))
    if bp is None:
        unbounded = broad_phase_aggregate.unbounded_shape_mask(
            scene["type"], scene["scale"], scene["body"], scene["flags"]
        )
        csr = sorted(np.flatnonzero(unbounded).tolist(), key=lambda u: (int(scene["world"][u]), u))
    else:
        csr = bp.unbounded_shapes.numpy().tolist()
    csr_position = {u: k for k, u in enumerate(csr)}
    grid = {}
    if bp is not None and bp.static_cell_size is not None:
        walk, aggregate_low = _static_plan(bp)
        static_low = bp.static_cell_lower.numpy()
        for g, u in enumerate(bp.static_shapes.numpy().tolist()):
            grid[u] = (int(scene["world"][u] >= 0), static_low[g])

    def key(pair: tuple[int, int]) -> tuple[int, ...]:
        s, t = pair
        a_s, a_t = int(aggregate[s]), int(aggregate[t])
        if a_s >= 0 and a_s == a_t:
            return (0, a_s, 0, int(local[s]), int(local[t]))
        if a_s < 0 or a_t < 0:
            member, u = (t, s) if a_s < 0 else (s, t)
            a = int(aggregate[member])
            if u in grid:
                own_world, low = grid[u]
                cell = np.maximum(aggregate_low[a], low) if walk[a] else np.zeros(3, dtype=np.int64)
                return (1, a, int(cell[2]), int(cell[1]), int(cell[0]), own_world, u, int(local[member]))
            return (0, a, 1, csr_position[u], int(local[member]))
        first, second = (s, t) if a_s < a_t else (t, s)
        return (2, int(aggregate[first]), int(aggregate[second]), int(local[first]), int(local[second]))

    return np.asarray(sorted(pairs, key=key), dtype=np.int32).reshape(-1, 2)


def test_aggregate_ordered_lists_are_identical(test, device):
    """Two ordered instances publish one list over 50 moving frames: the unordered set in the documented order."""
    scene = _aggregate_scene(5)
    instances = (
        _make_aggregate(scene, device, ordered=True),
        _make_aggregate(scene, device, ordered=True),
        _make_aggregate(scene, device, ordered=False),
    )
    test.assertEqual([bp.ordered for bp in instances], [True, True, False])
    arrays = _aggregate_arrays(scene, device)
    shape_count = len(scene["world"])
    capacity = shape_count * (shape_count - 1) // 2
    buffers = [
        (wp.zeros(capacity, dtype=wp.vec2i, device=device), wp.zeros(1, dtype=wp.int32, device=device))
        for _ in instances
    ]
    rng = np.random.default_rng(5)
    lower = scene["lower"].copy()
    upper = scene["upper"].copy()
    for frame in range(50):
        step = rng.uniform(-0.05, 0.05, (shape_count, 3)).astype(np.float32)
        lower += step
        upper += step
        arrays["lower"].assign(lower)
        arrays["upper"].assign(upper)
        # Odd frames sweep the boxes by the step, which takes the moving tests.
        displacement = _displacement_array(step, device) if frame % 2 else None
        results = []
        for bp, (pairs, count) in zip(instances, buffers, strict=True):
            _launch_aggregate(bp, arrays, pairs, count, device, displacement=displacement, limit=0.1)
            n = int(count.numpy()[0])
            results.append((pairs.numpy()[:n], bp.statistics()))
        (first, first_stats), (second, _), (unordered, unordered_stats) = results
        unordered_set = _pair_set(unordered, len(unordered))
        with test.subTest(frame=frame):
            test.assertTrue(np.array_equal(first, second))
            test.assertEqual(_pair_set(first, len(first)), unordered_set)
            test.assertEqual(len(first), len(unordered_set))
            test.assertEqual(first_stats["pair_demand"], unordered_stats["pair_demand"])
            test.assertEqual(first_stats["pair_overflow"], 0)
            test.assertTrue(np.array_equal(first, _documented_order(scene, unordered_set)))
    test.assertGreater(len(unordered_set), 1000)


def test_aggregate_ordered_list_order(test, device):
    """The group-rules scene publishes exactly the list of the documented order, with either aggregate numbering."""
    groups = (-1, -1, -1, -2, 1, 0, 0, 1, 2, -1, 3)
    aggregates = np.asarray((0, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5), dtype=np.int32)
    aggregate_worlds = (-1, -1, 0, 0, 0, -1)
    shape_count = len(groups)
    collide = int(ShapeFlags.COLLIDE_SHAPES)
    scene = {
        "lower": np.zeros((shape_count, 3), dtype=np.float32),
        "upper": np.ones((shape_count, 3), dtype=np.float32),
        "world": np.asarray([aggregate_worlds[a] for a in aggregates], dtype=np.int32),
        "group": np.asarray(groups, dtype=np.int32),
        "body": np.arange(shape_count, dtype=np.int32),
        "flags": np.full(shape_count, collide, dtype=np.int32),
        "type": np.full(shape_count, int(newton.GeoType.BOX), dtype=np.int32),
        "scale": np.full((shape_count, 3), 0.5, dtype=np.float32),
        "aggregate": aggregates,
        "body_flags": np.zeros(shape_count, dtype=np.int32),
        "filter_pairs": np.zeros((0, 2), dtype=np.int32),
    }
    # Self items (3, 4) and (9, 10) first, then the aggregate pairs (0, 2),
    # (0, 4), (0, 5), (1, 2), (1, 4), (1, 5), (2, 4), (2, 5) and (4, 5); the
    # other aggregate pairs have no member pair that passes the group rules.
    expected = [
        (3, 4),
        (9, 10),
        (0, 3), (0, 4), (1, 3), (1, 4),
        (0, 7), (0, 8), (1, 7), (1, 8),
        (0, 10), (1, 10),
        (2, 3), (2, 4),
        (2, 7), (2, 8),
        (2, 10),
        (3, 7), (3, 8), (4, 7),
        (3, 9), (3, 10), (4, 9),
        (7, 9), (8, 9),
    ]  # fmt: skip
    # Numbered in reverse, aggregate a's members follow b's in shape order, so
    # the member order (rows of a) differs from the sorted pair order.
    reversed_scene = dict(scene, aggregate=(5 - aggregates).astype(np.int32))
    expected_reversed = [
        (9, 10),
        (3, 4),
        (7, 9), (8, 9),
        (3, 9), (4, 9), (3, 10),
        (2, 10),
        (0, 10), (1, 10),
        (3, 7), (4, 7), (3, 8),
        (2, 7), (2, 8),
        (0, 7), (1, 7), (0, 8), (1, 8),
        (2, 3), (2, 4),
        (0, 3), (1, 3), (0, 4), (1, 4),
    ]  # fmt: skip
    for name, case, order in (("ids", scene, expected), ("reversed ids", reversed_scene, expected_reversed)):
        with test.subTest(case=name):
            bp = _make_aggregate(case, device, ordered=True)
            pair_set, count, raw = _aggregate_pair_set(bp, case, device)
            test.assertEqual(raw[:count].tolist(), [list(pair) for pair in order])
            test.assertTrue(np.array_equal(raw[:count], _documented_order(case, pair_set)))
            # With every aggregate-pair slot used, no sentinel slot ends the item list.
            stored = bp.statistics()["aggregate_pair_count"]
            full = _make_aggregate(case, device, ordered=True, aggregate_pairs_max=stored)
            _, count, raw = _aggregate_pair_set(full, case, device)
            test.assertEqual(raw[:count].tolist(), [list(pair) for pair in order])
            stats = full.statistics()
            test.assertEqual(stats["aggregate_pair_overflow"], 0)
            test.assertEqual(stats["pair_demand"], len(order))


def test_aggregate_ordered_overflow_keeps_a_prefix(test, device):
    """An overflowing ordered list keeps exactly the first entries of the uncapped list."""
    scene = _aggregate_scene(0)
    bp = _make_aggregate(scene, device, ordered=True)
    full_set, n, raw = _aggregate_pair_set(bp, scene, device)
    full = raw[:n].copy()
    test.assertEqual(n, len(full_set))
    arrays = _aggregate_arrays(scene, device)
    buffer = wp.full(n, wp.vec2i(-7, -7), dtype=wp.vec2i, device=device)
    count = wp.zeros(1, dtype=wp.int32, device=device)
    _launch_aggregate(bp, arrays, buffer[: n // 2], count, device)
    stats = bp.statistics()
    test.assertEqual(int(count.numpy()[0]), n // 2)
    test.assertEqual(stats["pair_overflow"], 1)
    test.assertEqual(stats["pair_demand"], n)
    raw = buffer.numpy()
    test.assertTrue(np.array_equal(raw[: n // 2], full[: n // 2]))
    test.assertTrue(np.all(raw[n // 2 :] == -7))


def test_aggregate_ordered_bound_raises(test, device):
    """Construction refuses an ordered list whose offsets could pass the int32 range."""
    scene = _aggregate_scene(0)
    test.assertEqual(int(np.bincount(scene["aggregate"][scene["aggregate"] >= 0]).max()), 40)
    with test.assertRaisesRegex(ValueError, "ordered"):
        _make_aggregate(scene, device, ordered=True, aggregate_pairs_max=2**26)


def _build_static(bp: BroadPhaseAggregate, arrays: dict[str, wp.array]) -> None:
    """Bin a scene's static boxes, as :meth:`CollisionPipeline.refresh_static_shapes` does with the model's."""
    bp.build_static(arrays["lower"], arrays["upper"])


def _box_scene(shapes, filter_pairs=()) -> dict[str, np.ndarray]:
    """A scene of boxes, each ``(lower, upper, world, group, body, aggregate)``; body -1 is a finite static box."""
    lower = np.asarray([shape[0] for shape in shapes], dtype=np.float32)
    upper = np.asarray([shape[1] for shape in shapes], dtype=np.float32)
    body = np.asarray([shape[4] for shape in shapes], dtype=np.int32)
    return {
        "lower": lower,
        "upper": upper,
        "world": np.asarray([shape[2] for shape in shapes], dtype=np.int32),
        "group": np.asarray([shape[3] for shape in shapes], dtype=np.int32),
        "body": body,
        "flags": np.full(len(shapes), int(ShapeFlags.COLLIDE_SHAPES), dtype=np.int32),
        "type": np.full(len(shapes), int(newton.GeoType.BOX), dtype=np.int32),
        "scale": 0.5 * (upper - lower),
        "aggregate": np.asarray([shape[5] for shape in shapes], dtype=np.int32),
        "body_flags": np.zeros(int(body.max()) + 1, dtype=np.int32),
        "filter_pairs": np.asarray(sorted(filter_pairs), dtype=np.int32).reshape(-1, 2),
    }


def _static_pairs(scene: dict[str, np.ndarray], pairs: set[tuple[int, int]]) -> tuple[int, int]:
    """Count the pairs with one static (body -1) shape and those with two."""
    static = scene["body"] < 0
    one = sum(1 for i, j in pairs if static[i] != static[j])
    two = sum(1 for i, j in pairs if static[i] and static[j])
    return one, two


def test_aggregate_statics_match_nxn_random(test, device):
    """Finite static boxes in the static grid give nxn's pair set, each pair once and none static-static."""
    for seed in range(3):
        scene = _aggregate_scene(seed, static_boxes=60)
        bp = _make_aggregate(scene, device)
        arrays = _aggregate_arrays(scene, device)
        _build_static(bp, arrays)
        for include in (True, False):
            with test.subTest(seed=seed, include_static_kinematic_pairs=include):
                nxn_set = _nxn_pair_set(scene, device, include_static_kinematic_pairs=include)
                agg_set, count, raw = _aggregate_pair_set(
                    bp, scene, device, include_static_kinematic_pairs=include, arrays=arrays
                )
                test.assertEqual(agg_set, nxn_set)
                test.assertEqual(count, len(agg_set))
                test.assertTrue(np.all(raw[:count, 0] < raw[:count, 1]))
                one, two = _static_pairs(scene, agg_set)
                test.assertEqual(two, 0)
                test.assertGreater(one, 0)
                stats = bp.statistics()
                test.assertEqual(stats["static_shapes"], 60)
                test.assertEqual(stats["static_entry_overflow"], 0)
                test.assertEqual(stats["pair_overflow"], 0)


def test_aggregate_statics_grid_variants_match_nxn(test, device):
    """Cell sizes that make the aggregates walk the grid or list the statics, or that make statics unbounded,
    world -1 aggregates with world -1 statics, kinematic bodies and a two-bucket table all give nxn's set."""
    kinematic = dict(_aggregate_scene(0, static_boxes=60))
    members = kinematic["aggregate"] >= 0
    kinematic["body_flags"] = np.zeros_like(kinematic["body_flags"])
    kinematic["body_flags"][np.unique(kinematic["body"][members & (kinematic["aggregate"] % 2 == 0)])] = _KINEMATIC
    two_buckets = mock.patch.object(broad_phase_aggregate, "_static_table_size", lambda capacity: 2)
    # Three slabs 200 m wide (in worlds -1, 0 and 1) exceed 64 derived cells.
    slabs = _aggregate_scene(0, static_boxes=60)
    slab = np.zeros(len(slabs["world"]), dtype=bool)
    slab[np.flatnonzero(slabs["body"] < 0)[2:5]] = True
    slabs = _moved_statics(slabs, shapes=slab, grow=100.0)
    cases = (
        ("derived cells", _aggregate_scene(0, static_boxes=60), {}, None),
        ("4 m cells", _aggregate_scene(0, static_boxes=60), {"static_cell_size": 4.0}, None),
        ("0.3 m cells", _aggregate_scene(0, static_boxes=60), {"static_cell_size": 0.3}, None),
        ("unbounded statics", slabs, {}, None),
        ("world -1 statics", _aggregate_scene(1, static_boxes=60, global_statics=True), {}, None),
        ("kinematic", kinematic, {}, None),
        ("two buckets", _aggregate_scene(2, static_boxes=60), {}, two_buckets),
    )
    walked = 0
    listed = 0
    for name, scene, kwargs, patch in cases:
        bp = _make_aggregate(scene, device, **kwargs)
        arrays = _aggregate_arrays(scene, device)
        if patch is None:
            _build_static(bp, arrays)
        else:
            with patch:
                _build_static(bp, arrays)
        nxn_sets = {}
        for include in (True, False):
            with test.subTest(case=name, include_static_kinematic_pairs=include):
                nxn_sets[include] = _nxn_pair_set(scene, device, include_static_kinematic_pairs=include)
                agg_set, count, _ = _aggregate_pair_set(
                    bp, scene, device, include_static_kinematic_pairs=include, arrays=arrays
                )
                test.assertEqual(agg_set, nxn_sets[include])
                test.assertEqual(count, len(agg_set))
                test.assertEqual(_static_pairs(scene, agg_set)[1], 0)
                walk, _ = _static_plan(bp)
                walked += int(np.count_nonzero(walk))
                listed += int(np.count_nonzero(~walk))
        stats = bp.statistics()
        if name == "unbounded statics":
            # Boxes with an edge over 64 cells join their world's unbounded list.
            test.assertEqual((stats["static_shapes"], stats["unbounded_shapes"]), (57, 5))
            test.assertEqual(sorted(slabs["world"][slab].tolist()), [-1, 0, 1])
        elif name == "two buckets":
            test.assertEqual(bp.static_bucket_start.shape[0], 3)
        elif name == "kinematic":
            dropped = nxn_sets[True] - nxn_sets[False]
            test.assertGreater(_static_pairs(scene, dropped)[0], 0)
        if name == "world -1 statics":
            test.assertTrue(np.any(bp.aggregate_world.numpy() < 0))
    test.assertGreater(walked, 0)
    test.assertGreater(listed, 0)


def test_aggregate_statics_knob_agree(test, device):
    """The static grid and one-shape static aggregates give nxn's pair set, in either emission and main level."""
    scene = _aggregate_scene(4, static_boxes=60)
    nxn_set = _nxn_pair_set(scene, device)
    for knobs in (
        {"statics": "grid"},
        {"statics": "aggregates"},
        {"statics": "grid", "emission": "tile"},
        {"statics": "grid", "ordered": True},
        {"statics": "aggregates", "ordered": True},
        {"statics": "grid", "main": "sap"},
        {"statics": "aggregates", "main": "sap"},
        {"statics": "grid", "main": "sap", "emission": "tile"},
        {"statics": "aggregates", "main": "sap", "ordered": True},
    ):
        with test.subTest(knobs=knobs):
            with mock.patch.dict(broad_phase_aggregate._KNOBS, knobs):
                bp = _make_aggregate(scene, device, shape_reach=_grid_reach(scene))
            test.assertEqual(bp.statistics()["main_level"], knobs.get("main", "grid"))
            arrays = _aggregate_arrays(scene, device)
            _build_static(bp, arrays)
            agg_set, count, _ = _aggregate_pair_set(bp, scene, device, arrays=arrays)
            test.assertEqual(agg_set, nxn_set)
            test.assertEqual(count, len(agg_set))
            stats = bp.statistics()
            if knobs["statics"] == "aggregates":
                test.assertEqual((stats["aggregates"], stats["static_shapes"]), (100, 0))
            else:
                test.assertEqual((stats["aggregates"], stats["static_shapes"]), (40, 60))


def test_aggregate_world_static_visibility(test, device):
    """A world -1 static box pairs with the aggregates of worlds 0 and 1; a world 0 box only with world 0's."""
    unit = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    shifted = ((0.5, 0.5, 0.5), (1.5, 1.5, 1.5))
    far = ((3.0, 3.0, 3.0), (4.0, 4.0, 4.0))
    scene = _box_scene(
        [
            (*unit, -1, -1, -1, -1),  # 0: world -1 static
            (*unit, 0, -1, -1, -1),  # 1: world 0 static
            (*shifted, 0, 1, 0, 0),  # 2: world 0 aggregate
            (*shifted, 1, 1, 1, 1),  # 3: world 1 aggregate
            (*far, 0, 1, 2, 2),  # 4: world 0 aggregate
            (*far, 1, -1, -1, -1),  # 5: world 1 static over shape 4
        ]
    )
    expected = {(0, 2), (0, 3), (1, 2)}
    test.assertEqual(_nxn_pair_set(scene, device), expected)
    for knobs in ({}, {"ordered": True}, {"statics": "aggregates"}):
        with test.subTest(knobs=knobs):
            with mock.patch.dict(broad_phase_aggregate._KNOBS, knobs):
                bp = _make_aggregate(scene, device)
            arrays = _aggregate_arrays(scene, device)
            _build_static(bp, arrays)
            agg_set, count, _ = _aggregate_pair_set(bp, scene, device, arrays=arrays)
            test.assertEqual(agg_set, expected)
            test.assertEqual(count, 3)


def test_aggregate_rejects_global_aggregates_with_world_statics(test, device):
    """World -1 aggregates beside per-world finite statics raise; world -1 statics, or static aggregates, do not."""
    unit = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    shapes = [
        (*unit, -1, 1, 0, 0),  # 0: world -1 aggregate
        (*unit, 0, 1, 1, 1),  # 1: world 0 aggregate
        (*unit, 0, -1, -1, -1),  # 2: world 0 static
        (*unit, -1, -1, -1, -1),  # 3: world -1 static
    ]
    scene = _box_scene(shapes)
    with test.assertRaisesRegex(NotImplementedError, r"1 world -1 aggregates.*1 finite static shapes in worlds"):
        _make_aggregate(scene, device)
    expected = _nxn_pair_set(scene, device)
    test.assertEqual(expected, {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)})
    with mock.patch.dict(broad_phase_aggregate._KNOBS, {"statics": "aggregates"}):
        bp = _make_aggregate(scene, device)
    test.assertEqual(_aggregate_pair_set(bp, scene, device)[0], expected)

    global_scene = _box_scene([*shapes[:2], (*unit, -1, -1, -1, -1), shapes[3]])
    bp = _make_aggregate(global_scene, device)
    arrays = _aggregate_arrays(global_scene, device)
    _build_static(bp, arrays)
    test.assertEqual(
        _aggregate_pair_set(bp, global_scene, device, arrays=arrays)[0], _nxn_pair_set(global_scene, device)
    )


def _moved_statics(scene: dict[str, np.ndarray], *, shift=(0.0, 0.0, 0.0), grow: float = 0.0, shapes=None):
    """A copy of ``scene`` with its finite static boxes (or ``shapes``) shifted and grown on every side [m]."""
    moved = dict(scene)
    if shapes is None:
        shapes = (scene["body"] < 0) & (scene["type"] == int(newton.GeoType.BOX))
    moved["lower"] = scene["lower"].copy()
    moved["upper"] = scene["upper"].copy()
    moved["lower"][shapes] += np.asarray(shift, dtype=np.float32) - np.float32(grow)
    moved["upper"][shapes] += np.asarray(shift, dtype=np.float32) + np.float32(grow)
    return moved


def test_aggregate_static_grid_rebuild_rules(test, device):
    """The grid is built before the first launch, goes stale when statics move, rebins in place, refuses a
    classification change and drops entries past its capacity with a flag."""
    scene = _aggregate_scene(3, static_boxes=60)
    bp = _make_aggregate(scene, device)
    arrays = _aggregate_arrays(scene, device)
    shape_count = len(scene["world"])
    pairs = wp.zeros(shape_count * (shape_count - 1) // 2, dtype=wp.vec2i, device=device)
    count = wp.zeros(1, dtype=wp.int32, device=device)
    with test.assertRaisesRegex(RuntimeError, "build_static"):
        _launch_aggregate(bp, arrays, pairs, count, device)
    test.assertEqual(bp.statistics()["static_shapes"], 0)
    with test.assertRaisesRegex(ValueError, "entries"):
        bp.build_static(arrays["lower"][:10], arrays["upper"][:10])
    _build_static(bp, arrays)
    test.assertEqual(_aggregate_pair_set(bp, scene, device, arrays=arrays)[0], _nxn_pair_set(scene, device))
    names = ("static_shapes", "static_cell_lower", "static_entries", "static_bucket_start", "unbounded_shapes")
    pointers = {name: getattr(bp, name).ptr for name in names}

    # Moved 1.5 m without a rebuild, the statics are still binned where they were:
    # pairs at their new place are missed, and only true pairs are found.
    moved = _moved_statics(scene, shift=(1.5, 0.0, 0.0))
    moved_arrays = _aggregate_arrays(moved, device)
    moved_nxn = _nxn_pair_set(moved, device)
    stale = _aggregate_pair_set(bp, moved, device, arrays=moved_arrays)[0]
    test.assertTrue(stale < moved_nxn)
    _build_static(bp, moved_arrays)
    test.assertEqual(_aggregate_pair_set(bp, moved, device, arrays=moved_arrays)[0], moved_nxn)
    test.assertEqual({name: getattr(bp, name).ptr for name in names}, pointers)

    # A box grown past 64 cells would change lists: refused, and the grid is kept.
    statics = np.flatnonzero((scene["body"] < 0) & (scene["type"] == int(newton.GeoType.BOX)))
    single = np.zeros(shape_count, dtype=bool)
    single[statics[0]] = True
    grown = _moved_statics(moved, grow=40.0 * bp.static_cell_size, shapes=single)
    with test.assertRaisesRegex(RuntimeError, "rebuild the pipeline"):
        _build_static(bp, _aggregate_arrays(grown, device))
    test.assertEqual(_aggregate_pair_set(bp, moved, device, arrays=moved_arrays)[0], moved_nxn)

    # Boxes grown to about 6 m need more entries than the capacity: flagged, warned, pairs only dropped.
    large = _moved_statics(moved, grow=3.0)
    large_arrays = _aggregate_arrays(large, device)
    with test.assertWarnsRegex(NewtonWarning, "static grid"):
        _build_static(bp, large_arrays)
    test.assertEqual(bp.statistics()["static_entry_overflow"], 1)
    large_set = _aggregate_pair_set(bp, large, device, arrays=large_arrays)[0]
    test.assertTrue(large_set < _nxn_pair_set(large, device))
    # The statics left out get empty cell ranges and are not seen by either path.
    left_out = bp.static_cell_upper.numpy()[:, 0] < bp.static_cell_lower.numpy()[:, 0]
    test.assertTrue(np.any(left_out))
    left_out_shapes = set(bp.static_shapes.numpy()[left_out].tolist())
    test.assertFalse(any(i in left_out_shapes or j in left_out_shapes for i, j in large_set))
    _build_static(bp, moved_arrays)
    test.assertEqual(bp.statistics()["static_entry_overflow"], 0)
    test.assertEqual(_aggregate_pair_set(bp, moved, device, arrays=moved_arrays)[0], moved_nxn)
    test.assertEqual({name: getattr(bp, name).ptr for name in names}, pointers)

    nan = dict(moved, lower=moved["lower"].copy())
    nan["lower"][statics[1]] = np.nan
    with test.assertRaisesRegex(ValueError, "finite"):
        _build_static(bp, _aggregate_arrays(nan, device))


def _left_out_statics(bp: BroadPhaseAggregate) -> set[int]:
    """The statics the last build left out of the grid: those with an empty binned cell range."""
    lower = bp.static_cell_lower.numpy()
    upper = bp.static_cell_upper.numpy()
    return set(bp.static_shapes.numpy()[np.any(upper < lower, axis=1)].tolist())


def test_aggregate_static_overflow_hides_left_out_statics(test, device):
    """Statics left out of a full grid are seen by neither the walking nor the listing aggregates."""
    scene = _aggregate_scene(3, static_boxes=60)
    arrays = _aggregate_arrays(scene, device)
    # Derived cells (about 1.3 m) let most aggregates walk; 0.3 m cells make them list.
    for name, kwargs, grow, walks in (("walk", {}, 3.0, True), ("list", {"static_cell_size": 0.3}, 1.0, False)):
        with test.subTest(path=name):
            bp = _make_aggregate(scene, device, **kwargs)
            _build_static(bp, arrays)
            large = _moved_statics(scene, grow=grow)
            large_arrays = _aggregate_arrays(large, device)
            with test.assertWarnsRegex(NewtonWarning, "static grid"):
                _build_static(bp, large_arrays)
            test.assertEqual(bp.statistics()["static_entry_overflow"], 1)
            left_out = _left_out_statics(bp)
            test.assertTrue(left_out)
            agg_set = _aggregate_pair_set(bp, large, device, arrays=large_arrays)[0]
            nxn_set = _nxn_pair_set(large, device)
            test.assertTrue(agg_set < nxn_set)
            test.assertFalse([pair for pair in agg_set if pair[0] in left_out or pair[1] in left_out])
            # The left-out statics overlap aggregates that take this path, so the check above is not vacuous.
            walk, _ = _static_plan(bp)
            missed = [
                bool(walk[scene["aggregate"][i if j in left_out else j]])
                for i, j in nxn_set
                if (i in left_out) != (j in left_out)
            ]
            test.assertIn(walks, missed)


def test_aggregate_statics_ordered_lists(test, device):
    """Ordered instances with statics publish one list: the unordered set in the documented order, the
    static pairs after every aggregate's own; stage replays and overflow keep it."""
    scene = _aggregate_scene(5, static_boxes=60)
    instances = (
        _make_aggregate(scene, device, ordered=True),
        _make_aggregate(scene, device, ordered=True),
        _make_aggregate(scene, device, ordered=False),
    )
    arrays = _aggregate_arrays(scene, device)
    for bp in instances:
        _build_static(bp, arrays)
    shape_count = len(scene["world"])
    capacity = shape_count * (shape_count - 1) // 2
    buffers = [
        (wp.zeros(capacity, dtype=wp.vec2i, device=device), wp.zeros(1, dtype=wp.int32, device=device))
        for _ in instances
    ]
    rng = np.random.default_rng(5)
    moving = scene["body"] >= 0
    lower = scene["lower"].copy()
    upper = scene["upper"].copy()
    for frame in range(10):
        if frame:
            step = rng.uniform(-0.05, 0.05, (shape_count, 3)).astype(np.float32) * moving[:, None]
            lower += step
            upper += step
            arrays["lower"].assign(lower)
            arrays["upper"].assign(upper)
        results = []
        for bp, (pairs, count) in zip(instances, buffers, strict=True):
            _launch_aggregate(bp, arrays, pairs, count, device)
            n = int(count.numpy()[0])
            results.append((pairs.numpy()[:n].copy(), bp.statistics()))
        (first, first_stats), (second, _), (unordered, unordered_stats) = results
        unordered_set = _pair_set(unordered, len(unordered))
        with test.subTest(frame=frame):
            test.assertTrue(np.array_equal(first, second))
            test.assertEqual(_pair_set(first, len(first)), unordered_set)
            test.assertEqual(len(first), len(unordered_set))
            test.assertEqual(first_stats["pair_demand"], unordered_stats["pair_demand"])
            test.assertTrue(np.array_equal(first, _documented_order(scene, unordered_set, instances[0])))
            if frame == 0:
                test.assertEqual(unordered_set, _nxn_pair_set(scene, device))
    test.assertGreater(_static_pairs(scene, unordered_set)[0], 0)

    # World -1's statics precede the aggregate's world's at one test cell (4 m cells, walked) and when
    # listed (0.1 m cells), whatever their shape order.
    unit = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    small = _box_scene([(*unit, 0, -1, -1, -1), (*unit, -1, -1, -1, -1), ((0.5,) * 3, (1.5,) * 3, 0, 1, 0, 0)])
    small_arrays = _aggregate_arrays(small, device)
    for cell_size, walks in ((4.0, True), (0.1, False)):
        with test.subTest(static_cell_size=cell_size):
            bp = _make_aggregate(small, device, ordered=True, static_cell_size=cell_size)
            _build_static(bp, small_arrays)
            _, n, raw = _aggregate_pair_set(bp, small, device, arrays=small_arrays)
            test.assertEqual(raw[:n].tolist(), [[1, 2], [0, 2]])
            test.assertEqual(bool(_static_plan(bp)[0][0]), walks)

    # At every stage, the ordered list is the launch's.
    bp, (pairs, count) = instances[0], buffers[0]
    test.assertIn("static", bp._stages)
    pairs.fill_(wp.vec2i(-7, -7))
    for stage in broad_phase_aggregate._STAGES:
        bp._launch_stage(stage)
    test.assertTrue(np.array_equal(pairs.numpy()[: int(count.numpy()[0])], first))

    # Overflow keeps the first entries.
    n = len(first)
    buffer = wp.full(n, wp.vec2i(-7, -7), dtype=wp.vec2i, device=device)
    _launch_aggregate(bp, arrays, buffer[: n // 2], count, device)
    test.assertEqual(int(count.numpy()[0]), n // 2)
    test.assertTrue(np.array_equal(buffer.numpy()[: n // 2], first[: n // 2]))
    test.assertTrue(np.all(buffer.numpy()[n // 2 :] == -7))

    # The int32 bound counts every finite static in an aggregate's view.
    sizes = np.bincount(scene["aggregate"][moving & (scene["flags"] != 0)]).astype(np.int64)
    world = np.asarray([scene["world"][np.flatnonzero(scene["aggregate"] == a)[0]] for a in range(len(sizes))])
    static_world = scene["world"][(scene["body"] < 0)]
    in_view = np.asarray([np.count_nonzero((static_world == -1) | (static_world == w)) for w in world])
    expected = int(np.sum(sizes * (sizes - 1) // 2 + sizes * in_view))
    expected += bp.aggregate_pairs_max * int(sizes.max()) ** 2
    test.assertEqual(bp._ordered_slots, expected)


def test_aggregate_statics_capture_replay(test, device):
    """A captured launch replays a grid rebinned outside the capture; binning inside a capture raises."""
    scene = _aggregate_scene(6, static_boxes=60)
    moved = _moved_statics(scene, shift=(0.7, -0.4, 0.0))
    moved_nxn = _nxn_pair_set(moved, device)
    shape_count = len(scene["world"])
    for ordered in (False, True):
        with test.subTest(ordered=ordered):
            bp = _make_aggregate(scene, device, ordered=ordered)
            arrays = _aggregate_arrays(scene, device)
            _build_static(bp, arrays)
            pairs = wp.zeros(shape_count * (shape_count - 1) // 2, dtype=wp.vec2i, device=device)
            count = wp.zeros(1, dtype=wp.int32, device=device)
            _launch_aggregate(bp, arrays, pairs, count, device)
            with wp.ScopedCapture(device=device) as capture:
                _launch_aggregate(bp, arrays, pairs, count, device)
                with test.assertRaisesRegex(RuntimeError, "capture"):
                    _build_static(bp, arrays)
            arrays["lower"].assign(moved["lower"])
            arrays["upper"].assign(moved["upper"])
            _build_static(bp, arrays)
            pairs.fill_(wp.vec2i(-7, -7))
            count.fill_(12345)
            wp.capture_launch(capture.graph)
            replay_count = int(count.numpy()[0])
            test.assertEqual(_pair_set(pairs.numpy(), replay_count), moved_nxn)
            test.assertEqual(replay_count, len(moved_nxn))


def _grid_reach(scene: dict[str, np.ndarray]) -> np.ndarray:
    """A reach of 0.65 m for every shape of an :func:`_aggregate_scene` scene.

    Each member box lies within 0.4 + 0.25 m of its aggregate's centre on every
    axis, so every union-box edge is at most twice that, the grid's cell size.
    """
    return np.full(len(scene["world"]), 0.65, dtype=np.float32)


def _grid_cell_size(reach: float, extent: float = 0.0) -> float:
    """The grid's cell for the largest member reach and the initial shape extent, as the class documents it."""
    return float(np.float32(2.0 * reach * (1.0 + 2.0**-8) + 2.0e-6 * max(1.0, extent + reach)))


def _host_aggregate_pairs(bp: BroadPhaseAggregate) -> set[tuple[int, int]]:
    """The aggregate pairs of the last launch's union boxes that overlap and pass the world and group rules."""
    lower = bp.aggregate_lower.numpy()
    upper = bp.aggregate_upper.numpy()
    world = bp.aggregate_world.numpy()
    group = bp.aggregate_group.numpy()
    overlap = np.all((lower[:, None, :] <= upper[None, :, :]) & (upper[:, None, :] >= lower[None, :, :]), axis=2)
    first, second = np.nonzero(np.triu(overlap, 1))
    return {
        (int(a), int(b))
        for a, b in zip(first, second, strict=True)
        if test_world_and_group_pair(int(world[a]), int(world[b]), int(group[a]), int(group[b]))
    }


def _aggregate_members(bp: BroadPhaseAggregate) -> list[list[int]]:
    """Each aggregate's member shapes, in shape order."""
    start = bp.aggregate_start.numpy()
    shapes = bp.aggregate_shapes.numpy()
    return [shapes[start[a] : start[a + 1]].tolist() for a in range(bp.aggregate_count)]


def test_aggregate_grid_matches_nxn_random(test, device):
    """The grid main level finds every overlapping aggregate pair once, and the shape pair set equals nxn's.

    The scenes have three worlds and world -1 aggregates. Ordered, the grid's
    list equals the sweep's.
    """
    for seed in range(3):
        scene = _aggregate_scene(seed, worlds=3)
        reach = _grid_reach(scene)
        grid = _make_aggregate(scene, device, main="grid", shape_reach=reach)
        test.assertAlmostEqual(grid.cell_size, _grid_cell_size(0.65), places=6)
        for include in (True, False):
            with test.subTest(seed=seed, include_static_kinematic_pairs=include):
                nxn_set = _nxn_pair_set(scene, device, include_static_kinematic_pairs=include)
                agg_set, count, raw = _aggregate_pair_set(grid, scene, device, include_static_kinematic_pairs=include)
                test.assertEqual(agg_set, nxn_set)
                test.assertEqual(count, len(agg_set))
                test.assertTrue(np.all(raw[:count, 0] < raw[:count, 1]))
                test.assertGreater(len(nxn_set), 1000)
                stats = grid.statistics()
                test.assertEqual(
                    (stats["pair_overflow"], stats["aggregate_pair_overflow"], stats["aggregate_extent_overflow"]),
                    (0, 0, 0),
                )
                aggregate_pairs = grid.aggregate_pairs.numpy()[: stats["aggregate_pair_count"]]
                found = {(int(a), int(b)) for a, b in aggregate_pairs}
                test.assertEqual(len(found), len(aggregate_pairs))
                test.assertTrue(np.all(aggregate_pairs[:, 0] < aggregate_pairs[:, 1]))
                test.assertEqual(found, _host_aggregate_pairs(grid))
        with test.subTest(seed=seed, ordered=True):
            lists = []
            for main in ("sap", "grid"):
                bp = _make_aggregate(scene, device, main=main, shape_reach=reach, ordered=True)
                _, count, raw = _aggregate_pair_set(bp, scene, device)
                lists.append(raw[:count])
            test.assertTrue(np.array_equal(lists[0], lists[1]))
            test.assertEqual(_pair_set(lists[1], len(lists[1])), _nxn_pair_set(scene, device))


def test_aggregate_grid_splits_prismatic_articulation(test, device):
    """Under ``main="grid"`` an articulation with a prismatic joint becomes one aggregate per body (one shape each)."""
    model, root_shape, child_shape, other_shape = _prismatic_model(device)
    bp = BroadPhaseAggregate.from_model(model, main="grid")
    stats = bp.statistics()
    test.assertEqual((stats["aggregates"], stats["split_aggregates"]), (3, 2))
    members = _aggregate_members(bp)
    test.assertIn([root_shape], members)
    test.assertIn([child_shape], members)
    test.assertIn([other_shape], members)
    # Each capsule is its own aggregate, so its reach is its own bound; the other capsule starts at x = 3.1.
    bound = (model.shape_collision_radius.numpy() + model.shape_margin.numpy() + model.shape_gap.numpy())[
        [root_shape, child_shape, other_shape]
    ]
    test.assertAlmostEqual(bp.cell_size, _grid_cell_size(float(bound.max()), 3.1), places=5)

    lower, upper, nxn_set = _model_aabbs_and_nxn(model, model.state(), device)
    agg_set = _model_aggregate_pair_set(bp, model, lower, upper, device)
    test.assertIn((child_shape, other_shape), agg_set)
    test.assertEqual(agg_set, nxn_set)
    test.assertEqual(bp.statistics()["aggregate_extent_overflow"], 0)

    sap = BroadPhaseAggregate.from_model(model, main="sap")
    test.assertEqual((sap.statistics()["aggregates"], sap.statistics()["split_aggregates"]), (2, 0))


def test_aggregate_grid_reach_from_model(test, device):
    """``from_model`` sizes the grid's cell from the joint offsets along each articulation and the shape bounds."""
    model = _two_world_chain_model(device)
    bp = BroadPhaseAggregate.from_model(model, main="grid")
    test.assertEqual((bp.statistics()["aggregates"], bp.statistics()["split_aggregates"]), (4, 0))
    # The chain's joints sit 0.4 m along x in their parent and at the child's origin, so the
    # third link is 0.8 m from the root; its capsule reaches 0.1 + 0.15 m further. The third
    # link also starts farthest out, at x = 0.8.
    bound = model.shape_collision_radius.numpy() + model.shape_margin.numpy() + model.shape_gap.numpy()
    shape_body = model.shape_body.numpy()
    third_link = int(model.joint_child.numpy()[2])
    capsule = int(np.flatnonzero(shape_body == third_link)[0])
    test.assertAlmostEqual(float(model.shape_collision_radius.numpy()[capsule]), 0.25, places=6)
    test.assertAlmostEqual(bp.cell_size, _grid_cell_size(0.8 + float(bound[capsule]), 0.8), places=5)

    lower, upper, nxn_set = _model_aabbs_and_nxn(model, model.state(), device)
    test.assertEqual(_model_aggregate_pair_set(bp, model, lower, upper, device), nxn_set)
    test.assertEqual(bp.statistics()["aggregate_extent_overflow"], 0)

    # Each chunk reaches from its own root: the longest chunk, links 0 and 1, spans one 0.4 m joint.
    chunked = BroadPhaseAggregate.from_model(model, main="grid", max_aggregate_size=2)
    test.assertEqual(chunked.statistics()["aggregates"], 6)
    test.assertAlmostEqual(chunked.cell_size, _grid_cell_size(0.4 + float(bound[capsule]), 0.8), places=5)
    test.assertEqual(_model_aggregate_pair_set(chunked, model, lower, upper, device), nxn_set)
    test.assertEqual(chunked.statistics()["aggregate_extent_overflow"], 0)


def test_aggregate_grid_from_model_without_articulations(test, device):
    """``from_model(main="grid")`` handles bodies outside any articulation, with or without joints.

    Each sphere's box is exactly twice its reach, and the second one's lower
    corner sits on a cell boundary at x = 0; the cell's slack holds its pad.
    """
    for free_joints in (False, True):
        with test.subTest(free_joints=free_joints):
            builder = newton.ModelBuilder()
            builder.add_ground_plane()
            spheres = []
            for k in range(2):
                body = builder.add_link(xform=wp.transform(wp.vec3(0.3 * k, 0.0, 1.0), wp.quat_identity()))
                spheres.append(builder.add_shape_sphere(body, radius=0.2))
                if free_joints:
                    builder.add_joint_free(body)
            model = builder.finalize(device=device)
            test.assertEqual(model.articulation_count, 0)
            bp = BroadPhaseAggregate.from_model(model, main="grid")
            stats = bp.statistics()
            test.assertEqual((stats["aggregates"], stats["split_aggregates"]), (2, 0))
            # Each sphere is its own root, so its reach is its own bound; the spheres start at z = 1.
            bound = model.shape_collision_radius.numpy() + model.shape_margin.numpy() + model.shape_gap.numpy()
            test.assertAlmostEqual(bp.cell_size, _grid_cell_size(float(bound[spheres].max()), 1.0), places=5)
            lower, upper, nxn_set = _model_aabbs_and_nxn(model, model.state(), device)
            agg_set = _model_aggregate_pair_set(bp, model, lower, upper, device)
            test.assertIn((spheres[0], spheres[1]), agg_set)
            test.assertEqual(agg_set, nxn_set)
            test.assertEqual(bp.statistics()["aggregate_extent_overflow"], 0)


def test_aggregate_grid_bucket_collisions(test, device):
    """Candidates that share a bucket are told apart by their cell and world key.

    Two one-shape aggregates, of world 0 and world -1, get a table of four
    buckets, so their two world keys often share a bucket where the two boxes
    share a cell; the pair is still found once.
    """
    collide = int(ShapeFlags.COLLIDE_SHAPES)
    rng = np.random.default_rng(11)
    for worlds in ((0, -1), (-1, 0)):
        with test.subTest(worlds=worlds):
            scene = {"world": np.asarray(worlds, dtype=np.int32)}
            bp = BroadPhaseAggregate(
                scene["world"],
                np.arange(2, dtype=np.int32),
                np.arange(2, dtype=np.int32),
                np.full(2, int(newton.GeoType.BOX), dtype=np.int32),
                np.full((2, 3), 0.2, dtype=np.float32),
                np.full(2, collide, dtype=np.int32),
                main="grid",
                shape_reach=np.full(2, 0.5, dtype=np.float32),
                device=device,
            )
            test.assertEqual(bp.grid_bucket_count.shape[0], 4)
            arrays = {
                "lower": wp.zeros(2, dtype=wp.vec3, device=device),
                "upper": wp.zeros(2, dtype=wp.vec3, device=device),
                "world": wp.array(scene["world"], dtype=wp.int32, device=device),
                "group": wp.array([1, 1], dtype=wp.int32, device=device),
                "body": wp.array([0, 1], dtype=wp.int32, device=device),
                "body_flags": wp.zeros(2, dtype=wp.int32, device=device),
                "filter_pairs": None,
            }
            shared = 0
            for _ in range(64):
                # Two overlapping boxes, each at most 0.8 m on every axis (the cell is about 1 m).
                centre = rng.uniform(-20.0, 20.0, 3) + rng.uniform(-0.1, 0.1, (2, 3))
                half = rng.uniform(0.1, 0.4, (2, 3))
                arrays["lower"].assign((centre - half).astype(np.float32))
                arrays["upper"].assign((centre + half).astype(np.float32))
                agg_set, count, _ = _aggregate_pair_set(bp, scene, device, arrays=arrays)
                test.assertEqual((agg_set, count), ({(0, 1)}, 1))
                test.assertEqual(bp.statistics()["aggregate_pair_count"], 1)
                cell = bp.grid_cell.numpy()
                bucket = bp.grid_bucket.numpy()[:2]
                shared += int(np.array_equal(cell[0], cell[1]) and bucket[0] == bucket[1])
            test.assertGreater(shared, 0)


def test_aggregate_grid_cell_holds_exact_fill(test, device):
    """Boxes exactly twice their reach keep the latch clear out to 1,000 m, and the set equals nxn's.

    4,096 one-sphere aggregates, each box exactly 2R wide, are placed at random
    in a 16 m cube at 10, 100 and 1,000 m, 20 times each. Without the cell's
    slack, boxes whose padded corners straddle a cell boundary span three cells.
    """
    count = 4096
    bound = np.float32(0.3)
    span = 16.0
    capacity = 16 * count
    ids = np.arange(count, dtype=np.int32)
    world = np.zeros(count, dtype=np.int32)
    flags = np.full(count, int(ShapeFlags.COLLIDE_SHAPES), dtype=np.int32)
    arrays = {
        "lower": wp.zeros(count, dtype=wp.vec3, device=device),
        "upper": wp.zeros(count, dtype=wp.vec3, device=device),
        "world": wp.array(world, dtype=wp.int32, device=device),
        "group": wp.full(count, 1, dtype=wp.int32, device=device),
        "body": wp.array(ids, dtype=wp.int32, device=device),
        "body_flags": wp.zeros(count, dtype=wp.int32, device=device),
        "filter_pairs": None,
    }
    nxn = BroadPhaseAllPairs(arrays["world"], shape_flags=flags, device=device)
    buffers = [
        (wp.zeros(capacity, dtype=wp.vec2i, device=device), wp.zeros(1, dtype=wp.int32, device=device))
        for _ in range(2)
    ]
    rng = np.random.default_rng(5)
    for offset in (10.0, 100.0, 1000.0):
        with test.subTest(offset=offset):
            bp = BroadPhaseAggregate(
                world,
                ids,
                ids,
                np.full(count, int(newton.GeoType.SPHERE), dtype=np.int32),
                np.full((count, 3), bound, dtype=np.float32),
                flags,
                main="grid",
                shape_reach=np.full(count, bound, dtype=np.float32),
                shape_extent=offset + span,
                aggregate_pairs_max=capacity,
                device=device,
            )
            for _ in range(20):
                centre = rng.uniform(offset, offset + span, (count, 3)).astype(np.float32)
                arrays["lower"].assign(centre - bound)
                arrays["upper"].assign(centre + bound)
                (pairs, pair_count), (nxn_pairs, nxn_count) = buffers
                _launch_aggregate(bp, arrays, pairs, pair_count, device)
                nxn.launch(
                    arrays["lower"],
                    arrays["upper"],
                    None,
                    arrays["group"],
                    arrays["world"],
                    count,
                    nxn_pairs,
                    nxn_count,
                    device=device,
                    shape_body=arrays["body"],
                    body_flags=arrays["body_flags"],
                )
                n = int(pair_count.numpy()[0])
                m = int(nxn_count.numpy()[0])
                test.assertLess(m, capacity)
                test.assertEqual(_pair_set(pairs.numpy(), n), _pair_set(nxn_pairs.numpy(), m))
                test.assertEqual(n, m)
            test.assertEqual(bp.statistics()["aggregate_extent_overflow"], 0)


def test_aggregate_from_model_defaults_to_grid(test, device):
    """``from_model`` builds the grid by default, and the statistics say so; ``main="sap"`` builds the sweep.

    The model has four aggregates in two worlds, so the size rule, which
    would build the sweep, is relaxed (``grid_min_aggregates`` 0).
    """
    knobs = broad_phase_aggregate._KNOBS
    test.assertEqual(
        tuple(
            knobs[name]
            for name in (
                "main",
                "max_aggregate_size",
                "grid_cell_ratio_max",
                "grid_min_aggregates",
                "grid_min_worlds",
                "grid_occupancy_max",
            )
        ),
        ("grid", 32, 3.0, 32768, 2048, 90.0),
    )
    model = _two_world_chain_model(device)
    lower, upper, nxn_set = _model_aabbs_and_nxn(model, model.state(), device)
    for main, level in ((None, "grid"), ("sap", "sap")):
        with test.subTest(main=main):
            with mock.patch.dict(broad_phase_aggregate._KNOBS, {"grid_min_aggregates": 0}):
                bp = BroadPhaseAggregate.from_model(model, main=main)
            stats = bp.statistics()
            test.assertEqual((stats["main_level"], stats["main_fallback"]), (level, ""))
            test.assertEqual(stats["cell_size"] > 0.0, level == "grid")
            test.assertEqual(_model_aggregate_pair_set(bp, model, lower, upper, device), nxn_set)
            test.assertEqual(bp.statistics()["aggregate_extent_overflow"], 0)

    # A non-finite initial pose is named as such, not as an argument the caller never passed.
    body_q = model.body_q.numpy()
    body_q[0, 0] = np.nan
    model.body_q.assign(body_q)
    with test.assertRaisesRegex(ValueError, "body_q"):
        BroadPhaseAggregate.from_model(model)


def test_aggregate_direct_construction_falls_back_to_sap(test, device):
    """A direct construction without ``shape_reach`` takes the sweep; with one, the default grid."""
    scene = _aggregate_scene(5)
    nxn_set = _nxn_pair_set(scene, device)
    for kwargs, level, fallback in (
        ({}, "sap", "reach"),
        ({"shape_reach": _grid_reach(scene)}, "grid", ""),
        ({"main": "sap"}, "sap", ""),
    ):
        with test.subTest(kwargs=sorted(kwargs)):
            bp = _make_aggregate(scene, device, **kwargs)
            test.assertEqual((bp.statistics()["main_level"], bp.statistics()["main_fallback"]), (level, fallback))
            agg_set, _, _ = _aggregate_pair_set(bp, scene, device)
            test.assertEqual(agg_set, nxn_set)
    with test.assertRaisesRegex(ValueError, "main='grid'"):
        _make_aggregate(scene, device, main="grid")


def test_aggregate_splits_long_chains_by_default(test, device):
    """Under the grid, a chain of 40 one-shape links becomes aggregates of at most 32 members by default.

    The set stays nxn's. (Left to ``main=None``, this model's 32-link chunk sets a
    cell 4 times its median aggregate's, and the cell guard builds the sweep.)
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    joints = []
    parent = -1
    for k in range(40):
        link = builder.add_link(xform=wp.transform(wp.vec3(0.4 * k, 0.0, 0.15), wp.quat_identity()))
        builder.add_shape_capsule(link, radius=0.1, half_height=0.15)
        parent_xform = wp.transform(
            wp.vec3(0.4 if parent >= 0 else 0.0, 0.0, 0.0 if parent >= 0 else 0.15), wp.quat_identity()
        )
        joints.append(builder.add_joint_revolute(parent, link, parent_xform=parent_xform, axis=newton.Axis.Y))
        parent = link
    builder.add_articulation(joints)
    box = builder.add_body(xform=wp.transform(wp.vec3(4.0, 0.3, 0.15), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.2, hy=0.2, hz=0.2)
    model = builder.finalize(device=device)
    lower, upper, nxn_set = _model_aabbs_and_nxn(model, model.state(), device)
    test.assertGreater(len(nxn_set), 40)
    for size, largest in ((None, 32), (0, 40)):
        with test.subTest(max_aggregate_size=size):
            bp = BroadPhaseAggregate.from_model(model, max_aggregate_size=size, main="grid")
            stats = bp.statistics()
            test.assertEqual((stats["main_level"], stats["largest_aggregate"]), ("grid", largest))
            test.assertEqual(_model_aggregate_pair_set(bp, model, lower, upper, device), nxn_set)
            test.assertEqual(bp.statistics()["aggregate_extent_overflow"], 0)


def _chain_model(device, links: int, small_bodies: int = 0) -> newton.Model:
    """A revolute chain of ``links`` capsule links 0.15 m apart, and ``small_bodies`` free spheres 1 m apart beside it."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for k in range(small_bodies):
        body = builder.add_body(xform=wp.transform(wp.vec3(1.0 * k, 5.0, 0.3), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.2)
    joints = []
    parent = -1
    for k in range(links):
        link = builder.add_link(xform=wp.transform(wp.vec3(0.15 * k, 0.0, 0.15), wp.quat_identity()))
        builder.add_shape_capsule(link, radius=0.1, half_height=0.075)
        offset = wp.vec3(0.15, 0.0, 0.0) if parent >= 0 else wp.vec3(0.0, 0.0, 0.15)
        joints.append(
            builder.add_joint_revolute(
                parent, link, parent_xform=wp.transform(offset, wp.quat_identity()), axis=newton.Axis.Y
            )
        )
        parent = link
    if joints:
        builder.add_articulation(joints)
    return builder.finalize(device=device)


def test_aggregate_grid_reach_per_chunk(test, device):
    """Each chunk of a long articulation reaches from its own root, so a 128-link chain gets a 32-link cell.

    The size rule, which would build the sweep for its four aggregates, is relaxed.
    """
    model = _chain_model(device, 128)
    with mock.patch.dict(broad_phase_aggregate._KNOBS, {"grid_min_aggregates": 0}):
        bp = BroadPhaseAggregate.from_model(model)
    stats = bp.statistics()
    test.assertEqual((stats["main_level"], stats["aggregates"], stats["largest_aggregate"]), ("grid", 4, 32))
    # Each chunk's links are 31 joints of 0.15 m from its first link, plus the capsule's 0.175 m and gaps.
    capsule = int(np.flatnonzero(model.shape_type.numpy() == int(newton.GeoType.CAPSULE))[0])
    bound = float(
        model.shape_collision_radius.numpy()[capsule]
        + model.shape_margin.numpy()[capsule]
        + model.shape_gap.numpy()[capsule]
    )
    test.assertAlmostEqual(bp.cell_size, _grid_cell_size(31 * 0.15 + bound, 127 * 0.15), places=4)
    lower, upper, nxn_set = _model_aabbs_and_nxn(model, model.state(), device)
    test.assertEqual(_model_aggregate_pair_set(bp, model, lower, upper, device), nxn_set)
    test.assertEqual(bp.statistics()["aggregate_extent_overflow"], 0)


def test_aggregate_grid_cell_guard(test, device):
    """By default ``from_model`` takes the sweep when the cell is far above the median aggregate.

    64 small spheres (2R = 0.6 m) beside a 128-link chain, whose 32-link chunks
    set a 9.9 m cell, more than 3 times the median: the sweep, with a warning.
    A 4-link chain (a 1.5 m cell), an explicit ``main="grid"`` and a relaxed
    ``grid_cell_ratio_max`` keep the grid. The size rule, which would build
    the sweep for these 65-68 aggregates first, is relaxed in every row.
    """
    for links, knobs, main, level, fallback in (
        (128, {}, None, "sap", "cell"),
        (128, {}, "grid", "grid", ""),
        (128, {"grid_cell_ratio_max": 20.0}, None, "grid", ""),
        (4, {}, None, "grid", ""),
    ):
        with test.subTest(links=links, knobs=knobs, main=main):
            model = _chain_model(device, links, small_bodies=64)
            relaxed = {"grid_min_aggregates": 0, **knobs}
            with mock.patch.dict(broad_phase_aggregate._KNOBS, relaxed), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                bp = BroadPhaseAggregate.from_model(model, main=main)
            stats = bp.statistics()
            test.assertEqual((stats["main_level"], stats["main_fallback"]), (level, fallback))
            warned = [w for w in caught if issubclass(w.category, NewtonWarning)]
            test.assertEqual(len(warned), int(fallback == "cell"))
            if warned:
                message = str(warned[0].message)
                for text in (
                    "cell",
                    "median",
                    "from_model(model, main='sap', verify=True)",
                    "max_aggregate_size",
                    "expert path",
                ):
                    test.assertIn(text, message)
                test.assertNotIn("_KNOBS", message)
                # The warning points at the caller, also through the pipeline.
                test.assertEqual(warned[0].filename, __file__)
                with (
                    mock.patch.dict(broad_phase_aggregate._KNOBS, relaxed),
                    warnings.catch_warnings(record=True) as caught,
                ):
                    warnings.simplefilter("always")
                    newton.CollisionPipeline(model, broad_phase="aggregate")
                test.assertEqual([w.filename for w in caught if issubclass(w.category, NewtonWarning)], [__file__])
            lower, upper, nxn_set = _model_aabbs_and_nxn(model, model.state(), device)
            test.assertEqual(_model_aggregate_pair_set(bp, model, lower, upper, device), nxn_set)
    model = _chain_model(device, 4, small_bodies=4)
    for ratio in (0.0, -1.0, float("inf")):
        with test.subTest(grid_cell_ratio_max=ratio):
            with mock.patch.dict(broad_phase_aggregate._KNOBS, {"grid_cell_ratio_max": ratio}):
                with test.assertRaisesRegex(ValueError, "grid_cell_ratio_max"):
                    BroadPhaseAggregate.from_model(model)


def _device_grid_occupancy(bp: BroadPhaseAggregate) -> float:
    """The mean over aggregates of how many share its world key and cell, from the cells the grid's last launch binned."""
    keys = np.column_stack((bp.aggregate_world.numpy().astype(np.int64) + 1, bp.grid_cell.numpy()))
    _, counts = np.unique(keys, axis=0, return_counts=True)
    return float(np.sum(counts.astype(np.float64) ** 2) / bp.aggregate_count)


def _chain_lattice_model(device, side: int, links: int, spacing: float) -> newton.Model:
    """A ground plane and ``side`` x ``side`` revolute chains of ``links`` capsule links 0.15 m apart along x.

    The chains stand ``spacing`` [m] apart on y and z from y = z = 0.5 m, one
    aggregate each, all in world -1 (no world counts them).
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for row in range(side):
        for column in range(side):
            y, z = 0.5 + spacing * row, 0.5 + spacing * column
            joints = []
            parent = -1
            for k in range(links):
                link = builder.add_link(xform=wp.transform(wp.vec3(0.15 * k, y, z), wp.quat_identity()))
                builder.add_shape_capsule(link, radius=0.1, half_height=0.075)
                offset = wp.vec3(0.15, 0.0, 0.0) if parent >= 0 else wp.vec3(0.0, y, z)
                joints.append(
                    builder.add_joint_revolute(
                        parent, link, parent_xform=wp.transform(offset, wp.quat_identity()), axis=newton.Axis.Y
                    )
                )
                parent = link
            builder.add_articulation(joints)
    return builder.finalize(device=device)


def test_aggregate_default_main_level_by_size(test, device):
    """By default ``from_model`` builds the sweep for fewer than ``grid_min_aggregates`` aggregates in few worlds.

    Eight humanoids 1 m apart in one world: the sweep at the default 32,768,
    the grid at 0 and at 8, the sweep at 9. The grid's occupancy, computed on
    the host from the initial pose, equals the mean the grid's own cells give
    after a launch on that pose.
    """
    model = _aggregate_humanoids(device, worlds=1, per_world=8, spacing=1.0)
    lower, upper, nxn_set = _model_aabbs_and_nxn(model, model.state(), device)
    for knobs, level, fallback in (
        ({}, "sap", "size"),
        ({"grid_min_aggregates": 0}, "grid", ""),
        ({"grid_min_aggregates": 8}, "grid", ""),
        ({"grid_min_aggregates": 9}, "sap", "size"),
    ):
        with test.subTest(knobs=knobs):
            with mock.patch.dict(broad_phase_aggregate._KNOBS, knobs):
                bp = BroadPhaseAggregate.from_model(model)
            stats = bp.statistics()
            test.assertEqual((stats["main_level"], stats["main_fallback"]), (level, fallback))
            test.assertEqual((stats["aggregates"], stats["aggregate_worlds"]), (8, 1))
            test.assertEqual(_model_aggregate_pair_set(bp, model, lower, upper, device), nxn_set)
            if level == "grid":
                test.assertGreater(stats["grid_occupancy"], 1.0)
                test.assertEqual(stats["grid_occupancy"], _device_grid_occupancy(bp))
            else:
                test.assertEqual(stats["grid_occupancy"], 0.0)


def test_aggregate_default_main_level_by_worlds(test, device):
    """From ``grid_min_worlds`` worlds holding aggregates on, the size rule is skipped; the guards still apply.

    Four worlds of one humanoid each, all at one place: with the knob at 4 the
    grid (each world bins under its own key, so the occupancy is 1), at 5 the
    sweep by the size rule, and with the knob at 4 but an occupancy limit of
    0.5 the sweep by the occupancy guard.
    """
    model = _aggregate_humanoids(device, worlds=4, per_world=1, spacing=0.4)
    lower, upper, nxn_set = _model_aabbs_and_nxn(model, model.state(), device)
    for knobs, level, fallback, occupancy in (
        ({"grid_min_worlds": 4}, "grid", "", 1.0),
        ({"grid_min_worlds": 5}, "sap", "size", 0.0),
        ({"grid_min_worlds": 4, "grid_occupancy_max": 0.5}, "sap", "occupancy", 1.0),
    ):
        with test.subTest(knobs=knobs):
            with mock.patch.dict(broad_phase_aggregate._KNOBS, knobs):
                bp = BroadPhaseAggregate.from_model(model)
            stats = bp.statistics()
            test.assertEqual((stats["main_level"], stats["main_fallback"]), (level, fallback))
            test.assertEqual((stats["aggregates"], stats["aggregate_worlds"]), (4, 4))
            test.assertEqual(stats["grid_occupancy"], occupancy)
            test.assertEqual(_model_aggregate_pair_set(bp, model, lower, upper, device), nxn_set)
            if level == "grid":
                test.assertEqual(_device_grid_occupancy(bp), occupancy)


def test_aggregate_default_main_level_by_occupancy(test, device):
    """By default ``from_model`` builds the sweep, without a warning, when the aggregates crowd the grid's cells.

    144 chains of 16 capsule links stand 0.3 m apart on y and z, so all their
    union boxes start in one cell of about 5.1 m: an occupancy of 144, above
    the default 90, builds the sweep; a limit of 200, or ``inf``, keeps the
    grid. The size rule is relaxed so that the guard decides; left alone, it
    builds the sweep first. An explicit ``main="grid"`` builds the grid and
    computes no occupancy. Invalid knobs, non-numeric ones included, raise a
    ``ValueError`` that names them.
    """
    model = _chain_lattice_model(device, side=12, links=16, spacing=0.3)
    lower, upper, nxn_set = _model_aabbs_and_nxn(model, model.state(), device)
    for knobs, main, level, fallback, occupancy in (
        ({"grid_min_aggregates": 0}, None, "sap", "occupancy", 144.0),
        ({"grid_min_aggregates": 0, "grid_occupancy_max": 200.0}, None, "grid", "", 144.0),
        ({"grid_min_aggregates": 0, "grid_occupancy_max": float("inf")}, None, "grid", "", 144.0),
        ({}, None, "sap", "size", 0.0),
        ({"grid_min_aggregates": 0}, "grid", "grid", "", 0.0),
    ):
        with test.subTest(knobs=knobs, main=main):
            with mock.patch.dict(broad_phase_aggregate._KNOBS, knobs), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                bp = BroadPhaseAggregate.from_model(model, main=main)
            test.assertEqual([w for w in caught if issubclass(w.category, NewtonWarning)], [])
            stats = bp.statistics()
            test.assertEqual((stats["main_level"], stats["main_fallback"]), (level, fallback))
            test.assertEqual((stats["aggregates"], stats["largest_aggregate"], stats["aggregate_worlds"]), (144, 16, 0))
            test.assertEqual(stats["grid_occupancy"], occupancy)
            test.assertEqual(_model_aggregate_pair_set(bp, model, lower, upper, device), nxn_set)
            if level == "grid":
                test.assertLess(bp.cell_size, 5.2)
                test.assertEqual(_device_grid_occupancy(bp), 144.0)

    model = _chain_model(device, 4, small_bodies=4)
    for name, value in (
        ("grid_min_aggregates", -1),
        ("grid_min_aggregates", 2.5),
        ("grid_min_aggregates", True),
        ("grid_min_worlds", 0),
        ("grid_occupancy_max", 0.0),
        ("grid_occupancy_max", float("nan")),
        ("grid_occupancy_max", "90"),
        ("grid_cell_ratio_max", None),
    ):
        with test.subTest(**{name: value}):
            with mock.patch.dict(broad_phase_aggregate._KNOBS, {name: value}):
                with test.assertRaisesRegex(ValueError, name):
                    BroadPhaseAggregate.from_model(model)


def test_aggregate_grid_extent_latch(test, device):
    """E6: a union box wider than the grid's cell latches ``aggregate_extent_overflow`` until reset."""
    scene = _aggregate_scene(0)
    reach = _grid_reach(scene)
    grid = _make_aggregate(scene, device, main="grid", shape_reach=reach)
    sap = _make_aggregate(scene, device, main="sap", shape_reach=reach)
    arrays = _aggregate_arrays(scene, device)
    _aggregate_pair_set(grid, scene, device, arrays=arrays)
    test.assertEqual(grid.statistics()["aggregate_extent_overflow"], 0)

    # After construction, one member's box grows to 10x its reach on every side.
    shape = int(np.flatnonzero(scene["aggregate"] == 1)[0])
    centre = 0.5 * (scene["lower"][shape] + scene["upper"][shape])
    grown_lower = scene["lower"].copy()
    grown_upper = scene["upper"].copy()
    grown_lower[shape] = centre - 10.0 * reach[shape]
    grown_upper[shape] = centre + 10.0 * reach[shape]
    arrays["lower"].assign(grown_lower)
    arrays["upper"].assign(grown_upper)
    for bp in (grid, sap):
        _aggregate_pair_set(bp, scene, device, arrays=arrays)
    test.assertEqual(grid.statistics()["aggregate_extent_overflow"], 1)
    test.assertEqual(sap.statistics()["aggregate_extent_overflow"], 0)

    # The latch holds after the box shrinks back, until reset_statistics().
    arrays["lower"].assign(scene["lower"])
    arrays["upper"].assign(scene["upper"])
    _aggregate_pair_set(grid, scene, device, arrays=arrays)
    test.assertEqual(grid.statistics()["aggregate_extent_overflow"], 1)
    grid.reset_statistics()
    test.assertEqual(grid.statistics()["aggregate_extent_overflow"], 0)
    agg_set, _, _ = _aggregate_pair_set(grid, scene, device, arrays=arrays)
    test.assertEqual(grid.statistics()["aggregate_extent_overflow"], 0)
    test.assertEqual(agg_set, _nxn_pair_set(scene, device))


def _device_output(launch) -> str:
    """Run ``launch`` and return what the device printed to stdout meanwhile.

    The capture redirects the process's file descriptor 1, which C ``printf``
    writes, also inside the test runner's own capture (which swaps
    ``sys.stdout`` for a file of its own).
    """
    wp.synchronize()
    outer = sys.stdout
    sys.stdout = sys.__stdout__
    try:
        capture = StdOutCapture()
        capture.begin()
        try:
            launch()
            wp.synchronize()
        finally:
            output = capture.end()
    finally:
        sys.stdout = outer
    # Lazily loaded modules announce themselves; CheckOutput lets those lines through too.
    return "".join(
        line
        for line in output.splitlines(keepends=True)
        if not (line.startswith("Module") and "load on device" in line)
    )


def test_aggregate_verify_prints_device_warnings(test, device):
    """With ``verify=True`` the device prints each of the three loss channels, without a host read.

    The pair and aggregate-pair overflows print from the one-thread publish on
    every collide that drops pairs, unordered and ordered; the grid's extent
    latch prints when it is set, once, and again after ``reset_statistics()``
    re-arms it. Zeroing ``verify_flag`` silences them, and the default
    (``verify=False``) prints nothing and allocates no flag.
    """
    scene = _aggregate_scene(0)
    n = len(_nxn_pair_set(scene, device))
    arrays = _aggregate_arrays(scene, device)
    for ordered in (False, True):
        with test.subTest(channel="pair_overflow", ordered=ordered):
            loud = _make_aggregate(scene, device, ordered=ordered, verify=True)
            quiet = _make_aggregate(scene, device, ordered=ordered)
            test.assertIsNone(quiet.verify_flag)
            np.testing.assert_array_equal(loud.verify_flag.numpy(), [1])
            for bp, expected in ((loud, 1), (quiet, 0)):
                output = _device_output(lambda bp=bp: _aggregate_pair_set(bp, scene, device, capacity=n // 2))
                test.assertEqual(output.count("candidate pair buffer overflowed"), expected, output)
                test.assertEqual(output.count(f"{n} > {n // 2}"), expected, output)
                test.assertEqual(bp.statistics()["pair_overflow"], 1)
            output = _device_output(lambda bp=loud: _aggregate_pair_set(bp, scene, device))
            test.assertEqual(output, "")
            loud.verify_flag.zero_()
            output = _device_output(lambda bp=loud: _aggregate_pair_set(bp, scene, device, capacity=n // 2))
            test.assertEqual(output, "")

    with test.subTest(channel="aggregate_pair_overflow"):
        small = _make_aggregate(scene, device, aggregate_pairs_max=4, verify=True)
        output = _device_output(lambda: _aggregate_pair_set(small, scene, device))
        demand = small.statistics()["aggregate_pair_count"]
        test.assertEqual(output.count("aggregate-pair list overflowed"), 1, output)
        test.assertIn(f"{demand} > 4", output)
        test.assertEqual(output.count("candidate pair buffer overflowed"), 0, output)

    with test.subTest(channel="aggregate_extent_overflow"):
        reach = _grid_reach(scene)
        grid = _make_aggregate(scene, device, main="grid", shape_reach=reach, verify=True)
        test.assertEqual(_device_output(lambda: _aggregate_pair_set(grid, scene, device, arrays=arrays)), "")
        # Two members' boxes grow to 10x the reach: the latch is set by one of them, and printed once.
        grown_lower = scene["lower"].copy()
        grown_upper = scene["upper"].copy()
        for a in (1, 2):
            shape = int(np.flatnonzero(scene["aggregate"] == a)[0])
            centre = 0.5 * (scene["lower"][shape] + scene["upper"][shape])
            grown_lower[shape] = centre - 10.0 * reach[shape]
            grown_upper[shape] = centre + 10.0 * reach[shape]
        arrays["lower"].assign(grown_lower)
        arrays["upper"].assign(grown_upper)
        outputs = [_device_output(lambda: _aggregate_pair_set(grid, scene, device, arrays=arrays)) for _ in range(2)]
        test.assertEqual(outputs[0].count("does not fit the grid cell"), 1, outputs[0])
        test.assertIn(f"{grid.cell_size:f}", outputs[0])
        test.assertEqual(outputs[1], "")
        grid.reset_statistics()
        output = _device_output(lambda: _aggregate_pair_set(grid, scene, device, arrays=arrays))
        test.assertEqual(output.count("does not fit the grid cell"), 1, output)
        test.assertEqual(grid.statistics()["aggregate_extent_overflow"], 1)
        quiet = _make_aggregate(scene, device, main="grid", shape_reach=reach)
        test.assertEqual(_device_output(lambda: _aggregate_pair_set(quiet, scene, device, arrays=arrays)), "")
        test.assertEqual(quiet.statistics()["aggregate_extent_overflow"], 1)


def test_aggregate_chunks_keep_bodies_whole(test, device):
    """Chunks are cut over whole bodies, so a body's shapes share an aggregate even when not consecutive.

    A 40-link chain gets a second shape on links 0 and 30 after all the other
    shapes. Cut in shape order, link 0's shapes would straddle the first cut;
    ordered by each body's first shape, every body stays in one chunk, and the
    first chunk takes link 30's two shapes at 31 members, so it holds 33, one
    body's shapes past ``max_aggregate_size``. The pipeline builds on both
    main levels and finds nxn's pairs.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    joints = []
    links = []
    parent = -1
    for k in range(40):
        link = builder.add_link(xform=wp.transform(wp.vec3(0.4 * k, 0.0, 0.15), wp.quat_identity()))
        builder.add_shape_capsule(link, radius=0.1, half_height=0.15)
        parent_xform = wp.transform(
            wp.vec3(0.4 if parent >= 0 else 0.0, 0.0, 0.0 if parent >= 0 else 0.15), wp.quat_identity()
        )
        joints.append(builder.add_joint_revolute(parent, link, parent_xform=parent_xform, axis=newton.Axis.Y))
        links.append(link)
        parent = link
    builder.add_articulation(joints)
    for k in (0, 30):
        builder.add_shape_sphere(links[k], xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity()), radius=0.05)
    model = builder.finalize(device=device)

    aggregate = broad_phase_aggregate.infer_shape_aggregates(model)
    body = model.shape_body.numpy()
    members = aggregate >= 0
    for b in np.unique(body[members]):
        test.assertEqual(len(np.unique(aggregate[members & (body == b)])), 1, f"body {b}")
    test.assertEqual(np.bincount(aggregate[members]).tolist(), [33, 9])

    state = model.state()
    _, _, nxn_set = _model_aabbs_and_nxn(model, state, device)
    test.assertGreaterEqual(len(nxn_set), 40)
    for knobs, level in (({}, "sap"), ({"grid_min_aggregates": 0}, "grid")):
        with test.subTest(level=level):
            with mock.patch.dict(broad_phase_aggregate._KNOBS, knobs):
                pipeline = newton.CollisionPipeline(model, broad_phase="aggregate")
            stats = pipeline.broad_phase.statistics()
            test.assertEqual((stats["main_level"], stats["aggregates"], stats["largest_aggregate"]), (level, 2, 33))
            contacts = pipeline.contacts()
            pipeline.collide(state, contacts)
            count = int(pipeline.broad_phase_pair_count.numpy()[0])
            test.assertEqual(_pair_set(pipeline.broad_phase_shape_pairs.numpy(), count), nxn_set)
            test.assertEqual(pipeline.broad_phase_statistics()["aggregate_extent_overflow"], 0)


def test_aggregate_grid_moving_boxes_match_nxn(test, device):
    """The grid follows moving, swept aggregates: every frame's set equals nxn's, and the latch stays 0.

    Each frame, every aggregate moves rigidly to within 1.5 m of its start on
    each axis (about a cell, so aggregates change cells and buckets), and every
    member sweeps by up to the 0.1 m clamp. The reach is the scene's 0.65 m
    plus the clamp, which bounds each swept member box about its aggregate's
    centre. Ordered, the grid's list equals the sweep's.
    """
    scene = _aggregate_scene(5)
    limit = 0.1
    reach = _grid_reach(scene) + np.float32(limit)
    members = scene["aggregate"] >= 0
    rng = np.random.default_rng(5)
    grid = _make_aggregate(scene, device, main="grid", shape_reach=reach)
    ordered = {
        main: _make_aggregate(scene, device, main=main, shape_reach=reach, ordered=True) for main in ("sap", "grid")
    }
    arrays = _aggregate_arrays(scene, device)
    cells = set()
    for frame in range(6):
        offset = rng.uniform(-1.5, 1.5, (_AGGREGATE_COUNT, 3)).astype(np.float32)
        lower = scene["lower"].copy()
        upper = scene["upper"].copy()
        lower[members] += offset[scene["aggregate"][members]]
        upper[members] += offset[scene["aggregate"][members]]
        displacement = rng.uniform(-limit, limit, (len(scene["world"]), 3)).astype(np.float32)
        displacement[scene["body"] < 0] = 0.0
        moved = {**scene, "lower": lower, "upper": upper}
        arrays["lower"].assign(lower)
        arrays["upper"].assign(upper)
        with test.subTest(frame=frame):
            nxn_set = _nxn_pair_set(moved, device, displacement=displacement)
            agg_set, count, _ = _aggregate_pair_set(
                grid, moved, device, displacement=displacement, limit=limit, arrays=arrays
            )
            test.assertEqual(agg_set, nxn_set)
            test.assertEqual(count, len(agg_set))
            stats = grid.statistics()
            test.assertEqual((stats["aggregate_extent_overflow"], stats["aggregate_pair_overflow"]), (0, 0))
            found = {(int(a), int(b)) for a, b in grid.aggregate_pairs.numpy()[: stats["aggregate_pair_count"]]}
            test.assertEqual(found, _host_aggregate_pairs(grid))
            cells.add(tuple(map(tuple, grid.grid_cell.numpy())))
            lists = []
            for bp in ordered.values():
                _, count, raw = _aggregate_pair_set(
                    bp, moved, device, displacement=displacement, limit=limit, arrays=arrays
                )
                lists.append(raw[:count])
            test.assertTrue(np.array_equal(lists[0], lists[1]))
            test.assertEqual(_pair_set(lists[1], len(lists[1])), nxn_set)
    test.assertEqual(len(cells), 6)
    test.assertEqual(ordered["grid"].statistics()["aggregate_extent_overflow"], 0)


_aggregate_devices = get_test_devices()
for _name, _func in (
    ("test_aggregate_matches_nxn_random", test_aggregate_matches_nxn_random),
    ("test_aggregate_kinematic_pairs_match_nxn", test_aggregate_kinematic_pairs_match_nxn),
    ("test_aggregate_group_rules_match_host_reference", test_aggregate_group_rules_match_host_reference),
    ("test_aggregate_runtime_group_change_matches_nxn", test_aggregate_runtime_group_change_matches_nxn),
    ("test_aggregate_membership_from_model", test_aggregate_membership_from_model),
    ("test_aggregate_finite_statics_from_model_match_nxn", test_aggregate_finite_statics_from_model_match_nxn),
    ("test_aggregate_prismatic_articulation_matches_nxn", test_aggregate_prismatic_articulation_matches_nxn),
    ("test_aggregate_swept_matches_nxn_within_clamp", test_aggregate_swept_matches_nxn_within_clamp),
    ("test_aggregate_two_instances_agree", test_aggregate_two_instances_agree),
    ("test_aggregate_pair_overflow_is_reported", test_aggregate_pair_overflow_is_reported),
    ("test_aggregate_level_overflow_is_reported", test_aggregate_level_overflow_is_reported),
    ("test_aggregate_knobs_agree", test_aggregate_knobs_agree),
    ("test_aggregate_ordered_lists_are_identical", test_aggregate_ordered_lists_are_identical),
    ("test_aggregate_ordered_list_order", test_aggregate_ordered_list_order),
    ("test_aggregate_ordered_overflow_keeps_a_prefix", test_aggregate_ordered_overflow_keeps_a_prefix),
    ("test_aggregate_ordered_bound_raises", test_aggregate_ordered_bound_raises),
    ("test_aggregate_statics_match_nxn_random", test_aggregate_statics_match_nxn_random),
    ("test_aggregate_statics_grid_variants_match_nxn", test_aggregate_statics_grid_variants_match_nxn),
    ("test_aggregate_statics_knob_agree", test_aggregate_statics_knob_agree),
    ("test_aggregate_world_static_visibility", test_aggregate_world_static_visibility),
    (
        "test_aggregate_rejects_global_aggregates_with_world_statics",
        test_aggregate_rejects_global_aggregates_with_world_statics,
    ),
    ("test_aggregate_static_grid_rebuild_rules", test_aggregate_static_grid_rebuild_rules),
    (
        "test_aggregate_static_overflow_hides_left_out_statics",
        test_aggregate_static_overflow_hides_left_out_statics,
    ),
    ("test_aggregate_statics_ordered_lists", test_aggregate_statics_ordered_lists),
    ("test_aggregate_grid_matches_nxn_random", test_aggregate_grid_matches_nxn_random),
    ("test_aggregate_grid_splits_prismatic_articulation", test_aggregate_grid_splits_prismatic_articulation),
    ("test_aggregate_grid_reach_from_model", test_aggregate_grid_reach_from_model),
    (
        "test_aggregate_grid_from_model_without_articulations",
        test_aggregate_grid_from_model_without_articulations,
    ),
    ("test_aggregate_grid_bucket_collisions", test_aggregate_grid_bucket_collisions),
    ("test_aggregate_grid_cell_holds_exact_fill", test_aggregate_grid_cell_holds_exact_fill),
    ("test_aggregate_from_model_defaults_to_grid", test_aggregate_from_model_defaults_to_grid),
    ("test_aggregate_direct_construction_falls_back_to_sap", test_aggregate_direct_construction_falls_back_to_sap),
    ("test_aggregate_splits_long_chains_by_default", test_aggregate_splits_long_chains_by_default),
    ("test_aggregate_grid_reach_per_chunk", test_aggregate_grid_reach_per_chunk),
    ("test_aggregate_grid_cell_guard", test_aggregate_grid_cell_guard),
    ("test_aggregate_default_main_level_by_size", test_aggregate_default_main_level_by_size),
    ("test_aggregate_default_main_level_by_worlds", test_aggregate_default_main_level_by_worlds),
    ("test_aggregate_default_main_level_by_occupancy", test_aggregate_default_main_level_by_occupancy),
    ("test_aggregate_grid_extent_latch", test_aggregate_grid_extent_latch),
    ("test_aggregate_verify_prints_device_warnings", test_aggregate_verify_prints_device_warnings),
    ("test_aggregate_chunks_keep_bodies_whole", test_aggregate_chunks_keep_bodies_whole),
    ("test_aggregate_grid_moving_boxes_match_nxn", test_aggregate_grid_moving_boxes_match_nxn),
):
    add_function_test(TestBroadPhaseAggregate, _name, _func, devices=_aggregate_devices)
for _name, _func in (
    ("test_aggregate_capture_replay_matches_eager", test_aggregate_capture_replay_matches_eager),
    ("test_aggregate_statics_capture_replay", test_aggregate_statics_capture_replay),
    ("test_aggregate_verify_capture_replay", test_aggregate_verify_capture_replay),
):
    add_function_test(TestBroadPhaseAggregate, _name, _func, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
