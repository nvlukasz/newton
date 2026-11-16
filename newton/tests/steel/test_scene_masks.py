# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Multi-word support masks of SolverSteel's sparse mass-matrix plan.

Steel's sparse path builds one-word masks, and its sparse mass-factor kernel at one word is the text pinned in
``STEEL_TEXT_SHA256``. The scene mode builds masks of ``ceil(DOFs / 64)`` words for articulations past 64 DOFs:
a plan's masks, their device layout and their reading as Python ints.
"""

import hashlib
import unittest

import numpy as np

from newton._src.solvers.steel.sparse_mass_matrix import (
    _get_crba_sparse_factor_kernel,
    _SparseMassMatrixPlan,
    distinct_mask_values,
    mask_value,
)
from newton.tests.steel._scene_fixtures import STEEL_TEXT_SHA256
from newton.tests.steel.test_scene_tripwire import FACTOR_SHAPE, factor_kernel_text
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _chain(links: int) -> tuple[np.ndarray, np.ndarray]:
    """``(joint_parent, joint_dof_count)`` of a serial chain of ``links`` one-DOF joints."""
    return np.arange(-1, links - 1, dtype=np.int32), np.ones(links, dtype=np.int32)


class TestSteelSceneMasks(unittest.TestCase):
    def test_one_word_factor_kernel_text_is_steels(self):
        """Build the sparse mass-factor kernel at one word with the text of Steel's, pinned in ``STEEL_TEXT_SHA256``.

        One word keeps Steel's bound of 64 DOFs, which two words lift.
        """
        text = factor_kernel_text(mask_words=1)
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), STEEL_TEXT_SHA256["factor_kernel"])
        self.assertEqual(text, factor_kernel_text())
        _, _, warps = FACTOR_SHAPE
        with self.assertRaisesRegex(ValueError, "Invalid sparse factor dimensions"):
            _get_crba_sparse_factor_kernel(65, 65, warps_per_block=warps, mask_words=1)
        _get_crba_sparse_factor_kernel(65, 65, warps_per_block=warps, mask_words=2)


def test_multi_word_masks_round_trip(test, device):
    """Give a plan support masks of ``mask_words`` words, laid out on the device word by word, read back as ints.

    A 65-link chain's plan has two-word masks, ``[coordinates, 2]`` with word 0 holding coordinates
    0 to 63: its tip's mask covers all 65, its root's one. The device indices hold a coordinate's
    words consecutively. Steel's one-word plan refuses the chain, and three words widen a two-link
    chain's masks without changing their coordinates. :func:`distinct_mask_values` on random one-,
    two- and three-word masks with repeats equals the set of every row's int.
    """
    chain = _SparseMassMatrixPlan.build(*_chain(65), mask_words=2)
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
        _SparseMassMatrixPlan.build(*_chain(65))
    # One word stays the one-dimensional masks of Steel's plans; a wider plan keeps their coordinates.
    one = _SparseMassMatrixPlan.build(*_chain(2))
    test.assertEqual(one.mask_words, 1)
    test.assertEqual(one.ancestor_mask.shape, (2,))
    test.assertEqual(one.ancestor_mask.dtype, np.uint64)
    wide = _SparseMassMatrixPlan.build(*_chain(2), mask_words=3)
    test.assertEqual(wide.ancestor_mask.shape, (2, 3))
    test.assertEqual([mask_value(mask) for mask in wide.ancestor_mask], [int(mask) for mask in one.ancestor_mask])
    test.assertTrue(np.array_equal(one.to_device(device).ancestor_mask.numpy(), one.ancestor_mask))
    with test.assertRaisesRegex(ValueError, "at least one word"):
        _SparseMassMatrixPlan.build(*_chain(2), mask_words=0)

    rng = np.random.default_rng(1707)
    for words in (1, 2, 3):
        with test.subTest(words=words):
            pool = rng.integers(0, 2**63, size=(12, words), dtype=np.uint64) * np.uint64(2) + np.uint64(1)
            pool[0] = 0
            pool[1] = np.iinfo(np.uint64).max
            masks = pool[rng.integers(0, len(pool), size=200)]
            masks = masks[:, 0] if words == 1 else masks
            test.assertEqual(distinct_mask_values(masks), sorted({mask_value(row) for row in masks}))


add_function_test(
    TestSteelSceneMasks,
    "test_multi_word_masks_round_trip",
    test_multi_word_masks_round_trip,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
