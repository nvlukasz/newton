# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The scene mode's test seams in SolverSteel (``solve_mode="scene"``): the kernel overrides it refuses or ignores,
the default scene kernels' tripwire and the host shim that compiles the scene's thread loop for the CPU."""

import hashlib
import re
import unittest

import numpy as np

from newton._src.solvers.steel._testing import kernel_overrides
from newton._src.solvers.steel.scene_pgs import scene_kernel_snippets
from newton._src.solvers.steel.scene_rows import _response_snippet
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import _run
from newton.tests.steel._scene_fixtures import DEFAULT_SNIPPET_SHA256, _humanoids
from newton.tests.steel.test_scene_sweep import SWEEP_CAPACITY, _launch_scene_serial
from newton.tests.steel.test_sparse import _SWEEP_CASES, _problem, _sweep
from newton.tests.unittest_utils import add_function_test, get_test_devices

HOST_SHIM_SUPPORT = 30
"""Entries per row of :func:`test_host_shim_compiles_the_thread_loop`'s serial sweep: ``_problem``'s rows hold 18, so
the even support loads its indices in pairs (``int2``), the rows with a 16th entry their wide pairs too, and a row
without one its wide pairs as ``make_int2(-1, -1)``."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_explicit_scene_with_dense_override_raises(test, device):
    """The scene with the dense mass matrix override raises; any other Steel-only kernel override is ignored.

    The scene has no dense factors, so ``sparse_mass_matrix=False`` raises at construction. Any other key of Steel's
    kernel choices is accepted and ignored (no new rejection): with ``pgs_kernel="loop"`` the scene builds and steps bit
    for bit as without the override.
    """
    model = _humanoids(device)
    with kernel_overrides({"sparse_mass_matrix": False}):
        with test.assertRaisesRegex(NotImplementedError, re.escape("sparse_mass_matrix=False")):
            SolverSteel(model, solve_mode="scene")
    with test.subTest(override="pgs_kernel"):
        _, q_plain, qd_plain, _ = _run(model, solve_mode="scene")
        solver, q, qd, _ = _run(model, overrides={"pgs_kernel": "loop"}, solve_mode="scene")
        test.assertEqual(solver.solve_mode, "scene")
        test.assertEqual(solver._kernel_overrides, {"pgs_kernel": "loop"})
        np.testing.assert_array_equal(q, q_plain)
        np.testing.assert_array_equal(qd, qd_plain)


def test_host_shim_compiles_the_thread_loop(test, device):
    """The serial sweep compiles for the CPU through the host shim and sweeps as the host reference.

    ``get_sweep_rows_serial_kernel("steel", 30)``, Steel's sums over :data:`HOST_SHIM_SUPPORT` entries, on
    ``test_sparse._problem(43, 18)``'s eight rows loaded at W = 1 (entries 18 to 29 empty) solves its tangent pairs with
    ``FRICTION_PAIR_CUDA``'s ``make_float2``, ``fmaxf``, ``sqrtf`` and ``fabsf``: unseeded and seeded impulses at every
    relaxation of ``test_unit_sweep_matches_host_reference``, against its host reference ``_sweep``.
    """
    support = HOST_SHIM_SUPPORT
    for seeded, omega in ((seeded, omega) for seeded in (False, True) for omega in _SWEEP_CASES):
        _, indices, rows, initial, bias, types, parents, mu, impulses, _, _, incident, diag = _problem(43, 18, seeded)
        expected, expected_impulses = _sweep(
            rows, rows, np.zeros_like(initial), incident + bias, diag, types, parents, mu, impulses, omega=omega
        )
        # The eight rows on their 18 coordinates, then -1 / 0 to the support; row 0 has one entry.
        row_dof = np.full((1, SWEEP_CAPACITY, support), -1, dtype=np.int32)
        row_dof[0, :8, : len(indices)] = indices
        row_dof[0, 0, 1:] = -1
        row_factor = np.zeros((1, SWEEP_CAPACITY, support), dtype=np.float32)
        row_factor[0, :8, : len(indices)] = rows[:, indices]
        with test.subTest(seeded=seeded, omega=omega):
            delta, actual_impulses = _launch_scene_serial(
                device,
                sums="steel",
                support=support,
                free_response=False,
                rows=8,
                row_dof=row_dof,
                row_factor=row_factor,
                row_free_response=np.zeros((1, 1, 1), dtype=np.float32),
                row_free_prefix=np.zeros(SWEEP_CAPACITY, dtype=np.int32),
                bias=bias,
                incident=incident,
                diag=diag,
                types=types,
                parents=parents,
                mu=mu,
                impulses=impulses,
                coordinates=43,
                omega=omega,
            )
            np.testing.assert_allclose(delta, expected, atol=2.0e-5, rtol=2.0e-5)
            np.testing.assert_allclose(actual_impulses, expected_impulses, atol=2.0e-5, rtol=2.0e-5)


class TestSteelSceneHarness(unittest.TestCase):
    def test_default_scene_kernels_are_unchanged(self):
        """The default scene kernels' native source is the source recorded in ``DEFAULT_SNIPPET_SHA256``."""
        hashes = {}
        for support, free_response, dof_max in ((21, False, 27), (30, False, 27), (12, True, 6)):
            for key, snippet in scene_kernel_snippets(support, free_response=free_response, dof_max=dof_max).items():
                hashes[f"{support}/{free_response}/{key}"] = _sha256(snippet)
        hashes["sparse_contact/27"] = _sha256(_response_snippet(27, mask_words=1, specialized=True))
        self.assertEqual(hashes, DEFAULT_SNIPPET_SHA256)


add_function_test(
    TestSteelSceneHarness,
    "test_explicit_scene_with_dense_override_raises",
    test_explicit_scene_with_dense_override_raises,
    devices=get_test_devices(),
)

add_function_test(
    TestSteelSceneHarness,
    "test_host_shim_compiles_the_thread_loop",
    test_host_shim_compiles_the_thread_loop,
    devices=[device for device in get_test_devices() if device.is_cpu],
)


if __name__ == "__main__":
    unittest.main()
