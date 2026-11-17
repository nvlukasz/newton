# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The scene mode's test seams in SolverSteel (``solve_mode="scene"``): the kernel overrides it refuses or ignores,
the default scene kernels' tripwire and the host shim that compiles the scene's thread loop for the CPU."""

import contextlib
import hashlib
import importlib
import json
import os
import re
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np

from newton import NewtonWarning
from newton._src.solvers.steel._testing import kernel_overrides
from newton._src.solvers.steel.scene_pgs import scene_kernel_snippets
from newton._src.solvers.steel.scene_rows import _response_snippet
from newton.solvers import SolverSteel
from newton.tests.steel import _scene_parity
from newton.tests.steel._fixtures import _run
from newton.tests.steel._scene_fixtures import DEFAULT_SNIPPET_SHA256, _chain_worlds, _humanoids
from newton.tests.steel.test_scene_sweep import SWEEP_CAPACITY, _launch_scene_serial
from newton.tests.steel.test_sparse import _SWEEP_CASES, _problem, _sweep
from newton.tests.unittest_utils import add_function_test, get_test_devices

HOST_SHIM_SUPPORT = 30
"""Entries per row of :func:`test_host_shim_compiles_the_thread_loop`'s serial sweep: ``_problem``'s rows hold 18, so
the even support loads its indices in pairs (``int2``), the rows with a 16th entry their wide pairs too, and a row
without one its wide pairs as ``make_int2(-1, -1)``."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@contextlib.contextmanager
def _fresh_parity():
    """Let a test install the forced run's wrappers afresh, then put back what was installed before.

    Inside, ``SolverSteel.__init__`` and ``unittest.TestCase.run`` are their unwrapped originals, so the test's own
    installs are the only ones active even when the whole suite runs forced; on exit the previous functions return.
    """
    init, run = SolverSteel.__init__, unittest.TestCase.run
    SolverSteel.__init__ = _scene_parity.unwrapped(init)
    unittest.TestCase.run = _scene_parity.unwrapped(run)
    try:
        yield
    finally:
        SolverSteel.__init__, unittest.TestCase.run = init, run


def _records(directory: str) -> list[dict]:
    """The recorder's records in ``directory`` (one file per process), in file order."""
    records = []
    for name in sorted(os.listdir(directory)):
        with open(os.path.join(directory, name), encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    return records


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

    def test_forced_scene_keeps_explicit_solve_modes(self):
        """The forced run builds ``solve_mode="scene"`` where no mode is given and keeps an explicit mode.

        A construction whose options the mode rejects (``_scene_parity.KEPT_ON_STEEL``) keeps Steel's default and is
        recorded with the option it was kept for; every record names the test that built the solver.
        """
        model = _chain_worlds([3], "cpu")
        modes = {}

        class Probe(unittest.TestCase):
            def test_builds(self):
                modes["default"] = SolverSteel(model).solve_mode
                modes["serial"] = SolverSteel(model, solve_mode="serial").solve_mode
                with warnings.catch_warnings():
                    # The colored mode warns that it runs the serial propagation sweep on the CPU.
                    warnings.simplefilter("ignore", NewtonWarning)
                    colored = SolverSteel(model, solve_mode="colored", articulated_contact_response="propagation")
                modes["colored"] = colored.solve_mode
                modes["reuse_kinematics"] = SolverSteel(model, reuse_kinematics=True).solve_mode
                with kernel_overrides({"sparse_mass_matrix": False}):
                    modes["dense"] = SolverSteel(model).solve_mode

        with tempfile.TemporaryDirectory() as directory:
            with _fresh_parity():
                _scene_parity.install_forced_scene()
                _scene_parity.install_recorder(directory)
                result = unittest.TestResult()
                Probe("test_builds").run(result)
            records = _records(directory)
        self.assertEqual(result.errors + result.failures, [])
        self.assertEqual(
            modes,
            {
                "default": "scene",
                "serial": "serial",
                "colored": "colored",
                "reuse_kinematics": "serial",
                "dense": "serial",
            },
        )
        self.assertEqual(
            [(r["test"], r["requested"], r["solve_mode"], r["effective"], r["kept"]) for r in records],
            [
                ("Probe::test_builds", None, "scene", True, None),
                ("Probe::test_builds", "serial", "serial", False, None),
                ("Probe::test_builds", "colored", "colored", False, None),
                ("Probe::test_builds", None, "serial", False, "reuse_kinematics=True"),
                ("Probe::test_builds", None, "serial", False, "kernel override sparse_mass_matrix=False"),
            ],
        )
        self.assertEqual([r["forced_from"] for r in records], [None] * 5)
        self.assertEqual(records[3]["options"], {"reuse_kinematics": True})
        self.assertEqual(records[4]["overrides"], ["sparse_mass_matrix"])
        self.assertEqual(
            _scene_parity.kept_on_steel({"nesterov_momentum": True, "reuse_kinematics": True}, {}),
            "nesterov_momentum=True, reuse_kinematics=True",
        )

    def test_forced_scene_forces_explicit_serial_in_upstream_modules(self):
        """The forced run builds an explicit ``solve_mode="serial"`` with the mode in a test of Steel's own modules.

        In a scene module (``_scene_parity.SCENE_TEST_MODULES``) an explicit ``"serial"`` is a reference and is kept;
        an explicit ``"colored"`` is kept everywhere; a construction with a ``KEPT_ON_STEEL`` option keeps ``"serial"``.
        A serial construction forced to the mode is recorded with ``forced_from="serial"``.
        """
        model = _chain_worlds([3], "cpu")
        modes = {}

        def build(self):
            where = type(self).__module__.rsplit(".", 1)[-1]
            modes[(where, "default")] = SolverSteel(model).solve_mode
            modes[(where, "serial")] = SolverSteel(model, solve_mode="serial").solve_mode
            with warnings.catch_warnings():
                # The colored mode warns that it runs the serial propagation sweep on the CPU.
                warnings.simplefilter("ignore", NewtonWarning)
                colored = SolverSteel(model, solve_mode="colored", articulated_contact_response="propagation")
            modes[(where, "colored")] = colored.solve_mode
            kept = SolverSteel(model, solve_mode="serial", reuse_kinematics=True)
            modes[(where, "serial reuse_kinematics")] = kept.solve_mode

        # Test classes as Steel's own test_options and the scene's test_scene_rows would define them.
        upstream = type("Probe", (unittest.TestCase,), {"test_builds": build, "__module__": "steel.test_options"})
        scene = type("Probe", (unittest.TestCase,), {"test_builds": build, "__module__": "steel.test_scene_rows"})
        with tempfile.TemporaryDirectory() as directory:
            with _fresh_parity():
                _scene_parity.install_forced_scene()
                _scene_parity.install_recorder(directory)
                result = unittest.TestResult()
                upstream("test_builds").run(result)
                scene("test_builds").run(result)
            records = _records(directory)
        self.assertEqual(result.errors + result.failures, [])
        self.assertTrue({"test_coloring", "test_scene_rows", "test_scene_harness"} <= _scene_parity.SCENE_TEST_MODULES)
        self.assertNotIn("test_options", _scene_parity.SCENE_TEST_MODULES)
        self.assertEqual(
            modes,
            {
                ("test_options", "default"): "scene",
                ("test_options", "serial"): "scene",
                ("test_options", "colored"): "colored",
                ("test_options", "serial reuse_kinematics"): "serial",
                ("test_scene_rows", "default"): "scene",
                ("test_scene_rows", "serial"): "serial",
                ("test_scene_rows", "colored"): "colored",
                ("test_scene_rows", "serial reuse_kinematics"): "serial",
            },
        )
        self.assertEqual(
            [(r["requested"], r["solve_mode"], r["kept"], r["forced_from"]) for r in records],
            [
                (None, "scene", None, None),
                ("serial", "scene", None, "serial"),
                ("colored", "colored", None, None),
                ("serial", "serial", "reuse_kinematics=True", None),
                (None, "scene", None, None),
                ("serial", "serial", None, None),
                ("colored", "colored", None, None),
                ("serial", "serial", None, None),
            ],
        )

    def test_forced_scene_skips_listed_tests_with_their_reason(self):
        """The forced run skips a test listed in ``_scene_parity.SKIPS`` with ``"Steel path: "`` and its reason.

        A reason key reads its text from ``STEEL_PATH_REASONS``; any other entry (a scene counterpart) is the text
        itself. A key without a device suffix covers the test's device variants. Without the forced run the listed
        tests run.
        """
        ran = []

        class Probe(unittest.TestCase):
            def test_listed_cpu(self):
                ran.append("listed")

            def test_counterpart(self):
                ran.append("counterpart")

            def test_unlisted(self):
                ran.append("unlisted")

        listed = {
            "Probe::test_listed": "STEEL_ROW_BUDGETS",
            "Probe::test_counterpart": "scene counterpart: test_counterpart_under_scene",
        }
        names = ("test_listed_cpu", "test_counterpart", "test_unlisted")
        with tempfile.TemporaryDirectory() as directory, _fresh_parity(), mock.patch.dict(_scene_parity.SKIPS, listed):
            _scene_parity.install_recorder(directory)
            unforced = unittest.TestResult()
            for name in names:
                Probe(name).run(unforced)
            self.assertEqual((ran, unforced.skipped), (["listed", "counterpart", "unlisted"], []))
            ran.clear()
            _scene_parity.install_forced_scene()
            forced = unittest.TestResult()
            probes = [Probe(name) for name in names]
            for probe in probes:
                probe.run(forced)
        self.assertEqual(ran, ["unlisted"])
        self.assertEqual(
            forced.skipped,
            [
                (probes[0], "Steel path: " + _scene_parity.STEEL_PATH_REASONS["STEEL_ROW_BUDGETS"]),
                (probes[1], "Steel path: scene counterpart: test_counterpart_under_scene"),
            ],
        )
        self.assertNotIn("test_listed_cpu", vars(probes[0]))
        # The table's entries: a reason key, or a scene counterpart, a rejected feature or a recorded exception.
        prefixes = ("scene counterpart: ", "rejected: ", "exception: ")
        self.assertEqual(len(_scene_parity.STEEL_PATH_REASONS), 13)
        for test, entry in _scene_parity.SKIPS.items():
            with self.subTest(test=test):
                self.assertTrue(entry in _scene_parity.STEEL_PATH_REASONS or entry.startswith(prefixes), entry)

    def _registered_tests(self) -> set[tuple[str, str, str]]:
        """``(module, class, name)`` of every test this package registers, ``name`` without its device suffix."""
        registered = set()
        directory = os.path.dirname(os.path.abspath(__file__))
        for stem in sorted(name[:-3] for name in os.listdir(directory) if re.fullmatch(r"test_\w+\.py", name)):
            module = importlib.import_module(f"{__package__}.{stem}")
            for cls in vars(module).values():
                if isinstance(cls, type) and issubclass(cls, unittest.TestCase) and cls.__module__ == module.__name__:
                    for method in dir(cls):
                        if method.startswith("test"):
                            registered.add((stem, cls.__name__, _scene_parity._DEVICE_SUFFIX.sub("", method)))
        return registered

    def test_skips_name_registered_tests(self):
        """Every ``_scene_parity.SKIPS`` key names a test this package registers, so no entry silently never skips.

        A key ``"<Class>::<name>"`` matches a method ``name`` of the test class ``Class``, with or without a device
        suffix (a test with no device on this machine registers under its bare name).
        """
        registered = {f"{cls}::{name}" for _, cls, name in self._registered_tests()}
        self.assertEqual([key for key in _scene_parity.SKIPS if key not in registered], [])

    def test_skip_counterparts_name_registered_scene_tests(self):
        """Every ``"scene counterpart: <test>"`` of ``_scene_parity.SKIPS`` names a test a ``test_scene_*`` module
        registers (with or without a device suffix), so a skipped test's replacement runs."""
        scene_tests = {name for stem, _, name in self._registered_tests() if stem.startswith("test_scene_")}
        prefix = "scene counterpart: "
        counterparts = {entry[len(prefix) :] for entry in _scene_parity.SKIPS.values() if entry.startswith(prefix)}
        self.assertTrue(counterparts)
        self.assertEqual(sorted(counterparts - scene_tests), [])


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
