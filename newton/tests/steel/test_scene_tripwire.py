# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The tripwire of the scene mode: hashes of Steel's default kernel text and of the scene kernels' text.

The texts are built by the kernel factories on the host and never compiled, so the tests run on any device.
"""

import hashlib
import inspect
import sys
import unittest

import warp as wp

from newton._src.solvers.steel import (
    contact_compliance,
    contact_torsion,
    friction,
    scene_pgs,
    scene_rows,
    solver_steel,
    sparse_contact,
    sparse_mass_matrix,
)
from newton.tests.steel._scene_fixtures import DEFAULT_SNIPPET_SHA256, STEEL_TEXT_SHA256

MF_GS_SHAPE = (128, 64, 27)
"""``(max_constraints, free_body_constraint_max, max_world_dofs)`` of the pinned fused matrix-free kernels: the
fixtures' row capacity, free-body row capacity and the DOFs of one ``nv_humanoid.xml`` humanoid."""

MF_GS_DEVICE_ARCH = 0
"""``device_arch`` of the pinned fused kernels: part of the factory's cache key only, not of the text."""

SPARSE_CONTACT_DOFS = 27
"""DOFs of the pinned sparse contact response kernel (one humanoid)."""

TORSION_DOFS = 12
"""DOFs of the pinned contact torsion sweeps (two free bodies)."""

FACTOR_SHAPE = (27, 243, 4)
"""``(dof_count, nonzero_count, warps_per_block)`` of the pinned sparse mass-factor kernel: the factor plan of one
``nv_humanoid.xml`` humanoid and Steel's default warps per block."""

SCENE_CONFIGURATIONS = ((21, False, 27), (30, False, 27), (12, True, 6))
"""``(support, free_response, dof_max)`` of the pinned default scene kernels: rows of 21 entries (two humanoid
articulations together) and 30 entries (the widest Steel-order sums), each with groups of one humanoid's 27 DOFs, and a
scene of free bodies only (12 entries, a free-body prefix, groups of 6 DOFs)."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _native_snippet(kernel: wp.Kernel) -> str:
    """The native snippet of the one ``wp.func_native`` function a factory-built ``kernel`` calls."""
    cells = (cell.cell_contents for cell in kernel.func.__closure__ or ())
    snippets = [cell.native_snippet for cell in cells if isinstance(cell, wp.Function) and cell.native_snippet]
    if len(snippets) != 1:
        raise AssertionError(f"{kernel.key} calls {len(snippets)} native functions, expected one")
    return snippets[0]


def _mf_gs_snippet(**flags) -> str:
    """The native snippet of Steel's fused matrix-free Gauss-Seidel kernel at :data:`MF_GS_SHAPE`.

    The flags without a default take the values Steel picks with default options for this shape: no joint
    velocity limits, and resident row metadata (2 KiB of it).
    """
    kernel = solver_steel._get_pgs_solve_mf_gs_kernel(
        *MF_GS_SHAPE,
        MF_GS_DEVICE_ARCH,
        has_dense_velocity_limit_rows=False,
        shared_metadata=True,
        **flags,
    )
    return _native_snippet(kernel)


def factor_kernel_text(**options) -> str:
    """The native snippet of Steel's sparse mass-factor kernel at :data:`FACTOR_SHAPE`, built with ``options``."""
    dof_count, nonzero_count, warps = FACTOR_SHAPE
    kernel = sparse_mass_matrix._get_crba_sparse_factor_kernel(
        dof_count, nonzero_count, warps_per_block=warps, **options
    )
    return _native_snippet(kernel)


def steel_kernel_texts() -> dict[str, str]:
    """The default path's kernel texts that :data:`STEEL_TEXT_SHA256` pins, by key."""
    return {
        "mf_gs/128/64/27": _mf_gs_snippet(),
        "mf_gs/128/64/27/nesterov": _mf_gs_snippet(nesterov_momentum=True),
        f"sparse_contact/{SPARSE_CONTACT_DOFS}": _native_snippet(
            sparse_contact._get_sparse_contact_response_kernel(SPARSE_CONTACT_DOFS)
        ),
        "joint_limit_rows/1": sparse_contact._build_sparse_joint_limit_rows.native_snippet,
        f"torsion_sweep/{TORSION_DOFS}": contact_torsion.torque_sweep_source(TORSION_DOFS),
        f"torsion_sweep/{TORSION_DOFS}/linked": contact_torsion.torque_sweep_source(TORSION_DOFS, linked=True),
        "friction_pair": friction.FRICTION_PAIR_CUDA,
        "dense_bisection_friction": solver_steel._DENSE_BISECTION_FRICTION_SOURCE,
        "mf_bisection_friction": solver_steel._MF_BISECTION_FRICTION_SOURCE,
        "mf_desaxce_bias": solver_steel._MF_DESAXCE_BIAS_SOURCE,
        "mf_coulomb_newton_friction": solver_steel._MF_COULOMB_NEWTON_FRICTION_SOURCE,
        "compliance_prepare_contacts": inspect.getsource(contact_compliance._prepare_contacts.func),
        "factor_kernel": factor_kernel_text(),
    }


def steel_kernel_hashes() -> dict[str, str]:
    """SHA-256 of each text of :func:`steel_kernel_texts`."""
    return {key: _sha256(text) for key, text in steel_kernel_texts().items()}


def scene_kernel_texts() -> dict[str, str]:
    """The default scene kernels' texts that :data:`DEFAULT_SNIPPET_SHA256` pins, by key.

    ``"<support>/<free_response>/<kernel>"`` for each snippet of ``scene_pgs.scene_kernel_snippets`` at each of
    :data:`SCENE_CONFIGURATIONS`, and ``sparse_contact/27``, the scene's response template at one word specialised on
    one factor plan, which is Steel's own contact response.
    """
    texts = {}
    for support, free_response, dof_max in SCENE_CONFIGURATIONS:
        snippets = scene_pgs.scene_kernel_snippets(support, free_response=free_response, dof_max=dof_max)
        texts.update({f"{support}/{free_response}/{key}": snippet for key, snippet in snippets.items()})
    texts[f"sparse_contact/{SPARSE_CONTACT_DOFS}"] = scene_rows._response_snippet(SPARSE_CONTACT_DOFS, specialized=True)
    return texts


def scene_kernel_hashes() -> dict[str, str]:
    """SHA-256 of each text of :func:`scene_kernel_texts`."""
    return {key: _sha256(text) for key, text in scene_kernel_texts().items()}


def record() -> None:
    """Print :data:`STEEL_TEXT_SHA256` and :data:`DEFAULT_SNIPPET_SHA256` as recorded from the current text."""
    for name, hashes in (
        ("STEEL_TEXT_SHA256", steel_kernel_hashes()),
        ("DEFAULT_SNIPPET_SHA256", scene_kernel_hashes()),
    ):
        print(f"{name}: dict[str, str] = {{")
        for key, digest in hashes.items():
            print(f'    "{key}": "{digest}",')
        print("}")


class TestSteelSceneTripwire(unittest.TestCase):
    def test_default_steel_kernels_are_unchanged(self):
        """Steel's default kernel text is the text recorded in ``STEEL_TEXT_SHA256``."""
        hashes = steel_kernel_hashes()
        changed = sorted(
            key for key in hashes.keys() | STEEL_TEXT_SHA256.keys() if hashes.get(key) != STEEL_TEXT_SHA256.get(key)
        )
        self.assertEqual(
            changed,
            [],
            "Steel's kernel text differs from STEEL_TEXT_SHA256; if the change is intended, re-record the table "
            "(python -m newton.tests.steel.test_scene_tripwire --record) with the reason in the same commit",
        )

    def test_scene_kernels_are_pinned(self):
        """The default scene kernels' text is the text recorded in ``DEFAULT_SNIPPET_SHA256``.

        The scene's own text with every feature off, not Steel's: the guards of ``test_scene_pgs`` compare the laws
        it restates with Steel's text, and ``sparse_contact/27`` equals Steel's pin of its contact response.
        """
        hashes = scene_kernel_hashes()
        self.assertEqual(len(hashes), 19)
        changed = sorted(
            key
            for key in hashes.keys() | DEFAULT_SNIPPET_SHA256.keys()
            if hashes.get(key) != DEFAULT_SNIPPET_SHA256.get(key)
        )
        self.assertEqual(
            changed,
            [],
            "the default scene kernels' text differs from DEFAULT_SNIPPET_SHA256; if the change is intended, re-record "
            "the table (python -m newton.tests.steel.test_scene_tripwire --record) with the reason in the same commit",
        )
        self.assertEqual(DEFAULT_SNIPPET_SHA256["sparse_contact/27"], STEEL_TEXT_SHA256["sparse_contact/27"])


if __name__ == "__main__":
    if "--record" in sys.argv:
        record()
    else:
        unittest.main()
