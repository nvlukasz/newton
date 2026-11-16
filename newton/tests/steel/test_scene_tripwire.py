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

from newton._src.solvers.steel import contact_compliance, contact_torsion, friction, solver_steel, sparse_contact
from newton.tests.steel._scene_fixtures import STEEL_TEXT_SHA256

MF_GS_SHAPE = (128, 64, 27)
"""``(max_constraints, free_body_constraint_max, max_world_dofs)`` of the pinned fused matrix-free kernels: the
fixtures' row capacity, free-body row capacity and the DOFs of one ``nv_humanoid.xml`` humanoid."""

MF_GS_DEVICE_ARCH = 0
"""``device_arch`` of the pinned fused kernels: part of the factory's cache key only, not of the text."""

SPARSE_CONTACT_DOFS = 27
"""DOFs of the pinned sparse contact response kernel (one humanoid)."""

TORSION_DOFS = 12
"""DOFs of the pinned contact torsion sweeps (two free bodies)."""


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
    }


def steel_kernel_hashes() -> dict[str, str]:
    """SHA-256 of each text of :func:`steel_kernel_texts`."""
    return {key: _sha256(text) for key, text in steel_kernel_texts().items()}


def record() -> None:
    """Print :data:`STEEL_TEXT_SHA256` as recorded from the current text, ready to paste."""
    print("STEEL_TEXT_SHA256: dict[str, str] = {")
    for key, digest in steel_kernel_hashes().items():
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

    @unittest.skip("the scene kernels do not exist yet")
    def test_scene_kernels_are_pinned(self):
        """The default scene kernels' text is the text recorded in ``DEFAULT_SNIPPET_SHA256``."""


if __name__ == "__main__":
    if "--record" in sys.argv:
        record()
    else:
        unittest.main()
