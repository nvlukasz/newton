# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Guards of the scene sweep's restated laws: each mirrors Steel's current text statement for statement.

The scene sweep restates some of Steel's laws instead of pasting them (its friction modes and contact torsion's budget
round every operation on its own, and its rows live in other arrays). ``scene_pgs._mirrored_lines`` normalises the
scene's text and Steel's (rounding intrinsics as operators, each side's names mapped onto common ones, comments and
layout dropped) so that a law Steel changes fails here instead of leaving the scene's copy behind.
"""

import unittest

from newton._src.solvers.steel import scene_pgs, solver_steel

LAWS = {
    "coulomb_newton",
    "bisection_free",
    "bisection_dense",
    "desaxce",
    "patch_load",
    "torsion_budget",
    "torsion_budget_linked",
    "torsion_budget_group",
    "torsion_budget_group_linked",
}
"""The restated laws: the friction modes' Coulomb-Newton solve, bisection (Steel's free-body and dense texts) and de
Saxce's bias, the friction patches' pooled normal load, and contact torsion's budget in each of the scene's row loops
(one thread, a lane group) against ``torque_sweep_source`` in each of its forms (rows scanned, rows linked)."""

COULOMB_NEWTON_CLAMP = "new_lambda_n = fmaxf(-(wNT0 * rT0 + wNT1 * rT1 + bN) / WN, 0.0f);"
"""Steel's Coulomb-Newton normal impulse, clamped at zero load."""


class TestSceneSweepLaws(unittest.TestCase):
    def test_restated_steel_laws_match_steel_source(self):
        """Each law the scene sweep restates is Steel's current law, statement for statement."""
        self.assertEqual(set(scene_pgs._MIRRORS), LAWS)
        for name in sorted(LAWS):
            with self.subTest(law=name):
                scene, steel = scene_pgs._mirrored_lines(name)
                self.assertGreaterEqual(len(steel), 2, steel)
                self.assertEqual(scene, steel)

    def test_mirrored_lines_see_a_changed_law(self):
        """A change to one of Steel's restated laws changes its mirrored lines, so the guard fails on it.

        Steel's Coulomb-Newton update without its zero-load clamp, and its torsion budget pooling the members' signed
        normal impulses instead of their positive parts, no longer match the scene's.
        """
        source = solver_steel._MF_COULOMB_NEWTON_FRICTION_SOURCE
        self.assertEqual(source.count(COULOMB_NEWTON_CLAMP), 1)
        unclamped = source.replace(COULOMB_NEWTON_CLAMP, "new_lambda_n = -(wNT0 * rT0 + wNT1 * rT1 + bN) / WN;")
        scene, steel = scene_pgs._mirrored_lines("coulomb_newton", steel=unclamped)
        self.assertNotEqual(scene, steel)
        torsion = scene_pgs._MIRRORS["torsion_budget"].steel()
        pooled = "normal_load += fmaxf(s_lam_dense[n], 0.0f);"
        self.assertEqual(torsion.count(pooled), 1)
        signed = torsion.replace(pooled, "normal_load += s_lam_dense[n];")
        scene, steel = scene_pgs._mirrored_lines("torsion_budget", steel=signed)
        self.assertNotEqual(scene, steel)


if __name__ == "__main__":
    unittest.main()
