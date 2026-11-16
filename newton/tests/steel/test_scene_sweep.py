# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The scene sweep of SolverSteel (``solve_mode="scene"``): phase A, the colors, the tail and the decode, against
Steel."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton import NewtonWarning
from newton._src.solvers.steel._testing import kernel_overrides
from newton._src.solvers.steel.scene_pgs import RANGE_ALL, SCENE_BLOCK_COLOR_MAX, get_sweep_rows_serial_kernel
from newton._src.solvers.steel.sparse_pgs import _get_pgs_solve_sparse_kernel
from newton.solvers import SolverBase, SolverSteel
from newton.tests.steel._fixtures import CONTACT_MAX, _build_contact_scene, _build_scissor_model, _heap, _tree
from newton.tests.steel._scene_fixtures import _humanoid, _humanoids, _leaning_humanoid, _steel_on
from newton.tests.steel.test_coloring import (
    _assert_color_order,
    _assert_valid_coloring,
    _chain_on_ground,
    _colored_view,
)
from newton.tests.steel.test_contacts import _net_body_forces, _stacked_boxes
from newton.tests.steel.test_sparse import _SWEEP_CASES, _mixed_problem, _problem, _sweep
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

DT = 1.0 / 200.0

# The options mavrix passes, without the scene solve.
COMMON = {"iterations": 4, "enable_joint_limits": True, "joint_limit_activation_gap": 0.1, "friction_anchor_beta": 0.0}

STEEL_ROWS = 128
"""Steel's rows per world of one humanoid, which scene mode sums over the worlds."""

STEEL_SUMS = {"scene_sum_order": "steel"}
SERIAL_SUMS = {"scene_sum_order": "serial"}

# Granularities and summation orders the serial oracle must reproduce bit for bit: Steel's sums at
# either granularity, or one thread per side and per unit with serial sums (8 lanes with serial sums
# add in another order than the oracle's one thread).
PHASE_CONFIGS = (
    {"scene_side_sweep": "lanes8", "scene_unit_sweep": "lanes8", "scene_sum_order": "steel"},
    {"scene_side_sweep": "thread", "scene_unit_sweep": "thread", "scene_sum_order": "steel"},
    {"scene_side_sweep": "thread", "scene_unit_sweep": "thread", "scene_sum_order": "serial"},
)

# The color loops that must reproduce the serial walk of the two-sided rows (``"serial"``) bit for
# bit: the default schedule (big colors in the while loop, the rest and the tail in one block), the
# while loop for every color with the tail on one thread, the block for every color, and the
# default schedule with one thread per unit.
COLOR_LOOP_CONFIGS = (
    {"scene_color_loop": "while_block"},
    {"scene_color_loop": "while"},
    {"scene_color_loop": "block"},
    {"scene_color_loop": "while_block", "scene_unit_sweep": "thread"},
)

CROWD_HUMANOIDS = 8
"""Lying humanoids of the crowd fixture, 0.4 m apart in one world."""

CROWD_PROP_X = -0.3
"""The crowd's prop box center on x [m]: its +x face at -0.15 m, 7.4 cm clear of the first humanoid."""

CROWD_BLOCK_COLOR_MAX = 3
"""A while/block threshold that splits the crowd's colors (3 or 4 units each) between the two."""

PHASE_ORDERS = ("BA", "AB", "ABA")
"""The phase orders of each iteration of the coupled sides (``scene_phase_order``), the default first."""

DRIFT_CONTROL_STEPS = 200
"""Control steps of the drift test, of ``DRIFT_SUBSTEPS`` steps of ``DT`` each: 5 s, as mavrix's lean."""

DRIFT_SUBSTEPS = 5
"""Steps of ``DT`` per control step: mavrix's 25 ms control step."""

DRIFT_MAX = 2.0e-3
"""Physics bound on the loaded drum's horizontal displacement [m]: 5x Steel's 0.40 mm at 4 iterations."""

DRIFT_STEEL_RATIO = 1.5
"""Physics bound on the default order's drum displacement over Steel's."""

needs_read_back = unittest.skipUnless(hasattr(SolverSteel, "scene_statistics"), "needs the read-back API")

SWEEP_CAPACITY = 10
"""Rows of the unit kernels' W = 1 arrays for ``_problem`` (8 rows), as Steel's kernel test."""

MIXED_CAPACITY = 16
"""Rows of the W = 1 arrays for ``_mixed_problem`` (14 rows)."""


class _Run:
    """One solver stepping its own states with its own deterministic collision pipeline."""

    def __init__(self, model, solver):
        self.model = model
        self.solver = solver
        self.pipeline = newton.CollisionPipeline(model, deterministic=True)
        self.contacts = self.pipeline.contacts()
        self.state_0 = model.state()
        self.state_1 = model.state()
        self.control = model.control()
        newton.eval_fk(model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def step(self):
        self.pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, DT)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def field(self, name):
        return getattr(self.state_0, name).numpy()

    def impulses(self):
        rows = int(self.solver._constraint_count.numpy()[0])
        return self.solver._impulses.numpy()[0, :rows]


def _steel(model, **options):
    """Steel's own per-world solve (``solve_mode="serial"``), deterministic, with the common options and 128 rows per
    world."""
    options = {**COMMON, "articulation_constraint_max": STEEL_ROWS, "solve_mode": "serial", **options}
    return SolverSteel(model, deterministic=wp.DeterministicMode.RUN_TO_RUN, **_steel_on(model, options))


def _scene(model, rows, overrides=None, **options):
    """Scene mode with ``rows`` articulation rows per world and the given kernel overrides."""
    with kernel_overrides(overrides or {}):
        return SolverSteel(model, solve_mode="scene", articulation_constraint_max=rows, **{**COMMON, **options})


def _scene_rows(model, humanoids=0):
    """Rows per world: 128 per humanoid, or three per contact of the buffer plus 128 per articulation."""
    if humanoids:
        return STEEL_ROWS * humanoids
    return 3 * CONTACT_MAX + STEEL_ROWS * model.articulation_count


def _scenes(device):
    """The scenes of the whole-step tests, each with its scene-mode rows and Steel's own options."""
    colored = {
        "solve_mode": "colored",
        "articulated_contact_response": "propagation",
        "free_body_constraint_max": CONTACT_MAX,
    }
    humanoids = _humanoids(device, per_world=4, spacing=0.4)
    scenes = {
        "heap": (_heap(device), None, colored),
        "tray": (_heap(device, kinematic_tray=True), None, colored),
        # 0.4 m apart the lying humanoids overlap by 4 cm, so units join pairs of them.
        "humanoids": (humanoids, _scene_rows(humanoids, 4), {"articulation_constraint_max": 4 * STEEL_ROWS}),
        "contact_scene": (_build_contact_scene(device), None, {}),
        "scissor": (_build_scissor_model(device), None, {}),
        "chain": (_chain_on_ground(device), None, {}),
    }
    return {
        name: (model, rows if rows is not None else _scene_rows(model), steel)
        for name, (model, rows, steel) in scenes.items()
    }


def _assert_equal_states(test, expected, actual, step, names=("joint_q", "joint_qd")):
    """Bitwise equality of state fields, naming the step and the largest difference."""
    for name in names:
        a, b = expected.field(name), actual.field(name)
        test.assertTrue(
            np.array_equal(a, b),
            msg=(
                f"{name} differs after step {step}: max |diff| {np.max(np.abs(a - b)):.3e} "
                f"at {np.argmax(np.abs(a - b))}"
            ),
        )


# --- the unit kernels ---------------------------------------------------------------------------


def _w1(value, dtype=np.float32, capacity=SWEEP_CAPACITY):
    """A ``[1, capacity, ...]`` array holding ``value`` in its first rows, zero after."""
    value = np.asarray(value)
    result = np.zeros((1, capacity, *value.shape[1:]), dtype=dtype)
    result[0, : len(value)] = value
    return result


def _launch_scene_serial(
    device,
    *,
    sums,
    support,
    free_response,
    rows,
    row_dof,
    row_factor,
    row_free_response,
    row_free_prefix,
    bias,
    incident,
    diag,
    types,
    parents,
    mu,
    impulses,
    coordinates,
    omega,
    sweeps=8,
    capacity=SWEEP_CAPACITY,
):
    """Sweep rows ``[0, rows)`` of one W = 1 problem with the scene's serial kernel, stationary exit on.

    The per-row vectors (``bias`` to ``impulses``) are padded to ``capacity`` rows; the row arrays
    come shaped ``[1, capacity, ...]``.
    """
    ranges = np.zeros((4, 2), dtype=np.int32)
    ranges[RANGE_ALL] = (0, rows)
    delta = wp.zeros((1, coordinates), device=device)
    actual_impulses = wp.array(_w1(impulses, capacity=capacity), device=device)
    wp.launch(
        get_sweep_rows_serial_kernel(sums, support, free_response=free_response),
        dim=1,
        inputs=[
            RANGE_ALL,
            sweeps,
            1,
            wp.array(ranges, dtype=wp.int32, device=device),
            wp.array(row_dof, dtype=wp.int32, device=device),
            wp.array(row_factor, dtype=wp.float32, device=device),
            wp.array(row_free_response, dtype=wp.float32, device=device),
            wp.array(row_free_prefix, dtype=wp.int32, device=device),
            wp.array(_w1(bias, capacity=capacity), device=device),
            wp.array(_w1(incident, capacity=capacity), device=device),
            wp.array(_w1(diag, capacity=capacity), device=device),
            wp.array(_w1(types, np.int32, capacity=capacity), device=device),
            wp.array(_w1(parents, np.int32, capacity=capacity), device=device),
            wp.array(_w1(mu, capacity=capacity), device=device),
            omega,
        ],
        outputs=[delta, actual_impulses],
        device=device,
    )
    return delta.numpy()[0], actual_impulses.numpy()[0, :rows]


def _problem_rows(indices, rows, support):
    """``_problem``'s rows on their support, padded with -1 / 0 to ``SWEEP_CAPACITY`` rows (row 0 has one entry)."""
    row_dof = np.full((1, SWEEP_CAPACITY, support), -1, dtype=np.int32)
    row_dof[0, :8] = indices
    row_dof[0, 0, 1:] = -1
    row_factor = np.zeros((1, SWEEP_CAPACITY, support), dtype=np.float32)
    row_factor[0, :8] = rows[:, indices]
    return row_dof, row_factor


def test_unit_sweep_matches_host_reference(test, device):
    """Sweep factor and free-body rows like the host reference, with serial and Steel-order sums."""
    for (dofs, support), seeded, omega in (
        (case, seeded, omega) for case in ((43, 18), (75, 39)) for seeded in (False, True) for omega in _SWEEP_CASES
    ):
        _, indices, rows, initial, bias, types, parents, mu, impulses, _, _, incident, diag = _problem(
            dofs, support, seeded
        )
        expected, expected_impulses = _sweep(
            rows, rows, np.zeros_like(initial), incident + bias, diag, types, parents, mu, impulses, omega=omega
        )
        row_dof, row_factor = _problem_rows(indices, rows, support)
        # Steel's 32-lane order exists only for supports of at most 32 entries.
        for sums in ("serial", "steel") if support <= 32 else ("serial",):
            with test.subTest(dofs=dofs, support=support, seeded=seeded, omega=omega, sums=sums):
                delta, actual_impulses = _launch_scene_serial(
                    device,
                    sums=sums,
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
                    coordinates=dofs,
                    omega=omega,
                )
                np.testing.assert_allclose(delta, expected, atol=2.0e-5, rtol=2.0e-4)
                np.testing.assert_allclose(actual_impulses, expected_impulses, atol=2.0e-5, rtol=2.0e-4)

    # Robot, robot-free and free-free rows: the free entries come first and respond through their
    # own physical response, the factor entries through the factor.
    _factor, z, r, jacobian, _response, initial, bias, types, parents, mu, diagonal = _mixed_problem()
    count, support, coordinates = 14, 21, 21
    row_dof = np.full((1, MIXED_CAPACITY, support), -1, dtype=np.int32)
    row_factor = np.zeros((1, MIXED_CAPACITY, support), dtype=np.float32)
    row_free_response = np.zeros((1, MIXED_CAPACITY, 12), dtype=np.float32)
    row_free_prefix = np.zeros(MIXED_CAPACITY, dtype=np.int32)
    for row in range(count):
        if 1 <= row < 4:
            indices = np.flatnonzero(np.any(z[1:4] != 0, axis=0))
        elif 5 <= row < 8:
            indices = np.flatnonzero(np.any(z[5:8] != 0, axis=0))
        else:
            indices = np.flatnonzero(z[row] != 0)
        indices = np.concatenate((indices[indices >= 9], indices[indices < 9]))
        free_indices = indices[indices >= 9]
        row_dof[0, row, : len(indices)] = indices
        row_factor[0, row, : len(indices)] = z[row, indices]
        row_free_response[0, row, : len(free_indices)] = r[row, free_indices]
        row_free_prefix[row] = len(free_indices)
    np.testing.assert_array_equal(row_free_prefix[:count], [0, 6, 6, 6, 0, 6, 6, 6, 12, 12, 12, 6, 6, 6])
    # The kernel's response: the factor on robot coordinates, the supplied response on free ones.
    response = np.where(np.arange(coordinates) >= 9, r, z)
    incident = jacobian @ initial
    for omega in _SWEEP_CASES:
        expected, expected_impulses = _sweep(
            z,
            response,
            np.zeros(coordinates),
            incident + bias,
            diagonal,
            types,
            parents,
            mu,
            np.zeros(count),
            omega=omega,
        )
        for sums in ("serial", "steel"):
            with test.subTest(problem="mixed", omega=omega, sums=sums):
                delta, actual_impulses = _launch_scene_serial(
                    device,
                    sums=sums,
                    support=support,
                    free_response=True,
                    rows=count,
                    row_dof=row_dof,
                    row_factor=row_factor,
                    row_free_response=row_free_response,
                    row_free_prefix=row_free_prefix,
                    bias=bias,
                    incident=incident,
                    diag=diagonal,
                    types=types,
                    parents=parents,
                    mu=mu,
                    impulses=np.zeros(count),
                    coordinates=coordinates,
                    omega=omega,
                    capacity=MIXED_CAPACITY,
                )
                np.testing.assert_allclose(delta, expected, atol=2.0e-5, rtol=2.0e-4)
                np.testing.assert_allclose(actual_impulses, expected_impulses, atol=2.0e-5, rtol=2.0e-4)


def _steel_kernel_sweep(
    device, *, row_dof, row_factor, bias, incident, diag, types, parents, mu, impulses, dofs, sweeps, omega
):
    """Steel's own sparse sweep (``_get_pgs_solve_sparse_kernel``) over one world of ``_problem``'s rows."""
    support = row_dof.shape[2]
    steel_impulses = wp.array(_w1(impulses), device=device)
    steel_delta = wp.full((1, dofs), float("nan"), device=device)
    wp.launch_tiled(
        _get_pgs_solve_sparse_kernel(SWEEP_CAPACITY, dofs, support),
        dim=[1],
        inputs=[
            1,
            wp.array([8], dtype=int, device=device),
            wp.array(_w1(bias), device=device),
            wp.array(_w1(diag), device=device),
            steel_impulses,
            wp.array(row_dof, device=device),
            wp.array(row_factor, device=device),
            wp.zeros((1, SWEEP_CAPACITY, 12), device=device),
            wp.array(_w1(incident), device=device),
            wp.array(_w1(types, np.int32), device=device),
            wp.array(_w1(parents, np.int32), device=device),
            wp.array(_w1(mu), device=device),
            wp.zeros((1, dofs), dtype=int, device=device),
            wp.array(np.arange(dofs).reshape(1, dofs), dtype=int, device=device),
            wp.zeros(dofs, device=device),
            wp.zeros(1, dtype=int, device=device),
            wp.zeros((1, 4), dtype=int, device=device),
            wp.zeros((1, 1), device=device),
            wp.zeros((1, 1, 6), device=device),
            wp.zeros((1, 1, 6), device=device),
            wp.zeros((1, 1, 6), device=device),
            wp.zeros((1, 1, 6), device=device),
            wp.zeros((1, 1), device=device),
            sweeps,
            omega,
        ],
        outputs=[steel_delta],
        block_dim=64,
        device=device,
    )
    return steel_delta.numpy()[0], steel_impulses.numpy()[0, :8]


def test_zero_diagonal_rows_match_steel_kernel(test, device):
    """Skip a normal row with a zero diagonal and still solve its tangent pair, exactly as Steel's kernel.

    With the pair's diagonals zeroed too, Steel's 2x2 friction solve has a zero operator and
    returns NaN; the scene sweep reproduces that bit for bit (NaN where Steel has NaN), which shows
    the pair is solved rather than skipped.
    """
    dofs, support, sweeps = 43, 18, 8
    _, indices, rows, _initial, bias, types, parents, mu, impulses, _, _, incident, diag = _problem(
        dofs, support, seeded=True
    )
    row_dof, row_factor = _problem_rows(indices, rows, support)
    for zeroed in ((1,), (1, 2, 3)):
        problem_diag = diag.copy()
        problem_diag[list(zeroed)] = 0.0
        for omega in _SWEEP_CASES:
            with test.subTest(zeroed=zeroed, omega=omega):
                arrays = {
                    "row_dof": row_dof,
                    "row_factor": row_factor,
                    "bias": bias,
                    "incident": incident,
                    "diag": problem_diag,
                    "types": types,
                    "parents": parents,
                    "mu": mu,
                    "impulses": impulses,
                }
                expected_delta, expected_impulses = _steel_kernel_sweep(
                    device, **arrays, dofs=dofs, sweeps=sweeps, omega=omega
                )
                delta, actual_impulses = _launch_scene_serial(
                    device,
                    sums="steel",
                    support=support,
                    free_response=False,
                    rows=8,
                    row_free_response=np.zeros((1, 1, 1), dtype=np.float32),
                    row_free_prefix=np.zeros(SWEEP_CAPACITY, dtype=np.int32),
                    coordinates=dofs,
                    omega=omega,
                    sweeps=sweeps,
                    **arrays,
                )
                test.assertTrue(np.array_equal(expected_delta, delta, equal_nan=True), msg="delta")
                test.assertTrue(np.array_equal(expected_impulses, actual_impulses, equal_nan=True), msg="impulses")
                # The normal row keeps its seeded impulse; the pair is solved on that load.
                test.assertEqual(actual_impulses[1], np.float32(impulses[1]))
                test.assertFalse(np.array_equal(actual_impulses[2:4], np.float32(impulses[2:4])))
                if zeroed == (1,):
                    test.assertTrue(np.all(np.isfinite(delta)))
                    test.assertTrue(np.all(np.isfinite(actual_impulses)))
                else:
                    test.assertTrue(np.all(np.isnan(actual_impulses[2:4])))


# --- whole steps against Steel ------------------------------------------------------------------


def test_one_humanoid_matches_steel_bitwise(test, device):
    """Step one lying humanoid exactly as Steel's deterministic per-world solve, for 100 steps."""
    model = _humanoids(device)
    steel = _Run(model, _steel(model))
    scene = _Run(model, _scene(model, STEEL_ROWS, STEEL_SUMS))
    rows = set()
    for step in range(1, 101):
        steel.step()
        scene.step()
        _assert_equal_states(test, steel, scene, step)
        rows.add(int(scene.solver._constraint_count.numpy()[0]))
    test.assertFalse(np.any(steel.solver.constraint_overflow.numpy()))
    test.assertFalse(np.any(scene.solver.constraint_overflow.numpy()))
    test.assertGreater(max(rows), 0)


def test_one_humanoid_matches_steel_with_serial_sums(test, device):
    """Agree with Steel to rounding when the sums run in entry order instead of Steel's lane tree."""
    model = _humanoids(device)
    steel = _Run(model, _steel(model))
    scene = _Run(model, _scene(model, STEEL_ROWS, SERIAL_SUMS))
    steel.step()
    scene.step()
    np.testing.assert_allclose(scene.field("joint_qd"), steel.field("joint_qd"), rtol=1.0e-5, atol=1.0e-6)
    for _ in range(19):
        steel.step()
        scene.step()
    hinges = scene.field("joint_q")[7:] - steel.field("joint_q")[7:]
    test.assertTrue(np.all(np.isfinite(hinges)))
    test.assertLess(float(np.sqrt(np.mean(hinges**2))), 1.0e-3)


def _sprung_humanoid(device):
    """One lying humanoid as ``_humanoids(device)`` builds it, with the MJCF's passive joint stiffness kept.

    ``_humanoids`` zeroes :attr:`~newton.Model.joint_stiffness`; this copies it back from the humanoid's own model, so
    stage 1's inverse dynamics apply the springs' torques ``stiffness * (rest_q - q)``.
    """
    model = _humanoids(device)
    model.joint_stiffness.assign(_humanoid().finalize(device=device).joint_stiffness)
    return model


def test_passive_springs_match_steel_under_scene(test, device):
    """Step one lying humanoid with its passive joint springs exactly as Steel's per-world solve, for 30 steps.

    The springs (:attr:`~newton.Model.joint_stiffness`, nonzero on the humanoid's hinges) act in the stage-1 inverse
    dynamics the scene shares with Steel. The humanoid touches only the ground, so no two moving bodies meet and the
    scene's rows are Steel's in Steel's order: ``joint_q`` and ``joint_qd`` agree bit for bit after every step. The
    springs act: after the 30 steps the humanoid's coordinates differ from the same scene's without them.
    """
    model = _sprung_humanoid(device)
    test.assertGreater(int(np.count_nonzero(model.joint_stiffness.numpy())), 0)
    steel = _Run(model, _steel(model))
    scene = _Run(model, _scene(model, STEEL_ROWS, STEEL_SUMS))
    limp_model = _humanoids(device)
    limp = _Run(limp_model, _scene(limp_model, STEEL_ROWS, STEEL_SUMS))
    rows = set()
    for step in range(1, 31):
        steel.step()
        scene.step()
        limp.step()
        _assert_equal_states(test, steel, scene, step)
        rows.add(int(scene.solver._constraint_count.numpy()[0]))
    test.assertFalse(np.any(steel.solver.constraint_overflow.numpy()))
    test.assertFalse(np.any(scene.solver.constraint_overflow.numpy()))
    test.assertGreater(max(rows), 0)
    test.assertFalse(np.array_equal(scene.field("joint_q"), limp.field("joint_q")), "the springs did not act")


def _assert_worlds_fold_bitwise(test, device, worlds, steps, side_sweeps):
    model = _humanoids(device, worlds=worlds)
    for side_sweep in side_sweeps:
        with test.subTest(worlds=worlds, scene_side_sweep=side_sweep):
            steel = _Run(model, _steel(model))
            scene = _Run(model, _scene(model, STEEL_ROWS, {**STEEL_SUMS, "scene_side_sweep": side_sweep}))
            for step in range(1, steps + 1):
                steel.step()
                scene.step()
                _assert_equal_states(test, steel, scene, step, names=("joint_q",))
            test.assertFalse(np.any(scene.solver.constraint_overflow.numpy()))
            test.assertEqual(_statistics(scene.solver)["two_sided_units"], 0)


def test_worlds_fold_matches_steel_bitwise(test, device):
    """Solve N worlds of one humanoid folded into one solve world exactly as Steel solves the N worlds."""
    for worlds in (1, 64):
        _assert_worlds_fold_bitwise(test, device, worlds, 20, ("lanes8", "thread"))


def test_worlds_fold_matches_steel_bitwise_4096(test, device):
    """The fold at the gate's smaller size: 4,096 worlds of one humanoid, bitwise for 10 steps."""
    _assert_worlds_fold_bitwise(test, device, 4096, 10, ("lanes8",))


def _finite(*arrays):
    return all(np.all(np.isfinite(array)) for array in arrays)


def _statistics(solver):
    """The last step's units, colors and rows, read from the scene's units (``SceneUnits.statistics``) and rows.

    The keys of the solver's read-back these tests use: ``units``, ``two_sided_units``, ``colors``, ``per_color``,
    ``tail``, ``unit_overflow``, ``rows`` (kept) and ``dropped_rows`` (requested past the rows kept).
    """
    units = solver._scene_units.statistics()
    rows = int(solver._constraint_count.numpy()[0])
    return {
        "units": units["units"],
        "two_sided_units": units["two_sided"],
        "colors": units["colors"],
        "per_color": units["per_color"],
        "tail": units["tail"],
        "unit_overflow": units["overflow"],
        "rows": rows,
        "dropped_rows": int(solver._scene.rows_requested.numpy()[0]) - rows,
    }


def test_phases_match_serial_oracle(test, device):
    """Reproduce one thread walking the rows in the phase order's sequence, bit for bit, with the phased sweep.

    ``"AB"``'s oracle walks every row in row order; ``"BA"``'s (the default) walks the isolated
    sides, then the two-sided rows, then the coupled sides (``test_phase_orders_match_their_oracles``
    covers all three orders on the crowd and the contact scene). Each scene runs 30 steps, or until
    the oracle's state turns non-finite: the scissor diverges within a few steps under Steel's own
    sweep too, so the phases must turn non-finite at the same step and the comparison stops there.
    """
    for name, (model, rows, _steel_options) in _scenes(device).items():
        for order in ("AB", "BA"):
            oracles = {}
            two_sided = 0
            for sums in ("steel", "serial"):
                oracle = _Run(
                    model,
                    _scene(model, rows, {"scene_sweep": "serial", "scene_sum_order": sums}, scene_phase_order=order),
                )
                history = []
                for _ in range(30):
                    oracle.step()
                    history.append((oracle.field("joint_q"), oracle.field("joint_qd"), oracle.impulses()))
                    two_sided = max(two_sided, _statistics(oracle.solver)["two_sided_units"])
                    if not _finite(*history[-1]):
                        break
                oracles[sums] = history
            for config in PHASE_CONFIGS:
                with test.subTest(scene=name, scene_phase_order=order, **config):
                    history = oracles[config["scene_sum_order"]]
                    phases = _Run(model, _scene(model, rows, config, scene_phase_order=order))
                    for step, (joint_q, joint_qd, impulses) in enumerate(history, start=1):
                        phases.step()
                        actual = (phases.field("joint_q"), phases.field("joint_qd"), phases.impulses())
                        if not _finite(joint_q, joint_qd, impulses):
                            test.assertFalse(_finite(*actual), msg=f"the oracle turned non-finite at step {step}")
                            test.assertGreater(step, 2, msg="too few finite steps to compare")
                            break
                        for field, expected, value in zip(
                            ("joint_q", "joint_qd", "impulses"), (joint_q, joint_qd, impulses), actual, strict=True
                        ):
                            test.assertTrue(np.array_equal(expected, value), msg=f"{field} at step {step}")
            if name in ("heap", "tray", "humanoids"):
                # These scenes exercise the coupled sides and the walk of the two-sided rows.
                test.assertGreater(two_sided, 0, msg=f"{name} {order}")


def test_scene_solve_rests_like_steel(test, device):
    """Come to rest where Steel's own solve does: no NaN, free bodies within 1 mm, the tree within 1e-3 rad.

    The overlapping humanoids push apart and one rolls over for longer than 150 steps, so they run
    300 (their coupled rows run in a different order than Steel's, so their paths differ while they
    move). The scissor diverges within a few steps under Steel's own sweep with any options, so it
    must match Steel bit for bit for as long as Steel stays finite, and turn non-finite with it.
    """
    for name, (model, rows, steel_options) in _scenes(device).items():
        with test.subTest(scene=name):
            steel = _Run(model, _steel(model, **steel_options))
            scene = _Run(model, _scene(model, rows))
            for step in range(1, (300 if name == "humanoids" else 150) + 1):
                steel.step()
                scene.step()
                if name == "scissor":
                    if not _finite(steel.field("joint_q")):
                        test.assertFalse(_finite(scene.field("joint_q")), msg=f"Steel turned non-finite at step {step}")
                        test.assertGreater(step, 2)
                        break
                    _assert_equal_states(test, steel, scene, step)
            if name == "scissor":
                continue
            for run in (steel, scene):
                test.assertTrue(_finite(run.field("joint_q"), run.field("body_q")))
            free = model.joint_child.numpy()[model.joint_type.numpy() == int(newton.JointType.FREE)]
            heights = scene.field("body_q")[free, 2] - steel.field("body_q")[free, 2]
            if free.size:
                test.assertLess(
                    float(np.max(np.abs(heights))), 1.0e-3, msg=f"worst body {free[np.argmax(np.abs(heights))]}"
                )
    with test.subTest(scene="tree"):
        model = _tree(device, 7, branched=True)
        steel = _Run(model, _steel(model))
        scene = _Run(model, _scene(model, _scene_rows(model)))
        for _ in range(150):
            steel.step()
            scene.step()
        difference = scene.field("joint_q") - steel.field("joint_q")
        test.assertTrue(np.all(np.isfinite(difference)))
        test.assertLess(float(np.sqrt(np.mean(difference**2))), 1.0e-3)


def test_captured_steps_match_eager_with_changing_limits(test, device):
    """Replay one captured step exactly as eager steps while the active limit rows change."""
    model = _humanoids(device, worlds=2, lying=False)
    eager = _Run(model, _scene(model, STEEL_ROWS))
    captured = _Run(model, _scene(model, STEEL_ROWS))

    def step(run):
        run.pipeline.collide(run.state_0, run.contacts)
        run.solver.step(run.state_0, run.state_1, run.control, run.contacts, DT)
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            wp.copy(getattr(run.state_0, name), getattr(run.state_1, name))

    row_counts = set()
    step(eager)
    step(captured)
    _assert_equal_states(test, eager, captured, 1, names=("joint_q",))
    row_counts.add(_statistics(eager.solver)["rows"])
    with wp.ScopedCapture(device) as recorded:
        step(captured)
    for index in range(2, 61):
        step(eager)
        wp.capture_launch(recorded.graph)
        _assert_equal_states(test, eager, captured, index, names=("joint_q",))
        row_counts.add(_statistics(eager.solver)["rows"])
    test.assertGreaterEqual(len(row_counts), 3, msg=f"row counts {sorted(row_counts)}")
    test.assertTrue(np.all(np.isfinite(captured.field("joint_q"))))


# --- phase B: the colors and the tail -----------------------------------------------------------


def _crowd(device):
    """Eight lying humanoids 0.4 m apart (neighbors overlap by 4 cm) and a free prop box clear of them.

    ``_humanoids`` puts the prop 2.6 cm inside the first humanoid; its free joint is moved to
    ``x = CROWD_PROP_X`` here, before any state is made from the model.
    """
    model = _humanoids(device, per_world=CROWD_HUMANOIDS, spacing=0.4, props=True)
    prop = model.body_count - 1
    joint = int(np.flatnonzero(model.joint_child.numpy() == prop)[0])
    assert model.joint_type.numpy()[joint] == int(newton.JointType.FREE)
    joint_q = model.joint_q.numpy()
    joint_q[model.joint_q_start.numpy()[joint]] = CROWD_PROP_X
    model.joint_q.assign(joint_q)
    body_q = model.body_q.numpy()
    body_q[prop, 0] = CROWD_PROP_X
    model.body_q.assign(body_q)
    return model


def _crowd_rows():
    """The crowd's scene rows: 128 per humanoid and 64 for the prop, as mavrix sizes them."""
    return STEEL_ROWS * CROWD_HUMANOIDS + 64


def _phase_b_scenes(device):
    """The scenes of the color tests, each with its scene rows.

    The heap's colors hold at most about 53 units, so the default schedule runs them all in the
    block; the wide heap's first colors hold more than ``SCENE_BLOCK_COLOR_MAX``, so it splits
    them between the while loop and the block.
    """
    heap = _heap(device)
    wide_heap = _heap(device, nx=12, ny=12, nz=2)
    contact_scene = _build_contact_scene(device)
    return {
        "heap": (heap, _scene_rows(heap)),
        "wide_heap": (wide_heap, _scene_rows(wide_heap)),
        "crowd": (_crowd(device), _crowd_rows()),
        "contact_scene": (contact_scene, _scene_rows(contact_scene)),
    }


def _assert_color_loop(test, solver, loop):
    """The solver runs the requested color loop (the override took effect)."""
    test.assertEqual(solver._scene.color_loop, loop)


def _record(run, steps):
    """Step ``run`` and keep its ``joint_q``, ``joint_qd`` and impulses after every step."""
    history = []
    for _ in range(steps):
        run.step()
        history.append((run.field("joint_q"), run.field("joint_qd"), run.impulses()))
    return history


def _color_split(solver):
    """The last step's colors through the while loop (``big_colors``) and colors in use (``used_colors``)."""
    scene = solver._scene
    return int(scene.big_colors.numpy()[0]), int(scene.used_colors.numpy()[0])


def _assert_replays(test, history, run):
    """Step ``run`` and require ``history`` bit for bit after every step.

    Returns the steps whose colors were split between the while loop and the block: some ran
    through the while loop (``big_colors > 0``) and some remained for the block after them.
    """
    split_steps = []
    for step, expected in enumerate(history, start=1):
        run.step()
        actual = (run.field("joint_q"), run.field("joint_qd"), run.impulses())
        for field, a, b in zip(("joint_q", "joint_qd", "impulses"), expected, actual, strict=True):
            test.assertTrue(np.array_equal(a, b), msg=f"{field} at step {step}")
        big, used = _color_split(run.solver)
        if 0 < big < used:
            split_steps.append(step)
    return split_steps


def test_color_sweep_matches_serial_walk(test, device):
    """Sweep the two-sided units color by color exactly as one thread walking their rows in row order.

    Units of one color share no side, so their order within the color cannot matter; every color
    loop (the while loop, the single block, the default schedule of both) and both unit
    granularities must therefore reproduce the serial walk bit for bit over 30 steps.
    """
    for name, (model, rows) in _phase_b_scenes(device).items():
        oracle = _Run(model, _scene(model, rows, {"scene_color_loop": "serial"}))
        _assert_color_loop(test, oracle.solver, "serial")
        history = []
        two_sided = colors = biggest = 0
        for _ in range(30):
            oracle.step()
            history.append((oracle.field("joint_q"), oracle.field("joint_qd"), oracle.impulses()))
            # A tail unit may appear; every loop walks the tail last, as the oracle does.
            statistics = _statistics(oracle.solver)
            two_sided = max(two_sided, statistics["two_sided_units"])
            colors = max(colors, statistics["colors"])
            biggest = max(biggest, *statistics["per_color"])
        if name != "contact_scene":
            test.assertGreater(two_sided, 0, msg=name)
            test.assertGreater(colors, 1, msg=name)
        if name == "wide_heap":
            # Some colors run through the while loop of the default schedule, the rest in the block.
            test.assertGreater(biggest, SCENE_BLOCK_COLOR_MAX)
        configs = COLOR_LOOP_CONFIGS
        if name == "crowd":
            # The crowd's colors hold at most 4 units, so force the default schedule's split.
            configs += ({"scene_color_loop": "while_block", "scene_block_color_max": CROWD_BLOCK_COLOR_MAX},)
        for config in configs:
            with test.subTest(scene=name, **config):
                run = _Run(model, _scene(model, rows, config))
                _assert_color_loop(test, run.solver, config["scene_color_loop"])
                split_steps = _assert_replays(test, history, run)
                if config["scene_color_loop"] == "while_block" and (
                    name == "wide_heap" or "scene_block_color_max" in config
                ):
                    # Both halves of the default schedule ran: the while loop, then the block.
                    test.assertTrue(split_steps, msg="no step split its colors between the while loop and the block")


def test_unroll_fallback_matches_while(test, device):
    """Launch every color below the cap with a constant color, bit for bit as the while loop (the heap)."""
    model = _heap(device)
    rows = _scene_rows(model)
    loop = _Run(model, _scene(model, rows, {"scene_color_loop": "while"}))
    history = _record(loop, 30)
    unroll = _Run(model, _scene(model, rows, {"scene_color_loop": "unroll"}))
    _assert_color_loop(test, unroll.solver, "unroll")
    _assert_replays(test, history, unroll)


def test_tail_sweep_matches_serial_walk(test, device):
    """Sweep a large tail bit for bit as the serial walk: the heap at ``scene_color_cap=2`` for 30 steps.

    Most two-sided units take the tail at every step, so the block's thread 0 (``"while_block"``,
    ``"block"``) and the one-thread walk of ``RANGE_TAIL`` (``"while"``) each sweep hundreds of tail
    units after the two colors.
    """
    model = _heap(device)
    rows = _scene_rows(model)
    oracle = _Run(model, _scene(model, rows, {"scene_color_loop": "serial", "scene_color_cap": 2}))
    history = []
    for step in range(1, 31):
        oracle.step()
        history.append((oracle.field("joint_q"), oracle.field("joint_qd"), oracle.impulses()))
        statistics = _statistics(oracle.solver)
        test.assertGreater(statistics["tail"], statistics["two_sided_units"] // 2, msg=f"step {step}")
    for loop in ("while_block", "while", "block"):
        with test.subTest(scene_color_loop=loop):
            run = _Run(model, _scene(model, rows, {"scene_color_loop": loop, "scene_color_cap": 2}))
            _assert_color_loop(test, run.solver, loop)
            _assert_replays(test, history, run)


def _settle_with_forces(model, solver, steps, pipeline=None):
    """Step ``solver`` ``steps`` times with the contact-force observable; return the run and each body's net contact
    force."""
    run = _Run(model, solver)
    if pipeline is not None:
        run.pipeline = pipeline
        run.contacts = pipeline.contacts()
    observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=run.contacts)
    for _ in range(steps):
        run.pipeline.collide(run.state_0, run.contacts)
        solver.step(run.state_0, run.state_1, run.control, run.contacts, DT, observables=observables)
        run.state_0, run.state_1 = run.state_1, run.state_0
    return run, _net_body_forces(model, run.contacts, observables.contact_f.numpy())


def _weights(model):
    return np.linalg.norm(model.gravity.numpy()[0]) * model.body_mass.numpy()


def _assert_weights(test, model, net, bodies, label):
    """Each body's net contact force is its weight, straight up, within 3 %."""
    weights = _weights(model)
    for body in bodies:
        expected = np.array([0.0, 0.0, weights[body]])
        np.testing.assert_allclose(net[body], expected, atol=0.03 * weights[body], err_msg=f"{label}: body {body}")


def _free_bodies(model):
    return model.joint_child.numpy()[model.joint_type.numpy() == int(newton.JointType.FREE)]


@needs_read_back
def test_cap_two_matches_cap_sixty_four_at_rest(test, device):
    """Rest where cap 64 rests with two-sided units in the tail (cap 2), and warn once that the cap is too small.

    The heap at cap 2 puts about 1,800 of its 1,900 two-sided units in the tail. Its boxes come to
    rest within 1 mm of cap 64's and carry the heap's weight within 3 %. Per box, 4 iterations
    leave Steel's own solve of the heap up to 14 % off a box's weight (48 of 108 boxes past 3 % at
    150 steps), so each box's own weight within 3 % is checked on the stacked boxes of the
    contact-force tests at 32 iterations, whose cap 2 puts two of the four box-box units in the
    tail.
    """
    model = _heap(device)
    rows = _scene_rows(model)
    boxes = _free_bodies(model)
    weight = _weights(model)[boxes].sum()
    runs = {}
    for cap in (2, 64):
        run, net = _settle_with_forces(model, _scene(model, rows, {"scene_color_cap": cap}), 150)
        runs[cap] = run
        test.assertTrue(_finite(run.field("body_q")), msg=f"cap {cap}")
        np.testing.assert_allclose(net[boxes].sum(axis=0)[2], weight, rtol=0.03, err_msg=f"cap {cap}")
    heights = runs[2].field("body_q")[boxes, 2] - runs[64].field("body_q")[boxes, 2]
    test.assertLess(float(np.max(np.abs(heights))), 1.0e-3, msg=f"worst box {boxes[np.argmax(np.abs(heights))]}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        capped = runs[2].solver.scene_statistics()
        runs[2].solver.scene_statistics()
    # Most two-sided units take the tail at cap 2; the solver warns once, on the first read.
    test.assertGreater(capped["tail"], capped["two_sided_units"] // 2)
    tail_warnings = [w for w in caught if issubclass(w.category, NewtonWarning) and "color_cap" in str(w.message)]
    test.assertEqual(len(tail_warnings), 1, msg=[str(w.message) for w in caught])
    # The heap's busiest box can have 65 two-sided units, so cap 64 may leave one in the tail.
    test.assertLessEqual(_statistics(runs[64].solver)["tail"], 1)

    stacked, pipeline = _stacked_boxes(device)
    for cap in (2, 64):
        solver = _scene(stacked, _scene_rows(stacked), {"scene_color_cap": cap}, iterations=32)
        run, net = _settle_with_forces(stacked, solver, 150, pipeline)
        test.assertTrue(_finite(run.field("body_q")), msg=f"stacked, cap {cap}")
        _assert_weights(test, stacked, net, _free_bodies(stacked), f"stacked, cap {cap}")
        tail = _statistics(solver)["tail"]
        if cap == 2:
            test.assertGreater(tail, 0)
        else:
            test.assertEqual(tail, 0)


def test_two_solvers_are_identical(test, device):
    """Two scene solvers on the same contacts give identical velocities, impulses and colors for 50 steps.

    One inherits Warp's determinism mode (``deterministic=None``) and the other runs
    ``RUN_TO_RUN``: scene mode has no atomics on its solve path, so the mode changes nothing.
    """
    model = _heap(device)
    rows = _scene_rows(model)
    first = _Run(model, _scene(model, rows, deterministic=None))
    second = _scene(model, rows, deterministic=wp.DeterministicMode.RUN_TO_RUN)
    states = [model.state(), model.state()]
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    two_sided = 0
    for step in range(1, 51):
        first.pipeline.collide(first.state_0, first.contacts)
        first.solver.step(first.state_0, first.state_1, first.control, first.contacts, DT)
        second.step(states[0], states[1], first.control, first.contacts, DT)
        first.state_0, first.state_1 = first.state_1, first.state_0
        states.reverse()
        count = int(first.solver._constraint_count.numpy()[0])
        test.assertEqual(count, int(second._constraint_count.numpy()[0]), msg=f"rows at step {step}")
        test.assertTrue(np.array_equal(first.solver._v_out.numpy(), second._v_out.numpy()), msg=f"v_out at {step}")
        test.assertTrue(
            np.array_equal(first.solver._impulses.numpy()[:, :count], second._impulses.numpy()[:, :count]),
            msg=f"impulses at step {step}",
        )
        colors = []
        for solver in (first.solver, second):
            units = solver._scene_units
            units_count = int(units.unit_count.numpy()[0])
            colors.append(units.unit_color.numpy()[units.unit_order.numpy()[:units_count]])
        test.assertTrue(np.array_equal(*colors), msg=f"colors at step {step}")
        test.assertTrue(np.array_equal(first.field("body_q"), states[0].body_q.numpy()), msg=f"body_q at {step}")
        two_sided = max(two_sided, _statistics(first.solver)["two_sided_units"])
    test.assertGreater(two_sided, 0)


def _copy_back_step(run):
    """Collide, step and copy the result back into ``state_0``, so a captured step replays on fixed buffers."""
    run.pipeline.collide(run.state_0, run.contacts)
    run.solver.step(run.state_0, run.state_1, run.control, run.contacts, DT)
    for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
        wp.copy(getattr(run.state_0, name), getattr(run.state_1, name))


LAYER_PITCH = 0.08
"""Box pitch of the single-layer heap: 4 cm boxes 4 cm apart."""


def _stack_every_second_box(model, run):
    """Lift every box of odd x index 4.5 cm and put it over its neighbor at x index - 1, still in ``state_0``."""
    nx = ny = 6
    joint_child = model.joint_child.numpy()
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    joint_q = run.state_0.joint_q.numpy()
    joint_qd = run.state_0.joint_qd.numpy()
    for ix in range(1, nx, 2):
        for iy in range(ny):
            body = ix * ny + iy
            joint = int(np.flatnonzero(joint_child == body)[0])
            start = q_start[joint]
            joint_q[start] -= LAYER_PITCH
            joint_q[start + 2] += 0.045
            joint_qd[qd_start[joint] : qd_start[joint] + 6] = 0.0
    run.state_0.joint_q.assign(joint_q)
    run.state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, run.state_0.joint_q, run.state_0.joint_qd, run.state_0)


def test_capture_before_coupling_replays_after(test, device):
    """Capture a step with no two-sided unit, then replay it while boxes land on boxes, bit for bit as eager steps.

    The colors' conditional nodes are captured with ``any_coupled == 0`` and only evaluated at
    replay; the colors after the replay are valid and equal the eager ones.
    """
    for loop in ("while_block", "while"):
        with test.subTest(scene_color_loop=loop):
            model = _heap(device, nz=1, pitch=LAYER_PITCH)
            rows = _scene_rows(model)
            eager = _Run(model, _scene(model, rows, {"scene_color_loop": loop}))
            captured = _Run(model, _scene(model, rows, {"scene_color_loop": loop}))
            for run in (eager, captured):
                _copy_back_step(run)
                statistics = _statistics(run.solver)
                test.assertGreater(statistics["units"], 0)
                test.assertEqual(statistics["two_sided_units"], 0)
                test.assertEqual(int(run.solver._scene.any_coupled.numpy()[0]), 0)
            with wp.ScopedCapture(device) as recorded:
                _copy_back_step(captured)
            # The capture ran nothing: put both twins back on the same state, then stack the boxes.
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                wp.copy(getattr(captured.state_0, name), getattr(eager.state_0, name))
            for run in (eager, captured):
                _stack_every_second_box(model, run)
            for step in range(1, 61):
                _copy_back_step(eager)
                wp.capture_launch(recorded.graph)
                test.assertTrue(
                    np.array_equal(eager.field("body_q"), captured.field("body_q")), msg=f"body_q after replay {step}"
                )
            statistics = _statistics(captured.solver)
            test.assertGreater(statistics["two_sided_units"], 0)
            test.assertGreater(statistics["colors"], 0)
            units = captured.solver._scene_units
            view = _colored_view(units)
            _assert_valid_coloring(test, units, view)
            _assert_color_order(test, units)
            test.assertTrue(np.array_equal(view["color"], _colored_view(eager.solver._scene_units)["color"]))


def test_captured_steps_match_eager_with_colors(test, device):
    """Replay one captured step exactly as eager steps on the crowd, whose colors change from step to step."""
    for loop in ("while_block", "while"):
        with test.subTest(scene_color_loop=loop):
            model = _crowd(device)
            rows = _crowd_rows()
            eager = _Run(model, _scene(model, rows, {"scene_color_loop": loop}))
            captured = _Run(model, _scene(model, rows, {"scene_color_loop": loop}))
            _copy_back_step(eager)
            _copy_back_step(captured)
            _assert_equal_states(test, eager, captured, 1)
            with wp.ScopedCapture(device) as recorded:
                _copy_back_step(captured)
            colors = set()
            two_sided = 0
            for step in range(2, 61):
                _copy_back_step(eager)
                wp.capture_launch(recorded.graph)
                _assert_equal_states(test, eager, captured, step)
                statistics = _statistics(eager.solver)
                colors.add(statistics["colors"])
                two_sided = max(two_sided, statistics["two_sided_units"])
            test.assertGreater(two_sided, 0)
            test.assertGreaterEqual(len(colors), 2, msg=f"color counts {sorted(colors)}")
            test.assertTrue(_finite(captured.field("joint_q")))


def test_contact_scene_matches_steel(test, device):
    """Rest the contact scene as Steel does: each free box carries its weight within 3 %, heights within 1 mm."""
    model = _build_contact_scene(device)
    steel, steel_net = _settle_with_forces(model, _steel(model), 150)
    scene, scene_net = _settle_with_forces(model, _scene(model, _scene_rows(model)), 150)
    boxes = _free_bodies(model)
    test.assertGreater(boxes.size, 0)
    for label, run, net in (("steel", steel, steel_net), ("scene", scene, scene_net)):
        test.assertTrue(_finite(run.field("body_q")), msg=label)
        _assert_weights(test, model, net, boxes, label)
    heights = scene.field("body_q")[:, 2] - steel.field("body_q")[:, 2]
    test.assertLess(float(np.max(np.abs(heights))), 1.0e-3, msg=f"worst body {np.argmax(np.abs(heights))}")


# --- the phase order ----------------------------------------------------------------------------


def _phase_order_scenes(device):
    """The coupled scenes of the phase-order tests, each with its scene rows: the crowd and the contact scene."""
    contact_scene = _build_contact_scene(device)
    return {
        "crowd": (_crowd(device), _crowd_rows()),
        "contact_scene": (contact_scene, _scene_rows(contact_scene)),
    }


def test_phase_orders_match_their_oracles(test, device):
    """Reproduce, in every phase order, one thread walking the rows in that order's sequence, bit for bit.

    The oracle (``scene_sweep="serial"``) runs each sweep as the order's iteration: ``"AB"`` walks
    every row in row order (the one-sided rows, then the two-sided ones); ``"BA"`` walks the
    isolated sides' one-sided rows, the two-sided rows (the colors, then the tail) and then the
    coupled sides' one-sided rows; ``"ABA"`` walks the isolated sides, the coupled sides, the
    two-sided rows and the coupled sides again. Both scenes have two-sided units, so the orders
    reach the coupled sides; 30 steps in each order and configuration.
    """
    for name, (model, rows) in _phase_order_scenes(device).items():
        for order in PHASE_ORDERS:
            oracles = {}
            two_sided = 0
            for sums in ("steel", "serial"):
                oracle = _Run(
                    model,
                    _scene(model, rows, {"scene_sweep": "serial", "scene_sum_order": sums}, scene_phase_order=order),
                )
                test.assertEqual(oracle.solver._scene.phase_order, order)
                history = []
                for _ in range(30):
                    oracle.step()
                    history.append((oracle.field("joint_q"), oracle.field("joint_qd"), oracle.impulses()))
                    two_sided = max(two_sided, _statistics(oracle.solver)["two_sided_units"])
                test.assertTrue(_finite(*history[-1]), msg=f"{name} {order}")
                oracles[sums] = history
            test.assertGreater(two_sided, 0, msg=f"{name} {order}")
            for config in PHASE_CONFIGS:
                with test.subTest(scene=name, scene_phase_order=order, **config):
                    run = _Run(model, _scene(model, rows, config, scene_phase_order=order))
                    test.assertEqual(run.solver.scene_phase_order, order)
                    _assert_replays(test, oracles[config["scene_sum_order"]], run)


def test_phase_orders_differ_only_on_coupled_scenes(test, device):
    """Step alike in every order without two-sided units, and differently on the coupled crowd.

    Four worlds of one humanoid have no two-sided unit, so every side is isolated and runs all its
    iterations in the first launch: the three orders give the same ``joint_q`` bit for bit after
    every one of 20 steps. Steel's own per-world solve (``solve_mode="serial"``) ignores the option.
    The crowd's neighbors overlap, so its coupled sides take the orders' different sequences and
    their states part within the 20 steps.
    """
    model = _humanoids(device, worlds=4)
    runs = {order: _Run(model, _scene(model, STEEL_ROWS, scene_phase_order=order)) for order in PHASE_ORDERS}
    steel = {order: _Run(model, _steel(model, scene_phase_order=order)) for order in PHASE_ORDERS}
    default = PHASE_ORDERS[0]
    for step in range(1, 21):
        for run in (*runs.values(), *steel.values()):
            run.step()
        for order in PHASE_ORDERS[1:]:
            with test.subTest(scene_phase_order=order):
                _assert_equal_states(test, runs[default], runs[order], step, names=("joint_q",))
                _assert_equal_states(test, steel[default], steel[order], step, names=("joint_q",))
        for order, run in runs.items():
            test.assertEqual(_statistics(run.solver)["two_sided_units"], 0, msg=f"{order} at step {step}")
    for order, run in steel.items():
        test.assertEqual(run.solver.scene_phase_order, order)
    crowd = _crowd(device)
    runs = {order: _Run(crowd, _scene(crowd, _crowd_rows(), scene_phase_order=order)) for order in PHASE_ORDERS}
    two_sided = 0
    for _ in range(20):
        for run in runs.values():
            run.step()
        two_sided = max(two_sided, _statistics(runs[default].solver)["two_sided_units"])
    test.assertGreater(two_sided, 0)
    joint_q = {order: run.field("joint_q") for order, run in runs.items()}
    for index, first in enumerate(PHASE_ORDERS):
        test.assertTrue(np.all(np.isfinite(joint_q[first])), msg=first)
        for second in PHASE_ORDERS[index + 1 :]:
            test.assertFalse(np.array_equal(joint_q[first], joint_q[second]), msg=f"{first} and {second}")


def test_invalid_phase_order_raises(test, device):
    """Default to ``"BA"`` and reject a phase order other than ``"BA"``, ``"AB"`` and ``"ABA"``; the option is
    read-only.

    The scene validates the option; Steel's own per-world solve (``solve_mode="serial"``) ignores it and reads it back.
    """
    model = _humanoids(device)
    for order in ("BAB", "ab", "A", "", None):
        with test.subTest(order=order):
            with test.assertRaisesRegex(ValueError, "scene_phase_order"):
                _scene(model, STEEL_ROWS, scene_phase_order=order)
    test.assertEqual(_steel(model, scene_phase_order="BAB").scene_phase_order, "BAB")
    test.assertEqual(_scene(model, STEEL_ROWS).scene_phase_order, "BA")
    test.assertEqual(_steel(model).scene_phase_order, "BA")
    solver = _scene(model, STEEL_ROWS, scene_phase_order="ABA")
    test.assertEqual(solver.scene_phase_order, "ABA")
    with test.assertRaises(AttributeError):
        solver.scene_phase_order = "AB"


# --- drift under a sustained load ---------------------------------------------------------------


def _drum_drift(model, drum, solver):
    """Step ``DRIFT_CONTROL_STEPS`` control steps and measure the drum's horizontal displacement [m].

    After each control step the drum's xy is read, and the step counts as loaded when the
    humanoid's contact forces on the drum (``CONTACT_F`` at the step's last substep; the ground is
    body -1 and left out) sum to a nonzero force. ``loaded`` is the displacement from the start of
    the first loaded control step to the end of the last one, ``whole`` the displacement over the
    run; the rest are the preconditions the test checks.
    """
    run = _Run(model, solver)
    observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=run.contacts)
    shape_body = model.shape_body.numpy()
    xy = [run.field("body_q")[drum, :2].astype(np.float64)]
    loaded = []
    contacts = two_sided = dropped = 0
    for _ in range(DRIFT_CONTROL_STEPS):
        for _ in range(DRIFT_SUBSTEPS):
            run.pipeline.collide(run.state_0, run.contacts)
            solver.step(run.state_0, run.state_1, run.control, run.contacts, DT, observables=observables)
            run.state_0, run.state_1 = run.state_1, run.state_0
        xy.append(run.field("body_q")[drum, :2].astype(np.float64))
        count = int(run.contacts.rigid_contact_count.numpy()[0])
        contacts = max(contacts, count)
        body0 = shape_body[run.contacts.rigid_contact_shape0.numpy()[:count]]
        body1 = shape_body[run.contacts.rigid_contact_shape1.numpy()[:count]]
        force = observables.contact_f.numpy()[:count, :3]
        push = force[(body0 == drum) & (body1 >= 0)].sum(axis=0) - force[(body1 == drum) & (body0 >= 0)].sum(axis=0)
        loaded.append(bool(np.any(push != 0.0)))
        if solver.solve_mode == "scene":
            statistics = _statistics(solver)
            two_sided = max(two_sided, statistics["two_sided_units"])
            dropped = max(dropped, statistics["dropped_rows"] + statistics["unit_overflow"])
    steps = np.flatnonzero(loaded)
    first, last = (int(steps[0]), int(steps[-1]) + 1) if steps.size else (0, 0)
    return {
        "loaded": float(np.linalg.norm(xy[last] - xy[first])),
        "whole": float(np.linalg.norm(xy[-1] - xy[0])),
        "loaded_steps": int(steps.size),
        "interval": (first, last),
        "finite": _finite(run.field("joint_q"), run.field("body_q")),
        "overflow": bool(np.any(solver.constraint_overflow.numpy())),
        "contacts": contacts,
        "two_sided": two_sided,
        "dropped": dropped,
    }


def _drift_runs(device):
    """The drift test's runs on ``_leaning_humanoid``: Steel (``"steel"``) and the scene solve in each phase order."""
    model, drum = _leaning_humanoid(device)
    rows = STEEL_ROWS + 64
    runs = {
        "steel": _drum_drift(model, drum, _steel(model, articulation_constraint_max=rows, free_body_constraint_max=48))
    }
    for order in PHASE_ORDERS:
        runs[order] = _drum_drift(
            model, drum, _scene(model, rows, free_body_constraint_max=48, scene_phase_order=order)
        )
    return model, runs


def test_loaded_body_does_not_drift(test, device):
    """Hold a drum that a humanoid lies across where Steel holds it, in the default phase order.

    In ``_leaning_humanoid`` a limp humanoid falls onto a free 51.2 kg drum and lies across it to
    the end, so the drum carries a sustained sideways load through two-sided units. Steel
    (``RUN_TO_RUN``) and the scene solve in each phase order run 200 control steps of 5 steps at 4
    iterations (mavrix's preset), with deterministic contacts. The drum's horizontal displacement
    over the loaded interval (control steps 10 to 200 in every run) on an NVIDIA L40:

    - Steel 0.403 mm (0.32 mm as the humanoid lands, 0.09 mm after control step 50);
    - ``"BA"`` (the default) 0.306 mm, 0.76x Steel's;
    - ``"ABA"`` 0.116 mm, 0.29x;
    - ``"AB"`` 25.1 mm, 62x: it keeps creeping, about 0.19 mm per control step from step 100 on.

    The whole-run displacements are within 0.002 mm of these. With the humanoid 5 cm further away,
    or tilted 12 or 20 degrees, the default order stayed at 0.84x to 1.17x Steel's while ``"AB"``
    moved the drum 19 to 45 mm; 5 cm closer, the humanoid slides off the drum under Steel.

    Physics bounds (tolerances): Steel, the default order and ``"ABA"`` keep the drum within
    ``DRIFT_MAX`` (2 mm, 5x Steel's displacement), and the default order within
    ``DRIFT_STEEL_RATIO`` (1.5x) of Steel's, over the loaded interval and over the whole run.

    Documentation of the finding (``"AB"`` solves a body's ground friction before the pushes it
    receives, so a loaded body creeps): ``"AB"`` moves the drum further than
    the default order and past both bounds, which shows that the bounds see the creep. A change
    that removes ``"AB"``'s creep fails these assertions and should update them with the numbers
    above.
    """
    model, runs = _drift_runs(device)
    summary = ", ".join(
        f"{name} {run['loaded'] * 1e3:.3f} mm ({run['whole'] * 1e3:.3f} mm whole)" for name, run in runs.items()
    )
    # Preconditions: finite, no overflow, and a load that lasts to the end through two-sided units.
    for name, run in runs.items():
        with test.subTest(run=name):
            test.assertTrue(run["finite"])
            test.assertFalse(run["overflow"])
            test.assertLess(run["contacts"], model.rigid_contact_max)
            test.assertEqual(run["interval"][1], DRIFT_CONTROL_STEPS, msg="the load must last to the end")
            test.assertGreaterEqual(run["loaded_steps"], 0.75 * DRIFT_CONTROL_STEPS)
            if name != "steel":
                test.assertGreater(run["two_sided"], 0)
                test.assertEqual(run["dropped"], 0)
    default, steel, creeping = runs[PHASE_ORDERS[0]], runs["steel"], runs["AB"]
    # Physics bounds (tolerances).
    for measure in ("loaded", "whole"):
        for name in ("steel", PHASE_ORDERS[0], "ABA"):
            test.assertLess(runs[name][measure], DRIFT_MAX, msg=f"{name} {measure}: {summary}")
        test.assertLessEqual(
            default[measure], DRIFT_STEEL_RATIO * steel[measure], msg=f"default order {measure}: {summary}"
        )
    # Documentation of the finding: "AB" creeps, and the bounds above would catch it.
    test.assertGreater(creeping["loaded"], default["loaded"], msg=summary)
    test.assertGreater(creeping["loaded"], DRIFT_MAX, msg=summary)
    test.assertGreater(creeping["loaded"], DRIFT_STEEL_RATIO * steel["loaded"], msg=summary)


class TestSteelSceneSweep(unittest.TestCase):
    pass


# On every device; on the CPU a Steel reference runs Steel's host sweep over its dense rows.
for _fn in (
    test_unit_sweep_matches_host_reference,
    test_one_humanoid_matches_steel_with_serial_sums,
    test_cap_two_matches_cap_sixty_four_at_rest,
    test_two_solvers_are_identical,
    test_contact_scene_matches_steel,
    test_phase_orders_differ_only_on_coupled_scenes,
    test_invalid_phase_order_raises,
    test_loaded_body_does_not_drift,
):
    add_function_test(TestSteelSceneSweep, _fn.__name__, _fn, devices=get_test_devices())

# CUDA only, each for the reason beside it. The phases' lane groups, colors and block kernel are CUDA's: the CPU sweeps
# the serial oracle alone, so a phases-against-oracle test would compare the oracle with itself there.
for _fn in (
    test_zero_diagonal_rows_match_steel_kernel,  # Steel's sparse sweep kernel is CUDA's alone
    test_one_humanoid_matches_steel_bitwise,  # bitwise to Steel's CUDA sweep; its CPU sweep is the host loop
    test_passive_springs_match_steel_under_scene,  # the same
    test_worlds_fold_matches_steel_bitwise,  # the same, and the side sweeps' lanes
    test_worlds_fold_matches_steel_bitwise_4096,  # the same
    test_phases_match_serial_oracle,  # the phases
    test_scene_solve_rests_like_steel,  # Steel's heap reference is its colored propagation response, matrix-free
    test_captured_steps_match_eager_with_changing_limits,  # graph capture
    test_color_sweep_matches_serial_walk,  # the phases
    test_unroll_fallback_matches_while,  # the phases' color loops
    test_tail_sweep_matches_serial_walk,  # the phases
    test_capture_before_coupling_replays_after,  # graph capture
    test_captured_steps_match_eager_with_colors,  # graph capture
    test_phase_orders_match_their_oracles,  # the phases
):
    add_function_test(TestSteelSceneSweep, _fn.__name__, _fn, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
