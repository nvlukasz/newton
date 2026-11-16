# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device coloring of SolverSteel's two-sided scene units: pair rounds, color cap, tail and color order."""

import json
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel.scene_units import COLOR_ROUND_MAX, SCAN_TILE, SceneUnits
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import CONTACT_MAX
from newton.tests.steel._scene_fixtures import _scene_heap, _units_for, _units_view
from newton.tests.steel.test_scene_units import (
    _collide,
    _initial_state,
    _one_sided,
    _scene,
    _solver,
    _sorted_view,
    _step,
    _stub,
    _stub_units,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

POISON = -7

STATISTICS_KEYS = {
    "units",
    "overflow",
    "colors",
    "tail",
    "tail_fallback",
    "rounds",
    "per_color",
    "two_sided",
    "limit_units",
}


def _chain_on_ground(device, links=3, worlds=2, box_on_link=None):
    """A free-rooted chain of box links lying on the ground, one articulation per world.

    With ``box_on_link``, a free box rests on that link of each chain.
    """
    template = newton.ModelBuilder(up_axis=newton.Axis.Z)
    template.add_ground_plane()
    joints = []
    parent = -1
    for i in range(links):
        link = template.add_link(xform=wp.transform(wp.vec3(0.25 * i, 0.0, 0.049), wp.quat_identity()))
        template.add_shape_box(link, hx=0.1, hy=0.08, hz=0.05)
        if parent < 0:
            joints.append(template.add_joint_free(child=link))
        else:
            joints.append(
                template.add_joint_revolute(
                    parent,
                    link,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=wp.transform(wp.vec3(0.125, 0.0, 0.0), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(-0.125, 0.0, 0.0), wp.quat_identity()),
                )
            )
        parent = link
    template.add_articulation(joints)
    if box_on_link is not None:
        # The link's top face is at z = 0.099; the box's bottom face rests 1 mm above it.
        box = template.add_body(xform=wp.transform(wp.vec3(0.25 * box_on_link, 0.0, 0.14), wp.quat_identity()))
        template.add_shape_box(box, hx=0.04, hy=0.04, hz=0.04)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.replicate(template, copy_count=worlds)
    model = builder.finalize(device=device)
    model.rigid_contact_max = CONTACT_MAX
    return model


def _colored_view(units):
    """The sorted view plus each unit's color: colors are stored per slot, like the other unit arrays."""
    view = _sorted_view(units)
    count = view["key"].size
    view["color"] = units.unit_color.numpy()[units.unit_order.numpy()[:count]]
    return view


def _two_sided(view):
    """Whether the unit at each sorted position joins two articulations: only those are colored."""
    return ~_one_sided(view["side_a"], view["side_b"])


def _lowest_clear_bit(used):
    bit = 0
    while bit < 64 and (used >> bit) & 1:
        bit += 1
    return bit


def _local_min_round(side_a, side_b, color, mask, pending, cap, side_count):
    """One ``local_min`` round: the first uncolored unit on both of its sides takes the lowest free color."""
    first = np.full(side_count, side_a.size)
    np.minimum.at(first, side_a[pending], pending)
    np.minimum.at(first, side_b[pending], pending)
    minima = pending[(first[side_a[pending]] == pending) & (first[side_b[pending]] == pending)]
    # Each side has at most one local minimum per round, so the order of this loop is immaterial.
    for p in minima:
        a, b = int(side_a[p]), int(side_b[p])
        bit = _lowest_clear_bit(mask[a] | mask[b])
        if bit < cap:
            color[p] = bit
            mask[a] |= 1 << bit
            mask[b] |= 1 << bit
        else:
            color[p] = cap


def _speculative_round(side_a, side_b, color, mask, pending, cap):
    """One ``speculative`` round: owners propose first-fit, and a proposal no earlier neighbor made wins."""
    # Each side_a proposes for its own units in sorted order, over the masks at the start of the round.
    owned = {}
    proposal = {}
    for p in pending:
        a, b = int(side_a[p]), int(side_b[p])
        own = owned.get(a, mask[a])
        bit = _lowest_clear_bit(own | mask[b])
        if bit < cap:
            proposal[p] = bit
            owned[a] = own | (1 << bit)
        elif _lowest_clear_bit(mask[a] | mask[b]) >= cap:
            # No color below the cap is free in the committed masks: the tail.
            proposal[p] = cap
        # Otherwise only the owner's proposals of this round fill the colors: no proposal, retry next round.
    # A proposal loses to any earlier unit on either side with the same proposal, winner or not.
    proposed = set()
    winners = []
    for p in pending:
        if p not in proposal:
            continue
        a, b, c = int(side_a[p]), int(side_b[p]), proposal[p]
        if c >= cap:
            color[p] = cap
            continue
        if (a, c) not in proposed and (b, c) not in proposed:
            winners.append(p)
        proposed.update(((a, c), (b, c)))
    for p in winners:
        c = proposal[p]
        color[p] = c
        mask[int(side_a[p])] |= 1 << c
        mask[int(side_b[p])] |= 1 << c


def _reference_coloring(view, side_count, cap, pair_rule):
    """The coloring rules on the host: pair rounds of ``pair_rule`` from empty masks, the cap and the tail fallback.

    Only two-sided units are colored; one-sided units keep ``-1``. Returns the colors in sorted
    order, each side's mask, the rounds run and the two-sided units the round cap left.
    """
    side_a, side_b = view["side_a"], view["side_b"]
    color = np.full(side_a.size, -1, dtype=np.int64)
    mask = [0] * side_count
    two_sided = np.flatnonzero(_two_sided(view))
    rounds = 0
    while np.any(color[two_sided] == -1) and rounds < COLOR_ROUND_MAX:
        rounds += 1
        pending = two_sided[color[two_sided] == -1]
        if pair_rule == "speculative":
            _speculative_round(side_a, side_b, color, mask, pending, cap)
        else:
            _local_min_round(side_a, side_b, color, mask, pending, cap, side_count)
    left = two_sided[color[two_sided] == -1]
    color[left] = cap
    return color, mask, rounds, int(left.size)


def _assert_valid_coloring(test, units, view):
    """One-sided units are uncolored; two-sided units sharing a side differ in color unless one has the tail color."""
    cap = units.color_cap
    color = view["color"]
    side_a, side_b = view["side_a"], view["side_b"]
    two_sided = _two_sided(view)
    test.assertGreater(color.size, 0)
    np.testing.assert_array_equal(color[~two_sided], -1, err_msg="a one-sided unit is colored")
    test.assertTrue(np.all((color[two_sided] >= 0) & (color[two_sided] <= cap)), msg=f"colors outside [0, {cap}]")
    masks = units.side_mask.numpy()
    for side in range(units.side_count):
        touching = np.flatnonzero(two_sided & ((side_a == side) | (side_b == side)))
        colored = color[touching][color[touching] < cap]
        test.assertEqual(np.unique(colored).size, colored.size, msg=f"side {side} repeats a color")
        expected_mask = 0
        for c in colored:
            expected_mask |= 1 << int(c)
        test.assertEqual(int(masks[side]), expected_mask, msg=f"side {side} mask")


def _assert_matches_reference(test, units, view):
    color, mask, rounds, left = _reference_coloring(view, units.side_count, units.color_cap, units.pair_rule)
    np.testing.assert_array_equal(view["color"], color)
    np.testing.assert_array_equal(units.side_mask.numpy().astype(np.uint64), np.asarray(mask, dtype=np.uint64))
    test.assertEqual(int(units.rounds.numpy()[0]), rounds)
    test.assertEqual(int(units.uncolored.numpy()[0]), left)


def test_one_sided_units_stay_uncolored(test, device):
    """Leave one-sided units uncolored and skip the coloring when no unit joins two articulations."""
    for limits in (False, True):
        with test.subTest(enable_joint_limits=limits):
            model = _chain_on_ground(device)
            solver = _solver(model, enable_joint_limits=limits)
            units = _units_for(solver, _step(model, solver))
            view = _colored_view(units)
            test.assertGreater(view["color"].size, 0)
            np.testing.assert_array_equal(view["side_b"], -1)
            test.assertEqual(bool(np.any(view["contact"] == -1)), limits)
            test.assertEqual(units.side_count, 2)
            np.testing.assert_array_equal(view["color"], -1)
            test.assertEqual(int(units.has_two_sided.numpy()[0]), 0)
            test.assertEqual(int(units.rounds.numpy()[0]), 0)
            test.assertEqual(int(units.uncolored.numpy()[0]), 0)
            np.testing.assert_array_equal(units.side_mask.numpy(), 0)
            np.testing.assert_array_equal(units.color_offsets.numpy(), 0)
            np.testing.assert_array_equal(units.color_count_per.numpy(), 0)
            np.testing.assert_array_equal(units.one_sided_count.numpy(), np.diff(units.side_start.numpy()))
            stats = units.statistics()
            test.assertEqual(stats["units"], view["color"].size)
            test.assertEqual((stats["two_sided"], stats["colors"], stats["tail"], stats["rounds"]), (0, 0, 0, 0))
            test.assertEqual(stats["per_color"], [0] * units.color_cap)


def test_valid_coloring_on_heap(test, device):
    """Color the box heap's two-sided units validly from empty masks, in few speculative rounds."""
    model = _scene("heap", device)
    solver = _solver(model)
    units = _units_for(solver, _step(model, solver))
    test.assertEqual(units.pair_rule, "speculative")
    test.assertEqual(units.color_cap, 64)
    view = _colored_view(units)
    side_a, side_b = view["side_a"], view["side_b"]
    two_sided = _two_sided(view)
    test.assertTrue(np.any(side_b < 0))
    test.assertTrue(np.any(two_sided))
    _assert_valid_coloring(test, units, view)
    _assert_matches_reference(test, units, view)
    # Colors count from 0 for the two-sided units alone.
    test.assertEqual(int(view["color"][two_sided].min()), 0)
    # Every unit finds a color within the rounds: none falls back to the tail.
    test.assertEqual(int(units.uncolored.numpy()[0]), 0)
    test.assertFalse(np.any(view["color"] == units.color_cap))
    test.assertGreater(int(units.rounds.numpy()[0]), 0)
    # Far below the 64-round cap: local_min would need hundreds of rounds here.
    test.assertLessEqual(int(units.rounds.numpy()[0]), 32)
    test.assertTrue(np.any(side_a[two_sided] != side_b[two_sided]))


def test_self_contact_is_one_sided(test, device):
    """Count contacts between two links of one articulation as one-sided units of that side, left uncolored."""
    for limits in (False, True):
        with test.subTest(enable_joint_limits=limits):
            model = _scene("scissor", device)
            solver = _solver(model, enable_joint_limits=limits)
            units = _units_for(solver, _step(model, solver))
            view = _colored_view(units)
            contacts = view["contact"] >= 0
            test.assertTrue(np.any(contacts))
            np.testing.assert_array_equal(view["side_a"][contacts], view["side_b"][contacts])
            test.assertEqual(units.side_count, 1)
            np.testing.assert_array_equal(view["color"], -1)
            test.assertEqual(int(units.one_sided_count.numpy()[0]), view["color"].size)
            test.assertEqual(int(units.side_mask.numpy()[0]), 0)
            test.assertEqual(int(units.has_two_sided.numpy()[0]), 0)
            test.assertEqual(int(units.rounds.numpy()[0]), 0)


def test_free_body_on_articulation_is_two_sided(test, device):
    """Color a contact between a chain link and a free box resting on it: a unit joining the two articulations."""
    model = _chain_on_ground(device, worlds=1, box_on_link=1)
    solver = _solver(model)
    units = _units_for(solver, _step(model, solver))
    view = _colored_view(units)
    body_to_art = solver._body_to_articulation.numpy()
    chain, box = int(body_to_art[1]), int(body_to_art[3])
    test.assertNotEqual(chain, box)
    test.assertEqual(int(solver._model_plan.is_free_rigid[box]), 1)
    test.assertEqual(int(solver._model_plan.is_free_rigid[chain]), 0)
    two_sided = np.flatnonzero(_two_sided(view))
    test.assertGreater(two_sided.size, 0)
    for p in two_sided:
        test.assertEqual(sorted((int(view["side_a"][p]), int(view["side_b"][p]))), sorted((chain, box)))
    # They share both sides, so first-fit gives them the colors 0, 1, ... in sorted order.
    np.testing.assert_array_equal(view["color"][two_sided], np.arange(two_sided.size))
    test.assertEqual(int(units.has_two_sided.numpy()[0]), 1)
    np.testing.assert_array_equal(units.side_coupled.numpy()[[chain, box]], 1)
    _assert_valid_coloring(test, units, view)
    _assert_matches_reference(test, units, view)
    _assert_color_order(test, units)


def test_tail_when_cap_exceeded(test, device):
    """With ``scene_color_cap=2``, give the two-sided units that find no free color below the cap the tail color 2.

    A speculative unit takes the tail only when its sides' committed colors leave none below the cap
    free; when only its owner's proposals of the round block it, it retries next round.
    """
    model = _scene("heap", device)
    solver = _solver(model, {"scene_color_cap": 2})
    units = _units_for(solver, _step(model, solver))
    test.assertEqual(units.color_cap, 2)
    view = _colored_view(units)
    color = view["color"]
    two_sided = _two_sided(view)
    test.assertTrue(np.all(np.isin(color[two_sided], (0, 1, 2))))
    test.assertLessEqual(int(units.rounds.numpy()[0]), COLOR_ROUND_MAX)
    test.assertTrue(np.all(units.side_mask.numpy() <= 0b11))
    # Units of the tail color may share a side.
    tail = color == 2
    tail_sides = np.concatenate([view["side_a"][tail], view["side_b"][tail]])
    test.assertGreater(int(np.bincount(tail_sides[tail_sides >= 0]).max()), 1)
    _assert_valid_coloring(test, units, view)
    _assert_matches_reference(test, units, view)
    # A side's used colors are one 64-bit mask.
    for bad_cap in (0, 65):
        with test.assertRaises(ValueError):
            SceneUnits(_units_view(_solver(model, {"scene_color_cap": bad_cap})))


def test_colors_across_worlds(test, device):
    """Color the units of a heap replicated into two worlds as one scene-wide list."""
    model = _scene_heap(device, worlds=2)
    test.assertEqual(model.world_count, 2)
    solver = _solver(model)
    units = _units_for(solver, _step(model, solver))
    view = _colored_view(units)
    # Both worlds have units, and no unit joins the two worlds.
    per_world = units.side_count // 2
    world_a = view["side_a"] // per_world
    test.assertEqual(set(world_a.tolist()), {0, 1})
    two_sided = view["side_b"] >= 0
    np.testing.assert_array_equal(view["side_b"][two_sided] // per_world, world_a[two_sided])
    _assert_valid_coloring(test, units, view)
    _assert_matches_reference(test, units, view)


def test_determinism(test, device):
    """Give two solvers on the same contacts and state identical colors, masks and round counts."""
    model = _scene("heap", device)
    contacts = _collide(model, _initial_state(model))
    results = []
    for _ in range(2):
        solver = _solver(model)
        _step(model, solver, contacts)
        units = _units_for(solver, contacts)
        view = _colored_view(units)
        results.append(
            {
                "color": view["color"],
                "side_mask": units.side_mask.numpy().copy(),
                "rounds": units.rounds.numpy().copy(),
                "uncolored": units.uncolored.numpy().copy(),
                "order_by_color": units.order_by_color.numpy()[: int(units.color_offsets.numpy()[-1])].copy(),
            }
        )
    test.assertGreater(results[0]["color"].size, 0)
    for name in results[0]:
        np.testing.assert_array_equal(results[0][name], results[1][name], err_msg=name)


def test_local_min_rule_on_heap(test, device):
    """Keep the ``local_min`` pair rule valid and deterministic on the box heap (it may fall back to the tail)."""
    model = _scene("heap", device)
    contacts = _collide(model, _initial_state(model))
    colors = []
    for _ in range(2):
        solver = _solver(model, {"scene_pair_rule": "local_min"})
        _step(model, solver, contacts)
        units = _units_for(solver, contacts)
        test.assertEqual(units.pair_rule, "local_min")
        view = _colored_view(units)
        _assert_valid_coloring(test, units, view)
        _assert_matches_reference(test, units, view)
        test.assertLessEqual(int(units.rounds.numpy()[0]), COLOR_ROUND_MAX)
        colors.append(view["color"])
    np.testing.assert_array_equal(colors[0], colors[1])


def test_invalid_pair_rule_raises(test, device):
    """Reject a pair rule other than ``"speculative"`` and ``"local_min"`` at construction."""
    model = _scene("heap", device)
    for rule in ("greedy", "", None):
        solver = _solver(model, {"scene_pair_rule": rule})
        with test.subTest(pair_rule=rule), test.assertRaises(ValueError):
            SceneUnits(_units_view(solver))


def _assert_color_order(test, units):
    """Count each color of the two-sided units, scan the counts into offsets and list their positions by color, stably."""
    view = _colored_view(units)
    cap = units.color_cap
    color = view["color"]
    two_sided = np.flatnonzero(_two_sided(view))
    counts = units.color_count_per.numpy()
    offsets = units.color_offsets.numpy()
    test.assertEqual(counts.shape, (cap + 2,))
    test.assertEqual(offsets.shape, (cap + 2,))
    # Slots 0..cap-1 are the colors and slot cap the tail; slot cap+1 counts nothing, so its offset is the total.
    np.testing.assert_array_equal(counts[: cap + 1], np.bincount(color[two_sided], minlength=cap + 1))
    test.assertEqual(int(counts[cap + 1]), 0)
    np.testing.assert_array_equal(offsets, np.concatenate([[0], np.cumsum(counts)[:-1]]))
    test.assertEqual(int(offsets[cap + 1]), two_sided.size)
    test.assertEqual(units.order_by_color.shape, (units.unit_max,))
    order = units.order_by_color.numpy()[: two_sided.size]
    # A permutation of the two-sided units' sorted positions, grouped by color, in increasing position within a color.
    np.testing.assert_array_equal(np.sort(order), two_sided)
    for c in range(cap + 1):
        run = order[offsets[c] : offsets[c + 1]]
        np.testing.assert_array_equal(color[run], c, err_msg=f"color {c}")
        test.assertTrue(np.all(np.diff(run) > 0), msg=f"color {c} is not in position order")
    np.testing.assert_array_equal(order, two_sided[np.argsort(color[two_sided], kind="stable")])


def test_color_offsets_partition_units(test, device):
    """Group the two-sided units' sorted positions by color: offsets scan the per-color counts, runs keep position order."""
    cases = (
        ("heap", {}, {}),
        ("heap", {"scene_color_cap": 2}, {}),
        ("heap", {"scene_pair_rule": "local_min"}, {}),
        ("articulated", {}, {"enable_joint_limits": True}),
    )
    for name, overrides, options in cases:
        with test.subTest(scene=name, **overrides, **options):
            model = _scene(name, device)
            solver = _solver(model, overrides, **options)
            units = _units_for(solver, _step(model, solver))
            test.assertGreater(int(units.unit_count.numpy()[0]), 0)
            _assert_color_order(test, units)
            if name == "heap":
                # The heap's two-sided units span many blocks of the counting scatter.
                test.assertGreater(int(units.incidence_start.numpy()[-1]) // 2, 2 * 64)
            if overrides:
                test.assertGreater(int(units.color_count_per.numpy()[units.color_cap]), 0)


def test_color_statistics_reads_back(test, device):
    """Read the coloring back outside capture: unit counts, two-sided per-color counts, tail, rounds."""
    cases = (
        ("heap", {"scene_color_cap": 2}, {}),
        ("heap", {"scene_pair_rule": "local_min"}, {}),
        ("articulated", {}, {"enable_joint_limits": True}),
    )
    for name, overrides, options in cases:
        with test.subTest(scene=name, **overrides, **options):
            model = _scene(name, device)
            solver = _solver(model, overrides, **options)
            units = _units_for(solver, _step(model, solver))
            stats = units.statistics()
            test.assertEqual(set(stats), STATISTICS_KEYS)
            # Plain Python values, so the statistics serialize to JSON.
            test.assertEqual(json.loads(json.dumps(stats)), stats)
            view = _colored_view(units)
            cap = units.color_cap
            two_sided = _two_sided(view)
            color = view["color"][two_sided]
            test.assertEqual(stats["units"], int(units.unit_count.numpy()[0]))
            test.assertEqual(stats["units"], view["color"].size)
            test.assertEqual(stats["overflow"], int(units.unit_overflow.numpy()[0]))
            test.assertEqual(stats["two_sided"], int(np.sum(two_sided)))
            test.assertEqual(len(stats["per_color"]), cap)
            test.assertEqual(sum(stats["per_color"]) + stats["tail"], stats["two_sided"])
            test.assertEqual(stats["per_color"], np.bincount(color, minlength=cap + 1)[:cap].tolist())
            test.assertEqual(stats["tail"], int(np.sum(color == cap)))
            used = color[color < cap]
            test.assertEqual(stats["colors"], int(used.max()) + 1 if used.size else 0)
            test.assertEqual(stats["rounds"], int(units.rounds.numpy()[0]))
            test.assertEqual(stats["tail_fallback"], int(units.uncolored.numpy()[0]))
            test.assertEqual(stats["limit_units"], int(np.sum(view["contact"] == -1)))
            if name == "articulated":
                test.assertGreater(stats["limit_units"], 0)
            if overrides.get("scene_color_cap") == 2:
                test.assertEqual(stats["colors"], 2)
                test.assertGreater(stats["tail"], 0)
            if overrides.get("scene_pair_rule") == "local_min":
                # The heap needs more rounds than the cap under local_min, so the fallback colors the rest.
                test.assertEqual(stats["rounds"], COLOR_ROUND_MAX)
                test.assertGreater(stats["tail_fallback"], 0)
                test.assertGreaterEqual(stats["tail"], stats["tail_fallback"])


@unittest.skipUnless(hasattr(SolverSteel, "scene_statistics"), "needs the read-back API")
def test_time_scene_units_keeps_the_step(test, device):
    """Time build and color on the last step's contacts, from graphs or eagerly, leaving the step's units as they were."""
    model = _scene("tray", device)
    solver = _solver(model, solve_mode="scene", enable_joint_limits=True)
    _step(model, solver)
    units = solver._scene_units

    def snapshot():
        view = _colored_view(units)
        view["order_by_color"] = units.order_by_color.numpy()[: int(units.color_offsets.numpy()[-1])].copy()
        names = ("side_mask", "side_start", "incidence_start", "color_count_per", "color_offsets", "one_sided_count")
        for name in (*names, "side_coupled", "has_two_sided"):
            view[name] = getattr(units, name).numpy().copy()
        view["incidence_unit"] = units.incidence_unit.numpy()[: view["incidence_start"][-1]].copy()
        return view, solver.color_statistics()

    before, before_stats = snapshot()
    test.assertGreater(before_stats["rounds"], 0)
    # Replays of captured graphs on a CUDA device, eager runs on the CPU or when asked.
    default_mode = "graph" if wp.get_device(device).is_cuda and wp.is_conditional_graph_supported() else "eager"
    for eager, mode in ((False, default_mode), (True, "eager")):
        with test.subTest(eager=eager):
            timing = solver.time_scene_units(n=3, eager=eager)
            test.assertEqual(set(timing), {"build_ms", "color_ms", "n", "mode"})
            test.assertEqual(timing["n"], 3)
            test.assertEqual(timing["mode"], mode)
            test.assertGreater(timing["build_ms"], 0.0)
            test.assertGreater(timing["color_ms"], 0.0)
            after, after_stats = snapshot()
            test.assertEqual(after_stats, before_stats)
            for name, values in before.items():
                np.testing.assert_array_equal(after[name], values, err_msg=name)
    with test.assertRaises(ValueError):
        solver.time_scene_units(n=0)


def test_full_mask_takes_the_tail(test, device):
    """Fill all 64 bits of a hub's mask at ``color_cap=64``: its 65th to 70th two-sided units take the tail."""
    # Side 70 is a hub with one two-sided unit against each of sides 0..69; sides 0 and 1 also touch the ground.
    pairs = [(70, k) for k in range(70)] + [(0, -1), (1, -1)]
    for rule in ("speculative", "local_min"):
        with test.subTest(pair_rule=rule):
            units = _stub_units(device, 71, pairs, color_cap=64, pair_rule=rule)
            view = _colored_view(units)
            hub = view["side_a"] == 70
            np.testing.assert_array_equal(view["side_b"][hub], np.arange(70))
            # The 65th two-sided unit (first-fit bit 64) and those after it take the tail; no shift by 64 happens.
            np.testing.assert_array_equal(view["color"][hub], np.concatenate([np.arange(64), np.full(6, 64)]))
            np.testing.assert_array_equal(view["color"][~hub], -1)
            masks = units.side_mask.numpy()
            test.assertEqual(int(masks[70]), 2**64 - 1)
            test.assertEqual([int(m) for m in masks[:70]], [1 << k for k in range(64)] + [0] * 6)
            # Speculative: two rounds, the second finds the hub full. local_min: one hub unit per round,
            # so the round cap leaves the last six to the tail fallback.
            test.assertEqual(int(units.uncolored.numpy()[0]), 0 if rule == "speculative" else 6)
            _assert_valid_coloring(test, units, view)
            _assert_matches_reference(test, units, view)
            _assert_color_order(test, units)
            stats = units.statistics()
            test.assertEqual(stats["two_sided"], 70)
            test.assertEqual(stats["tail"], 6)
            test.assertEqual(stats["colors"], 64)
            test.assertEqual(stats["per_color"], [1] * 64)
            test.assertEqual(stats["units"], 72)


# Six sides with seven two-sided units among twelve; the same contacts all on the ground have none.
_COUPLED_PAIRS = [(1, 0), (2, 1), (3, -1), (4, 2), (5, 5), (2, -1), (5, 4), (0, -1), (2, 0), (4, 1), (3, 2), (5, -1)]
_COUPLED_TWO_SIDED = 7


def _set_coupled(solver, coupled):
    """Route the stub's contacts as ``_COUPLED_PAIRS``, or with every second end on the ground."""
    solver._contact_art_b.assign([b if coupled else -1 for _, b in _COUPLED_PAIRS])


def _color_snapshot(units):
    """Everything the coloring and the order by color leave behind, read back."""
    view = _colored_view(units)
    two_sided = int(units.incidence_start.numpy()[-1]) // 2
    view["order_by_color"] = units.order_by_color.numpy()[:two_sided].copy()
    for name in ("has_two_sided", "side_mask", "rounds", "uncolored", "keep_going", "color_count_per", "color_offsets"):
        view[name] = getattr(units, name).numpy().copy()
    return view, units.statistics()


def _assert_nothing_colored(test, units, view, stats):
    """The state a coloring must leave when no unit is two-sided, whatever an earlier coloring left."""
    test.assertGreater(view["color"].size, 0)
    test.assertEqual(int(units.has_two_sided.numpy()[0]), 0)
    np.testing.assert_array_equal(view["color"], -1)
    np.testing.assert_array_equal(units.side_mask.numpy(), 0)
    np.testing.assert_array_equal(units.color_count_per.numpy(), 0)
    np.testing.assert_array_equal(units.color_offsets.numpy(), 0)
    # The order by color lists no unit: entries past color_offsets[color_cap + 1] are stale by contract.
    test.assertEqual(int(units.color_offsets.numpy()[-1]), 0)
    test.assertEqual(int(units.rounds.numpy()[0]), 0)
    test.assertEqual(int(units.uncolored.numpy()[0]), 0)
    test.assertEqual(int(units.keep_going.numpy()[0]), 0)
    test.assertEqual((stats["two_sided"], stats["colors"], stats["tail"], stats["rounds"]), (0, 0, 0, 0))
    test.assertEqual(stats["per_color"], [0] * units.color_cap)
    test.assertEqual(stats["units"], len(_COUPLED_PAIRS))


def test_recolor_without_two_sided_units_clears(test, device):
    """Recolor one SceneUnits after its two-sided units are gone: the skipped branch leaves nothing of the last coloring."""
    for rule in ("speculative", "local_min"):
        with test.subTest(pair_rule=rule):
            solver, contacts = _stub(device, 6, _COUPLED_PAIRS, pair_rule=rule)
            units = SceneUnits(solver)
            units.build(solver, contacts)
            units.color()
            view, stats = _color_snapshot(units)
            test.assertEqual(stats["two_sided"], _COUPLED_TWO_SIDED)
            test.assertGreater(stats["rounds"], 0)
            test.assertTrue(np.any(units.side_mask.numpy() != 0))
            test.assertEqual(int(units.color_offsets.numpy()[-1]), _COUPLED_TWO_SIDED)
            _assert_valid_coloring(test, units, view)
            _set_coupled(solver, False)
            units.build(solver, contacts)
            units.color()
            view, stats = _color_snapshot(units)
            _assert_nothing_colored(test, units, view, stats)


def test_captured_color_follows_two_sided_flips(test, device):
    """Replay one captured build and color while the two-sided units come, go and come back: each replay equals eager.

    The colour arrays are poisoned before every replay, so a clear missing before the conditional
    branch shows on the replay that skips it.
    """
    solver, contacts = _stub(device, 6, _COUPLED_PAIRS)
    units = SceneUnits(solver, 2 * len(_COUPLED_PAIRS))
    expected = {}
    for coupled in (True, False):
        _set_coupled(solver, coupled)
        units.build(solver, contacts)
        units.color()
        expected[coupled] = _color_snapshot(units)
    test.assertEqual(expected[True][1]["two_sided"], _COUPLED_TWO_SIDED)
    test.assertEqual(expected[False][1]["two_sided"], 0)
    with wp.ScopedCapture(units.device) as capture:
        units.build(solver, contacts)
        units.color()
    for coupled in (True, False, True):
        _set_coupled(solver, coupled)
        for name in ("unit_color", "order_by_color", "_color_proposal", "_color_counters", "_color_order_counts"):
            getattr(units, name).fill_(POISON)
        units.has_two_sided.fill_(POISON)
        units.side_mask.fill_(0xDEADBEEF)
        wp.capture_launch(capture.graph)
        view, stats = _color_snapshot(units)
        eager_view, eager_stats = expected[coupled]
        test.assertEqual(stats, eager_stats, msg=f"coupled={coupled}")
        for name, values in eager_view.items():
            np.testing.assert_array_equal(view[name], values, err_msg=f"{name}, coupled={coupled}")
        if not coupled:
            _assert_nothing_colored(test, units, view, stats)


def test_tile_scan_spans_several_trips(test, device):
    """Scan past three trips of the one-block tile-total scan exactly as NumPy does, and order a hub's colors with it.

    The order by color scans ``(color_cap + 2) * ceil(unit_max / 64)`` counts: at ``unit_max``
    2,100,000 that is 66 x 32,813 = 2,165,658 entries in 2,115 tiles of 1,024, so
    :func:`scan_tile_totals` loops three times, and the hub's tail run starts in the third trip.
    """
    pairs = [(70, k) for k in range(70)] + [(0, -1), (1, -1)]
    solver, contacts = _stub(device, 71, pairs)
    units = SceneUnits(solver, 2_100_000)
    count = units._color_block_count.shape[0]
    tiles = (count + SCAN_TILE - 1) // SCAN_TILE
    test.assertGreater(tiles, 2 * SCAN_TILE)
    values = np.random.default_rng(7).integers(0, 4, count).astype(np.int32)
    scanned = wp.zeros(count, dtype=wp.int32, device=units.device)
    units._scan_exclusive(wp.array(values, dtype=wp.int32, device=units.device), scanned)
    np.testing.assert_array_equal(scanned.numpy(), np.concatenate([[0], np.cumsum(values)[:-1]]))
    units.build(solver, contacts)
    units.color()
    test.assertGreater(units.color_cap * units._color_blocks // SCAN_TILE, 2 * SCAN_TILE)
    _assert_color_order(test, units)
    np.testing.assert_array_equal(units.color_count_per.numpy()[: units.color_cap + 1], [1] * 64 + [6])


class TestSteelColoring(unittest.TestCase):
    pass


# Every coloring kernel has a CPU version; an eager coloring reads its loop conditions back.
for _fn in (
    test_one_sided_units_stay_uncolored,
    test_valid_coloring_on_heap,
    test_self_contact_is_one_sided,
    test_free_body_on_articulation_is_two_sided,
    test_tail_when_cap_exceeded,
    test_colors_across_worlds,
    test_determinism,
    test_local_min_rule_on_heap,
    test_invalid_pair_rule_raises,
    test_color_offsets_partition_units,
    test_color_statistics_reads_back,
    test_time_scene_units_keeps_the_step,
    test_full_mask_takes_the_tail,
    test_recolor_without_two_sided_units_clears,
    test_tile_scan_spans_several_trips,
):
    add_function_test(TestSteelColoring, _fn.__name__, _fn, devices=get_test_devices())
# A captured graph needs a CUDA device.
add_function_test(
    TestSteelColoring,
    test_captured_color_follows_two_sided_flips.__name__,
    test_captured_color_follows_two_sided_flips,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
