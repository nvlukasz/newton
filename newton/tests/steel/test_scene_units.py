# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scene-wide constraint units of SolverSteel's scene mode: collection, counting sort, side CSR and incidence."""

import types
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel._testing import kernel_overrides
from newton._src.solvers.steel.scene_units import SCAN_TILE, SceneUnits
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import CONTACT_MAX, _build_contact_scene, _build_scissor_model, _heap
from newton.tests.steel._scene_fixtures import _units_for, _units_view
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

DT = 1.0 / 240.0

POISON = -7


def _scene(name, device):
    """Free-box heap, heap on a kinematic tray, articulated contacts, or one self-colliding articulation."""
    if name == "heap":
        model = _heap(device)
    elif name == "tray":
        model = _heap(device, kinematic_tray=True)
    elif name == "articulated":
        model = _build_contact_scene(device)
    elif name == "scissor":
        model = _build_scissor_model(device)
    else:
        raise ValueError(name)
    model.rigid_contact_max = CONTACT_MAX
    return model


def _solver(model, overrides=None, **options):
    """A solver without the scene mode, with room for every contact row, so no contact is dropped.

    ``overrides`` are its kernel overrides: the units' ``scene_color_cap`` and ``scene_pair_rule``.
    """
    settings = {"free_body_constraint_max": CONTACT_MAX, "articulation_constraint_max": 256, **options}
    with kernel_overrides(overrides or {}):
        return SolverSteel(model, **settings)


def _initial_state(model):
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state


def _collide(model, state):
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=CONTACT_MAX, deterministic=True)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return contacts


def _step(model, solver, contacts=None):
    """Take one step from the initial state; collide there unless ``contacts`` are given."""
    state = _initial_state(model)
    if contacts is None:
        contacts = _collide(model, state)
    solver.step(state, model.state(), model.control(), contacts, DT)
    return contacts


def _side(art, response):
    """Steel's side of an articulation index: itself when it has response DOFs, else -1."""
    art = np.asarray(art)
    return np.where((art >= 0) & (response[np.maximum(art, 0)] > 0), art, -1)


def _one_sided(side_a, side_b):
    """Units with one side: limits units, contacts with the ground or a static body, and self-contacts."""
    return (np.asarray(side_b) < 0) | (np.asarray(side_b) == np.asarray(side_a))


def _sort_field(side_a, side_b):
    """The sort key's middle field: ``0`` for a one-sided unit, ``side_b + 1`` for a two-sided one."""
    return np.where(_one_sided(side_a, side_b), 0, np.asarray(side_b) + 1)


def _limit_bearing(solver):
    """Responding articulations with a finite bound on a PRISMATIC, REVOLUTE or D6 DOF, recounted from the model."""
    model = solver.model
    lower = model.joint_limit_lower.numpy()
    upper = model.joint_limit_upper.numpy()
    qd_start = model.joint_qd_start.numpy()
    dof_dim = model.joint_dof_dim.numpy()
    limit_types = (int(newton.JointType.PRISMATIC), int(newton.JointType.REVOLUTE), int(newton.JointType.D6))
    limited = np.zeros(model.joint_dof_count, dtype=bool)
    for joint, joint_type in enumerate(model.joint_type.numpy()):
        if int(joint_type) in limit_types:
            dofs = slice(int(qd_start[joint]), int(qd_start[joint] + dof_dim[joint].sum()))
            limited[dofs] = np.isfinite(lower[dofs]) | np.isfinite(upper[dofs])
    response = solver._articulation_response_dof_count.numpy()
    dof_start = solver._articulation_dof_start.numpy()
    arts = [
        art
        for art in range(model.articulation_count)
        if response[art] > 0 and limited[dof_start[art] : dof_start[art] + response[art]].any()
    ]
    return np.asarray(arts, dtype=np.int64)


def _collected_units(solver, contacts, limits):
    """Units recounted on the host from the solver's contact routing, in slot order.

    Slots are a scan over flags: the limits units by articulation first, then the contacts by index.
    """
    count = int(contacts.rigid_contact_count.numpy()[0])
    path = solver._contact_path.numpy()[:count]
    response = solver._articulation_response_dof_count.numpy()
    routed = np.flatnonzero(path >= 0)
    side_0 = _side(solver._contact_art_a.numpy()[routed], response)
    side_1 = _side(solver._contact_art_b.numpy()[routed], response)
    keep = (side_0 >= 0) | (side_1 >= 0)
    contact = routed[keep]
    side_a = np.maximum(side_0, side_1)[keep]
    side_b = np.minimum(side_0, side_1)[keep]
    rows = solver._contact_slots_needed.numpy()[contact]
    if limits:
        arts = _limit_bearing(solver)
        contact = np.concatenate([np.full(arts.size, -1), contact])
        side_a = np.concatenate([arts, side_a])
        side_b = np.concatenate([np.full(arts.size, -1), side_b])
        rows = np.concatenate([np.zeros(arts.size, dtype=rows.dtype), rows])
    return {"side_a": side_a, "side_b": side_b, "contact": contact, "rows": rows}


def _expected_units(solver, contacts, limits):
    """The collected units sorted by ``(side_a, f, contact)``, with ``f`` from :func:`_sort_field`."""
    units = _collected_units(solver, contacts, limits)
    order = np.lexsort((units["contact"], _sort_field(units["side_a"], units["side_b"]), units["side_a"]))
    return {name: values[order] for name, values in units.items()}


def _slots(units):
    """The units by slot, ``[0, unit_count)``."""
    count = int(units.unit_count.numpy()[0])
    return {
        "side_a": units.unit_side_a.numpy()[:count],
        "side_b": units.unit_side_b.numpy()[:count],
        "contact": units.unit_contact.numpy()[:count],
        "rows": units.unit_rows.numpy()[:count],
    }


def _sorted_view(units):
    """The units in sorted order: the unit at sorted position ``p`` is slot ``unit_order[p]``."""
    count = int(units.unit_count.numpy()[0])
    order = units.unit_order.numpy()[:count]
    return {
        "side_a": units.unit_side_a.numpy()[order],
        "side_b": units.unit_side_b.numpy()[order],
        "contact": units.unit_contact.numpy()[order],
        "rows": units.unit_rows.numpy()[order],
        "key": units.unit_key.numpy()[order],
    }


def _packed_keys(view, units):
    """The sorted view's keys, packed with the field widths the units were allocated with."""
    side_a = view["side_a"].astype(np.int64)
    field = _sort_field(view["side_a"], view["side_b"]).astype(np.int64)
    contact = view["contact"].astype(np.int64)
    contact_bits = units.contact_bits
    return ((side_a + 1) << (units.side_bits + contact_bits)) | (field << contact_bits) | (contact + 1)


def _stub(device, side_count, pairs, color_cap=64, pair_rule="speculative"):
    """A stub solver and contact buffer whose contact ``c`` joins the articulations ``pairs[c]``.

    Side ``-1`` is the ground. Every contact is routed with one row and every articulation responds.
    """
    device = wp.get_device(device)
    count = len(pairs)
    solver = types.SimpleNamespace(
        model=types.SimpleNamespace(device=device, articulation_count=side_count),
        _enable_joint_limits=False,
        _enable_joint_friction=False,
        _has_drive_rows=False,
        _scene_late_rows_enabled=False,
        # The scene mode's per-articulation row-family bits (_scene_family_bearing): none on this stub.
        _scene_family_bearing=None,
        _kernel_overrides={"scene_color_cap": color_cap, "scene_pair_rule": pair_rule},
        _max_contacts_alloc=count,
        _contact_path=wp.zeros(count, dtype=wp.int32, device=device),
        _contact_art_a=wp.array([a for a, _ in pairs], dtype=wp.int32, device=device),
        _contact_art_b=wp.array([b for _, b in pairs], dtype=wp.int32, device=device),
        _articulation_response_dof_count=wp.full(side_count, 6, dtype=wp.int32, device=device),
        _contact_slots_needed=wp.ones(count, dtype=wp.int32, device=device),
    )
    contacts = types.SimpleNamespace(
        rigid_contact_count=wp.array([count], dtype=wp.int32, device=device), rigid_contact_max=count
    )
    return solver, contacts


def _stub_units(device, side_count, pairs, color_cap=64, pair_rule="speculative", limit_rows=None, color=True):
    """Build (and color) the units of a :func:`_stub` scene; ``limit_rows`` per articulation, if given."""
    solver, contacts = _stub(device, side_count, pairs, color_cap=color_cap, pair_rule=pair_rule)
    units = SceneUnits(solver)
    if limit_rows is not None:
        limit_rows = wp.array(limit_rows, dtype=wp.int32, device=solver.model.device)
    units.build(solver, contacts, limit_rows=limit_rows)
    if color:
        units.color()
    return units


def _hub_pairs(seed=3):
    """Five sides; side 4 holds 300 units of every kind (ground, self-contact, two-sided), shuffled among the rest."""
    hub = [(4, -1)] * 100 + [(4, 4)] * 50 + [(4, k) for k in range(4)] * 25 + [(k, 4) for k in range(4)] * 12
    hub += [(-1, 4)] * 2
    others = [(0, -1)] * 5 + [(1, 0)] * 5 + [(2, 1)] * 5 + [(3, -1)] * 5 + [(2, 2)] * 3 + [(3, 0)] * 4
    pairs = hub + others
    order = np.random.default_rng(seed).permutation(len(pairs))
    return 5, [pairs[i] for i in order]


def test_units_match_contacts(test, device):
    """Collect one unit per routed contact with a responding side, plus one limits unit per responding articulation.

    The units are built on a solver without the scene mode after its step (:func:`_units_for`), from the contact
    routes and row counts the expected units read; ``test_scene_rows`` checks the scene solve's units against its rows.
    """
    for name in ("heap", "tray", "articulated"):
        for limits in (False, True):
            with test.subTest(scene=name, enable_joint_limits=limits):
                model = _scene(name, device)
                solver = _solver(model, enable_joint_limits=limits)
                contacts = _step(model, solver)
                units = _units_for(solver, contacts)
                test.assertIsInstance(units, SceneUnits)
                test.assertEqual(units.side_count, model.articulation_count)
                expected = _expected_units(solver, contacts, limits)
                test.assertGreater(int(np.sum(expected["contact"] >= 0)), 0)
                test.assertEqual(int(units.unit_count.numpy()[0]), expected["contact"].size)
                test.assertEqual(int(units.unit_overflow.numpy()[0]), 0)
                view = _sorted_view(units)
                for field in ("side_a", "side_b", "contact", "rows"):
                    np.testing.assert_array_equal(view[field], expected[field], err_msg=field)
                # Slots are deterministic: the limits units by articulation, then the contacts by index.
                collected = _collected_units(solver, contacts, limits)
                slots = _slots(units)
                for field in ("side_a", "side_b", "contact", "rows"):
                    np.testing.assert_array_equal(slots[field], collected[field], err_msg=f"slot {field}")
                contact_units = view["contact"] >= 0
                test.assertTrue(np.all(np.isin(view["rows"][contact_units], (1, 3))))
                np.testing.assert_array_equal(
                    view["rows"][contact_units],
                    solver._contact_slots_needed.numpy()[view["contact"][contact_units]],
                )
                limit_units = view["contact"] == -1
                test.assertEqual(units.limit_unit_count, int(np.sum(limit_units)))
                if limits:
                    np.testing.assert_array_equal(np.flatnonzero(units.art_has_limits.numpy()), _limit_bearing(solver))
                    # A free rigid body has no joint limit, so it never gets a limits unit.
                    free = np.flatnonzero(solver._model_plan.is_free_rigid)
                    test.assertGreater(free.size, 0)
                    test.assertFalse(np.any(np.isin(view["side_a"][limit_units], free)))
                    # The box heaps hold free bodies only; the articulated scene's chains carry limits.
                    test.assertEqual(int(np.sum(limit_units)) > 0, name == "articulated")
                else:
                    test.assertFalse(np.any(limit_units))
                np.testing.assert_array_equal(view["side_b"][limit_units], -1)
                np.testing.assert_array_equal(view["rows"][limit_units], 0)
                test.assertTrue(np.all(view["side_a"] >= view["side_b"]))
                test.assertTrue(np.all(view["side_a"] >= 0))
                if name == "tray":
                    # The kinematic tray (body 0) is elided from the responses: a -1 side, like the ground.
                    tray = int(solver._body_to_articulation.numpy()[0])
                    test.assertTrue(np.any((view["side_b"] == -1) & (view["contact"] >= 0)))
                    test.assertNotIn(tray, np.concatenate([view["side_a"], view["side_b"]]))


def test_static_pair_makes_no_unit(test, device):
    """Make no unit for a kinematic tray resting on the ground, whose contacts have no responding side."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.rigid_gap = 0.002
    builder.add_ground_plane()
    tray = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.01), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(tray, hx=0.2, hy=0.2, hz=0.01)
    # A free box high above keeps a responding articulation in the world, far from any contact.
    box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.02, hy=0.02, hz=0.02)
    model = builder.finalize(device=device)
    model.rigid_contact_max = CONTACT_MAX
    solver = _solver(model)
    contacts = _step(model, solver)
    units = _units_for(solver, contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    np.testing.assert_array_equal(solver._contact_path.numpy()[:count], -1)
    test.assertEqual(int(units.unit_count.numpy()[0]), 0)
    test.assertEqual(int(units.unit_overflow.numpy()[0]), 0)


def test_order_is_sorted_and_deterministic(test, device):
    """Sort the units by (side_a, f, contact), identically for two solvers on the same contacts.

    ``f`` is ``0`` for a one-sided unit and ``side_b + 1`` for a two-sided one. The sorted view is
    ``unit_key[unit_order[:count]]``, strictly increasing.
    """
    for name in ("tray", "articulated"):
        with test.subTest(scene=name):
            model = _scene(name, device)
            contacts = _collide(model, _initial_state(model))
            views = []
            for _ in range(2):
                solver = _solver(model, enable_joint_limits=True)
                _step(model, solver, contacts)
                units = _units_for(solver, contacts)
                view = _sorted_view(units)
                count = view["key"].size
                if name == "tray":
                    # One-sided (box on tray or ground) and two-sided (box on box) contact units.
                    test.assertGreater(count, 100)
                    test.assertTrue(np.any((view["contact"] >= 0) & (view["side_b"] == -1)))
                    test.assertTrue(np.any(view["side_b"] >= 0))
                else:
                    # Limits units (contact -1) sort first within their side.
                    test.assertTrue(np.any(view["contact"] == -1))
                field = _sort_field(view["side_a"], view["side_b"])
                triples = np.stack([view["side_a"], field, view["contact"]])
                test.assertTrue(np.all(np.diff(np.lexsort(triples[::-1])) > 0))
                np.testing.assert_array_equal(view["key"], _packed_keys(view, units))
                test.assertTrue(np.all(np.diff(view["key"]) > 0))
                test.assertEqual(units.unit_order.shape, (units.unit_max,))
                test.assertEqual(units.unit_key.shape, (units.unit_max,))
                test.assertEqual(sorted(units.unit_order.numpy()[:count].tolist()), list(range(count)))
                views.append({**view, "order": units.unit_order.numpy()[:count].copy()})
            for field in ("side_a", "side_b", "contact", "rows", "key", "order"):
                np.testing.assert_array_equal(views[0][field], views[1][field], err_msg=field)


def test_overflow_counts_without_writing(test, device):
    """Keep the first ``scene_unit_max`` units in slot order, the same on every run, and count the rest.

    Slots come from a scan over flags, the limits units by articulation and then the contacts by
    index, so the kept units are the first ``scene_unit_max`` of that order. No slot past the
    capacity is written. The units are built on a solver without the scene mode after its step; the scene
    solve's unit overflow is ``test_scene_rows.test_unit_overflow_latches_every_world``.
    """
    for name, limits in (("heap", False), ("articulated", True)):
        with test.subTest(scene=name):
            model = _scene(name, device)
            contacts = _collide(model, _initial_state(model))
            runs = []
            for _ in range(2):
                solver = _solver(model, enable_joint_limits=limits)
                _step(model, solver, contacts)
                units = SceneUnits(_units_view(solver), 4)
                test.assertEqual(units.unit_max, 4)
                test.assertEqual(units.unit_contact.shape, (4,))
                test.assertEqual(units.unit_order.shape, (4,))
                test.assertEqual(units.unit_key.shape, (4,))
                # Oversized stand-ins expose any write past the capacity the kernels are given.
                for field in ("contact", "side_a", "side_b", "rows"):
                    setattr(units, f"unit_{field}", wp.full(16, POISON, dtype=wp.int32, device=model.device))
                units.unit_key = wp.full(16, POISON, dtype=wp.int64, device=model.device)
                units.build(solver, contacts)
                collected = _collected_units(solver, contacts, limits)
                total = collected["contact"].size
                test.assertGreater(total, 4)
                test.assertEqual(int(units.unit_count.numpy()[0]), 4)
                test.assertEqual(int(units.unit_overflow.numpy()[0]), total - 4)
                for field in ("contact", "side_a", "side_b", "rows"):
                    values = getattr(units, f"unit_{field}").numpy()
                    np.testing.assert_array_equal(values[4:], POISON, err_msg=field)
                    np.testing.assert_array_equal(values[:4], collected[field][:4], err_msg=field)
                np.testing.assert_array_equal(units.unit_key.numpy()[4:], POISON)
                if limits:
                    test.assertTrue(np.any(collected["contact"][:4] == -1))
                view = _sorted_view(units)
                np.testing.assert_array_equal(view["key"], _packed_keys(view, units))
                test.assertTrue(np.all(np.diff(view["key"]) > 0))
                runs.append({**view, "order": units.unit_order.numpy()[:4].copy()})
            for field in ("side_a", "side_b", "contact", "rows", "key", "order"):
                np.testing.assert_array_equal(runs[0][field], runs[1][field], err_msg=field)


def test_key_widths_follow_the_capacities(test, device):
    """Size the sort key's fields from the articulation count and the contact capacity; reject only keys over 63 bits."""
    model = _scene("heap", device)
    solver = _solver(model)
    units = SceneUnits(_units_view(solver))
    test.assertEqual(units.side_bits, (model.articulation_count + 1).bit_length())
    test.assertEqual(units.contact_bits, (CONTACT_MAX + 1).bit_length())
    test.assertEqual((units.side_bits, units.contact_bits), SceneUnits.key_bits(model.articulation_count, CONTACT_MAX))
    # A large scene: 65,536 humanoids and a buffer of 6,291,456 contacts, past 21-bit fields.
    test.assertEqual(SceneUnits.key_bits(65_536, 6_291_456), (17, 23))
    # The widest keys that fit: 21 + 2 x 21 = 63 bits.
    test.assertEqual(SceneUnits.key_bits(2**20, 2**20), (21, 21))
    for side_count, contact_max in ((2**20, 2**21), (2**30, 1)):
        with test.subTest(side_count=side_count, contact_max=contact_max):
            with test.assertRaisesRegex(ValueError, "63 bits"):
                SceneUnits.key_bits(side_count, contact_max)
            # The constructor checks the widths before it allocates anything.
            stub = types.SimpleNamespace(
                model=types.SimpleNamespace(device=model.device, articulation_count=side_count),
                _max_contacts_alloc=contact_max,
            )
            with test.assertRaisesRegex(ValueError, "63 bits"):
                SceneUnits(stub)


def _expected_incidence(view, side_count):
    """Each two-sided unit's sorted position under both its sides, by side and then position."""
    side_a, side_b = view["side_a"], view["side_b"]
    two_sided = ~_one_sided(side_a, side_b)
    starts = [0]
    entries = []
    for side in range(side_count):
        positions = np.flatnonzero(two_sided & ((side_a == side) | (side_b == side)))
        entries.extend(positions.tolist())
        starts.append(len(entries))
    return np.asarray(starts), np.asarray(entries, dtype=np.int64)


def test_csr_and_incidence(test, device):
    """Index the sorted units by side_a, and list each two-sided unit under both of its sides."""
    for name, limits in (("tray", True), ("articulated", True), ("scissor", False)):
        with test.subTest(scene=name):
            model = _scene(name, device)
            solver = _solver(model, enable_joint_limits=limits)
            units = _units_for(solver, _step(model, solver))
            view = _sorted_view(units)
            count = view["key"].size
            test.assertGreater(count, 0)
            side_count = units.side_count
            recount = np.bincount(view["side_a"], minlength=side_count)
            side_start = units.side_start.numpy()
            test.assertEqual(side_start.shape, (side_count + 1,))
            np.testing.assert_array_equal(side_start, np.concatenate([[0], np.cumsum(recount)]))
            for side in range(side_count):
                np.testing.assert_array_equal(view["side_a"][side_start[side] : side_start[side + 1]], side)

            starts, entries = _expected_incidence(view, side_count)
            incidence_start = units.incidence_start.numpy()
            test.assertEqual(incidence_start.shape, (side_count + 1,))
            np.testing.assert_array_equal(incidence_start, starts)
            incidence = units.incidence_unit.numpy()[: incidence_start[-1]]
            np.testing.assert_array_equal(incidence, entries)

            side_a, side_b = view["side_a"], view["side_b"]
            two_sided = np.flatnonzero(~_one_sided(side_a, side_b))
            occurrences = np.bincount(incidence, minlength=count)
            np.testing.assert_array_equal(occurrences[two_sided], 2)
            test.assertEqual(int(occurrences.sum()), 2 * two_sided.size)
            for position in two_sided:
                owners = [
                    side
                    for side in range(side_count)
                    if position in incidence[incidence_start[side] : incidence_start[side + 1]]
                ]
                test.assertEqual(owners, sorted((int(side_b[position]), int(side_a[position]))))
            if name == "tray":
                test.assertGreater(two_sided.size, 0)
            if name == "scissor":
                # Contacts between links of one articulation touch one side and stay out of the incidence.
                self_contacts = np.flatnonzero((side_a == side_b) & (view["contact"] >= 0))
                test.assertGreater(self_contacts.size, 0)
                test.assertEqual(two_sided.size, 0)
                # They are one-sided units, inside their side's one-sided prefix.
                one_sided = units.one_sided_count.numpy()
                for position in self_contacts:
                    side = int(side_a[position])
                    test.assertGreaterEqual(position, side_start[side])
                    test.assertLess(position, side_start[side] + one_sided[side])


def test_counting_sort_matches_lexsort(test, device):
    """Place every unit where ``np.lexsort`` of (side_a, f, contact) over the slots puts it.

    The keys are distinct, so the sorted order is unique and ``unit_order`` must equal the lexsort's
    permutation of the slots. The stub's side 4 holds 300 units of every kind.
    """
    for name in ("heap", "tray", "articulated", "scissor", "hub"):
        with test.subTest(scene=name):
            if name == "hub":
                side_count, pairs = _hub_pairs()
                units = _stub_units(device, side_count, pairs, color=False)
            else:
                model = _scene(name, device)
                solver = _solver(model, enable_joint_limits=name in ("articulated", "scissor"))
                units = _units_for(solver, _step(model, solver))
            slots = _slots(units)
            count = slots["contact"].size
            test.assertGreater(count, 0)
            field = _sort_field(slots["side_a"], slots["side_b"])
            expected = np.lexsort((slots["contact"], field, slots["side_a"]))
            np.testing.assert_array_equal(units.unit_order.numpy()[:count], expected)
            view = _sorted_view(units)
            test.assertTrue(np.all(np.diff(view["key"]) > 0))
            if name == "hub":
                test.assertEqual(int(np.bincount(slots["side_a"])[4]), 300)
                test.assertGreater(int(np.sum(~_one_sided(slots["side_a"], slots["side_b"]))), 100)


def test_order_is_independent_of_arrival(test, device):
    """Sort the same units identically whatever slots they occupy: the sorted view and incidence depend only on the units."""
    side_count, pairs = _hub_pairs()
    units = _stub_units(device, side_count, pairs, color=False)
    count = int(units.unit_count.numpy()[0])
    before = _sorted_view(units)
    order_before = units.unit_order.numpy()[:count].copy()
    two_sided = int(units.incidence_start.numpy()[side_count]) // 2
    test.assertGreater(two_sided, 0)
    names = ("side_start", "incidence_start", "one_sided_count", "side_coupled", "has_two_sided")
    csr_before = {name: getattr(units, name).numpy().copy() for name in names}
    incidence_before = units.incidence_unit.numpy()[: 2 * two_sided].copy()

    permutation = np.random.default_rng(5).permutation(count)
    for field in ("contact", "side_a", "side_b", "rows"):
        array = getattr(units, f"unit_{field}")
        values = array.numpy()
        values[:count] = values[:count][permutation]
        array.assign(values)
    units._sort()

    test.assertFalse(np.array_equal(units.unit_order.numpy()[:count], order_before))
    after = _sorted_view(units)
    for field in ("side_a", "side_b", "contact", "rows", "key"):
        np.testing.assert_array_equal(after[field], before[field], err_msg=field)
    for name, values in csr_before.items():
        np.testing.assert_array_equal(getattr(units, name).numpy(), values, err_msg=name)
    np.testing.assert_array_equal(units.incidence_unit.numpy()[: 2 * two_sided], incidence_before)


def test_limit_rows_make_limits_units(test, device):
    """Make one limits unit per articulation with active limit rows, carrying their count, first on its side."""
    pairs = [(0, -1), (2, 1), (1, -1), (2, -1), (0, 0), (1, 0), (2, 2)]
    solver, contacts = _stub(device, 3, pairs)
    units = SceneUnits(solver)
    test.assertFalse(units.limit_units)
    limit_rows = wp.array([3, 0, 5], dtype=wp.int32, device=solver.model.device)
    units.build(solver, contacts, limit_rows=limit_rows)
    view = _sorted_view(units)
    test.assertEqual(view["contact"].size, len(pairs) + 2)
    limits = np.flatnonzero(view["contact"] == -1)
    np.testing.assert_array_equal(view["side_a"][limits], [0, 2])
    np.testing.assert_array_equal(view["side_b"][limits], -1)
    np.testing.assert_array_equal(view["rows"][limits], [3, 5])
    # Each is first on its side, and the limits units take the first slots, by articulation.
    np.testing.assert_array_equal(limits, units.side_start.numpy()[[0, 2]])
    np.testing.assert_array_equal(units.unit_contact.numpy()[:2], -1)
    np.testing.assert_array_equal(units.unit_side_a.numpy()[:2], [0, 2])
    # The rows are read on the device at every build: new counts change the units.
    limit_rows.assign([0, 4, 0])
    units.build(solver, contacts, limit_rows=limit_rows)
    view = _sorted_view(units)
    limits = np.flatnonzero(view["contact"] == -1)
    np.testing.assert_array_equal(view["side_a"][limits], [1])
    np.testing.assert_array_equal(view["rows"][limits], [4])
    # Without limit rows the stub's rule applies: no joint limits, so no limits unit.
    units.build(solver, contacts)
    test.assertFalse(np.any(_sorted_view(units)["contact"] == -1))
    test.assertEqual(int(units.unit_count.numpy()[0]), len(pairs))


def test_one_sided_prefix_and_coupling(test, device):
    """Put each side's one-sided units first, and flag the sides that have two-sided units."""
    for name in ("heap", "scissor"):
        with test.subTest(scene=name):
            model = _scene(name, device)
            solver = _solver(model, enable_joint_limits=name == "scissor")
            units = _units_for(solver, _step(model, solver))
            side_count = units.side_count
            view = _sorted_view(units)
            side_a, side_b = view["side_a"], view["side_b"]
            one_sided = _one_sided(side_a, side_b)
            side_start = units.side_start.numpy()
            one_sided_count = units.one_sided_count.numpy()
            side_coupled = units.side_coupled.numpy()
            incidence_start = units.incidence_start.numpy()
            test.assertEqual(one_sided_count.shape, (side_count,))
            test.assertEqual(side_coupled.shape, (side_count,))
            np.testing.assert_array_equal(one_sided_count, np.bincount(side_a[one_sided], minlength=side_count))
            for side in range(side_count):
                prefix_end = side_start[side] + one_sided_count[side]
                test.assertTrue(np.all(one_sided[side_start[side] : prefix_end]), msg=f"side {side}")
                test.assertFalse(np.any(one_sided[prefix_end : side_start[side + 1]]), msg=f"side {side}")
            np.testing.assert_array_equal(side_coupled, (np.diff(incidence_start) > 0).astype(np.int32))
            test.assertEqual(int(units.has_two_sided.numpy()[0]), int(incidence_start[-1] > 0))
            if name == "heap":
                test.assertTrue(np.any(side_coupled))
                test.assertEqual(int(units.has_two_sided.numpy()[0]), 1)
            else:
                np.testing.assert_array_equal(side_coupled, 0)
                test.assertEqual(int(units.has_two_sided.numpy()[0]), 0)


def test_row_max_sizes_the_units(test, device):
    """Size the units from a row budget at construction: a contact unit needs three rows, so ``row_max // 3`` plus the limits units."""
    model = _scene("articulated", device)
    solver = _solver(model, enable_joint_limits=True)
    view = _units_view(solver)
    units = SceneUnits(view, row_max=300)
    test.assertGreater(units.limit_unit_count, 0)
    test.assertEqual(units.unit_max, 100 + units.limit_unit_count)
    # Allocated before any build, so the first build can be captured.
    test.assertFalse(units.built)
    test.assertEqual(units.unit_contact.shape, (units.unit_max,))
    test.assertEqual(units.unit_order.shape, (units.unit_max,))
    test.assertEqual(units.unit_key.shape, (units.unit_max,))
    test.assertEqual(units.incidence_unit.shape, (2 * units.unit_max,))
    # An explicit unit_max wins, and without either the first build allocates.
    test.assertEqual(SceneUnits(view, 7, row_max=300).unit_max, 7)
    test.assertIsNone(SceneUnits(view).unit_max)


def test_build_replays_under_capture(test, device):
    """Rebuild, recolor and reorder the same units from a captured graph: fixed launch shapes and no host readback.

    The build takes per-articulation limit rows, read on the device at every replay. The coloring
    sits in a conditional node on ``has_two_sided`` and its round loop in a nested one.
    """
    model = _scene("tray", device)
    solver = _solver(model, enable_joint_limits=True)
    contacts = _step(model, solver)
    units = _units_for(solver, contacts)
    side_count = units.side_count
    # Limit rows on responding articulations only, as the limits count pass would give.
    responds = solver._articulation_response_dof_count.numpy() > 0
    arts = np.arange(side_count, dtype=np.int32)
    rows = [np.where(responds, arts % 3, 0).astype(np.int32), np.where(responds, 4 * (arts % 2), 0).astype(np.int32)]
    limit_rows = wp.array(rows[0], dtype=wp.int32, device=model.device)
    names = ("unit_contact", "unit_side_a", "unit_side_b", "unit_rows", "unit_count", "unit_overflow")
    names += ("unit_order", "unit_key", "side_start", "incidence_start", "incidence_unit", "one_sided_count")
    names += ("side_coupled", "has_two_sided", "unit_color", "rounds", "uncolored", "keep_going")
    names += ("color_count_per", "color_offsets", "order_by_color", "_color_proposal")
    state_names = ("side_start", "incidence_start", "one_sided_count", "side_coupled", "has_two_sided")
    state_names += ("side_mask", "rounds", "uncolored", "color_count_per", "color_offsets")

    def snapshot():
        view = _sorted_view(units)
        count = view["key"].size
        view["color"] = units.unit_color.numpy()[units.unit_order.numpy()[:count]]
        view["order_by_color"] = units.order_by_color.numpy()[: int(units.color_offsets.numpy()[-1])].copy()
        view["incidence_unit"] = units.incidence_unit.numpy()[: units.incidence_start.numpy()[-1]].copy()
        for name in state_names:
            view[name] = getattr(units, name).numpy().copy()
        return view

    expected = []
    for values in rows:
        limit_rows.assign(values)
        units.build(solver, contacts, limit_rows=limit_rows)
        units.color()
        view = snapshot()
        np.testing.assert_array_equal(view["rows"][view["contact"] == -1], values[values > 0])
        test.assertEqual(int(view["color_offsets"][-1]), int(view["incidence_start"][-1]) // 2)
        test.assertGreater(int(view["rounds"][0]), 0)
        expected.append(view)
    with wp.ScopedCapture(model.device) as capture:
        units.build(solver, contacts, limit_rows=limit_rows)
        units.color()
    for values, eager in zip(rows, expected, strict=True):
        limit_rows.assign(values)
        for name in names:
            getattr(units, name).fill_(POISON)
        units.side_mask.fill_(0xDEADBEEF)
        wp.capture_launch(capture.graph)
        replay = snapshot()
        for name, value in eager.items():
            np.testing.assert_array_equal(replay[name], value, err_msg=name)


def test_exclusive_scan_matches_numpy(test, device):
    """Scan exclusively as NumPy's cumulative sum does, over one tile, its edges and several tiles.

    :meth:`SceneUnits._scan_exclusive` (the coloring's allocation-free scan) on a stand-in holding its tile scratch,
    for counts from 1 to five tiles of :data:`SCAN_TILE` entries: on the CPU its tile launches run one lane per block
    (Warp GH-1413), where each tile operation takes the whole tile.
    """
    rng = np.random.default_rng(7)
    for count in (1, 5, SCAN_TILE - 1, SCAN_TILE, SCAN_TILE + 1, 3000, 5 * SCAN_TILE):
        with test.subTest(count=count):
            tiles = (count + SCAN_TILE - 1) // SCAN_TILE
            holder = types.SimpleNamespace(
                device=wp.get_device(device),
                _scan_tile_totals=wp.full(tiles, POISON, dtype=wp.int32, device=device),
                _scan_tile_offsets=wp.full(tiles, POISON, dtype=wp.int32, device=device),
            )
            values = rng.integers(0, 9, size=count).astype(np.int32)
            scanned = wp.full(count, POISON, dtype=wp.int32, device=device)
            SceneUnits._scan_exclusive(holder, wp.array(values, dtype=wp.int32, device=device), scanned)
            np.testing.assert_array_equal(scanned.numpy(), np.concatenate(([0], np.cumsum(values)[:-1])))


class TestSteelSceneUnits(unittest.TestCase):
    pass


# Every kernel of the build has a CPU version; only the graph capture needs a CUDA device.
for _fn in (
    test_units_match_contacts,
    test_static_pair_makes_no_unit,
    test_order_is_sorted_and_deterministic,
    test_overflow_counts_without_writing,
    test_key_widths_follow_the_capacities,
    test_csr_and_incidence,
    test_counting_sort_matches_lexsort,
    test_order_is_independent_of_arrival,
    test_limit_rows_make_limits_units,
    test_one_sided_prefix_and_coupling,
    test_row_max_sizes_the_units,
    test_exclusive_scan_matches_numpy,
):
    add_function_test(TestSteelSceneUnits, _fn.__name__, _fn, devices=get_test_devices())
add_function_test(
    TestSteelSceneUnits,
    test_build_replays_under_capture.__name__,
    test_build_replays_under_capture,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
