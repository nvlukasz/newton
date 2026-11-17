# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Steel's contact warm start in the scene solve: the seed, the gather's coefficients and the reset.

Steel's warm start clears the whole impulse capacity, gathers each matched contact's impulses of the previous step into
its rows (``gather_contact_warmstart``, on the scene's one row family: world 0, route 0, a contact on Steel's free-body
path clamped with its coefficient before the patch link, as Steel gathers those rows before it:
:func:`scene_warmstart_gather_mu`) and replaces carried patch anchors' tangent impulses (``seed_patch_impulses``).
Steel then installs the seeds into its velocities; the scene, whose velocity is ``v_hat`` plus the decode of
``delta``, applies them to ``delta`` side by side (:func:`scene_seed_sides`) after zeroing it, so the sweeps start
from the seeds as Steel's do and the velocity pass continues from them. Steel's end-of-step snapshot keeps the rows
and each contact's first row; a reset discards a world's carry contact by contact
(:func:`scene_reset_contact_warmstart`), the rows being one world. Patch anchors on a moving mesh surface follow it as
Steel's do: the solver runs Steel's patch build (its displacement carried over the step's ``dt``) and
``finish_patch_impulses`` (the slip test relative to the surface) on the scene's route, so nothing here reads the
surface velocity.

The launchers :func:`seed`, :func:`keep_gather_mu`, :func:`finish_gather_mu` and :func:`clear_warmstart` take the
scene's solve object ``scene`` and the solver ``solver``, and read these attributes:

- ``scene``: ``device``; ``side_count`` (the sides, one per articulation); ``free_response`` (whether rows may start
  with free-body entries, the sweeps' build constant); ``side_row_begin`` and ``side_row_end`` (each side's one-sided
  rows); ``unit_row_start`` (each unit slot's first row, ``-1`` for a unit dropped past the row capacity);
  ``row_free_prefix`` (each row's free-body entries); ``gather_mu`` (``(1, row_max)`` float32, the gather's
  coefficients under friction patches with free bodies, else ``None``); ``_reset_all_selected`` (a one-entry int32
  array, the reset's every-world flag; allocated when the warm start is on).
- ``solver``: ``model.world_count``; ``_scene_units`` (``incidence_start``, ``incidence_unit``, ``unit_order``,
  ``unit_rows``); Steel's ``_articulation_world_dof_offset``, ``_articulation_response_dof_count``,
  ``_articulation_model_world``, ``_is_free_rigid``, ``_has_responding_global``; the row store ``_sparse_row_dof``,
  ``_sparse_row_factor``, ``_sparse_row_free_response``, ``_impulses``, ``_row_mu``,
  ``_sparse_factor_velocity_delta``; the contact routing ``_contact_path``, ``_contact_slot``,
  ``_contact_slots_needed``, ``_contact_art_a``, ``_contact_art_b``; the warm-start snapshot ``_ws_prev_dense_slot``,
  ``_ws_prev_contact_normal``, ``_ws_prev_impulses``, ``_ws_prev_row_type``, ``_ws_prev_row_parent``.
"""

import warp as wp

from .friction_patches import WARP_DETERMINISM_OFF
from .scene_units import _contact_unit_sides, _scene_contact_on_free_path, _unit_model_world

wp.set_module_options({"enable_backward": False, **WARP_DETERMINISM_OFF})


# --- Seed ----------------------------------------------------------------------------------------------------------


@wp.func
def _scene_entry_response(
    row_factor: wp.array3d[float], row_free_response: wp.array3d[float], prefix: int, row: int, k: int
) -> float:
    """Entry ``k``'s response to row ``row``'s impulse: its free-body response over the row's free ``prefix``, its
    factor value past it.

    The sweep's rule on a change (``value += (k < prefix ? rows_free_response : rows_factor) * change`` in the row
    loops' CUDA text, which the tripwire pins and so cannot call this function).
    """
    if k < prefix:
        return row_free_response[0, row, k]
    return row_factor[0, row, k]


@wp.func
def _scene_seed_row(
    row: int,
    first: int,
    last: int,
    free_response: int,
    rows_dof: wp.array3d[int],
    rows_factor: wp.array3d[float],
    rows_free_response: wp.array3d[float],
    rows_free_prefix: wp.array[int],
    impulses: wp.array2d[float],
    delta: wp.array2d[float],
):
    """Add row ``row``'s ``resp * impulse`` to its entries in coordinates ``[first, last)`` of ``delta``, entries in
    order; a zero impulse adds nothing (Steel's ``apply_mf_warmstart_impulses`` skips it too)."""
    impulse = impulses[0, row]
    if impulse == 0.0:
        return
    prefix = int(0)
    if free_response != 0:
        prefix = rows_free_prefix[row]
    for k in range(rows_dof.shape[2]):
        index = rows_dof[0, row, k]
        if index >= first and index < last:
            response = _scene_entry_response(rows_factor, rows_free_response, prefix, row, k)
            delta[0, index] = delta[0, index] + response * impulse


@wp.kernel(enable_backward=False, module="unique", module_options={"fuse_fp": False, **WARP_DETERMINISM_OFF})
def scene_seed_sides(
    side_count: int,
    free_response: int,
    side_row_begin: wp.array[int],
    side_row_end: wp.array[int],
    incidence_start: wp.array[int],
    incidence_unit: wp.array[int],
    unit_order: wp.array[int],
    unit_row_start: wp.array[int],
    unit_rows: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    rows_dof: wp.array3d[int],
    rows_factor: wp.array3d[float],
    rows_free_response: wp.array3d[float],
    rows_free_prefix: wp.array[int],
    impulses: wp.array2d[float],
    # in/out
    delta: wp.array2d[float],
):
    """Apply the warm start's seeded impulses to the velocity change ``delta``, side by side (grid: sides).

    The scene's counterpart of Steel's install (``apply_world_impulses_to_velocity`` and
    ``apply_mf_warmstart_impulses``, which add the seeds' velocities to ``v_out``): the scene's velocity is
    ``v_hat`` plus the decode of ``delta``, so the seeds go into ``delta`` as the sweep applies a change,
    ``delta[row_dof[k]] += resp_k * impulse`` (:func:`_scene_entry_response`), after ``SceneSolve.solve`` has zeroed it
    and before the first sweep. ``free_response`` is the sweeps' build constant (``0``: every row's prefix is 0).

    Side ``s`` owns the coordinates ``[articulation_world_dof_offset[s], + articulation_response_dof_count[s])`` and
    visits, in this order: its one-sided rows ``side_row_begin[s]`` to ``side_row_end[s]`` (every entry of a one-sided
    row is its own), then the two-sided units of its incidence list in order (``incidence_unit`` holds sorted positions,
    ``unit_order`` their slots; a unit dropped past the row capacity, ``unit_row_start < 0``, has no rows), each unit's
    ``unit_rows`` rows from its first, each row's entries in order, adding only the entries in its own coordinates: a
    two-sided row's entries are two contiguous runs, one per endpoint, each in its side's coordinates. Every entry of
    ``delta`` is thus written by the one side that owns it, in a fixed order, without atomics, and each entry of every
    row in these ranges is applied once. The seeded rows are contact and friction rows (Steel's gather seeds nothing
    else); the internal units' rows in the one-sided ranges hold the zero impulse of the row preparation and add
    nothing, and the late rows (velocity limits, spin rows) are not visited. Compiled without FMA contraction
    (``fuse_fp=False``), so each ``resp * impulse`` rounds before its sum.
    """
    s = wp.tid()
    if s >= side_count:
        return
    first = articulation_world_dof_offset[s]
    last = first + articulation_response_dof_count[s]
    for row in range(side_row_begin[s], side_row_end[s]):
        _scene_seed_row(
            row,
            first,
            last,
            free_response,
            rows_dof,
            rows_factor,
            rows_free_response,
            rows_free_prefix,
            impulses,
            delta,
        )
    for entry in range(incidence_start[s], incidence_start[s + 1]):
        slot = unit_order[incidence_unit[entry]]
        start = unit_row_start[slot]
        if start >= 0:
            for row in range(start, start + unit_rows[slot]):
                _scene_seed_row(
                    row,
                    first,
                    last,
                    free_response,
                    rows_dof,
                    rows_factor,
                    rows_free_response,
                    rows_free_prefix,
                    impulses,
                    delta,
                )


# --- Gather coefficients -------------------------------------------------------------------------------------------


@wp.kernel
def scene_warmstart_gather_mu(
    contact_count: wp.array[int],
    contact_path: wp.array[int],
    contact_slot: wp.array[int],
    contact_slots_needed: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    is_free_rigid: wp.array[int],
    row_mu: wp.array2d[float],
    # in/out
    gather_mu: wp.array2d[float],
):
    """Give the warm start's gather the friction coefficients each contact's rows have in Steel at its gather
    (grid: the contact capacity), under friction patches.

    ``gather_mu`` holds ``row_mu`` from before ``link_patch_rows`` divided the anchors' coefficients by their region's
    anchors. Steel gathers its free-body rows before that link (in ``SolverSteel._stage4_build_rows``) and its dense
    rows after it (in ``SolverSteel._solve_matrix_free``), so its transport clamps a carried tangent impulse with the
    undivided coefficient on its free-body path and the divided one on its dense path: this kernel copies the divided
    coefficients of every contact with its tangent pair on Steel's dense path
    (:func:`scene_units._scene_contact_on_free_path` false) into ``gather_mu``.
    """
    c = wp.tid()
    if c >= contact_count[0]:
        return
    slot = contact_slot[c]
    if contact_path[c] != 0 or slot < 0 or contact_slots_needed[c] != 3:
        return
    if _scene_contact_on_free_path(contact_art_a[c], contact_art_b[c], articulation_response_dof_count, is_free_rigid):
        return
    for t in range(1, 3):
        gather_mu[0, slot + t] = row_mu[0, slot + t]


# --- Reset ---------------------------------------------------------------------------------------------------------


# Steel's selection rule, which ``kernels.reset_row_warmstart`` keeps inline: Steel's kernels stay as they are, so the
# scene states the rule here, and test_scene_state.test_reset_selection_is_steels ties the two.
@wp.func
def _reset_selects_world(world_mask: wp.array[wp.bool], world: int, world_count: int, global_shares: int) -> bool:
    """Whether a reset mask selects the rows of solve world ``world``, as Steel's ``reset_row_warmstart`` decides it:
    the world's own entry, or for world 0 the global entry while a responding global articulation shares its rows."""
    selected = world_mask[world]
    if world == 0 and global_shares != 0 and world_mask[world_count]:
        selected = True
    return selected


@wp.kernel
def scene_reset_world_selection(
    world_mask: wp.array[wp.bool],
    world_count: int,
    global_shares: int,
    # in/out
    all_selected: wp.array[int],
):
    """Clear ``all_selected[0]`` (set to 1 before the launch) when a reset mask leaves a world out (grid: the model
    worlds), so that :func:`scene_reset_row_warmstart` clears the carried rows only when every world is selected.

    Each world the mask does not select (:func:`_reset_selects_world`) stores 0; every writer stores the same value,
    so the result does not depend on their order and needs no atomics.
    """
    world = wp.tid()
    if world >= world_count:
        return
    if not _reset_selects_world(world_mask, world, world_count, global_shares):
        all_selected[0] = 0


@wp.kernel
def scene_reset_contact_warmstart(
    world_mask: wp.array[wp.bool],
    world_count: int,
    global_shares: int,
    contact_path: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    art_model_world: wp.array[int],
    # in/out
    previous_slot: wp.array[int],
    previous_normal: wp.array[wp.vec3],
):
    """Discard the carried rows of the contacts a reset selects, contact by contact (grid: the contact capacity).

    The scene's rows are one solve world, so Steel's per-world ``reset_row_warmstart`` would clear every world's carry
    or none. This kernel clears instead each contact's first row of the snapshot (``previous_slot``, Steel's
    ``_ws_prev_dense_slot``: -1, which the next gather reads as no history) and its normal when the model world of its
    unit (:func:`scene_units._unit_model_world`, Steel's solve world of its rows) is selected by ``world_mask``
    (:func:`_reset_selects_world`). The sides are those of the step the snapshot holds (the solver's
    ``_contact_path`` and articulations, which the next step's build rewrites); an entry without a unit (path -1, or
    past that step's contacts) carries no row already. A ``None`` mask selects every contact.
    """
    c = wp.tid()
    if c >= previous_slot.shape[0]:
        return
    selected = True
    if world_mask:
        selected = False
        side_a, side_b = _contact_unit_sides(
            c, contact_path, contact_art_a, contact_art_b, articulation_response_dof_count
        )
        if side_a >= 0:
            world = _unit_model_world(side_a, side_b, art_model_world, world_count)
            selected = _reset_selects_world(world_mask, world, world_count, global_shares)
    if selected:
        previous_slot[c] = -1
        previous_normal[c] = wp.vec3(0.0)


@wp.kernel
def scene_reset_row_warmstart(
    all_selected: wp.array[int],
    # in/out
    prev_impulses: wp.array2d[float],
    prev_row_type: wp.array2d[int],
    prev_row_parent: wp.array2d[int],
):
    """Clear the carried rows when a reset selects every world (``all_selected[0]``), as Steel's
    ``reset_row_warmstart`` clears every world's rows (grid: the rows' shape)."""
    world, row = wp.tid()
    if all_selected[0] == 0:
        return
    prev_impulses[world, row] = 0.0
    prev_row_type[world, row] = -1
    prev_row_parent[world, row] = -1


# --- Launchers -----------------------------------------------------------------------------------------------------


def seed(scene, solver) -> None:
    """Apply the seeded impulses to ``delta``, one thread per side (:func:`scene_seed_sides`)."""
    units = solver._scene_units
    wp.launch(
        scene_seed_sides,
        dim=scene.side_count,
        inputs=[
            scene.side_count,
            int(scene.free_response),
            scene.side_row_begin,
            scene.side_row_end,
            units.incidence_start,
            units.incidence_unit,
            units.unit_order,
            scene.unit_row_start,
            units.unit_rows,
            solver._articulation_world_dof_offset,
            solver._articulation_response_dof_count,
            solver._sparse_row_dof,
            solver._sparse_row_factor,
            solver._sparse_row_free_response,
            scene.row_free_prefix,
            solver._impulses,
        ],
        outputs=[solver._sparse_factor_velocity_delta],
        device=scene.device,
    )


def keep_gather_mu(scene, solver) -> None:
    """Keep the rows' friction coefficients before the patch link for the warm start's gather (``gather_mu``)."""
    wp.copy(scene.gather_mu, solver._row_mu)


def finish_gather_mu(scene, solver, contacts) -> None:
    """After the patch link, give the contacts on Steel's dense path their divided coefficients in ``gather_mu``
    (:func:`scene_warmstart_gather_mu`)."""
    wp.launch(
        scene_warmstart_gather_mu,
        dim=contacts.rigid_contact_max,
        inputs=[
            contacts.rigid_contact_count,
            solver._contact_path,
            solver._contact_slot,
            solver._contact_slots_needed,
            solver._contact_art_a,
            solver._contact_art_b,
            solver._articulation_response_dof_count,
            solver._is_free_rigid,
            solver._row_mu,
        ],
        outputs=[scene.gather_mu],
        device=scene.device,
    )


def clear_warmstart(scene, solver, world_mask: wp.array | None) -> None:
    """Discard the warm start's carried impulses of the worlds ``world_mask`` selects (all for ``None``).

    Per contact (:func:`scene_reset_contact_warmstart`): the scene's rows are one solve world, so the carry is
    discarded contact by contact by its unit's model world, as Steel's per-world reset discards a world's rows; the
    rows themselves are cleared only when every world is selected (:func:`scene_reset_world_selection` clears a
    flag set to 1 for each world left out, :func:`scene_reset_row_warmstart` reads it). Launched on the device
    without a host read, so a reset with a live mask can be captured.
    """
    model = solver.model
    world_count = int(model.world_count)
    global_shares = int(solver._has_responding_global)
    # Every world selected unless the mask leaves one out (always with None).
    scene._reset_all_selected.fill_(1)
    if world_mask is not None:
        wp.launch(
            scene_reset_world_selection,
            dim=max(world_count, 1),
            inputs=[world_mask, world_count, global_shares],
            outputs=[scene._reset_all_selected],
            device=scene.device,
        )
    wp.launch(
        scene_reset_contact_warmstart,
        dim=solver._ws_prev_dense_slot.shape[0],
        inputs=[
            world_mask,
            world_count,
            global_shares,
            solver._contact_path,
            solver._contact_art_a,
            solver._contact_art_b,
            solver._articulation_response_dof_count,
            solver._articulation_model_world,
        ],
        outputs=[solver._ws_prev_dense_slot, solver._ws_prev_contact_normal],
        device=scene.device,
    )
    wp.launch(
        scene_reset_row_warmstart,
        dim=solver._ws_prev_impulses.shape,
        inputs=[scene._reset_all_selected],
        outputs=[solver._ws_prev_impulses, solver._ws_prev_row_type, solver._ws_prev_row_parent],
        device=scene.device,
    )
