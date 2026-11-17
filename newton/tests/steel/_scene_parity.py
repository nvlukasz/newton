# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Steel's test suite under ``solve_mode="scene"`` (the forced run). Test only.

With ``NEWTON_STEEL_FORCE_SCENE=1`` the test package (``newton/tests/steel/__init__.py``) calls :func:`install` as it
is imported. Every :class:`~newton.solvers.SolverSteel` a test then builds without an explicit ``solve_mode`` runs
``solve_mode="scene"``, and so does one built with an explicit ``solve_mode="serial"`` in a test of Steel's own modules,
whose test functions pass ``"serial"`` on as their default (a parameter their ``_colored`` twins set to ``"colored"``).
In the scene's own modules (:data:`SCENE_TEST_MODULES`) an explicit ``"serial"`` builds a Steel reference and is kept;
an explicit ``"colored"`` is kept everywhere, and so is any other explicit mode. A construction whose options the mode
rejects at construction (:data:`KEPT_ON_STEEL`) keeps its Steel mode, so that the test exercises what the mode does
not cover rather than its rejection. A test listed
in :data:`SKIPS` is skipped with ``"Steel path: "`` and its reason: it reads a Steel mechanism the mode does not have,
so under the forced run it would test nothing of the mode, or the mode's own test covers it (a scene counterpart).

With ``NEWTON_STEEL_RECORD_SOLVERS=DIR`` as well, every construction is appended to ``DIR/solvers-<pid>.jsonl`` with the
test that made it, so that a parity tool can tell a test that compares the mode with Steel from one that compares the
mode with itself, or one whose solvers all stayed on Steel.

The solver package reads no environment variable (``test_options``' ``test_solver_reads_no_environment_variables``):
the whole instrument is here, and wraps ``SolverSteel.__init__`` and ``unittest.TestCase.run`` from the outside. The
wrappers carry their state, so a second import of this module (the test package imported under two names) finds them
installed and installs nothing twice. The constructor's wrapper is one more frame between a test and the constructor,
so a constructor warning whose ``stacklevel`` names the constructor's caller names the wrapper while it is installed
(``test_options``' ``test_deprecated_spellings_map_to_the_new_options`` checks that attribution).
"""

import functools
import json
import os
import re
import unittest
from collections.abc import Callable, Mapping

from newton._src.solvers.steel._testing import current_kernel_overrides
from newton._src.solvers.steel.solver_steel import SolverSteel

STEEL_PATH_REASONS: dict[str, str] = {
    "STEEL_ROW_BUDGETS": (
        "Steel's per-world row budgets: a model world keeps the rows that fit its own budgets and flags or warns of "
        "its own loss (a global articulation's in world 0); the scene pools the budgets and keeps or drops each unit "
        "whole"
    ),
    "STEEL_ROW_FAMILIES": (
        "Steel's two row families and its contact routes (the free-body rows _mf_*, their counters and carried "
        "history, _contact_path 1); the scene keeps every row in its one row store and every contact on route 0"
    ),
    "STEEL_ROW_LAYOUT": (
        "Steel's per-world row layout (each model world's block in family order, its phase bounds and its slot maps "
        "_drive_slot, _velocity_limit_slot, _mimic_slot, _connect_slot); the scene orders its rows side-major, then "
        "by color"
    ),
    "STEEL_SOLVE_WORLDS": (
        "Steel's one solve world per model world (_world_count, _world_dof_count, _world_dof_indices); the mode folds "
        "the model worlds into one solve world (test_scene_mode's test_scene_mode_folds_model_worlds)"
    ),
    "STEEL_FACTOR_SELECTION": (
        "Steel's factor selection (dense, diagonal, fused CRBA or double-buffered factors and their per-world "
        "response _L_by_size, _J_by_size, _diag); the scene factors every topology in a sparse factor group"
    ),
    "STEEL_CONTACT_SOLVES": (
        "Steel's specialized contact solves (paired, sparse-diagonal, articulation-local), which the scene never "
        "selects"
    ),
    "STEEL_PROPAGATION": (
        "Steel's propagation responses (their tree factors and caches, rows, coloring, schedule and kernels), which "
        "the scene does not build: it solves them as the immediate response"
    ),
    "STEEL_MATRIX_FREE_SWEEP": (
        "Steel's matrix-free sweep kernels and row phases (_pgs_solve_mf_gs_kernel, _launch_mf_gs_phase), which the "
        "scene's own sweep replaces"
    ),
    "STEEL_PREELIMINATION": (
        "Steel's bilateral pre-elimination (its plan _preelim_* and direct closure solve), which the scene does not "
        "build: it runs plain mimic and connect rows"
    ),
    "STEEL_TORSION_PREPARATION": (
        "Steel's torsion preparation (host groups _torsion_stats, device preparation _device_torsion, their raises and "
        "capture rejection, the dense response and velocity post-solve they read); the scene prepares and latches its "
        "spin rows (test_scene_torsion)"
    ),
    "STEEL_GLOBAL_WORLD": (
        "Steel solves a global body in world 0's rows, so it warns about, drops and flags the global body's contacts "
        "with another world and flags world 0 with its row losses; the scene solves those contacts in its one solve "
        "world (test_scene_rows' test_global_body_couples_model_worlds)"
    ),
    "STEEL_DEFAULTS": 'Steel\'s default solve_mode ("serial"), which the forced run turns to solve_mode="scene"',
    "STEEL_WARMSTART_HISTORY": (
        "Steel's per-world warm-start history (_ws_prev_impulses, _ws_prev_mf_impulses, discarded world by world); "
        "the scene discards a world's carry contact by contact (test_scene_state's "
        "test_warmstart_resets_by_world_mask)"
    ),
}
"""The Steel mechanisms a listed test reads, by key: one text per mechanism, so that the forced run's skips group by
mechanism. The texts name the scene test that covers the mechanism under the mode, where one does."""

KEPT_ON_STEEL: dict[str, Callable[[Mapping[str, object], Mapping[str, object]], bool]] = {
    "nesterov_momentum=True": lambda options, overrides: bool(options.get("nesterov_momentum", False)),
    "reuse_kinematics=True": lambda options, overrides: bool(options.get("reuse_kinematics", False)),
    "kernel override sparse_mass_matrix=False": lambda options, overrides: (
        not overrides.get("sparse_mass_matrix", True)
    ),
}
"""The constructions the mode rejects, by the option the rejection names: each predicate takes the constructor's
keyword arguments and the active kernel overrides (``_testing.kernel_overrides``). A forced construction that matches
one keeps Steel's default mode. The mode's other rejections depend on the model (a topology its factor groups do not
plan) or come after construction (coupling, refused by the coupling hooks); a test that meets one fails with the
mode's ``solve_mode='scene' does not support`` message, which a parity tool files as a rejected feature."""

SKIPS: dict[str, str] = {
    # test_articulation
    "TestSteelConnect::test_closure_row_overflow_is_reported": "STEEL_ROW_BUDGETS",
    "TestSteelMimic::test_mimic_row_overflow_is_reported": "STEEL_ROW_BUDGETS",
    # A recorded exception: a behavioural test the mode does not hold.
    "TestSteelPreelimination::test_closure_held_at_low_iterations": "STEEL_PREELIMINATION",
    "TestSteelPreelimination::test_default_preserves_all_bilateral_trajectory": "STEEL_PREELIMINATION",
    # A recorded exception: a behavioural test the mode does not hold.
    "TestSteelPreelimination::test_mimic_held_at_low_iterations": "STEEL_PREELIMINATION",
    "TestSteelPreelimination::test_mimics_can_remain_iterative_when_total_rows_exceed_capacity": "STEEL_PREELIMINATION",
    # A recorded exception: a behavioural test the mode does not hold.
    "TestSteelPreelimination::test_preelimination_capture_matches_eager": "STEEL_PREELIMINATION",
    "TestSteelPreelimination::test_preelimination_with_free_body_in_world": "STEEL_PREELIMINATION",
    "TestSteelPreelimination::test_prescribed_parent_closure_warns_and_falls_back": "STEEL_PREELIMINATION",
    "TestSteelPreelimination::test_warmstart_with_preelimination_holds_a_loaded_closure": "STEEL_PREELIMINATION",
    "TestSteelPreeliminationPropagation::test_propagation_keeps_iterative_mimic_rows": "STEEL_PREELIMINATION",
    # test_capacity
    # Scene counterpart: test_scene_features::test_scene_capacity_failure_is_observable
    # A recorded exception: a behavioural test the mode does not hold.
    "TestSteelCapacityStatus::test_capacity_failure_is_observable": "STEEL_ROW_BUDGETS",
    # Scene counterpart: test_scene_features::test_scene_capacity_failure_is_observable
    "TestSteelCapacityStatus::test_dense_capacity_failure_is_world_local": "STEEL_ROW_BUDGETS",
    # Scene counterpart: test_scene_features::test_scene_overflow_warning_is_printed_once
    # A recorded exception: a behavioural test the mode does not hold.
    "TestSteelCapacityStatus::test_overflow_warning_is_printed_once": "STEEL_ROW_BUDGETS",
    # Scene counterpart: test_scene_features::test_scene_overflow_warning_is_printed_once
    # A recorded exception: a behavioural test the mode does not hold.
    "TestSteelCapacityStatus::test_overflow_warning_is_printed_once_propagation": "STEEL_ROW_BUDGETS",
    # Scene counterpart: test_scene_groups::test_heterogeneous_worlds_match_isolated_worlds_bitwise
    "TestSteelHeteroGuard::test_hetero_serial_constructs": "STEEL_SOLVE_WORLDS",
    # Scene counterpart: test_scene_groups::test_heterogeneous_worlds_match_isolated_worlds_bitwise
    "TestSteelHeteroGuard::test_homogeneous_propagation_constructs": "STEEL_SOLVE_WORLDS",
    # Scene counterpart: test_scene_groups::test_heterogeneous_worlds_match_isolated_worlds_bitwise
    "TestSteelHeteroGuard::test_homogeneous_serial_constructs": "STEEL_SOLVE_WORLDS",
    # Scene counterpart: test_scene_features::test_scene_watermark_excess_counts_dropped_rows
    "TestSteelRowWatermark::test_watermarks_record_capacity_overflow": "STEEL_ROW_BUDGETS",
    "TestSteelRowWatermark::test_watermarks_track_propagation_rows": "STEEL_PROPAGATION",
    # Scene counterpart: test_scene_features::test_row_watermark_counts_rows_per_model_world
    "TestSteelRowWatermark::test_watermarks_track_the_largest_row_counts": "STEEL_ROW_FAMILIES",
    # test_compliance
    "TestContactComplianceIntegration::test_native_hydro_free_body_manifold": "STEEL_ROW_FAMILIES",
    # test_contacts
    "TestSteelContactControls::test_overflow_warning_enabled_after_construction_latches": "STEEL_ROW_BUDGETS",
    # A recorded exception: a behavioural test the mode does not hold.
    "TestSteelContactObservable::test_contact_f_reports_each_box_weight_row_overflow": "STEEL_ROW_BUDGETS",
    "TestSteelPairedResponse::test_paired_response_selection": "STEEL_CONTACT_SOLVES",
    "TestSteelPersistentContacts::test_cached_friction_obeys_changed_material": "STEEL_ROW_FAMILIES",
    "TestSteelPersistentContacts::test_cached_friction_preserves_world_direction": "STEEL_ROW_FAMILIES",
    "TestSteelPersistentContacts::test_cached_friction_uses_solver_normal_convention": "STEEL_ROW_FAMILIES",
    "TestSteelPersistentContacts::test_combined_history_reset_replays_with_changed_world_mask": (
        "STEEL_WARMSTART_HISTORY"
    ),
    "TestSteelPersistentContacts::test_rotating_contact_clamps_the_projected_friction_cone": "STEEL_ROW_FAMILIES",
    # Scene counterpart: test_scene_features::test_restitution_rows_fire_where_steel_fires
    "TestSteelRestitution::test_restitution_uses_ordinary_rows_and_validates_configuration": "STEEL_ROW_LAYOUT",
    # Scene counterpart: test_scene_velocity_iterations::test_velocity_iterations_rebound_tracks_steel
    "TestSteelVelocityPassSpeculative::test_every_contact_response_route_lands": "STEEL_ROW_FAMILIES",
    # test_determinism
    # Scene counterpart: test_scene_rows::test_row_order_is_side_major_then_colors
    "TestSteelDeterminism::test_deterministic_bilateral_rows_follow_element_order": "STEEL_ROW_LAYOUT",
    # Scene counterpart: test_scene_internal_unit::test_joint_friction_bounds_match_steel_bitwise
    "TestSteelDeterminism::test_deterministic_joint_friction_rows_follow_dof_order": "STEEL_ROW_LAYOUT",
    # Scene counterpart: test_scene_rows::test_row_order_is_side_major_then_colors
    "TestSteelDeterminism::test_deterministic_joint_rows_follow_articulation_order": "STEEL_ROW_LAYOUT",
    # Scene counterpart: test_scene_rows::test_rows_overflow_in_scan_order
    "TestSteelDeterminism::test_deterministic_overflow_keeps_earliest_contacts": "STEEL_ROW_BUDGETS",
    # Scene counterpart: test_scene_features::test_scene_overflow_warning_is_printed_once
    "TestSteelDeterminism::test_overflow_warning_names_deterministic_mode": "STEEL_ROW_BUDGETS",
    "TestSteelPropagationDeterminism::test_deterministic_propagation_runs_are_bitwise_identical": "STEEL_PROPAGATION",
    # test_friction
    "TestSteelFrictionAnchorSelection::test_contact_churn_reports_linear_force_but_not_a_wrench": "STEEL_ROW_FAMILIES",
    # Scene counterpart: test_scene_state::test_patch_rows_follow_the_anchors
    "TestSteelFrictionAnchors::test_row_builders_store_anchor_separation_per_route": "STEEL_ROW_FAMILIES",
    # Scene counterpart: test_scene_friction_modes::test_friction_modes_track_steel
    "TestSteelFrictionMode::test_modes_select_distinct_kernels": "STEEL_MATRIX_FREE_SWEEP",
    # Scene counterpart: test_scene_friction_modes::test_friction_modes_track_steel
    "TestSteelFrictionMode::test_sparse_factors_require_the_current_mode": "STEEL_FACTOR_SELECTION",
    # test_kernels
    "TestSteelFusedCrba::test_fused_assembly_follows_the_mass_update_interval": "STEEL_FACTOR_SELECTION",
    "TestSteelFusedCrba::test_fused_assembly_is_selected_only_where_supported": "STEEL_FACTOR_SELECTION",
    "TestSteelFusedCrba::test_fused_assembly_matches_the_separate_pipeline": "STEEL_FACTOR_SELECTION",
    "TestSteelLaunchConfig::test_articulated_contact_response_validation": "STEEL_PROPAGATION",
    "TestSteelLaunchConfig::test_compact_world_dof_mapping_pads_heterogeneous_worlds": "STEEL_SOLVE_WORLDS",
    "TestSteelLaunchConfig::test_diagonal_fusion_requires_nonaliased_world_response": "STEEL_FACTOR_SELECTION",
    "TestSteelLaunchConfig::test_propagation_rejects_unsupported_options": "STEEL_PROPAGATION",
    "TestSteelStreams::test_capture_after_eager_orders_the_pending_clear": "STEEL_FACTOR_SELECTION",
    "TestSteelStreams::test_double_buffer_capture_sequences_match_single_buffer": "STEEL_FACTOR_SELECTION",
    "TestSteelStreams::test_reset_after_an_eager_error_steps_like_a_fresh_solver": "STEEL_FACTOR_SELECTION",
    "TestSteelStreams::test_streams_and_double_buffer_step_identically": "STEEL_FACTOR_SELECTION",
    # test_limits_drives
    "TestSteelJointFriction::test_dense_row_overflow_is_reported": "STEEL_ROW_BUDGETS",
    "TestSteelJointFriction::test_local_internal_fast_path_is_not_selected": "STEEL_CONTACT_SOLVES",
    # Scene counterpart: test_scene_velocity_limits::test_late_rows_follow_the_coupled_pass
    "TestSteelJointLimitActivationGap::test_combined_row_families_follow_the_documented_layout": "STEEL_ROW_LAYOUT",
    "TestSteelJointLimitActivationGap::test_default_finite_limits_keep_rows_and_warn_at_capacity": "STEEL_ROW_BUDGETS",
    "TestSteelJointLimitActivationGap::test_dense_row_families_respect_phase_bounds": "STEEL_ROW_LAYOUT",
    "TestSteelJointLimitActivationGap::test_propagation_solves_every_row_family": "STEEL_PROPAGATION",
    # Scene counterpart: test_scene_velocity_limits::test_free_body_velocity_limit_rows_match_steel_rows
    "TestSteelJointLimitActivationGap::test_velocity_limit_rows_require_the_option": "STEEL_ROW_FAMILIES",
    "TestSteelPGSDrives::test_drive_rows_precede_limit_rows_and_count_against_capacity": "STEEL_ROW_BUDGETS",
    # Scene counterpart: test_scene_drives::test_drive_rows_track_steel
    "TestSteelPGSDrives::test_undriven_dofs_get_no_drive_rows": "STEEL_ROW_LAYOUT",
    # test_mass_matrix
    "TestSteelDiagonalMass::test_diagonal_mass_rows_match_dense_and_sparse_factors": "STEEL_FACTOR_SELECTION",
    "TestSteelDiagonalMass::test_fixed_base_sibling_branches_detect_diagonal_mass_topology": "STEEL_FACTOR_SELECTION",
    "TestSteelDiagonalMass::test_fixed_base_sibling_branches_use_diagonal_mass_path": "STEEL_FACTOR_SELECTION",
    "TestSteelNotifyDevice::test_captured_world_mask_follows_the_live_mask": "STEEL_WARMSTART_HISTORY",
    "TestSteelNotifyDevice::test_world_mask_limits_warmstart_invalidation": "STEEL_WARMSTART_HISTORY",
    # test_options
    "TestSteelGlobalWorld::test_construction_warns_about_unsolvable_global_contacts": "STEEL_GLOBAL_WORLD",
    "TestSteelGlobalWorld::test_construction_warns_about_unsolvable_global_contacts_propagation": "STEEL_GLOBAL_WORLD",
    "TestSteelGlobalWorld::test_dynamic_global_body_flags_other_world_contacts": "STEEL_GLOBAL_WORLD",
    "TestSteelGlobalWorld::test_dynamic_global_body_flags_other_world_contacts_propagation": "STEEL_GLOBAL_WORLD",
    "TestSteelGlobalWorld::test_global_overflow_has_its_own_reset_slot": "STEEL_GLOBAL_WORLD",
    "TestSteelGlobalWorld::test_global_overflow_has_its_own_reset_slot_propagation": "STEEL_GLOBAL_WORLD",
    "TestSteelGlobalWorld::test_global_slot_reset_with_few_articulations": "STEEL_GLOBAL_WORLD",
    "TestSteelGlobalWorld::test_global_slot_reset_with_few_articulations_propagation": "STEEL_GLOBAL_WORLD",
    "TestSteelGlobalWorld::test_jointed_global_kinematic_flags_other_world_contacts": "STEEL_GLOBAL_WORLD",
    "TestSteelGlobalWorld::test_jointed_global_kinematic_flags_other_world_contacts_propagation": "STEEL_GLOBAL_WORLD",
    # test_propagation
    "TestPropagationSameArticulation::test_cross_response_effective_mass_matches_reference_operator": (
        "STEEL_PROPAGATION"
    ),
    "TestPropagationSameArticulation::test_default_routing_keeps_same_articulation_rows_dense": "STEEL_PROPAGATION",
    # Scene counterpart: test_scene_rows::test_rows_match_steel_rows_bitwise
    "TestPropagationSameArticulation::test_scissor_dense_diagonal_matches_reference": "STEEL_FACTOR_SELECTION",
    "TestSteelColoredPipeline::test_straight_line_units_match_the_row_loop": "STEEL_PROPAGATION",
    "TestSteelColoredPipeline::test_sweep_variants_match_the_reference_bitwise": "STEEL_PROPAGATION",
    "TestSteelColoredScheduling::test_colored_accepts_a_large_row_budget": "STEEL_PROPAGATION",
    "TestSteelColoredScheduling::test_coloring_is_deterministic_for_a_fixed_contact_set": "STEEL_PROPAGATION",
    "TestSteelColoredScheduling::test_dense_heap_colors_without_a_serial_tail": "STEEL_PROPAGATION",
    "TestSteelColoredScheduling::test_kinematic_free_body_is_recorded_as_static": "STEEL_PROPAGATION",
    "TestSteelColoredScheduling::test_kinematic_hub_does_not_serialize_its_contacts": "STEEL_PROPAGATION",
    "TestSteelColoredScheduling::test_prebuild_matches_the_first_fit_reference": "STEEL_PROPAGATION",
    "TestSteelColoredUnitCapacity::test_mixed_units_graph_matches_eager": "STEEL_PROPAGATION",
    "TestSteelColoredUnitCapacity::test_mixed_units_match_serial_propagation": "STEEL_PROPAGATION",
    "TestSteelPropagation::test_contact_effective_mass_matches_dense_response": "STEEL_PROPAGATION",
    "TestSteelPropagation::test_native_tree_kernels_match_generic": "STEEL_PROPAGATION",
    "TestSteelPropagation::test_phased_sweeps_touch_only_their_row_family": "STEEL_MATRIX_FREE_SWEEP",
    "TestSteelPropagation::test_propagation_row_overflow_is_flagged_and_reset": "STEEL_PROPAGATION",
    "TestSteelPropagationCachedResponse::test_cached_matrices_match_numpy_reference": "STEEL_PROPAGATION",
    "TestSteelPropagationCachedResponse::test_cached_path_detected": "STEEL_PROPAGATION",
    "TestSteelPropagationCachedResponse::test_clutter_does_not_evict_cache": "STEEL_PROPAGATION",
    "TestSteelPropagationCachedResponse::test_fixed_base_cached_matrices_match_numpy_reference": "STEEL_PROPAGATION",
    "TestSteelPropagationCachedResponse::test_impulses_consumed": "STEEL_PROPAGATION",
    "TestSteelPropagationCachedResponse::test_lockstep_colored": "STEEL_PROPAGATION",
    "TestSteelPropagationCachedResponse::test_lockstep_fixed_base_single_dof": "STEEL_PROPAGATION",
    "TestSteelPropagationCachedResponse::test_lockstep_multi_contact_bodies": "STEEL_PROPAGATION",
    "TestSteelPropagationCachedResponse::test_lockstep_second_size_group": "STEEL_PROPAGATION",
    "TestSteelPropagationCachedResponse::test_lockstep_single_contact": "STEEL_PROPAGATION",
    "TestSteelPropagationCachedResponse::test_overflow_falls_back_to_tree_walk": "STEEL_PROPAGATION",
    "TestSteelPropagationCombinations::test_regularization_weights_reach_the_propagation_rows": "STEEL_PROPAGATION",
    "TestSteelPropagationCombinations::test_warm_start_seeds_the_propagation_rows": "STEEL_PROPAGATION",
    "TestSteelPropagationCoupledContacts::test_graph_replay_matches_eager_steps": "STEEL_PROPAGATION",
    "TestSteelPropagationCoupledContacts::test_shared_dof_feet_split_their_whole_response": "STEEL_PROPAGATION",
    "TestSteelPropagationCoupledContacts::test_shared_root_split_matches_across_kernels": "STEEL_PROPAGATION",
    "TestSteelSchedule::test_contact_then_internal_gives_internal_rows_the_last_word": "STEEL_MATRIX_FREE_SWEEP",
    "TestSteelSchedule::test_row_phases_touch_only_their_row_families": "STEEL_MATRIX_FREE_SWEEP",
    # test_scene_mode
    "TestSteelSceneMode::test_scene_mode_is_off_by_default": "STEEL_DEFAULTS",
    # test_sparse
    "TestSteelSparseDiagonal::test_sparse_diagonal_contact_compliance_matches_general_owner": "STEEL_CONTACT_SOLVES",
    "TestSteelSparseDiagonal::test_sparse_diagonal_contact_friction_matches_general_owner": "STEEL_CONTACT_SOLVES",
    "TestSteelSparseDiagonal::test_sparse_diagonal_is_selected_only_where_supported": "STEEL_CONTACT_SOLVES",
    "TestSteelSparseDiagonal::test_sparse_diagonal_response_matches_dense_joint_limits": "STEEL_CONTACT_SOLVES",
    # Scene counterpart: test_scene_aliases::test_mimic_rows_track_steel
    "TestSteelSparseSelectionGuard::test_bilateral_rows_keep_dense_factors": "STEEL_FACTOR_SELECTION",
    # Scene counterpart: test_scene_groups::test_one_topology_scene_has_one_factor_group
    "TestSteelSparseSolver::test_default_selects_sparse_factors_for_branched_articulations": "STEEL_FACTOR_SELECTION",
    # Scene counterpart: test_scene_groups::test_heterogeneous_scene_tracks_steel
    "TestSteelSparseSolver::test_heterogeneous_worlds_do_not_invent_joint_support": "STEEL_FACTOR_SELECTION",
    # Scene counterpart: test_scene_groups::test_heterogeneous_scene_tracks_steel
    "TestSteelSparseSolver::test_mixed_contacts_match_existing_solver": "STEEL_FACTOR_SELECTION",
    # Scene counterpart: test_scene_groups::test_heterogeneous_scene_tracks_steel
    "TestSteelSparseSolver::test_moving_prescribed_support_and_reset": "STEEL_FACTOR_SELECTION",
    # Scene counterpart: test_scene_groups::test_heterogeneous_scene_tracks_steel
    "TestSteelSparseSolver::test_two_articulations_share_contact_solve": "STEEL_FACTOR_SELECTION",
    "TestSteelSparseSolver::test_unsupported_configurations_keep_existing_path": "STEEL_FACTOR_SELECTION",
    # test_torsion
    "TestContactTorsion::test_capture_is_explicitly_rejected": "STEEL_TORSION_PREPARATION",
    # Scene counterpart: test_scene_torsion::test_torsion_errors_are_latched
    "TestContactTorsion::test_hydro_combination_is_rejected": "STEEL_TORSION_PREPARATION",
    # Scene counterpart: test_scene_torsion::test_spin_rows_match_steel_rows
    "TestContactTorsion::test_joint_response_and_coupled_budget": "STEEL_TORSION_PREPARATION",
    "TestContactTorsion::test_timestep_and_capacity": "STEEL_TORSION_PREPARATION",
    # Scene counterpart: test_scene_torsion::test_torsion_errors_are_latched
    "TestContactTorsionRegressions::test_contact_input_overflow_is_rejected": "STEEL_TORSION_PREPARATION",
    # Scene counterpart: test_scene_torsion::test_torsion_errors_are_latched
    "TestContactTorsionRegressions::test_contact_row_loss_fails_without_diagnostics": "STEEL_TORSION_PREPARATION",
    # Scene counterpart: test_scene_torsion::test_torsion_errors_are_latched
    "TestContactTorsionRegressions::test_hydro_contact_input_is_rejected": "STEEL_TORSION_PREPARATION",
    "TestContactTorsionRegressions::test_patch_spanning_shape_pairs_takes_the_largest_radius": (
        "STEEL_TORSION_PREPARATION"
    ),
    # Scene counterpart: test_scene_torsion::test_one_sided_and_free_body_spin_rows_track_steel
    "TestContactTorsionRegressions::test_torsion_keeps_free_body_velocity_limits": "STEEL_ROW_FAMILIES",
    # Scene counterpart: test_scene_torsion::test_spin_rows_match_steel_rows
    "TestContactTorsionRegressions::test_torsion_uses_cfm_floor": "STEEL_TORSION_PREPARATION",
    "TestDeviceTorsionPreparation::test_cuda_cold_capture_reports_errors_until_reset": "STEEL_TORSION_PREPARATION",
    "TestDeviceTorsionPreparation::test_cuda_solver_profiles": "STEEL_TORSION_PREPARATION",
    # Scene counterpart: test_scene_torsion::test_torsion_errors_are_latched
    "TestDeviceTorsionPreparation::test_public_api_noop_and_fail_stop": "STEEL_TORSION_PREPARATION",
    # Scene counterpart: test_scene_torsion::test_spin_rows_match_steel_rows
    "TestFinalPatchBudget::test_final_budget_and_velocity_response": "STEEL_TORSION_PREPARATION",
    # Scene counterpart: test_scene_torsion::test_spin_rows_match_steel_rows
    "TestTorsionCombined::test_analytic_normal_load_position_soft_velocity_rigid": "STEEL_TORSION_PREPARATION",
    # Scene counterpart: test_scene_torsion::test_spin_rows_match_steel_rows
    "TestTorsionCombined::test_distributed_budget_and_export": "STEEL_TORSION_PREPARATION",
    # Scene counterpart: test_scene_torsion::test_retired_spin_row_refunds_under_velocity_iterations
    "TestTorsionCombined::test_no_load_near_positive_gap_and_release": "STEEL_TORSION_PREPARATION",
    # Scene counterpart: test_scene_torsion::test_spin_rows_match_steel_rows
    "TestTorsionRegularization::test_analytic_regularized_load_and_saturated_spin": "STEEL_TORSION_PREPARATION",
    # Scene counterpart: test_scene_torsion::test_spin_rows_match_steel_rows
    "TestTorsionRegularization::test_final_load_budget_across_iterations": "STEEL_TORSION_PREPARATION",
    # Scene counterpart: test_scene_torsion::test_retired_spin_row_refunds_under_velocity_iterations
    "TestTorsionVelocitySolve::test_carried_spin_is_refunded_when_postpass_retires_group": "STEEL_TORSION_PREPARATION",
    # Scene counterpart: test_scene_torsion::test_spin_rows_match_steel_rows
    "TestTorsionVelocitySolve::test_postpass_accumulated_impulse_response_and_position_split": (
        "STEEL_TORSION_PREPARATION"
    ),
}
"""The tests the forced run skips: ``"<Class>::<test>"`` (the test's method name without a device suffix, so that the
key covers its ``_cpu`` and ``_cuda_0`` variants) -> a key of :data:`STEEL_PATH_REASONS`, or the reason itself:
``"scene counterpart: <test>"`` (the mode's own test that replaces it), ``"rejected: <feature>"`` (a feature the mode
rejects) or ``"exception: <name>"`` (a documented divergence of the mode). A ``# Scene counterpart:`` comment above a
reason key names the scene test that covers the mechanism; a recorded exception is a reason-key entry whose test holds
a behaviour the mode does not."""

_DEVICE_SUFFIX = re.compile(r"_(cpu|cuda_\d+)$")

SCENE_TEST_MODULES: frozenset[str] = frozenset(
    name.removesuffix(".py")
    for name in os.listdir(os.path.dirname(os.path.abspath(__file__)))
    if name.startswith("test_scene_") and name.endswith(".py")
) | {"test_coloring"}
"""The scene's own test modules, by basename: every ``test_scene_*.py`` of this package and ``test_coloring.py`` (the
scene units' coloring), the modules that test the mode alone. Their tests compare the mode with Steel by design, so
their explicit ``solve_mode="serial"`` constructions are Steel references, which the forced run keeps; in every other
module of the package (Steel's own tests) an explicit ``"serial"`` is forced to the mode."""


def kept_on_steel(options: Mapping[str, object], overrides: Mapping[str, object]) -> str | None:
    """The :data:`KEPT_ON_STEEL` options that ``options`` and ``overrides`` set, comma-separated, or ``None``."""
    kept = [name for name, rejected in KEPT_ON_STEEL.items() if rejected(options, overrides)]
    return ", ".join(kept) if kept else None


def skip_reason(class_name: str, method_name: str) -> str | None:
    """The forced run's skip reason of a test (``"Steel path: "`` and its :data:`SKIPS` reason), or ``None``.

    The test's key is ``"<class_name>::<method_name>"``, else the same without the method's device suffix.
    """
    base = _DEVICE_SUFFIX.sub("", method_name)
    for key in (f"{class_name}::{method_name}", f"{class_name}::{base}"):
        entry = SKIPS.get(key)
        if entry is not None:
            return "Steel path: " + STEEL_PATH_REASONS.get(entry, entry)
    return None


def unwrapped(function: Callable) -> Callable:
    """``function`` without this module's wrappers (the function they wrapped, however many times)."""
    while hasattr(function, "_scene_parity_state"):
        function = function.__wrapped__
    return function


def _state() -> dict:
    """The installed wrappers' shared state, installing both wrappers first if they are not.

    ``forced``: whether forced constructions and skips are on; ``directory``: the recorder's directory or ``None``;
    ``test``: the running test's ``"<Class>::<method>"`` and ``module`` the basename of its class's module (both
    ``None`` outside a test).
    """
    state = getattr(SolverSteel.__init__, "_scene_parity_state", None)
    if state is not None:
        return state
    state = {"forced": False, "directory": None, "test": None, "module": None}
    init = SolverSteel.__init__
    run = unittest.TestCase.run

    @functools.wraps(init)
    def init_under_parity(self, model, *args, **kwargs):
        options = dict(kwargs)
        overrides = current_kernel_overrides()
        kept = forced_from = None
        requested = kwargs.get("solve_mode")
        module = state["module"]
        # No mode, or an explicit "serial" in a test of Steel's own modules (see SCENE_TEST_MODULES).
        forceable = "solve_mode" not in kwargs or (
            requested == "serial" and module is not None and module not in SCENE_TEST_MODULES
        )
        if state["forced"] and forceable:
            kept = kept_on_steel(options, overrides)
            if kept is None:
                kwargs["solve_mode"] = "scene"
                forced_from = requested
        try:
            init(self, model, *args, **kwargs)
        finally:
            if state["directory"] is not None:
                mode = getattr(self, "_solve_mode", kwargs.get("solve_mode", init.__kwdefaults__["solve_mode"]))
                record = {
                    "test": state["test"],
                    "requested": options.get("solve_mode"),
                    "solve_mode": mode,
                    "effective": mode == "scene",
                    "kept": kept,
                    "forced_from": forced_from,
                    "options": options,
                    "overrides": sorted(overrides),
                }
                line = json.dumps(record, sort_keys=True, default=repr)
                # The process id at write time: the test runner's workers import the package after it starts.
                path = os.path.join(state["directory"], f"solvers-{os.getpid()}.jsonl")
                with open(path, "a", encoding="utf-8") as stream:
                    stream.write(line + "\n")

    @functools.wraps(run)
    def run_under_parity(self, result=None):
        previous = state["test"], state["module"]
        state["test"] = f"{type(self).__name__}::{self._testMethodName}"
        state["module"] = type(self).__module__.rsplit(".", 1)[-1]
        reason = skip_reason(type(self).__name__, self._testMethodName) if state["forced"] else None
        if reason is not None:
            # unittest skips a test whose method carries the skip attributes, before its setUp; an instance
            # attribute shadows the method for this run only.
            def skipped():
                pass

            skipped.__unittest_skip__ = True
            skipped.__unittest_skip_why__ = reason
            setattr(self, self._testMethodName, skipped)
        try:
            return run(self, result)
        finally:
            if reason is not None:
                delattr(self, self._testMethodName)
            state["test"], state["module"] = previous

    init_under_parity._scene_parity_state = state
    run_under_parity._scene_parity_state = state
    SolverSteel.__init__ = init_under_parity
    unittest.TestCase.run = run_under_parity
    return state


def install_forced_scene() -> None:
    """Force ``solve_mode="scene"`` and skip the :data:`SKIPS` tests, from now on in this process.

    The mode is forced on every ``SolverSteel`` built without an explicit ``solve_mode``, and on every one built with
    an explicit ``"serial"`` outside :data:`SCENE_TEST_MODULES` (see the module docstring).
    """
    _state()["forced"] = True


def install_recorder(directory: str) -> None:
    """Append every ``SolverSteel`` construction from now on to ``directory/solvers-<pid>.jsonl``, one JSON line each.

    A record holds ``test`` (the running test's ``"<Class>::<method>"``, ``None`` outside a test), ``requested`` (the
    ``solve_mode`` argument or ``None``), ``solve_mode`` (the mode the solver resolved, or the one it was given when
    its construction raised first), ``effective`` (whether that is ``"scene"``), ``kept`` (the :data:`KEPT_ON_STEEL`
    options that kept a forced construction on Steel, or ``None``), ``forced_from`` (``"serial"`` for an explicit
    ``"serial"`` the forced run built with the mode, else ``None``), ``options`` (the keyword arguments) and
    ``overrides`` (the active kernel override keys). A construction that raises is recorded too.
    """
    os.makedirs(directory, exist_ok=True)
    _state()["directory"] = directory


def install() -> None:
    """Install what the environment asks for: ``NEWTON_STEEL_FORCE_SCENE=1`` the forced run
    (:func:`install_forced_scene`), ``NEWTON_STEEL_RECORD_SOLVERS=DIR`` the recorder (:func:`install_recorder`)."""
    if os.environ.get("NEWTON_STEEL_FORCE_SCENE") == "1":
        install_forced_scene()
    directory = os.environ.get("NEWTON_STEEL_RECORD_SOLVERS")
    if directory:
        install_recorder(directory)
