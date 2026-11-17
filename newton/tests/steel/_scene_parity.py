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
    "TestSteelConnectModelEdits::test_captured_device_release": (
        "scene counterpart: test_captured_device_release_under_scene"
    ),
    "TestSteelConnectModelEdits::test_in_place_release": "scene counterpart: test_in_place_release_under_scene",
    "TestSteelConnectModelEdits::test_in_place_release_with_preelimination": (
        "scene counterpart: test_in_place_release_with_preelimination_under_scene"
    ),
    "TestSteelMimic::test_mimic_row_overflow_is_reported": "STEEL_ROW_BUDGETS",
    "TestSteelMixedWorld::test_mixed_world_stack_rests": "scene counterpart: test_mixed_world_stack_rests_under_scene",
    "TestSteelPreelimination::test_closure_held_at_low_iterations": (
        "exception: closures held at a few iterations, which only Steel's direct closure solve achieves"
    ),
    "TestSteelPreelimination::test_default_preserves_all_bilateral_trajectory": "STEEL_PREELIMINATION",
    "TestSteelPreelimination::test_mimic_held_at_low_iterations": (
        "exception: closures held at a few iterations, which only Steel's direct closure solve achieves"
    ),
    "TestSteelPreelimination::test_mimics_can_remain_iterative_when_total_rows_exceed_capacity": "STEEL_PREELIMINATION",
    "TestSteelPreelimination::test_preelimination_capture_matches_eager": (
        "exception: closures held at a few iterations, which only Steel's direct closure solve achieves"
    ),
    "TestSteelPreelimination::test_preelimination_with_free_body_in_world": "STEEL_PREELIMINATION",
    "TestSteelPreelimination::test_prescribed_parent_closure_warns_and_falls_back": "STEEL_PREELIMINATION",
    "TestSteelPreelimination::test_warmstart_with_preelimination_holds_a_loaded_closure": "STEEL_PREELIMINATION",
    "TestSteelPreeliminationPropagation::test_propagation_keeps_iterative_mimic_rows": "STEEL_PREELIMINATION",
    "TestSteelPrescribedParentConnect::test_disabled_loop_joint_starts_released": (
        "scene counterpart: test_disabled_loop_joint_starts_released_under_scene"
    ),
    "TestSteelRevoluteLoop::test_revolute_loop_overflow_drops_all_five_rows": "STEEL_ROW_BUDGETS",
    "TestSteelRevoluteLoop::test_revolute_loop_pre_eliminates_five_rows": "STEEL_PREELIMINATION",
    # test_capacity
    # Scene counterpart: test_scene_features::test_scene_capacity_failure_is_observable
    "TestSteelCapacityStatus::test_capacity_failure_is_observable": (
        "exception: a world's row overflow, which the pooled row budget absorbs"
    ),
    "TestSteelCapacityStatus::test_dense_capacity_failure_is_world_local": (
        "scene counterpart: test_scene_capacity_failure_is_observable"
    ),
    # Scene counterpart: test_scene_features::test_scene_overflow_warning_is_printed_once
    "TestSteelCapacityStatus::test_overflow_warning_is_printed_once": (
        "exception: a world's row overflow, which the pooled row budget absorbs"
    ),
    # Scene counterpart: test_scene_features::test_scene_overflow_warning_is_printed_once
    "TestSteelCapacityStatus::test_overflow_warning_is_printed_once_propagation": (
        "exception: a world's row overflow, which the pooled row budget absorbs"
    ),
    "TestSteelCapacityStatus::test_status_snapshot_reports_status_at_snapshot": (
        "scene counterpart: test_scene_status_snapshot_is_kept_by_reset"
    ),
    "TestSteelCapacityStatus::test_unchecked_status_snapshot_is_kept_by_reset": (
        "scene counterpart: test_scene_status_snapshot_is_kept_by_reset"
    ),
    "TestSteelHeteroGuard::test_hetero_serial_constructs": (
        "scene counterpart: test_heterogeneous_worlds_match_isolated_worlds_bitwise"
    ),
    "TestSteelHeteroGuard::test_homogeneous_propagation_constructs": (
        "scene counterpart: test_heterogeneous_worlds_match_isolated_worlds_bitwise"
    ),
    "TestSteelHeteroGuard::test_homogeneous_serial_constructs": (
        "scene counterpart: test_heterogeneous_worlds_match_isolated_worlds_bitwise"
    ),
    "TestSteelRowAllocation::test_propagation_constraint_max_keeps_contact_rows_with_a_small_free_body_budget": (
        "STEEL_PROPAGATION"
    ),
    "TestSteelRowWatermark::test_watermarks_record_capacity_overflow": (
        "scene counterpart: test_scene_watermark_excess_counts_dropped_rows"
    ),
    "TestSteelRowWatermark::test_watermarks_track_propagation_rows": "STEEL_PROPAGATION",
    "TestSteelRowWatermark::test_watermarks_track_the_largest_row_counts": (
        "scene counterpart: test_row_watermark_counts_rows_per_model_world"
    ),
    # test_compliance
    "TestContactComplianceColored::test_colored_mixed_routes_match_immediate": "STEEL_ROW_FAMILIES",
    "TestContactComplianceColored::test_colored_native_hydro_matches_immediate": "STEEL_ROW_FAMILIES",
    "TestContactComplianceColored::test_colored_rising_kinematic_support_matches_immediate": "STEEL_ROW_FAMILIES",
    "TestContactComplianceIntegration::test_enabled_option_keeps_rigid_contacts_on_local_owner_scene": (
        "STEEL_CONTACT_SOLVES"
    ),
    "TestContactComplianceIntegration::test_mf_free_body_support": (
        "scene counterpart: test_mf_free_body_support_under_scene"
    ),
    "TestContactComplianceIntegration::test_native_hydro_free_body_manifold": "STEEL_ROW_FAMILIES",
    "TestContactComplianceIntegration::test_paired_factor_owner_matches_general_owner": "STEEL_CONTACT_SOLVES",
    "TestContactComplianceSafety::test_solver_overflow_and_dropped_contact_rejected": (
        "scene counterpart: test_compliance_counts_match_steel"
    ),
    "TestContactComplianceSafety::test_underflowing_compliance_keeps_friction_weight": "STEEL_ROW_FAMILIES",
    # test_contact_reduction
    "TestSteelBodyPairReductionConformance::test_steel_contact_mode_conformance": "STEEL_PROPAGATION",
    "TestSteelBodyPairReductionConformance::test_steel_same_articulation_rows_are_nonvacuous": "STEEL_PROPAGATION",
    # test_contacts
    "TestSteelContactControls::test_overflow_warning_enabled_after_construction_latches": "STEEL_ROW_BUDGETS",
    "TestSteelContactObservable::test_contact_f_in_a_captured_graph": (
        "scene counterpart: test_contact_f_in_a_captured_graph_under_scene"
    ),
    "TestSteelContactObservable::test_contact_f_reports_each_box_weight": (
        "scene counterpart: test_contact_f_reports_each_box_weight_under_scene"
    ),
    "TestSteelContactObservable::test_contact_f_reports_each_box_weight_row_overflow": (
        "exception: a world's row overflow, which the pooled row budget absorbs"
    ),
    "TestSteelNesterovMomentum::test_nesterov_momentum_on_articulated_rows": "rejected: nesterov_momentum",
    "TestSteelPairedResponse::test_paired_factor_coordinates_follow_patch_friction_and_torsion": "STEEL_CONTACT_SOLVES",
    "TestSteelPairedResponse::test_paired_factor_coordinates_with_contact_compliance_match_general": (
        "STEEL_CONTACT_SOLVES"
    ),
    "TestSteelPairedResponse::test_paired_response_falls_back_for_matrix_free_rows": "STEEL_CONTACT_SOLVES",
    "TestSteelPairedResponse::test_paired_response_matches_general_23_dof_trajectory": "STEEL_CONTACT_SOLVES",
    "TestSteelPairedResponse::test_paired_response_selection": "STEEL_CONTACT_SOLVES",
    "TestSteelPairedResponse::test_paired_response_without_factor_coordinates_matches_general": "STEEL_CONTACT_SOLVES",
    "TestSteelPersistentContacts::test_cached_friction_obeys_changed_material": "STEEL_ROW_FAMILIES",
    "TestSteelPersistentContacts::test_cached_friction_preserves_world_direction": "STEEL_ROW_FAMILIES",
    "TestSteelPersistentContacts::test_cached_friction_uses_solver_normal_convention": "STEEL_ROW_FAMILIES",
    "TestSteelPersistentContacts::test_combined_history_reset_replays_with_changed_world_mask": (
        "STEEL_WARMSTART_HISTORY"
    ),
    "TestSteelPersistentContacts::test_rotating_contact_clamps_the_projected_friction_cone": "STEEL_ROW_FAMILIES",
    "TestSteelRestitution::test_every_contact_response_route_enforces_restitution": (
        "scene counterpart: test_every_contact_response_route_enforces_restitution_under_scene"
    ),
    "TestSteelRestitution::test_multiworld_restitution_keeps_case_data_isolated": (
        "scene counterpart: test_multiworld_restitution_keeps_case_data_isolated_under_scene"
    ),
    "TestSteelRestitution::test_notify_reads_replaced_and_in_place_restitution": (
        "scene counterpart: test_notify_reads_replaced_and_in_place_restitution_under_scene"
    ),
    "TestSteelRestitution::test_penetrating_impact_replaces_baumgarte_with_restitution": (
        "scene counterpart: test_penetrating_impact_replaces_baumgarte_with_restitution_under_scene"
    ),
    "TestSteelRestitution::test_restitution_is_relative_to_a_moving_kinematic_surface": (
        "scene counterpart: test_restitution_is_relative_to_a_moving_kinematic_surface_under_scene"
    ),
    "TestSteelRestitution::test_restitution_uses_ordinary_rows_and_validates_configuration": (
        "scene counterpart: test_restitution_rows_fire_where_steel_fires"
    ),
    "TestSteelRestitution::test_two_body_impact_conserves_momentum_and_reverses_relative_speed": (
        "scene counterpart: test_two_body_impact_conserves_momentum_and_reverses_relative_speed_under_scene"
    ),
    "TestSteelRestitution::test_warm_start_history_does_not_change_or_repeat_restitution": (
        "scene counterpart: test_warm_start_history_does_not_change_or_repeat_restitution_under_scene"
    ),
    "TestSteelVelocityPassSpeculative::test_crossing_row_loses_its_speculative_allowance": (
        "scene counterpart: test_crossing_row_loses_its_speculative_allowance_under_scene"
    ),
    "TestSteelVelocityPassSpeculative::test_every_contact_response_route_lands": (
        "scene counterpart: test_velocity_iterations_rebound_tracks_steel"
    ),
    "TestSteelVelocityPassSpeculative::test_light_body_crossing_uses_end_gap_not_impulse_scale": (
        "scene counterpart: test_light_body_crossing_uses_end_gap_not_impulse_scale_under_scene"
    ),
    "TestSteelVelocityPassSpeculative::test_non_crossing_row_keeps_its_speculative_allowance": (
        "scene counterpart: test_non_crossing_row_keeps_its_speculative_allowance_under_scene"
    ),
    "TestSteelVelocityPassSpeculative::test_warm_start_with_velocity_iterations_is_supported": (
        "scene counterpart: test_warm_start_with_velocity_iterations_is_supported_under_scene"
    ),
    # test_determinism
    "TestSteelDeterminism::test_deterministic_bilateral_rows_follow_element_order": (
        "scene counterpart: test_row_order_is_side_major_then_colors"
    ),
    "TestSteelDeterminism::test_deterministic_joint_friction_rows_follow_dof_order": (
        "scene counterpart: test_joint_friction_bounds_match_steel_bitwise"
    ),
    "TestSteelDeterminism::test_deterministic_joint_rows_follow_articulation_order": (
        "scene counterpart: test_row_order_is_side_major_then_colors"
    ),
    "TestSteelDeterminism::test_deterministic_launches_replay_identically": (
        "scene counterpart: test_deterministic_launches_replay_identically_under_scene"
    ),
    "TestSteelDeterminism::test_deterministic_overflow_keeps_earliest_contacts": (
        "scene counterpart: test_rows_overflow_in_scan_order"
    ),
    "TestSteelDeterminism::test_overflow_warning_names_deterministic_mode": (
        "scene counterpart: test_scene_overflow_warning_is_printed_once"
    ),
    "TestSteelPropagationDeterminism::test_deterministic_propagation_launches_replay_identically": (
        "scene counterpart: test_deterministic_propagation_launches_replay_identically_under_scene"
    ),
    "TestSteelPropagationDeterminism::test_deterministic_propagation_runs_are_bitwise_identical": "STEEL_PROPAGATION",
    # test_friction
    "TestSteelFrictionAnchorSelection::test_anchor_selection_respects_contact_gap_filters": (
        "scene counterpart: test_anchor_selection_respects_contact_gap_filters_under_scene"
    ),
    "TestSteelFrictionAnchorSelection::test_contact_churn_reports_linear_force_but_not_a_wrench": "STEEL_ROW_FAMILIES",
    "TestSteelFrictionAnchorSelection::test_cuda_planar_patch_reduces_friction_rows": (
        "scene counterpart: test_cuda_planar_patch_reduces_friction_rows_under_scene"
    ),
    "TestSteelFrictionAnchors::test_row_builders_store_anchor_separation_per_route": (
        "scene counterpart: test_patch_rows_follow_the_anchors"
    ),
    "TestSteelFrictionMode::test_modes_select_distinct_kernels": "scene counterpart: test_friction_modes_track_steel",
    "TestSteelFrictionMode::test_sparse_factors_require_the_current_mode": (
        "scene counterpart: test_friction_modes_track_steel"
    ),
    "TestSteelFrictionRolling::test_faceted_rolling_across_tessellation_direction_and_timestep": (
        "scene counterpart: test_faceted_rolling_across_tessellation_direction_and_timestep_under_scene"
    ),
    "TestSteelKinematicBeltFriction::test_kinematic_belt_drags_the_box": (
        "scene counterpart: test_kinematic_belt_drags_the_box_under_scene"
    ),
    # test_kernels
    "TestSteelAdvancedOptions::test_serial_block_dim_is_bitwise": (
        "scene counterpart: test_serial_block_dim_is_bitwise_under_scene"
    ),
    "TestSteelFusedCrba::test_fused_assembly_follows_the_mass_update_interval": "STEEL_FACTOR_SELECTION",
    "TestSteelFusedCrba::test_fused_assembly_is_selected_only_where_supported": "STEEL_FACTOR_SELECTION",
    "TestSteelFusedCrba::test_fused_assembly_matches_the_separate_pipeline": "STEEL_FACTOR_SELECTION",
    "TestSteelLaunchConfig::test_articulated_contact_response_validation": "STEEL_PROPAGATION",
    "TestSteelLaunchConfig::test_compact_world_dof_mapping_pads_heterogeneous_worlds": "STEEL_SOLVE_WORLDS",
    "TestSteelLaunchConfig::test_cpu_runs_the_matrix_free_solve": "STEEL_DEFAULTS",
    "TestSteelLaunchConfig::test_defaults": "STEEL_DEFAULTS",
    "TestSteelLaunchConfig::test_diagonal_fusion_requires_nonaliased_world_response": "STEEL_FACTOR_SELECTION",
    "TestSteelLaunchConfig::test_propagation_rejects_unsupported_options": "STEEL_PROPAGATION",
    "TestSteelLaunchConfig::test_tiled_and_loop_kernels_step_identically": (
        "scene counterpart: test_tiled_and_loop_kernels_step_identically_under_scene"
    ),
    "TestSteelMatrixFreePairCache::test_pair_cache_matches_the_uncached_solve_contact_torsion": (
        "STEEL_MATRIX_FREE_SWEEP"
    ),
    "TestSteelMatrixFreePairCache::test_pair_cache_matches_the_uncached_solve_resident_metadata": (
        "STEEL_MATRIX_FREE_SWEEP"
    ),
    "TestSteelMatrixFreePairCache::test_pair_cache_matches_the_uncached_solve_streamed_metadata": (
        "STEEL_MATRIX_FREE_SWEEP"
    ),
    "TestSteelMatrixFreePairCache::test_pair_cache_selection": "STEEL_MATRIX_FREE_SWEEP",
    "TestSteelMatrixFreeRowPhase::test_compiled_row_phase_matches_runtime_phase_contact_then_internal": (
        "STEEL_MATRIX_FREE_SWEEP"
    ),
    "TestSteelMatrixFreeRowPhase::test_compiled_row_phase_matches_runtime_phase_phased": "STEEL_MATRIX_FREE_SWEEP",
    "TestSteelMatrixFreeRowPhase::test_compiled_row_phase_matches_runtime_phase_propagation": "STEEL_MATRIX_FREE_SWEEP",
    "TestSteelMatrixFreeRowPhase::test_compiled_row_phase_matches_runtime_phase_propagation_contact_rows_only": (
        "STEEL_MATRIX_FREE_SWEEP"
    ),
    "TestSteelMatrixFreeRowPhase::test_compiled_row_phase_matches_runtime_phase_propagation_contact_then_internal": (
        "STEEL_MATRIX_FREE_SWEEP"
    ),
    "TestSteelMatrixFreeRowPhase::test_compiled_row_phases_replay_a_captured_first_step": "STEEL_MATRIX_FREE_SWEEP",
    "TestSteelReusedKinematics::test_reused_kinematics_follow_resets_and_notices": "rejected: reuse_kinematics",
    "TestSteelReusedKinematics::test_reused_kinematics_match_the_recomputed_trajectory": "rejected: reuse_kinematics",
    "TestSteelReusedKinematics::test_torsion_rollback_drops_the_carried_kinematics": "STEEL_TORSION_PREPARATION",
    "TestSteelStreams::test_capture_after_eager_orders_the_pending_clear": "STEEL_FACTOR_SELECTION",
    "TestSteelStreams::test_double_buffer_capture_sequences_match_single_buffer": "STEEL_FACTOR_SELECTION",
    "TestSteelStreams::test_reset_after_an_eager_error_steps_like_a_fresh_solver": "STEEL_FACTOR_SELECTION",
    # Scene counterpart (streams alone): test_scene_sweep::test_parallel_streams_step_identically_under_scene
    "TestSteelStreams::test_streams_and_double_buffer_step_identically": "STEEL_FACTOR_SELECTION",
    # test_limits_drives
    "TestSteelJointFriction::test_dense_row_overflow_is_reported": "STEEL_ROW_BUDGETS",
    "TestSteelJointFriction::test_local_internal_fast_path_is_not_selected": "STEEL_CONTACT_SOLVES",
    "TestSteelJointFriction::test_schedules_routes_and_velocity_pass_agree": (
        "scene counterpart: test_schedules_routes_and_velocity_pass_agree_under_scene"
    ),
    "TestSteelJointLimitActivationGap::test_combined_row_families_follow_the_documented_layout": (
        "scene counterpart: test_late_rows_follow_the_coupled_pass"
    ),
    "TestSteelJointLimitActivationGap::test_default_finite_limits_keep_rows_and_warn_at_capacity": "STEEL_ROW_BUDGETS",
    "TestSteelJointLimitActivationGap::test_dense_row_families_respect_phase_bounds": "STEEL_ROW_LAYOUT",
    "TestSteelJointLimitActivationGap::test_propagation_solves_every_row_family": "STEEL_PROPAGATION",
    "TestSteelJointLimitActivationGap::test_velocity_limit_rows_require_the_option": (
        "scene counterpart: test_free_body_velocity_limit_rows_match_steel_rows"
    ),
    "TestSteelPGSDrives::test_drive_rows_precede_limit_rows_and_count_against_capacity": "STEEL_ROW_BUDGETS",
    "TestSteelPGSDrives::test_undriven_dofs_get_no_drive_rows": "scene counterpart: test_drive_rows_track_steel",
    "TestSteelVelocityDriveMode::test_frozen_drive_rows_keep_the_position_solve_impulse": (
        "scene counterpart: test_frozen_drive_rows_keep_the_position_solve_impulse_under_scene"
    ),
    # test_link_break
    "TestSteelLinkBreak::test_link_break_preelimination_falls_back": "STEEL_PREELIMINATION",
    # test_mass_matrix
    "TestSteelDiagonalMass::test_diagonal_mass_rows_match_dense_and_sparse_factors": "STEEL_FACTOR_SELECTION",
    "TestSteelDiagonalMass::test_fixed_base_sibling_branches_detect_diagonal_mass_topology": "STEEL_FACTOR_SELECTION",
    "TestSteelDiagonalMass::test_fixed_base_sibling_branches_use_diagonal_mass_path": "STEEL_FACTOR_SELECTION",
    "TestSteelDiagonalMass::test_sleeping_diagonal_dynamics_skip_is_exact": (
        "scene counterpart: test_sleeping_diagonal_dynamics_skip_is_exact_under_scene"
    ),
    "TestSteelNotifyDevice::test_armature_and_shape_notify_capture_into_a_graph": (
        "scene counterpart: test_armature_and_shape_notify_capture_into_a_graph_under_scene"
    ),
    "TestSteelNotifyDevice::test_armature_notify_matches_host_reference_and_fresh_solver": (
        "scene counterpart: test_armature_notify_matches_host_reference_and_fresh_solver_under_scene"
    ),
    "TestSteelNotifyDevice::test_captured_notify_reads_its_captured_arrays": (
        "scene counterpart: test_captured_notify_reads_its_captured_arrays_under_scene"
    ),
    "TestSteelNotifyDevice::test_captured_world_mask_follows_the_live_mask": "STEEL_WARMSTART_HISTORY",
    "TestSteelNotifyDevice::test_kinematic_notify_matches_host_reference_and_fresh_solver": (
        "scene counterpart: test_kinematic_notify_matches_host_reference_and_fresh_solver_under_scene"
    ),
    "TestSteelNotifyDevice::test_narrow_dof_flags_and_material_copies_capture_into_a_graph": (
        "scene counterpart: test_narrow_dof_flags_and_material_copies_capture_into_a_graph_under_scene"
    ),
    "TestSteelNotifyDevice::test_world_mask_limits_warmstart_invalidation": "STEEL_WARMSTART_HISTORY",
    # test_options
    "TestSteelFrictionOptions::test_friction_position_iterations_gate_every_solve_path": (
        "scene counterpart: test_friction_position_iterations_gate_the_scene_sweep"
    ),
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
    "TestSteelSolveModeOptions::test_deprecated_spellings_map_to_the_new_options": "STEEL_DEFAULTS",
    "TestSteelSolveModeOptions::test_new_spellings_are_accepted_and_invalid_ones_rejected": "STEEL_DEFAULTS",
    # test_parent_wrench
    "TestSteelParentWrench::test_parent_wrench_warns_without_preeliminated_closures": "STEEL_PREELIMINATION",
    # test_propagation
    "TestLeafPublication::test_cuda_actual_graph_publication": (
        "scene counterpart: test_cuda_actual_graph_publication_under_scene"
    ),
    "TestLeafPublication::test_current_frames_and_inertial_notifications": (
        "scene counterpart: test_current_frames_and_inertial_notifications_under_scene"
    ),
    "TestLeafPublication::test_interleaved_mixed_leaves_and_moving_parents": (
        "scene counterpart: test_interleaved_mixed_leaves_and_moving_parents_under_scene"
    ),
    "TestPropagationSameArticulation::test_closing_gripper_stops_at_finger_contact": "STEEL_PROPAGATION",
    "TestPropagationSameArticulation::test_cross_response_effective_mass_matches_reference_operator": (
        "STEEL_PROPAGATION"
    ),
    "TestPropagationSameArticulation::test_default_routing_keeps_same_articulation_rows_dense": "STEEL_PROPAGATION",
    "TestPropagationSameArticulation::test_same_articulation_torsion_rows_carry_the_cross_response": (
        "STEEL_PROPAGATION"
    ),
    "TestPropagationSameArticulation::test_scissor_dense_diagonal_matches_reference": (
        "scene counterpart: test_rows_match_steel_rows_bitwise"
    ),
    "TestSteelColoredPipeline::test_straight_line_units_match_the_row_loop": "STEEL_PROPAGATION",
    "TestSteelColoredPipeline::test_sweep_variants_match_the_reference_bitwise": "STEEL_PROPAGATION",
    "TestSteelColoredScheduling::test_colored_accepts_a_large_row_budget": "STEEL_PROPAGATION",
    "TestSteelColoredScheduling::test_colored_result_matches_the_immediate_response": (
        "scene counterpart: test_colored_result_matches_the_immediate_response_under_scene"
    ),
    "TestSteelColoredScheduling::test_coloring_is_deterministic_for_a_fixed_contact_set": "STEEL_PROPAGATION",
    "TestSteelColoredScheduling::test_dense_heap_colors_without_a_serial_tail": "STEEL_PROPAGATION",
    "TestSteelColoredScheduling::test_kinematic_free_body_is_recorded_as_static": "STEEL_PROPAGATION",
    "TestSteelColoredScheduling::test_kinematic_hub_does_not_serialize_its_contacts": "STEEL_PROPAGATION",
    "TestSteelColoredScheduling::test_prebuild_matches_the_first_fit_reference": "STEEL_PROPAGATION",
    "TestSteelColoredUnitCapacity::test_mixed_units_graph_matches_eager": "STEEL_PROPAGATION",
    "TestSteelColoredUnitCapacity::test_mixed_units_match_serial_propagation": "STEEL_PROPAGATION",
    "TestSteelPropagation::test_contact_effective_mass_matches_dense_response": "STEEL_PROPAGATION",
    "TestSteelPropagation::test_global_kinematic_floor_moves_articulated_bodies": (
        "scene counterpart: test_global_kinematic_floor_moves_articulated_bodies_under_scene"
    ),
    "TestSteelPropagation::test_native_tree_kernels_match_generic": "STEEL_PROPAGATION",
    "TestSteelPropagation::test_phased_sweeps_touch_only_their_row_family": "STEEL_MATRIX_FREE_SWEEP",
    "TestSteelPropagation::test_propagation_matches_immediate_when_converged": (
        "scene counterpart: test_propagation_matches_immediate_when_converged_under_scene"
    ),
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
    "TestSteelPropagationCombinations::test_options_match_the_immediate_response_at_convergence": (
        "scene counterpart: test_options_match_the_immediate_response_at_convergence_under_scene"
    ),
    "TestSteelPropagationCombinations::test_regularization_weights_reach_the_propagation_rows": "STEEL_PROPAGATION",
    "TestSteelPropagationCombinations::test_restitution_rebounds_on_propagation_rows": (
        "scene counterpart: test_restitution_rebounds_on_propagation_rows_under_scene"
    ),
    "TestSteelPropagationCombinations::test_warm_start_seeds_the_propagation_rows": "STEEL_PROPAGATION",
    "TestSteelPropagationCoupledContacts::test_graph_replay_matches_eager_steps": "STEEL_PROPAGATION",
    "TestSteelPropagationCoupledContacts::test_linkage_gripper_holds_lifted_box": (
        "scene counterpart: test_linkage_gripper_holds_lifted_box_under_scene"
    ),
    "TestSteelPropagationCoupledContacts::test_regularized_feet_reach_immediate_equilibrium": (
        "scene counterpart: test_regularized_feet_reach_immediate_equilibrium_under_scene"
    ),
    "TestSteelPropagationCoupledContacts::test_shared_dof_feet_split_their_whole_response": "STEEL_PROPAGATION",
    "TestSteelPropagationCoupledContacts::test_shared_root_split_matches_across_kernels": "STEEL_PROPAGATION",
    "TestSteelSchedule::test_contact_then_internal_gives_internal_rows_the_last_word": "STEEL_MATRIX_FREE_SWEEP",
    "TestSteelSchedule::test_row_phases_touch_only_their_row_families": "STEEL_MATRIX_FREE_SWEEP",
    "TestSteelTreeExecution::test_contact_trajectory_matches_serial": (
        "scene counterpart: test_contact_trajectory_matches_serial_under_scene"
    ),
    "TestSteelTreeExecution::test_serial_selection_skips_tree_setup": (
        "scene counterpart: test_serial_selection_skips_tree_setup_under_scene"
    ),
    # test_scene_mode
    "TestSteelSceneMode::test_scene_mode_is_off_by_default": "STEEL_DEFAULTS",
    # test_sleeping
    "TestSteelSleepWorkSkipping::test_graph_replay_restores_rows_on_force": (
        "scene counterpart: test_graph_replay_restores_rows_on_force_under_scene"
    ),
    "TestSteelSleeping::test_default_and_explicit_disabled_match": (
        "scene counterpart: test_default_and_explicit_disabled_match_under_scene"
    ),
    "TestSteelSleepingDynamics::test_alternate_kernel_paths_match": (
        "scene counterpart: test_alternate_kernel_paths_match_under_scene"
    ),
    "TestSteelSleepingDynamics::test_eager_trajectories_match": (
        "scene counterpart: test_eager_trajectories_match_under_scene"
    ),
    "TestSteelSleepingDynamics::test_graph_trajectories_match": (
        "scene counterpart: test_graph_trajectories_match_under_scene"
    ),
    "TestSteelSleepingDynamics::test_transient_divergence_is_reported": (
        "scene counterpart: test_transient_divergence_is_reported_under_scene"
    ),
    # test_sparse
    "TestSteelLocalOwners::test_contact_owners_remain_available_at_sm_count": "STEEL_CONTACT_SOLVES",
    "TestSteelLocalOwners::test_free_body_first_world_layout_matches_general": "STEEL_CONTACT_SOLVES",
    "TestSteelLocalOwners::test_free_body_velocity_limit_rows_stay_on_general_owner": "STEEL_CONTACT_SOLVES",
    "TestSteelLocalOwners::test_large_contact_batches_keep_single_owner_and_match_general": "STEEL_CONTACT_SOLVES",
    "TestSteelLocalOwners::test_local_internal_mixed_world_matches_general_response": "STEEL_CONTACT_SOLVES",
    "TestSteelLocalOwners::test_local_owner_matches_general_persistent_patch_friction": "STEEL_CONTACT_SOLVES",
    "TestSteelLocalOwners::test_local_owners_match_general_contact_regularization": "STEEL_CONTACT_SOLVES",
    "TestSteelLocalOwners::test_local_owners_stay_general_with_contact_torsion": "STEEL_CONTACT_SOLVES",
    "TestSteelLocalOwners::test_local_pair_matches_general_friction_and_restitution": "STEEL_CONTACT_SOLVES",
    "TestSteelSparseDiagonal::test_sparse_diagonal_contact_compliance_matches_general_owner": "STEEL_CONTACT_SOLVES",
    "TestSteelSparseDiagonal::test_sparse_diagonal_contact_friction_matches_general_owner": "STEEL_CONTACT_SOLVES",
    "TestSteelSparseDiagonal::test_sparse_diagonal_is_selected_only_where_supported": "STEEL_CONTACT_SOLVES",
    "TestSteelSparseDiagonal::test_sparse_diagonal_response_matches_dense_joint_limits": "STEEL_CONTACT_SOLVES",
    "TestSteelSparseSelectionGuard::test_bilateral_rows_keep_dense_factors": (
        "scene counterpart: test_mimic_rows_track_steel"
    ),
    "TestSteelSparseSolver::test_default_selects_sparse_factors_for_branched_articulations": (
        "scene counterpart: test_one_topology_scene_has_one_factor_group"
    ),
    "TestSteelSparseSolver::test_graph_replay_preserves_sparse_problem": (
        "scene counterpart: test_graph_replay_preserves_sparse_problem_under_scene"
    ),
    "TestSteelSparseSolver::test_heterogeneous_worlds_do_not_invent_joint_support": (
        "scene counterpart: test_heterogeneous_scene_tracks_steel"
    ),
    "TestSteelSparseSolver::test_mixed_contacts_match_existing_solver": (
        "scene counterpart: test_heterogeneous_scene_tracks_steel"
    ),
    "TestSteelSparseSolver::test_mixed_graph_replay_after_masked_reset": (
        "scene counterpart: test_mixed_graph_replay_after_masked_reset_under_scene"
    ),
    "TestSteelSparseSolver::test_moving_prescribed_support_and_reset": (
        "scene counterpart: test_heterogeneous_scene_tracks_steel"
    ),
    "TestSteelSparseSolver::test_patch_friction_matches_dense_factors": "STEEL_FACTOR_SELECTION",
    "TestSteelSparseSolver::test_trajectory_masked_reset_and_inertial_notify": (
        "scene counterpart: test_trajectory_masked_reset_and_inertial_notify_under_scene"
    ),
    "TestSteelSparseSolver::test_two_articulations_share_contact_solve": (
        "scene counterpart: test_heterogeneous_scene_tracks_steel"
    ),
    "TestSteelSparseSolver::test_unsupported_configurations_keep_existing_path": "STEEL_FACTOR_SELECTION",
    # test_torsion
    "TestColoredPropagationTorsion::test_colored_mixed_families_track_plain_through_the_contact_lifecycle": (
        "STEEL_PROPAGATION"
    ),
    "TestColoredPropagationTorsion::test_colored_rows_host_in_the_last_members_unit_at_its_highest_color": (
        "STEEL_PROPAGATION"
    ),
    "TestContactTorsion::test_capture_is_explicitly_rejected": "STEEL_TORSION_PREPARATION",
    "TestContactTorsion::test_hydro_combination_is_rejected": "scene counterpart: test_torsion_errors_are_latched",
    "TestContactTorsion::test_joint_response_and_coupled_budget": "scene counterpart: test_spin_rows_match_steel_rows",
    "TestContactTorsion::test_timestep_and_capacity": "STEEL_TORSION_PREPARATION",
    "TestContactTorsionRegressions::test_contact_input_overflow_is_rejected": (
        "scene counterpart: test_torsion_errors_are_latched"
    ),
    "TestContactTorsionRegressions::test_contact_row_loss_fails_without_diagnostics": (
        "scene counterpart: test_torsion_errors_are_latched"
    ),
    "TestContactTorsionRegressions::test_hydro_contact_input_is_rejected": (
        "scene counterpart: test_torsion_errors_are_latched"
    ),
    "TestContactTorsionRegressions::test_patch_spanning_shape_pairs_takes_the_largest_radius": (
        "STEEL_TORSION_PREPARATION"
    ),
    "TestContactTorsionRegressions::test_torsion_keeps_free_body_velocity_limits": (
        "scene counterpart: test_one_sided_and_free_body_spin_rows_track_steel"
    ),
    "TestContactTorsionRegressions::test_torsion_uses_cfm_floor": "scene counterpart: test_spin_rows_match_steel_rows",
    "TestDeviceTorsionPreparation::test_cuda_cold_capture_reports_errors_until_reset": "STEEL_TORSION_PREPARATION",
    "TestDeviceTorsionPreparation::test_cuda_solver_profiles": "STEEL_TORSION_PREPARATION",
    "TestDeviceTorsionPreparation::test_cuda_status_snapshot_reports_a_replay_error": "STEEL_TORSION_PREPARATION",
    "TestDeviceTorsionPreparation::test_public_api_noop_and_fail_stop": (
        "scene counterpart: test_torsion_errors_are_latched"
    ),
    "TestFinalPatchBudget::test_final_budget_and_velocity_response": (
        "scene counterpart: test_spin_rows_match_steel_rows"
    ),
    "TestPropagationTorsion::test_propagation_budget_holds_after_every_sweep": "STEEL_PROPAGATION",
    "TestPropagationTorsion::test_propagation_captured_steps_match_eager_steps_and_report_errors": "STEEL_PROPAGATION",
    "TestPropagationTorsion::test_propagation_converges_to_the_immediate_response": "STEEL_PROPAGATION",
    "TestPropagationTorsion::test_propagation_dropped_members_leave_the_highest_kept_host": "STEEL_PROPAGATION",
    "TestPropagationTorsion::test_propagation_free_body_contacts_reserve_no_spare": "STEEL_PROPAGATION",
    "TestPropagationTorsion::test_propagation_mixed_families_and_contact_lifecycle": "STEEL_PROPAGATION",
    "TestPropagationTorsion::test_propagation_patch_impulse_matches_immediate": "STEEL_PROPAGATION",
    "TestPropagationTorsion::test_propagation_prescribed_spin_turns_a_resting_sphere": "STEEL_PROPAGATION",
    "TestPropagationTorsion::test_propagation_rows_follow_the_recorded_reservation": "STEEL_PROPAGATION",
    "TestPropagationTorsion::test_propagation_rows_host_groups_in_spares": "STEEL_PROPAGATION",
    "TestPropagationTorsion::test_propagation_rows_keep_a_shared_static_body_out": "STEEL_PROPAGATION",
    "TestPropagationTorsion::test_propagation_spin_decelerates_at_the_coulomb_rate_and_sticks": "STEEL_PROPAGATION",
    "TestPropagationTorsion::test_propagation_worlds_solve_as_if_isolated": "STEEL_PROPAGATION",
    "TestTorsionCombined::test_analytic_normal_load_position_soft_velocity_rigid": (
        "scene counterpart: test_spin_rows_match_steel_rows"
    ),
    "TestTorsionCombined::test_distributed_budget_and_export": "scene counterpart: test_spin_rows_match_steel_rows",
    "TestTorsionCombined::test_no_load_near_positive_gap_and_release": (
        "scene counterpart: test_retired_spin_row_refunds_under_velocity_iterations"
    ),
    "TestTorsionRegularization::test_analytic_regularized_load_and_saturated_spin": (
        "scene counterpart: test_spin_rows_match_steel_rows"
    ),
    "TestTorsionRegularization::test_final_load_budget_across_iterations": (
        "scene counterpart: test_spin_rows_match_steel_rows"
    ),
    "TestTorsionRowOverflow::test_overflowing_world_is_flagged_and_others_are_unchanged": "STEEL_ROW_BUDGETS",
    "TestTorsionRowOverflow::test_torsion_overflow_warns_once": "STEEL_ROW_BUDGETS",
    "TestTorsionSweepLinks::test_large_capacity_falls_back": "STEEL_MATRIX_FREE_SWEEP",
    "TestTorsionSweepLinks::test_links_take_precedence_over_pair_cache": "STEEL_MATRIX_FREE_SWEEP",
    "TestTorsionSweepLinks::test_solver_selects_links_and_matches_fallback": "STEEL_MATRIX_FREE_SWEEP",
    "TestTorsionVelocitySolve::test_carried_spin_is_refunded_when_postpass_retires_group": (
        "scene counterpart: test_retired_spin_row_refunds_under_velocity_iterations"
    ),
    "TestTorsionVelocitySolve::test_postpass_accumulated_impulse_response_and_position_split": (
        "scene counterpart: test_spin_rows_match_steel_rows"
    ),
    # test_warmstart
    "TestSteelIdentityWarmstart::test_both_row_families_seed_impulses_scaled_by_the_step_ratio": (
        "scene counterpart: test_both_row_families_seed_impulses_scaled_by_the_step_ratio_under_scene"
    ),
    "TestSteelIdentityWarmstart::test_graph_replay_rescales_history_once_after_a_timestep_change": (
        "scene counterpart: test_graph_replay_rescales_history_once_after_a_timestep_change_under_scene"
    ),
    "TestSteelIdentityWarmstart::test_interleaved_contact_buffers_start_cold": (
        "scene counterpart: test_interleaved_contact_buffers_start_cold_under_scene"
    ),
    "TestSteelIdentityWarmstart::test_interleaved_contact_buffers_start_cold_under_graph_replay": (
        "scene counterpart: test_interleaved_contact_buffers_start_cold_under_graph_replay_under_scene"
    ),
    "TestSteelIdentityWarmstart::test_real_contact_insertion_moves_slots_without_cross_seeding": (
        "scene counterpart: test_real_contact_insertion_moves_slots_without_cross_seeding_under_scene"
    ),
    "TestSteelIdentityWarmstart::test_replaced_contact_buffer_starts_cold": (
        "scene counterpart: test_replaced_contact_buffer_starts_cold_under_scene"
    ),
    "TestSteelIdentityWarmstart::test_skipped_collision_pass_starts_cold": (
        "scene counterpart: test_skipped_collision_pass_starts_cold_under_scene"
    ),
    "TestSteelIdentityWarmstart::test_substeps_reuse_contacts_with_their_own_history": (
        "scene counterpart: test_substeps_reuse_contacts_with_their_own_history_under_scene"
    ),
    "TestSteelWarmstartDt::test_warm_carry_fixed_dt_identity": (
        "scene counterpart: test_warm_carry_fixed_dt_identity_under_scene"
    ),
    "TestSteelWarmstartDt::test_warm_carry_scales_with_dt": (
        "scene counterpart: test_warm_carry_scales_with_dt_under_scene"
    ),
}
"""The tests the forced run skips: ``"<Class>::<test>"`` (the test's method name without a device suffix, so that the
key covers its ``_cpu`` and ``_cuda_0`` variants) -> a key of :data:`STEEL_PATH_REASONS`, or the reason itself:
``"scene counterpart: <test>"`` (the mode's own test that replaces it), ``"rejected: <feature>"`` (a feature the mode
rejects) or ``"exception: <name>"`` (a recorded exception: a behaviour of Steel's that the test holds and the mode, by
design, does not, named by that divergence). A ``# Scene counterpart:`` comment above an entry names the scene test
that covers its mechanism under the mode."""

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
