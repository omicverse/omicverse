r"""Tools (``ov.tl``) — analysis helpers that transform an AnnData.

Currently hosts the cross-species preprocessing extracted from
:mod:`omicverse.single` so it can be reused independently of the integration
backend, and channel-level QC for multichannel electrophysiology
(:func:`channel_qc_table`, :func:`bridged_channel_pairs`,
:func:`erp_channel_outliers`), leakage-aware ranking of fixed prediction
models for transfer to new environments (:func:`rank_models_for_transfer`),
quantitative susceptibility mapping from multi-echo gradient-echo MRI
(:func:`qsm_reconstruct` and its steps) and open-set individual
re-identification from images (:func:`reid_embed`, :func:`reid_local_verify`,
:func:`reid_open_set_cluster`), data-entry error audits of linked clinical / EHR
tables (:func:`audit_clinical_tables`) and study-level effect-size extraction for
meta-analysis (:func:`extract_group_stats`, :func:`group_comparisons`,
:func:`meta_effect_sizes`), per-survey species-set prediction from
presence-absence / presence-only data (:func:`sdm_community_pipeline` and its
steps),
small-area estimation with reference-population priors, group-varying
coefficients, cross-validated calibration and benchmarked joint draws
(:func:`sae_grouped_model` and helpers, scored by :func:`crps_ensemble`),
and morphometry of boxed objects in stained bright-field smears with shape-vs-stain
field calls that transfer to unseen groups (:func:`smear_measure`,
:func:`two_axis_classifier_fit`).
"""
from ._cross_species import cross_species_align
from ._bonsai import bonsai
from ._channel_qc import (bandpass_filter, channel_qc_table, bridged_channel_pairs,
                          erp_channel_outliers)
from ._erp import repair_reject_epochs, channel_erps
from ._trigger import trigger_onsets
from ._choice_model import (fit_choice_model, choice_model_loglik,
                            choice_model_forward_validation, export_choice_predictor)
from ._reid import (load_reid_backbone, reid_embed, reid_view_center, reid_candidate_pairs,
                    reid_local_verify, reid_fuse_similarity, reid_pair_auc,
                    reid_fit_projection, reid_select_threshold, reid_open_set_cluster,
                    reid_pipeline)
from ._qsm import (qsm_load_multi_echo, qsm_save_nifti, qsm_brain_mask, qsm_unwrap_phase,
                   qsm_multi_echo_field, qsm_background_removal, qsm_dipole_inversion,
                   qsm_reconstruct, qsm_dipole_kernel, qsm_forward_field, qsm_numerical_phantom)
from ._transfer_ranking import (rank_models_for_transfer, dependence_components,
                                transfer_ranking_source)
from ._clinical_audit import (load_clinical_tables, infer_clinical_schema, audit_clinical_tables,
                              drug_sex_reference)
from ._meta_analysis import (parse_mean_sd, extract_group_stats, group_comparisons, pool_group_stats,
                             hedges_g, sd_from_summary, recompute_group_test, meta_effect_sizes,
                             apply_selection_rules, comparison_workbook, format_workbook,
                             method_measure_statements)
from ._meta_selection import (abbreviation_map, propose_selection, reconstruct_study_effects,
                              write_study_effects, record_deliverable, verify_deliverable)
from ._sdm import (sdm_sample_f1, sdm_decode_sets, sdm_sets_to_strings, sdm_spatial_holdout,
                   sdm_neighbor_composition, sdm_presence_only_histogram, sdm_cube_summary,
                   sdm_read_tensor_dir, sdm_standardize, sdm_fit_mlp, sdm_fuse,
                   sdm_community_pipeline)
from ._small_area import (sae_longitudinal_design, sae_reference_prior, sae_grouped_model,
                          sae_benchmark, crps_ensemble, energy_score, pit_histogram_tv)
from ._smear_morph import (smear_measure, smear_stain_vectors, smear_field_features,
                           two_axis_classifier_fit, two_axis_classifier_predict,
                           group_holdout_two_axis, species_balanced_recall, group_laplace_prior,
                           smear_morphometry_source, smear_fit_predictor, smear_export_predictor)

__all__ = ["sae_longitudinal_design", "sae_reference_prior", "sae_grouped_model",
           "sae_benchmark", "crps_ensemble", "energy_score", "pit_histogram_tv",
           "cross_species_align", "bonsai", "bandpass_filter", "channel_qc_table",
           "bridged_channel_pairs", "erp_channel_outliers",
           "repair_reject_epochs", "channel_erps", "trigger_onsets",
           "fit_choice_model", "choice_model_loglik",
           "choice_model_forward_validation", "export_choice_predictor",
           "load_reid_backbone", "reid_embed", "reid_view_center", "reid_candidate_pairs",
           "reid_local_verify", "reid_fuse_similarity", "reid_pair_auc",
           "reid_fit_projection", "reid_select_threshold", "reid_open_set_cluster",
           "reid_pipeline",
           "qsm_load_multi_echo", "qsm_save_nifti", "qsm_brain_mask", "qsm_unwrap_phase",
           "qsm_multi_echo_field", "qsm_background_removal", "qsm_dipole_inversion",
           "qsm_reconstruct", "qsm_dipole_kernel", "qsm_forward_field", "qsm_numerical_phantom",
           "rank_models_for_transfer", "dependence_components", "transfer_ranking_source",
           "load_clinical_tables", "infer_clinical_schema", "audit_clinical_tables", "drug_sex_reference",
           "parse_mean_sd", "extract_group_stats", "group_comparisons", "pool_group_stats", "hedges_g",
           "sd_from_summary", "recompute_group_test", "meta_effect_sizes", "apply_selection_rules", "comparison_workbook",
           "format_workbook", "method_measure_statements",
           "abbreviation_map", "propose_selection", "reconstruct_study_effects",
           "write_study_effects", "record_deliverable", "verify_deliverable",
           "sdm_sample_f1", "sdm_decode_sets", "sdm_sets_to_strings", "sdm_spatial_holdout",
           "sdm_neighbor_composition", "sdm_presence_only_histogram", "sdm_cube_summary",
           "sdm_read_tensor_dir", "sdm_standardize", "sdm_fit_mlp", "sdm_fuse",
           "sdm_community_pipeline",
           "smear_measure", "smear_stain_vectors", "smear_field_features",
           "two_axis_classifier_fit", "two_axis_classifier_predict", "group_holdout_two_axis",
           "species_balanced_recall", "group_laplace_prior", "smear_morphometry_source",
           "smear_fit_predictor", "smear_export_predictor"]
from ._meta_analysis import to_mean_sd  # noqa: E402  (summary statistics -> mean / SD)
__all__ += ["to_mean_sd"]
