r"""Tools (``ov.tl``) — analysis helpers that transform an AnnData.

Currently hosts the cross-species preprocessing extracted from
:mod:`omicverse.single` so it can be reused independently of the integration
backend, and channel-level QC for multichannel electrophysiology
(:func:`channel_qc_table`, :func:`bridged_channel_pairs`,
:func:`erp_channel_outliers`), leakage-aware ranking of fixed prediction
models for transfer to new environments (:func:`rank_models_for_transfer`) and
quantitative susceptibility mapping from multi-echo gradient-echo MRI
(:func:`qsm_reconstruct` and its steps).
"""
from ._cross_species import cross_species_align
from ._bonsai import bonsai
from ._channel_qc import (bandpass_filter, channel_qc_table, bridged_channel_pairs,
                          erp_channel_outliers)
from ._erp import repair_reject_epochs, channel_erps
from ._trigger import trigger_onsets
from ._choice_model import (fit_choice_model, choice_model_loglik,
                            choice_model_forward_validation, export_choice_predictor)
from ._qsm import (qsm_load_multi_echo, qsm_save_nifti, qsm_brain_mask, qsm_unwrap_phase,
                   qsm_multi_echo_field, qsm_background_removal, qsm_dipole_inversion,
                   qsm_reconstruct, qsm_dipole_kernel, qsm_forward_field, qsm_numerical_phantom)
from ._transfer_ranking import (rank_models_for_transfer, dependence_components,
                                transfer_ranking_source)

__all__ = ["cross_species_align", "bonsai", "bandpass_filter", "channel_qc_table",
           "bridged_channel_pairs", "erp_channel_outliers",
           "repair_reject_epochs", "channel_erps", "trigger_onsets",
           "fit_choice_model", "choice_model_loglik",
           "choice_model_forward_validation", "export_choice_predictor",
           "qsm_load_multi_echo", "qsm_save_nifti", "qsm_brain_mask", "qsm_unwrap_phase",
           "qsm_multi_echo_field", "qsm_background_removal", "qsm_dipole_inversion",
           "qsm_reconstruct", "qsm_dipole_kernel", "qsm_forward_field", "qsm_numerical_phantom",
           "rank_models_for_transfer", "dependence_components", "transfer_ranking_source"]
