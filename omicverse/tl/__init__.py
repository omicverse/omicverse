r"""Tools (``ov.tl``) — analysis helpers that transform an AnnData.

Currently hosts the cross-species preprocessing extracted from
:mod:`omicverse.single` so it can be reused independently of the integration
backend, and channel-level QC for multichannel electrophysiology
(:func:`channel_qc_table`, :func:`bridged_channel_pairs`,
:func:`erp_channel_outliers`), and open-set individual re-identification from
images (:func:`reid_embed`, :func:`reid_local_verify`,
:func:`reid_open_set_cluster`).
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
                    reid_select_threshold, reid_open_set_cluster)

__all__ = ["cross_species_align", "bonsai", "bandpass_filter", "channel_qc_table",
           "bridged_channel_pairs", "erp_channel_outliers",
           "repair_reject_epochs", "channel_erps", "trigger_onsets",
           "fit_choice_model", "choice_model_loglik",
           "choice_model_forward_validation", "export_choice_predictor",
           "load_reid_backbone", "reid_embed", "reid_view_center", "reid_candidate_pairs",
           "reid_local_verify", "reid_fuse_similarity", "reid_pair_auc",
           "reid_select_threshold", "reid_open_set_cluster"]
