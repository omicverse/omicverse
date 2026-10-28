r"""Tools (``ov.tl``) — analysis helpers that transform an AnnData.

Currently hosts the cross-species preprocessing extracted from
:mod:`omicverse.single` so it can be reused independently of the integration
backend, and channel-level QC for multichannel electrophysiology
(:func:`channel_qc_table`, :func:`bridged_channel_pairs`,
:func:`erp_channel_outliers`).
"""
from ._cross_species import cross_species_align
from ._bonsai import bonsai
from ._channel_qc import (bandpass_filter, channel_qc_table, bridged_channel_pairs,
                          erp_channel_outliers)
from ._erp import repair_reject_epochs, channel_erps
from ._trigger import trigger_onsets

__all__ = ["cross_species_align", "bonsai", "bandpass_filter", "channel_qc_table",
           "bridged_channel_pairs", "erp_channel_outliers",
           "repair_reject_epochs", "channel_erps", "trigger_onsets"]
