r"""Epoch-level artifact handling and ERP averaging with per-channel repair.

``repair_reject_epochs``
    per epoch × channel peak-to-peak; a channel over the *repair* limit is
    left out of that epoch (if at most ``max_repaired`` channels are), and the
    epoch is rejected when any remaining checked channel exceeds the
    *rejection* limit.
``channel_erps``
    per-condition ERP of every channel over accepted epochs, leaving out the
    epochs in which that channel was repaired; group ERPs as the mean of the
    member channels' ERPs (not a mean over per-epoch group averages).
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .._registry import register_function

__all__ = ["repair_reject_epochs", "channel_erps"]


@register_function(
    aliases=["伪迹剔除", "通道修复", "epoch rejection", "epoch channel repair",
             "artifact rejection with channel repair"],
    category="tl",
    description="Epoch x channel peak-to-peak repair (leave one channel out of one epoch) and "
                "whole-epoch rejection on the remaining checked channels; returns masks and "
                "per-channel repair counts over accepted epochs",
    requires={},
    produces={},
    auto_fix="none",
    examples=["rr = ov.tl.repair_reject_epochs(ep, ch_names, reject_uV=100, repair_uV=250, "
              "max_repaired=1, check_channels=scored_usable)"],
    related=["tl.channel_erps"],
)
def repair_reject_epochs(
    epochs,
    ch_names: Sequence[str],
    reject_uV: float,
    repair_uV: Optional[float] = None,
    max_repaired: int = 1,
    check_channels: Optional[Sequence[str]] = None,
    bad_channels: Optional[Sequence[str]] = None,
    dominant_frac: float = 0.5,
) -> Dict[str, object]:
    r"""Repair single-channel transients, then reject epochs.

    Parameters
    ----------
    epochs
        (n_epochs, n_channels, n_times), baseline-corrected, already without
        session-level bad channels' influence (bad channels may be present but
        must not be in ``check_channels``).
    ch_names
        Channel names for axis 1.
    reject_uV
        Peak-to-peak rejection limit applied to ``check_channels``.
    repair_uV
        Peak-to-peak limit above which a channel is repaired (left out) in that
        epoch; ``None`` disables repair.
    max_repaired
        An epoch with more than this many channels over ``repair_uV`` is not
        repairable; it is then judged by the rejection rule alone.
    check_channels
        Channels whose amplitude decides repair and rejection (default all).
    bad_channels
        Session-level unusable channels; always removed from
        ``check_channels`` (rejection must be computed **after** bad-channel
        exclusion — re-run this whenever the bad list changes, e.g. after the
        ERP-domain screen adds a channel).
    dominant_frac
        Warn when one checked channel alone exceeds the rejection limit in more
        than this fraction of epochs: that is a session-level defect left in
        the criterion, not epoch-level artifacts.

    Returns
    -------
    dict with ``accepted`` (n_epochs bool), ``repaired`` (n_epochs × n_channels
    bool, only within accepted epochs), ``repair_counts`` (channel → number of
    accepted epochs in which it was repaired) and ``ptp``.
    """
    ep = np.asarray(epochs, dtype=np.float64)
    names = list(ch_names)
    if ep.ndim != 3 or ep.shape[1] != len(names):
        raise ValueError("epochs must be (n_epochs, n_channels, n_times) matching ch_names")
    ptp = np.ptp(ep, axis=-1)
    chk = np.zeros(len(names), dtype=bool)
    chk[[names.index(c) for c in (check_channels if check_channels is not None else names)]] = True
    for c in (bad_channels or []):
        if c in names:
            chk[names.index(c)] = False
    if repair_uV is not None:
        cand = (ptp > repair_uV) & chk[None, :]
        k = cand.sum(axis=1)
        repairable = (k > 0) & (k <= int(max_repaired))
        rep = cand & repairable[:, None]
    else:
        rep = np.zeros_like(ptp, dtype=bool)
    p = np.where(rep, 0.0, ptp)
    accepted = ~(p[:, chk] > reject_uV).any(axis=1) if chk.any() else np.ones(len(ep), bool)
    rep = rep & accepted[:, None]
    counts = {names[j]: int(rep[:, j].sum()) for j in range(len(names)) if rep[:, j].any()}
    over_frac = {names[j]: float((ptp[:, j] > reject_uV).mean()) for j in range(len(names)) if chk[j]}
    warnings = [f"{c} exceeds the rejection limit in {f:.0%} of epochs — likely a session-level "
                f"bad channel left in the rejection criterion"
                for c, f in over_frac.items() if f > dominant_frac]
    if len(ep) and (~accepted).mean() > dominant_frac:
        warnings.append(f"{(~accepted).mean():.0%} of epochs rejected — check the bad-channel list "
                        "and that only usable checked channels decide rejection")
    return {"accepted": accepted, "repaired": rep, "repair_counts": counts, "ptp": ptp,
            "over_limit_frac": over_frac, "checked_channels": [names[j] for j in np.flatnonzero(chk)],
            "warnings": warnings}


@register_function(
    aliases=["ERP 平均", "通道组 ERP", "channel ERP average", "group ERP",
             "event-related potential average"],
    category="tl",
    description="Per-condition, per-channel ERPs over accepted epochs excluding each channel's "
                "repaired epochs; channel-group ERPs as the mean of member channel ERPs",
    requires={},
    produces={},
    auto_fix="none",
    examples=["erps = ov.tl.channel_erps(ep, ch_names, conditions, rr['accepted'], rr['repaired'], "
              "groups={'roi': usable_roi_channels})"],
    related=["tl.repair_reject_epochs"],
)
def channel_erps(
    epochs,
    ch_names: Sequence[str],
    conditions,
    accepted=None,
    repaired=None,
    groups: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, object]:
    r"""Average epochs per condition and channel, honouring per-channel repairs.

    A channel's ERP is the mean over accepted epochs **excluding** the epochs
    in which that channel was repaired. A group ERP is the mean of its member
    channels' ERPs.

    Returns
    -------
    dict with ``channel`` → {condition: (n_channels, n_times) array},
    ``n_epochs`` → {condition: per-channel epoch counts (Series)} and, if
    ``groups`` is given, ``group`` → {group: {condition: (n_times,) array}}.
    """
    ep = np.asarray(epochs, dtype=np.float64)
    names = list(ch_names)
    cond = np.asarray(conditions)
    n_ep = ep.shape[0]
    acc = np.ones(n_ep, bool) if accepted is None else np.asarray(accepted, bool)
    rep = np.zeros(ep.shape[:2], bool) if repaired is None else np.asarray(repaired, bool)
    use = acc[:, None] & ~rep                                    # (epochs, channels)
    out_ch, out_n, out_g = {}, {}, {}
    for c in pd.unique(cond):
        m = (cond == c)[:, None] & use
        n = m.sum(axis=0)
        s = np.einsum("ec,ect->ct", m.astype(float), np.nan_to_num(ep))
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = s / n[:, None]
        out_ch[c] = avg
        out_n[c] = pd.Series(n, index=names)
    if groups:
        for g, members in groups.items():
            idx = [names.index(m) for m in members]
            out_g[g] = {c: (np.nanmean(out_ch[c][idx], axis=0) if idx else
                            np.full(ep.shape[-1], np.nan)) for c in out_ch}
    return {"channel": out_ch, "n_epochs": out_n, "group": out_g}
