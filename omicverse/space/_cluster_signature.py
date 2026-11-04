r"""Cluster-level marker statistics and caller-defined signature scoring.

For panels of a few dozen markers (CODEX, MIBI, IMC, CyCIF, multiplex IF,
CyTOF) cluster labels are decided from a handful of markers per label.  This
module holds no biology: it computes the statistics a gating decision needs
and scores **signatures supplied by the caller** (``{label: {marker: weight}}``)
in a reproducible way.

``cluster_marker_profile``
    per-cluster means, positive fractions, cross-cluster z-scores and the
    markers on which each cluster stands out.
``score_cluster_signatures``
    scores the caller's signatures per cluster, assigns labels (presence test,
    best contrast, margin), resolves child labels inside a parent by contrast
    between that parent's clusters, and optionally assigns clusters that
    support no signature by their spatial neighbourhood.
``cluster_neighbor_composition``
    for each cluster, the label / cluster composition of its cells'
    neighbourhoods.
"""

from __future__ import annotations

import re
import warnings
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._registry import register_function

# the open / fixed review threshold on the winner's lineage evidence: the
# dataset mean (z = 0).  Callers may only make it stricter.
REVIEW_BELOW = 0.0

__all__ = ["cluster_marker_profile", "score_cluster_signatures",
           "cluster_neighbor_composition"]


def _canon(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _resolve(names: Iterable[str], columns: Sequence[str]) -> Tuple[Dict[str, str], List[str]]:
    """Map requested marker names onto panel columns.

    Exact match after lower-casing and dropping non-alphanumerics; otherwise a
    unique panel column that extends the request (``CD8`` → ``CD8a``).
    """
    canon_cols = {}
    for c in columns:
        canon_cols.setdefault(_canon(c), c)
    found, missing = {}, []
    for n in names:
        k = _canon(n)
        if k in canon_cols:
            found[n] = canon_cols[k]
            continue
        ext = [c for ck, c in canon_cols.items() if ck.startswith(k) and len(ck) - len(k) <= 2]
        if len(ext) == 1:
            found[n] = ext[0]
        else:
            missing.append(n)
    return found, missing


def _otsu(x: np.ndarray, bins: int = 64) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0 or np.ptp(x) == 0:
        return np.inf
    hist, edges = np.histogram(x, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    c = np.cumsum(hist * mids)
    m0 = c / np.maximum(w0, 1)
    m1 = (c[-1] - c) / np.maximum(w1, 1)
    return float(mids[int(np.argmax((w0 * w1 * (m0 - m1) ** 2)[:-1]))])


def _cells_by_markers(data, cluster_key, markers, layer):
    if isinstance(data, pd.DataFrame):
        if cluster_key not in data.columns:
            raise KeyError(f"cluster column {cluster_key!r} not in the table")
        if markers is None:
            markers = [c for c in data.columns
                       if c != cluster_key and pd.api.types.is_numeric_dtype(data[c])]
        return data[list(markers)].astype(float), data[cluster_key].astype(str)
    if cluster_key not in data.obs:
        raise KeyError(f"cluster column {cluster_key!r} not in adata.obs")
    M = data.layers[layer] if layer is not None else data.X
    M = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    X = pd.DataFrame(M, index=data.obs_names, columns=data.var_names.astype(str))
    if markers is not None:
        X = X[list(markers)]
    return X.astype(float), data.obs[cluster_key].astype(str)


@register_function(
    aliases=["簇 marker 统计", "蛋白 panel 簇概况", "cluster marker profile",
             "protein panel cluster summary", "CODEX MIBI IMC cluster statistics"],
    category="space",
    description="Per-cluster means, positive fractions (per-marker Otsu), cross-cluster "
                "z-scores and stand-out markers for a cells x markers table "
                "(CODEX / MIBI / IMC / CyCIF / CyTOF)",
    requires={},
    produces={},
    auto_fix="none",
    examples=["prof = ov.space.cluster_marker_profile(df, 'cluster')",
              "prof.attrs['standout']['3']  # markers on which cluster 3 stands out"],
)
def cluster_marker_profile(
    data,
    cluster_key: str = "cluster",
    markers: Optional[Sequence[str]] = None,
    *,
    layer: Optional[str] = None,
    transform: str = "auto",
    cofactor: float = 5.0,
    scale_quantile: float = 0.999,
    standout_z: float = 1.5,
) -> pd.DataFrame:
    r"""Per-cluster summary of one marker panel.

    Run once per dataset / panel: intensities are not comparable across
    platforms.

    Arguments:
        data: cells × markers ``DataFrame`` containing ``cluster_key`` or an
            ``AnnData`` (markers in ``var``, cluster in ``obs``).
        cluster_key: cluster id column.
        markers: marker columns (default: every numeric column except the key).
        layer: AnnData layer instead of ``X``.
        transform: ``'auto'`` applies ``arcsinh(x / cofactor)`` when the values
            look like raw counts (``scale_quantile`` quantile > 20), ``'arcsinh'``
            forces it, ``'none'`` skips it.  Every marker is then divided by its
            own ``scale_quantile`` quantile and clipped to [0, 1], so markers
            with different dynamic ranges can be compared.
        standout_z: z-score above which a marker is listed as a stand-out of a
            cluster.

    Returns:
        ``DataFrame`` indexed by cluster with ``n_cells`` and, per marker,
        ``mean:<m>``, ``rel:<m>`` (mean / largest cluster mean of that marker),
        ``z:<m>`` (z of the cluster mean across clusters), ``pos:<m>``
        (fraction of cells above the marker's Otsu threshold) and ``zpos:<m>``.
        ``attrs['standout']`` maps each cluster to its stand-out markers
        (highest z first); ``attrs['markers']`` lists the markers used.
    """
    X, clusters = _cells_by_markers(data, cluster_key, markers, layer)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0)
    if transform == "arcsinh" or (transform == "auto"
                                  and np.nanquantile(X.to_numpy(), scale_quantile) > 20):
        X = np.arcsinh(X / cofactor)
    hi = X.quantile(scale_quantile).replace(0, np.nan)
    S = (X / hi).clip(0, 1).fillna(0.0)

    g = clusters.to_numpy()
    means = S.groupby(g).mean()
    thr = {c: _otsu(S[c].to_numpy()) for c in S.columns}
    pos = pd.DataFrame({c: (S[c].to_numpy() > thr[c]) for c in S.columns}, index=S.index).groupby(g).mean()
    z = (means - means.mean()) / (means.std(ddof=0) + 1e-9)
    zpos = (pos - pos.mean()) / (pos.std(ddof=0) + 1e-9)
    rel = means / means.max().replace(0, np.nan)

    parts = {"n_cells": pd.Series(g).value_counts().reindex(means.index).astype(int)}
    for c in S.columns:
        parts[f"mean:{c}"] = means[c]
        parts[f"rel:{c}"] = rel[c].fillna(0.0)
        parts[f"z:{c}"] = z[c]
        parts[f"pos:{c}"] = pos[c]
        parts[f"zpos:{c}"] = zpos[c]
    out = pd.DataFrame(parts, index=means.index)
    out.attrs["markers"] = list(S.columns)
    out.attrs["standout"] = {
        cl: [m for m in z.loc[cl].sort_values(ascending=False).index if z.at[cl, m] >= standout_z]
        for cl in z.index
    }
    return out


def _sig_score(prof, cmap, sig, cl, min_weight=0.0, norm="l1", stat="z", neg="same"):
    """Signature score of one cluster.

    ``stat='z'``: cross-cluster z-scores; ``stat='rel'``: cluster mean relative
    to the marker's brightest cluster (0..1), so the score reads as the
    weighted fraction of the signature the cluster expresses.  ``norm``:
    ``'l1'`` divides by the total positive weight (weighted mean), ``'l2'`` by
    the weight vector's norm (matched filter); the denominator is at least
    ``min_weight``.  ``neg='whole'`` scales the negative part by the whole
    signature's mass (positive + negative), so a signature with little
    positive evidence cannot be blown up by its negatives.
    """
    sp, sn, wpos, wneg, w2 = 0.0, 0.0, 0.0, 0.0, 0.0
    for m, w in sig.items():
        if m.startswith("pos:"):
            kind, name = ("zpos" if stat == "z" else "pos", m[4:])
        else:
            kind, name = (stat, m)
        col = cmap.get(name)
        if col is None:
            continue
        x = prof.at[cl, f"{kind}:{col}"]
        if kind == "pos":                       # positive fraction, relative to the best cluster
            top = prof[f"pos:{col}"].max()
            x = x / top if top > 0 else 0.0
        if w > 0:
            sp += w * x
            wpos += w
        else:
            sn += -w * x
            wneg += -w
        w2 += w * w
    if wpos <= 0:
        return np.nan
    den = np.sqrt(w2) if norm == "l2" else wpos
    den = max(den, min_weight)
    if neg == "whole":
        return sp / den - sn / max(wpos + wneg, min_weight)
    return (sp - sn) / den


def _presence(prof, cmap, sig, cl, defining=None, standout_z=1.5):
    """Largest relative mean among a signature's positive markers.  With
    ``defining`` (a fraction of the largest positive weight, or the list of
    the label's defining markers), the largest relative mean among the
    defining markers the panel measures; 0 when the panel measures no more
    than half of them (the label is not identifiable on this panel)."""
    pos = {m: w for m, w in sig.items() if w > 0 and not m.startswith("pos:")}
    on = {m: w for m, w in pos.items() if m in cmap}
    if not on:
        return np.nan
    need = []
    if isinstance(defining, (int, float)):
        top = max(pos.values())
        need = [m for m, w in pos.items() if w >= defining * top]
    elif defining is not None:
        need = [m for m in defining if m in pos]
        if not need:
            # no defining marker on this panel: identifiable only where the label's
            # best measured marker is a stand-out of the cluster
            best = max(on, key=on.get)
            if prof.at[cl, f"z:{cmap[best]}"] < standout_z:
                return 0.0
            return prof.at[cl, f"rel:{cmap[best]}"]
    if not need:
        return max(prof.at[cl, f"rel:{cmap[m]}"] for m in on)
    need = {_canon(m): m for m in need}               # naming variants of one marker count once
    measured = [m for m in need.values() if m in cmap]
    if 2 * len(measured) <= len(need):
        return 0.0
    return max(prof.at[cl, f"rel:{cmap[m]}"] for m in measured)


def _presence_bg(prof, cmap, sig, cl, weighted=False, baseline=False):
    """Largest ``rel`` among the signature's positive markers; with
    ``weighted`` each marker's value is scaled by its weight relative to the
    signature's strongest marker, so a weakly supported marker cannot make a
    label eligible on its own.  With ``baseline`` the value is measured from
    the marker's background (median of the marker-negative clusters, see
    ``_background``) instead of from zero, so dim spill-over shared by many
    clusters does not count as presence."""
    pos = [(m, w) for m, w in sig.items() if w > 0 and not m.startswith("pos:") and m in cmap]
    if not pos:
        return np.nan
    wmax = max(w for _, w in pos)
    vals = []
    for m, w in pos:
        col = cmap[m]
        if baseline:
            x = prof[f"mean:{col}"]
            lo, hi = _background(x), float(x.max())
            v = (float(x[cl]) - lo) / (hi - lo) if hi > lo else 0.0
            v = min(max(v, 0.0), 1.0)
        else:
            v = prof.at[cl, f"rel:{col}"]
        vals.append(v * (w / wmax if weighted else 1.0))
    return max(vals)


def _background(x: pd.Series) -> float:
    """Background level of a marker across clusters: the median cluster mean,
    unless the median cluster is itself on the bright side of the Otsu split
    of the cluster means (the marker is carried by most clusters) — then the
    median of the dim clusters.  ``0`` when there is no dim mode at all."""
    v = x.to_numpy(dtype=float)
    if v.size < 3 or np.ptp(v) == 0:
        return float(np.min(v)) if v.size else 0.0
    t = _otsu(v, bins=32)
    med = float(np.median(v))
    if med <= t:
        return med
    low = v[v <= t]
    bg = float(np.median(low)) if low.size else 0.0
    return bg if bg < 0.5 * float(np.max(v)) else 0.0


def _label_shares(nc_by_cluster: pd.DataFrame, label: Mapping[str, str]) -> pd.DataFrame:
    """Neighbourhood label shares per cluster from a cluster-level neighbour
    composition, leaving the cluster's own cells out (a cluster cannot vote
    for its own label)."""
    rows = {}
    for c in nc_by_cluster.index:
        row = nc_by_cluster.loc[c].drop(labels=[c], errors="ignore")
        tot = float(row.sum())
        acc: Dict[str, float] = {}
        if tot > 0:
            for c2, v in row.items():
                lab = label.get(c2)
                if lab is not None:
                    acc[lab] = acc.get(lab, 0.0) + float(v) / tot
        rows[c] = acc
    return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)


def _contradiction(prof, cmap, signatures, stat="rel", expect_frac=0.0, skip=()):
    """For every cluster and label, the largest fraction of another
    signature's positive evidence that lies on markers the label does not
    expect."""
    kind = "rel" if stat == "rel" else "z"
    pos = {}
    for lab, sig in signatures.items():
        p = {cmap[m]: w for m, w in sig.items()
             if w > 0 and not m.startswith("pos:") and m in cmap}
        pos[lab] = p
    expect = {lab: {m for m, w in p.items() if p and w >= expect_frac * max(p.values())}
              for lab, p in pos.items()}
    X = {m: prof[f"{kind}:{m}"].clip(lower=0) for p in pos.values() for m in p}
    out = pd.DataFrame(0.0, index=prof.index, columns=list(signatures))
    for lab in signatures:
        best = pd.Series(0.0, index=prof.index)
        for other, p in pos.items():
            if other == lab or not p or other in skip:
                continue
            tot = sum(p.values())
            extra = [m for m in p if m not in expect[lab]]
            if not extra:
                continue
            r = sum(p[m] * X[m] for m in extra) / tot
            best = np.maximum(best, r)
        out[lab] = best
    return out


@register_function(
    aliases=["簇签名打分", "按 marker 签名注释簇", "score cluster signatures",
             "gating by marker signatures", "protein panel cluster annotation"],
    category="space",
    description="Score caller-supplied marker signatures ({label: {marker: weight}}) on "
                "cluster profiles: presence test, best contrast with margin, parent/child "
                "granularity by within-parent contrast, neighbourhood assignment of "
                "clusters that support no signature",
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "prof = ov.space.cluster_marker_profile(df, 'cluster')",
        "res = ov.space.score_cluster_signatures(prof, signatures, parents=parents)",
    ],
)
def score_cluster_signatures(
    profile: pd.DataFrame,
    signatures: Mapping[str, Mapping[str, float]],
    *,
    parents: Optional[Mapping[str, str]] = None,
    polar: Union[Iterable[str], Mapping[str, Sequence[str]]] = (),
    presence: Optional[float] = None,
    presence_from: str = "auto",
    split_rule: str = "contrast",
    residual_label: Optional[str] = None,
    neighbor_composition: Optional[pd.DataFrame] = None,
    residual_alternatives: Sequence[str] = (),
    min_weight: float = 1.0,
    defining: Union[None, float, Mapping[str, Sequence[str]]] = None,
    norm: str = "l2",
    children_support_parent: bool = True,
    lineage: Optional[Mapping[str, str]] = None,
    n_candidates: int = 5,
    defining_check: Optional[Tuple[float, float]] = (0.5, 1.5),
    promote_min: float = 1e-9,
    review_below: Optional[float] = 0.0,
    stat: str = "z",
    weighted_presence: bool = True,
    residual_ratio: float = 2.0,
    contrast: float = 0.0,
    weak_fraction: float = 0.25,
    shrink: float = 0.0,
    neg: str = "whole",
    split_gap: bool = True,
    rel_from: str = "background",
    peak: float = 0.5,
    residual_to_testable: bool = False,
) -> pd.DataFrame:
    r"""Assign cluster labels from caller-defined marker signatures.

    The caller owns all biology: which labels exist, which markers support
    (positive weight) or argue against (negative weight) each label, and how
    labels nest.  Prefix a marker with ``pos:`` to use its positive fraction
    instead of its mean (useful for markers that are dim on all cells of a
    population but present on many of them).

    Arguments:
        profile: output of :func:`cluster_marker_profile`.
        signatures: ``{label: {marker: weight}}``.  Marker names are matched
            to panel columns case- and punctuation-insensitively; markers absent
            from the panel are ignored and reported in ``attrs['unmatched']``.
        parents: ``{child_label: parent_label}``.  Top-level labels (no parent)
            compete in the first pass; children are resolved inside their
            parent in the second.
        polar: parents whose clusters must each take one of the children (a
            population with no resting state); or ``{parent: [children]}``
            naming opposed poles (e.g. ``kb['polar']`` from
            :func:`markerdb_signatures`): when the difference between the
            two poles separates the parent's clusters at least as much as
            any single child does, the clusters leaning furthest to each
            pole take it; otherwise, when the contrast rule promotes a
            cluster to one pole, the remaining member leaning furthest to
            the opposite pole (opposite minus promoted pole score, larger
            than the promoted cluster's) takes it.  Other parents follow
            ``split_rule``.
        presence: a top-level label is eligible for a cluster only if one of
            its positive markers reaches ``presence`` of the way from the
            ``presence_from`` level to that marker's largest cluster mean
            (weighted, see ``weighted_presence``).  ``None`` (default): 0.2
            from background, 0.3 from zero / defining markers.
        presence_from: ``'background'`` measures presence from the marker's
            background level (median cluster mean, or the median of the dim
            mode when most clusters carry the marker), so dim spill-over is
            not presence; ``'zero'`` measures from 0; ``'defining'`` (needs a
            ``defining`` mapping) uses the label's measured defining markers
            from 0.  ``'auto'`` (default): ``'background'`` when a
            ``residual_label`` is given (presence decides who escapes the
            label of exclusion), otherwise ``'defining'`` when ``defining``
            is given, else ``'zero'``.
        split_rule: children are only considered when they carry a positive
            panel marker weighted above the parent's weight for it.  For
            non-polar parents, ``'contrast'`` keeps the parent
            label when the parent has a single cluster and, when the clustering
            split the parent into several clusters, promotes the most extreme
            cluster along the child axis that separates them best;
            ``'best'`` gives every cluster its best child with positive score;
            ``'none'`` keeps parents.
        residual_label: label for clusters that support no top-level
            signature (``None`` → ``'unassigned'``).
        neighbor_composition: output of :func:`cluster_neighbor_composition`
            computed with the first-pass labels; when given together with
            ``residual_alternatives``, a residual cluster takes the
            alternative its cells are surrounded by when that share exceeds
            the share of residual-labelled neighbours.
        residual_alternatives: labels a residual cluster may take by
            neighbourhood.
        min_weight: a signature's score is its weighted marker sum divided by
            ``max(total positive weight, min_weight)``.  With weights scaled
            so that a label's best-supported marker weighs 1 (as
            :func:`markerdb_signatures` returns them), ``min_weight=1``
            shrinks labels whose panel markers carry less evidence than one
            defining marker, instead of inflating a weak, incidental marker
            to full strength.
        defining: ``{label: [markers]}`` (e.g. ``kb['defining']`` from
            :func:`markerdb_signatures`) or a fraction (markers whose weight
            is at least that fraction of the label's largest).  A top-level
            label is then eligible only if the panel measures more than half
            of its defining markers and one of them reaches ``presence``.  ``None``: any
            positive marker reaching ``presence`` suffices.
        norm: first-pass score: ``'l1'`` divides the weighted sum by the
            total positive weight (a weighted mean); ``'l2'`` by the weight
            vector's norm (a matched filter: agreeing secondary markers add
            evidence instead of diluting the defining one).  Splits always
            compare children by weighted mean, so a cluster bright on every
            marker does not go to the child that lists the most markers.
        children_support_parent: a parent's first-pass score (and presence
            test) is the best of its own and its descendants' (a cluster that
            fits a subtype fits its lineage) — with ``defining`` given, only
            in clusters where one of the parent's own defining markers is at
            or above the cross-cluster mean.
        lineage: optional ``{cluster: label}`` replacing the first pass for
            those clusters (e.g. after reviewing ``top_label``).  A child
            label is lifted to its top-level ancestor, so granularity is
            still decided by the split rules; a label without a signature
            (e.g. ``'unassigned'``) is kept as given.
        defining_check: ``(low, high)`` z-scores, used with a ``defining``
            mapping: when none of the first-pass winner's own defining markers
            reaches ``low``, the candidate (within ``n_candidates``) with the
            strongest defining marker wins if that marker reaches ``high``
            (its defining evidence is what the cluster stands out on).  With
            ``children_support_parent`` a lineage's defining markers include
            its descendants' in that comparison.  E.g. ``(0.5, cluster_marker_profile's standout_z)``.
        promote_min: a child is only considered in a split when one of its
            positive panel markers outweighs the parent's weight for it by at
            least this much (with :func:`markerdb_signatures` weights, a
            fraction of the child's best-supported marker, e.g. ``0.1``).
        review_below: with a ``defining`` mapping, ``lineage_z`` is the
            first-pass winner's lineage evidence (z of its best defining
            marker, or of a descendant whose defining markers are all
            elevated); ``review`` is ``'open: ...'`` when it is below this
            value (z is relative to the across-cluster mean, so ``0`` = below
            the dataset mean) or unmeasured, or when another reported
            candidate has stronger lineage evidence (the numbers are given);
            otherwise ``'fixed'``.  ``lineage=`` changes of fixed clusters
            are always ignored.  The threshold is the dataset mean
            (``REVIEW_BELOW = 0``); a caller value is used only if it is
            stricter (smaller: opens fewer clusters) — ``None`` or larger
            values are ignored with a warning and recorded in
            ``attrs['review_note']``.  ``alt_lineage_z`` records the best
            other candidate's lineage evidence, so the default flags can be
            recomputed (:func:`write_cluster_labels` does).
        n_candidates: how many ranked first-pass candidates to report per
            cluster in ``candidates`` (label, score and the z-scores of its
            defining markers; with ``children_support_parent``, also its
            descendants' defining markers that are elevated, after "via").

        neg: ``'whole'`` (default) scales a signature's negative part by its
            whole mass (positive + negative); ``'same'`` divides it by the
            same denominator as the positive part.
        split_gap: in a contrast split of three or more clusters, every
            cluster above the largest gap along the child axis takes the
            child (when that gap is at least half the spread; otherwise none
            does); ``False``: only the most extreme cluster.
        stat: per-marker statistic the signatures are scored on: ``'rel'``
            (default; cluster mean / brightest cluster mean, 0..1, so the score
            is the weighted fraction of the signature a cluster expresses and
            quiet clusters score near 0 for every label) or ``'z'``
            (cross-cluster z-score; a cluster that is low on everything then
            favours signatures made mostly of negative markers).
        weighted_presence: scale each marker's ``rel`` by its weight relative
            to the signature's strongest marker in the presence test (default),
            so a weakly supported marker cannot make a label eligible alone.
        contrast: weight of the contradiction penalty.  A label is penalised
            by the strongest other signature's evidence on markers the label
            itself does not support at all (no positive weight),
            as a fraction of that other signature — so a cluster carrying a
            coherent multi-marker programme of another label is not taken by a
            label resting on one marker it shares with the background.  The
            ``score:<label>`` columns are the penalised scores; ``0`` disables.
        weak_fraction: signatures whose strongest marker on this panel
            weighs less than ``weak_fraction`` × the median signature's
            strongest marker do not compete
            in the first pass (listed in ``attrs['weak']``).
        shrink: pseudo-mass (as a fraction of the median signature's positive
            weight on the panel) added to every signature's denominator, so a
            label resting on one or two markers does not beat a label with a
            broad, coherent signature on equal relative intensity; ``0``
            disables.
        rel_from: ``'background'`` (default) — with ``stat='rel'`` each
            marker is scored from its background level (see
            ``presence_from``) to its brightest cluster, so spill-over and
            background at the level of most clusters scores ~0; ``'zero'``
            uses ``mean / max``.
        peak: (``stat='rel'`` only) weight of the *peak* term — the label's strongest marker
            (weighted, from background) in the cluster — blended with the
            signature fraction: ``(1 - peak) * fraction + peak * peak_term``.
            A lineage whose defining marker is at full intensity then beats
            a broad signature that is only moderately present (contact
            spill-over), unless the latter dominates outright.
        residual_to_testable: allow a residual cluster to be reassigned by
            neighbourhood to labels the panel can test positively (default
            ``False``: a cluster that lacks a label's positive markers is
            evidence against it, so only labels without a usable signature
            — weak, empty or absent — are neighbourhood alternatives).
        residual_ratio: a residual cluster takes an alternative only when the
            alternative's neighbourhood share is at least ``residual_ratio`` ×
            the residual label's share (default 2: a clear majority is needed
            to overturn the label of exclusion; ``1`` = simple majority).

    Defaults (one configuration for every panel): cross-cluster z statistic,
    l2 matched filter with ``min_weight=1`` (weights as
    :func:`markerdb_signatures` scales them, defining marker = 1), defining
    markers checked, children supporting their lineage, presence from the
    marker background when a label of exclusion is given and from the
    defining markers otherwise.  ``stat='rel'``, ``contrast``, ``shrink``
    are alternatives (documented above), off by default.

    A ``residual_label`` that also has a signature does not compete in the
    first pass: it is the label of exclusion (its signature is reported in
    ``score:<label>`` only).  ``neighbor_composition`` may be computed by
    label (``labels=`` first-pass labels) or by cluster (no ``labels``); in the
    second form the shares are recomputed from the first-pass labels and
    exclude the cluster's own cells.

    Returns:
        ``DataFrame`` indexed by cluster: ``label``, ``top_label``,
        ``score``, ``runner_up``, ``lineage_z``, ``review``, ``margin``,
        ``eligible``, ``candidates`` and ``reason``, plus
        ``score:<label>`` for every signature.  ``attrs['unmatched']`` lists
        signature markers not found in the panel.
    """
    parents = dict(parents or {})
    # polar: parents, or {parent: [children]} restricting the polar choice
    polar = {p: None for p in polar} if not isinstance(polar, Mapping) \
        else {p: list(k) for p, k in polar.items()}
    markers = profile.attrs.get("markers") or [c[5:] for c in profile.columns if c.startswith("mean:")]
    req = {m[4:] if m.startswith("pos:") else m for sig in signatures.values() for m in sig}
    cmap, unmatched = _resolve(sorted(req), markers)

    if stat not in ("z", "rel"):
        raise ValueError("stat must be 'z' or 'rel'")
    if stat == "rel" and rel_from == "background":
        profile = profile.copy()
        for col in set(cmap.values()):
            x = profile[f"mean:{col}"]
            lo, hi = _background(x), float(x.max())
            profile[f"rel:{col}"] = ((x - lo) / (hi - lo)).clip(0, 1) if hi > lo else 0.0
    scores = pd.DataFrame({lab: [_sig_score(profile, cmap, sig, cl, min_weight, norm, stat, neg)
                                 for cl in profile.index]
                           for lab, sig in signatures.items()}, index=profile.index)
    if stat == "rel" and peak:
        # peak term (same 0..1 scale as the rel fraction): the label's strongest
        # weighted marker from background
        pk = pd.DataFrame({lab: [_presence_bg(profile, cmap, sig, cl, weighted_presence, True)
                                 for cl in profile.index]
                           for lab, sig in signatures.items()}, index=profile.index)
        scores = (1 - peak) * scores + peak * pk.reindex(columns=scores.columns).fillna(0.0)
    rl = residual_label if residual_label is not None else "unassigned"
    # presence test: with a label of exclusion (residual_label given) clusters
    # must show markers above background to escape it; without one, every
    # cluster takes its best supported label, so presence only asks that a
    # label's defining marker is present at all
    pfrom = presence_from
    if pfrom == "auto":
        pfrom = "background" if residual_label is not None else (
            "defining" if isinstance(defining, Mapping) else "zero")
    if presence is None:            # each measurement has its own scale
        presence = 0.2 if pfrom == "background" else 0.3
    # labels with little positive evidence on this panel do not compete (they
    # would win on one marker); they stay available as residual alternatives
    mass = {lab: sum(w for m, w in sig.items() if w > 0 and (m[4:] if m.startswith("pos:") else m) in cmap)
            for lab, sig in signatures.items()}
    pos_mass = sorted(v for v in mass.values() if v > 0)
    # testability = the label's strongest panel marker (one excellent marker
    # suffices to call a label; many feeble ones do not)
    top_w = {lab: max([w for m, w in sig.items() if w > 0
                       and (m[4:] if m.startswith("pos:") else m) in cmap] + [0.0])
             for lab, sig in signatures.items()}
    pos_top = sorted(v for v in top_w.values() if v > 0)
    floor_ = weak_fraction * pos_top[len(pos_top) // 2] if pos_top else 0.0
    weak = [lab for lab, v in top_w.items() if v < floor_]
    if shrink and pos_mass:
        # evidence / (mass + m0), m0 = shrink x the median signature mass
        m0 = shrink * pos_mass[len(pos_mass) // 2]
        for lab, v in mass.items():
            if v > 0:
                scores[lab] = scores[lab] * v / (v + m0)
    if contrast:
        scores = scores - contrast * _contradiction(profile, cmap, signatures, stat,
                                                    skip=set(weak) | {rl})
    top = [l for l in signatures if l not in parents and l != rl and l not in weak]
    def _pres(l, cl):
        if defining is None or pfrom != "defining":
            v = _presence_bg(profile, cmap, signatures[l], cl, weighted_presence,
                             pfrom == "background")
        else:
            v = _presence(profile, cmap, signatures[l], cl,
                          defining.get(l) if isinstance(defining, Mapping) else defining,
                          defining_check[1] if defining_check is not None else 1.5)
        return v if v is not None and np.isfinite(v) else 0.0
    pres = pd.DataFrame({l: [_pres(l, cl) for cl in profile.index] for l in signatures},
                        index=profile.index)
    first = scores.copy()
    pres_first = pres.copy()
    desc: Dict[str, List[str]] = {}
    if children_support_parent:
        for ch in parents:
            x, seen = parents.get(ch), {ch}
            while x is not None and x not in seen:
                desc.setdefault(x, []).append(ch)
                seen.add(x)
                x = parents.get(x)
        for pa, ds in desc.items():
            if pa not in first.columns:
                continue
            ds = [d for d in ds if d in first.columns]
            # a child vouches for its lineage only where it fits (positive score)
            fits = {d: (scores[d] > 0) for d in ds}
            child = pd.concat([scores[d].where(fits[d]) for d in ds], axis=1).max(axis=1) \
                if ds else pd.Series(np.nan, index=first.index)
            cpres = pd.concat([pres[d].where(fits[d], 0.0) for d in ds], axis=1).max(axis=1) \
                if ds else pd.Series(0.0, index=first.index)
            ok = pd.Series(True, index=first.index)
            # and only where the lineage's own defining marker is at or above the mean
            own = defining.get(pa) if isinstance(defining, Mapping) else None
            own = [cmap[m] for m in (own or []) if m in cmap]
            if own:
                ok = profile[[f"z:{c}" for c in own]].max(axis=1) >= 0
            first[pa] = pd.concat([first[pa], child.where(ok)], axis=1).max(axis=1)
            pres_first[pa] = np.maximum(pres[pa], cpres.where(ok, 0.0))
    # review policy: the threshold is the dataset mean of the lineage evidence
    # (z = 0, REVIEW_BELOW); a caller value is honoured only if it is stricter
    # (opens fewer clusters); looser values and None are ignored and recorded
    rb = REVIEW_BELOW
    rb_note = ""
    if review_below is not None and np.isfinite(review_below) and review_below <= REVIEW_BELOW:
        rb = float(review_below)
    elif review_below is None or review_below != REVIEW_BELOW:
        rb_note = (f"review_below={review_below!r} ignored (only stricter values than "
                   f"{REVIEW_BELOW} are accepted)")
        warnings.warn("score_cluster_signatures: " + rb_note)
    rows = {}
    for cl in profile.index:
        elig = [l for l in top if pres_first.at[cl, l] >= presence and np.isfinite(first.at[cl, l])]
        if not elig:
            rows[cl] = dict(top_label=None, score=np.nan, runner_up=None, margin=np.nan,
                            lineage_z=np.nan, alt_lineage_z=np.nan, review="open",
                            eligible="", candidates="", reason="no signature present")
            continue
        ranked = sorted(elig, key=lambda l: -first.at[cl, l])

        def _own_z(l):
            ms = [cmap[m] for m in {_canon(m): m for m in (defining.get(l, []) if isinstance(
                defining, Mapping) else []) if m in cmap}.values()]
            return ms

        def _lin_z(l, sub=True):
            """lineage evidence: the best of the label's own defining markers, or
            of a descendant whose defining markers are all elevated (min)"""
            own = _own_z(l)
            vals = [max(profile.at[cl, f"z:{c}"] for c in own)] if own else []
            if sub and children_support_parent:
                for d in desc.get(l, []):
                    ms = _own_z(d)
                    if ms:
                        vals.append(min(profile.at[cl, f"z:{c}"] for c in ms))
            return max(vals) if vals else np.nan

        def _zs(l, k=3):
            ms = defining.get(l, []) if isinstance(defining, Mapping) else \
                [m for m, w in signatures[l].items() if w > 0]
            ms = list({_canon(m): m for m in ms if m in cmap}.values())
            if not ms:
                ms = sorted((m for m, w in signatures[l].items() if w > 0 and m in cmap),
                            key=lambda m: -signatures[l][m])[:2]
            return ", ".join(f"{cmap[m]} z={profile.at[cl, 'z:' + cmap[m]]:.1f}" for m in ms[:k])

        def _ev(l):
            txt = _zs(l)
            if children_support_parent and isinstance(defining, Mapping):
                # the lineage's evidence includes its descendants' defining markers
                sub = []
                for d in desc.get(l, []):
                    ms = _own_z(d)
                    if ms:
                        z = min(profile.at[cl, f"z:{c}"] for c in ms)
                        sub.append((z, f"{d}: {_zs(d, 2)}"))
                sub = [t for z, t in sorted(sub, reverse=True)[:2] if z >= 0.5]
                if sub:
                    txt += " | via " + "; ".join(sub)
            return f"{l} {first.at[cl, l]:.2f} [{txt}]"
        cands = " | ".join(_ev(l) for l in ranked[:n_candidates])
        b = ranked[0]
        note = ""
        if defining_check is not None and isinstance(defining, Mapping):
            _dz = _lin_z
            lo, hi = defining_check
            if np.isfinite(_dz(b, False)) and _dz(b, False) < lo:
                ev = {l: _dz(l) for l in ranked[:n_candidates]}
                ev = {l: v for l, v in ev.items() if np.isfinite(v)}
                a = max(ev, key=ev.get) if ev else b
                if a != b and ev[a] >= hi:
                    note = f"; defining-marker check: {b} (z {_dz(b, False):.1f}) -> {a} (z {ev[a]:.1f})"
                    ranked = [a] + [l for l in ranked if l != a]
                    b = ranked[0]
        r = ranked[1] if len(ranked) > 1 else None
        lz = _lin_z(b) if isinstance(defining, Mapping) else np.nan
        alt = {l: _lin_z(l) for l in ranked[1:n_candidates]} if isinstance(defining, Mapping) else {}
        alt = {l: v for l, v in alt.items() if np.isfinite(v)}
        ba = max(alt, key=alt.get) if alt else None
        if not isinstance(defining, Mapping):
            rv = "fixed"
        elif not np.isfinite(lz) or lz < rb:
            rv = f"open: winner evidence z={lz:.1f} below {rb}"
        elif ba is not None and alt[ba] > lz:
            rv = f"open: {ba} evidence z={alt[ba]:.1f} > winner z={lz:.1f}"
        else:
            rv = "fixed"
        rows[cl] = dict(top_label=b, score=first.at[cl, b], runner_up=r, lineage_z=lz,
                        alt_lineage_z=alt[ba] if ba is not None else np.nan, review=rv,
                        margin=first.at[cl, b] - first.at[cl, r] if r else np.nan,
                        eligible=",".join(ranked), candidates=cands,
                        reason=f"{b} {first.at[cl, b]:.2f}" + (f" > {r} {first.at[cl, r]:.2f}" if r else "")
                        + note)
    out = pd.DataFrame.from_dict(rows, orient="index").reindex(profile.index)
    fixed = {}
    forced: List[str] = []
    for cl, lab in (lineage or {}).items():
        cl = str(cl) if str(cl) in out.index else cl
        if cl not in out.index:
            continue
        x = lab
        while x in parents:
            x = parents[x]
        if str(out.at[cl, "review"]) == "fixed" and x != out.at[cl, "top_label"]:
            out.at[cl, "reason"] = (f"{out.at[cl, 'reason']}; lineage change to {lab!r} ignored: "
                                    f"evidence for {out.at[cl, 'top_label']} is z={out.at[cl, 'lineage_z']:.1f}"
                                    f" (>= {rb})")
            continue
        x, seen = lab, set()
        while x in parents and x not in seen:
            seen.add(x)
            x = parents[x]
        if lab in signatures:
            out.at[cl, "top_label"] = x
            out.at[cl, "reason"] = f"lineage given: {lab}" + (f" -> {x}" if x != lab else "")
        else:
            fixed[cl] = lab
            out.at[cl, "top_label"] = None
            out.at[cl, "reason"] = f"given: {lab}"
    label = out["top_label"].copy()

    # residual clusters
    resid = [c for c in label.index[label.isna()] if c not in fixed]
    label[resid] = rl
    nc = neighbor_composition
    by_cluster = nc is not None and len(nc.columns) > 0 and \
        set(map(str, nc.columns)) <= set(map(str, profile.index))
    if by_cluster:
        nc = nc.copy()
        nc.index = nc.index.astype(str)
        nc.columns = nc.columns.astype(str)
    if len(resid) and nc is not None and residual_alternatives:
        shares = _label_shares(nc, label.to_dict()) if by_cluster else nc
        for c in [c for c in resid if str(c) in shares.index.astype(str)]:
            row = shares.loc[str(c)] if by_cluster else shares.loc[c]
            alts = [a for a in residual_alternatives
                    if residual_to_testable or a not in signatures or a in weak]
            if not alts:
                out.at[c, "reason"] = "no signature present; no untestable alternative"
                continue
            share = {a: float(row.get(a, 0.0)) for a in alts}
            own = float(row.get(rl, 0.0))
            a = max(share, key=share.get)
            if share[a] > 0 and share[a] >= residual_ratio * own and share[a] > own:
                label[c] = a
                out.at[c, "reason"] = (f"no signature present; neighbours {share[a]:.2f} {a} "
                                       f"vs {own:.2f} {rl}")
            else:
                out.at[c, "reason"] = (f"no signature present; neighbours {own:.2f} {rl} "
                                       f"vs {share[a]:.2f} {a}")

    # children inside parents (repeat for nested levels)
    children: Dict[str, List[str]] = {}
    for ch, pa in parents.items():
        children.setdefault(pa, []).append(ch)
    # splits compare children by weighted mean (l1): a cluster bright on
    # everything should not go to the child that simply lists more markers
    ss = scores if norm == "l1" else pd.DataFrame(
        {lab: [_sig_score(profile, cmap, sig, cl, min_weight, "l1", stat, neg) for cl in profile.index]
         for lab, sig in signatures.items()}, index=profile.index)
    split_done: set = set()
    for _ in range(4):
        changed = False
        for pa, kids in children.items():
            members = [c for c in label.index if label[c] == pa]
            kids = [k for k in kids if np.isfinite(ss[k]).any()]
            # a child can only be promoted on evidence of its own: a positive
            # panel marker weighted at least promote_min above the parent's
            if pa in signatures:
                pw = signatures[pa]
                kids = [k for k in kids if any(w - pw.get(m, 0.0) >= promote_min and m in cmap
                                               for m, w in signatures[k].items() if w > 0)]
            if not members or not kids:
                continue
            if pa in polar and polar[pa] is None:
                if len(members) == 1:
                    best = max(kids, key=lambda k: ss.at[members[0], k])
                    label[members[0]] = best
                else:
                    zz = {k: (ss.loc[members, k] - ss.loc[members, k].mean())
                          / (ss.loc[members, k].std(ddof=0) + 1e-9) for k in kids}
                    for c in members:
                        label[c] = max(kids, key=lambda k: zz[k][c])
                changed = True
            elif split_rule == "best":
                for c in members:
                    k = max(kids, key=lambda k: ss.at[c, k])
                    if ss.at[c, k] > 0:
                        label[c] = k
                        changed = True
            elif split_rule == "contrast" and len(members) >= 2 and pa not in split_done:
                split_done.add(pa)          # a parent is split once
                spread = {k: ss.loc[members, k].max() - ss.loc[members, k].min() for k in kids}
                poles = [x for x in (polar.get(pa) or []) if x in kids]
                if len(poles) == 2:
                    # an opposed pair that separates the clusters at least as well as
                    # any single child: split along that axis (each extreme to its pole)
                    d = ss.loc[members, poles[0]] - ss.loc[members, poles[1]]
                    if d.max() - d.min() >= max(spread.values()):
                        c1, c2 = d.idxmax(), d.idxmin()
                        for c, kk in ((c1, poles[0]), (c2, poles[1])):
                            label[c] = kk
                            out.at[c, "reason"] = f"{out.at[c, 'reason']}; split of {pa} by poles: {kk}"
                        changed = True
                        continue
                k = max(spread, key=spread.get)
                v = ss.loc[members, k].sort_values(ascending=False)
                c = v.index[0]
                take = [c]
                if split_gap and len(members) > 2:
                    # every cluster above the largest gap along the child axis
                    gaps = v.values[:-1] - v.values[1:]
                    cut = int(np.argmax(gaps))
                    if spread[k] > 0 and gaps[cut] >= 0.5 * spread[k]:
                        take = list(v.index[:cut + 1])
                    else:
                        take = []
                for c_ in take:
                    label[c_] = k
                    out.at[c_, "reason"] = f"{out.at[c_, 'reason']}; split of {pa}: {k}"
                if not take:
                    changed = True
                    continue
                # opposed poles: the most extreme remaining member on the other pole takes it
                poles = polar.get(pa) or []
                if k in poles:
                    rest = [m for m in members if m not in take]
                    for k2 in [x for x in poles if x != k and x in scores.columns]:
                        if not rest:
                            break
                        d = ss.loc[rest, k2] - ss.loc[rest, k]       # relative pull to the other pole
                        c2 = d.idxmax()
                        if d[c2] > ss.at[c, k2] - ss.at[c, k]:
                            label[c2] = k2
                            out.at[c2, "reason"] = f"{out.at[c2, 'reason']}; split of {pa}: {k2} (opposite of {k})"
                            rest.remove(c2)
                changed = True
        if not changed:
            break
    for c, lab in fixed.items():
        label[c] = lab
    out.insert(0, "label", label)
    # lineage changes applied to clusters the review policy marks fixed
    out["lineage_forced"] = out.index.isin(forced)
    for lab in signatures:
        out[f"score:{lab}"] = scores[lab]
    out.attrs["unmatched"] = unmatched
    out.attrs["weak"] = weak
    out.attrs["review_below"] = rb
    if rb_note:
        out.attrs["review_note"] = rb_note
    out.attrs["marker_map"] = cmap
    return out


@register_function(
    aliases=["簇邻域组成", "邻域组成", "cluster neighbor composition", "neighbourhood composition"],
    category="space",
    description="For each cluster, the average composition (by label or cluster) of its cells' "
                "spatial neighbourhoods; radius defaults to 3x the median nearest-neighbour distance",
    requires={},
    produces={},
    auto_fix="none",
    examples=["nc = ov.space.cluster_neighbor_composition(xy, clusters, labels=first_pass, samples=df.sample_id)"],
)
def cluster_neighbor_composition(
    coords,
    clusters: Sequence,
    labels: Optional[Mapping[str, str]] = None,
    *,
    samples: Optional[Sequence] = None,
    radius: Optional[float] = None,
) -> pd.DataFrame:
    r"""Average neighbourhood composition per cluster.

    Arguments:
        coords: ``(n_cells, 2)`` coordinates.
        clusters: cluster id per cell.
        labels: optional ``{cluster: label}``; composition is then reported
            by label, otherwise by cluster.
        samples: image / ROI id per cell; neighbourhoods never cross samples.
        radius: neighbourhood radius (default: 3 × median nearest-neighbour
            distance, per sample).

    Returns:
        ``DataFrame`` clusters × categories; each row is the mean, over the
        cluster's cells, of the fraction of neighbours (self excluded) in each
        category.
    """
    from scipy.spatial import cKDTree

    xy = np.asarray(coords, dtype=float)
    cl = np.asarray(clusters).astype(str)
    cat = np.array([labels.get(c, c) if labels is not None else c for c in cl], dtype=object)
    cat = np.array(["<none>" if v is None else str(v) for v in cat])
    cats = sorted(set(cat))
    ci = {c: i for i, c in enumerate(cats)}
    code = np.array([ci[c] for c in cat])
    smp = np.zeros(len(cl)) if samples is None else np.asarray(samples)
    acc: Dict[str, np.ndarray] = {}
    cnt: Dict[str, int] = {}
    for s in np.unique(smp):
        idx = np.where(smp == s)[0]
        if idx.size < 3:
            continue
        tree = cKDTree(xy[idx])
        r = radius
        if r is None:
            d, _ = tree.query(xy[idx], k=2)
            r = 3.0 * float(np.median(d[:, 1]))
        for i, js in zip(idx, tree.query_ball_point(xy[idx], r=r)):
            js = [idx[j] for j in js if idx[j] != i]
            if not js:
                continue
            frac = np.bincount(code[js], minlength=len(cats)) / len(js)
            acc[cl[i]] = acc.get(cl[i], 0) + frac
            cnt[cl[i]] = cnt.get(cl[i], 0) + 1
    return pd.DataFrame({k: v / cnt[k] for k, v in acc.items()}, index=cats).T.sort_index()
