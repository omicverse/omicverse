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
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._registry import register_function

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


def _sig_score(prof, cmap, sig, cl):
    s, wpos = 0.0, 0.0
    for m, w in sig.items():
        kind, name = ("zpos", m[4:]) if m.startswith("pos:") else ("z", m)
        col = cmap.get(name)
        if col is None:
            continue
        s += w * prof.at[cl, f"{kind}:{col}"]
        wpos += max(w, 0.0)
    return s / wpos if wpos > 0 else np.nan


def _presence(prof, cmap, sig, cl):
    vals = [prof.at[cl, f"rel:{cmap[m]}"] for m, w in sig.items()
            if w > 0 and not m.startswith("pos:") and m in cmap]
    return max(vals) if vals else np.nan


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
    polar: Iterable[str] = (),
    presence: float = 0.3,
    split_rule: str = "contrast",
    residual_label: Optional[str] = None,
    neighbor_composition: Optional[pd.DataFrame] = None,
    residual_alternatives: Sequence[str] = (),
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
            population with no resting state).  Other parents follow
            ``split_rule``.
        presence: a top-level label is eligible for a cluster only if one of
            its positive markers reaches ``presence`` × that marker's largest
            cluster mean.
        split_rule: for non-polar parents, ``'contrast'`` keeps the parent
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

    Returns:
        ``DataFrame`` indexed by cluster: ``label``, ``top_label``,
        ``score``, ``runner_up``, ``margin``, ``eligible`` and ``reason``, plus
        ``score:<label>`` for every signature.  ``attrs['unmatched']`` lists
        signature markers not found in the panel.
    """
    parents = dict(parents or {})
    polar = set(polar)
    markers = profile.attrs.get("markers") or [c[5:] for c in profile.columns if c.startswith("mean:")]
    req = {m[4:] if m.startswith("pos:") else m for sig in signatures.values() for m in sig}
    cmap, unmatched = _resolve(sorted(req), markers)

    scores = pd.DataFrame({lab: [_sig_score(profile, cmap, sig, cl) for cl in profile.index]
                           for lab, sig in signatures.items()}, index=profile.index)
    top = [l for l in signatures if l not in parents]
    rows = {}
    for cl in profile.index:
        elig = [l for l in top if (_presence(profile, cmap, signatures[l], cl) or 0) >= presence
                and np.isfinite(scores.at[cl, l])]
        if not elig:
            rows[cl] = dict(top_label=None, score=np.nan, runner_up=None, margin=np.nan,
                            eligible="", reason="no signature present")
            continue
        ranked = sorted(elig, key=lambda l: -scores.at[cl, l])
        b = ranked[0]
        r = ranked[1] if len(ranked) > 1 else None
        rows[cl] = dict(top_label=b, score=scores.at[cl, b], runner_up=r,
                        margin=scores.at[cl, b] - scores.at[cl, r] if r else np.nan,
                        eligible=",".join(ranked),
                        reason=f"{b} {scores.at[cl, b]:.2f}" + (f" > {r} {scores.at[cl, r]:.2f}" if r else ""))
    out = pd.DataFrame.from_dict(rows, orient="index").reindex(profile.index)
    label = out["top_label"].copy()

    # residual clusters
    resid = label.index[label.isna()]
    rl = residual_label if residual_label is not None else "unassigned"
    label[resid] = rl
    if len(resid) and neighbor_composition is not None and residual_alternatives:
        nc = neighbor_composition
        quiet = [c for c in resid if c in nc.index]
        if quiet:
            for c in quiet:
                share = {a: float(nc.at[c, a]) if a in nc.columns else 0.0 for a in residual_alternatives}
                own = float(nc.at[c, rl]) if rl in nc.columns else 0.0
                a = max(share, key=share.get)
                if share[a] > own:
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
    for _ in range(4):
        changed = False
        for pa, kids in children.items():
            members = [c for c in label.index if label[c] == pa]
            kids = [k for k in kids if np.isfinite(scores[k]).any()]
            if not members or not kids:
                continue
            if pa in polar:
                if len(members) == 1:
                    best = max(kids, key=lambda k: scores.at[members[0], k])
                    label[members[0]] = best
                else:
                    zz = {k: (scores.loc[members, k] - scores.loc[members, k].mean())
                          / (scores.loc[members, k].std(ddof=0) + 1e-9) for k in kids}
                    for c in members:
                        label[c] = max(kids, key=lambda k: zz[k][c])
                changed = True
            elif split_rule == "best":
                for c in members:
                    k = max(kids, key=lambda k: scores.at[c, k])
                    if scores.at[c, k] > 0:
                        label[c] = k
                        changed = True
            elif split_rule == "contrast" and len(members) >= 2:
                spread = {k: scores.loc[members, k].max() - scores.loc[members, k].min() for k in kids}
                k = max(spread, key=spread.get)
                c = scores.loc[members, k].idxmax()
                label[c] = k
                changed = True
        if not changed:
            break
    out.insert(0, "label", label)
    for lab in signatures:
        out[f"score:{lab}"] = scores[lab]
    out.attrs["unmatched"] = unmatched
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
