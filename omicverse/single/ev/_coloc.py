"""Marker colocalization on the same vesicle for single-EV proteomics.

The defining advantage of single-extracellular-vesicle (single-EV) proteomics
over bulk EV proteomics is **single-vesicle resolution**: it can tell which
protein markers physically co-occur on the *same* individual vesicle, rather
than only which markers are present in a pooled population. This module
implements that EV-signature analysis:

* :func:`colocalization`      — pairwise (and multi-way) marker co-occurrence
  statistics: Jaccard index, odds ratio, observed-vs-expected co-positivity
  and BH-corrected Fisher's-exact / hypergeometric p-values.
* :func:`coexpression_network`— a marker-marker co-occurrence graph built from
  the colocalization statistics.
* :func:`protein_combinations`— enumeration of the multi-marker combinations
  carried by EVs and, given a condition, differentially-expressed protein
  combinations (DEPCs).
* :func:`colocalization_plot` — a colocalization heatmap.

All functions are pure Python (numpy / scipy / pandas), keyword-only beyond the
``adata`` argument, and registered with :func:`omicverse._registry.register_function`
under ``category="ev"``.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from ..._registry import register_function


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _require(modname: str, role: str):
    """Lazy-import an optional backend with an actionable error message."""
    import importlib

    try:
        return importlib.import_module(modname)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"{role} needs the '{modname}' package. Install it with: "
            f"pip install {modname}."
        ) from exc


def _dense(adata):
    """Return ``adata.X`` as a dense float numpy array."""
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=float)


def _value_type(adata, override: Optional[str]) -> str:
    """Resolve the EV value type from ``adata.uns['ev']`` or an override."""
    if override is not None:
        return override
    ev = adata.uns.get("ev", {}) if hasattr(adata, "uns") else {}
    return ev.get("value_type", "intensity")


def _positivity(adata, *, markers, value_type, threshold):
    """Boolean EV x marker positivity matrix restricted to ``markers``.

    Returns ``(B, names)`` — the boolean array and the resolved marker names.
    """
    var_upper = {str(v).upper(): v for v in adata.var_names}
    if markers is None:
        names = list(adata.var_names)
    else:
        names = []
        for m in markers:
            hit = var_upper.get(str(m).upper())
            if hit is None:
                raise KeyError(f"marker {m!r} not found in adata.var.")
            names.append(hit)
    idx = [adata.var_names.get_loc(n) for n in names]
    X = _dense(adata)[:, idx]
    if value_type == "binary":
        B = X > 0
    else:
        B = X > threshold
    return B, names


def _bh_adjust(pvals):
    """Benjamini-Hochberg FDR adjustment of a 1-D array of p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(ranked, 0, 1)
    return out


# ---------------------------------------------------------------------------
# colocalization
# ---------------------------------------------------------------------------
@register_function(
    aliases=["colocalization", "ev_coloc", "marker_cooccurrence",
             "共定位分析", "标志物共定位"],
    category="ev",
    description=(
        "Pairwise (and optionally multi-way) marker colocalization across "
        "single EVs — the analysis bulk EV proteomics cannot do. For every "
        "marker pair: co-positive EV count, Jaccard index, odds ratio, "
        "observed/expected co-positivity ratio and Fisher's-exact p-value "
        "(BH-corrected). Returns a tidy table plus symmetric matrices in "
        ".attrs."
    ),
    examples=[
        "res = ov.single.ev.colocalization(adata)",
        "res = ov.single.ev.colocalization(adata, markers=['CD9','CD63','CD81'])",
        "res = ov.single.ev.colocalization(adata, markers=tetra, max_order=3)",
        "jac = res.attrs['jaccard']",
    ],
    related=["single.ev.coexpression_network", "single.ev.protein_combinations",
             "single.ev.colocalization_plot"],
)
def colocalization(adata, *, markers=None, value_type: Optional[str] = None,
                   threshold: float = 0.0, max_order: int = 2,
                   min_count: int = 1, method: str = "fisher"):
    """Pairwise / multi-way marker colocalization across single EVs.

    Quantifies, for every marker pair (and optionally every higher-order
    combination), how often the markers are detected together on the *same*
    individual vesicle — and whether that co-occurrence exceeds chance.

    For a marker pair the 2x2 contingency table of EV positivity gives:

    * **Jaccard index** — ``n11 / (n11 + n10 + n01)``;
    * **odds ratio** — ``(n11 * n00) / (n10 * n01)``;
    * **observed/expected** — ``n11 / (p_a * p_b * N)`` where ``p_a``,
      ``p_b`` are the marginal positive fractions;
    * a **p-value** from Fisher's exact test or the hypergeometric tail.

    Parameters
    ----------
    adata
        EV x protein :class:`anndata.AnnData`.
    markers
        Markers to test. ``None`` uses every protein in ``var``.
    value_type
        ``'count'`` | ``'intensity'`` | ``'binary'``. ``None`` reads
        ``adata.uns['ev']['value_type']``.
    threshold
        Positivity cutoff for ``'count'`` / ``'intensity'`` matrices.
    max_order
        Largest combination order to enumerate. ``2`` (default) gives the
        pairwise table; ``3`` additionally enumerates marker triplets, etc.
        Statistics beyond pairs are restricted to the co-positive count,
        Jaccard index and observed/expected ratio.
    min_count
        Drop combinations whose co-positive EV count is below ``min_count``.
    method
        Pairwise p-value test — ``'fisher'`` (Fisher's exact, default) or
        ``'hypergeometric'`` (one-sided enrichment tail).

    Returns
    -------
    :class:`pandas.DataFrame`
        A tidy colocalization table — one row per marker combination with
        ``order``, ``markers``, ``n_copos``, ``jaccard``, ``obs_exp`` and,
        for pairs, ``odds_ratio``, ``pvalue`` and ``padj``. The symmetric
        pairwise ``jaccard``, ``odds_ratio`` and ``obs_exp`` matrices are
        attached under ``.attrs``.
    """
    from scipy.stats import fisher_exact, hypergeom

    if max_order < 2:
        raise ValueError("max_order must be >= 2.")
    vtype = _value_type(adata, value_type)
    B, names = _positivity(adata, markers=markers, value_type=vtype,
                           threshold=threshold)
    n_ev, n_mk = B.shape
    if n_mk < 2:
        raise ValueError("colocalization needs at least 2 markers.")
    Bf = B.astype(float)
    marg = Bf.sum(axis=0)               # per-marker positive count
    pmarg = marg / n_ev if n_ev else marg

    # Pairwise co-positive counts via the boolean co-occurrence matrix.
    co = Bf.T @ Bf                      # n_mk x n_mk co-positive counts

    jac = np.eye(n_mk)
    orat = np.full((n_mk, n_mk), np.nan)
    oe = np.full((n_mk, n_mk), np.nan)

    rows = []
    for i, j in combinations(range(n_mk), 2):
        n11 = co[i, j]
        n10 = marg[i] - n11
        n01 = marg[j] - n11
        n00 = n_ev - n11 - n10 - n01
        union = n11 + n10 + n01
        jaccard = n11 / union if union > 0 else 0.0
        # Haldane-Anscombe 0.5 correction keeps the odds ratio finite.
        odds = ((n11 + 0.5) * (n00 + 0.5)) / ((n10 + 0.5) * (n01 + 0.5))
        expected = pmarg[i] * pmarg[j] * n_ev
        obs_exp = (n11 / expected) if expected > 0 else np.nan
        if method == "fisher":
            _, pval = fisher_exact(
                [[int(n11), int(n10)], [int(n01), int(n00)]],
                alternative="greater",
            )
        elif method in ("hypergeometric", "hypergeom"):
            pval = hypergeom.sf(int(n11) - 1, n_ev, int(marg[i]),
                                int(marg[j]))
        else:
            raise ValueError(
                f"method must be 'fisher' or 'hypergeometric', got {method!r}."
            )
        jac[i, j] = jac[j, i] = jaccard
        orat[i, j] = orat[j, i] = odds
        oe[i, j] = oe[j, i] = obs_exp
        if n11 >= min_count:
            rows.append({
                "order": 2,
                "markers": f"{names[i]}+{names[j]}",
                "marker_list": (names[i], names[j]),
                "n_copos": int(n11),
                "jaccard": jaccard,
                "odds_ratio": odds,
                "obs_exp": obs_exp,
                "pvalue": float(pval),
            })

    # Higher-order combinations — count / Jaccard / obs-exp only.
    for order in range(3, max_order + 1):
        for combo in combinations(range(n_mk), order):
            allpos = B[:, list(combo)].all(axis=1)
            n_all = int(allpos.sum())
            if n_all < min_count:
                continue
            anypos = B[:, list(combo)].any(axis=1).sum()
            jaccard = n_all / anypos if anypos > 0 else 0.0
            expected = np.prod(pmarg[list(combo)]) * n_ev
            obs_exp = (n_all / expected) if expected > 0 else np.nan
            rows.append({
                "order": order,
                "markers": "+".join(names[c] for c in combo),
                "marker_list": tuple(names[c] for c in combo),
                "n_copos": n_all,
                "jaccard": jaccard,
                "odds_ratio": np.nan,
                "obs_exp": obs_exp,
                "pvalue": np.nan,
            })

    res = pd.DataFrame(rows)
    if not res.empty:
        pair_mask = res["order"] == 2
        padj = np.full(len(res), np.nan)
        if pair_mask.any():
            padj[pair_mask.values] = _bh_adjust(
                res.loc[pair_mask, "pvalue"].values
            )
        res["padj"] = padj
        res = res.sort_values(
            ["order", "n_copos"], ascending=[True, False]
        ).reset_index(drop=True)
    else:
        res = pd.DataFrame(columns=[
            "order", "markers", "marker_list", "n_copos", "jaccard",
            "odds_ratio", "obs_exp", "pvalue", "padj",
        ])

    res.attrs["jaccard"] = pd.DataFrame(jac, index=names, columns=names)
    res.attrs["odds_ratio"] = pd.DataFrame(orat, index=names, columns=names)
    res.attrs["obs_exp"] = pd.DataFrame(oe, index=names, columns=names)
    res.attrs["marker_names"] = names
    res.attrs["n_ev"] = n_ev
    return res


# ---------------------------------------------------------------------------
# coexpression_network
# ---------------------------------------------------------------------------
@register_function(
    aliases=["coexpression_network", "ev_coloc_network", "marker_network",
             "共表达网络", "标志物共定位网络"],
    category="ev",
    description=(
        "Build a marker-marker co-occurrence graph from single-EV "
        "colocalization statistics. Nodes are markers, edges connect marker "
        "pairs whose co-positivity passes a weight / significance cutoff. "
        "Returns a networkx graph (or an edge-list DataFrame)."
    ),
    examples=[
        "G = ov.single.ev.coexpression_network(adata)",
        "G = ov.single.ev.coexpression_network(adata, weight='odds_ratio', "
        "min_weight=2.0, padj_cutoff=0.05)",
        "edges = ov.single.ev.coexpression_network(adata, as_dataframe=True)",
    ],
    related=["single.ev.colocalization", "single.ev.colocalization_plot"],
)
def coexpression_network(adata, *, coloc=None, markers=None,
                         weight: str = "jaccard", min_weight: float = 0.1,
                         padj_cutoff: Optional[float] = 0.05,
                         as_dataframe: bool = False, **coloc_kwargs):
    """Marker-marker co-occurrence graph from single-EV colocalization.

    Turns the pairwise colocalization statistics into a network: every marker
    is a node, and an edge joins two markers whose co-occurrence weight (and,
    optionally, BH-adjusted p-value) passes the requested cutoffs.

    Parameters
    ----------
    adata
        EV x protein :class:`anndata.AnnData`.
    coloc
        A precomputed :func:`colocalization` result. ``None`` computes one on
        the fly (passing ``markers`` and ``**coloc_kwargs``).
    markers
        Markers to include — forwarded to :func:`colocalization` when
        ``coloc`` is ``None``.
    weight
        Edge-weight statistic — ``'jaccard'`` (default), ``'odds_ratio'`` or
        ``'obs_exp'``.
    min_weight
        Minimum edge weight to keep an edge.
    padj_cutoff
        Keep only edges with ``padj <= padj_cutoff``. ``None`` disables the
        significance filter.
    as_dataframe
        If ``True`` return a tidy edge-list :class:`pandas.DataFrame` instead
        of a :class:`networkx.Graph`.
    **coloc_kwargs
        Extra keyword args forwarded to :func:`colocalization` (``value_type``,
        ``threshold``, ``method`` …) when ``coloc`` is computed here.

    Returns
    -------
    :class:`networkx.Graph` or :class:`pandas.DataFrame`
        An undirected weighted graph (markers as nodes, with a
        ``positive_fraction`` node attribute) — or the edge list when
        ``as_dataframe=True``.
    """
    if coloc is None:
        coloc = colocalization(adata, markers=markers, **coloc_kwargs)
    if weight not in ("jaccard", "odds_ratio", "obs_exp"):
        raise ValueError(
            "weight must be 'jaccard', 'odds_ratio' or 'obs_exp', "
            f"got {weight!r}."
        )

    pairs = coloc[coloc["order"] == 2].copy()
    edges = pairs[pairs[weight] >= min_weight]
    if padj_cutoff is not None and "padj" in edges.columns:
        edges = edges[edges["padj"].fillna(1.0) <= padj_cutoff]

    names = coloc.attrs.get("marker_names", [])
    edge_rows = []
    for _, r in edges.iterrows():
        a, b = r["marker_list"]
        edge_rows.append({
            "source": a, "target": b, "weight": float(r[weight]),
            "jaccard": float(r["jaccard"]), "odds_ratio": float(r["odds_ratio"]),
            "obs_exp": float(r["obs_exp"]), "n_copos": int(r["n_copos"]),
            "padj": float(r["padj"]) if pd.notna(r["padj"]) else np.nan,
        })
    edge_df = pd.DataFrame(edge_rows)

    if as_dataframe:
        return edge_df

    nx = _require("networkx", "Co-expression network construction")
    G = nx.Graph()
    # Per-marker positive fraction as a node attribute.
    n_ev = coloc.attrs.get("n_ev", adata.n_obs)
    jac_mat = coloc.attrs.get("jaccard")
    for name in names:
        G.add_node(name)
    if jac_mat is not None:
        B, _ = _positivity(adata, markers=names,
                           value_type=_value_type(
                               adata, coloc_kwargs.get("value_type")),
                           threshold=coloc_kwargs.get("threshold", 0.0))
        for k, name in enumerate(names):
            G.nodes[name]["positive_fraction"] = float(B[:, k].mean())
    for _, r in edge_df.iterrows():
        G.add_edge(r["source"], r["target"], weight=r["weight"],
                   jaccard=r["jaccard"], odds_ratio=r["odds_ratio"],
                   obs_exp=r["obs_exp"], n_copos=int(r["n_copos"]),
                   padj=r["padj"])
    return G


# ---------------------------------------------------------------------------
# protein_combinations
# ---------------------------------------------------------------------------
@register_function(
    aliases=["protein_combinations", "ev_protein_combos", "depc",
             "蛋白质组合", "差异表达蛋白组合"],
    category="ev",
    description=(
        "Enumerate the multi-marker protein combinations (EV signatures) "
        "carried by individual EVs and count how many vesicles carry each. "
        "With condition_key, compare combination frequencies between "
        "conditions to find DEPCs — differentially-expressed protein "
        "combinations — with BH-corrected Fisher tests."
    ),
    examples=[
        "combos = ov.single.ev.protein_combinations(adata, markers=tetra)",
        "depc = ov.single.ev.protein_combinations(adata, markers=tetra, "
        "condition_key='condition')",
    ],
    related=["single.ev.colocalization", "single.ev.coexpression_network"],
)
def protein_combinations(adata, *, markers=None, condition_key: Optional[str] = None,
                         value_type: Optional[str] = None,
                         threshold: float = 0.0, min_ev: int = 1,
                         max_markers: int = 12,
                         reference: Optional[str] = None):
    """Enumerate EV protein combinations and find DEPCs between conditions.

    Each EV carries a *signature* — the exact set of markers it is positive
    for. This enumerates those signatures, counts the vesicles carrying each,
    and (when a condition is given) tests which combinations differ in
    frequency between conditions: the differentially-expressed protein
    combinations (DEPCs).

    Parameters
    ----------
    adata
        EV x protein :class:`anndata.AnnData`.
    markers
        Markers defining the combination space. ``None`` uses every protein
        in ``var`` (subject to ``max_markers``).
    condition_key
        Optional ``obs`` column splitting EVs into conditions. When given,
        the result is a per-combination DEPC table comparing frequencies
        between conditions.
    value_type
        ``'count'`` | ``'intensity'`` | ``'binary'``. ``None`` reads
        ``adata.uns['ev']['value_type']``.
    threshold
        Positivity cutoff for ``'count'`` / ``'intensity'`` matrices.
    min_ev
        Drop combinations carried by fewer than ``min_ev`` EVs.
    max_markers
        Guard rail — refuse to enumerate when more than this many markers are
        requested (the signature space is 2**n).
    reference
        Reference condition for the DEPC comparison (only when
        ``condition_key`` is given and exactly two conditions are present).
        ``None`` uses the first condition by sort order.

    Returns
    -------
    :class:`pandas.DataFrame`
        Without ``condition_key`` — a combination table with ``combination``,
        ``n_markers``, ``n_ev`` and ``fraction``, sorted by abundance. With
        ``condition_key`` — a DEPC table with per-condition EV counts /
        fractions, ``log2_fold_change`` (two-condition case),
        Fisher ``pvalue`` and BH ``padj``.
    """
    from scipy.stats import fisher_exact

    vtype = _value_type(adata, value_type)
    if markers is None:
        markers = list(adata.var_names)
    if len(markers) > max_markers:
        raise ValueError(
            f"{len(markers)} markers requested but max_markers={max_markers}; "
            "the signature space grows as 2**n — pass a focused `markers` "
            "list or raise `max_markers`."
        )
    B, names = _positivity(adata, markers=markers, value_type=vtype,
                           threshold=threshold)
    n_ev = B.shape[0]

    def _sig(row):
        on = [names[k] for k in range(len(names)) if row[k]]
        return "+".join(on) if on else "(none)"

    sigs = np.array([_sig(B[i]) for i in range(n_ev)], dtype=object)

    if condition_key is None:
        vc = pd.Series(sigs).value_counts()
        vc = vc[vc >= min_ev]
        res = pd.DataFrame({
            "combination": vc.index,
            "n_markers": [0 if s == "(none)" else s.count("+") + 1
                          for s in vc.index],
            "n_ev": vc.values.astype(int),
            "fraction": vc.values / n_ev,
        }).reset_index(drop=True)
        res.attrs["n_ev"] = n_ev
        return res

    if condition_key not in adata.obs:
        raise KeyError(f"obs column {condition_key!r} not found.")
    cond = pd.Series(adata.obs[condition_key].values, dtype=object)
    conditions = sorted(cond.dropna().unique())
    if len(conditions) < 2:
        raise ValueError(
            f"condition_key {condition_key!r} has <2 conditions; "
            "cannot compute DEPCs."
        )

    tab = pd.crosstab(pd.Series(sigs, name="combination"), cond)
    tab = tab[tab.sum(axis=1) >= min_ev]
    cond_totals = {c: int((cond == c).sum()) for c in conditions}

    rows = []
    two = len(conditions) == 2
    ref = reference if reference is not None else conditions[0]
    if ref not in conditions:
        raise ValueError(f"reference {ref!r} not among conditions {conditions}.")
    other = [c for c in conditions if c != ref]

    for combo, counts in tab.iterrows():
        rec = {"combination": combo,
               "n_markers": 0 if combo == "(none)" else combo.count("+") + 1}
        for c in conditions:
            rec[f"n_{c}"] = int(counts.get(c, 0))
            rec[f"frac_{c}"] = (int(counts.get(c, 0)) / cond_totals[c]
                                if cond_totals[c] else 0.0)
        if two:
            c1, c2 = ref, other[0]
            a = int(counts.get(c2, 0))
            b = cond_totals[c2] - a
            c = int(counts.get(c1, 0))
            d = cond_totals[c1] - c
            _, pval = fisher_exact([[a, b], [c, d]])
            f2 = rec[f"frac_{c2}"]
            f1 = rec[f"frac_{c1}"]
            rec["log2_fold_change"] = float(
                np.log2((f2 + 1e-9) / (f1 + 1e-9))
            )
            rec["pvalue"] = float(pval)
        else:
            # Multi-condition: chi-square-style Fisher on the collapsed
            # ref-vs-rest 2x2 table.
            a = sum(int(counts.get(c, 0)) for c in other)
            b = sum(cond_totals[c] for c in other) - a
            c = int(counts.get(ref, 0))
            d = cond_totals[ref] - c
            _, pval = fisher_exact([[a, b], [c, d]])
            rec["pvalue"] = float(pval)
        rows.append(rec)

    res = pd.DataFrame(rows)
    if not res.empty:
        res["padj"] = _bh_adjust(res["pvalue"].values)
        sort_col = "log2_fold_change" if two else "pvalue"
        ascending = two is False
        res = res.sort_values(
            sort_col, ascending=ascending,
            key=(np.abs if two else None),
        ).reset_index(drop=True)
    res.attrs["conditions"] = conditions
    res.attrs["reference"] = ref
    return res


# ---------------------------------------------------------------------------
# colocalization_plot
# ---------------------------------------------------------------------------
@register_function(
    aliases=["colocalization_plot", "ev_coloc_heatmap", "coloc_heatmap",
             "共定位热图", "标志物共定位热图"],
    category="ev",
    description=(
        "Plot a single-EV marker colocalization heatmap from a "
        "colocalization result — a symmetric marker x marker matrix of "
        "Jaccard index, odds ratio or observed/expected co-positivity."
    ),
    examples=[
        "res = ov.single.ev.colocalization(adata, markers=tetra)",
        "ax = ov.single.ev.colocalization_plot(res)",
        "ax = ov.single.ev.colocalization_plot(res, value='odds_ratio')",
    ],
    related=["single.ev.colocalization", "single.ev.coexpression_network"],
)
def colocalization_plot(coloc, *, value: str = "jaccard", cmap: str = "magma",
                        annot: bool = True, ax=None, figsize=(6.0, 5.0),
                        title: Optional[str] = None, **kwargs):
    """Colocalization heatmap from a :func:`colocalization` result.

    Parameters
    ----------
    coloc
        A :func:`colocalization` result :class:`pandas.DataFrame` (its
        ``.attrs`` carry the symmetric statistic matrices).
    value
        Which symmetric matrix to draw — ``'jaccard'`` (default),
        ``'odds_ratio'`` or ``'obs_exp'``.
    cmap
        Matplotlib colormap name.
    annot
        Annotate each cell with its numeric value.
    ax
        Optional pre-existing :class:`matplotlib.axes.Axes` to draw into.
    figsize
        Figure size used when ``ax`` is ``None``.
    title
        Optional plot title; a sensible default is used when ``None``.
    **kwargs
        Extra keyword args forwarded to :func:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        The axes the heatmap was drawn on.
    """
    import matplotlib.pyplot as plt

    if value not in ("jaccard", "odds_ratio", "obs_exp"):
        raise ValueError(
            "value must be 'jaccard', 'odds_ratio' or 'obs_exp', "
            f"got {value!r}."
        )
    mat = coloc.attrs.get(value)
    if mat is None:
        raise ValueError(
            f"the colocalization result has no {value!r} matrix in .attrs — "
            "pass a DataFrame returned by ov.single.ev.colocalization."
        )
    M = np.asarray(mat.values, dtype=float)
    labels = list(mat.index)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    plot_M = np.where(np.isnan(M), 0.0, M)
    im = ax.imshow(plot_M, cmap=cmap, **kwargs)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title or f"Single-EV marker colocalization ({value})")

    if annot:
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = M[i, j]
                if np.isnan(v):
                    continue
                txt = f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                        color="white" if plot_M[i, j] < plot_M.max() * 0.6
                        else "black")

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(value)
    ax.figure.tight_layout()
    return ax
