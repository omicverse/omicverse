"""CellChat cell-cell communication inference (vendored pyCellChat, R-parity pipeline).

``run_cellchat`` reproduces the standard R CellChat workflow::

    subsetData -> identifyOverExpressedGenes -> identifyOverExpressedInteractions
    -> computeCommunProb(type = "triMean") -> filterCommunication
    -> computeCommunProbPathway -> aggregateNet

on top of the PyTorch engine in ``omicverse.external.cellchat``. The
result lives in ``adata.uns[key_added]`` and is read directly by
``ov.pl.ccc_heatmap`` / ``ccc_network_plot`` / ``ccc_stat_plot`` through
:func:`format_cellchat_results`.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Sequence

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .._registry import register_function

_RESULT_KEYS = ("prob", "pval", "groups", "LRsig")


def _looks_like_cellchat_results(value) -> bool:
    return isinstance(value, Mapping) and value.get("method") == "cellchat" and all(
        key in value for key in _RESULT_KEYS
    )


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Plain ``object``-of-``str`` columns so the table round-trips through h5ad."""
    frame = frame.copy()
    for col in frame.columns:
        if not (pd.api.types.is_numeric_dtype(frame[col]) or pd.api.types.is_bool_dtype(frame[col])):
            frame[col] = np.asarray(frame[col].astype(object).where(frame[col].notna(), "").astype(str), dtype=object)
    frame.index = np.asarray(frame.index.astype(str), dtype=object)
    return frame


def _get_matrix(adata: anndata.AnnData, layer: str | None):
    if layer is None:
        return adata.X
    if layer == "raw":
        if adata.raw is None:
            raise ValueError("layer='raw' requested but adata.raw is None.")
        return adata.raw.X
    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in adata.layers.")
    return adata.layers[layer]


def _looks_like_counts(X) -> bool:
    values = X.data[:20000] if sp.issparse(X) else np.asarray(X).ravel()[:20000]
    values = values[values > 0]
    return values.size > 0 and (float(values.max()) > 50 or np.allclose(values, np.round(values)))


@register_function(
    aliases=["CellChat", "run_cellchat", "cellchat", "细胞通讯CellChat", "CellChat细胞通讯", "pycellchat"],
    category="single",
    description=(
        "Run CellChat ligand-receptor inference (triMean + permutation test) with the R CellChat "
        "pipeline and store results in `adata.uns` for `ov.pl.ccc_*` plotting."
    ),
    prerequisites={"functions": [], "optional_functions": ["pp.normalize_total", "pp.log1p"]},
    requires={"obs": ["groupby"]},
    produces={"uns": ["cellchat_res"]},
    auto_fix="none",
    examples=[
        "ov.single.run_cellchat(adata, groupby='cell_type', species='human')",
        "ov.single.run_cellchat(adata, groupby='cell_type', db_categories=['Secreted Signaling'])",
        "ov.pl.ccc_heatmap(adata, plot_type='heatmap', display_by='aggregation', result_uns_key='cellchat_res')",
    ],
    related=["single.format_cellchat_results", "single.cellchat_centrality",
             "pl.ccc_heatmap", "pl.ccc_network_plot", "pl.ccc_stat_plot"],
)
def run_cellchat(
    adata: anndata.AnnData,
    *,
    groupby: str,
    species: str = "human",
    layer: str | None = None,
    db_categories: Sequence[str] | None = None,
    n_perms: int = 100,
    seed: int = 1,
    population_size: bool = False,
    min_cells: int = 10,
    thresh_p: float = 0.05,
    pvalue_threshold: float = 0.05,
    variable_both: bool = True,
    use_gpu: bool = False,
    progress: bool = True,
    key_added: str = "cellchat_res",
    inplace: bool = True,
):
    r"""Infer cell-cell communication with CellChat (Jin et al., 2021/2025).

    Mirrors R CellChat 2.x defaults, so results match the R package:
    over-expressed signaling genes (one-vs-rest Wilcoxon, ``p < thresh_p``,
    positive log fold-change), LR pairs whose ligand and receptor are both
    over-expressed, triMean group averages, Hill-function communication
    probability with agonist/antagonist/co-receptor terms, and a label
    permutation test (``n_perms`` shuffles). Groups with ``<= min_cells``
    cells are removed afterwards (``filterCommunication``).

    Parameters
    ----------
    adata
        Single-cell AnnData with **log-normalised** expression (CellChat's
        ``normalizeData``: library-size 1e4 + log1p). Gene symbols in
        ``var_names`` must match the database species casing.
    groupby
        Column in ``adata.obs`` with cell-group labels.
    species
        CellChatDB species: ``'human'``, ``'mouse'``, ``'rat'`` or ``'zebrafish'``.
    layer
        Expression source: ``None`` (``adata.X``), a key of ``adata.layers``,
        or ``'raw'`` (``adata.raw.X``).
    db_categories
        Restrict CellChatDB to these annotations, e.g. ``['Secreted Signaling']``
        (R ``subsetDB``). Default uses the full database.
    n_perms, seed
        Permutation count (R ``nboot``) and RNG seed.
    population_size
        Weight probabilities by group abundance (R ``population.size``).
    min_cells
        Groups with ``<= min_cells`` cells are filtered (R ``filterCommunication``).
    thresh_p
        P-value cutoff for over-expressed genes (R ``identifyOverExpressedGenes``).
    pvalue_threshold
        Significance cutoff for interactions, pathways and aggregated networks.
    variable_both
        Require both ligand and receptor to be over-expressed (R default).
    use_gpu
        Run on CUDA/MPS when available (float32 on MPS). CPU uses float64 and
        reproduces R's probabilities to machine precision.
    progress
        Show a permutation progress bar.
    key_added
        ``adata.uns`` key for the result.
    inplace
        Store in ``adata.uns[key_added]`` and return ``adata``; otherwise
        return the result dict.

    Returns
    -------
    AnnData or dict
        ``adata`` (``inplace=True``) or the result dict with ``prob`` / ``pval``
        ``(sender, receiver, LR)`` tensors, ``groups``, ``LRsig``, ``df_net``
        (significant interactions), ``netP`` pathway tensor, ``count`` /
        ``weight`` networks and the run parameters.
    """
    from ..external.cellchat import _engine as eng
    from ..external.cellchat import _pipeline as rp
    from ..external.cellchat._db import ANNOTATIONS, load_cellchat_db

    if groupby not in adata.obs.columns:
        raise KeyError(f"`{groupby}` not found in adata.obs.")
    labels = adata.obs[groupby]
    if labels.isna().any():
        raise ValueError(f"adata.obs['{groupby}'] contains missing labels; drop or fill them first.")
    labels = labels.astype(str).to_numpy()

    X = _get_matrix(adata, layer)
    var_names = (adata.raw.var_names if layer == "raw" else adata.var_names).astype(str)
    if _looks_like_counts(X):
        raise ValueError(
            "CellChat expects log-normalised expression, but the selected matrix looks like raw counts. "
            "Run ov.pp.normalize_total(adata, target_sum=1e4) and ov.pp.log1p(adata), or pass `layer=`."
        )

    db = load_cellchat_db(species)
    if db_categories is not None:
        db = db.subset(db_categories)

    # subsetData: data.signaling = every database gene present, in data order.
    db_genes = set(rp.extract_genes(db))
    sig_cols = np.where([g in db_genes for g in var_names])[0]
    if len(sig_cols) == 0:
        raise ValueError(
            f"No CellChatDB ({species}) genes found in var_names; check the species and gene-symbol casing."
        )
    signaling_genes = list(var_names[sig_cols])
    X_sig = X[:, sig_cols]
    X_sig = sp.csr_matrix(X_sig, dtype=np.float64) if sp.issparse(X_sig) else np.asarray(X_sig, dtype=np.float64)

    markers = rp.identify_over_expressed_genes(X_sig, labels, signaling_genes, thresh_p=thresh_p)
    features_sig = list(dict.fromkeys(markers["features"]))
    lrsig = rp.identify_over_expressed_interactions(db, features_sig, signaling_genes,
                                                    variable_both=variable_both)
    if lrsig.empty:
        raise RuntimeError("No over-expressed ligand-receptor pairs; nothing to infer.")

    groups = sorted(pd.unique(labels))
    codes = np.searchsorted(np.asarray(groups, dtype=object), labels).astype(np.int64)
    design = eng.build_design(lrsig, db, signaling_genes)
    scale = float(X_sig.max())  # R: data.signaling / max(data.signaling)
    prob, pval = eng.compute_commun_prob(
        X_sig, codes, len(groups), design, signaling_genes, scale, n_perms=n_perms, seed=seed,
        kh=0.5, n=1.0, population_size=population_size, use_gpu=use_gpu, progress=progress)

    sizes = pd.Series(labels).value_counts().reindex(groups).fillna(0).astype(int).to_numpy()
    prob = rp.filter_communication(prob, sizes, min_cells=min_cells)

    pathways, netp, _ = rp.compute_commun_prob_pathway(prob, pval, lrsig["pathway_name"].astype(str),
                                                       thresh=pvalue_threshold)
    count, weight = rp.aggregate_net(prob, pval, thresh=pvalue_threshold)
    df_net = rp.subset_communication(prob, pval, groups, lrsig, thresh=pvalue_threshold)

    result = {
        "method": "cellchat",
        "groups": np.asarray(groups, dtype=object),
        "prob": prob,
        "pval": pval,
        "LRsig": _clean_frame(lrsig),
        "df_net": _clean_frame(df_net),
        "netP": {"pathways": np.asarray(pathways, dtype=object), "prob": netp},
        "count": count,
        "weight": weight,
        "over_expressed_genes": np.asarray(features_sig, dtype=object),
        "parameters": {
            "groupby": groupby, "species": species, "db_version": "CellChatDB v2 (R CellChat 2.2.0)",
            "db_categories": list(db_categories) if db_categories else list(ANNOTATIONS),
            "type_mean": "triMean", "n_perms": int(n_perms), "seed": int(seed),
            "population_size": bool(population_size), "min_cells": int(min_cells),
            "thresh_p": float(thresh_p), "pvalue_threshold": float(pvalue_threshold),
            "variable_both": bool(variable_both), "n_LRsig": int(len(lrsig)),
        },
    }
    if inplace:
        adata.uns[key_added] = result
        return adata
    return result


def _get_result(adata=None, result=None, key: str = "cellchat_res") -> Mapping:
    if result is None:
        if adata is None:
            raise ValueError("Provide `adata` or `result`.")
        if key not in adata.uns:
            raise KeyError(f"`{key}` not found in adata.uns; run ov.single.run_cellchat first.")
        result = adata.uns[key]
    if not _looks_like_cellchat_results(result):
        raise ValueError("Object is not a CellChat result produced by ov.single.run_cellchat.")
    return result


@register_function(
    aliases=["format_cellchat_results", "CellChat结果格式化", "CellChat转通信AnnData"],
    category="single",
    description="Convert `ov.single.run_cellchat` results into communication AnnData for `ov.pl.ccc_*`.",
    prerequisites={"functions": ["run_cellchat"], "optional_functions": []},
    requires={"uns": ["cellchat_res"]},
    produces={
        "layers": ["means", "pvalues"],
        "obs": ["sender", "receiver", "cell_type_pair"],
        "var": ["interacting_pair", "classification", "gene_a", "gene_b"],
    },
    auto_fix="none",
    examples=["comm_adata = ov.single.format_cellchat_results(adata)"],
    related=["single.run_cellchat", "pl.ccc_heatmap", "pl.ccc_network_plot", "pl.ccc_stat_plot"],
)
def format_cellchat_results(adata=None, *, result=None, uns_key: str = "cellchat_res",
                            separator: str = "|") -> anndata.AnnData:
    r"""Communication AnnData from a CellChat result.

    ``obs`` are all ordered sender-receiver pairs (``sender|receiver``),
    ``var`` the inferred LR pairs (``interaction_name``). ``X`` and
    ``layers['means']`` hold the communication probability, ``layers['pvalues']``
    the permutation p-value; ``var['classification']`` is the CellChat pathway.

    Parameters
    ----------
    adata
        AnnData with ``adata.uns[uns_key]`` from :func:`run_cellchat`.
    result
        The result dict itself (alternative to ``adata``).
    uns_key
        Key in ``adata.uns``.
    separator
        Separator between sender and receiver in ``obs_names``.
    """
    res = _get_result(adata, result, uns_key)
    groups = [str(g) for g in res["groups"]]
    if any(separator in g for g in groups):
        raise ValueError(f"Separator '{separator}' occurs in a group label; choose another separator.")
    prob = np.asarray(res["prob"], dtype=np.float64)
    pval = np.asarray(res["pval"], dtype=np.float64)
    n = len(groups)
    lrsig = pd.DataFrame(res["LRsig"]).copy()

    senders = np.repeat(groups, n)
    receivers = np.tile(groups, n)
    obs = pd.DataFrame({"sender": senders, "receiver": receivers},
                       index=[f"{s}{separator}{r}" for s, r in zip(senders, receivers)])
    obs["cell_type_pair"] = obs.index.astype(str)

    # R's LR key (DB rownames) is unique; interaction_name can repeat.
    key = lrsig["id"] if "id" in lrsig.columns else lrsig["interaction_name"]
    var = pd.DataFrame(index=key.astype(str).to_numpy())
    var["interaction_name"] = lrsig["interaction_name"].astype(str).to_numpy()
    var["interaction_name_2"] = lrsig.get("interaction_name_2", lrsig["interaction_name"]).astype(str).to_numpy()
    var["interacting_pair"] = var["interaction_name"]
    var["classification"] = lrsig["pathway_name"].astype(str).replace("", "Unclassified").to_numpy()
    var["gene_a"] = lrsig["ligand"].astype(str).to_numpy()
    var["gene_b"] = lrsig["receptor"].astype(str).to_numpy()
    for col in ("annotation", "evidence"):
        if col in lrsig.columns:
            var[col] = lrsig[col].astype(str).to_numpy()

    means = prob.reshape(n * n, -1)  # C order: receiver varies fastest within a sender
    comm = anndata.AnnData(X=means.copy(), obs=obs, var=var)
    comm.layers["means"] = means.copy()
    comm.layers["pvalues"] = pval.reshape(n * n, -1).copy()
    comm.uns["comm_source"] = "cellchat"
    comm.uns["cellchat_parameters"] = dict(res.get("parameters", {}))
    return comm


@register_function(
    aliases=["cellchat_centrality", "CellChat中心性", "netAnalysis_computeCentrality", "signaling role"],
    category="single",
    description="Per-pathway network centrality of CellChat signaling (R netAnalysis_computeCentrality).",
    prerequisites={"functions": ["run_cellchat"], "optional_functions": []},
    requires={"uns": ["cellchat_res"]},
    produces={"uns": ["cellchat_res"]},
    auto_fix="none",
    examples=["centr = ov.single.cellchat_centrality(adata)", "centr['MIF']"],
    related=["single.run_cellchat", "pl.ccc_heatmap"],
)
def cellchat_centrality(adata=None, *, result=None, uns_key: str = "cellchat_res",
                        signaling: Sequence[str] | None = None) -> dict:
    r"""Compute sender/receiver/mediator/influencer centrality per pathway.

    Ports R ``netAnalysis_computeCentrality`` / ``computeCentralityLocal``:
    weighted out/in-degree (sender/receiver), flow betweenness (mediator),
    information centrality (influencer), plus hub, authority, eigenvector,
    PageRank and betweenness. Stored under ``result['netP']['centr']``.

    Returns
    -------
    dict
        ``{pathway: DataFrame(groups x measures)}``.
    """
    from ..external.cellchat import _pipeline as rp

    res = _get_result(adata, result, uns_key)
    pathways = [str(p) for p in res["netP"]["pathways"]]
    netp = np.asarray(res["netP"]["prob"], dtype=np.float64)
    if signaling is not None:
        idx = [pathways.index(p) for p in signaling if p in pathways]
        pathways, netp = [pathways[i] for i in idx], netp[:, :, idx]
    centr = rp.net_analysis_compute_centrality(netp, pathways, [str(g) for g in res["groups"]])
    res["netP"]["centr"] = centr
    return centr


@register_function(
    aliases=["cellchat_communication_patterns", "CellChat通讯模式", "identifyCommunicationPatterns"],
    category="single",
    description="Outgoing/incoming communication patterns of CellChat pathways via NMF.",
    prerequisites={"functions": ["run_cellchat"], "optional_functions": []},
    requires={"uns": ["cellchat_res"]},
    produces={"uns": ["cellchat_res"]},
    auto_fix="none",
    examples=["W, H = ov.single.cellchat_communication_patterns(adata, k=3, pattern='outgoing')"],
    related=["single.run_cellchat", "single.cellchat_centrality"],
)
def cellchat_communication_patterns(adata=None, *, k: int, pattern: str = "outgoing", result=None,
                                    uns_key: str = "cellchat_res", random_state: int = 0):
    r"""Identify global communication patterns (R ``identifyCommunicationPatterns``).

    Factorises the group x pathway signaling matrix (each pathway scaled to max 1)
    into ``k`` patterns with NMF. ``W`` (groups x patterns) is row-normalised and
    ``H`` (patterns x pathways) column-normalised, as in R.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        ``W`` cell-group contributions and ``H`` pathway contributions.
    """
    from ..external.cellchat import _pipeline as rp

    res = _get_result(adata, result, uns_key)
    W, H = rp.identify_communication_patterns(
        np.asarray(res["netP"]["prob"], dtype=np.float64), [str(p) for p in res["netP"]["pathways"]],
        [str(g) for g in res["groups"]], k=k, pattern=pattern, random_state=random_state)
    res["netP"].setdefault("pattern", {})[pattern] = {"cell": W, "signaling": H}
    return W, H


@register_function(
    aliases=["cellchat_subset_communication", "subsetCommunication", "CellChat显著互作表"],
    category="single",
    description="Table of significant CellChat interactions filtered by sender, receiver or pathway.",
    prerequisites={"functions": ["run_cellchat"], "optional_functions": []},
    requires={"uns": ["cellchat_res"]},
    produces={},
    auto_fix="none",
    examples=["df = ov.single.cellchat_subset_communication(adata, signaling=['MIF'])"],
    related=["single.run_cellchat"],
)
def cellchat_subset_communication(adata=None, *, result=None, uns_key: str = "cellchat_res",
                                  sources: Sequence[str] | None = None,
                                  targets: Sequence[str] | None = None,
                                  signaling: Sequence[str] | None = None,
                                  thresh: float = 0.05) -> pd.DataFrame:
    r"""R ``subsetCommunication``: significant interactions as a long table."""
    from ..external.cellchat import _pipeline as rp

    res = _get_result(adata, result, uns_key)
    df = rp.subset_communication(np.asarray(res["prob"]), np.asarray(res["pval"]),
                                 [str(g) for g in res["groups"]], pd.DataFrame(res["LRsig"]), thresh=thresh)
    if sources is not None:
        df = df[df["source"].isin(list(sources))]
    if targets is not None:
        df = df[df["target"].isin(list(targets))]
    if signaling is not None:
        df = df[df["pathway_name"].isin(list(signaling))]
    return df.reset_index(drop=True)
