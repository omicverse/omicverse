"""Gene-set scoring (native port of ``scanpy.tl.score_genes`` / ``score_genes_cell_cycle``).

Ported from scanpy ``scanpy/tools/_score_genes.py`` (BSD-3-Clause,
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab). The random
control-gene draw uses the same NumPy global RNG sequence as scanpy
(``np.random.seed(random_state)`` + ``Series.sample``) so identical inputs give
identical scores.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .._registry import register_function


def _check_use_raw(adata, use_raw, *, layer=None) -> bool:
    if use_raw is not None:
        return use_raw
    if layer is not None:
        return False
    return adata.raw is not None


def _sparse_nanmean(x, axis: int) -> np.ndarray:
    z = x.copy()
    z.data = np.isnan(z.data)
    z.eliminate_zeros()
    n_elements = z.shape[axis] - z.sum(axis)
    y = x.copy()
    y.data[np.isnan(y.data)] = 0
    y.eliminate_zeros()
    s = y.sum(axis, dtype="float64")
    return s / n_elements


def _nan_means(x, *, axis: int, dtype=None) -> np.ndarray:
    if sparse.issparse(x):
        return np.array(_sparse_nanmean(x, axis=axis)).flatten()
    return np.nanmean(np.asarray(x), axis=axis, dtype=dtype)


def _check_score_genes_args(adata, gene_list, gene_pool, *, layer, use_raw):
    var_names = adata.raw.var_names if use_raw else adata.var_names
    gene_list = pd.Index([gene_list] if isinstance(gene_list, str) else gene_list)
    genes_to_ignore = gene_list.difference(var_names, sort=False)
    gene_list = gene_list.intersection(var_names)
    if len(genes_to_ignore) > 0:
        import warnings
        warnings.warn(f"genes are not in var_names and ignored: {genes_to_ignore}",
                      UserWarning, stacklevel=3)
    if len(gene_list) == 0:
        raise ValueError("No valid genes were passed for scoring.")
    if gene_pool is None:
        gene_pool = var_names.astype("string")
    else:
        gene_pool = pd.Index(gene_pool, dtype="string").intersection(var_names)
    if len(gene_pool) == 0:
        raise ValueError("No valid genes were passed for reference set.")

    def get_subset(genes):
        if use_raw:
            x = adata.raw.X
        elif layer is not None:
            x = adata.layers[layer]
        else:
            x = adata.X
        if len(genes) == len(var_names):
            return x
        idx = var_names.get_indexer(genes)
        return x[:, idx]

    return gene_list, gene_pool, get_subset


def _score_genes_bins(gene_list, gene_pool, *, ctrl_as_ref, ctrl_size, n_bins, get_subset):
    obs_avg = pd.Series(_nan_means(get_subset(gene_pool), axis=0), index=gene_pool)
    obs_avg = obs_avg[np.isfinite(obs_avg)]
    n_items = int(np.round(len(obs_avg) / (n_bins - 1)))
    obs_cut = obs_avg.rank(method="min") // n_items
    keep_ctrl_in_obs_cut = False if ctrl_as_ref else obs_cut.index.isin(gene_list)
    for cut in np.unique(obs_cut.loc[gene_list]):
        r_genes = obs_cut[(obs_cut == cut) & ~keep_ctrl_in_obs_cut].index
        if len(r_genes) == 0:
            import warnings
            warnings.warn(
                f"No control genes for cut={cut}. You might want to increase "
                f"gene_pool size (current size: {len(gene_pool)})",
                UserWarning, stacklevel=3)
        if ctrl_size < len(r_genes):
            r_genes = r_genes.to_series().sample(ctrl_size).index
        if ctrl_as_ref:
            r_genes = r_genes.difference(gene_list)
        yield r_genes


@register_function(
    aliases=["score_genes", "基因集打分", "gene_set_score", "module_score", "signature_score"],
    category="preprocessing",
    description=(
        "Score a gene set per cell as mean expression minus a random, expression-"
        "matched control set (Seurat AddModuleScore / scanpy score_genes)."
    ),
    prerequisites={"optional_functions": ["normalize_total", "log1p"]},
    requires={},
    produces={"obs": ["score"]},
    auto_fix="none",
    examples=[
        "ov.pp.score_genes(adata, ['CD3D', 'CD3E', 'CD2'], score_name='T_score')",
    ],
    related=["pp.score_genes_cell_cycle"],
)
def score_genes(
    adata,
    gene_list: Sequence[str],
    *,
    ctrl_as_ref: bool = True,
    ctrl_size: int = 50,
    gene_pool: Sequence[str] | None = None,
    n_bins: int = 25,
    score_name: str = "score",
    random_state=0,
    copy: bool = False,
    use_raw: bool | None = None,
    layer: str | None = None,
):
    """Score a set of genes (drop-in for ``scanpy.tl.score_genes``).

    The score is the average expression of ``gene_list`` minus the average
    expression of a reference set randomly sampled from ``gene_pool`` for each
    expression bin (Satija et al. 2015).

    Parameters
    ----------
    adata
        Annotated data matrix (log-normalised expression).
    gene_list
        Genes to score.
    ctrl_as_ref
        Allow the scored genes to be used as reference genes.
    ctrl_size
        Number of reference genes sampled per bin.
    gene_pool
        Genes to sample the reference set from (default: all genes).
    n_bins
        Number of expression-level bins.
    score_name
        ``adata.obs`` column to write.
    random_state
        Seed for sampling the reference set.
    copy
        Return a copy instead of writing to ``adata``.
    use_raw
        Use ``adata.raw`` (default: ``True`` if ``.raw`` is present and
        ``layer`` is ``None``).
    layer
        Layer to use.

    Returns
    -------
    ``None`` (or the copy) with ``adata.obs[score_name]`` (float64).
    """
    adata = adata.copy() if copy else adata
    use_raw = _check_use_raw(adata, use_raw, layer=layer)
    if random_state is not None:
        np.random.seed(random_state)
    gene_list, gene_pool, get_subset = _check_score_genes_args(
        adata, gene_list, gene_pool, use_raw=use_raw, layer=layer)

    control_genes = pd.Index([], dtype="string")
    for r_genes in _score_genes_bins(gene_list, gene_pool, ctrl_as_ref=ctrl_as_ref,
                                     ctrl_size=ctrl_size, n_bins=n_bins,
                                     get_subset=get_subset):
        control_genes = control_genes.union(r_genes)
    if len(control_genes) == 0:
        msg = "No control genes found in any cut."
        if ctrl_as_ref:
            msg += " Try setting `ctrl_as_ref=False`."
        raise RuntimeError(msg)

    means_list, means_control = (
        _nan_means(get_subset(genes), axis=1, dtype="float64")
        for genes in (gene_list, control_genes)
    )
    score = means_list - means_control
    adata.obs[score_name] = pd.Series(np.array(score).ravel(), index=adata.obs_names,
                                      dtype="float64")
    return adata if copy else None


def score_genes_cell_cycle(
    adata,
    *,
    s_genes: Sequence[str],
    g2m_genes: Sequence[str],
    copy: bool = False,
    **kwargs,
):
    """Score S / G2M cell-cycle genes and assign ``phase`` (drop-in for
    ``scanpy.tl.score_genes_cell_cycle``)."""
    adata = adata.copy() if copy else adata
    ctrl_size = min(len(s_genes), len(g2m_genes))
    for genes, name in [(s_genes, "S_score"), (g2m_genes, "G2M_score")]:
        score_genes(adata, genes, score_name=name, ctrl_size=ctrl_size, **kwargs)
    scores = adata.obs[["S_score", "G2M_score"]]
    phase = pd.Series("S", index=scores.index)
    phase[scores["G2M_score"] > scores["S_score"]] = "G2M"
    phase[np.all(scores < 0, axis=1)] = "G1"
    adata.obs["phase"] = phase
    return adata if copy else None
