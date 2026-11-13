"""Scanpy-equivalent helpers built on omicverse internals.

Used by ``omicverse.space``, ``omicverse.bulk2single``, ``omicverse.genetics``
and ``omicverse.tl``. Several scanpy calls in those modules have public
omicverse counterparts with different defaults or storage conventions (e.g.
``ov.pp.scale`` writes ``layers['scaled']`` while ``sc.pp.scale`` overwrites
``X``; ``ov.pp.leiden`` defaults to the brisc backend while ``sc.tl.leiden``
uses leidenalg). The helpers below reproduce the scanpy *results* with
omicverse code so call sites keep their behaviour without importing scanpy.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd


def scale_X(adata, *, zero_center: bool = True, max_value: Optional[float] = None):
    """``sc.pp.scale(adata)``: z-score ``adata.X`` in place."""
    from .pp._scale import scale_anndata
    scale_anndata(adata, zero_center=zero_center, max_value=max_value)


def pca_X(adata, n_comps: Optional[int] = None, *, svd_solver: Optional[str] = None,
          zero_center: bool = True, random_state=0, **kwargs):
    """``sc.pp.pca(adata)`` / ``sc.tl.pca(adata)`` on ``adata.X``.

    Writes ``obsm['X_pca']``, ``varm['PCs']`` and ``uns['pca']`` like scanpy
    (``var['highly_variable']`` is used as mask when present).
    ``svd_solver=None`` maps to scanpy's default, ``'arpack'``.
    """
    from .pp._pca import pca as _pca
    if svd_solver is None:
        svd_solver = "arpack"
    _pca(adata, n_comps=n_comps, layer=None, zero_center=zero_center,
         svd_solver=svd_solver, random_state=random_state, **kwargs)


@contextmanager
def _cpu_mode():
    """Run on the CPU path whatever ``ov.settings.mode`` is.

    These helpers stand in for scanpy calls, which always ran on the CPU; the
    public ``ov.pp`` functions dispatch on the session mode (torch / RAPIDS
    paths take other arguments, e.g. no ``flavor`` for the torch Leiden).
    """
    from ._settings import settings
    old = settings.mode
    settings.mode = "cpu"
    try:
        yield
    finally:
        settings.mode = old


def neighbors(adata, n_neighbors: int = 15, *, use_rep=None, n_pcs=None,
              random_state=0, **kwargs):
    """``sc.pp.neighbors`` with scanpy defaults (umap connectivities)."""
    from .pp import neighbors as _neighbors
    with _cpu_mode():
        _neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep, n_pcs=n_pcs,
                   random_state=random_state, method="umap", **kwargs)


def leiden(adata, resolution: float = 1.0, *, key_added: str = "leiden",
           random_state=0, flavor: str = "leidenalg", n_iterations: int = -1,
           **kwargs):
    """``sc.tl.leiden`` with scanpy defaults.

    ``ov.pp.leiden`` would pick the brisc backend; ``flavor='leidenalg'`` and
    ``n_iterations=-1`` give scanpy's partition (exact C++ port of leidenalg
    when anndataoom-fast is available).
    """
    from .pp import leiden as _leiden
    with _cpu_mode():
        _leiden(adata, resolution=resolution, random_state=random_state,
                key_added=key_added, flavor=flavor, n_iterations=n_iterations,
                **kwargs)


def umap(adata, **kwargs):
    """``sc.tl.umap`` (min_dist=0.5, spread=1, random_state=0)."""
    from .pp import umap as _umap
    kwargs.setdefault("random_state", 0)
    with _cpu_mode():
        _umap(adata, **kwargs)


def _n_vars_used(adata, use_raw: Optional[bool], layer: Optional[str]) -> int:
    if use_raw is None:
        use_raw = adata.raw is not None and layer is None
    if use_raw and adata.raw is not None:
        return adata.raw.n_vars
    return adata.n_vars


def rank_genes_groups(
    adata,
    groupby: str,
    *,
    method: str = "t-test",
    groups: Union[str, Sequence[str]] = "all",
    reference: str = "rest",
    n_genes: Optional[int] = None,
    key_added: Optional[str] = None,
    use_raw: Optional[bool] = None,
    layer: Optional[str] = None,
    corr_method: str = "benjamini-hochberg",
    rankby_abs: bool = False,
    tie_correct: bool = False,
    pts: bool = False,
    **kwargs,
):
    """``sc.tl.rank_genes_groups`` with scanpy defaults, via
    :func:`omicverse.single.find_markers` (all genes kept, ``pts=False``,
    ``method='t-test'``)."""
    from .single._markers import find_markers
    if n_genes is None:
        n_genes = _n_vars_used(adata, use_raw, layer)
    find_markers(
        adata,
        groupby=groupby,
        method=method,
        n_genes=n_genes,
        key_added=key_added if key_added is not None else "rank_genes_groups",
        use_raw=use_raw,
        layer=layer,
        groups=groups,
        reference=reference,
        corr_method=corr_method,
        rankby_abs=rankby_abs,
        tie_correct=tie_correct,
        pts=pts,
        **kwargs,
    )


def paga(adata, groups: Optional[str] = None, *, model: str = "v1.2",
         neighbors_key: Optional[str] = None) -> None:
    """``sc.tl.paga(adata, groups)`` (connectivities and tree)."""
    from .utils._compat import PAGA
    if groups is None:
        for k in ("leiden", "louvain"):
            if k in adata.obs.columns:
                groups = k
                break
    if groups is None:
        raise ValueError(
            "You need to run `ov.pp.leiden` or `ov.pp.louvain` to compute "
            "community labels, or specify `groups='an_existing_key'`")
    if groups not in adata.obs.columns:
        raise KeyError(f"`groups` key {groups!r} not found in `adata.obs`.")
    if not isinstance(adata.obs[groups].dtype, pd.CategoricalDtype):
        adata.obs[groups] = adata.obs[groups].astype("category")
    p = PAGA(adata, groups, model=model, neighbors_key=neighbors_key)
    if "paga" not in adata.uns:
        adata.uns["paga"] = {}
    p.compute_connectivities()
    adata.uns["paga"]["connectivities"] = p.connectivities
    adata.uns["paga"]["connectivities_tree"] = p.connectivities_tree
    adata.uns[f"{groups}_sizes"] = np.array(p.ns)
    adata.uns["paga"]["groups"] = groups


default_20 = [
    "#1f77b4", "#ff7f0e", "#279e68", "#d62728", "#aa40fc", "#8c564b",
    "#e377c2", "#b5bd61", "#17becf", "#aec7e8", "#ffbb78", "#98df8a",
    "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5",
    "#ad494a", "#8c6d31",
]


def palette_28():
    """scanpy's ``default_28`` / ``zeileis_28`` palette."""
    from .pl._scanpy_compat import zeileis_28
    return list(zeileis_28)


def palette_102():
    """scanpy's ``default_102`` / ``godsnot_102`` palette."""
    from .pl._scanpy_compat import default_102
    return list(default_102)


def rank_genes_groups_df(
    adata,
    group: Union[str, Sequence[str], None],
    *,
    key: str = "rank_genes_groups",
    pval_cutoff: Optional[float] = None,
    log2fc_min: Optional[float] = None,
    log2fc_max: Optional[float] = None,
    gene_symbols: Optional[str] = None,
) -> pd.DataFrame:
    """Same output as ``sc.get.rank_genes_groups_df``."""
    result = adata.uns[key]
    if isinstance(group, str):
        group = [group]
    if group is None:
        names = result["names"]
        group = (names.columns.tolist() if isinstance(names, pd.DataFrame)
                 else list(names.dtype.names))
    group = list(group)
    method = result.get("params", {}).get("method")
    if method == "logreg":
        colnames = ["names", "scores"]
    else:
        colnames = ["names", "scores", "logfoldchanges", "pvals", "pvals_adj"]
    colnames = [c for c in colnames if c in result]

    frames = []
    for g in group:
        d = pd.DataFrame({c: pd.DataFrame(result[c])[g].to_numpy() for c in colnames})
        d.insert(0, "group", g)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["group"] = pd.Categorical(d["group"], categories=group)

    if method != "logreg":
        if pval_cutoff is not None and "pvals_adj" in d:
            d = d[d["pvals_adj"] < pval_cutoff]
        if log2fc_min is not None and "logfoldchanges" in d:
            d = d[d["logfoldchanges"] > log2fc_min]
        if log2fc_max is not None and "logfoldchanges" in d:
            d = d[d["logfoldchanges"] < log2fc_max]
    if gene_symbols is not None:
        d = d.join(adata.var[gene_symbols], on="names")

    for pts, name in {"pts": "pct_nz_group", "pts_rest": "pct_nz_reference"}.items():
        if pts in result and isinstance(result[pts], pd.DataFrame):
            pts_df = (
                result[pts][group]
                .rename_axis(index="names")
                .reset_index()
                .melt(id_vars="names", var_name="group", value_name=name)
            )
            d = d.merge(pts_df)

    if len(group) == 1:
        d = d.drop(columns="group")
    return d.reset_index(drop=True)


def paga_pos(adata, *, threshold: float = 0.01, random_state=0) -> None:
    """Store the PAGA layout in ``adata.uns['paga']['pos']``.

    Equivalent to ``sc.pl.paga(adata, plot=False)`` with the default
    Fruchterman-Reingold layout.
    """
    import random
    from sklearn.utils import check_random_state
    from .utils._compat import _get_igraph_from_adjacency

    adjacency_solid = adata.uns["paga"]["connectivities"].copy()
    if threshold > 0:
        adjacency_solid.data[adjacency_solid.data < threshold] = 0
        adjacency_solid.eliminate_zeros()
    rs = check_random_state(random_state)
    random.seed(rs.bytes(8))
    g = _get_igraph_from_adjacency(adjacency_solid)
    init_coords = rs.random_sample((adjacency_solid.shape[0], 2)).tolist()
    try:
        pos_list = g.layout("fr", seed=init_coords, weights="weight").coords
    except AttributeError:  # empty graphs
        pos_list = g.layout("fr", seed=init_coords).coords
    pos = {n: (x, -y) for n, (x, y) in enumerate(pos_list)}
    if len(pos) == 1:
        pos[0] = (0.5, 0.5)
    adata.uns["paga"]["pos"] = np.array([pos[n] for n in range(adjacency_solid.shape[0])])
