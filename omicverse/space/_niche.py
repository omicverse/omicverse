"""Recurrent multicellular structure — niches.

A niche is a composition that repeats: the same mix of cell types, found in
several places, doing the same job in each. Issue #760 asked for three flavours
of this and they turn out to be three different questions.

**Spatial niche.** Describe every spot by *what surrounds it* rather than by what
it is, then cluster those descriptions. :func:`neighborhood` clusters the raw
neighbourhood composition; :func:`utag` smooths expression along the spatial
graph before clustering, so the niche is defined by transcriptome-in-context.
Both are the missing flavours of squidpy's ``calculate_niche``; the third,
CellCharter, is already in ``ov.space.cellcharter``.

**Molecular niche.** Deconvolve each spot into cell-type abundances, factorise
that matrix, and read each factor as a recurrent multicellular programme. This is
what the myocardial-infarction and pulmonary-fibrosis atlases mean by the term.
omicverse has done this since ``nmf_tissue_zones`` shipped; :func:`molecular` is
the same function under the name the literature uses.

The distinction that matters: a spatial niche is defined by geometry, a molecular
niche by composition. On a slide where cell types are spatially segregated they
converge. Where they do not, they answer different questions, and it is worth
knowing which one a figure is showing.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import sparse

from .._registry import register_function

__all__ = ["neighborhood", "utag", "molecular"]


def _graph_and_codes(adata, cluster_key, connectivity_key):
    from ._neighborhood import _cluster_codes, _get_graph
    return _get_graph(adata, connectivity_key), *_cluster_codes(adata, cluster_key)


def _leiden_flavor_kwargs(leiden) -> dict:
    """``flavor='igraph'`` when this scanpy has the keyword, nothing when not.

    ``flavor=`` arrived in scanpy 1.10; omicverse pins ``scanpy>=1.9``, so on
    an otherwise valid 1.9 install passing it unconditionally is a TypeError
    and the whole niche module is unusable. Ask the signature rather than the
    version string: a distro backport or a fork can carry the keyword without
    the version number admitting it, and the signature is what actually
    decides whether the call succeeds.

    On 1.10+ we keep igraph — it is scanpy's own announced future default and
    1.10 warns when the flavour is left unset. On 1.9 we fall back to the
    leidenalg default. Both optimise the same objective over the same graph,
    so the fallback is a different partition of equal standing, not a
    degraded one; only the exact labels can differ between the two.
    """
    import inspect

    if "flavor" in inspect.signature(leiden).parameters:
        return {"flavor": "igraph", "n_iterations": 2, "directed": False}
    return {}


def _cluster_rows(matrix: np.ndarray, resolution: float, seed: int, n_neighbors: int):
    """Leiden over the rows of a feature matrix, via a temporary AnnData."""
    import scanpy as sc
    from anndata import AnnData

    tmp = AnnData(np.asarray(matrix, dtype=np.float32))
    sc.pp.neighbors(tmp, n_neighbors=min(n_neighbors, max(tmp.n_obs - 1, 2)),
                    use_rep="X", random_state=seed)
    sc.tl.leiden(tmp, resolution=resolution, key_added="niche", random_state=seed,
                 **_leiden_flavor_kwargs(sc.tl.leiden))
    return tmp.obs["niche"].to_numpy()


@register_function(
    aliases=["空间生态位", "spatial niche", "邻域组成聚类", "cellular neighborhood", "niche分析"],
    category="space",
    description="Define niches by clustering each observation's neighbourhood composition",
    requires={'obsp': ['spatial_connectivities'], 'obs': []},
    produces={'obs': ['{key_added}'], 'obsm': ['{key_added}_composition']},
    auto_fix='none',
    examples=[
        "ov.space.spatial_neighbors(adata, n_neighs=6, coord_type='grid')",
        "ov.space.niche.neighborhood(adata, 'cell_type', resolution=0.5)",
    ],
)
def neighborhood(
    adata,
    cluster_key: str,
    *,
    connectivity_key: Optional[str] = None,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    normalize: bool = True,
    scale: bool = True,
    min_niche_size: Optional[int] = None,
    key_added: str = "niche",
    seed: int = 0,
    copy: bool = False,
):
    """Cluster spots by the company they keep.

    Each spot is described by the mix of cluster labels among its spatial
    neighbours — not by its own label — and those descriptions are clustered.
    Two spots of different cell types sitting in the same kind of neighbourhood
    end up in the same niche, which is the point: the niche is the local
    community, not the cell.

    This is the ``flavor='neighborhood'`` arm of squidpy's ``calculate_niche``.

    Arguments:
        adata: AnnData with a spatial graph and a categorical cluster column.
        cluster_key: Column in ``adata.obs`` holding the cluster labels.
        connectivity_key: Graph in ``adata.obsp``. Default ``'spatial_connectivities'``.
        resolution: Leiden resolution over the composition profiles. Higher splits
            more finely.
        n_neighbors: Neighbours used when building the profile graph for Leiden.
        normalize: Turn neighbour counts into fractions, so a spot at the tissue
            edge with fewer neighbours is not pushed into its own niche purely for
            having a smaller total.
        scale: Standardise each cluster's column before clustering, so a rare cell
            type carries the same weight as an abundant one. Without it the
            partition is driven by whichever type is most common.
        min_niche_size: Niches smaller than this are relabelled ``'unassigned'``.
        key_added: ``obs`` column to write. The composition itself is written to
            ``adata.obsm[f'{key_added}_composition']``.
        seed: Seed for Leiden.
        copy: Return the labels instead of writing them to ``adata.obs``.

    Returns:
        A Series of niche labels when ``copy=True``, else ``None``.
    """
    graph, codes, cats = _graph_and_codes(adata, cluster_key, connectivity_key)

    onehot = sparse.csr_matrix(
        (np.ones(len(codes)), (np.arange(len(codes)), np.clip(codes, 0, None))),
        shape=(len(codes), len(cats)),
    )
    composition = np.asarray((graph @ onehot).todense(), dtype=np.float64)

    if normalize:
        totals = composition.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            composition = np.where(totals > 0, composition / totals, 0.0)

    features = composition
    if scale:
        from sklearn.preprocessing import StandardScaler
        features = StandardScaler().fit_transform(composition)

    labels = _cluster_rows(features, resolution, seed, n_neighbors)

    if min_niche_size is not None:
        counts = pd.Series(labels).value_counts()
        small = set(counts[counts < min_niche_size].index)
        labels = np.array(["unassigned" if x in small else x for x in labels])

    series = pd.Series(pd.Categorical(labels), index=adata.obs_names, name=key_added)
    if copy:
        return series
    adata.obs[key_added] = series
    adata.obsm[f"{key_added}_composition"] = pd.DataFrame(
        composition, index=adata.obs_names, columns=[str(c) for c in cats]
    )
    return None


@register_function(
    aliases=["UTAG", "utag", "空间平滑聚类", "spatial context clustering"],
    category="space",
    description="Define niches by smoothing expression along the spatial graph, then clustering (UTAG)",
    requires={'obsp': ['spatial_connectivities']},
    produces={'obs': ['{key_added}'], 'obsm': ['{key_added}_smoothed']},
    auto_fix='none',
    examples=[
        "ov.space.spatial_neighbors(adata, n_neighs=6, coord_type='grid')",
        "ov.space.niche.utag(adata, resolution=0.5)",
    ],
)
def utag(
    adata,
    *,
    connectivity_key: Optional[str] = None,
    use_rep: Optional[str] = None,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    normalize: bool = True,
    key_added: str = "utag_niche",
    seed: int = 0,
    copy: bool = False,
):
    """Smooth expression over the spatial graph, then cluster the result (UTAG).

    One matrix multiplication: replace each spot's profile with the average over
    itself and its spatial neighbours, then cluster. A spot is then represented by
    its transcriptome *in context*, so the clusters that come out are tissue
    domains rather than cell types — the same profile in two different
    surroundings lands in two different niches.

    This is the ``flavor='utag'`` arm of squidpy's ``calculate_niche``.

    Arguments:
        adata: AnnData with a spatial graph.
        connectivity_key: Graph in ``adata.obsp``. Default ``'spatial_connectivities'``.
        use_rep: ``obsm`` key to smooth, e.g. ``'X_pca'``. Defaults to ``adata.X``,
            which is usually slower and noisier — a PCA representation is the
            better input.
        resolution: Leiden resolution over the smoothed profiles.
        n_neighbors: Neighbours used when building the profile graph for Leiden.
        normalize: Average over the neighbourhood rather than summing over it, so
            spots with fewer neighbours are not systematically dimmer.
        key_added: ``obs`` column to write. The smoothed matrix goes to
            ``adata.obsm[f'{key_added}_smoothed']``.
        seed: Seed for Leiden.
        copy: Return the labels instead of writing them to ``adata.obs``.

    Returns:
        A Series of niche labels when ``copy=True``, else ``None``.
    """
    from ._neighborhood import _get_graph

    graph = _get_graph(adata, connectivity_key)

    if use_rep is not None:
        if use_rep not in adata.obsm:
            raise KeyError(f"adata.obsm has no key {use_rep!r}.")
        matrix = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    else:
        matrix = adata.X
        matrix = np.asarray(matrix.todense() if sparse.issparse(matrix) else matrix,
                            dtype=np.float64)

    # include the spot itself, so a spot's own profile is not discarded
    adjacency = (graph > 0).astype(np.float64).tocsr()
    adjacency.setdiag(1.0)
    smoothed = adjacency @ matrix
    if normalize:
        degree = np.asarray(adjacency.sum(axis=1)).ravel().reshape(-1, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed = np.where(degree > 0, smoothed / degree, 0.0)

    labels = _cluster_rows(smoothed, resolution, seed, n_neighbors)
    series = pd.Series(pd.Categorical(labels), index=adata.obs_names, name=key_added)
    if copy:
        return series
    adata.obs[key_added] = series
    adata.obsm[f"{key_added}_smoothed"] = smoothed
    return None


@register_function(
    aliases=["分子生态位", "molecular niche", "细胞生态位", "cellular niche", "NMF生态位", "tissue zone"],
    category="space",
    description="Factorise per-spot cell-type abundances into recurrent multicellular niches (NMF)",
    requires={'obsm': ['q05_cell_abundance_w_sf']},
    produces={'obsm': ['X_tissue_zones'], 'uns': ['tissue_zones']},
    auto_fix='none',
    examples=[
        "# after cell2location deconvolution",
        "zones = ov.space.niche.molecular(adata, n_factors=10)",
        "zones.factor_loadings          # cell types x factors",
        "zones.factor_top_cell_types    # the top cell types per factor",
    ],
)
def molecular(adata, *args, **kwargs):
    """Factorise deconvolved cell-type abundances into molecular niches.

    Take the spot-by-cell-type abundance matrix a deconvolution method produces —
    cell2location's ``q05_cell_abundance_w_sf`` by default — and decompose it with
    NMF. Each factor is a group of cell types that recur together across the
    slide, which is what the myocardial-infarction and pulmonary-fibrosis atlases
    call a molecular or cellular niche.

    This is :func:`omicverse.space.nmf_tissue_zones` under the name the literature
    uses. The two are the same function; nothing about the computation differs.
    It is aliased here because nobody looking for niche analysis will search for
    "tissue zones".

    Arguments:
        adata: Spatial AnnData carrying a per-spot cell-abundance matrix in
            ``adata.obsm``.
        *args: Forwarded to :func:`~omicverse.space.nmf_tissue_zones`.
        **kwargs: Forwarded to :func:`~omicverse.space.nmf_tissue_zones`, including
            ``obsm_key``, ``n_factors``, ``cell_type_names`` and ``top_k``.

    Returns:
        The :class:`~omicverse.space.TissueZones` object ``nmf_tissue_zones``
        returns. It is a plain dataclass, not a plotting handle: read
        ``.factor_loadings`` (cell types x factors), ``.spot_activations``
        (spots x factors) and ``.factor_top_cell_types``, and draw them with
        whatever plotter you like. The activations are also written to
        ``adata.obsm['X_tissue_zones']``, with the run's metadata under
        ``adata.uns['tissue_zones']['X_tissue_zones']``.
    """
    from ._tissue_zones import nmf_tissue_zones
    return nmf_tissue_zones(adata, *args, **kwargs)
