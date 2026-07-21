"""Cross-species preprocessing (ortholog alignment + standard prep).

Extracted so the ortholog *preparation* — map every species onto the reference
species' one-to-one orthologs, keep the shared genes, concatenate, and run the
usual normalize → log → HVG → scale → PCA — is reusable on its own, independent
of which integration backend runs afterwards (see
:class:`omicverse.single.CrossSpecies`).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import anndata

from .._registry import register_function


@register_function(
    aliases=["跨物种对齐", "cross_species_align", "cross species preprocessing", "同源对齐预处理"],
    category="tools",
    description="Align datasets across species onto shared one-to-one orthologs and preprocess",
    examples=[
        "adata = ov.tl.cross_species_align([adata_hs, adata_mm], ['human','mouse'])",
    ],
    related=["single.cross_species_integrate", "single.get_orthologs"],
)
def cross_species_align(
    adatas: Sequence[anndata.AnnData],
    species: Sequence[str],
    *,
    ref_species: Optional[str] = None,
    batch_key: str = "species",
    orthology_type: str = "one2one",
    n_top_genes: int = 2000,
    n_pcs: int = 50,
    target_sum: float = 1e4,
    preprocess: bool = True,
    host: str = "https://www.ensembl.org",
    cache_dir: Optional[str] = "data/orthologs",
    use_cache: bool = True,
) -> anndata.AnnData:
    """Ortholog-align and preprocess several single-cell datasets across species.

    Parameters
    ----------
    adatas
        One ``AnnData`` per species (**raw counts**, gene *symbols* as
        ``var_names``).
    species
        Species name for each entry of ``adatas`` (same order).
    ref_species
        Species whose gene symbols become the common axis (default ``species[0]``).
    batch_key
        ``obs`` column written with the species label.
    orthology_type, host, cache_dir, use_cache
        Forwarded to :func:`omicverse.single.get_orthologs`.
    n_top_genes, n_pcs, target_sum
        HVG / PCA / normalization settings applied when ``preprocess=True``.
    preprocess
        When ``True`` (default) run normalize → log1p → HVG(batch-aware) → scale
        → PCA; the raw counts are kept in ``layers['counts']``. Set ``False`` to
        return the concatenated raw-count object only.

    Returns
    -------
    anndata.AnnData
        The concatenated (and, by default, preprocessed) cross-species object,
        with ``uns['cross_species']`` describing the alignment.
    """
    import scanpy as sc

    from ..single._cross_species import map_var_to_orthologs, _normalize_species

    if len(adatas) != len(species):
        raise ValueError("`adatas` and `species` must have the same length.")
    if len(adatas) < 2:
        raise ValueError("Need at least two datasets to align across species.")
    if ref_species is None:
        ref_species = species[0]

    aligned: List[anndata.AnnData] = []
    for ad, sp in zip(adatas, species):
        a = ad.copy()
        if _normalize_species(sp) != _normalize_species(ref_species):
            a = map_var_to_orthologs(
                a, sp, ref_species, orthology_type=orthology_type,
                host=host, cache_dir=cache_dir, use_cache=use_cache)
        a.var_names_make_unique()
        a.obs[batch_key] = str(sp)
        aligned.append(a)

    common = set(aligned[0].var_names)
    for a in aligned[1:]:
        common &= set(a.var_names)
    common = sorted(common)
    if not common:
        raise ValueError(
            "No shared one-to-one orthologs across the datasets — check that "
            "var_names are gene symbols and the species names are correct.")

    aligned = [a[:, common].copy() for a in aligned]
    adata = anndata.concat(aligned, join="inner", index_unique="-")
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    adata.uns["cross_species"] = {
        "ref_species": str(ref_species),
        "species": [str(s) for s in species],
        "n_common_orthologs": len(common),
        "orthology_type": orthology_type,
    }

    if preprocess:
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        adata.raw = adata
        sc.pp.highly_variable_genes(
            adata, n_top_genes=min(n_top_genes, len(common)), batch_key=batch_key)
        adata = adata[:, adata.var["highly_variable"]].copy()
        sc.pp.scale(adata, max_value=10)
        sc.pp.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1, adata.n_obs - 1))

    return adata
