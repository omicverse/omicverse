"""Cross-species single-cell integration via one-to-one orthologs (Ensembl BioMart).

Answer to issue #856: bring cross-species integration + homologous-gene transfer
into omicverse. This module implements the *ortholog-based* route — the standard,
lightweight cross-species workflow — reusing omicverse's existing integration
backends (Harmony / scVI / Concord / Seurat-CCA, via
:func:`omicverse.single.batch_correction`).

Workflow
--------
1. :func:`get_orthologs` — fetch an ortholog table between two species from
   Ensembl BioMart (``martservice`` REST over HTTPS, no external ``biomart``
   package needed), cached to disk.
2. :func:`map_var_to_orthologs` — rename an ``AnnData``'s genes to the target
   species' one-to-one ortholog symbols (transfer of homologous genes).
3. :func:`cross_species_integrate` — align every dataset to a reference species'
   ortholog space, concatenate on the shared genes, and run a batch-integration
   backend *across species*.

By default only **one-to-one** orthologs are used: they give an unambiguous
shared gene axis, which is the conservative, widely-used choice for
ortholog-based cross-species integration. For many-to-many homology / very
large evolutionary distances, two heavier backends are vendored under
``omicverse.external`` and exposed here:

* :func:`samap_integrate` — **SAMap** (reciprocal-BLAST gene-homology graph +
  Self-Assembling Manifold).
* :func:`saturn_integrate` — **SATURN** (ESM2 protein-embedding macrogenes).
"""
from __future__ import annotations

import io
import os
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import anndata

from .._settings import add_reference

try:  # optional, only used for the registry decorators (kept soft)
    from .._registry import register_function
except Exception:  # pragma: no cover
    def register_function(*a, **k):  # type: ignore
        def _wrap(f):
            return f
        return _wrap


# Common names → Ensembl BioMart dataset names. Extend freely; any dataset name
# accepted by Ensembl BioMart also works when passed directly (e.g.
# ``"sscrofa_gene_ensembl"``).
_SPECIES_DATASETS = {
    "human": "hsapiens_gene_ensembl", "hsapiens": "hsapiens_gene_ensembl", "hs": "hsapiens_gene_ensembl",
    "mouse": "mmusculus_gene_ensembl", "mmusculus": "mmusculus_gene_ensembl", "mm": "mmusculus_gene_ensembl",
    "rat": "rnorvegicus_gene_ensembl", "rnorvegicus": "rnorvegicus_gene_ensembl",
    "zebrafish": "drerio_gene_ensembl", "drerio": "drerio_gene_ensembl",
    "chicken": "ggallus_gene_ensembl", "ggallus": "ggallus_gene_ensembl",
    "dog": "clfamiliaris_gene_ensembl", "clfamiliaris": "clfamiliaris_gene_ensembl",
    "pig": "sscrofa_gene_ensembl", "sscrofa": "sscrofa_gene_ensembl",
    "cow": "btaurus_gene_ensembl", "cattle": "btaurus_gene_ensembl", "btaurus": "btaurus_gene_ensembl",
    "macaque": "mmulatta_gene_ensembl", "rhesus": "mmulatta_gene_ensembl", "mmulatta": "mmulatta_gene_ensembl",
    "marmoset": "cjacchus_gene_ensembl", "cjacchus": "cjacchus_gene_ensembl",
    "fly": "dmelanogaster_gene_ensembl", "drosophila": "dmelanogaster_gene_ensembl",
    "dmelanogaster": "dmelanogaster_gene_ensembl",
    "frog": "xtropicalis_gene_ensembl", "xenopus": "xtropicalis_gene_ensembl",
    "worm": "celegans_gene_ensembl", "celegans": "celegans_gene_ensembl",
    "rabbit": "ocuniculus_gene_ensembl", "ocuniculus": "ocuniculus_gene_ensembl",
}


def _normalize_species(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _resolve_dataset(species: str) -> str:
    """Return the Ensembl BioMart dataset name for ``species``.

    Accepts a common name (``"human"``), an Ensembl prefix (``"hsapiens"``), or a
    full dataset name (``"hsapiens_gene_ensembl"``).
    """
    s = _normalize_species(species)
    if s in _SPECIES_DATASETS:
        return _SPECIES_DATASETS[s]
    if s.endswith("_gene_ensembl"):
        return s
    raise ValueError(
        f"Unknown species {species!r}. Pass one of {sorted(set(_SPECIES_DATASETS))}, "
        "an Ensembl prefix like 'hsapiens', or a full '<prefix>_gene_ensembl' name."
    )


# Ensembl mirrors, tried in order when a host is unreachable / in maintenance.
_ENSEMBL_MIRRORS = (
    "https://www.ensembl.org",
    "https://asia.ensembl.org",
    "https://useast.ensembl.org",
)


def _biomart_query(dataset: str, attributes: Sequence[str], *,
                   host: str = "https://www.ensembl.org", timeout: int = 120) -> pd.DataFrame:
    """Run a BioMart ``martservice`` query and return a TSV DataFrame.

    Uses the REST XML API directly over ``requests`` (HTTPS) — the ``biomart``
    PyPI package mishandles ``https://`` URLs, and Ensembl now returns 403 for
    plain ``http``. If the requested ``host`` is unreachable or serving a
    maintenance page, the other Ensembl mirrors are tried before giving up.
    """
    import requests

    attrs = list(attributes)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<!DOCTYPE Query>'
        '<Query virtualSchemaName="default" formatter="TSV" header="0" '
        'uniqueRows="1" count="" datasetConfigVersion="0.6">'
        f'<Dataset name="{dataset}" interface="default">'
        + "".join(f'<Attribute name="{a}"/>' for a in attrs)
        + "</Dataset></Query>"
    )

    hosts = [host] + [m for m in _ENSEMBL_MIRRORS if m.rstrip("/") != host.rstrip("/")]
    errors = []
    for h in hosts:
        try:
            resp = requests.get(
                h.rstrip("/") + "/biomart/martservice",
                params={"query": xml},
                headers={"User-Agent": "Mozilla/5.0 (omicverse cross-species)"},
                timeout=timeout)
            resp.raise_for_status()
            text = resp.text
            low = text.lstrip().lower()
            # A valid TSV never starts with an HTML/status/error page.
            if low.startswith(("query error", "<html", "<!doctype", "error")):
                raise RuntimeError(f"non-tabular response from {h}: {text[:120]!r}")
            df = pd.read_csv(io.StringIO(text), sep="\t", header=None,
                             names=attrs, dtype=str)
            if df.shape[1] != len(attrs):
                raise RuntimeError(f"unexpected column count from {h}")
            return df
        except Exception as exc:  # try the next mirror
            errors.append(f"{h}: {type(exc).__name__} {str(exc)[:80]}")
            continue
    raise RuntimeError(
        "BioMart query failed on all Ensembl mirrors for dataset "
        f"{dataset!r}:\n  " + "\n  ".join(errors))


@register_function(
    aliases=["跨物种同源基因", "get_orthologs", "ortholog table", "同源基因映射", "cross-species orthologs"],
    category="single",
    description="Fetch a cross-species ortholog mapping table from Ensembl BioMart",
    examples=[
        "orthologs = ov.single.get_orthologs('human', 'mouse')",
        "orthologs = ov.single.get_orthologs('mouse', 'human', orthology_type='one2one')",
    ],
    related=["single.cross_species_integrate", "single.map_var_to_orthologs"],
)
def get_orthologs(
    source_species: str,
    target_species: str,
    *,
    orthology_type: str = "one2one",
    host: str = "https://www.ensembl.org",
    cache_dir: Optional[str] = "data/orthologs",
    use_cache: bool = True,
    timeout: int = 120,
) -> pd.DataFrame:
    """Fetch a ``source → target`` ortholog table from Ensembl BioMart.

    Parameters
    ----------
    source_species, target_species
        Species names (``"human"``/``"mouse"``/...), Ensembl prefixes, or full
        ``<prefix>_gene_ensembl`` dataset names. See ``_SPECIES_DATASETS``.
    orthology_type
        ``"one2one"`` (default) keeps only unambiguous one-to-one orthologs.
        ``"all"`` keeps every ortholog regardless of type. Any explicit Ensembl
        value (e.g. ``"ortholog_one2many"``) filters to that type.
    host
        Ensembl mirror, e.g. ``"https://asia.ensembl.org"`` or
        ``"https://useast.ensembl.org"``.
    cache_dir, use_cache
        Where to cache the fetched table (Parquet-free TSV). ``use_cache=False``
        forces a fresh BioMart query.
    timeout
        Per-request timeout (seconds).

    Returns
    -------
    pandas.DataFrame
        Columns ``source_symbol``, ``target_symbol``, ``orthology_type``.
    """
    src_ds = _resolve_dataset(source_species)
    tgt_prefix = _resolve_dataset(target_species).split("_gene_ensembl")[0]

    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir, f"{src_ds}__to__{tgt_prefix}__{orthology_type}.tsv")
        if use_cache and os.path.exists(cache_path):
            return pd.read_csv(cache_path, sep="\t", dtype=str)

    attrs = [
        "external_gene_name",
        f"{tgt_prefix}_homolog_associated_gene_name",
        f"{tgt_prefix}_homolog_orthology_type",
    ]
    df = _biomart_query(src_ds, attrs, host=host, timeout=timeout)
    df.columns = ["source_symbol", "target_symbol", "orthology_type"]

    df = df.dropna(subset=["source_symbol", "target_symbol"])
    df = df[(df["source_symbol"] != "") & (df["target_symbol"] != "")]
    if orthology_type not in (None, "all"):
        want = orthology_type
        if not want.startswith("ortholog_"):
            want = "ortholog_" + want
        df = df[df["orthology_type"] == want]
    df = (df.drop_duplicates(subset=["source_symbol", "target_symbol"])
            .reset_index(drop=True))

    if cache_path is not None:
        df.to_csv(cache_path, sep="\t", index=False)
    return df


@register_function(
    aliases=["同源基因转换", "map_var_to_orthologs", "ortholog transfer", "基因物种转换"],
    category="single",
    description="Rename an AnnData's genes to another species' ortholog symbols",
    examples=[
        "adata_mm = ov.single.map_var_to_orthologs(adata_hs, 'human', 'mouse')",
    ],
    related=["single.get_orthologs", "single.cross_species_integrate"],
)
def map_var_to_orthologs(
    adata: anndata.AnnData,
    source_species: str,
    target_species: str,
    *,
    ortho: Optional[pd.DataFrame] = None,
    orthology_type: str = "one2one",
    collapse: str = "sum",
    copy: bool = True,
    **ortho_kwargs,
) -> anndata.AnnData:
    """Transfer an ``AnnData`` onto the *target* species' gene symbols.

    Subsets ``adata`` to genes that have a (one-to-one, by default) ortholog in
    the target species and renames ``var_names`` to the target symbols. Genes
    without an ortholog are dropped. On the rare duplicate target symbol,
    columns are collapsed by ``collapse`` (``"sum"`` or ``"mean"``).

    Parameters
    ----------
    adata
        Source-species data (genes = ``var_names`` as symbols).
    source_species, target_species
        See :func:`get_orthologs`.
    ortho
        Precomputed ortholog table (columns ``source_symbol``/``target_symbol``).
        When ``None`` it is fetched via :func:`get_orthologs`.
    orthology_type, collapse, copy
        ``orthology_type`` forwarded to :func:`get_orthologs`; ``collapse`` how to
        merge duplicate target genes; ``copy`` return a copy (always True here).
    **ortho_kwargs
        Forwarded to :func:`get_orthologs` (e.g. ``host``, ``cache_dir``).

    Returns
    -------
    anndata.AnnData
        A new object on the target species' gene space.
    """
    if ortho is None:
        ortho = get_orthologs(source_species, target_species,
                              orthology_type=orthology_type, **ortho_kwargs)
    mapper = dict(zip(ortho["source_symbol"].astype(str),
                      ortho["target_symbol"].astype(str)))

    keep = [g for g in adata.var_names if g in mapper]
    if not keep:
        raise ValueError(
            f"No genes of the input match {source_species}->{target_species} "
            f"orthologs. Are var_names gene symbols for {source_species}?")

    ad = adata[:, keep].copy()
    ad.var["source_symbol"] = list(ad.var_names)
    ad.var_names = pd.Index([mapper[g] for g in keep])

    if ad.var_names.duplicated().any():
        ad = _collapse_duplicate_vars(ad, how=collapse)
    return ad


def _collapse_duplicate_vars(adata: anndata.AnnData, how: str = "sum") -> anndata.AnnData:
    """Merge columns that share a ``var_name`` (sum/mean of expression)."""
    import scipy.sparse as sp

    names = pd.Index(adata.var_names)
    uniq = names.unique()
    if len(uniq) == len(names):
        return adata
    X = adata.X
    dense = not sp.issparse(X)
    cols = []
    for g in uniq:
        idx = np.where(names == g)[0]
        block = X[:, idx]
        if how == "mean":
            v = block.mean(axis=1)
        else:
            v = block.sum(axis=1)
        v = np.asarray(v).reshape(-1, 1)
        cols.append(v)
    newX = np.hstack(cols)
    if not dense:
        newX = sp.csr_matrix(newX)

    # Preserve per-gene var metadata: first row of each group, and join the
    # collapsed source symbols so provenance survives.
    var = adata.var.copy()
    var.index = names
    grouped = var.groupby(level=0, sort=False)
    new_var = grouped.first()
    if "source_symbol" in var.columns:
        new_var["source_symbol"] = grouped["source_symbol"].agg(
            lambda s: ",".join(map(str, s)))
    new_var = new_var.loc[list(uniq)]
    new_var.index = pd.Index(list(uniq), name=adata.var.index.name)

    out = anndata.AnnData(X=newX, obs=adata.obs.copy(), var=new_var)
    out.obsm = adata.obsm.copy()
    out.obs_names = adata.obs_names
    return out


@register_function(
    aliases=["跨物种整合", "cross_species_integrate", "cross species integration",
             "跨物种数据整合", "SAMap alternative"],
    category="single",
    description="Integrate single-cell datasets across species via one-to-one orthologs",
    examples=[
        "adata = ov.single.cross_species_integrate([adata_hs, adata_mm], ['human','mouse'])",
        "adata = ov.single.cross_species_integrate([a1,a2], ['human','mouse'], method='scVI')",
    ],
    related=["single.get_orthologs", "single.map_var_to_orthologs", "single.batch_correction"],
)
def cross_species_integrate(
    adatas: Sequence[anndata.AnnData],
    species: Sequence[str],
    *,
    ref_species: Optional[str] = None,
    batch_key: str = "species",
    method: str = "harmony",
    orthology_type: str = "one2one",
    n_top_genes: int = 2000,
    n_pcs: int = 50,
    target_sum: float = 1e4,
    preprocess: bool = True,
    host: str = "https://www.ensembl.org",
    cache_dir: Optional[str] = "data/orthologs",
    use_cache: bool = True,
    **integration_kwargs,
) -> anndata.AnnData:
    """Integrate multiple single-cell datasets **across species**.

    Every dataset is mapped onto the reference species' gene symbols via
    one-to-one Ensembl orthologs, restricted to the genes shared by all
    datasets, concatenated with a ``species`` batch label, and integrated with
    one of omicverse's :func:`batch_correction` backends.

    Parameters
    ----------
    adatas
        One ``AnnData`` per species, with **raw counts** in ``.X`` and gene
        *symbols* as ``var_names``.
    species
        Species name for each entry of ``adatas`` (same length/order).
    ref_species
        Species whose gene symbols become the common axis. Defaults to
        ``species[0]``.
    batch_key
        ``obs`` column written with the species label and used as the
        integration batch.
    method
        Integration backend forwarded to :func:`batch_correction`
        (``"harmony"``, ``"scVI"``, ``"Concord"``, ``"cca"``, ...).
    orthology_type
        Forwarded to :func:`get_orthologs` (default ``"one2one"``).
    n_top_genes, n_pcs, target_sum
        Standard HVG / PCA / normalization settings applied when
        ``preprocess=True``.
    preprocess
        When ``True`` (default) run normalize → log1p → HVG(batch-aware) →
        scale → PCA on the concatenated object before integration. Set
        ``False`` if ``adatas`` are already jointly preprocessed.
    host, cache_dir, use_cache
        Forwarded to :func:`get_orthologs`.
    **integration_kwargs
        Extra keyword arguments for the chosen :func:`batch_correction` method.

    Returns
    -------
    anndata.AnnData
        The integrated object. ``.uns['cross_species']`` records the reference
        species, the ortholog count, and the method; the integrated embedding is
        stored by :func:`batch_correction` (e.g. ``obsm['X_pca_harmony']``).
    """
    import scanpy as sc

    if len(adatas) != len(species):
        raise ValueError("`adatas` and `species` must have the same length.")
    if len(adatas) < 2:
        raise ValueError("Need at least two datasets to integrate across species.")
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
        "method": method,
        "orthology_type": orthology_type,
    }

    if preprocess:
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        adata.raw = adata
        sc.pp.highly_variable_genes(
            adata, n_top_genes=min(n_top_genes, len(common)),
            batch_key=batch_key)
        adata = adata[:, adata.var["highly_variable"]].copy()
        sc.pp.scale(adata, max_value=10)
        sc.pp.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1, adata.n_obs - 1))

    from ._batch import batch_correction
    batch_correction(adata, batch_key=batch_key, methods=method, n_pcs=n_pcs,
                     **integration_kwargs)

    add_reference(adata, 'cross-species',
                  'cross-species integration via one-to-one Ensembl orthologs')
    return adata


@register_function(
    aliases=["SAMap", "samap_integrate", "跨物种SAMap", "cross species SAMap", "同源图整合"],
    category="single",
    description="Cross-species integration with SAMap (BLAST protein-homology graph + SAM)",
    examples=[
        "adata = ov.single.samap_integrate([zeb, frog], ['zf','fr'], ['zebrafish','frog'], proteomes=..., blast_maps='maps/')",
    ],
    related=["single.cross_species_integrate", "single.saturn_integrate", "single.get_orthologs"],
)
def samap_integrate(*args, **kwargs):
    """Cross-species integration with **SAMap** (vendored under
    ``omicverse.external.samap``). Builds a reciprocal-BLAST gene-homology graph
    and runs the Self-Assembling Manifold — handles many-to-many homology and
    large evolutionary distances. See
    :func:`omicverse.external.samap._run.samap_integrate` for the full signature.
    """
    from ..external.samap._run import samap_integrate as _fn
    return _fn(*args, **kwargs)


@register_function(
    aliases=["SATURN", "saturn_integrate", "跨物种SATURN", "cross species SATURN", "蛋白embedding整合"],
    category="single",
    description="Cross-species integration with SATURN (ESM2 protein-embedding macrogenes)",
    examples=[
        "adata = ov.single.saturn_integrate([zeb, frog], ['zebrafish','frog'], embedding_paths=...)",
    ],
    related=["single.cross_species_integrate", "single.samap_integrate", "single.get_orthologs"],
)
def saturn_integrate(*args, **kwargs):
    """Cross-species integration with **SATURN** (vendored under
    ``omicverse.external.saturn``). Learns "macrogenes" from ESM2 protein
    language-model embeddings to map species into a shared space. See
    :func:`omicverse.external.saturn._run.run_saturn` for the full signature.
    """
    from ..external.saturn._run import run_saturn as _fn
    return _fn(*args, **kwargs)
