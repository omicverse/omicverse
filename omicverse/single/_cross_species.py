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

All methods share one entry point — :class:`CrossSpecies` (and the
:func:`cross_species_integrate` convenience wrapper) — selected via ``method=``:

* ortholog route (default): one-to-one Ensembl orthologs + omicverse's
  :func:`batch_correction` backends (``'harmony'``, ``'scVI'``, ...).
* ``method='samap'`` — **SAMap** (reciprocal-BLAST gene-homology graph +
  Self-Assembling Manifold), vendored under ``omicverse.external.samap``.
* ``method='saturn'`` — **SATURN** (ESM2 protein-embedding macrogenes),
  vendored under ``omicverse.external.saturn``.

The ortholog *preprocessing* (align + concat + normalize/HVG/scale/PCA) lives in
:func:`omicverse.tl.cross_species_align`.
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


# method values that route to the ortholog route (one-to-one orthologs +
# ov.single.batch_correction). Anything not 'samap'/'saturn' is treated as an
# ortholog backend and forwarded to batch_correction as its ``methods=``.
_HOMOLOGY_METHODS = {"samap", "saturn"}


@register_function(
    aliases=["跨物种整合类", "CrossSpecies", "cross species class", "跨物种整合对象"],
    category="single",
    description="Cross-species single-cell integration (orthologs / SAMap / SATURN) with save/load",
    examples=[
        "cs = ov.single.CrossSpecies([a1, a2], ['human','mouse'], method='scVI').fit()",
        "cs = ov.single.CrossSpecies([zeb, frog], ['zebrafish','frog'], method='saturn', embedding_paths=...).fit()",
        "ov.io.save(cs, 'run.pkl'); cs = ov.io.load('run.pkl')",
    ],
    related=["single.cross_species_integrate", "tl.cross_species_align", "io.save"],
)
class CrossSpecies:
    """Cross-species single-cell integration with pluggable methods.

    One object, one ``method=`` — dispatches to the ortholog route or to a
    vendored homology-aware backend, and can be persisted with
    :func:`omicverse.io.save` / :func:`omicverse.io.load` (e.g. to reuse a
    trained SATURN model).

    Parameters
    ----------
    adatas
        One ``AnnData`` per species (**raw counts**, gene *symbols* as
        ``var_names``).
    species
        Species name per entry of ``adatas`` (same order).
    method
        Integration backend:

        * **ortholog route** (one-to-one Ensembl orthologs, then
          :func:`batch_correction`): ``'harmony'`` (default), ``'scVI'``,
          ``'Concord'``, ``'cca'``, ``'combat'``, ``'scanorama'``, ...
        * ``'samap'`` — **SAMap** (reciprocal-BLAST homology graph). Needs
          ``proteomes=`` and ``blast_maps=`` (and ``ids=`` short species IDs);
          see :func:`omicverse.external.samap._run.samap_integrate`.
        * ``'saturn'`` — **SATURN** (ESM2 protein-embedding macrogenes). Needs
          ``embedding_paths=``; see
          :func:`omicverse.external.saturn._run.run_saturn`.
    ref_species, batch_key, orthology_type, n_top_genes, n_pcs, target_sum,
    host, cache_dir, use_cache
        Ortholog-route settings (forwarded to :func:`ov.tl.cross_species_align`).
    **kwargs
        Backend-specific arguments — extra ``batch_correction`` kwargs for the
        ortholog route, or ``proteomes``/``blast_maps``/``ids``/``embedding_paths``
        for SAMap / SATURN.

    Attributes
    ----------
    adata : the integrated ``AnnData`` (after :meth:`fit`).
    model : a trained backend model when the method exposes one (e.g. SATURN),
        else ``None``.
    """

    def __init__(
        self,
        adatas: Sequence[anndata.AnnData],
        species: Sequence[str],
        *,
        method: str = "harmony",
        ref_species: Optional[str] = None,
        batch_key: str = "species",
        orthology_type: str = "one2one",
        n_top_genes: int = 2000,
        n_pcs: int = 50,
        target_sum: float = 1e4,
        host: str = "https://www.ensembl.org",
        cache_dir: Optional[str] = "data/orthologs",
        use_cache: bool = True,
        **kwargs,
    ):
        if len(adatas) != len(species):
            raise ValueError("`adatas` and `species` must have the same length.")
        if len(adatas) < 2:
            raise ValueError("Need at least two datasets to integrate across species.")
        self.adatas = list(adatas)
        self.species = list(species)
        self.method = method
        self.ref_species = ref_species
        self.batch_key = batch_key
        self.orthology_type = orthology_type
        self.n_top_genes = n_top_genes
        self.n_pcs = n_pcs
        self.target_sum = target_sum
        self.host = host
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.kwargs = dict(kwargs)
        self.adata = None
        self.model = None

    def fit(self) -> "CrossSpecies":
        """Run the integration; populate ``self.adata`` (and ``self.model``)."""
        m = str(self.method).lower()
        if m == "samap":
            self._fit_samap()
        elif m == "saturn":
            self._fit_saturn()
        else:
            self._fit_ortholog()
        return self

    # ``run`` kept as an alias for backwards habit / readability.
    run = fit

    def _fit_ortholog(self) -> None:
        from ..tl import cross_species_align
        from ._batch import batch_correction

        adata = cross_species_align(
            self.adatas, self.species, ref_species=self.ref_species,
            batch_key=self.batch_key, orthology_type=self.orthology_type,
            n_top_genes=self.n_top_genes, n_pcs=self.n_pcs,
            target_sum=self.target_sum, host=self.host,
            cache_dir=self.cache_dir, use_cache=self.use_cache)
        adata.uns.setdefault("cross_species", {})["method"] = self.method
        batch_correction(adata, batch_key=self.batch_key, methods=self.method,
                         n_pcs=self.n_pcs, **self.kwargs)
        add_reference(adata, "cross-species",
                      "cross-species integration via one-to-one Ensembl orthologs")
        self.adata = adata

    def _fit_samap(self) -> None:
        from ..external.samap._run import samap_integrate as _samap

        kw = dict(self.kwargs)
        ids = kw.pop("ids", self.species)
        ensembl_species = kw.pop("ensembl_species", self.species)
        self.adata = _samap(self.adatas, ids, ensembl_species, **kw)
        self.adata.uns.setdefault("cross_species", {})["method"] = "samap"

    def _fit_saturn(self) -> None:
        from ..external.saturn._run import run_saturn as _saturn

        res = _saturn(self.adatas, self.species, return_model=True, **self.kwargs)
        if isinstance(res, tuple):
            self.adata, self.model = res
        else:
            self.adata = res
        self.adata.uns.setdefault("cross_species", {})["method"] = "saturn"

    def save(self, path: str) -> str:
        """Persist this run (config + integrated AnnData + model) to ``path``
        via :func:`omicverse.io.save` (pickle, cloudpickle fallback)."""
        from ..io import save as _save
        _save(self, path)
        return path

    @classmethod
    def load(cls, path: str) -> "CrossSpecies":
        """Load a run previously written by :meth:`save` (``ov.io.load``)."""
        from ..io import load as _load
        return _load(path)


@register_function(
    aliases=["跨物种整合", "cross_species_integrate", "cross species integration",
             "跨物种数据整合"],
    category="single",
    description="Integrate single-cell datasets across species (orthologs / SAMap / SATURN)",
    examples=[
        "adata = ov.single.cross_species_integrate([a1,a2], ['human','mouse'], method='scVI')",
        "adata = ov.single.cross_species_integrate([zeb,frog], ['zebrafish','frog'], method='samap', ids=['zf','fr'], proteomes=..., blast_maps='maps/')",
        "adata = ov.single.cross_species_integrate([zeb,frog], ['zebrafish','frog'], method='saturn', embedding_paths=...)",
    ],
    related=["single.CrossSpecies", "single.get_orthologs", "tl.cross_species_align"],
)
def cross_species_integrate(
    adatas: Sequence[anndata.AnnData],
    species: Sequence[str],
    *,
    method: str = "harmony",
    **kwargs,
) -> anndata.AnnData:
    """Integrate single-cell datasets **across species**.

    Thin wrapper over :class:`CrossSpecies`: pick the backend with ``method=``
    (ortholog backends such as ``'scVI'``/``'harmony'``, or ``'samap'`` /
    ``'saturn'``), fit, and return the integrated ``AnnData``. Use
    :class:`CrossSpecies` directly if you also need the trained model or
    save/load. See :class:`CrossSpecies` for all parameters.

    Returns
    -------
    anndata.AnnData
        The integrated object (``.uns['cross_species']`` records the method and,
        for the ortholog route, the reference species and ortholog count).
    """
    return CrossSpecies(adatas, species, method=method, **kwargs).fit().adata
