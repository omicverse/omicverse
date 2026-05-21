"""Bulk immune-repertoire analysis — thin wrappers over :mod:`pyimmunarch`.

These functions cover the bulk AIRR-seq side: diversity, overlap, gene usage,
clonality, public clonotypes and clonotype tracking, computed on the
*immunarch* data model — a list of per-sample repertoire DataFrames plus
sample metadata (the :class:`pyimmunarch.ImmunData` container).

The :mod:`pyimmunarch` backend is imported lazily, so ``import omicverse.airr``
succeeds even without the optional ``omicverse[airr]`` extra.
"""
from __future__ import annotations

from typing import Optional

from .._registry import register_function


def _require_immunarch():
    """Lazy-import :mod:`pyimmunarch` with an actionable error message."""
    try:
        import pyimmunarch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Bulk repertoire analysis needs the 'pyimmunarch' backend. "
            "Install with:  pip install omicverse[airr]   (or "
            "pip install pyimmunarch)."
        ) from exc
    return pyimmunarch


@register_function(
    aliases=["repertoire_diversity", "bulk_diversity", "组库多样性", "批量多样性"],
    category="airr",
    description=(
        "Bulk repertoire diversity via pyimmunarch.repDiversity. method "
        "selects 'chao1', 'hill', 'div' (true diversity), 'gini.simp', "
        "'inv.simp', 'gini', 'dxx', 'd50' or 'raref' (rarefaction)."
    ),
    examples=[
        "df = ov.airr.repertoire_diversity(immdata, method='chao1')",
        "df = ov.airr.repertoire_diversity(immdata, method='hill')",
    ],
    related=["airr.clonality", "airr.alpha_diversity"],
)
def repertoire_diversity(data, *, method: str = "chao1", col: str = "aa",
                         **kwargs):
    """Bulk repertoire diversity (``pyimmunarch.repDiversity``).

    Parameters
    ----------
    data
        A :class:`pyimmunarch.ImmunData` (or a list of per-sample repertoire
        DataFrames).
    method
        Diversity estimator — ``'chao1'`` | ``'hill'`` | ``'div'`` |
        ``'gini.simp'`` | ``'inv.simp'`` | ``'gini'`` | ``'dxx'`` |
        ``'d50'`` | ``'raref'``.
    col
        Clonotype-defining column — ``'aa'`` (CDR3 AA, default), ``'nt'``
        or ``'aa+v'``.
    **kwargs
        Forwarded to :func:`pyimmunarch.repDiversity` (``q``, ``max_q`` …).

    Returns
    -------
    :class:`pandas.DataFrame`
    """
    pim = _require_immunarch()
    return pim.repDiversity(data, method=method, col=col, **kwargs)


@register_function(
    aliases=["repertoire_overlap_bulk", "bulk_overlap", "批量组库重叠"],
    category="airr",
    description=(
        "Bulk repertoire overlap matrix via pyimmunarch.repOverlap "
        "(public / overlap / jaccard / tversky / cosine / morisita)."
    ),
    examples=[
        "mat = ov.airr.repertoire_overlap_bulk(immdata, method='jaccard')",
    ],
    related=["airr.repertoire_diversity", "airr.public_clonotypes"],
)
def repertoire_overlap_bulk(data, *, method: str = "public", col: str = "aa",
                            **kwargs):
    """Bulk repertoire-overlap matrix (``pyimmunarch.repOverlap``).

    Parameters
    ----------
    data
        A :class:`pyimmunarch.ImmunData`.
    method
        ``'public'`` | ``'overlap'`` | ``'jaccard'`` | ``'tversky'`` |
        ``'cosine'`` | ``'morisita'``.
    col
        Clonotype column (``'aa'`` / ``'nt'`` / ``'aa+v'``).
    **kwargs
        Forwarded to :func:`pyimmunarch.repOverlap`.

    Returns
    -------
    :class:`pandas.DataFrame`
        A symmetric sample x sample overlap matrix.
    """
    pim = _require_immunarch()
    return pim.repOverlap(data, method=method, col=col, **kwargs)


@register_function(
    aliases=["gene_usage_bulk", "bulk_gene_usage", "批量基因使用"],
    category="airr",
    description=(
        "Bulk V/D/J gene-segment usage table via pyimmunarch.geneUsage."
    ),
    examples=[
        "df = ov.airr.gene_usage_bulk(immdata, gene='hs.trbv', norm=True)",
    ],
    related=["airr.vdj_usage", "airr.repertoire_diversity"],
)
def gene_usage_bulk(data, *, gene: str = "hs.trbv", norm: bool = False,
                    **kwargs):
    """Bulk V/D/J gene usage (``pyimmunarch.geneUsage``).

    Parameters
    ----------
    data
        A :class:`pyimmunarch.ImmunData`.
    gene
        Gene-segment specifier, e.g. ``'hs.trbv'``, ``'hs.trbj'``,
        ``'hs.ighv'``.
    norm
        Normalise the counts to frequencies.
    **kwargs
        Forwarded to :func:`pyimmunarch.geneUsage`.

    Returns
    -------
    :class:`pandas.DataFrame`
    """
    pim = _require_immunarch()
    return pim.geneUsage(data, gene=gene, norm=norm, **kwargs)


@register_function(
    aliases=["clonality", "repertoire_clonality", "组库克隆性", "克隆空间"],
    category="airr",
    description=(
        "Bulk clonal-space analysis via pyimmunarch.repClonality. method "
        "selects 'clonal.prop', 'homeo' (homeostasis), 'top' or 'rare'."
    ),
    examples=[
        "df = ov.airr.clonality(immdata, method='homeo')",
        "df = ov.airr.clonality(immdata, method='clonal.prop')",
    ],
    related=["airr.repertoire_diversity", "airr.clonal_expansion"],
)
def clonality(data, *, method: str = "clonal.prop", **kwargs):
    """Bulk clonal-space analysis (``pyimmunarch.repClonality``).

    Parameters
    ----------
    data
        A :class:`pyimmunarch.ImmunData`.
    method
        ``'clonal.prop'`` | ``'homeo'`` | ``'top'`` | ``'rare'``.
    **kwargs
        Forwarded to :func:`pyimmunarch.repClonality` (``perc``,
        ``clone_types`` …).

    Returns
    -------
    :class:`pandas.DataFrame`
    """
    pim = _require_immunarch()
    return pim.repClonality(data, method=method, **kwargs)


@register_function(
    aliases=["public_clonotypes", "pubrep", "公共克隆型", "公共组库"],
    category="airr",
    description=(
        "Build the public-repertoire table — clonotypes shared across "
        "samples — via pyimmunarch.pubRep."
    ),
    examples=[
        "pr = ov.airr.public_clonotypes(immdata, col='aa+v')",
    ],
    related=["airr.repertoire_overlap_bulk", "airr.track_clonotypes"],
)
def public_clonotypes(data, *, col: str = "aa+v", quant: str = "count",
                      **kwargs):
    """Public-repertoire table (``pyimmunarch.pubRep``).

    Parameters
    ----------
    data
        A :class:`pyimmunarch.ImmunData`.
    col
        Clonotype-defining column (``'aa+v'`` default).
    quant
        ``'count'`` or ``'prop'``.
    **kwargs
        Forwarded to :func:`pyimmunarch.pubRep` (``min_samples`` …).

    Returns
    -------
    :class:`pandas.DataFrame`
    """
    pim = _require_immunarch()
    return pim.pubRep(data, col=col, quant=quant, **kwargs)


@register_function(
    aliases=["track_clonotypes", "clonotype_tracking", "克隆型追踪", "克隆动态"],
    category="airr",
    description=(
        "Track the abundance of selected clonotypes across samples / "
        "time-points via pyimmunarch.trackClonotypes."
    ),
    examples=[
        "df = ov.airr.track_clonotypes(immdata, which=(1, 15))",
    ],
    related=["airr.public_clonotypes", "airr.repertoire_diversity"],
)
def track_clonotypes(data, *, which=(1, 15), col: str = "aa",
                     norm: bool = True, **kwargs):
    """Track clonotype abundance across samples (``pyimmunarch.trackClonotypes``).

    Parameters
    ----------
    data
        A :class:`pyimmunarch.ImmunData`.
    which
        Clonotype selector — passed straight to immunarch's ``trackClonotypes``
        (e.g. ``(sample_index, n_top)`` or a list of CDR3 sequences).
    col
        Clonotype column (``'aa'`` / ``'nt'``).
    norm
        Normalise abundances to frequencies.
    **kwargs
        Forwarded to :func:`pyimmunarch.trackClonotypes`.

    Returns
    -------
    :class:`pandas.DataFrame`
    """
    pim = _require_immunarch()
    return pim.trackClonotypes(data, which=which, col=col, norm=norm, **kwargs)


@register_function(
    aliases=["simulate_immdata", "airr_simulate_immdata", "模拟批量组库"],
    category="airr",
    description=(
        "Simulate a small bulk-repertoire cohort as a pyimmunarch.ImmunData "
        "(several samples, two groups, power-law clone sizes) for tutorials "
        "and tests — no external download required."
    ),
    examples=[
        "immdata = ov.airr.simulate_immdata(n_samples=6, receptor='TCR')",
    ],
    related=["airr.repertoire_diversity", "airr.load_example_immdata"],
)
def simulate_immdata(
    n_samples: int = 6,
    n_clones: int = 200,
    receptor: str = "TCR",
    seed: int = 0,
):
    """Simulate a bulk-repertoire :class:`pyimmunarch.ImmunData`.

    Each sample draws clonotypes from a shared pool with power-law clone
    sizes, so samples share public clonotypes and differ in private ones —
    realistic input for diversity / overlap / clonality / public-repertoire
    analyses.

    Parameters
    ----------
    n_samples
        Number of repertoire samples.
    n_clones
        Size of the shared clonotype pool.
    receptor
        ``'TCR'`` or ``'BCR'`` — controls the V/J gene names.
    seed
        Random seed.

    Returns
    -------
    :class:`pyimmunarch.ImmunData`
        ``.data`` holds one repertoire DataFrame per sample;
        ``.meta`` carries a two-level ``group`` column.
    """
    import numpy as np
    import pandas as pd
    from collections import OrderedDict

    pim = _require_immunarch()
    rng = np.random.default_rng(seed)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    nt = list("ACGT")
    if receptor.upper() == "TCR":
        v_genes = [f"TRBV{i}-1" for i in range(1, 21)]
        j_genes = [f"TRBJ{i}-{k}" for i in range(1, 3) for k in range(1, 6)]
        d_genes = [f"TRBD{i}" for i in range(1, 3)]
    else:
        v_genes = [f"IGHV{i}-1" for i in range(1, 21)]
        j_genes = [f"IGHJ{i}" for i in range(1, 7)]
        d_genes = [f"IGHD{i}-1" for i in range(1, 7)]

    pool = []
    for _ in range(n_clones):
        L = int(rng.integers(11, 17))
        pool.append({
            "CDR3.aa": "".join(rng.choice(aa, L)),
            "CDR3.nt": "".join(rng.choice(nt, L * 3)),
            "V.name": rng.choice(v_genes),
            "D.name": rng.choice(d_genes),
            "J.name": rng.choice(j_genes),
        })

    data = OrderedDict()
    for s in range(n_samples):
        k = int(rng.integers(int(n_clones * 0.4), int(n_clones * 0.8)))
        idx = rng.choice(n_clones, size=k, replace=False)
        w = 1.0 / (np.arange(1, k + 1) ** rng.uniform(1.0, 1.4))
        counts = rng.multinomial(int(rng.integers(2000, 6000)), w / w.sum())
        # guarantee a tail of singleton clones — Chao1 / rarefaction need them
        n_single = max(int(k * 0.15), 3)
        counts[-n_single:] = 1
        rows = []
        for c, i in zip(counts, idx):
            if c == 0:
                continue
            r = dict(pool[i])
            r["Clones"] = int(c)
            rows.append(r)
        df = pd.DataFrame(rows)
        df["Proportion"] = df["Clones"] / df["Clones"].sum()
        df = df[["Clones", "Proportion", "CDR3.nt", "CDR3.aa",
                 "V.name", "D.name", "J.name"]]
        data[f"sample_{s + 1}"] = df.reset_index(drop=True)

    meta = pd.DataFrame({
        "Sample": list(data.keys()),
        "group": ["group_A" if i % 2 == 0 else "group_B"
                  for i in range(n_samples)],
    })
    return pim.ImmunData(data, meta)


@register_function(
    aliases=["load_example_immdata", "airr_example_immdata", "示例组库数据"],
    category="airr",
    description=(
        "Load the bundled example bulk-repertoire cohort shipped with "
        "pyimmunarch — a small TCR ImmunData for tutorials / tests."
    ),
    examples=[
        "immdata = ov.airr.load_example_immdata()",
    ],
    related=["airr.repertoire_diversity", "airr.simulate_immdata"],
)
def load_example_immdata(extdata_dir: Optional[str] = None):
    """Load the bundled example bulk TCR cohort (``pyimmunarch``).

    The per-sample count / proportion columns are repaired if the bundled
    loader leaves them all-NA: each unique clonotype row is assigned a unit
    count and the proportion is recomputed, so count-dependent analyses
    (:func:`clonality`, :func:`public_clonotypes`) run cleanly.

    Parameters
    ----------
    extdata_dir
        Optional override directory for the example files.

    Returns
    -------
    :class:`pyimmunarch.ImmunData`
    """
    import pandas as pd

    pim = _require_immunarch()
    imm = pim.load_example_immdata(extdata_dir)
    count_col = getattr(pim.IMMCOL, "count", "Clones")
    prop_col = getattr(pim.IMMCOL, "prop", "Proportion")
    data = getattr(imm, "data", None)
    if isinstance(data, dict):
        for name, df in data.items():
            counts = pd.to_numeric(df.get(count_col), errors="coerce")
            if counts is None or counts.isna().all():
                df[count_col] = 1
                df[prop_col] = 1.0 / max(len(df), 1)
    return imm
