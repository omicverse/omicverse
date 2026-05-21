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
    aliases=["kmer_analysis", "count_kmers", "k_mer", "K-mer分析", "K-mer计数"],
    category="airr",
    description=(
        "Count k-mer occurrences in CDR3 sequences across samples via "
        "pyimmunarch.getKmers — a sliding window of length k over each "
        "clonotype, giving a k-mer x sample occurrence table."
    ),
    examples=[
        "df = ov.airr.kmer_analysis(immdata, k=3)",
        "df = ov.airr.kmer_analysis(immdata, k=5, col='nt')",
    ],
    related=["airr.kmer_motif", "airr.cdr3_aa_properties"],
)
def kmer_analysis(data, *, k: int = 3, col: str = "aa",
                  coding_only: bool = True):
    """Count k-mer occurrences in CDR3 sequences (``pyimmunarch.getKmers``).

    Slides a window of length ``k`` over every CDR3 clonotype and tallies how
    often each distinct k-mer is observed, per sample — the standard
    immunarch k-mer statistics used to summarise repertoire sequence
    composition and feed motif analysis.

    Parameters
    ----------
    data
        A :class:`pyimmunarch.ImmunData` (or a list/dict of per-sample
        repertoire DataFrames, or a single repertoire DataFrame).
    k
        K-mer length — the window size in residues / nucleotides.
    col
        Sequence column the k-mers are drawn from — ``'aa'`` (CDR3 amino
        acid, default) or ``'nt'`` (CDR3 nucleotide).
    coding_only
        Drop non-coding clonotypes (those carrying stop codons or frame
        shifts) before counting.

    Returns
    -------
    :class:`pandas.DataFrame`
        One row per distinct k-mer (column ``Kmer``) and one occurrence
        column per sample.

    See Also
    --------
    kmer_motif : per-position PFM/PPM/PWM motif profile of equal-length k-mers.
    """
    pim = _require_immunarch()
    return pim.getKmers(data, k, col=col, coding_only=coding_only)


@register_function(
    aliases=["kmer_motif", "kmer_profile", "motif_profile", "K-mer基序", "基序谱"],
    category="airr",
    description=(
        "Build a per-position amino-acid k-mer profile (PFM / PPM / PWM) "
        "from a pyimmunarch.getKmers table for sequence-logo motif plots "
        "via pyimmunarch.kmer_profile."
    ),
    examples=[
        "kmers = ov.airr.kmer_analysis(immdata, k=5)",
        "pfm = ov.airr.kmer_motif(kmers, method='freq')",
        "pwm = ov.airr.kmer_motif(kmers, method='wei')",
    ],
    related=["airr.kmer_analysis", "airr.plotting.kmer_motif_plot"],
)
def kmer_motif(kmers, *, method: str = "freq", remove_stop: bool = True):
    """Per-position amino-acid k-mer profile for motif logos (``pyimmunarch``).

    Collapses a multi-sample :func:`kmer_analysis` table into a single
    set of weighted k-mers and computes a per-position residue profile —
    a position frequency / probability / weight matrix suitable for
    rendering a sequence-logo motif.

    Parameters
    ----------
    kmers
        A :func:`kmer_analysis` / :func:`pyimmunarch.getKmers` output
        (``Kmer`` column plus one or more per-sample occurrence columns),
        or an iterable of equal-length k-mer strings.
    method
        Profile type — ``'freq'`` (position frequency matrix / counts),
        ``'prob'`` (position probability matrix), ``'wei'`` (position
        weight matrix, ``log2(count / n_rows)``) or ``'self'``
        (self-information).
    remove_stop
        Drop k-mers carrying ``*`` / ``~`` characters before profiling.

    Returns
    -------
    :class:`pandas.DataFrame`
        Amino acids (rows) by k-mer position (columns ``V1``, ``V2`` …).

    Notes
    -----
    A multi-sample k-mer table is collapsed to a single ``Kmer`` / ``Count``
    table (occurrences summed across samples) before profiling, because the
    backend profiler operates on one weighted k-mer set at a time.
    """
    import pandas as pd

    pim = _require_immunarch()
    data = kmers
    if isinstance(kmers, pd.DataFrame) and "Kmer" in kmers.columns \
            and kmers.shape[1] > 2:
        sample_cols = [c for c in kmers.columns if c != "Kmer"]
        data = pd.DataFrame({
            "Kmer": kmers["Kmer"].values,
            "Count": kmers[sample_cols].sum(axis=1, skipna=True).values,
        })
    return pim.kmer_profile(data, method=method, remove_stop=remove_stop)


@register_function(
    aliases=["cdr3_aa_properties", "cdr3_aa_profile", "cdr3_property_profile",
             "CDR3理化谱", "CDR3氨基酸谱"],
    category="airr",
    description=(
        "Per-position CDR3 amino-acid composition or physicochemical "
        "property profile of a single repertoire via "
        "pyimmunarch.cdr3_aa_profile."
    ),
    examples=[
        "prof = ov.airr.cdr3_aa_properties(immdata, sample='MS1')",
        "prof = ov.airr.cdr3_aa_properties(rep_df, property='hydropathy')",
    ],
    related=["airr.kmer_motif", "airr.aa_properties"],
)
def cdr3_aa_properties(data, *, sample: Optional[str] = None,
                       property=None,
                       max_len: Optional[int] = None, align: str = "left"):
    """Per-position CDR3 amino-acid property profile (``pyimmunarch``).

    For a single repertoire, builds either a per-position amino-acid
    composition matrix (``property=None``) or a per-position average of a
    chosen physicochemical property — the data behind immunarch's CDR3
    property plots.

    Parameters
    ----------
    data
        A single repertoire :class:`pandas.DataFrame`, or a
        :class:`pyimmunarch.ImmunData` / dict of repertoires — in which
        case ``sample`` selects the repertoire to profile.
    sample
        Sample name to profile when ``data`` is a multi-sample container.
        If ``None`` the first repertoire is used.
    property
        Physicochemical property to average per position — e.g.
        ``'hydropathy'``, ``'charge'``, ``'volume'``, ``'polarity'`` …
        (the keys of :data:`pyimmunarch.AA_PROPERTIES`). ``None`` (default)
        returns the raw per-position amino-acid composition. May also be a
        **list** of property names — every property is profiled and the
        results are concatenated into a single multi-property table whose
        rows are named after the properties.
    max_len
        Pad / truncate CDR3 sequences to this length. ``None`` uses the
        longest observed CDR3.
    align
        Position alignment when CDR3 lengths differ — ``'left'`` (default),
        ``'right'`` or ``'center'``.

    Returns
    -------
    :class:`pandas.DataFrame`
        Rows = amino acids (or one row per requested ``property``),
        columns = CDR3 positions.
    """
    import pandas as pd

    pim = _require_immunarch()
    rep = data
    container = getattr(data, "data", data)
    if isinstance(container, dict):
        if sample is not None:
            if sample not in container:
                raise KeyError(
                    f"sample {sample!r} not found; available: "
                    f"{list(container)}"
                )
            rep = container[sample]
        else:
            rep = next(iter(container.values()))

    if isinstance(property, (list, tuple)):
        blocks = []
        for prop in property:
            prof = pim.cdr3_aa_profile(rep, property=prop, max_len=max_len,
                                       align=align)
            # cdr3_aa_profile returns a single-row frame per property
            blocks.append(prof.iloc[[0]].rename(index={prof.index[0]: prop}))
        return pd.concat(blocks, axis=0)

    return pim.cdr3_aa_profile(rep, property=property, max_len=max_len,
                               align=align)


@register_function(
    aliases=["cohort_groups", "airr_cohort_groups", "sample_groups",
             "队列分组", "样本分组"],
    category="airr",
    description=(
        "Derive cohort sample groups from a bulk-repertoire metadata "
        "status column — relabel the status values, attach a 'group' "
        "column to the metadata and return the per-group sample lists."
    ),
    examples=[
        "g = ov.airr.cohort_groups(immdata, status_col='Status', "
        "mapping={'MS': 'MS', 'C': 'Healthy'})",
        "ms = g['groups']['MS']",
    ],
    related=["airr.repertoire_summary", "airr.differential_gene_usage"],
)
def cohort_groups(immdata, *, status_col: str = "Status",
                  mapping: Optional[dict] = None) -> dict:
    """Derive cohort sample groups from repertoire metadata.

    Takes the per-sample metadata of a bulk-repertoire cohort, relabels a
    clinical / experimental status column into analysis groups and returns
    both the annotated metadata and the per-group sample lists — the cohort
    bookkeeping that downstream group comparisons rely on.

    Parameters
    ----------
    immdata
        A :class:`pyimmunarch.ImmunData` with a ``.meta`` metadata table
        carrying a ``Sample`` column and ``status_col``.
    status_col
        Metadata column holding the raw status / condition label.
    mapping
        Optional ``{raw_status: group}`` relabelling. ``None`` (default)
        keeps the raw status values as the group labels.

    Returns
    -------
    dict
        ``'meta'`` — a copy of the metadata with an added ``group`` column;
        ``'groups'`` — ``{group: [sample_names]}``;
        ``'sample_to_group'`` — ``{sample: group}``.
    """
    meta = immdata.meta.copy()
    if mapping is not None:
        meta["group"] = meta[status_col].map(mapping)
    else:
        meta["group"] = meta[status_col]

    groups: dict = {}
    for g in meta["group"].dropna().unique():
        groups[g] = meta.loc[meta["group"] == g, "Sample"].tolist()
    sample_to_group = dict(zip(meta["Sample"], meta["group"]))
    return {"meta": meta, "groups": groups,
            "sample_to_group": sample_to_group}


@register_function(
    aliases=["repertoire_summary", "airr_repertoire_summary", "rep_summary",
             "组库概览", "组库统计"],
    category="airr",
    description=(
        "Per-sample bulk-repertoire summary table — unique clonotypes and "
        "total clone count per sample, optionally annotated with the "
        "sample group."
    ),
    examples=[
        "df = ov.airr.repertoire_summary(immdata)",
        "df = ov.airr.repertoire_summary(immdata, meta=g['meta'])",
    ],
    related=["airr.cohort_groups", "airr.repertoire_diversity"],
)
def repertoire_summary(immdata, *, meta=None):
    """Per-sample bulk-repertoire size summary.

    Builds a one-row-per-sample overview of repertoire size — the number of
    unique clonotypes and the total clone count — and, when sample
    metadata carrying a ``group`` column is supplied, attaches the group
    label.

    Parameters
    ----------
    immdata
        A :class:`pyimmunarch.ImmunData` whose ``.data`` holds one
        repertoire DataFrame per sample (with a ``Clones`` column).
    meta
        Optional metadata :class:`pandas.DataFrame` with ``Sample`` and
        ``group`` columns — typically ``cohort_groups(...)['meta']``. When
        given, a ``group`` column is joined onto the result.

    Returns
    -------
    :class:`pandas.DataFrame`
        Indexed by sample name with columns ``unique_clonotypes``,
        ``total_clones`` and (if ``meta`` is given) ``group``.
    """
    import pandas as pd

    explore = pd.DataFrame({
        "unique_clonotypes": {s: df.shape[0]
                              for s, df in immdata.data.items()},
        "total_clones": {s: int(df["Clones"].sum())
                         for s, df in immdata.data.items()},
    })
    if meta is not None:
        explore = explore.join(meta.set_index("Sample")[["group"]])
    return explore


@register_function(
    aliases=["spectratype_bulk", "bulk_spectratype", "批量谱型",
             "批量CDR3长度分布"],
    category="airr",
    description=(
        "Bulk CDR3-length spectratype — the clone-weighted distribution of "
        "CDR3 lengths, pooled per sample group, for group-comparison "
        "spectratype plots."
    ),
    examples=[
        "df = ov.airr.spectratype_bulk(immdata, groupby=g['groups'])",
        "sr = ov.airr.spectratype_bulk(immdata, samples=ms_samples)",
    ],
    related=["airr.spectratype", "airr.cohort_groups"],
)
def spectratype_bulk(immdata, *, samples=None, groupby=None,
                     col: str = "CDR3.aa", weight: str = "Clones",
                     normalize: bool = True):
    """Bulk CDR3-length spectratype, pooled per sample group.

    Pools the CDR3 sequences of several bulk repertoires and tallies the
    clone-weighted distribution of CDR3 lengths — the spectratype that
    summarises repertoire length composition for group comparisons.

    Unlike the single-cell :func:`spectratype`, this operates on the bulk
    *immunarch* data model (per-sample repertoire DataFrames).

    Parameters
    ----------
    immdata
        A :class:`pyimmunarch.ImmunData` whose ``.data`` holds one
        repertoire DataFrame per sample.
    samples
        A single list of sample names to pool into one spectratype. Mutually
        exclusive with ``groupby``.
    groupby
        A ``{group: [sample_names]}`` mapping — one spectratype column is
        produced per group. Typically ``cohort_groups(...)['groups']``.
    col
        Sequence column whose length is measured (``'CDR3.aa'`` default).
    weight
        Per-clonotype weight column (``'Clones'`` default) — each CDR3
        length is weighted by this column.
    normalize
        Normalise each group's spectratype to sum to 1.

    Returns
    -------
    :class:`pandas.DataFrame`
        Indexed by CDR3 length, one column per group. When a single
        ``samples`` list is passed the single column is named
        ``'spectratype'``.
    """
    import pandas as pd

    def _one(sample_list):
        acc: dict = {}
        for s in sample_list:
            df = immdata.data[s]
            lens = df[col].str.len()
            for length, c in df.groupby(lens)[weight].sum().items():
                acc[length] = acc.get(length, 0) + c
        sr = pd.Series(acc).sort_index()
        if normalize and sr.sum() != 0:
            sr = sr / sr.sum()
        return sr

    if (samples is None) == (groupby is None):
        raise ValueError("pass exactly one of `samples` or `groupby`.")

    if samples is not None:
        return _one(samples).to_frame(name="spectratype")

    cols = {g: _one(s) for g, s in groupby.items()}
    return pd.DataFrame(cols).sort_index()


@register_function(
    aliases=["differential_gene_usage", "diff_gene_usage", "gene_usage_diff",
             "差异基因使用", "基因使用差异"],
    category="airr",
    description=(
        "Differential V/D/J gene usage between two sample groups — the "
        "per-group mean usage and the signed case-minus-control delta from "
        "a gene-usage table."
    ),
    examples=[
        "gu = ov.airr.gene_usage_bulk(immdata, gene='hs.trbv', norm=True)",
        "df = ov.airr.differential_gene_usage(gu, groups=g['groups'], "
        "case='MS', control='Healthy')",
    ],
    related=["airr.gene_usage_bulk", "airr.gene_usage_analysis"],
)
def differential_gene_usage(gene_usage, *, groups, case: str, control: str,
                            name_col: str = "Names"):
    """Differential gene usage between two sample groups.

    From a bulk gene-usage table, averages the gene-segment usage within a
    ``case`` and a ``control`` group and computes the signed difference —
    the simplest contrast behind a differential gene-usage plot.

    Parameters
    ----------
    gene_usage
        Output of :func:`gene_usage_bulk` / :func:`pyimmunarch.geneUsage`
        (a gene name column plus one usage column per sample).
    groups
        A ``{group: [sample_names]}`` mapping — typically
        ``cohort_groups(...)['groups']``.
    case, control
        Group names; ``delta`` is computed as ``case - control``.
    name_col
        Gene-name column of ``gene_usage`` (``'Names'`` default).

    Returns
    -------
    :class:`pandas.DataFrame`
        Indexed by gene name with one mean-usage column per group and a
        signed ``delta`` column (``case`` mean minus ``control`` mean),
        sorted ascending by ``delta``.
    """
    import pandas as pd

    gu = gene_usage.set_index(name_col).fillna(0.0)
    case_cols = [s for s in groups[case] if s in gu.columns]
    control_cols = [s for s in groups[control] if s in gu.columns]
    out = pd.DataFrame({
        control: gu[control_cols].mean(axis=1),
        case: gu[case_cols].mean(axis=1),
    })
    out["delta"] = out[case] - out[control]
    return out.sort_values("delta")


@register_function(
    aliases=["cdr3_aa_properties_by_group", "group_cdr3_properties",
             "分组CDR3理化谱", "分组CDR3属性"],
    category="airr",
    description=(
        "Per-position CDR3 physicochemical property profile averaged "
        "across the samples of each cohort group — for group-comparison "
        "CDR3 property plots."
    ),
    examples=[
        "df = ov.airr.cdr3_aa_properties_by_group(immdata, "
        "groups=g['groups'], property='hydropathy')",
    ],
    related=["airr.cdr3_aa_properties", "airr.cohort_groups"],
)
def cdr3_aa_properties_by_group(immdata, *, groups, property: str = "hydropathy",
                                max_len: int = 18, align: str = "left"):
    """Per-position CDR3 property profile averaged per cohort group.

    Profiles a chosen physicochemical CDR3 property for every sample of a
    group with :func:`cdr3_aa_properties` and averages the per-position
    values across samples — one mean property profile per group, ready for
    a group-comparison line plot.

    Parameters
    ----------
    immdata
        A :class:`pyimmunarch.ImmunData`.
    groups
        A ``{group: [sample_names]}`` mapping — typically
        ``cohort_groups(...)['groups']``.
    property
        Physicochemical property to average per position — e.g.
        ``'hydropathy'``, ``'charge'``, ``'volume'`` …
    max_len
        Pad / truncate CDR3 sequences to this length before profiling.
    align
        Position alignment — ``'left'`` (default), ``'right'`` or
        ``'center'``.

    Returns
    -------
    :class:`pandas.DataFrame`
        Indexed by CDR3 position, one column per group.
    """
    import pandas as pd

    cols = {}
    for g, samples in groups.items():
        profs = [
            cdr3_aa_properties(immdata, sample=s, property=property,
                               max_len=max_len, align=align).iloc[0]
            for s in samples
        ]
        cols[g] = pd.concat(profs, axis=1).mean(axis=1)
    return pd.DataFrame(cols)


@register_function(
    aliases=["antigen_load_summary", "antigen_load", "抗原负荷汇总",
             "抗原负荷统计"],
    category="airr",
    description=(
        "Summarise an antigen-annotated bulk repertoire — total annotated "
        "clones per antigen species, the species-by-sample matrix and the "
        "per-group antigen load."
    ),
    examples=[
        "ann = ov.airr.annotate_antigen_bulk(immdata, db=vdjdb_trb, "
        "db_col='cdr3_aa')",
        "s = ov.airr.antigen_load_summary(ann, sample_cols=hc+ms, "
        "groups=g['groups'])",
    ],
    related=["airr.annotate_antigen_bulk", "airr.cohort_groups"],
)
def antigen_load_summary(annotated, *, sample_cols, groups=None,
                         by: str = "Species") -> dict:
    """Summarise antigen load from an annotated bulk repertoire.

    Aggregates an antigen-annotated repertoire table (the output of
    :func:`annotate_antigen_bulk`) into the cohort-level antigen-load
    summaries used for antigen-source bar charts and heatmaps.

    Parameters
    ----------
    annotated
        An antigen-annotated table — :func:`annotate_antigen_bulk` with
        ``annotate=True`` — carrying the ``by`` label column, a
        per-clonotype ``total_clones`` column and one clone-count column
        per repertoire sample.
    sample_cols
        The per-sample clone-count column names to aggregate over.
    groups
        Optional ``{group: [sample_names]}`` mapping — when given, the
        per-group antigen load is added.
    by
        Annotation label column to group by (``'Species'`` default; e.g.
        also ``'Antigen'`` or ``'Epitope'``).

    Returns
    -------
    dict
        ``'by_species'`` — total ``total_clones`` per ``by`` label,
        sorted ascending;
        ``'species_by_sample'`` — a ``by`` x sample clone-count matrix;
        ``'by_group'`` — total annotated clones per group (only when
        ``groups`` is given, else ``None``).
    """
    by_species = (annotated.groupby(by)["total_clones"].sum()
                  .sort_values())
    species_by_sample = annotated.groupby(by)[list(sample_cols)].sum()

    by_group = None
    if groups is not None:
        import pandas as pd
        by_group = pd.Series({
            g: annotated[[s for s in samples if s in annotated.columns]]
            .sum().sum()
            for g, samples in groups.items()
        }, name="antigen_load")

    return {"by_species": by_species,
            "species_by_sample": species_by_sample,
            "by_group": by_group}


@register_function(
    aliases=["gene_usage_analysis", "gene_usage_post", "geneUsageAnalysis",
             "基因使用分析", "基因使用降维"],
    category="airr",
    description=(
        "Multivariate post-analysis of a gene-usage table: a sample "
        "distance matrix (JS / correlation / cosine), a 2-D reduction "
        "(PCA / MDS / t-SNE), sample clustering (hclust / kmeans / dbscan) "
        "and a per-gene Kruskal-Wallis group test."
    ),
    examples=[
        "gu = ov.airr.gene_usage_bulk(immdata, gene='hs.trbv', norm=True)",
        "res = ov.airr.gene_usage_analysis(gu, distance='js', "
        "reduction='mds', cluster='hclust', k=2)",
    ],
    related=["airr.gene_usage_bulk", "airr.plotting.gene_usage_analysis_plot"],
)
def gene_usage_analysis(gene_usage, *, distance: str = "js",
                        reduction: str = "mds", cluster: str = "hclust",
                        k: int = 2, groups=None, cor: str = "pearson",
                        base: float = 2.0, eps: float = 0.5,
                        min_samples: int = 2):
    """Multivariate gene-usage post-analysis (``pyimmunarch``).

    Combines several immunarch ``geneUsageAnalysis`` building blocks into
    one call: a sample-by-sample distance/divergence matrix, a 2-D
    embedding of the samples, a clustering of that embedding and — when
    sample group labels are supplied — a per-gene Kruskal-Wallis test for
    differential gene usage between groups.

    Parameters
    ----------
    gene_usage
        Output of :func:`gene_usage_bulk` / :func:`pyimmunarch.geneUsage`
        (a ``Names`` gene column plus one column per sample).
    distance
        Sample-distance metric — ``'js'`` (Jensen-Shannon divergence,
        default), ``'cor'`` (correlation distance) or ``'cosine'``.
    reduction
        2-D embedding of the distance matrix — ``'mds'`` (default),
        ``'pca'`` or ``'tsne'``.
    cluster
        Clustering of the embedding — ``'hclust'`` (default), ``'kmeans'``,
        ``'dbscan'`` or ``None`` to skip.
    k
        Number of clusters for ``'hclust'`` / ``'kmeans'``.
    groups
        Optional ``{sample: group}`` mapping (or a sequence aligned to the
        sample columns). When given, a per-gene Kruskal-Wallis test of
        usage across groups is added to the result.
    cor
        Correlation type for ``distance='cor'``.
    base
        Logarithm base for ``distance='js'``.
    eps, min_samples
        DBSCAN parameters for ``cluster='dbscan'``.

    Returns
    -------
    dict
        ``'distance'`` — the sample distance matrix;
        ``'embedding'`` — a 2-column sample coordinate DataFrame;
        ``'clusters'`` — per-sample cluster labels (omitted if
        ``cluster=None``);
        ``'kruskal'`` — per-gene Kruskal-Wallis statistic / p-value table
        (only when ``groups`` is supplied).
    """
    import numpy as np
    import pandas as pd

    pim = _require_immunarch()
    samples = [c for c in gene_usage.columns if c != "Names"]

    dist = pim.geneUsageAnalysis(gene_usage, method=distance, cor=cor,
                                 base=base)
    result: dict = {"distance": dist}

    dmat = dist.to_numpy(dtype=float).copy()
    np.fill_diagonal(dmat, 0.0)
    if distance == "cor":
        # correlation similarity -> distance
        dmat = 1.0 - np.nan_to_num(dmat, nan=0.0)
        np.fill_diagonal(dmat, 0.0)
    else:
        dmat = np.nan_to_num(dmat, nan=np.nanmax(dmat[np.isfinite(dmat)])
                             if np.isfinite(dmat).any() else 0.0)

    red = reduction.lower()
    if red == "pca":
        coords = pim.geneUsageAnalysis(gene_usage, method="pca")
    elif red == "mds":
        from sklearn.manifold import MDS
        emb = MDS(n_components=2, dissimilarity="precomputed",
                  random_state=42, normalized_stress="auto")
        coords = pd.DataFrame(emb.fit_transform(dmat), index=samples,
                              columns=["DimI", "DimII"])
    elif red == "tsne":
        from sklearn.manifold import TSNE
        perp = max(1, min(5, len(samples) - 1))
        emb = TSNE(n_components=2, metric="precomputed", init="random",
                   perplexity=perp, random_state=42)
        coords = pd.DataFrame(emb.fit_transform(dmat), index=samples,
                              columns=["DimI", "DimII"])
    else:
        raise ValueError("reduction must be 'mds', 'pca' or 'tsne'.")
    result["embedding"] = coords

    if cluster is not None:
        X = coords.to_numpy(dtype=float)
        clu = cluster.lower()
        if clu == "hclust":
            from scipy.cluster.hierarchy import fcluster, linkage
            labels = fcluster(linkage(X, method="complete"), t=k,
                              criterion="maxclust")
        elif clu == "kmeans":
            from sklearn.cluster import KMeans
            labels = KMeans(n_clusters=k, n_init=10,
                            random_state=42).fit_predict(X)
        elif clu == "dbscan":
            from sklearn.cluster import DBSCAN
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        else:
            raise ValueError(
                "cluster must be 'hclust', 'kmeans', 'dbscan' or None."
            )
        result["clusters"] = pd.Series(labels, index=samples, name="Cluster")

    if groups is not None:
        from scipy.stats import kruskal

        if isinstance(groups, dict):
            grp = pd.Series({s: groups.get(s) for s in samples})
        else:
            grp = pd.Series(list(groups), index=samples[:len(list(groups))])
        grp = grp.reindex(samples)
        usage = gene_usage.set_index("Names")[samples].astype(float)
        rows = []
        for gene, vals in usage.iterrows():
            blocks = [vals[grp == g].dropna().values
                      for g in grp.dropna().unique()]
            blocks = [b for b in blocks if len(b) > 0]
            if len(blocks) >= 2 and any(len(b) > 1 for b in blocks):
                try:
                    stat, pval = kruskal(*blocks)
                except ValueError:
                    stat, pval = np.nan, np.nan
            else:
                stat, pval = np.nan, np.nan
            rows.append({"Gene": gene, "statistic": stat, "p_value": pval})
        result["kruskal"] = pd.DataFrame(rows).sort_values(
            "p_value", na_position="last").reset_index(drop=True)

    return result


@register_function(
    aliases=["annotate_antigen_bulk", "db_annotate", "antigen_db_bulk",
             "抗原特异性注释", "批量抗原注释"],
    category="airr",
    description=(
        "Annotate bulk repertoire clonotypes against an antigen-specificity "
        "database (VDJdb / McPAS-TCR / TBAdb-PIRD) via pyimmunarch.dbLoad "
        "and pyimmunarch.dbAnnotate."
    ),
    examples=[
        "ann = ov.airr.annotate_antigen_bulk(immdata, "
        "db_path='vdjdb.tsv', db='vdjdb')",
        "ann = ov.airr.annotate_antigen_bulk(immdata, db=mcpas_df, "
        "data_col='CDR3.aa')",
        "ann = ov.airr.annotate_antigen_bulk(immdata, db=vdjdb_trb, "
        "db_col='cdr3_aa', annotate=True)",
    ],
    related=["airr.public_repertoire", "airr.antigen_load_summary",
             "airr.track_clonotypes"],
)
def annotate_antigen_bulk(data, *, db, db_path: Optional[str] = None,
                          data_col="CDR3.aa", db_col=None,
                          species=None, chain=None, pathology=None,
                          annotate: bool = True,
                          epitope_col="antigen_epitope",
                          antigen_col="antigen_gene",
                          species_col="antigen_species",
                          **load_kwargs):
    """Antigen-specificity DB annotation of bulk repertoires (``pyimmunarch``).

    Loads (or accepts) an antigen-specificity database — VDJdb, McPAS-TCR
    or TBAdb/PIRD — and locates the matching clonotypes in every repertoire,
    counting incidence per database record, exactly as immunarch's
    ``dbAnnotate``.

    When ``annotate=True`` (default) the raw ``dbAnnotate`` table is also
    enriched: the database is collapsed to one epitope / antigen / species
    label per match key and merged back, and a per-clonotype
    ``total_clones`` column (the sum of all per-sample clone counts) is
    added. The query-clonotype count and the database hit rate are exposed
    on the returned frame's ``.attrs`` (``n_query``, ``n_matched``,
    ``hit_rate``).

    Parameters
    ----------
    data
        A :class:`pyimmunarch.ImmunData` (or list/dict of repertoires).
    db
        Either a database-format string — ``'vdjdb'``, ``'vdjdb-search'``,
        ``'mcpas'`` / ``'mcpas-tcr'`` or ``'pird'`` / ``'tbadb'`` — used
        together with ``db_path``, or an already-loaded database
        :class:`pandas.DataFrame`.
    db_path
        Path to the tabular database file (TSV / CSV, optionally gzipped).
        Required when ``db`` is a format string.
    data_col
        Repertoire column(s) used as the match key — a string or a list
        (e.g. ``'CDR3.aa'`` or ``['CDR3.aa', 'V.name']``).
    db_col
        The matching column name(s) in the database; same length as
        ``data_col``. Defaults to ``data_col``.
    species, chain, pathology
        Optional filters applied when loading the database from a file.
    annotate
        If ``True`` (default) merge the collapsed epitope / antigen /
        species labels onto the match table and add a per-clonotype
        ``total_clones`` column. Set ``False`` for the bare ``dbAnnotate``
        output (the original behaviour).
    epitope_col, antigen_col, species_col
        Column names in ``db`` carrying the epitope, antigen-gene and
        antigen-species labels — used only when ``annotate=True`` and only
        if those columns exist in the database.
    **load_kwargs
        Extra keyword arguments forwarded to :func:`pyimmunarch.dbLoad`.

    Returns
    -------
    :class:`pandas.DataFrame`
        One row per matched database clonotype — the key column(s), a
        ``Samples`` incidence count and one clone-count column per
        repertoire, sorted by descending incidence. With ``annotate=True``
        the frame also carries ``Epitope`` / ``Antigen`` / ``Species`` and
        ``total_clones`` columns, plus ``.attrs`` with ``n_query``,
        ``n_matched`` and ``hit_rate``.
    """
    import pandas as pd

    pim = _require_immunarch()
    if isinstance(db, str):
        if db_path is None:
            raise ValueError(
                "db_path is required when `db` is a database-format string."
            )
        db_table = pim.dbLoad(db_path, db, species=species, chain=chain,
                              pathology=pathology, **load_kwargs)
    else:
        db_table = db
    if db_col is None:
        db_col = data_col
    ann = pim.dbAnnotate(data, db_table, data_col, db_col)

    if not annotate:
        return ann

    key = data_col[0] if isinstance(data_col, (list, tuple)) else data_col
    dkey = db_col[0] if isinstance(db_col, (list, tuple)) else db_col

    # collapse the database to one label per match key and merge it back
    if isinstance(db_table, pd.DataFrame) and dkey in db_table.columns:
        agg = {}
        if epitope_col in db_table.columns:
            agg["Epitope"] = (epitope_col, "first")
        if antigen_col in db_table.columns:
            agg["Antigen"] = (antigen_col, "first")
        if species_col in db_table.columns:
            agg["Species"] = (species_col, "first")
        if agg:
            labels = db_table.groupby(dkey).agg(**agg).reset_index()
            ann = ann.merge(labels, left_on=key, right_on=dkey, how="left")
            if dkey != key and dkey in ann.columns:
                ann = ann.drop(columns=[dkey])

    # per-clonotype total clone count across all repertoire samples
    label_cols = {key, "Samples", "Epitope", "Antigen", "Species"}
    sample_cols = [c for c in ann.columns if c not in label_cols]
    ann["total_clones"] = ann[sample_cols].sum(axis=1)

    # query-clonotype count and hit rate
    container = getattr(data, "data", data)
    if isinstance(container, dict):
        n_query = sum(len(set(df[key].dropna()))
                      for df in container.values() if key in df.columns)
    elif isinstance(container, pd.DataFrame) and key in container.columns:
        n_query = container[key].dropna().nunique()
    else:
        n_query = 0
    ann.attrs["n_query"] = int(n_query)
    ann.attrs["n_matched"] = int(ann.shape[0])
    ann.attrs["hit_rate"] = (ann.shape[0] / n_query) if n_query else float("nan")
    return ann


@register_function(
    aliases=["overlap_analysis", "rep_overlap_analysis", "repOverlapAnalysis",
             "组库重叠分析", "重叠后分析"],
    category="airr",
    description=(
        "Post-analysis of a repertoire-overlap matrix — embed samples "
        "(MDS / t-SNE) and cluster them (hclust / kmeans) — via "
        "pyimmunarch.repOverlapAnalysis."
    ),
    examples=[
        "mat = ov.airr.repertoire_overlap_bulk(immdata, method='public')",
        "res = ov.airr.overlap_analysis(mat, method='mds+hclust', k=2)",
    ],
    related=["airr.repertoire_overlap_bulk", "airr.gene_usage_analysis"],
)
def overlap_analysis(overlap, *, method: str = "mds+hclust", k: int = 2):
    """Sample embedding / clustering from an overlap matrix (``pyimmunarch``).

    Takes a square repertoire-overlap matrix (from
    :func:`repertoire_overlap_bulk`), turns it into a 2-D sample embedding
    and optionally clusters the samples — the immunarch ``repOverlapAnalysis``
    workflow used to visualise repertoire similarity structure.

    Parameters
    ----------
    overlap
        A square sample-by-sample overlap / similarity matrix — typically
        the output of :func:`repertoire_overlap_bulk`.
    method
        Embedding step optionally chained with a clustering step:
        ``'mds'`` or ``'tsne'``, followed by ``'+hclust'`` or
        ``'+kmeans'`` (e.g. ``'mds+hclust'``, ``'tsne+kmeans'``).
    k
        Number of clusters for the clustering step.

    Returns
    -------
    dict
        ``'coords'`` — a 2-column ``DimI`` / ``DimII`` sample embedding;
        ``'clusters'`` — per-sample cluster labels (only when a clustering
        step is requested).
    """
    pim = _require_immunarch()
    return pim.repOverlapAnalysis(overlap, method=method, k=k)


@register_function(
    aliases=["public_repertoire", "public_rep_workflow", "公共组库分析"],
    category="airr",
    description=(
        "Public-repertoire workflow: build the shared-clonotype table "
        "(pubRep), optionally filter it by sample metadata (pubRepFilter), "
        "compare two public repertoires (pubRepApply) and summarise "
        "incidence (pubRepStatistics)."
    ),
    examples=[
        "pr = ov.airr.public_repertoire(immdata, col='aa+v')",
        "res = ov.airr.public_repertoire(immdata, "
        "filter_by={'Status': 'C'}, statistics=True)",
    ],
    related=["airr.public_clonotypes", "airr.overlap_analysis"],
)
def public_repertoire(data, *, col: str = "aa+v", quant: str = "count",
                      coding_only: bool = True, min_samples: int = 1,
                      max_samples=None, filter_by: Optional[dict] = None,
                      meta=None, compare_to=None, apply_fun=None,
                      statistics: bool = False):
    """Public-repertoire build / filter / compare workflow (``pyimmunarch``).

    A composable wrapper over immunarch's public-repertoire family:
    :func:`pyimmunarch.pubRep` builds the table of clonotypes shared
    across samples, :func:`pyimmunarch.pubRepFilter` subsets it by sample
    metadata, :func:`pyimmunarch.pubRepApply` compares two public
    repertoires and :func:`pyimmunarch.pubRepStatistics` summarises the
    sample-incidence distribution.

    Parameters
    ----------
    data
        A :class:`pyimmunarch.ImmunData` (or list/dict of repertoires).
    col
        Clonotype-defining column — ``'aa+v'`` (default), ``'aa'``,
        ``'nt'`` …
    quant
        Quantity stored per sample — ``'count'`` or ``'prop'``.
    coding_only
        Restrict to coding clonotypes.
    min_samples, max_samples
        Keep clonotypes present in at least ``min_samples`` (and at most
        ``max_samples``) samples.
    filter_by
        Optional ``{meta_column: value}`` mapping; when given the public
        repertoire is filtered to the matching samples with
        :func:`pyimmunarch.pubRepFilter`.
    meta
        Sample-metadata DataFrame for ``filter_by``. Defaults to the
        ``.meta`` of an :class:`~pyimmunarch.ImmunData` ``data``.
    compare_to
        Optional second public-repertoire DataFrame; when given the two
        repertoires are compared with :func:`pyimmunarch.pubRepApply`.
    apply_fun
        Optional callable forwarded to :func:`pyimmunarch.pubRepApply`
        for the per-clonotype comparison.
    statistics
        If ``True`` also return the incidence-distribution summary from
        :func:`pyimmunarch.pubRepStatistics`.

    Returns
    -------
    :class:`pandas.DataFrame` or dict
        The public-repertoire table when no comparison / statistics are
        requested; otherwise a dict with keys ``'public_repertoire'`` and,
        as requested, ``'comparison'`` and ``'statistics'``.
    """
    pim = _require_immunarch()
    pr = pim.pubRep(data, col=col, quant=quant, coding_only=coding_only,
                    min_samples=min_samples, max_samples=max_samples)

    if filter_by is not None:
        if meta is None:
            meta = getattr(data, "meta", None)
        if meta is None:
            raise ValueError(
                "filter_by needs sample metadata — pass `meta=` or an "
                "ImmunData with a `.meta` table."
            )
        pr = pim.pubRepFilter(pr, meta, by=filter_by,
                              min_samples=min_samples)

    out: dict = {"public_repertoire": pr}
    if compare_to is not None:
        out["comparison"] = pim.pubRepApply(pr, compare_to, fun=apply_fun)
    if statistics:
        out["statistics"] = pim.pubRepStatistics(pr)

    if len(out) == 1:
        return pr
    return out


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
    try:
        imm = pim.load_example_immdata(extdata_dir)
    except TypeError:
        # newer pyimmunarch: load_example_immdata() takes no arguments
        imm = pim.load_example_immdata()
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
