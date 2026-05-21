"""B-cell receptor analysis — wrappers over the Immcantation backends.

These functions thread the B-cell / immunoglobulin-specific tasks behind
``method=``-style dispatchers:

* :func:`clonal_clustering`  — B-cell clonal partitioning (:mod:`pyscoper`).
* :func:`mutation_analysis`  — observed SHM frequencies (:mod:`pyshazam`).
* :func:`shm_targeting`      — SHM targeting models (:mod:`pyshazam`).
* :func:`baseline_selection` — BASELINe selection analysis (:mod:`pyshazam`).
* :func:`infer_genotype` / :func:`find_novel_alleles` — Ig genotyping
  (:mod:`pytigger`).
* :func:`lineage_trees` / :func:`lineage_tests` — B-cell phylogenetics
  (:mod:`pydowser`).
* :func:`hill_diversity` / :func:`aa_properties` — Immcantation core
  (:mod:`pyalakazam`).

All backends import lazily, so ``import omicverse.airr`` works without the
optional ``omicverse[airr]`` extra.  Functions take plain AIRR-format
:class:`pandas.DataFrame` objects, as the Immcantation backends expect.
"""
from __future__ import annotations

from typing import Optional

from .._registry import register_function


def _require(modname: str, role: str):
    """Lazy-import a backend with an actionable error message."""
    import importlib

    try:
        return importlib.import_module(modname)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"{role} needs the '{modname}' backend. Install with: "
            f"pip install omicverse[airr]   (or pip install {modname})."
        ) from exc


# ---------------------------------------------------------------------------
# Clonal clustering — pyscoper
# ---------------------------------------------------------------------------
@register_function(
    aliases=["clonal_clustering", "bcr_clones", "B细胞克隆聚类", "克隆划分"],
    category="airr",
    description=(
        "B-cell clonal partitioning via pyscoper. method selects "
        "'identical' (identical junctions), 'hierarchical' (hierarchical "
        "clustering of junction distances) or 'spectral' (adaptive-threshold "
        "spectral clustering). Returns the AIRR DataFrame with a clone_id "
        "column."
    ),
    examples=[
        "df = ov.airr.clonal_clustering(db, method='identical')",
        "df = ov.airr.clonal_clustering(db, method='hierarchical', threshold=0.15)",
        "df = ov.airr.clonal_clustering(db, method='spectral')",
    ],
    related=["airr.mutation_analysis", "airr.lineage_trees"],
)
def clonal_clustering(db, *, method: str = "hierarchical",
                      threshold: Optional[float] = None, **kwargs):
    """B-cell clonal partitioning (``pyscoper``).

    Parameters
    ----------
    db
        An AIRR-format :class:`pandas.DataFrame` (columns ``sequence_id``,
        ``v_call``, ``j_call``, ``junction`` …).
    method
        ``'identical'`` — clones by identical junction sequence;
        ``'hierarchical'`` — hierarchical clustering of junction distances
        (needs ``threshold``); ``'spectral'`` — scoper's adaptive-threshold
        spectral clustering.
    threshold
        Distance cutoff for ``method='hierarchical'`` (required) or an
        optional override for ``method='spectral'``.
    **kwargs
        Forwarded to the underlying ``pyscoper`` function.

    Returns
    -------
    :class:`pandas.DataFrame`
        The input frame with an integer ``clone_id`` column.
    """
    scoper = _require("pyscoper", "B-cell clonal clustering")
    m = method.lower()
    if m in ("identical", "identicalclones"):
        return scoper.identicalClones(db, **kwargs)
    if m in ("hierarchical", "hierarchicalclones"):
        if threshold is None:
            raise ValueError(
                "method='hierarchical' requires a `threshold` "
                "(e.g. from ov.airr.distance_threshold)."
            )
        return scoper.hierarchicalClones(db, threshold, **kwargs)
    if m in ("spectral", "spectralclones"):
        if threshold is not None:
            kwargs.setdefault("threshold", threshold)
        return scoper.spectralClones(db, **kwargs)
    raise ValueError(
        f"method must be 'identical', 'hierarchical' or 'spectral', "
        f"got {method!r}."
    )


@register_function(
    aliases=["distance_threshold", "dist_to_nearest", "克隆距离阈值", "最近邻距离"],
    category="airr",
    description=(
        "Compute the distance-to-nearest distribution and an automatic "
        "clonal-clustering threshold via pyshazam.distToNearest + "
        "findThreshold."
    ),
    examples=[
        "thr, db = ov.airr.distance_threshold(db)",
    ],
    related=["airr.clonal_clustering"],
)
def distance_threshold(db, *, model: str = "ham",
                       threshold_method: str = "density", **kwargs):
    """Distance-to-nearest distribution + automatic clonal threshold.

    Parameters
    ----------
    db
        An AIRR-format :class:`pandas.DataFrame`.
    model
        Distance model for :func:`pyshazam.distToNearest` (``'ham'`` …).
    threshold_method
        ``'density'`` or ``'gmm'`` for :func:`pyshazam.findThreshold`.
    **kwargs
        Forwarded to :func:`pyshazam.distToNearest`.

    Returns
    -------
    tuple
        ``(threshold, db_with_dist)`` — the inferred numeric threshold (or
        ``None`` if it could not be found) and the input DataFrame with a
        ``dist_nearest`` column.
    """
    shazam = _require("pyshazam", "Distance-to-nearest")
    db_dist = shazam.distToNearest(db, model=model, **kwargs)
    dists = db_dist["dist_nearest"].dropna().values
    thr_obj = shazam.findThreshold(dists, method=threshold_method)
    threshold = getattr(thr_obj, "threshold", thr_obj)
    return threshold, db_dist


# ---------------------------------------------------------------------------
# Somatic hypermutation — pyshazam
# ---------------------------------------------------------------------------
@register_function(
    aliases=["mutation_analysis", "observed_mutations", "突变分析", "SHM突变"],
    category="airr",
    description=(
        "Quantify observed somatic-hypermutation (SHM) counts / frequencies "
        "per sequence via pyshazam.observedMutations."
    ),
    examples=[
        "df = ov.airr.mutation_analysis(db, frequency=True)",
    ],
    related=["airr.shm_targeting", "airr.baseline_selection"],
)
def mutation_analysis(db, *, frequency: bool = False, combine: bool = False,
                      region: Optional[str] = "v", **kwargs):
    """Observed SHM mutation counts / frequencies (``pyshazam``).

    Parameters
    ----------
    db
        An AIRR-format :class:`pandas.DataFrame` with ``sequence_alignment``
        and ``germline_alignment`` columns.
    frequency
        Report mutation frequencies (per-base) instead of raw counts.
    combine
        Combine R + S mutations into a single total column.
    region
        Region scheme splitting mutations by FWR / CDR sub-region —
        ``'v'`` (IMGT V-segment FWR1-3 / CDR1-2, default), ``'vdj'`` (full
        V(D)J), or ``None`` for a single whole-sequence count. Ignored when
        ``combine=True``.
    **kwargs
        Forwarded to :func:`pyshazam.observedMutations`.

    Returns
    -------
    :class:`pandas.DataFrame`
        The input frame with ``mu_count_*`` / ``mu_freq_*`` columns —
        one R (replacement) and one S (silent) column per region.
    """
    shazam = _require("pyshazam", "Mutation analysis")
    region_def = None
    if region is not None and not combine:
        schemes = {
            "v": "IMGT_V_BY_REGIONS",
            "vdj": "IMGT_VDJ_BY_REGIONS",
        }
        key = schemes.get(str(region).lower())
        if key is None:
            raise ValueError("region must be 'v', 'vdj' or None.")
        region_def = getattr(shazam, key, None)
    kwargs.setdefault("regionDefinition", region_def)
    return shazam.observedMutations(db, frequency=frequency, combine=combine,
                                    **kwargs)


@register_function(
    aliases=["shm_targeting", "targeting_model", "SHM靶向模型", "突变靶向"],
    category="airr",
    description=(
        "Build a somatic-hypermutation targeting model (substitution + "
        "mutability + 5-mer targeting) from observed sequences via "
        "pyshazam.createTargetingModel."
    ),
    examples=[
        "model = ov.airr.shm_targeting(db)",
    ],
    related=["airr.mutation_analysis", "airr.baseline_selection"],
)
def shm_targeting(db, **kwargs):
    """Build an SHM targeting model (``pyshazam.createTargetingModel``).

    Parameters
    ----------
    db
        An AIRR-format :class:`pandas.DataFrame` with germline-aligned
        sequences.
    **kwargs
        Forwarded to :func:`pyshazam.createTargetingModel`.

    Returns
    -------
    :class:`pyshazam.TargetingModel`
        A fitted 5-mer targeting model (substitution + mutability).
    """
    shazam = _require("pyshazam", "SHM targeting")
    return shazam.createTargetingModel(db, **kwargs)


@register_function(
    aliases=["baseline_selection", "calc_baseline", "BASELINe选择", "选择压力"],
    category="airr",
    description=(
        "BASELINe selection-pressure analysis: compute, group and summarise "
        "selection (Sigma) on R/S mutations via pyshazam.calcBaseline + "
        "groupBaseline + summarizeBaseline."
    ),
    examples=[
        "summary = ov.airr.baseline_selection(db, group_by='clone_id')",
    ],
    related=["airr.mutation_analysis", "airr.shm_targeting"],
)
def baseline_selection(db, *, group_by: Optional[str] = None,
                       test_statistic: str = "focused",
                       region: Optional[str] = "v",
                       collapse: bool = True,
                       clone: str = "clone_id", **kwargs):
    """BASELINe selection-pressure analysis (``pyshazam``).

    Estimates antigen-driven selection (the BASELINe selection strength
    ``Sigma``) from the ratio of replacement to silent mutations: positive
    ``Sigma`` in CDRs indicates positive (affinity-maturing) selection,
    negative ``Sigma`` in FWRs indicates purifying selection.

    When ``collapse`` is ``True`` the per-clone consensus
    ``clonal_sequence`` / ``clonal_germline`` are first built with
    :func:`pyshazam.collapseClones`; selection is then computed
    (:func:`pyshazam.calcBaseline`), grouped (:func:`pyshazam.groupBaseline`)
    and summarised (:func:`pyshazam.summarizeBaseline`).

    Parameters
    ----------
    db
        An AIRR-format :class:`pandas.DataFrame` with a clonal partitioning
        (``clone_id``) and germline-aligned sequences.
    group_by
        Column to group selection scores by (e.g. ``'clone_id'``,
        ``'sample_id'``). If ``None`` the per-region summary is returned.
    test_statistic
        BASELINe test statistic — ``'focused'`` (default) or ``'local'``.
    region
        Region scheme — ``'v'`` (IMGT V FWR/CDR, default) or ``'vdj'``.
    collapse
        If ``True`` (default) build per-clone consensus sequences first.
    clone
        Clone-id column used for the consensus collapse.
    **kwargs
        Forwarded to :func:`pyshazam.calcBaseline`.

    Returns
    -------
    :class:`pandas.DataFrame`
        Per-region (and per-group) selection table — ``baseline_sigma`` with
        confidence interval and p-value.
    """
    shazam = _require("pyshazam", "BASELINe selection")
    schemes = {"v": "IMGT_V_BY_REGIONS", "vdj": "IMGT_VDJ_BY_REGIONS"}
    region_def = getattr(shazam, schemes.get(str(region).lower(), ""), None)

    work = db
    if collapse:
        work = shazam.collapseClones(
            db, cloneColumn=clone, regionDefinition=region_def,
        )
        if group_by is not None and group_by not in work.columns:
            keep = db.groupby(clone)[group_by].first()
            work[group_by] = keep.reindex(work[clone]).values

    baseline = shazam.calcBaseline(
        work, testStatistic=test_statistic, regionDefinition=region_def,
        **kwargs,
    )
    grouped = shazam.groupBaseline(
        baseline, groupBy=[group_by] if group_by else [],
    )
    summary = shazam.summarizeBaseline(grouped)
    stats = getattr(summary, "stats", summary)
    # summarizeBaseline drops the grouping label — re-attach it so a grouped
    # selection table is interpretable. Rows are blocked per group (one block
    # of regions per group) in the order held by the grouped Baseline's .db.
    if group_by is not None and hasattr(stats, "columns") \
            and group_by not in getattr(stats, "columns", []):
        gdb = getattr(grouped, "db", None)
        if gdb is not None and group_by in getattr(gdb, "columns", []):
            labels = list(gdb[group_by])
            n_rows = len(stats)
            if labels and n_rows % len(labels) == 0:
                per = n_rows // len(labels)
                stats = stats.copy()
                stats.insert(
                    0, group_by,
                    [lab for lab in labels for _ in range(per)],
                )
    return stats


# ---------------------------------------------------------------------------
# Genotyping — pytigger
# ---------------------------------------------------------------------------
@register_function(
    aliases=["find_novel_alleles", "novel_alleles", "新等位基因", "新等位基因发现"],
    category="airr",
    description=(
        "Discover novel immunoglobulin V alleles from AIRR-seq data via "
        "pytigger.find_novel_alleles (mutation-accumulation / y-intercept "
        "regression)."
    ),
    examples=[
        "novel = ov.airr.find_novel_alleles(db, germline_db)",
    ],
    related=["airr.infer_genotype"],
)
def find_novel_alleles(db, germline_db, **kwargs):
    """Discover novel immunoglobulin V alleles (``pytigger``).

    Parameters
    ----------
    db
        An AIRR-format :class:`pandas.DataFrame`.
    germline_db
        A ``{allele_name: sequence}`` dict of IMGT-gapped germline V
        sequences.
    **kwargs
        Forwarded to :func:`pytigger.find_novel_alleles`.

    Returns
    -------
    :class:`pandas.DataFrame`
        The novel-allele evidence table.
    """
    tigger = _require("pytigger", "Novel-allele discovery")
    return tigger.find_novel_alleles(db, germline_db, **kwargs)


@register_function(
    aliases=["infer_genotype", "genotype_inference", "基因型推断", "Ig基因型"],
    category="airr",
    description=(
        "Infer an individual's immunoglobulin V genotype from AIRR-seq data "
        "via pytigger. method selects 'frequency' (frequency method) or "
        "'bayesian' (Dirichlet-multinomial)."
    ),
    examples=[
        "geno = ov.airr.infer_genotype(db, germline_db=germ, method='frequency')",
        "geno = ov.airr.infer_genotype(db, method='bayesian')",
    ],
    related=["airr.find_novel_alleles"],
)
def infer_genotype(db, *, germline_db=None, novel=None,
                   method: str = "frequency", **kwargs):
    """Infer an immunoglobulin V genotype (``pytigger``).

    Parameters
    ----------
    db
        An AIRR-format :class:`pandas.DataFrame`.
    germline_db
        ``{allele: sequence}`` germline V dict.
    novel
        Optional novel-allele table from :func:`find_novel_alleles`.
    method
        ``'frequency'`` (default) — frequency method; ``'bayesian'`` —
        Dirichlet-multinomial Bayesian inference.
    **kwargs
        Forwarded to the underlying ``pytigger`` function.

    Returns
    -------
    :class:`pandas.DataFrame`
        The inferred genotype table.
    """
    tigger = _require("pytigger", "Genotype inference")
    m = method.lower()
    if m in ("frequency", "freq"):
        return tigger.infer_genotype(db, germline_db=germline_db, novel=novel,
                                     **kwargs)
    if m in ("bayesian", "bayes"):
        return tigger.infer_genotype_bayesian(db, germline_db=germline_db,
                                              novel=novel, **kwargs)
    raise ValueError(
        f"method must be 'frequency' or 'bayesian', got {method!r}."
    )


# ---------------------------------------------------------------------------
# Germline reconstruction — pydowser
# ---------------------------------------------------------------------------
@register_function(
    aliases=["reconstruct_germlines", "create_germlines", "germline_reconstruction",
             "种系重建", "克隆种系重建"],
    category="airr",
    description=(
        "Reconstruct per-clone germline sequences for an AIRR data frame "
        "via pydowser.createGermlines — the Immcantation preprocessing step "
        "that adds the germline_alignment / germline_alignment_d_mask "
        "columns required by SHM, BASELINe and lineage-tree analyses."
    ),
    examples=[
        "import pydowser",
        "refs = pydowser.readIMGT('/path/to/imgt')",
        "db = ov.airr.reconstruct_germlines(db, references=refs)",
    ],
    related=["airr.mutation_analysis", "airr.baseline_selection",
             "airr.lineage_trees"],
)
def reconstruct_germlines(db, *, references, locus: str = "locus",
                          seq: str = "sequence_alignment",
                          v_call: str = "v_call", d_call: str = "d_call",
                          j_call: str = "j_call", clone: str = "clone_id",
                          fields: Optional[list] = None,
                          trim_lengths: bool = False, na_rm: bool = True,
                          **kwargs):
    """Reconstruct per-clone germline sequences (``pydowser.createGermlines``).

    For every clone the consensus germline is rebuilt from the IMGT
    reference set and the ``germline_alignment`` /
    ``germline_alignment_d_mask`` columns are added to the AIRR frame.
    This is the core Immcantation preprocessing step that supplies the
    germline columns consumed by somatic-hypermutation quantification
    (:func:`mutation_analysis`), BASELINe selection
    (:func:`baseline_selection`) and lineage-tree construction
    (:func:`lineage_trees`).

    Parameters
    ----------
    db
        An AIRR-format :class:`pandas.DataFrame` carrying V/D/J gene calls,
        a clonal partitioning (``clone_id``) and the germline-index
        columns (``v_germline_start`` / ``v_germline_end`` …).
    references
        The nested IMGT reference mapping
        ``{locus: {'V': {...}, 'D': {...}, 'J': {...}}}`` — typically the
        output of :func:`pydowser.readIMGT` (or
        :func:`pydowser.load_imgt_human_vdj`).
    locus
        Name of the locus *column* in ``db`` (not a locus value).
    seq
        Aligned input-sequence column.
    v_call, d_call, j_call
        V / D / J gene-call columns.
    clone
        Clone-id column the germline consensus is reconstructed per.
    fields
        Optional extra grouping columns — a separate germline is built per
        unique combination of ``clone`` and ``fields``.
    trim_lengths
        Trim reconstructed germlines to the observed sequence lengths.
    na_rm
        Drop clones whose germline reconstruction failed.
    **kwargs
        Forwarded to :func:`pydowser.createGermlines` (germline-index
        column overrides …).

    Returns
    -------
    :class:`pandas.DataFrame`
        The input frame with ``germline_alignment`` and
        ``germline_alignment_d_mask`` columns added / updated.
    """
    dowser = _require("pydowser", "Germline reconstruction")
    return dowser.createGermlines(
        db, references, locus=locus, seq=seq, v_call=v_call, d_call=d_call,
        j_call=j_call, clone=clone, fields=fields,
        trim_lengths=trim_lengths, na_rm=na_rm, **kwargs,
    )


# ---------------------------------------------------------------------------
# B-cell phylogenetics — pydowser
# ---------------------------------------------------------------------------
@register_function(
    aliases=["lineage_trees", "build_lineage_trees", "谱系树", "B细胞谱系树"],
    category="airr",
    description=(
        "Build B-cell lineage (phylogenetic) trees per clone via "
        "pydowser.formatClones + getTrees."
    ),
    examples=[
        "trees = ov.airr.lineage_trees(db, build='pratchet')",
    ],
    related=["airr.lineage_tests", "airr.clonal_clustering"],
)
def lineage_trees(db, *, build: str = "pratchet", trait: Optional[str] = None,
                  format_kwargs: Optional[dict] = None, **kwargs):
    """Build B-cell lineage trees per clone (``pydowser``).

    Parameters
    ----------
    db
        A clonal AIRR-format :class:`pandas.DataFrame` (with ``clone_id``).
    build
        Tree-building route — ``'pratchet'`` (maximum parsimony, default) or
        ``'pml'`` (pure-Python maximum likelihood).
    trait
        Optional discrete trait column propagated onto the trees.
    format_kwargs
        Extra keyword args for :func:`pydowser.formatClones`.
    **kwargs
        Forwarded to :func:`pydowser.getTrees`.

    Returns
    -------
    :class:`pandas.DataFrame`
        A per-clone table with a ``trees`` column of phylo objects.
    """
    dowser = _require("pydowser", "B-cell phylogenetics")
    clones = dowser.formatClones(db, **(format_kwargs or {}))
    return dowser.getTrees(clones, build=build, trait=trait, **kwargs)


@register_function(
    aliases=["lineage_tests", "phylo_tests", "谱系检验", "系统发育检验"],
    category="airr",
    description=(
        "Discrete-trait / measurable-evolution phylogenetic tests on B-cell "
        "lineage trees via pydowser. method selects 'switches' (trait-switch "
        "tests) or 'correlation' (root-to-tip date-randomisation test)."
    ),
    examples=[
        "res = ov.airr.lineage_tests(clones, method='switches', trait='tissue')",
        "res = ov.airr.lineage_tests(clones, method='correlation', time='time')",
    ],
    related=["airr.lineage_trees"],
)
def lineage_tests(clones, *, method: str = "correlation", **kwargs):
    """Phylogenetic tests on B-cell lineage trees (``pydowser``).

    Parameters
    ----------
    clones
        A per-clone trees table from :func:`lineage_trees`.
    method
        ``'switches'`` — trait-state-switch tests (:func:`pydowser.findSwitches`,
        needs ``trait=`` and ``permutations=``); ``'correlation'`` —
        root-to-tip divergence-vs-time test
        (:func:`pydowser.correlationTest`).
    **kwargs
        Forwarded to the underlying ``pydowser`` function.

    Returns
    -------
    dict or :class:`pandas.DataFrame`
        ``findSwitches`` returns a dict of result tables;
        ``correlationTest`` returns a DataFrame.
    """
    dowser = _require("pydowser", "B-cell phylogenetic tests")
    m = method.lower()
    if m in ("switches", "switch", "findswitches"):
        kwargs.setdefault("permutations", 100)
        if "trait" not in kwargs:
            raise ValueError("method='switches' requires a `trait` argument.")
        return dowser.findSwitches(clones, **kwargs)
    if m in ("correlation", "correlationtest", "temporal"):
        return dowser.correlationTest(clones, **kwargs)
    raise ValueError(
        f"method must be 'switches' or 'correlation', got {method!r}."
    )


# ---------------------------------------------------------------------------
# Immcantation core — pyalakazam
# ---------------------------------------------------------------------------
@register_function(
    aliases=["hill_diversity", "alpha_diversity_curve", "Hill多样性", "希尔多样性"],
    category="airr",
    description=(
        "Hill-number diversity curve (alpha diversity over the diversity "
        "order q) with bootstrap confidence intervals via "
        "pyalakazam.alphaDiversity."
    ),
    examples=[
        "curve = ov.airr.hill_diversity(db, group='sample')",
    ],
    related=["airr.repertoire_diversity", "airr.aa_properties"],
)
def hill_diversity(db, *, min_q: float = 0, max_q: float = 4,
                   step_q: float = 0.1, **kwargs):
    """Hill-number diversity curve (``pyalakazam.alphaDiversity``).

    Parameters
    ----------
    db
        An AIRR-format :class:`pandas.DataFrame` (or an
        :class:`pyalakazam.AbundanceCurve`).
    min_q, max_q, step_q
        Range / resolution of the Hill diversity order ``q``.
    **kwargs
        Forwarded to :func:`pyalakazam.alphaDiversity` (``group``, ``clone``,
        ``ci`` …).

    Returns
    -------
    :class:`pyalakazam.DiversityCurve`
    """
    alakazam = _require("pyalakazam", "Hill diversity")
    return alakazam.alphaDiversity(db, min_q=min_q, max_q=max_q,
                                   step_q=step_q, **kwargs)


@register_function(
    aliases=["aa_properties", "cdr3_aa_properties", "氨基酸性质", "CDR3理化性质"],
    category="airr",
    description=(
        "Per-sequence CDR3 amino-acid physicochemical properties (length, "
        "gravy, bulkiness, polarity, charge, aliphatic index, aromaticity) "
        "via pyalakazam.aminoAcidProperties."
    ),
    examples=[
        "df = ov.airr.aa_properties(db, seq='junction')",
    ],
    related=["airr.hill_diversity", "airr.mutation_analysis"],
)
def aa_properties(db, *, seq: str = "junction", nt: bool = True, **kwargs):
    """Per-CDR3 amino-acid physicochemical properties (``pyalakazam``).

    Parameters
    ----------
    db
        An AIRR-format :class:`pandas.DataFrame`.
    seq
        Sequence column (``'junction'`` nucleotide, or ``'junction_aa'``).
    nt
        ``True`` if ``seq`` is a nucleotide column (it is translated first).
    **kwargs
        Forwarded to :func:`pyalakazam.aminoAcidProperties`.

    Returns
    -------
    :class:`pandas.DataFrame`
        The input frame with appended ``*_aa_length``, ``*_aa_gravy`` …
        property columns.
    """
    alakazam = _require("pyalakazam", "CDR3 AA properties")
    return alakazam.aminoAcidProperties(db, seq=seq, nt=nt, **kwargs)


@register_function(
    aliases=["clonal_abundance", "estimate_abundance", "rank_abundance",
             "克隆丰度", "克隆秩丰度"],
    category="airr",
    description=(
        "Clonal rank-abundance distribution with bootstrap confidence "
        "intervals via pyalakazam.estimateAbundance — the relative size of "
        "each clone ranked from largest to smallest."
    ),
    examples=[
        "ab = ov.airr.clonal_abundance(db, group='sample_id')",
        "ab, curve = ov.airr.clonal_abundance(db, group='sample_id', "
        "as_curve_data=True)",
    ],
    related=["airr.hill_diversity",
             "airr.plotting.clonal_abundance_plot"],
)
def clonal_abundance(db, *, clone: str = "clone_id",
                     group: Optional[str] = None, copy: Optional[str] = None,
                     min_n: int = 30, max_n: Optional[int] = None,
                     uniform: bool = True, ci: float = 0.95,
                     nboot: int = 200, seed: Optional[int] = None,
                     as_curve_data: bool = False, **kwargs):
    """Clonal rank-abundance distribution (``pyalakazam.estimateAbundance``).

    Estimates, per clone, its relative abundance in the repertoire with
    bootstrap confidence intervals — the rank-abundance distribution used
    to compare clonal-expansion structure across samples / groups.

    Parameters
    ----------
    db
        An AIRR-format :class:`pandas.DataFrame` with a clonal
        partitioning (``clone_id``).
    clone
        Clone-id column.
    group
        Optional column grouping sequences into repertoires — one
        rank-abundance distribution is estimated per group.
    copy
        Optional column of per-sequence copy numbers; clone sizes are
        summed over copies instead of counting sequences.
    min_n
        Minimum repertoire size — groups smaller than this are skipped.
    max_n
        Maximum repertoire size to rarefy to. ``None`` uses the smallest
        group (when ``uniform=True``) or each group's own size.
    uniform
        Rarefy every group to a common size for comparability.
    ci
        Confidence-interval width for the bootstrap (e.g. ``0.95``).
    nboot
        Number of bootstrap realisations.
    seed
        Random seed for the bootstrap.
    as_curve_data
        If ``True`` also return a tidy abundance-curve DataFrame
        (rank, abundance, lower / upper CI per group) suitable for
        plotting.
    **kwargs
        Forwarded to :func:`pyalakazam.estimateAbundance`.

    Returns
    -------
    :class:`pyalakazam.AbundanceCurve` or tuple
        The fitted abundance curve; when ``as_curve_data=True`` a
        ``(curve, dataframe)`` tuple — the second element being the tidy
        rank-abundance table from ``curve.abundance``.
    """
    alakazam = _require("pyalakazam", "Clonal abundance")
    curve = alakazam.estimateAbundance(
        db, clone=clone, copy=copy, group=group, min_n=min_n, max_n=max_n,
        uniform=uniform, ci=ci, nboot=nboot, seed=seed, **kwargs,
    )
    if as_curve_data:
        return curve, getattr(curve, "abundance", None)
    return curve


@register_function(
    aliases=["bcr_gene_usage", "bcr_count_genes", "ig_gene_usage",
             "BCR基因使用", "免疫球蛋白基因使用"],
    category="airr",
    description=(
        "V / J / D gene & gene-family usage for BCR/Ig AIRR data via "
        "pyalakazam.countGenes — supporting plain, group-stratified, "
        "clone-weighted and copy-weighted counting."
    ),
    examples=[
        "df = ov.airr.bcr_gene_usage(db, gene='v_call')",
        "df = ov.airr.bcr_gene_usage(db, gene='v_call', mode='family', "
        "groups='sample_id')",
        "df = ov.airr.bcr_gene_usage(db, gene='j_call', clone='clone_id')",
        "wide = ov.airr.bcr_gene_usage(db, gene='v_call', mode='family', "
        "groups='c_call', pivot=True)",
    ],
    related=["airr.gene_usage_bulk", "airr.clonal_abundance"],
)
def bcr_gene_usage(db, *, gene: str = "v_call", mode: str = "gene",
                   groups=None, clone: Optional[str] = None,
                   copy: Optional[str] = None, fill: bool = False,
                   first: bool = True, remove_na: bool = True,
                   pivot: bool = False, values: str = "seq_freq", **kwargs):
    """V/J/D gene & family usage for BCR data (``pyalakazam.countGenes``).

    Tabulates immunoglobulin gene-segment usage from an AIRR frame, at
    allele / gene / family resolution, with optional group stratification
    and clone- or copy-weighted counting (so highly expanded clones do
    not dominate the usage profile).

    Parameters
    ----------
    db
        An AIRR-format :class:`pandas.DataFrame`.
    gene
        Gene-call column to tabulate — ``'v_call'`` (default), ``'j_call'``
        or ``'d_call'``.
    mode
        Counting resolution — ``'gene'`` (default), ``'allele'``,
        ``'family'`` or ``'asis'``.
    groups
        Optional column (or list of columns) to stratify the usage table
        by (e.g. ``'sample_id'``).
    clone
        Optional clone-id column — when given each clone contributes once
        (clone-weighted counting) rather than once per sequence.
    copy
        Optional per-sequence copy-number column — usage is then weighted
        by copy number.
    fill
        Fill absent gene/group combinations with zero so every group
        spans the same gene set.
    first
        When a call lists several ambiguous genes, keep only the first.
    remove_na
        Drop sequences with missing gene calls before counting.
    pivot
        If ``True`` (and ``groups`` is given), reshape the long usage table
        into a wide gene x group matrix — gene rows, one column per group
        value — with absent combinations filled with zero. The old long-form
        output is returned unchanged when ``pivot=False`` (the default).
    values
        Column to populate the wide matrix with when ``pivot=True``
        (``'seq_freq'`` by default; ``'seq_count'``, ``'clone_freq'`` …).
    **kwargs
        Forwarded to :func:`pyalakazam.countGenes`.

    Returns
    -------
    :class:`pandas.DataFrame`
        A gene-usage table with ``seq_count`` / ``seq_freq`` columns (and
        ``clone_count`` / ``copy_count`` columns when ``clone`` / ``copy``
        are supplied), one block per group. When ``pivot=True`` a wide
        gene x group matrix of ``values``.
    """
    alakazam = _require("pyalakazam", "BCR gene usage")
    usage = alakazam.countGenes(
        db, gene=gene, mode=mode, groups=groups, clone=clone, copy=copy,
        fill=fill, first=first, remove_na=remove_na, **kwargs,
    )
    if not pivot:
        return usage
    if groups is None:
        raise ValueError("pivot=True requires a `groups` argument.")
    cols = groups if isinstance(groups, str) else list(groups)
    return usage.pivot(index="gene", columns=cols, values=values).fillna(0)


# ---------------------------------------------------------------------------
# Lightweight AIRR post-processing helpers — pure pandas, no backend
# ---------------------------------------------------------------------------
#: Default isotype -> naive / class-switched map (heavy-chain constant calls).
_ISOTYPE_CLASS = {
    "IGHM": "naive",
    "IGHD": "naive",
    "IGHG": "switched",
    "IGHA": "switched",
}


@register_function(
    aliases=["isotype_class", "bcr_isotype_class", "naive_vs_switched",
             "同种型分类", "幼稚与类别转换"],
    category="airr",
    description=(
        "Map immunoglobulin heavy-chain isotype calls (c_call) to a coarse "
        "naive (IgM/IgD) vs class-switched (IgG/IgA) label. Accepts an AIRR "
        "DataFrame or a c_call Series."
    ),
    examples=[
        "cls = ov.airr.isotype_class(db)",
        "cls = ov.airr.isotype_class(db['c_call'])",
        "cls = ov.airr.isotype_class(db, mapping={'IGHE': 'switched'})",
    ],
    related=["airr.isotype_composition", "airr.bcr_gene_usage"],
)
def isotype_class(data, *, col: str = "c_call", mapping=None):
    """Map isotype calls to naive vs class-switched (pure pandas).

    Collapses the heavy-chain constant-region call into a two-level
    differentiation label: naive (unswitched IgM / IgD) versus
    class-switched (IgG / IgA, optionally IgE) B cells.

    Parameters
    ----------
    data
        Either an AIRR-format :class:`pandas.DataFrame` (the isotype column
        named by ``col`` is used) or a :class:`pandas.Series` of isotype
        calls directly.
    col
        Name of the isotype-call column when ``data`` is a DataFrame
        (``'c_call'`` by default).
    mapping
        Optional ``{isotype: class}`` override / extension. It is merged on
        top of the default ``{IGHM, IGHD -> 'naive', IGHG, IGHA ->
        'switched'}`` map, so e.g. ``{'IGHE': 'switched'}`` adds IgE.

    Returns
    -------
    :class:`pandas.Series`
        A categorical-style Series (aligned to ``data``'s index) of
        ``'naive'`` / ``'switched'`` labels; isotypes absent from the map
        become ``NaN``.
    """
    import pandas as pd

    if isinstance(data, pd.Series):
        calls = data
    elif isinstance(data, pd.DataFrame):
        if col not in data.columns:
            raise KeyError(
                f"column {col!r} not found — pass `col=` for the isotype "
                f"column or a Series directly."
            )
        calls = data[col]
    else:
        raise TypeError(
            "data must be a pandas DataFrame or Series, got "
            f"{type(data).__name__}."
        )
    full = dict(_ISOTYPE_CLASS)
    if mapping:
        full.update(mapping)
    return calls.map(full)


@register_function(
    aliases=["clone_timepoint_distribution", "clone_time_matrix",
             "克隆时间分布", "克隆时间点矩阵"],
    category="airr",
    description=(
        "Build a clone x timepoint sequence-count matrix for the large "
        "clones of a clonal AIRR frame, and report the per-timepoint share "
        "of those clones in the result's .attrs."
    ),
    examples=[
        "tp = ov.airr.clone_timepoint_distribution(clones)",
        "tp = ov.airr.clone_timepoint_distribution(clones, group='sample_id', "
        "min_size=8)",
        "tp.attrs['timepoint_share']",
    ],
    related=["airr.clonal_abundance", "airr.isotype_composition"],
)
def clone_timepoint_distribution(clones, *, clone: str = "clone_id",
                                 group: str = "sample_id",
                                 min_size: int = 8):
    """Clone x timepoint sequence-count matrix for large clones.

    Cross-tabulates, for clones whose total size is at least ``min_size``,
    how many of their sequences fall in each timepoint / group — the
    standard view of how an expanded clone is distributed across an
    immunisation or infection time course.

    Parameters
    ----------
    clones
        A clonal AIRR-format :class:`pandas.DataFrame` (with a ``clone_id``
        partitioning and a timepoint / sample column).
    clone
        Clone-id column.
    group
        Timepoint / sample column forming the matrix columns.
    min_size
        Minimum total clone size (sequences) for a clone to be kept.

    Returns
    -------
    :class:`pandas.DataFrame`
        A clone (rows) x timepoint (columns) integer count matrix, sorted
        by descending total clone size. ``.attrs['timepoint_share']`` holds
        the fraction of these large-clone sequences contributed by each
        timepoint, and ``.attrs['min_size']`` the cutoff used.
    """
    for c in (clone, group):
        if c not in clones.columns:
            raise KeyError(f"column {c!r} not found in `clones`.")
    sizes = clones[clone].value_counts()
    big = sizes[sizes >= min_size].index
    sub = clones[clones[clone].isin(big)]
    tp = (sub.groupby([clone, group]).size()
          .unstack(fill_value=0))
    # Sort by descending total clone size for a readable largest-first table.
    tp = tp.loc[tp.sum(axis=1).sort_values(ascending=False).index]
    total = tp.to_numpy().sum()
    share = (tp.sum(axis=0) / total) if total else tp.sum(axis=0) * 0.0
    tp.attrs["timepoint_share"] = share
    tp.attrs["min_size"] = min_size
    return tp


@register_function(
    aliases=["mutation_by_region", "shm_by_region", "区域突变频率",
             "IMGT区域突变"],
    category="airr",
    description=(
        "Average per-IMGT-V-region somatic-hypermutation frequency from an "
        "observedMutations table — sums replacement + silent "
        "(mu_freq_<region>_r + mu_freq_<region>_s) per region and averages "
        "over sequences, with an optional boolean subset mask."
    ),
    examples=[
        "reg = ov.airr.mutation_by_region(mut_reg)",
        "switched = mut_reg['c_call'].isin(['IGHG', 'IGHA'])",
        "reg_sw = ov.airr.mutation_by_region(mut_reg, subset=switched)",
    ],
    related=["airr.mutation_analysis", "airr.baseline_selection"],
)
def mutation_by_region(mut_reg, *, regions=None, subset=None):
    """Mean SHM frequency per IMGT V-region (pure pandas).

    Consumes the region-split output of :func:`mutation_analysis`
    (``region='v'``, ``combine=False``) and, per IMGT V sub-region, sums the
    replacement and silent mutation-frequency columns
    (``mu_freq_<region>_r`` + ``mu_freq_<region>_s``) then averages over
    sequences.

    Parameters
    ----------
    mut_reg
        An AIRR-format :class:`pandas.DataFrame` carrying region-split
        ``mu_freq_<region>_r`` / ``mu_freq_<region>_s`` columns.
    regions
        Iterable of IMGT V-region names. Defaults to
        ``('fwr1', 'cdr1', 'fwr2', 'cdr2', 'fwr3')``.
    subset
        Optional boolean mask (a :class:`pandas.Series` or array aligned to
        ``mut_reg``) restricting the average to a sub-population — e.g. a
        class-switched mask ``mut_reg['c_call'].isin(['IGHG', 'IGHA'])``.

    Returns
    -------
    :class:`pandas.Series`
        Mean total (R + S) mutation frequency indexed by region, in the
        order given by ``regions``.
    """
    import pandas as pd

    if regions is None:
        regions = ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3"]
    regions = list(regions)
    work = mut_reg if subset is None else mut_reg.loc[subset]
    out = {}
    for r in regions:
        rc, sc = f"mu_freq_{r}_r", f"mu_freq_{r}_s"
        missing = [c for c in (rc, sc) if c not in work.columns]
        if missing:
            raise KeyError(
                f"region {r!r}: column(s) {missing} not found — run "
                f"ov.airr.mutation_analysis(..., combine=False, region='v')."
            )
        out[r] = (work[rc] + work[sc]).mean()
    return pd.Series(out, index=regions, name="mu_freq")


@register_function(
    aliases=["collapse_germlines", "consensus_germline", "种系一致序列",
             "克隆种系折叠"],
    category="airr",
    description=(
        "Replace each clone's germline-alignment columns with a per-clone "
        "column-wise majority-vote consensus sequence — the germline "
        "preprocessing step required before lineage-tree construction."
    ),
    examples=[
        "tree_in = ov.airr.collapse_germlines(clones)",
        "tree_in = ov.airr.collapse_germlines(clones, "
        "cols=('germline_alignment_d_mask',))",
    ],
    related=["airr.reconstruct_germlines", "airr.lineage_trees"],
)
def collapse_germlines(db, *, clone: str = "clone_id",
                       cols=("germline_alignment",
                             "germline_alignment_d_mask")):
    """Per-clone consensus germline sequences (pure pandas).

    For every clone the named germline-alignment columns are replaced by a
    single consensus sequence built by a column-wise (per-position)
    majority vote across the clone's members. Lineage-tree builders expect
    one germline per clone, so this collapses any per-sequence variation
    introduced upstream.

    Parameters
    ----------
    db
        A clonal AIRR-format :class:`pandas.DataFrame` carrying a
        ``clone_id`` partitioning and germline-alignment columns.
    clone
        Clone-id column the consensus is computed per.
    cols
        Germline-alignment columns to collapse. Columns absent from ``db``
        are silently skipped.

    Returns
    -------
    :class:`pandas.DataFrame`
        A copy of ``db`` whose ``cols`` hold, for every member of a clone,
        that clone's column-wise consensus germline.
    """
    from collections import Counter

    import numpy as np

    if clone not in db.columns:
        raise KeyError(f"column {clone!r} not found in `db`.")

    def _consensus(seqs):
        vals = [s for s in seqs if isinstance(s, str) and s]
        if not vals:
            return seqs.iloc[0] if len(seqs) else None
        arr = np.array([list(s) for s in vals])
        return "".join(
            Counter(col).most_common(1)[0][0] for col in arr.T
        )

    out = db.copy()
    for col in cols:
        if col not in out.columns:
            continue
        out[col] = out.groupby(clone)[col].transform(_consensus)
    return out


@register_function(
    aliases=["normalize_gene_calls", "clean_gene_calls", "tidy_gene_calls",
             "基因调用规范化", "清理基因名"],
    category="airr",
    description=(
        "Clean IMGT V/J gene-call decorations on an AIRR frame — strip "
        "tool prefixes ('Homsap '), drop allele-status suffixes "
        "(' F'/' ORF'/' P'), and keep the first of comma-separated "
        "ambiguous calls."
    ),
    examples=[
        "geno_in = ov.airr.normalize_gene_calls(clones)",
        "geno_in = ov.airr.normalize_gene_calls(db, cols=('v_call',))",
    ],
    related=["airr.infer_genotype", "airr.find_novel_alleles",
             "airr.bcr_gene_usage"],
)
def normalize_gene_calls(db, *, cols=("v_call", "j_call"),
                         strip_prefix: bool = True,
                         drop_allele_status: bool = True,
                         first_only: bool = True):
    """Clean IMGT V/J gene-call decorations (pure pandas).

    Normalises the cosmetic decorations IMGT / IgBLAST attach to gene
    calls so they match the bare allele names expected by genotyping
    (:func:`infer_genotype`, :func:`find_novel_alleles`): leading tool
    prefixes (``'Homsap '``), trailing allele-status flags (``' F'``,
    ``' ORF'``, ``' P'``) and comma-separated ambiguous lists.

    Mirrors the private ``_gene_root`` helper of :mod:`omicverse.airr._tcr`
    for consistency across the AIRR module.

    Parameters
    ----------
    db
        An AIRR-format :class:`pandas.DataFrame`.
    cols
        Gene-call columns to clean. Columns absent from ``db`` are skipped.
    strip_prefix
        Remove a leading species/tool prefix (e.g. ``'Homsap '``).
    drop_allele_status
        Remove a trailing IMGT allele-status flag (`` F`` / `` ORF`` /
        `` P``).
    first_only
        Keep only the first call of a comma-separated ambiguous list.

    Returns
    -------
    :class:`pandas.DataFrame`
        A copy of ``db`` with the requested gene-call columns cleaned.
    """
    out = db.copy()
    for col in cols:
        if col not in out.columns:
            continue
        s = out[col].astype("string")
        if strip_prefix:
            s = s.str.replace(r"^\S+sap ", "", regex=True)
        if drop_allele_status:
            s = s.str.replace(r" (F|ORF|P)$", "", regex=True)
        if first_only:
            s = s.str.split(",").str[0]
        out[col] = s.str.strip()
    return out


@register_function(
    aliases=["isotype_composition", "isotype_fraction", "同种型组成",
             "同种型比例"],
    category="airr",
    description=(
        "Isotype-fraction-by-timepoint matrix from a clonal AIRR frame — "
        "cross-tabulates c_call against a timepoint/sample column and "
        "normalises per timepoint, exposing the class-switched (IgG+IgA) "
        "fraction in .attrs."
    ),
    examples=[
        "iso_frac = ov.airr.isotype_composition(clones)",
        "iso_frac = ov.airr.isotype_composition(clones, group='sample_id')",
        "iso_frac.attrs['switched_fraction']",
    ],
    related=["airr.isotype_class", "airr.bcr_gene_usage"],
)
def isotype_composition(clones, *, group: str = "sample_id",
                        isotype: str = "c_call", normalize: bool = True):
    """Isotype composition by timepoint (pure pandas).

    Cross-tabulates the immunoglobulin isotype call against a timepoint /
    sample column to give the per-timepoint isotype profile — the standard
    view of class-switch dynamics over an immune response.

    Parameters
    ----------
    clones
        An AIRR-format :class:`pandas.DataFrame`.
    group
        Timepoint / sample column forming the matrix rows.
    isotype
        Isotype-call column forming the matrix columns (``'c_call'``).
    normalize
        If ``True`` (default) divide each row by its total so cells are
        per-timepoint fractions; if ``False`` keep raw sequence counts.

    Returns
    -------
    :class:`pandas.DataFrame`
        A timepoint (rows) x isotype (columns) matrix of fractions (or
        counts when ``normalize=False``). ``.attrs['switched_fraction']``
        holds the per-timepoint class-switched (IgG + IgA) fraction.
    """
    for c in (group, isotype):
        if c not in clones.columns:
            raise KeyError(f"column {c!r} not found in `clones`.")
    iso_tp = (clones.groupby([group, isotype]).size()
              .unstack(fill_value=0))
    if normalize:
        iso_tp = iso_tp.div(iso_tp.sum(axis=1), axis=0)
    switched_cols = [c for c in ("IGHG", "IGHA") if c in iso_tp.columns]
    if switched_cols:
        iso_tp.attrs["switched_fraction"] = iso_tp[switched_cols].sum(axis=1)
    return iso_tp
