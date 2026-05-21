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
