r"""Sample-level genetic QC from aligned reads at a SNP panel.

Paired tumour / normal (or any multi-sample) sequencing batches are checked
for four things that a genotype panel can settle without variant calling:

``snp_allele_counts``
    per-sample ref / alt / other read counts at every panel SNP (pysam pileup,
    one observation per read and site), with the target interval of each SNP.
``bam_read_summary``
    per-sample read-length, mapping and modified-base-tag summaries with
    robust batch z-scores (library / assay outliers).
``sample_genotype_concordance``
    all-vs-all genotype discordance on germline-informative sites (alleles
    seen clonally in at least two samples), ranked comparators per sample
    and mutual-best ("same individual") pairs.
``resolve_sample_relationships``
    checks each submitted tumour / baseline pair against the concordance and
    proposes the molecularly supported comparator of each member.
``estimate_cross_sample_contamination``
    joint binomial mixture of every other sample's allele profile into each
    sample's minority-allele reads at its homozygous sites: per-source
    contaminating fraction, likelihood-ratio test (over-dispersion
    corrected, Bonferroni over all ordered pairs) and source-private marker
    support.
``tumor_normal_allelic_imbalance``
    per-region folded-allele-fraction comparison of a tumour against its
    matched normal at normal-heterozygous sites (LOH / imbalance, tumour
    content implied by the deepest region).
``tumor_in_normal_shift``
    tumour-in-normal from germline SNPs: the normal's heterozygous allele
    fraction shifted toward the tumour's retained allele inside the tumour's
    LOH regions (direction-balanced, no somatic markers needed).
``tumor_normal_batch_qc``
    runs the steps above and returns one row per supported finding
    (identity mismatch, cross-sample contamination, tumour-in-normal,
    low tumour content, regional allelic imbalance, library outlier).

Nothing here is specific to a platform, panel or batch: every cut-off is a
statistical significance level or a genotype-state definition and can be
changed by the caller.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._registry import register_function

__all__ = [
    "snp_allele_counts",
    "bam_read_summary",
    "sample_genotype_concordance",
    "resolve_sample_relationships",
    "estimate_cross_sample_contamination",
    "tumor_normal_allelic_imbalance",
    "tumor_in_normal_shift",
    "tumor_normal_batch_qc",
]

PathLike = Union[str, Path]


# --------------------------------------------------------------------------
# input helpers
# --------------------------------------------------------------------------
def _as_bam_mapping(bams) -> Dict[str, str]:
    """``{sample: path}`` from a mapping, a directory, or a list of paths."""
    if isinstance(bams, Mapping):
        return {str(k): str(v) for k, v in bams.items()}
    if isinstance(bams, (str, Path)) and Path(bams).is_dir():
        paths = sorted(Path(bams).glob("*.bam")) + sorted(Path(bams).glob("*.cram"))
        return {p.name.rsplit(".", 1)[0]: str(p) for p in paths}
    if isinstance(bams, (str, Path)):
        bams = [bams]
    out = {}
    for p in bams:
        p = Path(p)
        out[p.name.rsplit(".", 1)[0]] = str(p)
    return out


def _read_targets(targets) -> pd.DataFrame:
    if targets is None:
        return pd.DataFrame(columns=["chrom", "start", "end", "region_id"])
    if isinstance(targets, pd.DataFrame):
        df = targets.copy()
        df.columns = [str(c) for c in df.columns]
        ren = {df.columns[0]: "chrom", df.columns[1]: "start", df.columns[2]: "end"}
        if len(df.columns) > 3:
            ren[df.columns[3]] = "region_id"
        df = df.rename(columns=ren)
    else:
        rows = []
        with open(targets) as fh:
            for i, raw in enumerate(fh, 1):
                if not raw.strip() or raw.startswith(("#", "track", "browser")):
                    continue
                f = raw.rstrip("\n").split("\t")
                rows.append((f[0], int(f[1]), int(f[2]),
                             f[3] if len(f) > 3 and f[3] else f"{f[0]}:{f[1]}-{f[2]}"))
        df = pd.DataFrame(rows, columns=["chrom", "start", "end", "region_id"])
    if "region_id" not in df:
        df["region_id"] = df["chrom"].astype(str) + ":" + df["start"].astype(str) + "-" + df["end"].astype(str)
    return df[["chrom", "start", "end", "region_id"]]


def _read_sites(sites) -> pd.DataFrame:
    """Bi-allelic SNPs as ``chrom, pos (1-based), ref, alt``."""
    if isinstance(sites, pd.DataFrame):
        df = sites.copy()
        df.columns = [str(c).lower().lstrip("#") for c in df.columns]
        df = df.rename(columns={"position": "pos", "chr": "chrom"})
        return df[["chrom", "pos", "ref", "alt"]].assign(pos=lambda d: d["pos"].astype(int))
    import pysam
    rows = []
    with pysam.VariantFile(str(sites)) as vcf:
        for rec in vcf.fetch() if vcf.index is not None else vcf:
            if not rec.alts or len(rec.ref) != 1 or len(rec.alts[0]) != 1:
                continue
            rows.append((rec.chrom, int(rec.pos), rec.ref.upper(), rec.alts[0].upper()))
    return pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt"])


def _contig_resolver(bam_contigs: Iterable[str]):
    contigs = set(bam_contigs)

    def resolve(c: str) -> Optional[str]:
        if c in contigs:
            return c
        alt = c[3:] if c.startswith("chr") else "chr" + c
        return alt if alt in contigs else None
    return resolve


def _count_one_bam(sample, path, recs, min_mapq, min_baseq, drop_duplicates):
    """Allele counts of one alignment file at ``recs`` (chrom, pos, ref, alt, region, in_target)."""
    import pysam

    rows, checks = [], []
    try:
        bam = pysam.AlignmentFile(path, "rb")
    except Exception as exc:  # unreadable input is itself a QC finding
        return rows, [f"{sample}: cannot open {path} ({exc})"]
    with bam:
        if not bam.has_index():
            return rows, [f"{sample}: alignment file has no index"]
        resolve = _contig_resolver(bam.references)
        missing = set()
        for chrom, pos, ref, alt, region_id, in_target in recs:
            contig = resolve(chrom)
            ref_n = alt_n = oth_n = 0
            if contig is None:
                missing.add(chrom)
            else:
                seen = set()
                for col in bam.pileup(contig, pos - 1, pos, truncate=True,
                                      stepper="nofilter", min_base_quality=0,
                                      ignore_orphans=False, ignore_overlaps=False,
                                      max_depth=1_000_000, compute_baq=False):
                    for pr in col.pileups:
                        if pr.is_del or pr.is_refskip or pr.query_position is None:
                            continue
                        r = pr.alignment
                        if (r.is_unmapped or r.is_secondary or r.is_supplementary
                                or r.is_qcfail or (drop_duplicates and r.is_duplicate)
                                or r.mapping_quality < min_mapq):
                            continue
                        if r.query_name in seen:
                            continue
                        seen.add(r.query_name)
                        q = r.query_qualities
                        if q is not None and len(q) and q[pr.query_position] < min_baseq:
                            continue
                        b = r.query_sequence[pr.query_position].upper()
                        if b == ref:
                            ref_n += 1
                        elif b == alt:
                            alt_n += 1
                        else:
                            oth_n += 1
            depth = ref_n + alt_n
            rows.append((sample, chrom, pos, ref, alt, region_id, ref_n, alt_n,
                         oth_n, depth, alt_n / depth if depth else np.nan, in_target))
        if missing:
            checks.append(f"{sample}: panel contigs absent from the alignment header: "
                          + ",".join(sorted(missing)))
    return rows, checks


# --------------------------------------------------------------------------
# allele counting
# --------------------------------------------------------------------------
@register_function(
    aliases=["snp_allele_counts", "bam_snp_pileup", "allele_counts_at_sites",
             "SNP等位计数", "位点等位基因计数"],
    category="genetics",
    description=(
        "Count ref / alt / other bases per sample at every bi-allelic SNP of a "
        "panel (VCF or table) from BAM/CRAM alignments with pysam; one "
        "observation per read and site, MAPQ / base-quality filters, region "
        "of each SNP from a BED of target intervals, chr-prefix tolerant. "
        "Returns a long table (sample x site) used by the genotype "
        "concordance, contamination and allelic-imbalance functions."
    ),
    examples=[
        "counts = ov.genetics.snp_allele_counts('data/bams', 'sites.vcf.gz', targets='targets.bed')",
    ],
    related=["ov.genetics.sample_genotype_concordance",
             "ov.genetics.estimate_cross_sample_contamination",
             "ov.genetics.tumor_normal_batch_qc"],
)
def snp_allele_counts(
    bams,
    sites,
    *,
    targets=None,
    min_mapq: int = 20,
    min_baseq: int = 10,
    drop_duplicates: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Per-sample allele counts at a SNP panel.

    Parameters
    ----------
    bams
        ``{sample: path}``, a list of BAM/CRAM paths (sample = file stem), or a
        directory holding ``*.bam`` (indexed).
    sites
        VCF / VCF.gz path or a DataFrame with ``chrom, pos (1-based), ref, alt``.
        Multi-allelic records and indels are skipped.
    targets
        Optional BED path / DataFrame (chrom, start, end, name). Each SNP gets
        the name of the interval containing it as ``region_id`` (``NA`` if
        none); SNPs outside every interval are kept and flagged.
    min_mapq, min_baseq
        Read mapping-quality and base-quality floors.
    n_jobs
        Alignment files counted in parallel processes.

    Returns
    -------
    DataFrame with ``sample, chrom, pos, ref, alt, region_id, ref_count,
    alt_count, other_count, depth (ref+alt), alt_fraction, in_target``.
    ``df.attrs['input_checks']`` lists contig / index problems found.
    """
    bam_map = _as_bam_mapping(bams)
    site_df = _read_sites(sites)
    tg = _read_targets(targets)
    region = np.array(["NA"] * len(site_df), dtype=object)
    in_target = np.zeros(len(site_df), dtype=bool)
    for _, t in tg.iterrows():
        m = (site_df["chrom"].values == t["chrom"]) & (site_df["pos"].values - 1 >= t["start"]) \
            & (site_df["pos"].values - 1 < t["end"])
        region[m & ~in_target] = t["region_id"]
        in_target |= m
    site_df["region_id"] = region
    site_df["in_target"] = in_target

    checks: List[str] = []
    rows = []
    recs = list(site_df[["chrom", "pos", "ref", "alt", "region_id", "in_target"]].itertuples(index=False, name=None))
    jobs = [(sample, path, recs, min_mapq, min_baseq, drop_duplicates) for sample, path in bam_map.items()]
    n_jobs = max(1, min(int(n_jobs or 1), len(jobs)))
    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(_count_one_bam, *zip(*jobs)))
    else:
        results = [_count_one_bam(*j) for j in jobs]
    for r_rows, r_checks in results:
        rows.extend(r_rows)
        checks.extend(r_checks)
    out = pd.DataFrame(rows, columns=["sample", "chrom", "pos", "ref", "alt", "region_id",
                                      "ref_count", "alt_count", "other_count", "depth",
                                      "alt_fraction", "in_target"])
    n_out = int((~site_df["in_target"]).sum()) if len(tg) else 0
    if n_out:
        checks.append(f"{n_out} panel SNPs fall outside every target interval")
    out.attrs["input_checks"] = checks
    return out


def _n50(lengths: np.ndarray) -> float:
    if lengths.size == 0:
        return np.nan
    s = np.sort(lengths)[::-1]
    c = np.cumsum(s)
    return float(s[np.searchsorted(c, c[-1] / 2)])


def _robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(0.0, index=x.index)
    return (x - med) / mad


@register_function(
    aliases=["bam_read_summary", "read_length_qc", "library_qc",
             "测序文库质控", "读长统计"],
    category="genetics",
    description=(
        "Per-sample read-level summary of BAM/CRAM alignments over target "
        "intervals: primary read count, read-length quantiles / N50, mean MAPQ, "
        "duplicate / secondary / supplementary fractions, modified-base (MM/ML) "
        "tag fraction and calls per kb, with robust batch z-scores and an "
        "outlier list (library fragmentation, depth or modified-base outliers)."
    ),
    examples=["rs = ov.genetics.bam_read_summary('data/bams', targets='targets.bed')"],
    related=["ov.genetics.snp_allele_counts", "ov.genetics.tumor_normal_batch_qc"],
)
def bam_read_summary(bams, *, targets=None, max_reads: int = 0, outlier_z: float = 3.5) -> pd.DataFrame:
    """Read-length / mapping / modified-base summary per sample.

    ``outlier_z`` is the robust z (median / MAD across the batch) beyond which a
    metric is listed in the ``outliers`` column.
    """
    import pysam

    bam_map = _as_bam_mapping(bams)
    tg = _read_targets(targets)
    rows = []
    for sample, path in bam_map.items():
        lengths, mapq = [], []
        n_dup = n_sec = n_sup = n_tag = 0
        mod_calls = 0
        seen = set()
        with pysam.AlignmentFile(path, "rb") as bam:
            resolve = _contig_resolver(bam.references)
            regions = ([(resolve(t.chrom), t.start, t.end) for t in tg.itertuples()]
                       if len(tg) else [(None, None, None)])
            for contig, start, end in regions:
                it = (bam.fetch(contig, start, end) if contig is not None
                      else bam.fetch(until_eof=True))
                for r in it:
                    if r.is_unmapped:
                        continue
                    if r.is_secondary:
                        n_sec += 1
                        continue
                    if r.is_supplementary:
                        n_sup += 1
                        continue
                    key = (r.query_name, r.flag)
                    if key in seen:
                        continue
                    seen.add(key)
                    if r.is_duplicate:
                        n_dup += 1
                    lengths.append(r.query_length or r.infer_read_length() or 0)
                    mapq.append(r.mapping_quality)
                    if r.has_tag("ML"):
                        n_tag += 1
                        try:
                            mod_calls += len(r.get_tag("ML"))
                        except Exception:
                            pass
                    elif r.has_tag("MM"):
                        n_tag += 1
                    if max_reads and len(lengths) >= max_reads:
                        break
        L = np.asarray(lengths, dtype=float)
        n = max(len(L), 1)
        rows.append({
            "sample": sample, "primary_reads": len(L),
            "median_read_length": float(np.median(L)) if len(L) else np.nan,
            "read_length_q25": float(np.percentile(L, 25)) if len(L) else np.nan,
            "read_length_q75": float(np.percentile(L, 75)) if len(L) else np.nan,
            "read_n50": _n50(L), "total_bases": float(L.sum()),
            "mean_mapq": float(np.mean(mapq)) if mapq else np.nan,
            "duplicate_fraction": n_dup / n, "secondary_per_primary": n_sec / n,
            "supplementary_per_primary": n_sup / n,
            "modbase_tagged_fraction": n_tag / n,
            "modbase_calls_per_kb": mod_calls * 1000 / L.sum() if L.sum() else np.nan,
        })
    df = pd.DataFrame(rows).set_index("sample")
    metrics = ["primary_reads", "median_read_length", "read_n50", "mean_mapq",
               "duplicate_fraction", "modbase_calls_per_kb"]
    flags = {s: [] for s in df.index}
    if len(df) >= 4:
        for m in metrics:
            x = df[m].astype(float)
            # lengths and counts are compared on the log scale
            if m in ("primary_reads", "median_read_length", "read_n50", "modbase_calls_per_kb"):
                x = np.log10(x.clip(lower=1e-9))
            z = _robust_z(x)
            df[f"z_{m}"] = z
            for s, v in z.items():
                if abs(v) >= outlier_z:
                    flags[s].append(f"{m}({'high' if v > 0 else 'low'}, z={v:.1f})")
    df["outliers"] = [";".join(flags[s]) for s in df.index]
    return df.reset_index()


# --------------------------------------------------------------------------
# genotype concordance / identity
# --------------------------------------------------------------------------
def _wide(counts: pd.DataFrame, col: str) -> pd.DataFrame:
    return counts.pivot_table(index=["chrom", "pos"], columns="sample", values=col, aggfunc="first")


def _genotype_states(af: np.ndarray, hom: float, het_lo: float) -> np.ndarray:
    """0 hom-ref, 1 het, 2 hom-alt, -1 ambiguous / missing."""
    g = np.full(af.shape, -1, dtype=int)
    g[af <= hom] = 0
    g[af >= 1 - hom] = 2
    g[(af >= het_lo) & (af <= 1 - het_lo)] = 1
    g[~np.isfinite(af)] = -1
    return g


@register_function(
    aliases=["sample_genotype_concordance", "sample_identity", "genotype_concordance",
             "sample_swap_check", "样本身份核验", "基因型一致性"],
    category="genetics",
    description=(
        "All-vs-all genotype discordance between samples from SNP-panel allele "
        "counts, restricted to germline-informative sites (alleles seen at clonal "
        "fraction in >=2 samples, so sample-private / somatic markers do not mask "
        "identity). Returns pairwise discordance, opposite-homozygote counts, "
        "comparator rank per sample and mutual-best (same-individual) flags."
    ),
    examples=["conc = ov.genetics.sample_genotype_concordance(counts)"],
    related=["ov.genetics.resolve_sample_relationships", "ov.genetics.snp_allele_counts"],
)
def sample_genotype_concordance(
    counts: pd.DataFrame,
    *,
    min_depth: int = 8,
    hom_af: float = 0.15,
    het_af: float = 0.30,
    informative: str = "shared",
) -> pd.DataFrame:
    """Pairwise genotype discordance.

    Parameters
    ----------
    counts
        Output of :func:`snp_allele_counts`.
    min_depth
        Minimum ref+alt depth in both samples for a site to be compared.
    hom_af, het_af
        Genotype states: AF <= ``hom_af`` hom-ref, >= 1-``hom_af`` hom-alt,
        ``het_af`` <= AF <= 1-``het_af`` het; anything else is ambiguous and
        not compared (contamination / imbalance does not create calls).
    informative
        ``'shared'`` keeps sites whose alt and ref alleles are each present at
        clonal fraction (AF >= ``het_af`` / <= 1-``het_af``) in at least two
        samples; ``'all'`` keeps every site.

    Returns
    -------
    Long DataFrame ``sample, comparator, n_compared, discordance,
    opposite_hom, het_hom, hom_het, loh_inconsistent, rank, mutual_best,
    gap_to_next`` (one row per ordered pair). ``het_hom`` counts sites where
    the sample is heterozygous and the comparator homozygous, ``hom_het`` the
    reverse. ``loh_inconsistent = opposite_hom + min(het_hom, hom_het)`` counts
    the mismatches that loss of heterozygosity in *one* of the two libraries
    cannot explain (LOH turns a heterozygote into a homozygote in the tumour
    only, never the other way round and never into the opposite homozygote):
    ~0 for two libraries of one person, tumour / normal included, and a
    sizeable fraction of the sites for two different people.
    """
    af = _wide(counts, "alt_fraction")
    dp = _wide(counts, "depth").reindex_like(af)
    samples = list(af.columns)
    A = af.values.astype(float)
    D = dp.values.astype(float)
    A[~(D >= min_depth)] = np.nan
    if informative == "shared":
        alt_present = (A >= het_af).sum(1)
        ref_present = (A <= 1 - het_af).sum(1)
        keep = (alt_present >= 2) & (ref_present >= 2)
        A = A[keep]
    G = _genotype_states(A, hom_af, het_af)
    rows = []
    for i, j in itertools.permutations(range(len(samples)), 2):
        v = (G[:, i] >= 0) & (G[:, j] >= 0)
        n = int(v.sum())
        disc = float((G[v, i] != G[v, j]).mean()) if n else np.nan
        opp = int((np.abs(G[v, i] - G[v, j]) == 2).sum())
        het_hom = int(((G[v, i] == 1) & (G[v, j] != 1)).sum())
        hom_het = int(((G[v, i] != 1) & (G[v, j] == 1)).sum())
        rows.append((samples[i], samples[j], n, disc, opp, het_hom, hom_het, opp + min(het_hom, hom_het)))
    df = pd.DataFrame(rows, columns=["sample", "comparator", "n_compared", "discordance", "opposite_hom",
                                     "het_hom", "hom_het", "loh_inconsistent"])
    df["rank"] = df.groupby("sample")["discordance"].rank(method="min").astype(int)
    best = df[df["rank"] == 1].groupby("sample")["comparator"].apply(set).to_dict()
    df["mutual_best"] = [(c in best.get(s, set())) and (s in best.get(c, set()))
                         for s, c in zip(df["sample"], df["comparator"])]
    second = df.sort_values("discordance").groupby("sample")["discordance"].apply(
        lambda x: x.iloc[1] if len(x) > 1 else np.nan)
    df["gap_to_next"] = np.where(df["rank"] == 1, df["sample"].map(second) - df["discordance"], np.nan)
    df.attrs["n_informative_sites"] = int(A.shape[0])
    # opposite homozygotes expected between two *unrelated* people at these
    # sites (Hardy-Weinberg, allele frequency from the batch's genotype calls)
    dose = np.where(G >= 0, G / 2.0, np.nan)
    with np.errstate(invalid="ignore"):
        p = np.nanmean(dose, axis=1) if dose.size else np.array([])
    p = p[np.isfinite(p)]
    df.attrs["unrelated_opposite_hom_rate"] = float(np.mean(2 * p ** 2 * (1 - p) ** 2)) if p.size else np.nan
    # and LOH-inconsistent mismatches (opposite homozygote, or het/hom in the
    # direction LOH of the other library does not produce): one direction of
    # het/hom is 2pq(p^2+q^2) per site; min(two directions) <= that, so this
    # is an upper bound used only when the batch has no unrelated pairs to
    # measure the rate on
    df.attrs["unrelated_loh_inconsistent_rate"] = (
        float(np.mean(2 * p ** 2 * (1 - p) ** 2 + 2 * p * (1 - p) * (p ** 2 + (1 - p) ** 2)))
        if p.size else np.nan)
    out = df.sort_values(["sample", "rank"]).reset_index(drop=True)
    out.attrs.update(df.attrs)
    return out


def _same_individual_groups(conc: pd.DataFrame) -> Dict[str, set]:
    """``{sample: set of same-individual samples}`` from the concordance.

    1. Mutual-best pairs are joined (two samples that are each other's
       closest genotype match).
    2. Groups are then merged while two groups are each other's nearest
       group (single linkage on discordance) **and** their closest pair's
       count of LOH-inconsistent mismatches (``loh_inconsistent``: opposite
       homozygotes plus the smaller of the two het/hom directions) is better
       explained by the same-individual error rate than by the rate between
       different people.

    Step 2 is what keeps several libraries of one person together: two
    libraries of the same normal are each other's best match (discordance
    ~0), so the tumour -- whose best match is one of them, or a tie between
    them -- is the best match of neither and step 1 alone leaves it on its
    own (a false identity mismatch). The statistic is robust to tumour LOH
    (a heterozygous site becomes homozygous in the tumour only, so all of
    its mismatches fall in one het/hom direction) and stops an unrelated
    sample from being absorbed by its nearest group: two different people
    disagree in both het/hom directions. Opposite homozygotes alone are not
    enough for that -- at panels of common, mostly-reference sites and with
    contaminated libraries (whose homozygous sites fall out as ambiguous)
    two different people can share every homozygous call at ~120 sites.

    The same-individual rate is estimated from the mutual-best pairs; the
    rate between different people from the pairs left in different groups
    and from Hardy-Weinberg at the panel's allele frequencies
    (``conc.attrs['unrelated_loh_inconsistent_rate']``), taking the smaller
    of the two that exceed the same-individual rate (fewer merges when they
    disagree).
    """
    samples = sorted(set(conc["sample"]) | set(conc["comparator"]))
    groups = {s: {s} for s in samples}
    for s, c in conc.loc[conc["mutual_best"], ["sample", "comparator"]].itertuples(index=False):
        g = groups[s] | groups[c]
        for x in g:
            groups[x] = g

    if "loh_inconsistent" in conc.columns:
        stat, attr = "loh_inconsistent", "unrelated_loh_inconsistent_rate"
    else:                                   # concordance from an older release
        stat, attr = "opposite_hom", "unrelated_opposite_hom_rate"
    d = {(a, b): (float(x), int(o), int(n)) for a, b, x, o, n in
         conc[["sample", "comparator", "discordance", stat, "n_compared"]].itertuples(index=False)
         if np.isfinite(x) and n > 0}
    if not d:
        return groups
    mb = conc[conc["mutual_best"] & (conc["n_compared"] > 0)]
    e_same = (float(mb[stat].sum()) + 1.0) / (float(mb["n_compared"].sum()) + 2.0)
    candidates = [float(conc.attrs.get(attr, np.nan))]
    apart = [(o, n) for (a, b), (_, o, n) in d.items() if b not in groups[a]]
    if apart:
        candidates.append((sum(o for o, _ in apart) + 1.0) / (sum(n for _, n in apart) + 2.0))
    candidates = [e for e in candidates if np.isfinite(e) and e > e_same]
    if not candidates:
        return groups
    e_unrel = min(candidates)

    def same_person(o: int, n: int) -> bool:
        # binomial log-likelihood ratio, same individual vs unrelated
        return o * np.log(e_same / e_unrel) + (n - o) * np.log((1 - e_same) / (1 - e_unrel)) > 0

    def link(G, H):
        best = None
        for a in G:
            for b in H:
                v = d.get((a, b)) or d.get((b, a))
                if v is not None and (best is None or v[0] < best[0]):
                    best = v
        return best

    while True:
        parts = list({id(g): g for g in groups.values()}.values())
        if len(parts) < 2:
            break
        near = {}
        for i, G in enumerate(parts):
            cands = [(link(G, H), j) for j, H in enumerate(parts) if j != i]
            cands = [(v, j) for v, j in cands if v is not None]
            if cands:
                dmin = min(v[0] for v, _ in cands)
                near[i] = {j for v, j in cands if v[0] <= dmin + 1e-12}
        merged = False
        for i, js in near.items():
            for j in js:
                if j <= i or i not in near.get(j, set()):
                    continue
                _, o, n = link(parts[i], parts[j])
                if same_person(o, n):
                    g = parts[i] | parts[j]
                    for x in g:
                        groups[x] = g
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    return groups


@register_function(
    aliases=["resolve_sample_relationships", "manifest_check", "tumor_normal_pairing_check",
             "样本配对核验", "manifest核对"],
    category="genetics",
    description=(
        "Check each submitted tumour / baseline (or any paired) record against "
        "the genotype concordance: rank of the submitted partner, whether the "
        "pair is mutual-best (same individual), and the molecularly supported "
        "comparator of each member (manifest / label / swap repair)."
    ),
    examples=["rel = ov.genetics.resolve_sample_relationships(conc, manifest, "
              "case_col='case_id', a_col='tumor_alias', b_col='submitted_baseline_alias')"],
    related=["ov.genetics.sample_genotype_concordance"],
)
def resolve_sample_relationships(
    concordance: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    case_col: str = "case_id",
    a_col: str = "tumor_alias",
    b_col: str = "submitted_baseline_alias",
) -> pd.DataFrame:
    """One row per submitted record.

    Columns: ``case_id, a, b, submitted_rank_for_a, submitted_discordance,
    best_for_a, best_for_a_discordance, best_for_b, status`` where status is
    ``SUPPORTED`` (the submitted pair is in one same-individual group) or
    ``MISMATCH``.

    Same-individual groups (see ``_same_individual_groups``) join mutual-best
    pairs and then reciprocal-nearest groups whose LOH-inconsistent mismatch
    count fits the same-individual error rate, so a batch may hold several
    libraries of one person (re-runs, replicate normals, a second tumour
    library) without breaking the tumour / normal record.
    """
    conc = concordance.set_index(["sample", "comparator"])
    groups = _same_individual_groups(concordance)
    rows = []
    for rec in manifest.itertuples(index=False):
        a, b = getattr(rec, a_col), getattr(rec, b_col)
        ca = concordance[concordance["sample"] == a].sort_values("discordance")
        cb = concordance[concordance["sample"] == b].sort_values("discordance")
        sub = conc.loc[(a, b)] if (a, b) in conc.index else None
        supported = b in groups.get(a, set())
        rows.append({
            "case_id": getattr(rec, case_col), "a": a, "b": b,
            "submitted_rank_for_a": int(sub["rank"]) if sub is not None else np.nan,
            "submitted_discordance": float(sub["discordance"]) if sub is not None else np.nan,
            "submitted_n_compared": int(sub["n_compared"]) if sub is not None else 0,
            "best_for_a": ca["comparator"].iloc[0] if len(ca) else None,
            "best_for_a_discordance": float(ca["discordance"].iloc[0]) if len(ca) else np.nan,
            "best_for_b": cb["comparator"].iloc[0] if len(cb) else None,
            "best_for_b_discordance": float(cb["discordance"].iloc[0]) if len(cb) else np.nan,
            "status": "SUPPORTED" if supported else "MISMATCH",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# contamination
# --------------------------------------------------------------------------
def _fit_mixture(k: np.ndarray, n: np.ndarray, X: np.ndarray, eps_max: float = 0.1):
    """MLE of p_i = eps + X_i . a (a >= 0) for binomial minority counts."""
    from scipy.optimize import minimize

    m = X.shape[1]

    def nll(th):
        p = np.clip(th[0] + X @ th[1:], 1e-7, 1 - 1e-7)
        return -(k * np.log(p) + (n - k) * np.log1p(-p)).sum()

    def grad(th):
        p = np.clip(th[0] + X @ th[1:], 1e-7, 1 - 1e-7)
        w = -(k / p - (n - k) / (1 - p))
        return np.r_[w.sum(), X.T @ w]

    x0 = np.r_[max(k.sum() / max(n.sum(), 1), 1e-4), np.full(m, 1e-3)]
    res = minimize(nll, x0, jac=grad, method="L-BFGS-B",
                   bounds=[(1e-6, eps_max)] + [(0.0, 1.0)] * m)
    return res.fun, res.x


@register_function(
    aliases=["estimate_cross_sample_contamination", "contamination_source", "carryover_detection",
             "cross_sample_contamination", "tumor_in_normal", "跨样本污染", "污染来源识别"],
    category="genetics",
    description=(
        "Estimate low-fraction cross-sample contamination / carryover and its "
        "source from SNP-panel allele counts: at each sample's homozygous-like "
        "sites the minority-allele reads are modelled as background error plus "
        "a non-negative mixture of every other batch sample's allele "
        "frequencies (joint binomial MLE), giving per-source fraction, "
        "over-dispersion-corrected likelihood-ratio test with Bonferroni "
        "correction, informative / hit site counts and source-private marker "
        "support. Detects 1-20% admixture that genotype thresholds hide."
    ),
    examples=["cont = ov.genetics.estimate_cross_sample_contamination(counts)",
              "cont[cont.called]"],
    related=["ov.genetics.snp_allele_counts", "ov.genetics.tumor_normal_batch_qc"],
)
def estimate_cross_sample_contamination(
    counts: pd.DataFrame,
    *,
    min_depth: int = 8,
    source_min_depth: int = 4,
    max_minor_af: float = 0.25,
    marker_min_source_af: float = 0.20,
    private_max_other_af: float = 0.05,
    alpha: float = 0.01,
    min_fraction: float = 0.005,
    overdispersion: bool = True,
    private_outlier_p: float = 1e-3,
    genotype_reference: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
    groups: Optional[Mapping[str, Iterable[str]]] = None,
    min_informative_sites: int = 5,
    min_hit_sites: int = 3,
) -> pd.DataFrame:
    """Joint mixture estimate of contamination fractions and sources.

    For affected sample *A* take its sites with depth >= ``min_depth`` where
    the minor allele is below ``max_minor_af`` (A is homozygous apart from
    error / admixture). With ``k_i`` minority reads of ``n_i`` and ``s_{S,i}``
    the frequency of that allele in candidate source *S*, fit
    ``k_i ~ Bin(n_i, eps + sum_S a_S s_{S,i})`` with ``a_S >= 0``. Each
    source's evidence is the likelihood ratio of the full fit against the fit
    without it, divided by the Pearson dispersion of the full fit
    (``overdispersion=True``), referred to a 50:50 chi2(0)/chi2(1) boundary
    mixture, Bonferroni-corrected over all ordered pairs. Sites whose
    minority allele no other sample carries and whose minority count is above
    the background rate (binomial p < ``private_outlier_p``) are alleles
    private to the affected sample (sub-clonal somatic variants, artefacts);
    they hold no source information and are left out of the fit
    (``affected_private_minor_sites``).

    ``genotype_reference`` maps an affected sample to other samples of the
    same individual (e.g. a tumour to its matched normal). Its sites are then
    restricted to loci homozygous in the *individual* (affected and every
    reference sample), and the minor allele is oriented by the individual's
    genotype. This removes loci where the affected sample is homozygous only
    through somatic loss of heterozygosity, whose residual minority reads are
    the sample's own normal admixture and otherwise dilute / confound a
    foreign source.

    ``groups`` maps each sample to its same-individual set (e.g. the
    ``groups`` of :func:`tumor_normal_batch_qc`). Samples of one individual
    are genetically collinear as sources, so the evidence for a *foreign*
    individual is the likelihood ratio of dropping all its samples at once
    (``lr`` / ``p_adj`` of that individual), and the source is attributed to
    the member whose own single-source fit has the highest likelihood
    (``best_in_individual``; the others are reported as alternates and are
    never ``called``). ``fraction`` of a foreign source is then its
    single-source mixture fraction (``joint_fraction`` keeps the joint fit).
    A call additionally needs ``min_informative_sites`` loci at which the
    source carries the affected sample's minor allele and ``min_hit_sites``
    of them with minority reads, so single noisy loci never make a call.

    Returns
    -------
    One row per ordered pair: ``affected, source, fraction, lr, p_value,
    p_adj, called, n_sites, informative_sites`` (source allele frequency
    >= ``marker_min_source_af``), ``hit_sites`` (informative with >= 1
    minority read), ``mean_minor_af_at_hits``, ``private_markers`` (source
    allele absent, <= ``private_max_other_af``, from every other sample),
    ``private_hits``, ``private_reads``, ``dominant_region``, ``dispersion``,
    ``background_error``, ``source_individual``, ``best_in_individual``,
    ``single_source_loglik``, ``joint_fraction``. ``called`` = ``p_adj <
    alpha``, ``fraction >= min_fraction``, the support minima and (with
    ``groups``) ``best_in_individual``.
    """
    from scipy.stats import binom, chi2

    af = _wide(counts, "alt_fraction")
    dp = _wide(counts, "depth").reindex_like(af)
    ac = _wide(counts, "alt_count").reindex_like(af)
    reg = counts.drop_duplicates(["chrom", "pos"]).set_index(["chrom", "pos"])["region_id"].reindex(af.index)
    samples = list(af.columns)
    AF, DP, AC = af.values.astype(float), dp.values.astype(float), ac.values.astype(float)
    n_tests = len(samples) * (len(samples) - 1)
    rows = []
    gref = {}
    for k_, v_ in (genotype_reference or {}).items():
        v_ = [v_] if isinstance(v_, str) else list(v_)
        gref[k_] = [samples.index(x) for x in v_ if x in samples and x != k_]
    grp = {s_: frozenset(groups.get(s_, [s_])) | {s_} for s_ in samples} if groups else \
        {s_: frozenset([s_]) for s_ in samples}
    for ia, A in enumerate(samples):
        a = AF[:, ia]
        refs = gref.get(A, [])
        # minor allele oriented by the individual's genotype (affected + references)
        g_af = np.nanmean(AF[:, [ia] + refs], axis=1) if refs else a
        minor_alt = g_af < 0.5
        sel = np.ones(len(a), dtype=bool)
        for j in [ia] + refs:
            mj = np.where(minor_alt, AF[:, j], 1 - AF[:, j])
            sel &= (DP[:, j] >= min_depth) & np.isfinite(mj) & (mj < max_minor_af)
        minor_af = np.where(minor_alt, a, 1 - a)
        if sel.sum() < 5:
            continue
        k = np.where(minor_alt, AC[:, ia], DP[:, ia] - AC[:, ia])[sel]
        n = DP[sel, ia]
        others = [j for j in range(len(samples)) if j != ia]
        S = np.where(minor_alt[:, None], AF[:, others], 1 - AF[:, others])
        S[~(DP[:, others] >= source_min_depth)] = np.nan
        X = np.nan_to_num(S[sel], nan=0.0)
        # Sites whose minority allele no batch sample carries cannot inform a
        # source; those that are far above the background rate are alleles
        # private to the affected sample (sub-clonal somatic variants, site
        # artefacts) and would inflate the error term and the dispersion.
        no_src = X.max(axis=1) < private_max_other_af
        bg_rate = max(k[no_src].sum() / max(n[no_src].sum(), 1), 1e-4) if no_src.any() else 1e-3
        outl = np.zeros(len(k), dtype=bool)
        for _ in range(5):
            outl = no_src & (binom.sf(k - 1, n, bg_rate) < private_outlier_p)
            keep_bg = no_src & ~outl
            new_rate = max(k[keep_bg].sum() / max(n[keep_bg].sum(), 1), 1e-4) if keep_bg.any() else bg_rate
            if abs(new_rate - bg_rate) < 1e-6:
                break
            bg_rate = new_rate
        keep = ~outl
        n_private_minor = int(outl.sum())
        sel_idx = np.flatnonzero(sel)[keep]
        sel = np.zeros_like(sel)
        sel[sel_idx] = True
        k, n, X = k[keep], n[keep], X[keep]
        L_full, th = _fit_mixture(k, n, X)
        p_hat = np.clip(th[0] + X @ th[1:], 1e-7, 1 - 1e-7)
        pear = ((k - n * p_hat) ** 2 / (n * p_hat * (1 - p_hat))).sum()
        dof = max(len(k) - X.shape[1] - 1, 1)
        phi = max(pear / dof, 1.0) if overdispersion else 1.0
        S_all = S[sel]
        Sfull = np.where(minor_alt[:, None], AF, 1 - AF)
        Sfull[~(DP >= source_min_depth)] = np.nan
        minor_sel = minor_af[sel]
        reg_sel = reg.values[sel]
        # single-source fits (background + one source) for attribution
        single = {}
        for jj in range(len(others)):
            Ls, ths = _fit_mixture(k, n, X[:, [jj]])
            single[jj] = (-Ls, float(ths[1]))
        group_lr: Dict[frozenset, float] = {}
        best_member: Dict[frozenset, int] = {}
        own = grp[A]
        for jj, js in enumerate(others):
            G = grp[samples[js]]
            if samples[js] in own or G in group_lr:
                continue
            cols = [c for c, jx in enumerate(others) if samples[jx] in G and samples[jx] not in own]
            if th[1 + np.array(cols)].sum() > 0:
                L0, _ = _fit_mixture(k, n, np.delete(X, cols, axis=1))
                group_lr[G] = max(2 * (L0 - L_full), 0.0)
            else:
                group_lr[G] = 0.0
            best_member[G] = max(cols, key=lambda c: single[c][0])
        for jj, js in enumerate(others):
            src = samples[js]
            foreign = src not in own and groups is not None
            if foreign:
                G = grp[src]
                lr = group_lr[G]
                is_best = best_member[G] == jj
                frac = single[jj][1]
            else:
                if th[1 + jj] > 0:
                    L0, _ = _fit_mixture(k, n, np.delete(X, jj, axis=1))
                    lr = max(2 * (L0 - L_full), 0.0)
                else:
                    lr = 0.0
                is_best = True
                frac = float(th[1 + jj])
            stat = lr / phi
            pval = 0.5 * chi2.sf(stat, 1) if stat > 0 else 1.0
            padj = min(pval * n_tests, 1.0)
            s = S_all[:, jj]
            inf = np.isfinite(s) & (s >= marker_min_source_af)
            hits = inf & (k > 0)
            rest = [x for x in range(len(samples)) if x not in (ia, js)]
            other_max = np.nanmax(Sfull[sel][:, rest], axis=1) if rest else np.zeros(sel.sum())
            priv = inf & (np.nan_to_num(other_max, nan=1.0) <= private_max_other_af)
            if hits.any():
                rc = pd.Series(reg_sel[hits]).value_counts()
                dom = rc.index[0] if (len(rc) == 1 or rc.iloc[0] >= 0.7 * rc.sum()) else "MULTIPLE"
            else:
                dom = "NA"
            rows.append({
                "affected": A, "source": src, "fraction": frac, "lr": lr / phi if overdispersion else lr,
                "lr_raw": lr, "p_value": pval, "p_adj": padj,
                "called": bool(padj < alpha and frac >= min_fraction and is_best
                               and inf.sum() >= min_informative_sites and hits.sum() >= min_hit_sites),
                "n_sites": int(sel.sum()), "informative_sites": int(inf.sum()),
                "hit_sites": int(hits.sum()),
                "mean_minor_af_at_hits": float(minor_sel[hits].mean()) if hits.any() else np.nan,
                "private_markers": int(priv.sum()), "private_hits": int((priv & (k > 0)).sum()),
                "private_reads": int(k[priv].sum()), "dominant_region": dom,
                "dispersion": phi, "background_error": float(th[0]),
                "affected_private_minor_sites": n_private_minor,
                "source_individual": ",".join(sorted(grp[src])),
                "best_in_individual": bool(is_best),
                "single_source_loglik": single[jj][0],
                "joint_fraction": float(th[1 + jj]),
                "genotype_reference": ",".join(samples[r] for r in refs) or "NA",
            })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["called", "lr"], ascending=[False, False]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# tumour / normal allelic imbalance
# --------------------------------------------------------------------------
@register_function(
    aliases=["tumor_normal_allelic_imbalance", "tumor_purity_from_loh", "loh_regions",
             "肿瘤纯度", "等位失衡", "LOH检测"],
    category="genetics",
    description=(
        "Compare a tumour with its matched normal at normal-heterozygous panel "
        "SNPs: per-region median folded allele fraction (min(AF,1-AF)), "
        "imbalance depth 1 - folded_T/folded_N with a one-sided Mann-Whitney "
        "test, LOH / imbalanced regions, and the tumour content implied by the "
        "deepest region (copy-neutral LOH model); flags tumours with no clonal "
        "imbalance anywhere (low tumour content / baseline-like specimen)."
    ),
    examples=["reg, summ = ov.genetics.tumor_normal_allelic_imbalance(counts, 'T1', 'N1')"],
    related=["ov.genetics.tumor_normal_batch_qc"],
)
def tumor_normal_allelic_imbalance(
    counts: pd.DataFrame,
    tumor: str,
    normal: str,
    *,
    min_depth: int = 8,
    het_band: float = 0.20,
    min_region_loci: int = 3,
    alpha: float = 0.01,
    clonal_imbalance: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Per-region allelic imbalance of ``tumor`` against ``normal``.

    Normal-heterozygous sites: ``|AF_N - 0.5| <= het_band``, both depths
    >= ``min_depth``. For region r, ``imbalance = 1 - median(fT)/median(fN)``
    (fraction of the normal minor-allele signal lost in the tumour; equals the
    tumour content for a clonal copy-neutral LOH). A region is imbalanced
    when the one-sided Mann-Whitney p < ``alpha``.

    Returns ``(regions, summary)``; summary keys: ``het_loci``,
    ``median_folded_tumor``, ``median_folded_normal``, ``median_folded_shift``,
    ``max_imbalance``, ``max_imbalance_region``, ``imbalanced_regions``,
    ``loh_regions`` (imbalance >= ``clonal_imbalance``),
    ``tumor_content_estimate`` (= max significant imbalance) and
    ``low_tumor_content`` (no region reaches ``clonal_imbalance``).
    """
    from scipy.stats import mannwhitneyu

    af = _wide(counts, "alt_fraction")
    dp = _wide(counts, "depth").reindex_like(af)
    reg = counts.drop_duplicates(["chrom", "pos"]).set_index(["chrom", "pos"])["region_id"].reindex(af.index)
    ok = (dp[tumor] >= min_depth) & (dp[normal] >= min_depth) & ((af[normal] - 0.5).abs() <= het_band)
    fT = np.minimum(af[tumor], 1 - af[tumor])[ok]
    fN = np.minimum(af[normal], 1 - af[normal])[ok]
    d = pd.DataFrame({"fT": fT.values, "fN": fN.values, "region_id": reg[ok].values})
    rows = []
    for r, g in d.groupby("region_id"):
        if len(g) < min_region_loci:
            continue
        mT, mN = float(g["fT"].median()), float(g["fN"].median())
        try:
            p = float(mannwhitneyu(g["fT"], g["fN"], alternative="less").pvalue)
        except ValueError:
            p = 1.0
        rows.append({"region_id": r, "het_loci": len(g), "median_folded_tumor": mT,
                     "median_folded_normal": mN,
                     "imbalance": 1 - mT / mN if mN > 0 else np.nan, "p_value": p,
                     "imbalanced": p < alpha})
    regions = pd.DataFrame(rows)
    summ: Dict[str, object] = {"tumor": tumor, "normal": normal, "het_loci": int(len(d))}
    if len(d):
        summ.update(median_folded_tumor=float(d["fT"].median()),
                    median_folded_normal=float(d["fN"].median()),
                    median_folded_shift=float((d["fT"] - d["fN"]).abs().median()))
    sig = regions[regions["imbalanced"]] if len(regions) else regions
    if len(sig):
        top = sig.sort_values("imbalance", ascending=False).iloc[0]
        summ.update(max_imbalance=float(top["imbalance"]), max_imbalance_region=top["region_id"],
                    tumor_content_estimate=float(np.clip(top["imbalance"], 0, 1)))
    else:
        summ.update(max_imbalance=0.0, max_imbalance_region="NA", tumor_content_estimate=np.nan)
    summ["imbalanced_regions"] = list(sig["region_id"]) if len(sig) else []
    loh = sig[sig["imbalance"] >= clonal_imbalance] if len(sig) else sig
    summ["loh_regions"] = list(loh["region_id"]) if len(loh) else []
    summ["loh_loci"] = int(loh["het_loci"].sum()) if len(loh) else 0
    summ["low_tumor_content"] = bool(len(d) >= 2 * min_region_loci and len(loh) == 0)
    return regions, summ


@register_function(
    aliases=["tumor_in_normal_shift", "tumor_in_normal_loh", "tin_estimate", "normal_contamination_by_tumor",
             "肿瘤污染正常", "正常样本肿瘤污染"],
    category="genetics",
    description=(
        "Tumour-in-normal (tumour-derived reads in the matched normal / "
        "baseline) from germline SNPs alone: inside the tumour's LOH regions the "
        "normal's heterozygous allele fraction is shifted toward the allele the "
        "tumour retained. Direction-balanced signed shift (ALT-retained and "
        "REF-retained loci averaged, cancelling reference / mapping bias), "
        "z-score, p-value and the implied tumour-derived fraction. Needs no "
        "tumour-private / somatic markers, so it works on any SNP panel."
    ),
    examples=["tin = ov.genetics.tumor_in_normal_shift(counts, 'T1', 'N1')"],
    related=["ov.genetics.tumor_normal_allelic_imbalance", "ov.genetics.tumor_normal_batch_qc"],
)
def tumor_in_normal_shift(
    counts: pd.DataFrame,
    tumor: str,
    normal: str,
    *,
    loh_regions: Optional[Sequence[str]] = None,
    min_depth: int = 15,
    het_band: float = 0.25,
    retained_min_shift: float = 0.25,
    min_loci_per_direction: int = 5,
    imbalance_kwargs: Optional[dict] = None,
) -> Dict[str, object]:
    """Signed allelic shift of ``normal`` toward ``tumor``'s retained alleles.

    Loci: ``normal`` heterozygous (``|AF_N - 0.5| <= het_band``), both depths
    >= ``min_depth``, inside ``loh_regions`` (default: the LOH regions of
    :func:`tumor_normal_allelic_imbalance` of the pair), and the tumour
    clearly retaining one allele (``|AF_T - 0.5| >= retained_min_shift``).
    With ``d = (AF_N - 0.5) * sign(AF_T - 0.5)`` the statistic is the mean of
    ``d`` over ALT-retained loci and over REF-retained loci, averaged (a
    reference bias in the normal moves the two means in opposite directions
    and cancels); ``z`` uses the two-sample standard error. A tumour-derived
    fraction ``f`` in the normal gives ``AF_N = 0.5 + f (AF_T - 0.5)``, so
    ``fraction = shift / mean |AF_T - 0.5|``. Restricting to LOH regions keeps
    isolated artefactual loci from driving the test; a tumour without LOH
    regions gives ``tested = False`` (no power, not evidence of absence).

    Returns a dict ``tumor, normal, tested, loh_regions, loci_alt_retained,
    loci_ref_retained, shift, se, z, p_value, fraction, dominant_region``.
    """
    from scipy.stats import norm

    if loh_regions is None:
        _, summ = tumor_normal_allelic_imbalance(counts, tumor, normal, **(imbalance_kwargs or {}))
        loh_regions = summ["loh_regions"]
    loh_regions = list(loh_regions)
    out: Dict[str, object] = {"tumor": tumor, "normal": normal, "tested": False, "loh_regions": loh_regions,
                              "loci_alt_retained": 0, "loci_ref_retained": 0, "shift": np.nan, "se": np.nan,
                              "z": np.nan, "p_value": 1.0, "fraction": np.nan, "dominant_region": "NA"}
    if not loh_regions:
        return out
    af = _wide(counts, "alt_fraction")
    dp = _wide(counts, "depth").reindex_like(af)
    if tumor not in af.columns or normal not in af.columns:
        return out
    reg = counts.drop_duplicates(["chrom", "pos"]).set_index(["chrom", "pos"])["region_id"].reindex(af.index).values
    aN, aT = af[normal].values.astype(float), af[tumor].values.astype(float)
    ok = (np.isin(reg, loh_regions) & (dp[normal].values >= min_depth) & (dp[tumor].values >= min_depth)
          & np.isfinite(aN) & np.isfinite(aT) & (np.abs(aN - 0.5) <= het_band))
    hi = ok & (aT - 0.5 >= retained_min_shift)
    lo = ok & (0.5 - aT >= retained_min_shift)
    out.update(loci_alt_retained=int(hi.sum()), loci_ref_retained=int(lo.sum()))
    if hi.sum() < min_loci_per_direction or lo.sum() < min_loci_per_direction:
        return out
    d_hi, d_lo = aN[hi] - 0.5, 0.5 - aN[lo]
    shift = float((d_hi.mean() + d_lo.mean()) / 2)
    se = float(np.sqrt(d_hi.var(ddof=1) / len(d_hi) + d_lo.var(ddof=1) / len(d_lo)) / 2)
    amp = float((np.abs(aT[hi] - 0.5).mean() + np.abs(aT[lo] - 0.5).mean()) / 2)
    z = shift / se if se > 0 else 0.0
    rc = pd.Series(reg[hi | lo]).value_counts()
    out.update(tested=True, shift=shift, se=se, z=float(z), p_value=float(norm.sf(z)),
               fraction=shift / amp if amp > 0 else np.nan,
               dominant_region=rc.index[0] if (len(rc) == 1 or rc.iloc[0] >= 0.7 * rc.sum()) else "MULTIPLE")
    return out


# --------------------------------------------------------------------------
# batch review
# --------------------------------------------------------------------------
def _replicate_libraries(counts: pd.DataFrame, groups: Mapping[str, set], listed: Iterable[str],
                         min_depth: int = 8) -> Dict[str, str]:
    """``{library: manifest sample}`` for libraries absent from the manifest
    whose same-individual group contains manifest samples: the manifest
    sample with the closest allele-fraction profile (a tumour re-run matches
    its tumour, not the normal, at LOH and somatic sites)."""
    listed = set(listed)
    extra = [s for s in groups if s not in listed and (groups[s] & listed)]
    if not extra:
        return {}
    af = _wide(counts, "alt_fraction")
    dp = _wide(counts, "depth").reindex_like(af)
    ok = dp >= min_depth
    out = {}
    for lib in extra:
        best, best_d = None, np.inf
        for m in sorted(groups[lib] & listed):
            v = (ok[lib] & ok[m]).values
            if v.sum() == 0:
                continue
            dist = float(np.nanmean(np.abs(af[lib].values[v] - af[m].values[v])))
            if dist < best_d:
                best, best_d = m, dist
        if best is not None:
            out[lib] = best
    return out


def _pool_libraries(counts: pd.DataFrame, replicate_of: Mapping[str, str]) -> pd.DataFrame:
    """Sum the allele counts of re-run libraries into their manifest sample."""
    c = counts.assign(sample=counts["sample"].replace(dict(replicate_of)))
    num = [x for x in ("ref_count", "alt_count", "other_count", "depth") if x in c.columns]
    key = [x for x in c.columns if x not in num + ["alt_fraction"]]
    out = c.groupby(key, as_index=False, sort=False, dropna=False)[num].sum()
    rd = out["ref_count"] + out["alt_count"] if {"ref_count", "alt_count"} <= set(out.columns) \
        else out["depth"]
    out["alt_fraction"] = out["alt_count"] / rd.where(rd > 0)
    out = out[[x for x in counts.columns if x in out.columns]]
    out.attrs.update(counts.attrs)
    return out


@register_function(
    aliases=["tumor_normal_batch_qc", "paired_tumor_normal_qc", "sequencing_batch_qc",
             "sample_qc_review", "肿瘤正常配对质控", "测序批次质控"],
    category="genetics",
    description=(
        "Batch QC review of paired tumour / normal (baseline) sequencing from "
        "SNP-panel allele counts: genotype identity (manifest / label "
        "mismatch with repaired comparators), cross-sample contamination with "
        "source attribution, tumour-in-normal, low tumour content (no clonal "
        "allelic imbalance vs the matched normal), regional allelic imbalance "
        "and library outliers. Normal admixture inside a tumour is reported as "
        "tumour content, not as contamination. Returns one row per finding, "
        "assigned to the submitted case of the affected sample."
    ),
    examples=[
        "counts = ov.genetics.snp_allele_counts('data/bams', 'sites.vcf.gz', targets='targets.bed')",
        "res = ov.genetics.tumor_normal_batch_qc(counts, manifest, read_summary=rs)",
        "res['findings']",
    ],
    related=["ov.genetics.snp_allele_counts", "ov.genetics.estimate_cross_sample_contamination",
             "ov.genetics.sample_genotype_concordance", "ov.genetics.tumor_normal_allelic_imbalance"],
)
def tumor_normal_batch_qc(
    counts: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    case_col: str = "case_id",
    tumor_col: str = "tumor_alias",
    normal_col: str = "submitted_baseline_alias",
    read_summary: Optional[pd.DataFrame] = None,
    contamination_kwargs: Optional[dict] = None,
    imbalance_kwargs: Optional[dict] = None,
    report_regional_imbalance: bool = False,
    max_findings_per_case: int = 4,
    tin_alpha: float = 1e-4,
    tin_min_fraction: float = 0.02,
    tin_kwargs: Optional[dict] = None,
    pool_replicates: bool = True,
) -> Dict[str, object]:
    """Run identity, contamination and tumour-content checks for a batch.

    Several libraries of one individual may be present (re-sequenced
    normals, a second tumour library). Identity groups are formed on every
    library; a library that is not in the manifest but belongs to a manifest
    individual is treated as a re-run of the manifest sample whose allele
    fractions it matches best and, with ``pool_replicates=True``, its counts
    are pooled into that sample for the contamination / imbalance steps (a
    re-run would otherwise be fitted as a "source" of its own sample). Put a
    library in the manifest if it is a distinct sample. ``pooled_libraries``
    in the result lists what was pooled.

    Finding types (``finding_type``):

    ``identity_mismatch``
        submitted pair is not the same individual; ``affected`` = both
        submitted samples, ``comparison`` = the best-matching sample of each.
    ``cross_sample_contamination``
        a discrete minority allele signal attributable to a sample of a
        *different* individual; ``comparison`` = source (+ samples of the
        source's individual, genetically indistinguishable as sources).
    ``tumor_in_normal``
        a normal / baseline contains reads of its own individual's tumour:
        tumour-private alleles at low fraction (mixture model) and / or the
        normal's heterozygous allele fraction shifted toward the tumour's
        retained alleles inside the tumour's LOH regions
        (:func:`tumor_in_normal_shift`, one-sided p < ``tin_alpha`` /
        number of tested pairs and fraction >= ``tin_min_fraction``).
    ``low_tumor_content``
        a tumour shows no clonal allelic imbalance anywhere against its
        (repaired) matched normal: broad attenuation toward the baseline.
    ``regional_allelic_imbalance``
        LOH / imbalance regions of a tumour (only if
        ``report_regional_imbalance``; usually tumour biology, ancillary).
    ``library_outlier``
        read-level batch outlier from ``read_summary`` (ancillary).

    Contamination of a tumour is tested at loci homozygous in its individual
    (tumour and repaired matched normal), and a foreign individual's samples
    are tested jointly with the source attributed to the best-fitting member
    (see :func:`estimate_cross_sample_contamination` ``genotype_reference`` /
    ``groups``).

    Normal-in-tumour admixture (a tumour's minority alleles explained by its
    own normal) is tumour content and is never reported as contamination.
    Each finding goes to the case in which its affected sample was submitted.

    Returns a dict with ``findings`` (DataFrame), ``concordance`` (every
    library), ``relationships``, ``contamination``, ``imbalance`` (per tumour
    summary), ``groups`` (same-individual sets over every library) and
    ``pooled_libraries`` (``{re-run library: manifest sample}``).
    """
    contamination_kwargs = contamination_kwargs or {}
    imbalance_kwargs = imbalance_kwargs or {}
    conc_all = sample_genotype_concordance(counts)
    groups_all = _same_individual_groups(conc_all)
    listed = set(manifest[tumor_col]) | set(manifest[normal_col])
    replicate_of = _replicate_libraries(counts, groups_all, listed) if pool_replicates else {}
    if replicate_of:
        counts = _pool_libraries(counts, replicate_of)
        conc = sample_genotype_concordance(counts)
        groups = _same_individual_groups(conc)
    else:
        conc, groups = conc_all, groups_all
    rel = resolve_sample_relationships(conc, manifest, case_col=case_col, a_col=tumor_col, b_col=normal_col)
    tumors = set(manifest[tumor_col])
    normals = set(manifest[normal_col])
    case_of: Dict[str, str] = {}
    for rec in manifest.itertuples(index=False):
        case_of.setdefault(getattr(rec, tumor_col), getattr(rec, case_col))
        case_of.setdefault(getattr(rec, normal_col), getattr(rec, case_col))

    findings: List[dict] = []
    for r in rel.itertuples(index=False):
        if r.status != "MISMATCH":
            continue
        comp = [x for x in (r.best_for_a, r.best_for_b) if x]
        findings.append({
            "case_id": r.case_id, "finding_type": "identity_mismatch", "priority": 1,
            "affected": f"{r.a},{r.b}", "comparison": ",".join(dict.fromkeys(comp)),
            "region_id": "MULTIPLE", "support_loci": r.submitted_n_compared, "effect_size": r.submitted_discordance,
            "confidence": "HIGH",
            "evidence": (f"submitted pair {r.a}/{r.b} genotype discordance {r.submitted_discordance:.3f} over "
                         f"{r.submitted_n_compared} germline-informative loci (rank {r.submitted_rank_for_a} for {r.a}); "
                         f"best match of {r.a} is {r.best_for_a} ({r.best_for_a_discordance:.3f}), "
                         f"of {r.b} is {r.best_for_b} ({r.best_for_b_discordance:.3f})"),
        })

    # matched normal of each tumour after repair: same-individual non-tumour sample
    def matched_normal(t: str) -> Optional[str]:
        cand = [s for s in groups.get(t, {t}) if s != t]
        cand_n = [s for s in cand if s not in tumors] or cand
        if not cand_n:
            return None
        cc = conc[(conc["sample"] == t) & conc["comparator"].isin(cand_n)].sort_values("discordance")
        return cc["comparator"].iloc[0] if len(cc) else cand_n[0]

    gref = {t: [matched_normal(t)] for t in tumors if matched_normal(t)}
    ckw = dict(genotype_reference=gref, groups=groups)
    ckw.update(contamination_kwargs)
    cont = estimate_cross_sample_contamination(counts, **ckw)

    called = cont[cont["called"]] if len(cont) else cont
    admixture = []
    for c in called.itertuples(index=False):
        same = c.source in groups.get(c.affected, {c.affected})
        if same and c.affected in tumors and c.source not in tumors:
            admixture.append(c)   # normal cells in a tumour = tumour content
            continue
        if same:
            ftype = "tumor_in_normal" if c.source in tumors else "same_individual_admixture"
            comp = c.source
        else:
            ftype = "cross_sample_contamination"
            own = groups.get(c.affected, {c.affected})
            alts = sorted(s for s in groups.get(c.source, {c.source}) if s != c.source and s not in own)
            comp = ",".join([c.source] + alts)
        conf = "HIGH" if c.p_adj < 1e-6 and c.hit_sites >= 10 else "MEDIUM"
        findings.append({
            "case_id": case_of.get(c.affected, "NA"), "finding_type": ftype, "priority": 2,
            "affected": c.affected, "comparison": comp,
            "region_id": c.dominant_region, "support_loci": c.hit_sites,
            "effect_size": round(c.fraction, 4), "confidence": conf,
            "evidence": (f"estimated {c.source} fraction in {c.affected} {c.fraction:.3f} (joint mixture LRT "
                         f"{c.lr:.1f}, adj p {c.p_adj:.1e}); minority {c.source} alleles at {c.hit_sites}/"
                         f"{c.informative_sites} homozygous {c.affected} loci, mean minor AF "
                         f"{c.mean_minor_af_at_hits:.3f}; {c.private_hits} {c.source}-private markers with "
                         f"{c.private_reads} reads"),
        })

    imb_rows: List[dict] = []
    tin_rows: List[dict] = []
    for rec in manifest.itertuples(index=False):
        t = getattr(rec, tumor_col)
        n = matched_normal(t)
        if n is None:
            continue
        reg, summ = tumor_normal_allelic_imbalance(counts, t, n, **imbalance_kwargs)
        summ["case_id"] = getattr(rec, case_col)
        mix = [a for a in admixture if a.affected == t and a.source == n]
        summ["normal_admixture_fraction"] = mix[0].fraction if mix else 0.0
        tin = tumor_in_normal_shift(counts, t, n, loh_regions=summ["loh_regions"], **(tin_kwargs or {}))
        summ.update({f"tin_{k}": v for k, v in tin.items() if k not in ("tumor", "normal", "loh_regions")})
        imb_rows.append(summ)
        tin_rows.append(tin)
        if summ.get("low_tumor_content"):
            findings.append({
                "case_id": getattr(rec, case_col), "finding_type": "low_tumor_content", "priority": 3,
                "affected": t, "comparison": n, "region_id": "MULTIPLE",
                "support_loci": summ["het_loci"], "effect_size": round(summ.get("median_folded_tumor", np.nan), 3),
                "confidence": "HIGH" if summ["het_loci"] >= 30 else "MEDIUM",
                "evidence": (f"{t} vs matched {n}: no region with clonal allelic imbalance over "
                             f"{summ['het_loci']} {n}-heterozygous loci; median folded AF "
                             f"{summ.get('median_folded_tumor', np.nan):.3f} vs {summ.get('median_folded_normal', np.nan):.3f}, "
                             f"max regional imbalance {summ['max_imbalance']:.2f} ({summ['max_imbalance_region']})"),
            })
        elif report_regional_imbalance and summ["loh_regions"]:
            regs = summ["loh_regions"]
            findings.append({
                "case_id": getattr(rec, case_col), "finding_type": "regional_allelic_imbalance", "priority": 4,
                "affected": t, "comparison": n, "region_id": regs[0] if len(regs) == 1 else "MULTIPLE",
                "support_loci": summ["loh_loci"], "effect_size": round(summ["max_imbalance"], 3),
                "confidence": "MEDIUM",
                "evidence": f"{t} loses the {n} minor allele in {','.join(regs)} (max imbalance {summ['max_imbalance']:.2f})",
            })

    # tumour-in-normal from the LOH-region allelic shift (germline SNPs only)
    n_tested = max(sum(1 for r in tin_rows if r["tested"]), 1)
    have_tin = {(f["affected"], f["comparison"]) for f in findings if f["finding_type"] == "tumor_in_normal"}
    for r in tin_rows:
        if not r["tested"] or r["p_value"] * n_tested >= tin_alpha or not (r["fraction"] >= tin_min_fraction):
            continue
        t, n = r["tumor"], r["normal"]
        if (n, t) in have_tin:
            continue
        findings.append({
            "case_id": case_of.get(n, "NA"), "finding_type": "tumor_in_normal", "priority": 2,
            "affected": n, "comparison": t, "region_id": r["dominant_region"],
            "support_loci": r["loci_alt_retained"] + r["loci_ref_retained"],
            "effect_size": round(float(r["fraction"]), 4),
            "confidence": "HIGH" if r["z"] >= 5 else "MEDIUM",
            "evidence": (f"{n} heterozygous allele fraction shifted toward {t}'s retained alleles inside {t}'s LOH "
                         f"regions ({','.join(r['loh_regions'])}): direction-balanced shift {r['shift']:.3f} over "
                         f"{r['loci_alt_retained']} ALT- and {r['loci_ref_retained']} REF-retained loci, z {r['z']:.1f}; "
                         f"estimated {t}-derived fraction {r['fraction']:.3f}"),
        })

    if read_summary is not None and "outliers" in read_summary:
        for r in read_summary.itertuples(index=False):
            case = case_of.get(r.sample) or case_of.get(replicate_of.get(r.sample, ""))
            if r.outliers and case:
                findings.append({
                    "case_id": case, "finding_type": "library_outlier", "priority": 5,
                    "affected": r.sample, "comparison": "NA", "region_id": "NA", "support_loci": np.nan,
                    "effect_size": float(r.median_read_length), "confidence": "MEDIUM",
                    "evidence": f"{r.sample} read-level batch outlier: {r.outliers}",
                })

    fdf = pd.DataFrame(findings, columns=["case_id", "finding_type", "priority", "affected", "comparison",
                                          "region_id", "support_loci", "effect_size", "confidence", "evidence"])
    if len(fdf):
        fdf = (fdf.sort_values(["case_id", "priority"]).groupby("case_id", sort=False)
               .head(max_findings_per_case).reset_index(drop=True))
    return {"findings": fdf, "concordance": conc_all, "relationships": rel, "contamination": cont,
            "imbalance": pd.DataFrame(imb_rows),
            "groups": {k: sorted(v) for k, v in groups_all.items()},
            "pooled_libraries": dict(replicate_of)}
