"""Tests for ov.genetics sample-level QC (identity / contamination / tumour-normal)."""

import numpy as np
import pandas as pd
import pytest

from omicverse.genetics._sample_qc import (
    estimate_cross_sample_contamination,
    resolve_sample_relationships,
    sample_genotype_concordance,
    tumor_normal_allelic_imbalance,
    tumor_normal_batch_qc,
)


def _simulate(seed=0, n_sites=320, depth=60, reruns=None):
    """4 individuals x (tumour, normal); known LOH, admixture and contamination.

    ``reruns`` = ``{new_library: sample}`` adds independently sequenced extra
    libraries of existing samples (same DNA, new read sampling)."""
    rng = np.random.default_rng(seed)
    regions = np.repeat([f"r{i}" for i in range(8)], n_sites // 8)
    maf = rng.uniform(0.2, 0.5, n_sites)
    ind = {i: rng.binomial(2, maf) / 2 for i in range(4)}           # germline AF 0/0.5/1
    true_af = {}
    for i in range(4):
        g = ind[i].copy()
        true_af[f"N{i}"] = g.copy()
        t = g.copy()
        if i != 3:                                                   # T3: no LOH (low content)
            loh = np.isin(regions, [f"r{i}", f"r{i + 4}"]) & (g == 0.5)
            t[loh] = rng.choice([0.0, 1.0], loh.sum())
        # tumour-private somatic markers
        som = rng.choice(np.flatnonzero(g == 0), 12, replace=False)
        t[som] = 0.5
        true_af[f"T{i}"] = t
    mix = {
        "T0": {"N0": 0.10},                 # normal admixture: tumour content, not contamination
        "T1": {"N1": 0.10, "T2": 0.06},     # cross-individual contamination of a tumour
        "N2": {"T2": 0.05},                 # tumour-in-normal
        "T3": {"N3": 0.60},                 # low tumour content
        "T2": {"N2": 0.10},
    }
    rows = []
    libraries = [(s, s) for s in true_af] + list((reruns or {}).items())
    for lib, s in libraries:
        af = true_af[s]
        m = mix.get(s, {})
        f = (1 - sum(m.values())) * af + sum(a * true_af[src] for src, a in m.items())
        p = np.clip(f * (1 - 0.004) + (1 - f) * 0.004, 0, 1)
        n = rng.poisson(depth, n_sites)
        k = rng.binomial(n, p)
        for j in range(n_sites):
            rows.append((lib, "chr1", 1000 + j * 50, "A", "G", regions[j], n[j] - k[j], k[j], 0, n[j],
                         k[j] / n[j] if n[j] else np.nan, True))
    counts = pd.DataFrame(rows, columns=["sample", "chrom", "pos", "ref", "alt", "region_id", "ref_count",
                                         "alt_count", "other_count", "depth", "alt_fraction", "in_target"])
    # P1 is a swapped record: T1 submitted with N0
    manifest = pd.DataFrame({
        "case_id": ["P0", "P1", "P2", "P3"],
        "tumor_alias": ["T0", "T1", "T2", "T3"],
        "submitted_baseline_alias": ["N1", "N0", "N2", "N3"],
    })
    return counts, manifest


@pytest.fixture(scope="module")
def sim():
    return _simulate()


def test_concordance_finds_same_individual(sim):
    counts, _ = sim
    conc = sample_genotype_concordance(counts)
    best = conc[conc["rank"] == 1].set_index("sample")["comparator"]
    for i in range(4):
        assert best[f"T{i}"] == f"N{i}"
        assert best[f"N{i}"] == f"T{i}"
    assert conc.loc[(conc["sample"] == "T1") & (conc["comparator"] == "N1"), "mutual_best"].item()


def test_relationships_flag_swapped_records(sim):
    counts, manifest = sim
    rel = resolve_sample_relationships(sample_genotype_concordance(counts), manifest)
    st = rel.set_index("case_id")["status"]
    assert st["P0"] == "MISMATCH" and st["P1"] == "MISMATCH"
    assert st["P2"] == "SUPPORTED" and st["P3"] == "SUPPORTED"
    r1 = rel.set_index("case_id").loc["P1"]
    assert r1["best_for_a"] == "N1" and r1["best_for_b"] == "T0"


def test_contamination_fraction_and_source(sim):
    counts, _ = sim
    cont = estimate_cross_sample_contamination(counts)
    called = cont[cont["called"]].set_index(["affected", "source"])
    assert ("T1", "T2") in called.index
    assert ("N2", "T2") in called.index
    # simulated 6% / 5% admixture, estimated within a factor ~2 (few informative markers)
    assert 0.025 < called.loc[("T1", "T2"), "fraction"] < 0.13
    assert 0.02 < called.loc[("N2", "T2"), "fraction"] < 0.12
    # clean normals carry no called source
    for clean in ("N0", "N1", "N3"):
        assert clean not in called.index.get_level_values(0)
    # the tumour-in-normal source is supported at most of its informative sites
    assert called.loc[("N2", "T2"), "hit_sites"] >= 0.5 * called.loc[("N2", "T2"), "informative_sites"]


def test_allelic_imbalance_and_low_content(sim):
    counts, _ = sim
    reg, summ = tumor_normal_allelic_imbalance(counts, "T1", "N1")
    assert set(summ["loh_regions"]) >= {"r1", "r5"}
    assert not summ["low_tumor_content"]
    _, summ3 = tumor_normal_allelic_imbalance(counts, "T3", "N3")
    assert summ3["low_tumor_content"]


def test_batch_qc_findings(sim):
    counts, manifest = sim
    res = tumor_normal_batch_qc(counts, manifest)
    f = res["findings"]
    types = {(r.case_id, r.finding_type, r.affected) for r in f.itertuples()}
    assert ("P0", "identity_mismatch", "T0,N1") in types
    assert ("P1", "identity_mismatch", "T1,N0") in types
    assert ("P1", "cross_sample_contamination", "T1") in types
    assert ("P2", "tumor_in_normal", "N2") in types
    assert ("P3", "low_tumor_content", "T3") in types
    # normal admixture of a tumour is never reported as contamination
    cont_rows = f[f["finding_type"].isin(["cross_sample_contamination", "tumor_in_normal"])]
    for r in cont_rows.itertuples():
        assert not (r.affected.startswith("T") and r.comparison.split(",")[0] == "N" + r.affected[1:])
    t1 = f[(f.finding_type == "cross_sample_contamination") & (f.affected == "T1")].iloc[0]
    assert t1["comparison"].split(",")[0] == "T2"
    assert (f.groupby("case_id").size() <= 4).all()


def test_snp_allele_counts_from_bam(tmp_path):
    pysam = pytest.importorskip("pysam")
    ref = "ACGT" * 50
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": len(ref)}]}
    path = tmp_path / "S1.bam"
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for i in range(10):
            seq = list(ref[90:110])
            if i < 3:
                seq[10] = "T"          # site at 1-based pos 101 (ref 'A')
            a = pysam.AlignedSegment()
            a.query_name = f"r{i}"
            a.query_sequence = "".join(seq)
            a.flag = 0
            a.reference_id = 0
            a.reference_start = 90
            a.mapping_quality = 60
            a.cigar = [(0, 20)]
            a.query_qualities = pysam.qualitystring_to_array("I" * 20)
            out.write(a)
    pysam.index(str(path))
    sites = pd.DataFrame({"chrom": ["chr1"], "pos": [101], "ref": [ref[100]], "alt": ["T"]})
    targets = pd.DataFrame({"chrom": ["chr1"], "start": [50], "end": [150], "name": ["t1"]})
    from omicverse.genetics._sample_qc import snp_allele_counts
    c = snp_allele_counts({"S1": str(path)}, sites, targets=targets)
    row = c.iloc[0]
    assert (row["ref_count"], row["alt_count"], row["region_id"]) == (7, 3, "t1")
    assert abs(row["alt_fraction"] - 0.3) < 1e-9


def _simulate_germline_only(seed=1, n_sites=640, depth=60, tin=0.10, cont=0.03):
    """3 individuals, germline SNPs only (no somatic markers), LOH tumours.

    N0 carries ``tin`` of its own tumour T0 (tumour-in-normal, visible only as
    an allelic shift inside T0's LOH regions); T1 carries ``cont`` of the
    normal N2 of another individual whose tumour T2 lost alleles by LOH.
    """
    rng = np.random.default_rng(seed)
    regions = np.repeat([f"r{i}" for i in range(8)], n_sites // 8)
    maf = rng.uniform(0.2, 0.5, n_sites)
    true_af = {}
    for i in range(3):
        g = rng.binomial(2, maf) / 2
        true_af[f"N{i}"] = g.copy()
        t = g.copy()
        loh = np.isin(regions, [f"r{i}", f"r{i + 3}", f"r{i + 5}"]) & (g == 0.5)
        t[loh] = rng.choice([0.0, 1.0], loh.sum())
        true_af[f"T{i}"] = 0.9 * t + 0.1 * g                       # 90 % tumour content
    mix = {"N0": {"T0": tin}, "T1": {"N2": cont}}
    rows = []
    for s, af in true_af.items():
        m = mix.get(s, {})
        f = (1 - sum(m.values())) * af + sum(a * true_af[src] for src, a in m.items())
        p = np.clip(f * (1 - 0.003) + (1 - f) * 0.003, 0, 1)
        n = rng.poisson(depth, n_sites)
        k = rng.binomial(n, p)
        for j in range(n_sites):
            rows.append((s, "chr1", 1000 + j * 50, "A", "G", regions[j], n[j] - k[j], k[j], 0, n[j],
                         k[j] / n[j] if n[j] else np.nan, True))
    counts = pd.DataFrame(rows, columns=["sample", "chrom", "pos", "ref", "alt", "region_id", "ref_count",
                                         "alt_count", "other_count", "depth", "alt_fraction", "in_target"])
    manifest = pd.DataFrame({"case_id": ["P0", "P1", "P2"], "tumor_alias": ["T0", "T1", "T2"],
                             "submitted_baseline_alias": ["N0", "N1", "N2"]})
    return counts, manifest


@pytest.fixture(scope="module")
def sim_germline():
    return _simulate_germline_only()


def test_tumor_in_normal_shift_without_markers(sim_germline):
    from omicverse.genetics._sample_qc import tumor_in_normal_shift
    counts, _ = sim_germline
    r = tumor_in_normal_shift(counts, "T0", "N0")
    assert r["tested"] and r["loci_alt_retained"] >= 5 and r["loci_ref_retained"] >= 5
    assert r["z"] > 4 and r["p_value"] < 1e-4
    assert 0.04 < r["fraction"] < 0.2                 # simulated 0.10
    for i in (1, 2):                                   # clean normals: no shift
        c = tumor_in_normal_shift(counts, f"T{i}", f"N{i}")
        assert c["tested"] and abs(c["z"]) < 3.5
    # a tumour without LOH regions gives no power, not a call
    nolo = tumor_in_normal_shift(counts, "T0", "N0", loh_regions=[])
    assert not nolo["tested"]


def test_contamination_individual_level_attribution(sim_germline):
    counts, _ = sim_germline
    groups = {"T0": ["T0", "N0"], "N0": ["T0", "N0"], "T1": ["T1", "N1"], "N1": ["T1", "N1"],
              "T2": ["T2", "N2"], "N2": ["T2", "N2"]}
    cont = estimate_cross_sample_contamination(
        counts, genotype_reference={"T0": ["N0"], "T1": ["N1"], "T2": ["N2"]}, groups=groups)
    called = cont[cont["called"]]
    hit = called[called["affected"] == "T1"]
    assert len(hit) == 1                                # one call per source individual
    row = hit.iloc[0]
    assert row["source"] == "N2" and row["best_in_individual"]   # normal fits better than its LOH tumour
    assert 0.015 < row["fraction"] < 0.06               # simulated 0.03
    alt = cont[(cont["affected"] == "T1") & (cont["source"] == "T2")].iloc[0]
    assert not alt["called"] and alt["lr"] == pytest.approx(row["lr"])
    # no foreign-individual call anywhere else
    assert set(called.loc[called["source"].map(lambda s: s[1:]) != called["affected"].map(lambda s: s[1:]),
                          "affected"]) <= {"T1"}


def test_batch_qc_germline_only_panel(sim_germline):
    counts, manifest = sim_germline
    f = tumor_normal_batch_qc(counts, manifest)["findings"]
    got = {(r.case_id, r.finding_type, r.affected, r.comparison.split(",")[0]) for r in f.itertuples()}
    assert ("P0", "tumor_in_normal", "N0", "T0") in got
    assert ("P1", "cross_sample_contamination", "T1", "N2") in got
    assert not any(t in ("identity_mismatch", "low_tumor_content") for _, t, _, _ in got)
    assert not ((f["finding_type"] == "tumor_in_normal") & (f["affected"] != "N0")).any()


# --- several libraries of one individual ------------------------------------
def test_extra_libraries_stay_with_their_individual():
    """A re-sequenced normal used to break the tumour/normal group.

    Two libraries of N2 are each other's best match, so T2 -- whose best is
    one of them -- was the mutual best of neither, was grouped with nobody
    and its correct record came out MISMATCH."""
    counts, manifest = _simulate(reruns={"N2-rerun": "N2", "N2-rerun2": "N2",
                                         "T3-lib2": "T3"})
    conc = sample_genotype_concordance(counts)
    rel = resolve_sample_relationships(conc, manifest).set_index("case_id")
    assert rel.loc["P2", "status"] == "SUPPORTED"          # was MISMATCH
    assert rel.loc["P3", "status"] == "SUPPORTED"
    # the planted swap is still caught
    assert rel.loc["P0", "status"] == "MISMATCH" and rel.loc["P1", "status"] == "MISMATCH"

    res = tumor_normal_batch_qc(counts, manifest)
    groups = res["groups"]
    assert set(groups["T2"]) == {"T2", "N2", "N2-rerun", "N2-rerun2"}
    assert set(groups["T3"]) == {"T3", "N3", "T3-lib2"}
    for i in (0, 1):                                       # individuals stay apart
        assert set(groups[f"T{i}"]) == {f"T{i}", f"N{i}"}
    assert res["pooled_libraries"] == {"N2-rerun": "N2", "N2-rerun2": "N2", "T3-lib2": "T3"}
    # same findings as the batch without the extra libraries: nothing on an
    # unlisted library, no re-run fitted as a "source" of its own sample
    f = res["findings"]
    got = {(r.case_id, r.finding_type, r.affected) for r in f.itertuples()}
    ref = tumor_normal_batch_qc(*_simulate())["findings"]
    assert got == {(r.case_id, r.finding_type, r.affected) for r in ref.itertuples()}
    assert ("P2", "tumor_in_normal", "N2") in got and "NA" not in set(f["case_id"])


def test_unrelated_orphan_is_not_absorbed():
    """A sample whose own partner is missing must stay on its own."""
    counts, manifest = _simulate(reruns={"N2-rerun": "N2"})
    counts = counts[counts["sample"] != "N3"]              # T3's normal is missing
    conc = sample_genotype_concordance(counts)
    from omicverse.genetics._sample_qc import _same_individual_groups
    groups = _same_individual_groups(conc)
    assert groups["T3"] == {"T3"}
    assert groups["T2"] == {"T2", "N2", "N2-rerun"}


def _simulate_parent_child(seed=0, n_sites=320, depth=60):
    """Parent (0), child (1) and an unrelated person (2), each tumour + normal.

    A child carries one allele of its parent at every site, so the two never
    show opposite homozygotes -- yet they are different people."""
    rng = np.random.default_rng(seed)
    regions = np.repeat([f"r{i}" for i in range(8)], n_sites // 8)
    maf = rng.uniform(0.2, 0.5, n_sites)
    hap = lambda: rng.binomial(1, maf)                               # noqa: E731
    parent = (hap(), hap())
    child = (np.where(rng.random(n_sites) < 0.5, parent[0], parent[1]), hap())
    germ = {0: (parent[0] + parent[1]) / 2, 1: (child[0] + child[1]) / 2, 2: (hap() + hap()) / 2}
    true_af = {}
    for i, g in germ.items():
        true_af[f"N{i}"] = g.copy()
        t = g.copy()
        loh = np.isin(regions, [f"r{i}", f"r{i + 4}"]) & (g == 0.5)
        t[loh] = rng.choice([0.0, 1.0], loh.sum())
        true_af[f"T{i}"] = 0.9 * t + 0.1 * g
    rows = []
    for s, f in true_af.items():
        p = np.clip(f * (1 - 0.004) + (1 - f) * 0.004, 0, 1)
        n = rng.poisson(depth, n_sites)
        k = rng.binomial(n, p)
        for j in range(n_sites):
            rows.append((s, "chr1", 1000 + j * 50, "A", "G", regions[j], n[j] - k[j], k[j], 0, n[j],
                         k[j] / n[j] if n[j] else np.nan, True))
    return pd.DataFrame(rows, columns=["sample", "chrom", "pos", "ref", "alt", "region_id", "ref_count",
                                       "alt_count", "other_count", "depth", "alt_fraction", "in_target"])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_relatives_are_not_merged_into_one_individual(seed):
    """Parent and child share an allele at every site (no opposite
    homozygotes), so the opposite-homozygote merge of #1030 joined them into
    one person: a record pairing the parent's tumour with the child's normal
    came out SUPPORTED. They disagree in both het/hom directions, which tumour
    LOH never does."""
    from omicverse.genetics._sample_qc import _same_individual_groups
    counts = _simulate_parent_child(seed)
    conc = sample_genotype_concordance(counts)
    groups = _same_individual_groups(conc)
    assert {tuple(sorted(g)) for g in groups.values()} == {("N0", "T0"), ("N1", "T1"), ("N2", "T2")}
    manifest = pd.DataFrame({"case_id": ["P0", "P1", "P2"], "tumor_alias": ["T0", "T1", "T2"],
                             "submitted_baseline_alias": ["N1", "N0", "N2"]})   # P0/P1 swapped normals
    st = resolve_sample_relationships(conc, manifest).set_index("case_id")["status"]
    assert st["P0"] == "MISMATCH" and st["P1"] == "MISMATCH" and st["P2"] == "SUPPORTED"
    f = tumor_normal_batch_qc(counts, manifest)["findings"]
    assert {"P0", "P1"} <= set(f.loc[f["finding_type"] == "identity_mismatch", "case_id"])
    pc = conc[(conc["sample"] == "N0") & (conc["comparator"] == "N1")].iloc[0]
    assert pc["opposite_hom"] == 0 and pc["loh_inconsistent"] > 0.1 * pc["n_compared"]
    for i in range(3):                                     # tumour vs own normal: LOH is one-directional
        tn = conc[(conc["sample"] == f"T{i}") & (conc["comparator"] == f"N{i}")].iloc[0]
        assert tn["loh_inconsistent"] <= 2
