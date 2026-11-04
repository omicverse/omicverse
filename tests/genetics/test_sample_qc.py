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


def _simulate(seed=0, n_sites=320, depth=60):
    """4 individuals x (tumour, normal); known LOH, admixture and contamination."""
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
    for s, af in true_af.items():
        m = mix.get(s, {})
        f = (1 - sum(m.values())) * af + sum(a * true_af[src] for src, a in m.items())
        p = np.clip(f * (1 - 0.004) + (1 - f) * 0.004, 0, 1)
        n = rng.poisson(depth, n_sites)
        k = rng.binomial(n, p)
        for j in range(n_sites):
            rows.append((s, "chr1", 1000 + j * 50, "A", "G", regions[j], n[j] - k[j], k[j], 0, n[j],
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
