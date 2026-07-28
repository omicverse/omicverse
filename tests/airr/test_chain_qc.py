"""``ov.airr.chain_qc`` chain-pairing granularity.

Dual TCRalpha expression is a documented consequence of incomplete allelic
exclusion at the *TRA* locus, so 2 TRA + 1 TRB cells are a biological
population rather than a technical artefact. Collapsing them into
``'multichain'`` alongside true doublets forces users to throw them away
(omicverse#902). These tests pin the finer categories and the reader flag
they depend on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

ad = pytest.importorskip("anndata")
ov = pytest.importorskip("omicverse")

from omicverse.airr._qc import (  # noqa: E402
    CHAIN_PAIRING_CATEGORIES,
    PAIRED_CATEGORIES,
    _chain_pairing,
)


def _cell(vj, vdj, *, multi=False, vj_locus="TRA", vdj_locus="TRB"):
    """One obs row with ``vj`` VJ chains and ``vdj`` VDJ chains."""
    rec = {"multi_chain": "True" if multi else "False"}
    for i in range(2):
        rec[f"VJ_{i + 1}_junction_aa"] = f"CAVJ{i}F" if i < vj else None
        rec[f"VJ_{i + 1}_locus"] = vj_locus if i < vj else None
        rec[f"VDJ_{i + 1}_junction_aa"] = f"CASVDJ{i}F" if i < vdj else None
        rec[f"VDJ_{i + 1}_locus"] = vdj_locus if i < vdj else None
    return rec


def _adata(records):
    obs = pd.DataFrame(records, index=[f"c{i}" for i in range(len(records))])
    return ad.AnnData(X=np.zeros((len(records), 1), dtype="float32"), obs=obs)


# --------------------------------------------------------------------------
# the categories themselves
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("n_vj", "n_vdj", "expected"),
    [
        (1, 1, "single pair"),
        (2, 1, "extra VJ"),        # 2 TRA + 1 TRB — the dual-alpha case
        (1, 2, "extra VDJ"),       # 1 TRA + 2 TRB
        (2, 2, "two full chains"),  # 2 TRA + 2 TRB
        (1, 0, "orphan VJ"),
        (2, 0, "orphan VJ"),
        (0, 1, "orphan VDJ"),
        (0, 2, "orphan VDJ"),
        (0, 0, "no IR"),
    ],
)
def test_chain_pairing_categories(n_vj, n_vdj, expected):
    assert _chain_pairing(n_vj, n_vdj, multi=False, ambiguous=False) == expected


def test_dual_alpha_is_not_multichain():
    """The whole point of #902: 2 TRA + 1 TRB must be distinguishable."""
    adata = _adata([_cell(2, 1)])
    ov.airr.chain_qc(adata)
    assert adata.obs["chain_pairing"].iloc[0] == "extra VJ"


def test_multichain_reserved_for_dropped_chains():
    """>2 chains in one arm is the configuration that is really suspicious."""
    adata = _adata([_cell(2, 2, multi=True)])
    ov.airr.chain_qc(adata)
    assert adata.obs["chain_pairing"].iloc[0] == "multichain"


def test_ambiguous_loci():
    adata = _adata([_cell(1, 1, vj_locus="TRA", vdj_locus="IGH")])
    ov.airr.chain_qc(adata)
    assert adata.obs["chain_pairing"].iloc[0] == "ambiguous"
    assert adata.obs["receptor_subtype"].iloc[0] == "ambiguous"


def test_chain_pairing_is_an_ordered_category():
    adata = _adata([_cell(1, 1), _cell(2, 1), _cell(0, 0)])
    ov.airr.chain_qc(adata)
    cats = list(adata.obs["chain_pairing"].cat.categories)
    assert cats == list(CHAIN_PAIRING_CATEGORIES)


def test_missing_multi_chain_column_is_tolerated():
    """h5ad files written before the reader gained the flag must still work."""
    rec = _cell(2, 1)
    del rec["multi_chain"]
    adata = _adata([rec])
    ov.airr.chain_qc(adata)
    assert adata.obs["chain_pairing"].iloc[0] == "extra VJ"


# --------------------------------------------------------------------------
# filter_chains
# --------------------------------------------------------------------------
def _labelled():
    adata = _adata([
        _cell(1, 1),            # single pair
        _cell(2, 1),            # extra VJ
        _cell(1, 2),            # extra VDJ
        _cell(2, 2),            # two full chains
        _cell(1, 0),            # orphan VJ
        _cell(2, 2, multi=True),  # multichain
        _cell(0, 0),            # no IR
    ])
    ov.airr.chain_qc(adata)
    return adata


def test_filter_strict_keeps_only_single_pair():
    out = ov.airr.filter_chains(_labelled(), keep="strict")
    assert set(out.obs["chain_pairing"].astype(str)) == {"single pair"}
    assert out.n_obs == 1


def test_filter_paired_keeps_dual_receptor_cells():
    out = ov.airr.filter_chains(_labelled(), keep="paired")
    assert set(out.obs["chain_pairing"].astype(str)) == set(PAIRED_CATEGORIES)
    assert out.n_obs == 4


def test_filter_any_ir_drops_only_no_ir():
    out = ov.airr.filter_chains(_labelled(), keep="any IR")
    assert "no IR" not in set(out.obs["chain_pairing"].astype(str))
    assert out.n_obs == 6


def test_filter_explicit_list():
    out = ov.airr.filter_chains(_labelled(), keep=["single pair", "extra VJ"])
    assert out.n_obs == 2


def test_filter_preserves_cell_order():
    adata = _labelled()
    out = ov.airr.filter_chains(adata, keep="any IR")
    assert list(out.obs_names) == [n for n in adata.obs_names if n != "c6"]


def test_filter_rejects_unknown_category():
    with pytest.raises(ValueError, match="unknown chain_pairing categories"):
        ov.airr.filter_chains(_labelled(), keep=["single pair", "nonsense"])


def test_filter_rejects_unknown_preset():
    with pytest.raises(ValueError, match="keep must be one of"):
        ov.airr.filter_chains(_labelled(), keep="everything")


def test_filter_without_chain_qc():
    with pytest.raises(KeyError, match="chain_qc"):
        ov.airr.filter_chains(_adata([_cell(1, 1)]))


# --------------------------------------------------------------------------
# the reader flag the classification depends on
# --------------------------------------------------------------------------
def _contigs(cell_chains):
    """Build a 10x-style contig table.

    ``{cell: [(locus, umis), ...]}``; a ``umis`` entry may instead be
    ``(umis, None)`` to mark a contig that carries no CDR3.
    """
    rows = []
    for cell, chains in cell_chains.items():
        for i, (locus, umis) in enumerate(chains):
            has_cdr3 = True
            if isinstance(umis, tuple):
                umis, has_cdr3 = umis[0], umis[1] is not None
            rows.append({
                "barcode": cell,
                "contig_id": f"{cell}_{i}",
                "chain": locus,  # 10x column name; the reader renames it to `locus`
                "v_gene": f"{locus}V1",
                "j_gene": f"{locus}J1",
                "cdr3": f"CA{i}F" if has_cdr3 else None,
                "cdr3_nt": "TGT" * 4 if has_cdr3 else None,
                "productive": "True",
                "umis": umis,
            })
    return pd.DataFrame(rows)


def test_reader_flags_more_than_two_chains_per_arm(tmp_path):
    csv = tmp_path / "filtered_contig_annotations.csv"
    _contigs({
        "AAA-1": [("TRA", 30), ("TRB", 25)],                 # single pair
        "BBB-1": [("TRA", 30), ("TRA", 20), ("TRB", 25)],    # genuine dual alpha
        "CCC-1": [("TRA", 30), ("TRA", 20), ("TRA", 10), ("TRB", 25)],  # dropped
    }).to_csv(csv, index=False)

    adata = ov.airr.read_10x_vdj(str(csv))
    flags = adata.obs["multi_chain"].astype(str).to_dict()
    assert flags["AAA-1"] == "False"
    assert flags["BBB-1"] == "False"
    assert flags["CCC-1"] == "True"

    ov.airr.chain_qc(adata)
    pairing = adata.obs["chain_pairing"].astype(str).to_dict()
    assert pairing["AAA-1"] == "single pair"
    assert pairing["BBB-1"] == "extra VJ"
    assert pairing["CCC-1"] == "multichain"


def test_reader_keeps_the_highest_umi_chains(tmp_path):
    """The dropped chain must be the weakest one, not an arbitrary one."""
    csv = tmp_path / "filtered_contig_annotations.csv"
    _contigs({"AAA-1": [("TRA", 5), ("TRA", 50), ("TRA", 30), ("TRB", 25)]}).to_csv(
        csv, index=False
    )
    adata = ov.airr.read_10x_vdj(str(csv))
    kept = [
        adata.obs["VJ_1_duplicate_count"].iloc[0],
        adata.obs["VJ_2_duplicate_count"].iloc[0],
    ]
    assert sorted(float(x) for x in kept) == [30.0, 50.0]


def test_overflow_is_counted_per_receptor_system(tmp_path):
    """An ambient BCR contig must not make a dual-alpha T cell 'multichain'.

    2 TRA + 1 TRB is a dual-alpha TCR; a stray IGK contig alongside it is
    contamination from another receptor system, not a third TCR chain.
    """
    csv = tmp_path / "filtered_contig_annotations.csv"
    _contigs({
        "AAA-1": [("TRA", 30), ("TRA", 20), ("IGK", 8), ("TRB", 25)],
        "BBB-1": [("TRA", 30), ("TRA", 20), ("TRA", 8), ("TRB", 25)],
    }).to_csv(csv, index=False)

    adata = ov.airr.read_10x_vdj(str(csv))
    ov.airr.chain_qc(adata)
    pairing = adata.obs["chain_pairing"].astype(str).to_dict()
    assert pairing["AAA-1"] == "extra VJ"     # dual alpha + ambient IGK
    assert pairing["BBB-1"] == "multichain"   # three genuine alpha chains


def test_junctionless_contigs_are_not_chains(tmp_path):
    """A contig with no CDR3 cannot define a clone, so it is not a chain.

    It must neither trip the overflow flag nor take a slot away from a
    contig that does carry a junction, however high its UMI count.
    """
    csv = tmp_path / "filtered_contig_annotations.csv"
    _contigs({
        # three TRA contigs, but one has no CDR3 -> two real chains
        "AAA-1": [("TRA", 30), ("TRA", 20), ("TRA", (99, None)), ("TRB", 25)],
        # the junction-less contig has the highest UMI count of the arm
        "BBB-1": [("TRA", (99, None)), ("TRA", 20), ("TRB", 25)],
    }).to_csv(csv, index=False)

    adata = ov.airr.read_10x_vdj(str(csv))
    assert adata.obs["multi_chain"].astype(str).to_dict()["AAA-1"] == "False"

    ov.airr.chain_qc(adata)
    pairing = adata.obs["chain_pairing"].astype(str).to_dict()
    assert pairing["AAA-1"] == "extra VJ"
    # the real chain outranks the junction-less one despite 20 UMIs vs 99
    assert adata.obs.loc["BBB-1", "VJ_1_junction_aa"] is not None
    assert float(adata.obs.loc["BBB-1", "VJ_1_duplicate_count"]) == 20.0
    assert pairing["BBB-1"] == "single pair"


def test_locus_without_junction_does_not_make_a_cell_ambiguous(tmp_path):
    """A junction-less IGH contig must not mark a clean TRA+TRB cell ambiguous."""
    csv = tmp_path / "filtered_contig_annotations.csv"
    _contigs({
        "AAA-1": [("TRA", 30), ("TRB", 25), ("IGH", (9, None))],
        "BBB-1": [("TRA", 30), ("TRB", 25), ("IGH", 9)],
    }).to_csv(csv, index=False)

    adata = ov.airr.read_10x_vdj(str(csv))
    ov.airr.chain_qc(adata)
    pairing = adata.obs["chain_pairing"].astype(str).to_dict()
    assert pairing["AAA-1"] == "single pair"  # IGH carries no CDR3 -> not a chain
    assert pairing["BBB-1"] == "ambiguous"    # a real IGH chain in a T cell
