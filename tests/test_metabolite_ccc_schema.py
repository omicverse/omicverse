"""Schema tests for the MetaboliteCCC comm-AnnData built by
``omicverse.single._metabolism._mebocost_to_comm_adata``.

These tests exercise the pivot from MEBOCOST's long-format result table
(one row per communication event) into the wide
``(cell_pair × interaction)`` matrix consumed by ``ov.pl.ccc_*``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def synthetic_mebocost_result() -> pd.DataFrame:
    """Synthetic result table covering 3 cell types × 2 metabolite-sensor pairs."""
    rows = [
        # Malignant receiving L-glutamine via SLC3A2
        ("Fibroblast",  "Malignant", "HMDB0000641", "L-Glutamine",
         "SLC3A2",  4.2, 0.001, 0.005),
        # Fibroblast receiving the same via SLC1A5
        ("Macrophage", "Malignant", "HMDB0000641", "L-Glutamine",
         "SLC1A5",  3.7, 0.002, 0.008),
        ("Fibroblast", "Fibroblast","HMDB0000122", "D-Glucose",
         "SLC2A1",   2.5, 0.010, 0.040),
        ("Macrophage", "Malignant", "HMDB0000122", "D-Glucose",
         "SLC2A1",   3.1, 0.003, 0.015),
        # one filtered-out event (pval > 0.05)
        ("Malignant",  "Fibroblast","HMDB0000641", "L-Glutamine",
         "SLC1A5",   0.8, 0.4,   0.5),
    ]
    return pd.DataFrame(rows, columns=[
        "Sender", "Receiver", "Metabolite", "Metabolite_Name", "Sensor",
        "Commu_Score", "permutation_test_pval", "permutation_test_fdr",
    ])


def test_wide_schema_basic(synthetic_mebocost_result):
    from omicverse.single._metabolism import _mebocost_to_comm_adata

    comm = _mebocost_to_comm_adata(synthetic_mebocost_result, pvalue_threshold=0.05)
    # All filtered events kept; the 0.5-fdr one dropped.
    expected_pairs = {"Fibroblast|Malignant", "Macrophage|Malignant",
                      "Fibroblast|Fibroblast"}
    expected_interactions = {"L-Glutamine → SLC3A2",
                             "L-Glutamine → SLC1A5",
                             "D-Glucose → SLC2A1"}
    assert set(comm.obs_names) == expected_pairs
    assert set(comm.var_names) == expected_interactions
    assert comm.layers["means"].shape == (3, 3)
    assert comm.layers["pvalues"].shape == (3, 3)


def test_var_metadata_has_ligand_receptor(synthetic_mebocost_result):
    """``ov.pl.ccc_*`` interaction-level plots read gene_a/gene_b — the
    metabolite plays the ligand role, the sensor plays the receptor role.
    """
    from omicverse.single._metabolism import _mebocost_to_comm_adata

    comm = _mebocost_to_comm_adata(synthetic_mebocost_result, pvalue_threshold=0.05)
    for col in ("gene_a", "gene_b", "ligand", "receptor",
                "metabolite", "sensor", "interaction_name",
                "classification", "classification_super"):
        assert col in comm.var.columns, f"missing {col}"

    gln = comm.var.loc["L-Glutamine → SLC3A2"]
    assert gln["gene_a"] == "L-Glutamine"
    assert gln["gene_b"] == "SLC3A2"
    assert gln["metabolite"] == "L-Glutamine"
    assert gln["sensor"] == "SLC3A2"


def test_means_pivot_correctness(synthetic_mebocost_result):
    """Score lands in the right (pair, interaction) cell; absent events = 0."""
    from omicverse.single._metabolism import _mebocost_to_comm_adata

    comm = _mebocost_to_comm_adata(synthetic_mebocost_result, pvalue_threshold=0.05)
    means = pd.DataFrame(comm.layers["means"], index=comm.obs_names,
                         columns=comm.var_names)
    # Score 4.2 for Fibroblast→Malignant via L-Glutamine→SLC3A2.
    assert means.loc["Fibroblast|Malignant", "L-Glutamine → SLC3A2"] == pytest.approx(4.2)
    # Fibroblast→Fibroblast did not send L-Glutamine, so cell should be 0.
    assert means.loc["Fibroblast|Fibroblast", "L-Glutamine → SLC3A2"] == 0.0


def test_pvalues_pivot_with_default_one(synthetic_mebocost_result):
    """Cells without an event must be filled with p=1 (so they are non-significant).
    Otherwise downstream FDR-thresholded plots would treat 'no event' as
    'highly significant'."""
    from omicverse.single._metabolism import _mebocost_to_comm_adata

    comm = _mebocost_to_comm_adata(synthetic_mebocost_result, pvalue_threshold=0.05)
    pvals = pd.DataFrame(comm.layers["pvalues"], index=comm.obs_names,
                         columns=comm.var_names)
    assert pvals.loc["Fibroblast|Fibroblast", "L-Glutamine → SLC3A2"] == pytest.approx(1.0)
    # The 0.005 fdr should land at the right cell.
    assert pvals.loc["Fibroblast|Malignant", "L-Glutamine → SLC3A2"] == pytest.approx(0.005)


def test_hmdb_classification_join(synthetic_mebocost_result):
    """L-Glutamine should resolve to an HMDB sub_class that is *not*
    'Unclassified' — the join with HMDB tables in the omicverse data bundle
    must succeed."""
    from omicverse.single._metabolism import _mebocost_to_comm_adata

    comm = _mebocost_to_comm_adata(synthetic_mebocost_result, pvalue_threshold=0.05)
    cls = comm.var.set_index("metabolite")["classification"]
    sup = comm.var.set_index("metabolite")["classification_super"]
    # L-Glutamine is "Amino acids, peptides, and analogues" in HMDB
    assert "Unclassified" not in cls.loc["L-Glutamine"]
    assert "Unclassified" not in sup.loc["L-Glutamine"]


def test_backward_compat_obs_sender_receiver(synthetic_mebocost_result):
    """``ov.pl.ccc_*`` reads obs['sender'] / obs['receiver']. Confirm those
    survive the long→wide refactor."""
    from omicverse.single._metabolism import _mebocost_to_comm_adata

    comm = _mebocost_to_comm_adata(synthetic_mebocost_result, pvalue_threshold=0.05)
    for col in ("sender", "receiver"):
        assert col in comm.obs.columns

    senders = set(comm.obs["sender"].unique())
    receivers = set(comm.obs["receiver"].unique())
    assert senders == {"Fibroblast", "Macrophage"}
    assert receivers == {"Malignant", "Fibroblast"}


def test_uns_marker_kept(synthetic_mebocost_result):
    from omicverse.single._metabolism import _mebocost_to_comm_adata

    comm = _mebocost_to_comm_adata(synthetic_mebocost_result, pvalue_threshold=0.05)
    assert comm.uns.get("mebocost_comm") is True


def _bigger_synthetic_result() -> pd.DataFrame:
    """A 5-celltype × 4-metabolite × 4-sensor random scaffold so the
    integration plots have enough material to render."""
    np.random.seed(0)
    rows = []
    celltypes = ["Malignant", "Fibroblast", "T cell", "Macrophage", "Endothelial"]
    metabolites = [
        ("HMDB0000641", "L-Glutamine"),
        ("HMDB0000122", "D-Glucose"),
        ("HMDB0000159", "L-Phenylalanine"),
        ("HMDB0000148", "L-Glutamic acid"),
    ]
    sensors = ["SLC3A2", "SLC1A5", "SLC7A5", "SLC2A1"]
    for s in celltypes:
        for r in celltypes:
            if s == r:
                continue
            for h, m in metabolites:
                for sn in sensors:
                    if np.random.rand() > 0.45:
                        continue
                    rows.append((
                        s, r, h, m, sn,
                        abs(np.random.randn() * 2) + 0.5,
                        np.random.rand() * 0.05,
                        np.random.rand() * 0.1,
                    ))
    return pd.DataFrame(rows, columns=[
        "Sender", "Receiver", "Metabolite", "Metabolite_Name", "Sensor",
        "Commu_Score", "permutation_test_pval", "permutation_test_fdr",
    ])


def test_categorical_obs_stripped_when_threaded():
    """``omicverse.external.mebocost.run_mebocost`` should defensively cast
    Categorical obs columns to str when ``thread > 1``, otherwise
    multiprocessing forks fail to unpickle Categorical arrays under
    newer pandas (``NotImplementedError`` in ``Categorical.__setstate__``).

    We don't actually run the full inference here — we test the
    pre-processing branch by intercepting ``create_obj`` and inspecting
    what AnnData lands inside it.
    """
    import importlib

    from anndata import AnnData
    from unittest import mock

    ad_obs = pd.DataFrame({
        "celltype": pd.Categorical(["A", "B", "A", "B"]),
        "patient": pd.Categorical(["P1", "P1", "P2", "P2"]),
        "extra_num": [1.0, 2.0, 3.0, 4.0],
    }, index=[f"c{i}" for i in range(4)])
    adata = AnnData(X=np.random.rand(4, 3).astype("float32"), obs=ad_obs,
                    var=pd.DataFrame(index=["g0", "g1", "g2"]))

    captured = {}

    def _stub_create_obj(*args, **kwargs):
        # Capture and short-circuit
        captured["adata"] = kwargs["adata"]
        raise RuntimeError("intercepted")

    mod = importlib.import_module("omicverse.external.mebocost")
    with mock.patch.object(mod, "_get_create_obj",
                           return_value=(_stub_create_obj, None, None)):
        with pytest.raises(RuntimeError, match="intercepted"):
            mod.run_mebocost(adata, group_key="celltype", thread=4,
                             verbose=True, n_shuffle=2)

    captured_obs = captured["adata"].obs
    assert str(captured_obs["celltype"].dtype) != "category"
    assert str(captured_obs["patient"].dtype) != "category"
    # The user's input adata must NOT have been mutated.
    assert str(adata.obs["celltype"].dtype) == "category"


@pytest.mark.parametrize("plot_call", [
    ("ccc_heatmap", dict(plot_type="heatmap", display_by="aggregation")),
    ("ccc_heatmap", dict(plot_type="dot", display_by="interaction", top_n=10)),
    ("ccc_heatmap", dict(plot_type="dot", display_by="interaction",
                         sender_use=["Fibroblast"], top_n=10)),
    ("ccc_heatmap", dict(plot_type="tile", display_by="interaction", top_n=8)),
    ("ccc_heatmap", dict(plot_type="pathway_bubble", top_n=5)),
    ("ccc_stat_plot", dict(plot_type="bar", display_by="interaction",
                           group_by="interaction", top_n=10)),
    ("ccc_stat_plot", dict(plot_type="scatter")),
    ("ccc_stat_plot", dict(plot_type="sankey", display_by="interaction", top_n=10)),
    ("ccc_network_plot", dict(plot_type="circle")),
    ("ccc_network_plot", dict(plot_type="bipartite", ligand="L-Glutamine", top_n=5)),
])
def test_ccc_plot_smoke(plot_call):
    """Smoke check: the rich interaction-level ``ov.pl.ccc_*`` plots run end-
    to-end on a metabolite-CCC comm-AnnData. We don't assert on the figure
    contents — only that the dispatcher does not raise."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    import omicverse as ov
    from omicverse.single._metabolism import _mebocost_to_comm_adata

    comm = _mebocost_to_comm_adata(_bigger_synthetic_result(), pvalue_threshold=0.05)
    fname, kwargs = plot_call
    func = getattr(ov.pl, fname)
    func(comm, show=False, **kwargs)
    plt.close("all")
