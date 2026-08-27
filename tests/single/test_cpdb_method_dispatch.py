from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad

from omicverse.single._cpdb import format_cpdb_results, run_cellphonedb_v5


def _degs_results() -> dict[str, pd.DataFrame]:
    means = pd.DataFrame(
        {
            "id_cp_interaction": [1],
            "interacting_pair": ["CXCL13_CXCR5"],
            "gene_a": ["CXCL13"],
            "gene_b": ["CXCR5"],
            "B|T": [0.8],
        }
    )
    relevant = means.copy()
    relevant["B|T"] = 1
    return {
        "means": means,
        "relevant_interactions": relevant,
        "deconvoluted": pd.DataFrame(
            {"complex_name": ["CXCL13_CXCR5", np.nan]}
        ),
        "CellSign_active_interactions": pd.DataFrame,
    }


def _statistical_results() -> dict[str, pd.DataFrame]:
    results = _degs_results()
    results.pop("relevant_interactions")
    results["pvalues"] = results["means"].assign(**{"B|T": [0.01]})
    return results


def _install_fake_cpdb(monkeypatch, *, degs_call, statistical_call) -> None:
    cellphonedb = ModuleType("cellphonedb")
    src = ModuleType("cellphonedb.src")
    core = ModuleType("cellphonedb.src.core")
    methods = ModuleType("cellphonedb.src.core.methods")
    methods.cpdb_degs_analysis_method = SimpleNamespace(call=degs_call)
    methods.cpdb_statistical_analysis_method = SimpleNamespace(call=statistical_call)
    monkeypatch.setitem(sys.modules, "cellphonedb", cellphonedb)
    monkeypatch.setitem(sys.modules, "cellphonedb.src", src)
    monkeypatch.setitem(sys.modules, "cellphonedb.src.core", core)
    monkeypatch.setitem(sys.modules, "cellphonedb.src.core.methods", methods)


def test_method3_dispatch(monkeypatch, tmp_path: Path) -> None:
    received = {}

    def degs_call(**kwargs):
        received.update(kwargs)
        return _degs_results()

    def statistical_call(**kwargs):  # pragma: no cover - failure sentinel
        raise AssertionError("statistical method should not be called")

    _install_fake_cpdb(
        monkeypatch,
        degs_call=degs_call,
        statistical_call=statistical_call,
    )
    monkeypatch.setattr(
        "omicverse.single._cpdb.validate_cpdb_database",
        lambda path: str(path),
    )

    degs_file = tmp_path / "degs.tsv"
    degs_file.write_text("cluster\tgene\nB\tCXCL13\n")
    adata = AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame({"celltype": ["B", "T"]}, index=["b", "t"]),
        var=pd.DataFrame(index=["CXCL13", "CXCR5"]),
    )

    results, comm = run_cellphonedb_v5(
        adata,
        cpdb_file_path=tmp_path / "cellphonedb.zip",
        method=3,
        degs_file_path=degs_file,
        min_genes=0,
        min_cells=0,
        temp_dir=tmp_path / "temp",
        output_dir=tmp_path / "output",
    )

    assert results["means"].shape[0] == 1
    assert comm.uns["cellphonedb_method"] == "degs"
    assert received["degs_file_path"] == str(degs_file)
    assert "iterations" not in received
    assert "pvalue" not in received
    saved = tmp_path / "method3.h5ad"
    adata.write_h5ad(saved)
    restored = read_h5ad(saved)
    restored_comm = format_cpdb_results(restored.uns["cpdb_results"])
    assert "CellSign_active_interactions" not in restored.uns["cpdb_results"]
    assert restored_comm.uns["support_kind"] == "relevance"
    assert restored.uns["cpdb_results"]["deconvoluted"]["complex_name"].tolist() == [
        "CXCL13_CXCR5",
        "",
    ]


def test_method3_requires_degs(tmp_path: Path) -> None:
    adata = AnnData(
        X=np.ones((1, 1)),
        obs=pd.DataFrame({"celltype": ["B"]}, index=["b"]),
        var=pd.DataFrame(index=["CXCL13"]),
    )
    with pytest.raises(ValueError, match="degs_file_path"):
        run_cellphonedb_v5(
            adata,
            cpdb_file_path=tmp_path / "cellphonedb.zip",
            method="degs",
        )


def test_method2_default(monkeypatch, tmp_path: Path) -> None:
    received = {}

    def statistical_call(**kwargs):
        received.update(kwargs)
        return _statistical_results()

    def degs_call(**kwargs):  # pragma: no cover - failure sentinel
        raise AssertionError("DEGs method should not be called")

    _install_fake_cpdb(
        monkeypatch,
        degs_call=degs_call,
        statistical_call=statistical_call,
    )
    monkeypatch.setattr(
        "omicverse.single._cpdb.validate_cpdb_database",
        lambda path: str(path),
    )
    adata = AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame({"celltype": ["B", "T"]}, index=["b", "t"]),
        var=pd.DataFrame(index=["CXCL13", "CXCR5"]),
    )

    _, comm = run_cellphonedb_v5(
        adata,
        cpdb_file_path=tmp_path / "cellphonedb.zip",
        min_genes=0,
        min_cells=0,
        temp_dir=tmp_path / "temp",
        output_dir=tmp_path / "output",
    )

    assert comm.uns["cellphonedb_method"] == "statistical"
    assert received["iterations"] == 1000
    assert received["pvalue"] == pytest.approx(0.05)
    assert "degs_file_path" not in received
