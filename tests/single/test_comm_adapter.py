from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

try:
    import omicverse as ov
except Exception as exc:  # pragma: no cover - environment guard
    ov = None
    pytestmark = pytest.mark.skip(reason=f"omicverse import failed in test env: {exc}")


@pytest.fixture()
def cpdb_results() -> dict[str, pd.DataFrame]:
    means = pd.DataFrame(
        {
            "id_cp_interaction": [1, 2],
            "interacting_pair": ["CXCL13_CXCR5", "TNF_TNFRSF1A"],
            "gene_a": ["CXCL13", "TNF"],
            "gene_b": ["CXCR5", "TNFRSF1A"],
            "classification": ["CXCL", "TNF"],
            "B|T": [0.8, 0.5],
            "T|B": [0.1, 0.3],
        }
    )
    pvalues = pd.DataFrame(
        {
            "id_cp_interaction": [1, 2],
            "interacting_pair": ["CXCL13_CXCR5", "TNF_TNFRSF1A"],
            "gene_a": ["CXCL13", "TNF"],
            "gene_b": ["CXCR5", "TNFRSF1A"],
            "classification": ["CXCL", "TNF"],
            "B|T": [0.01, 0.02],
            "T|B": [0.3, 0.2],
        }
    )
    return {"means": means, "pvalues": pvalues}


@pytest.fixture()
def cpdb_degs_results(cpdb_results: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    means = cpdb_results["means"]
    relevant = means.iloc[[1, 0]].copy()
    relevant["B|T"] = [0, 1]
    relevant["T|B"] = [1, 0]
    return {
        "means": means,
        "relevant_interactions": relevant,
        "significant_means": means.copy(),
    }


def test_format_liana_results_alias_matches_backward_compatible_name() -> None:
    liana_res = pd.DataFrame(
        {
            "source": ["B"],
            "target": ["T"],
            "ligand_complex": ["CXCL13"],
            "receptor_complex": ["CXCR5"],
            "specificity_rank": [0.02],
        }
    )
    new_comm = ov.single.format_liana_results(liana_res=liana_res)
    old_comm = ov.single.format_liana_results_for_viz(liana_res=liana_res)
    assert new_comm.shape == old_comm.shape
    assert list(new_comm.obs.columns) == list(old_comm.obs.columns)
    assert list(new_comm.var.columns) == list(old_comm.var.columns)


def test_format_cpdb_results_builds_comm_adata(cpdb_results: dict[str, pd.DataFrame]) -> None:
    comm_adata = ov.single.format_cpdb_results(cpdb_results)
    assert comm_adata.shape == (2, 2)
    assert list(comm_adata.obs.columns) == ["sender", "receiver"]
    assert "means" in comm_adata.layers
    assert "pvalues" in comm_adata.layers
    assert comm_adata.obs.loc["B|T", "sender"] == "B"
    assert comm_adata.obs.loc["B|T", "receiver"] == "T"
    assert comm_adata.var.iloc[0]["interaction_name"] == "CXCL13_CXCR5"
    assert comm_adata.uns["comm_source"] == "cellphonedb"


def test_to_comm_adata_extracts_cpdb_results_from_uns(cpdb_results: dict[str, pd.DataFrame]) -> None:
    adata = AnnData(X=np.zeros((1, 1), dtype=float))
    adata.uns["cpdb_results"] = cpdb_results
    comm_adata = ov.single.to_comm_adata(adata)
    extracted = ov.single.extract_comm_adata(adata, result_uns_key="cpdb_results")
    assert comm_adata.shape == extracted.shape == (2, 2)
    assert float(comm_adata.layers["means"][0, 0]) == pytest.approx(0.8)


def test_method3_relevance(
    cpdb_degs_results: dict[str, pd.DataFrame],
) -> None:
    comm_adata = ov.single.format_cpdb_results(cpdb_degs_results)

    assert comm_adata.uns["cellphonedb_method"] == "degs"
    assert comm_adata.uns["support_kind"] == "relevance"
    assert not comm_adata.uns["pvalues_are_statistical"]
    assert comm_adata.layers["relevance"].tolist() == [
        [True, False],
        [False, True],
    ]
    assert comm_adata.layers["pvalues"].tolist() == [
        [0.0, 1.0],
        [1.0, 0.0],
    ]


def test_method3_adapter(
    cpdb_degs_results: dict[str, pd.DataFrame],
) -> None:
    comm_adata = ov.single.to_comm_adata(data=cpdb_degs_results)
    assert comm_adata.uns["support_kind"] == "relevance"


def test_method3_filtering(
    cpdb_degs_results: dict[str, pd.DataFrame],
) -> None:
    from omicverse.pl._ccc import _communication_long_table

    comm_adata = ov.single.format_cpdb_results(cpdb_degs_results)
    strict = _communication_long_table(comm_adata, pvalue_threshold=0.0)
    permissive = _communication_long_table(comm_adata, pvalue_threshold=2.0)

    assert len(strict) == len(permissive) == 2
    assert set(strict["pair_lr"]) == {"CXCL13_CXCR5", "TNF_TNFRSF1A"}


def test_method3_rejects_pvalue_color(
    cpdb_degs_results: dict[str, pd.DataFrame],
) -> None:
    comm_adata = ov.single.format_cpdb_results(cpdb_degs_results)
    with pytest.raises(ValueError, match="does not provide statistical p-values"):
        ov.pl.ccc_heatmap(comm_adata, color_by="pvalue", show=False)
