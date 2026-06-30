from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from omicverse.external.cnmf.cnmf import cNMF


def _cnmf_instance():
    cls = getattr(cNMF, "__wrapped__", cNMF)
    return cls.__new__(cls)


def _adata(obs_names):
    return AnnData(
        X=np.ones((len(obs_names), 2)),
        obs=pd.DataFrame(index=list(obs_names)),
        var=pd.DataFrame(index=["g1", "g2"]),
    )


def _result_dict():
    return {
        "usage_norm": pd.DataFrame(
            {
                "cNMF_1": [0.9, 0.1, 0.2],
                "cNMF_2": [0.1, 0.9, 0.8],
            },
            index=["cell_a", "cell_b", "cell_c"],
        ),
        "gep_scores": pd.DataFrame(
            {
                1: [2.0, -1.0],
                2: [-1.0, 2.0],
            },
            index=["g1", "g2"],
        ),
    }


def test_cnmf_get_results_aligns_by_obs_names_when_adata_is_reordered():
    cnmf = _cnmf_instance()
    adata = _adata(["cell_c", "cell_a", "cell_b"])

    cnmf.get_results(adata, _result_dict())

    assert list(adata.obs["cNMF_cluster"]) == ["cNMF_2", "cNMF_1", "cNMF_2"]
    assert list(adata.obs["cNMF_1"]) == [0.2, 0.9, 0.1]


def test_cnmf_get_results_rejects_missing_obs_names_instead_of_silent_nan():
    cnmf = _cnmf_instance()
    adata = _adata(["cell_a", "missing_cell"])

    with pytest.raises(ValueError, match="Missing 1 cell"):
        cnmf.get_results(adata, _result_dict())

    assert "cNMF_cluster" not in adata.obs


def test_cnmf_get_results_can_be_called_repeatedly_on_same_adata():
    cnmf = _cnmf_instance()
    adata = _adata(["cell_a", "cell_b", "cell_c"])
    adata.obs["cNMF_note"] = ["keep", "keep", "keep"]
    adata.obs["cNMF_3"] = [1.0, 1.0, 1.0]
    adata.var[3] = [3.0, 3.0]
    adata.var["gene_note"] = ["keep", "keep"]

    cnmf.get_results(adata, _result_dict())
    cnmf.get_results(adata, _result_dict())

    cNMF_columns = {col for col in adata.obs if col.startswith("cNMF_") and col != "cNMF_cluster"}
    assert cNMF_columns == {"cNMF_1", "cNMF_2", "cNMF_note"}
    assert "cNMF_3" not in adata.obs
    assert list(adata.obs["cNMF_note"]) == ["keep", "keep", "keep"]
    assert list(adata.obs["cNMF_cluster"]) == ["cNMF_1", "cNMF_2", "cNMF_2"]
    assert 3 not in adata.var
    assert list(adata.var["gene_note"]) == ["keep", "keep"]
    assert list(adata.var.columns) == ["gene_note", 1, 2]


def test_cnmf_get_results_rfc_uses_alignment_validation():
    cnmf = _cnmf_instance()
    adata = _adata(["cell_a", "missing_cell"])

    with pytest.raises(ValueError, match="Missing 1 cell"):
        cnmf.get_results_rfc(adata, _result_dict(), use_rep="X_test")

    assert "cNMF_cluster_rfc" not in adata.obs


def test_cnmf_get_results_rfc_rejects_threshold_without_two_classes():
    cnmf = _cnmf_instance()
    adata = _adata(["cell_a", "cell_b", "cell_c"])
    adata.obsm["X_test"] = np.ones((3, 2))

    with pytest.raises(ValueError, match="At least two cNMF factors"):
        cnmf.get_results_rfc(adata, _result_dict(), use_rep="X_test", cNMF_threshold=1.0)
