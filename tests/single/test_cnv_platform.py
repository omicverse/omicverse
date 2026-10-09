"""ov.single.CNV(method='infercnv') platform / cutoff guard.

The gene-filter cutoff is platform-dependent (R inferCNV: 10x->0.1,
SmartSeq2->1.0). The wrong default silently over-/under-filters genes, so
``run()`` requires an explicit ``platform`` or ``cutoff``. A recording stub
exercises the real dispatcher without installing or running pyinfercnv.
These tests cover configuration/metadata forwarding, not CNV inference accuracy.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ad = pytest.importorskip("anndata")


@pytest.fixture
def cnv_class(monkeypatch):
    # Load the real dispatcher without importing unrelated optional single-cell
    # tools. Keep decorator registration isolated from the rest of the suite.
    from omicverse import _registry

    monkeypatch.setattr(_registry, "_global_registry", _registry.FunctionRegistry())
    path = Path(__file__).resolve().parents[2] / "omicverse" / "single" / "_cnv.py"
    name = "omicverse.single._cnv_platform_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module.CNV


@pytest.fixture(autouse=True)
def infercnv_backend(monkeypatch):
    """Record the backend boundary and supply the minimal write-back contract."""
    calls = []
    configurations = []
    result = object()

    class InferCNVConfig:
        def __init__(self, **kwargs):
            self.options = kwargs
            configurations.append(self)

    def infercnv(adata, **kwargs):
        calls.append((adata, kwargs))
        # The real backend writes these outputs when inplace=True. Distinct
        # signed values also exercise the dispatcher's score postprocessing.
        adata.obsm["X_cnv"] = np.tile([-2.0, 1.0], (adata.n_obs, 1))
        adata.uns["cnv"] = {"chr_pos": {"chr1": 0, "chr2": 1}}
        return result

    backend = types.ModuleType("pyinfercnv")
    backend.InferCNVConfig = InferCNVConfig
    backend.infercnv = infercnv
    monkeypatch.setitem(sys.modules, "pyinfercnv", backend)
    return types.SimpleNamespace(
        calls=calls, configurations=configurations, result=result,
    )


def _tiny_adata():
    rng = np.random.RandomState(0)
    X = rng.poisson(1.0, size=(20, 12)).astype("float32")
    var = pd.DataFrame(
        {"chromosome": ["chr1"] * 12, "start": range(12), "end": range(1, 13)},
        index=[f"g{i}" for i in range(12)],
    )
    obs = pd.DataFrame(
        {"cell_type": ["Macrophage"] * 10 + ["Tumor"] * 10},
        index=[f"c{i}" for i in range(20)],
    )
    return ad.AnnData(X=X, obs=obs, var=var)


def test_infercnv_requires_platform_or_cutoff(infercnv_backend, cnv_class):
    cnv = cnv_class(_tiny_adata(), method="infercnv")
    with pytest.raises(ValueError, match="platform"):
        cnv.run(reference_key="cell_type", reference_cat=["Macrophage"])
    assert not infercnv_backend.calls
    assert not infercnv_backend.configurations


def test_infercnv_rejects_unknown_platform(infercnv_backend, cnv_class):
    cnv = cnv_class(_tiny_adata(), method="infercnv")
    with pytest.raises(ValueError, match="unknown platform"):
        cnv.run(reference_key="cell_type", reference_cat=["Macrophage"], platform="nanopore")
    assert not infercnv_backend.calls
    assert not infercnv_backend.configurations


def test_infercnv_rejects_platform_and_cutoff_together(infercnv_backend, cnv_class):
    cnv = cnv_class(_tiny_adata(), method="infercnv")
    with pytest.raises(ValueError, match="not both"):
        cnv.run(
            reference_key="cell_type", reference_cat=["Macrophage"],
            platform="10x", cutoff=0.1,
        )
    assert not infercnv_backend.calls
    assert not infercnv_backend.configurations


@pytest.mark.parametrize(
    "platform, expected_cutoff",
    [
        ("10x", 0.1), ("10X", 0.1), ("tenx", 0.1), ("TENX", 0.1),
        ("smartseq2", 1.0), ("SmartSeq2", 1.0), ("smartseq", 1.0),
        ("ss2", 1.0), ("SS2", 1.0), ("Smart-Seq2", 1.0),
        ("smart_seq2", 1.0), ("Smart Seq2", 1.0),
    ],
)
def test_platform_cutoff_reaches_backend(infercnv_backend, cnv_class, platform, expected_cutoff):
    adata = _tiny_adata()
    cnv = cnv_class(adata, method="infercnv")

    returned = cnv.run(
        reference_key="cell_type", reference_cat=["Macrophage"], platform=platform,
    )

    assert returned is cnv
    assert cnv.result is infercnv_backend.result
    assert len(infercnv_backend.calls) == 1
    actual_adata, options = infercnv_backend.calls[0]
    assert actual_adata is adata
    assert options["config"].options == {"counts_layer": None, "cutoff": expected_cutoff}
    assert adata.uns["cnv"] == {
        "method": "infercnv", "chr_pos": {"chr1": 0, "chr2": 1},
    }
    np.testing.assert_array_equal(adata.obs["cnv_score"], np.full(adata.n_obs, 1.5))
    assert adata.obs["cnv_prediction"].isna().all()


@pytest.mark.parametrize("cutoff", [0.0, 0.25, 2.0])
def test_explicit_cutoff_reaches_backend_without_platform(infercnv_backend, cnv_class, cutoff):
    cnv_class(_tiny_adata(), method="infercnv").run(cutoff=cutoff)

    assert len(infercnv_backend.calls) == 1
    config = infercnv_backend.calls[0][1]["config"]
    assert config.options == {"counts_layer": None, "cutoff": cutoff}


@pytest.mark.parametrize("missing_column", ["chromosome", "start", "end"])
def test_missing_gene_coordinates_fail_before_backend(infercnv_backend, cnv_class, missing_column):
    adata = _tiny_adata()
    del adata.var[missing_column]
    cnv = cnv_class(adata, method="infercnv")

    with pytest.raises(ValueError, match=repr(missing_column)):
        cnv.run(platform="10x")

    assert not infercnv_backend.calls
    assert not infercnv_backend.configurations


@pytest.mark.parametrize("reference_cat", ["Macrophage", ["Macrophage"], ("Macrophage", "Tumor"), None])
@pytest.mark.parametrize("exclude_chromosomes", [("chrX", "chrM"), None])
def test_references_coordinates_layer_and_options_are_forwarded(
    infercnv_backend, cnv_class, reference_cat, exclude_chromosomes,
):
    adata = _tiny_adata()
    adata.layers["counts"] = adata.X.copy()
    original_var = adata.var.copy(deep=True)
    original_obs = adata.obs.copy(deep=True)
    cnv = cnv_class(adata, method="infercnv", layer="counts")

    cnv.run(
        reference_key="cell_type", reference_cat=reference_cat,
        exclude_chromosomes=exclude_chromosomes, platform="10x",
        denoise=True, HMM=False, window_length=7,
    )

    assert len(infercnv_backend.calls) == 1
    actual_adata, options = infercnv_backend.calls[0]
    assert actual_adata is adata
    pd.testing.assert_frame_equal(actual_adata.var, original_var)
    pd.testing.assert_frame_equal(actual_adata.obs[original_obs.columns], original_obs)
    config = options.pop("config")
    assert config.options == {
        "counts_layer": "counts", "cutoff": 0.1, "denoise": True,
        "HMM": False, "window_length": 7,
    }
    assert options == {
        "reference_key": "cell_type", "reference_cat": reference_cat,
        "exclude_chromosomes": exclude_chromosomes, "key_added": "cnv", "inplace": True,
    }


def test_default_reference_and_chromosome_options_reach_backend(infercnv_backend, cnv_class):
    cnv_class(_tiny_adata(), method="infercnv").run(platform="10x")

    options = infercnv_backend.calls[0][1]
    assert options["reference_key"] is None
    assert options["reference_cat"] is None
    assert options["exclude_chromosomes"] == ("chrX", "chrY", "chrM")
