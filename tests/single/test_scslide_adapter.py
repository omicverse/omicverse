"""Adapter contract tests with an explicit fake backend.

These do not certify scSLIDE numerical parity or full OmicVerse import.
A separate real-backend test is supplied and must run before merge.
"""
from __future__ import annotations
import copy
import builtins
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from scipy import sparse


class MiniAnnData:
    def __init__(self):
        self.X = np.ones((12, 4), dtype=float)
        self.obs = pd.DataFrame({"sample": np.repeat(["s0", "s1", "s2", "s3"], 3),
                                 "response": np.repeat([0., 1., 2., 3.], 3),
                                 "subject": np.repeat(["p0", "p1", "p2", "p3"], 3)},
                                index=[f"c{i}" for i in range(12)])
        self.var = pd.DataFrame({"highly_variable": [True] * 4}, index=[f"g{i}" for i in range(4)])
        self.layers = {"counts": self.X.copy(), "data": np.log1p(self.X)}
        self.obsm = {"X_pca": np.arange(24, dtype=float).reshape(12, 2)}
        self.uns = {}
        self.isbacked = False

    @property
    def obs_names(self): return self.obs.index
    @property
    def var_names(self): return self.var.index
    @property
    def n_obs(self): return self.X.shape[0]
    @property
    def n_vars(self): return self.X.shape[1]
    def copy(self): return copy.deepcopy(self)


@pytest.fixture
def adapter(monkeypatch):
    # An isolated namespace avoids importing all of OmicVerse in contract tests.
    for name in ("ov_contract", "ov_contract.single"):
        mod = ModuleType(name)
        mod.__path__ = []
        monkeypatch.setitem(sys.modules, name, mod)
    reg = ModuleType("ov_contract._registry")

    def register_function(**values):
        def decorate(func):
            func.registration = values
            return func
        return decorate
    reg.register_function = register_function
    monkeypatch.setitem(sys.modules, "ov_contract._registry", reg)
    ann = ModuleType("anndata")
    ann.AnnData = MiniAnnData
    monkeypatch.setitem(sys.modules, "anndata", ann)
    path = Path(__file__).resolve().parents[2] / "omicverse/single/_scslide.py"
    spec = importlib.util.spec_from_file_location("ov_contract.single._scslide", path)
    module = importlib.util.module_from_spec(spec)
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "scslide" or name.startswith("scslide."):
            raise AssertionError("Optional backend imported during module import")
        return real_import(name, *args, **kwargs)
    with monkeypatch.context() as imports:
        imports.setattr(builtins, "__import__", guard)
        spec.loader.exec_module(module)
    return module


@pytest.fixture
def backend(adapter, monkeypatch):
    calls = []

    def run(working, **kwargs):
        calls.append((working, kwargs))
        ids = pd.Index(["s3", "s2", "s1", "s0"])
        sample = SimpleNamespace(obs_names=ids, obsm={}, obs=pd.DataFrame(index=ids), uns={})
        dims = pd.DataFrame(np.arange(8).reshape(4, 2), index=["s0", "s1", "s2", "s3"])
        return SimpleNamespace(sample_adata=sample, diffusion=SimpleNamespace(embeddings=dims),
                               principal_curve=SimpleNamespace(lambda_=np.arange(4.0)),
                               prepared=SimpleNamespace(adata=working))
    monkeypatch.setattr(adapter, "_require_backend", lambda: (SimpleNamespace(run_scslide_workflow=run), "0.3.3"))
    return calls


def call(adapter, data=None, **kwargs):
    options = {"sample_key": "sample", "response_key": "response", "subject_key": "subject",
               "workflow_kwargs": {"ncomp_diffusion": 2}}
    options.update(kwargs)
    return adapter.scslide(MiniAnnData() if data is None else data, **options)


def test_public_contract_preserves_input_and_aligns_samples(adapter, backend):
    data = MiniAnnData()
    before = copy.deepcopy(data.__dict__)
    result = call(adapter, data)
    assert backend[0][0] is not data
    assert data.uns == before["uns"]
    np.testing.assert_array_equal(data.layers["counts"], before["layers"]["counts"])
    assert result.sample_adata.uns["omicverse_scslide"]["observation_unit"] == "sample"
    np.testing.assert_array_equal(result.sample_adata.obsm["X_scslide_diffmap"][0], [6, 7])
    assert result.sample_adata.obs["scslide_trajectory"].tolist() == [3., 2., 1., 0.]
    assert backend[0][1]["de_inference_mode"] == "descriptive_r_compat"


@pytest.mark.parametrize("format", ["csr", "csc"])
def test_sparse_inputs_are_not_densified_by_adapter(adapter, backend, format):
    data = MiniAnnData()
    constructor = sparse.csr_matrix if format == "csr" else sparse.csc_matrix
    data.layers = {k: constructor(v) for k, v in data.layers.items()}
    call(adapter, data)
    assert sparse.issparse(backend[0][0].layers["counts"])
    assert sparse.issparse(backend[0][0].layers["data"])


def test_layer_aliases_capture_before_mapping(adapter, backend):
    data = MiniAnnData()
    data.layers = {"raw": data.X.copy(), "lognorm": np.log1p(data.X)}
    call(adapter, data, counts_layer="raw", data_layer="lognorm")
    np.testing.assert_array_equal(backend[0][0].layers["counts"], data.X)
    assert "counts" not in data.layers


def test_counts_from_x(adapter, backend):
    data = MiniAnnData()
    del data.layers["counts"]
    call(adapter, data, counts_layer=None)
    np.testing.assert_array_equal(backend[0][0].layers["counts"], data.X)


@pytest.mark.parametrize("mutation,pattern", [
    (lambda a: a.layers.pop("counts"), "no silent X fallback"),
    (lambda a: a.layers.pop("data"), "Normalized data layer"),
    (lambda a: a.obs.__setitem__("response", np.nan), "missing values"),
    (lambda a: a.obs.__setitem__("response", 1), "two distinct"),
    (lambda a: a.var.__setitem__("highly_variable", [1, 1, 1, 1]), "boolean"),
    (lambda a: a.var.__setitem__("highly_variable", False), "boolean"),
    (lambda a: a.obsm.pop("X_pca"), "Missing cell embedding"),
    (lambda a: setattr(a, "isbacked", True), "Backed"),
    (lambda a: a.layers.__setitem__("counts", a.X + 0.5), "integer-valued"),
    (lambda a: a.layers.__setitem__("counts", a.X * -1), "nonnegative"),
    (lambda a: a.layers.__setitem__("data", a.X * np.inf), "finite"),
])
def test_bad_inputs_fail_before_backend(adapter, backend, mutation, pattern):
    data = MiniAnnData()
    mutation(data)
    with pytest.raises((ValueError, TypeError), match=pattern):
        call(adapter, data)
    assert not backend


def test_varying_response_in_sample(adapter, backend):
    data = MiniAnnData()
    data.obs.loc["c0", "response"] = 8.
    with pytest.raises(ValueError, match="constant within"):
        call(adapter, data)


def test_duplicate_feature_ids(adapter, backend):
    data = MiniAnnData()
    data.var.index = ["g", "g", "g2", "g3"]
    with pytest.raises(ValueError, match="var_names"):
        call(adapter, data)


@pytest.mark.parametrize(
    "key,value",
    [("de_inference_mode", "confirmatory_oof_countsplit"),
     ("sample_col", "other"), ("n_jobs", 99), ("seed", 8)],
)
def test_reserved_native_overrides_rejected(adapter, backend, key, value):
    with pytest.raises(ValueError, match="reserved"):
        call(adapter, workflow_kwargs={"ncomp_diffusion": 2, key: value})


@pytest.mark.parametrize("budget", [0, -2, True, 1.5])
def test_invalid_workers(adapter, backend, budget):
    with pytest.raises(ValueError, match="n_jobs"):
        call(adapter, n_jobs=budget)


def test_backend_errors_are_not_hidden(adapter, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("original fit failure")
    monkeypatch.setattr(adapter, "_require_backend", lambda: (SimpleNamespace(run_scslide_workflow=fail), "0.3.3"))
    with pytest.raises(RuntimeError, match="original fit failure"):
        call(adapter)


def test_unrelated_missing_dependency_propagates(adapter, monkeypatch):
    def fail(name):
        raise ModuleNotFoundError("broken transitive dependency", name="unrelated_dependency")
    monkeypatch.setattr(adapter.importlib, "import_module", fail)
    with pytest.raises(ModuleNotFoundError, match="transitive"):
        adapter._require_backend()


def test_missing_backend_has_actionable_error(adapter, monkeypatch):
    def fail(name):
        raise ModuleNotFoundError("no scslide", name="scslide")
    monkeypatch.setattr(adapter.importlib, "import_module", fail)
    with pytest.raises(ImportError, match="approved release artifact"):
        adapter._require_backend()


def test_returned_wrong_sample_axis_rejected(adapter, monkeypatch):
    sample = SimpleNamespace(
        obs_names=pd.Index(["wrong"]), obsm={},
        obs=pd.DataFrame(index=["wrong"]), uns={},
    )
    wrong = SimpleNamespace(
        sample_adata=sample,
        diffusion=SimpleNamespace(embeddings=pd.DataFrame([[1.]], index=["s0"])),
        principal_curve=None,
    )
    monkeypatch.setattr(
        adapter, "_require_backend",
        lambda: (SimpleNamespace(run_scslide_workflow=lambda *a, **k: wrong), "0.3.3"),
    )
    with pytest.raises(RuntimeError, match="IDs disagree"):
        call(adapter)


@pytest.mark.parametrize("covariates,pattern", [(None, "explicitly"), (["Age"], "missing"), (["varying"], "constant")])
def test_descriptive_de_covariates_are_not_silently_changed(adapter, backend, covariates, pattern):
    data = MiniAnnData()
    data.obs["celltype"] = "T"
    data.obs["varying"] = np.arange(12)
    options = {"ncomp_diffusion": 2, "de_celltype": "T"}
    if covariates is not None:
        options["de_covariates"] = covariates
    with pytest.raises(ValueError, match=pattern):
        call(adapter, data, celltype_key="celltype", workflow_kwargs=options)
    assert not backend


def test_descriptive_de_explicit_empty_covariates(adapter, backend):
    data = MiniAnnData()
    data.obs["celltype"] = "T"
    call(adapter, data, celltype_key="celltype",
         workflow_kwargs={"ncomp_diffusion": 2, "de_celltype": "T", "de_covariates": []})
    assert backend[0][1]["de_covariates"] == []


def test_optional_backend_is_not_imported_on_module_loading(adapter):
    assert adapter.scslide.registration["category"] == "single"
    assert "sample-level" in adapter.scslide.registration["description"]
