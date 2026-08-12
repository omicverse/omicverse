from __future__ import annotations

import builtins
import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


def _load_leaf_module(mod_path):
    from omicverse.mcp.manifest import _try_load_leaf_module

    assert _try_load_leaf_module(mod_path)
    return sys.modules[mod_path]


class _FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)
        self.shape = self.array.shape

    @property
    def T(self):
        return _FakeTensor(self.array.T)


def _install_fake_regvelo(monkeypatch):
    calls = {
        "setup_anndata": [],
        "init": [],
        "train": [],
        "save": [],
        "load": [],
        "set_output": [],
        "perturb": [],
        "perturbation_effect": [],
        "cellfate_perturbation": [],
        "preprocess_data": [],
        "set_prior_grn": [],
    }

    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = np.float32
    fake_torch.is_tensor = lambda value: isinstance(value, _FakeTensor)
    fake_torch.as_tensor = lambda value, **kwargs: _FakeTensor(value)
    fake_torch.tensor = lambda value, **kwargs: _FakeTensor(value)

    class FakeREGVELOVI:
        @classmethod
        def load(cls, path, adata=None, **kwargs):
            model = cls.__new__(cls)
            model.adata = adata
            model.W = None
            model.regulators = None
            model.kwargs = {}
            model.loaded_from = path
            calls["load"].append({"path": path, "adata": adata, "kwargs": kwargs})
            return model

        @classmethod
        def setup_anndata(cls, adata, spliced_layer, unspliced_layer):
            calls["setup_anndata"].append(
                {
                    "adata": adata,
                    "spliced_layer": spliced_layer,
                    "unspliced_layer": unspliced_layer,
                }
            )

        def __init__(self, adata, W, regulators=None, **kwargs):
            self.adata = adata
            self.W = W
            self.regulators = regulators
            self.kwargs = kwargs
            calls["init"].append(
                {
                    "adata": adata,
                    "W": W,
                    "regulators": regulators,
                    "kwargs": kwargs,
                }
            )

        def train(self, **kwargs):
            calls["train"].append(kwargs)

        def save(self, path, **kwargs):
            calls["save"].append({"path": path, "kwargs": kwargs})

    def fake_set_output(adata, model, n_samples, batch_size):
        calls["set_output"].append(
            {
                "adata": adata,
                "model": model,
                "n_samples": n_samples,
                "batch_size": batch_size,
            }
        )
        adata.layers["velocity"] = np.full(adata.shape, 2.0, dtype=np.float32)

    def fake_perturb(**kwargs):
        calls["perturb"].append(kwargs)
        return kwargs["adata"].copy(), kwargs["model"]

    def fake_perturbation_effect(adata_perturb, adata, terminal_state, **kwargs):
        calls["perturbation_effect"].append(
            {
                "adata_perturb": adata_perturb,
                "adata": adata,
                "terminal_state": terminal_state,
                "kwargs": kwargs,
            }
        )
        states = [terminal_state] if isinstance(terminal_state, str) else list(terminal_state)
        for i, state in enumerate(states):
            adata.obs[f"perturbation effect on {state}"] = np.full(adata.n_obs, i + 0.25)
        return adata

    def fake_cellfate_perturbation(**kwargs):
        calls["cellfate_perturbation"].append(kwargs)
        return pd.DataFrame(
            {
                "candidate": list(kwargs["perturbed"].keys()),
                "score": np.arange(len(kwargs["perturbed"]), dtype=float),
            }
        )

    fake_regvelo = types.ModuleType("regvelo")
    fake_regvelo.REGVELOVI = FakeREGVELOVI
    fake_regvelo.tl = types.SimpleNamespace(
        set_output=fake_set_output,
        in_silico_block_simulation=fake_perturb,
    )
    fake_regvelo.tools = types.SimpleNamespace(
        perturbation_effect=fake_perturbation_effect,
    )
    fake_regvelo.metrics = types.SimpleNamespace(
        cellfate_perturbation=fake_cellfate_perturbation,
    )
    fake_regvelo.pp = types.SimpleNamespace(
        preprocess_data=lambda adata, **kwargs: calls["preprocess_data"].append(kwargs) or adata,
        set_prior_grn=lambda adata, prior, **kwargs: (
            calls["set_prior_grn"].append({"prior": prior, "kwargs": kwargs})
            or adata.uns.__setitem__("skeleton", prior)
            or adata
        ),
    )
    fake_regvelo.datasets = types.SimpleNamespace(
        zebrafish_nc=_make_velocity_adata,
        zebrafish_grn=lambda: pd.DataFrame(
            np.eye(3, dtype=np.float32),
            index=["TF1", "Gene2", "TF3"],
            columns=["TF1", "Gene2", "TF3"],
        ),
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "regvelo", fake_regvelo)
    return calls


def _make_velocity_adata():
    adata = AnnData(X=np.ones((4, 3), dtype=np.float32))
    adata.var_names = ["TF1", "Gene2", "TF3"]
    adata.layers["Ms"] = np.ones((4, 3), dtype=np.float32)
    adata.layers["Mu"] = np.ones((4, 3), dtype=np.float32) * 0.5
    adata.uns["skeleton"] = np.eye(3, dtype=np.float32)
    adata.var["is_tf"] = [True, False, True]
    return adata


def test_cal_velocity_regvelo_trains_and_exports_outputs(monkeypatch):
    calls = _install_fake_regvelo(monkeypatch)
    Velo = _load_leaf_module("omicverse.single._velo").Velo

    adata = _make_velocity_adata()
    velo = Velo(adata)

    velo.cal_velocity(
        method="regvelo",
        velocity_key="velocity_regvelo",
        n_samples=7,
        batch_size=3,
        model_save_path="tmp/regvelo_model",
        regvelo_kwargs={"soft_constraint": False},
        train_kwargs={"max_epochs": 1},
    )

    assert calls["setup_anndata"] == [
        {"adata": adata, "spliced_layer": "Ms", "unspliced_layer": "Mu"}
    ]
    assert calls["init"][0]["regulators"] == ["TF1", "TF3"]
    assert calls["init"][0]["kwargs"] == {"soft_constraint": False}
    assert calls["init"][0]["W"].shape == (3, 3)
    assert calls["train"] == [{"max_epochs": 1}]
    assert calls["save"] == [{"path": "tmp/regvelo_model", "kwargs": {"overwrite": False}}]
    assert calls["set_output"][0]["n_samples"] == 7
    assert calls["set_output"][0]["batch_size"] == 3
    assert np.all(adata.layers["velocity_regvelo"] == 2.0)
    assert adata.var["velocity_regvelo_genes"].tolist() == [True, False, True]
    assert adata.uns["regvelo_model_path"] == "tmp/regvelo_model"
    assert adata.uns["regvelo"]["velocity_key"] == "velocity_regvelo"
    assert adata.uns["regvelo"]["model_overwrite"] is False


def test_cal_velocity_regvelo_overwrites_existing_model_path(monkeypatch):
    calls = _install_fake_regvelo(monkeypatch)
    Velo = _load_leaf_module("omicverse.single._velo").Velo

    adata = _make_velocity_adata()
    velo = Velo(adata)

    velo.cal_velocity(
        method="regvelo",
        model_save_path="tmp/regvelo_model",
        model_overwrite=True,
        train_kwargs={"max_epochs": 1},
    )

    assert calls["save"] == [{"path": "tmp/regvelo_model", "kwargs": {"overwrite": True}}]
    assert adata.uns["regvelo"]["model_overwrite"] is True


def test_cal_velocity_regvelo_loads_existing_model_without_training(monkeypatch):
    calls = _install_fake_regvelo(monkeypatch)
    Velo = _load_leaf_module("omicverse.single._velo").Velo

    adata = _make_velocity_adata()
    velo = Velo(adata)

    velo.cal_velocity(
        method="regvelo",
        velocity_key="velocity_loaded",
        model_load_path="tmp/regvelo_model",
        train_kwargs={"accelerator": "cpu", "devices": 1, "max_epochs": 99},
    )

    assert calls["setup_anndata"] == []
    assert calls["init"] == []
    assert calls["train"] == []
    assert calls["load"] == [
        {"path": "tmp/regvelo_model", "adata": adata, "kwargs": {"accelerator": "cpu", "device": 1}}
    ]
    assert calls["set_output"][0]["model"].loaded_from == "tmp/regvelo_model"
    assert np.all(adata.layers["velocity_loaded"] == 2.0)
    assert adata.uns["regvelo_model_path"] == "tmp/regvelo_model"
    assert adata.uns["regvelo"]["model_load_path"] == "tmp/regvelo_model"


def test_cal_velocity_regvelo_can_reuse_existing_output(monkeypatch):
    calls = _install_fake_regvelo(monkeypatch)
    Velo = _load_leaf_module("omicverse.single._velo").Velo

    adata = _make_velocity_adata()
    existing_velocity = np.full(adata.shape, 5.0, dtype=np.float32)
    adata.layers["velocity_loaded"] = existing_velocity
    velo = Velo(adata)

    velo.cal_velocity(
        method="regvelo",
        velocity_key="velocity_loaded",
        model_load_path="tmp/regvelo_model",
        reuse_regvelo_output=True,
    )

    assert calls["load"][0]["path"] == "tmp/regvelo_model"
    assert calls["set_output"] == []
    assert np.all(adata.layers["velocity"] == existing_velocity)
    assert np.all(adata.layers["velocity_loaded"] == existing_velocity)
    assert adata.uns["regvelo"]["reused_regvelo_output"] is True


def test_cal_velocity_regvelo_can_run_downstream_projection(monkeypatch):
    _install_fake_regvelo(monkeypatch)
    Velo = _load_leaf_module("omicverse.single._velo").Velo

    downstream_calls = {"graph": [], "embedding": []}

    def fake_graph(self, **kwargs):
        downstream_calls["graph"].append(kwargs)

    def fake_embedding(self, **kwargs):
        downstream_calls["embedding"].append(kwargs)

    monkeypatch.setattr(Velo, "velocity_graph", fake_graph)
    monkeypatch.setattr(Velo, "velocity_embedding", fake_embedding)

    adata = _make_velocity_adata()
    velo = Velo(adata)
    monkeypatch.setattr(velo, "velocity_graph", fake_graph.__get__(velo, Velo))
    monkeypatch.setattr(velo, "velocity_embedding", fake_embedding.__get__(velo, Velo))
    velo.cal_velocity(
        method="regvelo",
        velocity_key="velocity_regvelo",
        compute_velocity_graph=True,
        compute_velocity_embedding=True,
        basis="umap",
        graph_kwargs={"n_jobs": 2},
    )

    assert downstream_calls["graph"] == [
        {"basis": "umap", "vkey": "velocity_regvelo", "n_jobs": 2, "xkey": "Ms"}
    ]
    assert downstream_calls["embedding"] == [
        {"basis": "umap", "vkey": "velocity_regvelo"}
    ]


def test_prepare_regvelo_aligns_prior_and_regulators(monkeypatch):
    calls = _install_fake_regvelo(monkeypatch)
    Velo = _load_leaf_module("omicverse.single._velo").Velo

    adata = _make_velocity_adata()
    velo = Velo(adata)

    moment_calls = []

    def fake_moments(**kwargs):
        moment_calls.append(kwargs)

    monkeypatch.setattr(velo, "moments", fake_moments)

    prior_values = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=np.float32,
    )
    prior = pd.DataFrame(
        prior_values,
        index=["TF1", "Gene2", "TF3"],
        columns=["TF1", "Gene2", "TF3"],
    )
    skeleton, regulators = velo.prepare_regvelo(
        prior,
        use_ov_neighbors=False,
        n_neighbors=5,
        n_pcs=2,
        preprocess_kwargs={"foo": "bar"},
        set_prior_kwargs={"baz": "qux"},
    )

    assert moment_calls == [{"backend": "scvelo", "n_neighbors": 5, "n_pcs": 2}]
    assert calls["preprocess_data"] == [{"foo": "bar"}]
    assert calls["set_prior_grn"][0]["kwargs"] == {"baz": "qux"}
    assert calls["set_prior_grn"][0]["prior"].equals(prior.T)
    assert skeleton is adata.uns["skeleton"]
    assert regulators == ["TF1", "TF3"]
    assert adata.var["is_tf"].tolist() == [True, False, True]
    assert adata.uns["regvelo_prepare"]["n_regulators"] == 2
    assert adata.uns["regvelo_regulators"] == ["TF1", "TF3"]


def test_prepare_regvelo_accepts_scenic_style_edgelist(monkeypatch):
    calls = _install_fake_regvelo(monkeypatch)
    Velo = _load_leaf_module("omicverse.single._velo").Velo

    adata = _make_velocity_adata()
    velo = Velo(adata)
    monkeypatch.setattr(velo, "moments", lambda **kwargs: None)

    scenic_edges = pd.DataFrame(
        {
            "TF": ["TF1", "TF3", "missing_tf"],
            "target": ["Gene2", "TF1", "Gene2"],
            "importance": [0.7, 0.4, 1.0],
        }
    )

    skeleton, regulators = velo.prepare_regvelo(
        scenic_edges,
        use_ov_neighbors=False,
        prior_orientation="regulator_by_target",
    )

    prior_for_regvelo = calls["set_prior_grn"][0]["prior"]
    assert prior_for_regvelo.loc["Gene2", "TF1"] == pytest.approx(0.7)
    assert prior_for_regvelo.loc["TF1", "TF3"] == pytest.approx(0.4)
    assert prior_for_regvelo.loc["Gene2"].sum() == pytest.approx(0.7)
    assert skeleton is adata.uns["skeleton"]
    assert regulators == ["TF1", "TF3"]


def test_regvelo_perturb_uses_saved_model_path(monkeypatch):
    calls = _install_fake_regvelo(monkeypatch)
    Velo = _load_leaf_module("omicverse.single._velo").Velo

    adata = _make_velocity_adata()
    adata.uns["regvelo_model_path"] = "tmp/regvelo_model"
    velo = Velo(adata)

    perturbed, model = velo.regvelo_perturb("TF1", cutoff=0.0)

    assert perturbed.shape == adata.shape
    assert model == "tmp/regvelo_model"
    assert calls["perturb"][0]["TF"] == "TF1"
    assert calls["perturb"][0]["model"] == "tmp/regvelo_model"
    assert calls["perturb"][0]["batch_size"] == adata.n_obs
    assert adata.uns["regvelo_perturbations"]["TF1"]["cutoff"] == 0.0


def test_regvelo_perturb_saves_in_memory_model_for_upstream(monkeypatch):
    calls = _install_fake_regvelo(monkeypatch)
    Velo = _load_leaf_module("omicverse.single._velo").Velo

    class MemoryModel:
        def save(self, path, **kwargs):
            calls["save"].append({"path": path, "kwargs": kwargs})

    adata = _make_velocity_adata()
    adata.uns["regvelo_model"] = MemoryModel()
    velo = Velo(adata)

    perturbed, model = velo.regvelo_perturb("TF1", cutoff=0.0)

    assert perturbed.shape == adata.shape
    assert isinstance(model, str)
    assert "ov_regvelo_perturb_" in model
    assert calls["save"][0]["kwargs"] == {"overwrite": True}
    assert calls["perturb"][0]["model"] == model


def test_perturbation_effect_prefers_regvelo_tools(monkeypatch):
    calls = _install_fake_regvelo(monkeypatch)
    Velo = _load_leaf_module("omicverse.single._velo").Velo

    adata = _make_velocity_adata()
    perturbed = _make_velocity_adata()
    velo = Velo(adata)

    result = velo.perturbation_effect(
        perturbed,
        terminal_states=["state_a", "state_b"],
        extra_arg=True,
    )

    assert result is adata
    assert calls["perturbation_effect"][0]["adata_perturb"] is perturbed
    assert calls["perturbation_effect"][0]["adata"] is adata
    assert calls["perturbation_effect"][0]["terminal_state"] == ["state_a", "state_b"]
    assert calls["perturbation_effect"][0]["kwargs"] == {"extra_arg": True}
    assert adata.obs["perturbation effect on state_a"].tolist() == [0.25] * adata.n_obs
    assert adata.obs["perturbation effect on state_b"].tolist() == [1.25] * adata.n_obs


def test_perturbation_effect_falls_back_to_regvelo_tl_and_custom_prefix(monkeypatch):
    _install_fake_regvelo(monkeypatch)
    fake_regvelo = sys.modules["regvelo"]
    fake_regvelo.tools.perturbation_effect = None

    fallback_calls = []

    def fallback_effect(adata_perturb, adata, terminal_state, **kwargs):
        fallback_calls.append(terminal_state)
        adata.obs[f"perturbation effect on {terminal_state}"] = np.arange(adata.n_obs)
        return adata

    fake_regvelo.tl.perturbation_effect = fallback_effect
    Velo = _load_leaf_module("omicverse.single._velo").Velo

    adata = _make_velocity_adata()
    perturbed = _make_velocity_adata()
    velo = Velo(adata)

    result = velo.perturbation_effect(
        perturbed,
        terminal_states="state_a",
        key_prefix="effect on ",
    )

    assert result is adata
    assert fallback_calls == ["state_a"]
    assert "perturbation effect on state_a" not in adata.obs
    assert adata.obs["effect on state_a"].tolist() == [0, 1, 2, 3]


def test_cell_fate_perturbation_prefers_regvelo_metrics(monkeypatch):
    calls = _install_fake_regvelo(monkeypatch)
    Velo = _load_leaf_module("omicverse.single._velo").Velo

    adata = _make_velocity_adata()
    perturbed = {"TF1": _make_velocity_adata()}
    velo = Velo(adata)

    result = velo.cell_fate_perturbation(
        perturbed,
        terminal_states=["state_a"],
        score_method="t-statistics",
        solver="direct",
        terminal_indices={"state_a": ["cell0"]},
    )

    assert result.equals(pd.DataFrame({"candidate": ["TF1"], "score": [0.0]}))
    assert adata.uns["cell_fate_perturbation"] is result
    assert calls["cellfate_perturbation"][0]["perturbed"] is perturbed
    assert calls["cellfate_perturbation"][0]["baseline"] is adata
    assert calls["cellfate_perturbation"][0]["terminal_state"] == ["state_a"]
    assert calls["cellfate_perturbation"][0]["method"] == "t-statistics"
    assert calls["cellfate_perturbation"][0]["solver"] == "direct"
    assert calls["cellfate_perturbation"][0]["terminal_indices"] == {"state_a": ["cell0"]}


def test_cell_fate_perturbation_falls_back_to_regvelo_mt(monkeypatch):
    _install_fake_regvelo(monkeypatch)
    fake_regvelo = sys.modules["regvelo"]
    fake_regvelo.metrics = None

    fallback_calls = []

    def fallback_cellfate_perturbation(**kwargs):
        fallback_calls.append(kwargs)
        return pd.DataFrame({"candidate": ["fallback"], "score": [1.0]})

    fake_regvelo.mt = types.SimpleNamespace(cellfate_perturbation=fallback_cellfate_perturbation)
    Velo = _load_leaf_module("omicverse.single._velo").Velo

    adata = _make_velocity_adata()
    velo = Velo(adata)
    result = velo.cell_fate_perturbation(_make_velocity_adata(), terminal_states="state_a")

    assert result.equals(pd.DataFrame({"candidate": ["fallback"], "score": [1.0]}))
    assert list(fallback_calls[0]["perturbed"].keys()) == ["perturbation"]
    assert fallback_calls[0]["terminal_state"] == "state_a"


def test_velocity_effect_writes_obs_and_handles_zero_norms(monkeypatch):
    _install_fake_regvelo(monkeypatch)
    Velo = _load_leaf_module("omicverse.single._velo").Velo

    adata = _make_velocity_adata()
    perturbed = _make_velocity_adata()
    adata.layers["velo_regvelo"] = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    perturbed.layers["velocity"] = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    velo = Velo(adata)

    result = velo.velocity_effect(perturbed, target="TF1")

    assert result.name == "TF1_velocity_effect"
    assert np.all(np.isfinite(result))
    assert result.tolist() == pytest.approx([0.0, 2.0, 1.0, 1 - 1 / np.sqrt(2)])
    assert adata.obs["TF1_velocity_effect"].equals(result)


def test_velocity_effect_aligns_cells_and_genes_by_name(monkeypatch):
    from scipy.sparse import csr_matrix

    _install_fake_regvelo(monkeypatch)
    Velo = _load_leaf_module("omicverse.single._velo").Velo

    adata = _make_velocity_adata()
    adata.obs_names = ["cell_a", "cell_b", "cell_c", "cell_d"]
    adata.layers["velo_regvelo"] = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [4.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    perturbed = adata[[2, 0, 3, 1], [2, 0, 1]].copy()
    perturbed.layers["velocity"] = csr_matrix(
        perturbed.layers["velo_regvelo"]
    )

    result = Velo(adata).velocity_effect(perturbed)

    assert result.tolist() == pytest.approx([0.0] * adata.n_obs, abs=1e-7)


def test_velocity_effect_rejects_mismatched_cell_labels(monkeypatch):
    _install_fake_regvelo(monkeypatch)
    Velo = _load_leaf_module("omicverse.single._velo").Velo

    adata = _make_velocity_adata()
    perturbed = _make_velocity_adata()
    adata.layers["velo_regvelo"] = np.ones(adata.shape, dtype=np.float32)
    perturbed.layers["velocity"] = np.ones(perturbed.shape, dtype=np.float32)
    perturbed.obs_names = ["other", *perturbed.obs_names[1:]]

    with pytest.raises(ValueError, match="obs_names.*same labels"):
        Velo(adata).velocity_effect(perturbed)


def test_graphvelo_default_and_subset_gene_markers(monkeypatch):
    from scipy.sparse import csr_matrix

    calls = {"init": [], "train": [], "project": []}

    class FakeGraphVelo:
        def __init__(self, adata, **kwargs):
            self.adata = adata
            calls["init"].append(kwargs)

        def train(self, **kwargs):
            calls["train"].append(kwargs)

        def project_velocity(self, values):
            calls["project"].append(values)
            return np.asarray(values)

    fake_graph_velocity = types.ModuleType(
        "omicverse.external.graphvelo.graph_velocity"
    )
    fake_graph_velocity.GraphVelo = FakeGraphVelo
    fake_graph_utils = types.ModuleType("omicverse.external.graphvelo.utils")
    fake_graph_utils.adj_to_knn = lambda matrix: (
        np.tile(np.arange(matrix.shape[0]), (matrix.shape[0], 1)),
        None,
    )
    monkeypatch.setitem(
        sys.modules,
        "omicverse.external.graphvelo.graph_velocity",
        fake_graph_velocity,
    )
    monkeypatch.setitem(
        sys.modules,
        "omicverse.external.graphvelo.utils",
        fake_graph_utils,
    )

    Velo = _load_leaf_module("omicverse.single._velo").Velo
    adata = _make_velocity_adata()
    adata.layers["velocity_S"] = np.ones(adata.shape, dtype=np.float32)
    adata.obsp["connectivities"] = csr_matrix(np.eye(adata.n_obs))
    adata.uns["neighbors"] = {}
    adata.obsm["X_umap"] = np.ones((adata.n_obs, 2), dtype=np.float32)
    adata.obsm["X_pca"] = np.ones((adata.n_obs, 2), dtype=np.float32)

    Velo(adata).graphvelo()
    assert calls["init"][-1]["gene_subset"] == adata.var_names.tolist()
    assert adata.var["velocity_gv_genes"].tolist() == [True, True, True]

    Velo(adata).graphvelo(gene_subset=["TF1", "TF3"])
    assert calls["init"][-1]["gene_subset"] == ["TF1", "TF3"]
    assert adata.var["velocity_gv_genes"].tolist() == [True, False, True]


def test_graphvelo_rejects_unknown_subset_gene(monkeypatch):
    Velo = _load_leaf_module("omicverse.single._velo").Velo

    adata = _make_velocity_adata()
    with pytest.raises(KeyError, match="missing_gene"):
        Velo(adata).graphvelo(gene_subset=["TF1", "missing_gene"])


def test_adata_first_perturbation_wrappers(monkeypatch):
    _install_fake_regvelo(monkeypatch)
    mod = _load_leaf_module("omicverse.single._velo")

    adata = _make_velocity_adata()
    perturbed = _make_velocity_adata()
    adata.layers["velo_regvelo"] = np.ones(adata.shape, dtype=np.float32)
    perturbed.layers["velocity"] = np.ones(adata.shape, dtype=np.float32)

    effect_result = mod.perturbation_effect(
        adata,
        perturbed,
        terminal_states="state_a",
    )
    fate_result = mod.cell_fate_perturbation(
        adata,
        {"TF1": perturbed},
        terminal_states="state_a",
    )
    velocity_result = mod.velocity_effect(
        adata,
        perturbed,
        target=["TF1", "TF3"],
    )

    assert effect_result is adata
    assert "perturbation effect on state_a" in adata.obs
    assert fate_result is adata.uns["cell_fate_perturbation"]
    assert velocity_result.name == "TF1+TF3_velocity_effect"
    assert adata.obs["TF1+TF3_velocity_effect"].tolist() == pytest.approx([0.0] * adata.n_obs)


def test_cellrank_fate_skips_terminal_states_not_in_macrostates(monkeypatch):
    calls = {
        "velocity_kernel": [],
        "connectivity_kernel": [],
        "compute_macrostates": [],
        "set_terminal_states": [],
        "compute_fate_probabilities": 0,
        "plot_macrostates": [],
    }

    class FakeKernel:
        def __init__(self, adata=None, **kwargs):
            self.adata = adata
            self.kwargs = kwargs

        def compute_transition_matrix(self):
            return self

        def __mul__(self, other):
            return self

        __rmul__ = __mul__

        def __add__(self, other):
            return self

    class FakeVelocityKernel(FakeKernel):
        def __init__(self, adata=None, **kwargs):
            super().__init__(adata, **kwargs)
            calls["velocity_kernel"].append(kwargs)

    class FakeConnectivityKernel(FakeKernel):
        def __init__(self, adata=None, **kwargs):
            super().__init__(adata, **kwargs)
            calls["connectivity_kernel"].append(kwargs)

    class FakeGPCCA:
        def __init__(self, kernel):
            self.kernel = kernel
            self.macrostates = types.SimpleNamespace(names=["state_a", "state_c"])

        def compute_macrostates(self, **kwargs):
            calls["compute_macrostates"].append(kwargs)

        def set_terminal_states(self, states):
            calls["set_terminal_states"].append(states)

        def compute_fate_probabilities(self):
            calls["compute_fate_probabilities"] += 1

        def plot_macrostates(self, **kwargs):
            calls["plot_macrostates"].append(kwargs)
            warnings.warn(
                "No data for colormapping provided via 'c'. Parameters 'cmap' will be ignored",
                UserWarning,
            )

    fake_cellrank = types.ModuleType("cellrank")
    fake_cellrank.kernels = types.SimpleNamespace(
        VelocityKernel=FakeVelocityKernel,
        ConnectivityKernel=FakeConnectivityKernel,
    )
    fake_cellrank.estimators = types.SimpleNamespace(GPCCA=FakeGPCCA)
    monkeypatch.setitem(sys.modules, "cellrank", fake_cellrank)

    Velo = _load_leaf_module("omicverse.single._velo").Velo
    adata = _make_velocity_adata()
    adata.obs["cell_type"] = ["state_a", "state_b", "state_c", "state_b"]
    velo = Velo(adata)

    with pytest.warns(UserWarning, match="state_b") as captured:
        estimator = velo.cellrank_fate(
            velocity_key="velocity",
            xkey="Ms",
            cluster_key="cell_type",
            terminal_states=["state_a", "state_b"],
            n_states=2,
            compute_fate_probabilities=True,
            plot=True,
        )

    assert not any(
        "No data for colormapping provided via 'c'" in str(warning.message)
        for warning in captured
    )
    assert estimator.macrostates.names == ["state_a", "state_c"]
    assert calls["velocity_kernel"] == [{"xkey": "Ms", "vkey": "velocity"}]
    assert calls["set_terminal_states"] == [["state_a"]]
    assert calls["compute_fate_probabilities"] == 1
    assert calls["plot_macrostates"] == [
        {"which": "terminal", "basis": "umap", "legend_loc": "right", "s": 100}
    ]
    assert adata.uns["velocity_cellrank"]["terminal_states"] == ["state_a"]
    assert adata.uns["velocity_cellrank"]["missing_terminal_states"] == ["state_b"]
    assert adata.uns["velocity_cellrank"]["estimator"] is estimator
    assert "kernel" in adata.uns["velocity_cellrank"]


def test_adata_first_cellrank_fate_wrapper_stores_estimator(monkeypatch):
    calls = {"velocity_kernel": [], "compute_macrostates": []}

    class FakeKernel:
        def __init__(self, adata=None, **kwargs):
            calls["velocity_kernel"].append(kwargs)

        def compute_transition_matrix(self):
            return self

    class FakeGPCCA:
        def __init__(self, kernel):
            self.kernel = kernel
            self.macrostates = types.SimpleNamespace(names=["state_a"])

        def compute_macrostates(self, **kwargs):
            calls["compute_macrostates"].append(kwargs)

    fake_cellrank = types.ModuleType("cellrank")
    fake_cellrank.kernels = types.SimpleNamespace(
        VelocityKernel=FakeKernel,
        ConnectivityKernel=FakeKernel,
    )
    fake_cellrank.estimators = types.SimpleNamespace(GPCCA=FakeGPCCA)
    monkeypatch.setitem(sys.modules, "cellrank", fake_cellrank)

    mod = _load_leaf_module("omicverse.single._velo")
    adata = _make_velocity_adata()
    estimator = mod.cellrank_fate(
        adata,
        velocity_key="velocity",
        xkey="Ms",
        n_states=1,
        connectivity_weight=0,
    )

    assert adata.uns["velocity_cellrank"]["estimator"] is estimator
    assert calls["velocity_kernel"] == [{"xkey": "Ms", "vkey": "velocity"}]
    assert calls["compute_macrostates"] == [{"n_states": 1, "n_cells": 30}]


def test_adata_first_velocity_wrapper_returns_adata(monkeypatch):
    _install_fake_regvelo(monkeypatch)
    mod = _load_leaf_module("omicverse.single._velo")

    adata = _make_velocity_adata()
    result = mod.velocity(
        adata,
        method="regvelo",
        velocity_key="velocity_regvelo",
        train_kwargs={"max_epochs": 1},
    )

    assert result is adata
    assert "velocity_regvelo" in adata.layers


def test_default_velocity_color_prefers_cell_type():
    Velo = _load_leaf_module("omicverse.single._velo").Velo

    adata = _make_velocity_adata()
    adata.obs["clusters"] = ["a", "a", "b", "b"]
    adata.obs["cell_type"] = ["type1", "type1", "type2", "type2"]
    velo = Velo(adata)

    assert velo._default_velocity_color_key() == "cell_type"


def test_cal_velocity_regvelo_missing_dependency_has_install_hint(monkeypatch):
    original_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".")[0] == "regvelo":
            raise ImportError("blocked import: regvelo")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    sys.modules.pop("regvelo", None)

    Velo = _load_leaf_module("omicverse.single._velo").Velo

    adata = _make_velocity_adata()
    velo = Velo(adata)

    with pytest.raises(ImportError, match="pip install regvelo scvi-tools"):
        velo.cal_velocity(method="regvelo")
