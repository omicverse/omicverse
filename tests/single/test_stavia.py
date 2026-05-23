import types
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import omicverse.single._stavia as stavia_mod
import omicverse.single._via as via_mod
from omicverse.single._stavia import StaVIA


def _make_adata():
    obs = pd.DataFrame(
        {
            "clusters": ["stem", "stem", "late", "late"],
            "time": [0, 1, 2, 3],
            "slice": ["s1", "s1", "s2", "s2"],
        },
        index=[f"cell{i}" for i in range(4)],
    )
    adata = AnnData(
        X=np.ones((4, 3)),
        obs=obs,
        var=pd.DataFrame(index=[f"gene{i}" for i in range(3)]),
    )
    adata.obsm["X_pca"] = np.arange(20, dtype=float).reshape(4, 5)
    adata.obsm["X_umap"] = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    adata.obsm["spatial"] = np.array(
        [[10.0, 20.0, 1.0], [11.0, 20.0, 1.0], [50.0, 60.0, 2.0], [51.0, 61.0, 2.0]]
    )
    return adata


def _fake_via_backend(captured):
    class FakeVIA:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            n_obs = kwargs["data"].shape[0]
            self.single_cell_pt_markov = np.linspace(0.0, 1.0, n_obs)
            self.labels = np.array([0, 0, 1, 1])
            self.terminal_clusters = [5, 8]
            self.single_cell_bp = np.vstack(
                [np.linspace(1.0, 0.0, n_obs), np.linspace(0.0, 1.0, n_obs)]
            )
            self.ran = False

        def run_VIA(self):
            self.ran = True

    return types.SimpleNamespace(core=types.SimpleNamespace(VIA=FakeVIA))


def _install_fake_pygam(monkeypatch, linear_gam=None):
    class FakeLinearGAM:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setitem(
        sys.modules,
        "pygam",
        types.SimpleNamespace(LinearGAM=linear_gam or FakeLinearGAM),
    )


def test_stavia_fit_translates_anndata_keys_and_writes_results(monkeypatch):
    adata = _make_adata()
    captured = {}
    monkeypatch.setattr(
        stavia_mod,
        "_load_via_backend",
        lambda *, rw2_mode=False: _fake_via_backend(captured),
    )

    model = StaVIA(
        adata,
        use_rep="X_pca",
        n_comps=3,
        basis="X_umap",
        cluster_key="clusters",
        spatial_key="spatial",
        time_key="time",
        sample_key="slice",
        spatial_knn=7,
        root="stem",
        random_seed=11,
    ).fit()

    assert model.model.ran is True
    assert captured["data"].shape == (4, 3)
    np.testing.assert_allclose(captured["embedding"], adata.obsm["X_umap"])
    np.testing.assert_allclose(captured["spatial_coords"], adata.obsm["spatial"][:, :2])
    assert captured["do_spatial_knn"] is True
    assert captured["spatial_knn"] == 7
    assert captured["spatial_aux"] == ["s1", "s1", "s2", "s2"]
    assert captured["time_series"] is True
    assert captured["time_series_labels"] == [0, 1, 2, 3]
    assert captured["root_user"] == ["stem"]
    assert captured["random_seed"] == 11
    assert captured["do_compute_embedding"] is False

    assert "stavia_pseudotime" in adata.obs
    np.testing.assert_allclose(adata.obs["stavia_pseudotime"], np.linspace(0.0, 1.0, 4))
    assert list(adata.obs["stavia_cluster"].astype(str)) == ["0", "0", "1", "1"]
    assert "stavia_lineage_probabilities" in adata.obsm
    lineage = adata.obsm["stavia_lineage_probabilities"]
    assert list(lineage.columns) == ["lineage_5", "lineage_8"]
    assert lineage.shape == (4, 2)
    assert adata.uns["stavia"]["spatial_key"] == "spatial"
    assert adata.uns["stavia"]["pseudotime_key"] == "stavia_pseudotime"


def test_stavia_fit_suppresses_backend_plots_only(monkeypatch, capsys):
    adata = _make_adata()
    captured = {}

    class NoisyVIA:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            n_obs = kwargs["data"].shape[0]
            self.single_cell_pt_markov = np.linspace(0.0, 1.0, n_obs)
            self.labels = np.array([0, 0, 1, 1])
            self.terminal_clusters = [5]
            self.single_cell_bp = np.ones((1, n_obs))

        def run_VIA(self):
            print("backend output should stay visible")
            plt.figure()
            plt.plot([0, 1], [0, 1])
            plt.show()

    monkeypatch.setattr(
        stavia_mod,
        "_load_via_backend",
        lambda *, rw2_mode=False: types.SimpleNamespace(core=types.SimpleNamespace(VIA=NoisyVIA)),
    )

    existing_figures = set(plt.get_fignums())
    StaVIA(adata, n_comps=3, spatial_key=None).fit()
    output = capsys.readouterr()

    assert "backend output should stay visible" in output.out
    assert output.err == ""
    assert set(plt.get_fignums()) == existing_figures


def test_pyvia_run_suppresses_backend_plots_only(monkeypatch, capsys):
    adata = _make_adata()
    adata.obsm["tsne"] = adata.obsm["X_umap"]
    adata.obs["label"] = pd.Categorical(["HSC", "HSC", "late", "late"])

    class FakeVIA:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_VIA(self):
            print("pyVIA backend output should stay visible")
            plt.figure()
            plt.plot([0, 1], [1, 0])
            plt.show()

    monkeypatch.setattr(via_mod, "_load_via_modules", lambda: setattr(via_mod, "VIA", FakeVIA))

    model = via_mod.pyVIA(
        adata=adata,
        adata_key="X_pca",
        adata_ncomps=3,
        basis="tsne",
        clusters="label",
        knn=2,
        random_seed=4,
        root_user=[0],
    )

    existing_figures = set(plt.get_fignums())
    model.run()
    output = capsys.readouterr()

    assert "pyVIA backend output should stay visible" in output.out
    assert output.err == ""
    assert set(plt.get_fignums()) == existing_figures


def test_pyvia_lineage_probability_honors_figsize(monkeypatch):
    adata = _make_adata()
    adata.obsm["tsne"] = adata.obsm["X_umap"]
    adata.obs["label"] = pd.Categorical(["HSC", "HSC", "late", "late"])

    class FakeVIA:
        def __init__(self, **kwargs):
            self.terminal_clusters = list(range(7))

    def fake_plot_sc_lineage_probability(**kwargs):
        n_terminal = len(kwargs["marker_lineages"])
        ncols = min(3, n_terminal)
        nrows, mod = divmod(n_terminal, ncols)
        if mod:
            nrows += 1
        fig, axs = plt.subplots(nrows, ncols)
        return fig, axs

    monkeypatch.setattr(via_mod, "_load_via_modules", lambda: setattr(via_mod, "VIA", FakeVIA))
    monkeypatch.setattr(via_mod, "plot_sc_lineage_probability", fake_plot_sc_lineage_probability, raising=False)

    model = via_mod.pyVIA(
        adata=adata,
        adata_key="X_pca",
        adata_ncomps=3,
        basis="tsne",
        clusters="label",
    )

    fig, axs = model.plot_lineage_probability(
        figsize=(10, 5),
        marker_lineages=list(range(7)),
        ncol=4,
    )

    assert tuple(fig.get_size_inches()) == pytest.approx((10, 5))
    assert axs.shape == (2, 4)
    plt.close(fig)


def test_via_core_get_gene_expression_uses_shared_legend(monkeypatch):
    class FakeGAM:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, x, y, weights=None):
            self.y_mean = float(np.mean(y))
            return self

        def predict(self, X):
            X = np.asarray(X, dtype=float)
            return self.y_mean + 0.1 * X

        def confidence_intervals(self, X, width=0.95):
            y = self.predict(X)
            return np.column_stack([y - 0.05, y + 0.05])

    _install_fake_pygam(monkeypatch, FakeGAM)
    from omicverse.external.VIA import core as via_core

    class FakeVIA:
        terminal_clusters = [1, 2]
        labels = np.array([1, 1, 1, 2, 2, 2])
        true_label = np.array(["A", "A", "A", "B", "B", "B"])
        single_cell_pt_markov = np.linspace(0.0, 1.0, 6)
        single_cell_bp = np.array(
            [
                [1.0, 0.2],
                [0.95, 0.4],
                [0.9, 0.6],
                [0.5, 0.95],
                [0.3, 1.0],
                [0.2, 0.95],
            ],
            dtype=float,
        )

        @staticmethod
        def func_mode(values):
            return pd.Series(values).mode().iloc[0]

    gene_exp = pd.DataFrame(
        np.arange(30, dtype=float).reshape(6, 5),
        columns=[f"G{i}" for i in range(5)],
    )

    fig, axs = via_core.get_gene_expression(
        FakeVIA(),
        gene_exp=gene_exp,
        marker_genes=list(gene_exp.columns),
        marker_lineages=[1, 2],
        dpi=80,
        ncols=3,
        fontsize_=8,
    )

    assert np.asarray(axs).shape == (2, 3)
    assert len(fig.legends) == 1
    assert all(ax.get_legend() is None for ax in np.ravel(axs) if ax.axison)
    assert any(text.get_text() == "Time" for text in fig.texts)
    assert any(text.get_text() == "Intensity" for text in fig.texts)
    visible_axes = [ax for ax in np.ravel(axs) if ax.axison]
    bottom = min(ax.get_position().y0 for ax in visible_axes)
    left = min(ax.get_position().x0 for ax in visible_axes)
    time_text = [text for text in fig.texts if text.get_text() == "Time"][0]
    intensity_text = [text for text in fig.texts if text.get_text() == "Intensity"][0]
    assert 0.0 < bottom - time_text.get_position()[1] < 0.08
    assert 0.0 < left - intensity_text.get_position()[0] < 0.08
    plt.close(fig)


def test_via_get_gene_expression_skips_single_cell_lineage(monkeypatch):
    _install_fake_pygam(monkeypatch)
    from omicverse.external.VIA import core as via_core
    from omicverse.external.VIA import plotting_via_ov

    class FakeVIA:
        terminal_clusters = [1]
        labels = np.array([1, 0, 0])
        true_label = np.array(["A", "B", "B"])
        single_cell_pt_markov = np.array([0.0, 0.5, 1.0])
        single_cell_bp = np.array([[1.0], [0.0], [0.0]], dtype=float)

        @staticmethod
        def func_mode(values):
            return pd.Series(values).mode().iloc[0]

    gene_exp = pd.DataFrame({"G0": [1.0, 2.0, 3.0]})

    fig, axs = via_core.get_gene_expression(
        FakeVIA(),
        gene_exp=gene_exp,
        marker_genes=["G0"],
        marker_lineages=[1],
        dpi=80,
    )
    assert sum(len(ax.lines) for ax in np.ravel(axs)) == 0
    plt.close(fig)

    fig, axs = plotting_via_ov.get_gene_expression_ov(
        None,
        None,
        FakeVIA(),
        gene_exp=gene_exp,
        marker_genes=["G0"],
        marker_lineages=[1],
        dpi=80,
    )
    assert sum(len(ax.lines) for ax in np.ravel(axs)) == 0
    plt.close(fig)


def test_stavia_rejects_missing_spatial_key():
    adata = _make_adata()
    model = StaVIA(adata, spatial_key="missing")

    with pytest.raises(KeyError, match="spatial_key='missing'"):
        model.fit()


def test_stavia_plot_methods_are_not_exposed_on_wrapper():
    for name in (
        "plot_stream",
        "plot_graph",
        "plot_trajectory",
        "plot_lineage_probability",
    ):
        assert not hasattr(StaVIA, name)


def test_stavia_missing_core_dependency_message(monkeypatch):
    real_import_module = stavia_mod.importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "leidenalg":
            raise ImportError("blocked import: leidenalg")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(stavia_mod.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="VIA runtime dependencies"):
        stavia_mod._load_via_backend()


def test_stavia_dependency_helpers_avoid_module_level_required_lists():
    assert not hasattr(stavia_mod, "_STAVIA_REQUIRED_MODULES")
    assert not hasattr(stavia_mod, "_STAVIA_RW2_MODULES")
    assert stavia_mod._stavia_required_modules() == ("leidenalg", "hnswlib", "pygam")
    assert stavia_mod._stavia_required_modules(rw2=True) == (
        "leidenalg",
        "hnswlib",
        "pygam",
        "pecanpy",
        "numba_progress",
    )


def test_stavia_loader_checks_via_runtime_dependencies_for_basic_mode(monkeypatch):
    checked = []

    def fake_require_modules(dependencies, *, rw2=False):
        checked.append((tuple(dependencies), rw2))

    def fake_import_module(name, *args, **kwargs):
        if name == "..external.VIA":
            return types.SimpleNamespace(core=types.SimpleNamespace(VIA=object))
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(stavia_mod, "_require_modules", fake_require_modules)
    monkeypatch.setattr(stavia_mod.importlib, "import_module", fake_import_module)

    stavia_mod._load_via_backend()

    assert checked == [(("leidenalg", "hnswlib", "pygam"), False)]


def test_stavia_loader_checks_rw2_dependencies_only_when_enabled(monkeypatch):
    checked = []

    def fake_require_modules(dependencies, *, rw2=False):
        checked.append((tuple(dependencies), rw2))

    def fake_import_module(name, *args, **kwargs):
        if name == "..external.VIA":
            return types.SimpleNamespace(core=types.SimpleNamespace(VIA=object))
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(stavia_mod, "_require_modules", fake_require_modules)
    monkeypatch.setattr(stavia_mod.importlib, "import_module", fake_import_module)

    stavia_mod._load_via_backend(rw2_mode=True)

    assert checked == [
        (("leidenalg", "hnswlib", "pygam"), False),
        (("pecanpy", "numba_progress"), True),
    ]


def test_stavia_rw2_missing_dependency_message(monkeypatch):
    def fake_require_modules(dependencies, *, rw2=False):
        if rw2:
            stavia_mod._raise_stavia_dependency_error(dependencies, rw2=rw2)

    monkeypatch.setattr(stavia_mod, "_require_modules", fake_require_modules)

    with pytest.raises(ImportError, match="pip install pecanpy numba-progress"):
        stavia_mod._load_via_backend(rw2_mode=True)


def test_vendored_via_core_has_no_dependency_fallback_helpers(monkeypatch):
    _install_fake_pygam(monkeypatch)
    from omicverse.external.VIA import core as via_core
    from omicverse.external.VIA import utils_via

    assert not hasattr(via_core, "_is_missing_rw2_dependency")
    assert not hasattr(utils_via, "_SklearnKNNIndex")
    assert not hasattr(utils_via, "_build_knn_index")
    assert not hasattr(utils_via, "straight_edge_bundle")


def test_trajinfer_dispatches_stavia(monkeypatch):
    pytest.importorskip("torch")
    from omicverse.single._traj import TrajInfer

    adata = _make_adata()
    captured = {}
    monkeypatch.setattr(
        stavia_mod,
        "_load_via_backend",
        lambda *, rw2_mode=False: _fake_via_backend(captured),
    )

    traj = TrajInfer(adata, basis="X_umap", use_rep="X_pca", n_comps=2, groupby="clusters")
    model = traj.inference(
        method="stavia",
        spatial_key="spatial",
        time_key="time",
        sample_key="slice",
        key_added="traj_stavia",
    )

    assert model.__class__.__name__ == "StaVIA"
    assert traj.stavia is model
    assert captured["data"].shape == (4, 2)
    assert captured["true_label"].equals(adata.obs["clusters"])
    assert "traj_stavia_pseudotime" in adata.obs
