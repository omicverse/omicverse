import importlib.util
import sys
import types
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_module(module_name: str, relative_path: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _seed_packages():
    omicverse_pkg = types.ModuleType("omicverse")
    omicverse_pkg.__path__ = [str(PROJECT_ROOT / "omicverse")]
    sys.modules["omicverse"] = omicverse_pkg

    pl_pkg = types.ModuleType("omicverse.pl")
    pl_pkg.__path__ = [str(PROJECT_ROOT / "omicverse" / "pl")]
    sys.modules["omicverse.pl"] = pl_pkg

    utils_pkg = types.ModuleType("omicverse.utils")
    utils_pkg.__path__ = [str(PROJECT_ROOT / "omicverse" / "utils")]
    sys.modules["omicverse.utils"] = utils_pkg


def test_utils_plot_wrapper_warns_and_delegates(monkeypatch):
    _seed_packages()
    calls = []

    backend = types.ModuleType("omicverse.pl._plot_backend")

    def plot_set(*args, **kwargs):
        calls.append(("plot_set", args, kwargs))
        return "ok"

    backend.plot_set = plot_set
    backend.plotset = plot_set
    backend.ov_plot_set = plot_set
    backend.style = plot_set
    backend.palette = lambda: ["#000000"]
    backend.plot_text_set = plot_set
    backend.ticks_range = plot_set
    backend.plot_boxplot = plot_set
    backend.plot_network = plot_set
    backend.plot_cellproportion = plot_set
    backend.plot_embedding_celltype = plot_set
    backend.geneset_wordcloud = plot_set
    backend.plot_pca_variance_ratio = plot_set
    backend.plot_pca_variance_ratio1 = plot_set
    backend.gen_mpl_labels = plot_set
    backend._vector_friendly = True
    backend.pyomic_palette = []
    backend.blue_palette = []
    backend.orange_palette = []
    backend.red_palette = []
    backend.green_palette = []
    backend.sc_color = []
    backend.red_color = []
    backend.green_color = []
    backend.orange_color = []
    backend.blue_color = []
    backend.purple_color = []
    backend.palette_28 = []
    backend.cet_g_bw = []
    backend.palette_112 = []
    backend.palette_56 = []
    backend.vibrant_palette = []
    backend.earth_palette = []
    backend.pastel_palette = []
    monkeypatch.setitem(sys.modules, "omicverse.pl._plot_backend", backend)

    module = _load_module("omicverse.utils._plot", "omicverse/utils/_plot.py")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert module.plot_set(1, dpi=120) == "ok"

    assert calls == [("plot_set", (1,), {"dpi": 120})]
    assert caught
    messages = [str(warning.message) for warning in caught]
    assert any("will be removed in omicverse 2.2" in message for message in messages)
    assert any("ov.pl.plot_set" in message for message in messages)


def test_utils_scatterplot_wrapper_warns_and_delegates(monkeypatch):
    _seed_packages()
    calls = []

    backend = types.ModuleType("omicverse.pl._scatterplot_backend")

    def embedding(*args, **kwargs):
        calls.append(("embedding", args, kwargs))
        return "embedded"

    for name in [
        "embedding",
        "umap",
        "tsne",
        "pca",
        "spatial",
        "diffmap",
        "draw_graph",
        "_get_vector_friendly",
        "_get_vboundnorm",
        "_add_categorical_legend",
        "_get_basis",
        "_get_color_source_vector",
        "_get_palette",
        "_color_vector",
        "_basis2name",
        "_components_to_dimensions",
        "_embedding",
    ]:
        setattr(backend, name, embedding)
    monkeypatch.setitem(sys.modules, "omicverse.pl._scatterplot_backend", backend)

    module = _load_module("omicverse.utils._scatterplot", "omicverse/utils/_scatterplot.py")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert module.embedding("adata", basis="X_umap") == "embedded"

    assert calls == [("embedding", ("adata",), {"basis": "X_umap"})]
    assert caught
    assert "ov.pl.embedding" in str(caught[0].message)


def test_utils_venn_wrapper_warns_and_delegates(monkeypatch):
    _seed_packages()
    calls = []

    backend = types.ModuleType("omicverse.pl._venn_backend")

    def venny4py(*args, **kwargs):
        calls.append(("venny4py", args, kwargs))
        return "venn"

    backend.get_shared = venny4py
    backend.get_unique = venny4py
    backend.venny4py = venny4py
    monkeypatch.setitem(sys.modules, "omicverse.pl._venn_backend", backend)

    module = _load_module("omicverse.utils._venn", "omicverse/utils/_venn.py")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert module.venny4py(sets={"A": {1}}) == "venn"

    assert calls == [("venny4py", (), {"sets": {"A": {1}}})]
    assert caught
    assert "ov.pl.venn" in str(caught[0].message)
