import numpy as np
import pandas as pd
import pytest
import warnings
from anndata import AnnData
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import omicverse.pl._dotplot as dotplot_mod


class _StubAnnotation:
    def __init__(self, kind, *args, **kwargs):
        self.kind = kind
        self.args = args
        self.kwargs = kwargs


class _StubSizedHeatmap:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.figure = object()
        self.title_calls = []
        self.__class__.instances.append(self)

    def add_top(self, *args, **kwargs):
        return None

    def add_left(self, *args, **kwargs):
        return None

    def add_right(self, *args, **kwargs):
        return None

    def add_dendrogram(self, *args, **kwargs):
        return None

    def add_legends(self, *args, **kwargs):
        return None

    def add_title(self, *args, **kwargs):
        self.title_calls.append((args, kwargs))
        return None

    def group_cols(self, *args, **kwargs):
        return None

    def render(self):
        # Marsilea renders in place and exposes the result as ``.figure``.
        return None


@pytest.fixture
def simple_adata():
    adata = AnnData(
        np.array(
            [
                [1.0, 0.0, 2.0],
                [2.0, 1.0, 0.0],
                [0.0, 3.0, 1.0],
                [1.0, 2.0, 2.0],
            ]
        )
    )
    adata.var_names = ["g1", "g2", "g3"]
    adata.obs["cell_type"] = pd.Categorical(["A", "A", "B", "B"])
    return adata


@pytest.fixture
def stub_marsilea(monkeypatch):
    monkeypatch.setattr(dotplot_mod.ma, "SizedHeatmap", _StubSizedHeatmap)
    monkeypatch.setattr(dotplot_mod.mp, "Labels", lambda *a, **k: _StubAnnotation("Labels", *a, **k))
    monkeypatch.setattr(dotplot_mod.mp, "Colors", lambda *a, **k: _StubAnnotation("Colors", *a, **k))
    monkeypatch.setattr(dotplot_mod.mp, "Numbers", lambda *a, **k: _StubAnnotation("Numbers", *a, **k))


def test_dotplot_warns_for_broken_marsilea_056(simple_adata, stub_marsilea, monkeypatch):
    monkeypatch.setattr(dotplot_mod.ma, "__version__", "0.5.6")

    with pytest.warns(UserWarning, match="marsilea 0.5.6"):
        dotplot_mod.dotplot(
            simple_adata,
            ["g1", "g2"],
            groupby="cell_type",
            show=False,
        )


def test_dotplot_does_not_warn_for_fixed_marsilea_versions(simple_adata, stub_marsilea, monkeypatch):
    monkeypatch.setattr(dotplot_mod.ma, "__version__", "0.5.7")

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        dotplot_mod.dotplot(
            simple_adata,
            ["g1", "g2"],
            groupby="cell_type",
            show=False,
        )

    assert not [w for w in record if "marsilea 0.5.6" in str(w.message)]


def test_dotplot_honors_title_and_returns_rendered_figure(simple_adata, stub_marsilea):
    _StubSizedHeatmap.instances.clear()
    fig = dotplot_mod.dotplot(
        simple_adata,
        ["g1", "g2"],
        groupby="cell_type",
        title="Marker genes",
        return_fig=True,
    )

    assert fig is _StubSizedHeatmap.instances[0].figure
    assert _StubSizedHeatmap.instances[0].title_calls == [
        ((), {"top": "Marker genes", "pad": 0.2, "fontsize": 13, "fontweight": "bold"})
    ]


def test_dotplot_pdf_export(simple_adata, tmp_path):
    output = tmp_path / "dotplot.pdf"
    fig = dotplot_mod.dotplot(
        simple_adata,
        ["g1", "g2"],
        groupby="cell_type",
        title="Marker genes",
        save=str(output),
        return_fig=True,
    )

    assert isinstance(fig, Figure)
    assert output.read_bytes().startswith(b"%PDF")
    assert any(text.get_text() == "Marker genes" for ax in fig.axes for text in ax.texts)
    plt.close(fig)


def test_rank_dotplot_preserves_figure_return(simple_adata, monkeypatch):
    rendered_figure = object()
    captured = {}

    def fake_dotplot(*args, **kwargs):
        captured.update(kwargs)
        return rendered_figure

    monkeypatch.setattr(dotplot_mod, "dotplot", fake_dotplot)

    result = dotplot_mod.rank_genes_groups_dotplot(
        simple_adata,
        var_names=["g1", "g2"],
        groups=["cell_type"],
        groupby="cell_type",
        show=False,
    )

    assert result is rendered_figure
    assert captured["return_fig"] is True


def test_rank_genes_groups_df_drops_unaligned_long_statistics():
    adata = AnnData(np.zeros((4, 78)))
    adata.uns["cell_type_roughly_cosg"] = {
        "names": pd.DataFrame({"CMP": [f"g{i}" for i in range(50)]}),
        "scores": pd.DataFrame({"CMP": np.arange(50, dtype=float)}),
        "logfoldchanges": pd.DataFrame({"CMP": np.arange(50, dtype=float)}),
        "pvals": pd.DataFrame({"CMP": np.arange(78, dtype=float)}),
        "pvals_adj": pd.DataFrame({"CMP": np.arange(78, dtype=float)}),
    }

    df = dotplot_mod.rank_genes_groups_df(
        adata,
        "CMP",
        key="cell_type_roughly_cosg",
    )

    assert df.shape == (50, 3)
    assert df.columns.tolist() == ["names", "scores", "logfoldchanges"]


def test_rank_genes_groups_df_rejects_statistics_shorter_than_names():
    adata = AnnData(np.zeros((4, 5)))
    adata.uns["bad_markers"] = {
        "names": pd.DataFrame({"CMP": [f"g{i}" for i in range(5)]}),
        "scores": pd.DataFrame({"CMP": np.arange(4, dtype=float)}),
    }

    with pytest.raises(ValueError, match="Inconsistent rank_genes_groups"):
        dotplot_mod.rank_genes_groups_df(adata, "CMP", key="bad_markers")


def test_rank_genes_groups_df_keeps_aligned_pvals():
    adata = AnnData(np.zeros((4, 5)))
    adata.uns["rank_genes_groups"] = {
        "names": pd.DataFrame({"CMP": [f"g{i}" for i in range(5)]}),
        "scores": pd.DataFrame({"CMP": np.arange(5, dtype=float)}),
        "pvals": pd.DataFrame({"CMP": np.linspace(0.01, 0.05, 5)}),
        "pvals_adj": pd.DataFrame({"CMP": np.linspace(0.02, 0.1, 5)}),
    }

    df = dotplot_mod.rank_genes_groups_df(adata, "CMP")

    assert df.columns.tolist() == ["names", "scores", "pvals", "pvals_adj"]
    np.testing.assert_allclose(df["pvals_adj"], np.linspace(0.02, 0.1, 5))
