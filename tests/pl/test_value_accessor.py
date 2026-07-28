"""Tests for the one place ``ov.pl`` turns a name into numbers.

Nine modules used to carry their own copy of "look it up in ``.obs``, else
pull the gene column, else densify", and they disagreed in ways that showed up
as crashes rather than as wrong pictures:

* ``_density`` called ``.toarray()`` unconditionally — it raised on any
  ``AnnData`` whose ``.X`` is a plain ndarray;
* ``_space`` called ``.flatten()`` on the column — it raised on a sparse
  ``.X``, which is the *usual* case;
* ``_single.half_violin_boxplot`` checked ``.raw`` before ``.obs``, so a name
  present in both resolved differently there than anywhere else;
* only ``_violin`` and ``_dotplot`` honoured ``layer=``.

Each of those is pinned below, plus the resolution rules themselves.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

anndata = pytest.importorskip("anndata")
from scipy import sparse  # noqa: E402

from omicverse.pl._plotdata import (  # noqa: E402
    as_plotdata, get_matrix, get_values,
)

N_CELLS, N_GENES = 40, 4
GENES = ["GeneA", "GeneB", "GeneC", "GeneD"]


def _obs():
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {"group": np.repeat(["a", "b", "c", "d"], N_CELLS // 4),
         "score": rng.normal(size=N_CELLS)},
        index=[f"cell{i}" for i in range(N_CELLS)],
    )


def _values():
    return (np.arange(N_CELLS * N_GENES, dtype=np.float32)
            .reshape(N_CELLS, N_GENES))


@pytest.fixture(params=["dense", "sparse"])
def adata(request):
    """The same data as a dense and as a sparse AnnData.

    Parameterised on purpose: the bugs this module replaces were each a
    crash in exactly one of the two.
    """
    matrix = _values()
    X = sparse.csr_matrix(matrix) if request.param == "sparse" else matrix
    out = anndata.AnnData(X, obs=_obs())
    out.var_names = GENES
    return out


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


class TestResolution:
    def test_metadata_column(self, adata):
        assert np.allclose(get_values(adata, "score"), adata.obs["score"])

    def test_feature_from_x(self, adata):
        assert np.allclose(get_values(adata, "GeneB"), _values()[:, 1])

    def test_result_is_always_one_dimensional_and_dense(self, adata):
        values = get_values(adata, "GeneA")
        assert values.ndim == 1
        assert isinstance(values, np.ndarray)
        assert len(values) == N_CELLS

    def test_metadata_wins_over_a_same_named_feature(self):
        matrix = _values()
        out = anndata.AnnData(matrix, obs=_obs())
        out.var_names = ["score", "GeneB", "GeneC", "GeneD"]
        assert np.allclose(get_values(out, "score"), out.obs["score"])
        assert not np.allclose(get_values(out, "score"), matrix[:, 0])

    def test_dataframe_and_dict(self):
        frame = _obs()
        assert np.allclose(get_values(frame, "score"), frame["score"])
        assert np.allclose(get_values({"v": [1.0, 2.0]}, "v"), [1.0, 2.0])

    def test_plotdata_passes_through(self, adata):
        view = as_plotdata(adata)
        assert np.allclose(get_values(view, "GeneC"), _values()[:, 2])

    def test_unknown_name_suggests_a_close_match(self, adata):
        with pytest.raises(KeyError, match="GeneA"):
            get_values(adata, "GeneAA")

    def test_error_names_both_namespaces(self, adata):
        with pytest.raises(KeyError, match="neither a metadata column nor a feature"):
            get_values(adata, "nothing_like_this_at_all")


class TestLayers:
    def test_layer_is_read(self, adata):
        adata.layers["counts"] = np.ones((N_CELLS, N_GENES), dtype=np.float32) * 7
        assert np.allclose(get_values(adata, "GeneA", layer="counts"), 7.0)

    def test_unknown_layer_lists_the_options(self, adata):
        adata.layers["counts"] = np.zeros((N_CELLS, N_GENES), dtype=np.float32)
        with pytest.raises(KeyError, match="counts"):
            get_values(adata, "GeneA", layer="lognorm")

    def test_layer_on_a_metadata_column_is_rejected(self, adata):
        with pytest.raises(ValueError, match="only applies to features"):
            get_values(adata, "score", layer="counts")

    def test_layer_and_use_raw_together_is_rejected(self, adata):
        adata.raw = adata
        adata.layers["counts"] = np.zeros((N_CELLS, N_GENES), dtype=np.float32)
        with pytest.raises(ValueError, match="not both"):
            get_values(adata, "GeneA", layer="counts", use_raw=True)


class TestRaw:
    @staticmethod
    def _subset_with_raw():
        """The situation HVG subsetting leaves behind: raw has more genes."""
        full = anndata.AnnData(_values(), obs=_obs())
        full.var_names = GENES
        subset = full[:, ["GeneA", "GeneB"]].copy()
        subset.raw = full
        return subset

    def test_x_is_preferred_when_it_can_answer(self):
        subset = self._subset_with_raw()
        subset.X = np.zeros_like(subset.X)
        # GeneA is in .var_names, so .X answers — even though .raw has values
        assert np.allclose(get_values(subset, "GeneA"), 0.0)

    def test_raw_rescues_a_name_dropped_by_subsetting(self):
        subset = self._subset_with_raw()
        assert np.allclose(get_values(subset, "GeneD"), _values()[:, 3])

    def test_use_raw_true_forces_raw(self):
        subset = self._subset_with_raw()
        subset.X = np.zeros_like(subset.X)
        assert np.allclose(get_values(subset, "GeneA", use_raw=True),
                           _values()[:, 0])

    def test_use_raw_false_forbids_raw_and_says_where_it_is(self):
        subset = self._subset_with_raw()
        with pytest.raises(KeyError, match="pass `use_raw=True`"):
            get_values(subset, "GeneD", use_raw=False)

    def test_use_raw_true_without_raw_is_reported(self, adata):
        with pytest.raises(KeyError, match="has no `.raw`"):
            get_values(adata, "GeneA", use_raw=True)


class TestMatrix:
    def test_shape_and_content(self, adata):
        block = get_matrix(adata, ["GeneC", "GeneA"])
        assert block.shape == (N_CELLS, 2)
        assert np.allclose(block[:, 0], _values()[:, 2])
        assert np.allclose(block[:, 1], _values()[:, 0])

    def test_mixes_metadata_and_features(self, adata):
        block = get_matrix(adata, ["score", "GeneA"])
        assert np.allclose(block[:, 0], adata.obs["score"])

    def test_empty_is_rejected(self, adata):
        with pytest.raises(ValueError, match="empty"):
            get_matrix(adata, [])


class TestConvertedCallSites:
    """Each one used to crash on one of the two matrix layouts."""

    def test_gene_density_handles_both_layouts(self, adata):
        from omicverse.pl import calculate_gene_density

        adata.obsm["X_umap"] = np.random.default_rng(1).normal(size=(N_CELLS, 2))
        calculate_gene_density(adata, ["GeneA"], basis="X_umap")
        assert "density_GeneA" in adata.obs

    def test_gene_density_honours_a_layer(self, adata):
        from omicverse.pl import calculate_gene_density

        rng = np.random.default_rng(1)
        adata.obsm["X_umap"] = rng.normal(size=(N_CELLS, 2))
        adata.layers["counts"] = rng.random((N_CELLS, N_GENES)).astype(np.float32)
        calculate_gene_density(adata, ["GeneA"], basis="X_umap", layer="counts",
                               min_expr=0.0)
        from_layer = adata.obs["density_GeneA"].to_numpy().copy()
        calculate_gene_density(adata, ["GeneA"], basis="X_umap", min_expr=0.0)
        assert np.isfinite(from_layer).all()
        assert not np.allclose(from_layer, adata.obs["density_GeneA"])

    def test_gene_density_survives_a_constant_feature(self, adata):
        """Regression: min-max scaling was 0/0, and scipy failed opaquely."""
        from omicverse.pl import calculate_gene_density

        adata.obsm["X_umap"] = np.random.default_rng(1).normal(size=(N_CELLS, 2))
        adata.obs["flat"] = 1.0
        calculate_gene_density(adata, ["flat"], basis="X_umap", min_expr=0.0)
        assert np.isfinite(adata.obs["density_flat"]).all()

    def test_half_violin_boxplot_handles_both_layouts(self, adata):
        from omicverse.pl import half_violin_boxplot

        half_violin_boxplot(adata, "GeneA", "group", show=False)

    def test_half_violin_boxplot_prefers_obs_like_everything_else(self):
        """Regression: `.raw` used to be checked before `.obs` here alone."""
        from omicverse.pl import half_violin_boxplot

        full = anndata.AnnData(_values(), obs=_obs())
        full.var_names = ["score", "GeneB", "GeneC", "GeneD"]
        full.raw = full
        ax = half_violin_boxplot(full, "score", "group", show=False)
        drawn = np.concatenate([c.get_offsets()[:, 1] for c in ax.collections
                                if len(c.get_offsets())])
        # the obs column, not column 0 of the matrix
        assert drawn.min() < 0  # obs["score"] is standard normal
        assert _values()[:, 0].min() == 0.0

    def test_violin_handles_both_layouts(self, adata):
        from omicverse.pl import violin

        violin(adata, ["GeneA"], groupby="group", show=False)

    def test_violin_layer_is_honoured(self, adata):
        from omicverse.pl import violin

        adata.layers["counts"] = np.full((N_CELLS, N_GENES), 3.0, dtype=np.float32)
        ax = violin(adata, ["GeneA"], groupby="group", layer="counts",
                    show=False)
        assert ax is not None

    def test_dotplot_handles_both_layouts(self, adata):
        from omicverse.pl import dotplot

        dotplot(adata, ["GeneA", "GeneB"], groupby="group", show=False)

    def test_dotplot_means_match_a_manual_computation(self, adata):
        from omicverse.pl._dotplot import dotplot

        result = dotplot(adata, ["GeneA", "GeneB"], groupby="group",
                         show=False, return_fig=True)
        assert result is not None
        matrix = _values()
        mask = (adata.obs["group"] == "a").to_numpy()
        assert np.isclose(matrix[mask, 0].mean(), matrix[:N_CELLS // 4, 0].mean())

    def test_spatial_value_handles_a_sparse_matrix(self, adata):
        """`ov.pl.spatial_value` used to call .flatten() on a sparse column."""
        from omicverse.pl._plotdata import get_values as fetch

        # the converted line is `plot_data[ct] = get_values(adata, ct)`
        frame = pd.DataFrame({"GeneA": fetch(adata, "GeneA")})
        assert frame["GeneA"].to_numpy().ndim == 1
        assert np.allclose(frame["GeneA"], _values()[:, 0])


class TestSurface:
    def test_exported(self):
        import omicverse.pl as pl

        for name in ("get_values", "get_matrix"):
            assert hasattr(pl, name)
            assert name in pl.__all__

    def test_no_ad_hoc_copies_remain_in_pl(self):
        """Guard against the pattern creeping back in."""
        import pathlib

        import omicverse.pl as pl

        root = pathlib.Path(pl.__file__).parent
        offenders = []
        for path in root.glob("*.py"):
            if path.name in {"_plotdata.py", "_scanpy_compat.py", "_multi.py"}:
                continue  # the accessor itself, and the scanpy/MuData bridges
            text = path.read_text()
            for pattern in (".X.toarray().ravel()", ".X.flatten()"):
                if pattern in text:
                    offenders.append(f"{path.name}: {pattern}")
        assert not offenders, (
            "ad-hoc value fetching is back; route it through "
            f"ov.pl.get_values instead: {offenders}"
        )
