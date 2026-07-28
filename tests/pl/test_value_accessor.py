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


class TestRegistryCoverage:
    """Every public name in the new ov.pl modules must be discoverable.

    The registry is how an agent finds a capability it was not told about, so
    an exported function that is not in it is invisible — which is how
    ``kaplan_meier``, ``get_values`` and eleven others were shipped
    undiscoverable in the first place.
    """

    #: the modules added by the statistics / layout / adapter work
    MODULES = ("_survival", "_classification", "_forest", "_layout",
               "_stats_tests", "_categorical", "_distribution", "_relational",
               "_plotdata")

    def _public_names(self, callables_only: bool = True):
        """Public names of the new modules, keyed to the module they live in.

        ``callables_only`` skips three things, for three different reasons:

        * data constants such as ``JOURNAL_WIDTH_MM`` — the registry indexes
          *capabilities an agent can invoke*, and a dict of column widths is
          not one;
        * classes such as ``PlotData`` and ``ObsView`` — registering a class
          also registers its members, and ``PlotData.embedding`` then takes
          the lookup key that belongs to ``ov.pl.embedding``. They are reached
          through their factory, ``as_plotdata``, which is registered;
        * anything in ``_plot_backend``, which stubs ``register_function`` out
          on purpose because it holds the styling primitives.

        All three must still be exported, which the export test checks.
        """
        import importlib
        import inspect

        names = {}
        for module_name in self.MODULES:
            module = importlib.import_module(f"omicverse.pl.{module_name}")
            for name in getattr(module, "__all__", []):
                target = getattr(module, name, None)
                if callables_only:
                    if not callable(target) or inspect.isclass(target):
                        continue
                names[name] = module_name
        return names

    def test_every_public_name_is_exported(self):
        import omicverse.pl as pl

        missing = [f"{n} ({m})" for n, m
                   in self._public_names(callables_only=False).items()
                   if n not in pl.__all__]
        assert not missing, f"exported from the module but not from ov.pl: {missing}"

    def test_every_public_name_is_registered(self):
        import omicverse.pl  # noqa: F401

        from omicverse._registry import get_registry

        registered = {str(entry.get("full_name")).rsplit(".", 1)[-1]
                      for entry in get_registry().get_by_category("pl")}
        missing = sorted(n for n in self._public_names() if n not in registered)
        assert not missing, (
            "public but undiscoverable — add @register_function to: "
            f"{missing}"
        )

    def test_registered_entries_carry_usable_metadata(self):
        import omicverse.pl  # noqa: F401

        from omicverse._registry import get_registry

        public = set(self._public_names())
        thin = []
        for entry in get_registry().get_by_category("pl"):
            name = str(entry.get("full_name")).rsplit(".", 1)[-1]
            if name not in public:
                continue
            if not entry.get("description") or not entry.get("aliases"):
                thin.append(name)
        assert not thin, f"registered with no description or aliases: {thin}"

    def test_classes_survive_the_decorator(self):
        """``@register_function`` must not turn a class into a function."""
        import omicverse.pl as pl

        frame = pd.DataFrame({"a": [1.0, 2.0], "g": ["x", "y"]})
        assert isinstance(pl.as_plotdata(frame), pl.PlotData)
        assert isinstance(pl.ObsView(frame), pl.ObsView)


class TestMosaicPanelKeys:
    """A panel must be reachable by the tag printed on it, grid or mosaic."""

    def test_grid_keys_are_the_tags(self):
        from omicverse.pl import multipanel

        fig, axes = multipanel((2, 2), width=180, height=120)
        assert {"a", "b", "c", "d"} <= set(axes)
        plt.close(fig)

    def test_mosaic_accepts_both_its_own_key_and_the_printed_tag(self):
        """Regression: `axes['a']` raised KeyError on a mosaic but not a grid."""
        from omicverse.pl import multipanel

        fig, axes = multipanel("AAB\nCDB", width=180, height=95)
        assert {"A", "B", "C", "D"} <= set(axes)
        assert {"a", "b", "c", "d"} <= set(axes)
        # reading order: A spans the top-left, so it carries tag 'a'
        assert axes["a"] is axes["A"]
        plt.close(fig)

    def test_a_mosaic_that_uses_lowercase_keeps_its_own_meaning(self):
        """An alias must never shadow a key the mosaic actually declared."""
        from omicverse.pl import multipanel

        fig, axes = multipanel("ab\ncd", width=180, height=120)
        # 'a' is the caller's own panel, not an alias onto a different one
        assert axes["a"] is not axes["b"]
        assert len({id(v) for v in axes.values()}) == 4
        plt.close(fig)

    def test_labels_off_leaves_grid_tuples(self):
        from omicverse.pl import multipanel

        fig, axes = multipanel((1, 2), width=180, height=60, label=False)
        assert set(axes) == {(0, 0), (0, 1)}
        plt.close(fig)


class TestHouseStyle:
    """The new plots must follow ov.plot_set, not a second style system."""

    @staticmethod
    def _frame():
        rng = np.random.default_rng(3)
        return pd.DataFrame({"g": np.repeat(list("abcd"), 25),
                             "v": rng.normal(size=100),
                             "w": rng.lognormal(size=100)})

    def test_style_axes_applies_the_convention(self):
        import omicverse.pl as pl

        fig, ax = plt.subplots()
        pl.style_axes(ax)
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        for side in ("left", "bottom"):
            assert ax.spines[side].get_position() == ("outward", 10.0)
        plt.close(fig)

    def test_style_axes_can_drop_the_frame_entirely(self):
        import omicverse.pl as pl

        fig, ax = plt.subplots()
        pl.style_axes(ax, spines=False)
        assert not any(ax.spines[s].get_visible() for s in ax.spines)
        plt.close(fig)

    @pytest.mark.parametrize("call", [
        lambda pl, d: pl.barplot(d, "g", "v"),
        lambda pl, d: pl.violinplot(d, "g", "v"),
        lambda pl, d: pl.stripplot(d, "g", "v"),
        lambda pl, d: pl.scatterplot(d, "w", "v"),
        lambda pl, d: pl.histplot(d, "v"),
        lambda pl, d: pl.regplot(d, "w", "v"),
    ])
    def test_generic_plots_use_the_house_frame(self, call):
        import omicverse.pl as pl

        ax = call(pl, self._frame())
        assert not ax.spines["top"].get_visible()
        assert ax.spines["left"].get_position() == ("outward", 10.0)
        plt.close("all")

    @pytest.mark.parametrize("configured", [9, 14, 18])
    def test_font_size_follows_plot_set(self, configured):
        """Regression: the new plots hard-coded fontsize=9 of their own."""
        import omicverse.pl as pl

        with plt.rc_context({"axes.labelsize": configured,
                             "xtick.labelsize": configured - 1,
                             "axes.titlesize": configured + 2}):
            ax = pl.barplot(self._frame(), "g", "v", title="t")
            assert ax.xaxis.label.get_size() == pytest.approx(configured)
            assert ax.title.get_size() == pytest.approx(configured + 2)
        plt.close("all")

    def test_explicit_fontsize_still_overrides(self):
        import omicverse.pl as pl

        with plt.rc_context({"axes.labelsize": 20}):
            ax = pl.barplot(self._frame(), "g", "v", fontsize=6)
            assert ax.xaxis.label.get_size() == pytest.approx(7)  # 6 + 1
        plt.close("all")

    def test_font_size_resolves_named_rcparam_sizes(self):
        import omicverse.pl as pl

        with plt.rc_context({"font.size": 10, "axes.labelsize": "large"}):
            assert pl.font_size(None, "label") > 10


class TestDeprecatedAliases:
    """Legacy names must keep working, warn, and render the modern plot."""

    @staticmethod
    def _frame():
        rng = np.random.default_rng(4)
        return pd.DataFrame({
            "Gene": np.repeat(["GENE1", "GENE2"], 40),
            "Type": np.tile(np.repeat(["Treatment", "Control"], 20), 2),
            "Value": rng.normal(5, 1, 80),
        })

    def test_plot_boxplot_warns_and_forwards(self):
        import omicverse.pl as pl

        with pytest.warns(DeprecationWarning, match="plot_boxplot"):
            fig, ax = pl.plot_boxplot(self._frame(), hue="Type",
                                      x_value="Gene", y_value="Value")
        assert ax is not None
        plt.close("all")

    def test_plot_boxplot_keeps_the_old_hue_order(self):
        """`boxplot` sorts hue levels; the old function did not.

        Flipping them would silently swap Treatment and Control in the
        DESeq2 plots that call this through ov.bulk.
        """
        import omicverse.pl as pl

        with pytest.warns(DeprecationWarning):
            _, ax = pl.plot_boxplot(self._frame(), hue="Type",
                                    x_value="Gene", y_value="Value")
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["Treatment", "Control"]
        plt.close("all")

    def test_explicit_hue_order_still_wins(self):
        import omicverse.pl as pl

        with pytest.warns(DeprecationWarning):
            _, ax = pl.plot_boxplot(self._frame(), hue="Type", x_value="Gene",
                                    y_value="Value",
                                    hue_order=["Control", "Treatment"])
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["Control", "Treatment"]
        plt.close("all")

    @pytest.mark.parametrize("name", ["violin_old", "violin_box"])
    def test_legacy_violins_warn(self, name):
        import omicverse.pl as pl

        adata = pytest.importorskip("anndata").AnnData(
            np.zeros((40, 1), dtype=np.float32),
            obs=pd.DataFrame({"g": np.repeat(list("ab"), 20),
                              "v": np.random.default_rng(0).normal(size=40)}),
        )
        adata.var_names = ["gene"]
        with pytest.warns(DeprecationWarning, match=name):
            getattr(pl, name)(adata, keys="v", groupby="g")
        plt.close("all")


class TestRegistryLookupIsUnambiguous:
    """A name must resolve to the function that owns it.

    The registry keys entries by alias and by short name, so an alias that
    happens to equal another function's real name silently takes the key over
    — asking for ``kaplan_meier`` returned ``survival``, because ``survival``
    listed ``kaplan_meier`` among its aliases. Decorating a *class* had the
    same effect through its members: ``PlotData.embedding`` claimed the key
    ``embedding`` that belongs to ``ov.pl.embedding``.

    These assertions cover the entries this work added. Capability branches
    (``name[method=...]``) are excluded: sharing a method alias across the
    functions that accept it is what that mechanism is for.
    """

    OWNED = (
        "survival", "cumulative_incidence", "kaplan_meier", "logrank_test",
        "aalen_johansen", "grays_test", "roc", "confusion", "roc_auc_ci",
        "forest", "meta_analysis", "compare_groups", "add_stat_annotation",
        "format_pvalue", "barplot", "stripplot", "violinplot", "stackplot",
        "lollipopplot", "pieplot", "donutplot", "slopeplot", "histplot",
        "kdeplot", "ridgeplot", "qqplot", "scatterplot", "lineplot", "regplot",
        "as_plotdata", "accepts_frame", "get_values", "get_matrix",
        "figure", "multipanel", "add_panel_label", "savefig",
        "set_editable_text", "take_legend_out",
    )

    @staticmethod
    def _registry():
        import omicverse.pl  # noqa: F401

        from omicverse._registry import get_registry

        return get_registry()

    def test_each_name_resolves_to_its_own_function(self):
        registry = self._registry()
        wrong = {}
        for name in self.OWNED:
            entry = registry._registry.get(name)
            assert entry is not None, f"{name} is not in the registry at all"
            full = str(entry.get("full_name"))
            if not full.endswith(f".{name}"):
                wrong[name] = full
        assert not wrong, (
            "these names resolve to a different function — an alias is "
            f"shadowing them: {wrong}"
        )

    def test_no_alias_shadows_another_functions_name(self):
        registry = self._registry()
        owned = set(self.OWNED)
        offenders = []
        for entry in registry._registry.values():
            full = str(entry.get("full_name") or "")
            if "[" in full or not full.startswith("omicverse.pl."):
                continue
            short = full.rsplit(".", 1)[-1]
            for alias in entry.get("aliases") or []:
                alias = str(alias).lower()
                if alias in owned and alias != short.lower():
                    offenders.append(f"{short} claims the alias {alias!r}")
        assert not offenders, sorted(set(offenders))

    def test_decorating_a_class_does_not_leak_its_members(self):
        registry = self._registry()
        leaked = sorted({str(e.get("full_name")) for e in registry._registry.values()
                         if ".PlotData." in str(e.get("full_name"))
                         or ".ObsView." in str(e.get("full_name"))})
        assert not leaked, (
            "class members reached the lookup table; register the factory "
            f"function instead of the class: {leaked}"
        )

    def test_the_factory_is_still_discoverable(self):
        """Dropping the class decorators must not hide `as_plotdata`."""
        registry = self._registry()
        hits = {str(e.get("full_name")) for e in registry.find("as_plotdata")}
        assert any(h.endswith("as_plotdata") for h in hits)
