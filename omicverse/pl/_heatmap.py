import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from pandas.api.types import is_numeric_dtype

from .._registry import register_function
from ._scanpy_compat import _prepare_dataframe, default_palette, obs_df

pycomplexheatmap_install = False


def _global_imports(modulename, shortname=None, asfunction=False):
    if shortname is None:
        shortname = modulename
    globals()[shortname] = __import__(modulename)


def check_pycomplexheatmap():
    r"""Check if PyComplexHeatmap is installed and available."""
    global pycomplexheatmap_install
    try:
        import PyComplexHeatmap as pch

        pycomplexheatmap_install = True
        print("PyComplexHeatmap have been install version:", pch.__version__)
    except ImportError:
        raise ImportError("Please install the tangram: `pip install PyComplexHeatmap`.")


@register_function(
    aliases=["复杂热图", "complexheatmap", "complex_heatmap", "PyComplexHeatmap"],
    category="pl",
    description="Create a complex heatmap with annotations using PyComplexHeatmap.",
    examples=[
        "ov.pl.complexheatmap(adata, groupby='cell_type', var_names=marker_genes)",
        "ov.pl.complexheatmap(adata, groupby='leiden', marker_genes_dict={'T cell': ['CD3D'], 'B cell': ['CD19']})",
    ],
    related=["pl.group_heatmap", "pl.marker_heatmap", "pl.dotplot"],
)
def complexheatmap(
    adata,
    groupby="",
    figsize=(6, 10),
    layer: str = None,
    use_raw: bool = False,
    var_names=None,
    gene_symbols=None,
    standard_scale: str = None,
    col_color_bars: dict = None,
    col_color_labels: dict = None,
    left_color_bars: dict = None,
    left_color_labels: dict = None,
    right_color_bars: dict = None,
    right_color_labels: dict = None,
    marker_genes_dict: dict = None,
    index_name: str = "",
    value_name: str = "",
    cmap: str = "parula",
    xlabel: str = None,
    ylabel: str = None,
    label: str = "",
    save: bool = False,
    save_pathway: str = "",
    legend_gap: int = 7,
    legend_hpad: int = 0,
    show: bool = False,
    left_add_text: bool = False,
    col_split_gap: int = 1,
    row_split_gap: int = 1,
    col_height: int = 4,
    left_height: int = 4,
    right_height: int = 4,
    col_cluster: bool = False,
    row_cluster: bool = False,
    row_split=None,
    col_split=None,
    legend: bool = True,
    plot_legend: bool = True,
    right_fontsize: int = 12,
):
    r"""
    Generate a complex annotated heatmap using PyComplexHeatmap.

    Args:
        adata: Annotated data object containing expression data
        groupby: Grouping variable for cell categorization
        figsize: Figure dimensions as (width, height)
        layer: Data layer to use for expression values
        use_raw: Whether to use raw expression data
        var_names: List of genes to include in heatmap
        gene_symbols: Gene symbol column name
        standard_scale: Standardization method - 'obs', 'var', or None
        col_color_bars: Color mapping dictionary for column annotations
        col_color_labels: Color mapping for column labels
        left_color_bars: Color mapping for left annotations
        left_color_labels: Color mapping for left labels
        right_color_bars: Color mapping for right annotations
        right_color_labels: Color mapping for right labels
        marker_genes_dict: Dictionary mapping cell types to marker genes
        index_name: Name for index column in melted data
        value_name: Name for value column in melted data
        cmap: Colormap for heatmap values
        xlabel: X-axis label
        ylabel: Y-axis label
        label: Plot label
        save: Whether to save the plot
        save_pathway: File path for saving
        legend_gap: Gap between legend items
        legend_hpad: Horizontal padding for legend
        show: Whether to display the plot
        left_add_text: Whether to add text to left annotations
        col_split_gap: Gap between column splits
        row_split_gap: Gap between row splits
        col_height: Height of column annotations
        left_height: Height of left annotations
        right_height: Height of right annotations
        col_cluster: Whether to cluster columns
        row_cluster: Whether to cluster rows
        row_split: Row splitting variable
        col_split: Column splitting variable
        legend: Whether to show legend
        plot_legend: Whether to plot legend
        right_fontsize: Font size for right annotations

    Returns:
        cm: PyComplexHeatmap ClusterMapPlotter object
    """
    check_pycomplexheatmap()
    global pycomplexheatmap_install
    if pycomplexheatmap_install:
        _global_imports("PyComplexHeatmap", "pch")

    if layer is not None:
        use_raw = False
    if use_raw:
        adata = adata.raw.to_adata()
        use_raw = False

    if var_names is None:
        var_names = adata.var_names
    if isinstance(var_names, str):
        var_names = [var_names]
    groupby_copy = groupby

    groupby_index = None
    if groupby is not None:
        if isinstance(groupby, str):
            groupby = [groupby]
        for group in groupby:
            if group not in list(adata.obs_keys()) + [adata.obs.index.name]:
                if adata.obs.index.name is not None:
                    msg = f' or index name "{adata.obs.index.name}"'
                else:
                    msg = ""
                raise ValueError(
                    "groupby has to be a valid observation."
                    f"Given {group}, is not in observations: {adata.obs_keys()}" + msg
                )
            if group in adata.obs.keys() and group == adata.obs.index.name:
                raise ValueError(
                    f"Given group {group} is both and index and a column level, "
                    "which is ambiguous."
                )
            if group == adata.obs.index.name:
                groupby_index = group
    if groupby_index is not None:
        groupby = groupby.copy()
        groupby.remove(groupby_index)

    if col_color_bars is None:
        print("Error, please input col_color before run this function.")
    if col_color_labels is None:
        print("Error, please input col_color before run this function.")
    if marker_genes_dict is None:
        print("Error, please input marker_genes_dict before run this function.")

    keys = list(groupby) + list(np.unique(var_names))
    obs_tidy = obs_df(
        adata, keys=keys, layer=layer, use_raw=use_raw, gene_symbols=gene_symbols
    )
    assert np.all(np.array(keys) == np.array(obs_tidy.columns))

    if groupby_index is not None:
        obs_tidy.reset_index(inplace=True)
        groupby.append(groupby_index)

    if groupby is None:
        categorical = pd.Series(np.repeat("", len(obs_tidy))).astype("category")
    elif len(groupby) == 1 and is_numeric_dtype(obs_tidy[groupby[0]]):
        categorical = pd.cut(obs_tidy[groupby[0]], 7)
    elif len(groupby) == 1:
        categorical = obs_tidy[groupby[0]].astype("category")
        categorical.name = groupby[0]
    else:
        categorical = obs_tidy[groupby].apply("_".join, axis=1).astype("category")
        categorical.name = "_".join(groupby)
        from itertools import product

        order = {
            "_".join(k): idx
            for idx, k in enumerate(
                product(*(obs_tidy[g].cat.categories for g in groupby))
            )
        }
        categorical = categorical.cat.reorder_categories(
            sorted(categorical.cat.categories, key=lambda x: order[x])
        )
    obs_tidy = obs_tidy[var_names].set_index(categorical)
    obs_tidy = obs_tidy.groupby(groupby).mean()

    if standard_scale == "obs":
        obs_tidy = obs_tidy.sub(obs_tidy.min(1), axis=0)
        obs_tidy = obs_tidy.div(obs_tidy.max(1), axis=0).fillna(0)
    elif standard_scale == "var":
        obs_tidy -= obs_tidy.min(0)
        obs_tidy = (obs_tidy / obs_tidy.max(0)).fillna(0)

    if right_color_bars is None:
        gene_color_dict = {}
        for cell_type, genes in marker_genes_dict.items():
            cell_type_color = [
                color
                for color, category in zip(
                    adata.uns[groupby_copy + "_colors"],
                    adata.obs[groupby_copy].cat.categories,
                )
                if category == cell_type
            ][0]
            for gene in genes:
                gene_color_dict[gene] = cell_type_color
        right_color_bars = gene_color_dict

    if right_color_labels is None:
        right_color_labels = right_color_bars

    # Avoid duplicate/empty melt column names: when both names are blank (the default),
    # pandas returns a two-column DataFrame for ``loc[:, value_name]`` and the downstream
    # gene index becomes tuples instead of gene names.
    internal_index_name = index_name or "__marker_group__"
    internal_value_name = value_name or "__marker_gene__"
    if internal_index_name == internal_value_name:
        internal_value_name = f"{internal_value_name}__value"

    df_col = obs_tidy.copy()
    df_col[groupby_copy] = df_col.index
    col_ha = pch.HeatmapAnnotation(
        label=pch.anno_label(
            df_col[groupby_copy],
            merge=True,
            rotation=90,
            extend=True,
            colors=col_color_bars,
            adjust_color=True,
            luminance=0.75,
            relpos=(0.5, 0),
        ),
        Celltype=pch.anno_simple(
            df_col[groupby_copy], colors=col_color_labels, height=col_height
        ),
        verbose=1,
        axis=1,
        plot=False,
    )

    marker_genes_df = pd.DataFrame.from_dict(marker_genes_dict, orient="index")
    marker_genes_df = marker_genes_df.transpose()
    melted_df = marker_genes_df.melt(
        var_name=internal_index_name, value_name=internal_value_name
    ).dropna()
    melted_df.index = melted_df.loc[:, internal_value_name]
    df_row = melted_df
    del melted_df

    if left_color_labels is None:
        left_ha = pch.HeatmapAnnotation(
            Marker_Gene=pch.anno_simple(
                df_row[internal_index_name],
                legend=True,
                colors=left_color_bars,
                add_text=left_add_text,
                height=left_height,
            ),
            verbose=1,
            axis=0,
            plot_legend=False,
            plot=False,
        )
    else:
        left_ha = pch.HeatmapAnnotation(
            label=pch.anno_label(
                df_row[internal_index_name],
                merge=True,
                extend=False,
                colors=left_color_labels,
                adjust_color=True,
                luminance=0.75,
                relpos=(1, 0.5),
            ),
            Marker_Gene=pch.anno_simple(
                df_row[internal_index_name],
                legend=True,
                colors=left_color_bars,
                add_text=left_add_text,
                height=left_height,
            ),
            verbose=1,
            axis=0,
            plot_legend=False,
            plot=False,
        )

    if right_color_labels is None:
        right_ha = pch.HeatmapAnnotation(
            Group=pch.anno_simple(
                df_row[internal_index_name],
                legend=True,
                colors=right_color_bars,
                height=right_height,
            ),
            verbose=1,
            axis=0,
            plot_legend=False,
            label_kws=dict(visible=False),
            plot=False,
        )
    else:
        right_ha = pch.HeatmapAnnotation(
            Group=pch.anno_simple(
                df_row[internal_index_name],
                legend=True,
                colors=right_color_bars,
                height=right_height,
            ),
            label=pch.anno_label(
                df_row[internal_value_name],
                merge=True,
                extend=True,
                colors=right_color_labels,
                adjust_color=True,
                luminance=0.75,
                relpos=(0, 0.5),
                fontsize=right_fontsize,
            ),
            verbose=1,
            axis=0,
            plot_legend=False,
            label_kws=dict(visible=False),
            plot=False,
        )

    if row_split is not None:
        row_split_copy = df_row.loc[:, internal_index_name]
    else:
        row_split_copy = row_split
    if col_split is not None:
        col_split_copy = df_col.loc[:, index_name]
    else:
        col_split_copy = col_split

    plt.figure(figsize=figsize)
    obs_copy = obs_tidy.copy().loc[df_col.index.tolist(), df_row.index.tolist()]
    cm = pch.ClusterMapPlotter(
        data=obs_copy.T,
        top_annotation=col_ha,
        left_annotation=left_ha,
        right_annotation=right_ha,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        label=label,
        row_dendrogram=False,
        legend_hgap=legend_gap,
        legend_vgap=legend_gap,
        row_split=row_split_copy,
        col_split=col_split_copy,
        col_split_gap=col_split_gap,
        row_split_gap=row_split_gap,
        row_split_order=list(marker_genes_dict.keys()),
        cmap=cmap,
        rasterized=True,
        xlabel=xlabel,
        legend_hpad=legend_hpad,
        ylabel=ylabel,
        xlabel_kws=dict(color="black", fontsize=14, labelpad=0),
        legend=legend,
        plot_legend=plot_legend,
    )

    if save:
        plt.savefig(save_pathway, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    return cm


@register_function(
    aliases=["marker热图", "marker_heatmap", "标记基因热图", "细胞类型热图"],
    category="pl",
    description="Create a marker-gene dot heatmap using PyComplexHeatmap.",
    examples=[
        "marker_genes = {'T cell': ['CD3D', 'CD3E'], 'B cell': ['CD19', 'MS4A1']}",
        "ov.pl.marker_heatmap(adata, marker_genes_dict=marker_genes, groupby='cell_type')",
    ],
    related=["pl.complexheatmap", "pl.feature_heatmap", "pl.dotplot"],
)
def marker_heatmap(
    adata: AnnData,
    marker_genes_dict: dict = None,
    groupby: str = None,
    color_map: str = "RdBu_r",
    use_raw: bool = True,
    standard_scale: str = "var",
    expression_cutoff: float = 0.0,
    bbox_to_anchor: tuple = (5, -0.5),
    figsize: tuple = (8, 4),
    spines: bool = False,
    fontsize: int = 12,
    show_rownames: bool = True,
    show_colnames: bool = True,
    save_path: str = None,
    ax=None,
):
    r"""
    Create a dot plot heatmap showing marker gene expression using PyComplexHeatmap.

    Args:
        adata: Annotated data object with expression data
        marker_genes_dict: Dictionary mapping cell types to marker genes
        groupby: Column name for cell type grouping
        color_map: Colormap for expression values
        use_raw: Whether to use raw expression data
        standard_scale: Expression standardization method
        expression_cutoff: Minimum expression threshold
        bbox_to_anchor: Legend position
        figsize: Figure dimensions
        spines: Whether to show plot spines
        fontsize: Font size for labels
        show_rownames: Whether to display row names
        show_colnames: Whether to display column names
        save_path: File path for saving plot
        ax: Existing matplotlib axes object

    Returns:
        fig: matplotlib.figure.Figure object
        ax: matplotlib.axes.Axes object
    """
    if marker_genes_dict is None:
        print(
            "Please provide a dictionary containing the marker genes for each cell type."
        )
        return
    if groupby is None:
        print("Please provide a key in adata.obs for grouping the cells.")
        return

    try:
        import PyComplexHeatmap as pch
        from PyComplexHeatmap import (
            DotClustermapPlotter,
            HeatmapAnnotation,
            anno_simple,
            anno_label,
        )

        print("PyComplexHeatmap have been install version:", pch.__version__)
        if pch.__version__ < "1.7.5":
            raise ImportError(
                "Please install PyComplexHeatmap with version > 1.7.5: "
                "`pip install PyComplexHeatmap`."
            )
    except ImportError:
        raise ImportError(
            "Please install PyComplexHeatmap with version > 1.7.5: "
            "`pip install PyComplexHeatmap`."
        )

    if f"{groupby}_colors" in adata.uns:
        type_color_all = dict(
            zip(adata.obs[groupby].cat.categories, adata.uns[f"{groupby}_colors"])
        )
    else:
        if len(adata.obs[groupby].cat.categories) > 28:
            type_color_all = dict(
                zip(
                    adata.obs[groupby].cat.categories,
                    default_palette(len(adata.obs[groupby].cat.categories)),
                )
            )
        else:
            type_color_all = dict(
                zip(
                    adata.obs[groupby].cat.categories,
                    default_palette(len(adata.obs[groupby].cat.categories)),
                )
            )

    var_group_labels = []
    _var_names = []
    start = 0
    for label_name, vars_list in marker_genes_dict.items():
        if isinstance(vars_list, str):
            vars_list = [vars_list]
        _var_names.extend(list(vars_list))
        var_group_labels.append(label_name)

    categories, obs_tidy = _prepare_dataframe(
        adata,
        _var_names,
        groupby=groupby,
        use_raw=use_raw,
        log=False,
        num_categories=7,
        layer=None,
        gene_symbols=None,
    )

    obs_bool = obs_tidy > expression_cutoff
    dot_size_df = (
        obs_bool.groupby(level=0, observed=True).sum()
        / obs_bool.groupby(level=0, observed=True).count()
    )
    dot_color_df = obs_tidy.groupby(level=0, observed=True).mean()
    if standard_scale == "group":
        dot_color_df = dot_color_df.sub(dot_color_df.min(1), axis=0)
        dot_color_df = dot_color_df.div(dot_color_df.max(1), axis=0).fillna(0)
    elif standard_scale == "var":
        dot_color_df -= dot_color_df.min(0)
        dot_color_df = (dot_color_df / dot_color_df.max(0)).fillna(0)

    Gene_list = []
    for celltype in marker_genes_dict.keys():
        for gene in marker_genes_dict[celltype]:
            Gene_list.append(gene)

    df_row = dot_color_df.index.to_frame()
    df_row["Celltype"] = dot_color_df.index
    df_row.set_index("Celltype", inplace=True)
    df_row.columns = ["Celltype_name"]
    df_row = df_row.loc[list(marker_genes_dict.keys()), :]

    df_col = pd.DataFrame()
    for celltype in marker_genes_dict.keys():
        df_col_tmp = pd.DataFrame(index=marker_genes_dict[celltype])
        df_col_tmp["Gene"] = marker_genes_dict[celltype]
        df_col_tmp["Celltype_name"] = celltype
        df_col = pd.concat([df_col, df_col_tmp])
    df_col.columns = ["Gene_name", "Celltype_name"]
    df_col = df_col.loc[Gene_list, :]

    color_df = pd.melt(
        dot_color_df.reset_index(),
        id_vars=groupby,
        var_name="gene",
        value_name="Mean\nexpression\nin group",
    )
    color_df[groupby] = color_df[groupby].astype(str)
    color_df.index = color_df[groupby] + "_" + color_df["gene"]
    size_df = pd.melt(
        dot_size_df.reset_index(),
        id_vars=groupby,
        var_name="gene",
        value_name="Fraction\nof cells\nin group",
    )
    size_df[groupby] = size_df[groupby].astype(str)
    size_df.index = size_df[groupby] + "_" + size_df["gene"]
    color_df["Fraction\nof cells\nin group"] = size_df.loc[
        color_df.index.tolist(), "Fraction\nof cells\nin group"
    ]

    Gene_color = []
    for celltype in df_row.Celltype_name:
        for gene in marker_genes_dict[celltype]:
            Gene_color.append(type_color_all[celltype])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    row_ha = HeatmapAnnotation(
        TARGET=anno_simple(
            df_row.Celltype_name,
            colors=[type_color_all[i] for i in df_row.Celltype_name],
            add_text=False,
            text_kws={"color": "black", "rotation": 0, "fontsize": fontsize},
            legend=False,
        ),
        legend_gap=7,
        axis=0,
        verbose=0,
        label_kws={
            "rotation": 90,
            "horizontalalignment": "right",
            "fontsize": 0,
        },
    )

    col_ha = HeatmapAnnotation(
        TARGET=anno_simple(
            df_col.Gene_name,
            colors=Gene_color,
            add_text=False,
            text_kws={"color": "black", "rotation": 0, "fontsize": fontsize},
            legend=False,
        ),
        verbose=0,
        label_kws={"horizontalalignment": "right", "fontsize": 0},
        legend_kws={"ncols": 1},
        legend=False,
        legend_hpad=7,
        legend_vpad=5,
        axis=1,
    )

    cm = DotClustermapPlotter(
        color_df,
        y=groupby,
        x="gene",
        value="Mean\nexpression\nin group",
        c="Mean\nexpression\nin group",
        s="Fraction\nof cells\nin group",
        cmap=color_map,
        vmin=0,
        top_annotation=col_ha,
        left_annotation=row_ha,
        row_dendrogram=False,
        col_dendrogram=False,
        col_split_order=list(df_col.Celltype_name.unique()),
        col_split=df_col.Celltype_name,
        col_split_gap=1,
        xticklabels_kws={"labelsize": fontsize},
        yticklabels_kws={"labelsize": fontsize},
        dot_legend_kws={
            "fontsize": fontsize,
            "title_fontsize": fontsize,
        },
        color_legend_kws={"fontsize": fontsize},
        x_order=df_col.Gene_name.unique(),
        y_order=df_col.Celltype_name.unique(),
        row_cluster=False,
        col_cluster=False,
        show_rownames=show_rownames,
        show_colnames=show_colnames,
        col_names_side="left",
        spines=spines,
        grid="minor",
        legend=True,
    )

    cm.ax_heatmap.grid(which="minor", color="gray", linestyle="--", alpha=0.5)
    cm.ax_heatmap.grid(which="major", color="black", linestyle="-", linewidth=0.5)
    cm.cmap_legend_kws = {"ncols": 1}
    plt.grid(False)
    plt.tight_layout()

    for ax1 in plt.gcf().axes:
        ax1.grid(False)

    handles = [
        plt.Line2D([0], [0], color=type_color_all[cell], lw=4)
        for cell in type_color_all.keys()
    ]
    labels = type_color_all.keys()
    legend_kws = {
        "fontsize": fontsize,
        "bbox_to_anchor": bbox_to_anchor,
        "loc": "center left",
    }
    plt.legend(
        handles,
        labels,
        borderaxespad=1,
        handletextpad=0.5,
        labelspacing=0.2,
        **legend_kws,
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.tight_layout()
    return fig, ax
