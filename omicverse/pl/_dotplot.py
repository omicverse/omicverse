from __future__ import annotations

from typing import Sequence, Mapping, Literal, Union, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import marsilea as ma
import marsilea.plotter as mp
import warnings
from matplotlib.colors import Normalize, Colormap
from matplotlib.axes import Axes as _AxesSubplot
from anndata import AnnData
import functools
import operator
from typing import Any
from ._palette import palette_28, palette_56
from .._registry import register_function

_VarNames = Union[str, Sequence[str]]

@register_function(
    aliases=["点图", "dotplot", "dot_plot", "表达点图", "基因表达点图"],
    category="pl",
    description="Create dot plot showing gene expression levels and fraction of expressing cells",
    examples=[
        "# Basic dot plot",
        "ov.pl.dotplot(adata, var_names=['CD3D', 'CD8A'], groupby='cell_type')",
        "# Grouped genes",
        "genes = {'T_cells': ['CD3D', 'CD8A'], 'B_cells': ['CD19', 'MS4A1']}",
        "ov.pl.dotplot(adata, var_names=genes, groupby='cell_type')",
        "# Customized dot plot",
        "ov.pl.dotplot(adata, var_names=marker_genes, groupby='leiden',",
        "              standard_scale='var', figsize=(8,6))",
        "# With dendrogram",
        "ov.pl.dotplot(adata, var_names=genes, groupby='cluster',",
        "              dendrogram=True, swap_axes=True)"
    ],
    related=["pl.violin", "pl.heatmap"]
)
def dotplot(
    adata: AnnData,
    var_names: Union[_VarNames, Mapping[str, _VarNames]],
    groupby: Union[str, Sequence[str]],
    *,
    use_raw: Optional[bool] = None,
    log: bool = False,
    num_categories: int = 7,
    categories_order: Optional[Sequence[str]] = None,
    expression_cutoff: float = 0.0,
    mean_only_expressed: bool = False,
    standard_scale: Optional[Literal['var', 'group']] = None,
    title: Optional[str] = None,
    colorbar_title: Optional[str] = 'Mean expression\nin group',
    size_title: Optional[str] = 'Fraction of cells\nin group (%)',
    figsize: Optional[Tuple[float, float]] = None,
    dendrogram: Union[bool, str] = False,
    gene_symbols: Optional[str] = None,
    var_group_positions: Optional[Sequence[Tuple[int, int]]] = None,
    var_group_labels: Optional[Sequence[str]] = None,
    var_group_rotation: Optional[float] = None,
    layer: Optional[str] = None,
    swap_axes: Optional[bool] = False,
    dot_color_df: Optional[pd.DataFrame] = None,
    show: Optional[bool] = None,
    save: Optional[Union[str, bool]] = None,
    ax: Optional[_AxesSubplot] = None,
    return_fig: Optional[bool] = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vcenter: Optional[float] = None,
    norm: Optional[Normalize] = None,
    cmap: Union[Colormap, str, None] = 'Reds',
    dot_max: Optional[float] = None,
    dot_min: Optional[float] = None,
    smallest_dot: float = 0.0,
    fontsize: int = 12,
    preserve_dict_order: bool = False,
    **kwds,
) -> Optional[Union[Dict, 'DotPlot']]:
    r"""
    Make a dot plot of the expression values of `var_names`.
    
    For each var_name and each `groupby` category a dot is plotted.
    Each dot represents two values: mean expression within each category
    (visualized by color) and fraction of cells expressing the `var_name` in the
    category (visualized by the size of the dot).
    
    Args:
        adata: AnnData
            Annotated data matrix.
        var_names: str or list of str or dict
            Variables to plot.
        groupby: str or list of str
            The key of the observation grouping to consider.
        use_raw: bool, optional (default=None)
            Use `raw` attribute of `adata` if present.
        log: bool, optional (default=False)
            Whether to log-transform the data.
        num_categories: int, optional (default=7)
            Number of categories to show.
        categories_order: list of str, optional (default=None)
            Order of categories to display.
        expression_cutoff: float, optional (default=0.0)
            Expression cutoff for calculating fraction of expressing cells.
        mean_only_expressed: bool, optional (default=False)
            Whether to calculate mean only for expressing cells.
        standard_scale: {'var', 'group'} or None, optional (default=None)
            Whether to standardize data.
        title: str, optional (default=None)
            Title for the plot.
        colorbar_title: str, optional (default='Mean expression\nin group')
            Title for the color bar.
        size_title: str, optional (default='Fraction of cells\nin group (%)')
            Title for the size legend.
        figsize: tuple, optional (default=None)
            Figure size (width, height) in inches. If provided, the plot dimensions will be scaled accordingly.
        dendrogram: bool or str, optional (default=False)
            Whether to add dendrogram to the plot.
        gene_symbols: str, optional (default=None)
            Key for gene symbols in `adata.var`.
        var_group_positions: list of tuples, optional (default=None)
            Positions for variable groups.
        var_group_labels: list of str, optional (default=None)
            Labels for variable groups.
        var_group_rotation: float, optional (default=None)
            Rotation angle for variable group labels.
        layer: str, optional (default=None)
            Layer to use for expression data.
        swap_axes: bool, optional (default=False)
            Whether to swap x and y axes.
        dot_color_df: pandas.DataFrame, optional (default=None)
            DataFrame for dot colors.
        show: bool, optional (default=None)
            Whether to show the plot.
        save: str or bool, optional (default=None)
            Whether to save the plot.
        ax: matplotlib.axes.Axes, optional (default=None)
            Axes object to plot on.
        return_fig: bool, optional (default=False)
            Whether to return the figure object.
        vmin: float, optional (default=None)
            Minimum value for color scaling.
        vmax: float, optional (default=None)
            Maximum value for color scaling.
        vcenter: float, optional (default=None)
            Center value for diverging colormap.
        norm: matplotlib.colors.Normalize, optional (default=None)
            Normalization object for colors.
        cmap: str or matplotlib.colors.Colormap, optional (default='Reds')
            Colormap for the plot.
        dot_max: float, optional (default=None)
            Maximum dot size.
        dot_min: float, optional (default=None)
            Minimum dot size.
        smallest_dot: float, optional (default=0.0)
            Size of the smallest dot.
        fontsize: int, optional (default=12)
            Font size for labels and legends. Titles will be one point larger.
        preserve_dict_order: bool, optional (default=False)
            When var_names is a dictionary, whether to preserve the original dictionary order.
            If True, genes will be ordered according to the dictionary's insertion order.
            If False (default), genes will be ordered according to cell type categories.
    
    Returns:
        If `return_fig` is True, returns the figure object.
        If `show` is False, returns axes dictionary.
    """
    marsilea_version = getattr(ma, "__version__", "")
    if marsilea_version.startswith("0.5.6"):
        warnings.warn(
            "marsilea 0.5.6 has an upstream legend regression that can hide "
            "the dotplot colorbar. Please upgrade marsilea to 0.5.7 or newer.",
            UserWarning,
            stacklevel=2,
        )

    # Convert var_names to list if string
    original_var_names_dict = None
    if isinstance(var_names, str):
        var_names = [var_names]
    elif isinstance(var_names, Mapping):
        # Save original dictionary reference for color bar ordering
        if preserve_dict_order:
            original_var_names_dict = var_names
        
        # Get gene groups
        gene_groups = []
        var_names_list = []
        
        if preserve_dict_order:
            # Preserve the original dictionary order
            for group, genes in var_names.items():
                if isinstance(genes, str):
                    genes = [genes]
                var_names_list.extend(genes)
                gene_groups.extend([group] * len(genes))
        else:
            # Get cell type order (original behavior)
            if categories_order is not None:
                group_order = categories_order
            elif pd.api.types.is_categorical_dtype(adata.obs[groupby]):
                group_order = list(adata.obs[groupby].cat.categories)
            else:
                group_order = list(adata.obs[groupby].unique())
            
            # Order gene groups according to cell types
            for group in group_order:
                if group in var_names:
                    genes = var_names[group]
                    if isinstance(genes, str):
                        genes = [genes]
                    var_names_list.extend(genes)
                    gene_groups.extend([group] * len(genes))
            
            # Add any remaining groups that weren't in the cell types
            for group, genes in var_names.items():
                if group not in group_order:
                    if isinstance(genes, str):
                        genes = [genes]
                    var_names_list.extend(genes)
                    gene_groups.extend([group] * len(genes))
        
        var_names = var_names_list
    
    # Get expression matrix
    # Auto-detect if we need to use raw data when use_raw is not explicitly set
    if use_raw is None and adata.raw is not None:
        # Check if any genes are only in raw
        genes_not_in_var = [name for name in var_names if name not in adata.var_names]
        genes_in_raw = [name for name in genes_not_in_var if name in adata.raw.var_names]

        if genes_in_raw:
            use_raw = True
            print(f"Auto-detected: {len(genes_in_raw)} genes only in raw data, using raw data automatically")

    if use_raw and adata.raw is not None:
        matrix = adata.raw.X
        var_names_idx = [adata.raw.var_names.get_loc(name) for name in var_names]
    else:
        matrix = adata.X if layer is None else adata.layers[layer]
        var_names_idx = [adata.var_names.get_loc(name) for name in var_names]
    
    # Determine category order
    if categories_order is not None:
        cats = categories_order
    else:
        # Use the categorical order from adata if available
        if pd.api.types.is_categorical_dtype(adata.obs[groupby]):
            cats = adata.obs[groupby].cat.categories
        else:
            # If not categorical, get unique values
            cats = adata.obs[groupby].unique()
    
    # Get aggregated data with specified order
    agg = adata.obs[groupby].value_counts().reindex(cats)
    cell_counts = agg.to_numpy()
    
    # Get colors for cell types if available
    cell_colors = None
    color_dict = None
    try:
        color_key = f"{groupby}_colors"
        if color_key in adata.uns:
            colors = adata.uns[color_key]
            # Create color dictionary mapping cell types to colors
            if pd.api.types.is_categorical_dtype(adata.obs[groupby]):
                # Use categorical order for colors
                color_dict = dict(zip(adata.obs[groupby].cat.categories, colors))
            else:
                # Use unique order for colors
                unique_cats = adata.obs[groupby].unique()
                color_dict = dict(zip(unique_cats, colors[:len(unique_cats)]))
            
            # Get colors for the actual categories in the plot
            cell_colors = [color_dict.get(cat, '#CCCCCC') for cat in agg.index]
    except (KeyError, IndexError):
        cell_colors = None
        color_dict = None
    
    # Calculate mean expression and fraction of expressing cells
    means = np.zeros((len(agg), len(var_names)))
    fractions = np.zeros_like(means)
    
    for i, group in enumerate(agg.index):
        mask = (adata.obs[groupby] == group).values  # Convert to numpy array
        group_matrix = matrix[mask][:, var_names_idx]
        
        # Calculate mean expression
        if mean_only_expressed:
            expressed = group_matrix > expression_cutoff
            means[i] = np.array([
                group_matrix[:, j][expressed[:, j]].mean() if expressed[:, j].any() else 0
                for j in range(group_matrix.shape[1])
            ])
        else:
            means[i] = np.mean(group_matrix, axis=0)
        
        # Calculate fraction of expressing cells
        fractions[i] = np.mean(group_matrix > expression_cutoff, axis=0)
    
    # Scale if requested
    if standard_scale == 'group':
        means = (means - means.min(axis=1, keepdims=True)) / (means.max(axis=1, keepdims=True) - means.min(axis=1, keepdims=True))
    elif standard_scale == 'var':
        means = (means - means.min(axis=0)) / (means.max(axis=0) - means.min(axis=0))
    
    # Handle dot size limits
    if dot_max is not None:
        fractions = np.minimum(fractions, dot_max)
    if dot_min is not None:
        fractions = np.maximum(fractions, dot_min)
    
    # Scale dot sizes to account for smallest_dot
    if smallest_dot > 0:
        fractions = smallest_dot + (1 - smallest_dot) * fractions
    
    # Create the plot
    h, w = means.shape
    
    # Calculate dimensions based on figsize if provided
    if figsize is not None:
        # Use figsize to determine height and width
        # Adjust for the number of rows and columns to maintain aspect ratio
        base_height = figsize[1] * 0.7  # Use 70% of figsize height for main plot
        base_width = figsize[0] * 0.7   # Use 70% of figsize width for main plot
        
        # Scale based on data dimensions
        height = base_height * (h / max(h, w))
        width = base_width * (w / max(h, w))
    else:
        # Default behavior
        height = h / 3
        width = w / 3
    
    
    # Create SizedHeatmap
    m = ma.SizedHeatmap(
        size=fractions,
        color=means,
        cluster_data=fractions if dendrogram else None,
        height=height,
        width=width,
        edgecolor="lightgray",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        #norm=norm,
        size_legend_kws=dict(
            colors="#c2c2c2",
            title=size_title,
            labels=[f"{int(x*100)}%" for x in [0.2, 0.4, 0.6, 0.8, 1.0]],
            show_at=[0.2, 0.4, 0.6, 0.8, 1.0],
            fontsize=fontsize,
            ncol=3,
            title_fontproperties={"size": fontsize + 1, "weight": 100}
        ),
        color_legend_kws=dict(
            title=colorbar_title,
            fontsize=fontsize,
            orientation="horizontal",
            title_fontproperties={"size": fontsize + 1, "weight": 100}
        ),
    )
    
    # Add labels
    m.add_top(mp.Labels(var_names, fontsize=fontsize), pad=0.1)
    
    # Group genes if var_names was a dictionary
    if 'gene_groups' in locals():
        # Get colors for gene groups
        try:
            # Use the same color_dict that was created for cell types
            if color_dict is not None:
                # Get unique groups and check which ones are not in color_dict
                unique_groups = list(dict.fromkeys(gene_groups))
                missing_groups = [g for g in unique_groups if g not in color_dict]
                
                # If there are missing groups, add colors from palette
                if missing_groups:
                    if len(palette_28) >= len(color_dict) + len(missing_groups):
                        extra_colors = palette_28[len(color_dict):len(color_dict) + len(missing_groups)]
                    else:
                        extra_colors = palette_56[len(color_dict):len(color_dict) + len(missing_groups)]
                    color_dict.update(dict(zip(missing_groups, extra_colors)))
            else:
                # If no colors found in uns, use default palette
                unique_groups = list(dict.fromkeys(gene_groups))
                if len(unique_groups) <= 28:
                    palette = palette_28
                else:
                    palette = palette_56
                color_dict = dict(zip(unique_groups, palette[:len(unique_groups)]))
        except (KeyError, AttributeError):
            # If colors not found in uns, use default palette
            unique_groups = list(dict.fromkeys(gene_groups))
            if len(unique_groups) <= 28:
                palette = palette_28
            else:
                palette = palette_56
            color_dict = dict(zip(unique_groups, palette[:len(unique_groups)]))
        
        # Add color bars with matching order
        # Add group labels
        # Only show used colors in legend and increase group spacing
        if preserve_dict_order and original_var_names_dict is not None:
            # When preserving dict order, use the original dictionary key order
            used_groups = list(original_var_names_dict.keys())
        else:
            # Use the order as they appear in gene_groups
            used_groups = list(dict.fromkeys(gene_groups))

        # Ensure all used_groups have colors in color_dict
        missing_in_used = [g for g in used_groups if g not in color_dict]
        if missing_in_used:
            # Determine which palette to use based on total unique groups
            all_unique_groups = list(dict.fromkeys(gene_groups))
            if len(all_unique_groups) <= 28:
                palette = palette_28
            else:
                palette = palette_56
            # Add missing colors
            start_idx = len(color_dict)
            for i, group in enumerate(missing_in_used):
                color_dict[group] = palette[(start_idx + i) % len(palette)]

        used_color_dict = {k: color_dict[k] for k in used_groups}
        m.add_top(
            mp.Colors(gene_groups, palette=used_color_dict),
            pad=0.1,
            size=0.15,
        )
        # Add group labels with increased spacing
        m.group_cols(gene_groups,order=used_groups)
    
    # Add cell type colors if available
    if color_dict is not None:
        # Add color bar using the properly created color_dict
        m.add_left(
            mp.Colors(agg.index, palette=color_dict),
            size=0.15,
            pad=0.1,
            legend=False,
        )
    
    # Add cell type labels
    m.add_left(mp.Labels(agg.index, align="right", fontsize=fontsize), pad=0.1)
    
    # Add cell counts
    m.add_right(
        mp.Numbers(
            cell_counts,
            color="#EEB76B",
            label="Count",
            label_props={'size': fontsize},
            props={'size': fontsize},
            show_value=False
        ),
        size=0.5,
        pad=0.1,
    )
    
    # Add dendrogram if requested
    if dendrogram:
        m.add_dendrogram("right", pad=0.1)
    
    # Add legends
    m.add_legends(box_padding=2)
    
    # Render the plot
    fig = m.render()
    
    if return_fig:
        return fig
    elif not show:
        return m
    return None

def rank_genes_groups_df(
    adata: AnnData,
    group: str,
    key: str = "rank_genes_groups",
    gene_symbols: str | None = None,
    log2fc_min: float | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with the results of rank_genes_groups."""
    import numpy as np

    result = adata.uns[key]

    # Resolve group index (needed for plain 2D arrays stored by cosg)
    names_data = result.get("names")
    group_index: int | None = None
    if names_data is not None:
        if isinstance(names_data, pd.DataFrame):
            cols = names_data.columns.tolist()
        elif hasattr(names_data, "dtype") and names_data.dtype.names:
            cols = list(names_data.dtype.names)
        else:
            cols = []
        if group in cols:
            group_index = cols.index(group)

    def _extract(data, grp: str, gi: int | None):
        """Extract the column for `grp` from various storage formats."""
        if isinstance(data, pd.DataFrame):
            return data[grp].values if grp in data.columns else None
        if hasattr(data, "dtype") and data.dtype.names:
            return data[grp] if grp in data.dtype.names else None
        # Plain 2D ndarray — use positional index
        if gi is not None and isinstance(data, np.ndarray) and data.ndim == 2:
            return data[:, gi]
        return None

    cols = {}
    for k in ['names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj']:
        if k in result:
            col = _extract(result[k], group, group_index)
            if col is not None:
                cols[k] = col
    if cols:
        lengths = {k: len(v) for k, v in cols.items()}
        if "names" in lengths:
            names_len = lengths["names"]
            shorter = {k: v for k, v in lengths.items() if v < names_len}
            if shorter:
                raise ValueError(
                    f"Inconsistent rank_genes_groups result lengths for group {group!r}: "
                    f"{lengths}"
                )
            # Longer columns may come from a different full-gene result table (for
            # example COSG markers paired with Scanpy p-values). Keep only columns
            # that are safely aligned to the marker names.
            cols = {k: v for k, v in cols.items() if len(v) == names_len}
        elif len(set(lengths.values())) != 1:
            raise ValueError(
                f"Inconsistent rank_genes_groups result lengths for group {group!r}: "
                f"{lengths}"
            )
    d = pd.DataFrame(cols)

    if log2fc_min is not None and 'logfoldchanges' in d.columns:
        d = d[d['logfoldchanges'].abs() > log2fc_min]

    return d

def _get_values_to_plot(
    adata: AnnData,
    values_to_plot: str,
    var_names: Sequence[str],
    key: str = 'rank_genes_groups',
    gene_symbols: str | None = None,
) -> pd.DataFrame:
    """Get values to plot from rank_genes_groups results."""
    if values_to_plot not in adata.uns[key]:
        raise ValueError(
            f'The key {values_to_plot} is not available in adata.uns["{key}"]'
        )
    
    # Get the values for each group
    values = pd.DataFrame(adata.uns[key][values_to_plot])
    values.index = pd.DataFrame(adata.uns[key]['names']).iloc[:, 0]
    values = values.loc[var_names]
    
    return values

@register_function(
    aliases=["差异基因点图", "rank_genes_dotplot", "DEG点图", "差异基因可视化"],
    category="pl",
    description="Create dot plot from rank_genes_groups differential expression results",
    examples=[
        "# After running rank_genes_groups, plot top 5 marker genes per cluster",
        "ov.pp.find_markers(adata, groupby='leiden')",
        "ov.pl.rank_genes_groups_dotplot(adata, n_genes=5, groupby='leiden')",
        "# Plot with logfoldchange filter",
        "ov.pl.rank_genes_groups_dotplot(adata, n_genes=5, groupby='leiden', min_logfoldchange=1.0)",
        "# Select specific groups",
        "ov.pl.rank_genes_groups_dotplot(adata, groups=['0', '1', '2'], n_genes=5)",
    ],
    related=["pl.markers_dotplot", "pl.dotplot", "single.find_markers"]
)
def rank_genes_groups_dotplot(
    adata: AnnData,
    plot_type: str = "dotplot",
    *,
    groups: Optional[Union[str, Sequence[str]]] = None,
    n_genes: Optional[int] = None,
    groupby: Optional[str] = None,
    values_to_plot: Optional[str] = None,
    var_names: Optional[Union[Sequence[str], Mapping[str, Sequence[str]]]] = None,
    min_logfoldchange: Optional[float] = None,
    key: Optional[str] = None,
    show: Optional[bool] = None,
    save: Optional[bool] = None,
    return_fig: bool = False,
    gene_symbols: Optional[str] = None,
    **kwds: Any,
) -> Optional[Union[Dict, Any]]:
    """
    Create a dot plot from rank_genes_groups results.
    
    Args:
        adata: AnnData
            Annotated data matrix.
        plot_type: str
            Currently only 'dotplot' is supported.
        groups: str or list of str, optional
            Groups to include in the plot.
        n_genes: int, optional
            Number of genes to include in the plot.
        groupby: str, optional
            Key in `adata.obs` to group by.
        values_to_plot: str, optional
            Key in rank_genes_groups results to plot (e.g. 'logfoldchanges', 'scores').
        var_names: str or list of str or dict, optional
            Variables to include in the plot. Can be:
            - A list of gene names: ['gene1', 'gene2', ...]
            - A dictionary mapping group names to gene lists: {'group1': ['gene1', 'gene2'], 'group2': ['gene3', 'gene4']}
            When a dictionary is provided, genes will be grouped and labeled accordingly in the plot.
        min_logfoldchange: float, optional
            Minimum log fold change to include in the plot.
        key: str, optional
            Key in `adata.uns` to use for rank_genes_groups results.
        show: bool, optional
            Whether to show the plot.
        save: bool, optional
            Whether to save the plot.
        return_fig: bool
            Whether to return the figure object.
        gene_symbols: str, optional
            Key for gene symbols in `adata.var`.
        **kwds: dict
            Additional keyword arguments to pass to dotplot.
    
    Returns:
        If `return_fig` is True, returns the figure object.
        If `show` is False, returns axes dictionary.
    
    Examples:
        >>> # Basic usage with top genes
        >>> ov.pl.rank_genes_groups_dotplot(adata, n_genes=5)
    
        >>> # Using logfoldchanges for coloring
        >>> ov.pl.rank_genes_groups_dotplot(adata, n_genes=5, values_to_plot='logfoldchanges')
    
        >>> # Grouping genes manually
        >>> gene_groups = {
        ...     'Group1': ['gene1', 'gene2'],
        ...     'Group2': ['gene3', 'gene4']
        ... }
        >>> ov.pl.rank_genes_groups_dotplot(adata, var_names=gene_groups)
    """
    if plot_type != "dotplot":
        raise ValueError("Only 'dotplot' is currently supported")
        
    if var_names is not None and n_genes is not None:
        msg = (
            "The arguments n_genes and var_names are mutually exclusive. Please "
            "select only one."
        )
        raise ValueError(msg)

    if key is None:
        key = "rank_genes_groups"

    if groupby is None:
        groupby = str(adata.uns[key]["params"]["groupby"])

    # Handle both DataFrame and numpy structured array
    if groups is None:
        names_data = adata.uns[key]["names"]
        if isinstance(names_data, pd.DataFrame):
            group_names = names_data.columns.tolist()
        else:
            # Assume it's a numpy structured array
            group_names = names_data.dtype.names
    else:
        group_names = groups

    if var_names is not None:
        if isinstance(var_names, Mapping):
            # get a single list of all gene names in the dictionary
            var_names_list = functools.reduce(
                operator.iadd, [list(x) for x in var_names.values()], []
            )
        elif isinstance(var_names, str):
            var_names_list = [var_names]
        else:
            var_names_list = var_names
    else:
        # set n_genes = 10 as default when none of the options is given
        if n_genes is None:
            n_genes = 10

        # dict in which each group is the key and the n_genes are the values
        var_names = {}
        var_names_list = []
        for group in group_names:
            df = rank_genes_groups_df(
                adata,
                group,
                key=key,
                gene_symbols=gene_symbols,
                log2fc_min=min_logfoldchange,
            )

            if gene_symbols is not None:
                df["names"] = df[gene_symbols]

            genes_list = df.names[df.names.notnull()].tolist()

            if len(genes_list) == 0:
                print(f"Warning: No genes found for group {group}")
                continue
            genes_list = genes_list[n_genes:] if n_genes < 0 else genes_list[:n_genes]
            var_names[group] = genes_list
            var_names_list.extend(genes_list)

    # by default add dendrogram to plots
    kwds.setdefault("dendrogram", True)

    # Auto-detect if we need to use raw data
    # Check if genes are in adata.var_names or only in adata.raw.var_names
    if 'use_raw' not in kwds and adata.raw is not None:
        genes_in_var = [g for g in var_names_list if g in adata.var_names]
        genes_in_raw = [g for g in var_names_list if g in adata.raw.var_names]

        # If some genes are only in raw but not in var, use raw
        if len(genes_in_raw) > len(genes_in_var):
            kwds['use_raw'] = True
            print(f"Auto-detected: {len(genes_in_raw) - len(genes_in_var)} genes only in raw data, using use_raw=True")
        elif len(genes_in_var) < len(var_names_list):
            # Some genes are missing entirely
            missing_genes = [g for g in var_names_list if g not in adata.var_names and (adata.raw is None or g not in adata.raw.var_names)]
            if missing_genes:
                print(f"Warning: {len(missing_genes)} genes not found in adata: {missing_genes[:5]}")

    # Get values to plot if specified
    title = None
    values_df = None
    if values_to_plot is not None:
        values_df = _get_values_to_plot(
            adata,
            values_to_plot,
            var_names_list,
            key=key,
            gene_symbols=gene_symbols,
        )
        title = values_to_plot
        if values_to_plot == "logfoldchanges":
            title = "log fold change"
        else:
            title = values_to_plot.replace("_", " ").replace("pvals", "p-value")

    # Create the plot
    _pl = dotplot(
        adata,
        var_names,
        groupby,
        dot_color_df=values_df,
        return_fig=True,
        gene_symbols=gene_symbols,
        preserve_dict_order=True,
        **kwds,
    )
    
    if title is not None and "colorbar_title" not in kwds:
        _pl.legend(colorbar_title=title)
    
    if return_fig:
        return _pl
    elif not show:
        return _pl
    return None


@register_function(
    aliases=["标记基因点图", "markers_dotplot", "marker点图", "marker_dotplot", "markers_dot"],
    category="pl",
    description="Create a dot plot from marker genes — a clean drop-in replacement for rank_genes_groups_dotplot",
    examples=[
        "# Basic usage after find_markers",
        "ov.single.find_markers(adata, groupby='leiden')",
        "ov.pl.markers_dotplot(adata, groupby='leiden')",
        "",
        "# Custom gene count and colormap",
        "ov.pl.markers_dotplot(adata, groupby='leiden', n_genes=5, cmap='RdBu_r')",
        "",
        "# Use a custom key (e.g. from cosg with key_added)",
        "ov.pl.markers_dotplot(adata, groupby='leiden', key='leiden_cosg', n_genes=3)",
        "",
        "# Standard-scale by variable and disable dendrogram",
        "ov.pl.markers_dotplot(adata, groupby='leiden', standard_scale='var',",
        "                      dendrogram=False, n_genes=5)",
    ],
    related=["pl.rank_genes_groups_dotplot", "pl.dotplot", "single.find_markers", "single.get_markers"],
)
def markers_dotplot(
    adata: AnnData,
    groupby: Optional[str] = None,
    key: Optional[str] = None,
    n_genes: int = 5,
    groups: Optional[Union[str, Sequence[str]]] = None,
    standard_scale: Optional[Literal["var", "group"]] = "var",
    cmap: Union[Colormap, str, None] = "Spectral_r",
    dendrogram: bool = False,
    min_logfoldchange: Optional[float] = None,
    use_raw: Optional[bool] = None,
    layer: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    show: Optional[bool] = None,
    save: Optional[Union[str, bool]] = None,
    return_fig: bool = False,
    **kwds,
) -> Optional[Any]:
    r"""Dot plot of marker genes — clean drop-in for :func:`rank_genes_groups_dotplot`.

    Wraps :func:`rank_genes_groups_dotplot` with more convenient defaults:
    ``standard_scale='var'``, ``cmap='Spectral_r'``, and ``dendrogram=False``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    groupby : str or None, default=None
        Key in ``adata.obs`` to group by. If ``None``, uses
        ``adata.uns[key]['params']['groupby']``.
    key : str or None, default=None
        Key in ``adata.uns`` containing ``rank_genes_groups`` results.
    n_genes : int, default=5
        Number of top marker genes shown per group.
    groups : str or Sequence[str] or None, default=None
        Subset of groups to include.
    standard_scale : {'var', 'group'} or None, default='var'
        Standardization mode for color values.
    cmap : Colormap or str or None, default='Spectral_r'
        Colormap for mean expression.
    dendrogram : bool, default=False
        Whether to include a dendrogram.
    min_logfoldchange : float or None, default=None
        Minimum log-fold-change threshold for marker inclusion.
    use_raw : bool or None, default=None
        Whether to use ``adata.raw`` values.
    layer : str or None, default=None
        Layer used for expression values.
    figsize : tuple[float, float] or None, default=None
        Figure size in inches.
    show : bool or None, default=None
        Whether to display the figure.
    save : str or bool or None, default=None
        Save option/path.
    return_fig : bool, default=False
        Whether to return the generated plot object.
    **kwds
        Additional keyword arguments forwarded to :func:`dotplot`.

    Returns:
        Figure or axes object when ``return_fig=True`` or ``show=False``;
        ``None`` otherwise.

    Examples:
        >>> import omicverse as ov
        >>> ov.single.find_markers(adata, groupby='leiden', method='cosg')
        >>> ov.pl.markers_dotplot(adata, groupby='leiden', n_genes=5)
        >>> # Using wilcoxon results stored under a different key
        >>> ov.single.find_markers(adata, groupby='leiden', method='wilcoxon',
        ...                        key_added='wilcoxon_markers')
        >>> ov.pl.markers_dotplot(adata, groupby='leiden', key='wilcoxon_markers')
    """
    return rank_genes_groups_dotplot(
        adata,
        groups=groups,
        n_genes=n_genes,
        groupby=groupby,
        key=key,
        standard_scale=standard_scale,
        cmap=cmap,
        dendrogram=dendrogram,
        min_logfoldchange=min_logfoldchange,
        use_raw=use_raw,
        layer=layer,
        figsize=figsize,
        show=show,
        save=save,
        return_fig=return_fig,
        **kwds,
    )
