r"""
Plotting utilities for omics data visualization.

This module provides comprehensive plotting functions for visualizing various types
of omics data including single-cell, bulk, spatial transcriptomics, multi-omics,
and network data. It offers publication-ready plots with customizable aesthetics.

Visualization categories:
    Single-cell plots: UMAP, t-SNE, PCA, violin plots, dot plots
    Bulk data plots: Heatmaps, volcano plots, MA plots, PCA plots
    Spatial plots: Spatial feature plots, domain visualization
    Multi-omics plots: Factor plots, integration visualizations
    Network plots: Protein-protein interactions, pathway networks
    General plots: Heatmaps, scatter plots, bar plots
    Clinical plots: Kaplan-Meier survival, competing-risk incidence, forest
    Model evaluation: ROC curves with DeLong CIs, confusion matrices
    Figure assembly: journal-sized canvases, labelled panels, editable-text export

Key modules:
    _single: Single-cell specific plotting functions
    _bulk: Bulk RNA-seq plotting functions
    _space: Spatial transcriptomics visualizations
    _multi: Multi-omics integration plots
    _heatmap: Heatmap and clustering visualizations
    _general: General-purpose plotting utilities
    _palette: Color palette and aesthetic functions
    _cpdb: Cell-cell communication network plots
    _flowsig: Flow cytometry-style visualizations
    _embedding: Dimensionality reduction visualizations
    _density: Density and distribution plots
    _categorical: barplot/stripplot/violinplot/stackplot/pie/donut/slope (table in)
    _distribution: histplot/kdeplot/ridgeplot/qqplot
    _relational: scatterplot/lineplot/regplot
    _stats_tests: compare_groups + significance brackets
    _plotdata: get_values / get_matrix — the one place a name becomes numbers;
        as_plotdata / accepts_frame — AnnData plots that also take a table
    _survival: Kaplan-Meier and Aalen-Johansen curves, log-rank / Gray's test
    _classification: ROC curves and confusion matrices
    _forest: forest plots and fixed/random-effects meta-analysis
    _layout: figure(), multipanel(), savefig() with Illustrator-editable text
    _panelflow: panelflow() — panels declared at their finished axes size,
        decorations measured at draw time and reserved outside the box
    _funkyheatmap: dynbenchmark-style benchmark / multi-metric heatmaps
        (funky rectangles + circles + bars + pies + text + image glyphs,
        wraps the pyfunkyheatmap PyPI package)

Features:
    - Publication-ready figures with customizable aesthetics
    - Integration with matplotlib and seaborn
    - Support for interactive plots
    - Consistent color schemes and themes
    - Export capabilities for various formats
    - Integration with AnnData objects

Examples:
    >>> import omicverse as ov
    >>> # Single-cell visualization
    >>> ov.pl.embedding(adata, basis='umap', color='cell_type')
    >>> ov.pl.violin(adata, keys=['CD3D', 'CD8A'], groupby='cell_type')
    >>> 
    >>> # Bulk data visualization  
    >>> ov.pl.volcano(deg_results, pval_threshold=0.05)
    >>> ov.pl.heatmap(adata, var_names=marker_genes)
    >>> 
    >>> # Spatial visualization
    >>> ov.pl.spatial(adata, color='total_counts')
    >>> ov.pl.spatial_domains(adata, color='domain')
    >>>
    >>> # Benchmark / multi-metric tables — dynbenchmark / scIB style
    >>> # (requires the pyfunkyheatmap PyPI package: pip install pyfunkyheatmap)
    >>> import pandas as pd
    >>> bench = pd.DataFrame({
    ...     'id':       ['UMAP', 't-SNE', 'PHATE', 'PCA'],
    ...     'accuracy': [0.83, 0.71, 0.94, 0.62],
    ...     'speed':    [0.42, 0.30, 0.20, 0.91],
    ...     'memory':   [0.60, 0.55, 0.85, 0.30],
    ... })
    >>> fh = ov.pl.funky_heatmap(bench)
    >>> fh.save('benchmark.png', dpi=150)
"""
from warnings import warn

from ._palette import (
    ForbiddenCity,
    Forbidden_Cmap,
    Forbiddencity,
    blue_color,
    cet_g_bw,
    colormaps_palette,
    earth_palette,
    get_forbidden,
    green_color,
    optim_palette,
    orange_color,
    palette_112,
    palette_28,
    palette_56,
    pastel_palette,
    purple_color,
    red_color,
    sc_color,
    sync_categorical_palette,
    vibrant_palette,
    palplot,
)
from ._single import (
    ConvexHull,
    add_arrow,
    bardotplot,
    cellproportion,
    cellstackarea,
    contour,
    dotplot_doublegroup,
    embedding,
    embedding_adjust,
    embedding_celltype,
    embedding_density,
    mde,
    half_violin_boxplot,
    pca,
    plot_boxplots,
    single_group_boxplot,
    tsne,
    umap,
    violin_box,
    violin_old,
)
from ._dynamic_trends import dynamic_trends, plot_gam_trends
from ._branch_streamplot import (
    branch_streamplot,
    compute_group_kde_profiles,
    make_branch_centerline,
    sigmoid_curve,
    tapered_kde,
)
from ._trajectory import (
    cell_fate,
    cellrank_macrostates,
    plot_stream,
    trajectory,
    trajectory_graph,
    trajectory_overlay,
    trajectory_projection,
    trajectory_tree,
)
from ._general import (
    add_palue,
    create_transparent_gradient_colormap,
)
from ._heatmap import (
    check_pycomplexheatmap,
    complexheatmap,
    marker_heatmap,
    pycomplexheatmap_install,
)
from ._heatmap_marsilea import (
    cell_cor_heatmap,
    dynamic_heatmap,
    feature_heatmap,
    global_imports,
    group_heatmap,
)
from ._multi import embedding_multi
from ._bulk import boxplot, plot_grouped_fractions, venn, volcano
from ._upset import upset
from ._space import (
    add_pie2spatial,
    add_pie_charts_to_spatial,
    create_colormap,
    get_rgb_function,
    html_to_rgb,
    plot_spatial,
    plot_spatial_general,
    rgb_to_ryb,
    ryb_to_rgb,
    spatial_value,
    to_rgb_grayscale,
)
from ._cpdb import (
    cpdb_chord,
    cpdb_group_heatmap,
    cpdb_heatmap,
    cpdb_interacting_heatmap,
    cpdb_interacting_network,
    cpdb_network,
    curved_graph as cpdb_curved_graph,
    curved_line as cpdb_curved_line,
    plot_curve_network as cpdb_plot_curve_network,
)
from ._flowsig import (
    curved_graph as flowsig_curved_graph,
    curved_line as flowsig_curved_line,
    plot_curve_network as flowsig_plot_curve_network,
    plot_flowsig_network,
)
from ._embedding import embedding_atlas
from ._cnv import cnv_heatmap, cnv_summary, cnv_umap
from ._metabolism import metabolism_heatmap
from ._metacell import (
    metacell_metrics,
    metacell_purity_box,
    rigor_scatter,
    metacell_codebook_umap,
    metacell_soft_heatmap,
)
from ._perturbation import (
    perturbation_shift_violin,
    perturbation_embedding_shift,
    perturbation_top_downstream_genes,
    perturb_quiver,
    perturb_cell_quiver,
    perturb_sankey,
    perturb_volcano,
    perturb_inner_product_on_grid,
    perturb_development_layout,
    perturb_celloracle_layout,
    perturb_markov_endpoints,
)
from ._density import add_density_contour, calculate_gene_density
from ._plot1cell import plot1cell
from ._cpdbviz import CellChatViz
from ._ccc import ccc_heatmap, ccc_network_plot, ccc_stat_plot
from ._dotplot import dotplot, rank_genes_groups_dotplot, rank_genes_groups_df, markers_dotplot
from ._spatial import spatial, spatial_segment, spatial_segment_overlay
from ._spatialseg import (
    create_custom_colormap,
    highlight_spatial_region,
    spatialseg,
)
from ._nanostring import nanostring, nanostringseg
from ._violin import violin
from ._qc import qc
from ._report import (
    auto_resolution_curve,
    champ_landscape,
    cluster_sizes_bar,
    doublet_score_histogram,
    highly_variable_genes_scatter,
    neighbor_degree_histogram,
)
from ._animation_lines import (
    Streamlines,
    add_streamplot,
    animate_streamplot,
    compute_velocity_on_grid,
    nan_helper,
)
from ._plot_backend import (
    palette,
    plot_set,
    plotset,
    style_axes,
    ov_plot_set,
    style,
    plot_text_set,
    ticks_range,
    plot_boxplot,
    plot_network,
    plot_cellproportion,
    plot_embedding_celltype,
    geneset_wordcloud,
    plot_pca_variance_ratio,
    plot_pca_variance_ratio1,
    plot_ConvexHull,
    stacking_vol,
    gen_mpl_labels,
)
from ._funkyheatmap import (
    funky_heatmap,
    position_arguments as funky_position_arguments,
    scale_minmax as funky_scale_minmax,
)
from ._layout import (
    EDITABLE_TEXT_RCPARAMS,
    JOURNAL_WIDTH_MM,
    add_panel_label,
    figure,
    multipanel,
    savefig,
    set_editable_text,
    take_legend_out,
)
from ._panelflow import (
    PanelFlow,
    panelflow,
)
from ._survival import (
    aalen_johansen,
    cumulative_incidence,
    grays_test,
    kaplan_meier,
    logrank_test,
    survival,
)
from ._classification import confusion, roc, roc_auc_ci
from ._forest import forest, meta_analysis
from ._stats_tests import add_stat_annotation, compare_groups, format_pvalue
from ._categorical import (
    barplot,
    donutplot,
    lollipopplot,
    pieplot,
    sankey,
    slopeplot,
    stackplot,
    stripplot,
    violinplot,
)
from ._distribution import histplot, kdeplot, qqplot, ridgeplot
from ._relational import lineplot, regplot, scatterplot
from ._plotdata import (ObsView, PlotData, accepts_frame, as_plotdata,
                        get_matrix, get_values)
from ._stats_common import font_size, font_sizes, kde_curve


def curved_graph(*args, **kwargs):
    warn(
        "`ov.pl.curved_graph` is deprecated and ambiguous; use `ov.pl.flowsig_curved_graph` "
        "or `ov.pl.cpdb_curved_graph` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return flowsig_curved_graph(*args, **kwargs)


def curved_line(*args, **kwargs):
    warn(
        "`ov.pl.curved_line` is deprecated and ambiguous; use `ov.pl.flowsig_curved_line` "
        "or `ov.pl.cpdb_curved_line` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return flowsig_curved_line(*args, **kwargs)


def plot_curve_network(*args, **kwargs):
    warn(
        "`ov.pl.plot_curve_network` is deprecated and ambiguous; use `ov.pl.flowsig_plot_curve_network` "
        "or `ov.pl.cpdb_plot_curve_network` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return flowsig_plot_curve_network(*args, **kwargs)

# Explicit public exports for stable, non-wildcard imports
__all__ = [
    # @ _palette
    "ForbiddenCity",
    "Forbidden_Cmap",
    "Forbiddencity",
    "blue_color",
    "cet_g_bw",
    "colormaps_palette",
    "earth_palette",
    "get_forbidden",
    "green_color",
    "optim_palette",
    "orange_color",
    "palette_112",
    "palette_28",
    "palette_56",
    "pastel_palette",
    "dynamic_trends",
    "plot_gam_trends",
    "branch_streamplot",
    "compute_group_kde_profiles",
    "make_branch_centerline",
    "sigmoid_curve",
    "tapered_kde",
    "cell_fate",
    "cellrank_macrostates",
    "plot_stream",
    "trajectory",
    "trajectory_graph",
    "trajectory_overlay",
    "trajectory_projection",
    "trajectory_tree",
    "purple_color",
    "red_color",
    "sc_color",
    "vibrant_palette",
    "palplot",
    # @ _single
    "ConvexHull",
    "add_arrow",
    "bardotplot",
    "cellproportion",
    "cellstackarea",
    "contour",
    "dotplot_doublegroup",
    "embedding",
    "embedding_adjust",
    "embedding_celltype",
    "embedding_density",
    "qc",
    "half_violin_boxplot",
    "mde",
    "pca",
    "plot_boxplots",
    "single_group_boxplot",
    "tsne",
    "umap",
    "violin_box",
    "violin_old",
    # @ _general
    "add_palue",
    "create_transparent_gradient_colormap",
    # @ _heatmap
    "cell_cor_heatmap",
    "check_pycomplexheatmap",
    "complexheatmap",
    "dynamic_heatmap",
    "feature_heatmap",
    "group_heatmap",
    "global_imports",
    "marker_heatmap",
    "pycomplexheatmap_install",
    # @ _multi
    "embedding_multi",
    # @ _bulk
    "boxplot",
    "plot_grouped_fractions",
    "upset",
    "venn",
    "volcano",
    # @ _space
    "add_pie2spatial",
    "add_pie_charts_to_spatial",
    "create_colormap",
    "get_rgb_function",
    "html_to_rgb",
    "plot_spatial",
    "plot_spatial_general",
    "rgb_to_ryb",
    "ryb_to_rgb",
    "spatial_value",
    # @ _cpdb
    "cpdb_chord",
    "cpdb_group_heatmap",
    "cpdb_heatmap",
    "cpdb_interacting_heatmap",
    "cpdb_interacting_network",
    "cpdb_network",
    "cpdb_curved_graph",
    "cpdb_curved_line",
    "cpdb_plot_curve_network",
    # deprecated generic aliases
    "curved_graph",
    "curved_line",
    "plot_curve_network",
    # @ _flowsig
    "flowsig_curved_graph",
    "flowsig_curved_line",
    "flowsig_plot_curve_network",
    "plot_flowsig_network",
    # @ _embedding
    "embedding_atlas",
    # @ _density
    "add_density_contour",
    "calculate_gene_density",
    # @ _cpdbviz
    "CellChatViz",
    # @ _ccc
    "ccc_heatmap",
    "ccc_network_plot",
    "ccc_stat_plot",
    # @ _dotplot
    "dotplot",
    "rank_genes_groups_dotplot",
    "rank_genes_groups_df",
    "markers_dotplot",
    # @ _spatial
    "spatial",
    "spatial_segment",
    "spatial_segment_overlay",
    # @ _spatialseg
    "spatialseg",
    "highlight_spatial_region",
    "create_custom_colormap",
    # @ _nanostring
    "nanostring",
    "nanostringseg",
    # @ _violin
    "violin",
    # @ _report
    "auto_resolution_curve",
    "champ_landscape",
    "cluster_sizes_bar",
    "doublet_score_histogram",
    "highly_variable_genes_scatter",
    "neighbor_degree_histogram",
    # @ _animation_lines
    "Streamlines",
    "add_streamplot",
    "animate_streamplot",
    "compute_velocity_on_grid",
    "nan_helper",
    # @ _plot_backend
    "palette",
    "plot_set",
    "plotset",
    "style_axes",
    "ov_plot_set",
    "style",
    "plot_text_set",
    "ticks_range",
    "plot_boxplot",
    "plot_network",
    "plot_cellproportion",
    "plot_embedding_celltype",
    "geneset_wordcloud",
    "plot_pca_variance_ratio",
    "plot_pca_variance_ratio1",
    "plot_ConvexHull",
    "stacking_vol",
    "gen_mpl_labels",
    # @ _cnv
    "cnv_heatmap",
    "cnv_summary",
    "cnv_umap",
    "metabolism_heatmap",
    # @ _perturbation
    "perturbation_shift_violin",
    "perturbation_embedding_shift",
    "perturbation_top_downstream_genes",
    "perturb_quiver",
    "perturb_cell_quiver",
    "perturb_sankey",
    "perturb_volcano",
    "perturb_inner_product_on_grid",
    "perturb_development_layout",
    "perturb_celloracle_layout",
    "perturb_markov_endpoints",
    # @ _layout — figure geometry, panels, publication export
    "JOURNAL_WIDTH_MM",
    "EDITABLE_TEXT_RCPARAMS",
    "figure",
    "multipanel",
    "add_panel_label",
    "savefig",
    "set_editable_text",
    "take_legend_out",
    # @ _panelflow — panels sized by their axes box, decorations measured
    "panelflow",
    "PanelFlow",
    # @ _survival — time-to-event
    "survival",
    "cumulative_incidence",
    "kaplan_meier",
    "logrank_test",
    "aalen_johansen",
    "grays_test",
    # @ _classification — classifier evaluation
    "roc",
    "confusion",
    "roc_auc_ci",
    # @ _forest — effect sizes and meta-analysis
    "forest",
    "meta_analysis",
    # @ _stats_tests — group comparison and significance brackets
    "compare_groups",
    "add_stat_annotation",
    "format_pvalue",
    # @ _categorical — generic categorical plots (table in, no AnnData)
    "barplot",
    "stripplot",
    "violinplot",
    "stackplot",
    "lollipopplot",
    "pieplot",
    "donutplot",
    "slopeplot",
    "sankey",
    # @ _distribution
    "histplot",
    "kdeplot",
    "ridgeplot",
    "qqplot",
    # @ _relational
    "scatterplot",
    "lineplot",
    "regplot",
    # @ _plotdata — let AnnData-shaped plots take a table
    "as_plotdata",
    "accepts_frame",
    "get_values",
    "get_matrix",
    "font_size",
    "font_sizes",
    "kde_curve",
    "PlotData",
    "ObsView",
]
