import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
from typing import Sequence, Union, Optional, Literal, List, Tuple, Any
from matplotlib.axes import Axes
from collections import OrderedDict
from .._registry import register_function
from ._plotdata import accepts_frame
from ._plotdata import get_values

try:
    from anndata import AnnData
    ANNDATA_AVAILABLE = True
except ImportError:
    # Fallback for when AnnData is not available
    AnnData = object
    ANNDATA_AVAILABLE = False

# Import Colors from settings
try:
    from .._settings import Colors
except Exception:
    # Fallback Colors class if import fails
    class Colors:
        HEADER = '\033[95m'
        BLUE = '\033[94m'
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'


DensityNorm = Literal["area", "count", "width"]

@register_function(
    aliases=["小提琴图", "violin", "violin_plot", "小提琴", "表达分布图"],
    category="pl",
    description="Enhanced violin plot for visualizing gene expression distribution across groups",
    examples=[
        "# Basic violin plot",
        "ov.pl.violin(adata, keys=['CD3D', 'CD8A'], groupby='cell_type')",
        "# Enhanced violin plot with statistics",
        "ov.pl.violin(adata, keys=['GAPDH'], groupby='leiden',",
        "             stripplot=True, statistical_tests=True)",
        "# Multiple genes",
        "ov.pl.violin(adata, keys=['CD3D', 'CD4', 'CD8A'], groupby='cell_type',",
        "             custom_colors=['red', 'blue', 'green'], figsize=(8,6))",
        "# With boxplot overlay",
        "ov.pl.violin(adata, keys=['marker_gene'], groupby='cluster',",
        "             show_boxplot=True, show_means=True)",
        "# With alternating background bands",
        "ov.pl.violin(adata, keys=['CD3D'], groupby='cell_type',",
        "             alternating_background=True)"
    ],
    related=["pl.embedding", "pl.dotplot"]
)
@accepts_frame
def violin(
    adata: AnnData,
    keys: Union[str, Sequence[str]],
    groupby: Optional[str] = None,
    *,
    log: bool = False,
    use_raw: Optional[bool] = None,
    stripplot: bool = True,
    jitter: Union[float, bool] = True,
    size: int = 1,
    layer: Optional[str] = None,
    density_norm: DensityNorm = "width",
    order: Optional[Sequence[str]] = None,
    multi_panel: Optional[bool] = None,
    xlabel: str = "",
    ylabel: Optional[Union[str, Sequence[str]]] = None,
    rotation: Optional[float] = None,
    show: Optional[bool] = None,
    save: Optional[Union[bool, str]] = None,
    ax: Optional[Axes] = None,
    # Enhanced features
    enhanced_style: bool = True,
    show_means: bool = False,
    show_boxplot: bool = False,
    add_box: bool = False,
    comparisons: Optional[List] = None,
    jitter_method: str = 'uniform',  # 'uniform', 't_dist'
    jitter_alpha: float = 0.4,
    violin_alpha: float = 0.8,
    rasterized: bool = False,
    background_color: str = 'white',
    alternating_background: bool = True,
    alternating_background_colors: Sequence[str] = ("white", "#e8e8e8"),
    use_group_colors_background: bool = False,
    background_lighten_amount: float = 0.8,
    spine_color: str = '#b4aea9',
    grid_lines: bool = True,
    statistical_tests: Union[bool, str, List[str]] = False,
    custom_colors: Optional[Sequence[str]] = None,
    figsize: Optional[tuple] = None,
    fontsize=13,
    ticks_fontsize=None,
    verbose: bool = False,
    hue: Optional[str] = None,
    hue_order: Optional[Sequence[str]] = None,
    split: bool = False,
    cut: Optional[float] = None,
    clip: Optional[Tuple[Optional[float], Optional[float]]] = None,
    inner: Optional[str] = None,
    orient: str = "v",
    scale: Optional[str] = None,
    width: float = 0.8,
    palette: Any = None,
    test: Optional[str] = None,
    correction: Optional[str] = "holm",
    annot_style: str = "star",
    hide_ns: bool = False,
    return_stats: bool = False,
    **kwds
) -> Union[Axes, None]:
    r"""
    Enhanced violin plot compatible with omicverse's interface.
    
    This function provides all the functionality of omicverse's violin plot
    with additional customization options for enhanced visualization,
    implemented using pure matplotlib.
    
    Args:
        adata: AnnData. Annotated data matrix.
        keys: str | Sequence[str]. Keys for accessing variables of `.var_names` or fields of `.obs`.
        groupby: str | None. The key of the observation grouping to consider. (None)
        log: bool. Plot on logarithmic axis. (False)
        use_raw: bool | None. Whether to use `raw` attribute of `adata`. Defaults to `True` if `.raw` is present. (None)
        stripplot: bool. Add a stripplot on top of the violin plot. (True)
        jitter: float | bool. Add jitter to the stripplot (only when stripplot is True). (True)
        size: int. Size of the jitter points. (1)
        layer: str | None. Name of the AnnData object layer that wants to be plotted. (None)
        density_norm: str. The method used to scale the width of each violin. If 'width' (the default), each violin will have the same width. If 'area', each violin will have the same area. If 'count', a violin's width corresponds to the number of observations. ('width')
        order: Sequence[str] | None. Order in which to show the categories. (None)
        multi_panel: bool | None. Display keys in multiple panels also when `groupby is not None`. (None)
        xlabel: str. Label of the x axis. ('')
        ylabel: str | Sequence[str] | None. Label of the y axis. (None)
        rotation: float | None. Rotation of xtick labels. (None)
        show: bool | None. Whether to show the plot. (None)
        save: bool | str | None. Path to save the figure. (None)
        ax: Axes | None. A matplotlib axes object. (None)
        enhanced_style: bool. Whether to apply enhanced styling. (True)
        show_means: bool. Whether to show mean values with annotations. (False)
        show_boxplot: bool. Whether to overlay box plots on violins. (False)
        jitter_method: str. Method for jittering: 'uniform' or 't_dist'. ('uniform')
        jitter_alpha: float. Transparency of jittered points. (0.4)
        violin_alpha: float. Transparency of violin plots. (0.8)
        background_color: str. Background color of the plot. ('white')
        alternating_background: bool. Whether to add background bands using each group's lightened color. (True)
        background_lighten_amount: float. How much to lighten each group color for its background band (0=original, 1=white). (0.8)
        spine_color: str. Color of plot spines. ('#b4aea9')
        grid_lines: bool. Whether to show horizontal grid lines. (True)
        statistical_tests: bool | str | List[str]. Statistical tests to perform. Options: False (no tests), True (auto-select), 'wilcox', 'ttest', 'anova', 'kruskal', 'mannwhitney', or list of methods. (False)
        custom_colors: Sequence[str] | None. Custom colors for groups. (None)
        figsize: tuple | None. Figure size (width, height). (None)
        fontsize: int. Font size for labels and ticks. (13)
        ticks_fontsize: int | None. Font size for axis ticks. If None, uses fontsize-1. (None)
        **kwds: Additional keyword arguments passed to violinplot.
    
    Returns:
        ax: matplotlib.axes.Axes | None. A matplotlib axes object if `ax` is `None` else `None`.
    """
    
    # ------------------------------------------------------------------
    # Options the matplotlib engine below cannot express — a mirrored
    # (split) violin, a density clipped at a hard bound, a horizontal
    # layout, per-observation inner marks, or significance brackets. Only
    # those route to the kernel-density renderer; with none of them the
    # original code path runs untouched, so the default figure is
    # unchanged to the pixel.
    # ------------------------------------------------------------------
    _advanced = (hue is not None or split or cut is not None or clip is not None
                 or inner is not None or orient != "v" or scale is not None
                 or test is not None or return_stats)
    if _advanced:
        from ._categorical import _violin_kde

        if isinstance(keys, (list, tuple)):
            if len(keys) != 1:
                raise ValueError(
                    "hue / split / cut / clip / inner / orient / test handle "
                    f"one key at a time, got {len(keys)}. Call violin() once "
                    "per key, or drop those options for the multi-key layout."
                )
            _key = keys[0]
        else:
            _key = keys
        return _violin_kde(
            adata, x=groupby, y=_key, hue=hue, hue_order=hue_order,
            order=order, split=split, inner=inner,
            cut=2.0 if cut is None else cut, clip=clip,
            scale="width" if scale is None else scale, orient=orient,
            width=width, palette=palette if palette is not None else custom_colors,
            ax=ax, figsize=figsize, test=test, correction=correction,
            annot_style=annot_style, hide_ns=hide_ns,
            stripplot=stripplot, rotation=rotation or 0,
            fontsize=fontsize, return_stats=return_stats, **kwds,
        )

    # Handle AnnData availability
    if not ANNDATA_AVAILABLE:
        raise ImportError("AnnData is required for this function. Install with: pip install anndata")
    
    # Ensure keys is a list
    if isinstance(keys, str):
        keys = [keys]
    keys = list(OrderedDict.fromkeys(keys))  # remove duplicates, preserving order
    
    # Handle ylabel
    if isinstance(ylabel, (str, type(None))):
        ylabel = [ylabel] * (1 if groupby is None else len(keys))
    ylabel=keys
    
    # Validate ylabel length
    if groupby is None:
        if len(ylabel) != 1:
            raise ValueError(f"Expected number of y-labels to be `1`, found `{len(ylabel)}`.")
    elif len(ylabel) != len(keys):
        raise ValueError(f"Expected number of y-labels to be `{len(keys)}`, found `{len(ylabel)}`.")
    
    # Extract data from AnnData object
    obs_df = _extract_data_from_adata(adata, keys, groupby, layer, use_raw)
    
    if verbose:
        # Colorful data analysis and parameter optimization suggestions
        print(f"{Colors.HEADER}{Colors.BOLD}🎻 Violin Plot Analysis:{Colors.ENDC}")
        print(f"   {Colors.CYAN}Total cells: {Colors.BOLD}{len(obs_df)}{Colors.ENDC}")
        print(f"   {Colors.BLUE}Variables to plot: {Colors.BOLD}{len(keys)} {keys}{Colors.ENDC}")

        if groupby is not None:
            group_counts = obs_df[groupby].value_counts()
            print(f"   {Colors.GREEN}Groupby variable: '{Colors.BOLD}{groupby}{Colors.ENDC}{Colors.GREEN}' with {Colors.BOLD}{len(group_counts)}{Colors.ENDC}{Colors.GREEN} groups{Colors.ENDC}")

            # Show group distribution
            for group, count in group_counts.head(10).items():  # Show top 10 groups
                if count < 10:
                    color = Colors.WARNING
                elif count < 50:
                    color = Colors.BLUE
                else:
                    color = Colors.GREEN
                print(f"     {color}• {group}: {Colors.BOLD}{count}{Colors.ENDC}{color} cells{Colors.ENDC}")

            if len(group_counts) > 10:
                print(f"     {Colors.CYAN}... and {Colors.BOLD}{len(group_counts) - 10}{Colors.ENDC}{Colors.CYAN} more groups{Colors.ENDC}")

            # Check for imbalanced groups
            min_count = group_counts.min()
            max_count = group_counts.max()
            if max_count / min_count > 10:
                print(f"   {Colors.WARNING}⚠️  Imbalanced groups detected: {Colors.BOLD}{min_count}-{max_count}{Colors.ENDC}{Colors.WARNING} cells per group{Colors.ENDC}")
        else:
            print(f"   {Colors.BLUE}Groupby: {Colors.BOLD}None{Colors.ENDC}{Colors.BLUE} (comparing variables){Colors.ENDC}")

        # Analyze data distribution for each variable
        print(f"\n{Colors.HEADER}{Colors.BOLD}📊 Data Distribution Analysis:{Colors.ENDC}")
        for key in keys:
            if key in obs_df.columns:
                data_vals = obs_df[key].dropna()
                if len(data_vals) > 0:
                    zero_fraction = (data_vals == 0).sum() / len(data_vals)

                    log_suggestion = ""
                    if data_vals.min() >= 0 and data_vals.max() > 100:
                        log_suggestion = f" {Colors.BLUE}(consider log=True){Colors.ENDC}"
                    elif data_vals.min() < 0:
                        log_suggestion = f" {Colors.WARNING}(negative values detected){Colors.ENDC}"

                    print(f"   {Colors.BLUE}'{key}': range {Colors.BOLD}{data_vals.min():.2f}-{data_vals.max():.2f}{Colors.ENDC}{Colors.BLUE}, {Colors.BOLD}{zero_fraction*100:.1f}%{Colors.ENDC}{Colors.BLUE} zeros{log_suggestion}")

        # Display current function parameters
        print(f"\n{Colors.HEADER}{Colors.BOLD}⚙️  Current Function Parameters:{Colors.ENDC}")
        print(f"   {Colors.BLUE}Plot style: enhanced_style={Colors.BOLD}{enhanced_style}{Colors.ENDC}{Colors.BLUE}, stripplot={Colors.BOLD}{stripplot}{Colors.ENDC}{Colors.BLUE}, jitter={Colors.BOLD}{jitter}{Colors.ENDC}")
        print(f"   {Colors.BLUE}Additional features: show_means={Colors.BOLD}{show_means}{Colors.ENDC}{Colors.BLUE}, show_boxplot={Colors.BOLD}{show_boxplot}{Colors.ENDC}{Colors.BLUE}, statistical_tests={Colors.BOLD}{statistical_tests}{Colors.ENDC}")
        print(f"   {Colors.BLUE}Figure settings: figsize={Colors.BOLD}{figsize}{Colors.ENDC}{Colors.BLUE}, fontsize={Colors.BOLD}{fontsize}{Colors.ENDC}{Colors.BLUE}, violin_alpha={Colors.BOLD}{violin_alpha}{Colors.ENDC}")
        if custom_colors is not None:
            print(f"   {Colors.BLUE}Colors: {Colors.BOLD}{len(custom_colors)} custom colors specified{Colors.ENDC}")
        else:
            print(f"   {Colors.BLUE}Colors: {Colors.BOLD}Default palette{Colors.ENDC}")

        # Parameter optimization suggestions
        print(f"\n{Colors.HEADER}{Colors.BOLD}💡 Parameter Optimization Suggestions:{Colors.ENDC}")
        suggestions = []

        if groupby is not None:
            n_groups = len(obs_df[groupby].unique())
            if n_groups > 8:
                suggestions.append(f"   {Colors.WARNING}▶ Many groups detected ({n_groups}):{Colors.ENDC}")
                suggestions.append(f"     {Colors.CYAN}Current: figsize={Colors.BOLD}{figsize}{Colors.ENDC}")
                suggested_width = max(8, n_groups * 1.2)
                suggestions.append(f"     {Colors.GREEN}Suggested: {Colors.BOLD}figsize=({suggested_width}, {figsize[1] if figsize else 6}){Colors.ENDC}")
                if rotation is None:
                    suggestions.append(f"     {Colors.GREEN}Consider adding: {Colors.BOLD}rotation=45{Colors.ENDC} for better label readability")

            group_counts = obs_df[groupby].value_counts()
            if group_counts.min() < 10:
                suggestions.append(f"   {Colors.WARNING}▶ Small sample sizes detected (min: {group_counts.min()}):{Colors.ENDC}")
                suggestions.append(f"     {Colors.GREEN}Suggested: {Colors.BOLD}stripplot=True, jitter=0.3{Colors.ENDC} to show individual points")

        for key in keys:
            if key in obs_df.columns:
                data_vals = obs_df[key].dropna()
                if len(data_vals) > 0 and data_vals.min() >= 0 and data_vals.max() / data_vals.min() > 100:
                    suggestions.append(f"   {Colors.WARNING}▶ Wide data range for '{key}' ({data_vals.max()/data_vals.min():.1f}x):{Colors.ENDC}")
                    suggestions.append(f"     {Colors.CYAN}Current: log={Colors.BOLD}{log}{Colors.ENDC}")
                    suggestions.append(f"     {Colors.GREEN}Suggested: {Colors.BOLD}log=True{Colors.ENDC} for better visualization")
                    break

        if len(keys) > 3 and (figsize is None or figsize[0] < len(keys) * 2):
            suggestions.append(f"   {Colors.WARNING}▶ Multiple variables with small figure:{Colors.ENDC}")
            suggested_width = max(len(keys) * 2, 8)
            suggestions.append(f"     {Colors.CYAN}Current: figsize={Colors.BOLD}{figsize}{Colors.ENDC}")
            suggestions.append(f"     {Colors.GREEN}Suggested: {Colors.BOLD}figsize=({suggested_width}, 6){Colors.ENDC} or {Colors.BOLD}multi_panel=True{Colors.ENDC}")

        if groupby is not None:
            max_label_length = max(len(str(x)) for x in obs_df[groupby].unique())
            if max_label_length > 8 and fontsize > 12:
                suggestions.append(f"   {Colors.WARNING}▶ Long group labels detected:{Colors.ENDC}")
                suggestions.append(f"     {Colors.CYAN}Current: fontsize={Colors.BOLD}{fontsize}{Colors.ENDC}")
                suggestions.append(f"     {Colors.GREEN}Suggested: {Colors.BOLD}fontsize=10, rotation=45{Colors.ENDC}")

        if not show_means and not show_boxplot and not statistical_tests:
            suggestions.append(f"   {Colors.BLUE}▶ Consider enhancing your plot:{Colors.ENDC}")
            suggestions.append(f"     {Colors.GREEN}Options: {Colors.BOLD}show_means=True{Colors.ENDC}{Colors.GREEN}, {Colors.BOLD}show_boxplot=True{Colors.ENDC}{Colors.GREEN}, or {Colors.BOLD}statistical_tests=True{Colors.ENDC}")

        if suggestions:
            for suggestion in suggestions:
                print(suggestion)

            print(f"\n   {Colors.BOLD}📋 Copy-paste ready function call:{Colors.ENDC}")
            optimized_params = []
            if isinstance(keys, list) and len(keys) == 1:
                optimized_params.append(f"adata, keys='{keys[0]}'")
            else:
                optimized_params.append(f"adata, keys={keys}")
            if groupby is not None:
                optimized_params.append(f"groupby='{groupby}'")
                n_groups = len(obs_df[groupby].unique())
                if n_groups > 8:
                    suggested_width = max(8, n_groups * 1.2)
                    optimized_params.append(f"figsize=({suggested_width}, 6)")
                if rotation is None and n_groups > 6:
                    optimized_params.append("rotation=45")
                group_counts = obs_df[groupby].value_counts()
                if group_counts.min() < 10:
                    optimized_params.append("stripplot=True")
                    optimized_params.append("jitter=0.3")
            for key in keys:
                if key in obs_df.columns:
                    data_vals = obs_df[key].dropna()
                    if len(data_vals) > 0 and data_vals.min() >= 0 and data_vals.max() / data_vals.min() > 100:
                        optimized_params.append("log=True")
                        break
            if len(keys) > 3 and (figsize is None or figsize[0] < len(keys) * 2):
                optimized_params.append("multi_panel=True")
            if not show_means and not show_boxplot:
                optimized_params.append("show_means=True")
            print(f"   {Colors.GREEN}ov.pl.violin({', '.join(optimized_params)}){Colors.ENDC}")
        else:
            print(f"   {Colors.GREEN}✅ Current parameters are optimal for your data!{Colors.ENDC}")

        print(f"{Colors.CYAN}{'─' * 60}{Colors.ENDC}")
    
    # Prepare data for plotting
    if groupby is None:
        obs_tidy = pd.melt(obs_df, value_vars=keys)
        x_col = "variable"
        y_cols = ["value"]
        group_categories = keys
    else:
        obs_tidy = obs_df
        x_col = groupby
        y_cols = keys
        obs_df[groupby] = obs_df[groupby].astype('category')
        group_categories = obs_df[groupby].cat.categories if order is None else order
    
    # Set up colors
    colors = _setup_colors(custom_colors, group_categories, adata, groupby)
    
    # Handle multi-panel case
    if multi_panel and groupby is None and len(y_cols) == 1:
        return _create_multi_panel_plot(
            obs_tidy, keys, y_cols[0], colors, log, stripplot, jitter, 
            size, density_norm, enhanced_style, rasterized=rasterized, **kwds
        )
    
    # Create single or multiple axis plots
    if len(y_cols) > 1:
        # Multiple keys: create subplots directly
        if ax is not None and verbose:
            print(f"{Colors.WARNING}Warning: ax parameter ignored when plotting multiple keys{Colors.ENDC}")
        
        # Use user-provided figsize or calculate appropriate size
        if figsize is not None:
            fig, axes = plt.subplots(1, len(y_cols), figsize=figsize)
        else:
            # Default: width scales with number of keys
            fig, axes = plt.subplots(1, len(y_cols), figsize=(5*len(y_cols), 6))
        
        # Ensure axes is always a list
        if len(y_cols) == 1:
            axes = [axes]
    else:
        # Single key: use provided ax or create new one
        if ax is None:
            if figsize is not None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
        axes = [ax]
    
    # Create plots for each y variable
    for i, (y_col, y_label) in enumerate(zip(y_cols, ylabel)):
        current_ax = axes[i]
        
        # Apply enhanced styling to each subplot
        if enhanced_style:
            _apply_enhanced_styling(current_ax, background_color, spine_color, grid_lines)

        if alternating_background:
            _add_alternating_background(
                current_ax,
                group_categories,
                colors if use_group_colors_background else None,
                alternating_background_colors,
                background_lighten_amount,
            )

        # Prepare data for current y variable
        plot_data = _prepare_plot_data(obs_tidy, x_col, y_col, group_categories, order)

        # Create violin plots
        _create_violin_plots(
            current_ax, plot_data, group_categories, colors, density_norm,
            violin_alpha, enhanced_style, **kwds
        )

        # Add box plots if requested
        if show_boxplot or add_box:
            _add_box_plots(current_ax, plot_data, group_categories)
        
        # Add strip plot (jittered points)
        if stripplot:
            _add_strip_plot(
                current_ax, plot_data, group_categories, jitter, jitter_method, 
                size, jitter_alpha, colors, rasterized
            )
        
        # Add mean annotations
        if show_means:
            _add_mean_annotations(current_ax, plot_data, group_categories)
        
        # Add statistical tests
        if statistical_tests:
            _add_statistical_tests(current_ax, plot_data, group_categories, statistical_tests)

        # Add pairwise comparison brackets
        if comparisons:
            _add_comparisons(current_ax, plot_data, group_categories, comparisons, fontsize=fontsize - 2)

        # Customize axis
        _customize_axis(
            current_ax, group_categories, xlabel, y_label, groupby, 
            rotation, log, order
        )
        
        # Apply consistent styling to all subplots
        current_ax.spines['top'].set_visible(False)
        current_ax.spines['right'].set_visible(False)
        current_ax.spines['bottom'].set_visible(True)
        current_ax.spines['left'].set_visible(True)
        current_ax.spines['left'].set_position(('outward', 10))
        current_ax.spines['bottom'].set_position(('outward', 10))
        
        # Set font sizes
        if ticks_fontsize is None:
            ticks_fontsize = fontsize - 1
    
        current_ax.set_xticklabels(current_ax.get_xticklabels(), fontsize=ticks_fontsize, rotation=rotation)
        current_ax.set_yticklabels(current_ax.get_yticklabels(), fontsize=ticks_fontsize)
        current_ax.set_xlabel(groupby, fontsize=fontsize)
        current_ax.set_ylabel(y_label, fontsize=fontsize)
        current_ax.yaxis.grid(grid_lines)
    
    # Apply tight layout for better spacing
    if len(y_cols) > 1:
        plt.tight_layout()
    

    
    # Save figure if requested
    if save:
        save_path = save if isinstance(save, str) else "violin_plot.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show figure if requested
    if show is True or (show is None and ax is None):
        plt.show()
        return None
    
    # Return appropriate object based on number of subplots
    if len(y_cols) > 1:
        return axes
    else:
        return axes[0]

def _extract_data_from_adata(adata, keys, groupby, layer, use_raw):
    r"""
    Extract data from AnnData object for violin plotting.

    Args:
        adata: AnnData object
        keys: list of str
            Variables to extract
        groupby: str
            Grouping variable
        layer: str, optional
            Layer to use for gene expression data
        use_raw: bool, optional
            Whether to use raw data. Defaults to True if `.raw` is present.

    Returns:
        dict: Dictionary containing extracted data
    """
    # Default behavior: use raw if it exists and use_raw is not explicitly False
    if use_raw is None:
        use_raw = hasattr(adata, 'raw') and adata.raw is not None

    data_dict = {}

    # Handle groupby column
    if groupby is not None:
        if groupby in adata.obs.columns:
            data_dict[groupby] = adata.obs[groupby]
        else:
            raise ValueError(f"groupby '{groupby}' not found in adata.obs")

    # Handle keys (variables)
    _has_raw = hasattr(adata, 'raw') and adata.raw is not None
    for key in keys:
        if key in adata.obs.columns:
            # Key is in observations
            data_dict[key] = adata.obs[key]
        elif key in adata.var_names or (_has_raw and key in adata.raw.var_names):
            # A gene: ov.pl.get_values applies the layer / raw rules, densifies
            # a sparse column, and rescues names left only in .raw by HVG
            # subsetting
            data_dict[key] = get_values(
                adata, key, layer=layer if not use_raw else None,
                use_raw=True if use_raw and _has_raw else None,
            )
        else:
            available = "adata.obs, adata.var_names"
            if _has_raw:
                available += ", or adata.raw.var_names (with use_raw=True)"
            raise ValueError(f"Key '{key}' not found in {available}")

    return pd.DataFrame(data_dict)

def _setup_colors(custom_colors, group_categories, adata, groupby):
    """Set up colors for groups."""
    if custom_colors is not None:
        return custom_colors
    
    # Default color palette
    default_colors = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D", "#666666"]
    
    # Try to get colors from adata if groupby is specified
    if groupby is not None and hasattr(adata, 'uns') and f"{groupby}_colors" in adata.uns:
        return adata.uns[f"{groupby}_colors"]
    
    # Return default colors, cycling if necessary
    n_groups = len(group_categories)
    return (default_colors * ((n_groups // len(default_colors)) + 1))[:n_groups]

def _apply_enhanced_styling(ax, background_color, spine_color, grid_lines):
    """Apply enhanced styling to the plot."""
    ax.set_facecolor(background_color)
    ax.set_axisbelow(True)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color(spine_color)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_color(spine_color)
    ax.spines["bottom"].set_linewidth(1.5)

    if grid_lines:
        ax.yaxis.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    else:
        ax.yaxis.grid(False)

    ax.tick_params(length=0)

def _lighten_color(color, amount=0.8):
    """Return a lightened version of *color* by blending it with white.

    amount=0 → original color, amount=1 → white.
    """
    import matplotlib.colors as mc
    try:
        r, g, b, a = mc.to_rgba(color)
        return (r + (1 - r) * amount,
                g + (1 - g) * amount,
                b + (1 - b) * amount,
                a)
    except ValueError:
        return color


def _add_alternating_background(ax, group_categories, group_colors=None,
                                alternating_colors=("white", "#e8e8e8"),
                                lighten_amount=0.8):
    """Add background bands for each category.

    If *group_colors* is provided, each band is the lightened version of that
    group's color.  Otherwise the two *alternating_colors* are cycled.
    """
    for idx, _ in enumerate(group_categories):
        if group_colors is not None:
            color = _lighten_color(group_colors[idx % len(group_colors)], amount=lighten_amount)
        else:
            color = alternating_colors[idx % len(alternating_colors)]
        ax.axvspan(idx - 0.5, idx + 0.5, facecolor=color, edgecolor='none', zorder=0)


def _add_comparisons(ax, plot_data, group_categories, comparisons, fontsize=11):
    """Draw pairwise significance brackets above the violin plots.

    Each item in *comparisons* is a 2-element sequence of group names.
    A Mann-Whitney U test is run for each pair and the result annotated with
    stars (*** p<0.001, ** p<0.01, * p<0.05, ns otherwise).
    """
    from scipy.stats import mannwhitneyu

    cat_to_pos = {cat: i for i, cat in enumerate(group_categories)}

    # Compute data max to anchor bracket positions
    all_vals = np.concatenate([v for v in plot_data.values() if len(v) > 0])
    data_max = float(np.max(all_vals)) if len(all_vals) > 0 else 1.0
    data_range = data_max - (float(np.min(all_vals)) if len(all_vals) > 0 else 0.0)
    if data_range == 0:
        data_range = abs(data_max) or 1.0

    tick_h = data_range * 0.04    # height of the vertical ticks
    gap    = data_range * 0.04    # gap between data and first bracket
    step   = data_range * 0.12   # vertical spacing between brackets

    needed_top = data_max + gap

    for pair_idx, pair in enumerate(comparisons):
        if len(pair) != 2:
            continue
        g1, g2 = str(pair[0]), str(pair[1])
        if g1 not in cat_to_pos or g2 not in cat_to_pos:
            continue

        x1, x2 = cat_to_pos[g1], cat_to_pos[g2]
        d1 = plot_data.get(g1, np.array([]))
        d2 = plot_data.get(g2, np.array([]))
        if len(d1) < 2 or len(d2) < 2:
            continue

        _, p_val = mannwhitneyu(d1, d2, alternative='two-sided')
        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        elif p_val < 0.05:
            sig = '*'
        else:
            sig = 'ns'

        y_base = data_max + gap + step * pair_idx

        # Bracket: two vertical ticks connected by a horizontal bar
        ax.plot([x1, x1, x2, x2],
                [y_base, y_base + tick_h, y_base + tick_h, y_base],
                lw=1.2, color='black', clip_on=False)

        # Significance label
        ax.text((x1 + x2) / 2, y_base + tick_h,
                sig, ha='center', va='bottom',
                fontsize=fontsize, color='black')

        needed_top = y_base + tick_h + step * 0.5

    # Expand y-axis to show all brackets
    cur_lo, cur_hi = ax.get_ylim()
    ax.set_ylim(cur_lo, max(cur_hi, needed_top))

def _prepare_plot_data(obs_tidy, x_col, y_col, group_categories, order):
    """Prepare data for plotting."""
    plot_data = {}
    categories = order if order is not None else group_categories
    
    for category in categories:
        if x_col == "variable":
            mask = obs_tidy[x_col] == category
        else:
            mask = obs_tidy[x_col] == category
        plot_data[category] = obs_tidy[mask][y_col].dropna().values
    
    return plot_data

def _create_violin_plots(ax, plot_data, group_categories, colors, density_norm, violin_alpha, enhanced_style, **kwds):
    """Create violin plots using matplotlib."""
    
    positions = list(range(len(group_categories)))
    data_arrays = [plot_data[cat] for cat in group_categories]
    
    # Create violin plots
    parts = ax.violinplot(
        data_arrays, 
        positions=positions,
        widths=0.6,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        **kwds
    )
    
    # Customize violin appearance
    for i, (pc, color) in enumerate(zip(parts['bodies'], colors)):
        if enhanced_style:
            pc.set_facecolor(color)
            pc.set_alpha(violin_alpha)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.2)
        else:
            pc.set_facecolor(color)
            pc.set_alpha(violin_alpha)

def _add_box_plots(ax, plot_data, group_categories):
    """Add box plots overlay."""
    positions = list(range(len(group_categories)))
    data_arrays = [plot_data[cat] for cat in group_categories]
    
    # Box plot properties
    medianprops = dict(linewidth=2, color='black')
    boxprops = dict(linewidth=1.5, color='black')
    
    ax.boxplot(
        data_arrays,
        positions=positions,
        widths=0.2,
        showfliers=False,
        showcaps=False,
        medianprops=medianprops,
        whiskerprops=boxprops,
        boxprops=boxprops
    )

def _add_strip_plot(ax, plot_data, group_categories, jitter, jitter_method, size, jitter_alpha, colors, rasterized=False):
    """Add jittered strip plot."""
    jitter_width = 0.04 if isinstance(jitter, bool) and jitter else float(jitter) * 0.04
    
    for i, (category, color) in enumerate(zip(group_categories, colors)):
        y_data = plot_data[category]
        n_points = len(y_data)
        
        if n_points == 0:
            continue
            
        # Generate jittered x coordinates
        x_base = np.array([i] * n_points)
        
        if jitter_method == 't_dist':
            x_jittered = x_base + st.t(df=6, scale=jitter_width).rvs(n_points)
        else:  # uniform
            x_jittered = x_base + np.random.uniform(-jitter_width, jitter_width, n_points)
        
        # Plot jittered points
        ax.scatter(
            x_jittered,
            y_data,
            s=size,
            color=color,
            alpha=jitter_alpha,
            edgecolors='black',
            linewidth=0.5,
            rasterized=rasterized,
        )

def _add_mean_annotations(ax, plot_data, group_categories):
    """Add mean value annotations."""
    RED_DARK = "#850e00"
    
    for i, category in enumerate(group_categories):
        data = plot_data[category]
        if len(data) == 0:
            continue
            
        mean_val = np.mean(data)
        
        # Add dot for mean
        ax.scatter(i, mean_val, s=150, color=RED_DARK, zorder=10, edgecolors='white', linewidth=1)
        
        # Add mean label
        ax.annotate(
            f'μ={mean_val:.2f}',
            xy=(i, mean_val),
            xytext=(10, 10),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8),
            fontsize=9,
            ha='left'
        )

def _add_statistical_tests(ax, plot_data, group_categories, statistical_tests):
    """Add statistical test results."""
    if len(group_categories) < 2:
        return
    
    try:
        from scipy.stats import (
            f_oneway, ttest_ind, mannwhitneyu, kruskal, 
            ranksums, wilcoxon
        )
        
        data_arrays = [plot_data[cat] for cat in group_categories if len(plot_data[cat]) > 0]
        
        if len(data_arrays) < 2:
            return
            
        # Determine which tests to perform
        if statistical_tests is True:
            # Auto-select based on number of groups (backward compatibility)
            if len(data_arrays) == 2:
                tests_to_perform = ['ttest']
            else:
                tests_to_perform = ['anova']
        elif isinstance(statistical_tests, str):
            tests_to_perform = [statistical_tests]
        elif isinstance(statistical_tests, list):
            tests_to_perform = statistical_tests
        else:
            return
        
        test_results = []
        
        for test_method in tests_to_perform:
            test_method = test_method.lower()
            
            try:
                if test_method in ['wilcox', 'wilcoxon'] and len(data_arrays) == 2:
                    # Wilcoxon rank-sum test (Mann-Whitney U test)
                    stat, p_val = mannwhitneyu(data_arrays[0], data_arrays[1], 
                                              alternative='two-sided')
                    test_results.append(f"Wilcox: p = {p_val:.2e}")
                    
                elif test_method == 'mannwhitney' and len(data_arrays) == 2:
                    # Mann-Whitney U test (same as wilcox rank-sum)
                    stat, p_val = mannwhitneyu(data_arrays[0], data_arrays[1], 
                                              alternative='two-sided')
                    test_results.append(f"Mann-Whitney: p = {p_val:.2e}")
                    
                elif test_method == 'ttest' and len(data_arrays) == 2:
                    # T-test for two groups
                    stat, p_val = ttest_ind(data_arrays[0], data_arrays[1])
                    test_results.append(f"t-test: p = {p_val:.2e}")
                    
                elif test_method == 'anova' and len(data_arrays) >= 2:
                    # ANOVA for multiple groups
                    stat, p_val = f_oneway(*data_arrays)
                    test_results.append(f"ANOVA: F = {stat:.2f}, p = {p_val:.2e}")
                    
                elif test_method == 'kruskal' and len(data_arrays) >= 2:
                    # Kruskal-Wallis test (non-parametric alternative to ANOVA)
                    stat, p_val = kruskal(*data_arrays)
                    test_results.append(f"Kruskal-Wallis: H = {stat:.2f}, p = {p_val:.2e}")
                    
                elif test_method in ['wilcox', 'wilcoxon', 'mannwhitney'] and len(data_arrays) > 2:
                    # Perform pairwise tests for multiple groups
                    from itertools import combinations
                    pairwise_results = []
                    for i, (cat1, cat2) in enumerate(combinations(group_categories, 2)):
                        if cat1 in plot_data and cat2 in plot_data:
                            if len(plot_data[cat1]) > 0 and len(plot_data[cat2]) > 0:
                                stat, p_val = mannwhitneyu(plot_data[cat1], plot_data[cat2], 
                                                          alternative='two-sided')
                                pairwise_results.append(f"{cat1}-{cat2}: p = {p_val:.2e}")
                                # Only show first 3 pairwise comparisons to avoid clutter
                                if i >= 2:
                                    pairwise_results.append("...")
                                    break
                    if pairwise_results:
                        test_results.append(f"Pairwise Wilcox: {'; '.join(pairwise_results)}")
                        
                else:
                    print(f"Warning: Test method '{test_method}' not supported or not appropriate for {len(data_arrays)} groups")
                    
            except Exception as e:
                print(f"Warning: Failed to perform {test_method} test: {str(e)}")
        
        # Add test result text
        if test_results:
            test_text = "\n".join(test_results)
            ax.text(0.02, 0.98, test_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                   family='monospace')
               
    except ImportError as e:
        print(f"Statistical tests require scipy: {str(e)}")

def _customize_axis(ax, group_categories, xlabel, ylabel, groupby, rotation, log, order):
    """Customize axis labels and ticks."""
    # Set x-axis
    positions = list(range(len(group_categories)))
    ax.set_xticks(positions)
    ax.set_xticklabels(group_categories, rotation=rotation)
    
    if xlabel:
        ax.set_xlabel(xlabel,)
    elif groupby:
        ax.set_xlabel(groupby.replace('_', ' '))
    
    # Set y-axis
    if ylabel:
        ax.set_ylabel(ylabel)
    
    # Set log scale if requested
    if log:
        ax.set_yscale('log')

def _create_multi_panel_plot(obs_tidy, keys, y_col, colors, log, stripplot, jitter, size, density_norm, enhanced_style, rasterized=False, **kwds):
    """Create multi-panel plot for multiple keys."""
    n_panels = len(keys)
    fig, axes = plt.subplots(1, n_panels, figsize=(4*n_panels, 6), sharey=False)
    
    if n_panels == 1:
        axes = [axes]
    
    for i, (key, ax_panel) in enumerate(zip(keys, axes)):
        # Filter data for current key
        key_data = obs_tidy[obs_tidy['variable'] == key]
        
        # Create violin plot
        data_array = [key_data['value'].values]
        parts = ax_panel.violinplot(data_array, positions=[0], **kwds)
        
        # Customize
        for pc in parts['bodies']:
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.8)
        
        # Add stripplot if requested
        if stripplot:
            y_data = key_data['value'].values
            x_jittered = np.random.uniform(-0.1, 0.1, len(y_data))
            ax_panel.scatter(
                x_jittered,
                y_data,
                s=size * 20,
                color='black',
                alpha=0.6,
                rasterized=rasterized,
            )
        
        # Set title and labels
        ax_panel.set_title(key)
        ax_panel.set_xticks([])
        
        if log:
            ax_panel.set_yscale('log')
    
    plt.tight_layout()
    return fig
