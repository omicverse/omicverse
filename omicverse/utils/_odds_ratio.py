import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from anndata import AnnData
from typing import Optional, List, Tuple


def odds_ratio(
    adata: AnnData,
    sample_key: str,
    cell_type_key: str,
    reference_group: Optional[str] = None,
    confidence_level: float = 0.95,
    correction_method: str = 'bonferroni'
) -> pd.DataFrame:
    """Compute cell-type enrichment odds ratios between sample groups.

    Parameters
    ----------
    adata : AnnData
        Input object with sample and cell-type annotations in ``adata.obs``.
    sample_key : str
        Column name in ``adata.obs`` defining biological groups/conditions.
    cell_type_key : str
        Column name in ``adata.obs`` containing cell-type labels.
    reference_group : str or None, default=None
        Baseline group used in pairwise comparisons. If ``None``, uses the
        alphabetically first group.
    confidence_level : float, default=0.95
        Confidence level for odds-ratio interval estimation.
    correction_method : str, default='bonferroni'
        Multiple-testing correction method: ``'bonferroni'``, ``'fdr_bh'``,
        or ``'none'``.

    Returns
    -------
    pandas.DataFrame
        Long-format table containing odds ratios, confidence intervals,
        p-values, and adjusted p-values.
    
    Examples:
        >>> import omicverse as ov
        >>> 
        >>> # Load data
        >>> adata = ov.datasets.pbmc3k(processed=True)
        >>> 
        >>> # Add mock sample groups for demonstration
        >>> adata.obs['condition'] = np.random.choice(['Control', 'Treatment'], adata.n_obs)
        >>> 
        >>> # Calculate odds ratios
        >>> or_results = ov.utils.odds_ratio(
        ...     adata, 
        ...     sample_key='condition', 
        ...     cell_type_key='louvain'
        ... )
        >>> print(or_results.head())
    """
    
    # Create contingency table
    crosstab = pd.crosstab(adata.obs[cell_type_key], adata.obs[sample_key])
    
    # Get unique groups
    groups = crosstab.columns.tolist()
    if reference_group is None:
        reference_group = sorted(groups)[0]
    elif reference_group not in groups:
        raise ValueError(f"Reference group '{reference_group}' not found in {sample_key}")
    
    # Initialize results list
    results = []
    
    # Calculate odds ratios for each cell type and group comparison
    cell_types = crosstab.index.tolist()
    
    for cell_type in cell_types:
        for group in groups:
            if group == reference_group:
                continue
                
            # Create 2x2 contingency table for this comparison
            # [cell_type_in_group, cell_type_in_ref]
            # [other_cells_in_group, other_cells_in_ref]
            
            cell_type_group = crosstab.loc[cell_type, group]
            cell_type_ref = crosstab.loc[cell_type, reference_group]
            other_cells_group = crosstab[group].sum() - cell_type_group
            other_cells_ref = crosstab[reference_group].sum() - cell_type_ref
            
            # Create 2x2 table
            table_2x2 = np.array([
                [cell_type_group, cell_type_ref],
                [other_cells_group, other_cells_ref]
            ])
            
            # Calculate odds ratio and p-value using Fisher's exact test
            try:
                odds_ratio_val, p_value = fisher_exact(table_2x2, alternative='two-sided')
                
                # Calculate confidence interval
                log_or = np.log(odds_ratio_val) if odds_ratio_val > 0 else np.nan
                
                if np.isfinite(log_or) and all(table_2x2.flatten() > 0):
                    se_log_or = np.sqrt(np.sum(1.0 / table_2x2))
                    alpha = 1 - confidence_level
                    z_score = stats.norm.ppf(1 - alpha/2)
                    
                    ci_lower = np.exp(log_or - z_score * se_log_or)
                    ci_upper = np.exp(log_or + z_score * se_log_or)
                else:
                    ci_lower = np.nan
                    ci_upper = np.nan
                    
            except (ValueError, ZeroDivisionError):
                odds_ratio_val = np.nan
                p_value = np.nan
                ci_lower = np.nan
                ci_upper = np.nan
            
            results.append({
                'cell_type': cell_type,
                'group': group,
                'reference_group': reference_group,
                'odds_ratio': odds_ratio_val,
                'p_value': p_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'cell_type_group_count': cell_type_group,
                'cell_type_ref_count': cell_type_ref,
                'total_group_count': crosstab[group].sum(),
                'total_ref_count': crosstab[reference_group].sum()
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Apply multiple testing correction
    if correction_method != 'none' and len(results_df) > 0:
        valid_pvals = ~results_df['p_value'].isna()
        if valid_pvals.sum() > 0:
            if correction_method == 'bonferroni':
                results_df.loc[valid_pvals, 'p_value_adjusted'] = (
                    results_df.loc[valid_pvals, 'p_value'] * valid_pvals.sum()
                )
                results_df.loc[valid_pvals, 'p_value_adjusted'] = (
                    results_df.loc[valid_pvals, 'p_value_adjusted'].clip(upper=1.0)
                )
            elif correction_method == 'fdr_bh':
                from statsmodels.stats.multitest import multipletests
                _, pvals_corrected, _, _ = multipletests(
                    results_df.loc[valid_pvals, 'p_value'], 
                    method='fdr_bh'
                )
                results_df.loc[valid_pvals, 'p_value_adjusted'] = pvals_corrected
            
            results_df.loc[~valid_pvals, 'p_value_adjusted'] = np.nan
    else:
        results_df['p_value_adjusted'] = results_df['p_value']
    
    # Store results in AnnData object
    adata.uns['odds_ratio_results'] = results_df
    adata.uns[f'odds_ratio_{sample_key}_{cell_type_key}'] = {
        'crosstab': crosstab,
        'reference_group': reference_group,
        'confidence_level': confidence_level,
        'correction_method': correction_method
    }
    
    return results_df


def plot_odds_ratio_heatmap(
    adata: AnnData,
    figsize: Tuple[int, int] = (10, 8),
    log_scale: bool = True,
    show_ci: bool = False,
    significance_threshold: float = 0.05,
    cmap: str = 'RdBu_r',
    save_path: Optional[str] = None
) -> None:
    """Plot a heatmap of odds ratios across groups and cell types.

    Parameters
    ----------
    adata : AnnData
        Object containing ``odds_ratio_results`` in ``adata.uns``.
    figsize : Tuple[int, int], default=(10, 8)
        Figure size in inches.
    log_scale : bool, default=True
        Whether to display ``log2(odds_ratio)`` values.
    show_ci : bool, default=False
        Whether to annotate confidence intervals in each heatmap cell.
    significance_threshold : float, default=0.05
        Adjusted p-value threshold used for significance marks.
    cmap : str, default='RdBu_r'
        Colormap name.
    save_path : str or None, default=None
        Optional path for saving the generated figure.

    Returns
    -------
    None
        Displays and optionally saves the figure.
    """
    
    if 'odds_ratio_results' not in adata.uns:
        raise ValueError("No odds ratio results found. Run odds_ratio() first.")
    
    results_df = adata.uns['odds_ratio_results']
    
    # Create pivot table for heatmap
    if log_scale:
        pivot_data = results_df.pivot(
            index='cell_type', 
            columns='group', 
            values='odds_ratio'
        ).apply(lambda x: np.log2(x))
        title = "Log2 Odds Ratios"
        center_val = 0
    else:
        pivot_data = results_df.pivot(
            index='cell_type', 
            columns='group', 
            values='odds_ratio'
        )
        title = "Odds Ratios"
        center_val = 1
    
    # Create significance mask
    pval_pivot = results_df.pivot(
        index='cell_type', 
        columns='group', 
        values='p_value_adjusted'
    )
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create annotations
    if show_ci:
        # Format annotations with confidence intervals
        annotations = results_df.set_index(['cell_type', 'group'])
        annot_matrix = pivot_data.copy()
        for idx in annot_matrix.index:
            for col in annot_matrix.columns:
                if (idx, col) in annotations.index:
                    row = annotations.loc[(idx, col)]
                    if not pd.isna(row['odds_ratio']):
                        ci_text = f"{row['odds_ratio']:.2f}\n[{row['ci_lower']:.2f}-{row['ci_upper']:.2f}]"
                        annot_matrix.loc[idx, col] = ci_text
                    else:
                        annot_matrix.loc[idx, col] = "N/A"
        
        sns.heatmap(
            pivot_data, 
            annot=annot_matrix, 
            fmt='', 
            cmap=cmap, 
            center=center_val,
            cbar_kws={'label': title},
            ax=ax
        )
    else:
        # Add significance stars
        annot_matrix = pivot_data.copy().astype(str)
        for idx in annot_matrix.index:
            for col in annot_matrix.columns:
                val = pivot_data.loc[idx, col]
                pval = pval_pivot.loc[idx, col]
                if not pd.isna(val) and not pd.isna(pval):
                    stars = ""
                    if pval < 0.001:
                        stars = "***"
                    elif pval < 0.01:
                        stars = "**"
                    elif pval < significance_threshold:
                        stars = "*"
                    annot_matrix.loc[idx, col] = f"{val:.2f}{stars}"
                else:
                    annot_matrix.loc[idx, col] = ""
        
        sns.heatmap(
            pivot_data, 
            annot=annot_matrix, 
            fmt='', 
            cmap=cmap, 
            center=center_val,
            cbar_kws={'label': title},
            ax=ax
        )
    
    ax.set_title(f"Odds Ratios by Cell Type and Group\n* p<{significance_threshold}, ** p<0.01, *** p<0.001")
    ax.set_xlabel("Sample Group")
    ax.set_ylabel("Cell Type")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
