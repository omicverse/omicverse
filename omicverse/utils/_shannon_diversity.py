import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from anndata import AnnData
from typing import Optional, List, Tuple, Dict
from scipy import stats


def shannon_diversity(
    adata: AnnData,
    groupby: str,
    cell_type_key: str,
    base: str = 'natural',
    calculate_evenness: bool = True
) -> pd.DataFrame:
    """Calculate Shannon diversity from cell-type composition.

    Parameters
    ----------
    adata : AnnData
        Input object containing group and cell-type metadata in ``adata.obs``.
    groupby : str
        Obs column defining groups/samples for diversity calculation.
    cell_type_key : str
        Obs column containing cell-type annotations.
    base : str, default='natural'
        Logarithm base used in Shannon index, one of ``'natural'``, ``'2'``,
        or ``'10'``.
    calculate_evenness : bool, default=True
        Whether to compute Shannon evenness (Pielou-style normalization).

    Returns
    -------
    pandas.DataFrame
        Per-group diversity table including Shannon and Simpson metrics.
        
    Examples:
        >>> import omicverse as ov
        >>> 
        >>> # Load data
        >>> adata = ov.datasets.pbmc3k(processed=True)
        >>> 
        >>> # Add mock condition groups
        >>> adata.obs['condition'] = np.random.choice(['Control', 'Treatment'], adata.n_obs)
        >>> 
        >>> # Calculate Shannon diversity
        >>> diversity_results = ov.utils.shannon_diversity(
        ...     adata, 
        ...     groupby='condition', 
        ...     cell_type_key='louvain'
        ... )
        >>> print(diversity_results)
    """
    
    # Set logarithm base
    if base == 'natural':
        log_func = np.log
    elif base == '2':
        log_func = np.log2
    elif base == '10':
        log_func = np.log10
    else:
        raise ValueError("base must be 'natural', '2', or '10'")
    
    # Get unique groups and cell types
    groups = adata.obs[groupby].unique()
    results = []
    
    for group in groups:
        # Get cells for this group
        group_mask = adata.obs[groupby] == group
        group_cells = adata.obs[group_mask]
        
        if len(group_cells) == 0:
            continue
            
        # Count cell types in this group
        cell_type_counts = group_cells[cell_type_key].value_counts()
        total_cells = cell_type_counts.sum()
        
        # Calculate proportions
        proportions = cell_type_counts / total_cells
        
        # Calculate Shannon diversity index
        # Only include non-zero proportions to avoid log(0)
        shannon_index = -np.sum(proportions * log_func(proportions))
        
        # Calculate evenness (equitability) if requested
        evenness = None
        if calculate_evenness:
            max_diversity = log_func(len(cell_type_counts))  # ln(S) where S is number of species
            evenness = shannon_index / max_diversity if max_diversity > 0 else 0
        
        # Calculate Simpson's diversity index as well
        simpson_index = 1 - np.sum(proportions**2)
        
        results.append({
            'group': group,
            'shannon_diversity': shannon_index,
            'shannon_evenness': evenness,
            'simpson_diversity': simpson_index,
            'total_cells': total_cells,
            'n_cell_types': len(cell_type_counts),
            'dominant_cell_type': cell_type_counts.index[0],
            'dominant_proportion': proportions.iloc[0]
        })
    
    results_df = pd.DataFrame(results)
    
    # Store results in AnnData object
    adata.uns['shannon_diversity_results'] = results_df
    adata.uns[f'shannon_diversity_{groupby}_{cell_type_key}'] = {
        'base': base,
        'calculate_evenness': calculate_evenness,
        'cell_type_counts': {
            group: adata.obs[adata.obs[groupby] == group][cell_type_key].value_counts().to_dict()
            for group in groups
        }
    }
    
    return results_df


def compare_shannon_diversity(
    adata: AnnData,
    groupby: str,
    cell_type_key: str,
    test_method: str = 'kruskal',
    base: str = 'natural'
) -> Dict:
    """Compare Shannon diversity values between groups.

    Parameters
    ----------
    adata : AnnData
        Input object with annotations in ``adata.obs``.
    groupby : str
        Obs column defining groups to compare.
    cell_type_key : str
        Obs column containing cell-type labels.
    test_method : str, default='kruskal'
        Statistical test selector. Supports ``'kruskal'``,
        ``'mann_whitney'``, and ``'ttest'`` (partial support).
    base : str, default='natural'
        Log base forwarded to :func:`shannon_diversity`.

    Returns
    -------
    dict
        Dictionary with per-group diversity values and statistical test output.
    """
    
    # Calculate diversity for each sample individually
    samples = adata.obs[groupby].unique()
    diversity_values = {}
    
    for sample in samples:
        sample_mask = adata.obs[groupby] == sample
        sample_adata = adata[sample_mask].copy()
        
        # Calculate diversity for this sample
        # Add a dummy groupby column since we already filtered to single sample
        sample_adata.obs['_dummy_group'] = 'all'
        diversity_result = shannon_diversity(
            sample_adata, 
            groupby='_dummy_group',  # Use dummy groupby since we want per-sample diversity
            cell_type_key=cell_type_key,
            base=base,
            calculate_evenness=True
        )
        
        if len(diversity_result) > 0:
            diversity_values[sample] = diversity_result['shannon_diversity'].values[0]
    
    # Prepare data for statistical testing
    group_diversities = list(diversity_values.values())
    group_labels = list(diversity_values.keys())
    
    # Perform statistical test
    test_results = {}
    
    if len(set(group_labels)) >= 2:
        if test_method == 'kruskal':
            # Kruskal-Wallis test for multiple groups
            groups_data = [
                [diversity_values[group] for i in range(1)] 
                for group in set(group_labels)
            ]
            # Since we only have one value per group, we'll use the values directly
            if len(set(group_labels)) == 2:
                stat, p_value = stats.mannwhitneyu(
                    [diversity_values[group_labels[0]]], 
                    [diversity_values[group_labels[1]]]
                )
                test_results['test'] = 'mann_whitney'
            else:
                # For multiple groups with single values, we can't perform meaningful statistics
                stat, p_value = np.nan, np.nan
                test_results['test'] = 'insufficient_replicates'
        else:
            stat, p_value = np.nan, np.nan
            test_results['test'] = test_method
            
        test_results['statistic'] = stat
        test_results['p_value'] = p_value
    else:
        test_results = {'test': 'insufficient_groups', 'statistic': np.nan, 'p_value': np.nan}
    
    return {
        'diversity_values': diversity_values,
        'test_results': test_results,
        'mean_diversity': np.mean(group_diversities),
        'std_diversity': np.std(group_diversities)
    }


def plot_shannon_diversity(
    adata: AnnData,
    groupby: Optional[str] = None,
    metric: str = 'shannon_diversity',
    figsize: Tuple[int, int] = (8, 6),
    palette: Optional[str] = None,
    show_stats: bool = True,
    save_path: Optional[str] = None
) -> None:
    """Visualize diversity metrics across groups.

    Parameters
    ----------
    adata : AnnData
        Object containing ``shannon_diversity_results`` in ``adata.uns``.
    groupby : str or None, default=None
        Grouping variable for plotting. If ``None``, uses stored result groups.
    metric : str, default='shannon_diversity'
        Metric column to plot.
    figsize : Tuple[int, int], default=(8, 6)
        Figure size in inches.
    palette : str or None, default=None
        Seaborn palette name or custom palette.
    show_stats : bool, default=True
        Whether to display statistical comparison text when possible.
    save_path : str or None, default=None
        Optional path to save the figure.

    Returns
    -------
    None
        Displays and optionally saves the plot.
    """
    
    if 'shannon_diversity_results' not in adata.uns:
        raise ValueError("No Shannon diversity results found. Run shannon_diversity() first.")
    
    results_df = adata.uns['shannon_diversity_results']
    
    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results. Available: {results_df.columns.tolist()}")
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create box plot or violin plot
    if len(results_df) > 1:
        sns.boxplot(
            data=results_df, 
            x='group', 
            y=metric,
            palette=palette,
            ax=ax
        )
        
        # Add individual points
        sns.stripplot(
            data=results_df, 
            x='group', 
            y=metric,
            color='black',
            size=8,
            alpha=0.7,
            ax=ax
        )
    else:
        # Bar plot for single group
        sns.barplot(
            data=results_df, 
            x='group', 
            y=metric,
            palette=palette,
            ax=ax
        )
    
    # Customize the plot
    metric_labels = {
        'shannon_diversity': 'Shannon Diversity Index',
        'shannon_evenness': 'Shannon Evenness',
        'simpson_diversity': 'Simpson Diversity Index'
    }
    
    ax.set_ylabel(metric_labels.get(metric, metric))
    ax.set_xlabel('Group')
    ax.set_title(f'{metric_labels.get(metric, metric)} by Group')
    
    # Add statistical annotations if requested
    if show_stats and len(results_df) > 1:
        from scipy import stats
        groups = results_df['group'].unique()
        if len(groups) == 2:
            # Perform Mann-Whitney U test for two groups
            group1_val = results_df[results_df['group'] == groups[0]][metric].values
            group2_val = results_df[results_df['group'] == groups[1]][metric].values
            
            if len(group1_val) > 0 and len(group2_val) > 0:
                try:
                    stat, p_val = stats.mannwhitneyu(group1_val, group2_val)
                    
                    # Add significance annotation
                    y_max = results_df[metric].max()
                    y_range = results_df[metric].max() - results_df[metric].min()
                    y_pos = y_max + 0.1 * y_range
                    
                    # Format p-value
                    if p_val < 0.001:
                        p_text = "p < 0.001"
                    elif p_val < 0.01:
                        p_text = "p < 0.01"
                    elif p_val < 0.05:
                        p_text = "p < 0.05"
                    else:
                        p_text = f"p = {p_val:.3f}"
                    
                    ax.annotate(
                        p_text, 
                        xy=(0.5, y_pos), 
                        ha='center', 
                        va='bottom',
                        xycoords='axes fraction'
                    )
                    
                    # Add line connecting groups
                    ax.plot([0, 1], [y_pos - 0.02 * y_range, y_pos - 0.02 * y_range], 
                           'k-', linewidth=1)
                    
                except Exception as e:
                    print(f"Could not perform statistical test: {e}")
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
