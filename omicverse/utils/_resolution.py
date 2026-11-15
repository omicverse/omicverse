import pandas as pd
import numpy as np
from anndata import AnnData
from typing import Optional, List, Tuple, Dict, Union
from ._compat import leiden_like_scanpy as _leiden
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
import warnings


def _louvain(adata, **kwargs):
    from ..pp import louvain
    return louvain(adata, **kwargs)


def optimal_resolution(
    adata: AnnData,
    resolution_range: Tuple[float, float] = (0.1, 2.0),
    n_resolutions: int = 20,
    clustering_method: str = 'leiden',
    metric: str = 'silhouette',
    use_rep: str = 'X_pca',
    key_added: Optional[str] = None,
    random_state: int = 42,
    copy: bool = False
) -> Union[float, AnnData]:
    """Find an empirically optimal clustering resolution.

    Parameters
    ----------
    adata : AnnData
        Input object with neighborhood graph already computed in ``adata.uns``.
    resolution_range : Tuple[float, float], default=(0.1, 2.0)
        Minimum and maximum resolution values to scan.
    n_resolutions : int, default=20
        Number of evenly spaced resolution values in the scan range.
    clustering_method : str, default='leiden'
        Clustering backend, either ``'leiden'`` or ``'louvain'``.
    metric : str, default='silhouette'
        Optimization metric: ``'silhouette'``, ``'modularity'``, or ``'both'``.
    use_rep : str, default='X_pca'
        Embedding key used for silhouette computation.
    key_added : str or None, default=None
        Obs key used to store clustering at the selected resolution.
    random_state : int, default=42
        Random seed for clustering.
    copy : bool, default=False
        Whether to run on and return a copied ``AnnData`` object.

    Returns
    -------
    float or AnnData
        Best resolution if ``copy=False``; otherwise modified copied object.
        
    Examples:
        >>> import omicverse as ov
        >>> 
        >>> # Load and preprocess data
        >>> adata = ov.datasets.pbmc3k(processed=True)
        >>> ov.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        >>> 
        >>> # Find optimal resolution
        >>> optimal_res = ov.utils.optimal_resolution(
        ...     adata, 
        ...     resolution_range=(0.1, 1.5),
        ...     metric='silhouette'
        ... )
        >>> print(f"Optimal resolution: {optimal_res}")
    """
    
    adata_work = adata.copy() if copy else adata
    
    # Check if neighbors graph exists
    if 'neighbors' not in adata_work.uns:
        raise ValueError("Neighbors graph not found. Run ov.pp.neighbors() first.")
    
    # Generate resolution values to test
    if resolution_range[0] >= resolution_range[1]:
        raise ValueError("resolution_range[0] must be less than resolution_range[1]")
    
    resolutions = np.linspace(resolution_range[0], resolution_range[1], n_resolutions)
    
    results = []
    
    # Test each resolution
    for res in resolutions:
        # Perform clustering
        if clustering_method == 'leiden':
            _leiden(adata_work, resolution=res, random_state=random_state, key_added='temp_clusters')
        elif clustering_method == 'louvain':
            _louvain(adata_work, resolution=res, random_state=random_state, key_added='temp_clusters')
        else:
            raise ValueError("clustering_method must be 'leiden' or 'louvain'")
        
        cluster_labels = adata_work.obs['temp_clusters'].astype(int)
        n_clusters = len(cluster_labels.unique())
        
        # Skip if only one cluster (can't calculate silhouette)
        if n_clusters < 2:
            results.append({
                'resolution': res,
                'n_clusters': n_clusters,
                'silhouette_score': np.nan,
                'modularity': np.nan,
                'valid': False
            })
            continue
        
        # Calculate evaluation metrics
        metrics_dict = {
            'resolution': res,
            'n_clusters': n_clusters,
            'valid': True
        }
        
        # Silhouette score
        if metric in ['silhouette', 'both']:
            try:
                if use_rep in adata_work.obsm:
                    X = adata_work.obsm[use_rep]
                elif use_rep == 'X':
                    X = adata_work.X
                    if hasattr(X, 'toarray'):  # Handle sparse matrices
                        X = X.toarray()
                else:
                    raise ValueError(f"Representation '{use_rep}' not found in adata.obsm")
                
                sil_score = silhouette_score(X, cluster_labels)
                metrics_dict['silhouette_score'] = sil_score
            except Exception as e:
                warnings.warn(f"Could not calculate silhouette score for resolution {res}: {e}")
                metrics_dict['silhouette_score'] = np.nan
        else:
            metrics_dict['silhouette_score'] = np.nan
        
        # Modularity score
        if metric in ['modularity', 'both']:
            try:
                if clustering_method == 'leiden':
                    # Calculate modularity for leiden clustering
                    modularity = _leiden(
                        adata_work, 
                        resolution=res, 
                        random_state=random_state, 
                        key_added='temp_clusters_mod',
                        copy=True
                    ).uns['leiden']['modularity'][-1]  # Get final modularity
                else:
                    # For louvain, modularity is stored differently
                    modularity = np.nan  # Placeholder - would need proper implementation
                
                metrics_dict['modularity'] = modularity
            except Exception as e:
                warnings.warn(f"Could not calculate modularity for resolution {res}: {e}")
                metrics_dict['modularity'] = np.nan
        else:
            metrics_dict['modularity'] = np.nan
        
        results.append(metrics_dict)
        
        # Clean up temporary columns
        if 'temp_clusters' in adata_work.obs.columns:
            del adata_work.obs['temp_clusters']
        if 'temp_clusters_mod' in adata_work.obs.columns:
            del adata_work.obs['temp_clusters_mod']
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    valid_results = results_df[results_df['valid']].copy()
    
    if len(valid_results) == 0:
        raise ValueError("No valid clustering results found. Try adjusting resolution_range.")
    
    # Find optimal resolution based on metric
    optimal_res = None
    
    if metric == 'silhouette':
        if valid_results['silhouette_score'].isna().all():
            raise ValueError("Could not calculate silhouette scores for any resolution.")
        optimal_idx = valid_results['silhouette_score'].idxmax()
        optimal_res = valid_results.loc[optimal_idx, 'resolution']
        
    elif metric == 'modularity':
        if valid_results['modularity'].isna().all():
            raise ValueError("Could not calculate modularity scores for any resolution.")
        optimal_idx = valid_results['modularity'].idxmax()
        optimal_res = valid_results.loc[optimal_idx, 'resolution']
        
    elif metric == 'both':
        # Use combined score (normalized silhouette + modularity)
        sil_scores = valid_results['silhouette_score'].fillna(0)
        mod_scores = valid_results['modularity'].fillna(0)
        
        # Normalize scores to 0-1 range
        if sil_scores.max() != sil_scores.min():
            sil_norm = (sil_scores - sil_scores.min()) / (sil_scores.max() - sil_scores.min())
        else:
            sil_norm = sil_scores
            
        if mod_scores.max() != mod_scores.min():
            mod_norm = (mod_scores - mod_scores.min()) / (mod_scores.max() - mod_scores.min())
        else:
            mod_norm = mod_scores
        
        combined_score = (sil_norm + mod_norm) / 2
        optimal_idx = combined_score.idxmax()
        optimal_res = valid_results.loc[optimal_idx, 'resolution']
    
    # Apply optimal clustering
    if clustering_method == 'leiden':
        if key_added is None:
            key_added = f'leiden_optimal_res_{optimal_res:.2f}'
        _leiden(adata_work, resolution=optimal_res, random_state=random_state, key_added=key_added)
    else:
        if key_added is None:
            key_added = f'louvain_optimal_res_{optimal_res:.2f}'
        _louvain(adata_work, resolution=optimal_res, random_state=random_state, key_added=key_added)
    
    # Store results in adata
    adata_work.uns['optimal_resolution_results'] = results_df
    adata_work.uns['optimal_resolution_params'] = {
        'optimal_resolution': optimal_res,
        'metric': metric,
        'clustering_method': clustering_method,
        'resolution_range': resolution_range,
        'n_resolutions': n_resolutions,
        'optimal_key': key_added
    }
    
    if copy:
        return adata_work
    else:
        return optimal_res


def plot_resolution_optimization(
    adata: AnnData,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> None:
    """Visualize metric curves from resolution optimization.

    Parameters
    ----------
    adata : AnnData
        Object containing ``optimal_resolution_results`` in ``adata.uns``.
    figsize : Tuple[int, int], default=(12, 5)
        Figure size in inches.
    save_path : str or None, default=None
        Optional output path for saving the figure.

    Returns
    -------
    None
        Displays the plot and optionally saves it to disk.
    """
    
    if 'optimal_resolution_results' not in adata.uns:
        raise ValueError("No resolution optimization results found. Run optimal_resolution() first.")
    
    results_df = adata.uns['optimal_resolution_results']
    params = adata.uns['optimal_resolution_params']
    
    # Create subplots
    n_metrics = int('silhouette_score' in results_df.columns) + int('modularity' in results_df.columns)
    if n_metrics == 0:
        raise ValueError("No valid metrics found in results.")
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    plot_idx = 0
    
    # Plot silhouette scores
    if 'silhouette_score' in results_df.columns:
        ax = axes[plot_idx]
        valid_data = results_df[results_df['valid'] & ~results_df['silhouette_score'].isna()]
        
        ax.plot(valid_data['resolution'], valid_data['silhouette_score'], 
                'o-', color=colors[0], linewidth=2, markersize=6)
        
        # Highlight optimal resolution
        if params['metric'] in ['silhouette', 'both']:
            optimal_res = params['optimal_resolution']
            optimal_score = results_df[results_df['resolution'] == optimal_res]['silhouette_score'].iloc[0]
            ax.scatter([optimal_res], [optimal_score], color='red', s=100, zorder=5, 
                      label=f'Optimal (res={optimal_res:.2f})')
        
        ax.set_xlabel('Resolution')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Silhouette Score vs Resolution')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plot_idx += 1
    
    # Plot modularity scores
    if 'modularity' in results_df.columns:
        ax = axes[plot_idx]
        valid_data = results_df[results_df['valid'] & ~results_df['modularity'].isna()]
        
        if len(valid_data) > 0:
            ax.plot(valid_data['resolution'], valid_data['modularity'], 
                    'o-', color=colors[1], linewidth=2, markersize=6)
            
            # Highlight optimal resolution
            if params['metric'] in ['modularity', 'both']:
                optimal_res = params['optimal_resolution']
                optimal_score = results_df[results_df['resolution'] == optimal_res]['modularity'].iloc[0]
                if not pd.isna(optimal_score):
                    ax.scatter([optimal_res], [optimal_score], color='red', s=100, zorder=5,
                              label=f'Optimal (res={optimal_res:.2f})')
            
            ax.set_xlabel('Resolution')
            ax.set_ylabel('Modularity')
            ax.set_title('Modularity vs Resolution')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    # Add number of clusters as secondary y-axis
    if len(axes) > 0:
        ax2 = axes[0].twinx()
        ax2.plot(results_df['resolution'], results_df['n_clusters'], 
                's--', color=colors[2], alpha=0.7, markersize=4, label='# Clusters')
        ax2.set_ylabel('Number of Clusters', color=colors[2])
        ax2.tick_params(axis='y', labelcolor=colors[2])
        
        # Add legend for number of clusters
        lines1, labels1 = axes[0].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.suptitle(f'Resolution Optimization Results\n'
                f'Method: {params["clustering_method"]}, '
                f'Metric: {params["metric"]}, '
                f'Optimal: {params["optimal_resolution"]:.2f}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def resolution_stability_analysis(
    adata: AnnData,
    resolution_range: Tuple[float, float] = (0.1, 2.0),
    n_resolutions: int = 20,
    clustering_method: str = 'leiden',
    n_iterations: int = 10,
    use_rep: str = 'X_pca',
    random_state: int = 42
) -> pd.DataFrame:
    """Evaluate clustering stability across resolutions and random seeds.

    Parameters
    ----------
    adata : AnnData
        Input object with precomputed neighbor graph.
    resolution_range : Tuple[float, float], default=(0.1, 2.0)
        Scan range for resolution.
    n_resolutions : int, default=20
        Number of tested resolution values.
    clustering_method : str, default='leiden'
        Clustering backend, ``'leiden'`` or ``'louvain'``.
    n_iterations : int, default=10
        Number of repeated clustering runs per resolution.
    use_rep : str, default='X_pca'
        Reserved for future extension; currently not used.
    random_state : int, default=42
        Base seed for repeated runs.

    Returns
    -------
    pandas.DataFrame
        Per-resolution stability metrics including mean ARI.
    """
    
    # Check if neighbors graph exists
    if 'neighbors' not in adata.uns:
        raise ValueError("Neighbors graph not found. Run ov.pp.neighbors() first.")
    
    resolutions = np.linspace(resolution_range[0], resolution_range[1], n_resolutions)
    stability_results = []
    
    for res in resolutions:
        # Store clustering results for this resolution
        clusterings = []
        
        for i in range(n_iterations):
            current_seed = random_state + i
            
            # Perform clustering
            if clustering_method == 'leiden':
                _leiden(adata, resolution=res, random_state=current_seed, key_added='temp_stability')
            else:
                _louvain(adata, resolution=res, random_state=current_seed, key_added='temp_stability')
            
            clusterings.append(adata.obs['temp_stability'].astype(int).values.copy())
            
            # Clean up
            del adata.obs['temp_stability']
        
        # Calculate pairwise ARI between all clustering iterations
        ari_scores = []
        n_clusters_list = []
        
        for i in range(n_iterations):
            n_clusters_list.append(len(np.unique(clusterings[i])))
            for j in range(i+1, n_iterations):
                ari = adjusted_rand_score(clusterings[i], clusterings[j])
                ari_scores.append(ari)
        
        # Calculate stability metrics
        mean_ari = np.mean(ari_scores) if ari_scores else np.nan
        std_ari = np.std(ari_scores) if ari_scores else np.nan
        mean_n_clusters = np.mean(n_clusters_list)
        std_n_clusters = np.std(n_clusters_list)
        
        stability_results.append({
            'resolution': res,
            'mean_ari': mean_ari,
            'std_ari': std_ari,
            'mean_n_clusters': mean_n_clusters,
            'std_n_clusters': std_n_clusters,
            'stability_score': mean_ari  # Higher is more stable
        })
    
    stability_df = pd.DataFrame(stability_results)
    
    # Store results
    adata.uns['resolution_stability_results'] = stability_df
    adata.uns['resolution_stability_params'] = {
        'resolution_range': resolution_range,
        'n_resolutions': n_resolutions,
        'clustering_method': clustering_method,
        'n_iterations': n_iterations,
        'random_state': random_state
    }
    
    return stability_df
