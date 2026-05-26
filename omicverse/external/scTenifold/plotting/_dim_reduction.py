from typing import Callable, Optional, Tuple, Union

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, MDS, SpectralEmbedding, LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from enum import Enum

__all__ = ["prepare_PCA_dfs", "prepare_embedding_dfs"]


class Reducer(Enum):
    """Supported non-PCA dimensionality reducers for :func:`prepare_embedding_dfs`."""
    TSNE = "TSNE"
    Isomap = "Isomap"
    MDS = "MDS"
    SpectralEmbedding = "SpectralEmbedding"
    LocallyLinearEmbedding = "LocallyLinearEmbedding"
    UMAP = "UMAP"


REDUCER_DICT = {Reducer.TSNE: TSNE,
                Reducer.MDS: MDS,
                Reducer.Isomap: Isomap,
                Reducer.LocallyLinearEmbedding: LocallyLinearEmbedding,
                Reducer.SpectralEmbedding: SpectralEmbedding}


def prepare_PCA_dfs(feature_df: pd.DataFrame,
                    transform_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                    n_components: Optional[int] = None,
                    standardize: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run PCA on a genes-by-cells DataFrame.

    Parameters
    ----------
    feature_df
        Input expression DataFrame (rows are features, columns are samples).
    transform_func
        Optional pre-PCA transform applied to ``feature_df``.
    n_components
        Number of components; defaults to ``min(n_samples, n_features)``.
    standardize
        If True, z-score columns before PCA.

    Returns
    -------
    Tuple ``(scores, explained_variance, loadings)`` as DataFrames.
    """
    if transform_func is not None:
        x = transform_func(feature_df)
    else:
        x = feature_df
    x = StandardScaler().fit_transform(x.values.T) if standardize else x.values.T
    pca = PCA(n_components=n_components)
    if not n_components:
        n_components = min(x.shape[0], x.shape[1])
    principal_components = pca.fit_transform(x)
    final_df = pd.DataFrame(data=principal_components,
                            columns=[f'PC {num + 1}' for num in range(principal_components.shape[1])],
                            index=feature_df.columns)
    exp_var_df = pd.DataFrame(data=pca.explained_variance_ratio_,
                              index=[f'PC {num + 1}' for num in range(n_components)])
    component_df = pd.DataFrame(data=pca.components_.T,
                                columns=[f'PC {num + 1}' for num in range(n_components)],
                                index=feature_df.index)
    return final_df, exp_var_df, component_df


def prepare_embedding_dfs(feature_df: pd.DataFrame,
                          transform_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                          n_components: int = 2,
                          reducer: Union[str, "Reducer"] = "TSNE",
                          standardize: bool = True, **kwargs: object) -> pd.DataFrame:
    """Run a non-PCA dimensionality reducer on a feature DataFrame.

    Parameters
    ----------
    feature_df
        Input expression DataFrame (features x samples).
    transform_func
        Optional pre-embedding transform applied to ``feature_df.values``.
    n_components
        Number of embedding dimensions.
    reducer
        Reducer name or :class:`Reducer` member. ``"UMAP"`` requires
        the optional ``umap-learn`` package.
    standardize
        If True, z-score columns before reduction.
    **kwargs
        Forwarded to the underlying reducer class.

    Returns
    -------
    Sample-by-component DataFrame.
    """
    if transform_func:
        x = transform_func(feature_df.values)
    else:
        x = feature_df.values
    if isinstance(reducer, str):
        reducer = Reducer(reducer)
    sample_names = feature_df.columns.to_list()
    x = StandardScaler().fit_transform(x.T) if standardize else x.T
    if reducer == Reducer.UMAP:
        try:
            from importlib import import_module
            umap = import_module("umap")
        except ImportError as exc:
            raise ImportError("Install umap-learn to use reducer='UMAP'.") from exc
        reducer_cls = umap.UMAP
    else:
        reducer_cls = REDUCER_DICT[reducer]
    X_embedded = reducer_cls(n_components=n_components, **kwargs).fit_transform(x)
    df = pd.DataFrame(X_embedded,
                      columns=["{reducer} {i}".format(reducer=reducer.value, i=i) for i in range(1, n_components + 1)],
                      index=sample_names)
    return df
