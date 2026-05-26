from typing import Dict, List, Optional, Tuple

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import chi2

from omicverse.external.scTenifold.plotting._dim_reduction import *


def plot_network_graph(network: np.ndarray,
                       weight_thres: float = 0.1,
                       con_thres: float = 0) -> None:
    """
    Plot graph of a PCnet

    Parameters
    ----------
    network: np.ndarray
        A pc net
    weight_thres: float
        Minimum threshold of the pcnet's weights
    con_thres: float or int
        Minimum threshold of sum of weights
    Returns
    -------
    None
    """
    network = abs(network.copy())
    network[network < weight_thres] = 0
    valid_rows, valid_cols = (network.sum(axis=1) > con_thres), (network.sum(axis=0) > con_thres)
    network = network[valid_rows,:][:, valid_cols]
    G = nx.from_numpy_array(network)
    pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos,
                           ax=ax, nodelist=[0], alpha=0.4)
    nx.draw_networkx_nodes(G, pos,
                           ax=ax,
                           node_size=10,
                           cmap=plt.cm.Reds_r)
    plt.show()


def plot_network_heatmap(network: np.ndarray,
                         figsize: Tuple[int, int] = (12, 12)) -> None:
    """
    Plot a heatmap of a PC network

    Parameters
    ----------
    network: np.ndarray
        A pcnet
    figsize: tuple of ints
        output figure size
    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(network, center=0.0, ax=ax)


def plot_qqplot(df: pd.DataFrame,
                exp_col: str = "FC",
                stat_col: str = "adjusted p-value",
                plot_qqline: bool = True,
                sig_threshold: float = 0.1) -> None:
    """
    Plot QQ-plot using a d_regulation dataframe

    Parameters
    ----------
    df: pd.DataFrame
        A d_regulation dataframe
    exp_col: str
        Column name of data used to put the y-axis
    stat_col: str
        Column name of data used to check significance
    plot_qqline: bool
        Plot Q-Q line on the plot
    sig_threshold: float
        The significance
    Returns
    -------
    None
    """
    the_col = "Theoretical quantiles"
    len_x = df.shape[0]
    data = df.loc[:, [exp_col, stat_col]]
    data["significant"] = data[stat_col].apply(lambda x: x < sig_threshold)
    data.sort_values(exp_col, inplace=True)
    data[the_col] = chi2.ppf(q=np.linspace(0, 1, len_x + 2)[1:-1], df=1)
    sns.scatterplot(data=data, x="Theoretical quantiles", y=exp_col, hue="significant")
    if plot_qqline:
        xl_1, xl_2 = plt.gca().get_xlim()
        x1, x2 = data[the_col].quantile(0.25), data[the_col].quantile(0.75)
        y1, y2 = data[exp_col].quantile(0.25), data[exp_col].quantile(0.75)
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        plt.plot([xl_1, xl_2],
                 [slope * xl_1 + intercept, slope * xl_2 + intercept])
        plt.xlim([xl_1, xl_2])
    plt.show()


def plot_embedding(df: pd.DataFrame,
                   groups: Optional[Dict[str, List[str]]],
                   method: str = "UMAP",
                   plot_2D: bool = True,
                   figsize: Tuple[int, int] = (8, 8),
                   size: int = 10,
                   title: Optional[str] = None,
                   palette: str = "muted",
                   **kwargs: object) -> None:
    """
    Do dimension reduction and plot the embeddings onto a 2D plot

    Parameters
    ----------
    df: pd.DataFrame
        A dataframe to perform dimension reduction
    groups: dict(str, list)
        A dict indicating the groups
    method: str
        The name of used method, could be: PCA, TSNE, UMAP, Isomap, MDS, SpectralEmbedding, LocallyLinearEmbedding
    plot_2D: bool
        Draw a 2D or 3D (if false) plot
    figsize: tuple of int
        The figure size of the plot: (width, height)
    title: str
        The subplot's title
    palette: str
        The name of used seaborn color palette,
        reference: https://seaborn.pydata.org/generated/seaborn.color_palette.html
    kwargs: keyword arguments of doing dimension reduction

    Returns
    -------
    None
    """

    if method == "PCA":
        feature_df, exp_var_df, component_df = prepare_PCA_dfs(df, **kwargs)
        emb_name = "PC"
    else:
        feature_df = prepare_embedding_dfs(df, reducer=method, **kwargs)
        emb_name = method

    if groups is None:
        groups = {"all": df.columns.to_list()}
    colors = sns.color_palette(palette)
    if plot_2D:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    for i, (group_name, sample_names) in enumerate(groups.items()):
        em1, em2 = np.array([feature_df.loc[name, '{} 1'.format(emb_name)] for name in sample_names]), \
                   np.array([feature_df.loc[name, '{} 2'.format(emb_name)] for name in sample_names])

        if plot_2D:
            ax.scatter(em1, em2, s=size, label=group_name, c=[colors[i]])
        else:
            em3 = np.array([feature_df.loc[name, '{} 3'.format(emb_name)] for name in sample_names])
            ax.scatter(em1, em2, em3, s=size, label=group_name, c=[colors[i]])

    x_label = '{} 1'.format(emb_name)
    y_label = '{} 2'.format(emb_name)
    z_label = None if plot_2D else '{} 3'.format(emb_name)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if z_label is not None:
        ax.set_zlabel(z_label)
    if title is not None:
        ax.set_title(title)
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()


def plot_hist(df_1: pd.DataFrame,
              df_1_name: str,
              df_2: Optional[pd.DataFrame] = None,
              df_2_name: Optional[str] = None,
              sum_axis: int = 0,
              label: str = "Sample",
              figsize: Tuple[int, int] = (10, 8)) -> None:
    """Plot library-size histograms for one or two QC matrices.

    Parameters
    ----------
    df_1
        Genes-by-cells (or cells-by-genes) DataFrame.
    df_1_name
        Legend label for ``df_1``.
    df_2
        Optional second DataFrame plotted on the same axes.
    df_2_name
        Legend label for ``df_2``.
    sum_axis
        Axis to sum over before histogramming (``0`` for genes-by-cells,
        ``1`` for cells-by-genes).
    label
        X-axis label for the histogram.
    figsize
        Figure size in inches.
    """
    fig, ax = plt.subplots(figsize=figsize)
    df_1 = df_1.copy()
    df_2 = df_2.copy() if df_2 is not None else None
    if sum_axis == 0:
        df_1 = df_1.T
        df_2 = df_2.T if df_2 is not None else None
    elif sum_axis != 1:
        raise ValueError("Passed df should be a 2D df")
    df_1 = df_1.sum(axis=1).to_frame()
    df_2 = df_2.sum(axis=1).to_frame() if df_2 is not None else None
    df_1.columns = [label]
    df_1["name"] = df_1_name
    if df_2 is not None:
        df_2.columns = [label]
        df_2["name"] = df_2_name
        df_1 = pd.concat([df_1, df_2])
        sns.histplot(data=df_1, x=label, hue="name", ax=ax)
    else:
        sns.histplot(data=df_1, x=label, ax=ax)
    plt.show()
