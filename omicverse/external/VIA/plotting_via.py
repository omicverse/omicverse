import numpy as np
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
from scipy.stats import norm as normal
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import distance
from scipy.sparse import csr_matrix, csgraph, find
import math
import pandas as pd
import numpy as np
from numpy import ndarray
from scipy.sparse import issparse, spmatrix

import time
import matplotlib
import igraph as ig
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.cm as cm

from sklearn.preprocessing import normalize
from typing import Optional, Union
#from utils_via import *
from .utils_via import *
import random
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

# Numpy 2.0 compatibility: trapz renamed to trapezoid
try:
    from numpy import trapezoid as np_trapz
except ImportError:
    from numpy import trapz as np_trapz


def _pl_velocity_embedding(via_object, X_emb, smooth_transition, b, use_sequentially_augmented=False):
    '''

    :param X_emb:
    :param smooth_transition:
    :return: V_emb
    '''
    # T transition matrix at single cell level
    n_obs = X_emb.shape[0]
    V_emb = np.zeros(X_emb.shape)
    print('inside _plotting')
    if via_object.single_cell_transition_matrix is None:
        print('inside _plotting compute sc_transitionmatrix')
        via_object.single_cell_transition_matrix = via_object.sc_transition_matrix(smooth_transition, b,
                                                                       use_sequentially_augmented=use_sequentially_augmented)
        T = via_object.single_cell_transition_matrix
    else:
        print('get _plotting compute sc_transitionmatrix')
        T = via_object.single_cell_transition_matrix

    # the change in embedding distance when moving from cell i to its neighbors is given by dx
    for i in range(n_obs):
        indices = T[i].indices
        dX = X_emb[indices] - X_emb[i, None]  # shape (n_neighbors, 2)
        dX /= l2_norm(dX)[:, None]

        # dX /= np.sqrt(dX.multiply(dX).sum(axis=1).A1)[:, None]
        dX[np.isnan(dX)] = 0  # zero diff in a steady-state
        # neighbor edge weights are used to weight the overall dX or velocity from cell i.
        probs = T[i].data
        # if probs.size ==0: print('velocity embedding probs=0 length', probs, i, self.true_label[i])
        V_emb[i] = probs.dot(dX) - probs.mean() * dX.sum(0)
    V_emb /= 3 * quiver_autoscale(X_emb, V_emb)
    return V_emb


def geodesic_distance(data: ndarray, knn: int = 10, root: int = 0, mst_mode: bool = False,
                      cluster_labels: ndarray = None):
    n_samples = data.shape[0]
    # make knn graph on low dimensional data "data"
    knn_struct = construct_knn_utils(data, knn=knn)
    neighbors, distances = knn_struct.knn_query(data, k=knn)
    msk = np.full_like(distances, True, dtype=np.bool_)
    # https://igraph.org/python/versions/0.10.1/tutorials/shortest_paths/shortest_paths.html
    # Remove self-loops
    msk &= (neighbors != np.arange(neighbors.shape[0])[:, np.newaxis])
    rows = np.array([np.repeat(i, len(x)) for i, x in enumerate(neighbors)])[msk]
    cols = neighbors[msk]
    weights = distances[msk]  # we keep the distances as the weights here will actually be edge distances
    result = csr_matrix((weights, (rows, cols)), shape=(len(neighbors), len(neighbors)), dtype=np.float32)

    if mst_mode:
        print(f'MST geodesic mode')
        from scipy.sparse.csgraph import minimum_spanning_tree
        MST_ = minimum_spanning_tree(result)
        result = result + MST_
    result.eliminate_zeros()
    sources, targets = result.nonzero()
    edgelist = list(zip(sources.tolist(), targets.tolist()))

    G = ig.Graph(edgelist, edge_attrs={'weight': result.data.tolist()})
    if cluster_labels is not None:
        graph = ig.VertexClustering(G, membership=cluster_labels).cluster_graph(combine_edges='sum')

        graph = recompute_weights(graph, Counter(cluster_labels))  # returns csr matrix

        weights = graph.data / (np.std(graph.data))

        edges = list(zip(*graph.nonzero()))

        G = ig.Graph(edges, edge_attrs={'weight': weights})
        root = cluster_labels[root]
    # get shortest distance from root to each point
    geo_distance_list = []
    print(f'start computing shortest paths')
    for i in range(G.vcount()):
        if cluster_labels is None:
            if i % 1000 == 0: print(f'{datetime.now()}\t{i} out of {n_samples} complete')
        shortest_path = G.get_shortest_paths(root, to=i, weights=G.es["weight"], output="epath")

        if len(shortest_path[0]) > 0:
            # Add up the weights across all edges on the shortest path
            distance = 0
            for e in shortest_path[0]:
                distance += G.es[e]["weight"]
            geo_distance_list.append(distance)
            # print("Shortest weighted distance is: ", distance)
        else:
            geo_distance_list.append(0)
    return geo_distance_list


def corr_geodesic_distance_lowdim(embedding, knn=10, time_labels: list = [], root: int = 0,
                                  saveto='/home/shobi/Trajectory/Datasets/geodesic_distance.csv',
                                  mst_mode: bool = False, cluster_labels: ndarray = None):
    geodesic_dist = geodesic_distance(embedding, knn=knn, root=root, mst_mode=mst_mode, cluster_labels=cluster_labels)
    df_ = pd.DataFrame()

    df_['true_time'] = time_labels
    if cluster_labels is not None:
        df_['cluster_labels'] = cluster_labels
        df_ = df_.sort_values(['cluster_labels'], ascending=True).groupby('cluster_labels').mean()
        print('df_groupby', df_.head())
    df_['geo'] = geodesic_dist
    df_['geo'] = df_['geo'].fillna(0)

    correlation = df_['geo'].corr(df_['true_time'])
    print(f'{datetime.now()}\tcorrelation geo 2d and true time, {correlation}')
    df_.to_csv(saveto)
    return correlation


def make_edgebundle_milestone(embedding: ndarray = None, sc_graph=None, via_object=None, sc_pt: list = None,
                              initial_bandwidth=0.03, decay=0.7, n_milestones: int = None, milestone_labels: list = [],
                              sc_labels_numeric: list = None, weighted: bool = True, global_visual_pruning: float = 0.5,
                              terminal_cluster_list: list = [], single_cell_lineage_prob: ndarray = None,
                              random_state: int = 0):
    '''
    Perform Edgebundling of edges in a milestone level to return a hammer bundle of milestone-level edges. This is more granular than the original parc-clusters but less granular than single-cell level and hence also less computationally expensive
    requires some type of embedding (n_samples x 2) to be available

    :param embedding: optional (not required if via_object is provided) embedding single cell. also looks nice when done on via_mds as more streamlined continuous diffused graph structure. Umap is a but "clustery"
    :param graph: optional (not required if via_object is provided) igraph single cell graph level
    :param via_object: via_object (best way to run this function by simply providing via_object)
    :param sc_graph: igraph graph set as the via attribute self.ig_full_graph (affinity graph)
    :param initial_bandwidth: increasing bw increases merging of minor edges
    :param decay: increasing decay increases merging of minor edges #https://datashader.org/user_guide/Networks.html
    :param milestone_labels: default list=[]. Usually autocomputed. but can provide as single-cell level labels (clusters, groups, which function as milestone groupings of the single cells)
    :param sc_labels_numeric: default is None which automatically chooses via_object's pseudotime or time_series_labels (when available). otherwise set to a list of numerical values representing some sequential/chronological information
    :param terminal_cluster_list: default list [] and automatically uses all terminal clusters. otherwise set to any of the terminal cluster numbers within a list
    :param global_visual_pruning: prune the edges of the visualized StaVia clustergraph before edgebundling. default =0.5. Can take values (float) 0-3 (standard deviations), smaller number means fewer edges retained
    :return: dictionary containing keys: hb_dict['hammerbundle'] = hb hammerbundle class with hb.x and hb.y containing the coords
                hb_dict['milestone_embedding'] dataframe with 'x' and 'y' columns for each milestone and hb_dict['edges'] dataframe with columns ['source','target'] milestone for each each and ['cluster_pop'], hb_dict['sc_milestone_labels'] is a list of milestone label for each single cell

    '''
    if embedding is None:
        if via_object is not None: embedding = via_object.embedding

    if sc_graph is None:
        if via_object is not None: sc_graph = via_object.ig_full_graph
    if embedding is None:
        if via_object is None:
            print(f'{datetime.now()}\tERROR: Please provide via_object')
            return
        else:
            print(
                f'{datetime.now()}\tWARNING: VIA will now autocompute an embedding. It would be better to precompute an embedding using embedding = via_umap() or via_mds() and setting this as the embedding attribute via_object = embedding.')
            embedding = via_mds(via_object=via_object, random_seed=random_state)
    n_samples = embedding.shape[0]
    if n_milestones is None:
        n_milestones = min(via_object.nsamples, min(250, int(0.1 * via_object.nsamples)))
        print(f'{datetime.now()}\t n_milestones is {n_milestones}')
    # milestone_indices = random.sample(range(n_samples), n_milestones)  # this is sampling without replacement
    if len(milestone_labels) == 0:
        print(f'{datetime.now()}\tStart finding milestones')
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_milestones, random_state=random_state, n_init=10).fit(embedding)
        milestone_labels = kmeans.labels_.flatten().tolist()
        #df_ = pd.DataFrame()
        #df_['kmeans'] = milestone_labels
        #df_.to_csv('/home/user/Trajectory/Datasets/Zebrafish_Lange2023/kmeans_milestones'+str(n_milestones)+'.csv')
        print(f'{datetime.now()}\tEnd milestones with {n_milestones}')
        # plt.scatter(embedding[:, 0], embedding[:, 1], c=milestone_labels, cmap='tab20', s=1, alpha=0.3)
        # plt.show()
    if sc_labels_numeric is None:
        if via_object is not None:
            sc_labels_numeric = via_object.time_series_labels
        else:
            print(
                f'{datetime.now()}\tWill use via-pseudotime for edges, otherwise consider providing a list of numeric labels (single cell level) or via_object')
    if sc_pt is None:
        sc_pt = via_object.single_cell_pt_markov
    '''
    numeric_val_of_milestone = []
    if len(sc_labels_numeric)>0:
        for cluster_i in set(milestone_labels):
            loc_cluster_i = np.where(np.asarray(milestone_labels)==cluster_i)[0]
            majority_ = func_mode(list(np.asarray(sc_labels_numeric)[loc_cluster_i]))
            numeric_val_of_milestone.append(majority_)
    '''
    vertex_milestone_graph = ig.VertexClustering(sc_graph, membership=milestone_labels).cluster_graph(
        combine_edges='sum')

    print(f'{datetime.now()}\tRecompute weights')
    vertex_milestone_graph = recompute_weights(vertex_milestone_graph, Counter(milestone_labels))
    print(f'{datetime.now()}\tpruning milestone graph based on recomputed weights')

    edgeweights_pruned_milestoneclustergraph, edges_pruned_milestoneclustergraph, comp_labels = pruning_clustergraph(
        vertex_milestone_graph,
        global_pruning_std=global_visual_pruning,
        preserve_disconnected=True,
        preserve_disconnected_after_pruning=False, do_max_outgoing=False)

    print(f'{datetime.now()}\tregenerate igraph on pruned edges')
    vertex_milestone_graph = ig.Graph(edges_pruned_milestoneclustergraph,
                                      edge_attrs={'weight': edgeweights_pruned_milestoneclustergraph}).simplify(
        combine_edges='sum')
    vertex_milestone_csrgraph = get_sparse_from_igraph(vertex_milestone_graph, weight_attr='weight')

    weights_for_layout = np.asarray(vertex_milestone_csrgraph.data)
    # clip weights to prevent distorted visual scale
    weights_for_layout = np.clip(weights_for_layout, np.percentile(weights_for_layout, 20),
                                 np.percentile(weights_for_layout,
                                               80))  # want to clip the weights used to get the layout
    # print('weights for layout', (weights_for_layout))
    # print('weights for layout std', np.std(weights_for_layout))

    weights_for_layout = weights_for_layout / np.std(weights_for_layout)
    # print('weights for layout post-std', weights_for_layout)
    # print(f'{datetime.now()}\tregenerate igraph after clipping')
    vertex_milestone_graph = ig.Graph(list(zip(*vertex_milestone_csrgraph.nonzero())),
                                      edge_attrs={'weight': list(weights_for_layout)})

    # layout = vertex_milestone_graph.layout_fruchterman_reingold()
    # embedding = np.asarray(layout.coords)

    # print(f'{datetime.now()}\tmake node dataframe')
    data_node = [node for node in range(embedding.shape[0])]
    nodes = pd.DataFrame(data_node, columns=['id'])
    nodes.set_index('id', inplace=True)
    nodes['x'] = embedding[:, 0]
    nodes['y'] = embedding[:, 1]
    nodes['pt'] = sc_pt
    if via_object is not None:
        terminal_cluster_list = via_object.terminal_clusters
        single_cell_lineage_prob = via_object.single_cell_bp_rownormed  # _rownormed#_rownormed does not make a huge difference whether or not rownorming is applied. (default not rownormed)
    if (len(terminal_cluster_list) > 0) and (single_cell_lineage_prob is not None):
        for i, c_i in enumerate(terminal_cluster_list):
            nodes['sc_lineage_probability_' + str(c_i)] = single_cell_lineage_prob[:, i]
    if sc_labels_numeric is not None:
        print(
            f'{datetime.now()}\tSetting numeric label as time_series_labels or other sequential metadata for coloring edges')
        nodes['numeric label'] = sc_labels_numeric

    else:
        print(f'{datetime.now()}\tSetting numeric label as single cell pseudotime for coloring edges')
        nodes['numeric label'] = sc_pt

    nodes['kmeans'] = milestone_labels
    group_pop = []
    for i in sorted(set(milestone_labels)):
        group_pop.append(milestone_labels.count(i))

    nodes_mean = nodes.groupby('kmeans').mean()
    nodes_mean['cluster population'] = group_pop

    edges = pd.DataFrame([e.tuple for e in vertex_milestone_graph.es], columns=['source', 'target'])

    edges['weight0'] = vertex_milestone_graph.es['weight']
    edges = edges[edges['source'] != edges['target']]

    # seems to work better when allowing the bundling to occur on unweighted representation and later using length of segments to color code significance
    if weighted == True:
        edges['weight'] = edges[
            'weight0']  # 1  # [1/i for i in edges['weight0']]np.where((edges['source_cluster'] != edges['target_cluster']) , 1,0.1)#[1/i for i in edges['weight0']]#
    else:
        edges['weight'] = 1
    print(f'{datetime.now()}\tMaking smooth edges')
    from datashader.bundling import hammer_bundle
    hb = hammer_bundle(nodes_mean, edges, weight='weight', initial_bandwidth=initial_bandwidth,
                       decay=decay)  # default bw=0.05, dec=0.7
    # hb.x and hb.y contain all the x and y coords of the points that make up the edge lines.
    # each new line segment is separated by a nan value
    # https://datashader.org/_modules/datashader/bundling.html#hammer_bundle
    # nodes_mean contains the averaged 'x' and 'y' milestone locations based on the embedding
    hb_dict = {}
    hb_dict['hammerbundle'] = hb
    hb_dict['milestone_embedding'] = nodes_mean
    hb_dict['edges'] = edges[['source', 'target']]
    hb_dict['sc_milestone_labels'] = milestone_labels

    return hb_dict


def plot_gene_trend_heatmaps(via_object, df_gene_exp: pd.DataFrame, marker_lineages: list = [], fontsize: int = 8,
                             cmap: str = 'viridis', normalize: bool = True, ytick_labelrotation: int = 0,
                             fig_width: int = 7):
    '''

    Plot the gene trends on heatmap: a heatmap is generated for each lineage (identified by terminal cluster number). Default selects all lineages

    :param via_object:
    :param df_gene_exp: pandas DataFrame single-cell level expression [cells x genes]
    :param marker_lineages: list default = None and plots all detected all lineages. Optionally provide a list of integers corresponding to the cluster number of terminal cell fates
    :param fontsize: int default = 8
    :param cmap: str default = 'viridis'
    :param normalize: bool = True
    :param ytick_labelrotation: int default = 0
    :return: fig and list of axes
    '''
    import seaborn as sns

    if len(marker_lineages) == 0: marker_lineages = via_object.terminal_clusters
    dict_trends = get_gene_trend(via_object=via_object, marker_lineages=marker_lineages, df_gene_exp=df_gene_exp)
    branches = list(dict_trends.keys())
    print('branches', branches)
    genes = dict_trends[branches[0]]['trends'].index
    height = len(genes) * len(branches)
    # Standardize the matrix (standardization along each gene. Since SS function scales the columns, we first transpose the df)
    #  Set up plot
    fig = plt.figure(figsize=[fig_width, height])
    ax_list = []
    for i, branch in enumerate(branches):
        ax = fig.add_subplot(len(branches), 1, i + 1)
        df_trends = dict_trends[branch]['trends']
        # normalize each genes (feature)
        if normalize == True:
            df_trends = pd.DataFrame(
                StandardScaler().fit_transform(df_trends.T).T,
                index=df_trends.index,
                columns=df_trends.columns)

        ax.set_title('Lineage: ' + str(branch) + '-' + str(dict_trends[branch]['name']), fontsize=int(fontsize * 1.3))
        # sns.set(size=fontsize)  # set fontsize 2
        b = sns.heatmap(df_trends, yticklabels=True, xticklabels=False, cmap=cmap)
        b.tick_params(labelsize=fontsize, labelrotation=ytick_labelrotation)
        b.figure.axes[-1].tick_params(labelsize=fontsize)
        ax_list.append(ax)
    b.set_xlabel("pseudotime", fontsize=int(fontsize * 1.3))
    return fig, ax_list


def plot_scatter(embedding: ndarray, labels: list, cmap='rainbow', s=5, alpha=0.3, edgecolors='None', title: str = '',
                 text_labels: bool = True, color_dict=None, via_object=None, sc_index_terminal_states: list = None,
                 true_labels: list = [], show_legend: bool = True, hide_axes_ticks:bool=True, color_labels_reverse:bool = False):
    '''
    General scatter plotting tool for numeric and categorical labels on the single-cell level

    :param embedding: ndarray n_samples x 2
    :param labels: list single cell labels list of number or strings
    :param cmap: str default = 'rainbow'
    :param s: int size of scatter dot
    :param alpha: float with 0 transparent to 1 opaque default =0.3
    :param edgecolors:
    :param title: str
    :param text_labels: bool default =True
    :param via_object:
    :param sc_index_terminal_states: list of integers corresponding to one cell in each of the terminal states
    :param color_dict: {'true_label_group_1': #COLOR,'true_label_group_2': #COLOR2,....} where the dictionary keys correspond to the provided labels
    :param true_labels: list of single cell labels used to annotate the terminal states
    :return: matplotlib pyplot fig, ax
    '''
    fig, ax = plt.subplots()
    if isinstance(labels[0], str):
        categorical = True
    else:
        categorical = False
    ax.set_facecolor('white')
    if categorical and color_dict is None:
        color_dict = {}
        set_labels = list(set(labels))
        set_labels.sort(reverse=color_labels_reverse)#True) #used to be True until Feb 2024
        palette = plt.get_cmap(cmap, len(set_labels))
        cmap_ = palette(range(len(set_labels)))
        for value, color in zip(set_labels,cmap_):
            color_dict[value] = color
    if color_dict is not None:
        #:param color_dict: {'true_label_group_1': #COLOR,'true_label_group_2': #COLOR2,....} where the dictionary keys correspond to the provided labels
        for key in color_dict:
            loc_key = np.where(np.asarray(labels) == key)[0]
            ax.scatter(embedding[loc_key, 0], embedding[loc_key, 1], color=color_dict[key], label=key, s=s,
                       alpha=alpha, edgecolors=edgecolors)
            x_mean = embedding[loc_key, 0].mean()
            y_mean = embedding[loc_key, 1].mean()
            if text_labels == True: ax.text(x_mean, y_mean, key, style='italic', fontsize=10, color="black")
    else:
        im = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels,
                        cmap=cmap, s=s, alpha=alpha)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical',
                     label='pseudotime')
        if via_object is not None:
            tsi_list = []
            for tsi in via_object.terminal_clusters:
                loc_i = np.where(np.asarray(via_object.labels) == tsi)[0]
                val_pt = [via_object.single_cell_pt_markov[i] for i in loc_i]

                th_pt = np.percentile(val_pt, 50)  # 50
                loc_i = [loc_i[i] for i in range(len(val_pt)) if val_pt[i] >= th_pt]
                temp = np.mean(via_object.data[loc_i], axis=0)
                labelsq, distances = via_object.knn_struct.knn_query(temp, k=1)

                tsi_list.append(labelsq[0][0])

            ax.set_title(
                'root:' + str(via_object.root_user[0]) + 'knn' + str(via_object.knn) + 'Ncomp' + str(via_object.ncomp))
            for i in tsi_list:
                # print(i, ' has traj and cell type', self.df_annot.loc[i, ['Main_trajectory', 'Main_cell_type']])
                ax.text(embedding[i, 0], embedding[i, 1], str(true_labels[i]) + '_Cell' + str(i))
                ax.scatter(embedding[i, 0], embedding[i, 1], c='black', s=10)
    if (via_object is None) & (sc_index_terminal_states is not None):
        for i in sc_index_terminal_states:
            ax.text(embedding[i, 0], embedding[i, 1], str(true_labels[i]) + '_Cell' + str(i))
            ax.scatter(embedding[i, 0], embedding[i, 1], c='black', s=10)

    if len(title) == 0:
        ax.set_title(label='scatter plot', color='blue')
    else:
        ax.set_title(label=title, color='blue')
    ax.grid(False)
    # Hide axes ticks
    if hide_axes_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    # Hide grid lines
    ax.grid(False)
    fig.patch.set_visible(True)
    if show_legend and categorical:
        leg_handles = []
        for label in color_dict:
            lh = ax.scatter([], [], c=color_dict[label], label=label)
            leg_handles.append(lh)
        ax.legend(
            handles=leg_handles, 
            fontsize=12, 
            frameon=False, 
            bbox_to_anchor=(1, 0.5),
            loc='center left', 
            ncol=(1 if len(color_dict) <= 14 else 2 if len(color_dict) <= 30 else 3)
            )
    return fig, ax


def _make_knn_embeddedspace(embedding):
    import hnswlib
    # knn struct built in the embedded space to be used for drawing the lineage trajectories onto the 2D plot
    knn = hnswlib.Index(space='l2', dim=embedding.shape[1])
    knn.init_index(max_elements=embedding.shape[0], ef_construction=200, M=16)
    knn.add_items(embedding)
    knn.set_ef(50)
    return knn


def via_forcelayout(X_pca, viagraph_full: csr_matrix = None, k: int = 10,
                    n_milestones=2000, time_series_labels: list = [],
                    knn_seq: int = 5, saveto='', random_seed: int = 0) -> ndarray:
    '''
    Compute force directed layout. #TODO not complete

    :param X_pca:
    :param viagraph_full: optional. if calling before via, then None. if calling after or from within via, then we can use the via-graph to reinforce the layout
    :param k:
    :param random_seed:
    :param t_diffusion:
    :param n_milestones:
    :param time_series_labels:
    :param knn_seq:
    :return: ndarray
    '''
    # use the csr_full_graph from via and subsample it.
    # but this results in a very fragmented graph because the subsampling index is too small a fraction of the total number of possible edges.
    # only works if you take a high enough percentage of the original samples

    print(f"{datetime.now()}\tCommencing Force Layout")
    np.random.seed(random_seed)
    milestone_indices = random.sample(range(X_pca.shape[0]), n_milestones)  # this is sampling without replacement
    if viagraph_full is not None:
        milestone_knn = viagraph_full[milestone_indices]  #
        milestone_knn = milestone_knn[:, milestone_indices]
        milestone_knn = normalize(milestone_knn, axis=1)

    knn_struct = construct_knn_utils(X_pca[milestone_indices, :], knn=k)
    # we need to add the new knn (milestone_knn_new) built on the subsampled indices to ensure connectivity. o/w graph is fragmented if only relying on the subsampled graph
    if time_series_labels is None: time_series_labels = []
    if len(time_series_labels) >= 1: time_series_labels = np.array(time_series_labels)[milestone_indices].tolist()
    milestone_knn_new = affinity_milestone_knn(data=X_pca[milestone_indices, :], knn_struct=knn_struct, k=k,
                                               time_series_labels=time_series_labels, knn_seq=knn_seq)

    print('milestone knn new', milestone_knn_new.shape, milestone_knn_new.data[0:10])
    if viagraph_full is None:
        milestone_knn = milestone_knn_new
    else:
        milestone_knn = milestone_knn + milestone_knn_new
    print('final reinforced milestone knn', milestone_knn.shape, 'number of nonzero edges', len(milestone_knn.data))

    print('force layout')
    g_layout = ig.Graph(list(zip(*milestone_knn.nonzero())))  # , edge_attrs={'weight': weights_for_layout})
    layout = g_layout.layout_fruchterman_reingold()
    force_layout = np.asarray(layout.coords)
    # compute knn used to estimate the embedding values of the full sample set based on embedding values computed just for a milestone subset of the full sample
    neighbor_array, distance_array = knn_struct.knn_query(X_pca, k=k)
    print('shape of ', X_pca.shape, neighbor_array.shape)
    row_mean = np.mean(distance_array, axis=1)
    row_var = np.var(distance_array, axis=1)
    row_znormed_dist_array = -(distance_array - row_mean[:, np.newaxis]) / row_var[:, np.newaxis]
    # when k is very small, then you can get very large affinities due to var being ~0
    row_znormed_dist_array = np.nan_to_num(row_znormed_dist_array, copy=True, nan=1, posinf=1, neginf=1)
    row_znormed_dist_array[row_znormed_dist_array > 10] = 0
    affinity_array = np.exp(row_znormed_dist_array)
    affinity_array = normalize(affinity_array, norm='l1', axis=1)  # row stoch

    row_list = []
    n_neighbors = neighbor_array.shape[1]
    n_cells = neighbor_array.shape[0]
    print('ncells and neighs', n_cells, n_neighbors)

    row_list.extend(list(np.transpose(np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten()))

    col_list = neighbor_array.flatten().tolist()

    list_affinity = affinity_array.flatten().tolist()

    csr_knn = csr_matrix((list_affinity, (row_list, col_list)),
                         shape=(n_cells, len(milestone_indices)))  # n_samples*n_milestones

    milestone_force = csr_matrix(force_layout)  ##TODO remove this we are just testing force layout
    full_force = csr_knn * milestone_force  # is a matrix
    full_force = np.asarray(full_force.todense())

    plt.scatter(full_force[:, 0].tolist(), full_force[:, 1].tolist(), s=1, alpha=0.3, c='red')
    plt.title('full mds')
    plt.show()
    full_force = np.reshape(full_force, (n_cells, 2))
    if len(saveto) > 0:
        U_df = pd.DataFrame(full_force)
        U_df.to_csv(saveto)
    return full_force


def via_mds(via_object=None, X_pca: ndarray = None, viagraph_full: csr_matrix = None, k: int = 15,
            random_seed: int = 0, diffusion_op: int = 1, n_milestones=2000, time_series_labels: list = [],
            knn_seq: int = 5, k_project_milestones: int = 3, t_difference: int = 2, saveto='',
            embedding_type: str = 'mds', double_diffusion: bool = False) -> ndarray:
    '''

    Fast computation of a 2D embedding
    FOR EXAMPLE:
    via_object.embedding = via.via_mds(via_object = v0)
    plot_scatter(embedding = via_object.embedding, labels = via_object.true_labels)

    :param via_object:
    :param X_pca: dimension reduced (only if via_object is not passed)
    :param viagraph_full: optional. if calling before or without via, then None and a milestone graph will be computed. if calling after or from within via, then we can use the via-graph to reinforce the layout of the milestone graph
    :param k: number of knn for the via_mds reinforcement graph on milestones. default =15. integers 5-20 are reasonable
    :param random_seed: randomseed integer
    :param t_diffusion: default integer value = 1 with higher values generate more smoothing
    :param n_milestones: number of milestones used to generate the initial embedding
    :param time_series_labels: numerical values in list form representing some sequentual information
    :param knn_seq: if time-series data is available, this will augment the knn with sequential neighbors (2-10 are reasonable values) default =5
    :param embedding_type: default = 'mds' or set to 'umap'
    :param double_diffusion: default is False. To achieve sharper strokes/lineages, set to True
    :param k_project_milestones: number of milestones in the milestone-knngraph used to compute the single-cell projection
    :param n_iterations: number of iterations to run
    :param neighbors_distances: array of distances of each neighbor for each cell (n_cells x knn) used when called from within via.run() for autocompute via-mds
    :return: numpy array of size n_samples x 2
    '''

    # use the csr_full_graph from via and subsample it.
    # but this results in a very fragmented graph because the subsampling index is too small a fraction of the total number of possible edges.
    # only works if you take a high enough percentage of the original samples
    # however, omitting the integration of csr_full_graph also compromises the ability of the embedding to better reflect the underlying trajectory in terms of global structure

    print(f"{datetime.now()}\tCommencing Via-MDS")
    if via_object is not None:
        if X_pca is None: X_pca = via_object.data
        if viagraph_full is None: viagraph_full = via_object.csr_full_graph
    n_samples = X_pca.shape[0]
    if n_milestones is None:
        n_milestones = min(n_samples, max(2000, int(0.01 * n_samples)))
    if n_milestones > n_samples:
        n_milestones = min(n_samples, max(2000, int(0.01 * n_samples)))
        print(f"{datetime.now()}\tResetting n_milestones to {n_milestones} as n_samples > original n_milestones")

    '''
    if n_milestones < n_samples:
        if via_object is not None: milestone_indices = density_sampling(neighbors_distances= via_object.full_neighbor_array, desired_samples = n_milestones)
        else: milestone_indices = density_sampling(neighbors_distances= neighbors_distances, desired_samples = n_milestones)
        print(f'number of milestone indices from density sampling {milestone_indices.shape}')
        print('exp=True, dens sampling')
    '''
    np.random.seed(random_seed)

    milestone_indices = random.sample(range(X_pca.shape[0]), n_milestones)  # this is sampling without replacement

    if viagraph_full is not None:
        milestone_knn = viagraph_full[milestone_indices]  #
        milestone_knn = milestone_knn[:, milestone_indices]
        milestone_knn = normalize(milestone_knn,
                                  axis=1)  # using these effectively emphasises the edges that are pass an even more stringent requirement on Nearest neighbors (since they are selected from the full set of cells, rather than a subset of milestones)
    X_pca[milestone_indices, :]
    knn_struct = construct_knn_utils(X_pca[milestone_indices, :], knn=k)
    # we need to add the new knn (milestone_knn_new) built on the subsampled indices to ensure connectivity. o/w graph is fragmented if only relying on the subsampled graph
    if time_series_labels is None: time_series_labels = []
    if len(time_series_labels) >= 1: time_series_labels = np.array(time_series_labels)[milestone_indices].tolist()
    milestone_knn_new = affinity_milestone_knn(data=X_pca[milestone_indices, :], knn_struct=knn_struct, k=k,
                                               time_series_labels=time_series_labels, knn_seq=knn_seq,
                                               t_difference=t_difference)

    if viagraph_full is None:
        milestone_knn = milestone_knn_new
    else:
        milestone_knn = milestone_knn + milestone_knn_new

    # build a knn to project the input n_samples based on milestone knn

    neighbor_array, distance_array = knn_struct.knn_query(X_pca, k=k_project_milestones)  # [n_samples x n_milestones]

    row_mean = np.mean(distance_array, axis=1)
    row_var = np.var(distance_array, axis=1)
    row_znormed_dist_array = -(distance_array - row_mean[:, np.newaxis]) / row_var[:, np.newaxis]
    # when k is very small, then you can get very large affinities due to var being ~0
    row_znormed_dist_array = np.nan_to_num(row_znormed_dist_array, copy=True, nan=1, posinf=1, neginf=1)
    row_znormed_dist_array[row_znormed_dist_array > 10] = 0
    affinity_array = np.exp(row_znormed_dist_array)
    affinity_array = normalize(affinity_array, norm='l1', axis=1)  # row stoch

    row_list = []
    n_neighbors = neighbor_array.shape[1]
    n_cells = neighbor_array.shape[0]

    row_list.extend(list(np.transpose(np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten()))

    col_list = neighbor_array.flatten().tolist()

    list_affinity = affinity_array.flatten().tolist()

    csr_knn = csr_matrix((list_affinity, (row_list, col_list)),
                         shape=(n_cells, len(milestone_indices)))  # n_samples*n_milestones
    print(f"{datetime.now()}\tStart computing with diffusion power:{diffusion_op}")

    # r2w_input = pd.read_csv(        '/home/shobi/Trajectory/Datasets/EB_Phate/RW2/pc20_knn100kseq50krev50RW2_sparse_matrix029_P1_Q10.csv')
    # r2w_input = r2w_input.drop(['Unnamed: 0'], axis=1).values
    # input = r2w_input[:, 0:30]
    # input = input[milestone_indices, :]
    # print('USING RW2 COMPS')

    if embedding_type == 'mds':
        milestone_mds = sgd_mds(via_graph=milestone_knn, X_pca=X_pca[milestone_indices, :], diff_op=diffusion_op,
                                ndims=2,
                                random_seed=random_seed, double_diffusion=double_diffusion)  # returns an ndarray
    elif embedding_type == 'umap':
        milestone_mds = via_umap(X_input=X_pca[milestone_indices, :], graph=milestone_knn)

    print(f"{datetime.now()}\tEnd computing mds with diffusion power:{diffusion_op}")
    # TESTING
    # plt.scatter(milestone_mds[:, 0], milestone_mds[:, 1], s=1)
    # plt.title('sampled')
    # plt.show()

    milestone_mds = csr_matrix(milestone_mds)

    full_mds = csr_knn * milestone_mds  # is a matrix
    full_mds = np.asarray(full_mds.todense())

    # TESTING
    # plt.scatter(full_mds[:, 0].tolist(), full_mds[:, 1].tolist(), s=1, alpha=0.3, c='green')
    # plt.title('full')
    # plt.show()
    full_mds = np.reshape(full_mds, (n_cells, 2))

    if len(saveto) > 0:
        U_df = pd.DataFrame(full_mds)
        U_df.to_csv(saveto)

    return full_mds


def via_atlas_emb(via_object=None, X_input: ndarray = None, graph: csr_matrix = None, n_components: int = 2,
                  alpha: float = 1.0, negative_sample_rate: int = 5,
                  gamma: float = 1.0, spread: float = 1.0, min_dist: float = 0.1, init_pos: Union[str, ndarray] = 'via',
                  random_state: int = 0,
                  n_epochs: int = 100, distance_metric: str = 'euclidean', layout: Optional[list] = None,
                  cluster_membership: Optional[list] = None, parallel: bool = False, saveto='',
                  n_jobs: int = 2) -> ndarray:
    '''

    Run dimensionality reduction using the VIA modified HNSW graph using via cluster graph initialization when Via_object is provided

    :param via_object: if via_object is provided then X_input and graph are ignored
    :param X_input: ndarray nsamples x features (PCs)
    :param graph: csr_matrix of knngraph. This usually is via's pruned, sequentially augmented sc-knn graph accessed as an attribute of via via_object.csr_full_graph
    :param n_components:
    :param alpha:
    :param negative_sample_rate:
    :param gamma: Weight to apply to negative samples.
    :param spread: The effective scale of embedded points. In combination with min_dist this determines how clustered/clumped the embedded points are.
    :param min_dist: The effective minimum distance between embedded points. Smaller values will result in a more clustered/clumped embedding where nearby points on the manifold are drawn closer together, while larger values will result on a more even dispersal of points
    :param init_pos: either a string (default) 'via' (uses via graph to initialize), or 'spectral'. Or a n_cellx2 dimensional ndarray with initial coordinates
    :param random_state:
    :param n_epochs: The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings. If 0 is specified a value will be selected based on the size of the input dataset (200 for large datasets, 500 for small).
    :param distance_metric:
    :param layout: ndarray . custom initial layout. (n_cells x2). also requires cluster_membership labels
    :param cluster_membership: via_object.labels (cluster level labels of length n_samples corresponding to the layout)
    :return: ndarray of shape (nsamples,n_components)
    '''
    if via_object is None:
        if (X_input is None) or (graph is None):
            print(f"{datetime.now()}\tERROR: please provide both X_input and graph")

    if via_object is not None:
        if X_input is None: X_input = via_object.data
        print('X-input', X_input.shape)
        graph = via_object.csr_full_graph
        if cluster_membership is None: cluster_membership = via_object.labels
    # X_input = via0.data
    n_cells = X_input.shape[0]
    print('len membership and n_cells', len(cluster_membership), n_cells)
    print(f'n cell {n_cells}')
    # graph = graph+graph.T
    # graph = via0.csr_full_graph
    print(f"{datetime.now()}\tComputing embedding on sc-Viagraph")

    from umap.umap_ import find_ab_params, simplicial_set_embedding
    # graph is a csr matrix
    # weight all edges as 1 in order to prevent umap from pruning weaker edges away
    layout_array = np.zeros(shape=(n_cells, 2))

    if (init_pos == 'via') and (via_object is None):
        # list of lists [[x,y], [x1,y1], []]
        if (layout is None) or (cluster_membership is None):
            print('please provide via object or values for arguments: layout and cluster_membership')
        else:
            for i in range(n_cells):
                layout_array[i, 0] = layout[cluster_membership[i]][0]
                layout_array[i, 1] = layout[cluster_membership[i]][1]
            init_pos = layout_array
            print(f'{datetime.now()}\tusing via cluster graph to initialize embedding')
    elif (init_pos == 'via') and (via_object is not None):
        layout = via_object.graph_node_pos
        cluster_membership = via_object.labels
        for i in range(n_cells):
            layout_array[i, 0] = layout[cluster_membership[i]][0]
            layout_array[i, 1] = layout[cluster_membership[i]][1]
        init_pos = layout_array
        print(f'{datetime.now()}\tusing via cluster graph to initialize embedding')
    elif init_pos == 'spatial':
        init_pos = layout
    a, b = find_ab_params(spread, min_dist)
    # print('a,b, spread, dist', a, b, spread, min_dist)
    t0 = time.time()
    # m = graph.data.max()
    graph.data = np.clip(graph.data, np.percentile(graph.data, 1), np.percentile(graph.data, 99))
    # graph.data = 1 + graph.data/m
    # graph.data.fill(1)
    # print('average graph.data', round(np.mean(graph.data),4), round(np.max(graph.data),2))
    # graph.data = graph.data + np.mean(graph.data)

    # transpose =graph.transpose()

    # prod_matrix = graph.multiply(transpose)
    # graph = graph + transpose - prod_matrix
    if parallel:
        import numba
        print('before setting numba threads')
        print(f'there are {numba.get_num_threads()} threads')
        numba.set_num_threads(n_jobs)
        print(f'there are now {numba.get_num_threads()} threads')
    random_state = np.random.RandomState(random_state)
    if parallel:
        print('using parallel, the random_state will not be used.')

        do_randomize_init = True
        if do_randomize_init:
            init_pos = init_pos + random_state.normal(
                scale=0.001, size=init_pos.shape
            ).astype(np.float32)
    X_emb, aux_data = simplicial_set_embedding(data=X_input, graph=graph, n_components=n_components,
                                               initial_alpha=alpha,
                                               a=a, b=b, n_epochs=n_epochs, metric_kwds={}, gamma=gamma,
                                               metric=distance_metric,
                                               negative_sample_rate=negative_sample_rate, init=init_pos,
                                               random_state=random_state,
                                               verbose=1, output_dens=False, densmap_kwds={}, densmap=False,
                                               parallel=parallel)

    if len(saveto) > 0:
        U_df = pd.DataFrame(X_emb)
        U_df.to_csv(saveto)
    return X_emb


def run_umap_hnsw(via_object=None, X_input: ndarray = None, graph: csr_matrix = None, n_components: int = 2,
                  alpha: float = 1.0, negative_sample_rate: int = 5,
                  gamma: float = 1.0, spread: float = 1.0, min_dist: float = 0.1,
                  init_pos: Union[str, ndarray] = 'spectral', random_state: int = 0,
                  n_epochs: int = 0, distance_metric: str = 'euclidean', layout: Optional[list] = None,
                  cluster_membership: list = [], saveto='') -> ndarray:
    print(f"{datetime.now()}\tWarning: in future call via_umap() to run this function")

    return via_umap(via_object=via_object, X_input=X_input, graph=graph, n_components=n_components, alpha=alpha,
                    negative_sample_rate=negative_sample_rate,
                    gamma=gamma, spread=spread, min_dist=min_dist, init_pos=init_pos, random_state=random_state,
                    n_epochs=n_epochs, distance_metric=distance_metric, layout=layout,
                    cluster_membership=cluster_membership, saveto=saveto)


def plot_population_composition(via_object, time_labels: list = None, celltype_list: list = None, cmap: str = 'rainbow',
                                legend: bool = True,
                                alpha: float = 0.5, linewidth: float = 0.2, n_intervals: int = 20, xlabel: str = 'time',
                                ylabel: str = '', title: str = 'Cell populations', color_dict: dict = None,
                                fraction: bool = True):
    '''
    :param via_object: optional. this is required unless both time_labels and cell_labels are provided as arguments to the function
    :param time_labels: list length n_cells of pseudotime or known stage numeric labels
    :param cell_labels:  list of cell type or cluster length n_cells
    :return: ax
    '''

    if time_labels is None:
        pt = via_object.single_cell_pt_markov
        maxpt = max(pt)
        pt = [i / maxpt for i in pt]
    else:
        pt = time_labels
    maxpt = max(pt)
    minpt = min(pt)
    if celltype_list is None: celltype_list = via_object.true_label
    df_full = pd.DataFrame()
    df_full['pt'] = [i for i in pt]
    df_full['celltype'] = celltype_list
    print(f'head df full {df_full.head()}')
    n_intervals = n_intervals
    interval_step = (max(pt) - min(pt)) / n_intervals
    interval_i = 0
    from collections import Counter
    set_celltype_sorted = list(sorted(list(set(celltype_list))))
    df_population = pd.DataFrame(0, index=[minpt + (i) * interval_step for i in range(n_intervals)],
                                 columns=set_celltype_sorted)

    index_i = 0
    while interval_i <= max(pt) + 0.01:
        df_temp = df_full[((df_full['pt'] < interval_i + interval_step) & (df_full['pt'] >= minpt + interval_i))]

        dict_temp = Counter(df_temp['celltype'])
        if fraction:
            n_samples_temp = df_temp.shape[0]
            for key in dict_temp:
                dict_temp[key] = dict_temp[key] / n_samples_temp
        print('dict temp', dict_temp)
        dict_temp = dict(sorted(dict_temp.items()))

        interval_i += interval_step
        for key_pop_i in dict_temp:
            df_population.loc[minpt + (index_i + 1) * interval_step, key_pop_i] = dict_temp[key_pop_i]
        index_i += 1
    title = title + 'n_intervals' + str(n_intervals)
    if color_dict is not None:
        ax = df_population.plot.area(grid=False, legend=legend, color=color_dict, alpha=alpha, linewidth=linewidth,
                                     xlabel=xlabel, ylabel=ylabel, title=title)
    else:
        ax = df_population.plot.area(grid=False, legend=legend, colormap=cmap, alpha=alpha, linewidth=linewidth,
                                     xlabel=xlabel, ylabel=ylabel, title=title)

    return ax


def plot_differentiation_flow(via_object, idx: list = None, dpi=150, marker_lineages=[], label_node: list = [],
                              do_log_flow: bool = True, fontsize: int = 8, alpha_factor: float = 0.9,
                              majority_cluster_population_dict: dict = None, cmap_sankey='rainbow',
                              title_str: str = 'Differentiation Flow', root_cluster_list: list = None):
    '''
    #SANKEY PLOTS
    G is the igraph knn (low K) used for shortest path in high dim space. no idx needed as it's made on full sample
    knn_hnsw is the knn made in the embedded space used for query to find the nearest point in the downsampled embedding
    that corresponds to the single cells in the full graph

    :param via_object:
    :param embedding: n_samples x 2. embedding is 2D representation of the full dataset.
    :param idx: if one uses a downsampled embedding of the original data, then idx is the selected indices of the downsampled samples used in the visualization
    :param cmap_name:
    :param dpi:
    :param do_log_flow bool True (default) take the natural log (1+edge flow value)
    :param label_node list of labels for each cell (could be cell type, stage level) length is n_cells
    :param scatter_size: if None, then auto determined based on n_cells
    :param marker_lineages: Default is to use all lineage pathways. other provide a list of lineage number (terminal cluster number).
    :param alpha_factor: float transparency
    :param root_cluster_list: list of roots by cluster number e.g. [5] means a good root is cluster number 5
    :return: fig, axs
    '''
    import math
    import hnswlib

    if len(marker_lineages) == 0:
        marker_lineages = via_object.terminal_clusters
    if root_cluster_list is None:
        root_cluster_list = via_object.root


    else:
        marker_lineages = [i for i in marker_lineages if i in via_object.labels]  # via_object.terminal_clusters]
    print(f'{datetime.now()}\tMarker_lineages: {marker_lineages}')
    '''
    if embedding is None:
        if via_object.embedding is None:
            print('ERROR: please provide a single cell embedding or run re-via with do_compute_embedding==True using either embedding_type = via-umap OR via-mds')
            return
        else:
            print(f'automatically setting embedding to via_object.embedding')
            embedding = via_object.embedding
    '''
    # make the sankey node labels either using via_obect.true_label or the labels provided by the user
    print(f'{datetime.now()}\tStart dictionary modes')
    df_mode = pd.DataFrame()
    df_mode['cluster'] = via_object.labels
    # df_mode['celltype'] = pre_labels_celltype_df['fine'].tolist()#v0.true_label
    if len(label_node) > 0:
        df_mode['celltype'] = label_node  # v0.true_label
    else:
        df_mode['celltype'] = via_object.true_label
    majority_cluster_population_dict = df_mode.groupby(['cluster'])['celltype'].agg(
        lambda x: pd.Series.mode(x)[0])  # agg(pd.Series.mode would give all modes) #series
    majority_cluster_population_dict = majority_cluster_population_dict.to_dict()
    print(f'{datetime.now()}\tEnd dictionary modes')

    if idx is None: idx = np.arange(0, via_object.nsamples)
    # G = via_object.full_graph_shortpath
    n_original_comp, n_original_comp_labels = connected_components(via_object.csr_full_graph, directed=False)
    # G = via_object.full_graph_paths(via_object.data, n_original_comp)
    # knn_hnsw = _make_knn_embeddedspace(embedding)
    y_root = []
    x_root = []
    root1_list = []
    p1_sc_bp = np.nan_to_num(via_object.single_cell_bp[idx, :], nan=0.0, posinf=0.0, neginf=0.0)
    # row normalize
    row_sums = p1_sc_bp.sum(axis=1)
    p1_sc_bp = p1_sc_bp / row_sums[:,
                          np.newaxis]  # make rowsums a column vector where i'th entry is sum of i'th row in p1-sc-bp
    print(f'{datetime.now()}\tCheck sc pb {p1_sc_bp[0, :].sum()} ')

    p1_labels = np.asarray(via_object.labels)[idx]

    p1_cc = via_object.connected_comp_labels
    p1_sc_pt_markov = list(np.asarray(via_object.single_cell_pt_markov)[idx])
    X_data = via_object.data

    X_ds = X_data[idx, :]
    p_ds = hnswlib.Index(space='l2', dim=X_ds.shape[1])
    p_ds.init_index(max_elements=X_ds.shape[0], ef_construction=200, M=16)
    p_ds.add_items(X_ds)
    p_ds.set_ef(max(1, min(50, X_ds.shape[0])))
    num_cluster = len(set(via_object.labels))
    G_orange = ig.Graph(n=num_cluster, edges=via_object.edgelist_maxout,
                        edge_attrs={'weight': via_object.edgeweights_maxout})
    for ii, r_i in enumerate(root_cluster_list):
        sankey_edges = []
        '''
        loc_i = np.where(p1_labels == via_object.root[ii])[0]
        x = [embedding[xi, 0] for xi in loc_i]
        y = [embedding[yi, 1] for yi in loc_i]

        labels_root, distances_root = knn_hnsw.knn_query(np.array([np.mean(x), np.mean(y)]), k=1)
        x_root.append(embedding[labels_root, 0][0])
        y_root.append(embedding[labels_root, 1][0])

        labelsroot1, distances1 = via_object.knn_struct.knn_query(X_ds[labels_root[0][0], :], k=1)
        root1_list.append(labelsroot1[0][0])

        print('f getting majority comp')
        '''
        '''
        #VERY SLOW maybe try with dataframe mode  df.groupby(['team'])['points'].agg(pd.Series.mode)
        for labels_i in via_object.labels:
            loc_labels = np.where(np.asarray(via_object.labels) == labels_i)[0]
            majority_composition = func_mode(list(np.asarray(via_object.true_label)[loc_labels]))
            majority_cluster_population_dict[labels_i] = majority_composition
        print('f End getting majority comp')
        '''
        for fst_i in marker_lineages:

            path_orange = G_orange.get_shortest_paths(root_cluster_list[ii], to=fst_i)[0]
            '''
            if fst_i in [1,22,71,89,136,10,83,115]: #CNS and periderm for Zebrahub Lange we want the root to be the early CNS
                #path_orange = G_orange.get_shortest_paths(52, to=fst_i)[0]
                #path_orange = G_orange.get_shortest_paths(via_object.root[ii], to=fst_i)[0]
                path_orange = G_orange.get_shortest_paths(3, to=fst_i)[0] #for Zebralange use this CNS root
            elif fst_i in [10,69,90,95]:
                path_orange = G_orange.get_shortest_paths(75, to=fst_i)[0]  # for Zebralange use this endoderm (pharynx, liver, intestine) root
            else:
                path_orange = G_orange.get_shortest_paths(via_object.root[ii], to=fst_i)[0]
            '''
            # path_orange = G_orange.get_shortest_paths(3, to=fst_i)[0]
            # if the roots is in the same component as the terminal cluster, then print the path to output
            if len(path_orange) > 0:
                print(
                    f'{datetime.now()}\tCluster path on clustergraph starting from Root Cluster {root_cluster_list[ii]} to Terminal Cluster {fst_i}: {path_orange}')
                do_sankey = True
                '''
                cluster_population_dict = {}
                for group_i in set(via_object.labels):
                    loc_i = np.where(via_object.labels == group_i)[0]
                    cluster_population_dict[group_i] = len(loc_i)
                '''
                if do_sankey:

                    import holoviews as hv
                    hv.extension('bokeh')
                    from bokeh.plotting import show
                    from holoviews import opts, dim

                    print(f"{datetime.now()}\tHoloviews for TC {fst_i}")

                    cluster_adjacency = via_object.cluster_adjacency
                    # row normalize
                    row_sums = cluster_adjacency.sum(axis=1)
                    cluster_adjacency_rownormed = cluster_adjacency / row_sums[:, np.newaxis]

                    for n_i in range(len(path_orange) - 1):
                        source = path_orange[n_i]
                        dest = path_orange[n_i + 1]
                        if n_i < len(path_orange) - 2:
                            if do_log_flow:

                                val_edge = round(math.log1p(cluster_adjacency_rownormed[source, dest]),
                                                 2)  # * cluster_population_dict[source] #  natural logarithm (base e) of 1 + x
                            else:
                                val_edge = round(cluster_adjacency_rownormed[source, dest], 2)
                                # print("clipping val edge")
                                # if val_edge > 0.5: val_edge = 0.5
                        else:
                            if dest in via_object.terminal_clusters:
                                ts_array_original = np.asarray(via_object.terminal_clusters)
                                loc_ts_current = np.where(ts_array_original == dest)[0][0]
                                print(f'dest {dest}, is at loc {loc_ts_current} on the bp_array')
                                if do_log_flow:
                                    val_edge = round(math.log1p(via_object.cluster_bp[source, loc_ts_current]),
                                                     2)  # * cluster_population_dict[source]
                                else:
                                    val_edge = round(via_object.cluster_bp[source, loc_ts_current], 2)
                                    # print("clipping val edge")
                                    # if val_edge > 0.5: val_edge = 0.5
                            else:
                                if do_log_flow:

                                    val_edge = round(math.log1p(cluster_adjacency_rownormed[source, dest]),
                                                     2)  # * cluster_population_dict[source] #  natural logarithm (base e) of 1 + x
                                else:
                                    val_edge = round(cluster_adjacency_rownormed[source, dest], 2)
                                    # print("clipping val edge")
                                    # val_edge = 0.5
                        # sankey_edges.append((majority_cluster_population_dict[source]+'_C'+str(source), majority_cluster_population_dict[dest]+'_C'+str(dest), val_edge))#, majority_cluster_population_dict[source],majority_cluster_population_dict[dest]))
                        sankey_edges.append((source, dest,
                                             val_edge))  # ,majority_cluster_population_dict[source]+'_C'+str(source),'magenta' ))

        # print(f'pre-final sankey set of edges and vals {len(sankey_edges)}, {sankey_edges}')
        source_dest = list(set(sankey_edges))
        # print(f'final sankey set of edges and vals {len(source_dest)}, {source_dest}')
        source_dest_df = pd.DataFrame(source_dest, columns=['Source', 'Dest', 'Count'])  # ,'Label','Color'])

        nodes_in_source_dest = list(set(set(source_dest_df.Source) | set(source_dest_df.Dest)))
        nodes_in_source_dest.sort()
        convert_old_to_new = {}
        convert_new_to_old = {}
        majority_newcluster_population_dict = {}
        for ei, ii in enumerate(nodes_in_source_dest):
            convert_old_to_new[ii] = ei
            convert_new_to_old[ei] = ii
            majority_newcluster_population_dict[ei] = majority_cluster_population_dict[ii]
        source_dest_new = []
        for tuple_ in source_dest:
            source_dest_new.append((convert_old_to_new[tuple_[0]], convert_old_to_new[tuple_[1]], tuple_[2]))
        # print('new source dest after reindexing', source_dest_new)
        # nodes = [majority_cluster_population_dict[i] for i in range(len(majority_cluster_population_dict))]
        # nodes = [majority_cluster_population_dict[i] for i in nodes_in_source_dest]
        nodes = [majority_newcluster_population_dict[key] + '_C' + str(convert_new_to_old[key]) for key in
                 majority_newcluster_population_dict]
        # nodes = ['C' + str(convert_new_to_old[key]) for key in                 majority_newcluster_population_dict]
        # print('nodes', len(nodes), nodes,)
        nodes = hv.Dataset(enumerate(nodes), 'index', 'label')

        from holoviews.plotting.util import process_cmap
        print(f'{datetime.now()}\tStart sankey')
        cmap_list = process_cmap("glasbey_hv")
        p2 = hv.Sankey((source_dest_new, nodes), ['Source', "Dest"])
        p2_2 = hv.Sankey((source_dest_new, nodes), ['Source', "Dest"])
        print(f'{datetime.now()}\tmake sankey color dict')
        # Make color map
        # Extract Unique values dictionary values
        # Using set comprehension + values() + sorted()
        set_majority_truth = list(set(list(majority_newcluster_population_dict.values())))
        set_majority_truth.sort(reverse=True)

        color_dict = {}
        for index, value in enumerate(set_majority_truth):
            # assign each celltype a number
            color_dict[value] = index

        palette = plt.get_cmap(cmap_sankey, len(color_dict.keys()))
        cmap_ = palette(range(len(color_dict.keys())))
        cmap_colors_dict_sankey = {}
        for key in majority_newcluster_population_dict:
            cmap_colors_dict_sankey[int(key)] = matplotlib.colors.rgb2hex(
                cmap_[color_dict[majority_newcluster_population_dict[key]]])

        print(f'{datetime.now()}\tset options and render')

        p2.opts(
            opts.Sankey(show_values=False, edge_cmap=cmap_colors_dict_sankey, edge_color=dim('Source').str(),
                        node_color=dim('Source').str(),
                        edge_line_width=2, width=1800, height=1200, cmap=cmap_colors_dict_sankey, node_padding=15,
                        fontsize={'labels': 1}, title=title_str))
        show(hv.render(p2))

        p2_2.opts(
            opts.Sankey(show_values=False, edge_cmap=cmap_colors_dict_sankey, edge_color=dim('Source').str(),
                        node_color=dim('Source').str(),
                        edge_line_width=2, width=1800, height=1200, node_padding=20, cmap=cmap_colors_dict_sankey,
                        title=title_str))
        # show(hv.render(p2_2))

        p2_2.opts(
            opts.Sankey(labels='label', edge_cmap=cmap_colors_dict_sankey, edge_color=dim('Source').str(),
                        node_color=dim('Source').str(),
                        edge_line_width=2, width=1800, height=1200, node_padding=20, cmap=cmap_colors_dict_sankey,
                        title=title_str))
        show(hv.render(p2_2))

        '''
        p0 = hv.Sankey(source_dest_df)
        show(hv.render(p0))

        p = hv.Sankey(source_dest_df, kdims=["Source", "Dest"], vdims=["Count"])
        p.opts(
            opts.Sankey(edge_color=dim('Source').str(), node_color=dim('Source').str(),
                        edge_line_width=2,
                        edge_cmap='tab20', node_cmap='tab20', width=1800, height=1800, title='test title', node_padding=3))
        show(hv.render(p))
        '''
        # https://stackoverflow.com/questions/57085026/how-do-i-colour-the-individual-categories-in-a-holoviews-sankey-diagram
        # https://stackoverflow.com/questions/76505156/draw-sankey-diagram-with-holoviews-and-bokeh
        # https://holoviews.org/reference/elements/bokeh/Sankey.html
        # https://malouche.github.io/notebooks/Sankey_graphs.html
        # https://github.com/holoviz/holoviews/issues/3501

    return


def plot_sc_lineage_probability(via_object, embedding: ndarray = None, idx: list = None, cmap_name='plasma', dpi=150,
                                scatter_size=None, marker_lineages=[], fontsize: int = 8, alpha_factor: float = 0.9,
                                majority_cluster_population_dict: dict = None, cmap_sankey='rainbow',
                                do_sankey: bool = False):
    '''

    G is the igraph knn (low K) used for shortest path in high dim space. no idx needed as it's made on full sample
    knn_hnsw is the knn made in the embedded space used for query to find the nearest point in the downsampled embedding
    that corresponds to the single cells in the full graph

    :param via_object:
    :param embedding: n_samples x 2. embedding is either the full or downsampled 2D representation of the full dataset.
    :param idx: if one uses a downsampled embedding of the original data, then idx is the selected indices of the downsampled samples used in the visualization
    :param cmap_name:
    :param dpi:
    :param scatter_size: if None, then auto determined based on n_cells
    :param marker_lineages: Default is to use all lineage pathways. other provide a list of lineage number (terminal cluster number).
    :param alpha_factor: float transparency
    :return: fig, axs
    '''
    import hnswlib

    if len(marker_lineages) == 0:
        marker_lineages = via_object.terminal_clusters

    else:
        marker_lineages = [i for i in marker_lineages if i in via_object.terminal_clusters]
    print(f'{datetime.now()}\tMarker_lineages: {marker_lineages}')
    if embedding is None:
        if via_object.embedding is None:
            print(
                f'{datetime.now()}\tERROR: please provide a single cell embedding or run re-via with do_compute_embedding==True using either embedding_type = via-umap OR via-mds')
            return
        else:
            print(f'{datetime.now()}\tAutomatically setting embedding to via_object.embedding')
            embedding = via_object.embedding

    if idx is None: idx = np.arange(0, via_object.nsamples)
    # G = via_object.full_graph_shortpath
    n_original_comp, n_original_comp_labels = connected_components(via_object.csr_full_graph, directed=False)
    G = via_object.full_graph_paths(via_object.data, n_original_comp)
    knn_hnsw = _make_knn_embeddedspace(embedding)
    y_root = []
    x_root = []
    root1_list = []
    p1_sc_bp = np.nan_to_num(via_object.single_cell_bp[idx, :], nan=0.0, posinf=0.0, neginf=0.0)
    # row normalize
    row_sums = p1_sc_bp.sum(axis=1)
    p1_sc_bp = p1_sc_bp / row_sums[:,
                          np.newaxis]  # make rowsums a column vector where i'th entry is sum of i'th row in p1-sc-bp
    print(f'{datetime.now()}\tCheck sc pb {p1_sc_bp[0, :].sum()} ')

    p1_labels = np.asarray(via_object.labels)[idx]

    p1_cc = via_object.connected_comp_labels
    p1_sc_pt_markov = list(np.asarray(via_object.single_cell_pt_markov)[idx])
    X_data = via_object.data

    X_ds = X_data[idx, :]
    p_ds = hnswlib.Index(space='l2', dim=X_ds.shape[1])
    p_ds.init_index(max_elements=X_ds.shape[0], ef_construction=200, M=16)
    p_ds.add_items(X_ds)
    p_ds.set_ef(max(1, min(50, X_ds.shape[0])))
    num_cluster = len(set(via_object.labels))
    G_orange = ig.Graph(n=num_cluster, edges=via_object.edgelist_maxout,
                        edge_attrs={'weight': via_object.edgeweights_maxout})
    for ii, r_i in enumerate(via_object.root):
        loc_i = np.where(p1_labels == via_object.root[ii])[0]
        x = [embedding[xi, 0] for xi in loc_i]
        y = [embedding[yi, 1] for yi in loc_i]

        labels_root, distances_root = knn_hnsw.knn_query(np.array([np.mean(x), np.mean(y)]), k=1)
        x_root.append(embedding[labels_root, 0][0])
        y_root.append(embedding[labels_root, 1][0])

        labelsroot1, distances1 = via_object.knn_struct.knn_query(X_ds[labels_root[0][0], :], k=1)
        root1_list.append(labelsroot1[0][0])
        sankey_edges = []
        print('f getting majority comp')

        '''
        #VERY SLOW maybe try with dataframe mode  df.groupby(['team'])['points'].agg(pd.Series.mode)
        for labels_i in via_object.labels:
            loc_labels = np.where(np.asarray(via_object.labels) == labels_i)[0]
            majority_composition = func_mode(list(np.asarray(via_object.true_label)[loc_labels]))
            majority_cluster_population_dict[labels_i] = majority_composition
        print('f End getting majority comp')
        '''
        for fst_i in marker_lineages:#via_object.terminal_clusters:
            path_orange = G_orange.get_shortest_paths(via_object.root[ii], to=fst_i)[0]
            if len(path_orange) > 0:
                print(
                    f'{datetime.now()}\tCluster path on clustergraph starting from Root Cluster {via_object.root[ii]} to Terminal Cluster {fst_i}: {path_orange}')

                '''
                cluster_population_dict = {}
                for group_i in set(via_object.labels):
                    loc_i = np.where(via_object.labels == group_i)[0]
                    cluster_population_dict[group_i] = len(loc_i)
                '''
    # single-cell branch probability evolution probability
    n_terminal_clusters = len(marker_lineages)
    fig_ncols = min(3, n_terminal_clusters)
    fig_nrows, mod = divmod(n_terminal_clusters, fig_ncols)
    if mod == 0:
        if fig_nrows == 0:
            fig_nrows += 1
        else:
            fig_nrows = fig_nrows
    if mod != 0:        fig_nrows += 1

    fig, axs = plt.subplots(fig_nrows, fig_ncols, dpi=dpi)

    ts_array_original = np.asarray(via_object.terminal_clusters)

    ti = 0  # counter for terminal cluster
    for r in range(fig_nrows):
        for c in range(fig_ncols):
            if ti < n_terminal_clusters:
                ts_current = marker_lineages[ti]
                loc_ts_current = np.where(ts_array_original == ts_current)[0][0]
                loc_labels = np.where(np.asarray(via_object.labels) == ts_current)[0]
                majority_composition = func_mode(list(np.asarray(via_object.true_label)[loc_labels]))

                if fig_nrows == 1:
                    if fig_ncols == 1:
                        plot_sc_pb(axs, fig, embedding, p1_sc_bp[:, loc_ts_current],
                                   ti=str(ts_current) + '-' + str(majority_composition), cmap_name=cmap_name,
                                   scatter_size=scatter_size, fontsize=fontsize)
                    else:
                        plot_sc_pb(axs[c], fig, embedding, p1_sc_bp[:, loc_ts_current],
                                   ti=str(ts_current) + '-' + str(majority_composition), cmap_name=cmap_name,
                                   scatter_size=scatter_size, fontsize=fontsize, alpha_factor=alpha_factor)

                else:
                    plot_sc_pb(axs[r, c], fig, embedding, p1_sc_bp[:, loc_ts_current],
                               ti=str(ts_current) + '-' + str(majority_composition), cmap_name=cmap_name,
                               scatter_size=scatter_size, fontsize=fontsize, alpha_factor=alpha_factor)

                loc_i = np.where(p1_labels == ts_current)[0]
                val_pt = [p1_sc_pt_markov[i] for i in loc_i]
                th_pt = np.percentile(val_pt, 50)  # 50
                loc_i = [loc_i[i] for i in range(len(val_pt)) if val_pt[i] >= th_pt]
                x = [embedding[xi, 0] for xi in
                     loc_i]  # location of sc nearest to average location of terminal clus in the EMBEDDED space
                y = [embedding[yi, 1] for yi in loc_i]
                labels, distances = knn_hnsw.knn_query(np.array([np.mean(x), np.mean(y)]),
                                                       k=1)  # knn_hnsw is knn of embedded space
                x_sc = embedding[labels[0], 0]  # terminal sc location in the embedded space
                y_sc = embedding[labels[0], 1]

                labelsq1, distances1 = via_object.knn_struct.knn_query(X_ds[labels[0][0], :],
                                                                       k=1)  # find the nearest neighbor in the PCA-space full graph

                path = G.get_shortest_paths(root1_list[p1_cc[loc_ts_current]], to=labelsq1[0][0])  # weights='weight')
                # G is the knn of all sc points

                path_idx = []  # find the single-cell which is nearest to the average-location of a terminal cluster
                # get the nearest-neighbor in this downsampled PCA-space graph. These will make the new path-way points
                path = path[0]

                # clusters of path
                cluster_path = []
                for cell_ in path:
                    cluster_path.append(via_object.labels[cell_])

                revised_cluster_path = []
                revised_sc_path = []
                for enum_i, clus in enumerate(cluster_path):
                    num_instances_clus = cluster_path.count(clus)
                    if (clus == cluster_path[0]) | (clus == cluster_path[-1]):
                        revised_cluster_path.append(clus)
                        revised_sc_path.append(path[enum_i])
                    else:
                        '''
                        if num_instances_clus > 1:  # typically intermediate stages spend a few transitions at the sc level within a cluster
                            if clus not in revised_cluster_path: revised_cluster_path.append(clus)  # cluster
                            revised_sc_path.append(path[enum_i])  # index of single cell
                        '''
                        if  num_instances_clus >= 1:  # typically intermediate stages spend a few transitions at the sc level within a cluster
                            if clus not in revised_cluster_path: revised_cluster_path.append(clus)  # cluster
                            revised_sc_path.append(path[enum_i])  # index of single cell

                print(
                    f"{datetime.now()}\tRevised Cluster level path on sc-knnGraph from Root Cluster {via_object.root[p1_cc[ti - 1]]} to Terminal Cluster {ts_current} along path: {revised_cluster_path}")
                ti += 1
            fig.patch.set_visible(False)
            if fig_nrows == 1:
                if fig_ncols == 1:
                    axs.axis('off')
                    axs.grid(False)
                else:
                    axs[c].axis('off')
                    axs[c].grid(False)
            else:
                axs[r, c].axis('off')
                axs[r, c].grid(False)
    return fig, axs


def plot_viagraph(via_object, type_data='gene', df_genes=None, gene_list:list = [],arrow_head:float=0.1, n_col:int = None, n_row:int = None,
                  edgeweight_scale:float=1.5, cmap=None, label_text:bool=True, size_factor_node: float = 1, tune_edges:bool = False,initial_bandwidth=0.05, decay=0.9, edgebundle_pruning=0.5):
    '''
    cluster level expression of gene/feature intensity
    :param via_object:
    :param type_data:
    :param gene_exp: pd.Dataframe size n_cells x genes. Otherwise defaults to plotting pseudotime
    :param gene_list: list of gene names corresponding to the column name
    :param arrow_head:
    :param edgeweight_scale:
    :param cmap:
    :param label_text: bool to add numeric values of the gene exp level
    :param size_factor_node size of graph nodes
    :param tune_edges: bool (false). if you want to change the number of edges visualized, then set this to True and modify the tuning parameters (initial_bandwidth, decay, edgebundle_pruning)
    :param initial_bandwidth: (float = 0.05)  increasing bw increases merging of minor edges.  Only used when tune_edges = True
    :param decay: (decay = 0.9) increasing decay increases merging of minor edges . Only used when tune_edges = True
    :param edgebundle_pruning (float = 0.5). takes on values between 0-1. smaller value means more pruning away edges that can be visualised. Only used when tune_edges = True
    :param n_col: Number of columns to plot (if None, compute n_col if n_row is given, else 4)
    :param n_row: Number of rows to plot (if None, compute n_row if n_col is given)
    :return: fig, axs
    '''
    '''
    #draws the clustergraph for cluster level gene or pseudotime values
    # type_pt can be 'pt' pseudotime or 'gene' for gene expression
    # ax1 is the pseudotime graph
    '''

    n_genes = len(gene_list)
    pt = via_object.markov_hitting_times
    if n_genes == 0:
        gene_list=['pseudotime']
        df_genes = pd.DataFrame()
        df_genes['pseudotime'] = via_object.single_cell_pt_markov
        n_genes = 1
    if tune_edges:
        hammer_bundle, layout = make_edgebundle_viagraph(via_object = via_object, layout=via_object.layout, decay=decay,initial_bandwidth=initial_bandwidth, edgebundle_pruning=edgebundle_pruning) #hold the layout fixed. only change the edges

    else:
        hammer_bundle = via_object.hammerbundle_cluster
        layout = via_object.layout#graph_node_pos
    if n_col is None and n_row is None :
        n_col = 4
        n_row = int(np.ceil(n_genes/n_col))
    elif n_col is None:
        n_col = int(np.ceil(n_genes/n_row))
    elif n_row is None:
        n_row = int(np.ceil(n_genes/n_col))
    
    if n_col*n_row < n_genes:
        raise ValueError('n_col and n_row does not match number of genes in gene_list')

    fig, axes = plt.subplots(n_row, n_col)
    axs = axes.flatten()

    if cmap is None: cmap = 'coolwarm' if type_data == 'gene' else 'viridis_r'

    node_pos = layout.coords# via_object.graph_node_pos

    node_pos = np.asarray(node_pos)

    df_genes['cluster'] = via_object.labels
    df_genes = df_genes.groupby('cluster', as_index=False).mean()

    n_groups = len(set(via_object.labels))  # node_pos.shape[0]

    group_pop = np.zeros([n_groups, 1])
    via_object.cluster_population_dict = {}
    for group_i in set(via_object.labels):
        loc_i = np.where(via_object.labels == group_i)[0]

        group_pop[group_i] = len(loc_i)  # np.sum(loc_i) / 1000 + 1
        via_object.cluster_population_dict[group_i] = len(loc_i)

    for i in range(n_genes):
        if n_genes ==1:
            ax_i = axs
        else: ax_i = axs[i]
        gene_i = gene_list[i]

        c_edge, l_width = [], []
        for ei, pti in enumerate(pt):
            if ei in via_object.terminal_clusters:
                c_edge.append('red')
                l_width.append(1.5)
            else:
                c_edge.append('gray')
                l_width.append(0.0)
        ax_i = plot_viagraph_(ax_i, hammer_bundle=hammer_bundle, layout=layout,
                              CSM=via_object.CSM,
                              velocity_weight=via_object.velo_weight, pt=pt, headwidth_bundle=arrow_head,
                              alpha_bundle=0.4, linewidth_bundle=edgeweight_scale)
        group_pop_scale = .5 * group_pop * 1000 / max(group_pop)
        pos = ax_i.scatter(node_pos[:, 0], node_pos[:, 1], s=group_pop_scale * size_factor_node,
                           c=df_genes[gene_i].values, cmap=cmap,
                           edgecolors=c_edge, alpha=1, zorder=3, linewidth=l_width)
        if label_text == True:
            for ii in range(node_pos.shape[0]):
                ax_i.text(node_pos[ii, 0] + 0.1, node_pos[ii, 1] + 0.1,
                          'C' + str(ii) + ' ' + str(round(df_genes[gene_i].values[ii], 1)),
                          color='black', zorder=4, fontsize=6)
        divider = make_axes_locatable(ax_i)
        cax = divider.append_axes('right', size='10%', pad=0.05)

        cbar = fig.colorbar(pos, cax=cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=8)
        ax_i.set_title(gene_i)
        ax_i.grid(False)
        ax_i.set_xticks([])
        ax_i.set_yticks([])
        ax_i.axis('off')
    fig.patch.set_visible(False)
    fig.set_size_inches(5*n_col,5*n_row)
    fig.tight_layout()
    for ax in axs[n_genes:]:
        fig.delaxes(ax)
    return fig, axes


def plot_atlas_view(hammerbundle_dict=None, via_object=None, alpha_bundle_factor=1, linewidth_bundle=2,
                    facecolor: str = 'white', cmap: str = 'plasma', extra_title_text='', alpha_milestones: float = 0.3,
                    headwidth_bundle: float = 0.1, headwidth_alpha: float = 0.8, arrow_frequency: float = 0.05,
                    show_arrow: bool = True, sc_labels_sequential: list = None, sc_labels_expression: list = None,
                    initial_bandwidth=0.03, decay=0.7, n_milestones: int = None, scale_scatter_size_pop: bool = False,
                    show_milestones: bool = True, sc_labels: list = None, text_labels: bool = False,
                    lineage_pathway: list = [], dpi: int = 300, fontsize_title: int = 6, fontsize_labels: int = 6,
                    global_visual_pruning=0.5, use_sc_labels_sequential_for_direction: bool = False, sc_scatter_size=3,
                    sc_scatter_alpha: float = 0.4, add_sc_embedding: bool = True, size_milestones: int = 5,
                    colorbar_legend='pseudotime',scale_arrow_headwidth:bool = False):
    '''

    Edges can be colored by time-series numeric labels, pseudotime, lineage pathway probabilities,  or gene expression. If not specificed then time-series is chosen if available, otherwise falls back to pseudotime. to use gene expression the sc_labels_expression is provided as a list.
    To specify other numeric sequential data provide a list of sc_labels_sequential = [] n_samples in length. via_object.embedding must be an ndarray of shape (nsamples,2)

    :param hammer_bundle_dict: dictionary with keys: hammerbundle object with coordinates of all the edges to draw. If hammer_bundle and layout are None, then this will be computed internally
    :param via_object: type via object, if hammerbundle_dict is None, then you must provide a via_object. Ensure that via_object has embedding attribute
    :param layout: coords of cluster nodes and optionally also contains the numeric value associated with each cluster (such as time-stamp) layout[['x','y','numeric label']] sc/cluster/milestone level
    :param CSM: cosine similarity matrix. cosine similarity between the RNA velocity between neighbors and the change in gene expression between these neighbors. Only used when available
    :param velocity_weight: percentage weightage given to the RNA velocity based transition matrix
    :param pt: cluster-level pseudotime
    :param alpha_bundle: alpha when drawing lines
    :param linewidth_bundle: linewidth of bundled lines
    :param edge_color:
    :param alpha_milestones: float 0.3 alpha of milestones
    :param size_milestones: scatter size of the milestones (use sc_size_scatter to control single cell scatter when using in conjunction with lineage probs/ sc embeddings)
    :param arrow_frequency: min dist between arrows (bundled edges otherwise have overcrowding of arrows)
    :param show_direction: True will draw arrows along the lines to indicate direction
    :param milestone_edges: pandas DataFrame milestoone_edges[['source','target']]
    :param milestone_numeric_values: the milestone average of numeric values such as time (days, hours), location (position), or other numeric value used for coloring edges in a sequential manner
            if this is None then the edges are colored by length to distinguish short and long range edges
    :param arrow_frequency: 0.05. higher means fewer arrows
    :param n_milestones: int  None. if no hammerbundle_dict is provided, but via_object is provided, then the user can specify level of granularity by setting the n_milestones. otherwise it will be automatically selected
    :param scale_scatter_size_pop: bool default False
    :param sc_labels_expression: list single cell numeric values used for coloring edges and nodes of corresponding milestones mean expression levels (len n_single_cell samples)
            edges can be colored by time-series numeric (gene expression)/string (cell type) labels, pseudotime, or gene expression. If not specificed then time-series is chosen if available, otherwise falls back to pseudotime. to use gene expression the sc_labels_expression is provided as a list
    :param sc_labels_sequential: list single cell numeric sequential values used for directionality inference as replacement for  pseudotime or via_object.time_series_labels (len n_samples single cell)
    :param sc_labels: list None list of single-cell level labels (categorial or discrete set of numerical values) to label the nodes
    :param text_labels: bool False if you want to label the nodes based on sc_labels (or true_label if via_object is provided)
    :param lineage_pathway: list of terminal states to plot lineage pathways
    :param use_sc_labels_sequential_for_direction: use the sequential data (timeseries labels or other provided by user) to direct the arrows
    :param lineage_alpha_threshold number representing the percentile (0-100) of lineage likelikhood in a particular lineage pathway, below which edges will be drawn with lower alpha transparency factor
    :param sc_scatter_alpha: transparency of the background singlecell scatter when plotting lineages
    :param add_sc_embedding: add background of single cell scatter plot for Atlas
    :param scatter_size_sc_embedding
    :param colorbar_legend str title of colorbar
    :return: fig, axis with bundled edges plotted
    '''

    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        import matplotlib.colors as colors
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    sc_scatter_alpha = 1 - sc_scatter_alpha
    cmap_name = cmap
    headwidth_alpha_og = headwidth_alpha
    linewidth_bundle_og = linewidth_bundle
    alpha_bundle_factor_og = alpha_bundle_factor
    if hammerbundle_dict is None:
        if via_object is None:
            print('if hammerbundle_dict is not provided, then you must provide via_object')
        else:
            hammerbundle_dict = via_object.hammerbundle_milestone_dict
            if hammerbundle_dict is None:
                if n_milestones is None: n_milestones = min(via_object.nsamples, 150)
                if sc_labels_sequential is None:
                    if via_object.time_series_labels is not None:
                        sc_labels_sequential = via_object.time_series_labels
                    else:
                        sc_labels_sequential = via_object.single_cell_pt_markov
                print(f'{datetime.now()}\tComputing Edges')
                hammerbundle_dict = make_edgebundle_milestone(via_object=via_object,
                                                              embedding=via_object.embedding,
                                                              sc_graph=via_object.ig_full_graph,
                                                              n_milestones=n_milestones,
                                                              sc_labels_numeric=sc_labels_sequential,
                                                              initial_bandwidth=initial_bandwidth, decay=decay,
                                                              weighted=True,
                                                              global_visual_pruning=global_visual_pruning)
                via_object.hammerbundle_dict = hammerbundle_dict
            hammer_bundle = hammerbundle_dict['hammerbundle']
            layout = hammerbundle_dict['milestone_embedding'][['x', 'y']].values
            milestone_edges = hammerbundle_dict['edges']
            if sc_labels_expression is None:
                milestone_numeric_values = hammerbundle_dict['milestone_embedding']['numeric label']
            else:
                if (isinstance(sc_labels_expression[0], str)) == True:
                    color_dict = {}
                    set_labels = list(set(sc_labels_expression))
                    set_labels.sort(reverse=True)
                    for index, value in enumerate(set_labels):
                        color_dict[value] = index
                    milestone_numeric_values = [color_dict[i] for i in sc_labels_expression]
                    sc_labels_expression = milestone_numeric_values
                else:
                    milestone_numeric_values = sc_labels_expression
            milestone_pt = hammerbundle_dict['milestone_embedding']['pt']
            if use_sc_labels_sequential_for_direction: milestone_pt = hammerbundle_dict['milestone_embedding'][
                'numeric label']
            if sc_labels_expression is not None:  # if both sclabelexpression and sequential are provided, then sc_labels_expression takes precedence
                df = pd.DataFrame()
                df['sc_milestone_labels'] = hammerbundle_dict['sc_milestone_labels']
                df['sc_expression'] = sc_labels_expression
                df = df.groupby('sc_milestone_labels').mean()

                milestone_numeric_values = df[
                    'sc_expression'].values  # used to color edges. direction is based on milestone_pt

    else:
        hammer_bundle = hammerbundle_dict['hammerbundle']
        layout = hammerbundle_dict['milestone_embedding'][['x', 'y']].values
        milestone_edges = hammerbundle_dict['edges']
        milestone_numeric_values = hammerbundle_dict['milestone_embedding']['numeric label']

        if sc_labels_expression is not None:  # if both sclabelexpression and sequential are provided, then sc_labels_expression takes precedence
            df = pd.DataFrame()
            df['sc_milestone_labels'] = hammerbundle_dict['sc_milestone_labels']
            df['sc_expression'] = sc_labels_expression
            df = df.groupby('sc_milestone_labels').mean()
            milestone_numeric_values = df['sc_expression'].values  # used to color edges

        milestone_pt = hammerbundle_dict['milestone_embedding']['pt']
        if use_sc_labels_sequential_for_direction: milestone_pt = hammerbundle_dict['milestone_embedding'][
            'numeric label']
    if len(lineage_pathway) == 0:
        # fig, ax = plt.subplots(facecolor=facecolor)
        fig_nrows, fig_ncols = 1, 1
    else:
        lineage_pathway_temp = [i for i in lineage_pathway if
                                i in via_object.terminal_clusters]  # checking the clusters are actually in terminal_clusters
        lineage_pathway = lineage_pathway_temp
        n_terminal_clusters = len(lineage_pathway)
        fig_ncols = min(3, n_terminal_clusters)
        fig_nrows, mod = divmod(n_terminal_clusters, fig_ncols)
        if mod == 0:
            if fig_nrows == 0:
                fig_nrows += 1
            else:
                fig_nrows = fig_nrows
        if mod != 0: fig_nrows += 1
    fig, ax = plt.subplots(fig_nrows, fig_ncols, dpi=dpi, facecolor=facecolor)
    counter_ = 0
    n_real_subplots = max(len(lineage_pathway), 1)
    majority_composition = ''
    for r in range(fig_nrows):
        for c in range(fig_ncols):
            if (counter_ < n_real_subplots):
                if len(lineage_pathway) > 0:
                    milestone_numeric_values = hammerbundle_dict['milestone_embedding'][
                        'sc_lineage_probability_' + str(lineage_pathway[counter_])]
                    p1_sc_bp = np.nan_to_num(via_object.single_cell_bp, nan=0.0, posinf=0.0,
                                             neginf=0.0)  # single cell lineage probabilities sc pb
                    # row normalize
                    row_sums = p1_sc_bp.sum(axis=1)
                    p1_sc_bp = p1_sc_bp / row_sums[:,
                                          np.newaxis]  # make rowsums a column vector where i'th entry is sum of i'th row in p1-sc-bp
                    ts_cluster_number = lineage_pathway[counter_]
                    ts_array_original = np.asarray(via_object.terminal_clusters)
                    loc_ts_current = np.where(ts_array_original == ts_cluster_number)[0][0]
                    print(
                        f'location of {lineage_pathway[counter_]} is at {np.where(ts_array_original == ts_cluster_number)[0]} and {loc_ts_current}')
                    p1_sc_bp = p1_sc_bp[:, loc_ts_current]

                    # print(f'{datetime.now()}\tCheck sc pb {p1_sc_bp[0, :].sum()} ')
                    if via_object is not None:
                        ts_current = lineage_pathway[counter_]
                        loc_labels = np.where(np.asarray(via_object.labels) == ts_current)[0]
                        majority_composition = func_mode(list(np.asarray(via_object.true_label)[loc_labels]))
                x_ = [l[0] for l in layout]
                y_ = [l[1] for l in layout]
                # min_x, max_x = min(x_), max(x_)
                # min_y, max_y = min(y_), max(y_)
                delta_x = max(x_) - min(x_)
                delta_y = max(y_) - min(y_)

                layout = np.asarray(layout)
                # get each segment. these are separated by nans.
                hbnp = hammer_bundle.to_numpy()
                splits = (np.isnan(hbnp[:, 0])).nonzero()[0]  # location of each nan values
                edgelist_segments = []
                start = 0
                segments = []
                arrow_coords = []
                seg_len = []  # length of a segment
                for stop in splits:
                    seg = hbnp[start:stop, :]
                    segments.append(seg)
                    seg_len.append(seg.shape[0])
                    start = stop

                min_seg_length = min(seg_len)
                max_seg_length = max(seg_len)
                seg_len = np.asarray(seg_len)
                seg_len = np.clip(seg_len, a_min=np.percentile(seg_len, 10),
                                  a_max=np.percentile(seg_len, 90))
                # mean_seg_length = sum(seg_len)/len(seg_len)

                step = 1  # every step'th segment is plotted

                cmap = plt.get_cmap(cmap)

                if milestone_numeric_values is not None:
                    max_numerical_value = max(milestone_numeric_values)
                    min_numerical_value = min(milestone_numeric_values)
                ##inserting edits here

                from matplotlib.patches import Rectangle
                sc_embedding = via_object.embedding
                max_r = np.max(via_object.embedding[:, 0]) + 1
                max_l = np.min(via_object.embedding[:, 0]) - 1

                max_up = np.max(via_object.embedding[:, 1]) + 1
                max_dw = np.min(via_object.embedding[:, 1]) - 1
                if add_sc_embedding:

                    if len(lineage_pathway) == 0:
                        print('inside add sc embedding second if')
                        if sc_labels_expression is not None:
                            gene_expression = False
                            if gene_expression:
                                val_alph = [i if i > 0.3 else 0 for i in sc_labels_expression]
                                max_alph = max(val_alph)
                                val_alph = [i / max_alph for i in val_alph]
                                ax.scatter(via_object.embedding[:, 0], via_object.embedding[:, 1], alpha=val_alph,
                                           c=sc_labels_expression, s=sc_scatter_size, cmap=cmap_name,
                                           zorder=2)  # alpha=1 change back to
                            # ax.scatter(via_object.embedding[:, 0], via_object.embedding[:, 1], alpha=0.1,                                       c='lightgray', s=5)
                            # new_cmap= truncate_colormap(cmap, 0.25, 1.0) #use this for gene expression plotting in zebrahub non-neuro ecto

                            else:
                                ax.scatter(via_object.embedding[:, 0], via_object.embedding[:, 1], alpha=1,
                                           c=sc_labels_expression, s=sc_scatter_size, cmap=cmap,
                                           zorder=1)  # alpha=1 change back to
                            ax.add_patch(Rectangle((max_l, max_dw), max_r - max_l, max_up - max_dw, facecolor=facecolor, #'white'
                                                   alpha=sc_scatter_alpha))

                if len(lineage_pathway) > 0:
                    if fig_nrows == 1:
                        if fig_ncols == 1:
                            plot_sc_pb(ax, fig, embedding=via_object.embedding, prob=p1_sc_bp,
                                       ti=str(ts_current) + '-' + str(majority_composition), cmap_name=cmap_name,
                                       scatter_size=sc_scatter_size, fontsize=4, alpha_factor=1, show_legend=False)
                            ax.add_patch(
                                Rectangle((max_l, max_dw), max_r - max_l, max_up - max_dw, facecolor=facecolor,#"white",
                                          alpha=sc_scatter_alpha))
                            # ax.scatter(via_object.embedding[:, 0], via_object.embedding[:, 1], alpha=0.05, c='white',                                             s=5)

                        else:

                            plot_sc_pb(ax[c], fig, embedding=sc_embedding, prob=p1_sc_bp,
                                       ti=str(ts_current) + '-' + str(majority_composition), cmap_name=cmap_name,
                                       scatter_size=sc_scatter_size, fontsize=4, alpha_factor=1, show_legend=False)
                            ax[c].add_patch(
                                Rectangle((max_l, max_dw), max_r - max_l, max_up - max_dw, facecolor=facecolor,#"white",
                                          alpha=sc_scatter_alpha))
                            # ax[c].scatter(via_object.embedding[:, 0], via_object.embedding[:, 1], alpha=0.1, c='white', s=4)

                    else:

                        plot_sc_pb(ax[r, c], fig, embedding=sc_embedding, prob=p1_sc_bp,
                                   ti=str(ts_current) + '-' + str(majority_composition), cmap_name=cmap_name,
                                   scatter_size=sc_scatter_size, fontsize=4, alpha_factor=1, show_legend=False)

                        ax[r, c].add_patch(Rectangle((max_l, max_dw), max_r - max_l, max_up - max_dw, facecolor=facecolor,#"white",
                                                     alpha=sc_scatter_alpha))  # 0.7
                        # ax[r, c].scatter(layout[:, 0], layout[:, 1], s=40,                                         c='white', cmap=cmap_name,                                         alpha=0.5, edgecolors='none')  # vmax=1)
                        # ax[r, c].scatter(via_object.embedding[:, 0], via_object.embedding[:, 1], alpha=0.1, c='white', s=4,edgecolors='none')

                # end white edits
                seg_count = 0

                for seg in segments[::step]:
                    do_arrow = True

                    # seg_weight = max(0.3, math.log(1+seg[-1,2])) seg[-1,2] column index 2 has the weight information

                    seg_weight = seg[-1, 2] * seg_len[seg_count] / (
                                max_seg_length - min_seg_length)  ##seg.shape[0] / (max_seg_length - min_seg_length)#seg.shape[0]

                    # cant' quite decide yet if sigmoid is desirable
                    # seg_weight=sigmoid_scalar(seg.shape[0] / (max_seg_length - min_seg_length), scale=5, shift=mean_seg_length / (max_seg_length - min_seg_length))
                    alpha_bundle = max(seg_weight * alpha_bundle_factor, 0.1)  # max(0.1, math.log(1 + seg[-1, 2]))

                    if alpha_bundle > 1: alpha_bundle = 1

                    source_milestone = milestone_edges['source'].values[seg_count]
                    target_milestone = milestone_edges['target'].values[seg_count]

                    direction = milestone_pt[target_milestone] - milestone_pt[source_milestone]
                    if direction < 0:
                        direction = -1
                    else:
                        direction = 1
                    source_milestone_numerical_value = milestone_numeric_values[source_milestone]

                    target_milestone_numerical_value = milestone_numeric_values[target_milestone]
                    # print('source milestone', source_milestone_numerical_value)
                    # print('target milestone', target_milestone_numerical_value)
                    min_source_target_numerical_value = min(source_milestone_numerical_value,
                                                            target_milestone_numerical_value)  # ORIGINALLY USING MIN()
                    # min_source_target_numerical_value =(source_milestone_numerical_value+       target_milestone_numerical_value)/2
                    max_source_target_numerical_value = max(source_milestone_numerical_value,
                                                            target_milestone_numerical_value)
                    # consider using the max value for lineage pathways to better highlight the high probabilties near the cell fate
                    if len(lineage_pathway) > 0:  # print('change remove this back to >0 in plotting_via.py and gray segment zorder =2')

                        # rgba = cmap((min_source_target_numerical_value - min_numerical_value) / (max_numerical_value - min_numerical_value))

                        if min_source_target_numerical_value <= 0.3 * np.max(
                                milestone_numeric_values):  # 0.1:#np.percentile(milestone_numeric_values,lineage_alpha_threshold):
                            alpha_bundle = 0.01  # 0.1#0.01
                            headwidth_alpha = 0.01  # 0.2
                            linewidth_bundle = 0.1 * linewidth_bundle_og
                        elif ((min_source_target_numerical_value > 0.3 * np.max(milestone_numeric_values)) & (
                                min_source_target_numerical_value < 0.7 * np.max(
                                milestone_numeric_values))):  # 0.1:#np.percentile(milestone_numeric_values,lineage_alpha_threshold):

                            alpha_bundle = 0.05  # 0.2#max(min_source_target_numerical_value/np.max(milestone_numeric_values) *alpha_bundle,0.01)
                            headwidth_alpha = 0.01  # 0.2
                            linewidth_bundle = min_source_target_numerical_value / np.max(
                                milestone_numeric_values) * linewidth_bundle_og
                        else:

                            headwidth_alpha = headwidth_alpha_og
                            linewidth_bundle = linewidth_bundle_og * 1.4
                    rgba = cmap((min_source_target_numerical_value - min_numerical_value) / (
                                max_numerical_value - min_numerical_value))
                    # rgba = new_cmap((min_source_target_numerical_value - min_numerical_value) / (                                max_numerical_value - min_numerical_value)) #use for non-neuro-ecto zebrahub gene expression

                    # else: rgba = cmap(min(seg_weight,0.95))#cmap(seg.shape[0]/(max_seg_length-min_seg_length))
                    # if seg_weight>0.05: seg_weight=0.1
                    # if seg_count%10000==0: print('seg weight', seg_weight)
                    seg = seg[:, 0:2].reshape(-1, 2)
                    seg_p = seg[~np.isnan(seg)].reshape((-1, 2))

                    if fig_nrows == 1:

                        if fig_ncols == 1:

                            # ax.plot(seg_p[:, 0], seg_p[:, 1], linewidth=0.2,                                    alpha=0.1, color='gray')
                            ax.plot(seg_p[:, 0], seg_p[:, 1], linewidth=linewidth_bundle * seg_weight,
                                    alpha=alpha_bundle, color=rgba)  # , zorder=2)#edge_color )
                        else:
                            ax[c].plot(seg_p[:, 0], seg_p[:, 1], linewidth=linewidth_bundle * seg_weight,
                                       alpha=alpha_bundle, color=rgba)  # edge_color )
                    else:

                        ax[r, c].plot(seg_p[:, 0], seg_p[:, 1], linewidth=linewidth_bundle * seg_weight,
                                      alpha=alpha_bundle, color=rgba)  # edge_color )

                    if (show_arrow) & (seg_p.shape[0] > 3):
                        mid_point = math.floor(seg_p.shape[0] / 2)

                        if len(arrow_coords) > 0:  # dont draw arrows in overlapping segments
                            for v1 in arrow_coords:
                                dist_ = dist_points(v1, v2=[seg_p[mid_point, 0], seg_p[mid_point, 1]])

                                if dist_ < arrow_frequency * delta_x: do_arrow = False
                                if dist_ < arrow_frequency * delta_y: do_arrow = False

                        if (do_arrow == True) & (seg_p.shape[0] > 3):
                            if scale_arrow_headwidth:
                                headwidth_bundle_ = headwidth_bundle*min(seg_weight,1)
                                print('doing scale arrow headwidth:', headwidth_bundle_)
                            else: headwidth_bundle_ = headwidth_bundle
                            if fig_nrows == 1:
                                if fig_ncols == 1:
                                    ax.arrow(seg_p[mid_point, 0], seg_p[mid_point, 1],
                                             seg_p[mid_point + (direction * step), 0] - seg_p[mid_point, 0],
                                             seg_p[mid_point + (direction * step), 1] - seg_p[mid_point, 1],
                                             lw=0, length_includes_head=False, head_width=headwidth_bundle_, color=rgba,
                                             shape='full', alpha=headwidth_alpha, zorder=5)

                                else:
                                    ax[c].arrow(seg_p[mid_point, 0], seg_p[mid_point, 1],
                                                seg_p[mid_point + (direction * step), 0] - seg_p[mid_point, 0],
                                                seg_p[mid_point + (direction * step), 1] - seg_p[mid_point, 1],
                                                lw=0, length_includes_head=False, head_width=headwidth_bundle_,
                                                color=rgba, shape='full', alpha=headwidth_alpha, zorder=5)

                            else:
                                ax[r, c].arrow(seg_p[mid_point, 0], seg_p[mid_point, 1],
                                               seg_p[mid_point + (direction * step), 0] - seg_p[mid_point, 0],
                                               seg_p[mid_point + (direction * step), 1] - seg_p[mid_point, 1],
                                               lw=0, length_includes_head=False, head_width=headwidth_bundle_,
                                               color=rgba, shape='full', alpha=headwidth_alpha, zorder=5)
                            arrow_coords.append([seg_p[mid_point, 0], seg_p[mid_point, 1]])

                    seg_count += 1
                if show_milestones == False:
                    size_milestones = 0.01
                    show_milestones = True
                    scale_scatter_size_pop = False
                if show_milestones == True:
                    milestone_numeric_values_normed = []
                    milestone_numeric_values_rgba = []
                    for ei, i in enumerate(milestone_numeric_values):

                        # if you have numeric values (days, hours) that need to be scaled to 0-1 so they can be used in cmap()
                        if len(lineage_pathway) > 0:  # these are probabilities and we dont want to normalize the lineage likelihooods

                            rgba_ = cmap(i)
                            color_numeric = (i)
                        else:

                            rgba_ = cmap((i - min_numerical_value) / (max_numerical_value - min_numerical_value))
                            color_numeric = (i - min_numerical_value) / (max_numerical_value - min_numerical_value)
                        milestone_numeric_values_normed.append(color_numeric)
                        milestone_numeric_values_rgba.append(
                            rgba_)  # need a list of rgb when also plotting labels as plot colors are done one-by-one
                    if scale_scatter_size_pop == True:

                        n_samples = layout.shape[0]
                        sqrt_nsamples = math.sqrt(n_samples)
                        group_pop_scale = [math.log(6 + i / sqrt_nsamples) for i in
                                           hammerbundle_dict['milestone_embedding']['cluster population']]
                        size_scatter_scaled = [size_milestones * i for i in group_pop_scale]
                    else:
                        size_scatter_scaled = size_milestones  # constant value
                    # NOTE # using vmax=1 in the scatter plot would mean that all values are plotted relative to a 0-1 scale and the legend for all plots is 0-1.
                    # If we want to allow that each legend is unique then there is autoscaling of the colors such that the max color is set to the max value of that particular subplot (even if that max value is well below 1)
                    if fig_nrows == 1:
                        if fig_ncols == 1:

                            im = ax.scatter(layout[:, 0], layout[:, 1], s=0.01,
                                            c=milestone_numeric_values_normed, cmap=cmap_name,
                                            edgecolors='None')  # without alpha parameter which otherwise gets passed onto the colorbar

                            ax.scatter(layout[:, 0], layout[:, 1], s=size_scatter_scaled,
                                       c=milestone_numeric_values_normed, cmap=cmap_name, alpha=alpha_milestones,
                                       edgecolors='None')

                        else:
                            im = ax[c].scatter(layout[:, 0], layout[:, 1], s=0.01, c=milestone_numeric_values_normed,
                                               cmap=cmap_name,
                                               edgecolors='None')  # without alpha parameter which otherwise gets passed onto the colorbar

                            ax[c].scatter(layout[:, 0], layout[:, 1], s=size_scatter_scaled,
                                          c=milestone_numeric_values_normed, cmap=cmap_name, alpha=alpha_milestones,
                                          edgecolors='None')
                    else:
                        '''

                        ax[r, c].scatter(layout[:, 0], layout[:, 1], s=size_scatter_scaled*3,
                                         c=milestone_numeric_values_normed, cmap=cmap_name,
                                         alpha=alpha_milestones*0.5, edgecolors='none', vmin=min_numerical_value)  # vmax=1)


                        '''
                        im = ax[r, c].scatter(layout[:, 0], layout[:, 1], c=milestone_numeric_values_normed, s=0.01,
                                              cmap=cmap_name, edgecolors='none', vmin=min_numerical_value)
                        ax[r, c].scatter(layout[:, 0], layout[:, 1], s=size_scatter_scaled,
                                         c=milestone_numeric_values_normed, cmap=cmap_name,
                                         alpha=alpha_milestones, edgecolors='none', vmin=min_numerical_value)  # vmax=1)

                        ''''
                        if len(lineage_pathway)>0:
                            #accentuate the scatter size for nodes significant to a lineage
                            for j in range(layout.shape[0]):
                                if milestone_numeric_values_normed[j] > 0.5*max_numerical_value:
                                    if scale_scatter_size_pop:
                                        ax[r, c].scatter(layout[j, 0], layout[j, 1], s=size_scatter_scaled[j] * 1.5,
                                             c=milestone_numeric_values_normed[j], cmap=cmap_name,
                                             alpha=alpha_milestones * 1.5, edgecolors='None',
                                             vmin=min_numerical_value)  # vmax=1)
                                    else:
                                        print(f'node {j} {milestone_numeric_values_normed[j]}')
                                        ax[r, c].scatter(layout[j, 0], layout[j, 1], s=size_scatter_scaled*1.5,
                                                     c=milestone_numeric_values_normed[j], cmap=cmap_name,
                                                     alpha=alpha_milestones*1.5, edgecolors='None',
                                                     vmin=min_numerical_value)  # vmax=1)
                        '''
                    if text_labels == True:

                        # if text labels is true but user has not provided any labels at the sc level from which to create milestone categorical labels
                        if sc_labels is None:
                            if via_object is not None:
                                sc_labels = via_object.true_label
                            else:
                                print(
                                    f'{datetime.now()}\t ERROR: in order to show labels, please provide list of sc_labels at the single cell level OR via_object')
                        for i in range(layout.shape[0]):
                            sc_milestone_labels = hammerbundle_dict['sc_milestone_labels']
                            loc_milestone = np.where(np.asarray(sc_milestone_labels) == i)[0]

                            mode_label = func_mode(list(np.asarray(sc_labels)[loc_milestone]))
                            if scale_scatter_size_pop == True:
                                if fig_nrows == 1:
                                    if fig_ncols == 1:
                                        ax.scatter(layout[i, 0], layout[i, 1], s=size_scatter_scaled[i],
                                                   c=np.array([milestone_numeric_values_rgba[i]]),
                                                   alpha=alpha_milestones, edgecolors='None', label=mode_label)
                                    else:
                                        ax[c].scatter(layout[i, 0], layout[i, 1], s=size_scatter_scaled[i],
                                                      c=np.array([milestone_numeric_values_rgba[i]]),
                                                      alpha=alpha_milestones, edgecolors='None', label=mode_label)
                                else:
                                    ax[r, c].scatter(layout[i, 0], layout[i, 1], s=size_scatter_scaled[i],
                                                     c=np.array([milestone_numeric_values_rgba[i]]),
                                                     alpha=alpha_milestones, edgecolors='None', label=mode_label)
                            else:
                                if fig_nrows == 1:
                                    if fig_ncols == 1:
                                        ax.scatter(layout[i, 0], layout[i, 1], s=size_scatter_scaled,
                                                   c=np.array([milestone_numeric_values_rgba[i]]),
                                                   alpha=alpha_milestones, edgecolors='None', label=mode_label)
                                    else:
                                        ax[c].scatter(layout[i, 0], layout[i, 1], s=size_scatter_scaled,
                                                      c=np.array([milestone_numeric_values_rgba[i]]),
                                                      alpha=alpha_milestones, edgecolors='None', label=mode_label)
                                else:
                                    ax[r, c].scatter(layout[i, 0], layout[i, 1], s=size_scatter_scaled,
                                                     c=np.array([milestone_numeric_values_rgba[i]]),
                                                     alpha=alpha_milestones, edgecolors='None', label=mode_label)
                            if fig_nrows == 1:
                                if fig_ncols == 1:
                                    ax.text(layout[i, 0], layout[i, 1], mode_label, style='italic',
                                            fontsize=fontsize_labels, color="black")
                                else:
                                    ax[c].text(layout[i, 0], layout[i, 1], mode_label, style='italic',
                                               fontsize=fontsize_labels, color="black")
                            else:
                                ax[r, c].text(layout[i, 0], layout[i, 1], mode_label, style='italic',
                                              fontsize=fontsize_labels, color="black")
                time = datetime.now()
                time = time.strftime("%H:%M")
                if len(lineage_pathway) == 0:
                    title_ = extra_title_text + ' n_milestones = ' + str(int(layout.shape[0]))  # + ' time: ' + time
                else:

                    title_ = 'lineage:' + str(lineage_pathway[counter_]) + '-' + str(majority_composition)

                if fig_nrows == 1:
                    if fig_ncols == 1:
                        ax.axis('off')
                        ax.grid(False)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.set_facecolor(facecolor)
                        ax.set_title(label=title_, color='black', fontsize=fontsize_title)
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        if len(lineage_pathway) > 0:
                            cb = fig.colorbar(im, cax=cax, orientation='vertical', label='lineage likelihood')
                        else:
                            cb = fig.colorbar(im, cax=cax, orientation='vertical', label='pseudotime')
                        ax_cb = cb.ax
                        text = ax_cb.yaxis.label
                        font = matplotlib.font_manager.FontProperties(
                            size=fontsize_title)  # family='times new roman', style='italic',
                        text.set_font_properties(font)
                        ax_cb.tick_params(labelsize=int(fontsize_title * 0.8))
                        cb.outline.set_visible(False)

                    else:
                        ax[c].axis('off')
                        ax[c].grid(False)
                        ax[c].spines['top'].set_visible(False)
                        ax[c].spines['right'].set_visible(False)
                        ax[c].set_facecolor(facecolor)
                        ax[c].set_title(label=title_, color='black', fontsize=fontsize_title)

                        divider = make_axes_locatable(ax[c])
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        if (len(lineage_pathway)) > 0:
                            colorbar_legend = 'lineage likelihood'
                            cb = fig.colorbar(im, cax=cax, orientation='vertical', label=colorbar_legend)
                        else:
                            cb = fig.colorbar(im, cax=cax, orientation='vertical', label=colorbar_legend)

                        ax_cb = cb.ax
                        text = ax_cb.yaxis.label
                        font = matplotlib.font_manager.FontProperties(
                            size=fontsize_title)  # family='times new roman', style='italic',
                        text.set_font_properties(font)
                        ax_cb.tick_params(labelsize=int(fontsize_title * 0.8))
                        cb.outline.set_visible(False)

                else:

                    ax[r, c].axis('off')
                    ax[r, c].grid(False)
                    ax[r, c].spines['top'].set_visible(False)
                    ax[r, c].spines['right'].set_visible(False)
                    ax[r, c].set_facecolor(facecolor)
                    ax[r, c].set_title(label=title_, color='black', fontsize=fontsize_title)

                    divider = make_axes_locatable(ax[r, c])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    if len(lineage_pathway) > 0:
                        cb = fig.colorbar(im, cax=cax, orientation='vertical', label='Lineage likelihood')
                    else:
                        cb = fig.colorbar(im, cax=cax, orientation='vertical', label='pseudotime')

                    ax_cb = cb.ax
                    text = ax_cb.yaxis.label
                    font = matplotlib.font_manager.FontProperties(
                        size=fontsize_title)  # family='times new roman', style='italic',
                    text.set_font_properties(font)
                    ax_cb.tick_params(labelsize=int(fontsize_title * 0.8))
                    cb.outline.set_visible(False)

                counter_ += 1
            else:
                if fig_nrows == 1:
                    if fig_ncols == 1:
                        ax.axis('off')
                        ax.grid(False)
                    else:
                        ax[c].axis('off')
                        ax[c].grid(False)
                else:
                    ax[r, c].axis('off')
                    ax[r, c].grid(False)
    return fig, ax


def animate_atlas(hammerbundle_dict=None, via_object=None, linewidth_bundle=2, frame_interval: int = 10,
                  n_milestones: int = None, facecolor: str = 'white', cmap: str = 'plasma_r', extra_title_text='',
                  size_scatter: int = 1, alpha_scatter: float = 0.2,
                  saveto='/home/user/Trajectory/Datasets/animation_default.gif', time_series_labels: list = None, lineage_pathway = [],
                  sc_labels_numeric: list = None,  show_sc_embedding:bool=False, sc_emb=None, sc_size_scatter:float=10, sc_alpha_scatter:float=0.2, n_intervals:int = 50, n_repeat:int = 2):
    '''
    :param ax: axis to plot on
    :param hammer_bundle: hammerbundle object with coordinates of all the edges to draw
    :param layout: coords of cluster nodes and optionally also contains the numeric value associated with each cluster (such as time-stamp) layout[['x','y','numeric label']] sc/cluster/milestone level
    :param CSM: cosine similarity matrix. cosine similarity between the RNA velocity between neighbors and the change in gene expression between these neighbors. Only used when available
    :param velocity_weight: percentage weightage given to the RNA velocity based transition matrix
    :param pt: cluster-level pseudotime
    :param alpha_bundle: alpha when drawing lines
    :param linewidth_bundle: linewidth of bundled lines
    :param edge_color:
    :param frame_interval: smaller number, faster refresh and video
    :param facecolor: default = white
    :param headwidth_bundle: headwidth of arrows used in bundled edges
    :param arrow_frequency: min dist between arrows (bundled edges otherwise have overcrowding of arrows)
    :param show_direction: True will draw arrows along the lines to indicate direction
    :param milestone_edges: pandas DataFrame milestone_edges[['source','target']]
    :param t_diff_factor scaling the average the time intervals (0.25 means that for each frame, the time is progressed by 0.25* mean_time_differernce_between adjacent times (only used when sc_labels_numeric are directly passed instead of using pseudotime)
    :param show_sc_embedding: plot the single cell embedding under the edges
    :param sc_emb numpy array of single cell embedding (ncells x 2)
    :param sc_alpha_scatter, Alpha transparency value of points of single cells (1 is opaque, 0 is fully transparent)
    :param sc_size_scatter. size of scatter points of single cells
    :param n_repeat. number of times you repeat the whole process
    :return: axis with bundled edges plotted
    '''
    import tqdm
    cmap = plt.get_cmap(cmap)
    if show_sc_embedding:
        if sc_emb is None:
            sc_emb= via_object.embedding
            if sc_emb is None:
                print('please provide a single cell embedding as an array')
                return
    if hammerbundle_dict is None:
        if via_object is None:
            print(
                f'{datetime.now()}\tERROR: Hammerbundle_dict needs to be provided either through via_object or by running make_edgebundle_milestone()')
        else:
            hammerbundle_dict = via_object.hammerbundle_milestone_dict
            if hammerbundle_dict is None:
                if n_milestones is None: n_milestones = min(via_object.nsamples, 150)
                if sc_labels_numeric is None:
                    if via_object.time_series_labels is not None:
                        sc_labels_numeric = via_object.time_series_labels
                    else:
                        sc_labels_numeric = via_object.single_cell_pt_markov

                hammerbundle_dict = make_edgebundle_milestone(via_object=via_object,
                                                              embedding=via_object.embedding,
                                                              sc_graph=via_object.ig_full_graph,
                                                              n_milestones=n_milestones,
                                                              sc_labels_numeric=sc_labels_numeric,
                                                              initial_bandwidth=0.02, decay=0.7, weighted=True)
            hammer_bundle = hammerbundle_dict['hammerbundle']
            layout = hammerbundle_dict['milestone_embedding'][['x', 'y']].values
            milestone_edges = hammerbundle_dict['edges']
            milestone_numeric_values = hammerbundle_dict['milestone_embedding']['numeric label']


    else:
        hammer_bundle = hammerbundle_dict['hammerbundle']
        layout = hammerbundle_dict['milestone_embedding'][['x', 'y']].values
        milestone_edges = hammerbundle_dict['edges']
        milestone_numeric_values = hammerbundle_dict['milestone_embedding']['numeric label']

    fig, ax = plt.subplots(facecolor=facecolor, figsize=(15, 12))
    n_milestones = len(milestone_numeric_values)

    if len(lineage_pathway) > 0:
                milestone_lin_values = hammerbundle_dict['milestone_embedding'][
                    'sc_lineage_probability_' + str(lineage_pathway[0])]
                p1_sc_bp = np.nan_to_num(via_object.single_cell_bp, nan=0.0, posinf=0.0,
                                         neginf=0.0)  # single cell lineage probabilities sc pb
                # row normalize
                row_sums = p1_sc_bp.sum(axis=1)
                p1_sc_bp = p1_sc_bp / row_sums[:,
                                      np.newaxis]  # make rowsums a column vector where i'th entry is sum of i'th row in p1-sc-bp
                ts_cluster_number = lineage_pathway[0]
                ts_array_original = np.asarray(via_object.terminal_clusters)
                loc_ts_current = np.where(ts_array_original == ts_cluster_number)[0][0]
                print(f'{datetime.now()}\tlocation of {lineage_pathway[0]} is at {np.where(ts_array_original == ts_cluster_number)[0]} and {loc_ts_current}')
                p1_sc_bp = p1_sc_bp[:, loc_ts_current]
                rgba_lineage_sc = []
                rgba_lineage_milestone = []
                min_p1_sc_pb = min(p1_sc_bp)
                max_p1_sc_pb = max(p1_sc_bp)
                min_milestone_lin_values = min(milestone_lin_values)
                max_milestone_lin_values = max(milestone_lin_values)
                print(f"{datetime.now()}\t making rgba_lineage_sc")
                for i in p1_sc_bp:
                    rgba_lineage_sc_ = cmap((i - min_p1_sc_pb) / (max_p1_sc_pb - min_p1_sc_pb))
                    rgba_lineage_sc.append(rgba_lineage_sc_)
                print(f"{datetime.now()}\t making rgba_lineage_sc")
                for i in milestone_lin_values:
                    rgba_lineage_milestone_ = cmap((i - min_milestone_lin_values) / (max_milestone_lin_values - min_milestone_lin_values))
                    rgba_lineage_milestone.append(rgba_lineage_milestone_)

    # ax.set_facecolor(facecolor)
    ax.grid(False)
    x_ = [l[0] for l in layout]
    y_ = [l[1] for l in layout]

    layout = np.asarray(layout)
    # make a knn so we can find which clustergraph nodes the segments start and end at

    # get each segment. these are separated by nans.
    hbnp = hammer_bundle.to_numpy()
    splits = (np.isnan(hbnp[:, 0])).nonzero()[0]  # location of each nan values
    edgelist_segments = []
    start = 0
    segments = []
    arrow_coords = []
    seg_len = []  # length of a segment
    for stop in splits:
        seg = hbnp[start:stop, :]
        segments.append(seg)
        seg_len.append(seg.shape[0])
        start = stop

    min_seg_length = min(seg_len)
    max_seg_length = max(seg_len)
    # mean_seg_length = sum(seg_len)/len(seg_len)
    seg_len = np.asarray(seg_len)
    seg_len = np.clip(seg_len, a_min=np.percentile(seg_len, 10),
                      a_max=np.percentile(seg_len, 90))



    if milestone_numeric_values is not None:
        max_numerical_value = max(milestone_numeric_values)
        min_numerical_value = min(milestone_numeric_values)

    seg_count = 0

    i_sorted_numeric_values = np.argsort(milestone_numeric_values)

    ee = int(n_milestones / n_intervals)
    print('ee',ee)

    loc_time_thresh = i_sorted_numeric_values[0:ee]
    for ll in loc_time_thresh:
        print('sorted numeric milestone',milestone_numeric_values[ll])
    # print('loc time thres', loc_time_thresh)
    milestone_edges['source_thresh'] = milestone_edges['source'].isin(
        loc_time_thresh)  # apply(lambda x: any([k in x for k in loc_time_thresh]))

    # print(milestone_edges[0:10])
    idx = milestone_edges.index[milestone_edges['source_thresh']].tolist()
    # print('loc time thres', time_thresh, loc_time_thresh)
    for i in idx:
        seg = segments[i]
        source_milestone = milestone_edges['source'].values[i]
        target_milestone = milestone_edges['target'].values[i]

        # seg_weight = max(0.3, math.log(1+seg[-1,2])) seg[-1,2] column index 2 has the weight information

        seg_weight = seg[-1, 2] * seg_len[i] / (
                    max_seg_length - min_seg_length)  ##seg.shape[0] / (max_seg_length - min_seg_length)

        # cant' quite decide yet if sigmoid is desirable
        # seg_weight=sigmoid_scalar(seg.shape[0] / (max_seg_length - min_seg_length), scale=5, shift=mean_seg_length / (max_seg_length - min_seg_length))

        alpha_bundle_firstsegments = max(seg_weight, 0.1)
        if alpha_bundle_firstsegments > 1:
            alpha_bundle_firstsegments = 1
        if len(lineage_pathway)>0:
            #alpha_bundle_firstsegments = milestone_lin_values[loc_time_thresh[0]] #the alpha should be propotional to the lineage_pb of these segments
            alpha_bundle_firstsegments = milestone_lin_values[source_milestone]
            if alpha_bundle_firstsegments < 0.7: alpha_bundle_firstsegments *= alpha_bundle_firstsegments

        if milestone_numeric_values is not None:

            if len(lineage_pathway) > 0:
                source_milestone_numerical_value = milestone_lin_values[source_milestone]
                target_milestone_numerical_value = milestone_lin_values[target_milestone]
                rgba_milestone_value = min(source_milestone_numerical_value, target_milestone_numerical_value)
                rgba = cmap((rgba_milestone_value - min_milestone_lin_values) / (max_milestone_lin_values - min_milestone_lin_values))
            else:
                source_milestone_numerical_value = milestone_numeric_values[source_milestone]
                target_milestone_numerical_value = milestone_numeric_values[target_milestone]
                rgba_milestone_value = min(source_milestone_numerical_value, target_milestone_numerical_value)
                rgba = cmap((rgba_milestone_value - min_numerical_value) / (max_numerical_value - min_numerical_value))
        else:
            rgba = cmap(min(seg_weight, 0.95))  # cmap(seg.shape[0]/(max_seg_length-min_seg_length))
        # if seg_weight>0.05: seg_weight=0.1
        if seg_count % 10000 == 0: print('seg weight', seg_weight)
        seg = seg[:, 0:2].reshape(-1, 2)
        seg_p = seg[~np.isnan(seg)].reshape((-1, 2))

        ax.plot(seg_p[:, 0], seg_p[:, 1], linewidth=linewidth_bundle * seg_weight, alpha=alpha_bundle_firstsegments,
                color=rgba)  # edge_color )
        seg_count += 1
    milestone_numeric_values_rgba = []

    print(f'{datetime.now()}\there1 in animate()')
    if milestone_numeric_values is not None:
        for i in milestone_numeric_values:
            rgba_ = cmap((i - min_numerical_value) / (max_numerical_value - min_numerical_value))
            milestone_numeric_values_rgba.append(rgba_)

        if show_sc_embedding:
            if len(lineage_pathway)>0:
                weighted_alpha = [sc_alpha_scatter*i for i in p1_sc_bp]
                weighted_alpha = [0.5*sc_alpha_scatter if i<= 0.5*sc_alpha_scatter  else i for i in weighted_alpha]
                ax.scatter(sc_emb[:, 0], sc_emb[:, 1], s=sc_size_scatter,
                       c=p1_sc_bp, alpha=weighted_alpha, cmap=cmap)
            else: ax.scatter(sc_emb[:, 0], sc_emb[:, 1], s=size_scatter,                       c='blue', alpha=0)
        if len(lineage_pathway) > 0:
            ax.scatter(layout[loc_time_thresh, 0], layout[loc_time_thresh, 1], s=size_scatter,
                       c=np.asarray(rgba_lineage_milestone)[loc_time_thresh], alpha=alpha_scatter)
        else: ax.scatter(layout[loc_time_thresh, 0], layout[loc_time_thresh, 1], s=size_scatter,
                   c=np.asarray(milestone_numeric_values_rgba)[loc_time_thresh], alpha=alpha_scatter)
        # if we dont plot all the points, then the size of axis changes and the location of the graph moves/changes as more points are added
        ax.scatter(layout[:, 0], layout[:, 1], s=size_scatter,
                   c=np.asarray(milestone_numeric_values_rgba), alpha=0)

    else:
        ax.scatter(layout[:, 0], layout[:, 1], s=size_scatter, c='red', alpha=alpha_scatter)
    print('here2 in animate()')
    ax.set_facecolor(facecolor)
    ax.axis('off')
    time = datetime.now()
    time = time.strftime("%H:%M")
    title_ = 'n_milestones = ' + str(int(layout.shape[0])) + ' time: ' + time + ' ' + extra_title_text
    ax.set_title(label=title_, color='black')
    print(f"{datetime.now()}\tFinished plotting edge bundle")
    if time_series_labels is None: #over-ride via_object's saved time_series_labels and/or pseudotime
        if via_object is not None:
            time_series_labels = via_object.time_series_labels
            if time_series_labels is None:
                time_series_labels = via_object.single_cell_pt_markov


    cycles = n_intervals

    min_time_series_labels = min(time_series_labels)
    max_time_series_labels = max(time_series_labels)
    sc_rgba = []
    for i in time_series_labels:
        sc_rgba_ = cmap((i - min_time_series_labels) / (max_time_series_labels - min_time_series_labels))
        sc_rgba.append(sc_rgba_)

    if show_sc_embedding:
        print(f"{datetime.now()}\tdoing argsort of single cell time_series_labels")
        i_sorted_sc_time = np.argsort(time_series_labels)
        print(f"{datetime.now()}\tfinish argsort of single cell time_series_labels")
    n_cells = len(time_series_labels)
    def update_edgebundle(frame_no):

        print(frame_no, 'out of', n_intervals, 'cycles')

        rem = (frame_no % n_intervals)
        if (frame_no % n_intervals)==0:
            loc_time_thresh = i_sorted_numeric_values[0:int(n_milestones/n_intervals)]

            if show_sc_embedding:
                sc_loc_time_thresh = i_sorted_sc_time[0:int(n_milestones / n_intervals)]

        else:
            loc_time_thresh = i_sorted_numeric_values[rem*int(n_milestones/n_intervals):(rem+1)*int(n_milestones/n_intervals)]

            if show_sc_embedding:
                sc_loc_time_thresh = i_sorted_sc_time[rem*int(n_cells/n_intervals):(rem+1)*int(n_cells/n_intervals)]
        #sc_loc_time_thresh = np.where((np.asarray(time_series_labels) <= time_thresh) & (                    np.asarray(time_series_labels) > time_thresh - t_diff_mean))[0].tolist()

        milestone_edges['source_thresh'] = milestone_edges['source'].isin(
            loc_time_thresh)  # apply(lambda x: any([k in x for k in loc_time_thresh]))


        idx = milestone_edges.index[milestone_edges['source_thresh']].tolist()

        for i in idx:
            seg = segments[i]
            source_milestone = milestone_edges['source'].values[i]

            # seg_weight = max(0.3, math.log(1+seg[-1,2])) seg[-1,2] column index 2 has the weight information

            seg_weight = seg[-1, 2] * seg_len[i] / (
                    max_seg_length - min_seg_length)  ##seg.shape[0] / (max_seg_length - min_seg_length)

            # cant' quite decide yet if sigmoid is desirable
            # seg_weight=sigmoid_scalar(seg.shape[0] / (max_seg_length - min_seg_length), scale=5, shift=mean_seg_length / (max_seg_length - min_seg_length))
            alpha_bundle = max(seg_weight, 0.1)  # max(0.1, math.log(1 + seg[-1, 2]))
            if alpha_bundle > 1: alpha_bundle = 1

            if milestone_numeric_values is not None:
                source_milestone_numerical_value = milestone_numeric_values[source_milestone]
                if len(lineage_pathway)==0:
                    rgba = cmap((source_milestone_numerical_value - min_numerical_value) / (                        max_numerical_value - min_numerical_value))
                else:
                    rgba = list(rgba_lineage_milestone[source_milestone])
                    #print('pre alpha modified', rgba)
                    #rgba[3] = milestone_lin_values[source_milestone] #source_milestone
                    #rgba = tuple(rgba)
                    #print('alpha modified',rgba)
            else:
                rgba = cmap(min(seg_weight, 0.95))  # cmap(seg.shape[0]/(max_seg_length-min_seg_length))
            # if seg_weight>0.05: seg_weight=0.1

            seg = seg[:, 0:2].reshape(-1, 2)
            seg_p = seg[~np.isnan(seg)].reshape((-1, 2))

            if len(lineage_pathway)>0:
                squared_alpha = milestone_lin_values[source_milestone]
                if squared_alpha<0.6: squared_alpha*=squared_alpha
                #print('squared_alpha',squared_alpha)
                ax.plot(seg_p[:, 0], seg_p[:, 1], linewidth=linewidth_bundle * seg_weight,                     color=rgba, alpha=squared_alpha)#milestone_lin_values[source_milestone])
            else: ax.plot(seg_p[:, 0], seg_p[:, 1], linewidth=linewidth_bundle * seg_weight,                     color=rgba,alpha=alpha_bundle)

        milestone_numeric_values_rgba = []

        if milestone_numeric_values is not None:
            for i in milestone_numeric_values:
                rgba_ = cmap((i - min_numerical_value) / (max_numerical_value - min_numerical_value))
                milestone_numeric_values_rgba.append(rgba_)

            if ((frame_no > n_repeat*n_intervals) and (rem ==0)): #by using > rather than >= sign, two complete cycles run before the axis is cleared and the animation is restarted
                ax.clear()
                ax.axis('off')
            else:
                ax.scatter(layout[:, 0], layout[:, 1], s=size_scatter,
                           c=np.asarray(milestone_numeric_values_rgba), alpha=0)
                if len(lineage_pathway) > 0:
                    ax.scatter(layout[loc_time_thresh, 0], layout[loc_time_thresh, 1], s=size_scatter,
                               c=np.asarray(rgba_lineage_milestone)[loc_time_thresh], alpha=alpha_scatter)
                else: ax.scatter(layout[loc_time_thresh, 0], layout[loc_time_thresh, 1], s=size_scatter,
                           c=np.asarray(milestone_numeric_values_rgba)[loc_time_thresh], alpha=alpha_scatter)

                if show_sc_embedding:
                    if len(lineage_pathway)>0:
                        ax.scatter(sc_emb[sc_loc_time_thresh, 0], sc_emb[sc_loc_time_thresh, 1], s=sc_size_scatter, edgecolors = None,
                               c=np.asarray(rgba_lineage_sc)[sc_loc_time_thresh], alpha=[p1_sc_bp[sc_loc_time_thresh]]) #no *sc_alpha_scatter

                    else: ax.scatter(sc_emb[sc_loc_time_thresh, 0], sc_emb[sc_loc_time_thresh, 1], s=sc_size_scatter,
                               c=np.asarray(sc_rgba)[sc_loc_time_thresh], alpha=sc_alpha_scatter)
        else:
            ax.scatter(layout[:, 0], layout[:, 1], s=size_scatter, c='red', alpha=alpha_scatter)
        # pbar.update()

    frame_no = int(int(cycles)*n_repeat)
    animation = FuncAnimation(fig, update_edgebundle, frames=frame_no, interval=frame_interval, repeat=False)  # 100
    # pbar = tqdm.tqdm(total=frame_no)
    # pbar.close()
    print('complete animate')

    animation.save(saveto, writer='imagemagick')  # , fps=30)
    print('saved animation')
    plt.show()
    return


def animate_streamplot(via_object, embedding, density_grid=1,
                       linewidth=0.5, min_mass=1, cutoff_perc=None, scatter_size=500, scatter_alpha=0.2,
                       marker_edgewidth=0.1, smooth_transition=1, smooth_grid=0.5, color_scheme='annotation',
                       other_labels=[], b_bias=20, n_neighbors_velocity_grid=None, fontsize=8, alpha_animate=0.7,
                       cmap_scatter='rainbow', cmap_stream='Blues', segment_length=1,
                       saveto='/home/shobi/Trajectory/Datasets/animation.gif', use_sequentially_augmented=False,
                       facecolor_='white', random_seed=0):
    '''
    Draw Animated vector plots. the Saved .gif file saved at the saveto address, is the best for viewing the animation as the fig, ax output can be slow

    :param via_object: viaobject
    :param embedding: ndarray (nsamples,2) umap, tsne, via-umap, via-mds
    :param density_grid:
    :param linewidth:
    :param min_mass:
    :param cutoff_perc:
    :param scatter_size:
    :param scatter_alpha:
    :param marker_edgewidth:
    :param smooth_transition:
    :param smooth_grid:
    :param color_scheme: 'annotation', 'cluster', 'other'
    :param add_outline_clusters:
    :param cluster_outline_edgewidth:
    :param gp_color:
    :param bg_color:
    :param title:
    :param b_bias:
    :param n_neighbors_velocity_grid:
    :param fontsize:
    :param alpha_animate:
    :param cmap_scatter:
    :param cmap_stream: string of a cmap for streamlines, default = 'Blues' (for dark blue lines) . Consider 'Blues_r' for white lines OR 'Greys/_r' 'gist_yard/_r'
    :param color_stream: string like 'white'. will override cmap_stream
    :param segment_length:

    :return: fig, ax.
    '''
    import tqdm

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, writers
    from matplotlib.collections import LineCollection
    from .windmap import Streamlines
    import matplotlib.patheffects as PathEffects
    print(f'{datetime.now()}\tStep1 velocity embedding')
    # import cartopy.crs as ccrs
    if embedding is None:
        embedding = via_object.embedding
        if embedding is None: print(
            f'ERROR: please provide input parameter embedding of ndarray with shape (nsamples, 2)')
    V_emb = _pl_velocity_embedding(via_object, embedding, smooth_transition, b=b_bias,
                                           use_sequentially_augmented=use_sequentially_augmented)

    V_emb *= 10  # the velocity of the samples has shape (n_samples x 2).*100
    print(f'{datetime.now()}\tStep2 interpolate')
    # interpolate the velocity along all grid points based on the velocities of the samples in V_emb
    X_grid, V_grid = compute_velocity_on_grid(
        X_emb=embedding,
        V_emb=V_emb,
        density=density_grid,
        smooth=smooth_grid,
        min_mass=min_mass,
        autoscale=False,
        adjust_for_stream=True,
        cutoff_perc=cutoff_perc, n_neighbors=n_neighbors_velocity_grid)
    print(f'{datetime.now()}\tInside animated. File will be saved to location {saveto}')


    # lengths = np.sqrt((V_grid ** 2).sum(0))

    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(1, 1, 1)
    fig.patch.set_visible(False)
    if color_scheme == 'time':
        ax.scatter(embedding[:, 0], embedding[:, 1], c=via_object.single_cell_pt_markov, alpha=scatter_alpha, zorder=0,
                   s=scatter_size, linewidths=marker_edgewidth, cmap=cmap_scatter)
    else:
        if color_scheme == 'annotation': color_labels = via_object.true_label
        if color_scheme == 'cluster': color_labels = via_object.labels
        if color_scheme == 'other': color_labels = other_labels

        n_true = len(set(color_labels))
        lin_col = np.linspace(0, 1, n_true)
        col = 0
        cmap = plt.get_cmap(cmap_scatter)  # 'twilight' is nice too
        cmap = cmap(np.linspace(0.01, 0.80, n_true))  # .95
        # cmap = cmap(np.linspace(0.5, 0.95, n_true))
        for color, group in zip(lin_col, sorted(set(color_labels))):
            color_ = np.asarray(cmap[col]).reshape(-1, 4)

            color_[0, 3] = scatter_alpha

            where = np.where(np.array(color_labels) == group)[0]

            ax.scatter(embedding[where, 0], embedding[where, 1], label=group,
                       c=color_,
                       alpha=scatter_alpha, zorder=0, s=scatter_size,
                       linewidths=marker_edgewidth)  # plt.cm.rainbow(color))

            x_mean = embedding[where, 0].mean()
            y_mean = embedding[where, 1].mean()
            ax.text(x_mean, y_mean, '', fontsize=fontsize, zorder=4,
                    path_effects=[PathEffects.withStroke(linewidth=linewidth, foreground='w')],
                    weight='bold')  # str(group)
            col += 1
        ax.set_facecolor(facecolor_)

    lengths = []
    colors = []
    lines = []
    linewidths = []
    count = 0
    # X, Y, U, V = interpolate_static_stream(X_grid[0], X_grid[1], V_grid[0],V_grid[1])
    s = Streamlines(X_grid[0], X_grid[1], V_grid[0], V_grid[1])
    # s = Streamlines(X,Y,U,V)

    for streamline in s.streamlines:
        random_seed += 1
        count += 1
        x, y = streamline
        # interpolate x, y data to handle nans
        x_ = np.array(x)
        nans, func_ = nan_helper(x_)
        x_[nans] = np.interp(func_(nans), func_(~nans), x_[~nans])

        y_ = np.array(y)
        nans, func_ = nan_helper(y_)
        y_[nans] = np.interp(func_(nans), func_(~nans), y_[~nans])

        # test=proj.transform_points(x=np.array(x),y=np.array(y),src_crs=proj)
        # points = np.array([x, y]).T.reshape(-1, 1, 2)
        points = np.array([x_, y_]).T.reshape(-1, 1, 2)
        # print('points')
        # print(points.shape)

        segments = np.concatenate([points[:-1], points[1:]], axis=1)  # nx2x2

        n = len(segments)

        D = np.sqrt(((points[1:] - points[:-1]) ** 2).sum(axis=-1)) / segment_length
        np.random.seed(random_seed)
        L = D.cumsum().reshape(n, 1) + np.random.uniform(0, 1)

        C = np.zeros((n, 4))  # 3
        C[::-1] = (L * 1.5) % 1
        C[:, 3] = alpha_animate

        lw = L.flatten().tolist()

        line = LineCollection(segments, color=C, linewidth=1)  # 0.1 when changing linewidths in update
        # line = LineCollection(segments, color=C_locationbased, linewidth=1)
        lengths.append(L)

        colors.append(C)
        linewidths.append(lw)
        lines.append(line)

        ax.add_collection(line)
    print('total number of stream lines', count)

    ax.set_xlim(min(X_grid[0]), max(X_grid[0]))
    ax.set_xticks([])
    ax.set_ylim(min(X_grid[1]), max(X_grid[1]))
    ax.set_yticks([])
    plt.tight_layout()

    # print('colors', colors)
    def update(frame_no):
        cmap = plt.get_cmap(cmap_stream)
        # cmap = cmap(np.linspace(0.1, 0.2, 100)) #darker portion
        cmap = cmap(np.linspace(0.8, 0.9, 100))  # lighter portion
        for i in range(len(lines)):
            lengths[i] -= 0.05
            # esthetic factors here by adding 0.1 and doing some clipping, 0.1 ensures no "hard blacks"
            colors[i][::-1] = np.clip(0.1 + (lengths[i] * 1.5) % 1, 0.2, 0.9)
            colors[i][:, 3] = alpha_animate
            temp = (lengths[i] * 1) % 2  # *1.5

            # temp = (lengths[i] * 1.5) % 1  # *1.5 original until Sep 7 2022
            linewidths[i] = temp.flatten().tolist()
            # if i==5: print('temp', linewidths[i])
            '''
            if i%5 ==0:
                print('lengths',i, lengths[i])
                colors[i][::-1] = (lengths[i] * 1.5) % 1
                colors[i][:, 0] = 1
            '''

            # CMAP COLORS
            # cmap_colors = [cmap(j) for j in colors[i][:,0]] #when using full cmap_stream
            cmap_colors = [cmap[int(j * 100)] for j in colors[i][:, 0]]  # when using truncated cmap_stream
            '''
            cmap_colors = [cmap[int(j*100)] for j in colors[i][:, 0]]
            linewidths[i] = [f[0]*2 for f in cmap_colors]
            if i ==5: print('colors', [f[0]for f in cmap_colors])
            '''
            for row in range(colors[i].shape[0]):
                colors[i][row, :] = cmap_colors[row][0:4]
                # colors[i][row, 3] = (1-colors[i][row][0])*0.6#alpha_animate
                # linewidths[i][row] = 2-((colors[i][row][0])%2) #1-colors[i]... #until 7 sept 2022

                # if color_stream is not None: colors[i][row, :] =  matplotlib.colors.to_rgba_array(color_stream)[0] #monochrome is nice 1 or 0
            # if i == 5: print('lw', linewidths[i])
            colors[i][:, 3] = alpha_animate
            lines[i].set_linewidth(linewidths[i])
            lines[i].set_color(colors[i])
        pbar.update()

    n = 250  # 27

    animation = FuncAnimation(fig, update, frames=n, interval=40)
    pbar = tqdm.tqdm(total=n)
    pbar.close()
    animation.save(saveto, writer='imagemagick', fps=25)
    # animation.save('/home/shobi/Trajectory/Datasets/Toy3/wind_density_ffmpeg.mp4', writer='ffmpeg', fps=60)

    # fig.patch.set_visible(False)
    # ax.axis('off')
    plt.show()
    return fig, ax


def via_streamplot(via_object, embedding: ndarray = None, density_grid: float = 0.5, arrow_size: float = 0.7,
                   arrow_color: str = 'k', color_dict: dict = None,
                   arrow_style="-|>", max_length: int = 4, linewidth: float = 1, min_mass=1, cutoff_perc: int = 5,
                   scatter_size: int = 500, scatter_alpha: float = 0.5, marker_edgewidth: float = 0.1,
                   density_stream: int = 2, smooth_transition: int = 1, smooth_grid: float = 0.5,
                   color_scheme: str = 'annotation', add_outline_clusters: bool = False,
                   cluster_outline_edgewidth=0.001, gp_color='white', bg_color='black', dpi=300, title='Streamplot',
                   b_bias=20, n_neighbors_velocity_grid=None, labels: list = None, use_sequentially_augmented=False,
                   cmap: str = 'rainbow', show_text_labels: bool = True):
    '''
    Construct vector streamplot on the embedding to show a fine-grained view of inferred directions in the trajectory

    :param via_object:
    :param embedding:  np.ndarray of shape (n_samples, 2) umap or other 2-d embedding on which to project the directionality of cells
    :param density_grid:
    :param arrow_size:
    :param arrow_color:
    :param arrow_style:
    :param max_length:
    :param linewidth:  width of  lines in streamplot, default = 1
    :param min_mass:
    :param cutoff_perc:
    :param scatter_size: size of scatter points default =500
    :param scatter_alpha: transpsarency of scatter points
    :param marker_edgewidth: width of outline arround each scatter point, default = 0.1
    :param density_stream:
    :param smooth_transition:
    :param smooth_grid:
    :param color_scheme: str, default = 'annotation' corresponds to self.true_labels. Other options are 'time' (uses single-cell pseudotime) and 'cluster' (via cluster graph) and 'other'. Alternatively provide labels as a list
    :param add_outline_clusters:
    :param cluster_outline_edgewidth:
    :param gp_color:
    :param bg_color:
    :param dpi:
    :param title:
    :param b_bias: default = 20. higher value makes the forward bias of pseudotime stronger
    :param n_neighbors_velocity_grid:
    :param labels: list (will be used for the color scheme) or if a color_dict is provided these labels should match
    :param use_sequentially_augmented:
    :param cmap:
    :return: fig, ax
    '''
    """



   Parameters
   ----------
   X_emb:

   scatter_size: int, default = 500

   linewidth:

   marker_edgewidth: 

   streamplot matplotlib.pyplot instance of fine-grained trajectories drawn on top of scatter plot
   """

    import matplotlib.patheffects as PathEffects
    if embedding is None:
        embedding = via_object.embedding
        if embedding is None:
            print(
                f'{datetime.now()}\tWARNING: please assign ambedding attribute to via_object as via_object.embedding = ndarray of [n_cells x 2]')

    V_emb = via_object._velocity_embedding(embedding, smooth_transition, b=b_bias,
                                           use_sequentially_augmented=use_sequentially_augmented)

    V_emb *= 20  # 5

    X_grid, V_grid = compute_velocity_on_grid(
        X_emb=embedding,
        V_emb=V_emb,
        density=density_grid,
        smooth=smooth_grid,
        min_mass=min_mass,
        autoscale=False,
        adjust_for_stream=True,
        cutoff_perc=cutoff_perc, n_neighbors=n_neighbors_velocity_grid)

    # adapted from : https://github.com/theislab/scvelo/blob/1805ab4a72d3f34496f0ef246500a159f619d3a2/scvelo/plotting/velocity_embedding_grid.py#L27
    lengths = np.sqrt((V_grid ** 2).sum(0))

    linewidth = 1 if linewidth is None else linewidth
    # linewidth *= 2 * lengths / np.percentile(lengths[~np.isnan(lengths)],90)
    linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()

    # linewidth=0.5
    fig, ax = plt.subplots(dpi=dpi)
    ax.grid(False)
    ax.streamplot(X_grid[0], X_grid[1], V_grid[0], V_grid[1], color=arrow_color, arrowsize=arrow_size,
                  arrowstyle=arrow_style, zorder=3, linewidth=linewidth, density=density_stream, maxlength=max_length)

    # num_cluster = len(set(super_cluster_labels))

    if add_outline_clusters:
        # add black outline to outer cells and a white inner rim
        # adapted from scanpy (scVelo utils adapts this from scanpy)
        gp_size = (2 * (scatter_size * cluster_outline_edgewidth * .1) + 0.1 * scatter_size) ** 2

        bg_size = (2 * (scatter_size * cluster_outline_edgewidth) + math.sqrt(gp_size)) ** 2

        ax.scatter(embedding[:, 0], embedding[:, 1], s=bg_size, marker=".", c=bg_color, zorder=-2)
        ax.scatter(embedding[:, 0], embedding[:, 1], s=gp_size, marker=".", c=gp_color, zorder=-1)
    if labels is None:
        if color_scheme == 'time':
            ax.scatter(embedding[:, 0], embedding[:, 1], c=via_object.single_cell_pt_markov, alpha=scatter_alpha,
                       zorder=0, s=scatter_size, linewidths=marker_edgewidth, cmap='viridis_r')
        else:
            if color_scheme == 'annotation': color_labels = via_object.true_label
            if color_scheme == 'cluster': color_labels = via_object.labels
            cmap_ = plt.get_cmap(cmap)
            # plt.cm.rainbow(color)

            line = np.linspace(0, 1, len(set(color_labels)))
            for color, group in zip(line, sorted(set(color_labels))):
                where = np.where(np.array(color_labels) == group)[0]
                ax.scatter(embedding[where, 0], embedding[where, 1], label=group,
                           c=np.asarray(cmap_(color)).reshape(-1, 4),
                           alpha=scatter_alpha, zorder=0, s=scatter_size, linewidths=marker_edgewidth)

                if show_text_labels:
                    x_mean = embedding[where, 0].mean()
                    y_mean = embedding[where, 1].mean()
                    ax.text(x_mean, y_mean, str(group), fontsize=5, zorder=4,
                            path_effects=[PathEffects.withStroke(linewidth=1, foreground='w')], weight='bold')
    elif labels is not None:
        if (isinstance(labels[0], str)) == True:  # labels are categorical
            if color_dict is not None:
                for key in color_dict:
                    loc_key = np.where(np.asarray(labels) == key)[0]
                    ax.scatter(embedding[loc_key, 0], embedding[loc_key, 1], color=color_dict[key], label=key,
                               s=scatter_size,
                               alpha=scatter_alpha, zorder=0, linewidths=marker_edgewidth)
                    x_mean = embedding[loc_key, 0].mean()
                    y_mean = embedding[loc_key, 1].mean()
                    if show_text_labels == True: ax.text(x_mean, y_mean, key, style='italic', fontsize=10,
                                                         color="black")
            else:  # there is no color_dict but labels are categorical
                cmap_ = plt.get_cmap(cmap)

                line = np.linspace(0, 1, len(set(labels)))
                for color, group in zip(line, sorted(set(labels))):
                    where = np.where(np.array(labels) == group)[0]
                    ax.scatter(embedding[where, 0], embedding[where, 1], label=group,
                               c=np.asarray(cmap_(color)).reshape(-1, 4),
                               alpha=scatter_alpha, zorder=0, s=scatter_size, linewidths=marker_edgewidth)

                    if show_text_labels:
                        x_mean = embedding[where, 0].mean()
                        y_mean = embedding[where, 1].mean()
                        ax.text(x_mean, y_mean, str(group), fontsize=5, zorder=4,
                                path_effects=[PathEffects.withStroke(linewidth=1, foreground='w')], weight='bold')
        else:  # not categorical
            ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, alpha=scatter_alpha, zorder=0,
                       s=scatter_size, linewidths=marker_edgewidth, cmap=cmap)

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.set_title(title)
    return fig, ax


def get_gene_expression(via_object, gene_exp: pd.DataFrame, cmap: str = 'jet', dpi: int = 150, marker_genes: list = [],
                        linewidth: float = 2.0, n_splines: int = 10, spline_order: int = 4, fontsize_: int = 8, 
                        marker_lineages=[], optional_title_text: str = '', cmap_dict: dict = None, conf_int:float=0.95, driver_genes:bool=False, driver_lineage:int=None,
                        figsize=None, ncols: int = 4, show_legend: bool = True, legend_loc: str = 'right',
                        legend_fontsize: int = None):
    '''
    :param via_object: via object
    :param gene_exp: dataframe where columns are features (gene) and rows are single cells
    :param cmap: default: 'jet'
    :param dpi: default:150
    :param marker_genes: Default is to use all genes in gene_exp. other provide a list of marker genes that will be used from gene_exp.
    :param linewidth: default:2
    :param n_slines: default:10 Note n_splines must be > spline_order.
    :param spline_order: default:4 n_splines must be > spline_order.
    :param marker_lineages: Default is to use all lineage pathways. other provide a list of lineage number (terminal cluster number).
    :param cmap_dict: {lineage number: 'color'}
    :param conf_int: Confidence interval of gene expressions. Also used for identifying driver genes if driver_genes = True.
    :param driver_genes: Set True to compute and plot top 3 upregulated & downregulated driver genes expressions given terminal cell fates.
    :param driver_lineage: Provide lineage used to compute driver genes if driver_genes=True.
    :param figsize: Optional matplotlib figure size.
    :param ncols: Number of subplot columns.
    :param show_legend: Whether to draw one shared figure legend.
    :param legend_loc: Shared legend location. Use 'right' or any matplotlib legend location string.
    :param legend_fontsize: Shared legend fontsize. Defaults to fontsize_.
    :return: fig, axs
    '''
    import pygam as pg
    sc_bp_original = via_object.single_cell_bp

    if driver_genes:
        if driver_lineage is None: 
            raise KeyError(f'Please provide a lineage from {via_object.terminal_clusters} for driver genes computation')
        df_driver = compute_driver_genes(via_object, gene_exp, lineage=driver_lineage, conf_int=conf_int)
        df_driver = df_driver[df_driver['pvalue']<0.05]
        df_driver = df_driver.sort_values('corr', ascending=False)
        marker_genes = df_driver.head(3).index.tolist()+df_driver.tail(3).index.tolist()
        print(marker_genes)

    if len(marker_lineages) == 0:
        marker_lineages = via_object.terminal_clusters
        n_terminal_states = len(via_object.terminal_clusters)

    else:
        n_terminal_states = len(marker_lineages)
    if len(marker_genes) > 0: gene_exp = gene_exp[marker_genes]
    sc_pt = via_object.single_cell_pt_markov

    if cmap_dict is None:
        palette = plt.get_cmap(cmap, n_terminal_states)
        cmap_ = palette(range(n_terminal_states))
    else:
        cmap_ = cmap_dict
    n_genes = gene_exp.shape[1]

    fig_ncols = max(1, min(int(ncols), max(n_genes, 1)))
    fig_nrows, mod = divmod(n_genes, fig_ncols)
    if mod == 0: fig_nrows = fig_nrows
    if mod != 0: fig_nrows += 1

    fig, axs = plt.subplots(fig_nrows, fig_ncols, dpi=dpi, figsize=figsize)
    fig.patch.set_visible(False)
    i_gene = 0  # counter for number of genes
    i_terminal = 0  # counter for terminal cluster
    # for i in range(n_terminal_states): #[0]

    for r in range(fig_nrows):
        for c in range(fig_ncols):
            if (i_gene < n_genes):
                for enum_i_lineage, i_lineage in enumerate(marker_lineages):
                    valid_scbp = False
                    if i_lineage in via_object.terminal_clusters:
                        i_terminal = np.where(np.asarray(via_object.terminal_clusters) == i_lineage)[0]
                        if len(i_terminal) > 0:
                            sc_bp = sc_bp_original.copy()
                            valid_scbp = len(np.where(sc_bp[:, i_terminal] > 0.9)[0]) > 0
                            i_terminal = i_terminal[0]

                    # if (via_object.terminal_clusters[i_terminal] in marker_lineages and len(np.where(sc_bp[:, i_terminal] > 0.8)[ 0]) > 0): # check if terminal state is in marker_lineage and in case this terminal state i cannot be reached (sc_bp is all 0)
                    if (i_lineage in via_object.terminal_clusters) and (valid_scbp):
                        cluster_i_loc = \
                        np.where(np.asarray(via_object.labels) == via_object.terminal_clusters[i_terminal])[0]
                        majority_true = via_object.func_mode(list(np.asarray(via_object.true_label)[cluster_i_loc]))

                        gene_i = gene_exp.columns[i_gene]
                        loc_i = np.where(sc_bp[:, i_terminal] > 0.9)[0]
                        val_pt = [sc_pt[pt_i] for pt_i in loc_i]  # TODO,  replace with array to speed up

                        max_val_pt = max(val_pt)

                        loc_i_bp = np.where(sc_bp[:, i_terminal] > 0.000)[0]  # 0.001
                        loc_i_sc = np.where(np.asarray(sc_pt) <= max_val_pt)[0]

                        loc_ = np.intersect1d(loc_i_bp, loc_i_sc)

                        gam_in = np.asarray(sc_pt)[loc_]
                        x = gam_in.reshape(-1, 1)

                        y = np.asarray(gene_exp[gene_i])[loc_].reshape(-1, 1)

                        weights = np.asarray(sc_bp[:, i_terminal])[loc_].reshape(-1, 1)

                        if len(loc_) > 1:

                            geneGAM = pg.LinearGAM(n_splines=n_splines, spline_order=spline_order, lam=10).fit(x, y,
                                                                                                               weights=weights)
                            xval = np.linspace(min(sc_pt), max_val_pt, 100 * 2)
                            yg = geneGAM.predict(X=xval)
                            x_conf_int = geneGAM.confidence_intervals(xval, width=conf_int)
                            A_under_curve = np_trapz(yg, x=xval)
                            print(f'Area under curve {gene_i} for branch {majority_true} is {A_under_curve}')

                        else:
                            print(
                                f'{datetime.now()}\tLineage {i_terminal} cannot be reached. Exclude this lineage in trend plotting')
                            continue
                        if cmap_dict is None:
                            color_ = cmap_[enum_i_lineage]
                        else:
                            print('cmap dict', cmap_)
                            print('i_lineage', i_lineage)
                            color_ = cmap_[i_lineage]
                        if fig_nrows > 1:
                            axs[r, c].plot(xval, yg, color=color_, linewidth=linewidth, zorder=3,
                                           label=f"Lineage:{majority_true} {via_object.terminal_clusters[i_terminal]}")
                            axs[r, c].fill_between(xval, x_conf_int[:,0], x_conf_int[:,1], color=color_, alpha=0.2)
                            axs[r, c].set_title(gene_i + optional_title_text, fontsize=fontsize_)
                            # Set tick font size
                            for label in (axs[r, c].get_xticklabels() + axs[r, c].get_yticklabels()):
                                label.set_fontsize(fontsize_ - 1)
                            axs[r, c].spines['top'].set_visible(False)
                            axs[r, c].spines['right'].set_visible(False)
                            axs[r, c].grid(False)
                        else:
                            axs[c].plot(xval, yg, color=color_, linewidth=linewidth, zorder=3,
                                        label=f"Lineage:{majority_true} {via_object.terminal_clusters[i_terminal]}")
                            axs[c].fill_between(xval, x_conf_int[:,0], x_conf_int[:,1], color=color_, alpha=0.2)
                            axs[c].set_title(gene_i + optional_title_text, fontsize=fontsize_)
                            # Set tick font size
                            for label in (axs[c].get_xticklabels() + axs[c].get_yticklabels()):
                                label.set_fontsize(fontsize_ - 1)
                            axs[c].spines['top'].set_visible(False)
                            axs[c].spines['right'].set_visible(False)
                            axs[c].grid(False)
                i_gene += 1
            else:
                if fig_nrows > 1:
                    axs[r, c].axis('off')
                    axs[r, c].grid(False)
                else:
                    axs[c].axis('off')
                    axs[c].grid(False)
    axes = np.ravel(axs)
    handles = []
    labels = []
    for ax in axes[:n_genes]:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
        ax.set_xlabel('')
        ax.set_ylabel('')

    legend_fontsize = fontsize_ if legend_fontsize is None else legend_fontsize
    right_margin = 0.96
    if show_legend and len(handles) > 0:
        if legend_loc in {'right', 'right margin'}:
            right_margin = 0.82
            fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.84, 0.5),
                       frameon=False, fontsize=legend_fontsize)
        else:
            fig.legend(handles, labels, loc=legend_loc, frameon=False, fontsize=legend_fontsize)

    try:
        fig.tight_layout(rect=[0.05, 0.06, right_margin, 0.95])
    except Exception:
        fig.subplots_adjust(left=0.08, right=right_margin, bottom=0.12, top=0.9, wspace=0.3, hspace=0.45)
    visible_axes = [ax for ax in axes[:n_genes] if ax.axison]
    if len(visible_axes) > 0:
        positions = [ax.get_position() for ax in visible_axes]
        left = min(pos.x0 for pos in positions)
        right = max(pos.x1 for pos in positions)
        bottom = min(pos.y0 for pos in positions)
        top = max(pos.y1 for pos in positions)
        fig.text((left + right) / 2, max(0.02, bottom - 0.055), 'Time',
                 ha='center', va='top', fontsize=fontsize_)
        fig.text(max(0.01, left - 0.055), (bottom + top) / 2, 'Intensity',
                 ha='right', va='center', rotation='vertical', fontsize=fontsize_)
    return fig, axs


def plot_trajectory_curves(via_object, embedding: ndarray = None, idx: Optional[list] = None,
                           title_str: str = "Pseudotime", draw_all_curves: bool = True,
                           arrow_width_scale_factor: float = 15.0,
                           scatter_size: float = 50, scatter_alpha: float = 0.5,
                           linewidth: float = 1.5, marker_edgewidth: float = 1, cmap_pseudotime: str = 'viridis_r',
                           dpi: int = 150, highlight_terminal_states: bool = True, use_maxout_edgelist: bool = False):
    '''

    projects the graph based coarse trajectory onto a umap/tsne embedding

    :param via_object: via object

    :param embedding: 2d array [n_samples x 2] with x and y coordinates of all n_samples. Umap, tsne, pca OR use the via computed embedding via_object.embedding
    :param idx: default: None. Or List. if you had previously computed a umap/tsne (embedding) only on a subset of the total n_samples (subsampled as per idx), then the via objects and results will be indexed according to idx too
    :param title_str: title of figure
    :param draw_all_curves: if the clustergraph has too many edges to project in a visually interpretable way, set this to False to get a simplified view of the graph pathways
    :param arrow_width_scale_factor:
    :param scatter_size:
    :param scatter_alpha:
    :param linewidth:
    :param marker_edgewidth:
    :param cmap_pseudotime:
    :param dpi: int default = 150. Use 300 for paper figures
    :param highlight_terminal_states: whether or not to highlight/distinguish the clusters which are detected as the terminal states by via
    :return: f, ax1, ax2
    '''
    import pygam as pg
    if embedding is None:
        embedding = via_object.embedding
        if embedding is None: print(
            f'{datetime.now()}\t ERROR please provide an embedding or compute using via_mds() or via_umap()')

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if idx is None: idx = np.arange(0, via_object.nsamples)
    cluster_labels = list(np.asarray(via_object.labels)[idx])
    super_cluster_labels = list(np.asarray(via_object.labels)[idx])
    super_edgelist = via_object.edgelist
    if use_maxout_edgelist == True:
        super_edgelist = via_object.edgelist_maxout
    true_label = list(np.asarray(via_object.true_label)[idx])
    knn = via_object.knn
    ncomp = via_object.ncomp
    final_super_terminal = via_object.terminal_clusters

    sub_terminal_clusters = via_object.terminal_clusters
    sc_pt_markov = list(np.asarray(via_object.single_cell_pt_markov)[idx])
    super_root = via_object.root[0]

    sc_supercluster_nn = sc_loc_ofsuperCluster_PCAspace(via_object, np.arange(0, len(cluster_labels)))
    # draw_all_curves. True draws all the curves in the piegraph, False simplifies the number of edges
    # arrow_width_scale_factor: size of the arrow head
    X_dimred = embedding * 1. / np.max(embedding, axis=0)
    x = X_dimred[:, 0]
    y = X_dimred[:, 1]
    max_x = np.percentile(x, 90)
    noise0 = max_x / 1000

    df = pd.DataFrame({'x': x, 'y': y, 'cluster': cluster_labels, 'super_cluster': super_cluster_labels,
                       'projected_sc_pt': sc_pt_markov},
                      columns=['x', 'y', 'cluster', 'super_cluster', 'projected_sc_pt'])
    df_mean = df.groupby('cluster', as_index=False).mean()
    sub_cluster_isin_supercluster = df_mean[['cluster', 'super_cluster']]

    sub_cluster_isin_supercluster = sub_cluster_isin_supercluster.sort_values(by='cluster')
    sub_cluster_isin_supercluster['int_supercluster'] = sub_cluster_isin_supercluster['super_cluster'].round(0).astype(
        int)

    df_super_mean = df.groupby('super_cluster', as_index=False).mean()
    pt = df_super_mean['projected_sc_pt'].values

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=[20, 10], dpi=dpi)
    num_true_group = len(set(true_label))
    num_cluster = len(set(super_cluster_labels))
    line = np.linspace(0, 1, num_true_group)
    for color, group in zip(line, sorted(set(true_label))):
        where = np.where(np.array(true_label) == group)[0]
        ax1.scatter(X_dimred[where, 0], X_dimred[where, 1], label=group,
                    c=np.asarray(plt.cm.rainbow(color)).reshape(-1, 4),
                    alpha=scatter_alpha, s=scatter_size, linewidths=marker_edgewidth * .1)  # 10 # 0.5 and 4
    ax1.legend(fontsize=6, frameon=False)
    ax1.set_title('True Labels: ncomps:' + str(ncomp) + '. knn:' + str(knn))

    G_orange = ig.Graph(n=num_cluster, edges=super_edgelist)
    ll_ = []  # this can be activated if you intend to simplify the curves
    for fst_i in final_super_terminal:

        path_orange = G_orange.get_shortest_paths(super_root, to=fst_i)[0]
        len_path_orange = len(path_orange)
        for enum_edge, edge_fst in enumerate(path_orange):
            if enum_edge < (len_path_orange - 1):
                ll_.append((edge_fst, path_orange[enum_edge + 1]))

    edges_to_draw = super_edgelist if draw_all_curves else list(set(ll_))
    for e_i, (start, end) in enumerate(edges_to_draw):
        if pt[start] >= pt[end]:
            start, end = end, start

        x_i_start = df[df['super_cluster'] == start]['x'].values
        y_i_start = df[df['super_cluster'] == start]['y'].values
        x_i_end = df[df['super_cluster'] == end]['x'].values
        y_i_end = df[df['super_cluster'] == end]['y'].values

        super_start_x = X_dimred[sc_supercluster_nn[start], 0]
        super_end_x = X_dimred[sc_supercluster_nn[end], 0]
        super_start_y = X_dimred[sc_supercluster_nn[start], 1]
        super_end_y = X_dimred[sc_supercluster_nn[end], 1]
        direction_arrow = -1 if super_start_x > super_end_x else 1

        minx = min(super_start_x, super_end_x)
        maxx = max(super_start_x, super_end_x)

        miny = min(super_start_y, super_end_y)
        maxy = max(super_start_y, super_end_y)

        x_val = np.concatenate([x_i_start, x_i_end])
        y_val = np.concatenate([y_i_start, y_i_end])

        idx_keep = np.where((x_val <= maxx) & (x_val >= minx))[0]
        idy_keep = np.where((y_val <= maxy) & (y_val >= miny))[0]

        idx_keep = np.intersect1d(idy_keep, idx_keep)

        x_val = x_val[idx_keep]
        y_val = y_val[idx_keep]

        super_mid_x = (super_start_x + super_end_x) / 2
        super_mid_y = (super_start_y + super_end_y) / 2
        from scipy.spatial import distance

        very_straight = False
        straight_level = 3
        noise = noise0
        x_super = np.array(
            [super_start_x, super_end_x, super_start_x, super_end_x, super_start_x, super_end_x, super_start_x,
             super_end_x, super_start_x + noise, super_end_x + noise,
             super_start_x - noise, super_end_x - noise])
        y_super = np.array(
            [super_start_y, super_end_y, super_start_y, super_end_y, super_start_y, super_end_y, super_start_y,
             super_end_y, super_start_y + noise, super_end_y + noise,
             super_start_y - noise, super_end_y - noise])

        if abs(minx - maxx) <= 1:
            very_straight = True
            straight_level = 10
            x_super = np.append(x_super, super_mid_x)
            y_super = np.append(y_super, super_mid_y)

        for i in range(straight_level):  # DO THE SAME FOR A MIDPOINT TOO
            y_super = np.concatenate([y_super, y_super])
            x_super = np.concatenate([x_super, x_super])

        list_selected_clus = list(zip(x_val, y_val))

        if len(list_selected_clus) >= 1 & very_straight:
            dist = distance.cdist([(super_mid_x, super_mid_y)], list_selected_clus, 'euclidean')
            k = min(2, len(list_selected_clus))
            midpoint_loc = dist[0].argsort()[:k]

            midpoint_xy = []
            for i in range(k):
                midpoint_xy.append(list_selected_clus[midpoint_loc[i]])

            noise = noise0 * 2

            if k == 1:
                mid_x = np.array([midpoint_xy[0][0], midpoint_xy[0][0] + noise, midpoint_xy[0][0] - noise])
                mid_y = np.array([midpoint_xy[0][1], midpoint_xy[0][1] + noise, midpoint_xy[0][1] - noise])
            if k == 2:
                mid_x = np.array(
                    [midpoint_xy[0][0], midpoint_xy[0][0] + noise, midpoint_xy[0][0] - noise, midpoint_xy[1][0],
                     midpoint_xy[1][0] + noise, midpoint_xy[1][0] - noise])
                mid_y = np.array(
                    [midpoint_xy[0][1], midpoint_xy[0][1] + noise, midpoint_xy[0][1] - noise, midpoint_xy[1][1],
                     midpoint_xy[1][1] + noise, midpoint_xy[1][1] - noise])
            for i in range(3):
                mid_x = np.concatenate([mid_x, mid_x])
                mid_y = np.concatenate([mid_y, mid_y])

            x_super = np.concatenate([x_super, mid_x])
            y_super = np.concatenate([y_super, mid_y])
        x_val = np.concatenate([x_val, x_super])
        y_val = np.concatenate([y_val, y_super])

        x_val = x_val.reshape((len(x_val), -1))
        y_val = y_val.reshape((len(y_val), -1))
        xp = np.linspace(minx, maxx, 500)

        gam50 = pg.LinearGAM(n_splines=4, spline_order=3, lam=10).gridsearch(x_val, y_val)
        XX = gam50.generate_X_grid(term=0, n=500)
        preds = gam50.predict(XX)

        idx_keep = np.where((xp <= (maxx)) & (xp >= (minx)))[0]
        ax2.plot(XX, preds, linewidth=linewidth, c='#323538')  # 3.5#1.5

        mean_temp = np.mean(xp[idx_keep])
        closest_val = xp[idx_keep][0]
        closest_loc = idx_keep[0]

        for i, xp_val in enumerate(xp[idx_keep]):
            if abs(xp_val - mean_temp) < abs(closest_val - mean_temp):
                closest_val = xp_val
                closest_loc = idx_keep[i]
        step = 1

        head_width = noise * arrow_width_scale_factor  # arrow_width needs to be adjusted sometimes # 40#30  ##0.2 #0.05 for mESC #0.00001 (#for 2MORGAN and others) # 0.5#1
        if direction_arrow == 1:
            ax2.arrow(xp[closest_loc], preds[closest_loc], xp[closest_loc + step] - xp[closest_loc],
                      preds[closest_loc + step] - preds[closest_loc], shape='full', lw=0, length_includes_head=False,
                      head_width=head_width, color='#323538')

        else:
            ax2.arrow(xp[closest_loc], preds[closest_loc], xp[closest_loc - step] - xp[closest_loc],
                      preds[closest_loc - step] - preds[closest_loc], shape='full', lw=0, length_includes_head=False,
                      head_width=head_width, color='#323538')

    c_edge = []
    width_edge = []
    pen_color = []
    super_cluster_label = []
    terminal_count_ = 0
    dot_size = []

    for i in sc_supercluster_nn:
        if i in final_super_terminal:
            print(f'{datetime.now()}\tSuper cluster {i} is a super terminal with sub_terminal cluster',
                  sub_terminal_clusters[terminal_count_])
            c_edge.append('yellow')  # ('yellow')
            if highlight_terminal_states == True:
                width_edge.append(2)
                super_cluster_label.append('TS' + str(sub_terminal_clusters[terminal_count_]))
            else:
                width_edge.append(0)
                super_cluster_label.append('')
            pen_color.append('black')
            # super_cluster_label.append('TS' + str(i))  # +'('+str(i)+')')
            # +'('+str(i)+')')
            dot_size.append(60)  # 60
            terminal_count_ = terminal_count_ + 1
        else:
            width_edge.append(0)
            c_edge.append('black')
            pen_color.append('red')
            super_cluster_label.append(str(' '))  # i or ' '
            dot_size.append(00)  # 20

    ax2.set_title(title_str)

    im2 = ax2.scatter(X_dimred[:, 0], X_dimred[:, 1], c=sc_pt_markov, cmap=cmap_pseudotime, s=0.01)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(im2, cax=cax, orientation='vertical',
               label='pseudotime')  # to avoid lines drawn on the colorbar we need an image instance without alpha variable
    ax2.scatter(X_dimred[:, 0], X_dimred[:, 1], c=sc_pt_markov, cmap=cmap_pseudotime, alpha=scatter_alpha,
                s=scatter_size, linewidths=marker_edgewidth * .1)
    count_ = 0
    loci = [sc_supercluster_nn[key] for key in sc_supercluster_nn]
    for i, c, w, pc, dsz, lab in zip(loci, c_edge, width_edge, pen_color, dot_size,
                                     super_cluster_label):  # sc_supercluster_nn
        ax2.scatter(X_dimred[i, 0], X_dimred[i, 1], c='black', s=dsz, edgecolors=c, linewidth=w)
        ax2.annotate(str(lab), xy=(X_dimred[i, 0], X_dimred[i, 1]))
        count_ = count_ + 1

    ax1.grid(False)
    ax2.grid(False)
    f.patch.set_visible(False)
    ax1.axis('off')
    ax2.axis('off')
    return f, ax1, ax2


def plot_viagraph_(ax=None, hammer_bundle=None, layout: ndarray = None, CSM: ndarray = None,
                   velocity_weight: float = None, pt: list = None, alpha_bundle=1, linewidth_bundle=2,
                   edge_color='darkblue', headwidth_bundle=0.1, arrow_frequency=0.05, show_direction=True,
                   ax_text: bool = True, title: str = '', plot_clusters: bool = False, cmap: str = 'viridis',
                   via_object=None, fontsize: float = 9, dpi: int = 300,tune_edges:bool = False,initial_bandwidth=0.05, decay=0.9, edgebundle_pruning=0.5):
    '''
    this plots the edgebundles on the via clustergraph level and also adds the relevant arrow directions based on the TI directionality

    :param ax: axis to plot on
    :param hammer_bundle: hammerbundle object with coordinates of all the edges to draw. self.hammer
    :param layout: coords of cluster nodes
    :param CSM: cosine similarity matrix. cosine similarity between the RNA velocity between neighbors and the change in gene expression between these neighbors. Only used when available
    :param velocity_weight: percentage weightage given to the RNA velocity based transition matrix
    :param pt: cluster-level pseudotime (or other intensity level of features at average-cluster level)
    :param alpha_bundle: alpha when drawing lines
    :param linewidth_bundle: linewidth of bundled lines
    :param edge_color:
    :param headwidth_bundle: headwidth of arrows used in bundled edges
    :param arrow_frequency: min dist between arrows (bundled edges otherwise have overcrowding of arrows)
    :param show_direction: bool default True. will draw arrows along the lines to indicate direction
    :param plot_clusters: bool default False. When this function is called on its own (and not from within draw_piechart_graph() then via_object must be provided
    :param ax_text: bool default True. Show labels of the clusters with the cluster population and PARC cluster label
    :param fontsize: float default 9 Font size of labels
    :return: fig, ax with bundled edges plotted
    '''
    return_fig_ax = False  # return only the ax
    if ax == None:
        fig, ax = plt.subplots(dpi=dpi)
        ax.set_facecolor('white')
        fig.patch.set_visible(False)
        return_fig_ax = True
    if (plot_clusters == True) and (via_object is None):
        print('Warning: please provide a via object in order to `plot` the clusters on the graph')
    if via_object is not None:
        if hammer_bundle is None: hammer_bundle = via_object.hammerbundle_cluster
        if layout is None: layout = via_object.graph_node_pos
        if CSM is None: CSM = via_object.CSM
        if velocity_weight is None: velocity_weight = via_object.velo_weight
        if pt is None: pt = via_object.scaled_hitting_times
    if tune_edges:
        print('make new edgebundle')
        hammer_bundle, layout = make_edgebundle_viagraph(via_object=via_object, layout=via_object.layout, decay=decay,
                                                         initial_bandwidth=initial_bandwidth,
                                                         edgebundle_pruning=edgebundle_pruning)  # hold the layout fixed. only change the edges
    x_ = [l[0] for l in layout]
    y_ = [l[1] for l in layout]
    # min_x, max_x = min(x_), max(x_)
    # min_y, max_y = min(y_), max(y_)
    delta_x = max(x_) - min(x_)

    delta_y = max(y_) - min(y_)

    layout = np.asarray(layout)
    # make a knn so we can find which clustergraph nodes the segments start and end at

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(layout)
    # get each segment. these are separated by nans.
    hbnp = hammer_bundle.to_numpy()
    splits = (np.isnan(hbnp[:, 0])).nonzero()[0]  # location of each nan values
    edgelist_segments = []
    start = 0
    segments = []
    arrow_coords = []
    for stop in splits:
        seg = hbnp[start:stop, :]
        segments.append(seg)
        start = stop

    n = 1  # every nth segment is plotted
    step = 1
    for seg in segments[::n]:
        do_arrow = True

        # seg_weight = max(0.3, math.log(1+seg[-1,2]))
        seg_weight = max(0.05, math.log(1 + seg[-1, 2]))

        # print('seg weight', seg_weight)
        seg = seg[:, 0:2].reshape(-1, 2)
        seg_p = seg[~np.isnan(seg)].reshape((-1, 2))

        start = neigh.kneighbors(seg_p[0, :].reshape(1, -1), return_distance=False)[0][0]
        end = neigh.kneighbors(seg_p[-1, :].reshape(1, -1), return_distance=False)[0][0]
        # print('start,end',[start, end])

        if ([start, end] in edgelist_segments) | ([end, start] in edgelist_segments):
            do_arrow = False
        edgelist_segments.append([start, end])

        direction_ = infer_direction_piegraph(start_node=start, end_node=end, CSM=CSM, velocity_weight=velocity_weight,
                                              pt=pt)

        direction = -1 if direction_ < 0 else 1

        ax.plot(seg_p[:, 0], seg_p[:, 1], linewidth=linewidth_bundle * seg_weight, alpha=alpha_bundle, color=edge_color)
        mid_point = math.floor(seg_p.shape[0] / 2)

        if len(arrow_coords) > 0:  # dont draw arrows in overlapping segments
            for v1 in arrow_coords:
                dist_ = dist_points(v1, v2=[seg_p[mid_point, 0], seg_p[mid_point, 1]])
                # print('dist between points', dist_)
                if dist_ < arrow_frequency * delta_x: do_arrow = False
                if dist_ < arrow_frequency * delta_y: do_arrow = False

        if (do_arrow == True) & (seg_p.shape[0] > 3):
            ax.arrow(seg_p[mid_point, 0], seg_p[mid_point, 1],
                     seg_p[mid_point + (direction * step), 0] - seg_p[mid_point, 0],
                     seg_p[mid_point + (direction * step), 1] - seg_p[mid_point, 1],
                     lw=0, length_includes_head=False, head_width=headwidth_bundle, color=edge_color, shape='full',
                     alpha=0.6, zorder=5)
            arrow_coords.append([seg_p[mid_point, 0], seg_p[mid_point, 1]])
        if plot_clusters == True:
            group_pop = np.ones([layout.shape[0], 1])

            if via_object is not None:
                for group_i in set(via_object.labels):
                    # n_groups = len(set(via_object.labels))
                    loc_i = np.where(via_object.labels == group_i)[0]
                    group_pop[group_i] = len(loc_i)
            gp_scaling = 1000 / max(group_pop)  # 500 / max(group_pop)
            group_pop_scale = group_pop * gp_scaling * 0.5

            c_edge, l_width = [], []
            if via_object is not None:
                terminal_clusters_placeholder = via_object.terminal_clusters
            else:
                terminal_clusters_placeholder = []
            for ei, pti in enumerate(pt):
                if ei in terminal_clusters_placeholder:
                    c_edge.append('red')
                    l_width.append(1.5)
                else:
                    c_edge.append('gray')
                    l_width.append(0.0)

            ax.scatter(layout[:, 0], layout[:, 1], s=group_pop_scale, c=pt, cmap=cmap,
                       edgecolors=c_edge,
                       alpha=1, zorder=3, linewidth=l_width)
            if ax_text:
                x_max_range = np.amax(layout[:, 0]) / 100
                y_max_range = np.amax(layout[:, 1]) / 100

                for ii in range(layout.shape[0]):
                    ax.text(layout[ii, 0] + max(x_max_range, y_max_range),
                            layout[ii, 1] + min(x_max_range, y_max_range),
                            'C' + str(ii) + 'pop' + str(int(group_pop[ii][0])),
                            color='black', zorder=4, fontsize=fontsize)
    ax.set_title(title)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.set_facecolor('white')
    if return_fig_ax == True:
        return fig, ax
    else:
        return ax


def _slow_sklearn_mds(via_graph: csr_matrix, X_pca: ndarray, t_diff_op: int = 1):
    '''

    :param via_graph: via_graph =via_object.csr_full_graph #single cell knn graph representation based on hnsw
    :param t_diff_op:
    :param X_pca ndarray adata_counts.obsm['X_pca'][:, 0:ncomps]
    :return: ndarray
    '''
    from sklearn.preprocessing import normalize

    via_graph.data = np.clip(via_graph.data, np.percentile(via_graph.data, 10), np.percentile(via_graph.data, 90))
    row_stoch = normalize(via_graph, norm='l1', axis=1)

    # note that the edge weights are affinities in via_graph

    from sklearn import manifold
    mds = manifold.MDS(
        n_components=2,
        max_iter=3000,
        eps=1e-9,
        random_state=0,
        dissimilarity="precomputed",
        n_jobs=2,
    )

    row_stoch = row_stoch ** t_diff_op  # level of diffusion

    temp = csr_matrix(X_pca)

    X_mds = row_stoch * temp  # matrix multiplication

    X_mds = squareform(pdist(X_mds.todense()))
    X_mds = mds.fit(X_mds).embedding_
    # X_mds = squareform(pdist(adata_counts.obsm['X_pca'][:, 0:ncomps+20])) #no diffusion makes is less streamlined and compact. more fuzzy
    return X_mds

def plot_piechart_only_viagraph(via_object, type_data='pt', gene_exp: list = [], cmap_piechart: str = 'rainbow', title='',
                           cmap: str = None, ax_text=True, dpi=150, headwidth_arrow=0.1, alpha_edge=0.4,
                           linewidth_edge=2, edge_color='darkblue', reference_labels=None, show_legend: bool = True,
                           pie_size_scale: float = 0.8, fontsize: float = 8, pt_visual_threshold: int = 99,
                           highlight_terminal_clusters: bool = True, tune_edges:bool = False,initial_bandwidth=0.05, decay=0.9, edgebundle_pruning=0.5):
    '''
    plot clustergraph level representation of the viagraph showing true-label composition (lhs) and pseudotime/gene expression (rhs)
    Returns matplotlib figure with two axes that plot the clustergraph using edge bundling
    left axis shows the clustergraph with each node colored by annotated ground truth membership.
    right axis shows the same clustergraph with each node colored by the pseudotime or gene expression

    :param via_object: is class VIA (the same function also exists as a method of the class and an external plotting function
    :param type_data: string  default 'pt' for pseudotime colored nodes. or 'gene'
    :param gene_exp: list of values (or column of dataframe) corresponding to feature or gene expression to be used to color nodes at CLUSTER level
    :param cmap_piechart: str cmap for piechart categories
    :param title: string
    :param cmap: default None. automatically chooses coolwarm for gene expression or viridis_r for pseudotime
    :param ax_text: Bool default= True. Annotates each node with cluster number and population of membership
    :param dpi: int default = 150
    :param headwidth_arrow: default = 0.1. width of arrowhead used to directed edges
    :param reference_labels: None or list. list of categorical (str) labels for cluster composition of the piecharts (LHS subplot) length = n_samples.
    :param pie_size_scale: float default=0.8 scaling factor of the piechart nodes
    :param pt_visual_threshold: int (percentage) default = 95 corresponding to rescaling the visual color scale by clipping outlier cluster pseudotimes
    :param highlight_terminal_clusters:bool = True (red border around terminal clusters)
    :param size_node_notpiechart: scaling factor for node size of the viagraph (not the piechart part)
    :param initial_bandwidth: (float = 0.05)  increasing bw increases merging of minor edges.  Only used when tune_edges = True
    :param decay: (decay = 0.9) increasing decay increases merging of minor edges . Only used when tune_edges = True
    :param edgebundle_pruning (float = 0.5). takes on values between 0-1. smaller value means more pruning away edges that can be visualised. Only used when tune_edges = True
    :return: f, ax, ax1
    '''

    f, ax = plt.subplots( dpi=dpi)

    node_pos = via_object.graph_node_pos

    node_pos = np.asarray(node_pos)
    if cmap is None: cmap = 'coolwarm' if type_data == 'gene' else 'viridis_r'

    if type_data == 'pt':
        pt = via_object.markov_hitting_times  # via_object.scaled_hitting_times
        threshold_high = np.percentile(pt, pt_visual_threshold)

        pt_subset = [x for x in pt if x < threshold_high]  # remove high outliers
        new_upper_pt = np.percentile(pt_subset, pt_visual_threshold)  # 'true' upper percentile after removing outliers

        pt = [x if x < new_upper_pt else new_upper_pt for x in pt]
        title_ax1 = "Pseudotime " + title

    if (type_data == 'gene') | (len(gene_exp) > 0):
        pt = gene_exp
        title_ax1 = title
    if reference_labels is None: reference_labels = via_object.true_label

    n_groups = len(set(via_object.labels))
    n_truegroups = len(set(reference_labels))
    group_pop = np.zeros([n_groups, 1])
    if type(reference_labels[0]) == int or type(reference_labels[0]) == float:
        sorted_col_ = sorted(list(set(reference_labels)))

        group_frac = pd.DataFrame(np.zeros([n_groups, n_truegroups]), columns=sorted_col_)
    else:
        sorted_col_ = list(set(reference_labels))
        sorted_col_.sort()

        group_frac = pd.DataFrame(np.zeros([n_groups, n_truegroups]),
                                  columns=sorted_col_)  # list(set(reference_labels))

    via_object.cluster_population_dict = {}

    set_labels = list(set(via_object.labels))
    set_labels.sort()

    for group_i in set_labels:
        loc_i = np.where(via_object.labels == group_i)[0]

        group_pop[group_i] = len(loc_i)  # np.sum(loc_i) / 1000 + 1
        via_object.cluster_population_dict[group_i] = len(loc_i)
        true_label_in_group_i = list(np.asarray(reference_labels)[loc_i])
        ll_temp = list(set(true_label_in_group_i))

        for ii in ll_temp:
            group_frac[ii][group_i] = true_label_in_group_i.count(ii)

    line_true = np.linspace(0, 1, n_truegroups)
    cmap_piechart_ = plt.get_cmap(cmap_piechart)
    color_true_list = [cmap_piechart_(color) for color in line_true]  # plt.cm.rainbow(color)

    sct = ax.scatter(node_pos[:, 0], node_pos[:, 1],
                     c='white', edgecolors='face', s=group_pop, cmap=cmap_piechart)

    bboxes = getbb(sct, ax)

    ax = plot_viagraph_(ax, via_object= via_object, pt=pt, headwidth_bundle=headwidth_arrow,
                        alpha_bundle=alpha_edge, linewidth_bundle=linewidth_edge, edge_color=edge_color,
                        tune_edges=tune_edges, initial_bandwidth=initial_bandwidth, decay=decay,
                        edgebundle_pruning=edgebundle_pruning)

    trans = ax.transData.transform
    bbox = ax.get_position().get_points()
    ax_x_min = bbox[0, 0]
    ax_x_max = bbox[1, 0]
    ax_y_min = bbox[0, 1]
    ax_y_max = bbox[1, 1]
    ax_len_x = ax_x_max - ax_x_min
    ax_len_y = ax_y_max - ax_y_min
    trans2 = ax.transAxes.inverted().transform
    pie_axs = []
    pie_size_ar = ((group_pop - np.min(group_pop)) / (np.max(group_pop) - np.min(group_pop)) + 0.5) / 10  # 10

    for node_i in range(n_groups):

        cluster_i_loc = np.where(np.asarray(via_object.labels) == node_i)[0]
        majority_true = via_object.func_mode(list(np.asarray(reference_labels)[cluster_i_loc]))

        pie_size = pie_size_ar[node_i][0] * pie_size_scale

        x1, y1 = trans(node_pos[node_i])  # data coordinates
        xa, ya = trans2((x1, y1))  # axis coordinates

        xa = ax_x_min + (xa - pie_size / 2) * ax_len_x
        ya = ax_y_min + (ya - pie_size / 2) * ax_len_y
        # clip, the fruchterman layout sometimes places below figure
        if ya < 0: ya = 0
        if xa < 0: xa = 0
        rect = [xa, ya, pie_size * ax_len_x, pie_size * ax_len_y]
        frac = np.asarray([ff for ff in group_frac.iloc[node_i].values])

        pie_axs.append(plt.axes(rect, frameon=False))
        pie_axs[node_i].pie(frac, wedgeprops={'linewidth': 0.0}, colors=color_true_list)
        pie_axs[node_i].set_xticks([])
        pie_axs[node_i].set_yticks([])
        pie_axs[node_i].set_aspect('equal')
        # pie_axs[node_i].text(0.5, 0.5, graph_node_label[node_i])
        if ax_text == True:
            pie_axs[node_i].text(0.5, 0.5, str(majority_true)+'_c'+str(node_i), fontsize=fontsize)
            #pie_axs[node_i].text(0.5, 0.5, 'c' + str(node_i), fontsize=fontsize)

    patches, texts = pie_axs[node_i].pie(frac, wedgeprops={'linewidth': 0.0}, colors=color_true_list)
    labels = list(set(reference_labels))
    labels.sort()
    if show_legend == True: ax.legend(patches, labels, loc='best', fontsize=6, frameon=False)

    if via_object.time_series == True:
        ti = 'Cluster Composition. K=' + str(via_object.knn) + '. ncomp = ' + str(via_object.ncomp) + 'knnseq_' + str(
            via_object.knn_sequential)
    elif via_object.do_spatial_knn == True:
        ti = 'Cluster Composition. K=' + str(via_object.knn) + '. ncomp = ' + str(via_object.ncomp) + 'SpatKnn_' + str(
            via_object.spatial_knn)
    else:
        ti = 'Cluster Composition. K=' + str(via_object.knn) + '. ncomp = ' + str(via_object.ncomp)
    ax.set_title(ti)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


    f.patch.set_visible(False)

    ax.axis('off')
    ax.set_facecolor('white')
    return f, ax

def plot_piechart_viagraph(via_object, type_data='pt', gene_exp: list = [], cmap_piechart: str = 'rainbow', title='',
                           cmap: str = None, ax_text=True, dpi=150, headwidth_arrow=0.1, alpha_edge=0.4,
                           linewidth_edge=2, edge_color='darkblue', reference_labels=None, show_legend: bool = True,
                           pie_size_scale: float = 0.8, fontsize: float = 8, pt_visual_threshold: int = 99,
                           highlight_terminal_clusters: bool = True, size_node_notpiechart: float = 1,tune_edges:bool = False,initial_bandwidth=0.05, decay=0.9, edgebundle_pruning=0.5):
    '''
    plot two subplots with a clustergraph level representation of the viagraph showing true-label composition (lhs) and pseudotime/gene expression (rhs)
    Returns matplotlib figure with two axes that plot the clustergraph using edge bundling
    left axis shows the clustergraph with each node colored by annotated ground truth membership.
    right axis shows the same clustergraph with each node colored by the pseudotime or gene expression

    :param via_object: is class VIA (the same function also exists as a method of the class and an external plotting function
    :param type_data: string  default 'pt' for pseudotime colored nodes. or 'gene'
    :param gene_exp: list of values (or column of dataframe) corresponding to feature or gene expression to be used to color nodes at CLUSTER level
    :param cmap_piechart: str cmap for piechart categories
    :param title: string
    :param cmap: default None. automatically chooses coolwarm for gene expression or viridis_r for pseudotime
    :param ax_text: Bool default= True. Annotates each node with cluster number and population of membership
    :param dpi: int default = 150
    :param headwidth_arrow: default = 0.1. width of arrowhead used to directed edges
    :param reference_labels: None or list. list of categorical (str) labels for cluster composition of the piecharts (LHS subplot) length = n_samples.
    :param pie_size_scale: float default=0.8 scaling factor of the piechart nodes
    :param pt_visual_threshold: int (percentage) default = 95 corresponding to rescaling the visual color scale by clipping outlier cluster pseudotimes
    :param highlight_terminal_clusters:bool = True (red border around terminal clusters)
    :param size_node_notpiechart: scaling factor for node size of the viagraph (not the piechart part)
    :param initial_bandwidth: (float = 0.05)  increasing bw increases merging of minor edges.  Only used when tune_edges = True
    :param decay: (decay = 0.9) increasing decay increases merging of minor edges . Only used when tune_edges = True
    :param edgebundle_pruning (float = 0.5). takes on values between 0-1. smaller value means more pruning away edges that can be visualised. Only used when tune_edges = True
    :return: f, ax, ax1
    '''

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    f, ((ax, ax1)) = plt.subplots(1, 2, sharey=True, dpi=dpi)

    node_pos = via_object.graph_node_pos

    node_pos = np.asarray(node_pos)
    if cmap is None: cmap = 'coolwarm' if type_data == 'gene' else 'viridis_r'

    if type_data == 'pt':
        pt = via_object.markov_hitting_times  # via_object.scaled_hitting_times
        threshold_high = np.percentile(pt, pt_visual_threshold)

        pt_subset = [x for x in pt if x < threshold_high]  # remove high outliers
        new_upper_pt = np.percentile(pt_subset, pt_visual_threshold)  # 'true' upper percentile after removing outliers

        pt = [x if x < new_upper_pt else new_upper_pt for x in pt]
        title_ax1 = "Pseudotime " + title

    if (type_data == 'gene') | (len(gene_exp) > 0):
        pt = gene_exp
        title_ax1 = title
    if reference_labels is None: reference_labels = via_object.true_label

    n_groups = len(set(via_object.labels))
    n_truegroups = len(set(reference_labels))
    group_pop = np.zeros([n_groups, 1])
    if type(reference_labels[0]) == int or type(reference_labels[0]) == float:
        sorted_col_ = sorted(list(set(reference_labels)))

        group_frac = pd.DataFrame(np.zeros([n_groups, n_truegroups]), columns=sorted_col_)
    else:
        sorted_col_ = list(set(reference_labels))
        sorted_col_.sort()

        group_frac = pd.DataFrame(np.zeros([n_groups, n_truegroups]),
                                  columns=sorted_col_)  # list(set(reference_labels))

    via_object.cluster_population_dict = {}

    set_labels = list(set(via_object.labels))
    set_labels.sort()

    for group_i in set_labels:
        loc_i = np.where(via_object.labels == group_i)[0]

        group_pop[group_i] = len(loc_i)  # np.sum(loc_i) / 1000 + 1
        via_object.cluster_population_dict[group_i] = len(loc_i)
        true_label_in_group_i = list(np.asarray(reference_labels)[loc_i])
        ll_temp = list(set(true_label_in_group_i))

        for ii in ll_temp:
            group_frac[ii][group_i] = true_label_in_group_i.count(ii)

    line_true = np.linspace(0, 1, n_truegroups)
    cmap_piechart_ = plt.get_cmap(cmap_piechart)
    color_true_list = [cmap_piechart_(color) for color in line_true]  # plt.cm.rainbow(color)

    sct = ax.scatter(node_pos[:, 0], node_pos[:, 1],
                     c='white', edgecolors='face', s=group_pop, cmap=cmap_piechart)

    bboxes = getbb(sct, ax)
    print('tune edges', tune_edges)
    '''
    ax = plot_viagraph_(ax, via_object.hammerbundle_cluster, layout=via_object.graph_node_pos, CSM=via_object.CSM,
                        velocity_weight=via_object.velo_weight, pt=pt, headwidth_bundle=headwidth_arrow,
                        alpha_bundle=alpha_edge, linewidth_bundle=linewidth_edge, edge_color=edge_color,tune_edges = tune_edges,initial_bandwidth=initial_bandwidth, decay=decay, edgebundle_pruning=edgebundle_pruning)
    '''
    ax = plot_viagraph_(ax, via_object= via_object, pt=pt, headwidth_bundle=headwidth_arrow,
                        alpha_bundle=alpha_edge, linewidth_bundle=linewidth_edge, edge_color=edge_color,
                        tune_edges=tune_edges, initial_bandwidth=initial_bandwidth, decay=decay,
                        edgebundle_pruning=edgebundle_pruning)

    trans = ax.transData.transform
    bbox = ax.get_position().get_points()
    ax_x_min = bbox[0, 0]
    ax_x_max = bbox[1, 0]
    ax_y_min = bbox[0, 1]
    ax_y_max = bbox[1, 1]
    ax_len_x = ax_x_max - ax_x_min
    ax_len_y = ax_y_max - ax_y_min
    trans2 = ax.transAxes.inverted().transform
    pie_axs = []
    pie_size_ar = ((group_pop - np.min(group_pop)) / (np.max(group_pop) - np.min(group_pop)) + 0.5) / 10  # 10

    for node_i in range(n_groups):

        cluster_i_loc = np.where(np.asarray(via_object.labels) == node_i)[0]
        majority_true = via_object.func_mode(list(np.asarray(reference_labels)[cluster_i_loc]))

        pie_size = pie_size_ar[node_i][0] * pie_size_scale

        x1, y1 = trans(node_pos[node_i])  # data coordinates
        xa, ya = trans2((x1, y1))  # axis coordinates

        xa = ax_x_min + (xa - pie_size / 2) * ax_len_x
        ya = ax_y_min + (ya - pie_size / 2) * ax_len_y
        # clip, the fruchterman layout sometimes places below figure
        if ya < 0: ya = 0
        if xa < 0: xa = 0
        rect = [xa, ya, pie_size * ax_len_x, pie_size * ax_len_y]
        frac = np.asarray([ff for ff in group_frac.iloc[node_i].values])

        pie_axs.append(plt.axes(rect, frameon=False))
        pie_axs[node_i].pie(frac, wedgeprops={'linewidth': 0.0}, colors=color_true_list)
        pie_axs[node_i].set_xticks([])
        pie_axs[node_i].set_yticks([])
        pie_axs[node_i].set_aspect('equal')
        # pie_axs[node_i].text(0.5, 0.5, graph_node_label[node_i])
        if ax_text == True:
            pie_axs[node_i].text(0.5, 0.5, str(majority_true)+'_c'+str(node_i), fontsize=fontsize)
            #pie_axs[node_i].text(0.5, 0.5, 'c' + str(node_i), fontsize=fontsize)

    patches, texts = pie_axs[node_i].pie(frac, wedgeprops={'linewidth': 0.0}, colors=color_true_list)
    labels = list(set(reference_labels))
    labels.sort()
    if show_legend == True: ax.legend(patches, labels, loc='best', fontsize=6, frameon=False)

    if via_object.time_series == True:
        ti = 'Cluster Composition. K=' + str(via_object.knn) + '. ncomp = ' + str(via_object.ncomp) + 'Temporalknn_' + str(
            via_object.knn_sequential)
        if via_object.do_spatial_knn == True:
            ti = 'Cluster Composition. K=' + str(via_object.knn) + '. ncomp = ' + str(
                via_object.ncomp) + 'SpatKnn_' + str(
                via_object.spatial_knn) + 'Temporalknn_' + str(
            via_object.knn_sequential)
    elif via_object.do_spatial_knn == True:
        ti = 'Cluster Composition. K=' + str(via_object.knn) + '. ncomp = ' + str(via_object.ncomp) + 'SpatKnn_' + str(
            via_object.spatial_knn)
    else:
        ti = 'Cluster Composition. K=' + str(via_object.knn) + '. ncomp = ' + str(via_object.ncomp)
    ax.set_title(ti)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    title_list = [title_ax1]
    for i, ax_i in enumerate([ax1]):
        # pt = via_object.markov_hitting_times if type_data == 'pt' else gene_exp

        c_edge, l_width = [], []
        for ei, pti in enumerate(pt):
            if ei in via_object.terminal_clusters:
                c_edge.append('red')
                if not highlight_terminal_clusters:
                    l_width.append(0)
                else:
                    l_width.append(1.5)
            else:
                c_edge.append('gray')
                l_width.append(0.0)

        gp_scaling = 1000 / max(group_pop)

        group_pop_scale = group_pop * gp_scaling * 0.5
        '''
        ax_i = plot_viagraph_(ax_i, via_object.hammerbundle_cluster, layout=via_object.graph_node_pos,
                              CSM=via_object.CSM, velocity_weight=via_object.velo_weight, pt=pt,
                              headwidth_bundle=headwidth_arrow, alpha_bundle=alpha_edge,
                              linewidth_bundle=linewidth_edge, edge_color=edge_color,tune_edges=tune_edges,initial_bandwidth=initial_bandwidth, decay=decay, edgebundle_pruning=edgebundle_pruning)
        '''
        ax_i = plot_viagraph_(ax_i, via_object=via_object, pt=pt, headwidth_bundle=headwidth_arrow,
                            alpha_bundle=alpha_edge, linewidth_bundle=linewidth_edge, edge_color=edge_color,
                            tune_edges=tune_edges, initial_bandwidth=initial_bandwidth, decay=decay,
                            edgebundle_pruning=edgebundle_pruning)

        im1 = ax_i.scatter(node_pos[:, 0], node_pos[:, 1], s=group_pop_scale * size_node_notpiechart, c=pt, cmap=cmap,
                           edgecolors=c_edge,
                           alpha=1, zorder=3, linewidth=l_width)
        if ax_text:
            x_max_range = np.amax(node_pos[:, 0]) / 100
            y_max_range = np.amax(node_pos[:, 1]) / 100

            for ii in range(node_pos.shape[0]):
                ax_i.text(node_pos[ii, 0] + max(x_max_range, y_max_range),
                          node_pos[ii, 1] + min(x_max_range, y_max_range),
                          'C' + str(ii) + 'pop' + str(int(group_pop[ii][0])),
                          color='black', zorder=4, fontsize=fontsize)
        ax_i.set_title(title_list[i])
        ax_i.grid(False)
        ax_i.set_xticks([])
        ax_i.set_yticks([])

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    if type_data == 'pt':
        f.colorbar(im1, cax=cax, orientation='vertical', label='pseudotime')
    else:
        f.colorbar(im1, cax=cax, orientation='vertical', label='Gene expression')
    f.patch.set_visible(False)

    ax1.axis('off')
    ax.axis('off')
    ax.set_facecolor('white')
    ax1.set_facecolor('white')
    return f, ax, ax1


def plot_clusters_spatial(spatial_coords, clusters=[], via_labels= [], title_sup='', fontsize_=6,color='green', s:int=5, alpha=0.5, xlim_max=None, ylim_max=None,xlim_min=None, ylim_min=None, reference_labels:list=[], reference_labels2:list = [],equal_axes_lim: bool = True):
    '''

    :param spatial_coords: ndarray of spatial coords ncellsx2 dims
    :param clusters: the clusters in via_object.labels which you want to plot (usually a subset of the total number of clusters)
    :param via_labels: via_object.labels (cluster level labels, list of n_cells length)
    :param title_sup: title of the overall figure
    :param fontsize_: fontsize for legend
    :param color: color of scatter points
    :param s: size of scatter points
    :param alpha: float alpha transparency of scatter (0 fully transporent, 1 is opaque)
    :param xlim_max: limits of axes
    :param ylim_max: limits of axes
    :param xlim_min: limits of axes
    :param ylim_min: limits of axes
    :param reference_labels:  optional list of single-cell labels (e.g. time, annotation). this will be used in the title of each subplot to note the majority cell (ref2) type for each cluster
    :param reference_labels2: optional list of single-cell labels (e.g. time, annotation). this will be used in the title of each subplot to note the majority cell (ref2) type for each cluster
    :return: fig, axs
    '''

    if xlim_max is None:  xlim_max = np.max(spatial_coords[:, 0])
    if ylim_max is None: ylim_max = np.max(spatial_coords[:, 1])

    if xlim_min is None:  xlim_min = np.min(spatial_coords[:, 0])
    if ylim_min is None: ylim_min = np.min(spatial_coords[:, 1])
    n_clusters = len(clusters)
    col_init = min(4,n_clusters)
    fig_nrows, mod = divmod(n_clusters, col_init)
    if mod == 0: fig_nrows = fig_nrows
    if mod != 0: fig_nrows += 1

    fig_ncols = col_init
    fig, axs = plt.subplots(fig_nrows, fig_ncols)
    fig.patch.set_visible(False)
    i_gene = 0  # counter for number of genes
    i_terminal = 0  # counter for terminal cluster
    # for i in range(n_terminal_states): #[0]

    for r in range(fig_nrows):
        for c in range(fig_ncols):
            if (i_gene < n_clusters):
                cluster_i = clusters[i_gene]
                df = pd.DataFrame(spatial_coords, columns=['x', 'y'])
                df['v0'] = via_labels
                if len(reference_labels)>0:
                    df['reference'] = reference_labels
                if len(reference_labels2) > 0:
                    df['reference2'] = reference_labels2


                df = df[df.v0 == cluster_i]
                df_coords_majref = pd.DataFrame(spatial_coords, columns=['x', 'y'])


                majority_reference = ''
                majority_reference2 = ''


                if len(reference_labels)>0:
                    reference_labels_sub = list(df['reference'])

                    majority_reference = func_mode(reference_labels_sub)
                    if len(reference_labels2) > 0:
                        reference2_labels_sub = list(df['reference2'])
                        majority_reference2 = func_mode(reference2_labels_sub)
                    df_coords_majref['reference'] = reference_labels
                    df_coords_majref = df_coords_majref[df_coords_majref.reference == majority_reference]
                    df_coords_majref = df_coords_majref[['x', 'y']].values
                else: df_coords_majref = df_coords_majref.values
                emb = df[['x', 'y']].values

                if not equal_axes_lim:
                    xlim_max = np.max(emb[:, 0]) *1.2
                    ylim_max = np.max(emb[:, 1]) *1.2

                    xlim_min = np.min(emb[:, 0]) *1.2
                    ylim_min = np.min(emb[:, 1]) *1.2
                #color = cmap_[i_gene]
                if fig_nrows > 1:

                    axs[r, c].scatter(df_coords_majref[:, 0], df_coords_majref[:, 1], c='gray', s=1, alpha=0.2)
                    axs[r, c].scatter(emb[:, 0], emb[:, 1], c=color, s=s, alpha=alpha )
                    if len(reference_labels)>0: axs[r,c].set_title('c:' + str(cluster_i)+'_'+str(majority_reference)+'_'+str(majority_reference2))
                    else:                    axs[r,c].set_title('c:' + str(cluster_i))#

                    axs[r,c].set_xlim([xlim_min, xlim_max]) #axis limits
                    axs[r, c].set_ylim([ylim_min, ylim_max])  # axis limits
                    # Set tick font size
                    for label in (axs[r, c].get_xticklabels() + axs[r, c].get_yticklabels()):
                        label.set_fontsize(fontsize_ - 1)
                    if i_gene == n_clusters - 1:
                        axs[r, c].legend(frameon=False, fontsize=fontsize_)

                    axs[r, c].spines['top'].set_visible(False)
                    axs[r, c].spines['right'].set_visible(False)

                    axs[r, c].grid(False)
                elif fig_ncols ==1:
                    axs.scatter(df_coords_majref[:, 0], df_coords_majref[:, 1], c='gray', s=s, alpha=alpha*0.5)
                    axs.scatter(emb[:, 0], emb[:, 1], c=color, s=s,alpha=alpha )
                    if len(reference_labels)>0: axs.set_title('c:' + str(cluster_i)+'_'+str(majority_reference)+'_'+str(majority_reference2))
                    else:                    axs.set_title('c:' + str(cluster_i))#

                    axs.set_xlim([xlim_min, xlim_max])  # axis limits
                    axs.set_ylim([ylim_min, ylim_max])  # axis limits
                    # Set tick font size
                    for label in (axs.get_xticklabels() + axs.get_yticklabels()):
                        label.set_fontsize(fontsize_ - 1)
                    if i_gene == n_clusters - 1:
                        axs.legend(frameon=False, fontsize=fontsize_)

                    axs.spines['top'].set_visible(False)
                    axs.spines['right'].set_visible(False)
                    axs.grid(False)
                else:
                    print('df_coords_majref  shape', df_coords_majref.shape)
                    axs[c].scatter(df_coords_majref[:, 0], df_coords_majref[:, 1], c='gray', s=s, alpha=0.1)
                    axs[c].scatter(emb[:, 0], emb[:, 1], c=color, s=s,alpha=alpha )
                    if len(reference_labels)>0: axs[c].set_title('c:' + str(cluster_i)+'_'+str(majority_reference)+'_'+str(majority_reference2))
                    else:                    axs[c].set_title('c:' + str(cluster_i))#
                    axs[ c].set_xlim([xlim_min, xlim_max])  # axis limits
                    axs[ c].set_ylim([ylim_min, ylim_max])  # axis limits
                    # Set tick font size
                    for label in (axs[c].get_xticklabels() + axs[c].get_yticklabels()):
                        label.set_fontsize(fontsize_ - 1)
                    if i_gene == n_clusters - 1:
                        axs[c].legend(frameon=False, fontsize=fontsize_)

                    axs[c].spines['top'].set_visible(False)
                    axs[c].spines['right'].set_visible(False)
                    axs[c].grid(False)
                i_gene += 1
            else:
                if fig_nrows > 1:
                    axs[r, c].axis('off')
                    axs[r, c].grid(False)
                else:
                    axs[c].axis('off')
                    axs[c].grid(False)
    fig.suptitle(title_sup, fontsize=8)
    return fig, axs

def make_dict_of_clusters_for_each_celltype(via_labels:list = [], true_label:list = [], verbose:bool = False):
    '''

    :param via_labels: usually set to via_object.labels. list of length n_cells of cluster membership
    :param true_label: cell type labels (list of length n_cells)
    :return:
    '''

    df_mode = pd.DataFrame()
    df_mode['cluster'] = via_labels
    df_mode['class_str'] = true_label
    majority_cluster_population_dict = df_mode.groupby(['cluster'])['class_str'].agg(
        lambda x: pd.Series.mode(x)[0])
    majority_cluster_population_dict = majority_cluster_population_dict.to_dict()
    print(f'dict cluster to majority pop: {majority_cluster_population_dict}')
    class_to_cluster_dict = collect_dictionary(majority_cluster_population_dict)
    print('list of clusters for each majority', class_to_cluster_dict)
    return class_to_cluster_dict
def plot_all_spatial_clusters(spatial_coords, true_label, via_labels, save_to:str = '', color_dict:dict = {}, cmap:str = 'rainbow', alpha = 0.4, s=5, verbose:bool=False, reference_labels:list=[],reference_labels2:list=[]):
    '''
    :param spatial_coords: ndarray of x,y coords of tissue location of cells (ncells x2)
    :param true_label: categorial labels (list of length n_cells)
    :param via_labels: cluster membership labels (list of length n_cells)
    :param save_to:
    :param color_dict: optional dict with keys corresponding to true_label type. e.g. {true_label_celltype1: 'green',true_label_celltype2: 'red'}
    :param cmap: string default = rainbow
    :param reference_labels:  optional list of single-cell labels (e.g. time, annotation). Used to selectively provide a grey background to cells not in the cluster being inspected. If you have multipe time points, then set reference_labels to the time_points. All cells in the most prevalent timepoint seen in the cluster of interest will be plotted as a background
    :param reference_labels2: optional list of single-cell labels (e.g. time, annotation). this will be used in the title of each subplot to note the majority cell (ref2) type for each cluster
    :return: list lists of [[fig1, axs_set1], [fig2, axs_set2],...]
    '''
    clusters_for_each_celltype_dict=make_dict_of_clusters_for_each_celltype(via_labels, true_label)
    if verbose: print(clusters_for_each_celltype_dict)
    keys = list(sorted(list(clusters_for_each_celltype_dict.keys())))
    print('keys', keys)
    potential_majority_celltype_keys = list(sorted(list(set(true_label))))
    list_of_figs=[]
    if len(color_dict) ==0:

        set_labels = list(set(true_label))
        set_labels.sort(reverse=False)  # True)
        palette = plt.get_cmap(cmap, len(set_labels))
        cmap_ = palette(range(len(set_labels)))
        for index, value in enumerate(set_labels):
            color_dict[value] = cmap_[index]
    for i, keyi in enumerate(potential_majority_celltype_keys):
        if keyi in keys:
            color = color_dict[keyi]  # cmap_[i]

            clusters_list = clusters_for_each_celltype_dict[keyi]
            f, axs = plot_clusters_spatial(spatial_coords, clusters=clusters_list, via_labels=via_labels,
                                               title_sup=keyi, color=color, s=s,
                                               alpha=alpha, reference_labels2=reference_labels2, reference_labels=reference_labels)
            list_of_figs.append([f, axs])
            fig_nrows, mod = divmod(len(clusters_list), 4)
            if mod == 0: fig_nrows = fig_nrows
            if mod != 0: fig_nrows += 1
            f.set_size_inches(10, 2 * fig_nrows)
            #f.savefig(                WORK_PATH + 'Viagraphs/Bregma' + str(int(bregma)) + 'Spatknn' + str(spatial_knn) + 'cluster_' + keyi[                                                                                                                0:4] + '.png')
            f.savefig(save_to + 'cluster_' + keyi[0:4] + '.png')
        else:
            print('No cluster has a majority population of ', keyi)

    return list_of_figs

def animate_atlas_old(hammerbundle_dict=None, via_object=None, linewidth_bundle=2, frame_interval: int = 10,
                  n_milestones: int = None, facecolor: str = 'white', cmap: str = 'plasma_r', extra_title_text='',
                  size_scatter: int = 1, alpha_scatter: float = 0.2,
                  saveto='/home/user/Trajectory/Datasets/animation_default.gif', time_series_labels: list = None, lineage_pathway = [],
                  sc_labels_numeric: list = None, t_diff_factor:float=0.25, show_sc_embedding:bool=False, sc_emb=None, sc_size_scatter:float=10, sc_alpha_scatter:float=0.2, n_intervals:int = 50):
    '''
    :param ax: axis to plot on
    :param hammer_bundle: hammerbundle object with coordinates of all the edges to draw
    :param layout: coords of cluster nodes and optionally also contains the numeric value associated with each cluster (such as time-stamp) layout[['x','y','numeric label']] sc/cluster/milestone level
    :param CSM: cosine similarity matrix. cosine similarity between the RNA velocity between neighbors and the change in gene expression between these neighbors. Only used when available
    :param velocity_weight: percentage weightage given to the RNA velocity based transition matrix
    :param pt: cluster-level pseudotime
    :param alpha_bundle: alpha when drawing lines
    :param linewidth_bundle: linewidth of bundled lines
    :param edge_color:
    :param frame_interval: smaller number, faster refresh and video
    :param facecolor: default = white
    :param headwidth_bundle: headwidth of arrows used in bundled edges
    :param arrow_frequency: min dist between arrows (bundled edges otherwise have overcrowding of arrows)
    :param show_direction: True will draw arrows along the lines to indicate direction
    :param milestone_edges: pandas DataFrame milestone_edges[['source','target']]
    :param t_diff_factor scaling the average the time intervals (0.25 means that for each frame, the time is progressed by 0.25* mean_time_differernce_between adjacent times (only used when sc_labels_numeric are directly passed instead of using pseudotime)
    :param show_sc_embedding: plot the single cell embedding under the edges
    :param sc_emb numpy array of single cell embedding (ncells x 2)
    :param sc_alpha_scatter, Alpha transparency value of points of single cells (1 is opaque, 0 is fully transparent)
    :param sc_size_scatter. size of scatter points of single cells
    :param time_series_labels, should be a single-cell level list (n_cells) of numerical values that form a discrete set. I.e. not continuous like pseudotime,
    :return: axis with bundled edges plotted
    '''
    import tqdm
    if show_sc_embedding:
        if sc_emb is None:
            sc_emb= via_object.embedding
            if sc_emb is None:
                print('please provide a single cell embedding as an array')
                return
    if hammerbundle_dict is None:
        if via_object is None:
            print(
                f'{datetime.now()}\tERROR: Hammerbundle_dict needs to be provided either through via_object or by running make_edgebundle_milestone()')
        else:
            hammerbundle_dict = via_object.hammerbundle_milestone_dict
            if hammerbundle_dict is None:
                if n_milestones is None: n_milestones = min(via_object.nsamples, 150)
                if sc_labels_numeric is None:
                    if via_object.time_series_labels is not None:
                        sc_labels_numeric = via_object.time_series_labels
                    else:
                        sc_labels_numeric = via_object.single_cell_pt_markov

                hammerbundle_dict = make_edgebundle_milestone(via_object=via_object,
                                                              embedding=via_object.embedding,
                                                              sc_graph=via_object.ig_full_graph,
                                                              n_milestones=n_milestones,
                                                              sc_labels_numeric=sc_labels_numeric,
                                                              initial_bandwidth=0.02, decay=0.7, weighted=True)
            hammer_bundle = hammerbundle_dict['hammerbundle']
            layout = hammerbundle_dict['milestone_embedding'][['x', 'y']].values
            milestone_edges = hammerbundle_dict['edges']
            milestone_numeric_values = hammerbundle_dict['milestone_embedding']['numeric label']
            milestone_pt = hammerbundle_dict['milestone_embedding']['pt']  # used when plotting arrows

    else:
        hammer_bundle = hammerbundle_dict['hammerbundle']
        layout = hammerbundle_dict['milestone_embedding'][['x', 'y']].values
        milestone_edges = hammerbundle_dict['edges']
        milestone_numeric_values = hammerbundle_dict['milestone_embedding']['numeric label']
        milestone_pt = hammerbundle_dict['milestone_embedding']['pt']  # used when plotting arrows
    fig, ax = plt.subplots(facecolor=facecolor, figsize=(15, 12))
    time_thresh = min(milestone_numeric_values)
    # ax.set_facecolor(facecolor)
    ax.grid(False)
    x_ = [l[0] for l in layout]
    y_ = [l[1] for l in layout]
    # min_x, max_x = min(x_), max(x_)
    # min_y, max_y = min(y_), max(y_)
    delta_x = max(x_) - min(x_)

    delta_y = max(y_) - min(y_)

    layout = np.asarray(layout)
    # make a knn so we can find which clustergraph nodes the segments start and end at

    # get each segment. these are separated by nans.
    hbnp = hammer_bundle.to_numpy()
    splits = (np.isnan(hbnp[:, 0])).nonzero()[0]  # location of each nan values
    edgelist_segments = []
    start = 0
    segments = []
    arrow_coords = []
    seg_len = []  # length of a segment
    for stop in splits:
        seg = hbnp[start:stop, :]
        segments.append(seg)
        seg_len.append(seg.shape[0])
        start = stop

    min_seg_length = min(seg_len)
    max_seg_length = max(seg_len)
    # mean_seg_length = sum(seg_len)/len(seg_len)
    seg_len = np.asarray(seg_len)
    seg_len = np.clip(seg_len, a_min=np.percentile(seg_len, 10),
                      a_max=np.percentile(seg_len, 90))
    step = 1  # every step'th segment is plotted

    cmap = plt.get_cmap(cmap)
    if milestone_numeric_values is not None:
        max_numerical_value = max(milestone_numeric_values)
        min_numerical_value = min(milestone_numeric_values)

    seg_count = 0

    # print('numeric vals', milestone_numeric_values)
    loc_time_thresh = np.where(np.asarray(milestone_numeric_values) <= time_thresh)[0].tolist()
    i_sorted_numeric_values = np.argsort(milestone_numeric_values)

    ee = int(len(milestone_numeric_values) / n_intervals)
    print('ee',ee)

    loc_time_thresh = i_sorted_numeric_values[0:ee]
    for ll in loc_time_thresh:
        print('sorted numeric milestone',milestone_numeric_values[ll])
    # print('loc time thres', loc_time_thresh)
    milestone_edges['source_thresh'] = milestone_edges['source'].isin(
        loc_time_thresh)  # apply(lambda x: any([k in x for k in loc_time_thresh]))

    # print(milestone_edges[0:10])
    idx = milestone_edges.index[milestone_edges['source_thresh']].tolist()
    # print('loc time thres', time_thresh, loc_time_thresh)
    for i in idx:
        seg = segments[i]
        source_milestone = milestone_edges['source'].values[i]
        target_milestone = milestone_edges['target'].values[i]

        # seg_weight = max(0.3, math.log(1+seg[-1,2])) seg[-1,2] column index 2 has the weight information

        seg_weight = seg[-1, 2] * seg_len[i] / (
                    max_seg_length - min_seg_length)  ##seg.shape[0] / (max_seg_length - min_seg_length)

        # cant' quite decide yet if sigmoid is desirable
        # seg_weight=sigmoid_scalar(seg.shape[0] / (max_seg_length - min_seg_length), scale=5, shift=mean_seg_length / (max_seg_length - min_seg_length))
        alpha_bundle = max(seg_weight, 0.1)  # max(0.1, math.log(1 + seg[-1, 2]))
        if alpha_bundle > 1: alpha_bundle = 1

        if milestone_numeric_values is not None:
            source_milestone_numerical_value = milestone_numeric_values[source_milestone]
            target_milestone_numerical_value = milestone_numeric_values[target_milestone]
            # print('source milestone', source_milestone_numerical_value)
            # print('target milestone', target_milestone_numerical_value)
            rgba_milestone_value = min(source_milestone_numerical_value, target_milestone_numerical_value)
            rgba = cmap((rgba_milestone_value - min_numerical_value) / (max_numerical_value - min_numerical_value))
        else:
            rgba = cmap(min(seg_weight, 0.95))  # cmap(seg.shape[0]/(max_seg_length-min_seg_length))
        # if seg_weight>0.05: seg_weight=0.1
        if seg_count % 10000 == 0: print('seg weight', seg_weight)
        seg = seg[:, 0:2].reshape(-1, 2)
        seg_p = seg[~np.isnan(seg)].reshape((-1, 2))

        ax.plot(seg_p[:, 0], seg_p[:, 1], linewidth=linewidth_bundle * seg_weight, alpha=alpha_bundle,
                color=rgba)  # edge_color )
        seg_count += 1
    milestone_numeric_values_rgba = []

    if len(lineage_pathway) > 0:
                milestone_lin_values = hammerbundle_dict['milestone_embedding'][
                    'sc_lineage_probability_' + str(lineage_pathway[0])]
                p1_sc_bp = np.nan_to_num(via_object.single_cell_bp, nan=0.0, posinf=0.0,
                                         neginf=0.0)  # single cell lineage probabilities sc pb
                # row normalize
                row_sums = p1_sc_bp.sum(axis=1)
                p1_sc_bp = p1_sc_bp / row_sums[:,
                                      np.newaxis]  # make rowsums a column vector where i'th entry is sum of i'th row in p1-sc-bp
                ts_cluster_number = lineage_pathway[0]
                ts_array_original = np.asarray(via_object.terminal_clusters)
                loc_ts_current = np.where(ts_array_original == ts_cluster_number)[0][0]
                print(
                    f'location of {lineage_pathway[0]} is at {np.where(ts_array_original == ts_cluster_number)[0]} and {loc_ts_current}')
                p1_sc_bp = p1_sc_bp[:, loc_ts_current]
                rgba_lineage_sc = []
                rgba_lineage_milestone = []
                for i in p1_sc_bp:
                    rgba_lineage_sc_ = cmap((i - min(p1_sc_bp)) / (max(p1_sc_bp) - min(p1_sc_bp)))
                    rgba_lineage_sc.append(rgba_lineage_sc_)
                for i in milestone_lin_values:
                    rgba_lineage_milestone_ = cmap((i - min(milestone_lin_values)) / (max(milestone_lin_values) - min(milestone_lin_values)))
                    rgba_lineage_milestone.append(rgba_lineage_milestone_)
    print('here1 in animate()')
    if milestone_numeric_values is not None:
        for i in milestone_numeric_values:
            rgba_ = cmap((i - min_numerical_value) / (max_numerical_value - min_numerical_value))
            milestone_numeric_values_rgba.append(rgba_)

        ax.scatter(layout[loc_time_thresh, 0], layout[loc_time_thresh, 1], s=size_scatter,
                   c=np.asarray(milestone_numeric_values_rgba)[loc_time_thresh], alpha=alpha_scatter)
        # if we dont plot all the points, then the size of axis changes and the location of the graph moves/changes as more points are added
        ax.scatter(layout[:, 0], layout[:, 1], s=size_scatter,
                   c=np.asarray(milestone_numeric_values_rgba), alpha=0)
        if show_sc_embedding:
            if len(lineage_pathway)>0:     ax.scatter(sc_emb[:, 0], sc_emb[:, 1], s=sc_size_scatter,
                       c=p1_sc_bp, alpha=sc_alpha_scatter, cmap=cmap)
            ax.scatter(sc_emb[:, 0], sc_emb[:, 1], s=size_scatter,
                       c='blue', alpha=0)
    else:
        ax.scatter(layout[:, 0], layout[:, 1], s=size_scatter, c='red', alpha=alpha_scatter)
    print('here2 in animate()')
    ax.set_facecolor(facecolor)
    ax.axis('off')
    time = datetime.now()
    time = time.strftime("%H:%M")
    title_ = 'n_milestones = ' + str(int(layout.shape[0])) + ' time: ' + time + ' ' + extra_title_text
    ax.set_title(label=title_, color='black')
    print(f"{datetime.now()}\tFinished plotting edge bundle")
    if time_series_labels is not None: #over-ride via_object's saved time_series_labels and/or pseudotime
        time_series_set_order = list(sorted(list(set(time_series_labels))))
        t_diff_mean = t_diff_factor * np.mean(
            np.array([int(abs(y - x)) for x, y in zip(time_series_set_order[:-1], time_series_set_order[1:])]))
        print('t_diff_mean:', t_diff_mean)
        cycles = (max_numerical_value - min_numerical_value) / t_diff_mean
        print('number cycles', cycles)
    else:
        if via_object is not None:
            time_series_labels = via_object.time_series_labels
            if time_series_labels is None:
                time_series_labels = via_object.single_cell_pt_markov
            time_series_labels_int = [int(i * 10) for i in time_series_labels] #to be able to get "mean t_diff" if time_series_labels are continuous rather than a set of discrete values
            time_series_set_order = list(sorted(list(set(time_series_labels_int))))
            print('times series order set using pseudotime', time_series_set_order)

            t_diff_mean = t_diff_factor * np.mean(np.array([int(abs(y - x)) for x, y in zip(time_series_set_order[:-1],
                                                                                   time_series_set_order[
                                                                                   1:])])) / 10  # divide by 10 because we multiplied the single_cell_pt_markov by 10


            cycles = (max_numerical_value - min_numerical_value) / (t_diff_mean)
            print('number cycles if no time_series labels given', cycles)



    min_time_series_labels = min(time_series_labels)
    max_time_series_labels = max(time_series_labels)
    sc_rgba = []
    for i in time_series_labels:
        sc_rgba_ = cmap((i - min_time_series_labels) / (max_time_series_labels - min_time_series_labels))
        sc_rgba.append(sc_rgba_)

    if show_sc_embedding:
        i_sorted_sc_time = np.argsort(time_series_labels)

    def update_edgebundle(frame_no):

        print('inside update', frame_no, 'out of', int(cycles), 'cycles')
        if len(time_series_labels) > 0:

            time_thresh = min_numerical_value + frame_no % (cycles + 1) * t_diff_mean

            print('time thresh', time_thresh)

        else:
            #n_intervals = 10
            time_thresh = min_numerical_value + (frame_no % n_intervals) * (
                        max_numerical_value - min_numerical_value) / n_intervals
        #time-based loc_time_thresh
        loc_time_thresh = np.where((np.asarray(milestone_numeric_values) <= time_thresh) & (                    np.asarray(milestone_numeric_values) > time_thresh - t_diff_mean))[0].tolist()





        sc_loc_time_thresh = np.where((np.asarray(time_series_labels) <= time_thresh) & (                    np.asarray(time_series_labels) > time_thresh - t_diff_mean))[0].tolist()

        milestone_edges['source_thresh'] = milestone_edges['source'].isin(
            loc_time_thresh)  # apply(lambda x: any([k in x for k in loc_time_thresh]))


        idx = milestone_edges.index[milestone_edges['source_thresh']].tolist()
        print('len of number of edges in this cycle', len(idx), 'for REM=', rem)
        for i in idx:
            seg = segments[i]
            source_milestone = milestone_edges['source'].values[i]

            # seg_weight = max(0.3, math.log(1+seg[-1,2])) seg[-1,2] column index 2 has the weight information

            seg_weight = seg[-1, 2] * seg_len[i] / (
                    max_seg_length - min_seg_length)  ##seg.shape[0] / (max_seg_length - min_seg_length)

            # cant' quite decide yet if sigmoid is desirable
            # seg_weight=sigmoid_scalar(seg.shape[0] / (max_seg_length - min_seg_length), scale=5, shift=mean_seg_length / (max_seg_length - min_seg_length))
            alpha_bundle = max(seg_weight, 0.1)  # max(0.1, math.log(1 + seg[-1, 2]))
            if alpha_bundle > 1: alpha_bundle = 1

            if milestone_numeric_values is not None:
                source_milestone_numerical_value = milestone_numeric_values[source_milestone]
                if len(lineage_pathway)==0:
                    rgba = cmap((source_milestone_numerical_value - min_numerical_value) / (                        max_numerical_value - min_numerical_value))
                else:
                    rgba = list(rgba_lineage_milestone[source_milestone])
                    rgba[3] = milestone_lin_values[source_milestone]
                    rgba = tuple(rgba)
            else:
                rgba = cmap(min(seg_weight, 0.95))  # cmap(seg.shape[0]/(max_seg_length-min_seg_length))
            # if seg_weight>0.05: seg_weight=0.1

            seg = seg[:, 0:2].reshape(-1, 2)
            seg_p = seg[~np.isnan(seg)].reshape((-1, 2))

            if len(lineage_pathway)>0:
                ax.plot(seg_p[:, 0], seg_p[:, 1], linewidth=linewidth_bundle * seg_weight,                     color=rgba)  # edge_color ) #alpha=alpha_bundle,
            else: ax.plot(seg_p[:, 0], seg_p[:, 1], linewidth=linewidth_bundle * seg_weight,                     color=rgba,alpha=alpha_bundle)

        milestone_numeric_values_rgba = []

        if milestone_numeric_values is not None:
            for i in milestone_numeric_values:
                rgba_ = cmap((i - min_numerical_value) / (max_numerical_value - min_numerical_value))
                milestone_numeric_values_rgba.append(rgba_)


            if time_thresh > 1.1 * max_numerical_value:
                ax.clear()
            else:
                ax.scatter(layout[:, 0], layout[:, 1], s=size_scatter,
                           c=np.asarray(milestone_numeric_values_rgba), alpha=0)
                ax.scatter(layout[loc_time_thresh, 0], layout[loc_time_thresh, 1], s=size_scatter,
                           c=np.asarray(milestone_numeric_values_rgba)[loc_time_thresh], alpha=alpha_scatter)

                if show_sc_embedding:
                    if len(lineage_pathway)>0:
                        ax.scatter(sc_emb[sc_loc_time_thresh, 0], sc_emb[sc_loc_time_thresh, 1], s=sc_size_scatter,
                               c=np.asarray(rgba_lineage_sc)[sc_loc_time_thresh], alpha=p1_sc_bp[sc_loc_time_thresh])

                    else: ax.scatter(sc_emb[sc_loc_time_thresh, 0], sc_emb[sc_loc_time_thresh, 1], s=sc_size_scatter,
                               c=np.asarray(sc_rgba)[sc_loc_time_thresh], alpha=sc_alpha_scatter)
        else:
            ax.scatter(layout[:, 0], layout[:, 1], s=size_scatter, c='red', alpha=alpha_scatter)
        #pbar.update()

    frame_no = int(cycles)*2
    animation = FuncAnimation(fig, update_edgebundle, frames=frame_no, interval=frame_interval, repeat=False)  # 100
    # pbar = tqdm.tqdm(total=frame_no)
    # pbar.close()
    print('complete animate')

    animation.save(saveto, writer='imagemagick')  # , fps=30)
    print('saved animation')
    plt.show()
    return
