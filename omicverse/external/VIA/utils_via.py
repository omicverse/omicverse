# based on : https://github.com/theislab/scvelo/blob/1805ab4a72d3f34496f0ef246500a159f619d3a2/scvelo/plotting/velocity_embedding_grid.py#L27
import numpy as np
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
from scipy.stats import norm as normal
from typing import Union
from scipy.spatial import distance
from scipy.sparse import csr_matrix, csgraph, find
import math
import pandas as pd
import numpy as np
from numpy import ndarray
from scipy.sparse import issparse, spmatrix

import matplotlib
import igraph as ig
import matplotlib.pyplot as plt
from matplotlib.path import get_path_collection_extents

from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import normalize
import random
from collections import Counter

import scipy
import pygam as pg
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
# import utils_sampling
from matplotlib.animation import FuncAnimation, writers

from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from collections import defaultdict
from tqdm.auto import tqdm

def collect_dictionary(obj):
    #obj is the dict to be inverted
  inv_obj = defaultdict(list)
  for key, value in obj.items():
    inv_obj[value].append(key)
  return dict(inv_obj)

def DEMaP(data, embedding, knn=30, subsample_idx=None):
    # https://github.com/scottgigante/DEMaP/blob/master/demap/demap.py
    # geodesic_dist = geodesic_distance(data, knn=knn)
    if subsample_idx is not None:
        geodesic_dist = geodesic_dist[subsample_idx, :][:, subsample_idx]
    geodesic_dist = squareform(geodesic_dist)
    embedded_dist = pdist(embedding)
    return spearmanr(geodesic_dist, embedded_dist).correlation


def func_mode(ll):
    # return MODE of list ll
    # If multiple items are maximal, the function returns the first one encountered.
    return max(set(ll), key=ll.count)

def compute_driver_genes(via_object, gene_exp:pd.DataFrame, lineage:int, clusters:list=None, conf_int:float=0.95, q = 0.05):
    '''
    Compute driver genes of each terminal cell fates using Pearson correlation.
    :param via_object: via object
    :param gene_exp: Dataframe where columns are features (gene) and rows are single cells
    :param lineage: Terminal cluster number to compute lineage driver 
    :param conf_int: default: 0.95 Confidence interval of correlation 
    :param q: Quantile threshold to select cells based on lineage probility. Used when clusters is not given. 
    :return: DataFrame with correlation, confidence intervals, and p-values for each terminal cell fates.
    :rtype: pandas.DataFrame
    '''
    
    if lineage not in via_object.terminal_clusters:
        raise KeyError(f"Lineage {lineage} not in terminal clusters {via_object.terminal_clusters}.")
    
    df_bp = pd.DataFrame(via_object.single_cell_bp, index=gene_exp.index, columns=via_object.terminal_clusters)[lineage]
    
    cell_mask = []
    if clusters is None or clusters == []:
        # Cell mask based on lineage probability inspired by palantir
        eps=1e-2
        fate_probs = via_object.single_cell_bp[:,via_object.terminal_clusters.index(lineage)]
        pseudotime = via_object.single_cell_pt_markov
        idx = np.argsort(pseudotime)
        sorted_fate_probs = fate_probs[idx]
        prob_thresholds = np.empty_like(fate_probs)
        n = fate_probs.shape[0]
        pseudotime_resolution = min(len(set(pseudotime)), n)
        pseudotime_resolution = min(n, n)
        step = n // pseudotime_resolution
        nsteps = n // step
        for i in range(nsteps):
            l, r = i * step, (i + 1) * step
            mprob = np.quantile(sorted_fate_probs[:r], 1 - q, axis=0)
            prob_thresholds[l:r] = mprob[None]
        mprob = np.quantile(sorted_fate_probs, 1 - q, axis=0)
        prob_thresholds[r:] = mprob[None]
        prob_thresholds = np.maximum.accumulate(prob_thresholds, axis=0)
        cell_mask = np.empty_like(fate_probs).astype(bool)
        cell_mask[idx] = prob_thresholds - eps < sorted_fate_probs
    else:
        # Cell mask based on given cluster list
        cell_mask = [True if i in clusters else False for i in via_object.labels]

    # Select cells in given lineage
    print(f'Selected {sum(cell_mask)} cells for lineage {lineage}')
    print(f'Cells from clusters {list(sorted(set(np.array(via_object.labels)[cell_mask])))}')
    df_bp = df_bp[cell_mask]
    gene_exp = gene_exp[cell_mask]

    # Compute pearson correlation
    print(f"Computing driver genes")
    corr = ['']*gene_exp.shape[1]
    for i, gene in enumerate(tqdm(gene_exp.columns)):
        res = pearsonr(gene_exp[gene].values, df_bp.values)
        conf = res.confidence_interval(conf_int)
        corr[i] = pd.Series([res.statistic, res.pvalue, conf.low, conf.high], name=gene, 
                            index=['corr', 'pvalue', 'ci_low', 'ci_high'])
    df_corr = pd.concat(corr, axis=1).T
    
    # Remove invalid correlations outside (-1,1)
    df_corr['corr'][df_corr['corr'] < -1] = np.nan
    df_corr['corr'][df_corr['corr'] > 1] = np.nan
    df_corr = df_corr.dropna()
    return df_corr

def get_gene_trend(via_object, marker_lineages: list = [], df_gene_exp=None, n_splines: int = 10,
                   spline_order: int = 4):
    '''
    Get the gene trend vs pseudotime for a lineage (terminal cell fate)
    :param via_object:
    :param marker_lineages:
    :param df_gene_exp:
    :param n_splines:
    :param spline_order:
    :return: dict of dicts. First dict keys corresponding to terminal cluster of a lineage, second dict having keys "trends": entries with pandas DataFrame with genes (rows) x ("pseudotime") for that lineage and "name" : majority true label
    '''
    sc_pt = via_object.single_cell_pt_markov
    sc_bp = via_object.single_cell_bp
    n_terminal_states = sc_bp.shape[1]
    ts = via_object.terminal_clusters
    trends_dict = {}
    for i_terminal in range(n_terminal_states):
        df_trends = pd.DataFrame()
        # print('ts[i_terminal]',ts[i_terminal],len(np.where(sc_bp[:, i_terminal] > 0.9)[0]))
        if (ts[i_terminal] in marker_lineages and len(np.where(sc_bp[:, i_terminal] > 0.9)[
                                                          0]) > 0):  # check if terminal state is in marker_lineage and i_terminal can be reached
            loc_i = np.where(sc_bp[:, i_terminal] > 0.9)[0]
            val_pt = [sc_pt[pt_i] for pt_i in loc_i]  # TODO,  replace with array to speed up

            max_val_pt = max(val_pt)

            loc_i_bp = np.where(sc_bp[:, i_terminal] > 0.000)[0]  # 0.001
            loc_i_sc = np.where(np.asarray(sc_pt) <= max_val_pt)[0]

            loc_ = np.intersect1d(loc_i_bp, loc_i_sc)
            if len(loc_) > 1:
                gam_in = np.asarray(sc_pt)[loc_]
                x = gam_in.reshape(-1, 1)
                for gene_i in df_gene_exp.columns:
                    y = np.asarray(df_gene_exp[gene_i])[loc_].reshape(-1, 1)

                    weights = np.asarray(sc_bp[:, i_terminal])[loc_].reshape(-1, 1)

                    geneGAM = pg.LinearGAM(n_splines=n_splines, spline_order=spline_order, lam=10).fit(x, y,
                                                                                                       weights=weights)
                    xval = np.linspace(min(sc_pt), max_val_pt, 100 * 2)
                    yg = geneGAM.predict(X=xval)
                    df_trends[str(gene_i)] = yg

            else:
                print(
                    f'{datetime.now()}\tLineage {i_terminal} cannot be reached. Exclude this lineage in trend plotting')
            loc_labels = np.where(np.asarray(via_object.labels) == ts[i_terminal])[0]
            majority_composition = func_mode(list(np.asarray(via_object.true_label)[loc_labels]))
            trends_dict[ts[i_terminal]] = {'trends': df_trends.transpose(), 'name': majority_composition}
    return trends_dict


def csr_mst(adjacency):
    # return minimum spanning tree from adjacency matrix (csr)
    Tcsr = adjacency.copy()
    Tcsr.data *= -1
    Tcsr.data -= np.min(Tcsr.data) - 1
    Tcsr = minimum_spanning_tree(Tcsr)
    return (Tcsr + Tcsr.T) * .5


def connect_all_components(MSTcsr, cluster_graph_csr, adjacency):
    # connect forest of MSTs (csr)
    n, labels = connected_components(csgraph=cluster_graph_csr, directed=False, return_labels=True)
    while n > 1:
        sub_td = MSTcsr[labels == 0, :][:, labels != 0]

        locxy = scipy.sparse.find(MSTcsr == np.min(sub_td.data))

        for i in range(len(locxy[0])):
            if (labels[locxy[0][i]] == 0) & (labels[locxy[1][i]] != 0):
                x, y = locxy[0][i], locxy[1][i]

        cluster_graph_csr[x, y] = adjacency[x, y]
        n, labels = connected_components(csgraph=cluster_graph_csr, directed=False, return_labels=True)
    return cluster_graph_csr


def pruning_clustergraph(adjacency, global_pruning_std=1, max_outgoing=30, preserve_disconnected=True,
                         preserve_disconnected_after_pruning=False, do_max_outgoing=True):
    # neighbors in the adjacency matrix (neighbor-matrix) are not listed in in any order of proximity
    # larger pruning_std factor means less pruning
    # the mst is only used to reconnect components that become disconnect due to pruning
    # print('global pruning std', global_pruning_std, 'max outoing', max_outgoing)
    from scipy.sparse.csgraph import minimum_spanning_tree

    Tcsr = csr_mst(adjacency)
    initial_links_n = len(adjacency.data)

    n_comp, comp_labels = connected_components(csgraph=adjacency, directed=False, return_labels=True)
    print(f"{datetime.now()}\tGraph has {n_comp} connected components before pruning")

    if do_max_outgoing == True:
        adjacency = scipy.sparse.csr_matrix.todense(adjacency)
        row_list = []
        col_list = []
        weight_list = []

        rowi = 0

        for i in range(adjacency.shape[0]):
            row = np.asarray(adjacency[i, :]).flatten()
            n_nonz = min(np.sum(row > 0), max_outgoing)

            to_keep_index = np.argsort(row)[::-1][0:n_nonz]  # np.where(row>np.mean(row))[0]#
            # print('to keep', to_keep_index)
            updated_nn_weights = list(row[to_keep_index])
            for ik in range(len(to_keep_index)):
                row_list.append(rowi)
                col_list.append(to_keep_index[ik])
                dist = updated_nn_weights[ik]
                weight_list.append(dist)
            rowi = rowi + 1
        final_links_n = len(weight_list)

        cluster_graph_csr = csr_matrix((weight_list, (row_list, col_list)), shape=adjacency.shape)
    else:
        cluster_graph_csr = adjacency.copy()
    n_comp, comp_labels = connected_components(csgraph=adjacency, directed=False, return_labels=True)

    sources, targets = cluster_graph_csr.nonzero()
    mask = np.zeros(len(cluster_graph_csr.data), dtype=bool)

    cluster_graph_csr.data = cluster_graph_csr.data / (np.std(cluster_graph_csr.data))  # normalize
    threshold_global = np.mean(cluster_graph_csr.data) - global_pruning_std * np.std(cluster_graph_csr.data)
    mask |= (cluster_graph_csr.data < threshold_global)  # smaller Jaccard weight means weaker edge

    cluster_graph_csr.data[mask] = 0
    cluster_graph_csr.eliminate_zeros()

    prev_n_comp, prev_comp_labels = n_comp, comp_labels  # before pruning
    n_comp, comp_labels = connected_components(csgraph=cluster_graph_csr, directed=False,
                                               return_labels=True)  # n comp after pruning
    print(f"{datetime.now()}\tGraph has {n_comp} connected components after pruning")
    n_comp_preserve = n_comp if preserve_disconnected_after_pruning else prev_n_comp

    # preserve initial disconnected components
    if (preserve_disconnected == True) and (n_comp > prev_n_comp):
        Td = Tcsr.todense()
        Td[Td == 0] = 999.999
        n_comp_ = n_comp
        while n_comp_ > n_comp_preserve:
            for i in range(n_comp_preserve):
                loc_x = np.where(prev_comp_labels == i)[0]
                len_i = len(set(comp_labels[loc_x]))

                while len_i > 1:
                    s = list(set(comp_labels[loc_x]))
                    loc_notxx = np.intersect1d(loc_x, np.where((comp_labels != s[0]))[0])
                    loc_xx = np.intersect1d(loc_x, np.where((comp_labels == s[0]))[0])
                    sub_td = Td[loc_xx, :][:, loc_notxx]
                    locxy = np.where(Td == np.min(sub_td))
                    for i in range(len(locxy[0])):
                        if comp_labels[locxy[0][i]] != comp_labels[locxy[1][i]]:
                            x, y = locxy[0][i], locxy[1][i]

                    cluster_graph_csr[x, y] = adjacency[x, y]
                    cluster_graph_csr[y, x] = adjacency[y, x]

                    n_comp_, comp_labels = connected_components(csgraph=cluster_graph_csr, directed=False,
                                                                return_labels=True)
                    loc_x = np.where(prev_comp_labels == i)[0]
                    len_i = len(set(comp_labels[loc_x]))


    elif (preserve_disconnected == False) and (n_comp > 1):
        cluster_graph_csr = connect_all_components(Tcsr, cluster_graph_csr, adjacency)
        n_comp, comp_labels = connected_components(csgraph=cluster_graph_csr, directed=False, return_labels=True)

    # print('sum in utils',cluster_graph_csr.sum(axis=1))
    n_comp_, comp_labels = connected_components(csgraph=cluster_graph_csr, directed=False, return_labels=True)
    print(f"{datetime.now()}\tGraph has {n_comp_} connected components after reconnecting")
    '''
    for i in range(n_comp_):
        count_ = comp_labels.tolist().count(i)
        print(f'number of clusters in component: {i} is {count_}')
        if count_ <3: print(np.where(comp_labels == i)[0])
    '''
    cluster_graph_dense = cluster_graph_csr.todense()
    sum_degree = cluster_graph_dense.sum(axis=1)  # out degree #need to make it dense for proper indexing
    cluster_graph_dense_transpose = np.transpose(cluster_graph_dense)
    sum_degree_in0 = cluster_graph_dense_transpose.sum(axis=1)  # out degree
    where_condition = np.where(sum_degree == 0)[0]
    where_condition_in0 = np.where(sum_degree_in0 == 0)[
        0]  # out degree of transpose is the same as in-degree of original graph

    intersection_ = [value for value in where_condition_in0 if value in where_condition]
    if len(intersection_) > 0:
        for i in intersection_:
            cluster_graph_dense[i, i] = 1.0
            print(f'handling intersection condition where a singleton cluster {i} without edges exists')

    cluster_graph_csr = csr_matrix(cluster_graph_dense)

    n_comp_, comp_labels = connected_components(csgraph=cluster_graph_csr, directed=False, return_labels=True)

    weights = cluster_graph_csr.data / (np.std(cluster_graph_csr.data))

    edges = list(zip(*cluster_graph_csr.nonzero()))

    if do_max_outgoing == True: trimmed_n = (initial_links_n - final_links_n) * 100. / initial_links_n
    trimmed_n_glob = (initial_links_n - len(weights)) * 100. / initial_links_n
    if do_max_outgoing == True: print(
        f"{datetime.now()}\t{round(trimmed_n, 1)}% links trimmed from local pruning relative to start")
    if global_pruning_std < 0.5:
        print(f"{datetime.now()}\t{round(trimmed_n_glob, 1)}% links trimmed from global pruning relative to start")
        if do_max_outgoing: print(f'initial links {initial_links_n} and final_links_n {final_links_n}')

    return weights, edges, comp_labels


def get_sparse_from_igraph(graph: ig.Graph, weight_attr=None):
    '''

    :param graph: igrapaph
    :param weight_attr:
    :return: csr matrix
    '''
    edges = graph.get_edgelist()
    weights = graph.es[weight_attr] if weight_attr else [1] * len(edges)

    if not graph.is_directed():
        edges.extend([(v, u) for u, v in edges])
        weights.extend(weights)

    shape = graph.vcount()
    shape = (shape, shape)
    if len(edges) > 0:
        row_indices, col_indices = zip(*edges)
        return csr_matrix((weights, (row_indices, col_indices)), shape=shape)
    else:
        # Handle the case when there are no edges - return empty sparse matrix
        return csr_matrix(shape)


def recompute_weights(graph: ig.Graph, label_counts: Counter):
    # harmonic weights
    graph = get_sparse_from_igraph(graph, weight_attr='weight')

    weights, scale_factor, w_min = [], 1., 0
    for s, t, w in zip(*[*graph.nonzero(), graph.data]):
        ns, nt = label_counts[s], label_counts[t]
        nw = w * (ns + nt) / (1. * ns * nt)
        weights.append(nw)

    scale_factor = max(weights) - min(weights)
    w_min = min(weights)
    # if w_min > nw: w_min = nw
    weights = [(w + w_min) / scale_factor for w in weights]

    return csr_matrix((weights, graph.nonzero()), shape=graph.shape)


def affinity_milestone_knn(data, knn_struct, k: int = 10, time_series_labels: list = [], knn_seq: int = 5,
                           t_difference: int = 3) -> csr_matrix:
    '''
    Receives as input "data" which is the subset of the original data provided to VIA on which to make a 'milestone'-KNN-graph and convert the distances to affinities
    For datasets larger than 10,000 points it is advisable from a memory usage point of view to make a milestone knn  as the pairwise distance pdist computation used for mds
    is a very large matrix
    :param data: the subset of the original data on which to make a milestone-KNNgraph and convert the distances to affinity
    :param knn_struct: the index of the knn-graph made of milestone (subset of original data) samples is provided to construct the milestone knngraph
    :param k: number of k-neighbors. since the number of milestones is usually <10,000, we dont want a huge k number
    :param time_series_labels: if using time-series data then the user can optionally guide the milestone knngraph with the sequential time labels
    :param knn_seq: number of sequential neighbors in addition to the regular neighbors
    :return:csr_matrix of the (optionally sequentially augmented) milestone knngraph where edge weights are affinities
    '''

    neighbor_array, distance_array = knn_struct.knn_query(data, k=k)

    if len(time_series_labels) >= 1:
        t_diff_step = t_difference
        n_augmented, d_augmented = sequential_knn(data, time_series_labels, neighbor_array,
                                                  distance_array, k_seq=knn_seq,
                                                  k_reverse=0,
                                                  num_threads=-1, distance='l2',
                                                  )
        neighbor_array = n_augmented
        distance_array = d_augmented
        print('shape neighbor array augmented ', neighbor_array.shape)
        msk = np.full_like(distance_array, True, dtype=np.bool_)
        # print('all edges', np.sum(msk))
        # Remove self-loops
        msk &= (neighbor_array != np.arange(neighbor_array.shape[0])[:, np.newaxis])
        # print('non-self edges', np.sum(msk))

        '''
        #doing local pruning and then adding back the augmented edges does not work so well because when the edge weights are scaled and inverted,
        # the sequentially added edges appear very weak and noisy compared to the fairly strong edges that remain after the local pruning from the inital round of knngraph. If you retain all edges from initial graph construction,
        # then the average weight of edges is exaggeratedly higher than those edge weights from the sequentially added edges, and creates a better gradient of edge weights
        # Local pruning based on neighbor being too far. msk where we want to keep neighbors
        msk = distances <= (np.mean(distances, axis=1) + self.dist_std_local * np.std(distances, axis=1))[:,
                           np.newaxis]
        # Remove self-loops
        msk &= (neighbors != np.arange(neighbors.shape[0])[:, np.newaxis])
        last_n_columns = self.knn_sequential+1
        msk[:,-last_n_columns:] = True # add back the edges belonging to knn-sequentially built part of the graph
        '''
        # remove edges between nodes that >t_diff_step far apart in time_series_labels
        time_series_set_order = list(sorted(list(set(time_series_labels))))
        t_diff_mean = np.mean(
            np.array([int(abs(y - x)) for x, y in zip(time_series_set_order[:-1], time_series_set_order[1:])]))
        print(
            f"{datetime.now()}\tActual average allowable time difference between nodes is {round(t_diff_mean * t_diff_step, 2)}")
        time_series_labels = np.asarray(time_series_labels)
        # print(colored(f"inside time_series msk"))

        rr = 0
        count = 0
        for row in neighbor_array:
            # if rr%20000==0: print(row, type(row), row[0])
            rr += 1
            t_row = time_series_labels[row[0]]  # first neighbor is itself

            for e_i, n_i in enumerate(row):
                if abs(time_series_labels[n_i] - t_row) > t_diff_mean * t_diff_step:
                    count = count + 1
                    if np.sum(msk[row[0]]) > 4: msk[
                        row[0], e_i] = False  # we want to ensure that each cell has at least 5 nn
        print(f'{datetime.now()}\tNumber of non temporal neighbors removed', count)
    else:
        msk = np.full_like(distance_array, True, dtype=np.bool_)
        # print('all edges', np.sum(msk))
        # Remove self-loops
        msk &= (neighbor_array != np.arange(neighbor_array.shape[0])[:, np.newaxis])

    row_mean = np.mean(distance_array, axis=1)
    row_var = np.var(distance_array, axis=1)
    row_znormed_dist_array = (distance_array - row_mean[:, np.newaxis]) / row_var[:, np.newaxis]
    row_znormed_dist_array = np.nan_to_num(row_znormed_dist_array, copy=True, nan=1, posinf=1, neginf=1)
    row_znormed_dist_array[row_znormed_dist_array > 10] = 0
    print(f'minrowznormed', np.min(row_znormed_dist_array))
    row_znormed_dist_array[row_znormed_dist_array < -3] = 1

    affinity_array = np.exp(-row_znormed_dist_array)

    n_neighbors = neighbor_array.shape[1]
    n_cells = neighbor_array.shape[0]

    affinity_array = affinity_array[msk]

    rows = np.array([np.repeat(i, len(x)) for i, x in enumerate(neighbor_array)])[msk]
    cols = neighbor_array[msk]
    result = csr_matrix((affinity_array, (rows, cols)), shape=(n_cells, n_cells), dtype=np.float64)
    result = normalize(result, axis=1)
    '''
    row_list = []

    print('ncells and neighs', n_cells, n_neighbors)
    row_list.extend(list(np.transpose(np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten()))

    col_list = neighbor_array.flatten().tolist()
    list_affinity = affinity_array.flatten().tolist()
    print('affinity list for milestone_knn_new', len(list_affinity), list_affinity[0:20])
    csr_knn = csr_matrix((list_affinity, (row_list, col_list)), shape=(n_cells, n_cells))
    csr_knn = normalize(csr_knn,axis=1)
    return csr_knn
    '''
    return result


def sgd_mds(via_graph: csr_matrix, X_pca, diff_op: int = 1, ndims: int = 2, random_seed=0,
            double_diffusion: bool = True):
    '''

    :param via_graph: via_graph = v0.csr_full_graph #single cell knn graph representation based on hnsw
    :param diff_op: power of diffusion operation
    :param ndims:
    :return:
    '''
    # outlier handling of graph edges - handles instabilities at tails of the transition probabilities
    via_graph.data = np.clip(via_graph.data, np.percentile(via_graph.data, 10), np.percentile(via_graph.data, 90))
    row_stoch = normalize(via_graph, norm='l1', axis=1)
    row_stoch = row_stoch ** diff_op
    # msk = row_stoch==0
    from scipy.sparse.csgraph import connected_components
    n_components, labels_cc = connected_components(csgraph=row_stoch, directed=False, return_labels=True)
    if n_components > 1: print('Considering re-running with higher knn value. disconnected components exist')

    temp_pca = csr_matrix(X_pca)

    X_mds = row_stoch * temp_pca  # matrix multiplication to diffuse the pcs. This is not the same as using diffusion components which are the eigenvectors of the transition matrix (in our case given by csr_full_graph)

    X_mds = squareform(pdist(X_mds.todense()))

    # print(X_mds[0:10,:])
    # print(X_mds[490:499,:])
    # X_mds[msk.todense()]=np.amax(X_mds)
    # X_mds = X_mds+X_mds.transpose() #has to be symmetric
    # np.fill_diagonal(X_mds,0)
    # X_mds = squareform(pdist(temp_pca.todense()))# testing when no viagraph diffusion

    print(f'{datetime.now()}\tStarting MDS on milestone')

    Y_classic = classic(X_mds, n_components=ndims, random_state=random_seed)

    X_mds = sgd(X_mds, n_components=ndims, random_state=random_seed, init=Y_classic)
    if double_diffusion == True: X_mds = row_stoch * X_mds  # added Dec 12 10pm to test diffusion of x-mds after mds
    return X_mds


def sgd(D, n_components=2, random_state=None, init=None):
    """Metric MDS using stochastic gradient descent
    Parameters
    ----------
    D : array-like, shape=[n_samples, n_samples]
        pairwise distances
    n_components : int, optional (default: 2)
        number of dimensions in which to embed `D`
    random_state : int or None, optional (default: None)
        numpy random state
    init : array-like or None
        Initialization algorithm or state to use for MMDS
    Returns
    -------
    Y : array-like, embedded data [n_sample, ndim]
    """
    import s_gd2
    N = D.shape[0]
    D = squareform(D)
    # Metric MDS from s_gd2
    Y = s_gd2.mds_direct(N, D, init=init, random_seed=random_state)
    return Y


def classic(D, n_components=2, random_state=None):
    """Fast CMDS using random SVD
    Parameters
    ----------
    D : array-like, shape=[n_samples, n_samples]
        pairwise distances
    n_components : int, optional (default: 2)
        number of dimensions in which to embed `D`
    random_state : int, RandomState or None, optional (default: None)
        numpy random state
    Returns
    -------
    Y : array-like, embedded data [n_sample, ndim]
    """
    from sklearn.decomposition import PCA
    D = D ** 2
    D = D - D.mean(axis=0)[None, :]
    D = D - D.mean(axis=1)[:, None]
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    Y = pca.fit_transform(D)
    return Y


def construct_knn_utils(data: np.ndarray, too_big: bool = False, distance='l2', num_threads: int = -1,
                        knn: int = 20):
    """
    Construct K-NN graph for given data. This is also featured within VIA class, but since we use it outside the class, we declare it in utils too
    too_big: if constructing knn during an iteration of PARC that tries to break up very large clusters. typically False unless called within too_big SubPARC

    Parameters
    ----------
    data: np.ndarray of shape (n_samples, n_features)
        Data matrix over which to construct knn graph

    too_big: bool, default = False

    Returns
    -------
    Initialized instance of Index to be used over given data
    """
    import hnswlib
    # if self.knn > 100:
    # print(colored(f'Passed number of neighbors exceeds max value. Setting number of neighbors to 100'))
    # k = min(100, self.knn + 1)
    k = knn + 1  # since first knn is itself

    nsamples, dim = data.shape
    ef_const, M = 200, 30
    if not too_big:
        if nsamples < 10000:
            k = ef_const = min(nsamples - 10, 500)
        if nsamples <= 50000 and dim > 30:
            M = 48  # good for scRNA-seq where dimensionality is high

    p = hnswlib.Index(space=distance, dim=dim)
    p.set_num_threads(num_threads)
    p.init_index(max_elements=nsamples, ef_construction=ef_const, M=M)
    p.add_items(data)
    p.set_ef(k)
    return p


def spatial_knn(coords: np.ndarray, neighbors: np.ndarray, distances: np.ndarray, k_spatial: int,
                ) -> np.ndarray:
    '''
    Make the sequential knn graph connecting cells in adjacent time points and merge this sequential graph
    together with the original KNN graph (distances, neighbors) made without any prior knowledge of the teim-series information

    :param data: the data we want to make a sequential graph with
    :param time_series_labels: numerical labels used to make the sequential graph
    :param neighbors: array of n_samples (data.shape[0]) * n_knn in the original knn graph made without knowledge of the time-series sequences
    :param distances: array n_samples x n_knn in the original knn graph graph made without any time-series info
    :param k_seq: number of knn for sequential
    :param k_reverse: number of sequential edges from time{i} to time{i-1}
    :param knn: number of knn in the index constriction that we subsequently will query on
    :return: 2 ndarrays augmented_nn, augmented_nn_data that contain the neighbor and distance values of the final sequential graph + original knn graph. the data is distances NOT affinity.
    '''

    knn_struct = construct_knn_utils(coords, knn=k_spatial)
    spatial_neighbors, spatial_distances = knn_struct.knn_query(coords, k=k_spatial)
    ## Remove self-loops
    # spatial_neighbors = spatial_neighbors[:,1:]
    # spatial_distances = spatial_distances[:, 1:]
    print('spatial_neighbors shape', spatial_neighbors.shape)
    print('spatial_distances shape', spatial_distances.shape)
    # msk = np.full_like(spatial_distances, True, dtype=np.bool_)
    # rows = np.array([np.repeat(i, len(x)) for i, x in enumerate(neighbors)])[msk]
    # cols = neighbors[msk]
    # print('spatial distances', spatial_distances.shape)
    print(f'min spatial distances { np.min(spatial_distances)}')
    print(f'max spatial distances { np.max(spatial_distances)}')
    print(f'mean spatial distances { np.mean(spatial_distances)}')
    print(f'std spatial distances {np.std(spatial_distances)}')
    from sklearn.preprocessing import normalize
    spatial_distances = spatial_distances / (
    np.mean(spatial_distances, axis=1)[:, np.newaxis])  # normalize(spatial_distances, norm="l1", axis=1)

    # scaling the spatial distances to be on the same magnitude scale as the gene expression distances
    spatial_distances = spatial_distances * (np.mean(distances, axis=1)[:, np.newaxis])


    print(f'min gene distances { np.min(distances)}')
    print(f'max gene distances { np.max(distances)}')
    print(f'mean gene distances { np.mean(distances)}')
    print(f'std gene distances {np.std(distances)}')

    print(f"{datetime.now()}\tShape neighbors {neighbors.shape} and sequential neighbors {spatial_neighbors.shape}")

    augmented_nn = np.concatenate((neighbors, spatial_neighbors), axis=1).astype('int')
    #print(f'augmented nn {spatial_neighbors[0, :]}')
    print(f'augmented distances {spatial_distances[0, 0:5]}')

    print(f'k-nn {neighbors[0, :]}')
    print(f'gene distances {distances[0, 0:5]}')

    augmented_nn_data = np.concatenate((distances, spatial_distances), axis=1)
    print(f"{datetime.now()}\tShape augmented neighbors {augmented_nn.shape}")

    return augmented_nn, augmented_nn_data


def spatial_input_old(X_genes:ndarray, X_coords:ndarray, knn_spatial:int=5, spatial_weight:float=0.3, weight_transformation='inverse'):
    '''

    :param X_genes:
    :param X_coords:
    :param knn_spatial:
    :param spatial_weight:
    :param weight_transformation: 'inverse' or 'gaussian'
    :return:
    '''
    # make knn graph on spatial coordinates
    # Import scikit-learn preprocessing
    from sklearn.preprocessing import normalize

    print(f'x_coords shape {X_coords.shape}')
    print(f'x_genes shape {X_genes.shape}')
    n_samples = X_coords.shape[0]
    knn_struct = construct_knn_utils(X_coords, knn=knn_spatial)
    neighbors, distances = knn_struct.knn_query(X_coords, k=knn_spatial)
    msk = np.full_like(distances, True, dtype=np.bool_)

    # Remove self-loops
    msk &= (neighbors != np.arange(neighbors.shape[0])[:, np.newaxis])
    rows = np.array([np.repeat(i, len(x)) for i, x in enumerate(neighbors)])[msk]
    cols = neighbors[msk]
    distances_ones = np.ones_like(distances)
    weights = distances[msk]

    weights = (np.mean(distances[msk]) ** 2) / (
            distances[msk] + np.min(weights))  # larger weight is a stronger edge
    # weights = np.exp(-distances[msk]/stdd[:,None])

    #result = csr_matrix((distances_ones[msk], (rows, cols)), shape=(n_samples, n_samples), dtype=np.float32)
    result = csr_matrix((weights, (rows, cols)), shape=(n_samples, n_samples), dtype=np.float32)

    X_knn_genes = result * X_genes

    X_knn_genes = normalize(X_knn_genes, norm="l1", axis=1) #divide each value by sum of the row (so each row sums to 1)

    X_genes = normalize(X_genes, norm="l1", axis=1)

    X_spatial = spatial_weight * X_knn_genes + (1-spatial_weight) *X_genes

    return X_spatial

def spatial_input(X_genes:ndarray, spatial_coords:ndarray, spatial_slice_labels: list=[],knn_spatial:int=5, spatial_weight:float=0.3, weight_transformation='inverse'):
    '''

    :param X_genes:
    :param X_coords:
    :param knn_spatial:
    :param spatial_weight:
    :param weight_transformation: 'inverse' or 'gaussian'
    :return:
    '''

    all_new_nn = np.ones((spatial_coords.shape[0], knn_spatial))
    all_new_nn_data = np.ones((spatial_coords.shape[0], knn_spatial))

    if len(spatial_slice_labels)>0: spatial_slice_labels = np.asarray(spatial_slice_labels)
    else: spatial_slice_labels=['slice1'] * spatial_coords.shape[0]
    spatial_slice_labels = np.asarray(spatial_slice_labels)

    slices_set = list(sorted(list(set(spatial_slice_labels))))  # values sorted in ascending order
    print(f"{datetime.now()}\tThese slices are present: {slices_set}")
    # make knn graph on spatial coordinates
    # Import scikit-learn preprocessing
    from sklearn.preprocessing import normalize

    print(f'x_coords shape {spatial_coords.shape}')
    print(f'x_genes shape {X_genes.shape}')
    n_samples = spatial_coords.shape[0]
    print('nsamples (slices)', n_samples)
    #knn_struct = construct_knn_utils(spatial_coords, knn=knn_spatial)
    #neighbors, distances = knn_struct.knn_query(spatial_coords, k=knn_spatial)
    for slice_i in slices_set:

        tj_loc = np.where(spatial_slice_labels == slice_i)[0]

        tj_data = spatial_coords[tj_loc, :] #the subset of spatial coords that belong to slice_i


        tj_knn = construct_knn_utils(tj_data, knn=knn_spatial)
        ti_query_nn, d_ij = tj_knn.knn_query(tj_data,   k=knn_spatial)
        #d_ij = d_ij / (            np.mean(d_ij, axis=1)[:, np.newaxis])
        # normalize the spatial distances, later scale them. Works marginally better without this normalization here. Given the distance scales across tisues should be comparable. we dont need this here.

        for xx_i, xx in enumerate(tj_loc):
            all_new_nn[xx, 0:knn_spatial] = tj_loc[ti_query_nn[
                xx_i]]  # need to convert the tj_query_nn indices back to the indices of tj_loc in full data
            all_new_nn_data[xx, 0:knn_spatial] = d_ij[xx_i]

    #print(f"{datetime.now()}\tShape neighbors {neighbors.shape} and spatial neighbors {all_new_nn.shape}")
    print(f"{datetime.now()}\tShape spatial neighbors and data shape {all_new_nn.shape} {all_new_nn_data.shape}")


    msk = np.full_like(all_new_nn_data, True, dtype=np.bool_)


    # Remove self-loops
    msk &= (all_new_nn != np.arange(all_new_nn.shape[0])[:, np.newaxis])
    rows = np.array([np.repeat(i, len(x)) for i, x in enumerate(all_new_nn)])[msk]
    cols = all_new_nn[msk]
    distances_ones = np.ones_like(all_new_nn_data)
    weights = all_new_nn_data[msk]

    weights = (np.mean(all_new_nn_data[msk]) ** 2) / (
            all_new_nn_data[msk] + np.min(weights))  # larger weight is a stronger edge

    result = csr_matrix((weights, (rows, cols)), shape=(n_samples, n_samples), dtype=np.float32)

    X_knn_genes = result * X_genes

    X_knn_genes = normalize(X_knn_genes, norm="l1", axis=1) #divide each value by sum of the row (so each row sums to 1)

    X_genes = normalize(X_genes, norm="l1", axis=1)

    print(f"{datetime.now()}\tSpatial gene smoothing... {all_new_nn.shape}")
    X_spatial = spatial_weight * X_knn_genes + (1-spatial_weight) *X_genes

    return X_spatial

def spatial_knn_new(spatial_coords: np.ndarray, spatial_slice_labels: list, neighbors: np.ndarray, distances: np.ndarray, k_spatial: int,
                   distance_metric: str = 'l2', num_threads: int = -1,
                   too_big: bool = False) -> np.ndarray:
    '''
    Make the sequential knn graph connecting cells in adjacent time points and merge this sequential graph
    together with the original KNN graph (distances, neighbors) made without any prior knowledge of the teim-series information

    :param data: the data we want to make a sequential graph with
    :param time_series_labels: numerical labels used to make the sequential graph
    :param neighbors: array of n_samples (data.shape[0]) * n_knn in the original knn graph made without knowledge of the time-series sequences
    :param distances: array n_samples x n_knn in the original knn graph graph made without any time-series info
    :param k_seq: number of knn for sequential
    :param k_reverse: number of sequential edges from time{i} to time{i-1}
    :param knn: number of knn in the index constriction that we subsequently will query on
    :return: 2 ndarrays augmented_nn, augmented_nn_data that contain the neighbor and distance values of the final sequential graph + original knn graph. the data is distances NOT affinity.
    '''
    all_new_nn = np.ones((spatial_coords.shape[0], k_spatial))
    all_new_nn_data = np.ones((spatial_coords.shape[0], k_spatial))
    slices_set = list(sorted(list(set(spatial_slice_labels))))  # values sorted in ascending order
    print(f"{datetime.now()}\tThese slices are present: {slices_set}")
    spatial_slice_labels = np.asarray(spatial_slice_labels)

    for slice_i in slices_set:
        #print('slice _i', slice_i)
        tj_loc = np.where(spatial_slice_labels == slice_i)[0]
        #print('indices of slice',slice_i,'are:',tj_loc[0:10])
        tj_data = spatial_coords[tj_loc, :] #the subset of spatial coords that belong to slice_i
        tj_data_distances = distances[tj_loc,:] #the distances of nn in gene space of this slice

        tj_knn = _construct_knn(tj_data, knn=k_spatial, distance=distance_metric, num_threads=num_threads, too_big=too_big)
        ti_query_nn, d_ij = tj_knn.knn_query(tj_data,   k=k_spatial)
        d_ij = d_ij / (
            np.mean(d_ij, axis=1)[:, np.newaxis])  # normalize the spatial distances, later scale them
        # scaling the spatial distances to be on the same magnitude scale as the gene expression based distances of nn
        d_ij= d_ij * (np.mean(tj_data_distances, axis=1)[:, np.newaxis])
        for xx_i, xx in enumerate(tj_loc):
            all_new_nn[xx, 0:k_spatial] = tj_loc[ti_query_nn[
                xx_i]]  # need to convert the tj_query_nn indices back to the indices of tj_loc in full data
            all_new_nn_data[xx, 0:k_spatial] = d_ij[xx_i]

    print(f"{datetime.now()}\tShape neighbors {neighbors.shape} and spatial neighbors {all_new_nn.shape}")

    augmented_nn = np.concatenate((neighbors, all_new_nn), axis=1).astype('int')
    augmented_nn_data = np.concatenate((distances, all_new_nn_data), axis=1)
    print(f"{datetime.now()}\tShape of spatially augmented neighbors {augmented_nn.shape}")

    return augmented_nn, augmented_nn_data

def sequential_knn(data: np.ndarray, time_series_labels: list, neighbors: np.ndarray, distances: np.ndarray, k_seq: int,
                   k_reverse: int = 0, distance: str = 'l2', num_threads: int = -1,
                   too_big: bool = False) -> np.ndarray:
    '''
    Make the sequential knn graph connecting cells in adjacent time points and merge this sequential graph
    together with the original KNN graph (distances, neighbors) made without any prior knowledge of the teim-series information

    :param data: the data we want to make a sequential graph with
    :param time_series_labels: numerical labels used to make the sequential graph
    :param neighbors: array of n_samples (data.shape[0]) * n_knn in the original knn graph made without knowledge of the time-series sequences
    :param distances: array n_samples x n_knn in the original knn graph graph made without any time-series info
    :param k_seq: number of knn for sequential
    :param k_reverse: number of sequential edges from time{i} to time{i-1}
    :param knn: number of knn in the index constriction that we subsequently will query on
    :return: 2 ndarrays augmented_nn, augmented_nn_data that contain the neighbor and distance values of the final sequential graph + original knn graph. the data is distances NOT affinity.
    '''
    all_new_nn = np.ones((data.shape[0], k_seq + k_reverse))
    all_new_nn_data = np.ones((data.shape[0], k_seq + k_reverse))
    time_series_set = sorted(list(set(time_series_labels)))  # values sorted in ascending order
    print(f"{datetime.now()}\tTime series ordered set {time_series_set}")
    time_series_labels = np.asarray(time_series_labels)

    for counter, tj in enumerate(time_series_set[1:]):
        ti = time_series_set[counter]

        tj_loc = np.where(time_series_labels == tj)[0]

        ti_loc = np.where(time_series_labels == ti)[0]

        ti_data = data[ti_loc, :]
        tj_data = data[tj_loc, :]

        if k_seq > 0:
            tj_knn = _construct_knn(tj_data, knn=k_seq, distance=distance, num_threads=num_threads, too_big=too_big)
            ti_query_nn, d_ij = tj_knn.knn_query(ti_data,
                                                 k=k_seq)  # find the cells in tj that are closest to those in ti

            for xx_i, xx in enumerate(ti_loc):
                all_new_nn[xx, 0:k_seq] = tj_loc[ti_query_nn[
                    xx_i]]  # need to convert the ti_query_nn indices back to the indices of tj_loc in full data
                all_new_nn_data[xx, 0:k_seq] = d_ij[xx_i]

        if k_reverse > 0:
            ti_knn = _construct_knn(ti_data, knn=k_reverse, distance=distance, num_threads=num_threads, too_big=too_big)
            tj_query_nn, d_ji = ti_knn.knn_query(tj_data, k=k_reverse)
            for xx_i, xx in enumerate(tj_loc):
                all_new_nn[xx, k_seq:] = ti_loc[tj_query_nn[
                    xx_i]]  # need to convert the tj_query_nn indices back to the indices of tj_loc in full data
                all_new_nn_data[xx, k_seq:] = d_ji[xx_i]

    print(f"{datetime.now()}\tShape neighbors {neighbors.shape} and sequential neighbors {all_new_nn.shape}")

    augmented_nn = np.concatenate((neighbors, all_new_nn), axis=1).astype('int')
    augmented_nn_data = np.concatenate((distances, all_new_nn_data), axis=1)
    print(f"{datetime.now()}\tShape augmented neighbors {augmented_nn.shape}")

    return augmented_nn, augmented_nn_data


def _construct_knn(data: np.ndarray, knn: int, distance: str, num_threads: int, too_big: bool = False):
    """
    Construct K-NN graph index for given data. This is not the knngraph in itself. that is made by querying this index

    Parameters
    ----------
    data: np.ndarray of shape (n_samples, n_features)
        Data matrix over which to construct knn graph

    too_big: bool, default = False #in the subparc_toobig routine is set to True
    knn: int self.knn +1
    distance: str self.distance (type of metric) e.g. 'l2'
    num_threads:int default =-1
    Returns
    -------
    Initialized instance of hnswlib.Index to be used over given data
    """
    import hnswlib
    k = knn + 1  # since first knn is itself

    nsamples, dim = data.shape
    ef_const, M = 200, 30
    if not too_big:
        if nsamples < 10000:
            # k = ef_const = min(nsamples - 10, 500) #was this until Sept62023
            ef_const = min(nsamples - 1, 500)
            k = min(nsamples - 1, 500)
        if nsamples <= 50000 and dim > 30:
            M = 48  # good for scRNA-seq where dimensionality is high

    p = hnswlib.Index(space=distance, dim=dim)
    p.set_num_threads(num_threads)
    p.init_index(max_elements=nsamples, ef_construction=ef_const, M=M)
    p.add_items(data)
    p.set_ef(k)
    return p


def getbb(sc, ax):
    """
    Function to return a list of bounding boxes in data coordinates for a scatter plot.
    Adapted from https://stackoverflow.com/questions/55005272/
    """
    ax.figure.canvas.draw()  # need to draw before the transforms are set.
    transform = sc.get_transform()
    transOffset = sc.get_offset_transform()
    offsets = sc._offsets
    paths = sc.get_paths()
    transforms = sc.get_transforms()

    if not transform.is_affine:
        paths = [transform.transform_path_non_affine(p) for p in paths]
        transform = transform.get_affine()
    if not transOffset.is_affine:
        offsets = transOffset.transform_non_affine(offsets)
        transOffset = transOffset.get_affine()

    if isinstance(offsets, np.ma.MaskedArray):
        offsets = offsets.filled(np.nan)

    bboxes = []

    if len(paths) and len(offsets):
        if len(paths) < len(offsets):
            # for usual scatters you have one path, but several offsets
            paths = [paths[0]] * len(offsets)
        if len(transforms) < len(offsets):
            # often you may have a single scatter size, but several offsets
            transforms = [transforms[0]] * len(offsets)

        for p, o, t in zip(paths, offsets, transforms):
            result = get_path_collection_extents(
                transform.frozen(), [p], [t], [o], transOffset.frozen()
            )
            # bboxes.append(result.inverse_transformed(ax.transData))
            bboxes.append(result.transformed(ax.transData.inverted()))
    return bboxes


def sc_loc_ofsuperCluster_PCAspace(p0, idx):
    '''
    #helper function for draw_trajectory_gams in order to find the PCA location of the terminal and intermediate clusters and roots
    :param p0: coarse via object
    :param idx: if using a subsampled PCA space for visualization. otherwise just range(0,n_samples)
    :return:
    '''
    import hnswlib
    # ci_list first finds location in unsampled PCA space of the location of the super-cluster or sub-terminal-cluster and root
    # Returns location (index) of cell nearest to the ci_list in the downsampled space
    # print("dict of terminal state pairs, Super: sub: ", p1.dict_terminal_super_sub_pairs)
    p0_labels = np.asarray(p0.labels)

    p0_sc_markov_pt = p0.single_cell_pt_markov
    ci_list = []
    for ci in range(len(list(set(p0.labels)))):

        if (ci in p0.root) & (len(p0.root) == 1):
            loc_root = np.where(np.asarray(p0.root) == ci)[0][0]

            p0_root_label = p0.root[loc_root]
            loc_i = np.where(np.asarray(p0_labels) == p0_root_label)[0]

            # loc_i = np.where(p0.labels == ci)[0]
            val_pt = [p0_sc_markov_pt[i] for i in loc_i]
            th_pt = np.percentile(val_pt, 20)  # 50
            loc_i = [loc_i[i] for i in range(len(val_pt)) if val_pt[i] <= th_pt]
            temp = np.mean(p0.data[loc_i], axis=0)
            labelsq, distances = p0.knn_struct.knn_query(temp, k=1)
            ci_list.append(labelsq[0][0])
        else:
            loc_i = np.where(p0_labels == ci)[0]
            temp = np.mean(p0.data[loc_i], axis=0)
            labelsq, distances = p0.knn_struct.knn_query(temp, k=1)
            ci_list.append(labelsq[0][0])

        X_ds = p0.data[idx]
        p_ds = hnswlib.Index(space='l2', dim=p0.data.shape[1])
        p_ds.init_index(max_elements=X_ds.shape[0], ef_construction=200, M=16)
        p_ds.add_items(X_ds)
        p_ds.set_ef(max(1, min(50, X_ds.shape[0])))

        new_superclust_index_ds = {}
        for en_item, item in enumerate(ci_list):
            labelsq, distances = p_ds.knn_query(p0.data[item, :], k=1)
            # new_superclust_index_ds.append(labelsq[0][0])
            new_superclust_index_ds.update({en_item: labelsq[0][0]})
    # print('new_superclust_index_ds',new_superclust_index_ds)
    return new_superclust_index_ds


def plot_sc_pb(ax, fig, embedding, prob, ti, cmap_name: str = 'plasma', scatter_size=None, vmax=99, vmin=1, fontsize: int = 10,
               alpha_factor=0.9, show_legend: bool = True):
    '''
    This is a helper function called by draw_sc_lineage_probability which plots the single-cell lineage probabilities

    :param ax:
    :param fig:
    :param embedding:
    :param prob:
    :param ti:
    :param cmap_name:
    :param scatter_size:
    :return:
    '''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # prob = np.sqrt(prob)  # scale values to improve visualization of colors
    vmax = np.percentile(prob, vmax)

    vmin = np.percentile(prob,vmin)
    print('setting vmin to', vmin)
    # vmax=1
    cmap = plt.get_cmap(cmap_name)
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(prob))
    if scatter_size is None:
        scatter_size = 3 if embedding.shape[0] > 10000 else 10

    # changing the alpha transparency parameter for plotting points

    c = cmap(prob).reshape(-1, 4)
    im = ax.scatter(embedding[:, 0], embedding[:, 1], c=prob, s=0.01, cmap=cmap_name, edgecolors='none', vmin=vmin, #vmin=0
                    vmax=vmax)  # prevent auto-normalization of colors
    # im = ax.scatter(embedding[:, 0], embedding[:, 1], c=c, s=0.01,  edgecolors='none')

    if show_legend:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ax.set_title('Lineage: ' + str(ti), fontsize=int(fontsize * 1.2))
        cb = fig.colorbar(im, cax=cax, orientation='vertical', label='lineage likelihood')

        ax_cb = cb.ax
        text = ax_cb.yaxis.label
        font = matplotlib.font_manager.FontProperties(size=fontsize)  # family='times new roman', style='italic',
        text.set_font_properties(font)
        ax_cb.tick_params(labelsize=int(fontsize * 0.8))

    c = cmap(prob).reshape(-1, 4)

    # c = cmap(norm(prob)).reshape(-1, 4)
    loc_c = np.where(prob <= 0.3)[0]
    ax.scatter(embedding[loc_c, 0], embedding[loc_c, 1], c=prob[loc_c], s=scatter_size, edgecolors='none',
               alpha=0.2 * alpha_factor, cmap=cmap_name, vmin=0, vmax=vmax)
    c[loc_c, 3] = 0.2
    loc_c = np.where((prob > 0.3) & (prob <= 0.5))[0]
    c[loc_c, 3] = 0.5
    ax.scatter(embedding[loc_c, 0], embedding[loc_c, 1], c=prob[loc_c], s=scatter_size, edgecolors='none',
               alpha=0.5 * alpha_factor, cmap=cmap_name, vmin=0, vmax=vmax)
    loc_c = np.where((prob > 0.5) & (prob <= 0.7))[0]
    c[loc_c, 3] = 0.8  # changing the alpha
    ax.scatter(embedding[loc_c, 0], embedding[loc_c, 1], c=prob[loc_c], s=scatter_size, edgecolors='none',
               alpha=0.8 * alpha_factor, cmap=cmap_name, vmin=0, vmax=vmax)
    loc_c = np.where((prob > 0.7))[0]
    c[loc_c, 3] = 0.8
    ax.scatter(embedding[loc_c, 0], embedding[loc_c, 1], c=prob[loc_c], s=scatter_size, edgecolors='none',
               alpha=0.8 * alpha_factor, cmap=cmap_name, vmin=0, vmax=vmax)





def sigmoid_func(X):
    return 1 / (1 + np.exp(-X))


def sigmoid_scalar(x, scale=1, shift=0):
    return 1 / (1 + math.exp(-scale * (x - shift)))


def logistic_function(X, par_slope=1):
    '''
    https://en.wikipedia.org/wiki/Generalised_logistic_function
    :param X: input matrix X on which elementwise logistic function will be performed
    :param par_slope:
    :return:
    '''
    return 1 / (1 + np.exp(par_slope * -X))


def cosine_sim(A, B):
    '''
    :param A: matrix with n_samples and n_var
    :param B: matrix with m_samples and n_var
    :return: matrix of cosine similarity between A and B
    https://towardsdatascience.com/cosine-similarity-matrix-using-broadcasting-in-python-2b1998ab3ff3
    '''
    # numerator
    num = np.dot(A, B.T)

    # denominator
    p1 = np.sqrt(np.sum(A ** 2, axis=1))  # [:, np.newaxis] when A and B are both matrices
    p2 = np.sqrt(np.sum(B ** 2))
    # print(p1, p2, p1*p2)
    # if A and B is a matrix rather than vector, use below
    # p1 = np.sqrt(np.sum(A ** 2, axis=1))[:, np.newaxis] when A and B are both matrices
    # p2 = np.sqrt(np.sum(B ** 2), axis = 1))[np.newaxis, :]
    return num / (p1 * p2)


def make_edgebundle_viagraph(layout=None, graph=None, initial_bandwidth=0.05, decay=0.9, edgebundle_pruning=0.5,
                             via_object=None):
    '''
    # Perform Edgebundling of edges in clustergraph of VIA to return a hammer bundle. hb.x and hb.y contain all the x and y coords of the points that make up the edge lines.
    # each new line segment is separated by a nan value
    # reference: https://datashader.org/_modules/datashader/bundling.html#hammer_bundle
    :param layout: force-directed layout coordinates of graph
    :param graph: igraph clustergraph
    :param initial_bandwidth: increasing bw increases merging of minor edges
    :param decay: increasing decay increases merging of minor edges #https://datashader.org/user_guide/Networks.html
    :param edgebundle_pruning: float (0-1), smaller value means more pruning away edges that can be visualised. Only required when no layout is precomputed
    :return: hb hammerbundle class with hb.x and hb.y containing the coords
    '''
    from datashader.bundling import connect_edges, hammer_bundle

    if (graph is None):# or (layout is None):
        graph = via_object.cluster_graph_csr_not_pruned
        edgeweights_layout, edges_layout, comp_labels_layout = pruning_clustergraph(graph,
                                                                                    global_pruning_std=edgebundle_pruning,
                                                                                    preserve_disconnected=True,
                                                                                    preserve_disconnected_after_pruning=True,
                                                                                    do_max_outgoing=False)

        # layout = locallytrimmed_g.layout_fruchterman_reingold(weights='weight') #uses non-clipped weights but this can skew layout due to one or two outlier edges
        layout_g = ig.Graph(edges_layout, edge_attrs={'weight': edgeweights_layout}).simplify(combine_edges='sum')
        layout_g_csr = get_sparse_from_igraph(layout_g, weight_attr='weight')
        weights_for_layout = np.asarray(layout_g_csr.data)
        # clip weights to prevent distorted visual scale in layout
        weights_for_layout = np.clip(weights_for_layout, np.percentile(weights_for_layout, 10),
                                     np.percentile(weights_for_layout, 90))
        weights_for_layout = list(weights_for_layout)
        graph = ig.Graph(list(zip(*layout_g_csr.nonzero())), edge_attrs={'weight': weights_for_layout})
        # the layout of the graph is determine by a pruned clustergraph and the directionality of edges will be based on the final markov pseudotimes
        # the edgeweights of the bundle-edges is determined by the distance based metrics and jaccard similarities and not by the pseudotimes
        # for the transition matrix used in the markov pseudotime and differentiation probability computations, the edges will be further biased by the hittings times and markov pseudotimes
        if layout is None: layout = graph.layout_fruchterman_reingold(weights='weight')

    print(f'{datetime.now()}\tMake via clustergraph edgebundle')
    data_node = [[node] + layout.coords[node] for node in range(graph.vcount())]

    nodes = pd.DataFrame(data_node, columns=['id', 'x', 'y'])

    nodes.set_index('id', inplace=True)

    edges = pd.DataFrame([e.tuple for e in graph.es], columns=['source', 'target'])

    edges['weight'] = graph.es['weight']
    hb = hammer_bundle(nodes, edges, weight='weight', initial_bandwidth=initial_bandwidth,
                       decay=decay)  # default bw=0.05, dec=0.7
    print(f'{datetime.now()}\tHammer dims: Nodes shape: {nodes.shape} Edges shape: {edges.shape}')
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.plot(hb.x, hb.y, 'y', zorder=1, linewidth=3)
    # hb.plot(x="x", y="y",figsize=(9,9))
    # plt.show()
    '''
    if via_object is not None:
        print(f'{datetime.now()}\tUpdating the hammerbundle and graph layout of via clustergraph')
        via_object.hammerbundle_cluster = hb
        via_object.graph_node_pos = layout.coords
    '''
    return hb, layout


def _make_edgebundle_sc(embedding, sc_graph, initial_bandwidth=0.05, decay=0.70, sc_clusterlabel: list = []):
    '''
     initial_bandwidth = 0.30, decay=0.90,
    # Perform Edgebundling of edges in clustergraph to return a hammer bundle of single-cell level edges. hb.x and hb.y contain all the x and y coords of the points that make up the edge lines.
    # each new line segment is separated by a nan value
    # https://datashader.org/_modules/datashader/bundling.html#hammer_bundle
    :param embedding: embedding single cell. looks nicer when done on via_mds as more streamlined continuous diffused graph structure. Umap is a but "clustery"
    :param graph: igraph cluster graph level
    :param sc_graph: igraph set as the via attribute self.ig_full_graph
    :param initial_bandwidth: increasing bw increases merging of minor edges
    :param decay: increasing decay increases merging of minor edges #https://datashader.org/user_guide/Networks.html
    :return: hb hammerbundle class with hb.x and hb.y containing the coords
    '''
    from datashader.bundling import connect_edges, hammer_bundle
    print(f"{datetime.now()}\tComputing Edgebundling at single-cell level")
    data_node = [node for node in range(embedding.shape[0])]
    nodes = pd.DataFrame(data_node, columns=['id'])
    nodes.set_index('id', inplace=True)
    nodes['x'] = embedding[:, 0]
    nodes['y'] = embedding[:, 1]

    edges = pd.DataFrame([e.tuple for e in sc_graph.es], columns=['source', 'target'])
    print(edges)
    edges['weight0'] = sc_graph.es['weight']

    print(edges['source'].max())
    edges['source_cluster'] = [sc_clusterlabel[i] for i in edges['source']]
    edges['target_cluster'] = [sc_clusterlabel[i] for i in edges['target']]
    # seems to work better when allowing the bundling to occur on unweighted representation and later using length of segments to color code significance
    edges[
        'weight'] = 1  # [1/i for i in edges['weight0']]np.where((edges['source_cluster'] != edges['target_cluster']) , 1,0.1)#[1/i for i in edges['weight0']]#

    # remove rows by filtering
    edges = edges[edges['source'] != edges['target']]
    # edges = edges[edges['source_cluster'] != edges['target_cluster']]

    edges.drop('target_cluster', inplace=True, axis=1)
    edges.drop('source_cluster', inplace=True, axis=1)

    hb = hammer_bundle(nodes, edges, weight='weight', initial_bandwidth=initial_bandwidth,
                       decay=decay)  # default bw=0.05, dec=0.7

    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.plot(hb.x, hb.y, 'y', zorder=1, linewidth=3)
    # hb.plot(x="x", y="y",  figsize=(9,9))
    # title = 'initial bw:' + str(initial_bandwidth)
    # plt.title(title)
    # plt.show()
    hb_dict = {}
    hb_dict['hammerbundle'] = hb
    hb_dict['milestone_embedding'] = embedding
    hb_dict['edges'] = edges[['source', 'target']]
    return hb_dict


def dist_points(v1, v2):
    # euclidean distance between two points (x,y) (x1,y1)
    x = (v1[0] - v2[0]) * (v1[0] - v2[0])
    y = (v1[1] - v2[1]) * (v1[1] - v2[1])
    return math.sqrt(x + y)


def infer_direction_piegraph(start_node, end_node, CSM, velocity_weight, pt, tanh_scaling_factor=1):
    '''
    infers directionality between start and end node using both the velocity metric and pseudotime change.
    The Level of influence from pseudotime vs. velocity is based on the velocity weight
    :param start_node: the start node of the edge
    :param end_node: the end node of the end
    :param CSM: Cosine similarity matrix. i,j entry is the Cosine similarity between the velocity vector of the i'th
            cell and the change in spliced gene expression from cell i to j
    :param velocity_weight: parameter between 0-1, 0 signifies 0 weight of velocity (CSM matrix)
    :param pt: the cluster level pseudotimes
    :param tanh_scaling_factor: tanh(kx), default k=1. slope of the tanh curve
    :return: value between -1 and 1. a negative value signifies that the start and end should be swapped. start, end = end, start
    '''
    if CSM is None:
        velocity_weight = csm_es = csm_se = 0
    else:
        csm_se = 3 * CSM[start_node, end_node]  # cosine similarity from start-to-end
        csm_es = 3 * CSM[end_node, start_node]  # cosine similarity from end-to-start

        # print('csm_start', csm_se)
        # print('csm_end', -1 * csm_es)
        # print('csm', csm_se + -1 * csm_es)

        # Note csm_es does not equal -csm_se because the velocity vector used in the dot product refers to the originating cell

    mpt = max(pt)
    pt = [i * 3 / mpt for i in pt]  # bring the values to 0-3 so that the tanh function has a range of values

    # print('pt of end node', end_node, round(pt[end_node], 2), 'pt of start node', start_node, round(pt[start_node], 2))
    tanh_ = math.tanh((pt[end_node] - pt[start_node]) * tanh_scaling_factor)
    # print('tanh', tanh_)
    direction = velocity_weight * 0.5 * (csm_se + (-1 * csm_es)) + (1 - velocity_weight) * tanh_
    # print('direction for start',start_node,' to end node',end_node,'is', direction)
    return direction


def get_projected_distances(loadings, gene_matrix, velocity_matrix, edgelist, current_pc):
    '''
    # revise the distance between the start cell and the neighbor based on expected location of the neighbor after one step in velocity/pseudotime
    # the start cell location is given by the "projected" distance implied by the velocity, and the end cell location is based on the current gene based location in PCA space
    # based on idea described in Veloviz (Atta Lyla 2022)
    :param loadings: PCA loadings adata.varm['PCs']
    :param gene_matrix: single-cell gene matrix of the filtered genes
    :param velocity_matrix: single-cell velocity from sc-Velo/veloctyo
    :param edgelist: list of tuples of graph (start, end)
    :param current_pc: PCs of the current gene space
    :return: ndarray
    '''
    # loadings = adata.varm['PCs']
    # print('loadings shape', loadings.shape)
    proj = gene_matrix + velocity_matrix

    proj[proj < 0] = 0
    # print('inside projected distances')
    # print(gene_matrix[0:3,0:20])#pcs one step ahead

    proj -= np.mean(proj, axis=0)
    proj[proj < 0] = 0
    # print('size projected', proj.shape)
    proj = np.matmul(proj, loadings)
    # print('size proj-pca', proj.shape)
    # checking computation approach used for proj. by using the same approach on the gene matrix without adding velocity, we see if we get the current PCs
    curr_loading_pca = gene_matrix - np.mean(gene_matrix, axis=0)
    curr_loading_pca = np.matmul(curr_loading_pca, loadings)

    new_edgelist = []
    closer = 0
    farther = 0
    for (s, e) in edgelist:
        # dist_prime = np.linalg.norm(current_pc[s,:]- proj[e,:]) #revise the distance between start-cell and it's neighbor based on expected location of the neighbor after one step in velocity/pseudotime
        dist_prime = np.linalg.norm(proj[s, :] - current_pc[e, :])
        new_edgelist.append(dist_prime)
        '''
        if s==0: #just to see what is happening when you shift the PCs
            print('regular pca dist: start-end',s,'-',e, round(np.linalg.norm(current_pc[s, :] - current_pc[e, :]),2))#np.linalg.norm(current_pc[s,:]- current_pc[e,:])
            print('regular pca dist using loadings: start-end', s, '-', e, round(np.linalg.norm(curr_loading_pca[e,:]- curr_loading_pca[s,:]),2))#np.linalg.norm(curr_loading_pca[s,:]- curr_loading_pca[e,:])
            print('projected pca dist: start-end', s, '-', e, round(dist_prime,2))
            if dist_prime<np.linalg.norm(current_pc[e,:]- current_pc[s,:]):
                print('proj is closer')
                closer +=1
            else: farther +=1
        '''
    # print('distance between projected start and current neighbor location closer', closer)
    # print('farther', farther)

    weights = np.array(new_edgelist)
    weights = np.clip(weights, a_min=np.percentile(weights, 10),
                      a_max=None)  # the first neighbor is usually itself and hence has distance 0

    # print('min clip weights in utils get projected distsances', np.percentile(weights, 10))
    # print('clipped weights for projected distances', weights)
    # scaled to between 0.5 and 2
    weights = 0.5 + (weights - np.min(weights)) * (2 - 0.5) / (np.max(weights) - np.min(weights))
    weights = 1 / weights
    return weights


def stationary_probability_naive(A_velo):
    '''
    Find the stationary probability of the cluster-graph transition matrix
    :param A_velo: transition matrix of cluster graph based on velocity (cluster level)
    :return: stationary probability of each cluster
    '''

    n_clus = A_velo.shape[0]
    A = np.append(np.transpose(A_velo) - np.identity(n_clus), [[1 for i in range(n_clus)]], axis=0)

    b = np.transpose(np.array([0 for i in range(n_clus + 1)]))
    b[n_clus] = 1
    lin_alg = np.linalg.solve(np.transpose(A).dot(A), np.transpose(A).dot(b))

    print(np.round(lin_alg, 3))
    return lin_alg


def stationary_probability_(A_velo):
    '''
    Find the stationary probability of the cluster-graph transition matrix
    :param A_velo: transition matrix of cluster graph based on velocity (cluster level)
    :return: array [n_clusters x 1] with stationary probability of each cluster, list of top 3 root candidates corresponding to cluster labels
    '''

    n = A_velo.shape[0]

    A_velo /= np.max(A_velo)
    for i, r in enumerate(A_velo):
        if np.all(r == 0):
            A_velo[i, i] = 1
    # ensure probabilities sum to 1 along the rows, normalize across the rows
    A_velo = A_velo / A_velo.sum(axis=1).reshape((n, 1))
    # print('A_velo normed', np.round(A_velo,2))
    # print('A_velo colsum',np.sum(A_velo, axis=1))

    # print('using eigenvalue decomp')
    D, V_l = np.linalg.eig(A_velo.T)
    # print('D, V_l', D, V_l)
    D_r, V_r = np.linalg.eig(A_velo)
    # print('D_r, V_R', D_r, V_r)
    # Sort the eigenvalues and eigenvectors and take the real part
    # print("Sorting eigenvalues by their real part")
    # print('D.real', np.round(D.real, 2))
    p = np.flip(np.argsort(D.real))
    # print('p sorted and flipped', p)
    D, V_l, V_r = D[p], V_l[:, p], V_r[:, p]

    pi = np.abs(V_l[:, 0].real)

    pi /= np.sum(pi)

    print(f"{datetime.now()}\tStationary distribution normed {np.round(pi, 3)}")
    sorted_pi = np.argsort(pi)
    velo_root_top10 = sorted_pi[0:10]
    print(
        f"{datetime.now()}\tTop 10 candidates for root: {np.round(sorted_pi[0:10], 2)} with stationary prob (%) {np.round(pi[sorted_pi[0:10]] * 100, 3)}")
    print(f"{datetime.now()}\tTop 5 candidates for terminal: {np.flip(sorted_pi)[0:5]}")

    return pi, velo_root_top10


def velocity_transition(A, V, G, slope=4):
    '''
    Reweighting the cluster level transition matrix based on the cosine similarities of velocity vectors.
    negative direction is suppresed using logistic function, and positive directions are emphasize using logistic function
    positive direction is when velocity relative to the change in gene expression from the ith cell to its neighbors in knn_gene graph (at cluster level) is also positive, and converse
    :param A: Adjacency of clustergraph
    :param V: velocity matrix, cluster average
    :param G: Gene expression matrix, cluster average
    :return A_velo: reweighted transition matrix of cluster graph [n_clusxn_clus], array of cosine similarities between clusters [n_clus x n_clus], list of top3 roots
    '''
    # from scipy.spatial import distance
    CSM = np.zeros_like(A)
    A_velo = np.zeros_like(A)
    for i in range(A.shape[0]):
        delta_gene = G - G[i, :]  # change in gene expression when going from i'th cell to other cells
        # print('shape delta_gene', delta_gene.shape, delta_gene)
        delta_gene = np.delete(delta_gene, obj=i, axis=0)  # delete ith row to avoid divide by zero in csm

        CSM_i = cosine_sim(delta_gene, V[i, :])  # the cosine similarity between delta_gene and the i'th cell's velocity
        # for j in range(3): print('scipy cosine similarity', 1-distance.cosine(delta_gene[j,:],V[i,:]))

        # A_velo_: convert the Cosine similarity to a weight between 0-1, such that edges going from i to j that have negative csm, have a low weight near 0
        # this means that rather than reverse the implied directionality of the edge, we make the transition very unlikely by lowering the weight in the new transition matrix. In this way we also ensure that the Markov process is irreducible
        A_velo_i = logistic_function(CSM_i, par_slope=slope)
        A_velo_i = np.insert(A_velo_i, obj=i, values=0)  # set the ith value to 0
        CSM_i = np.insert(CSM_i, obj=i, values=0)

        CSM[i, :] = CSM_i
        A_velo[i, :] = A_velo_i

    mask = A == 0  # identify the non-neighbors
    # CSM[mask] = 0 #remove non-neighbor edges
    A_velo[mask] = 0
    A_velo = np.multiply(A_velo,
                         A)  # multiply element-wise the edge-weight of the transition matrix A by the velocity-factor

    print(f"{datetime.now()}\tLooking for initial states")
    pi, velo_root_top10 = stationary_probability_(A_velo)

    return A_velo, CSM, velo_root_top10


def sc_CSM(A, V, G):
    '''
    :param A: single-cell csr knn graph with neighbors. [n_samples x knn] v0.self.full_csr_matrix
    :param V: cell x velocity matrix (dim: n_samples x n_genes)
    :param G: cell x genes matrix (dim: n_samples x n_genes)
    :return: single cell level cosine similarity between cells and neighbors of size [n_cells x knn]
    '''
    CSM = np.zeros_like(A)
    find_A = find(A)
    size_A = A.size

    for i in range(A.shape[0]):
        delta_gene = G - G[i, :]  # change in gene expression when going from i'th cell to other cells
        # print('shape delta_gene', delta_gene.shape, delta_gene)
        delta_gene = np.delete(delta_gene, obj=i,
                               axis=0)  # delete ith row to avoid divide by zero in cosine similarity calculation
        CSM_i = cosine_sim(delta_gene, V[i, :])  # the cosine similarity between delta_gene and the i'th cell's velocity
        # for j in range(3): print('scipy cosine similarity', 1-distance.cosine(delta_gene[j,:],V[i,:]))

        # A_velo_: convert the Cosine similarity to a weight between 0-1, such that edges going from i to j that have negative csm, have a low weight near 0
        # this means that rather than reverse the implied directionality of the edge, we make the transition very unlikely by lowering the weight in the new transition matrix.
        # A_velo_i = logistic_function(CSM_i, par_slope=4)
        # A_velo_i = np.insert(A_velo_i,obj=i,values=0)
        CSM_i = np.insert(CSM_i, obj=i, values=0)
        CSM[i, :] = CSM_i

    CSM_list = []

    for i in range(size_A):
        start = find_A[0][i]
        end = find_A[1][i]
        # weight = find_A[2][i]
        delta_gene = G[end, :] - G[start, :]  # change in gene expression when going from start cell to end cells
        # print('shape delta_gene', delta_gene.shape, delta_gene)
        CSM_list = CSM_list.append(cosine_sim(delta_gene, V[start,
                                                          :]))  # the cosine similarity between delta_gene and the i'th cell's velocity
        # for j in range(3): print('scipy cosine similarity', 1-distance.cosine(delta_gene[j,:],V[i,:]))

        # A_velo_: convert the Cosine similarity to a weight between 0-1, such that edges going from i to j that have negative csm, have a low weight near 0
        # this means that rather than reverse the implied directionality of the edge, we make the transition very unlikely by lowering the weight in the new transition matrix.
        # A_velo_i = logistic_function(CSM_i, par_slope=4)
        # A_velo_i = np.insert(A_velo_i,obj=i,values=0)
    CSM = csr_matrix((CSM_list, (np.array(find_A[0]), np.array(find_A[1]))),
                     shape=size_A)

    return CSM


def interpolate_stream(array):
    from scipy import interpolate
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    # mask invalid values
    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)
    # get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    array_corrected = interpolate.griddata((x1, y1), newarr.ravel(),
                                           (xx, yy),
                                           method='cubic')
    return array_corrected


def interpolate_density(a, density_factor=2):
    from scipy.interpolate import UnivariateSpline
    old_indices = np.arange(0, a.size)  # len(a)
    new_length = int(len(a) * density_factor)
    new_indices = np.linspace(0, len(a) - 1, new_length)
    spl = UnivariateSpline(old_indices, a, k=3, s=0)
    new_array = spl(new_indices)
    return new_array


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> y= array([1, 1, 1, NaN, NaN, 2, 2, NaN, 0])
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])

    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def quiver_autoscale(X_emb, V_emb):
    import matplotlib.pyplot as pl

    scale_factor = np.abs(X_emb).max()  # just so that it handles very large values
    fig, ax = pl.subplots()
    Q = ax.quiver(
        X_emb[:, 0] / scale_factor,
        X_emb[:, 1] / scale_factor,
        V_emb[:, 0],
        V_emb[:, 1],
        angles="xy",
        scale_units="xy",
        scale=None,
    )
    Q._init()
    fig.clf()
    pl.close(fig)
    return Q.scale / scale_factor


def compute_velocity_on_grid(
        X_emb,
        V_emb,
        density=None,
        smooth=None,
        n_neighbors=None,
        min_mass=None,
        autoscale=True,
        adjust_for_stream=True,
        cutoff_perc=None,
):
    # adapted from scVelo Volker Bergen Nature Biotechnology 2020
    # print(X_emb.shape, V_emb.shape)

    # remove invalid cells
    idx_valid = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb = X_emb[idx_valid]
    V_emb = V_emb[idx_valid]

    # prepare grid
    n_obs, n_dim = X_emb.shape
    density = 1 if density is None else density
    smooth = 0.5 if smooth is None else smooth  # 0.5

    grs = []
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(50 * density))
        grs.append(gr)
    # print('grs', grs)
    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T

    # estimate grid velocities
    if n_neighbors is None:
        n_neighbors = min(int(n_obs / 50), 20)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X_emb)
    dists, neighs = nn.kneighbors(X_grid)

    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth  # diagonal distance of a grid-square

    weight = normal.pdf(x=dists, scale=scale)

    p_mass = weight.sum(1)

    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1)
    # print("V_grid in utils before norm")
    # print(V_grid)
    # print('V_grid intermediate 1', V_grid)
    V_grid /= np.maximum(1, p_mass)[:, None]

    # print('V_grid /= p_mass', V_grid)
    if min_mass is None:
        min_mass = 1

    if adjust_for_stream == True:

        X_grid = np.stack([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])])
        ns = int(np.sqrt(len(V_grid[:, 0])))
        V_grid = V_grid.T.reshape(2, ns, ns)

        mass = np.sqrt((V_grid ** 2).sum(0))
        min_mass = 10 ** (min_mass - 6)  # default min_mass = 1e-5
        min_mass = np.clip(min_mass, None, np.max(mass) * 0.9)

        # min_mass = np.clip(0, None, np.max(mass) * 0.9999)
        cutoff = mass.reshape(V_grid[0].shape) < min_mass

        if cutoff_perc is None:
            cutoff_perc = 5
        length = np.sum(np.mean(np.abs(V_emb[neighs]), axis=1), axis=1).T
        length = length.reshape(ns, ns)
        cutoff |= length < np.percentile(length, cutoff_perc)

        V_grid[0][cutoff] = np.nan
    else:

        min_mass *= np.percentile(p_mass, 99) / 100

        X_grid, V_grid = X_grid[p_mass > min_mass], V_grid[p_mass > min_mass]

        # if autoscale:            V_grid /= 3 * quiver_autoscale(X_grid, V_grid)
    return X_grid, V_grid


def map_velocity_to_color(X1D, Y1D, U, V, segments):
    '''

    :param X: nx1 array of x-coors of grid
    :param Y: nx1 array of y-coors of grid
    :param U: nxn array of u-velocity on X-Y grid
    :param V:
    :return:
    '''
    # reshape U and V so we can match coords with velocities
    print(U, V)
    U_ = U.reshape((np.prod(U.shape),))
    V_ = V.reshape((np.prod(V.shape),))
    velo_coords = np.vstack((U_, V_)).T
    # velo_coords = velo_coords / velo_coords.max(axis=0)
    print('velo coords', velo_coords.shape)
    print(velo_coords)
    # reshape meshgrid to extract coords
    X, Y = np.meshgrid(X1D, Y1D)

    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))

    coords = np.vstack((X, Y)).T  # (nx2) array of grid coords
    neigh_graph = NearestNeighbors(n_neighbors=1)
    neigh_graph.fit(coords)

    seg_coors = np.squeeze(segments[:, ::2, :])

    neigh_seg = neigh_graph.kneighbors(seg_coors, return_distance=False)  # [0]

    n = len(segments)
    u_seg = velo_coords[neigh_seg, 1]

    print(u_seg)
    C = np.zeros((n, 4))
    C[::-1] = 1 - np.clip(u_seg, 0.01, 1)

    return C


def interpolate_static_stream(x, y, u, v):
    from scipy.interpolate import griddata
    x, y = np.meshgrid(x, y)

    # print(x, y)
    points = np.array((x.flatten(), y.flatten())).T
    u = np.nan_to_num(u.flatten())
    v = np.nan_to_num(v.flatten())
    xi = np.linspace(x.min(), x.max(), 25)
    yi = np.linspace(y.min(), y.max(), 25)
    X, Y = np.meshgrid(xi, yi)

    U = griddata(points, u, (X, Y), method='cubic')
    V = griddata(points, v, (X, Y), method='cubic')

    return X, Y, U, V


def l2_norm(x: Union[ndarray, spmatrix], axis: int = 1) -> Union[float, ndarray]:
    """Calculate l2 norm along a given axis.
    Arguments
    ---------
    x
        Array to calculate l2 norm of.
    axis
        Axis along which to calculate l2 norm.
    Returns
    -------
    Union[float, ndarray]
        L2 norm along a given axis.
    """

    if issparse(x):
        return np.sqrt(x.multiply(x).sum(axis=axis).A1)
    elif x.ndim == 1:
        return np.sqrt(np.einsum("i, i -> ", x, x))
    elif axis == 0:
        return np.sqrt(np.einsum("ij, ij -> j", x, x))
    elif axis == 1:
        return np.sqrt(np.einsum("ij, ij -> i", x, x))


def _median_min_distance(neighbors_distances):
    """This function computes a graph of nearest-neighbors for each sample point in
        'data' and returns the median of the distribution of distances between those
        nearest-neighbors, the distance metric being specified by 'metric'.

    Parameters
    ----------
    data : array of shape (n_samples, k_neighbors)
        The data-set, a fraction of whose sample points will be extracted
        by density sampling.


    Returns
    -------
    median_min_dist : float
        The median of the distribution of distances between nearest-neighbors.
    """

    median_min_dist = np.median(neighbors_distances, overwrite_input=True)
    p1 = np.percentile(neighbors_distances, 10)
    p2 = np.percentile(neighbors_distances, 25)
    p3 = np.percentile(neighbors_distances, 75)
    p4 = np.percentile(neighbors_distances, 90)
    print(
        f'median, p1 to p4 {round(median_min_dist, 4)}, {round(p1, 4)}, {round(p2, 4)}, {round(p3, 4)}, {round(p4, 4)}')
    return round(median_min_dist, 4)


def _get_local_densities(neighbors_distances, kernel_mult=1.0):
    """For each sample point of the data-set 'data', estimate a local density in feature
        space by counting the number of neighboring data-points within a particular
        region centered around that sample point.

    Parameters
    ----------
    neighbor_distances : array of shape (n_samples, n_features)
        The data-set, a fraction of whose sample points will be extracted
        by density sampling.

    kernel_mult : float, optional (default = 2.0)
        The kernel multiplier, which determine (in terms of the median of the distribution
        of distances among nearest neighbors) the extent of the regions centered
        around each sample point to consider for the computation of the local density
        associated to that particular sample point.


    Returns
    -------
    local_densities : array of shape (n_samples,)
        The i-th entry of this vector corresponds to the local density of the i-th sample
        point in the order of the rows of 'data'.
    """

    kernel_width = kernel_mult * _median_min_distance(neighbors_distances=neighbors_distances)
    print('kernel multiplier', kernel_mult)
    print(f'kernel width {kernel_width}')

    N_samples = neighbors_distances.shape[0]

    # local_densities = np.zeros(N_samples, dtype=int)

    D = (neighbors_distances <= kernel_width)

    local_densities = D.sum(axis=1)
    print(f'local densities shape and values {local_densities.shape} {local_densities}')
    return local_densities


def density_sampling(neighbors_distances, local_densities=None,
                     kernel_mult=1.5, outlier_percentile=0.005,
                     target_percentile=0.05, desired_samples=None):
    """The i-th sample point of the data-set 'data' is selected by density sampling
            with a probability given by:

                                          | 0 if outlier_density > LD[i];
            P(keep the i-th data-point) = | 1 if outlier_density <= LD[i] <= target_density;
                                          | target_density / LD[i] if LD[i] > target_density.

            Here 'LD[i]' denotes the local density of the i-th sample point of the data-set,
            whereas 'outlier_density' and 'target_density' are computed as particular percentiles
            of that distribution of local densities.

        Parameters
        ----------
        data : array of shape (n_samples, n_features)
            The data-set, a fraction of whose sample points will be extracted
            by density sampling.

        local_densities : array of shape (n_samples,), optional (default = None)
            The i-th entry of this vector corresponds to the local density of the i-th sample
            point in the order of the rows of 'data'.

        kernel_mult : float, optional (default = 2.0)
            The kernel multiplier, which determine (in terms of the median of the distribution
            of distances among nearest neighbors) the extent of the regions centered
            around each sample point to consider for the computation of the local density
            associated to that particular sample point.

        outlier_percentile : float, optional (default = 0.01)
            Specify the outlier density as a percentile of the distribution of local densities.

        target_percentile : float, optional (default = 0.05)
            Specifiy the target density as a percentile of the distribution of local densities.
            Relevant only if 'desired_samples' is left unspecified.

        desired_samples : int, optional (default = None)
            The number of samples to be selected from the whole data-set such that members
            of rare populations and members of more common populations are roughly
            equally represented. To that purpose, a target density is computed that to selects about
            'desired_samples' data-points.

        Returns
        -------
        samples_kept : array of shape (n_selected_samples,)
            If the 'i'-th sample point of 'data' has been selected by a given instance of
            density sampling, number 'i' is featured in the array returned by
            the present function.
        """

    random_state = np.random.RandomState()

    if local_densities is None:
        local_densities = _get_local_densities(neighbors_distances, kernel_mult)

    outlier_density = np.percentile(local_densities, outlier_percentile)
    target_density = np.percentile(local_densities, target_percentile)

    samples_kept = np.where(local_densities > outlier_density)[0]
    N_kept = samples_kept.size
    print(f'{N_kept} cells are greater than outlier density')

    local_densities = local_densities[samples_kept]

    if desired_samples is None:
        probs = np.divide(target_density + 0.0, local_densities)
        ind = np.where(probs > random_state.uniform(size=N_kept))[0]
        samples_kept = samples_kept[ind]
    elif desired_samples <= N_kept:
        sorted_densities = np.sort(local_densities)

        temp = np.reciprocal(sorted_densities[::-1].astype(float))
        cdf = np.cumsum(temp)[::-1]

        target_density = (desired_samples + 0.0) / cdf[0]
        if target_density > sorted_densities[0]:
            temp = desired_samples - np.arange(1.0, N_kept + 1.0)
            possible_targets = np.divide(temp, cdf)

            ind = np.argmax(possible_targets < sorted_densities)
            target_density = possible_targets[ind]

        probs = np.divide(target_density + 0.0, local_densities)
        ind = np.where(probs > random_state.uniform(size=N_kept))[0]
        samples_kept = samples_kept[ind]
    else:
        print("\nERROR: Density_Sampling: density_sampling: 'desired_samples' has been "
              "assigned a value of {desired_samples}, larger than {N_kept}, "
              "the number of samples whose local densities are high enough "
              "(i.e. excluded are the local densities in the lowest {outlier_percentile} "
              "percentile).\n".format(**locals()))
        kernel_mult += 0.25
        print(f'increasing kernel multiplier by 0.25, {kernel_mult}')
        samples_kept = density_sampling(neighbors_distances=neighbors_distances, kernel_mult=kernel_mult,
                                        desired_samples=desired_samples)
    print(f'samples_kept are: {samples_kept.shape}, {samples_kept}')
    return samples_kept


def accuracy(onevsall=1, true_label=[], cluster_label=[]):
    true_labels = true_label
    Index_dict = {}
    PARC_labels = cluster_label
    N = len(PARC_labels)
    n_cancer = list(true_labels).count(onevsall)
    n_pbmc = N - n_cancer

    for k in range(N):
        Index_dict.setdefault(PARC_labels[k], []).append(true_labels[k])
    num_groups = len(Index_dict)
    sorted_keys = list(sorted(Index_dict.keys()))
    error_count = []
    pbmc_labels = []
    thp1_labels = []
    fp, fn, tp, tn, precision, recall, f1_score = 0, 0, 0, 0, 0, 0, 0

    for kk in sorted_keys:
        vals = [t for t in Index_dict[kk]]
        majority_val = func_mode(vals)
        # if majority_val == onevsall: print('cluster', kk, ' has majority', onevsall, 'with population', len(vals))
        if kk == -1:
            len_unknown = len(vals)
            # print('len unknown', len_unknown)
        if (majority_val == onevsall) and (kk != -1):
            thp1_labels.append(kk)
            fp = fp + len([e for e in vals if e != onevsall])
            tp = tp + len([e for e in vals if e == onevsall])
            list_error = [e for e in vals if e != majority_val]
            e_count = len(list_error)
            error_count.append(e_count)
        elif (majority_val != onevsall) and (kk != -1):
            pbmc_labels.append(kk)
            tn = tn + len([e for e in vals if e != onevsall])
            fn = fn + len([e for e in vals if e == onevsall])
            error_count.append(len([e for e in vals if e != majority_val]))

    predict_class_array = np.array(PARC_labels)
    PARC_labels_array = np.array(PARC_labels)
    number_clusters_for_target = len(thp1_labels)
    for cancer_class in thp1_labels:
        predict_class_array[PARC_labels_array == cancer_class] = 1
    for benign_class in pbmc_labels:
        predict_class_array[PARC_labels_array == benign_class] = 0
    predict_class_array.reshape((predict_class_array.shape[0], -1))
    error_rate = sum(error_count) / N
    n_target = tp + fn
    tnr = tn / n_pbmc
    fnr = fn / n_cancer
    tpr = tp / n_cancer
    fpr = fp / n_pbmc

    if tp != 0 or fn != 0: recall = tp / (tp + fn)  # ability to find all positives
    if tp != 0 or fp != 0: precision = tp / (tp + fp)  # ability to not misclassify negatives as positives
    if precision != 0 or recall != 0:
        f1_score = precision * recall * 2 / (precision + recall)

    majority_truth_labels = np.empty((len(true_labels), 1), dtype=object)
    for cluster_i in set(PARC_labels):
        cluster_i_loc = np.where(np.asarray(PARC_labels) == cluster_i)[0]
        true_labels = np.asarray(true_labels)
        majority_truth = func_mode(list(true_labels[cluster_i_loc]))
        majority_truth_labels[cluster_i_loc] = majority_truth

    majority_truth_labels = list(majority_truth_labels.flatten())
    accuracy_val = [error_rate, f1_score, tnr, fnr, tpr, fpr, precision,
                    recall, num_groups, n_target]

    return accuracy_val, predict_class_array, majority_truth_labels, number_clusters_for_target


def main_accuracy(true_label=[], cluster_label=[], filename='/home/shobi/Trajectory/df_accuracy.csv'):
    targets = list(set(true_label))
    targets.sort()
    N = len(true_label)
    f1_accumulated = 0
    f1_mean = 0

    # self.majority_truth_labels = []
    list_roc = []
    if len(targets) > 1:
        f1_accumulated, f1_acc_noweighting = 0, 0
        for onevsall_val in targets:
            # print('target is', onevsall_val)
            vals_roc, predict_class_array, majority_truth_labels, numclusters_targetval = \
                accuracy(onevsall=onevsall_val, true_label=true_label, cluster_label=cluster_label)
            f1_current = vals_roc[1]
            f1_accumulated = f1_accumulated + f1_current * (list(true_label).count(onevsall_val)) / N
            f1_acc_noweighting = f1_acc_noweighting + f1_current

            list_roc.append([onevsall_val] +
                            vals_roc + [vals_roc[-1] / N] + [numclusters_targetval])

        f1_mean = f1_acc_noweighting / len(targets)

        df_accuracy = pd.DataFrame(list_roc,
                                   columns=['onevsall-target', 'error rate',
                                            'f1-score', 'tnr', 'fnr',
                                            'tpr', 'fpr', 'precision', 'recall', 'num_groups',
                                            'population of target', 'percentage of population', 'num clusters'])
        print(df_accuracy.head)
        print(df_accuracy.shape)
        df_accuracy.to_csv(filename)
        return df_accuracy


'''
def rw2_emb(filename='/home/user/Trajectory/Datasets/'):
    from pecanpy import pecanpy as node2vec
    #https://pecanpy.readthedocs.io/en/latest/pecanpy.html#pecanpy.pecanpy.Base.embed
    # initialize node2vec object, similarly for SparseOTF and DenseOTF
    g = node2vec.SparseOTF(p=0.5, q=1, workers=4, verbose=True, extend=True)
    # alternatively, can specify ``extend=True`` for using node2vec+

    # load graph from edgelist file
    g.read_npz(filename+'.npz', weighted=True)


    # generate random walks, which could then be used to train w2v
    #walks = g.simulate_walks(num_walks=10, walk_length=80)

    # alternatively, generate the embeddings directly using ``embed``
    print(f'{datetime.now()}\tStart g.embed')
    emd = g.embed(num_walks=20, walk_length=80, dim=64) #dim default = 128
    print(f'{datetime.now()}\tEnd g.embed and save csv')
    df=pd.DataFrame(emd)
    df.to_csv(filename+'csv')
    return
'''


def save_sparse_knn_as_npz(via_object, working_directory: str = None):
    '''

    :param via_object:
    :param working_directory: defaults to working directory pathway stored in via_object.working_dir_fp. e.g.  working_dir_fp='/home/user/Trajectory/Zebrafish/'
    :return:
    '''
    import random
    pseudorand = random.randint(0, 1000)
    print("saving csv for RW2")
    from scipy.sparse import save_npz

    print(f'Unique ID for RW2 file {pseudorand}')

    fname = working_directory + 'pc' + str(via_object.ncomp) + '_knn' + str(via_object.knn) + 'rw2_' + str(pseudorand)
    save_npz(fname + '.npz', via_object.csr_full_graph)
    print('npz saved to', fname + '.npz')
    return fname


def rw2_feature_representation(via_object, filepath_npz: str = None,
                               working_directory: str = '/home/user/Trajectory/Datasets/', n_dim: int = 128,
                               p_return: float = 0.5, q_inward=1, ):
    '''
    computes the graph feature representation based on Pecaypy implementation of Node2Vec and saves it as a csv file using the same filename
    :param via_object: required if this is the first time the sc-knn graph is being saved into an npz file which is the format required to run the feature representation
    :param filename: filepath and name of npz file if one exists already. otherwise provide via_object to save an npz file of the single-cell graph. These are like PCs for graph networks. It's best to compute it once and save e.g. 128 features that you can use at any time
    :param n_dim: number of features (components) default = 128. other values": 32,64,128
    :param p_inward. larger number means more exploration
    :param q_inward. larger number means more introspective search
    :return:filepath_npz (name of csv filepath for features), feature_rep ndarray (n_cellsxn_dim)
    '''
    from pecanpy import pecanpy as node2vec
    print(
        f'{datetime.now()}\tStart feature representation computations')
    # initialize node2vec object, similarly for SparseOTF and DenseOTF

    g = node2vec.SparseOTF(p=p_return, q=q_inward, workers=4, verbose=True, extend=True)
    # g = node2vec.SparseOTF(p=1, q=0.01, workers=4, verbose=True, extend=True)
    # alternatively, can specify ``extend=True`` for using node2vec+

    if filepath_npz is None:
        # save down the knn graph as an npz file
        print(f'working directory is {working_directory}')
        filepath_npz = save_sparse_knn_as_npz(via_object, working_directory)
    # load graph from edgelist file
    g.read_npz(filepath_npz + '.npz', weighted=True)
    # precompute and save 2nd order transition probs (for PreComp only)
    # g.preprocess_transition_probs()

    # generate random walks, which could then be used to train w2v
    # walks = g.simulate_walks(num_walks=10, walk_length=80)

    # alternatively, generate the embeddings directly using ``embed``
    print(f'{datetime.now()}\tStart g.embed. takes 5-8 mins for 100K cells - saving down takes a few mins. ')
    print(f'{datetime.now()}\tSuffice to run this once and going forward load the saved .csv file with the features')
    feature_rep = g.embed(num_walks=20, walk_length=80, dim=n_dim)  # dim default = 128
    filepath_csv = filepath_npz + '_ndim' + str(n_dim) + '_qin' + str(q_inward) + '_pret' + str(p_return)
    print(f'{datetime.now()}\tEnd feature representation computations and save csv to file {filepath_csv}.csv')
    print(f'{datetime.now()}\tGoing forward these features can be used as X_input in the via_atlas_emb() function')
    df = pd.DataFrame(feature_rep)

    df.to_csv(filepath_csv + '.csv')
    return filepath_npz, feature_rep


def lineage_corr(marker_lineages: list = [], list_filenames: list = [], cmap='Blues', fontscale: float = 0.75):
    import seaborn as sns
    from matplotlib.colors import ListedColormap
    sns.set(font_scale=fontscale)
    n_terminal_clusters = len(marker_lineages)
    fig_ncols = min(3, n_terminal_clusters)
    fig_nrows, mod = divmod(n_terminal_clusters, fig_ncols)
    if mod == 0:
        if fig_nrows == 0:
            fig_nrows += 1
        else:
            fig_nrows = fig_nrows
    if mod != 0:        fig_nrows += 1
    ti = 0
    fig, ax = plt.subplots(fig_nrows, fig_ncols)
    for r in range(fig_nrows):
        for c in range(fig_ncols):
            if ti < n_terminal_clusters:
                marker_lineage_i = marker_lineages[ti]
                df_store = pd.DataFrame()
                for f in list_filenames:
                    df_ = pd.read_csv(f + '.csv')
                    memvalue = f.partition('memory')[2]
                    column_name = str(memvalue)
                    df_store[column_name] = df_[marker_lineage_i]
                corrM = df_store.corr()

                print(corrM)
                # Plot correlation matrix

                # ax=sns.heatmap(corrM, annot=True, cmap=cmap)
                dict_ = {0: 'cartilage', 2: 'dermis', 1: 'forebrain', 136: 'RGC',
                         145: 'musculature', 37: 'erythrocyte',
                         102: 'vasculature', 105: 'oligodendrocyte', 55: 'myotome'}

                if fig_nrows == 1:
                    if fig_ncols == 1:
                        mask = np.triu(np.ones(corrM.shape)).astype(bool)

                        sns.heatmap(corrM, ax=ax, mask=mask, cmap=cmap, fmt='', square=True, linewidths=1)
                        mask = np.ones((corrM.shape[0], corrM.shape[0])) - mask

                        ax = sns.heatmap(corrM, ax=ax, mask=mask, cmap=ListedColormap(['white']), cbar=False, fmt='',
                                         linewidths=1)
                        ax.set_title('Lineage:' + str(marker_lineage_i))

                    else:
                        mask = np.triu(np.ones(corrM.shape)).astype(bool)
                        sns.heatmap(corrM, ax=ax[c], mask=mask, cmap=cmap, fmt='', square=True, linewidths=1)
                        mask = np.ones((corrM.shape[0], corrM.shape[0])) - mask
                        sns.heatmap(corrM, ax=ax[c], mask=mask, cmap=ListedColormap(['white']), cbar=False, fmt='',
                                    linewidths=1)
                        ax[c].set_title('Lineage:' + str(marker_lineage_i))
                else:
                    mask = np.triu(np.ones(corrM.shape)).astype(bool)
                    sns.heatmap(corrM, ax=ax[r, c], mask=mask, cmap=cmap, fmt='', square=True, linewidths=1)
                    mask = np.ones((corrM.shape[0], corrM.shape[0])) - mask
                    sns.heatmap(corrM, ax=ax[r, c], mask=mask, cmap=ListedColormap(['white']), cbar=False, fmt='',
                                linewidths=1, xticklabels=True, yticklabels=True)

                    n, m = corrM.shape
                    val_ = corrM.values[np.tri(N=n, M=m, k=-1, dtype=bool)]
                    ax[r, c].set_title(
                        dict_[int(marker_lineage_i)] + ' Max:' + str(round(np.max(val_), 2)) + '/Med:' + str(
                            round(np.median(val_), 2)))  #
                    # ax[r, c].set_title('Lineage:' +str(marker_lineage_i) + ' Max:' + str(                        round(np.max(val_), 2)) + '/Med:' + str(round(np.median(val_), 2)))

                    # print(val_)
                    print(
                        f'marker lineage:{dict_[int(marker_lineage_i)]} Max = {str(round(np.max(val_), 2))} /Med:  {str(round(np.median(val_), 2))}')

            if fig_nrows == 1:
                if fig_ncols == 1:
                    if ti >= n_terminal_clusters: ax.axis('off')
                    ax.grid(False)
                else:
                    if ti >= n_terminal_clusters: ax[c].axis('off')
                    ax[c].grid(False)
            else:
                if ti >= n_terminal_clusters: ax[r, c].axis('off')
                ax[r, c].grid(False)
            ti += 1
    plt.show()
    return
