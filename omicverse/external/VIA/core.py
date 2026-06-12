import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix, csgraph, find
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
import igraph as ig
import leidenalg
import time
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.path import get_path_collection_extents
import multiprocessing
from collections import Counter
from typing import Optional, Union

from .plotting_via import *
from .plotting_via_ov import via_streamplot_ov,plot_trajectory_curves_ov,plot_piechart_viagraph_ov,animate_streamplot_ov,animate_atlas_ov,plot_atlas_view_ov
from .utils_via import *
from .utils_via import _construct_knn, sequential_knn

#from plotting_via import *
#from utils_via import *
#from utils_via import _construct_knn, sequential_knn
from sklearn.preprocessing import normalize
import math



### to upload to pip core via2 file


def _composite_jacAffinity_distanceAffinity(sc_csr_graph: csr_matrix, projected_distances: ndarray = None):
    '''
    make a composite score for distance based affinity and  jaccard score
    :param sc_csr_graph: distances based affinities (.data is affinity)
    :param projected_distances: if using projected distances adjusted based on velocity then provide them
    :return: igraph of affinities based on Jaccard and distance-based affinities
    '''
    edges_ = list(zip(*sc_csr_graph.nonzero()))  # in the case of time-series, this is the augmented graph
    # Jaccard computation in igraph does not consider weights. However, doing the global pruning on distance based weights seems less effective than pruning on Jaccard similarities
    jac_edgeweights_list = ig.Graph(edges_, edge_attrs={'weight': sc_csr_graph.data}).similarity_jaccard(
        pairs=edges_)

    # the graph has self loops, so we need to clip these values as they have a jacc of 1, and much higher than other neighbors
    jac_edgeweights_list = np.clip(np.array(jac_edgeweights_list), a_max=np.percentile(np.array(jac_edgeweights_list),
                                                                                       95), a_min=None).tolist()

    if projected_distances is None:
        composite_edgeweights = np.multiply(sc_csr_graph.data, np.array(
            jac_edgeweights_list))  # 0<J<1 #combining distance-based affinities and jac-based affinities
    else:

        composite_edgeweights = np.multiply(projected_distances, np.array(
            jac_edgeweights_list))

    composite_igraph = ig.Graph(edges_, edge_attrs={'weight': composite_edgeweights.tolist()}).simplify(
        combine_edges='sum')  # used for clustergraph #in the case of time-series, this is the augmented graph

    return composite_igraph


def _get_loc_terminal_states(via0, X_input: ndarray):
    '''

    we need the location of terminal states from first iteration (Via0) to pass onto the second iterations of Via (Via1)
    this will allow identification of the terminal-cluster in fine-grained Via1 that best captures the terminal state from coarse Via0

    :param via0: coarse grained iteration of object class VIA
    :param X_input: via0.data. the data matrix (PCs) on which the TI is inferred
    :return:
    '''

    tsi_list = []  # find the single-cell which is nearest to the average-location of a terminal cluster in PCA space
    for tsi in via0.terminal_clusters:
        loc_i = np.where(np.asarray(via0.labels) == tsi)[0]
        val_pt = [via0.single_cell_pt_markov[i] for i in loc_i]
        th_pt = np.percentile(val_pt, 50)  # 50
        loc_i = [loc_i[i] for i in range(len(val_pt)) if val_pt[i] >= th_pt]
        temp = np.mean(X_input[loc_i], axis=0)
        labelsq, distances = via0.knn_struct.knn_query(temp, k=1)
        tsi_list.append(labelsq[0][0])
    return tsi_list

def _prob_reaching_terminal_state(terminal_state, A, root, n_simulations, q,
                                 cumstateChangeHist, cumstateChangeHist_all, seed):
    # this function is defined outside the VIA class to enable smooth parallel processing in Windows

    n = A.shape[0]

    A /= np.max(A)
    for i, r in enumerate(A):
        if np.all(r == 0):
            A[i, i] = 1
    # ensure probabilities sum to 1 along the rows
    P = A / A.sum(axis=1).reshape((n, 1))

    n_steps, count_reach_terminal_state = 2 * n, 0
    for _ in range(n_simulations):
        cur_state = root
        change_hist = np.zeros((n, n))
        change_hist[root, root] = 1

        x, terminal_state_found = 0, False

        while x < n_steps and not terminal_state_found:
            seed += 1
            np.random.seed(seed)
            next_state = np.random.choice(range(P.shape[0]), p=P[cur_state])
            if next_state == terminal_state:
                terminal_state_found = True

            change_hist[cur_state, next_state] += 1
            cur_state = next_state
            x += 1

        if terminal_state_found:
            cumstateChangeHist += np.any(change_hist > 0, axis=0)
            count_reach_terminal_state += 1
        cumstateChangeHist_all += np.any(change_hist > 0, axis=0)

    cumstateChangeHist_all[cumstateChangeHist_all == 0] = 1
    q.append([cumstateChangeHist, cumstateChangeHist_all])


def _rw2_walks(A: ndarray, root: int, memory: float = 0.8, weighted: bool = True, implicit_ids: bool = False,
              num_walks: int = 100, p_memory: float = 1.0, x_lazy: float = 0.95,
              alpha_teleport: float = 0.99):
    '''
    :param A: cluster level transition matrix (forward biased)
    :param root: root cluster index
    :param memory: 1/q * edge weight to a next-node that is not a neighbor of previous node. larger number means more memory and more introspective walk. small number <1 means more exploration
    :param p_memory: 1/p * edge weight to next node = previous node
    :param weighted: default True edge weights are provided as part of the array A
    :param walk_length: default None (walk_length = num_nodes *2)
    :return: implicit_ids
    '''
    from pecanpy.graph import AdjlstGraph
    # initialize SparseGraph object
    # g = AdjlstGraph()
    # setting up the lazy-teleporting behaviour
    n_states = A.shape[0]
    P = A / A.sum(axis=1).reshape((n_states, 1))
    # print('row normed P',P.shape, P, P.sum(axis=1))

    # bias_P is the transition probability matrix

    P = x_lazy * P + (1 - x_lazy) * np.identity(n_states)
    # print(P, P.sum(axis=1))
    P = alpha_teleport * P + ((1 - alpha_teleport) * (1 / n_states) * (np.ones((n_states, n_states))))

    from numba_progress import ProgressBar
    from pecanpy import pecanpy as node2vec
    A_csr = csr_matrix(A)  # csr_matrix(P) #use P if you want to add in the Lazy-teleporting. I find using A directly works a bit better.
    # print(f'A_csr {A_csr}')
    # print(f'A dense {A}')

    g = node2vec.SparseOTF(p=p_memory, q=memory, workers=4, verbose=True,
                           extend=True)  # larger q, more memory and introverted search #
    # using the extend=True which is the node2vec+ implementation from Jan 2023 Renming Liu Bioinformatics paper that differentiates between weakly connected (prev, next) and strongly connected (prev, next)
    # https://pecanpy.readthedocs.io/en/latest/pecanpy.html
    g.indptr = A_csr.indptr  # ["indptr"].astype(np.uint32)
    g.indices = A_csr.indices  # ["indices"].astype(np.uint32)
    g.data = A_csr.data  # ["data"].astype(np.float32)

    if g.data is None:
        raise ValueError("Adjacency matrix data not found.")
    elif not weighted:
        print('no weights so assign 1 to all edges')
        g.data[:] = 1.0  # overwrite edge weights with constant
    num_nodes = g.indptr.size
    walk_length = num_nodes * 2
    #print(f'g.indptr.size, {num_nodes}')
    # look deeper at how the node id mapping is done: https://pecanpy.readthedocs.io/en/latest/_modules/pecanpy/graph.html#DenseGraph.read_edg
    g.set_node_ids(node_ids=None,
                   implicit_ids=implicit_ids,
                   num_nodes=int(num_nodes - 1))

    # generate random walks, which could then be used to train w2v
    # walks = g.simulate_walks(num_walks=2, walk_length=3)
    # https://pecanpy.readthedocs.io/en/latest/_modules/pecanpy/pecanpy.html#Base.simulate_walks
    g._preprocess_transition_probs()

    nodes = np.array([root], dtype=np.uint32)  # range(g.num_nodes
    # print(f'nodes {range(g.num_nodes)} and nodes {nodes}')
    start_node_idx_ary = np.concatenate([nodes] * num_walks)  # root<=> nodes
    # print(f'start_node_array {start_node_idx_ary}')
    tot_num_jobs = start_node_idx_ary.size

    random_state = g.random_state
    np.random.seed(random_state)
    np.random.shuffle(start_node_idx_ary)  # for balanced work load

    move_forward = g.get_move_forward()
    has_nbrs = g.get_has_nbrs()
    verbose = g.verbose

    # Acquire numba progress proxy for displaying the progress bar
    with ProgressBar(total=tot_num_jobs, disable=not verbose) as progress:
        walk_idx_mat = g._random_walks(
            tot_num_jobs,
            walk_length,
            random_state,
            start_node_idx_ary,
            has_nbrs,
            move_forward,
            progress,
        )
    # print(f'walk mat {walk_idx_mat}')
    # Map node index back to node ID
    walks = [g._map_walk(walk_idx_ary) for walk_idx_ary in walk_idx_mat]
    walks = np.asarray(walks).astype(int)
    # print(f'via walks and shape of n_walks x n_steps {walks.shape}') #ndarray n_walks x n_steps

    # emd = g.embed(num_walks=5, walk_length=num_nodes*2)
    # print(f'emd {emd.shape} {emd}')
    return walks


def _compute_rw2_lineageprobability(A: ndarray, root: int, memory: float = 0.8, p_memory: float = 1.0,
                                   num_walks: int = 1000, terminal_states: list = None, x_lazy: float = 0.95,
                                   alpha_teleport: float = 0.99):
    print(f'{datetime.now()}\tCalculating lineage probability at memory {memory}')
    n_states = A.shape[0]
    n_terminal_states = len(terminal_states)
    ts_index_counter = 0
    prob_mat = np.zeros((n_states, n_terminal_states))
    walks = _rw2_walks(A=A, root=root, memory=memory, num_walks=num_walks, p_memory=p_memory, x_lazy=x_lazy,
                      alpha_teleport=alpha_teleport)

    # if n_steps is None: n_steps = 2 * n_states
    n_steps = walks.shape[1] - 1
    for terminal_state in terminal_states:

        cumstateChangeHist = np.zeros((1, n_states))
        cumstateChangeHist_all = np.zeros((1, n_states))
        terminal_state_found_inAllWalks = False

        count_reached_bool = 0
        count_reach_terminal_state = 0
        for row in walks:
            change_hist = np.zeros((n_states, n_states))
            change_hist[root, root] = 1
            x = 0
            # print(f'walk {row}')
            terminal_state_found = False
            while x < n_steps and not terminal_state_found:
                cur_state = row[x]
                next_state = row[x + 1]
                if next_state == terminal_state:
                    terminal_state_found = True
                    count_reached_bool += 1

                change_hist[cur_state, next_state] += 1
                x += 1

            if terminal_state_found:
                cumstateChangeHist += np.any(change_hist > 0, axis=0)
                count_reach_terminal_state += 1
            cumstateChangeHist_all += np.any(change_hist > 0, axis=0)
        # print(f'count reached bool {count_reached_bool}')
        cumstateChangeHist_all[
            cumstateChangeHist_all == 0] = 1  # to avoid division by zero we say that the state has been visited once
        count_reached = cumstateChangeHist_all[0, terminal_state]
        # print(f'terminal state HistAll {terminal_state} is reached {count_reached} times')
        count_reached = cumstateChangeHist[0, terminal_state]
        # print(f'terminal state {terminal_state} is reached {count_reached} times')
        # print( f"{datetime.now()}\tFrom root {root},  TS {terminal_state} is reached {int(count_reached)} times.")
        np.set_printoptions(precision=3)
        prob_ = cumstateChangeHist / cumstateChangeHist_all
        # print(f'cumstatechangeHist {cumstateChangeHist}')
        # print(f'cumstatechangeHist_all {cumstateChangeHist_all}')
        # print(f'prob_ pre scaling {prob_}')
        # given a cluster is traversed (cumhistory_vec_all), how many of these traversals led to a successful path to Terminal State X
        # i.e. with what probability does traversal of a cluster lead to the desired terminal state (number of times traversed along a sucessful path)  #cumhistory_vec: number of times cluster is traversed on a successful path

        if count_reached == 0:
            prob_[:, terminal_state] = 0
            print(
                f'{datetime.now()}\t{terminal_state} cluster/state is never reached. try increase number of KNN (which will increase number of edges) or try to increase the value of jac_std_global and cluster_graph_pruning which will lower edge pruning')
        else:
            loc_1 = np.where(prob_ == 1)
            loc_1 = loc_1[1]
            prob_[0, loc_1] = 0
            # print('zerod out prob', prob_)
            temp_ = np.max(prob_)
            if temp_ == 0: temp_ = 1
            prob_ = prob_ / min(1,
                                1.1 * temp_)  # do some scaling to amplify the probabiltiies of those clusters that are not 1 to make the range of values more compact
            prob_[0, loc_1] = 1  # put back probability =1
            print(
                f'{datetime.now()}\tCluster or terminal cell fate {terminal_state} is reached {count_reached} times')  # {prob_}
        prob_mat[:, ts_index_counter] = prob_
        ts_index_counter += 1

    return prob_mat


def _compute_rw2_hittingtimes(A: ndarray, root: int, memory: float = 0.8, num_walks: int = 1300, x_lazy: float = 0.95,
                             alpha_teleport: float = 0.99):
    '''

    :param walks: ndarray walks of shape n_walks x n_steps
    :param A: ndarray of clustergraph (values are edge weights, higher means stronger edge)
    :return:
    '''
    # from scipy.sparse import csc_matrix
    # A_csr = csr_matrix(A)
    # print(f'cluster graph A', A)
    # generate ndarray of walks (n_walks x n_steps+1)
    print(f'{datetime.now()}\tstart computing walks with rw2 method')
    walks = _rw2_walks(A=A, root=root, memory=memory, num_walks=num_walks, x_lazy=x_lazy, alpha_teleport=alpha_teleport)
    num_nodes = A.shape[0]
    num_walks = walks.shape[0]
    walk_length = walks.shape[                      1] - 1  # (the first element in the walk is the root, so the walks array has n_steps + 1 element
    hitting_array = np.zeros((num_nodes, num_walks))

    walk_count = 0
    for walk_i in walks:
        walk_i_dist_cumulative = [0]
        # print(f'walk_i {walk_i}')
        for step in range(walk_length):
            edge_weight = A[walk_i[step], walk_i[step + 1]]
            # print(f'edge weight {edge_weight}')
            edge_distance = 1. / (1 + math.exp(edge_weight - 1))  # convert to distance
            walk_i_dist_cumulative.append(walk_i_dist_cumulative[-1] + edge_distance)
            # print(f'edge distance',edge_distance)
            # print(f'cumulative walk {walk_i_dist_cumulative}')
        for node_i in range(num_nodes):
            # print(f'node {node_i}')
            first_time_at_statei = np.where(walk_i == node_i)[0]

            if len(first_time_at_statei) == 0:
                hitting_array[node_i, walk_count] = walk_length + 1
            else:
                # print(f'first time hitting node {node_i} is {first_time_at_statei[0]}')
                hitting_array[node_i, walk_count] = walk_i_dist_cumulative[first_time_at_statei[0]]
        walk_count += 1
    # print(f'hitting_array {hitting_array}')

    # extract relevant hitting times
    hitting_array_final = np.zeros((1, num_nodes))
    no_times_state_reached_array = np.zeros((1, num_nodes))
    for i in range(num_nodes):
        rowtemp = hitting_array[i, :]
        no_times_state_reached_array[0, i] = np.sum(rowtemp != (walk_length + 1))

    for i in range(num_nodes):
        rowtemp = hitting_array[i, :]
        no_times_state_reached = np.sum(rowtemp != (walk_length + 1))
        if no_times_state_reached != 0:
            perc = np.percentile(rowtemp[rowtemp != walk_length + 1], 20) + 0.001  # 15 for Human and Toy
            # print('state ', i,' has perc' ,perc)
            hitting_array_final[0, i] = np.mean(rowtemp[rowtemp <= perc])
        else:
            hitting_array_final[0, i] = (walk_length + 1)

    return hitting_array_final[0]


def _simulate_markov_sub(A, num_sim, hitting_array, q, root, seed):
    # sub process that actually does the mcmcs
    n_states = A.shape[0]
    P = A / A.sum(axis=1).reshape((n_states, 1))

    hitting_array_temp = np.zeros((P.shape[0], 1)).astype('float64')
    n_steps = int(2 * n_states)

    currentState = root

    state = np.zeros((1, n_states))
    state[0, currentState] = 1
    state_root = state.copy()
    for i in range(num_sim):
        dist_list = []
        state = state_root
        currentState = root
        stateHist = state
        for x in range(n_steps):
            seed += 1
            np.random.seed(seed)
            nextState = np.random.choice(range(P.shape[0]), p=P[currentState])
            dist = A[currentState, nextState]
            dist_list.append(1. / (1 + math.exp(dist - 1)))

            state = np.zeros((1, n_states))
            state[0, nextState] = 1.0
            currentState = nextState
            # Keep track of state history
            stateHist = np.append(stateHist, state, axis=0)

        for state_i in range(P.shape[0]):
            first_time_at_statei = np.where(stateHist[:, state_i] == 1)[0]
            if len(first_time_at_statei) == 0:
                hitting_array_temp[state_i, 0] = n_steps + 1
            else:
                total_dist = 0
                for ff in range(first_time_at_statei[0]):
                    total_dist = dist_list[ff] + total_dist

                hitting_array_temp[state_i, 0] = total_dist  # first_time_at_statei[0]
        hitting_array = np.append(hitting_array, hitting_array_temp, axis=1)

    hitting_array = hitting_array[:, 1:]
    q.append(hitting_array)


def get_biased_weights(edges, weights, pt, round=1):
    # small nu means less forward biasing (0.5 is quite mild)
    # larger nu (in our case 1/nu) means more aggressive forwards biasing https://en.wikipedia.org/wiki/Generalised_logistic_function

    # using the pseudotime calculated from lazy-jumping walk. Otherwise using the refined MCMC Psuedotimes before
    # calculating lineage likelihood paths
    b = 1 if round == 1 else 20

    weights_thr, pct_thr = weights.mean(), np.percentile(pt, 80)
    loc_high_pt = set(np.where(pt > pct_thr)[0])
    for i in np.where(weights > weights_thr)[0]:
        start, end = edges[i]
        if start in loc_high_pt or end in loc_high_pt:
            weights[i] = 0.5 * weights.mean()
    weights.clip(np.percentile(weights, 10), np.percentile(weights, 90))

    bias_weight, K, c, C, nu = [], 1, 0, 1, 1
    for (s, t), w in zip(edges, weights):
        t_ab = pt[s] - pt[t]
        bias_weight.append(w * K / (C + math.exp(b * (t_ab + c))) ** nu)

    return bias_weight


def expected_num_steps(start_i, N):
    return np.dot(N, np.ones(N.shape[0]))[start_i]


def absorption_probability(N, R, absorption_state_j):
    M = np.dot(N, R)
    return M, M[:, absorption_state_j]


class VIA:
    '''

    A class to represent the VIA analysis

    Parameters
    ----------

    data: ndarray
        input matrix of size n_cells x n_dims. Expects the PCs or features that will be used in the TI computation. Can be e.g. adata.obsm['X_pca][:,0:20]
    true_label:list
        list of str/int that correspond to the ground truth or reference annotations. Can also be None when no labels are available
    labels: ndarray (nsamples, )
        default is None. and PARC clusters are used for the viagraph. alternatively provide a list of clustermemberships that are integer values (not strings) to construct the viagraph using another clustering method or available annotations
    edgepruning_clustering_resolution_local:float
        default = 2
        local level of pruning for PARC graph clustering stage. Range (0.1,3) higher numbers mean more edge retention. For large datasets can stick to just tuning edgepruning_clustering_resolution
    edgepruning_clustering_resolution: float
        (optional, default = 0.15, can also set as 'median') graph pruning for PARC clustering stage. Higher value keeps more edges, results in fewer clusters. Smaller value removes more edges and results in more clusters. Number of standard deviations below the network’s mean-jaccard-weighted edges. 0.1-1 provide reasonable pruning.
        higher value means less pruning (more edges retained). e.g. a value of 0.15 means all edges that are above mean(edgeweight)-0.15*std(edge-weights) are retained.
        We find both 0.15 and ‘median’ to yield good results/starting point and resulting in pruning away ~ 50-60% edges
    keep_all_local_dist: bool, str
        default value of 'auto' means that for smaller datasets local-pruning is done prior to clustering, but for large datasets local pruning is set to False for speed.
        can also set to be bool of True or False
    do_clustergraph_edgecontrol:bool=True
        limits the max number of edges per cluster in the clustergraph to 30 edges. Applied after any pruning of edges and serves as a final sense-check that the number of edges per cluster doesnt become intractable
    too_big_factor: float
        (optional, default=0.4). Forces clusters > 0.4*n_cells to be re-clustered
    resolution_parameter: float
        (default =1) larger value means more and smaller clusters
    partition_type:str
        (default "ModularityVP") Options
    small_pop: int
         (default 10) Via attempts to merge Clusters with a population < 10 cells with larger clusters. If you have a very small dataset (e.g. few hundred cells), then consider lowering to e.g. 5
    jac_weighted_edges:bool
        (default = True) Use weighted edges in the PARC clustering step
    knn: int
        (optional, default = 30) number of K-Nearest Neighbors for HNSWlib KNN graph. Larger knn means more graph connectivity. Lower knn means more loosely connected clusters/cells
    n_iter_leiden:int
    random_seed: int
        Random seed to pass to clustering
    num_threads:
    distance:str
        (default 'l2') Euclidean distance 'l2' by default; other options 'ip' and 'cosine' for graph construction and similarity
    visual_cluster_graph_pruning: float
        (optional, default = 0.15) This only comes into play if the user deliberately chooses not to use the default edge-bundling method of visualizating edges (draw_piechart_graph()) and instead calls draw_piechart_graph_nobundle().
        It is often set to the same value as the PARC clustering level of edgepruning_clustering_resolution. This does not impact computation of terminal states, pseudotime or lineage likelihoods.
        It controls the number of edges plotted for visual effect
    cluster_graph_pruning: float
        (optional, default =0.15) Pruning level of the cluster graph (does not impact number of clusters). Only impacts the connectivity of the clustergraph. Often set to the same value as the PARC clustering level of edgepruning_clustering_resolution.Reasonable range [0.1,1]
        To retain more connectivity in the clustergraph underlying the trajectory computations, increase the value
    time_smallpop: max time to be allowed handling singletons

    x_lazy:float
        (default =0.95) 1-x = probability of staying in same node (lazy). Values between 0.9-0.99 are reasonable
    alpha_teleport: float
        (default = 0.99) 1-alpha is probability of jumping. Values between 0.95-0.99 are reasonable unless prior knowledge of teleportation
    root_user: list, None
        can be a list of strings, a list of int or None
        (default is None) When the root_user is set as None and an RNA velocity matrix is available, a root will be automatically computed
        if the root_user is None and not velocity matrix is provided, then an arbitrary root is selected
        if the root_user is ['celltype_earlystage'] where the str corresponds to an item in true_label, then a suitable starting point will be selected corresponding to this group
        if the root_user is [678], where 678 is the index of the cell chosen as a start cell, then this will be the designated starting cell.
        It is possible to give a list of root indices and groups. [120, 699] or ['traj1_earlystage', 'traj2_earlystage'] when there are more than one trajectories
    preserve_disconnected: bool
        (default = True) If you believe there may be disconnected trajectories then set this to False
    dataset: str
        Can be set to 'group' or '' (default). this refers to the type of root label (group level root or single cell index) you are going to provide.
        if your true_label has a sensible group of cells for a root then you can set dataset to 'group' and make the root parameter ['labelname_root_cell_type']
        if your root corresponds to one particular cell then set dataset = '' (default)
    embedding: ndarray
        (optional, default = None) embedding (e.g. precomputed tsne, umap, phate, via-umap) for plotting data. Size n_cells x 2
        If an embedding is provided when running VIA, then a scatterplot colored by pseudotime, highlighting terminal fates
    velo_weight: float
        (optional, default = 0.5) #float between [0,1]. the weight assigned to directionality and connectivity derived from scRNA-velocity
    neighboring_terminal_states_threshold:int
        (default = 3). Candidates for terminal states that are neighbors of each other may be removed from the list if they have this number of more of terminal states as neighbors
    knn_sequential:int
        (default =10) number of knn in the adjacent time-point for time-series data (t_i and t_i+1)
    knn_sequential_reverse: int
        (default = 0) number of knn enforced from current to previous time point
    t_diff_step: int
        (default =1) Number of permitted temporal intervals between connected nodes. If time data is labeled as [0,25,50,75,100,..]
        then t_diff_step=1 corresponds to '25' and only edges within t_diff_steps are retained
    is_coarse:bool
        (default = True) If running VIA in two iterations where you wish to link the second fine-grained iteration with the initial iteration, then you set to False
    via_coarse: VIA
        (default = None) If instantiating a second iteration of VIA that needs to be linked to a previous iteration (e.g. via0), then set via_coarse to the previous via0 object
    df_annot: DataFrame
        (default None) used for the Mouse Organ data
    preserve_disconnected_after_pruning:bool
        (default = False) If you believe there are disconnected trajectories then set this to True and test your hypothesis
    A_velo: ndarray
        Cluster Graph Transition matrix based on rna velocity [n_clus x n_clus]
    velocity_matrix: matrix
            (default None) matrix of size [n_samples x n_genes]. this is the velocity matrix computed by scVelo (or similar package) and stored in adata.layers['velocity']. The genes used for computing velocity should correspond to those useing in gene_matrix
            Requires gene_matrix to be provided too.
    gene_matrix: matrix
            (default None) Only used if Velocity_matrix is available. matrix of size [n_samples x n_genes]. We recommend using a subset like HVGs rather than full set of genes. (need to densify input if taking from adata = adata.X.todense())
    time_series:bool
        (default False) if the data has time-series labels then set to True
    time_series_labels:list
        (default None) list of integer values of temporal annoataions corresponding to e.g. hours (post fert), days, or sequential ordering
    pca_loadings: array
        (default None) the loadings of the pcs used to project the cells (to projected euclidean location based on velocity). n_cells x n_pcs
    secondary_annotations: None
        (default None)
    edgebundle_pruning:float
        (default=None) will by default be set to the same as the cluster_graph_pruning and influences the visualized level of pruning of edges.
        Typical values can be between [0,1] with higher numbers retaining more edges
    edgebundle_pruning_twice:bool
        default: False. When True, the edgebundling is applied to a further visually pruned (visual_cluster_graph_pruning) and can sometimes simplify the visualization. it does not impact the pseudotime and lineage computations
     piegraph_arrow_head_width: float
        (default = 0.1) size of arrow heads in via cluster graph
    piegraph_edgeweight_scalingfactor:
        (defaulf = 1.5) scaling factor for edge thickness in via cluster graph
    max_visual_outgoing_edges: int
        (default =2) Only allows max_visual_outgoing_edges to come out of any given node. Used in differentiation_flow()

    edgebundle_pruning:float
        (default=None) will by default be set to the same as the cluster_graph_pruning and influences the visualized level of pruning of edges.
        Typical values can be between [0,1] with higher numbers retaining more edges

    edgebundle_pruning_twice:bool
        default: False. When True, the edgebundling is applied to a further visually pruned (visual_cluster_graph_pruning) and can sometimes simplify the visualization for very cluttered graphs. it does not impact the pseudotime and lineage computations
    pseudotime_threshold_TS: int
        (default = 30) corresponds to the criteria for a state to be considered a candidate terminal cell fate to be 30% or later of the computed psuedotime range
    num_mcmc_simulations:int
        (default = 1300) number of random walk simulations conducted
    embedding_type: str
        (default = 'via-mds', other options are 'via-atlas' and 'via-force'
    do_compute_embedding: bool
        (default = False) If you want an embedding (n_samples x2) to be computed on the basis of the via sc graph then set this to True
    do_gaussian_kernel_edgeweights: bool
        (default = False) Type of edgeweighting on the graph edges
    memory: float
        (default = 5) higher value means more memory and a more retrospective/inwards randomwalk. memory = 0 means run using the non-memory Via 1.0 mode
    viagraph_decay: float
        (default = 0.9) increasing decay causes more edges to merge
   
    p_memory: 1/p * edge weight to de-emphasize returning to previous node. i.e. when next node = previous node. large value of p_memory value means more exploration
    graph_init_pos: matrix (or list of lists) to initialize the viagraph
    spatial_coords: np.ndarray of size n_cells x 2 (denoting x,y coordinates) of each spot/cell
    do_spatial_knn: Whether or not to do spatial mode of StaVia for graph augmentation
    do_spatial_layout: whether to use spatial coords for layout of the clustergraph
    spatial_knn:int = 15. number of knn's added based on spatial proximity indiciated by spatial_coords
    spatial_aux:list = [] a list of slice IDs so that only cells/spots on the same slice are considered when building the spatial_knn graph

    Attributes
    ------------
    labels: array
        length (n_samples, ) of cluster labels ndarray pre determined cluster labels user defined. #np.asarray(pre_labels).flatten()
    single_cell_pt_markov: list
        length n_samples of pseudotime
    single_cell_bp: ndarray
        [n_lineages x n_samples] array of single cell branching probabilities towards each lineage (lineage normalized).
        Each column corresponds to a terminal state, in the order presented by the terminal_clusters attribute
    single_cell_bp_rownormed: ndarray
        [n_lineages x n_samples] array of single cell branching probabilities towards each lineage (cell normalized).
        Each column corresponds to a terminal state, in the order presented by the terminal_clusters attribute
    terminal_clusters: list
        list of clusters that are cell fates/ unique lineages
    cluster_bp: ndarray
        [n_clusters x n_terminal_states]. Lineage probability of cluster towards a particular terminal cluster state
    CSM: ndarray
        [n_cluster x n_clusters] array of cosine similarity used to weight the cluster graph transition matrix by velocity
    single_cell_transition_matrix: ndarray
        [n_samples x n_samples]
    terminal_clusters:list
        (default None) list of terminal clusters
    csr_full_graph: csr matrix of single-cell graph (augmented with sequential data when providing time_series information)
    csr_array_locally_pruned: csr matrix
    ig_full_graph:
    full_neighbor_array:
    user_defined_terminal_cell:list=[] list of cell indices corresponding to terminal fate cells
    user_defined_terminal_group:list=[] list of group level labels corresponding to labels found in true_label, that represent cell fates
    n_milestones: int = None Number of milestones in the via-mds computation (anything more than 10,000 can be computationally heavy and time consuming) Typically auto-determined within the via-mds function
    embedding: ndarray
        [n_cells x 2] provided by user or autocomputed with via-mds or via-umap

    '''

    def __init__(self, data: ndarray, true_label=None, edgepruning_clustering_resolution_local: float = 1, edgepruning_clustering_resolution=0.15,
                 labels: ndarray = None,
                 keep_all_local_dist='auto', do_clustergraph_edgecontrol:bool=True,too_big_factor: float = 0.4, resolution_parameter: float = 1.0,
                 partition_type: str = "ModularityVP", small_pop: int = 10,
                 jac_weighted_edges: bool = True, knn: int = 30, n_iter_leiden: int = 5, random_seed: int = 42,
                 num_threads=-1, distance='l2', time_smallpop=15,
                 super_cluster_labels: bool = False, super_node_degree_list: bool = False,
                 super_terminal_cells: bool = False, x_lazy: float = 0.99, alpha_teleport: float = 0.99,
                 root_user=None, preserve_disconnected: bool = True, dataset: str = '',
                 super_terminal_clusters: list = [],
                 is_coarse=True, csr_full_graph: csr_matrix = '', csr_array_locally_pruned='', ig_full_graph='',
                 full_neighbor_array='', full_distance_array='', embedding: ndarray = None, df_annot=None,
                 preserve_disconnected_after_pruning: bool = False,
                 secondary_annotations: list = None, pseudotime_threshold_TS: int = 30,
                 cluster_graph_pruning: float = 0.15,
                 visual_cluster_graph_pruning: float = 0.15, neighboring_terminal_states_threshold=3,
                 num_mcmc_simulations=1300,
                 piegraph_arrow_head_width=0.1,
                 piegraph_edgeweight_scalingfactor=1.5, max_visual_outgoing_edges: int = 2, via_coarse=None,
                 velocity_matrix=None,
                 gene_matrix=None, velo_weight=0.5, edgebundle_pruning=None, A_velo=None, CSM=None,
                 edgebundle_pruning_twice=False, pca_loadings=None, time_series=False,
                 time_series_labels: list = None, knn_sequential: int = 10, knn_sequential_reverse: int = 0,
                 t_diff_step: int = 1, single_cell_transition_matrix=None,
                 embedding_type: str = 'via-mds', do_compute_embedding: bool = False, color_dict: dict = None,
                 user_defined_terminal_cell: list = [], user_defined_terminal_group: list = [],
                 do_gaussian_kernel_edgeweights: bool = False, RW2_mode: bool = False,
                 working_dir_fp: str = '/home/', memory=5, viagraph_decay=0.9, p_memory=1, graph_init_pos: np.ndarray=None, spatial_coords:np.ndarray=None, do_spatial_knn:bool=False, do_spatial_layout:bool=False, spatial_knn:int = 15, spatial_aux:list = []):

        self.data = data
        self.nsamples, self.ncomp = data.shape
        if true_label is not None:
            self.true_label = true_label
        else:
            self.true_label = [1] * self.nsamples
        if velocity_matrix is None: velo_weight = 0
        self.velo_weight = velo_weight  # float between 0,1. the weight assigned to directionality and connectivity derived from scRNA-velocity
        if edgebundle_pruning is None: edgebundle_pruning = cluster_graph_pruning
        self.edgebundle_pruning = edgebundle_pruning
        if (root_user is None) & (velocity_matrix is None):
            root_user = []
            dataset = ''  
        elif root_user is None: 
            dataset = ''
        elif (type(root_user[0]) == str): 
            dataset = 'group'
        else: 
            dataset = ''
        self.root_user = root_user
        self.dataset = dataset
        self.knn_struct = None
        if isinstance(labels, list):
            labels = np.asarray(labels).flatten()
        self.labels = labels  # np.asarray(pre_labels).flatten() where pre_labels is a list
        self.connected_comp_labels = None
        self.edgelist = None
        self.edgelist_unique = None

        # higher edgepruning_clustering_resolution_local means more edges are kept
        # higher edgepruning_clustering_resolution means more edges are kept
        if keep_all_local_dist == 'auto':
            # If large dataset skip local pruning to increase speed
            keep_all_local_dist = data.shape[0] > 50000

        if resolution_parameter != 1:
            partition_type = "RBVP"  # Reichardt and Bornholdt’s Potts model. Note that this is the same as ModularityVertexPartition when setting 𝛾 = 1 and normalising by 2m

        self.edgepruning_clustering_resolution_local = edgepruning_clustering_resolution_local
        self.edgepruning_clustering_resolution = edgepruning_clustering_resolution  ##0.15 is also a recommended value performing empirically similar to 'median'
        self.keep_all_local_dist = keep_all_local_dist
        self.do_clustergraph_edgecontrol = do_clustergraph_edgecontrol
        self.too_big_factor = too_big_factor  ##if a cluster exceeds this share of the entire cell population, then the PARC will be run on the large cluster. at 0.4 it does not come into play
        self.resolution_parameter = resolution_parameter
        self.partition_type = partition_type
        self.small_pop = small_pop  # smallest cluster population to be considered a community
        self.jac_weighted_edges = jac_weighted_edges
        self.knn = knn
        self.knn_sequential = knn_sequential  # number of knn in the adjacent time-point for time-series data
        self.knn_sequential_reverse = knn_sequential_reverse  # number of knn made between timepoint and previous time point
        self.n_iter_leiden = n_iter_leiden
        self.random_seed = random_seed  # enable reproducible Leiden clustering
        self.num_threads = num_threads  # number of threads used in KNN search/construction
        self.distance = distance  # Euclidean distance 'l2' by default; other options 'ip' and 'cosine'
        self.time_smallpop = time_smallpop

        self.super_cluster_labels = super_cluster_labels
        self.super_node_degree_list = super_node_degree_list
        self.super_terminal_clusters = super_terminal_clusters
        self.full_neighbor_array = full_neighbor_array
        self.full_distance_array = full_distance_array
        self.ig_full_graph = ig_full_graph
        self.csr_array_locally_pruned = csr_array_locally_pruned
        self.csr_full_graph = csr_full_graph
        self.super_terminal_cells = super_terminal_cells


        self.x_lazy = x_lazy  # 1-x = probability of staying in same node
        self.alpha_teleport = alpha_teleport  # 1-alpha is probability of jumping
        self.preserve_disconnected = preserve_disconnected
        self.is_coarse = is_coarse  # set to True for first round of VIA. if one chooses to run a second iteration of VIA that uses the terminal states from the first round, then set this to False for second iteration
        self.embedding = embedding
        self.df_annot = df_annot
        self.preserve_disconnected_after_pruning = preserve_disconnected_after_pruning  # pruning can cause some fragmentation of small clusters, these should be reattached to the relevant component (so typically this is False as we dont want to preserve these small extra components resulting from pruning)
        self.secondary_annotations = secondary_annotations
        self.pseudotime_threshold_TS = pseudotime_threshold_TS
        self.cluster_graph_pruning = cluster_graph_pruning
        self.visual_cluster_graph_pruning = visual_cluster_graph_pruning  # higher value means more edges retained. This is applied to the clustergraph before visulizing only if two rounds of edgbebundlng is done.
        self.neighboring_terminal_states_threshold = neighboring_terminal_states_threshold  # number of neighbors of a terminal state has before it is eliminated as a TS
        self.num_mcmc_simulations = num_mcmc_simulations  # number of mcmc simulations in second state of pseudotime computation
        self.piegraph_arrow_head_width = piegraph_arrow_head_width
        self.piegraph_edgeweight_scalingfactor = piegraph_edgeweight_scalingfactor
        self.max_visual_outgoing_edges = max_visual_outgoing_edges  # higher value means more edges retained. This is applied to the clustergraph and is a strong threshold for number of edges shown
        if velocity_matrix is not None: velocity_matrix[np.isnan(velocity_matrix)] = 0
        self.velocity_matrix = velocity_matrix  # matrix from scVelo with velocities
        if gene_matrix is not None: gene_matrix[np.isnan(gene_matrix)] = 0
        self.gene_matrix = gene_matrix  # matrix (not numpy array) with gene expression #such as adata.X.todense() or adata.layers["Mu"] from scVelo (first moments of spliced gene counts
        self.A_velo = A_velo  # the transition matrix weighted by the rna velocity [n_clus x n_clus]
        self.CSM = CSM  # the cosine similarity matrix using velocity [n_clusters x n_clusters]
        self.edgebundle_pruning_twice = edgebundle_pruning_twice  # default: False. When True, the edgebundling is applied to a further visually pruned (visual_cluster_graph_pruning) and can sometimes simplify the visualization. it does not impact the pseudotime and lineage computations
        self.pca_loadings = pca_loadings  # the loadings of the pcs used to project the cells (to projected euclidean location based on velocity)
        self.time_series = time_series
        self.time_series_labels = time_series_labels
        self.t_diff_step = t_diff_step
        self.single_cell_transition_matrix = single_cell_transition_matrix
        self.embedding_type = embedding_type
        self.do_compute_embedding = do_compute_embedding
        self.color_dict = color_dict
        self.user_defined_terminal_cell = user_defined_terminal_cell
        self.user_defined_terminal_group = user_defined_terminal_group
        self.do_gaussian_kernel_edgeweights = do_gaussian_kernel_edgeweights
        self.RW2_mode = RW2_mode
        self.working_dir_fp = working_dir_fp
        self.memory = memory
        self.p_memory = p_memory
        self.viagraph_decay = viagraph_decay
        self.graph_init_pos = graph_init_pos
        self.spatial_coords = spatial_coords
        self.do_spatial_knn = do_spatial_knn
        self.do_spatial_layout = do_spatial_layout
        self.spatial_knn = spatial_knn
        self.spatial_aux = spatial_aux  #list of labels that identify cells from the same tissue slice

    def _make_pt_augmented_adjacency_igraph(self, neighbors: ndarray, distances: ndarray, k_reverse: int = 10,
                                           k_seq: int = 0):
        '''
        If you dont have time-series labels to adjust the construction of the single-cell graph, you can try using the
        pseudotime (single-cell level) to achieve the same sequencing effect. However, since the pseudo-times are derived from
        the original unguided structure, there may not neccessarily be a large visible impact on the graph structure

        :param neighbors: ndarray (n_cells x n_neighbors) original unguided knn graph structure
        :param distances: ndarray (n_cell x n_neighbors)
        :param k_reverse: (int) default = 10
        :param k_seq: this is zero because terminal states that have lower pseudotimes should not be forced to look for neighbors in later pseudotimes
        :param knn:
        :return:
        '''
        n_pt_augmented, d_pt_augmented = sequential_knn(self.data,
                                                        [int(i * 10) for i in self.single_cell_pt_markov],
                                                        neighbors, distances,
                                                        k_seq=k_seq, k_reverse=k_reverse,
                                                        num_threads=self.num_threads, distance=self.distance)

        adjacency_pt_augmented = self._make_csrmatrix_noselfloop(n_pt_augmented, d_pt_augmented, time_series=True,
                                                                 time_series_labels=[int(i * 10) for i in
                                                                                     self.single_cell_pt_markov],
                                                                 t_diff_step=3)  # this function has local pruning which removes neighbors that are more than t_dif apart. Since the same type of local pruning wrt t_dif is applied pre-clustergraph, we only need to call this function once in the case of time_series data
        ig_full_graph_pt_guided = _composite_jacAffinity_distanceAffinity(sc_csr_graph=adjacency_pt_augmented)
        return ig_full_graph_pt_guided, adjacency_pt_augmented

    def _get_terminal_clusters_user_defined_(self, user_defined_terminal_cell: list = [],
                                            user_defined_terminal_group: list = []):
        '''
        Allow the user to optionally select group or cell index level terminal fates that override the automated cell fate detection

        :param user_defined_terminal_cell: list of cell indices corresponding to terminal fate cells
        :param user_defined_terminal_cell_group: list of group level labels corresponding to labels found in true_label, that represent cell fates
        :return: list of clusters to represent the cell fate clusters in the via-clustergraph
        '''
        dict_user_defined_terminal_clusters = {}
        terminal_cluster_list = []
        if len(user_defined_terminal_cell) > 0:
            for i in user_defined_terminal_cell:
                clus_ = self.labels[i]
                if clus_ not in terminal_cluster_list:
                    terminal_cluster_list.append(clus_)
                    dict_user_defined_terminal_clusters[user_defined_terminal_cell] = clus_
                else:
                    print(f'{i} is a repeated terminal cluster')
                    #terminal_cluster_list.append(clus_)

        else:

            for user_terminal_group in user_defined_terminal_group:

                # location in indices of which cells belong to this user_terminal_group
                index_terminal_group = np.where(np.asarray(self.true_label) == user_terminal_group)[0]
                potential_clusters = self.labels[index_terminal_group]
                clus_ = func_mode(potential_clusters.tolist())
                if clus_ not in terminal_cluster_list:
                    terminal_cluster_list.append(clus_)
                    dict_user_defined_terminal_clusters[user_terminal_group] = clus_
        print(f'{datetime.now()}\tTerminal cluster list based on user defined cells/groups:',
              [(key, dict_user_defined_terminal_clusters[key]) for key in dict_user_defined_terminal_clusters])
        return terminal_cluster_list

    def _get_terminal_clusters(self, A, markov_pt, root_ai):
        n_ = A.shape[0]  # number of states in the graph component

        if n_ <= 10: n_outlier_std = 3
        if (n_ <= 40) & (n_ > 10): n_outlier_std = 2

        if n_ >= 40: n_outlier_std = 2  # 1

        pop_list = []

        # print('get terminal', set(self.labels), np.where(self.labels == 0))
        for i in list(set(self.labels)):
            pop_list.append(len(np.where(self.labels == i)[0]))
        # we weight the out-degree based on the population of clusters to avoid allowing small clusters to become the terminals based on population alone
        A_new = A.copy()
        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                A_new[i, j] = A[i, j] * (pop_list[i] + pop_list[j]) / (pop_list[i] * pop_list[j])

        # make an igraph graph to compute the closeness
        g_dis = ig.Graph.Adjacency(
            (A_new > 0).tolist())  # need to manually add the weights as igraph treates A>0 as boolean
        g_dis.es['weights'] = 1 / A_new[A_new.nonzero()]  # we want "distances" not weights for closeness and betweeness

        betweenness_score = g_dis.betweenness(weights='weights')
        betweenness_score_array = np.asarray(betweenness_score)
        betweenness_score_takeout_outlier = betweenness_score_array[betweenness_score_array < (
                np.mean(betweenness_score_array) + n_outlier_std * np.std(betweenness_score_array))]
        betweenness_list = [i for i, score in enumerate(betweenness_score) if score < (
                np.mean(betweenness_score_takeout_outlier) - 0 * np.std(betweenness_score_takeout_outlier))]

        closeness_score = g_dis.closeness(mode='ALL', cutoff=None, weights='weights', normalized=True)
        closeness_score_array = np.asarray(closeness_score)
        closeness_score_takeout_outlier = closeness_score_array[
            closeness_score_array < (np.mean(closeness_score_array) + n_outlier_std * np.std(closeness_score_array))]
        closeness_list = [i for i, score in enumerate(closeness_score) if
                          score < (np.mean(closeness_score_takeout_outlier) - 0 * np.std(
                              closeness_score_takeout_outlier))]

        out_deg = A_new.sum(axis=1)
        out_deg = np.asarray(out_deg)

        outdegree_score_takeout_outlier = out_deg[out_deg < (np.mean(out_deg) + n_outlier_std * np.std(out_deg))]
        loc_deg = [i for i, score in enumerate(out_deg) if
                   score < (np.mean(outdegree_score_takeout_outlier) - 0 * np.std(outdegree_score_takeout_outlier))]
        print(f"{datetime.now()}\tIdentifying terminal clusters corresponding to unique lineages...")
        print(f"{datetime.now()}\tCloseness:{closeness_list}")
        print(f"{datetime.now()}\tBetweenness:{betweenness_list}")
        print(f"{datetime.now()}\tOut Degree:{loc_deg}")

        markov_pt = np.asarray(markov_pt)
        pct = 10 if n_ <= 40 else 30
        loc_pt = np.where(markov_pt >= np.percentile(markov_pt, pct))[0]

        terminal_clusters_1 = list(set(closeness_list) & set(betweenness_list))
        terminal_clusters_2 = list(set(closeness_list) & set(loc_deg))
        terminal_clusters_3 = list(set(betweenness_list) & set(loc_deg))
        terminal_clusters = list(set(terminal_clusters_1) | set(terminal_clusters_2))
        terminal_clusters = list(set(terminal_clusters) | set(terminal_clusters_3))
        terminal_clusters = list(set(terminal_clusters) & set(loc_pt))

        terminal_org = terminal_clusters.copy()

        for terminal_i in terminal_org:
            if terminal_i in terminal_clusters:
                removed_terminal_i = False
            else:
                removed_terminal_i = True
            # print('terminal state', terminal_i)
            count_nn = 0
            ts_neigh = []
            neigh_terminal = np.where(A[:, terminal_i] > 0)[0]
            if neigh_terminal.size > 0:
                for item in neigh_terminal:
                    if item in terminal_clusters:
                        ts_neigh.append(item)
                        count_nn = count_nn + 1

                    if n_ >= 10:
                        if item == root_ai:  # if the terminal state is a neighbor of
                            if terminal_i in terminal_clusters:
                                terminal_clusters.remove(terminal_i)
                                print(
                                    f"{datetime.now()}\tWe removed cluster {terminal_i} from the shortlist of terminal states")
                                removed_terminal_i = True
                if count_nn >= self.neighboring_terminal_states_threshold:  # 2
                    if removed_terminal_i == False:
                        temp_remove = terminal_i
                        temp_time = markov_pt[terminal_i]
                        for to_remove_i in ts_neigh:
                            if markov_pt[to_remove_i] < temp_time:
                                temp_remove = to_remove_i
                                temp_time = markov_pt[to_remove_i]
                        terminal_clusters.remove(temp_remove)
                        print(
                            f"{datetime.now()}\tCluster {terminal_i} had {self.neighboring_terminal_states_threshold} or more neighboring terminal states {ts_neigh} and so we removed cluster {temp_remove}")
        if len(terminal_clusters) == 0: terminal_clusters = loc_deg
        # print('terminal_clusters', terminal_clusters)
        return terminal_clusters

    def compute_hitting_time(self, sparse_graph, root, x_lazy, alpha_teleport, number_eig=0):
        # 1- alpha is the probability of teleporting
        # 1- x_lazy is the probability of staying in current state (be lazy)
        # Computing lazy-teleporting expected hitting time
        beta_teleport = 2 * (1 - alpha_teleport) / (2 - alpha_teleport)
        N = sparse_graph.shape[0]

        Lsym = csgraph.laplacian(sparse_graph, normed=True)
        eigval, eigvec = scipy.sparse.linalg.eigsh(Lsym, number_eig or Lsym.shape[0] - 1)

        Greens, beta_norm_lap = np.zeros((N, N), float), np.zeros((N, N), float)
        Xu = np.zeros((N, N))
        Xu[:, root] = 1
        Xv_Xu = np.eye(N) - Xu

        for i in range(1 if alpha_teleport == 1 else 0, len(eigval)):
            vv = np.outer(eigvec[:, i], eigvec[:, i])
            factor = beta_teleport + 2 * eigval[i] * x_lazy * (1 - beta_teleport)

            Greens += vv / factor
            beta_norm_lap += vv * factor

        D = np.diag(np.nan_to_num(np.array(sparse_graph.sum(axis=1)).reshape(-1) ** -.5, posinf=0))
        t = D @ Greens @ D * beta_teleport

        hitting_matrix = np.diagonal(t)[np.newaxis, :] - t
        # Calculate only diagonal elements of Xv_Xu @ t
        return np.abs((Xv_Xu * t.T).sum(-1)), (hitting_matrix + hitting_matrix.T)[root]

    def simulate_branch_probability(self, terminal_state, A, root, num_sim=500):

        n_states = A.shape[0]

        ncpu = multiprocessing.cpu_count()
        if (ncpu == 1) | (ncpu == 2):
            n_jobs = 1
        elif ncpu > 2:
            n_jobs = min(ncpu - 1, 5)
        # print('njobs', n_jobs)
        num_sim_pp = int(num_sim / n_jobs)  # num of simulations per process
        jobs = []

        manager = multiprocessing.Manager()

        q = manager.list()
        seed_list_ = list(range(n_jobs))
        seed_list = [i + self.random_seed for i in seed_list_]

        for i in range(n_jobs):
            cumstateChangeHist = np.zeros((1, n_states))
            cumstateChangeHist_all = np.zeros((1, n_states))
            process = multiprocessing.Process(target=_prob_reaching_terminal_state, args=(
                terminal_state, A, root, num_sim_pp, q, cumstateChangeHist,
                cumstateChangeHist_all,
                seed_list[i]))
            jobs.append(process)

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        cumhistory_vec = q[0][0]
        cumhistory_vec_all = q[0][1]

        count_reached = cumhistory_vec_all[0, terminal_state]

        for i in range(1, len(q)):
            cumhistory_vec = cumhistory_vec + q[i][0]
            cumhistory_vec_all = cumhistory_vec_all + q[i][1]

            count_reached = count_reached + q[i][1][0, terminal_state]

        print(
            f"{datetime.now()}\tFrom root {root},  the Terminal state {terminal_state} is reached {int(count_reached)} times.")
        # print('cumhistory_vec_all', cumhistory_vec_all)
        cumhistory_vec_all[
            cumhistory_vec_all == 0] = 1  # to avoid division by zero we say that the state has been visited once
        # print('cumhistory_vec_all', cumhistory_vec_all)
        # print('cumhistory_vec', cumhistory_vec)
        prob_ = cumhistory_vec / cumhistory_vec_all
        # given a cluster is traversed (cumhistory_vec_all), how many of these traversals led to a successful path to Terminal State X
        # i.e. with what probability does traversal of a cluster lead to the desired terminal state (number of times traversed along a sucessful path)  #cumhistory_vec: number of times cluster is traversed on a successful path

        np.set_printoptions(precision=3)

        if count_reached == 0:
            prob_[:, terminal_state] = 0
            print('never reached state', terminal_state)
        else:
            loc_1 = np.where(prob_ == 1)

            loc_1 = loc_1[1]

            prob_[0, loc_1] = 0
            # print('zerod out prob', prob_)
            temp_ = np.max(prob_)
            if temp_ == 0: temp_ = 1
            prob_ = prob_ / min(1,
                                1.1 * temp_)  # do some scaling to amplify the probabiltiies of those clusters that are not 1 to make the range of values more compact
        prob_[0, loc_1] = 1  # put back probability =1

        return list(prob_)[0]

    def _simulate_markov(self, A, root):
        '''
        Computes the MCMC based hitting times on the forward biased graph
        :param A: cluster graph adjacency matrix (forward biased)
        :param root:
        :return: mcmc hitting times
        '''
        n_states = A.shape[0]
        P = A / A.sum(axis=1).reshape((n_states, 1))
        # print('row normed P',P.shape, P, P.sum(axis=1))
        x_lazy = self.x_lazy  # 1-x is prob lazy
        alpha_teleport = self.alpha_teleport
        # bias_P is the transition probability matrix

        P = x_lazy * P + (1 - x_lazy) * np.identity(n_states)
        # print(P, P.sum(axis=1))
        P = alpha_teleport * P + ((1 - alpha_teleport) * (1 / n_states) * (np.ones((n_states, n_states))))
        # print('check prob of each row sum to one', P.sum(axis=1))

        currentState = root
        state = np.zeros((1, n_states))
        state[0, currentState] = 1
        num_sim = self.num_mcmc_simulations

        ncpu = multiprocessing.cpu_count()
        if (ncpu == 1) | (ncpu == 2):
            n_jobs = 1
        elif ncpu > 2:
            n_jobs = ncpu - 2  # min(ncpu - 2, 5)
        # print('njobs', n_jobs)
        num_sim_pp = int(num_sim / n_jobs)  # num of simulations per process

        n_steps = int(2 * n_states)

        jobs = []
        seed = self.random_seed
        dummy_list = []
        for dummy in range(n_jobs):
            hitting_array = np.ones((P.shape[0], 1)) * 1000
            dummy_list.append(hitting_array)

        with multiprocessing.Manager() as manager:
            q = manager.list()
            for i in range(2):  # range(n_jobs):
                hitting_array = np.ones((P.shape[0], 1)) * 1000
                process = multiprocessing.Process(target=_simulate_markov_sub,
                                                  args=(P, num_sim_pp, hitting_array, q, root, seed))
                jobs.append(process)
                process.start()
            for proc in jobs:
                proc.join()
            # for j in jobs:
            # j.start()

            # for j in jobs:
            # j.join()

            print(f"{datetime.now()}\tEnded all multiprocesses, will retrieve and reshape")
            hitting_array = q[0]
            for qi in q[1:]:
                hitting_array = np.append(hitting_array, qi, axis=1)  # .get(), axis=1)
            # print('finished getting from queue', hitting_array.shape)
        hitting_array_final = np.zeros((1, n_states))
        no_times_state_reached_array = np.zeros((1, n_states))

        for i in range(n_states):
            rowtemp = hitting_array[i, :]
            no_times_state_reached_array[0, i] = np.sum(rowtemp != (n_steps + 1))

        for i in range(n_states):
            rowtemp = hitting_array[i, :]
            no_times_state_reached = np.sum(rowtemp != (n_steps + 1))
            if no_times_state_reached != 0:
                perc = np.percentile(rowtemp[rowtemp != n_steps + 1], 15) + 0.001  # 15 for Human and Toy
                # print('state ', i,' has perc' ,perc)
                hitting_array_final[0, i] = np.mean(rowtemp[rowtemp <= perc])
            else:
                hitting_array_final[0, i] = (n_steps + 1)

        return hitting_array_final[0]

    def _compute_hitting_time_onbias(self, laplacian, inv_sqr_deg, root, x_lazy, alpha_teleport, number_eig=0):
        # 1- alpha is the probabilty of teleporting
        # 1- x_lazy is the probability of staying in current state (be lazy)
        beta_teleport = 2 * (1 - alpha_teleport) / (2 - alpha_teleport)
        N = laplacian.shape[0]
        print('is laplacian of biased symmetric', (laplacian.transpose() == laplacian).all())
        Id = np.zeros((N, N), float)
        np.fill_diagonal(Id, 1)
        # norm_lap = scipy.sparse.csr_matrix.todense(laplacian)

        eig_val, eig_vec = np.linalg.eig(
            laplacian)  # eig_vec[:,i] is eigenvector for eigenvalue eig_val[i] not eigh as this is only for symmetric. the eig vecs are not in decsending order
        print('eig val', eig_val.shape)
        if number_eig == 0: number_eig = eig_vec.shape[1]
        print('number of eig vec', number_eig)
        Greens_matrix = np.zeros((N, N), float)
        beta_norm_lap = np.zeros((N, N), float)
        Xu = np.zeros((N, N))
        Xu[:, root] = 1
        Id_Xv = np.zeros((N, N), int)
        np.fill_diagonal(Id_Xv, 1)
        Xv_Xu = Id_Xv - Xu
        start_ = 0
        if alpha_teleport == 1:
            start_ = 1  # if there are no jumps (alph_teleport ==1), then the first term in beta-normalized Green's function will have 0 in denominator (first eigenvalue==0)

        for i in range(start_, number_eig):  # 0 instead of 1st eg
            # print(i, 'th eigenvalue is', eig_val[i])
            vec_i = eig_vec[:, i]
            factor = beta_teleport + 2 * eig_val[i] * x_lazy * (1 - beta_teleport)

            vec_i = np.reshape(vec_i, (-1, 1))
            eigen_vec_mult = vec_i.dot(vec_i.T)
            Greens_matrix = Greens_matrix + (
                    eigen_vec_mult / factor)  # Greens function is the inverse of the beta-normalized laplacian
            beta_norm_lap = beta_norm_lap + (eigen_vec_mult * factor)  # beta-normalized laplacian

        temp = Greens_matrix.dot(inv_sqr_deg)
        temp = inv_sqr_deg.dot(temp) * beta_teleport
        hitting_matrix = np.zeros((N, N), float)
        diag_row = np.diagonal(temp)
        for i in range(N):
            hitting_matrix[i, :] = diag_row - temp[i, :]

        roundtrip_commute_matrix = hitting_matrix + hitting_matrix.T
        temp = Xv_Xu.dot(temp)
        final_hitting_times = np.diagonal(
            temp)  ## number_eig x 1 vector of hitting times from root (u) to number_eig of other nodes
        roundtrip_times = roundtrip_commute_matrix[root, :]
        return abs(final_hitting_times), roundtrip_times

    def _project_branch_probability_sc(self, bp_array_clus, pt):
        # print('sum of branch probabilities at cluster level', np.sum(bp_array_clus, axis=1))
        n_clus = len(list(set(self.labels)))
        n_cells = self.data.shape[0]

        knn_sc = 3 if self.data.shape[0] > 1000 else 10
        neighbors, _ = self.knn_struct.knn_query(self.data, k=knn_sc)

        rows, cols, weights = [], [], []
        for i, row in enumerate(neighbors):
            neighboring_clus = self.labels[row]
            for c in set(list(neighboring_clus)):
                rows.append(i)
                cols.append(c)
                weights.append(np.sum(neighboring_clus == c) / knn_sc)

        weights = csr_matrix((weights, (rows, cols)), shape=(n_cells, n_clus))

        bp_array_sc = weights.dot(bp_array_clus)
        bp_array_sc /= np.max(bp_array_sc,
                              axis=0)  # divide cell probability by max value in that column so that rare lineages dont have distortedly low lineage probabilities

        for i, label_ts in enumerate(self.terminal_clusters):  # list(set(self.terminal_clusters))

            loc_i = np.where(self.labels == label_ts)[0]
            loc_noti = np.where(self.labels != label_ts)[0]
            if np.max(bp_array_sc[loc_noti, i]) > 0.8: bp_array_sc[loc_i, i] = 1.2
        pt = np.asarray(pt)
        pt = np.reshape(pt, (n_clus, 1))
        pt_sc = weights.dot(pt)
        pt_sc /= np.amax(pt_sc)
        return bp_array_sc, pt_sc.flatten()

    def sc_transition_matrix(self, smooth_transition, b=10, use_sequentially_augmented=False):
        '''
        #computes the single cell level transition directions that are later used to calculate velocity of embedding
        #based on changes at single cell level in genes and single cell level velocity

        :param smooth_transition:
        :param b: slope of logistic function
        :return:
        '''
        # n_clus = len(list(set(self.labels)))
        # n_cells = self.data.shape[0]
        pt = self.single_cell_pt_markov * 2 / max(
            self.single_cell_pt_markov)  # some scaling so that the input to the logistic function covers a range of sensible inputs
        labels = self.labels
        if use_sequentially_augmented:
            T = self.csr_array_locally_pruned_augmented  # single cell edges are weighted as inverse of distance

        else:
            T = self.csr_array_locally_pruned
        # T = self.csr_full_graph #single cell

        thr_global = np.mean(T.data) - 2 * np.std(T.data)  # 2*
        T.data[T.data < thr_global] = 0
        T.setdiag(0)
        T.eliminate_zeros()

        size_T = T.size

        T.data = T.data.clip(np.percentile(T.data, 10), np.percentile(T.data, 90))
        find_T = find(T)
        bias_weight, K, c, C, nu = [], 1, 0, 1, 1
        bias_weight_pt, bias_weight_velo = [], []
        # print('transition before biasing')
        # print(T)
        if self.CSM is not None:
            # if rna velocity is available, then combine the single-cell directions inferred by sc-pt and scRNA velocity to get the sc directions
            for i in range(size_T):
                start = find_T[0][i]
                end = find_T[1][i]
                weight = find_T[2][i]
                t_dif = pt[find_T[1][i]] - pt[find_T[0][i]]

                delta_gene = np.array(self.gene_matrix[end, :] - self.gene_matrix[start,
                                                                 :])  # change in gene expression when going from start cell to end cells

                # print('shape delta_gene', delta_gene.shape, delta_gene)

                # csm_ = 1- distance.cosine(delta_gene[0,:], self.velocity_matrix[start, :])
                # print(csm_)
                csm_ = 0

                csm_ = cosine_sim(delta_gene, self.velocity_matrix[start, :])[0] * 10  # based on sc level CSM

                # csm_ = self.CSM[labels[find_T[0][i]],labels[find_T[1][i]]] #based on cluster level CSM
                # if (csm_<0)&(t_dif<0): csm_ *=10
                # if (csm_ >0) & (t_dif > 0): csm_ *= 10
                # if i%100000==0:  print(i,'out of', size_T, 'start', labels[start], 'end', labels[end], 'csm_', round(csm_,3), 't_diff', round(t_dif,3))
                # print('shape gene_matrix', self.gene_matrix.shape)
                # print('start-end, csm', start, end, csm_)
                # bias_weight_pt.append(t_dif)
                # bias_weight_velo.append(csm_)
                bias_weight_pt.append(1 * K / ((C + math.exp(b * (-t_dif + c))) ** nu))
                # bias_weight_velo.append(csm_)
                bias_weight_velo.append(1 * K / ((C + math.exp(b * (- csm_ + c))) ** nu))
            # scale both the pt and velocity weights
            bias_weight_pt = 2.0 * np.asarray(bias_weight_pt) / np.mean(
                np.array(bias_weight_pt))  # need to normalize the pt and velo weights
            # print('list shape',bias_weight_pt.shape)
            # print('argwhere is nan')
            # argwhere=np.argwhere(np.isnan(bias_weight_velo))
            # print(argwhere.shape)
            # print('bias weight velo is anywhere nan', np.isnan(bias_weight_velo).any(), np.mean(np.array(bias_weight_velo)))
            bias_weight_velo = np.nan_to_num(bias_weight_velo)
            bias_weight_velo = 2.0 * np.asarray(bias_weight_velo) / np.mean(np.array(bias_weight_velo))  # USE THIS
            # bias_weight_velo = np.asarray(bias_weight_velo)#
            # bias_weight_velo = np.clip(bias_weight_velo, 0, 1) #this approach zeros out negative edges
            # print('pt weight ', bias_weight_pt)
            bias_weight_velo = np.multiply(np.asarray(weight),
                                           bias_weight_velo)  ## consider changing this weight to the weight based on projected "i" -to- current "neighbor"
            bias_weight_pt = np.multiply(np.asarray(weight), bias_weight_pt)
            # print('velo weight * weigt', bias_weight_velo)
            # print('pt weight *weight', bias_weight_pt)
            bias_weight = (1 - self.velo_weight) * bias_weight_pt + self.velo_weight * bias_weight_velo
            # print('bias_weight size', bias_weight.shape)
            T = csr_matrix((bias_weight / np.mean(np.array(bias_weight)), (np.array(find_T[0]), np.array(find_T[1]))),
                           shape=T.shape)
            # print('sc transition weights T',np.isnan(T.data).any())
            T.data = np.nan_to_num(T.data)
            # print('sc transition weights T', np.isnan(T.data).any())
        else:
            # case where you dont have rna velocity, then the single-cell directions are only inferred using the single-cell pt
            b = 10
            for i in range(size_T):
                # start = find_T[0][i]
                # end = find_T[1][i]
                weight = find_T[2][i]
                t_dif = pt[find_T[1][i]] - pt[find_T[0][i]]
                bias_weight.append(weight * K / ((C + math.exp(b * (-t_dif + c))) ** nu))  # b * (-t_dif

            T = csr_matrix((bias_weight / np.mean(np.array(bias_weight)), (np.array(find_T[0]), np.array(find_T[1]))),
                           shape=T.shape)

        T.setdiag(0)

        T.eliminate_zeros()
        T = np.expm1(T)

        T = normalize(T, norm='l1', axis=1) ** smooth_transition

        T = T.multiply(csr_matrix(1.0 / np.abs(T).sum(1)))  # rows sum to one
        return T

    def _velocity_embedding(self, X_emb, smooth_transition, b, use_sequentially_augmented=False):
        '''

        :param X_emb:
        :param smooth_transition:
        :return: V_emb
        '''
        # T transition matrix at single cell level
        n_obs = X_emb.shape[0]
        V_emb = np.zeros(X_emb.shape)
        if self.single_cell_transition_matrix is None:
            self.single_cell_transition_matrix = self.sc_transition_matrix(smooth_transition, b,
                                                                           use_sequentially_augmented=use_sequentially_augmented)
            T = self.single_cell_transition_matrix
        else:
            T = self.single_cell_transition_matrix

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

    def _make_csrmatrix_noselfloop(self, neighbors: np.ndarray, distances: np.ndarray,
                                   auto_: bool = True, distance_factor=.01, weights_as_inv_dist=True,
                                   min_max_scale=False, time_series=False, time_series_labels=None,
                                   t_diff_step=2) -> csr_matrix:
        """
        Create sparse matrix from weighted knn graph. Default does local pruning and returns an affinity graph

        Parameters
        ----------
        neighbors: np.ndarray of shape (n_samples, n_neighbors)
            Indicating neighbors of each sample. neighbors[i,j] means that sample j is a neighbor of sample i

        distances: np.ndarray of shape (n_samples, n_neighbors)
            Distances between neighboring samples corresponding `neighbors`

        auto_: bool, default=True
            If `False` and `self.keep_all_local_dist = False` perform local pruning (according to self.edgepruning_clustering_resolution_local)
            and remove self-loops

        distance_factor: float, default=0.01
            Factor used in calculation of edge weights. mean(sqrt(distances))^2 / (sqrt(distances) + distance_factor)
        weights_as_inv_dist: bool, default = True
            Whether to convert the distances into weights that are proportional to inverse of the distance
        min_max_scale: bool, default = False
            This is called with True for constructing the sc-knn used for vertexclustergraph.
            we use the inverse of the distance but scaled from 0.5-2 such that we can combine the distance derived weights with the
            Jaccard similarity (0-1). The jaccard similiarity in igraph does not consider weights, so we re-introduce the impact of the
            distances which then play a role in the transition matrix for trajectories as well as the visualization of edge-weights
            In the case of the sc-KNN used for clustering, we do not re-introduce the distance based weights but leave it at the
            jaccard similarity as this empirically seems to work well
        time_series: bool, default = False. Whether or not this is a time course data set. will require time_series_labels in order to be actioned on
        time_series_labels: list, default = None. a list of numerical values reflecting the sampling time of each cell
        t_diff_step=int, default =2, number of sequential time steps permissible between edges. (e.g. edges allowed to be 2 or fewer time steps apart)
        Returns
        -------
            sparse matrix representing the locally pruned weighted knn affinity graph. edge weight represents level of similarity. higher values mean stronger edge
        """
        distances = np.sqrt(distances.astype(np.float32))
        # distances =distances.astype(np.float32)

        # print(neighbors[0,:])
        # print(distances[0,:])
        neigh01 = neighbors[0, 1]
        # print('norm2', np.linalg.norm(self.data[0,:]-self.data[neigh01,:]))
        if time_series:
            msk = np.full_like(distances, True, dtype=np.bool_)
            # print('all edges', np.sum(msk))
            # Remove self-loops
            msk &= (neighbors != np.arange(neighbors.shape[0])[:, np.newaxis])
            # print('non-self edges', np.sum(msk))

            '''
            #doing local pruning and then adding back the augmented edges does not work so well because when the edge weights are scaled and inverted,
            # the sequentially added edges appear very weak and noisy compared to the fairly strong edges that remain after the local pruning from the inital round of knngraph. If you retain all edges from initial graph construction,
            # then the average weight of edges is exaggeratedly higher than those edge weights from the sequentially added edges, and creates a better gradient of edge weights
            # Local pruning based on neighbor being too far. msk where we want to keep neighbors
            msk = distances <= (np.mean(distances, axis=1) + self.edgepruning_clustering_resolution_local * np.std(distances, axis=1))[:,
                               np.newaxis]
            # Remove self-loops
            msk &= (neighbors != np.arange(neighbors.shape[0])[:, np.newaxis])
            last_n_columns = self.knn_sequential+1
            msk[:,-last_n_columns:] = True # add back the edges belonging to knn-sequentially built part of the graph
            '''
            # remove edges between nodes that >t_diff_step far apart in time_series_labels

            if t_diff_step is not None:
                time_series_set_order = list(sorted(list(set(time_series_labels))))
                t_diff_mean = np.mean(
                    np.array([abs(y - x) for x, y in zip(time_series_set_order[:-1], time_series_set_order[1:])]))
                print(
                    f"{datetime.now()}\tActual average allowable time difference between nodes is {round(t_diff_mean * t_diff_step, 4)}")
            time_series_labels = np.asarray(time_series_labels)
            # print(colored(f"inside time_series msk"))

            rr = 0
            count = 0
            for row in neighbors:
                # if rr%20000==0: print(row, type(row), row[0])
                rr += 1
                t_row = time_series_labels[row[0]]  # first neighbor is itself

                if t_diff_step is not None:
                    for e_i, n_i in enumerate(row):
                        if abs(time_series_labels[n_i] - t_row) > t_diff_mean * t_diff_step:
                            count = count + 1
                            if np.sum(msk[row[0]]) > 4: msk[
                                row[0], e_i] = False  # we want to ensure that each cell has at least 5 nn
            # print('links within tdiff',np.sum(msk))

        elif auto_ and not self.keep_all_local_dist:
            # print('elif condition')
            # Local pruning based on neighbor being too far. msk where we want to keep neighbors
            msk = distances <= (np.mean(distances, axis=1) + self.edgepruning_clustering_resolution_local * np.std(distances, axis=1))[:,
                               np.newaxis]
            # Remove self-loops
            msk &= (neighbors != np.arange(neighbors.shape[0])[:, np.newaxis])

        else:

            msk = np.full_like(distances, True, dtype=np.bool_)
            # print('msk shape', msk.shape)
            # Remove self-loops
            msk &= (neighbors != np.arange(neighbors.shape[0])[:, np.newaxis])

            # print('msk shape', msk.shape)
        # Inverting the distances outputs values in range [0-1]. This also causes many ``good'' neighbors ending up
        # having a weight near zero (misleading as non-neighbors have a weight of zero). Therefore we scale by the
        # mean distance. The inversed weights without min_max_scaling are used for the community detection graph

        if self.distance == 'cosine':
            rows = np.array([np.repeat(i, len(x)) for i, x in enumerate(neighbors)])[msk]
            cols = neighbors[msk]
            weights = distances[msk]
            weights.fill(1)  # CHECK THIS
            # weights = logistic_function(weights)

        elif (weights_as_inv_dist == True) & (min_max_scale == False):

            # weights = (np.mean(distances[msk]) ** 2) / (distances[msk] + distance_factor) #larger weight is a stronger edge

            # weights = 1 / (distances[msk] + distance_factor)
            # do_exp_weight= True
            if self.do_gaussian_kernel_edgeweights == True:
                print('do_exp edge weights true in core_working')
                stdd = np.std(distances, axis=1)
                import numpy.ma as ma
                import umap.umap_ as u
                mx = ma.masked_array(distances, mask=distances == 0)

                minn = mx.min(axis=1)  # np.min(distances,axis=1) min of nonzero entries per row
                new_distances = np.exp(-(distances - minn[:, None]) / stdd[:, None])
                weights = new_distances[msk]
                sigmas, rhos = u.smooth_knn_dist(
                    distances, k=self.knn, local_connectivity=float(1.0))

                rows, cols, weights, dist_weights = u.compute_membership_strengths(
                    neighbors, distances, sigmas, rhos)

                # print('local connectivity', float(distances.shape[1]))
                # print('k', self.knn)
                # print('sigmas', len(sigmas), sigmas)
                # print('rhos', len(rhos), rhos)
                result = csr_matrix((weights, (rows, cols)), shape=(len(neighbors), len(neighbors)), dtype=np.float32)
                result.eliminate_zeros()

                # print('applying symmetry in umap_')
                transpose = result.transpose()
                # print('size of symmetry only', (transpose + result).size)
                prod_matrix = result.multiply(transpose)

                result = (result + transpose - prod_matrix)
                # print('after fuzzy symm result.data.size, max, mean', result.data.size, max(result.data), np.mean(result.data))
                return result
            else:
                weights = (np.mean(distances[msk]) ** 2) / (
                        distances[msk] +np.min(distances[msk]))#distance_factor)
                # larger weight is a stronger edge #Dec12 replacing distance_factor with np.min(distances)
                #print('np.min(distances) or distance_factor', np.min(distances[msk]), distance_factor)
            # weights = np.exp(-distances[msk]/stdd[:,None])
            rows = np.array([np.repeat(i, len(x)) for i, x in enumerate(neighbors)])[msk]
            cols = neighbors[msk]
        elif min_max_scale == True:  # the jaccard similarity does not consider edge weights, we will use these min-max scaled weights to make a composite weight together with Jaccard to retain the distance based info
            weights = distances[msk]  # larger weight is a stronger edge
            weights = np.clip(weights, a_min=np.percentile(weights, 10),
                              a_max=None)  # the first neighbor is usually itself and hence has distance 0
            # print('min clip',np.percentile(weights,10))
            # print('clipped weights for current pc-dist', weights)
            weights = 0.5 + (weights - np.min(weights)) * (2 - 0.5) / (np.max(weights) - np.min(weights))
            weights = 1 / weights  # all weights between 0.5 and 2. these will later be multiplied by the Jac similarity which are between 0-1 to create a composite weight

            rows = np.array([np.repeat(i, len(x)) for i, x in enumerate(neighbors)])[msk]
            cols = neighbors[msk]

        # result = scipy.sparse.coo_matrix( (weights, (rows, cols)), shape=(len(neighbors), len(neighbors)) )
        result = csr_matrix((weights, (rows, cols)), shape=(len(neighbors), len(neighbors)), dtype=np.float32)
        # result.eliminate_zeros()
        # transpose = result.transpose()

        # prod_matrix = result.multiply(transpose)
        # result = result + transpose# - prod_matrix
        return result

    def func_mode(self, ll):
        # return MODE of list
        # If multiple items are maximal, the function returns the first one encountered.
        return max(set(ll), key=ll.count)

    def make_JSON(self, folderpath='/home/user/JavaCode/basicgraph/', filename='VIA_JSON.js'):
        import networkx as nx

        from networkx.readwrite import json_graph
        from collections import defaultdict
        edgelist = self.edgelist_maxout

        weightlist = self.edgeweights_maxout
        min_w = min(weightlist)
        max_w = max(weightlist)
        weightlist = [(10 * (i - min_w) / (max_w - min_w)) + 1 for i in weightlist]
        n_groups = len(set(self.labels))
        group_pop = np.zeros([n_groups, 1])
        cluster_population_dict = {}
        for group_i in set(self.labels):
            loc_i = np.where(self.labels == group_i)[0]
            group_pop[group_i] = len(loc_i)  # np.sum(loc_i) / 1000 + 1
            cluster_population_dict[group_i] = len(loc_i)

        pop_max = cluster_population_dict[max(cluster_population_dict, key=cluster_population_dict.get)]
        pop_min = cluster_population_dict[min(cluster_population_dict, key=cluster_population_dict.get)]
        print(cluster_population_dict, pop_min, pop_max)
        pt_max = max(self.scaled_hitting_times)
        pt_min = min(self.scaled_hitting_times)
        scaled_hitting_times = [10 * (i - pt_min) / (pt_max - pt_min) for i in self.scaled_hitting_times]
        node_majority_truth_labels = []
        for ci, cluster_i in enumerate(sorted(list(set(self.labels)))):
            cluster_i_loc = np.where(np.asarray(self.labels) == cluster_i)[0]
            majority_truth = str(self.func_mode(list(np.asarray(self.true_label)[cluster_i_loc])))
            node_majority_truth_labels.append(majority_truth)

        majority_truth_labels_dict = dict(enumerate(node_majority_truth_labels))
        temp = defaultdict(lambda: len(temp))

        df_edges = pd.DataFrame(edgelist, columns=['source', 'target'])
        df_edges['weight'] = weightlist
        df_edges['distance'] = [150 / i for i in weightlist]
        G = nx.DiGraph()  # directed graph
        for key in majority_truth_labels_dict:
            print('node', key, majority_truth_labels_dict[key], round(scaled_hitting_times[key], 1),
                  cluster_population_dict[key],
                  temp[majority_truth_labels_dict[key]])
            # val denotes size in d3 by default
            G.add_node(key, group=majority_truth_labels_dict[key], pseudotime=scaled_hitting_times[key],
                       val=(10 * (cluster_population_dict[key] - pop_min) / (pop_max - pop_min)) + 1,
                       group_num=temp[majority_truth_labels_dict[key]])
        for enum_i, i in enumerate(edgelist):
            print('edge', i, weightlist[enum_i], cluster_population_dict[i[0]], 150 / weightlist[enum_i])
            if (scaled_hitting_times[i[0]] < scaled_hitting_times[i[1]]):
                source_node = i[0]
                target_node = i[1]
            else:
                source_node = i[1]
                target_node = i[0]
            # val edge controls number of emitted particles
            G.add_edge(u_of_edge=source_node, v_of_edge=target_node, weight=weightlist[enum_i],
                       val=(5 * (cluster_population_dict[source_node] - pop_min) / (pop_max - pop_min)) + 1,
                       distance=150 / weightlist[enum_i])

        # Visualize the network:
        nx.draw_networkx(G)
        plt.show()
        import json

        j = json_graph.node_link_data(G)

        js = json.dumps(['var gData=', j], ensure_ascii=False, indent=2)
        with open(folderpath + filename, "w") as file:
            file.write(js)
        return

    def run_toobig_subPARC(self, X_data, jac_std_toobig=1,
                           jac_weighted_edges=True):
        n_elements = X_data.shape[0]
        hnsw = _construct_knn(X_data, knn=self.knn, distance=self.distance, num_threads=self.num_threads, too_big=True)
        if self.knn >= 0.8 * n_elements:
            k = int(0.5 * n_elements)
        else:
            k = self.knn
        neighbor_array, distance_array = hnsw.knn_query(X_data, k=k)
        csr_array = self._make_csrmatrix_noselfloop(neighbor_array, distance_array)
        sources, targets = csr_array.nonzero()
        mask = np.zeros(len(sources), dtype=bool)
        mask |= (csr_array.data > (
                np.mean(csr_array.data) + np.std(csr_array.data) * 5))  # smaller distance means stronger edge
        csr_array.data[mask] = 0
        csr_array.eliminate_zeros()
        sources, targets = csr_array.nonzero()
        edgelist = list(zip(sources.tolist(), targets.tolist()))
        edgelist_copy = edgelist.copy()
        G = ig.Graph(edgelist, edge_attrs={'weight': csr_array.data.tolist()})
        sim_list = G.similarity_jaccard(pairs=edgelist_copy)  # list of jaccard weights
        new_edgelist = []
        sim_list_array = np.asarray(sim_list)
        if jac_std_toobig == 'median':
            threshold = np.median(sim_list)
        else:
            threshold = np.mean(sim_list) - jac_std_toobig * np.std(sim_list)
        strong_locs = np.where(sim_list_array > threshold)[0]
        for ii in strong_locs: new_edgelist.append(edgelist_copy[ii])
        sim_list_new = list(sim_list_array[strong_locs])

        if jac_weighted_edges == True:
            G_sim = ig.Graph(n=n_elements, edges=list(new_edgelist), edge_attrs={'weight': sim_list_new})
        else:
            G_sim = ig.Graph(n=n_elements, edges=list(new_edgelist))
        G_sim.simplify(combine_edges='sum')
        if jac_weighted_edges == True:
            if self.partition_type == 'ModularityVP':
                partition = leidenalg.find_partition(G_sim, leidenalg.ModularityVertexPartition, weights='weight',
                                                     n_iterations=self.n_iter_leiden, seed=self.random_seed)
                # print('partition type MVP')
            else:
                partition = leidenalg.find_partition(G_sim, leidenalg.RBConfigurationVertexPartition, weights='weight',
                                                     n_iterations=self.n_iter_leiden, seed=self.random_seed,
                                                     resolution_parameter=self.resolution_parameter)

        else:
            if self.partition_type == 'ModularityVP':
                # print('partition type MVP')
                partition = leidenalg.find_partition(G_sim, leidenalg.ModularityVertexPartition,
                                                     n_iterations=self.n_iter_leiden, seed=self.random_seed)
            else:
                print('partition type RBC')
                partition = leidenalg.find_partition(G_sim, leidenalg.RBConfigurationVertexPartition,
                                                     n_iterations=self.n_iter_leiden, seed=self.random_seed,
                                                     resolution_parameter=self.resolution_parameter)
        PARC_labels_leiden = np.asarray(partition.membership)
        PARC_labels_leiden = np.reshape(PARC_labels_leiden, (n_elements, 1))
        small_pop_list = []
        small_cluster_list = []
        small_pop_exist = False
        dummy, PARC_labels_leiden = np.unique(list(PARC_labels_leiden.flatten()), return_inverse=True)
        for cluster in set(PARC_labels_leiden):
            population = len(np.where(PARC_labels_leiden == cluster)[0])
            if population < 5:
                small_pop_exist = True
                small_pop_list.append(list(np.where(PARC_labels_leiden == cluster)[0]))
                small_cluster_list.append(cluster)

        for small_cluster in small_pop_list:
            for single_cell in small_cluster:
                old_neighbors = neighbor_array[single_cell, :]
                group_of_old_neighbors = PARC_labels_leiden[old_neighbors]
                group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                available_neighbours = set(group_of_old_neighbors) - set(small_cluster_list)
                if len(available_neighbours) > 0:
                    available_neighbours_list = [value for value in group_of_old_neighbors if
                                                 value in list(available_neighbours)]
                    best_group = max(available_neighbours_list, key=available_neighbours_list.count)
                    PARC_labels_leiden[single_cell] = best_group

        do_while_time = time.time()
        while (small_pop_exist == True) & (time.time() - do_while_time < 5):
            small_pop_list = []
            small_pop_exist = False
            for cluster in set(list(PARC_labels_leiden.flatten())):
                population = len(np.where(PARC_labels_leiden == cluster)[0])
                if population < 10:
                    small_pop_exist = True
                    small_pop_list.append(np.where(PARC_labels_leiden == cluster)[0])
            for small_cluster in small_pop_list:
                for single_cell in small_cluster:
                    old_neighbors = neighbor_array[single_cell, :]
                    group_of_old_neighbors = PARC_labels_leiden[old_neighbors]
                    group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                    best_group = max(set(group_of_old_neighbors), key=group_of_old_neighbors.count)
                    PARC_labels_leiden[single_cell] = best_group

        dummy, PARC_labels_leiden = np.unique(list(PARC_labels_leiden.flatten()), return_inverse=True)
        self.labels = PARC_labels_leiden
        return PARC_labels_leiden

    def find_root_group(self, graph_dense, PARC_labels_leiden, root_user, true_labels, super_cluster_labels_sub,
                        super_node_degree_list):
        # PARC_labels_leiden is the subset belonging to the component of the graph being considered. graph_dense is a component of the full graph
        # returns the cluster most likely corresponding to the root (initial) state on the cluster the graph
        majority_truth_labels = np.empty((len(PARC_labels_leiden), 1), dtype=object)
        graph_node_label = []
        min_deg = 1000
        super_min_deg = 1000
        found_super_and_sub_root = False
        found_any_root = False
        true_labels = np.asarray(true_labels)

        deg_list = graph_dense.sum(axis=1).reshape((1, -1)).tolist()[0]

        for ci, cluster_i in enumerate(sorted(list(set(PARC_labels_leiden)))):
            cluster_i_loc = np.where(np.asarray(PARC_labels_leiden) == cluster_i)[0]
            majority_truth = str(self.func_mode(list(true_labels[cluster_i_loc])))
            if self.super_cluster_labels != False:
                super_majority_cluster = self.func_mode(list(np.asarray(super_cluster_labels_sub)[cluster_i_loc]))
                super_majority_cluster_loc = np.where(np.asarray(super_cluster_labels_sub) == super_majority_cluster)[0]
                super_majority_truth = self.func_mode(list(true_labels[super_majority_cluster_loc]))
                super_node_degree = super_node_degree_list[super_majority_cluster]
                if (str(root_user) in majority_truth) & (str(root_user) in str(super_majority_truth)):
                    if super_node_degree < super_min_deg:
                        found_super_and_sub_root = True
                        root = cluster_i
                        found_any_root = True
                        min_deg = deg_list[ci]
                        super_min_deg = super_node_degree
                        print(f"{datetime.now()}\tNew root is {root}")
            majority_truth_labels[cluster_i_loc] = str(majority_truth) + 'c' + str(cluster_i)

            graph_node_label.append(str(majority_truth) + 'c' + str(cluster_i))
        if (self.super_cluster_labels == False) | (found_super_and_sub_root == False):
            # print('self.super_cluster_labels', super_cluster_labels_sub, ' foundsuper_cluster_sub and super root',found_super_and_sub_root)
            for ic, cluster_i in enumerate(sorted(list(set(PARC_labels_leiden)))):
                cluster_i_loc = np.where(np.asarray(PARC_labels_leiden) == cluster_i)[0]
                # print('cluster', cluster_i, 'set true labels', set(true_labels))
                true_labels = np.asarray(true_labels)

                majority_truth = str(self.func_mode(list(true_labels[cluster_i_loc])))

                print('cluster', cluster_i, 'has majority', majority_truth)  # , 'with degree list', deg_list)

                if (str(root_user) in str(majority_truth)):  # 'in' not ==
                    if deg_list[ic] < min_deg:
                        root = cluster_i
                        found_any_root = True
                        min_deg = deg_list[ic]
                        print(f"{datetime.now()}\tNew root is {root} and majority {majority_truth}")
        # print('len graph node label', graph_node_label)
        if found_any_root == False:
            print(f"{datetime.now()}\tSetting arbitrary root {cluster_i}")
            root = cluster_i
        return graph_node_label, majority_truth_labels, deg_list, root

    def make_majority_truth_cluster_labels(self, PARC_labels_leiden, true_labels, graph_dense):
        majority_truth_labels = np.empty((len(PARC_labels_leiden), 1), dtype=object)
        graph_node_label = []
        true_labels = np.asarray(true_labels)
        deg_list = graph_dense.sum(axis=1).reshape((1, -1)).tolist()[0]

        for ci, cluster_i in enumerate(sorted(list(set(PARC_labels_leiden)))):
            cluster_i_loc = np.where(np.asarray(PARC_labels_leiden) == cluster_i)[0]
            majority_truth = str(self.func_mode(list(true_labels[cluster_i_loc])))

            majority_truth_labels[cluster_i_loc] = str(majority_truth) + 'c' + str(cluster_i)

            graph_node_label.append(str(majority_truth) + 'c' + str(cluster_i))
            deg_list = graph_dense.sum(axis=1).reshape((1, -1)).tolist()[0]

        return graph_node_label, majority_truth_labels, deg_list

    def find_root(self, graph_dense, PARC_labels_leiden, root_user, true_labels):
        # root-user is the singlecell index given by the user when running VIA
        majority_truth_labels = np.empty((len(PARC_labels_leiden), 1), dtype=object)
        graph_node_label = []
        true_labels = np.asarray(true_labels)

        deg_list = graph_dense.sum(axis=1).reshape((1, -1)).tolist()[0]

        for ci, cluster_i in enumerate(sorted(list(set(PARC_labels_leiden)))):
            # print('cluster i', cluster_i)
            cluster_i_loc = np.where(np.asarray(PARC_labels_leiden) == cluster_i)[0]

            majority_truth = self.func_mode(list(true_labels[cluster_i_loc]))

            majority_truth_labels[cluster_i_loc] = str(majority_truth) + 'c' + str(cluster_i)

            graph_node_label.append(str(majority_truth) + 'c' + str(cluster_i))
        root = PARC_labels_leiden[root_user]
        return graph_node_label, majority_truth_labels, deg_list, root

    def find_root_2Morgan(self, graph_dense, PARC_labels_leiden, root_idx, true_labels):
        # single cell index given corresponding to user defined root cell
        majority_truth_labels = np.empty((len(PARC_labels_leiden), 1), dtype=object)
        graph_node_label = []
        true_labels = np.asarray(true_labels)

        deg_list = graph_dense.sum(axis=1).reshape((1, -1)).tolist()[0]
        secondary_annotations = np.asarray(self.secondary_annotations)
        for ci, cluster_i in enumerate(sorted(list(set(PARC_labels_leiden)))):
            cluster_i_loc = np.where(np.asarray(PARC_labels_leiden) == cluster_i)[0]

            majority_truth = self.func_mode(list(true_labels[cluster_i_loc]))
            majority_truth_secondary = str(self.func_mode(list(secondary_annotations[cluster_i_loc])))
            majority_truth_labels[cluster_i_loc] = str(majority_truth) + 'c' + str(cluster_i)

            # graph_node_label.append(str(majority_truth)[0:5] + 'C' + str(cluster_i))
            graph_node_label.append(str(majority_truth)[0:5] + 'C' + str(cluster_i) + str(majority_truth_secondary))
        root = PARC_labels_leiden[root_idx]
        return graph_node_label, majority_truth_labels, deg_list, root

    def full_graph_paths(self, data, n_components_original=1):
        # make igraph object of very low-K KNN using the knn_struct PCA-dimension space made in PARC.
        # This is later used by find_shortest_path for sc_bp visual
        # neighbor array is not listed in in any order of proximity

        print(f"{datetime.now()}\tThe number of components in the original full graph is {n_components_original}")
        print(f"{datetime.now()}\tFor downstream visualization purposes we are also constructing a low knn-graph ")
        first, k0, n_comp = True, 3, n_components_original + 1
        while (n_components_original == 1 and n_comp > 1) or \
                (n_components_original > 1 and k0 <= 5 and n_comp > n_components_original):
            neighbors, distances = self.knn_struct.knn_query(data, k=k0)
            csr_array = self._make_csrmatrix_noselfloop(neighbors, distances, auto_=False)
            n_comp, comp_labels = connected_components(csr_array, return_labels=True)
            if first:
                first = False
            else:
                k0 += 1

        # print(f"{datetime.now()}\t the size of neighbor array in low-KNN in pca-space for visualization is {neighbors.shape}")
        n_cells, n_neighbors = neighbors.shape
        rows = np.transpose(np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten()
        cols = neighbors.flatten()
        csr_full_graph = csr_matrix((distances.flatten(), (rows, cols)), shape=(n_cells, n_cells))

        return ig.Graph(list(zip(*csr_full_graph.nonzero()))).simplify(combine_edges='sum')



    def do_impute(self, df_gene, magic_steps=3, gene_list=[]):
        # ad_gene is an ann data object from scanpy
        # normalize across columns to get Transition matrix.
        transition_full_graph = normalize(self.csr_full_graph, norm='l1', axis=1) ** magic_steps

        print('shape of transition matrix raised to power', magic_steps, transition_full_graph.shape)
        subset = df_gene[gene_list].values
        return pd.DataFrame(transition_full_graph.dot(subset), index=df_gene.index, columns=gene_list)

    def run_subVIA(self):

        # Construct graph or obtain from previous run
        if self.is_coarse:

            neighbors, distances = self.knn_struct.knn_query(self.data,
                                                             k=self.knn)  # these n, d will be used as a basis for both the clustergraph and graph which we apply community detection. These two graphs are constructed differently because in the community detection we want to emphasize dissimiliarities,
            # but the in the clustergraph we want to enhance connectivity
            adjacency_augmented = None
            adjacency = None
            if (self.time_series == True) and (
                    self.time_series_labels is not None):  # refine the graph for clustering (leiden) using time_series labels if available
                print(f"{datetime.now()}\tUsing time series information to guide knn graph construction ")

                n_augmented, d_augmented = sequential_knn(self.data, self.time_series_labels, neighbors, distances,
                                                          k_seq=self.knn_sequential,
                                                          k_reverse=self.knn_sequential_reverse,
                                                          num_threads=self.num_threads, distance=self.distance)

                adjacency_augmented = self._make_csrmatrix_noselfloop(n_augmented, d_augmented,
                                                                      time_series=self.time_series,
                                                                      time_series_labels=self.time_series_labels,
                                                                      t_diff_step=self.t_diff_step)  # this function has local pruning which removes neighbors that are more than t_dif apart. Since the same type of local pruning wrt t_dif is applied pre-clustergraph, we only need to call this function once in the case of time_series data


                adjacency = self._make_csrmatrix_noselfloop(neighbors, distances)  # not augmented or t_diff'ed

            if (self.do_spatial_knn) and (
                    self.spatial_coords is not None):  # refine the graph for clustering (leiden) using time_series labels if available

                if len(self.spatial_aux) == 0:
                    print(f"{datetime.now()}\tSince no slice-labels were provided, all cells are assumed to be from the same tissue slice ")
                    self.spatial_aux =  ['slice1'] * self.nsamples#['slice1' for i in range(self.nsamples)]
                if adjacency_augmented is None: #we did not augment based on time series data
                    print(f"{datetime.now()}\tUsing spatial information to guide knn graph construction. Note! If cells are from different slices of tissue, provide a list of tissue-slice IDs ")
                    #n_augmented, d_augmented = spatial_knn(coords=self.spatial_coords, neighbors=neighbors,                                                           distances=distances,                                                           k_spatial=self.spatial_knn)
                    n_augmented, d_augmented=                    spatial_knn_new(spatial_coords=self.spatial_coords, spatial_slice_labels=self.spatial_aux,
                                    neighbors=neighbors,
                                    distances=distances, k_spatial=self.spatial_knn, distance_metric='l2',
                                    num_threads=-1, too_big=False)
                    adjacency_augmented = self._make_csrmatrix_noselfloop(n_augmented, d_augmented)  # this function has local pruning which removes neighbors that are more than t_dif apart. Since the same type of local pruning wrt t_dif is applied pre-clustergraph, we only need to call this function once in the case of time_series data
                else:
                    print(f"{datetime.now()}\tUsing spatial information to guide knn graph construction + combining with time-series data")
                    n_augmented, d_augmented = spatial_knn_new(spatial_coords=self.spatial_coords,  spatial_slice_labels = self.spatial_aux, neighbors=n_augmented,
                                                           distances=d_augmented,     k_spatial=self.spatial_knn,distance_metric = 'l2', num_threads= -1,                     too_big= False)
                    adjacency_augmented = self._make_csrmatrix_noselfloop(n_augmented, d_augmented)
                if adjacency is  None: adjacency = self._make_csrmatrix_noselfloop(neighbors, distances)  # not augmented or t_diff'ed

            if adjacency is None:
                adjacency = self._make_csrmatrix_noselfloop(neighbors,
                                                            distances)  # , min_max_scale=True) #this function has local pruning based on edge distances


        else:
            neighbors, distances = self.full_neighbor_array, self.full_distance_array
            adjacency = self.csr_array_locally_pruned

        edges = np.array(list(zip(*adjacency.nonzero())))
        sim = np.array(
            ig.Graph(n=self.nsamples, edges=edges.tolist(), edge_attrs={'weight': adjacency.data}).similarity_jaccard(
                pairs=edges))  # used to do igraph jaccard does not use  weight information to compute Jacc
        # sim = adjacency.data
        tot = len(sim)

        # Prune Jacc weighted edges off graph on global level
        threshold = np.median(sim) if self.edgepruning_clustering_resolution == 'median' else sim.mean() - self.edgepruning_clustering_resolution * sim.std()
        strong_locs = np.asarray(np.where(sim > threshold)[0])
        # to be used for leiden paritionining clustering
        g_local_global_jac = ig.Graph(n=self.nsamples, edges=list(edges[strong_locs]),
                                      edge_attrs={'weight': list(sim[strong_locs])}).simplify(
            combine_edges='sum')  # used for the clustering step and has been locally and globally pruned
        # jac_sim_list = g_local_global_jac.similarity_jaccard(pairs=list(edges[strong_locs]))
        # g_local_global_jac =ig.Graph(n=self.nsamples, edges=list(edges[strong_locs]), edge_attrs={'weight': jac_sim_list}).simplify(combine_edges='sum') #used for clustering (globally and locally pruned) and then jacc weighted (Jacc does not consider weights)
        self.sparse_glob_loc_pruned = get_sparse_from_igraph(g_local_global_jac, weight_attr='weight')

        print(
            f"{datetime.now()}\tFinished global pruning of {self.knn}-knn graph used for clustering at level of {self.edgepruning_clustering_resolution}. Kept {round(100 * len(strong_locs) / tot, 1)} % of edges. ")

        if self.is_coarse:
            # Construct full graph with no pruning - used for cluster graph edges, not listed in any order of proximity
            if (self.time_series == True) and (self.time_series_labels is not None):

                csr_full_graph = adjacency_augmented.copy()  # when time_series data is available
                if self.RW2_mode:
                    print("saving csr for RW2")
                    from scipy.sparse import save_npz
                    str_date = str(str(datetime.now())[-3:])
                    print(f'Unique ID for RW2 file {str_date}')
                    save_npz(self.working_dir_fp + 'pc' + str(self.ncomp) + '_knn' + str(self.knn) + 'kseq' + str(
                        self.knn_sequential) + 'krev' + str(self.knn_sequential_reverse) + 'RW2_' + str_date + '.npz',
                             csr_full_graph)

            # when no time-series data is available, in this iteration of weighting the edges we do not locally prune based on edge-weights becasue we intend to use all neighborhood info towards the edges in the clustergraph and ensuring the clustergraph is connected
            elif (self.do_spatial_knn == True):
                print(f"{datetime.now()}\tusing spatial coords to augment clustergraph")
                csr_full_graph = adjacency_augmented.copy()
            else:
                csr_full_graph = self._make_csrmatrix_noselfloop(neighbors, distances, auto_=False, min_max_scale=False,
                                                                 time_series=self.time_series,
                                                                 time_series_labels=self.time_series_labels)  # no local pruning: auto_ set to false #min_max_scale=True
            n_original_comp, n_original_comp_labels = connected_components(csr_full_graph, directed=False)
            print(f"{datetime.now()}\tNumber of connected components used for clustergraph  is {n_original_comp}")

            if ((self.pca_loadings is not None) & (self.velocity_matrix is not None)) & (self.gene_matrix is not None):

                edges_preClustergraph = list(
                    zip(*csr_full_graph.nonzero()))  # in the case of time-series, this is the augmented graph
                # projected_distances are the inverted Projected_distances (affinities) based on velocity adjusted location of neighbors
                projected_distances = get_projected_distances(loadings=self.pca_loadings, gene_matrix=self.gene_matrix,
                                                              velocity_matrix=self.velocity_matrix,
                                                              edgelist=edges_preClustergraph, current_pc=self.data)

                self.ig_full_graph = _composite_jacAffinity_distanceAffinity(sc_csr_graph=csr_full_graph,
                                                                             projected_distances=projected_distances)

            else:
                # this will be used for making the clustergraph. Note that in the case of time-series, this is the augmented graph with weights given by a composite metric of Jaccard and affinities
                self.ig_full_graph = _composite_jacAffinity_distanceAffinity(
                    csr_full_graph)  # 0<J<1 #combining distance-based affinities and jac-based affinities
            # for VIA we local and global prune the vertex cluster graph *after* making the clustergraph
            self.csr_array_locally_pruned = adjacency  # this graph has only been locally pruned and in the case of time-series, it is not the augmented knn. it then subsequently in later code will be subject to global pruning, followed by Leiden clustering. saving it here for non-coarse runs

            if self.time_series == True:
                self.csr_array_locally_pruned_augmented = adjacency_augmented  # t_diff edges removed and is sequentially augmented affinity graph (no Jaccard metric is added)
            else:
                self.csr_array_locally_pruned_augmented = None

            self.csr_full_graph = csr_full_graph  # for timeseries, this is the sequentially augmented graph, and later used in sc_transitions. This can also be used for hnsw_umap
            self.full_neighbor_array = neighbors
            self.full_distance_array = distances

            # knn graph used for making trajectory drawing on the visualization
            # self.full_graph_shortpath = self.full_graph_paths(self.data, n_original_comp)
            neighbors = self.full_neighbor_array
        if self.labels is None:
            print(
                f"{datetime.now()}\tCommencing community detection")  # this is done on the non-time-series augmented graph to avoid clustering largely based on prior known time-level data
            weights = 'weight' if self.jac_weighted_edges else None
            # type = leidenalg.ModularityVertexPartition if self.partition_type == 'ModularityVP' else leidenalg.RBConfigurationVertexPartition
            if self.partition_type == 'ModularityVP':
                partition = leidenalg.find_partition(g_local_global_jac, leidenalg.ModularityVertexPartition,
                                                     weights=weights,
                                                     n_iterations=self.n_iter_leiden, seed=self.random_seed)
                # print('partition type MVP')
            else:
                partition = leidenalg.find_partition(g_local_global_jac, leidenalg.RBConfigurationVertexPartition,
                                                     weights=weights,
                                                     n_iterations=self.n_iter_leiden, seed=self.random_seed,
                                                     resolution_parameter=self.resolution_parameter)
            # partition = leidenalg.find_partition(g_local_global_jac, partition_type=type, weights=weights,                                             n_iterations=self.n_iter_leiden, seed=self.random_seed)
            labels = np.array(partition.membership)

            print(f"{datetime.now()}\tFinished community detection. Found {len(set(labels))} clusters.")

            # Searching for clusters that are too big and split them
            too_big_clusters = [k for k, v in Counter(labels).items() if v > self.too_big_factor * self.nsamples]
            if len(too_big_clusters):
                print(f"{datetime.now()}\tFound {len(too_big_clusters)} clusters that are too big")

            time0_big = time.time()
            count_big_pop = len(too_big_clusters)
            num_times_expanded = 0
            # TODO - add max running time condition
            while len(too_big_clusters) > 0 & (
                    not ((time.time() - time0_big > 200) & (num_times_expanded >= count_big_pop))):
                # while len(too_big_clusters) & (not((time.time() - time0_big >200) & (num_times_expanded >= count_big_pop))):
                print(f"{datetime.now()}\tExamining clusters that are above the too_big threshold")
                cluster = too_big_clusters.pop(0)
                idx = labels == cluster
                print(f"{datetime.now()}\tCluster {cluster} contains "
                      f"{idx.sum()}>{round(self.too_big_factor * self.nsamples)} samples and is too big")

                data = self.data[idx]
                membership = max(labels) + 1 + np.array(self.run_toobig_subPARC(data))
                num_times_expanded += 1

                if len(set(membership)) > 1:
                    labels[idx] = membership
                    too_big_clusters.extend(
                        [k for k, v in Counter(membership).items() if v > self.too_big_factor * self.nsamples])
                else:
                    print(f"{datetime.now()}\tCould not expand cluster {cluster}")

            # Search for clusters that are too small (like singletons) and merge them to non-small clusters based on neighbors' majority vote
            # first we make a quick pass through all clusters to remove very small outliers by merging with a larger cluster
            # print('before final small cluster handling we have',len(set(labels)), 'communities')

            too_small_clusters = {k for k, v in Counter(labels).items() if v < self.small_pop}
            print(f"{datetime.now()}\tMerging {len(set(too_small_clusters))} very small clusters (<{self.small_pop})")
            idx = np.where(np.isin(labels, list(too_small_clusters)))[0]
            neighbours_labels = labels[neighbors[idx]]
            for i, nl in zip(*[idx, neighbours_labels]):
                # Retrieve the first non small cluster, with highest number of neighbours
                label = next((label for label, n in Counter(nl).most_common() if label not in too_small_clusters), None)
                # label = next((label for label, n in Counter(nl).most_common()), None)
                if label is not None:  # recall 0 is a valid label value
                    labels[i] = label

                # too_small_clusters = {k for k, v in Counter(labels).items() if v < self.small_pop}
            too_small_clusters = {k for k, v in Counter(labels).items() if v < self.small_pop}
            # in this pass we allow clusters to be merged even if they are not a Large Cluster.. as multiple smaller ones might come together to form an acceptably large cluster
            do_while_time = time.time()
            while len(too_small_clusters) & (time.time() - do_while_time < 15):
                # membership of neighbours of samples in small clusters
                idx = np.where(np.isin(labels, list(too_small_clusters)))[0]
                neighbours_labels = labels[neighbors[idx]]
                for i, nl in zip(*[idx, neighbours_labels]):
                    # Retrieve the first non small cluster, with highest number of neighbours
                    # label = next((label for label, n in Counter(nl).most_common() if label not in too_small_clusters), None)
                    label = next((label for label, n in Counter(nl).most_common()), None)
                    if label is not None:  # recall 0 is a valid label value
                        labels[i] = label
                # Update set of too small clusters, stopping if converged
                too_small_clusters = {k for k, v in Counter(labels).items() if v < self.small_pop}
            # Reset labels to begin from zero and with no missing numbers
            self.labels = labels = np.unique(labels, return_inverse=True)[1]

            print(f"{datetime.now()}\tFinished detecting communities. Found", len(set(self.labels)), 'communities')
        else:
            print(f'{datetime.now()}\tUsing predfined labels provided by user (this must be provided as an array)')

        # end community detection
        # do kmeans instead
        '''
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=20, random_state=1).fit(X_data)
        self.labels = kmeans.labels_
        n_clus = len(set(self.labels))
        self.labels = kmeans.labels_.flatten().tolist()

        pop_list = []
        pop_list_raw = []
        for item in range(len(set(PARC_labels_leiden))):
            pop_item = PARC_labels_leiden.count(item)
            pop_list.append((item, pop_item))
            pop_list_raw.append(pop_item)
        '''
        # Make cluster-graph
        print(
            f"{datetime.now()}\tMaking cluster graph. Global cluster graph pruning level: {self.cluster_graph_pruning}")
        graph = ig.VertexClustering(self.ig_full_graph, membership=self.labels).cluster_graph(combine_edges='sum')

        graph = recompute_weights(graph, Counter(self.labels))  # returns csr matrix

        self.cluster_graph_csr_not_pruned = graph  # type csr NOT PRUNED
        edgeweights_pruned_clustergraph, edges_pruned_clustergraph, comp_labels = pruning_clustergraph(graph,
                                                                                                       global_pruning_std=self.cluster_graph_pruning,
                                                                                                       preserve_disconnected=self.preserve_disconnected,
                                                                                                       preserve_disconnected_after_pruning=self.preserve_disconnected_after_pruning, do_max_outgoing=self.do_clustergraph_edgecontrol)

        self.connected_comp_labels = comp_labels
        # ig.graph() creates an undirected graph,in which case get_spare_from_igraph will be symmetric for each edge (u,v) and edge (v,u) is created
        locallytrimmed_g = ig.Graph(edges_pruned_clustergraph, edge_attrs={
            'weight': edgeweights_pruned_clustergraph})  # .simplify(combine_edges='sum') #forward biasing happens later

        locallytrimmed_sparse_vc = get_sparse_from_igraph(locallytrimmed_g, weight_attr='weight')

        if self.edgebundle_pruning == self.cluster_graph_pruning:

            weights_for_layout = np.asarray(locallytrimmed_sparse_vc.data)
            # clip weights to prevent distorted visual scale
            weights_for_layout = np.clip(weights_for_layout, np.percentile(weights_for_layout, 10),
                                         np.percentile(weights_for_layout,
                                                       90))  # want to clip the weights used to get the layout
            weights_for_layout = list(weights_for_layout)

            g_layout = ig.Graph(list(zip(*locallytrimmed_sparse_vc.nonzero())),
                                edge_attrs={'weight': weights_for_layout})


        # if using a different pruning for visualized edge bundles than the one specified in self.cluster_graph_pruning
        else:

            edgeweights_layout, edges_layout, comp_labels_layout = pruning_clustergraph(graph,
                                                                                        global_pruning_std=self.edgebundle_pruning,
                                                                                        preserve_disconnected=self.preserve_disconnected,
                                                                                        preserve_disconnected_after_pruning=self.preserve_disconnected_after_pruning,do_max_outgoing=self.do_clustergraph_edgecontrol)

            # layout = locallytrimmed_g.layout_fruchterman_reingold(weights='weight') #uses non-clipped weights but this can skew layout due to one or two outlier edges
            layout_g = ig.Graph(edges_layout, edge_attrs={
                'weight': edgeweights_layout})  # .simplify(combine_edges='sum') remving the 'combined_edges='sum' as it does not handle diagonal self loops which we require to be retained for singleton clusters that are not connected to other clusters

            layout_g_csr = get_sparse_from_igraph(layout_g, weight_attr='weight')
            weights_for_layout = np.asarray(layout_g_csr.data)

            for i in range(weights_for_layout.shape[0]):
                if weights_for_layout[i] == 0: print(i)
            # clip weights to prevent distorted visual scale in layout
            weights_for_layout = np.clip(weights_for_layout, np.percentile(weights_for_layout, 10),
                                         np.percentile(weights_for_layout, 90))
            weights_for_layout = list(weights_for_layout)

            g_layout = ig.Graph(list(zip(*layout_g_csr.nonzero())), edge_attrs={'weight': weights_for_layout})
        self.clustergraph_igraph_forlayout = g_layout
        # the layout of the graph is determine by a pruned clustergraph and the directionality of edges will be based on the final markov pseudotimes
        # the edgeweights of the bundle-edges is determined by the distance based metrics and jaccard similarities and not by the pseudotimes
        # for the transition matrix used in the markov pseudotime and differentiation probability computations, the edges will be further biased by the hittings times and markov pseudotimes

        # layout = g_layout.layout_fruchterman_reingold(weights='weight')

        # globally trimmed link
        self.edgelist_unique = set(
            tuple(sorted(l)) for l in zip(*locallytrimmed_sparse_vc.nonzero()))  # keep only one of (0,1) and (1,0)
        self.edgelist = edges_pruned_clustergraph  # after one round of global and local pruning of graph to be used for MCMCs

        # number of components
        n_components, labels_cc = connected_components(csgraph=locallytrimmed_sparse_vc, directed=False,
                                                       return_labels=True)

        df_graph = pd.DataFrame(locallytrimmed_sparse_vc.todense())

        if (self.velocity_matrix is not None) & (self.gene_matrix is not None):
            df_velocity = pd.DataFrame(self.velocity_matrix)
            print('size velocity matrix', len(self.labels), self.velocity_matrix.shape)
            df_velocity['labels'] = self.labels
            velocity_mean = df_velocity.groupby('labels', as_index=True).mean().to_numpy()
            gene_mean = pd.DataFrame(self.gene_matrix)
            gene_mean['labels'] = self.labels
            gene_mean = gene_mean.groupby('labels', as_index=True).mean().to_numpy()

            self.A_velo, self.CSM, velo_root_top3 = velocity_transition(A=df_graph.values, V=velocity_mean,
                                                                        G=gene_mean)  # velocity directed transition matrix

            print('cell type of suggested roots:', [self.true_label[x] for x in velo_root_top3])
            if self.time_series == True:  print('cell stage of suggested roots:',
                                                [self.time_series_labels[x] for x in velo_root_top3])

        df_graph['cc'] = labels_cc
        df_graph['pt'] = float('NaN')
        df_graph['majority_truth'] = 'maj truth'
        df_graph['graph_node_label'] = 'node label'

        PARC_labels_leiden = self.labels
        set_parc_labels = list(set(PARC_labels_leiden))
        set_parc_labels.sort()

        tsi_list = []
        df_graph['markov_pt'] = float('NaN')
        terminal_clus = []  # list of cluster labels that are found to be terminal clusters
        node_deg_list = []
        super_terminal_clus_revised = []
        pd_columnnames_terminal = []
        dict_terminal_super_sub_pairs = {}
        self.root = []
        large_components = []
        for comp_i in range(n_components):
            loc_compi = np.where(labels_cc == comp_i)[0]

            if len(loc_compi) > 1:  # need at least 2 nodes in in a component to make edges
                large_components.append(comp_i)
            elif len(loc_compi) == 0:
                df_graph.at[loc_compi[0], 'markov_pt'] = 1  # 0
                df_graph.at[loc_compi[0], 'pt'] = 1  # 0

        for comp_i in large_components:  # range(n_components):
            print(f'{datetime.now()}\tcomponent number', comp_i, 'out of ', large_components)
            loc_compi = np.where(labels_cc == comp_i)[0]

            a_i = df_graph.iloc[loc_compi][loc_compi].values  # transition matrix of relevant component
            # A_velo, CSM = velocity_transition(A=a_i, V=velocity_mean,   G=gene_mean)  # velocity directed transition matrix
            if self.A_velo is not None: A_velo, CSM = self.A_velo[loc_compi, :][loc_compi, :], self.CSM[loc_compi, :][
                                                                                               loc_compi, :]
            a_i = csr_matrix(a_i, (a_i.shape[0], a_i.shape[0]))

            cluster_labels_subi = [x for x in loc_compi]
            # print('cluster_labels_subi', cluster_labels_subi)
            sc_labels_subi = [PARC_labels_leiden[i] for i in range(len(PARC_labels_leiden)) if
                              (PARC_labels_leiden[i] in cluster_labels_subi)]

            sc_truelabels_subi = [self.true_label[i] for i in range(len(PARC_labels_leiden)) if
                                  (PARC_labels_leiden[i] in cluster_labels_subi)]

            if (self.root_user is None) and (self.velocity_matrix is not None):
                # in case the root cluster is a singleton, move on to the next nominee in initial states
                root_i = velo_root_top3[0]
                cluster_i_loc = np.where(np.asarray(sc_labels_subi) == root_i)[0]
                if len(cluster_i_loc) < 2:
                    root_i = velo_root_top3[1]
                    cluster_i_loc = np.where(np.asarray(sc_labels_subi) == root_i)[0]
                    if len(cluster_i_loc) < 2:
                        root_i = velo_root_top3[2]
                        cluster_i_loc = np.where(np.asarray(sc_labels_subi) == root_i)[0]
                        if len(cluster_i_loc) < 2:
                            root_i = velo_root_top3[0]

                graph_node_label, majority_truth_labels, node_deg_list_i = self.make_majority_truth_cluster_labels(
                    PARC_labels_leiden=sc_labels_subi, true_labels=sc_truelabels_subi, graph_dense=a_i)
                for velo_root_candidate_i in velo_root_top3:
                    cluster_i_loc = np.where(np.asarray(sc_labels_subi) == velo_root_candidate_i)[0]
                    majority_truth = str(self.func_mode(list(np.asarray(sc_truelabels_subi)[cluster_i_loc])))
                    if velo_root_candidate_i == root_i: majority_truth_veloroot0 = majority_truth
                    print(
                        f"{datetime.now()}\tUsing the RNA velocity graph, A top3 candidate for initial state is {velo_root_candidate_i} comprising predominantly of {majority_truth} cells",
                    )
                print(
                    f"{datetime.now()}\tUsing the RNA velocity graph, the suggested initial root state is {root_i} comprising predominantly of {majority_truth_veloroot0} cells")


            elif (self.dataset in ['toy', 'faced', 'mESC', 'iPSC', 'group']) and (
                    self.root_user is not None):
                root_user_ = None
                for ri in self.root_user:
                    print(f"{datetime.now()}\tgroup root method")
                    if ri in sc_truelabels_subi:
                        root_user_ = ri
                        print(f"{datetime.now()}\tfor component {comp_i}, the root is {root_user_} and ri {ri}")
                if root_user_ is None:
                    root_user_ = sc_truelabels_subi[0]
                    print(f"{datetime.now()}\tsetting a dummy root")


                graph_node_label, majority_truth_labels, node_deg_list_i, root_i = self.find_root_group(a_i,
                                                                                                            sc_labels_subi,
                                                                                                            root_user_,
                                                                                                            sc_truelabels_subi,
                                                                                                            [], [])


            elif (self.dataset == '2M'):
                for ri in self.root_user:
                    if PARC_labels_leiden[ri] in cluster_labels_subi: root_user_ = ri
                graph_node_label, majority_truth_labels, node_deg_list_i, root_i = self.find_root_2Morgan(a_i,
                                                                                                          sc_labels_subi,
                                                                                                          root_user_,
                                                                                                          sc_truelabels_subi)

            # when root_user is given as a cell index
            else:
                root_user_ = None
                if (comp_i > len(self.root_user) - 1):
                    root_user_ = 0

                else:
                    for ri in self.root_user:
                        if PARC_labels_leiden[ri] in cluster_labels_subi:
                            root_user_ = ri
                            print(
                                f"{datetime.now()}\tThe root index, {ri} provided by the user belongs to cluster number {PARC_labels_leiden[ri]} and corresponds to cell type {self.true_label[ri]}")
                if root_user_ is None: root_user_ = 0
                graph_node_label, majority_truth_labels, node_deg_list_i, root_i = self.find_root(a_i,
                                                                                                  PARC_labels_leiden,
                                                                                                  root_user_,
                                                                                                  self.true_label)

            self.root.append(root_i)
            self.majority_truth_labels = majority_truth_labels  # single cell level "Majority Truth of that Cluster + Clusterlabel"

            for item in node_deg_list_i:
                node_deg_list.append(item)

            new_root_index_found = False
            for ii, llabel in enumerate(cluster_labels_subi):
                if root_i == llabel:
                    new_root_index = ii
                    new_root_index_found = True

            if not new_root_index_found:
                print('cannot find the new root index')
                new_root_index = 0

            print(f"{datetime.now()}\tComputing lazy-teleporting expected hitting times")
            hitting_times, roundtrip_times = self.compute_hitting_time(a_i, new_root_index, self.x_lazy,
                                                                       self.alpha_teleport)
            # rescale hitting times
            very_high = np.mean(hitting_times) + 1.5 * np.std(hitting_times)
            without_very_high_pt = [iii for iii in hitting_times if iii < very_high]
            new_very_high = np.mean(without_very_high_pt) + np.std(without_very_high_pt)
            # print('very high, and new very high', very_high, new_very_high)
            new_hitting_times = [x if x < very_high else very_high for x in hitting_times]
            hitting_times = np.asarray(new_hitting_times)
            scaling_fac = 10 / max(hitting_times)
            hitting_times = hitting_times * scaling_fac
            s_ai, t_ai = a_i.nonzero()
            edgelist_ai = list(zip(s_ai, t_ai))
            edgeweights_ai = a_i.data
            # print('edgelist ai', edgelist_ai)
            # print('edgeweight ai', edgeweights_ai)
            # bias the edges based on pseudotime from "undirected" pruned graph
            biased_edgeweights_ai = get_biased_weights(edgelist_ai, edgeweights_ai, hitting_times)

            # biased_sparse = csr_matrix((biased_edgeweights, (row, col)))
            adjacency_matrix_ai = np.zeros((a_i.shape[0], a_i.shape[0]))

            for i, (start, end) in enumerate(edgelist_ai):
                adjacency_matrix_ai[start, end] = biased_edgeweights_ai[i]

            markov_hitting_times_ai = self._simulate_markov(adjacency_matrix_ai,
                                                           new_root_index)  # +adjacency_matrix.T))
            '''
            df_sc = pd.DataFrame()
            df_sc['parc'] = self.labels
            df_sc['true_time'] = self.time_series_labels
            df_ = df_sc.groupby(['parc']).mean()
            df_['via_pt'] = markov_hitting_times_ai
            df_['via_pt'] = df_['via_pt'].fillna(0)

            correlation = df_['via_pt'].corr(df_['true_time'])
            print(f'correlation cluster level via original mcmc pt before scaling, {correlation}')
            '''

            #print('try rw2 hitting times setup')
            memory_pt = 2  # 2
            rw2_hittingtimes = _compute_rw2_hittingtimes(A=adjacency_matrix_ai, root=new_root_index, memory=memory_pt,
                                                        x_lazy=self.x_lazy, alpha_teleport=self.alpha_teleport)

            '''
            df_sc = pd.DataFrame()
            df_sc['parc'] = self.labels
            df_sc['true_time'] = self.time_series_labels
            df_ = df_sc.groupby(['parc']).mean()
            df_['via_pt'] = rw2_hittingtimes
            df_['via_pt'] = df_['via_pt'].fillna(0)


            correlation = df_['via_pt'].corr(df_['true_time'])
            print(f'correlation via rw2 pt at memory q={memory} before scaling, {correlation}')
            '''
            print(f'memory for rw2 hittings times  {memory_pt}. Using rw2 based pt')
            markov_hitting_times_ai = rw2_hittingtimes

            # print('no rw2')
            # print('skip scaling of pt')
            #print('do scaling of pt')

            very_high = np.mean(markov_hitting_times_ai) + 1.5 * np.std(markov_hitting_times_ai)  # 1.5
            very_high = min(very_high, max(markov_hitting_times_ai))
            # without_very_high_pt = [iii for iii in markov_hitting_times_ai if iii < very_high]
            # new_very_high = min(np.mean(without_very_high_pt) + np.std(without_very_high_pt), very_high)

            new_markov_hitting_times_ai = [x if x < very_high else very_high for x in markov_hitting_times_ai]

            markov_hitting_times_ai = np.asarray(new_markov_hitting_times_ai)
            scaling_fac = 1 / max(markov_hitting_times_ai)
            markov_hitting_times_ai = markov_hitting_times_ai * scaling_fac

            # for eee, ttt in enumerate(markov_hitting_times_ai):print('cluster ', eee, ' had markov time', ttt)

            # print('markov hitting times', [(i, j) for i, j in enumerate(markov_hitting_times_ai)])
            # print('hitting times', [(i, j) for i, j in enumerate(hitting_times)])
            # markov_hitting_times_ai = (markov_hitting_times_ai)  # + hitting_times)*.5 #consensus
            adjacency_matrix_csr_ai = sparse.csr_matrix(adjacency_matrix_ai)
            # A_velo = velocity_transition(A=adjacency_matrix_ai, V=velocity_mean, G=gene_mean) #velocity directed transition matrix
            # print('A_velo', A_velo.shape)
            # print(A_velo)
            (sources, targets) = adjacency_matrix_csr_ai.nonzero()
            edgelist_ai = list(zip(sources, targets))
            weights_ai = adjacency_matrix_csr_ai.data
            bias_weights_2_ai = get_biased_weights(edgelist_ai, weights_ai, markov_hitting_times_ai, round=2)
            adjacency_matrix2_ai = np.zeros((adjacency_matrix_ai.shape[0], adjacency_matrix_ai.shape[
                0]))  # this will be the cluster transition matrix forward biased by pseudotimes
            for i, (start, end) in enumerate(edgelist_ai):
                adjacency_matrix2_ai[start, end] = bias_weights_2_ai[i]
            # print('adjacency_matrix2_ai col sum', np.sum(adjacency_matrix2_ai, axis=1))
            # adjacency matrix used for branch prob and terminal states will be the weighted average of the pt_trans_matrix and the velo_trans_matrix
            # note that although A_velo and adjaccency_matrix2_ai do not have rows that sum to one due to the biasing steps, the transition matrix used for simulating branch probabilities will be normalized such that each row_sum = 1. This is done in the simulate_branch_prob function
            # however the biased weights without row normalization is used for visualization
            if self.A_velo is not None: adjacency_matrix2_ai = (
                                                                           1 - self.velo_weight) * adjacency_matrix2_ai + self.velo_weight * A_velo
            user_defined_terminal_clus_exists = False
            if (len(self.user_defined_terminal_group) > 0) | (len(self.user_defined_terminal_cell) > 0):
                user_defined_terminal_clus_exists = True

            if (self.super_terminal_cells == False) & (
                    user_defined_terminal_clus_exists == False):  # when is_coarse = True, there is no list of terminal clusters/cells that are passed into VIA based on a previous iteration.
                # print('new_root_index', new_root_index, ' before get terminal')
                terminal_clus_ai = self._get_terminal_clusters(adjacency_matrix2_ai, markov_hitting_times_ai,
                                                              new_root_index)
                temp_terminal_clus_ai = []
                threshold_ = np.percentile(np.asarray(markov_hitting_times_ai), self.pseudotime_threshold_TS)
                #print('!!! Remove the [0:2] just using to speed up testing')
                for i in terminal_clus_ai:#[0:2]:

                    if markov_hitting_times_ai[i] > threshold_:
                        terminal_clus.append(cluster_labels_subi[i])
                        temp_terminal_clus_ai.append(i)
                terminal_clus_ai = temp_terminal_clus_ai
            elif (self.super_terminal_cells == False) & (user_defined_terminal_clus_exists == True):
                terminal_clus_ai = []
                terminal_clus = []
                threshold_ = np.percentile(np.asarray(markov_hitting_times_ai), self.pseudotime_threshold_TS)
                terminal_clus_temp = self._get_terminal_clusters_user_defined_(self.user_defined_terminal_cell,
                                                                              self.user_defined_terminal_group)
                for ti in terminal_clus_temp:
                    if (ti in loc_compi):
                        terminal_clus_ai.append(ti)
                        terminal_clus.append(cluster_labels_subi[ti])

            print(
                f"{datetime.now()}\tTerminal clusters corresponding to unique lineages in this component are {terminal_clus_ai} ")

            if self.memory == 0:
                use_rw2_prob = False
            else:
                use_rw2_prob = True
            if not use_rw2_prob:
                print(f'Via 1.0 lineage prob')
                for target_terminal in terminal_clus_ai:

                    prob_ai = self.simulate_branch_probability(target_terminal,
                                                               adjacency_matrix2_ai,
                                                               new_root_index,
                                                               num_sim=int(self.num_mcmc_simulations / 2))
                    print(f'terminal state {target_terminal} has probability {prob_ai}')
                    df_graph['terminal_clus' + str(cluster_labels_subi[target_terminal])] = 0.0000000

                    pd_columnnames_terminal.append('terminal_clus' + str(cluster_labels_subi[target_terminal]))

                    for k, prob_ii in enumerate(prob_ai):
                        df_graph.at[cluster_labels_subi[k], 'terminal_clus' + str(
                            cluster_labels_subi[target_terminal])] = prob_ii
            else:
                memory = self.memory
                p_memory = self.p_memory

                prob_lin_rw2 = _compute_rw2_lineageprobability(A=adjacency_matrix2_ai, memory=memory,
                                                              root=new_root_index,
                                                              terminal_states=terminal_clus_ai, x_lazy=self.x_lazy,
                                                              alpha_teleport=self.alpha_teleport)
                for enum_ts, target_terminal in enumerate(terminal_clus_ai):
                    df_graph['terminal_clus' + str(cluster_labels_subi[target_terminal])] = 0.0000000

                    pd_columnnames_terminal.append('terminal_clus' + str(cluster_labels_subi[target_terminal]))

                    for k, prob_ii in enumerate(prob_lin_rw2[:, enum_ts]):
                        df_graph.at[cluster_labels_subi[k], 'terminal_clus' + str(
                            cluster_labels_subi[target_terminal])] = prob_ii
            bp_array = df_graph[pd_columnnames_terminal].values
            bp_array[np.isnan(bp_array)] = 1e-8

            bp_array = bp_array / bp_array.sum(axis=1)[:, None]  # row normalization at the cluster level
            bp_array[np.isnan(bp_array)] = 1e-8

            for ei, ii in enumerate(loc_compi):
                df_graph.at[ii, 'pt'] = hitting_times[ei]
                df_graph.at[ii, 'graph_node_label'] = graph_node_label[ei]
                df_graph.at[ii, 'majority_truth'] = graph_node_label[ei]
                df_graph.at[ii, 'markov_pt'] = markov_hitting_times_ai[ei]

            locallytrimmed_g.vs["label"] = df_graph['graph_node_label'].values
            hitting_times = df_graph['pt'].values

        self.cluster_adjacency = adjacency_matrix2_ai
        self.hitting_times = hitting_times  # not markov chain simulated
        # print('hitting times', hitting_times[0:10])
        self.markov_hitting_times = df_graph['markov_pt'].values  # hitting_times on the forward biased#
        # print('markov hitting times,', self.markov_hitting_times[0:10])
        self.terminal_clusters = terminal_clus

        dict_ts_mode = {}
        for i in terminal_clus:
            loc__ = np.where(np.asarray(self.labels) == i)[0]
            i_mode = func_mode(list(np.asarray(self.true_label)[loc__]))
            dict_ts_mode[i] = i_mode
        print(
            f"{datetime.now()}\tThere are ({len(terminal_clus)}) terminal clusters corresponding to unique lineages {dict_ts_mode}")  # {self.terminal_clusters} ")
        self.node_degree_list = node_deg_list
        print(f"{datetime.now()}\tBegin projection of pseudotime and lineage likelihood")
        self.cluster_bp = bp_array  # there is row normalization at the cluster level so each row sums to 1

        df_graph['markov_pt'] = df_graph['markov_pt'].fillna(10)  # 0
        # nan_count = df_graph['markov_pt'].isna().sum()
        # print('df graph markov pt nan count', nan_count)
        self.single_cell_bp, self.single_cell_pt_markov = self._project_branch_probability_sc(bp_array, df_graph[
            'markov_pt'].values)

        self.single_cell_pt_markov = [item for item in self.single_cell_pt_markov if not (math.isnan(item)) == True]
        if self.time_series_labels is not None:
            df_ = pd.DataFrame()
            df_['via_pt'] = self.single_cell_pt_markov
            df_['true_time'] = self.time_series_labels
            df_['via_pt'] = df_['via_pt'].fillna(0)

            correlation = df_['via_pt'].corr(df_['true_time'])
            print(f'{datetime.now()}\tStart reading data')
            print(
                f'{datetime.now()}\tCorrelation of Via pseudotime with developmental stage {round(correlation * 100, 2)} %')

        # the single_cell_bp are not re-rownormalized. In fact we scale each column (lineage) by the max value in the lineage (column) to prevent rarer/smaller lineages from being under-represented.
        # to get a row-normalized results of the single-cell branching probabilities, we offer single_cell_bp_rownormed as an attribute so that the probabilities of each cell sum to 1.

        single_cell_bp_rownormed = np.nan_to_num(self.single_cell_bp, nan=0.0, posinf=0.0, neginf=0.0)
        # row normalize
        row_sums = single_cell_bp_rownormed.sum(axis=1)
        single_cell_bp_rownormed = single_cell_bp_rownormed / row_sums[:, np.newaxis]
        # print(f'{datetime.now()}\tCheck cell 0 of sc pb rownormed {single_cell_bp_rownormed[0, :]},{single_cell_bp_rownormed.sum(axis=1)}')
        self.single_cell_bp_rownormed = single_cell_bp_rownormed
        self.dict_terminal_super_sub_pairs = dict_terminal_super_sub_pairs
        hitting_times = self.markov_hitting_times  # cluster level

        ### CONSTRUCT GRAPH USED FOR VISUALIZATION by doing further pruning so we can easily visualize the TI

        # rebias the edge weights based on the recomputed final markov hitting time and do a final pruning used for visualization (not MCMCs)
        bias_weights_2_all = get_biased_weights(self.edgelist, edgeweights_pruned_clustergraph,
                                                self.markov_hitting_times,
                                                round=2)  # edgeweights of initial globally and locally pruned graph (all components) need to be forward biased

        n_clus = len(set(self.labels))
        temp_csr = csr_matrix((bias_weights_2_all, tuple(zip(*self.edgelist))),
                              shape=(n_clus, n_clus))  # locally and globally pruned and used for adjacencyMatrix2ai

        if self.A_velo is None:
            final_transition_matrix_all_components = temp_csr.toarray()
        else:
            final_transition_matrix_all_components = (self.velo_weight) * self.A_velo + (1 - self.velo_weight) * (
                temp_csr.toarray())
            print(f"{datetime.now()}\tTransition matrix with weight of {self.velo_weight} on RNA velocity")
        # the final array used for making the visual edgelists will be weighted average of A_velo and pt_based_trans (temp_csr)
        final_transition_matrix_all_components = csr_matrix(final_transition_matrix_all_components)

        print(f'{datetime.now()}\tCluster graph layout based on forward biasing')
        # print("**** USING OLD LAYOUT BEFORE BIASING ****")

        # ensure no extreme values distort layout:
        '''
        print('NEW, KEEP OR NOT? ensure no extreme values distort layout:')
        bias_weights_2_all_hi = np.mean(bias_weights_2_all) + 3*np.std(bias_weights_2_all)
        bias_weights_2_all_lo = np.mean(bias_weights_2_all) - 3*np.std(bias_weights_2_all)
        bias_weights_2_subset = [x for x in bias_weights_2_all if ((x < bias_weights_2_all_hi) & (x> bias_weights_2_all_lo))]
        threshold_hi = np.percentile(bias_weights_2_subset,99)
        threshold_lo = np.percentile(bias_weights_2_subset,1)
        bias_weights_2_all = [x if x < threshold_hi else threshold_hi for x in bias_weights_2_all ]
        bias_weights_2_all = [x if x > threshold_lo else threshold_lo for x in bias_weights_2_all ]
        '''
        visual_g = ig.Graph(self.edgelist, edge_attrs={'weight': bias_weights_2_all}).simplify(
            combine_edges='sum')  # used to be commented out
        # random.seed(self.random_seed)


        if (self.graph_init_pos is not None) & (self.do_spatial_layout):
            if self.graph_init_pos is None: self.graph_init_pos = self.spatial_coords
            list_graph_init_pos = []
            print('using provided spatial coords to initialise viagraph layout. Only use this if all cells are from the same slice')
            for i in range(len(set(self.labels))):
                where = np.where(np.array(self.labels) == i)[0]
                #print('where are the clusters cells',where)
                x_mean = self.graph_init_pos[where, 0].mean()
                print('x_mean',x_mean)
                y_mean = self.graph_init_pos[where, 1].mean()
                list_graph_init_pos.append([x_mean, y_mean]) #list of lists
                print('list graph init pos')
            print(list_graph_init_pos)

            layout = visual_g.layout_fruchterman_reingold(weights='weight', seed=list_graph_init_pos)
            #layout = visual_g.layout_fruchterman_reingold(weights='weight')
        else:
            layout = visual_g.layout_fruchterman_reingold(weights='weight')
        #layout = visual_g.layout_fruchterman_reingold(weights='weight',  seed=np.matrix(layout))  # used to be commented out

        self.graph_node_pos = layout.coords
        self.layout = layout  # reassign

        if (self.embedding is None) & (self.do_compute_embedding == True):
            if self.embedding_type == 'via-atlas':
                print(f'{datetime.now()}\tRun via-atlas')

                from sklearn.preprocessing import normalize
                # row_stoch = normalize(self.csr_full_graph, norm='l1', axis=1)
                ##row_stoch = row_stoch ** 3
                # temp_pca = csr_matrix(self.data)
                # X_diffused_data = row_stoch * temp_pca  # matrix multiplication to diffuse the pcs
                # X_input=self.data
                for random_state in [self.random_seed]:  # 6 self.random_seed,6
                    for min_dist in [0.1]:  # 35,0.3,0.25,0.2,0.1]:
                        for rw2_comp in [self.ncomp]:
                            import random
                            str_date = random.randint(0, 1000)
                            #print('embedding pseudorand', str_date)
                            distance_metric = 'euclidean'  # 'cosine' #
                            #print(                                f'distance metric {distance_metric} and min_dist {min_dist} and randomstate {random_state} and pseudoid {str_date}')
                            # input = self.data
                            # '/home/user/Trajectory/Datasets/EB_Phate/RW2_sparse_matrix510Pp1p5_R2Wemd.csv'

                            # r2w_input = pd.read_csv( '/home/user/Trajectory/Datasets/Cao_ProtoVert/RW2/RW2_sparse_matrix5094_pc30_knn30_krev15_kseq15_rs0.csv')
                            # r2w_input = pd.read_csv('/home/user/Trajectory/Datasets/Cao_ProtoVert/RW2/RW2_P1_Q1_sparse_matrix5820_pc20_knn30_krev15_kseq15_rs0.csv')
                            # r2w_input = pd.read_csv('/home/user/Trajectory/Datasets/WagnerZebrafish/RW2/knn100_pc30_kseq50RW2_sparse_matrix5820.csv')

                            # r2w_input = pd.read_csv( '/home/user/Trajectory/Datasets/Zebrafish_Lange2023/RW2/5000hvg/pc100_knn10kseq5krev5RW2_noparc137noK0871_walklength20_numwalks20_dim128.csv')
                            # r2w_input = pd.read_csv('/home/user/Trajectory/Datasets/Zebrafish_Lange2023/RW2/5000hvg/pc100_knn10kseq5krev5RW2_672_walklength80_numwalks20_dim64.csv')

                            # r2w_input = pd.read_csv('/home/user/Trajectory/Datasets/Zebrafish_Lange2023/RW2/2000hvg/pc100_knn20kseq5krev5RW2_120q10walklength20798.csv')
                            # r2w_input = pd.read_csv( '/home/user/Trajectory/Datasets/EB_Phate/RW2/pc20_knn100kseq50krev50RW2_emd029.csv')

                            # r2w_input = pd.read_csv('/home/user/Trajectory/Datasets/MouseNeuron/RW2/pc30_knn20kseq0krev0RW2_887_P20Q20.csv')# not this
                            # r2w_input = pd.read_csv('/home/user/Trajectory/Datasets/MouseNeuron/RW2/pc30_knn50kseq0krev0RW2_gaussTrue_051_P1Q1.csv')  # not this
                            # r2w_input = pd.read_csv('/home/user/Trajectory/Datasets/MouseNeuron/RW2/pc30_knn20kseq0krev0RW2_887.csv') not this

                            # r2w_input = pd.read_csv( '/home/user/Trajectory/Datasets/EB_Phate/RW2/pc20_knn100kseq50krev50RW2_sparse_matrix029_P1_Qp001_numwalk20.csv')
                            # r2w_input = pd.read_csv('/home/user/Trajectory/Datasets/EB_Phate/RW2/pc20_knn100kseq50krev50RW2_sparse_matrix029_P1_Q100_numwalk20.csv')
                            # r2w_input = pd.read_csv('/home/user/Trajectory/Datasets/MouseNeuron/RW2/pc30_P1_Q1000_knn50kseq10krev10RW2_988.csv')
                            # r2w_input = pd.read_csv('/home/user/Trajectory/Datasets/MouseNeuron/RW2/pc30_P1_Q1_knn50kseq10krev10RW2_988.csv')
                            # r2w_input = pd.read_csv('/home/user/Trajectory/Datasets/2MOrgan/RW2/pc50_knn30kseq10krev0RW2_970_P1_Q10_rs42_tdiffstep2_gaussFalse_970.csv')
                            #r2w_input = pd.read_csv(                                '/home/user/Trajectory/Datasets/Pijuan_Gastrulation/RW2/pc30_knn30kseq15krev15RW2_sparse_matrix_122.csv')  # pc30_knn30kseq15krev15RW2_emd_122.csv')
                            # r2w_input = pd.read_csv('/home/user/Trajectory/Datasets/Pijuan_Gastrulation/RW2/pc30_P1_Qp001_knn30kseq15krev15RW2_sparse_matrix_122.csv')

                            #r2w_input = r2w_input.drop(['Unnamed: 0'], axis=1).values
                            #input = r2w_input[:, 0:rw2_comp]
                            # print('NOT using rw2 embedding CANNOT USE THIS CODE ON GITHUB YET AS IT USES LOCAL FILES')
                            input = self.data
                            #print('USING rw2 embedding CANNOT USE THIS CODE ON GITHUB YET AS IT USES LOCAL FILES')

                            '''
                            from sklearn.preprocessing import normalize
                            row_stoch = self.csr_full_graph
                            row_stoch.data = np.clip(row_stoch.data, np.percentile(row_stoch.data, 10),
                                                     np.percentile(row_stoch.data, 90))
                            row_stoch = normalize(row_stoch, norm='l1', axis=1)
                            row_stoch = row_stoch ** 2  # level of diffusion

                            temp = csr_matrix(input)

                            input = row_stoch * temp  # matrix multiplication
                            '''
                            do_initVia = True  # when calling via_atlas_emb() from within the class, it automatically initializes using the via cluster graph
                            n_epochs = 100  # 100 usually

                            if do_initVia:
                                self.embedding = via_atlas_emb(X_input=self.data, graph=self.csr_full_graph,
                                                               n_epochs=n_epochs,
                                                               spread=1,
                                                               distance_metric=distance_metric, min_dist=min_dist,
                                                               saveto='',
                                                               random_state=random_state, init_pos='via',
                                                               cluster_membership=self.labels, layout=layout.coords)
                            else:
                                self.embedding = via_atlas_emb(X_input=self.data, graph=self.csr_full_graph,
                                                               n_epochs=n_epochs,
                                                               spread=1,
                                                               distance_metric=distance_metric, min_dist=min_dist,
                                                               saveto='',
                                                               random_state=random_state)

                            print(f'{datetime.now()}\tCompleted via-atlas embedding')
                            title_umap = 'via-atlas rw2 knn/pc/knnseq:' + str(self.knn) + '/' + str(
                                self.ncomp) + '/' + str(self.knn_sequential) + '_knnreverse' + str(
                                self.knn_sequential_reverse) + 'tdiff' + str(
                                self.t_diff_step) + 'mindist' + str(
                                min_dist) + 'rs' + str(self.random_seed)
                            if self.time_series_labels is not None:
                                color_labels = self.time_series_labels
                                f1, ax = plot_scatter(embedding=self.embedding, labels=color_labels, cmap='plasma', s=5,
                                                      alpha=0.5, edgecolors='None',
                                                      title=title_umap, text_labels=False)
                                f1.set_size_inches(10, 10)

                                # str_date = str(str(datetime.now())[-3:])
                                # save_str = '/home/user/Trajectory/Datasets/WagnerZebrafish/viaumap_RW2_5820_k' + str(                                    self.knn) + 'TimeseriesAug' + str(self.time_series) + '_kseq' + str(                                   self.knn_sequential) + '_knnreverse' + str(self.knn_sequential_reverse) + 'npc' + str(self.ncomp) + 'tdiff' + str(                                    self.t_diff_step) + 'mindist' + str(min_dist) + '_rsUmap' + str(                                    random_state) + '_rsVIA' + str(self.random_seed) + 'doGauss' + str(                                    self.do_gaussian_kernel_edgeweights) + 'viaInit' + str(do_initVia) + str_date

                                # save_str = '/home/user/Trajectory/Datasets/WagnerZebrafish/viaumap_PCA_k' + str(self.knn) + 'TimeseriesAug'+str(self.time_series)+ '_kseq' + str(self.knn_sequential) + '_knnreverse' + str(  self.knn_sequential_reverse) + 'npc' + str(self.ncomp) + 'tdiff' + str(                                self.t_diff_step) + 'mindist' + str(min_dist) + '_rsUmap' + str(                                random_state) + '_rsVIA' + str(self.random_seed) + 'doGauss' + str(                                self.do_gaussian_kernel_edgeweights) + 'viaInit'+str(do_initVia)+ str_date
                                # save_str = '/home/user/Trajectory/Datasets/MEF_Schiebinger/viaumap_k' + str(                                self.knn) + 'kseq' + str(self.knn_sequential) + 'npc' + str(self.ncomp) + 'tdiff' + str(                                self.t_diff_step) + 'mindist' + str(min_dist) + 'rs' + str(                                self.random_seed) + 'doGauss' + str(                                self.do_gaussian_kernel_edgeweights) + str_date + 'stage'

                                # save_str = '/home/user/Trajectory/Datasets/EB_Phate/viaumap_R2W_029_P1_Qp001_k' + str( self.knn) +'TimeseriesAug'+str(self.time_series)+ 'kseq' + str(self.knn_sequential) +'_knnreverse'+str(self.knn_sequential_reverse) + 'npc' + str(self.ncomp) + 'tdiff' + str(                                    self.t_diff_step) + '_mainRS'+str(self.random_seed)+'mindist' + str(min_dist) + 'rsUmap' + str(random_state) + 'doGauss' + str( self.do_gaussian_kernel_edgeweights) +'_ViaInitLayout'+str(do_initVia)+'_nEpochs'+str(n_epochs)+'_'+ str_date
                                # save_str = '/home/user/Trajectory/Datasets/EB_Phate/viaumap_PCA_k' + str(                                    self.knn) + 'TimeseriesAug' + str(self.time_series) + 'kseq' + str(                                    self.knn_sequential) + '_knnreverse' + str(                                    self.knn_sequential_reverse) + 'npc' + str(self.ncomp) + 'tdiff' + str(                                    self.t_diff_step) + '_mainRS' + str(self.random_seed) + 'mindist' + str(                                    min_dist) + 'rsUmap' + str(random_state) + 'doGauss' + str(                                    self.do_gaussian_kernel_edgeweights) + '_ViaInitLayout' + str(                                    do_initVia) + '_nEpochs' + str(n_epochs) + '_' + str_date
                                # save_str = '/home/user/Trajectory/Datasets/Cao_ProtoVert/viaumap_RW2_P1_Q1_5820_k' + str( self.knn) + 'TimeseriesAug'+str(self.time_series)+ '_kseq' + str(self.knn_sequential) + '_knnreverse' + str(                                self.knn_sequential_reverse) + 'npc' + str(self.ncomp) + 'tdiff' + str(                                self.t_diff_step) + 'mindist' + str(min_dist) + '_rsUmap' + str(                                random_state) +'_rsVIA'+str(self.random_seed)+ 'doGauss' + str(                                self.do_gaussian_kernel_edgeweights) +'viaInitLayout'+str(do_initVia)+'_nEpochs'+str(n_epochs)+'_'+ str_date
                                # save_str = '/home/user/Trajectory/Datasets/Cao_ProtoVert/viaumap_PCA_k' + str(                                    self.knn) + 'TimeseriesAug' + str(self.time_series) + '_kseq' + str(                                    self.knn_sequential) + '_knnreverse' + str(                                    self.knn_sequential_reverse) + 'npc' + str(self.ncomp) + 'tdiff' + str(                                    self.t_diff_step) + 'mindist' + str(min_dist) + '_rsUmap' + str(                                    random_state) + '_rsVIA' + str(self.random_seed) + 'doGauss' + str(                                    self.do_gaussian_kernel_edgeweights) + 'viaInitLayout' + str(                                    do_initVia) + '_nEpochs' + str(n_epochs) + '_' + str_date
                                # save_str = '/home/user/Trajectory/Datasets/Pijuan_Gastrulation/viaumap_RW2_P1_Qp001_rw2comp'+str(rw2_comp)+'_122_k' + str(                                    self.knn) +'_TimeAug'+str(self.time_series)+ '_kseq' + str(self.knn_sequential) +'_knnreverse'+str(self.knn_sequential_reverse) + 'npc' + str(self.ncomp) + 'tdiff' + str(                                    self.t_diff_step) + 'mindist' + str(min_dist) + 'rs' + str(                                    random_state) + 'doGauss' + str(                                    self.do_gaussian_kernel_edgeweights) + str_date +'_viaInitTrue'
                                # save_str = '/home/user/Trajectory/Datasets/Pijuan_Gastrulation/viaumap_PCA_k' + str(self.knn) + '_TimeAug' + str(                                    self.time_series) + '_kseq' + str(self.knn_sequential) + '_knnreverse' + str(                                    self.knn_sequential_reverse) + 'npc' + str(self.ncomp) + 'tdiff' + str(                                    self.t_diff_step) + 'mindist' + str(min_dist) + 'rs' + str(                                    random_state) + 'doGauss' + str(                                    self.do_gaussian_kernel_edgeweights)+'_viaInit'+str(do_initVia)      +str(n_epochs)+'_'+ str_date
                                # save_str = '/home/user/Trajectory/Datasets/MouseNeuron/viaumap_RW2_P1_Q1000_rw2comp' + str(rw2_comp) + '_988_k' + str(self.knn) + 'TimeseriesAug'+str(self.time_series)+'_kseq' + str(self.knn_sequential) + '_knnreverse' + str( self.knn_sequential_reverse) + 'npc' + str(self.ncomp) + 'tdiff' + str(self.t_diff_step) + 'mindist' + str(min_dist) +'_rsMain'+str(self.random_seed)+ '_rs' + str(random_state) + 'doGauss' + str( self.do_gaussian_kernel_edgeweights) +'_viaInitLayout'+str(do_initVia)+'_nEpochs'+str(n_epochs)+'_'+ str_date
                                # save_str = '/home/user/Trajectory/Datasets/MouseNeuron/viaumap_PCA_k' + str(self.knn) + 'TimeseriesAug'+str(self.time_series)+'_kseq' + str(self.knn_sequential) + '_knnreverse' + str( self.knn_sequential_reverse) + 'npc' + str(self.ncomp) + 'tdiff' + str(self.t_diff_step) + 'mindist' + str(min_dist) +'_rsMain'+str(self.random_seed)+ '_rs' + str(random_state) + 'doGauss' + str( self.do_gaussian_kernel_edgeweights) +'_viaInitLayout'+str(do_initVia)+'_nEpochs'+str(n_epochs)+'_'+ str_date
                                # save_str = '/home/user/Trajectory/Datasets/2MOrgan/viaumap_RW2_P1_Q10_rw2comp' + str(                                    rw2_comp) + '_970_k' + str(self.knn) + 'kseq' + str(                                    self.knn_sequential) + '_knnreverse' + str(                                    self.knn_sequential_reverse) + 'npc' + str(self.ncomp) + 'tdiff' + str(                                    self.t_diff_step) + 'mindist' + str(min_dist) + 'rs' + str(                                    random_state) + 'doGauss' + str(self.do_gaussian_kernel_edgeweights) +'viaInitTrue'+ str_date
                                # save_str = '/home/user/Trajectory/Datasets/Zebrafish_Lange2023/via_umaps_all/viaumap_2000hvg_k' + str(                                    self.knn) + 'TimeseriesAug' + str(self.time_series) + '_kseq' + str(                                    self.knn_sequential) + '_knnreverse' + str(                                    self.knn_sequential_reverse) + 'npc' + str(self.ncomp) + 'tdiff' + str(                                    self.t_diff_step) + 'mindist' + str(min_dist) + '_rsUmap' + str(                                    random_state) + '_rsVIA' + str(self.random_seed) + 'doGauss' + str(                                    self.do_gaussian_kernel_edgeweights) +'_distMet'+distance_metric[0:3]+ 'viaInit' + str(do_initVia) + str_date
                                # save_str = '/home/user/Trajectory/Datasets/Zebrafish_Lange2023/RW2/embeddings/rw2_npz871_2000hvg_k' + str(                                    self.knn) + 'TimeseriesAug' + str(self.time_series) + '_kseq' + str(                                    self.knn_sequential) + '_knnreverse' + str(                                    self.knn_sequential_reverse) + '_cluster_gp'+str(self.cluster_graph_pruning)+'_npc' + str(self.ncomp) + 'tdiff' + str(                                    self.t_diff_step) + 'mindist' + str(min_dist) + '_rsUmap' + str(                                    random_state) + '_rsVIA' + str(self.random_seed) + 'doGauss' + str(                                    self.do_gaussian_kernel_edgeweights) + '_distMet' + distance_metric[                                                                                        0:3] + 'viaInit' + str(                                    do_initVia) + str_date                                #df_umap.to_csv('/home/user/Trajectory/Datasets/MouseNeuron/2000hvg_viaumap_k'+str(self.knn)+'kseq'+str(self.knn_sequential)+'nps'+str(self.ncomp)+'tdiff'+str(self.t_diff_step)+'mindist'+str(min_dist)+'rs'+str(self.random_seed)+'stage'+str_date+".csv")
                                # savefig_ = '/home/user/Trajectory/Datasets/MouseNeuron/2000hvg_viaumap_k'+str(self.knn)+'kseq'+str(self.knn_sequential)+'nps'+str(self.ncomp)+'tdiff'+str(self.t_diff_step)+'mindist'+str(min_dist)+'rs'+str(self.random_seed)+str_date+'stage.png'
                                # save_str = '/home/user/Trajectory/Datasets/Qiu_Mouse_GastrulaPup/via-umap/5000hvg_viaumap_k'+str(self.knn)+'kseq'+str(self.knn_sequential)+'npcs'+str(self.ncomp)+'tdiff'+str(self.t_diff_step)+'doGaussianEdge'+str(self.do_gaussian_kernel_edgeweights)+'mindist'+str(min_dist)+'rs'+str(self.random_seed)+'_distmet'+distance_metric+str(str_date)
                                save_str = '/home/user/Trajectory/Datasets/Packer_Elegans2019/via-umap/5000hvg_viaumap_k' + str(
                                    self.knn) + 'kseq' + str(self.knn_sequential) + 'npcs' + str(
                                    self.ncomp) + 'tdiff' + str(self.t_diff_step) + 'doGaussianEdge' + str(
                                    self.do_gaussian_kernel_edgeweights) + 'mindist' + str(min_dist) + 'rs' + str(
                                    self.random_seed) + '_distmet' + distance_metric + str(str_date)
                                df_umap = pd.DataFrame(self.embedding)
                                # df_umap.to_csv(save_str + '.csv')
                                # f1.savefig(save_str + 'stage.png', facecolor='white', transparent=False)

                            # savefig_ = '/home/user/Trajectory/Datasets/MouseNeuron/2000hvg_viaumap_k'+str(self.knn)+'kseq'+str(self.knn_sequential)+'nps'+str(self.ncomp)+'tdiff'+str(self.t_diff_step)+'mindist'+str(min_dist)+'rs'+str(self.random_seed)+str_date+'celltype.png'

                            celltype_label = [i.split("_")[0] for i in self.true_label]  # self.true_label
                            f2, ax = plot_scatter(embedding=self.embedding, labels=celltype_label, title=title_umap,
                                                  alpha=0.5, s=5, color_dict=self.color_dict)
                            f2.set_size_inches(10, 10)
                            # save_str = '/home/user/Trajectory/Datasets/Rashmi2023/emt/'+self.embedding_type+'_knn'+str(self.knn)
                            # f2.savefig(save_str + 'celltype.png', facecolor='white', transparent=False)
                            plt.show()

            elif self.embedding_type == 'via-mds':
                str_date = str(str(datetime.now())[-3:])
                print(f'{datetime.now()}\tRun via-mds')
                # n_milestones = min(self.nsamples, max(10000, int(0.1*self.nsamples)))

                for rs_i in [self.random_seed]:
                    for k_project_milestones in [3]:  # ,5,10]:
                        for k_mds_i in [25]:  # [5,10,15,25]:
                            for diffusion_i in [2]:  # [2,5,10]:
                                t_difference = self.t_diff_step
                                k_seq_i = 2
                                n_milestones_mds = min(3000, self.data.shape[0])

                                self.embedding = via_mds(X_pca=self.data, n_milestones=n_milestones_mds, k=k_mds_i,
                                                         knn_seq=k_seq_i, k_project_milestones=k_project_milestones,
                                                         # neighbors_distances=self.full_neighbor_array,

                                                         diffusion_op=diffusion_i, random_seed=rs_i,
                                                         viagraph_full=self.csr_full_graph, t_difference=t_difference,
                                                         time_series_labels=self.time_series_labels, saveto='',
                                                         double_diffusion=False)

                                str_date = str(str(datetime.now())[-3:])
                                # save_str = '/home/user/Trajectory/Datasets/Pijuan_Gastrulation/mds/viamds_singlediffusion_pcs' + str(                                    shape_data) + '_k' + str(self.knn) + '_milestones' + str(                                   n_milestones_mds) + '_kprojectmilestones' + str(                                    k_project_milestones) + 't_step' + str(t_difference) + '_knnmds' + str(                                    k_mds_i) + '_kseqmds' + str(k_seq_i) + '_kseq' + str(                                    self.knn_sequential) + '_nps' + str(self.ncomp) + '_tdiff' + str(self.t_diff_step) + '_randseed' + str(self.random_seed) + '_diffusionop' + str(                                    diffusion_i) + '_RsMds' + str(rs_i) + '_' + str_date
                                # save_str = '/home/user/Trajectory/Datasets/Zebrafish_Lange2023/via_mds/viamds_singlediffusion_pcs' + str(                                    shape_data) + '_k' + str(self.knn) + '_milestones' + str(                                    n_milestones_mds) + '_kprojectmilestones' + str(                                    k_project_milestones) + 't_step' + str(t_difference) + '_knnmds' + str(                                    k_mds_i) + '_kseqmds' + str(k_seq_i) + '_kseq' + str(                                    self.knn_sequential) + '_nps' + str(self.ncomp) + '_tdiff' + str(                                    self.t_diff_step) + '_randseed' + str(self.random_seed) + '_diffusionop' + str(                                    diffusion_i) + '_RsMds' + str(rs_i) + '_' + str_date
                                # save_str = '/home/user/Trajectory/Datasets/Qiu_Mouse_GastrulaPup/mds/viamds_singlediffusion_pcs' + str(                                    shape_data) + '_k' + str(self.knn) + '_milestones' + str(                                    n_milestones_mds) + '_kprojectmilestones' + str(                                    k_project_milestones) + 't_step' + str(t_difference) + '_knnmds' + str(                                    k_mds_i) + '_kseqmds' + str(k_seq_i) + '_kseq' + str(                                    self.knn_sequential) + '_nps' + str(self.ncomp) + '_tdiff' + str(                                    self.t_diff_step) + '_randseed' + str(self.random_seed) + '_diffusionop' + str(                        diffusion_i) + '_RsMds' + str(rs_i) + '_' + str_date
                                # save_str = '/home/user/Trajectory/Datasets/WagnerZebrafish/2000hvg/mds/singlediffusion_viamds/2000hvg_viamds_singlediffusion_pcs'+str(shape_data)+'_k' + str(self.knn) + '_milestones'+str(n_milestones_mds)+'_kprojectmilestones'+str(k_project_milestones)+'t_step'+str(t_difference)+'_knnmds' + str(k_mds_i) +'_kseqmds' + str(k_seq_i) +'_kseq'+str(self.knn_sequential)+'_nps' + str(self.ncomp) + '_tdiff' + str(self.t_diff_step)+'_randseed'+str(self.random_seed)+ '_diffusionop'+str(diffusion_i)+'_RsMds'+str(rs_i)+'_'+str_date
                                # save_str = '/home/user/Trajectory/Datasets/Cao_ProtoVert/mds_theseare1000hvg/singlediffusion_viamds/1000hvg_viamds_singlediffusion_doExp'+str(self.do_gaussian_kernel_edgeweights)+'_k' + str(self.knn) + '_milestones'+str(n_milestones_mds)+'_kprojectmilestones'+str(k_project_milestones)+'t_stepmds'+str(t_difference)+'_knnmds' + str(k_mds_i) +'_kseqmds' + str(k_seq_i) +'_kseq'+str(self.knn_sequential)+'_nps' + str(self.ncomp) + '_tdiff' + str(self.t_diff_step)+'_randseed'+str(self.random_seed)+ '_diffusionop'+str(diffusion_i)+'_rsMds'+str(rs_i)+'_'+str_date
                                # save_str = '/home/user/Trajectory/Datasets/EB_Phate/viamds_R2W_029_P1_Qp001_singlediffusion_prescaled_doExp' + str(                                    self.do_gaussian_kernel_edgeweights) + '_k' + str(self.knn) + '_milestones' + str(                                    n_milestones_mds) + '_kprojectmilestones' + str(                                    k_project_milestones) + 't_stepmds' + str(t_difference) + '_knnmds' + str(                                    k_mds_i) + '_kseqmds' + str(k_seq_i) + '_kseq' + str(                                    self.knn_sequential) + '_npc' + str(self.ncomp) + '_tdiff' + str(                                    self.t_diff_step) + '_randseed' + str(self.random_seed) + '_diffusionop' + str(                                    diffusion_i) + '_rsMds' + str(rs_i) + '_' + str_date                                # save_str = '/home/user/Trajectory/Datasets/MEF_Schiebinger/viamds_singlediffusion_prescaled_doExp'+str(self.do_gaussian_kernel_edgeweights)+'_k' + str(self.knn) + '_milestones'+str(n_milestones_mds)+'_kprojectmilestones'+str(k_project_milestones)+'t_stepmds'+str(t_difference)+'_knnmds' + str(k_mds_i) +'_kseqmds' + str(k_seq_i) +'_kseq'+str(self.knn_sequential)+'_npc' + str(self.ncomp) + '_tdiff' + str(self.t_diff_step)+'_randseed'+str(self.random_seed)+ '_diffusionop'+str(diffusion_i)+'_rsMds'+str(rs_i)+'_'+str_date
                                print(f'{datetime.now()}\tCompleted via-mds')

                                # save_str = '/home/user/Trajectory/Datasets/Qiu_Mouse_GastrulaPup/mds/' + self.embedding_type + '_knn' + str(                                    self.knn) + '_projectmilestones' + str(k_project_milestones) + '_kmds' + str(                                    k_mds_i) + '_diffusion' + str(diffusion_i)+'_strdate'+str(str_date)
                                save_str = '/home/user/Trajectory/Datasets/Packer_Elegans2019/via-mds/' + self.embedding_type + '_knn' + str(
                                    self.knn) + '_projectmilestones' + str(k_project_milestones) + '_kmds' + str(
                                    k_mds_i) + '_diffusion' + str(diffusion_i) + '_strdate' + str(str_date)
                                df_mds = pd.DataFrame(self.embedding)
                                df_mds.to_csv(save_str + ".csv")
                                celltype_label = [i.split("_")[0] for i in self.true_label]  # self.true_label
                                f1, ax = plot_scatter(embedding=self.embedding, labels=celltype_label, title='via-mds',
                                                      alpha=0.5, cmap='rainbow', text_labels=True)

                                f1.set_size_inches(10, 10)
                                #f1.savefig(save_str + '_celltype.png', facecolor='white', transparent=False)

                                if self.time_series_labels is not None:
                                    f2, ax2 = plot_scatter(embedding=self.embedding, labels=self.time_series_labels,
                                                           title='via-mds', alpha=0.5, cmap='plasma')

                                    f2.set_size_inches(10, 10)
                                    #f2.savefig(save_str + '_stage.png', facecolor='white', transparent=False)



            elif self.embedding_type == 'via-force':
                print(f'{datetime.now()}\tRun via-force')
                # n_milestone = min(self.nsamples, max(5000, int(0.1*self.nsamples)))
                n_milestones = 3000
                self.embedding = via_forcelayout(X_pca=self.data, viagraph_full=self.csr_full_graph, k=10, knn_seq=10,
                                                 n_milestones=n_milestones)
                if self.time_series_labels is None:
                    color_labels = self.true_label
                else:
                    color_labels = self.time_series_labels
                if (isinstance(color_labels[0], str)) == True:
                    categorical = True
                else:
                    categorical = False

                plot_scatter(embedding=self.embedding, labels=color_labels, title='via-force', categorical=categorical)
                plot_scatter(embedding=self.embedding, labels=self.true_label, title='via-mds', categorical=True)
                plt.show()
            else:
                print(
                    f'{datetime.now()}\tNo embedding will be computed: if you wish to compute a via-embedding specify one of via-force, via-mds or via-umap')

        if self.edgebundle_pruning_twice == False:
            # print('creating bundle with single round of global pruning at a level of', self.edgebundle_pruning)
            print(f"{datetime.now()}\tStarting make edgebundle viagraph...")
            self.hammerbundle_cluster, layout = make_edgebundle_viagraph(layout, g_layout, decay=self.viagraph_decay)

        # simplifying structure of edges used on the visual layout (Used when doing edgebundling_pruning_twice AND differention flows)
        edgeweights_maxout_2, edgelist_maxout_2, comp_labels_2 = pruning_clustergraph(
            final_transition_matrix_all_components,
            global_pruning_std=self.visual_cluster_graph_pruning,
            max_outgoing=self.max_visual_outgoing_edges,
            preserve_disconnected=self.preserve_disconnected)

        if self.edgebundle_pruning_twice == True:
            print(
                f"{datetime.now()}\tAdditional Visual cluster graph pruning for edge bundling at level: {self.visual_cluster_graph_pruning}")
            layout_g = ig.Graph(self.edgelist, edge_attrs={'weight': edgeweights_maxout_2}).simplify(
                combine_edges='sum')
            layout_g_csr = get_sparse_from_igraph(layout_g, weight_attr='weight')
            weights_for_layout = np.asarray(layout_g_csr.data)
            # clip weights to prevent distorted visual scale in layout
            weights_for_layout = np.clip(weights_for_layout, np.percentile(weights_for_layout, 10),
                                         np.percentile(weights_for_layout, 90))
            weights_for_layout = list(weights_for_layout)
            graph_for_layout = ig.Graph(list(zip(*layout_g_csr.nonzero())), edge_attrs={'weight': weights_for_layout})
            self.clustergraph_igraph_forlayout = graph_for_layout
            # edge bundle is based on the visually (double) pruned graph rather than the inital graph used for Velo and Pseudotime
            self.hammerbundle_cluster, layout = make_edgebundle_viagraph(layout=layout, graph=graph_for_layout, decay=self.viagraph_decay)
        '''
        else:
            print(f'redoing cluster graph layout based on forward biased edges')
            layout_g = ig.Graph(edgelist_maxout_2, edge_attrs={'weight': bias_weights_2_all}).simplify(
                combine_edges='sum')
            layout_g_csr = get_sparse_from_igraph(layout_g, weight_attr='weight')
            weights_for_layout = np.asarray(layout_g_csr.data)
            # clip weights to prevent distorted visual scale in layout
            weights_for_layout = np.clip(weights_for_layout, np.percentile(weights_for_layout, 10),
                                         np.percentile(weights_for_layout, 90))
            weights_for_layout = list(weights_for_layout)
            graph_for_layout = ig.Graph(list(zip(*layout_g_csr.nonzero())), edge_attrs={'weight': weights_for_layout})
            # edge bundle is based on the visually (double) pruned graph rather than the inital graph used for Velo and Pseudotime
            self.hammerbundle_cluster, layout = make_edgebundle_viagraph(layout, graph_for_layout)
        '''
        temp_csr = csr_matrix((np.array(edgeweights_maxout_2), tuple(zip(*edgelist_maxout_2))), shape=(n_clus, n_clus))
        temp_csr = temp_csr.transpose().todense() + temp_csr.todense()
        temp_csr = np.tril(temp_csr, -1)  # elements along the main diagonal and above are set to zero
        temp_csr = csr_matrix(temp_csr)
        edgeweights_maxout_2 = temp_csr.data
        scale_factor = max(edgeweights_maxout_2) - min(edgeweights_maxout_2)
        edgeweights_maxout_2 = [((wi + .1) * 2.5 / scale_factor) + 0.1 for wi in edgeweights_maxout_2]

        sources, targets = temp_csr.nonzero()
        edgelist_maxout_2 = list(zip(sources.tolist(), targets.tolist()))
        self.edgelist_maxout = edgelist_maxout_2
        self.edgeweights_maxout = edgeweights_maxout_2

        remove_outliers = hitting_times
        threshold = np.percentile(remove_outliers, 95)  # np.mean(remove_outliers) + 1* np.std(remove_outliers)
        th_hitting_times = [x if x < threshold else threshold for x in hitting_times]
        remove_outliers_low = hitting_times[hitting_times < (np.mean(hitting_times) - 0.3 * np.std(hitting_times))]
        threshold_low = 0 if remove_outliers_low.size else np.percentile(remove_outliers_low, 5)
        th_hitting_times = [x if x > threshold_low else threshold_low for x in th_hitting_times]

        scaled_hitting_times = (th_hitting_times - np.min(th_hitting_times))
        npmax = np.max(scaled_hitting_times) or 1
        scaled_hitting_times = scaled_hitting_times * (1000 / npmax)

        self.scaled_hitting_times = scaled_hitting_times  # cluster mcmc pt cluster level
        scaled_hitting_times = scaled_hitting_times.astype(int)
        pal = ig.drawing.colors.AdvancedGradientPalette(['yellow', 'green', 'blue'], n=1001)

        # making a new "augmented" single-cell graph based on the computed pseudotimes - potentially useful when there are no time-series labels for (optionally) guiding the graph structure
        use_pt_to_guide_graph = False
        if use_pt_to_guide_graph == True:
            print(f'{datetime.now()}\tMake pt-augmented knn')
            pt_augmented_adjacency_igraph, adjacency_augmented = self._make_pt_augmented_adjacency_igraph(
                neighbors=neighbors, distances=distances, k_reverse=10, knn=20)

            print(f'{datetime.now()}\tRun via-umap on pt-augmented knn')  # graph=csr_full_graph
            self.embedding = via_atlas_emb(X_input=self.data, graph=self.adjacency_pt_augmented, n_epochs=100, spread=1,
                                           distance_metric='euclidean', min_dist=0.3,
                                           saveto='/home/user/Trajectory/Datasets/HumanCD34/pt-aug/not_ptaug_umap_.csv')  # usually min_dist default =0.1, for cd34 0.8

        '''
        self.embedding = via_mds(X_pca=self.data, k=15, t_diffusion=2,
                                 n_milestones=min(self.nsamples, max(10000, int(0.1 * self.nsamples))),
                                 viagraph_full=adjacency_pt_augmented, time_series_labels=[int(i*10) for i in self.single_cell_pt_markov],
                                 saveto='')
        '''


        all_colors = []  # TODO clean up this paragraph/put into a separate function
        for i in scaled_hitting_times:
            all_colors.append(pal.get(int(i))[0:3])

        locallytrimmed_g.vs['hitting_times'] = scaled_hitting_times
        locallytrimmed_g.vs['color'] = [pal.get(i)[0:3] for i in scaled_hitting_times]

        self.group_color = [colors.to_hex(v) for v in locallytrimmed_g.vs['color']]  # based on ygb scale
        viridis_cmap = plt.get_cmap('viridis_r')

        self.group_color_cmap = [colors.to_hex(v) for v in
                                 viridis_cmap(scaled_hitting_times / 1000)]  # based on ygb scale

        self.graph_node_label = df_graph['graph_node_label'].values
        self.edgeweight = [e['weight'] * 1 for e in locallytrimmed_g.es]

        # print('To draw viagraph edges without bundling and plot piechart composition of cell types: use self.draw_piechart_graph_nobundle()')
        # self.draw_piechart_graph_nobundle()
        # print('drawing with bundle piechart... Use self.draw_piechart_graph() to reproduce this plot')

        initial_bandwidth = 0.05  # 0.05
        decay = 0.7  # 0.7
        n_milestones = 150  # f you min(self.nsamples, max(500, int(0.1 * self.nsamples)))

        # print(f'milestones for edgebundling {n_milestones}')
        global_visual_pruning = 0.15
        extra_title_text = 'jac_visual:' + str(global_visual_pruning) + ' bw:' + str(
            initial_bandwidth) + ' decay:' + str(decay)
        # sc_graph = self.csr_full_graph
        self.hammerbundle_milestone_dict = None
        if self.embedding is not None:
            print(
                f"{datetime.now()}\tStart making edgebundle milestone with {n_milestones} milestones...This can be recomputed with make_edgebundle_milestone()")
            self.hammerbundle_milestone_dict = make_edgebundle_milestone(embedding=self.embedding,
                                                                         sc_graph=self.ig_full_graph,
                                                                         n_milestones=n_milestones,
                                                                         global_visual_pruning=global_visual_pruning,
                                                                         initial_bandwidth=initial_bandwidth,
                                                                         decay=decay,
                                                                         weighted=True,
                                                                         sc_labels_numeric=self.time_series_labels,
                                                                         sc_pt=self.single_cell_pt_markov,
                                                                         terminal_cluster_list=self.terminal_clusters,
                                                                         single_cell_lineage_prob=self.single_cell_bp)

        self.labels = list(self.labels)


        for tsi in self.terminal_clusters:
            loc_i = np.where(np.asarray(self.labels) == tsi)[0]
            val_pt = [self.single_cell_pt_markov[i] for i in loc_i]

            th_pt = np.percentile(val_pt, 50)  # 50
            loc_i = [loc_i[i] for i in range(len(val_pt)) if val_pt[i] >= th_pt]
            temp = np.mean(self.data[loc_i], axis=0)
            labelsq, distances = self.knn_struct.knn_query(temp, k=1)

            tsi_list.append(labelsq[0][0])

        if self.embedding is not None:
            print('REMEMBER TO RE-INCLUDE the PLT.SHOW HERE - COMMENTING IT OUT FOR NOW')
            plot_scatter(embedding=self.embedding, labels=self.single_cell_pt_markov, sc_index_terminal_states=tsi_list,
                         title='pseudotime and terminal states', cmap='plasma', true_labels=self.true_label)

            plot_scatter(embedding=self.embedding, labels=self.true_label,
                         sc_index_terminal_states=tsi_list, title='lineage and terminal states', cmap='rainbow',
                         true_labels=self.labels)
            #plt.show()

        return


    def accuracy(self, onevsall=1):

        true_labels = self.true_label
        Index_dict = {}
        PARC_labels = self.labels
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
            majority_val = self.func_mode(vals)
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
            majority_truth = self.func_mode(list(true_labels[cluster_i_loc]))
            majority_truth_labels[cluster_i_loc] = majority_truth

        majority_truth_labels = list(majority_truth_labels.flatten())
        accuracy_val = [error_rate, f1_score, tnr, fnr, tpr, fpr, precision,
                        recall, num_groups, n_target]

        return accuracy_val, predict_class_array, majority_truth_labels, number_clusters_for_target

    def run_VIA(self):
        print(
            f'{datetime.now()}\tRunning VIA over input data of {self.data.shape[0]} (samples) x {self.data.shape[1]} (features)')
        print(f'{datetime.now()}\tKnngraph has {self.knn} neighbors')

        self.knn_struct = _construct_knn(self.data, knn=self.knn, distance=self.distance, num_threads=self.num_threads)
        st = time.time()
        self.run_subVIA()
        run_time = time.time() - st
        print(f'{datetime.now()}\tTime elapsed {round(run_time, 1)} seconds')
        do_accuracy = False

        if do_accuracy:
            print(f'{datetime.now()}\tCalculating accuracy scores for clusters in the StaVia graph. saved to via_object.stats_df as a dataframe')
            targets = list(set(self.true_label))
            targets.sort()

            N = len(self.true_label)
            self.f1_accumulated = 0
            self.f1_mean = 0
            self.stats_df = pd.DataFrame(
                {'edgepruning_clustering_resolution': [self.edgepruning_clustering_resolution], 'edgepruning_clustering_resolution_local': [self.edgepruning_clustering_resolution_local],
                 'runtime(s)': [run_time]})

            list_roc = []
            if len(targets) > 1:
                f1_accumulated, f1_acc_noweighting = 0, 0
                for onevsall_val in targets:
                    # print('target is', onevsall_val)
                    vals_roc, predict_class_array, majority_truth_labels, numclusters_targetval = \
                        self.accuracy(onevsall=onevsall_val)
                    f1_current = vals_roc[1]
                    f1_accumulated = f1_accumulated + f1_current * (list(self.true_label).count(onevsall_val)) / N
                    f1_acc_noweighting = f1_acc_noweighting + f1_current

                    list_roc.append([self.edgepruning_clustering_resolution, self.edgepruning_clustering_resolution_local, onevsall_val] +
                                    vals_roc + [numclusters_targetval] + [run_time])

                f1_mean = f1_acc_noweighting / len(targets)

                df_accuracy = pd.DataFrame(list_roc,
                                           columns=['edgepruning_clustering_resolution', 'edgepruning_clustering_resolution_local', 'onevsall-target', 'error rate',
                                                    'f1-score', 'tnr', 'fnr',
                                                    'tpr', 'fpr', 'precision', 'recall', 'num_groups',
                                                    'population of target', 'num clusters', 'clustering runtime'])
                # df_accuracy.to_csv('/home/user/Trajectory/Datasets/Cao_ProtoVert/df_accuracy_.csv')

                self.f1_accumulated = f1_accumulated
                self.f1_mean = f1_mean
                self.stats_df = df_accuracy
