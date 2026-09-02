from argparse import ArgumentTypeError
from . import mpi_wrapper as mpi_wrapper
import sys
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import optimize
from scipy.optimize import minimize
from . import bonsai_globals as bs_glob
import os, shutil
import csv
import pickle
from pathlib import Path
import resource
from datetime import datetime
from collections import namedtuple
from ruamel.yaml import YAML

custom_colors = ['#d7191c', '#fdae61', '#abd9e9']
custom_colors_rgb = [tuple(float(int(h.lstrip('#')[i:i + 2], 16)) / 255 for i in (0, 2, 4)) for h in custom_colors]
custom_colors_rgba = [tuple(list(rgb) + [1.0]) for rgb in custom_colors_rgb]
blackish = '#%02x%02x%02x' % (35, 31, 32)
blackish = (0.08578431372549018, 0.08578428015768168, 0.11935208866155156, 1.0)

# omicverse: matplotlib removed cm.get_cmap in 3.9; matplotlib.colormaps is
# the supported lookup from 3.5 onwards.
def _get_cmap(name):
    import matplotlib
    try:
        return matplotlib.colormaps[name]
    except (AttributeError, KeyError):
        return cm.get_cmap(name)

grey_cmap = _get_cmap('gray')
grey = grey_cmap(0.7)
# celltype_colors = _get_cmap('tab10')
other_colors = _get_cmap('tab20')
# gradient_colors = _get_cmap('terrain')
gradient_colors = _get_cmap('viridis')

import logging
FORMAT = '%(asctime)s %(funcName)s %(levelname)s %(message)s'
log_level = logging.WARNING
log_level = logging.DEBUG
logging.basicConfig(format=FORMAT,
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.WARNING)   # silence all libraries

# Create your app logger
logger = logging.getLogger("myapp")
logger.setLevel(log_level)

plt.set_loglevel(level='warning')


class Run_Configs:
    config_yaml = None

    def __init__(self, yaml_filepath=None, args=None, args_to_copy=None, step=None, create_empty_configs=False):

        pars_defaults = {'step': 'all',
                         'dataset': 'new_dataset',
                         'data_folder': None,
                         'filenames_data': 'delta.txt,d_delta.txt',
                         'results_folder': None,
                         'verbose': True,
                         'input_is_sanity_output': True,
                         'zscore_cutoff': 1.0,
                         'rescale_by_var': True,
                         'nnn_n_randommoves': 1000,
                         'nnn_n_randomtrees': 10,
                         'use_knn': 10,
                         'UB_ellipsoid_size': 1.0,
                         'skip_greedy_merging': False,
                         'skip_redo_starry': False,
                         'skip_opt_times': False,
                         'skip_spr_moves': False,
                         'skip_nnn_reordering': False,
                         'skip_reorder_edges': False,
                         'pickup_intermediate': False,
                         'tmp_folder': None,
                         'spr_strategy': 'large_tree'}

        if create_empty_configs:
            for label in pars_defaults:
                setattr(self, label, pars_defaults[label])
            return

        if yaml_filepath is not None and os.path.exists(yaml_filepath):
            yaml = YAML()
            with open(yaml_filepath, 'r') as file_obj:
                self.config_yaml = yaml.load(file_obj)
        else:
            print("No YAML-file with run configurations found{}".format(
                "." if yaml_filepath is None else " at {}".format(yaml_filepath)))
            print("Create a YAML-file using the script bonsai/create_config_file.py, or copy the template at "
                  "'bonsai/config_template_do_not_change/config_template.yaml' "
                  "and insert the correct configuration there.\n"
                  "Then add the path to this YAML-file in the '--config_filepath' parameters")
            exit()

        for label in pars_defaults:
            if (label in self.config_yaml) and (self.config_yaml[label] is not None):
                setattr(self, label, self.config_yaml[label])
            else:
                setattr(self, label, pars_defaults[label])

        if (args_to_copy is not None) and (args is not None):
            for label in args_to_copy:
                if hasattr(args, label):
                    setattr(self, label, getattr(args, label))

        # The following run-configurations are still used in Bonsai, but can no longer be set by users. Instead,
        # picking up intermediate results is only done through --pickup_intermediate and providing the correct
        # intermediate-results folders in the results-folder
        deprecated_args = ['skip_greedy_merging', 'skip_opt_times', 'skip_redo_starry', 'skip_spr_moves',
                           'skip_nnn_reordering', 'skip_reorder_edges']
        for dep_arg in deprecated_args:
            if hasattr(self, dep_arg) and getattr(self, dep_arg):
                setattr(self, dep_arg, False)
                mp_print(
                    "The argument '{}' is deprecated and no longer have any effect. "
                    "Use --pickup_intermediate True to pick up intermediate results from "
                    "previous Bonsai runs.".format(dep_arg),
                    WARNING=True)

        if step is not None:
            self.step = step
        # Add timestamp
        self.config_yaml['start_time'] = datetime.now().strftime('%d.%m.%y_%H:%M:%S')

    def store_yaml(self, yaml_filepath=None):
        if yaml_filepath is not None:
            yaml = YAML()
            with open(yaml_filepath, 'w') as file_obj:
                yaml.dump(self.config_yaml, file_obj)


def get_pdist_inds(shape, i, j):
    if i < j:
        return shape * i + j - ((i + 2) * (i + 1)) // 2
    else:
        return shape * j + i - ((j + 2) * (j + 1)) // 2


# def dist_ind_to_pair_ind(d, i):
#     b = 1 - 2 * d
#     x = np.floor((-b - np.sqrt(b**2 - 8*i))/2).astype(int)
#     y = (i + x * (b + x + 2) / 2 + 1).astype(int)
#     return x, y


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def mp_print(*args, **kwargs):
    """
    Multiprocessing wrapper for print().
    Prints the given arguments, but only on process 0 unless
    named argument ALL_RANKS is set to true.
    :return:
    """
    print_message = None
    error = False
    debug = False
    warning = False
    if ('ERROR' in kwargs) and kwargs['ERROR']:
        error = True
    if ('DEBUG' in kwargs) and kwargs['DEBUG']:
        debug = True
    if ('WARNING' in kwargs) and kwargs['WARNING']:
        warning = True
    mpi_rank = mpi_wrapper.get_process_rank()
    if 'ALL_RANKS' in kwargs and kwargs['ALL_RANKS']:
        print_message = "Process %d: " % (mpi_rank) + ' '.join(map(str, args))
    elif 'ONLY_RANK' in kwargs:
        if kwargs['ONLY_RANK'] == mpi_rank:
            print_message = "Process %d: " % (mpi_rank) + ' '.join(map(str, args))
    elif mpi_rank == 0:
        print_message = ' '.join(map(str, args))

    if print_message is None:
        return

    if error:
        logger.error(print_message)
    elif debug:
        logger.debug(print_message)
    elif warning:
        logger.warning(print_message)
    else:
        logger.info(print_message)

    sys.stdout.flush()
    sys.stderr.flush()


# Should go to tree vis file
def get_celltype_colors(n_celltypes, colortype=None, gradientType='hsv'):
    if (colortype is None) and (n_celltypes <= 8):
        col_HSC = "#0B5345"  # darkgreen
        col_MPP = "#229954"  # green
        col_LMPP = "#48C9B0"  # turchqoise
        col_CMP = "#AF601A"
        # col_UNK = "#E5E7E9"
        col_MEP = "#FE776D"
        col_pDC = "#A690A4"
        col_GMP = "#FCD0A1"
        col_CLP = "#AFD2E9"
        col_5h1 = "#FF7F11"  # orange
        col_5h2 = "#FF1B1C"  # red
        col_10h1 = "#10AFF8"  # light blue
        col_10h2 = "#0E2DF5"  # darker blue

        celltype_colors = [col_CLP, col_CMP, col_GMP, col_HSC, col_LMPP, col_MEP, col_MPP, col_pDC]
        celltype_colors = ListedColormap(celltype_colors)
        if n_celltypes == 4:
            celltype_colors = ListedColormap([col_10h1, col_10h2, col_5h1, col_5h2])
        # col_dict = {"MEP": col_MEP,
        #             "pDC": col_pDC,
        #             "GMP": col_GMP,
        #             "CLP": col_CLP,
        #             "HSC": col_HSC,
        #             "MPP": col_MPP,
        #             "LMPP": col_LMPP,
        #             "CMP": col_CMP,
        #             "UNK": col_UNK,
        #             "GMP-A": col_GMPA,
        #             "GMP-B": col_GMPB,
        #             "GMP-C": col_GMPC
        #             }
    elif (colortype is None) and (n_celltypes < 10):
        celltype_colors = _get_cmap('tab10')
    elif (colortype is None) and (n_celltypes < 20):
        celltype_colors = _get_cmap('tab20')
    elif colortype == 'offOn':
        tab10 = _get_cmap('tab10')
        two_colors = tab10([1, 2])
        two_colors[1, :] = grey_cmap(0.5)
        celltype_colors = ListedColormap(two_colors)
    else:
        gradient = _get_cmap(gradientType)
        celltype_colors = ListedColormap(gradient(np.linspace(0, 1, n_celltypes)))
    return celltype_colors


# Should go to tree vis file
def transform_coords_poincare(coords, origin=np.zeros(2), zoom=1, no_transform=False):
    # if no_transform:
    #     return (coords - origin) * zoom
    if no_transform:
        new_coords = coords - origin
        radii_euclidean = np.sqrt(np.sum(new_coords ** 2, axis=1))
        radii_euclidean *= zoom
        large_radii = radii_euclidean > 1
        radii_euclidean[large_radii] = 1.1
        nonzeros = radii_euclidean != 0
    else:
        new_coords = coords - origin
        radii_euclidean = np.sqrt(np.sum(new_coords ** 2, axis=1))
        radii_euclidean *= zoom
        nonzeros = radii_euclidean != 0
        radii_euclidean[nonzeros] = -1 / (2 * radii_euclidean[nonzeros]) + np.sqrt(
            1 + 1 / (4 * (radii_euclidean[nonzeros] ** 2)))
    coords_transformed = np.zeros(new_coords.shape)
    coords_transformed[nonzeros, :] = radii_euclidean[nonzeros, np.newaxis] * (
            new_coords[nonzeros, :] / np.linalg.norm(new_coords[nonzeros, :], axis=1, ord=2)[:, np.newaxis])
    return coords_transformed


# Should go to tree vis file
def invert_coords_poincare(coords, origin=np.zeros(2), zoom=1, no_transform=False):
    if no_transform:
        radii = np.sqrt(np.sum(coords ** 2, axis=1))
        small_radii = radii <= 1
        new_coords = coords.copy()
        new_coords[small_radii] = coords[small_radii] / zoom + origin
        return coords / zoom + origin
    coords_polar = np.zeros(coords.shape)
    coords_polar[:, 0] = np.sqrt(np.sum(coords ** 2, axis=1))
    coords_polar[:, 1] = np.arctan2(coords[:, 1], coords[:, 0])
    coords_polar[:, 0] = (1 / zoom) * coords_polar[:, 0] / (1 - (coords_polar[:, 0] ** 2))
    coords_transformed = np.zeros(coords_polar.shape)
    coords_transformed[:, 0] = coords_polar[:, 0] * np.cos(coords_polar[:, 1])
    coords_transformed[:, 1] = coords_polar[:, 0] * np.sin(coords_polar[:, 1])
    coords_transformed += origin
    return coords_transformed


# Should go to tree vis file
def pause_anim(t):  # This is taken from plt.pause(...), but without unnecessary
    # stuff. Note that the time module should be previously imported.
    # Again, this use the controversial event_loop of Matplotlib.
    backend = plt.rcParams['backend']
    if backend in plt._interactive_bk:
        figManager = plt._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            figManager.canvas.start_event_loop(t)
            return
    else:
        time.sleep(t)


# Should go to tree vis file
def get_opt_zoom_origin_poincare(coords, frac_within=0.8, within_radius=0.8, tol=1e-6, max_iter=20, verbose=False):
    converged = False
    old_zoom = 1
    origin0 = np.mean(coords, axis=0)
    counter = 0
    while not converged:
        counter += 1
        new_zoom = get_zoom_given_origin(origin0, coords, frac_within=frac_within, within_radius=within_radius)
        origin0 = get_centroid_poincare(coords, zoom=new_zoom, verbose=verbose)
        if verbose:
            print('Zoom is ' + str(new_zoom))
            print('Origin is ' + str(origin0) + '\n')
        if np.abs(old_zoom - new_zoom) < tol:
            converged = True
        else:
            old_zoom = new_zoom
        if counter == max_iter:
            print("Maximum number of iterations reached: just returning last found value.")
            print("Zoom value converged to " + str(np.abs(old_zoom - new_zoom)))
            converged = True
    return new_zoom, origin0


# Should go to tree vis file
def get_zoom_given_origin(origin, coords, frac_within=0.8, within_radius=.8):
    n_points = coords.shape[0]
    n_points_within = int(np.ceil(frac_within * n_points))

    def radius_needed(logzoom):
        zoom = np.exp(logzoom)
        needed_radius = np.sort(get_radial_poincare(coords, origin=origin, zoom=zoom))[
                            n_points_within - 1] - within_radius
        return needed_radius

    bracket = (-6, 1)
    bracket_ok = False
    while not bracket_ok:
        if radius_needed(bracket[0]) >= 0:
            bracket = (bracket[0] - 1, bracket[1])
        elif radius_needed(bracket[1]) <= 0:
            bracket = (bracket[0], bracket[1] + 1)
        else:
            bracket_ok = True

    opt_result = optimize.root_scalar(radius_needed, bracket=bracket)
    if opt_result.converged:
        result = np.exp(opt_result.root)
    else:
        print(opt_result.flag)
        print("Finding optimal zoom failed. Using 0.001 instead.")
        result = 0.001

    return result


# Should go to tree vis file
def get_radial_poincare(coords, origin=np.zeros(2), zoom=1):
    new_coords = coords - origin
    radii_euclidean = np.sqrt(np.sum(new_coords ** 2, axis=1))
    # nonzeros = coords[:, 0] != 0
    # coords_polar[~nonzeros, 1] = np.sign(coords[~nonzeros, 1])
    radii_euclidean *= zoom
    # coords_transformed_polar = coords_polar.copy()
    # coords_transformed_polar[:, 0] = -1 / (2 * coords_polar[:, 0]) + np.sqrt(1 + 1 / (4 * (coords_polar[:, 0] ** 2)))
    nonzeros = radii_euclidean != 0
    radii_euclidean[nonzeros] = -1 / (2 * radii_euclidean[nonzeros]) + np.sqrt(
        1 + 1 / (4 * (radii_euclidean[nonzeros] ** 2)))
    return radii_euclidean


# Should go to tree vis file
def get_centroid_poincare(coords, zoom=1, verbose=False):
    origin0 = np.mean(coords, axis=0)

    def dist_fun(ori):
        return np.mean(get_radial_poincare(coords, origin=ori, zoom=zoom))

    opt_result = minimize(dist_fun, origin0)
    if opt_result.success:
        if verbose:
            print(opt_result.message)
        origin = opt_result.x
    else:
        print(opt_result.message)
        print("We take the centroid of the graph in Euclidean space instead.")
        origin = origin0
    return origin


# Should go to tree vis file
class InteractionEvent:
    function = None
    kwargs = None

    def __init__(self, func=None, kwargs=None):
        if func is None:
            print("You should pass a funcion.")
            return
        self.function = func
        self.kwargs = kwargs

    def execute(self):
        if self.kwargs is not None:
            self.function(**self.kwargs)
        else:
            self.function()


def checkGrad(funGrad, coords, args, dx=1e-6, dims=None):
    """
    This function checks for a (multivariate) function that returns a function and a gradient as its two outputs
    whether the gradient matches a numerical gradient.
    :param funGrad: function that takes a numpy-vector of size coords.shape and returns a function value and its
    gradient
    :param coords: Point at which we want to check the  gradient
    :return: numpy vector of deviation of gradient from numerical derivative
    """
    if dims is None:
        nDims = len(coords)
        dims = np.array(range(nDims))
    else:
        nDims = len(dims)
    deviations = np.zeros(nDims)
    relDeviations = np.zeros(nDims)
    funOrig, gradOrig = funGrad(coords, *args)
    for dimInd, dim in enumerate(dims):
        if dimInd % 10 == 0:
            mp_print("Checking derivative for dimension " + str(dimInd) + " out of " + str(nDims))
        coordsdx = coords.copy()
        coordsMinusDx = coords.copy()
        coordsdx[dim] += dx
        coordsMinusDx[dim] -= dx
        numGrad = (funGrad(coordsdx, *args)[0] - funGrad(coordsMinusDx, *args)[0]) / (2 * dx)
        deviations[dimInd] = gradOrig[dim] - numGrad
        relDeviations[dimInd] = abs(deviations[dimInd] / gradOrig[dim])

    return deviations, relDeviations


# Used
def hierarchy_to_newick(pathToMerger, clustIds, pathToOutput):
    """
    Function for getting a newick string from a hierarchy DataFrame.
    Parameters
    ----------
    Returns
    -------
    newick_string : str
        string of cluster hierarchy in Newick format
    """
    import pandas as pd
    mergers_df = pd.read_csv(pathToMerger, names=['cluster_new', 'cluster_old', 'dist1', 'dist2'], delimiter=' ')

    cluster_string_dict = {}
    # cluster_distance = {c: min_distance for c in cluster_names}
    for cInd, cId in enumerate(clustIds):
        cluster_string = f'C{cId}'
        cluster_string_dict[cInd] = cluster_string

    c_low = min(cluster_string_dict.keys())

    # if distance:
    #     distances = np.cumsum(np.where((hierarchy_df.delta_LL >= 0), 1, -hierarchy_df.delta_LL + 1))

    # use index i instead of step in case hierarchy_df.index is non-standard
    i = mergers_df.shape[0] - 1
    for step, row in mergers_df.iterrows():
        c_old = row.cluster_old
        c_new = row.cluster_new
        s_old = cluster_string_dict[c_old]
        s_new = cluster_string_dict[c_new]

        # d = distances[-i - 1]
        d_old = row[2]
        d_new = row[3]
        cluster_string_new = f'({s_new}:{d_new},{s_old}:{d_old})I{i}'

        cluster_string_dict[c_new] = cluster_string_new
        del cluster_string_dict[c_old]
        # del cluster_distance[c_old]

        i -= 1

    newick_string = cluster_string_dict[c_new] + ';'
    with open(pathToOutput, 'w') as f:
        f.write("%s" % newick_string)


# Should go to tree vis file
def compile_tree_from_newick(newickFilename):
    from skbio import TreeNode
    njTreenode = TreeNode.read(newickFilename, convert_underscores=False)
    return compile_tree_from_nj_treenode(njTreenode)


# Should go to tree vis file
def compile_tree_from_nj_treenode(njTreenode):
    # Convert to igraph tree
    tree = njTreenode
    internal_counter = 0
    id_ind_dict = {}
    curr_ind = 0
    edge_list = []
    dist_list = []
    orig_vert_names = []
    for node in tree.preorder():
        # If node is internal it does not have an ID, so we give it one.
        if node.name is None:
            node.name = 'internal_' + str(internal_counter)
            internal_counter += 1

        # We keep track of which node is at which index by updating a dict
        id_ind_dict[node.name] = curr_ind
        orig_vert_names.append(node.name)

        # For each node that we traverse, we add an edge to its parent with the right length
        if not node.is_root():
            edge_list.append((id_ind_dict[node.parent.name], curr_ind))
            dist_list.append(node.length)
        curr_ind += 1
    return edge_list, dist_list, orig_vert_names


# Used
def startMPI(verbose=True):
    mpi_wrapper.mpi_init()
    mpiSize = mpi_wrapper.get_process_size()
    mpiRank = mpi_wrapper.get_process_rank()
    if verbose and (mpiRank in [mpiSize - 1, 0]):
        mp_print("Process " + str(mpiRank) + " out of " + str(mpiSize) + " has started.", ALL_RANKS=True)
    return mpiRank, mpiSize


# TODO: Do this without networkx. Only used to get sparse distance matrix
def get_shortest_distances_on_tree(ig_graph, indices=None):
    from scipy.sparse.csgraph import floyd_warshall, shortest_path
    from scipy.sparse import csr_matrix
    from scipy.spatial.distance import squareform
    edgelist = ig_graph.get_edge_dataframe()[['source', 'target', 'weight']]

    # weighted_edges = list(edgelist.to_records(index=False))
    # import networkx as nx
    # G = nx.Graph()
    # G.add_weighted_edges_from(weighted_edges)
    # distance_csr = ig_graph.get_adjacency_sparse(attribute="weight")
    # distances = squareform(floyd_warshall(csgraph=distance_csr, directed=False))

    cols = edgelist['source'].values
    rows = edgelist['target'].values
    weights = edgelist['weight'].values
    colsComplete = np.concatenate((cols, rows))
    rowsComplete = np.concatenate((rows, cols))
    weightsComplete = np.concatenate((weights, weights))
    nVerts = np.max(colsComplete) + 1
    distance_csr = csr_matrix((weightsComplete, (rowsComplete, colsComplete)), shape=(nVerts, nVerts))

    distances = squareform(shortest_path(distance_csr, method='auto', directed=False, return_predecessors=False,
                                         unweighted=False, overwrite=False, indices=indices)[:, indices], checks=False)
    return distances


# Used
def getOutputFolder(zscore_cutoff=-1, greedy=True, redo_starry=True, opt_times=True, final=False,
                    reorderedEdges=False, spr_moves=False, nnn_reorder=False, tmp_file='', get_all_possibilities=False):
    if get_all_possibilities:
        all_possibilities = []

    # Distinguish preprocessed-results from tree-reconstruction results
    if final:
        mergerFolder = 'final_bonsai'
    elif not greedy:
        mergerFolder = 'preprocessed'
    else:
        mergerFolder = 'intermediate_bonsai'

    # Store zscore_cutoff in the filename
    if zscore_cutoff > 0:
        mergerFolder += "_zscore" + str(zscore_cutoff)
    if get_all_possibilities:
        all_possibilities.append(mergerFolder)

    # Add the several steps that we already did
    if (redo_starry or get_all_possibilities) and (not final):
        mergerFolder += "_redoStarry"
    if get_all_possibilities:
        all_possibilities.append(mergerFolder)
    if (opt_times or get_all_possibilities) and (not final):
        mergerFolder += "_optTimes"
    if get_all_possibilities:
        all_possibilities.append(mergerFolder)
    if (spr_moves or get_all_possibilities) and (not final):
        mergerFolder += "_sprMoves"
    if get_all_possibilities:
        all_possibilities.append(mergerFolder)
    if (nnn_reorder or get_all_possibilities) and (not final):
        mergerFolder += "_nnnReorder"
    if get_all_possibilities:
        all_possibilities.append(mergerFolder)
    if (reorderedEdges or get_all_possibilities) and (not final):
        mergerFolder += "_reorderedEdges"
    if get_all_possibilities:
        all_possibilities.append(mergerFolder)
    if len(tmp_file) > 0:
        mergerFolder += "_tmpStart" + tmp_file
        if get_all_possibilities:
            for ind in range(len(all_possibilities)):
                all_possibilities[ind] += "_tmpStart" + tmp_file
    if get_all_possibilities:
        return all_possibilities
    return mergerFolder


def find_latest_tree_folder(results_folder, not_final=False):
    tree_folder = None
    for file in os.listdir(results_folder):
        if (file[:12] == 'final_bonsai') and (not not_final):
            tree_folder = file
            break
        elif file[:19] == 'intermediate_bonsai':
            if tree_folder is None:
                tree_folder = file
            elif len(file) > len(tree_folder):
                tree_folder = file
    if tree_folder is None:
        tree_folder = results_folder
    return tree_folder


def find_latest_tree_folder_new(args, results_folder, not_final=False, set_skip_args=False):
    # tree_folder = None

    if set_skip_args:
        # Start with assuming we can skip everything
        args.skip_greedy_merging = True
        args.skip_redo_starry = True
        args.skip_opt_times = True
        args.skip_spr_moves = True
        args.skip_nnn_reordering = True
        args.skip_reorder_edges = True

    # We go over the possible tree folders in reverse order, i.e., most advanced first, and check whether it's present
    # 1. Finished tree folder
    if not not_final:
        dir_path = getOutputFolder(zscore_cutoff=args.zscore_cutoff, tmp_file=os.path.basename(args.tmp_folder),
                                   final=True)
        if os.path.exists(os.path.join(results_folder, dir_path, 'vertInfo.txt')):
            return dir_path

    # 2. Check if done all computation steps, except for the final metadata
    dir_path = getOutputFolder(zscore_cutoff=args.zscore_cutoff, redo_starry=True, opt_times=True, final=False,
                               reorderedEdges=True, nnn_reorder=True, spr_moves=True,
                               tmp_file=os.path.basename(args.tmp_folder))
    if os.path.exists(os.path.join(results_folder, dir_path, 'vertInfo.txt')):
        return dir_path

    # 3. Check if done everything but swapping the order of branches and setting root
    if set_skip_args:
        args.skip_reorder_edges = False
    dir_path = getOutputFolder(zscore_cutoff=args.zscore_cutoff, redo_starry=True, opt_times=True, final=False,
                               nnn_reorder=True, reorderedEdges=False, spr_moves=True,
                               tmp_file=os.path.basename(args.tmp_folder))
    if os.path.exists(os.path.join(results_folder, dir_path, 'vertInfo.txt')):
        return dir_path

    # 4. Check if done everything up to nn-interchanges
    if set_skip_args:
        args.skip_nnn_reordering = False
    dir_path = getOutputFolder(zscore_cutoff=args.zscore_cutoff, redo_starry=True, opt_times=True, final=False,
                               nnn_reorder=False, reorderedEdges=False, spr_moves=True,
                               tmp_file=os.path.basename(args.tmp_folder))
    if os.path.exists(os.path.join(results_folder, dir_path, 'vertInfo.txt')):
        return dir_path

    # 4. Check if done everything up to SPR-moves
    if set_skip_args:
        args.skip_spr_moves = False
    dir_path = getOutputFolder(zscore_cutoff=args.zscore_cutoff, redo_starry=True, opt_times=True, final=False,
                               nnn_reorder=False, reorderedEdges=False, spr_moves=False,
                               tmp_file=os.path.basename(args.tmp_folder))
    if os.path.exists(os.path.join(results_folder, dir_path, 'vertInfo.txt')):
        return dir_path

    # 4. Check if done everything up to time optimization
    if set_skip_args:
        args.skip_opt_times = False
    dir_path = getOutputFolder(zscore_cutoff=args.zscore_cutoff, redo_starry=True, opt_times=False, final=False,
                               nnn_reorder=False, reorderedEdges=False, spr_moves=False,
                               tmp_file=os.path.basename(args.tmp_folder))
    if os.path.exists(os.path.join(results_folder, dir_path, 'vertInfo.txt')):
        return dir_path

    # 4. Check if done everything up to redoing-starry, i.e., only the greedy merging
    if set_skip_args:
        args.skip_redo_starry = False
    dir_path = getOutputFolder(zscore_cutoff=args.zscore_cutoff, redo_starry=False, opt_times=False, final=False,
                               nnn_reorder=False, reorderedEdges=False, spr_moves=False,
                               tmp_file=os.path.basename(args.tmp_folder))
    if os.path.exists(os.path.join(results_folder, dir_path, 'vertInfo.txt')):
        return dir_path

    # 5. Check if only preprocessing has been done
    if set_skip_args:
        args.skip_greedy_merging = False
    dir_path = getOutputFolder(zscore_cutoff=args.zscore_cutoff, greedy=False, redo_starry=False, opt_times=False,
                               final=False, tmp_file=os.path.basename(args.tmp_folder))
    if os.path.exists(os.path.join(results_folder, dir_path, 'vertInfo.txt')):
        return dir_path

    # If nothing has been done, just return the results_folder
    return results_folder

def clean_up_redundant_data_files(scData, args, verbose=True):
    if verbose:
        mp_print("Cleaning up intermediate datafiles.")
    all_possible_folders = getOutputFolder(zscore_cutoff=args.zscore_cutoff,
                                           tmp_file=os.path.basename(args.tmp_folder), get_all_possibilities=True)
    try:
        for tree_folder in all_possible_folders:
            tree_folder = scData.result_path(tree_folder)
            if os.path.exists(tree_folder) and os.path.isdir(tree_folder):
                for filename in os.listdir(tree_folder):
                    if filename[-15:] == '_vertByGene.npy':
                        os.remove(os.path.join(tree_folder, filename))
    except:
        mp_print("Somehow couldn't fully remove directory with intermediate tree-files. Not a problem, these "
                 "directories don't take much memory.")


# Used
def getNodePairInfo(pairInfoDict, sortedNodeInds):
    info = None
    if sortedNodeInds in pairInfoDict:
        info = pairInfoDict[sortedNodeInds]
    return info


# Used
def communicateNodePairInfo(oldPairInfoDict, newPairInfoDict, mpi_rank, allRanksGetInfo=True):
    newPairInfoDicts = mpi_wrapper.world_allgather(newPairInfoDict)
    if (mpi_rank == 0) or allRanksGetInfo:
        for infoDict in newPairInfoDicts:
            oldPairInfoDict.update(infoDict)
    else:
        oldPairInfoDict = None
    return oldPairInfoDict


# Used
def cleanUpPairDict(deletedNodes, pairDict):
    keys, vals = zip(*pairDict.items())
    keyArray = np.array(keys)
    valArray = np.array(vals)
    for node in deletedNodes:
        toBeDeleted = np.where(keyArray == node)[0]
        keyArray = np.delete(keyArray, toBeDeleted, axis=0)
        valArray = np.delete(valArray, toBeDeleted, axis=0)
    return dict(zip(list(map(tuple, keyArray)), list(valArray)))


# Used
def communicateGlobalVars():
    allGlobalVars = sorted(list(bs_glob.globalVars))
    for globalVar in allGlobalVars:
        thisVar = mpi_wrapper.bcast(getattr(bs_glob, globalVar), root=0)
        setattr(bs_glob, globalVar, thisVar)


# Used
def calcTInit(tOld1, tOld2, sequential):
    tAnc0 = min(tOld1, tOld2) / 2
    if sequential:
        t0_i = [tOld1 + tOld2 - 2 * tAnc0, tAnc0]
    else:
        t0_i = [tOld1 - tAnc0, tOld2 - tAnc0, tAnc0]
    return t0_i


# Used
def getAllDLogLs(dLogLDict, mpi_rank, nChildren):
    # Gather all dLogLs on the first process
    dLogLsDicts = mpi_wrapper.world_allgather(dLogLDict)
    if mpi_rank != 0:
        dLogLs = None
    else:
        allDLogLsDict = {}
        for dLogLDict in dLogLsDicts:
            allDLogLsDict.update(dLogLDict)
        dLogLs = np.zeros(int(nChildren * (nChildren - 1) / 2))

        curr_pairs = nChildren - 1
        done = 0
        for i in range(nChildren - 1):
            dLogLs[done:done + curr_pairs] = allDLogLsDict[i]
            done += curr_pairs
            curr_pairs -= 1
    return dLogLs


# Used
def getMyTaskNumbers(nTasks, mpiSize, mpiRank, skippingSteps=False):
    if skippingSteps:
        return np.arange(mpiRank, nTasks, mpiSize)
    quotient, remainder = divmod(nTasks, mpiSize)
    section_sizes = ([] +
                     remainder * [quotient + 1] +
                     (mpiSize - remainder) * [quotient])
    prevTasks = np.append(0, np.cumsum(section_sizes))
    return np.arange(prevTasks[mpiRank], prevTasks[mpiRank + 1])


def printTiming(taskDescription, oldTime):
    currTime = time.time()
    mp_print("Timing: %s took %f seconds." % (taskDescription, currTime - oldTime))
    return currTime


def writeVectToEndCsv(filename, vect, init=False):
    if mpi_wrapper.get_process_rank() == 0:
        if init:
            if os.path.exists(filename):
                os.remove(filename)
        if len(vect) > 0:
            with open(filename, "a") as file:
                file.write('\t'.join(np.char.mod('%.8e', vect)) + '\n')


def readListFromTxt(filename):
    inputList = []
    if not os.path.exists(filename):
        print("Can't read from %s, file not found." % filename)
        return None
    else:
        with open(filename, 'r') as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                inputList.append(row[0])
        return inputList


def create_celltypefile_from_cellstates(cellstates_file, celltype_file, multiplicityCutoff=0):
    with open(cellstates_file, 'r') as file:
        reader = csv.reader(file, delimiter="\t")
        cellstates = []
        for row in reader:
            cellstates.append("CS_" + row[0])
    unq_cellstates, inverse, counts = np.unique(cellstates, return_inverse=True, return_counts=True)
    if os.path.exists(celltype_file):
        os.remove(celltype_file)
    with open(celltype_file, "a") as file:
        for ind, cellstate in enumerate(cellstates):
            cellstate = cellstate if counts[inverse[
                ind]] > multiplicityCutoff else "cs with <= %d cells" % multiplicityCutoff
            file.write(cellstate + '\n')


# def storeCurrentState(outputFolder, scData, filename='tmp_tree.dat', dataOrResults='data', args=None, dataFolder=None):
#     if mpi_wrapper.get_process_rank() == 0:
#         if dataFolder is None:
#             if args.dataset is not None:
#                 dataFolder = os.path.join(dataOrResults, args.dataset)
#             else:
#                 dataFolder = args.data_folder
#         tmpDataPath = os.path.join(dataFolder, outputFolder)
#         Path(tmpDataPath).mkdir(parents=True, exist_ok=True)
#
#         scData.tree.nNodes = bs_glob.nNodes
#         mp_print("Printing tree at %s" % os.path.join(tmpDataPath, filename))
#         try:
#             with open(os.path.join(tmpDataPath, filename), 'wb') as file:
#                 pickle.dump(scData, file)
#         except RecursionError:
#             with RecursionLimit():
#                 with open(os.path.join(tmpDataPath, filename), 'wb') as file:
#                     pickle.dump(scData, file)


def get_latest_intermediate(intermediateFiles, base=None):
    tmpTreeInd = None
    try:
        indexed_files = []
        for ind, intFile in enumerate(intermediateFiles):
            if base is not None:
                if intFile.split('_')[0] != base:
                    indexed_files.append(np.nan)
                    continue
            index_string = intFile.split('.')[0].split('_')[-1]
            indexed_files.append(int(index_string)) if index_string.isdigit() else indexed_files.append(np.nan)
        if len(indexed_files):
            latestIntFile = np.nanargmax(indexed_files)
            tmpTreeInd = indexed_files[latestIntFile]
        else:
            mp_print("Could not find intermediate file, starting from scratch", ALL_RANKS=True)
            return None, None
    except:
        mp_print("Could not find intermediate file, starting from scratch.", ALL_RANKS=True)
        return None, None
    return intermediateFiles[latestIntFile], tmpTreeInd


# def pickleTree(pickleFolder, filename, tree):
#     Path(pickleFolder).mkdir(parents=True, exist_ok=True)
#     tree.nNodes = bs_glob.nNodes
#     try:
#         with open(os.path.join(pickleFolder, filename), 'wb') as file:
#             pickle.dump(tree, file)
#     except RecursionError:
#         with RecursionLimit():
#             with open(os.path.join(pickleFolder, filename), 'wb') as file:
#                 pickle.dump(tree, file)


# def unpickleTree(pickleFolder, filename):
#     mp_print("Trying to unpickle tree from ", os.path.join(pickleFolder, filename), ALL_RANKS=True)
#     try:
#         with open(os.path.join(pickleFolder, filename), 'rb') as file:
#             tree = pickle.load(file)
#     except RecursionError:
#         with RecursionLimit():
#             with open(os.path.join(pickleFolder, filename), 'rb') as file:
#                 tree = pickle.load(file)
#     bs_glob.nNodes = tree.nNodes
#     bs_glob.nGenes = len(tree.root.ltqs)
#     return tree


def remove_tree_folders(tree_folder, removeDir=False, notRemove=None, base=None):
    empty = True
    if not os.path.exists(tree_folder):
        return
    for intFile in os.listdir(tree_folder):
        try:
            if (base is not None) and (intFile.split('_')[0] != base):
                empty = False
                continue
            if notRemove is not None:
                index_string = intFile.split('.')[0].split('_')[-1]
                if index_string.isdigit() and (int(index_string) in [notRemove, notRemove - 1]):
                    empty = False
                    continue
            if os.path.isdir(os.path.join(tree_folder, intFile)):
                shutil.rmtree(os.path.join(tree_folder, intFile))
            else:
                os.remove(os.path.join(tree_folder, intFile))
        except:
            mp_print("Somehow couldn't remove intermediate trees.")
            empty = False
    if removeDir and empty:
        try:
            shutil.rmtree(tree_folder)
        except FileNotFoundError or OSError:
            mp_print("Process %d: Apparently, folder %s was already removed." % (
                mpi_wrapper.get_process_rank(), tree_folder), ALL_RANKS=True)


def remove_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
    except FileNotFoundError or OSError:
        mp_print("Apparently, folder %s was already removed." % folder_path)


def empty_folder(folder_path):
    try:
        for file_path in os.listdir(folder_path):
            os.remove(os.path.join(folder_path, file_path))
    except:
        mp_print("Somehow couldn't empty folder %s." % folder_path)


def removePickledTrees(pickleFolder, removeDir=False, notRemove=None, base=None):
    try:
        for intFile in os.listdir(pickleFolder):
            if (base is not None) and (intFile.split('_')[0] != base):
                continue
            if notRemove is not None:
                index_string = intFile.split('.')[0].split('_')[-1]
                if index_string.isdigit() and (int(index_string) in [notRemove, notRemove - 1]):
                    continue
            os.remove(os.path.join(pickleFolder, intFile))
    except:
        mp_print("Somehow couldn't remove pickled intermediate trees.")
    if removeDir:
        try:
            for intFile in os.listdir(pickleFolder):
                os.remove(os.path.join(pickleFolder, intFile))
            os.rmdir(pickleFolder)
        except FileNotFoundError or OSError:
            mp_print("Process %d: Apparently, folder %s was already removed." % (
                mpi_wrapper.get_process_rank(), pickleFolder))


def recursionWrap(fun, *args, **kwargs):
    with RecursionLimit():
        result = fun(*args, **kwargs)
    return result


def set_recursion_limits(new_limit):
    if sys.platform.startswith('linux'):
        old_resourcelimit = resource.getrlimit(resource.RLIMIT_STACK)
        # resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
        resource.setrlimit(resource.RLIMIT_STACK, (old_resourcelimit[0], resource.RLIM_INFINITY))
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, new_limit))


class RecursionLimit:
    limit = 10 ** 6
    old_limit = None
    old_resourcelimit = None

    def __init__(self, limit=None):
        if limit is not None:
            self.limit = limit

    def __enter__(self):
        if sys.platform.startswith('linux'):
            self.old_resourcelimit = resource.getrlimit(resource.RLIMIT_STACK)
            # resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
            resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        self.old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(10 ** 6)

    def __exit__(self, type, value, tb):
        sys.setrecursionlimit(self.old_limit)
        if sys.platform.startswith('linux'):
            resource.setrlimit(resource.RLIMIT_STACK, self.old_resourcelimit)


def broadcastRecursiveStruct(recursiveStruct, root=0):
    try:
        recursiveStruct = mpi_wrapper.bcast(recursiveStruct, root=root)
    except RecursionError:
        with RecursionLimit():
            recursiveStruct = mpi_wrapper.bcast(recursiveStruct, root=root)
    return recursiveStruct


def time_format():
    now = datetime.now()
    timestr = now.strftime("%H:%M:%S")
    return f'{timestr}|> '


def convert_dict_to_named_tuple(d):
    namedTupleConstructor = namedtuple('myNamedTuple', ' '.join(sorted(d.keys())))
    nt = namedTupleConstructor(**d)
    return nt


def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


# Kruskal's algorithm in Python
class OwnGraph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])

    # Search function

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def apply_union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    #  Applying Kruskal algorithm
    def kruskal_algo(self, verbose=True, very_verbose=False):
        result = []
        i, e = 0, 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent = []
        rank = []
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
        while e < self.V - 1:
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)
            if x != y:
                if verbose and ((e + 1) % 1000 == 0):
                    print("Adding edge " + str(e + 1) + " out of " + str(self.V - 1))
                e = e + 1
                result.append([u, v, w])
                self.apply_union(parent, rank, x, y)
        if very_verbose:
            for u, v, weight in result:
                print("%d - %d: %d" % (u, v, weight))
        return result

    def kruskal_algo_maxdegree_two(self, verbose=True, very_verbose=False):
        result = []
        i, e = 0, 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent = []
        rank = []
        degree = []
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
            degree.append(0)
        while e < self.V - 1:
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)
            if x != y:
                if (degree[u] < 2) and (degree[v] < 2):
                    if verbose and ((e + 1) % 1000 == 0):
                        print("Adding edge " + str(e + 1) + " out of " + str(self.V - 1))
                    e = e + 1
                    result.append([u, v])
                    self.apply_union(parent, rank, x, y)
                    degree[u] += 1
                    degree[v] += 1
        if very_verbose:
            for u, v in result:
                print("%d - %d" % (u, v))
        return result


def add_celltype_info_to_tree(treenode, cell_id_to_ct):
    if treenode.nodeId in cell_id_to_ct:
        treenode.celltype = cell_id_to_ct[treenode.nodeId]
    else:
        treenode.celltype = 'unknown'
    for child in treenode.childNodes:
        add_celltype_info_to_tree(child, cell_id_to_ct)


def read_ids(filepath):
    if not os.path.exists(filepath):
        logger.error("Can't find IDs stored at {}.".format(filepath))
        return []
    ids = []
    with open(filepath, 'r') as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            ids.append(row[0])
    return ids


def write_ids(filepath, ids_list):
    with open(filepath, 'w') as f:
        for ID in ids_list:
            f.write("%s\n" % ID)


def print_memory(location=None):
    import psutil
    import tracemalloc
    import resource
    message = ''
    if location is not None:
        message += "{}: ".format(location)
    # mp_print("Current RSS memory usage is ",
    #       psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ALL_RANKS=True)
    # print("Current shared memory usage is ",
    #       psutil.Process(os.getpid()).memory_info().shared / 1024 ** 2, " MB.")
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    message += "Current-mem: {} MB, Peak-mem since last: {} MB".format(current / 1000 ** 2, peak / 1000 ** 2)
    # mp_print("Python data segment =",
    #       resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, "MB", ALL_RANKS=True)
    mp_print(message, ALL_RANKS=True)


def store_scdata_and_communicate_path(scData, specific_results_folder, general_results_folder=None, prefix=None,
                                      with_coords=True):
    mpi_info = mpi_wrapper.get_mpi_info()
    if mpi_info.rank == 0:
        no_scdata_path = True
        while no_scdata_path:
            if prefix is None:
                subfoldername = 'scdata_state_{}'.format(np.random.randint(1e6))
            else:
                subfoldername = prefix + '_' + 'scdata_state_{}'.format(np.random.randint(1e6))
            if general_results_folder is None:
                scdata_path = scData.result_path(os.path.join(specific_results_folder, subfoldername))
            else:
                scdata_path = os.path.join(general_results_folder, specific_results_folder, subfoldername)
            if not os.path.exists(scdata_path):
                no_scdata_path = False
        # We store the tree to make sure we have the data of all vertices
        scData.storeTreeInFolder(os.path.join(scdata_path), with_coords=with_coords, verbose=False, nwk=False)

        # Communicate the path where scData is stored with the other processes. This also serves as a barrier between
        # storing the tree and loading it on other processes
        scdata_path = mpi_wrapper.bcast(scdata_path, root=0)
    else:
        scdata_path = None
        # Other processes receive the path to scData
        scdata_path = mpi_wrapper.bcast(scdata_path, root=0)
    return scdata_path


def look_for_output_folder(specific_results_folder, general_results_folder=None, prefix=None):
    if general_results_folder is not None:
        full_path = os.path.join(general_results_folder, specific_results_folder)
    else:
        full_path = specific_results_folder

    full_path = Path(full_path)

    if not full_path.exists() or not full_path.is_dir():
        return False, None

    subfolders = [p for p in full_path.iterdir() if p.is_dir() and (prefix is None or p.name.startswith(prefix))]
    if len(subfolders) == 0:
        return False, None

    # Return the folder that was last modified
    scdata_path = max(subfolders, key=lambda p: p.stat().st_mtime)
    return True, scdata_path
