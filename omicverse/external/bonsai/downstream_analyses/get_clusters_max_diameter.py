from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
import sys
from .get_cluster_helpers import Cluster_Tree, entropy_non_norm
from ..bonsai.bonsai_helpers import mp_print
from itertools import combinations
import random
from collections import defaultdict
import csv
import time
from scipy.stats import entropy


# parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# # Add the parent directory to sys.path
# sys.path.append(parent_dir)
# os.chdir(parent_dir)
# sys.path.append(os.path.join(parent_dir, 'tree_layout'))

# def get_cluster_assignments(clusters_list, assign_singlets_together=False):
#     cluster_idx = 0
#     cluster_assigment = []
#     cell_names = []
#     for cluster in clusters_list:
#         # if singleton, assign -1
#         if assign_singlets_together and (len(cluster) == 1):
#             cluster_assigment.append("cl_{}".format(-1))
#             cell_names.append(cluster[0])
#             # fout.write("{}\t{}\n".format(cluster[0], -1))
#         else:
#             for leaf in cluster:
#                 # fout.write("{}\t{}\n".format(leaf, cluster_idx))
#                 cluster_assigment.append("cl_{}".format(cluster_idx))
#                 cell_names.append(leaf)
#             cluster_idx += 1
#
#     # make dict:
#
#     cl_dict = dict(zip(cell_names, cluster_assigment))
#     return cl_dict


def get_cluster_assignments(all_clusterings, node_ids_multiple_cs_ids=None):
    if node_ids_multiple_cs_ids is None:
        node_ids_multiple_cs_ids = {}
    cluster_assignments = {}
    # Create dictionary with dictionaries for each cs-ID, containing for each clustering in which cluster it falls
    for label, clusters in all_clusterings.items():
        for cluster_index, ids in enumerate(clusters):
            for id_ in ids:
                if id_ in node_ids_multiple_cs_ids:
                    for cs_id in node_ids_multiple_cs_ids[id_]:
                        cluster_assignments.setdefault(cs_id, {})[label] = f"cl_{cluster_index}"
                else:
                    cluster_assignments.setdefault(id_, {})[label] = f"cl_{cluster_index}"

    df = pd.DataFrame.from_dict(cluster_assignments, orient="index")
    df = df.sort_index()
    return df


def get_footfall_clustering_from_nwk_str_deprecated(tree_nwk_str, n_clusters, cell_ids=None,
                                                    get_cell_ids_all_splits=False):
    print("\nInit footfall clustering-tree")
    cluster_tree = Cluster_Tree()
    cluster_tree.from_newick_string(nwk_str=tree_nwk_str)  # Works
    if get_cell_ids_all_splits:
        clusters, footfall_edges, ids_splits = get_footfall_clustering(cluster_tree, n_clusters, cell_ids=cell_ids,
                                                                       get_cell_ids_all_splits=get_cell_ids_all_splits)
        return clusters, footfall_edges, ids_splits
    else:
        clusters, footfall_edges = get_footfall_clustering(cluster_tree, n_clusters, cell_ids=cell_ids)
    return clusters, footfall_edges


def get_annotation_based_clustering_from_nwk_str(tree_nwk_str, annotation_dict, cell_id_to_node_id=None,
                                                 verbose=True, node_ids_to_clst=None, random_sampling=False,
                                                 tracking_path=None, max_moves=1e6, cutting_tol=1e-4,
                                                 prohibit_small_clsts=False):
    """

    :param tree_nwk_str: String read from a .nwk-file
    :param annotation_dict: Dictionary having key-value pairs with cell-ID to annotation
    :param cell_id_to_node_id: (Optional) Annotation is given for each cell, but multiple cells may have been mapped to
    the same node. This optional dictionary allows the user to give the mapping. If left to None, we assume that
    annotated IDs coincide with node-IDs.
    :param node_ids_to_clst: Node Ids of nodes that should be in reported clustering (otherwise leaf-nodes are returned)
    then we assume that we should cluster the leaves of the given newick string
    :param random_sampling: Boolean, default=False. If False, then we do a pure greedy hierarchical clustering that
    maximizes the Normalized Mutual Information at each clustering step. If True, we randomly sample clustering-steps
    weighted by their NMI. In addition, we allow for reverting a cut.
    :param tracking_path: Just for debugging purposes, we can give a folder where we will store some stats.
    :param verbose:
    :return:
    """
    cluster_tree = Cluster_Tree()
    cluster_tree.from_newick_string(nwk_str=tree_nwk_str)
    clusters, cut_edges, mut_info = get_annotation_based_clustering_random(cluster_tree,
                                                                           cell_id_to_node_id=cell_id_to_node_id,
                                                                           annotation_dict=annotation_dict,
                                                                           verbose=verbose,
                                                                           node_ids_to_clst=node_ids_to_clst,
                                                                           random_sampling=random_sampling,
                                                                           tracking_path=tracking_path,
                                                                           max_moves=max_moves,
                                                                           cutting_tol=cutting_tol,
                                                                           prohibit_small_clsts=prohibit_small_clsts)
    return clusters, cut_edges, mut_info


def get_min_pdists_clustering_from_nwk_str(tree_nwk_str, n_clusters, cell_ids=None, get_cell_ids_all_splits=False,
                                           node_id_to_n_cells=None, verbose=True, footfall=False):
    if verbose:
        mp_print("\nInit min-dist clustering-tree")
    cluster_tree = Cluster_Tree()
    cluster_tree.from_newick_string(nwk_str=tree_nwk_str)  # Works
    if node_id_to_n_cells is not None:
        cluster_tree.root.add_info_to_nodes(node_id_to_info=node_id_to_n_cells, info_key='n_cells')

    all_clusterings, footfall_edges = get_min_pdists_clustering(cluster_tree, n_clusters, cell_ids=cell_ids,
                                                                verbose=verbose, footfall=footfall)
    return all_clusterings, footfall_edges


def get_min_pdists_clustering_from_nwk_str_deprecated(tree_nwk_str, n_clusters, cell_ids=None,
                                                      get_cell_ids_all_splits=False,
                                                      node_id_to_n_cells=None, verbose=True):
    if verbose:
        print("\nInit min-dist clustering-tree")
    cluster_tree = Cluster_Tree()
    cluster_tree.from_newick_string(nwk_str=tree_nwk_str)  # Works
    if node_id_to_n_cells is not None:
        cluster_tree.root.add_info_to_nodes(node_id_to_info=node_id_to_n_cells, info_key='n_cells')
    if get_cell_ids_all_splits:
        clusters, footfall_edges, ids_splits = get_min_pdists_clustering_deprecated(cluster_tree, n_clusters,
                                                                                    cell_ids=cell_ids,
                                                                                    get_cell_ids_all_splits=get_cell_ids_all_splits)
        return clusters, footfall_edges, ids_splits
    else:
        clusters, footfall_edges = get_min_pdists_clustering_deprecated(cluster_tree, n_clusters, cell_ids=cell_ids,
                                                                        verbose=verbose)
    return clusters, footfall_edges


def get_max_diam_clustering_from_nwk_str(tree_nwk_str, max_diam_threshold, cell_ids=None):
    print("\nInit max diameter clustering-tree")
    cluster_tree = Cluster_Tree()
    cluster_tree.from_newick_string(nwk_str=tree_nwk_str)  # Works

    clusters = get_max_diam_clustering(cluster_tree, max_diam_threshold, cell_ids=cell_ids)

    return clusters


def get_footfall_clustering(cluster_tree, n_clusters, cell_ids=None, get_cell_ids_all_splits=False):
    if cell_ids is not None:
        cell_id_set = set(cell_ids)
    if get_cell_ids_all_splits:
        cell_ids_splits = {}

    tree_ensmbl = [cluster_tree]
    # Make sure each node knows how many ds leafs it has
    cluster_tree.n_leafs = cluster_tree.root.get_ds_and_parent_info()
    footfall_edges = []

    while len(tree_ensmbl) < n_clusters:
        # Loop over all edges to find max footfall one
        max_footfall_tree_ind = None
        max_footfall_node = None
        max_footfall_score = -1e9
        for ind_tree, tree in enumerate(tree_ensmbl):
            for vert_ind, node in tree.vert_ind_to_node.items():
                if node.parentNode is not None:
                    footfall_score = node.ds_leafs * (tree.n_leafs - node.ds_leafs) * node.tParent
                    if footfall_score > max_footfall_score:
                        max_footfall_tree_ind = ind_tree
                        max_footfall_node = node
                        max_footfall_score = footfall_score

        # Cut the tree into two pieces at the max footfall edge
        ds_node = max_footfall_node
        us_node = max_footfall_node.parentNode
        footfall_edges.append((ds_node.nodeId, us_node.nodeId))

        # Make one new tree:
        max_tree = tree_ensmbl[max_footfall_tree_ind]
        new_tree = Cluster_Tree()

        # Remove ds node from original tree
        us_node.childNodes = [child for child in us_node.childNodes if child.vert_ind != ds_node.vert_ind]

        # Make ds-node the root of the new tree
        new_tree.root = ds_node
        ds_node.parentNode = None

        # tree_1 = max_footfall_tree.copy(minimal_copy=True)
        # tree_2 = max_footfall_tree.copy(minimal_copy=True)
        # Reset the roots of these trees to the connecting nodes
        # tree_1.reset_root(new_root_ind=ds_node_vert_ind)
        # tree_2.reset_root(new_root_ind=us_node_vert_ind)
        # Cut off the redundant parts of the trees
        # tree_1.root.childNodes = [child for child in tree_1.root.childNodes if child.vert_ind != us_node_vert_ind]
        # tree_2.root.childNodes = [child for child in tree_2.root.childNodes if child.vert_ind != ds_node_vert_ind]
        # Update vert_ind_to_node
        max_tree.vert_ind_to_node, max_tree.nNodes = max_tree.root.renumber_verts(vertIndToNode={}, vert_count=0)
        new_tree.vert_ind_to_node, new_tree.nNodes = new_tree.root.renumber_verts(vertIndToNode={}, vert_count=0)
        max_tree.n_leafs = max_tree.root.get_ds_and_parent_info()
        new_tree.n_leafs = new_tree.root.get_ds_and_parent_info()

        tree_ensmbl.append(new_tree)
        if get_cell_ids_all_splits:
            cell_ids_splits[(ds_node.nodeId, us_node.nodeId)] = []
            for tree in [max_tree, new_tree]:
                leaf_ids_tree = []
                for vert_ind, node in tree.vert_ind_to_node.items():
                    if cell_ids is None:
                        if node.isLeaf:
                            leaf_ids_tree.append(node.nodeId)
                    else:
                        if node.nodeId in cell_id_set:
                            leaf_ids_tree.append(node.nodeId)
                cell_ids_splits[(ds_node.nodeId, us_node.nodeId)].append(leaf_ids_tree)

    clusters = []
    for ind_tree, tree in enumerate(tree_ensmbl):
        leaf_ids_tree = []
        for vert_ind, node in tree.vert_ind_to_node.items():
            if cell_ids is None:
                if node.isLeaf:
                    leaf_ids_tree.append(node.nodeId)
            else:
                if node.nodeId in cell_id_set:
                    leaf_ids_tree.append(node.nodeId)
        clusters.append(leaf_ids_tree)
    # Should produce a list of lists with the node-IDs of the various clusterings
    print("Clustering done")
    if get_cell_ids_all_splits:
        return clusters, footfall_edges, cell_ids_splits
    return clusters, footfall_edges


def get_min_pdists_clustering_deprecated(cluster_tree, n_clusters, cell_ids=None, get_cell_ids_all_splits=False,
                                         verbose=True):
    if cell_ids is not None:
        cell_id_set = set(cell_ids)
    if get_cell_ids_all_splits:
        cell_ids_splits = {}

    tree_ensmbl = [cluster_tree]
    # Make sure each node knows how many ds leafs it has
    cluster_tree.n_leafs = cluster_tree.get_min_pdists_info()
    # cluster_tree.n_leafs, cluster_tree.root.ds_dists = cluster_tree.root.get_ds_and_parent_info_plus_dists()
    # cluster_tree.root.us_dists = 0
    # cluster_tree.root.store_us_dists(total_leafs=cluster_tree.n_leafs)
    footfall_edges = []

    while len(tree_ensmbl) < n_clusters:
        # Loop over all edges to find max footfall one
        max_footfall_tree_ind = None
        max_footfall_node = None
        max_footfall_score = -1e9
        for ind_tree, tree in enumerate(tree_ensmbl):
            for vert_ind, node in tree.vert_ind_to_node.items():
                if node.parentNode is not None:
                    footfall_score = node.ds_leafs * (tree.n_leafs - node.ds_leafs) * node.tParent
                    footfall_score += node.ds_dists * (tree.n_leafs - node.ds_leafs)
                    footfall_score += node.us_dists * node.ds_leafs
                    if footfall_score > max_footfall_score:
                        max_footfall_tree_ind = ind_tree
                        max_footfall_node = node
                        max_footfall_score = footfall_score

        # Cut the tree into two pieces at the max footfall edge
        ds_node = max_footfall_node
        us_node = max_footfall_node.parentNode
        footfall_edges.append((ds_node.nodeId, us_node.nodeId))

        # Make one new tree:
        max_tree = tree_ensmbl[max_footfall_tree_ind]
        new_tree = Cluster_Tree()

        # Remove ds node from original tree
        us_node.childNodes = [child for child in us_node.childNodes if child.vert_ind != ds_node.vert_ind]

        # Make ds-node the root of the new tree
        new_tree.root = ds_node
        ds_node.parentNode = None

        # tree_1 = max_footfall_tree.copy(minimal_copy=True)
        # tree_2 = max_footfall_tree.copy(minimal_copy=True)
        # Reset the roots of these trees to the connecting nodes
        # tree_1.reset_root(new_root_ind=ds_node_vert_ind)
        # tree_2.reset_root(new_root_ind=us_node_vert_ind)
        # Cut off the redundant parts of the trees
        # tree_1.root.childNodes = [child for child in tree_1.root.childNodes if child.vert_ind != us_node_vert_ind]
        # tree_2.root.childNodes = [child for child in tree_2.root.childNodes if child.vert_ind != ds_node_vert_ind]
        # Update vert_ind_to_node
        max_tree.vert_ind_to_node, max_tree.nNodes = max_tree.root.renumber_verts(vertIndToNode={}, vert_count=0)
        new_tree.vert_ind_to_node, new_tree.nNodes = new_tree.root.renumber_verts(vertIndToNode={}, vert_count=0)
        max_tree.n_leafs = max_tree.get_min_pdists_info()
        new_tree.n_leafs = new_tree.get_min_pdists_info()

        tree_ensmbl.append(new_tree)
        if get_cell_ids_all_splits:
            cell_ids_splits[(ds_node.nodeId, us_node.nodeId)] = []
            for tree in [max_tree, new_tree]:
                leaf_ids_tree = []
                for vert_ind, node in tree.vert_ind_to_node.items():
                    if cell_ids is None:
                        if node.isLeaf:
                            leaf_ids_tree.append(node.nodeId)
                    else:
                        if node.nodeId in cell_id_set:
                            leaf_ids_tree.append(node.nodeId)
                cell_ids_splits[(ds_node.nodeId, us_node.nodeId)].append(leaf_ids_tree)
    # Should produce a list of lists with the node-IDs of the various clusterings
    clusters = []
    for ind_tree, tree in enumerate(tree_ensmbl):
        leaf_ids_tree = []
        for vert_ind, node in tree.vert_ind_to_node.items():
            if cell_ids is None:
                if node.isLeaf:
                    leaf_ids_tree.append(node.nodeId)
            else:
                if node.nodeId in cell_id_set:
                    leaf_ids_tree.append(node.nodeId)
        clusters.append(leaf_ids_tree)
    if verbose:
        print("Clustering done")

    if get_cell_ids_all_splits:
        return clusters, footfall_edges, cell_ids_splits
    return clusters, footfall_edges


def get_ent_change_merge_move(node, n_labels=None):
    """
    This assumes that node is the root of a subtree that was already split off from an original tree.
    :param node:
    :param n_labels:
    :return:
    """
    old_parent = node.old_parent_node
    # Find out which tree the old-parent currently belongs to
    old_tree_root = old_parent.belongs_to_tree.root
    # Find out the total counts in this old tree
    if n_labels is None:
        n_labels = len(node.ds_annot_counts)
    comb_clade_ent_cont = - entropy_non_norm(node.ds_n_annots + old_tree_root.ds_n_annots, num_annots=1)
    comb_annot_ent_cont = - entropy_non_norm(node.ds_annot_counts + old_tree_root.ds_annot_counts, num_annots=n_labels)
    node.revert_clade_ent_change = node.clade_ent_contrib_ds + old_tree_root.clade_ent_contrib_ds - comb_clade_ent_cont
    node.revert_annot_ent_change = node.annot_ent_contrib_ds + old_tree_root.annot_ent_contrib_ds - comb_annot_ent_cont


def get_new_mut_info(orig_ent_diff, annot_entropy, clade_entropy, clade_ent_change, annot_ent_change):
    new_ent_diff = (orig_ent_diff + clade_ent_change - annot_ent_change)
    new_normalization_sq = annot_entropy * (clade_entropy + clade_ent_change)
    if new_normalization_sq > 0.:
        new_mut_info = new_ent_diff / np.sqrt(new_normalization_sq)
    else:
        new_mut_info = 0.0
    return new_mut_info


def test_move_acceptance(sampling_beta, orig_mut_info, new_mut_info):
    sampling_prob = np.exp(-sampling_beta * (1 - new_mut_info / orig_mut_info))
    accepted = sampling_prob > random.random()
    return accepted


def get_annotation_based_clustering_random(cluster_tree, annotation_dict, cell_id_to_node_id=None, verbose=True,
                                           node_ids_to_clst=None, random_sampling=False, seed=1231,
                                           tracking_path=None, max_moves=1e6, cutting_tol=1e-4,
                                           prohibit_small_clsts=False):
    """
    Greedily cuts tree into clades such that mutual information with some annotation is maximized.
    *NOTE:* We allow for partial annotation, such that some cells have an annotation, while others do not. The use case
    for this is that people may have partial annotation-information, and want to use the tree to annotate the rest. In
    that case, these un-annotated cells will not be counted in the mutual annotation score, but they will still be
    clustered in the final clustering.
    :param cluster_tree: Cluster_tree object
    :param node_ids_to_clst: Node Ids of nodes that should be in reported clustering (otherwise leaf-nodes are reurned)
    :param annotation_dict: Dictionary having key-value pairs with cell-ID to annotation.
    :param cell_id_to_node_id: (Optional) Annotation is given for each cell, but multiple cells may have been mapped to
    the same node. This optional dictionary allows the user to give the mapping. If left to None, we assume that
    annotated IDs coincide with node-IDs.
    :param random_sampling: Boolean, default=False. If False, then we do a pure greedy hierarchical clustering that
    maximizes the Normalized Mutual Information at each clustering step. If True, we randomly sample clustering-steps
    weighted by their NMI. In addition, we allow for reverting a cut.
    :param verbose:
    :param tracking_folder: Just a path where I can store some stats for performance tracking.
    :return:
    """
    very_verbose = False

    if tracking_path is not None:
        # tracking_path, _ = os.path.splitext(tracking_path)
        # tracking_path += '_random.tsv' if random_sampling else '.tsv'
        stats = []
    np.random.seed(seed)
    random.seed(np.random.randint(1e6))
    n_moves = -1
    if node_ids_to_clst is not None:
        node_ids_to_clst_set = set(node_ids_to_clst)
    else:
        node_ids_to_clst_set = None

    if cluster_tree.vert_ind_to_node is None:
        cluster_tree.vert_ind_to_node, cluster_tree.nNodes = cluster_tree.root.renumber_verts(vertIndToNode={},
                                                                                              vert_count=0)

    if random_sampling:
        # We will do random-sampling at each step, such that an option that is a factor <optimality_decrease> below the
        # best option is picked with <prob_decrease> lower probability than the maximal option
        # We start with optimality_decrease = .95, and may lower it when we move on.
        # I think we can keep prob_decrease fixed at .5
        prob_decrease = .5
        optimality_decrease = .95
        # We want a formula like exp(-beta * (1 - (NMI/NMI_max)^2)), so we want
        # exp(-beta * (1 - optimality_decrease^2)) = prob_decrease
        # beta = -log(prob_decrease) / (1 - optimality_decrease ^ 2)
        # So, if we set:
        sampling_beta = - np.log(prob_decrease) / (1 - optimality_decrease)
        # n_annealing = 100
        n_annealing = int(min(max(10, cluster_tree.nNodes / 1000), 200))
        optimality_decrease_lb = .999
        annealing_factor = 5
        # Then the function for the (unnormalized) probability becomes
        # prob(NMI) = exp(-beta * (1 - (NMI/NMI_max) * optimality_decrease))

        # Add these random moves to the max_moves. With the current settings these are 3 annealing phases
        max_moves += 3 * n_annealing

        # We first do a burn-in phase though, where we just pick greedily
        greedy = True

    # First, put the annotation-labels on the tree
    # cluster_tree.root.add_info_to_nodes(node_id_to_info=annotation_dict, info_key='annotation')
    # Get some statistics on the annotation-labels
    annot_labels = np.unique(list(annotation_dict.values()))
    n_labels = len(annot_labels)
    if n_labels in [1, len(annotation_dict)]:
        if n_labels == 1:
            mp_print("Only one annotation-label, nothing to cluster.")
        elif n_labels == len(annotation_dict):
            mp_print("All annotation-labels are different, no point in trying to match this with clustering.")
        tree_ensmbl = [cluster_tree]
        clusters = get_clustering_lists(tree_ensmbl, node_ids_to_clst_set)
        return clusters, [], 1.0

    label_to_ind = {annot: ind for ind, annot in enumerate(annot_labels)}

    # Store the downstream annotation-counts on each node
    node_id_to_node = {}
    for _, node in cluster_tree.vert_ind_to_node.items():
        node_id_to_node[node.nodeId] = node
        node.own_annot_counts = np.zeros(n_labels, dtype=int)
        node.own_n_annots = 0

    # Store on each node to which tree it belongs
    cluster_tree.root.add_info_to_nodes(info_key='belongs_to_tree', const_val=cluster_tree)
    label_counts = np.zeros(n_labels, dtype=int)

    # TODO: Remove when sklearn test is no longer necessary
    # labels_true = np.zeros(len(annotation_dict), dtype=int)
    # labels_pred = np.zeros(len(annotation_dict), dtype=int)
    # node_id_to_cell_ids = defaultdict(list)
    # cell_id_to_ind = {cell_id: ind for ind, cell_id in enumerate(annotation_dict)}
    # for cell_id, node_id in cell_id_to_node_id.items():
    #     node_id_to_cell_ids[node_id].append(cell_id)

    for cell_id, annot in annotation_dict.items():
        if cell_id_to_node_id is not None:
            node_id = cell_id_to_node_id[cell_id]
        else:
            node_id = cell_id
        if node_id not in node_id_to_node:
            mp_print("WARNING: Node_id {} is not present in the tree. "
                     "Cannot take its annotation into account in clustering.".format(node_id), WARNING=True)
            continue
        else:
            node = node_id_to_node[node_id]

        node.own_annot_counts[label_to_ind[annot]] += 1
        node.own_n_annots += 1
        label_counts[label_to_ind[annot]] += 1

    n_cells = np.sum(label_counts)

    if prohibit_small_clsts:
        # Here, we calculate based on the label-counts some cutoffs for how small clusters can be before we cut them
        # Ad-hoc rule: Clusters should be at least .1 of the average category, or at least .01 of the entire dataset.
        avg_cat = n_cells / n_labels
        clst_size_lb = min(avg_cat / 10, n_cells / 200)
        # Clusters cannot be cut if they contain only very small fractions of all categories.
        clst_cat_capture_lb_factor = .025
        clst_cat_capture_lbs = label_counts * clst_cat_capture_lb_factor
        # However, clusters can always be cut if they contain a certain fraction of all cells of a certain category.
        clst_cat_capture_ub_factor = .3
        clst_cat_capture_ubs = label_counts * clst_cat_capture_ub_factor

    # Here, we can already calculate the entropy in the annotation. This is independent of the clustering
    annot_entropy = entropy_non_norm(label_counts / n_cells)
    # Initial clade entropy is 0, everyone is in the same cluster
    clade_entropy = 0
    # Initial joint entropy is the same as the annot_entropy, since all cells are in same cluster
    joint_entropy = annot_entropy

    # Multiply these three entropies by total number of annotations. This makes calculations easier later
    annot_entropy *= n_cells
    clade_entropy *= n_cells
    joint_entropy *= n_cells

    # Initialize the necessary information on the nodes of the full tree
    cluster_tree.root.get_ds_annot_info()
    if prohibit_small_clsts:
        cluster_tree.root.set_include_as_cand(total_n_annots=cluster_tree.root.ds_n_annots,
                                              total_annot_counts=cluster_tree.root.ds_annot_counts,
                                              clst_size_lb=clst_size_lb,
                                              clst_cat_capture_lbs=clst_cat_capture_lbs,
                                              clst_cat_capture_ubs=clst_cat_capture_ubs)
    # Store on each node their contribution to the clade and annot-entropy
    cluster_tree.root.get_ent_contribs(total_n=cluster_tree.root.ds_n_annots,
                                       total_annot_counts=cluster_tree.root.ds_annot_counts,
                                       num_annots=n_labels)
    # Store on each node how clade- and joint-entropy would change when edge to this node was cut
    cluster_tree.get_ent_changes()
    tree_ensmbl = [cluster_tree]

    # TODO: Maybe remove
    # Per tree that we create (i.e., cluster) we keep track of the total number of cells for each label
    # total_annot_counts = [label_counts]

    # TODO: Check if, for each tree, we can keep track of the maximal mutual information increase,
    #  and which node we'd have to cut off for that maximal score
    # max_scores = [None]
    # max_nodes = [None]
    # clusters = [None]
    cut_edges = []

    # Can't cut more branches between cells than there are annotated nodes
    n_clusters = n_cells

    print_i = 2
    n_random_moves = 0
    while (len(tree_ensmbl) < n_clusters) and (n_moves < max_moves):
        n_moves += 1
        n_trees = len(tree_ensmbl)
        if verbose and (n_moves == print_i):
            mp_print("Annotation-based clustering has done {} moves, "
                     "currently {} clusters, "
                     "and Normalized Mutual Information of {}.".format(n_moves, n_trees, max_new_mut_info))
            print_i *= 2
        # Loop over all edges to find which increases mutual information most
        orig_ent_diff = annot_entropy + clade_entropy - joint_entropy
        orig_normalization = np.sqrt(annot_entropy * clade_entropy)
        if orig_normalization == 0:
            orig_mut_info = 0.
        else:
            orig_mut_info = orig_ent_diff / orig_normalization
        # max_tree_ind = None
        max_node = None
        max_new_mut_info = -1e9
        if random_sampling and (not greedy):
            if (n_random_moves % n_annealing == 0) and n_random_moves != 0:
                optimality_decrease = 1 - ((1 - optimality_decrease) / annealing_factor)
                sampling_beta = - np.log(prob_decrease) / (1 - optimality_decrease)
                if optimality_decrease > optimality_decrease_lb:
                    random_sampling = False
                    continue
            n_random_moves += 1
            node_list_split = []
            node_list_merge = []
            sampling_probs_split = []
            sampling_probs_merge = []
            for ind_tree, tree in enumerate(tree_ensmbl):
                for vert_ind, node in tree.vert_ind_to_node.items():
                    # We set the sampling prob. of a node equal to how much it changes the clade-entropy.
                    # We make sure to never sample a pure-cluster-split (i.e. that do not change clade-annot-ent)
                    if node.parentNode is None:
                        if node.old_parent_node is None:
                            continue
                        else:
                            # In this case, this is a "merge"-move
                            get_ent_change_merge_move(node, n_labels=n_labels)
                            node_list_merge.append(node)
                            sampling_probs_merge.append(node.revert_clade_ent_change)
                    else:
                        # In this case, this is a "split"-move
                        if not node.include_as_cand:
                            # We can skip this one because the created clusters would be too small
                            continue
                        if node.annot_ent_change == 0:
                            # We can skip this node because it splits up all annotated labels from a cluster.
                            continue
                        node_list_split.append(node)
                        sampling_probs_split.append(node.clade_ent_change)
            # Now sample nodes prop. to  sampling_probs until we find a successful one
            sampling_probs_split = np.abs(np.array(sampling_probs_split))
            sampling_probs_merge = np.abs(np.array(sampling_probs_merge))
            n_probs_split = len(sampling_probs_split)
            n_probs_merge = len(sampling_probs_merge)
            n_probs_total = n_probs_merge + n_probs_split
            sampling_probs_split /= sampling_probs_split.sum()
            sampling_probs_merge /= sampling_probs_merge.sum()
            n_counter = 0
            while (max_node is None) and (n_counter < 2 * n_probs_total):
                n_counter += 1
                # First decide whether you pick a merge or a split
                pick_split = random.random() > 0.5
                if (pick_split and (n_probs_split > 0)) or (n_probs_merge == 0):
                    sampled_ind = np.random.choice(n_probs_split, p=sampling_probs_split)
                    node = node_list_split[sampled_ind]
                else:  # pick_merge
                    sampled_ind = np.random.choice(n_probs_merge, p=sampling_probs_merge)
                    node = node_list_merge[sampled_ind]

                if node.parentNode is None:
                    new_mut_info = get_new_mut_info(orig_ent_diff, annot_entropy, clade_entropy,
                                                    node.revert_clade_ent_change,
                                                    node.revert_annot_ent_change)
                else:
                    new_mut_info = get_new_mut_info(orig_ent_diff, annot_entropy, clade_entropy,
                                                    node.clade_ent_change,
                                                    node.annot_ent_change)
                if (new_mut_info > orig_mut_info) or test_move_acceptance(sampling_beta, orig_mut_info, new_mut_info):
                    max_node = node
                    max_tree = node.belongs_to_tree
                    max_new_mut_info = new_mut_info
            if max_node is None:
                # Wasn't able to find any successful move. Stopping random_sampling and going for a greedy finish
                random_sampling = False
                continue

        else:
            for ind_tree, tree in enumerate(tree_ensmbl):
                for vert_ind, node in tree.vert_ind_to_node.items():
                    if node.parentNode is None:
                        if node.old_parent_node is None:
                            continue  # This is the root of original tree. Can't merge nor split
                        else:
                            # This is a sub-tree root, ask it if wants to go back to its daddy
                            # This function stores revert_clade_ent_change and revert_annot_ent_change on the node
                            get_ent_change_merge_move(node, n_labels=n_labels)
                            new_mut_info = get_new_mut_info(orig_ent_diff, annot_entropy, clade_entropy,
                                                            node.revert_clade_ent_change,
                                                            node.revert_annot_ent_change)
                    else:
                        if not node.include_as_cand:
                            # We can skip this one because the created clusters would be too small
                            continue
                        if node.annot_ent_change == 0:
                            # if (node.ds_n_annots == 0) or (tree.root.ds_n_annots == node.ds_n_annots):
                            # This node, we can skip because it splits up all annotated labels from a cluster.
                            continue
                        new_mut_info = get_new_mut_info(orig_ent_diff, annot_entropy, clade_entropy,
                                                        node.clade_ent_change,
                                                        node.annot_ent_change)
                    # if we're grouping clusters, we always accept moves that improve the NMI, if we're cutting
                    # we only accept if they're improving to a reasonable extent
                    tol = -1e-5 if (node.parentNode is None) else cutting_tol
                    if new_mut_info > max(max_new_mut_info, orig_mut_info + tol):
                        max_node = node
                        max_tree = tree
                        max_new_mut_info = new_mut_info

        # Determine which tree has the maximum score
        if not random_sampling:
            # if we're grouping clusters, we always accept moves that improve the NMI, if we're cutting we only accept
            # if they're improving to a reasonable extent
            quitting = False
            if max_node is None:
                quitting = True
            else:
                tol = -1e-5 if (max_node.parentNode is None) else cutting_tol
                if max_new_mut_info < orig_mut_info + tol:
                    quitting = True
            if quitting:
                mp_print("\nAfter making {} clusters, norm. mutual information doesn't significantly increase. "
                         "Stopping here.".format(n_trees))
                break

        if random_sampling and greedy:
            if (max_new_mut_info / (orig_mut_info + 1e-9)) < 1.01:
                mp_print("Progress is stalling, stopping the greedy burn-in phase, starting the random phase")
                greedy = False
                continue

        # Cut the tree into two pieces at the edge that maximizes the norm. mut. info.
        ds_node = max_node
        if max_node.parentNode is None:
            if max_node.old_parent_node is None:
                mp_print("Something went terribly wrong. Node doesn't have parent, and no former parent, "
                         "and was still selected in clustering. Check this.", ERROR=True)
                exit()

            # In this case, we do not create a new tree, but rather merge two trees
            us_node = max_node.old_parent_node
            cut_edges = [edge for edge in cut_edges if edge != (ds_node.nodeId, us_node.nodeId)]
            # Update current entropies
            clade_entropy += max_node.revert_clade_ent_change
            joint_entropy += max_node.revert_annot_ent_change

            # Delete the tree
            tree_ensmbl = [tree_inst for tree_inst in tree_ensmbl if tree_inst.root.nodeId != max_tree.root.nodeId]
            # del tree_ensmbl[max_tree_ind]

            # Add node to original tree
            us_node.childNodes.append(ds_node)
            # ds-node should no longer be the root
            ds_node.parentNode = us_node
            ds_node.old_parent_node = None
            ds_node.isRoot = False

            # Update vert_ind_to_node for the now merged tree
            new_tree = us_node.belongs_to_tree
            new_tree.vert_ind_to_node, new_tree.nNodes = new_tree.root.renumber_verts(vertIndToNode={}, vert_count=0)

            # Initialize the necessary information on the nodes of the full tree
            # We don't have to do: new_tree.root.get_ds_annot_info(), because new_tree still has correct ds-info
            # For cluster_tree, we just subtract the ds_info for everything upstream of new root
            us_node.subtract_or_add_ds_info_us(ds_node, add=True)
            if prohibit_small_clsts:
                new_tree.root.set_include_as_cand(total_n_annots=new_tree.root.ds_n_annots,
                                                  total_annot_counts=new_tree.root.ds_annot_counts,
                                                  clst_size_lb=clst_size_lb,
                                                  clst_cat_capture_lbs=clst_cat_capture_lbs,
                                                  clst_cat_capture_ubs=clst_cat_capture_ubs)
            # Store on each node their contribution to the clade and annot-entropy.
            new_tree.root.get_ent_contribs(total_n=new_tree.root.ds_n_annots,
                                           total_annot_counts=new_tree.root.ds_annot_counts, num_annots=n_labels)
            # Store on each node how clade- and joint-entropy would change when edge to this node was cut
            new_tree.get_ent_changes()
            # Add for all downstream nodes to which tree it belongs
            ds_node.add_info_to_nodes(info_key='belongs_to_tree', const_val=new_tree)

            if very_verbose:
                mp_print("\nMerging two clusters with {} and {} labeled-cells. "
                         "Norm. mut. info. now: {}".format(new_tree.root.ds_n_annots - ds_node.ds_n_annots,
                                                           ds_node.ds_n_annots,
                                                           max_new_mut_info),
                         DEBUG=True)
                mp_print("Cluster 1, annot_counts: {}".format(','.join(map(str,
                                                                           new_tree.root.ds_annot_counts - ds_node.ds_annot_counts))),
                         DEBUG=True)
                mp_print("Cluster 2, annot_counts: {}".format(','.join(map(str, ds_node.ds_annot_counts))),
                         DEBUG=True)
        else:
            us_node = max_node.parentNode
            cut_edges.append((ds_node.nodeId, us_node.nodeId))

            # Update current entropies
            clade_entropy += max_node.clade_ent_change
            joint_entropy += max_node.annot_ent_change

            # Make one new tree:
            # max_tree = tree_ensmbl[max_tree_ind]
            new_tree = Cluster_Tree()

            # Remove ds node from original tree
            us_node.childNodes = [child for child in us_node.childNodes if child.vert_ind != ds_node.vert_ind]

            # Make ds-node the root of the new tree
            new_tree.root = ds_node
            ds_node.parentNode = None
            ds_node.old_parent_node = us_node

            # Update vert_ind_to_node
            # TODO: Check at some point whether this can be done efficient by re-using information
            max_tree.vert_ind_to_node, max_tree.nNodes = max_tree.root.renumber_verts(vertIndToNode={}, vert_count=0)
            new_tree.vert_ind_to_node, new_tree.nNodes = new_tree.root.renumber_verts(vertIndToNode={}, vert_count=0)
            # Initialize the necessary information on the nodes of the full tree
            # We don't have to do: new_tree.root.get_ds_annot_info(), because new_tree still has correct ds-info
            # For cluster_tree, we just subtract the ds_info for everything upstream of new root
            us_node.subtract_or_add_ds_info_us(ds_node, add=False)

            if prohibit_small_clsts:
                max_tree.root.set_include_as_cand(total_n_annots=max_tree.root.ds_n_annots,
                                                  total_annot_counts=max_tree.root.ds_annot_counts,
                                                  clst_size_lb=clst_size_lb,
                                                  clst_cat_capture_lbs=clst_cat_capture_lbs,
                                                  clst_cat_capture_ubs=clst_cat_capture_ubs)
                new_tree.root.set_include_as_cand(total_n_annots=new_tree.root.ds_n_annots,
                                                  total_annot_counts=new_tree.root.ds_annot_counts,
                                                  clst_size_lb=clst_size_lb,
                                                  clst_cat_capture_lbs=clst_cat_capture_lbs,
                                                  clst_cat_capture_ubs=clst_cat_capture_ubs)

            # Store on each node their contribution to the clade and annot-entropy.
            # Ideally, this is done only for us-ent-contribs for nodes downstream of change, and for ds-ent-contribs for
            # nodes upstream of change. But it's a bit of work for little gain (only a factor 2, I think)
            max_tree.root.get_ent_contribs(total_n=max_tree.root.ds_n_annots,
                                           total_annot_counts=max_tree.root.ds_annot_counts, num_annots=n_labels)
            new_tree.root.get_ent_contribs(total_n=new_tree.root.ds_n_annots,
                                           total_annot_counts=new_tree.root.ds_annot_counts, num_annots=n_labels)

            # Store on each node how clade- and joint-entropy would change when edge to this node was cut
            max_tree.get_ent_changes()
            new_tree.get_ent_changes()

            new_tree.root.add_info_to_nodes(info_key='belongs_to_tree', const_val=new_tree)

            # Add tree to ensemble, and make space for the new tree in the lists
            tree_ensmbl.append(new_tree)

            if very_verbose:
                mp_print("\nCreating two clusters with {} and {} labeled-cells. "
                         "Norm. mut. info. now: {}".format(max_tree.root.ds_n_annots, new_tree.root.ds_n_annots,
                                                           max_new_mut_info),
                         DEBUG=True)
                mp_print("Cluster 1, annot_counts: {}".format(','.join(map(str, max_tree.root.ds_annot_counts))),
                         DEBUG=True)
                mp_print("Cluster 2, annot_counts: {}".format(','.join(map(str, new_tree.root.ds_annot_counts))),
                         DEBUG=True)

        if tracking_path is not None:
            if not random_sampling:
                greedy = optimality_decrease = None
            stats.append({
                "n_moves": n_moves,
                "norm_mut_info": max_new_mut_info,
                "n_clusters": len(tree_ensmbl),
                "random_sampling": random_sampling,
                "greedy": greedy,
                "optimality_decrease": optimality_decrease
            })

        # TODO: Write independent norm. mut. info. check and eventually comment that out.
        DO_TEST = False
        if DO_TEST:
            contingency = np.zeros((n_labels, len(tree_ensmbl)), dtype=int)
            for ind_tree, tree in enumerate(tree_ensmbl):
                contingency[:, ind_tree] = tree.root.ds_annot_counts

            labels_true = []
            labels_pred = []
            for i, row in enumerate(contingency):
                for j, count in enumerate(row):
                    labels_true.extend([i] * count)
                    labels_pred.extend([j] * count)
            from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, rand_score, \
                adjusted_rand_score

            nmi_test = normalized_mutual_info_score(np.array(labels_true), np.array(labels_pred),
                                                    average_method='geometric')
            adjusted_mut_info = adjusted_mutual_info_score(np.array(labels_true), np.array(labels_pred),
                                                           average_method='geometric')
            rand_index = rand_score(np.array(labels_true), np.array(labels_pred))
            adjusted_rand_index = adjusted_rand_score(np.array(labels_true), np.array(labels_pred))
            print(nmi_test, max_new_mut_info)
            if abs(max_new_mut_info - nmi_test) > 1e-9:
                mp_print("Predicted mutual information deviates from independently calculated mut.info."
                         "Check this!", ERROR=True)
                exit()

            if tracking_path is not None:
                stats[-1].update({"adjusted_mut_info": adjusted_mut_info,
                                  "rand_index": rand_index,
                                  "adjusted_rand_index": adjusted_rand_index})

        # if DO_TEST:
        #     joint_entropy_test = 0.0
        #     clade_sizes_test = []
        #     joint_counts = []
        #     annot_counts_test = np.zeros(n_labels)
        #     for ind_tree, tree in enumerate(tree_ensmbl):
        #         clade_sizes_test.append(tree.root.ds_n_annots)
        #         joint_entropy_test += entropy_non_norm(tree.root.ds_annot_counts / n_cells)
        #         joint_counts.append(tree.root.ds_annot_counts)
        #         annot_counts_test += tree.root.ds_annot_counts
        #
        #     joint_counts = np.hstack(joint_counts)
        #     # joint_entropy_test2 = entropy_non_norm(joint_counts / n_cells)
        #     clade_entropy_test = entropy_non_norm(np.array(clade_sizes_test) / n_cells)
        #     annot_entropy_test = entropy_non_norm(annot_counts_test / n_cells)
        #
        #     norm_mut_inf_test = (clade_entropy_test + annot_entropy_test - joint_entropy_test) / np.sqrt(
        #         clade_entropy_test * annot_entropy_test)
        #     # print("label_counts is equal to annot_counts_test: {}".format(annot_counts_test == label_counts))
        #     # print("n_cells is equal to sum(clade_sizes_test): {}".format(n_cells == np.sum(clade_sizes_test)))
        #
        #     mp_print("Predicted mutual information is {}, and independent test gives {}".format(max_new_mut_info,
        #                                                                                         norm_mut_inf_test))
        #     if abs(max_new_mut_info - norm_mut_inf_test) > 1e-9:
        #         mp_print("Predicted mutual information deviates from independently calculated mut.info."
        #                  "Check this!", ERROR=True)
        #         exit()

    mp_print("Final annotation-based clustering did {} moves, "
             "created {} clusters, "
             "with a Normalized Mutual Information of {}.".format(n_moves + 1, n_trees, orig_mut_info))

    clusters = get_clustering_lists(tree_ensmbl, node_ids_to_clst_set)

    if tracking_path is not None:
        df = pd.DataFrame(stats)
        df.to_csv(tracking_path, index=False, sep='\t')

    # Store current clustering in dictionary of clusterings
    if verbose:
        mp_print("Clustering done")
    return clusters, cut_edges, orig_mut_info


def get_clustering_lists(tree_ensmbl, node_ids_to_clst_set):
    # Should produce a list of lists with the node-IDs of the various clusters
    clusters = [None] * len(tree_ensmbl)
    for ind_tree, tree in enumerate(tree_ensmbl):
        if clusters[ind_tree] is not None:
            continue
        leaf_ids_tree = []
        for vert_ind, node in tree.vert_ind_to_node.items():
            if node_ids_to_clst_set is None:
                if node.isLeaf:
                    leaf_ids_tree.append(node.nodeId)
            else:
                if node.nodeId in node_ids_to_clst_set:
                    leaf_ids_tree.append(node.nodeId)
        clusters[ind_tree] = leaf_ids_tree
    return clusters


def get_min_pdists_clustering(cluster_tree, n_clusters, cell_ids=None, get_cell_ids_all_splits=False, verbose=True,
                              footfall=False):
    # if get_cell_ids_all_splits:
    #     cell_ids_splits = {}
    if cell_ids is not None:
        cell_ids_set = set(cell_ids)
    if cluster_tree.vert_ind_to_node is None:
        cluster_tree.vert_ind_to_node, cluster_tree.nNodes = cluster_tree.root.renumber_verts(vertIndToNode={},
                                                                                              vert_count=0)

    all_clusterings = {}
    tree_ensmbl = [cluster_tree]
    # Make sure each node knows how many ds leafs it has
    cluster_tree.n_leafs = cluster_tree.get_min_pdists_info()
    max_scores = [None]
    max_nodes = [None]
    clusters = [None]
    footfall_edges = []

    # Can't cut more branches between cells than there are cell-associated nodes
    n_clusters = min(n_clusters, cluster_tree.n_cell_nodes)

    print_i = 2
    while len(tree_ensmbl) < n_clusters:
        n_trees = len(tree_ensmbl)
        if verbose and (n_trees == print_i):
            mp_print(
                "Clustering has created {} subtrees, {} branches still to cut.".format(n_trees, n_clusters - n_trees))
            print_i *= 2
        # Loop over all edges to find max footfall one
        for ind_tree, tree in enumerate(tree_ensmbl):
            if max_scores[ind_tree] is not None:
                continue
            if tree.n_cell_nodes == 1:
                max_scores[ind_tree] = 0.0
                continue
            max_footfall_node = None
            max_footfall_score = -1e9
            for vert_ind, node in tree.vert_ind_to_node.items():
                if node.parentNode is not None:
                    footfall_score = node.ds_leafs * (tree.n_leafs - node.ds_leafs) * node.tParent
                    if not footfall:
                        footfall_score += node.ds_dists * (tree.n_leafs - node.ds_leafs)
                        footfall_score += node.us_dists * node.ds_leafs
                    if footfall_score > max_footfall_score:
                        max_footfall_node = node
                        max_footfall_score = footfall_score
            max_scores[ind_tree] = max_footfall_score
            max_nodes[ind_tree] = max_footfall_node

        # Determine which tree has the maximum score
        max_tree_ind = np.argmax(max_scores)
        if max_scores[max_tree_ind] < 1e-9:
            # This should never happen
            print("Cannot find more than {} clusters. Cannot find edge that reduces the pairwise distances.")
            break

        max_node = max_nodes[max_tree_ind]

        # Cut the tree into two pieces at the max footfall edge
        ds_node = max_node
        us_node = max_node.parentNode
        footfall_edges.append((ds_node.nodeId, us_node.nodeId))

        # Make one new tree:
        max_tree = tree_ensmbl[max_tree_ind]
        new_tree = Cluster_Tree()

        # Remove ds node from original tree
        us_node.childNodes = [child for child in us_node.childNodes if child.vert_ind != ds_node.vert_ind]

        # Make ds-node the root of the new tree
        new_tree.root = ds_node
        ds_node.parentNode = None

        # Update vert_ind_to_node
        # TODO: Check at some point whether this can be done efficient by re-using information
        max_tree.vert_ind_to_node, max_tree.nNodes = max_tree.root.renumber_verts(vertIndToNode={}, vert_count=0)
        new_tree.vert_ind_to_node, new_tree.nNodes = new_tree.root.renumber_verts(vertIndToNode={}, vert_count=0)
        max_tree.n_leafs = max_tree.get_min_pdists_info()
        new_tree.n_leafs = new_tree.get_min_pdists_info()

        # Add tree to ensemble, and make space for the new tree in the lists
        tree_ensmbl.append(new_tree)
        max_scores[max_tree_ind] = None  # This will make sure the score is recalculated
        max_nodes[max_tree_ind] = None
        clusters[max_tree_ind] = None
        max_scores.append(None)  # This will give a place for the new tree to store
        max_nodes.append(None)
        clusters.append(None)

        # Should produce a list of lists with the node-IDs of the various clusters
        for ind_tree, tree in enumerate(tree_ensmbl):
            if clusters[ind_tree] is not None:
                continue
            leaf_ids_tree = []
            for vert_ind, node in tree.vert_ind_to_node.items():
                if cell_ids is None:
                    if node.isLeaf:
                        leaf_ids_tree.append(node.nodeId)
                else:
                    if node.nodeId in cell_ids_set:
                        leaf_ids_tree.append(node.nodeId)
            clusters[ind_tree] = leaf_ids_tree
        # Store current clustering in dictionary of clusterings
        clustering_name = 'annot_bnsi_cluster_n{}'.format(len(tree_ensmbl))
        all_clusterings[clustering_name] = clusters.copy()

    if verbose:
        print("Clustering done")
    return all_clusterings, footfall_edges


def get_max_diam_clustering(cluster_tree, max_diam_threshold, cell_ids=None):
    # print("\nInit tree")
    # cluster_tree = Cluster_Tree()
    # cluster_tree.from_newick_file(nwk_file=tree_nwk_file)  # Works

    # get post traversal order
    vert_ind_in_postOrder_v3 = cluster_tree.root.getPostOrder_only_internalNodes()

    print("Do clustering with maxdiam: {}".format(max_diam_threshold))
    clusters = []
    # traverse tree in post_order_traversal
    for vert_ind in vert_ind_in_postOrder_v3:
        # print("traversing node: {}".format(vert_ind))
        node = cluster_tree.vert_ind_to_node[vert_ind]

        # if for some reason the node has already been visited...
        if node.is_deleted:
            # print("node {} is deleted".format(node.vert_ind))
            continue

        # Check constraint
        # For that calculate first all pairwise max distances to leaf through parent (node)
        # print("number of children: {}".format(len(node.childNodes)))
        # print("find all combinations of children")

        comb = combinations(range(len(node.childNodes)), 2)
        max_pairwise_distances = {}
        # print("calc max pairwise distances of children")
        for i1, i2 in list(comb):  # store in dict (i1,i2) : dist
            u1 = node.childNodes[i1]
            u2 = node.childNodes[i2]
            if not u1.is_deleted and not u2.is_deleted:
                # TODO : store only if  node is not deleted: done
                dist = u1.len_to_most_distant_leaf + u1.tParent + u2.len_to_most_distant_leaf + u2.tParent
                max_pairwise_distances[(i1, i2)] = dist
        # print("sort max pairwise distances of children")
        # sort distances:
        pair_wise_comb = list(max_pairwise_distances.keys())
        distances = np.array(list(max_pairwise_distances.values()))
        distances_sortidx = np.argsort(-distances)  # largest will be at first position

        # check if all max pairwise distances fullfill the constraint
        # for that I can check if the largest max pairwise distance fullfills the constraint
        # print("len(pair_wise_comb): {}".format(len(pair_wise_comb)))
        # print("start while loop")
        while len(pair_wise_comb) > 0:
            # print("len(pair_wise_comb): {}".format(len(pair_wise_comb)))
            # print("check if contraint is met for the largest distances between two leafs")
            if distances[distances_sortidx][0] < max_diam_threshold:
                # print("contraint is met, we can continue")
                longest_distances_to_leafs_from_node = [x.len_to_most_distant_leaf + x.tParent for x in node.childNodes
                                                        if not x.is_deleted]
                node.len_to_most_distant_leaf = np.max(longest_distances_to_leafs_from_node)
                break
            else:
                # print("contraint is NOT met, we have to cut the longest branch")
                # cut longest branch

                i1, i2 = np.array(pair_wise_comb)[distances_sortidx][0]
                u1 = node.childNodes[i1]
                u2 = node.childNodes[i2]
                # cut longest branch and get cluster
                if u1.len_to_most_distant_leaf + u1.tParent >= u2.len_to_most_distant_leaf + u2.tParent:
                    # print("longest branch corresponds to child: {}".format(i1))
                    cluster = cluster_tree.cut(subtree_root=u1, cell_ids=cell_ids)
                    # todo set u1 to deleted
                    deleted_node_idx = i1
                    clusters.append(cluster)
                    node.len_to_most_distant_leaf = u2.len_to_most_distant_leaf + u2.tParent
                else:
                    # print("longest branch corresponds to child: {}".format(i2))
                    cluster = cluster_tree.cut(subtree_root=u2, cell_ids=cell_ids)
                    deleted_node_idx = i2
                    clusters.append(cluster)
                    node.len_to_most_distant_leaf = u1.len_to_most_distant_leaf + u1.tParent

                # print("Now remove all that corresponds to that longest branch and redo while loop")
                # then delete the node and subtree of the longest branch, and check again for the distances

                valid_pairs = [True if x[0] != deleted_node_idx and x[1] != deleted_node_idx else False for x in
                               pair_wise_comb]

                pair_wise_comb = [b for a, b in zip(valid_pairs, pair_wise_comb) if a]
                distances = distances[valid_pairs]
                distances_sortidx = np.argsort(
                    -distances)  # i think i have to sort new TODO check if this is right, make more efficient

    # at the end get the rest of the cluster
    cluster = cluster_tree.cut(subtree_root=node, cell_ids=cell_ids)
    clusters.append(cluster)
    print("Clustering done")
    return clusters


if __name__ == "__main__":

    parser = ArgumentParser(
        description='Starts from a reconstructed tree output by Bonsai and creates a data-object necessary for further '
                    'visualization and usage in the Bonsai-Shiny app.')

    parser.add_argument('--dataset', type=str, default='test_sarah',
                        help='Name of dataset. This will determine name of results-folder where information is stored.')
    # Arguments that define where to find the data and where to store results. Also, whether data comes from Sanity or not
    # parser.add_argument('--results_folder', type=str, default="/Users/sarahmorillo/bz_mnt/software/waddington-code-github/python_waddington_code/downstream_analyses/test_res",
    #                     help='path to folder where results will be stored.')
    parser.add_argument('--tree_folder', type=str,
                        # default='/Users/sarahmorillo/bz_mnt/software/waddington-code-github/python_waddington_code/downstream_analyses/test_data/tamara_ecoli_isolates/mergers_zscore1.0_ellipsoidsize1.0_smallerrorbars_redoStarry_optTimes_nnnReorder_reorderedEdges',
                        default='/Users/sarahmorillo/bz_mnt/software/waddington-code-github/downstream_analyses/test/test_data/',
                        help='Path to folder that determines tree topology. Should contain edgeInfo and vertInfo and tree.nwk')
    parser.add_argument('--output_file', type=str,
                        default="/Users/sarahmorillo/bz_mnt/software/waddington-code-github/downstream_analyses/test/test_res/test_binary_tree_8_leafs",
                        help="output file name")
    parser.add_argument('--t', type=float, default=5, dest='max_diam_threshold', help="max diam threshold")
    # Arguments that determine running configurations of MLTree. How much is printed, which steps are run?
    # parser.add_argument('--verbose', type=str2bool, default=True,
    #                     help='--verbose False only shows essential print messages (default: True)')

    args = parser.parse_args()
    print(args)

    """Test max footfall and min pdists clustering on some test case"""
    nwk_file = '/Users/Daan/Documents/postdoc/bonsai-development/results/simulated_datasets/simulated_binary_6_gens_samplingnoise_seed_1231/final_bonsai_zscore1.0/tree.nwk'
    with open(nwk_file, "r") as f:
        nwk_str = f.readline()
    clusters_list, cut_edges_main = get_footfall_clustering_from_nwk_str(tree_nwk_str=nwk_str,
                                                                         n_clusters=8)
    clusters_list_min_pd, cut_edges_min_pd = get_min_pdists_clustering_from_nwk_str(tree_nwk_str=nwk_str,
                                                                                    n_clusters=8)

    """In the following lines the tree is read in from the output generated by Bonsai"""

    clusters_md = get_max_diam_clustering_from_nwk_file(
        tree_nwk_file=os.path.join(args.tree_folder, 'test_binary_tree_8_leafs.nwk'),
        max_diam_threshold=args.max_diam_threshold)

    cl_dict = get_cluster_assignments(clusters_list=clusters_md)

    # print output:
    print("number of clusters found: {}".format(len(clusters_md)))
    # write to file:
    with open(args.output_file + "-max_diam_{}.cluster".format(args.max_diam_threshold), "w") as fout:
        cluster_idx = 0
        for cluster in clusters_md:
            # if singleton, assign -1
            if len(cluster) == 1:
                fout.write("{}\t{}\n".format(cluster[0], -1))
            else:
                for leaf in cluster:
                    fout.write("{}\t{}\n".format(leaf, cluster_idx))

                cluster_idx += 1

    print("done")

    # try out to get longest path from root to a leaf
    cluster_tree_md = Cluster_Tree()
    cluster_tree_md.from_newick_file(
        nwk_file=os.path.join(args.tree_folder, 'test_binary_tree_8_leafs-longest_path_to_leaf_11.5.nwk'))  # Works
    longest_path_from_root_to_leaf, _ = cluster_tree_md.root.find_longest_path_between_two_leafs()
