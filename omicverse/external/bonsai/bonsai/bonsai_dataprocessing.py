from .bonsai_helpers import *
import os
from scipy.optimize import fminbound
from scipy.spatial import distance
import pandas as pd
import csv
from .bonsai_treeHelpers import Tree, TreeNode, optimiseTStar
from . import bonsai_globals as bs_glob
import copy
import gc
import json
import heapq

import logging

FORMAT = '%(asctime)s %(funcName)s %(levelname)s %(message)s'
log_level = logging.DEBUG
logging.basicConfig(format=FORMAT,
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.WARNING)  # silence all libraries

# Create your app logger
logger = logging.getLogger("myapp")
logger.setLevel(log_level)


class Metadata:
    pathToOrigData = None
    dataset = None
    nCells = None
    nGenes = None
    cellIds = None
    geneIds = None
    loglik = None
    loglikVarCorr = None
    geneVariances = None
    geneDiffusionScaling = None
    processedDatafolder = None
    results_folder = None

    def __repr__(self):
        return "Metadata(\n" \
               "pathToOrigData = %r \n" \
               "dataset = %r \n" \
               "nCells = %r \n" \
               "nGenes = %r \n" \
               "cellIds = %r \n" \
               "geneIds = %r \n" \
               "loglik = %r \n" \
               "loglikVarCorr = %r \n" \
               "geneVariances = %r \n" \
               "geneDiffusionScaling = %r \n" \
               "results_folder = %r \n" \
               "processedDatafolder = %r \n)" \
               % (self.pathToOrigData, self.dataset, self.nCells, self.nGenes, self.cellIds, self.geneIds, self.loglik,
                  self.loglikVarCorr, self.geneVariances, self.geneDiffusionScaling,
                  self.results_folder, self.processedDatafolder)

    def __init__(self, json_filepath=None, curr_metadata=None):
        attr_list = ['pathToOrigData', 'dataset', 'nCells', 'nGenes', 'cellIds', 'geneIds', 'loglik', 'loglikVarCorr',
                     'geneVariances', 'geneDiffusionScaling', 'processedDatafolder', 'results_folder']
        if json_filepath is not None:
            self.from_json(json_filepath)
        else:
            for attr in attr_list:
                setattr(self, attr, None)
        if curr_metadata is not None:
            for attr in attr_list:
                if getattr(curr_metadata, attr) is not None:
                    if attr in ['cellIds', 'geneIds', 'geneVariances']:
                        copied_attr = getattr(curr_metadata, attr).copy()
                    else:
                        copied_attr = getattr(curr_metadata, attr)
                    setattr(self, attr, copied_attr)

        # Check if the stored number-of-cells matches the number of stored cell-IDs
        if hasattr(self, 'nCells') and (self.nCells is not None) and \
                hasattr(self, 'cellIds') and (self.cellIds is not None):
            if self.nCells != len(self.cellIds):
                mp_print("Number of cell-IDs ({}) stored in metadata, is not equal to value of nCells ({}). "
                         "Check this! Now exiting.".format(len(self.cellIds), self.nCells), ERROR=True)
                exit()

    def to_dict(self):
        self_dict = self.__dict__
        new_dict = {}
        for self_label, self_val in self_dict.items():
            if self_label in ['geneVariances'] and (self_val is not None):
                new_dict[self_label] = self_val.tolist()
            elif hasattr(self_val, 'dtype'):
                if self_val.dtype == 'float64':
                    new_dict[self_label] = float(self_val)
                elif self_val.dtype == 'int64':
                    new_dict[self_label] = int(self_val)
                else:
                    new_dict[self_label] = self_val
            else:
                new_dict[self_label] = self_val
        return new_dict

    def from_dict(self, metadata_dict):
        for md_label, md_val in metadata_dict.items():
            if md_label in ['geneVariances'] and (md_val is not None):
                setattr(self, md_label, np.array(md_val))
            elif isinstance(md_val, float):
                setattr(self, md_label, np.float64(md_val))
            else:
                setattr(self, md_label, md_val)

    def to_json(self, filepath):
        with open(filepath, "w") as file:
            json.dump(self.to_dict(), file, indent=4)

    def from_json(self, filepath):
        with open(filepath, 'r') as file:
            self.from_dict(metadata_dict=json.load(file))

    def take_subset_genes(self, genes_to_keep):
        new_nGenes = len(genes_to_keep)
        if self.nGenes < new_nGenes:
            logger.error("Trying to take a subset of genes with more genes than original.")
            exit()
        elif self.nGenes > new_nGenes:
            self.geneIds = [self.geneIds[ind] for ind in genes_to_keep]
            if self.geneVariances is not None:
                self.geneVariances = self.geneVariances[genes_to_keep]
            self.nGenes = new_nGenes


class OriginalData:
    umiCounts = None
    ltqs = None
    ltqsVars = None
    geneVariances = None
    geneDiffusionScaling = None
    priorVariances = None

    def __repr__(self):
        return "OriginalData(\numiCounts = %r \nltqs = %r \nltqsVars = %r \n" \
               "geneVariances = %r \npriorVariances = %r \n)" \
               % (self.umiCounts, self.ltqs, self.ltqsVars, self.geneVariances, self.priorVariances)


class SCData:
    # Object with useful information about the dataset
    metadata = None

    # The original data itself
    originalData = None
    unscaled = None

    # The tree object
    tree = None
    mergers = None  # List of node mergers to recover tree from star-tree
    nVerts = None

    def __repr__(self):
        return "SCData(\n metadata = %r \noriginalData = %r \nuncorrected = %r \ntree = %r \nmergers = %r \n)" \
               % (self.metadata, self.originalData, self.unscaled, self.tree, self.mergers)

    def __init__(self, dataset=None, filenamesData=None, verbose=False, pathToOrigData=None,
                 zscoreCutoff=-1, onlyObject=False, returnUncorrected=False,
                 createStarTree=True, optTimes=True, getOrigData=False, results_folder=None,
                 noDataNeeded=False, sanityOutput=True, mpiInfo=None, rescale_by_var=True, all_genes=False):
        if all_genes:
            zscoreCutoff = -1
        if mpiInfo is None:
            mpiInfo = mpi_wrapper.get_mpi_info()
        if onlyObject:
            self.metadata = Metadata()
            if dataset:
                self.metadata.dataset = dataset
            if results_folder and len(results_folder):
                self.metadata.results_folder = os.path.abspath(results_folder)
            if mpiInfo.rank == 0:
                Path(self.data_path()).mkdir(parents=True, exist_ok=True)
                Path(self.result_path()).mkdir(parents=True, exist_ok=True)
            return
        if not dataset:
            mp_print("Dataset-ID should be given.")
            exit()

        self.metadata = Metadata()
        if results_folder and len(results_folder):
            self.metadata.results_folder = os.path.abspath(results_folder)
        if pathToOrigData is not None:
            self.metadata.pathToOrigData = os.path.abspath(pathToOrigData)
        self.metadata.dataset = dataset
        if mpiInfo.rank == 0:
            Path(self.data_path()).mkdir(parents=True, exist_ok=True)
            Path(self.result_path()).mkdir(parents=True, exist_ok=True)

        ltqStdsFound, originalData = self.read_in_data(filenamesData=filenamesData, verbose=verbose,
                                                       noDataNeeded=noDataNeeded, sanityOutput=sanityOutput,
                                                       getOrigData=getOrigData or (not createStarTree),
                                                       zscoreCutoff=zscoreCutoff, mpiInfo=mpiInfo, all_genes=all_genes)
        if mpiInfo.rank != 0:
            return
        # if returnUncorrected:
        #     self.unscaled = copy.deepcopy(originalData)

        self.metadata.geneVariances = originalData.geneVariances
        if self.metadata.geneVariances is None:
            self.metadata.geneVariances = np.ones(self.metadata.nGenes)

        # Scale diffusion by estimated gene variance.
        if rescale_by_var:
            if originalData.ltqs is not None:
                originalData.ltqs /= np.sqrt(originalData.geneVariances[:, None])
                if originalData.ltqsVars is not None:
                    originalData.ltqsVars /= (originalData.geneVariances[:, None])
                self.metadata.loglikVarCorr = - (self.metadata.nCells - 1) * np.sum(
                    np.log(
                        originalData.geneVariances))  # - self.metadata.nCells * self.metadata.nGenes * np.log(2* np.pi)
                originalData.geneDiffusionScaling = 'geneVariances'
                self.metadata.geneDiffusionScaling = 'geneVariances'
            # elif originalData.ltqs is not None:
            #     originalData.ltqs /= np.sqrt(originalData.geneVariances[:, None])
            #     originalData.geneDiffusionScaling = self.metadata.geneVariances
            #     self.metadata.geneDiffusionScaling = self.metadata.geneVariances
            #     self.metadata.loglikVarCorr = - (self.metadata.nCells - 1) * np.sum(
            #         np.log(originalData.geneVariances))  #-self.metadata.nCells*self.metadata.nGenes * np.log(2* np.pi)
        else:
            if originalData.ltqs is not None:
                nVars = self.metadata.nGenes if self.metadata.nGenes is not None else originalData.geneVariances.shape
                mean_var = np.mean(originalData.geneVariances)
                originalData.geneDiffusionScaling = mean_var
                self.metadata.geneDiffusionScaling = originalData.geneDiffusionScaling
                originalData.ltqs /= np.sqrt(mean_var)
                if originalData.ltqsVars is not None:
                    originalData.ltqsVars /= mean_var
                self.metadata.loglikVarCorr = - (self.metadata.nCells - 1) * nVars * np.log(mean_var)

        if originalData.ltqs is not None:
            # Store data filtered by zscore
            self.metadata.processedDatafolder = self.result_path('zscorefiltered_%.3f_and_processed' % zscoreCutoff)
            storeData(self.metadata, originalData.ltqs, originalData.ltqsVars, verbose=verbose)

        # Some variables are nice to have access to from all functions
        bs_glob.nCells = self.metadata.nCells
        bs_glob.nGenes = self.metadata.nGenes

        if (bs_glob.nCells is not None) and (bs_glob.nCells > 25000) and (self.tree is not None):
            bs_glob.mem_friendly = True
            self.tree.root.keep_one_ltqsvars_or_W(keep_ltqsvars=True)

        # Update recursion limits so that very deep trees don't create errors
        set_recursion_limits(int(2 * bs_glob.nCells))

        if getOrigData:
            self.originalData = originalData

        if createStarTree:
            self.tree = Tree()
            self.tree.initialize_star_tree(originalData.ltqs, originalData.ltqsVars, self.metadata,
                                           opt_times=optTimes, verbose=verbose)

    """----------TREE MANIPULATIONS-------------------------------"""

    # Used
    def compile_tree_from_mergers(self, mergers=None):
        if mergers is None:
            mergers = self.mergers
        edge_list = []
        mergers_array = np.array(mergers)
        orig_vert_names = [self.metadata.cellIds[int(ind)] for ind in
                           np.setdiff1d(np.arange(self.metadata.nCells), mergers_array[:, 1])]
        # orig_vert_names = [self.metadata.cellIds[int(mergers[-1][0])]]
        dist_list = []
        internal_node_counter = 0

        starryYN = len(orig_vert_names) > 1
        if starryYN:  # This happens when reading from tmp-file, where root still had multiple children
            orig_vert_names = ['internal_0'] + orig_vert_names
            internal_node_counter += 1
            edge_list = [(0, ind) for ind in range(1, len(orig_vert_names))]
            dist_list = [1.0] * len(edge_list)

        for split in reversed(mergers):
            # Find out who the father is
            left_child = self.metadata.cellIds[int(split[0])]
            right_child = self.metadata.cellIds[int(split[1])]
            parent_ind = [ind for ind, vert_name in enumerate(orig_vert_names) if vert_name == left_child][0]

            # Change the name of the parent, this is now an internal node
            orig_vert_names[parent_ind] = 'internal_' + str(internal_node_counter)
            internal_node_counter += 1

            # Find out division times of ancestor and dist to children
            # tdiv_anc, dist_child1, dist_child2 = split[2:]
            dist_child1, dist_child2 = split[2:]

            # Add two edges for the two children
            curr_n_vertices = len(orig_vert_names)
            # curr_fij = fijs[parent_ind]
            # summed_dist = split[2]
            edge_list.append((parent_ind, curr_n_vertices))
            dist_list.append(dist_child1)
            edge_list.append((parent_ind, curr_n_vertices + 1))
            dist_list.append(dist_child2)

            # Add both children under their own name
            orig_vert_names.append(left_child)
            orig_vert_names.append(right_child)
        return edge_list, dist_list, orig_vert_names, starryYN

    # Not Used
    # def buildTreeFromMergers(self, mergers):
    #     edgeList, distList, origVertNames, starryYN = self.compile_tree_from_mergers(mergers)
    #     tree = Tree()
    #     tree.starryYN = starryYN
    #     rootInd = edgeList[0][0]
    #     tree.root.nodeInd = rootInd
    #     tree.root.childNodes = []
    #     currInds = [rootInd]
    #     currNodes = [tree.root]
    #     # Build tree from edgeList, distList
    #     for ind, edge in enumerate(edgeList):
    #         dist = distList[ind]
    #         if edge[0] in currInds:
    #             parentInd = currInds.index(edge[0])
    #             childInd = edge[1]
    #         else:
    #             parentInd = currInds.index(edge[1])
    #             childInd = edge[0]
    #         parent = currNodes[parentInd]
    #         if dist > 0 or (origVertNames[ind + 1][:8] != 'internal'):
    #             parent.isLeaf = False
    #             child = TreeNode(nodeInd=childInd, tParent=dist, childNodes=[], isLeaf=True)
    #             parent.childNodes.append(child)
    #             bs_glob.nNodes += 1
    #             currInds.append(childInd)
    #             currNodes.append(child)
    #         else:
    #             currInds.append(childInd)
    #             currNodes.append(parent)
    #
    #     # Add ltq-information to leafs of the tree
    #     for cellInd, cellId in enumerate(self.metadata.cellIds):
    #         nodeInd = origVertNames.index(cellId)
    #         node = currNodes[nodeInd]
    #         node.nodeId = cellId
    #         node.isCell = True
    #         if self.originalData is not None:
    #             node.ltqs = self.originalData.ltqs[:, cellInd]
    #             node.ltqsVars = self.originalData.ltqsVars[:, cellInd]
    #         node.nodeInd = cellInd
    #     tree.root.renumberNodes(change_node_inds=False)
    #     return tree

    # Used
    def storeTreeInFolder(self, tree_folder, with_coords=False, verbose=False, all_ranks=False, cleanup_tree=True,
                          nwk=True, store_posterior_ltqs=False):
        coords_folder = tree_folder if with_coords else None
        mpiRank = mpi_wrapper.get_process_rank()
        if cleanup_tree:
            self.tree.root.mergeZeroTimeChilds()
            self.cleanup_node_inds()
            # self.tree.root.renumberNodes(change_node_inds=False)
            self.tree.nNodes = bs_glob.nNodes
        if (mpiRank == 0) or all_ranks:
            Path(tree_folder).mkdir(parents=True, exist_ok=True)
            self.tree.store_tree_info_and_coords(tree_folder, coords_folder=coords_folder,
                                                 verbose=False,
                                                 store_posterior_ltqs=store_posterior_ltqs,
                                                 geneDiffusionScaling=self.metadata.geneDiffusionScaling,
                                                 variances=self.metadata.geneVariances)

            self.metadata.to_json(os.path.join(tree_folder, 'metadata.json'))
            if nwk:
                self.tree.to_newick(use_ids=True, results_path=os.path.join(tree_folder, 'tree.nwk'))

    def communicate_tree_topology(self):
        if mpi_wrapper.get_process_rank() == 0:
            # Make copy of tree
            tree_copy = self.tree.copy_tree_topology()
            tree_copy.root.storeParent()
        else:
            tree_copy = None
        tree_copy = mpi_wrapper.bcast(tree_copy, root=0)
        if mpi_wrapper.get_process_rank() != 0:
            self.tree = tree_copy

    # Should go to tree layout file
    # def makeIgraphTreeFromSCData(self, merge_tol=1e-9, from_scData=False):
    #     if from_scData:
    #         edge_list, dist_list, orig_vert_names, starryYN, _ = self.compile_tree_from_scData_tree()
    #     else:
    #         edge_list, dist_list, orig_vert_names, starryYN = self.compile_tree_from_mergers()
    #
    #     # First we define every vertex in the binary tree as being in their own cluster
    #     # clustering = list(range(len(orig_vert_names)))
    #     # clustering = {orig_vert_names[nodeInd]: ind for ind, nodeInd in enumerate(orig_vert_names)}
    #     # # Then merge nodes that have an edge length of 0 between them
    #     # edges_to_remove = []
    #     # for ind, edge in enumerate(edge_list):
    #     #     if dist_list[ind] <= merge_tol:
    #     #         clustering[orig_vert_names[edge[1]]] = clustering[orig_vert_names[edge[0]]]
    #     #         edges_to_remove.append(ind)
    #     #     edge_list[ind] = (clustering[orig_vert_names[edge[0]]], clustering[orig_vert_names[edge[1]]])
    #     #
    #     # # This results in a clustering-list where some cells go to the same cluster (vertex on the tree)
    #     # edge_list = [edge for ind, edge in enumerate(edge_list) if ind not in edges_to_remove]
    #     # dist_list = [dist for ind, dist in enumerate(dist_list) if ind not in edges_to_remove]
    #
    #     # # We number the remaining vertices
    #     # new_vert_inds = {label: ind for ind, label in enumerate(np.sort(np.unique(clustering)))}
    #     # # And adjust the edge_list to map to these new indices
    #     # edge_list = [(new_vert_inds[edge[0]], new_vert_inds[edge[1]]) for edge in edge_list]
    #
    #     # We store to which clst every original node in the binary tree went
    #     self.clustering = np.arange(len(orig_vert_names))
    #     nodeIndToVertInd = {nodeInd: ind for ind, nodeInd in enumerate(orig_vert_names)}
    #     # Also, we store how many vertices there are in our final tree
    #     self.nVert = len(orig_vert_names)
    #     # And we  make up some names
    #     self.vertNames = ['vert_' + str(ind) for ind in range(self.nVert)]
    #
    #     # If the tree reconstruction is run on cellstates-output, we should first account for mapping the cells to
    #     # cellstates
    #     self.cellsToCellstates = {}
    #     self.cellstatesToVerts = {}
    #     self.cellsToVerts = {}
    #     self.cellIndToVertInd = {}
    #
    #     for ind in range(self.metadata.nCells):
    #         cellId = self.metadata.cellIds[ind]
    #         # We have not yet implemented that we ran it on cs, so now we just map cells to their own cellstate
    #         cellstateId = cellId
    #         self.cellsToCellstates[cellId] = cellstateId
    #
    #         # Find out which node_ind this cellstate was mapped to
    #         node_inds = [nodeInd for nodeInd in orig_vert_names if orig_vert_names[nodeInd] == cellId]
    #         if len(node_inds) != 1:
    #             print("Couldn't find " + cellId)
    #             node_ind = -1
    #         else:
    #             node_ind = node_inds[0]
    #         # Find out to which vertex ind this was again mapped
    #         vert_ind = nodeIndToVertInd[node_ind]
    #         # Find out to which new vertex this was again send to
    #         self.cellstatesToVerts[cellstateId] = self.vertNames[vert_ind]
    #         # Store in cell_assignment to which eventual vertex the original cell was sent
    #         self.cellsToVerts[cellId] = self.cellstatesToVerts[self.cellsToCellstates[cellId]]
    #         # Also store a map of which cellInd went to which vertInd
    #         self.cellIndToVertInd[ind] = vert_ind
    #
    #     # Make edge-list between vert_inds instead of between node_inds
    #     edge_list = [(nodeIndToVertInd[edge[0]], nodeIndToVertInd[edge[1]]) for edge in edge_list]
    #
    #     # Build up the tree
    #     self.mst = igraph.Graph(directed=True)
    #     self.mst.add_vertices(self.nVert)
    #
    #     self.mst.add_edges(edge_list)
    #     self.mst.es["weight"] = dist_list
    #     self.mst.vs["name"] = self.vertNames
    #     self.mst.es["name"] = ['e_' + str(ind) for ind in range(self.nVert - 1)]
    #     igraph.summary(self.mst)
    #     self.nCellsPerVert = np.zeros(self.nVert)
    #     self.vertIndToCellInds = {ind: [] for ind in range(self.nVert)}
    #     for ind in range(self.metadata.nCells):
    #         self.vertIndToCellInds[self.cellIndToVertInd[ind]].append(ind)
    #         self.nCellsPerVert[self.cellIndToVertInd[ind]] += 1
    #
    # # Should go to tree layout file
    # def makeIgraphTree(self, merge_tol=1e-9, from_scData=False):
    #     if from_scData:
    #         edge_list, dist_list, orig_vert_names, starryYN, _ = self.compile_tree_from_scData_tree()
    #     else:
    #         edge_list, dist_list, orig_vert_names, starryYN = self.compile_tree_from_mergers()
    #
    #     # First we define every vertex in the binary tree as being in their own cluster
    #     clustering = list(range(len(orig_vert_names)))
    #     # Then merge nodes that have an edge length of 0 between them
    #     edges_to_remove = []
    #     for ind, edge in enumerate(edge_list):
    #         if dist_list[ind] <= merge_tol:
    #             clustering[edge[1]] = clustering[edge[0]]
    #             edges_to_remove.append(ind)
    #         edge_list[ind] = (clustering[edge[0]], clustering[edge[1]])
    #
    #     # This results in a clustering-list where some cells go to the same cluster (vertex on the tree)
    #     edge_list = [edge for ind, edge in enumerate(edge_list) if ind not in edges_to_remove]
    #     dist_list = [dist for ind, dist in enumerate(dist_list) if ind not in edges_to_remove]
    #
    #     # We number the remaining vertices
    #     new_vert_inds = {label: ind for ind, label in enumerate(np.sort(np.unique(clustering)))}
    #     # And adjust the edge_list to map to these new indices
    #     edge_list = [(new_vert_inds[edge[0]], new_vert_inds[edge[1]]) for edge in edge_list]
    #
    #     # We store to which clst every original node in the binary tree went
    #     self.clustering = np.array([new_vert_inds[clst] for clst in clustering])
    #     # Also, we store how many vertices there are in our final tree
    #     self.nVert = len(np.unique(self.clustering))
    #     # And we  make up some names
    #     self.vertNames = ['vert_' + str(ind) for ind in range(self.nVert)]
    #
    #     # Now we would still like to know to which vertex the different cells were mapped
    #     self.cellsToVerts = []
    #
    #     # If the tree reconstruction is run on cellstates-output, we should first account for mapping the cells to
    #     # cellstates
    #     self.cellsToCellstates = {}
    #     self.cellstatesToVerts = {}
    #     self.cellsToVerts = {}
    #     self.cellIndToVertInd = {}
    #
    #     for ind in range(self.metadata.nCells):
    #         cellId = self.metadata.cellIds[ind]
    #         # We have not yet implemented that we ran it on cs, so now we just map cells to their own cellstate
    #         cellstateId = cellId
    #         self.cellsToCellstates[cellId] = cellstateId
    #
    #         # Find out which original vertex (before merging zero distance pairs) this cellstate was mapped to
    #         vert_ind = [vert_ind for vert_ind, vert_name in enumerate(orig_vert_names) if vert_name == cellstateId][0]
    #         # Find out to which new vertex this was again send to
    #         self.cellstatesToVerts[cellstateId] = self.vertNames[self.clustering[vert_ind]]
    #         # Store in cell_assignment to which eventual vertex the original cell was sent
    #         self.cellsToVerts[cellId] = self.cellstatesToVerts[self.cellsToCellstates[cellId]]
    #         # Also store a map of which cellInd went to which vertInd
    #         self.cellIndToVertInd[ind] = self.clustering[vert_ind]
    #
    #     # Build up the tree
    #     self.mst = igraph.Graph(directed=True)
    #     self.mst.add_vertices(self.nVert)
    #
    #     self.mst.add_edges(edge_list)
    #     self.mst.es["weight"] = dist_list
    #     self.mst.vs["name"] = self.vertNames
    #     self.mst.es["name"] = ['e_' + str(ind) for ind in range(self.nVert - 1)]
    #     igraph.summary(self.mst)
    #     self.nCellsPerVert = np.zeros(self.nVert)
    #     self.vertIndToCellInds = {ind: [] for ind in range(self.nVert)}
    #     for ind in range(self.metadata.nCells):
    #         self.vertIndToCellInds[self.cellIndToVertInd[ind]].append(ind)
    #         self.nCellsPerVert[self.cellIndToVertInd[ind]] += 1

    # Should go to tree layout file
    def find_branching_points(self, level=1, verbose=False):
        # leaves = self.mst.vs.select(_degree_eq=1)
        # parents = []
        # dsInds = [leaf.index for leaf in leaves]
        # allNonBranching = dsInds
        # for ind in range(level):
        #     parents.append([])
        #     for dsInd in dsInds:
        #         parent_edge = self.mst.es.select(_target=dsInd)
        #         if len(parent_edge) > 0:
        #             parents[ind].append(parent_edge[0].source)
        #     dsInds = parents[ind]
        #     allNonBranching += dsInds
        # branching_inds = [vert.index for vert in self.mst.vs if
        #                   (vert.index not in allNonBranching)]
        # self.branchingInds = branching_inds

        leaves = self.mst.vs.select(_degree_eq=1)
        dsInds = [leaf.index for leaf in leaves]
        levelDict = {}
        lev = 0
        for ind in dsInds:
            levelDict[ind] = lev
        stillParents = True
        while stillParents:
            lev += 1
            parents = []
            for dsInd in dsInds:
                parent_edge = self.mst.es.select(_target=dsInd)
                if len(parent_edge) > 0:
                    parents.append(parent_edge[0].source)
            dsInds = parents.copy()
            for ind in dsInds:
                levelDict[ind] = lev
            stillParents = (len(parents) > 0)
        allNonBranching = []
        for ind, lev in levelDict.items():
            if lev < level:
                allNonBranching.append(ind)
        branching_inds2 = [vert.index for vert in self.mst.vs if
                           (vert.index not in allNonBranching)]
        self.branchingInds = branching_inds2

    # Should go to tree helpers file
    def merge_clusters_acc_cellstates(self):
        cellMask = self.nCellsPerVert != 0
        summed_counts_vert = np.zeros((bs_glob.nGenes, self.nVerts))
        if self.umiCounts is None:
            self.vertUMIs = None
        else:
            for cellInd in range(self.metadata.nCells):
                summed_counts_vert[:, self.cellIndToVertInd[cellInd]] += self.umiCounts[:, cellInd]
            self.vertUMIs = summed_counts_vert
        if (self.originalData is None) or (self.originalData.ltqs is None):
            self.vertLtqs = None
        else:
            mean_ltqs_vert = np.zeros((self.originalData.ltqs.shape[0], self.nVerts))
            for cellInd in range(self.metadata.nCells):
                mean_ltqs_vert[:, self.cellIndToVertInd[cellInd]] += self.originalData.ltqs[:, cellInd]
            cellMask = self.nCellsPerVert != 0
            mean_ltqs_vert[:, cellMask] /= self.nCellsPerVert[cellMask]
            self.vertLtqs = mean_ltqs_vert

    # Should go to tree visualisation file
    def set_node_edge_style(self, style_type='default'):
        from .bonsai_helpers import custom_colors_rgba, gradient_colors, blackish, grey

        counts = self.nCellsPerVert
        cnt_min, cnt_max = np.min(counts), np.max(counts)

        # non_cell_nodes = [ind for ind in range(self.nVert) if counts[ind] == 0]
        non_cell_nodes = np.where(counts == 0)[0]

        node_style = {}
        node_style['s'] = np.ones(self.nVerts) * 4.
        nonzero_counts = np.where(counts > 0)[0]
        node_style['s'][nonzero_counts] += 20 * np.sqrt(counts[nonzero_counts])
        node_style['s'] = list(node_style['s'])

        edge_style = {}
        edge_style['color'] = blackish
        edge_style['linewidth'] = .375

        if style_type == 'default':
            node_style["color"] = [custom_colors_rgba[0]] * self.nVerts

        if style_type == 'umi_counts':
            summed_umi_counts = np.log(np.sum(self.vertUMIs, axis=0))
            min_sum = np.min(summed_umi_counts)
            max_sum = np.max(summed_umi_counts)
            rescaled_sums = (summed_umi_counts - min_sum) / (max_sum - min_sum)
            node_style["color"] = [gradient_colors(scaled_sum) for scaled_sum in rescaled_sums]

        node_style['edgecolor'] = [blackish] * self.nVerts
        node_style['linewidths'] = [.1] * self.nVerts

        for ind in non_cell_nodes:
            node_style['color'][ind] = grey
            node_style['edgecolor'][ind] = grey
        node_style = np.array(node_style)

        return node_style, edge_style

    # Should go to tree visualisation file
    def get_celltype_annotations(self, celltype_metadata_path=None, vertIndToNodeId=None):
        create_celltype_metadata = celltype_metadata_path is not None
        # from helpers_full_gene_expression import celltype_colors, blackish, gradient_colors
        celltypes_found = False

        allcelltype_files = list(Path(self.data_path()).glob('Celltype*.txt'))
        allcelltype_files.sort()
        if create_celltype_metadata:
            metadata_index = [vertIndToNodeId[vertInd] for vertInd in self.vertIndToCellInds]
            ct_metadata = {}
            metadata_labels = []
        for celltype_file in allcelltype_files:
            celltypes_input = np.array(pd.read_csv(celltype_file, header=None).iloc[:, 0].tolist())
            celltypes_input = np.array([str(ct) for ct in celltypes_input])
            if not celltypes_found:
                celltypes_found = True
                self.celltypes = celltypes_input
                self.celltypesAlts = [self.celltypes]
            else:
                self.celltypesAlts.append(celltypes_input)
            if create_celltype_metadata:
                metadata_labels.append(os.path.basename(celltype_file).split('.txt')[0])
        if not celltypes_found:
            print("No celltype-file was found. Assigning same celltype to all cells")
            self.celltypesAlts = [np.array(['default'] * self.metadata.nCells)]
            metadata_labels.append("default")

        self.cellCategoriesAlts = []
        self.cellstateTypeListAlts = []
        for ct_alt_ind, celltype_alt in enumerate(self.celltypesAlts):
            cell_categories = np.unique(celltype_alt)
            cell_categories = np.append(cell_categories, 'internal_node')
            cellstate_type_list = ['internal_node'] * self.nVerts
            for vertInd, cellInds in self.vertIndToCellInds.items():
                celltypesPresent = np.array([celltype_alt[cellInd] for cellInd in cellInds])
                if len(celltypesPresent) > 0:
                    types, counts = np.unique(celltypesPresent, return_counts=True)
                    n_types = len(types)
                    if n_types == 1:
                        cellstate_type_list[vertInd] = types[0]
                    elif n_types > 1:
                        types_dec_abundance = types[np.argsort(-counts)]
                        mixed_type = types_dec_abundance[0]
                        for ind in range(1, n_types):
                            cell_type = types_dec_abundance[ind]
                            mixed_type = mixed_type + '++' + cell_type
                        cellstate_type_list[vertInd] = mixed_type

            self.cellCategoriesAlts.append(cell_categories)
            self.cellstateTypeListAlts.append(cellstate_type_list)
            if create_celltype_metadata:
                ct_metadata[metadata_labels[ct_alt_ind]] = cellstate_type_list

        self.celltypes = self.celltypesAlts[0]
        self.cellCategories = self.cellCategoriesAlts[0]
        self.cellstateTypeList = self.cellstateTypeListAlts[0]
        if create_celltype_metadata:
            metadata_df = pd.DataFrame(ct_metadata, index=metadata_index)
            metadata_df.to_csv(celltype_metadata_path, index=True, index_label='node_id')

    # def get_celltypes_new(self, annotation_path, celltype_metadata_path=None, vertIndToNodeId=None):
    #     """
    #     Requires a celltype-file which is a csv with as first column the cell-IDs and as other columns different
    #     celltype-annotations. A header line should give a label to the type of annotations.
    #     :param celltype_metadata_path:
    #     :param vertIndToNodeId:
    #     :return:
    #     """
    #     metadata_index = [vertIndToNodeId[vertInd] for vertInd in self.vertIndToCellInds]
    #     ct_metadata = {}
    #
    #     if os.path.isfile(annotation_path):
    #         celltypes_input = pd.read_csv(annotation_path, header=0, index_col=0).astype(dtype=str)
    #         if celltypes_input.shape[0] == (self.metadata.nCells - 1):
    #             celltypes_input = pd.read_csv(annotation_path, header=None, index_col=0).astype(dtype=str)
    #             celltypes_input.columns = ['Annotation {}'.format(ind) for ind in range(celltypes_input.shape[1])]
    #
    #         self.celltypes = np.array(celltypes_input.iloc[:,0].tolist())
    #         self.celltypesAlts = [np.array(celltypes_input.iloc[:,ind].tolist()) for ind in range(celltypes_input.shape[1])]
    #
    #         metadata_labels = list(celltypes_input.columns)
    #     else:
    #         print("No celltype-file was found. Assigning same celltype to all cells")
    #         self.celltypesAlts = [np.array(['default'] * self.metadata.nCells)]
    #         self.celltypes = self.celltypesAlts
    #         metadata_labels = ["default"]
    #
    #     self.cellCategoriesAlts = []
    #     self.cellstateTypeListAlts = []
    #     for ct_alt_ind, celltype_alt in enumerate(self.celltypesAlts):
    #         cell_categories = np.unique(celltype_alt)
    #         cell_categories = np.append(cell_categories, 'internal_node')
    #         cellstate_type_list = ['internal_node'] * self.nVert
    #         for vertInd, cellInds in self.vertIndToCellInds.items():
    #             celltypesPresent = np.array([celltype_alt[cellInd] for cellInd in cellInds])
    #             if len(celltypesPresent) > 0:
    #                 types, counts = np.unique(celltypesPresent, return_counts=True)
    #                 n_types = len(types)
    #                 if n_types == 1:
    #                     cellstate_type_list[vertInd] = types[0]
    #                 elif n_types > 1:
    #                     types_dec_abundance = types[np.argsort(-counts)]
    #                     mixed_type = types_dec_abundance[0]
    #                     for ind in range(1, n_types):
    #                         cell_type = types_dec_abundance[ind]
    #                         mixed_type = mixed_type + '++' + cell_type
    #                     cellstate_type_list[vertInd] = mixed_type
    #
    #         self.cellCategoriesAlts.append(cell_categories)
    #         self.cellstateTypeListAlts.append(cellstate_type_list)
    #         ct_metadata[metadata_labels[ct_alt_ind]] = cellstate_type_list
    #
    #     self.celltypes = self.celltypesAlts[0]
    #     self.cellCategories = self.cellCategoriesAlts[0]
    #     self.cellstateTypeList = self.cellstateTypeListAlts[0]
    #
    #     metadata_df = pd.DataFrame(ct_metadata, index=metadata_index)
    #     metadata_df.to_csv(celltype_metadata_path, index=True, index_label='node_id')
    #     return ct_metadata

    # def get_celltypes_new(self, annotation_path, celltype_metadata_path=None, vertIndToNodeId=None):
    #     """
    #     Requires a celltype-file which is a csv with as first column the cell-IDs and as other columns different
    #     celltype-annotations. A header line should give a label to the type of annotations.
    #     :param celltype_metadata_path:
    #     :param vertIndToNodeId:
    #     :return:
    #     """
    #     metadata_index = [vertIndToNodeId[vertInd] for vertInd in self.vertIndToCellInds]
    #     ct_metadata = {}
    #
    #     if os.path.isfile(annotation_path):
    #         celltypes_input = pd.read_csv(annotation_path, header=0, index_col=0).astype(dtype=str)
    #         if celltypes_input.shape[0] == (self.metadata.nCells - 1):
    #             celltypes_input = pd.read_csv(annotation_path, header=None, index_col=0).astype(dtype=str)
    #             celltypes_input.columns = ['Annotation {}'.format(ind) for ind in range(celltypes_input.shape[1])]
    #
    #         self.celltypes = np.array(celltypes_input.iloc[:, 0].tolist())
    #         self.celltypesAlts = [np.array(celltypes_input.iloc[:, ind].tolist()) for ind in
    #                               range(celltypes_input.shape[1])]
    #
    #         metadata_labels = list(celltypes_input.columns)
    #     else:
    #         print("No celltype-file was found. Assigning same celltype to all cells")
    #         self.celltypesAlts = [np.array(['default'] * self.metadata.nCells)]
    #         self.celltypes = self.celltypesAlts
    #         metadata_labels = ["default"]
    #
    #     self.cellCategoriesAlts = []
    #     self.cellstateTypeListAlts = []
    #     for ct_alt_ind, celltype_alt in enumerate(self.celltypesAlts):
    #         cell_categories = np.unique(celltype_alt)
    #         cell_categories = np.append(cell_categories, 'internal_node')
    #         cellstate_type_list = ['internal_node'] * self.nVerts
    #         for vertInd, cellInds in self.vertIndToCellInds.items():
    #             celltypesPresent = np.array([celltype_alt[cellInd] for cellInd in cellInds])
    #             if len(celltypesPresent) > 0:
    #                 types, counts = np.unique(celltypesPresent, return_counts=True)
    #                 n_types = len(types)
    #                 if n_types == 1:
    #                     cellstate_type_list[vertInd] = types[0]
    #                 elif n_types > 1:
    #                     types_dec_abundance = types[np.argsort(-counts)]
    #                     mixed_type = types_dec_abundance[0]
    #                     for ind in range(1, n_types):
    #                         cell_type = types_dec_abundance[ind]
    #                         mixed_type = mixed_type + '++' + cell_type
    #                     cellstate_type_list[vertInd] = mixed_type
    #
    #         self.cellCategoriesAlts.append(cell_categories)
    #         self.cellstateTypeListAlts.append(cellstate_type_list)
    #         ct_metadata[metadata_labels[ct_alt_ind]] = cellstate_type_list
    #
    #     self.celltypes = self.celltypesAlts[0]
    #     self.cellCategories = self.cellCategoriesAlts[0]
    #     self.cellstateTypeList = self.cellstateTypeListAlts[0]
    #
    #     metadata_df = pd.DataFrame(ct_metadata, index=metadata_index)
    #     metadata_df.to_csv(celltype_metadata_path, index=True, index_label='node_id')
    #     return ct_metadata

    def get_celltypes_minim(self, annotation_path):
        """
        Requires a celltype-file which is a csv with as first column the cell-IDs and as other columns different
        celltype-annotations. A header line should give a label to the type of annotations.
        :param celltype_metadata_path:
        :param vertIndToNodeId:
        :return:
        """
        # bonvis_data.obs["ct_{}".format(annot)] = pd.Categorical(ct_metadata[annot])
        if os.path.isfile(annotation_path):
            celltypes_input = pd.read_csv(annotation_path, header=0, index_col=0).astype(dtype=str)
            if celltypes_input.shape[0] == (self.metadata.nCells - 1):
                celltypes_input = pd.read_csv(annotation_path, header=None, index_col=0).astype(dtype=str)
                celltypes_input.columns = ['Annotation {}'.format(ind) for ind in range(celltypes_input.shape[1])]

            cell_id_to_ind = {cell_id: ind for ind, cell_id in enumerate(self.metadata.cellIds)}
            cell_inds = np.array([cell_id_to_ind[cell_id] for cell_id in celltypes_input.index])
            celltype_df = celltypes_input.iloc[cell_inds, :]
            # Determine which columns are numerical values
            new_colnames = []
            for ind, col in enumerate(celltype_df.columns):
                if np.all(celltype_df[col].apply(is_numeric)):
                    celltype_df[col] = celltype_df[col].astype(float)
                    new_colnames.append('annot_num_{}'.format(col))
                else:
                    celltype_df[col] = pd.Categorical(celltype_df[col])
                    new_colnames.append('annot_{}'.format(col))
            celltype_df.columns = new_colnames
        else:
            print("No celltype-file was found. Assigning same celltype to all cells")
            celltype_df = pd.DataFrame({'annot_default': np.array(['default'] * self.metadata.nCells)},
                                       index=self.metadata.cellIds)
            celltype_df['annot_default'] = pd.Categorical(celltype_df['annot_default'])
        # celltype_df.reset_index(inplace=True)
        return celltype_df

    def get_annotations(self, annotation_folder):
        """
        Requires a path to a folder where files are stored with annotation information. These files should be
        csvs with as first column the cell-IDs and as other columns different
        celltype-annotations. A header line should give a label to the type of annotations.
        :return:
        """
        if (annotation_folder is not None) and os.path.isfile(annotation_folder):
            file_list = [annotation_folder]
        elif (annotation_folder is not None) and os.path.isdir(annotation_folder):
            file_list = os.listdir(annotation_folder)
        else:
            print("No celltype-file was found. Assigning same celltype to all cells")
            annotation_df = pd.DataFrame({'annot_default': np.array(['default'] * self.metadata.nCells)},
                                         index=self.metadata.cellIds)
            annotation_df['annot_default'] = pd.Categorical(annotation_df['annot_default'])
            return annotation_df, {}

        file_list = [file for file in file_list if (os.path.splitext(file)[1] == '.csv')]
        annotation_dfs = []
        data_matrices = {}
        for ind_file, filename in enumerate(file_list):
            filepath = os.path.join(annotation_folder, filename)
            annot_input = pd.read_csv(filepath, header=0, index_col=0)  # .astype(dtype=str)
            if annot_input.shape[0] in [self.metadata.nCells - 1, self.metadata.nCells]:
                if annot_input.shape[0] == (self.metadata.nCells - 1):
                    annot_input = pd.read_csv(filepath, header=None, index_col=0)
                    annot_input.columns = ['Annotation {}_{}'.format(ind_file, ind) for ind in
                                           range(annot_input.shape[1])]
                cell_id_to_ind = {cell_id: ind for ind, cell_id in enumerate(annot_input.index)}
                cell_inds = np.array([cell_id_to_ind[cell_id] for cell_id in self.metadata.cellIds])
                annotation_df = annot_input.iloc[cell_inds, :]
            # elif hasattr(self.metadata, 'nCss') and (annot_input.shape[0] in [self.metadata.nCss -1, self.metadata.nCss]):
            #     if annot_input.shape[0] == (self.metadata.nCss - 1):
            #         annot_input = pd.read_csv(filepath, header=None, index_col=0)
            #         annot_input.columns = ['Annotation {}_{}'.format(ind_file, ind) for ind in range(annot_input.shape[1])]
            #     cs_id_to_ind = {cell_id: ind for ind, cell_id in enumerate(annot_input.index)}
            #     cs_inds = np.array([cs_id_to_ind[cs_id] for cs_id in self.metadata.csIds])
            #     annotation_df = annot_input.iloc[cs_inds, :]
            if filename[:4] == 'mat_':
                mat_label = ''.join(filename[4:].split('.')[:-1])
                data_matrices[mat_label] = annotation_df.T
            else:
                # Determine which columns are numerical values
                new_colnames = []
                for ind, col in enumerate(annotation_df.columns):
                    if np.all(annotation_df[col].apply(is_numeric)):
                        annotation_df[col] = annotation_df[col].astype(float)
                        new_colnames.append('annot_num_{}'.format(col))
                    else:
                        try:
                            annotation_df[col] = annotation_df[col].fillna("NaN")
                        except:
                            print("Could not convert nans in column {}.".format(col))
                        annotation_df[col] = pd.Categorical(annotation_df[col])
                        new_colnames.append('annot_{}'.format(col))
                annotation_df.columns = new_colnames
                annotation_dfs.append(annotation_df)
        annotation_dfs = pd.concat(annotation_dfs, axis=1)
        # celltype_df.reset_index(inplace=True)
        return annotation_dfs, data_matrices

    def get_annotations_with_cs(self, annotation_folder):
        """
        Requires a path to a folder where files are stored with annotation information. These files should be
        csvs with as first column the cell-IDs and as other columns different
        celltype-annotations. A header line should give a label to the type of annotations.
        :return:
        """
        if (annotation_folder is not None) and os.path.isfile(annotation_folder):
            file_list = [annotation_folder]
        elif (annotation_folder is not None) and os.path.isdir(annotation_folder):
            file_list = os.listdir(annotation_folder)
        else:
            file_list = []
            logger.info("No celltype-annotation was found. Assigning same celltype to all cells")

        file_list = [file for file in file_list if (os.path.splitext(file)[1] in ['.csv', '.tsv'])]
        annotation_dfs_cell = []
        annotation_dfs_cs = []
        data_matrices = {}
        cell_or_cs = None
        for ind_file, filename in enumerate(file_list):
            filepath = os.path.join(annotation_folder, filename)
            ext = os.path.splitext(filepath)[1]
            annotation_df = None
            if ext == '.tsv':
                delim = '\t'
            elif ext == '.csv':
                delim = ','
            else:
                continue
            annot_input = pd.read_csv(filepath, header=0, index_col=0, delimiter=delim,
                                      dtype={0: str})  # .astype(dtype=str)
            if hasattr(self.metadata, 'nCss') and (
                    annot_input.shape[0] in [self.metadata.nCss - 1, self.metadata.nCss]):
                if annot_input.shape[0] == (self.metadata.nCss - 1):
                    annot_input = pd.read_csv(filepath, header=None, index_col=0, delimiter=delim)
                    annot_input.columns = ['Annotation {}_{}'.format(ind_file, ind) for ind in
                                           range(annot_input.shape[1])]
                cs_id_to_ind = {cell_id: ind for ind, cell_id in enumerate(annot_input.index)}
                try:
                    cs_inds = np.array([cs_id_to_ind[cs_id] for cs_id in self.metadata.csIds])
                    annotation_df = annot_input.iloc[cs_inds, :]
                    cell_or_cs = 'cs'
                except KeyError as e:
                    mp_print("The cell-ID {} was present in the data but not in annotation-file '{}'. "
                             "Please check this. "
                             "For now, we are trying to still use this annotation-file.".format(e, filename),
                             WARNING=True)
                    annotation_df = None
            elif annot_input.shape[0] in [self.metadata.nCells - 1, self.metadata.nCells]:
                if annot_input.shape[0] == (self.metadata.nCells - 1):
                    annot_input = pd.read_csv(filepath, header=None, index_col=0, delimiter=delim, dtype={0: str})
                    annot_input.columns = ['Annotation {}_{}'.format(ind_file, ind) for ind in
                                           range(annot_input.shape[1])]
                cell_id_to_ind = {cell_id: ind for ind, cell_id in enumerate(annot_input.index)}
                try:
                    cell_inds = np.array([cell_id_to_ind[cell_id] for cell_id in self.metadata.cellIds])
                    annotation_df = annot_input.iloc[cell_inds, :]
                    cell_or_cs = 'cell'
                except KeyError as e:
                    mp_print("The cell-ID {} was present in the data but not in annotation-file '{}'. "
                             "Please check this. "
                             "For now, we are trying to still use this annotation-file.".format(e, filename),
                             WARNING=True)
            else:
                mp_print("Annotation-file '{}' has a number of rows that is not equal to the number of cells "
                         "(or cellstates) in the dataset. Please check this. For now, we are trying to still use this "
                         "annotation-file".format(filename), WARNING=True)
            if annotation_df is None:
                try:
                    # Try to find out if we can save this annotation-file even though it doesn't have the correct number
                    # of rows
                    saved_df = False
                    annotation_df = annot_input.reindex(self.metadata.csIds)
                    if annotation_df.shape[0] and annotation_df.notna().any().any():
                        cell_or_cs = 'cs'
                        mp_print("Managed to save the annotation in {}. "
                                 "Do check if results are correct.".format(filename), WARNING=True)
                        saved_df = True
                    if not saved_df:
                        annotation_df = annot_input.reindex(self.metadata.cellIds)
                        if annotation_df.shape[0] and annotation_df.notna().any().any():
                            cell_or_cs = 'cell'
                            mp_print("Managed to save the annotation in {}. "
                                     "Do check if results are correct.".format(filename), WARNING=True)
                            saved_df = True
                    if not saved_df:
                        mp_print("Annotation-file '{}' has a number of rows that is not equal to the number of cells "
                                 "(or cellstates) in the dataset. Please check this. For now, we are discarding this "
                                 "annotation-file".format(filename), WARNING=True)
                        annotation_df = None
                except Exception as e:
                    mp_print("An error occurred: {}".format(e))
                    mp_print("Annotation-file '{}' can somehow not be mapped to any annotation of the cells or "
                             "cellstates in the dataset.".format(filename), WARNING=True)
                    annotation_df = None
            if annotation_df is not None:
                if filename[:4] == 'mat_':
                    mat_label = ''.join(filename[4:].split('.')[:-1])
                    data_matrices[mat_label] = annotation_df.T
                else:
                    # Determine which columns are numerical values
                    new_colnames = []
                    cols_to_add = []
                    colnames_to_add = []
                    annotation_df = annotation_df.drop(columns=annotation_df.columns[annotation_df.isna().all()])
                    for ind, col in enumerate(annotation_df.columns):
                        if np.all(annotation_df[col].apply(is_numeric)):
                            annotation_df[col] = annotation_df[col].astype(float)
                            new_colnames.append(
                                'annot_num_{}'.format(col.replace(' ', '_').replace('-', '_').replace('.', '_')))
                            if len(np.unique(annotation_df[col])) < 20:
                                # In this case, add a colname to the annotation that treats these as categorical annotation
                                cols_to_add.append(['c_{}'.format(orig_annot) for orig_annot in annotation_df[col]])
                                colnames_to_add.append('annot_categorized_{}'.format(
                                    col.replace(' ', '_').replace('-', '_').replace('.', '_')))
                        else:
                            try:
                                annotation_df[col] = annotation_df[col].fillna("NaN")
                            except:
                                print("Could not convert nans in column {}.".format(col))
                            annotation_df[col] = pd.Categorical(annotation_df[col])

                            if col.startswith("num_"):
                                col = col.replace("num_", "usernum_")

                            new_colnames.append(
                                'annot_{}'.format(col.replace(' ', '_').replace('-', '_').replace('.', '_')))
                    annotation_df.columns = new_colnames
                    for name, vector in zip(colnames_to_add, cols_to_add):
                        annotation_df[name] = vector
                    if cell_or_cs == 'cell':
                        annotation_dfs_cell.append(annotation_df)
                    else:
                        annotation_dfs_cs.append(annotation_df)

        if len(annotation_dfs_cell) == 0:
            annotation_dfs_cell = pd.DataFrame({'annot_default': np.array(['default'] * self.metadata.nCells)},
                                               index=self.metadata.cellIds)
            annotation_dfs_cell['annot_default'] = pd.Categorical(annotation_dfs_cell['annot_default'])
        else:
            annotation_dfs_cell = pd.concat(annotation_dfs_cell, axis=1)
            if annotation_dfs_cell.columns.has_duplicates:
                annotation_dfs_cell.columns = [col_name + "{}_".format(ind) for ind, col_name in
                                               enumerate(annotation_dfs_cell.columns)]
        if len(annotation_dfs_cs) == 0:
            annotation_dfs_cs = pd.DataFrame({'annot_default': np.array(['default'] * self.metadata.nCss)},
                                             index=self.metadata.csIds)
            annotation_dfs_cs['annot_default'] = pd.Categorical(annotation_dfs_cs['annot_default'])
        else:
            annotation_dfs_cs = pd.concat(annotation_dfs_cs, axis=1)
            if annotation_dfs_cs.columns.has_duplicates:
                annotation_dfs_cs.columns = [col_name + "_{}".format(ind) for ind, col_name in
                                             enumerate(annotation_dfs_cs.columns)]

        return annotation_dfs_cell, annotation_dfs_cs, data_matrices

    """---------------------------Reading and writing data---------------------––––-----------------------"""

    def data_path(self, filename=""):
        if self.metadata.pathToOrigData is None:
            if not len(filename):
                if self.metadata.dataset:
                    return os.path.abspath(os.path.join("../data", self.metadata.dataset))
                else:
                    return os.path.abspath(os.path.join("../data", 'tmp_dataset'))
            else:
                if self.metadata.dataset:
                    return os.path.abspath(os.path.join("../data", self.metadata.dataset, filename))
                else:
                    return os.path.abspath(os.path.join("../data", 'tmp_dataset', filename))
        else:
            if not len(filename):
                return os.path.abspath(self.metadata.pathToOrigData)
            else:
                return os.path.abspath(os.path.join(self.metadata.pathToOrigData, filename))

    def result_path(self, filename=""):
        if self.metadata.results_folder is None:
            if not len(filename):
                if self.metadata.dataset:
                    return os.path.abspath(os.path.join("../results", self.metadata.dataset))
                else:
                    return os.path.abspath(os.path.join("../results", 'tmp_dataset'))
            else:
                if self.metadata.dataset:
                    return os.path.abspath(os.path.join("../results", self.metadata.dataset, filename))
                else:
                    return os.path.abspath(os.path.join("../results", 'tmp_dataset', filename))
        else:
            if not len(filename):
                return os.path.abspath(self.metadata.results_folder)
            else:
                return os.path.abspath(os.path.join(self.metadata.results_folder, filename))

    # Used
    def read_in_data(self, filenamesData=None, getOrigData=False, verbose=False, noDataNeeded=False, sanityOutput=False,
                     zscoreCutoff=-1, mpiInfo=None, all_genes=False):
        if mpiInfo is None:
            mpiInfo = mpi_wrapper.get_mpi_info()
        originalData = OriginalData()
        # Determine correct filenames
        if filenamesData is None:
            filenameMeans = 'delta_vmax.txt'
            filenameStds = 'd_delta_vmax.txt'
        else:
            filenamesData = filenamesData.split(',')
            filenameMeans = filenamesData[0]
            if len(filenamesData) > 1:
                filenameStds = filenamesData[1]
            else:
                filenameStds = None

        try:
            tmp_folder = self.result_path('processed_data_communication')
            if (mpiInfo.size > 1) and (mpiInfo.rank == 0):
                Path(tmp_folder).mkdir(parents=True, exist_ok=True)
            originalData.ltqs, originalData.ltqsVars, originalData.geneVariances, \
            self.metadata.nCells, \
            self.metadata.nGenes, genes_to_keep, ltqStdsFound, \
            n_genes_orig = read_and_filter(self.data_path(), filenameMeans, filenameStds, sanityOutput,
                                           zscoreCutoff, mpiInfo, tmp_folder=tmp_folder, verbose=verbose,
                                           all_genes=all_genes)

        except FileNotFoundError:
            if noDataNeeded:
                print("filenamesData: {}".format(filenamesData))
                # mp_print("WARNING: File {} not found.".format(self.data_path(filenameMeans)), WARNING=True)
                # mp_print("WARNING: File {} not found.".format(self.data_path(filenameStds)), WARNING=True)
                mp_print("Did not find data-file, but is not (strictly) needed now.", WARNING=True)
                originalData.ltqs = None
                originalData.ltqsVars = None
                self.metadata.nCells = None
                self.metadata.nGenes = None
                genes_to_keep = None
                ltqStdsFound = False
                n_genes_orig = None
            else:
                # mp_print("WARNING: File {} not found.".format(self.data_path(filenameMeans)), WARNING=True)
                # mp_print("WARNING: File {} not found.".format(self.data_path(filenameStds)), WARNING=True)
                exit("Data-file was not found!")
        if mpiInfo.rank != 0:
            return None, None

        # Reading in gene-identifiers (or other sc-features)
        made_up_gene_ids = False
        if os.path.exists(self.data_path("geneID.txt")):
            self.metadata.geneIds = []
            with open(self.data_path(os.path.join('geneID.txt')), 'r') as file:
                reader = csv.reader(file, delimiter="\t")
                for row in reader:
                    self.metadata.geneIds.append(row[0])
            if (n_genes_orig is not None) and (len(self.metadata.geneIds) != n_genes_orig):
                print("Number of gene-Ids in file geneID.txt does not match number of features in data-matrix!")
                exit()
        elif os.path.exists(self.data_path('log_transcription_quotients.txt')):
            geneIds = pd.read_csv(self.data_path('log_transcription_quotients.txt'), header=None, usecols=[0],
                                  sep='\t')[1:].values.flatten()
            self.metadata.geneIds = [ID for ID in geneIds]
            if (n_genes_orig is not None) and (len(self.metadata.geneIds) != n_genes_orig):
                print("Number of gene-Ids in file log_transcription_quotients.txt does not match genes in data-matrix!")
                exit()
        else:
            mp_print("No geneID-file was found. Giving generic names.")
            if originalData.ltqs is None:
                self.metadata.nGenes = 10  # Make up 10 fake genes, not used anywhere
            self.metadata.geneIds = ['Gene_' + str(ind) for ind in range(self.metadata.nGenes)]
            made_up_gene_ids = True
        if genes_to_keep is not None:
            self.metadata.geneIds = list(
                np.array(self.metadata.geneIds)[genes_to_keep]) if not made_up_gene_ids else self.metadata.geneIds

        if self.metadata.nGenes is None:
            self.metadata.nGenes = len(self.metadata.geneIds)

        self.metadata.cellIds = []
        forbidden = {":", ",", "(", ")", ";", ' '}  # Otherwise, writing and reading to Nwk-string fails
        warned = False
        if os.path.exists(self.data_path('cellID.txt')):
            with open(self.data_path(os.path.join('cellID.txt')), 'r') as file:
                reader = csv.reader(file, delimiter="\t")
                for row in reader:
                    cell_id = row[0]
                    if any(c in cell_id for c in forbidden):
                        logger.error("At least one cell-ID: {} contains a character that is problematic for"
                                     "writing and reading Nwk-strings: ':', ',' '(', ')',';'.\n"
                                     "Please replace these characters, for example by underscores: '_'.\n"
                                     "Note that you should also do these "
                                     "replacements in potential annotation-files.".format(cell_id))
                        exit()
                        # if not warned:
                        #     logger.warning("At least one cell-ID: {} contains a character that is problematic for"
                        #                    "writing and reading Nwk-strings: ':', ',' '(', ')',';'.\n"
                        #                    "We'll replace these characters by underscores in the output.")
                        #     warned = True
                        # for c in forbidden:
                        #     cell_id = cell_id.replace(c, "_")
                    self.metadata.cellIds.append(cell_id)
            n_unq_ids = len(np.unique(self.metadata.cellIds))
            if self.metadata.nCells is not None:
                if (n_unq_ids != self.metadata.nCells) or (len(self.metadata.cellIds) != self.metadata.nCells):
                    mp_print("Number of unique object-IDs given in {} does not "
                             "match the number of columns in your feature"
                             "matrix.\n "
                             "Please resolve this and start Bonsai again.".format(self.data_path('cellID.txt')),
                             ERROR=True)
                    exit()
        else:
            mp_print("No cellID-file was found. Giving generic names.")
            if self.metadata.nCells is None:
                self.metadata.cellIds = None
            else:
                self.metadata.cellIds = ['Cell_' + str(ind) for ind in range(self.metadata.nCells)]
        if (self.metadata.nCells is None) and (self.metadata.cellIds is not None):
            self.metadata.nCells = len(self.metadata.cellIds)

        self.originalData = originalData if getOrigData else None
        return ltqStdsFound, originalData

    # Used
    def read_in_data_no_ltqs(self):
        # Reading in gene-identifiers (or other sc-features)
        if os.path.exists(self.data_path("geneID.txt")):
            self.metadata.geneIds = []
            with open(self.data_path(os.path.join('geneID.txt')), 'r') as file:
                reader = csv.reader(file, delimiter="\t")
                for row in reader:
                    self.metadata.geneIds.append(row[0])
        elif os.path.exists(self.data_path('log_transcription_quotients.txt')):
            geneIds = pd.read_csv(self.data_path('log_transcription_quotients.txt'), header=None, usecols=[0],
                                  sep='\t')[1:].values.flatten()
            self.metadata.geneIds = [ID for ID in geneIds]
        self.metadata.nGenes = len(self.metadata.geneIds)

        self.metadata.cellIds = []
        if os.path.exists(self.data_path('cellID.txt')):
            with open(self.data_path(os.path.join('cellID.txt')), 'r') as file:
                reader = csv.reader(file, delimiter="\t")
                for row in reader:
                    self.metadata.cellIds.append(row[0])
        else:
            mp_print("No cellID-file was found. Giving generic names.")
            self.metadata.cellIds = ['Cell_' + str(ind) for ind in range(self.metadata.nCells)]
        self.metadata.nCells = len(self.metadata.cellIds)

    # Should go to tree visualisation file
    def read_umi_counts(self, filename_data=None):
        if filename_data is None:
            filename_data = 'Gene_table.txt'
        try:
            tmp = pd.read_csv(self.data_path(filename_data), sep='\t', index_col=0)
            self.umiCounts = tmp.values.astype(dtype='int')
            if self.umiCounts.shape[0] != self.metadata.nGenes:
                self.throw_out_zeros()
                if self.umiCounts.shape[0] != self.metadata.nGenes:
                    exit("Umi-count file does not have the same number as genes as the rest of the data.")
        except FileNotFoundError:
            print("Warning, not an error: No UMI-count file was found.")
            self.umiCounts = None

    # Should go to tree visualisation file
    def throw_out_zeros(self):
        nonzero_inds = np.sum(self.umiCounts, axis=1) != 0
        self.umiCounts = self.umiCounts[nonzero_inds, :]

    # Used
    def getGeneVariances(self, originalData, posterior_data=None, ltqStdsFound=False):
        if originalData.geneVariances is not None:
            return

        if posterior_data is None:
            posterior_data = originalData
        if ltqStdsFound:
            originalData.geneVariances = np.var(posterior_data.ltqs, axis=1) + np.mean(posterior_data.ltqsVars,
                                                                                       axis=1)
            # Old options for calculating the variances are below
            # pjotrs = infer_true_var_log(originalData.ltqs, originalData.ltqsVars)
            # # Infer variance from fixed point procedure
            # C = self.metadata.nCells
            # newGeneVars = np.zeros(self.metadata.nGenes)
            # ltqsSq = originalData.ltqs ** 2
            # for gene in range(self.metadata.nGenes):
            #     if gene % 1000 == 0:
            #         mp_print("Optimising gene variance for gene " + str(gene))
            #     vg0 = originalData.geneVariances[gene]
            #     ltqsVarsG = originalData.ltqsVars[gene, :]
            #     lb = ltqsVarsG.max() + 1e-6
            #     optRes = minimize(logLGradDepOnV, vg0, args=(C, ltqsSq[gene, :], ltqsVarsG), jac=True,
            #                       bounds=[(lb, None)])
            #
            #     newGeneVars[gene] = optRes.x
        elif posterior_data.ltqs is not None:
            originalData.geneVariances = np.var(posterior_data.ltqs, axis=1)
        else:
            originalData.geneVariances = None

    def cleanup_node_inds(self):
        if self.metadata.cellIds is None:
            mp_print("Cannot cleanup node-indices without defined metadata.cellIds.", WARNING=True)
            return
        cell_id_to_ind = {cell_id: ind for ind, cell_id in enumerate(self.metadata.cellIds)}
        nodes_list = self.tree.root.getNodeList([], returnRoot=True, returnLeafs=True)
        bs_glob.max_node_ind = -1
        n_cells = 0
        for node in nodes_list:
            if node.nodeId in cell_id_to_ind:
                n_cells += 1
                node.nodeInd = cell_id_to_ind[node.nodeId]
                node.isCell = True
                bs_glob.max_node_ind = max(node.nodeInd, bs_glob.max_node_ind)

        # bs_glob.max_node_ind = np.max(list(cell_id_to_ind.values()))
        self.tree.max_node_ind = bs_glob.max_node_ind
        self.metadata.nCells = n_cells
        bs_glob.nCells = self.metadata.nCells
        self.tree.root.renumberNodes(change_node_inds=True)


def do_spr_moveset(scdata_path, args, strategy='determ_random_determ', pickup_intermediate=False):
    """

    :param strategy: Current options are 'large_tree', 'long_branch_sorted', 'super_sure'.
    This relates to what kind of SPR-moves we propose in what order.
    :return:
    """
    mpi_info = mpi_wrapper.get_mpi_info()
    # info_dict = None
    # if (mpi_info.rank == 0) and tracking:
    #     info_dict = {'steps': [], 'runtimes': [], 'd_loglik': [], 'successes': []}
    #     orig_loglik = scData.tree.calcLogLComplete(mem_friendly=True)
    #     orig_time = time.time()

    output_prefixes = ['after_sprmoveset{}'.format(ind) for ind in range(1, 3)]
    output_present = [False] * len(output_prefixes)
    if pickup_intermediate:
        for ind, prefix in enumerate(output_prefixes):
            # Check whether one of the intermediate paths is already there
            output_present[ind], scdata_path_new = look_for_output_folder('spr_intermediates',
                                                                          general_results_folder=args.results_folder,
                                                                          prefix=prefix)
            if output_present[ind]:
                scdata_path = scdata_path_new

    if strategy == 'large_tree':
        if output_present[0]:
            mp_print("Found results for first SPR moveset in {}. Skipping first moveset.".format(scdata_path))
        else:
            scData = do_spr_moves_with_postprocessing(scdata_path, args=args,
                                                      select_cand='long_branches_first',
                                                      select_target='cluster_centers',
                                                      max_moves=None,
                                                      min_branch_length=1e-2,
                                                      large_tree=True)

            scdata_path = store_scdata_and_communicate_path(scData, specific_results_folder='spr_intermediates',
                                                            general_results_folder=args.results_folder,
                                                            prefix=output_prefixes[0])
            scData = None
            gc.collect()

        scData = do_spr_moves_with_postprocessing(scdata_path, args=args,
                                                  select_cand='long_branches_first',
                                                  select_target='cluster_centers',
                                                  max_moves=None,
                                                  min_branch_length=1e-2,
                                                  large_tree=True)
    else:
        if mpi_info.rank != 0:
            mp_print("Can't contribute to doing SPR moves in any other mode than 'large_tree'. "
                     "Choose --spr_strategy='large_tree' to benefit from parallelization "
                     "(at the cost of some memory usage) for the SPR-moves."
                     "Now, this process will wait to see whether it can help in the postprocessing.", ALL_RANKS=True)

    if strategy == 'long_branch_sorted':
        if output_present[0]:
            mp_print("Found results for first SPR moveset in {}. Skipping first moveset.".format(scdata_path))
        else:
            scData = do_spr_moves_with_postprocessing(scdata_path, args=args,
                                                      select_cand='long_branches_first',
                                                      select_target='cluster_centers',
                                                      max_moves=None,
                                                      min_branch_length=1e-2)

            scdata_path = store_scdata_and_communicate_path(scData, specific_results_folder='spr_intermediates',
                                                            general_results_folder=args.results_folder,
                                                            prefix=output_prefixes[0])
            scData = None
            gc.collect()

        # if output_present[1]:
        #     mp_print("Found results for second SPR moveset in {}. Skipping first moveset.".format(scdata_path))
        # else:
        # scData = do_spr_moves_with_postprocessing(scdata_path, args=args,
        #                                                   select_cand='random',
        #                                                   select_target='random',
        #                                                   max_moves=None,
        #                                                   moves_id='C_random_T_random',
        #                                                   tracking=tracking, info_dict=info_dict)
        # scdata_path = store_scdata_and_communicate_path(scData, specific_results_folder='spr_intermediates',
        #                                                 general_results_folder=args.results_folder,
        #                                                 prefix=output_prefixes[1])
        # scData = None
        # gc.collect()
        # scData = None
        # gc.collect()

        scData = do_spr_moves_with_postprocessing(scdata_path, args=args,
                                                  select_cand='long_branches_first',
                                                  select_target='cluster_centers',
                                                  max_moves=None,
                                                  min_branch_length=1e-2)

    elif strategy == 'super_sure':
        if output_present[0]:
            mp_print("Found results for first SPR moveset in {}. Skipping first moveset.".format(scdata_path))
        else:
            scData = do_spr_moves_with_postprocessing(scdata_path, args=args,
                                                      select_cand='long_branches_first',
                                                      select_target='cluster_centers',
                                                      max_moves=None)

            scdata_path = store_scdata_and_communicate_path(scData, specific_results_folder='spr_intermediates',
                                                            general_results_folder=args.results_folder,
                                                            prefix=output_prefixes[0])
            scData = None
            gc.collect()

        if output_present[1]:
            mp_print("Found results for second SPR moveset in {}. Skipping first moveset.".format(scdata_path))
        else:
            scData = do_spr_moves_with_postprocessing(scdata_path, args=args,
                                                      select_cand='long_branches_first',
                                                      select_target='all',
                                                      max_moves=1000)

            scdata_path = store_scdata_and_communicate_path(scData, specific_results_folder='spr_intermediates',
                                                            general_results_folder=args.results_folder,
                                                            prefix=output_prefixes[1])
            scData = None
            gc.collect()

        scData = do_spr_moves_with_postprocessing(scdata_path, args=args,
                                                  select_cand='long_branches_first',
                                                  select_target='cluster_centers',
                                                  max_moves=None)

    return scData


def do_spr_moves_with_postprocessing(scdata_path, args, select_cand, select_target, max_moves=None,
                                     min_branch_length=-1, large_tree=False, parallel=False):
    """

    :param args:
    :param select_cand:
    :param select_target:
    :param max_moves:
    :param min_branch_length:
    :param large_tree: If True, SPR moves in a sequence are done normally until we drop below 10 successes in 1000
    attempts. After that, we start scanning all moves without executing successes, because this can be parallelized.
    Finally, we take only the successes and execute them in order of predicted dLogL
    :return:
    """
    mpi_info = mpi_wrapper.get_mpi_info()
    if max_moves is None:
        max_moves = bs_glob.nNodes

    # Just some tracking
    # if (mpi_info.rank == 0) and tracking:
    #     start_logl = scData.tree.calcLogLComplete(mem_friendly=True)
    #     mp_print("Loglikelihood of inferred tree before "
    #              "{}: {}".format(moves_id, start_logl))
    #     start = time.time()

    # The actual moves!
    if not large_tree:
        if mpi_info.rank != 0:
            mp_print("This process will only contribute to SPR-postprocessing.", ALL_RANKS=True)
            scData = None
        else:
            scData = loadReconstructedTreeAndData(args, scdata_path,
                                                  reprocess_data=False, all_genes=False, get_cell_info=False,
                                                  all_ranks=True, rel_to_results=False, verbose=False)

            successful_moves, total_dlogl, _ = scData.tree.do_spr_moves(max_moves=max_moves,
                                                                        select_cand=select_cand,
                                                                        select_target=select_target,
                                                                        min_branch_length=min_branch_length)
    else:
        # In this case, do three phases:
        # Phase 1: only moves are performed until they occur at low frequency only,
        # Phase 2: no moves are performed, but we scan all candidates and store successful candidates
        # Phase 3: the successful candidates are sorted for their predicted loglikelihood and executed

        # Phase 1: only moves are performed until they occur at low frequency only,
        # intermediate_folder = scData.result_path('spr_intermediate_{}'.format(np.random.randint(1e6)))
        if mpi_info.rank == 0:
            scData = loadReconstructedTreeAndData(args, scdata_path,
                                                  reprocess_data=False, all_genes=False, get_cell_info=False,
                                                  all_ranks=True, rel_to_results=False, single_process=True,
                                                  verbose=False)

            # To save on memory, do the preparing tree already here, after that store and load the tree again. The
            # reason is that ltqs at internal nodes will be mem-mapped in this way, rather than read into RAM
            orig_loglik = scData.tree.prepare_tree_spr_moves()
        else:
            scData = None

        # Store tree again, and communicate tree-folder with all processes.
        scdata_path = store_scdata_and_communicate_path(scData, specific_results_folder='spr_intermediates')
        scData = None
        gc.collect()

        if mpi_info.rank == 0:
            scData = loadReconstructedTreeAndData(args, scdata_path,
                                                  reprocess_data=False, all_genes=False, get_cell_info=False,
                                                  all_ranks=False, rel_to_results=False, single_process=True,
                                                  verbose=False)

            # Do the first SPR moves (coming at high freq) only at first process
            successful_moves, total_dlogl, remain_cands = scData.tree.do_spr_moves(max_moves=max_moves,
                                                                                   select_cand=select_cand,
                                                                                   select_target=select_target,
                                                                                   min_branch_length=min_branch_length,
                                                                                   freq_cutoff=.05, mem_friendly=True,
                                                                                   skip_prepare_tree=True)

            # Prepare the tree already for the next phase, do this only on Process 0
            orig_loglik = scData.tree.prepare_tree_spr_moves()

            # Store the resulting tree with coordinates, such that it can be read in by other processes
            # mp_print("Before storing tree, memory usage is ",
            #          psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ALL_RANKS=True)
            # scData.storeTreeInFolder(os.path.join(intermediate_folder), with_coords=True,
            #                          verbose=False, nwk=False)

            # Communicate the node indices of the remaining candidates. This also serves as a barrier between
            # storing the tree and loading it on other processes
            df_nodes_list = scData.tree.root.getNodeList([], returnRoot=True, returnLeafs=True)
            remain_cands_set = set(remain_cands)
            remain_cands_df_sorted = [node.nodeInd for node in df_nodes_list if node.nodeInd in remain_cands_set]

            remain_cands = np.array(remain_cands_df_sorted)
            remain_cands = mpi_wrapper.bcast(remain_cands, root=0)
            # intermediate_folder = mpi_wrapper.bcast(intermediate_folder, root=0)
        else:
            remain_cands = None
            # Other processes receive the node-indices that need to be checked
            remain_cands = mpi_wrapper.bcast(remain_cands, root=0)
            # intermediate_folder = mpi_wrapper.bcast(intermediate_folder, root=0)
            # Then they read the tree that was stored from process 0

        scdata_path_parallelphase = store_scdata_and_communicate_path(scData,
                                                                      specific_results_folder='spr_intermediates')

        # print_memory("Stored the tree before parallel-phase of SPR moves")
        scData = None
        if mpi_info.rank == 0:
            del df_nodes_list
        gc.collect()
        # print_memory("Removed old scData-object before parallel-phase of SPR moves")

        scData = loadReconstructedTreeAndData(args, scdata_path_parallelphase,
                                              reprocess_data=False, all_genes=False, get_cell_info=False,
                                              all_ranks=True, rel_to_results=False, verbose=False)
        # scData.tree.root.storeParent()
        # print_memory("Loaded fresh scData from file before parallel-phase of SPR.")

        # Now, every process should have the full tree-result of the first SPR moves. Ready to start Phase 2
        # Phase 2: no moves are performed, but we scan all candidates and store successful candidates

        # First, distribute node-inds to check over the cores. We'll do this by cutting a depth-first list up in
        # blocks. Eventually, this may help keep memory-usage friendly
        my_tasks = getMyTaskNumbers(len(remain_cands), mpi_info.size, mpi_info.rank, skippingSteps=False)
        my_tasks = remain_cands[my_tasks]  # This now contains the node-inds for candidates to be treated per proc.

        # Scan over these candidates
        negdlogl_nodeind_tuples = scData.tree.do_spr_moves(select_cand='list', select_target=select_target,
                                                           cand_list=my_tasks, only_scan=True, mem_friendly=True,
                                                           skip_prepare_tree=True)
        # The variable negdlogl_nodeind_tuples should contain the successful-candidates that were found on this core,
        # as a list of tuples (-d_log_l, cand_node_ind)

        # Sorting the candidates calculated on this process
        if len(negdlogl_nodeind_tuples):
            negdlogl_nodeind_tuples.sort(key=lambda x: x[0], reverse=False)
        else:
            negdlogl_nodeind_tuples = []

        # Gather the succesful candidates from all processes
        if mpi_info.size > 1:
            all_negdlogl_nodeind_tuples = mpi_wrapper.gather(negdlogl_nodeind_tuples, root=0)
            if mpi_info.rank != 0:
                # The memory on these processes can be freed, as they will be idle for a while
                try:
                    del negdlogl_nodeind_tuples
                    del scData
                    gc.collect()
                except UnboundLocalError:
                    pass
                mp_print("My job in the SPR-moves is done. Will wait to see if I can help in postprocessing.",
                         ALL_RANKS=True)
                scData = None
            else:
                # So only process zero goes here. It merges all the sorted lists using the heapq.merge algorithm
                negdlogl_nodeind_tuples = list(heapq.merge(*all_negdlogl_nodeind_tuples))

        if mpi_info.rank == 0:
            # Phase 3: the successful candidates are sorted for their predicted loglikelihood and executed
            # OPTIONAL: Re-load tree that was stored as intermediate, because then we the scanning behavior didn't change
            # the ordering of children, which makes it perfectly comparable when running on different numbers of cores
            del scData
            gc.collect()
            # print_memory('Deleted scData after parallel phase')
            scData = loadReconstructedTreeAndData(args, scdata_path_parallelphase,
                                                  reprocess_data=False, all_genes=False, get_cell_info=False,
                                                  all_ranks=True, rel_to_results=False, single_process=True,
                                                  verbose=False)
            # scData.tree.root.storeParent()
            # print_memory("Loaded tree again after parallel phase")

            my_tasks = [node_ind_dlogl[1] for node_ind_dlogl in negdlogl_nodeind_tuples]
            successful_moves, total_dlogl, remain_cands = scData.tree.do_spr_moves(select_cand='list',
                                                                                   select_target=select_target,
                                                                                   cand_list=my_tasks, only_scan=False,
                                                                                   mem_friendly=True,
                                                                                   skip_prepare_tree=True)

    # Just some tracking
    # if tracking:
    #     info_dict['steps'].append(moves_id)
    #     info_dict['runtimes'].append(time.time() - start)
    #     new_logl = scData.tree.calcLogLComplete(mem_friendly=True)
    #     info_dict['d_loglik'].append(new_logl - start_logl)
    #     info_dict['successes'].append(successful_moves)
    #     mp_print("Loglikelihood of inferred tree after "
    #              "{} successful SPR-moves: {}".format(successful_moves, new_logl))

    # start = time.time()
    # start_logl = new_logl

    # All processes should come togethere here.
    mp_print("Start round of postprocessing after SPR-moves.", ALL_RANKS=True)
    scData = extensive_spr_postprocessing(args, scData, verbose=large_tree)
    # print_memory("After round of postprocessing after SPR-moves.")

    # Just some tracking
    # if tracking:
    #     info_dict['steps'].append(moves_id + "_postproc")
    #     info_dict['runtimes'].append(time.time() - start)
    #     new_logl = scData.tree.calcLogLComplete(mem_friendly=True)
    #     info_dict['d_loglik'].append(new_logl - start_logl)
    #     info_dict['successes'].append(0)
    #     mp_print("Loglikelihood of inferred tree after postprocessing: {}".format(new_logl))
    return scData


def extensive_spr_postprocessing(args, scdata, verbose=False):
    mpi_info = mpi_wrapper.get_mpi_info()
    # First do smaller postprocessing just on zero-th process
    if mpi_info.rank == 0:
        scdata.tree.do_spr_postprocessing(verbose=verbose, only_cleanup=False, only_time_opt=True)

    # Then communicate the tree-topology (without the data coordinates), necessary to resolve polytomies
    scdata_path_postproc = store_scdata_and_communicate_path(scdata, specific_results_folder='spr_intermediates')
    if (mpi_info.size > 1) and (mpi_info.rank != 0):
        scdata = loadReconstructedTreeAndData(args, scdata_path_postproc, reprocess_data=False, all_genes=False,
                                              get_cell_info=False, all_ranks=False, rel_to_results=True)
    else:
        # Get as-if-root information, necessary for resolving polytomies
        scdata.tree.root.getAIRootInfo(None, None)
    # Make sure, node counts are correct on all processes
    scdata.tree.root.renumberNodes(change_node_inds=False)
    scdata.tree.nNodes = bs_glob.nNodes

    # Resolve polytomies
    scdata.tree.root.mergeChildrenRecursive(scdata.tree.root.ltqs, scdata.tree.root.getW(), sequential=False,
                                            verbose=verbose, ellipsoidSize=1.0, single_process=False,
                                            mergeDownstream=True, tree=scdata.tree, nChildNN=50, kNN=10)
    if mpi_info.rank != 0:
        mp_print("Now, my job in the SPR postprocessing is also done. Returning to main function.", ALL_RANKS=True)
        del scdata
        gc.collect()
        return None

    # Only the first process goes here.
    scdata.tree.root.renumberNodes(change_node_inds=False)  # Some nodes were merged, need to re-count the nodes
    scdata.tree.nNodes = bs_glob.nNodes
    scdata.tree.root.storeParent()
    scdata.tree.optTimes(verbose=verbose, singleProcess=True, mem_friendly=True, maxiter=100, tol=1e-4)

    return scdata

    # Used
    # def filter_variable_genes(self, originalData, zscoreCutoff=-1, nGenesToKeep=-1, verbose=False):
    #     if (zscoreCutoff > 0) or (nGenesToKeep > 0):
    #         # Compute zscores for all genes.
    #         # zscores = np.sqrt(np.maximum(np.mean(((originalData.ltqs - np.mean(originalData.ltqs, axis=1)[:,
    #         #                                                            np.newaxis]) ** 2 - originalData.ltqsVars)
    #         #                                      / originalData.ltqsVars, axis=1), 1e-12))
    #         zscores = np.sqrt(np.maximum(np.mean(
    #             ((originalData.ltqs - np.mean(originalData.ltqs, axis=1)[:, np.newaxis]) ** 2) / originalData.ltqsVars,
    #             axis=1), 1e-12))
    #         if zscoreCutoff > 0:
    #             genesToKeep = np.where(zscores > zscoreCutoff)[0]
    #             mp_print("Z-score cutoff is: %f, %d genes are kept." % (zscoreCutoff, len(genesToKeep)))
    #         else:
    #             nGenesToKeep = min(nGenesToKeep, self.metadata.nGenes)
    #             ordered_genes = np.argsort(-zscores)
    #             genesToKeep = ordered_genes[:nGenesToKeep]
    #         if verbose:
    #             mp_print("%d genes are kept, gene with lowest z-score becomes %f." % (
    #                 nGenesToKeep, -np.sort(-zscores)[nGenesToKeep - 1]))
    #     else:
    #         genesToKeep = np.array(range(self.metadata.nGenes))
    #     originalData.ltqs = originalData.ltqs[genesToKeep, :]
    #     originalData.ltqsVars = originalData.ltqsVars[genesToKeep, :]
    #     if originalData.geneVariances is not None:
    #         originalData.geneVariances = originalData.geneVariances[genesToKeep]
    #         # originalData.trueVariances = originalData.trueVariances[genesToKeep]
    #     if originalData.priorVariances is not None:
    #         originalData.priorVariances = originalData.priorVariances[genesToKeep]
    #     self.metadata.nGenes = len(genesToKeep)
    #     self.metadata.geneIds = list(np.array(self.metadata.geneIds)[genesToKeep])

    # def calc_euclidean_distances(self, distsFile=None):
    #     avgDists = pdist(self.ltqs.T, metric='sqeuclidean') / self.nGenes
    #     mpiRank = mpi_wrapper.get_process_rank()
    #     if (mpiRank == 0) and distsFile:
    #         np.savetxt(self.data_path(distsFile), avgDists)
    #     return avgDists


# Used
def nnnReorderRandom(args, outputFolder, verbose=False, randomMoves=0,
                     veryVerbose=False, randomTries=None, resultsFolder=None):
    """
    In this function we take single edges in the tree and consider all nodes connected to it. We start by
    calculating the tree likelihood for the current configuration. Then we delete the edge such that all nodes
    are connected to one root. Then we just do our usual (mergeChildrenUB) to find the best configuration. We
    compare the resulting tree likelihood with the previous one. Then we either loop over all edges and take the
    best few reconfigurations, or we just do a reconfiguration whenever we find a good one.
    :param verbose:
    :return:
    """
    mpiInfo = mpi_wrapper.get_mpi_info()
    tmp_folder = os.path.join(resultsFolder, 'intermediate_trees')
    stored_tree_ind = None
    if mpiInfo.rank == 0:
        found_intermediate = False
        if args.pickup_intermediate and os.path.exists(tmp_folder):
            intermediateFolders = os.listdir(tmp_folder)
            if len(intermediateFolders):
                intermediateFolder, stored_tree_ind = get_latest_intermediate(intermediateFolders, base='nnn')
                if intermediateFolder is not None:
                    # scData.tree = unpickleTree(tmp_folder, intermediateFile)
                    scData = loadReconstructedTreeAndData(args, os.path.join(tmp_folder, intermediateFolder),
                                                          reprocess_data=False, all_genes=False, get_cell_info=False,
                                                          all_ranks=False, rel_to_results=False)
                    randomMoves = 0
                    found_intermediate = True
            elif os.path.exists(os.path.join(resultsFolder, 'random_trees', 'orig_tree')):
                scData = loadReconstructedTreeAndData(args, os.path.join(resultsFolder, 'random_trees', 'orig_tree'),
                                                      reprocess_data=False, all_genes=False, get_cell_info=False,
                                                      all_ranks=False, rel_to_results=False)
                found_intermediate = True

        if not found_intermediate:
            scData = loadReconstructedTreeAndData(args, outputFolder, reprocess_data=False, all_genes=False,
                                                  get_cell_info=False, all_ranks=False, rel_to_results=True)
        Path(tmp_folder).mkdir(parents=True, exist_ok=True)
        scData.tree.root.mergeZeroTimeChilds()
        scData.tree.root.renumberNodes(change_node_inds=False)
        scData.tree.nNodes = bs_glob.nNodes

        # First make sure every node knows where its parent node is
        # mp_print("Before storing parent, memory usage is ",
        #          psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ALL_RANKS=True)
        scData.tree.root.storeParent()
        # mp_print("Before getting ltqs, memory usage is ",
        #          psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ALL_RANKS=True)
        scData.tree.root.getLtqsComplete(mem_friendly=True)
        origLoglik = scData.tree.calcLogLComplete(mem_friendly=True, loglikVarCorr=scData.metadata.loglikVarCorr,
                                                  recalc=False)
        scData.metadata.loglik = origLoglik

    randomMoves = mpi_wrapper.bcast(randomMoves, root=0)
    if randomMoves == 0:
        randomTries = 0
    else:
        if (randomTries is None) or ((randomTries > 0) and (randomTries < mpiInfo.size)):
            randomTries = mpiInfo.size
    np.random.seed(42)
    seeds = np.random.choice(int(randomTries * 1e4), size=randomTries, replace=False)
    # Start by doing random moves. Different move series are done in parallel.
    # First create a copy of the original tree from which each move series will start
    random_folder = os.path.join(resultsFolder, 'random_trees')
    # mp_print("Before storing tree, memory usage is ",
    #          psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ALL_RANKS=True)
    if mpiInfo.rank == 0:
        # pickleTree(pickleFolder, 'orig_tree.dat', self.tree)
        scData.storeTreeInFolder(os.path.join(random_folder, 'orig_tree'), with_coords=True, verbose=False, nwk=False)
        scData = None
        gc.collect()
    mpi_wrapper.barrier()

    myTasks = np.arange(mpiInfo.rank, randomTries, mpiInfo.size)
    treeLogliks = {}

    for task in myTasks:
        moveCounter = 0
        np.random.seed(seeds[task])
        # First load the original tree again
        # mp_print("Before loading tree, memory usage is ",
        #          psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ALL_RANKS=True)
        # self.tree = unpickleTree(pickleFolder, 'orig_tree.dat')
        # noinspection PyMethodFirstArgAssignment
        scData = loadReconstructedTreeAndData(args, os.path.join(random_folder, 'orig_tree'),
                                              reprocess_data=False, all_genes=False, get_cell_info=False,
                                              all_ranks=True, rel_to_results=False)
        # scData.tree.root.storeParent()
        # mp_print(scData.metadata.loglik, scData.tree.root.ltqs[:5], ALL_RANKS=True)
        currLoglik = scData.metadata.loglik
        # Then do random moves on this copy of the tree
        while moveCounter < randomMoves:
            # if moveCounter % 100 == 0:
            #     mp_print("After %d random moves, memory usage is " % moveCounter,
            #              psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ALL_RANKS=True)
            # TODO: Optimize the following couple of lines
            # Create a list of nodes that are not the root and not leafs
            nodesList = recursionWrap(scData.tree.root.getNodeList, [], returnLeafs=True,
                                      returnRoot=True)  # TODO: Fast to maintain?
            nodeIndToNode = {node.nodeInd: node for node in nodesList}
            candidatesList = [node for node in nodesList if (not node.isLeaf) and (not node.isRoot)]  # TODO: Same
            nCandidates = len(candidatesList)

            randInd = np.random.randint(nCandidates)
            # Make sure this node knows what its ltqs and ltqsVars would be if it would be the root
            # candidatesList[randInd].getAIRootUpstream()
            nnnLoglik, optEdgeList, mostUSInfo = scData.tree.getOptEdgeList(candidatesList[randInd], args,
                                                                            finalOptTimes=True,
                                                                            returnEdgelist=True, random=True,
                                                                            trackCloseness=False, singleProcess=True,
                                                                            mem_friendly=True)

            if optEdgeList is None:
                if verbose:
                    cand = candidatesList[randInd]
                    message = "node %d" % cand.nodeInd if cand.parentNode is None \
                        else "nodes %d and %d" % (cand.nodeInd, cand.parentNode.nodeInd)
                    mp_print("Random reconfiguration around %s is skipped because of many connected nodes." % message,
                             ALL_RANKS=True)
                continue
            # Process optimal reordering in the original tree
            scData.tree.processOptReconfig(optEdgeList, nodeIndToNode, nodesList, candidatesList, mostUSInfo,
                                           trackCloseness=False)
            currLoglik += nnnLoglik
            if veryVerbose:
                mp_print("Move %d: Random reconfiguration of the nodes around node %d "
                         "changed the loglikelihood by %f" % (moveCounter, randInd, nnnLoglik))
            if verbose and (moveCounter % 1000 == 0):
                mp_print(
                    "Tree %d, Move %d: Random reconfiguration of next-nearest neighbours. Loglik now: %f" % (
                        task, moveCounter, currLoglik), ALL_RANKS=True)
            moveCounter += 1

        # Store tree loglikelihood after this random series of moves
        scData.metadata.loglik = scData.tree.calcLogLComplete(mem_friendly=True,
                                                              loglikVarCorr=scData.metadata.loglikVarCorr)
        treeLogliks[task] = scData.metadata.loglik
        # Store this tree in a tmp-file
        # pickleTree(nnn_folder, 'randomTree_%d' % task, self.tree)
        scData.storeTreeInFolder(os.path.join(random_folder, 'random_tree_%d' % task), with_coords=False,
                                 verbose=verbose, all_ranks=True)
        scData = None
        gc.collect()
        # mp_print("After storing random tree, memory usage is ",
        #          psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ALL_RANKS=True)
        if verbose:
            mp_print("Process %d made tree number %d by doing %d random moves with a loglikelihood of %f." % (
                mpiInfo.rank, task, moveCounter, treeLogliks[task]), ALL_RANKS=True)

    if randomTries > 0:
        # Random bunch of trees has been created. Now select the best one
        allTreeLogliks = mpi_wrapper.gather(treeLogliks, root=0)
        if mpiInfo.rank == 0:
            for oneTreeLogliks in allTreeLogliks:
                treeLogliks.update(oneTreeLogliks)
            tasks, logliks = zip(*treeLogliks.items())
            bestTreeInd = np.argmax(logliks)
            max_rand_loglik = logliks[bestTreeInd]
            take_original_tree = False
            if max_rand_loglik < origLoglik:
                mp_print("Best random tree still has lower likelihood than the original tree. This is probably normal "
                         "and desired behavior, but maybe check if something didn't go terribly wrong.", WARNING=True)
                take_original_tree = True
            if verbose and (not take_original_tree):
                mp_print("Taking tree number %d. The random moves increased the loglikelihood from %f to %f." % (
                    tasks[bestTreeInd], origLoglik, logliks[bestTreeInd]), ALL_RANKS=True)
            bestTree = tasks[bestTreeInd]
            # currLoglik = logliks[bestTreeInd]
            # Then load this tree to move on
            # self.tree = unpickleTree(pickleFolder, 'randomTree_%d.dat' % bestTree)
            # mp_print("Before loading optimal random tree, memory usage is ",
            #          psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ALL_RANKS=True)
            if take_original_tree:
                scData = loadReconstructedTreeAndData(args, os.path.join(random_folder, 'orig_tree'),
                                                      reprocess_data=False, all_genes=False, get_cell_info=False,
                                                      all_ranks=False, rel_to_results=False)
            else:
                scData = loadReconstructedTreeAndData(args, os.path.join(random_folder, 'random_tree_%d' % bestTree),
                                                      reprocess_data=False, all_genes=False, get_cell_info=False,
                                                      all_ranks=False, rel_to_results=False)
            mp_print("Loaded optimal tree has loglikelihood: %r" % scData.metadata.loglik)
            # mp_print("Loaded optimal tree has true loglikelihood: %r" % scData.tree.calcLogLComplete(mem_friendly=True,
            #                                                                                  loglikVarCorr=scData.metadata.loglikVarCorr))
            # mp_print("After loading optimal random tree, memory usage is ",
            #          psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ALL_RANKS=True)
        mpi_wrapper.barrier()
    elif mpiInfo.rank == 0:
        scData = loadReconstructedTreeAndData(args, os.path.join(random_folder, 'orig_tree'),
                                              reprocess_data=False, all_genes=False, get_cell_info=False,
                                              all_ranks=True, rel_to_results=False)

    # Process 0 now has the tree on which we should proceed. Communicate this.
    # Then do on this tree all remaining beneficial reconfigurations in a greedy manner
    if mpiInfo.rank == 0:
        stored_tree_ind = 0 if stored_tree_ind is None else stored_tree_ind
        scData.storeTreeInFolder(os.path.join(tmp_folder, 'nnn_tree_%d' % stored_tree_ind), with_coords=True,
                                 verbose=verbose, cleanup_tree=False)
        scData = None
        gc.collect()
        # mp_print("After storing optimal random tree, memory usage is ",
        #          psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ALL_RANKS=True)
        # pickleTree(nnn_folder, 'nnn_start', self.tree)
    if mpiInfo.rank == 0:
        remove_tree_folders(random_folder, removeDir=True)
    stored_tree_ind = mpi_wrapper.bcast(stored_tree_ind, root=0)
    return stored_tree_ind, tmp_folder


def nnnReorder(args, tmp_folder, stored_tree_ind, maxMoves=100, closenessBound=0.5, verbose=False, mem_friendly=True):
    mpiInfo = mpi_wrapper.get_mpi_info()
    moveCounter = 0
    # mp_print("Before loading initial tree, memory usage is ",
    #          psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ALL_RANKS=True)
    scData = loadReconstructedTreeAndData(args, os.path.join(tmp_folder, 'nnn_tree_%d' % stored_tree_ind),
                                          reprocess_data=False, all_genes=False, get_cell_info=False,
                                          all_ranks=True, rel_to_results=False)
    mp_print("Loaded tree has loglikelihood: %r" % scData.metadata.loglik)
    currLoglik = scData.metadata.loglik
    # mp_print("After loading initial tree, memory usage is ",
    #          psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ALL_RANKS=True)
    # scData.tree.root.storeParent()
    stored_tree_ind += 1
    goodMovesLeft = True
    while goodMovesLeft and (moveCounter < maxMoves):
        # TODO: Optimize this as above
        # Create a list of nodes that are not the root and not leafs
        nodesList = recursionWrap(scData.tree.root.getNodeList, [], returnLeafs=True,
                                  returnRoot=True)  # TODO: Fast to maintain?
        nodeIndToNode = {node.nodeInd: node for node in nodesList}  # TODO: Maybe add to node-object
        candidatesList = [node for node in nodesList if (not node.isLeaf) and (not node.isRoot)]  # TODO: Same
        nCandidates = len(candidatesList)

        if not mem_friendly:
            # Since we will calculate reorderings from every node, it is beneficial to get AIRootInfo all at once
            scData.tree.root.getAIRootInfo(None, None)

        myTasks = getMyTaskNumbers(nCandidates, mpiInfo.size, mpiInfo.rank, skippingSteps=False)
        nTasks = len(myTasks)
        calcCounter = 0
        maxReorder = (-1, -1e9)
        maxReorderNoTopChange = (-1, -1e9)
        # as reorderDlogL we will store tuples (dLogL, topChangeYN) where second boolean indicates topology-change
        reorderDLogLs = {}
        for myInd, ind in enumerate(myTasks):
            dsNode = candidatesList[ind]
            if verbose and ((myInd + 1) % 1000 == 0):
                mp_print("Move %d. Considering edge %d out of %d." % (moveCounter, myInd + 1, nTasks),
                         ALL_RANKS=True)

            if (dsNode.cumClosenessNNN is not None) and (dsNode.cumClosenessNNN <= closenessBound):
                reorderDLogL = dsNode.nnnLoglik
                topChange = dsNode.nnnLoglikTopChange
            else:
                calcCounter += 1
                reorderDLogL, topChange = scData.tree.getOptEdgeList(dsNode, args, finalOptTimes=False,
                                                                     returnEdgelist=False, singleProcess=True,
                                                                     mem_friendly=mem_friendly)
                if reorderDLogL is None:
                    if verbose:
                        message = "node %d" % dsNode.nodeInd if dsNode.parentNode is None \
                            else "nodes %d and %d" % (dsNode.nodeInd, dsNode.parentNode.nodeInd)
                        mp_print(
                            "Reconfiguration around %s is skipped because of many connected nodes." % message,
                            ALL_RANKS=True)
                    reorderDLogL = -1
                    topChange = False
                reorderDLogLs[dsNode.nodeInd] = (reorderDLogL, topChange)
            if topChange and (reorderDLogL > maxReorder[1]):
                maxReorder = (dsNode.nodeInd, reorderDLogL)
            elif (not topChange) and (reorderDLogL > maxReorderNoTopChange[1]):
                maxReorderNoTopChange = (dsNode.nodeInd, reorderDLogL)

        # Communicate maxima and take only the maximum
        maxReorderTuples = mpi_wrapper.world_allgather(maxReorder)
        _, maxReorders = zip(*maxReorderTuples)
        maxReorder = maxReorderTuples[np.argmax(maxReorders)]
        maxReorderNoTopChangeTuples = mpi_wrapper.world_allgather(maxReorderNoTopChange)
        _, maxReordersNoTopChange = zip(*maxReorderNoTopChangeTuples)
        maxReorderNoTopChange = maxReorderNoTopChangeTuples[np.argmax(maxReordersNoTopChange)]

        # mp_print("Reconfig without topology change: %f. With topology change: %f" % (
        #       maxReorderNoTopChange[1], maxReorder[1]))
        # if maxReorderNoTopChange[1] > maxReorder[1]:
        #     mp_print("Reconfig without topology change is larger than with topology change.")

        # Communicate calculated reorderDLogLs and add to dsNodes
        reorderDLogLsAll = mpi_wrapper.world_allgather(reorderDLogLs)
        for reorderDLogLsSingle in reorderDLogLsAll:
            for nodeInd in reorderDLogLsSingle:
                nodeOI = nodeIndToNode[nodeInd]
                nodeOI.nnnLoglik = reorderDLogLsSingle[nodeInd][0]
                nodeOI.nnnLoglikTopChange = reorderDLogLsSingle[nodeInd][1]
                nodeOI.cumClosenessNNN = 0.
        calcCounters = mpi_wrapper.gather(calcCounter, root=0)
        if mpiInfo.rank == 0:
            calcCounter = sum(calcCounters)

        # If loglikelihood increased enough, do these optimal merges in the original tree as well
        # Find optimal reorder
        optNodeInd = maxReorder[0]
        if (maxReorder[1] / (scData.metadata.nCells * scData.metadata.nGenes)) < 1e-6:
            mp_print(
                "Couldn't find any more reconfigurations of next-nearest neighbours that increased the "
                "likelihood by more than 1e-6 per gene per cell.")
            goodMovesLeft = False
            continue

        nnnLoglik, optEdgeList, mostUSInfo = scData.tree.getOptEdgeList(nodeIndToNode[optNodeInd], args,
                                                                        finalOptTimes=True, returnEdgelist=True,
                                                                        mem_friendly=mem_friendly)

        if (nnnLoglik / (scData.metadata.nCells * scData.metadata.nGenes)) < 1e-6:
            mp_print(
                "Couldn't find any more reconfigurations of next-nearest neighbours that increased the "
                "likelihood by more than 1e-6 per gene per cell.")
            goodMovesLeft = False
            continue

        currLoglik += nnnLoglik
        if verbose:
            mp_print("Move %d: After %d calculations, it was found that re-ordering the nodes around node %d "
                     "maximally increased the "
                     "loglikelihood. Loglik now %f" % (moveCounter, calcCounter, optNodeInd, currLoglik))

        # Process optimal reordering in the original tree
        # Find most-upstream-node in original tree. In most cases, not all children will be reconfigured, only these
        # should be deleted
        scData.tree.processOptReconfig(optEdgeList, nodeIndToNode, nodesList, candidatesList, mostUSInfo)
        moveCounter += 1

        # TODO: Remove eventually, only uncomment this for making animations
        # if (mpiInfo.rank == 0) and bs_glob.nwk_counter and (scData.tree is not None):
        #     scData.tree.to_newick(use_ids=True,
        #                    results_path=os.path.join(bs_glob.nwk_folder,
        #                                              'tree_{}.nwk'.format(bs_glob.nwk_counter)))
        #     bs_glob.nwk_counter += 1

        if moveCounter % 100 == 0:
            if mpiInfo.rank == 0:
                scData.storeTreeInFolder(os.path.join(tmp_folder, 'nnn_tree_%d' % stored_tree_ind),
                                         with_coords=False, verbose=verbose, cleanup_tree=False)
                remove_tree_folders(tmp_folder, removeDir=False, notRemove=stored_tree_ind, base='nnn')
                stored_tree_ind += 1
                # pickleTree(tmp_folder, 'nnn_tree_%d.dat' % stored_tree_ind, self.tree)
                # removePickledTrees(tmpFolder, removeDir=False, notRemove=stored_tree_ind, base='random')
                # stored_tree_ind += 1
    if mpiInfo.rank == 0:
        remove_tree_folders(tmp_folder, removeDir=True, base='nnn')
        # removePickledTrees(tmpFolder, removeDir=False, base='random')
        return scData
    else:
        return None


def logLGradDepOnV(v, C, ltqsSq, ltqsVars):
    denominator = 1 / (v - ltqsVars)
    ltqsSqDenom = ltqsSq * denominator
    logL = - 2 * C * np.log(v)
    logL -= np.sum(np.log(denominator) + ltqsSqDenom)
    dLogL = - 2 * C / v
    dLogL += np.sum(denominator * (1 + ltqsSqDenom))
    return -logL, -dLogL


def neg_loglik_grad_using_measurements_and_errors(logv, measurements, variances, return_mean_ML=False):
    v = np.exp(logv)
    total_var_i = v + variances
    precision_i = 1 / total_var_i
    total_precision = np.sum(precision_i)
    mean_ML = np.sum(precision_i * measurements) / total_precision
    # Full loglik:
    # loglik = 0.5 * np.sum(-np.log(2* np.pi) + np.log(precision_i) - precision_i * (measurements - mean_ML) ** 2)
    weighted_dev = precision_i * (measurements - mean_ML) ** 2
    loglik = 0.5 * np.sum(np.log(precision_i) - weighted_dev)

    grad = 0.5 * np.sum(precision_i * (-1 + weighted_dev))
    grad_log = v * grad
    if not return_mean_ML:
        return -loglik, -grad_log
    else:
        return -loglik, -grad_log, mean_ML


# Used
def correct_means_stds(originalData, priorVariances, all_genes=False):
    # TODO: Check this derivation and computation
    cutoff = 1e-4
    uncorrected = copy.deepcopy(originalData)
    denominatorFactor = np.maximum(priorVariances[:, np.newaxis] - originalData.ltqsVars, cutoff)
    if not all_genes:
        genes_to_keep = np.where(np.min(denominatorFactor, axis=1) > cutoff)[0]
    else:
        genes_to_keep = np.arange(originalData.ltqsVars.shape[0])
    if len(genes_to_keep) == 0:
        problematic_cells = np.argsort(-np.sum(denominatorFactor == cutoff, axis=0))[
                            :min(10, originalData.ltqsVars.shape[1])]
    else:
        problematic_cells = None
    factor = priorVariances[genes_to_keep, np.newaxis] / denominatorFactor[genes_to_keep, :]
    originalData.ltqs = factor * originalData.ltqs[genes_to_keep, :]
    originalData.ltqsVars = originalData.ltqsVars[genes_to_keep, :] * factor
    if originalData.geneVariances is not None:
        originalData.geneVariances = originalData.geneVariances[genes_to_keep]
    if originalData.priorVariances is not None:
        originalData.priorVariances = originalData.priorVariances[genes_to_keep]

    uncorrected.ltqs = uncorrected.ltqs[genes_to_keep, :]
    uncorrected.ltqsVars = uncorrected.ltqsVars[genes_to_keep, :]
    if uncorrected.geneVariances is not None:
        uncorrected.geneVariances = uncorrected.geneVariances[genes_to_keep]
    if uncorrected.priorVariances is not None:
        uncorrected.priorVariances = uncorrected.priorVariances[genes_to_keep]

    return uncorrected, genes_to_keep, problematic_cells


# Not used
# def recoverTmpTree(args, tmpFolder, optimizeTimes=True):
#     scData = recoverTreeFromFile(args, tmpFolder, allRanks=False)
#
#     # If tmp-tree comes from unfinished run, optimal branch lengths to root have not been stored.
#     mpiRank = mpi_wrapper.get_process_rank()
#     if (mpiRank == 0) and (scData.tree.starryYN or optimizeTimes):
#         startTimeOpt = time.time()
#         # Get ltqs and ltqsVars of children
#         scData.tree.root.getLtqsComplete(mem_friendly=True)
#         scData.tree.optTimes(verbose=args.verbose, mem_friendly=True)
#         scData.metadata.loglik = scData.tree.calcLogLComplete(mem_friendly=True,
#                                                               loglikVarCorr=scData.metadata.loglikVarCorr)
#         # scData.tree.root.setLtqsVarsOrW(W_g=W_g)
#         dTimeOpt = time.time() - startTimeOpt
#         if args.verbose:
#             mp_print("Initial optimisation of times of recovered tree took " + str(dTimeOpt) + " seconds.")
#             mp_print("Optimal loglikelihood is: " + str(scData.metadata.loglik))
#
#     scData = broadcastRecursiveStruct(scData, root=0)
#     communicateGlobalVars()
#     return scData


# Not used
# def recoverTreeFromFile(args, mergerFilename, allRanks=True):
#     mpiRank = mpi_wrapper.get_process_rank()
#     if mpiRank == 0:
#         scData = SCData(dataset=args.dataset, filenamesData=args.filenames_data, verbose=args.verbose,
#                         pathToOrigData=args.data_folder, zscoreCutoff=args.zscore_cutoff,
#                         createStarTree=False)
#
#         scData.mergers = np.loadtxt(scData.result_path(mergerFilename))
#         scData.tree = scData.buildTreeFromMergers(scData.mergers)
#         scData.metadata.loglik = scData.tree.calcLogLComplete(mem_friendly=True,
#                                                               loglikVarCorr=scData.metadata.loglikVarCorr)
#         if args.verbose:
#             mp_print("\nLoglikelihood of tree recovered from file: " + str(scData.metadata.loglik) + '\n')
#     else:
#         scData = None
#         gc.collect()
#     if allRanks:
#         # scData = mpi_wrapper.bcast(scData, root=0)
#         scData = broadcastRecursiveStruct(scData, root=0)
#         communicateGlobalVars()
#     return scData


# Used
def initializeSCData(args, createStarTree=True, allGenes=False, allRanks=True, otherRanksMinimalInfo=False,
                     optTimes=True, getOrigData=False, returnUncorrected=False, noDataNeeded=False, verbose=True):
    zscoreCutoff = -1 if allGenes else args.zscore_cutoff
    mpiInfo = mpi_wrapper.get_mpi_info(singleProcess=(not allRanks))
    scData = SCData(dataset=args.dataset, filenamesData=args.filenames_data, verbose=args.verbose,
                    pathToOrigData=args.data_folder, zscoreCutoff=zscoreCutoff, getOrigData=getOrigData,
                    returnUncorrected=returnUncorrected, createStarTree=createStarTree, optTimes=optTimes,
                    noDataNeeded=noDataNeeded, sanityOutput=args.input_is_sanity_output, mpiInfo=mpiInfo,
                    results_folder=args.results_folder, rescale_by_var=args.rescale_by_var, all_genes=allGenes)

    if allRanks:
        if not otherRanksMinimalInfo:
            # scData = mpi_wrapper.bcast(scData, root=0)
            scData = broadcastRecursiveStruct(scData, root=0)
        else:
            if mpiInfo.rank != 0:
                scData = SCData(onlyObject=True)
                scData.tree = Tree()
            scData.metadata = mpi_wrapper.bcast(scData.metadata, root=0)
        communicateGlobalVars()
    return scData


# def loadCurrentState(outputFolder, filename='tmp_tree.dat', args=None, dataFolder=None, fullPath=None, allRanks=True,
#                      dataOrResults='data', singleProcess=False):
#     if (mpi_wrapper.get_process_rank() == 0) or singleProcess:
#         if fullPath is None:
#             if dataFolder is None:
#                 if args.dataset is not None:
#                     dataFolder = os.path.join(dataOrResults, args.dataset)
#                 else:
#                     dataFolder = args.data_folder
#             tmpDataPath = os.path.join(dataFolder, outputFolder)
#             fullPath = os.path.join(tmpDataPath, filename)
#             mp_print("Loading tree from %s" % fullPath)
#         try:
#             with open(fullPath, 'rb') as file:
#                 scData = pickle.load(file)
#         except RecursionError:
#             with RecursionLimit():
#                 with open(fullPath, 'rb') as file:
#                     scData = pickle.load(file)
#
#         bs_glob.nCells = scData.metadata.nCells
#         bs_glob.nGenes = scData.metadata.nGenes
#         bs_glob.nNodes = scData.tree.nNodes
#
#         scData.metadata.loglik = scData.tree.calcLogLComplete(mem_friendly=True,
#                                                               loglikVarCorr=scData.metadata.loglikVarCorr)
#         if args.verbose:
#             mp_print("\nLoglikelihood of tree recovered from file: " + str(scData.metadata.loglik) + '\n')
#     else:
#         scData = SCData(onlyObject=True)
#         scData.tree = Tree()
#     if not singleProcess:
#         if allRanks:  # So in this case we communicate all information with all processes
#             with RecursionLimit():
#                 scData = mpi_wrapper.bcast(scData)
#         else:  # In this case, we only communicate the metadata
#             scData.metadata = mpi_wrapper.bcast(scData.metadata, root=0)
#         communicateGlobalVars()
#     return scData


# Used
def getMetadata(args, scData, outputFolder_raw, computationTime):
    outputFolder = scData.result_path(outputFolder_raw)
    mpiRank = mpi_wrapper.get_process_rank()
    if mpiRank == 0:
        scData.metadata.loglik = scData.tree.calcLogLComplete(mem_friendly=True,
                                                              loglikVarCorr=None)
        # Read in data with all genes
        # scDataFull = loadReconstructedTreeAndData(args, outputFolder, all_genes=True, get_cell_info=False,
        #                                           reprocess_data=True, all_ranks=False)
        # scDataFull = initializeSCData(args, createStarTree=False, allGenes=True, allRanks=False)
        # scDataFull.tree = scDataFull.buildTreeFromMergers(scData.mergers)
        # scDataFull.metadata.loglik = scDataFull.tree.calcLogLComplete(mem_friendly=True,
        #                                                               loglikVarCorr=scDataFull.metadata.loglikVarCorr)
        # mp_print("Final loglikelihood of inferred tree on filtered genes: " + str(scDataFull.metadata.loglik))

        if os.path.exists(scData.data_path('true_tree')):
            scDataFull = loadReconstructedTreeAndData(args, outputFolder_raw, all_genes=True, get_cell_info=False,
                                                      reprocess_data=True, all_ranks=False, rel_to_results=True,
                                                      no_data_needed=False)
            scDataFull.metadata.loglik = scDataFull.tree.calcLogLComplete(mem_friendly=True,
                                                                          loglikVarCorr=None)
            mp_print(
                "Loglikelihood of reconstructed tree (all genes, before optTimes): " + str(scDataFull.metadata.loglik))
            scDataFull.tree.optTimes(mem_friendly=True, tol=1e-8)
            scDataFull.metadata.loglik = scDataFull.tree.calcLogLComplete(mem_friendly=True,
                                                                          loglikVarCorr=None)
            full_loglik = scDataFull.metadata.loglik

            scData_true = loadReconstructedTreeAndData(args, scData.data_path('true_tree'), all_genes=True,
                                                       get_cell_info=False,
                                                       reprocess_data=True, all_ranks=False)
            true_loglik = scData_true.tree.calcLogLComplete(mem_friendly=True,
                                                            loglikVarCorr=None)
            mp_print("Loglikelihood of ground truth tree (before optTimes): " + str(true_loglik))
            scData_true.tree.optTimes_single_scalar(mem_friendly=True, tol=1e-8)
            true_loglik = scData_true.tree.calcLogLComplete(mem_friendly=True,
                                                            loglikVarCorr=None)
            # true_mergers = np.loadtxt(scData.data_path('true_mergers.txt'))
            # true_tree = scDataFull.buildTreeFromMergers(true_mergers)
            # true_loglik = true_tree.calcLogLComplete(mem_friendly=True, loglikVarCorr=scDataFull.metadata.loglikVarCorr)
            mp_print("Loglikelihood of reconstructed tree on all genes: " + str(scDataFull.metadata.loglik))
            mp_print("Loglikelihood of ground truth tree: " + str(true_loglik))
        else:
            true_loglik = np.nan
            full_loglik = np.nan

        metadata = pd.DataFrame(
            {'nGenes': scData.metadata.nGenes, 'nCells': scData.metadata.nCells, 'reconsTime': computationTime,
             'zscoreCutoff': args.zscore_cutoff, 'coresUsed': mpi_wrapper.get_process_size(),
             'finalLoglik': scData.metadata.loglik, 'trueLoglik': true_loglik, 'fullLoglik': full_loglik,
             'remainingRootChildren': len(scData.tree.root.childNodes)},
            index=['metadata'])
        return metadata


def loadReconstructedTreeAndData(args, tree_folder, reprocess_data=False, all_genes=False, all_ranks=True,
                                 get_cell_info=False, corrected_data=True, rel_to_results=False, no_data_needed=False,
                                 single_process=False, keep_original_data=False, calc_loglik=False, get_data=True,
                                 get_posterior_ltqs=False, otherRanksMinimalInfo=False, verbose=True,
                                 calc_posteriors=True):
    """

    :param args:
    :param tree_folder:
    :param reprocess_data:
    :param all_genes:
    :param all_ranks:
    :param get_cell_info:
    :param corrected_data:
    :param rel_to_results:
    :param no_data_needed:
    :param single_process:
    :param keep_original_data:
    :param calc_loglik:
    :param get_data:
    :param get_posterior_ltqs:
    :param otherRanksMinimalInfo:
    :param verbose:
    :param calc_posteriors: This boolean may be false, even when get_posterior_ltqs is True. This means that, even
    though we want to read in the posteriors if they are stored in the folder, we do not calculate them if they are not
    available. This is to save the memory of storing these posteriors on the tree. Instead, we only have to calculate
    them when we immediately write them to the file.
    :return:
    """
    if type(args) is dict:
        args = convert_dict_to_named_tuple(args)
    mpi_info = mpi_wrapper.get_mpi_info(singleProcess=single_process)
    if reprocess_data:
        scData = initializeSCData(args, createStarTree=False, allGenes=all_genes, allRanks=all_ranks, optTimes=False,
                                  getOrigData=True, returnUncorrected=(not corrected_data), noDataNeeded=no_data_needed,
                                  otherRanksMinimalInfo=otherRanksMinimalInfo, verbose=verbose)
    else:
        scData = SCData(onlyObject=True, dataset=args.dataset, results_folder=args.results_folder)
    if rel_to_results:
        tree_folder = scData.result_path(tree_folder)

    get_all_data = get_data and ((mpi_info.rank == 0) or (all_ranks and (not otherRanksMinimalInfo)))
    tree_tuple = reconstructTreeFromEdgeVertInfo(scData, tree_folder, verbose=verbose)
    vertIndToNode, vertIndToNodeInd, vertIndToNodeId, edgeList, distList = tree_tuple
    data_found = load_data_for_tree(scData, tree_folder, vertIndToNode, get_all_data=get_all_data,
                                    load_data=(not reprocess_data),
                                    verbose=False, keep_original_data=keep_original_data,
                                    get_posterior_ltqs=get_posterior_ltqs,
                                    no_data_needed=no_data_needed)
    if data_found and get_all_data and calc_loglik and (mpi_info.rank == 0):
        scData.metadata.loglik = scData.tree.calcLogLComplete(mem_friendly=True,
                                                              loglikVarCorr=scData.metadata.loglikVarCorr)
        if verbose:
            mp_print("Loaded tree has loglikelihood %.4f" % scData.metadata.loglik)

    if (bs_glob.nCells is not None) and (bs_glob.nCells > 25000):
        bs_glob.mem_friendly = True
        scData.tree.root.keep_one_ltqsvars_or_W(keep_ltqsvars=True)

    # if (not corrected_data) and (scData.unscaled is not None) and (scData.unscaled.ltqs is not None) \
    #         and get_all_data and data_found:
    #     # scData.originalData = scData.unscaled
    #     scData.unscaled = None
    #     addDataToTree(scData, vertIndToNode)
    #
    #     mp_print("Calculating ltqs of internal nodes", ALL_RANKS=True)
    #     scData.tree.root.getLtqsComplete(mem_friendly=True)

    if data_found and get_posterior_ltqs and get_all_data and (scData.tree.root.ltqsAIRoot is None) and calc_posteriors:
        if scData.tree.root.ltqs is None:
            scData.tree.root.getLtqsComplete(mem_friendly=True)
        scData.tree.root.getAIRootInfo(None, None)

    # Also store for every node who its parent is
    scData.tree.root.storeParent()

    if not get_cell_info:
        return scData

    get_cell_info_tree(scData, vertIndToNode)
    return scData, vertIndToNodeId


def reconstructTreeFromEdgeVertInfo(scData, tree_folder, verbose=False):
    # Read reconstructed tree information
    edgeList = []
    distList = []
    edgeFile = os.path.join(tree_folder, 'edgeInfo.txt')
    with open(edgeFile, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            edgeList.append((int(row[0]), int(row[1])))
            distList.append(float(row[2]))

    vertIndToNodeInd = {}
    vertIndToNodeId = {}
    vertFile = os.path.join(tree_folder, 'vertInfo.txt')
    with open(vertFile, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        skipRow = True
        for row in reader:
            if skipRow:
                skipRow = False
                continue
            vertIndToNodeInd[int(row[0])] = int(row[1])
            vertIndToNodeId[int(row[0])] = row[2]

    # Update recursion limits so that very deep trees don't create errors
    set_recursion_limits(len(edgeList))

    scData.tree = Tree()
    vertIndToNode = {}
    for ind, edge in enumerate(edgeList):
        if ind == 0:
            scData.tree.root = TreeNode(nodeInd=vertIndToNodeInd[edge[0]], childNodes=[], isLeaf=False, isRoot=True,
                                        ltqs=None, ltqsVars=None, tParent=None, nodeId=vertIndToNodeId[edge[0]])
            vertIndToNode[edge[0]] = scData.tree.root
            childNode = TreeNode(nodeInd=vertIndToNodeInd[edge[1]], childNodes=[], isLeaf=True, isRoot=False,
                                 ltqs=None, ltqsVars=None, tParent=distList[ind], nodeId=vertIndToNodeId[edge[1]])
            scData.tree.root.childNodes.append(childNode)
            vertIndToNode[edge[1]] = childNode
            scData.tree.nNodes = 2
            continue
        if edge[0] in vertIndToNode:
            parentVert = edge[0]
            childVert = edge[1]
        else:
            parentVert = edge[1]
            childVert = edge[0]
        childNode = TreeNode(nodeInd=vertIndToNodeInd[childVert], childNodes=[], isLeaf=True, isRoot=False,
                             ltqs=None, ltqsVars=None, tParent=distList[ind], nodeId=vertIndToNodeId[childVert])
        parentNode = vertIndToNode[parentVert]
        parentNode.childNodes.append(childNode)
        parentNode.isLeaf = False
        vertIndToNode[childVert] = childNode
        scData.tree.nNodes += 1
    bs_glob.nNodes = scData.tree.nNodes
    # Store vert_inds on nodes
    for vert_ind, node in vertIndToNode.items():
        node.vert_ind = vert_ind
    scData.tree.vert_ind_to_node = vertIndToNode
    # Commented out the below on January 6, 2026. This was supposed to make vert-indices consistent with depth-first
    # search, but this does not seem necessary, and it may mess up reading in the coordinates.

    # vertIndToNode_test, scData.tree.nNodes = scData.tree.root.renumber_verts(vertIndToNode={}, vert_count=0)
    # vertIndToNodeId_test = {}
    # vertIndToNodeInd_test = {}
    # for vert_ind, node in vertIndToNode_test.items():
    #     vertIndToNodeId_test[vert_ind] = node.nodeId
    #     vertIndToNodeInd_test[vert_ind] = node.nodeInd

    tree_tuple = vertIndToNode, vertIndToNodeInd, vertIndToNodeId, edgeList, distList
    scData.tree.max_node_ind = scData.tree.root.get_max_node_ind(-np.inf)
    bs_glob.max_node_ind = scData.tree.max_node_ind
    if verbose:
        mp_print("\n\nReconstructed tree loaded from: \n%s \n%s" % (edgeFile, vertFile))
    return tree_tuple


def load_data_for_tree(scData, tree_folder, vertind_to_node, get_all_data=True, load_data=True, verbose=False,
                       keep_original_data=False, get_posterior_ltqs=False, no_data_needed=False):
    if not load_data:  # In this case, data was already loaded, we only add it to the tree and do some checks
        if scData.metadata.nCells is None:
            scData.metadata.nCells = scData.tree.root.countDSLeafs(0)
            bs_glob.nCells = scData.metadata.nCells
            scData.metadata.cellIds = ["Cell_%d" % ind for ind in range(scData.metadata.nCells)]
        if get_all_data and (scData.originalData.ltqs is not None):
            addDataToTree(scData, vertind_to_node)
            # scData.metadata.loglik = scData.tree.calcLogLComplete(mem_friendly=True)
        if not keep_original_data:
            scData.originalData = None
        if verbose and (scData.metadata.loglik is not None):
            mp_print("\nLoglikelihood of tree recovered from file: " + str(scData.metadata.loglik))
        return True

    # Read all metadata
    scData.metadata = Metadata(json_filepath=os.path.join(tree_folder, 'metadata.json'), curr_metadata=scData.metadata)
    bs_glob.nGenes = scData.metadata.nGenes
    bs_glob.nCells = scData.metadata.nCells
    bs_glob.nNodes = scData.tree.nNodes

    # Try to read data in following order:
    # - see if there is data for all vertices stored where the tree is stored
    # - if not, read the data of all leafs from the folder that is indicated in the metadata-file, then calculate all
    # data for all vertices
    data_found = True
    posteriors_found = False
    ltqs_found = False
    if get_posterior_ltqs:
        meanspath = os.path.join(tree_folder, 'posterior_ltqs_vertByGene.npy')
        if os.path.exists(meanspath):
            posteriors_found = True
            varspath = os.path.join(tree_folder, 'posterior_ltqsVars_vertByGene.npy')
            readFolder = tree_folder

    if (not get_posterior_ltqs) or (not posteriors_found):
        meanspath = os.path.join(tree_folder, 'ltqs_vertByGene.npy')
        if os.path.exists(meanspath):
            ltqs_found = True
            varspath = os.path.join(tree_folder, 'ltqsVars_vertByGene.npy')
            readFolder = tree_folder

    if (not ltqs_found) and (not posteriors_found):
        readFolder = scData.metadata.processedDatafolder

    if scData.metadata.processedDatafolder is None:
        scData.metadata.processedDatafolder = readFolder

    if get_all_data:
        cell_id_set = set(scData.metadata.cellIds)
        try:
            start = time.time()
            if ltqs_found or posteriors_found:
                print_ind = 1000
                ltqs_cg = np.load(meanspath, allow_pickle=False, mmap_mode='r')
                ltqsVars_cg = np.load(varspath, allow_pickle=False, mmap_mode='r')
                if ltqs_cg.shape[0] != bs_glob.nNodes:
                    raise Exception("Trying to load data for {} nodes, "
                                    "while tree has {} nodes.".format(ltqs_cg.shape[0], bs_glob.nNodes))
                for vert_ind in range(bs_glob.nNodes):
                    if (vert_ind == print_ind) and verbose:
                        print_ind *= 2
                        mp_print("Loaded coordinates for %d vertices out of %d in %.2f seconds." % (
                            vert_ind, bs_glob.nNodes, time.time() - start))
                    node = vertind_to_node[vert_ind]
                    if posteriors_found:
                        node.ltqsAIRoot = ltqs_cg[vert_ind, :]
                        node.setLtqsVarsOrW(ltqsVars=ltqsVars_cg[vert_ind, :], AIRoot=True)
                    else:
                        node.ltqs = ltqs_cg[vert_ind, :]
                        node.setLtqsVarsOrW(ltqsVars=ltqsVars_cg[vert_ind, :])
                    node.isCell = node.nodeId in cell_id_set
                del ltqs_cg
                del ltqsVars_cg
                gc.collect()
                if verbose:
                    mp_print("Loaded coordinates for %d vertices out of %d in %.2f seconds." % (
                        vert_ind, bs_glob.nNodes, time.time() - start))
            else:
                origFolder = scData.metadata.processedDatafolder
                scData.originalData = OriginalData()
                if os.path.exists(os.path.join(origFolder, 'delta.npy')):
                    meanspath = os.path.join(origFolder, 'delta.npy')
                    varspath = os.path.join(origFolder, 'delta_vars.npy')
                    scData.originalData.ltqs = np.load(meanspath, allow_pickle=False, mmap_mode='r')
                    scData.originalData.ltqsVars = np.load(varspath, allow_pickle=False, mmap_mode='r')
                    if scData.originalData.ltqs.shape[1] != len(scData.metadata.cellIds):
                        raise Exception("Trying to load data for {} cells from folder {}, "
                                        "while tree seems to have {} cells.".format(scData.originalData.ltqs.shape[1],
                                                                                    origFolder,
                                                                                    len(scData.metadata.cellIds)))
                    # scData.originalData.ltqs = ltqs_cg.T
                    # scData.originalData.ltqsVars = ltqsVars_cg.T
                elif os.path.exists(os.path.join(origFolder, 'delta_vmax.txt')):
                    raise Exception("Trying to load data from folder {}, "
                                    "but no pre-processed data can be found.".format(origFolder))
                    scData.originalData.ltqs = np.loadtxt(os.path.join(origFolder, 'delta_vmax.txt'))
                    scData.originalData.ltqsVars = np.loadtxt(os.path.join(origFolder, 'd_delta_vmax.txt')) ** 2
                elif no_data_needed:
                    print("Original data was not found, but is not strictly needed now. Will continue without it.")
                    data_found = False
                else:
                    exit("No data was found, please provide correct paths to where the original data is stored.")
                addDataToTree(scData, vertind_to_node)
                if not keep_original_data:
                    scData.originalData = None
                if data_found:
                    scData.metadata.loglik = scData.tree.calcLogLComplete(mem_friendly=True,
                                                                          loglikVarCorr=scData.metadata.loglikVarCorr)
        except Exception as e:
            mp_print("Encountered exception '{}'".format(e), WARNING=True)
            if no_data_needed:
                mp_print("Original data was not found, but is not strictly needed now. Will continue without it.",
                         WARNING=True)
                data_found = False
        if verbose:
            if scData.metadata.loglik is not None:
                mp_print("\nLoglikelihood of tree recovered from file: " + str(scData.metadata.loglik))

    return data_found


def addDataToTree(scData, vertIndToNode):
    cell_id_to_ind = {cell_id: ind for ind, cell_id in enumerate(scData.metadata.cellIds)}
    for vert, node in vertIndToNode.items():
        if node.nodeId in cell_id_to_ind:
            cellInd = cell_id_to_ind[node.nodeId]
            if (scData.originalData is not None) and (scData.originalData.ltqs is not None):
                node.ltqs = scData.originalData.ltqs[:, cellInd]
                node.setLtqsVarsOrW(ltqsVars=scData.originalData.ltqsVars[:, cellInd])
            node.isCell = True


# Old
# def addIgraph(scData, vertIndToNode, vertIndToNodeInd, edgeList, distList):
#     # We store how many vertices there are in our final tree
#     scData.nVert = len(vertIndToNode)
#     # And we make up some names
#     scData.vertNames = ['vert_' + str(ind) for ind in range(scData.nVert)]
#
#     # If the tree reconstruction is run on cellstates-output, we should first account for mapping the cells to
#     # cellstates
#     cellsToCellstates = {}
#     cellstatesToVerts = {}
#     scData.cellsToVerts = {}
#     scData.cellIndToVertInd = {}
#     cellIdToVertInd = {node.nodeId: vertInd for vertInd, node in vertIndToNode.items()}
#     for ind in range(scData.metadata.nCells):
#         cellId = scData.metadata.cellIds[ind]
#         cellstateId = cellId
#         cellsToCellstates[cellId] = cellstateId
#         if cellId in cellIdToVertInd:
#             vert_ind = cellIdToVertInd[cellId]
#         else:
#             vert_ind = -1
#         # Find out to which new vertex this was again send to
#         cellstatesToVerts[cellstateId] = scData.vertNames[vert_ind]
#         # Store in cell_assignment to which eventual vertex the original cell was sent
#         scData.cellsToVerts[cellId] = cellstatesToVerts[cellsToCellstates[cellId]]
#         # Also store a map of which cellInd went to which vertInd
#         scData.cellIndToVertInd[ind] = vert_ind
#
#     # Build up the tree
#     scData.mst = igraph.Graph(directed=True)
#     scData.mst.add_vertices(scData.nVert)
#
#     scData.mst.add_edges(edgeList)
#     scData.mst.es["weight"] = distList
#     scData.mst.vs["name"] = scData.vertNames
#     scData.mst.es["name"] = ['e_' + str(ind) for ind in range(scData.nVert - 1)]
#     igraph.summary(scData.mst)
#     scData.nCellsPerVert = np.zeros(scData.nVert)
#     scData.vertIndToCellInds = {ind: [] for ind in range(scData.nVert)}
#     for ind in range(scData.metadata.nCells):
#         if scData.cellIndToVertInd[ind] == -1:
#             # TODO: Clean this up
#             continue
#         scData.vertIndToCellInds[scData.cellIndToVertInd[ind]].append(ind)
#         scData.nCellsPerVert[scData.cellIndToVertInd[ind]] += 1


# Used
def get_cell_info_tree(scData, vertIndToNode):
    # We store how many vertices there are in our final tree
    scData.nVerts = len(vertIndToNode)
    # And we make up some names
    scData.vertNames = ['vert_' + str(ind) for ind in range(scData.nVerts)]

    scData.cellsToVerts = {}
    scData.cellIndToVertInd = {}
    if hasattr(scData.metadata, 'nCss'):
        scData.csToVerts = {}
        scData.csIndToVertInd = {}
        get_cs_info = True
        logger.error("I didn't think getting cs-info here would ever be necessary. Check why this happens.")
        exit()
    else:
        get_cs_info = False
    nodeIdToVertInd = {node.nodeId: vertInd for vertInd, node in vertIndToNode.items()}
    cells_not_found = []
    for ind in range(scData.metadata.nCells):
        cellId = scData.metadata.cellIds[ind]
        if cellId in nodeIdToVertInd:
            vert_ind = nodeIdToVertInd[cellId]
        else:
            cells_not_found.append(cellId)
            vert_ind = None
        if vert_ind is not None:
            # Store in cell_assignment to which vertex the original cell was sent
            scData.cellsToVerts[cellId] = scData.vertNames[vert_ind]
            # Also store a map of which cellInd went to which vertInd
            # TODO: Maybe eventually only store this.
            scData.cellIndToVertInd[ind] = vert_ind

    if len(cells_not_found):
        mp_print("{}\n\n Could not find the above cells in the tree. "
                 "Check this! Now exiting!".format(cells_not_found), ERROR=True)
        exit()

    scData.nCellsPerVert = np.zeros(scData.nVerts)
    scData.vertIndToCellInds = {ind: [] for ind in range(scData.nVerts)}
    # for ind in range(scData.metadata.nCells):
    for cell_ind, vert_ind in scData.cellIndToVertInd.items():
        # if scData.cellIndToVertInd[ind] == -1:
        #     continue
        scData.vertIndToCellInds[vert_ind].append(cell_ind)
        scData.nCellsPerVert[vert_ind] += 1


def infer_true_var_log(inferred_vals, inferred_vars):
    """ finds tau """
    n_genes = inferred_vals.shape[0]
    true_vars = np.zeros(n_genes)
    for ind in range(n_genes):
        if (ind % 1000) == 0:
            print("Done %d genes." % ind)
        true_var_log, fval, ierr, numfunc = fminbound(loglik_given_true_var_log, -13, 13,
                                                      args=(inferred_vals[ind, :], inferred_vars[ind, :]),
                                                      full_output=1, xtol=1.e-06)
        if ierr:
            sys.exit("log_activities has not converged after %d iterations.\n"
                     % numfunc)
        true_vars[ind] = np.exp(true_var_log)

    return true_vars


def loglik_given_true_var_log(true_var_log, inferred_vals, inferred_vars):
    """ Calculate likelihood """
    true_var = np.exp(true_var_log)
    alpha = 1. / (inferred_vars + true_var)

    m0 = np.sum(alpha)
    m1 = np.sum(alpha * inferred_vals)
    m2 = np.sum(alpha * np.power(inferred_vals, 2))
    mu = m1 / m0

    return (0.5 * (-m1 * mu + m2) - 0.5 * np.sum(np.log(alpha))) + 0.5 * np.log(
        np.sum(alpha))  # the middle term was missing in the other implementation


def read_and_filter(data_folder, meansfile, stdsfile, sanityOutput, zscoreCutoff, mpiInfo, tmp_folder=None,
                    verbose=False, all_genes=False):
    """
    Reads in means and standard deviations line by line (i.e per gene/feature), possibly parallelized over multiple
    processes. For each gene, we determine if it makes the zscore-cutoff before adding it to the data to save memory.
    :param meanspath: Full path to means-file. The centers of the Gaussian likelihood of the data (the measurements).
    :param stdspath: Full path to standard deviations corresponding to the Gaussian likelihoods (the error-bars)
    :param sanityOutput: Boolean determining if the input is Sanity output, in which case we know how to correct the
    means and stds
    :param zscoreCutoff: Cutoff for the signal-to-noise ratio of genes
    :param mpiInfo: Information on how many processes are working in parallel, and the rank of the current process
    :param verbose:
    :return:
    """
    # Read in means and stds per feature, immediately determine zscore and discard if under cutoff
    genes_to_keep = []
    tmp_means = []
    tmp_vars = []
    tmp_gene_vars = []
    print_ind = 1000
    if tmp_folder is None:
        tmp_folder = data_folder
        Path(tmp_folder).mkdir(parents=True, exist_ok=True)

    # Define cutoff for how close the variances on LTQs can be to the true variances before throwing out a gene.
    # In error-less data, variances on the LTQ-posterior should always be smaller than the true variance
    if not all_genes:
        divide_out_prior_cutoff = 1e-4
    else:
        divide_out_prior_cutoff = -1

    if sanityOutput:
        mp_print("Since the argument '--input_is_sanity_output' is 'True', "
                 "we are assuming the input-files are Sanity output.\n"
                 "If this is not the desired behavior, change this argument in the Bonsai config-yaml file.")
        meanspath = os.path.join(data_folder, 'delta_vmax.txt')
        input_incomplete = False
        if not os.path.exists(meanspath):
            mp_print("File {} not found.".format(meanspath), WARNING=True)
            exit()
            input_incomplete = True
        stdspath = os.path.join(data_folder, 'd_delta_vmax.txt')
        if not os.path.exists(stdspath):
            mp_print("File {} not found.".format(stdspath), WARNING=True)
            input_incomplete = True
        gene_variancepath = os.path.join(data_folder, 'variance_vmax.txt')
        if not os.path.exists(gene_variancepath):
            mp_print("File {} not found.".format(gene_variancepath), WARNING=True)
            input_incomplete = True
        gene_meanspath = os.path.join(data_folder, 'mu_vmax.txt')
        if not os.path.exists(gene_meanspath):
            mp_print("File {} not found.".format(gene_meanspath), WARNING=True)
            input_incomplete = True
        # Check if correct version of Sanity was run
        if input_incomplete:
            if os.path.exists(os.path.join(data_folder, 'delta.txt')):
                mp_print("Only found delta.txt, not delta_vmax.txt. "
                         "Make sure to run Sanity with the argument '-max_v only_max_output'", ERROR=True)
            else:
                mp_print("Could not find (the right) Sanity-output.\n"
                         "Are you sure the argument --input_is_sanity_output should be set to True?"
                         "\nAre you sure you are running Sanity with the extended-output flag -e 1, "
                         "and the vmax-argument: -vmax true?", ERROR=True)
            exit()
    else:
        meanspath = os.path.join(data_folder, meansfile)
        stdspath = os.path.join(data_folder, stdsfile) if (stdsfile is not None) else None
        if not os.path.exists(meanspath):
            mp_print("Did not find necessary input-file: {}".format(meanspath), ERROR=True)
            exit()

    # First determine the number of genes on process 0
    if mpiInfo.rank == 0:
        nGenesOrig = None
        with open(meanspath, 'r') as fp:
            for (nGenesOrig, _) in enumerate(fp, 1):
                pass
    else:
        nGenesOrig = None
    nGenesOrig = mpi_wrapper.bcast(nGenesOrig, root=0)  # Communicate number of genes with other processes
    # Get number of cells
    with open(meanspath, 'r') as fmeans:
        reader_means = csv.reader(fmeans, delimiter='\t')
        nCells = len(next(reader_means))

    # Distribute genes that should be read in over the different processes
    myTasks = getMyTaskNumbers(nGenesOrig, mpiInfo.size, mpiInfo.rank, skippingSteps=False)
    if len(myTasks) > 0:
        myTasks = [myTasks[0], myTasks[-1] + 1]
        nTasks = myTasks[1] - myTasks[0]
    else:
        myTasks = [-1, -1]
        nTasks = 0
    start = time.time()

    if (stdspath is not None) and os.path.exists(stdspath):
        # In this case, we read in standard deviations, and use them to estimate a signal-to-noise ratio for each gene.
        ltqStdsFound = True
        zscoreCutoffSq = zscoreCutoff ** 2 if zscoreCutoff > 0 else 0.0
        if sanityOutput:
            # If input is sanity output, we know that we get the posterior means and variances, instead of the
            # likelihood means and variances. So the posteriors can immediately be used for calculating the
            # signal-to-noise ratio.
            # The likelihood means and variances can be reconstructed by reading in the estimated variances so
            # that we can divide out the prior. In addition, we can read in the gene-means, so that we get LTQs instead
            # of the reported log-fold changes.
            # Start by reading the gene-means and gene-variances
            gene_variances = np.loadtxt(gene_variancepath)
            if len(gene_variances) != nGenesOrig:
                print("Number of gene variances in file {} does not match number of features in the ltq-matrix.\n"
                      "Check this.".format(gene_variancepath))
                exit()
            gene_means = np.loadtxt(gene_meanspath)
            if len(gene_means) != nGenesOrig:
                print("Number of gene means in file {} does not match number of features in the ltq-matrix.\n"
                      "Check this.".format(gene_meanspath))
                exit()

            with open(meanspath, 'r') as fmeans:
                with open(stdspath, 'r') as fstd:
                    for row_ind in range(myTasks[0]):
                        # Skip over every row that other processes read, until first task for this process is reached
                        next(fmeans)
                        next(fstd)
                    # When reaching first task, read in first row.
                    reader_means = csv.reader(fmeans, delimiter='\t')
                    reader_stds = csv.reader(fstd, delimiter='\t')
                    for row_ind in range(myTasks[0], myTasks[1]):
                        read = next(reader_stds)
                        vars = np.maximum(np.asarray(read, dtype='float') ** 2, 1e-12)
                        means = np.asarray(next(reader_means), dtype='float')

                        # Do not subtract mean of means anymore
                        # zscoreSq = np.sum((means - np.mean(means)) ** 2 / vars) / nCells
                        zscoreSq = np.sum(means ** 2 / vars) / nCells

                        if zscoreSq > zscoreCutoffSq:
                            # Because Sanity reports means and variances of the posteriors, rather than of the
                            # likelihood, we here correct for that by dividing out the prior
                            gene_var = gene_variances[row_ind]
                            # denominatorFactor = np.maximum(priorVariances[:, np.newaxis] - originalData.ltqsVars,
                            #                                cutoff)
                            denominator_factor = gene_var - vars
                            if np.any(denominator_factor < divide_out_prior_cutoff):
                                continue
                            denominator_factor = np.maximum(denominator_factor, 1e-6)

                            # factor = priorVariances[genes_to_keep, np.newaxis] / denominatorFactor[genes_to_keep, :]
                            factor = gene_var / denominator_factor
                            # originalData.ltqs = factor * originalData.ltqs[genes_to_keep, :]
                            means *= factor
                            # originalData.ltqsVars = originalData.ltqsVars[genes_to_keep, :] * factor
                            vars *= factor

                            means += gene_means[row_ind]

                            tmp_means.append(means)
                            tmp_vars.append(vars)
                            tmp_gene_vars.append(gene_var)
                            genes_to_keep.append(row_ind)
                        if (row_ind - myTasks[0]) > print_ind:
                            print_ind *= 2
                            mp_print(
                                "Processing data for the the %d-th feature out of %d, this took %.2f seconds." % (
                                    row_ind - myTasks[0], nTasks, time.time() - start), ONLY_RANK=0)
        else:
            with open(meanspath, 'r') as fmeans:
                with open(stdspath, 'r') as fstd:
                    for row_ind in range(myTasks[0]):
                        # Skip over every row that other processes read, until first task for this process is reached
                        next(fmeans)
                        next(fstd)
                    # When reaching first task, read in first row.
                    reader_means = csv.reader(fmeans, delimiter='\t')
                    reader_stds = csv.reader(fstd, delimiter='\t')
                    for row_ind in range(myTasks[0], myTasks[1]):
                        read = next(reader_stds)
                        vars = np.maximum(np.asarray(read, dtype='float') ** 2, 1e-12)
                        means = np.asarray(next(reader_means), dtype='float')

                        # TODO: Check whether it helps to give a derivative to that function as well
                        try:
                            log_v0 = np.log(np.var(means))
                        except RuntimeWarning:
                            log_v0 = 0
                        opt_res = minimize(neg_loglik_grad_using_measurements_and_errors, x0=log_v0,
                                           jac=True, args=(means, vars, False))
                        inferred_gene_var = np.exp(opt_res.x[0])
                        if not opt_res.success:
                            mp_print("Could not optimize variance for feature {}.\n"
                                     "Discarding this feature as variance is likely very small".format(row_ind))
                            continue
                        _, _, inferred_gene_mean = neg_loglik_grad_using_measurements_and_errors(opt_res.x, means, vars,
                                                                                                 return_mean_ML=True)
                        # zscoreSq = np.sum((means - np.mean(means)) ** 2 / vars) / nCells
                        zscoreSq = np.sum((inferred_gene_var / (inferred_gene_var + vars)) * (
                                means - inferred_gene_mean) ** 2 / vars) / nCells

                        if zscoreSq > zscoreCutoffSq:
                            tmp_means.append(means)
                            tmp_vars.append(vars)
                            genes_to_keep.append(row_ind)
                            tmp_gene_vars.append(inferred_gene_var)
                        if (row_ind - myTasks[0]) == print_ind:
                            print_ind *= 2
                            mp_print(
                                "Processing data for the the %d-th feature out of %d, this took %.2f seconds." % (
                                    row_ind - myTasks[0], nTasks, time.time() - start), ONLY_RANK=0)

        if len(tmp_vars) == 0:
            ltqsVars = np.zeros((0, nCells))
            ltqs = np.zeros((0, nCells))
            genes_to_keep = np.zeros(0, dtype=int)
            gene_vars = np.zeros(0)
        else:
            ltqsVars = np.vstack(tmp_vars)
            ltqs = np.vstack(tmp_means)
            genes_to_keep = np.array(genes_to_keep)
            gene_vars = np.array(tmp_gene_vars)
    else:
        # In this case, we do not read in standard deviations, all genes are kept.
        ltqStdsFound = False
        if stdspath is not None:
            mp_print("****\nNo standard-deviations found for features at {}. Assuming very small error-bar instead. "
                     "\nCheck if this is the desired behaviour!!!\n****".format(stdspath), ERROR=True)
        else:
            mp_print("****\nNo filename is given for standard-deviations. Check if this is the desired behavior.\n****",
                     WARNING=True)
        with open(meanspath, 'r') as fmeans:
            for row_ind in range(myTasks[0]):
                next(fmeans)
            reader_means = csv.reader(fmeans, delimiter='\t')
            for row_ind in range(myTasks[0], myTasks[1]):
                means = np.asarray(next(reader_means), dtype='float')
                inferred_gene_var = np.var(means)
                if inferred_gene_var > 0.0:
                    tmp_means.append(means)
                    genes_to_keep.append(row_ind)
                    tmp_gene_vars.append(inferred_gene_var)
                if (row_ind - myTasks[0]) == print_ind:
                    print_ind *= 2
                    mp_print("Processing data for the the %d-th feature out of %d, this took %.2f seconds." % (
                        row_ind - myTasks[0], nTasks, time.time() - start), ONLY_RANK=0)
        if len(tmp_means) == 0:
            ltqsVars = np.zeros((0, nCells))
            ltqs = np.zeros((0, nCells))
            genes_to_keep = np.zeros(0, dtype=int)
            gene_vars = np.zeros(0)
        else:
            ltqs = np.vstack(tmp_means)
            ltqsVars = np.ones(ltqs.shape) * 1e-6
            nGenes = ltqs.shape[0]
            genes_to_keep = np.array(genes_to_keep)
            gene_vars = np.array(tmp_gene_vars)

    if mpiInfo.size > 1:
        # Write the data gathered on each process to a numpy file
        if not os.path.exists(tmp_folder):
            Path(tmp_folder).mkdir(parents=True, exist_ok=True)
        fname = os.path.join(tmp_folder, 'tmp_data_proc_{}.npz'.format(mpiInfo.rank))
        np.savez(fname, genes_to_keep=genes_to_keep, gene_vars=gene_vars, ltqs=ltqs, ltqsVars=ltqsVars)

        # Wait until all processes are done
        mpi_wrapper.barrier()

        # Gather data only on process 0
        if mpiInfo.rank == 0:
            genes_to_keep_list = []
            gene_vars_list = []
            ltqs_list = []
            ltqs_vars_list = []
            for proc in range(mpiInfo.size):
                fname = os.path.join(tmp_folder, 'tmp_data_proc_{}.npz'.format(proc))
                data = np.load(fname)

                genes_to_keep_list.append(data["genes_to_keep"])
                gene_vars_list.append(data["gene_vars"])
                ltqs_list.append(data["ltqs"])
                ltqs_vars_list.append(data["ltqsVars"])

            # Concatenate vertically
            genes_to_keep = np.concatenate(genes_to_keep_list, axis=0)
            gene_vars = np.concatenate(gene_vars_list, axis=0)
            ltqs = np.vstack(ltqs_list)
            ltqsVars = np.vstack(ltqs_vars_list)
            del ltqs_list
            del ltqs_vars_list
            gc.collect()

            # Check if there are genes to continue, otherwise communicate with other processes to exit
            if len(genes_to_keep) <= 1:
                continueYN = False
                mpi_wrapper.bcast(continueYN, root=0)
                if len(genes_to_keep) == 0:
                    exit("No gene made the zscore-cutoff. Considering lowering the cutoff.")
                else:
                    exit("Only one gene made the zscore-cutoff. Considering lowering the zscore-cutoff.")
            continueYN = True
            mpi_wrapper.bcast(continueYN, root=0)

            # Remove the communicated data
            if (mpiInfo.rank == 0) and (tmp_folder is not None):
                remove_folder(tmp_folder)
        else:
            continueOrBreak = None
            continueYN = mpi_wrapper.bcast(continueOrBreak, root=0)
            if not continueYN:
                exit()

    else:
        if ltqs.shape[0] == 0:
            exit("Exiting: No gene made the zscore-cutoff. Consider lowering the cutoff.")
        elif ltqs.shape[0] == 1:
            exit("Exiting: Only one gene made the zscore-cutoff. Consider lowering the cutoff.")

    """REPLACE FROM HERE!"""
    # if mpiInfo.size > 1:
    #     # Make all processes communicate the read-in data with process 0.
    #     ltqsInfo = np.concatenate((genes_to_keep[:, None], gene_vars[:, None], ltqs, ltqsVars), axis=1)
    #     mp_print("Size of ltqsInfo that is communicated with other processes: ", ltqsInfo.shape, ONLY_RANK=0)
    #     ltqsInfo = mpi_wrapper.GatherNpUnknownSize(ltqsInfo, root=0)
    #     if mpiInfo.rank == 0:
    #         if ltqsInfo.shape[0] <= 1:
    #             # Check if there are genes to continue, otherwise communicate with other processes to exit
    #             continueYN = False
    #             mpi_wrapper.bcast(continueYN, root=0)
    #             if ltqsInfo.shape[0] == 0:
    #                 exit("No gene made the zscore-cutoff. Considering lowering the cutoff.")
    #             else:
    #                 exit("Only one gene made the zscore-cutoff. Considering lowering the zscore-cutoff.")
    #         # Gather all information from all processes, this is now a matrix with a gene on each row, and then blocks
    #         # with, respectively, gene_inds, means, variances
    #         ltqsInfo = np.vstack(ltqsInfo)
    #         genes_to_keep, gene_vars, ltqs, ltqsVars = np.array_split(ltqsInfo, indices_or_sections=[1, 2, nCells + 2],
    #                                                                   axis=1)
    #         continueYN = True
    #         mpi_wrapper.bcast(continueYN, root=0)
    #     else:
    #         continueOrBreak = None
    #         continueYN = mpi_wrapper.bcast(continueOrBreak, root=0)
    #         if not continueYN:
    #             exit()
    # else:
    #     if ltqs.shape[0] == 0:
    #         exit("Exiting: No gene made the zscore-cutoff. Consider lowering the cutoff.")
    #     elif ltqs.shape[0] == 1:
    #         exit("Exiting: Only one gene made the zscore-cutoff. Consider lowering the cutoff.")
    """REPLACE UNTIL HERE!"""

    if mpiInfo.rank != 0:
        # Save memory by removing all data-variables from processes other than 0
        del ltqs
        del ltqsVars
        gc.collect()
        return None, None, None, None, None, None, None, None
    # genes_to_keep = genes_to_keep.flatten().astype(dtype=int)
    # gene_vars = gene_vars.flatten()
    nGenes = ltqs.shape[0]
    if verbose:
        mp_print("Processed %d genes in %.2f seconds, found %d cells. At zscore-cutoff of %f, %d genes were kept." % (
            nGenesOrig, time.time() - start, nCells, zscoreCutoff, nGenes))

    return ltqs, ltqsVars, gene_vars, nCells, nGenes, genes_to_keep, ltqStdsFound, nGenesOrig


def storeData(metadata, ltqs, ltqsVars, verbose=False):
    datafolder = metadata.processedDatafolder
    Path(datafolder).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(datafolder, "delta.npy"), ltqs, allow_pickle=False)
    np.save(os.path.join(datafolder, "delta_vars.npy"), ltqsVars, allow_pickle=False)
    metadata.to_json(os.path.join(datafolder, 'metadata.json'))


def get_bonsai_posteriors(final_bonsai_folder, vert_ids):
    bonsai_ltqs_vg = np.load(os.path.join(final_bonsai_folder, 'posterior_ltqs_vertByGene.npy'))
    bonsai_ltqsVars_vg = np.load(os.path.join(final_bonsai_folder, 'posterior_ltqsVars_vertByGene.npy'))
    num_dims = bonsai_ltqs_vg.shape[1]
    bonsai_ltqs_gc = np.zeros((num_dims, len(vert_ids)))
    bonsai_ltqsVars_gc = np.zeros((num_dims, len(vert_ids)))
    # Read in vertInfo and cellIds
    node_id_to_vert_ind = {}
    vertFile = os.path.join(final_bonsai_folder, 'vertInfo.txt')
    with open(vertFile, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        skipRow = True
        for row in reader:
            if skipRow:
                skipRow = False
                continue
            node_id_to_vert_ind[row[2]] = int(row[0])
    # metadata = Metadata(json_filepath=os.path.join(args.bonsai_results, 'metadata.json'))
    for ind_cell, vert_id in enumerate(vert_ids):
        if vert_id in node_id_to_vert_ind:
            bonsai_ltqs_gc[:, ind_cell] = bonsai_ltqs_vg[node_id_to_vert_ind[vert_id], :]
            bonsai_ltqsVars_gc[:, ind_cell] = bonsai_ltqsVars_vg[node_id_to_vert_ind[vert_id], :]
        else:
            mp_print("Couldn't find vert_id {} in Bonsai-output.", ERROR=True)
    return bonsai_ltqs_gc, bonsai_ltqsVars_gc


def get_bonsai_euclidean_distances(final_bonsai_folder, cell_ids):
    bonsai_ltqs_gc, _ = get_bonsai_posteriors(final_bonsai_folder, vert_ids=cell_ids)
    num_dims = bonsai_ltqs_gc.shape[0]
    bonsai_dists = distance.pdist(bonsai_ltqs_gc.T, metric='sqeuclidean') / num_dims
    return bonsai_dists
