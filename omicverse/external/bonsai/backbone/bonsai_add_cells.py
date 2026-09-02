from argparse import ArgumentParser
import numpy as np
import time
import os, sys, csv
from pathlib import Path

# import tracemalloc
#
# tracemalloc.start()


from ..bonsai.bonsai_helpers import Run_Configs, remove_tree_folders, find_latest_tree_folder_new, \
    convert_dict_to_named_tuple, str2bool, get_latest_intermediate


def main(args):
    """Run this step end to end.

    Wrapped by omicverse: upstream executed this body at module level, so the
    module could not be imported without running a whole step. ``args`` is the
    namespace upstream's ArgumentParser produced.
    """
    np.random.seed(args.seed)

    # args = Run_Configs(args.config_filepath)
    args_to_copy = ['preprocessed_data_folder', 'growth_before_cleanup', 'select_target', 'guide_tree_folder',
                    'cells_to_be_added', 'pickup_intermediate', 'search_tol']
    args = Run_Configs(args.config_filepath, args=args, args_to_copy=args_to_copy)

    from ..bonsai.bonsai_dataprocessing import initializeSCData, loadReconstructedTreeAndData, SCData, \
        OriginalData, Metadata
    from ..bonsai.bonsai_helpers import mp_print, startMPI
    from ..bonsai import bonsai_globals as bs_glob

    if args.guide_tree_folder is None:
        mp_print("Cannot add cells if 'guide_tree_folder' is not defined, don't know where to find the guide tree.",
                 ERROR=True)
        exit()

    # Some of the code can be run in parallel, parallelization is done via mpi4py
    mpiRank, mpiSize = startMPI(args.verbose)
    origEllipsoidSize = None

    start_all = time.time()

    # Read in the tree that was created on a subset of the data (read in without data)
    # scdata_tmp = SCData(onlyObject=True, dataset=args.dataset, results_folder=args.results_folder)
    # results_folder = scdata_tmp.result_path()

    all_genes = False

    # Now try to find if some cells were already added in an earlier (aborted) run
    scdata_tmp = SCData(onlyObject=True, dataset=args.dataset, results_folder=args.results_folder)
    tmp_folder = scdata_tmp.result_path('tmp_added')
    Path(tmp_folder).mkdir(parents=True, exist_ok=True)
    tmp_found = False
    if (mpiRank == 0) and args.pickup_intermediate and os.path.exists(tmp_folder):
        try:
            intermediateFolders = os.listdir(tmp_folder)
            if len(intermediateFolders):
                intermediateFolder, tmp_tree_ind = get_latest_intermediate(intermediateFolders, base='added')
                if intermediateFolder is not None:
                    # scdata_guide = loadReconstructedTreeAndData(args, os.path.join(tmp_folder, intermediateFolder),
                    #                                             reprocess_data=False, all_genes=False, get_cell_info=False,
                    #                                             all_ranks=False, rel_to_results=False, calc_loglik=True)
                    scdata_guide = loadReconstructedTreeAndData(args, os.path.join(tmp_folder, intermediateFolder),
                                                                all_genes=all_genes, get_cell_info=False,
                                                                reprocess_data=False, all_ranks=False, rel_to_results=False,
                                                                get_data=False, no_data_needed=True,
                                                                get_posterior_ltqs=False, otherRanksMinimalInfo=True)
                    tmp_found = True
        except:
            mp_print("Tried to find intermediate results in {}, but could not find any. "
                     "Starting from guide tree.".format(tmp_folder))

    if not tmp_found:
        # If no intermediate result is present, take the guide tree here. Otherwise take the intermediate result.
        tmp_tree_ind = 0
        scdata_guide = loadReconstructedTreeAndData(args, args.guide_tree_folder, all_genes=all_genes, get_cell_info=False,
                                                    reprocess_data=False, all_ranks=True, rel_to_results=True,
                                                    get_data=False, no_data_needed=True, get_posterior_ltqs=False,
                                                    otherRanksMinimalInfo=True)

    # Read in the data for all cells
    if args.preprocessed_data_folder is not None:
        scdata_all_cells = SCData(onlyObject=True, dataset=args.dataset, results_folder=args.results_folder)
        scdata_all_cells.metadata = Metadata(json_filepath=os.path.join(args.preprocessed_data_folder, 'metadata.json'),
                                             curr_metadata=scdata_all_cells.metadata)

        scdata_all_cells.originalData = OriginalData()
        scdata_all_cells.originalData.ltqs = np.load(os.path.join(args.preprocessed_data_folder, 'delta.npy'),
                                                     allow_pickle=False, mmap_mode='r')
        scdata_all_cells.originalData.ltqsVars = np.load(os.path.join(args.preprocessed_data_folder, 'delta_vars.npy'),
                                                         allow_pickle=False, mmap_mode='r')
    else:
        scdata_all_cells = initializeSCData(args, createStarTree=False, getOrigData=False, otherRanksMinimalInfo=True)

    cell_id_to_ind = {cell_id: ind for ind, cell_id in enumerate(scdata_all_cells.metadata.cellIds)}

    # Create list of cells that is already on the tree, and get cell-ID to node dictionary to be able to add ltq-info
    nodes_list = scdata_guide.tree.root.getNodeList([], returnRoot=True, returnLeafs=True)

    # The gene-selection will be based on all cells, so we copy the metadata based on scdata_all_cells
    scdata_guide.metadata.geneIds = scdata_all_cells.metadata.geneIds
    scdata_guide.metadata.nGenes = len(scdata_guide.metadata.geneIds)
    # Note that we call only the original cells on the guide-tree "cells" at the moment. This is for technical reasons,
    # for numbering the nodes
    bs_glob.nCells = scdata_guide.metadata.nCells
    bs_glob.nGenes = scdata_guide.metadata.nGenes

    scdata_guide.metadata.geneVariances = scdata_all_cells.metadata.geneVariances
    scdata_guide.metadata.loglikVarCorr = scdata_all_cells.metadata.loglikVarCorr
    scdata_guide.metadata.pathToOrigData = scdata_all_cells.metadata.pathToOrigData
    scdata_guide.metadata.processedDatafolder = scdata_all_cells.metadata.processedDatafolder
    scdata_guide.metadata.results_folder = args.results_folder

    # Add data to guide-tree, take signal-to-noise genes based on all cells
    guide_cell_inds = []
    for node in nodes_list:
        if node.nodeId in cell_id_to_ind:
            # In this case, the node corresponds to a cell. We're assuming that each node has max 1 cell
            cell_ind = cell_id_to_ind[node.nodeId]
            guide_cell_inds.append(cell_ind)
            node.ltqs = scdata_all_cells.originalData.ltqs[:, cell_ind]
            node.setLtqsVarsOrW(ltqsVars=scdata_all_cells.originalData.ltqsVars[:, cell_ind])
            node.isCell = True

    guide_cell_inds = np.unique(guide_cell_inds)
    non_guide_cell_inds = np.setdiff1d(np.arange(scdata_all_cells.metadata.nCells), guide_cell_inds)
    # Select subset of non_guide-cells that we want to add based on the "cells_to_be_added"-argument
    cell_id_list_after_adding = []
    if (args.cells_to_be_added is not None) and os.path.exists(os.path.abspath(args.cells_to_be_added)):
        cells_to_be_added = os.path.abspath(args.cells_to_be_added)
        # cell_ids_to_be_added = []
        cell_inds_to_add = []
        with open(cells_to_be_added, 'r') as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                cell_id_list_after_adding.append(row[0])
                cell_inds_to_add.append(cell_id_to_ind[row[0]])
        cell_inds_to_add = np.array(cell_inds_to_add)
        cell_inds_to_add = np.intersect1d(cell_inds_to_add, non_guide_cell_inds)
    else:
        cell_inds_to_add = non_guide_cell_inds

    # Create random order of cells to be added
    np.random.shuffle(cell_inds_to_add)
    # ltqs_to_add = scdata_all_cells.originalData.ltqs[:, cell_inds_to_add]
    # ltqsvars_to_add = scdata_all_cells.originalData.ltqsVars[:, cell_inds_to_add]

    # Store ltqs to be added in a file, such that they can be memory-mapped when reading in
    n_to_add = len(cell_inds_to_add)
    ltqs_file = os.path.join(tmp_folder, 'ltqs_to_add_cell_by_gene.npy')
    ltqsvars_file = os.path.join(tmp_folder, 'ltqsvars_to_add_cell_by_gene.npy')
    ltqs_to_add = np.lib.format.open_memmap(ltqs_file, dtype='float64', mode='w+',
                                            shape=(n_to_add, bs_glob.nGenes))
    ltqsvars_to_add = np.lib.format.open_memmap(ltqsvars_file, dtype='float64', mode='w+',
                                                shape=(n_to_add, bs_glob.nGenes))
    for out_row, orig_col in enumerate(cell_inds_to_add):
        ltqs_to_add[out_row, :] = scdata_all_cells.originalData.ltqs[:, orig_col]
        ltqsvars_to_add[out_row, :] = scdata_all_cells.originalData.ltqsVars[:, orig_col]

    # Flush to disk
    ltqs_to_add.flush()
    ltqsvars_to_add.flush()
    del ltqs_to_add
    del ltqsvars_to_add
    ltqs_to_add_cg = np.load(ltqs_file, allow_pickle=False, mmap_mode='r')
    ltqsvars_to_add_cg = np.load(ltqsvars_file, allow_pickle=False, mmap_mode='r')
    cell_ids_to_add = [scdata_all_cells.metadata.cellIds[ind] for ind in cell_inds_to_add]

    # Make sure that all ltqs are calculated at all nodes (automatically done when calculating a loglikelihood)
    scdata_guide.metadata.loglik = scdata_guide.tree.calcLogLComplete(mem_friendly=True,
                                                                      loglikVarCorr=scdata_guide.metadata.loglikVarCorr)
    mp_print("Loglikelihood of guide tree before adding cells: " + str(scdata_guide.metadata.loglik))

    """
    This is the core of the script, the cells will be added iteratively to the guide-tree
    """
    if args.select_target == 'root':
        n_centers = 1
        args.select_target = 'cluster_centers'
    else:
        n_centers = None

    scdata_guide.tree.add_cells(ltqs_to_add_cg, ltqsvars_to_add_cg, cell_ids_to_add,
                                growth_before_cleanup=args.growth_before_cleanup,
                                select_target=args.select_target,
                                scdata=scdata_guide, tmp_folder=tmp_folder, tmp_tree_ind=tmp_tree_ind,
                                search_tol=args.search_tol, n_centers=n_centers, only_count_search_moves=False)

    # Make node-indices nice again: The cells have the node-ind matching position in the input cell-ID list. Root has -1,
    # Internal nodes start at nCells and increase in depth-first manner.
    scdata_guide.metadata.cellIds = cell_id_list_after_adding
    scdata_guide.metadata.nCells = len(scdata_guide.metadata.cellIds)
    bs_glob.nCells = scdata_guide.metadata.nCells
    scdata_guide.cleanup_node_inds()

    """
    Store the resulting tree in a folder, such that another script can pick it up. In that script, we should then
    decide (through some arguments) how many tree-moves we still do, or if we just store the information in the final 
    format.
    """

    mp_print("Adding cells took " + str(time.time() - start_all) + " seconds.")
    scdata_guide.metadata.loglik = scdata_guide.tree.calcLogLComplete(mem_friendly=True,
                                                                      loglikVarCorr=scdata_guide.metadata.loglikVarCorr)
    mp_print("Loglikelihood of inferred tree after adding cells: " + str(scdata_guide.metadata.loglik))

    mp_print("Storing result after adding all cells in " + scdata_guide.result_path() + "\n\n")
    scdata_guide.storeTreeInFolder(scdata_guide.result_path(), with_coords=True, verbose=args.verbose)

    if tmp_folder is not None:
        remove_tree_folders(tmp_folder, removeDir=True, base='added')

    mp_print("Time necessary for the whole calculation was {} seconds.".format(time.time() - start_all))
    # print_memory("Memory usage after the whole calculation")



def _build_parser():
    """Upstream's ArgumentParser, lifted out of ``__main__`` so omicverse can
    reuse it to build an args namespace without spawning a process."""
    parser = ArgumentParser(
        description='Starts from cell-data and a tree on which some of the cells are already placed. Cells that are not on'
                    'the tree but are in the data will be placed one by one on the tree.')
    parser.add_argument('--config_filepath', type=str, default=None,
                        help='Absolute (or relative to "bonsai-development") path to YAML-file that contains all arguments'
                             'needed to run Bonsai.')
    parser.add_argument('--guide_tree_folder', type=str, default=None,
                        help="Path that should point to folder where an existing Bonsai-tree is stored. This folder"
                             "is typically called 'final_...' in a Bonsai-results folder. Then this script will start "
                             "adding cells to that 'guide-tree'.")
    parser.add_argument('--preprocessed_data_folder', type=str, default=None,
                        help="Folderpath where data is stored that was already preprocessed for all cells.")
    parser.add_argument('--growth_before_cleanup', type=float, default=.5,
                        help="When adding cells, this factor (larger than 0) determines after what growth factor (so .1 "
                             "means 10% growth) we re-optimize the tree before adding another set of cells.")
    # parser.add_argument('--resolve_polytomies_immediately', type=str2bool, default=True,
    #                     help="Determines whether, when adding a cell downstream of a node, we immediately test whether it "
    #                          "wants to merge with one of the other children of that node.")
    # TODO: To be moved to config-file
    parser.add_argument('--select_target', type=str, default='cluster_centers',
                        help="This will determine what strategy we follow for the "
                             "spr-moves. Current options are 'root', 'cluster_centers', 'exhaustive'.")
    parser.add_argument('--nodes_to_add_to', type=str, default=None,
                        help="[NOT YET IMPLEMENTED]"
                             "One can give a filename to a txt-file containing node-IDs here (one per line). In that case,"
                             "cells will only be added directly to one of these nodes.")
    parser.add_argument('--cells_to_be_added', type=str, default=None,
                        help="One can give a filename to a txt-file containing cell-IDs here (one per line). In that case,"
                             "only these cell-IDs will be added.")
    parser.add_argument('--search_tol', type=float, default=2,
                        help="Gives the loglikelihood-margin by which a new SPR-search has to be worse than a previous one"
                             "before we discard this search direction")
    parser.add_argument('--pickup_intermediate', type=str2bool, default=True,
                        help='Optional: Set this argument to true if Bonsai needs to search in the indicated results-folder'
                             'for the furthest developed tree reconstruction, and pick it up from there. If this argument'
                             'is True, it over-rules the same argument in bonsai_config.yaml, and the other way around.')
    parser.add_argument('--seed', type=int, default=1231,
                        help="Random seed for order of adding the cells.")
    return parser


if __name__ == "__main__":
    main(_build_parser().parse_args())
