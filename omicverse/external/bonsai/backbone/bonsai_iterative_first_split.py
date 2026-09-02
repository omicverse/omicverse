from argparse import ArgumentParser
import numpy as np
import time
from pathlib import Path
import os, sys, csv
import subprocess


# import tracemalloc
# tracemalloc.start()

from ..bonsai.bonsai_helpers import Run_Configs, remove_tree_folders, find_latest_tree_folder_new, \
    convert_dict_to_named_tuple, str2bool


def main(args):
    """Run this step end to end.

    Wrapped by omicverse: upstream executed this body at module level, so the
    module could not be imported without running a whole step. ``args`` is the
    namespace upstream's ArgumentParser produced.
    """

    start_all = time.time()

    # TODO: Make sure that Run_configs initialization just copies all arguments that were originally in args
    config_filepath = args.config_filepath
    subset_results_folders = args.subset_results_folders

    args = Run_Configs(args.config_filepath)

    args.subset_results_folders = subset_results_folders
    args.config_filepath = config_filepath

    from ..bonsai import mpi_wrapper as mpi_wrapper
    from ..bonsai.bonsai_dataprocessing import initializeSCData, SCData, Metadata, storeData
    from ..bonsai.bonsai_helpers import mp_print, startMPI, getOutputFolder, read_ids
    from ..bonsai.bonsai_treeHelpers import Tree
    from ..bonsai import bonsai_globals as bs_glob

    # Some of the code can be run in parallel, parallelization is done via mpi4py
    mpi_rank, mpi_size = startMPI(verbose=True)

    scdata = initializeSCData(args, createStarTree=False, getOrigData=True, otherRanksMinimalInfo=True)
    orig_cell_id_to_ind = {cell_id: ind for ind, cell_id in enumerate(scdata.metadata.cellIds)}

    if mpi_rank != 0:
        mp_print("My job here is done.", ALL_RANKS=True)
        exit()

    subset_folders = args.subset_results_folders.split(',')
    for ind_subset, subset_folder in enumerate(subset_folders):
        cell_ids_subset = read_ids(os.path.join(subset_folder, 'non_refined_tree', 'starting_cell_ids.txt'))
        subset_inds = np.array([orig_cell_id_to_ind[cell_id] for cell_id in cell_ids_subset])

        metadata_subset = Metadata(curr_metadata=scdata.metadata)
        metadata_subset.cellIds = cell_ids_subset
        metadata_subset.nCells = len(cell_ids_subset)
        bs_glob.nCells = metadata_subset.nCells
        metadata_subset.dataset = os.path.join(scdata.metadata.dataset, os.path.basename(subset_folder))
        metadata_subset.results_folder = subset_folder

        scdata_subset = SCData(onlyObject=True, dataset=metadata_subset.dataset,
                               results_folder=metadata_subset.results_folder)

        scdata_subset.metadata = metadata_subset
        ltqs_subset = scdata.originalData.ltqs[:, subset_inds].copy()
        ltqsvars_subset = scdata.originalData.ltqsVars[:, subset_inds].copy()

        scdata_subset.metadata.processedDatafolder = scdata_subset.result_path('zscorefiltered_%.3f_and_processed'
                                                                               % args.zscore_cutoff)
        storeData(scdata_subset.metadata, ltqs_subset, ltqsvars_subset)
        mp_print("Storing subset {} with {} cells in folder: {}".format(ind_subset, len(subset_inds),
                                                                        scdata_subset.metadata.processedDatafolder))
        # storeData(scdata_subset.metadata, ltqs_subset, ltqsvars_subset)

        if (mpi_rank == 0) and (ind_subset == 0):
            scdata_subset.metadata.dataset = os.path.join(scdata_subset.metadata.dataset, 'non_refined_tree')
            results_folder_initial = scdata_subset.result_path('non_refined_tree')
            Path(results_folder_initial).mkdir(parents=True, exist_ok=True)
            scdata_subset.tree = Tree()
            scdata_subset.tree.initialize_star_tree(ltqs_subset, ltqsvars_subset, scdata_subset.metadata,
                                                    opt_times=True, verbose=args.verbose)
            # Store tree topology with optimised times, and the data only for selected genes, such that it can be read
            # in by multiple cores such that the next part of the program can be run in parallel
            # outputFolder = getOutputFolder(zscore_cutoff=args.zscore_cutoff, greedy=False,
            #                                redo_starry=False, opt_times=False)
            mp_print("Storing result of preprocessing in " + results_folder_initial + "\n\n")
            scdata_subset.storeTreeInFolder(results_folder_initial, with_coords=True, verbose=args.verbose,
                                            cleanup_tree=False)

    mp_print("Reading, filtering, and storing data took " + str(time.time() - start_all) + " seconds.")
    mp_print("Time necessary for the whole calculation was {} seconds.".format(time.time() - start_all))
    # print_memory("Memory usage after the whole calculation")



def _build_parser():
    """Upstream's ArgumentParser, lifted out of ``__main__`` so omicverse can
    reuse it to build an args namespace without spawning a process."""
    parser = ArgumentParser(
        description='First script to be called in backbone_based_bonsai.py.'
                    'Creates a first star-tree on a first subset of the data. Preprocesses all the data to store the'
                    'preprocessed data in the results-folder of all subsets already.')
    parser.add_argument('--config_filepath', type=str, default=None,
                        help='Absolute (or relative to "bonsai-development") path to YAML-file that contains all arguments'
                             'needed to run Bonsai.')
    parser.add_argument('--subset_results_folders', type=str, default=None,
                        help="Comma-separated string containing the results folders where tree for the subsets should "
                             "eventually be stored. "
                             "Also folders where we can find 'non_refined_tree/starting_cell_ids.txt' to know from "
                             "which cells to start."
                             "NOTE: It is assumed that making a star-tree is only necessary for the FIRST subset.")
    return parser


if __name__ == "__main__":
    main(_build_parser().parse_args())
