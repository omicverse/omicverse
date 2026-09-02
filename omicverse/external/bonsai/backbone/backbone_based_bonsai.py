from argparse import ArgumentParser
import numpy as np
import time
from pathlib import Path
import os, sys, csv
import subprocess
import pandas as pd

# TODO: REMOVE THIS EVENTUALLY
# import tracemalloc
#
# tracemalloc.start()


from ..bonsai.bonsai_helpers import Run_Configs, remove_tree_folders, find_latest_tree_folder_new, \
    convert_dict_to_named_tuple, str2bool, read_ids, write_ids



class _Completed:
    """Stands in for subprocess.CompletedProcess.

    The call sites only read ``.stdout``/``.stderr`` to echo them; running the
    step in process sends its output straight to the console instead, so these
    are empty.
    """
    stdout = ""
    stderr = ""


# omicverse: upstream dispatches each step by spawning `python <script> <args>`.
# The scripts are vendored inside this package, so they are imported and called
# instead -- same steps, same arguments, one process.
_STEPS = {
    "bonsai_iterative_first_split.py": ("backbone", "bonsai_iterative_first_split"),
    "bonsai_add_cells.py": ("backbone", "bonsai_add_cells"),
    "bonsai_main.py": ("bonsai", "bonsai_main"),
    "create_config_file.py": ("bonsai", "create_config_file"),
}


def _run_step(argv):
    """Run one vendored step in process. ``argv`` is upstream's command list,
    whose first element is the script path."""
    import importlib

    script = str(argv[0]).split("/")[-1]
    if script not in _STEPS:
        raise ValueError(f"unknown Bonsai step: {argv[0]}")
    pkg, name = _STEPS[script]
    mod = importlib.import_module(f"...{pkg}.{name}", __name__)
    mod.main(mod._build_parser().parse_args([str(a) for a in argv[1:]]))
    return _Completed()


def main(args):
    """Run this step end to end.

    Wrapped by omicverse: upstream executed this body at module level, so the
    module could not be imported without running a whole step. ``args`` is the
    namespace upstream's ArgumentParser produced.
    """

    # TODO: Make sure that Run_configs initialization just copies all arguments that were originally in args
    pickup_intermediate = args.pickup_intermediate
    # growth_before_cleanup = args.growth_before_cleanup
    # select_target = args.select_target
    # iterative_cell_lists = args.iterative_cell_lists
    # n_initial_cells = args.n_initial_cells
    # return_commands = args.return_commands
    # config_filepath = args.config_filepath
    # growth_factor_guide = args.growth_factor_guide
    # search_tol = args.search_tol
    seed = args.seed
    args_to_copy = ['growth_before_cleanup', 'select_target', 'iterative_cell_lists', 'n_initial_cells', 'return_commands',
                    'config_filepath', 'growth_factor_guide', 'search_tol']

    args = Run_Configs(args.config_filepath, args=args, args_to_copy=args_to_copy)

    # args.search_tol = search_tol
    # args.select_target = select_target
    # args.iterative_cell_lists = iterative_cell_lists
    # args.growth_before_cleanup = growth_before_cleanup
    # args.n_initial_cells = n_initial_cells
    # args.return_commands = return_commands
    # args.config_filepath = config_filepath
    # args.growth_factor_guide = growth_factor_guide
    if pickup_intermediate:
        args.pickup_intermediate = True

    from ..bonsai.bonsai_dataprocessing import SCData, Metadata
    from ..bonsai.bonsai_helpers import mp_print, startMPI, getOutputFolder
    from ..bonsai import bonsai_globals as bs_glob

    # Some of the code can be run in parallel, parallelization is done via mpi4py
    mpi_rank, mpi_size = startMPI(verbose=True)

    start_all = time.time()

    if mpi_rank != 0:
        mp_print("The script 'backbone_based_bonsai.py' is not designed to use multiple CPUs in all steps."
                 "Instead, please run this script with the command '--return_commands True'. "
                 "It will then return the commands in a text-file "
                 "(and printed out on the console) that you can then run yourself using multiple CPUs.",
                 ALL_RANKS=True, WARNING=True)
        exit()

    """
    ------------------------------------------------------------------------------------------
    Step 1: Get a subset of the cells and store their information in a subfolder.
    ------------------------------------------------------------------------------------------
    """

    scdata = SCData(onlyObject=True, dataset=args.dataset, results_folder=args.results_folder)
    if args.return_commands:
        commands_file = scdata.result_path('iterative_build_commands.txt')

    scdata.metadata.pathToOrigData = args.data_folder
    # Get cell-IDs of whole dataset
    if os.path.exists(scdata.data_path('cellID.txt')):
        scdata.metadata.cellIds = read_ids(scdata.data_path('cellID.txt'))
        scdata.metadata.nCells = len(scdata.metadata.cellIds)
        n_unq_ids = len(np.unique(scdata.metadata.cellIds))
        if scdata.metadata.nCells is not None:
            if n_unq_ids != scdata.metadata.nCells:
                mp_print("Cell-IDs are not unique.\n "
                         "Please resolve this and start this script again.", ERROR=True)
                exit()

    scdata.metadata.processedDatafolder = scdata.result_path('zscorefiltered_%.3f_and_processed' % args.zscore_cutoff)

    # Select a subset of the cell-IDs to build a first guide-tree on
    np.random.seed(seed)

    if args.n_initial_cells >= scdata.metadata.nCells:
        mp_print("The dataset is smaller than your proposed starting tree. In that case, you can just run the normal"
                 "Bonsai-algorithm.", ERROR=True)
        exit()
    n_initial_cells = min(args.n_initial_cells, scdata.metadata.nCells)
    guide_tree_sizes = [n_initial_cells]
    # Check how often we have to grow the tree by a factor of growth_factor_guide
    while guide_tree_sizes[-1] < scdata.metadata.nCells:
        guide_tree_sizes.append(int(min(args.growth_factor_guide * guide_tree_sizes[-1], scdata.metadata.nCells)))
    subsets = [None] * len(guide_tree_sizes)
    subsets[-1] = np.arange(scdata.metadata.nCells)
    for subset_ind in range(len(guide_tree_sizes) - 2, -1, -1):
        subsets[subset_ind] = np.random.choice(subsets[subset_ind + 1], size=guide_tree_sizes[subset_ind], replace=False)

    # Create annotation file that stores which cell belongs to which subset
    which_subset = [None] * scdata.metadata.nCells
    for subset_ind, subset in enumerate(subsets):
        for cell_ind in subset:
            if which_subset[cell_ind] is None:
                which_subset[cell_ind] = "subset_{}".format(subset_ind)
    annotation_df = pd.DataFrame({'iteration_subset': which_subset}, index=scdata.metadata.cellIds)
    annotation_df.to_csv(os.path.join(scdata.result_path(), 'iteration_subset_annotation.tsv'), sep='\t')

    scdata_subsets = []
    for subset_ind, subset in enumerate(subsets):
        cell_ids_subset = [scdata.metadata.cellIds[ind] for ind in subset]
        if subset_ind == len(subsets) - 1:
            scdata_subsets.append(scdata)
            folder_for_nonrefined_tree = scdata.result_path('non_refined_tree')
            Path(folder_for_nonrefined_tree).mkdir(parents=True, exist_ok=True)
            write_ids(os.path.join(folder_for_nonrefined_tree, 'starting_cell_ids.txt'), cell_ids_subset)
            break

        # Also create a scdata-object for the subset
        metadata_subset = Metadata(curr_metadata=scdata.metadata)
        metadata_subset.cellIds = cell_ids_subset
        metadata_subset.nCells = len(cell_ids_subset)
        bs_glob.nCells = metadata_subset.nCells
        metadata_subset.dataset = os.path.join(scdata.metadata.dataset, 'subset_{}'.format(subset_ind))
        metadata_subset.results_folder = os.path.join(scdata.metadata.results_folder, 'subset_{}'.format(subset_ind))

        scdata_subset = SCData(onlyObject=True, dataset=metadata_subset.dataset,
                               results_folder=metadata_subset.results_folder)

        scdata_subset.metadata = metadata_subset
        scdata_subset.metadata.processedDatafolder = scdata_subset.result_path('zscorefiltered_%.3f_and_processed' %
                                                                               args.zscore_cutoff)

        # Store this subset in a file such that the other script can read it in.
        mp_print("Writing cell IDs to file:")
        folder_for_nonrefined_tree = scdata_subset.result_path('non_refined_tree')
        Path(folder_for_nonrefined_tree).mkdir(parents=True, exist_ok=True)
        write_ids(os.path.join(folder_for_nonrefined_tree, 'starting_cell_ids.txt'), cell_ids_subset)
        scdata_subsets.append(scdata_subset)

    """-----------For the first subset, we run a script that creates a star-tree for the first guide-tree---------"""
    # Run the script that will read in and preprocess *all* the data, and create the star-tree for the first guide-tree
    subset_results_folders = [scdata_subsets[ind].metadata.results_folder for ind in range(len(scdata_subsets)-1)]
    subset_results_folders = ','.join(subset_results_folders)
    preprocess_cmd = ['backbone_based_bonsai/bonsai_iterative_first_split.py',
                      '--config_filepath', args.config_filepath,
                      '--subset_results_folder', subset_results_folders]

    if not args.return_commands:
        output1 = _run_step(preprocess_cmd)
        mp_print(output1.stdout)
        mp_print(output1.stderr)
    else:
        cmd = ' '.join(preprocess_cmd)
        mp_print("Command for creating initial Bonsai guide-tree (stored in {}):\n "
                 "{}".format(commands_file, cmd))
        with open(commands_file, "w") as file:
            file.write(cmd + '\n')

    """
    ------------------------------------------------------------------------------------------
    Step 2: Run Bonsai on the next subset, then run adding cells. Iterate, and finish with a Bonsai-run.
    ------------------------------------------------------------------------------------------
    """


    def get_command_config_file(args, dataset, result_path, data_folder=None, tmp_folder=None):
        config_filepath = os.path.join(result_path, 'iterative_configs.yaml')
        if data_folder is None:
            data_folder = result_path
        command1 = [sys.executable,
                    'bonsai/create_config_file.py',
                    '--new_yaml_path',
                    config_filepath,
                    '--dataset',
                    dataset,
                    '--data_folder',
                    data_folder,
                    '--results_folder',
                    result_path,
                    '--input_is_sanity_output',
                    str(args.input_is_sanity_output),
                    '--zscore_cutoff',
                    str(args.zscore_cutoff),
                    '--UB_ellipsoid_size',
                    str(args.UB_ellipsoid_size),
                    '--nnn_n_randommoves',
                    str(args.nnn_n_randommoves),
                    '--nnn_n_randomtrees',
                    str(args.nnn_n_randomtrees),
                    '--pickup_intermediate',
                    'True',
                    '--use_knn',
                    str(args.use_knn),
                    '--rescale_by_var',
                    str(args.rescale_by_var)]
        if tmp_folder is not None:
            command1 += ['--tmp_folder', tmp_folder]
        return command1, config_filepath


    for subset_ind, scdata_subset in enumerate(scdata_subsets):
        data_folder = scdata_subset.result_path() if subset_ind != len(scdata_subsets) - 1 else args.data_folder
        non_refined_folder = scdata_subset.result_path('non_refined_tree')
        # Create new config-file for the subset
        create_config_command_1, config_filepath = get_command_config_file(args, scdata_subset.metadata.dataset,
                                                                           scdata_subset.result_path(),
                                                                           data_folder=data_folder,
                                                                           tmp_folder=non_refined_folder)
        output1 = _run_step(create_config_command_1[1:])
        mp_print(output1.stdout)
        mp_print(output1.stderr)

        # Run Bonsai on the new config-file
        bonsai_subset1_cmd = ['bonsai/bonsai_main.py',
                              '--config_filepath', config_filepath,
                              '--step', 'all',
                              '--pickup_intermediate', 'True']

        if not args.return_commands:
            output1 = _run_step(bonsai_subset1_cmd)
            mp_print(output1.stdout)
            mp_print(output1.stderr)
        else:
            cmd = ' '.join(bonsai_subset1_cmd)
            mp_print("Command for creating refined Bonsai-tree based on subset {}, (stored in {}):\n "
                     "{}".format(subset_ind, commands_file, cmd))
            with open(commands_file, "a") as file:
                file.write(cmd + '\n')

        results_dir_subset = scdata_subset.result_path(getOutputFolder(zscore_cutoff=args.zscore_cutoff, final=True,
                                                                       tmp_file=os.path.basename(non_refined_folder)))

        """
        ------------------------------------------------------------------------------------------
        Add the next batch of cells to the just created Bonsai-tree.
        ------------------------------------------------------------------------------------------
        """

        if subset_ind == len(scdata_subsets) - 1:
            # In this case, we have added all the cells, so we can end the for-loop now
            break
        # final_dataset_name = os.path.join(scdata.metadata.dataset, 'subset_final')
        # final_result_folder = os.path.join(scdata.result_path(), 'subset_final')
        # create_config_command_1, config_filepath = get_command_config_file(args, final_dataset_name, final_result_folder,
        #                                                                    data_folder=args.data_folder,
        #                                                                    tmp_folder=scdata_subset.result_path())
        # output2 = subprocess.run(create_config_command_1, stdout=subprocess.PIPE, text=True)
        # mp_print(output1.stdout)
        # mp_print(output1.stderr)

        # Create new config-file for the subset
        new_non_refined_folder = os.path.join(scdata_subsets[subset_ind + 1].result_path(), 'non_refined_tree')
        create_config_command_addcells, config_filepath_add = get_command_config_file(args, scdata_subset.metadata.dataset,
                                                                                      new_non_refined_folder,
                                                                                      data_folder=scdata_subset.result_path())
        output1 = _run_step(create_config_command_addcells[1:])
        mp_print(output1.stdout)
        mp_print(output1.stderr)

        add_cells_seed = np.random.randint(1e6)
        add_cells_cmd = ['backbone_based_bonsai/bonsai_add_cells.py',
                         '--config_filepath', config_filepath_add,
                         '--guide_tree_folder', results_dir_subset,
                         '--cells_to_be_added', os.path.join(new_non_refined_folder, 'starting_cell_ids.txt'),
                         '--preprocessed_data_folder', scdata_subsets[subset_ind+1].metadata.processedDatafolder,
                         '--growth_before_cleanup', str(args.growth_before_cleanup),
                         '--select_target', args.select_target,
                         '--search_tol', str(args.search_tol),
                         '--seed', str(add_cells_seed),
                         '--pickup_intermediate', str(args.pickup_intermediate)]
        # TODO: Add arguments '--nodes_to_add_to', '--cels_to_be_added' later

        if not args.return_commands:
            output1 = _run_step(add_cells_cmd)
            mp_print(output1.stdout)
            mp_print(output1.stderr)
        else:
            cmd = ' '.join(add_cells_cmd)
            mp_print("Command for adding remaining cells to guide tree (stored in {}):\n{}".format(commands_file, cmd))
            with open(commands_file, "a") as file:
                file.write(cmd + '\n')

    """
    # ------------------------------------------------------------------------------------------
    # Step 4: Run Bonsai one more time starting from the created tree to see what spr-moves and NNI moves are still possible
    # ------------------------------------------------------------------------------------------
    # """
    #
    # create_config_command_1, config_filepath = get_command_config_file(args, scdata.metadata.dataset, scdata.result_path(),
    #                                                                    tmp_folder=final_result_folder)
    # output2 = subprocess.run(create_config_command_1, stdout=subprocess.PIPE, text=True)
    # mp_print(output1.stdout)
    # mp_print(output1.stderr)
    #
    # # Run Bonsai on the new config-file
    # bonsai_subset1_cmd = ['bonsai/bonsai_main.py',
    #                       '--config_filepath', config_filepath,
    #                       '--step', 'all',
    #                       '--pickup_intermediate', 'True']
    #
    # if not args.return_commands:
    #     output1 = subprocess.run([sys.executable] + bonsai_subset1_cmd, stdout=subprocess.PIPE, text=True)
    #     mp_print(output1.stdout)
    #     mp_print(output1.stderr)
    # else:
    #     cmd = ' '.join(bonsai_subset1_cmd)
    #     mp_print("Command for creating initial Bonsai guide-tree (stored in {}):\n "
    #              "{}".format(commands_file, cmd))
    #     with open(commands_file, "a") as file:
    #         file.write(cmd + '\n')

    print_text = 'only printing commands' if args.return_commands else "performing all calculations"
    mp_print("Running the iterative-bonsai script while {} took {} seconds.".format(print_text, time.time() - start_all))



def _build_parser():
    """Upstream's ArgumentParser, lifted out of ``__main__`` so omicverse can
    reuse it to build an args namespace without spawning a process."""
    parser = ArgumentParser(
        description='This script provides a way of running backbone-based-Bonsai, consisting of the steps'
                    '1) Preprocess the data'
                    '2) Build a normal Bonsai-tree on a subset of the data as a backbone'
                    '3) Grow this backbone by a factor "growth_factor_guide" (or just add all remaining cells)'
                    '4) Refine the tree using Bonsai.'
                    ''
                    'There are a few arguments that one can set below. Importantly, if you run this script with '
                    'return_commands=False, it will try to run the whole calculation with just a single CPU. If you set'
                    'return_commands=True, it will just print out the commands to a file, which you can then use to run'
                    'the separate steps with more optimized computational resources.')
    parser.add_argument('--config_filepath', type=str, default=None,
                        help='Absolute (or relative to "bonsai-development") path to YAML-file that contains all arguments'
                             'needed to run Bonsai.')
    parser.add_argument('--select_target', type=str, default='cluster_centers',
                        help="This will determine what strategy we follow for the "
                             "spr-moves. Current options are 'root', 'cluster_centers', 'exhaustive'")
    parser.add_argument('--search_tol', type=float, default=2,
                        help="Gives the loglikelihood-margin by which a new SPR-search has to be worse than a previous one"
                             "before we discard this search direction")
    parser.add_argument('--iterative_cell_lists', type=str, default=None,
                        help="[NOT YET IMPLEMENTED]"
                             "One can give a filename here for a file where we have all cell-IDs (one per line) in the "
                             "first column. Second column contains an integer indicating at which iteration the cells "
                             "should be added. Useful for modeling time-course data.")
    parser.add_argument('--n_initial_cells', type=int, default=10000,
                        help="If --iterative_cell_lists is None, this number will determine the size of the subset of "
                             "cells on which we will make the initial guide-tree.")
    parser.add_argument('--growth_before_cleanup', type=float, default=.5,
                        help="When adding cells, this factor (larger than 0) determines after what growth factor (so .1 "
                             "means 10% growth) we re-optimize the tree before adding another set of cells.")
    parser.add_argument('--growth_factor_guide', type=float, default=10,
                        help="Starting from the guide tree, we add cells until the tree is growth_factor_guide as large,"
                             "then we refine this tree with a normal Bonsai-run, and then add the next factor of cells..")
    parser.add_argument('--return_commands', type=str2bool, default=False,
                        help="If True, this script will only return a file with the commands that should be run, "
                             "but not run them. This is useful for running these commands using parallel computing.\n"
                             "If False, this script will just run these commands on a single CPU. MPI will not be used.")
    parser.add_argument('--pickup_intermediate', type=str2bool, default=False,
                        help='Optional: Set this argument to true if Bonsai needs to search in the indicated results-folder'
                             'for the furthest developed tree reconstruction, and pick it up from there. If this argument'
                             'is True, it over-rules the same argument in bonsai_config.yaml, and the other way around.')
    parser.add_argument('--seed', type=int, default=1231,
                        help="Sets the random seed necessary for getting the subset of cells.")
    return parser


if __name__ == "__main__":
    main(_build_parser().parse_args())
