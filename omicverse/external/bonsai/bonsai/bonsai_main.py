from argparse import ArgumentParser
import numpy as np
import time
from pathlib import Path
import os, sys, psutil, gc

# TODO: Remove this later maybe
# import tracemalloc
#
# tracemalloc.start()


if sys.platform == "win32":
    import types
    m = types.ModuleType("resource")
    m.RLIMIT_AS = 0
    m.RUSAGE_SELF = 0
    m.getrusage = lambda x: types.SimpleNamespace(
        ru_maxrss=0, ru_utime=0, ru_stime=0
    )
    m.setrlimit = lambda x, y: None
    sys.modules["resource"] = m

from .bonsai_helpers import Run_Configs, remove_tree_folders, find_latest_tree_folder_new, str2bool, \
    store_scdata_and_communicate_path


def main(args):
    """Run Bonsai end to end.

    Wrapped by omicverse: upstream ran this body at module level, which made
    the module impossible to import without executing a whole run. ``args`` is
    the namespace upstream's ArgumentParser produced, so the block under
    ``__main__`` below still reproduces the original command-line behaviour.
    """

    pickup_intermediate = args.pickup_intermediate
    # TODO: Remove
    store_all_nwk_folder = args.store_all_nwk_folder
    print_annotations = args.print_annotations

    args = Run_Configs(args.config_filepath, step=args.step)

    if pickup_intermediate:
        args.pickup_intermediate = True

    from . import mpi_wrapper as mpi_wrapper
    from .bonsai_dataprocessing import initializeSCData, getMetadata, loadReconstructedTreeAndData, SCData, \
        nnnReorder, nnnReorderRandom, do_spr_moveset
    from .bonsai_helpers import mp_print, startMPI, getOutputFolder, get_latest_intermediate, \
        clean_up_redundant_data_files, str2bool
    from . import bonsai_globals as bs_glob

    # TODO: Remove eventually, only uncomment this for making animations
    if len(store_all_nwk_folder):
        bs_glob.nwk_counter = 1
        bs_glob.nwk_folder = os.path.join(store_all_nwk_folder, args.dataset)
        Path(bs_glob.nwk_folder).mkdir(parents=True, exist_ok=True)
    else:
        bs_glob.nwk_counter = None
        bs_glob.nwk_folder = None

    # Some of the code can be run in parallel, parallelization is done via mpi4py
    mpiRank, mpiSize = startMPI(args.verbose)
    scData = None
    startGML = None
    ellipsoidSize = args.UB_ellipsoid_size if (args.UB_ellipsoid_size > 0) else None
    origEllipsoidSize = ellipsoidSize

    # The SEQUENTIAL-variables determines whether we first optimise diff. times between merged nodes, and then to ancestor
    # from root, or all three at the same time. SEQUENTIAL=True is faster and leads to better tree likelihoods in tests
    SEQUENTIAL = True
    STORED_GREEDY_RESULT = False

    start_all = time.time()
    # print_memory("Start")

    # Make sure that args.results_folder is set:
    sc_data = SCData(onlyObject=True, dataset=args.dataset, results_folder=args.results_folder)
    args.results_folder = sc_data.result_path()
    del sc_data

    """-----------------Greedily maximizing tree likelihood starting from star-tree------"""
    if args.step in ['preprocess', 'all']:
        outputFolder = getOutputFolder(zscore_cutoff=args.zscore_cutoff, greedy=False,
                                       redo_starry=False, opt_times=False, tmp_file=os.path.basename(args.tmp_folder))
        # If args.pickup_intermediate and preprocessing result is already present, we skip this step.
        preprocessing_done = False
        if args.pickup_intermediate:
            try:
                scData = loadReconstructedTreeAndData(args, outputFolder, reprocess_data=False, all_genes=False,
                                                      get_cell_info=False,
                                                      all_ranks=False, rel_to_results=True)
                mp_print("Since --pickup_intermediate = True, Bonsai successfully loaded the preprocessing result from {}."
                         "\nPlease set --pickup_intermediate False "
                         "for redoing the preprocessing.".format(scData.result_path(outputFolder)))
                preprocessing_done = True
            except Exception as error:
                mp_print("Encountered an error when trying to pick up the preprocessing "
                         "results from a previous run: {}".format(error))
                mp_print("Even though --pickup_intermediate = True, Bonsai could not load the preprocessing result from "
                         "{}. Bonsai will now re-do the "
                         "preprocessing.".format(os.path.join(args.results_folder, outputFolder)),
                         WARNING=True)

        if not preprocessing_done:
            # Read in data and do initial time optimisation for star-tree
            if len(args.tmp_folder):
                if os.path.isdir(args.tmp_folder):
                    # scData = recoverTmpTree(args, args.tmp_folder, optimizeTimes=True)
                    scData = loadReconstructedTreeAndData(args, args.tmp_folder, all_genes=False, get_cell_info=False,
                                                          corrected_data=True)
                else:
                    mp_print("Could not find tmp-folder at {}. Loading tree from start.".format(args.tmp_folder),
                             ERROR=True)
                    scData = initializeSCData(args, createStarTree=True, getOrigData=False, otherRanksMinimalInfo=True)
            else:
                scData = initializeSCData(args, createStarTree=True, getOrigData=False, otherRanksMinimalInfo=True)

            # Store run configurations in YAML-file in output-folder
            args.store_yaml(scData.result_path('used_run_configs.yaml'))

            if mpiRank == 0:
                # Store tree topology with optimised times, and the data only for selected genes, such that it can be read
                # in by multiple cores such that the next part of the program can be run in parallel
                mp_print("Storing result of preprocessing in " + scData.result_path(outputFolder) + "\n\n")
                scData.storeTreeInFolder(scData.result_path(outputFolder), with_coords=True, verbose=args.verbose,
                                         cleanup_tree=False)
        if args.step in ['preprocess']:
            exit()

    # print_memory("After preprocessing.")

    if args.step in ['core_calc', 'all']:
        if scData is None:
            scDataTmp = SCData(onlyObject=True, dataset=args.dataset, results_folder=args.results_folder)
        results_folder = scData.result_path() if (scData is not None) else scDataTmp.result_path()

        if args.pickup_intermediate:
            # If args.pickup_intermediate, we first check which steps already successfully finished in the previous run, by
            # checking the output-folder that is furthest along. We will then pick up the reconstruction there.
            outputFolder = find_latest_tree_folder_new(args, results_folder, not_final=False, set_skip_args=True)
            # outputFolder2 = find_latest_tree_folder(results_folder, not_final=True)

        if args.skip_greedy_merging:
            mp_print("Skipping greedy merging, since --pickup_intermediate is True, and "
                     "the results of this step are already present in {}".format(results_folder))
        else:
            # Determine where to load the results
            outputFolder = getOutputFolder(zscore_cutoff=args.zscore_cutoff, greedy=False,
                                           redo_starry=False, opt_times=False, tmp_file=os.path.basename(args.tmp_folder))
            if scData is None:
                scData = loadReconstructedTreeAndData(args, outputFolder, reprocess_data=False, all_genes=False,
                                                      get_cell_info=False,
                                                      all_ranks=False, rel_to_results=True)

            # TODO: Remove this later. Only uncomment this for printing cell-annotations
            # if len(print_annotations) > 0:
            #     cell_info_df, data_matrices = scData.get_annotations(print_annotations)
            #     cell_id_to_ct = dict(zip(list(cell_info_df.index), list(cell_info_df.iloc[:, 0].values)))
            #     add_celltype_info_to_tree(scData.tree.root, cell_id_to_ct)
            #     bs_glob.geneIds = scData.metadata.geneIds

            if args.verbose:
                mp_print("Starting to greedily merge nodes to maximise tree likelihood.")

            # TODO: Remove eventually, only uncomment this for making animations
            # if (args.step == 'core_calc') and (bs_glob.nwk_folder is not None):
            #     all_nwk_files = [os.path.basename(filepath) for filepath in Path(bs_glob.nwk_folder).glob('*.nwk')]
            #     tree_nums = [int(nwk_file.split('.nwk')[0].split('_')[-1]) for nwk_file in all_nwk_files]
            #     bs_glob.nwk_counter = np.max(tree_nums) + 1

            if mpiRank == 0 and args.verbose:
                mp_print("Loaded tree loglikelihood is:",
                         scData.tree.calcLogLComplete(mem_friendly=True, loglikVarCorr=scData.metadata.loglikVarCorr))
            startGML = time.time()
            outputFolder = getOutputFolder(zscore_cutoff=args.zscore_cutoff, greedy=True,
                                           redo_starry=False, opt_times=False, tmp_file=os.path.basename(args.tmp_folder))
            outputFolder = scData.result_path(outputFolder)
            tmpTreeInd = None
            tmp_folder = os.path.join(outputFolder, 'tmp_trees')
            if (mpiRank == 0) and args.pickup_intermediate and os.path.exists(tmp_folder):
                intermediateFolders = os.listdir(tmp_folder)
                if len(intermediateFolders):
                    intermediateFolder, tmpTreeInd = get_latest_intermediate(intermediateFolders, base='greedy')
                    if intermediateFolder is not None:
                        # scData.tree = unpickleTree(tmp_folder, intermediateFile)
                        scData = loadReconstructedTreeAndData(args, os.path.join(tmp_folder, intermediateFolder),
                                                              reprocess_data=False, all_genes=False, get_cell_info=False,
                                                              all_ranks=False, rel_to_results=False, calc_loglik=True)
            Path(tmp_folder).mkdir(parents=True, exist_ok=True)
            nChildNN = -1 if args.use_knn < 0 else 50
            scData.tree.root.mergeChildrenUB(scData.tree.root.ltqs, scData.tree.root.getW(), scData=scData,
                                             sequential=SEQUENTIAL, verbose=args.verbose,
                                             ellipsoidSize=origEllipsoidSize, outputFolder=outputFolder, nChildNN=nChildNN,
                                             kNN=args.use_knn, mergeDownstream=True, tree=scData.tree,
                                             tmpTreeInd=tmpTreeInd)
            if (scData is not None) and (scData.tree is not None):
                scData.tree.root.clear_memory()
            if mpiRank == 0:
                mp_print("First greedy maximisation of tree likelihood took " + str(time.time() - startGML) + " seconds.")
                scData.metadata.loglik = scData.tree.calcLogLComplete(mem_friendly=True,
                                                                      loglikVarCorr=scData.metadata.loglikVarCorr)
                mp_print("Loglikelihood of tree after first greedy optimisation: " + str(scData.metadata.loglik))
                # Store intermediate results in merge-file.

                # TODO: Remove eventually, only uncomment this for making animations
                # if (mpi_wrapper.get_process_rank() == 0) and bs_glob.nwk_counter:
                #     bs_glob.nwk_counter += 100

                outputFolder = getOutputFolder(zscore_cutoff=args.zscore_cutoff,
                                               greedy=True, redo_starry=False, opt_times=False,
                                               tmp_file=os.path.basename(args.tmp_folder))

                mp_print("Storing result of greedy optimisation in " + scData.result_path(outputFolder) + "\n\n")

                # Store tree topology with optimised times, and the data only for selected genes, such that it can be read
                # in by multiple cores such that the next part of the program can be run in parallel
                # storeCurrentState(outputFolder, scData, dataOrResults='results', filename='tmp_tree.dat', args=args)
                scData.storeTreeInFolder(scData.result_path(outputFolder), with_coords=True, verbose=args.verbose)
                STORED_GREEDY_RESULT = True

        # We can go over the tree once more to see whether some nodes have more than 2 children and are therefore candidates
        # for adding an additional ancestor
        """---------------------Redoing starry nodes."""
        if args.skip_redo_starry:
            mp_print("Skipping redoing starry-nodes (resolving polytomies), since --pickup_intermediate is True, and "
                     "the results of this step are already present in {}".format(results_folder))
        else:
            mpi_wrapper.barrier()
            # Determine where to load results from
            outputFolder = getOutputFolder(zscore_cutoff=args.zscore_cutoff,
                                           redo_starry=False, opt_times=False, tmp_file=os.path.basename(args.tmp_folder))
            # Before doing the nnn_reordering, all processes must have the latest tree topology, but not the data. Before,
            # the tree topology was only necessary on process 0. Therefore, we will load the results from the just stored
            # file on process 0, and send only the tree topology to other processes.
            if (mpiRank != 0) or (not STORED_GREEDY_RESULT):
                scData = loadReconstructedTreeAndData(args, outputFolder, reprocess_data=False, all_genes=False,
                                                      get_cell_info=False, all_ranks=False, rel_to_results=True)

            startRedoingStarry = time.time()
            scData.tree.root.mergeZeroTimeChilds()
            scData.tree.root.renumberNodes(change_node_inds=False)  # Some nodes may have merged, need to renumber them
            scData.tree.nNodes = bs_glob.nNodes

            # Then start merging
            if mpiRank == 0:
                scData.tree.root.getAIRootInfo(None, None)
            nChildNN = -1 if args.use_knn < 0 else 50
            scData.tree.root.mergeChildrenRecursive(scData.tree.root.ltqs, scData.tree.root.getW(),
                                                    sequential=SEQUENTIAL, verbose=args.verbose,
                                                    ellipsoidSize=origEllipsoidSize, nChildNN=nChildNN, kNN=args.use_knn,
                                                    mergeDownstream=True, tree=scData.tree)
            if (scData is not None) and (scData.tree is not None):
                scData.tree.root.clear_memory()
            if mpiRank == 0:
                mp_print("Redoing starry nodes took " + str(time.time() - startRedoingStarry) + " seconds.")
                scData.metadata.loglik = scData.tree.calcLogLComplete(mem_friendly=True,
                                                                      loglikVarCorr=scData.metadata.loglikVarCorr)
                mp_print("Loglikelihood of inferred tree after redoing starry nodes: " + str(scData.metadata.loglik))

            # Store intermediate results.
            outputFolder = getOutputFolder(zscore_cutoff=args.zscore_cutoff,
                                           redo_starry=True, opt_times=False, tmp_file=os.path.basename(args.tmp_folder))
            mp_print("Storing result after redoing starry nodes in " + scData.result_path(outputFolder) + "\n\n")
            scData.storeTreeInFolder(scData.result_path(outputFolder), with_coords=True, verbose=args.verbose)
            # storeCurrentState(outputFolder, scData, dataOrResults='results', args=args)

            # TODO: Remove eventually, only uncomment this for making animations
            # if (mpi_wrapper.get_process_rank() == 0) and bs_glob.nwk_counter:
            #     bs_glob.nwk_counter += 100

        """--------------------Optimise all times on the tree to finalise the tree reconstruction """
        if args.skip_opt_times:
            mp_print("Skipping optimizing times, since --pickup_intermediate is True, and "
                     "the results of this step are already present in {}".format(results_folder))
        else:
            # Time optimization is not parallelized. Only do this on process 0
            if mpiRank == 0:
                mp_print("Starting final optimization of all diffusion times.")
                if scData is None or args.skip_greedy_merging:
                    # Determine where to load results from
                    outputFolder = getOutputFolder(zscore_cutoff=args.zscore_cutoff,
                                                   redo_starry=True, opt_times=False,
                                                   tmp_file=os.path.basename(args.tmp_folder))
                    scData = loadReconstructedTreeAndData(args, outputFolder, reprocess_data=False, all_genes=False,
                                                          get_cell_info=False, all_ranks=False, rel_to_results=True)

                startOptTimes = time.time()
                optTimes = scData.tree.optTimes(verbose=True, singleProcess=True, mem_friendly=True, maxiter=100)

                if (scData is not None) and (scData.tree is not None):
                    scData.tree.root.clear_memory()
                mp_print("Optimization of diffusion times took " + str(time.time() - startOptTimes) + " seconds.")
                scData.metadata.loglik = scData.tree.calcLogLComplete(mem_friendly=True,
                                                                      loglikVarCorr=scData.metadata.loglikVarCorr)
                mp_print("Loglikelihood of inferred tree after optimising diffusion times: " + str(scData.metadata.loglik))

                # self.metadata.geneVariances = self.tree.optGeneVars(scData.metadata.geneVariances, verbose=True, singleProcess=True, genesSimultaneously=True)
                # self.metadata.loglikVarCorr = - self.metadata.nCells * np.sum(np.log(self.metadata.geneVariances))

                # Store intermediate results in merge-file.
                outputFolder = getOutputFolder(zscore_cutoff=args.zscore_cutoff,
                                               redo_starry=True, opt_times=True, tmp_file=os.path.basename(args.tmp_folder))
                mp_print("Storing result after optimising diffusion times in " + scData.result_path(outputFolder) + "\n")
                scData.storeTreeInFolder(scData.result_path(outputFolder), with_coords=True, verbose=args.verbose)
                # Store tree topology with optimised times, and the data only for selected genes, such that it can be read in
                # by multiple cores such that the next part of the program can be run in parallel
                # storeCurrentState(outputFolder, scData, dataOrResults='results', args=args)

                # TODO: Remove eventually, only uncomment this for making animations
                # if (mpiRank == 0) and bs_glob.nwk_counter and (scData.tree is not None):
                #     scData.tree.to_newick(use_ids=True,
                #                           results_path=os.path.join(bs_glob.nwk_folder,
                #                                                     'tree_{}.nwk'.format(bs_glob.nwk_counter)))
                #     bs_glob.nwk_counter += 1
                # if (mpi_wrapper.get_process_rank() == 0) and bs_glob.nwk_counter:
                #     bs_glob.nwk_counter += 100

        """--------------------Optimise all times on the tree to finalise the tree reconstruction """
        if args.skip_spr_moves:
            mp_print("Skipping the SPR moves, since --pickup_intermediate is True, and "
                     "the results of this step are already present in {}".format(results_folder))
        else:
            mp_print("Starting SPR moves.")
            start_spr = time.time()
            if mpiRank == 0:
                if scData is None or args.skip_greedy_merging:
                    # Determine where to load results from
                    outputFolder = getOutputFolder(zscore_cutoff=args.zscore_cutoff,
                                                   redo_starry=True, opt_times=True,
                                                   tmp_file=os.path.basename(args.tmp_folder))
                    scData = loadReconstructedTreeAndData(args, outputFolder, reprocess_data=False, all_genes=False,
                                                          get_cell_info=False, all_ranks=False, rel_to_results=True,
                                                          verbose=True)

                rootsetting_success = scData.tree.set_mindist_root(cell_ids=scData.metadata.cellIds)
                scData.metadata.loglik = scData.tree.calcLogLComplete(mem_friendly=True,
                                                                      loglikVarCorr=scData.metadata.loglikVarCorr)
                mp_print("Loglikelihood of inferred tree before starting SPR moves: " + str(scData.metadata.loglik))

            # For memory usage reasons, it's best to here delete scData, then read it in using memmap in the SPR-moves
            # function
            scdata_path = store_scdata_and_communicate_path(scData, specific_results_folder='spr_intermediates',
                                                            general_results_folder=args.results_folder)
            scData = None
            gc.collect()
            # print_memory("Ready to start SPR moves")

            # Do the actual moves here:
            scData = do_spr_moveset(scdata_path, args, strategy=args.spr_strategy,
                                    pickup_intermediate=args.pickup_intermediate)
            # print_memory("Done with SPR moves")
            if (scData is not None) and (scData.tree is not None):
                scData.tree.root.clear_memory()

            if mpiRank == 0:
                scData.tree.remove_two_child_root()

                mp_print("SPR moves took " + str(time.time() - start_spr) + " seconds.")
                scData.metadata.loglik = scData.tree.calcLogLComplete(mem_friendly=True,
                                                                      loglikVarCorr=scData.metadata.loglikVarCorr)
                mp_print("Loglikelihood of inferred tree after SPR moves: " + str(scData.metadata.loglik))

                # Store intermediate results in merge-file.
                outputFolder = getOutputFolder(zscore_cutoff=args.zscore_cutoff, spr_moves=True,
                                               redo_starry=True, opt_times=True, tmp_file=os.path.basename(args.tmp_folder))
                mp_print("Storing result after SPR moves in " + scData.result_path(outputFolder) + "\n")
                scData.storeTreeInFolder(scData.result_path(outputFolder), with_coords=True, verbose=args.verbose)

                redundant_folder = scData.result_path('spr_intermediates')
                if os.path.exists(redundant_folder):
                    remove_tree_folders(redundant_folder, removeDir=True)

                # Store tree topology with optimised times, and the data only for selected genes, such that it can be read
                # by multiple cores such that the next part of the program can be run in parallel
                # storeCurrentState(outputFolder, scData, dataOrResults='results', args=args)

            # TODO: Remove eventually, only uncomment this for making animations
            # if (mpiRank == 0) and bs_glob.nwk_counter and (scData.tree is not None):
            #     scData.tree.to_newick(use_ids=True,
            #                           results_path=os.path.join(bs_glob.nwk_folder,
            #                                                     'tree_{}.nwk'.format(bs_glob.nwk_counter)))
            #     bs_glob.nwk_counter += 1
            # if (mpi_wrapper.get_process_rank() == 0) and bs_glob.nwk_counter:
            #     bs_glob.nwk_counter += 100

        """--------------------Do random re-ordering of next-nearest-neighbour-nodes """
        if args.skip_nnn_reordering:
            mp_print("Skipping Nearest-Neighbor interchanges, since --pickup_intermediate is True, and "
                     "the results of this step are already present in {}".format(results_folder))
        else:
            # Before doing the nnn_reordering, all processes must have the latest tree, which was before only necessary on
            # process 0. Therefore, we will just load the results from the just stored file here.

            # Barrier to make sure that also process 0 is done with storing the results from optTimes
            mpi_wrapper.barrier()
            # Determine where to load results from
            outputFolder = getOutputFolder(zscore_cutoff=args.zscore_cutoff, spr_moves=True,
                                           redo_starry=True, opt_times=True, tmp_file=os.path.basename(args.tmp_folder))

            mp_print("Starting re-ordering of next-nearest-neighbour-nodes.")
            startNnnReorder = time.time()
            np.random.seed(42)

            scData = SCData(onlyObject=True, dataset=args.dataset, results_folder=args.results_folder)
            # mp_print("Before starting nnnReorderRandom, memory usage is ",
            #          psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ALL_RANKS=True)
            stored_tree_ind, tmp_folder = nnnReorderRandom(args, outputFolder, verbose=args.verbose,
                                                           randomMoves=args.nnn_n_randommoves,
                                                           randomTries=args.nnn_n_randomtrees,
                                                           resultsFolder=scData.result_path())

            # TODO: Remove eventually, only uncomment this for making animations
            # if (mpiRank == 0) and bs_glob.nwk_counter and (scData.tree is not None):
            #     scData.tree.to_newick(use_ids=True,
            #                           results_path=os.path.join(bs_glob.nwk_folder,
            #                                                     'tree_{}.nwk'.format(bs_glob.nwk_counter)))
            #     bs_glob.nwk_counter += 1
            # if (mpi_wrapper.get_process_rank() == 0) and bs_glob.nwk_counter:
            #     bs_glob.nwk_counter += 100

            # mp_print("Before starting nnnReorder, memory usage is ",
            #          psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ALL_RANKS=True)
            scData = nnnReorder(args, tmp_folder, stored_tree_ind, maxMoves=1000, closenessBound=0.5, verbose=args.verbose)
            if (scData is not None) and (scData.tree is not None):
                scData.tree.root.clear_memory()

            if mpiRank != 0:
                mp_print("This process's job is done. Closing down.")
                exit()
            else:
                bs_glob.nNodes = scData.tree.nNodes

            mpiSize = 1
            mp_print("Reordering next-to-nearest neighbours took " + str(time.time() - startNnnReorder) + " seconds.")
            scData.metadata.loglik = scData.tree.calcLogLComplete(mem_friendly=True,
                                                                  loglikVarCorr=scData.metadata.loglikVarCorr)
            mp_print(
                "Loglikelihood of inferred tree after next-to-nearest-neighbour reordering: " + str(scData.metadata.loglik))

            startOptTimes = time.time()
            mp_print("Starting a final optimization of diffusion times.")
            optTimes = scData.tree.optTimes(verbose=True, mem_friendly=True, maxiter=100, singleProcess=True)
            new_opt_times = scData.tree.test_for_zero_times(verbose=True, mem_friendly=True, singleProcess=True)
            mp_print("Optimization of diffusion times took " + str(time.time() - startOptTimes) + " seconds.")
            scData.metadata.loglik = scData.tree.calcLogLComplete(mem_friendly=True,
                                                                  loglikVarCorr=scData.metadata.loglikVarCorr)
            if (scData is not None) and (scData.tree is not None):
                scData.tree.root.clear_memory()

            mp_print("Loglikelihood of inferred tree after optimising diffusion times: " + str(scData.metadata.loglik))

            # Store intermediate results
            outputFolder = getOutputFolder(zscore_cutoff=args.zscore_cutoff, spr_moves=True,
                                           redo_starry=True, opt_times=True, nnn_reorder=True,
                                           tmp_file=os.path.basename(args.tmp_folder))
            mp_print(
                "Storing result after next-to-nearest neighbour reorder in " + scData.result_path(outputFolder) + "\n\n")
            scData.storeTreeInFolder(scData.result_path(outputFolder), with_coords=True, verbose=args.verbose)
            # storeCurrentState(outputFolder, scData, dataOrResults='results', args=args)

            # TODO: Remove eventually, only uncomment this for making animations
            # if (mpiRank == 0) and bs_glob.nwk_counter and (scData.tree is not None):
            #     scData.tree.to_newick(use_ids=True,
            #                           results_path=os.path.join(bs_glob.nwk_folder,
            #                                                     'tree_{}.nwk'.format(bs_glob.nwk_counter)))
            #     bs_glob.nwk_counter += 1

        if (hasattr(args, 'skip_reorder_edges')) and args.skip_reorder_edges:
            mp_print("Skipping reordering branches and resetting the root, since --pickup_intermediate is True, and "
                     "the results of this step are already present in {}".format(results_folder))
        else:
            """----------------Swap children order to minimize cousin distance to improve visual apperance.--------------"""
            # The tree likelihood is independent of swapping the left-right order of children of the same node, so everyone is
            # free to choose their own order. We here try to minimize distances between 'cousins', but ladderizing the tree is
            # another option.
            if mpiRank == 0:
                mp_print("Starting setting midpoint root and reordering (left-right order) of children of all nodes.")
                if scData is None:
                    # Determine where to load results from
                    outputFolder = getOutputFolder(zscore_cutoff=args.zscore_cutoff, spr_moves=True,
                                                   redo_starry=True, opt_times=True, nnn_reorder=True, reorderedEdges=False,
                                                   tmp_file=os.path.basename(args.tmp_folder))
                    scData = loadReconstructedTreeAndData(args, outputFolder, reprocess_data=False, all_genes=False,
                                                          get_cell_info=False, all_ranks=False, rel_to_results=True)

                start_setting_root = time.time()
                rootsetting_success = scData.tree.set_mindist_root(cell_ids=scData.metadata.cellIds)
                if not rootsetting_success:
                    mp_print("Setting minimal-distance root didn't succeed, setting midpoint root instead.")
                    scData.tree.set_midpoint_root()

                mp_print("Setting root took " + str(time.time() - start_setting_root) + " seconds.")
                # scData.metadata.loglik = scData.tree.calcLogLComplete(mem_friendly=True,
                #                                                       loglikVarCorr=scData.metadata.loglikVarCorr)
                # mp_print("Loglikelihood of inferred tree after reordering children: " + str(scData.metadata.loglik))

                startReorderEdges = time.time()
                nChildren = scData.tree.root.gatherInfoDepthFirst([])
                scData.tree.root.deleteParentsWithOneChild()
                scData.tree.root.mergeZeroTimeChilds()
                scData.tree.root.renumberNodes(change_node_inds=False)
                scData.tree.nNodes = bs_glob.nNodes
                # scData.tree.root.reorderChildrenRoot(verbose=args.verbose, maxChild=8)
                scData.tree.root.ladderize_in_main()
                if (scData is not None) and (scData.tree is not None):
                    scData.tree.root.clear_memory()

                mp_print("Reordering children took " + str(time.time() - startReorderEdges) + " seconds.")
                scData.metadata.loglik = scData.tree.calcLogLComplete(mem_friendly=True,
                                                                      loglikVarCorr=scData.metadata.loglikVarCorr)
                mp_print("Loglikelihood of inferred tree after reordering children: " + str(scData.metadata.loglik))

                # Store intermediate results
                outputFolder = getOutputFolder(zscore_cutoff=args.zscore_cutoff, spr_moves=True,
                                               redo_starry=True, opt_times=True, nnn_reorder=True, reorderedEdges=True,
                                               tmp_file=os.path.basename(args.tmp_folder))
                mp_print("Storing result after reordering children in " + scData.result_path(outputFolder) + "\n\n")
                scData.storeTreeInFolder(scData.result_path(outputFolder), with_coords=True, verbose=args.verbose)
                # storeCurrentState(outputFolder, scData, dataOrResults='results', args=args)

        if startGML is not None:
            computationTime = time.time() - startGML
            mp_print("Calculation took %f seconds." % computationTime)
        else:
            computationTime = np.nan

    """--------------Storing final tree and storing some metadata-----------"""
    if args.step in ['metadata', 'all']:
        if args.step == 'metadata':
            computationTime = np.nan
        # outputFolder = getOutputFolder(zscore_cutoff=args.zscore_cutoff,
        #                                redo_starry=(not args.skip_redo_starry), opt_times=(not args.skip_opt_times),
        #                                reorderedEdges=True, tmp_file=os.path.basename(args.tmp_folder),
        #                                nnn_reorder=(not args.skip_nnn_reordering))
        if scData is None:
            scDataTmp = SCData(onlyObject=True, dataset=args.dataset, results_folder=args.results_folder)
        results_folder = scData.result_path() if (scData is not None) else scDataTmp.result_path()
        if args.pickup_intermediate:
            dir_path = getOutputFolder(zscore_cutoff=args.zscore_cutoff, tmp_file=os.path.basename(args.tmp_folder),
                                       final=True)
            if os.path.exists(os.path.join(results_folder, dir_path, 'vertInfo.txt')):
                mp_print("\nYou set --pickup_intermediate True, but the final results folder is already present."
                         "\nTree reconstruction thus seems to be finished.\n"
                         "\nIf this is not the desired behaviour, please either"
                         "\n - set --pickup_intermediate False, which will start "
                         "the tree reconstruction from scratch, or "
                         "\n - delete the results-folders of the steps that you want to and keep "
                         "--pickup_intermediate True.", WARNING=True)
                exit()

        if scData is None:
            # if type(args) is dict:
            #     args = convert_dict_to_named_tuple(args)
            scDataTmp = SCData(onlyObject=True, dataset=args.dataset, results_folder=args.results_folder)
            results_folder = scDataTmp.result_path()
        else:
            results_folder = scData.result_path()
        outputFolder = find_latest_tree_folder_new(args, results_folder, not_final=True)

        if scData is None:
            all_genes = False
            scData = loadReconstructedTreeAndData(args, outputFolder, all_genes=all_genes, get_cell_info=False,
                                                  reprocess_data=True, all_ranks=True, rel_to_results=True,
                                                  no_data_needed=False, get_posterior_ltqs=True, otherRanksMinimalInfo=True,
                                                  calc_posteriors=False)

        # scDataUncorrected = loadReconstructedTreeAndData(args, outputFolder, reprocess_data=True, all_genes=False,
        #                                                  all_ranks=False, get_cell_info=False, corrected_data=False,
        #                                                  rel_to_results=True, no_data_needed=False, single_process=False,
        #                                                  keep_original_data=False, calc_loglik=False, get_data=True)

        if mpiRank == 0:
            finalFolder = getOutputFolder(zscore_cutoff=args.zscore_cutoff, tmp_file=os.path.basename(args.tmp_folder),
                                          final=True)
            scData.storeTreeInFolder(scData.result_path(finalFolder), with_coords=True, verbose=args.verbose,
                                     store_posterior_ltqs=True)
            mp_print("\n\nStored final tree in " + scData.result_path(finalFolder) + ".\n\n")

            metadata = getMetadata(args, scData, outputFolder, computationTime)
            metadata.to_csv(os.path.join(scData.result_path(finalFolder), 'metadata_bonsai.txt'))
            mp_print("Stored metadata in {}".format(os.path.join(scData.result_path(finalFolder), 'metadata_bonsai.txt')))

            # Finally clean up some redundant data-files:
            clean_up_redundant_data_files(scData, args, verbose=args.verbose)
            redundant_folders = ['random_trees', 'intermediate_trees']
            for redundant_folder in redundant_folders:
                redundant_folder = scData.result_path(redundant_folder)
                if os.path.exists(redundant_folder):
                    remove_tree_folders(redundant_folder, removeDir=True)

    mp_print("Time necessary for the whole calculation was {} seconds.".format(time.time() - start_all))
    # print_memory("Memory usage after the whole calculation")



def _build_parser():
    """Upstream's ArgumentParser, lifted out of ``__main__`` so omicverse can
    reuse it to build an args namespace without spawning a process."""
    parser = ArgumentParser(
        description='Infers a cell-tree to approximate the distances in gene expression space between cells in single'
                    ' cell data.')
    parser.add_argument('--config_filepath', type=str, default=None,
                        help='Absolute (or relative to "bonsai-development") path to YAML-file that contains all arguments'
                             'needed to run Bonsai.')
    parser.add_argument('--step', type=str, default='all',
                        help='Optional: Use this argument to run all or only a single step of the script, continuing from '
                             'the state of the previous step. Can be used when parallelizing on HPC-clusters, because '
                             'mostly core_calc benefits from parallelization, and preprocess has the highest memory '
                             'requirements.'
                             'Allowed values (in order of execution): all, preprocess, core_calc, metadata.')
    parser.add_argument('--pickup_intermediate', type=str2bool, default=False,
                        help='Optional: Set this argument to true if Bonsai needs to search in the indicated results-folder'
                             'for the furthest developed tree reconstruction, and pick it up from there. If this argument'
                             'is True, it over-rules the same argument in bonsai_config.yaml, and the other way around.')
    # TODO: To be removed
    parser.add_argument('--store_all_nwk_folder', type=str, default='',
                        help='REMOVE LATER! This will slow down the program by storing a newick file after every change to'
                             'the tree. This is only necessary for making a tree reconstruction animation.')
    parser.add_argument('--print_annotations', type=str, default='',
                        help='REMOVE LATER! Only for debugging, string given here should give path to annotation-folder.')
    #
    # # Arguments that define where to find the data and where to store results. Also, whether data comes from Sanity or not
    # parser.add_argument('--dataset', type=str, default='Zeisel',
    #                     help='Name of dataset. This will determine name of results-folder that is created.')
    # parser.add_argument('--data_folder', type=str, default='data',
    #                     help='path to folder where input data can be found. This folder should contain a file with means'
    #                          'and standard-deviations in files "delta.txt" and "d_delta.txt" unless argument '
    #                          'filenames_data changes this behaviour.')
    # parser.add_argument('--results_folder', type=str, default=None,
    #                     help='path to folder where results will be stored.')
    # parser.add_argument('--filenames_data', type=str, default='delta.txt,d_delta.txt',
    #                     help='Filenames of input-files for means and standard deviations separated by a comma. '
    #                          'These files should have different cells in the columns, and '
    #                          'features (like gene expression quotients) as rows.')
    # parser.add_argument('--input_is_sanity_output', type=str2bool, default=True,
    #                     help='Whether we divide out Sanitys prior, to get likelihood of data given LTQs, instead of vice '
    #                          'versa. This should be turned off when tree reconstruction is used on data that is not the'
    #                          'output of Sanity.')
    # parser.add_argument('--tmp_folder', type=str, default='',
    #                     help='Foldername pointing towards tmp-file (absolute path). '
    #                          'Relevant if one wants to start from an intermediate tree,'
    #                          ' either for debugging reasons, or because the previous run did not finalize. This tmp-tree is'
    #                          ' automatically stored as a file in the results-folder during a normal run.')
    # parser.add_argument('--pickup_intermediate', type=str2bool, default=False,
    #                     help='Decides whether we look for intermediate results from previous runs or not.')
    # # parser.add_argument('--manual_merge', type=str, default='',
    # #                     help='Filename pointing towards merge-file (relative to results-folder). '
    # #                          'Relevant if one wants to quickly repeat the merge steps of an earlier run of the algorithm.')
    #
    # # Arguments that determine running configurations of bonsai. How much is printed, which steps are run?
    # parser.add_argument('--verbose', type=str2bool, default=True,
    #                     help='--verbose False only shows essential print messages (default: True)')
    # parser.add_argument('--skip_greedy_merging', type=str2bool, default=False,
    #                     help='Used to skip tree reconstruction when this is already done and stored')
    #
    # # Arguments that decide on how many genes are kept for the inference
    # parser.add_argument('--zscore_cutoff', type=float, default=-1.,
    #                     help='Genes with a variability under this cutoff will be dropped. Negative means: keep all.'
    #                          'zscore_cutoff.')
    #
    # # Arguments that decide how much post-optimisation is done
    # parser.add_argument("--skip_opt_times", type=str2bool, default=False,
    #                     help="Decides whether all times are optimised after tree reconstruction.")
    # parser.add_argument("--skip_redo_starry", type=str2bool, default=False,
    #                     help="Decides whether, after the first greedy merging, for nodes with more than 2 children, "
    #                          "pairs of children are considered for merge.")
    # parser.add_argument("--use_knn", type=int, default=20,
    #                     help="Decides whether nearest-neighbours are used to get candidate pairs to merge. Set to -1 for"
    #                          "considering all pairs of leafs.")
    # parser.add_argument("--skip_nnn_reordering", type=str2bool, default=False,
    #                     help="Decides whether we go over edges and try to reconfigure all connected nodes (which are thus"
    #                          "next-nearest-neighbours).")
    # parser.add_argument("--nnn_n_randommoves", type=int, default=1000,
    #                     help="Decides how many random nnn-moves we do before finally doing them greedily.")
    # parser.add_argument("--nnn_n_randomtrees", type=int, default=1,
    #                     help="Decides how many random trees we create before taking the best loglikelihood and doing nnn "
    #                          "greedily.")
    #
    # # Arguments determining which computational speedups are done
    # parser.add_argument("--UB_ellipsoid_size", type=float, default=-1,
    #                     help="Decides whether we make an estimate of an upper bound (UB) for dLogL based on the current"
    #                          "calculation, such that some calculations in the future may be skipped. If this arg < 0, this"
    #                          "estimation will not be used. Otherwise, the argument decides how large the ellipsoid is in "
    #                          "root-position/precision space for which we estimate the UB. Larger f will result in looser "
    #                          "UB, so that more candidate pairs per merge have to be considered. However, it will also "
    #                          "result in longer validity of the UB-estimation, so that more merges can be done without"
    #                          "re-calculating the upper bounds. This is a computational optimization that should not affect "
    #                          "the final result. Ellipsoid sizes below 5 are reasonable to try, I recommend to start at 2.")
    #
    # parser.add_argument("--rescale_by_var", type=str2bool, default=True,
    #                     help="By default coordinates are rescaled by the inferred variance per gene (feature). This"
    #                          "implicitly assumes that changes in highly-varying genes are more likely")
    return parser


if __name__ == "__main__":
    main(_build_parser().parse_args())
