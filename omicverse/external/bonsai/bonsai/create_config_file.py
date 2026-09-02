from argparse import ArgumentParser, ArgumentTypeError
from ruamel.yaml import YAML
import os, sys
from pathlib import Path

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')




def main(args):
    """Write the run YAML. Wrapped by omicverse; see the package docstring."""

    # Warn about deprecated arguments
    deprecated_args = ['skip_greedy_merging', 'skip_opt_times', 'skip_redo_starry', 'skip_nnn_reordering']
    for dep_arg in deprecated_args:
        if hasattr(args, dep_arg) and getattr(args, dep_arg):
            print("WARNING: The arguments {} are deprecated and no longer have any effect. Use --pickup_intermediate True"
                  " to pick up intermediate results from previous Bonsai runs.")

    # omicverse: was a path relative to the repo root, which only resolved
    # because upstream chdir'd there at import. Anchor it to this file instead,
    # so it works from any working directory.
    template_yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'config_template_do_not_change', 'config_template.yaml')
    if not os.path.exists(template_yaml_path):
        print("Can't find template for YAML-file, should be at: {}".format(template_yaml_path))
        exit()

    yaml = YAML()
    with open(template_yaml_path, 'r') as file_obj:
        yaml_file = yaml.load(file_obj)

    yaml_dict = dict(yaml_file)
    for label in yaml_dict:
        try:
            args_attrib = getattr(args, label)
            if args_attrib is not None:
                yaml_file[label] = getattr(args, label)
        except AttributeError:
            pass


    print("Storing new config-file at {}".format(args.new_yaml_path))
    Path(os.path.dirname(args.new_yaml_path)).mkdir(parents=True, exist_ok=True)
    with open(args.new_yaml_path, 'w') as file_obj:
        yaml.dump(yaml_file, file_obj)



def _build_parser():
    """Upstream's ArgumentParser, lifted out of ``__main__`` so omicverse can
    reuse it to build an args namespace without spawning a process."""
    parser = ArgumentParser(
        description='Infers a binary tree to approximate the distances in gene expression space between cells in single'
                    ' cell data.')
    parser.add_argument('--new_yaml_path', type=str, default='my_config.yaml',
                        help='Path (absolute or relative to "bonsai-development) to where the created YAML-file with '
                             'configuration-parameters needs to be stored.')
    # Arguments that define where to find the data and where to store results. Also, whether data comes from Sanity or not
    parser.add_argument('--dataset', type=str, default='new_dataset',
                        help='Identifier of dataset.')
    parser.add_argument('--data_folder', type=str, default='data/new_dataset',
                        help='Path (absolute or relative to "bonsai-development") to folder where input data can be found. '
                             'This folder should contain a file with means and standard-deviations in files "delta.txt" '
                             'and "d_delta.txt" unless filenames_data changes this behaviour.')
    parser.add_argument('--results_folder', type=str, default=None,
                        help='# Path (absolute or relative to "bonsai-development") to folder where results will be '
                             'stored.')
    parser.add_argument('--filenames_data', type=str, default='features.txt,standard_deviations.txt',
                        help='Filenames of input-files for means and standard deviations separated by a comma. These files '
                             'should have different cells as columns, and features (such as log transcription quotients) '
                             'as rows.')
    parser.add_argument('--input_is_sanity_output', type=str2bool, default=True,
                        help='The provided means and stds by Sanity are the means and stds of the posterior for the '
                             'feature: P(x | D), but in Bonsai we need the likelihood of the data given the feature value: '
                             'P(D | x). These are related through P(x | D) ~ P(D | x) P(x). So, we need to divide '
                             'out the prior. If the input is Sanity-output we know how to do this because the reported '
                             'posterior is Gaussian and the prior was Gaussian around zero (see the SI of the '
                             'Bonsai-publication. '
                             'If your input-data is not Sanity-output, set this flag to False, but make sure to provide '
                             'Bonsai with the required means and variances of the likelihood (P(D | x)).')
    # Arguments that determine running configurations of bonsai. How much is printed, which steps are run?
    parser.add_argument('--verbose', type=str2bool, default=True,
                        help='False only shows essential print messages')
    # Arguments that decide on how data is preprocessed
    parser.add_argument('--zscore_cutoff', type=float, default=1.0,
                        help='# Genes with a signal-to-noise ratio under this cutoff will be dropped. '
                             'Default=1.0, Negative means: keep all. '
                             'Due to the large number of genes dominated by noise, tree-reconstruction is usually better '
                             'when we discard the most noisy genes, even though we account for error-bars rigorously. '
                             'zscore is defined as: '
                             'z_g = frac{1}{C} sum_c frac{(delta_{gc} - bar{delta_g})^2}{epsilon_{gc}^2} '
                             'where delta_{gc} is the feature value for gene g and cell c, '
                             'epsilon_{gc} is the standard-deviation')
    parser.add_argument("--rescale_by_var", type=str2bool, default=True,
                        help="This determines whether oordinates are rescaled by the inferred variance per gene (feature). "
                             "This is equivalent to putting the prior assumption that it is more likely to see change in a "
                             "certain gene's expression when the gene shows much variation over the whole dataset.")
    # Arguments determining which computational speedups are done
    parser.add_argument("--use_knn", type=int, default=10,
                        help="Decides whether nearest-neighbours are used to get candidate pairs to merge. Set to -1 for "
                             "consideringall pairs of leafs. Values between 5 and 20 give good results. Computation time "
                             "will scale approximately linear with use_knn, but tree likelihood may also increase "
                             "slightly with this parameter.")
    parser.add_argument("--nnn_n_randommoves", type=int, default=1000,
                        help="Decides how many random nearest-neighbor-interchange-moves we do before doing them greedily.")
    parser.add_argument("--nnn_n_randomtrees", type=int, default=10,
                        help="Decides how many random trees we create before taking the tree with the highest "
                             "loglikelihood and doing nnn greedily. Since the creation of one random tree is not "
                             "parallelized, it never hurts to set nnn_n_randomtrees equal to the number of cores that are "
                             "anyhow reserved, since then the different random trees will be created in parallel.")
    parser.add_argument("--UB_ellipsoid_size", type=float, default=1.0,
                        help="This is a purely computational optimization that does not affect the final result. Decides "
                             "whether we calculate an upper bound (UB) for the increase in loglikelihood that merging a "
                             "certain pair can yield, given that a the root stays in a certain ellipsoid. Using these "
                             "upper bounds, the calculation can be sped up enormously. If the UB_ellipsoid_size < 0, "
                             "this estimation will not be used. Otherwise, it decides how large the ellipsoid is in "
                             "root-position/precision space for which we estimate the UB. Larger values will result in "
                             "looser UB, so that more candidate pairs per merge have to be considered. However, it will "
                             "also result in longer validity of the UB-estimation, so that more merges can be done "
                             "without re-calculating the upper bounds. Ellipsoid sizes below 5 are reasonable to try, I "
                             "recommend to start at 1. Ellipsoid size will be updated dynamically by Bonsai based on the "
                             "results.")
    parser.add_argument('--spr_strategy', type=str, default='large_tree',
                        help="Move to general config-file later. This will determine what strategy we follow for the "
                             "spr-moves. Current options are 'large_tree', 'long_branch_sorted', 'super_sure'."
                             "NOTE: Only large_tree allows for parallelization, the rest will only use 1 CPU.")
    """Information on starting from previous runs configurations"""
    # If Bonsai was already ran with these configurations, but failed (e.g. because of a time limit), it can be re-started
    # from the last completed results by setting --pickup_intermediate True.
    # In that case, we will look for the furthest advanced result in the results-folder. Note that the results-folder thus
    # has to be cleaned of results that you don't want to be picked up.
    # The main Bonsai calculation takes the 5 below steps. After each step, the current tree will be stored in the results-
    # folder. Therefore, if Bonsai fails (or runs out of time) in some step, it can be picked up from the end-result of the
    # previous step.
    #  - greedy_merging: greedily merging pairs starting from the star-tree
    #  - redo_starry: detecting nodes that still have more than 2 children, and checking if more merges can be done
    #  - opt_times: optimizing all branch length simultaneously once
    #  - nnn_reordering: taking any edge and interchanging children of the two connected nodes
    #  - reorder_edges: swapping the order of branches to ladderize the tree and re-setting the root
    parser.add_argument('--pickup_intermediate', type=str2bool, default=False,
                        help='Decides whether we look for intermediate results from previous runs or not. '
                             'These intermediate results are periodically stored during any normal run, and can thus be '
                             'used when a run did not finalize.')
    # The following arguments are DEPRECATED! They won't do anything anymore.
    # Arguments that decide how much post-optimisation is done
    parser.add_argument('--skip_greedy_merging', type=str2bool, default=False,
                        help='[DEPRECATED!] Used to skip tree reconstruction when this is already done and stored')
    parser.add_argument("--skip_opt_times", type=str2bool, default=False,
                        help="[DEPRECATED!] Decides whether all times are optimised after tree reconstruction.")
    parser.add_argument("--skip_redo_starry", type=str2bool, default=False,
                        help="[DEPRECATED!] Decides whether, after the first greedy merging, for nodes with more than 2 children, "
                             "pairs of children are considered for merge.")
    parser.add_argument("--skip_nnn_reordering", type=str2bool, default=False,
                        help="[DEPRECATED!] Decides whether we go over edges and try to reconfigure all connected nodes (which are thus"
                             "next-nearest-neighbours).")
    parser.add_argument('--tmp_folder', type=str, default='',
                        help='(ADVANCED): Path (absolute path or relative to "bonsai-development") pointing to a '
                             'tree-folder from which Bonsai reconstruction will start. Relevant if one wants to start '
                             'from a tree other than the usual star-tree, for example if one already created a '
                             'tree-object where cellstates were connected to a common ancestor.')
    return parser


if __name__ == "__main__":
    main(_build_parser().parse_args())
