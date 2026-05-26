from typing import Dict, List, Optional
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from omicverse.external.scTenifold.data._sim import *


def adobo_score(X: pd.DataFrame,
                genes: List[str],
                n_bins: int = 25,
                n_ctrl: int = 50,
                random_state: int = 42,
                file_path: Optional[Path] = None) -> pd.Series:
    """Adobo-style signature score: mean target expression minus binned controls.

    Parameters
    ----------
    X
        Expression DataFrame (genes x cells).
    genes
        Target signature gene names.
    n_bins
        Number of mean-expression bins used to draw controls.
    n_ctrl
        Number of control genes sampled per target.
    random_state
        Seed for the control sampler.
    file_path
        Optional CSV output destination.

    Returns
    -------
    Per-cell signature scores.
    """
    if len(genes) == 0:
        raise ValueError('Gene list ("genes") is empty.')
    gene_mean = X.mean(axis=1)
    gene_mean = gene_mean.sort_values()
    binned = pd.qcut(gene_mean, n_bins)
    ret = []
    for g in genes:
        sampled_bin = binned[binned == binned[binned.index == g].values[0]]
        if n_ctrl > sampled_bin.shape[0]:
            ret.append(sampled_bin.index)
        else:
            ret.append(
                sampled_bin.sample(n_ctrl, replace=True, random_state=random_state).index
            )
    con = []
    for g in ret:
        con.append(X[X.index.isin(g)].mean(axis=0))

    con = pd.concat(con, axis=1).transpose()
    con.index = genes
    targets = X[X.index.isin(genes)]
    targets = targets.reindex(genes)
    scores = (targets-con).mean(axis=0)
    if file_path:
        scores.to_csv(file_path)
    return scores


def _get_assigned_bins(data_avg: np.ndarray,
                       cluster_len: int,
                       n_bins: int) -> np.ndarray:
    assigned_bin = np.zeros(shape=(cluster_len, ), dtype=np.int32)  # (G,)
    bin_size = cluster_len / n_bins
    for i_bin in range(n_bins):
        assigned_bin[(assigned_bin == 0) &
                     (data_avg <= data_avg[int(np.round(bin_size * i_bin))])] = i_bin
    return assigned_bin


def _get_ctrl_use(assigned_bin: np.ndarray,
                  gene_arr: np.ndarray,
                  target_dict: Dict[str, List[str]],
                  n_ctrl: int,
                  random_state: np.random.Generator) -> List[str]:
    selected_bins = list(set(assigned_bin[np.isin(gene_arr, target_dict["Pos"])]))
    genes_in_same_bin = gene_arr[np.isin(assigned_bin, selected_bins)]
    ctrl_use = list()
    for _ in range(len(target_dict["Pos"])):
        ctrl_use.extend(random_state.choice(genes_in_same_bin, n_ctrl))
    return list(set(ctrl_use))


def cell_cycle_score(X: np.ndarray,
                     gene_list: List[str],
                     sample_list: List[str],
                     target_dict: Optional[Dict[str, List[str]]] = None,
                     n_bins: int = 25,
                     n_ctrl: int = 50,
                     random_state: int = 42,
                     file_path: Optional[Path] = None) -> np.ndarray:
    """Score cell cycle (or arbitrary signature) per cell, Seurat-style.

    Parameters
    ----------
    X
        Expression matrix (genes x cells).
    gene_list
        Gene names aligned with rows of ``X``.
    sample_list
        Sample names aligned with columns of ``X`` (used for CSV output).
    target_dict
        ``{"Pos": [...], "Neg": [...]}`` signature genes; defaults to
        :data:`scTenifold.data._sim.DEFAULT_POS`/``DEFAULT_NEG``.
    n_bins
        Mean-expression bin count for control selection.
    n_ctrl
        Number of control genes per target.
    random_state
        Seed for the RNG.
    file_path
        Optional CSV destination.

    Returns
    -------
    Per-cell signature scores.
    """
    random_state = np.random.default_rng(random_state)
    if target_dict is None:
        target_dict = {"Pos": DEFAULT_POS,
                       "Neg": DEFAULT_NEG}
    else:
        target_dict = {k: [i.upper() for i in v] for k, v in target_dict.items()}

    if len(set(gene_list) & set(target_dict["Pos"])) == 0:
        raise ValueError('No feature genes found in gene_list.')

    gene_list = [i.upper() for i in gene_list]
    cluster_len = X.shape[0]
    data_avg = X.mean(axis=1)
    sort_arg = np.argsort(data_avg)
    data_avg = data_avg[sort_arg]
    gene_list = np.array(gene_list)[sort_arg]
    X = X[sort_arg, :]

    assigned_bin = _get_assigned_bins(data_avg, cluster_len, n_bins)
    used_ctrl = _get_ctrl_use(assigned_bin, gene_list, target_dict,
                              n_ctrl, random_state)
    ctrl_score = X[np.isin(gene_list, used_ctrl), :].mean(axis=0).T
    features_score = X[np.isin(gene_list, target_dict["Pos"]), :].mean(axis=0).T
    scores = features_score - ctrl_score
    if file_path:
        pd.DataFrame({"score": scores}, index=sample_list).to_csv(file_path)

    return scores


if __name__ == '__main__':
    from importlib import import_module
    try:
        score_genes = import_module("scanpy.tools").score_genes
    except ImportError as exc:
        raise ImportError("Install scTenifoldpy[scanpy] to run the Scanpy scoring comparison.") from exc
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--random_state",
                        help="random seed", default=42, type=int)
    parser.add_argument("-o", "--output_path",
                        help="output directory, it will be automatically and recursively created",
                        default=".", type=str)
    parser.add_argument("-g", "--genes",
                        help="number of the genes in the test data",
                        default=1000, type=int)
    parser.add_argument("-s", "--samples",
                        help="number of the samples (cells/observations) in the test data",
                        default=100, type=int)
    parser.add_argument("-b", "--bins",
                        help="number of bins",
                        default=25, type=int)
    parser.add_argument("-c", "--ctrls",
                        help="number of controls",
                        default=50, type=int)
    args = parser.parse_args()

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_obj = TestDataGenerator(n_genes=args.genes,
                                 n_samples=args.samples,
                                 n_bins=args.bins,
                                 n_ctrl=args.ctrls,
                                 random_state=args.random_state)
    data_obj.save_data(output_dir / Path("test_data.csv"), use_normalized=True)
    np_data = data_obj.get_data("numpy", True)
    np_data["file_path"] = output_dir / Path("cell_scores.csv")
    pd_data = data_obj.get_data("pandas", True)
    pd_data["file_path"] = output_dir / Path("adobo_cell_scores.csv")

    cell_cycle_score(**np_data)
    score_genes(**(data_obj.get_data("ann_data", True))).write_csvs(output_dir / Path("scanpy_result"))
    adobo_score(**pd_data)
