from functools import reduce
from timeit import default_timer as timer
from typing import List, Optional, Sequence

import anndata as ad
import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import issparse


def _as_flow_matrix(adata: ad.AnnData, flow_expr_key: str) -> np.ndarray:
    """Extract a dense, numeric flow-expression matrix in the parent process."""
    if flow_expr_key not in adata.obsm:
        raise ValueError(f"adata.obsm does not contain flow_expr_key={flow_expr_key!r}.")

    samples = adata.obsm[flow_expr_key]
    if issparse(samples):
        samples = samples.toarray()
    samples = np.asarray(samples)

    if samples.ndim != 2:
        raise ValueError(
            f"adata.obsm[{flow_expr_key!r}] must be a two-dimensional matrix."
        )
    if samples.shape[0] != adata.n_obs:
        raise ValueError(
            f"adata.obsm[{flow_expr_key!r}] has {samples.shape[0]} rows, "
            f"but adata has {adata.n_obs} observations."
        )
    if not np.issubdtype(samples.dtype, np.number):
        raise ValueError(f"adata.obsm[{flow_expr_key!r}] must contain numeric values.")

    return samples


def _indices_by_block(labels: np.ndarray) -> List[np.ndarray]:
    """Return observation indices grouped by their spatial block label."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("Spatial block labels must be one-dimensional.")
    if labels.size == 0:
        raise ValueError("Spatial block labels cannot be empty.")

    _, inverse = np.unique(labels, return_inverse=True)
    return [np.flatnonzero(inverse == block) for block in range(inverse.max() + 1)]


def _resample_matrix(
    samples: np.ndarray,
    rng: np.random.Generator,
    indices_by_blocks: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    """Bootstrap rows globally or within each supplied spatial block."""
    if indices_by_blocks is None:
        return samples[rng.integers(0, samples.shape[0], size=samples.shape[0])]

    resampled = samples.copy()
    for indices in indices_by_blocks:
        indices = np.asarray(indices, dtype=int)
        if indices.size == 0:
            continue
        sampled_indices = rng.choice(indices, size=indices.size, replace=True)
        resampled[indices] = samples[sampled_indices]
    return resampled


def _drop_zero_std_columns(
    matrices: Sequence[np.ndarray],
) -> tuple[np.ndarray, List[np.ndarray]]:
    """Keep features with non-zero standard deviation in every matrix."""
    nonzero_masks = [np.std(matrix, axis=0) > 0 for matrix in matrices]
    keep = np.flatnonzero(reduce(np.logical_and, nonzero_masks))
    if keep.size == 0:
        raise ValueError("No flow variables have non-zero variance in this bootstrap.")
    return keep, [matrix[:, keep] for matrix in matrices]


def _learn_gsp(samples: np.ndarray, alpha: float) -> np.ndarray:
    from causaldag import gsp
    from causaldag import (
        MemoizedCI_Tester,
        partial_correlation_suffstat,
        partial_correlation_test,
    )

    nodes = set(range(samples.shape[1]))
    obs_suffstat = partial_correlation_suffstat(samples, invert=True)
    ci_tester = MemoizedCI_Tester(
        partial_correlation_test, obs_suffstat, alpha=alpha
    )
    est_dag = gsp(nodes, ci_tester, nruns=20)
    return est_dag.cpdag().to_amat()[0]


def _learn_utigsp(
    control_samples: np.ndarray,
    perturbed_samples: Sequence[np.ndarray],
    alpha: float,
    alpha_inv: float,
) -> tuple[np.ndarray, Sequence[set[int]]]:
    from causaldag import (
        MemoizedCI_Tester,
        MemoizedInvarianceTester,
        gauss_invariance_suffstat,
        gauss_invariance_test,
        partial_correlation_suffstat,
        partial_correlation_test,
        unknown_target_igsp,
    )

    nodes = set(range(control_samples.shape[1]))
    obs_suffstat = partial_correlation_suffstat(control_samples, invert=True)
    invariance_suffstat = gauss_invariance_suffstat(
        control_samples, list(perturbed_samples)
    )
    ci_tester = MemoizedCI_Tester(
        partial_correlation_test, obs_suffstat, alpha=alpha
    )
    invariance_tester = MemoizedInvarianceTester(
        gauss_invariance_test, invariance_suffstat, alpha=alpha_inv
    )
    setting_list = [dict(known_interventions=[]) for _ in perturbed_samples]

    est_dag, est_targets_list = unknown_target_igsp(
        setting_list,
        nodes,
        ci_tester,
        invariance_tester,
        nruns=20,
    )
    est_icpdag = est_dag.interventional_cpdag(
        est_targets_list, cpdag=est_dag.cpdag()
    )
    return est_icpdag.to_amat()[0], est_targets_list


def run_gsp(
    samples: np.ndarray,
    use_spatial: bool = False,
    indices_by_blocks: Optional[Sequence[np.ndarray]] = None,
    alpha: float = 1e-3,
    seed: int = 0,
):
    """Run one observational GSP bootstrap using serialization-safe inputs."""
    if use_spatial and indices_by_blocks is None:
        raise ValueError("Spatial block indices are required when use_spatial=True.")

    rng = np.random.default_rng(seed)
    resampled = _resample_matrix(
        samples,
        rng,
        indices_by_blocks=indices_by_blocks if use_spatial else None,
    )
    keep, (resampled,) = _drop_zero_std_columns([resampled])
    adjacency_cpdag = _learn_gsp(resampled, alpha)

    return {
        "nonzero_flow_vars_indices": keep,
        "adjacency_cpdag": adjacency_cpdag,
    }


def run_utigsp(
    control_samples: np.ndarray,
    perturbed_samples: Sequence[np.ndarray],
    use_spatial: bool = False,
    indices_by_blocks_control: Optional[Sequence[np.ndarray]] = None,
    indices_by_blocks_perturbed: Optional[Sequence[Sequence[np.ndarray]]] = None,
    alpha: float = 1e-3,
    alpha_inv: float = 1e-3,
    seed: int = 0,
):
    """Run one interventional GSP bootstrap using serialization-safe inputs."""
    if use_spatial and (
        indices_by_blocks_control is None or indices_by_blocks_perturbed is None
    ):
        raise ValueError("Spatial block indices are required when use_spatial=True.")

    rng = np.random.default_rng(seed)
    control_resampled = _resample_matrix(
        control_samples,
        rng,
        indices_by_blocks_control if use_spatial else None,
    )
    perturbed_resampled = [
        _resample_matrix(
            samples,
            rng,
            indices_by_blocks_perturbed[index] if use_spatial else None,
        )
        for index, samples in enumerate(perturbed_samples)
    ]

    keep, matrices = _drop_zero_std_columns(
        [control_resampled, *perturbed_resampled]
    )
    control_resampled, *perturbed_resampled = matrices
    adjacency_cpdag, targets = _learn_utigsp(
        control_resampled, perturbed_resampled, alpha, alpha_inv
    )
    perturbed_targets_indices = [
        keep[np.asarray(sorted(target_set), dtype=int)] for target_set in targets
    ]

    return {
        "nonzero_flow_vars_indices": keep,
        "adjacency_cpdag": adjacency_cpdag,
        "perturbed_targets_indices": perturbed_targets_indices,
    }


def learn_intercellular_flows(
    adata: ad.AnnData,
    condition_key: str = None,
    control_key: str = None,
    flowsig_key: str = "flowsig_network",
    flow_expr_key: str = "X_flow",
    use_spatial: Optional[bool] = False,
    block_key: Optional[str] = None,
    n_jobs: int = 1,
    n_bootstraps: int = 100,
    alpha_ci: float = 1e-3,
    alpha_inv: float = 1e-3,
):
    """
    Learn a causal intercellular signaling network by bootstrap aggregation.

    The public API and output schema are retained for compatibility. Parallel
    workers receive only NumPy matrices and integer indices; the AnnData object
    is read and mutated exclusively in the parent process.

    Parameters
    ----------
    adata
        Annotated data containing flow expressions and FlowSig variable metadata.
    condition_key
        Observation column separating control and perturbed conditions.
    control_key
        Control value in ``adata.obs[condition_key]``.
    flowsig_key
        Key in ``adata.uns`` containing FlowSig metadata and receiving results.
    flow_expr_key
        Key in ``adata.obsm`` containing the flow-expression matrix.
    use_spatial
        Whether to bootstrap observations within spatial blocks.
    block_key
        Observation column defining spatial blocks.
    n_jobs
        Number of joblib worker processes.
    n_bootstraps
        Number of bootstrap samples.
    alpha_ci
        Conditional-independence significance level.
    alpha_inv
        Conditional-invariance significance level.
    """
    if not isinstance(n_bootstraps, int) or n_bootstraps < 1:
        raise ValueError("n_bootstraps must be an integer greater than or equal to 1.")
    if not isinstance(n_jobs, int) or n_jobs == 0:
        raise ValueError("n_jobs must be a non-zero integer.")
    if flowsig_key not in adata.uns:
        raise ValueError(f"adata.uns does not contain flowsig_key={flowsig_key!r}.")
    if "flow_var_info" not in adata.uns[flowsig_key]:
        raise ValueError(
            f"adata.uns[{flowsig_key!r}] does not contain 'flow_var_info'."
        )

    samples = _as_flow_matrix(adata, flow_expr_key)
    flow_vars = list(adata.uns[flowsig_key]["flow_var_info"].index)
    if len(flow_vars) != samples.shape[1]:
        raise ValueError(
            "The number of flow variables does not match the columns in the "
            f"flow-expression matrix ({len(flow_vars)} != {samples.shape[1]})."
        )

    block_labels = None
    if use_spatial:
        if block_key is None:
            raise ValueError("block_key must be provided when use_spatial=True.")
        if block_key not in adata.obs:
            raise ValueError(f"adata.obs does not contain block_key={block_key!r}.")
        block_labels = np.asarray(adata.obs[block_key])
        if block_labels.shape[0] != samples.shape[0]:
            raise ValueError("Spatial block labels must match the number of observations.")

    if condition_key is not None:
        if condition_key not in adata.obs:
            raise ValueError(
                f"adata.obs does not contain condition_key={condition_key!r}."
            )
        if control_key is None:
            raise ValueError(
                "control_key must be provided when condition_key is specified."
            )

        condition_labels = np.asarray(adata.obs[condition_key])
        if condition_labels.shape[0] != samples.shape[0]:
            raise ValueError("Condition labels must match the number of observations.")
        conditions = list(adata.obs[condition_key].unique())
        if control_key not in conditions:
            raise ValueError(
                f"control_key={control_key!r} is not present in "
                f"adata.obs[{condition_key!r}]."
            )
        perturbed_keys = [condition for condition in conditions if condition != control_key]
        if not perturbed_keys:
            raise ValueError("At least one perturbed condition is required.")

        control_mask = condition_labels == control_key
        control_samples = samples[control_mask]
        perturbed_samples = [
            samples[condition_labels == condition] for condition in perturbed_keys
        ]
        control_blocks = (
            _indices_by_block(block_labels[control_mask]) if use_spatial else None
        )
        perturbed_blocks = (
            [
                _indices_by_block(block_labels[condition_labels == condition])
                for condition in perturbed_keys
            ]
            if use_spatial
            else None
        )

        print(f"starting computations on {n_jobs} cores")
        start = timer()
        bootstrap_results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(run_utigsp)(
                control_samples,
                perturbed_samples,
                use_spatial,
                control_blocks,
                perturbed_blocks,
                alpha_ci,
                alpha_inv,
                bootstrap,
            )
            for bootstrap in range(n_bootstraps)
        )
        print(f"elapsed time: {timer() - start}")

        bagged_adjacency = np.zeros((len(flow_vars), len(flow_vars)))
        bagged_perturbed_targets = [
            np.zeros(len(flow_vars)) for _ in perturbed_keys
        ]
        for result in bootstrap_results:
            keep = result["nonzero_flow_vars_indices"]
            bagged_adjacency[np.ix_(keep, keep)] += result["adjacency_cpdag"]
            for index, targets in enumerate(result["perturbed_targets_indices"]):
                bagged_perturbed_targets[index][targets] += 1

        bagged_adjacency /= float(n_bootstraps)
        bagged_perturbed_targets = [
            targets / float(n_bootstraps) for targets in bagged_perturbed_targets
        ]
        network_results = {
            "flow_vars": flow_vars,
            "adjacency": bagged_adjacency,
            "perturbed_targets": bagged_perturbed_targets,
        }
    else:
        indices_by_blocks = _indices_by_block(block_labels) if use_spatial else None

        print(f"starting computations on {n_jobs} cores")
        start = timer()
        bootstrap_results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(run_gsp)(
                samples,
                use_spatial,
                indices_by_blocks,
                alpha_ci,
                bootstrap,
            )
            for bootstrap in range(n_bootstraps)
        )
        print(f"elapsed time: {timer() - start}")

        bagged_adjacency = np.zeros((len(flow_vars), len(flow_vars)))
        for result in bootstrap_results:
            keep = result["nonzero_flow_vars_indices"]
            bagged_adjacency[np.ix_(keep, keep)] += result["adjacency_cpdag"]
        bagged_adjacency /= float(n_bootstraps)
        network_results = {
            "flow_vars": flow_vars,
            "adjacency": bagged_adjacency,
        }

    adata.uns[flowsig_key]["network"] = network_results
