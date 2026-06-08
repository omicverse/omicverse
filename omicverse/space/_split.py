"""Core SPLIT workflow functions for AnnData objects."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

from .._registry import register_function


_EPSILON = 1e-10


@dataclass(frozen=True)
class _AlignedInputs:
    counts: sparse.spmatrix | np.ndarray
    weights: pd.DataFrame
    reference: pd.DataFrame
    primary: pd.Series
    obs_names: pd.Index
    var_names: pd.Index
    cell_types: pd.Index


def _layer_matrix(adata, layer: str | None):
    if layer is None or layer == "X":
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"`adata.layers[{layer!r}]` was not found.")
    return adata.layers[layer]


def _as_dataframe(value, index: pd.Index | None = None, columns: pd.Index | None = None, name: str = "matrix") -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    arr = value.toarray() if sparse.issparse(value) else np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f"`{name}` must be two-dimensional.")
    if index is not None and len(index) != arr.shape[0]:
        index = None
    if columns is not None and len(columns) != arr.shape[1]:
        columns = None
    return pd.DataFrame(arr, index=index, columns=columns)


def _align_weights(deconvolution_weights, obs_names: pd.Index) -> pd.DataFrame:
    weights = _as_dataframe(deconvolution_weights, name="deconvolution_weights")
    if weights.index.is_unique and set(obs_names).issubset(set(weights.index)):
        weights = weights.loc[obs_names]
    elif weights.shape[0] == len(obs_names):
        weights.index = obs_names
    else:
        raise ValueError(
            "`deconvolution_weights` must have one row per AnnData observation, "
            "or an index containing all `adata.obs_names`."
        )
    if weights.columns.hasnans or not weights.columns.is_unique:
        raise ValueError("`deconvolution_weights` columns must be unique cell-type names.")
    weights = weights.astype(float)
    weights.columns = weights.columns.astype(str)
    weights = weights.loc[:, sorted(weights.columns)]
    values = weights.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("`deconvolution_weights` must contain only finite values.")
    if (values < 0).any():
        raise ValueError("`deconvolution_weights` cannot contain negative values.")
    return weights


def _align_reference(reference, cell_types: pd.Index, var_names: pd.Index) -> pd.DataFrame:
    ref = _as_dataframe(reference, name="reference")
    ref.index = ref.index.astype(str)
    ref.columns = ref.columns.astype(str)
    cell_types = pd.Index(cell_types.astype(str))
    var_names = pd.Index(var_names.astype(str))

    index_celltypes = set(cell_types).issubset(set(ref.index))
    columns_genes = set(var_names).issubset(set(ref.columns))
    index_genes = set(var_names).issubset(set(ref.index))
    columns_celltypes = set(cell_types).issubset(set(ref.columns))

    if index_celltypes and columns_genes:
        ref = ref.loc[cell_types, var_names]
    elif index_genes and columns_celltypes:
        ref = ref.loc[var_names, cell_types].T
    elif ref.shape == (len(cell_types), len(var_names)):
        ref.index = cell_types
        ref.columns = var_names
    elif ref.shape == (len(var_names), len(cell_types)):
        ref = ref.T
        ref.index = cell_types
        ref.columns = var_names
    else:
        raise ValueError(
            "`reference` must be either cell-types x genes or genes x cell-types "
            "and align to deconvolution weights and `adata.var_names`."
        )
    ref = ref.astype(float)
    values = ref.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("`reference` must contain only finite values.")
    if (values < 0).any():
        raise ValueError("`reference` cannot contain negative values.")
    return ref


def _infer_primary(weights: pd.DataFrame, primary_cell_type) -> pd.Series:
    if primary_cell_type is None:
        primary = weights.idxmax(axis=1)
    elif isinstance(primary_cell_type, pd.Series):
        primary = primary_cell_type.copy()
        if set(weights.index).issubset(set(primary.index)):
            primary = primary.loc[weights.index]
        elif len(primary) == len(weights.index):
            primary.index = weights.index
        else:
            raise ValueError("`primary_cell_type` does not align to `adata.obs_names`.")
    else:
        primary = pd.Series(primary_cell_type, index=weights.index)
    primary = primary.astype(str)
    missing = sorted(set(primary) - set(weights.columns))
    if missing:
        raise ValueError(f"`primary_cell_type` contains types absent from weights: {missing[:5]}")
    return primary


def _prepare_inputs(adata, deconvolution_weights, reference, primary_cell_type, layer: str | None) -> _AlignedInputs:
    obs_names = pd.Index(adata.obs_names.astype(str))
    var_names = pd.Index(adata.var_names.astype(str))
    weights = _align_weights(deconvolution_weights, obs_names)
    reference = _align_reference(reference, pd.Index(weights.columns), var_names)
    primary = _infer_primary(weights, primary_cell_type)
    counts = _layer_matrix(adata, layer)
    if counts.shape != (adata.n_obs, adata.n_vars):
        raise ValueError("The selected AnnData matrix must be cells x genes.")
    data = counts.data if sparse.issparse(counts) else np.asarray(counts)
    if not np.isfinite(data).all():
        raise ValueError("The selected AnnData matrix must contain only finite values.")
    if (data < 0).any():
        raise ValueError("The selected AnnData matrix cannot contain negative counts.")
    return _AlignedInputs(counts, weights, reference, primary, obs_names, var_names, pd.Index(weights.columns))


def _normalize_weights(weights: pd.DataFrame) -> pd.DataFrame:
    row_sums = weights.sum(axis=1).astype(float)
    if (row_sums <= 0).any():
        bad = list(row_sums.index[row_sums <= 0][:5])
        raise ValueError(f"Each deconvolution row must have positive total weight; bad cells: {bad}")
    if not np.allclose(row_sums.to_numpy(), 1.0):
        warnings.warn("Some deconvolution weights do not sum to 1; rows were rescaled.", RuntimeWarning, stacklevel=2)
        weights = weights.div(row_sums, axis=0)
    return weights


def _second_type_and_weight(weights: pd.DataFrame, primary: pd.Series) -> tuple[pd.Series, pd.Series]:
    second_type = []
    second_weight = []
    for cell, row in weights.iterrows():
        candidates = row.copy()
        candidates.loc[primary.loc[cell]] = -np.inf
        if np.isneginf(candidates.max()) or candidates.max() <= 0:
            second_type.append(pd.NA)
            second_weight.append(0.0)
        else:
            ct = str(candidates.idxmax())
            second_type.append(ct)
            second_weight.append(float(row.loc[ct]))
    return pd.Series(second_type, index=weights.index, dtype="object"), pd.Series(second_weight, index=weights.index, dtype=float)


def _identity_rows_for_unselected(weights: pd.DataFrame, primary: pd.Series, selected: Iterable[str] | None) -> pd.DataFrame:
    if selected is None:
        return weights
    selected = set(map(str, selected))
    missing = selected - set(weights.index)
    if missing:
        raise ValueError(f"`cells_to_purify` contains cells absent from adata: {sorted(missing)[:5]}")
    out = weights.copy()
    keep_raw = [cell for cell in out.index if cell not in selected]
    if keep_raw:
        out.loc[keep_raw, :] = 0.0
        for cell in keep_raw:
            out.loc[cell, primary.loc[cell]] = 1.0
    return out


def _purify_matrix(counts: sparse.spmatrix | np.ndarray, weights: pd.DataFrame, reference: pd.DataFrame, primary: pd.Series, chunk_size: int):
    if chunk_size <= 0:
        raise ValueError("`chunk_size` must be positive.")
    weights_arr = weights.to_numpy(dtype=float)
    ref_arr = reference.to_numpy(dtype=float)
    primary_idx = reference.index.get_indexer(primary.to_numpy())
    primary_weight = weights_arr[np.arange(weights_arr.shape[0]), primary_idx]
    n_celltypes = np.maximum((weights_arr > 0).sum(axis=1), 1)
    is_sparse = sparse.issparse(counts)
    chunks = []
    for start in range(0, weights_arr.shape[0], chunk_size):
        stop = min(start + chunk_size, weights_arr.shape[0])
        denom = weights_arr[start:stop] @ ref_arr + _EPSILON
        numer = primary_weight[start:stop, None] * ref_arr[primary_idx[start:stop], :] + _EPSILON / n_celltypes[start:stop, None]
        ratio = numer / denom
        block_counts = counts[start:stop]
        chunks.append(block_counts.multiply(ratio).tocsr() if is_sparse else np.asarray(block_counts, dtype=float) * ratio)
    return sparse.vstack(chunks, format="csr") if is_sparse else np.vstack(chunks)


@register_function(
    aliases=["SPLIT", "split_purify", "spatial transcript purification", "transcript spillover cleanup"],
    category="space",
    description="Run SPLIT-style transcript purification from generic deconvolution weights and reference profiles.",
    requires={"layers": ["counts or selected layer"], "obs": ["optional primary cell type labels"]},
    produces={"layers": ["split_purified"], "obs": ["SPLIT purification metadata"]},
    examples=[
        "ov.space.split_purify(adata, weights, reference, primary_cell_type=adata.obs['cell_type'])",
        "ov.space.split_purify(adata, weights, reference, cells_to_purify=adata.obs_names[:100])",
    ],
    related=["space.split_spatial_score", "space.split_balance", "space.split_reassign_residuals"],
)
def split_purify(
    adata,
    deconvolution_weights,
    reference,
    primary_cell_type=None,
    layer: str | None = "counts",
    result_layer: str = "split_purified",
    cells_to_purify: Iterable[str] | None = None,
    chunk_size: int = 50000,
    copy: bool = False,
):
    """Purify spatial transcript counts with generic SPLIT inputs.

    Parameters
    ----------
    adata
        AnnData object containing raw counts.
    deconvolution_weights
        Deconvolution weights with shape `(n_obs, n_cell_types)`.
    reference
        Reference profile matrix in `(cell_type, gene)` format.
    primary_cell_type
        Optional primary labels; if `None`, inferred from max weight.
    layer
        Input count layer name. Use `None` or `"X"` for `adata.X`.
    result_layer
        Output layer name to store purified counts.
    cells_to_purify
        Optional list-like of cells to purify; unselected cells remain raw.
    chunk_size
        Internal block size for large matrices.
    copy
        If True, returns a copied object; otherwise operates in-place.

    Returns
    -------
    AnnData or None
        Updated AnnData when `copy=True`, else in-place with `None` return.
    """
    target = adata.copy() if copy else adata
    inputs = _prepare_inputs(target, deconvolution_weights, reference, primary_cell_type, layer)
    weights = _identity_rows_for_unselected(_normalize_weights(inputs.weights), inputs.primary, cells_to_purify)
    second_type, second_weight = _second_type_and_weight(weights, inputs.primary)
    target.layers[result_layer] = _purify_matrix(inputs.counts, weights, inputs.reference, inputs.primary, chunk_size)

    primary_weight = np.array([weights.loc[cell, inputs.primary.loc[cell]] for cell in weights.index], dtype=float)
    n_celltypes = (weights > 0).sum(axis=1).astype(int)
    target.obs["first_type"] = inputs.primary.to_numpy()
    target.obs["second_type"] = second_type.to_numpy()
    target.obs["weight_first_type"] = primary_weight
    target.obs["weight_second_type"] = second_weight.to_numpy()
    target.obs["split_primary_cell_type"] = inputs.primary.to_numpy()
    target.obs["split_w1_primary"] = primary_weight
    target.obs["split_n_cell_types"] = n_celltypes.to_numpy()
    target.obs["purification_status"] = np.where(n_celltypes.to_numpy() > 1, "purified", "raw")
    return target if copy else None


def _resolve_series(values, index: pd.Index, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.copy()
        if set(index).issubset(set(series.index)):
            series = series.loc[index]
        elif len(series) == len(index):
            series.index = index
        else:
            raise ValueError(f"`{name}` does not align to `adata.obs_names`.")
    else:
        series = pd.Series(values, index=index)
    return series


def _knn_indices(coords: np.ndarray, k: int, radius: float | None) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        raise ValueError("`k` must be positive.")
    if radius is not None and radius < 0:
        raise ValueError("`radius` must be non-negative when provided.")
    if coords.ndim != 2:
        raise ValueError("Spatial coordinates must be a two-dimensional array.")
    if not np.isfinite(coords).all():
        raise ValueError("Spatial coordinates must contain only finite values.")
    n = coords.shape[0]
    query_k = min(k + 1, n)
    dist, idx = cKDTree(coords).query(coords, k=query_k)
    if query_k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    if radius is not None:
        idx = idx.astype(int)
        idx[dist > radius] = -1
    return idx.astype(int), dist


@register_function(
    aliases=["split_spatial_score", "SPLIT spatially-aware score", "local diffusion score"],
    category="space",
    description="Compute SPLIT neighborhood second-type diffusion scores from spatial coordinates.",
    requires={"obsm": ["spatial"], "obs": ["primary and optional secondary labels"]},
    produces={"obs": ["neighborhood_weights_second_type", "neighborhood_weights_first_type"]},
    examples=["ov.space.split_spatial_score(adata, weights, adata.obs['first_type'])"],
    related=["space.split_purify", "space.split_balance"],
)
def split_spatial_score(
    adata,
    deconvolution_weights,
    primary_cell_type,
    secondary_cell_type=None,
    spatial_key: str = "spatial",
    k: int = 20,
    radius: float | None = None,
):
    """Compute local support for each cell's primary and secondary type."""
    if spatial_key not in adata.obsm:
        raise KeyError(f"`adata.obsm[{spatial_key!r}]` was not found.")
    obs_names = pd.Index(adata.obs_names.astype(str))
    weights = _normalize_weights(_align_weights(deconvolution_weights, obs_names))
    primary = _resolve_series(primary_cell_type, obs_names, "primary_cell_type").astype(str)
    if secondary_cell_type is None:
        secondary, second_weight = _second_type_and_weight(weights, primary)
    else:
        secondary = _resolve_series(secondary_cell_type, obs_names, "secondary_cell_type").astype("object")
        missing = sorted({str(value) for value in secondary.dropna()} - set(weights.columns))
        if missing:
            raise ValueError(f"`secondary_cell_type` contains types absent from weights: {missing[:5]}")
        second_weight = pd.Series([0.0 if pd.isna(secondary.loc[cell]) else float(weights.loc[cell, secondary.loc[cell]]) for cell in obs_names], index=obs_names)
    idx, dist = _knn_indices(np.asarray(adata.obsm[spatial_key], dtype=float), k=k, radius=radius)
    cell_types = list(weights.columns.astype(str))
    composition = []
    first_scores = []
    second_scores = []
    first_counts = []
    second_counts = []
    for i, neighbors in enumerate(idx):
        valid = [j for j in neighbors if j >= 0 and j != i]
        comp = pd.Series(0.0, index=cell_types)
        for j in valid:
            cell = obs_names[j]
            ft = primary.loc[cell]
            st = secondary.loc[cell]
            comp.loc[ft] += float(weights.loc[cell, ft])
            if not pd.isna(st) and st in comp.index:
                comp.loc[st] += float(second_weight.loc[cell])
        total = comp.sum()
        if total > 0:
            comp = comp / total
        focal = obs_names[i]
        ft = primary.loc[focal]
        st = secondary.loc[focal]
        first_scores.append(float(comp.loc[ft]) if ft in comp.index else 0.0)
        second_scores.append(float(comp.loc[st]) if not pd.isna(st) and st in comp.index else 0.0)
        first_counts.append(sum(primary.iloc[j] == ft for j in valid))
        second_counts.append(0 if pd.isna(st) else sum(primary.iloc[j] == st for j in valid))
        composition.append(comp.to_numpy())
    adata.obs["second_type"] = secondary.to_numpy()
    adata.obs["weight_second_type"] = second_weight.to_numpy()
    adata.obs["neighborhood_weights_first_type"] = first_scores
    adata.obs["neighborhood_weights_second_type"] = second_scores
    adata.obs["first_type_neighbors_N"] = first_counts
    adata.obs["second_type_neighbors_N"] = second_counts
    adata.uns["split_spatial_neighbors"] = {
        "indices": idx,
        "distances": dist,
        "cell_types": cell_types,
        "weight_composition": np.vstack(composition),
        "params": {"spatial_key": spatial_key, "k": k, "radius": radius},
    }
    return None


def _matrix_as_csr(mat):
    return mat.tocsr() if sparse.issparse(mat) else sparse.csr_matrix(np.asarray(mat))


def _swap_mask(obs: pd.DataFrame) -> np.ndarray:
    required = {"first_type_neighborhood_agreement", "first_type_class_neighborhood_agreement", "second_type_class", "first_type_class_neighborhood"}
    if required.issubset(obs.columns):
        return (
            (~obs["first_type_neighborhood_agreement"].astype(bool))
            & (~obs["first_type_class_neighborhood_agreement"].astype(bool))
            & (obs["second_type_class"].astype(str) == obs["first_type_class_neighborhood"].astype(str))
        ).to_numpy()
    if {"first_type", "second_type", "first_type_neighborhood"}.issubset(obs.columns):
        return (
            obs["second_type"].notna()
            & (obs["second_type"].astype(str) == obs["first_type_neighborhood"].astype(str))
            & (obs["first_type"].astype(str) != obs["first_type_neighborhood"].astype(str))
        ).to_numpy()
    return np.zeros(len(obs), dtype=bool)


@register_function(
    aliases=["split_balance", "spatially-aware SPLIT", "selective SPLIT purification"],
    category="space",
    description="Merge raw and SPLIT-purified counts using a neighborhood contamination score.",
    requires={"layers": ["counts", "split_purified"], "obs": ["neighborhood score"]},
    produces={"layers": ["split_balanced"], "obs": ["split_balance_status"]},
    examples=["ov.space.split_balance(adata, threshold=0.15)"],
    related=["space.split_purify", "space.split_spatial_score", "space.split_reassign_residuals"],
)
def split_balance(
    adata,
    purified_layer: str = "split_purified",
    score_key: str = "neighborhood_weights_second_type",
    threshold: float = 0.15,
    spot_class_key: str | None = None,
    result_layer: str = "split_balanced",
    raw_layer: str | None = "counts",
    swap_labels: bool = False,
):
    """Keep raw profiles for low-contamination cells and purified profiles otherwise."""
    if not np.isfinite(threshold):
        raise ValueError("`threshold` must be finite.")
    if purified_layer not in adata.layers:
        raise KeyError(f"`adata.layers[{purified_layer!r}]` was not found.")
    if score_key not in adata.obs:
        raise KeyError(f"`adata.obs[{score_key!r}]` was not found.")
    raw = _matrix_as_csr(_layer_matrix(adata, raw_layer))
    purified = _matrix_as_csr(adata.layers[purified_layer])
    if raw.shape != purified.shape:
        raise ValueError("Raw and purified matrices must have the same shape.")
    scores = np.asarray(adata.obs[score_key], dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError(f"`adata.obs[{score_key!r}]` must contain only finite values.")
    use_purified = scores > threshold
    removed = np.zeros(adata.n_obs, dtype=bool)
    if spot_class_key is not None:
        if spot_class_key not in adata.obs:
            raise KeyError(f"`adata.obs[{spot_class_key!r}]` was not found.")
        spot = adata.obs[spot_class_key].astype(str).to_numpy()
        use_purified |= spot == "doublet_uncertain"
        removed = spot == "reject"
    balanced = raw.copy().tolil()
    if use_purified.any():
        balanced[use_purified, :] = purified[use_purified, :]
    if removed.any():
        balanced[removed, :] = 0
    adata.layers[result_layer] = balanced.tocsr()
    status = np.full(adata.n_obs, "raw", dtype=object)
    status[use_purified] = "purified"
    status[removed] = "removed"
    adata.obs["split_balance_status"] = status

    if swap_labels:
        swap = _swap_mask(adata.obs)
        adata.obs["split_shift_swap"] = swap
        for first, second in (("first_type", "second_type"), ("weight_first_type", "weight_second_type")):
            if first in adata.obs and second in adata.obs:
                a = adata.obs[first].copy()
                adata.obs.loc[swap, first] = adata.obs.loc[swap, second]
                adata.obs.loc[swap, second] = a.loc[swap]
    else:
        adata.obs["split_shift_swap"] = False
    return None


@register_function(
    aliases=["split_reassign_residuals", "SPLIT residual reassignment", "residual transcript reassignment"],
    category="space",
    description="Redistribute SPLIT residual transcripts across spatial neighbors.",
    requires={"layers": ["counts", "split_balanced"], "obsm": ["spatial"]},
    produces={"layers": ["split_reassigned"], "uns": ["split_reassignment_operator", "split_residual_stats"]},
    examples=["ov.space.split_reassign_residuals(adata, mode='count_proportional')"],
    related=["space.split_balance", "space.split_spatial_score"],
)
def split_reassign_residuals(
    adata,
    raw_layer: str | None = "counts",
    purified_layer: str = "split_balanced",
    spatial_key: str = "spatial",
    mode: Literal["uniform", "count_proportional"] = "count_proportional",
    result_layer: str = "split_reassigned",
    k: int = 20,
    radius: float | None = None,
    self_keep: float = 0.0,
):
    """Reassign positive residual counts from purified cells to neighbors."""
    if mode not in {"uniform", "count_proportional"}:
        raise ValueError("`mode` must be 'uniform' or 'count_proportional'.")
    if purified_layer not in adata.layers:
        raise KeyError(f"`adata.layers[{purified_layer!r}]` was not found.")
    if spatial_key not in adata.obsm:
        raise KeyError(f"`adata.obsm[{spatial_key!r}]` was not found.")
    if not 0 <= self_keep <= 1:
        raise ValueError("`self_keep` must be between 0 and 1.")
    raw = _matrix_as_csr(_layer_matrix(adata, raw_layer))
    purified = _matrix_as_csr(adata.layers[purified_layer])
    if raw.shape != purified.shape:
        raise ValueError("Raw and purified matrices must have the same shape.")
    residual = raw - purified
    residual.data[residual.data < 0] = 0
    residual.eliminate_zeros()

    idx, _ = _knn_indices(np.asarray(adata.obsm[spatial_key], dtype=float), k=k, radius=radius)
    if "split_balance_status" in adata.obs:
        senders = adata.obs["split_balance_status"].astype(str).to_numpy() == "purified"
    elif "purification_status" in adata.obs:
        senders = adata.obs["purification_status"].astype(str).to_numpy() == "purified"
    else:
        senders = np.asarray(residual.sum(axis=1)).ravel() > 0
    raw_counts_per_cell = np.asarray(raw.sum(axis=1)).ravel()
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    sender_has_receivers = np.zeros(adata.n_obs, dtype=bool)
    for sender, is_sender in enumerate(senders):
        if not is_sender:
            continue
        receivers = np.array([j for j in idx[sender] if j >= 0 and j != sender], dtype=int)
        if receivers.size == 0:
            continue
        sender_has_receivers[sender] = True
        if self_keep > 0:
            rows.append(sender)
            cols.append(sender)
            vals.append(float(self_keep))
        if mode == "uniform":
            weights = np.full(receivers.size, 1.0 / receivers.size)
        elif mode == "count_proportional":
            weights = raw_counts_per_cell[receivers].astype(float)
            weights[~np.isfinite(weights) | (weights < 0)] = 0
            total = weights.sum()
            weights = weights / total if total > 0 else np.full(receivers.size, 1.0 / receivers.size)
        rows.extend([sender] * receivers.size)
        cols.extend(receivers.tolist())
        vals.extend((weights * (1 - self_keep)).tolist())
    operator = sparse.csr_matrix((vals, (rows, cols)), shape=(adata.n_obs, adata.n_obs))
    reassigned = purified + operator.T @ residual
    reassigned.data[reassigned.data < 0] = 0
    reassigned.eliminate_zeros()
    adata.layers[result_layer] = reassigned.tocsr()
    adata.uns["split_reassignment_operator"] = operator
    residual_mass_by_sender = np.asarray(residual.sum(axis=1)).ravel()
    assigned_mass = float(residual_mass_by_sender[sender_has_receivers].sum())
    unassigned_mass = float(residual_mass_by_sender[senders & ~sender_has_receivers].sum())
    adata.uns["split_residual_stats"] = {
        "raw_total": float(raw.sum()),
        "purified_total": float(purified.sum()),
        "positive_residual_total": float(residual.sum()),
        "assigned_residual_total": assigned_mass * (1 - self_keep),
        "self_kept_residual_total": assigned_mass * self_keep,
        "unassigned_residual_total": unassigned_mass,
        "n_sender_cells": int(senders.sum()),
        "n_sender_cells_with_receivers": int(sender_has_receivers.sum()),
        "mode": mode,
        "k": k,
        "radius": radius,
        "self_keep": self_keep,
    }
    return None


purify = split_purify
spatial_score = split_spatial_score
balance = split_balance
reassign_residuals = split_reassign_residuals

__all__ = [
    "split_purify",
    "split_spatial_score",
    "split_balance",
    "split_reassign_residuals",
]
