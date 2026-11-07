"""Validation shared by optional analysis adapters (no backend imports)."""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
from scipy import sparse


def validate_adata(adata):
    from anndata import AnnData
    if not isinstance(adata, AnnData):
        raise TypeError("Expected an AnnData object.")
    if adata.isbacked or adata.is_view:
        raise ValueError("Use an in-memory AnnData copy, not a backed object or view.")
    if not adata.n_obs or not adata.n_vars:
        raise ValueError("AnnData must contain cells and features.")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Cell and feature names must be unique.")


def validate_matrix(value, name, *, counts=False):
    data = value.data if sparse.issparse(value) else np.asarray(value)
    if (not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data)
            or not np.isfinite(data).all()):
        raise ValueError(f"{name} must contain finite numeric values.")
    if counts and (np.any(data < 0) or not np.allclose(data, np.rint(data), rtol=0, atol=1e-6)):
        raise ValueError(f"{name} must contain nonnegative integer counts, not normalized expression.")


def expression(adata, layer, *, counts=False):
    validate_adata(adata)
    if layer is not None and layer not in adata.layers:
        raise KeyError(f"Expression layer {layer!r} is missing.")
    value = adata.X if layer is None else adata.layers[layer]
    if value is None:
        raise ValueError("Expression matrix is missing.")
    validate_matrix(value, "Expression", counts=counts)
    return value


def obs_column(adata, key, *, minimum_categories=1):
    if not isinstance(key, str) or key not in adata.obs:
        raise KeyError(f"Observation column {key!r} is missing.")
    values = adata.obs[key]
    if values.isna().any() or values.nunique() < minimum_categories:
        raise ValueError(f"obs[{key!r}] must have no missing values and at least {minimum_categories} distinct values.")
    return values


def coordinates(adata, key):
    if key not in adata.obsm:
        raise KeyError(f"obsm[{key!r}] is missing.")
    value = adata.obsm[key]
    if len(value.shape) != 2 or value.shape[0] != adata.n_obs or value.shape[1] < 2:
        raise ValueError(f"obsm[{key!r}] must have one row per cell and at least two columns.")
    validate_matrix(value, key)


def kwargs_copy(value, *, reserved=()):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("Backend keyword arguments must be a mapping.")
    overlap = set(value).intersection(reserved)
    if overlap:
        raise ValueError(f"Use the named arguments instead of overriding {sorted(overlap)}.")
    return dict(value)


def write_embedding(adata, key, value):
    value = np.asarray(value)
    if value.ndim != 2 or value.shape[0] != adata.n_obs or value.shape[1] == 0:
        raise ValueError("Backend embedding must have one row per input cell.")
    validate_matrix(value, "Backend embedding")
    adata.obsm[key] = value


def output_key(adata, key, *, overwrite=False):
    if not isinstance(key, str) or not key:
        raise ValueError("Output key must be a nonempty string.")
    if key in adata.obsm and not overwrite:
        raise ValueError(f"obsm[{key!r}] already exists; choose another key or use overwrite=True.")
