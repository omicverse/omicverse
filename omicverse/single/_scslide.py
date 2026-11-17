"""Optional scSLIDE adapter for sample-level analysis; never imports R.

The independent scslide-py package supplies the numerical implementation.
The adapter owns input validation and the cell/sample axis boundary only.
"""
from __future__ import annotations

import importlib
from importlib import metadata
from typing import TYPE_CHECKING, Any, Mapping

from .._registry import register_function

if TYPE_CHECKING:
    from anndata import AnnData
    from scslide.pipeline import WorkflowResult

_ALLOWED_OPTIONS = frozenset({
    "dims_embedding", "ncells_per_group", "ncells_landmark", "ncomp_pls",
    "pls_function", "k_nn", "knn_range", "fix_wnn_weights", "min_lmc_count",
    "recompute_residuals_after_filter", "scale_max_sample", "ncomp_diffusion",
    "fit_curve", "de_celltype", "de_covariates", "reference_profile",
})


def _require_backend():
    # Only the optional package itself being absent becomes an install hint.
    # An ImportError inside an installed backend must retain its real cause.
    try:
        backend = importlib.import_module("scslide")
    except ModuleNotFoundError as exc:
        if exc.name != "scslide":
            raise
        raise ImportError(
            "ov.single.scslide requires the independent scslide-py package "
            "with its [anndata] extra. Install its approved release artifact; "
            "see the scSLIDE tutorial for the verified artifact and availability."
        ) from exc
    from packaging.version import Version
    try:
        version_text = metadata.version("scslide-py")
    except metadata.PackageNotFoundError as exc:
        raise ImportError("Install scslide-py as a distribution; an unrelated local scslide.py is not supported") from exc
    version = Version(version_text)
    if version < Version("0.3.3") or version >= Version("0.4") or version.is_prerelease:
        raise ImportError(
            f"This adapter supports the scslide-py 0.3.x API starting at 0.3.3; found {version_text}. "
            "Check the documented tested artifact and support matrix."
        )
    if not callable(getattr(backend, "run_scslide_workflow", None)):
        raise ImportError("Installed scslide-py does not expose run_scslide_workflow")
    return backend, version_text


def _validate_matrix(matrix, *, name: str, counts: bool = False) -> None:
    import numpy as np
    from scipy import sparse
    if sparse.issparse(matrix):
        if getattr(matrix, "format", None) not in {"csr", "csc"}:
            raise TypeError(f"{name} must be a NumPy array or CSR/CSC matrix")
        values = matrix.data
    elif isinstance(matrix, np.ndarray):
        values = matrix
    else:
        raise TypeError(f"{name} must be in-memory NumPy or CSR/CSC; lazy/backed/GPU arrays are not supported by this adapter")
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if np.asarray(values).dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name} must be finite and nonnegative")
    if counts and not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{name} must contain raw integer-valued counts, not normalized expression")


@register_function(
    aliases=["scSLIDE", "scslide", "sample-level landmark density", "样本级细胞状态分布"],
    category="single",
    description=("Run the independent native-Python scSLIDE backend on explicitly "
                 "prepared cell AnnData and return a separate sample-level AnnData "
                 "inside WorkflowResult. Input is not mutated. This is not RNA velocity "
                 "or a general confirmatory gene-discovery procedure."),
    examples=[
        "result = ov.single.scslide(adata, sample_key='sample', response_key='condition', "
        "subject_key='subject', embedding_key='X_pca')",
        "sample_adata = result.sample_adata  # obs=samples, var=landmarks",
    ],
    related=["single.mrvi", "single.pseudobulk"],
)
def scslide(
    adata: "AnnData",
    *,
    sample_key: str,
    response_key: str,
    subject_key: str | None = None,
    celltype_key: str | None = None,
    counts_layer: str | None = "counts",
    data_layer: str = "data",
    hvg_key: str = "highly_variable",
    embedding_key: str | None = "X_pca",
    n_jobs: int = 1,
    random_state: int = 123,
    neighbor_backend: str = "exact",
    workflow_kwargs: Mapping[str, Any] | None = None,
) -> "WorkflowResult":
    """Run scSLIDE with an explicit cell-to-sample data contract.

    Parameters
    ----------
    adata
        In-memory cell-by-gene AnnData. This initial adapter supports NumPy,
        CSR and CSC expression arrays only, not backed, Dask or GPU data.
        A working copy is made; the input object is never modified.
    sample_key, response_key
        Columns in ``obs``. Response is constant within each biological
        sample, nonmissing, and has at least two distinct values.
    subject_key
        Optional independent-subject identifier. Repeated samples are not
        automatically independent; absence of this field does not establish
        independence. No subject prefix is guessed.
    celltype_key
        Optional cell-type column for grouped feature selection and
        descriptive, cell-type-specific pseudobulk analyses.
    counts_layer
        Raw nonnegative integer-valued counts; ``None`` means ``X``.
    data_layer
        Existing nonnegative log-normalized expression. This adapter does
        not silently normalize data or recompute highly variable genes.
    hvg_key
        Boolean, nonmissing ``var`` column selecting at least one gene.
    embedding_key
        Existing cell-state embedding in ``obsm``. ``None`` explicitly uses
        the single-modality PLS route instead of the multimodal workflow.
    n_jobs, random_state
        Forwarded to the backend. Requested and effective worker metadata
        remain available in ``result.inference_summary``.
    neighbor_backend
        ``'exact'`` (default) or explicitly selected ``'annoy'``. This adapter
        does not silently fall back when an optional backend fails.
    workflow_kwargs
        Optional native workflow parameters such as ``ncomp_pls``,
        ``ncells_landmark``, ``ncomp_diffusion`` and ``fit_curve``. Reserved
        identity, data and inference arguments cannot be overridden.
        ``de_celltype`` is opt-in and uses descriptive R-compatible inference.
        When requesting DE, set ``de_covariates`` explicitly (an empty list is
        allowed); missing covariates are rejected rather than silently dropped.

    Returns
    -------
    scslide.pipeline.WorkflowResult
        Native model/state and outputs. ``sample_adata`` has observations
        equal to samples, NOT input cells, and variables equal to landmarks.
        Its ``obsm['X_scslide_diffmap']`` is aligned by sample ID. Optional
        ``obs['scslide_trajectory']`` has arbitrary geometric orientation;
        it is not automatically a disease progression direction.
        Rectangular cell-to-landmark information remains in the backend's
        copied cell object, never in the original cell ``obsp``.

    Notes
    -----
    Supervision uses response labels; separation by those same labels is
    not independent predictive validation. Gene p/q values from a learned
    trajectory are descriptive/post-selection. Experimental permutations
    and count-splitting modes are intentionally outside this adapter.
    The standalone package's further APIs remain available separately.
    Version strings do not distinguish the historical private 0.3.3 builds;
    use the documented artifact SHA256 when reproducing their results.
    """
    import numpy as np
    import pandas as pd
    from anndata import AnnData

    if not isinstance(adata, AnnData):
        raise TypeError("adata must be an in-memory anndata.AnnData")
    if adata.isbacked:
        raise ValueError(
            "Backed AnnData is not supported by this adapter; use an explicit "
            "in-memory subset or standalone scslide API"
        )
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("adata must contain cells and genes")
    for axis_name, index in (("obs_names", adata.obs_names), ("var_names", adata.var_names)):
        if not index.is_unique or index.hasnans or not index.map(str).is_unique:
            raise ValueError(f"{axis_name} must be unique, nonmissing and remain unique when stringified")
    for key in (sample_key, response_key, subject_key, celltype_key):
        if key is None:
            continue
        if not isinstance(key, str) or not key or key not in adata.obs:
            raise ValueError(f"Missing or invalid obs column {key!r}")
        if adata.obs[key].isna().any():
            raise ValueError(f"obs[{key!r}] contains missing values; subset or repair metadata explicitly")
    if not isinstance(sample_key, str) or not isinstance(response_key, str):
        raise TypeError("sample_key and response_key must be column names")
    for key in (sample_key, subject_key):
        if key is not None:
            values = adata.obs[key]
            if values.nunique() != values.astype(str).nunique():
                raise ValueError(f"obs[{key!r}] has IDs that collide when stringified")
    for key in (response_key, subject_key):
        if key is not None:
            sizes = adata.obs.groupby(sample_key, observed=True, sort=False)[key].nunique()
            if (sizes > 1).any():
                raise ValueError(f"obs[{key!r}] must be constant within each sample")
    if adata.obs[response_key].nunique() < 2:
        raise ValueError("response_key must have at least two distinct values")
    n_samples = adata.obs[sample_key].nunique()
    if n_samples < 3:
        raise ValueError("At least three samples are required for sample diffusion analysis")
    if counts_layer is not None and counts_layer not in adata.layers:
        raise ValueError(f"Raw counts layer {counts_layer!r} is missing; no silent X fallback")
    if not isinstance(data_layer, str) or data_layer not in adata.layers:
        raise ValueError(f"Normalized data layer {data_layer!r} is missing")
    if counts_layer == data_layer:
        raise ValueError("counts_layer and data_layer must describe distinct raw and normalized matrices")
    counts = adata.X if counts_layer is None else adata.layers[counts_layer]
    _validate_matrix(counts, name="counts", counts=True)
    _validate_matrix(adata.layers[data_layer], name="normalized data")
    if hvg_key not in adata.var:
        raise ValueError(f"HVG column {hvg_key!r} is missing; select features explicitly first")
    hvg = adata.var[hvg_key]
    if not pd.api.types.is_bool_dtype(hvg.dtype) or hvg.isna().any() or not hvg.any():
        raise ValueError("HVG mask must be boolean, nonmissing and select at least one feature")
    if embedding_key is not None:
        if embedding_key not in adata.obsm:
            raise ValueError(f"Missing cell embedding {embedding_key!r}")
        embedding = np.asarray(adata.obsm[embedding_key])
        if embedding.ndim != 2 or embedding.shape[0] != adata.n_obs or embedding.shape[1] < 1:
            raise ValueError("Embedding must be a cell-by-dimension array")
        if embedding.dtype.kind not in "iuf" or not np.isfinite(embedding).all():
            raise ValueError("Embedding must be finite and real-valued")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be -1 or a positive integer")
    if isinstance(random_state, bool) or not isinstance(random_state, int) or random_state < 0:
        raise ValueError("random_state must be a nonnegative integer")
    if neighbor_backend not in {"exact", "annoy"}:
        raise ValueError("neighbor_backend must be 'exact' or explicitly selected 'annoy'")
    options = dict(workflow_kwargs or {})
    unknown = set(options) - _ALLOWED_OPTIONS
    if unknown:
        raise ValueError(f"Unsupported or reserved workflow_kwargs: {sorted(unknown)}")
    components = options.get("ncomp_diffusion", 10)
    if isinstance(components, bool) or not isinstance(components, int) or not 1 <= components < n_samples:
        raise ValueError("Set workflow_kwargs['ncomp_diffusion'] between 1 and n_samples - 1")
    if options.get("de_celltype") is not None:
        if celltype_key is None or options["de_celltype"] not in set(adata.obs[celltype_key].astype(str)):
            raise ValueError("de_celltype must name an observed category of celltype_key")
        if "de_covariates" not in options:
            raise ValueError("Set de_covariates explicitly for descriptive DE; use [] for no covariates")
        covariates = options["de_covariates"]
        if not isinstance(covariates, (list, tuple)) or any(not isinstance(c, str) for c in covariates):
            raise TypeError("de_covariates must be a list or tuple of column names")
        for key in covariates:
            if key == "log_count":
                continue  # Generated by the native pseudobulk operation.
            if key not in adata.obs or adata.obs[key].isna().any():
                raise ValueError(f"DE covariate {key!r} is missing or contains missing values")
            sizes = adata.obs.groupby(sample_key, observed=True, sort=False)[key].nunique()
            if (sizes > 1).any():
                raise ValueError(f"DE covariate {key!r} must be constant within each sample")
    backend, backend_version = _require_backend()
    working = adata.copy()
    raw_copy = working.X if counts_layer is None else working.layers[counts_layer]
    normalized_copy = working.layers[data_layer]
    working.layers["counts"] = raw_copy
    working.layers["data"] = normalized_copy
    working.var["highly_variable"] = hvg.to_numpy(dtype=bool)
    result = backend.run_scslide_workflow(
        working, sample_col=sample_key, response_col=response_key,
        subject_col=subject_key, celltype_col=celltype_key,
        embedding_key=embedding_key, n_jobs=n_jobs, seed=random_state,
        nn_method=neighbor_backend, de_inference_mode="descriptive_r_compat",
        **options,
    )
    sample = result.sample_adata
    dims = result.diffusion.embeddings
    if not sample.obs_names.is_unique or not dims.index.is_unique:
        raise RuntimeError("Backend returned duplicate sample IDs")
    if set(sample.obs_names) != set(dims.index):
        raise RuntimeError("Backend sample AnnData and diffusion IDs disagree")
    sample.obsm["X_scslide_diffmap"] = dims.loc[sample.obs_names].to_numpy(copy=True)
    if result.principal_curve is not None:
        trajectory = pd.Series(result.principal_curve.lambda_, index=dims.index)
        sample.obs["scslide_trajectory"] = trajectory.loc[sample.obs_names].to_numpy()
    sample.uns["omicverse_scslide"] = {
        "adapter_schema": "1", "backend_distribution": "scslide-py",
        "backend_version": backend_version, "sample_key": sample_key,
        "response_key": response_key, "subject_key": subject_key or "",
        "celltype_key": celltype_key or "", "embedding_key": embedding_key or "",
        "counts_source": "X" if counts_layer is None else counts_layer,
        "normalized_source": data_layer, "hvg_key": hvg_key,
        "observation_unit": "sample", "variable_unit": "landmark",
        "input_mutated": False, "seed": random_state, "n_jobs_requested": n_jobs,
        "neighbor_backend_requested": neighbor_backend,
        "inference_mode": "descriptive_r_compat",
        "confirmatory_calibration_established": False,
        "trajectory_direction": "unoriented; interpret with an explicit scientific endpoint",
    }
    return result


__all__ = ["scslide"]
