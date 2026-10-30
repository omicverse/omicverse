"""Optimal transport through the optional moscot backend."""
from __future__ import annotations

import numpy as np

from ._optional import import_optional_module
from ._registry import register_function
from ._backend_validation import validate_adata, obs_column, coordinates, kwargs_copy


def _solve(problem, prepare_kwargs, solve_kwargs, require_convergence):
    problem.prepare(**prepare_kwargs)
    problem.solve(**solve_kwargs)
    if not problem.solutions:
        raise RuntimeError("moscot returned no transport solutions.")
    if require_convergence:
        failed = [key for key, value in problem.solutions.items() if not bool(value.converged)]
        if failed:
            raise RuntimeError(f"moscot did not converge for {failed}; adjust solve_kwargs.")
    return problem


@register_function(aliases=["moscot temporal", "temporal optimal transport"], category="single",
                   description="Solve time-course transport with moscot and check convergence")
def moscot_temporal(adata, *, time_key, joint_attr=None, prepare_kwargs=None,
                    solve_kwargs=None, require_convergence=True):
    """Prepare and solve temporal OT; return the native TemporalProblem.

    ``time_key`` must contain numeric sampling times, not category codes invented
    from condition labels. ``joint_attr`` names an obsm representation (e.g.
    ``X_pca``); None delegates local PCA to moscot. Parameters for prepare and
    solve are kept separate. No dense transport matrix is copied into AnnData.
    A nonconverged solve raises by default. The returned problem supports native
    moscot downstream analyses; its coupling is not a lineage ground truth.
    """
    validate_adata(adata)
    times = obs_column(adata, time_key, minimum_categories=2)
    try:
        numeric = np.asarray(times, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("time_key must contain numeric sampling times.") from exc
    if not np.isfinite(numeric).all() or any(isinstance(x, str) for x in times):
        raise ValueError("time_key must contain finite numeric sampling times.")
    if isinstance(joint_attr, str):
        if joint_attr not in adata.obsm:
            raise KeyError(f"obsm[{joint_attr!r}] is missing.")
    prepare = kwargs_copy(prepare_kwargs, reserved=("time_key", "joint_attr"))
    solve = kwargs_copy(solve_kwargs)
    backend = import_optional_module("moscot.problems.time", feature="Temporal OT",
                                     dependencies=("moscot",), install_hint="Install moscot.")
    return _solve(backend.TemporalProblem(adata),
                  dict(time_key=time_key, joint_attr=joint_attr, **prepare), solve,
                  require_convergence)


@register_function(aliases=["moscot alignment"], category="space",
                   description="Solve spatial slice couplings with moscot")
def moscot_alignment(adata, *, batch_key, spatial_key="spatial", joint_attr=None,
                     prepare_kwargs=None, solve_kwargs=None, require_convergence=True):
    """Solve spatial slice OT and return AlignmentProblem.

    Call the returned problem's ``align`` method explicitly to generate aligned
    coordinates. Solving a coupling alone does not overwrite physical positions.
    """
    validate_adata(adata)
    obs_column(adata, batch_key, minimum_categories=2)
    coordinates(adata, spatial_key)
    prepare = kwargs_copy(prepare_kwargs, reserved=("batch_key", "spatial_key", "joint_attr"))
    solve = kwargs_copy(solve_kwargs)
    backend = import_optional_module("moscot.problems.space", feature="Spatial OT",
                                     dependencies=("moscot",), install_hint="Install moscot.")
    return _solve(backend.AlignmentProblem(adata),
                  dict(batch_key=batch_key, spatial_key=spatial_key, joint_attr=joint_attr, **prepare),
                  solve, require_convergence)


@register_function(aliases=["moscot mapping"], category="space",
                   description="Map reference single cells to spatial observations with moscot")
def moscot_mapping(adata_sc, adata_sp, *, sc_attr=None, spatial_key="spatial",
                   batch_key=None, prepare_kwargs=None, solve_kwargs=None,
                   require_convergence=True):
    """Map single cells to space, returning the native MappingProblem.

    moscot's transport matrices have spatial observations on rows and reference
    single cells on columns. The two datasets must share gene identifiers.
    ``sc_attr=None`` supports linear OT when ``solve_kwargs={'alpha': 0}``;
    quadratic mapping requires a reference representation, e.g. ``X_pca``.
    """
    validate_adata(adata_sc)
    validate_adata(adata_sp)
    coordinates(adata_sp, spatial_key)
    if not len(adata_sc.var_names.intersection(adata_sp.var_names)):
        raise ValueError("Reference and spatial data must share feature names.")
    if batch_key is not None:
        obs_column(adata_sp, batch_key)
    if isinstance(sc_attr, str) and sc_attr not in adata_sc.obsm:
        raise KeyError(f"Reference obsm[{sc_attr!r}] is missing.")
    prepare = kwargs_copy(prepare_kwargs, reserved=("sc_attr", "spatial_key", "batch_key"))
    solve = kwargs_copy(solve_kwargs)
    if sc_attr is None and solve.get("alpha", 0.5) != 0:
        raise ValueError("Provide sc_attr for quadratic mapping or use solve_kwargs={'alpha': 0}.")
    if sc_attr is not None and solve.get("alpha", 0.5) == 0:
        raise ValueError("Linear mapping (alpha=0) requires sc_attr=None.")
    backend = import_optional_module("moscot.problems.space", feature="Spatial mapping",
                                     dependencies=("moscot",), install_hint="Install moscot.")
    return _solve(backend.MappingProblem(adata_sc=adata_sc, adata_sp=adata_sp),
                  dict(sc_attr=sc_attr, spatial_key=spatial_key, batch_key=batch_key, **prepare),
                  solve, require_convergence)
