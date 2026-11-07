"""Pertpy adapters for guide assignment, perturbation classification and distances."""
from __future__ import annotations

import numpy as np

from .._optional import import_optional_module
from .._registry import register_function
from .._backend_validation import expression, obs_column, validate_matrix, kwargs_copy


def _pertpy():
    return import_optional_module("pertpy", feature="Perturbation analysis",
                                  dependencies=("pertpy",), install_hint="Install pertpy.")


@register_function(aliases=["pertpy guide assignment"], category="single",
                   description="Assign guides from a guide-count modality using pertpy")
def assign_guides(guide_adata, *, assignment_threshold, layer=None,
                  key_added="assigned_guide", overwrite=False):
    """Assign each cell its highest-count guide and return its label Series.

    Input features must be guides, not RNA genes. Ties follow pertpy's first
    maximum rule. Below-threshold cells receive ``Negative``. This does not
    infer target genes, multiplexed perturbations, or perturbation efficacy.
    RNA metadata can be joined by cell ID explicitly after this step.
    """
    expression(guide_adata, layer, counts=True)
    if not np.isfinite(assignment_threshold) or assignment_threshold <= 0:
        raise ValueError("assignment_threshold must be positive and finite.")
    if not isinstance(key_added, str) or not key_added:
        raise ValueError("key_added must be a nonempty string.")
    if key_added in guide_adata.obs and not overwrite:
        raise ValueError(f"obs[{key_added!r}] already exists; use overwrite=True.")
    backend = _pertpy()
    work = guide_adata.copy()
    backend.pp.GuideAssignment().assign_to_max_guide(
        work, assignment_threshold=assignment_threshold, layer=layer, obs_key=key_added)
    if key_added not in work.obs or not work.obs_names.equals(guide_adata.obs_names):
        raise RuntimeError("pertpy did not return guide labels in input cell order.")
    guide_adata.obs[key_added] = work.obs[key_added].copy()
    return guide_adata.obs[key_added].copy()


@register_function(aliases=["pertpy Mixscape", "perturbation efficacy"], category="single",
                   description="Compute perturbation signatures and classify cells using Mixscape")
def mixscape(adata, *, pert_key, control, layer=None, split_by=None,
             signature_kwargs=None, mixscape_kwargs=None):
    """Return a new AnnData with perturbation signatures and Mixscape labels.

    ``layer`` supplies unscaled log-normalized expression. Every split must
    contain controls. Results describe cell-level efficacy, not a biological
    replicate-level hypothesis test. The input AnnData is not modified.
    """
    matrix = expression(adata, layer)
    labels = obs_column(adata, pert_key, minimum_categories=2)
    if control not in set(labels):
        raise ValueError("control is not present in the perturbation labels.")
    if split_by is not None:
        splits = obs_column(adata, split_by)
        if any(not (labels[splits == value] == control).any() for value in splits.unique()):
            raise ValueError("Every split must contain control cells.")
    signature = kwargs_copy(signature_kwargs, reserved=("adata", "pert_key", "control", "split_by", "copy"))
    classify = kwargs_copy(mixscape_kwargs, reserved=("adata", "pert_key", "control", "layer", "split_by", "copy"))
    work = adata.copy()
    work.X = matrix.copy()
    model = _pertpy().tl.Mixscape()
    model.perturbation_signature(work, pert_key=pert_key, control=control, split_by=split_by, **signature)
    model.mixscape(work, pert_key=pert_key, control=control, layer="X_pert", split_by=split_by, **classify)
    if "X_pert" not in work.layers:
        raise RuntimeError("Mixscape did not produce perturbation signatures.")
    class_key = classify.get("new_class_name", "mixscape_class")
    if class_key not in work.obs:
        raise RuntimeError("Mixscape did not produce cell classifications.")
    validate_matrix(work.layers["X_pert"], "Perturbation signatures")
    return work


@register_function(aliases=["pertpy E-distance", "perturbation distance"], category="single",
                   description="Compute descriptive group distances on an explicit representation")
def perturbation_distance(adata, *, groupby, use_rep="X_pca", metric="edistance",
                           distance_kwargs=None, pairwise_kwargs=None):
    """Return pertpy's group-distance table (or bootstrap mean/variance tuple).

    ``use_rep`` is an explicit obsm key. Distances quantify distributional
    differences; they are not p-values. With pooled cells they do not account
    for biological replication. The backend may cache pairwise distances in
    ``adata.obsp``; computation can require quadratic memory in cell count.
    """
    from .._backend_validation import validate_adata
    validate_adata(adata)
    obs_column(adata, groupby, minimum_categories=2)
    if use_rep not in adata.obsm:
        raise KeyError(f"obsm[{use_rep!r}] is missing.")
    validate_matrix(adata.obsm[use_rep], "Representation")
    init = kwargs_copy(distance_kwargs, reserved=("metric", "obsm_key", "layer_key"))
    pairwise = kwargs_copy(pairwise_kwargs, reserved=("adata", "groupby"))
    model = _pertpy().tl.Distance(metric=metric, obsm_key=use_rep, **init)
    result = model.pairwise(adata, groupby=groupby, **pairwise)
    for table in result if isinstance(result, tuple) else (result,):
        validate_matrix(table.to_numpy(), "Group distances")
    return result
