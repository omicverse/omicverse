"""Spatial expression denoising with the optional ResolVI backend."""
from __future__ import annotations

import numpy as np

from .._optional import import_optional_module
from .._registry import register_function
from .._backend_validation import (
    expression, coordinates, obs_column, kwargs_copy, output_key, write_embedding,
)


@register_function(aliases=["ResolVI"], category="space",
                   description="Train ResolVI to model spatial expression noise and contamination")
def resolvi(adata, *, layer="counts", spatial_key="spatial", batch_key=None,
            n_neighbors=10, key_added="X_resolvi", setup_kwargs=None,
            model_kwargs=None, train_kwargs=None, overwrite=False):
    """Train scvi-tools ResolVI and return the model for denoising/DE.

    The embedding is stored in ``obsm[key_added]``. Corrected expression can be
    computed explicitly with ``model.get_normalized_expression()``; it is not
    raw counts and never replaces the input count layer. ``batch_key`` separates
    physical slides when building spatial neighbors. scvi-tools 1.5.x is the
    tested backend; its future migration to scviva-tools is not silently applied.
    """
    matrix = expression(adata, layer, counts=True)
    coordinates(adata, spatial_key)
    if np.any(np.asarray(matrix.sum(axis=1)).ravel() <= 0):
        raise ValueError("ResolVI requires cells with positive total counts.")
    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int) or n_neighbors < 6:
        raise ValueError("ResolVI empirical priors require n_neighbors >= 6.")
    sizes = [adata.n_obs]
    if batch_key is not None:
        sizes = obs_column(adata, batch_key).value_counts().loc[lambda x: x > 0].to_numpy()
    if min(sizes) <= n_neighbors + 5:
        raise ValueError("Each spatial batch must have more than n_neighbors + 5 cells.")
    if adata.n_vars < 50:
        raise ValueError("ResolVI's empirical background prior requires at least 50 genes.")
    output_key(adata, key_added, overwrite=overwrite)
    setup = kwargs_copy(setup_kwargs, reserved=("adata", "layer", "batch_key", "prepare_data", "prepare_data_kwargs"))
    model_args = kwargs_copy(model_kwargs, reserved=("adata",))
    train = kwargs_copy(train_kwargs)
    backend = import_optional_module("scvi.external", feature="ResolVI", dependencies=("scvi",),
                                     install_hint="Install scvi-tools with ResolVI support.")
    backend.RESOLVI.setup_anndata(adata, layer=layer, batch_key=batch_key,
                                 prepare_data=True, prepare_data_kwargs={
                                     "spatial_rep": spatial_key, "n_neighbors": n_neighbors}, **setup)
    model = backend.RESOLVI(adata, **model_args)
    model.train(**train)
    write_embedding(adata, key_added, model.get_latent_representation())
    return model
