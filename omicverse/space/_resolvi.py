"""Spatial expression denoising with ``ov.vi.RESOLVI`` (no scvi-tools)."""
from __future__ import annotations

from .._registry import register_function

_PROTECTED_SUFFIXES = ("index_neighbor", "distance_neighbor")


def _reserved_keys(spatial_key: str) -> set:
    return {spatial_key, "X_spatial", *_PROTECTED_SUFFIXES}


@register_function(
    aliases=["ResolVI", "resolvi", "space.resolvi", "空间表达去噪", "spatial expression denoising"],
    category="space",
    description="Train omicverse's native ResolVI (ov.vi.RESOLVI, no scvi-tools) to denoise "
                "imaging-based spatial transcriptomics counts into cell-intrinsic, spatial "
                "diffusion and ambient-background channels; writes the latent embedding (and "
                "optionally proportions / a denoised layer) without touching the input spatial "
                "coordinates or neighbour graph.",
    examples=[
        "m = ov.space.resolvi(adata, layer='counts', spatial_key='spatial', batch_key='slide')",
        "denoised = m.get_normalized_expression()",
    ],
    related=["vi.RESOLVI", "vi.SCVI"],
)
def resolvi(adata, *, layer="counts", spatial_key="spatial", batch_key=None, labels_key=None,
           n_neighbors=10, key_added="X_resolvi", proportions_key="resolvi_proportions",
           denoised_layer=None, model_kwargs=None, train_kwargs=None, overwrite=False):
    """Train :class:`omicverse.vi.RESOLVI` and write its outputs to ``adata``.

    The spatial coordinates (``obsm[spatial_key]``) and the model's internal
    neighbour graph are never written back to ``adata`` by this function (the
    model keeps them in its own registry, see :class:`omicverse.vi.RESOLVI`);
    ``key_added`` / ``proportions_key`` / ``denoised_layer`` must not collide
    with ``spatial_key`` or the reserved ``'X_spatial'``, ``'index_neighbor'``,
    ``'distance_neighbor'`` names, so a previous fit's spatial inputs can never
    be overwritten by this function's outputs. Pass a copy of ``adata`` to keep
    the input object itself unchanged otherwise (gene / count validation only).

    Arguments:
        adata: AnnData, AnnDataOOM or Dask-backed AnnData with raw counts in
            ``layer`` and 2-D/3-D coordinates in ``obsm[spatial_key]``.
        layer: Layer with raw counts.
        spatial_key: ``obsm`` key with spatial coordinates.
        batch_key: ``obs`` column separating physical slides.
        labels_key: ``obs`` column with cell-type labels (semi-supervised variant).
        n_neighbors: Spatial kNN size (>= 6).
        key_added: ``obsm`` key for the latent embedding.
        proportions_key: ``obsm`` key for the per-cell true/diffusion/background
            mixture weights (``None``: skip).
        denoised_layer: ``layers`` key for the denoised expression
            (``get_normalized_expression(library_size='latent')``; ``None``: skip
            — this can be expensive to materialise for many genes).
        model_kwargs: Passed to :class:`omicverse.vi.RESOLVI`.
        train_kwargs: Passed to ``model.train()``.
        overwrite: Replace an existing ``key_added`` / ``proportions_key`` /
            ``denoised_layer``.

    Returns:
        The trained :class:`omicverse.vi.RESOLVI` model.
    """
    from ..vi._resolvi import RESOLVI
    reserved = _reserved_keys(spatial_key)
    for key in (key_added, proportions_key, denoised_layer):
        if key is not None and key in reserved:
            raise ValueError(f"{key!r} must not replace the spatial coordinates or the "
                             "model's spatial neighbour graph")
    if not overwrite:
        if key_added in adata.obsm:
            raise ValueError(f"adata.obsm[{key_added!r}] already exists; pass overwrite=True")
        if proportions_key is not None and proportions_key in adata.obsm:
            raise ValueError(f"adata.obsm[{proportions_key!r}] already exists; "
                             "pass overwrite=True")
        if denoised_layer is not None and denoised_layer in adata.layers:
            raise ValueError(f"adata.layers[{denoised_layer!r}] already exists; "
                             "pass overwrite=True")

    model = RESOLVI(adata, layer=layer, spatial_key=spatial_key, batch_key=batch_key,
                    labels_key=labels_key, n_neighbors=n_neighbors, **(model_kwargs or {}))
    model.train(**(train_kwargs or {}))

    adata.obsm[key_added] = model.get_latent_representation()
    if proportions_key is not None:
        adata.obsm[proportions_key] = model.get_proportions().to_numpy()
    if denoised_layer is not None:
        adata.layers[denoised_layer] = model.get_normalized_expression(
            library_size="latent", return_numpy=True)
    return model


__all__ = ["resolvi"]
