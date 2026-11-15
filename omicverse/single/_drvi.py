"""``ov.single.drvi`` — thin convenience wrapper over :class:`ov.vi.DRVI`.

Public function name proposed by the closed contributor PR #1014 (which wrapped
``drvi-py``/scvi-tools); this is the omicverse-native replacement, backed by
``omicverse.vi.DRVI`` (see ``omicverse/vi/_drvi.py``) instead.
"""
from __future__ import annotations

from .._registry import register_function


@register_function(
    aliases=["drvi", "DRVI", "single.drvi", "disentangled representation VI"],
    category="single",
    description="Train DRVI (disentangled representation VI: an additive, "
                "per-latent-dimension decoder) and store its latent embedding in "
                "adata.obsm.",
    examples=[
        "m = ov.single.drvi(adata, layer='counts', batch_key='batch', n_latent=32)",
        "adata.obsm['X_drvi']",
    ],
    related=["vi.DRVI", "single.batch_correction"],
)
def drvi(adata, *, layer="counts", batch_key=None, labels_key=None,
         categorical_covariate_keys=None, continuous_covariate_keys=None, genes=None,
         key_added: str = "X_drvi", model_kwargs: dict | None = None,
         train_kwargs: dict | None = None, overwrite: bool = False):
    """Train DRVI on ``adata`` and write its latent representation.

    Arguments:
        adata: AnnData, AnnDataOOM (``ov.read(path, storage='disk')``) or
            Dask-backed AnnData.
        layer: Layer with raw counts (``None`` = ``X``).
        batch_key, labels_key, categorical_covariate_keys,
        continuous_covariate_keys, genes: As :class:`ov.vi.DRVI`.
        key_added: ``obsm`` key the latent mean is written to.
        model_kwargs: Extra constructor arguments (``n_latent``,
            ``gene_likelihood``, ``n_split_latent``, ``split_aggregation``, ...).
        train_kwargs: Arguments forwarded to :meth:`ov.vi.DRVI.train`
            (``max_epochs``, ``batch_size``, ``device``, ...).
        overwrite: Allow overwriting an existing ``adata.obsm[key_added]``.

    Returns:
        The trained :class:`ov.vi.DRVI` model (``model.differential_vars(dim)``
        and ``model.latent_traverse()`` give the per-latent-dimension gene
        effects used to interpret the embedding).

    Example::

        m = ov.single.drvi(adata, layer="counts", batch_key="batch", n_latent=32)
        genes, effect = m.differential_vars(dim=0, n_genes=20)
    """
    if key_added in adata.obsm and not overwrite:
        raise ValueError(f"adata.obsm[{key_added!r}] already exists; pass overwrite=True")
    from .. import vi

    m = vi.DRVI(adata, layer=layer, batch_key=batch_key, labels_key=labels_key,
               categorical_covariate_keys=categorical_covariate_keys,
               continuous_covariate_keys=continuous_covariate_keys, genes=genes,
               **dict(model_kwargs or {}))
    m.train(**dict(train_kwargs or {}))
    adata.obsm[key_added] = m.get_latent_representation()
    return m


__all__ = ["drvi"]
