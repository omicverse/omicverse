"""Optional DRVI adapter."""
from __future__ import annotations

from .._optional import import_optional_module
from .._registry import register_function
from .._backend_validation import expression, obs_column, kwargs_copy, output_key, write_embedding


from ._scvi_training import _fit


@register_function(aliases=["DRVI"], category="single",
                   description="Learn disentangled count-based representations with DRVI")
def drvi(adata, *, layer="counts", batch_key=None, key_added="X_drvi", setup_kwargs=None,
         model_kwargs=None, train_kwargs=None, overwrite=False):
    """Train count-based scvi-tools DRVI; return the model and save its embedding.

    Supports count likelihoods only. Gaussian DRVI needs a distinct preprocessing
    contract and is deliberately rejected instead of accepting normalized data
    through a count-based interface. Latent dimensions are not gene regulons.
    """
    expression(adata, layer, counts=True)
    if batch_key is not None:
        obs_column(adata, batch_key)
    init = kwargs_copy(model_kwargs)
    if init.get("gene_likelihood", "pnb") not in ("nb", "pnb", "zinb", "poisson"):
        raise ValueError("This DRVI adapter supports count likelihoods only.")
    setup = kwargs_copy(setup_kwargs, reserved=("adata", "layer", "batch_key"))
    return _fit(adata, "DRVI", layer, dict(batch_key=batch_key, **setup), init,
                train_kwargs, key_added, overwrite, counts=True)
