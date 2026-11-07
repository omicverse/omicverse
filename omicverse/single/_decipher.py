"""Optional Decipher adapter."""
from __future__ import annotations

from .._optional import import_optional_module
from .._registry import register_function
from .._backend_validation import expression, obs_column, kwargs_copy, output_key, write_embedding


from ._scvi_training import _fit


@register_function(aliases=["Decipher"], category="single",
                   description="Train Decipher and return its top-level latent representation")
def decipher(adata, *, layer="counts", key_added="X_decipher", setup_kwargs=None,
             model_kwargs=None, train_kwargs=None, overwrite=False):
    """Train Decipher on counts and store the top-level ``v`` embedding.

    Returns the native model; intermediate ``z`` representations and trajectory
    analysis remain available through the backend. An embedding alone is not
    an inferred developmental direction or pseudotime.
    """
    setup = kwargs_copy(setup_kwargs, reserved=("adata", "layer"))
    return _fit(adata, "Decipher", layer, setup, model_kwargs, train_kwargs,
                key_added, overwrite, counts=True, latent_kwargs={"give_z": False})
