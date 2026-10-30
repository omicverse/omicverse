"""Optional sysVI adapter."""
from __future__ import annotations

from .._optional import import_optional_module
from .._registry import register_function
from .._backend_validation import expression, obs_column, kwargs_copy, output_key, write_embedding


from ._scvi_training import _fit


@register_function(aliases=["sysVI", "SysVI"], category="single",
                   description="Integrate systems using SysVI on normalized expression")
def sysvi(adata, *, batch_key, layer=None, key_added="X_sysvi", setup_kwargs=None,
          model_kwargs=None, train_kwargs=None, overwrite=False):
    """Train SysVI on normalized expression; return its native model.

    ``batch_key`` identifies the systems to integrate and must have at least two
    levels. Input must already be normalized (typically log-normalized and
    scaled); this adapter does not convert raw counts to SysVI input.
    """
    expression(adata, layer)
    obs_column(adata, batch_key, minimum_categories=2)
    setup = kwargs_copy(setup_kwargs, reserved=("adata", "layer", "batch_key"))
    return _fit(adata, "SysVI", layer, dict(batch_key=batch_key, **setup), model_kwargs,
                train_kwargs, key_added, overwrite, counts=False)
