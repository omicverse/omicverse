"""Explicit adapters for scvi-tools' SysVI, DRVI and Decipher models."""
from __future__ import annotations

from .._optional import import_optional_module
from .._registry import register_function
from .._backend_validation import expression, obs_column, kwargs_copy, output_key, write_embedding


def _fit(adata, class_name, layer, setup, model_kwargs, train_kwargs, key_added, overwrite,
         *, counts, latent_kwargs=None):
    expression(adata, layer, counts=counts)
    output_key(adata, key_added, overwrite=overwrite)
    init = kwargs_copy(model_kwargs, reserved=("adata", "registry"))
    train = kwargs_copy(train_kwargs)
    backend = import_optional_module("scvi.external", feature=class_name, dependencies=("scvi",),
                                     install_hint=f"Install scvi-tools providing {class_name}.")
    if not hasattr(backend, class_name):
        raise ImportError(f"Installed scvi-tools does not expose {class_name}; use a supporting version.")
    cls = getattr(backend, class_name)
    cls.setup_anndata(adata, layer=layer, **setup)
    model = cls(adata, **init)
    model.train(**train)
    write_embedding(adata, key_added, model.get_latent_representation(**(latent_kwargs or {})))
    return model
