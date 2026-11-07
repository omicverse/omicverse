"""Sample-aware representation learning with scvi-tools MrVI."""
from __future__ import annotations

from .._optional import import_optional_module
from .._registry import register_function
from .._backend_validation import expression, obs_column, kwargs_copy, output_key, write_embedding


@register_function(aliases=["MrVI", "sample-aware embedding"], category="single",
                   description="Train MrVI on counts with biological sample identities")
def mrvi(adata, *, sample_key, layer="counts", batch_key=None, key_added="X_mrvi",
         setup_kwargs=None, model_kwargs=None, train_kwargs=None, overwrite=False):
    """Train scvi.external.MRVI and return its native model.

    Requires raw counts and at least two biological samples. Writes the
    sample-independent ``u`` representation to ``obsm[key_added]``. The returned
    model exposes sample distances and differential analyses; training an
    embedding alone does not perform a differential-expression test.
    ``setup_kwargs``, ``model_kwargs`` and ``train_kwargs`` go only to their
    corresponding backend stages. Backend registration may add obs/uns fields.
    """
    expression(adata, layer, counts=True)
    obs_column(adata, sample_key, minimum_categories=2)
    if batch_key is not None:
        obs_column(adata, batch_key)
    output_key(adata, key_added, overwrite=overwrite)
    setup = kwargs_copy(setup_kwargs, reserved=("adata", "sample_key", "layer", "batch_key"))
    model_args = kwargs_copy(model_kwargs, reserved=("adata",))
    train = kwargs_copy(train_kwargs)
    backend = import_optional_module("scvi.external", feature="MrVI", dependencies=("scvi",),
                                     install_hint="Install scvi-tools with MrVI support.")
    cls = backend.MRVI
    cls.setup_anndata(adata, sample_key=sample_key, layer=layer, batch_key=batch_key, **setup)
    model = cls(adata, **model_args)
    model.train(**train)
    write_embedding(adata, key_added, model.get_latent_representation(give_z=False))
    return model
