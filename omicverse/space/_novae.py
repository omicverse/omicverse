"""Spatial domain learning with Novae."""
from __future__ import annotations

from .._optional import import_optional_module
from .._registry import register_function
from .._backend_validation import (
    expression, coordinates, obs_column, kwargs_copy, output_key, write_embedding,
)


@register_function(aliases=["Novae", "spatial domain learning"], category="space",
                   description="Train Novae or run a supplied pretrained model for spatial domains")
def novae(adata, *, model=None, slide_key=None, key_added="X_novae",
          domains_key="novae_domains", neighbors_kwargs=None, model_kwargs=None,
          train_kwargs=None, representation_kwargs=None, domains_kwargs=None,
          overwrite=False):
    """Train a new Novae model or infer with a caller-provided pretrained model.

    Pass a loaded native Novae model for zero-shot inference; this wrapper never
    downloads a checkpoint implicitly. With ``model=None``, a new model is
    trained. Work is performed on a copy: expression and spatial graphs on the
    original are preserved. Embeddings and domain labels are copied back in
    cell order. Novae may leave isolated cells unassigned (missing domain label);
    these cells must not be treated as a new biological domain.
    The returned native model retains its working AnnData for downstream use.
    """
    expression(adata, None)
    coordinates(adata, "spatial")
    if slide_key is not None:
        obs_column(adata, slide_key)
    output_key(adata, key_added, overwrite=overwrite)
    if not isinstance(domains_key, str) or not domains_key:
        raise ValueError("domains_key must be a nonempty string.")
    if domains_key in adata.obs and not overwrite:
        raise ValueError(f"obs[{domains_key!r}] already exists; use overwrite=True or another key.")
    neighbors = kwargs_copy(neighbors_kwargs, reserved=("adata", "slide_key"))
    init = kwargs_copy(model_kwargs, reserved=("adata",))
    train = kwargs_copy(train_kwargs, reserved=("adata",))
    representation = kwargs_copy(representation_kwargs, reserved=("adata", "zero_shot"))
    domains = kwargs_copy(domains_kwargs, reserved=("adata", "key_added"))
    if model is not None and (init or train):
        raise ValueError("model_kwargs/train_kwargs are only used when training a new model.")
    backend = import_optional_module("novae", feature="Novae", dependencies=("novae",),
                                     install_hint="Install novae.")
    work = adata.copy()
    backend.spatial_neighbors(work, slide_key=slide_key, **neighbors)
    zero_shot = model is not None
    if model is None:
        model = backend.Novae(work, **init)
        model.fit(**train)
    model.compute_representations(work, zero_shot=zero_shot, **representation)
    source_key = model.assign_domains(work, **domains)
    if source_key not in work.obs or "novae_latent" not in work.obsm:
        raise RuntimeError("Novae did not return domain labels and representations.")
    if not work.obs_names.equals(adata.obs_names):
        raise RuntimeError("Novae changed cell order; results were not written.")
    write_embedding(adata, key_added, work.obsm["novae_latent"])
    adata.obs[domains_key] = work.obs[source_key].copy()
    return model
