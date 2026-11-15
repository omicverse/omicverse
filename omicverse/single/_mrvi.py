"""Sample-aware representation learning with omicverse's native MrVI (``ov.vi.MRVI``).

No scvi-tools dependency: trains on in-memory AnnData, AnnDataOOM (disk) and
Dask-backed AnnData.  This is a thin convenience wrapper (train + write the
sample-independent ``u`` embedding) around :class:`omicverse.vi.MRVI`, whose
constructor and methods (``get_local_sample_representation``,
``get_local_sample_distances``, ``differential_abundance``,
``differential_expression``...) expose the rest of MrVI's sample-level analyses.
"""
from __future__ import annotations

from .._registry import register_function


@register_function(
    aliases=["MrVI", "mrvi", "sample-aware embedding", "多样本层级变分推断"],
    category="single",
    description="Train omicverse-native MrVI on counts with biological sample "
                "identities (no scvi-tools); writes the sample-independent u "
                "representation and returns the trained ov.vi.MRVI model for "
                "sample-level downstream analyses (distances, differential "
                "abundance, differential expression).",
    examples=[
        "model = ov.single.mrvi(adata, sample_key='donor', layer='counts')",
        "model = ov.single.mrvi(adata, sample_key='donor', batch_key='batch',"
        " train_kwargs={'max_epochs': 100})",
    ],
    related=["vi.MRVI", "single.batch_correction"],
)
def mrvi(adata, *, sample_key, layer="counts", batch_key=None, labels_key=None,
         key_added="X_mrvi", model_kwargs=None, train_kwargs=None, overwrite=False):
    """Train :class:`omicverse.vi.MRVI` and return it.

    Requires raw counts and at least two biological samples. Writes the
    sample-independent ``u`` representation to ``obsm[key_added]``. The returned
    model exposes sample distances and differential analyses; training an
    embedding alone does not perform a differential-expression test.
    ``model_kwargs`` go to the :class:`~omicverse.vi.MRVI` constructor (e.g.
    ``n_latent``, ``n_latent_u``, ``u_prior_mixture_k``); ``train_kwargs`` to
    :meth:`~omicverse.vi.MRVI.train` (e.g. ``max_epochs``, ``batch_size``).

    Arguments:
        adata: AnnData, AnnDataOOM (``ov.read(path, storage='disk')``) or
            Dask-backed AnnData.
        sample_key: ``obs`` column with each cell's biological sample identity.
        layer: Layer with raw counts (None = ``X``).
        batch_key: ``obs`` column with a technical batch (None = one batch).
        labels_key: ``obs`` column with labels (informs the ``u`` mixture prior).
        key_added: ``obsm`` key for the ``u`` embedding.
        overwrite: Allow overwriting an existing ``obsm[key_added]``.

    Example::

        model = ov.single.mrvi(adata, sample_key="donor", layer="counts",
                               train_kwargs={"max_epochs": 100})
        adata.obsm["X_mrvi"]  # already written
        da = model.differential_abundance(sample_cov_keys=["disease"])
    """
    if not overwrite and key_added in adata.obsm:
        raise ValueError(f"adata.obsm[{key_added!r}] already exists; pass overwrite=True")
    from ..vi import MRVI
    model = MRVI(adata, sample_key=sample_key, layer=layer, batch_key=batch_key,
                labels_key=labels_key, **(model_kwargs or {}))
    model.train(**(train_kwargs or {}))
    adata.obsm[key_added] = model.get_latent_representation()
    return model


__all__ = ["mrvi"]
