"""Decipher (Nazaret et al., 2024) convenience adapter.

Trains ``ov.vi.Decipher`` -- the omicverse-native, pure-PyTorch rewrite of
``scvi.external.Decipher`` (no scvi-tools / Pyro) -- and stores its top-level,
interpretable 2-D latent ``v`` ("Decipher space") in ``adata.obsm``.  The
richer latent ``z``, importance-weighted marginal log-likelihood, imputed
gene expression / covariances and the trajectory utilities (``ov.vi.Trajectory``,
``ov.vi.rotate_decipher_components``, ``ov.vi.compute_decipher_time``) stay
available through the returned model.
"""
from __future__ import annotations

from .._registry import register_function


@register_function(
    aliases=["Decipher", "decipher", "Decipher space", "trajectory VAE", "轨迹VAE"],
    category="single",
    description="Train omicverse's native Decipher (two-level VAE with a 2-D interpretable "
                "latent v and a richer latent z) and store v in adata.obsm; useful for "
                "visualising normal vs. perturbed differentiation trajectories. Trains on "
                "in-memory AnnData, AnnDataOOM (disk) and Dask-backed AnnData.",
    produces={"obsm": ["X_decipher"]},
    examples=[
        "model = ov.single.decipher(adata, layer='counts')",
        "adata.obsm['X_decipher']  # (n_obs, 2) Decipher space",
        "model = ov.single.decipher(adata, layer='counts', key_added_z='X_decipher_z')",
    ],
    related=["vi.Decipher", "vi.Trajectory", "vi.rotate_decipher_components",
             "vi.compute_decipher_time"],
)
def decipher(adata, *, layer: str = "counts", key_added: str = "X_decipher",
             key_added_z: str | None = None, model_kwargs: dict | None = None,
             train_kwargs: dict | None = None, overwrite: bool = False):
    """Train Decipher and store its top-level latent representation.

    Parameters
    ----------
    adata
        AnnData, AnnDataOOM (``ov.read(path, storage='disk')``) or Dask-backed AnnData.
    layer
        Layer with raw counts (``None`` = ``adata.X``).
    key_added
        ``adata.obsm`` key for the 2-D Decipher space ``v``.
    key_added_z
        If given, also store the posterior mean of the intermediate latent
        ``z`` under this ``adata.obsm`` key.
    model_kwargs
        Extra keywords for :class:`ov.vi.Decipher` (``dim_v``, ``dim_z``,
        ``layers_v_to_z``, ``layers_z_to_x``, ``beta``, ``genes``, ``seed``, ...).
    train_kwargs
        Extra keywords for :meth:`ov.vi.Decipher.train` (``max_epochs``,
        ``batch_size``, ``lr``, ``early_stopping``, ...).
    overwrite
        Overwrite ``key_added`` / ``key_added_z`` if already present in ``adata.obsm``.

    Returns
    -------
    ov.vi.Decipher
        The trained model. An embedding alone is not an inferred developmental
        direction or pseudotime: build one with ``ov.vi.Trajectory.from_clusters``
        and ``ov.vi.compute_decipher_time``, after optionally orienting the space
        with ``ov.vi.rotate_decipher_components``.
    """
    if not overwrite:
        for k in (key_added, key_added_z):
            if k is not None and k in adata.obsm:
                raise KeyError(f"adata.obsm[{k!r}] already exists; pass overwrite=True")
    from .. import vi
    model = vi.Decipher(adata, layer=layer, **(model_kwargs or {}))
    model.train(**(train_kwargs or {}))
    adata.obsm[key_added] = model.get_latent_representation()
    if key_added_z is not None:
        adata.obsm[key_added_z] = model.get_latent_representation(give_z=True)
    return model


__all__ = ["decipher"]
