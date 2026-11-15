"""``ov.single.sysvi`` — thin, contributor-proposed entry point for
:class:`omicverse.vi.SysVI` (train + write the latent representation).

omicverse-native re-implementation of PR #1013's ``scvi.external.SysVI``
adapter (no scvi-tools dependency); the model itself lives in
:mod:`omicverse.vi._sysvi`, reusing the ``ov.vi`` core so it also trains on
AnnDataOOM (disk) and Dask-backed AnnData.
"""
from __future__ import annotations

from .._registry import register_function


@register_function(
    aliases=["sysVI", "SysVI", "sysvi", "系统整合", "cross-system integration"],
    category="single",
    description="Integrate systems (species, protocol, modality...) with SysVI: a cVAE "
                "with a VampPrior and a latent cycle-consistency loss on normalized "
                "expression; writes the latent representation to adata.obsm.",
    prerequisites={"optional_functions": ["preprocess", "pp.normalize_total", "pp.log1p"]},
    requires={"obs": []},
    produces={"obsm": ["X_sysvi"]},
    examples=[
        "m = ov.single.sysvi(adata, batch_key='system')",
        "adata.obsm['X_sysvi']  # written in place",
        "# weaker covariate (e.g. donor/sample) conditions the model without cycle-correction",
        "m = ov.single.sysvi(adata, batch_key='system', categorical_covariate_keys=['donor'])",
    ],
    related=["vi.SysVI", "single.batch_correction"],
)
def sysvi(adata, *, batch_key: str, layer=None, categorical_covariate_keys=None,
          continuous_covariate_keys=None, key_added: str = "X_sysvi", model_kwargs=None,
          train_kwargs=None, overwrite: bool = False):
    """Train :class:`omicverse.vi.SysVI` and write its latent representation.

    Arguments:
        adata: AnnData, AnnDataOOM or Dask-backed AnnData.
        batch_key: ``obs`` column identifying the *systems* to integrate
            (species, protocol, modality...); needs >= 2 categories. Corrected
            via conditioning and a latent cycle-consistency loss.
        layer: Layer with normalized expression (``None`` = ``X``). Input
            must already be normalized (e.g. log1p); this function does not
            convert raw counts.
        categorical_covariate_keys / continuous_covariate_keys: Weaker
            covariates (e.g. donor/sample) used only to condition the cVAE,
            not cycle-corrected.
        key_added: ``obsm`` key for the latent representation.
        model_kwargs: Extra keywords for :class:`omicverse.vi.SysVI`
            (``n_latent``, ``prior``, ``n_prior_components``, ...).
        train_kwargs: Extra keywords for ``SysVI.train()`` (``max_epochs``,
            ``device``, ...).
        overwrite: Allow overwriting an existing ``adata.obsm[key_added]``.

    Returns:
        The trained :class:`omicverse.vi.SysVI` model.
    """
    if key_added in adata.obsm and not overwrite:
        raise ValueError(f"adata.obsm[{key_added!r}] already exists; pass overwrite=True")
    from .. import vi
    model = vi.SysVI(adata, layer=layer, batch_key=batch_key,
                     categorical_covariate_keys=categorical_covariate_keys,
                     continuous_covariate_keys=continuous_covariate_keys,
                     **(model_kwargs or {}))
    model.train(**(train_kwargs or {}))
    adata.obsm[key_added] = model.get_latent_representation()
    return model


__all__ = ["sysvi"]
