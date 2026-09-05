r"""``method=`` dispatchers for HE→ST prediction and super-resolution."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

from ..._registry import register_function

if TYPE_CHECKING:
    from anndata import AnnData
    from wsidata import WSIData


PredictMethod = Literal["stpath", "stflow", "hest_fm"]
SuperResMethod = Literal["istar"]


@register_function(
    aliases=["HE预测空转", "predict_expression", "HE2ST", "H&E预测基因", "histo_predict"],
    category="space",
    description="Predict spot-level spatial gene expression from a tiled WSI using a HE→ST backend (method='stpath' / 'stflow' / 'hest_fm'). Output AnnData is stored as wsi.tables[{key_added or method}_{tile_key}] with tile-pixel centroids in obsm['spatial'].",
    examples=[
        "ov.space.histo.predict_expression(wsi, method='stpath', organ='Breast', tech='Visium')",
        "ov.space.histo.predict_expression(wsi, method='hest_fm', reference=adata, fm_backbone='ctranspath')",
        "ov.space.histo.predict_expression(wsi, method='stflow', reference=adata, n_epochs=80)",
    ],
    related=["space.histo.super_resolve", "space.histo.embed", "space.histo.spot_features", "space.histo.tile"],
)
def predict_expression(
    wsi: "WSIData",
    *,
    method: PredictMethod = "stpath",
    tile_key: str = "tiles",
    key_added: str | None = None,
    genes: Sequence[str] | None = None,
    organ: str | None = None,
    tech: str | None = "Visium",
    reference: "AnnData | None" = None,
    feature_key: str | None = None,
    fm_backbone: str | None = None,
    weight_path: str | None = None,
    fm_weight_path: str | None = None,
    cache_dir: str | None = None,
    hf_token: str | None = None,
    device: str | None = None,
    **kwargs,
) -> "AnnData":
    """Predict spot-level spatial gene expression from a tiled WSI.

    Writes the predicted ``AnnData`` to
    ``wsi.tables['{key_added or method}_{tile_key}']`` and also returns it.
    The output AnnData has tile barcodes as ``obs_names``, gene symbols as
    ``var_names``, predicted log1p expression as ``X``, and tile pixel
    centroids in ``obsm['spatial']``.

    Parameters
    ----------
    method
        Prediction backend (see :mod:`ov.space.histo`).
    genes
        Gene symbols to retain. ``None`` returns the model's full vocabulary
        (37k for STPath/STFlow) or all reference genes (HEST-FM).
    organ, tech
        Hint tokens used by STPath/STFlow. Examples: ``organ='Breast'``,
        ``tech='Visium'``. Ignored by HEST-FM.
    reference
        Paired Visium :class:`AnnData` used to fit a per-slide head
        (HEST-FM). Not required by STPath/STFlow zero-shot.
    feature_key
        Name of the tile-level feature table in ``wsi.tables``. Defaults to
        ``'gigapath'`` for STPath/STFlow and ``fm_backbone`` for HEST-FM.
    fm_backbone
        Pathology FM used to extract patch features when ``feature_key`` is
        not already present. Defaults to ``'gigapath'`` (STPath/STFlow) or
        ``'ctranspath'`` (HEST-FM).
    weight_path
        Path to an already-downloaded **predictor** checkpoint, bypassing the
        HuggingFace download. STPath: a local ``stfm.pth`` file. STFlow:
        ignored (trained per-slide). HEST-FM: ignored (linear head fit per
        slide).
    fm_weight_path
        Path to an already-downloaded **patch-encoder** checkpoint
        (CTransPath / GigaPath / UNI / …). Forwarded to
        :func:`lazyslide.tl.feature_extraction` as ``model_path``. Use this
        when the slide-host doesn't have HuggingFace network access or when
        you've pre-staged the encoder weights.
    cache_dir
        Override the default ``$OV_HISTO_CACHE``
        (``~/.cache/omicverse/histo``) where this module stores
        auto-cloned repos (STPath, STFlow), downloaded predictor
        weights, and on-disk feature caches.
    hf_token
        Explicit HuggingFace access token. Falls back to
        ``$HUGGING_FACE_HUB_TOKEN`` then to
        ``~/.cache/huggingface/token``.
    """
    if method == "stpath":
        from ._stpath import predict_stpath
        return predict_stpath(
            wsi,
            tile_key=tile_key,
            key_added=key_added,
            genes=genes,
            organ=organ,
            tech=tech,
            feature_key=feature_key,
            fm_backbone=fm_backbone or "gigapath",
            weight_path=weight_path,
            fm_weight_path=fm_weight_path,
            cache_dir=cache_dir,
            hf_token=hf_token,
            device=device,
            **kwargs,
        )
    if method == "stflow":
        from ._stflow import predict_stflow
        return predict_stflow(
            wsi,
            tile_key=tile_key,
            key_added=key_added,
            genes=genes,
            organ=organ,
            tech=tech,
            reference=reference,
            feature_key=feature_key,
            fm_backbone=fm_backbone or "gigapath",
            fm_weight_path=fm_weight_path,
            cache_dir=cache_dir,
            hf_token=hf_token,
            device=device,
            **kwargs,
        )
    if method == "hest_fm":
        from ._hest_fm import predict_hest_fm
        if reference is None:
            raise ValueError(
                "method='hest_fm' requires `reference=` (a paired Visium AnnData "
                "with H&E in the same physical frame)."
            )
        return predict_hest_fm(
            wsi,
            reference=reference,
            tile_key=tile_key,
            key_added=key_added,
            genes=genes,
            feature_key=feature_key,
            fm_backbone=fm_backbone or "ctranspath",
            fm_weight_path=fm_weight_path,
            hf_token=hf_token,
            device=device,
            **kwargs,
        )
    raise ValueError(
        f"Unknown method={method!r}. Pick one of: stpath, stflow, hest_fm."
    )


@register_function(
    aliases=["空转超分", "super_resolve", "Visium超分", "iStar超分"],
    category="space",
    description="Super-resolve a paired (Visium, H&E) sample to near-single-cell sub-spot tiles via iStar. Trains a per-slide HIPT regression head on the paired Visium counts (NOT a zero-shot model — needs paired ST). For H&E-only prediction use predict_expression(method='stpath') instead.",
    examples=[
        "pred = ov.space.histo.super_resolve(adata, wsi=wsi, method='istar')",
        "pred = ov.space.histo.super_resolve(adata, wsi=wsi, method='istar', pixel_size=0.5, epochs=80, n_top_genes=50)",
    ],
    related=["space.histo.predict_expression", "space.histo.read_visium_with_image"],
)
def super_resolve(
    adata: "AnnData",
    *,
    wsi: "WSIData | None" = None,
    he_image: str | None = None,
    method: SuperResMethod = "istar",
    factor: int = 8,
    genes: Sequence[str] | None = None,
    device: str | None = None,
    cache_dir: str | None = None,
    hipt256_path: str | None = None,
    hipt4k_path: str | None = None,
    **kwargs,
) -> "AnnData":
    """Super-resolve a paired (Visium, H&E) sample to near-single-cell tiles.

    .. note::
       iStar is **not** an H&E-only model. The per-slide regression head
       is trained on the paired Visium counts you pass as ``adata``;
       super-resolution then extrapolates that fit to sub-spot pixels on
       the same slide. For H&E-only prediction (no Visium reference)
       reach for :func:`predict_expression` with ``method='stpath'``.

    Parameters
    ----------
    adata
        Visium :class:`AnnData` carrying spot counts and
        ``obsm['spatial']`` — the **required** paired reference iStar
        trains its head on.
    wsi
        Optional :class:`wsidata.WSIData` wrapping the source H&E. If absent,
        ``he_image`` must point to the slide and the wrapper opens it.
    he_image
        Path to the full-resolution H&E slide.
    method
        Currently only ``'istar'`` is supported (Nature Biotechnology 2024).
    factor
        Super-resolution factor; ``8`` gives ~8 µm sub-spot tiles for
        Visium (55 µm spots).
    cache_dir
        Override the default ``$OV_HISTO_CACHE``
        (``~/.cache/omicverse/histo``) where HIPT checkpoints and per-
        slide iStar working directories live.
    hipt256_path, hipt4k_path
        Paths to already-downloaded HIPT checkpoints, bypassing the
        mahmoodlab/HIPT LFS download. Use this when the host doesn't
        have GitHub network access or when you've pre-staged the
        weights.
    """
    if method == "istar":
        from ._istar import super_resolve_istar
        return super_resolve_istar(
            adata,
            wsi=wsi,
            he_image=he_image,
            factor=factor,
            genes=genes,
            device=device,
            cache_dir=cache_dir,
            hipt256_path=hipt256_path,
            hipt4k_path=hipt4k_path,
            **kwargs,
        )
    raise ValueError(f"Unknown super-resolution method={method!r}.")
