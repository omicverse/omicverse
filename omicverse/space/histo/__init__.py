r"""H&E → spatial transcriptomics prediction — ``ov.space.histo``.

Predicts spatial gene expression directly from H&E histology images. Wraps
state-of-the-art generative foundation models (STPath, STFlow), pathology
foundation-model + linear-head pipelines (HEST-FM), and super-resolution
methods (iStar) behind one ``method=`` dispatcher.

Data contract
-------------
The canonical container is :class:`wsidata.WSIData` (a
:class:`spatialdata.SpatialData` subclass with WSI-aware accessors). Tiles
live in ``shapes['tiles']`` (GeoDataFrame); foundation-model embeddings and
predicted gene-expression matrices live in ``tables['{key}_tiles']``
(AnnData). The same downstream tools (``ov.space.svg``, ``ov.pl.spatial``,
``zs.pl.tiles``) can therefore consume both real Visium tables and HE→ST
predictions without changes.

Pipeline
--------
::

    import omicverse as ov

    # 1. Open a WSI and create tiles (delegates to LazySlide).
    wsi = ov.space.histo.open_wsi('slide.svs')
    ov.space.histo.tile(wsi, tile_px=224, mpp=0.5)
    ov.space.histo.embed(wsi, model='gigapath')

    # 2. Predict expression — zero-shot generative foundation model.
    ov.space.histo.predict_expression(
        wsi, method='stpath',
        organ='Breast', tech='Visium',
        genes=['EPCAM', 'CDH1', 'KRT8'],
    )

    # 3. Predict expression — FM features + ridge head trained on a
    #    paired Visium reference.
    ov.space.histo.predict_expression(
        wsi, method='hest_fm',
        reference=ref_adata, fm_backbone='ctranspath',
    )

    # 4. Super-resolve a Visium adata using the paired H&E.
    ov.space.histo.super_resolve(adata, wsi=wsi, method='istar')

    # 5. Visualise — reuse LazySlide / omicverse plotting.
    import lazyslide as zs
    zs.pl.tiles(wsi, feature_key='stpath', color=['EPCAM', 'CDH1'])

Backends
--------
``stpath``     Generative foundation model (npj Digital Medicine 2025),
               37k+ genes, 17 organs, zero-shot via HuggingFace
               ``tlhuang/STPath``. Default.
``stflow``     Whole-slide flow matching (ICML 2025 Spotlight).
``istar``      HIPT + per-slide self-supervised super-resolution
               (Nature Biotechnology 2024). Vendored under
               ``omicverse.external.istar``.
``hest_fm``    Pathology foundation-model embedding + Ridge / MLP head
               fitted on the user's reference Visium slide. The
               default-and-recommended path when no zero-shot vocabulary
               match is available; works with as little as one paired
               reference slide.
Install
-------
``pip install 'omicverse[histo]'`` pulls ``lazyslide``, ``wsidata``,
``spatialdata``, ``tiffslide``/``openslide``, ``timm``, ``huggingface_hub``.
iStar remains an experimental backend with additional runtime requirements;
there is currently no separate ``histo-istar`` extra.
"""
from __future__ import annotations

from ..._optional import bind_optional_symbols

_DEPS_CORE = ("torch", "wsidata", "spatialdata", "lazyslide")
_HF_DEPS = _DEPS_CORE + ("huggingface_hub", "timm")
_INSTALL_HINT = "Install with `pip install 'omicverse[histo]'`."


bind_optional_symbols(
    globals(),
    "._io",
    ["open_wsi", "tile", "read_visium_with_image"],
    package=__name__,
    feature="omicverse.space.histo IO",
    dependencies=_DEPS_CORE,
    install_hint=_INSTALL_HINT,
)

bind_optional_symbols(
    globals(),
    "._demo",
    ["download_breast", "load_breast"],
    package=__name__,
    feature="omicverse.space.histo demo",
    dependencies=_DEPS_CORE,
    install_hint=_INSTALL_HINT,
)

bind_optional_symbols(
    globals(),
    "._embed",
    ["embed", "available_backbones"],
    package=__name__,
    feature="omicverse.space.histo.embed",
    dependencies=_HF_DEPS,
    install_hint=_INSTALL_HINT,
)

bind_optional_symbols(
    globals(),
    "._dispatch",
    ["predict_expression", "super_resolve"],
    package=__name__,
    feature="omicverse.space.histo prediction",
    dependencies=_HF_DEPS,
    install_hint=_INSTALL_HINT,
)

bind_optional_symbols(
    globals(),
    "._hest_fm",
    ["spot_features"],
    package=__name__,
    feature="omicverse.space.histo.spot_features",
    dependencies=_HF_DEPS,
    install_hint=_INSTALL_HINT,
)


__all__ = [
    # IO
    "open_wsi",
    "tile",
    "read_visium_with_image",
    # Demo data
    "download_breast",
    "load_breast",
    # Embedding
    "embed",
    "available_backbones",
    # Prediction
    "predict_expression",
    "super_resolve",
    "spot_features",
]
