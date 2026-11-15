r"""HEST-style baseline: pathology FM features → Ridge / MLP head.

Mirrors the HEST-Bench baseline that was shown to match or beat several
end-to-end neural architectures (Jaume et al., NeurIPS 2024 Spotlight):

1. Extract per-tile foundation-model embeddings on the reference Visium
   slide and on the query WSI (same backbone).
2. Project to a low-dimensional space (PCA).
3. Fit one Ridge regression per gene on the reference spots.
4. Predict on the query tiles.

Works with as little as one paired reference slide because a single Visium
sample provides 3,000–5,000 (spot, expression) training pairs — enough to
fit a stable linear head.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from ..._registry import register_function

if TYPE_CHECKING:
    from anndata import AnnData
    from wsidata import WSIData


@register_function(
    aliases=["spot特征", "spot_features", "提取spot特征", "Visium_spot_embed"],
    category="space",
    description="Embed every Visium spot's H&E patch with a pathology FM. Cuts spot-diameter patches on the WSI at each spot centroid and runs the chosen backbone, returning an AnnData of (n_spots × feature_dim). Result cached to $OV_HISTO_CACHE/ref_features/.",
    examples=[
        "ref_emb = ov.space.histo.spot_features(adata, wsi, fm_backbone='ctranspath')",
        "ref_emb = ov.space.histo.spot_features(adata, wsi, fm_backbone='gigapath', hf_token='hf_...')",
    ],
    related=["space.histo.embed", "space.histo.predict_expression"],
)
def spot_features(
    reference: "AnnData",
    wsi: "WSIData",
    *,
    fm_backbone: str = "ctranspath",
    fm_weight_path: str | None = None,
    hf_token: str | None = None,
    device: str | None = None,
    tile_key: str = "tiles",
) -> "AnnData":
    """Embed every Visium spot's H&E patch with a pathology FM.

    Returns an :class:`anndata.AnnData` whose rows are Visium spots
    (same order as ``reference.obs``) and columns are the pathology
    FM's feature dimensions. Mostly a thin public re-export of the
    internal helper used by HEST-FM / STFlow so notebooks can build
    custom held-out evaluations, k-fold CV, etc., without touching
    private internals.

    The result is cached on disk under
    ``$OV_HISTO_CACHE/ref_features/{backbone}_{slide}_n{n_spots}.h5ad``
    so subsequent calls (same slide + spot count + backbone) return
    instantly.
    """
    return _spot_features_from_reference(
        reference, wsi,
        fm_backbone=fm_backbone, tile_key=tile_key,
        device=device, feature_key=None,
        fm_weight_path=fm_weight_path, token=hf_token,
    )


def _ensure_features(
    wsi: "WSIData",
    *,
    feature_key: str | None,
    fm_backbone: str,
    tile_key: str,
    device: str | None,
    fm_weight_path: str | None = None,
    token: str | None = None,
):
    table_key = f"{feature_key or fm_backbone}_{tile_key}"
    if table_key in wsi.tables:
        return wsi.tables[table_key], (feature_key or fm_backbone)
    from ._embed import embed
    embed(wsi, model=fm_backbone, tile_key=tile_key, key_added=fm_backbone,
          model_path=fm_weight_path, token=token, device=device)
    return wsi.tables[f"{fm_backbone}_{tile_key}"], fm_backbone


def _spot_features_from_reference(
    reference: "AnnData",
    wsi: "WSIData",
    *,
    fm_backbone: str,
    tile_key: str,
    device: str | None,
    feature_key: str | None,
    fm_weight_path: str | None = None,
    token: str | None = None,
):
    """Tile the WSI on the reference spot centroids, then embed.

    Stores a one-shot tile collection under ``tile_key + '_ref_spots'`` so
    the original ``tiles`` grid (used for prediction) stays untouched.
    Caches the resulting embedding on disk under
    ``$OV_HISTO_CACHE/ref_features/{backbone}_{slide_stem}.h5ad`` so repeat
    calls (notebook re-runs, parameter sweeps) avoid the heavy WSI patch
    extraction.
    """
    import lazyslide as zs
    import geopandas as gpd
    from shapely.geometry import Point
    import os
    from pathlib import Path
    import anndata as ad

    if "spatial" not in reference.obsm:
        raise KeyError("reference.obsm['spatial'] missing — must be Visium.")

    # On-disk cache key — based on the slide file path + reference shape
    # + backbone name. Skips the heavy patch-extraction loop on re-runs.
    cache_dir = Path(os.environ.get("OV_HISTO_CACHE",
                                    Path.home() / ".cache" / "omicverse" / "histo")) / "ref_features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    slide_stem = Path(getattr(wsi.reader, "file", "wsi")).stem
    cache_path = cache_dir / f"{fm_backbone}_{slide_stem}_n{reference.n_obs}.h5ad"
    if cache_path.exists():
        return ad.read_h5ad(cache_path)

    library_id = next(iter(reference.uns["spatial"]))
    sf = reference.uns["spatial"][library_id]["scalefactors"]
    spot_diameter = float(sf.get("spot_diameter_fullres", 100.0))
    mpp = wsi.properties.mpp or 0.5
    tile_px = max(64, int(round(spot_diameter)))

    coords = np.asarray(reference.obsm["spatial"], dtype=float)
    spot_key = f"{tile_key}__ref_spots"
    # Build a tile grid by registering point-centred boxes directly.
    from shapely.geometry import box
    from spatialdata.models import ShapesModel
    r = tile_px / 2.0
    polys = [box(x - r, y - r, x + r, y + r) for x, y in coords]
    spot_tiles = gpd.GeoDataFrame(
        {
            "tile_id": np.arange(len(coords)),
            "tissue_id": 0,
            "x": coords[:, 0],
            "y": coords[:, 1],
        },
        geometry=polys,
    )
    # SpatialData requires the GeoDataFrame to carry a default `transform`
    # entry; ShapesModel.parse adds the identity transform in-place.
    spot_tiles = ShapesModel.parse(spot_tiles)
    wsi.shapes[spot_key] = spot_tiles
    # Register a TileSpec matching this synthetic tile grid so
    # feature_extraction can iterate.
    from wsidata import TileSpec
    spec = TileSpec(
        height=tile_px, width=tile_px,
        stride_height=tile_px, stride_width=tile_px,
        mpp=float(mpp), ops_level=0, base_level=0,
        tissue_name="ref_spots",
    ).to_dict()
    wsi.attrs.setdefault("tile_spec", {})[spot_key] = spec

    from ._embed import embed
    ref_key = f"{fm_backbone}__ref"
    embed(wsi, model=fm_backbone, tile_key=spot_key, key_added=ref_key,
          model_path=fm_weight_path, token=token, device=device)
    table = wsi.tables[f"{ref_key}_{spot_key}"]
    try:
        table.write_h5ad(cache_path)
    except Exception:
        pass
    return table


def predict_hest_fm(
    wsi: "WSIData",
    *,
    reference: "AnnData",
    tile_key: str = "tiles",
    key_added: str | None = None,
    genes: Sequence[str] | None = None,
    feature_key: str | None = None,
    fm_backbone: str = "ctranspath",
    fm_weight_path: str | None = None,
    hf_token: str | None = None,
    n_components: int | None = 256,
    alpha: float = 1.0,
    head: str = "ridge",
    device: str | None = None,
) -> "AnnData":
    """Train a Ridge head on the reference, predict on the query WSI.

    The reference :class:`AnnData` must carry spot coordinates in
    ``obsm['spatial']`` *and* refer to the same physical H&E that ``wsi``
    wraps — i.e. ``wsi`` is the full-resolution slide associated with the
    Visium output of ``reference``.

    Parameters
    ----------
    head
        ``'ridge'`` (default, fast) or ``'mlp'`` (2-layer MLP, slower but
        often a bit more accurate on small panels).
    """
    import anndata as ad
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge

    ref_emb = _spot_features_from_reference(
        reference, wsi,
        fm_backbone=fm_backbone, tile_key=tile_key,
        device=device, feature_key=feature_key,
        fm_weight_path=fm_weight_path, token=hf_token,
    )
    query_emb, eff_key = _ensure_features(
        wsi, feature_key=feature_key, fm_backbone=fm_backbone,
        tile_key=tile_key, device=device,
        fm_weight_path=fm_weight_path, token=hf_token,
    )

    X_ref = np.asarray(ref_emb.X, dtype=np.float32)
    X_q = np.asarray(query_emb.X, dtype=np.float32)

    if n_components is not None and n_components < X_ref.shape[1]:
        pca = PCA(n_components=n_components, random_state=0).fit(X_ref)
        X_ref = pca.transform(X_ref).astype(np.float32)
        X_q = pca.transform(X_q).astype(np.float32)

    if genes is None:
        gene_panel = list(reference.var_names)
    else:
        gene_panel = [g for g in genes if g in reference.var_names]
        if not gene_panel:
            raise ValueError("None of the requested genes are in reference.var_names.")

    Y_ref = reference[:, gene_panel].X
    if hasattr(Y_ref, "toarray"):
        Y_ref = Y_ref.toarray()
    Y_ref = np.log1p(Y_ref.astype(np.float32))

    if head == "ridge":
        model = Ridge(alpha=alpha)
        model.fit(X_ref, Y_ref)
        Y_pred = model.predict(X_q).astype(np.float32)
    elif head == "mlp":
        Y_pred = _fit_mlp(X_ref, Y_ref, X_q, device=device)
    else:
        raise ValueError(f"head={head!r} not in {{'ridge','mlp'}}.")

    pred = ad.AnnData(
        X=Y_pred,
        obs=query_emb.obs.copy(),
        var=reference.var.loc[gene_panel].copy(),
    )
    pred.uns["histo"] = {
        "method": "hest_fm",
        "fm_backbone": eff_key,
        "n_components": n_components,
        "alpha": alpha,
        "head": head,
    }
    if "spatial" in query_emb.obsm:
        pred.obsm["spatial"] = query_emb.obsm["spatial"].copy()
    else:
        # Derive pixel centroids from the tile GeoDataFrame so downstream
        # spatial plotters (scanpy, ov.pl, zs.pl) work without further
        # bookkeeping.
        tiles_gdf = wsi.shapes[tile_key]
        pred.obsm["spatial"] = np.stack(
            [tiles_gdf.geometry.centroid.x.to_numpy(),
             tiles_gdf.geometry.centroid.y.to_numpy()],
            axis=1,
        ).astype(np.float32)

    out_key = f"{key_added or 'hest_fm'}_{tile_key}"
    wsi.tables[out_key] = pred
    return pred


def _fit_mlp(X_ref, Y_ref, X_q, *, device=None, hidden: int = 512, epochs: int = 100):
    import torch
    import torch.nn as nn

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xtr = torch.from_numpy(X_ref).to(device)
    Ytr = torch.from_numpy(Y_ref).to(device)
    Xq = torch.from_numpy(X_q).to(device)
    net = nn.Sequential(
        nn.Linear(X_ref.shape[1], hidden), nn.GELU(),
        nn.Linear(hidden, Y_ref.shape[1]),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(Xtr), Ytr)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return net(Xq).cpu().numpy().astype("float32")
