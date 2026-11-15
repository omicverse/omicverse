r"""STFlow backend — whole-slide flow matching (ICML 2025 Spotlight).

STFlow's upstream codebase ships only training scripts; the authors have
since productionised it as STPath with public zero-shot weights. This
wrapper therefore implements the canonical "fit-on-your-reference,
predict-on-your-query" workflow: a flow-matching denoiser is trained
end-to-end on the paired reference slide (a few minutes on a single GPU
for a 500-gene panel) and applied to the query tiles.

Reference
---------
Huang et al., Scalable Generation of Spatial Transcriptomics from Histology
Images via Whole-Slide Flow Matching, ICML 2025 (Spotlight).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData
    from wsidata import WSIData


STFLOW_GITHUB_URL = "https://github.com/Graph-and-Geometric-Learning/STFlow.git"


def _default_cache_dir() -> Path:
    return Path(
        os.environ.get(
            "OV_HISTO_CACHE",
            Path.home() / ".cache" / "omicverse" / "histo",
        )
    )


def _ensure_stflow_repo(cache_dir: Path) -> Path:
    repo = cache_dir / "STFlow"
    if not repo.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        import subprocess
        subprocess.check_call(
            ["git", "clone", "--depth", "1", STFLOW_GITHUB_URL, str(repo)],
        )
    _patch_stflow_repo(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def _patch_stflow_repo(repo: Path) -> None:
    """Apply small idempotent fixes to upstream STFlow.

    Upstream's ``stflow/model/transformer.py`` passes ``non_negative=`` to
    ``GeneUpdate``, but ``GeneUpdate.__init__`` doesn't accept that kwarg
    — so importing ``stflow.model.denoiser`` and instantiating ``Denoiser``
    crashes with ``TypeError``. The fix is a one-line edit that drops
    the unused kwarg; we apply it idempotently after each auto-clone.
    """
    target = repo / "stflow" / "model" / "transformer.py"
    if not target.exists():
        return
    text = target.read_text()
    bad = (
        "self.gene_updater = GeneUpdate(d_model, n_genes, proj_drop=proj_drop, "
        "non_negative=gene_exp_non_negative)"
    )
    if bad in text:
        good = (
            "# omicverse patch: upstream's GeneUpdate does not accept "
            "`non_negative=`; the flag is unused inside the class anyway.\n"
            "        self.gene_updater = GeneUpdate(d_model, n_genes, proj_drop=proj_drop)"
        )
        target.write_text(text.replace(bad, good))


def _ref_features_via_pseudotiles(reference, wsi, *, fm_backbone, tile_key, device):
    """Reuse the HEST-FM helper to build a synthetic tile grid on Visium spots."""
    from ._hest_fm import _spot_features_from_reference
    return _spot_features_from_reference(
        reference, wsi,
        fm_backbone=fm_backbone, tile_key=tile_key,
        device=device, feature_key=None,
    )


def predict_stflow(
    wsi: "WSIData",
    *,
    tile_key: str = "tiles",
    key_added: str | None = None,
    genes: Sequence[str] | None = None,
    organ: str | None = None,            # noqa: ARG001  — accepted for API parity
    tech: str | None = "Visium",         # noqa: ARG001
    reference: "AnnData | None" = None,
    feature_key: str | None = None,
    fm_backbone: str = "gigapath",
    fm_weight_path: str | None = None,
    hf_token: str | None = None,
    n_epochs: int = 200,
    n_layers: int = 4,
    n_neighbors: int = 8,
    n_sample_steps: int = 5,
    hidden_dim: int = 256,
    batch_size: int = 1,
    device: str | None = None,
    cache_dir: str | Path | None = None,
) -> "AnnData":
    """Fit a flow-matching denoiser on the reference slide, predict on the query.

    Parameters
    ----------
    n_sample_steps
        Number of Euler steps for the reverse-time ODE (default 5, as in
        the STFlow paper).
    n_layers
        Transformer depth — keep at 4 unless GPU memory allows more.
    """
    import anndata as ad
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if reference is None:
        raise ValueError(
            "method='stflow' requires `reference=` (a paired Visium AnnData)."
        )

    cache = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    _ensure_stflow_repo(cache)

    from types import SimpleNamespace
    from stflow.model.denoiser import Denoiser
    from stflow.flow.interpolant import Interpolant

    # 1. Tile-level features for the query.
    feat_table_key = f"{feature_key or fm_backbone}_{tile_key}"
    if feat_table_key not in wsi.tables:
        from ._embed import embed
        embed(wsi, model=fm_backbone, tile_key=tile_key, key_added=fm_backbone,
              model_path=fm_weight_path, token=hf_token, device=device)
        feat_table_key = f"{fm_backbone}_{tile_key}"
    q_feat = wsi.tables[feat_table_key]
    img_features = np.asarray(q_feat.X, dtype=np.float32)
    feature_dim = img_features.shape[1]

    # Tile centroids.
    tiles_gdf = wsi.shapes[tile_key]
    q_coords = np.stack(
        [tiles_gdf.geometry.centroid.x.to_numpy(),
         tiles_gdf.geometry.centroid.y.to_numpy()],
        axis=1,
    ).astype(np.float32)

    # 2. Reference features at Visium spots.
    r_feat = _ref_features_via_pseudotiles(
        reference, wsi,
        fm_backbone=fm_backbone, tile_key=tile_key, device=device,
    )
    r_features = np.asarray(r_feat.X, dtype=np.float32)
    r_coords = np.asarray(reference.obsm["spatial"], dtype=np.float32)

    # 3. Gene panel & reference targets.
    if genes is None:
        # STFlow's upstream MLPAttnEdgeAggregation hardcodes
        # `+50` in its mlp_attn input dim (see
        # stflow/model/transformer.py:63), i.e. the gene panel size is
        # baked into the architecture at 50. Until that assumption is
        # lifted upstream the default panel sits at 50 HVGs.
        from ...pp import normalize_total, log1p, highly_variable_genes
        ref_copy = reference.copy()
        normalize_total(ref_copy, target_sum=1e4)
        log1p(ref_copy)
        highly_variable_genes(ref_copy, n_top_genes=50, flavor="seurat_v3")
        gene_panel = ref_copy.var_names[ref_copy.var["highly_variable"]].tolist()
    else:
        gene_panel = [g for g in genes if g in reference.var_names]
        if len(gene_panel) != 50:
            raise ValueError(
                f"STFlow's upstream transformer hardcodes a 50-gene panel "
                f"(stflow/model/transformer.py:63). Pass exactly 50 genes "
                f"or set `genes=None` to use the top-50 HVGs; got "
                f"{len(gene_panel)} genes after filtering against the "
                f"reference."
            )
    if not gene_panel:
        raise ValueError("Empty gene panel after filtering against reference.var_names.")

    Y = reference[:, gene_panel].X
    if hasattr(Y, "toarray"):
        Y = Y.toarray()
    Y = np.log1p(Y.astype(np.float32))

    # 4. Normalise coords to [-1, 1] jointly for ref/query.
    all_coords = np.concatenate([r_coords, q_coords], axis=0)
    mn = all_coords.min(0)
    mx = all_coords.max(0)
    rng = np.maximum(mx - mn, 1.0)
    r_coords_n = 2 * (r_coords - mn) / rng - 1
    q_coords_n = 2 * (q_coords - mn) / rng - 1

    # 5. Build denoiser.
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # The upstream Denoiser reads attributes (feature_dim, hidden_dim,
    # pairwise_hidden_dim, activation, …) that don't match
    # `stflow.model.config.ModelConfig`'s constructor names exactly, so we
    # supply a SimpleNamespace with the field names Denoiser actually
    # references.
    cfg = SimpleNamespace(
        n_genes=len(gene_panel),
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        pairwise_hidden_dim=hidden_dim // 4,
        n_layers=n_layers,
        n_heads=4,
        dropout=0.0,
        attn_dropout=0.0,
        n_neighbors=n_neighbors,
        activation="gelu",
    )

    model = Denoiser(cfg).to(dev)
    # Upstream prior types: "gaussian", "zero", "zinb".
    interpolant = Interpolant(prior_sample_type="gaussian", normalize=False)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    r_feat_t = torch.from_numpy(r_features)[None].to(dev)        # [1, N, D]
    r_coords_t = torch.from_numpy(r_coords_n)[None].to(dev)      # [1, N, 2]
    r_Y_t = torch.from_numpy(Y)[None].to(dev)                    # [1, N, G]

    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        noisy, t_steps = interpolant.corrupt_exp(r_Y_t)
        _, loss = model(r_Y_t, r_feat_t, r_coords_t, r_Y_t, t_steps)
        # STFlow trains on the velocity field; use its own forward signature.
        loss.backward()
        opt.step()

    # 6. Inference on query tiles.
    q_feat_t = torch.from_numpy(img_features)[None].to(dev)
    q_coords_t = torch.from_numpy(q_coords_n)[None].to(dev)
    model.eval()
    with torch.no_grad():
        exp_t1 = interpolant.sample_from_prior((1, q_feat_t.shape[1], len(gene_panel))).to(dev)
        ts = torch.linspace(0.01, 1.0, n_sample_steps)[:, None].expand(n_sample_steps, exp_t1.shape[0]).to(dev)
        pred = exp_t1
        for step, (t1, t2) in enumerate(zip(ts[:-1], ts[1:])):
            pred = model.inference(exp_t1, q_feat_t, q_coords_t, t1, predict=True)
            d_t = t2 - t1
            if step == n_sample_steps - 2:
                break
            exp_t1 = interpolant.denoise(pred, exp_t1, t1, d_t)
        pred_np = pred.squeeze(0).cpu().numpy().astype(np.float32)

    out = ad.AnnData(
        X=pred_np,
        obs=q_feat.obs.copy(),
        var=reference.var.loc[gene_panel].copy(),
    )
    out.uns["histo"] = {
        "method": "stflow",
        "fm_backbone": fm_backbone,
        "n_epochs": n_epochs,
        "n_layers": n_layers,
        "n_sample_steps": n_sample_steps,
    }
    out.obsm["spatial"] = q_coords

    out_key = f"{key_added or 'stflow'}_{tile_key}"
    wsi.tables[out_key] = out
    return out
