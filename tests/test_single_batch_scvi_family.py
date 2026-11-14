"""Smoke + kwarg-routing tests for the scvi-tools-family branches of
``ov.single.batch_correction``: SCVI, scANVI, totalVI, scPoli.

Each test uses a tiny synthetic AnnData and trains for a single epoch
(or a small ``max_epochs`` value) — enough to exercise the full code path
(setup_anndata → kwarg split → model construction → train → latent
write-back) without becoming a slow integration test. The scvi tests gate
on ``pytest.importorskip("torch")  # ov.vi (no scvi-tools)``; the scPoli test additionally gates on
``pytest.importorskip("scarches")``.

The intent is to lock the kwarg-routing contract (named init params land
in ``__init__``, named train params land in ``.train``) against future
scvi-tools / scArches API drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import anndata as ad


def _toy_adata_with_counts(n_per_batch=80, n_genes=120, n_batches=3, seed=0):
    """Tiny synthetic counts AnnData with a per-batch additive shift and a
    binary cell-type covariate. Counts go in ``adata.layers['counts']`` (raw)
    and ``adata.X`` (also raw — scvi-family methods read from `layer='counts'`)."""
    rng = np.random.default_rng(seed)
    Xs, batches, cell_types = [], [], []
    for bi in range(n_batches):
        X = rng.poisson(lam=2.0, size=(n_per_batch, n_genes)).astype(np.float32)
        # Per-batch overdispersion on the first 20 genes
        X[:, :20] += rng.poisson(lam=bi + 1, size=(n_per_batch, 20))
        # Cell-type A on first half, B on second half — adds counts on
        # genes 20:40 vs 40:60
        X[: n_per_batch // 2, 20:40] += 3
        X[n_per_batch // 2:, 40:60] += 3
        Xs.append(X)
        batches.extend([f"batch_{bi}"] * n_per_batch)
        cell_types.extend(["A"] * (n_per_batch // 2) + ["B"] * (n_per_batch - n_per_batch // 2))
    X_all = np.vstack(Xs)
    obs = pd.DataFrame({
        "batch": batches, "celltype": cell_types,
    }, index=[f"cell_{i}" for i in range(X_all.shape[0])])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = ad.AnnData(X=X_all, obs=obs, var=var)
    adata.layers["counts"] = X_all.copy()
    return adata


# ── scVI ─────────────────────────────────────────────────────────────────────

class TestSCVIRouting:
    def test_scvi_kwargs_route_to_init_and_train(self) -> None:
        pytest.importorskip("torch")  # ov.vi (no scvi-tools)
        from omicverse.single import batch_correction

        adata = _toy_adata_with_counts()
        model = batch_correction(
            adata, batch_key="batch", methods="scVI",
            # Init-side architecture params:
            n_hidden=32, n_latent=8, dropout_rate=0.05,
            # Train-side optimisation params:
            max_epochs=1, batch_size=64, accelerator="cpu",
        )
        # Init kwargs took effect (not silently swallowed).
        assert model.module.n_latent == 8
        # Latent embedding written back.
        assert "X_scVI" in adata.obsm
        assert adata.obsm["X_scVI"].shape == (adata.n_obs, 8)


# ── scANVI ───────────────────────────────────────────────────────────────────

class TestSCANVI:
    def test_scanvi_requires_labels_key(self) -> None:
        pytest.importorskip("torch")  # ov.vi (no scvi-tools)
        from omicverse.single import batch_correction

        adata = _toy_adata_with_counts()
        with pytest.raises(ValueError, match="labels_key"):
            batch_correction(adata, batch_key="batch", methods="scANVI")

    def test_scanvi_runs_and_writes_latent(self) -> None:
        pytest.importorskip("torch")  # ov.vi (no scvi-tools)
        from omicverse.single import batch_correction

        adata = _toy_adata_with_counts()
        # Mark a fraction of cells as "Unknown" so scANVI has a target.
        adata.obs["celltype"] = adata.obs["celltype"].astype(object)
        adata.obs.loc[adata.obs.index[::5], "celltype"] = "Unknown"
        adata.obs["celltype"] = adata.obs["celltype"].astype("category")

        model = batch_correction(
            adata, batch_key="batch", methods="scANVI",
            labels_key="celltype", unlabeled_category="Unknown",
            n_latent=8,            # init-side
            max_epochs=1,          # train-side
            accelerator="cpu",     # train-side
        )
        assert model.module.n_latent == 8
        assert "X_scANVI" in adata.obsm
        assert adata.obsm["X_scANVI"].shape == (adata.n_obs, 8)


# ── totalVI ──────────────────────────────────────────────────────────────────

class TestTotalVI:
    def test_totalvi_requires_protein_obsm_key(self) -> None:
        pytest.importorskip("torch")  # ov.vi (no scvi-tools)
        from omicverse.single import batch_correction

        adata = _toy_adata_with_counts()
        with pytest.raises(ValueError, match="protein_expression_obsm_key"):
            batch_correction(adata, batch_key="batch", methods="totalVI")

    def test_totalvi_runs_with_fake_protein_matrix(self) -> None:
        pytest.importorskip("torch")  # ov.vi (no scvi-tools)
        from omicverse.single import batch_correction

        adata = _toy_adata_with_counts()
        # Fake CITE-seq ADT counts: 15 proteins per cell.
        rng = np.random.default_rng(1)
        protein_counts = rng.poisson(lam=4.0, size=(adata.n_obs, 15)).astype(np.float32)
        adata.obsm["protein_counts"] = protein_counts

        model = batch_correction(
            adata, batch_key="batch", methods="totalVI",
            protein_expression_obsm_key="protein_counts",
            n_latent=8,            # init-side
            max_epochs=1,          # train-side
            accelerator="cpu",     # train-side
            batch_size=64,         # train-side
        )
        assert model.module.n_latent == 8
        assert "X_totalVI" in adata.obsm
        assert adata.obsm["X_totalVI"].shape == (adata.n_obs, 8)


# ── scPoli ───────────────────────────────────────────────────────────────────

class TestScPoli:
    def test_scpoli_runs_and_writes_latent(self) -> None:
        pytest.importorskip("scarches")
        from omicverse.single import batch_correction

        adata = _toy_adata_with_counts()
        model = batch_correction(
            adata, batch_key="batch", methods="scPoli",
            cell_type_keys="celltype",
            # Init-side scPoli params:
            embedding_dims=5,
            # Train-side scPoli.train params:
            n_epochs=2, pretraining_epochs=1,
        )
        assert "X_scPoli" in adata.obsm
        # scPoli's latent dimension is determined internally (latent_dim
        # default 10 unless overridden) — assert presence + shape contract.
        assert adata.obsm["X_scPoli"].shape[0] == adata.n_obs


# ── ov.vi specifics ─────────────────────────────────────────────────────────

class TestOvVIBackend:
    def test_scvi_uses_ov_vi_and_plan_kwargs(self) -> None:
        pytest.importorskip("torch")
        from omicverse.single import batch_correction
        from omicverse.vi import SCVI

        adata = _toy_adata_with_counts()
        model = batch_correction(adata, batch_key="batch", methods="scVI", n_latent=4,
                                 max_epochs=2, lr=2e-3, n_epochs_kl_warmup=10,
                                 accelerator="cpu", enable_progress_bar=False)
        assert isinstance(model, SCVI)
        assert model.module.train_config.lr == 2e-3
        assert model.history_["kl_weight"] == [0.0, 0.1]

    def test_scvi_on_anndataoom_is_native(self, tmp_path) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("anndataoom")
        import omicverse as ov
        from omicverse.single import batch_correction

        adata = _toy_adata_with_counts()
        path = tmp_path / "toy.h5ad"
        import scipy.sparse as sp
        ad.AnnData(X=sp.csr_matrix(adata.layers["counts"]), obs=adata.obs,
                   var=adata.var).write_h5ad(path)
        ov.settings.cpu_init()
        oom = ov.read(str(path), storage="disk")
        model = batch_correction(oom, batch_key="batch", methods="scVI", n_latent=4,
                                 max_epochs=2, accelerator="cpu")
        assert model._source().kind == "h5"  # streamed from disk, not materialised
        assert oom.obsm["X_scVI"].shape == (oom.n_obs, 4)
        oom.close()
