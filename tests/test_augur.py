"""Tests for ov.single.Augur — cell type prioritization via ML cross-validation."""

import numpy as np
import pandas as pd
import anndata as ad
import pytest

import omicverse as ov


class TestAugur:
    """Test suite for ov.single.Augur."""

    @pytest.fixture
    def sample_adata(self):
        """Synthetic AnnData: 2 conditions, 3 cell types, 200 genes."""
        rng = np.random.default_rng(42)
        n_cells = 600
        cell_types = rng.choice(["T_cell", "B_cell", "Macrophage"], n_cells)
        labels = rng.choice(["control", "treatment"], n_cells)
        X = rng.standard_normal((n_cells, 200))
        # Add signal: treatment T_cells have shifted expression
        treat_t = (labels == "treatment") & (cell_types == "T_cell")
        X[treat_t, :50] += 2.0
        obs = pd.DataFrame({"cell_type": cell_types, "label": labels})
        return ad.AnnData(X, obs=obs)

    @pytest.fixture
    def adata2(self):
        """Second synthetic AnnData for differential test."""
        rng = np.random.default_rng(99)
        n_cells = 600
        cell_types = rng.choice(["T_cell", "B_cell", "Macrophage"], n_cells)
        labels = rng.choice(["control", "stimulus"], n_cells)
        X = rng.standard_normal((n_cells, 200))
        treat_b = (labels == "stimulus") & (cell_types == "B_cell")
        X[treat_b, :50] += 1.5
        obs = pd.DataFrame({"cell_type": cell_types, "label": labels})
        return ad.AnnData(X, obs=obs)

    def test_import(self):
        """Augur class is accessible via ov.single."""
        assert hasattr(ov.single, "Augur")

    def test_run_basic(self, sample_adata):
        """run() produces valid results with 'AUC' key."""
        augur = ov.single.Augur(
            sample_adata, label_col="label", cell_type_col="cell_type",
        )
        augur.run(n_subsamples=5, subsample_size=10, folds=2)

        assert augur.result is not None
        assert "AUC" in augur.result
        assert "results" in augur.result
        assert "feature_importance" in augur.result
        assert "parameters" in augur.result

        auc_df = augur.result["AUC"]
        assert "cell_type" in auc_df.columns
        assert "auc" in auc_df.columns
        assert len(auc_df) > 0
        assert auc_df["auc"].between(0, 1).all()

    def test_results_in_uns(self, sample_adata):
        """Results are stored in adata.uns['augur']."""
        augur = ov.single.Augur(
            sample_adata, label_col="label", cell_type_col="cell_type",
        )
        augur.run(n_subsamples=5, subsample_size=10, folds=2)

        assert "augur" in sample_adata.uns
        uns = sample_adata.uns["augur"]
        assert "AUC" in uns
        assert "results" in uns

    def test_augur_auc_in_obs(self, sample_adata):
        """Per-cell AUC scores are mapped into adata.obs['augur_auc']."""
        augur = ov.single.Augur(
            sample_adata, label_col="label", cell_type_col="cell_type",
        )
        augur.run(n_subsamples=5, subsample_size=10, folds=2)

        assert "augur_auc" in sample_adata.obs.columns
        assert sample_adata.obs["augur_auc"].notna().all()

    def test_lr_classifier(self, sample_adata):
        """Logistic regression classifier works."""
        augur = ov.single.Augur(
            sample_adata, label_col="label", cell_type_col="cell_type",
            classifier="lr",
        )
        augur.run(n_subsamples=3, subsample_size=10, folds=2)
        assert "AUC" in augur.result

    def test_plot_lollipop(self, sample_adata):
        """plot_lollipop returns a figure and axes."""
        import matplotlib.pyplot as plt

        augur = ov.single.Augur(
            sample_adata, label_col="label", cell_type_col="cell_type",
        )
        augur.run(n_subsamples=3, subsample_size=10, folds=2)
        fig, ax = augur.plot_lollipop()
        assert fig is not None
        assert ax is not None
        plt.close("all")

    def test_plot_important_features(self, sample_adata):
        """plot_important_features returns a figure and axes."""
        import matplotlib.pyplot as plt

        augur = ov.single.Augur(
            sample_adata, label_col="label", cell_type_col="cell_type",
        )
        augur.run(n_subsamples=3, subsample_size=10, folds=2)
        fig, ax = augur.plot_important_features(top_n=5)
        assert fig is not None
        plt.close("all")

    def test_plot_augur_combined(self, sample_adata):
        """plot_augur returns a combined figure."""
        import matplotlib.pyplot as plt

        augur = ov.single.Augur(
            sample_adata, label_col="label", cell_type_col="cell_type",
        )
        augur.run(n_subsamples=3, subsample_size=10, folds=2)
        fig = augur.plot_augur()
        assert fig is not None
        plt.close("all")

    def test_run_returns_self(self, sample_adata):
        """run() returns self for method chaining."""
        augur = ov.single.Augur(
            sample_adata, label_col="label", cell_type_col="cell_type",
        )
        result = augur.run(n_subsamples=3, subsample_size=10, folds=2)
        assert result is augur

    def test_plot_before_run_raises(self, sample_adata):
        """Plotting before run() raises RuntimeError."""
        augur = ov.single.Augur(
            sample_adata, label_col="label", cell_type_col="cell_type",
        )
        with pytest.raises(RuntimeError, match=r"Run \.run\(\)"):
            augur.plot_lollipop()

    def test_velocity_mode(self, sample_adata):
        """augur_mode='velocity' sets feature_perc and var_quantile to 1.0."""
        augur = ov.single.Augur(
            sample_adata, label_col="label", cell_type_col="cell_type",
        )
        augur.run(n_subsamples=3, subsample_size=10, folds=2, augur_mode="velocity")
        params = augur.result["parameters"]
        assert params["feature_perc"] == 1.0
        assert params["var_quantile"] == 1.0

    def test_run_differential(self, sample_adata, adata2):
        """run_differential returns DataFrame with expected columns."""
        augur = ov.single.Augur(
            sample_adata, label_col="label", cell_type_col="cell_type",
        )
        augur.run(n_subsamples=3, subsample_size=10, folds=2)
        dp = augur.run_differential(adata2, n_subsamples=3, subsample_size=10,
                                    folds=2, n_permutations=10)
        assert isinstance(dp, pd.DataFrame)
        for col in ("cell_type", "auc.x", "auc.y", "delta_auc", "pval", "padj"):
            assert col in dp.columns
        assert len(dp) > 0

    def test_plot_scatterplot(self, sample_adata):
        """plot_scatterplot returns a figure and axes."""
        import matplotlib.pyplot as plt

        augur1 = ov.single.Augur(
            sample_adata, label_col="label", cell_type_col="cell_type", seed=42,
        )
        augur1.run(n_subsamples=3, subsample_size=10, folds=2)
        augur2 = ov.single.Augur(
            sample_adata.copy(), label_col="label", cell_type_col="cell_type", seed=99,
        )
        augur2.run(n_subsamples=3, subsample_size=10, folds=2)
        fig, ax = augur1.plot_scatterplot(augur2)
        assert fig is not None
        assert ax is not None
        plt.close("all")

    def test_plot_umap(self, sample_adata):
        """plot_umap returns a figure and axes with X_umap in obsm."""
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(42)
        sample_adata.obsm["X_umap"] = rng.standard_normal((sample_adata.n_obs, 2))
        augur = ov.single.Augur(
            sample_adata, label_col="label", cell_type_col="cell_type",
        )
        augur.run(n_subsamples=3, subsample_size=10, folds=2)
        fig, ax = augur.plot_umap()
        assert fig is not None
        assert ax is not None
        plt.close("all")

    def test_plot_differential_prioritization(self, sample_adata):
        """plot_differential_prioritization runs on synthetic dp results."""
        import matplotlib.pyplot as plt

        augur = ov.single.Augur(
            sample_adata, label_col="label", cell_type_col="cell_type",
        )
        augur.run(n_subsamples=3, subsample_size=10, folds=2)

        # Build synthetic differential results (avoids slow permute run)
        auc_df = augur.result["AUC"].copy()
        dp = pd.DataFrame({
            "cell_type": auc_df["cell_type"],
            "auc.x": auc_df["auc"],
            "auc.y": auc_df["auc"] + np.array([0.1, -0.1, 0.05]),
            "delta_auc": np.array([0.1, -0.1, 0.05]),
            "pval": np.array([0.01, 0.03, 0.5]),
            "padj": np.array([0.03, 0.05, 0.5]),
            "z": np.array([2.5, -2.1, 0.5]),
        })
        fig, ax = augur.plot_differential_prioritization(dp)
        assert fig is not None
        assert ax is not None
        plt.close("all")
