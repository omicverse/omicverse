"""Tests for the redesigned ``ov.single.autoResolution``.

The algorithm picks the most stable Leiden resolution via bootstrap-ARI:
for each resolution, it reclusters N subsamples and scores the mean
ARI against the full-data reference labels. These tests build a small
synthetic AnnData with 3 well-separated Gaussian blobs and verify:

- The chosen resolution produces close to 3 clusters on this dataset.
- The ``uns['autoResolution']`` payload has the expected schema.
- Temporary obs columns leaked during the search are cleaned up.
- The min_clusters guard rejects degenerate resolutions.
- The function fails loudly when the neighbor graph is missing.
"""
from __future__ import annotations

import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest


def _three_blob_adata(n_per_blob: int = 100, n_genes: int = 50,
                       seed: int = 0) -> ad.AnnData:
    """Three well-separated blobs in gene space → leiden picks ~3 clusters."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((3, n_genes))
    centers[0, :n_genes // 3] = 5
    centers[1, n_genes // 3:2 * n_genes // 3] = 5
    centers[2, 2 * n_genes // 3:] = 5
    X = np.vstack([
        centers[0] + rng.normal(0, 0.5, size=(n_per_blob, n_genes)),
        centers[1] + rng.normal(0, 0.5, size=(n_per_blob, n_genes)),
        centers[2] + rng.normal(0, 0.5, size=(n_per_blob, n_genes)),
    ]).astype(np.float32)
    obs = pd.DataFrame({
        "true_label": np.repeat(["A", "B", "C"], n_per_blob),
    }, index=[f"c{i}" for i in range(3 * n_per_blob)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture(scope="module")
def adata_with_neighbors():
    """Synthetic AnnData with 3 well-separated clusters + neighbor graph."""
    os.environ.setdefault("OMICVERSE_DISABLE_LLM", "1")
    import scanpy as sc

    a = _three_blob_adata(n_per_blob=120, n_genes=60, seed=0)
    sc.pp.pca(a, n_comps=10)
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X_pca")
    return a


def test_autoresolution_picks_three_clusters(adata_with_neighbors):
    """Three well-separated blobs → chosen resolution should be the
    one whose reference clustering recovers ~3 clusters."""
    import omicverse as ov

    a = adata_with_neighbors.copy()
    _, best, df = ov.single.autoResolution(
        a,
        resolutions=[0.1, 0.3, 0.5, 0.8, 1.2],
        n_subsamples=3,
        random_state=0,
        verbose=False,
    )

    # Best resolution is one of the candidates we passed in.
    assert best in [0.1, 0.3, 0.5, 0.8, 1.2]
    # On three well-separated blobs the chosen resolution should land
    # near three clusters (allow 2-4 for noise / leiden tie-breaks).
    assert 2 <= int(df.loc[best, "n_clusters"]) <= 4
    # Mean ARI on this trivial data should be very high.
    assert df.loc[best, "mean_ari"] >= 0.8


def test_autoresolution_writes_uns_and_obs(adata_with_neighbors):
    import omicverse as ov

    a = adata_with_neighbors.copy()
    _, best, df = ov.single.autoResolution(
        a,
        resolutions=[0.3, 0.6],
        n_subsamples=2,
        random_state=0,
        key_added="leiden_auto",
        verbose=False,
    )
    # The chosen resolution's labels are written under key_added.
    assert "leiden_auto" in a.obs.columns
    assert a.obs["leiden_auto"].nunique() == int(df.loc[best, "n_clusters"])

    # uns payload schema: best_resolution + scores table + bookkeeping.
    payload = a.uns["autoResolution"]
    assert payload["best_resolution"] == best
    assert payload["method"] == "bootstrap-ARI"
    assert payload["n_subsamples"] == 2
    # `scores` is a DataFrame.to_dict('list') — round-trippable.
    scores_df = pd.DataFrame(payload["scores"])
    assert set(scores_df.columns) >= {"resolution", "mean_ari",
                                        "std_ari", "n_clusters"}
    assert sorted(scores_df["resolution"].tolist()) == [0.3, 0.6]


def test_autoresolution_cleans_temp_obs_cols(adata_with_neighbors):
    """The intermediate '_autores_*' obs columns must NOT leak."""
    import omicverse as ov

    a = adata_with_neighbors.copy()
    obs_cols_before = set(a.obs.columns)
    ov.single.autoResolution(
        a, resolutions=[0.3, 0.5], n_subsamples=2,
        random_state=0, verbose=False,
    )
    leaked = [c for c in a.obs.columns
               if c.startswith("_autores") and c not in obs_cols_before]
    assert leaked == []


def test_autoresolution_rejects_degenerate_min_clusters(adata_with_neighbors):
    """If every candidate produces fewer than min_clusters clusters,
    raise rather than return a meaningless 'best'."""
    import omicverse as ov

    a = adata_with_neighbors.copy()
    with pytest.raises(RuntimeError, match="No resolution"):
        # Set min_clusters absurdly high — no resolution will reach it.
        ov.single.autoResolution(
            a, resolutions=[0.1, 0.3], n_subsamples=2,
            min_clusters=999, random_state=0, verbose=False,
        )


def test_autoresolution_requires_neighbor_graph():
    import omicverse as ov

    a = _three_blob_adata(n_per_blob=60, n_genes=20, seed=0)
    # No PCA, no neighbors.
    with pytest.raises(ValueError, match="connectivities"):
        ov.single.autoResolution(a, verbose=False)


def test_autoresolution_rejects_tiny_adata():
    import omicverse as ov

    a = _three_blob_adata(n_per_blob=10, n_genes=20, seed=0)
    with pytest.raises(ValueError, match="at least 50 cells"):
        ov.single.autoResolution(a, verbose=False)


def test_autoresolution_records_provenance(adata_with_neighbors):
    """Confirm the @tracked decorator wires this into adata.uns['_ov_provenance']
    with the expected name + viz spec."""
    import omicverse as ov
    from omicverse.report._provenance import (
        clear_provenance, get_provenance,
    )

    a = adata_with_neighbors.copy()
    clear_provenance(a)
    ov.single.autoResolution(
        a, resolutions=[0.3, 0.6], n_subsamples=2,
        random_state=0, verbose=False,
    )
    prov = get_provenance(a)
    # Internal leiden invocations must be silenced by the nesting guard.
    names = [e["name"] for e in prov]
    assert names == ["autoResolution"]
    e = prov[0]
    assert e["function"] == "ov.single.autoResolution"
    assert "ARI-stability" in e["backend"]
    # viz spec: at least the cluster_sizes_bar.
    viz_fns = [v["function"] for v in e["viz"]]
    assert "ov.pl.cluster_sizes_bar" in viz_fns
