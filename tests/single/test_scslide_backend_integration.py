"""Real optional-backend test. A skip is not a successful integration test."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from scipy import sparse


def test_real_scslide_matches_direct_workflow_without_mutating_input():
    ad = pytest.importorskip("anndata")
    backend = pytest.importorskip("scslide", reason="Install the audited scslide-py artifact to run integration")
    ov = pytest.importorskip("omicverse")
    rng = np.random.default_rng(17)
    samples = np.repeat(np.arange(12), 30)
    counts = rng.poisson(2.0, (360, 40)).astype(float)
    counts[:, :8] += rng.poisson((samples / 3 + 1)[:, None], (360, 8))
    obs = pd.DataFrame({"sample": [f"s{i}" for i in samples],
                        "subject": [f"p{i}" for i in samples],
                        "response": samples.astype(float)}, index=[f"c{i}" for i in range(360)])
    data = ad.AnnData(sparse.csr_matrix(counts), obs=obs,
                     var=pd.DataFrame({"highly_variable": np.ones(40, dtype=bool)},
                                      index=[f"g{i}" for i in range(40)]))
    data.layers["counts"] = data.X.copy()
    data.layers["data"] = sparse.csr_matrix(np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4))
    data.obsm["X_pca"] = rng.normal(size=(360, 5))
    options = dict(ncells_per_group=15, ncells_landmark=60, ncomp_pls=3,
                   k_nn=3, knn_range=30, ncomp_diffusion=3, fit_curve=False)
    wrapped = ov.single.scslide(data, sample_key="sample", response_key="response",
                                subject_key="subject", n_jobs=1, random_state=17,
                                workflow_kwargs=options)
    direct = backend.run_scslide_workflow(data.copy(), sample_col="sample", response_col="response",
                                          subject_col="subject", embedding_key="X_pca", n_jobs=1,
                                          seed=17, nn_method="exact", de_inference_mode="descriptive_r_compat",
                                          **options)
    np.testing.assert_array_equal(wrapped.sample_object.counts, direct.sample_object.counts)
    np.testing.assert_allclose(wrapped.diffusion.embeddings, direct.diffusion.embeddings)
    assert data.uns == {}
    assert "X_proj.pls" not in data.obsm
    assert wrapped.sample_adata.n_obs == 12
    assert wrapped.sample_adata.n_vars == 60
    assert wrapped.sample_adata.obsm["X_scslide_diffmap"].shape == (12, 3)
