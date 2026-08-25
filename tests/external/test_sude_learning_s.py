from __future__ import annotations

import numpy as np
import pytest

from omicverse.external.sude_py import learning_s as learning_s_module


def test_torch_learning_s_matches_cpu_gradient_update(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(
        learning_s_module,
        "tqdm",
        lambda iterable, **kwargs: iterable,
    )

    rng = np.random.default_rng(7)
    samples = rng.normal(size=(12, 5)).astype(np.float32)
    unused_neighbors = np.empty((0, 0), dtype=int)
    sample_ids = np.arange(samples.shape[0])
    common_args = (
        samples,
        0,
        unused_neighbors,
        unused_neighbors,
        sample_ids,
        2,
        "pca",
        1.0,
        1,
    )

    cpu_embedding, cpu_k = learning_s_module._learning_s_cpu(*common_args)
    torch_embedding, torch_k = learning_s_module._learning_s_torch(
        *common_args,
        device="cpu",
    )

    assert torch_k == cpu_k
    assert np.isfinite(torch_embedding).all()
    np.testing.assert_allclose(torch_embedding, cpu_embedding, rtol=1e-5, atol=1e-6)
