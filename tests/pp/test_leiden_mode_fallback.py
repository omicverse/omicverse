"""ov.pp.leiden with CPU-only arguments (flavor / n_iterations / directed) runs the
CPU implementation whatever ov.settings.mode is; the torch / RAPIDS paths reject
those arguments (regression: scanpy-replacement call sites in mixed mode)."""
import sys
import types

import numpy as np
import pytest


def _graph():
    import anndata as ad
    import omicverse as ov
    rs = np.random.default_rng(0)
    centers = rs.normal(size=(3, 10)) * 6
    X = np.vstack([c + rs.normal(size=(60, 10)) for c in centers]).astype(np.float32)
    a = ad.AnnData(X)
    ov.settings.cpu_init()
    ov.pp.neighbors(a, n_neighbors=10, use_rep="X", method="umap")
    return a


@pytest.mark.parametrize("mode", ["cpu-gpu-mixed", "gpu"])
def test_cpu_only_kwargs_use_cpu_leiden(mode, monkeypatch):
    import omicverse as ov
    a = _graph()

    def boom(*args, **kwargs):
        raise AssertionError("torch / RAPIDS Leiden must not run for flavor=...")
    # stand-ins for the torch and RAPIDS backends (need not be installed)
    fake_torch_leiden = types.ModuleType("omicverse.pp._leiden_gpu")
    fake_torch_leiden.leiden_gpu_sparse_multilevel = boom
    fake_rsc = types.ModuleType("rapids_singlecell")
    fake_rsc.tl = types.SimpleNamespace(leiden=boom)
    monkeypatch.setitem(sys.modules, "omicverse.pp._leiden_gpu", fake_torch_leiden)
    monkeypatch.setitem(sys.modules, "rapids_singlecell", fake_rsc)
    monkeypatch.setattr(ov.settings, "mode", mode)
    ov.pp.leiden(a, resolution=0.5, key_added="lv", flavor="igraph", n_iterations=2,
                 directed=False, random_state=0)
    assert a.obs["lv"].nunique() == 3
