"""ov.single.CNV(method='infercnv') platform / cutoff guard.

The gene-filter cutoff is platform-dependent (R inferCNV: 10x->0.1,
SmartSeq2->1.0). The wrong default silently over-/under-filters genes, so
``run()`` requires an explicit ``platform`` or ``cutoff``. These guards fire
before any heavy computation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyinfercnv")
ad = pytest.importorskip("anndata")
ov = pytest.importorskip("omicverse")


def _tiny_adata():
    rng = np.random.RandomState(0)
    X = rng.poisson(1.0, size=(20, 12)).astype("float32")
    var = pd.DataFrame(
        {"chromosome": ["chr1"] * 12, "start": range(12), "end": range(1, 13)},
        index=[f"g{i}" for i in range(12)],
    )
    obs = pd.DataFrame(
        {"cell_type": ["Macrophage"] * 10 + ["Tumor"] * 10},
        index=[f"c{i}" for i in range(20)],
    )
    return ad.AnnData(X=X, obs=obs, var=var)


def test_infercnv_requires_platform_or_cutoff():
    cnv = ov.single.CNV(_tiny_adata(), method="infercnv")
    with pytest.raises(ValueError, match="platform"):
        cnv.run(reference_key="cell_type", reference_cat=["Macrophage"])


def test_infercnv_rejects_unknown_platform():
    cnv = ov.single.CNV(_tiny_adata(), method="infercnv")
    with pytest.raises(ValueError, match="unknown platform"):
        cnv.run(reference_key="cell_type", reference_cat=["Macrophage"], platform="nanopore")


def test_infercnv_rejects_platform_and_cutoff_together():
    cnv = ov.single.CNV(_tiny_adata(), method="infercnv")
    with pytest.raises(ValueError, match="not both"):
        cnv.run(
            reference_key="cell_type", reference_cat=["Macrophage"],
            platform="10x", cutoff=0.1,
        )
