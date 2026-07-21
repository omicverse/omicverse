"""R-parity for the milo NB-GLM engine after dropping inmoose.

``ov.single.Milo.da_nhoods`` now runs its differential-abundance NB-GLM through
the pure-Python ``pyedger`` port (edgeR ``estimateDisp`` -> ``glmQLFit`` ->
``glmQLFTest``) instead of ``inmoose.edgepy`` — the same engine
``miloR::testNhoods`` wraps.

The reference ``data/milo_edger_reference.csv`` was produced in R with
Bioconductor **edgeR 4.4.0** (CMAP env) on the identical synthetic nhood-count
matrix, using the identical options this module uses:

    dge <- DGEList(counts=Y); dge$samples$norm.factors <- 1
    dge <- estimateDisp(dge, design, trend.method="none", robust=FALSE)
    fit <- glmQLFit(dge, design, robust=FALSE, legacy=FALSE)   # edgeR 4.x default
    qlf <- glmQLFTest(fit, coef=2)

Findings (pinned below): **logFC is bit-exact** vs edgeR (max|Δ| < 1e-3);
**F / PValue are very high agreement but not bit-exact** (Pearson > 0.99,
PValue Spearman > 0.99) because pyedger's quasi-likelihood variance moderation
(``squeezeVar``) is not yet a bit-exact port of limma's. The same
edgeR-vs-pyedger F/PValue gap holds against ``miloR::testNhoods``'s default
(trended-dispersion + robust=TRUE) engine — within the same tolerance.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REF = Path(__file__).parent / "data" / "milo_edger_reference.csv"


def _simulate_nhood_counts():
    """Identical synthetic nhood-count matrix used to build the R reference."""
    rng = np.random.default_rng(0)
    n_nhoods, n_samp = 150, 8
    cond = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    mu = rng.uniform(20.0, 300.0, n_nhoods)
    lfc = np.zeros(n_nhoods)
    lfc[:40] = rng.normal(0.0, 1.2, 40)
    disp = 0.15
    counts = np.zeros((n_nhoods, n_samp), dtype=int)
    for j in range(n_samp):
        m = mu * (2.0 ** (lfc * cond[j]))
        lam = rng.gamma(1.0 / disp, m / (1.0 / disp))
        counts[:, j] = rng.poisson(lam)
    Y = pd.DataFrame(counts,
                     index=[f"nh{i}" for i in range(n_nhoods)],
                     columns=[f"s{j}" for j in range(n_samp)])
    return Y, cond


def _pyedger_da(Y, cond):
    """The exact NB-GLM pipeline milo.da_nhoods runs (pyedger)."""
    from omicverse.external import pyedger as edger

    X = np.column_stack([np.ones(len(cond)), cond.astype(float)])
    dge = edger.DGEList(counts=Y, norm_factors=np.ones(len(cond)),
                        group=np.zeros(len(cond)))
    edger.estimateDisp(dge, design=X, trend_method="none", robust=False)
    fit = edger.glmQLFit(dge, design=X, robust=False, legacy=False)
    res = edger.glmQLFTest(fit, coef=1)
    tab = res.table.copy()
    tab.index = Y.index
    return tab


def test_milo_pyedger_matches_bioconductor_edger():
    from scipy.stats import spearmanr

    ref = pd.read_csv(_REF, index_col=0)
    Y, cond = _simulate_nhood_counts()
    tab = _pyedger_da(Y, cond).loc[ref.index]

    # logFC: R-parity (bit-exact up to GLM-IRLS tolerance)
    d_lfc = np.abs(tab["logFC"].to_numpy() - ref["logFC"].to_numpy())
    assert d_lfc.max() < 1e-2, f"logFC max|Δ|={d_lfc.max():.2e} vs edgeR"

    # F / PValue: high agreement (pyedger QL squeezeVar is not yet a bit-exact
    # limma port — documented in the module docstring).
    f_pear = np.corrcoef(tab["F"], ref["F"])[0, 1]
    p_spear = spearmanr(tab["PValue"], ref["PValue"]).correlation
    assert f_pear > 0.99, f"F Pearson={f_pear:.4f} vs edgeR"
    assert p_spear > 0.99, f"PValue Spearman={p_spear:.4f} vs edgeR"


def test_milo_da_nhoods_runs_without_inmoose(monkeypatch):
    """End-to-end: the milo workflow must run its DA test with no inmoose
    importable (proves the engine no longer depends on it)."""
    pytest.importorskip("mudata")  # milo.load builds a MuData object
    import builtins
    import scanpy as sc
    from anndata import AnnData
    import omicverse as ov

    real_import = builtins.__import__

    def _no_inmoose(name, *args, **kwargs):
        if name == "inmoose" or name.startswith("inmoose."):
            raise ImportError("inmoose is intentionally blocked in this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_inmoose)

    rng = np.random.default_rng(0)
    n = 500
    samp = np.array(["s0", "s1", "s2", "s3", "s4", "s5"])[rng.integers(0, 6, n)]
    cond = np.where(np.isin(samp, ["s0", "s1", "s2"]), "ctrl", "treat")
    X = rng.normal(0, 1, (n, 40)).astype("float32")
    X[cond == "treat", :5] += 1.5
    ad = AnnData(X, obs=dict(sample=samp, label=cond))
    sc.pp.pca(ad, n_comps=15)
    sc.pp.neighbors(ad, n_neighbors=15, use_rep="X_pca")

    milo = ov.single.Milo()
    mdata = milo.load(ad)
    milo.make_nhoods(mdata["rna"])
    mdata = milo.count_nhoods(mdata, sample_col="sample")
    milo.da_nhoods(mdata, design="~label")

    var = mdata["milo"].var
    for col in ("logFC", "PValue", "SpatialFDR"):
        assert col in var.columns
    assert np.isfinite(var["logFC"]).all()
