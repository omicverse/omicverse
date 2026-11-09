"""CytoSignal port vs R CytoSignal 0.5.1 on a real Slide-tags (E14 mouse brain) sub-region.

Fixture: tests/single/data/cytosignal_r_reference.{npz,R} -- 360 contiguous
cells of SCP2170 (Russell et al. 2023) and the R outputs of the vignette
pipeline (``set.seed(42)``, ``perm.size = 5000``, ``reads/sig.thresh = 10``).
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

FIX = Path(__file__).parent / "data" / "cytosignal_r_reference.npz"
PERM, THR = 5000, 10


@pytest.fixture(scope="module")
def ref():
    return np.load(FIX, allow_pickle=False)


@pytest.fixture(scope="module")
def adata(ref):
    X = sp.csr_matrix((ref["X_data"], ref["X_indices"], ref["X_indptr"]), shape=tuple(ref["X_shape"]))
    a = ad.AnnData(X=X.astype(np.float32), obs=pd.DataFrame(index=ref["obs_names"]),
                   var=pd.DataFrame(index=ref["var_names"]))
    a.obs["cell_type"] = ref["cell_type"]
    a.obsm["spatial"] = ref["spatial"]
    return a


@pytest.fixture(scope="module")
def pipeline(adata):
    from omicverse.external.cytosignal import _core
    from omicverse.external.cytosignal._db import load_cytosignal_db

    return _core.run_pipeline(adata.X, list(adata.var_names), adata.obsm["spatial"], db=load_cytosignal_db(),
                              scale_factor=0.73, perm_size=PERM, seed=42, reads_thresh=THR, sig_thresh=THR)


def test_rng_matches_r():
    from omicverse.external.cytosignal._rrng import RRandom

    r = RRandom(42)  # R: set.seed(42); sample(4623)[1:3]; sample(10)
    assert list(r.sample(4623)[:3] + 1) == [2609, 4069, 2369]
    r = RRandom(7)  # R: set.seed(7); runif(1)
    assert r.unif_rand() == pytest.approx(0.9889092978555709, abs=1e-16)


@pytest.mark.parametrize("slot", ["diffusion", "contact", "raw"])
def test_scores_pvalues_match_r(ref, pipeline, slot):
    _, _, res = pipeline
    r = res[slot]
    assert r.interactions == list(ref[f"{slot}_interactions"])
    j = [r.interactions.index(i) for i in ref[f"{slot}_pick"]]
    np.testing.assert_allclose(r.score[:, j], ref[f"{slot}_score"], rtol=1e-12, atol=1e-14)
    # identical R RNG stream -> identical permutation nulls -> identical p-values
    np.testing.assert_array_equal(r.pval[:, j], ref[f"{slot}_pval"])
    np.testing.assert_allclose(r.padj[:, j], ref[f"{slot}_padj"], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("slot", ["diffusion", "contact", "raw"])
def test_significant_cells_match_r(ref, pipeline, slot):
    _, _, res = pipeline
    r = res[slot]
    assert list(r.result) == list(ref[f"{slot}_result_order"])
    assert list(r.result_hq) == list(ref[f"{slot}_hq_order"])
    hq = ref[f"{slot}_hq_order"]
    want = pd.DataFrame({"i": hq[ref[f"{slot}_hq_intr"]], "c": ref[f"{slot}_hq_cell"]})
    for intr, cells in r.result_hq.items():
        assert set(cells) == set(want.loc[want.i == intr, "c"])


def test_run_adapter_contract(adata):
    from omicverse.single import _ccc_cytosignal as cs

    a = adata.copy()
    out = cs.run(a, groupby="cell_type", species="mouse", seed=42, scale_factor=0.73, perm_size=PERM,
                 reads_thresh=THR, sig_thresh=THR, min_cells=5, n_perms=200, progress=False)
    assert cs.NEEDS_SPATIAL and cs.SCORE_SCALE == "expression"
    long = out["long"]
    assert list(long.columns) == ["source", "target", "ligand", "receptor", "ligand_genes", "receptor_genes",
                                  "interaction", "pathway", "annotation", "score", "pvalue", "significant"]
    assert set(long["annotation"]) <= {"diffusion", "contact"}
    assert (long["score"] > 0).all() and long["pvalue"].between(0, 1).all()
    assert long["significant"].any()
    sizes = a.obs["cell_type"].value_counts()
    assert set(long["source"]) | set(long["target"]) <= set(sizes[sizes >= 5].index)
    for k in ("cytosignal_scores", "cytosignal_padj"):
        df = out["obsm"][k]
        assert list(df.index) == list(a.obs_names)
    assert {"interactions", "significant_cells", "parameters"} <= set(out["raw"])


def test_group_score_conserves_receiver_score(adata, pipeline):
    """Summed over senders, the group score equals the receivers' mean CytoSignal score."""
    from omicverse.external.cytosignal import _group

    keep, st, res = pipeline
    r = res["diffusion"]
    labels = adata.obs["cell_type"].to_numpy()[keep]
    groups = sorted(set(labels))
    codes = np.searchsorted(np.asarray(groups, dtype=object), labels)
    lc = _group.ligand_counts(st.counts, r.intr.lig)
    sig = np.zeros(r.score.shape, dtype=bool)
    score, pval = _group.group_summary(st.graphs["diffusion"], lc, r.score, sig, codes, len(groups), n_perms=0)
    total = np.asarray(st.graphs["diffusion"].T @ lc)
    for t in range(len(groups)):
        cells = codes == t
        expect = np.where(total[cells] > 0, r.score[cells], 0).mean(axis=0)
        np.testing.assert_allclose(score[:, t, :].sum(axis=0), expect, rtol=1e-10, atol=1e-12)
    assert (pval == 1).all()  # nothing significant -> nothing tested


def test_rejects_normalised_input(adata):
    from omicverse.single import _ccc_cytosignal as cs

    a = adata.copy()
    a.X = np.log1p(a.X.toarray())
    with pytest.raises(ValueError, match="raw counts"):
        cs.run(a, groupby="cell_type", progress=False)
