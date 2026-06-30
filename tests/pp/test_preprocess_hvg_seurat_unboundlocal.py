"""Regression test for issue #706.

`ov.pp.preprocess(mode="shiftlog|seurat", ...)` raised::

    UnboundLocalError: cannot access local variable 'highly_variable_genes'
    where it is not associated with a value

…while the sibling `mode="shiftlog|pearson"` worked. Root cause was a
local-name shadowing bug in `omicverse/pp/_preprocess.py::preprocess`:
the `'pearson'` branch did ``from .experimental import
highly_variable_genes`` inline, which Python promoted to a function-scope
local-name binding. The `'seurat'` branch then referenced the same
name *without* importing it — so when it ran (the `'pearson'` branch
hadn't), Python raised `UnboundLocalError`.

Even if the import had worked, the experimental
`highly_variable_genes` only supports ``flavor='pearson_residuals'``
(see ``omicverse/pp/experimental/_highly_variable_genes.py:324``),
so the seurat branch should have been calling
``sc.pp.highly_variable_genes(flavor='seurat_v3', ...)`` from the
start — mirroring the rapids GPU branch a few lines below.

This test pins both modes (pearson + seurat) so neither path can
regress to the old bug.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData


def _synthetic_counts_adata(n_cells: int = 200, n_genes: int = 300,
                            seed: int = 0) -> AnnData:
    """Small Poisson-counts AnnData with a plausible HVG signal.

    The first 30 genes get a 5× count boost so HVG selection finds
    real candidates instead of returning an arbitrary 2k slice from
    noise. Pre-fixate one absurdly high count so any future
    `qc`-style filter that looks at `max(layer) < log(1e4)` doesn't
    misclassify the matrix as already-log-normalised.
    """
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=2.0, size=(n_cells, n_genes)).astype(np.int32)
    counts[:, :30] += rng.poisson(lam=8.0, size=(n_cells, 30)).astype(np.int32)
    counts[0, 0] = 30  # ensures max > log(1e4) ≈ 9.21
    adata = AnnData(X=sp.csr_matrix(counts.astype(np.float32)))
    adata.var_names = [f"Gene{i:04d}" for i in range(n_genes)]
    adata.obs_names = [f"cell{i:04d}" for i in range(n_cells)]
    # `preprocess()` reads from `adata.layers['counts']` for HVG
    # selection regardless of the chosen flavor.
    adata.layers['counts'] = adata.X.copy()
    return adata


@pytest.mark.parametrize("hvg_method", ["seurat", "pearson"])
def test_preprocess_shiftlog_modes_do_not_raise_unboundlocal(hvg_method):
    """Both `shiftlog|<hvg>` modes must run end-to-end and write
    `adata.var['highly_variable']`.

    Before the fix, the `seurat` parametrisation raised:

        UnboundLocalError: cannot access local variable
        'highly_variable_genes' where it is not associated with a value

    while the `pearson` parametrisation passed. We pin both so a
    future refactor that re-introduces the local-shadow can't drop
    only the seurat path silently.
    """
    import omicverse as ov

    adata = _synthetic_counts_adata()

    # NB: `no_cc=False` skips the cell-cycle gene removal that would
    # otherwise pull a curated mouse / human gene list from the
    # package data — irrelevant to the HVG-selection bug we're pinning.
    ov.pp.preprocess(
        adata,
        mode=f"shiftlog|{hvg_method}",
        n_HVGs=200,
        target_sum=10000,
        no_cc=False,
    )

    assert "highly_variable" in adata.var.columns, (
        f"preprocess(mode='shiftlog|{hvg_method}') did not set "
        f"`adata.var['highly_variable']`"
    )
    n_hvg = int(adata.var["highly_variable"].sum())
    # We asked for 200 HVGs; the actual count can drift slightly
    # because of ties / batch effects, but it should clearly be
    # in the right ballpark.
    assert 100 <= n_hvg <= 250, (
        f"preprocess(mode='shiftlog|{hvg_method}') selected {n_hvg} HVGs; "
        f"expected ~200 from `n_HVGs=200`"
    )


def test_preprocess_shiftlog_can_disable_high_expression_exclusion():
    """Expose Scanpy's high-expression exclusion switch through preprocess.

    With exclusion enabled, the size factor is computed from non-excluded
    genes and then applied to the whole matrix, so a dominant gene can exceed
    ``log1p(target_sum)``. Disabling exclusion restores the usual row-sum bound.
    """
    import omicverse as ov

    target_sum = 10000
    adata_without_exclusion = _synthetic_counts_adata()
    adata_without_exclusion.X = adata_without_exclusion.X.toarray()
    adata_without_exclusion.X[0, 0] = 1000
    adata_without_exclusion.X[0, 1:] = 1
    adata_without_exclusion.layers["counts"] = sp.csr_matrix(
        adata_without_exclusion.X.astype(np.float32)
    )
    adata_with_exclusion = _synthetic_counts_adata()
    adata_with_exclusion.X = adata_without_exclusion.X.copy()
    adata_with_exclusion.layers["counts"] = adata_without_exclusion.layers[
        "counts"
    ].copy()

    ov.pp.preprocess(
        adata_with_exclusion,
        mode="shiftlog|pearson",
        n_HVGs=200,
        target_sum=target_sum,
        identify_robust=False,
        no_cc=False,
    )

    ov.pp.preprocess(
        adata_without_exclusion,
        mode="shiftlog|pearson",
        n_HVGs=200,
        target_sum=target_sum,
        exclude_highly_expressed=False,
        identify_robust=False,
        no_cc=False,
    )

    assert float(adata_with_exclusion.X.max()) > np.log1p(target_sum)
    assert float(adata_without_exclusion.X.max()) <= np.log1p(target_sum) + 1e-4
    assert (
        adata_without_exclusion.uns["status_args"]["preprocess"][
            "exclude_highly_expressed"
        ]
        is False
    )
