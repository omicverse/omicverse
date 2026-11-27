# RUVg control-gene factors

`ov.bulk.ruvg` implements the RUVg **matrix method** from RUVSeq. It is a
first subset of [roadmap #989](https://github.com/omicverse/omicverse/issues/989),
not a full RUVSeq port. The algorithm now lives in the optional pyruvseq standalone package. Install the pinned backend following [the standalone adapter guide](roadmap989-ports.md). Neither R nor rpy2 is required at runtime.

## Usage

```python
import numpy as np
import pandas as pd
import omicverse as ov

# Synthetic example: genes on rows, independent samples on columns.
rng = np.random.default_rng(42)
counts = pd.DataFrame(rng.poisson(20, (100, 12)),
                      index=[f"gene_{i}" for i in range(100)],
                      columns=[f"sample_{i}" for i in range(12)])
negative_controls = counts.index[:20]  # choose validated controls for real data
result = ov.bulk.ruvg(counts, negative_controls, k=2)
W = result["W"]  # samples x factors; indexed in the original sample order
reconstructed = result["normalized_counts"]  # genes x samples
```

Use the columns of `W` as covariates in the DE design along with the biological
condition, and keep the **original counts** as the response for count models.
Check design rank and confounding before fitting. This function does not choose
control genes or `k`, fit DE, or prove that a biological effect was preserved.

DataFrames use gene **labels** for `controls`; arrays use zero-based integer row
positions. Both accept boolean row masks. Dense and sparse inputs are accepted,
but SVD and reconstructed outputs are dense; this is intended for bulk or
pseudobulk sample matrices, not out-of-core single-cell normalization.

## R parameter mapping and boundaries

| RUVSeq matrix method | Python |
| --- | --- |
| `x` (genes x samples) | `expression` (same orientation) |
| `cIdx` | `controls` (Python array positions are zero-based) |
| `k`, `drop`, `center`, `epsilon`, `tolerance` | same names |
| `round` | `round_counts` |
| `isLog` | `is_log` |
| result `W` | result `W` |
| result `normalizedCounts` | result `normalized_counts` |

`drop` discards leading singular vectors from the first `k`, so there are
`k-drop` factors when controls have sufficient rank. Rank deficiency reduces
`k` with a warning. No remaining factors is an error. Uncentered, already-logged,
nondefault-offset and unrounded paths are supported. Unrounded reconstructed
counts can be negative, as in the R matrix method.

Stricter input boundaries are deliberate: fractional raw counts, duplicate
controls/labels, invalid indices and nonfinite values fail clearly. Unlike the
R matrix method's warning-and-continue behavior, fractional values must be
explicitly supplied on a transformed scale with `is_log=True`.

RUVr, RUVs, automatic control selection, EDASeq/SeqExpressionSet methods,
standalone packaging and full-package equivalence are outside this PR.

## Reproducible reference validation

Reference: [RUVSeq RUVg matrix method](https://github.com/drisso/RUVSeq/blob/f560e84eb493f0c2d72279612f9e4ca99493b434/R/RUVg-methods.R),
commit `f560e84eb493f0c2d72279612f9e4ca99493b434`; its DESCRIPTION is version
1.33.1. The reference generator downloads pinned upstream source and executes
the **unmodified matrix-method body in R**, intercepting only S4 registration.
It does not install or validate the full RUVSeq package.

```bash
Rscript tests/bulk/generate_ruvg_reference.R
python -m pytest tests/bulk/test_ruvg.py -q
```

Checked-in fixtures allow pytest to run offline without R. Seven cases cover
rounded counts, unrounded counts, no centering, dropped leading factors,
already-logged values, a different log offset, and deficient control rank.
The gate was fixed before comparison: `W @ W.T` absolute error <= 1e-10;
correction absolute tolerance 1e-8 and relative tolerance 1e-10; rounded counts
must match exactly. Projection comparison accounts for arbitrary SVD signs.
Tests also check axis preservation, sample permutation, sparse/dense agreement,
unchanged input, orthogonality of the logged residuals and invalid inputs.

Method citation: Risso et al. (2014), *Normalization of RNA-seq data using factor
analysis of control genes or samples*, doi:10.1038/nbt.2931.
