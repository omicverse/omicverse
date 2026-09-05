# Spatial feature statistical defaults

Moran, SOMDE and SpatialDE now default to selection='significant': apply the
q-value threshold before n_svgs. Use selection='top_n' explicitly to reproduce
rank-based prefilter intent; top_n does not imply significant spatial evidence.
PROST and Spateo retain their method-specific selection. Pearson residuals
(including pearsonr alias) remain a non-spatial HVG prefilter and are labeled so.

```python
ov.space.svg(adata, mode='moran', selection='significant', qval_threshold=0.05,
             n_svgs=2000, n_perms=999, seed=0, library_key='slice')
table = adata.uns['spatial_features_by_library']
per_slice = adata.varm['space_variable_features_by_library']
```

When multiple libraries are supplied, graph construction and tests run within
each library. Results are a long table of library/gene/statistic/raw p-value/
q-value/selection. BH correction uses all testable library-gene pairs in the
call; n_svgs is a per-library cap. var['space_variable_features'] is the union
of those candidate sets, not evidence of replicated differential expression.

moranI/spatial_autocorr likewise return library/gene long tables for multiple
libraries and ordinary gene-indexed tables for single libraries. Library identity
can be read from the graph's recorded library_key; ambiguous multi-image objects
without identity are rejected. Cross-library edges are rejected. A pure shift
between libraries cannot generate within-library variation. Constant genes,
fewer-than-four-spot slices and edgeless graphs are untestable (NaN), not p=0.

copy=True leaves X, obs, var, uns and obsp unchanged, including automatic graph
construction. Empirical permutation p-values use (exceedances+1)/(B+1). Two-sided
tests compare departures from the null expectation. z_sim stores the z-score;
pval_z_sim now stores a probability rather than a mislabeled z-score.

Single-library statistics and normal-approximation p-values are compared with
ESDA using a prespecified positive-autocorrelation tail (ESDA's default chooses
its one-sided tail based on the observed direction). Multi-library statistics
are compared with separate single-library runs followed by joint BH adjustment.
Real SOMDE/SpatialDE backend outputs are checked against wrapper fields. Their
removed scipy.misc.derivative dependency is replaced by the same three-point
formula, with no global SciPy modification.

Local validation: 48 passed, 5 skipped. The skips are unrelated optional Squidpy
neighborhood comparisons; ESDA, real SOMDE and real SpatialDE tests ran. These
checks establish numerical/runtime contracts, not universal calibration across
all biological datasets. No CI, setup.py or dependency configuration changes.
