# pertpy integration

Use pertpy to assign guides from counts, compute Mixscape signatures and classifications, and measure distances between groups of cells.

Part of [roadmap #989](https://github.com/omicverse/omicverse/issues/989).

## Usage

Use Python 3.12 or later. Install the optional backend with `pip install "omicverse[pertpy]"`. OmicVerse imports pertpy when you call these functions.

```python
import omicverse as ov

labels = ov.single.assign_guides(guide_adata, assignment_threshold=5)
classified = ov.single.mixscape(adata, pert_key="target", control="NT", layer="logcounts")
distances = ov.single.perturbation_distance(adata, groupby="target", use_rep="X_pca")
```

## Contract and validation

- `assign_guides` returns a pandas Series and writes the labels to `guide_adata.obs[key_added]`. It accepts dense counts and sparse counts in CSR or CSC format. Use `overwrite=True` to replace an existing label column.
- `mixscape` returns a new AnnData with perturbation signatures in `layers["X_pert"]` and classifications in `obs`. It leaves the input AnnData unchanged.
- `perturbation_distance` returns a pandas DataFrame, or a pair of mean and variance DataFrames when `pairwise_kwargs={"bootstrap": True}`. It uses the current representation without reading or writing a cache on the input AnnData. Some metrics and bootstrap settings can require quadratic memory in the number of cells. These distances describe differences between cell groups; replicate-level inference requires a separate analysis.

With pertpy installed, run `python -m pytest tests/test_backend_validation.py tests/test_roadmap989_pertpy.py -q`.
