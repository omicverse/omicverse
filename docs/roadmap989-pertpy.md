# pertpy integration

Adds guide-count assignment, Mixscape signatures/classification and group-distance tables. Preserves input expression during Mixscape and keeps descriptive distances separate from replicate-level inference. This is the core workflow, not full pertpy/Systema coverage.

Part of [roadmap #989](https://github.com/omicverse/omicverse/issues/989).

## Usage

Install the optional backend with `pip install "omicverse[pertpy]"` in an environment supported by the backend. Importing OmicVerse does not import the backend.

```python
import omicverse as ov

labels = ov.single.assign_guides(guide_adata, assignment_threshold=5)
classified = ov.single.mixscape(adata, pert_key="target", control="NT", layer="logcounts")
distances = ov.single.perturbation_distance(adata, groupby="target", use_rep="X_pca")
```

## Contract and validation

Setup, constructor and training/solver options are passed separately. Inputs and outputs are checked; existing output embeddings require explicit overwrite. Native model/problem objects are returned rather than stored in `AnnData.uns`. Backend failures propagate.

Run `python -m pytest tests/test_backend_validation.py tests/test_roadmap989_pertpy.py -q`. Integration tests require the real backend; skipped tests do not establish real execution. Small synthetic training runs check execution and output contracts, not convergence on arbitrary biological data.

The generic validation helper is byte-identical across the independent roadmap PRs. This PR introduces only the pertpy public API and its optional dependency.
