# ResolVI integration

Adds ResolVI training with raw-count, spatial-coordinate and neighborhood checks. Returns the native model and an embedding; normalized expression is requested explicitly and never replaces counts.

Part of [roadmap #989](https://github.com/omicverse/omicverse/issues/989).

## Usage

Use Python 3.12 or later with scvi-tools 1.5.2. Install the optional backend with `pip install "omicverse[resolvi]"`. OmicVerse imports the backend when you call `resolvi()`.

```python
import omicverse as ov

model = ov.space.resolvi(adata, layer="counts", spatial_key="spatial", batch_key="slide",
                         train_kwargs={"max_epochs": 50, "accelerator": "cpu"})
corrected_expression = model.get_normalized_expression()
```

## Contract and validation

ResolVI rebuilds spatial neighbors from the current coordinates and slide labels on every fit. Setup writes the neighbor indices and distances to `adata.obsm` and registers the data in `adata.obs` and `adata.uns`. These changes can replace fields from a previous fit. Pass a copy of `adata` if you want to keep the original object unchanged.

The function returns the native ResolVI model and writes the embedding to `adata.obsm["X_resolvi"]` by default. Pass `overwrite=True` to replace an existing embedding. Choose an output key distinct from the coordinate and neighbor keys.

Run `python -m pytest tests/test_backend_validation.py tests/test_roadmap989_resolvi.py -q`. Integration tests require the real backend; skipped tests do not establish real execution. Small synthetic training runs check execution and output contracts, not convergence on arbitrary biological data.

scvi-tools 1.5.2 emits a migration warning for ResolVI moving to scviva-tools. This adapter does not silently switch backends.
