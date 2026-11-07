# ResolVI integration

Adds ResolVI training with raw-count, spatial-coordinate and neighborhood checks. Returns the native model and an embedding; normalized expression is requested explicitly and never replaces counts.

Part of [roadmap #989](https://github.com/omicverse/omicverse/issues/989).

## Usage

Install the optional backend with `pip install "omicverse[resolvi]"` in an environment supported by the backend. Importing OmicVerse does not import the backend.

```python
import omicverse as ov

model = ov.space.resolvi(adata, layer="counts", spatial_key="spatial", batch_key="slide",
                         train_kwargs={"max_epochs": 50, "accelerator": "cpu"})
corrected_expression = model.get_normalized_expression()
```

## Contract and validation

Setup, constructor and training/solver options are passed separately. Inputs and outputs are checked; existing output embeddings require explicit overwrite. Native model/problem objects are returned rather than stored in `AnnData.uns`. Backend failures propagate.

Run `python -m pytest tests/test_backend_validation.py tests/test_roadmap989_resolvi.py -q`. Integration tests require the real backend; skipped tests do not establish real execution. Small synthetic training runs check execution and output contracts, not convergence on arbitrary biological data.

The generic validation helper is byte-identical across the independent roadmap PRs. This PR introduces only the ResolVI public API and its optional dependency.

scvi-tools 1.5.2 emits a migration warning for ResolVI moving to scviva-tools. This adapter does not silently switch backends.
