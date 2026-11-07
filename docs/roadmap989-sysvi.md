# sysVI integration

Adds SysVI integration on caller-normalized expression with at least two systems. Separates setup, model and training arguments, validates the resulting embedding, and returns the native model.

Part of [roadmap #989](https://github.com/omicverse/omicverse/issues/989).

## Usage

Install the optional backend with `pip install "omicverse[sysvi]"` in an environment supported by the backend. Importing OmicVerse does not import the backend.

```python
import omicverse as ov

model = ov.single.sysvi(adata, batch_key="system", layer="logcounts",
                        train_kwargs={"max_epochs": 100, "accelerator": "cpu"})
```

## Contract and validation

Setup, constructor and training/solver options are passed separately. Inputs and outputs are checked; existing output embeddings require explicit overwrite. Native model/problem objects are returned rather than stored in `AnnData.uns`. Backend failures propagate.

Run `python -m pytest tests/test_backend_validation.py tests/test_roadmap989_sysvi.py -q`. Integration tests require the real backend; skipped tests do not establish real execution. Small synthetic training runs check execution and output contracts, not convergence on arbitrary biological data.

The generic validation helper is byte-identical across the independent roadmap PRs. This PR introduces only the sysVI public API and its optional dependency.
