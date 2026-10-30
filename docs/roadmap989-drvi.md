# DRVI integration

Adds count-based DRVI training and validated embedding output. Rejects Gaussian likelihoods at this count-only interface instead of silently changing the preprocessing contract.

Part of [roadmap #989](https://github.com/omicverse/omicverse/issues/989).

## Usage

Install the optional backend with `pip install "omicverse[drvi]"` in an environment supported by the backend. Importing OmicVerse does not import the backend.

```python
import omicverse as ov

model = ov.single.drvi(adata, layer="counts", batch_key="batch",
                       train_kwargs={"max_epochs": 100, "accelerator": "cpu"})
```

## Contract and validation

Setup, constructor and training/solver options are passed separately. Inputs and outputs are checked; existing output embeddings require explicit overwrite. Native model/problem objects are returned rather than stored in `AnnData.uns`. Backend failures propagate.

Run `python -m pytest tests/test_backend_validation.py tests/test_roadmap989_drvi.py -q`. Integration tests require the real backend; skipped tests do not establish real execution. Small synthetic training runs check execution and output contracts, not convergence on arbitrary biological data.

The generic validation helper is byte-identical across the independent roadmap PRs. This PR introduces only the DRVI public API and its optional dependency.
