# Novae integration

Trains Novae or runs zero-shot inference using a caller-supplied pretrained model. Copies embeddings and domain labels back in cell order while preserving input expression and graphs. No checkpoint is downloaded implicitly.

Part of [roadmap #989](https://github.com/omicverse/omicverse/issues/989).

## Usage

Install the optional backend with `pip install "omicverse[novae]"` using **Python 3.11 or later**. The optional backend has a higher Python floor than the base OmicVerse package. Importing OmicVerse does not import the backend.

```python
import omicverse as ov

model = ov.space.novae(adata, slide_key="slide",
                       train_kwargs={"max_epochs": 20, "num_workers": 0},
                       representation_kwargs={"num_workers": 0},
                       domains_kwargs={"level": 7})
```

## Contract and validation

Setup, constructor and training/solver options are passed separately. Inputs and outputs are checked; existing output embeddings require explicit overwrite. Native model/problem objects are returned rather than stored in `AnnData.uns`. Backend failures propagate.

Run `python -m pytest tests/test_backend_validation.py tests/test_roadmap989_novae.py -q`. Integration tests require the real backend; skipped tests do not establish real execution. Small synthetic training runs check execution and output contracts, not convergence on arbitrary biological data.

The generic validation helper is byte-identical across the independent roadmap PRs. This PR introduces only the Novae public API and its optional dependency.

Isolated cells may remain unassigned. An unavailable requested domain count raises instead of silently changing clustering. Pretrained inference requires a caller-loaded model; the integration test trains a small synthetic model, saves and reloads its local checkpoint, then infers on a 40-gene subset across two slides. It checks that graph edges do not cross slide boundaries and that original expression, coordinates, graphs and cell order are preserved. It does not download or benchmark a public foundation checkpoint.
