# pyLemur integration

Fits LEMUR on explicitly selected variance-stabilized expression, preserves the input, and returns the model for condition predictions. Does not claim to implement R lemur neighborhood or pseudobulk hypothesis tests.

Part of [roadmap #989](https://github.com/omicverse/omicverse/issues/989).

## Usage

Install the optional backend with `pip install "omicverse[lemur]"` in an environment supported by the backend. Importing OmicVerse does not import the backend.

```python
import omicverse as ov

model = ov.single.lemur(adata, design="~ condition", layer="logcounts", n_embedding=15)
prediction = model.predict(new_condition=model.cond(condition="treated"))
```

## Contract and validation

Setup, constructor and training/solver options are passed separately. Inputs and outputs are checked; existing output embeddings require explicit overwrite. Native model/problem objects are returned rather than stored in `AnnData.uns`. Backend failures propagate.

Run `python -m pytest tests/test_backend_validation.py tests/test_roadmap989_lemur.py -q`. Integration tests require the real backend; skipped tests do not establish real execution. Small synthetic training runs check execution and output contracts, not convergence on arbitrary biological data.

The generic validation helper is byte-identical across the independent roadmap PRs. This PR introduces only the pyLemur public API and its optional dependency.

On Windows, the validation environment used `harmonypy==0.0.10` because harmonypy 2.1.0 failed to build with MSVC. No global pin is imposed.
