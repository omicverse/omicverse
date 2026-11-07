# moscot integration

Adds temporal optimal transport, single-cell to spatial mapping, and slice coupling. Checks convergence by default and returns native moscot problems without overwriting physical coordinates.

Part of [roadmap #989](https://github.com/omicverse/omicverse/issues/989).

## Usage

Install the optional backend with `pip install "omicverse[moscot]"` in an environment supported by the backend. Importing OmicVerse does not import the backend.

```python
import omicverse as ov

transport = ov.single.moscot_temporal(adata, time_key="day", joint_attr="X_pca",
                                      solve_kwargs={"epsilon": 0.1})
```

## Contract and validation

Setup, constructor and training/solver options are passed separately. Inputs and outputs are checked; existing output embeddings require explicit overwrite. Native model/problem objects are returned rather than stored in `AnnData.uns`. Backend failures propagate.

Run `python -m pytest tests/test_backend_validation.py tests/test_roadmap989_moscot.py -q`. Integration tests require the real backend; skipped tests do not establish real execution. Small synthetic training runs check execution and output contracts, not convergence on arbitrary biological data.

The generic validation helper is byte-identical across the independent roadmap PRs. This PR introduces only the moscot public API and its optional dependency.
