# COMMOT and FlowSig: verified stages

The public runners follow the AnnData-first style of `ov.single.run_liana`.
COMMOT and FlowSig are sequential tasks, not interchangeable methods.
Existing `ov.external.commot` and `ov.external.flowsig` APIs remain available.

## Run COMMOT on one library

Install the runtime with `pip install "omicverse[commot]"`. The vendored
optimal-transport implementation requires Torch even for CPU execution.

```python
import omicverse as ov
import pandas as pd

# Explicit, version-controlled input database; keep a copy with your analysis.
lr = pd.read_csv('ligand_receptor_database.csv')
# Required named columns: ligand, receptor, pathway.
# Use coordinates in a known unit and choose a biologically appropriate cutoff.
out = ov.space.run_commot(
    adata,
    database_name='my_database',
    df_ligrec=lr,
    layer='lognorm',
    spatial_key='spatial',
    dis_thr=150,
    distance_unit='micron',
    inplace=False,
)

comm = ov.space.create_communication_anndata(
    out, 'cell_type', database_name='my_database',
    level='lr', statistic='sum', n_permutations=1000, seed=0,
)
```

The cutoff above is an example, not a universal default. The runner does not
normalize expression or convert coordinate units. `layer=None` explicitly uses
X. Raw inputs, layers and coordinate keys are preserved. Coordinates supplied
to this call define distances; cached `spatial_distance` is not reused.

The dense distance matrix is checked against `max_distance_matrix_mb` before
allocation. This estimate covers only that matrix, not total solver memory.
No automatic downsampling occurs. Multiple libraries must be analyzed
separately, with both observations and image metadata subset consistently.

Output remains in native `commot-<database>-*` AnnData slots. The database
checksum, input layer, coordinate key, units and parameters are recorded in
`uns['commot-<database>-info']['omicverse_run']`. Existing output namespaces
require `overwrite=True`; failed backend calls do not write partial results.

## Interpret summaries correctly

- `level='lr'`, `'pathway'`, and `'total'` select separate result levels.
- Legacy `level='all'` remains available with a warning: those columns overlap.
- Omit `database_name` only when exactly one database metadata entry exists.
- Missing metadata is an error rather than a guess from hyphenated names.
- `layers['means']` is a compatibility name for plotting; its actual statistic
  is recorded in `uns['commot_summary']['statistic']`.
- `sum` is total transport across all sender/receiver pairs; `mean` divides by
  the number of possible pairs. Group sizes are stored in observation metadata.
- P-values are cell-type-label permutations on a fixed communication matrix;
  they are not condition-level tests. Unknown secretion/integrin annotations
  remain the string `Unknown`; do not cast these annotation columns to bool.

The CPU scaling bug fix changes previously unscaled CPU outputs, including
default `auto`. FlowSig inputs affected by similar ligand names such as WNT1
and WNT10 must be reconstructed and their networks rerun.

## Run FlowSig from supplied GEMs

Install the combined stack with `pip install "omicverse[commot,flowsig]"`. GEM training
is separate: supply a validated non-negative `obsm['X_gem']` matrix. This runner
does not substitute NMF for NSF and does not perform implicit Moran filtering.

```python
flow = ov.space.run_flowsig(
    out,
    commot_output_key='commot-my_database',
    gem_expr_key='X_gem',
    layer='lognorm',
    block_key='spatial_block',
    n_bootstraps=500,
    edge_threshold=0.8,
    inplace=False,
)
flow.write_h5ad('communication_and_flow.h5ad')
```

Blocks must be supplied from an explicit spatial blocking design. They are not
biological replicates. The runner retains all constructed flow variables and
uses the backend's `scale_gem_expr=False` mode (which still normalizes GEM
columns). Inflow is COMMOT received signal; outflow is ligand expression.
The backend's bootstrap-index seed schedule is recorded; the runner does not
expose an unsupported seed parameter.

`uns['flowsig_network']['network']` retains `adjacency`,
`adjacency_validated`, and `adjacency_validated_filtered`. The latter stages
apply biological direction constraints and bootstrap-frequency thresholds.
These weights are not p-values, and directed edges are hypotheses rather than
experimentally established causality. Advanced users can use the existing
FlowSig preprocessing and inference functions to choose other workflows.

## Reproduce the test stages

```text
python -m pytest tests/external/test_commot_flowsig_correctness.py -q
python -m pytest tests/space/test_commot_summary_contract.py -q
python -m pytest tests/external/test_flowsig_network.py tests/space/test_communication_runners.py -q
```

CuPy tests require a functioning CUDA installation. The local Windows validation
used a separate environment with causaldag 0.1a163, CuPy CUDA12 14.2.0 and NVIDIA
CUDA runtime/NVRTC wheels. The obsolete `typing` and `dataclasses` backports
pulled by causaldag were removed from that test environment because `typing`
shadowed the Python standard library in a CUDA subprocess. The user's original
analysis environment was not modified. These commands can be run locally;
this change does not add or modify any CI workflow. Real GPU parity is a
separately reported local test, never counted as passed when skipped.
