# `ov.flow` — Flow, spectral and mass cytometry

Cytometry is a **modality**, not a sub-analysis of single-cell. An event is a cell, but the
display scalings, the compensation model and above all the sequential gating hierarchy have no
analogue anywhere else in omicverse.

Reading the file is **not** in this namespace — that is [`ov.io.read_fcs`](#reading-a-file),
because it is I/O.

```python
import omicverse as ov

adata = ov.io.read_fcs('sample.fcs')     # events x channels
ov.flow.compensate(adata)                # undo spillover
logicle = ov.flow.make_transform('logicle', t=262144, w=0.5, m=4.5, a=0.0)

gs = (ov.flow.GatingStrategy('PBMC')
      .add_gate(ov.flow.RectangleGate(name='Cells', dims=('FSC-A', 'SSC-A'),
                                      bounds=((3e4, 1.1e5), (1.5e4, 6e4))))
      .add_gate(ov.flow.PolygonGate(name='Singlets', dims=('FSC-A', 'FSC-H'),
                                    vertices=verts), parent='Cells')
      .add_gate(ov.flow.RectangleGate(name='CD3+', dims=('CD3',), bounds=((0.45, None),),
                                      transforms={'CD3': logicle}), parent='Singlets'))

res = gs.apply(adata)
res.stats()
```

Full worked examples live with the rest of the user documentation, in the
`omicverse_guide` submodule: [`Tutorials-flow`](../omicverse_guide/docs/Tutorials-flow/index.md)
— four notebooks covering reading and compensation, gating, Gating-ML
interchange, and FlowSOM. Each writes its own synthetic `.fcs`, so none of them
needs data of your own.

## Reading a file

`ov.io.read_fcs` keeps the two things a cytometry file exists to carry:

| | |
|---|---|
| **`$PnN` vs `$PnS`** | the DETECTOR (`FITC-A`) versus the MARKER on it (`CD3`), as separate `var` columns. The same antibody sits on a different detector between panels, so conflating them is the classic FCS mistake. |
| **`$SPILLOVER`** | parsed into a labelled square DataFrame in `uns['fcs']['spillover']`, not left as the raw comma string other readers return. |

`var` is indexed by the marker where the file names one, matching
[pytometry](https://github.com/scverse/pytometry)'s schema, so the result feeds
`pytometry.pp.compensate` and friends unchanged.

## Display transforms

A linear axis cannot show compensated fluorescence — after compensation a real population sits
partly **below zero** — and a log axis cannot represent that at all. The field's answer is a
family of biexponential scalings.

| | |
|---|---|
| `logicle` | Parks/Roederer/Moore 2006. The default for fluorescence. |
| `hyperlog` | Bagwell 2005. Strictly increasing with no branch in the positive region. |
| `asinh` | The CyTOF workhorse — cofactor 5 for mass cytometry, ~150 for fluorescence. |
| `log`, `linear` | GatingML's `flog` / `flin`. `log` is `NaN` at and below zero, which is the honest answer. |

Transforms are **objects**, not functions: `apply` / `inverse` / `ticks` / `to_dict`. A gate is
meaningless without the scaling its vertices were drawn on, so storing the transform beside the
gate — and round-tripping it through JSON — is what lets a strategy be saved and reapplied.

These are derived from the GatingML 2.0 and Parks 2006 specifications rather than wrapped, and
verified bit-for-bit (**4.4e-16**, twice float64 epsilon) against the reference C implementation
as a black-box oracle across 16 parameter sets including `A ≠ 0`. See `_transforms.py` for the
licensing and numerics argument, and `tests/flow/test_transforms.py` for the parity data.

## Compensation

Observed values are `true @ S`, so recovering the truth is `observed @ inv(S)`.

```python
ov.flow.compensate(adata)                                  # the file's own $SPILLOVER
ov.flow.compensate(adata, spillover=from_controls)         # a matrix you measured
ov.flow.compensate(adata, spillover=M, matrix_type='compensation')   # already inverted
```

* Only the detectors the matrix names are touched — scatter and time are not fluorescence.
* A **bare numpy array is refused**: an unlabelled matrix cannot say which detector each row is,
  and guessing the order is how channels get compensated against the wrong dye.
* **Compensating twice is refused.** It is silently destructive — the numbers stay plausible —
  so it has to be explicit.

`spillover_spreading_matrix` is the panel-design diagnostic: compensation removes the *mean* of
the spillover but cannot remove the variance the extra photons added, which is why a dim
population can be unresolvable on a detector a bright dye spills into no matter how good the
compensation is.

## Gates and the strategy tree

Five gate types — `RectangleGate`, `PolygonGate`, `EllipsoidGate`, `QuadrantGate`, `BooleanGate`
— following Gating-ML 2.0 geometry so they round-trip without a second representation.

The **tree belongs to the analysis, not to a sample**. That is the whole reason this is a module:
it is what makes applying one strategy to ninety files possible, and two strategies diffable.

* Rectangle bounds are half-open `[min, max)`, so adjacent gates partition rather than
  double-count the shared edge. `None` is unbounded — how "CD3 positive" is expressed without
  inventing a ceiling.
* Quadrants are **one gate with several outputs**, not four rectangles: they share their
  dividers, and four independent gates drift apart the first time someone edits one.
* Booleans are evaluated by the strategy and deliberately **not** parent-restricted — restricting
  would silently intersect with the parent and change the meaning of `NOT`.

`stats()` reports `count`, `parent`, `parent_count`, `freq_parent`, `freq_total` and `low_n`.
The parent is named in its own column because *"62% of CD3+"* is meaningless without its
denominator, and `low_n` flags populations where the percentage is noise — a frequency off 30
events has a 95% CI of roughly ±18 points.

`to_dict()` is plain JSON, and is the seam an interactive gate editor works against.

## Interchange — Gating-ML 2.0

```python
ov.flow.write_gatingml(gs, 'strategy.xml')
gs2 = ov.flow.read_gatingml('strategy.xml')
```

There is a trap in the spec worth knowing about. `gating:id` is typed `xs:ID` — an XML NCName —
and **almost no real gate name is one**:

| gate name | valid `xs:ID`? |
|---|---|
| `CD3+` | no |
| `CD4+CD8-` | no |
| `Live cells` | no |
| `CD45RA+CCR7+` | no |
| `4-1BB+` | no |
| `Singlets` | yes |

`+` is the single most common character in a cytometry gate name. `ov.flow` maps display names
onto generated valid ids and carries the human name in `custom_info`, restoring it on read.
`+`/`-` become `pos`/`neg` rather than being stripped, because `CD4+CD8-` and `CD4-CD8+` are
**opposite populations** and silently merging them would be the worst thing an interchange format
could do. `write_gatingml` re-parses what it just wrote, because *"writes fine, will not read
back"* is exactly the failure being avoided.

## Plots

Every other part of this module can be checked by reading a number; a gate
cannot. *"CD3+ is 69.5% of Live"* is not evidence that the CD3 gate is in the
right place — the only thing that settles it is seeing the boundary lying on the
population, **on the scale the boundary was drawn on**.

```python
ov.flow.biaxial(adata, 'CD4', 'CD8', strategy=gs, gates=['CD4/CD8'],
                result=res, population='CD3+')     # density + gate + the four %
ov.flow.histogram(adata, 'CD3', gates=[cd3], result=res)
ov.flow.backgate(adata, 'FSC-A', 'SSC-A', result=res, population='CD4 T')
ov.flow.hierarchy(gs, result=res)
ov.flow.spillover_heatmap(adata)
ov.flow.flowsom_heatmap(adata)
```

* **The axis is the scale.** Events are drawn in scale space using the transform
  the *gate* carries, and the ticks are re-labelled into data units via
  `Transform.ticks`. The plot cannot disagree with the mask, because it reads the
  scaling off the gate rather than being told it a second time.
* **The events are the gate's events.** Channel lookup goes through the same
  accessor `GatingStrategy.apply` uses, so a marker/detector alias resolves
  identically in the picture and in the mask.
* **The percentages come from the `GatingResult`**, not from re-running the gate
  on whatever is being displayed. Those differ — a child is restricted by its
  parent — and the number a reader should see is the one the analysis produced.
* Gate dimensions are matched **by name**: a polygon stored as `('CD8','CD4')`
  drawn on a CD4 × CD8 plot is transposed, not drawn mirrored.
* Colour is per-event binned density, drawn low-to-high so rare events survive;
  a plain scatter at 1e5 events is a black blob in which every population looks
  equally dense.

`backgate` is the one that catches a bad gate: a "CD4 T" population scattered
through the debris corner of FSC × SSC is not a T-cell population, however clean
the CD4 × CD8 plot looked, and no statistics table will say so.

## Unsupervised — FlowSOM

```python
ov.flow.flowsom(adata, n_clusters=12, markers=['CD3', 'CD4', 'CD8'], layer='logicle')
```

Pure numpy, no R or Java. Cluster on **transformed** values and on the markers only — including
scatter lets it dominate and the result is shape clusters rather than phenotypes.

Clustering **complements** gating rather than replacing it: a gate is auditable and a reviewer can
argue with it; a metacluster is neither. Use it to find populations a strategy missed, then draw
the gate.

`ov.single.ev.flowsom` and `ov.flow.flowsom` are two entry points onto **one** implementation
(`ov.flow._som`), differing only in which layer they default to and whether the result lands in
`uns['ev']` or `uns['flow']`.

## Installing

The reader needs `flowio` (BSD-3, pure Python, numpy only). Everything in `ov.flow` itself is
numpy + pandas.

```bash
pip install omicverse[cytometry]
```

## Function discovery

Everything is registered, so `ov.find_function` and any omicOS agent can reach it:

```python
ov.find_function('荧光补偿')     # -> ov.flow.compensate
ov.find_function('gating')       # -> ov.flow.GatingStrategy
```

## Deliberately not here

* **A second FCS reader.** `ov.io.read_fcs` owns the format.
* **EV proteomics.** `ov.single.ev` owns `uns['ev']`, the MISEV panels and the vesicle vocabulary.
* **Spectral unmixing.** SpectroFlo already ships pre-hoc control QC; the unmet slice is post-hoc
  audit, which is a different feature.
* **`.wsp` export.** FlowJo v11's native format is `.flowjo` and v11↔v10 conversion is lossy in
  both directions. Import is a bridge; export would be a bridge to nowhere.
* **A blanket `auto_qc_gates()`.** CyTOF has no FSC/SSC, `$TIMESTEP` is optional, and viability is
  an arbitrary `$PnS` string with panel-dependent polarity — a five-step auto-gater would
  `KeyError` or, worse, silently mis-gate.
