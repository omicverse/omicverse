# `ov.synbio` — Synthetic Biology

`ov.synbio` is a self-contained, three-layer design-build stack for synthetic
biology inside OmicVerse. It bridges **metabolism**, **protein/enzyme
engineering**, and **DNA** — and, crucially, lets the layers talk to each other.

```bash
pip install "omicverse[synbio]"
```

All imports are lazy and every backend is gated behind an actionable
`ImportError`, so `import omicverse` never requires any synbio dependency. GPU
protein models raise a clear error on CPU instead of hanging; set
`OMICOS_SYNBIO_DEVICE` or pass `device=` to override, and
`OMICOS_SYNBIO_WEIGHTS` to relocate downloaded model weights (default
`~/.omicverse/synbio_weights`).

---

## Layer A — metabolic networks (CPU, COBRApy)

Constraint-based reconstruction and analysis on genome-scale metabolic models
(GEMs). The currency of this layer is a `cobra.Model`.

| Function | What it does |
|---|---|
| `load_gem(path_or_id)` | Load a GEM from a local SBML/JSON or a BiGG id (`e_coli_core`, `iML1515`, `iCW773`, …) |
| `fba(model, objective=None)` | Flux Balance Analysis → growth/production + fluxes |
| `pfba(model)` | Parsimonious FBA (minimal total flux at the optimum) |
| `fva(model, fraction_of_optimum=…)` | Flux Variability Analysis → per-reaction min/max flux |
| `single_gene_deletion` / `double_gene_deletion` | Knockout scans → post-KO growth |
| `strain_design(model, target)` | Ranked over-expression (FSEOF) + growth-coupled knockout targets |
| `production_envelope(model, target)` | Growth-vs-product-yield trade-off curve |
| `ec_model(model, kcat_map)` / `apply_kcat` | Enzyme-constrained (GECKO-light) model from `{reaction: kcat}` |

```python
import omicverse as ov
m = ov.synbio.load_gem("e_coli_core")
ov.synbio.fba(m).objective_value                 # 0.874 /h
res = ov.synbio.strain_design(m, "EX_succ_e")    # succinate over-production
res.amplify.head()    # FSEOF: FRD7, FUM, PPC …
res.knockout.head()   # growth-coupled KOs (e.g. PFL under anaerobic)
```

## Layer B — proteins & enzymes (GPU-capable)

Public open models: ESM-2 / ESM-1v / ESMFold (`facebookresearch/esm`),
ProteinMPNN (`dauparas/ProteinMPNN`, auto-cloned on first use), with hooks for
DLKcat, ThermoMPNN, CLEAN and RFdiffusion.

| Function | Model | GPU | Input → Output |
|---|---|---|---|
| `predict_structure(seq)` | ESMFold | required | sequence → PDB + pLDDT |
| `inverse_design(pdb)` | ProteinMPNN | CPU ok | backbone → candidate sequences |
| `denovo_backbone(spec)` | RFdiffusion | required | constraints → new backbone (optional) |
| `variant_effect(seq, mutations=None)` | ESM-1v/2 | recommended | zero-shot saturation ΔΔ scores |
| `stability_ddg(pdb, mutations=None)` | ProteinMPNN ΔΔG proxy | CPU ok | mutation → destabilisation score |
| `enzyme_kcat(seq, substrate_smiles)` | DLKcat / ESM baseline | recommended | enzyme+substrate → k_cat (1/s) |
| `enzyme_function(seq)` | CLEAN / ESM k-NN | recommended | sequence → EC number |
| `protein_embed(seqs)` | ESM-2 | recommended | sequences → embedding vectors |

```python
X    = ov.synbio.protein_embed([seq], model="esm2_t33_650M")   # (1, 1280)
scan = ov.synbio.variant_effect(seq)                            # full DMS table
pred = ov.synbio.predict_structure(seq)                         # ESMFold on GPU
pred.save("mypro.pdb"); pred.mean_plddt
designs = ov.synbio.inverse_design("mypro.pdb", num_sequences=8)
ddg = ov.synbio.stability_ddg("mypro.pdb")                      # ΔΔG proxy
```

> **Honesty note.** `enzyme_kcat`'s default engine and `enzyme_function`'s
> default `knn` are dependency-light, sequence-sensitive **baselines** — good
> for wiring and relative comparison, not calibrated quantitative prediction.
> For quantitative results plug in DLKcat / UniKP weights (`engine="dlkcat"`)
> or CLEAN (`engine="clean"`). `stability_ddg`'s default is the well-established
> ProteinMPNN log-likelihood ΔΔG proxy; `engine="thermompnn"` uses calibrated
> weights when available.

## Layer C — DNA (CPU)

| Function | Library | Input → Output |
|---|---|---|
| `codon_optimize(seq, host=…)` | DNAchisel | AA/DNA → host-optimised DNA |
| `design_primers(seq)` | primer3 | DNA → validated primer pairs |

```python
opt = ov.synbio.codon_optimize("MKTAYIAK…", host="e_coli")   # GC/enzyme-site aware
primers = ov.synbio.design_primers(opt.sequence)
```

---

## The moat — A↔B coupling

The feature that a metabolism-only or protein-only tool can't offer: **edit an
enzyme, and the metabolic network re-solves its yield.**

```python
m   = ov.synbio.load_gem("e_coli_core")
k   = ov.synbio.enzyme_kcat(enzyme_seq, substrate_smiles)   # protein layer → kcat
ecm = ov.synbio.ec_model(m, {"PFK": k.kcat})                # metabolic layer
sol = ov.synbio.fba(ecm)                                    # yield recomputed
```

Compared under a **fixed protein budget**, a faster enzyme variant (higher
`k_cat`) relaxes the enzyme-capacity constraint and raises the attainable
growth/yield — a quantitative link from sequence to phenotype. See
`tests/synbio_coupling_demo.py` and the tutorial notebook.

---

## Function discovery (omicOS registry)

Every function is registered under the `synthetic_biology` category with
Chinese + English aliases:

```python
ov.find_function("酶")                              # fuzzy search
ov.find_function("代谢模型")
ov.list_functions(category="synthetic_biology")     # list all synbio functions
```
