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

## Genetic circuits & regulation (CPU)

The regulatory heart of synthetic biology — design gene circuits and simulate
them deterministically (ODEs) or stochastically (Gillespie SSA).

| Function | What it does |
|---|---|
| `genetic_circuit(template)` | Build a circuit — `toggle_switch`, `repressilator`, `logic_gate`, `feed_forward_loop`, or from parts |
| `simulate_circuit(c, method='ode'\|'stochastic')` | Simulate → time × species DataFrame |
| `plot_circuit(df)` | Timecourse figure |
| `rbs_strength(utr)` | Translation-initiation rate (Salis-style SD:anti-SD thermodynamics, ViennaRNA) |
| `promoter_strength(seq)` | σ70 strength from −10/−35 consensus |
| `cai(cds)` / `predict_expression(...)` | Codon Adaptation Index / combined expression estimate |
| `rna_fold` / `rna_accessibility` / `rna_duplex` | RNA structure, site accessibility, hybridisation (ViennaRNA) |

```python
c  = ov.synbio.genetic_circuit("repressilator")
df = ov.synbio.simulate_circuit(c, t_end=300)     # sustained oscillation
ov.synbio.plot_circuit(df)
ov.synbio.rbs_strength("AAAGGAGGACAACATG…").initiation_rate
```

## CRISPR & genome editing (CPU)

| Function | What it does |
|---|---|
| `design_grnas(seq, enzyme='SpCas9')` | PAM scan (SpCas9/SaCas9/Cas12a) + on-target efficiency, ranked |
| `offtarget_search(spacer, background)` | Near-matches + seed-weighted CFD-style scoring |
| `base_editor_window(spacer, editor='ABE'\|'CBE')` | Editable bases in the activity window |
| `hdr_arms(seq, cut_site, insert=…)` | Homology arms / donor for HDR knock-in |

```python
guides = ov.synbio.design_grnas(target_dna, enzyme="SpCas9")
ov.synbio.offtarget_search(guides[0].spacer, genome)
```

## DNA assembly & standards (CPU, Biopython)

| Function | What it does |
|---|---|
| `restriction_map(seq, enzymes)` | Restriction-site map |
| `golden_gate(fragments, enzyme='BsaI')` | Type IIS assembly by 4-nt overhangs → circular construct |
| `gibson_assembly(fragments)` | Join fragments by terminal homology |
| `annotate_construct(seq)` | ORFs + common parts (promoters/RBS/terminators) |
| `read_genbank` / `write_genbank` | Annotated construct I/O |

## Pathway design & libraries

| Function | What it does |
|---|---|
| `reaction_dg(reaction)` | Reaction ΔG'° (built-in baseline / eQuilibrator hook) |
| `max_min_driving_force(reactions, dg0)` | MDF as an exact LP — feasibility + thermodynamic bottleneck |
| `pathway_search(model, target)` | Retrosynthesis: shortest production routes over a GEM |
| `degenerate_codon(aas)` / `saturation_library` / `dms_library` | Combinatorial library design |
| `ml_guided_design(seq)` | ESM-guided combinatorial directed evolution |

```python
res = ov.synbio.max_min_driving_force(reactions, dg0)   # res.mdf, res.bottleneck
paths = ov.synbio.pathway_search(ov.synbio.load_gem("e_coli_core"), "succ_c")
ov.synbio.degenerate_codon(list("ACDEFGHIKLMNPQRSTVWY"))   # -> NNK/NNS
```

> The circuit ODE/SSA, MDF LP, assembly, degenerate-codon and pathway search are
> **exact**; RBS/promoter/gRNA-efficiency/off-target/ΔG are transparent
> biophysical or consensus **baselines** with hooks to the Salis RBS Calculator,
> Azimuth, the Doench-2016 CFD matrix, eQuilibrator and RetroRules for calibrated
> work.

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
