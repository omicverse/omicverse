r"""Synthetic biology for omicverse — ``ov.synbio``.

A self-contained, three-layer design-build stack that bridges **metabolism**,
**protein/enzyme engineering**, and **DNA**:

Layer A — metabolic networks (CPU, COBRApy)
    ``load_gem`` · ``fba`` · ``pfba`` · ``fva`` · ``single_gene_deletion`` ·
    ``double_gene_deletion`` · ``strain_design`` · ``production_envelope`` ·
    ``ec_model`` · ``apply_kcat``

Layer B — proteins & enzymes (GPU-capable, ESM / ProteinMPNN)
    ``predict_structure`` (ESMFold) · ``inverse_design`` (ProteinMPNN) ·
    ``denovo_backbone`` (RFdiffusion, optional) · ``variant_effect`` (ESM
    zero-shot saturation scan) · ``stability_ddg`` (ProteinMPNN ΔΔG proxy) ·
    ``enzyme_kcat`` (DLKcat / baseline) · ``enzyme_function`` (CLEAN / k-NN) ·
    ``protein_embed`` (ESM-2)

Layer C — DNA (CPU, DNAchisel / primer3)
    ``codon_optimize`` · ``design_primers``

The hinge — A↔B coupling
------------------------
Predict a turnover number from an enzyme sequence, push it into a genome-scale
model as an enzyme-capacity constraint, and re-solve the achievable yield::

    import omicverse as ov
    m   = ov.synbio.load_gem("e_coli_core")
    k   = ov.synbio.enzyme_kcat(enzyme_seq, substrate_smiles)   # protein layer
    ecm = ov.synbio.ec_model(m, {"PFK": k.kcat})                # metabolic layer
    sol = ov.synbio.fba(ecm)                                    # yield recomputed

Dependencies are **optional**: ``import omicverse`` (and even
``import omicverse.synbio``) do no heavy work.  Every backend is gated behind an
actionable ``ImportError`` — install with ``pip install 'omicverse[synbio]'``.
Heavy GPU models raise clearly on CPU instead of hanging; set
``OMICOS_SYNBIO_DEVICE`` or pass ``device=`` to override, and
``OMICOS_SYNBIO_WEIGHTS`` to relocate downloaded weights.
"""
from __future__ import annotations

import importlib as _importlib

# public name -> (submodule, attribute)
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # ---- Layer A: metabolic networks -------------------------------------
    "load_gem":              ("._gem", "load_gem"),
    "fba":                   ("._gem", "fba"),
    "pfba":                  ("._gem", "pfba"),
    "fva":                   ("._gem", "fva"),
    "single_gene_deletion":  ("._gem", "single_gene_deletion"),
    "double_gene_deletion":  ("._gem", "double_gene_deletion"),
    "strain_design":         ("._strain", "strain_design"),
    "production_envelope":   ("._strain", "production_envelope"),
    "StrainDesignResult":    ("._strain", "StrainDesignResult"),
    "ec_model":              ("._ec", "ec_model"),
    "apply_kcat":            ("._ec", "apply_kcat"),
    # ---- Layer B: proteins & enzymes -------------------------------------
    "predict_structure":     ("._structure", "predict_structure"),
    "StructurePrediction":   ("._structure", "StructurePrediction"),
    "inverse_design":        ("._design", "inverse_design"),
    "denovo_backbone":       ("._design", "denovo_backbone"),
    "DesignedSequence":      ("._design", "DesignedSequence"),
    "variant_effect":        ("._variant", "variant_effect"),
    "stability_ddg":         ("._stability", "stability_ddg"),
    "enzyme_kcat":           ("._kcat", "enzyme_kcat"),
    "KcatPrediction":        ("._kcat", "KcatPrediction"),
    "enzyme_function":       ("._function", "enzyme_function"),
    "plot_ec_prediction":    ("._function", "plot_ec_prediction"),
    "ECPrediction":          ("._function", "ECPrediction"),
    "protein_embed":         ("._embed", "protein_embed"),
    # ---- Layer C: DNA ----------------------------------------------------
    "codon_optimize":        ("._codon", "codon_optimize"),
    "design_primers":        ("._codon", "design_primers"),
    "CodonResult":           ("._codon", "CodonResult"),
    "PrimerPair":            ("._codon", "PrimerPair"),
    # ---- visualisation ---------------------------------------------------
    "view_structure":            ("._plot", "view_structure"),
    "plot_variant_effect":       ("._plot", "plot_variant_effect"),
    "plot_enzyme_yield_response": ("._plot", "plot_enzyme_yield_response"),
    "plot_production_envelope":  ("._plot", "plot_production_envelope"),
    # ---- genetic circuits & regulation -----------------------------------
    "genetic_circuit":       ("._circuit", "genetic_circuit"),
    "simulate_circuit":      ("._circuit", "simulate_circuit"),
    "plot_circuit":          ("._circuit", "plot_circuit"),
    "GeneticCircuit":        ("._circuit", "GeneticCircuit"),
    "toggle_switch":         ("._circuit", "toggle_switch"),
    "repressilator":         ("._circuit", "repressilator"),
    "logic_gate":            ("._circuit", "logic_gate"),
    "feed_forward_loop":     ("._circuit", "feed_forward_loop"),
    "rbs_strength":          ("._expression", "rbs_strength"),
    "promoter_strength":     ("._expression", "promoter_strength"),
    "cai":                   ("._expression", "cai"),
    "predict_expression":    ("._expression", "predict_expression"),
    "rna_fold":              ("._rna", "rna_fold"),
    "rna_accessibility":     ("._rna", "rna_accessibility"),
    "rna_duplex":            ("._rna", "rna_duplex"),
    "gc_content":            ("._rna", "gc_content"),
    # ---- CRISPR & genome editing -----------------------------------------
    "design_grnas":          ("._crispr", "design_grnas"),
    "offtarget_search":      ("._crispr", "offtarget_search"),
    "base_editor_window":    ("._crispr", "base_editor_window"),
    "hdr_arms":              ("._crispr", "hdr_arms"),
    "plot_grna_efficiency":  ("._crispr", "plot_grna_efficiency"),
    "plot_offtargets":       ("._crispr", "plot_offtargets"),
    "Guide":                 ("._crispr", "Guide"),
    # ---- DNA assembly & standards ----------------------------------------
    "restriction_map":       ("._assembly", "restriction_map"),
    "golden_gate":           ("._assembly", "golden_gate"),
    "gibson_assembly":       ("._assembly", "gibson_assembly"),
    "annotate_construct":    ("._assembly", "annotate_construct"),
    "write_genbank":         ("._assembly", "write_genbank"),
    "read_genbank":          ("._assembly", "read_genbank"),
    # ---- pathway thermodynamics & retrosynthesis -------------------------
    "reaction_dg":           ("._thermo", "reaction_dg"),
    "max_min_driving_force": ("._thermo", "max_min_driving_force"),
    "plot_driving_forces":   ("._thermo", "plot_driving_forces"),
    "pathway_search":        ("._retro", "pathway_search"),
    "Pathway":               ("._retro", "Pathway"),
    # ---- combinatorial libraries & directed evolution --------------------
    "degenerate_codon":      ("._library", "degenerate_codon"),
    "saturation_library":    ("._library", "saturation_library"),
    "dms_library":           ("._library", "dms_library"),
    "ml_guided_design":      ("._library", "ml_guided_design"),
    # ---- device helper (handy for users) ---------------------------------
    "resolve_device":        ("._device", "resolve_device"),
}

# submodules that carry @register_function decorators
_REGISTRY_SUBMODULES = (
    "._gem", "._strain", "._ec", "._structure", "._design", "._variant",
    "._stability", "._kcat", "._function", "._embed", "._codon", "._plot",
    "._circuit", "._expression", "._rna", "._crispr", "._assembly",
    "._thermo", "._retro", "._library",
)


def _hydrate_registry() -> None:
    """Force-import every ``@register_function``-bearing submodule so the global
    registry sees ``ov.synbio`` at export time.  Called from
    :func:`omicverse._registry._hydrate_registry_for_export`.  Submodules whose
    optional deps are missing still register their functions, because the heavy
    imports live *inside* the function bodies, not at module top level."""
    for mod in _REGISTRY_SUBMODULES:
        try:
            _importlib.import_module(mod, __name__)
        except Exception:
            continue


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module = _importlib.import_module(module_path, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(list(globals().keys()) + list(_LAZY_ATTRS.keys())))


__version__ = "0.1.0"
__all__ = list(_LAZY_ATTRS.keys())
