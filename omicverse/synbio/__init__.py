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
    "ECPrediction":          ("._function", "ECPrediction"),
    "protein_embed":         ("._embed", "protein_embed"),
    # ---- Layer C: DNA ----------------------------------------------------
    "codon_optimize":        ("._codon", "codon_optimize"),
    "design_primers":        ("._codon", "design_primers"),
    "CodonResult":           ("._codon", "CodonResult"),
    "PrimerPair":            ("._codon", "PrimerPair"),
    # ---- device helper (handy for users) ---------------------------------
    "resolve_device":        ("._device", "resolve_device"),
}

# submodules that carry @register_function decorators
_REGISTRY_SUBMODULES = (
    "._gem", "._strain", "._ec", "._structure", "._design", "._variant",
    "._stability", "._kcat", "._function", "._embed", "._codon",
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
