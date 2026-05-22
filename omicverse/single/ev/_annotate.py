"""MISEV2023 marker annotation and EV subtyping for single-EV proteomics.

Single-extracellular-vesicle (single-EV) proteomics yields an **EV x protein**
matrix — every observation is one individual vesicle, every variable a protein
marker.  This module ships the MISEV2023 (Welsh *et al.*, *J Extracell Vesicles*
2024) five-category marker framework as built-in panels and uses it to:

* :func:`misev_markers`      — return the built-in MISEV2023 marker panels.
* :func:`classify_markers`   — label every protein in ``var`` with its
  MISEV2023 category (``var['misev_category']``).
* :func:`annotate_ev_subtype`— assign each EV to a tetraspanin-defined surface
  subset (CD9 / CD63 / CD81 single / double / triple-positive / negative).
* :func:`tissue_of_origin`   — score each EV / cluster for cell- and
  tissue-of-origin marker sets.
* :func:`marker_enrichment`  — hypergeometric enrichment of cluster markers
  against an EV-cargo reference (Vesiclepedia / ExoCarta).
* :func:`purity_report`      — a MISEV-style EV purity score (positive EV
  markers present vs contaminant markers present).

All functions are pure Python (numpy / scipy / pandas), keyword-only beyond the
``adata`` argument, and registered with :func:`omicverse._registry.register_function`
under ``category="ev"``.
"""
from __future__ import annotations

from typing import Optional

import re

import numpy as np
import pandas as pd

from ..._registry import register_function


# ---------------------------------------------------------------------------
# Built-in MISEV2023 marker panels
# ---------------------------------------------------------------------------
#: MISEV2023 five-category protein-marker framework.
#:
#: Category keys follow Welsh *et al.* 2024:
#:   * ``transmembrane``  — category 1: transmembrane / lipid-bound EV markers.
#:   * ``cytosolic``      — category 2: cytosolic EV markers recovered in EVs.
#:   * ``contaminant``    — category 3: co-isolated non-EV contaminants.
#:   * ``organelle``      — category 4: intracellular / organelle contaminants.
#:   * ``functional``     — category 5: functional / cell-type / disease markers.
_MISEV2023_PANELS = {
    # Category 1 — transmembrane or GPI-anchored / lipid-bound EV markers.
    "transmembrane": [
        "CD9", "CD63", "CD81", "CD82", "CD37", "CD53",
        "FLOT1", "FLOT2",                       # flotillins
        "ITGA1", "ITGA2", "ITGA4", "ITGB1", "ITGB3",  # integrins
        "ICAM1", "BSG", "SDC1", "SDC4",         # other transmembrane
        "HLA-A", "HLA-B", "HLA-C",              # MHC class I
        "HLA-DRA", "HLA-DRB1", "CD86", "ADAM10",  # MHC class II / surface
        "PTGFRN", "LAMP1", "LAMP2",
    ],
    # Category 2 — cytosolic proteins enriched in / recovered within EVs.
    "cytosolic": [
        "TSG101", "PDCD6IP",                    # ALIX = PDCD6IP
        "SDCBP",                                # syntenin-1
        "CHMP4A", "CHMP4B", "CHMP2A", "VPS4A", "VPS4B",  # ESCRT
        "HSPA8", "HSP90AA1", "HSP90AB1",        # heat-shock / HSP70-90
        "ARRDC1", "ACTB", "GAPDH", "ANXA2", "ANXA5", "ANXA11",
        "RAB5A", "RAB7A", "RAB11A", "RAB27A", "RAB27B", "EHD1", "EHD4",
        "ARF6", "PDCD6", "YWHAE", "YWHAZ",
    ],
    # Category 3 — co-isolated soluble contaminants (lipoproteins, serum).
    "contaminant": [
        "APOA1", "APOA2", "APOB", "APOC3", "APOE",  # lipoproteins
        "ALB",                                       # albumin
        "AHSG", "TF", "HP", "FGA", "FGB", "FGG",     # serum / coagulation
        "C3", "ORM1", "SERPINA1", "A2M",
    ],
    # Category 4 — intracellular / organelle contaminants (non-EV compartments).
    "organelle": [
        "CANX",                                 # calnexin — ER
        "HSPA5",                                # BiP / GRP78 — ER
        "GOLGA2",                               # GM130 — Golgi
        "CYCS",                                 # cytochrome c — mitochondria
        "TOMM20", "VDAC1",                      # mitochondria
        "HIST1H1C", "HIST1H2BK", "H3-3A", "H4C1",  # histones — nucleus
        "LMNA", "LMNB1",                        # nuclear lamina
        "RPL7", "RPS6",                         # ribosome
        "ACTN4", "EEF1A1",
    ],
    # Category 5 — functional / cell-type / disease (parent-cell) markers.
    "functional": [
        "EPCAM",                                # epithelial / tumor
        "ERBB2",                                # HER2
        "EGFR", "MET", "MUC1", "KRT8", "KRT18",
        "L1CAM", "NCAM1", "ENO2", "MAP2", "RBFOX3",  # neuronal
        "ITGA2B", "PF4", "SELP", "GP1BA",       # platelet (CD41=ITGA2B)
        "CD3D", "CD3E", "CD4", "CD8A", "PTPRC", "CD19", "MS4A1",  # immune
        "CD274", "PDCD1LG2",                    # PD-L1 / PD-L2
        "PECAM1", "CDH5", "VWF", "ENG",         # endothelial
        "CD14", "ITGAM", "CD68",                # myeloid
        "PSCA", "FOLH1", "MLANA",               # disease / tumor
    ],
}

#: Human-readable label for each MISEV2023 category key.
_MISEV2023_LABELS = {
    "transmembrane": "Transmembrane/lipid-bound EV marker (MISEV cat.1)",
    "cytosolic": "Cytosolic EV marker (MISEV cat.2)",
    "contaminant": "Co-isolated contaminant (MISEV cat.3)",
    "organelle": "Organelle contaminant (MISEV cat.4)",
    "functional": "Functional/cell-type/disease marker (MISEV cat.5)",
}

#: Cell- / tissue-of-origin marker sets for :func:`tissue_of_origin`.
_TISSUE_MARKERS = {
    "epithelial_tumor": ["EPCAM", "ERBB2", "EGFR", "MUC1", "KRT8", "KRT18",
                         "MET", "PSCA", "FOLH1"],
    "neuronal": ["L1CAM", "NCAM1", "ENO2", "MAP2", "RBFOX3"],
    "platelet": ["ITGA2B", "PF4", "SELP", "GP1BA", "ITGB3"],
    "immune": ["PTPRC", "CD3D", "CD3E", "CD4", "CD8A", "CD19", "MS4A1",
               "CD14", "ITGAM", "CD68"],
    "endothelial": ["PECAM1", "CDH5", "VWF", "ENG", "ICAM1"],
}

#: Tetraspanins used to define EV surface subtypes.
_TETRASPANINS = ("CD9", "CD63", "CD81")


#: CD-antigen / assay-shorthand -> canonical gene symbol. Single-EV panels
#: routinely use CD numbers or antibody-clone shorthand instead of HGNC
#: symbols, so marker matching must resolve these aliases.
_MARKER_ALIASES = {
    "CD107A": "LAMP1", "CD107B": "LAMP2",
    "CD227": "MUC1", "CD318": "CDCP1", "CD340": "ERBB2", "CD309": "KDR",
    "TSG": "TSG101", "ALIX": "PDCD6IP", "SYNTENIN": "SDCBP",
    "DEL1": "EDIL3", "FLOT": "FLOT1",
    "HER2": "ERBB2", "PDL1": "CD274", "PD-L1": "CD274",
}


def _resolve_marker_name(name) -> str:
    """Resolve a single-EV panel protein name to a canonical marker symbol.

    Single-EV assays label one antibody with several DNA barcodes
    (``CD63_C`` / ``CD63_D``, ``CD9_A`` / ``CD9_B``) and use CD-antigen or
    antibody-clone shorthand (``CD107a``, ``CD81sc``) rather than HGNC gene
    symbols. This strips the antibody-barcode suffix and antibody-format
    shorthand and applies a CD-antigen alias table, so the canonical marker
    (``CD63``, ``CD9``, ``LAMP1`` …) is recovered for panel matching.
    """
    s = str(name).strip()
    # strip a trailing single-character antibody-barcode suffix: CD63_C -> CD63
    m = re.match(r"^(.+?)_[A-Za-z0-9]$", s)
    if m:
        s = m.group(1)
    up = s.upper()
    # strip a trailing antibody-format shorthand: CD81SC -> CD81
    for suf in ("SC", "HC", "MAB"):
        if up.endswith(suf) and len(up) > len(suf) + 1:
            up = up[: -len(suf)]
            break
    return _MARKER_ALIASES.get(up, up)


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _require(modname: str, role: str):
    """Lazy-import an optional backend with an actionable error message."""
    import importlib

    try:
        return importlib.import_module(modname)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"{role} needs the '{modname}' package. Install it with: "
            f"pip install {modname}."
        ) from exc


def _dense(adata):
    """Return ``adata.X`` as a dense float numpy array."""
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=float)


def _binarise(X, *, value_type: str, threshold: float):
    """Boolean EV x protein positivity matrix from a value matrix.

    ``'binary'`` value types are taken as-is (non-zero = positive); ``'count'``
    and ``'intensity'`` matrices are thresholded at ``threshold``.
    """
    if value_type == "binary":
        return X > 0
    return X > threshold


def _value_type(adata, override: Optional[str]) -> str:
    """Resolve the EV value type from ``adata.uns['ev']`` or an override."""
    if override is not None:
        return override
    ev = adata.uns.get("ev", {}) if hasattr(adata, "uns") else {}
    return ev.get("value_type", "intensity")


# ---------------------------------------------------------------------------
# misev_markers
# ---------------------------------------------------------------------------
@register_function(
    aliases=["misev_markers", "misev2023_panels", "ev_marker_panels",
             "MISEV标志物", "MISEV2023标志物面板"],
    category="ev",
    description=(
        "Return the built-in MISEV2023 five-category EV protein-marker "
        "framework (transmembrane/lipid-bound, cytosolic, co-isolated "
        "contaminants, organelle contaminants, functional/cell-type/disease "
        "markers). Pass category= for one panel, or flat=True for a "
        "protein -> category lookup."
    ),
    examples=[
        "panels = ov.single.ev.misev_markers()",
        "tetra = ov.single.ev.misev_markers(category='transmembrane')",
        "lookup = ov.single.ev.misev_markers(flat=True)",
    ],
    related=["single.ev.classify_markers", "single.ev.purity_report"],
)
def misev_markers(*, category: Optional[str] = None, flat: bool = False):
    """Built-in MISEV2023 EV protein-marker panels.

    Ships the five-category protein-marker framework defined by MISEV2023
    (Welsh *et al.*, *J Extracell Vesicles* 2024) as offline reference panels.

    Parameters
    ----------
    category
        Return only one panel: ``'transmembrane'`` (cat.1 transmembrane /
        lipid-bound EV markers — tetraspanins, flotillins, integrins,
        MHC-I/II), ``'cytosolic'`` (cat.2 cytosolic EV markers — TSG101,
        ALIX/PDCD6IP, syntenin-1/SDCBP, ESCRT, HSP70/HSPA8), ``'contaminant'``
        (cat.3 co-isolated contaminants — lipoproteins, albumin),
        ``'organelle'`` (cat.4 organelle contaminants — calnexin, GM130,
        cytochrome c, histones), or ``'functional'`` (cat.5 functional /
        cell-type / disease markers). ``None`` returns every panel.
    flat
        If ``True`` return a flat ``{protein: category}`` dict instead of the
        nested ``{category: [proteins]}`` mapping. Ignored when ``category``
        is given.

    Returns
    -------
    dict or list
        ``{category: [proteins]}`` (default), a single ``[proteins]`` list
        (when ``category`` is given) or a ``{protein: category}`` lookup
        (when ``flat=True``).
    """
    if category is not None:
        if category not in _MISEV2023_PANELS:
            raise ValueError(
                f"category must be one of {sorted(_MISEV2023_PANELS)}, "
                f"got {category!r}."
            )
        return list(_MISEV2023_PANELS[category])
    if flat:
        lookup = {}
        for cat, proteins in _MISEV2023_PANELS.items():
            for p in proteins:
                lookup.setdefault(p, cat)
        return lookup
    return {k: list(v) for k, v in _MISEV2023_PANELS.items()}


# ---------------------------------------------------------------------------
# classify_markers
# ---------------------------------------------------------------------------
@register_function(
    aliases=["classify_markers", "annotate_misev_category", "misev_classify",
             "标志物分类", "MISEV类别注释"],
    category="ev",
    description=(
        "Classify every protein in adata.var into its MISEV2023 category "
        "(transmembrane, cytosolic, contaminant, organelle, functional, or "
        "'other' for unmatched proteins). Writes var['misev_category'] and "
        "a human-readable var['misev_category_label']."
    ),
    examples=[
        "ov.single.ev.classify_markers(adata)",
        "adata.var['misev_category'].value_counts()",
    ],
    related=["single.ev.misev_markers", "single.ev.purity_report"],
)
def classify_markers(adata, *, key_added: str = "misev_category",
                     extra_panels: Optional[dict] = None,
                     case_insensitive: bool = True):
    """Classify ``var`` proteins into MISEV2023 marker categories.

    Each protein in ``adata.var`` is matched against the built-in MISEV2023
    panels (:func:`misev_markers`) and labelled with its category; proteins
    that match no panel are labelled ``'other'``.

    Parameters
    ----------
    adata
        EV x protein :class:`anndata.AnnData`; ``var`` index holds protein /
        marker names.
    key_added
        ``var`` column to write the category label into
        (default ``'misev_category'``). A companion ``<key_added>_label``
        column with the full human-readable label is also written.
    extra_panels
        Optional ``{category: [proteins]}`` mapping merged on top of the
        built-in panels — e.g. to add a custom assay-specific panel. New
        category keys are kept as-is.
    case_insensitive
        Match protein names case-insensitively (default ``True``).

    Returns
    -------
    :class:`anndata.AnnData`
        The input ``adata`` with ``var[key_added]`` (and the ``_label``
        companion) filled in-place.
    """
    panels = {k: list(v) for k, v in _MISEV2023_PANELS.items()}
    if extra_panels:
        for cat, proteins in extra_panels.items():
            panels.setdefault(cat, [])
            panels[cat] = list(panels[cat]) + list(proteins)

    # Build the panel lookup keyed by canonical (alias-resolved) marker name
    # so antibody-barcode-suffixed / CD-shorthand var names match correctly.
    lookup = {}
    for cat, proteins in panels.items():
        for p in proteins:
            lookup.setdefault(_resolve_marker_name(p), cat)
            # also keep the raw upper-case form for non-standard panel names
            lookup.setdefault(str(p).upper(), cat)

    cats = [lookup.get(_resolve_marker_name(v),
                       lookup.get(str(v).upper(), "other"))
            for v in adata.var_names]
    labels = {**_MISEV2023_LABELS}
    adata.var[key_added] = pd.Categorical(cats)
    adata.var[f"{key_added}_label"] = [
        labels.get(c, c) for c in cats
    ]
    return adata


# ---------------------------------------------------------------------------
# annotate_ev_subtype
# ---------------------------------------------------------------------------
@register_function(
    aliases=["annotate_ev_subtype", "tetraspanin_subtype", "ev_surface_subset",
             "EV亚型注释", "四跨膜蛋白亚型"],
    category="ev",
    description=(
        "Assign each EV to a tetraspanin-defined surface subset from CD9 / "
        "CD63 / CD81 positivity: single- (CD9-only / CD63-only / CD81-only), "
        "double-, triple-positive or tetraspanin-negative. MISEV2023: "
        "tetraspanins are NOT universal EV markers, so the negative class is "
        "kept explicit. Writes obs['ev_subtype'] and obs['tetraspanin_class']."
    ),
    examples=[
        "ov.single.ev.annotate_ev_subtype(adata)",
        "ov.single.ev.annotate_ev_subtype(adata, threshold=1.0)",
        "adata.obs['ev_subtype'].value_counts()",
    ],
    related=["single.ev.classify_markers", "single.ev.tissue_of_origin",
             "single.ev.colocalization"],
)
def annotate_ev_subtype(adata, *, tetraspanins=_TETRASPANINS,
                        value_type: Optional[str] = None,
                        threshold: float = 0.0,
                        key_added: str = "ev_subtype"):
    """Assign each EV to a tetraspanin-defined surface subset.

    From per-EV CD9 / CD63 / CD81 positivity, every vesicle is placed in one
    of the canonical single-EV surface subsets. MISEV2023 stresses that
    tetraspanins are *not* universal EV markers, so vesicles negative for all
    three are explicitly kept as a ``'tetraspanin-negative'`` class rather
    than being discarded.

    Parameters
    ----------
    adata
        EV x protein :class:`anndata.AnnData`.
    tetraspanins
        The three tetraspanin protein names to test, in the order used to
        build single-positive labels (default ``('CD9', 'CD63', 'CD81')``).
        Tetraspanins absent from ``var`` are treated as all-negative.
    value_type
        ``'count'`` | ``'intensity'`` | ``'binary'``. ``None`` reads
        ``adata.uns['ev']['value_type']`` (falling back to ``'intensity'``).
    threshold
        Positivity cutoff for ``'count'`` / ``'intensity'`` matrices — a
        protein is *present* on an EV when its value exceeds ``threshold``.
        Ignored for ``'binary'`` data.
    key_added
        ``obs`` column for the fine subtype label (default ``'ev_subtype'``).
        A coarse ``'tetraspanin_class'`` column (``positive`` /
        ``negative``) and an integer ``'n_tetraspanins'`` column are also
        written.

    Returns
    -------
    :class:`anndata.AnnData`
        The input ``adata`` with the new ``obs`` columns filled in-place.
    """
    tetraspanins = list(tetraspanins)
    vtype = _value_type(adata, value_type)

    # Map every var to its canonical marker so antibody-barcode-suffixed
    # names (CD9_A / CD9_B, CD63_C / CD63_D) all resolve to one tetraspanin.
    resolved_to_vars: dict = {}
    for v in adata.var_names:
        resolved_to_vars.setdefault(_resolve_marker_name(v), []).append(v)
    n_ev = adata.n_obs
    pos = np.zeros((n_ev, len(tetraspanins)), dtype=bool)
    present = []
    for j, t in enumerate(tetraspanins):
        hits = resolved_to_vars.get(_resolve_marker_name(t), [])
        if not hits:
            present.append(False)
            continue
        present.append(True)
        # an EV is positive for the tetraspanin if ANY of its antibody
        # barcodes is positive (the barcodes report the same antibody).
        for hit in hits:
            col = adata[:, hit].X
            if hasattr(col, "toarray"):
                col = col.toarray()
            col = np.asarray(col, dtype=float).ravel()
            pos[:, j] |= _binarise(col, value_type=vtype, threshold=threshold)

    if not any(present):
        import warnings
        warnings.warn(
            "none of the requested tetraspanins "
            f"{tetraspanins} are present in adata.var — every EV will be "
            "labelled 'tetraspanin-negative'.",
            stacklevel=2,
        )

    n_pos = pos.sum(axis=1)
    labels = np.empty(n_ev, dtype=object)
    for i in range(n_ev):
        k = int(n_pos[i])
        if k == 0:
            labels[i] = "tetraspanin-negative"
        elif k == len(tetraspanins):
            labels[i] = "triple-positive"
        elif k == 1:
            j = int(np.argmax(pos[i]))
            labels[i] = f"{tetraspanins[j]}-only"
        else:
            members = "/".join(tetraspanins[j] for j in np.where(pos[i])[0])
            labels[i] = f"double-positive ({members})"

    adata.obs[key_added] = pd.Categorical(labels)
    adata.obs["n_tetraspanins"] = n_pos.astype(int)
    adata.obs["tetraspanin_class"] = pd.Categorical(
        np.where(n_pos > 0, "positive", "negative")
    )
    return adata


# ---------------------------------------------------------------------------
# tissue_of_origin
# ---------------------------------------------------------------------------
@register_function(
    aliases=["tissue_of_origin", "ev_cell_origin", "parent_cell_score",
             "组织来源", "EV细胞来源评分"],
    category="ev",
    description=(
        "Score each EV (or each cluster) for cell/tissue-of-origin marker "
        "sets — epithelial/tumor, neuronal, platelet, immune, endothelial. "
        "Returns a per-EV (or per-cluster) score table and writes the "
        "argmax call to obs['tissue_of_origin']."
    ),
    examples=[
        "scores = ov.single.ev.tissue_of_origin(adata)",
        "scores = ov.single.ev.tissue_of_origin(adata, groupby='leiden')",
    ],
    related=["single.ev.annotate_ev_subtype", "single.ev.marker_enrichment"],
)
def tissue_of_origin(adata, *, marker_sets: Optional[dict] = None,
                     groupby: Optional[str] = None,
                     value_type: Optional[str] = None,
                     threshold: float = 0.0,
                     key_added: str = "tissue_of_origin",
                     min_score: float = 0.0):
    """Score each EV / cluster for cell- and tissue-of-origin marker sets.

    For every EV (or every cluster when ``groupby`` is given) the mean
    positivity over each origin marker set is computed; the highest-scoring
    set is the assigned tissue of origin.

    Parameters
    ----------
    adata
        EV x protein :class:`anndata.AnnData`.
    marker_sets
        ``{tissue: [proteins]}`` mapping. ``None`` uses the built-in sets
        (``epithelial_tumor``, ``neuronal``, ``platelet``, ``immune``,
        ``endothelial``).
    groupby
        Optional ``obs`` clustering column. When given, scores are averaged
        per cluster and the per-EV call is broadcast from the cluster call;
        otherwise scoring is per individual EV.
    value_type
        ``'count'`` | ``'intensity'`` | ``'binary'``. ``None`` reads
        ``adata.uns['ev']['value_type']``.
    threshold
        Positivity cutoff for ``'count'`` / ``'intensity'`` matrices.
    key_added
        ``obs`` column for the assigned tissue (default
        ``'tissue_of_origin'``).
    min_score
        EVs / clusters whose top score does not exceed ``min_score`` are
        labelled ``'unassigned'``.

    Returns
    -------
    :class:`pandas.DataFrame`
        Per-EV (or per-cluster, indexed by the cluster label) score table —
        one column per marker set plus an ``assigned`` column. The argmax
        call is also written to ``adata.obs[key_added]``.
    """
    sets = marker_sets if marker_sets is not None else _TISSUE_MARKERS
    vtype = _value_type(adata, value_type)
    var_upper = {str(v).upper(): v for v in adata.var_names}

    X = _dense(adata)
    B = _binarise(X, value_type=vtype, threshold=threshold)

    score_cols = {}
    for tissue, proteins in sets.items():
        idx = [adata.var_names.get_loc(var_upper[p.upper()])
               for p in proteins if p.upper() in var_upper]
        if idx:
            score_cols[tissue] = B[:, idx].mean(axis=1)
        else:
            score_cols[tissue] = np.zeros(adata.n_obs)

    per_ev = pd.DataFrame(score_cols, index=adata.obs_names)

    if groupby is not None:
        if groupby not in adata.obs:
            raise KeyError(f"obs column {groupby!r} not found.")
        grp = per_ev.groupby(adata.obs[groupby].values, observed=True).mean()
        top = grp.idxmax(axis=1)
        topval = grp.max(axis=1)
        assigned = top.where(topval > min_score, "unassigned")
        grp["assigned"] = assigned
        ev_call = adata.obs[groupby].map(assigned.to_dict())
        adata.obs[key_added] = pd.Categorical(ev_call.astype(object))
        return grp

    top = per_ev.idxmax(axis=1)
    topval = per_ev.max(axis=1)
    assigned = top.where(topval > min_score, "unassigned")
    per_ev["assigned"] = assigned
    adata.obs[key_added] = pd.Categorical(assigned.astype(object))
    return per_ev


# ---------------------------------------------------------------------------
# marker_enrichment
# ---------------------------------------------------------------------------
@register_function(
    aliases=["marker_enrichment", "ev_cargo_enrichment", "vesiclepedia_enrichment",
             "标志物富集", "EV货物富集"],
    category="ev",
    description=(
        "Hypergeometric enrichment of a cluster's marker proteins against an "
        "EV-cargo reference set (Vesiclepedia / ExoCarta). The reference is "
        "passed in as a protein list or DataFrame so the test runs fully "
        "offline. Returns a per-reference enrichment table with BH-corrected "
        "p-values."
    ),
    examples=[
        "res = ov.single.ev.marker_enrichment(adata, markers=cluster_markers, "
        "reference=vesiclepedia_proteins)",
        "res = ov.single.ev.marker_enrichment(adata, markers=cluster_markers, "
        "reference={'Vesiclepedia': vp_list, 'ExoCarta': ec_list})",
    ],
    related=["single.ev.tissue_of_origin", "single.ev.classify_markers"],
)
def marker_enrichment(adata, *, markers, reference,
                      background: Optional[list] = None):
    """Hypergeometric enrichment of cluster markers vs an EV-cargo reference.

    Tests whether a set of cluster-defining marker proteins overlaps an EV
    cargo reference (e.g. Vesiclepedia or ExoCarta) more than expected by
    chance, using the hypergeometric distribution.

    Parameters
    ----------
    adata
        EV x protein :class:`anndata.AnnData`. Its ``var`` index supplies the
        default statistical background (the assayed protein universe).
    markers
        The query marker set — an iterable of protein names.
    reference
        The EV-cargo reference. Accepts an iterable of protein names, a
        :class:`pandas.DataFrame` (a ``'gene'`` / ``'protein'`` column, else
        the index, is used) or a ``{name: list_or_DataFrame}`` mapping of
        several references — every reference is tested independently. Passing
        the reference in keeps the test fully offline.
    background
        Optional explicit background protein universe. ``None`` uses
        ``adata.var_names``.

    Returns
    -------
    :class:`pandas.DataFrame`
        One row per reference with columns ``n_markers``, ``n_reference``,
        ``n_overlap``, ``expected``, ``fold_enrichment``, ``pvalue`` and the
        BH-corrected ``padj``, sorted by ``pvalue``.
    """
    from scipy.stats import hypergeom

    def _as_set(ref):
        if isinstance(ref, pd.DataFrame):
            for col in ("gene", "protein", "Gene", "Protein", "symbol"):
                if col in ref.columns:
                    return set(str(x).upper() for x in ref[col].dropna())
            return set(str(x).upper() for x in ref.index)
        return set(str(x).upper() for x in ref)

    bg = set(str(x).upper() for x in
             (background if background is not None else adata.var_names))
    query = set(str(x).upper() for x in markers) & bg
    M = len(bg)
    n = len(query)
    if M == 0 or n == 0:
        raise ValueError(
            "empty background or marker set after intersecting with the "
            "background protein universe."
        )

    if isinstance(reference, dict):
        refs = {k: _as_set(v) for k, v in reference.items()}
    else:
        refs = {"reference": _as_set(reference)}

    rows = []
    for name, refset in refs.items():
        ref_in_bg = refset & bg
        N = len(ref_in_bg)
        k = len(query & ref_in_bg)
        expected = n * N / M if M else 0.0
        # P(X >= k) survival function, k-1 for the inclusive tail.
        pval = hypergeom.sf(k - 1, M, N, n) if N > 0 and k > 0 else 1.0
        fold = (k / expected) if expected > 0 else np.nan
        rows.append({
            "reference": name,
            "n_markers": n,
            "n_reference": N,
            "n_overlap": k,
            "expected": expected,
            "fold_enrichment": fold,
            "pvalue": float(pval),
        })

    res = pd.DataFrame(rows).sort_values("pvalue").reset_index(drop=True)
    res["padj"] = _bh_adjust(res["pvalue"].values)
    return res


def _bh_adjust(pvals):
    """Benjamini-Hochberg FDR adjustment of a 1-D array of p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # Enforce monotonicity from the largest p downwards.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(ranked, 0, 1)
    return out


# ---------------------------------------------------------------------------
# purity_report
# ---------------------------------------------------------------------------
@register_function(
    aliases=["purity_report", "ev_purity", "misev_purity_score",
             "纯度报告", "EV纯度评分"],
    category="ev",
    description=(
        "MISEV-style EV preparation purity report: counts and mean signal of "
        "EV-positive markers (transmembrane + cytosolic) versus contaminant "
        "markers (co-isolated + organelle), and a purity score in [0, 1]. "
        "Returns a summary dict; writes per-EV purity to obs when groupby is "
        "not used."
    ),
    examples=[
        "report = ov.single.ev.purity_report(adata)",
        "report = ov.single.ev.purity_report(adata, groupby='sample')",
    ],
    related=["single.ev.classify_markers", "single.ev.misev_markers"],
)
def purity_report(adata, *, groupby: Optional[str] = None,
                  value_type: Optional[str] = None,
                  threshold: float = 0.0,
                  key_added: str = "ev_purity"):
    """MISEV-style EV-preparation purity report.

    Contrasts the EV-positive markers (MISEV categories 1 + 2 — transmembrane
    and cytosolic) against the contaminant markers (categories 3 + 4 —
    co-isolated and organelle) to give a purity score, following the
    MISEV2023 recommendation that an EV preparation be characterised by both
    the EV markers it carries *and* the contaminants it lacks.

    The purity score is the fraction of detected positive signal that comes
    from genuine EV markers::

        purity = positive_signal / (positive_signal + contaminant_signal)

    Parameters
    ----------
    adata
        EV x protein :class:`anndata.AnnData`.
    groupby
        Optional ``obs`` column — when given, a purity score is reported per
        group (e.g. per sample); otherwise one global score plus a per-EV
        ``obs`` column are produced.
    value_type
        ``'count'`` | ``'intensity'`` | ``'binary'``. ``None`` reads
        ``adata.uns['ev']['value_type']``.
    threshold
        Positivity cutoff for ``'count'`` / ``'intensity'`` matrices.
    key_added
        Per-EV ``obs`` column for the purity score (written only when
        ``groupby`` is ``None``; default ``'ev_purity'``).

    Returns
    -------
    dict or :class:`pandas.DataFrame`
        A summary dict (global) with keys ``purity_score``,
        ``n_positive_markers``, ``n_contaminant_markers``,
        ``positive_markers_detected``, ``contaminant_markers_detected`` and
        ``flag`` (``'pass'`` / ``'caution'`` / ``'fail'``); or, when
        ``groupby`` is given, a per-group :class:`pandas.DataFrame` of the
        same columns.
    """
    vtype = _value_type(adata, value_type)
    var_upper = {str(v).upper(): v for v in adata.var_names}

    pos_panel = set(_MISEV2023_PANELS["transmembrane"]) | \
        set(_MISEV2023_PANELS["cytosolic"])
    con_panel = set(_MISEV2023_PANELS["contaminant"]) | \
        set(_MISEV2023_PANELS["organelle"])

    def _idx(panel):
        return [adata.var_names.get_loc(var_upper[p.upper()])
                for p in panel if p.upper() in var_upper]

    pos_idx = _idx(pos_panel)
    con_idx = _idx(con_panel)

    X = _dense(adata)
    B = _binarise(X, value_type=vtype, threshold=threshold)
    # Signal value used in the ratio: raw value for count/intensity, presence
    # for binary; clip negatives so intensity backgrounds cannot bias it.
    sig = B.astype(float) if vtype == "binary" else np.clip(X, 0, None)

    def _flag(score):
        if score >= 0.8:
            return "pass"
        if score >= 0.5:
            return "caution"
        return "fail"

    def _summary(rows_idx):
        sub_sig = sig[rows_idx]
        sub_B = B[rows_idx]
        pos_signal = sub_sig[:, pos_idx].sum() if pos_idx else 0.0
        con_signal = sub_sig[:, con_idx].sum() if con_idx else 0.0
        denom = pos_signal + con_signal
        score = float(pos_signal / denom) if denom > 0 else float("nan")
        pos_det = int((sub_B[:, pos_idx].any(axis=0)).sum()) if pos_idx else 0
        con_det = int((sub_B[:, con_idx].any(axis=0)).sum()) if con_idx else 0
        return {
            "purity_score": score,
            "n_positive_markers": len(pos_idx),
            "n_contaminant_markers": len(con_idx),
            "positive_markers_detected": pos_det,
            "contaminant_markers_detected": con_det,
            "flag": _flag(score) if denom > 0 else "unknown",
        }

    if groupby is not None:
        if groupby not in adata.obs:
            raise KeyError(f"obs column {groupby!r} not found.")
        rows = []
        for g, gi in pd.Series(range(adata.n_obs)).groupby(
                adata.obs[groupby].values, observed=True):
            d = _summary(gi.values)
            d[groupby] = g
            rows.append(d)
        return pd.DataFrame(rows).set_index(groupby)

    # Per-EV purity score written to obs.
    pos_ev = sig[:, pos_idx].sum(axis=1) if pos_idx else np.zeros(adata.n_obs)
    con_ev = sig[:, con_idx].sum(axis=1) if con_idx else np.zeros(adata.n_obs)
    denom_ev = pos_ev + con_ev
    with np.errstate(invalid="ignore", divide="ignore"):
        ev_purity = np.where(denom_ev > 0, pos_ev / denom_ev, np.nan)
    adata.obs[key_added] = ev_purity

    summary = _summary(np.arange(adata.n_obs))
    summary["mean_per_ev_purity"] = float(np.nanmean(ev_purity))
    return summary
