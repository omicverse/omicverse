"""Real public single-extracellular-vesicle (single-EV) proteomic datasets.

Each loader downloads a dataset asset from the ``omicverse-data`` GitHub
release (``single-ev-v1``) on first use, caches it under ``dir``, and
returns it ready for the ``ov.single.ev`` single-EV proteomics workflow
(EV QC, normalisation, EV-subpopulation clustering, marker-set
enrichment).

Single-EV proteomics produces an **EV x protein** matrix — each row an
individual vesicle, each column a protein/marker.

| Loader | Returns | Modality | Source |
|---|---|---|---|
| :func:`ev_pba` | AnnData (EVs x proteins) | sequencing surface-protein counts | Wu et al., Nat Commun 2019 |
| :func:`ev_masev` | AnnData (EVs x markers) | cyclic-immunofluorescence per-EV | Spitzberg et al., Nat Commun 2023 |
| :func:`ev_marker_reference` | DataFrame (protein reference) | EV marker set | ExoCarta + Vesiclepedia + MISEV2023 |

All datasets are real, published single-EV proteomic data redistributed
at tutorial scale from open sources — the Wu et al. 2019 Proximity
Barcoding Assay deposit on Figshare, the Spitzberg et al. 2023 MASEV
Nature Communications Source Data file, and the open ExoCarta /
Vesiclepedia EV-protein compendia.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .._registry import register_function
from ._datasets import download_data

if TYPE_CHECKING:  # pragma: no cover
    from anndata import AnnData

_RELEASE = (
    "https://github.com/omicverse/omicverse-data/releases/download/single-ev-v1"
)


def _fetch(asset: str, dir: str) -> str:
    """Download ``asset`` from the single-ev-v1 release; return local path."""
    return download_data(f"{_RELEASE}/{asset}", file_path=asset, dir=dir)


@register_function(
    aliases=[
        "ev_pba", "pba", "single_ev_pba", "ev_sequencing", "exosome_pba",
        "单囊泡PBA", "单囊泡蛋白质组", "邻位条形码",
    ],
    category="datasets",
    description=(
        "Real sequencing-based single-extracellular-vesicle surface-protein "
        "count matrix for the primary modality of the single-EV tutorial — "
        "the Proximity Barcoding Assay (PBA) data of Wu et al. (Nat Commun "
        "2019, 10:3854; PMID 31477692, 'Profiling surface proteins on "
        "individual exosomes using a proximity barcoding assay'), obtained "
        "from the authors' Figshare deposit (10.6084/m9.figshare.7963742). "
        "Each row is one individual exosome (a PBA complex identified by its "
        "ComplexTag barcode); each column is one of 40 surface-protein / "
        "marker antibodies; values are next-generation-sequencing read "
        "counts. Curated to a 75,000-EV x 40-protein tutorial subset across "
        "15 samples — 13 cancer / normal cell-line exosome populations "
        "(A549, AGS, BLC21, Daudi, HCT116, HEK293, K562, MM1, MKN45, MKN7, "
        "PC3, SK-N-SH, U87MG) and 2 human-serum exosome samples — keeping "
        "only informative EVs (>=3 total reads) so EV-subpopulation "
        "structure is preserved. Returned as an AnnData: ``.X`` holds raw "
        "per-EV read counts, ``.obs`` carries ``sample`` / ``source`` / "
        "``sample_type`` / ``condition`` / ``complex_tag`` / "
        "``total_counts`` / ``n_proteins``, and ``.uns['ev']['value_type']`` "
        "is ``'count'``. Feed straight into the ``ov.single.ev`` workflow."
    ),
    examples=[
        "adata = ov.datasets.ev_pba()",
        "adata.obs['sample'].value_counts()",
    ],
)
def ev_pba(dir: str = "./data") -> "AnnData":
    """Load the real Wu 2019 PBA single-EV surface-protein count dataset."""
    import anndata as ad
    return ad.read_h5ad(_fetch("ev_pba.h5ad", dir))


@register_function(
    aliases=[
        "ev_masev", "masev", "ev_intensity", "single_ev_intensity",
        "ev_immunofluorescence", "单囊泡免疫荧光", "单囊泡强度",
    ],
    category="datasets",
    description=(
        "Real intensity-modality single-extracellular-vesicle dataset for "
        "the imaging arm of the single-EV tutorial — the MASEV "
        "(Multiplexed Analysis of a Single EV) cyclic-immunofluorescence "
        "data of Spitzberg, Yang et al. (Nat Commun 2023, 14:1239; PMID "
        "36869028), parsed from the article's open Source Data file. Each "
        "row is one individual EV interrogated over 5 cycles of "
        "multi-channel fluorescence staining; each column is one of 16 EV "
        "markers (single markers CD63 / CD47 / TSG101 / MUC1 / CD81 / CD98 "
        "/ syntenin / EGFR / CD9 / CD29 / ALIX / calnexin, plus "
        "co-localisation combination markers CD63-CD81 / CD9-CD81 / "
        "CD63-CD9 / CD81-CD63-CD9); values are binary per-EV positivity "
        "calls. 12,000 EVs across 4 parental cancer cell lines (PANC1, "
        "CAPAN-2, ASPC1, A549), 3,000 EVs each. Returned as an AnnData: "
        "``.X`` holds the 0/1 marker positivity, ``.obs`` carries "
        "``sample`` / ``cell_type`` / ``condition`` / "
        "``n_markers_positive``, ``.var['marker_type']`` flags single vs "
        "combination markers, and ``.uns['ev']['value_type']`` is "
        "``'binary'``."
    ),
    examples=[
        "adata = ov.datasets.ev_masev()",
        "adata.var['marker_type'].value_counts()",
    ],
)
def ev_masev(dir: str = "./data") -> "AnnData":
    """Load the real Spitzberg 2023 MASEV single-EV per-EV marker dataset."""
    import anndata as ad
    return ad.read_h5ad(_fetch("ev_masev.h5ad", dir))


@register_function(
    aliases=[
        "ev_marker_reference", "ev_markers", "exocarta", "vesiclepedia",
        "ev_marker_set", "misev_markers", "囊泡标志物参考", "EV标志物",
    ],
    category="datasets",
    description=(
        "Curated extracellular-vesicle protein-marker reference for the "
        "marker-set-enrichment arm of the single-EV tutorial — built from "
        "the two open community EV-protein compendia ExoCarta and "
        "Vesiclepedia 2024 (microvesicles.org / exocarta.org) plus the "
        "MISEV2023 minimal-information marker panels. 7,243 human EV "
        "proteins, each ranked by the number of independent EV studies it "
        "appears in. Returned as a pandas DataFrame with columns "
        "``gene_symbol``, ``exocarta_studies``, ``vesiclepedia_studies``, "
        "``total_studies``, ``rank``, ``misev2023_category`` "
        "(transmembrane / cytosolic / intracellular_control / "
        "non_ev_contaminant), ``is_misev2023_marker`` and "
        "``is_core_ev_marker`` (the canonical tetraspanin / ESCRT panel "
        "CD9 / CD63 / CD81 / TSG101 / ALIX / syntenin / flotillins / "
        "HSPA8). Use as a marker set to score / annotate EV "
        "subpopulations or to QC EV-vs-contaminant signal."
    ),
    examples=[
        "ref = ov.datasets.ev_marker_reference()",
        "ref[ref['is_core_ev_marker']]",
    ],
)
def ev_marker_reference(dir: str = "./data") -> pd.DataFrame:
    """Load the curated EV protein-marker reference (ExoCarta + Vesiclepedia)."""
    return pd.read_csv(_fetch("ev_marker_reference.tsv.gz", dir), sep="\t")
