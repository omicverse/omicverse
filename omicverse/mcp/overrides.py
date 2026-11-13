"""
Manual overrides for manifest entries, schemas, and naming.

This module is the single place for:
- Resolving duplicate registrations (e.g. ``convert_to_pandas``)
- Forcing specific tool names
- Patching JSON Schemas for kwargs-heavy or weakly-typed functions
- Controlling rollout phase whitelists
"""

from __future__ import annotations

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Phase whitelists — only these full_names are enabled per phase
# ---------------------------------------------------------------------------

PHASE_WHITELIST: Dict[str, List[str]] = {
    "P0": [
        "omicverse.io.single._read.read",
        "omicverse.utils._data.read",
        "omicverse.datasets._datasets.pbmc3k",
        "omicverse.datasets._datasets.pbmc8k",
        "omicverse.datasets._datasets.seqfish",
        "omicverse.pp._qc.qc",
        "omicverse.pp._qc.filter_cells",
        "omicverse.pp._qc.filter_genes",
        "omicverse.pp._normalization.log1p",
        "omicverse.pp._preprocess.highly_variable_genes",
        "omicverse.pp._preprocess.scale",
        "omicverse.pp._preprocess.pca",
        "omicverse.pp._preprocess.neighbors",
        "omicverse.pp._preprocess.umap",
        "omicverse.pp._preprocess.leiden",
        "omicverse.pp._preprocess.louvain",
    ],
    "P0.5": [
        "omicverse.single._markers.find_markers",
        "omicverse.single._markers.get_markers",
        "omicverse.single._cosg.cosg",
        "omicverse.single._scgsea.pathway_enrichment",
        "omicverse.single._scgsea.pathway_enrichment_plot",
        "omicverse.pl._single.embedding",
        "omicverse.pl._single.umap",
        "omicverse.pl._single.tsne",
        "omicverse.pl._violin.violin",
        "omicverse.pl._dotplot.dotplot",
        "omicverse.pl._dotplot.markers_dotplot",
        "omicverse.pl._dotplot.rank_genes_groups_dotplot",
        "omicverse.pl._heatmap.marker_heatmap",
    ],
    "P2": [
        "omicverse.bulk._Deseq2.pyDEG",
        "omicverse.single._anno.pySCSA",
        "omicverse.single._metacell.MetaCell",
        "omicverse.single._deg_ct.DCT",
        "omicverse.single._scenic.SCENIC",
        "omicverse.utils._cluster.LDA_topic",
    ],
}

# ---------------------------------------------------------------------------
# Name overrides — full_name → forced canonical tool name
# ---------------------------------------------------------------------------

NAME_OVERRIDES: Dict[str, str] = {
    # gpu/cpu init live under core, not utils
    "omicverse._settings.gpu_init": "ov.core.gpu_init",
    "omicverse._settings.cpu_gpu_mixed_init": "ov.core.cpu_gpu_mixed_init",
    # read() is implemented in io.single but exposed as a core utility tool
    "omicverse.io.single._read.read": "ov.utils.read",
    "omicverse.utils._data.read": "ov.utils.read",
}

# ---------------------------------------------------------------------------
# Manifest overrides — full_name → partial manifest field patches
# ---------------------------------------------------------------------------

MANIFEST_OVERRIDES: Dict[str, dict] = {
    # read() creates a new adata_id rather than consuming one
    "omicverse.utils._data.read": {
        "state_contract": {
            "needs_session": False,
            "session_inputs": [],
            "mutates_state": False,
            "returns_updated_adata_ref": True,
        },
        "return_contract": {
            "primary_output": "object_ref",
            "secondary_outputs": ["json"],
            "emits_artifacts": False,
            "artifact_types": [],
        },
        "risk_level": "low",
    },
    "omicverse.io.single._read.read": {
        "state_contract": {
            "needs_session": False,
            "session_inputs": [],
            "mutates_state": False,
            "returns_updated_adata_ref": True,
        },
        "return_contract": {
            "primary_output": "object_ref",
            "secondary_outputs": ["json"],
            "emits_artifacts": False,
            "artifact_types": [],
        },
        "risk_level": "low",
    },
    # get_markers returns a table, does not mutate
    "omicverse.single._markers.get_markers": {
        "state_contract": {
            "needs_session": True,
            "session_inputs": ["adata_id"],
            "mutates_state": False,
            "returns_updated_adata_ref": False,
        },
        "return_contract": {
            "primary_output": "table",
            "secondary_outputs": [],
            "emits_artifacts": False,
            "artifact_types": [],
        },
    },
}

# ---------------------------------------------------------------------------
# Schema overrides — full_name → JSON Schema patches (merged into generated)
# ---------------------------------------------------------------------------

SCHEMA_OVERRIDES: Dict[str, dict] = {
    # qc() accepts **kwargs — expose common params explicitly
    "omicverse.pp._qc.qc": {
        "properties": {
            "adata_id": {"type": "string", "description": "Session dataset reference"},
            "mode": {
                "type": "string",
                "enum": ["seurat", "mads"],
                "default": "seurat",
                "description": "QC mode: seurat (threshold-based) or mads (MAD-based)",
            },
            "tresh": {
                "type": "object",
                "description": "Thresholds dict with keys mito_perc, nUMIs, detected_genes",
                "properties": {
                    "mito_perc": {"type": "number", "default": 0.2},
                    "nUMIs": {"type": "integer", "default": 500},
                    "detected_genes": {"type": "integer", "default": 250},
                },
            },
            "min_cells": {"type": "integer", "default": 3},
            "min_genes": {"type": "integer", "default": 200},
            "doublets": {"type": "boolean", "default": True},
            "mt_startswith": {
                "type": "string",
                "default": "MT-",
                "description": "Prefix identifying mitochondrial genes",
            },
        },
        "required": ["adata_id"],
        "additionalProperties": True,
    },
    # umap() accepts **kwargs — minimal schema
    "omicverse.pp._preprocess.umap": {
        "properties": {
            "adata_id": {"type": "string", "description": "Session dataset reference"},
        },
        "required": ["adata_id"],
        "additionalProperties": True,
    },
    # read() — path is required, storage optional
    "omicverse.utils._data.read": {
        "properties": {
            "path": {"type": "string", "description": "File path to read (h5ad, csv, tsv, txt)"},
            "backend": {
                "type": "string",
                "enum": ["python", "rust"],
                "default": "python",
            },
        },
        "required": ["path"],
        "additionalProperties": True,
    },
    "omicverse.io.single._read.read": {
        "properties": {
            "path": {"type": "string", "description": "File path to read (h5ad, csv, tsv, txt)"},
            "storage": {
                "type": "string",
                "enum": ["memory", "disk"],
                "default": "memory",
                "description": "h5ad only: 'memory' loads X into RAM; 'disk' keeps it on "
                               "disk (out-of-core on CPU, streamed blocks in GPU mode)",
            },
            "backend": {
                "type": "string",
                "enum": ["python", "rust"],
                "description": "Deprecated alias: 'python' = storage='memory', 'rust' = 'disk'",
            },
        },
        "required": ["path"],
        "additionalProperties": True,
    },
}

# ---------------------------------------------------------------------------
# Duplicate resolution rules
# ---------------------------------------------------------------------------

DUPLICATE_RESOLUTION: Dict[str, str] = {
    # If two entries share the same full_name, keep the one from this module
    "omicverse.utils._data.convert_to_pandas": "omicverse.utils._data",
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_manifest_override(full_name: str) -> dict:
    """Return manifest field patches for *full_name*, or empty dict."""
    return MANIFEST_OVERRIDES.get(full_name, {})


def get_schema_override(full_name: str) -> dict:
    """Return JSON Schema patches for *full_name*, or empty dict."""
    return SCHEMA_OVERRIDES.get(full_name, {})


def get_name_override(full_name: str) -> Optional[str]:
    """Return a forced canonical tool name, or ``None``."""
    return NAME_OVERRIDES.get(full_name)


def is_enabled_in_phase(full_name: str, phase: str) -> bool:
    """Check whether *full_name* is whitelisted for the given rollout *phase*."""
    entries = PHASE_WHITELIST.get(phase, [])
    return full_name in entries


def get_rollout_phase(full_name: str) -> str:
    """Return the rollout phase for *full_name*."""
    for phase, entries in PHASE_WHITELIST.items():
        if full_name in entries:
            return phase
    return "defer"
