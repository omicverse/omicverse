"""I/O for single-cell immune-repertoire (AIRR-seq) data.

This module reads TCR / BCR sequencing output into an :class:`anndata.AnnData`
object using a clean, documented per-cell receptor data model — an
omicverse-style reimplementation of the relevant parts of scirpy's reader.

Data model
----------
After reading, the returned :class:`~anndata.AnnData` carries one row per
*cell* (barcode).  The per-cell immune-receptor (IR) information lives in
``adata.obs`` as a fixed set of columns.  Up to two chains of each receptor
arm are stored — a *primary* (most-expressed) and a *secondary* chain:

``VJ_1_*`` / ``VJ_2_*``
    The VJ-arm chains — TRA (alpha) or IGK / IGL (light).
``VDJ_1_*`` / ``VDJ_2_*``
    The VDJ-arm chains — TRB (beta) or IGH (heavy).

For each of those four slots the following sub-fields are stored
(``*`` is one of ``VJ_1``, ``VJ_2``, ``VDJ_1``, ``VDJ_2``)::

    {*}_v_gene  {*}_d_gene  {*}_j_gene  {*}_c_gene
    {*}_junction      (CDR3 nucleotide)
    {*}_junction_aa   (CDR3 amino-acid)
    {*}_locus
    {*}_duplicate_count   (UMI / read support)
    {*}_productive

Plus the cell-level columns ``has_ir`` (``"True"``/``"False"``) and
``receptor_type`` (``"TCR"`` / ``"BCR"`` / ``"ambiguous"`` / ``"no IR"``).

The raw, un-collapsed per-contig table is kept in
``adata.uns['airr_contigs']`` for downstream re-processing.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from .._registry import register_function


# ---------------------------------------------------------------------------
# Constants — the per-cell AIRR schema
# ---------------------------------------------------------------------------
_CHAIN_SLOTS = ("VJ_1", "VJ_2", "VDJ_1", "VDJ_2")
_CHAIN_FIELDS = (
    "v_gene", "d_gene", "j_gene", "c_gene",
    "junction", "junction_aa", "locus",
    "duplicate_count", "productive",
)

# loci that belong to each receptor arm
_VJ_LOCI = {"TRA", "TRG", "IGK", "IGL"}
_VDJ_LOCI = {"TRB", "TRD", "IGH"}
_TCR_LOCI = {"TRA", "TRB", "TRG", "TRD"}
_BCR_LOCI = {"IGH", "IGK", "IGL"}

# The three receptor systems a cell can express. Chain overflow is counted
# within a system, not across all of them: a cell with 2 TRA + 1 IGK carries
# a dual-alpha TCR plus one ambient BCR contig, not a three-chain receptor.
_RECEPTOR_SYSTEMS = (
    {"TRA", "TRB"},          # alpha/beta TCR
    {"TRG", "TRD"},          # gamma/delta TCR
    {"IGH", "IGK", "IGL"},   # BCR
)


def airr_obs_columns() -> list[str]:
    """Return the full ordered list of per-cell AIRR ``obs`` columns."""
    cols: list[str] = ["has_ir", "receptor_type", "multi_chain"]
    for slot in _CHAIN_SLOTS:
        for field in _CHAIN_FIELDS:
            cols.append(f"{slot}_{field}")
    return cols


# ---------------------------------------------------------------------------
# Contig-table normalisation
# ---------------------------------------------------------------------------
def _normalize_10x_contigs(df: pd.DataFrame) -> pd.DataFrame:
    """Map a 10x ``filtered_contig_annotations.csv`` to the AIRR schema."""
    rename = {
        "barcode": "cell_id",
        "v_gene": "v_call",
        "d_gene": "d_call",
        "j_gene": "j_call",
        "c_gene": "c_call",
        "cdr3": "junction_aa",
        "cdr3_nt": "junction",
        "umis": "duplicate_count",
        "reads": "consensus_count",
        "chain": "locus",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "duplicate_count" not in df.columns and "consensus_count" in df.columns:
        df["duplicate_count"] = df["consensus_count"]
    return df


def _normalize_airr_rearrangement(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise an AIRR-rearrangement TSV into the per-contig schema."""
    if "cell_id" not in df.columns:
        # AIRR allows the barcode to be parsed from the sequence_id prefix
        if "sequence_id" in df.columns:
            df = df.copy()
            df["cell_id"] = (
                df["sequence_id"].astype(str).str.split("_contig").str[0]
            )
        else:
            raise ValueError("AIRR table lacks both 'cell_id' and 'sequence_id'.")
    if "duplicate_count" not in df.columns:
        for alt in ("umi_count", "consensus_count", "reads"):
            if alt in df.columns:
                df = df.copy()
                df["duplicate_count"] = df[alt]
                break
        else:
            df = df.copy()
            df["duplicate_count"] = 1
    return df


def _infer_locus(row: pd.Series) -> str:
    """Best-effort locus inference from V/J gene prefixes."""
    for col in ("v_call", "j_call", "c_call"):
        val = row.get(col, None)
        if isinstance(val, str) and len(val) >= 3:
            prefix = val[:3].upper()
            if prefix in _VJ_LOCI or prefix in _VDJ_LOCI:
                return prefix
    return "None"


def _is_productive(val) -> bool:
    if isinstance(val, str):
        return val.strip().lower() in {"true", "t", "yes", "productive"}
    return bool(val) if val is not None and val == val else False


def _has_chain_overflow(vj: pd.DataFrame, vdj: pd.DataFrame) -> bool:
    """True if any single receptor system contributed more than two chains to an arm.

    Only two slots per arm survive :func:`_contigs_to_cells`, so a cell that
    brought more chains than that has had some dropped — the configuration
    that suggests a doublet. Two refinements keep the flag honest:

    * counting is per receptor system, so ambient contamination from another
      system (an IGK contig in a T cell) does not masquerade as a multichain
      TCR;
    * contigs without a ``junction_aa`` are not chains and do not count, which
      is the same rule :func:`~omicverse.airr._qc.chain_qc` applies when it
      counts the filled slots.
    """
    def _real(frame: pd.DataFrame) -> pd.DataFrame:
        if "junction_aa" not in frame.columns:
            return frame
        ja = frame["junction_aa"]
        return frame[ja.notna() & ~ja.astype(str).isin(["", "None", "nan"])]

    vj, vdj = _real(vj), _real(vdj)
    for system in _RECEPTOR_SYSTEMS:
        n_vj = int(vj["locus"].isin(system).sum())
        n_vdj = int(vdj["locus"].isin(system).sum())
        if n_vj > 2 or n_vdj > 2:
            return True
    return False


def _contigs_to_cells(contigs: pd.DataFrame) -> pd.DataFrame:
    """Collapse a per-contig table into the per-cell AIRR ``obs`` frame.

    For every cell the chains are split into the VJ and VDJ receptor arms;
    within each arm chains are ranked by ``duplicate_count`` and the top two
    kept as the primary / secondary slot. Cells that carried more than two
    chains in either arm are flagged in ``multi_chain`` — without it the
    overflow would be indistinguishable from a genuine dual-receptor cell
    once the extra contigs are dropped.
    """
    contigs = contigs.copy()

    # ensure a locus for every contig
    if "locus" not in contigs.columns:
        contigs["locus"] = contigs.apply(_infer_locus, axis=1)
    contigs["locus"] = contigs["locus"].astype(str).str.upper().replace(
        {"MULTI": "None", "NONE": "None", "NAN": "None", "": "None"}
    )
    # impute missing locus by gene prefix
    miss = contigs["locus"] == "NONE"
    if "duplicate_count" not in contigs.columns:
        contigs["duplicate_count"] = 1
    contigs["duplicate_count"] = (
        pd.to_numeric(contigs["duplicate_count"], errors="coerce").fillna(0)
    )
    # A contig without a CDR3 cannot define a clone, so it must never outrank
    # one that can — otherwise a junction-less contig takes the VJ_2 slot and
    # the real second chain is dropped.
    if "junction_aa" in contigs.columns:
        ja = contigs["junction_aa"]
        contigs["_has_junction"] = (
            ja.notna() & ~ja.astype(str).isin(["", "None", "nan"])
        ).astype(int)
    else:
        contigs["_has_junction"] = 0
    _rank = ["_has_junction", "duplicate_count"]

    rows: dict[str, dict] = {}
    for cell_id, sub in contigs.groupby("cell_id"):
        rec: dict = {c: None for c in airr_obs_columns()}
        vj = sub[sub["locus"].isin(_VJ_LOCI)].sort_values(_rank, ascending=False)
        vdj = sub[sub["locus"].isin(_VDJ_LOCI)].sort_values(_rank, ascending=False)
        for arm, frame in (("VJ", vj), ("VDJ", vdj)):
            for i, (_, contig) in enumerate(frame.iterrows()):
                if i >= 2:
                    break
                slot = f"{arm}_{i + 1}"
                rec[f"{slot}_v_gene"] = contig.get("v_call")
                rec[f"{slot}_d_gene"] = contig.get("d_call")
                rec[f"{slot}_j_gene"] = contig.get("j_call")
                rec[f"{slot}_c_gene"] = contig.get("c_call")
                rec[f"{slot}_junction"] = contig.get("junction")
                rec[f"{slot}_junction_aa"] = contig.get("junction_aa")
                rec[f"{slot}_locus"] = contig.get("locus")
                rec[f"{slot}_duplicate_count"] = contig.get("duplicate_count")
                rec[f"{slot}_productive"] = _is_productive(contig.get("productive"))

        rec["multi_chain"] = "True" if _has_chain_overflow(vj, vdj) else "False"

        loci = set(sub["locus"]) - {"NONE"}
        has_tcr = bool(loci & _TCR_LOCI)
        has_bcr = bool(loci & _BCR_LOCI)
        if has_tcr and has_bcr:
            rec["receptor_type"] = "ambiguous"
        elif has_tcr:
            rec["receptor_type"] = "TCR"
        elif has_bcr:
            rec["receptor_type"] = "BCR"
        else:
            rec["receptor_type"] = "no IR"
        rec["has_ir"] = "True" if (has_tcr or has_bcr) else "False"
        rows[str(cell_id)] = rec

    obs = pd.DataFrame.from_dict(rows, orient="index")
    obs = obs.reindex(columns=airr_obs_columns())
    obs.index.name = "cell_id"
    return obs


def _build_anndata(contigs: pd.DataFrame, obs: pd.DataFrame):
    """Assemble a minimal AnnData carrying the per-cell AIRR table."""
    from anndata import AnnData

    n = obs.shape[0]
    X = np.zeros((n, 1), dtype=np.float32)
    adata = AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=["_ir_placeholder"]),
    )
    adata.uns["airr_contigs"] = contigs.reset_index(drop=True)
    return adata


# ---------------------------------------------------------------------------
# Public readers
# ---------------------------------------------------------------------------
@register_function(
    aliases=["read_10x_vdj", "load_10x_vdj", "读取10x免疫组库", "10x VDJ读取"],
    category="airr",
    description=(
        "Read 10x Cell Ranger V(D)J output (filtered_contig_annotations.csv "
        "or airr_rearrangement.tsv) into an AnnData with one row per cell and "
        "the per-cell TCR/BCR receptor data stored in obs (VJ_1/VJ_2/VDJ_1/"
        "VDJ_2 chain slots)."
    ),
    examples=[
        "adata = ov.airr.read_10x_vdj('filtered_contig_annotations.csv')",
        "adata = ov.airr.read_10x_vdj('airr_rearrangement.tsv')",
    ],
    related=["airr.read_airr", "airr.chain_qc"],
)
def read_10x_vdj(path: str):
    """Read 10x Cell Ranger V(D)J output into an AnnData.

    Parameters
    ----------
    path
        Path to a 10x ``filtered_contig_annotations.csv`` (CSV) or an
        ``airr_rearrangement.tsv`` (tab-separated AIRR format). The format is
        auto-detected from the extension and the column header.

    Returns
    -------
    :class:`~anndata.AnnData`
        One row per cell; per-cell AIRR fields in ``.obs``; the raw contig
        table in ``.uns['airr_contigs']``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    sep = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep, dtype=str)
    cols = set(df.columns)
    if "barcode" in cols and "cdr3" in cols:
        contigs = _normalize_10x_contigs(df)
    else:
        contigs = _normalize_airr_rearrangement(df)
    obs = _contigs_to_cells(contigs)
    return _build_anndata(contigs, obs)


@register_function(
    aliases=["read_airr", "load_airr", "读取AIRR", "AIRR读取"],
    category="airr",
    description=(
        "Read an AIRR-format rearrangement TSV (one row per contig/sequence) "
        "into an AnnData with one row per cell and the per-cell TCR/BCR "
        "receptor data stored in obs."
    ),
    examples=[
        "adata = ov.airr.read_airr('rearrangement.tsv')",
        "adata = ov.airr.read_airr(['tra.tsv', 'trb.tsv'])",
    ],
    related=["airr.read_10x_vdj", "airr.chain_qc"],
)
def read_airr(path):
    """Read an AIRR rearrangement TSV (or several) into an AnnData.

    Parameters
    ----------
    path
        Path to an AIRR rearrangement TSV, or a list of paths that are
        concatenated (e.g. a separate TRA and TRB file).

    Returns
    -------
    :class:`~anndata.AnnData`
    """
    paths = [path] if isinstance(path, str) else list(path)
    frames = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        frames.append(pd.read_csv(p, sep="\t", dtype=str))
    df = pd.concat(frames, ignore_index=True)
    contigs = _normalize_airr_rearrangement(df)
    obs = _contigs_to_cells(contigs)
    return _build_anndata(contigs, obs)


@register_function(
    aliases=["read_tracer", "read_immune_repertoire", "通用免疫组库读取"],
    category="airr",
    description=(
        "Generic reader for a per-contig immune-repertoire DataFrame or "
        "delimited file. Accepts arbitrary column names mapped via the "
        "``column_map`` argument; produces the standard per-cell AIRR AnnData."
    ),
    examples=[
        "adata = ov.airr.read_tracer(contig_df)",
        "adata = ov.airr.read_tracer('contigs.csv', column_map={'cell':'cell_id'})",
    ],
    related=["airr.read_airr", "airr.read_10x_vdj"],
)
def read_tracer(data, column_map: Optional[dict] = None):
    """Generic per-contig reader.

    Parameters
    ----------
    data
        A :class:`pandas.DataFrame` of per-contig records, or a path to a
        CSV/TSV file.
    column_map
        Optional mapping ``{source_column: airr_column}`` applied before
        normalisation. AIRR columns expected downstream are ``cell_id``,
        ``v_call``, ``d_call``, ``j_call``, ``c_call``, ``junction``,
        ``junction_aa``, ``locus``, ``productive``, ``duplicate_count``.

    Returns
    -------
    :class:`~anndata.AnnData`
    """
    if isinstance(data, str):
        sep = "\t" if data.lower().endswith((".tsv", ".txt")) else ","
        df = pd.read_csv(data, sep=sep, dtype=str)
    else:
        df = data.copy()
    if column_map:
        df = df.rename(columns=column_map)
    contigs = _normalize_airr_rearrangement(df)
    obs = _contigs_to_cells(contigs)
    return _build_anndata(contigs, obs)


@register_function(
    aliases=["from_airr_array", "airr_from_obsm", "obsm_airr读取", "awkward组库读取"],
    category="airr",
    description=(
        "Convert an AnnData whose per-cell TCR/BCR contigs are stored in "
        "obsm['airr'] (the scirpy awkward-array layout, e.g. as returned by "
        "ov.datasets.airr_singlecell) into the per-cell ov.airr obs schema "
        "(VJ_1/VJ_2/VDJ_1/VDJ_2 chain slots), preserving the gene-expression "
        "matrix and the original obs metadata."
    ),
    examples=[
        "adata = ov.datasets.airr_singlecell()",
        "adata = ov.airr.from_airr_array(adata)",
    ],
    related=["airr.read_10x_vdj", "airr.chain_qc"],
)
def from_airr_array(adata, *, airr_key: str = "airr"):
    """Bridge a scirpy-style ``obsm['airr']`` AnnData to the ov.airr schema.

    Single-cell AIRR datasets are often distributed with the gene-expression
    matrix in ``.X`` and the per-cell receptor contigs in an awkward array at
    ``obsm['airr']`` (one variable-length list of chains per cell — the
    scirpy on-disk layout). This reader flattens those contigs, collapses
    them into the per-cell ``VJ_1`` / ``VJ_2`` / ``VDJ_1`` / ``VDJ_2`` chain
    slots and merges the result back into ``adata.obs`` so the rest of the
    ``ov.airr`` single-cell stack (``chain_qc``, ``define_clonotypes`` …) can
    run directly — without losing the transcriptome.

    Parameters
    ----------
    adata
        AnnData with a per-cell receptor awkward array in
        ``adata.obsm[airr_key]``.
    airr_key
        ``obsm`` key holding the receptor contigs (default ``'airr'``).

    Returns
    -------
    :class:`~anndata.AnnData`
        The same cells x genes AnnData with the per-cell AIRR ``obs`` columns
        added; the raw contig table is kept in ``uns['airr_contigs']``.
    """
    if airr_key not in adata.obsm:
        raise KeyError(
            f"obsm[{airr_key!r}] not found — expected the per-cell receptor "
            "contigs (e.g. from ov.datasets.airr_singlecell)."
        )
    arr = adata.obsm[airr_key]
    fields = list(getattr(arr, "fields", []))
    keep = [f for f in (
        "locus", "v_call", "d_call", "j_call", "c_call",
        "junction", "junction_aa", "productive",
        "duplicate_count", "consensus_count",
    ) if f in fields]

    records = []
    for i in range(len(arr)):
        cell_id = str(adata.obs_names[i])
        for chain in arr[i]:
            rec = {f: chain[f] for f in keep}
            rec["cell_id"] = cell_id
            records.append(rec)
    contigs = pd.DataFrame.from_records(records)
    air_obs = _contigs_to_cells(contigs)
    air_obs = air_obs.reindex(adata.obs_names)

    out = adata.copy()
    for col in air_obs.columns:
        out.obs[col] = air_obs[col].values
    out.uns["airr_contigs"] = contigs.reset_index(drop=True)
    return out


@register_function(
    aliases=["airr_simulate", "simulate_airr", "模拟免疫组库", "AIRR模拟数据"],
    category="airr",
    description=(
        "Simulate a small single-cell TCR/BCR AnnData with known clonotype "
        "structure for tutorials and tests — no external download required."
    ),
    examples=[
        "adata = ov.airr.simulate_airr(n_cells=300, receptor='TCR')",
        "adata = ov.airr.simulate_airr(n_cells=200, receptor='BCR', seed=0)",
    ],
    related=["airr.read_10x_vdj", "airr.chain_qc"],
)
def simulate_airr(
    n_cells: int = 300,
    n_clones: int = 40,
    receptor: str = "TCR",
    seed: int = 0,
):
    """Simulate a single-cell immune-repertoire AnnData.

    Parameters
    ----------
    n_cells
        Number of cells (barcodes) to generate.
    n_clones
        Number of distinct clonotypes; clone sizes follow a power law so a
        few clones are expanded.
    receptor
        ``'TCR'`` (TRA/TRB chains) or ``'BCR'`` (IGH/IGK chains).
    seed
        Random seed.

    Returns
    -------
    :class:`~anndata.AnnData`
        Per-cell AIRR AnnData, plus a random ``obs['group']`` column with two
        sample groups for downstream comparisons.
    """
    rng = np.random.default_rng(seed)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    nt = "ACGT"
    if receptor.upper() == "TCR":
        vj_locus, vdj_locus = "TRA", "TRB"
        vj_v = [f"TRAV{i}" for i in range(1, 16)]
        vj_j = [f"TRAJ{i}" for i in range(1, 30)]
        vdj_v = [f"TRBV{i}" for i in range(1, 21)]
        vdj_d = [f"TRBD{i}" for i in range(1, 3)]
        vdj_j = [f"TRBJ{i}-{k}" for i in range(1, 3) for k in range(1, 6)]
    else:
        vj_locus, vdj_locus = "IGK", "IGH"
        vj_v = [f"IGKV{i}" for i in range(1, 16)]
        vj_j = [f"IGKJ{i}" for i in range(1, 6)]
        vdj_v = [f"IGHV{i}" for i in range(1, 21)]
        vdj_d = [f"IGHD{i}" for i in range(1, 7)]
        vdj_j = [f"IGHJ{i}" for i in range(1, 7)]

    # build clonotype templates
    clones = []
    for _ in range(n_clones):
        L = int(rng.integers(11, 17))
        clones.append({
            "vj_v": rng.choice(vj_v), "vj_j": rng.choice(vj_j),
            "vdj_v": rng.choice(vdj_v), "vdj_d": rng.choice(vdj_d),
            "vdj_j": rng.choice(vdj_j),
            "vj_cdr3": "".join(rng.choice(list(aa), L)),
            "vdj_cdr3": "".join(rng.choice(list(aa), L)),
            "vj_cdr3_nt": "".join(rng.choice(list(nt), L * 3)),
            "vdj_cdr3_nt": "".join(rng.choice(list(nt), L * 3)),
        })
    # clone-size weights — power law
    weights = 1.0 / (np.arange(1, n_clones + 1) ** 1.1)
    weights = weights / weights.sum()
    clone_idx = rng.choice(n_clones, size=n_cells, p=weights)

    records = []
    for i in range(n_cells):
        bc = f"CELL{i:05d}-1"
        cl = clones[clone_idx[i]]
        records.append({
            "cell_id": bc, "locus": vj_locus,
            "v_call": cl["vj_v"], "d_call": None, "j_call": cl["vj_j"],
            "c_call": vj_locus + "C",
            "junction": cl["vj_cdr3_nt"], "junction_aa": cl["vj_cdr3"],
            "productive": "True",
            "duplicate_count": int(rng.integers(3, 60)),
        })
        records.append({
            "cell_id": bc, "locus": vdj_locus,
            "v_call": cl["vdj_v"], "d_call": cl["vdj_d"], "j_call": cl["vdj_j"],
            "c_call": vdj_locus + "M",
            "junction": cl["vdj_cdr3_nt"], "junction_aa": cl["vdj_cdr3"],
            "productive": "True",
            "duplicate_count": int(rng.integers(3, 60)),
        })
    contigs = pd.DataFrame.from_records(records)
    obs = _contigs_to_cells(contigs)
    adata = _build_anndata(contigs, obs)
    adata.obs["group"] = rng.choice(["group_A", "group_B"], size=adata.n_obs)
    adata.obs["sample"] = rng.choice(
        ["s1", "s2", "s3", "s4"], size=adata.n_obs
    )
    return adata


# ---------------------------------------------------------------------------
# Per-cell heavy-chain extraction — bridges sc-BCR AnnData → AIRR DataFrame
# ---------------------------------------------------------------------------
@register_function(
    aliases=[
        "extract_heavy_chains", "heavy_chain_table", "bcr_heavy_chain_table",
        "igh_table", "重链表", "BCR重链提取",
    ],
    category="airr",
    description=(
        "Extract the dominant productive IgH (or, for TCR, VDJ-arm) contig "
        "per cell from a single-cell AIRR AnnData (``obsm['airr']`` awkward "
        "array) into a flat AIRR-format ``pandas.DataFrame``. This is the "
        "bridge between the AnnData-native single-cell side and the "
        "DataFrame-native B-cell / Immcantation side of ``ov.airr`` — feed "
        "the result into ``mutation_analysis``, ``clonal_clustering``, "
        "``lineage_trees`` etc."
    ),
    examples=[
        "db = ov.airr.extract_heavy_chains(adata)",
        "db = ov.airr.extract_heavy_chains(adata, locus='IGH', fields=['sequence_alignment','germline_alignment_d_mask','mu_freq'])",
    ],
    related=["airr.from_airr_array", "airr.mutation_analysis",
             "airr.clonal_clustering"],
)
def extract_heavy_chains(
    adata,
    *,
    airr_key: str = "airr",
    locus: str = "IGH",
    fields: Optional[list] = None,
    obs_cols: Optional[list] = None,
):
    """Extract the dominant productive heavy-chain contig per cell.

    Single-cell AIRR data stores per-cell contigs as a ragged awkward array
    in ``adata.obsm[airr_key]`` — convenient for storage, awkward for the
    Immcantation backends (which expect a flat AIRR-format
    ``pandas.DataFrame``). This helper:

    * Iterates over each cell;
    * Keeps only **productive** contigs at the requested ``locus`` (``IGH``
      by default for BCR; pass e.g. ``'TRB'`` for TCR beta);
    * Picks the **dominant** contig per cell, ranked by
      ``duplicate_count`` (UMI support);
    * Emits one row per cell into an AIRR-format ``DataFrame`` carrying
      the standard sequence / germline / mutation columns plus any extra
      ``obs`` columns requested.

    Parameters
    ----------
    adata
        Single-cell AIRR AnnData with ``obsm[airr_key]`` (e.g. as returned
        by :func:`ov.datasets.airr_singlecell_bcr` or any scirpy-style
        loader).
    airr_key
        ``obsm`` key holding the awkward contig array (default ``'airr'``).
    locus
        Locus to extract: ``'IGH'`` (default — heavy chain) for BCR,
        ``'TRB'`` for TCR beta, etc.
    fields
        AIRR fields to pull from each contig. If ``None``, a useful default
        is used: ``v_call`` / ``d_call`` / ``j_call`` / ``c_call`` /
        ``junction`` / ``junction_aa`` / ``sequence_alignment`` /
        ``germline_alignment_d_mask`` / ``mu_freq`` / ``duplicate_count`` /
        ``sample_id``. Missing fields are silently skipped.
    obs_cols
        ``adata.obs`` columns to attach to each row (default: none).

    Returns
    -------
    :class:`pandas.DataFrame`
        One row per cell with the requested fields and a ``cell_id`` column
        matching ``adata.obs_names``. Cells without a productive heavy
        chain are dropped.
    """
    if airr_key not in adata.obsm:
        raise KeyError(
            f"obsm[{airr_key!r}] not found — expected per-cell receptor "
            "contigs (e.g. from ov.datasets.airr_singlecell_bcr)."
        )
    arr = adata.obsm[airr_key]
    default_fields = [
        "v_call", "d_call", "j_call", "c_call",
        "junction", "junction_aa",
        "sequence_alignment", "germline_alignment_d_mask",
        "mu_freq", "duplicate_count", "sample_id", "productive",
    ]
    wanted = list(fields) if fields is not None else default_fields
    avail_fields = set(getattr(arr, "fields", []) or [])
    if len(arr) > 0:
        avail_fields |= set(getattr(arr[0], "fields", []) or [])
    wanted = [f for f in wanted if f in avail_fields]

    def _str_or_none(v):
        if v is None:
            return None
        s = str(v)
        return None if s in ("None", "nan", "") else s

    def _to_float(v):
        if v is None:
            return float("nan")
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    rows = []
    for i in range(len(arr)):
        cell_id = str(adata.obs_names[i])
        best = None
        for chain in arr[i]:
            loc = _str_or_none(chain["locus"]) if "locus" in avail_fields else None
            if loc != locus:
                continue
            prod = (
                _str_or_none(chain["productive"])
                if "productive" in avail_fields else "True"
            )
            if prod not in ("T", "True", "true", "TRUE"):
                continue
            dc_v = (
                _to_float(chain["duplicate_count"])
                if "duplicate_count" in avail_fields else 0.0
            )
            if not (best and dc_v <= best[0]):
                rec = {}
                for f in wanted:
                    val = chain[f]
                    if f in ("duplicate_count", "mu_freq", "junction_length"):
                        rec[f] = _to_float(val)
                    else:
                        rec[f] = _str_or_none(val)
                rec["cell_id"] = cell_id
                best = (dc_v, rec)
        if best is not None:
            rows.append(best[1])

    db = pd.DataFrame(rows)
    if obs_cols:
        meta = adata.obs[list(obs_cols)].copy()
        meta.index = meta.index.astype(str)
        db = db.merge(meta, left_on="cell_id", right_index=True, how="left")
    return db
