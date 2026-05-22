r"""Known-drug / bioactivity lookup for :mod:`omicverse.mol`.

:func:`known_drugs` answers "does my omics target already have approved
drugs or bioactive compounds?" by querying **ChEMBL** — its
mechanism-of-action table (the clean "known drugs" set) and, optionally,
its broader bioactivity records.
"""

from __future__ import annotations

import re
from typing import List, Optional

import pandas as pd

from .._registry import register_function
from ._check import _need

_UNIPROT_RE = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"
)

_COLUMNS = ["drug_name", "chembl_id", "max_phase", "mechanism_of_action",
            "action_type", "smiles", "target_chembl_id", "evidence"]


def _resolve_target(target: str, organism: str):
    """Resolve a gene symbol / UniProt accession to a ChEMBL target id."""
    from chembl_webresource_client.new_client import new_client

    if _UNIPROT_RE.match(target):
        acc = target
    else:
        from ._structure import _gene_to_uniprot
        acc = _gene_to_uniprot(target)

    hits = list(new_client.target.filter(
        target_components__accession=acc).only(
        ["target_chembl_id", "target_type", "organism", "pref_name"]))
    if not hits:
        raise ValueError(
            f"no ChEMBL target for {target!r} (UniProt {acc}); the protein "
            f"may have no recorded bioactivity.")
    # prefer a single-protein target in the requested organism
    single = [h for h in hits if h.get("target_type") == "SINGLE PROTEIN"]
    pool = single or hits
    for h in pool:
        if (h.get("organism") or "").lower() == organism.lower():
            return h["target_chembl_id"], h.get("pref_name")
    return pool[0]["target_chembl_id"], pool[0].get("pref_name")


def _fetch_molecules(chembl_ids: List[str]) -> dict:
    """Batch-fetch molecule name / phase / SMILES for ChEMBL ids."""
    from chembl_webresource_client.new_client import new_client

    info = {}
    ids = sorted(set(chembl_ids))
    for i in range(0, len(ids), 40):
        chunk = ids[i:i + 40]
        for mol in new_client.molecule.filter(
                molecule_chembl_id__in=chunk).only(
                ["molecule_chembl_id", "pref_name", "max_phase",
                 "molecule_structures"]):
            struct = mol.get("molecule_structures") or {}
            info[mol["molecule_chembl_id"]] = {
                "drug_name": mol.get("pref_name"),
                "max_phase": mol.get("max_phase"),
                "smiles": struct.get("canonical_smiles"),
            }
    return info


@register_function(
    aliases=["known_drugs", "已知药物", "drugs", "chembl_drugs",
             "known_inhibitors", "靶点药物", "drug_lookup"],
    category="mol",
    description=(
        "Look up known drugs and bioactive compounds against an omics "
        "target via ChEMBL. Returns the mechanism-of-action records "
        "(approved / clinical drugs with an annotated target mechanism — "
        "the clean 'known drugs' set) and, optionally, broader bioactivity "
        "hits. Existing chemical matter is itself strong evidence a target "
        "is druggable, and feeds drug-repurposing follow-up."
    ),
    examples=[
        "df = ov.mol.known_drugs('EGFR')",
        "df = ov.mol.known_drugs('EGFR', max_phase=4)   # approved only",
        "df = ov.mol.known_drugs('P00533', only_mechanism=True)",
    ],
    related=["mol.fetch_structure", "mol.dock", "mol.druggability"],
)
def known_drugs(target: str, *, organism: str = "Homo sapiens",
                max_phase: Optional[int] = None, only_mechanism: bool = False,
                max_records: int = 300) -> pd.DataFrame:
    r"""Known drugs / bioactive compounds for a target, from ChEMBL.

    Parameters
    ----------
    target
        Gene symbol (``'EGFR'``) or UniProt accession (``'P00533'``).
    organism
        Organism name for target disambiguation.
    max_phase
        Keep only molecules whose maximum clinical phase is at least this
        (``4`` = approved). ``None`` keeps all.
    only_mechanism
        ``True`` restricts to compounds with an annotated mechanism of
        action; ``False`` also appends broader bioactivity hits.
    max_records
        Cap on broader bioactivity molecules fetched.

    Returns
    -------
    pandas.DataFrame
        Columns: ``drug_name, chembl_id, max_phase, mechanism_of_action,
        action_type, smiles, target_chembl_id, evidence``.
    """
    _need("chembl_webresource_client", "mol", "ov.mol known-drug lookup")
    from chembl_webresource_client.new_client import new_client

    tid, _pref = _resolve_target(target, organism)

    # mechanism-of-action records — the clean "known drugs" set
    mechs = list(new_client.mechanism.filter(target_chembl_id=tid).only(
        ["molecule_chembl_id", "mechanism_of_action", "action_type"]))
    rows = []
    mech_ids = {m["molecule_chembl_id"] for m in mechs}

    extra_ids: List[str] = []
    if not only_mechanism:
        acts = new_client.activity.filter(
            target_chembl_id=tid, pchembl_value__isnull=False).only(
            ["molecule_chembl_id", "pchembl_value", "standard_type"])
        seen = set()
        for act in acts:
            mid = act.get("molecule_chembl_id")
            if mid and mid not in mech_ids and mid not in seen:
                seen.add(mid)
                extra_ids.append(mid)
            if len(extra_ids) >= max_records:
                break

    info = _fetch_molecules(list(mech_ids) + extra_ids)

    for m in mechs:
        mid = m["molecule_chembl_id"]
        meta = info.get(mid, {})
        rows.append({
            "drug_name": meta.get("drug_name") or mid,
            "chembl_id": mid,
            "max_phase": meta.get("max_phase"),
            "mechanism_of_action": m.get("mechanism_of_action"),
            "action_type": m.get("action_type"),
            "smiles": meta.get("smiles"),
            "target_chembl_id": tid,
            "evidence": "mechanism",
        })
    for mid in extra_ids:
        meta = info.get(mid, {})
        rows.append({
            "drug_name": meta.get("drug_name") or mid,
            "chembl_id": mid,
            "max_phase": meta.get("max_phase"),
            "mechanism_of_action": None,
            "action_type": None,
            "smiles": meta.get("smiles"),
            "target_chembl_id": tid,
            "evidence": "bioactivity",
        })

    df = pd.DataFrame(rows, columns=_COLUMNS)
    if not df.empty:
        df = df.drop_duplicates("chembl_id").reset_index(drop=True)
        if max_phase is not None:
            phase = pd.to_numeric(df["max_phase"], errors="coerce")
            df = df[phase.fillna(-1) >= max_phase].reset_index(drop=True)
        # approved / late-phase first, mechanism evidence first
        df["_p"] = pd.to_numeric(df["max_phase"], errors="coerce").fillna(-1)
        df["_e"] = (df["evidence"] == "mechanism").astype(int)
        df = (df.sort_values(["_e", "_p"], ascending=False)
                .drop(columns=["_p", "_e"]).reset_index(drop=True))
    return df
