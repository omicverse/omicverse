r"""De-novo protein **binder** design — the RFdiffusion → ProteinMPNN pipeline.

The headline protein-design workflow: given a target protein and hotspot
residues, generate a new mini-protein that binds it.

1. **RFdiffusion** (PPI mode) diffuses binder backbones docked against the
   target's hotspots (:func:`~omicverse.synbio._design.denovo_backbone`).
2. **ProteinMPNN** designs sequences for the binder chain while keeping the
   target fixed (the interface-aware inverse-folding step).
3. Optionally **validate** each design by re-folding / co-folding it with the
   target (ESMFold self-consistency, or Boltz-2 complex + affinity).

:func:`denovo_binder` runs the whole thing and returns ranked
:class:`BinderDesign` objects. RFdiffusion needs its own environment
(see :func:`denovo_backbone`); ProteinMPNN runs in the main env.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .._registry import register_function


@dataclass
class BinderDesign:
    backbone_pdb: str               # RFdiffusion binder backbone (+ target)
    binder_chain: str
    sequence: str                   # designed binder sequence
    mpnn_score: float               # ProteinMPNN score (lower = more confident)
    binder_length: int
    validation: Optional[Dict] = None   # {'plddt', 'affinity', ...} if validate!='none'

    def __repr__(self) -> str:  # pragma: no cover
        v = ""
        if self.validation:
            if "plddt" in self.validation:
                v = f", pLDDT={self.validation['plddt']:.2f}"
            if self.validation.get("affinity"):
                v += f", p_bind={self.validation['affinity'].get('binder_probability', float('nan')):.2f}"
        return (f"BinderDesign(len={self.binder_length}, "
                f"mpnn={self.mpnn_score:.3f}{v}, seq={self.sequence[:20]}...)")


def _chain_range(pdb: str, chain: str) -> tuple:
    lo, hi = None, None
    for line in open(pdb):
        if line.startswith(("ATOM", "HETATM")) and line[21] == chain:
            r = int(line[22:26])
            lo = r if lo is None else min(lo, r)
            hi = r if hi is None else max(hi, r)
    if lo is None:
        raise ValueError(f"target_pdb 中找不到链 {chain!r}。")
    return lo, hi


def _pdb_chains(pdb: str) -> List[str]:
    seen = []
    for line in open(pdb):
        if line.startswith("ATOM") and line[21] not in seen:
            seen.append(line[21])
    return seen


def _design_binder_chain(complex_pdb: str, design_chain: str,
                         num_seqs: int, sampling_temp: float,
                         seed: int) -> List[tuple]:
    """Run ProteinMPNN on *complex_pdb* designing only *design_chain* (target
    chains fixed). Returns [(sequence, score), ...] for the binder chain."""
    from ._proteinmpnn import ensure_proteinmpnn
    repo = ensure_proteinmpnn()
    helper = os.path.join(repo, "helper_scripts")
    work = tempfile.mkdtemp(prefix="ovsynbio_binder_mpnn_")
    pdb_dir = os.path.join(work, "pdbs")
    os.makedirs(pdb_dir, exist_ok=True)
    import shutil
    shutil.copy(complex_pdb, pdb_dir)

    parsed = os.path.join(work, "parsed.jsonl")
    assigned = os.path.join(work, "assigned.jsonl")
    subprocess.run([sys.executable, os.path.join(helper, "parse_multiple_chains.py"),
                    f"--input_path={pdb_dir}", f"--output_path={parsed}"],
                   check=True, capture_output=True, text=True)
    subprocess.run([sys.executable, os.path.join(helper, "assign_fixed_chains.py"),
                    f"--input_path={parsed}", f"--output_path={assigned}",
                    "--chain_list", design_chain], check=True,
                   capture_output=True, text=True)
    run_py = os.path.join(repo, "protein_mpnn_run.py")
    env = dict(os.environ)
    proc = subprocess.run(
        [sys.executable, run_py, "--jsonl_path", parsed,
         "--chain_id_jsonl", assigned, "--out_folder", work,
         "--num_seq_per_target", str(num_seqs),
         "--sampling_temp", str(sampling_temp), "--seed", str(seed),
         "--batch_size", "1"], capture_output=True, text=True, env=env)

    fa_dir = os.path.join(work, "seqs")
    fastas = [os.path.join(fa_dir, f) for f in os.listdir(fa_dir)] \
        if os.path.isdir(fa_dir) else []
    if not fastas:
        raise RuntimeError("ProteinMPNN 未产出序列:\n" + (proc.stderr or proc.stdout)[-800:])

    out = []
    name = None
    score = 0.0
    for line in open(fastas[0]):
        line = line.strip()
        if line.startswith(">"):
            name = line
            m = [x for x in line.split(",") if "score=" in x]
            score = float(m[0].split("score=")[1]) if m else 0.0
        elif line and name is not None:
            # ProteinMPNN concatenates designed chains with '/'; take the binder
            seq = line.split("/")[0] if "/" in line else line
            # skip the first record (native recovery, T=?) — keep designs
            out.append((seq, score))
    # first entry is the native sequence recovery; drop it
    return out[1:] if len(out) > 1 else out


@register_function(
    aliases=["denovo_binder", "从头结合蛋白", "binder_design", "de_novo_binder",
             "结合蛋白设计", "minibinder", "从头binder设计", "蛋白结合子设计"],
    category="synthetic_biology",
    description="从头结合蛋白设计全流程:RFdiffusion(PPI 模式,按 hotspot 在靶点上扩散 binder 骨架)→ ProteinMPNN(固定靶点、设计 binder 链序列)→ 可选 ESMFold/Boltz-2 验证(结构自洽 + 结合亲和力)。De-novo binder design: RFdiffusion → ProteinMPNN → optional validation.",
    examples=[
        "designs = ov.synbio.denovo_binder('target.pdb', hotspot_res=['A59','A83','A91'], binder_length=70)",
        "designs[0].sequence, designs[0].validation",
    ],
    related=["synbio.denovo_backbone", "synbio.inverse_design", "synbio.predict_complex",
             "synbio.predict_structure"],
    requires={},
    produces={},
)
def denovo_binder(target_pdb: str, hotspot_res: List[str], binder_length: int = 70,
                  target_chain: str = "A", num_backbones: int = 2,
                  num_seqs: int = 2, sampling_temp: float = 0.1,
                  validate: str = "none", device: Optional[str] = None,
                  seed: int = 37, out_dir: Optional[str] = None) -> List[BinderDesign]:
    """Design de-novo binders against *target_pdb* at the given *hotspot_res*.

    Parameters
    ----------
    target_pdb : str
        The target protein PDB.
    hotspot_res : list[str]
        Hotspot residues to bind, e.g. ``['A59','A83','A91']``.
    binder_length : int
        Length of the designed binder.
    num_backbones : int
        How many RFdiffusion backbones to diffuse.
    num_seqs : int
        ProteinMPNN sequences per backbone.
    validate : {'none','esmfold','boltz'}
        Optionally re-fold each design (ESMFold self-consistency) or co-fold with
        the target and score binding (Boltz-2).

    Returns ranked :class:`BinderDesign` objects (best ProteinMPNN score first;
    or, if validated, best validation score)."""
    from ._design import denovo_backbone
    if validate not in ("none", "esmfold", "boltz"):
        raise ValueError(
            f"validate must be one of ['none','esmfold','boltz'], got {validate!r}")
    lo, hi = _chain_range(target_pdb, target_chain)
    contigs = f"[{target_chain}{lo}-{hi}/0 {binder_length}-{binder_length}]"
    work = out_dir or tempfile.mkdtemp(prefix="ovsynbio_binder_")
    os.makedirs(work, exist_ok=True)

    designs: List[BinderDesign] = []
    for i in range(num_backbones):
        bb = denovo_backbone(
            {"target_pdb": target_pdb, "contigs": contigs, "hotspots": hotspot_res},
            out_path=os.path.join(work, f"backbone_{i}.pdb"),
            num_designs=1, device=device)
        chains = _pdb_chains(bb)
        binder_chain = next((c for c in chains if c != target_chain), "B")
        for seq, score in _design_binder_chain(bb, binder_chain, num_seqs,
                                               sampling_temp, seed):
            d = BinderDesign(backbone_pdb=bb, binder_chain=binder_chain,
                             sequence=seq, mpnn_score=score,
                             binder_length=len(seq))
            if validate != "none":
                d.validation = _validate_binder(seq, target_pdb, target_chain,
                                                validate, device,
                                                backbone_pdb=bb)
            designs.append(d)

    def _rank(d: BinderDesign):
        """Rank on interface evidence first, monomer confidence last.

        Ranking on ``plddt`` alone ranked on *monomer* folding confidence, which
        for an idealised helical bundle is ~94 whether or not it binds anything —
        so the ordering carried essentially no information about the interface.
        The field's triage criteria are self-consistency RMSD (design backbone vs
        refold, < ~2 A) and interface confidence (ipTM / PAE_interaction); those
        come first here when they are available, and monomer pLDDT only breaks
        remaining ties.
        """
        v = d.validation or {}
        # lower is better for scrmsd, higher for iptm/plddt
        scrmsd = v.get("scrmsd")
        iptm = v.get("iptm")
        return (
            0 if scrmsd is not None else 1,
            scrmsd if scrmsd is not None else 0.0,
            -(iptm if iptm is not None else -1.0),
            -(v.get("plddt", 0.0)),
            d.mpnn_score,
        )
    designs.sort(key=_rank)
    return designs


def _validate_binder(seq: str, target_pdb: str, target_chain: str,
                     mode: str, device, backbone_pdb: Optional[str] = None) -> Dict:
    if mode == "esmfold":
        from ._structure import predict_structure
        pred = predict_structure(seq, device=device)
        out = {"plddt": float(pred.mean_plddt) / 100.0}
        # Self-consistency RMSD is what "esmfold validation" means in this field:
        # refold the designed sequence and compare against the backbone it was
        # designed onto. Returning monomer pLDDT alone and calling it
        # self-consistency was a misuse of the term — the RMSD was never computed.
        if backbone_pdb:
            try:
                from ._evaluate import structure_rmsd
                path = getattr(pred, "path", None)
                if path:
                    out["scrmsd"] = float(structure_rmsd(backbone_pdb, path))
                    out["self_consistent"] = out["scrmsd"] < 2.0
            except Exception as exc:                     # pragma: no cover
                out["scrmsd_error"] = f"{type(exc).__name__}: {exc}"
        return out
    # boltz: co-fold binder + target sequence, score affinity
    from ._boltz import predict_complex
    tgt_seq = _chain_sequence(target_pdb, target_chain)
    # affinity_binder names the chain whose affinity is wanted; without it the
    # affinity head never runs and validation='boltz' could not return the
    # affinity its docstring promised.
    r = predict_complex(
        [{"id": "A", "protein": tgt_seq}, {"id": "B", "protein": seq}],
        affinity_binder="B", device=device)
    return {"plddt": r.plddt, "iptm": r.iptm, "affinity": r.affinity}


_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def _chain_sequence(pdb: str, chain: str) -> str:
    seq, seen = [], set()
    for line in open(pdb):
        if line.startswith("ATOM") and line[21] == chain and line[12:16].strip() == "CA":
            res = line[17:20].strip()
            key = (line[22:27])
            if key not in seen:
                seen.add(key)
                seq.append(_THREE_TO_ONE.get(res, "X"))
    return "".join(seq)


__all__ = ["denovo_binder", "BinderDesign"]
