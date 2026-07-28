r"""Design evaluation — quantitative "did it get better?" metrics.

A generative design is only useful if you can *score* it. This module packages
the ov.synbio predictors into an in-silico **scorecard** that compares a designed
sequence/structure against a reference (wild-type / baseline), plus the
structural **self-consistency** metric the field uses to judge de-novo designs.

* :func:`structure_rmsd` — Cα RMSD between two structures after superposition
  (the basis of *self-consistency RMSD*: design a backbone → predict its
  sequence → re-fold → RMSD to the original; < ~2 Å ≈ a realisable design).
* :func:`evaluate_design` — run a panel of metrics on a design and report them
  with **reliability tags**, and Δ-vs-reference where a reference is given.

**Honesty first.** These are *in-silico proxies*, not wet-lab measurements. Some
are well-calibrated (stability ΔΔG, foldability pLDDT, self-consistency RMSD,
EC-retention); catalytic **kcat** prediction is noisy — treat small Δkcat between
close variants as *no signal*, not evidence of higher activity. Real proof of
improved activity needs experiments; these metrics are for *prioritising*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .._registry import register_function

# per-metric reliability of the in-silico proxy (for honest reporting)
#: How much weight each metric can bear. These labels are load-bearing — a
#: reader ranks the panel by them — so they describe what the metric *can
#: distinguish*, not how sophisticated the model behind it is.
_RELIABILITY = {
    "pLDDT": "high", "ddG": "high",
    # Detects the fold family, not catalysis. An E. coli DHFR with its catalytic
    # Asp27 removed still scores 1.5.1.3 at 0.993, and a quadruple active-site
    # knockout scores *higher* than the wild type; only a fully scrambled
    # sequence fails. It was labelled "high", which invited exactly the reading
    # that a near-1.0 score means the enzyme still works.
    "EC_confidence": "low",
    "scRMSD": "high", "ESM_fitness_delta": "medium", "ipTM": "medium",
    "affinity_p_bind": "medium", "kcat": "low", "kcat_delta": "low",
}

#: What each metric is blind to. Attached to the scorecard so the caveat travels
#: with the number instead of living in a docstring.
_BLIND_SPOTS = {
    "EC_confidence": "只反映折叠家族,不反映催化能力;敲掉催化残基分数几乎不变。",
    "pLDDT": "单体折叠置信度,与结合、活性、稳定性都无关。",
    "ESM_fitness_delta": "进化似然,不是稳定性也不是活性;若设计就是用同一模型"
                         "挑出来的,这个指标是循环的。",
    "kcat": "序列到 kcat 的预测未经校准;失活突变体常常打分高于野生型。",
    "scRMSD": "传入野生型结构时,它量的是野生型折叠与变体折叠的差,"
              "对少数点突变本来就接近 0。",
}


@dataclass
class DesignScorecard:
    metrics: Dict[str, float] = field(default_factory=dict)
    deltas: Dict[str, float] = field(default_factory=dict)   # design − reference
    reliability: Dict[str, str] = field(default_factory=dict)
    #: What each reported metric cannot see, keyed by metric name.
    blind_spots: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)      # e.g. EC string
    notes: List[str] = field(default_factory=list)            # skipped/failed

    def to_frame(self):
        """Tidy pandas DataFrame: metric · value · Δ vs reference · reliability."""
        import pandas as pd
        rows = []
        for k, v in self.metrics.items():
            rows.append({"metric": k, "value": v,
                         "delta_vs_ref": self.deltas.get(k, float("nan")),
                         "reliability": self.reliability.get(k, ""),
                         "blind_to": self.blind_spots.get(k, "")})
        return pd.DataFrame(rows)

    def __repr__(self) -> str:  # pragma: no cover
        parts = [f"{k}={v:.3g}" + (f"(Δ{self.deltas[k]:+.3g})" if k in self.deltas else "")
                 for k, v in self.metrics.items()]
        return "DesignScorecard(" + ", ".join(parts) + ")"


def _ca_atoms(pdb: str, chain: Optional[str]):
    from Bio.PDB import PDBParser
    s = PDBParser(QUIET=True).get_structure("s", pdb)
    atoms = []
    for model in s:
        for ch in model:
            if chain and ch.id != chain:
                continue
            for res in ch:
                if "CA" in res:
                    atoms.append(res["CA"])
        break
    return atoms


@register_function(
    aliases=["structure_rmsd", "结构RMSD", "rmsd", "自洽RMSD", "scRMSD",
             "结构比对", "self_consistency_rmsd"],
    category="synthetic_biology",
    description="两个结构的 Cα RMSD(先叠合再算),是设计自洽性(self-consistency RMSD)的基础:设计骨架→预测序列→重折叠→与原骨架比 RMSD,<~2Å 视为可实现的设计。Cα RMSD between two structures after superposition.",
    examples=[
        "rmsd = ov.synbio.structure_rmsd('design_backbone.pdb', 'refolded.pdb')",
    ],
    related=["synbio.evaluate_design", "synbio.predict_structure", "synbio.denovo_binder"],
    requires={},
    produces={},
)
def structure_rmsd(pdb_a: str, pdb_b: str, chain_a: Optional[str] = None,
                   chain_b: Optional[str] = None) -> float:
    """Cα RMSD (Å) between *pdb_a* and *pdb_b* after optimal superposition.

    Atoms are paired in residue order; the shorter chain sets the length."""
    from Bio.PDB import Superimposer
    a = _ca_atoms(pdb_a, chain_a)
    b = _ca_atoms(pdb_b, chain_b)
    n = min(len(a), len(b))
    if n == 0:
        raise ValueError("找不到可比对的 Cα 原子(检查 PDB / chain)。")
    sup = Superimposer()
    sup.set_atoms(a[:n], b[:n])
    return float(sup.rms)


@register_function(
    aliases=["evaluate_design", "评估设计", "设计评分", "design_scorecard",
             "score_design", "评分卡", "设计评估", "变体评估"],
    category="synthetic_biology",
    description="设计评分卡:对一个设计序列/结构跑一组 in-silico 指标(pLDDT 可折叠性、CLEAN EC 功能保留、DLKcat kcat 活性、ESM fitness、ThermoMPNN ΔΔG 稳定性、Boltz ipTM/亲和力、scRMSD 自洽性),给出数值、相对参考(野生型)的 Δ 与每项可信度。诚实标注:kcat 等为弱信号,金标准仍是湿实验。Panel of design-quality metrics vs a reference, with reliability tags.",
    examples=[
        "sc = ov.synbio.evaluate_design(variant, reference=WT, substrate=smiles)",
        "sc.to_frame()",
    ],
    related=["synbio.structure_rmsd", "synbio.stability_ddg", "synbio.enzyme_kcat",
             "synbio.variant_effect", "synbio.enzyme_function", "synbio.predict_complex"],
    requires={},
    produces={},
)
def evaluate_design(sequence: str, reference: Optional[str] = None,
                    substrate: Optional[str] = None, target: Optional[str] = None,
                    pdb: Optional[str] = None, backbone_pdb: Optional[str] = None,
                    device: Optional[str] = None, verbose: bool = True
                    ) -> DesignScorecard:
    """Score a designed protein *sequence* with an in-silico metric panel.

    Parameters
    ----------
    sequence : str
        The designed protein sequence.
    reference : str, optional
        Wild-type / baseline sequence for Δ comparison (same length → ESM
        fitness delta and ΔΔG on the differing positions).
    substrate : str, optional
        Substrate SMILES → DLKcat kcat (and Δkcat vs reference).
    target : str, optional
        A target protein sequence → co-fold with Boltz-2 for interface ipTM
        (binder quality).
    pdb : str, optional
        A structure of the design → ThermoMPNN ΔΔG on the mutations (needs
        ``reference``).
    backbone_pdb : str, optional
        A design backbone (e.g. RFdiffusion output) → self-consistency RMSD
        (re-fold ``sequence`` and compare).

    Returns a :class:`DesignScorecard`. Each metric is tagged with its
    reliability; every step degrades gracefully (missing deps/inputs → a note,
    never a crash)."""
    sc = DesignScorecard()

    def _try(name, fn):
        try:
            fn()
        except Exception as exc:  # pragma: no cover - best-effort panel
            sc.notes.append(f"{name}: skipped ({type(exc).__name__}: {str(exc)[:80]})")

    def _rel(k):
        sc.reliability[k] = _RELIABILITY.get(k, "medium")
        blind = _BLIND_SPOTS.get(k)
        if blind:
            sc.blind_spots[k] = blind

    # --- foldability (pLDDT) — also save the fold for the scRMSD step ---
    import tempfile
    def _plddt():
        from ._structure import predict_structure
        tmp = tempfile.mktemp(suffix="_refold.pdb")
        pred = predict_structure(sequence, out_path=tmp, device=device)
        sc.metrics["pLDDT"] = round(float(pred.mean_plddt), 2)
        sc._refold_pdb = getattr(pred, "path", None) or tmp
        _rel("pLDDT")
    _try("pLDDT", _plddt)

    # --- retained function (CLEAN EC) ---
    def _ec():
        from ._function import enzyme_function
        ec = enzyme_function(sequence, method="clean", verbose=False)
        top = ec.predictions[0] if getattr(ec, "predictions", None) else None
        if top:
            sc.labels["EC"] = str(top[0])
            sc.metrics["EC_confidence"] = round(float(top[1]), 3)
            _rel("EC_confidence")
    _try("EC_confidence", _ec)

    # --- catalytic activity (DLKcat) — LOW reliability ---
    def _kcat():
        from ._kcat import enzyme_kcat
        kd = enzyme_kcat(sequence, substrate, method="dlkcat", verbose=False)
        sc.metrics["kcat"] = round(float(kd.kcat), 4)
        _rel("kcat")
        if reference:
            kr = enzyme_kcat(reference, substrate, method="dlkcat", verbose=False)
            sc.deltas["kcat"] = round(float(kd.kcat - kr.kcat), 4)
    if substrate:
        _try("kcat", _kcat)

    # --- evolutionary fitness delta (ESM) ---
    def _fitness():
        if not reference or len(reference) != len(sequence):
            sc.notes.append("ESM_fitness_delta: needs same-length reference")
            return
        muts = [f"{r}{i + 1}{d}" for i, (r, d) in enumerate(zip(reference, sequence))
                if r != d]
        if not muts:
            return
        from ._variant import variant_effect
        df = variant_effect(reference, mutations=muts, model="esm1v", device=device)
        sc.metrics["ESM_fitness_delta"] = round(float(df["score"].sum()), 3)
        sc.labels["mutations"] = "+".join(muts)
        _rel("ESM_fitness_delta")
    if reference:
        _try("ESM_fitness_delta", _fitness)

    # --- stability ΔΔG (ThermoMPNN) — needs a structure + mutations ---
    def _ddg():
        if not (pdb and reference and len(reference) == len(sequence)):
            return
        muts = [f"{r}{i + 1}{d}" for i, (r, d) in enumerate(zip(reference, sequence))
                if r != d]
        if not muts:
            return
        from ._stability import stability_ddg
        df = stability_ddg(pdb, mutations=muts, method="thermompnn", device=device)
        col = "ddg" if "ddg" in df.columns else df.columns[-1]
        sc.metrics["ddG"] = round(float(df[col].sum()), 3)
        _rel("ddG")
    if pdb:
        _try("ddG", _ddg)

    # --- interface quality vs a target (Boltz-2 ipTM) ---
    def _iptm():
        from ._boltz import predict_complex
        r = predict_complex([{"id": "A", "protein": target},
                             {"id": "B", "protein": sequence}], device=device)
        sc.metrics["ipTM"] = round(float(r.iptm), 3)
        _rel("ipTM")
    if target:
        _try("ipTM", _iptm)

    # --- self-consistency RMSD (design backbone vs re-fold) ---
    def _scrmsd():
        refold = getattr(sc, "_refold_pdb", None)
        if not refold:
            from ._structure import predict_structure
            tmp = tempfile.mktemp(suffix="_refold.pdb")
            predict_structure(sequence, out_path=tmp, device=device)
            refold = tmp
        sc.metrics["scRMSD"] = round(structure_rmsd(backbone_pdb, refold), 3)
        _rel("scRMSD")
    if backbone_pdb:
        _try("scRMSD", _scrmsd)

    if verbose:
        print("[evaluate_design]", sc)
        if sc.notes:
            print("  notes:", "; ".join(sc.notes))
    return sc


__all__ = ["evaluate_design", "structure_rmsd", "DesignScorecard"]
