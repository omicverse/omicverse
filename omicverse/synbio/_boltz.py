r"""Biomolecular complex structure & binding affinity — Boltz-2.

`Boltz-2 <https://github.com/jwohlwend/boltz>`_ (Passaro *et al.* 2025) is an
open, AlphaFold3-class model that co-folds proteins, nucleic acids and small
molecules **and** predicts binding affinity — the first open model to approach
FEP-level affinity accuracy. This module drives its ``boltz predict`` CLI:

* :func:`predict_complex` — co-fold a complex from protein / DNA / RNA chains
  and/or a small-molecule ligand (SMILES / CCD), returning the structure, the
  confidence metrics (pLDDT / pTM / ipTM) and, when a binder is named, the
  predicted **binding affinity** (log-scale IC50 + a binary binder probability).

Boltz is heavy (its own torch stack + ~1 GB weights), so it lives in a **separate
environment** and is called as a subprocess — ``import omicverse`` never needs
it. Point :envvar:`OMICOS_BOLTZ_PYTHON` at a Python that has ``boltz`` installed
(default: ``$SCRATCH/env/boltz/bin/python``). First run downloads the weights and
(with ``use_msa_server=True``) fetches MSAs from the public ColabFold server.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from .._registry import register_function


@dataclass
class ComplexPrediction:
    structure_path: str                 # predicted mmCIF
    confidence: float                   # overall confidence (0..1)
    plddt: float                        # mean pLDDT (0..1)
    ptm: float                          # predicted TM-score
    iptm: float                         # interface pTM (multi-chain)
    affinity: Optional[Dict] = None     # {'ic50_log', 'binder_probability'} if requested
    out_dir: str = ""

    def __repr__(self) -> str:  # pragma: no cover
        aff = ""
        if self.affinity:
            aff = (f", affinity(logIC50={self.affinity.get('ic50_log'):.2f}, "
                   f"p_bind={self.affinity.get('binder_probability'):.2f})")
        return (f"ComplexPrediction(pLDDT={self.plddt:.2f}, pTM={self.ptm:.2f}, "
                f"ipTM={self.iptm:.2f}{aff})")


def _boltz_python() -> str:
    """Path to a Python interpreter with ``boltz`` installed."""
    env = os.environ.get("OMICOS_BOLTZ_PYTHON")
    if env and os.path.exists(env):
        return env
    scratch = os.environ.get("SCRATCH", os.path.expanduser("~"))
    cand = os.path.join(scratch, "env", "boltz", "bin", "python")
    return cand


def _check_boltz(py: str) -> None:
    if not os.path.exists(py):
        raise ImportError(
            "ov.synbio.predict_complex(method='boltz2') 需要独立的 boltz 环境。"
            "请建一个 venv 并 `pip install boltz`,然后把 OMICOS_BOLTZ_PYTHON 指向"
            f"它的 python(当前查找:{py})。")


def _weights_dir() -> str:
    from ._esm_common import weights_dir
    d = os.path.join(weights_dir(), "boltz_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _build_yaml(components: List[Dict], affinity_binder: Optional[str]) -> str:
    """Render a Boltz-2 input YAML from a list of component dicts."""
    lines = ["version: 1", "sequences:"]
    for comp in components:
        cid = comp["id"]
        if "protein" in comp:
            lines += [f"  - protein:", f"      id: {cid}",
                      f"      sequence: {comp['protein']}"]
        elif "dna" in comp:
            lines += [f"  - dna:", f"      id: {cid}",
                      f"      sequence: {comp['dna']}"]
        elif "rna" in comp:
            lines += [f"  - rna:", f"      id: {cid}",
                      f"      sequence: {comp['rna']}"]
        elif "smiles" in comp:
            lines += [f"  - ligand:", f"      id: {cid}",
                      f"      smiles: '{comp['smiles']}'"]
        elif "ccd" in comp:
            lines += [f"  - ligand:", f"      id: {cid}",
                      f"      ccd: {comp['ccd']}"]
    if affinity_binder:
        lines += ["properties:", "  - affinity:",
                  f"      binder: {affinity_binder}"]
    return "\n".join(lines) + "\n"


@register_function(
    aliases=["predict_complex", "复合物预测", "结合亲和力", "boltz", "boltz2",
             "蛋白复合物", "配体对接", "binding_affinity", "complex_structure",
             "亲和力预测", "docking"],
    category="synthetic_biology",
    description="生物大分子复合物共折叠 + 结合亲和力(Boltz-2,AF3 级开源模型)。输入蛋白/DNA/RNA 链与/或小分子(SMILES/CCD),返回结构、置信度(pLDDT/pTM/ipTM),指定 binder 时给出结合亲和力(log IC50 + 结合概率)。Co-fold a complex and predict binding affinity with Boltz-2.",
    examples=[
        "r = ov.synbio.predict_complex([{'id':'A','protein':seq},{'id':'B','smiles':smi}], affinity_binder='B')",
        "r.plddt, r.affinity",
    ],
    related=["synbio.predict_structure", "synbio.inverse_design", "synbio.enzyme_kcat"],
    requires={},
    produces={},
)
def predict_complex(components: List[Dict], affinity_binder: Optional[str] = None,
                    method: str = "boltz2", out_dir: Optional[str] = None,
                    use_msa_server: bool = True, device: Optional[str] = None,
                    diffusion_samples: int = 1,
                    no_kernels: bool = True) -> ComplexPrediction:
    """Co-fold a complex and (optionally) predict binding affinity with Boltz-2.

    Parameters
    ----------
    components : list of dict
        Each has an ``id`` plus one of ``protein`` / ``dna`` / ``rna`` (a
        sequence) or ``smiles`` / ``ccd`` (a small-molecule ligand). E.g.
        ``[{'id':'A','protein':'MVT...'}, {'id':'B','smiles':'CC(=O)O'}]``.
    affinity_binder : str, optional
        The ``id`` of the ligand chain whose binding affinity to predict.
    use_msa_server : bool
        Fetch MSAs from the public ColabFold server (needs internet).
    device : str, optional
        ``'cuda'`` / ``'cpu'`` (default: auto).
    """
    if method != "boltz2":
        raise ValueError(f"method must be one of ['boltz2'], got {method!r}")
    if not components or not all("id" in c for c in components):
        raise ValueError("每个 component 需要 'id' 以及 protein/dna/rna/smiles/ccd 之一。")
    py = _boltz_python()
    _check_boltz(py)
    from ._device import resolve_device, is_cuda
    dev = resolve_device(device)

    import tempfile
    work = out_dir or tempfile.mkdtemp(prefix="ovsynbio_boltz_")
    os.makedirs(work, exist_ok=True)
    yaml_path = os.path.join(work, "complex.yaml")
    with open(yaml_path, "w") as fh:
        fh.write(_build_yaml(components, affinity_binder))

    boltz_cli = os.path.join(os.path.dirname(py), "boltz")
    cmd = [boltz_cli, "predict", yaml_path, "--out_dir", work,
           "--cache", _weights_dir(), "--output_format", "mmcif",
           "--diffusion_samples", str(diffusion_samples),
           "--accelerator", "gpu" if is_cuda(dev) else "cpu"]
    if no_kernels:
        # native PyTorch triangle path — avoids the optional cuEquivariance
        # kernels (a heavy extra CUDA dep) at a small speed cost.
        cmd += ["--no_kernels"]
    if use_msa_server:
        cmd += ["--use_msa_server"]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    cifs = glob.glob(os.path.join(work, "**", "*_model_0.cif"), recursive=True) \
        or glob.glob(os.path.join(work, "**", "*.cif"), recursive=True)
    if not cifs:
        raise RuntimeError(
            "Boltz 预测未产出结构:\n" + (proc.stderr or proc.stdout)[-1200:])
    structure = cifs[0]

    conf = _load_json(work, "confidence")
    aff = None
    if affinity_binder:
        aj = _load_json(work, "affinity")
        if aj:
            aff = {"ic50_log": float(aj.get("affinity_pred_value", float("nan"))),
                   "binder_probability":
                       float(aj.get("affinity_probability_binary", float("nan")))}
    return ComplexPrediction(
        structure_path=structure,
        confidence=float(conf.get("confidence_score", conf.get("complex_plddt", 0.0))),
        plddt=float(conf.get("complex_plddt", 0.0)),
        ptm=float(conf.get("ptm", 0.0)), iptm=float(conf.get("iptm", 0.0)),
        affinity=aff, out_dir=work)


def _load_json(work: str, key: str) -> Dict:
    for p in glob.glob(os.path.join(work, "**", f"*{key}*.json"), recursive=True):
        try:
            with open(p) as fh:
                return json.load(fh)
        except Exception:
            continue
    return {}


__all__ = ["predict_complex", "ComplexPrediction"]
