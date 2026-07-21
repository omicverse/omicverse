r"""Layer B — single-sequence structure prediction with ESMFold.

Fold a protein sequence into 3D coordinates (PDB) with ESMFold
(``facebook/esmfold_v1`` via 🤗 transformers — no MSA, no openfold install).
ESMFold is heavy: by default it **requires a GPU** and raises a clear error on
CPU rather than churning for hours.  Weights (~2.8 GB) download once to
``~/.omicverse/synbio_weights`` (``OMICOS_SYNBIO_WEIGHTS``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .._registry import register_function
from ._device import resolve_device, require_gpu, describe_device, is_cuda, _torch
from ._esm_common import weights_dir

_MODEL_ID = "facebook/esmfold_v1"
_cache = {}


@dataclass
class StructurePrediction:
    """ESMFold result."""

    sequence: str
    pdb: str
    mean_plddt: float
    path: Optional[str] = None
    device: str = "cpu"

    def save(self, path: str) -> str:
        with open(path, "w") as fh:
            fh.write(self.pdb)
        self.path = path
        return path

    def __repr__(self) -> str:  # pragma: no cover
        return (f"StructurePrediction(len={len(self.sequence)}, "
                f"mean_pLDDT={self.mean_plddt:.1f}, device={self.device!r}, "
                f"path={self.path!r})")


def _load_esmfold(device: str):
    if device in _cache:
        return _cache[device]
    try:
        from transformers import EsmForProteinFolding
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.synbio.predict_structure 需要 transformers 的 ESMFold。请 "
            "pip install 'omicverse[synbio]' (transformers>=4.30)。"
        ) from exc
    os.environ.setdefault("HF_HOME", weights_dir())
    torch = _torch()
    model = EsmForProteinFolding.from_pretrained(
        _MODEL_ID, cache_dir=weights_dir(),
    )
    model = model.eval().to(device)
    if is_cuda(device):
        # chunked attention keeps long sequences within memory
        try:
            model.trunk.set_chunk_size(64)
        except Exception:
            pass
    _cache[device] = (model, torch)
    return _cache[device]


@register_function(
    aliases=[
        "predict_structure", "结构预测", "蛋白结构预测", "folding", "esmfold",
        "protein_structure", "折叠",
    ],
    category="synthetic_biology",
    description="蛋白结构预测:用 ESMFold 从单条序列直接折叠出 3D 结构 (PDB),无需 MSA。默认需 GPU。ESMFold single-sequence structure prediction → PDB + pLDDT.",
    examples=[
        "pred = ov.synbio.predict_structure('MKTAYIAKQRQISFVK...')",
        "pred.save('mypro.pdb'); pred.mean_plddt",
    ],
    related=["synbio.inverse_design", "synbio.stability_ddg", "synbio.variant_effect"],
    requires={},
    produces={},
)
def predict_structure(
    seq: str,
    device: Optional[str] = None,
    out_path: Optional[str] = None,
    allow_cpu: bool = False,
    verbose: bool = True,
) -> StructurePrediction:
    """Fold *seq* with ESMFold.

    Parameters
    ----------
    seq
        Amino-acid sequence (single chain).
    device
        ``None`` = auto (CUDA if available).  ESMFold on CPU is impractical, so
        a CPU device raises unless ``allow_cpu=True``.
    out_path
        Optional path to write the PDB.
    allow_cpu
        Override the GPU requirement (expect minutes-to-hours on CPU).

    Returns
    -------
    StructurePrediction
    """
    import numpy as np

    device = resolve_device(device)
    if not allow_cpu:
        device = require_gpu(device, "predict_structure",
                             small_model_hint="(ESMFold 无小模型;请提供 GPU)")
    if verbose:
        print(f"[ov.synbio.predict_structure] device={describe_device(device)} "
              f"len={len(seq)}")

    model, torch = _load_esmfold(device)
    with torch.no_grad():
        pdb = model.infer_pdb(seq)

    # mean pLDDT lives in the B-factor column of the PDB
    plddts = []
    for line in pdb.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                plddts.append(float(line[60:66]))
            except ValueError:
                pass
    mean_plddt = float(np.mean(plddts)) if plddts else float("nan")
    # transformers ESMFold writes pLDDT on a 0–1 scale in the B-factor column;
    # report on the conventional 0–100 scale (AlphaFold convention).
    if plddts and max(plddts) <= 1.0:
        mean_plddt *= 100.0

    pred = StructurePrediction(sequence=seq, pdb=pdb, mean_plddt=mean_plddt,
                               device=device)
    if out_path:
        pred.save(out_path)
    if verbose:
        print(f"[ov.synbio.predict_structure] done: mean pLDDT={mean_plddt:.1f}")
    return pred


__all__ = ["predict_structure", "StructurePrediction"]
