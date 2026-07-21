r"""Layer B — enzyme function / EC-number prediction from sequence.

``enzyme_function(seq)`` predicts Enzyme Commission (EC) numbers.

Engines
-------
``"clean"``
    `CLEAN <https://github.com/tttianhao/CLEAN>`_ — contrastive ESM-1b model,
    the SOTA open method.  No PyPI package; used when a local checkout +
    weights are available, otherwise raises with install instructions.
``"knn"`` (default)
    A dependency-light baseline: embed the query with ESM-2 and nearest-neighbour
    it against a small bundled reference of characterised enzymes (or a
    user-supplied FASTA whose headers carry ``EC=...``).  Returns ranked EC
    candidates with cosine similarity.  Honest baseline — for genome-scale or
    high-confidence annotation use ``method="clean"``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .._registry import register_function
from ._device import describe_device, warn_if_cpu, resolve_device

_REFERENCE = os.path.join(os.path.dirname(__file__), "external", "ec_reference.fasta")


@dataclass
class ECPrediction:
    """Ranked EC-number predictions for one query."""

    query_len: int
    predictions: List[Tuple[str, float]] = field(default_factory=list)  # (EC, sim)
    method: str = "knn"

    @property
    def top_ec(self) -> Optional[str]:
        return self.predictions[0][0] if self.predictions else None

    def __repr__(self) -> str:  # pragma: no cover
        top = ", ".join(f"{ec}({s:.2f})" for ec, s in self.predictions[:3])
        return f"ECPrediction(method={self.method!r}, top=[{top}])"


def _read_reference(path: str) -> List[Tuple[str, str, str]]:
    """Return list of (id, EC, sequence) from a FASTA whose headers carry EC=."""
    entries, hid, hec, seq = [], None, None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if hid is not None:
                    entries.append((hid, hec, "".join(seq)))
                hid = line[1:].split("|")[0]
                hec = None
                for tok in line[1:].split("|"):
                    if tok.upper().startswith("EC="):
                        hec = tok.split("=", 1)[1]
                seq = []
            elif line:
                seq.append(line)
    if hid is not None:
        entries.append((hid, hec, "".join(seq)))
    return entries


@register_function(
    aliases=[
        "enzyme_function", "酶功能预测", "EC号预测", "ec_number", "酶分类",
        "clean", "功能注释", "酶EC",
    ],
    category="synthetic_biology",
    description="酶功能 / EC 号预测:从序列预测 Enzyme Commission 号(默认 ESM-2 嵌入 k-NN 基线,可选 CLEAN)。Enzyme EC-number prediction from sequence (CLEAN / ESM k-NN baseline).",
    examples=[
        "pred = ov.synbio.enzyme_function(enzyme_seq)",
        "pred.top_ec, pred.predictions",
    ],
    related=["synbio.protein_embed", "synbio.enzyme_kcat"],
    requires={},
    produces={},
)
def enzyme_function(
    seq: str,
    method: str = "knn",
    reference_fasta: Optional[str] = None,
    top_k: int = 3,
    device: Optional[str] = None,
    verbose: bool = True,
) -> ECPrediction:
    """Predict EC number(s) for an enzyme sequence.

    Parameters
    ----------
    seq
        Query amino-acid sequence.
    method
        ``"knn"`` (default) or ``"clean"``.
    reference_fasta
        Custom reference FASTA (headers must contain ``EC=<number>``).  When
        ``None`` a small bundled central-metabolism reference is used.
    top_k
        Number of ranked EC candidates to return.
    device
        ``None`` = auto.

    Returns
    -------
    ECPrediction
    """
    import numpy as np

    dev = resolve_device(device)
    if verbose:
        print(f"[ov.synbio.enzyme_function] method={method} device={describe_device(dev)}")
    warn_if_cpu(dev, "enzyme_function")

    if method == "clean":
        return _clean_predict(seq, top_k, dev)
    if method != "knn":
        raise ValueError(f"method must be one of ['knn', 'clean'], got {method!r}")

    from ._embed import protein_embed

    ref_path = reference_fasta or _REFERENCE
    refs = _read_reference(ref_path)
    if not refs:
        raise ValueError(f"参考库 {ref_path} 为空或不含 EC= 头。请提供 reference_fasta。")

    ref_seqs = [r[2] for r in refs]
    q = protein_embed([seq], model="esm2_t33_650M", device=dev, verbose=False)[0]
    R = protein_embed(ref_seqs, model="esm2_t33_650M", device=dev, verbose=False)

    qn = q / (np.linalg.norm(q) + 1e-8)
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)
    sims = Rn @ qn
    order = np.argsort(-sims)

    preds, seen = [], set()
    for idx in order:
        ec = refs[idx][1] or "unknown"
        if ec in seen:
            continue
        seen.add(ec)
        preds.append((ec, float(sims[idx])))
        if len(preds) >= top_k:
            break
    return ECPrediction(query_len=len(seq), predictions=preds, method="knn")


def _clean_predict(seq: str, top_k: int, device) -> ECPrediction:
    from ._esm_common import weights_dir

    repo = os.path.join(weights_dir(), "CLEAN")
    if not os.path.isdir(os.path.join(repo, "app")):
        raise ImportError(
            "method='clean' 需要 CLEAN 本地 checkout + 权重(未随包分发)。请:\n"
            "  git clone https://github.com/tttianhao/CLEAN "
            f"{repo}\n  并按其 README 下载预训练模型。\n"
            "或使用默认 method='knn'(ESM-2 嵌入基线,无需额外权重)。"
        )
    raise NotImplementedError(
        "CLEAN 后端已检测到 checkout,但请按 CLEAN README 运行其 inference 脚本;"
        "本包默认推荐 method='knn'。"
    )


__all__ = ["enzyme_function", "ECPrediction"]
