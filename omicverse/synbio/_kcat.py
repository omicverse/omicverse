r"""Layer B — enzyme turnover-number (k_cat) prediction.

``enzyme_kcat(sequence, substrate_smiles)`` predicts a turnover number in 1/s.
This is the protein-side input to the A↔B hinge: its output feeds
:func:`omicverse.synbio.ec_model` to rebuild an enzyme-constrained metabolic
model, so "edit the enzyme → re-solve the pathway yield" becomes one pipeline.

Engines
-------
``"dlkcat"``
    `DLKcat <https://github.com/SysBioChalmers/DLKcat>`_ — a sequence+substrate
    deep model.  No PyPI package; used when a local checkout + weights are
    available (auto-detected), otherwise it raises with install instructions.
``"baseline"`` (default)
    A dependency-light, **sequence-sensitive** estimator: an ESM-2 embedding of
    the enzyme is mapped through a fixed projection into a plausible
    log-turnover range, modestly modulated by a coarse substrate descriptor.
    It is deterministic and *responds to mutations* (a faster/slower variant
    gives a different k_cat), which is what the coupling demo needs.  It is an
    **honest baseline for wiring and relative comparison — not a calibrated
    quantitative predictor.** For quantitative k_cat, use ``method="dlkcat"``
    (or UniKP/TurNuP) with trained weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

from .._registry import register_function
from ._device import describe_device, warn_if_cpu, resolve_device


@dataclass
class KcatPrediction:
    """A single k_cat prediction."""

    sequence: str
    substrate: str
    kcat: float          # 1/s
    method: str
    log10_kcat: float = 0.0

    def __repr__(self) -> str:  # pragma: no cover
        return (f"KcatPrediction(kcat={self.kcat:.3g} /s, method={self.method!r}, "
                f"substrate={self.substrate!r})")


# median BRENDA k_cat ≈ 13 /s; spread ~2 orders of magnitude either side.
_LOG_CENTER = 1.1     # log10(~13)
_LOG_SPREAD = 1.2


def _substrate_factor(smiles: str) -> float:
    """Coarse [-0.3, +0.3] modulation from a substrate SMILES (no rdkit).

    Larger / more complex substrates trend to slightly lower turnover; this is
    a weak, monotone heuristic, not chemistry."""
    if not smiles:
        return 0.0
    heavy = sum(1 for c in smiles if c.isalpha() and c.upper() in "CNOSPFI")
    rings = sum(1 for c in smiles if c.isdigit())
    x = heavy + 2 * rings
    # map size 5..60 -> +0.2..-0.3
    import math
    return float(max(-0.3, min(0.2, 0.2 - (x - 5) / 55.0 * 0.5)))


def _baseline_kcat(seq: str, substrate: str, device) -> float:
    import numpy as np
    from ._embed import protein_embed

    emb = protein_embed([seq], model="esm2_t33_650M", device=device,
                        verbose=False)[0]
    # fixed, deterministic projection -> standard normal-ish scalar
    rng = np.random.default_rng(20240607)
    w = rng.standard_normal(emb.shape[0])
    w /= np.linalg.norm(w)
    z = float(np.dot(emb - emb.mean(), w) / (emb.std() + 1e-8))
    z = max(-3.0, min(3.0, z))
    log10 = _LOG_CENTER + _LOG_SPREAD * (z / 3.0) + _substrate_factor(substrate)
    return float(10.0 ** log10)


@register_function(
    aliases=[
        "enzyme_kcat", "酶动力学预测", "kcat", "酶周转数", "turnover_number",
        "dlkcat", "酶kcat预测", "催化常数",
    ],
    category="synthetic_biology",
    description="酶周转数 (kcat) 预测:输入酶序列 + 底物 SMILES → 预测 kcat (1/s)。作为 A↔B 咬合的蛋白端输入,喂给 ec_model 重算通量。Enzyme k_cat prediction (DLKcat / sequence baseline).",
    examples=[
        "k = ov.synbio.enzyme_kcat(enzyme_seq, 'C(C(=O)O)N')   # substrate SMILES",
        "ecm = ov.synbio.ec_model(m, {'PFK': k.kcat})",
    ],
    related=["synbio.ec_model", "synbio.apply_kcat", "synbio.protein_embed", "synbio.fba"],
    requires={},
    produces={},
)
def enzyme_kcat(
    seq: str,
    substrate_smiles: str = "",
    method: str = "baseline",
    device: Optional[str] = None,
    verbose: bool = True,
) -> KcatPrediction:
    """Predict the turnover number k_cat (1/s) for an enzyme–substrate pair.

    Parameters
    ----------
    seq
        Enzyme amino-acid sequence.
    substrate_smiles
        Substrate SMILES (optional; modulates the baseline slightly, required
        by DLKcat).
    method
        ``"baseline"`` (default, sequence-sensitive, no extra weights) or
        ``"dlkcat"`` (needs a DLKcat checkout + weights).
    device
        ``None`` = auto.

    Returns
    -------
    KcatPrediction
    """
    import numpy as np

    dev = resolve_device(device)
    if verbose:
        print(f"[ov.synbio.enzyme_kcat] method={method} device={describe_device(dev)}")
    warn_if_cpu(dev, "enzyme_kcat")

    if method == "dlkcat":
        kcat = _dlkcat_predict(seq, substrate_smiles, dev)
    elif method == "baseline":
        kcat = _baseline_kcat(seq, substrate_smiles, dev)
    else:
        raise ValueError(f"method must be one of ['baseline', 'dlkcat'], got {method!r}")

    return KcatPrediction(sequence=seq, substrate=substrate_smiles, kcat=kcat,
                          method=method, log10_kcat=float(np.log10(kcat)))


def _dlkcat_predict(seq: str, smiles: str, device) -> float:
    """Run DLKcat if a local checkout with weights is present."""
    import os
    from ._esm_common import weights_dir

    repo = os.path.join(weights_dir(), "DLKcat")
    pred_script = os.path.join(repo, "DeeplearningApproach", "Code", "example",
                               "prediction_for_input.py")
    if not os.path.exists(pred_script):
        raise ImportError(
            "method='dlkcat' 需要 DLKcat 本地 checkout + 权重(未随包分发)。请:\n"
            "  git clone https://github.com/SysBioChalmers/DLKcat "
            f"{repo}\n  并按其 README 准备训练好的模型。\n"
            "或使用默认 method='baseline'(基于 ESM,序列敏感,无需额外权重)。"
        )
    # DLKcat's own CLI expects a TSV; delegate via subprocess.
    import subprocess, sys, tempfile, csv
    tmp = tempfile.mkdtemp(prefix="ovsynbio_dlkcat_")
    inp = os.path.join(tmp, "input.tsv")
    with open(inp, "w") as fh:
        fh.write("Protein\tSMILES\n")
        fh.write(f"{seq}\t{smiles}\n")
    out = os.path.join(tmp, "output.tsv")
    subprocess.run([sys.executable, pred_script, inp, out], check=True,
                   cwd=os.path.dirname(pred_script))
    with open(out) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    for r in rows:
        for k, v in r.items():
            if "kcat" in k.lower():
                return float(v)
    raise RuntimeError("DLKcat 输出中未找到 kcat 列")


__all__ = ["enzyme_kcat", "KcatPrediction"]
