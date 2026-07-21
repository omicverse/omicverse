r"""Real Doench-2016 CFD (Cutting Frequency Determination) off-target scoring.

The CFD score (Doench *et al.* 2016, Nat Biotechnol) is the field-standard
CRISPR off-target metric: a product of empirically-measured per-position,
per-mismatch-type retention factors times a PAM factor. The scoring tables
(``mismatch_score.pkl`` / ``pam_scores.pkl``, 240 + 16 values) are vendored
under ``external/cfd/`` from the CRISPOR distribution; the algorithm here is a
faithful reimplementation of the official ``cfd-score-calculator.py``.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, Optional, Tuple

_CFD_DIR = os.path.join(os.path.dirname(__file__), "external", "cfd")
_COMP = {"A": "T", "C": "G", "G": "C", "T": "A", "U": "A"}
_cache: Dict[str, dict] = {}


def _revcomp1(nt: str) -> str:
    return _COMP[nt]


def _load() -> Tuple[dict, dict]:
    if "mm" not in _cache:
        try:
            _cache["mm"] = pickle.load(open(os.path.join(_CFD_DIR, "mismatch_score.pkl"), "rb"))
            _cache["pam"] = pickle.load(open(os.path.join(_CFD_DIR, "pam_scores.pkl"), "rb"))
        except FileNotFoundError as exc:  # pragma: no cover
            raise ImportError(
                "CFD 打分表缺失(external/cfd/*.pkl)。请重新安装 omicverse 或从 "
                "CRISPOR 获取 mismatch_score.pkl / pam_scores.pkl。") from exc
    return _cache["mm"], _cache["pam"]


def cfd_score(guide: str, offtarget: str, pam2: str) -> float:
    """Doench-2016 CFD score in [0, 1] for a *guide* vs an *offtarget* protospacer.

    Parameters
    ----------
    guide, offtarget
        20-nt protospacers (the intended guide and the genomic near-match).
    pam2
        The two PAM-defining nucleotides of the off-target (for SpCas9 NGG, the
        ``GG``, i.e. the last two of the 3-nt PAM).

    Returns
    -------
    float
        1.0 = identical / full activity; → 0 = unlikely to be cut.
    """
    mm_scores, pam_scores = _load()
    sg = offtarget.upper().replace("T", "U")
    wt = guide.upper().replace("T", "U")
    score = 1.0
    for i in range(min(len(wt), len(sg))):
        if wt[i] == sg[i]:
            continue
        key = "r" + wt[i] + ":d" + _revcomp1(sg[i].replace("U", "T")) + "," + str(i + 1)
        score *= mm_scores.get(key, 0.0)
    score *= pam_scores.get(pam2.upper().replace("U", "T"), 1.0)
    return float(score)
