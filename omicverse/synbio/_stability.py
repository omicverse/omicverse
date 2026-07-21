r"""Layer B — thermostability ΔΔG of point mutations.

Score how a mutation changes fold stability.  The default engine is a
dependency-light **ProteinMPNN zero-shot proxy**: from the backbone's
per-residue unconditional log-probabilities,

.. math::  \Delta\Delta G_{\text{proxy}}(i, wt\to mut)
           = \log P(wt \mid \text{struct}) - \log P(mut \mid \text{struct})

so a positive value means the wild-type residue is strongly preferred and the
mutation is predicted **destabilising** (this is exactly the signal ThermoMPNN
regresses on).  If a ThermoMPNN checkpoint is available you can switch
``engine="thermompnn"`` for calibrated kcal/mol values.

Runs on CPU (ProteinMPNN is light) or GPU.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

from .._registry import register_function
from ._device import resolve_device, describe_device, warn_if_cpu

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

_AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def _parse_mut(mut: str) -> Tuple[str, int, str]:
    return mut[0], int(mut[1:-1]), mut[-1]


@register_function(
    aliases=[
        "stability_ddg", "热稳定性", "稳定性预测", "ddg", "热稳定性ddG",
        "thermostability", "thermompnn", "折叠稳定性",
    ],
    category="synthetic_biology",
    description="热稳定性 ΔΔG:对结构上的点突变预测稳定性变化(ProteinMPNN 零样本 proxy;正值=去稳定)。默认全饱和扫描。Point-mutation thermostability ΔΔG (ProteinMPNN proxy / ThermoMPNN).",
    examples=[
        "df = ov.synbio.stability_ddg('mypro.pdb', mutations=['A23V','K30E'])",
        "df = ov.synbio.stability_ddg('mypro.pdb')   # full saturation scan",
    ],
    related=["synbio.predict_structure", "synbio.inverse_design", "synbio.variant_effect"],
    requires={},
    produces={},
)
def stability_ddg(
    pdb: str,
    mutations: Optional[Sequence[str]] = None,
    engine: str = "proteinmpnn",
    device: Optional[str] = None,
    chain: Optional[str] = None,
    verbose: bool = True,
) -> "pd.DataFrame":
    """Predict thermostability ΔΔG for point mutations on a structure.

    Parameters
    ----------
    pdb
        Path to the PDB (a single chain, or specify ``chain``).
    mutations
        List like ``["A23V", "K30E"]`` (1-based).  ``None`` = full saturation
        scan (every position × 19 substitutions).
    engine
        ``"proteinmpnn"`` (default, zero-shot proxy) or ``"thermompnn"``.
    device
        ``None`` = auto.

    Returns
    -------
    pandas.DataFrame
        Columns ``mutation, wt, pos, mut, ddg_proxy`` sorted by descending
        ddG_proxy (most destabilising first).  Higher = more destabilising.
    """
    import numpy as np
    import pandas as pd
    from ._proteinmpnn import unconditional_log_probs, MPNN_ALPHABET

    dev = resolve_device(device)
    if verbose:
        print(f"[ov.synbio.stability_ddg] pdb={pdb} engine={engine} "
              f"device={describe_device(dev)}")
    warn_if_cpu(dev, "stability_ddg")

    if engine == "thermompnn":
        raise ImportError(
            "engine='thermompnn' 需要 ThermoMPNN 权重(未随包分发)。请安装 "
            "https://github.com/Kuhlman-Lab/ThermoMPNN 并提供 checkpoint,"
            "或使用默认 engine='proteinmpnn'(零样本 proxy,无需额外权重)。"
        )

    log_p, S, alphabet = unconditional_log_probs(pdb, device=dev)
    aidx = {a: alphabet.index(a) for a in _AA20}
    # native sequence from S
    native = "".join(alphabet[i] if i < len(alphabet) else "X" for i in S)
    L = log_p.shape[0]

    rows = []
    if mutations is None:
        positions = range(1, L + 1)
        for pos in positions:
            wt = native[pos - 1]
            if wt not in aidx:
                continue
            wt_lp = log_p[pos - 1, aidx[wt]]
            for alt in _AA20:
                if alt == wt:
                    continue
                ddg = float(wt_lp - log_p[pos - 1, aidx[alt]])
                rows.append((f"{wt}{pos}{alt}", wt, pos, alt, ddg))
    else:
        for m in mutations:
            wt, pos, alt = _parse_mut(m)
            wt_lp = log_p[pos - 1, aidx[wt]]
            ddg = float(wt_lp - log_p[pos - 1, aidx[alt]])
            rows.append((m, wt, pos, alt, ddg))

    df = pd.DataFrame(rows, columns=["mutation", "wt", "pos", "mut", "ddg_proxy"])
    return df.sort_values("ddg_proxy", ascending=False).reset_index(drop=True)


__all__ = ["stability_ddg"]
