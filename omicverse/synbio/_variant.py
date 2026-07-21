r"""Layer B — zero-shot variant-effect prediction (in-silico directed evolution).

Score amino-acid substitutions with an ESM language model, no labels required.
For a mutation ``p.A123V`` the effect is the log-likelihood ratio

.. math::  \text{score} = \log P(x_i = \text{mut}) - \log P(x_i = \text{wt})

positive ⇒ the model prefers the mutant (often tolerated / stabilising),
negative ⇒ disfavoured.  Two estimators: ``"masked-marginals"`` (Meier *et al.*
2021, one masked forward pass per position — more accurate) and
``"wt-marginals"`` (a single forward pass — fast).

Default is a **full saturation scan**: every position × the 19 alternative
amino acids, returned as a tidy DataFrame — a computational deep-mutational-scan.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

from .._registry import register_function
from ._device import describe_device, warn_if_cpu, _torch
from ._esm_common import load_esm

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

_AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def _parse_mutation(mut: str) -> Tuple[str, int, str]:
    """Parse ``A123V`` (1-based) → ('A', 123, 'V')."""
    mut = mut.strip()
    wt, pos, alt = mut[0], mut[1:-1], mut[-1]
    return wt, int(pos), alt


@register_function(
    aliases=[
        "variant_effect", "突变效应", "饱和突变", "定向进化", "in_silico_evolution",
        "saturation_mutagenesis", "deep_mutational_scan", "突变打分", "esm1v",
    ],
    category="synthetic_biology",
    description="变体效应预测 (零样本 in-silico 定向进化):用 ESM-1v/2 masked-marginals 对突变打分,默认全饱和扫描 → ΔΔ 打分表。Zero-shot variant-effect / saturation mutagenesis with ESM.",
    examples=[
        "df = ov.synbio.variant_effect(seq)                    # full saturation scan",
        "df = ov.synbio.variant_effect(seq, mutations=['A23V','K30E'])",
    ],
    related=["synbio.protein_embed", "synbio.stability_ddg", "synbio.predict_structure"],
    requires={},
    produces={},
)
def variant_effect(
    seq: str,
    mutations: Optional[Sequence[str]] = None,
    model: str = "esm1v",
    scoring: str = "masked-marginals",
    device: Optional[str] = None,
    verbose: bool = True,
) -> "pd.DataFrame":
    """Score substitutions in *seq* with an ESM language model.

    Parameters
    ----------
    seq
        Wild-type amino-acid sequence.
    mutations
        List like ``["A23V", "K30E"]`` (1-based, wt-pos-alt).  ``None`` runs a
        full saturation scan (all positions × 19 substitutions).
    model
        ``esm1v`` (default, tuned for variant effects), ``esm2_t33_650M``, etc.
    scoring
        ``"masked-marginals"`` (default) or ``"wt-marginals"``.
    device
        ``None`` = auto.

    Returns
    -------
    pandas.DataFrame
        Columns ``mutation, wt, pos, mut, score`` sorted by descending score
        (most-favoured substitutions first).
    """
    import numpy as np
    import pandas as pd

    net, alphabet, batch_converter, dev, repr_layer = load_esm(model, device)
    if verbose:
        print(f"[ov.synbio.variant_effect] model={model} scoring={scoring} "
              f"device={describe_device(dev)}")
    warn_if_cpu(dev, "variant_effect")
    torch = _torch()

    aa_to_idx = {aa: alphabet.get_idx(aa) for aa in _AA20}
    data = [("wt", seq)]
    _, _, toks = batch_converter(data)
    toks = toks.to(dev)

    # positions to evaluate (1-based sequence positions)
    if mutations is None:
        positions = list(range(1, len(seq) + 1))
    else:
        positions = sorted({_parse_mutation(m)[1] for m in mutations})

    # log-prob table: {pos: np.array over 20 aa}
    logprobs = {}
    if scoring == "wt-marginals":
        with torch.no_grad():
            logits = net(toks)["logits"]
        lp = torch.log_softmax(logits, dim=-1)[0]  # (L+2, vocab)
        for pos in positions:
            row = lp[pos]  # token index = pos (after BOS at 0)
            logprobs[pos] = np.array([row[aa_to_idx[a]].item() for a in _AA20])
    elif scoring == "masked-marginals":
        mask_idx = alphabet.mask_idx
        for pos in positions:
            batch = toks.clone()
            batch[0, pos] = mask_idx
            with torch.no_grad():
                logits = net(batch)["logits"]
            row = torch.log_softmax(logits, dim=-1)[0, pos]
            logprobs[pos] = np.array([row[aa_to_idx[a]].item() for a in _AA20])
    else:
        raise ValueError(f"未知 scoring='{scoring}' (masked-marginals/wt-marginals)")

    rows = []
    if mutations is None:
        for pos in positions:
            wt = seq[pos - 1]
            if wt not in aa_to_idx:
                continue
            wt_lp = logprobs[pos][_AA20.index(wt)]
            for j, alt in enumerate(_AA20):
                if alt == wt:
                    continue
                rows.append((f"{wt}{pos}{alt}", wt, pos, alt,
                             float(logprobs[pos][j] - wt_lp)))
    else:
        for m in mutations:
            wt, pos, alt = _parse_mutation(m)
            wt_lp = logprobs[pos][_AA20.index(wt)]
            alt_lp = logprobs[pos][_AA20.index(alt)]
            rows.append((m, wt, pos, alt, float(alt_lp - wt_lp)))

    df = pd.DataFrame(rows, columns=["mutation", "wt", "pos", "mut", "score"])
    return df.sort_values("score", ascending=False).reset_index(drop=True)


__all__ = ["variant_effect"]
