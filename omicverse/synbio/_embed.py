r"""Layer B — ESM-2 protein embeddings for downstream ML.

Turn a list of protein sequences into fixed-length vectors (mean-pooled
per-residue representations from an ESM-2 layer).  These feed any downstream
model — enzyme property regressors, clustering, function transfer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Union

from .._registry import register_function
from ._device import describe_device, warn_if_cpu
from ._esm_common import load_esm

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


@register_function(
    aliases=[
        "protein_embed", "蛋白嵌入", "蛋白质嵌入", "protein_embedding",
        "esm_embedding", "蛋白特征", "esm嵌入",
    ],
    category="synthetic_biology",
    description="蛋白质嵌入:用 ESM-2 把序列编码为定长向量(均值池化 per-residue 表示),供下游 ML/聚类/性质回归。ESM-2 protein embeddings (mean-pooled) → numpy array.",
    examples=[
        "X = ov.synbio.protein_embed([seq1, seq2], model='esm2_t33_650M')",
        "X.shape  # (n_seqs, embed_dim)",
    ],
    related=["synbio.variant_effect", "synbio.enzyme_kcat", "synbio.predict_structure"],
    requires={},
    produces={},
)
def protein_embed(
    seqs: Union[str, Sequence[str]],
    model: str = "esm2_t33_650M",
    device: Optional[str] = None,
    pooling: str = "mean",
    batch_size: int = 8,
    return_per_residue: bool = False,
    verbose: bool = True,
) -> "np.ndarray":
    """Embed protein sequences with ESM-2.

    Parameters
    ----------
    seqs
        One sequence or a list of sequences (single-letter amino acids).
    model
        ESM-2 alias (``esm2_t6_8M`` … ``esm2_t36_3B``) or a full fair-esm name.
    device
        ``None`` = auto (CUDA if available).  The device actually used is
        printed when ``verbose``.
    pooling
        ``"mean"`` (default), ``"cls"`` (BOS token), or ``"max"``.
    batch_size
        Sequences per forward pass.
    return_per_residue
        If True, return a list of ``(L_i, dim)`` arrays instead of pooled
        vectors.

    Returns
    -------
    numpy.ndarray
        ``(n_seqs, embed_dim)`` when pooled, else a list of per-residue arrays.
    """
    import numpy as np

    if isinstance(seqs, str):
        seqs = [seqs]
    seqs = list(seqs)

    net, alphabet, batch_converter, dev, repr_layer = load_esm(model, device)
    if verbose:
        print(f"[ov.synbio.protein_embed] model={model} device={describe_device(dev)}")
    warn_if_cpu(dev, "protein_embed")

    from ._device import _torch
    torch = _torch()

    pooled: List[np.ndarray] = []
    per_res: List[np.ndarray] = []
    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start:start + batch_size]
        data = [(f"seq{start+i}", s) for i, s in enumerate(chunk)]
        _, _, toks = batch_converter(data)
        toks = toks.to(dev)
        with torch.no_grad():
            out = net(toks, repr_layers=[repr_layer], return_contacts=False)
        reps = out["representations"][repr_layer]  # (B, L, D)
        for i, s in enumerate(chunk):
            L = len(s)
            # tokens: [BOS] + residues + [EOS]; residues are 1..L
            res = reps[i, 1:L + 1]
            if return_per_residue:
                per_res.append(res.cpu().numpy())
                continue
            if pooling == "mean":
                vec = res.mean(0)
            elif pooling == "max":
                vec = res.max(0).values
            elif pooling == "cls":
                vec = reps[i, 0]
            else:
                raise ValueError(f"未知 pooling='{pooling}' (mean/max/cls)")
            pooled.append(vec.cpu().numpy())

    if return_per_residue:
        return per_res
    return np.vstack(pooled)


__all__ = ["protein_embed"]
