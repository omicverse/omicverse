r"""Shared ESM (Evolutionary Scale Modeling) plumbing for the protein layer.

Loads `fair-esm <https://github.com/facebookresearch/esm>`_ models once, caches
them per (name, device), and points the torch-hub download cache at
``~/.omicverse/synbio_weights`` (override with ``OMICOS_SYNBIO_WEIGHTS``) so the
2–3 GB checkpoints don't land in ``$HOME``.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

from ._device import _torch, resolve_device, is_cuda, cuda_available

# short aliases -> fair-esm pretrained loader names
ESM_MODELS = {
    "esm2_t6_8M": "esm2_t6_8M_UR50D",
    "esm2_t12_35M": "esm2_t12_35M_UR50D",
    "esm2_t30_150M": "esm2_t30_150M_UR50D",
    "esm2_t33_650M": "esm2_t33_650M_UR50D",
    "esm2_t36_3B": "esm2_t36_3B_UR50D",
    "esm2_t48_15B": "esm2_t48_15B_UR50D",
    "esm1v": "esm1v_t33_650M_UR90S_1",
    "esm1b": "esm1b_t33_650M_UR50S",
}

# models that realistically need a GPU (params too large for comfortable CPU).
_GPU_ONLY = {"esm2_t36_3B", "esm2_t48_15B"}

_CACHE: Dict[Tuple[str, str], tuple] = {}


def weights_dir() -> str:
    d = os.environ.get(
        "OMICOS_SYNBIO_WEIGHTS",
        os.path.join(os.path.expanduser("~"), ".omicverse", "synbio_weights"),
    )
    os.makedirs(d, exist_ok=True)
    return d


def _set_hub_cache():
    """Route torch.hub downloads to the synbio weights dir."""
    os.environ.setdefault("TORCH_HOME", weights_dir())


def resolve_esm_name(model: str) -> str:
    if model in ESM_MODELS:
        return ESM_MODELS[model]
    # allow passing the full fair-esm name directly
    return model


def num_layers_for(model: str) -> int:
    """Best-effort final-layer index for representation extraction."""
    name = model.lower()
    for tag in ("t48", "t36", "t33", "t30", "t12", "t6"):
        if tag in name:
            return int(tag[1:])
    return 33


def load_esm(model: str, device: str = None):
    """Return ``(model, alphabet, batch_converter, device, repr_layer)``.

    Caches per (name, device).  Raises an actionable error if a genuinely
    GPU-only checkpoint is requested on CPU.
    """
    try:
        import esm  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.synbio 的蛋白语言模型需要 fair-esm。请 pip install "
            "'omicverse[synbio]' (或 pip install fair-esm)。"
        ) from exc

    torch = _torch()
    device = resolve_device(device)

    short = model
    if short in _GPU_ONLY and not is_cuda(device):
        raise RuntimeError(
            f"protein model '{short}' 需要 GPU;当前 device='{device}' 无 CUDA。"
            f" 可设 device='cuda' 或换小模型 model='esm2_t33_650M'。"
        )

    key = (model, device)
    if key in _CACHE:
        return _CACHE[key]

    _set_hub_cache()
    import esm as _esm

    loader_name = resolve_esm_name(model)
    loader = getattr(_esm.pretrained, loader_name)
    net, alphabet = loader()
    net = net.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    repr_layer = num_layers_for(loader_name)
    out = (net, alphabet, batch_converter, device, repr_layer)
    _CACHE[key] = out
    return out
