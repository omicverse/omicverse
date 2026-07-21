r"""Device selection helper for the GPU-backed corners of :mod:`omicverse.synbio`.

The synbio layer mixes CPU-only tools (COBRApy, DNAchisel, primer3) with
protein models that want a GPU (ESMFold, ESM-2, ProteinMPNN, …).  Every
GPU-capable function funnels its device choice through :func:`resolve_device`
so behaviour is consistent and overridable:

Resolution order (first hit wins):

1. an explicit ``device=`` argument passed to the calling function,
2. the ``OMICOS_SYNBIO_DEVICE`` environment variable,
3. ``"cuda"`` when :func:`torch.cuda.is_available` is true, else ``"cpu"``.

Heavy models (ESMFold, large ESM-2, RFdiffusion) are effectively unusable on
CPU; instead of letting them silently churn for hours, the calling function
should gate on :func:`require_gpu`, which raises an actionable error naming a
smaller fallback model.
"""
from __future__ import annotations

import os
from typing import Optional

_ENV_VAR = "OMICOS_SYNBIO_DEVICE"


def _torch():
    """Import torch lazily with an actionable message (torch is a core
    omicverse dependency, but keep the error friendly just in case)."""
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.synbio 的蛋白模型需要 PyTorch。请 pip install torch "
            "(or `pip install 'omicverse[synbio]'`)."
        ) from exc


def cuda_available() -> bool:
    """True when a CUDA device is visible to PyTorch."""
    try:
        return bool(_torch().cuda.is_available())
    except Exception:  # pragma: no cover - torch missing / broken CUDA
        return False


def resolve_device(device: Optional[str] = None) -> str:
    """Return the concrete device string (``"cuda"``/``"cuda:0"``/``"cpu"``).

    Parameters
    ----------
    device
        Explicit override.  ``None`` (the default) consults
        ``OMICOS_SYNBIO_DEVICE`` and then CUDA availability.
    """
    if device is not None:
        return str(device)
    env = os.environ.get(_ENV_VAR)
    if env:
        return env
    return "cuda" if cuda_available() else "cpu"


def is_cuda(device: str) -> bool:
    """True for any CUDA device string (``"cuda"`` or ``"cuda:N"``)."""
    return str(device).lower().startswith("cuda")


def require_gpu(device: str, fn_name: str, small_model_hint: str = "") -> str:
    """Raise if *device* is not a GPU, for functions that are impractical on CPU.

    Returns the device unchanged when it is a CUDA device so callers can write
    ``device = require_gpu(resolve_device(device), "predict_structure")``.
    """
    if is_cuda(device):
        if not cuda_available():
            raise RuntimeError(
                f"{fn_name} 被要求在 '{device}' 上运行,但当前没有可用的 CUDA 设备"
                f"(torch.cuda.is_available() == False)。"
            )
        return device
    hint = f" 或换小模型 {small_model_hint}" if small_model_hint else ""
    raise RuntimeError(
        f"{fn_name} 需要 GPU;当前 device='{device}' 无 CUDA。"
        f"请设置可用的 device=（或 OMICOS_SYNBIO_DEVICE=cuda）{hint}。"
    )


def describe_device(device: str) -> str:
    """Human-readable one-liner about *device* (name + memory for CUDA)."""
    if not is_cuda(device):
        return "cpu"
    try:
        torch = _torch()
        idx = 0
        if ":" in device:
            idx = int(device.split(":", 1)[1])
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        return f"{device} ({name}, {total:.0f} GB)"
    except Exception:  # pragma: no cover
        return device


def warn_if_cpu(device: str, fn_name: str) -> None:
    """Emit a soft log note when a GPU-capable-but-CPU-ok function runs on CPU."""
    if not is_cuda(device):
        import logging
        logging.getLogger("omicverse.synbio").info(
            "%s: 未检测到 GPU,使用 CPU (较慢)。", fn_name
        )
