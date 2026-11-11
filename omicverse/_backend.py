"""Compute backend selection: multithreaded C++ kernels or the reference Python code.

Every accelerated function takes ``backend=None | 'cpp' | 'python'``; ``None``
uses ``ov.settings.backend`` (default ``'cpp'``).  ``'cpp'`` runs the
anndataoom-fast kernels when the package is installed and the input is
supported, and otherwise falls back to ``'python'`` (with a one-time notice).
``'python'`` always runs omicverse's original implementation.
"""
from __future__ import annotations

import warnings

_VALID = ("cpp", "python")
_warned = False


def cpp_available() -> bool:
    try:
        import anndataoom_fast  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend(backend: str | None = None) -> str:
    """Return the effective backend, 'cpp' or 'python'."""
    global _warned
    if backend is None:
        from ._settings import settings
        backend = getattr(settings, "backend", "cpp")
    backend = str(backend).lower()
    if backend not in _VALID:
        raise ValueError(f"backend must be one of {_VALID}, got {backend!r}")
    if backend == "cpp" and not cpp_available():
        if not _warned:
            warnings.warn(
                "backend='cpp' requested but anndataoom-fast is not installed; using the "
                "Python implementation (pip install anndataoom-fast to enable it).",
                stacklevel=3)
            _warned = True
        return "python"
    return backend


def oom_ops(backend: str | None = None):
    """Module providing anndataoom's ``chunked_*`` functions for this backend."""
    if resolve_backend(backend) == "cpp":
        from anndataoom_fast import oom
        return oom
    import anndataoom
    return anndataoom


def cpp_memory():
    """anndataoom_fast.memory (in-memory CSR kernels)."""
    from anndataoom_fast import memory
    return memory
