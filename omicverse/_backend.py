"""Compute backend selection: multithreaded C++ kernels or the reference Python code.

Every accelerated function takes ``backend=None | 'cpp' | 'python'``; ``None``
uses ``ov.settings.backend`` (default ``'cpp'``).  ``'cpp'`` runs the
anndataoom-fast kernels when the package is installed and the input is
supported, and otherwise falls back to ``'python'`` (with a one-time notice).
``'python'`` always runs omicverse's original implementation.
"""
from __future__ import annotations

import os
import re
import warnings

_VALID = ("cpp", "python")
_warned = False


def cpp_available() -> bool:
    try:
        import anndataoom_fast  # noqa: F401
    except ImportError:
        return False
    return True


_threads_set = False


def _init_threads():
    """Once: make anndataoom-fast default to the effective CPU count (it would
    otherwise use the host core count, oversubscribing cgroup / Slurm limits)."""
    global _threads_set
    _threads_set = True
    try:
        import anndataoom_fast
        if os.environ.get("OMP_NUM_THREADS", "").strip().isdigit():
            return  # user choice, honoured by the backend itself
        anndataoom_fast.set_num_threads(effective_cpu_count())
    except Exception:  # pragma: no cover - best effort
        pass


def resolve_backend(backend: str | None = None) -> str:
    """Return the effective backend, 'cpp' or 'python'."""
    global _warned
    if backend is None:
        from ._settings import settings
        backend = getattr(settings, "backend", "cpp")
    backend = str(backend).lower()
    if backend not in _VALID:
        raise ValueError(f"backend must be one of {_VALID}, got {backend!r}")
    if backend == "cpp" and not _threads_set and cpp_available():
        _init_threads()
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


# ---------------------------------------------------------------------------
# effective CPU count (affinity / cgroup quota / Slurm aware)
# ---------------------------------------------------------------------------
def _cgroup_cpu_limit(v2_path="/sys/fs/cgroup/cpu.max",
                      v1_quota="/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
                      v1_period="/sys/fs/cgroup/cpu/cpu.cfs_period_us"):
    """CPU quota of the current cgroup as a CPU count, or None when unlimited."""
    try:
        with open(v2_path) as f:  # cgroup v2: "<quota|max> <period>"
            quota, period = f.read().split()[:2]
        return None if quota == "max" else max(1, int(-(-int(quota) // int(period))))
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        with open(v1_quota) as f:
            quota = int(f.read().strip())
        with open(v1_period) as f:
            period = int(f.read().strip())
        return None if quota <= 0 or period <= 0 else max(1, int(-(-quota // period)))
    except (OSError, ValueError):
        return None


def _slurm_cpu_limit(environ=None):
    environ = os.environ if environ is None else environ
    v = environ.get("SLURM_CPUS_PER_TASK")
    if v and v.isdigit() and int(v) > 0:
        return int(v)
    v = environ.get("SLURM_JOB_CPUS_PER_NODE")  # e.g. "16", "16(x2)", "8(x2),4"
    if v:
        m = re.match(r"\d+", v)
        if m and int(m.group()) > 0:
            return int(m.group())
    return None


def effective_cpu_count(environ=None, **cgroup_paths) -> int:
    """CPUs this process may really use: ``OMP_NUM_THREADS`` when the user set
    it, else ``min(sched_getaffinity, cgroup quota, Slurm allocation)``."""
    env = os.environ if environ is None else environ
    omp = env.get("OMP_NUM_THREADS", "")
    if omp.strip().isdigit() and int(omp) > 0:
        return int(omp)
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        n = os.cpu_count() or 1
    for lim in (_cgroup_cpu_limit(**cgroup_paths), _slurm_cpu_limit(env)):
        if lim:
            n = min(n, lim)
    return max(1, n)


def default_threads(n_threads=None) -> int:
    """Thread count for anndataoom-fast calls: explicit ``n_threads`` (>0) wins."""
    if n_threads is not None and int(n_threads) > 0:
        return int(n_threads)
    return effective_cpu_count()
