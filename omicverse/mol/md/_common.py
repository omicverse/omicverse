r"""Shared plumbing for :mod:`omicverse.mol.md` — dependency gates, OpenMM
platform selection, and provenance helpers.

Nothing heavy is imported at module scope: ``openmm`` / ``mdtraj`` /
``pdbfixer`` are pulled in *inside* functions so ``import omicverse.mol``
stays free even without the ``[md]`` extra.
"""

from __future__ import annotations

import hashlib
import importlib
import os
from typing import Any, Dict, Optional, Tuple

# Environment override for the compute device, e.g. OMICOS_MD_DEVICE=CUDA
DEVICE_ENV = "OMICOS_MD_DEVICE"

_INSTALL_HINT = (
    "  pip install 'omicverse[md]'\n"
    "or, for the most reliable CUDA build:\n"
    "  conda install -c conda-forge openmm mdtraj "
    "openmmforcefields openff-toolkit\n"
    "(conda-forge is recommended on GPU machines — the CUDA build of OpenMM "
    "is far more likely to match your driver than a pip wheel, and pip's "
    "OpenMM <8.2 is incompatible with NumPy 2. PDBFixer ships inside "
    "omicverse, so it never needs installing separately.)"
)


def need_md(module: str, feature: str = "ov.mol molecular dynamics") -> Any:
    """Import an MD backend module or raise an actionable :class:`ImportError`."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on the env
        raise ImportError(
            f"{feature} needs the '{module}' package, which is not installed.\n"
            f"Install the molecular-dynamics backend with:\n{_INSTALL_HINT}"
        ) from exc


def check_md() -> None:
    """Gate for the core MD stack.

    Only OpenMM is checked: PDBFixer is vendored inside omicverse
    (:mod:`omicverse.external.pdbfixer`) because upstream does not publish it
    on PyPI, and it needs nothing beyond OpenMM + NumPy.

    OpenMM < 8.2 was compiled against the NumPy 1.x C API and dies on
    ``import openmm.app`` under NumPy 2 with a message that names neither
    package ("numpy.core.multiarray failed to import"). That combination is
    detected here and explained.
    """
    try:
        need_md("openmm")
    except ImportError:
        raise
    except Exception as exc:                       # pragma: no cover
        _raise_if_numpy2_conflict(exc)
        raise

    import openmm
    from openmm import app  # noqa: F401  — the import that actually breaks


def _raise_if_numpy2_conflict(exc: Exception) -> None:
    """Turn the opaque NumPy-2 ABI failure into an actionable message."""
    if "multiarray failed to import" not in str(exc):
        return
    import numpy as _np

    try:
        import openmm
        version = openmm.version.version
    except Exception:
        version = "<unknown>"
    raise ImportError(
        f"OpenMM {version} is incompatible with NumPy {_np.__version__}: it "
        "was built against the NumPy 1.x C API.\n"
        "Install OpenMM >= 8.2, which supports NumPy 2:\n"
        "    pip install -U 'openmm>=8.2'\n"
        "If pip reports no matching distribution, this machine's glibc is too "
        "old for the newer wheels (8.2 needs glibc >= 2.27, 8.5 >= 2.34). Use "
        "conda-forge, whose build does not depend on the system glibc:\n"
        "    conda install -c conda-forge openmm mdtraj\n"
        "(Downgrading NumPy below 2 would also work, but that usually breaks "
        "the rest of the omicverse stack.)"
    ) from exc


def check_mdtraj() -> None:
    """Gate for the trajectory-analysis layer."""
    need_md("mdtraj", "ov.mol trajectory analysis")


# ------------------------------------------------------------------ #
# platform / device                                                    #
# ------------------------------------------------------------------ #
def resolve_platform(platform: str = "auto", precision: str = "mixed",
                     verbose: bool = True) -> Tuple[Any, Dict[str, str]]:
    """Pick an OpenMM ``Platform`` and its properties.

    ``platform='auto'`` tries CUDA → OpenCL → CPU. The ``OMICOS_MD_DEVICE``
    environment variable overrides the argument. Returns
    ``(platform, properties)``; ``properties`` carries the mixed/single/double
    precision request on the GPU platforms (ignored on CPU).

    A CPU fallback emits a loud warning — MD on CPU is orders of magnitude
    slower and is only appropriate for toy systems.
    """
    import warnings

    openmm = need_md("openmm")
    Platform = openmm.Platform

    override = os.environ.get(DEVICE_ENV)
    if override:
        platform = override

    if platform is None or str(platform).lower() == "auto":
        order = ["CUDA", "OpenCL", "CPU"]
    else:
        order = [str(platform)]

    if precision not in ("mixed", "single", "double"):
        raise ValueError(
            f"precision must be 'mixed', 'single' or 'double', got {precision!r}")

    errors = []
    for name in order:
        try:
            plat = Platform.getPlatformByName(name)
        except Exception as exc:  # unknown / unavailable platform
            errors.append(f"{name}: {exc}")
            continue
        props: Dict[str, str] = {}
        if name in ("CUDA", "OpenCL"):
            props["Precision"] = precision
        if name == "CPU" and order == ["CUDA", "OpenCL", "CPU"]:
            warnings.warn(
                "No GPU platform available — falling back to the CPU platform. "
                "Molecular dynamics on CPU is 10-100x slower; expect toy "
                f"systems only. Set {DEVICE_ENV}=CUDA (and install a CUDA "
                "build of OpenMM) to use the GPU.",
                RuntimeWarning, stacklevel=2)
        if verbose:
            print(f"   OpenMM platform: {name}"
                  + (f" (precision={precision})" if props else ""))
        return plat, props

    raise RuntimeError(
        "could not initialise any OpenMM platform. Tried: "
        + "; ".join(errors)
        + f"\nInstall a working OpenMM build:\n{_INSTALL_HINT}")


def platform_report(simulation) -> str:
    """Name of the platform a running :class:`Simulation` actually uses."""
    return simulation.context.getPlatform().getName()


# ------------------------------------------------------------------ #
# provenance                                                           #
# ------------------------------------------------------------------ #
def file_hash(path: Optional[str], *, algo: str = "sha256",
              chunk: int = 1 << 20) -> Optional[str]:
    """Short content hash of a file (first 16 hex chars), or ``None``."""
    if not path or not os.path.exists(path):
        return None
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:16]


def engine_versions() -> Dict[str, Optional[str]]:
    """Versions of whichever MD backends are importable (for provenance)."""
    out: Dict[str, Optional[str]] = {}
    for mod in ("openmm", "mdtraj", "pdbfixer", "openmmforcefields",
                "openff.toolkit"):
        try:
            m = importlib.import_module(mod)
            out[mod] = getattr(m, "__version__", None)
        except Exception:
            out[mod] = None
    return out


def base_provenance(**kwargs) -> Dict[str, Any]:
    """Seed a provenance dict with engine versions plus caller-supplied keys."""
    prov: Dict[str, Any] = {"engine": "openmm", "versions": engine_versions()}
    prov.update(kwargs)
    return prov
