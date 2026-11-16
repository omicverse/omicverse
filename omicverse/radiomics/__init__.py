"""Optional radiomics integration backed by the independent ov-radiomics package."""

from importlib import import_module
from types import ModuleType
from typing import Any, List

_EXPORTS = (
    "image",
    "mask",
    "ImageSample",
    "extract",
    "FeatureTable",
    "fit_fusion",
    "load_model",
    "evaluate",
    "interop",
)
__all__ = list(_EXPORTS)

_INSTALL = "Install it with: pip install 'omicverse[radiomics]'"


def _load() -> ModuleType:
    try:
        backend = import_module("ov_radiomics")
    except ImportError as exc:
        raise ImportError(
            "omicverse.radiomics requires the optional 'ov-radiomics' package. "
            + _INSTALL
        ) from exc

    # The adapter supports the stable 0.1 release series. Reject prerelease or
    # malformed identities as well as incompatible major/minor releases before
    # forwarding API objects. No packaging or imaging library is imported here.
    version = getattr(backend, "__version__", "")
    release = version.partition("+")[0] if isinstance(version, str) else ""
    components = release.split(".")
    if (
        len(components) < 3
        or components[:2] != ["0", "1"]
        or not all(component.isdecimal() for component in components)
    ):
        raise ImportError(
            "omicverse.radiomics requires ov-radiomics>=0.1.0,<0.2 "
            f"(found {version!r}). " + _INSTALL
        )
    return backend


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(_load(), name)


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_EXPORTS))
