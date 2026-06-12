from __future__ import annotations

import importlib
from typing import Iterable


def normalize_torch_device(device=None, *, prefer_cuda=True, fallback_to_cpu=False):
    """Return a torch.device with bare CUDA normalized to the current index."""
    import torch

    if device is None:
        device = "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"
    if isinstance(device, int):
        if not torch.cuda.is_available() and fallback_to_cpu:
            return torch.device("cpu")
        return torch.device("cuda", device)

    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            return torch.device("cpu") if fallback_to_cpu else device
        if device.index is None:
            return torch.device("cuda", torch.cuda.current_device())
    return device


def _normalize_dependencies(dependencies: Iterable[str]) -> tuple[str, ...]:
    normalized = []
    for dep in dependencies:
        if dep and dep not in normalized:
            normalized.append(dep)
    return tuple(normalized)


def missing_optional_dependency(exc: BaseException, dependencies: Iterable[str]) -> bool:
    deps = _normalize_dependencies(dependencies)
    if not deps:
        return False

    missing_name = getattr(exc, "name", None)
    if isinstance(missing_name, str):
        if any(missing_name == dep or missing_name.startswith(dep + ".") for dep in deps):
            return True

    text = str(exc)
    return any(
        f"No module named '{dep}'" in text
        or f'No module named "{dep}"' in text
        or f"blocked import: {dep}" in text
        or f"blocked import: {dep}." in text
        for dep in deps
    )


def build_optional_dependency_error(
    feature: str,
    dependencies: Iterable[str],
    *,
    install_hint: str | None = None,
) -> ImportError:
    deps = _normalize_dependencies(dependencies)
    dep_text = ", ".join(f"`{dep}`" for dep in deps)
    message = f"{feature} requires the optional dependencies {dep_text}."
    if install_hint:
        message = f"{message} {install_hint}"
    return ImportError(message)


def has_optional_dependencies(dependencies: Iterable[str]) -> bool:
    deps = _normalize_dependencies(dependencies)
    for dep in deps:
        try:
            importlib.import_module(dep)
        except Exception:
            return False
    return True


def import_optional_module(
    module_path: str,
    *,
    package: str | None = None,
    feature: str,
    dependencies: Iterable[str],
    install_hint: str | None = None,
):
    try:
        return importlib.import_module(module_path, package=package)
    except ImportError as exc:
        if missing_optional_dependency(exc, dependencies):
            raise build_optional_dependency_error(
                feature,
                dependencies,
                install_hint=install_hint,
            ) from exc
        raise


class OptionalDependencyProxy:
    def __init__(
        self,
        symbol_name: str,
        *,
        feature: str,
        dependencies: Iterable[str],
        install_hint: str | None = None,
    ) -> None:
        self.__name__ = symbol_name
        self.__qualname__ = symbol_name
        self._feature = feature
        self._dependencies = _normalize_dependencies(dependencies)
        self._install_hint = install_hint

    def _raise(self) -> None:
        raise build_optional_dependency_error(
            self._feature,
            self._dependencies,
            install_hint=self._install_hint,
        )

    def __call__(self, *args, **kwargs):
        self._raise()

    def __getattr__(self, name: str):
        self._raise()

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        deps = ", ".join(self._dependencies)
        return f"<OptionalDependencyProxy {self.__name__} requires {deps}>"


def optional_dependency_proxy(
    symbol_name: str,
    *,
    feature: str,
    dependencies: Iterable[str],
    install_hint: str | None = None,
) -> OptionalDependencyProxy:
    return OptionalDependencyProxy(
        symbol_name,
        feature=feature,
        dependencies=dependencies,
        install_hint=install_hint,
    )


def export_optional_dependency_proxies(
    namespace: dict,
    symbol_names: Iterable[str],
    *,
    feature: str,
    dependencies: Iterable[str],
    install_hint: str | None = None,
) -> None:
    for symbol_name in symbol_names:
        namespace[symbol_name] = optional_dependency_proxy(
            symbol_name,
            feature=feature,
            dependencies=dependencies,
            install_hint=install_hint,
        )


def bind_optional_symbols(
    namespace: dict,
    module_path: str,
    symbol_names: Iterable[str],
    *,
    package: str | None = None,
    feature: str,
    dependencies: Iterable[str],
    install_hint: str | None = None,
) -> None:
    symbols = tuple(symbol_names)

    if not has_optional_dependencies(dependencies):
        export_optional_dependency_proxies(
            namespace,
            symbols,
            feature=feature,
            dependencies=dependencies,
            install_hint=install_hint,
        )
        return

    try:
        module = importlib.import_module(module_path, package=package)
    except ImportError as exc:
        if missing_optional_dependency(exc, dependencies):
            export_optional_dependency_proxies(
                namespace,
                symbols,
                feature=feature,
                dependencies=dependencies,
                install_hint=install_hint,
            )
            return
        raise

    for symbol_name in symbols:
        namespace[symbol_name] = getattr(module, symbol_name)
