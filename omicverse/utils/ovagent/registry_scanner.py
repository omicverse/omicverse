"""Static registry scanner — AST-based function discovery for code generation.

Extracted from ``smart_agent.py`` to reduce the facade to a thin composer.
Scans ``@register_function``-decorated definitions across the omicverse
package tree and produces registry-like records that the codegen pipeline
and tool runtime can use for function retrieval and scoring.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..._registry import _global_registry, _hydrate_registry_for_export

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


class RegistryScanner:
    """AST-based static registry scanner with caching and scoring.

    Provides all functionality previously located directly on
    ``OmicVerseAgent`` for static function discovery.
    """

    def __init__(self) -> None:
        self._static_registry_entries_cache: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # Aggregate collection
    # ------------------------------------------------------------------

    def collect_relevant_entries(
        self,
        request: str,
        max_entries: int = 8,
    ) -> List[Dict[str, Any]]:
        """Return a compact set of registry entries relevant to *request*."""

        if max_entries <= 0:
            return []

        self.ensure_runtime_registry()

        merged: Dict[str, Dict[str, Any]] = {}
        for raw_entry in [*self._iter_runtime_entries(), *self.load_static_entries()]:
            entry = self.normalize_entry(raw_entry)
            full_name = str(entry.get("full_name", "") or "")
            if not full_name:
                continue
            current = merged.get(full_name)
            if current is None:
                merged[full_name] = entry
            else:
                merged[full_name] = self._merge_entries(current, entry)

        ranked: List[Tuple[Tuple[float, ...], Dict[str, Any]]] = []
        for entry in merged.values():
            sort_key = self.rank_entry(request, entry)
            if sort_key is None:
                continue
            ranked.append((sort_key, entry))

        ranked.sort(
            key=lambda item: (item[0], item[1].get("full_name", "")),
            reverse=True,
        )
        return [entry for _, entry in ranked[:max_entries]]

    # ------------------------------------------------------------------
    # Runtime registry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def ensure_runtime_registry() -> None:
        """Hydrate the runtime registry.

        Prefer the same broad hydration path used by ``ov.Agent`` registry
        export helpers so lookup wrappers see the full public registry rather
        than only MCP rollout-whitelisted modules. Fall back to MCP manifest
        hydration when broad package imports are unavailable in the current
        runtime.
        """

        # A partially-populated registry (e.g. just the handful of functions
        # whose modules happened to import at ``import omicverse`` time) must
        # NOT skip full hydration — otherwise every lookup only ever sees those
        # few. Gate on a *hydration-done* memo, not on emptiness.
        if getattr(_global_registry, "_hydrated", False):
            return

        try:
            ensure = getattr(_global_registry, "_ensure_hydrated", None)
            if callable(ensure):
                ensure()
            else:
                _hydrate_registry_for_export()
        except Exception:
            logger.warning(
                "Failed to hydrate the runtime registry via full package import",
                exc_info=True,
            )

        if getattr(_global_registry, "_registry", None):
            return

        try:
            from ...mcp.manifest import ensure_registry_populated

            ensure_registry_populated()
        except Exception:
            logger.warning(
                "Failed to hydrate the runtime registry from MCP manifest",
                exc_info=True,
            )

    @staticmethod
    def collect_runtime_entries(
        request: str,
        max_entries: int = 8,
    ) -> List[Dict[str, Any]]:
        """Search hydrated runtime registry entries using unified scoring."""

        if max_entries <= 0:
            return []

        ranked: List[Tuple[Tuple[float, ...], Dict[str, Any]]] = []
        for raw_entry in RegistryScanner._iter_runtime_entries():
            entry = RegistryScanner.normalize_entry(raw_entry)
            sort_key = RegistryScanner.rank_entry(request, entry)
            if sort_key is None:
                continue
            ranked.append((sort_key, entry))

        ranked.sort(
            key=lambda item: (item[0], item[1].get("full_name", "")),
            reverse=True,
        )
        return [entry for _, entry in ranked[:max_entries]]

    # ------------------------------------------------------------------
    # Normalization / scoring
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Convert registry entries to public-facing ov.* names for code generation."""

        normalized = dict(entry)
        source = normalized.get("source")
        normalized["source"] = source if str(source).startswith("static_ast") else "runtime"

        original_full_name = str(normalized.get("full_name", "") or "")
        normalized["registry_full_name"] = original_full_name

        public_name = original_full_name
        parent_full_name = str(normalized.get("parent_full_name", "") or "")

        if original_full_name.startswith("omicverse."):
            branch_suffix = ""
            stem = original_full_name
            if "[" in original_full_name and original_full_name.endswith("]"):
                stem, branch_suffix = original_full_name.split("[", 1)
                branch_suffix = "[" + branch_suffix

            if parent_full_name and stem.startswith(parent_full_name + "."):
                parent_public = RegistryScanner.normalize_entry(
                    {"full_name": parent_full_name}
                ).get("full_name", parent_full_name)
                method_name = stem[len(parent_full_name) + 1 :]
                public_name = f"{parent_public}.{method_name}{branch_suffix}"
            elif parent_full_name and stem == parent_full_name:
                parent_public = RegistryScanner.normalize_entry(
                    {"full_name": parent_full_name}
                ).get("full_name", parent_full_name)
                public_name = f"{parent_public}{branch_suffix}"
            else:
                parts = stem.split(".")
                if len(parts) >= 2:
                    domain = parts[1]
                    leaf_name = parts[-1]
                    if domain == "_settings":
                        public_name = f"ov.core.{leaf_name}{branch_suffix}"
                    elif domain:
                        public_name = f"ov.{domain}.{leaf_name}{branch_suffix}"

        normalized["full_name"] = public_name
        if parent_full_name:
            normalized["parent_full_name"] = RegistryScanner.normalize_entry(
                {"full_name": parent_full_name}
            ).get("full_name", parent_full_name)
        return normalized

    @staticmethod
    def score_entry(
        request: str,
        entry: Dict[str, Any],
    ) -> float:
        """Return a scalar score preserving the same ordering as ``rank_entry``."""

        sort_key = RegistryScanner.rank_entry(request, entry)
        if sort_key is None:
            return 0.0

        exact_rank, alias_score, name_score, kind_score, weak_score, contract_score, source_rank = sort_key
        return (
            exact_rank * 1000.0
            + alias_score * 100.0
            + name_score * 10.0
            + kind_score
            + weak_score * 0.1
            + contract_score * 0.01
            + source_rank * 0.001
        )

    @staticmethod
    def rank_entry(
        request: str,
        entry: Dict[str, Any],
    ) -> Optional[Tuple[float, float, float, float, float, float, float]]:
        """Return a structured sort key for unified runtime/static ranking."""

        query = (request or "").strip().lower()
        if not query:
            return None

        normalized = RegistryScanner.normalize_entry(entry)
        tokens = RegistryScanner._query_tokens(query)

        full_name = str(normalized.get("full_name", "") or "").lower()
        registry_full_name = str(normalized.get("registry_full_name", "") or "").lower()
        short_name = str(
            normalized.get("short_name") or normalized.get("name") or ""
        ).lower()
        aliases = [
            str(alias).strip().lower()
            for alias in (normalized.get("aliases") or [])
            if str(alias).strip()
        ]

        exact_rank = 0.0
        if query == full_name or query == registry_full_name:
            exact_rank = 4.0
        elif query == short_name:
            exact_rank = 3.0
        elif query in aliases:
            exact_rank = 3.0

        alias_score = 0.0
        strong_fields = [full_name, registry_full_name, short_name]
        if query and any(query in field for field in strong_fields if field):
            alias_score += 2.0
        if query:
            alias_score += max(
                (
                    RegistryScanner._token_overlap_score(query, alias)
                    for alias in aliases
                ),
                default=0.0,
            )
            if any(query in alias for alias in aliases):
                alias_score += 1.0

        name_score = 0.0
        for field in strong_fields:
            if not field:
                continue
            name_score = max(
                name_score,
                RegistryScanner._token_overlap_score(query, field),
            )
            if query in field:
                name_score += 1.0
        if any(token == short_name for token in tokens if short_name):
            name_score += 1.0

        kind_score = float(RegistryScanner._entry_kind_rank(normalized))

        weak_fields = [
            str(normalized.get("category", "") or ""),
            str(normalized.get("description", "") or ""),
            str(normalized.get("docstring", "") or ""),
            " ".join(normalized.get("examples", []) or []),
            " ".join(normalized.get("related", []) or []),
            " ".join(normalized.get("imports", []) or []),
        ]
        weak_score = sum(
            RegistryScanner._token_overlap_score(query, field)
            for field in weak_fields
            if field
        )

        contract_fields = [
            RegistryScanner._flatten_contract_map(normalized.get("prerequisites")),
            RegistryScanner._flatten_contract_map(normalized.get("requires")),
            RegistryScanner._flatten_contract_map(normalized.get("produces")),
        ]
        contract_score = 0.0
        for field in contract_fields:
            if not field:
                continue
            contract_score += min(
                RegistryScanner._token_overlap_score(query, field),
                1.5,
            )
            if query in field.lower():
                contract_score += 0.25

        public_name = full_name
        if public_name.startswith("ov.datasets.") and not any(
            word in query for word in ("dataset", "download", "read", "load", "example", "demo")
        ):
            weak_score -= 1.0

        if public_name.startswith("ov.core.") and not any(
            word in query for word in ("reference", "table", "gpu", "cpu", "settings")
        ):
            weak_score -= 1.0

        source_rank = 1.0 if normalized.get("source") == "runtime" else 0.0
        if max(exact_rank, alias_score, name_score, weak_score, contract_score) <= 0:
            return None
        return (
            exact_rank,
            alias_score,
            name_score,
            kind_score,
            weak_score,
            contract_score,
            source_rank,
        )

    # ------------------------------------------------------------------
    # Static AST scanning
    # ------------------------------------------------------------------

    def load_static_entries(self) -> List[Dict[str, Any]]:
        """Parse @register_function metadata plus nested method/branch capabilities."""

        if self._static_registry_entries_cache is not None:
            return self._static_registry_entries_cache

        package_root = Path(__file__).resolve().parents[2]
        search_roots = (
            "pp", "pl", "single", "bulk", "space", "utils",
            "io", "alignment", "external", "biocontext", "bulk2single", "datasets",
        )
        entries: List[Dict[str, Any]] = []
        seen: set = set()

        for root_name in search_roots:
            root = package_root / root_name
            if not root.exists():
                continue
            for file_path in sorted(root.rglob("*.py")):
                if file_path.name == "__init__.py":
                    continue
                if "__pycache__" in file_path.parts or ".ipynb_checkpoints" in file_path.parts:
                    continue
                try:
                    source = file_path.read_text(encoding="utf-8")
                    tree = ast.parse(source, filename=str(file_path))
                except Exception:
                    logger.warning("Failed to statically scan registry entries from %s", file_path, exc_info=True)
                    continue

                for node in tree.body:
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        continue
                    decorator = _find_register_function_decorator(node)
                    if decorator is None:
                        continue
                    for entry in _build_static_entries(file_path, node, decorator):
                        full_name = entry.get("full_name", "")
                        if not full_name or full_name in seen:
                            continue
                        seen.add(full_name)
                        entries.append(entry)

        self._static_registry_entries_cache = entries
        return entries

    def collect_static_entries(
        self,
        request: str,
        max_entries: int = 8,
    ) -> List[Dict[str, Any]]:
        """Search the static AST-derived registry snapshot."""

        if not request or max_entries <= 0:
            return []

        entries = self.load_static_entries()
        scored: List[Tuple[Tuple[float, ...], Dict[str, Any]]] = []

        for entry in entries:
            sort_key = self.rank_entry(request, entry)
            if sort_key is not None:
                scored.append((sort_key, entry))

        scored.sort(
            key=lambda item: (item[0], item[1].get("full_name", "")),
            reverse=True,
        )
        return [entry for _, entry in scored[:max_entries]]

    @staticmethod
    def _iter_runtime_entries() -> List[Dict[str, Any]]:
        """Return unique runtime registry entries keyed by canonical full name."""

        if not getattr(_global_registry, "_registry", None):
            return []

        unique: Dict[str, Dict[str, Any]] = {}
        for entry in getattr(_global_registry, "_registry", {}).values():
            full_name = str(entry.get("full_name", "") or "")
            if full_name and full_name not in unique:
                unique[full_name] = entry
        return list(unique.values())

    @staticmethod
    def _merge_entries(primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
        """Merge duplicate runtime/static entries under one public-facing name."""

        merged = dict(primary)

        list_fields = ("aliases", "examples", "related", "imports", "parameters")
        dict_fields = ("prerequisites", "requires", "produces")

        for field in list_fields:
            values: List[Any] = []
            for source in (primary, secondary):
                values.extend(source.get(field) or [])
            if values:
                merged[field] = list(dict.fromkeys(values))

        for field in dict_fields:
            combined: Dict[str, List[Any]] = {}
            for source in (primary, secondary):
                for key, values in (source.get(field) or {}).items():
                    combined.setdefault(str(key), [])
                    combined[str(key)].extend(values or [])
            if combined:
                merged[field] = {
                    key: list(dict.fromkeys(values))
                    for key, values in combined.items()
                }

        for field in (
            "description",
            "docstring",
            "signature",
            "module",
            "parent_full_name",
            "branch_parameter",
            "branch_value",
            "category",
            "registry_full_name",
        ):
            primary_value = merged.get(field)
            secondary_value = secondary.get(field)
            if (not primary_value) and secondary_value:
                merged[field] = secondary_value

        if primary.get("source") != "runtime" and secondary.get("source") == "runtime":
            merged["source"] = "runtime"

        return merged

    @staticmethod
    def _query_tokens(text: str) -> List[str]:
        """Tokenize mixed English/CJK queries for lightweight lexical retrieval."""

        tokens = [
            token
            for token in re.findall(r"[a-z0-9_\\.\\-]+|[\u4e00-\u9fff]+", text.lower())
            if token.strip()
        ]
        return list(dict.fromkeys(tokens))

    @staticmethod
    def _token_overlap_score(query: str, field: str) -> float:
        """Return a normalized token-overlap score between query and one field."""

        field_lower = str(field).lower()
        if not field_lower:
            return 0.0

        tokens = RegistryScanner._query_tokens(query)
        if not tokens:
            return 0.0

        score = 0.0
        for token in tokens:
            if token == field_lower:
                score += 2.0
            elif token in field_lower:
                score += 1.0
        return score

    @staticmethod
    def _entry_kind_rank(entry: Dict[str, Any]) -> int:
        """Prefer parent functions over methods, and methods over branches."""

        source = str(entry.get("source", "") or "")
        if entry.get("branch_parameter") or source.endswith("branch"):
            return 0
        if entry.get("parent_full_name") or source.endswith("method"):
            return 1
        return 2

    @staticmethod
    def _flatten_contract_map(value: Any) -> str:
        """Flatten prerequisite/contract metadata into lightweight search text."""

        if isinstance(value, dict):
            parts: List[str] = []
            for key, items in value.items():
                if isinstance(items, list):
                    parts.append(f"{key} {' '.join(str(item) for item in items)}")
                else:
                    parts.append(f"{key} {items}")
            return " ".join(parts)
        return str(value or "")


def build_compact_registry_summary(
    scanner: Optional["RegistryScanner"] = None,
) -> str:
    """Return the compact category-level registry summary used by ``ov.Agent``."""

    active_scanner = scanner or RegistryScanner()
    category_map: Dict[str, List[str]] = {}
    seen: set[str] = set()

    entries = active_scanner.load_static_entries()
    if not entries:
        for entry in getattr(_global_registry, "_registry", {}).values():
            full_name = entry.get("full_name", "")
            if full_name and full_name not in seen:
                seen.add(full_name)
                cat = entry.get("category", "other") or "other"
                category_map.setdefault(cat, []).append(full_name)
    else:
        for entry in entries:
            full_name = entry.get("full_name", "")
            if full_name and full_name not in seen:
                seen.add(full_name)
                cat = entry.get("category", "other") or "other"
                category_map.setdefault(cat, []).append(full_name)

    lines: List[str] = []
    for cat in sorted(category_map):
        names = category_map[cat]
        sample = ", ".join(names[:5])
        suffix = f" (+{len(names) - 5} more)" if len(names) > 5 else ""
        lines.append(f"- **{cat}** ({len(names)} functions): {sample}{suffix}")
    return "\n".join(lines) if lines else "No registered functions detected."


# ---------------------------------------------------------------------------
# Module-level helpers (stateless AST utilities)
# ---------------------------------------------------------------------------


def _find_register_function_decorator(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
) -> Optional[ast.Call]:
    """Return the register_function decorator call when present."""
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        func = decorator.func
        if isinstance(func, ast.Name) and func.id == "register_function":
            return decorator
        if isinstance(func, ast.Attribute) and func.attr == "register_function":
            return decorator
    return None


def _literal_eval_or_default(node: Optional[ast.AST], default: Any) -> Any:
    if node is None:
        return default
    try:
        return ast.literal_eval(node)
    except Exception:
        logger.debug("Falling back to default during AST literal evaluation", exc_info=True)
        return default


def _derive_static_signature(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
) -> str:
    """Build a lightweight signature string from AST."""
    if isinstance(node, ast.ClassDef):
        return f"{node.name}(...)"
    arg_names = [arg.arg for arg in node.args.args]
    if arg_names and arg_names[0] == "self":
        arg_names = arg_names[1:]
    return f"{node.name}({', '.join(arg_names)})"


def _collect_import_targets(statements: List[ast.stmt]) -> List[str]:
    """Collect import targets mentioned inside a function or branch body."""
    module = ast.Module(body=list(statements), type_ignores=[])
    imports: List[str] = []
    for child in ast.walk(module):
        if isinstance(child, ast.Import):
            for alias in child.names:
                if alias.name:
                    imports.append(alias.name.split(".")[0])
        elif isinstance(child, ast.ImportFrom):
            if child.module:
                imports.append(child.module.split(".")[0])
            for alias in child.names:
                if alias.name and alias.name != "*":
                    imports.append(alias.name.split(".")[0])
    return list(dict.fromkeys(imports))


def _filter_examples_for_method(examples: List[str], method_name: str) -> List[str]:
    """Keep examples most relevant to a nested method entry."""
    if not examples:
        return []
    matches = [example for example in examples if method_name in str(example)]
    return matches[:3] if matches else list(examples[:2])


def _filter_examples_for_branch(
    examples: List[str], param_name: str, branch_value: str,
) -> List[str]:
    """Keep examples mentioning the relevant branch parameter/value when possible."""
    if not examples:
        return []
    value_lower = branch_value.lower()
    param_lower = param_name.lower()
    matches = [
        example for example in examples
        if value_lower in str(example).lower() or param_lower in str(example).lower()
    ]
    return matches[:3] if matches else list(examples[:2])


def _branch_subject_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _branch_string_values(node: ast.AST) -> List[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values: List[str] = []
        for child in node.elts:
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                values.append(child.value)
        return values
    return []


def _extract_branch_variants(test: ast.AST) -> List[Tuple[str, str]]:
    """Extract simple string-dispatch branches from an ``if`` test."""
    variants: List[Tuple[str, str]] = []
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
        for value in test.values:
            variants.extend(_extract_branch_variants(value))
        return variants

    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return variants

    subject = _branch_subject_name(test.left)
    comparator = test.comparators[0]
    op = test.ops[0]
    values: List[str] = []

    if isinstance(op, ast.Eq):
        values = _branch_string_values(comparator)
    elif isinstance(op, ast.In):
        values = _branch_string_values(comparator)

    if subject and values:
        variants.extend((subject, value) for value in values)
    return variants


def _build_static_entry(
    file_path: Path,
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
    decorator: ast.Call,
) -> Optional[Dict[str, Any]]:
    """Convert one @register_function AST node into a registry-like record."""

    package_root = Path(__file__).resolve().parents[2]
    rel = file_path.relative_to(package_root)
    if not rel.parts:
        return None
    domain = rel.parts[0]
    if domain not in {"pp", "pl", "single", "bulk", "space", "utils"}:
        return None

    args = list(decorator.args)
    aliases = _literal_eval_or_default(args[0], []) if len(args) >= 1 else []
    category = _literal_eval_or_default(args[1], "") if len(args) >= 2 else ""
    description = _literal_eval_or_default(args[2], "") if len(args) >= 3 else ""
    examples = _literal_eval_or_default(args[3], []) if len(args) >= 4 else []

    kw = {item.arg: item.value for item in decorator.keywords if item.arg}
    aliases = _literal_eval_or_default(kw.get("aliases"), aliases)
    category = _literal_eval_or_default(kw.get("category"), category)
    description = _literal_eval_or_default(kw.get("description"), description)
    examples = _literal_eval_or_default(kw.get("examples"), examples)
    related = _literal_eval_or_default(kw.get("related"), [])
    prerequisites = _literal_eval_or_default(kw.get("prerequisites"), {})
    requires = _literal_eval_or_default(kw.get("requires"), {})
    produces = _literal_eval_or_default(kw.get("produces"), {})

    full_name = f"ov.{domain}.{node.name}"
    module_name = "omicverse." + ".".join(rel.with_suffix("").parts)

    return {
        "name": node.name,
        "short_name": node.name,
        "full_name": full_name,
        "module": module_name,
        "aliases": aliases or [],
        "category": category or domain,
        "description": description or (ast.get_docstring(node) or ""),
        "examples": examples or [],
        "related": related or [],
        "signature": _derive_static_signature(node),
        "docstring": ast.get_docstring(node) or "",
        "prerequisites": prerequisites or {},
        "requires": requires or {},
        "produces": produces or {},
        "source": "static_ast",
    }


def _build_static_method_entry(
    base_entry: Dict[str, Any],
    class_node: ast.ClassDef,
    method_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> Optional[Dict[str, Any]]:
    """Create a searchable entry for a public method on a registered class."""

    aliases: List[str] = [
        method_node.name,
        f"{class_node.name}.{method_node.name}",
        f"{base_entry.get('short_name', class_node.name)} {method_node.name}",
    ]
    for alias in (base_entry.get("aliases") or [])[:8]:
        alias = str(alias).strip()
        if alias:
            aliases.append(f"{alias} {method_node.name}")

    description = ast.get_docstring(method_node) or (
        f"Method `{method_node.name}` on {base_entry.get('full_name', class_node.name)}."
    )

    return {
        "name": method_node.name,
        "short_name": method_node.name,
        "full_name": f"{base_entry.get('full_name', class_node.name)}.{method_node.name}",
        "module": base_entry.get("module", ""),
        "aliases": list(dict.fromkeys(aliases)),
        "category": base_entry.get("category", ""),
        "description": description,
        "examples": _filter_examples_for_method(base_entry.get("examples", []), method_node.name),
        "related": [base_entry.get("full_name", "")],
        "signature": _derive_static_signature(method_node),
        "docstring": ast.get_docstring(method_node) or "",
        "prerequisites": base_entry.get("prerequisites", {}) or {},
        "requires": base_entry.get("requires", {}) or {},
        "produces": base_entry.get("produces", {}) or {},
        "source": "static_ast_method",
        "parent_full_name": base_entry.get("full_name", ""),
        "imports": _collect_import_targets(method_node.body),
    }


def _build_static_branch_entries(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    parent_entry: Dict[str, Any],
    owner_entry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Create searchable entries for string-dispatched branches."""

    entries: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()

    for child in ast.walk(node):
        if not isinstance(child, ast.If):
            continue
        for param_name, branch_value in _extract_branch_variants(child.test):
            key = (param_name, branch_value)
            if key in seen:
                continue
            seen.add(key)
            imports = _collect_import_targets(child.body)
            aliases = [
                branch_value,
                f"{node.name} {branch_value}",
                f"{param_name} {branch_value}",
                f"{parent_entry.get('short_name', node.name)} {branch_value}",
            ]
            for alias in (owner_entry.get("aliases") or [])[:8]:
                alias = str(alias).strip()
                if alias:
                    aliases.append(f"{alias} {branch_value}")
            examples = _filter_examples_for_branch(
                parent_entry.get("examples", []),
                param_name,
                branch_value,
            )
            description = (
                f"Variant of `{parent_entry.get('full_name', node.name)}` when "
                f"`{param_name}='{branch_value}'`."
            )
            if imports:
                description += " Imports/uses: " + ", ".join(imports) + "."

            entries.append({
                "name": branch_value,
                "short_name": branch_value,
                "full_name": f"{parent_entry.get('full_name', node.name)}[{param_name}={branch_value}]",
                "module": parent_entry.get("module", ""),
                "aliases": list(dict.fromkeys(aliases)),
                "category": parent_entry.get("category", ""),
                "description": description,
                "examples": examples,
                "related": [parent_entry.get("full_name", "")],
                "signature": parent_entry.get("signature", _derive_static_signature(node)),
                "docstring": parent_entry.get("docstring", ""),
                "prerequisites": parent_entry.get("prerequisites", {}) or {},
                "requires": parent_entry.get("requires", {}) or {},
                "produces": parent_entry.get("produces", {}) or {},
                "source": "static_ast_branch",
                "parent_full_name": parent_entry.get("full_name", ""),
                "branch_parameter": param_name,
                "branch_value": branch_value,
                "imports": imports,
            })

    return entries


def _build_nested_entries(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
    base_entry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Expand registered classes/functions into searchable method and branch entries."""

    nested: List[Dict[str, Any]] = []
    if isinstance(node, ast.ClassDef):
        for child in node.body:
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if child.name.startswith("_"):
                continue
            method_entry = _build_static_method_entry(base_entry, node, child)
            if not method_entry:
                continue
            nested.append(method_entry)
            nested.extend(_build_static_branch_entries(child, method_entry, base_entry))
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        nested.extend(_build_static_branch_entries(node, base_entry, base_entry))
    return nested


def _build_static_entries(
    file_path: Path,
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
    decorator: ast.Call,
) -> List[Dict[str, Any]]:
    """Build the primary static entry plus nested method/branch entries."""
    base_entry = _build_static_entry(file_path, node, decorator)
    if not base_entry:
        return []
    entries = [base_entry]
    entries.extend(_build_nested_entries(node, base_entry))
    return entries
