"""
Function Registry System for OmicVerse

This module provides a decorator-based function registration system that allows
users to discover functions using natural language queries, aliases, and semantic search.
"""

from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from functools import wraps
from datetime import datetime, timezone
import ast
import inspect
import textwrap
from difflib import get_close_matches
import warnings
import json
from pathlib import Path

_CAPABILITY_BRANCH_PARAMS = {
    "method",
    "backend",
    "mode",
    "source",
    "task",
    "provider",
    "api_type",
    "model",
    "doublets_method",
    "correction_method",
    "prediction_mode",
    "classifier",
    "organism",
}

_NON_CAPABILITY_BRANCH_PARAMS = {
    "name",
    "format",
    "key",
    "tag",
    "cmd",
    "kind",
    "platform",
    "cmap_name",
    "__name__",
    "file_type",
}

_CAPABILITY_BRANCH_SUFFIXES = (
    "_method",
    "_backend",
    "_provider",
    "_source",
    "_classifier",
)

_CAPABILITY_BRANCH_IMPORT_SUFFIXES = (
    "_mode",
    "_model",
    "_type",
    "_organism",
    "_task",
)


class FunctionRegistry:
    """
    Central registry for all decorated functions in OmicVerse.
    
    This class maintains a searchable index of functions with their metadata,
    including aliases, categories, descriptions, and usage examples.
    """
    
    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._function_map: Dict[Callable, str] = {}  # Maps functions to their primary keys
        self._categories: Dict[str, List[str]] = {}  # Category to function mapping

    def _store_entry(
        self,
        entry: Dict[str, Any],
        *,
        register_short_name: bool = True,
        override_aliases: bool = True,
        map_function: Optional[Callable] = None,
    ) -> None:
        """Store one registry entry and its aliases."""

        full_name = str(entry.get("full_name", "") or "").lower()
        short_name = str(entry.get("short_name", "") or "").lower()
        aliases = [str(alias).strip().lower() for alias in (entry.get("aliases") or []) if str(alias).strip()]

        def _maybe_store(key: str) -> None:
            if not key:
                return
            if override_aliases or key not in self._registry:
                self._registry[key] = entry

        for alias in aliases:
            _maybe_store(alias)

        if register_short_name:
            _maybe_store(short_name)

        if full_name:
            self._registry[full_name] = entry

        category = str(entry.get("category", "") or "")
        full_name_value = str(entry.get("full_name", "") or "")
        if category and full_name_value:
            if category not in self._categories:
                self._categories[category] = []
            if full_name_value not in self._categories[category]:
                self._categories[category].append(full_name_value)

        if map_function is not None and not entry.get("virtual_entry"):
            self._function_map[map_function] = entry.get("full_name", "")
        
    def register(self,
                 func: Callable,
                 aliases: List[str],
                 category: str,
                 description: str,
                 examples: Optional[List[str]] = None,
                 related: Optional[List[str]] = None,
                 prerequisites: Optional[Dict[str, List[str]]] = None,
                 requires: Optional[Dict[str, List[str]]] = None,
                 produces: Optional[Dict[str, List[str]]] = None,
                 auto_fix: str = 'none') -> Callable:
        """
        Register a function with its metadata.

        Parameters
        ----------
        func : Callable
            The function to register
        aliases : List[str]
            List of aliases (including Chinese, abbreviations, etc.)
        category : str
            Function category (e.g., "preprocessing", "analysis", "visualization")
        description : str
            Detailed description of the function
        examples : List[str], optional
            Usage examples
        related : List[str], optional
            Related function names
        prerequisites : Dict[str, List[str]], optional
            Function prerequisites, format:
            {'functions': ['scale'], 'optional_functions': ['qc', 'preprocess']}
        requires : Dict[str, List[str]], optional
            Required data structures, format:
            {'layers': ['scaled'], 'obsm': ['X_pca'], 'uns': ['neighbors'], ...}
        produces : Dict[str, List[str]], optional
            Data structures created by this function, format:
            {'layers': ['scaled'], 'obsm': ['X_pca'], 'uns': ['pca'], ...}
        auto_fix : str, optional
            Auto-fix strategy: 'auto' (can auto-insert prerequisites),
            'escalate' (suggest workflow), 'none' (just warn). Default: 'none'

        Returns
        -------
        Callable
            The original function unchanged
        """
        # Validate metadata completeness
        if not aliases or not all(alias.strip() for alias in aliases):
            raise ValueError("Function registration requires at least one non-empty alias.")

        if not category or not category.strip():
            raise ValueError("Function registration requires a category.")

        if not description or not description.strip():
            raise ValueError("Function registration requires a description.")

        if examples is not None:
            if isinstance(examples, (tuple, set)):
                examples = list(examples)
            elif not isinstance(examples, list):
                raise TypeError("Examples must be provided as a list of strings when specified.")

        if related is not None:
            if isinstance(related, (tuple, set)):
                related = list(related)
            elif not isinstance(related, list):
                raise TypeError("Related entries must be provided as a list of strings when specified.")

        # Validate prerequisite metadata
        if prerequisites is not None:
            if not isinstance(prerequisites, dict):
                raise TypeError("Prerequisites must be a dictionary.")
            valid_keys = {'functions', 'optional_functions'}
            for key in prerequisites.keys():
                if key not in valid_keys:
                    raise ValueError(f"Invalid prerequisites key '{key}'. Valid keys: {valid_keys}")
                if not isinstance(prerequisites[key], list):
                    raise TypeError(f"Prerequisites['{key}'] must be a list.")

        if requires is not None:
            if not isinstance(requires, dict):
                raise TypeError("Requires must be a dictionary.")
            valid_keys = {'layers', 'obsm', 'obsp', 'uns', 'var', 'obs', 'varm'}
            for key in requires.keys():
                if key not in valid_keys:
                    raise ValueError(f"Invalid requires key '{key}'. Valid keys: {valid_keys}")
                if not isinstance(requires[key], list):
                    raise TypeError(f"Requires['{key}'] must be a list.")

        if produces is not None:
            if not isinstance(produces, dict):
                raise TypeError("Produces must be a dictionary.")
            valid_keys = {'layers', 'obsm', 'obsp', 'uns', 'var', 'obs', 'varm'}
            for key in produces.keys():
                if key not in valid_keys:
                    raise ValueError(f"Invalid produces key '{key}'. Valid keys: {valid_keys}")
                if not isinstance(produces[key], list):
                    raise TypeError(f"Produces['{key}'] must be a list.")

        if auto_fix not in ('auto', 'escalate', 'none'):
            raise ValueError(f"Invalid auto_fix value '{auto_fix}'. Valid values: 'auto', 'escalate', 'none'")

        # Generate function key
        module_name = func.__module__
        func_name = func.__name__
        full_name = f"{module_name}.{func_name}"

        # Extract function signature and parameters
        try:
            sig = inspect.signature(func)
            signature = str(sig)
            
            # Extract parameter details with defaults
            params_info = []
            for param_name, param in sig.parameters.items():
                param_str = param_name
                if param.default != inspect.Parameter.empty:
                    param_str += f"={repr(param.default)}"
                params_info.append(param_str)
            
            # Get full docstring for help
            raw_doc = inspect.getdoc(func)
            if not raw_doc:
                warnings.warn(
                    f"Function '{full_name}' is missing a docstring; agent help output may be limited.",
                    UserWarning,
                )
            docstring = raw_doc or "No documentation available"
            
        except Exception:
            signature = "(adata, **kwargs)"
            params_info = ["adata", "**kwargs"]
            docstring = "No documentation available"
        
        # Create registry entry
        entry = {
            'function': func,
            'full_name': full_name,
            'short_name': func_name,
            'module': module_name,
            'aliases': aliases,
            'category': category,
            'description': description,
            'examples': examples or [],
            'related': related or [],
            'signature': signature,
            'parameters': params_info,
            'docstring': docstring,
            # Prerequisite tracking metadata
            'prerequisites': prerequisites or {},
            'requires': requires or {},
            'produces': produces or {},
            'auto_fix': auto_fix,
            'virtual_entry': False,
            'source': 'runtime',
        }

        self._store_entry(
            entry,
            register_short_name=True,
            override_aliases=True,
            map_function=func,
        )

        for derived_entry in self._derive_runtime_entries(func, entry):
            self._store_entry(
                derived_entry,
                register_short_name=False,
                override_aliases=False,
            )
        
        return func

    def _derive_runtime_entries(self, func: Callable, base_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Expand a registered callable into searchable virtual registry entries."""

        try:
            source = inspect.getsource(func)
        except Exception:
            return []

        try:
            tree = ast.parse(textwrap.dedent(source))
        except Exception:
            return []

        node = None
        for child in tree.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                node = child
                break
        if node is None:
            return []

        entries: List[Dict[str, Any]] = []
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if child.name.startswith("_"):
                    continue
                method_entry = self._build_runtime_method_entry(base_entry, node, child)
                entries.append(method_entry)
                entries.extend(self._build_runtime_branch_entries(child, method_entry, base_entry, func))
        else:
            entries.extend(self._build_runtime_branch_entries(node, base_entry, base_entry, func))

        return entries

    def _build_runtime_method_entry(
        self,
        base_entry: Dict[str, Any],
        class_node: ast.ClassDef,
        method_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    ) -> Dict[str, Any]:
        aliases = [
            f"{class_node.name}.{method_node.name}",
            f"{base_entry.get('short_name', class_node.name)} {method_node.name}",
        ]
        for alias in (base_entry.get("aliases") or [])[:8]:
            alias = str(alias).strip()
            if alias:
                aliases.append(f"{alias} {method_node.name}")

        return {
            'function': base_entry.get('function'),
            'full_name': f"{base_entry.get('full_name', class_node.name)}.{method_node.name}",
            'short_name': method_node.name,
            'module': base_entry.get('module', ''),
            'aliases': list(dict.fromkeys(aliases)),
            'category': base_entry.get('category', ''),
            'description': ast.get_docstring(method_node) or f"Method `{method_node.name}` on {base_entry.get('full_name', class_node.name)}.",
            'examples': self._filter_examples_for_variant(base_entry.get('examples', []), method_node.name),
            'related': [base_entry.get('full_name', '')],
            'signature': self._derive_signature_from_ast(method_node),
            'parameters': self._parameters_from_ast(method_node),
            'docstring': ast.get_docstring(method_node) or "",
            'prerequisites': base_entry.get('prerequisites', {}) or {},
            'requires': base_entry.get('requires', {}) or {},
            'produces': base_entry.get('produces', {}) or {},
            'auto_fix': base_entry.get('auto_fix', 'none'),
            'virtual_entry': True,
            'source': 'runtime_derived_method',
            'parent_full_name': base_entry.get('full_name', ''),
            'imports': self._collect_import_targets(method_node.body),
        }

    def _build_runtime_branch_entries(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        parent_entry: Dict[str, Any],
        owner_entry: Dict[str, Any],
        func: Callable,
    ) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str]] = set()

        for child in ast.walk(node):
            if not isinstance(child, ast.If):
                continue
            for param_name, branch_value in self._extract_branch_variants(child.test):
                if not self._should_index_branch_variant(
                    param_name,
                    branch_value,
                    child.body,
                ):
                    continue
                key = (param_name, branch_value)
                if key in seen:
                    continue
                seen.add(key)
                imports = self._collect_import_targets(child.body)
                aliases = [
                    branch_value,
                    f"{node.name} {branch_value}",
                    f"{param_name} {branch_value}",
                    f"{parent_entry.get('short_name', node.name)} {branch_value}",
                ]
                for alias in (owner_entry.get('aliases') or [])[:8]:
                    alias = str(alias).strip()
                    if alias:
                        aliases.append(f"{alias} {branch_value}")
                entries.append({
                    'function': func,
                    'full_name': f"{parent_entry.get('full_name', node.name)}[{param_name}={branch_value}]",
                    'short_name': branch_value,
                    'module': parent_entry.get('module', ''),
                    'aliases': list(dict.fromkeys(aliases)),
                    'category': parent_entry.get('category', ''),
                    'description': self._build_branch_description(parent_entry, param_name, branch_value, imports),
                    'examples': self._filter_examples_for_variant(parent_entry.get('examples', []), branch_value, param_name),
                    'related': [parent_entry.get('full_name', '')],
                    'signature': parent_entry.get('signature', self._derive_signature_from_ast(node)),
                    'parameters': list(parent_entry.get('parameters', []) or []),
                    'docstring': parent_entry.get('docstring', ''),
                    'prerequisites': parent_entry.get('prerequisites', {}) or {},
                    'requires': parent_entry.get('requires', {}) or {},
                    'produces': parent_entry.get('produces', {}) or {},
                    'auto_fix': parent_entry.get('auto_fix', 'none'),
                    'virtual_entry': True,
                    'source': 'runtime_derived_branch',
                    'parent_full_name': parent_entry.get('full_name', ''),
                    'branch_parameter': param_name,
                    'branch_value': branch_value,
                    'imports': imports,
                })

        return entries

    @staticmethod
    def _build_branch_description(
        parent_entry: Dict[str, Any],
        param_name: str,
        branch_value: str,
        imports: List[str],
    ) -> str:
        description = (
            f"Variant of `{parent_entry.get('full_name', '')}` when "
            f"`{param_name}='{branch_value}'`."
        )
        if imports:
            description += " Imports/uses: " + ", ".join(imports) + "."
        return description

    @staticmethod
    def _filter_examples_for_variant(examples: List[str], *keywords: str) -> List[str]:
        if not examples:
            return []
        lowered = [str(keyword).lower() for keyword in keywords if str(keyword).strip()]
        matches = [
            example for example in examples
            if any(keyword in str(example).lower() for keyword in lowered)
        ]
        return matches[:3] if matches else list(examples[:2])

    @staticmethod
    def _derive_signature_from_ast(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
        """Derive a docstring-style signature from an AST node.

        Includes positional args (with defaults) **and** keyword-only args
        (with defaults). The latter were missing previously, which hid the
        ``mode=`` / ``doublets_method=`` / ``flavor=`` style dispatch
        parameters from ``registry_lookup`` output. Without those, agents
        searching for e.g. ``"scdblfinder"`` would see ``ov.pp.qc(adata)``
        and never realise the function dispatches via
        ``doublets_method='scdblfinder'``.
        """
        params: list[str] = []

        # Positional args (skip ``self`` if present).
        arg_names = list(node.args.args)
        defaults = list(node.args.defaults)
        offset = len(arg_names) - len(defaults)
        for idx, arg in enumerate(arg_names):
            if idx == 0 and arg.arg == "self":
                continue
            p = arg.arg
            if idx >= offset:
                d = defaults[idx - offset]
                try:
                    p += "=" + repr(ast.literal_eval(d))
                except Exception:
                    pass
            params.append(p)

        # ``*args`` separator if there are kwonly args without a vararg.
        if node.args.kwonlyargs and node.args.vararg is None:
            params.append("*")
        elif node.args.vararg is not None:
            params.append("*" + node.args.vararg.arg)

        # Keyword-only args (with defaults). This is the fix for the
        # dispatch-parameter visibility issue.
        for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
            p = arg.arg
            if default is not None:
                try:
                    p += "=" + repr(ast.literal_eval(default))
                except Exception:
                    pass
            params.append(p)

        if node.args.kwarg is not None:
            params.append("**" + node.args.kwarg.arg)

        return f"({', '.join(params)})"

    @staticmethod
    def _parameters_from_ast(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
        params: List[str] = []
        arg_names = list(node.args.args)
        defaults = list(node.args.defaults)
        offset = len(arg_names) - len(defaults)
        for idx, arg in enumerate(arg_names):
            if idx == 0 and arg.arg == "self":
                continue
            param = arg.arg
            default_node = defaults[idx - offset] if idx >= offset else None
            if default_node is not None:
                try:
                    param += "=" + repr(ast.literal_eval(default_node))
                except Exception:
                    pass
            params.append(param)
        return params

    def _extract_branch_variants(self, test: ast.AST) -> List[Tuple[str, str]]:
        variants: List[Tuple[str, str]] = []
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
            for value in test.values:
                variants.extend(self._extract_branch_variants(value))
            return variants

        if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
            return variants

        subject = self._branch_subject_name(test.left)
        comparator = test.comparators[0]
        op = test.ops[0]
        values: List[str] = []
        if isinstance(op, ast.Eq):
            values = self._branch_string_values(comparator)
        elif isinstance(op, ast.In):
            values = self._branch_string_values(comparator)

        if subject and values:
            variants.extend((subject, value) for value in values)
        return variants

    def _should_index_branch_variant(
        self,
        param_name: str,
        branch_value: str,
        statements: List[ast.stmt],
    ) -> bool:
        """Return True when a string-dispatch branch likely represents a user-facing capability."""

        name = (param_name or "").strip().lower()
        value = (branch_value or "").strip().lower()
        if not name or not value:
            return False
        if name in _NON_CAPABILITY_BRANCH_PARAMS:
            return False
        if name in _CAPABILITY_BRANCH_PARAMS:
            return True
        if any(name.endswith(suffix) for suffix in _CAPABILITY_BRANCH_SUFFIXES):
            return True

        imports = self._collect_import_targets(statements)
        if imports and any(name.endswith(suffix) for suffix in _CAPABILITY_BRANCH_IMPORT_SUFFIXES):
            return True
        if imports:
            return True

        return False

    @staticmethod
    def _branch_subject_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    @staticmethod
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

    @staticmethod
    def _collect_import_targets(statements: List[ast.stmt]) -> List[str]:
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
    
    def find(self, query: str, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Find functions matching the query.
        
        Parameters
        ----------
        query : str
            Search query (can be Chinese, English, abbreviation, etc.)
        threshold : float
            Similarity threshold for fuzzy matching (0-1)
        
        Returns
        -------
        List[Dict[str, Any]]
            List of matching function entries, sorted by relevance
        """
        query_lower = query.lower()
        results = []
        
        # Exact match
        if query_lower in self._registry:
            results.append(self._registry[query_lower])
            
        # Fuzzy match on aliases
        all_keys = list(self._registry.keys())
        close_matches = get_close_matches(query_lower, all_keys, n=5, cutoff=threshold)
        
        for match in close_matches:
            if self._registry[match] not in results:
                results.append(self._registry[match])
        
        # Search in descriptions, examples, docstrings, imports, and aliases
        for entry in self._registry.values():
            if entry not in results:
                haystack = " ".join([
                    str(entry.get('description', '') or ''),
                    str(entry.get('docstring', '') or ''),
                    " ".join(entry.get('aliases', []) or []),
                    " ".join(entry.get('examples', []) or []),
                    " ".join(entry.get('related', []) or []),
                    " ".join(entry.get('imports', []) or []),
                    str(entry.get('full_name', '') or ''),
                    str(entry.get('short_name', '') or ''),
                ]).lower()
                if query_lower in haystack:
                    results.append(entry)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for entry in results:
            if entry['full_name'] not in seen:
                seen.add(entry['full_name'])
                unique_results.append(entry)
        
        return unique_results
    
    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all functions in a specific category.
        
        Parameters
        ----------
        category : str
            Category name
        
        Returns
        -------
        List[Dict[str, Any]]
            List of functions in the category
        """
        if category not in self._categories:
            return []
        
        results = []
        for func_name in self._categories[category]:
            # Find the entry
            for entry in self._registry.values():
                if entry['full_name'] == func_name:
                    results.append(entry)
                    break
        
        return results
    
    def list_categories(self) -> List[str]:
        """List all available categories."""
        return list(self._categories.keys())
    
    def get_function(self, query: str) -> Optional[Callable]:
        """
        Get the actual function object for a query.
        
        Parameters
        ----------
        query : str
            Function name or alias
        
        Returns
        -------
        Optional[Callable]
            The function object if found, None otherwise
        """
        matches = self.find(query)
        if matches:
            return matches[0]['function']
        return None
    
    def format_results(self, results: List[Dict[str, Any]], verbose: bool = False) -> str:
        """
        Format search results for display.
        
        Parameters
        ----------
        results : List[Dict[str, Any]]
            Search results
        verbose : bool
            Whether to include detailed information
        
        Returns
        -------
        str
            Formatted string for display
        """
        if not results:
            return "❌ No matching functions found."
        
        output = []
        output.append(f"🔍 Found {len(results)} matching function(s):\n")
        
        for i, entry in enumerate(results, 1):
            output.append(f"\n{i}. 📦 {entry['full_name']}")
            output.append(f"   📝 {entry['description']}")
            output.append(f"   🏷️  Aliases: {', '.join(entry['aliases'])}")
            output.append(f"   📁 Category: {entry['category']}")
            
            if verbose:
                output.append(f"   🔧 Signature: {entry['signature']}")
                if entry['examples']:
                    output.append(f"   💡 Examples:")
                    for example in entry['examples']:
                        output.append(f"      {example}")
                if entry['related']:
                    output.append(f"   🔗 Related: {', '.join(entry['related'])}")

        return "\n".join(output)

    def get_prerequisites(self, func_name: str) -> Dict[str, Any]:
        """
        Get full prerequisite information for a function.

        Parameters
        ----------
        func_name : str
            Function name or alias

        Returns
        -------
        Dict with keys:
            - required_functions: List[str] - Must run these first
            - optional_functions: List[str] - Recommended to run first
            - requires: Dict - Required data structures
            - produces: Dict - What the function creates
            - auto_fix: str - Auto-fix strategy

        Examples
        --------
        >>> registry.get_prerequisites('pca')
        {
            'required_functions': ['scale'],
            'optional_functions': ['qc', 'preprocess'],
            'requires': {'layers': ['scaled']},
            'produces': {'obsm': ['X_pca'], 'varm': ['PCs'], 'uns': ['pca']},
            'auto_fix': 'escalate'
        }
        """
        matches = self.find(func_name)
        if not matches:
            return {
                'required_functions': [],
                'optional_functions': [],
                'requires': {},
                'produces': {},
                'auto_fix': 'none'
            }

        entry = matches[0]
        prereqs = entry.get('prerequisites', {})

        return {
            'required_functions': prereqs.get('functions', []),
            'optional_functions': prereqs.get('optional_functions', []),
            'requires': entry.get('requires', {}),
            'produces': entry.get('produces', {}),
            'auto_fix': entry.get('auto_fix', 'none')
        }

    def get_prerequisite_chain(self, func_name: str, include_optional: bool = False) -> List[str]:
        """
        Get ordered list of functions to run for prerequisites.

        Parameters
        ----------
        func_name : str
            Target function name
        include_optional : bool
            Whether to include optional prerequisites. Default: False

        Returns
        -------
        List[str]
            Ordered prerequisite chain, e.g., ['scale', 'pca']

        Examples
        --------
        >>> registry.get_prerequisite_chain('pca')
        ['scale', 'pca']

        >>> registry.get_prerequisite_chain('leiden')
        ['neighbors', 'leiden']

        >>> registry.get_prerequisite_chain('pca', include_optional=True)
        ['qc', 'preprocess', 'scale', 'pca']
        """
        prereq_info = self.get_prerequisites(func_name)
        chain = []

        # Add optional functions if requested
        if include_optional:
            chain.extend(prereq_info['optional_functions'])

        # Add required functions
        chain.extend(prereq_info['required_functions'])

        # Add the target function itself
        matches = self.find(func_name)
        if matches:
            chain.append(matches[0]['short_name'])

        return chain

    def check_prerequisites(self, func_name: str, adata) -> Dict[str, Any]:
        """
        Validate if all prerequisites are satisfied for an AnnData object.

        Parameters
        ----------
        func_name : str
            Function to check
        adata : AnnData
            Data object to validate

        Returns
        -------
        Dict with keys:
            - satisfied: bool - All requirements met
            - missing_functions: List[str] - Functions likely not run yet
            - missing_structures: List[str] - Missing data layers/structures
            - recommendation: str - What to do next
            - auto_fixable: bool - Can prerequisites be auto-inserted

        Examples
        --------
        >>> result = registry.check_prerequisites('pca', raw_adata)
        >>> result
        {
            'satisfied': False,
            'missing_functions': ['scale'],
            'missing_structures': ['adata.layers["scaled"]'],
            'recommendation': 'Run ov.pp.scale() first or use ov.pp.preprocess()',
            'auto_fixable': False
        }
        """
        prereq_info = self.get_prerequisites(func_name)
        requires = prereq_info['requires']

        missing_structures = []

        # Check required layers
        if 'layers' in requires:
            for layer in requires['layers']:
                if not hasattr(adata, 'layers') or layer not in adata.layers:
                    missing_structures.append(f'adata.layers["{layer}"]')

        # Check required obsm
        if 'obsm' in requires:
            for key in requires['obsm']:
                if not hasattr(adata, 'obsm') or key not in adata.obsm:
                    missing_structures.append(f'adata.obsm["{key}"]')

        # Check required obsp
        if 'obsp' in requires:
            for key in requires['obsp']:
                if not hasattr(adata, 'obsp') or key not in adata.obsp:
                    missing_structures.append(f'adata.obsp["{key}"]')

        # Check required uns
        if 'uns' in requires:
            for key in requires['uns']:
                if not hasattr(adata, 'uns') or key not in adata.uns:
                    missing_structures.append(f'adata.uns["{key}"]')

        # Check required var columns
        if 'var' in requires:
            for col in requires['var']:
                if not hasattr(adata, 'var') or col not in adata.var.columns:
                    missing_structures.append(f'adata.var["{col}"]')

        # Check required obs columns
        if 'obs' in requires:
            for col in requires['obs']:
                if not hasattr(adata, 'obs') or col not in adata.obs.columns:
                    missing_structures.append(f'adata.obs["{col}"]')

        # Check required varm
        if 'varm' in requires:
            for key in requires['varm']:
                if not hasattr(adata, 'varm') or key not in adata.varm:
                    missing_structures.append(f'adata.varm["{key}"]')

        satisfied = len(missing_structures) == 0
        missing_functions = prereq_info['required_functions'] if not satisfied else []

        # Generate recommendation
        if satisfied:
            recommendation = "All prerequisites satisfied"
        else:
            auto_fix = prereq_info['auto_fix']
            if auto_fix == 'auto' and len(missing_functions) <= 2:
                func_list = ', '.join([f'ov.pp.{f}()' for f in missing_functions])
                recommendation = f"Auto-fixable: Will insert {func_list}"
            elif auto_fix == 'escalate' or len(missing_functions) > 2:
                recommendation = f"Complex prerequisite chain. Consider using a workflow function."
            else:
                func_list = ', '.join([f'ov.pp.{f}()' for f in missing_functions])
                recommendation = f"Run {func_list} first"

        # Auto-fixable if: auto_fix='auto' AND <= 2 missing functions
        auto_fixable = (prereq_info['auto_fix'] == 'auto' and
                       len(missing_functions) > 0 and
                       len(missing_functions) <= 2)

        return {
            'satisfied': satisfied,
            'missing_functions': missing_functions,
            'missing_structures': missing_structures,
            'recommendation': recommendation,
            'auto_fixable': auto_fixable
        }

    def format_prerequisites_for_llm(self, func_name: str) -> str:
        """
        Format prerequisite info for LLM consumption in system prompt.

        Parameters
        ----------
        func_name : str
            Function to format prerequisites for

        Returns
        -------
        str
            Formatted text with prerequisite chain, requirements, and guidance

        Examples
        --------
        >>> print(registry.format_prerequisites_for_llm('pca'))
        Function: ov.pp.pca()
        Prerequisites:
          - Required functions: scale
          - Optional functions: qc, preprocess
          - Requires: adata.layers['scaled']
          - Produces: adata.obsm['X_pca'], adata.varm['PCs'], adata.uns['pca']
        Prerequisite Chain: scale → pca
        Full Chain (with optional): qc → preprocess → scale → pca
        Auto-fix Strategy: ESCALATE (suggest workflow for complex cases)
        """
        matches = self.find(func_name)
        if not matches:
            return f"Function '{func_name}' not found in registry."

        entry = matches[0]
        prereq_info = self.get_prerequisites(func_name)
        chain = self.get_prerequisite_chain(func_name, include_optional=False)
        full_chain = self.get_prerequisite_chain(func_name, include_optional=True)

        output = []
        output.append(f"Function: {entry['full_name']}()")
        output.append("Prerequisites:")

        # Required functions
        if prereq_info['required_functions']:
            output.append(f"  - Required functions: {', '.join(prereq_info['required_functions'])}")
        else:
            output.append("  - Required functions: None")

        # Optional functions
        if prereq_info['optional_functions']:
            output.append(f"  - Optional functions: {', '.join(prereq_info['optional_functions'])}")

        # Required structures
        requires = prereq_info['requires']
        req_list = []
        for key, values in requires.items():
            for val in values:
                req_list.append(f"adata.{key}['{val}']")
        if req_list:
            output.append(f"  - Requires: {', '.join(req_list)}")

        # Produced structures
        produces = prereq_info['produces']
        prod_list = []
        for key, values in produces.items():
            for val in values:
                prod_list.append(f"adata.{key}['{val}']")
        if prod_list:
            output.append(f"  - Produces: {', '.join(prod_list)}")

        # Prerequisite chain
        if len(chain) > 1:
            output.append(f"Prerequisite Chain: {' → '.join(chain)}")
        if len(full_chain) > len(chain):
            output.append(f"Full Chain (with optional): {' → '.join(full_chain)}")

        # Auto-fix strategy
        auto_fix = prereq_info['auto_fix']
        auto_fix_desc = {
            'auto': 'AUTO (can auto-insert simple prerequisites)',
            'escalate': 'ESCALATE (suggest workflow for complex cases)',
            'none': 'NONE (just warn user)'
        }
        output.append(f"Auto-fix Strategy: {auto_fix_desc.get(auto_fix, auto_fix)}")

        return "\n".join(output)


# Global registry instance
_global_registry = FunctionRegistry()


def get_registry() -> FunctionRegistry:
    """Return the global function registry singleton."""
    return _global_registry


def _hydrate_registry_for_export() -> None:
    """Best-effort import of major public modules before registry export.

    OmicVerse relies on module import side effects from ``@register_function``.
    Without a hydration pass, ``export_registry()`` may only include whichever
    modules the caller happened to import first.
    """
    import importlib

    module_names = (
        "omicverse._settings",
        "omicverse.alignment",
        "omicverse.utils.biocontext",
        "omicverse.bulk",
        "omicverse.bulk2single",
        "omicverse.datasets",
        "omicverse.io",
        "omicverse.pl",
        "omicverse.pp",
        "omicverse.single",
        "omicverse.space",
        "omicverse.metabol",
        "omicverse.mol",
        "omicverse.protein",
        "omicverse.genetics",
        "omicverse.airr",
    )
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except Exception:
            # Optional dependencies may block some modules; export whatever
            # could be safely registered in the current environment.
            continue
    # metabol is itself lazy — its __init__ doesn't import the submodules
    # that carry the @register_function decorators. Drive its hydration hook
    # explicitly so export_registry sees metabol functions.
    try:
        import omicverse.metabol as _metabol  # noqa: F401

        _metabol._hydrate_registry()
    except Exception:
        pass
    # Same drill for ov.protein — its __init__ is lazy too.
    try:
        import omicverse.protein as _protein  # noqa: F401

        _protein._hydrate_registry()
    except Exception:
        pass
    # And ov.genetics — its __init__ is lazy too.
    try:
        import omicverse.genetics as _genetics  # noqa: F401

        _genetics._hydrate_registry()
    except Exception:
        pass
    # And ov.airr — its __init__ is lazy too.
    try:
        import omicverse.airr as _airr  # noqa: F401

        _airr._hydrate_registry()
    except Exception:
        pass
    # And ov.mol — its __init__ is lazy too.
    try:
        import omicverse.mol as _mol  # noqa: F401

        _mol._hydrate_registry()
    except Exception:
        pass


def register_function(aliases: List[str],
                     category: str,
                     description: str,
                     examples: Optional[List[str]] = None,
                     related: Optional[List[str]] = None,
                     prerequisites: Optional[Dict[str, List[str]]] = None,
                     requires: Optional[Dict[str, List[str]]] = None,
                     produces: Optional[Dict[str, List[str]]] = None,
                     auto_fix: str = 'none'):
    """
    Decorator to register a function with metadata.

    Parameters
    ----------
    aliases : List[str]
        List of aliases for the function (Chinese, English, abbreviations)
    category : str
        Function category
    description : str
        Function description
    examples : List[str], optional
        Usage examples
    related : List[str], optional
        Related function names
    prerequisites : Dict[str, List[str]], optional
        Function prerequisites, format:
        {'functions': ['scale'], 'optional_functions': ['qc', 'preprocess']}
    requires : Dict[str, List[str]], optional
        Required data structures, format:
        {'layers': ['scaled'], 'obsm': ['X_pca'], 'uns': ['neighbors'], ...}
    produces : Dict[str, List[str]], optional
        Data structures created by this function, format:
        {'layers': ['scaled'], 'obsm': ['X_pca'], 'uns': ['pca'], ...}
    auto_fix : str, optional
        Auto-fix strategy: 'auto' (can auto-insert prerequisites),
        'escalate' (suggest workflow), 'none' (just warn). Default: 'none'

    Examples
    --------
    >>> @register_function(
    ...     aliases=["质控", "qc", "quality_control"],
    ...     category="preprocessing",
    ...     description="Perform quality control on single-cell data"
    ... )
    ... def qc(adata, min_genes=200):
    ...     pass

    >>> @register_function(
    ...     aliases=["pca", "PCA"],
    ...     category="preprocessing",
    ...     description="Perform Principal Component Analysis",
    ...     prerequisites={'functions': ['scale']},
    ...     requires={'layers': ['scaled']},
    ...     produces={'obsm': ['X_pca'], 'varm': ['PCs'], 'uns': ['pca']},
    ...     auto_fix='escalate'
    ... )
    ... def pca(adata, n_pcs=50):
    ...     pass
    """
    def decorator(func: Callable) -> Callable:
        _global_registry.register(
            func=func,
            aliases=aliases,
            category=category,
            description=description,
            examples=examples,
            related=related,
            prerequisites=prerequisites,
            requires=requires,
            produces=produces,
            auto_fix=auto_fix
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Add registry info to function
        wrapper._registry_info = {
            'aliases': aliases,
            'category': category,
            'description': description
        }

        # If func is a class, copy over class methods and static methods
        import inspect
        if inspect.isclass(func):
            for name, value in inspect.getmembers(func):
                if isinstance(inspect.getattr_static(func, name), (classmethod, staticmethod)):
                    setattr(wrapper, name, value)

        return wrapper
    
    return decorator


# User-friendly API functions
def find_function(query: str, verbose: bool = False) -> Union[str, None]:
    """
    Find functions matching the query.
    
    Parameters
    ----------
    query : str
        Search query (Chinese, English, or abbreviation)
    verbose : bool
        Whether to show detailed information
    
    Returns
    -------
    str
        Formatted search results
    
    Examples
    --------
    >>> import omicverse as ov
    >>> ov.find_function("质控")
    >>> ov.find_function("differential expression")
    >>> ov.find_function("pca")
    """
    results = _global_registry.find(query)
    print(_global_registry.format_results(results, verbose=verbose))
    
    if results:
        return results[0]['function']
    return None


def list_functions(category: Optional[str] = None) -> None:
    """
    List all registered functions or functions in a category.
    
    Parameters
    ----------
    category : str, optional
        Category to filter by
    
    Examples
    --------
    >>> import omicverse as ov
    >>> ov.list_functions()  # List all functions
    >>> ov.list_functions("preprocessing")  # List preprocessing functions
    """
    if category:
        results = _global_registry.get_by_category(category)
        print(f"📚 Functions in category '{category}':")
    else:
        # Get all unique functions
        all_functions = {}
        for entry in _global_registry._registry.values():
            all_functions[entry['full_name']] = entry
        results = list(all_functions.values())
        print(f"📚 All registered functions ({len(results)} total):")
    
    if not results:
        print("   No functions found.")
        return
    
    # Group by category if showing all
    if not category:
        categories = {}
        for entry in results:
            cat = entry['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(entry)
        
        for cat, entries in sorted(categories.items()):
            print(f"\n🏷️  {cat.upper()}:")
            for entry in sorted(entries, key=lambda x: x['short_name']):
                print(f"   • {entry['short_name']}: {entry['description']}")
    else:
        for entry in sorted(results, key=lambda x: x['short_name']):
            print(f"   • {entry['short_name']}: {entry['description']}")


def get_function_help(query: str) -> None:
    """
    Get detailed help for a function.
    
    Parameters
    ----------
    query : str
        Function name or alias
    
    Examples
    --------
    >>> import omicverse as ov
    >>> ov.help("qc")
    >>> ov.help("质控")
    """
    results = _global_registry.find(query)
    
    if not results:
        print(f"❌ No function found for '{query}'")
        return
    
    entry = results[0]
    print(f"📦 {entry['full_name']}")
    print(f"{'=' * 50}")
    print(f"📝 Description: {entry['description']}")
    print(f"🏷️  Aliases: {', '.join(entry['aliases'])}")
    print(f"📁 Category: {entry['category']}")
    print(f"🔧 Signature: {entry['signature']}")
    
    if entry['docstring']:
        print(f"\n📖 Documentation:")
        print(entry['docstring'])
    
    if entry['examples']:
        print(f"\n💡 Examples:")
        for example in entry['examples']:
            print(f"   {example}")
    
    if entry['related']:
        print(f"\n🔗 Related functions: {', '.join(entry['related'])}")


def recommend_function(task_description: str, n: int = 5) -> List[Callable]:
    """
    Recommend functions based on task description.
    
    This is a simple keyword-based recommendation. 
    Could be enhanced with embedding-based search in the future.
    
    Parameters
    ----------
    task_description : str
        Description of what you want to do
    n : int
        Number of recommendations
    
    Returns
    -------
    List[Callable]
        List of recommended functions
    
    Examples
    --------
    >>> import omicverse as ov
    >>> ov.recommend_function("我想过滤低质量细胞")
    >>> ov.recommend_function("perform dimensionality reduction")
    """
    # Extract keywords from description
    keywords = task_description.lower().split()
    
    # Score each function based on keyword matches
    scores = {}
    for entry in _global_registry._registry.values():
        if entry['full_name'] in scores:
            continue
            
        score = 0
        # Check aliases
        for keyword in keywords:
            for alias in entry['aliases']:
                if keyword in alias.lower():
                    score += 2
            # Check description
            if keyword in entry['description'].lower():
                score += 1
        
        if score > 0:
            scores[entry['full_name']] = (score, entry)
    
    # Sort by score and return top n
    sorted_entries = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)[:n]
    
    if not sorted_entries:
        print(f"❌ No recommendations found for: '{task_description}'")
        return []
    
    print(f"🎯 Top {min(n, len(sorted_entries))} recommendations for: '{task_description}'")
    results = [entry[1][1] for entry in sorted_entries]
    print(_global_registry.format_results(results, verbose=False))
    
    return [entry[1][1]['function'] for entry in sorted_entries]


def export_registry(filepath: Optional[str] = None, 
                   format: str = "json",
                   include_source: bool = False) -> Union[str, Dict]:
    """
    Export the current registry to JSON format.
    
    Parameters
    ----------
    filepath : str, optional
        File path to save the registry. If None, returns the data as dict/string
    format : str
        Export format: 'json' or 'dict'
    include_source : bool
        Whether to include source code of functions (warning: large file)
    
    Returns
    -------
    Union[str, Dict]
        Registry data as JSON string or dictionary
        
    Examples
    --------
    >>> # Export to file
    >>> ov.export_registry('registry.json')
    
    >>> # Get as dictionary
    >>> data = ov.export_registry(format='dict')
    
    >>> # Get as JSON string
    >>> json_str = ov.export_registry(format='json')
    """
    _hydrate_registry_for_export()

    export_data = {}
    
    # Get all unique functions
    processed_functions = set()
    
    for entry in _global_registry._registry.values():
        full_name = entry['full_name']
        if full_name in processed_functions:
            continue
        processed_functions.add(full_name)
        
        func_data = {
            'full_name': entry['full_name'],
            'short_name': entry['short_name'],
            'module': entry['module'],
            'aliases': entry['aliases'],
            'category': entry['category'],
            'description': entry['description'],
            'examples': entry['examples'],
            'related': entry['related'],
            'signature': entry['signature'],
            'docstring': entry['docstring'],
            'source': entry.get('source', 'runtime'),
            'virtual_entry': bool(entry.get('virtual_entry', False)),
        }

        for optional_key in (
            'parent_full_name',
            'branch_parameter',
            'branch_value',
            'imports',
            'parameters',
        ):
            if optional_key in entry:
                func_data[optional_key] = entry.get(optional_key)
        
        if include_source:
            try:
                func_data['source_code'] = inspect.getsource(entry['function'])
            except Exception:
                func_data['source_code'] = "Source not available"
        
        export_data[full_name] = func_data
    
    # Add metadata
    export_metadata = {
        'exported_at': datetime.now(timezone.utc).isoformat(),
        'total_functions': len(export_data),
        'categories': list(_global_registry._categories.keys()),
        'omicverse_version': getattr(__import__('omicverse'), '__version__', 'unknown')
    }
    
    final_data = {
        'metadata': export_metadata,
        'functions': export_data
    }
    
    if format == 'dict':
        result = final_data
    else:  # json
        result = json.dumps(final_data, indent=2, ensure_ascii=False)
    
    if filepath:
        filepath = Path(filepath)
        with open(filepath, 'w', encoding='utf-8') as f:
            if format == 'dict':
                json.dump(final_data, f, indent=2, ensure_ascii=False)
            else:
                f.write(result)
        print(f"Registry exported to: {filepath}")
        print(f"Total functions: {len(export_data)}")
        return str(filepath)
    
    return result


def import_registry(filepath: str) -> Dict:
    """
    Import registry data from JSON file.
    
    Parameters
    ----------
    filepath : str
        Path to the JSON file
        
    Returns
    -------
    Dict
        Imported registry data
        
    Examples
    --------
    >>> data = ov.import_registry('registry.json')
    """
    filepath = Path(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Registry imported from: {filepath}")
    print(f"Total functions: {data['metadata']['total_functions']}")
    print(f"Categories: {', '.join(data['metadata']['categories'])}")
    
    return data


# Export registry instance and API functions
__all__ = [
    'register_function',
    'find_function', 
    'get_registry',
    'list_functions',
    'get_function_help',
    'recommend_function',
    'export_registry',
    'import_registry',
    'FunctionRegistry',
    '_global_registry'
]
