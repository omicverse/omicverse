"""Sandbox execution helpers for agent-generated code.

Extracted from ``analysis_executor.py`` during Phase 4 decomposition.
Contains sandbox globals construction, the main execution entry point,
read-only snippet execution, approval gating, doublet harmonization,
figure autosave injection, and context-directive processing.

All functions receive an explicit ``ctx`` (AgentContext) parameter where
state access is needed, rather than relying on ``self``.
"""

from __future__ import annotations

import builtins
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..agent_errors import SandboxDeniedError, SecurityViolationError
from ..agent_sandbox import ApprovalMode, SafeOsProxy
from ..agent_config import SandboxFallbackPolicy
from ..agent_reporter import EventLevel

if TYPE_CHECKING:
    from .protocol import AgentContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Figure autosave
# ---------------------------------------------------------------------------

def figure_autosave_dir(ctx: "AgentContext") -> Optional[Path]:
    """Return the workspace figures directory, or None if unavailable."""
    fs_ctx = getattr(ctx, "_filesystem_context", None)
    if fs_ctx is None:
        return None
    workspace_dir = getattr(fs_ctx, "workspace_dir", None)
    if workspace_dir is None:
        return None
    return Path(workspace_dir) / "figures"


def inject_figure_autosave(ctx: "AgentContext", code: str) -> str:
    """Wrap *code* with figure-autosave prelude and epilogue."""
    fig_dir = figure_autosave_dir(ctx)
    if fig_dir is None:
        return code

    safe_dir = json.dumps(str(fig_dir))
    prelude = (
        "from pathlib import Path as _ov_fig_path\n"
        "import matplotlib.pyplot as _ov_fig_plt\n"
        f"_ov_fig_dir = _ov_fig_path({safe_dir})\n"
        "_ov_fig_dir.mkdir(parents=True, exist_ok=True)\n"
        "for _ov_fig_num in list(_ov_fig_plt.get_fignums()):\n"
        "    try:\n"
        "        _ov_fig_plt.close(_ov_fig_num)\n"
        "    except Exception:\n"
        "        pass\n"
    )
    epilogue = (
        "\nimport time as _ov_fig_time\n"
        "import matplotlib.pyplot as _ov_fig_plt\n"
        "_ov_fig_nums = list(_ov_fig_plt.get_fignums())\n"
        "for _ov_fig_idx, _ov_fig_num in enumerate(_ov_fig_nums, start=1):\n"
        "    try:\n"
        "        _ov_fig = _ov_fig_plt.figure(_ov_fig_num)\n"
        "        _ov_fig.savefig(\n"
        "            _ov_fig_dir / f'auto_figure_{int(_ov_fig_time.time() * 1000)}_{_ov_fig_idx:02d}.png',\n"
        "            dpi=200,\n"
        "            bbox_inches='tight',\n"
        "        )\n"
        "    except Exception as _ov_fig_exc:\n"
        "        print(f'[ovagent autosave warning] {_ov_fig_exc}')\n"
    )
    return prelude + "\n" + code + "\n" + epilogue


# ---------------------------------------------------------------------------
# Approval gate
# ---------------------------------------------------------------------------

def request_approval(ctx: "AgentContext", code: str, violations: list) -> bool:
    """Prompt the user (or call the approval handler) for execution consent."""
    from ..harness import make_turn_id

    if ctx._approval_handler is not None:
        trace = ctx._last_run_trace
        payload = {
            "request_id": make_turn_id(),
            "title": "Execution approval required",
            "message": "Generated code requires approval before execution.",
            "code": code,
            "violations": [v.__dict__ if hasattr(v, "__dict__") else str(v) for v in violations],
            "trace_id": getattr(trace, "trace_id", ""),
            "session_id": ctx._get_harness_session_id(),
            "approval_mode": ctx._security_config.approval_mode.value,
        }
        try:
            return bool(ctx._approval_handler(payload))
        except Exception as exc:
            logger.warning("Approval handler failed, denying execution: %s", exc)
            return False

    print("\n" + "=" * 60)
    print("GENERATED CODE REVIEW")
    print("=" * 60)
    display = code if len(code) < 2000 else code[:2000] + "\n... (truncated)"
    for i, line in enumerate(display.split("\n"), 1):
        print(f"  {i:3d} | {line}")
    if violations:
        print()
        print(ctx._security_scanner.format_report(violations))
    print("=" * 60)
    try:
        response = input("Execute this code? [y/N]: ").strip().lower()
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


# ---------------------------------------------------------------------------
# Read-only snippet execution
# ---------------------------------------------------------------------------

def execute_snippet_readonly(ctx: "AgentContext", code: str, adata: Any) -> str:
    """Run *code* for read-only inspection — always in-process.

    Returns captured stdout (or error message).
    """
    try:
        violations = ctx._security_scanner.scan(code)
    except SyntaxError:
        violations = []
    if violations:
        report = ctx._security_scanner.format_report(violations)
        logger.warning("Security scan report:\n%s", report)
        if ctx._security_scanner.has_critical(violations):
            return f"ERROR: Code blocked by security scanner:\n{report}"

    import io
    from contextlib import redirect_stdout

    compiled = compile(code, "<omicverse-snippet>", "exec")
    sandbox_globals = build_sandbox_globals(ctx)
    sandbox_locals: Dict[str, Any] = {"adata": adata}
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(compiled, sandbox_globals, sandbox_locals)  # noqa: S102
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        _undo_scvi_shims()
    out = buf.getvalue()
    return out if out.strip() else "(no stdout output)"


# ---------------------------------------------------------------------------
# Main execution entry point
# ---------------------------------------------------------------------------

def execute_generated_code(
    ctx: "AgentContext", code: str, adata: Any, capture_stdout: bool = False,
    *, _executor: Any = None,
) -> Any:
    """Execute agent-generated code in the sandbox."""
    code = inject_figure_autosave(ctx, code)

    # --- Pre-execution security scan ---
    try:
        violations = ctx._security_scanner.scan(code)
    except SyntaxError:
        violations = []

    if violations:
        report = ctx._security_scanner.format_report(violations)
        logger.warning("Security scan report:\n%s", report)
        if ctx._security_scanner.has_critical(violations):
            raise SecurityViolationError(
                f"Code blocked by security scanner:\n{report}",
                violations=violations,
            )

    # --- Approval gate ---
    approval_mode = ctx._security_config.approval_mode
    if approval_mode == ApprovalMode.ALWAYS:
        if not request_approval(ctx, code, violations):
            raise SecurityViolationError("User declined code execution.")
    elif approval_mode == ApprovalMode.ON_VIOLATION and violations:
        if not request_approval(ctx, code, violations):
            raise SecurityViolationError(
                "User declined code execution after security warnings."
            )

    # Use notebook execution if enabled
    if ctx.use_notebook_execution and ctx._notebook_executor is not None:
        try:
            result_adata = ctx._notebook_executor.execute(code, adata)
            if hasattr(result_adata, "uns"):
                result_adata.uns["_ovagent_session"] = {
                    "session_id": ctx._notebook_executor.current_session["session_id"],
                    "notebook_path": str(ctx._notebook_executor.current_session["notebook_path"]),
                    "prompt_number": ctx._notebook_executor.session_prompt_count,
                }
            if ctx.enable_filesystem_context and ctx._filesystem_context:
                process_context_directives(ctx, code, {})
            return result_adata

        except Exception as e:
            policy = getattr(getattr(ctx, "_config", None), "execution", None)
            fb = getattr(policy, "sandbox_fallback_policy", SandboxFallbackPolicy.WARN_AND_FALLBACK)
            if fb == SandboxFallbackPolicy.RAISE:
                raise SandboxDeniedError(
                    f"Notebook execution failed and fallback is disabled: {e}"
                ) from e
            elif fb == SandboxFallbackPolicy.WARN_AND_FALLBACK:
                if _executor is not None:
                    _executor._notebook_fallback_error = str(e)
                if hasattr(ctx, "_emit"):
                    ctx._emit(EventLevel.WARNING, f"Session execution failed: {e}", "execution")
                    ctx._emit(EventLevel.INFO, "Falling back to in-process execution...", "execution")
                else:
                    print(f"\u26a0\ufe0f  Session execution failed: {e}")
                    print("   Falling back to in-process execution...")

    # Legacy in-process execution
    compiled = compile(code, "<omicverse-agent>", "exec")
    sandbox_globals = build_sandbox_globals(ctx)
    sandbox_locals: Dict[str, Any] = {"adata": adata}
    try:
        if hasattr(adata, "obs_names"):
            sandbox_locals.setdefault("obs_names", adata.obs_names)
        if hasattr(adata, "var_names"):
            sandbox_locals.setdefault("var_names", adata.var_names)
    except Exception:
        pass

    _is_mudata = type(adata).__name__ == "MuData"

    # Normalize HVG column naming
    try:
        if not _is_mudata and hasattr(adata, "var") and adata.var is not None:
            if "highly_variable" not in adata.var.columns and "highly_variable_features" in adata.var.columns:
                adata.var["highly_variable"] = adata.var["highly_variable_features"]
        if hasattr(adata, "uns"):
            adata.uns.setdefault("initial_cells", getattr(adata, "n_obs", None) or getattr(adata, "shape", [None])[0])
            adata.uns.setdefault("initial_genes", getattr(adata, "n_vars", None) or getattr(adata, "shape", [None, None])[1])
            adata.uns.setdefault("omicverse_qc_original_cells", getattr(adata, "n_obs", None))
        if not _is_mudata and getattr(adata, "raw", None) is None:
            try:
                adata.raw = adata
            except Exception:
                pass
        if not _is_mudata and hasattr(adata, "layers") and getattr(adata, "layers", None) is not None and "scaled" not in adata.layers:
            try:
                adata.layers["scaled"] = adata.X.copy()
            except Exception:
                pass
        try:
            import pandas as _pd
            if not _is_mudata and hasattr(adata, "obs"):
                for col in adata.obs.columns:
                    col_data = adata.obs[col]
                    if _pd.api.types.is_numeric_dtype(col_data):
                        if col_data.isna().any():
                            adata.obs[col] = col_data.fillna(0)
                if "total_counts" not in adata.obs.columns:
                    try:
                        import numpy as _np
                        data_matrix = None
                        if hasattr(adata, "layers") and getattr(adata, "layers", None) is not None and "counts" in adata.layers:
                            data_matrix = adata.layers["counts"]
                        else:
                            data_matrix = adata.X
                        sums = _np.asarray(data_matrix.sum(axis=1)).ravel()
                        adata.obs["total_counts"] = sums
                    except Exception:
                        adata.obs["total_counts"] = 0
                if "n_counts" not in adata.obs.columns:
                    adata.obs["n_counts"] = adata.obs.get("total_counts", 0)
                if "n_genes_by_counts" not in adata.obs.columns:
                    try:
                        import numpy as _np
                        data_matrix = None
                        if hasattr(adata, "layers") and getattr(adata, "layers", None) is not None and "counts" in adata.layers:
                            data_matrix = adata.layers["counts"]
                        else:
                            data_matrix = adata.X
                        adata.obs["n_genes_by_counts"] = _np.asarray((data_matrix > 0).sum(axis=1)).ravel()
                    except Exception:
                        adata.obs["n_genes_by_counts"] = 0
                if "pct_counts_mito" not in adata.obs.columns and "pct_counts_mt" in adata.obs.columns:
                    adata.obs["pct_counts_mito"] = adata.obs["pct_counts_mt"]
                elif "pct_counts_mito" not in adata.obs.columns:
                    adata.obs["pct_counts_mito"] = 0
            if hasattr(adata, "var") and adata.var is not None:
                if "mito" not in adata.var.columns and "mt" in adata.var.columns:
                    adata.var["mito"] = adata.var["mt"]
                elif "mito" not in adata.var.columns:
                    adata.var["mito"] = False
            try:
                if not getattr(_pd.cut, "_ov_wrapped", False):
                    _orig_cut = _pd.cut

                    def _safe_cut(*args, **kwargs):
                        kwargs.setdefault("duplicates", "drop")
                        return _orig_cut(*args, **kwargs)

                    _safe_cut._ov_wrapped = True  # type: ignore[attr-defined]
                    _pd.cut = _safe_cut
            except Exception:
                pass
        except Exception:
            pass
        try:
            Path("genesets").mkdir(exist_ok=True)
        except Exception:
            pass
    except Exception as exc:
        warnings.warn(
            f"Failed to normalize HVG columns for agent execution: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    warnings.warn(
        "Executing agent-generated code. Ensure the input model and prompts come from a trusted source.",
        RuntimeWarning,
        stacklevel=2,
    )

    import io as _io

    stdout_buffer = _io.StringIO() if capture_stdout else None

    with ctx._temporary_api_keys():
        try:
            if capture_stdout:
                old_stdout = sys.stdout
                sys.stdout = stdout_buffer  # type: ignore[assignment]
                try:
                    exec(compiled, sandbox_globals, sandbox_locals)
                finally:
                    sys.stdout = old_stdout
            else:
                exec(compiled, sandbox_globals, sandbox_locals)
        finally:
            _undo_scvi_shims()

    result_adata = sandbox_locals.get("adata", adata)
    normalize_doublet_obs(result_adata)

    if ctx.enable_filesystem_context and ctx._filesystem_context:
        process_context_directives(ctx, code, sandbox_locals)

    if capture_stdout:
        stdout_text = stdout_buffer.getvalue()  # type: ignore[union-attr]
        return {"adata": result_adata, "stdout": stdout_text}

    return result_adata


# ---------------------------------------------------------------------------
# Doublet harmonization
# ---------------------------------------------------------------------------

def normalize_doublet_obs(adata: Any) -> None:
    """Harmonize doublet-related obs columns."""
    try:
        obs = getattr(adata, "obs", None)
        if obs is None:
            return
        if "predicted_doublet" not in obs.columns and "predicted_doublets" in obs.columns:
            obs["predicted_doublet"] = obs["predicted_doublets"]
        col = None
        for c in ("predicted_doublet", "predicted_doublets", "doublet", "doublets"):
            if c in obs.columns:
                col = c
                break
        if col is None:
            return
        try:
            rate = float(obs[col].mean()) * 100.0
            if hasattr(adata, "uns"):
                adata.uns.setdefault("doublet_summary", {})["rate_percent"] = rate
        except Exception:
            pass
    except Exception:
        return


# ---------------------------------------------------------------------------
# Context directives
# ---------------------------------------------------------------------------

def process_context_directives(
    ctx: "AgentContext", code: str, local_vars: Dict[str, Any],
) -> None:
    """Parse and execute context directives embedded in *code*."""
    fc = ctx._filesystem_context
    if not fc:
        return
    try:
        lines = code.split("\n")
        collecting_plan = False
        plan_steps: list[dict[str, Any]] = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# CONTEXT_WRITE:"):
                _handle_context_write(ctx, stripped, local_vars)
            elif stripped.startswith("# CONTEXT_PLAN:"):
                collecting_plan = True
                plan_steps = []
            elif collecting_plan:
                if stripped.startswith("# - "):
                    step_info = _parse_plan_step(stripped[4:])
                    if step_info:
                        plan_steps.append(step_info)
                elif stripped.startswith("#"):
                    if not stripped.startswith("# CONTEXT_"):
                        continue
                    else:
                        if plan_steps:
                            fc.write_plan(plan_steps)
                        collecting_plan = False
                        plan_steps = []
                else:
                    if plan_steps:
                        fc.write_plan(plan_steps)
                    collecting_plan = False
                    plan_steps = []
            elif stripped.startswith("# CONTEXT_UPDATE:"):
                _handle_context_update(ctx, stripped)

        if collecting_plan and plan_steps:
            fc.write_plan(plan_steps)

    except Exception as e:
        logger.debug("Error processing context directives: %s", e)


def _safe_resolve_expr(value_expr: str, local_vars: Dict[str, Any]) -> Any:
    """Resolve *value_expr* without unconstrained ``eval``.

    Supported forms (in priority order):
    1. Direct name lookup in *local_vars* (e.g. ``my_var``).
    2. Dotted attribute access on a local variable (e.g. ``adata.shape``).
    3. Python literal values via ``ast.literal_eval`` (strings, numbers,
       lists, dicts, tuples, booleans, ``None``).
    4. Fallback: return *value_expr* as a plain string.
    """
    import ast as _ast

    # 1. Direct name lookup
    if value_expr in local_vars:
        return local_vars[value_expr]

    # 2. Dotted attribute access: ``name.attr1.attr2``
    parts = value_expr.split(".")
    if len(parts) > 1 and all(p.isidentifier() for p in parts):
        root = parts[0]
        if root in local_vars:
            obj = local_vars[root]
            try:
                for attr in parts[1:]:
                    obj = getattr(obj, attr)
                return obj
            except (AttributeError, TypeError):
                pass

    # 3. Literal value (string, number, list, dict, tuple, bool, None)
    try:
        return _ast.literal_eval(value_expr)
    except (ValueError, SyntaxError):
        pass

    # 4. Fallback: return the raw string
    return value_expr


def _handle_context_write(
    ctx: "AgentContext", directive: str, local_vars: Dict[str, Any],
) -> None:
    fc = ctx._filesystem_context
    if not fc:
        return
    try:
        content = directive.replace("# CONTEXT_WRITE:", "").strip()
        if " -> " in content:
            key, value_expr = content.split(" -> ", 1)
            key = key.strip()
            value_expr = value_expr.strip()
            value = _safe_resolve_expr(value_expr, local_vars)
            category = "notes"
            if any(kw in key.lower() for kw in ["result", "stats", "metrics", "output"]):
                category = "results"
            elif any(kw in key.lower() for kw in ["decision", "choice", "why"]):
                category = "decisions"
            elif any(kw in key.lower() for kw in ["error", "fail", "exception"]):
                category = "errors"
            fc.write_note(key, value, category)
            logger.debug("Context write: %s -> %s", key, category)
    except Exception as e:
        logger.debug("Failed to process CONTEXT_WRITE: %s", e)


def _handle_context_update(ctx: "AgentContext", directive: str) -> None:
    fc = ctx._filesystem_context
    if not fc:
        return
    try:
        content = directive.replace("# CONTEXT_UPDATE:", "").strip()
        parts: dict[str, str] = {}
        for part in content.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                parts[k.strip()] = v.strip().strip('"').strip("'")
        step = int(parts.get("step", "0"))
        status = parts.get("status", "completed")
        result = parts.get("result")
        fc.update_plan_step(step, status, result)
        logger.debug("Context update: step %d -> %s", step, status)
    except Exception as e:
        logger.debug("Failed to process CONTEXT_UPDATE: %s", e)


def _parse_plan_step(step_text: str) -> Optional[Dict[str, Any]]:
    """Parse a single plan step from a context directive comment."""
    try:
        status = "pending"
        if "[" in step_text and "]" in step_text:
            status_start = step_text.rfind("[")
            status_end = step_text.rfind("]")
            status = step_text[status_start + 1 : status_end].strip().lower()
            step_text = step_text[:status_start].strip()
        if step_text.lower().startswith("step "):
            colon_idx = step_text.find(":")
            if colon_idx > 0:
                step_text = step_text[colon_idx + 1 :].strip()
        return {"description": step_text, "status": status}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sandbox globals
# ---------------------------------------------------------------------------

def _make_workspace_open(ctx: "AgentContext"):
    """Return a ``open()`` wrapper that constrains file access to the workspace.

    Allowed roots:
    - The agent workspace directory (from ``ctx._filesystem_context``)
    - The current working directory
    - The system temp directory (``/tmp`` or platform equivalent)
    - The ``genesets`` sub-directory (used by OmicVerse pathway helpers)

    Relative paths are resolved against the current working directory so
    existing agent code that does ``open("output.csv", "w")`` keeps working.
    """
    import tempfile

    _builtin_open = builtins.open

    def _allowed_roots():
        roots = []
        # Workspace dir from filesystem context
        fs_ctx = getattr(ctx, "_filesystem_context", None)
        if fs_ctx is not None:
            ws = getattr(fs_ctx, "workspace_dir", None)
            if ws is not None:
                roots.append(Path(ws).resolve())
        # Current working directory
        try:
            roots.append(Path.cwd().resolve())
        except Exception:
            pass
        # System temp directory
        try:
            roots.append(Path(tempfile.gettempdir()).resolve())
        except Exception:
            pass
        return roots

    def _workspace_open(file, mode="r", *args, **kwargs):
        # Resolve the target path
        try:
            target = Path(str(file)).resolve()
        except Exception:
            target = None

        if target is not None:
            roots = _allowed_roots()
            if roots:
                allowed = any(
                    target == root or str(target).startswith(str(root) + os.sep)
                    for root in roots
                )
                if not allowed:
                    raise PermissionError(
                        f"Sandbox file access denied: path is outside the workspace boundary."
                    )

        return _builtin_open(file, mode, *args, **kwargs)

    return _workspace_open


def _undo_scvi_shims() -> None:
    """Restore MULTIVI.train to its original un-patched state.

    Called after sandbox execution to prevent the monkey-patch from
    persisting globally across runs.
    """
    try:
        from scvi.model import MULTIVI
    except Exception:
        return
    train_fn = getattr(MULTIVI, "train", None)
    orig = getattr(train_fn, "_ovbench_original", None)
    if orig is not None:
        MULTIVI.train = orig  # type: ignore[assignment]


def build_sandbox_globals(ctx: "AgentContext") -> Dict[str, Any]:
    """Construct the restricted namespace for agent code execution."""
    allowed_builtins = [
        "abs", "all", "any", "bool", "dict", "enumerate", "Exception",
        "float", "int", "isinstance", "iter", "len", "list", "map",
        "max", "min", "next", "pow", "print", "range", "round", "set",
        "sorted", "str", "sum", "tuple", "zip", "filter", "type",
        "hasattr", "getattr", "setattr",
        "ImportError", "AttributeError", "IndexError",
        "FileNotFoundError", "OSError", "Exception", "ValueError",
        "RuntimeError", "TypeError", "KeyError", "AssertionError",
    ]
    if not ctx._security_config.restrict_introspection:
        allowed_builtins.extend(["locals", "globals"])

    safe_builtins = {name: getattr(builtins, name) for name in allowed_builtins if hasattr(builtins, name)}
    safe_builtins["open"] = _make_workspace_open(ctx)
    allowed_modules: Dict[str, Any] = {}
    deny_roots = {
        "subprocess", "socket", "ssl", "urllib", "http",
        "ftplib", "smtplib", "telnetlib", "paramiko", "requests",
        "importlib", "ctypes", "multiprocessing",
    }
    deny_roots |= ctx._security_config.extra_blocked_modules
    safe_import_submodules = {
        "importlib.metadata",
        "importlib.resources",
    }
    core_modules = (
        "omicverse", "numpy", "pandas",
        "time", "math", "json", "re", "pathlib",
        "itertools", "functools", "collections",
        "statistics", "random", "warnings", "datetime", "typing",
    )
    skill_modules = (
        "openpyxl", "reportlab", "matplotlib", "seaborn",
        "scipy", "statsmodels", "sklearn",
        # scanpy is an optional dependency of omicverse; expose it (as ``sc``)
        # only when installed.
        "scanpy",
    )
    # Warn once per missing module instead of on every execute_code call —
    # reportlab/openpyxl/etc. are optional skill modules; their absence is
    # not actionable and the noise drowns out real errors in the trace.
    global _SANDBOX_MODULE_WARNED
    try:
        _SANDBOX_MODULE_WARNED
    except NameError:
        _SANDBOX_MODULE_WARNED = set()
    for module_name in core_modules + skill_modules:
        try:
            allowed_modules[module_name] = __import__(module_name)
        except ImportError:
            if module_name not in _SANDBOX_MODULE_WARNED:
                _SANDBOX_MODULE_WARNED.add(module_name)
                # Demote optional skill modules to debug; only core is warning-worthy.
                if module_name in core_modules:
                    warnings.warn(
                        f"Module '{module_name}' (core) is not available inside the agent sandbox.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    import logging as _lg
                    _lg.getLogger(__name__).debug(
                        "Optional sandbox module %r not installed (warned once); skipping.",
                        module_name,
                    )

    allowed_modules["os"] = SafeOsProxy()

    def _apply_scvi_shims() -> None:
        try:
            import functools
            from scvi.model import MULTIVI
        except Exception:
            return
        try:
            already = getattr(MULTIVI.train, "_ovbench_patched", False)
        except Exception:
            already = False
        if already:
            return
        orig_train = MULTIVI.train

        @functools.wraps(orig_train)
        def _train_wrapper(self_m, *args, **kwargs):
            kwargs.pop("early_stopping_patience", None)
            return orig_train(self_m, *args, **kwargs)

        _train_wrapper._ovbench_patched = True  # type: ignore[attr-defined]
        _train_wrapper._ovbench_original = orig_train  # type: ignore[attr-defined]
        MULTIVI.train = _train_wrapper  # type: ignore[assignment]

    def limited_import(name, globals=None, locals=None, fromlist=(), level=0):
        root_name = name.split(".")[0]
        fromlist_names = {str(item) for item in (fromlist or ())}
        allow_safe_importlib = (
            name in safe_import_submodules
            or any(name.startswith(f"{mod}.") for mod in safe_import_submodules)
            or (name == "importlib" and fromlist_names and fromlist_names <= {"metadata", "resources"})
        )
        if root_name in deny_roots and not allow_safe_importlib:
            caller_pkg = (globals or {}).get("__package__", "") or ""
            if not caller_pkg.startswith("omicverse"):
                raise ImportError(
                    f"Module '{name}' is blocked inside the OmicVerse agent sandbox."
                )
        if root_name not in allowed_modules:
            allowed_modules[root_name] = __import__(root_name)
        if root_name == "scvi":
            _apply_scvi_shims()
        return __import__(name, globals, locals, fromlist, level)

    safe_builtins["__import__"] = limited_import

    sandbox_globals: Dict[str, Any] = {"__builtins__": safe_builtins}
    sandbox_globals.update(allowed_modules)
    if "pandas" in allowed_modules:
        sandbox_globals.setdefault("pd", allowed_modules["pandas"])
    if "numpy" in allowed_modules:
        sandbox_globals.setdefault("np", allowed_modules["numpy"])
    if "scanpy" in allowed_modules:
        sandbox_globals.setdefault("sc", allowed_modules["scanpy"])
    if "omicverse" in allowed_modules:
        sandbox_globals.setdefault("ov", allowed_modules["omicverse"])
    return sandbox_globals
