"""
JarvisSession and SessionManager — per-user state for the Telegram bot.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import importlib
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# WorkspaceShell
# ---------------------------------------------------------------------------

ALLOWED_CMDS = {"ls", "find", "cat", "head", "wc", "file", "du", "pwd", "tree"}
UNLIMITED_PROMPTS_SENTINEL = 10**9
DEFAULT_KERNEL_NAME = "main"


def _ensure_local_omicverse_package_path() -> None:
    """Ensure imports can resolve the local omicverse source tree in editable-like installs."""
    package_dir = Path(__file__).resolve().parents[1]
    utils_dir = package_dir / "utils"
    repo_root = package_dir.parent

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    pkg = sys.modules.get("omicverse")
    if pkg is None:
        return

    package_paths = getattr(pkg, "__path__", None)
    if package_paths is None:
        importlib.invalidate_caches()
        return

    package_dir_str = str(package_dir)
    if package_dir_str not in package_paths:
        package_paths.append(package_dir_str)

    utils_pkg = sys.modules.get("omicverse.utils")
    if utils_pkg is not None:
        utils_paths = getattr(utils_pkg, "__path__", None)
        utils_dir_str = str(utils_dir)
        if utils_paths is not None and utils_dir_str not in utils_paths:
            utils_paths.append(utils_dir_str)

    importlib.invalidate_caches()


def _load_agent_factory() -> Any:
    """Return the OmicVerse Agent constructor without relying on top-level package exports."""
    try:
        from ..utils.smart_agent import Agent

        return Agent
    except ImportError:
        _ensure_local_omicverse_package_path()
        from omicverse.utils.smart_agent import Agent

        return Agent


class WorkspaceShell:
    """Execute a whitelist of read-only shell commands inside the workspace."""

    def exec(self, cmd: str, cwd: Path, timeout: int = 10) -> str:
        try:
            parts = shlex.split(cmd)
        except ValueError as exc:
            return f"❌ 命令解析失败：{exc}"
        if not parts:
            return "❌ 空命令"
        if parts[0] not in ALLOWED_CMDS:
            return (
                f"❌ 不允许执行 '{parts[0]}'。\n"
                f"允许的命令：{sorted(ALLOWED_CMDS)}"
            )
        try:
            result = subprocess.run(
                parts,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = (result.stdout + result.stderr).strip()
            return out if out else "(无输出)"
        except subprocess.TimeoutExpired:
            return f"❌ 命令超时（>{timeout}s）"
        except Exception as exc:
            return f"❌ 执行出错：{exc}"


# ---------------------------------------------------------------------------
# JarvisSession
# ---------------------------------------------------------------------------

@dataclass
class JarvisSession:
    """Holds per-user state: workspace path, live Agent, and loaded adata."""

    user_id: int
    workspace_dir: Path          # ~/.ovjarvis/<user_id>/
    agent: Any                   # ov.Agent (OmicVerseAgent)
    max_prompts_setting: int = 0
    adata: Optional[Any] = None
    prompt_count: int = 0
    shell: WorkspaceShell = field(default_factory=WorkspaceShell)
    last_usage: Optional[Any] = None   # token usage from most recent analysis

    # Convenience property: the actual data directory
    @property
    def workspace(self) -> Path:
        return self.workspace_dir / "workspace"

    # ------------------------------------------------------------------
    # Workspace helpers
    # ------------------------------------------------------------------

    def list_h5ad_files(self) -> List[Path]:
        """Return all .h5ad files under the workspace directory."""
        return sorted(self.workspace.rglob("*.h5ad"))

    def load_from_workspace(self, filename: str) -> Optional[Any]:
        """Find *filename* in workspace and load it as adata.

        Returns the loaded AnnData on success, None on failure.
        """
        # Accept bare filename or relative path
        candidates = list(self.workspace.rglob(filename))
        if not candidates:
            return None
        target = candidates[0]
        import anndata
        self.adata = anndata.read_h5ad(str(target))
        self.save_adata()
        return self.adata

    def get_agents_md(self) -> Optional[str]:
        """Return contents of workspace/AGENTS.md if it exists."""
        agents_md = self.workspace / "AGENTS.md"
        if agents_md.exists():
            try:
                return agents_md.read_text(encoding="utf-8").strip()
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    # Memory system (OpenClaw-style)
    # ------------------------------------------------------------------

    @property
    def memory_dir(self) -> Path:
        """Return (and create) the workspace/memory/ directory."""
        d = self.workspace / "memory"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def append_memory_log(
        self, request: str, summary: str, adata_info: str = ""
    ) -> None:
        """Append one analysis entry to today's daily log.

        File: ``workspace/memory/YYYY-MM-DD.md``
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M")

        log_file = self.memory_dir / f"{date_str}.md"
        is_new = not log_file.exists()

        # Truncate long fields so the context doesn't blow up
        req_short = request[:200] + ("..." if len(request) > 200 else "")
        sum_short = summary[:500] + ("..." if len(summary) > 500 else "")

        entry = f"\n## {time_str}\n**请求**: {req_short}\n**摘要**: {sum_short}"
        if adata_info:
            entry += f"\n**数据**: {adata_info}"
        entry += "\n"

        with open(str(log_file), "a", encoding="utf-8") as fh:
            if is_new:
                fh.write(f"# 分析日志 {date_str}\n")
            fh.write(entry)

    def get_memory_context(self) -> Optional[str]:
        """Return recent memory for injection into the Agent prompt.

        Loads (in order):
        1. ``workspace/MEMORY.md`` — long-term curated notes
        2. ``workspace/memory/yesterday.md``
        3. ``workspace/memory/today.md``
        """
        parts: List[str] = []

        # Long-term memory
        long_term = self.workspace / "MEMORY.md"
        if long_term.exists():
            try:
                content = long_term.read_text(encoding="utf-8").strip()
                if content:
                    parts.append(f"## Long-term memory\n{content}")
            except Exception:
                pass

        # Daily logs: yesterday + today
        today = datetime.now().date()
        for date in (today - timedelta(days=1), today):
            log_file = self.memory_dir / f"{date}.md"
            if log_file.exists():
                try:
                    content = log_file.read_text(encoding="utf-8").strip()
                    if content:
                        parts.append(content)
                except Exception:
                    pass

        return "\n\n".join(parts) if parts else None

    def get_recent_memory_text(self, max_entries: int = 5) -> str:
        """Return a human-readable summary of recent analyses (for /memory cmd)."""
        lines: List[str] = []
        today = datetime.now().date()
        for date in (today - timedelta(days=1), today):
            log_file = self.memory_dir / f"{date}.md"
            if log_file.exists():
                try:
                    lines.append(log_file.read_text(encoding="utf-8").strip())
                except Exception:
                    pass
        return "\n\n".join(lines) if lines else "(暂无分析历史)"

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_adata(self) -> Optional[Path]:
        """Write adata to ``workspace_dir/current.h5ad``.  Returns path or None."""
        if self.adata is None:
            return None
        path = self.workspace_dir / "current.h5ad"
        self.adata.write_h5ad(str(path))
        return path

    def load_adata(self) -> Optional[Any]:
        """Read ``workspace_dir/current.h5ad`` if it exists.  Updates self.adata."""
        path = self.workspace_dir / "current.h5ad"
        if path.exists():
            import anndata
            self.adata = anndata.read_h5ad(str(path))
        return self.adata

    def kernel_status(self) -> dict:
        """Return kernel health info from the underlying notebook executor."""
        default_max = "∞" if self.max_prompts_setting <= 0 else "?"
        info: dict = {"alive": False, "prompt_count": 0, "max_prompts": default_max, "session_id": None}
        try:
            executor = getattr(self.agent, "_notebook_executor", None)
            if executor is None:
                return info
            max_prompts = getattr(executor, "max_prompts_per_session", "?")
            if self.max_prompts_setting > 0:
                info["max_prompts"] = max_prompts
            info["prompt_count"] = getattr(executor, "session_prompt_count", 0)
            session = getattr(executor, "current_session", None)
            if session is None:
                return info
            info["session_id"] = session.get("session_id")
            # Check if kernel process is alive
            try:
                alive = executor._is_kernel_alive()
            except Exception:
                alive = False
            info["alive"] = alive
        except Exception:
            pass
        return info

    def reset(self) -> None:
        """Clear adata and restart the agent session."""
        self.adata = None
        self.prompt_count = 0
        h5ad = self.workspace_dir / "current.h5ad"
        if h5ad.exists():
            try:
                h5ad.unlink()
            except OSError:
                pass
        try:
            self.agent.restart_session()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    """Manage per-user multi-kernel sessions with one active kernel pointer."""

    class _SharedKernelNotebookExecutor:
        """Adapter that lets Jarvis reuse the web app's in-process kernel."""

        def __init__(self, kernel_executor: Any, *, timeout: int = 600) -> None:
            self._kernel_executor = kernel_executor
            self.timeout = timeout
            self.max_prompts_per_session = UNLIMITED_PROMPTS_SENTINEL
            self.session_prompt_count = 0
            self.session_history: List[Dict[str, Any]] = []
            self.current_session: Optional[Dict[str, Any]] = None

        def _ensure_session(self) -> Dict[str, Any]:
            if self.current_session is None:
                self.current_session = {
                    "session_id": f"gateway-shared-{uuid.uuid4().hex[:12]}",
                    "session_dir": str(Path.cwd()),
                    "notebook_path": None,
                    "notebook": None,
                }
            return self.current_session

        def execute(self, code: str, adata: Any) -> Any:
            self._ensure_session()
            result = self._kernel_executor.execute(code, adata=adata, timeout=self.timeout)
            self.session_prompt_count += 1
            if isinstance(result, dict):
                # Pipe captured matplotlib figures into the notebook structure so
                # AgentBridge._harvest_notebook_figures() can find them.
                figures_b64 = result.get("figures") or []
                if figures_b64:
                    nb = self.current_session.get("notebook")
                    if nb is None:
                        nb = {"cells": []}
                        self.current_session["notebook"] = nb
                    nb["cells"].append({
                        "cell_type": "code",
                        "outputs": [
                            {
                                "output_type": "display_data",
                                "data": {"image/png": b64},
                                "metadata": {},
                            }
                            for b64 in figures_b64
                        ],
                    })
                return result.get("adata")
            return result

        def _archive_current_session(self) -> None:
            if self.current_session is not None:
                self.session_history.append(dict(self.current_session))
            self.current_session = None
            self.session_prompt_count = 0

        def shutdown(self) -> None:
            # The shared kernel belongs to the web process. Jarvis sessions must
            # not tear it down when an individual agent is discarded.
            return

    def __init__(
        self,
        *,
        session_dir: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        max_prompts: int = 0,
        verbose: bool = False,
        gateway_web_bridge: Optional[Any] = None,
        shared_kernel: bool = False,
        shared_kernel_executor: Optional[Any] = None,
    ) -> None:
        self._base      = Path(session_dir or os.path.expanduser("~/.ovjarvis"))
        self._model     = model
        self._api_key   = api_key
        self._endpoint  = endpoint
        self._max_prompts_setting = max_prompts
        self._max_prompts = (
            max_prompts if max_prompts > 0 else UNLIMITED_PROMPTS_SENTINEL
        )
        self._verbose = verbose
        self._sessions: Dict[int, Dict[str, JarvisSession]] = {}
        self._active_kernel: Dict[int, str] = {}
        self._kernel_name_re = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")
        self._shared_adata: Any = None
        self._lock = threading.Lock()
        self._shared_kernel = bool(shared_kernel)
        self._shared_kernel_user_id = 0
        self._shared_kernel_executor = shared_kernel_executor
        # Optional WebSessionBridge — set when running with --with-web
        self.gateway_web_bridge: Optional[Any] = gateway_web_bridge

    # ------------------------------------------------------------------

    def _read_adata_from_shared_kernel(self) -> Any:
        executor = self._shared_kernel_executor
        if executor is None:
            return None
        try:
            ensure_kernel = getattr(executor, "_ensure_kernel", None)
            if callable(ensure_kernel):
                ensure_kernel()
            shell = getattr(executor, "shell", None)
            user_ns = getattr(shell, "user_ns", None)
            if not isinstance(user_ns, dict):
                return None
            return user_ns.get("odata") or user_ns.get("adata")
        except Exception:
            return None

    def _resolve_shared_adata_locked(self) -> Any:
        kernel_adata = self._read_adata_from_shared_kernel()
        if kernel_adata is not None and kernel_adata is not self._shared_adata:
            self._shared_adata = kernel_adata
        return self._shared_adata

    def _attach_shared_executor_to_agent(self, agent: Any) -> None:
        if agent is None or self._shared_kernel_executor is None:
            return
        timeout = getattr(agent, "notebook_timeout", 600)
        agent.use_notebook_execution = True
        agent._notebook_executor = self._SharedKernelNotebookExecutor(
            self._shared_kernel_executor,
            timeout=timeout,
        )

    def attach_shared_kernel_executor(self, kernel_executor: Any) -> None:
        with self._lock:
            self._shared_kernel_executor = kernel_executor
            shared = self._resolve_shared_adata_locked()
            for user_sessions in self._sessions.values():
                for session in user_sessions.values():
                    self._attach_shared_executor_to_agent(session.agent)
                    if shared is not None:
                        session.adata = shared

    def _effective_user_id(self, user_id: int) -> int:
        if self._shared_kernel:
            return self._shared_kernel_user_id
        return user_id

    def get_or_create(self, user_id: int) -> JarvisSession:
        """Return the active kernel session for *user_id* (default: ``main``)."""
        effective_user_id = self._effective_user_id(user_id)
        active = self.get_active_kernel(effective_user_id)
        session = self._get_or_create_kernel_session(effective_user_id, active)
        with self._lock:
            shared = self._resolve_shared_adata_locked()
            if session.adata is None and shared is not None:
                session.adata = shared
        return session

    def find_session_by_bridge_session_id(self, bridge_session_id: str) -> Optional[JarvisSession]:
        """Return an existing JarvisSession whose live notebook session matches."""
        target = str(bridge_session_id or "").strip()
        if not target:
            return None

        try:
            from ._bridge_session import resolve_bridge_session_id
        except Exception:
            return None

        with self._lock:
            sessions = [
                session
                for user_sessions in self._sessions.values()
                for session in user_sessions.values()
            ]

        for session in sessions:
            try:
                current = resolve_bridge_session_id(session, create=False)
            except Exception:
                current = ""
            if current == target:
                return session
        return None

    def set_shared_adata(self, adata: Any) -> None:
        """Set the shared AnnData reference used by all Jarvis sessions."""
        with self._lock:
            self._shared_adata = adata
            for user_sessions in self._sessions.values():
                for session in user_sessions.values():
                    session.adata = adata
        executor = self._shared_kernel_executor
        if executor is not None:
            try:
                sync_adata = getattr(executor, "sync_adata", None)
                if callable(sync_adata):
                    sync_adata(adata)
            except Exception:
                pass

    def get_shared_adata(self) -> Any:
        """Return the current shared AnnData reference, if any."""
        with self._lock:
            return self._resolve_shared_adata_locked()

    def get_active_kernel(self, user_id: int) -> str:
        user_id = self._effective_user_id(user_id)
        if user_id not in self._active_kernel:
            self._active_kernel[user_id] = DEFAULT_KERNEL_NAME
        return self._active_kernel[user_id]

    def list_kernels(self, user_id: int) -> List[str]:
        user_id = self._effective_user_id(user_id)
        names = set(self._discover_kernel_names(user_id))
        names.update(self._sessions.get(user_id, {}).keys())
        if not names:
            names.add(DEFAULT_KERNEL_NAME)
        return sorted(names)

    def switch_kernel(self, user_id: int, kernel_name: str, create: bool = False) -> JarvisSession:
        user_id = self._effective_user_id(user_id)
        name = self.normalize_kernel_name(kernel_name)
        exists_in_mem = name in self._sessions.get(user_id, {})
        exists_on_disk = (name == DEFAULT_KERNEL_NAME) or self._kernel_exists_on_disk(user_id, name)
        if not exists_in_mem and not exists_on_disk and not create:
            raise KeyError(f"Kernel '{name}' 不存在")
        session = self._get_or_create_kernel_session(user_id, name)
        self._active_kernel[user_id] = name
        return session

    def create_kernel(self, user_id: int, kernel_name: str, switch: bool = True) -> JarvisSession:
        user_id = self._effective_user_id(user_id)
        name = self.normalize_kernel_name(kernel_name)
        if self._kernel_exists_on_disk(user_id, name) or name in self._sessions.get(user_id, {}):
            raise ValueError(f"Kernel '{name}' 已存在")
        session = self._get_or_create_kernel_session(user_id, name)
        if switch:
            self._active_kernel[user_id] = name
        return session

    def normalize_kernel_name(self, kernel_name: str) -> str:
        name = (kernel_name or "").strip()
        if not name:
            raise ValueError("Kernel 名称不能为空")
        if not self._kernel_name_re.match(name):
            raise ValueError("Kernel 名称仅允许字母/数字/._-，且长度 1-32")
        return name

    def _get_or_create_kernel_session(self, user_id: int, kernel_name: str) -> JarvisSession:
        with self._lock:
            user_sessions = self._sessions.setdefault(user_id, {})
            if kernel_name in user_sessions:
                return user_sessions[kernel_name]

            kernel_root = self._kernel_root(user_id, kernel_name)
            self._ensure_kernel_dirs(kernel_root)
            agent = self._build_agent(kernel_root)
            self._attach_shared_executor_to_agent(agent)
            session = JarvisSession(
                user_id=user_id,
                workspace_dir=kernel_root,
                agent=agent,
                max_prompts_setting=self._max_prompts_setting,
                adata=self._resolve_shared_adata_locked(),
            )
            user_sessions[kernel_name] = session
            return session

    def _kernel_root(self, user_id: int, kernel_name: str) -> Path:
        user_dir = self._base / str(user_id)
        if kernel_name == DEFAULT_KERNEL_NAME:
            return user_dir
        return user_dir / "kernels" / kernel_name

    def _ensure_kernel_dirs(self, kernel_root: Path) -> None:
        workspace_dir = kernel_root / "workspace"
        sessions_dir = kernel_root / "sessions"
        context_dir = kernel_root / "context"
        memory_dir = workspace_dir / "memory"
        for d in (kernel_root, workspace_dir, sessions_dir, context_dir, memory_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _kernel_exists_on_disk(self, user_id: int, kernel_name: str) -> bool:
        root = self._kernel_root(user_id, kernel_name)
        if kernel_name == DEFAULT_KERNEL_NAME:
            return (root / "workspace").exists() or (root / "sessions").exists() or (root / "context").exists()
        return root.exists()

    def _discover_kernel_names(self, user_id: int) -> List[str]:
        user_dir = self._base / str(user_id)
        names: List[str] = []
        if self._kernel_exists_on_disk(user_id, DEFAULT_KERNEL_NAME):
            names.append(DEFAULT_KERNEL_NAME)
        kernels_root = user_dir / "kernels"
        if kernels_root.exists():
            for d in sorted(kernels_root.iterdir()):
                if d.is_dir():
                    names.append(d.name)
        return names

    def _build_agent(self, kernel_root: Path) -> Any:
        kwargs: Dict[str, Any] = dict(
            model=self._model,
            use_notebook_execution=self._shared_kernel_executor is None,
            strict_kernel_validation=False,
            verbose=self._verbose,
            max_prompts_per_session=self._max_prompts,
            notebook_storage_dir=str(kernel_root / "sessions"),
            context_storage_dir=str(kernel_root / "context"),
            enable_filesystem_context=True,
        )
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._endpoint:
            kwargs["endpoint"] = self._endpoint

        agent_factory = _load_agent_factory()
        return agent_factory(**kwargs)
