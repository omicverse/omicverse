"""
OmicVerse Smart Agent — thin facade / composer.

This module provides the public ``OmicVerseAgent`` class and the ``Agent()``
factory.  All substantial logic now lives in the ``ovagent/`` subpackage;
this file wires the subsystems together and exposes the stable public API.

Usage:
    import omicverse as ov
    result = ov.Agent("quality control with nUMI>500, mito<0.2", adata)
"""

import asyncio
import inspect
import json
import logging
import os
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, TypeVar, Union

_T = TypeVar("_T")

# Internal LLM backend
from .agent_backend import OmicVerseLLMBackend, Usage

# Registry system and model configuration
from .._registry import _global_registry
from .model_config import ModelConfig, PROVIDER_API_KEYS

# Grouped configuration dataclasses
from .agent_config import AgentConfig, SandboxFallbackPolicy

# Structured error hierarchy
from .agent_errors import (
    OVAgentError,
    ProviderError,
    ConfigError,
    ExecutionError,
    SandboxDeniedError,
    SecurityViolationError,
)

# Sandbox security hardening
from .agent_sandbox import (
    ApprovalMode,
    CodeSecurityScanner,
    SafeOsProxy,
    SecurityConfig,
)

# Structured event reporting
from .agent_reporter import (
    AgentEvent,
    EventLevel,
    Reporter,
    make_reporter,
)

# Harness helpers
from .harness import (
    RunTraceRecorder,
    RunTraceStore,
    build_stream_event,
    coerce_usage_payload,
    hash_code_block,
    make_turn_id,
)
from .harness.tool_catalog import (
    get_default_loaded_tool_names,
    get_visible_tool_schemas,
)

# Context compactor
from .context_compactor import ContextCompactor

# OVAgent extracted modules
from .ovagent.runtime import OmicVerseRuntime
from .ovagent.auth import (
    ResolvedBackend as _ResolvedBackend,
    resolve_model_and_provider as _resolve_model_and_provider,
    collect_api_key_env as _collect_api_key_env,
    temporary_api_keys as _temporary_api_keys_cm,
    display_backend_info as _display_backend_info,
    resolve_credentials as _resolve_agent_llm_credentials,
    # Backward-compat re-exports (moved from this module to ovagent.auth)
    _normalize_model_for_routing,
)
from .ovagent.bootstrap import (
    format_skill_overview as _format_skill_overview,
    initialize_skill_registry as _initialize_skill_registry,
    initialize_notebook_executor as _initialize_notebook_executor,
    initialize_security as _initialize_security,
    create_llm_backend as _create_llm_backend,
    display_reflection_config as _display_reflection_config,
)
from .ovagent.prompt_builder import (
    PromptBuilder as _PromptBuilder,
    CODE_QUALITY_RULES as _CODE_QUALITY_RULES_EXT,
    build_filesystem_context_instructions as _build_filesystem_context_instructions,
)
from .ovagent.analysis_executor import (
    AnalysisExecutor as _AnalysisExecutor,
    ProactiveCodeTransformer,
)
from .ovagent.tool_runtime import (
    LEGACY_AGENT_TOOLS as _LEGACY_AGENT_TOOLS,
    ToolRuntime as _ToolRuntime,
)
from .ovagent.codegen_pipeline import CodegenPipeline as _CodegenPipeline
from .ovagent.subagent_controller import SubagentController as _SubagentController
from .ovagent.turn_controller import TurnController as _TurnController
from .ovagent.session_facade import SessionContextFacadeMixin
from .ovagent.codegen_tool_facade import CodegenToolDispatchFacadeMixin
from .ovagent.registry_scanner import (
    RegistryScanner as _RegistryScanner,
    build_compact_registry_summary as _build_compact_registry_summary,
)

# Session history
from .session_history import HistoryEntry, SessionHistory

# Skill registry
from .skill_registry import (
    SkillMatch,
    SkillMetadata,
    SkillDefinition,
    SkillRegistry,
    SkillRouter,
    build_skill_registry,
    discover_multi_path_skill_roots,
    build_multi_path_skill_registry,
)

# Filesystem context management
from .filesystem_context import FilesystemContextManager


logger = logging.getLogger(__name__)

def _wire_package_import_compatibility() -> None:
    """Expose parent package references without probing lazy __getattr__ hooks."""
    parent_pkg = sys.modules.get("omicverse")
    utils_pkg = sys.modules.get("omicverse.utils")
    module = sys.modules.get(__name__)
    if parent_pkg is None or utils_pkg is None or module is None:
        return

    parent_attrs = getattr(parent_pkg, "__dict__", {})
    if "utils" not in parent_attrs:
        setattr(parent_pkg, "utils", utils_pkg)

    module_name = __name__.rsplit(".", 1)[-1]
    utils_attrs = getattr(utils_pkg, "__dict__", {})
    if module_name not in utils_attrs:
        setattr(utils_pkg, module_name, module)


_wire_package_import_compatibility()


def _run_coroutine_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run *coro* synchronously, preserving exception tracebacks.

    * **No running loop** — delegates to ``asyncio.run()``.
    * **Running loop detected** (e.g. Jupyter / Jarvis) — spawns a daemon
      worker thread with its own event loop.  Exceptions are transported via
      ``sys.exc_info()`` and re-raised with ``with_traceback()`` so the
      caller sees the full original traceback chain instead of a collapsed
      single-frame ``raise err`` from a dict container.

    This function never imports or patches ``nest_asyncio``.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        _result: list = [None]
        _exc_info: list = [None]

        def _worker() -> None:
            try:
                _result[0] = asyncio.run(coro)
            except BaseException:
                _exc_info[0] = sys.exc_info()

        thread = threading.Thread(
            target=_worker, name="OmicVerseAgentSync", daemon=True,
        )
        thread.start()
        thread.join()

        if _exc_info[0] is not None:
            _tp, exc, tb = _exc_info[0]
            _exc_info[0] = None
            try:
                raise exc.with_traceback(tb)
            finally:
                del exc, tb

        return _result[0]  # type: ignore[return-value]

    return asyncio.run(coro)


class OmicVerseAgent(CodegenToolDispatchFacadeMixin, SessionContextFacadeMixin):
    """
    Intelligent agent for OmicVerse function discovery and execution.

    This agent uses an internal LLM backend to understand natural language
    requests and automatically execute appropriate OmicVerse functions.

    Usage:
        agent = ov.Agent(api_key="your-api-key")  # Uses gpt-5.2 by default
        result_adata = agent.run("quality control with nUMI>500, mito<0.2", adata)
    """
    # Class-level tool schemas (backward-compat surface)
    LEGACY_AGENT_TOOLS = _LEGACY_AGENT_TOOLS
    AGENT_TOOLS = get_visible_tool_schemas(get_default_loaded_tool_names()) + _LEGACY_AGENT_TOOLS

    @staticmethod
    def _emit(level: "EventLevel", message: str, category: str = "") -> None:
        """No-op fallback; replaced by ``__init__`` with a reporter-backed emitter."""

    def __init__(self, model: str = "gpt-5.2", api_key: Optional[str] = None, endpoint: Optional[str] = None, auth_mode: str = "environment", auth_provider: Optional[str] = None, auth_file: Optional[str] = None, enable_reflection: bool = True, reflection_iterations: int = 1, enable_result_review: bool = True, use_notebook_execution: bool = True, max_prompts_per_session: int = 5, notebook_storage_dir: Optional[str] = None, keep_execution_notebooks: bool = True, notebook_timeout: int = 600, strict_kernel_validation: bool = True, enable_filesystem_context: bool = True, context_storage_dir: Optional[str] = None, approval_mode: str = "never", agent_mode: str = "agentic", max_agent_turns: int = 15, security_level: Optional[str] = None, *, config: Optional[AgentConfig] = None, reporter: Optional[Reporter] = None, verbose: bool = True):
        """
        Initialize the OmicVerse Smart Agent.

        Parameters
        ----------
        model : str
            LLM model to use for reasoning (default: "gpt-5.2")
        api_key : str, optional
            API key for the model provider. If not provided, will use environment variable
        endpoint : str, optional
            Custom API endpoint. If not provided, will use default for the provider
        auth_mode : str, optional
            Authentication mode. Use ``"openai_oauth"`` to reuse Jarvis/Codex login state.
        auth_provider : str, optional
            OAuth provider identifier. Currently only ``"codex"`` is supported.
        auth_file : str, optional
            Path to the saved Jarvis auth file. Defaults to ``~/.ovjarvis/auth.json``.
        enable_reflection : bool, optional
            Enable reflection step to review and improve generated code (default: True)
        reflection_iterations : int, optional
            Maximum number of reflection iterations (default: 1, range: 1-3)
        enable_result_review : bool, optional
            Enable result review to validate output matches user intent (default: True)
        use_notebook_execution : bool, optional
            Execute code in separate Jupyter notebook for isolation and debugging (default: True).
            Set to False to use legacy in-process execution.
        max_prompts_per_session : int, optional
            Number of prompts to execute in same notebook session before restart (default: 5).
            This prevents memory bloat while maintaining context for iterative analysis.
        notebook_storage_dir : str, optional
            Directory to store session notebooks. Defaults to ~/.ovagent/sessions
        keep_execution_notebooks : bool, optional
            Whether to keep session notebooks after execution (default: True)
        notebook_timeout : int, optional
            Execution timeout in seconds (default: 600)
        strict_kernel_validation : bool, optional
            If True, raise error if kernel not found. If False, fall back to python3 kernel (default: True)
        enable_filesystem_context : bool, optional
            Enable filesystem-based context management for offloading intermediate results,
            plans, and notes to disk. This reduces context window usage and enables
            selective context retrieval. Default: True.
        context_storage_dir : str, optional
            Directory for storing context files. Defaults to ~/.ovagent/context/
        config : AgentConfig, optional
            Grouped configuration object.  When provided, its values take priority
            over the flat keyword arguments above.
        reporter : Reporter, optional
            Structured event reporter.  When omitted a default reporter is
            created based on the *verbose* flag.
        verbose : bool, optional
            Whether to emit events to stdout (default: True).
        """

        # --- Build AgentConfig (P0-2) ------------------------------------------
        if config is not None:
            self._config = config
        else:
            if agent_mode != "agentic":
                import warnings
                warnings.warn(
                    "agent_mode='legacy' is deprecated and ignored. "
                    "Agentic mode is now the only execution mode.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            self._config = AgentConfig.from_flat_kwargs(
                model=model,
                api_key=api_key,
                endpoint=endpoint,
                auth_mode=auth_mode,
                auth_provider=auth_provider,
                auth_file=auth_file,
                enable_reflection=enable_reflection,
                reflection_iterations=reflection_iterations,
                enable_result_review=enable_result_review,
                use_notebook_execution=use_notebook_execution,
                max_prompts_per_session=max_prompts_per_session,
                notebook_storage_dir=notebook_storage_dir,
                keep_execution_notebooks=keep_execution_notebooks,
                notebook_timeout=notebook_timeout,
                strict_kernel_validation=strict_kernel_validation,
                enable_filesystem_context=enable_filesystem_context,
                context_storage_dir=context_storage_dir,
                approval_mode=approval_mode,
                max_agent_turns=max_agent_turns,
                security_level=security_level,
            )
            self._config.verbose = verbose

        # --- Build Reporter (P1-1) ---------------------------------------------
        self._reporter: Reporter = make_reporter(
            verbose=self._config.verbose,
            reporter=reporter,
        )

        def _emit(level: EventLevel, message: str, category: str = "") -> None:
            self._reporter.emit(AgentEvent(level=level, message=message, category=category))

        self._emit = _emit

        _emit(EventLevel.INFO, "Initializing OmicVerse Smart Agent (internal backend)...", "init")
        llm_cfg = self._config.llm
        model, api_key, endpoint, resolved_auth_mode = _resolve_agent_llm_credentials(
            model=llm_cfg.model,
            api_key=llm_cfg.api_key,
            endpoint=llm_cfg.endpoint,
            auth_mode=llm_cfg.auth_mode,
            auth_provider=llm_cfg.auth_provider,
            auth_file=llm_cfg.auth_file,
        )
        llm_cfg.model = model
        llm_cfg.api_key = api_key
        llm_cfg.endpoint = endpoint
        llm_cfg.auth_mode = resolved_auth_mode
        model = llm_cfg.model
        api_key = llm_cfg.api_key
        endpoint = llm_cfg.endpoint
        self.auth_mode = llm_cfg.auth_mode
        self.auth_provider = llm_cfg.auth_provider
        self.auth_file = str(llm_cfg.auth_file) if llm_cfg.auth_file else None
        
        # When using a custom endpoint (proxy), keep the model name as-is.
        # Proxies expect the exact model name the user typed.
        if endpoint:
            _emit(EventLevel.INFO, f"Proxy mode: model={model}, endpoint={endpoint}", "init")
        else:
            # Normalize model ID for aliases and variations, then validate
            original_model = model
            try:
                model = ModelConfig.normalize_model_id(model)  # type: ignore[attr-defined]
            except (AttributeError, KeyError, ValueError, TypeError) as exc:
                logger.debug("Model ID normalization fallback for %r: %s", model, exc)
                model = model
            if model != original_model:
                _emit(EventLevel.INFO, f"Model ID normalized: {original_model} → {model}", "init")

        # --- Auth & backend resolution (ovagent.auth) --------------------------
        backend = _resolve_model_and_provider(model, api_key, endpoint)
        self.model = backend.model
        self.api_key = api_key
        self.endpoint = backend.endpoint
        self.provider = backend.provider

        # --- Attribute defaults ------------------------------------------------
        self._llm: Optional[OmicVerseLLMBackend] = None
        self.skill_registry: Optional[SkillRegistry] = None
        self._skill_overview_text: str = ""
        self._use_llm_skill_matching: bool = True
        self._managed_api_env: Dict[str, str] = {}
        # Reflection configuration
        self.enable_reflection = self._config.reflection.enabled
        self.reflection_iterations = self._config.reflection.iterations
        # Result review configuration
        self.enable_result_review = self._config.reflection.result_review
        # Notebook execution configuration
        self.use_notebook_execution = self._config.execution.use_notebook
        self.max_prompts_per_session = self._config.execution.max_prompts_per_session
        self._notebook_executor = None
        # Filesystem context configuration (set early to avoid AttributeError)
        self.enable_filesystem_context = self._config.context.enabled
        self._filesystem_context: Optional[FilesystemContextManager] = None
        self.last_usage = None
        self.last_usage_breakdown: Dict[str, Any] = {
            'generation': None,
            'reflection': [],
            'review': [],
            'total': None
        }
        self._code_only_mode: bool = False
        self._code_only_captured_code: str = ""
        self._code_only_captured_history: List[Dict[str, Any]] = []
        self._session_history: Optional[SessionHistory] = None
        self._trace_store: Optional[RunTraceStore] = None
        self._context_compactor: Optional[ContextCompactor] = None
        self._last_run_trace = None
        self._approval_handler = None
        self._web_session_id = ""
        self._ov_runtime: Optional[OmicVerseRuntime] = None
        self._active_run_id = ""
        self._registry_scanner = _RegistryScanner()

        # --- API key env collection (ovagent.auth) -----------------------------
        try:
            self._managed_api_env = _collect_api_key_env(
                self.model, self.endpoint, api_key,
            )
        except (KeyError, ValueError, TypeError, OSError) as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to collect API key environment variables: %s", exc)
            self._managed_api_env = {}

        # --- Skill registry (ovagent.bootstrap) --------------------------------
        self.skill_registry, self._skill_overview_text = _initialize_skill_registry()

        # --- Display backend info (ovagent.auth) -------------------------------
        _display_backend_info(
            self.model, self.endpoint, self.provider,
            self.api_key, self._managed_api_env,
        )

        # --- Subsystem bootstrap (ovagent.bootstrap) ---------------------------
        try:
            with self._temporary_api_keys():
                self._setup_agent()
            stats = self._get_registry_stats()
            _emit(EventLevel.INFO, f"Function registry loaded: {stats['total_functions']} functions in {stats['categories']} categories", "init")

            _display_reflection_config(
                self.enable_reflection,
                self.reflection_iterations,
                self.enable_result_review,
            )

            # Notebook executor
            nb_storage = Path(notebook_storage_dir) if notebook_storage_dir else None
            self.use_notebook_execution, self._notebook_executor = (
                _initialize_notebook_executor(
                    use_notebook=self.use_notebook_execution,
                    storage_dir=nb_storage,
                    max_prompts_per_session=max_prompts_per_session,
                    keep_notebooks=keep_execution_notebooks,
                    timeout=notebook_timeout,
                    strict_kernel_validation=strict_kernel_validation,
                )
            )

            # Session, context, tracing, OV runtime (SessionContextFacadeMixin)
            ctx_storage = Path(context_storage_dir) if context_storage_dir else None
            self._initialize_session_context_tracing(ctx_storage_dir=ctx_storage)

            # Security scanner
            self._security_config, self._security_scanner = _initialize_security(
                self._config,
            )

            # Session/context service delegates (SessionContextFacadeMixin)
            self._wire_session_context_services()
            self._prompt_builder = _PromptBuilder(self)
            self._analysis_executor = _AnalysisExecutor(self)
            self._tool_runtime = _ToolRuntime(self, self._analysis_executor)
            self._subagent_controller = _SubagentController(
                self, self._prompt_builder, self._tool_runtime,
            )
            self._tool_runtime.set_subagent_controller(self._subagent_controller)
            self._turn_controller = _TurnController(
                self, self._prompt_builder, self._tool_runtime,
            )
            self._codegen_pipeline = _CodegenPipeline(self)
            self._analysis_executor._set_codegen_pipeline(self._codegen_pipeline)

            _emit(EventLevel.SUCCESS, "Smart Agent initialized successfully!", "init")
        except Exception as e:
            _emit(EventLevel.ERROR, f"Agent initialization failed: {e}", "init")
            raise

    # =====================================================================
    # Internal setup (called once from __init__)
    # =====================================================================

    def _get_registry_stats(self) -> dict:
        """Get statistics about the function registry."""
        static_entries = self._scanner.load_static_entries()
        unique_functions = set(e['full_name'] for e in static_entries)
        categories = set(e['category'] for e in static_entries)

        if not unique_functions:
            for entry in _global_registry._registry.values():
                unique_functions.add(entry['full_name'])
                categories.add(entry['category'])

        return {
            'total_functions': len(unique_functions),
            'categories': len(categories),
            'category_list': list(categories)
        }

    def _get_compact_registry_summary(self) -> str:
        """Build the shared compact registry summary used in prompts."""
        return _build_compact_registry_summary(self._scanner)

    def _setup_agent(self):
        """Setup the internal agent backend with dynamic instructions.

        Uses a compact category-level registry summary instead of
        dumping every function as JSON.  The LLM discovers detailed
        signatures through the ``search_functions`` tool at runtime.
        """
        compact_summary = self._get_compact_registry_summary()

        instructions = (
            "You are an intelligent OmicVerse assistant that can automatically "
            "discover and execute functions based on natural language requests.\n\n"
            "## OmicVerse Function Registry (compact overview)\n\n"
            "The following categories of functions are registered.  "
            "Use the `search_functions` tool to look up signatures, parameters, "
            "prerequisites, and examples for any function before generating code.\n\n"
            + compact_summary + "\n\n"
            "## Your Task\n\n"
            "When given a natural language request and an adata object, you should:\n\n"
            "Quick-start examples (non-experts can copy/paste):\n"
            '- "basic single-cell QC and clustering" (uses QC -> preprocess -> neighbors/UMAP -> Leiden -> markers)\n'
            '- "batch integration with harmony on this adata" (uses harmony then neighbors/UMAP/Leiden, use_raw=False)\n'
            '- "simple trajectory with DPT, root on paul15_clusters=7MEP, list top genes"\n'
            '- "find markers for each Leiden cluster (wilcoxon)"\n'
            '- "doublet check and report rate"\n\n'
            "1. **Analyze the request** to understand what the user wants to accomplish\n"
            "2. **Call `search_functions`** with relevant keywords to find the appropriate function, "
            "its exact signature, prerequisites, and examples\n"
            "3. **Extract parameters** from the user's request (e.g., \"nUMI>500\" means min_genes=500)\n"
            "4. **Generate and execute Python code** using the appropriate OmicVerse function\n"
            "5. **Return the modified adata object**\n\n"
            "## Parameter Extraction Rules\n\n"
            "Extract parameters dynamically based on patterns in the user request:\n\n"
            "- For qc function: Create tresh dict with 'mito_perc', 'nUMIs', 'detected_genes'\n"
            "  - \"nUMI>X\", \"umi>X\" -> tresh={'nUMIs': X, 'detected_genes': 250, 'mito_perc': 0.15}\n"
            "  - \"mito<X\", \"mitochondrial<X\" -> include in tresh dict as 'mito_perc': X\n"
            "  - \"genes>X\" -> include in tresh dict as 'detected_genes': X\n"
            "  - Always provide complete tresh dict with all three keys\n"
            "- \"resolution=X\" -> resolution=X\n"
            "- \"n_pcs=X\", \"pca=X\" -> n_pcs=X\n"
            "- \"max_value=X\" -> max_value=X\n"
            "- Mode indicators: \"seurat\", \"mads\", \"pearson\" -> mode=\"seurat\"\n"
            "- Boolean indicators: \"no doublets\", \"skip doublets\" -> doublets=False\n\n"
            "## Code Execution Rules\n\n"
            "1. **Always import omicverse as ov** at the start\n"
            "2. **Use the exact function signature** returned by search_functions\n"
            "3. **Handle the adata variable** - it will be provided in the context\n"
            "4. **Update adata in place** when possible\n"
            "5. **Print success messages** and basic info about the result\n\n"
            "## Important Notes\n\n"
            "- Always work with the provided `adata` variable\n"
            "- Use the function signatures exactly as returned by search_functions\n"
            "- Provide helpful feedback about what was executed\n"
            "- Do not create dummy AnnData objects; operate directly on the provided data\n"
            "- Prefer `use_raw=False` unless the user explicitly requests raw\n"
            "- Handle errors gracefully and suggest alternatives if needed\n\n"
        ) + _CODE_QUALITY_RULES_EXT

        if self._skill_overview_text:
            instructions += (
                "\n\n## Project Skill Catalog\n"
                "OmicVerse provides curated Agent Skills that capture end-to-end workflows. "
                "Before executing complex tasks, call `_list_project_skills` to view the catalog and `_load_skill_guidance` "
                "to read detailed instructions for relevant skills. Follow the selected skill guidance when planning code "
                "execution.\n\n"
                f"{self._skill_overview_text}"
            )

        if self.enable_filesystem_context and self._filesystem_context:
            instructions += self._context_service.build_filesystem_context_instructions() if hasattr(self, '_context_service') else _build_filesystem_context_instructions(self._filesystem_context)

        if self.api_key:
            required_key = PROVIDER_API_KEYS.get(self.model)
            if required_key and not os.getenv(required_key):
                os.environ[required_key] = self.api_key

        self._llm = _create_llm_backend(
            system_prompt=instructions,
            model=self.model,
            api_key=self.api_key,
            endpoint=self.endpoint,
        )

    # =====================================================================
    # Protocol-required methods (called by ovagent modules via AgentContext)
    # =====================================================================

    @contextmanager
    def _temporary_api_keys(self):
        """Temporarily inject API keys into the environment and clean up afterwards."""
        with _temporary_api_keys_cm(self._managed_api_env):
            yield

    def _resolve_local_path(self, file_path: str, *, allow_relative: bool = False) -> Path:
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            if not allow_relative:
                raise ValueError("file_path must be an absolute path")
            path = Path(self._refresh_runtime_working_directory()) / path
        return path.resolve()

    def _ensure_server_tool_mode(self, tool_name: str) -> None:
        harness_config = getattr(self._config, "harness", None)
        if harness_config is None or not getattr(harness_config, "server_tool_mode", True):
            raise RuntimeError(f"{tool_name} is disabled because server_tool_mode is off.")

    def _request_interaction(self, payload: dict[str, Any]) -> Any:
        if self._approval_handler is None:
            return True
        return self._approval_handler(payload)

    def _request_tool_approval(self, tool_name: str, *, reason: str, payload: dict[str, Any]) -> None:
        from .harness.tool_catalog import get_tool_spec
        spec = get_tool_spec(tool_name)
        if spec is None or not spec.requires_approval:
            return
        interaction = {
            "kind": "approval",
            "title": f"{tool_name} approval required",
            "message": reason,
            "tool_name": tool_name,
            "payload": payload,
            "session_id": self._get_runtime_session_id(),
            "trace_id": getattr(getattr(self, "_last_run_trace", None), "trace_id", ""),
        }
        approved = self._request_interaction(interaction)
        if not approved:
            raise PermissionError(f"{tool_name} was not approved by the user.")

    def _build_agentic_system_prompt(self) -> str:
        return self._prompt_builder.build_agentic_system_prompt()

    async def _run_agentic_loop(self, request: str, adata: Any,
                               event_callback=None,
                               cancel_event=None,
                               history=None,
                               approval_handler=None,
                               request_content=None) -> Any:
        return await self._turn_controller.run_agentic_loop(
            request, adata,
            event_callback=event_callback,
            cancel_event=cancel_event,
            history=history,
            approval_handler=approval_handler,
            request_content=request_content,
        )

    def _load_skill_guidance(self, skill_name: str) -> str:
        """Return the detailed instructions for a requested skill."""
        if not self.skill_registry or not self.skill_registry.skill_metadata:
            return json.dumps({"error": "No project skills are available."})
        if not skill_name or not skill_name.strip():
            return json.dumps({"error": "Provide a skill name to load guidance."})
        slug = skill_name.strip().lower()
        definition = self.skill_registry.load_full_skill(slug)
        if not definition:
            return json.dumps({"error": f"Skill '{skill_name}' not found."})
        return json.dumps(
            {
                "name": definition.name,
                "description": definition.description,
                "instructions": definition.prompt_instructions(provider=getattr(self, "provider", None)),
                "path": str(definition.path),
                "metadata": definition.metadata,
            },
            indent=2,
        )

    # =====================================================================
    # Skill helpers
    # =====================================================================

    def _list_project_skills(self) -> str:
        """Return a JSON catalog of the discovered project skills."""
        if not self.skill_registry or not self.skill_registry.skill_metadata:
            return json.dumps({"skills": [], "message": "No project skills available."}, indent=2)
        skills_payload = [
            {
                "name": skill.name,
                "slug": skill.slug,
                "description": skill.description,
                "path": str(skill.path),
                "metadata": skill.metadata,
            }
            for skill in sorted(self.skill_registry.skill_metadata.values(), key=lambda item: item.name.lower())
        ]
        return json.dumps({"skills": skills_payload}, indent=2)

    async def _select_skill_matches_llm(self, request: str, top_k: int = 2) -> List[str]:
        """Use LLM to select relevant skills based on the request."""
        if not self.skill_registry or not self.skill_registry.skill_metadata:
            return []
        skills_list = []
        for skill in sorted(self.skill_registry.skill_metadata.values(), key=lambda s: s.name.lower()):
            skills_list.append(f"- **{skill.slug}**: {skill.description}")
        skills_catalog = "\n".join(skills_list)
        matching_prompt = f"""You are a skill matching system. Given a user request and a list of available skills, determine which skills (if any) are relevant.

User Request: "{request}"

Available Skills:
{skills_catalog}

Your task:
1. Analyze the user request to understand their intent
2. Review the skill descriptions
3. Select the {top_k} most relevant skills (or fewer if not many are relevant)
4. Respond with ONLY the skill slugs as a JSON array, e.g., ["skill-slug-1", "skill-slug-2"]
5. If no skills are relevant, return an empty array: []

IMPORTANT: Respond with ONLY the JSON array, nothing else."""

        try:
            with self._temporary_api_keys():
                if not self._llm:
                    return []
                response = await self._llm.run(matching_prompt)
            import re as _re
            json_match = _re.search(r'\[.*?\]', response, _re.DOTALL)
            if json_match:
                matched_slugs = json.loads(json_match.group(0))
                return [slug for slug in matched_slugs if slug in self.skill_registry.skill_metadata]
            return []
        except (RuntimeError, ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
            logger.warning("LLM skill matching failed: %s", exc)
            return []

    def _format_skill_guidance(self, matches: List[SkillMatch]) -> str:
        """Format skill instructions for prompt injection."""
        if not matches:
            return ""
        blocks = []
        for match in matches:
            instructions = match.skill.prompt_instructions(max_chars=2000)
            blocks.append(
                f"- {match.skill.name} (score={match.score:.3f})\n"
                f"{instructions}"
            )
        return "\n\n".join(blocks)

    def _format_skill_overview(self) -> str:
        """Generate a bullet overview of available project skills."""
        return _format_skill_overview(self.skill_registry)

    def help_short(self) -> str:
        """Return a short, non-expert help string with sample prompts."""
        examples = [
            "basic single-cell QC and clustering",
            "batch integration with harmony on this adata",
            "simple trajectory with DPT, root on paul15_clusters=7MEP, list top genes",
            "find markers for each Leiden cluster (wilcoxon)",
            "doublet check and report rate",
        ]
        return (
            "ov.Agent quick start (use only your provided adata):\n"
            "- " + "\n- ".join(examples) + "\n"
            "Notes: do not create new/dummy AnnData; prefer use_raw=False unless you need raw; "
            "allowed libs: omicverse/matplotlib."
        )

    # =====================================================================
    # Direct-Python seams (overridable by test doubles)
    # =====================================================================

    def _detect_direct_python_request(self, request: str) -> Optional[str]:
        """Detect explicit Python code in *request*.

        Default implementation delegates to the codegen pipeline subsystem.
        Tests may override this at instance level to inject fake behaviour
        without requiring `_codegen_pipeline` / `_analysis_executor` to exist.
        """
        return self._codegen.detect_direct_python_request(request)

    def _execute_generated_code(self, code: str, adata: Any) -> Any:
        """Execute *code* against *adata* and return the (possibly mutated) result.

        Default implementation delegates to the analysis executor subsystem.
        Overridable at instance level for lightweight test doubles.
        """
        return self._analysis_executor.execute_generated_code(code, adata)

    # =====================================================================
    # Legacy thin delegation seams (codegen / tool-runtime / code-only)
    # =====================================================================

    def _extract_python_code(self, response: str) -> str:
        """Delegate to CodegenPipeline.extract_python_code."""
        return self._codegen.extract_python_code(response)

    def _extract_python_code_strict(self, response: str) -> str:
        """Delegate to CodegenPipeline.extract_python_code_strict."""
        return self._codegen.extract_python_code_strict(response)

    def _normalize_code_candidate(self, code: str) -> str:
        """Delegate to CodegenPipeline.normalize_code_candidate."""
        return self._codegen.normalize_code_candidate(code)

    def _gather_code_candidates(self, response: str) -> list:
        """Delegate to CodegenPipeline.gather_code_candidates."""
        return self._codegen.gather_code_candidates(response)

    def _looks_like_python(self, text: str) -> bool:
        """Delegate to CodegenPipeline.looks_like_python."""
        return self._codegen.looks_like_python(text)

    def _get_visible_agent_tools(self, *, allowed_names=None):
        """Delegate to ToolRuntime.get_visible_agent_tools."""
        return self._tool_runtime.get_visible_agent_tools(allowed_names=allowed_names)

    def _tool_blocked_in_plan_mode(self, tool_name: str) -> bool:
        """Delegate to ToolRuntime.tool_blocked_in_plan_mode."""
        return self._tool_runtime.tool_blocked_in_plan_mode(tool_name)

    def _capture_code_only_snippet(self, code: str, description: str = "") -> None:
        """Store code snippet captured during code-only mode."""
        history = getattr(self, "_code_only_captured_history", None)
        if history is None:
            history = []
            self._code_only_captured_history = history
        history.append({"code": code, "description": description or ""})
        self._code_only_captured_code = code

    # =====================================================================
    # Public API: run / stream
    # =====================================================================

    async def run_async(self, request: str, adata: Any) -> Any:
        """
        Process a natural language request using the agentic tool-calling loop.

        Parameters
        ----------
        request : str
            Natural language description of what to do.
        adata : Any
            AnnData/MuData object to process.

        Returns
        -------
        Any
            Processed adata object.
        """
        if adata is not None and hasattr(adata, 'shape'):
            dataset_desc = f"{adata.shape[0]} cells × {adata.shape[1]} genes"
        else:
            dataset_desc = "None (knowledge query)"
        self._emit(EventLevel.INFO, f"Processing request: \"{request}\" | Dataset: {dataset_desc}", "execution")

        # Direct execution path for explicit Python snippets (no LLM required)
        direct_code = self._detect_direct_python_request(request)
        if direct_code:
            self._emit(EventLevel.INFO, "Direct Python detected → executing without model calls", "execution")
            self.last_usage = None
            self.last_usage_breakdown = {
                'generation': None, 'reflection': [], 'review': [], 'total': None
            }
            try:
                result_adata = self._execute_generated_code(direct_code, adata)
                self._emit(EventLevel.SUCCESS, "Python code executed directly.", "execution")
                return result_adata
            except Exception as exc:
                self._emit(EventLevel.ERROR, f"Direct Python execution failed: {exc}", "execution")
                raise

        if self.provider == "python":
            raise ValueError("Python provider requires executable Python code in the request.")

        return await self._run_agentic_mode(request, adata)

    async def _run_agentic_mode(self, request: str, adata: Any) -> Any:
        """Agentic loop mode: LLM autonomously calls tools to complete the task."""
        self._emit(EventLevel.INFO, "Mode: Agentic Loop (tool-calling)", "execution")

        try:
            result = await self._run_agentic_loop(request, adata)
            self._turn_controller._persist_harness_history(request)
            self._emit(EventLevel.SUCCESS, "Agentic loop completed!", "execution")
            return result
        except Exception as e:
            self._turn_controller._persist_harness_history(request)
            self._emit(EventLevel.ERROR, f"Agentic loop failed: {e}", "execution")
            raise

    async def generate_code_async(self, request: str, adata: Any = None, *, max_functions: int = 8, progress_callback: Optional[Callable[[str], None]] = None) -> str:
        """Generate OmicVerse Python code without executing it."""
        return await self._codegen.generate_code_async(
            request, adata, max_functions=max_functions, progress_callback=progress_callback,
        )

    def generate_code(self, request: str, adata: Any = None, *, max_functions: int = 8, progress_callback: Optional[Callable[[str], None]] = None) -> str:
        """Synchronous wrapper for code-only OmicVerse generation."""
        return _run_coroutine_sync(
            self.generate_code_async(
                request, adata, max_functions=max_functions,
                progress_callback=progress_callback,
            )
        )

    async def stream_async(self, request: str, adata: Any,
                           cancel_event=None, history=None,
                           approval_handler=None,
                           request_content=None):
        """
        Stream agentic-loop events as the agent processes a request.

        Wraps ``_run_agentic_loop`` with an event callback so that callers
        can observe tool calls, code execution, results, and completion in
        real time.

        Parameters
        ----------
        request : str
            Natural language description of what to do.
        adata : Any
            AnnData/MuData object to process.
        cancel_event : threading.Event, optional
            When set, signals the agentic loop to stop at the next safe
            checkpoint.
        history : list, optional
            Prior conversation messages for multi-turn context.

        Yields
        ------
        dict
            Dictionary with ``'type'`` and ``'content'`` keys.
        """
        queue: asyncio.Queue = asyncio.Queue()

        async def _event_callback(event):
            await queue.put(event)

        async def _run_loop():
            try:
                await self._run_agentic_loop(
                    request, adata,
                    event_callback=_event_callback,
                    cancel_event=cancel_event,
                    history=history,
                    approval_handler=approval_handler,
                    request_content=request_content,
                )
            except Exception as exc:
                trace_id = getattr(getattr(self, "_last_run_trace", None), "trace_id", "")
                turn_id = getattr(getattr(self, "_last_run_trace", None), "turn_id", "")
                await queue.put(build_stream_event(
                    "error",
                    str(exc),
                    turn_id=turn_id,
                    trace_id=trace_id,
                    session_id=self._get_harness_session_id(),
                    category="runtime",
                ))
                await queue.put(build_stream_event(
                    "done",
                    f"Error: {exc}",
                    turn_id=turn_id,
                    trace_id=trace_id,
                    session_id=self._get_harness_session_id(),
                    category="lifecycle",
                ) | {"error": True})
            finally:
                await queue.put(None)  # sentinel

        task = asyncio.create_task(_run_loop())

        while True:
            event = await queue.get()
            if event is None:
                break
            yield event

        await task

    def run(self, request: str, adata: Any) -> Any:
        """
        Process a natural language request with the provided adata (main method).

        Parameters
        ----------
        request : str
            Natural language description of what to do
        adata : Any
            AnnData object to process

        Returns
        -------
        Any
            Processed adata object (modified)

        Examples
        --------
        >>> agent = ov.Agent(model="gpt-4o-mini")
        >>> result = agent.run("quality control with nUMI>500, mito<0.2", adata)
        """
        return _run_coroutine_sync(self.run_async(request, adata))

    # ===================================================================
    # Cleanup
    # ===================================================================

    def __del__(self):
        """Cleanup on agent deletion.

        Uses broad ``except Exception`` to avoid surfacing teardown noise —
        __del__ runs in unpredictable interpreter states where any subsystem
        may already be partially torn down.
        """
        if hasattr(self, '_notebook_executor') and self._notebook_executor:
            try:
                self._notebook_executor.shutdown()
            except Exception as exc:
                logger.debug("__del__: notebook executor shutdown failed: %s", exc)
        if hasattr(self, '_filesystem_context') and self._filesystem_context:
            try:
                self._filesystem_context.cleanup_session(keep_summary=True)
            except Exception as exc:
                logger.debug("__del__: filesystem context cleanup failed: %s", exc)


# =====================================================================
# Module-level API
# =====================================================================


def list_supported_models(show_all: bool = False) -> str:
    """
    List all supported models for OmicVerse Smart Agent.

    Parameters
    ----------
    show_all : bool, optional
        If True, show all models. If False, show top 3 per provider (default: False)

    Returns
    -------
    str
        Formatted list of supported models with API key status

    Examples
    --------
    >>> import omicverse as ov
    >>> print(ov.list_supported_models())
    >>> print(ov.list_supported_models(show_all=True))
    """
    return ModelConfig.list_supported_models(show_all)

# Canonical default values derived from OmicVerseAgent.__init__ so the
# Agent() factory stays in sync without duplicating them.  Computed once
# at module load time; the dict is keyed by parameter name.
_INIT_DEFAULTS: Dict[str, Any] = {
    name: param.default
    for name, param in inspect.signature(OmicVerseAgent.__init__).parameters.items()
    if name != "self" and param.default is not inspect.Parameter.empty
}
_D = _INIT_DEFAULTS  # short alias used in the Agent() signature below


def Agent(
    model: str = _D["model"],
    api_key: Optional[str] = _D["api_key"],
    endpoint: Optional[str] = _D["endpoint"],
    auth_mode: str = _D["auth_mode"],
    auth_provider: Optional[str] = _D["auth_provider"],
    auth_file: Optional[str] = _D["auth_file"],
    enable_reflection: bool = _D["enable_reflection"],
    reflection_iterations: int = _D["reflection_iterations"],
    enable_result_review: bool = _D["enable_result_review"],
    use_notebook_execution: bool = _D["use_notebook_execution"],
    max_prompts_per_session: int = _D["max_prompts_per_session"],
    notebook_storage_dir: Optional[str] = _D["notebook_storage_dir"],
    keep_execution_notebooks: bool = _D["keep_execution_notebooks"],
    notebook_timeout: int = _D["notebook_timeout"],
    strict_kernel_validation: bool = _D["strict_kernel_validation"],
    enable_filesystem_context: bool = _D["enable_filesystem_context"],
    context_storage_dir: Optional[str] = _D["context_storage_dir"],
    approval_mode: str = _D["approval_mode"],
    agent_mode: str = _D["agent_mode"],
    max_agent_turns: int = _D["max_agent_turns"],
    security_level: Optional[str] = _D["security_level"],
    *,
    config: Optional[AgentConfig] = _D["config"],
    reporter: Optional[Reporter] = _D["reporter"],
    verbose: bool = _D["verbose"],
) -> OmicVerseAgent:
    """Convenience factory — creates an :class:`OmicVerseAgent`.

    Accepts the same parameters as :meth:`OmicVerseAgent.__init__`.
    Defaults are derived from the constructor at import time so they
    cannot drift.
    """
    return OmicVerseAgent(**locals())


__all__ = [
    "Agent",
    "OmicVerseAgent",
    "list_supported_models",
]
