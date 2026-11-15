"""CodegenPipeline — code generation, extraction, review and reflection.

Extracted from ``smart_agent.py``.  ``CodegenPipeline`` owns:

* Code extraction from LLM responses (fenced blocks, inline heuristics)
* Direct-Python request detection
* Code-only agentic-loop orchestration (``generate_code_async``)
* Lightweight review, scanpy rewriting, full reflection
* Codegen prompt building and registry-context formatting
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import re
import textwrap
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from ..agent_backend import OmicVerseLLMBackend, Usage
from ..._registry import _global_registry
from ..skill_registry import SkillMatch, SkillRouter

if TYPE_CHECKING:
    from .protocol import AgentContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CodegenPipeline
# ---------------------------------------------------------------------------


class CodegenPipeline:
    """Code generation, extraction, review and reflection engine.

    Parameters
    ----------
    ctx : AgentContext
        The agent instance (accessed via protocol surface).
    """

    def __init__(self, ctx: "AgentContext") -> None:
        self._ctx = ctx

    # ------------------------------------------------------------------
    # Code extraction helpers (stateless — no LLM calls)
    # ------------------------------------------------------------------

    def extract_python_code(self, response_text: str) -> str:
        """Extract executable Python code from LLM response using AST validation."""
        from .analysis_executor import ProactiveCodeTransformer

        candidates = self.gather_code_candidates(response_text)
        if not candidates:
            error_msg = (
                f"Could not extract executable code: no code candidates found in the response.\n"
                f"Response length: {len(response_text)} characters\n"
                f"Response preview (first 500 chars):\n{response_text[:500]}\n"
                f"Response preview (last 300 chars):\n...{response_text[-300:]}"
            )
            logger.error(error_msg)
            return self._fallback_minimal_workflow()

        logger.debug(f"Found {len(candidates)} code candidate(s) to validate")

        syntax_errors = []
        for i, candidate in enumerate(candidates):
            logger.debug(f"Validating candidate {i+1}/{len(candidates)} (length: {len(candidate)} chars)")
            logger.debug(f"Candidate preview (first 200 chars): {candidate[:200]}")

            try:
                normalized = self.normalize_code_candidate(candidate)
            except ValueError as exc:
                error = f"Candidate {i+1}: normalization failed - {exc}"
                logger.debug(error)
                syntax_errors.append(error)
                continue

            try:
                ast.parse(normalized)
                logger.debug(f"✓ Candidate {i+1} validated successfully")
                transformer = ProactiveCodeTransformer()
                transformed = transformer.transform(normalized)
                if transformed != normalized:
                    logger.debug("✓ Proactive transformations applied to fix potential errors")
                return transformed
            except SyntaxError as exc:
                error = f"Candidate {i+1}: syntax error - {exc}"
                logger.debug(error)
                syntax_errors.append(error)
                continue

        error_msg = (
            f"Could not extract executable code: all {len(candidates)} candidate(s) failed validation.\n"
            f"Errors:\n" + "\n".join(f"  - {err}" for err in syntax_errors)
        )
        logger.error(error_msg)
        return self._fallback_minimal_workflow()

    def extract_python_code_strict(self, response_text: str) -> str:
        """Extract executable Python code without logging errors or falling back."""
        from .analysis_executor import ProactiveCodeTransformer

        candidates = self.gather_code_candidates(response_text)
        if not candidates:
            raise ValueError("no code candidates found")

        syntax_errors: List[str] = []
        for i, candidate in enumerate(candidates, start=1):
            try:
                normalized = self.normalize_code_candidate(candidate)
            except ValueError as exc:
                syntax_errors.append(f"Candidate {i}: normalization failed - {exc}")
                continue

            try:
                ast.parse(normalized)
                transformer = ProactiveCodeTransformer()
                transformed = transformer.transform(normalized)
                ast.parse(transformed)
                return transformed
            except SyntaxError as exc:
                syntax_errors.append(f"Candidate {i}: syntax error - {exc}")
                continue

        raise ValueError(
            "Could not extract executable code: all "
            f"{len(candidates)} candidate(s) failed validation.\nErrors:\n  - "
            + "\n  - ".join(syntax_errors)
        )

    def gather_code_candidates(self, response_text: str) -> List[str]:
        """Enhanced code extraction with multiple strategies to handle various formats."""

        candidates = []

        # Strategy 1: Standard fenced code blocks with python identifier
        fenced_python = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)
        for match in fenced_python.finditer(response_text):
            code = textwrap.dedent(match.group(1)).strip()
            if code:
                candidates.append(code)

        # Strategy 2: Generic fenced code blocks (```...```)
        if not candidates:
            fenced_generic = re.compile(r"```\s*(.*?)```", re.DOTALL)
            for match in fenced_generic.finditer(response_text):
                code = textwrap.dedent(match.group(1)).strip()
                first_line = code.split('\n')[0].strip().lower()
                if first_line in ['bash', 'shell', 'json', 'yaml', 'xml', 'html', 'css', 'javascript']:
                    continue
                if code and self.looks_like_python(code):
                    candidates.append(code)

        # Strategy 3: Code blocks with alternative language identifiers (py, python3)
        if not candidates:
            fenced_alt = re.compile(r"```(?:py|python3)\s*(.*?)```", re.DOTALL | re.IGNORECASE)
            for match in fenced_alt.finditer(response_text):
                code = textwrap.dedent(match.group(1)).strip()
                if code:
                    candidates.append(code)

        # Strategy 4: Code following "Here's the code:" or similar phrases
        if not candidates:
            code_intro = re.compile(
                r"(?:here'?s? (?:the )?code|code:|solution:)\s*[:\n]\s*```(?:python)?\s*(.*?)```",
                re.DOTALL | re.IGNORECASE
            )
            for match in code_intro.finditer(response_text):
                code = textwrap.dedent(match.group(1)).strip()
                if code:
                    candidates.append(code)

        # Strategy 5: prefer the last code block when reasoning precedes final code
        if len(candidates) > 1:
            candidates = list(reversed(candidates))

        # Strategy 6: Inline extraction as fallback
        if not candidates:
            inline = self.extract_inline_python(response_text)
            if inline:
                candidates.append(inline)

        return candidates

    @staticmethod
    def looks_like_python(code: str) -> bool:
        """Heuristic check if code snippet looks like Python."""

        python_indicators = [
            r'\bimport\b',
            r'\bdef\b',
            r'\bclass\b',
            r'\badata\b',
            r'\bov\.',
            r'\bsc\.',
            r'\breturn\b',
            r'\bfor\b.*\bin\b',
            r'\bif\b.*:',
            r'\.obs\[',
            r'\.var\[',
            r'=\s*\w+\(',
        ]

        matches = sum(1 for pattern in python_indicators if re.search(pattern, code))
        return matches >= 2

    @staticmethod
    def extract_inline_python(response_text: str) -> str:
        """Heuristically gather inline Python statements for AST validation."""

        python_line_pattern = re.compile(
            r"^\s*(?:async\s+def |def |class |import |from |for |while |if |elif |else:|try:|except |with |return |@|print|adata|ov\.|sc\.)"
        )
        assignment_pattern = re.compile(r"^\s*[\w\.]+\s*=.*")
        call_pattern = re.compile(r"^\s*[\w\.]+\s*\(.*")
        collected: List[str] = []

        for raw_line in response_text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                continue
            if (
                python_line_pattern.match(line)
                or assignment_pattern.match(line)
                or call_pattern.match(line)
                or stripped.startswith("#")
            ):
                collected.append(line)

        snippet = "\n".join(collected).strip()
        return textwrap.dedent(snippet) if snippet else ""

    @staticmethod
    def normalize_code_candidate(code: str) -> str:
        """Ensure imports and formatting are in place for execution."""

        dedented = textwrap.dedent(code).strip()
        if not dedented:
            raise ValueError("empty code candidate")

        import_present = re.search(r"^\s*(?:import|from)\s+omicverse", dedented, re.MULTILINE)
        if not import_present:
            dedented = "import omicverse as ov\n" + dedented

        return dedented

    @staticmethod
    def _fallback_minimal_workflow() -> str:
        """Return a minimal safe workflow when code extraction fails."""
        return textwrap.dedent(
            """
            import omicverse as ov
            # Fallback minimal workflow when code extraction fails
            ov.pp.normalize_total(adata, target_sum=1e4)
            ov.pp.log1p(adata)
            ov.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat')
            ov.pp.pca(adata)
            ov.pp.neighbors(adata)
            try:
                ov.pp.leiden(adata)
            except Exception:
                pass
            try:
                ov.pp.umap(adata)
            except Exception:
                pass
            """
        ).strip()

    # ------------------------------------------------------------------
    # Direct-Python detection
    # ------------------------------------------------------------------

    def detect_direct_python_request(self, request: str) -> Optional[str]:
        """Detect and return user-provided Python code to execute directly."""
        trimmed = (request or "").strip()
        if not trimmed:
            return None

        python_markers = (
            "```",
            "import ",
            "from ",
            "def ",
            "class ",
            "adata",
            "ov.",
            "sc.",
            "pd.",
            "np.",
        )

        if self._ctx.provider != "python" and not any(marker in trimmed for marker in python_markers):
            return None

        candidates = self.gather_code_candidates(trimmed)
        if not candidates and self._ctx.provider == "python":
            candidates = [trimmed]

        for candidate in candidates:
            try:
                normalized = self.normalize_code_candidate(candidate)
            except ValueError:
                continue
            try:
                ast.parse(normalized)
                return normalized
            except SyntaxError:
                continue

        return None

    # ------------------------------------------------------------------
    # Code-only mode helpers
    # ------------------------------------------------------------------

    def capture_code_only_snippet(self, code: str, description: str = "") -> None:
        """Store the latest code snippet captured from execute_code in code-only mode."""

        history = getattr(self._ctx, "_code_only_captured_history", None)
        if history is None:
            history = []
            self._ctx._code_only_captured_history = history
        history.append({
            "code": code,
            "description": description or "",
        })
        self._ctx._code_only_captured_code = code

    @staticmethod
    def build_code_only_agentic_request(request: str, adata: Any) -> str:
        """Wrap a raw claw request so the normal agentic loop produces code instead of running it."""

        dataset_hint = (
            "A live dataset object is available as `adata`. Reuse the normal Jarvis workflow, "
            "but stop at code generation."
            if adata is not None
            else "No live dataset object is available. Unless the user explicitly asks to load data, "
            "assume an AnnData object named `adata` already exists in the generated code."
        )
        return (
            f"{request}\n\n"
            "CLAW REQUEST MODE:\n"
            "- Use the same OmicVerse Agent logic as Jarvis.\n"
            "- Use search_functions and search_skills when helpful.\n"
            "- Produce the final answer by calling execute_code with the final Python script.\n"
            "- In this mode, execute_code captures code without running it.\n"
            "- After execute_code, call finish.\n"
            f"{dataset_hint}"
        )

    async def generate_code_via_agentic_loop(
        self,
        request: str,
        adata: Any,
        *,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Run the normal Jarvis agentic loop and capture the final execute_code snippet."""

        captured_chunks: List[str] = []
        captured_errors: List[str] = []

        def _progress(message: str) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(message)
            except Exception as exc:
                logger.debug("Progress callback failed (best-effort): %s", exc)

        async def _event_callback(event: Dict[str, Any]) -> None:
            event_type = str(event.get("type") or "")
            content = event.get("content")
            if event_type == "tool_call":
                tool_name = ""
                if isinstance(content, dict):
                    tool_name = str(content.get("name") or "")
                _progress(f"tool: {tool_name or 'unknown'}")
            elif event_type == "code":
                code = str(content or "")
                if code:
                    captured_chunks.append(code)
                _progress("captured code")
            elif event_type == "error":
                captured_errors.append(str(content or "unknown error"))
                _progress("agent error")
            elif event_type == "done":
                _progress("finish")

        previous_mode = getattr(self._ctx, "_code_only_mode", False)
        previous_code = getattr(self._ctx, "_code_only_captured_code", "")
        previous_history = getattr(self._ctx, "_code_only_captured_history", None)
        self._ctx._code_only_mode = True
        self._ctx._code_only_captured_code = ""
        self._ctx._code_only_captured_history = []
        try:
            _progress("start agentic loop")
            await self._ctx._run_agentic_loop(
                self.build_code_only_agentic_request(request, adata),
                adata,
                event_callback=_event_callback,
            )
        finally:
            captured_code = str(getattr(self._ctx, "_code_only_captured_code", "") or "")
            captured_history = list(getattr(self._ctx, "_code_only_captured_history", []) or [])
            self._ctx._code_only_mode = previous_mode
            self._ctx._code_only_captured_code = previous_code
            self._ctx._code_only_captured_history = previous_history

        if captured_code:
            return captured_code

        for item in reversed(captured_history):
            code = str(item.get("code", "") or "")
            if code:
                return code

        for chunk in reversed(captured_chunks):
            try:
                return self.extract_python_code_strict(chunk)
            except ValueError:
                continue

        if captured_errors:
            raise RuntimeError(
                "Jarvis-style code generation did not reach execute_code. "
                + "; ".join(captured_errors[:3])
            )
        raise RuntimeError(
            "Jarvis-style code generation did not emit executable code. "
            "The agent finished without calling execute_code."
        )

    async def generate_code_async(
        self,
        request: str,
        adata: Any = None,
        *,
        max_functions: int = 8,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate OmicVerse Python code without executing it."""

        if not request or not request.strip():
            raise ValueError("request cannot be empty")

        direct_code = self.detect_direct_python_request(request)
        if direct_code:
            self._ctx.last_usage = None
            self._ctx.last_usage_breakdown = {
                'generation': None,
                'reflection': [],
                'review': [],
                'total': None,
            }
            return direct_code

        code = await self.generate_code_via_agentic_loop(
            request,
            adata,
            progress_callback=progress_callback,
        )
        return code

    def generate_code(
        self,
        request: str,
        adata: Any = None,
        *,
        max_functions: int = 8,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Synchronous wrapper for code-only OmicVerse generation."""

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            result_container: Dict[str, Any] = {}
            error_container: Dict[str, BaseException] = {}

            def _run_in_thread() -> None:
                try:
                    result_container["value"] = asyncio.run(
                        self.generate_code_async(
                            request,
                            adata,
                            max_functions=max_functions,
                            progress_callback=progress_callback,
                        )
                    )
                except BaseException as exc:
                    error_container["error"] = exc

            thread = threading.Thread(target=_run_in_thread, name="OmicVerseCodegenRunner")
            thread.start()
            thread.join()

            if "error" in error_container:
                raise error_container["error"]

            return result_container.get("value", "")

        return asyncio.run(
            self.generate_code_async(
                request,
                adata,
                max_functions=max_functions,
                progress_callback=progress_callback,
            )
        )

    # ------------------------------------------------------------------
    # Scanpy rewriting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def contains_forbidden_scanpy_usage(code: str) -> bool:
        """Disallow raw scanpy usage in registry-first claw generation."""

        if not code:
            return False
        patterns = [
            r"^\s*import\s+scanpy\s+as\s+sc\b",
            r"^\s*from\s+scanpy\b",
            r"\bsc\.",
        ]
        return any(re.search(pattern, code, re.MULTILINE) for pattern in patterns)

    def rewrite_scanpy_calls_with_registry(
        self,
        code: str,
        entries: List[Dict[str, Any]],
    ) -> str:
        """Best-effort mechanical rewrite from scanpy-style calls to ov.* calls."""

        if not code:
            return code

        from .registry_scanner import RegistryScanner

        lookup: Dict[str, str] = {}
        for raw_entry in entries:
            entry = RegistryScanner.normalize_entry(raw_entry)
            public_name = str(entry.get("full_name", "") or "")
            short_name = str(entry.get("short_name") or entry.get("name") or "").strip()
            if not public_name.startswith("ov.") or not short_name:
                continue
            lookup.setdefault(short_name, public_name)
            for alias in entry.get("aliases", []) or []:
                alias_key = str(alias).strip().split(".")[-1]
                if alias_key:
                    lookup.setdefault(alias_key, public_name)

        rewritten = re.sub(r"^\s*import\s+scanpy\s+as\s+sc\s*$", "", code, flags=re.MULTILINE)
        rewritten = re.sub(r"^\s*from\s+scanpy\b.*$", "", rewritten, flags=re.MULTILINE)

        def _replace(match: re.Match[str]) -> str:
            func_name = match.group(1)
            replacement = lookup.get(func_name)
            if replacement:
                return replacement + "("
            return match.group(0)

        rewritten = re.sub(r"\bsc\.(?:pp|tl|pl)\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", _replace, rewritten)
        rewritten = re.sub(r"\n{3,}", "\n\n", rewritten).strip()
        return rewritten

    # ------------------------------------------------------------------
    # Codegen prompt building
    # ------------------------------------------------------------------

    def build_code_generation_system_prompt(self, adata: Any) -> str:
        """Build the code-only prompt on top of the fully initialized Agent prompt."""

        base_prompt = ""
        if self._ctx._llm is not None and getattr(self._ctx._llm, "config", None) is not None:
            base_prompt = getattr(self._ctx._llm.config, "system_prompt", "") or ""
        if not base_prompt:
            base_prompt = self._ctx._build_agentic_system_prompt()

        dataset_line = (
            f"Dataset summary: {adata.shape[0]} cells x {adata.shape[1]} features."
            if adata is not None and hasattr(adata, "shape")
            else "No dataset object was provided."
        )

        return (
            f"{base_prompt}\n\n"
            "## Claw Code-Only Mode\n"
            "Return ONLY executable Python code for the user's request.\n"
            "Do not explain the code. Do not describe future actions. Do not execute tools.\n"
            "Use OmicVerse APIs (`import omicverse as ov`) for all analysis steps.\n"
            "Assume every needed Scanpy-style step has an OmicVerse `ov.*` wrapper available.\n"
            "Do NOT import scanpy and do NOT call `sc.*` anywhere in the output.\n"
            "If no dataset object is provided, assume an AnnData object named `adata` already exists unless the user explicitly asks to load data from disk.\n"
            "When using in-place OmicVerse preprocessing functions, call them without assigning their return value.\n"
            "Include imports that are actually needed.\n"
            "Produce a single coherent snippet, not multiple alternatives.\n"
            f"{dataset_line}\n"
        )

    @staticmethod
    def build_code_generation_user_prompt(request: str, adata: Any) -> str:
        """Build the lightweight user prompt for code-only generation."""

        dataset_hint = (
            "A live dataset object is available as `adata`."
            if adata is not None
            else "No live dataset object is available. Generate reusable code that assumes `adata` already exists."
        )
        return (
            f"User request: {request}\n\n"
            f"{dataset_hint}\n"
            "Return Python code only."
        )

    # ------------------------------------------------------------------
    # Skill / registry context for codegen
    # ------------------------------------------------------------------

    def select_codegen_skill_matches(self, request: str, top_k: int = 2) -> List[SkillMatch]:
        """Select skills for codegen using the same loaded skill registry as Jarvis."""

        if not self._ctx.skill_registry or not self._ctx.skill_registry.skill_metadata:
            return []

        try:
            router = SkillRouter(self._ctx.skill_registry, min_score=0.1)
            return router.route(request, top_k=top_k)
        except Exception as exc:
            logger.warning("Skill routing failed for codegen: %s", exc)
            return []

    def format_registry_context_for_codegen(
        self,
        entries: List[Dict[str, Any]],
    ) -> str:
        """Format a compact registry snippet for code-only generation."""

        if not entries:
            return "No highly relevant registry matches were found. Use the best OmicVerse API you know."

        blocks: List[str] = []
        for entry in entries:
            full_name = entry.get("full_name", "")
            signature = entry.get("signature", "")
            description = entry.get("description", "")
            examples = entry.get("examples", [])[:2]

            block = [
                f"Function: {full_name}",
                f"Signature: {signature}",
                f"Description: {description}",
            ]
            if examples:
                block.append("Examples:")
                for example in examples:
                    block.append(f"- {example}")

            prereq_text = self.format_prerequisites_for_codegen_entry(entry)
            if prereq_text:
                block.append("Prerequisites:")
                block.append(prereq_text)

            blocks.append("\n".join(block))

        return "\n\n".join(blocks)

    @staticmethod
    def format_prerequisites_for_codegen_entry(entry: Dict[str, Any]) -> str:
        """Format prerequisites from runtime registry or static AST metadata."""

        full_name = entry.get("registry_full_name", entry.get("full_name", ""))
        if getattr(_global_registry, "_registry", None):
            prereq_text = _global_registry.format_prerequisites_for_llm(full_name)
            if prereq_text and "not found" not in prereq_text.lower():
                return prereq_text

        prerequisites = entry.get("prerequisites", {}) or {}
        requires = entry.get("requires", {}) or {}
        produces = entry.get("produces", {}) or {}

        parts: List[str] = []
        required_functions = prerequisites.get("required", []) or prerequisites.get("functions", []) or []
        optional_functions = prerequisites.get("optional", []) or []

        if required_functions:
            parts.append("  - Required functions: " + ", ".join(required_functions))
        if optional_functions:
            parts.append("  - Optional functions: " + ", ".join(optional_functions))

        req_items = []
        for key, values in requires.items():
            for value in values:
                req_items.append(f"adata.{key}['{value}']")
        if req_items:
            parts.append("  - Requires: " + ", ".join(req_items))

        prod_items = []
        for key, values in produces.items():
            for value in values:
                prod_items.append(f"adata.{key}['{value}']")
        if prod_items:
            parts.append("  - Produces: " + ", ".join(prod_items))

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Review and reflection (LLM-based)
    # ------------------------------------------------------------------

    async def review_generated_code_lightweight(
        self,
        code: str,
        request: str,
        adata: Any,
    ) -> Tuple[str, Optional[Usage]]:
        """Run a lightweight reflection pass and return improved code if possible."""

        dataset_line = (
            f"Dataset summary: {adata.shape[0]} cells x {adata.shape[1]} features."
            if adata is not None and hasattr(adata, "shape")
            else "No dataset object is available."
        )

        backend = OmicVerseLLMBackend(
            system_prompt=(
                "You are a strict reviewer of OmicVerse Python code.\n"
                "Return ONLY corrected executable Python code.\n"
                "Do not add explanations.\n"
            ),
            model=self._ctx.model,
            api_key=self._ctx.api_key,
            endpoint=self._ctx.endpoint,
            max_tokens=4096,
            temperature=0.0,
        )

        prompt = (
            f"User request: {request}\n"
            f"{dataset_line}\n\n"
            "Review the code below for correctness, OmicVerse API misuse, bad assumptions, "
            "and obvious syntax/runtime issues. Keep it concise and executable.\n\n"
            f"```python\n{code}\n```"
        )

        with self._ctx._temporary_api_keys():
            response = await backend.run(prompt)

        try:
            return self.extract_python_code_strict(response), backend.last_usage
        except ValueError:
            return code, backend.last_usage

    async def rewrite_code_without_scanpy(
        self,
        code: str,
        request: str,
        adata: Any,
        registry_context: str = "",
        skill_guidance: str = "",
    ) -> Tuple[str, Optional[Usage]]:
        """Rewrite code to strict OmicVerse-only style when scanpy slips in."""

        backend = OmicVerseLLMBackend(
            system_prompt=(
                "You rewrite Python snippets to use OmicVerse APIs only.\n"
                "Return ONLY executable Python code.\n"
                "Do not import scanpy. Do not call sc.*.\n"
            ),
            model=self._ctx.model,
            api_key=self._ctx.api_key,
            endpoint=self._ctx.endpoint,
            max_tokens=4096,
            temperature=0.0,
        )

        dataset_line = (
            f"Dataset summary: {adata.shape[0]} cells x {adata.shape[1]} features."
            if adata is not None and hasattr(adata, "shape")
            else "No dataset object is available."
        )

        prompt = (
            f"User request: {request}\n"
            f"{dataset_line}\n\n"
            "Rewrite the code below so that all analysis calls use `ov.*` APIs only.\n"
            "Keep the behavior aligned with the initialized OmicVerse Agent prompt.\n"
            f"```python\n{code}\n```"
        )

        with self._ctx._temporary_api_keys():
            response = await backend.run(prompt)

        try:
            rewritten = self.extract_python_code_strict(response)
        except ValueError:
            return code, backend.last_usage
        return rewritten, backend.last_usage

    async def review_result(
        self,
        original_adata: Any,
        result_adata: Any,
        request: str,
        code: str,
    ) -> Dict[str, Any]:
        """Review execution result to validate it matches the user's task assignment."""

        original_shape = (original_adata.shape[0], original_adata.shape[1])
        result_shape = (result_adata.shape[0], result_adata.shape[1])

        original_obs_cols = list(getattr(original_adata, 'obs', {}).columns) if hasattr(original_adata, 'obs') else []
        result_obs_cols = list(getattr(result_adata, 'obs', {}).columns) if hasattr(result_adata, 'obs') else []
        new_obs_cols = [col for col in result_obs_cols if col not in original_obs_cols]

        original_uns_keys = list(getattr(original_adata, 'uns', {}).keys()) if hasattr(original_adata, 'uns') else []
        result_uns_keys = list(getattr(result_adata, 'uns', {}).keys()) if hasattr(result_adata, 'uns') else []
        new_uns_keys = [key for key in result_uns_keys if key not in original_uns_keys]

        review_prompt = f"""You are an expert bioinformatics analyst reviewing the results of an OmicVerse operation.

User Request: "{request}"

Executed Code:
```python
{code}
```

Original Data:
- Shape: {original_shape[0]} cells × {original_shape[1]} genes
- Observation columns: {len(original_obs_cols)} columns
- Uns keys: {len(original_uns_keys)} keys

Result Data:
- Shape: {result_shape[0]} cells × {result_shape[1]} genes
- Observation columns: {len(result_obs_cols)} columns (new: {new_obs_cols if new_obs_cols else 'none'})
- Uns keys: {len(result_uns_keys)} keys (new: {new_uns_keys if new_uns_keys else 'none'})

Changes Detected:
- Cells: {original_shape[0]} → {result_shape[0]} (change: {result_shape[0] - original_shape[0]:+d})
- Genes: {original_shape[1]} → {result_shape[1]} (change: {result_shape[1] - original_shape[1]:+d})
- New observation columns: {new_obs_cols if new_obs_cols else 'none'}
- New uns keys: {new_uns_keys if new_uns_keys else 'none'}

Your task:
1. **Evaluate if the result matches the user's intent**:
   - Does the transformation align with the request?
   - Are the changes expected for this operation?
   - Is the data integrity maintained?

2. **Identify any issues or concerns**:
   - Unexpected data loss (too many cells/genes filtered)
   - Missing expected outputs
   - Suspicious transformations

3. **Provide assessment as JSON**:
{{
  "matched": true,
  "assessment": "Brief assessment of the result quality",
  "changes_detected": ["change 1", "change 2"],
  "issues": ["issue 1"] or [],
  "confidence": 0.92,
  "recommendation": "accept"
}}

Recommendation values:
- "accept": Result looks good, matches intent
- "review": Result may have issues, user should review
- "retry": Result appears incorrect, suggest retry

IMPORTANT:
- Return ONLY the JSON object
- Keep confidence between 0.0 and 1.0
- Be specific about changes and issues
- Consider the context of the user's request
"""

        try:
            with self._ctx._temporary_api_keys():
                if not self._ctx._llm:
                    raise RuntimeError("LLM backend is not initialized")

                response_text = await self._ctx._llm.run(review_prompt)

                if self._ctx._llm.last_usage:
                    if 'review' not in self._ctx.last_usage_breakdown:
                        self._ctx.last_usage_breakdown['review'] = []
                    self._ctx.last_usage_breakdown['review'].append(self._ctx._llm.last_usage)

            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                return {
                    'matched': True,
                    'assessment': 'Result review completed (JSON extraction failed)',
                    'changes_detected': [f'Shape changed: {original_shape} → {result_shape}'],
                    'issues': [],
                    'confidence': 0.7,
                    'recommendation': 'accept'
                }

            review_result = json.loads(json_match.group(0))

            result = {
                'matched': bool(review_result.get('matched', True)),
                'assessment': review_result.get('assessment', 'No assessment provided'),
                'changes_detected': review_result.get('changes_detected', []),
                'issues': review_result.get('issues', []),
                'confidence': max(0.0, min(1.0, float(review_result.get('confidence', 0.8)))),
                'recommendation': review_result.get('recommendation', 'accept')
            }

            return result

        except Exception as exc:
            logger.warning(f"Result review failed: {exc}")
            return {
                'matched': True,
                'assessment': f'Result review failed: {exc}',
                'changes_detected': [f'Shape: {original_shape} → {result_shape}'],
                'issues': [],
                'confidence': 0.6,
                'recommendation': 'review'
            }

    async def reflect_on_code(
        self,
        code: str,
        request: str,
        adata: Any,
        iteration: int = 1,
    ) -> Dict[str, Any]:
        """Reflect on generated code to identify issues and improvements."""

        reflection_prompt = f"""You are a code reviewer for OmicVerse bioinformatics code.

Original User Request: "{request}"

Generated Code (Iteration {iteration}):
```python
{code}
```

Dataset Information:
{f"- Shape: {adata.shape[0]} cells × {adata.shape[1]} genes" if adata is not None and hasattr(adata, 'shape') else "- No dataset provided (knowledge query mode)"}

Your task is to review this code and provide feedback:

1. **Check for correctness**:
   - Are the function calls correct?
   - Are parameters properly formatted (especially dict parameters like 'tresh')?
   - Are there any syntax errors?
   - Does the code match the user's request?

2. **Common issues to check**:
   - Missing or incorrect imports
   - Wrong parameter types or values
   - Incorrect function selection
   - Parameter extraction errors (e.g., nUMI>500 should map to correct parameter)
   - Missing required parameters
   - Using wrong parameter names

3. **CRITICAL VALIDATION CHECKLIST** (These cause frequent errors!):

   **Parameter Name Validation:**
   - pySCSA.cell_auto_anno() uses `clustertype='leiden'`, NOT `cluster='leiden'`!
   - COSG/rank_genes uses `groupby='leiden'`, NOT `cluster='leiden'`
   - These are DIFFERENT parameters with DIFFERENT meanings!

   **Output Storage Validation:**
   - Cell annotations → stored in `adata.obs['column_name']`
   - Marker gene results (COSG, rank_genes_groups) → stored in `adata.uns['key']`
   - COSG does NOT create `adata.obs['cosg_celltype']` - it stores results in `adata.uns['rank_genes_groups']`!

   **Pandas/DataFrame Pitfalls:**
   - DataFrame uses `.dtypes` (PLURAL) for all column types
   - Series uses `.dtype` (SINGULAR) for single column type
   - `df.dtype` will cause AttributeError - use `df.dtypes` instead!

   **Batch Column Validation:**
   - Before batch operations, check if batch column exists and has no NaN values
   - Use `adata.obs['batch'].fillna('unknown')` to handle missing values

   **Geneset Enrichment:**
   - `pathways_dict` must be a dictionary loaded via `ov.utils.geneset_prepare()`, NOT a file path string!
   - WRONG: `ov.bulk.geneset_enrichment(gene_list, pathways_dict='file.gmt')`
   - CORRECT: First load with `pathways_dict = ov.utils.geneset_prepare('file.gmt')`, then pass dict

   **HVG (Highly Variable Genes) - Small Dataset Pitfalls:**
   - `flavor='seurat_v3'` uses LOESS regression which FAILS on:
     - Small batches (<500 cells per batch)
     - Log-normalized data (expects raw counts)
   - Error message: "ValueError: Extrapolation not allowed with blending"
   - ALWAYS wrap HVG in try/except with fallback:
   ```python
   try:
       ov.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=2000)
   except ValueError:
       ov.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=2000)
   ```
   - For batch-aware HVG with small batches, prefer `flavor='cell_ranger'` or `flavor='seurat'`

   **In-Place Function Pitfalls:**
   - OmicVerse preprocessing functions operate IN-PLACE by default!
   - Functions: `ov.pp.pca()`, `ov.pp.scale()`, `ov.pp.neighbors()`, `ov.pp.leiden()`, `ov.pp.umap()`, `ov.pp.tsne()`
   - WRONG: `adata = ov.pp.pca(adata)` → returns None, adata becomes None!
   - CORRECT: `ov.pp.pca(adata)` (call without assignment)
   - Alternative: `adata = ov.pp.pca(adata, copy=True)` (explicit copy)
   - Same pattern for `ov.pp.scale()`, `ov.pp.neighbors()`, `ov.pp.umap()`, etc.

   **Print Statement Pitfalls:**
   - NEVER use f-strings in print statements - they cause format errors with special characters
   - WRONG: `print(f"Value: {{val:.2%}}")` → format code errors
   - CORRECT: `print("Value: " + str(round(val * 100, 2)) + "%")`
   - ALWAYS use string concatenation with str() for print statements

   **Categorical Column Access Pitfalls:**
   - NEVER assume a column is categorical - it may be string/object dtype
   - WRONG: `adata.obs['leiden'].cat.categories` → AttributeError if not categorical
   - CORRECT: `adata.obs['leiden'].value_counts()` (works for any dtype)
   - If you MUST access categories: `if hasattr(adata.obs['col'], 'cat'): ...`

   **AUTOMATIC FIXES REQUIRED** (You MUST apply these fixes if found):
   - If code has f-strings in print() → Convert to string concatenation
   - If code has `adata = ov.pp.func(adata)` → Remove the assignment
   - If code has `.cat.categories` without check → Add hasattr() guard or use value_counts()
   - If code has HVG without try/except → Add seurat fallback wrapper
   - If code has batch operations without validation → Add fillna('unknown') guard

4. **Provide feedback as a JSON object**:
{{
  "issues_found": ["specific issue 1", "specific issue 2"],
  "needs_revision": true,
  "confidence": 0.85,
  "improved_code": "the corrected code here",
  "explanation": "brief explanation of what was fixed"
}}

If no issues are found:
{{
  "issues_found": [],
  "needs_revision": false,
  "confidence": 0.95,
  "improved_code": "{code}",
  "explanation": "Code looks correct"
}}

IMPORTANT:
- Return ONLY the JSON object, nothing else
- Keep confidence between 0.0 and 1.0
- If you fix the code, put the complete corrected code in 'improved_code'
- Be specific about issues found
"""

        try:
            with self._ctx._temporary_api_keys():
                if not self._ctx._llm:
                    raise RuntimeError("LLM backend is not initialized")

                response_text = await self._ctx._llm.run(reflection_prompt)

                if self._ctx._llm.last_usage:
                    self._ctx.last_usage_breakdown['reflection'].append(self._ctx._llm.last_usage)

            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                return {
                    'improved_code': code,
                    'issues_found': [],
                    'confidence': 0.8,
                    'needs_revision': False,
                    'explanation': 'Reflection completed (JSON extraction failed, assuming code is OK)'
                }

            reflection_result = json.loads(json_match.group(0))

            result = {
                'improved_code': reflection_result.get('improved_code', code),
                'issues_found': reflection_result.get('issues_found', []),
                'confidence': max(0.0, min(1.0, float(reflection_result.get('confidence', 0.8)))),
                'needs_revision': bool(reflection_result.get('needs_revision', False)),
                'explanation': reflection_result.get('explanation', 'No explanation provided')
            }

            return result

        except Exception as exc:
            logger.warning(f"Reflection failed: {exc}")
            return {
                'improved_code': code,
                'issues_found': [],
                'confidence': 0.7,
                'needs_revision': False,
                'explanation': f'Reflection failed: {exc}'
            }
