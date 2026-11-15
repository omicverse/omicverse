"""Prompt template engine — composable prompt construction for OVAgent.

Breaks monolithic prompt strings into reusable template blocks and
deterministic overlay composition.  The engine guarantees that the same
set of overlays always renders in the same order.

Prompt composition contract
---------------------------
* A **base template** defines the agent identity and core behaviour.
* **Overlays** add workflow, skill, provider, and runtime context.
* Overlays are sorted by ``(priority, name)`` before concatenation,
  so rendering is deterministic regardless of registration order.
* ``PromptBuilder`` delegates to this engine internally; callers that
  only need the builder API do not need to touch the engine directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# PromptOverlay — a named, prioritized text block
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptOverlay:
    """A named text block composable into a rendered prompt.

    Parameters
    ----------
    name : str
        Unique key used for replacement and ordering.
    content : str
        The text content of this overlay.
    priority : int
        Lower values render earlier.  Default 100.
    """

    name: str
    content: str
    priority: int = 100


# ---------------------------------------------------------------------------
# PromptTemplateEngine — deterministic composition
# ---------------------------------------------------------------------------


class PromptTemplateEngine:
    """Compose a prompt from a base template plus ordered overlays.

    Rendering is deterministic: overlays are sorted by
    ``(priority, name)`` and concatenated onto the base template
    separated by blank lines.

    Examples
    --------
    >>> engine = PromptTemplateEngine()
    >>> engine.set_base("You are an assistant.")
    >>> engine.add_overlay(PromptOverlay("tools", "Use tools.", priority=10))
    >>> engine.add_overlay(PromptOverlay("skills", "Domain skills.", priority=20))
    >>> print(engine.render())
    You are an assistant.
    <BLANKLINE>
    Use tools.
    <BLANKLINE>
    Domain skills.
    """

    def __init__(self) -> None:
        self._base: str = ""
        self._overlays: Dict[str, PromptOverlay] = {}

    # -- base ---------------------------------------------------------------

    def set_base(self, template: str) -> None:
        """Set the base system template."""
        self._base = template

    @property
    def base(self) -> str:
        """Return the current base template."""
        return self._base

    # -- overlay management -------------------------------------------------

    def add_overlay(self, overlay: PromptOverlay) -> None:
        """Register an overlay.  Replaces any existing with the same name."""
        self._overlays[overlay.name] = overlay

    def remove_overlay(self, name: str) -> None:
        """Remove an overlay by name (no-op if absent)."""
        self._overlays.pop(name, None)

    def has_overlay(self, name: str) -> bool:
        """Return whether an overlay with *name* is registered."""
        return name in self._overlays

    def get_overlay(self, name: str) -> Optional[PromptOverlay]:
        """Return the overlay with *name*, or ``None``."""
        return self._overlays.get(name)

    @property
    def overlay_names(self) -> List[str]:
        """Return overlay names in render order."""
        return [
            o.name
            for o in sorted(
                self._overlays.values(), key=lambda o: (o.priority, o.name)
            )
        ]

    # -- rendering ----------------------------------------------------------

    def render(self) -> str:
        """Render the complete prompt: base + sorted overlays.

        Returns
        -------
        str
            Concatenation of non-empty parts separated by ``\\n\\n``.
        """
        parts: List[str] = []
        if self._base:
            parts.append(self._base)
        for overlay in sorted(
            self._overlays.values(), key=lambda o: (o.priority, o.name)
        ):
            if overlay.content:
                parts.append(overlay.content)
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Predefined template blocks — extracted from PromptBuilder string literals
# ---------------------------------------------------------------------------

# Priority bands:
#   10  identity / base
#   20  tool catalog / coding instructions
#   30  web access
#   40  workflow steps
#   50  code quality rules
#   60  guidelines
#   70  delegation strategy
#   80  code-only mode (conditional)
#   90  skill listing (dynamic)
#  100  provider / runtime overlay (dynamic)

BASE_IDENTITY = (
    "You are OmicVerse Agent, an expert bioinformatics assistant that processes "
    "single-cell, bulk RNA-seq, and spatial transcriptomics data.\n\n"
    "You have OmicVerse data tools plus a Claude-style coding tool catalog.\n"
    "Core Claude tools are ToolSearch, Bash, and Read. Deferred Claude tools "
    "must be loaded through ToolSearch before use. High-risk tools require approval.\n\n"
    "Use the legacy OmicVerse tools for dataset inspection and execution, and "
    "use Claude-style tools for files, tasks, planning, worktrees, and questions.\n\n"
    "Do not narrate future actions without taking them in the same turn. "
    "If a tool is needed, call it now instead of saying that you will call it later."
)

TOOL_INSTRUCTIONS = (
    "CODING / FILESYSTEM:\n"
    "- Use ToolSearch when you need deferred tools such as Edit, Write, Glob, "
    "Grep, NotebookEdit, Task*, or EnterWorktree\n"
    "- Use Read for local files instead of shelling out to cat/head/sed when possible\n"
    "- Use AskUserQuestion when a turn genuinely needs user clarification"
)

WEB_ACCESS = (
    "WEB ACCESS:\n"
    "- Use WebFetch(url) or web_fetch(url) to read any web page "
    "(GEO datasets, papers, docs)\n"
    "- Use WebSearch(query) or web_search(query) to search the internet "
    "for information\n"
    "- Use web_download(url) to download files (datasets, h5ad, csv, gz) to disk\n"
    "- When the user provides a URL, ALWAYS fetch it first instead of "
    "asking them to paste content\n"
    "- When you need background info (gene functions, diseases, methods), "
    "search the web\n"
    "- To download GEO data, fetch the GEO page first to find download links, "
    "then use web_download"
)

WORKFLOW_STEPS = (
    "WORKFLOW:\n"
    "1. If the user provides a URL, use web_fetch to read it\n"
    "2. Use inspect_data to understand the dataset structure\n"
    "3. Use search_functions to find relevant OmicVerse functions "
    "(includes prerequisite chains and examples)\n"
    "4. For complex multi-step workflows, use search_skills for domain guidance\n"
    "5. Execute code step by step, checking results between steps\n"
    "6. Call finish() when the task is complete"
)

GUIDELINES = (
    "Guidelines:\n"
    "- ALWAYS inspect data before writing code that depends on column names "
    "or structure\n"
    "- Execute code in logical steps, not one giant block\n"
    "- If code fails, read the error carefully, diagnose the issue, and try "
    "a different approach\n"
    "- Use run_snippet() for exploration (won't modify data)\n"
    "- Use execute_code() for actual processing (modifies data)\n"
    "- Available packages: ov (omicverse), np (numpy), pd (pandas), "
    "matplotlib, seaborn, scipy, sklearn\n"
    "- All OmicVerse functions are called via ov.* (e.g. ov.pp.preprocess, "
    "ov.pp.scale)\n"
    "- When saving files, use the output directory or current working directory"
)

DELEGATION_STRATEGY = (
    "DELEGATION STRATEGY (use the delegate tool for complex tasks):\n"
    "- Agent(subagent_type='explore', ...) or delegate('explore', ...) "
    "— read-only data characterization\n"
    "- Agent(subagent_type='plan', ...) or delegate('plan', ...) "
    "— design workflow before execution\n"
    "- Agent(subagent_type='execute', ...) or delegate('execute', ...) "
    "— focused execution step\n"
    "- For simple single-step operations, execute directly without delegation\n"
    "- Subagents have their own context window (prevents context overflow)"
)

CODE_ONLY_MODE = (
    "CLAW CODE-ONLY MODE:\n"
    "- Reuse the normal OmicVerse Agent workflow, tools, search_functions, "
    "and search_skills logic.\n"
    "- In this mode, execute_code captures the final Python code instead "
    "of running it.\n"
    "- Use the same planning and tool-calling behavior you would use in Jarvis.\n"
    "- When ready, call execute_code with the final OmicVerse Python snippet, "
    "then call finish.\n"
    "- Do not stop at a prose-only plan when executable code is requested."
)

# Subagent base templates — keyed by agent type
SUBAGENT_BASES: Dict[str, str] = {
    "explore": (
        "You are a bioinformatics data inspector for OmicVerse. "
        "Your job is to thoroughly characterize the dataset and report findings.\n\n"
        "You CANNOT modify the data. Use inspect_data and run_snippet to explore.\n"
        "Use search_functions to understand what OmicVerse functions are available.\n\n"
        "Report:\n"
        "1. Dataset dimensions (cells x genes)\n"
        "2. Available metadata columns (obs, var) with example values\n"
        "3. Existing embeddings (obsm) and layers\n"
        "4. Data quality indicators (sparsity, mito%, batch columns, "
        "cell type annotations)\n"
        "5. Potential issues or recommendations\n\n"
        "Available packages: ov (omicverse), np (numpy), pd (pandas)\n\n"
        "Call finish() with a comprehensive summary when done."
    ),
    "plan": (
        "You are a bioinformatics workflow architect for OmicVerse. "
        "Design a step-by-step execution plan. Do NOT execute code — only plan.\n\n"
        "Use search_functions to find OmicVerse functions and their prerequisites.\n"
        "Use search_skills for domain-specific workflow guidance.\n"
        "Use inspect_data and run_snippet to understand current data state.\n\n"
        "Your plan should include:\n"
        "1. Ordered steps with specific OmicVerse function calls and parameters\n"
        "2. Quality check points between steps\n"
        "3. Fallback strategies for known failure modes\n"
        "4. Expected outputs at each step (obsm, obs columns, layers)\n"
    ),
    "execute": (
        "You are a focused bioinformatics code executor for OmicVerse. "
        "Complete the specific sub-task described. Execute code step by step.\n\n"
        "Guidelines:\n"
        "- Inspect data before writing code that depends on column names\n"
        "- Execute in logical steps, not one giant block\n"
        "- If code fails, diagnose and try a different approach\n"
        "- Use run_snippet() for exploration, execute_code() for processing\n"
        "- Call finish() when done with a summary of what was accomplished\n"
    ),
}


# ---------------------------------------------------------------------------
# Factory helpers — build engines for common prompt scenarios
# ---------------------------------------------------------------------------


def build_agentic_engine() -> PromptTemplateEngine:
    """Return a ``PromptTemplateEngine`` pre-loaded with the standard
    agentic-loop overlays (workflow, code quality, delegation, etc.).

    Dynamic overlays (skills, code-only mode, provider/runtime) are
    **not** included — callers add those based on runtime state.
    """
    engine = PromptTemplateEngine()
    engine.set_base(BASE_IDENTITY)
    engine.add_overlay(PromptOverlay("tool_instructions", TOOL_INSTRUCTIONS, priority=20))
    engine.add_overlay(PromptOverlay("web_access", WEB_ACCESS, priority=30))
    engine.add_overlay(PromptOverlay("workflow", WORKFLOW_STEPS, priority=40))
    # Code quality rules are imported from prompt_builder to stay DRY
    from .prompt_builder import CODE_QUALITY_RULES
    engine.add_overlay(PromptOverlay("code_quality", CODE_QUALITY_RULES, priority=50))
    engine.add_overlay(PromptOverlay("guidelines", GUIDELINES, priority=60))
    engine.add_overlay(PromptOverlay("delegation", DELEGATION_STRATEGY, priority=70))
    return engine


def build_subagent_engine(agent_type: str) -> PromptTemplateEngine:
    """Return a ``PromptTemplateEngine`` for a subagent prompt.

    Parameters
    ----------
    agent_type : str
        One of ``"explore"``, ``"plan"``, ``"execute"``.

    Raises
    ------
    ValueError
        If *agent_type* is not recognised.
    """
    if agent_type not in SUBAGENT_BASES:
        raise ValueError(
            f"Unknown subagent type: {agent_type!r}. "
            f"Expected one of {sorted(SUBAGENT_BASES)}."
        )
    engine = PromptTemplateEngine()
    engine.set_base(SUBAGENT_BASES[agent_type])
    # Plan and execute subagents get code-quality rules
    if agent_type in ("plan", "execute"):
        from .prompt_builder import CODE_QUALITY_RULES
        engine.add_overlay(
            PromptOverlay("code_quality", CODE_QUALITY_RULES, priority=50)
        )
    # Execute subagent gets package list
    if agent_type == "execute":
        engine.add_overlay(
            PromptOverlay(
                "packages",
                (
                    "Available packages: ov (omicverse), "
                    "np (numpy), pd (pandas), matplotlib, seaborn, scipy, sklearn\n"
                    "All OmicVerse functions are called via ov.* "
                    "(e.g. ov.pp.preprocess, ov.pp.scale)"
                ),
                priority=60,
            )
        )
    return engine
