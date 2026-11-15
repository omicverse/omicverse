"""Prompt construction for OVAgent agentic and subagent loops.

Extracted from ``smart_agent.py`` to keep prompt assembly in one place.
All methods are pure string construction with no side effects.

Since task-034, the heavy lifting is delegated to the
``PromptTemplateEngine`` in ``prompt_templates.py``; this module
remains the public-facing composition API so existing callers are
unaffected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from .prompt_templates import (
    CODE_ONLY_MODE,
    PromptOverlay,
    build_agentic_engine,
    build_subagent_engine,
)

if TYPE_CHECKING:
    from .protocol import AgentContext


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

CODE_QUALITY_RULES = (
    "MANDATORY CODE QUALITY RULES:\n"
    "- NEVER use f-strings in print() - use string concatenation: "
    "print('Result: ' + str(value))\n"
    "- NEVER assign result of in-place OmicVerse functions. Call them directly:\n"
    "  ov.pp.pca(adata, n_pcs=50)   # CORRECT\n"
    "  adata = ov.pp.pca(adata)      # WRONG - returns None!\n"
    "  In-place functions: pca, scale, neighbors, leiden, umap, tsne, "
    "sude, scrublet, mde, louvain, phate\n"
    "- NEVER use .cat.categories - use .value_counts() instead\n"
    "- ALWAYS wrap HVG selection in try/except with fallback:\n"
    "  try:\n"
    "      ov.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=2000)\n"
    "  except ValueError:\n"
    "      ov.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=2000)\n"
    "- ALWAYS validate batch column before batch operations "
    "(check existence, fillna, astype('category'))\n"
)


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """Build system / user prompts for the agentic loop and subagents.

    Internally delegates to :class:`PromptTemplateEngine` for
    deterministic overlay composition while keeping the public API
    unchanged.
    """

    def __init__(self, ctx: "AgentContext") -> None:
        self._ctx = ctx

    # -- subagent prompts ---------------------------------------------------

    def build_explore_prompt(self, context: str) -> str:
        engine = build_subagent_engine("explore")
        if context:
            engine.add_overlay(
                PromptOverlay("parent_context", "Additional context from parent: " + context, priority=200)
            )
        return engine.render()

    def build_plan_prompt(self, context: str) -> str:
        engine = build_subagent_engine("plan")
        engine.add_overlay(
            PromptOverlay("finish_call", "Call finish() with the complete plan when done.", priority=60)
        )
        if context:
            engine.add_overlay(
                PromptOverlay("parent_context", "Additional context from parent: " + context, priority=200)
            )
        # Skill listing overlay
        registry = self._ctx.skill_registry
        if registry is not None and getattr(registry, "skill_metadata", None):
            lines = ["Available domain skills (use search_skills for details):"]
            for meta in sorted(registry.skill_metadata.values(), key=lambda s: s.slug):
                lines.append("  - " + meta.slug + ": " + meta.description[:80])
            engine.add_overlay(PromptOverlay("skills", "\n".join(lines), priority=90))
        return engine.render()

    def build_execute_prompt(self, context: str) -> str:
        engine = build_subagent_engine("execute")
        if context:
            engine.add_overlay(
                PromptOverlay("parent_context", "Context from parent: " + context, priority=200)
            )
        return engine.render()

    def build_subagent_system_prompt(self, agent_type: str, context: str = "") -> str:
        if agent_type == "explore":
            return self.build_explore_prompt(context)
        elif agent_type == "plan":
            return self.build_plan_prompt(context)
        elif agent_type == "execute":
            return self.build_execute_prompt(context)
        raise ValueError("Unknown subagent type: " + agent_type)

    def build_subagent_user_message(self, task: str, adata: Any) -> str:
        msg = f"Task: {task}\n\n"
        if adata is not None and hasattr(adata, "shape"):
            msg += f"Dataset: {adata.shape[0]} cells x {adata.shape[1]} genes\n"
        return msg

    # -- main agentic prompt ------------------------------------------------

    def build_agentic_system_prompt(self) -> str:
        engine = build_agentic_engine()

        # Conditional: code-only mode overlay
        if getattr(self._ctx, "_code_only_mode", False):
            engine.add_overlay(PromptOverlay("code_only_mode", CODE_ONLY_MODE, priority=80))

        # Always-on: mandate registry-first workflow before any analysis code.
        # Conditional rules ("if you use ov.* then search") get bypassed by
        # models that fall back to plain scanpy/numpy. So we mandate UNCONDITIONALLY:
        # Step 1 of every analysis MUST be search_functions to discover the
        # specialist omicverse implementations the registry exposes.
        engine.add_overlay(PromptOverlay(
            "registry_first",
            (
                "MANDATORY WORKFLOW for any single-cell / spatial / multi-omics task:\n"
                "  Step 1 (REQUIRED, FIRST tool call): call "
                "search_functions(query=\"<task keywords>\") to discover the "
                "omicverse functions designed for this task. The omicverse registry "
                "contains 870 curated, GPU-aware, validated functions that handle "
                "edge cases (sparse matrices, mt-prefix detection, version drift) "
                "that handwritten scanpy/numpy code typically gets wrong.\n"
                "  Step 2: prefer the ov.* functions returned by search_functions over "
                "writing your own scanpy / numpy code. Falling back to handwritten "
                "code is allowed ONLY after search_functions returns no relevant matches.\n"
                "  Skipping Step 1 is a workflow violation that will fail the task."
            ),
            priority=10,    # Very high priority — render near the top of system prompt
        ))

        # Always-on: when a tool call fails with FileNotFoundError, missing-DB,
        # or 'please download X first' messages, the model must search the
        # function registry for the matching `download_*` helper before retrying
        # the call or hand-coding a workaround. Many ov.* functions (cellphonedb_v5,
        # TOSICA, geneset enrichment, gene-id mapping) require external reference
        # files that have a paired `ov.*.download_*` function in the registry.
        engine.add_overlay(PromptOverlay(
            "missing_file_recovery",
            (
                "ERROR-RECOVERY RULE: if a tool call fails with one of:\n"
                "  - FileNotFoundError\n"
                "  - 'database not found' / 'please download'\n"
                "  - 'missing required positional argument' for an *_path / *_file param\n"
                "  - 'No such file or directory' for an external reference (genesets, "
                "    cellphonedb DB, model checkpoint, GMT pathway file, gene-id annotation)\n"
                "your NEXT tool call must be search_functions(query=\"download <missing thing>\") "
                "to find the paired `download_*` helper, call that helper to fetch the file, "
                "then retry the original call. Do NOT fabricate a path, write the algorithm "
                "from scratch, or skip the step. Most ov.* functions that need a reference "
                "file have a `ov.*.download_*` companion in the registry."
            ),
            priority=11,    # Just below registry_first
        ))

        # Dynamic: skill listing overlay
        registry = self._ctx.skill_registry
        if registry is not None and getattr(registry, "skill_metadata", None):
            lines = ["Available domain skills (use search_skills for detailed guidance):"]
            for meta in sorted(registry.skill_metadata.values(), key=lambda s: s.slug):
                lines.append("  - " + meta.slug + ": " + meta.description[:80])
            engine.add_overlay(PromptOverlay("skills", "\n".join(lines), priority=90))

        prompt = engine.render()

        # Dynamic: provider/runtime overlay (OV runtime compose)
        if self._ctx._ov_runtime is not None:
            prompt = self._ctx._ov_runtime.compose_system_prompt(prompt)

        return prompt

    def build_initial_user_message(
        self,
        request: str,
        adata: Any,
        *,
        extra_content: Optional[List[dict]] = None,
    ) -> Any:
        msg = f"Task: {request}\n\n"
        if getattr(self._ctx, "_code_only_mode", False):
            if adata is not None:
                dtype = type(adata).__name__
                if hasattr(adata, "shape"):
                    msg += f"Dataset ({dtype}): {adata.shape[0]} cells x {adata.shape[1]} features\n"
                if dtype == "MuData" and hasattr(adata, "mod"):
                    msg += f"Modalities: {list(adata.mod.keys())}\n"
                msg += (
                    "This is a Claw code-only request. Inspect/search as needed, then call "
                    "execute_code with the final script. The code will be captured, not run."
                )
            else:
                msg += (
                    "No live dataset object is loaded for this Claw code-only request.\n"
                    "Assume an AnnData object named `adata` already exists unless the user "
                    "explicitly asks to load data from disk.\n"
                    "Use the normal Jarvis tool workflow, then call execute_code with the "
                    "final OmicVerse Python script. The code will be captured, not run."
                )
            if extra_content:
                return [{"type": "input_text", "text": msg}] + list(extra_content)
            return msg
        if adata is not None:
            dtype = type(adata).__name__
            if hasattr(adata, "shape"):
                msg += f"Dataset ({dtype}): {adata.shape[0]} cells x {adata.shape[1]} features\n"
            if dtype == "MuData" and hasattr(adata, "mod"):
                msg += f"Modalities: {list(adata.mod.keys())}\n"
            msg += "Use inspect_data to learn more about the dataset structure before writing code."
        else:
            msg += (
                "No dataset is loaded yet. You can still help with:\n"
                "- Use web_fetch/web_search to look up information\n"
                "- Use web_download to download datasets from URLs\n"
                "- Answer bioinformatics questions\n"
                "- Plan analysis workflows\n"
            )
        if extra_content:
            return [{"type": "input_text", "text": msg}] + list(extra_content)
        return msg


# ---------------------------------------------------------------------------
# Filesystem context instructions (extracted from smart_agent._build_*)
# ---------------------------------------------------------------------------

def build_filesystem_context_instructions(session_id: str = "N/A") -> str:
    """Build instructions for using the filesystem context workspace.

    This teaches LLMs how to use the filesystem-based context management
    system for offloading intermediate results, plans, and notes.

    Parameters
    ----------
    session_id : str
        Current filesystem context session identifier.

    Returns
    -------
    str
        Instructions for filesystem context usage.
    """
    return f"""

## Context Engineering with Filesystem Workspace

You have access to a **filesystem-based context workspace** that allows you to:
- Offload intermediate results to reduce memory/context usage
- Save and track execution plans across multiple steps
- Search for relevant context using patterns
- Share context with sub-agents

**Current Session**: `{session_id}`

### Why Use the Workspace?

1. **Reduce Context Window Usage**: Instead of keeping all results in memory, write them to disk
2. **Track Multi-Step Workflows**: Save plans and update progress as you complete steps
3. **Retrieve Relevant Context**: Search for notes when you need specific information
4. **Debug and Audit**: All notes are persisted for later review

### Available Context Operations

#### 1. Writing Notes (Offload Results)
Use `# CONTEXT_WRITE:` comments in your code to indicate what should be saved:

```python
# After completing a step, offload the result
# CONTEXT_WRITE: qc_result -> {{"n_cells_before": original_count, "n_cells_after": adata.n_obs, "removed": removed_count}}

# Example: Save intermediate statistics
qc_stats = {{
    "n_cells": adata.n_obs,
    "n_genes": adata.n_vars,
    "mito_pct_mean": float(adata.obs['pct_counts_mt'].mean()) if 'pct_counts_mt' in adata.obs else None
}}
# CONTEXT_WRITE: qc_stats -> qc_stats
```

#### 2. Saving Execution Plans
For multi-step workflows, define a plan upfront:

```python
# CONTEXT_PLAN:
# - Step 1: Quality Control [pending]
# - Step 2: Normalization [pending]
# - Step 3: Feature Selection [pending]
# - Step 4: Dimensionality Reduction [pending]
# - Step 5: Clustering [pending]
```

#### 3. Updating Plan Progress
As you complete steps, update the plan:

```python
# CONTEXT_UPDATE: step=0, status=completed, result="QC removed 500 low-quality cells"
```

#### 4. Searching for Context
When you need to reference previous results:

```python
# CONTEXT_SEARCH: pattern="qc*", type="glob"
# Or for content search:
# CONTEXT_SEARCH: pattern="resolution", type="grep"
```

### Context Categories

Organize your notes by category:
- **notes**: General observations and comments
- **results**: Computation results (statistics, parameters)
- **decisions**: Important choices and their rationale
- **snapshots**: Data state at key points
- **errors**: Error logs and debugging information

### Best Practices

1. **Write Early, Write Often**: Offload results as soon as they're computed
2. **Use Descriptive Keys**: `clustering_leiden_res1.0` is better than `result1`
3. **Include Metadata**: Add function names, parameters, timestamps
4. **Reference Previous Context**: Check workspace before repeating computations
5. **Update Plans Promptly**: Mark steps complete immediately after finishing

### Example: Multi-Step Workflow with Context

```python
import omicverse as ov

# CONTEXT_PLAN:
# - Step 1: Quality Control [in_progress]
# - Step 2: Preprocessing [pending]
# - Step 3: Clustering [pending]

# Step 1: QC
original_cells = adata.n_obs
adata = ov.pp.qc(adata, tresh={{'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250}})
removed = original_cells - adata.n_obs

# CONTEXT_WRITE: qc_result -> {{"original": original_cells, "remaining": adata.n_obs, "removed": removed}}
# CONTEXT_UPDATE: step=0, status=completed, result="Removed " + str(removed) + " cells"

print("QC completed: " + str(adata.n_obs) + " cells remaining")
```

### Automatic Context Injection

The workspace context is automatically searched and injected into prompts when relevant.
You can reference previous results without explicitly searching:

- Recent notes are included automatically
- Plan status is always visible
- Relevant context is retrieved based on the current task
"""
