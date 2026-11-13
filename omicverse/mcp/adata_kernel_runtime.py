"""Persistent Jupyter-kernel execution for adata-backed MCP tools."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .session_store import SessionStore

logger = logging.getLogger(__name__)


class AdataKernelRuntime:
    """Execute adata-backed tools in one long-lived ipykernel session.

    This preserves state across calls and allows cancellation to be handled as
    a kernel interrupt instead of killing the entire execution environment.
    """

    def __init__(self, *, store: SessionStore) -> None:
        self._store = store
        self._executor = None
        self._lock = asyncio.Lock()
        self._kernel_state: Dict[str, dict[str, Any]] = {}
        self._runtime_dir = Path(tempfile.mkdtemp(prefix="ov_mcp_kernel_runtime_"))

    def _get_executor(self):
        if self._executor is None:
            from ..utils.session_notebook_executor import SessionNotebookExecutor

            self._executor = SessionNotebookExecutor(
                max_prompts_per_session=10**9,
                storage_dir=self._runtime_dir / "sessions",
                keep_notebooks=True,
                timeout=3600,
                strict_kernel_validation=False,
            )
            if self._executor._should_start_new_session():
                self._executor._start_new_session()
        return self._executor

    async def execute(self, entry: dict, args: dict) -> dict:
        adata_id = args["adata_id"]
        async with self._lock:
            await asyncio.to_thread(self._ensure_synced, adata_id)
            runner = asyncio.create_task(asyncio.to_thread(self._execute_blocking, entry, args))
            try:
                return await asyncio.shield(runner)
            except asyncio.CancelledError:
                logger.warning("Kernel runtime cancelling tool=%s adata_id=%s", entry.get("tool_name", ""), adata_id)
                await asyncio.to_thread(self._interrupt_kernel)
                self._kernel_state.pop(adata_id, None)
                try:
                    await asyncio.wait_for(asyncio.shield(runner), timeout=10)
                except asyncio.CancelledError:
                    logger.info("Kernel runtime background execution acknowledged cancellation for %s", adata_id)
                except Exception as exc:
                    logger.info("Kernel runtime background execution settled after cancellation: %s", exc)
                except asyncio.TimeoutError:
                    logger.warning("Kernel interrupt timed out; restarting kernel")
                    await asyncio.to_thread(self._reset_executor)
                if self._executor is not None and getattr(self._executor, "current_session", None) is None:
                    logger.warning("Kernel runtime left without an active session after cancellation; resetting executor")
                    await asyncio.to_thread(self._reset_executor)
                elif self._executor is not None:
                    await asyncio.to_thread(self._probe_kernel_after_cancel)
                raise

    async def flush_dirty(self, adata_id: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._flush_dirty_blocking, adata_id)

    async def drop(self, adata_id: str) -> None:
        async with self._lock:
            self._kernel_state.pop(adata_id, None)
            exec_ = self._get_executor()
            code = (
                f"OV_MCP_ADATA.pop({json.dumps(adata_id)}, None)\n"
                f"OV_MCP_META.pop({json.dumps(adata_id)}, None)\n"
            )
            exec_._execute_code_in_kernel(code, exec_.current_session["kernel_client"])

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        self._kernel_state.clear()

    def _ensure_synced(self, adata_id: str) -> None:
        state = self._kernel_state.get(adata_id, {})
        if state.get("dirty"):
            self._flush_dirty_blocking(adata_id)
            state = self._kernel_state.get(adata_id, {})

        current_revision = self._store.get_adata_revision(adata_id)
        if state.get("revision") == current_revision:
            return

        exec_ = self._get_executor()
        snapshot_path = self._runtime_dir / f"{adata_id}.sync.h5ad"
        self._store.persist_adata(adata_id, path=str(snapshot_path))
        code = (
            "import anndata as _ov_ad\n"
            "OV_MCP_ADATA = globals().setdefault('OV_MCP_ADATA', {})\n"
            "OV_MCP_META = globals().setdefault('OV_MCP_META', {})\n"
            f"OV_MCP_ADATA[{json.dumps(adata_id)}] = _ov_ad.read_h5ad({json.dumps(str(snapshot_path))})\n"
            f"OV_MCP_META[{json.dumps(adata_id)}] = {{'revision': {current_revision}}}\n"
        )
        exec_._execute_code_in_kernel(code, exec_.current_session["kernel_client"])
        self._kernel_state[adata_id] = {"revision": current_revision, "dirty": False}

    def _flush_dirty_blocking(self, adata_id: str) -> None:
        state = self._kernel_state.get(adata_id)
        if not state or not state.get("dirty"):
            return
        exec_ = self._get_executor()
        snapshot_path = self._runtime_dir / f"{adata_id}.flush.h5ad"
        code = (
            "OV_MCP_ADATA = globals().setdefault('OV_MCP_ADATA', {})\n"
            f"_adata = OV_MCP_ADATA.get({json.dumps(adata_id)})\n"
            "if _adata is None:\n"
            "    raise RuntimeError('adata not present in kernel cache')\n"
            f"_adata.write_h5ad({json.dumps(str(snapshot_path))})\n"
        )
        outputs = exec_._execute_code_in_kernel(code, exec_.current_session["kernel_client"])
        if outputs["errors"]:
            err = outputs["errors"][0]
            raise RuntimeError(f"{err['ename']}: {err['evalue']}")
        if not snapshot_path.exists():
            raise RuntimeError(
                "Kernel flush completed without writing adata snapshot "
                f"for {adata_id}. stdout={outputs['stdout']} stderr={outputs['stderr']}"
            )
        self._store.restore_adata(str(snapshot_path), adata_id=adata_id)
        self._kernel_state[adata_id] = {
            "revision": self._store.get_adata_revision(adata_id),
            "dirty": False,
        }

    def _interrupt_kernel(self) -> None:
        if self._executor is None or getattr(self._executor, "current_session", None) is None:
            return
        self._executor._interrupt_kernel()

    def _restart_kernel(self) -> None:
        if self._executor is None or getattr(self._executor, "current_session", None) is None:
            self._reset_executor()
            return
        self._executor._restart_kernel()

    def _reset_executor(self) -> None:
        if self._executor is not None:
            try:
                self._executor.shutdown()
            except Exception as exc:
                logger.warning("Kernel runtime shutdown during reset failed: %s", exc)
            self._executor = None
        self._kernel_state.clear()

    def _probe_kernel_after_cancel(self) -> None:
        if self._executor is None or getattr(self._executor, "current_session", None) is None:
            return
        try:
            outputs = self._executor._execute_code_in_kernel(
                "pass",
                self._executor.current_session["kernel_client"],
                auto_recover=True,
            )
            if outputs["errors"]:
                raise RuntimeError(str(outputs["errors"][0]))
        except Exception as exc:
            logger.warning("Kernel runtime post-cancel probe failed; resetting executor: %s", exc)
            self._reset_executor()

    def _execute_blocking(self, entry: dict, args: dict) -> dict:
        from .adapters.adata_adapter import AdataAdapter

        tool_name = entry["tool_name"]
        adata_id = args["adata_id"]
        func = entry["_function"]
        adata = self._store.get_adata(adata_id)
        helper = AdataAdapter()
        pre_state = helper.snapshot_pre_state(adata, entry)

        exec_ = self._get_executor()
        result_path = self._runtime_dir / f"{adata_id}.result.json"
        adata_out = self._runtime_dir / f"{adata_id}.out.h5ad"
        image_path = self._runtime_dir / f"{adata_id}.png"

        kwargs = dict(args)
        kwargs.pop("adata_id", None)
        kwargs.pop("instance_id", None)
        kwargs_json = json.dumps(kwargs, ensure_ascii=False, default=str)
        module_path, attr_name = entry["full_name"].rsplit(".", 1)
        code = f"""
import importlib, json, os, matplotlib.pyplot as plt
OV_MCP_ADATA = globals().setdefault('OV_MCP_ADATA', {{}})
OV_MCP_META = globals().setdefault('OV_MCP_META', {{}})
_adata_id = {json.dumps(adata_id)}
adata = OV_MCP_ADATA[_adata_id]
_kwargs = json.loads({json.dumps(kwargs_json)})
_mod = importlib.import_module({json.dumps(module_path)})
_func = getattr(_mod, {json.dumps(attr_name)})
_result = _func(adata=adata, **_kwargs)
if type(_result).__name__ in ('AnnData', 'MuData') and _result is not adata:
    adata = _result
    OV_MCP_ADATA[_adata_id] = adata
adata.write_h5ad({json.dumps(str(adata_out))})
_payload = {{
    "result_type": type(_result).__name__ if _result is not None else "NoneType",
    "dataframe": None,
    "json_data": None,
    "text_data": None,
    "image_path": None,
}}
if type(_result).__name__ == 'DataFrame':
    _subset = _result.head(200)
    _payload["dataframe"] = {{
        "columns": list(_subset.columns),
        "data": _subset.values.tolist(),
        "shape": list(_result.shape),
        "truncated": len(_result) > 200,
    }}
elif isinstance(_result, (dict, list, str, int, float, bool)) or _result is None:
    _payload["json_data"] = _result
else:
    try:
        json.dumps(_result)
        _payload["json_data"] = _result
    except Exception:
        _payload["text_data"] = str(_result)
try:
    _fig = plt.gcf()
    if _fig.get_axes():
        _fig.savefig({json.dumps(str(image_path))}, dpi=150, bbox_inches='tight')
        _payload["image_path"] = {json.dumps(str(image_path))}
    plt.close(_fig)
except Exception:
    pass
with open({json.dumps(str(result_path))}, 'w', encoding='utf-8') as _fh:
    json.dump(_payload, _fh, ensure_ascii=False, default=str)
"""
        outputs = exec_._execute_code_in_kernel(code, exec_.current_session["kernel_client"])
        if outputs["errors"]:
            err = outputs["errors"][0]
            raise RuntimeError(f"{err['ename']}: {err['evalue']}")

        if not adata_out.exists():
            raise RuntimeError(
                "Kernel execution completed without writing adata output "
                f"for {tool_name}. stdout={outputs['stdout']} stderr={outputs['stderr']}"
            )
        if not result_path.exists():
            raise RuntimeError(
                "Kernel execution completed without writing result metadata "
                f"for {tool_name}. stdout={outputs['stdout']} stderr={outputs['stderr']}"
            )

        self._store.restore_adata(str(adata_out), adata_id=adata_id)
        updated = self._store.get_adata(adata_id)
        state_updates = helper.detect_state_updates(updated, pre_state, entry)
        revision = self._store.get_adata_revision(adata_id)
        self._kernel_state[adata_id] = {"revision": revision, "dirty": False}

        with open(result_path, encoding="utf-8") as fh:
            payload = json.load(fh)

        outputs_payload = []
        primary_output = entry.get("return_contract", {}).get("primary_output", "object_ref")
        if primary_output == "image" and payload.get("image_path"):
            artifact_id = self._store.create_artifact(
                payload["image_path"],
                "image/png",
                artifact_type="image",
                source_tool=tool_name,
            )
            outputs_payload.append({
                "type": "image",
                "artifact_id": artifact_id,
                "path": payload["image_path"],
                "content_type": "image/png",
            })
            outputs_payload.append({
                "type": "object_ref",
                "ref_type": "adata",
                "ref_id": adata_id,
            })
        elif primary_output == "table":
            if payload.get("dataframe") is not None:
                outputs_payload.append({"type": "table", "data": payload["dataframe"]})
            elif payload.get("json_data") is not None:
                outputs_payload.append({"type": "json", "data": payload["json_data"]})
            elif payload.get("text_data") is not None:
                outputs_payload.append({"type": "text", "data": payload["text_data"]})
        else:
            outputs_payload.append({
                "type": "object_ref",
                "ref_type": "adata",
                "ref_id": adata_id,
            })
            if primary_output == "image" and not payload.get("image_path"):
                logger.warning("Expected image output for %s but none was produced", tool_name)
            if primary_output == "json" and payload.get("json_data") is not None:
                outputs_payload.insert(0, {"type": "json", "data": payload["json_data"]})

        summary_parts = [entry.get("description", tool_name)]
        produced = state_updates.get("produced", {})
        if produced:
            keys = []
            for slot, names in produced.items():
                keys.extend(f"{slot}[{n}]" for n in names)
            summary_parts.append(f"Produced: {', '.join(keys)}")
        if hasattr(updated, "shape"):
            summary_parts.append(f"Shape: {updated.shape[0]} x {updated.shape[1]}")

        return {
            "ok": True,
            "tool_name": tool_name,
            "summary": " | ".join(summary_parts),
            "outputs": outputs_payload,
            "state_updates": state_updates,
            "warnings": [],
        }
