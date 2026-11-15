"""
scMulan model wrapper backed by ``omicverse.external.scMulan``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from anndata import AnnData

from .base import ModelExecutionBlockedError, SCLLMBase


class ScMulanModel(SCLLMBase):
    """Thin wrapper around the vendored scMulan inference code."""

    def __init__(self, device: Optional[str] = None, **kwargs):
        super().__init__("scmulan", device)
        self.checkpoint_path: Optional[Path] = None
        self.meta_info_path = Path(kwargs.get(
            "meta_info_path",
            Path(__file__).resolve().parents[1] / "external" / "scMulan" / "utils" / "meta_info.pt",
        ))
        self.kv_cache = kwargs.get("kv_cache", False)

    def load_model(self, model_path: Optional[Union[str, Path]], **kwargs) -> None:
        if model_path in (None, ""):
            raise ModelExecutionBlockedError(
                "scmulan",
                "scMulan checkpoint not configured",
                checkpoint_url="https://cloud.tsinghua.edu.cn/f/2250c5df51034b2e9a85/?dl=1",
            )
        checkpoint = Path(model_path)
        if checkpoint.is_dir():
            candidates = [
                checkpoint / "ckpt_scMulan.pt",
                checkpoint / "model.pt",
                checkpoint / "model.pth",
            ]
            checkpoint = next((candidate for candidate in candidates if candidate.exists()), checkpoint)
            if checkpoint.is_dir():
                pt_files = sorted(checkpoint.glob("*.pt")) + sorted(checkpoint.glob("*.pth"))
                if pt_files:
                    checkpoint = pt_files[0]

        if not checkpoint.exists():
            raise ModelExecutionBlockedError(
                "scmulan",
                f"scMulan checkpoint not found: {checkpoint}",
                checkpoint_url="https://cloud.tsinghua.edu.cn/f/2250c5df51034b2e9a85/?dl=1",
            )
        if not self.meta_info_path.exists():
            raise FileNotFoundError(f"scMulan meta_info not found: {self.meta_info_path}")

        try:
            import torch  # noqa: F401
            from scipy.sparse import csc_matrix  # noqa: F401
            from omicverse.external import scMulan  # noqa: F401
        except Exception as exc:
            raise ImportError(f"scMulan dependencies not available: {exc}") from exc

        self.checkpoint_path = checkpoint
        self.is_loaded = True

    def preprocess(self, adata: AnnData, **kwargs) -> AnnData:
        from scipy.sparse import csc_matrix
        from omicverse.external import scMulan
        from omicverse.pp import normalize_total, log1p

        working = adata.copy()
        if not isinstance(working.X, csc_matrix):
            working.X = csc_matrix(working.X)

        with tempfile.TemporaryDirectory(prefix="scmulan_") as tmpdir:
            unified = scMulan.GeneSymbolUniform(
                input_adata=working,
                output_dir=tmpdir,
                output_prefix="scmulan",
            )
        if unified.X.max() > 10:
            normalize_total(unified, target_sum=1e4)
            log1p(unified)
        return unified

    def _infer(self, adata: AnnData, *, annotate: bool = False, **kwargs) -> AnnData:
        if not self.is_loaded or self.checkpoint_path is None:
            raise ModelExecutionBlockedError(
                "scmulan",
                "scMulan checkpoint not loaded",
                checkpoint_url="https://cloud.tsinghua.edu.cn/f/2250c5df51034b2e9a85/?dl=1",
            )

        from omicverse.external import scMulan

        prepared = self.preprocess(adata, **kwargs)
        model = scMulan.model_inference(
            str(self.checkpoint_path),
            prepared,
            meta_info_path=str(self.meta_info_path),
            kv_cache=self.kv_cache,
        )

        parallel = kwargs.pop("parallel", False)
        n_process = kwargs.pop("n_process", None)
        if annotate:
            model.get_cell_types_and_embds_for_adata(parallel=parallel, n_process=n_process, **kwargs)
        else:
            model.get_cell_embeddings_for_adata(parallel=parallel, n_process=n_process, **kwargs)
        return model.adata

    @staticmethod
    def _runtime_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        filtered = dict(kwargs)
        for key in ("batch_size", "batch_key", "label_key", "device"):
            filtered.pop(key, None)
        return filtered

    def predict(self, adata: AnnData, task: str = "annotation", **kwargs) -> dict[str, Any]:
        if task not in {"annotation", "annotate"}:
            raise ValueError(f"scMulan predict only supports annotation, got: {task}")
        result_adata = self._infer(adata, annotate=True, **self._runtime_kwargs(kwargs))
        predictions = result_adata.obs.get("cell_type_from_scMulan")
        embeddings = result_adata.obsm.get("X_scMulan")
        return {
            "predicted_celltypes": predictions.to_numpy() if predictions is not None else None,
            "embeddings": np.asarray(embeddings, dtype=np.float32) if embeddings is not None else None,
        }

    def fine_tune(
        self,
        train_adata: AnnData,
        valid_adata: Optional[AnnData] = None,
        **kwargs,
    ) -> dict[str, Any]:
        raise ModelExecutionBlockedError(
            "scmulan",
            "scMulan fine-tuning is not implemented in omicverse.llm yet.",
            checkpoint_url="https://github.com/SuperBianC/scMulan",
        )

    def get_embeddings(self, adata: AnnData, **kwargs) -> np.ndarray:
        result_adata = self._infer(adata, annotate=False, **self._runtime_kwargs(kwargs))
        embeddings = result_adata.obsm.get("X_scMulan")
        if embeddings is None:
            raise RuntimeError("scMulan did not produce X_scMulan embeddings")
        return np.asarray(embeddings, dtype=np.float32)

    def _save_model_specific(self, save_path: Path, **kwargs) -> None:
        raise ModelExecutionBlockedError(
            "scmulan",
            "Saving scMulan checkpoints is not implemented in omicverse.llm yet.",
            checkpoint_url="https://github.com/SuperBianC/scMulan",
        )
