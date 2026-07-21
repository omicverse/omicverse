r"""Layer B — protein sequence & backbone design.

* :func:`inverse_design` — ProteinMPNN inverse folding: given a backbone (PDB),
  sample sequences predicted to fold to it.  Runs on CPU (slower) or GPU.
* :func:`denovo_backbone` — de-novo backbone generation with RFdiffusion
  (optional / heavy; gated behind an actionable error when unavailable).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from .._registry import register_function
from ._device import resolve_device, describe_device, warn_if_cpu


@dataclass
class DesignedSequence:
    """One ProteinMPNN design."""

    seq: str
    score: float
    recovery: Optional[float] = None
    is_native: bool = False

    def __repr__(self) -> str:  # pragma: no cover
        tag = " (native)" if self.is_native else ""
        return f"DesignedSequence(score={self.score:.3f}{tag}, seq={self.seq})"


@register_function(
    aliases=[
        "inverse_design", "逆向设计", "反向折叠", "序列设计", "inverse_folding",
        "proteinmpnn", "骨架设计序列", "固定骨架设计",
    ],
    category="synthetic_biology",
    description="逆向设计 / 固定骨架序列设计:给定蛋白骨架 (PDB),用 ProteinMPNN 采样能折叠成该骨架的候选序列。CPU 可跑。ProteinMPNN inverse folding → candidate sequences.",
    examples=[
        "designs = ov.synbio.inverse_design('backbone.pdb', num_sequences=8)",
        "designs[0].seq, designs[0].score",
    ],
    related=["synbio.predict_structure", "synbio.stability_ddg", "synbio.denovo_backbone"],
    requires={},
    produces={},
)
def inverse_design(
    pdb: str,
    num_sequences: int = 8,
    sampling_temp: float = 0.1,
    device: Optional[str] = None,
    seed: int = 37,
    verbose: bool = True,
) -> List[DesignedSequence]:
    """Design sequences for the backbone in *pdb* with ProteinMPNN.

    Parameters
    ----------
    pdb
        Path to a backbone PDB file.
    num_sequences
        Number of sequences to sample.
    sampling_temp
        Sampling temperature (lower = more conservative / native-like).
    device
        ``None`` = auto.  ProteinMPNN is light enough to run on CPU.
    seed
        RNG seed for reproducibility.

    Returns
    -------
    list[DesignedSequence]
        The native (recovered) sequence first, then designs sorted by score
        (lower ProteinMPNN score = more confident).
    """
    from ._proteinmpnn import design_sequences

    dev = resolve_device(device)
    if verbose:
        print(f"[ov.synbio.inverse_design] backbone={pdb} device={describe_device(dev)} "
              f"n={num_sequences}")
    warn_if_cpu(dev, "inverse_design")

    entries = design_sequences(pdb, num_sequences=num_sequences,
                               sampling_temp=sampling_temp, device=dev, seed=seed)
    out: List[DesignedSequence] = []
    for i, e in enumerate(entries):
        meta = e.get("meta", {})
        out.append(DesignedSequence(
            seq=str(e["seq"]),
            score=float(meta.get("score", meta.get("global_score", float("nan")))),
            recovery=meta.get("seq_recovery"),
            is_native=(i == 0),
        ))
    # keep native first, sort the rest by ascending score
    native = [d for d in out if d.is_native]
    designs = sorted((d for d in out if not d.is_native), key=lambda d: d.score)
    return native + designs


@register_function(
    aliases=[
        "denovo_backbone", "从头设计骨架", "de_novo_backbone", "rfdiffusion",
        "骨架生成", "从头蛋白设计",
    ],
    category="synthetic_biology",
    description="从头骨架生成 (RFdiffusion):按约束(长度/对称/结合位点)扩散生成新蛋白骨架 PDB。需 GPU + RFdiffusion 权重(可选,选做)。De-novo backbone generation with RFdiffusion.",
    examples=[
        "pdb = ov.synbio.denovo_backbone({'length': 100})",
    ],
    related=["synbio.inverse_design", "synbio.predict_structure"],
    requires={},
    produces={},
)
def denovo_backbone(
    spec: Dict,
    out_path: Optional[str] = None,
    device: Optional[str] = None,
    rfdiffusion_dir: Optional[str] = None,
) -> str:
    """Generate a de-novo backbone with RFdiffusion (optional feature).

    RFdiffusion has no PyPI package and needs its own weights; this wrapper
    drives a local checkout when available and otherwise raises an actionable
    error.  Marked optional in the ov.synbio spec.
    """
    import os
    from ._device import require_gpu
    from ._esm_common import weights_dir

    dev = require_gpu(resolve_device(device), "denovo_backbone",
                      small_model_hint="(RFdiffusion 无小模型)")
    repo = rfdiffusion_dir or os.path.join(weights_dir(), "RFdiffusion")
    run_py = os.path.join(repo, "scripts", "run_inference.py")
    if not os.path.exists(run_py):
        raise ImportError(
            "denovo_backbone 需要 RFdiffusion(选做功能,未自动安装)。请:\n"
            "  git clone https://github.com/RosettaCommons/RFdiffusion "
            f"{repo}\n  并按其 README 下载权重到 {repo}/models。\n"
            "然后重试,或使用 predict_structure + inverse_design 的组合。"
        )
    # minimal driver: length-only unconditional generation
    import subprocess, sys, tempfile
    length = int(spec.get("length", 100))
    out = out_path or tempfile.mktemp(suffix=".pdb")
    prefix = out[:-4] if out.endswith(".pdb") else out
    cmd = [
        sys.executable, run_py,
        f"inference.output_prefix={prefix}",
        "inference.num_designs=1",
        f"contigmap.contigs=[{length}-{length}]",
    ]
    subprocess.run(cmd, check=True, cwd=repo)
    produced = f"{prefix}_0.pdb"
    return produced if os.path.exists(produced) else prefix


__all__ = ["inverse_design", "denovo_backbone", "DesignedSequence"]
