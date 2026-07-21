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
    method: str = "proteinmpnn",
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

    if method != "proteinmpnn":
        raise ValueError(
            f"method must be one of ['proteinmpnn'], got {method!r}. "
            "(LigandMPNN backend is planned.)")
    dev = resolve_device(device)
    if verbose:
        print(f"[ov.synbio.inverse_design] method={method} backbone={pdb} "
              f"device={describe_device(dev)} n={num_sequences}")
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
def _rfdiff_python() -> str:
    """Path to a Python with RFdiffusion installed (its own env — old torch/dgl
    stack, incompatible with the main env)."""
    import os
    env = os.environ.get("OMICOS_RFDIFFUSION_PYTHON")
    if env and os.path.exists(env):
        return env
    scratch = os.environ.get("SCRATCH", os.path.expanduser("~"))
    return os.path.join(scratch, "env", "rfdiff", "bin", "python")


def denovo_backbone(
    spec: Dict,
    out_path: Optional[str] = None,
    device: Optional[str] = None,
    rfdiffusion_dir: Optional[str] = None,
    num_designs: int = 1,
) -> str:
    """Generate a de-novo backbone with RFdiffusion.

    ``spec`` selects the task:

    * ``{'length': 100}`` — unconditional monomer of that length.
    * ``{'contigs': '[A1-100/0 50-50]', 'target_pdb': 'target.pdb',
      'hotspots': ['A30','A33']}`` — **binder** design against a target
      (RFdiffusion PPI mode).

    RFdiffusion runs from its own environment (``OMICOS_RFDIFFUSION_PYTHON`` or
    ``$SCRATCH/env/rfdiff/bin/python``) since it needs an older torch/dgl stack.
    Returns the path to the generated backbone PDB."""
    import os
    from ._device import require_gpu
    from ._esm_common import weights_dir

    dev = require_gpu(resolve_device(device), "denovo_backbone",
                      small_model_hint="(RFdiffusion 无小模型)")
    repo = rfdiffusion_dir or os.path.join(weights_dir(), "RFdiffusion")
    run_py = os.path.join(repo, "scripts", "run_inference.py")
    models = os.path.join(repo, "models")
    if not os.path.exists(run_py):
        raise ImportError(
            "denovo_backbone 需要 RFdiffusion(选做功能)。请:\n"
            "  git clone https://github.com/RosettaCommons/RFdiffusion "
            f"{repo}\n  并 bash scripts/download_models.sh {models},\n"
            "  且建一个装好 RFdiffusion 的独立 env,把 OMICOS_RFDIFFUSION_PYTHON 指向它。"
        )
    py = _rfdiff_python()
    if not os.path.exists(py):
        raise ImportError(
            "RFdiffusion 需要独立环境(老 torch/dgl 栈)。请建 env 并 pip install "
            "其 SE3Transformer + rfdiffusion,然后设 OMICOS_RFDIFFUSION_PYTHON。"
            f"(当前查找:{py})")

    import subprocess, tempfile
    out = out_path or tempfile.mktemp(suffix=".pdb")
    prefix = out[:-4] if out.endswith(".pdb") else out
    cmd = [py, run_py, f"inference.output_prefix={prefix}",
           f"inference.model_directory_path={models}",
           f"inference.num_designs={num_designs}"]
    if spec.get("target_pdb"):                     # binder / PPI mode
        contigs = spec.get("contigs")
        if not contigs:
            raise ValueError("binder 设计需要 spec['contigs'](如 '[A1-100/0 50-50]')。")
        cmd += [f"inference.input_pdb={spec['target_pdb']}",
                f"contigmap.contigs={contigs}"]
        if spec.get("hotspots"):
            hs = "[" + ",".join(spec["hotspots"]) + "]"
            cmd += [f"ppi.hotspot_res={hs}"]
        # binder runs use the beta complex model + no noise for tighter designs
        cmd += ["denoiser.noise_scale_ca=0", "denoiser.noise_scale_frame=0"]
    else:
        length = int(spec.get("length", 100))
        contigs = spec.get("contigs", f"[{length}-{length}]")
        cmd += [f"contigmap.contigs={contigs}"]
    subprocess.run(cmd, check=True, cwd=repo)
    produced = f"{prefix}_0.pdb"
    return produced if os.path.exists(produced) else prefix


__all__ = ["inverse_design", "denovo_backbone", "DesignedSequence"]
