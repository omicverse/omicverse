r"""Vendored access to `ThermoMPNN <https://github.com/Kuhlman-Lab/ThermoMPNN>`_
for real thermostability ΔΔG prediction.

ThermoMPNN (Dieckhaus *et al.* 2024, PNAS) is a transfer-learning head on a
frozen ProteinMPNN encoder, trained on the mega-scale stability dataset. It has
no PyPI package and its inference script has a few environment quirks (a name
clash with the HuggingFace ``datasets`` package, a training-only ``wandb``
import, and a hard-coded repo path in ``local.yaml``). This module clones it on
first use, applies those fixes automatically, and drives its
``analysis/custom_inference.py`` as a subprocess — returning per-mutation ΔΔG
(positive = destabilising) for a PDB.

Weights ship in the repo (``models/thermoMPNN_default.pt``, ~39 MB, +
ProteinMPNN ``vanilla_model_weights``), so no separate download is needed.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Dict, Optional, Tuple

from ._esm_common import weights_dir

_REPO_URL = "https://github.com/Kuhlman-Lab/ThermoMPNN"


def _stub_dir() -> str:
    """A dir holding a no-op ``wandb`` stub (ThermoMPNN imports it for training
    logging only; inference never calls it, and wandb is heavy to install)."""
    d = os.path.join(weights_dir(), "_stubs")
    os.makedirs(d, exist_ok=True)
    stub = os.path.join(d, "wandb.py")
    if not os.path.exists(stub):
        with open(stub, "w") as fh:
            fh.write(
                "def init(*a, **k): return None\n"
                "def log(*a, **k): return None\n"
                "def finish(*a, **k): return None\n"
                "def watch(*a, **k): return None\n"
                "run = None\nconfig = {}\n")
    return d


def ensure_thermompnn() -> str:
    """Clone ThermoMPNN (if needed) and patch it for standalone inference."""
    repo = os.path.join(weights_dir(), "ThermoMPNN")
    if not os.path.exists(os.path.join(repo, "analysis", "custom_inference.py")):
        try:
            subprocess.run(["git", "clone", "--depth", "1", _REPO_URL, repo],
                           check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover
            raise ImportError(
                "ov.synbio stability_ddg(method='thermompnn') 需要 ThermoMPNN,"
                f"自动 git clone 失败。请手动 git clone {_REPO_URL} 到 "
                f"{repo}。({getattr(exc, 'stderr', exc)})") from exc
    # point the hard-coded repo path in local.yaml at this checkout
    lyaml = os.path.join(repo, "local.yaml")
    if os.path.exists(lyaml):
        txt = open(lyaml).read()
        if "/proj/kuhl_lab/ThermoMPNN" in txt:
            open(lyaml, "w").write(
                txt.replace('"/proj/kuhl_lab/ThermoMPNN"', f'"{repo}"'))
    # model checkpoint must be present
    ckpt = os.path.join(repo, "models", "thermoMPNN_default.pt")
    if not os.path.exists(ckpt):  # pragma: no cover
        raise ImportError(
            f"ThermoMPNN 权重缺失({ckpt})。请从仓库重新拉取 models/。")
    return repo


def run_thermompnn(pdb_path: str, chain: str = "A",
                   device: Optional[str] = None) -> Dict[Tuple[int, str], float]:
    """Run ThermoMPNN site-saturation inference on *pdb_path*.

    Returns ``{(position_1based, mut_aa): (ddG, wildtype_aa)}`` where positive
    ΔΔG is destabilising."""
    import pandas as pd
    from ._device import resolve_device, is_cuda

    repo = ensure_thermompnn()
    ckpt = os.path.join(repo, "models", "thermoMPNN_default.pt")
    pdb_path = os.path.abspath(pdb_path)
    out_dir = tempfile.mkdtemp(prefix="ovsynbio_thermompnn_")

    env = dict(os.environ)
    # local datasets.py must win over the installed HuggingFace `datasets`;
    # the wandb stub must resolve before any real wandb.
    env["PYTHONPATH"] = os.pathsep.join(
        [_stub_dir(), repo, os.path.join(repo, "analysis"),
         env.get("PYTHONPATH", "")])
    if not is_cuda(resolve_device(device)):
        env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = [sys.executable, "custom_inference.py", "--pdb", pdb_path,
           "--chain", chain, "--model_path", ckpt, "--out_dir", out_dir]
    proc = subprocess.run(cmd, cwd=os.path.join(repo, "analysis"),
                          env=env, capture_output=True, text=True)
    base = os.path.splitext(os.path.basename(pdb_path))[0]
    csv = os.path.join(out_dir, f"ThermoMPNN_inference_{base}.csv")
    if not os.path.exists(csv):
        raise RuntimeError(
            "ThermoMPNN 推理失败:\n" + (proc.stderr or proc.stdout)[-1500:])
    df = pd.read_csv(csv)
    # ThermoMPNN position is 0-based over the parsed chain sequence; keep the
    # wildtype residue so callers can build proper mutation strings.
    return {(int(r["position"]) + 1, str(r["mutation"])):
            (float(r["ddG_pred"]), str(r["wildtype"]))
            for _, r in df.iterrows()}
