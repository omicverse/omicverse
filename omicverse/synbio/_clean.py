r"""Vendored access to `CLEAN <https://github.com/tttianhao/CLEAN>`_ for real
enzyme EC-number prediction (Yu *et al.* 2023, Science).

CLEAN is a contrastive-learning model over ESM-1b embeddings that assigns EC
numbers by max-separation from ~5,200 EC cluster centres of reviewed SwissProt.
It has no PyPI package and its weights + precomputed EC-centre embeddings live
on Google Drive. This module, on first use, clones the repo, fetches the
pretrained bundle (gdown) and the fair-esm ``extract.py`` embedding script,
places them where CLEAN expects, and drives ``CLEAN_infer_fasta.py`` as a
subprocess — parsing the predicted EC numbers and confidences.

Downloads on first use: the CLEAN pretrained bundle (~150 MB) and the ESM-1b
weights (~7.8 GB, via fair-esm) into ``OMICOS_SYNBIO_WEIGHTS``.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

from ._esm_common import weights_dir

_REPO_URL = "https://github.com/tttianhao/CLEAN"
_BUNDLE_GDRIVE_ID = "1kwYd4VtzYuMvJMWXy6Vks91DSUAOcKpZ"
_EXTRACT_URL = "https://raw.githubusercontent.com/facebookresearch/esm/main/scripts/extract.py"


def ensure_clean() -> str:
    """Clone CLEAN + fetch its weights/embedding-script; return the app dir."""
    root = os.path.join(weights_dir(), "CLEAN")
    app = os.path.join(root, "app")
    if not os.path.exists(os.path.join(app, "CLEAN_infer_fasta.py")):
        try:
            subprocess.run(["git", "clone", "--depth", "1", _REPO_URL, root],
                           check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover
            raise ImportError(
                "enzyme_function(method='clean') 需要 CLEAN,自动 git clone 失败。"
                f"请手动 git clone {_REPO_URL} 到 {root}。"
                f"({getattr(exc, 'stderr', exc)})") from exc

    # fair-esm's extract.py embedding script (CLEAN calls `python esm/scripts/extract.py`)
    extract = os.path.join(app, "esm", "scripts", "extract.py")
    if not os.path.exists(extract):
        os.makedirs(os.path.dirname(extract), exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(_EXTRACT_URL, extract)

    # pretrained bundle: split100.pth (model) + 100.pt (EC-centre embeddings) + gmm
    pre = os.path.join(app, "data", "pretrained")
    if not os.path.exists(os.path.join(pre, "split100.pth")):
        os.makedirs(pre, exist_ok=True)
        try:
            import gdown
        except ImportError as exc:
            raise ImportError(
                "enzyme_function(method='clean') 需要 gdown 下载 CLEAN 预训练权重:"
                "pip install gdown。") from exc
        zpath = os.path.join(app, "data", "clean_pretrained.zip")
        gdown.download(id=_BUNDLE_GDRIVE_ID, output=zpath, quiet=True)
        import zipfile
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(os.path.join(app, "data", "_clean_bundle"))
        for dirpath, _dirs, files in os.walk(os.path.join(app, "data", "_clean_bundle")):
            for f in files:
                if f.endswith((".pth", ".pt", ".pkl")):
                    import shutil
                    shutil.copy(os.path.join(dirpath, f), os.path.join(pre, f))
    return app


def run_clean(sequences: Sequence[str], names: Optional[Sequence[str]] = None,
              device: Optional[str] = None) -> List[List[Tuple[str, float]]]:
    """Predict EC numbers for *sequences* with CLEAN.

    Returns, per input sequence, a list of ``(EC_number, confidence)`` sorted by
    confidence (CLEAN's max-separation calls, possibly several per query)."""
    from ._device import resolve_device, is_cuda

    app = ensure_clean()
    names = list(names) if names is not None else [f"q{i}" for i in range(len(sequences))]
    uid = "ovsynbio_" + next(tempfile._get_candidate_names())
    inputs = os.path.join(app, "data", "inputs")
    os.makedirs(inputs, exist_ok=True)
    fasta = os.path.join(inputs, f"{uid}.fasta")
    with open(fasta, "w") as fh:
        for nm, sq in zip(names, sequences):
            fh.write(f">{nm}\n{sq}\n")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(app, "src"), env.get("PYTHONPATH", "")])
    env.setdefault("TORCH_HOME", weights_dir())
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    if not is_cuda(resolve_device(device)):
        env["CUDA_VISIBLE_DEVICES"] = ""

    proc = subprocess.run(
        [sys.executable, "CLEAN_infer_fasta.py", "-d", uid],
        cwd=app, env=env, capture_output=True, text=True)
    out_csv = os.path.join(app, "results", "inputs", f"{uid}_maxsep.csv")
    if not os.path.exists(out_csv):
        raise RuntimeError("CLEAN 推理失败:\n" + (proc.stderr or proc.stdout)[-1500:])

    # parse: each line "name,EC:x.x.x.x/conf,EC:.../conf,..."
    by_name: Dict[str, List[Tuple[str, float]]] = {}
    with open(out_csv) as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            nm = row[0]
            preds = []
            for tok in row[1:]:
                tok = tok.strip()
                if tok.startswith("EC:"):
                    ec, _, conf = tok[3:].partition("/")
                    preds.append((ec, float(conf) if conf else 1.0))
            preds.sort(key=lambda t: t[1], reverse=True)
            by_name[nm] = preds
    return [by_name.get(nm, []) for nm in names]
