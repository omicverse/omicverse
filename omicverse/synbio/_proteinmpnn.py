r"""Vendored access to `ProteinMPNN <https://github.com/dauparas/ProteinMPNN>`_.

ProteinMPNN has no PyPI package, so on first use we ``git clone`` it into the
synbio weights dir (``OMICOS_SYNBIO_WEIGHTS``, default ``~/.omicverse/
synbio_weights``) — the repo bundles its own model weights, no separate
download.  We drive its officially-supported ``protein_mpnn_run.py`` CLI as a
subprocess (robust, matches upstream featurisation) rather than reimplementing
the graph featuriser.

Two capabilities are exposed:

* :func:`design_sequences` — sample sequences for a backbone (inverse folding).
* :func:`score_sequences` — global negative-log-likelihood of given sequences
  threaded onto a backbone (the basis of the ProteinMPNN ΔΔG stability proxy).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Sequence

from ._esm_common import weights_dir

_REPO_URL = "https://github.com/dauparas/ProteinMPNN"


def ensure_proteinmpnn() -> str:
    """Return the local ProteinMPNN repo path, cloning it on first use."""
    root = os.path.join(weights_dir(), "ProteinMPNN")
    if os.path.isdir(root) and os.path.exists(os.path.join(root, "protein_mpnn_run.py")):
        return root
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", _REPO_URL, root],
            check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover
        raise ImportError(
            "ov.synbio 需要 ProteinMPNN,但自动 git clone 失败。请手动:"
            f" git clone {_REPO_URL} 到 $OMICOS_SYNBIO_WEIGHTS/ProteinMPNN。"
            f" (原始错误: {getattr(exc, 'stderr', exc)})"
        ) from exc
    return root


def _run_cli(args: List[str]) -> subprocess.CompletedProcess:
    repo = ensure_proteinmpnn()
    cmd = [sys.executable, os.path.join(repo, "protein_mpnn_run.py"), *args]
    env = dict(os.environ)
    return subprocess.run(cmd, check=True, capture_output=True, text=True,
                          cwd=repo, env=env)


def _parse_fasta(path: str) -> List[Dict[str, object]]:
    """Parse ProteinMPNN output FASTA → list of {header, seq, meta}."""
    entries: List[Dict[str, object]] = []
    header, seq = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    entries.append({"header": header, "seq": "".join(seq)})
                header, seq = line[1:], []
            elif line:
                seq.append(line)
    if header is not None:
        entries.append({"header": header, "seq": "".join(seq)})
    # parse key=value metadata in headers (score=, global_score=, T=, sample=)
    for e in entries:
        meta = {}
        for tok in str(e["header"]).replace(",", " ").split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                try:
                    meta[k] = float(v)
                except ValueError:
                    meta[k] = v
        e["meta"] = meta
    return entries


def design_sequences(
    pdb_path: str,
    num_sequences: int = 8,
    sampling_temp: float = 0.1,
    device: Optional[str] = None,
    seed: int = 37,
) -> List[Dict[str, object]]:
    """Sample sequences that fold to the backbone in *pdb_path*.

    Returns a list of ``{header, seq, meta}`` (the first entry is the native
    sequence recovered by ProteinMPNN)."""
    from ._device import resolve_device, is_cuda

    dev = resolve_device(device)
    out_dir = tempfile.mkdtemp(prefix="ovsynbio_mpnn_")
    args = [
        "--pdb_path", pdb_path,
        "--out_folder", out_dir,
        "--num_seq_per_target", str(num_sequences),
        "--sampling_temp", str(sampling_temp),
        "--seed", str(seed),
        "--batch_size", "1",
    ]
    if not is_cuda(dev):
        args += ["--device", "cpu"]
    _run_cli(args)
    base = os.path.splitext(os.path.basename(pdb_path))[0]
    fa = os.path.join(out_dir, "seqs", f"{base}.fa")
    return _parse_fasta(fa)


MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def unconditional_log_probs(
    pdb_path: str,
    device: Optional[str] = None,
    seed: int = 37,
):
    """Per-residue unconditional log-probabilities over the 21-letter alphabet.

    Returns ``(log_p, S, alphabet)`` where ``log_p`` is ``(L, 21)``, ``S`` is
    the native amino-acid indices, and ``alphabet`` is ``MPNN_ALPHABET``.  This
    is the basis of the ProteinMPNN zero-shot ΔΔG stability proxy: a mutation's
    destabilisation ≈ ``log_p[i, wt] - log_p[i, mut]``.
    """
    import numpy as np
    from ._device import resolve_device, is_cuda

    dev = resolve_device(device)
    out_dir = tempfile.mkdtemp(prefix="ovsynbio_mpnnprob_")
    args = [
        "--pdb_path", pdb_path,
        "--out_folder", out_dir,
        "--unconditional_probs_only", "1",
        "--seed", str(seed),
        "--batch_size", "1",
    ]
    if not is_cuda(dev):
        args += ["--device", "cpu"]
    _run_cli(args)
    prob_dir = os.path.join(out_dir, "unconditional_probs_only")
    npz = None
    for f in os.listdir(prob_dir):
        if f.endswith(".npz"):
            npz = os.path.join(prob_dir, f)
            break
    if npz is None:  # pragma: no cover
        raise RuntimeError("ProteinMPNN unconditional_probs_only 未产生 .npz")
    d = np.load(npz)
    log_p = d["log_p"]
    if log_p.ndim == 3:  # (1, L, 21)
        log_p = log_p[0]
    S = d["S"]
    if S.ndim == 2:
        S = S[0]
    return log_p, S, MPNN_ALPHABET


def score_sequences(
    pdb_path: str,
    device: Optional[str] = None,
    seed: int = 37,
) -> Dict[str, float]:
    """Score the native sequence of *pdb_path* under ProteinMPNN.

    Returns ``{"global_score": float, "score": float}`` where lower is more
    favourable (mean negative log-likelihood per residue)."""
    import numpy as np
    from ._device import resolve_device, is_cuda

    dev = resolve_device(device)
    out_dir = tempfile.mkdtemp(prefix="ovsynbio_mpnnscore_")
    args = [
        "--pdb_path", pdb_path,
        "--out_folder", out_dir,
        "--score_only", "1",
        "--seed", str(seed),
        "--batch_size", "1",
    ]
    if not is_cuda(dev):
        args += ["--device", "cpu"]
    _run_cli(args)
    base = os.path.splitext(os.path.basename(pdb_path))[0]
    score_dir = os.path.join(out_dir, "score_only")
    # ProteinMPNN writes <base>_pdb.npz for the native sequence
    npz = None
    for f in os.listdir(score_dir):
        if f.endswith(".npz"):
            npz = os.path.join(score_dir, f)
            break
    if npz is None:  # pragma: no cover
        raise RuntimeError("ProteinMPNN score_only 未产生 .npz 输出")
    d = np.load(npz)
    score = float(np.mean(d["score"])) if "score" in d else float("nan")
    gscore = float(np.mean(d["global_score"])) if "global_score" in d else score
    return {"score": score, "global_score": gscore}
