"""SecAct / CytoSig signature matrices: on-demand download, cache and grouping.

The SecAct signature (7,919 genes x 1,170 secreted proteins, ~80 MB gzipped)
is too large for the omicverse wheel. It is fetched once from the authors'
GitHub repository at a pinned commit, checked against a SHA-256 digest and
cached. The cache directory is, in order of precedence, the ``cache_dir``
argument, ``$OMICVERSE_SECACT_DIR``, or ``./data/secact`` (the ``./data``
convention of :func:`omicverse.datasets.download_data`).
"""
from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

_SECACT_R_COMMIT = "8d9ce8355b3c7c05fff28f4dc3d393747a9f93cc"   # data2intelligence/SecAct
_SECACTPY_COMMIT = "13bb6d2a6ff6ceffcb1c52f8dabccb3b3ebea6f3"   # data2intelligence/SecActpy
_R_EXTDATA = f"https://raw.githubusercontent.com/data2intelligence/SecAct/{_SECACT_R_COMMIT}/inst/extdata"
_PY_DATA = f"https://raw.githubusercontent.com/data2intelligence/SecActpy/{_SECACTPY_COMMIT}/secactpy/data"

RESOURCES = {
    "secact": {
        "filename": "SecAct.tsv.gz",
        "sha256": "eda4d822132ce173f23e19104549de3c8894aa93a0bb4ec25810608c525b3f83",
        "urls": [
            f"{_R_EXTDATA}/SecAct.tsv.gz",
            f"{_PY_DATA}/SecAct.tsv.gz",
        ],
        "lambda": 5e5,
        "group": True,
    },
    "cytosig": {
        "filename": "CytoSig.tsv.gz",
        "sha256": "e72365938f9482b3c211610d873c037c222591a8325eb95f602c79e29c5ea2af",
        "urls": [
            f"{_PY_DATA}/CytoSig.tsv.gz",
        ],
        "lambda": 1e4,
        "group": False,
    },
    "mouse2human": {
        "filename": "Mouse2Human_filter.csv",
        "sha256": "059b7d8b4540dfceeaa71a63dc436d8a9ebf42c9a3bcc2592fc16ec040ba53c5",
        "urls": [
            f"{_R_EXTDATA}/Mouse2Human_filter.csv",
        ],
    },
    "alias2symbol": {
        "filename": "NCBI_20251008_gene_result_alias2symbol.csv.gz",
        "sha256": "dd4c0d3e44fbb0e0fe39f817132c4f2ced1f4135fc7c2dbac35a6716a27d1df4",
        "urls": [
            f"{_R_EXTDATA}/NCBI_20251008_gene_result_alias2symbol.csv.gz",
        ],
    },
}


def cache_dir(path: str | os.PathLike | None = None) -> Path:
    """Resolve (and create) the SecAct cache directory."""
    root = path or os.environ.get("OMICVERSE_SECACT_DIR") or os.path.join(".", "data", "secact")
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_resource(name: str, path: str | os.PathLike | None = None) -> Path:
    """Download ``RESOURCES[name]`` into the cache (once) and verify its digest."""
    info = RESOURCES[name]
    target = cache_dir(path) / info["filename"]
    if target.exists() and (info["sha256"] is None or _sha256(target) == info["sha256"]):
        return target
    from ...datasets._datasets import download_data_requests

    errors = []
    for url in info["urls"]:
        if target.exists():
            target.unlink()
        try:
            download_data_requests(url, file_path=info["filename"], dir=str(target.parent))
        except Exception as exc:  # pragma: no cover - network dependent
            errors.append(f"{url}: {exc}")
            continue
        if info["sha256"] is None or _sha256(target) == info["sha256"]:
            return target
        errors.append(f"{url}: SHA-256 mismatch")
    if target.exists():
        target.unlink()
    raise RuntimeError(
        f"Could not obtain the SecAct resource '{name}'. Download it manually from one of "
        f"{info['urls']} into {target.parent} (or set OMICVERSE_SECACT_DIR). Errors: {errors}"
    )


def read_signature_table(path: str | os.PathLike) -> pd.DataFrame:
    """Read a SecAct-format signature (genes x proteins TSV, unnamed gene column)."""
    frame = pd.read_csv(path, sep="\t", index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    frame.index.name = None
    frame = frame.astype(np.float64)
    if frame.isna().to_numpy().any():
        frame = frame.fillna(0.0)
    return frame


@functools.lru_cache(maxsize=4)
def _load_builtin(name: str, path: str | None) -> pd.DataFrame:
    cache = cache_dir(path)
    npz = cache / (RESOURCES[name]["filename"].split(".")[0] + ".parsed.npz")
    if npz.exists():
        data = np.load(npz, allow_pickle=False)
        if str(data["sha256"]) == RESOURCES[name]["sha256"]:
            return pd.DataFrame(data["values"], index=data["genes"].astype(str),
                                columns=data["proteins"].astype(str))
    frame = read_signature_table(fetch_resource(name, path))
    try:
        np.savez(npz, values=frame.to_numpy(), genes=frame.index.to_numpy(dtype=str),
                 proteins=frame.columns.to_numpy(dtype=str), sha256=RESOURCES[name]["sha256"])
    except OSError:  # pragma: no cover - read-only cache
        pass
    return frame


def load_signature(sig_matrix="SecAct", cache_dir: str | os.PathLike | None = None):
    """Return ``(signature DataFrame genes x proteins, default lambda, default grouping)``.

    ``sig_matrix`` is ``"SecAct"``, ``"CytoSig"``, a path to a SecAct-format TSV,
    or a ``DataFrame`` (genes x proteins) such as a pre-loaded subset.
    """
    if isinstance(sig_matrix, pd.DataFrame):
        frame = sig_matrix.copy()
        frame.index = frame.index.astype(str)
        frame.columns = frame.columns.astype(str)
        return frame.astype(np.float64), 5e5, True
    key = str(sig_matrix).lower()
    if key in ("secact", "cytosig"):
        info = RESOURCES[key]
        frame = _load_builtin(key, None if cache_dir is None else str(cache_dir)).copy()
        return frame, info["lambda"], info["group"]
    if os.path.exists(str(sig_matrix)):
        return read_signature_table(sig_matrix), 5e5, True
    raise ValueError(f"Unknown signature {sig_matrix!r}: use 'SecAct', 'CytoSig', a TSV path or a DataFrame.")


def group_signatures(X: pd.DataFrame, cor: float = 0.9) -> tuple[pd.DataFrame, list[list[str]]]:
    """R ``cluster_signatures``: merge proteins whose signatures correlate above ``cor``.

    Complete-linkage clustering on ``1 - Pearson r``, cut at ``1 - cor``;
    each cluster becomes the mean signature named ``"A|B|C"``.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    names = list(X.columns)
    if len(names) < 2:
        return X.copy(), [[n] for n in names]
    corr = np.corrcoef(X.to_numpy(dtype=np.float64), rowvar=False)
    dist = 1.0 - np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(dist, 0.0)
    labels = fcluster(linkage(squareform(dist, checks=False), method="complete"), t=1.0 - cor,
                      criterion="distance")
    groups: dict[int, list[str]] = {}
    for name, label in zip(names, labels):  # R: unique(group_labels) in column order
        groups.setdefault(int(label), []).append(name)
    members = list(groups.values())
    values = np.column_stack([X[m].to_numpy().mean(axis=1) for m in members])
    return pd.DataFrame(values, index=X.index, columns=["|".join(m) for m in members]), members
