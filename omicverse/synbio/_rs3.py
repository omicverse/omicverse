r"""Runtime access to `Rule Set 3 <https://github.com/gpp-rnd/rs3>`_ — the real
on-target sgRNA activity model (DeWeirdt *et al.* 2021, *Nature Biotechnology*;
the maintained successor to Azimuth / Rule Set 2).

``rs3`` cannot be a declared dependency of ``omicverse[synbio]``. Its metadata
pins ``scikit-learn<=1.0.2``, ``numpy<=1.26.4`` and ``lightgbm<=3.3.5``, all of
which contradict omicverse's own floors — and ``rs3 0.0.18`` is its only release
satisfying ``rs3>=0.0.18``, so no resolver can back off. Listing it made the
whole extra unsatisfiable (see the note in ``pyproject.toml``).

Those pins are, however, **stale rather than real**: ``RuleSet3.pkl`` unpickles
to a plain ``lightgbm.sklearn.LGBMRegressor`` — no scikit-learn estimator is
stored in it at all, scikit-learn is merely lightgbm's own dependency. Verified
by running the same 30-mers through rs3's pinned stack (numpy 1.26.4 /
scikit-learn 1.0.2 / lightgbm 3.3.5 / pandas 1.5.3) and through a modern one
(numpy 2.4.6 / scikit-learn 1.9.0 / lightgbm 4.7.0 / pandas 3.0.5): the
predictions are bit-identical.

So on first use we install rs3 + sglearn with ``--no-deps`` into
``$OMICOS_SYNBIO_WEIGHTS/rs3`` (default ``~/.omicverse/synbio_weights/rs3``,
~20 MB, mostly the 17 MB ``RuleSet3.pkl``) and import them from there, letting
the host environment provide modern lightgbm/pandas/joblib. ``--no-deps`` is
what makes this safe: a plain ``pip install rs3`` into an omicverse env would
try to downgrade scikit-learn to 1.0.2 and break omicverse.

rs3 is Apache-2.0, sglearn is MIT; both are fetched from PyPI at runtime, not
redistributed here.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Optional, Sequence

from ._esm_common import weights_dir

# pinned so a silent upstream change can't move the model under us
_RS3_SPEC = "rs3==0.0.18"
_SGLEARN_SPEC = "sglearn==1.2.5"


def _install_root() -> str:
    return os.path.join(weights_dir(), "rs3")


def _install_commands(root: str) -> List[List[str]]:
    """Installers to try, in order.

    ``--no-deps`` is load-bearing in both: it skips rs3's stale scikit-learn /
    numpy / lightgbm ceilings, which would otherwise downgrade — and break —
    the omicverse environment. ``--target`` keeps the whole thing out of
    site-packages.

    pip is tried first but is not always there: environments created by ``uv
    venv`` (which is how the omicOS kernel env is built) ship without pip, so
    ``python -m pip`` fails with "No module named pip". uv itself is the
    fallback for exactly that case.
    """
    specs = [_RS3_SPEC, _SGLEARN_SPEC]
    return [
        [sys.executable, "-m", "pip", "install", "--quiet",
         "--no-deps", "--target", root, *specs],
        ["uv", "pip", "install", "--quiet", "--python", sys.executable,
         "--no-deps", "--target", root, *specs],
    ]


def _rs3_importable() -> bool:
    try:
        import rs3.seq  # noqa: F401
        import sglearn  # noqa: F401
    except Exception:
        return False
    return True


def ensure_rs3() -> Optional[str]:
    """Make ``rs3`` and ``sglearn`` importable; return the install dir if we own it.

    Returns ``None`` when the host environment already provides them (nothing
    was installed). Raises an actionable ``ImportError`` if the install fails.
    """
    if _rs3_importable():
        return None

    root = _install_root()
    if os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)
        if _rs3_importable():
            return root

    os.makedirs(root, exist_ok=True)
    errors = []
    for cmd in _install_commands(root):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            break
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            errors.append(f"{cmd[0]}: {getattr(exc, 'stderr', None) or exc}")
    else:
        raise ImportError(
            "design_grnas(method='rs3') 需要 Rule Set 3 模型,自动安装失败。请手动:\n"
            f"    pip install --no-deps --target {root} {_RS3_SPEC} {_SGLEARN_SPEC}\n"
            "  (--no-deps 必须加 —— rs3 钉死 scikit-learn<=1.0.2,直接 pip install rs3 "
            "会把 scikit-learn 降级从而弄坏 omicverse。)\n"
            "  也可以改用 method='heuristic'(默认),无需任何额外安装。\n"
            "  (尝试过: " + " | ".join(errors) + ")"
        ) from None

    if root not in sys.path:
        sys.path.insert(0, root)
    if not _rs3_importable():  # pragma: no cover
        raise ImportError(
            f"rs3 已安装到 {root},但仍然导入失败。它需要宿主环境提供 lightgbm 与 "
            "seqfold:请 pip install 'omicverse[synbio]',或 pip install lightgbm seqfold。"
        )
    return root


def _load_model():
    """Load ``RuleSet3.pkl``, repairing the one attribute lightgbm 4.x needs.

    The pickle was written by lightgbm 3.x, whose ``LGBMRegressor`` had no
    ``_n_classes``; 4.x reads it in ``_process_params`` and trips over ``None``.
    For a regressor the value is 1.
    """
    from rs3.seq import load_seq_model

    model = load_seq_model()
    if getattr(model, "_n_classes", None) is None:
        model._n_classes = 1
    return model


def predict_rs3(contexts: Sequence[str], sequence_tracr: str = "Hsu2013") -> List[float]:
    """Rule Set 3 on-target scores for 30-mer *contexts* (4nt + 20nt + PAM + 3nt).

    Mirrors ``rs3.seq.predict_seq``, but goes through :func:`ensure_rs3` and the
    patched loader. Feature columns are passed in the featurizer's own order —
    do **not** reorder them by ``booster_.feature_name()``, which reports
    lightgbm-normalised names (``Hsu2013_tracr`` for ``"Hsu2013 tracr"``) and
    would raise ``KeyError``.
    """
    ensure_rs3()
    from rs3.seq import featurize_context

    model = _load_model()
    features = featurize_context(list(contexts), sequence_tracr=sequence_tracr)
    return [float(v) for v in model.predict(features)]
