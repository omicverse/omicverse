"""Guard rails around ``ov.synbio.design_grnas`` and the ``[synbio]`` extra.

``rs3`` (Rule Set 3) pins ``scikit-learn<=1.0.2`` while omicverse requires
``>=1.2``, and 0.0.18 is its only release satisfying ``>=0.0.18`` — so a
resolver cannot back off. Listing it in the ``synbio`` extra made
``pip install "omicverse[synbio]"`` fail with ResolutionImpossible on every
Python version, taking all its sibling dependencies down with it. It is now
fetched at runtime with ``--no-deps`` instead (see ``omicverse/synbio/_rs3.py``).
These tests keep it out of the extra, keep the runtime path honest, and keep
the dependency-free fallback working.

Nothing here touches the network: the install step is always monkeypatched.
"""
import re
import sys
import types
from pathlib import Path

import pytest

import omicverse as ov
from omicverse.synbio import _rs3


# A short target with 10-nt flanks so guides get a full 30-mer rs3 context.
TARGET = "GCATGCATGC" + "ATGGCTAGCTAGGATCCATCGATCGGGCTAAACCGGTTAGCTAGCTTGACC" + "GCATGCATGC"


def _synbio_extra_requirements():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("running against an installed omicverse, no pyproject.toml")
    extra = pyproject.read_text(encoding="utf-8").split("synbio = [", 1)[1].split("\n]", 1)[0]
    reqs = re.findall(r'^\s*"([^"]+)"', extra, re.M)
    assert reqs, "failed to parse the synbio extra — did the block move?"
    return reqs


def test_design_grnas_heuristic_needs_no_optional_dependency():
    """The default path must work on a bare install — no synbio extra at all."""
    guides = ov.synbio.design_grnas(TARGET)
    assert guides, "expected at least one SpCas9 guide in the target"
    top = guides[0]
    assert len(top.spacer) == 20
    assert 0.0 <= top.efficiency <= 1.0
    assert guides == sorted(guides, key=lambda g: g.efficiency, reverse=True)


def test_design_grnas_rejects_unknown_method():
    with pytest.raises(ValueError, match="heuristic"):
        ov.synbio.design_grnas(TARGET, method="azimuth")


def test_rs3_stays_out_of_the_synbio_extra():
    """Regression guard: re-adding rs3 makes the whole extra uninstallable."""
    offenders = [r for r in _synbio_extra_requirements() if re.match(r"rs3\b", r)]
    assert not offenders, (
        f"rs3 is back in the synbio extra ({offenders}). It pins "
        "scikit-learn<=1.0.2 against omicverse's >=1.2, which makes "
        '`pip install "omicverse[synbio]"` unresolvable for every dependency in '
        "the extra. It is fetched at runtime by omicverse/synbio/_rs3.py instead."
    )


def test_extra_carries_what_the_runtime_fetch_needs():
    """``--no-deps`` means the host env must supply rs3/sglearn's real deps."""
    reqs = _synbio_extra_requirements()
    for pkg in ("lightgbm", "seqfold"):
        assert any(re.match(rf"{pkg}\b", r) for r in reqs), (
            f"{pkg} missing from the synbio extra — _rs3.py installs rs3/sglearn "
            "with --no-deps, so nothing else will pull it in and method='rs3' "
            "would fail at import time."
        )


def test_install_commands_cover_pip_and_uv_and_always_pass_no_deps():
    """uv-created venvs (including the omicOS kernel env) ship without pip.

    Found the hard way: with only the ``python -m pip`` path, method='rs3'
    died with "No module named pip" on any uv-built environment.
    """
    cmds = _rs3._install_commands("/tmp/whatever")
    assert len(cmds) >= 2, "need a fallback installer, not just pip"
    assert cmds[0][1:4] == ["-m", "pip", "install"], "pip should be tried first"
    assert cmds[1][0] == "uv", "uv is the fallback for pip-less environments"
    for cmd in cmds:
        assert "--no-deps" in cmd, (
            "--no-deps must never be dropped: without it the installer "
            "downgrades scikit-learn to 1.0.2 and breaks omicverse"
        )
        assert "--target" in cmd, "must install out-of-tree, not into site-packages"


def test_ensure_rs3_falls_back_to_uv_when_pip_is_missing(monkeypatch):
    monkeypatch.setattr(_rs3, "_rs3_importable", lambda: False)
    import subprocess

    attempted = []

    def _fake_run(cmd, **kwargs):
        attempted.append(cmd[0] if cmd[0] == "uv" else "pip")
        if attempted[-1] == "pip":
            raise subprocess.CalledProcessError(1, cmd, stderr="No module named pip")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    # the post-install import check would fail with a fake installer; skip past it
    monkeypatch.setattr(_rs3, "_rs3_importable", lambda: len(attempted) >= 2)

    _rs3.ensure_rs3()
    assert attempted == ["pip", "uv"], f"expected pip then uv, got {attempted}"


def test_ensure_rs3_failure_is_actionable(monkeypatch):
    """A failed install must not leave the user guessing (or wrecking their env)."""
    import subprocess

    monkeypatch.setattr(_rs3, "_rs3_importable", lambda: False)

    def _boom(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "pip", stderr="network unreachable")

    monkeypatch.setattr(subprocess, "run", _boom)

    with pytest.raises(ImportError) as excinfo:
        _rs3.ensure_rs3()

    msg = str(excinfo.value)
    assert "--no-deps" in msg, "the manual recipe must keep --no-deps or it breaks the env"
    assert "method='heuristic'" in msg, "must offer the dependency-free fallback"
    assert "scikit-learn<=1.0.2" in msg, "must say why --no-deps is mandatory"


def test_ensure_rs3_is_a_noop_when_host_already_has_it(monkeypatch):
    import subprocess

    monkeypatch.setattr(_rs3, "_rs3_importable", lambda: True)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("must not install"))
    assert _rs3.ensure_rs3() is None


def test_load_model_repairs_the_lightgbm_3_pickle(monkeypatch):
    """RuleSet3.pkl was written by lightgbm 3.x; 4.x trips over _n_classes=None."""
    class _FakeRegressor:
        _n_classes = None

    fake_seq = types.ModuleType("rs3.seq")
    fake_seq.load_seq_model = lambda: _FakeRegressor()
    fake_rs3 = types.ModuleType("rs3")
    fake_rs3.seq = fake_seq
    monkeypatch.setitem(sys.modules, "rs3", fake_rs3)
    monkeypatch.setitem(sys.modules, "rs3.seq", fake_seq)

    assert _rs3._load_model()._n_classes == 1


def test_load_model_leaves_a_healthy_pickle_alone(monkeypatch):
    class _FakeRegressor:
        _n_classes = 3          # would be wrong to clobber

    fake_seq = types.ModuleType("rs3.seq")
    fake_seq.load_seq_model = lambda: _FakeRegressor()
    fake_rs3 = types.ModuleType("rs3")
    fake_rs3.seq = fake_seq
    monkeypatch.setitem(sys.modules, "rs3", fake_rs3)
    monkeypatch.setitem(sys.modules, "rs3.seq", fake_seq)

    assert _rs3._load_model()._n_classes == 3
