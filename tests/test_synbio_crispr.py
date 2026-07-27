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


# ---------------------------------------------------------------------------
# Ranking semantics: rs3 is a z-score, ``efficiency`` is a [0,1] heuristic,
# and the two must never be sorted against each other.
#
# ``TARGET`` above carries 10-nt flanks so every guide gets a 30-mer context.
# These tests need the opposite: a target whose end-adjacent guides *cannot* be
# scored, because that is where the scales used to collide.
# ---------------------------------------------------------------------------
RANKING_TARGET = (
    "GGCCATGGAGTCTAGGACTTCAGGTACCGGATCAGGCTAACGGTTAGGCCATTAGGACGTTAGG"
    "ACCTTGGCATGCAGGTTACCGGAATTCAGGCCTTAAGGCATTCAGGTTAACGGCCATTAGGTAC"
)


@pytest.fixture
def stub_rs3(monkeypatch):
    """Replace ``_rs3.predict_rs3`` with a stub returning negative z-scores.

    Score = -(index + 1) * 0.5, so the *first* context handed to the model gets
    the highest (least negative) score. Every value is < 0 — the regime real
    Rule Set 3 lives in, and the one that used to let unscored heuristic guides
    float to the top of an rs3 ranking.

    Patching here also keeps ``ensure_rs3()`` out of the test: no ``--no-deps``
    pip fetch, no ~20 MB model download, no lightgbm import.
    """
    calls = {}

    def predict_rs3(contexts, sequence_tracr="Hsu2013"):
        calls["contexts"] = list(contexts)
        calls["tracr"] = sequence_tracr
        return [-(i + 1) * 0.5 for i in range(len(contexts))]

    monkeypatch.setattr(_rs3, "predict_rs3", predict_rs3)
    return calls


def test_heuristic_leaves_rs3_score_unset():
    guides = ov.synbio.design_grnas(RANKING_TARGET)
    assert guides
    assert all(g.rs3_score is None for g in guides)


def test_rs3_score_populated_only_where_context_exists(stub_rs3):
    for g in ov.synbio.design_grnas(RANKING_TARGET, method="rs3"):
        assert (g.rs3_score is not None) == (len(g.context) == 30)


def test_rs3_does_not_clobber_heuristic_efficiency(stub_rs3):
    """The regression: ``efficiency`` used to *become* the z-score."""
    heur = {(g.spacer, g.start, g.strand): g.efficiency
            for g in ov.synbio.design_grnas(RANKING_TARGET)}
    for g in ov.synbio.design_grnas(RANKING_TARGET, method="rs3"):
        assert 0.0 <= g.efficiency <= 1.0
        assert g.efficiency == pytest.approx(heur[(g.spacer, g.start, g.strand)])


def test_rs3_ranking_is_by_z_score_and_tolerates_negatives(stub_rs3):
    guides = ov.synbio.design_grnas(RANKING_TARGET, method="rs3")
    scored = [g for g in guides if g.rs3_score is not None]
    assert scored, "stub should have scored at least one guide"
    assert all(g.rs3_score < 0 for g in scored), "stub returns negative z-scores"
    zs = [g.rs3_score for g in scored]
    assert zs == sorted(zs, reverse=True)


def test_unscorable_guides_rank_last(stub_rs3):
    """A top-heuristic guide with no context must not outrank a real rs3 hit.

    ``RANKING_TARGET`` has guides too close to both ends to yield a 30-mer, and
    one of them carries the *highest* heuristic efficiency in the pool — under
    the old scale-mixing sort it landed at rank 0, ahead of every guide the
    model actually scored.
    """
    guides = ov.synbio.design_grnas(RANKING_TARGET, method="rs3")
    unscored = [g for g in guides if g.rs3_score is None]
    scored = [g for g in guides if g.rs3_score is not None]
    assert unscored and scored, "target must exercise both cases"

    n = len(scored)
    assert all(g.rs3_score is not None for g in guides[:n]), "scored guides lead"
    assert all(g.rs3_score is None for g in guides[n:]), "unscored form the tail"

    # The trap: an unscored guide beats every scored one on the heuristic scale.
    assert max(g.efficiency for g in unscored) >= max(g.efficiency for g in scored)


def test_no_fabricated_flanks_reach_the_model(stub_rs3):
    """Contexts are real 30-mers taken from the input, never padded out."""
    ov.synbio.design_grnas(RANKING_TARGET, method="rs3")
    revcomp = RANKING_TARGET.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]
    for ctx in stub_rs3["contexts"]:
        assert len(ctx) == 30
        assert set(ctx) <= set("ACGT")
        assert ctx in RANKING_TARGET or ctx in revcomp


def test_tracr_passed_through(stub_rs3):
    ov.synbio.design_grnas(RANKING_TARGET, method="rs3")
    assert stub_rs3["tracr"] == "Hsu2013"


def test_top_n_applies_after_the_rs3_sort(stub_rs3):
    full = ov.synbio.design_grnas(RANKING_TARGET, method="rs3")
    top = ov.synbio.design_grnas(RANKING_TARGET, method="rs3", top_n=3)
    assert len(top) == 3
    assert [g.spacer for g in top] == [g.spacer for g in full[:3]]


def test_rs3_failure_is_not_silently_downgraded(monkeypatch):
    """An unavailable rs3 raises — it does not quietly fall back to heuristic.

    A silent fallback would hand back a heuristic ranking under the name the
    caller asked rs3 for: the same scale confusion, just hidden better.
    """
    def boom(contexts, sequence_tracr="Hsu2013"):
        raise ImportError("rs3 install failed")

    monkeypatch.setattr(_rs3, "predict_rs3", boom)
    with pytest.raises(ImportError, match="rs3"):
        ov.synbio.design_grnas(RANKING_TARGET, method="rs3")


def test_guide_rs3_score_defaults_to_none():
    from omicverse.synbio._crispr import Guide

    g = Guide(spacer="A" * 20, pam="AGG", start=0, strand="+", gc=0.0,
              efficiency=0.5, poly_t=False)
    assert g.rs3_score is None


def test_guide_repr_shows_rs3_only_when_scored():
    from omicverse.synbio._crispr import Guide

    g = Guide(spacer="A" * 20, pam="AGG", start=0, strand="+", gc=0.0,
              efficiency=0.5, poly_t=False)
    assert "rs3" not in repr(g)
    g.rs3_score = -1.25
    assert "rs3=-1.25" in repr(g)
