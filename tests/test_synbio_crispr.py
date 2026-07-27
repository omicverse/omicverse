"""Guard rails around ``ov.synbio.design_grnas`` and the ``[synbio]`` extra.

``rs3`` (Rule Set 3) pins ``scikit-learn<=1.0.2`` while omicverse requires
``>=1.2``, and 0.0.18 is its only release satisfying ``>=0.0.18`` — so a
resolver cannot back off. Listing it in the ``synbio`` extra made
``pip install "omicverse[synbio]"`` fail with ResolutionImpossible on every
Python version, taking all 19 sibling dependencies down with it. These tests
keep it out and keep the fallback path honest.
"""
import re
from pathlib import Path

import pytest

import omicverse as ov


# A short target with 10-nt flanks so guides get a full 30-mer rs3 context.
TARGET = "GCATGCATGC" + "ATGGCTAGCTAGGATCCATCGATCGGGCTAAACCGGTTAGCTAGCTTGACC" + "GCATGCATGC"


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


def test_rs3_method_raises_actionable_error(monkeypatch):
    """``method='rs3'`` must explain the conflict, not just say 'pip install rs3'.

    Naively installing rs3 into an omicverse env downgrades scikit-learn to
    1.0.2 and breaks omicverse, so the message has to steer users to a separate
    environment and to the dependency-free default.
    """
    import builtins

    real_import = builtins.__import__

    def _no_rs3(name, *args, **kwargs):
        if name == "rs3" or name.startswith("rs3."):
            raise ImportError("blocked import: rs3")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_rs3)

    with pytest.raises(ImportError) as excinfo:
        ov.synbio.design_grnas(TARGET, method="rs3")

    msg = str(excinfo.value)
    assert "scikit-learn<=1.0.2" in msg, "must name the pin that causes the conflict"
    assert "method='heuristic'" in msg, "must offer the dependency-free fallback"
    assert "venv" in msg, "must point at a separate environment, not the current one"


def test_rs3_stays_out_of_the_synbio_extra():
    """Regression guard: re-adding rs3 makes the whole extra uninstallable."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("running against an installed omicverse, no pyproject.toml")

    body = pyproject.read_text(encoding="utf-8")
    extra = body.split("synbio = [", 1)[1].split("\n]", 1)[0]
    requirements = re.findall(r'^\s*"([^"]+)"', extra, re.M)

    assert requirements, "failed to parse the synbio extra — did the block move?"
    offenders = [r for r in requirements if re.match(r"rs3\b", r)]
    assert not offenders, (
        f"rs3 is back in the synbio extra ({offenders}). It pins "
        "scikit-learn<=1.0.2 against omicverse's >=1.2, which makes "
        '`pip install "omicverse[synbio]"` unresolvable for every dependency in '
        "the extra. Keep it out; design_grnas raises an actionable ImportError."
    )
