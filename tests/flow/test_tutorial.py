"""Keep the committed tutorials honest.

The `ov.flow` tutorials live in the **omicverse_guide submodule**
(`omicverse_guide/docs/Tutorials-flow/`), because that is the repository the
documentation site is built from — Sphinx reads `omicverse_guide/docs`, so a
notebook sitting in this repository's `docs/` is not published anywhere. They
were briefly committed to the wrong repository; these tests now assert the
correct location, which is also what stops the mistake recurring.

CI checks out submodules (`.github/workflows/python-package.yml`), so these run
with teeth there. A developer without the submodule initialised gets skips —
`git submodule update --init omicverse_guide` if you want them locally.

The notebooks ship WITH THEIR OUTPUTS, which makes them documentation people
read without running, which means a stale or broken one teaches the wrong thing
silently. Nothing else in this repo executes notebooks (no nbmake, nbval or
papermill in .github/ or pyproject.toml), so this file is the only guard.

Defects these checks were written in response to, all of which were green under
"it ran without an exception":

* a notebook committed with **no figures at all** — every output a table —
  because the module had no plotting functions at the time;
* a `np.allclose` shape bug that printed "scatter untouched: False" when scatter
  was in fact untouched;
* a boolean gate captioned "CD3+ but not CD4+" that was in fact a tautology
  returning its own operand.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
TUTORIALS = ROOT / "omicverse_guide" / "docs" / "Tutorials-flow"
SIMULATED = [
    "t_flow_01_reading_and_compensation.ipynb",
    "t_flow_02_gating.ipynb",
    "t_flow_03_gatingml.ipynb",
    "t_flow_04_flowsom.ipynb",
]
#: Runs on real downloaded data, so it is excluded from the re-execution test:
#: that would pull ~100 MB from Zenodo and PLOS on every CI run, and a red
#: build caused by someone else's server being down teaches nothing.
REAL_DATA = ["t_flow_05_real_data.ipynb"]
NOTEBOOKS = SIMULATED + REAL_DATA

HAS_SUBMODULE = (ROOT / "omicverse_guide" / "docs").is_dir()
HAS_NBCLIENT = importlib.util.find_spec("nbclient") is not None
HAS_FLOWIO = importlib.util.find_spec("flowio") is not None

needs_submodule = pytest.mark.skipif(
    not HAS_SUBMODULE,
    reason="omicverse_guide submodule not checked out "
           "(git submodule update --init omicverse_guide)",
)


def _load(name):
    return json.loads((TUTORIALS / name).read_text())


def _text(nb, kind):
    return " ".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == kind
    )


@needs_submodule
def test_the_tutorials_are_in_the_guide_submodule():
    """Not in this repository's docs/. That directory holds developer notes and
    is not part of the documentation build, so a tutorial there is invisible."""
    assert TUTORIALS.is_dir(), f"{TUTORIALS} is missing"
    missing = [n for n in NOTEBOOKS if not (TUTORIALS / n).exists()]
    assert not missing, f"missing tutorials: {missing}"


def test_no_tutorial_notebook_was_left_behind_in_this_repo():
    """Runs with or without the submodule: the misplacement this guards against
    is a file appearing HERE, which needs no submodule to detect.

    Deliberately not scoped to `t_flow*` — it has happened twice, to synbio
    (PR #887) and then to flow (PR #907), and a guard that only knows about the
    last occurrence would not have caught either of them in advance. Any
    `t_*.ipynb` under this repository's docs/ is in the wrong repository:
    the documentation site is built by `sphinx-build omicverse_guide/docs`
    (.github/workflows/deploy-docs.yml), and docs/ here is gitignored developer
    notes. Module OVERVIEWS (`ov_*.md`) do belong here; notebooks do not.
    """
    strays = sorted(p.name for p in (ROOT / "docs").glob("t_*.ipynb"))
    assert not strays, (
        f"{strays} is in the main repo's docs/, which the documentation build "
        "does not read. Tutorials belong in the omicverse_guide submodule, as "
        "omicverse_guide/docs/Tutorials-<domain>/, registered in Tutorial.md "
        "and in tutorials/index_<domain>.md"
    )


@needs_submodule
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_no_cell_errored_when_it_was_executed(name):
    """The committed outputs must not contain a traceback. This catches someone
    committing a notebook they ran after breaking the API."""
    bad = [
        (i, out.get("ename"))
        for i, cell in enumerate(_load(name)["cells"])
        for out in (cell.get("outputs") or [])
        if out.get("output_type") == "error"
    ]
    assert not bad, f"{name}: error outputs in committed notebook: {bad}"


@needs_submodule
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_every_code_cell_actually_ran(name):
    """An un-run cell means the outputs below it describe a different state than
    the code above it — worse than no outputs at all."""
    code = [c for c in _load(name)["cells"] if c["cell_type"] == "code"]
    assert code, f"{name}: no code cells"
    unrun = [i for i, c in enumerate(code) if c.get("execution_count") is None]
    assert not unrun, f"{name}: code cells never executed: {unrun}"


@needs_submodule
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_outputs_are_in_execution_order(name):
    """Out-of-order counts mean cells were re-run piecemeal, so the narrative
    and the numbers no longer correspond."""
    counts = [c["execution_count"] for c in _load(name)["cells"]
              if c["cell_type"] == "code" and c.get("execution_count")]
    assert counts == sorted(counts), f"{name}: execution counts out of order: {counts}"


@needs_submodule
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_tutorial_shows_figures(name):
    """The one that would have caught the original notebook.

    Gating is a visual procedure: a reader cannot tell whether a gate is in the
    right place from a table of counts. A flow-cytometry tutorial with no image
    outputs is not a flow-cytometry tutorial, and every check that existed
    before this one passed on exactly that.
    """
    images = sum(
        1
        for cell in _load(name)["cells"]
        for out in (cell.get("outputs") or [])
        for key in (out.get("data") or {})
        if key.startswith("image/")
    )
    assert images >= 2, f"{name}: only {images} figure(s) in a cytometry tutorial"


@needs_submodule
def test_the_series_covers_the_module():
    """Each notebook must actually reach the part of ov.flow it claims to."""
    expected = {
        "t_flow_01_reading_and_compensation.ipynb": ["read_fcs", "compensate",
                                                     "Logicle", "spillover_heatmap"],
        "t_flow_02_gating.ipynb": ["GatingStrategy", "PolygonGate", "QuadrantGate",
                                   "BooleanGate", "hierarchy", "backgate"],
        "t_flow_03_gatingml.ipynb": ["write_gatingml", "read_gatingml", "from_dict"],
        "t_flow_04_flowsom.ipynb": ["flowsom", "flowsom_heatmap"],
        "t_flow_05_real_data.ipynb": ["flow_pbmc_fortessa", "flow_pbmc_spectral",
                                      "compensate", "GatingStrategy",
                                      "write_gatingml", "flowsom", "PnR"],
    }
    for name, symbols in expected.items():
        src = _text(_load(name), "code")
        missing = [s for s in symbols if s not in src]
        assert not missing, f"{name} never calls: {missing}"


@needs_submodule
@pytest.mark.parametrize("name", SIMULATED)
def test_it_says_the_data_is_simulated(name):
    """A tutorial whose numbers look like a real experiment must say plainly
    that they are not."""
    text = _text(_load(name), "markdown").lower()
    assert "simulated" in text or "synthetic" in text, f"{name}"


@needs_submodule
@pytest.mark.parametrize("name", REAL_DATA)
def test_the_real_data_notebook_cites_its_sources(name):
    """Both datasets are CC-BY-4.0, and attribution is a licence CONDITION, not
    a courtesy. If the citation ever falls out of the notebook, redistribution
    of the figures stops being compliant — so it is a test, not a convention."""
    text = _text(_load(name), "markdown")
    for required in ("10.5281/zenodo.14311616", "10.1371/journal.pone.0351131",
                     "CC-BY-4.0"):
        assert required in text, f"{name} does not carry {required}"


@needs_submodule
@pytest.mark.parametrize("name", REAL_DATA)
def test_the_real_data_notebook_reads_the_top_of_scale_from_the_file(name):
    """Its central lesson. A hard-coded 262144 is exactly the bug it warns
    about, and the Cytek instrument in the same notebook runs to 4,194,304."""
    src = _text(_load(name), "code")
    assert "PnR" in src, f"{name}: never reads $PnR"
    assert "t=262144" not in src.replace(" ", ""), (
        f"{name} hard-codes a top of scale while telling the reader not to"
    )


@needs_submodule
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_it_only_uses_the_public_api(name):
    """A tutorial reaching into a private module teaches an API that can be
    renamed without notice."""
    src = _text(_load(name), "code")
    assert "ov.flow." in src, f"{name}: never touches ov.flow"
    for private in ("._transforms", "._gates", "._strategy", "._compensate",
                    "._gatingml", "._som", "._cluster"):
        assert private not in src, f"{name} reaches into {private}"


@needs_submodule
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_each_notebook_stands_alone(name):
    """A reader who lands on notebook 3 from a search must not have to run 1
    first, so each one loads its own data."""
    src = _text(_load(name), "code")
    assert "ov.datasets.flow_" in src, f"{name}: no data of its own"


@needs_submodule
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_demo_data_comes_from_ov_datasets(name):
    """Not from a generator pasted into the notebook.

    It was inline in all four to begin with — forty lines of simulation the
    reader had to scroll past before reaching any cytometry, repeated verbatim,
    and four places for it to drift out of sync with the module it feeds.
    """
    src = _text(_load(name), "code")
    for smell in ("def synthesise", "def write_demo", "flowio.create_fcs"):
        assert smell not in src, (
            f"{name} carries its own data generator ({smell}); "
            "use ov.datasets.flow_demo / flow_demo_fcs"
        )


@needs_submodule
@pytest.mark.skipif(not (HAS_NBCLIENT and HAS_FLOWIO),
                    reason="needs nbclient + flowio to re-execute")
@pytest.mark.parametrize("name", SIMULATED)
def test_the_notebook_still_runs(name, tmp_path, monkeypatch):
    """Re-execute end to end. These notebooks are the only thing in the repo
    exercising read_fcs -> compensate -> transform -> gate -> plot -> GatingML
    -> FlowSOM as one chain; the unit tests each cover a link, not the chain.

    tmp_path as the working directory on purpose: they write .fcs and .xml
    files, and none of those belong beside the notebooks.

    The kernel is a subprocess, so it does not inherit the sys.path pytest set
    up — it resolves `omicverse` however the environment happens to. On a
    machine with more than one checkout installed that is silently the WRONG
    one, and the failure surfaces as a baffling AttributeError twelve cells in.
    Pinning PYTHONPATH makes the test mean "the tutorial runs against THIS
    source", which is the only version of the question worth asking.
    """
    import nbformat
    from nbclient import NotebookClient

    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", str(ROOT) + (os.pathsep + existing if existing else "")
    )

    nb = nbformat.read(str(TUTORIALS / name), as_version=4)
    nb.cells.insert(0, nbformat.v4.new_code_cell(
        "import omicverse as ov, pathlib\n"
        f"assert pathlib.Path(ov.__file__).resolve().parents[1] == "
        f"pathlib.Path({str(ROOT)!r}).resolve(), (\n"
        "    'the kernel imported a different omicverse checkout: ' + ov.__file__)\n"
        "assert hasattr(ov.io, 'read_fcs')\n"
    ))
    client = NotebookClient(nb, timeout=900, kernel_name="python3",
                            resources={"metadata": {"path": str(tmp_path)}})
    client.execute()
