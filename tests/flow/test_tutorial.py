"""Keep the committed tutorial honest.

`docs/t_flow.ipynb` ships WITH ITS OUTPUTS, following `docs/t_synbio.ipynb`. That
makes it documentation people read without running — which means a stale or
broken notebook teaches the wrong thing silently.

Nothing in this repo executes notebooks (no nbmake, nbval or papermill anywhere
in .github/ or pyproject.toml), so these are the only guard. The cheap checks
run everywhere; the full re-execution runs wherever the toolchain is present.

Writing this notebook already caught two defects in itself that no unit test
would have: a shape bug in a `np.allclose` check that made it print
"scatter untouched: False" when scatter was in fact untouched, and a
before/after comparison that printed the wrong column under a label claiming
otherwise.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

NB = pathlib.Path(__file__).resolve().parents[2] / "docs" / "t_flow.ipynb"

HAS_NBFORMAT = importlib.util.find_spec("nbformat") is not None
HAS_NBCLIENT = importlib.util.find_spec("nbclient") is not None
HAS_FLOWIO = importlib.util.find_spec("flowio") is not None


def test_the_tutorial_exists():
    assert NB.exists(), "ov.flow ships a namespace; it ships a tutorial with it"


def test_no_cell_errored_when_it_was_executed():
    """The committed outputs must not contain a traceback. This is what catches
    someone committing a notebook they ran after breaking the API."""
    nb = json.loads(NB.read_text())
    bad = []
    for i, cell in enumerate(nb["cells"]):
        for out in cell.get("outputs", []) or []:
            if out.get("output_type") == "error":
                bad.append((i, out.get("ename")))
    assert not bad, f"error outputs in committed notebook: {bad}"


def test_every_code_cell_actually_ran():
    """An un-run cell means the outputs below it describe a different state than
    the code above it — worse than no outputs at all."""
    nb = json.loads(NB.read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code, "no code cells"
    unrun = [i for i, c in enumerate(code) if c.get("execution_count") is None]
    assert not unrun, f"code cells never executed: {unrun}"


def test_the_outputs_are_in_execution_order():
    """Out-of-order counts mean cells were re-run piecemeal, so the narrative
    and the numbers no longer correspond."""
    nb = json.loads(NB.read_text())
    counts = [c["execution_count"] for c in nb["cells"]
              if c["cell_type"] == "code" and c.get("execution_count")]
    assert counts == sorted(counts), f"execution counts out of order: {counts}"


def test_it_says_the_data_is_simulated():
    """A tutorial whose numbers look like a real experiment must say plainly
    that they are not."""
    text = " ".join(
        "".join(c["source"]) for c in json.loads(NB.read_text())["cells"]
        if c["cell_type"] == "markdown"
    ).lower()
    assert "simulated" in text or "synthetic" in text


def test_it_only_uses_the_public_api():
    """A tutorial reaching into a private module is teaching an API that can be
    renamed without notice."""
    src = " ".join(
        "".join(c["source"]) for c in json.loads(NB.read_text())["cells"]
        if c["cell_type"] == "code"
    )
    assert "ov.flow." in src
    assert "._transforms" not in src and "._gates" not in src and "._strategy" not in src


@pytest.mark.skipif(not (HAS_NBCLIENT and HAS_FLOWIO),
                    reason="needs nbclient + flowio to re-execute")
def test_the_notebook_still_runs(tmp_path):
    """Re-execute end to end. The notebook is the only thing in the repo that
    exercises read_fcs -> compensate -> transform -> gate -> GatingML -> FlowSOM
    as one pipeline; the unit tests each cover a link, not the chain."""
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(NB, as_version=4)
    client = NotebookClient(nb, timeout=900, kernel_name="python3",
                            resources={"metadata": {"path": str(tmp_path)}})
    client.execute()
