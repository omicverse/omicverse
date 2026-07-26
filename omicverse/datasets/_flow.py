r"""A **simulated** flow-cytometry sample for the ``ov.flow`` tutorials.

Everything else in ``ov.datasets`` is real published data. This is not, and the
docstrings say so in every place someone might read only one of them — a
cytometry matrix that looks like a PBMC panel is exactly the kind of thing that
gets cited by accident.

WHY IT IS SIMULATED RATHER THAN A DOWNLOAD
    A real FCS file cannot teach compensation, because you cannot show what the
    spillover did — the uncompensated truth is not recoverable once you only
    have the file. Here the *true* fluorescence is known, the spillover matrix
    is applied on purpose, and the tutorial can put the before and after side by
    side and quote the number. The same goes for gating: ``obs['population']``
    carries which population each event really came from, so a gate can be
    checked against the answer rather than against how the plot looks. Real data
    has no such column, which is the whole difficulty of the field.

WHAT MAKES IT REALISTIC ENOUGH TO BE WORTH ANYTHING
    Nine populations at plausible frequencies, including the rare ones that
    break naive analyses: 1,400 double-negative and 500 double-positive T cells.
    Doublets with the pulse geometry that actually distinguishes them — same
    area, ~half the height — rather than a label. A detector noise floor added
    *after* spillover, so around 11% of compensated fluorescence values are at or
    below zero. That last one is not a detail: it is why a log axis cannot
    display cytometry data, and a simulation without it makes logicle look like
    a complication rather than a fix.
"""
from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd

from .._registry import register_function

if TYPE_CHECKING:  # pragma: no cover
    from anndata import AnnData

__all__ = ["flow_demo", "flow_demo_fcs", "FLOW_DEMO_PANEL", "FLOW_DEMO_SPILLOVER"]

#: Detector names, in file order. Scatter first, then fluorescence.
FLOW_DEMO_PANEL: Tuple[str, ...] = (
    "FSC-A", "FSC-H", "SSC-A", "FITC-A", "PE-A", "PerCP-A", "APC-A", "Zombie-A",
)

#: The marker on each detector (``$PnS``); ``None`` for scatter.
FLOW_DEMO_MARKERS: Tuple = (
    None, None, None, "CD3", "CD4", "CD19", "CD8", "Viability",
)

_FLUORS: Tuple[str, ...] = ("FITC-A", "PE-A", "PerCP-A", "APC-A")

#: Row = fluorochrome, column = detector. FITC into PE and PerCP into APC are
#: the classic pairs; magnitudes are in the 3-18% range real panels show.
FLOW_DEMO_SPILLOVER = pd.DataFrame(
    np.array([
        [1.00, 0.18, 0.06, 0.01],
        [0.03, 1.00, 0.11, 0.01],
        [0.01, 0.04, 1.00, 0.14],
        [0.00, 0.01, 0.03, 1.00],
    ]),
    index=list(_FLUORS),
    columns=list(_FLUORS),
)

#: name, n, FSC, SSC, CD3, CD4, CD19, CD8, viability, is-doublet
_POPULATIONS = (
    ("debris",  11000,  17000,  8500,   25,   22,   20,    24,    55, False),
    ("CD4 T",   17000,  78000, 24000, 7500, 9500,   22,    26,    60, False),
    ("CD8 T",   10000,  76000, 23000, 8000,   24,   21,  7000,    58, False),
    ("DN T",     1400,  74000, 22000, 6500,   26,   23,    28,    62, False),
    ("DP T",      500,  79000, 25000, 7200, 4200,   22,  3500,    61, False),
    ("B",        7500,  82000, 28000,   28,   25, 6500,    25,    64, False),
    ("mono",     5200, 142000, 58000,   30,  900,   27,    30,    70, False),
    ("doublet",  3800,  80000, 26000, 6800, 8200,   24,    27,    60, True),
    ("dead",     3200,  76000, 24000, 5200, 6000,   30,    26,  7500, False),
)

_NOISE_SD = 55.0


def _simulate(seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Events and their true population labels, in file order."""
    rng = np.random.default_rng(seed)

    def block(n, fsc, ssc, cd3, cd4, cd19, cd8, via, doublet):
        area = rng.normal(fsc, fsc * 0.09, n)
        # A doublet has two cells' worth of signal in one pulse: the same
        # integrated area, spread over roughly twice the time, so the HEIGHT is
        # about half. That geometry — not a label — is what a singlet gate uses.
        height = area / (1.85 if doublet else 1.0) + rng.normal(0, fsc * 0.025, n)
        side = rng.normal(ssc, ssc * 0.13, n)

        def channel(level):
            # Negatives are a wide smear against the noise floor, not a tight
            # cluster: an unstained population has nothing holding it together.
            sigma = 0.42 if level > 300 else 0.85
            return rng.lognormal(np.log(level), sigma, n)

        return (
            np.column_stack([area, height, side]),
            np.column_stack([channel(cd3), channel(cd4), channel(cd19), channel(cd8)]),
            channel(via),
        )

    scatter, true_fl, viability, labels = [], [], [], []
    for name, n, *params in _POPULATIONS:
        s, f, v = block(n, *params)
        scatter.append(s)
        true_fl.append(f)
        viability.append(v)
        labels.append(np.repeat(name, n))

    scatter = np.vstack(scatter)
    n_events = scatter.shape[0]

    # Spillover first, THEN the noise floor. The other order would let
    # compensation remove the noise too, and the negative values that make this
    # sample worth having would not survive.
    observed = np.vstack(true_fl) @ FLOW_DEMO_SPILLOVER.to_numpy()
    observed = observed + rng.normal(0, _NOISE_SD, observed.shape)
    via = np.concatenate(viability) + rng.normal(0, _NOISE_SD, n_events)

    X = np.column_stack([scatter, observed, via])
    truth = np.concatenate(labels)

    # Shuffle: a file whose events arrive grouped by population would let a
    # reader "gate" by row index.
    order = rng.permutation(n_events)
    return X[order], truth[order]


@register_function(
    aliases=[
        "flow_demo_fcs", "write_flow_demo", "demo_fcs", "示例FCS", "模拟流式文件",
        "simulated_fcs", "flow_cytometry_demo_file",
    ],
    category="datasets",
    description=(
        "Write a SIMULATED flow-cytometry FCS 3.1 file for the ov.flow "
        "tutorials — not real data. 59,600 events x 8 channels (FSC-A, FSC-H, "
        "SSC-A and five fluorescence detectors carrying CD3 / CD4 / CD19 / CD8 "
        "/ viability), nine populations including rare double-negative and "
        "double-positive T cells, doublets with realistic pulse geometry "
        "(same area, ~half the height), a genuine $SPILLOVER keyword, and a "
        "detector noise floor that puts ~11% of compensated fluorescence at or "
        "below zero. Returns the path. Use ov.datasets.flow_demo() to get the "
        "same sample already read into an AnnData."
    ),
    examples=[
        "path = ov.datasets.flow_demo_fcs()",
        "adata = ov.io.read_fcs(ov.datasets.flow_demo_fcs('demo.fcs'))",
    ],
    related=["datasets.flow_demo", "io.read_fcs", "flow.compensate"],
)
def flow_demo_fcs(path: str = "flow_demo.fcs", *, seed: int = 0) -> str:
    """Write the simulated cytometry sample as a real FCS 3.1 file.

    This is the entry point the tutorials use, because a cytometry tutorial that
    hands you an AnnData has skipped the part where the file format is the
    problem — the detector-versus-marker naming and the ``$SPILLOVER`` keyword
    both live in the file and nowhere else.

    Arguments
    ---------
    path
        Where to write. Relative paths land in the current working directory.
    seed
        Changes the events, not the populations or their frequencies.

    Returns the path, so it composes: ``ov.io.read_fcs(ov.datasets.flow_demo_fcs())``.
    """
    try:
        import flowio
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "writing an FCS file needs flowio (the same dependency "
            "ov.io.read_fcs uses): pip install flowio"
        ) from exc

    X, _ = _simulate(seed=seed)
    spill = FLOW_DEMO_SPILLOVER
    keyword = ",".join(
        [str(spill.shape[0])]
        + list(spill.columns)
        + [f"{v:g}" for v in spill.to_numpy().ravel()]
    )
    with open(path, "wb") as fh:
        flowio.create_fcs(
            fh,
            X.ravel().tolist(),
            channel_names=list(FLOW_DEMO_PANEL),
            opt_channel_names=list(FLOW_DEMO_MARKERS),
            metadata_dict={"SPILLOVER": keyword, "CYT": "omicverse-demo"},
        )
    return path


@register_function(
    aliases=[
        "flow_demo", "demo_cytometry", "示例流式数据", "模拟流式数据",
        "simulated_cytometry", "flow_cytometry_demo",
    ],
    category="datasets",
    description=(
        "SIMULATED flow-cytometry sample as an AnnData for the ov.flow "
        "tutorials and tests — not real data. 59,600 events x 8 channels, nine "
        "populations (debris, CD4/CD8/DN/DP T cells, B cells, monocytes, "
        "doublets, dead cells), spillover applied and recorded in "
        "uns['fcs']['spillover'], and a noise floor that produces the negative "
        "values a biexponential display scale exists to show. obs['population'] "
        "carries the true origin of every event, which real data never has — it "
        "is what lets a tutorial check a gate against the answer."
    ),
    examples=[
        "adata = ov.datasets.flow_demo()",
        "adata.obs['population'].value_counts()",
        "ov.flow.compensate(adata)",
    ],
    related=["datasets.flow_demo_fcs", "io.read_fcs", "flow.GatingStrategy"],
)
def flow_demo(*, seed: int = 0) -> "AnnData":
    """The simulated cytometry sample, already read in.

    Goes out through an FCS file and back in through :func:`ov.io.read_fcs`
    rather than building the ``AnnData`` directly. That is deliberate:
    ``flowio`` writes float32, so a directly-constructed float64 matrix differs
    from the one a reader gets from the file — small, but enough to move an
    event across a gate boundary. Two "identical" demo samples that disagree
    about a population count is a bug nobody would ever look for, so there is
    only one path and it is the one the tutorials take.

    ``obs['population']`` is the ground truth. Real data does not come with it;
    it is here so a tutorial can ask "does this gate actually exclude the
    doublets?" and get an answer instead of an impression.
    """
    from .. import io as _io

    tmpdir = tempfile.mkdtemp(prefix="ov-flow-demo-")
    path = os.path.join(tmpdir, "flow_demo.fcs")
    try:
        flow_demo_fcs(path, seed=seed)
        adata = _io.read_fcs(path)
    finally:
        try:
            os.remove(path)
            os.rmdir(tmpdir)
        except OSError:  # pragma: no cover
            pass

    _, truth = _simulate(seed=seed)
    if truth.shape[0] != adata.n_obs:  # pragma: no cover
        raise RuntimeError(
            f"simulation produced {truth.shape[0]} events but the file read back "
            f"{adata.n_obs} — the ground-truth labels would be misaligned"
        )
    adata.obs["population"] = pd.Categorical(truth)
    adata.uns.setdefault("flow_demo", {})["simulated"] = True
    return adata
