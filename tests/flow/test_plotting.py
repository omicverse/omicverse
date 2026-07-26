"""``ov.flow.plotting`` — tests that assert what was DRAWN, not that it ran.

A plotting test that only checks "no exception" passes on a blank figure, and a
blank figure is exactly the failure this module exists to prevent. So every test
here reads the artists back off the axes and compares them to the gate geometry
and to ``GatingResult.stats()``.

Three of these were written after looking at the rendered PNG, because the
suite was fully green while the figures were wrong:

* a linear FSC axis was given decade ticks, piling 0, 10¹, 10², 10³ and 10⁴
  into the leftmost 4% of the plot as an unreadable smudge;
* a quadrant gate printed one useless "100.0%" label over the title instead of
  the four corner percentages that are the entire point of a quadrant plot;
* back-gating labelled its background layer with the parent's NAME and the
  count of parent-minus-population — "CD3+ (9,000)" beside a population with
  23,001 events.

Every check below has been mutation-tested: the bug was put back and the test
confirmed to go red.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import anndata as ad  # noqa: E402

from omicverse.flow import (  # noqa: E402
    EllipsoidGate,
    GatingStrategy,
    Linear,
    Logicle,
    PolygonGate,
    QuadrantGate,
    RectangleGate,
    backgate,
    biaxial,
    flowsom,
    flowsom_heatmap,
    hierarchy,
    histogram,
    spillover_heatmap,
)

LIN = Linear(t=262144.0)
LG = Logicle(t=262144.0, m=4.5, w=0.5, a=0.0)


def _at(value: float) -> float:
    """A data value on the logicle scale — how a threshold is really written."""
    return float(LG.apply(np.array([value]))[0])


@pytest.fixture(scope="module")
def sample():
    """A synthetic panel with populations whose membership is known."""
    rng = np.random.default_rng(7)

    def pop(n, fsc, ssc, cd3, cd4, cd8, cd19, via, doublet=False):
        f = rng.normal(fsc, fsc * 0.10, n)
        h = f / (1.9 if doublet else 1.0) + rng.normal(0, fsc * 0.02, n)
        return np.column_stack([
            f, h, rng.normal(ssc, ssc * 0.14, n),
            rng.lognormal(np.log(cd3), 0.55, n), rng.lognormal(np.log(cd4), 0.55, n),
            rng.lognormal(np.log(cd8), 0.55, n), rng.lognormal(np.log(cd19), 0.55, n),
            rng.lognormal(np.log(via), 0.5, n),
        ])

    blocks = [
        ("debris",  pop(9000, 18000, 9000, 30, 30, 30, 30, 60)),
        ("CD4T",    pop(14000, 82000, 26000, 9000, 12000, 60, 40, 70)),
        ("CD8T",    pop(9000, 80000, 25000, 9500, 55, 9000, 45, 70)),
        ("B",       pop(7000, 86000, 30000, 45, 40, 50, 8000, 75)),
        ("mono",    pop(5000, 150000, 62000, 60, 1200, 55, 70, 80)),
        ("doublet", pop(4000, 84000, 27000, 8000, 9000, 55, 50, 70, doublet=True)),
        ("dead",    pop(3000, 80000, 26000, 7000, 8000, 50, 50, 9000)),
    ]
    X = np.vstack([b for _, b in blocks])
    truth = np.repeat([n for n, _ in blocks], [b.shape[0] for _, b in blocks])
    perm = rng.permutation(X.shape[0])
    var = pd.DataFrame(
        index=["FSC-A", "FSC-H", "SSC-A", "CD3", "CD4", "CD8", "CD19", "Viability"]
    )
    var["channel"] = ["FSC-A", "FSC-H", "SSC-A", "FITC-A", "PE-A", "APC-A",
                      "PerCP-A", "Zombie-A"]
    adata = ad.AnnData(X[perm], var=var)
    adata.obs["truth"] = pd.Categorical(truth[perm])
    return adata


@pytest.fixture(scope="module")
def gates():
    cells = PolygonGate(
        name="Cells", dims=("FSC-A", "SSC-A"),
        transforms={"FSC-A": LIN, "SSC-A": LIN},
        vertices=np.array([[0.13, 0.02], [0.13, 0.19], [0.42, 0.34],
                           [0.72, 0.34], [0.72, 0.10], [0.40, 0.02]]),
    )
    singlets = PolygonGate(
        name="Singlets", dims=("FSC-A", "FSC-H"),
        transforms={"FSC-A": LIN, "FSC-H": LIN},
        vertices=np.array([[0.05, 0.02], [0.05, 0.09], [0.80, 0.90], [0.80, 0.72]]),
    )
    live = RectangleGate(name="Live", dims=("Viability",),
                         transforms={"Viability": LG},
                         bounds=((None, _at(1500.0)),))
    cd3 = RectangleGate(name="CD3+", dims=("CD3",), transforms={"CD3": LG},
                        bounds=((_at(500.0), None),))
    quad = QuadrantGate(
        name="CD4/CD8", dims=("CD4", "CD8"), transforms={"CD4": LG, "CD8": LG},
        dividers=(_at(500.0), _at(500.0)),
        quadrant_names=("DN", "CD4+CD8-", "CD4-CD8+", "DP"),
    )
    return dict(cells=cells, singlets=singlets, live=live, cd3=cd3, quad=quad)


@pytest.fixture(scope="module")
def strategy(gates):
    gs = GatingStrategy("T cell panel")
    gs.add_gate(gates["cells"])
    gs.add_gate(gates["singlets"], parent="Cells")
    gs.add_gate(gates["live"], parent="Singlets")
    gs.add_gate(gates["cd3"], parent="Live")
    gs.add_gate(gates["quad"], parent="CD3+")
    return gs


@pytest.fixture(scope="module")
def result(strategy, sample):
    return strategy.apply(sample, write_obs=False)


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


# ── the boundary is the gate ────────────────────────────────────────────────


def test_boundary_is_drawn_at_the_gates_own_coordinates(sample, gates, result, ax):
    biaxial(sample, "FSC-A", "SSC-A", gates=[gates["cells"]], result=result,
            ax=ax, max_events=3000)
    lines = ax.get_lines()
    assert len(lines) == 1
    drawn = np.column_stack(lines[0].get_data())
    expect = np.vstack([gates["cells"].vertices, gates["cells"].vertices[:1]])
    assert np.allclose(drawn, expect)


def test_gate_dimensions_are_matched_by_name_not_by_position(sample, gates, result, ax):
    """A gate stored as (FSC, SSC) drawn on an SSC × FSC plot must be
    transposed. Positional matching draws a mirrored gate that still looks
    entirely plausible, which is why this has its own test."""
    biaxial(sample, "SSC-A", "FSC-A", gates=[gates["cells"]], result=result,
            ax=ax, max_events=2000)
    drawn = np.column_stack(ax.get_lines()[0].get_data())
    v = gates["cells"].vertices[:, ::-1]
    assert np.allclose(drawn, np.vstack([v, v[:1]]))


def test_one_sided_threshold_spans_the_other_axis(sample, gates, result, ax):
    biaxial(sample, "CD3", "SSC-A", gates=[gates["cd3"]],
            transforms={"CD3": LG, "SSC-A": LIN}, result=result, ax=ax,
            max_events=2000)
    drawn = np.column_stack(ax.get_lines()[0].get_data())
    assert np.isclose(drawn[:, 0].min(), gates["cd3"].bounds[0][0])
    assert np.isclose(drawn[:, 0].max(), 1.0)      # unbounded above
    assert np.isclose(drawn[:, 1].min(), 0.0)
    assert np.isclose(drawn[:, 1].max(), 1.0)


def test_ellipse_contour_is_the_mahalanobis_level_set(sample, ax):
    ell = EllipsoidGate(
        name="E", dims=("CD4", "CD8"), transforms={"CD4": LG, "CD8": LG},
        mean=np.array([0.72, 0.30]),
        covariance=np.array([[0.004, 0.0015], [0.0015, 0.003]]),
        distance_square=4.0,
    )
    biaxial(sample, "CD4", "CD8", gates=[ell], transforms={"CD4": LG, "CD8": LG},
            ax=ax, max_events=1000)
    pts = np.column_stack(ax.get_lines()[0].get_data())[:-1]
    d = pts - ell.mean
    d2 = np.einsum("ij,jk,ik->i", d, np.linalg.inv(ell.covariance), d)
    assert np.allclose(d2, 4.0, atol=1e-8)


# ── the numbers on the plot are the numbers in stats() ──────────────────────


def test_gate_label_carries_the_strategys_own_frequency(sample, strategy, result, ax):
    stats = result.stats().set_index("population")
    biaxial(sample, "CD3", "SSC-A", strategy=strategy, gates=["CD3+"],
            result=result, population="Live", ax=ax, max_events=2000)
    texts = [t.get_text() for t in ax.texts]
    assert len(texts) == 1
    want = (f"{int(stats.loc['CD3+', 'count']):,} "
            f"({stats.loc['CD3+', 'freq_parent']:.1%})")
    assert want in texts[0]


def test_frequency_is_of_the_parent_not_of_all_events(sample, strategy, result, ax):
    """Recomputing the mask here instead of reading the result would quote a
    percentage of every event in the file — a different, larger, wrong
    number that looks entirely reasonable on the plot."""
    stats = result.stats().set_index("population")
    biaxial(sample, "CD3", "SSC-A", strategy=strategy, gates=["CD3+"],
            result=result, population="Live", ax=ax, max_events=2000)
    label = ax.texts[0].get_text()
    of_all = int(result.masks["CD3+"].sum()) / sample.n_obs
    assert stats.loc["CD3+", "freq_parent"] != pytest.approx(of_all)
    assert f"{of_all:.1%}" not in label


def test_quadrant_gate_labels_all_four_corners(sample, strategy, result, ax):
    stats = result.stats().set_index("population")
    biaxial(sample, "CD4", "CD8", strategy=strategy, gates=["CD4/CD8"],
            result=result, population="CD3+", ax=ax, max_events=2000)
    texts = [t.get_text() for t in ax.texts]
    assert len(texts) == 4
    for pop in ("DN", "CD4+CD8-", "CD4-CD8+", "DP"):
        match = [t for t in texts if t.startswith(pop + "\n")]
        assert len(match) == 1, f"quadrant {pop} unlabelled"
        assert f"{stats.loc[pop, 'freq_parent']:.1%}" in match[0]
    for t in ax.texts:                      # each in its own corner, inside
        px, py = t.get_position()
        assert 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0


def test_population_restricts_the_events_that_are_drawn(sample, strategy, result, ax):
    biaxial(sample, "CD4", "CD8", strategy=strategy, result=result,
            population="CD3+", ax=ax, max_events=None)
    drawn = int(ax.collections[0].get_offsets().shape[0])
    assert drawn == int(result.masks["CD3+"].sum())


# ── axes ────────────────────────────────────────────────────────────────────


def test_linear_axis_gets_evenly_spaced_ticks_in_data_units(sample, gates, result, ax):
    """Regression: decade ticks on a linear scatter axis pile every label
    below 10⁴ into the leftmost few percent of the plot."""
    biaxial(sample, "FSC-A", "SSC-A", gates=[gates["cells"]], result=result,
            ax=ax, max_events=1000)
    pos = np.sort(np.asarray(ax.get_xticks()))
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert len(pos) >= 3
    gaps = np.diff(pos)
    assert np.allclose(gaps, gaps[0]), f"linear axis has uneven ticks: {pos}"
    assert all(("K" in s) or s == "0" for s in labels), labels


def test_tick_labels_survive_a_font_change(sample, strategy, result, ax):
    """Regression: Unicode superscripts are font-dependent.

    ``¹²³`` are Latin-1 and near-universal; ``⁴⁵⁶⁷⁸⁹`` are U+2074+ and missing
    from many fonts. Under ``ov.style()`` the axes rendered 10² and 10³ fine and
    drew tofu boxes for 10⁴ and 10⁵ — a partial, silent failure that no
    exception and no numeric check would catch. Mathtext is drawn by matplotlib
    itself, so it cannot depend on the font in use.
    """
    biaxial(sample, "CD4", "CD8", strategy=strategy, result=result,
            population="CD3+", ax=ax, max_events=1000)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    exotic = [l for l in labels if any(ch in l for ch in "⁰¹²³⁴⁵⁶⁷⁸⁹")]
    assert not exotic, f"font-dependent superscripts in tick labels: {exotic}"
    powers = [l for l in labels if l not in ("0", "")]
    assert powers, "no decade labels at all"
    assert all(l.startswith("$") and l.endswith("$") for l in powers), powers


def test_biexponential_axis_is_labelled_in_data_units(sample, strategy, result, ax):
    biaxial(sample, "CD4", "CD8", strategy=strategy, result=result,
            population="CD3+", ax=ax, max_events=1000)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert ax.get_xlim() == (0.0, 1.0)
    assert "0" in labels                      # the linear region is marked
    assert any("10" in s for s in labels)     # and the decades are named


def test_tick_labels_near_zero_do_not_overprint(sample, strategy, result, ax):
    """The linear region of a logicle scale packs -10¹, 0 and 10¹ into a few
    percent of the axis; unthinned they render as a smudge."""
    biaxial(sample, "CD4", "CD8", strategy=strategy, result=result,
            population="CD3+", ax=ax, max_events=1000)
    for ticks in (ax.get_xticks(), ax.get_yticks()):
        gaps = np.diff(np.sort(np.asarray(ticks)))
        assert gaps.size and gaps.min() >= 0.05, f"labels overprint: {ticks}"
    assert len([t.get_text() for t in ax.get_xticklabels()]) >= 4


# ── histogram, back-gate, hierarchy ─────────────────────────────────────────


def test_histogram_normalises_each_population_to_its_own_mode(
    sample, strategy, result, ax
):
    histogram(sample, "CD3", strategy=strategy, gates=["CD3+"], result=result,
              populations=["Live", "CD3+"], ax=ax)
    curves = [l for l in ax.get_lines() if l.get_linestyle() == "-"]
    assert len(curves) == 2
    for c in curves:
        assert np.isclose(np.nanmax(c.get_ydata()), 1.0)
    dashed = [l for l in ax.get_lines() if l.get_linestyle() == "--"]
    assert len(dashed) == 1


def test_backgate_legend_counts_mean_what_the_labels_say(sample, result, ax):
    """Regression: the background layer excludes the highlighted events so they
    are not drawn twice, but it is the PARENT that is named — so it is the
    parent's count that has to be quoted."""
    backgate(sample, "FSC-A", "SSC-A", result=result, population="CD4+CD8-",
             transforms={"FSC-A": LIN, "SSC-A": LIN}, ax=ax, max_events=None)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels[0] == f"CD3+ ({int(result.masks['CD3+'].sum()):,})"
    assert labels[1] == f"CD4+CD8- ({int(result.masks['CD4+CD8-'].sum()):,})"


def test_backgate_does_not_draw_the_population_twice(sample, result, ax):
    backgate(sample, "FSC-A", "SSC-A", result=result, population="CD4+CD8-",
             transforms={"FSC-A": LIN, "SSC-A": LIN}, ax=ax, max_events=None)
    counts = [c.get_offsets().shape[0] for c in ax.collections]
    assert counts[1] == int(result.masks["CD4+CD8-"].sum())
    assert counts[0] == int(
        (result.masks["CD3+"] & ~result.masks["CD4+CD8-"]).sum()
    )


def test_hierarchy_shows_every_population_with_its_frequency(strategy, result, ax):
    stats = result.stats().set_index("population")
    hierarchy(strategy, result=result, ax=ax)
    labels = [a.get_text() for a in ax.texts]
    assert any(l.startswith("root") for l in labels)
    for name in ("Cells", "Singlets", "Live", "CD3+", "CD4+CD8-", "DP"):
        assert any(l.startswith(name) for l in labels), f"{name} missing"
    cd3 = next(l for l in labels if l.startswith("CD3+"))
    assert f"{stats.loc['CD3+', 'freq_parent']:.1%}" in cd3


def test_hierarchy_refuses_an_empty_strategy(ax):
    with pytest.raises(ValueError, match="no gates"):
        hierarchy(GatingStrategy("empty"), ax=ax)


# ── matrices ────────────────────────────────────────────────────────────────


def test_spillover_colour_scale_excludes_the_diagonal(ax):
    """With the diagonal in the normalisation every off-diagonal value — the
    entire content of the matrix — is squeezed into the bottom few percent of
    the colour map."""
    det = ["FITC-A", "PE-A", "PerCP-A", "APC-A"]
    spill = pd.DataFrame(
        np.eye(4) + np.array([[0, .21, .04, .00], [.02, 0, .17, .01],
                              [.00, .03, 0, .09], [.00, .00, .05, 0]]),
        index=det, columns=det,
    )
    spillover_heatmap(spill, ax=ax)
    assert np.isclose(ax.images[0].get_clim()[1], 0.21)


def test_spillover_says_where_it_looked_when_there_is_no_matrix(sample, ax):
    with pytest.raises(KeyError, match="uns\\['flow'\\]\\['spillover'\\]"):
        spillover_heatmap(sample, ax=ax)


def test_flowsom_heatmap_reports_cluster_sizes(sample, ax):
    adata = sample.copy()
    adata.layers["scaled"] = np.column_stack(
        [LIN.apply(adata.X[:, i]) for i in range(3)]
        + [LG.apply(adata.X[:, i]) for i in range(3, 8)]
    )
    flowsom(adata, n_clusters=6, grid=(6, 6), n_epochs=6, layer="scaled",
            markers=["CD3", "CD4", "CD8", "CD19"], random_state=0)
    flowsom_heatmap(adata, layer="scaled", ax=ax)
    assert ax.images[0].get_array().shape[1] == 4
    assert all("n=" in t.get_text() for t in ax.get_yticklabels())


def test_flowsom_heatmap_names_the_function_you_forgot_to_run(sample, ax):
    with pytest.raises(KeyError, match="ov.flow.flowsom"):
        flowsom_heatmap(sample.copy(), ax=ax)
