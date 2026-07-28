"""Designs over discrete part choices, and the loop they close."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

import omicverse.synbio as sb

PARTS = {"p35": [f"m{i}" for i in range(8)],
         "spacer": [f"s{i}" for i in range(8)],
         "p10": [f"d{i}" for i in range(8)]}


# --- orthogonal arrays -----------------------------------------------------

@pytest.mark.parametrize("q", [2, 3, 4, 5, 7, 8, 9, 16])
def test_orthogonal_array_is_strength_two(q):
    """Every ordered pair of levels between any two columns equally often.

    This is the property the whole method rests on: it is what makes the
    main-effect estimates mutually uncorrelated.
    """
    k = min(q + 1, 5)
    oa = sb.orthogonal_array(q, k)
    assert oa.shape == (q * q, k)
    for i, j in itertools.combinations(range(k), 2):
        counts = np.zeros((q, q), dtype=int)
        for row in oa:
            counts[row[i], row[j]] += 1
        assert counts.min() == counts.max(), (i, j, counts)
    for column in range(k):
        assert np.ptp(np.bincount(oa[:, column], minlength=q)) == 0


def test_orthogonal_array_rejects_non_prime_power_levels():
    with pytest.raises(ValueError, match="素数"):
        sb.orthogonal_array(6, 3)


def test_orthogonal_array_rejects_too_many_factors():
    with pytest.raises(ValueError, match="q\\+1"):
        sb.orthogonal_array(8, 10)


# --- designs ---------------------------------------------------------------

def test_combinatorial_design_is_balanced_and_much_smaller():
    design = sb.combinatorial_design(PARTS, design="orthogonal_array")
    assert design.n_runs == 64
    assert design.full_size() == 512
    assert design.balance()["balanced"].all()


def test_block_is_balanced_and_excluded_from_the_factors():
    parts = dict(PARTS, locus=[f"bg{i}" for i in range(8)])
    design = sb.combinatorial_design(parts, design="orthogonal_array", block="locus")
    assert design.block == "locus"
    assert "locus" not in design.factor_names
    assert design.balance().loc["locus", "balanced"]


def test_orthogonal_array_needs_equal_level_counts():
    with pytest.raises(ValueError, match="水平数相同"):
        sb.combinatorial_design({"a": ["x", "y", "z"], "b": list("abcdefgh")},
                                design="orthogonal_array")


def test_balanced_design_is_the_mixed_level_fallback():
    design = sb.combinatorial_design({"a": ["x", "y", "z"], "b": list("abcdefgh")},
                                     design="balanced")
    assert design.balance()["balanced"].all()


def test_single_level_factor_is_rejected():
    with pytest.raises(ValueError, match="无法估计效应"):
        sb.combinatorial_design({"a": ["only"], "b": ["x", "y"]}, design="balanced")


# --- analysis --------------------------------------------------------------

def _planted(design, effects, interaction=0.0, seed=0):
    """Response built from known per-level effects, so truth is known."""
    rng = np.random.default_rng(seed)
    frame = design.to_frame()
    y = []
    for _, row in frame.iterrows():
        value = sum(effects[name][row[name]] for name in effects)
        if interaction:
            value += interaction * (row["p35"] == "m0") * (row["p10"] == "d0")
        y.append(value + rng.normal(0, 0.05))
    return frame, np.asarray(y)


def test_analyse_parts_recovers_planted_effects():
    design = sb.combinatorial_design(PARTS, design="orthogonal_array")
    truth = {"p35": {lvl: float(i) for i, lvl in enumerate(PARTS["p35"])},
             "spacer": {lvl: 0.1 * i for i, lvl in enumerate(PARTS["spacer"])},
             "p10": {lvl: -0.5 * i for i, lvl in enumerate(PARTS["p10"])}}
    frame, y = _planted(design, truth)
    effects = sb.analyse_parts(frame, y)
    for factor, expected in truth.items():
        got = effects.level_effects[factor]
        levels = sorted(expected)
        centred = np.array([expected[l] for l in levels])
        centred = centred - centred.mean()
        # 0.99 rather than 0.999: the spacer's planted spread is 0.70 against a
        # noise SD of 0.05, so 0.996 is already near-perfect recovery for it.
        assert np.corrcoef([got[l] for l in levels], centred)[0, 1] > 0.99, factor
    assert effects.r_squared > 0.99
    assert effects.residual_df > 0


def test_analyse_parts_ranks_the_dominant_factor_first():
    design = sb.combinatorial_design(PARTS, design="orthogonal_array")
    truth = {"p35": {lvl: 3.0 * i for i, lvl in enumerate(PARTS["p35"])},
             "spacer": {lvl: 0.01 * i for i, lvl in enumerate(PARTS["spacer"])},
             "p10": {lvl: 0.5 * i for i, lvl in enumerate(PARTS["p10"])}}
    frame, y = _planted(design, truth)
    effects = sb.analyse_parts(frame, y)
    assert [name for name, _ in effects.ranked_factors][0] == "p35"
    assert effects.p_values["p35"] < 0.01


@pytest.mark.parametrize("seed", range(5))
def test_analyse_parts_says_so_when_nothing_matters(seed):
    """Judged on a Bonferroni-corrected F-test, not on eta-squared.

    With 64 runs and 22 parameters a pure-noise response gives the top factor
    eta2 ~ 0.24, so an "eta2 < 0.05" rule fires essentially never. Three factors
    also means a naive p < 0.05 fires on noise about 14% of the time.
    """
    design = sb.combinatorial_design(PARTS, design="orthogonal_array")
    noise = np.random.default_rng(seed).normal(0, 1.0, size=design.n_runs)
    effects = sb.analyse_parts(design.to_frame(), noise)
    assert any("别在噪声上排序" in note for note in effects.notes), effects.notes


def test_analyse_parts_does_not_cry_noise_on_real_signal():
    design = sb.combinatorial_design(PARTS, design="orthogonal_array")
    truth = {"p35": {l: float(i) for i, l in enumerate(PARTS["p35"])},
             "spacer": {l: 0.1 * i for i, l in enumerate(PARTS["spacer"])},
             "p10": {l: -0.5 * i for i, l in enumerate(PARTS["p10"])}}
    frame, y = _planted(design, truth)
    effects = sb.analyse_parts(frame, y)
    assert not any("别在噪声上排序" in note for note in effects.notes)
    assert max(effects.p_values.values()) < 1e-10


def test_analyse_parts_removes_the_block_effect():
    parts = dict(PARTS, locus=[f"bg{i}" for i in range(8)])
    design = sb.combinatorial_design(parts, design="orthogonal_array", block="locus")
    frame = design.to_frame()
    shift = {lvl: 5.0 * i for i, lvl in enumerate(parts["locus"])}
    y = np.array([shift[row["locus"]] + (row["p35"] == "m0") * 2.0
                  for _, row in frame.iterrows()], dtype=float)
    effects = sb.analyse_parts(design, y)
    assert "locus" not in effects.level_effects
    assert max(effects.block_effects.values()) > 5.0
    assert effects.variance_explained["p35"] > effects.variance_explained["spacer"]


def test_analyse_parts_refuses_an_underdetermined_design():
    design = sb.combinatorial_design(PARTS, design="random", n_runs=8, seed=0)
    with pytest.raises(ValueError, match="估不出"):
        sb.analyse_parts(design, np.zeros(design.n_runs))


def test_effects_predict_and_best_levels_agree():
    design = sb.combinatorial_design(PARTS, design="orthogonal_array")
    truth = {"p35": {lvl: float(i) for i, lvl in enumerate(PARTS["p35"])},
             "spacer": {lvl: 0.0 for lvl in PARTS["spacer"]},
             "p10": {lvl: -float(i) for i, lvl in enumerate(PARTS["p10"])}}
    frame, y = _planted(design, truth)
    effects = sb.analyse_parts(frame, y)
    best = effects.best_levels()
    assert best["p35"] == "m7"
    assert best["p10"] == "d0"
    assert effects.predict(best) > effects.predict({"p35": "m0", "p10": "d7"})


# --- discrete proposals ----------------------------------------------------

def test_propose_combinations_only_returns_buildable_ones():
    design = sb.combinatorial_design(PARTS, design="orthogonal_array")
    truth = {"p35": {lvl: float(i) for i, lvl in enumerate(PARTS["p35"])},
             "spacer": {lvl: 0.1 * i for i, lvl in enumerate(PARTS["spacer"])},
             "p10": {lvl: 0.3 * i for i, lvl in enumerate(PARTS["p10"])}}
    frame, y = _planted(design, truth)
    catalogue = sb.combinatorial_design(PARTS, design="full_factorial").to_frame()
    proposal = sb.propose_combinations(frame, y, catalogue, batch=5, seed=0)
    assert len(proposal.combinations) == 5
    for combo in proposal.combinations:
        assert not sb.lookup_combination(catalogue, combo).empty
        for name, level in combo.items():
            assert level in PARTS[name]


def test_propose_combinations_excludes_what_was_measured():
    design = sb.combinatorial_design(PARTS, design="orthogonal_array")
    frame, y = _planted(design, {"p35": {l: float(i) for i, l in enumerate(PARTS["p35"])},
                                 "spacer": {l: 0.0 for l in PARTS["spacer"]},
                                 "p10": {l: 0.0 for l in PARTS["p10"]}})
    catalogue = sb.combinatorial_design(PARTS, design="full_factorial").to_frame()
    proposal = sb.propose_combinations(frame, y, catalogue, batch=4, seed=0)
    assert proposal.n_candidates == len(catalogue) - len(frame)
    seen = {tuple(row) for row in frame[list(PARTS)].astype(str).to_numpy()}
    for combo in proposal.combinations:
        assert tuple(str(combo[name]) for name in PARTS) not in seen


def test_propose_combinations_gives_distinct_points():
    """A naive batch asks for the current optimum several times over."""
    design = sb.combinatorial_design(PARTS, design="orthogonal_array")
    frame, y = _planted(design, {"p35": {l: float(i) for i, l in enumerate(PARTS["p35"])},
                                 "spacer": {l: 0.0 for l in PARTS["spacer"]},
                                 "p10": {l: 0.0 for l in PARTS["p10"]}})
    catalogue = sb.combinatorial_design(PARTS, design="full_factorial").to_frame()
    proposal = sb.propose_combinations(frame, y, catalogue, batch=6, seed=0)
    keys = {tuple(sorted(c.items())) for c in proposal.combinations}
    assert len(keys) == 6


def test_propose_combinations_needs_observations():
    catalogue = sb.combinatorial_design(PARTS, design="full_factorial").to_frame()
    with pytest.raises(ValueError, match="观测"):
        sb.propose_combinations(catalogue.head(2), [1.0, 2.0], catalogue, batch=2)


def test_propose_combinations_refuses_an_exhausted_catalogue():
    catalogue = sb.combinatorial_design({"a": list("abcd"), "b": list("wxyz")},
                                        design="full_factorial").to_frame()
    y = np.arange(len(catalogue), dtype=float)
    with pytest.raises(ValueError, match="全部测过"):
        sb.propose_combinations(catalogue, y, catalogue, batch=2)


# --- helpers ---------------------------------------------------------------

def test_lookup_combination_returns_empty_for_something_never_built():
    catalogue = sb.combinatorial_design(PARTS, design="orthogonal_array").to_frame()
    assert sb.lookup_combination(catalogue, {"p35": "not_a_part"}).empty


def test_lookup_combination_rejects_an_unknown_factor():
    catalogue = sb.combinatorial_design(PARTS, design="orthogonal_array").to_frame()
    with pytest.raises(KeyError, match="不在表里"):
        sb.lookup_combination(catalogue, {"nope": "x"})


def test_compare_part_effects_detects_agreement_and_disagreement():
    design = sb.combinatorial_design(PARTS, design="orthogonal_array")
    truth = {"p35": {l: float(i) for i, l in enumerate(PARTS["p35"])},
             "spacer": {l: 0.1 * i for i, l in enumerate(PARTS["spacer"])},
             "p10": {l: -0.4 * i for i, l in enumerate(PARTS["p10"])}}
    frame, y = _planted(design, truth, seed=0)
    a = sb.analyse_parts(frame, y)
    b = sb.analyse_parts(frame, y + 1e-9)
    same = sb.compare_part_effects(a, b)
    assert same["best_level_agrees"].all()
    assert (same["effect_r"] > 0.999).all()

    _frame2, y2 = _planted(design, truth, seed=1)
    scrambled = sb.analyse_parts(frame, np.random.default_rng(2).permutation(y2))
    mixed = sb.compare_part_effects(scrambled, a)
    assert not mixed["best_level_agrees"].all()


def test_plot_helpers_return_axes():
    design = sb.combinatorial_design(PARTS, design="orthogonal_array")
    frame, y = _planted(design, {"p35": {l: float(i) for i, l in enumerate(PARTS["p35"])},
                                 "spacer": {l: 0.0 for l in PARTS["spacer"]},
                                 "p10": {l: 0.0 for l in PARTS["p10"]}})
    effects = sb.analyse_parts(frame, y)
    fig, _ = sb.plot_part_effects(effects)
    assert fig is not None
    fig2, _ = sb.plot_design_balance(design)
    assert fig2 is not None
    fig3, _ = sb.plot_round_progress({"r1": y[:10], "r2": y[10:20]},
                                     best_possible=float(y.max()))
    assert fig3 is not None
