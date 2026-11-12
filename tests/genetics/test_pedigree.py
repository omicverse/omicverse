"""Tests for the family haplotype core (ov.genetics._pedigree_core) on simulated pedigrees."""
import importlib.util
import itertools
import math

import numpy as np
import pytest

from omicverse.genetics import _pedigree_core as pc


@pytest.fixture(scope="module")
def cohort():
    return pc.simulate_family_cohort(n_families=3000, n_loci=10, n_haplotypes=30, seed=3)


def _validated(sim):
    fam = {f["family_id"]: f for f in sim["families"]}
    rows = [p for p in sim["people"] if fam[p["family_id"]]["validated"]]
    w = pc.two_phase_weights([fam[p["family_id"]]["first_phase_prob"] for p in rows],
                             [fam[p["family_id"]]["second_phase_prob"] for p in rows])
    return rows, w


def test_two_phase_weights():
    w = pc.two_phase_weights([0.5, 0.25, 1.0], [0.5, 0.5, 0.5], normalize=False)
    assert np.allclose(w, [4.0, 8.0, 2.0])
    wn = pc.two_phase_weights([0.5, 0.25, 1.0], [0.5, 0.5, 0.5])
    assert math.isclose(wn.mean(), 1.0)
    wt = pc.two_phase_weights([0.5, 0.25, 1.0], trim_quantile=0.5, normalize=False)
    assert wt.max() == pytest.approx(2.0)
    with pytest.raises(ValueError):
        pc.two_phase_weights([0.0, 0.5])


def test_weighted_frequencies_undo_ascertainment(cohort):
    rows, w = _validated(cohort)
    truth = dict(zip(cohort["catalog"], cohort["true_population_freq"]))

    def tv(cat, freq):
        est = dict(zip(cat, freq))
        return 0.5 * sum(abs(truth.get(h, 0) - est.get(h, 0)) for h in set(truth) | set(est))

    a = [p["hap_a"] for p in rows]
    b = [p["hap_b"] for p in rows]
    cat_w, f_w = pc.haplotype_frequencies(a, b, w)
    cat_u, f_u = pc.haplotype_frequencies(a, b)
    assert math.isclose(f_w.sum(), 1.0)
    assert tv(cat_w, f_w) < tv(cat_u, f_u) - 0.02


def test_prior_learns_covariates_and_shared_class(cohort):
    rows, w = _validated(cohort)
    x = np.array([p["x"] for p in rows])
    model = pc.fit_haplotype_prior([p["hap_a"] for p in rows], [p["hap_b"] for p in rows], x,
                                   [p["family_id"] for p in rows], [p["group"] for p in rows], w,
                                   penalties=(1.0, 0.1), random_state=1)
    assert model.n_components >= 2
    assert model.selection["basis"] != "constant"
    rng = np.random.default_rng(7)
    xs = rng.dirichlet(np.ones(3), size=200)
    groups = [f"G{1 + i % 3}" for i in range(200)]
    idx = [cohort["catalog"].index(h) for h in model.catalog]
    err_model, err_flat = [], []
    flat = dict(zip(*pc.haplotype_frequencies([p["hap_a"] for p in rows], [p["hap_b"] for p in rows], w)))
    flat_vec = np.array([flat[h] for h in model.catalog])
    for xi, g in zip(xs, groups):
        true = cohort["true_haplotype_probs"](xi[None], g)[0][idx]
        err_model.append(model.haplotype_probs(xi[None], g)[0] - true)
        err_flat.append(flat_vec - true)
    rmse_model = float(np.sqrt(np.mean(np.square(err_model))))
    rmse_flat = float(np.sqrt(np.mean(np.square(err_flat))))
    assert rmse_model < 0.6 * rmse_flat

    # probabilities are normalised; conditional diplotype prior sums to one
    p = model.haplotype_probs(xs[:3], "G1")
    assert np.allclose(p.sum(axis=1), 1.0)
    H = len(model.catalog)
    given = (model.catalog[0], model.catalog[1])
    total = sum(model.conditional_diplotype_prob((model.catalog[i], model.catalog[j]), xs[0], given, xs[1], "G2")
                for i in range(H) for j in range(i, H))
    assert total == pytest.approx(1.0, abs=1e-8)
    dm = model.diplotype_matrix(xs[0], "G2")
    assert dm.sum() == pytest.approx(1.0, abs=1e-8)
    # with the shared class, the father's prior depends on the mother's diplotype
    fam = rows[0]["family_id"]
    mother = [r for r in rows if r["family_id"] == fam and r["role"] == "maternal"][0]
    post = model.class_posterior([(mother["x"], (mother["hap_a"], mother["hap_b"]))], mother["group"])
    assert post.sum() == pytest.approx(1.0)
    pop = model.population_average(xs, groups)
    assert pop.sum() == pytest.approx(1.0)


def test_gamete_sequences():
    g = pc.gamete_sequences("0000", "1111", 1)
    assert len(g) == 2 + 2 * 3
    assert set(pc.gamete_sequences("0101", "0101", 1)) == {"0101"}
    assert len(pc.gamete_sequences("00000", "11111", 2)) == 2 + 2 * 4 + 2 * 6


def test_crossover_intervals_recovered(cohort):
    fam = {f["family_id"]: f for f in cohort["families"]}
    by_fam = {}
    for p in cohort["people"]:
        by_fam.setdefault(p["family_id"], {})[p["role"]] = (p["hap_a"], p["hap_b"])
    pairs, trans, fids = [], [], []
    for c in cohort["children"]:
        pairs.append(by_fam[c["family_id"]]["maternal"])
        trans.append(c["maternal_transmitted"])
        fids.append(c["family_id"])
        pairs.append(by_fam[c["family_id"]]["paternal"])
        trans.append(c["paternal_transmitted"])
        fids.append(c["family_id"])
    rec = pc.fit_crossover_intervals(pairs, trans, fids)
    truth = cohort["crossover"]
    assert rec["no_crossover"] == pytest.approx(truth["no_crossover"], abs=0.04)
    assert np.max(np.abs(rec["interval_masses"] - truth["interval_masses"])) < 0.03
    assert rec["no_crossover"] + rec["interval_masses"].sum() == pytest.approx(1.0)


def test_confident_copy_calls():
    lik = np.array([[0.9, 0.1, 0.0], [0.5, 0.4, 0.1], [np.nan] * 3, [0.0, 0.2, 0.8], [0.0, 0.25, 0.75]])
    cc = pc.confident_copy_calls(lik, [True, True, False, True, True], min_probability=0.72, min_ratio=3.0)
    assert cc.tolist() == [0, -1, -1, 2, 2]
    cc = pc.confident_copy_calls(lik, None, min_probability=0.8, min_ratio=3.5)
    assert cc.tolist() == [0, -1, -1, 2, -1]


def _brute_force_count(catalog, mc, pc_, kids):
    def ok_parent(pair, cc):
        return all(int(pair[0][l]) + int(pair[1][l]) == cc[l] for l in range(len(cc)) if cc[l] >= 0)

    dips = [(a, b) for a, b in itertools.combinations_with_replacement(catalog, 2)]
    n = 0
    for m in dips:
        if not ok_parent(m, mc):
            continue
        for p in dips:
            if not ok_parent(p, pc_):
                continue
            good = True
            for k in kids:
                if not any(all(int(gm[l]) + int(gp[l]) == k[l] for l in range(len(k)) if k[l] >= 0)
                           for gm in pc.gamete_sequences(*m, 1) for gp in pc.gamete_sequences(*p, 1)):
                    good = False
                    break
            n += good
    return n


def test_structural_identifiability_matches_brute_force():
    rng = np.random.default_rng(11)
    L = 6
    catalog = sorted({"".join(rng.choice(["0", "1"], size=L)) for _ in range(12)})
    n_ident = n_amb = 0
    for trial in range(60):
        m = tuple(sorted(rng.choice(catalog, size=2)))
        p = tuple(sorted(rng.choice(catalog, size=2)))

        def obs(copy):
            copy = np.array(copy)
            mask = rng.random(L) < 0.3
            copy[mask] = -1
            return copy

        mc = obs([int(m[0][l]) + int(m[1][l]) for l in range(L)])
        pcc = obs([int(p[0][l]) + int(p[1][l]) for l in range(L)])
        kids = []
        for _ in range(int(rng.integers(0, 3))):
            gm = pc.gamete_sequences(*m, 1)[int(rng.integers(0, len(pc.gamete_sequences(*m, 1))))]
            gp = pc.gamete_sequences(*p, 1)[int(rng.integers(0, len(pc.gamete_sequences(*p, 1))))]
            kids.append(obs([int(a) + int(b) for a, b in zip(gm, gp)]))
        expected = _brute_force_count(catalog, mc, pcc, kids)
        label, n, states = pc.structural_identifiability(catalog, mc, pcc, kids, return_states=True)
        assert n == expected
        assert label == ("IDENTIFIABLE" if expected == 1 else "AMBIGUOUS")
        lab2, n2 = pc.structural_identifiability(catalog, mc, pcc, kids)
        assert lab2 == label and n2 == min(expected, 2)
        n_ident += label == "IDENTIFIABLE"
        n_amb += label == "AMBIGUOUS"
    assert n_ident > 0 and n_amb > 0


def test_core_source_is_standalone(tmp_path):
    from omicverse.genetics._pedigree import pedigree_core_source

    path = tmp_path / "pedigree_core.py"
    text = pedigree_core_source(path)
    assert "import omicverse" not in text and "from omicverse" not in text and "from ." not in text
    spec = importlib.util.spec_from_file_location("vendored_core", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.structural_identifiability(["00", "11"], np.array([0, 0]), np.array([2, 2]))[0] == "IDENTIFIABLE"
