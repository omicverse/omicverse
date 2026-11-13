r"""Family-based haplotype inference (``ov.genetics``): registered entry points.

The numerics live in :mod:`omicverse.genetics._pedigree_core`, which imports
only numpy / scipy so that a deliverable that must run without omicverse
(a re-executed analysis script, a clinical pipeline) can embed it verbatim
via :func:`pedigree_core_source`.

``two_phase_weights``            inverse-probability weights of a biased /
                                 two-phase (ascertained + validated) sample.
``haplotype_frequencies``        design-weighted (Hajek) haplotype catalog and
                                 population frequencies.
``fit_haplotype_prior``          covariate- and group-dependent population
                                 haplotype prior with shared-family latent
                                 classes, chosen on held-out families.
``fit_crossover_intervals``      no-crossover / per-interval crossover masses
                                 from phased parents + ordered transmissions.
``gamete_sequences``             transmitted sequences with <= k crossovers.
``gamete_distribution``          their probabilities for any pair of sequences.
``offspring_outcome_probability`` expected offspring outcome of a parental state.
``genotype_likelihood_arrays``   long assay rows -> per-sample arrays.
``confident_copy_calls``         hard copy-number constraints from
                                 genotype-likelihood triplets.
``compatible_diplotypes``        catalog diplotypes consistent with them.
``structural_identifiability``   exact count of joint parental states
                                 compatible with all hard family evidence.
``simulate_family_cohort``       synthetic families with known truth.
``pedigree_core_source``         the core's source text, for embedding.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .._registry import register_function
from . import _pedigree_core as _core

__all__ = [
    "two_phase_weights", "haplotype_frequencies", "fit_haplotype_prior", "HaplotypePrior",
    "fit_crossover_intervals", "gamete_sequences", "gamete_distribution",
    "genotype_likelihood_arrays", "offspring_outcome_probability", "confident_copy_calls",
    "compatible_diplotypes", "structural_identifiability", "simulate_family_cohort",
    "pedigree_core_source",
]

HaplotypePrior = _core.HaplotypePrior

two_phase_weights = register_function(
    aliases=["two_phase_weights", "inverse probability weights", "IPW sampling weights",
             "ascertainment correction weights", "两阶段抽样权重", "逆概率加权"],
    category="genetics",
    description="Inverse-probability weights 1/(pi1*pi2) for a biased (ascertained) first-phase "
                "sample and a randomised second phase (e.g. phase validation, deep assay); "
                "optional quantile trimming and mean-1 normalisation. Estimates refer to the "
                "population before both sampling phases.",
    examples=["w = ov.genetics.two_phase_weights(fam_inclusion_prob, validation_prob, trim_quantile=0.99)"],
    related=["ov.genetics.haplotype_frequencies", "ov.genetics.fit_haplotype_prior"],
)(_core.two_phase_weights)

haplotype_frequencies = register_function(
    aliases=["haplotype_frequencies", "population haplotype frequencies", "weighted haplotype catalog",
             "Hajek haplotype frequency", "单倍型频率", "群体单倍型频率"],
    category="genetics",
    description="Design-weighted (Hajek) population haplotype catalog and frequencies from phased "
                "individuals (two haplotypes per person) and their sampling weights; corrects "
                "biased / ascertained sampling.",
    examples=["catalog, freq = ov.genetics.haplotype_frequencies(hap1, hap2, weights=w)"],
    related=["ov.genetics.two_phase_weights", "ov.genetics.fit_haplotype_prior"],
)(_core.haplotype_frequencies)

fit_haplotype_prior = register_function(
    aliases=["fit_haplotype_prior", "ancestry-dependent haplotype prior", "covariate haplotype frequency model",
             "founder mixture haplotype prior", "population haplotype prior", "单倍型先验",
             "祖源依赖单倍型频率", "家系单倍型先验"],
    category="genetics",
    description="Learn a population haplotype prior p(h | covariates, group) from phased families "
                "under a sampling design: latent shared-family (founder / community) classes whose "
                "mixing depends on a family-level group label, and within classes a penalised "
                "multinomial-logit surface on continuous covariates (ancestry, PCs). Class number, "
                "basis (constant / linear / quadratic / quadratic+RBF), class-shared vs per-class "
                "surface and penalty are chosen on held-out families (one-SE rule). The fitted "
                "model gives marginal haplotype priors, diplotype priors, the prior of one parent's "
                "diplotype given the other's (shared class), joint candidate priors and "
                "design-weighted population averages.",
    examples=["prior = ov.genetics.fit_haplotype_prior(h1, h2, ancestry, family_ids, groups=proxy, weights=w)",
              "p = prior.haplotype_probs(x_new, group='Q2')",
              "pg = prior.conditional_diplotype_prob(pat_pair, x_father, mat_pair, x_mother, group='Q2')"],
    related=["ov.genetics.two_phase_weights", "ov.genetics.haplotype_frequencies"],
)(_core.fit_haplotype_prior)

fit_crossover_intervals = register_function(
    aliases=["fit_crossover_intervals", "recombination interval masses", "crossover rate per interval",
             "sex-specific recombination", "重组率估计", "交换区间概率"],
    category="genetics",
    description="EM estimate of the no-crossover mass and per-interval single-crossover masses from "
                "phased parental homologues and ordered transmitted sequences, marginalising "
                "transmissions consistent with several crossover positions; Dirichlet smoothing "
                "chosen on held-out families. Fit separately per parental sex/role.",
    examples=["rec = ov.genetics.fit_crossover_intervals(maternal_pairs, maternal_transmitted, family_ids)"],
    related=["ov.genetics.gamete_sequences"],
)(_core.fit_crossover_intervals)

gamete_sequences = register_function(
    aliases=["gamete_sequences", "transmitted haplotypes", "recombinant gametes", "配子序列"],
    category="genetics",
    description="All distinct transmitted sequences of a parent's two homologues with at most k "
                "crossovers (k=0: the homologues themselves).",
    examples=["g = ov.genetics.gamete_sequences('0110', '1011', max_crossovers=1)"],
    related=["ov.genetics.structural_identifiability"],
)(_core.gamete_sequences)

gamete_distribution = register_function(
    aliases=["gamete_distribution", "transmission probabilities", "offspring haplotype distribution",
             "配子分布", "传递概率"],
    category="genetics",
    description="Probability of every sequence a parent transmits in one meiosis (start homologue 1/2, "
                "no crossover or one crossover per interval with given masses), for any pair of "
                "sequences -- including sequences outside a haplotype catalog. Basis for state-"
                "conditional offspring risks and transmission likelihoods.",
    examples=["seqs, p = ov.genetics.gamete_distribution(h1, h2, rec['no_crossover'], rec['interval_masses'])"],
    related=["ov.genetics.fit_crossover_intervals", "ov.genetics.gamete_sequences"],
)(_core.gamete_distribution)

offspring_outcome_probability = register_function(
    aliases=["offspring_outcome_probability", "state-conditional offspring risk", "parental state penetrance",
             "next conceptus risk given parents", "子代风险", "给定父母状态的子代患病概率"],
    category="genetics",
    description="Expected per-offspring outcome probability for one joint parental state: the outcome "
                "model (any callable of the maternal and paternal transmitted sequences, e.g. a "
                "logistic penetrance) averaged over both meioses' transmission distributions with "
                "role-specific crossover masses. Defined for any sequences (not limited to a catalog "
                "or candidate list); building block for offspring risk and recruitment / sibship "
                "history likelihoods of a supplied state.",
    examples=["q = ov.genetics.offspring_outcome_probability(mat_pair, pat_pair, penetrance, rec_m, rec_p)"],
    related=["ov.genetics.gamete_distribution", "ov.genetics.fit_crossover_intervals"],
)(_core.offspring_outcome_probability)

genotype_likelihood_arrays = register_function(
    aliases=["genotype_likelihood_arrays", "assay likelihood table to arrays", "基因型似然矩阵"],
    category="genetics",
    description="Convert long-format genotype-likelihood rows (sample, locus, call status, likelihoods "
                "of copy number 0/1/2) into per-sample (n_loci, 3) arrays plus a called mask, in a "
                "given locus order; uncalled loci are NaN.",
    examples=["lik = ov.genetics.genotype_likelihood_arrays(df.sample_id, df.locus_id, df.assay_status, "
              "df.likelihood_0, df.likelihood_1, df.likelihood_2, loci_order)"],
    related=["ov.genetics.confident_copy_calls", "ov.genetics.structural_identifiability"],
)(_core.genotype_likelihood_arrays)

confident_copy_calls = register_function(
    aliases=["confident_copy_calls", "high-confidence genotype calls", "hard genotype constraints",
             "高置信基因型"],
    category="genetics",
    description="Per-locus hard copy-number constraints (0/1/2, -1 = none) from genotype-likelihood "
                "triplets: the arg-max when it reaches a minimum value and a minimum ratio to the "
                "runner-up (both thresholds supplied by the study design).",
    examples=["cc = ov.genetics.confident_copy_calls(lik, called, min_probability=p_min, min_ratio=r_min)"],
    related=["ov.genetics.structural_identifiability"],
)(_core.confident_copy_calls)

compatible_diplotypes = register_function(
    aliases=["compatible_diplotypes", "diplotypes consistent with genotypes", "相容双倍型"],
    category="genetics",
    description="Unordered catalog diplotypes whose copy numbers match every hard constraint.",
    examples=["pairs = ov.genetics.compatible_diplotypes(catalog, copy_calls)"],
    related=["ov.genetics.structural_identifiability"],
)(_core.compatible_diplotypes)

structural_identifiability = register_function(
    aliases=["structural_identifiability", "identifiability certificate", "joint parental state identifiable",
             "family phase identifiability", "可识别性证书", "家系可识别性"],
    category="genetics",
    description="Exact count of joint parental diplotype states (mother, father; homologue swaps "
                "equivalent) compatible with hard parental copy-number constraints and, for every "
                "child, at least one ordered transmission with <= k crossovers matching its hard "
                "constraints. Exactly one -> IDENTIFIABLE, else AMBIGUOUS. A structural certificate, "
                "distinct from the maximum-posterior state.",
    examples=["label, n = ov.genetics.structural_identifiability(catalog, mother_cc, father_cc, [child_cc])"],
    related=["ov.genetics.confident_copy_calls", "ov.genetics.compatible_diplotypes"],
)(_core.structural_identifiability)

simulate_family_cohort = register_function(
    aliases=["simulate_family_cohort", "simulate pedigrees", "synthetic families haplotypes", "模拟家系"],
    category="genetics",
    description="Synthetic nuclear families with a known covariate- and class-dependent population "
                "haplotype prior, ascertained first-phase inclusion, randomised phase validation and "
                "children with <= 1 crossover per meiosis (tests / tutorials).",
    examples=["sim = ov.genetics.simulate_family_cohort(n_families=1500, seed=0)"],
    related=["ov.genetics.fit_haplotype_prior"],
)(_core.simulate_family_cohort)


@register_function(
    aliases=["pedigree_core_source", "vendor pedigree engine", "embed haplotype engine", "家系引擎源码"],
    category="genetics",
    description="Return (or write) the numpy/scipy-only source of the family haplotype core so a "
                "self-contained analysis script that is re-run without omicverse can import it "
                "(write it next to the script and `import` it, or embed the text).",
    examples=["ov.genetics.pedigree_core_source('results/pedigree_core.py')"],
    related=["ov.genetics.fit_haplotype_prior", "ov.genetics.structural_identifiability"],
)
def pedigree_core_source(path: Optional[Union[str, Path]] = None) -> str:
    """Source text of :mod:`omicverse.genetics._pedigree_core`; written to ``path`` if given."""
    text = Path(_core.__file__).read_text(encoding="utf-8")
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text
