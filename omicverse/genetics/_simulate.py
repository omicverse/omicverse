"""Coherent synthetic GWAS cohort for ``ov.genetics`` tutorials and tests.

:func:`simulate_gwas_study` generates ONE internally-consistent study —
genotypes with realistic LD block structure and population
substructure, a heritable quantitative trait, a matched bulk-expression
panel carrying real cis-eQTLs, and a small scRNA-seq atlas — together
with the planted ground truth. The same causal SNP drives both the
trait and one gene's expression, so colocalization, TWAS and Mendelian
randomization all have a real mediated signal to recover, and the
causal gene is selectively expressed in one cell type so scDRS can
pinpoint a disease-relevant cell type.

The whole study is deterministic given ``seed`` and is built to run a
full best-practice GWAS pipeline end to end in a few seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .._registry import register_function


@dataclass
class GWASStudy:
    """A coherent simulated statistical-genetics study.

    Attributes
    ----------
    genotype
        ``AnnData`` of ``samples x SNPs`` — integer 0/1/2 allele dosages
        in ``.X``. ``.var`` carries ``chrom``, ``pos``, ``maf``,
        ``block`` and a boolean ``causal`` flag; ``.obs`` carries a
        ``population`` label and the quantitative ``phenotype``.
    phenotype
        Per-sample quantitative trait (a :class:`pandas.Series` aligned
        to ``genotype.obs_names``); also stored in ``genotype.obs``.
    expression
        ``AnnData`` of ``samples x genes`` — a bulk-expression panel
        sharing the genotype's samples. ``.var`` carries ``chrom``,
        ``pos``, an ``eqtl_snp`` column (the cis driver, or ``''``) and
        an ``is_eqtl_gene`` flag.
    scrna
        ``AnnData`` of ``cells x genes`` — a small scRNA-seq atlas with
        a ``cell_type`` column; the trait's causal gene is selectively
        over-expressed in one cell type.
    truth
        Dict of planted ground truth — see :func:`simulate_gwas_study`.
    params
        The simulation parameters used.
    """

    genotype: "object"
    phenotype: "pd.Series"
    expression: "object"
    scrna: "object"
    truth: Dict[str, object] = field(default_factory=dict)
    params: Dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        g, e, s = self.genotype, self.expression, self.scrna
        return (
            "GWASStudy(\n"
            f"  genotype  : {g.n_obs} samples x {g.n_vars} SNPs\n"
            f"  phenotype : quantitative trait, h2 = {self.params.get('h2')}\n"
            f"  expression: {e.n_obs} samples x {e.n_vars} genes\n"
            f"  scrna     : {s.n_obs} cells x {s.n_vars} genes\n"
            f"  truth     : causal_snps={self.truth.get('causal_snps')}, "
            f"causal_gene={self.truth.get('causal_gene')!r}, "
            f"cell_type={self.truth.get('relevant_cell_type')!r}\n"
            ")"
        )


def _make_blocks(n_snps: int, n_blocks: int, rng) -> np.ndarray:
    """Assign each SNP to an LD block of roughly equal size."""
    base = n_snps // n_blocks
    sizes = np.full(n_blocks, base, dtype=int)
    sizes[: n_snps - base * n_blocks] += 1
    return np.repeat(np.arange(n_blocks), sizes)


@register_function(
    aliases=[
        "simulate_gwas_study", "simulate_gwas", "gwas_simulate",
        "模拟GWAS数据", "GWAS模拟数据", "遗传学模拟",
    ],
    category="genetics",
    description=(
        "Simulate one coherent, internally-consistent statistical-"
        "genetics study for tutorials and tests. Returns a genotype "
        "AnnData with realistic LD-block structure and population "
        "substructure, a heritable quantitative trait, a matched bulk-"
        "expression panel carrying true cis-eQTLs, and a small scRNA-seq "
        "atlas — plus the planted ground truth. Crucially the SAME causal "
        "SNP drives both the trait and one gene's expression, so "
        "colocalization, TWAS and Mendelian randomization all recover a "
        "real mediated signal, and that causal gene is selectively "
        "expressed in one cell type so scDRS recovers a disease-relevant "
        "cell type. Deterministic given ``seed``."
    ),
    produces={
        "obs": ["population", "phenotype", "cell_type"],
        "var": ["chrom", "pos", "maf", "causal", "eqtl_snp", "is_eqtl_gene"],
    },
    auto_fix="none",
    examples=[
        "study = ov.genetics.simulate_gwas_study(seed=0)",
        "study = ov.genetics.simulate_gwas_study(n_samples=3000, n_snps=8000)",
        "geno, pheno = study.genotype, study.phenotype",
    ],
    related=[
        "ov.genetics.gwas_qc", "ov.genetics.gwas_association",
        "ov.genetics.finemap", "ov.genetics.colocalize",
    ],
)
def simulate_gwas_study(
    *,
    n_samples: int = 2000,
    n_snps: int = 6000,
    n_blocks: int = 30,
    n_causal: int = 5,
    n_genes: int = 200,
    n_pops: int = 2,
    n_cells: int = 1500,
    h2: float = 0.4,
    seed: int = 0,
) -> GWASStudy:
    """Simulate a coherent GWAS study with known ground truth.

    The study is built so every step of a post-GWAS pipeline has a real
    planted signal to recover:

    * **Genotypes** — ``n_snps`` SNPs spread over ``n_blocks`` LD blocks.
      Each block is drawn from a correlated latent multivariate normal,
      then thresholded to 0/1/2 dosages at per-SNP minor-allele
      frequencies, giving realistic within-block LD. ``n_pops``
      subpopulations have slightly divergent allele frequencies, so
      genotype PCA recovers genuine structure.
    * **Phenotype** — a quantitative trait equal to the additive effect
      of ``n_causal`` causal SNPs (scaled so they explain a fraction
      ``h2`` of the variance) **plus** a small population-structure
      effect **plus** Gaussian noise. The structure effect makes a
      naive association scan inflated and PC adjustment necessary.
    * **Expression** — a bulk-expression panel over the same samples;
      a handful of genes carry true cis-eQTLs. One of the trait's causal
      SNPs is *also* the cis-eQTL of one designated gene (the *causal
      gene*) — i.e. the trait signal at that locus is mediated by that
      gene, so colocalization / TWAS / MR all have a true signal.
    * **scRNA-seq** — a small single-cell atlas over the same genes; the
      causal gene is selectively over-expressed in one cell type, so
      scDRS recovers a disease-relevant cell type.

    Parameters
    ----------
    n_samples
        Number of individuals (rows of the genotype / expression
        AnnData).
    n_snps
        Number of SNPs (columns of the genotype AnnData).
    n_blocks
        Number of LD blocks; SNPs within a block are correlated.
    n_causal
        Number of SNPs with a true effect on the trait.
    n_genes
        Number of genes in the bulk-expression and scRNA-seq panels.
    n_pops
        Number of subpopulations with divergent allele frequencies.
    n_cells
        Number of cells in the scRNA-seq atlas.
    h2
        Narrow-sense heritability — the fraction of trait variance
        explained by the causal SNPs.
    seed
        RNG seed; the whole study is deterministic given this value.

    Returns
    -------
    GWASStudy
        A dataclass with ``.genotype`` (AnnData, samples x SNPs),
        ``.phenotype`` (Series), ``.expression`` (AnnData, samples x
        genes), ``.scrna`` (AnnData, cells x genes), ``.truth`` and
        ``.params``. ``.truth`` carries:

        * ``causal_snps`` — list of trait-causal SNP ids
        * ``lead_causal_snp`` — the causal SNP that is also the cis-eQTL
          of the causal gene (the mediated locus)
        * ``gene_instruments`` — independent eQTLs of the causal gene in
          other LD blocks (the Mendelian-randomization instruments)
        * ``causal_gene`` — the gene mediating the trait at that locus
        * ``causal_block`` — the LD block of the lead causal SNP
        * ``eqtl_map`` — dict ``{gene: cis_eQTL_snp}`` for all eGenes
        * ``relevant_cell_type`` — the disease-relevant cell type
        * ``snp_effects`` — dict ``{snp: trait beta}`` for causal SNPs
    """
    from anndata import AnnData

    if n_causal < 1:
        raise ValueError("n_causal must be at least 1.")
    if n_genes < 5:
        raise ValueError("n_genes must be at least 5.")
    if not 0.0 < h2 < 1.0:
        raise ValueError("h2 must be in (0, 1).")

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # 1. SNP annotation — chromosomes, positions, LD blocks, MAF.        #
    # ------------------------------------------------------------------ #
    block = _make_blocks(n_snps, n_blocks, rng)
    # Map every LD block to a chromosome (a few blocks per chromosome).
    blocks_per_chr = max(1, n_blocks // 22)
    chrom_of_block = np.minimum(1 + np.arange(n_blocks) // blocks_per_chr, 22)
    chrom = chrom_of_block[block]
    # Genomic position: blocks are ~1 Mb apart, SNPs ~5 kb apart in a block.
    pos = np.zeros(n_snps, dtype=int)
    for b in range(n_blocks):
        idx = np.where(block == b)[0]
        start = 1_000_000 + (b % blocks_per_chr) * 2_000_000
        pos[idx] = start + np.arange(idx.size) * 5_000
    # A baseline MAF per SNP (population-level reference frequency).
    maf = rng.uniform(0.05, 0.5, n_snps)

    snp_ids = np.array([f"rs{i:06d}" for i in range(n_snps)])

    # ------------------------------------------------------------------ #
    # 2. Population structure — divergent allele frequencies.            #
    # ------------------------------------------------------------------ #
    pop = rng.integers(0, n_pops, n_samples)
    # Each population shifts every SNP's allele frequency by a small,
    # SNP-specific amount (Fst-like drift), kept inside [0.02, 0.98].
    pop_shift = rng.normal(0.0, 0.06, size=(n_pops, n_snps))
    pop_freq = np.clip(maf[None, :] + pop_shift, 0.02, 0.98)

    # ------------------------------------------------------------------ #
    # 3. Genotypes — correlated LD blocks thresholded to 0/1/2 dosages.  #
    # ------------------------------------------------------------------ #
    geno = np.zeros((n_samples, n_snps), dtype=np.int8)
    for b in range(n_blocks):
        idx = np.where(block == b)[0]
        k = idx.size
        # A latent within-block correlation: AR(1)-style decay so nearby
        # SNPs are in tighter LD than distant ones.
        rho = 0.85
        d = np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
        corr = rho ** d
        chol = np.linalg.cholesky(corr + 1e-6 * np.eye(k))
        # Two latent haplotype draws per individual -> additive dosage.
        for hap in range(2):
            latent = rng.standard_normal((n_samples, k)) @ chol.T
            # Per-individual threshold from that individual's pop frequency.
            thr = np.array(
                [_norm_isf(pop_freq[pop[i], idx]) for i in range(n_samples)]
            )
            geno[:, idx] += (latent > thr).astype(np.int8)

    # Realised (in-sample) minor-allele frequency after drawing genotypes.
    realised_af = geno.mean(axis=0) / 2.0
    realised_maf = np.minimum(realised_af, 1.0 - realised_af)

    # ------------------------------------------------------------------ #
    # 4. Causal architecture — designate causal SNPs, the causal gene    #
    #    and its instruments.                                            #
    # ------------------------------------------------------------------ #
    # Pick causal SNPs from common variants in distinct LD blocks. The
    # FIRST causal SNP is the *lead* — it acts on the trait only through
    # the causal gene's expression (a true mediated locus); the others
    # act on the trait directly (independent pleiotropy-free loci).
    common = np.where(realised_maf > 0.1)[0]
    rng.shuffle(common)
    causal_idx: List[int] = []
    used_blocks: set = set()
    for j in common:
        if block[j] not in used_blocks:
            causal_idx.append(int(j))
            used_blocks.add(block[j])
        if len(causal_idx) == n_causal:
            break
    causal_idx = np.array(sorted(causal_idx))
    lead_causal_snp_idx = int(causal_idx[0])
    causal_block = int(block[lead_causal_snp_idx])
    # The remaining causal SNPs act on the trait directly.
    direct_causal_idx = np.array(
        [j for j in causal_idx if j != lead_causal_snp_idx]
    )

    gene_ids = np.array([f"GENE{i:04d}" for i in range(n_genes)])
    causal_gene = gene_ids[0]
    n_extra_egenes = min(6, n_genes - 1)
    egene_idx = np.array([0] + list(range(1, 1 + n_extra_egenes)))

    # The causal gene's primary cis driver is the lead trait-causal SNP.
    eqtl_snp_for_gene: Dict[str, str] = {causal_gene: snp_ids[lead_causal_snp_idx]}
    eqtl_idx_for_gene: Dict[int, int] = {0: lead_causal_snp_idx}
    # The causal gene additionally has several independent eQTLs in OTHER
    # LD blocks. These are valid Mendelian-randomization instruments for
    # the gene's expression — and, because the gene mediates the trait,
    # they move the trait only through the gene (no horizontal pleiotropy).
    n_gene_instruments = 6
    avail = np.setdiff1d(common, causal_idx)
    avail = np.array([j for j in avail if block[j] != causal_block])
    rng.shuffle(avail)
    extra_instruments: List[int] = []
    inst_blocks: set = set()
    for j in avail:
        if block[j] not in inst_blocks:
            extra_instruments.append(int(j))
            inst_blocks.add(block[j])
        if len(extra_instruments) == n_gene_instruments:
            break
    extra_instruments = np.array(extra_instruments, dtype=int)
    # Other eGenes get their own cis SNPs (from remaining LD blocks).
    other_pool = np.array(
        [j for j in avail if j not in set(extra_instruments.tolist())]
    )
    other_snps = rng.choice(other_pool, size=n_extra_egenes, replace=False)
    for g_local, snp_j in zip(egene_idx[1:], other_snps):
        eqtl_snp_for_gene[gene_ids[g_local]] = snp_ids[snp_j]
        eqtl_idx_for_gene[int(g_local)] = int(snp_j)

    # ------------------------------------------------------------------ #
    # 5. Causal-gene expression — driven by its cis SNP + instruments.   #
    # ------------------------------------------------------------------ #
    def _std(col):
        col = col.astype(float)
        return (col - col.mean()) / (col.std() + 1e-9)

    # The causal gene's "true" expression: the lead cis SNP (much the
    # largest weight, so it is the dominant cis-eQTL and fine-maps to a
    # tight credible set) plus several independent eQTL instruments,
    # plus residual noise. The cis-eQTL h2 of the gene is ~0.7.
    cg_genetic = 3.0 * _std(geno[:, lead_causal_snp_idx])
    inst_weights = np.abs(rng.normal(0.0, 1.0, extra_instruments.size)) + 0.6
    for w, j in zip(inst_weights, extra_instruments):
        cg_genetic = cg_genetic + w * _std(geno[:, j])
    cg_genetic = cg_genetic / (cg_genetic.std() + 1e-9)   # var = 1
    cg_eqtl_h2 = 0.7
    cg_expr_noise = rng.normal(
        0.0, np.sqrt(1.0 / cg_eqtl_h2 - 1.0), n_samples,
    )
    causal_gene_expr = cg_genetic + cg_expr_noise          # the mediator

    # ------------------------------------------------------------------ #
    # 6. The heritable quantitative trait.                               #
    # ------------------------------------------------------------------ #
    # mediated component : the trait responds to the causal gene's
    #   expression (so every eQTL of that gene is also a trait locus) —
    #   weighted to carry most of the genetic signal so the mediated
    #   locus is the lead genome-wide hit;
    # direct component   : the other causal SNPs act on the trait
    #   directly; structure : a small population-mean shift; noise.
    mediator_effect = 2.2
    mediated = mediator_effect * (
        causal_gene_expr - causal_gene_expr.mean()
    ) / (causal_gene_expr.std() + 1e-9)
    if direct_causal_idx.size:
        G_d = np.column_stack([_std(geno[:, j]) for j in direct_causal_idx])
        d_beta = np.abs(rng.normal(0.0, 1.0, direct_causal_idx.size)) + 0.5
        d_beta *= rng.choice([-1.0, 1.0], direct_causal_idx.size)
        direct = G_d @ d_beta
    else:
        direct = np.zeros(n_samples)
    genetic = mediated + direct
    genetic = genetic / (genetic.std() + 1e-9)            # var = 1
    # Population-structure effect — a small per-population mean shift.
    pop_effect_size = 0.45
    pop_means = rng.normal(0.0, 1.0, n_pops)
    pop_means -= pop_means.mean()
    structure = pop_effect_size * pop_means[pop]
    # Noise scaled so genetic variance / total variance == h2.
    noise_var = genetic.var() * (1.0 - h2) / h2
    noise = rng.normal(0.0, np.sqrt(noise_var), n_samples)
    phenotype_vals = genetic + structure + noise

    # Per-causal-SNP marginal trait effect (additive-coded), for the truth.
    snp_effects = {}
    for j in causal_idx:
        g = geno[:, j].astype(float)
        snp_effects[snp_ids[j]] = float(
            np.cov(g, phenotype_vals)[0, 1] / (g.var() + 1e-9)
        )

    sample_ids = np.array([f"IND{i:05d}" for i in range(n_samples)])

    obs = pd.DataFrame(
        {
            "population": pd.Categorical(
                [f"pop{p}" for p in pop],
                categories=[f"pop{p}" for p in range(n_pops)],
            ),
            "phenotype": phenotype_vals,
        },
        index=sample_ids,
    )
    causal_flag = np.zeros(n_snps, dtype=bool)
    causal_flag[causal_idx] = True
    var = pd.DataFrame(
        {
            "chrom": chrom.astype(int),
            "pos": pos.astype(int),
            "maf": realised_maf,
            "block": block.astype(int),
            "causal": causal_flag,
        },
        index=snp_ids,
    )
    genotype = AnnData(
        X=geno.astype(np.float32), obs=obs, var=var,
    )
    genotype.uns["genetics_source"] = "simulate_gwas_study"

    # ------------------------------------------------------------------ #
    # 7. Bulk expression — the causal gene plus other cis-eQTL genes.     #
    # ------------------------------------------------------------------ #
    # Place each gene on a chromosome / position.
    gene_chrom = rng.integers(1, 23, n_genes)
    gene_pos = rng.integers(1_000_000, 50_000_000, n_genes)
    # The causal gene's cis SNP defines its locus coordinates.
    gene_chrom[0] = chrom[lead_causal_snp_idx]
    gene_pos[0] = pos[lead_causal_snp_idx] + 20_000
    for g_local, snp_j in eqtl_idx_for_gene.items():
        if g_local == 0:
            continue
        gene_chrom[g_local] = chrom[snp_j]
        gene_pos[g_local] = pos[snp_j] + 20_000

    # Build the bulk-expression matrix. The causal gene gets the mediator
    # expression built above (so its eQTLs and the trait truly share the
    # same drivers); the other eGenes get a single cis-eQTL effect.
    expr = rng.normal(0.0, 1.0, size=(n_samples, n_genes))
    expr[:, 0] = causal_gene_expr
    eqtl_effect = 0.7
    for g_local, snp_j in eqtl_idx_for_gene.items():
        if g_local == 0:
            continue
        expr[:, g_local] += eqtl_effect * _std(geno[:, snp_j])

    is_eqtl_gene = np.zeros(n_genes, dtype=bool)
    is_eqtl_gene[egene_idx] = True
    eqtl_snp_col = np.array(
        [eqtl_snp_for_gene.get(g, "") for g in gene_ids], dtype=object,
    )
    expr_var = pd.DataFrame(
        {
            "chrom": gene_chrom.astype(int),
            "pos": gene_pos.astype(int),
            "eqtl_snp": eqtl_snp_col,
            "is_eqtl_gene": is_eqtl_gene,
        },
        index=gene_ids,
    )
    expression = AnnData(
        X=expr.astype(np.float32),
        obs=obs[["population"]].copy(),
        var=expr_var,
    )
    expression.uns["genetics_source"] = "simulate_gwas_study"

    # ------------------------------------------------------------------ #
    # 8. scRNA-seq atlas — causal gene selective in one cell type.        #
    # ------------------------------------------------------------------ #
    cell_types = np.array(["T_cell", "B_cell", "Monocyte", "NK_cell"])
    n_ct = cell_types.size
    ct_label = cell_types[rng.integers(0, n_ct, n_cells)]
    relevant_cell_type = "Monocyte"

    # Poisson counts: a baseline rate per gene, with the causal gene (and
    # the other eGenes, more weakly) up-regulated in the relevant type.
    base_rate = rng.uniform(0.5, 3.0, n_genes)
    counts = rng.poisson(
        base_rate[None, :], size=(n_cells, n_genes),
    ).astype(np.float32)
    in_ct = ct_label == relevant_cell_type
    # Causal gene: strongly selective in the relevant cell type.
    counts[in_ct, 0] += rng.poisson(12.0, size=in_ct.sum()).astype(np.float32)
    # Other eGenes: a weaker selective bump (so the GWAS gene set as a
    # whole points at the same cell type).
    for g_local in egene_idx[1:]:
        counts[in_ct, g_local] += rng.poisson(
            3.0, size=in_ct.sum(),
        ).astype(np.float32)
    # Give each cell type one extra marker gene, for realism.
    for ci, ct in enumerate(cell_types):
        marker = n_genes - 1 - ci
        sel = ct_label == ct
        counts[sel, marker] += rng.poisson(
            10.0, size=sel.sum(),
        ).astype(np.float32)

    sc_obs = pd.DataFrame(
        {"cell_type": pd.Categorical(ct_label, categories=list(cell_types))},
        index=[f"CELL{i:05d}" for i in range(n_cells)],
    )
    scrna = AnnData(
        X=counts,
        obs=sc_obs,
        var=pd.DataFrame(
            {"chrom": gene_chrom.astype(int), "pos": gene_pos.astype(int)},
            index=gene_ids,
        ),
    )
    scrna.uns["genetics_source"] = "simulate_gwas_study"

    # ------------------------------------------------------------------ #
    # 9. Assemble the study + ground truth.                              #
    # ------------------------------------------------------------------ #
    phenotype = pd.Series(phenotype_vals, index=sample_ids, name="phenotype")
    truth = {
        "causal_snps": [str(s) for s in snp_ids[causal_idx]],
        "lead_causal_snp": str(snp_ids[lead_causal_snp_idx]),
        "gene_instruments": [str(s) for s in snp_ids[extra_instruments]],
        "causal_gene": str(causal_gene),
        "causal_block": causal_block,
        "causal_chrom": int(chrom[lead_causal_snp_idx]),
        "eqtl_map": {str(k): str(v) for k, v in eqtl_snp_for_gene.items()},
        "relevant_cell_type": relevant_cell_type,
        "snp_effects": {str(k): v for k, v in snp_effects.items()},
    }
    params = {
        "n_samples": n_samples, "n_snps": n_snps, "n_blocks": n_blocks,
        "n_causal": n_causal, "n_genes": n_genes, "n_pops": n_pops,
        "n_cells": n_cells, "h2": h2, "seed": seed,
    }
    return GWASStudy(
        genotype=genotype, phenotype=phenotype, expression=expression,
        scrna=scrna, truth=truth, params=params,
    )


def _norm_isf(p):
    """Inverse survival function of the standard normal (vectorised)."""
    from scipy.special import ndtri

    return ndtri(1.0 - np.asarray(p, dtype=float))
