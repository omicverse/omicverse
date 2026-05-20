"""I/O for ``ov.genetics`` — GWAS summary statistics, PLINK and VCF.

Flexible readers (and writers) for the heterogeneous file formats of
statistical genetics: :func:`read_sumstats` parses a GWAS
summary-statistics table with column-name auto-detection,
:func:`read_plink` loads a PLINK ``.bed`` / ``.bim`` / ``.fam`` triple
into a samples x SNPs AnnData, :func:`read_vcf` extracts a genotype
dosage matrix from a VCF, and :func:`write_plink` / :func:`write_sumstats`
serialise a genotype AnnData / a GWAS results table back out in the
PLINK and GWAS-summary formats that downstream tools (and these readers)
consume.
"""
from __future__ import annotations

import gzip
import os
from typing import Optional

import numpy as np
import pandas as pd

from .._registry import register_function


# Canonical column name -> recognised aliases (lower-case).
_SUMSTAT_ALIASES = {
    "SNP": ["snp", "rsid", "rs", "variant", "variant_id", "markername",
            "id", "snpid"],
    "CHR": ["chr", "chrom", "chromosome", "#chrom"],
    "BP": ["bp", "pos", "position", "base_pair_location"],
    "A1": ["a1", "effect_allele", "allele1", "ea", "alt"],
    "A2": ["a2", "other_allele", "allele2", "non_effect_allele", "nea", "ref"],
    "BETA": ["beta", "effect", "b", "effect_size"],
    "SE": ["se", "stderr", "standard_error", "se_beta"],
    "OR": ["or", "odds_ratio"],
    "Z": ["z", "zscore", "z_score", "zstat"],
    "P": ["p", "pval", "pvalue", "p_value", "p-value", "p.value"],
    "N": ["n", "samplesize", "n_total", "sample_size"],
    "EAF": ["eaf", "freq", "maf", "af", "effect_allele_frequency"],
    "INFO": ["info", "imputation_info", "rsq"],
}


@register_function(
    aliases=[
        "read_sumstats", "read_gwas", "load_sumstats", "gwas_reader",
        "读取GWAS汇总统计", "GWAS数据读取", "汇总统计读取",
    ],
    category="genetics",
    description=(
        "Flexible reader for a GWAS summary-statistics file. Auto-detects "
        "the delimiter and the column names (SNP / CHR / BP / A1 / A2 / "
        "BETA / SE / OR / Z / P / N / EAF / INFO) from a wide set of "
        "aliases used across GWAS Catalog, PLINK, BOLT-LMM, SAIGE, "
        "Regenie and consortium releases, and returns a tidy "
        "DataFrame with canonical column names. Pure pandas."
    ),
    examples=[
        "ov.genetics.read_sumstats('gwas.txt.gz')",
        "ov.genetics.read_sumstats('gwas.tsv', sep='\\t', rename=False)",
    ],
    related=["ov.genetics.munge_sumstats", "ov.genetics.gwas_association",
             "ov.genetics.manhattan"],
)
def read_sumstats(
    path: str,
    *,
    sep: Optional[str] = None,
    rename: bool = True,
    nrows: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """Read a GWAS summary-statistics file.

    Parameters
    ----------
    path
        Path to the summary-statistics file (plain or ``.gz``).
    sep
        Field delimiter; ``None`` (default) auto-detects whitespace / TSV
        / CSV.
    rename
        If ``True`` (default), rename detected columns to the canonical
        names (``SNP``, ``CHR``, ``BP``, ``A1``, ``A2``, ``BETA``, ``SE``,
        ``OR``, ``Z``, ``P``, ``N``, ``EAF``, ``INFO``).
    nrows
        Optionally read only the first ``nrows`` rows.
    **kwargs
        Forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        The summary-statistics table; ``.attrs['sumstats_columns']`` maps
        each canonical name to the original column it came from.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"sumstats file not found: {path}")

    read_kw = dict(kwargs)
    if sep is None:
        read_kw.setdefault("sep", r"\s+|,|\t")
        read_kw.setdefault("engine", "python")
    else:
        read_kw["sep"] = sep
    if nrows is not None:
        read_kw["nrows"] = nrows

    df = pd.read_csv(path, **read_kw)

    # Detect canonical columns.
    lower_map = {c.lower().strip(): c for c in df.columns}
    detected: dict[str, str] = {}
    for canon, aliases in _SUMSTAT_ALIASES.items():
        for alias in aliases:
            if alias in lower_map:
                detected[canon] = lower_map[alias]
                break

    if rename:
        df = df.rename(columns={orig: canon for canon, orig in detected.items()})

    df.attrs["sumstats_columns"] = detected
    return df


def _read_bim(prefix: str) -> pd.DataFrame:
    bim = pd.read_csv(
        prefix + ".bim", sep=r"\s+", header=None,
        names=["chr", "snp", "cm", "bp", "a1", "a2"],
        dtype={"snp": str},
    )
    return bim


def _read_fam(prefix: str) -> pd.DataFrame:
    fam = pd.read_csv(
        prefix + ".fam", sep=r"\s+", header=None,
        names=["fid", "iid", "father", "mother", "sex", "phenotype"],
        dtype={"fid": str, "iid": str},
    )
    return fam


def _read_bed(prefix: str, n_samples: int, n_snps: int) -> np.ndarray:
    """Read a PLINK SNP-major ``.bed`` into a (samples, snps) dosage array."""
    with open(prefix + ".bed", "rb") as fh:
        magic = fh.read(3)
        if magic[:2] != b"\x6c\x1b":
            raise ValueError(f"{prefix}.bed is not a valid PLINK .bed file.")
        if magic[2] != 1:
            raise ValueError(
                f"{prefix}.bed is not SNP-major; only SNP-major .bed is "
                "supported."
            )
        raw = np.frombuffer(fh.read(), dtype=np.uint8)

    bytes_per_snp = (n_samples + 3) // 4
    if raw.size != bytes_per_snp * n_snps:
        raise ValueError(
            f"{prefix}.bed size mismatch: expected "
            f"{bytes_per_snp * n_snps} bytes, got {raw.size}."
        )
    raw = raw.reshape(n_snps, bytes_per_snp)
    # Unpack 2 bits per genotype, 4 genotypes per byte (little-endian order).
    geno_codes = np.zeros((n_snps, bytes_per_snp * 4), dtype=np.uint8)
    for shift in range(4):
        geno_codes[:, shift::4] = (raw >> (2 * shift)) & 0b11
    geno_codes = geno_codes[:, :n_samples]
    # PLINK 2-bit codes -> dosage of A1 allele:
    #   00 -> 2 (hom A1), 01 -> NaN (missing), 10 -> 1 (het), 11 -> 0 (hom A2).
    dosage = np.full(geno_codes.shape, np.nan, dtype=np.float32)
    dosage[geno_codes == 0] = 2.0
    dosage[geno_codes == 2] = 1.0
    dosage[geno_codes == 3] = 0.0
    # Return (samples, snps).
    return dosage.T


@register_function(
    aliases=[
        "read_plink", "load_plink", "plink_reader", "read_bed",
        "读取PLINK", "PLINK数据读取", "基因型读取",
    ],
    category="genetics",
    description=(
        "Read a PLINK binary genotype fileset (``.bed`` / ``.bim`` / "
        "``.fam``) into a samples x SNPs AnnData. ``.X`` holds the A1-"
        "allele dosage (0 / 1 / 2, NaN for missing), ``.obs`` carries the "
        "sample table (FID / IID / sex / phenotype) and ``.var`` carries "
        "the SNP table (chromosome / position / alleles). Pure numpy — "
        "reads the SNP-major .bed bitstream directly."
    ),
    examples=[
        "ov.genetics.read_plink('cohort')",
        "adata = ov.genetics.read_plink('/data/study/plink')",
    ],
    related=["ov.genetics.gwas_qc", "ov.genetics.gwas_association",
             "ov.genetics.read_vcf"],
)
def read_plink(prefix: str):
    """Read a PLINK ``.bed`` / ``.bim`` / ``.fam`` fileset.

    Parameters
    ----------
    prefix
        Fileset prefix; ``prefix.bed``, ``prefix.bim`` and ``prefix.fam``
        must all exist.

    Returns
    -------
    AnnData
        ``samples x SNPs`` AnnData with A1-allele dosages in ``.X``.
    """
    import anndata

    # Allow passing the .bed path directly.
    if prefix.endswith(".bed"):
        prefix = prefix[:-4]
    for ext in (".bed", ".bim", ".fam"):
        if not os.path.exists(prefix + ext):
            raise FileNotFoundError(f"missing PLINK file: {prefix + ext}")

    bim = _read_bim(prefix)
    fam = _read_fam(prefix)
    dosage = _read_bed(prefix, len(fam), len(bim))

    obs = fam.copy()
    obs.index = obs["iid"].astype(str)
    var = bim.copy()
    var.index = var["snp"].astype(str)

    adata = anndata.AnnData(X=dosage, obs=obs, var=var)
    adata.uns["genetics_source"] = "plink"
    return adata


@register_function(
    aliases=[
        "read_vcf", "load_vcf", "vcf_reader", "读取VCF", "VCF数据读取",
    ],
    category="genetics",
    description=(
        "Read a (small) VCF file into a samples x variants AnnData of "
        "genotype dosages. Parses the GT field (counting ALT alleles; "
        "missing calls become NaN), with ``.var`` carrying CHROM / POS / "
        "REF / ALT. Intended for modest VCFs — for large cohorts use "
        ":func:`read_plink`. Pure-Python parser."
    ),
    examples=[
        "ov.genetics.read_vcf('variants.vcf')",
        "ov.genetics.read_vcf('variants.vcf.gz')",
    ],
    related=["ov.genetics.read_plink", "ov.genetics.gwas_qc"],
)
def read_vcf(path: str, *, max_variants: Optional[int] = None):
    """Read a VCF into a genotype-dosage AnnData.

    Parameters
    ----------
    path
        Path to the VCF file (plain or ``.gz``).
    max_variants
        Optional cap on the number of variants read.

    Returns
    -------
    AnnData
        ``samples x variants`` AnnData of ALT-allele dosages.
    """
    import anndata

    if not os.path.exists(path):
        raise FileNotFoundError(f"VCF file not found: {path}")

    opener = gzip.open if path.endswith(".gz") else open
    samples: list[str] = []
    var_rows: list[dict] = []
    dosage_rows: list[np.ndarray] = []

    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                continue
            if not samples:
                raise ValueError("VCF has no #CHROM header line.")
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                continue
            chrom, pos, vid, ref, alt = fields[:5]
            fmt = fields[8].split(":")
            try:
                gt_idx = fmt.index("GT")
            except ValueError:
                continue
            dos = np.full(len(samples), np.nan, dtype=np.float32)
            for k, cell in enumerate(fields[9:]):
                gt = cell.split(":")[gt_idx]
                alleles = gt.replace("|", "/").split("/")
                if any(a in (".", "") for a in alleles):
                    continue
                try:
                    dos[k] = float(sum(int(a) > 0 for a in alleles))
                except ValueError:
                    continue
            var_rows.append({"chr": chrom, "bp": pos, "snp": vid,
                             "ref": ref, "alt": alt})
            dosage_rows.append(dos)
            if max_variants is not None and len(var_rows) >= max_variants:
                break

    if not var_rows:
        raise ValueError(f"no variant records parsed from {path}")

    var = pd.DataFrame(var_rows)
    var.index = [v if v not in (".", "") else f"var{i}"
                 for i, v in enumerate(var["snp"])]
    X = np.vstack(dosage_rows).T  # samples x variants
    obs = pd.DataFrame(index=[str(s) for s in samples])
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.uns["genetics_source"] = "vcf"
    return adata


@register_function(
    aliases=[
        "write_plink", "save_plink", "to_plink", "plink_writer",
        "写入PLINK", "保存PLINK", "导出PLINK",
    ],
    category="genetics",
    description=(
        "Write a genotype AnnData out as a PLINK binary fileset "
        "(``.bed`` / ``.bim`` / ``.fam``) — the round-trip partner of "
        ":func:`read_plink`. Packs ``.X`` (0/1/2 A1-allele dosages) into "
        "the SNP-major 2-bit ``.bed`` bitstream and writes the SNP table "
        "(``.bim``) and sample table (``.fam``). This is the format "
        "downstream tools such as LDSC and PLINK consume. Pure numpy."
    ),
    examples=[
        "ov.genetics.write_plink(geno_qc, './cohort')",
        "ov.genetics.write_plink(geno_qc, prefix, a1='A', a2='G')",
    ],
    related=["ov.genetics.read_plink", "ov.genetics.heritability",
             "ov.genetics.write_sumstats"],
)
def write_plink(
    adata,
    prefix: str,
    *,
    chrom: str = "chrom",
    pos: str = "pos",
    a1: Union[str, "pd.Series"] = "A",
    a2: Union[str, "pd.Series"] = "G",
) -> str:
    """Write a genotype AnnData as a PLINK ``.bed`` / ``.bim`` / ``.fam``.

    Parameters
    ----------
    adata
        Genotype AnnData of ``samples x SNPs`` — ``.X`` holds 0/1/2
        A1-allele dosages.
    prefix
        Output fileset prefix; ``prefix.bed``, ``.bim``, ``.fam`` are
        written.
    chrom, pos
        ``.var`` columns carrying the SNP chromosome / position.
    a1, a2
        A1 (effect) and A2 (other) alleles — a constant string applied to
        every SNP, or a per-SNP Series aligned to ``.var``.

    Returns
    -------
    str
        The fileset ``prefix``.
    """
    parent = os.path.dirname(prefix)
    if parent:
        os.makedirs(parent, exist_ok=True)

    X = np.asarray(adata.X, dtype=float)
    if hasattr(adata.X, "toarray"):
        X = adata.X.toarray().astype(float)
    n, m = X.shape

    # .fam — sample table.
    pd.DataFrame({
        "fid": adata.obs_names, "iid": adata.obs_names,
        "pat": 0, "mat": 0, "sex": 0, "phen": -9,
    }).to_csv(prefix + ".fam", sep=" ", header=False, index=False)

    # .bim — SNP table.
    a1_col = (adata.var[a1] if isinstance(a1, str) and a1 in adata.var.columns
              else a1)
    a2_col = (adata.var[a2] if isinstance(a2, str) and a2 in adata.var.columns
              else a2)
    pd.DataFrame({
        "chr": adata.var[chrom].to_numpy(),
        "snp": adata.var_names,
        "cm": 0,
        "bp": adata.var[pos].to_numpy(),
        "a1": a1_col,
        "a2": a2_col,
    }).to_csv(prefix + ".bim", sep="\t", header=False, index=False)

    # .bed — SNP-major 2-bit bitstream. Dosage->code is the inverse of
    # _read_bed: 2 (hom A1) -> 00, 1 (het) -> 10, 0 (hom A2) -> 11,
    # NaN (missing) -> 01.
    geno = np.rint(X)
    code = np.full((n, m), 1, dtype=np.uint8)            # default = missing
    code[geno == 2] = 0
    code[geno == 1] = 2
    code[geno == 0] = 3
    with open(prefix + ".bed", "wb") as fh:
        fh.write(bytes([0x6c, 0x1b, 0x01]))              # magic + SNP-major
        for j in range(m):
            col = code[:, j]
            buf = bytearray((n + 3) // 4)
            for i in range(n):
                buf[i // 4] |= int(col[i]) << (2 * (i % 4))
            fh.write(bytes(buf))
    return prefix


@register_function(
    aliases=[
        "write_sumstats", "save_sumstats", "to_sumstats", "sumstats_writer",
        "写入GWAS汇总统计", "保存汇总统计", "导出汇总统计",
    ],
    category="genetics",
    description=(
        "Write a GWAS association results table to disk in the standard "
        "GWAS-summary format — one row per SNP with canonical column "
        "names (SNP / CHR / BP / A1 / A2 / BETA / SE / Z / P / N), sorted "
        "by genomic position. This is the format a study shares publicly "
        "and the format :func:`read_sumstats` reads back. Pure pandas."
    ),
    examples=[
        "ov.genetics.write_sumstats(res_adj, './gwas_sumstats.tsv')",
        "ov.genetics.write_sumstats(res, path, chrom='chrom', pos='pos')",
    ],
    related=["ov.genetics.read_sumstats", "ov.genetics.gwas_association",
             "ov.genetics.munge_sumstats"],
)
def write_sumstats(
    results: pd.DataFrame,
    path: str,
    *,
    snp: str = "snp",
    chrom: str = "chrom",
    pos: str = "pos",
    beta: str = "beta",
    se: str = "se",
    stat: Optional[str] = "stat",
    pvalue: str = "pvalue",
    n: Optional[str] = "n",
    a1: str = "A",
    a2: str = "G",
    sep: str = "\t",
) -> pd.DataFrame:
    """Write GWAS summary statistics in the standard sumstats format.

    Parameters
    ----------
    results
        A GWAS association results table (e.g. from
        :func:`ov.genetics.gwas_association`, merged with SNP annotation).
    path
        Output file path (``.tsv`` / ``.txt``; ``.gz`` is gzip-compressed).
    snp, chrom, pos, beta, se, stat, pvalue, n
        Source column names; ``stat`` and ``n`` are optional (pass
        ``None`` to omit).
    a1, a2
        Effect (A1) and other (A2) alleles written for every SNP.
    sep
        Field delimiter (default tab).

    Returns
    -------
    pandas.DataFrame
        The written sumstats table (canonical columns, position-sorted).
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    cols = {snp: "SNP", chrom: "CHR", pos: "BP", beta: "BETA",
            se: "SE", pvalue: "P"}
    if stat is not None and stat in results.columns:
        cols[stat] = "STAT"
    if n is not None and n in results.columns:
        cols[n] = "N"
    out = results[list(cols)].rename(columns=cols).copy()
    out["A1"] = a1
    out["A2"] = a2
    out["Z"] = out["BETA"] / out["SE"]
    order = [c for c in ["SNP", "CHR", "BP", "A1", "A2", "BETA", "SE",
                         "STAT", "Z", "P", "N"] if c in out.columns]
    out = (out[order]
           .sort_values(["CHR", "BP"])
           .reset_index(drop=True))
    out.to_csv(path, sep=sep, index=False)
    return out
