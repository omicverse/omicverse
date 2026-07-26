"""High-level SAMap runner used by ``ov.single.samap_integrate``.

Orchestrates the vendored SAMap: build per-species SAM objects from AnnData,
translate BLAST protein IDs to the dataset gene symbols (via the Ensembl peptide
FASTA + BioMart ``ensembl_gene_id -> external_gene_name``), then run SAMAP on a
precomputed BLAST ``maps/`` directory.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import anndata


def _which(name: str, env: dict) -> bool:
    import shutil
    return shutil.which(name, path=env.get("PATH")) is not None


def build_blast_maps(proteomes: Dict[str, str], out_dir: str = "maps",
                     *, engine: str = "auto", evalue: float = 1e-6,
                     max_target_seqs: int = 10, threads: int = 8,
                     sensitivity: str = "sensitive", blast_bin: str = "") -> str:
    """Build the SAMap ``maps/`` directory with reciprocal protein alignment
    between two species' peptide FASTAs.

    The two directions are the slow step. Two accelerations over SAMap's original
    ``map_genes.sh`` (which shells out to ``blastp`` sequentially):

    * ``engine='diamond'`` (default when DIAMOND is on ``PATH``) — DIAMOND is
      ~100-1000x faster than NCBI ``blastp`` at comparable sensitivity, and its
      ``--outfmt 6`` is column-identical to BLAST tabular, so SAMap reads it
      unchanged.
    * the two query directions are run **concurrently**.

    ``engine='blastp'`` reproduces the upstream NCBI path (``blastp -max_hsps 1``).

    Parameters
    ----------
    proteomes
        ``{species_id: peptide_fasta_path}`` — exactly two entries.
    out_dir
        Output ``maps`` directory (created).
    engine
        ``'auto'`` | ``'diamond'`` | ``'blastp'``. ``'auto'`` uses DIAMOND if
        available, else falls back to ``blastp``.
    evalue, max_target_seqs, threads
        Alignment parameters (shared by both engines).
    sensitivity
        DIAMOND sensitivity mode (e.g. ``'sensitive'``, ``'more-sensitive'``,
        ``'ultra-sensitive'``); ignored for ``blastp``.
    blast_bin
        Directory holding the aligner binaries (prepended to ``PATH``).

    Returns
    -------
    str
        Path to ``out_dir`` (pass as ``blast_maps`` to :func:`samap_integrate`).
    """
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    ids = list(proteomes)
    if len(ids) != 2:
        raise ValueError("build_blast_maps supports exactly two species.")
    a, b = ids
    env = dict(os.environ)
    if blast_bin:
        env["PATH"] = blast_bin + os.pathsep + env.get("PATH", "")

    if engine == "auto":
        engine = "diamond" if _which("diamond", env) else "blastp"
    sub = os.path.join(out_dir, f"{a}{b}")
    os.makedirs(sub, exist_ok=True)

    def _run(cmd):
        subprocess.run(cmd, check=True, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # The engine plumbing lives in `ov.alignment.homology_search` — one home for
    # the DIAMOND/BLAST+ command construction instead of a private copy here.
    # SAMap needs the raw outfmt-6 FILES (its loader reads `maps/`), so we ask
    # for `out=` and ignore the returned frame.
    from ...alignment._homology import homology_search

    def _align(pair):
        qid, db = pair
        homology_search(
            proteomes[qid], proteomes[db],
            method="diamond" if engine == "diamond" else "blastp",
            evalue=evalue, max_target_seqs=max_target_seqs, threads=threads,
            sensitivity=sensitivity, bin_dir=blast_bin,
            out=os.path.join(sub, f"{qid}_to_{db}.txt"),
        )

    # Run the two directions concurrently (independent BLAST/DIAMOND jobs).
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(_align, [(a, b), (b, a)]))
    return out_dir


def _protein_to_symbol(fasta_path: str, ensembl_species: str, *,
                       host: str = "https://www.ensembl.org") -> List[tuple]:
    """Map BLAST protein IDs -> dataset gene symbols.

    Parses the Ensembl peptide FASTA (``>PROTEIN ... gene:GENE ...``) for the
    protein->gene link, then resolves gene->symbol via BioMart. Returns the list
    of ``(protein_id, gene_symbol)`` tuples SAMap's ``names`` argument expects.
    """
    import gzip

    from ...single._cross_species import _biomart_query, _resolve_dataset

    opener = gzip.open if fasta_path.endswith(".gz") else open
    p2g = {}
    with opener(fasta_path, "rt") as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            pid = line[1:].split(None, 1)[0]
            m = re.search(r"gene:(\S+)", line)
            if m:
                p2g[pid] = m.group(1).split(".")[0]   # strip gene version

    dataset = _resolve_dataset(ensembl_species)
    df = _biomart_query(dataset, ["ensembl_gene_id", "external_gene_name"], host=host)
    df.columns = ["gene_id", "symbol"]
    g2s = {g: s for g, s in zip(df["gene_id"], df["symbol"]) if isinstance(s, str) and s}

    return [(pid, g2s[gid]) for pid, gid in p2g.items() if gid in g2s]


def samap_integrate(
    adatas: Sequence[anndata.AnnData],
    ids: Sequence[str],
    ensembl_species: Sequence[str],
    *,
    proteomes: Dict[str, str],
    blast_maps: str = "maps",
    host: str = "https://www.ensembl.org",
    keys: Optional[dict] = None,
    resolutions: Optional[dict] = None,
    n_top_genes: int = 3000,
    npcs: int = 100,
    k: int = 20,
    run_kwargs: Optional[dict] = None,
    verbose: bool = False,
) -> anndata.AnnData:
    """Cross-species integration with **SAMap** (BLAST homology + SAM manifold).

    Unlike the one-to-one ortholog route, SAMap builds a *gene-homology graph*
    from reciprocal protein BLAST, so it handles many-to-many homology and large
    evolutionary distances (its reason for existing).

    Parameters
    ----------
    adatas
        One AnnData per species (**raw counts**, gene *symbols* as ``var_names``).
    ids
        Short species IDs (e.g. ``['zf', 'fr']``) — SAMap keys, and the prefixes
        used in the ``maps/`` directory (``maps/zffr/zf_to_fr.txt``).
    ensembl_species
        Ensembl species name per entry (e.g. ``['zebrafish', 'frog']``) — used to
        translate BLAST protein IDs to gene symbols via BioMart.
    proteomes
        ``{id: peptide_fasta_path}`` (the FASTAs BLAST was run on).
    blast_maps
        Path to the SAMap ``maps/`` directory (see :func:`build_blast_maps`).
    host, keys, resolutions, n_top_genes, npcs, k, run_kwargs, verbose
        Passed through to SAM preprocessing / SAMAP.

    Returns
    -------
    anndata.AnnData
        Combined object with SAMap's joint embedding in ``obsm`` and ``species``
        in ``obs``.
    """
    try:
        from .samalg import SAM
        from .mapping import SAMAP
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Vendored SAMap failed to import. It needs `hnswlib`, `numba`, and "
            "`leidenalg`; install those into the environment."
        ) from exc

    ids = list(ids)
    if not (len(adatas) == len(ids) == len(ensembl_species)):
        raise ValueError("adatas, ids and ensembl_species must have equal length.")

    sams = {}
    for aid, ad in zip(ids, adatas):
        s = SAM(counts=ad.copy())
        s.preprocess_data(sum_norm="cell_median", norm="log",
                          thresh_low=0.0, thresh_high=0.96, min_expression=1)
        s.run(preprocessing="StandardScaler", npcs=npcs, weight_PCs=False,
              k=k, n_genes=n_top_genes, weight_mode="rms")
        sams[aid] = s

    names = {aid: _protein_to_symbol(proteomes[aid], sp, host=host)
             for aid, sp in zip(ids, ensembl_species)}

    sm = SAMAP(sams, f_maps=blast_maps.rstrip("/") + "/", names=names,
               keys=keys, resolutions=resolutions)
    sm.run(**(run_kwargs or {}))

    out = sm.samap.adata
    if "species" not in out.obs:
        out.obs["species"] = [x.split("_")[0] for x in out.obs_names]
    out.uns["samap"] = {"ids": ids, "ensembl_species": list(ensembl_species)}
    return out
