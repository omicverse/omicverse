r"""Sequence homology search — DIAMOND and NCBI BLAST+ behind one API.

Why this module exists
----------------------
The fast path was already in the codebase, but it was trapped: SAMap's
``build_blast_maps`` (``omicverse/external/samap/_run.py``) drove DIAMOND with
an ``auto``-detect and a concurrent two-direction run, ~100-1000x faster than
NCBI ``blastp``. Nothing outside SAMap could reach it and it carried no
``@register_function``, so an agent could not discover it either.

This lifts that capability to a first-class, registered API and generalises it:
``method`` selects the *program*, and the engine follows from it — DIAMOND for
its two supported programs, NCBI BLAST+ for the five it uniquely covers.

Both engines emit BLAST tabular ``outfmt 6`` with **identical columns**, which
is what makes one DataFrame contract possible.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Sequence, Union

from .._registry import register_function
from ._cli_utils import build_env, resolve_executable

__all__ = ["homology_search", "reciprocal_best_hits", "BLAST_TAB_COLUMNS"]

#: The 12 columns of BLAST tabular ``outfmt 6`` — DIAMOND's ``--outfmt 6``
#: is column-identical, which is why both engines share one parser.
BLAST_TAB_COLUMNS = [
    "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
    "qstart", "qend", "sstart", "send", "evalue", "bitscore",
]

#: program -> (engine, query molecule, subject molecule).
#: DIAMOND implements protein-database search only (``blastp``/``blastx``);
#: the nucleotide-database programs exist solely in NCBI BLAST+.
_PROGRAMS: Dict[str, Dict[str, Any]] = {
    "diamond":         {"engine": "diamond", "program": "blastp",  "db": "prot"},
    "diamond-blastp":  {"engine": "diamond", "program": "blastp",  "db": "prot"},
    "diamond-blastx":  {"engine": "diamond", "program": "blastx",  "db": "prot"},
    "blastp":          {"engine": "blast",   "program": "blastp",  "db": "prot"},
    "blastx":          {"engine": "blast",   "program": "blastx",  "db": "prot"},
    "blastn":          {"engine": "blast",   "program": "blastn",  "db": "nucl"},
    "tblastn":         {"engine": "blast",   "program": "tblastn", "db": "nucl"},
    "tblastx":         {"engine": "blast",   "program": "tblastx", "db": "nucl"},
}

# Every one of these is a real DIAMOND mode flag, verified against DIAMOND
# 2.2.4 by invoking it. `fast` in particular IS a distinct mode and not merely
# the default — dropping its flag would silently run a different search than the
# caller asked for, which is worse than erroring on an ancient build that
# predates it (the aligner's own stderr is surfaced either way).
_DIAMOND_SENSITIVITY = {
    "faster", "fast", "mid-sensitive", "sensitive", "more-sensitive",
    "very-sensitive", "ultra-sensitive",
}


def _which(name: str, env: Optional[dict] = None) -> Optional[str]:
    path = (env or os.environ).get("PATH")
    return shutil.which(name, path=path)


def resolve_method(method: str = "auto", env: Optional[dict] = None) -> str:
    """Resolve ``method='auto'`` to a concrete program name.

    ``'auto'`` means protein-protein search with whichever engine is installed,
    preferring DIAMOND. Any explicit method is returned unchanged (after
    validation) so a caller can pin the engine for reproducibility.
    """
    if method != "auto":
        if method not in _PROGRAMS:
            raise ValueError(
                f"unknown method {method!r}. Choose one of: "
                + ", ".join(sorted(_PROGRAMS)) + ", or 'auto'."
            )
        return method
    return "diamond" if _which("diamond", env) else "blastp"


def _read_hits(path: str):
    """Parse an ``outfmt 6`` table into a DataFrame (empty table -> empty frame)."""
    import pandas as pd

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=BLAST_TAB_COLUMNS)
    # Sequence IDs stay STRINGS. Left to pandas, an all-numeric FASTA header
    # ("12345", common in NCBI GI and in generated proteomes) is inferred as
    # int64 — and when the two directions of a reciprocal run happen to infer
    # different dtypes for the same column, the merge in `reciprocal_best_hits`
    # dies with a pandas dtype ValueError on data that is perfectly valid.
    df = pd.read_csv(path, sep="\t", header=None, names=BLAST_TAB_COLUMNS,
                     dtype={"qseqid": str, "sseqid": str})
    for col in ("pident", "evalue", "bitscore"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("length", "mismatch", "gapopen", "qstart", "qend", "sstart", "send"):
        df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")
    return df


def _run(cmd: Sequence[str], env: dict, quiet: bool = True) -> None:
    """Run an aligner, surfacing its stderr on failure.

    ``subprocess.run(check=True)`` alone would raise with an empty message and
    the actual cause (a malformed FASTA, a missing db) would be lost.
    """
    proc = subprocess.run(
        list(cmd), env=env,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(
            f"`{cmd[0]} {cmd[1] if len(cmd) > 1 else ''}` failed "
            f"(exit {proc.returncode}): {err or '<no stderr>'}"
        )


def _build_db(engine: str, fasta: str, db_prefix: str, dbtype: str,
              threads: int, env: dict, exe: Dict[str, str]) -> None:
    if engine == "diamond":
        _run([exe["diamond"], "makedb", "--in", fasta, "--db", db_prefix,
              "--threads", str(threads), "--quiet"], env)
    else:
        _run([exe["makeblastdb"], "-in", fasta, "-dbtype", dbtype,
              "-out", db_prefix], env)


def _align_cmd(spec: Dict[str, Any], *, exe: Dict[str, str], query: str,
               db_prefix: str, out_path: str, evalue: float,
               max_target_seqs: int, threads: int, sensitivity: str,
               extra_args: Optional[Sequence[str]]) -> list:
    if spec["engine"] == "diamond":
        cmd = [exe["diamond"], spec["program"], "--query", query, "--db", db_prefix,
               "--outfmt", "6", "--evalue", str(evalue),
               "--max-target-seqs", str(max_target_seqs),
               "--threads", str(threads), "--quiet", "--out", out_path]
        if sensitivity:
            if sensitivity not in _DIAMOND_SENSITIVITY:
                raise ValueError(
                    f"unknown DIAMOND sensitivity {sensitivity!r}; choose from "
                    + ", ".join(sorted(_DIAMOND_SENSITIVITY))
                )
            cmd.append(f"--{sensitivity}")
    else:
        cmd = [exe[spec["program"]], "-query", query, "-db", db_prefix,
               "-outfmt", "6", "-evalue", str(evalue), "-max_hsps", "1",
               "-max_target_seqs", str(max_target_seqs),
               "-num_threads", str(threads), "-out", out_path]
    if extra_args:
        cmd.extend(str(a) for a in extra_args)
    return cmd


def _resolve_binaries(spec: Dict[str, Any], *, bin_dir: str, auto_install: bool,
                      env: dict) -> Dict[str, str]:
    """Resolve each binary to an absolute path, honouring ``bin_dir`` FIRST.

    `resolve_executable` searches the ambient ``PATH`` and the interpreter's own
    ``env/bin`` — it has no idea about ``bin_dir``. SAMap's ``build_blast_maps``
    used to prepend its ``blast_bin`` to ``env["PATH"]`` and then run a bare
    ``diamond``, so a DIAMOND installed ONLY in ``blast_bin`` worked. Resolving
    through `resolve_executable` alone would drop that and raise
    FileNotFoundError where the previous implementation succeeded, so
    ``bin_dir`` gets first refusal here.
    """
    def _find(name: str) -> str:
        found = _which(name, env)
        if found:
            return found
        return resolve_executable(name, auto_install=auto_install)

    exe: Dict[str, str] = {}
    if spec["engine"] == "diamond":
        exe["diamond"] = _find("diamond")
    else:
        exe[spec["program"]] = _find(spec["program"])
        exe["makeblastdb"] = _find("makeblastdb")
    return exe


@register_function(
    aliases=["homology_search", "同源搜索", "blast", "blastp", "diamond",
             "sequence_search", "序列比对", "homology"],
    category="alignment",
    description=(
        "Search a query FASTA against a subject FASTA with DIAMOND or NCBI "
        "BLAST+ and return the hits as a tidy DataFrame (BLAST tabular "
        "outfmt-6 columns). `method` picks the program: 'auto' (DIAMOND if "
        "installed, else blastp), 'diamond' / 'diamond-blastx', or any NCBI "
        "program — 'blastp', 'blastx', 'blastn', 'tblastn', 'tblastx'. DIAMOND "
        "is ~100-1000x faster than blastp at comparable sensitivity, so it is "
        "the default whenever the search is protein-vs-protein."
    ),
    examples=[
        "hits = ov.alignment.homology_search('query.faa', 'subject.faa')",
        "hits = ov.alignment.homology_search('q.faa', 's.faa', method='diamond', sensitivity='ultra-sensitive')",
        "hits = ov.alignment.homology_search('genes.fna', 'genome.fna', method='blastn')",
        "hits = ov.alignment.homology_search('contigs.fna', 'proteins.faa', method='diamond-blastx')",
    ],
    related=["alignment.reciprocal_best_hits", "alignment.mafft", "single.get_orthologs"],
)
def homology_search(
    query: str,
    subject: str,
    *,
    method: str = "auto",
    evalue: float = 1e-5,
    max_target_seqs: int = 10,
    threads: int = 8,
    sensitivity: str = "sensitive",
    out: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    bin_dir: str = "",
    auto_install: bool = False,
):
    r"""Align ``query`` against ``subject`` and return the hit table.

    Arguments
    ---------
    query, subject
        Paths to FASTA files. ``subject`` is turned into a search database
        (DIAMOND ``makedb`` or NCBI ``makeblastdb``) in a temporary directory,
        unless ``out`` is given, in which case the database is kept beside it.
    method
        ``'auto'`` | ``'diamond'`` | ``'diamond-blastp'`` | ``'diamond-blastx'``
        | ``'blastp'`` | ``'blastx'`` | ``'blastn'`` | ``'tblastn'`` |
        ``'tblastx'``. ``'auto'`` runs protein-protein search with DIAMOND when
        it is on ``PATH``, else ``blastp``. Pin an explicit method when you need
        a reproducible engine.
    evalue, max_target_seqs, threads
        Shared alignment parameters.
    sensitivity
        DIAMOND sensitivity mode (``fast`` | ``mid-sensitive`` | ``sensitive`` |
        ``more-sensitive`` | ``very-sensitive`` | ``ultra-sensitive``). Ignored
        by the NCBI engine.
    out
        Where to write the ``outfmt 6`` table. Omit to use a temporary file
        (the DataFrame is still returned).
    extra_args
        Raw flags appended to the aligner command, for anything this wrapper
        does not model.
    bin_dir
        Directory holding the aligner binaries, prepended to ``PATH``.
    auto_install
        Try to install a missing aligner (conda/bioconda) before failing.

    Returns
    -------
    pandas.DataFrame
        One row per HSP with the 12 ``outfmt 6`` columns
        (:data:`BLAST_TAB_COLUMNS`), sorted by ``evalue`` then descending
        ``bitscore``. Empty when nothing passed the threshold.

    Examples
    --------
    >>> import omicverse as ov
    >>> hits = ov.alignment.homology_search("designed.faa", "uniprot_sprot.faa")
    >>> hits.head()
    """
    for label, path in (("query", query), ("subject", subject)):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} FASTA not found: {path}")

    env = build_env(extra_paths=[bin_dir] if bin_dir else None)
    resolved = resolve_method(method, env)
    spec = _PROGRAMS[resolved]
    exe = _resolve_binaries(spec, bin_dir=bin_dir, auto_install=auto_install, env=env)

    # The search database is an intermediate artifact and ALWAYS lives in a
    # private temp dir — never beside `out`. Putting it next to `out` with a
    # fixed name made two concurrent searches writing into the same directory
    # (exactly what a reciprocal run does) clobber each other's database, so one
    # direction silently searched against the wrong subject.
    tmp = tempfile.TemporaryDirectory(prefix="ov_homology_")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        out_path = out
    else:
        out_path = os.path.join(tmp.name, "hits.tsv")

    try:
        db_prefix = os.path.join(tmp.name, "subjectdb")
        _build_db(spec["engine"], subject, db_prefix, spec["db"], threads, env, exe)
        _run(_align_cmd(spec, exe=exe, query=query, db_prefix=db_prefix,
                        out_path=out_path, evalue=evalue,
                        max_target_seqs=max_target_seqs, threads=threads,
                        sensitivity=sensitivity, extra_args=extra_args), env)
        hits = _read_hits(out_path)
    finally:
        tmp.cleanup()

    if len(hits):
        hits = hits.sort_values(["evalue", "bitscore"], ascending=[True, False],
                                kind="mergesort").reset_index(drop=True)
    hits.attrs["method"] = resolved
    hits.attrs["engine"] = spec["engine"]
    return hits


@register_function(
    aliases=["reciprocal_best_hits", "RBH", "双向最佳匹配", "rbh", "ortholog_rbh"],
    category="alignment",
    description=(
        "Reciprocal best hits between two FASTAs — the standard operational "
        "definition of a one-to-one ortholog pair when no curated orthology "
        "table exists. Runs both directions concurrently (DIAMOND by default) "
        "and keeps pairs that are each other's top hit."
    ),
    examples=[
        "rbh = ov.alignment.reciprocal_best_hits('speciesA.faa', 'speciesB.faa')",
        "rbh = ov.alignment.reciprocal_best_hits('a.faa', 'b.faa', method='blastp', threads=16)",
    ],
    related=["alignment.homology_search", "single.get_orthologs", "tl.cross_species_align"],
)
def reciprocal_best_hits(
    fasta_a: str,
    fasta_b: str,
    *,
    method: str = "auto",
    evalue: float = 1e-5,
    threads: int = 8,
    sensitivity: str = "sensitive",
    max_target_seqs: int = 5,
    bin_dir: str = "",
    auto_install: bool = False,
):
    r"""Reciprocal best hits (RBH) between two FASTA files.

    A pair ``(a, b)`` is kept when ``b`` is the best hit for ``a`` in the
    A->B search **and** ``a`` is the best hit for ``b`` in B->A. "Best" is the
    lowest e-value, ties broken by the higher bitscore.

    Note
    ----
    For species with curated Ensembl orthology, prefer
    :func:`omicverse.single.get_orthologs` — looking up a precomputed
    one-to-one table is both faster and better curated than re-deriving it.
    RBH is for the cases that lookup cannot serve: unannotated assemblies,
    distant species, or your own designed sequences.

    Returns
    -------
    pandas.DataFrame
        Columns ``a``, ``b``, ``evalue_ab``, ``bitscore_ab``, ``pident_ab``,
        ``evalue_ba``, ``bitscore_ba``, ``pident_ba``.
    """
    import pandas as pd

    def _best(df):
        if not len(df):
            return df.assign(_k=[]).iloc[0:0]
        ranked = df.sort_values(["qseqid", "evalue", "bitscore"],
                                ascending=[True, True, False], kind="mergesort")
        return ranked.groupby("qseqid", as_index=False, sort=False).head(1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fwd_f = pool.submit(homology_search, fasta_a, fasta_b, method=method,
                            evalue=evalue, threads=threads, sensitivity=sensitivity,
                            max_target_seqs=max_target_seqs, bin_dir=bin_dir,
                            auto_install=auto_install)
        rev_f = pool.submit(homology_search, fasta_b, fasta_a, method=method,
                            evalue=evalue, threads=threads, sensitivity=sensitivity,
                            max_target_seqs=max_target_seqs, bin_dir=bin_dir,
                            auto_install=auto_install)
        fwd, rev = _best(fwd_f.result()), _best(rev_f.result())

    if not len(fwd) or not len(rev):
        return pd.DataFrame(columns=["a", "b", "evalue_ab", "bitscore_ab", "pident_ab",
                                     "evalue_ba", "bitscore_ba", "pident_ba"])

    ab = fwd.rename(columns={"qseqid": "a", "sseqid": "b", "evalue": "evalue_ab",
                             "bitscore": "bitscore_ab", "pident": "pident_ab"})
    ba = rev.rename(columns={"qseqid": "b", "sseqid": "a", "evalue": "evalue_ba",
                             "bitscore": "bitscore_ba", "pident": "pident_ba"})
    merged = ab[["a", "b", "evalue_ab", "bitscore_ab", "pident_ab"]].merge(
        ba[["a", "b", "evalue_ba", "bitscore_ba", "pident_ba"]], on=["a", "b"], how="inner")
    return merged.sort_values(["evalue_ab", "bitscore_ab"],
                              ascending=[True, False], kind="mergesort").reset_index(drop=True)
