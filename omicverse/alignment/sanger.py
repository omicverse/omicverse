r"""Sanger sequencing (.ab1) — basecalling, reference alignment, and the
construct-verification loop.

Why this lives in ``ov.alignment``
---------------------------------
Reading a chromatogram is a sequence-alignment job that shells out to an
external binary — exactly the shape this module already has for ``mafft`` /
``fastp`` / ``STAR`` (see :func:`omicverse.alignment._cli_utils.resolve_executable`).
It also closes a loop the rest of omicverse opens: ``ov.synbio`` designs a
construct and writes its GenBank, and this is where the sequencing that came
back gets checked against it.

Engine
------
`tracy <https://github.com/gear-genomics/tracy>`_ (EMBL Heidelberg, BSD) does
the heavy lifting: peak-ratio basecalling, kmer-anchored alignment to a
reference, and — the part nothing else open-source does well — **decomposition
of a mixed trace**, which is how a heterozygous variant or a CRISPR-edited
allele is read off a single overlapping chromatogram.

``read_ab1`` alone works without tracy (Biopython's ABI parser), so a caller can
always at least load a trace.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .._registry import register_function
from ._cli_utils import build_env, resolve_executable

__all__ = ["SangerTrace", "read_ab1", "basecall", "align_to_reference",
           "decompose_trace", "verify_construct"]

_TRACY_HINT = (
    "tracy is required for this step. Install it with "
    "`conda install -c bioconda tracy` (or set `tracy_path=`)."
)


@dataclass
class SangerTrace:
    """One chromatogram: the called sequence plus the four dye channels."""

    path: str
    #: Primary basecall (the sequence you would submit).
    sequence: str = ""
    #: Secondary basecall — where it differs from :attr:`sequence` the trace is
    #: mixed (two alleles, or a poor read).
    secondary: str = ""
    #: Per-base quality; empty when the engine did not report it.
    quality: List[int] = field(default_factory=list)
    #: Trace sample index of each called base (x-position for a viewer).
    basecall_pos: List[int] = field(default_factory=list)
    #: Raw dye channels, ``{'A': [...], 'C': [...], 'G': [...], 'T': [...]}``.
    peaks: Dict[str, List[float]] = field(default_factory=dict)

    def __len__(self) -> int:            # noqa: D105
        return len(self.sequence)

    def __repr__(self) -> str:           # noqa: D105
        return (f"SangerTrace({os.path.basename(self.path)!r}, "
                f"{len(self.sequence)} bp, {len(self.peaks)} channels)")

    @property
    def mixed_positions(self) -> List[int]:
        """0-based positions where primary and secondary basecalls disagree.

        A run of these is the signature of a heterozygous site or an edited
        pool; :func:`decompose_trace` is what resolves them into alleles.
        """
        return [i for i, (p, s) in enumerate(zip(self.sequence, self.secondary))
                if s and p != s and s != "N" and p != "N"]

    def to_fasta(self, path: str, name: Optional[str] = None) -> str:
        """Write the primary basecall as FASTA and return the path."""
        name = name or os.path.splitext(os.path.basename(self.path))[0]
        with open(path, "w") as fh:
            fh.write(f">{name}\n")
            for i in range(0, len(self.sequence), 60):
                fh.write(self.sequence[i:i + 60] + "\n")
        return path


def _tracy(tracy_path: str = "", auto_install: bool = False) -> str:
    try:
        return resolve_executable("tracy", explicit=tracy_path or None,
                                  auto_install=auto_install)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc}\n{_TRACY_HINT}") from exc


def _run(cmd: List[str], env: dict) -> None:
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(f"`tracy {cmd[1] if len(cmd) > 1 else ''}` failed "
                           f"(exit {proc.returncode}): {err or '<no stderr>'}")


def _trace_from_tracy_json(path: str, doc: Dict[str, Any]) -> SangerTrace:
    """Build a :class:`SangerTrace` from ``tracy basecall -f json`` output.

    Keys verified against tracy 0.8.1: ``pos``, ``peakA/C/G/T``,
    ``basecallPos``, ``basecallQual``, ``basecalls``, ``primarySeq``,
    ``secondarySeq``.
    """
    return SangerTrace(
        path=path,
        sequence=doc.get("primarySeq", "") or "",
        secondary=doc.get("secondarySeq", "") or "",
        quality=[int(q) for q in doc.get("basecallQual", []) or []],
        basecall_pos=[int(p) for p in doc.get("basecallPos", []) or []],
        peaks={b: list(doc.get(f"peak{b}", []) or []) for b in "ACGT"},
    )


@register_function(
    aliases=["read_ab1", "sanger", "chromatogram", "读取ab1", "测序峰图", "abi"],
    category="alignment",
    description=(
        "Read a Sanger .ab1/.abi chromatogram into a SangerTrace: primary and "
        "secondary basecalls, per-base quality, and the four dye channels for "
        "plotting. Uses tracy when available (it also reports the secondary "
        "basecall that reveals a mixed trace) and falls back to Biopython's ABI "
        "parser so a trace can always be loaded."
    ),
    examples=[
        "tr = ov.alignment.read_ab1('clone_A.ab1')",
        "tr.sequence[:60]; tr.mixed_positions",
    ],
    related=["alignment.verify_construct", "alignment.decompose_trace", "synbio.write_genbank"],
)
def read_ab1(path: str, *, engine: str = "auto", pratio: float = 0.33,
             tracy_path: str = "", auto_install: bool = False) -> SangerTrace:
    r"""Load a chromatogram.

    Arguments
    ---------
    path
        ``.ab1`` / ``.abi`` file.
    engine
        ``'auto'`` (tracy when installed, else Biopython), ``'tracy'``, or
        ``'biopython'``. Only tracy reports the **secondary** basecall, so
        :attr:`SangerTrace.mixed_positions` is empty under Biopython.
    pratio
        tracy's peak ratio for calling a base.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"trace not found: {path}")
    if engine not in ("auto", "tracy", "biopython"):
        raise ValueError("engine must be 'auto', 'tracy' or 'biopython'")

    # Resolve with the SAME rule the rest of the module uses. `shutil.which`
    # alone only looks at PATH, so a tracy sitting in the interpreter's own
    # env bin (where `resolve_executable` finds it) was missed and `auto`
    # silently degraded to Biopython — no secondary basecall, no dye channels,
    # and no warning that the trace had been read the lesser way.
    exe = None
    if engine in ("auto", "tracy"):
        try:
            exe = _tracy(tracy_path, auto_install)
        except FileNotFoundError:
            # Falling back is only correct when tracy was merely ABSENT. If the
            # caller named a path, a typo in it must not degrade silently to the
            # lesser reader — they asked for a specific binary.
            if engine == "tracy" or tracy_path:
                raise
    if exe:
        env = build_env()
        with tempfile.TemporaryDirectory(prefix="ov_sanger_") as tmp:
            out = os.path.join(tmp, "bc.json")
            _run([exe, "basecall", "-f", "json", "-p", str(pratio), "-o", out, path], env)
            with open(out) as fh:
                return _trace_from_tracy_json(path, json.load(fh))

    try:
        from Bio import SeqIO
    except ImportError as exc:                                    # pragma: no cover
        raise ImportError(
            "reading .ab1 without tracy needs Biopython (`pip install biopython`), "
            f"or install tracy. {_TRACY_HINT}"
        ) from exc
    rec = SeqIO.read(path, "abi")
    quals = list(rec.letter_annotations.get("phred_quality", []))

    # Pull the dye channels and basecall positions out of the raw ABIF
    # directories too. An earlier cut returned only the sequence and quality,
    # which made this path look like a success while handing every caller an
    # unplottable trace — a viewer drew a blank canvas with no error to explain
    # it. Only the SECONDARY basecall genuinely needs tracy (it comes from
    # tracy's peak-ratio calling), so that stays empty and `mixed_positions`
    # stays correctly empty with it.
    #
    # DATA9-12 are the processed (baseline-corrected, mobility-shifted) traces;
    # DATA1-4 are the raw ones. FWO_1 is the filter-wheel order — the string
    # that says which channel is which base, e.g. b"GATC" means DATA9 is G.
    # PLOC2 is the edited peak location per base, PLOC1 the original.
    raw = rec.annotations.get("abif_raw", {}) or {}
    order = raw.get("FWO_1") or b"GATC"
    if isinstance(order, str):
        order = order.encode()
    peaks: Dict[str, List[float]] = {}
    for i, base in enumerate(order.decode("ascii", "replace")[:4]):
        channel = raw.get(f"DATA{9 + i}")
        if channel is not None and base in "ACGT":
            peaks[base] = [float(v) for v in channel]
    positions = raw.get("PLOC2")
    if positions is None:
        positions = raw.get("PLOC1")

    return SangerTrace(
        path=path,
        sequence=str(rec.seq),
        secondary="",
        quality=quals,
        basecall_pos=[int(p) for p in (positions or [])],
        peaks=peaks,
    )


@register_function(
    aliases=["basecall", "碱基识别", "sanger_basecall"],
    category="alignment",
    description=(
        "Basecall a Sanger .ab1 trace with tracy and return the sequence "
        "(optionally writing FASTA/FASTQ). Trimming stringency and peak ratio "
        "are exposed because a raw trace's ends are usually unusable."
    ),
    examples=[
        "seq = ov.alignment.basecall('clone_A.ab1', trim=5)",
        "ov.alignment.basecall('clone_A.ab1', out='clone_A.fasta', fmt='fasta')",
    ],
    related=["alignment.read_ab1", "alignment.align_to_reference"],
)
def basecall(path: str, *, fmt: str = "fasta", out: Optional[str] = None,
             otype: str = "primary", trim: int = 0, pratio: float = 0.33,
             tracy_path: str = "", auto_install: bool = False) -> str:
    r"""Basecall a trace; returns the sequence (and writes ``out`` when given).

    ``fmt`` is ``fasta`` | ``fastq`` | ``tsv`` | ``json``; ``otype`` selects
    ``primary`` | ``secondary`` | ``consensus``. ``trim`` is tracy's stringency
    (1-9, 0 = no automatic trimming).
    """
    if fmt not in ("fasta", "fastq", "tsv", "json"):
        raise ValueError("fmt must be 'fasta', 'fastq', 'tsv' or 'json'")
    exe = _tracy(tracy_path, auto_install)
    env = build_env()
    with tempfile.TemporaryDirectory(prefix="ov_sanger_") as tmp:
        target = out or os.path.join(tmp, f"bc.{fmt}")
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        _run([exe, "basecall", "-f", fmt, "-y", otype, "-t", str(trim),
              "-p", str(pratio), "-o", target, path], env)
        text = open(target).read()
    if fmt == "fasta":
        return "".join(l.strip() for l in text.splitlines() if not l.startswith(">"))
    if fmt == "fastq":
        lines = text.splitlines()
        return lines[1] if len(lines) > 1 else ""
    return text


@register_function(
    aliases=["align_to_reference", "sanger_align", "比对参考", "trace_align"],
    category="alignment",
    description=(
        "Align a Sanger trace to a reference (FASTA, or a wild-type .ab1) with "
        "tracy and return the alignment: reference/trace rows, identity, and "
        "the per-position differences. This is how a clone is checked against "
        "the sequence it was supposed to be."
    ),
    examples=[
        "aln = ov.alignment.align_to_reference('clone_A.ab1', 'expected.fa')",
        "aln['identity'], aln['differences'][:5]",
    ],
    related=["alignment.verify_construct", "alignment.decompose_trace"],
)
def align_to_reference(trace: str, reference: str, *, trim_left: int = 50,
                       trim_right: int = 50, pratio: float = 0.33,
                       edge_margin: int = 50, out_prefix: Optional[str] = None,
                       tracy_path: str = "",
                       auto_install: bool = False) -> Dict[str, Any]:
    r"""Align ``trace`` to ``reference``.

    Returns
    -------
    dict
        ``ref_align`` / ``alt_align`` (the two gapped rows), ``identity`` (0-1
        over aligned columns), ``aligned`` (column count), ``differences``
        (``[{pos, ref, alt}]``, 0-based over the alignment), ``forward``
        (whether the trace matched the forward strand), and ``files`` (tracy's
        outputs when ``out_prefix`` was given).
    """
    for label, p in (("trace", trace), ("reference", reference)):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{label} not found: {p}")
    exe = _tracy(tracy_path, auto_install)
    env = build_env()

    tmp = tempfile.TemporaryDirectory(prefix="ov_sanger_")
    try:
        prefix = out_prefix or os.path.join(tmp.name, "aln")
        if out_prefix:
            os.makedirs(os.path.dirname(os.path.abspath(prefix)) or ".", exist_ok=True)
        _run([exe, "align", "-r", reference, "-o", prefix,
              "-q", str(trim_left), "-u", str(trim_right), "-p", str(pratio), trace], env)
        with open(f"{prefix}.json") as fh:
            doc = json.load(fh)
        files = sorted(glob.glob(f"{prefix}*")) if out_prefix else []
    finally:
        # The JSON is read inside the `try` above, so the directory is only
        # needed until here in both cases. (An earlier cut had two identical
        # branches and a dead `payload = None` guarding a lifetime concern that
        # does not exist.)
        tmp.cleanup()

    def _identity(diffs, span):
        return (span - len(diffs)) / span if span > 0 else 0.0

    ref_row = doc.get("refalign", "") or ""
    alt_row = doc.get("altalign", "") or ""
    diffs = [{"pos": i, "ref": r, "alt": a}
             for i, (r, a) in enumerate(zip(ref_row, alt_row)) if r != a]
    aligned = min(len(ref_row), len(alt_row))
    # A Sanger read decays at both ends: the last ~50 bases are routinely all
    # gaps against a perfectly correct clone. Judging a construct on whole-read
    # identity therefore fails good clones, so report BOTH — `identity` over
    # everything, and `identity_core` over the trustworthy middle, which is what
    # `verify_construct` actually decides on.
    core_lo, core_hi = edge_margin, aligned - edge_margin
    core_span = max(0, core_hi - core_lo)
    core_diffs = [d for d in diffs if core_lo <= d["pos"] < core_hi]
    # A short read (or a large `edge_margin`) can leave NO core at all. That is
    # "cannot be judged", not "0% identity" — reporting 0.0 made
    # `verify_construct` fail a perfect clone while handing back an empty
    # `core_differences`, i.e. a rejection with no evidence. `None` forces the
    # caller (and the agent reading it) to say so instead.
    return {
        "ref_align": ref_row,
        "alt_align": alt_row,
        "aligned": aligned,
        "identity": _identity(diffs, aligned),
        "identity_core": _identity(core_diffs, core_span) if core_span else None,
        "core_span": core_span,
        "edge_margin": edge_margin,
        "differences": diffs,
        "core_differences": core_diffs,
        "forward": bool(doc.get("forward", True)),
        "ref_chrom": doc.get("refchr", ""),
        "ref_pos": doc.get("refpos", None),
        "files": files,
    }


@register_function(
    aliases=["decompose_trace", "峰图解卷积", "trace_decompose", "heterozygous",
             "crispr_edit_readout"],
    category="alignment",
    description=(
        "Decompose a MIXED Sanger trace against a reference — the way a "
        "heterozygous variant or a CRISPR-edited allele is read off a single "
        "overlapping chromatogram. Wraps `tracy decompose`, which is the only "
        "well-maintained open implementation of this."
    ),
    examples=[
        "res = ov.alignment.decompose_trace('edited.ab1', 'wildtype.fa')",
        "res['variants']",
    ],
    related=["alignment.align_to_reference", "synbio.design_grnas"],
)
def decompose_trace(trace: str, reference: str, *, call_variants: bool = True,
                    max_indel: int = 1000, pratio: float = 0.33,
                    out_prefix: Optional[str] = None, tracy_path: str = "",
                    auto_install: bool = False) -> Dict[str, Any]:
    r"""Deconvolute a mixed trace into its alleles.

    Returns a dict with ``variants`` (whatever tracy called), the raw
    ``report`` document, and ``files`` when ``out_prefix`` was given.
    """
    for label, p in (("trace", trace), ("reference", reference)):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{label} not found: {p}")
    exe = _tracy(tracy_path, auto_install)
    env = build_env()

    with tempfile.TemporaryDirectory(prefix="ov_sanger_") as tmp:
        prefix = out_prefix or os.path.join(tmp, "dec")
        if out_prefix:
            os.makedirs(os.path.dirname(os.path.abspath(prefix)) or ".", exist_ok=True)
        cmd = [exe, "decompose", "-r", reference, "-o", prefix,
               "-i", str(max_indel), "-p", str(pratio)]
        if call_variants:
            cmd.append("-v")
        cmd.append(trace)
        _run(cmd, env)
        doc: Dict[str, Any] = {}
        jpath = f"{prefix}.json"
        if os.path.exists(jpath):
            with open(jpath) as fh:
                doc = json.load(fh)
        files = sorted(glob.glob(f"{prefix}*")) if out_prefix else []

    variants = doc.get("variants", doc.get("variant", []))
    return {"variants": variants, "report": doc, "files": files}


@register_function(
    aliases=["verify_construct", "construct_verification", "克隆验证",
             "sanger_verify", "验证构建体"],
    category="alignment",
    description=(
        "Verify a cloned construct against the sequence it was designed to be: "
        "align one or more Sanger traces to the expected reference (FASTA or "
        "GenBank, e.g. the output of ov.synbio.write_genbank) and report "
        "pass/fail with the exact mismatches. Closes the design->build->verify "
        "loop."
    ),
    examples=[
        "rep = ov.alignment.verify_construct(['fwd.ab1','rev.ab1'], 'pUC19_insert.gb')",
        "rep['ok'], rep['traces'][0]['identity'], rep['traces'][0]['differences']",
    ],
    related=["synbio.write_genbank", "synbio.design_primers", "alignment.align_to_reference"],
)
def verify_construct(traces, reference: str, *, min_identity: float = 0.995,
                     trim_left: int = 50, trim_right: int = 50,
                     edge_margin: int = 50, tracy_path: str = "",
                     auto_install: bool = False) -> Dict[str, Any]:
    r"""Check sequencing reads against the construct that was designed.

    Arguments
    ---------
    traces
        One ``.ab1`` path or a list of them (e.g. forward and reverse reads).
    reference
        Expected sequence — FASTA, or **GenBank** (``.gb``/``.gbk``, which is
        what :func:`omicverse.synbio.write_genbank` produces; it is converted to
        FASTA internally via Biopython).
    min_identity
        Identity required to pass, measured over the **core** region (the
        alignment minus ``edge_margin`` on each side).
    edge_margin
        Bases excluded from the verdict at each end. Sanger reads decay at both
        ends, so a perfectly correct clone routinely shows gap-only differences
        there; counting them would fail good clones.

    Returns
    -------
    dict
        ``ok`` (all traces passed), ``min_identity``, and ``traces`` — one entry
        per read with ``trace``, ``identity`` (whole read), ``identity_core``,
        ``aligned``, ``core_span``, ``differences``, ``core_differences``,
        ``forward`` and ``passed``.

    Note
    ----
    For a construct you intend to USE, require ``core_differences == []``
    rather than trusting ``ok`` alone: a single substitution in a ~1 kb read is
    still ~0.999 identity and will clear the default threshold, yet a
    single base is exactly what breaks a coding sequence.

        rep = ov.alignment.verify_construct("clone.ab1", "expected.gb")
        clean = rep["ok"] and not any(t["core_differences"] for t in rep["traces"])
    """
    if isinstance(traces, (str, os.PathLike)):
        traces = [str(traces)]
    traces = [str(t) for t in traces]
    if not traces:
        raise ValueError("no traces given")
    if not os.path.exists(reference):
        raise FileNotFoundError(f"reference not found: {reference}")

    tmp = tempfile.TemporaryDirectory(prefix="ov_sanger_ref_")
    try:
        ref_fa = reference
        if os.path.splitext(reference)[1].lower() in (".gb", ".gbk", ".genbank"):
            try:
                from Bio import SeqIO
            except ImportError as exc:                            # pragma: no cover
                raise ImportError(
                    "a GenBank reference needs Biopython (`pip install biopython`); "
                    "pass a FASTA instead."
                ) from exc
            ref_fa = os.path.join(tmp.name, "reference.fa")
            SeqIO.convert(reference, "genbank", ref_fa, "fasta")

        results = []
        for t in traces:
            aln = align_to_reference(t, ref_fa, trim_left=trim_left,
                                     trim_right=trim_right, edge_margin=edge_margin,
                                     tracy_path=tracy_path, auto_install=auto_install)
            # Judge on the CORE region. A correct clone routinely shows dozens
            # of gap-only differences in the last ~50 bases (Sanger end decay);
            # judging on whole-read identity would fail it. Whole-read numbers
            # are still returned so the caller can see everything.
            results.append({
                "trace": t,
                "identity": aln["identity"],
                "identity_core": aln["identity_core"],
                "aligned": aln["aligned"],
                "core_span": aln["core_span"],
                "differences": aln["differences"],
                "core_differences": aln["core_differences"],
                "forward": aln["forward"],
                # `identity_core is None` means the alignment was not longer
                # than 2*edge_margin, so there is no trustworthy middle to judge
                # on. That is UNJUDGEABLE, not a failure — but it must not
                # silently pass either, so it fails with `unjudged` set and the
                # caller is told to lower `edge_margin` or get a longer read.
                "unjudged": aln["identity_core"] is None,
                "passed": aln["identity_core"] is not None
                          and aln["identity_core"] >= min_identity,
            })
    finally:
        tmp.cleanup()

    return {
        "ok": all(r["passed"] for r in results),
        "min_identity": min_identity,
        "reference": reference,
        "traces": results,
    }
