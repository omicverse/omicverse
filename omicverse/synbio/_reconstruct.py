r"""Layer A entry point — build a GEM for an organism that has none.

:func:`load_gem` can only *read* a model that already exists. Hand it an
in-house industrial isolate and the whole layer-A chain stops at step one,
because BiGG has no model for it. These functions close that gap:

* :func:`reconstruct_gem` — draft a model for a new proteome.
  ``method='homology'`` (default, dependency-free beyond an aligner) transfers a
  template GEM through reciprocal best hits: every template reaction whose gene
  rule can no longer be satisfied is carved away. ``method='carveme'`` and
  ``method='modelseed'`` drive the real external reconstructors when installed.
* :func:`gapfill_model` — a carved draft usually cannot grow. Pull the smallest
  set of reactions from a universal database that restores flux through the
  objective (MILP via COBRApy, or an LP relaxation that needs no MILP solver).
* :func:`universal_reactions` — assemble that universal database from one or
  more template models.
* :func:`validate_gem` — the checks worth running before trusting a draft: mass
  and charge balance, orphan/dead-end metabolites, blocked reactions, growth.
* :func:`plot_reconstruction` — where the reactions went, and what gap-filling
  put back.

The GPR evaluator here is deliberately our own (:func:`_eval_gpr`) rather than
COBRApy's: ``GPR.eval`` has changed argument semantics across releases, and
"which reactions survive" is the single decision the whole reconstruction rests
on — it needs to be pinned and testable, not version-coupled.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .._registry import register_function

if TYPE_CHECKING:  # pragma: no cover - typing only
    import cobra
    import pandas as pd


_MISSING = (
    "ov.synbio.{fn} 需要额外依赖 COBRApy。请 pip install 'omicverse[synbio]' "
    "(或直接 pip install cobra optlang)。"
)


def _cobra(fn: str):
    try:
        import cobra
        return cobra
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise ImportError(_MISSING.format(fn=fn)) from exc


# Reactions that must never be carved away by gene evidence. Exchange /
# demand / sink reactions and the biomass equation carry no GPR at all, so a
# naive "no genes -> drop it" rule would delete the objective and every way in
# or out of the model.
_KEEP_PREFIXES = ("EX_", "DM_", "SK_", "BIOMASS", "ATPM")


def _is_structural(rxn) -> bool:
    """True for boundary / biomass / maintenance reactions — never gene-gated."""
    rid = rxn.id.upper()
    if any(rid.startswith(p) for p in _KEEP_PREFIXES):
        return True
    if "BIOMASS" in rid:
        return True
    # cobra marks exchanges/demands/sinks by having a single metabolite
    try:
        if rxn.boundary:
            return True
    except Exception:  # pragma: no cover - very old cobra
        pass
    return False


# ---------------------------------------------------------------------------
# GPR evaluation
# ---------------------------------------------------------------------------

_GPR_TOKEN = re.compile(r"\(|\)|\band\b|\bor\b|[^\s()]+", re.IGNORECASE)


def _eval_gpr(rule: str, present: Set[str]) -> bool:
    """Evaluate a gene-protein-reaction rule against a set of present genes.

    ``rule`` uses COBRApy's grammar: gene ids joined by ``and`` / ``or`` with
    parentheses. An empty rule means "no gene evidence required" and evaluates
    True — the caller decides whether such a reaction is structural.

    Implemented as a shunting-yard pass rather than ``eval()`` so that a gene id
    like ``and1`` or ``lambda`` cannot change the parse, and so behaviour is
    independent of the COBRApy version.
    """
    rule = (rule or "").strip()
    if not rule:
        return True

    tokens = [t for t in _GPR_TOKEN.findall(rule) if t.strip()]
    # precedence: and binds tighter than or
    prec = {"or": 1, "and": 2}
    out: List[object] = []
    ops: List[str] = []

    def _apply(op: str) -> None:
        if len(out) < 2:
            # malformed rule — treat the dangling operand as the result
            return
        b = out.pop()
        a = out.pop()
        out.append((a and b) if op == "and" else (a or b))

    for tok in tokens:
        low = tok.lower()
        if low in ("and", "or"):
            while ops and ops[-1] in prec and prec[ops[-1]] >= prec[low]:
                _apply(ops.pop())
            ops.append(low)
        elif tok == "(":
            ops.append(tok)
        elif tok == ")":
            while ops and ops[-1] != "(":
                _apply(ops.pop())
            if ops:
                ops.pop()
        else:
            out.append(tok in present)
    while ops:
        op = ops.pop()
        if op in prec:
            _apply(op)
    return bool(out[-1]) if out else True


# ---------------------------------------------------------------------------
# result containers
# ---------------------------------------------------------------------------

@dataclass
class ReconstructionReport:
    """What :func:`reconstruct_gem` produced and what it threw away."""

    model: "cobra.Model"
    method: str
    template: str
    n_reactions: int
    n_metabolites: int
    n_genes: int
    template_reactions: int
    dropped_reactions: List[str] = field(default_factory=list)
    mapped_genes: Dict[str, str] = field(default_factory=dict)   # query -> template
    unmapped_template_genes: List[str] = field(default_factory=list)
    growth: float = 0.0

    @property
    def grows(self) -> bool:
        return self.growth > 1e-6

    @property
    def coverage(self) -> float:
        """Fraction of the template's reactions retained."""
        if not self.template_reactions:
            return 0.0
        return self.n_reactions / self.template_reactions

    def __repr__(self) -> str:  # pragma: no cover
        return (f"ReconstructionReport({self.method}, template={self.template!r}, "
                f"{self.n_reactions}/{self.template_reactions} rxns "
                f"({self.coverage:.0%}), {self.n_genes} genes, "
                f"growth={self.growth:.4f})")


@dataclass
class GapfillReport:
    """Reactions gap-filling added, and the growth it bought."""

    model: "cobra.Model"
    added: List[str] = field(default_factory=list)
    growth_before: float = 0.0
    growth_after: float = 0.0
    method: str = "cobra"
    universe_size: int = 0

    @property
    def solved(self) -> bool:
        return self.growth_after > 1e-6

    def __repr__(self) -> str:  # pragma: no cover
        return (f"GapfillReport({self.method}, +{len(self.added)} rxns, "
                f"growth {self.growth_before:.4f} -> {self.growth_after:.4f})")


# ---------------------------------------------------------------------------
# reconstruct_gem
# ---------------------------------------------------------------------------

@register_function(
    aliases=["reconstruct_gem", "GEM重建", "代谢模型重建", "draft_gem",
             "carveme", "build_gem", "模型重建", "从头构建GEM"],
    category="synthetic_biology",
    description="为 BiGG 上没有模型的菌株重建 GEM。method='homology'(默认,用互惠最佳命中把模板 GEM 移植过来,基因规则不满足的反应被剪掉)/ 'carveme'(真实 CarveMe CLI)/ 'modelseed'(ModelSEEDpy)。Reconstruct a draft genome-scale model for a new proteome.",
    examples=[
        "rep = ov.synbio.reconstruct_gem('strain.faa', template='e_coli_core', template_proteome='ecoli_k12.faa')",
        "rep.model, rep.coverage, rep.grows",
        "rep = ov.synbio.reconstruct_gem('strain.faa', method='carveme')",
    ],
    related=["synbio.gapfill_model", "synbio.validate_gem", "synbio.load_gem",
             "alignment.reciprocal_best_hits"],
    requires={},
    produces={},
)
def reconstruct_gem(
    proteome: str,
    template: "str | cobra.Model" = "e_coli_core",
    template_proteome: Optional[str] = None,
    method: str = "homology",
    *,
    min_identity: float = 40.0,
    min_coverage: float = 0.5,
    evalue: float = 1e-10,
    threads: int = 8,
    keep_orphan_reactions: bool = False,
    gene_map: Optional[Mapping[str, str]] = None,
    carveme_args: Optional[Sequence[str]] = None,
    output: Optional[str] = None,
) -> ReconstructionReport:
    """Draft a genome-scale model for an organism that has none.

    Parameters
    ----------
    proteome
        FASTA of the target organism's predicted proteins. For
        ``method='carveme'`` this is the input CarveMe consumes directly.
    template
        Template GEM — a BiGG id or an in-memory :class:`cobra.Model`. Only used
        by ``method='homology'``.
    template_proteome
        FASTA of the template organism's proteins, with ids matching the
        template model's gene ids. Required for ``method='homology'`` unless
        ``gene_map`` is supplied.
    method
        ``'homology'`` — transfer the template by reciprocal best hits (default;
        needs DIAMOND or BLAST+ on ``PATH``, via
        :func:`omicverse.alignment.reciprocal_best_hits`).
        ``'carveme'`` — run the real ``carve`` CLI.
        ``'modelseed'`` — run ModelSEEDpy's reconstruction.
    min_identity, min_coverage, evalue, threads
        Homology thresholds. ``min_coverage`` is alignment length over query
        length.
    keep_orphan_reactions
        By default a template reaction whose GPR is unsatisfiable is dropped.
        Set True to keep everything and only annotate the report — useful when
        the proteome is fragmentary and you would rather gap-fill down than up.
    gene_map
        Skip the alignment and supply ``{query_gene: template_gene}`` directly.
        This is what makes the homology path testable offline.
    carveme_args
        Extra flags for the ``carve`` command line.
    output
        Write the drafted model here (``.xml`` SBML or ``.json``).

    Returns
    -------
    ReconstructionReport
    """
    if method not in ("homology", "carveme", "gapseq", "modelseed"):
        raise ValueError(
            f"method must be one of ['homology', 'carveme', 'gapseq', "
            f"'modelseed'], got {method!r}")

    if method == "carveme":
        rep = _reconstruct_carveme(proteome, carveme_args, output)
    elif method == "gapseq":
        rep = _reconstruct_gapseq(proteome, carveme_args, output)
    elif method == "modelseed":
        rep = _reconstruct_modelseed(proteome, output)
    else:
        rep = _reconstruct_homology(
            proteome, template, template_proteome,
            min_identity=min_identity, min_coverage=min_coverage,
            evalue=evalue, threads=threads,
            keep_orphan_reactions=keep_orphan_reactions, gene_map=gene_map,
        )

    if output and method == "homology":
        _write_model(rep.model, output)
    return rep


def _write_model(model, path: str) -> None:
    cobra = _cobra("reconstruct_gem")
    ext = os.path.splitext(path)[1].lower()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if ext == ".json":
        cobra.io.save_json_model(model, path)
    else:
        cobra.io.write_sbml_model(model, path)


def _load_template(template) -> "cobra.Model":
    if isinstance(template, str):
        from ._gem import load_gem
        return load_gem(template)
    return template


def _rbh_gene_map(
    proteome: str,
    template_proteome: str,
    *,
    min_identity: float,
    min_coverage: float,
    evalue: float,
    threads: int,
) -> Dict[str, str]:
    """``{query_gene: template_gene}`` from reciprocal best hits."""
    from ..alignment import reciprocal_best_hits

    rbh = reciprocal_best_hits(
        proteome, template_proteome, evalue=evalue, threads=threads,
    )
    if rbh is None or len(rbh) == 0:
        return {}

    # outfmt-6 names these qseqid/sseqid (alignment.BLAST_TAB_COLUMNS) and does
    # NOT carry qlen — reading a missing 'qlen' made min_coverage a silently
    # inert parameter. Query lengths come from the FASTA instead.
    cols = set(rbh.columns)
    qcol = "qseqid" if "qseqid" in cols else rbh.columns[0]
    scol = "sseqid" if "sseqid" in cols else rbh.columns[1]
    qlens = _fasta_lengths(proteome)

    out: Dict[str, str] = {}
    for _, row in rbh.iterrows():
        pident = float(row.get("pident", 100.0) or 100.0)
        if pident < min_identity:
            continue
        qid = str(row[qcol])
        length = float(row.get("length", 0) or 0)
        qlen = qlens.get(qid, 0)
        if qlen and length / qlen < min_coverage:
            continue
        out[qid] = str(row[scol])
    return out


def _fasta_lengths(path: str) -> Dict[str, int]:
    """``{sequence_id: length}`` for a FASTA, ids truncated at whitespace the
    way the aligners report them."""
    lengths: Dict[str, int] = {}
    name = None
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    lengths[name] = count
                name = line[1:].strip().split()[0] if line[1:].strip() else ""
                count = 0
            else:
                count += len(line.strip())
    if name is not None:
        lengths[name] = count
    return lengths


def _reconstruct_homology(
    proteome: str,
    template,
    template_proteome: Optional[str],
    *,
    min_identity: float,
    min_coverage: float,
    evalue: float,
    threads: int,
    keep_orphan_reactions: bool,
    gene_map: Optional[Mapping[str, str]],
) -> ReconstructionReport:
    cobra = _cobra("reconstruct_gem")
    tmpl = _load_template(template)
    tmpl_name = getattr(tmpl, "id", None) or str(template)
    n_template_rxns = len(tmpl.reactions)

    if gene_map is None:
        if not template_proteome:
            raise ValueError(
                "method='homology' 需要 template_proteome(其序列 id 必须与模板模型的 "
                "基因 id 对应),或直接传 gene_map={query_gene: template_gene}。")
        for label, path in (("proteome", proteome),
                            ("template_proteome", template_proteome)):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{label} FASTA not found: {path}")
        gene_map = _rbh_gene_map(
            proteome, template_proteome, min_identity=min_identity,
            min_coverage=min_coverage, evalue=evalue, threads=threads,
        )

    gene_map = dict(gene_map)
    present_template_genes = set(gene_map.values())
    template_genes = {g.id for g in tmpl.genes}
    unmapped = sorted(template_genes - present_template_genes)

    draft = tmpl.copy()
    draft.id = f"{tmpl_name}_draft"

    to_drop = []
    for rxn in list(draft.reactions):
        if _is_structural(rxn):
            continue
        rule = rxn.gene_reaction_rule or ""
        if not rule:
            # no gene evidence in the template either — keep, it is spontaneous
            # or diffusion, not something homology can speak to
            continue
        if not _eval_gpr(rule, present_template_genes):
            to_drop.append(rxn.id)

    if not keep_orphan_reactions and to_drop:
        draft.remove_reactions([draft.reactions.get_by_id(r) for r in to_drop],
                               remove_orphans=True)

    try:
        growth = float(draft.slim_optimize()) or 0.0
    except Exception:
        growth = 0.0
    if growth != growth:  # NaN — infeasible
        growth = 0.0

    return ReconstructionReport(
        model=draft, method="homology", template=str(tmpl_name),
        n_reactions=len(draft.reactions), n_metabolites=len(draft.metabolites),
        n_genes=len(draft.genes), template_reactions=n_template_rxns,
        dropped_reactions=to_drop, mapped_genes=gene_map,
        unmapped_template_genes=unmapped, growth=growth,
    )


def _reconstruct_carveme(proteome, carveme_args, output) -> ReconstructionReport:
    """Drive the real CarveMe CLI (Machado 2018)."""
    import shutil
    import subprocess
    import tempfile

    exe = shutil.which("carve")
    if exe is None:
        raise ImportError(
            "method='carveme' 需要 CarveMe 的 carve 命令。请 "
            "pip install carveme(它还需要 diamond 与一个 MILP 求解器,"
            "conda install -c bioconda diamond)。或改用 method='homology',"
            "只需要 DIAMOND/BLAST+。")
    out = output or os.path.join(tempfile.mkdtemp(prefix="ov_carveme_"), "model.xml")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    cmd = [exe, proteome, "-o", out, *(str(a) for a in (carveme_args or []))]
    subprocess.run(cmd, check=True)

    from ._gem import load_gem
    model = load_gem(out)
    try:
        growth = float(model.slim_optimize()) or 0.0
    except Exception:
        growth = 0.0
    return ReconstructionReport(
        model=model, method="carveme", template="carveme_universe",
        n_reactions=len(model.reactions), n_metabolites=len(model.metabolites),
        n_genes=len(model.genes), template_reactions=len(model.reactions),
        growth=growth if growth == growth else 0.0,
    )


def _reconstruct_gapseq(genome, extra_args, output) -> ReconstructionReport:
    """Drive the real gapseq pipeline (Zimmermann 2021).

    gapseq differs from CarveMe in what it eats: CarveMe takes a **proteome**
    and carves a universal model down, gapseq takes a **genome** (nucleotide
    contigs), predicts pathways from it with its own reaction database, and
    builds up. Passing a ``.faa`` here is the usual mistake, so it is caught
    rather than left to fail inside the pipeline.
    """
    import shutil
    import subprocess
    import tempfile

    exe = shutil.which("gapseq")
    if exe is None:
        raise ImportError(
            "method='gapseq' 需要 gapseq 在 PATH 上。它是一套 R/bash 流水线,"
            "不在 PyPI 上:conda install -c bioconda gapseq,或见 "
            "github.com/jotech/gapseq。它还需要 blast/hmmer/exonerate 等外部工具。"
            "只想要一个可跑的草稿模型的话,method='homology' 只需要 DIAMOND/BLAST+。")

    ext = os.path.splitext(str(genome))[1].lower()
    if ext in (".faa", ".fa.faa", ".pep"):
        raise ValueError(
            f"gapseq 需要**基因组**核苷酸序列(.fna/.fasta/.gbk),"
            f"给到的是蛋白序列 {genome!r}。CarveMe 吃蛋白组、gapseq 吃基因组,"
            f"两者不能互换 —— 用 method='carveme' 传蛋白组,或给 gapseq 一份基因组。")
    if not os.path.exists(genome):
        raise FileNotFoundError(f"genome FASTA not found: {genome}")

    workdir = tempfile.mkdtemp(prefix="ov_gapseq_")
    cmd = [exe, "doall", os.path.abspath(genome),
           *(str(a) for a in (extra_args or []))]
    subprocess.run(cmd, check=True, cwd=workdir)

    # gapseq doall writes <basename>.xml next to its working directory
    stem = os.path.splitext(os.path.basename(genome))[0]
    produced = None
    for candidate in (os.path.join(workdir, f"{stem}.xml"),
                      os.path.join(workdir, f"{stem}-draft.xml")):
        if os.path.exists(candidate):
            produced = candidate
            break
    if produced is None:
        found = sorted(f for f in os.listdir(workdir) if f.endswith(".xml"))
        if not found:
            raise RuntimeError(
                f"gapseq 跑完了但在 {workdir} 里没找到 SBML 输出。"
                f"目录内容:{sorted(os.listdir(workdir))[:10]}")
        produced = os.path.join(workdir, found[0])

    from ._gem import load_gem
    model = load_gem(produced)
    if output:
        _write_model(model, output)
    try:
        growth = float(model.slim_optimize()) or 0.0
    except Exception:
        growth = 0.0
    return ReconstructionReport(
        model=model, method="gapseq", template="gapseq_database",
        n_reactions=len(model.reactions), n_metabolites=len(model.metabolites),
        n_genes=len(model.genes), template_reactions=len(model.reactions),
        growth=growth if growth == growth else 0.0,
    )


def _reconstruct_modelseed(proteome, output) -> ReconstructionReport:
    """Drive ModelSEEDpy's reconstruction (Seaver 2021)."""
    try:
        import modelseedpy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "method='modelseed' 需要 ModelSEEDpy。请 "
            "pip install modelseedpy(它需要一个 ModelSEED 数据库快照,"
            "见 github.com/ModelSEED/ModelSEEDpy)。或改用 method='homology'。"
        ) from exc

    from modelseedpy import MSBuilder, MSGenome  # type: ignore

    genome = MSGenome.from_fasta(proteome, split=" ")
    model = MSBuilder.build_metabolic_model(
        model_id="modelseed_draft", genome=genome, allow_all_non_grp_reactions=True,
    )
    if output:
        _write_model(model, output)
    try:
        growth = float(model.slim_optimize()) or 0.0
    except Exception:
        growth = 0.0
    return ReconstructionReport(
        model=model, method="modelseed", template="ModelSEED",
        n_reactions=len(model.reactions), n_metabolites=len(model.metabolites),
        n_genes=len(model.genes), template_reactions=len(model.reactions),
        growth=growth if growth == growth else 0.0,
    )


# ---------------------------------------------------------------------------
# universal reaction database
# ---------------------------------------------------------------------------

@register_function(
    aliases=["universal_reactions", "通用反应库", "universal_model",
             "反应数据库", "universe"],
    category="synthetic_biology",
    description="从一个或多个模板 GEM 组装通用反应库(去掉基因规则与边界反应),作为 gap-filling 的候选池。Assemble a universal reaction database from template models.",
    examples=[
        "uni = ov.synbio.universal_reactions(['e_coli_core', 'iJO1366'])",
        "rep = ov.synbio.gapfill_model(draft, universe=uni)",
    ],
    related=["synbio.gapfill_model", "synbio.reconstruct_gem"],
    requires={},
    produces={},
)
def universal_reactions(
    templates: "Sequence[str | cobra.Model]",
    *,
    include_exchanges: bool = False,
    model_id: str = "universal",
) -> "cobra.Model":
    """Pool the reactions of several models into a gap-filling candidate set.

    Gene rules are stripped: a universal reaction is a *chemical* possibility,
    not something the target organism is claimed to encode. Boundary reactions
    are excluded by default — gap-filling should not be allowed to solve a dead
    end by inventing a transporter to the medium unless you ask for it.
    """
    cobra = _cobra("universal_reactions")
    uni = cobra.Model(model_id)
    seen: Set[str] = set()
    for t in templates:
        m = _load_template(t)
        add = []
        for rxn in m.reactions:
            if rxn.id in seen:
                continue
            if not include_exchanges and _is_structural(rxn):
                continue
            r = rxn.copy()
            r.gene_reaction_rule = ""
            add.append(r)
            seen.add(rxn.id)
        if add:
            uni.add_reactions(add)
    return uni


# ---------------------------------------------------------------------------
# gapfill_model
# ---------------------------------------------------------------------------

@register_function(
    aliases=["gapfill_model", "gap_filling", "补缺", "填补缺口", "gapfill",
             "模型补全", "gapseq"],
    category="synthetic_biology",
    description="Gap-filling:从通用反应库中挑出最小的一组反应,使草稿模型恢复目标(生长)通量。method='cobra'(COBRApy MILP)或 'lp'(L1 松弛,纯 LP,不需要 MILP 求解器,能处理需要同时补多个反应的缺口)。Fill gaps in a draft model so it can grow.",
    examples=[
        "rep = ov.synbio.gapfill_model(draft, universe=uni)",
        "rep.added, rep.growth_before, rep.growth_after",
        "rep = ov.synbio.gapfill_model(draft, universe=uni, method='lp')",
    ],
    related=["synbio.reconstruct_gem", "synbio.universal_reactions",
             "synbio.validate_gem"],
    requires={},
    produces={},
)
def gapfill_model(
    model: "cobra.Model",
    universe: "Optional[cobra.Model]" = None,
    *,
    method: str = "cobra",
    growth_threshold: float = 0.01,
    max_added: int = 20,
    demand_reactions: bool = False,
    exchange_reactions: bool = False,
    integer_threshold: float = 1e-6,
) -> GapfillReport:
    """Restore objective flux by adding reactions from a universal database.

    Parameters
    ----------
    model
        The draft model, modified only on a copy.
    universe
        Candidate reactions (see :func:`universal_reactions`). Defaults to a
        universe built from ``iJO1366`` — a superset of *E. coli* core
        metabolism, which is the sensible default for a bacterial draft.
    method
        ``'cobra'`` — COBRApy's MILP :class:`~cobra.flux_analysis.gapfilling.GapFiller`,
        which finds a genuinely minimal set. Needs a MILP-capable solver (GLPK
        is fine).
        ``'lp'`` — the L1 relaxation: add every candidate, require growth, then
        minimise total candidate flux, then prune. Pure LP, so it works on any
        solver, and it handles gaps that need several reactions at once.
    growth_threshold
        Objective value that counts as "grows".
    max_added
        Cap on how many reactions may be added.
    demand_reactions, exchange_reactions
        Passed to COBRApy's gap-filler — whether it may also invent demand or
        exchange reactions.
    integer_threshold
        MILP integrality tolerance.

    Returns
    -------
    GapfillReport
    """
    if method not in ("cobra", "lp"):
        raise ValueError(f"method must be one of ['cobra', 'lp'], got {method!r}")
    cobra = _cobra("gapfill_model")

    if universe is None:
        universe = universal_reactions(["iJO1366"])

    before = _safe_objective(model)
    work = model.copy()

    present = {r.id for r in work.reactions}
    candidates = [r for r in universe.reactions if r.id not in present]

    if method == "cobra":
        added = _gapfill_milp(
            work, universe, candidates, growth_threshold, max_added,
            demand_reactions, exchange_reactions, integer_threshold,
        )
    else:
        added = _gapfill_lp(work, candidates, growth_threshold, max_added)

    after = _safe_objective(work)
    return GapfillReport(model=work, added=added, growth_before=before,
                         growth_after=after, method=method,
                         universe_size=len(candidates))


def _safe_objective(model) -> float:
    try:
        v = float(model.slim_optimize())
    except Exception:
        return 0.0
    return 0.0 if v != v else v


def _gapfill_milp(work, universe, candidates, threshold, max_added,
                  demand, exchange, integer_threshold) -> List[str]:
    from cobra.flux_analysis import gapfilling as _gf

    uni = universe.copy()
    keep = {r.id for r in candidates}
    drop = [r for r in uni.reactions if r.id not in keep]
    if drop:
        uni.remove_reactions(drop, remove_orphans=False)

    try:
        filler = _gf.GapFiller(
            work, universal=uni, lower_bound=threshold,
            demand_reactions=demand, exchange_reactions=exchange,
            integer_threshold=integer_threshold,
        )
        solutions = filler.fill(iterations=1)
    except Exception as exc:
        raise RuntimeError(
            f"COBRApy gap-filling 失败({exc})。常见原因是求解器不支持 MILP 或"
            f" universe 太大。可以改用 method='lp'(L1 松弛,不需要 MILP)。") from exc

    if not solutions:
        return []
    chosen = solutions[0][:max_added]
    ids = [r.id for r in chosen]
    work.add_reactions([r.copy() for r in chosen])
    return ids


def _gapfill_lp(work, candidates, threshold, max_added, flux_tol=1e-7) -> List[str]:
    """L1 relaxation of the gap-filling MILP — an LP, so no MILP solver needed.

    Add every candidate at once, require the objective to clear ``threshold``,
    then minimise the total absolute flux carried by candidate reactions. The
    candidates left with non-zero flux are a set that works.

    This replaces the obvious "add whichever single reaction helps most, repeat":
    that never starts when a gap needs two reactions at once, because no single
    addition moves the objective off zero — which is the common case, since a
    carved-out pathway loses consecutive steps.

    The L1 optimum is not guaranteed minimal the way the MILP's is (a reaction
    carrying tiny flux still counts as selected), hence the pruning pass at the
    end: drop any selected reaction the model can grow without.
    """
    if not candidates:
        return []

    trial = work.copy()
    trial.add_reactions([r.copy() for r in candidates])
    cand_ids = [r.id for r in candidates]

    prob = trial.problem
    aux_vars = []
    cons = []
    for rid in cand_ids:
        rxn = trial.reactions.get_by_id(rid)
        aux = prob.Variable(f"_gf_abs_{rid}", lb=0)
        aux_vars.append(aux)
        # aux >= v  and  aux >= -v   =>  aux >= |v|
        cons.append(prob.Constraint(aux - rxn.flux_expression, lb=0,
                                    name=f"_gf_pos_{rid}"))
        cons.append(prob.Constraint(aux + rxn.flux_expression, lb=0,
                                    name=f"_gf_neg_{rid}"))
    obj_expr = trial.objective.expression
    cons.append(prob.Constraint(obj_expr, lb=threshold, name="_gf_growth"))
    trial.add_cons_vars(aux_vars + cons)

    trial.objective = prob.Objective(sum(aux_vars), direction="min")
    sol = trial.optimize()
    if sol.status != "optimal":
        return []

    selected = [rid for rid in cand_ids
                if abs(float(sol.fluxes.get(rid, 0.0))) > flux_tol]

    # prune: keep only what growth actually needs, smallest flux tried first
    selected.sort(key=lambda rid: abs(float(sol.fluxes.get(rid, 0.0))))
    keep = list(selected)
    for rid in selected:
        probe = work.copy()
        probe.add_reactions([_candidate_by_id(candidates, r).copy()
                             for r in keep if r != rid])
        if _safe_objective(probe) >= threshold:
            keep.remove(rid)

    keep = keep[:max_added]
    if keep:
        work.add_reactions([_candidate_by_id(candidates, r).copy() for r in keep])
    return keep


def _candidate_by_id(candidates, rid):
    for r in candidates:
        if r.id == rid:
            return r
    raise KeyError(rid)


# ---------------------------------------------------------------------------
# validate_gem
# ---------------------------------------------------------------------------

@register_function(
    aliases=["validate_gem", "模型检查", "模型质量", "gem_qc", "check_model",
             "模型验证", "memote"],
    category="synthetic_biology",
    description="草稿 GEM 质量检查:质量/电荷不平衡反应、孤立与死端代谢物、阻塞反应、能量漏洞(无底物产 ATP)、是否生长。Quality checks for a draft GEM before you trust it.",
    examples=[
        "qc = ov.synbio.validate_gem(draft)",
        "qc['unbalanced'], qc['blocked'], qc['energy_leak']",
    ],
    related=["synbio.reconstruct_gem", "synbio.gapfill_model"],
    requires={},
    produces={},
)
def validate_gem(
    model: "cobra.Model",
    *,
    check_blocked: bool = True,
    check_energy_leak: bool = True,
    check_cycles: bool = True,
) -> Dict[str, object]:
    """Run the checks that catch a broken draft.

    Returns a dict with ``growth``, ``unbalanced`` (reactions whose mass or
    charge does not close), ``orphan_metabolites`` (produced but never consumed,
    or the reverse), ``blocked`` (reactions that cannot carry flux in any
    steady state), and ``energy_leak`` — whether the model can make ATP with all
    uptake shut off, which is the classic sign that gap-filling introduced a
    thermodynamically impossible cycle.
    """
    cobra = _cobra("validate_gem")

    growth = _safe_objective(model)

    # check_mass_balance() returns {} when a metabolite has no formula, so a
    # draft with unannotated metabolites gets a clean bill of health. Count the
    # coverage alongside the failures, or "0 unbalanced" is unreadable.
    unbalanced = []
    for rxn in model.reactions:
        if _is_structural(rxn):
            continue
        try:
            imbalance = rxn.check_mass_balance()
        except Exception:
            continue
        if imbalance:
            unbalanced.append(rxn.id)
    without_formula = sorted(m.id for m in model.metabolites
                             if not getattr(m, "formula", None))
    without_charge = sorted(m.id for m in model.metabolites
                            if getattr(m, "charge", None) is None)

    produced: Set[str] = set()
    consumed: Set[str] = set()
    for rxn in model.reactions:
        for met, coeff in rxn.metabolites.items():
            if coeff > 0 or rxn.reversibility:
                produced.add(met.id)
            if coeff < 0 or rxn.reversibility:
                consumed.add(met.id)
    orphans = sorted((produced ^ consumed))

    blocked: List[str] = []
    if check_blocked:
        try:
            blocked = sorted(
                cobra.flux_analysis.find_blocked_reactions(model))
        except Exception:
            blocked = []

    leak = None
    if check_energy_leak:
        leak = _energy_leak(model)

    cycles: List[str] = []
    if check_cycles:
        cycles = _balanced_cycles(model)

    return {
        "growth": growth,
        "grows": growth > 1e-6,
        "n_reactions": len(model.reactions),
        "n_metabolites": len(model.metabolites),
        "n_genes": len(model.genes),
        "unbalanced": unbalanced,
        "n_without_formula": len(without_formula),
        "n_without_charge": len(without_charge),
        "metabolites_without_formula": without_formula,
        "orphan_metabolites": orphans,
        "blocked": blocked,
        "energy_leak": leak,
        "balanced_cycles": cycles,
    }


#: Energy currencies MEMOTE probes for free-lunch cycles, as the dissipation
#: reaction that would consume them if the model can make them from nothing.
ENERGY_LEAK_PROBES = ("ATPM", "ATPS4r")


def _balanced_cycles(model, tolerance: float = 1.0,
                     max_reactions: int = 400) -> List[str]:
    """Reactions that can carry flux with every exchange sealed.

    A sealed model has no source and no sink, so any non-zero steady-state flux
    closes a loop and violates the second law. ``FRD7``/``SUCDi`` in
    *e_coli_core* is the textbook example and both hit the 1000 flux bound.

    This is distinct from :func:`_energy_leak`: a balanced cycle need not produce
    ATP, so the energy test cannot see it. ``validate_gem`` claimed to catch
    gap-filling artefacts while running neither check in a working form.

    Skipped above ``max_reactions`` because it costs one LP per reaction; pass
    ``check_cycles=False`` to skip it explicitly.
    """
    reactions = [r for r in model.reactions if not r.boundary]
    if len(reactions) > max_reactions:
        return []
    looping: List[str] = []
    try:
        with model as m:
            for rxn in m.reactions:
                if rxn.boundary:
                    rxn.bounds = (0.0, 0.0)
                elif rxn.lower_bound > 0.0:
                    # A sealed model cannot satisfy a forced flux such as the
                    # ATP maintenance demand, so leaving it in makes every LP
                    # infeasible and the scan silently finds nothing — the same
                    # trap that made _energy_leak unable to fail.
                    rxn.lower_bound = 0.0
            for rxn in reactions:
                target = m.reactions.get_by_id(rxn.id)
                for direction in ("max", "min"):
                    m.objective = target.id
                    m.objective.direction = direction
                    value = m.slim_optimize()
                    if value == value and abs(float(value)) > tolerance:
                        looping.append(rxn.id)
                        break
    except Exception:                                     # pragma: no cover
        return []
    return sorted(set(looping))


def _energy_leak(model, probe: str = "ATPM") -> Optional[float]:
    """Max dissipation flux with every exchange sealed. ``> 0`` is a free lunch.

    A positive value means the network can generate a high-energy compound from
    nothing, which is almost always an artefact of gap-filling or of a reaction
    added in the wrong direction.

    Returns ``None`` when the test could not be run — deliberately distinct from
    ``0.0``, which asserts that there is no leak. The previous implementation
    conflated the two and so could never report a leak at all: it closed only the
    *lower* bounds of the exchanges (leaving secretion open) and never relaxed the
    maintenance reaction's own lower bound. ATPM in essentially every BiGG model
    carries a fixed non-zero NGAM demand, so with uptake closed the LP is
    **infeasible**, ``slim_optimize()`` returns nan, and ``0.0 if val != val``
    turned that nan into a clean bill of health for every model, leaky or not.
    """
    _cobra("validate_gem")
    reaction = None
    for candidate in (probe,) + ENERGY_LEAK_PROBES:
        try:
            reaction = model.reactions.get_by_id(candidate)
            break
        except Exception:
            continue
    if reaction is None:
        return None
    try:
        with model as m:
            for rxn in m.reactions:
                if rxn.boundary:
                    rxn.bounds = (0.0, 0.0)          # sealed, both directions
            target = m.reactions.get_by_id(reaction.id)
            target.bounds = (0.0, 1000.0)            # let the LP choose the flux
            m.objective = target.id
            solution = m.optimize()
            if solution.status != "optimal":
                return None
            val = float(solution.objective_value)
        return None if val != val else val
    except Exception:
        return None


# ---------------------------------------------------------------------------
# visualisation
# ---------------------------------------------------------------------------

@register_function(
    aliases=["plot_reconstruction", "重建图", "plot_gem_reconstruction",
             "重建报告图", "模型重建可视化"],
    category="synthetic_biology",
    description="画 GEM 重建结果:模板→草稿的反应保留/剪除瀑布图、基因映射覆盖度,以及 gap-filling 前后的生长对比。Visualise a GEM reconstruction (retained vs carved reactions, gene coverage, growth rescue).",
    examples=[
        "ov.synbio.plot_reconstruction(rep)",
        "ov.synbio.plot_reconstruction(rep, gapfill=gf_rep)",
    ],
    related=["synbio.reconstruct_gem", "synbio.gapfill_model"],
    requires={},
    produces={},
)
def plot_reconstruction(report: ReconstructionReport, gapfill: Optional[GapfillReport] = None,
                        axes=None):
    """Three panels: reaction waterfall, gene mapping coverage, growth rescue."""
    from ._plot import _mpl
    plt = _mpl()

    n_panels = 3 if gapfill is not None else 2
    if axes is None:
        fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.4))
    else:
        fig = axes[0].figure
    axes = list(axes)

    # --- panel 1: reactions kept vs carved -------------------------------
    ax = axes[0]
    kept = report.n_reactions
    carved = len(report.dropped_reactions)
    ax.bar(["template", "draft"], [report.template_reactions, kept],
           color=["#B0B0B0", "#377EB8"])
    if carved:
        ax.bar(["draft"], [carved], bottom=[kept], color="#E41A1C",
               label=f"carved ({carved})")
        ax.legend(fontsize=8)
    ax.set_ylabel("reactions")
    ax.set_title(f"{report.method}: {report.coverage:.0%} of template retained")

    # --- panel 2: gene mapping -------------------------------------------
    ax = axes[1]
    mapped = len(set(report.mapped_genes.values()))
    unmapped = len(report.unmapped_template_genes)
    if mapped or unmapped:
        ax.pie([mapped, unmapped], labels=[f"mapped\n{mapped}", f"unmapped\n{unmapped}"],
               colors=["#4DAF4A", "#E0E0E0"], autopct="%1.0f%%",
               startangle=90, textprops={"fontsize": 8})
        ax.set_title("template genes with a homolog")
    else:
        ax.text(0.5, 0.5, "no gene mapping\n(non-homology method)",
                ha="center", va="center", fontsize=9, transform=ax.transAxes)
        ax.set_axis_off()

    # --- panel 3: growth rescue ------------------------------------------
    if gapfill is not None:
        ax = axes[2]
        vals = [gapfill.growth_before, gapfill.growth_after]
        ax.bar(["draft", f"+{len(gapfill.added)} gapfilled"], vals,
               color=["#E41A1C" if vals[0] <= 1e-6 else "#377EB8", "#4DAF4A"])
        ax.set_ylabel("growth (1/h)")
        ax.set_title("gap-filling")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig, axes


__all__ = [
    "reconstruct_gem", "ReconstructionReport",
    "gapfill_model", "GapfillReport",
    "universal_reactions", "validate_gem", "plot_reconstruction",
]
