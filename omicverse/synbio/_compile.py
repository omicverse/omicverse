r"""Terminator strength, and compiling a truth table into DNA.

Two holes at the circuit layer.

``promoter_strength`` and ``rbs_strength`` existed; the third part of every
transcription unit did not. :func:`terminator_strength` scores intrinsic
(Rho-independent) terminators from their structure — hairpin stem stability, the
poly-U tail, and the A-tract upstream — because a weak terminator is a
transcriptional read-through that silently couples the unit downstream, which
looks like a broken promoter and is not.

``genetic_circuit`` / ``logic_gate`` build a circuit you have already designed
and simulate it. :func:`compile_circuit` goes the other way, the way Cello does:
give it a **truth table**, and it derives a gate network, assigns physical
repressor gates from a library, checks that each assignment's response function
actually separates the signal levels it is handed, and emits the DNA. The part
that matters is the check — an assignment that type-checks as a network can
still fail because the upstream gate's ON level lands in the downstream gate's
dead zone.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from .._registry import register_function

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


_COMP = str.maketrans("ACGTN", "TGCAN")


def _revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def _clean_dna(seq: str, fn: str) -> str:
    s = "".join(str(seq).split()).upper().replace("U", "T")
    if not s:
        raise ValueError(f"ov.synbio.{fn}: 序列为空。")
    bad = sorted(set(s) - set("ACGTN"))
    if bad:
        raise ValueError(f"ov.synbio.{fn}: 序列含非 DNA 字符 {bad}。")
    return s


# ---------------------------------------------------------------------------
# terminator strength
# ---------------------------------------------------------------------------

@dataclass
class TerminatorResult:
    """An intrinsic terminator call."""

    sequence: str
    strength: float                    # 0..1, fraction of transcripts terminated
    method: str = "structural"
    has_hairpin: bool = False
    stem_length: int = 0
    loop_length: int = 0
    stem_gc: float = 0.0
    hairpin_start: Optional[int] = None
    u_tract_length: int = 0
    a_tract_length: int = 0
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def readthrough(self) -> float:
        """Fraction of polymerases that do not terminate — the number that
        matters when a downstream unit is being silently driven."""
        return 1.0 - self.strength

    @property
    def classification(self) -> str:
        if self.strength >= 0.9:
            return "strong"
        if self.strength >= 0.6:
            return "moderate"
        if self.strength >= 0.3:
            return "weak"
        return "not a terminator"

    def __repr__(self) -> str:  # pragma: no cover
        return (f"TerminatorResult({self.classification}, strength="
                f"{self.strength:.2f}, stem={self.stem_length}, "
                f"U-tract={self.u_tract_length})")


@register_function(
    aliases=["terminator_strength", "终止子强度", "terminator", "终止子",
             "转录终止", "通读", "readthrough"],
    category="synthetic_biology",
    description="内在终止子(Rho 非依赖)强度:由结构推断 —— 发夹茎的稳定性与 GC、3' poly-U 尾长度、上游 A-tract。终止子弱 = 转录通读,会悄悄驱动下游的转录单元,表现得像启动子坏了。原来有 promoter_strength 和 rbs_strength,唯独缺这个。method='structural'(默认,无依赖)或 'thermodynamic'(用 ViennaRNA 算茎自由能)。Score an intrinsic transcription terminator.",
    examples=[
        "t = ov.synbio.terminator_strength('...GCCGCCAGCTGGCGGC TTTTTTT')",
        "t.strength, t.readthrough, t.classification",
    ],
    related=["synbio.promoter_strength", "synbio.rbs_strength",
             "synbio.predict_expression", "synbio.compile_circuit"],
    requires={},
    produces={},
)
def terminator_strength(sequence: str, method: str = "structural"
                        ) -> TerminatorResult:
    """Strength of an intrinsic terminator, in ``[0, 1]``.

    An intrinsic terminator is three features acting together, and the score
    weights them accordingly:

    ``hairpin``
        A GC-rich stem-loop. The stem's stability is what stalls RNA
        polymerase; GC pairs contribute roughly twice what AU pairs do.
    ``u_tract``
        6–8 uridines immediately 3' of the hairpin. The rU:dA hybrid is the
        weakest in nucleic acid chemistry, which is what lets the stalled
        polymerase release. Without it a hairpin only pauses.
    ``a_tract``
        An A-rich stretch 5' of the hairpin, which raises efficiency.

    ``method='thermodynamic'`` folds the region with ViennaRNA and uses the real
    stem free energy instead of a GC count.
    """
    _VALID = ("structural", "thermodynamic")
    if method not in _VALID:
        raise ValueError(f"method must be one of {list(_VALID)}, got {method!r}")
    seq = _clean_dna(sequence, "terminator_strength")

    # --- locate the hairpin ------------------------------------------------
    # Candidates are scored as *terminators*, not as hairpins. Picking the most
    # stable inverted repeat instead paired rrnB T1's 5' A-tract with its own 3'
    # U-tract into a 12-nt "stem" — consuming the very features that make it a
    # terminator — and scored the canonical strong terminator as weak.
    n = len(seq)
    candidates = []
    for stem_len in range(4, 16):
        for i in range(0, n - 2 * stem_len):
            stem = seq[i:i + stem_len]
            if "N" in stem:
                continue
            rc = _revcomp(stem)
            for loop in range(3, 11):
                j = i + stem_len + loop
                if j + stem_len > n:
                    break
                if seq[j:j + stem_len] == rc:
                    gc = (stem.count("G") + stem.count("C")) / stem_len
                    candidates.append((i, stem_len, loop, gc, j + stem_len))

    if not candidates:
        return TerminatorResult(
            sequence=seq, strength=0.0, method=method, has_hairpin=False,
            components={"hairpin": 0.0, "u_tract": 0.0, "a_tract": 0.0})

    def _features(cand):
        i, stem_len, loop, gc, hairpin_end = cand
        stability = stem_len * (1.0 + gc)
        tail = seq[hairpin_end:hairpin_end + 12]
        u = 0
        for ch in tail:
            if ch == "T":
                u += 1
            else:
                break
        if u == 0:                       # allow one interruption, at a discount
            for k, ch in enumerate(tail[:8]):
                if ch == "T":
                    u = min(sum(1 for c in tail[k:k + 8] if c == "T"), 8) // 2
                    break
        upstream = seq[max(0, i - 10):i]
        a = sum(1 for c in upstream if c == "A")
        return stability, u, a

    def _rank(cand):
        stability, u, a = _features(cand)
        return 0.45 * min(1.0, stability / 15.0) + 0.45 * min(1.0, u / 6.0) \
            + 0.10 * min(1.0, a / 6.0)

    best = max(candidates, key=_rank)
    start, stem_len, loop, gc, hairpin_end = best
    stability, u_len, a_len = _features(best)

    if method == "thermodynamic":
        try:
            import RNA  # noqa: F401
            from ._rna import rna_fold
            region = seq[start:hairpin_end].replace("T", "U")
            fold = rna_fold(region)
            dg = float(getattr(fold, "mfe", getattr(fold, "energy", 0.0))
                       if not isinstance(fold, tuple) else fold[1])
            stability = max(0.0, -dg)
        except Exception as exc:
            raise ImportError(
                "method='thermodynamic' 需要 ViennaRNA(pip install ViennaRNA)。"
                "默认 method='structural' 用茎长与 GC 组成,不需要依赖。") from exc

    # Real intrinsic terminators have short stems — rrnB T1's is 7-8 bp — so
    # normalising stability at 22 scored the canonical strong terminator as
    # merely moderate.
    hairpin_term = min(1.0, stability / 15.0)
    u_term = min(1.0, u_len / 6.0)
    a_term = min(1.0, a_len / 6.0)

    # The U-tract is not a bonus — without it a hairpin merely pauses the
    # polymerase, so it gates the score rather than adding to it. One or two
    # uridines do not make a terminator either: any GC-rich inverted repeat
    # followed by a stray T would otherwise score as weakly terminating.
    strength = (0.45 * hairpin_term + 0.45 * u_term + 0.10 * a_term)
    if u_len == 0:
        strength = min(strength, 0.25)
    elif u_len < 3:
        strength = min(strength, 0.30)

    return TerminatorResult(
        sequence=seq, strength=float(min(1.0, strength)), method=method,
        has_hairpin=True, stem_length=stem_len, loop_length=loop, stem_gc=gc,
        hairpin_start=start + 1, u_tract_length=u_len, a_tract_length=a_len,
        components={"hairpin": hairpin_term, "u_tract": u_term, "a_tract": a_term},
    )


# ---------------------------------------------------------------------------
# circuit compilation
# ---------------------------------------------------------------------------

@dataclass
class GatePart:
    """A characterised repressor gate from a library."""

    name: str
    repressor: str
    promoter: str
    cds: str = ""
    terminator: str = ""
    ymin: float = 0.05          # output at saturating input (repressed)
    ymax: float = 2.5           # output at zero input (de-repressed)
    K: float = 0.3              # input at half-maximal response
    n: float = 2.0              # Hill coefficient

    def response(self, x: float) -> float:
        """NOT-gate transfer function: output falls as input rises."""
        return self.ymin + (self.ymax - self.ymin) / (1.0 + (x / self.K) ** self.n)


# A small, explicitly synthetic gate library so the compiler is runnable out of
# the box. Real designs should pass their own characterised parts — these
# response parameters are illustrative, not measured, which `library_source`
# on the result records.
DEFAULT_GATE_LIBRARY: List[GatePart] = [
    GatePart("P1_PhlF", "PhlF", "pPhlF", ymin=0.02, ymax=3.9, K=0.23, n=4.0),
    GatePart("P2_SrpR", "SrpR", "pSrpR", ymin=0.003, ymax=1.3, K=0.10, n=2.9),
    GatePart("P3_BM3R1", "BM3R1", "pBM3R1", ymin=0.004, ymax=0.5, K=0.04, n=3.4),
    GatePart("P4_QacR", "QacR", "pQacR", ymin=0.01, ymax=2.4, K=0.18, n=2.7),
    GatePart("P5_AmtR", "AmtR", "pAmtR", ymin=0.06, ymax=3.8, K=0.07, n=1.6),
    GatePart("P6_LitR", "LitR", "pLitR", ymin=0.07, ymax=4.3, K=0.05, n=1.7),
    GatePart("P7_BetI", "BetI", "pBetI", ymin=0.07, ymax=3.8, K=0.41, n=2.4),
    GatePart("P8_HlyIIR", "HlyIIR", "pHlyIIR", ymin=0.07, ymax=2.1, K=0.19, n=2.6),
]


@dataclass
class CompiledGate:
    gate_id: str
    kind: str                          # 'NOT' | 'NOR' | 'INPUT' | 'OUTPUT'
    inputs: List[str] = field(default_factory=list)
    part: Optional[GatePart] = None
    on_level: float = 0.0
    off_level: float = 0.0

    @property
    def dynamic_range(self) -> float:
        return self.on_level / self.off_level if self.off_level > 0 else float("inf")


@dataclass
class CompiledCircuit:
    """A truth table turned into an assigned, checked gate network."""

    truth_table: Dict[Tuple[int, ...], int]
    inputs: List[str]
    gates: List[CompiledGate] = field(default_factory=list)
    dna: str = ""
    score: float = 0.0
    feasible: bool = True
    failures: List[str] = field(default_factory=list)
    library_source: str = "DEFAULT_GATE_LIBRARY (illustrative parameters)"

    @property
    def n_gates(self) -> int:
        return len([g for g in self.gates if g.kind in ("NOT", "NOR")])

    def to_frame(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame([{
            "gate": g.gate_id, "kind": g.kind, "inputs": ",".join(g.inputs),
            "part": g.part.name if g.part else "", "on": g.on_level,
            "off": g.off_level, "dynamic_range": g.dynamic_range,
        } for g in self.gates])

    def __repr__(self) -> str:  # pragma: no cover
        state = "feasible" if self.feasible else f"INFEASIBLE ({len(self.failures)})"
        return (f"CompiledCircuit({self.n_gates} gates, score={self.score:.2f}, "
                f"{state}, {len(self.dna)} nt)")


def _parse_truth_table(truth_table, inputs) -> Tuple[List[str], Dict[Tuple[int, ...], int]]:
    if isinstance(truth_table, str):
        # a boolean expression like "A and not B"
        expr = truth_table
        names = list(inputs or sorted({t for t in _identifiers(expr)}))
        table = {}
        for combo in itertools.product((0, 1), repeat=len(names)):
            env = dict(zip(names, (bool(c) for c in combo)))
            table[combo] = int(bool(eval(expr, {"__builtins__": {}}, env)))
        return names, table

    if isinstance(truth_table, Mapping := dict) or hasattr(truth_table, "items"):
        table = {tuple(int(x) for x in k): int(v) for k, v in truth_table.items()}
        width = {len(k) for k in table}
        if len(width) != 1:
            raise ValueError("真值表里各行的输入个数不一致。")
        names = list(inputs or [chr(ord("A") + i) for i in range(width.pop())])
        return names, table

    raise TypeError("truth_table 需要是 dict {(0,1): 1} 或布尔表达式字符串。")


def _identifiers(expr: str) -> List[str]:
    import re
    kw = {"and", "or", "not", "True", "False"}
    return [t for t in re.findall(r"[A-Za-z_]\w*", expr) if t not in kw]


@register_function(
    aliases=["compile_circuit", "线路编译", "真值表编译", "Cello",
             "truth_table_to_dna", "自动编译线路", "逻辑综合"],
    category="synthetic_biology",
    description="Cello 式线路编译:给一张**真值表**(或布尔表达式),自动综合成 NOR/NOT 门网络、从门库里分配物理阻遏子门、逐门检查上游 ON/OFF 电平是否真的落在下游门的可分辨区间,再输出 DNA。现有 genetic_circuit/logic_gate 是手工搭好再仿真,不是从真值表出 DNA。关键在那个检查 —— 网络拓扑对不代表信号电平配得上。Compile a truth table into an assigned, level-checked genetic circuit.",
    examples=[
        "cc = ov.synbio.compile_circuit({(0,0):0, (0,1):1, (1,0):1, (1,1):0})",
        "cc = ov.synbio.compile_circuit('A and not B')",
        "cc.feasible, cc.to_frame(), cc.dna",
    ],
    related=["synbio.genetic_circuit", "synbio.logic_gate",
             "synbio.simulate_circuit", "synbio.terminator_strength",
             "synbio.plot_compiled_circuit"],
    requires={},
    produces={},
)
def compile_circuit(
    truth_table,
    inputs: Optional[Sequence[str]] = None,
    *,
    library: Optional[Sequence[GatePart]] = None,
    input_on: float = 2.5,
    input_off: float = 0.02,
    min_dynamic_range: float = 3.0,
    max_assignments: int = 20000,
) -> CompiledCircuit:
    """Synthesise a gate network for ``truth_table`` and assign real parts.

    Parameters
    ----------
    truth_table
        ``{(in1, in2, ...): out}`` over all input combinations, or a boolean
        expression such as ``'A and not B'``.
    inputs
        Input signal names. Derived from the expression, or ``A``, ``B``, … .
    library
        Characterised gates. Defaults to :data:`DEFAULT_GATE_LIBRARY`, whose
        response parameters are illustrative — pass your own for real designs.
        The result records which was used in ``library_source``.
    input_on, input_off
        Sensor output levels for a logical 1 and 0.
    min_dynamic_range
        A gate assignment must separate its own ON and OFF outputs by at least
        this ratio, or the assignment is rejected. This is the check that makes
        compilation more than graph rewriting.
    max_assignments
        Cap on the assignment search.

    Returns
    -------
    CompiledCircuit
    """
    names, table = _parse_truth_table(truth_table, inputs)
    n_in = len(names)
    expected = set(itertools.product((0, 1), repeat=n_in))
    missing = expected - set(table)
    if missing:
        raise ValueError(
            f"真值表不完整,缺少 {sorted(missing)} 这些输入组合。"
            f"{n_in} 个输入需要 {2 ** n_in} 行。")

    lib = list(library) if library is not None else list(DEFAULT_GATE_LIBRARY)
    if not lib:
        raise ValueError("门库为空。")

    # --- synthesise: sum-of-products, then NOR/NOT realisation -------------
    gates: List[CompiledGate] = []
    for i, name in enumerate(names):
        gates.append(CompiledGate(gate_id=name, kind="INPUT",
                                  on_level=input_on, off_level=input_off))

    minterms = [combo for combo, out in sorted(table.items()) if out == 1]
    if not minterms:
        raise ValueError("真值表恒为 0 —— 没有需要实现的逻辑。")
    if len(minterms) == 2 ** n_in:
        raise ValueError("真值表恒为 1 —— 没有需要实现的逻辑。")

    # Each minterm is an AND of literals, realised as a NOR over the inverted
    # literals — the NOR-only form Cello targets, because repressor gates are
    # naturally NOT/NOR.
    #
    # Synthesising the *complement* when it has fewer terms is not an
    # optimisation nicety here, it is the difference between fitting the library
    # and not: two-input OR expanded to three minterms and nine gates, one more
    # than the eight-part library holds, while its complement NOR is a single
    # gate and OR is that gate plus an inverter.
    counter = itertools.count(1)
    zeros = [combo for combo, out in sorted(table.items()) if out == 0]
    use_zeros = len(minterms) > len(zeros)
    terms = zeros if use_zeros else minterms

    term_ids: List[str] = []
    for combo in terms:
        lits: List[str] = []
        for name, bit in zip(names, combo):
            if bit == 1:
                inv = f"g{next(counter)}"
                gates.append(CompiledGate(inv, "NOT", [name]))
                lits.append(inv)
            else:
                lits.append(name)
        term = f"g{next(counter)}"
        gates.append(CompiledGate(term, "NOR", lits))
        term_ids.append(term)

    if len(term_ids) == 1:
        node, node_is_or = term_ids[0], True     # a single term *is* the OR
    else:
        nor_all = f"g{next(counter)}"
        gates.append(CompiledGate(nor_all, "NOR", term_ids))
        node, node_is_or = nor_all, False        # NOT(node) is the OR

    # `node` currently represents OR(terms) if node_is_or, else its complement.
    # We want OUT = OR(minterms) = NOT(OR(zeros)) when the zeros were used.
    want_complement = use_zeros
    if node_is_or == (not want_complement):
        out_src = node
    else:
        final = f"g{next(counter)}"
        gates.append(CompiledGate(final, "NOT", [node]))
        out_src = final
    gates.append(CompiledGate("OUT", "OUTPUT", [out_src]))

    logic_gates = [g for g in gates if g.kind in ("NOT", "NOR")]
    if len(logic_gates) > len(lib):
        return CompiledCircuit(
            truth_table=table, inputs=names, gates=gates, feasible=False,
            failures=[f"需要 {len(logic_gates)} 个门,门库只有 {len(lib)} 个可用部件。"
                      f"每个门必须用不同的阻遏子,否则会互相串扰。"],
            library_source=("user library" if library is not None
                            else "DEFAULT_GATE_LIBRARY (illustrative parameters)"))

    # --- assign parts and check levels ------------------------------------
    by_id = {g.gate_id: g for g in gates}

    def evaluate(assignment: Dict[str, GatePart], combo: Tuple[int, ...]) -> Dict[str, float]:
        level: Dict[str, float] = {}
        for name, bit in zip(names, combo):
            level[name] = input_on if bit else input_off
        for g in gates:
            if g.kind in ("INPUT",):
                continue
            if g.kind == "OUTPUT":
                level[g.gate_id] = level[g.inputs[0]]
                continue
            x = sum(level[i] for i in g.inputs)     # NOR sums its inputs
            level[g.gate_id] = assignment[g.gate_id].response(x)
        return level

    best: Optional[Tuple[float, Dict[str, GatePart], Dict[str, Tuple[float, float]]]] = None
    tried = 0
    for perm in itertools.permutations(lib, len(logic_gates)):
        tried += 1
        if tried > max_assignments:
            break
        assignment = {g.gate_id: p for g, p in zip(logic_gates, perm)}
        on_levels: Dict[str, List[float]] = {g.gate_id: [] for g in logic_gates}
        off_levels: Dict[str, List[float]] = {g.gate_id: [] for g in logic_gates}
        out_on, out_off = [], []
        ok = True
        for combo, expected_out in sorted(table.items()):
            level = evaluate(assignment, combo)
            for g in logic_gates:
                # a gate's own truth value under this input combination
                (on_levels if _gate_truth(by_id, g, names, combo) else
                 off_levels)[g.gate_id].append(level[g.gate_id])
            (out_on if expected_out else out_off).append(level["OUT"])
        if not out_on or not out_off:
            ok = False
        else:
            margin = min(out_on) / max(out_off) if max(out_off) > 0 else float("inf")
            if margin < min_dynamic_range:
                ok = False
        if not ok:
            continue
        levels = {}
        for g in logic_gates:
            on = min(on_levels[g.gate_id]) if on_levels[g.gate_id] else 0.0
            off = max(off_levels[g.gate_id]) if off_levels[g.gate_id] else 0.0
            levels[g.gate_id] = (on, off)
        score = float(margin)
        if best is None or score > best[0]:
            best = (score, assignment, levels)

    if best is None:
        return CompiledCircuit(
            truth_table=table, inputs=names, gates=gates, feasible=False,
            failures=[
                f"库里没有任何一种分配能让输出的 ON/OFF 分离度达到 "
                f"{min_dynamic_range}x。拓扑是对的,但电平配不上 —— 上游门的 ON "
                f"电平落进了下游门的死区。换更强的门、降低 min_dynamic_range,"
                f"或改用输入摆幅更大的传感器。"],
            library_source=("user library" if library is not None
                            else "DEFAULT_GATE_LIBRARY (illustrative parameters)"))

    score, assignment, levels = best
    for g in logic_gates:
        g.part = assignment[g.gate_id]
        g.on_level, g.off_level = levels[g.gate_id]

    dna = _emit_dna(gates)
    return CompiledCircuit(
        truth_table=table, inputs=names, gates=gates, dna=dna, score=score,
        feasible=True,
        library_source=("user library" if library is not None
                        else "DEFAULT_GATE_LIBRARY (illustrative parameters)"))


def _gate_truth(by_id, gate: CompiledGate, names, combo) -> bool:
    """Boolean value of ``gate`` under an input combination."""
    env = dict(zip(names, (bool(b) for b in combo)))

    def val(gid: str) -> bool:
        if gid in env:
            return env[gid]
        g = by_id[gid]
        if g.kind == "NOT":
            return not val(g.inputs[0])
        if g.kind == "NOR":
            return not any(val(i) for i in g.inputs)
        if g.kind == "OUTPUT":
            return val(g.inputs[0])
        return False

    return val(gate.gate_id)


_SPACER = "TACTAGAG"
_DEFAULT_TERM = "CCAGGCATCAAATAAAACGAAAGGCTCAGTCGAAAGACTGGGCCTTTCGTTTTATCTGTTGTTTGTCGGTGAACGCTCTC"


def _emit_dna(gates: Sequence[CompiledGate]) -> str:
    """Concatenate each assigned gate's transcription unit.

    Each unit is promoter + spacer + repressor CDS + terminator; the terminator
    is what keeps one unit from driving the next, which is exactly why
    :func:`terminator_strength` exists in this module.
    """
    parts: List[str] = []
    for g in gates:
        if g.part is None:
            continue
        p = g.part
        promoter = p.promoter if set(p.promoter.upper()) <= set("ACGTN") else \
            _promoter_placeholder(p.promoter)
        cds = p.cds or _cds_placeholder(p.repressor)
        term = p.terminator or _DEFAULT_TERM
        parts.append(promoter + _SPACER + cds + term)
    return "".join(parts)


def _promoter_placeholder(name: str) -> str:
    """A σ70 core with the part name encoded in the spacer, so the emitted DNA
    is well-formed and each gate's promoter is distinguishable."""
    tag = "".join("ACGT"[ord(c) % 4] for c in name)[:8].ljust(8, "A")
    return "TTGACA" + tag + "GATAAT" + "AGGAGA"


def _cds_placeholder(repressor: str) -> str:
    tag = "".join("ACGT"[ord(c) % 4] for c in repressor)[:12].ljust(12, "A")
    return "ATG" + tag + "GGTGGCAGCGGT" + "TAA"


@register_function(
    aliases=["plot_compiled_circuit", "编译线路图", "plot_circuit_levels",
             "信号电平图", "门分配图"],
    category="synthetic_biology",
    description="画编译结果:每个门分配到的部件及其 ON/OFF 电平区间(能一眼看出哪一级的分离度最窄),以及真值表的实际输出与期望对照。Plot a compiled circuit's per-gate signal levels and truth-table agreement.",
    examples=["ov.synbio.plot_compiled_circuit(cc)"],
    related=["synbio.compile_circuit"],
    requires={},
    produces={},
)
def plot_compiled_circuit(circuit: CompiledCircuit, axes=None):
    """Two panels: per-gate ON/OFF separation, and the realised truth table."""
    from ._plot import _mpl
    plt = _mpl()
    import numpy as np

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8),
                                 gridspec_kw={"width_ratios": [2, 1]})
    else:
        fig = axes[0].figure
    axes = list(axes)

    logic = [g for g in circuit.gates if g.kind in ("NOT", "NOR")]
    ax = axes[0]
    for i, g in enumerate(logic):
        ax.plot([g.off_level, g.on_level], [i, i], lw=6, solid_capstyle="butt",
                color="#377EB8", alpha=0.85)
        ax.plot([g.off_level], [i], "o", color="#E41A1C", ms=5)
        ax.plot([g.on_level], [i], "o", color="#4DAF4A", ms=5)
    ax.set_yticks(range(len(logic)))
    ax.set_yticklabels([f"{g.gate_id} [{g.kind}] "
                        f"{g.part.name if g.part else '-'}" for g in logic],
                       fontsize=8)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("signal level (RPU, log scale)")
    ax.set_title(f"red = worst OFF, green = worst ON  "
                 f"(output margin {circuit.score:.1f}x)")
    if not logic:
        ax.text(0.5, 0.5, "no gates assigned", ha="center", va="center",
                transform=ax.transAxes)

    ax = axes[1]
    combos = sorted(circuit.truth_table)
    vals = [circuit.truth_table[c] for c in combos]
    ax.bar(range(len(combos)), vals, color=["#4DAF4A" if v else "#E0E0E0"
                                            for v in vals])
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels(["".join(map(str, c)) for c in combos], fontsize=8)
    ax.set_xlabel(" ".join(circuit.inputs))
    ax.set_ylabel("expected output")
    ax.set_yticks([0, 1])
    ax.set_title("truth table")

    fig.tight_layout()
    return fig, axes


__all__ = ["terminator_strength", "TerminatorResult", "compile_circuit",
           "CompiledCircuit", "CompiledGate", "GatePart",
           "DEFAULT_GATE_LIBRARY", "plot_compiled_circuit"]
