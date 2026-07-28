r"""Genetic-circuit design & simulation — the regulatory heart of synbio.

Build a gene-regulatory network from parts (genes + activation/repression
edges with Hill kinetics), then simulate it **deterministically** (ODEs, scipy)
or **stochastically** (Gillespie SSA, numpy — captures the molecular noise that
makes small circuits behave probabilistically).

Ready-made templates for the canonical circuits:

* :func:`constitutive` — a single unregulated gene.
* :func:`toggle_switch` — Gardner *et al.* 2000 bistable switch (two mutually
  repressing genes).
* :func:`repressilator` — Elowitz & Leibler 2000 three-gene oscillator.
* :func:`logic_gate` — transcriptional AND / OR / NOT / NAND / NOR gates.
* :func:`feed_forward_loop` — coherent/incoherent FFL motifs.

All CPU, no heavy dependencies (scipy is already an omicverse dependency).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from .._registry import register_function

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


@dataclass
class Regulation:
    regulator: str
    target: str
    kind: str = "repress"   # 'repress' | 'activate'
    K: float = 40.0          # half-max threshold (molecules)
    n: float = 2.0           # Hill coefficient (cooperativity)


@dataclass
class Gene:
    name: str
    basal: float = 1.0       # leaky production (molecules/min)
    beta: float = 100.0      # max regulated production (molecules/min)
    degradation: float = 1.0  # 1/min (protein turnover + dilution)
    inputs: List[str] = field(default_factory=list)  # external inducer names
    #: How multiple ``inputs`` combine at the promoter: ``'and'`` multiplies the
    #: Hill terms (both inducers needed), ``'or'`` takes ``1 - Π(1 - hᵢ)``
    #: (either suffices).
    input_logic: str = "and"
    input_K: float = 40.0    # inducer half-max, in inducer units
    input_n: float = 2.0     # inducer Hill coefficient


@dataclass
class GeneticCircuit:
    """A gene-regulatory network: genes + Hill-type regulatory edges.

    Production of each gene is ``basal + beta * f(regulators)`` where ``f`` is a
    product of Hill terms (repressors: ``K^n/(K^n+R^n)``; activators:
    ``A^n/(K^n+A^n)``); every gene also degrades first-order.
    """

    name: str = "circuit"
    genes: Dict[str, Gene] = field(default_factory=dict)
    regulations: List[Regulation] = field(default_factory=list)
    inducers: Dict[str, float] = field(default_factory=dict)

    def add_gene(self, name: str, **kw) -> "GeneticCircuit":
        self.genes[name] = Gene(name=name, **kw)
        return self

    def add_regulation(self, regulator: str, target: str, kind: str = "repress",
                       K: float = 40.0, n: float = 2.0) -> "GeneticCircuit":
        self.regulations.append(Regulation(regulator, target, kind, K, n))
        return self

    def set_inducer(self, name: str, level: float) -> "GeneticCircuit":
        """Set an external inducer level (relieves a repressor / drives input)."""
        self.inducers[name] = level
        return self

    # -- kinetics -----------------------------------------------------------
    def _production(self, name: str, state: Dict[str, float]) -> float:
        g = self.genes[name]
        factor = 1.0
        regs = [r for r in self.regulations if r.target == name]
        for r in regs:
            level = state.get(r.regulator, 0.0)
            # an inducer of the same name as the regulator sequesters it
            eff = max(0.0, level - self.inducers.get(r.regulator, 0.0))
            hill = (eff ** r.n) / (r.K ** r.n + eff ** r.n)
            factor *= (r.K ** r.n) / (r.K ** r.n + eff ** r.n) if r.kind == "repress" else hill
        if g.inputs:
            hills = []
            for inp in g.inputs:
                a = self.inducers.get(inp, 0.0)
                hills.append((a ** g.input_n)
                             / (g.input_K ** g.input_n + a ** g.input_n))
            if g.input_logic == "or":
                # either activator suffices — the behaviour of a promoter with
                # two independent activator sites. Multiplying Hill terms (the
                # only combination the model used to offer) is AND and cannot
                # express this, which is why `logic_gate('OR')` was a buffer on
                # in1 with a second, unconnected gene hanging off in2.
                combined = 1.0
                for h in hills:
                    combined *= (1.0 - h)
                factor *= (1.0 - combined)
            else:
                for h in hills:
                    factor *= h
        return g.basal + g.beta * factor

    def species(self) -> List[str]:
        return list(self.genes.keys())

    def __repr__(self) -> str:  # pragma: no cover
        return (f"GeneticCircuit({self.name!r}: {len(self.genes)} genes, "
                f"{len(self.regulations)} regulatory edges)")


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------

@register_function(
    aliases=["genetic_circuit", "基因线路", "基因电路", "gene_circuit",
             "regulatory_network", "GRN", "基因调控网络"],
    category="synthetic_biology",
    description="构建基因线路(基因调控网络):从内置模板(组成型/toggle/振荡子/逻辑门/FFL)或手动加基因+调控边,得到可仿真的 GeneticCircuit。Build a gene-regulatory circuit from templates or parts.",
    examples=[
        "c = ov.synbio.genetic_circuit('toggle_switch')",
        "c = ov.synbio.genetic_circuit('repressilator')",
        "df = ov.synbio.simulate_circuit(c, t_end=200)",
    ],
    related=["synbio.simulate_circuit", "synbio.plot_circuit"],
    requires={},
    produces={},
)
def genetic_circuit(template: str = "constitutive", **kw) -> GeneticCircuit:
    """Return a :class:`GeneticCircuit` from a named template.

    Templates: ``constitutive``, ``toggle_switch``, ``repressilator``,
    ``logic_gate`` (pass ``gate='AND'|'OR'|'NOT'|'NAND'|'NOR'``),
    ``feed_forward_loop`` (pass ``kind='coherent'|'incoherent'``)."""
    t = template.lower()
    if t in ("constitutive", "single"):
        return constitutive()
    if t in ("toggle", "toggle_switch"):
        return toggle_switch()
    if t in ("repressilator", "oscillator"):
        return repressilator()
    if t in ("logic_gate", "gate"):
        return logic_gate(kw.get("gate", "AND"))
    if t in ("feed_forward_loop", "ffl"):
        return feed_forward_loop(kw.get("kind", "incoherent"))
    raise ValueError(
        f"未知 template='{template}'。可用:constitutive / toggle_switch / "
        f"repressilator / logic_gate / feed_forward_loop。")


def constitutive() -> GeneticCircuit:
    c = GeneticCircuit("constitutive")
    c.add_gene("X", basal=0.0, beta=50.0, degradation=1.0)
    c.genes["X"].inputs = ["inducer"]
    return c


def toggle_switch() -> GeneticCircuit:
    """Gardner 2000 bistable toggle: A ⊣ B, B ⊣ A."""
    c = GeneticCircuit("toggle_switch")
    c.add_gene("A", basal=1.0, beta=100.0, degradation=1.0)
    c.add_gene("B", basal=1.0, beta=100.0, degradation=1.0)
    c.add_regulation("A", "B", "repress", K=40, n=3)
    c.add_regulation("B", "A", "repress", K=40, n=3)
    return c


def repressilator() -> GeneticCircuit:
    """Elowitz 2000 three-gene ring oscillator: A ⊣ B ⊣ C ⊣ A."""
    # params in the oscillatory regime (Hopf-unstable fixed point): n=3, high β.
    c = GeneticCircuit("repressilator")
    for g in ("A", "B", "C"):
        c.add_gene(g, basal=0.1, beta=300.0, degradation=1.0)
    c.add_regulation("A", "B", "repress", K=40, n=3)
    c.add_regulation("B", "C", "repress", K=40, n=3)
    c.add_regulation("C", "A", "repress", K=40, n=3)
    return c


LOGIC_GATES = ("AND", "OR", "NAND", "NOR", "NOT")


def logic_gate(gate: str = "AND") -> GeneticCircuit:
    """Transcriptional logic gate driven by inducers ``in1`` / ``in2``.

    The inverting gates get an actual inverting stage: an intermediate repressor
    that integrates the inputs, which then represses the output. Without it
    ``NAND`` was byte-identical to ``AND`` and ``NOR`` to ``OR`` — four of the
    five gates returned the wrong truth table, and ``OR`` was not an OR either,
    because two Hill terms multiplied are an AND and the second gene ``Y2`` was
    never combined with ``Y``.

    Output species is always ``Y``. Use
    :meth:`~GeneticCircuit.set_inducer` to drive ``in1`` / ``in2``; the default
    half-max is 40 inducer units, so 0 is OFF and 100 is comfortably ON.
    """
    gate = gate.upper()
    if gate not in LOGIC_GATES:
        raise ValueError(f"gate must be one of {list(LOGIC_GATES)}, got {gate!r}")
    c = GeneticCircuit(f"logic_{gate}")

    if gate in ("AND", "OR"):
        c.add_gene("Y", basal=1.0, beta=100.0, degradation=1.0)
        c.genes["Y"].inputs = ["in1", "in2"]
        c.genes["Y"].input_logic = "and" if gate == "AND" else "or"
    elif gate in ("NAND", "NOR"):
        # inverting stage: R integrates the inputs, R represses Y
        c.add_gene("R", basal=0.0, beta=100.0, degradation=1.0)
        c.genes["R"].inputs = ["in1", "in2"]
        c.genes["R"].input_logic = "and" if gate == "NAND" else "or"
        c.add_gene("Y", basal=1.0, beta=100.0, degradation=1.0)
        c.add_regulation("R", "Y", "repress", K=40, n=2)
    else:                                     # NOT
        c.add_gene("R", basal=0.0, beta=100.0, degradation=1.0)
        c.genes["R"].inputs = ["in1"]
        c.add_gene("Y", basal=1.0, beta=100.0, degradation=1.0)
        c.add_regulation("R", "Y", "repress", K=40, n=2)
    return c


def feed_forward_loop(kind: str = "incoherent") -> GeneticCircuit:
    """3-node feed-forward loop (Alon motif). X→Y, X→Z, Y (⊣|→) Z."""
    c = GeneticCircuit(f"ffl_{kind}")
    c.add_gene("X", basal=0.0, beta=80.0, degradation=1.0)
    c.genes["X"].inputs = ["inducer"]
    c.add_gene("Y", basal=1.0, beta=80.0, degradation=1.0)
    c.add_gene("Z", basal=1.0, beta=80.0, degradation=1.0)
    c.add_regulation("X", "Y", "activate", K=40, n=2)
    c.add_regulation("X", "Z", "activate", K=40, n=2)
    c.add_regulation("Y", "Z", "repress" if kind == "incoherent" else "activate",
                     K=40, n=2)
    return c


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------

@register_function(
    aliases=["simulate_circuit", "线路仿真", "电路仿真", "circuit_simulation",
             "gillespie", "ode_simulation", "随机仿真", "确定性仿真"],
    category="synthetic_biology",
    description="仿真基因线路:确定性 ODE(scipy,Hill 动力学)或随机 Gillespie SSA(分子噪声)→ 时间×物种的 DataFrame。Simulate a genetic circuit (deterministic ODE or stochastic Gillespie).",
    examples=[
        "df = ov.synbio.simulate_circuit(c, t_end=200, method='ode')",
        "df = ov.synbio.simulate_circuit(c, t_end=200, method='stochastic')",
    ],
    related=["synbio.genetic_circuit", "synbio.plot_circuit"],
    requires={},
    produces={},
)
def simulate_circuit(circuit: GeneticCircuit, t_end: float = 200.0,
                     method: str = "ode", n_points: int = 400,
                     init: Optional[Dict[str, float]] = None,
                     seed: int = 0) -> "pd.DataFrame":
    """Simulate *circuit* over ``[0, t_end]``.

    Parameters
    ----------
    circuit
        A :class:`GeneticCircuit`.
    method
        ``"ode"`` (deterministic, scipy ``solve_ivp``) or ``"stochastic"``
        (Gillespie direct SSA — production & degradation as discrete events).
    n_points
        Output time resolution.
    init
        Initial molecule counts (default: A=20, others small, to break symmetry).

    Returns
    -------
    pandas.DataFrame
        Index ``time``, one column per species.
    """
    import numpy as np
    import pandas as pd

    species = circuit.species()
    if init is None:
        init = {s: (20.0 if i == 0 else 5.0) for i, s in enumerate(species)}
    x0 = np.array([init.get(s, 0.0) for s in species], dtype=float)

    if method == "ode":
        from scipy.integrate import solve_ivp

        def deriv(t, x):
            state = {s: max(0.0, x[i]) for i, s in enumerate(species)}
            dx = np.zeros(len(species))
            for i, s in enumerate(species):
                dx[i] = circuit._production(s, state) - circuit.genes[s].degradation * x[i]
            return dx

        ts = np.linspace(0, t_end, n_points)
        sol = solve_ivp(deriv, (0, t_end), x0, t_eval=ts, method="LSODA",
                        rtol=1e-6, atol=1e-6)
        df = pd.DataFrame(sol.y.T, columns=species)
        df.insert(0, "time", sol.t)
        return df

    if method in ("stochastic", "gillespie", "ssa"):
        rng = np.random.default_rng(seed)
        x = x0.copy()
        t = 0.0
        rec_t = np.linspace(0, t_end, n_points)
        rec = np.zeros((n_points, len(species)))
        ri = 0
        while t < t_end and ri < n_points:
            state = {s: x[i] for i, s in enumerate(species)}
            # reactions: for each species, production (+1) and degradation (-1)
            props = []
            changes = []
            for i, s in enumerate(species):
                props.append(max(0.0, circuit._production(s, state)))
                changes.append((i, +1))
                props.append(max(0.0, circuit.genes[s].degradation * x[i]))
                changes.append((i, -1))
            total = sum(props)
            if total <= 0:
                break
            tau = rng.exponential(1.0 / total)
            t += tau
            # record any grid points passed
            while ri < n_points and rec_t[ri] <= t:
                rec[ri] = x
                ri += 1
            r = rng.random() * total
            cum = 0.0
            for p, (i, d) in zip(props, changes):
                cum += p
                if r <= cum:
                    x[i] = max(0.0, x[i] + d)
                    break
        while ri < n_points:
            rec[ri] = x
            ri += 1
        df = pd.DataFrame(rec, columns=species)
        df.insert(0, "time", rec_t)
        return df

    raise ValueError(f"未知 method='{method}' (ode / stochastic)")


@register_function(
    aliases=["plot_circuit", "线路时程图", "circuit_timecourse", "画线路"],
    category="synthetic_biology",
    description="画基因线路仿真时程:各物种浓度随时间变化(toggle 双稳、振荡子周期等)。Plot a genetic-circuit simulation timecourse.",
    examples=["ov.synbio.plot_circuit(ov.synbio.simulate_circuit(c, 200))"],
    related=["synbio.simulate_circuit", "synbio.genetic_circuit"],
    requires={},
    produces={},
)
def plot_circuit(df: "pd.DataFrame", ax=None, title: str = "genetic circuit"):
    """Plot a simulation DataFrame (from :func:`simulate_circuit`)."""
    from ._plot import _mpl
    plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.2))
    else:
        fig = ax.figure
    for col in df.columns:
        if col == "time":
            continue
        ax.plot(df["time"], df[col], label=col, lw=1.8)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("molecules / cell")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    return fig, ax


__all__ = ["GeneticCircuit", "Gene", "Regulation", "genetic_circuit",
           "constitutive", "toggle_switch", "repressilator", "logic_gate",
           "feed_forward_loop", "simulate_circuit", "plot_circuit"]
