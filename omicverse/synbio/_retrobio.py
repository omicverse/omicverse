r"""Retrobiosynthesis — enzymatic reaction-rule retrosynthesis over a target
molecule.

Where :func:`~omicverse.synbio._retro.pathway_search` searches for routes to a
metabolite *within a genome-scale model* (native reactions only), this module
proposes **novel enzymatic steps** by applying reaction rules — the RetroRules /
RetroPath approach. Each rule is a reaction SMARTS keyed to an EC class; applied
in the retro direction to a target SMILES it enumerates the precursor(s) an
enzyme of that class could have come from.

* :func:`retro_biosynthesis` — one- or multi-step retrobiosynthetic expansion of
  a target molecule into candidate precursors + the enzyme class for each step.

Ships a curated, chemically-checked rule set covering the major reversible
biotransformations (oxidoreduction, transamination, (de)phosphorylation,
(de)methylation, (de)acetylation, hydration/dehydration, aldehyde↔acid). Supply
your own ``rules`` (e.g. exported from the full RetroRules database) to extend
coverage. Uses RDKit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .._registry import register_function

# curated retro reaction rules: name -> (EC class, "product>>precursor" SMARTS).
# Written in the RETRO direction (LHS matches the target/product, RHS is the
# precursor the enzyme would have acted on).
_RULES: Dict[str, Tuple[str, str]] = {
    "carbonyl_reduction":  ("1.1.1  oxidoreductase (alcohol←carbonyl)",
                            "[C;!$(C=O);!$(C[OX1]):1][OX2H1:2]>>[C:1]=[O:2]"),
    "aldehyde_oxidation":  ("1.2.1  oxidoreductase (acid←aldehyde)",
                            "[C:1][CX3](=[O:2])[OX2H1]>>[C:1][CX3H1]=[O:2]"),
    "transamination":      ("2.6.1  transaminase (amino-acid←2-oxo-acid)",
                            "[CX4:1]([NX3H2])[CX3:2](=O)[OX2H1]>>[CX4:1](=O)[CX3:2](=O)[OX2H1]"),
    "dephosphorylation":   ("3.1.3 / 2.7  (de)phosphorylation",
                            "[OX2:1]P(=O)([OX2H,OX1-])[OX2H,OX1-]>>[OX2H:1]"),
    "O_demethylation":     ("2.1.1  methyltransferase (O-methyl←OH)",
                            "[#6:1][OX2][CH3]>>[#6:1][OX2H]"),
    "O_deacetylation":     ("2.3.1  acyltransferase (O-acetyl←OH)",
                            "[#6:1][OX2]C(=O)[CH3]>>[#6:1][OX2H]"),
    "dehydration":         ("4.2.1  hydro-lyase (alkene←alcohol)",
                            "[CX4:1][CX4:2][OX2H1]>>[C:1]=[C:2]"),
}


def _rdkit(fn: str):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        return Chem, AllChem
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"ov.synbio.{fn} 需要 RDKit。请 pip install rdkit "
            "(或 pip install 'omicverse[synbio]')。"
        ) from exc


@dataclass
class RetroStep:
    rule: str               # rule name
    ec: str                 # EC class / description
    precursors: List[str]   # precursor SMILES
    product: str            # the target this step makes

    def __repr__(self) -> str:  # pragma: no cover
        return f"RetroStep({self.rule}: {' + '.join(self.precursors)} -> {self.product})"


@dataclass
class RetroRoute:
    target: str
    steps: List[RetroStep]
    precursors: List[str]   # leaf precursors of the whole route

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"RetroRoute({self.n_steps} step(s): "
                f"{' | '.join(s.rule for s in self.steps)} -> {self.target})")


def _canon(Chem, smi: str) -> Optional[str]:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else None


def _apply_rules(Chem, AllChem, target: str, rules) -> List[RetroStep]:
    steps: List[RetroStep] = []
    mol = Chem.MolFromSmiles(target)
    if mol is None:
        return steps
    seen = set()
    for name, (ec, smarts) in rules.items():
        try:
            rxn = AllChem.ReactionFromSmarts(smarts)
        except Exception:
            continue
        try:
            products = rxn.RunReactants((mol,))
        except Exception:
            continue
        for prods in products:
            precs = []
            ok = True
            for p in prods:
                try:
                    Chem.SanitizeMol(p)
                    precs.append(Chem.MolToSmiles(p))
                except Exception:
                    ok = False
                    break
            if not ok or not precs:
                continue
            key = (name, tuple(sorted(precs)))
            if key in seen:
                continue
            seen.add(key)
            steps.append(RetroStep(rule=name, ec=ec, precursors=precs,
                                   product=target))
    return steps


@register_function(
    aliases=["retro_biosynthesis", "逆生物合成", "retrobiosynthesis", "反应规则逆合成",
             "retrosynthesis", "逆合成", "retropath", "酶促逆合成"],
    category="synthetic_biology",
    description="逆生物合成:用酶促反应规则(RetroRules/RetroPath 思路)对目标分子做逆合成扩展,枚举每一步可能的前体与酶类(EC)。区别于在 GEM 内检索的 pathway_search,这里可提出非天然的全新酶促步骤。Reaction-rule retrobiosynthesis of a target molecule (novel enzymatic steps).",
    examples=[
        "routes = ov.synbio.retro_biosynthesis('CC(=O)C(=O)O')  # pyruvate",
        "routes[0].steps, routes[0].precursors",
    ],
    related=["synbio.pathway_search", "synbio.reaction_dg", "synbio.max_min_driving_force"],
    requires={},
    produces={},
)
def retro_biosynthesis(target_smiles: str, generations: int = 1,
                       max_routes: int = 25,
                       rules: Optional[Dict[str, Tuple[str, str]]] = None
                       ) -> List[RetroRoute]:
    """Retrobiosynthetically expand *target_smiles* into precursor routes.

    Applies enzymatic reaction rules in the retro direction. ``generations``
    controls recursion depth (multi-step routes); ``rules`` overrides the
    curated default set (pass reaction SMARTS from the full RetroRules DB to
    broaden coverage). Returns routes ranked by fewest steps."""
    Chem, AllChem = _rdkit("retro_biosynthesis")
    rules = rules or _RULES
    tgt = _canon(Chem, target_smiles)
    if tgt is None:
        raise ValueError(f"无法解析 target_smiles={target_smiles!r}(不是合法 SMILES)。")

    # generation 0: one-step expansions of the target
    routes: List[RetroRoute] = []
    frontier: List[RetroRoute] = []
    for step in _apply_rules(Chem, AllChem, tgt, rules):
        r = RetroRoute(target=tgt, steps=[step], precursors=list(step.precursors))
        routes.append(r)
        frontier.append(r)

    # further generations: expand each precursor of frontier routes
    for _ in range(max(0, generations - 1)):
        nxt: List[RetroRoute] = []
        for route in frontier:
            for i, prec in enumerate(route.precursors):
                for step in _apply_rules(Chem, AllChem, prec, rules):
                    new_precs = (route.precursors[:i] + list(step.precursors)
                                 + route.precursors[i + 1:])
                    r = RetroRoute(target=tgt, steps=route.steps + [step],
                                   precursors=new_precs)
                    routes.append(r)
                    nxt.append(r)
        frontier = nxt
        if not frontier:
            break

    routes.sort(key=lambda r: r.n_steps)
    return routes[:max_routes]


@register_function(
    aliases=["plot_retro_routes", "逆合成图", "retro_plot", "逆合成路径图",
             "retrosynthesis_plot", "分子图"],
    category="synthetic_biology",
    description="画逆生物合成路径:把目标分子与每一步的前体用 RDKit 结构式画成网格,标注反应规则/EC。Draw retrobiosynthetic routes (target + precursors) as an RDKit molecule grid.",
    examples=[
        "routes = ov.synbio.retro_biosynthesis('CC(=O)C(=O)O')",
        "ov.synbio.plot_retro_routes(routes, n=3)",
    ],
    related=["synbio.retro_biosynthesis"],
    requires={},
    produces={},
)
def plot_retro_routes(routes: List[RetroRoute], n: int = 3, ax=None,
                      mols_per_row: int = 4):
    """Draw the top *n* one-step routes as a molecule grid (target → precursors)."""
    Chem, _ = _rdkit("plot_retro_routes")
    from rdkit.Chem import Draw
    from ._plot import _mpl
    plt = _mpl()
    import numpy as np

    mols, legends = [], []
    if routes:
        t = Chem.MolFromSmiles(routes[0].target)
        if t is not None:
            mols.append(t)
            legends.append("TARGET")
    for r in routes[:n]:
        step = r.steps[0]
        for p in step.precursors:
            m = Chem.MolFromSmiles(p)
            if m is not None:
                mols.append(m)
                legends.append(step.rule)
    if not mols:
        raise ValueError("没有可画的分子(routes 为空或 SMILES 非法)。")
    per_row = min(mols_per_row, len(mols))
    # force a PIL image (in a notebook RDKit's IPython integration can otherwise
    # return SVG/PNG-bytes, which np.asarray can't turn into a float array)
    img = Draw.MolsToGridImage(mols, legends=legends, molsPerRow=per_row,
                               subImgSize=(220, 180), useSVG=False,
                               returnPNG=False)
    if isinstance(img, (bytes, bytearray)):
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(img))
    arr = np.asarray(img.convert("RGB"))
    if ax is None:
        fig, ax = plt.subplots(figsize=(per_row * 2.4,
                                        2.2 * (1 + (len(mols) - 1) // per_row)))
    else:
        fig = ax.figure
    ax.imshow(arr)
    ax.axis("off")
    ax.set_title("Retrobiosynthesis: target → precursors", fontsize=11)
    fig.tight_layout()
    return fig, ax


__all__ = ["retro_biosynthesis", "plot_retro_routes", "RetroStep", "RetroRoute"]
