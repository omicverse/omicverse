r"""The **B** in design-build-test-learn — turning a design into something
executable.

Everything upstream of this module produces knowledge: a pathway, an enzyme, a
construct, a vector, a screening verdict. None of it produces anything a robot
or a person at a bench can carry out. That made "design-build-test-learn" a
claim about three of the four letters.

* :func:`plate_layout` — assign designs to wells. Everything else here consumes
  a plate, so this is the foundation rather than a convenience.
* :func:`assembly_worklist` — the calculation actually done by hand before every
  assembly: equimolar fragment volumes from concentrations and lengths, plus
  master mix and water to a final volume.
* :func:`echo_picklist` — a Beckman/Labcyte Echo pick list, in the column format
  the Echo Cherry Pick application reads.
* :func:`opentrons_protocol` — a runnable Opentrons Python Protocol API v2
  script for the transfers.
* :func:`pcr_protocol` — a thermocycler program derived from the primers' Tm
  and the amplicon length, not from a default anyone forgot to change.
* :func:`plot_plate` — the layout as a plate map, because a picklist is not
  reviewable and a plate map is.

**On correctness.** The Echo columns and the Opentrons API surface are
externally specified, so they are followed exactly rather than approximated:
``Source Plate Name`` / ``Source Well`` / ``Destination Plate Name`` /
``Destination Well`` / ``Transfer Volume`` for the Echo, and
``metadata`` / ``requirements`` / ``run(protocol: protocol_api.ProtocolContext)``
for Opentrons. Volumes come out in the units each instrument expects — nanolitres
for the Echo, microlitres for Opentrons — and the conversion is done once, here,
rather than left to whoever reads the file.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .._registry import register_function

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


PLATE_FORMATS: Dict[str, Tuple[int, int]] = {
    "6": (2, 3), "12": (3, 4), "24": (4, 6), "48": (6, 8),
    "96": (8, 12), "384": (16, 24), "1536": (32, 48),
}

_ROW_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZAA"

# Average molecular weight of a base pair of double-stranded DNA, g/mol.
# Used to convert ng/µL into nM, which is what equimolar mixing needs.
_BP_MW = 650.0


def _row_label(index: int) -> str:
    if index < 26:
        return _ROW_LETTERS[index]
    return "A" + _ROW_LETTERS[index - 26]


def well_name(row: int, column: int) -> str:
    """``(0, 0)`` → ``'A1'``. Rows and columns are 0-based."""
    return f"{_row_label(row)}{column + 1}"


def parse_well(well: str) -> Tuple[int, int]:
    """``'B7'`` → ``(1, 6)``."""
    w = str(well).strip().upper()
    letters = "".join(c for c in w if c.isalpha())
    digits = "".join(c for c in w if c.isdigit())
    if not letters or not digits:
        raise ValueError(f"无法解析孔位 {well!r};期望像 'A1' / 'H12' 这样的格式。")
    row = 0
    for c in letters:
        row = row * 26 + (ord(c) - ord("A") + 1)
    return row - 1, int(digits) - 1


@dataclass
class Plate:
    """A plate with named contents in wells."""

    name: str
    format: str = "96"
    contents: Dict[str, str] = field(default_factory=dict)   # well -> item
    volumes_ul: Dict[str, float] = field(default_factory=dict)
    concentrations: Dict[str, float] = field(default_factory=dict)  # ng/µL

    @property
    def shape(self) -> Tuple[int, int]:
        return PLATE_FORMATS[self.format]

    @property
    def capacity(self) -> int:
        rows, cols = self.shape
        return rows * cols

    @property
    def n_used(self) -> int:
        return len(self.contents)

    def well_of(self, item: str) -> Optional[str]:
        for well, name in self.contents.items():
            if name == item:
                return well
        return None

    def to_frame(self) -> "pd.DataFrame":
        import pandas as pd
        rows, cols = self.shape
        grid = [[self.contents.get(well_name(r, c), "") for c in range(cols)]
                for r in range(rows)]
        return pd.DataFrame(grid,
                            index=[_row_label(r) for r in range(rows)],
                            columns=[str(c + 1) for c in range(cols)])

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Plate({self.name!r}, {self.format}-well, "
                f"{self.n_used}/{self.capacity} used)")


@register_function(
    aliases=["plate_layout", "板布局", "孔位分配", "plate_map", "微孔板",
             "well_assignment"],
    category="synthetic_biology",
    description="把设计分配到微孔板孔位(6/12/24/48/96/384/1536)。按行或按列填充、可预留对照孔、可指定起始孔。Build 层其余功能(picklist、Opentrons 协议、组装工作单)都消费 Plate,所以这是基础而非便利函数。Assign designs to plate wells.",
    examples=[
        "plate = ov.synbio.plate_layout(['v1', 'v2', 'v3'], plate='96')",
        "plate = ov.synbio.plate_layout(designs, plate='384', by='column', controls=['blank', 'wt'])",
        "plate.to_frame()",
    ],
    related=["synbio.echo_picklist", "synbio.opentrons_protocol",
             "synbio.assembly_worklist", "synbio.plot_plate"],
    requires={},
    produces={},
)
def plate_layout(
    items: Sequence[str],
    *,
    plate: str = "96",
    name: str = "plate1",
    by: str = "row",
    start_well: str = "A1",
    controls: Optional[Sequence[str]] = None,
    replicates: int = 1,
    randomise: bool = False,
    avoid_edges: bool = False,
    seed: int = 0,
    volumes_ul: Optional[Mapping[str, float]] = None,
    concentrations: Optional[Mapping[str, float]] = None,
) -> Plate:
    """Lay ``items`` out on a plate.

    Parameters
    ----------
    items
        Names of the things being plated — designs, variants, strains.
    plate
        Plate format: ``'6'`` … ``'1536'``.
    by
        ``'row'`` fills A1, A2, A3 … ; ``'column'`` fills A1, B1, C1 … . Column
        order matters when an 8-channel pipette will read the plate.
    start_well
        First well to use, for leaving a block free.
    controls
        Placed *first*, before the items, so they land in a predictable spot
        rather than wherever the items happen to run out.
    replicates
        Plate each item this many times; replicate names get a ``_r2`` suffix.
    volumes_ul, concentrations
        Per-item volume (µL) and DNA concentration (ng/µL), carried on the plate
        so downstream worklists do not need them passed again.

    Returns
    -------
    Plate
    """
    if plate not in PLATE_FORMATS:
        raise ValueError(
            f"plate must be one of {sorted(PLATE_FORMATS, key=int)}, got {plate!r}")
    if by not in ("row", "column"):
        raise ValueError(f"by must be 'row' or 'column', got {by!r}")
    if replicates < 1:
        raise ValueError("replicates 必须 >= 1")

    rows, cols = PLATE_FORMATS[plate]
    start_row, start_col = parse_well(start_well)
    if not (0 <= start_row < rows and 0 <= start_col < cols):
        raise ValueError(
            f"起始孔 {start_well!r} 超出 {plate} 孔板范围({rows}x{cols})。")

    # Replicates are laid out in *rounds*, not consecutively. Placing r1, r2, r3
    # of the same item in adjacent wells is pseudo-replication: adjacent wells
    # share evaporation, thermal and optical micro-environment, so the replicates
    # agree for reasons that have nothing to do with the construct, and the
    # precision they buy is illusory exactly where replication was supposed to
    # protect against plate effects.
    expanded: List[str] = list(controls or [])
    for r in range(replicates):
        for item in items:
            expanded.append(str(item) if r == 0 else f"{item}_r{r + 1}")

    order: List[str] = []
    if by == "row":
        for r in range(rows):
            for c in range(cols):
                if (r, c) >= (start_row, start_col):
                    order.append(well_name(r, c))
    else:
        for c in range(cols):
            for r in range(rows):
                if (c, r) >= (start_col, start_row):
                    order.append(well_name(r, c))

    if avoid_edges and rows > 2 and cols > 2:
        # The perimeter of a microplate evaporates fastest and reads differently.
        # Reserving it costs wells and buys comparability; fill it with water.
        interior = [w for w in order
                    if 0 < parse_well(w)[0] < rows - 1
                    and 0 < parse_well(w)[1] < cols - 1]
        if len(expanded) <= len(interior):
            order = interior

    if len(expanded) > len(order):
        raise ValueError(
            f"{len(expanded)} 个条目(含对照与重复)放不进 {plate} 孔板的 "
            f"{len(order)} 个可用孔"
            + ("(已排除边缘孔;传 avoid_edges=False 可用满板)" if avoid_edges else "")
            + "。换更大的板,减少 replicates,或分多块板。")

    if randomise:
        # Run order and plate position are nuisance factors. Filling in design
        # order makes column index collinear with the factor settings, so a
        # position effect is indistinguishable from a real effect. There was no
        # way to randomise at all before this.
        import numpy as np
        rng = np.random.default_rng(seed)
        wells = list(order[:len(expanded)])
        rng.shuffle(wells)
        order = wells + list(order[len(expanded):])

    contents = {well: item for well, item in zip(order, expanded)}
    vols = {}
    concs = {}
    for well, item in contents.items():
        base = item.split("_r")[0]
        if volumes_ul and (item in volumes_ul or base in volumes_ul):
            vols[well] = float(volumes_ul.get(item, volumes_ul.get(base)))
        if concentrations and (item in concentrations or base in concentrations):
            concs[well] = float(concentrations.get(item,
                                                  concentrations.get(base)))
    return Plate(name=name, format=plate, contents=contents,
                 volumes_ul=vols, concentrations=concs)


# ---------------------------------------------------------------------------
# assembly worklist
# ---------------------------------------------------------------------------

@dataclass
class Transfer:
    """One liquid transfer."""

    source_plate: str
    source_well: str
    dest_plate: str
    dest_well: str
    volume_ul: float
    item: str = ""
    #: Which construct this transfer belongs to. Downstream emitters resolve the
    #: destination well from ``dest_plate.well_of(construct)``, which is what
    #: lets one worklist cover a whole library instead of a single reaction.
    construct: str = ""

    @property
    def volume_nl(self) -> float:
        return self.volume_ul * 1000.0

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Transfer({self.item or '?'}: {self.source_plate}/"
                f"{self.source_well} -> {self.dest_plate}/{self.dest_well}, "
                f"{self.volume_ul:.2f} µL)")


@dataclass
class Worklist:
    """A set of transfers plus what they are for."""

    transfers: List[Transfer] = field(default_factory=list)
    reaction: str = ""
    final_volume_ul: float = 20.0
    notes: List[str] = field(default_factory=list)

    @property
    def total_volume_ul(self) -> float:
        return sum(t.volume_ul for t in self.transfers)

    def to_frame(self) -> "pd.DataFrame":
        import pandas as pd
        if not self.transfers:
            return pd.DataFrame(columns=["item", "source_plate", "source_well",
                                         "dest_plate", "dest_well", "volume_ul"])
        return pd.DataFrame([{
            "item": t.item, "source_plate": t.source_plate,
            "source_well": t.source_well, "dest_plate": t.dest_plate,
            "dest_well": t.dest_well, "volume_ul": t.volume_ul,
        } for t in self.transfers])

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Worklist({self.reaction or 'transfers'}, "
                f"{len(self.transfers)} transfers, "
                f"{self.total_volume_ul:.1f} µL total)")


def dna_nm(concentration_ng_ul: float, length_bp: int) -> float:
    """Convert ng/µL to nM for double-stranded DNA of ``length_bp``.

    Equimolar mixing is molar, and stock concentrations are measured by mass —
    so this conversion is where "equal volumes of each fragment" quietly stops
    being equimolar as soon as the fragments differ in length.
    """
    if length_bp <= 0:
        raise ValueError("length_bp 必须为正。")
    # ng/µL = µg/mL; µg/mL ÷ (g/mol) × 1e6 = µM ... resolved to nM below
    return (float(concentration_ng_ul) * 1e6) / (_BP_MW * length_bp)


@register_function(
    aliases=["assembly_worklist", "组装工作单", "等摩尔混合", "assembly_setup",
             "反应配液", "worklist"],
    category="synthetic_biology",
    description="算出组装反应实际怎么配:按长度和浓度把片段换算成 nM 再取等摩尔体积,加上 master mix 与补水至终体积。等体积混合在片段长度不同时就已经不是等摩尔了,而这一步换算正是手工最容易算错的地方。Golden Gate / Gibson 皆可。Compute the actual pipetting for an assembly reaction.",
    examples=[
        "wl = ov.synbio.assembly_worklist({'backbone': (2700, 50), 'insert': (900, 30)})",
        "wl.to_frame(), wl.notes",
    ],
    related=["synbio.golden_gate", "synbio.gibson_assembly",
             "synbio.echo_picklist", "synbio.opentrons_protocol"],
    requires={},
    produces={},
)
def assembly_worklist(
    fragments: Mapping[str, Tuple[int, float]],
    *,
    method: str = "golden_gate",
    final_volume_ul: float = 20.0,
    backbone: Optional[str] = None,
    backbone_fmol: float = 20.0,
    insert_ratio: float = 2.0,
    master_mix_fraction: float = 0.5,
    min_volume_ul: float = 0.5,
    source_plate: str = "fragments",
    dest_plate: str = "reactions",
    dest_well: str = "A1",
    construct: str = "",
) -> Worklist:
    """Volumes for one assembly reaction.

    Parameters
    ----------
    fragments
        ``{name: (length_bp, concentration_ng_per_ul)}``.
    method
        ``'golden_gate'`` or ``'gibson'`` — changes only the master-mix note and
        the default insert:backbone ratio convention.
    final_volume_ul
        Reaction volume.
    backbone
        Which fragment is the vector. Defaults to the longest, which is the
        usual case and is stated in the notes so a wrong guess is visible.
    backbone_fmol
        Femtomoles of backbone per reaction.
    insert_ratio
        Molar excess of each insert over the backbone.
    master_mix_fraction
        Fraction of the final volume taken by enzyme/buffer master mix.
    min_volume_ul
        Smallest volume a hand pipette should be asked for. Anything below is
        reported in ``notes`` with the dilution needed, rather than emitted as
        an unpipettable number.

    Returns
    -------
    Worklist
    """
    _VALID = ("golden_gate", "gibson")
    if method not in _VALID:
        raise ValueError(f"method must be one of {list(_VALID)}, got {method!r}")
    if not fragments:
        raise ValueError("fragments 不能为空。")

    lengths = {k: int(v[0]) for k, v in fragments.items()}
    concs = {k: float(v[1]) for k, v in fragments.items()}
    nm = {k: dna_nm(concs[k], lengths[k]) for k in fragments}

    if backbone is None:
        backbone = max(lengths, key=lambda k: lengths[k])
    if backbone not in fragments:
        raise ValueError(f"backbone={backbone!r} 不在 fragments 里。")

    notes: List[str] = [
        f"{method}: master mix {master_mix_fraction:.0%} of {final_volume_ul} µL",
        f"backbone taken as {backbone!r} ({lengths[backbone]} bp) — the longest "
        f"fragment; pass backbone= if that is wrong",
    ]

    # fmol needed per fragment -> volume = fmol / (nM) since 1 nM = 1 fmol/µL
    transfers: List[Transfer] = []
    for name in fragments:
        fmol = backbone_fmol if name == backbone else backbone_fmol * insert_ratio
        if nm[name] <= 0:
            raise ValueError(f"片段 {name!r} 的浓度必须为正。")
        vol = fmol / nm[name]
        if vol < min_volume_ul:
            notes.append(
                f"{name}: 需要 {vol:.3f} µL,低于 {min_volume_ul} µL 的手工下限 — "
                f"把储液稀释 {min_volume_ul / vol:.0f}x 后再取 {min_volume_ul} µL")
        transfers.append(Transfer(
            source_plate=source_plate, source_well="", dest_plate=dest_plate,
            dest_well=dest_well, volume_ul=round(vol, 3), item=name,
            construct=construct))

    dna_volume = sum(t.volume_ul for t in transfers)
    mm_volume = final_volume_ul * master_mix_fraction
    water = final_volume_ul - dna_volume - mm_volume
    if water < 0:
        # Previously this only appended a note and skipped the else-branch, so
        # the master mix and the water were never emitted at all: the function
        # returned a runnable protocol for a ligation with no enzyme, no ligase
        # and no buffer. Refusing is the only safe answer.
        raise ValueError(
            f"DNA 体积 {dna_volume:.2f} µL + master mix {mm_volume:.2f} µL 超过了终"
            f"体积 {final_volume_ul} µL,配不出这个反应。把 final_volume_ul 提高到 "
            f"至少 {dna_volume + mm_volume:.1f} µL,或把储液稀释 "
            f"{dna_volume / max(final_volume_ul - mm_volume, 1e-9):.1f}x,"
            f"或降低 backbone_fmol。")
    transfers.append(Transfer(source_plate="reagents", source_well="",
                              dest_plate=dest_plate, dest_well=dest_well,
                              volume_ul=round(mm_volume, 3), construct=construct,
                              item=f"{method}_master_mix"))
    transfers.append(Transfer(source_plate="reagents", source_well="",
                              dest_plate=dest_plate, dest_well=dest_well,
                              volume_ul=round(water, 3), item="water",
                              construct=construct))

    return Worklist(transfers=transfers, reaction=method,
                    final_volume_ul=final_volume_ul, notes=notes)


# ---------------------------------------------------------------------------
# Echo pick list
# ---------------------------------------------------------------------------

#: The Echo Cherry Pick application's explicit pick-list columns, in order.
ECHO_COLUMNS = ("Source Plate Name", "Source Plate Type", "Source Well",
                "Destination Plate Name", "Destination Well",
                "Transfer Volume", "Sample Name")


@register_function(
    aliases=["echo_picklist", "Echo取样表", "picklist", "声波移液",
             "cherry_pick", "Labcyte"],
    category="synthetic_biology",
    description="生成 Beckman/Labcyte Echo 声波移液的 pick list(Echo Cherry Pick 读的显式列格式:Source Plate Name / Source Well / Destination Plate Name / Destination Well / Transfer Volume)。体积按 Echo 要求输出**纳升**并对齐到分辨率(2.5 nL),换算只在这里做一次,不留给读文件的人。Emit an Echo acoustic-transfer pick list.",
    examples=[
        "csv = ov.synbio.echo_picklist(worklist, source_plate=src, dest_plate=dst)",
        "ov.synbio.echo_picklist(worklist, out='transfer.csv')",
    ],
    related=["synbio.assembly_worklist", "synbio.plate_layout",
             "synbio.opentrons_protocol"],
    requires={},
    produces={},
)
def echo_picklist(
    worklist: "Worklist | Sequence[Transfer]",
    *,
    source_plate: Optional[Plate] = None,
    dest_plate: Optional[Plate] = None,
    source_plate_type: str = "384PP_AQ_BP",
    resolution_nl: float = 2.5,
    max_volume_nl: float = 2000.0,
    dead_volume_ul: float = 2.5,
    reagents_on_tips: bool = True,
    out: Optional[str] = None,
) -> str:
    """Render transfers as an Echo pick list, returned as CSV text.

    Parameters
    ----------
    worklist
        A :class:`Worklist` or a list of :class:`Transfer`.
    source_plate, dest_plate
        Used to resolve any transfer whose well is blank — the worklist from
        :func:`assembly_worklist` names items rather than wells, because which
        well an item sits in is a property of the plate, not of the reaction.
    source_plate_type
        Echo source-plate type string.
    resolution_nl
        Echo droplet resolution. Volumes are rounded to a multiple of this,
        because a volume the instrument cannot dispense is not a volume.
    max_volume_nl
        Transfers above this are split into several, since the Echo has a
        per-transfer ceiling.
    out
        Also write the CSV here.
    """
    transfers = (worklist.transfers if isinstance(worklist, Worklist)
                 else list(worklist))
    if not transfers:
        raise ValueError("没有要生成的转移步骤。")
    if resolution_nl <= 0:
        raise ValueError("resolution_nl 必须为正。")

    # Only DNA goes on an Echo. A Golden Gate master mix is a glycerol-containing
    # enzyme mix and falls outside every calibrated Echo fluid class, which is why
    # in practice the DNA is acoustically dispensed and the reagents are added by a
    # tip-based handler. Emitting them here also asked for wells of a "reagents"
    # plate that was never laid out.
    reagent_items: List[str] = []
    if reagents_on_tips:
        dna, held_back = [], []
        for t in transfers:
            item = (t.item or "").lower()
            if item == "water" or "master_mix" in item or "mastermix" in item:
                held_back.append(t)
            else:
                dna.append(t)
        if held_back:
            reagent_items = [t.item for t in held_back]
            transfers = dna

    lines = [",".join(ECHO_COLUMNS)]
    skipped: List[str] = []
    drawn: Dict[Tuple[str, str], float] = {}
    for t in transfers:
        src_well = t.source_well
        src_name = t.source_plate
        if not src_well and source_plate is not None:
            src_well = source_plate.well_of(t.item) or ""
            if src_well:
                # The well came from *this* plate, so it must be labelled with
                # this plate's name. Emitting `t.source_plate` here named a plate
                # ("reagents") that the well number did not come from, so the CSV
                # pointed the Echo at wells of a plate that was never laid out.
                src_name = source_plate.name
        if not src_well:
            skipped.append(t.item or "?")
            continue
        # Destination: prefer the plate map keyed by construct. `dest_well`
        # defaulted to "A1" for every transfer, so a plate whose A1 held the
        # `blank` control received every assembly.
        dst_well = ""
        if dest_plate is not None:
            dst_well = (dest_plate.well_of(t.construct) if t.construct else "") or ""
            if not dst_well and t.item:
                dst_well = dest_plate.well_of(t.item) or ""
        dst_well = dst_well or t.dest_well or "A1"

        remaining = t.volume_nl
        while remaining > 1e-9:
            chunk = min(remaining, max_volume_nl)
            snapped = round(chunk / resolution_nl) * resolution_nl
            if snapped <= 0:
                break
            lines.append(",".join([
                src_name, source_plate_type, src_well,
                t.dest_plate, dst_well, f"{snapped:.1f}", t.item,
            ]))
            # subtract what was actually emitted, not the un-snapped request,
            # or split transfers drift by a droplet per chunk
            remaining -= snapped
            drawn[(src_name, src_well)] = drawn.get((src_name, src_well), 0.0) + snapped

    # Source-volume feasibility. Plate.volumes_ul was populated by plate_layout
    # and read by nothing, so a pick list could ask for 10 µL out of a well that
    # holds 2 — or out of a well that holds nothing at all.
    if source_plate is not None and source_plate.volumes_ul:
        short = []
        for (plate_name, well), nl in sorted(drawn.items()):
            if plate_name != source_plate.name:
                continue
            available = float(source_plate.volumes_ul.get(well, 0.0))
            usable = max(available - dead_volume_ul, 0.0)
            if nl / 1000.0 > usable + 1e-9:
                short.append(f"{well} 需要 {nl / 1000.0:.2f} µL,可用 "
                             f"{usable:.2f} µL(装载 {available:.2f} − 死体积 "
                             f"{dead_volume_ul:.2f})")
        if short:
            raise ValueError(
                "源板体积不够,这份 pick list 跑不完:" + "; ".join(short)
                + "。提高 plate_layout(volumes_ul=...) 里的装载体积,或降低 "
                  "backbone_fmol/final_volume_ul。")

    if skipped:
        raise ValueError(
            f"这些转移没有源孔位,也无法从 source_plate 解析出来:{skipped}。"
            f"传 source_plate=(plate_layout 的结果),或在 Transfer 上直接给 "
            f"source_well。")

    if reagent_items:
        # Warn rather than writing a comment row: the CSV has to contain exactly
        # what the Echo software parses, and nothing else.
        import warnings
        warnings.warn(
            f"以下试剂未包含在这份 Echo pick list 里,请用枪头式移液器加:"
            f"{sorted(set(reagent_items))} —— 含酶/甘油的 master mix 不在 Echo 的"
            f"已校准流体类别内。传 reagents_on_tips=False 可强制包含。",
            stacklevel=2)

    text = "\n".join(lines) + "\n"
    if out:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text


# ---------------------------------------------------------------------------
# Opentrons protocol
# ---------------------------------------------------------------------------

#: ``pipette load name -> (min µL, max µL, tiprack load name)``.
#: Keyed explicitly because substring matching on the pipette name put OT-2
#: tipracks on a Flex deck and reported a 20 µL minimum for a Flex 1000 (whose
#: real minimum is 5 µL), producing five false warnings and a protocol that
#: cannot pass Flex analysis.
_PIPETTE_SPECS: Dict[str, Tuple[float, float, str]] = {
    "p20_single_gen2": (1.0, 20.0, "opentrons_96_tiprack_20ul"),
    "p300_single_gen2": (20.0, 300.0, "opentrons_96_tiprack_300ul"),
    "p1000_single_gen2": (100.0, 1000.0, "opentrons_96_tiprack_1000ul"),
    "p20_multi_gen2": (1.0, 20.0, "opentrons_96_tiprack_20ul"),
    "p300_multi_gen2": (20.0, 300.0, "opentrons_96_tiprack_300ul"),
    "flex_1channel_50": (1.0, 50.0, "opentrons_flex_96_tiprack_50ul"),
    "flex_1channel_1000": (5.0, 1000.0, "opentrons_flex_96_tiprack_1000ul"),
    "flex_8channel_50": (1.0, 50.0, "opentrons_flex_96_tiprack_50ul"),
    "flex_8channel_1000": (5.0, 1000.0, "opentrons_flex_96_tiprack_1000ul"),
}

@register_function(
    aliases=["opentrons_protocol", "Opentrons协议", "液体处理协议", "OT2",
             "自动化协议", "liquid_handling"],
    category="synthetic_biology",
    description="把转移步骤生成为可直接运行的 Opentrons Python Protocol API v2 协议脚本(metadata / requirements / run(protocol) 三段结构,load_labware / load_instrument / transfer 调用)。体积按 Opentrons 要求用**微升**,并按移液器量程自动挑选与拆分。Generate a runnable Opentrons Python Protocol API v2 script.",
    examples=[
        "script = ov.synbio.opentrons_protocol(worklist, source_plate=src, dest_plate=dst)",
        "ov.synbio.opentrons_protocol(worklist, out='protocol.py', robot='Flex')",
    ],
    related=["synbio.assembly_worklist", "synbio.echo_picklist",
             "synbio.plate_layout"],
    requires={},
    produces={},
)
def opentrons_protocol(
    worklist: "Worklist | Sequence[Transfer]",
    *,
    source_plate: Optional[Plate] = None,
    dest_plate: Optional[Plate] = None,
    robot: str = "OT-2",
    api_level: str = "2.16",
    protocol_name: str = "ov.synbio assembly",
    author: str = "omicverse",
    source_labware: str = "biorad_96_wellplate_200ul_pcr",
    dest_labware: str = "biorad_96_wellplate_200ul_pcr",
    source_slot: int = 1,
    dest_slot: int = 2,
    tiprack_slot: int = 3,
    reagent_labware: str = "nest_12_reservoir_15ml",
    reagent_slot: int = 4,
    pipette: Optional[str] = None,
    mount: str = "right",
    new_tip: str = "always",
    out: Optional[str] = None,
) -> str:
    """Render transfers as an Opentrons protocol script.

    Parameters
    ----------
    robot
        ``'OT-2'`` or ``'Flex'`` — goes into ``requirements['robotType']``.
    api_level
        ``requirements['apiLevel']``.
    pipette
        Instrument load name. Chosen from the transfer volumes when omitted: a
        p20 for sub-20 µL work and a p300 otherwise, because asking a p300 for
        0.5 µL is the most common way an automated protocol quietly fails.
    new_tip
        ``'always'`` | ``'once'`` | ``'never'``. Defaults to ``'always'``, which
        is the only safe choice when transferring different DNA fragments.
    out
        Also write the script here.
    """
    if robot not in ("OT-2", "Flex"):
        raise ValueError(f"robot must be 'OT-2' or 'Flex', got {robot!r}")
    if new_tip not in ("always", "once", "never"):
        raise ValueError(
            f"new_tip must be one of ['always', 'once', 'never'], got {new_tip!r}")

    transfers = (worklist.transfers if isinstance(worklist, Worklist)
                 else list(worklist))
    if not transfers:
        raise ValueError("没有要生成的转移步骤。")

    volumes = [t.volume_ul for t in transfers if t.volume_ul > 0]
    if not volumes:
        raise ValueError("所有转移体积都是 0。")

    if pipette is None:
        if max(volumes) <= 20.0:
            pipette = "p20_single_gen2" if robot == "OT-2" else "flex_1channel_50"
            tiprack = ("opentrons_96_tiprack_20ul" if robot == "OT-2"
                       else "opentrons_flex_96_tiprack_50ul")
        else:
            pipette = "p300_single_gen2" if robot == "OT-2" else "flex_1channel_1000"
            tiprack = ("opentrons_96_tiprack_300ul" if robot == "OT-2"
                       else "opentrons_flex_96_tiprack_1000ul")
    else:
        tiprack = _PIPETTE_SPECS.get(pipette, _PIPETTE_SPECS["p300_single_gen2"])[2]

    min_vol = _PIPETTE_SPECS.get(pipette, _PIPETTE_SPECS["p300_single_gen2"])[0]
    warnings: List[str] = []

    # Reagents before DNA. A sub-microlitre DNA transfer dispensed into a dry
    # well stays on the wall; water and master mix go in first so there is a bulk
    # to dispense into. The emitted order used to be exactly the reverse.
    def _rank(t):
        item = (t.item or "").lower()
        if item == "water":
            return 0
        if "master_mix" in item or "mastermix" in item:
            return 1
        return 2
    ordered = sorted(transfers, key=_rank)

    # Reagents live in a reservoir, not on the DNA source plate. Without this
    # they had no source well at all and the protocol could not be generated
    # (or, before the destination fix, was generated without them entirely).
    reagent_wells: Dict[str, str] = {}
    for t in ordered:
        item = (t.item or "").lower()
        if (item == "water" or "master_mix" in item or "mastermix" in item) \
                and t.item not in reagent_wells:
            reagent_wells[t.item] = well_name(0, len(reagent_wells))
    uses_reagents = bool(reagent_wells)

    body: List[str] = []
    for t in ordered:
        if t.volume_ul <= 0:
            continue
        from_reservoir = t.item in reagent_wells
        if from_reservoir:
            src_well = reagent_wells[t.item]
            src_label = "reagents"
        else:
            src_well = t.source_well or (source_plate.well_of(t.item)
                                         if source_plate else None)
            src_label = "source"
        # Resolve the destination from the plate map, keyed by construct. A
        # hardcoded "A1" fallback meant every assembly landed in whatever A1
        # held — on a plate laid out with controls first, that is the `blank`.
        dst_well = ""
        if dest_plate is not None:
            dst_well = (dest_plate.well_of(t.construct) if t.construct else "") or ""
            if not dst_well and t.item:
                dst_well = dest_plate.well_of(t.item) or ""
        dst_well = dst_well or t.dest_well or "A1"
        if not src_well:
            raise ValueError(
                f"转移 {t.item!r} 没有源孔位。传 source_plate=,或在 Transfer 上"
                f"给 source_well。")
        if t.volume_ul < min_vol:
            # protocol.comment() reaches the run log and the Opentrons app; a
            # Python comment reaches neither, so the operator never saw it.
            warnings.append(
                f'    protocol.comment("WARNING {t.item}: {t.volume_ul} µL is '
                f'below the {pipette} minimum ({min_vol} µL) — dilute the stock")')
        body.append(
            f'    pipette.transfer({t.volume_ul}, {src_label}["{src_well}"], '
            f'dest["{dst_well}"], new_tip="{new_tip}")'
            + (f'   # {t.item}' if t.item else ""))

    # mix the completed reaction — a ligation has to be homogeneous
    mix_wells = sorted({(dest_plate.well_of(t.construct) if (dest_plate and t.construct)
                         else t.dest_well) or "A1" for t in ordered})
    for well in mix_wells:
        body.append(f'    pipette.pick_up_tip()')
        body.append(f'    pipette.mix(3, {max(min_vol, 5.0)}, dest["{well}"])')
        body.append(f'    pipette.drop_tip()')

    slot = (lambda s: f'"{chr(ord("A") + (s - 1) // 3)}{((s - 1) % 3) + 1}"'
            if robot == "Flex" else str(s))

    script = "\n".join([
        '"""Generated by ov.synbio.opentrons_protocol — review before running."""',
        "from opentrons import protocol_api",
        "",
        "metadata = {",
        f'    "protocolName": "{protocol_name}",',
        f'    "author": "{author}",',
        f'    "description": "{len(body)} transfers generated from an '
        f'ov.synbio worklist",',
        "}",
        "",
        "requirements = {",
        f'    "robotType": "{robot}",',
        f'    "apiLevel": "{api_level}",',
        "}",
        "",
        "",
        "def run(protocol: protocol_api.ProtocolContext) -> None:",
        f'    source = protocol.load_labware("{source_labware}", {slot(source_slot)})',
        f'    dest = protocol.load_labware("{dest_labware}", {slot(dest_slot)})',
        f'    tips = protocol.load_labware("{tiprack}", {slot(tiprack_slot)})',
        *([f'    reagents = protocol.load_labware("{reagent_labware}", '
           f'{slot(reagent_slot)})'] if uses_reagents else []),
        f'    pipette = protocol.load_instrument("{pipette}", "{mount}", '
        f'tip_racks=[tips])',
        # A Flex has no fixed trash from apiLevel 2.16, so a protocol without an
        # explicit trash bin fails analysis before it ever reaches the deck.
        *(['    trash = protocol.load_trash_bin("A3")'] if robot == "Flex" else []),
        "",
        *warnings,
        *body,
        "",
    ])

    if out:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(script)
    return script


# ---------------------------------------------------------------------------
# PCR program
# ---------------------------------------------------------------------------

@dataclass
class ThermocyclerProgram:
    """A thermocycler program."""

    steps: List[Tuple[str, float, float]] = field(default_factory=list)
    cycles: int = 30
    annealing_C: float = 55.0
    extension_s: float = 30.0
    polymerase: str = "q5"
    notes: List[str] = field(default_factory=list)
    #: Heated-lid setpoint. A block programme without one condenses on the cap.
    lid_C: float = 105.0
    #: Post-run hold temperature.
    hold_C: float = 4.0

    #: Steps that run once per programme rather than once per cycle.
    ONCE_ONLY = ("initial", "final_extension", "hold")

    def to_text(self) -> str:
        """The programme as a block-format listing a thermocycler can be set from.

        Every other Build-layer emitter produces something an instrument reads;
        this one returned a DataFrame and nothing else, so "a thermocycler program
        from the primers" stopped at a table a human had to retype.
        """
        lines = [f"# {self.polymerase} — lid {self.lid_C:.0f} C, "
                 f"{self.cycles} cycles, {self.total_minutes:.1f} min total"]
        for name, temp, seconds in self.steps:
            if name == "hold":
                lines.append(f"HOLD           {temp:5.1f} C  infinite")
                continue
            tag = "1x " if name in self.ONCE_ONLY else f"{self.cycles}x"
            lines.append(f"{tag} {name:<14s} {temp:5.1f} C  {seconds:6.1f} s")
        for note in self.notes:
            lines.append(f"# {note}")
        return "\n".join(lines) + "\n"

    @property
    def total_minutes(self) -> float:
        """Wall-clock runtime of the programme, in minutes.

        The obvious spelling of this is a trap: ``sum(s for _, _, s in
        self.steps if _ != "initial")`` rebinds ``_`` to the *temperature* while
        unpacking, so the condition compares a float against a string, is always
        true, and folds the initial denaturation and the final extension into
        every cycle. That reported 95.5 minutes for a 22.5-minute programme.
        """
        once = set(self.ONCE_ONLY)
        per_cycle = sum(seconds for name, _temp, seconds in self.steps
                        if name not in once)
        one_off = sum(seconds for name, _temp, seconds in self.steps
                      if name in once)
        return (one_off + self.cycles * per_cycle) / 60.0

    def to_frame(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame(self.steps,
                            columns=["step", "temperature_C", "seconds"])

    def __repr__(self) -> str:  # pragma: no cover
        return (f"ThermocyclerProgram({self.polymerase}, {self.cycles} cycles, "
                f"Ta={self.annealing_C:.1f} °C, "
                f"extension {self.extension_s:.0f} s)")


@register_function(
    aliases=["pcr_protocol", "PCR程序", "热循环程序", "thermocycler",
             "退火温度", "annealing_temperature"],
    category="synthetic_biology",
    description="由引物 Tm 和产物长度算出热循环程序:退火温度取两条引物 Tm 的较低者加固定偏移(Q5 用 +3 °C、Taq 用 -5 °C 的常规),延伸时间按聚合酶速度和产物长度算,而不是沿用谁忘了改的默认值。Derive a thermocycler program from primer Tm and amplicon length.",
    examples=[
        "prog = ov.synbio.pcr_protocol(tm_forward=62.1, tm_reverse=60.8, amplicon_bp=1200)",
        "prog.annealing_C, prog.extension_s, prog.to_frame()",
    ],
    related=["synbio.design_primers", "synbio.assembly_worklist"],
    requires={},
    produces={},
)
def pcr_protocol(
    *,
    tm_forward: float,
    tm_reverse: float,
    amplicon_bp: int,
    polymerase: str = "q5",
    cycles: int = 30,
    final_extension_s: float = 120.0,
    annealing_s: float = 20.0,
    lid_C: float = 105.0,
    hold_C: float = 4.0,
) -> ThermocyclerProgram:
    """A thermocycler program for one amplicon.

    ``polymerase`` selects the vendor conventions that actually differ between
    enzymes — denaturation temperature, extension rate, and how annealing
    relates to primer Tm:

    ``'q5'``
        98 °C denaturation, ~20–30 s/kb, anneal at the *lower* primer Tm **+3 °C**.
    ``'phusion'``
        98 °C, ~15–30 s/kb, anneal at the lower Tm + 3 °C.
    ``'taq'``
        95 °C, ~60 s/kb, anneal at the lower Tm − 5 °C.

    Getting this from the primers rather than from a template is the point: a
    default annealing temperature carried over from another reaction is one of
    the most common reasons a designed primer pair does not amplify.
    """
    _POL = {
        "q5": {"denature_C": 98.0, "initial_s": 30.0, "denature_s": 10.0,
               "s_per_kb": 25.0, "ta_offset": 3.0, "extend_C": 72.0},
        "phusion": {"denature_C": 98.0, "initial_s": 30.0, "denature_s": 10.0,
                    "s_per_kb": 20.0, "ta_offset": 3.0, "extend_C": 72.0},
        "taq": {"denature_C": 95.0, "initial_s": 120.0, "denature_s": 30.0,
                "s_per_kb": 60.0, "ta_offset": -5.0, "extend_C": 72.0},
    }
    if polymerase not in _POL:
        raise ValueError(
            f"polymerase must be one of {sorted(_POL)}, got {polymerase!r}")
    if amplicon_bp <= 0:
        raise ValueError("amplicon_bp 必须为正。")
    if cycles < 1:
        raise ValueError("cycles 必须 >= 1")

    p = _POL[polymerase]
    lower_tm = min(float(tm_forward), float(tm_reverse))
    ta_raw = lower_tm + p["ta_offset"]
    ta = max(45.0, min(72.0, ta_raw))
    clamped = abs(ta - ta_raw) > 1e-9

    extension = max(10.0, p["s_per_kb"] * amplicon_bp / 1000.0)

    notes = [
        # Report the arithmetic *and* the clamp. Printing the pre-clamp equation
        # with the post-clamp answer produced statements like
        # "min(Tm) 80.0 °C +3 °C = 72.0 °C".
        (f"annealing = min(Tm) {lower_tm:.1f} °C {p['ta_offset']:+g} °C = "
         f"{ta_raw:.1f} °C, clamped to {ta:.1f} °C ({polymerase} convention)"
         if clamped else
         f"annealing = min(Tm) {lower_tm:.1f} °C {p['ta_offset']:+g} °C = "
         f"{ta:.1f} °C ({polymerase} convention)"),
        f"extension = {p['s_per_kb']:g} s/kb x {amplicon_bp / 1000:.2f} kb = "
        f"{extension:.0f} s",
    ]
    if abs(float(tm_forward) - float(tm_reverse)) > 5.0:
        notes.append(
            f"两条引物 Tm 差 {abs(tm_forward - tm_reverse):.1f} °C(> 5 °C)— "
            f"退火只能照顾较低的那条,另一条会结合过强;建议重设引物。")

    if clamped and ta_raw > 72.0:
        notes.append(
            f"退火温度被 72 °C 上限截断。Tm 这么高时通常改用两步 PCR"
            f"(变性 + 72 °C 退火/延伸合并),而不是硬压退火温度。")

    steps = [
        ("initial", p["denature_C"], p["initial_s"]),
        ("denature", p["denature_C"], p["denature_s"]),
        ("anneal", ta, annealing_s),
        ("extend", p["extend_C"], extension),
        ("final_extension", p["extend_C"], final_extension_s),
        ("hold", hold_C, 0.0),
    ]
    return ThermocyclerProgram(
        steps=steps, cycles=cycles, annealing_C=ta, extension_s=extension,
        polymerase=polymerase, notes=notes, lid_C=lid_C, hold_C=hold_C)


# ---------------------------------------------------------------------------
# visualisation
# ---------------------------------------------------------------------------

@register_function(
    aliases=["plot_plate", "板图", "孔位图", "plate_map_plot", "微孔板图"],
    category="synthetic_biology",
    description="画微孔板图:每个孔标注内容,空孔留白,对照孔高亮。picklist 不可复核,板图可以 —— 这是把自动化输出交给人检查的那一步。Plot a plate map.",
    examples=[
        "ov.synbio.plot_plate(plate)",
        "ov.synbio.plot_plate(plate, highlight=['blank', 'wt'])",
    ],
    related=["synbio.plate_layout", "synbio.echo_picklist"],
    requires={},
    produces={},
)
def plot_plate(plate: Plate, highlight: Optional[Sequence[str]] = None, ax=None):
    """Plate map with well contents labelled."""
    from ._plot import _mpl
    plt = _mpl()

    rows, cols = plate.shape
    highlight = set(highlight or [])

    if ax is None:
        fig, ax = plt.subplots(figsize=(min(18, 0.62 * cols + 2),
                                        0.62 * rows + 1.6))
    else:
        fig = ax.figure

    for r in range(rows):
        for c in range(cols):
            well = well_name(r, c)
            item = plate.contents.get(well, "")
            if not item:
                colour, edge = "#FFFFFF", "#DDDDDD"
            elif item in highlight or item.split("_r")[0] in highlight:
                colour, edge = "#FDB462", "#B36A00"
            else:
                colour, edge = "#B3DE69", "#5C8A1F"
            ax.add_patch(plt.Circle((c, -r), 0.42, facecolor=colour,
                                    edgecolor=edge, lw=1.0))
            if item and rows * cols <= 96:
                ax.text(c, -r, item[:6], ha="center", va="center", fontsize=5.5)

    ax.set_xlim(-0.7, cols - 0.3)
    ax.set_ylim(-rows + 0.3, 0.7)
    ax.set_xticks(range(cols))
    ax.set_xticklabels([str(c + 1) for c in range(cols)], fontsize=7)
    ax.set_yticks([-r for r in range(rows)])
    ax.set_yticklabels([_row_label(r) for r in range(rows)], fontsize=7)
    ax.set_aspect("equal")
    ax.set_title(f"{plate.name} ({plate.format}-well, "
                 f"{plate.n_used}/{plate.capacity} filled)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig, ax


__all__ = [
    "plate_layout", "Plate", "PLATE_FORMATS", "well_name", "parse_well",
    "assembly_worklist", "Worklist", "Transfer", "dna_nm",
    "echo_picklist", "ECHO_COLUMNS", "opentrons_protocol",
    "pcr_protocol", "ThermocyclerProgram", "plot_plate",
]
