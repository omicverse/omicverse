r"""Real public flow-cytometry datasets for the ``ov.flow`` tutorials.

The simulated sample in :mod:`omicverse.datasets._flow` teaches the mechanics —
it can show what spillover *did*, because it applied it. These are real
experiments, and the point of them is that nothing is arranged for the reader's
convenience.

Both are downloaded **from their original repositories**, not re-hosted here.
Attribution is required by both licences and is printed by the loaders.

WHY TWO, AND WHY NEITHER ONE ALONE
    No openly-licensed cytometry dataset carries a full CD3/CD4/CD8/CD19
    hierarchy *and* a real spillover matrix *and* the FSC-H needed for a singlet
    gate. Every candidate has two of the three, so the tutorials use one of each
    and say which lesson each supports:

    :func:`flow_pbmc_fortessa`  BD LSRFortessa, conventional fluorescence. A
        genuine 13x13 spillover matrix with 93 non-zero off-diagonal terms, and
        the data are UNCOMPENSATED — so compensation is a real step with a
        visible effect. 32 files share a byte-identical panel, which is what
        makes "one gating strategy, a whole batch" demonstrable. No CD19, and no
        FSC-H, so no doublet gate.

    :func:`flow_pbmc_spectral`  Cytek Aurora, full-spectrum. Carries FSC-A and
        FSC-H, so the singlet gate is real, and the full CD3 -> CD4/CD8 + CD19 +
        CD14 + CD16/CD56 hierarchy. Its ``$SPILLOVER`` is the 9x9 identity —
        the instrument unmixes spectrally at acquisition, so there is nothing to
        compensate. That is not a defect of the dataset; it is what a spectral
        instrument does, and a tutorial that quietly ran ``compensate()`` on it
        anyway would be teaching a superstition.
"""
from __future__ import annotations

import os
import zipfile
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._registry import register_function
from ._datasets import download_data, download_data_requests

if TYPE_CHECKING:  # pragma: no cover
    from anndata import AnnData

__all__ = ["flow_pbmc_fortessa", "flow_pbmc_spectral"]

# ── LSRFortessa: Kim 2024, Zenodo, CC-BY-4.0 ────────────────────────────────

_FORTESSA_RECORD = "14311616"
_FORTESSA_URL = f"https://zenodo.org/api/records/{_FORTESSA_RECORD}/files/{{}}/content"
_FORTESSA_DOI = "10.5281/zenodo.14311616"

#: (filename, group). The isotype-control arm only, so a group comparison is
#: disease versus healthy rather than a drug effect, and the smallest files in
#: each arm, because this downloads over the network at tutorial time.
_FORTESSA_FILES: Tuple[Tuple[str, str], ...] = (
    ("96_Healthy_Anti-Nectin2_Isotype_3.fcs", "Healthy"),
    ("108_CA_Anti-Nectin2_Isotype_3.fcs", "CardiacArrest"),
    ("97_Healthy_Anti-Nectin2_Isotype_4.fcs", "Healthy"),
    ("110_CA_Anti-Nectin2_Isotype_5.fcs", "CardiacArrest"),
    ("94_Healthy_Anti-Nectin2_Isotype_1.fcs", "Healthy"),
    ("106_CA_Anti-Nectin2_Isotype_1.fcs", "CardiacArrest"),
)

_FORTESSA_CITE = (
    "Kim, Edy (2024). Immunophenotyping of peripheral blood mononuclear cells "
    "after out-of-hospital cardiac arrest. Zenodo. "
    f"https://doi.org/{_FORTESSA_DOI} (CC-BY-4.0)"
)

# ── Cytek Aurora: Gootjes et al., PLOS ONE, CC-BY-4.0 ───────────────────────

_SPECTRAL_DOI = "10.1371/journal.pone.0351131"
_SPECTRAL_URL = (
    "https://journals.plos.org/plosone/article/file"
    f"?id={_SPECTRAL_DOI}.s{{:03d}}&type=supplementary"
)
#: Supporting-information item per donor; S1 is not flow data.
_SPECTRAL_ITEMS: Tuple[int, ...] = (2, 3, 4, 5, 6, 7)

_SPECTRAL_CITE = (
    "Gootjes C, Zwaginga JJ, Nikolic T, Roep BO (2026). Limited efficacy of a "
    "therapeutic anti-CD40 monoclonal antibody to inhibit activated CD4 T cell "
    "autoimmunity in vitro. PLOS ONE. "
    f"https://doi.org/{_SPECTRAL_DOI} (CC-BY-4.0)"
)


def _subsample(adata, max_events: Optional[int], seed: int):
    """Thin one file to ``max_events``. Real files run to 1.5 million events;
    a tutorial does not need them and a notebook cannot carry them."""
    if max_events is None or adata.n_obs <= max_events:
        return adata
    rng = np.random.default_rng(seed)
    keep = rng.choice(adata.n_obs, size=max_events, replace=False)
    keep.sort()          # keep acquisition order, so a Time gate still means something
    return adata[keep].copy()


def _concat(parts, names, extra):
    import anndata as ad

    for a, name in zip(parts, names):
        a.obs["sample"] = name
        for key, value in extra.get(name, {}).items():
            a.obs[key] = value
    out = ad.concat(parts, join="outer", index_unique="-", label=None)
    out.obs["sample"] = pd.Categorical(out.obs["sample"])
    # anndata.concat drops uns; the spillover matrix is the point of the file.
    out.uns["fcs"] = dict(parts[0].uns.get("fcs", {}))
    out.var = parts[0].var.copy()
    return out


@register_function(
    aliases=[
        "flow_pbmc_fortessa", "flow_real_fortessa", "真实流式数据",
        "外周血流式", "cardiac_arrest_pbmc_flow", "lsrfortessa_pbmc",
    ],
    category="datasets",
    description=(
        "REAL conventional flow-cytometry PBMC data on a BD LSRFortessa — "
        "Kim (2024), Zenodo 14311616, CC-BY-4.0, downloaded from the original "
        "repository. Immunophenotyping after out-of-hospital cardiac arrest: "
        "healthy versus cardiac-arrest donors, isotype-control arm. 16 "
        "parameters (FSC-A, SSC-A, CD3/CD4/CD8/CD14/CD16/CD56, LIVE-DEAD, "
        "Tim3/TIGIT/Nectin2, IFNg/TNFa/IL10). Carries a genuine 13x13 "
        "$SPILLOVER matrix (93 non-zero off-diagonal terms) and is "
        "UNCOMPENSATED, so ov.flow.compensate does real work. All files share "
        "an identical panel, so one gating strategy applies to the batch. No "
        "CD19 and no FSC-H (so no doublet gate) — use flow_pbmc_spectral for "
        "those."
    ),
    examples=[
        "adata = ov.datasets.flow_pbmc_fortessa()",
        "adata = ov.datasets.flow_pbmc_fortessa(n_per_group=3, max_events=50000)",
    ],
    related=["datasets.flow_pbmc_spectral", "datasets.flow_demo",
             "flow.compensate", "io.read_fcs"],
)
def flow_pbmc_fortessa(
    dir: str = "./data",
    *,
    n_per_group: int = 2,
    max_events: Optional[int] = 30_000,
    seed: int = 0,
    verbose: bool = True,
) -> "AnnData":
    """Real uncompensated PBMC flow data, healthy versus post-cardiac-arrest.

    Arguments
    ---------
    n_per_group
        Files per group (max 3 each). Each file is 6-25 MB and is downloaded
        from Zenodo on first use, then cached in ``dir``.
    max_events
        Events kept per file, sampled without replacement in acquisition order.
        ``None`` keeps everything.

    Returns one AnnData with ``obs['sample']`` and ``obs['group']``, and the
    acquisition spillover matrix in ``uns['fcs']['spillover']``.
    """
    from .. import io as _io

    if not 1 <= n_per_group <= 3:
        raise ValueError("n_per_group must be between 1 and 3")

    chosen = []
    for group in ("Healthy", "CardiacArrest"):
        in_group = [f for f, g in _FORTESSA_FILES if g == group]
        chosen += [(f, group) for f in in_group[:n_per_group]]

    parts, names, extra = [], [], {}
    for filename, group in chosen:
        path = download_data(_FORTESSA_URL.format(filename), file_path=filename, dir=dir)
        a = _io.read_fcs(path)
        a = _subsample(a, max_events, seed)
        name = filename.split("_")[0] + "_" + group
        parts.append(a)
        names.append(name)
        extra[name] = {"group": group, "source_file": filename}

    out = _concat(parts, names, extra)
    out.obs["group"] = pd.Categorical(out.obs["group"])
    out.uns["dataset"] = {"name": "flow_pbmc_fortessa", "doi": _FORTESSA_DOI,
                          "license": "CC-BY-4.0", "citation": _FORTESSA_CITE,
                          "compensated": False}
    if verbose:
        print(f"{out.n_obs:,} events x {out.n_vars} channels from "
              f"{len(parts)} files\n{_FORTESSA_CITE}")
    return out


@register_function(
    aliases=[
        "flow_pbmc_spectral", "flow_real_spectral", "光谱流式数据",
        "cytek_aurora_pbmc", "spectral_flow_pbmc",
    ],
    category="datasets",
    description=(
        "REAL full-spectrum flow-cytometry PBMC data on a Cytek Aurora — "
        "Gootjes et al. (2026) PLOS ONE, CC-BY-4.0, downloaded from the "
        "publisher's supporting information. 16 parameters including FSC-A AND "
        "FSC-H (so a real singlet gate) and the complete CD3 / CD4 / CD8 / "
        "CD19 / CD14 / CD16 / CD56 / CD15 / CD40 immunophenotyping hierarchy. "
        "Its $SPILLOVER is the 9x9 identity because the instrument unmixes "
        "spectrally at acquisition — there is nothing to compensate, which is "
        "itself worth knowing. Use flow_pbmc_fortessa for the compensation "
        "lesson."
    ),
    examples=[
        "adata = ov.datasets.flow_pbmc_spectral()",
        "adata = ov.datasets.flow_pbmc_spectral(n_donors=3)",
    ],
    related=["datasets.flow_pbmc_fortessa", "datasets.flow_demo", "io.read_fcs"],
)
def flow_pbmc_spectral(
    dir: str = "./data",
    *,
    n_donors: int = 1,
    max_events: Optional[int] = 60_000,
    seed: int = 0,
    verbose: bool = True,
) -> "AnnData":
    """Real spectral PBMC flow data with the full lineage panel.

    Arguments
    ---------
    n_donors
        Donors to load (max 6). Each is an ~83 MB zip holding one FCS of ~1.5
        million events, downloaded on first use and cached in ``dir``.
    max_events
        Events kept per donor.

    Returns one AnnData with ``obs['sample']``. ``uns['fcs']['spillover']`` is
    present but is the identity — see the module docstring.
    """
    from .. import io as _io

    if not 1 <= n_donors <= len(_SPECTRAL_ITEMS):
        raise ValueError(f"n_donors must be between 1 and {len(_SPECTRAL_ITEMS)}")

    parts, names, extra = [], [], {}
    for i, item in enumerate(_SPECTRAL_ITEMS[:n_donors], start=1):
        zip_name = f"pone.0351131.s{item:03d}.zip"
        # PLOS rejects the default urllib agent, so this one needs the
        # header-setting downloader rather than download_data.
        zip_path = download_data_requests(
            _SPECTRAL_URL.format(item), file_path=zip_name, dir=dir
        )
        with zipfile.ZipFile(zip_path) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith(".fcs")]
            if not members:
                raise RuntimeError(f"{zip_name} contains no .fcs file: {zf.namelist()}")
            target = os.path.join(dir, os.path.basename(members[0]))
            if not os.path.exists(target):
                with zf.open(members[0]) as src, open(target, "wb") as dst:
                    dst.write(src.read())

        a = _io.read_fcs(target)
        a = _subsample(a, max_events, seed)
        name = f"Donor {i}"
        parts.append(a)
        names.append(name)
        extra[name] = {"source_file": os.path.basename(target)}

    out = _concat(parts, names, extra)
    out.uns["dataset"] = {"name": "flow_pbmc_spectral", "doi": _SPECTRAL_DOI,
                          "license": "CC-BY-4.0", "citation": _SPECTRAL_CITE,
                          "spectrally_unmixed": True}
    if verbose:
        print(f"{out.n_obs:,} events x {out.n_vars} channels from "
              f"{len(parts)} donor(s)\n{_SPECTRAL_CITE}")
    return out
