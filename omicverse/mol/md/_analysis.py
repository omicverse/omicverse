r"""Trajectory analysis for :mod:`omicverse.mol.md`, built on MDTraj.

Every function accepts an :class:`~omicverse.mol.md.MDTrajectory`, an
``mdtraj.Trajectory``, or a path to a trajectory file.

.. important::
   On a periodic (explicit-solvent) system, molecules that cross a box
   boundary are wrapped back by the integrator, which makes distances jump.
   These helpers therefore **image molecules automatically** before analysing
   (equivalent to :meth:`MDTrajectory.wrap`). Pass ``wrap=False`` if you have
   already imaged the trajectory yourself.

   This applies to the inputs the helpers load themselves — an
   :class:`~omicverse.mol.md.MDTrajectory`, or a path plus ``top=``. An
   already-loaded ``mdtraj.Trajectory`` is used exactly as handed over and is
   **never** imaged, whatever ``wrap`` says: the caller has clearly taken the
   trajectory in hand, and re-imaging it would silently undo their own
   treatment. Call :func:`load_trajectory` (or ``traj.image_molecules()``)
   before passing one in.
"""

from __future__ import annotations

import itertools
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from ..._registry import register_function
from ._common import check_mdtraj
from ._types import MDTrajectory

_MAX_PAIRS = 5_000_000       # guard against an accidental O(N^2) explosion


def _image(traj):
    """Image molecules into the primary cell, anchored on the protein."""
    if traj.unitcell_vectors is None:
        return traj
    try:
        protein = traj.top.select("protein")
        anchors = ([set(traj.top.atom(int(i)) for i in protein)]
                   if len(protein) else None)
        return traj.image_molecules(anchor_molecules=anchors)
    except Exception:
        try:
            return traj.image_molecules()
        except Exception:
            return traj          # non-fatal: analysis still runs unwrapped


def _as_traj(traj, top=None, stride: int = 1, wrap: bool = True):
    """Normalise any accepted trajectory input to an ``mdtraj.Trajectory``."""
    check_mdtraj()
    import mdtraj as md

    if isinstance(traj, MDTrajectory):
        t = traj.load(stride=stride, top=top)
        already = bool(traj.provenance.get("wrapped"))
        return _image(t) if (wrap and not already) else t
    if isinstance(traj, md.Trajectory):
        # `wrap` is deliberately not honoured here: an in-memory Trajectory has
        # no provenance saying whether it was already imaged, and re-imaging an
        # imaged trajectory can shift molecules again. The caller holding the
        # object is the one who knows — documented in the module docstring.
        t = traj[::stride] if stride > 1 else traj
        return t
    if isinstance(traj, str):
        if top is None:
            raise ValueError(
                "loading a trajectory from a path needs `top=` (the topology "
                "PDB written next to it), or pass the MDTrajectory object")
        t = md.load(traj, top=top, stride=stride)
        return _image(t) if wrap else t
    raise TypeError(
        "traj must be an MDTrajectory, an mdtraj.Trajectory, or a path — "
        f"got {type(traj).__name__}")


def _select(traj, selection: str) -> np.ndarray:
    """Resolve an MDTraj selection string, with a helpful error when empty."""
    idx = traj.top.select(selection)
    if len(idx) == 0:
        raise ValueError(
            f"selection {selection!r} matched no atoms. Check the topology "
            "(e.g. 'protein and name CA', 'resname LIG', 'backbone').")
    return idx


@register_function(
    aliases=["load_trajectory", "载入轨迹", "md_load", "read_trajectory"],
    category="mol",
    description=(
        "Load an MD trajectory into MDTraj, imaging molecules back into the "
        "periodic box so distance-based analysis is not corrupted by "
        "periodic-boundary jumps. Accepts an MDTrajectory, an "
        "mdtraj.Trajectory or a path plus topology."
    ),
    examples=["t = ov.mol.load_trajectory(traj, stride=10)"],
    related=["mol.production", "mol.rmsd", "mol.rmsf"],
)
def load_trajectory(traj, top: Optional[str] = None, *, stride: int = 1,
                    wrap: bool = True):
    r"""Load a trajectory (imaged by default). Returns an ``mdtraj.Trajectory``."""
    return _as_traj(traj, top=top, stride=stride, wrap=wrap)


@register_function(
    aliases=["rmsd", "RMSD", "均方根偏差", "md_rmsd", "structural_drift"],
    category="mol",
    description=(
        "Per-frame RMSD to a reference frame after optimal superposition — "
        "the standard readout for whether a structure has equilibrated or is "
        "drifting. Returns a 1-D array in nanometres, one value per frame."
    ),
    examples=[
        "r = ov.mol.rmsd(traj)",
        "r = ov.mol.rmsd(traj, selection='backbone')",
    ],
    related=["mol.rmsf", "mol.plot_rmsd", "mol.radius_of_gyration"],
)
def rmsd(traj, *, reference: int = 0,
         selection: str = "protein and name CA", stride: int = 1,
         wrap: bool = True) -> np.ndarray:
    r"""Per-frame RMSD (nm) to frame ``reference``, over ``selection``."""
    import mdtraj as md
    t = _as_traj(traj, stride=stride, wrap=wrap)
    idx = _select(t, selection)
    return np.asarray(md.rmsd(t, t, frame=reference, atom_indices=idx))


@register_function(
    aliases=["rmsf", "RMSF", "残基涨落", "flexibility", "md_rmsf"],
    category="mol",
    description=(
        "Root-mean-square fluctuation per selected atom (typically per "
        "residue via CA atoms) around the mean structure — the per-residue "
        "flexibility profile. Peaks mark loops and disordered termini; "
        "useful for reading where an AlphaFold model is floppy."
    ),
    examples=["f = ov.mol.rmsf(traj)", "ov.mol.plot_rmsf(traj)"],
    related=["mol.rmsd", "mol.plot_rmsf", "mol.secondary_structure"],
)
def rmsf(traj, *, selection: str = "protein and name CA", stride: int = 1,
         wrap: bool = True) -> np.ndarray:
    r"""RMSF (nm) per selected atom, about the average structure."""
    import mdtraj as md
    t = _as_traj(traj, stride=stride, wrap=wrap)
    idx = _select(t, selection)
    t = t.superpose(t, frame=0, atom_indices=idx)
    return np.asarray(md.rmsf(t, t, frame=0, atom_indices=idx))


@register_function(
    aliases=["radius_of_gyration", "rg", "回转半径", "compactness", "md_rg"],
    category="mol",
    description=(
        "Radius of gyration per frame (nm) — a compactness measure. A steady "
        "Rg means the fold is holding; a rising Rg signals unfolding or "
        "expansion."
    ),
    examples=["rg = ov.mol.radius_of_gyration(traj)"],
    related=["mol.rmsd", "mol.plot_rg"],
)
def radius_of_gyration(traj, *, selection: str = "protein",
                       stride: int = 1, wrap: bool = True) -> np.ndarray:
    r"""Radius of gyration (nm) per frame."""
    import mdtraj as md
    t = _as_traj(traj, stride=stride, wrap=wrap)
    idx = _select(t, selection)
    return np.asarray(md.compute_rg(t.atom_slice(idx)))


@register_function(
    aliases=["hbonds", "hydrogen_bonds", "氢键", "md_hbonds"],
    category="mol",
    description=(
        "Hydrogen bonds present in at least a given fraction of frames "
        "(Baker-Hubbard criterion). Returns a DataFrame of donor/acceptor "
        "residue and atom names with occupancy — the standard way to show "
        "which protein-ligand or intra-protein contacts actually persist."
    ),
    examples=["hb = ov.mol.hbonds(traj, freq=0.2)"],
    related=["mol.contacts", "mol.native_contacts", "mol.mmgbsa"],
)
def hbonds(traj, *, freq: float = 0.1, stride: int = 1, wrap: bool = True):
    r"""Persistent hydrogen bonds as a :class:`pandas.DataFrame`.

    Columns: ``donor_residue``, ``donor_atom``, ``hydrogen``,
    ``acceptor_residue``, ``acceptor_atom``, ``occupancy``.
    """
    import mdtraj as md
    import pandas as pd

    t = _as_traj(traj, stride=stride, wrap=wrap)
    triplets = md.baker_hubbard(t, freq=freq, periodic=t.unitcell_vectors is not None)
    if len(triplets) == 0:
        return pd.DataFrame(columns=["donor_residue", "donor_atom", "hydrogen",
                                     "acceptor_residue", "acceptor_atom",
                                     "occupancy"])
    # occupancy = fraction of frames the bond satisfies the distance/angle cut
    da = triplets[:, [1, 2]]
    dist = md.compute_distances(t, da, periodic=t.unitcell_vectors is not None)
    angle_triplets = triplets[:, [0, 1, 2]]
    ang = md.compute_angles(t, angle_triplets,
                            periodic=t.unitcell_vectors is not None)
    ok = (dist < 0.25) & (ang > 2.0 * np.pi / 3.0)
    occupancy = ok.mean(axis=0)

    rows = []
    for (d, h, a), occ in zip(triplets, occupancy):
        da_, ha_, aa_ = t.top.atom(d), t.top.atom(h), t.top.atom(a)
        rows.append({
            "donor_residue": str(da_.residue), "donor_atom": da_.name,
            "hydrogen": ha_.name,
            "acceptor_residue": str(aa_.residue), "acceptor_atom": aa_.name,
            "occupancy": float(occ),
        })
    return (pd.DataFrame(rows).sort_values("occupancy", ascending=False)
            .reset_index(drop=True))


@register_function(
    aliases=["contacts", "接触", "atomic_contacts", "md_contacts"],
    category="mol",
    description=(
        "Number of atom pairs between two selections closer than a cutoff, "
        "per frame — e.g. protein-ligand contact count over time, a simple "
        "readout of whether a docked pose stays bound during MD."
    ),
    examples=[
        "n = ov.mol.contacts(traj, 'protein', 'resname LIG')",
    ],
    related=["mol.hbonds", "mol.native_contacts", "mol.mmgbsa"],
)
def contacts(traj, sel1: str, sel2: str, *, cutoff_nm: float = 0.45,
             stride: int = 1, wrap: bool = True) -> np.ndarray:
    r"""Per-frame count of ``sel1``-``sel2`` atom pairs within ``cutoff_nm``."""
    import mdtraj as md
    t = _as_traj(traj, stride=stride, wrap=wrap)
    i1, i2 = _select(t, sel1), _select(t, sel2)
    n_pairs = len(i1) * len(i2)
    if n_pairs > _MAX_PAIRS:
        raise ValueError(
            f"{len(i1)} x {len(i2)} = {n_pairs} atom pairs is too many; "
            "narrow the selections (e.g. add 'and not element H', or use "
            "residue-level selections).")
    pairs = np.array(list(itertools.product(i1, i2)), dtype=np.int32)
    d = md.compute_distances(t, pairs,
                             periodic=t.unitcell_vectors is not None)
    return np.asarray((d < cutoff_nm).sum(axis=1))


@register_function(
    aliases=["native_contacts", "Q", "天然接触", "fraction_native_contacts"],
    category="mol",
    description=(
        "Fraction of native contacts Q(t) retained relative to a reference "
        "frame — the classic folding/stability order parameter. Q near 1 "
        "means the fold is intact; a drop marks partial unfolding."
    ),
    examples=["q = ov.mol.native_contacts(traj)"],
    related=["mol.rmsd", "mol.contacts", "mol.radius_of_gyration"],
)
def native_contacts(traj, *, reference: int = 0, cutoff_nm: float = 0.45,
                    min_seq_sep: int = 3, tolerance: float = 1.2,
                    stride: int = 1, wrap: bool = True) -> np.ndarray:
    r"""Fraction of the reference frame's native contacts present per frame."""
    import mdtraj as md
    t = _as_traj(traj, stride=stride, wrap=wrap)
    heavy = t.top.select("protein and element != H")
    if len(heavy) == 0:
        raise ValueError("no protein heavy atoms found for native contacts")

    ref = t[reference]
    coords = ref.xyz[0][heavy]
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        close = tree.query_pairs(r=cutoff_nm, output_type="ndarray")
    except Exception:                                  # pragma: no cover
        d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        close = np.array(np.where(np.triu(d < cutoff_nm, k=1))).T

    if len(close) == 0:
        raise ValueError("no native contacts found in the reference frame")

    res = np.array([t.top.atom(int(i)).residue.index for i in heavy])
    keep = np.abs(res[close[:, 0]] - res[close[:, 1]]) >= min_seq_sep
    pairs = np.column_stack([heavy[close[keep, 0]], heavy[close[keep, 1]]])
    if len(pairs) == 0:
        raise ValueError(
            f"no native contacts with sequence separation >= {min_seq_sep}")

    d = md.compute_distances(t, pairs.astype(np.int32),
                             periodic=t.unitcell_vectors is not None)
    return np.asarray((d < cutoff_nm * tolerance).mean(axis=1))


@register_function(
    aliases=["secondary_structure", "dssp", "二级结构", "md_dssp"],
    category="mol",
    description=(
        "Per-frame, per-residue secondary structure via DSSP (H/E/C). Shows "
        "whether helices and sheets survive the simulation — the clearest "
        "evidence that a predicted model is or is not stable."
    ),
    examples=["ss = ov.mol.secondary_structure(traj)"],
    related=["mol.rmsf", "mol.rmsd"],
)
def secondary_structure(traj, *, simplified: bool = True, stride: int = 1,
                        wrap: bool = True) -> np.ndarray:
    r"""DSSP assignment, shape ``(n_frames, n_residues)`` of one-letter codes."""
    import mdtraj as md
    t = _as_traj(traj, stride=stride, wrap=wrap)
    protein = t.top.select("protein")
    if len(protein) == 0:
        raise ValueError("no protein atoms found for DSSP")
    return np.asarray(md.compute_dssp(t.atom_slice(protein),
                                      simplified=simplified))


@register_function(
    aliases=["rmsd_matrix", "pairwise_rmsd", "RMSD矩阵", "frame_rmsd"],
    category="mol",
    description=(
        "Frame-by-frame pairwise RMSD matrix (nm) — the input for "
        "conformational clustering and for spotting distinct basins visited "
        "during a run."
    ),
    examples=["m = ov.mol.rmsd_matrix(traj, stride=10)"],
    related=["mol.cluster", "mol.pca", "mol.rmsd"],
)
def rmsd_matrix(traj, *, selection: str = "name CA", stride: int = 1,
                wrap: bool = True) -> np.ndarray:
    r"""Symmetric ``(n_frames, n_frames)`` pairwise-RMSD matrix, in nm."""
    import mdtraj as md
    t = _as_traj(traj, stride=stride, wrap=wrap)
    idx = _select(t, selection)
    n = t.n_frames
    if n > 5000:
        raise ValueError(
            f"{n} frames would need a {n}x{n} matrix; pass a larger `stride`.")
    mat = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        mat[i] = md.rmsd(t, t, frame=i, atom_indices=idx)
    return np.maximum(mat, mat.T)      # enforce exact symmetry


@register_function(
    aliases=["cluster", "conformational_clustering", "构象聚类",
             "md_cluster"],
    category="mol",
    description=(
        "Cluster trajectory frames into conformational states by pairwise "
        "RMSD (k-means or hierarchical). Returns per-frame labels plus the "
        "representative (medoid) frame of each cluster — feed a medoid back "
        "to ov.mol.view to look at the dominant conformation."
    ),
    examples=["labels, centers = ov.mol.cluster(traj, n_clusters=5)"],
    related=["mol.rmsd_matrix", "mol.pca", "mol.view"],
)
def cluster(traj, *, n_clusters: int = 5, method: str = "kmeans",
            selection: str = "name CA", stride: int = 1,
            wrap: bool = True, seed: int = 0
            ) -> Tuple[np.ndarray, np.ndarray]:
    r"""Cluster frames; returns ``(labels, medoid_frame_indices)``."""
    mat = rmsd_matrix(traj, selection=selection, stride=stride, wrap=wrap)
    n = mat.shape[0]
    k = int(min(n_clusters, n))

    meth = str(method).lower()
    if meth == "kmeans":
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=k, random_state=seed,
                        n_init=10).fit_predict(mat)
    elif meth in ("hierarchical", "ward", "average"):
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        link = linkage(squareform(mat, checks=False),
                       method="average" if meth == "hierarchical" else meth)
        labels = fcluster(link, t=k, criterion="maxclust") - 1
    else:
        raise ValueError(
            f"method must be 'kmeans' or 'hierarchical', got {method!r}")

    medoids = []
    for c in range(labels.max() + 1):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        sub = mat[np.ix_(members, members)]
        medoids.append(int(members[np.argmin(sub.sum(axis=1))]))
    return np.asarray(labels), np.asarray(medoids)


@register_function(
    aliases=["pca", "essential_dynamics", "主成分分析", "md_pca"],
    category="mol",
    description=(
        "Principal-component analysis of the trajectory's Cartesian "
        "fluctuations (essential dynamics) after superposition. Returns the "
        "projection of each frame onto the leading components plus their "
        "explained variance — the standard way to visualise the dominant "
        "collective motions."
    ),
    examples=["proj, var = ov.mol.pca(traj, n_components=2)"],
    related=["mol.cluster", "mol.rmsd_matrix"],
)
def pca(traj, *, selection: str = "name CA", n_components: int = 2,
        stride: int = 1, wrap: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    r"""Essential-dynamics PCA; returns ``(projection, explained_variance_ratio)``."""
    from sklearn.decomposition import PCA as _PCA

    t = _as_traj(traj, stride=stride, wrap=wrap)
    idx = _select(t, selection)
    t = t.superpose(t, frame=0, atom_indices=idx)
    X = t.xyz[:, idx, :].reshape(t.n_frames, -1)
    n_comp = int(min(n_components, min(X.shape)))
    model = _PCA(n_components=n_comp)
    proj = model.fit_transform(X)
    return np.asarray(proj), np.asarray(model.explained_variance_ratio_)


# ------------------------------------------------------------------ #
# plotting                                                            #
# ------------------------------------------------------------------ #
def _new_ax(figsize):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


@register_function(
    aliases=["plot_rmsd", "RMSD图", "rmsd_plot"],
    category="mol",
    description="Plot per-frame RMSD against simulated time.",
    examples=["ov.mol.plot_rmsd(traj)"],
    related=["mol.rmsd", "mol.plot_rmsf"],
)
def plot_rmsd(traj, *, reference: int = 0,
              selection: str = "protein and name CA", stride: int = 1,
              figsize: Tuple[float, float] = (5, 3), color: str = "#1f77b4",
              ax=None, title: str = "RMSD"):
    r"""Plot RMSD vs time. Returns the matplotlib ``Axes``."""
    values = rmsd(traj, reference=reference, selection=selection, stride=stride)
    dt = getattr(traj, "dt_ps", 0.0) * stride
    x = np.arange(len(values)) * (dt / 1000.0 if dt else 1.0)
    xlabel = "time (ns)" if dt else "frame"
    if ax is None:
        _, ax = _new_ax(figsize)
    ax.plot(x, values, color=color, lw=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMSD (nm)")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


@register_function(
    aliases=["plot_rmsf", "RMSF图", "rmsf_plot", "flexibility_plot"],
    category="mol",
    description="Plot per-residue RMSF (flexibility profile).",
    examples=["ov.mol.plot_rmsf(traj)"],
    related=["mol.rmsf", "mol.plot_rmsd"],
)
def plot_rmsf(traj, *, selection: str = "protein and name CA",
              stride: int = 1, figsize: Tuple[float, float] = (5, 3),
              color: str = "#d62728", ax=None, title: str = "RMSF"):
    r"""Plot RMSF per residue. Returns the matplotlib ``Axes``."""
    values = rmsf(traj, selection=selection, stride=stride)
    t = _as_traj(traj, stride=stride)
    idx = _select(t, selection)
    resids = [t.top.atom(int(i)).residue.resSeq for i in idx]
    if ax is None:
        _, ax = _new_ax(figsize)
    ax.plot(resids, values, color=color, lw=1.2)
    ax.set_xlabel("residue")
    ax.set_ylabel("RMSF (nm)")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


@register_function(
    aliases=["plot_rg", "回转半径图", "rg_plot"],
    category="mol",
    description="Plot radius of gyration against simulated time.",
    examples=["ov.mol.plot_rg(traj)"],
    related=["mol.radius_of_gyration", "mol.plot_rmsd"],
)
def plot_rg(traj, *, selection: str = "protein", stride: int = 1,
            figsize: Tuple[float, float] = (5, 3), color: str = "#2ca02c",
            ax=None, title: str = "Radius of gyration"):
    r"""Plot Rg vs time. Returns the matplotlib ``Axes``."""
    values = radius_of_gyration(traj, selection=selection, stride=stride)
    dt = getattr(traj, "dt_ps", 0.0) * stride
    x = np.arange(len(values)) * (dt / 1000.0 if dt else 1.0)
    if ax is None:
        _, ax = _new_ax(figsize)
    ax.plot(x, values, color=color, lw=1.2)
    ax.set_xlabel("time (ns)" if dt else "frame")
    ax.set_ylabel("Rg (nm)")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


@register_function(
    aliases=["plot_energy", "能量图", "energy_plot", "md_log_plot"],
    category="mol",
    description=(
        "Plot the quantities OpenMM logged during production (potential "
        "energy, temperature, density) from the run's state log — the first "
        "thing to check when a simulation looks wrong."
    ),
    examples=["ov.mol.plot_energy(traj)"],
    related=["mol.production", "mol.plot_rmsd"],
)
def plot_energy(traj, *, quantity: str = "Potential Energy (kJ/mole)",
                figsize: Tuple[float, float] = (5, 3), color: str = "#9467bd",
                ax=None, log_path: Optional[str] = None):
    r"""Plot a column of the OpenMM state log. Returns the matplotlib ``Axes``."""
    import pandas as pd

    path = log_path or getattr(traj, "log_path", None)
    if not path:
        raise ValueError(
            "no state log available — pass log_path=, or use an MDTrajectory "
            "returned by ov.mol.production")
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip().lstrip("#").strip('"').strip() for c in df.columns]

    if quantity not in df.columns:
        matches = [c for c in df.columns if quantity.lower() in c.lower()]
        if not matches:
            raise ValueError(
                f"{quantity!r} not in the log; available columns: "
                f"{list(df.columns)}")
        quantity = matches[0]

    xcol = next((c for c in df.columns if "Time" in c), df.columns[0])
    if ax is None:
        _, ax = _new_ax(figsize)
    ax.plot(df[xcol], df[quantity], color=color, lw=1.0)
    ax.set_xlabel(xcol)
    ax.set_ylabel(quantity)
    ax.set_title(quantity.split("(")[0].strip())
    ax.spines[["top", "right"]].set_visible(False)
    return ax
