"""Paired microbe ↔ metabolite integration.

Three analysis layers that all consume a pair of AnnDatas sharing
``obs_names`` (i.e. the same samples measured on both platforms):

- :func:`paired_spearman` — Spearman rank correlation between each
  CLR-transformed microbe and each log1p-transformed metabolite,
  FDR-corrected.
- :func:`paired_cca` — Canonical Correlation Analysis (sklearn) for
  global joint covariance structure.
- :class:`MMvec` — a faithful Python/PyTorch implementation of the
  Morton *et al.* 2019 microbe-metabolite co-occurrence model.
  Learns log-conditional probabilities
  ``log P(metabolite_j | microbe_i) ∝ u_i · v_j + β_j``.

Plus a :func:`simulate_paired` synthesiser used by the tutorial and the
unit tests, and four plotting helpers (``plot_mmvec_training``,
``plot_cooccurrence``, ``plot_embedding_biplot``,
``plot_paired_method_comparison``) so notebooks stay one call per cell.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

try:
    import anndata as ad
except ImportError as exc:  # pragma: no cover
    raise ImportError("anndata is required for ov.micro._pair") from exc

from .._registry import register_function
from ._da import _bh_fdr
from ._pp import clr as _clr_inplace


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dense(X) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray(X.toarray())
    return np.asarray(X)


def _check_paired(adata_mb: "ad.AnnData", adata_mt: "ad.AnnData") -> None:
    if list(adata_mb.obs_names) != list(adata_mt.obs_names):
        raise ValueError(
            "adata_microbe.obs_names and adata_metabolite.obs_names must be "
            "identical (same samples in the same order). Re-index before "
            "calling paired_spearman / paired_cca / MMvec.fit()."
        )


# ---------------------------------------------------------------------------
# fetch_franzosa_ibd_2019 — curated real paired 16S + LC-MS dataset
# ---------------------------------------------------------------------------


_FRANZOSA_2019_BASE = (
    "https://raw.githubusercontent.com/borenstein-lab/"
    "microbiome-metabolome-curated-data/main/data/processed_data/"
    "FRANZOSA_IBD_2019"
)
_FRANZOSA_2019_FILES = ("genera.tsv", "mtb.tsv", "metadata.tsv")


@register_function(
    aliases=[
        "fetch_franzosa_ibd_2019", "fetch_franzosa_2019",
        "franzosa_2019_ibd_paired",
    ],
    category="microbiome",
    description="Download and parse the Franzosa 2019 IBD paired 16S + LC-MS metabolomics dataset (PRISM cohort, 220 samples, 88 CD / 76 UC / 56 Control).",
    examples=[
        "mb, mt = ov.micro.fetch_franzosa_ibd_2019('/scratch/data/franzosa_2019')",
    ],
    related=["micro.paired_spearman", "micro.paired_cca", "micro.MMvec"],
)
def fetch_franzosa_ibd_2019(
    data_dir: str,
    overwrite: bool = False,
    microbe_count_scale: float = 1e6,
) -> Tuple["ad.AnnData", "ad.AnnData"]:
    """Download + parse the Franzosa *et al.* 2019 paired IBD dataset.

    Files are fetched from the Borenstein lab's curated
    ``microbiome-metabolome-curated-data`` GitHub repository — three
    TSVs (genera.tsv, mtb.tsv, metadata.tsv) totalling about 30 MB.
    Once the files exist in ``data_dir`` the function is offline.

    Parameters
    ----------
    data_dir
        Absolute path the three TSVs are cached under. No ``$HOME``
        fallback — you pick where it goes (recommended: a scratch
        directory).
    overwrite
        Re-download even if the files already exist.
    microbe_count_scale
        The Borenstein TSV delivers per-sample *relative abundances*.
        To make the tables look like familiar 16S count matrices
        (integer counts, range 1 to ~100,000) we multiply by this
        scale and round — a pseudo-count-per-million by default.
        Pass ``1.0`` to keep proportions (most useful if you plan to
        CLR-transform immediately and don't need integer counts). All
        downstream ``ov.micro`` APIs (``filter_by_prevalence``,
        ``paired_spearman``, ``paired_cca``, ``MMvec``) work on either,
        but ``min_count`` filters expect counts >= 1.

    Returns
    -------
    ``(adata_microbe, adata_metabolite)`` — two AnnDatas sharing
    ``obs_names`` (same 220 samples, same order). The microbe ``var``
    carries parsed GTDB 7-rank taxonomy (``domain / phylum / class /
    order / family / genus / species`` + the raw GTDB string as
    ``taxonomy``). The metabolite ``var`` carries the cluster ID and,
    where annotated, the HMDB name (``name`` column; ``NaN`` for
    unannotated clusters). Both ``obs`` frames carry the same cohort
    metadata from metadata.tsv (``Study.Group`` = CD / UC / Control,
    ``Subject``, ``Age``, ``Fecal.Calprotectin``, drug covariates).
    """
    import os
    from urllib.request import urlretrieve

    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    paths: Dict[str, str] = {}
    for name in _FRANZOSA_2019_FILES:
        p = os.path.join(data_dir, name)
        if overwrite or not os.path.exists(p) or os.path.getsize(p) == 0:
            url = f"{_FRANZOSA_2019_BASE}/{name}"
            urlretrieve(url, p)
        paths[name] = p

    meta = pd.read_csv(paths["metadata.tsv"], sep="\t")
    meta.index = meta["Sample"].astype(str)

    df_mb = pd.read_csv(paths["genera.tsv"], sep="\t")
    df_mb.index = df_mb["Sample"].astype(str)
    df_mb = df_mb.drop(columns=["Sample"])

    df_mt = pd.read_csv(paths["mtb.tsv"], sep="\t")
    df_mt.index = df_mt["Sample"].astype(str)
    df_mt = df_mt.drop(columns=["Sample"])

    # Align all three on the intersection of samples, preserving metadata order.
    samples = [s for s in meta.index if s in df_mb.index and s in df_mt.index]
    meta  = meta.loc[samples]
    df_mb = df_mb.loc[samples]
    df_mt = df_mt.loc[samples]

    var_mb = _parse_gtdb_taxonomy(df_mb.columns)
    var_mt = _parse_metabolite_annotations(df_mt.columns)

    # Relabel columns so var_names = taxonomy tail / cluster id (short + unique).
    df_mb.columns = var_mb.index
    df_mt.columns = var_mt.index

    mb_X = df_mb.values.astype(np.float64)
    if microbe_count_scale and microbe_count_scale != 1.0:
        mb_X = np.rint(mb_X * float(microbe_count_scale)).astype(np.int64)
    adata_mb = ad.AnnData(X=mb_X, obs=meta.copy(), var=var_mb)
    adata_mt = ad.AnnData(X=df_mt.values.astype(np.float64),
                          obs=meta.copy(), var=var_mt)
    return adata_mb, adata_mt


_GTDB_RANKS = ("domain", "phylum", "class", "order", "family", "genus", "species")


def _parse_gtdb_taxonomy(columns) -> pd.DataFrame:
    """Parse ``d__X;p__Y;c__Z;…`` strings into a (n_features × 8) DataFrame."""
    rows: List[Dict[str, str]] = []
    for col in columns:
        parts = str(col).split(";")
        rank_map = {r: "" for r in _GTDB_RANKS}
        for p in parts:
            for prefix, rank in zip(
                ("d__", "p__", "c__", "o__", "f__", "g__", "s__"),
                _GTDB_RANKS,
            ):
                if p.startswith(prefix):
                    rank_map[rank] = p[len(prefix):]
                    break
        rank_map["taxonomy"] = str(col)
        rows.append(rank_map)
    var = pd.DataFrame(rows)

    # Build a short, unique var_name: prefer genus, fall back to first
    # non-empty rank, suffix a counter if duplicates exist.
    short: List[str] = []
    seen: Dict[str, int] = {}
    for i, row in var.iterrows():
        base = row["genus"] or row["family"] or row["order"] or row["class"] \
               or row["phylum"] or row["domain"] or f"tax_{i}"
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        short.append(base)
    var.index = pd.Index(short, name="feature")
    return var


def _parse_metabolite_annotations(columns) -> pd.DataFrame:
    """Split ``C18-neg_Cluster_0004: 4-hydroxystyrene`` → (cluster, name)."""
    rows: List[Dict[str, Any]] = []
    for col in columns:
        s = str(col)
        if ":" in s:
            cluster, name = s.split(":", 1)
            cluster = cluster.strip()
            name = name.strip()
            if name.upper() in {"NA", "N/A", ""}:
                name = np.nan
        else:
            cluster, name = s.strip(), np.nan
        rows.append({"cluster": cluster, "name": name, "raw": s})
    var = pd.DataFrame(rows)
    var.index = pd.Index(var["cluster"], name="feature")
    return var


# ---------------------------------------------------------------------------
# simulate_paired — synthetic data for the tutorial + the tests
# ---------------------------------------------------------------------------


@register_function(
    aliases=["simulate_paired", "simulate_microbe_metabolite"],
    category="microbiome",
    description="Simulate a paired 16S + metabolomics cohort with a known set of microbe→metabolite producer relationships for teaching, benchmarking and unit tests.",
    examples=[
        "mb, mt, truth = ov.micro.simulate_paired(n_samples=30, n_pairs=5, seed=0)",
    ],
    related=["micro.paired_spearman", "micro.MMvec"],
)
def simulate_paired(
    n_samples: int = 30,
    n_microbes: int = 40,
    n_metabolites: int = 20,
    n_pairs: int = 5,
    effect_range: Tuple[float, float] = (1.0, 2.0),
    depth_range: Tuple[int, int] = (1_000, 10_000),
    seed: int = 0,
) -> Tuple["ad.AnnData", "ad.AnnData", pd.DataFrame]:
    """Build a paired microbe + metabolite cohort with planted producer pairs.

    Returns ``(adata_microbe, adata_metabolite, truth)`` where ``truth``
    is a DataFrame with columns ``microbe / metabolite / effect`` listing
    the planted microbe→metabolite log-linear associations.
    """
    rng = np.random.default_rng(seed)
    sample_ids = [f"S{i:02d}" for i in range(n_samples)]

    # Microbe counts — multinomial from per-sample Dirichlet-style proportions,
    # scaled by variable sequencing depth so the compositional nature is real.
    log_mu = rng.normal(0, 1.0, size=(n_samples, n_microbes))
    mu = np.exp(log_mu)
    mu = mu / mu.sum(axis=1, keepdims=True)
    depths = rng.integers(depth_range[0], depth_range[1], size=n_samples)
    X_mb = np.array([rng.multinomial(d, p) for d, p in zip(depths, mu)],
                    dtype=np.int64)

    log_intensity = rng.normal(5.0, 0.6, size=(n_samples, n_metabolites))
    effects = rng.uniform(effect_range[0], effect_range[1], size=n_pairs)
    truth = pd.DataFrame({
        "microbe":    [f"ASV_{i}" for i in range(n_pairs)],
        "metabolite": [f"MET_{j}" for j in range(n_pairs)],
        "effect":     effects,
    })
    log_rel = np.log(mu + 1e-9)
    for i, row in truth.iterrows():
        m_idx = int(row["microbe"].split("_")[1])
        metab_idx = int(row["metabolite"].split("_")[1])
        log_intensity[:, metab_idx] += row["effect"] * (
            log_rel[:, m_idx] - log_rel[:, m_idx].mean()
        )
    X_mt = np.exp(log_intensity)

    microbe_names = [f"ASV_{i}" for i in range(n_microbes)]
    metab_names   = [f"MET_{j}" for j in range(n_metabolites)]

    adata_mb = ad.AnnData(
        X=sparse.csr_matrix(X_mb),
        obs=pd.DataFrame(index=sample_ids),
        var=pd.DataFrame(index=microbe_names),
    )
    adata_mt = ad.AnnData(
        X=X_mt,
        obs=pd.DataFrame(index=sample_ids),
        var=pd.DataFrame(index=metab_names),
    )
    return adata_mb, adata_mt, truth


# ---------------------------------------------------------------------------
# paired_spearman — classical baseline
# ---------------------------------------------------------------------------


@register_function(
    aliases=["paired_spearman", "paired_corr", "microbe_metabolite_spearman"],
    category="microbiome",
    description="Per-pair Spearman rank correlation between CLR-transformed microbe counts and log1p metabolite intensities, with BH-FDR across all microbe × metabolite pairs.",
    examples=[
        "ov.micro.paired_spearman(adata_microbe, adata_metabolite)",
    ],
    related=["micro.paired_cca", "micro.MMvec"],
)
def paired_spearman(
    adata_microbe: "ad.AnnData",
    adata_metabolite: "ad.AnnData",
    clr_microbe: bool = True,
    log1p_metabolite: bool = True,
    min_prevalence: float = 0.0,
) -> pd.DataFrame:
    """Rank correlation between every (microbe, metabolite) pair.

    Parameters
    ----------
    adata_microbe, adata_metabolite
        Must share ``obs_names`` (same samples, same order).
    clr_microbe
        CLR-transform the microbes first (recommended — compositional data).
    log1p_metabolite
        ``log(1 + x)``-transform the metabolites first.
    min_prevalence
        Drop microbes present in < this fraction of samples before testing
        (Spearman is undefined on constant rows).

    Returns
    -------
    DataFrame with columns ``microbe / metabolite / rho / p_value / fdr_bh``
    sorted by ``p_value`` ascending.
    """
    from scipy.stats import spearmanr

    _check_paired(adata_microbe, adata_metabolite)

    if clr_microbe:
        a = adata_microbe.copy()
        _clr_inplace(a)
        Y_mb = np.asarray(a.layers["clr"])
    else:
        Y_mb = _dense(adata_microbe.X)

    X_mt = _dense(adata_metabolite.X)
    Y_mt = np.log1p(X_mt) if log1p_metabolite else X_mt

    prev = (_dense(adata_microbe.X) > 0).mean(axis=0)
    keep_mb = prev >= min_prevalence

    mb_names = np.asarray(adata_microbe.var_names)[keep_mb]
    mt_names = np.asarray(adata_metabolite.var_names)
    Y_mb = Y_mb[:, keep_mb]

    records: List[Dict[str, Any]] = []
    for i, mb in enumerate(mb_names):
        for j, mt in enumerate(mt_names):
            rho, p = spearmanr(Y_mb[:, i], Y_mt[:, j])
            records.append({
                "microbe":    mb,
                "metabolite": mt,
                "rho":        float(rho) if np.isfinite(rho) else np.nan,
                "p_value":    float(p)   if np.isfinite(p)   else np.nan,
            })
    df = pd.DataFrame(records)
    valid = np.isfinite(df["p_value"].values)
    df["fdr_bh"] = np.nan
    if valid.any():
        df.loc[valid, "fdr_bh"] = _bh_fdr(df.loc[valid, "p_value"].values)
    return df.sort_values("p_value").reset_index(drop=True)


# ---------------------------------------------------------------------------
# paired_cca — global joint covariance
# ---------------------------------------------------------------------------


@register_function(
    aliases=["paired_cca", "canonical_correlation_microbe_metabolite"],
    category="microbiome",
    description="Canonical Correlation Analysis of CLR-transformed microbes against log1p metabolites. Returns scores, loadings, and canonical correlations.",
    examples=[
        "ov.micro.paired_cca(adata_microbe, adata_metabolite, n_components=3)",
    ],
    related=["micro.paired_spearman", "micro.MMvec"],
)
def paired_cca(
    adata_microbe: "ad.AnnData",
    adata_metabolite: "ad.AnnData",
    n_components: int = 3,
    clr_microbe: bool = True,
    log1p_metabolite: bool = True,
    max_iter: int = 500,
) -> Dict[str, Any]:
    """Run sklearn CCA on the paired tables.

    Returns a dict with keys:

    - ``cca``               — fitted :class:`sklearn.cross_decomposition.CCA`
    - ``x_scores``          — sample × components (microbe side)
    - ``y_scores``          — sample × components (metabolite side)
    - ``microbe_loadings``  — DataFrame (features × components)
    - ``metabolite_loadings`` — DataFrame (features × components)
    - ``canonical_correlations`` — list of correlations per component
    """
    from sklearn.cross_decomposition import CCA

    _check_paired(adata_microbe, adata_metabolite)

    if clr_microbe:
        a = adata_microbe.copy()
        _clr_inplace(a)
        Y_mb = np.asarray(a.layers["clr"])
    else:
        Y_mb = _dense(adata_microbe.X)
    X_mt = _dense(adata_metabolite.X)
    Y_mt = np.log1p(X_mt) if log1p_metabolite else X_mt

    cca = CCA(n_components=n_components, max_iter=max_iter)
    x_scores, y_scores = cca.fit_transform(Y_mb, Y_mt)
    canon = [float(np.corrcoef(x_scores[:, k], y_scores[:, k])[0, 1])
             for k in range(n_components)]
    microbe_loadings = pd.DataFrame(
        cca.x_loadings_,
        index=pd.Index(adata_microbe.var_names, name="microbe"),
        columns=[f"comp_{k+1}" for k in range(n_components)],
    )
    metab_loadings = pd.DataFrame(
        cca.y_loadings_,
        index=pd.Index(adata_metabolite.var_names, name="metabolite"),
        columns=[f"comp_{k+1}" for k in range(n_components)],
    )
    return {
        "cca":                    cca,
        "x_scores":               x_scores,
        "y_scores":               y_scores,
        "microbe_loadings":       microbe_loadings,
        "metabolite_loadings":    metab_loadings,
        "canonical_correlations": canon,
    }


# ---------------------------------------------------------------------------
# MMvec — Morton et al. 2019 microbe-metabolite co-occurrence model
# ---------------------------------------------------------------------------


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ov.micro.MMvec requires PyTorch (pip install torch)."
        ) from exc
    return __import__("torch")


@register_function(
    aliases=["MMvec", "microbe_metabolite_vec"],
    category="microbiome",
    description="Morton et al. 2019 microbe-metabolite co-occurrence model — learns low-rank embeddings U (microbes) / V (metabolites) such that log P(metabolite_j | microbe_i) ∝ u_i · v_j + β_j.",
    examples=[
        "mmvec = ov.micro.MMvec(n_latent=3).fit(adata_microbe, adata_metabolite)",
    ],
    related=["micro.paired_spearman", "micro.paired_cca"],
)
class MMvec:
    """MMvec (Morton *et al.* 2019) in ~80 lines of PyTorch.

    The objective is the exact expected multinomial log-likelihood

    .. math::
        \\ell \\;=\\; \\sum_{i,j} W_{ij} \\,\\log \\mathrm{softmax}(u_i \\cdot V^\\top + \\beta)_j

    where :math:`W_{ij} = \\sum_s c_{s,i} \\cdot m_{s,j} / M_s` is the
    co-occurrence weight matrix (total microbe-i count × expected
    metabolite-j fraction over the cohort). For the tutorial-scale
    data we use the full softmax; the upstream ``mmvec`` package uses
    negative sampling to scale to thousands of features.

    Parameters
    ----------
    n_latent
        Embedding dimensionality ``K``.
    lr
        Adam learning rate.
    epochs
        Maximum training epochs.
    val_frac
        Fraction of samples held out for the validation loss curve /
        early stopping. Set to 0 to skip validation.
    patience
        Early-stopping patience on validation loss (epochs without
        improvement before training halts).
    l2
        Weight-decay on ``U`` / ``V`` / ``beta``.
    seed
        Torch RNG seed.
    device
        ``'cpu'`` / ``'cuda'`` / ``None`` (auto-pick based on
        availability).
    """

    def __init__(
        self,
        n_latent: int = 3,
        lr: float = 0.05,
        epochs: int = 1000,
        val_frac: float = 0.1,
        patience: int = 100,
        l2: float = 1e-3,
        seed: int = 0,
        device: Optional[str] = None,
    ):
        self.n_latent = int(n_latent)
        self.lr       = float(lr)
        self.epochs   = int(epochs)
        self.val_frac = float(val_frac)
        self.patience = int(patience)
        self.l2       = float(l2)
        self.seed     = int(seed)
        self.device   = device

        # Populated by .fit()
        self.microbe_names_:    Optional[List[str]] = None
        self.metabolite_names_: Optional[List[str]] = None
        self.U_:    Optional[np.ndarray] = None
        self.V_:    Optional[np.ndarray] = None
        self.beta_: Optional[np.ndarray] = None
        self.loss_history_:     List[float] = []
        self.val_loss_history_: List[float] = []
        self.best_epoch_:       int = -1

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    @register_function(
        aliases=["MMvec.fit", "mmvec_fit", "fit_mmvec"],
        category="microbiome",
        description="Train MMvec on paired microbe / metabolite count tables; returns self with U_/V_/beta_/best_epoch_ populated.",
        examples=[
            "mmvec = ov.micro.MMvec(n_latent=3, epochs=600, val_frac=0.15).fit(adata_microbe, adata_metabolite)",
        ],
        related=["micro.MMvec", "micro.MMvec.cooccurrence", "micro.MMvec.conditional_probabilities"],
    )
    def fit(
        self,
        adata_microbe: "ad.AnnData",
        adata_metabolite: "ad.AnnData",
        verbose: bool = False,
    ) -> "MMvec":
        """Fit MMvec on paired microbe / metabolite count tables and return ``self``.

        Trains a low-rank latent-space model
        ``logits[i, j] = U[i] · V[j] + β[j]``, where rows of ``U`` are
        microbe embeddings, rows of ``V`` are metabolite embeddings,
        and the loss is per-microbe negative log-likelihood of the
        sample-weighted metabolite distribution. The cooccurrence
        weights ``W[i, j] = Σ_s X_mb[s, i] · X_mt[s, j] / Σ_s X_mt[s, :]``
        capture which microbes covary with which metabolites across
        samples.

        Both AnnDatas must share ``obs_names`` in the same order
        (`_check_paired` enforces this). Training reports an early-stop
        validation loss every epoch when ``val_frac > 0``; ``patience``
        epochs without validation improvement stops training and
        restores the best checkpoint.

        Parameters
        ----------
        adata_microbe, adata_metabolite : AnnData
            Sample-aligned count tables.
        verbose : bool, default False
            Print loss every ``epochs // 10`` epochs.

        Returns
        -------
        self — fitted; access ``microbe_embeddings_``,
        ``metabolite_embeddings_``, ``cooccurrence()``,
        ``conditional_probabilities()``, ``top_pairs(n)``.
        """
        torch = _require_torch()
        _check_paired(adata_microbe, adata_metabolite)
        torch.manual_seed(self.seed)

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        X_mb = _dense(adata_microbe.X).astype(np.float64)
        X_mt = _dense(adata_metabolite.X).astype(np.float64)
        self.microbe_names_    = list(adata_microbe.var_names)
        self.metabolite_names_ = list(adata_metabolite.var_names)

        # Train/val sample split.
        n_samples = X_mb.shape[0]
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(n_samples)
        n_val = int(round(self.val_frac * n_samples))
        val_idx   = order[:n_val]
        train_idx = order[n_val:] if n_val > 0 else order

        W_train = _cooccurrence_weights(X_mb[train_idx], X_mt[train_idx])
        W_val   = (_cooccurrence_weights(X_mb[val_idx], X_mt[val_idx])
                   if n_val > 0 else None)

        W_train_t = torch.as_tensor(W_train, dtype=torch.float32, device=device)
        W_val_t   = (torch.as_tensor(W_val, dtype=torch.float32, device=device)
                     if W_val is not None else None)

        M, N = X_mb.shape[1], X_mt.shape[1]
        U = torch.nn.Parameter(
            torch.randn(M, self.n_latent, device=device) * 0.1
        )
        V = torch.nn.Parameter(
            torch.randn(N, self.n_latent, device=device) * 0.1
        )
        beta = torch.nn.Parameter(torch.zeros(N, device=device))
        opt = torch.optim.Adam([U, V, beta], lr=self.lr, weight_decay=self.l2)

        best_val = float("inf")
        best_state: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        best_epoch = -1
        patience_left = self.patience
        self.loss_history_     = []
        self.val_loss_history_ = []

        for epoch in range(self.epochs):
            opt.zero_grad()
            logits = U @ V.T + beta          # (M, N)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(W_train_t * log_probs).sum() / W_train_t.sum().clamp(min=1.0)
            loss.backward()
            opt.step()
            self.loss_history_.append(float(loss.item()))

            if W_val_t is not None:
                with torch.no_grad():
                    val_log_probs = torch.log_softmax(U @ V.T + beta, dim=1)
                    val_loss = (-(W_val_t * val_log_probs).sum()
                                / W_val_t.sum().clamp(min=1.0))
                vl = float(val_loss.item())
                self.val_loss_history_.append(vl)
                if vl < best_val - 1e-6:
                    best_val = vl
                    best_state = (U.detach().cpu().numpy(),
                                  V.detach().cpu().numpy(),
                                  beta.detach().cpu().numpy())
                    best_epoch = epoch
                    patience_left = self.patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        if verbose:
                            print(f"MMvec: early stop at epoch {epoch} "
                                  f"(best val = {best_val:.4f} at {best_epoch})")
                        break

            if verbose and epoch % max(1, self.epochs // 10) == 0:
                print(f"MMvec epoch {epoch:4d}  train_loss = "
                      f"{self.loss_history_[-1]:.4f}"
                      + (f"   val_loss = {self.val_loss_history_[-1]:.4f}"
                         if self.val_loss_history_ else ""))

        if best_state is not None:
            self.U_, self.V_, self.beta_ = best_state
            self.best_epoch_ = best_epoch
        else:
            self.U_    = U.detach().cpu().numpy()
            self.V_    = V.detach().cpu().numpy()
            self.beta_ = beta.detach().cpu().numpy()
            self.best_epoch_ = len(self.loss_history_) - 1
        return self

    # ------------------------------------------------------------------
    # scoring / post-fit accessors
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.U_ is None:
            raise RuntimeError("MMvec is not fitted — call .fit() first.")

    @property
    def microbe_embeddings_(self) -> pd.DataFrame:
        self._check_fitted()
        return pd.DataFrame(self.U_, index=self.microbe_names_,
                            columns=[f"K{i+1}" for i in range(self.n_latent)])

    @property
    def metabolite_embeddings_(self) -> pd.DataFrame:
        self._check_fitted()
        return pd.DataFrame(self.V_, index=self.metabolite_names_,
                            columns=[f"K{i+1}" for i in range(self.n_latent)])

    @register_function(
        aliases=["MMvec.cooccurrence", "mmvec_cooccurrence", "cooccurrence_matrix"],
        category="microbiome",
        description="Symmetric microbe × metabolite log-odds matrix U · Vᵀ; signed scores (use top_pairs to rank by |score|).",
        examples=["co = mmvec.cooccurrence()"],
        related=["micro.MMvec", "micro.MMvec.top_pairs", "micro.MMvec.conditional_probabilities"],
    )
    def cooccurrence(self) -> pd.DataFrame:
        """Raw log-odds co-occurrence matrix ``U · Vᵀ`` (microbes × metabolites).

        Symmetric scoring before per-microbe softmax normalisation:
        positive entries indicate microbes and metabolites that
        appear together in the same samples; negative entries the
        opposite. Useful when you want **signed** scores and don't
        need them to sum to 1 per microbe — for example, downstream
        co-occurrence heatmaps and ``top_pairs(n)``. Use
        :meth:`conditional_probabilities` instead when you want
        proper P(metabolite | microbe) probabilities.
        """
        self._check_fitted()
        return pd.DataFrame(self.U_ @ self.V_.T,
                            index=self.microbe_names_,
                            columns=self.metabolite_names_)

    @register_function(
        aliases=["MMvec.conditional_probabilities", "mmvec_p_metabolite_given_microbe"],
        category="microbiome",
        description="Per-microbe softmax P(metabolite | microbe) — rows sum to 1; the canonical MMvec output.",
        examples=["P = mmvec.conditional_probabilities(); P.loc['<microbe>'].sort_values(ascending=False).head(10)"],
        related=["micro.MMvec", "micro.MMvec.cooccurrence", "micro.MMvec.top_pairs"],
    )
    def conditional_probabilities(self) -> pd.DataFrame:
        """Per-microbe P(metabolite | microbe) — softmax of ``U @ V.T + β``.

        For each microbe (row), the values across metabolites (columns)
        sum to 1 — this is what MMvec's loss directly optimises. Use
        these when the question is "given that microbe X is present,
        which metabolite is most likely?" rather than the symmetric
        co-occurrence question. Numerically stable: subtracts the
        per-row max before exponentiating.
        """
        self._check_fitted()
        logits = self.U_ @ self.V_.T + self.beta_
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return pd.DataFrame(probs,
                            index=self.microbe_names_,
                            columns=self.metabolite_names_)

    @register_function(
        aliases=["MMvec.top_pairs", "mmvec_top_pairs"],
        category="microbiome",
        description="Top-N (microbe, metabolite) pairs by |log-odds|; sign preserved on the score column for direction.",
        examples=["mmvec.top_pairs(n=20)"],
        related=["micro.MMvec", "micro.MMvec.cooccurrence"],
    )
    def top_pairs(self, n: int = 20) -> pd.DataFrame:
        """Top-``n`` (microbe, metabolite) pairs ranked by ``|log-odds|``.

        Stacks :meth:`cooccurrence` into long format and sorts by the
        absolute score, so the top of the list mixes strong positive
        and strong negative pairs (read the ``score`` column to see
        the sign). Returns a DataFrame with columns
        ``microbe``, ``metabolite``, ``score``.
        """
        co = self.cooccurrence().stack().rename("score").reset_index()
        co.columns = ["microbe", "metabolite", "score"]
        idx = co["score"].abs().sort_values(ascending=False).index
        return co.reindex(idx).reset_index(drop=True).head(n)


def _cooccurrence_weights(X_mb: np.ndarray, X_mt: np.ndarray) -> np.ndarray:
    """W[i, j] = Σ_s (X_mb[s, i] · X_mt[s, j] / M_s), M_s = X_mt[s, :].sum()."""
    M_s = X_mt.sum(axis=1, keepdims=True).clip(min=1.0)
    P_mt = X_mt / M_s
    return X_mb.T @ P_mt


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


@register_function(
    aliases=["plot_mmvec_training", "mmvec_training_curve"],
    category="microbiome",
    description="MMvec training + validation loss curve with the best-validation epoch marked.",
    examples=["ov.micro.plot_mmvec_training(mmvec)"],
    related=["micro.MMvec", "micro.MMvec.fit"],
)
def plot_mmvec_training(
    mmvec: "MMvec",
    ax: Optional[Any] = None,
) -> Any:
    """Training (and validation) loss curve for a fitted :class:`MMvec`.

    Plots ``mmvec.loss_history_`` (and ``val_loss_history_`` if a
    validation split was kept) per epoch. Marks the best-validation
    epoch with a vertical line — that's where the saved checkpoint
    came from. A monotonically falling validation loss is healthy;
    a rising val loss while train keeps falling is overfitting.

    Parameters
    ----------
    mmvec : MMvec
        Fitted model.
    ax : matplotlib axes, optional
        Axes to draw on; created if not provided.

    Returns
    -------
    matplotlib axes.
    """
    import matplotlib.pyplot as plt

    mmvec._check_fitted()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 3.6))
    epochs = np.arange(1, len(mmvec.loss_history_) + 1)
    ax.plot(epochs, mmvec.loss_history_,
            label="train", color="#1565C0", lw=1.4)
    if mmvec.val_loss_history_:
        ax.plot(epochs, mmvec.val_loss_history_,
                label="val", color="#EF6C00", lw=1.4)
        if 0 <= mmvec.best_epoch_ < len(mmvec.val_loss_history_):
            ax.axvline(mmvec.best_epoch_ + 1, color="#EF6C00",
                       lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("NLL (mean per weight)")
    ax.set_title("MMvec training")
    ax.legend(fontsize=8)
    return ax


def plot_cca_scatter(
    cca_result: Dict[str, Any],
    component: int = 1,
    ax: Optional[Any] = None,
) -> Any:
    """Scatter of microbe vs metabolite canonical variables on a given axis.

    Parameters
    ----------
    cca_result
        The dict returned by :func:`paired_cca`.
    component
        1-based index of the canonical component to plot.
    ax
        Optional existing matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    k = component - 1
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    x = cca_result["x_scores"][:, k]
    y = cca_result["y_scores"][:, k]
    r = cca_result["canonical_correlations"][k]
    ax.scatter(x, y, s=35, alpha=0.8,
               c="#5C6BC0", edgecolors="k", linewidths=0.5)
    ax.axhline(0, color="grey", lw=0.4)
    ax.axvline(0, color="grey", lw=0.4)
    ax.set_xlabel(f"microbe canonical variable {component}")
    ax.set_ylabel(f"metabolite canonical variable {component}")
    ax.set_title(f"CCA axis {component}  (r = {r:.2f})")
    return ax


def plot_cooccurrence(
    score_df: pd.DataFrame,
    top_n: int = 15,
    ax: Optional[Any] = None,
    cmap: str = "RdBu_r",
) -> Any:
    """Heatmap of a (microbe × metabolite) co-occurrence matrix.

    Picks the ``top_n`` rows and columns with the largest per-row /
    per-column absolute sums so the picture stays readable on big
    tables. Pass ``top_n=None`` to plot the full matrix.
    """
    import matplotlib.pyplot as plt

    mat = score_df.copy()
    if top_n is not None and (mat.shape[0] > top_n or mat.shape[1] > top_n):
        row_scores = mat.abs().sum(axis=1).sort_values(ascending=False)
        col_scores = mat.abs().sum(axis=0).sort_values(ascending=False)
        mat = mat.loc[row_scores.head(top_n).index,
                      col_scores.head(top_n).index]

    if ax is None:
        _, ax = plt.subplots(figsize=(0.35 * mat.shape[1] + 2,
                                      0.35 * mat.shape[0] + 1.6))
    vmax = float(np.abs(mat.values).max()) if mat.size else 1.0
    im = ax.imshow(mat.values, cmap=cmap, vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=8)
    ax.set_title(f"co-occurrence  (top {mat.shape[0]} microbes × "
                 f"{mat.shape[1]} metabolites by |score|)")
    plt.colorbar(im, ax=ax, shrink=0.75, label="log-odds")
    return ax


@register_function(
    aliases=["plot_embedding_biplot", "mmvec_biplot", "mmvec_embedding_biplot"],
    category="microbiome",
    description="MMvec joint microbe + metabolite latent-space biplot; annotates label_top items farthest from origin in each cloud.",
    examples=["ov.micro.plot_embedding_biplot(mmvec, components=(0, 1), label_top=8)"],
    related=["micro.MMvec", "micro.MMvec.fit"],
)
def plot_embedding_biplot(
    mmvec: "MMvec",
    components: Tuple[int, int] = (0, 1),
    label_top: int = 5,
    ax: Optional[Any] = None,
) -> Any:
    """Biplot of microbe + metabolite embeddings in the MMvec latent space.

    Draws two scatter point clouds in the same panel: microbes (rows
    of ``mmvec.U_``) and metabolites (rows of ``mmvec.V_``) projected
    onto the chosen ``components`` (default the first two latent
    factors). Annotates the ``label_top`` items farthest from the
    origin in each cloud — these are the entities most strongly
    associated with that pair of latent factors.

    Parameters
    ----------
    mmvec : MMvec
        Fitted model.
    components : (int, int), default (0, 1)
        Zero-indexed latent factors to plot.
    label_top : int, default 5
        Number of microbes / metabolites to annotate in each cloud.
    ax : matplotlib axes, optional

    Returns
    -------
    matplotlib axes.
    """
    import matplotlib.pyplot as plt

    mmvec._check_fitted()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.0))

    i, j = components
    U = mmvec.U_; V = mmvec.V_
    ax.scatter(U[:, i], U[:, j], s=40, alpha=0.7,
               c="#1E88E5", edgecolors="k", linewidths=0.4,
               label="microbes")
    ax.scatter(V[:, i], V[:, j], s=60, alpha=0.8, marker="^",
               c="#E53935", edgecolors="k", linewidths=0.4,
               label="metabolites")

    top_mb = np.argsort(-np.linalg.norm(U, axis=1))[:label_top]
    top_mt = np.argsort(-np.linalg.norm(V, axis=1))[:label_top]
    for idx in top_mb:
        ax.annotate(mmvec.microbe_names_[idx],
                    (U[idx, i], U[idx, j]),
                    fontsize=7, color="#0D47A1",
                    ha="left", va="bottom")
    for idx in top_mt:
        ax.annotate(mmvec.metabolite_names_[idx],
                    (V[idx, i], V[idx, j]),
                    fontsize=7, color="#B71C1C",
                    ha="left", va="bottom")
    ax.axhline(0, color="grey", lw=0.4)
    ax.axvline(0, color="grey", lw=0.4)
    ax.set_xlabel(f"MMvec component {i+1}")
    ax.set_ylabel(f"MMvec component {j+1}")
    ax.set_title("MMvec embedding — microbes (●) + metabolites (▲)")
    ax.legend(fontsize=8, loc="best")
    return ax


def plot_paired_method_comparison(
    truth: pd.DataFrame,
    spearman_df: Optional[pd.DataFrame] = None,
    mmvec_model: Optional["MMvec"] = None,
    ax: Optional[Any] = None,
) -> Any:
    """Grouped bar chart of planted-pair ranks under each method.

    Lower bars are better (the pair appears earlier in the hit list).
    Missing pairs are plotted as a hollow tall bar for visibility.
    """
    import matplotlib.pyplot as plt

    methods: Dict[str, List[int]] = {}
    if spearman_df is not None:
        methods["Spearman"] = _ranks(spearman_df, truth,
                                     microbe_col="microbe",
                                     metab_col="metabolite")
    if mmvec_model is not None:
        tp = mmvec_model.top_pairs(n=truth.shape[0] * 50 + 5)
        methods["MMvec"] = _ranks(tp, truth,
                                  microbe_col="microbe",
                                  metab_col="metabolite")

    if not methods:
        raise ValueError("Pass at least one of spearman_df or mmvec_model.")

    if ax is None:
        _, ax = plt.subplots(figsize=(max(5, 0.6 * len(truth) + 1.5), 3.6))

    n_pairs = truth.shape[0]
    x = np.arange(n_pairs)
    width = 0.8 / len(methods)
    max_rank = max((max(r for r in ranks if r > 0) for ranks in methods.values()
                    if any(r > 0 for r in ranks)), default=1)
    colors = {"Spearman": "#5C6BC0", "MMvec": "#EF5350", "CCA": "#66BB6A"}

    for k, (name, ranks) in enumerate(methods.items()):
        offset = (k - (len(methods) - 1) / 2.0) * width
        heights = [r if r > 0 else max_rank + 5 for r in ranks]
        ax.bar(x + offset, heights, width=width,
               label=name, color=colors.get(name, "#777"),
               edgecolor="k", linewidth=0.4)
        for xi, r in zip(x, ranks):
            if r <= 0:
                ax.text(xi + offset, max_rank + 5, "×",
                        ha="center", va="bottom",
                        color="red", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{a}\n→{b}" for a, b in zip(truth["microbe"], truth["metabolite"])],
        fontsize=8,
    )
    ax.set_ylabel("rank of planted pair (↓ better)")
    ax.set_title("Planted pair rank per method")
    ax.legend(fontsize=8, loc="best")
    return ax


def _ranks(
    hit_df: pd.DataFrame,
    truth: pd.DataFrame,
    microbe_col: str,
    metab_col: str,
) -> List[int]:
    """1-based rank of each truth pair within the hit_df; -1 if missing."""
    out: List[int] = []
    for _, row in truth.iterrows():
        m = hit_df[(hit_df[microbe_col] == row["microbe"]) &
                   (hit_df[metab_col]   == row["metabolite"])]
        out.append(int(m.index[0]) + 1 if len(m) else -1)
    return out
