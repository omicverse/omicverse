r"""Decipher (Nazaret et al., 2024): a two-level deep generative model whose 2-D
top latent ``v`` ("Decipher space") is an interpretable map of cell-state
trajectories, with a richer intermediate latent ``z``.

Generative model (per cell, ``G`` genes, library size ``l = sum_g x_g``)::

    v ~ N(0, I_dim_v)
    z | v ~ N(mu_z(v), softplus(s_z(v)))
    x | z ~ NB(mean = l * softmax(f(z)), inverse dispersion theta)   (per-gene theta)

Amortised posterior: ``q(z | x) q(v | z, x)`` on ``log1p(x)``; the ``v`` terms
are weighted by ``beta`` (default 0.1) in both the prior and the posterior.
The objective is the single-sample Monte-Carlo ELBO of Pyro's ``Trace_ELBO``
(summed over the minibatch), optimised with Pyro's ``ClippedAdam`` (lr 5e-3,
weight decay 1e-4, element-wise gradient clamp 10) -- the scvi-tools 1.5.2
``scvi.external.Decipher`` model rewritten with plain ``torch`` (no Pyro).
Network names and layouts match scvi-tools, so its weights load directly
(:meth:`Decipher.load_scvi_weights`).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .._registry import register_function
from ._base import BaseVIModel, _max_epochs_heuristic

_LOG_2PI = math.log(2 * math.pi)
TEST_BETA = 1.0


def _normal_log_prob(x, loc, scale):
    """Per-cell sum of log N(x; loc, scale)."""
    return (-0.5 * ((x - loc) / scale) ** 2 - torch.log(scale) - 0.5 * _LOG_2PI).sum(-1)


class ConditionalDenseNN(nn.Module):
    """Dense network with several outputs (scvi-tools / Pyro ``ConditionalDenseNN``
    without context): Linear -> BatchNorm -> ReLU per hidden layer, one linear
    output layer split into ``output_dims``.  Parameter names match scvi-tools."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dims: Sequence[int] = (1,)):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dims = list(output_dims)
        dims = [input_dim] + self.hidden_dims
        self.layers = nn.ModuleList(
            [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]
            + [nn.Linear(dims[-1], sum(self.output_dims))])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(h) for h in self.hidden_dims])

    def forward(self, x):
        h = x
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < last:
                h = F.relu(self.batch_norms[i](h))
        if len(self.output_dims) == 1:
            return h
        return tuple(torch.split(h, self.output_dims, dim=-1))


def _nb_logits_rowsum(x, mu_lib, theta, b, eps):
    """Per-cell sum of log NB(x; total_count = theta + eps, logits =
    log(mu_lib + eps) - log(theta + eps)) -- ``torch.distributions.NegativeBinomial``
    as used by scvi-tools' Decipher.  With the non-zeros of ``x`` in ``b`` the
    count-dependent terms are evaluated on the stored entries only."""
    r = theta + eps
    logits = torch.log(mu_lib + eps) - torch.log(r)
    dense = (r * F.logsigmoid(-logits)).sum(-1)
    if "row" in b and mu_lib.dim() == 2:
        flat, col, val = b["flat"], b["col"], b["val"]
        lg = F.logsigmoid(logits).reshape(-1).index_select(0, flat)
        r_s = r.index_select(0, col)
        term = val * lg + torch.lgamma(r_s + val) - torch.lgamma(r_s) - torch.lgamma(val + 1)
        return dense.index_add(0, b["row"], term)
    return dense + (x * F.logsigmoid(logits) + torch.lgamma(r + x) - torch.lgamma(r)
                    - torch.lgamma(x + 1)).sum(-1)


class DecipherModule(nn.Module):
    """The Decipher generative model and guide (scvi-tools ``DecipherPyroModule``).

    ``forward(batch)`` draws one reparameterised sample of ``(z, v)`` from the
    guide and returns the negative ELBO summed over the minibatch (``loss``) plus
    per-cell terms: ``reconstruction_loss`` = -log p(x|z), ``kl_local`` =
    [log q(z) - log p(z|v)] + beta [log q(v) - log p(v)] (``kl_z`` / ``kl_v``
    separately).  In eval mode it also reports ``nll`` (per-cell importance-weighted
    estimate of -log p(x) with ``n_iw_samples`` samples, beta = 1)."""

    def __init__(self, n_input: int, dim_v: int = 2, dim_z: int = 10,
                 layers_v_to_z: Sequence[int] = (64,), layers_z_to_x: Sequence[int] = (),
                 beta: float = 0.1, n_iw_samples: int = 5):
        super().__init__()
        self.n_input = self.dim_genes = n_input
        self.dim_v, self.dim_z = dim_v, dim_z
        self.n_latent = dim_v
        self.beta = beta
        self.n_iw_samples = n_iw_samples
        self.decoder_v_to_z = ConditionalDenseNN(dim_v, layers_v_to_z, [dim_z] * 2)
        self.decoder_z_to_x = ConditionalDenseNN(dim_z, layers_z_to_x, [n_input])
        self.encoder_x_to_z = ConditionalDenseNN(n_input, [128], [dim_z] * 2)
        self.encoder_zx_to_v = ConditionalDenseNN(n_input + dim_z, [128], [dim_v] * 2)
        # Pyro's positive constraint is an exp transform of an unconstrained value
        self.log_theta = nn.Parameter(torch.zeros(n_input))
        self._epsilon = 1e-5
        self.freeze_bn_from: int | None = None
        self._bn_frozen = False
        self.train_config = None

    # ------------------------------------------------------------------
    @property
    def theta(self):
        return torch.exp(self.log_theta)

    @property
    def cuda_graph_safe(self) -> bool:
        # a captured graph keeps BatchNorm in the mode it was captured in
        return self.freeze_bn_from is None

    def on_train_epoch_start(self, epoch: int):
        """scvi-tools freezes BatchNorm (eval mode) after the second validation
        epoch; ``freeze_bn_from`` is the first frozen epoch (None = never)."""
        self._bn_frozen = self.freeze_bn_from is not None and epoch >= self.freeze_bn_from
        if self._bn_frozen:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._bn_frozen:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
        return self

    # ------------------------------------------------------------------
    def encode(self, x):
        """Guide parameters: (z_loc, z_scale) from log1p(x)."""
        xl = torch.log1p(x)
        z_loc, z_s = self.encoder_x_to_z(xl)
        return xl, z_loc, F.softplus(z_s)

    def encode_v(self, z, xl):
        v_loc, v_s = self.encoder_zx_to_v(torch.cat([z, xl], dim=-1))
        return v_loc, F.softplus(v_s)

    def decode_z(self, v):
        z_loc, z_s = self.decoder_v_to_z(v)
        return z_loc, F.softplus(z_s)

    def decode_x(self, z):
        """Gene proportions softmax(f(z))."""
        return F.softmax(self.decoder_z_to_x(z), dim=-1)

    def latent_mean(self, b):
        """Posterior mean of v (used by sharded inference)."""
        xl, z_loc, _ = self.encode(b["x"])
        return self.encode_v(z_loc, xl)[0]

    def log_weights(self, b):
        """One guide sample: per-cell log p(x|z), log p(z|v) - log q(z|x),
        log p(v) - log q(v|z,x) (unweighted; the caller applies ``beta`` to the
        ``v`` term where needed -- :meth:`forward` for the ELBO, unweighted for
        :meth:`marginal_ll`, matching scvi-tools' ``beta=1`` test-time scale)."""
        x = b["x"]
        xl, z_loc, z_scale = self.encode(x)
        z = z_loc + z_scale * torch.randn_like(z_loc)
        v_loc, v_scale = self.encode_v(z, xl)
        v = v_loc + v_scale * torch.randn_like(v_loc)
        log_qz = _normal_log_prob(z, z_loc, z_scale)
        log_qv = _normal_log_prob(v, v_loc, v_scale)
        log_pv = (-0.5 * v ** 2 - 0.5 * _LOG_2PI).sum(-1)
        pz_loc, pz_scale = self.decode_z(v)
        log_pz = _normal_log_prob(z, pz_loc, pz_scale)
        mu = self.decode_x(z)
        lib = x.sum(-1, keepdim=True)
        log_px = _nb_logits_rowsum(x, lib * mu, self.theta, b, self._epsilon)
        return log_px, log_pz - log_qz, log_pv - log_qv

    def forward(self, b, kl_weight=1.0, beta: float | None = None):
        beta = self.beta if beta is None else beta
        log_px, wz, wv = self.log_weights(b)
        kl_z, kl_v = -wz, -wv
        kl_local = kl_z + beta * kl_v
        rec = -log_px
        out = {"loss": (rec + kl_local).sum(), "reconstruction_loss": rec,
               "kl_local": kl_local, "kl_z": kl_z, "kl_v": kl_v,
               "kl_global": torch.zeros((), device=rec.device)}
        if not self.training and self.n_iw_samples:
            out["nll"] = -self.marginal_ll(b, self.n_iw_samples)
        return out

    def marginal_ll(self, b, n_samples: int = 5):
        """Per-cell importance-weighted estimate of log p(x) (beta = 1)."""
        lw = []
        for _ in range(n_samples):
            log_px, wz, wv = self.log_weights(b)
            lw.append(log_px + wz + wv)
        lw = torch.stack(lw, 0)
        return torch.logsumexp(lw, 0) - math.log(n_samples)


# ---------------------------------------------------------------------------
# trajectories in the Decipher space
# ---------------------------------------------------------------------------
class Trajectory:
    """Piecewise-linear trajectory through a latent space (scvi-tools
    ``decipher.utils.Trajectory``): consecutive ``cluster_locations`` are joined
    by straight segments, time runs from 0 with the distance travelled, and the
    path is sampled at ``point_density`` points per unit length.

    Attributes: ``trajectory_latent`` (points), ``trajectory_time`` (their times),
    ``cumulative_length``, ``cluster_locations``, ``cluster_ids``, ``rep_key``,
    ``n_points``."""

    def __init__(self, rep_key: str, cluster_locations, cluster_ids, point_density: int = 50):
        self._point_density = point_density
        self.cluster_ids = list(cluster_ids)
        self.cluster_locations = np.asarray(cluster_locations)
        seg = np.linalg.norm(np.diff(self.cluster_locations, axis=0), axis=1)
        self.cumulative_length = np.cumsum(seg)
        self.n_points = int(self._point_density * self.cumulative_length[-1])
        self.rep_key = rep_key
        self.trajectory_latent, self.trajectory_time = self._linspace(self.n_points)

    @classmethod
    def from_clusters(cls, adata, rep_key: str, cluster_key: str, cluster_ids,
                      point_density: int = 50, reduce: str = "mean") -> "Trajectory":
        """Trajectory through the per-cluster centroids (``reduce`` = 'mean' or
        'median') of ``adata.obsm[rep_key]`` for ``cluster_ids`` in order."""
        rep = np.asarray(adata.obsm[rep_key])
        lab = np.asarray(adata.obs[cluster_key].astype(str))
        f = np.median if reduce == "median" else np.mean
        locs = []
        for c in cluster_ids:
            m = lab == str(c)
            if not m.any():
                raise ValueError(f"cluster {c!r} not found in adata.obs[{cluster_key!r}]")
            locs.append(f(rep[m], axis=0))
        return cls(rep_key, np.stack(locs), [str(c) for c in cluster_ids], point_density)

    def _linspace(self, num: int = 100):
        times = np.linspace(0, self.cumulative_length[-1], num)
        return self.at_time(times).astype(np.float32), times

    def at_time(self, t):
        """Coordinates at time(s) ``t`` along the trajectory."""
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)
        cl = self.cumulative_length
        i = np.minimum(np.searchsorted(cl, t, side="left"), len(cl) - 1)
        start = np.where(i > 0, cl[np.maximum(i - 1, 0)], 0.0)
        frac = (t - start) / (cl[i] - start)
        loc = self.cluster_locations
        out = loc[i] * (1 - frac)[:, None] + frac[:, None] * loc[i + 1]
        return out[0] if scalar else out

    def to_dict(self) -> dict:
        return {"cluster_locations": self.cluster_locations, "cluster_ids": self.cluster_ids,
                "points": self.trajectory_latent, "times": self.trajectory_time,
                "density": self._point_density, "rep_key": self.rep_key}

    @staticmethod
    def from_dict(d: dict) -> "Trajectory":
        return Trajectory(rep_key=d["rep_key"], cluster_locations=d["cluster_locations"],
                          cluster_ids=d["cluster_ids"], point_density=d.get("density", 50))

    def __repr__(self):
        return (f"Trajectory(rep_key={self.rep_key!r}, clusters={self.cluster_ids}, "
                f"length={self.cumulative_length[-1]:.3g}, n_points={self.n_points})")


def _rot(t, u=1):
    if u not in (-1, 1):
        raise ValueError("u must be -1 or 1")
    return np.array([[np.cos(t), np.sin(t) * u], [-np.sin(t), np.cos(t) * u]])


def _corr_cols(a, y):
    """Pearson correlation of each column of ``a`` (n, k) with ``y`` (n,)."""
    a = a - a.mean(0)
    y = y - y.mean()
    return (a * y[:, None]).sum(0) / np.sqrt((a * a).sum(0) * (y * y).sum())


def rotate_decipher_components(adata, v_obsm_key: str, z_obsm_key: str,
                               v1_obs_col: str | None = None, v1_order: list | None = None,
                               v2_obs_col: str | None = None, v2_order: list | None = None,
                               auto_flip_decipher_z: bool = True):
    """Rotate / flip the Decipher space so that component 1 (2) correlates with
    ``adata.obs[v1_obs_col]`` (``v2_obs_col``) and not with the other one; ordered
    categories are given with ``v1_order`` / ``v2_order``.  Same grid search as
    scvi-tools (100 angles x 2 reflections).  ``auto_flip_decipher_z`` flips each
    z dimension to correlate positively with the rotated v.

    The returned ``decipher_v`` is for **display** (plots, ``Trajectory`` built
    only to draw a path over a rotated embedding). The model's decoders
    (``decoder_v_to_z``, used by :meth:`Decipher.compute_gene_patterns`) only
    understand the native, un-rotated ``v`` -- if you overwrite
    ``adata.obsm[v_obsm_key]`` with this rotated array and then build a
    ``Trajectory`` from it for :meth:`Decipher.compute_gene_patterns`, pass the
    returned ``rotation`` to ``compute_gene_patterns(..., rotation=rotation)``
    so it can be undone before decoding; otherwise the imputed gene patterns
    are decoded from meaningless (rotated) coordinates.

    Returns ``(decipher_v, decipher_z, rotation)``."""
    def process(col, order):
        if col is None:
            return None, None
        s = adata.obs[col]
        if order is not None:
            s = s.astype("category").cat.set_categories(order)
            s = s.cat.codes.replace(-1, np.nan)
        elif s.dtype == "category":
            raise ValueError("v_order must be provided if v_obs is a category column")
        valid = ~s.isna()
        return np.asarray(s[valid], dtype=float), np.asarray(valid)

    if v1_obs_col is None and v2_obs_col is None:
        raise ValueError("At least one of v1_obs_col or v2_obs_col must be provided")
    y1, m1 = process(v1_obs_col, v1_order)
    y2, m2 = process(v2_obs_col, v2_order)
    v = np.asarray(adata.obsm[v_obsm_key])
    rots = [_rot(t, u) for t in np.linspace(0, 2 * np.pi, 100) for u in (1, -1)]
    R = np.stack(rots)                                    # (200, 2, 2)
    scores = np.zeros(len(rots))
    if y1 is not None:
        c = np.stack([_corr_cols(v[m1] @ r, y1) for r in R])
        scores += c[:, 0] - np.abs(c[:, 1])
    if y2 is not None:
        c = np.stack([_corr_cols(v[m2] @ r, y2) for r in R])
        scores += c[:, 1] - np.abs(c[:, 0])
    best = R[int(np.argmax(scores))]
    v = v @ best
    z = np.asarray(adata.obsm[z_obsm_key])
    if auto_flip_decipher_z:
        dz = z.shape[1]
        corr = np.corrcoef(z, y=v, rowvar=False)
        z = z * np.sign(corr[:dz, dz:].sum(axis=1))
    return v, z, best


def compute_decipher_time(adata, cluster_obs_key: str, trajectory: Trajectory,
                          n_neighbors: int = 10) -> np.ndarray:
    """Decipher time of each cell: KNN regression of the trajectory times on
    the cells' ``adata.obsm[trajectory.rep_key]``; NaN for cells outside the
    trajectory's clusters."""
    from sklearn.neighbors import KNeighborsRegressor
    out = np.full(adata.n_obs, np.nan)
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(trajectory.trajectory_latent, trajectory.trajectory_time)
    lab = adata.obs[cluster_obs_key]
    on = np.flatnonzero(np.asarray(lab.astype(str).isin([str(c) for c in trajectory.cluster_ids])))
    if len(on):
        out[on] = knn.predict(np.asarray(adata.obsm[trajectory.rep_key])[on])
    return out


# ---------------------------------------------------------------------------
@register_function(
    aliases=["Decipher", "decipher", "vi.Decipher", "Decipher space", "trajectory VAE"],
    category="single",
    description="omicverse-native Decipher (no scvi-tools / Pyro): two-level VAE with a 2-D "
                "interpretable latent v and a 10-D latent z for visualising normal and "
                "perturbed differentiation trajectories; trains on AnnData, AnnDataOOM and "
                "Dask-backed AnnData.",
    examples=[
        "m = ov.vi.Decipher(adata, layer='counts')",
        "m.train()",
        "adata.obsm['decipher_v'] = m.get_latent_representation()",
        "adata.obsm['decipher_z'] = m.get_latent_representation(give_z=True)",
    ],
    related=["single.decipher", "vi.SCVI"],
)
class Decipher(BaseVIModel):
    """Decipher (Nazaret et al., 2024), omicverse implementation of
    ``scvi.external.Decipher`` (scvi-tools 1.5.2) in plain PyTorch.

    Arguments:
        adata: AnnData, AnnDataOOM (``ov.read(path, storage='disk')``) or
            Dask-backed AnnData.  Raw counts are read from ``layer``.
        layer: Layer with raw counts (None = ``X``).
        genes: Training genes (boolean mask, names or a ``var`` column such as
            ``'highly_variable'``).
        dim_v: Dimension of the interpretable latent ``v`` (2).
        dim_z: Dimension of the intermediate latent ``z`` (10).
        layers_v_to_z: Hidden sizes of the v -> z decoder ((64,)).
        layers_z_to_x: Hidden sizes of the z -> x decoder (() = linear).
        beta: Weight of the ``v`` KL term (0.1).
        seed: Seed of the weight initialisation (None = current torch RNG).

    Example::

        m = ov.vi.Decipher(adata, layer="counts")
        m.train()
        adata.obsm["decipher_v"] = m.get_latent_representation()
        adata.obsm["decipher_z"] = m.get_latent_representation(give_z=True)
    """

    _module_cls = DecipherModule

    def __init__(self, adata, layer=None, genes=None, dim_v: int = 2, dim_z: int = 10,
                 layers_v_to_z: Sequence[int] = (64,), layers_z_to_x: Sequence[int] = (),
                 beta: float = 0.1, seed: int | None = 0, **module_kwargs):
        super().__init__(adata, layer=layer, genes=genes)
        self._build(seed=seed, dim_v=dim_v, dim_z=dim_z, layers_v_to_z=list(layers_v_to_z),
                    layers_z_to_x=list(layers_z_to_x), beta=beta, **module_kwargs)

    def _build(self, seed=None, **kw):
        kw = {k: v for k, v in kw.items() if k != "seed"}
        self._module_kwargs = kw
        self._module_args = dict(n_input=self.n_vars, **kw)
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(seed)
            self.module = self._module_cls(**self._module_args)
        self.module.to(self._device)

    # ------------------------------------------------------------------
    def train(self, max_epochs: int | None = None, *, batch_size: int = 128,
              train_size: float | None = 0.9, early_stopping: bool = False,
              check_val_every_n_epoch: int | None = None, lr: float | None = None,
              freeze_batchnorm: str | int | None = "auto", plan_kwargs: dict | None = None,
              **kwargs):
        """Train with scvi-tools' Decipher defaults: ``ClippedAdam`` (lr 5e-3,
        weight decay 1e-4, clamp 10, eps 1e-8), loss summed over the minibatch,
        incomplete last minibatch dropped, max epochs
        ``min(round(20000 / n_cells * 1000), 1000)``, early stopping on
        ``nll_validation``.

        Arguments:
            freeze_batchnorm: Epoch from which BatchNorm layers stay in eval mode.
                ``'auto'`` reproduces scvi-tools (frozen after the second epoch
                with a validation pass, i.e. only when validation runs, e.g. with
                ``early_stopping``); an int sets the epoch; None never freezes.
            **kwargs: See :meth:`BaseVIModel.train` (device, seed, shuffle,
                block_size, ... and optimisation options such as ``weight_decay``,
                ``clip_norm``, ``early_stopping_patience``).
        """
        if max_epochs is None:
            max_epochs = _max_epochs_heuristic(self.adata.n_obs, epochs_cap=1000)
        plan = {"optimizer": "ClippedAdam", "lr": 5e-3, "weight_decay": 1e-4, "eps": 1e-8,
                "clip_norm": 10.0, "n_epochs_kl_warmup": None, "drop_last": True,
                "early_stopping_monitor": "nll_validation"}
        plan.update(plan_kwargs or {})
        if lr is not None:
            plan["lr"] = lr
        if freeze_batchnorm == "auto":
            # scvi-tools freezes BatchNorm on the first validation pass with
            # current_epoch > 0; training epochs after it run frozen
            n = check_val_every_n_epoch or (1 if early_stopping else None)
            freeze = None if n is None else (2 if n == 1 else n)
        else:
            freeze = freeze_batchnorm
        self.module.freeze_bn_from = freeze
        self.module._bn_frozen = False
        try:
            return super().train(max_epochs, batch_size=batch_size, train_size=train_size,
                                 early_stopping=early_stopping,
                                 check_val_every_n_epoch=check_val_every_n_epoch,
                                 plan_kwargs=plan, **kwargs)
        finally:
            self.module._bn_frozen = False

    # ------------------------------------------------------------------
    def get_latent_representation(self, adata=None, indices=None, batch_size: int | None = None,
                                  give_z: bool = False) -> np.ndarray:
        """Posterior mean of the Decipher space ``v`` (``give_z=False``) or of ``z``."""
        mod = self.module

        def fn(b):
            xl, z_loc, _ = mod.encode(b["x"])
            return z_loc if give_z else mod.encode_v(z_loc, xl)[0]

        return self._collect(fn, adata, indices, batch_size)

    def get_marginal_ll(self, adata=None, indices=None, n_samples: int = 100,
                        batch_size: int | None = None, return_mean: bool = True):
        """Importance-weighted estimate of log p(x) per cell (beta = 1)."""
        mod = self.module
        out = self._collect(lambda b: mod.marginal_ll(b, n_samples), adata, indices, batch_size)
        return float(np.mean(out)) if return_mean else out

    def compute_imputed_gene_expression(self, adata=None, indices=None,
                                        batch_size: int | None = None,
                                        compute_covariances: bool = False,
                                        v_obsm_key: str | None = None,
                                        z_obsm_key: str | None = None,
                                        return_imputed: bool = True):
        """Imputed expression ``library_size * softmax(f(z_loc))`` per cell (cells x
        genes); with ``compute_covariances`` also the gene x v and gene x z
        covariances with ``adata.obsm[v_obsm_key]`` / ``[z_obsm_key]`` (computed
        in one streaming pass, so ``return_imputed=False`` never holds the cells x
        genes matrix -- use it for disk-backed data)."""
        if compute_covariances and (v_obsm_key is None or z_obsm_key is None):
            raise ValueError("`v_obsm_key` and `z_obsm_key` must be provided if "
                             "`compute_covariances` is True.")
        if not compute_covariances and not return_imputed:
            raise ValueError("nothing to return")
        mod = self.module
        ad = self.adata if adata is None else adata

        def fn(b):
            _, z_loc, _ = mod.encode(b["x"])
            return b["x"].sum(-1, keepdim=True) * mod.decode_x(z_loc)

        if not compute_covariances:
            return self._collect(fn, adata, indices, batch_size)
        self._check_trained()
        _, it, *_ = self._batches(adata, indices, batch_size)
        sel = np.arange(ad.n_obs) if indices is None else np.asarray(indices)
        if sel.dtype == bool:
            sel = np.flatnonzero(sel)
        Y = np.concatenate([np.asarray(ad.obsm[v_obsm_key])[sel],
                            np.asarray(ad.obsm[z_obsm_key])[sel]], 1).astype(np.float64)
        dv = np.asarray(ad.obsm[v_obsm_key]).shape[1]
        pos_of = {int(r): i for i, r in enumerate(sel)} if indices is not None else None
        G = self.n_vars
        sx = np.zeros(G); sxy = np.zeros((G, Y.shape[1]))
        imputed = np.empty((len(sel), G), np.float32) if return_imputed else None
        mod.eval()
        with torch.no_grad():
            for b in it:
                blk = fn(b).double().cpu().numpy()
                idx = b["idx"].cpu().numpy()
                pos = idx if pos_of is None else np.array([pos_of[int(i)] for i in idx])
                sx += blk.sum(0)
                sxy += blk.T @ Y[pos]
                if imputed is not None:
                    imputed[pos] = blk
        m = len(sel)
        cov = (sxy - np.outer(sx, Y.sum(0)) / m) / (m - 1)
        res = (cov[:, :dv], cov[:, dv:])
        return (imputed,) + res if return_imputed else res

    @staticmethod
    def compute_decipher_time(adata, cluster_obs_key: str, trajectory: Trajectory,
                              n_neighbors: int = 10) -> np.ndarray:
        """See :func:`compute_decipher_time`."""
        return compute_decipher_time(adata, cluster_obs_key, trajectory, n_neighbors)

    def compute_gene_patterns(self, adata=None, trajectory: Trajectory = None,
                              l_scale: float = 10_000, n_samples: int = 100,
                              rotation: np.ndarray | None = None) -> dict:
        """Gene expression along a trajectory of the Decipher space: the trajectory
        points go through the v -> z -> x decoders.  Returns ``mean`` (z at its
        mean), ``q25`` / ``q75`` (quantiles over ``n_samples`` z samples), all
        times ``l_scale``, and ``times``.

        ``decoder_v_to_z`` only understands the model's *native* (un-rotated) ``v``
        -- the one straight out of :meth:`get_latent_representation`.  If
        ``trajectory`` was built from a Decipher space rotated for display with
        :func:`rotate_decipher_components` (e.g. the obsm key you plotted
        ``adata.obsm[key]`` into also got overwritten with the rotated array, or
        you built the ``Trajectory`` directly from the rotated array), decoding
        those points directly is meaningless -- pass the same ``rotation`` matrix
        returned by :func:`rotate_decipher_components` here and it is undone
        before decoding. Leave ``None`` if ``trajectory`` is already in the
        model's native space."""
        if trajectory is None:
            raise ValueError("pass a Trajectory")
        mod = self.module
        was = mod.training
        mod.eval()
        pts = trajectory.trajectory_latent
        if rotation is not None:
            pts = pts @ np.asarray(rotation).T
        with torch.no_grad():
            t = torch.as_tensor(pts, dtype=torch.float32,
                                device=self.device)
            z_mean, z_scale = mod.decode_z(t)
            zs = torch.distributions.Normal(z_mean, z_scale).sample((n_samples,))
            mean = mod.decode_x(z_mean).cpu().numpy() * l_scale
            samples = mod.decode_x(zs).cpu().numpy() * l_scale
        mod.train(was)
        return {"mean": mean, "q25": np.quantile(samples, 0.25, axis=0),
                "q75": np.quantile(samples, 0.75, axis=0), "times": trajectory.trajectory_time}

    # ------------------------------------------------------------------
    def load_scvi_weights(self, scvi_model):
        """Copy the weights of a trained ``scvi.external.Decipher`` (networks and the
        Pyro parameter ``theta``) into this model (same genes and architecture)."""
        sd = {k: v.detach().cpu() for k, v in scvi_model.module.state_dict().items()
              if not k.startswith("_dummy")}
        import pyro
        theta = pyro.get_param_store()["theta"].detach().cpu()
        sd["log_theta"] = torch.log(theta)
        self.module.load_state_dict(sd)
        self.module.to(self.device)
        self.is_trained_ = True
        self.module.eval()
        return self

    def __repr__(self):
        k = self._module_kwargs
        return (f"Decipher(n_obs={self.adata.n_obs}, n_vars={self.n_vars}, dim_v={k['dim_v']}, "
                f"dim_z={k['dim_z']}, beta={k['beta']}, trained={self.is_trained_})")


__all__ = ["Decipher", "DecipherModule", "Trajectory", "rotate_decipher_components",
           "compute_decipher_time"]
