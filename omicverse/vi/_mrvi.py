"""MrVI (Boyeau, Hong, Xu et al. 2024): multi-resolution variational inference for
multi-sample single-cell data.

MrVI learns two nested latent variables per cell: a sample-*unaware* ``u`` (captures
cell state / cell type, independent of which biological sample the cell came from) and
a sample-*aware* ``z`` obtained from ``u`` through an attention mechanism conditioned on
the cell's sample.  Because the sample-conditioning is a small attention block rather
than a per-sample encoder, every cell can be re-encoded *as if* it had come from any
other sample (counterfactual ``z``), which is what the downstream analyses below use:
sample-sample distances local to groups of cells, differential abundance of samples in
the ``u`` space, and multivariate differential expression of sample covariates in the
``z`` space.

Ported from scvi-tools 1.5.2's PyTorch implementation (``scvi.external.mrvi``): the
*torch* backend has been the default for MrVI since scvi-tools 1.4.3 (the original JAX
backend was deprecated in 1.5), so this port is against the torch reference, not jax.
No scvi-tools / jax import anywhere here; only ``torch`` and ``torch.distributions``
(``Normal``, ``Independent``, ``Categorical``, ``MixtureSameFamily``, all part of core
PyTorch).  Contributed via PR #1009 (hurry); re-implemented natively on ``ov.vi``'s
block-shuffle data sources (in-memory, AnnDataOOM, Dask) so it needs no scvi-tools.

Latent hierarchy
-----------------
``qu = EncoderXU(x, sample)``                     amortised posterior of ``u``
``u ~ qu``                                        (or the mean, ``use_mean=True``)
``z_base, eps_params = EncoderUZ(u, cf_sample)``  attention from ``u`` to ``z``,
                                                   conditioned on ``cf_sample`` (the
                                                   cell's own sample during training;
                                                   any sample for counterfactuals)
``eps ~ N(eps_params)`` (or ``z_base = u`` if ``n_latent == n_latent_u``, no residual)
``z = z_base + eps``
``px = DecoderZXAttention(z, batch)``             NB likelihood, attention over a small
                                                   batch embedding
``p(u)``: standard normal, or a learned mixture of ``u_prior_mixture_k`` Gaussians
          (``u_prior_mixture=True``, the default — one component per label when
          ``n_labels > 1``).
``p(z | u)``: ``N(z_base, exp(z_u_prior_scale))`` on the residual ``eps`` only
             (``z_u_prior=True``, the default).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal, kl_divergence

from .._registry import register_function
from ._base import BaseVIModel, _categories, _codes
from ._distributions import log_nb_positive, nb_rowsum

DEFAULT_PX_KWARGS = {"n_hidden": 32, "stop_gradients": False, "stop_gradients_mlp": True,
                     "dropout_rate": 0.03}
DEFAULT_QZ_KWARGS = {"use_map": True, "stop_gradients": False, "stop_gradients_mlp": True,
                     "dropout_rate": 0.03}


# ---------------------------------------------------------------------------
# building blocks (scvi-tools' scvi.external.mrvi._components, ported verbatim;
# MrVI-specific, not shared with the rest of ov.vi -> kept local to this module)
# ---------------------------------------------------------------------------
def _gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU (tanh approximation), matching the original JAX/Flax ``nn.gelu``."""
    return F.gelu(x, approximate="tanh")


class _ResnetBlock(nn.Module):
    def __init__(self, n_in, n_out, n_hidden=128, internal_activation=F.relu,
                 output_activation=F.relu):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.ln1 = nn.LayerNorm(n_hidden, eps=1e-6)
        self.fc_match = nn.Linear(n_in, n_hidden) if n_in != n_hidden else None
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.ln2 = nn.LayerNorm(n_out, eps=1e-6)
        self.internal_activation = internal_activation
        self.output_activation = output_activation

    def forward(self, x):
        h = self.internal_activation(self.ln1(self.fc1(x)))
        h = h + (self.fc_match(x) if self.fc_match is not None else x)
        return self.output_activation(self.ln2(self.fc2(h)))


class _MLP(nn.Module):
    """Resnet-block MLP (scvi-tools' ``MLP``)."""

    def __init__(self, n_in, n_out, n_hidden=128, n_layers=1, activation=F.relu):
        super().__init__()
        # NB: the resnet block's own internal width stays at its default (128),
        # matching scvi-tools' ``MLP`` -- only ``n_out`` (== ``n_hidden`` here) is set.
        self.blocks = nn.Sequential(*[
            _ResnetBlock(n_in if i == 0 else n_hidden, n_hidden,
                        internal_activation=activation, output_activation=activation)
            for i in range(n_layers)])
        self.fc = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        return self.fc(self.blocks(x))


class _NormalDistOutputNN(nn.Module):
    """Resnet MLP parameterising a diagonal Normal (scvi-tools ``NormalDistOutputNN``)."""

    def __init__(self, n_in, n_out, n_hidden=128, n_layers=1, scale_eps=1e-5):
        super().__init__()
        self.blocks = nn.ModuleList([
            _ResnetBlock(n_in if i == 0 else n_hidden, n_hidden) for i in range(n_layers)])
        self.fc_mean = nn.Linear(n_hidden, n_out)
        self.fc_scale = nn.Sequential(nn.Linear(n_hidden, n_out), nn.Softplus())
        self.scale_eps = scale_eps

    def forward(self, x):
        h = x
        for b in self.blocks:
            h = b(h)
        return Normal(self.fc_mean(h), self.fc_scale(h) + self.scale_eps)


class _ConditionalNormalization(nn.Module):
    """LayerNorm / BatchNorm followed by condition-specific scale and shift."""

    def __init__(self, n_features, n_conditions, normalization_type="layer"):
        super().__init__()
        self.gamma = nn.Embedding(n_conditions, n_features)
        self.beta = nn.Embedding(n_conditions, n_features)
        nn.init.normal_(self.gamma.weight, mean=1.0, std=0.02)
        nn.init.zeros_(self.beta.weight)
        if normalization_type == "batch":
            self.norm = nn.BatchNorm1d(n_features, affine=False, track_running_stats=True)
        elif normalization_type == "layer":
            self.norm = nn.LayerNorm(n_features, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError("normalization_type must be 'batch' or 'layer'")

    def forward(self, x, condition, training=None):
        if training is not None:
            self.norm.train(training)
        x = self.norm(x)
        cond = condition.reshape(-1).to(torch.int64)
        return self.gamma(cond) * x + self.beta(cond)


class _AttentionBlock(nn.Module):
    """Outer-product multi-head attention, matching scvi-tools' PyTorch port of the
    Flax ``MultiHeadDotProductAttention``-based ``AttentionBlock``: project query / kv
    to ``(outerprod_dim, 1)``, run multi-head attention from that singleton feature
    dimension, then an MLP on the flattened attention output plus a residual MLP."""

    def __init__(self, query_dim, kv_dim, out_dim, outerprod_dim=16, n_channels=4, n_heads=2,
                 dropout_rate=0.0, n_hidden_mlp=32, n_layers_mlp=1, stop_gradients_mlp=False,
                 activation=_gelu):
        super().__init__()
        self.outerprod_dim = outerprod_dim
        self.n_heads = n_heads
        self.depth_per_head = n_channels
        self.stop_gradients_mlp = stop_gradients_mlp
        self.dropout_rate = dropout_rate
        qkv_dim = n_heads * n_channels
        self.query_proj = nn.Linear(query_dim, outerprod_dim, bias=False)
        self.kv_proj = nn.Linear(kv_dim, outerprod_dim, bias=False)
        self.q_proj = nn.Linear(1, qkv_dim)
        self.k_proj = nn.Linear(1, qkv_dim)
        self.v_proj = nn.Linear(1, qkv_dim)
        self.out_proj = nn.Linear(qkv_dim, n_channels)
        self.mlp_eps = _MLP(outerprod_dim * n_channels, outerprod_dim, n_hidden_mlp,
                            n_layers_mlp, activation)
        self.mlp_residual = _MLP(outerprod_dim + query_dim, out_dim, n_hidden_mlp,
                                 n_layers_mlp, activation)

    def forward(self, query_embed, kv_embed):
        has_mc = query_embed.ndim == 3
        q_in = query_embed.detach() if self.stop_gradients_mlp else query_embed
        q_att = self.query_proj(q_in).unsqueeze(-1)
        kv_att = self.kv_proj(kv_embed).unsqueeze(-1)
        if has_mc:
            mc, b = query_embed.shape[0], query_embed.shape[1]
            q_flat = query_embed.reshape(mc * b, -1)
            q_att = q_att.reshape(mc * b, self.outerprod_dim, 1)
            kv_att = kv_att.reshape(mc * b, self.outerprod_dim, 1)
        else:
            q_flat = query_embed
        q, k, v = self.q_proj(q_att), self.k_proj(kv_att), self.v_proj(kv_att)
        fb = q.shape[0]

        def heads(t):
            return t.view(fb, self.outerprod_dim, self.n_heads, self.depth_per_head).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)
        dp = self.dropout_rate if self.training else 0.0
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=dp)
        attn = attn.transpose(1, 2).reshape(fb, self.outerprod_dim, self.n_heads * self.depth_per_head)
        eps = self.out_proj(attn).reshape(fb, -1)
        eps = self.mlp_eps(eps)
        residual = self.mlp_residual(torch.cat([q_flat, eps], dim=-1))
        if has_mc:
            residual = residual.reshape(mc, b, -1)
        return residual


class DecoderZXAttention(nn.Module):
    """``p(x | z, batch)``: attention-based NB decoder (``n_batch`` embeddings as the
    attention's key/value)."""

    def __init__(self, n_in, n_out, n_batch, n_latent_sample=16, n_channels=4, n_heads=2,
                 dropout_rate=0.1, stop_gradients=False, stop_gradients_mlp=False, n_hidden=32,
                 n_layers=1, low_dim_batch=True, activation=_gelu):
        super().__init__()
        self.n_batch = n_batch
        self.stop_gradients = stop_gradients
        self.low_dim_batch = low_dim_batch
        self.layer_norm = nn.LayerNorm(n_in, eps=1e-6)
        self.layer_norm_batch = nn.LayerNorm(n_latent_sample, eps=1e-6)
        self.batch_embedding = nn.Embedding(max(n_batch, 1), n_latent_sample)
        nn.init.normal_(self.batch_embedding.weight, std=0.1)
        res_dim = n_in if low_dim_batch else n_out
        self.attention_block = _AttentionBlock(n_in, n_latent_sample, res_dim, n_latent_sample,
                                               n_channels, n_heads, dropout_rate, n_hidden,
                                               n_layers, stop_gradients_mlp, activation)
        self.fc = nn.Linear(n_in, n_out)
        self.px_r = nn.Parameter(torch.zeros(n_out))
        nn.init.normal_(self.px_r)

    def forward(self, z, batch_index, size_factor):
        has_mc = z.ndim == 3
        z_in = z.detach() if self.stop_gradients else z
        z_ = self.layer_norm(z_in)
        batch_index = batch_index.to(torch.int64).flatten()
        if self.n_batch >= 2:
            be = self.layer_norm_batch(self.batch_embedding(batch_index))
            if has_mc:
                be = be.unsqueeze(0).expand(z_.shape[0], -1, -1)
            residual = self.attention_block(z_, be)
            mu = self.fc(z + residual) if self.low_dim_batch else self.fc(z) + residual
        else:
            mu = self.fc(z_)
        mu = torch.softmax(mu, dim=-1)
        return _NB(mu * size_factor, torch.exp(self.px_r))


class _NB:
    """Negative binomial with ``mean`` and inverse dispersion ``theta`` (scvi-tools
    parameterisation); log-likelihood goes through ``ov.vi``'s ``log_nb_positive`` /
    ``nb_rowsum`` rather than a distribution object."""

    __slots__ = ("mean", "theta")

    def __init__(self, mean, theta):
        self.mean, self.theta = mean, theta


class EncoderUZ(nn.Module):
    """``u -> z`` attention encoder, conditioned on the (possibly counterfactual)
    sample index.  Isomorphic case (``n_latent == n_latent_u``): ``z_base = u``, no
    extra projection layer, matching the reference's parameter count exactly."""

    def __init__(self, n_latent, n_sample, n_latent_u=None, n_latent_sample=16, n_channels=4,
                n_heads=2, dropout_rate=0.0, stop_gradients=False, stop_gradients_mlp=False,
                use_map=True, n_hidden=32, n_layers=1, activation=_gelu):
        super().__init__()
        self.n_latent_u = n_latent_u if n_latent_u is not None else n_latent
        self.stop_gradients = stop_gradients
        self.layer_norm = nn.LayerNorm(self.n_latent_u, eps=1e-6)
        self.embedding = nn.Embedding(n_sample, n_latent_sample)
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.layer_norm_embed = nn.LayerNorm(n_latent_sample, eps=1e-6)
        n_outs = 1 if use_map else 2
        self.attention_block = _AttentionBlock(self.n_latent_u, n_latent_sample,
                                               n_outs * n_latent, n_latent_sample, n_channels,
                                               n_heads, dropout_rate, n_hidden, n_layers,
                                               stop_gradients_mlp, activation)
        self.fc = nn.Linear(self.n_latent_u, n_latent) if n_latent_u is not None else None

    def forward(self, u, sample_covariate):
        sample_covariate = sample_covariate.to(torch.int64).flatten()
        has_mc = u.ndim == 3
        u_in = u.detach() if self.stop_gradients else u
        u_ = self.layer_norm(u_in)
        se = self.layer_norm_embed(self.embedding(sample_covariate))
        if has_mc:
            se = se.unsqueeze(0).expand(u_.shape[0], -1, -1)
        residual = self.attention_block(u_, se)
        if self.fc is not None:
            return self.fc(u_in), residual
        return u, residual


class EncoderXU(nn.Module):
    """``x -> u`` amortised encoder with sample-conditional normalisation
    (``ConditionalNormalization``) and an additive sample embedding, matching
    scvi-tools' ``EncoderXU``."""

    def __init__(self, n_input, n_latent, n_sample, n_hidden=128, n_layers=1, activation=_gelu):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.cn1 = _ConditionalNormalization(n_hidden, n_sample)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.cn2 = _ConditionalNormalization(n_hidden, n_sample)
        self.sample_embed = nn.Embedding(n_sample, n_hidden)
        nn.init.normal_(self.sample_embed.weight, std=0.1)
        self.output_nn = _NormalDistOutputNN(n_hidden, n_latent, n_hidden, n_layers)
        self.activation = activation

    def forward(self, x, sample_covariate, training=None):
        h = torch.log1p(x)
        h = self.activation(self.cn1(self.fc1(h), sample_covariate, training))
        h = self.activation(self.cn2(self.fc2(h), sample_covariate, training))
        se = self.sample_embed(sample_covariate.reshape(-1).to(torch.int64))
        return self.output_nn(h + se)


# ---------------------------------------------------------------------------
# module
# ---------------------------------------------------------------------------
class MRVAEModule(nn.Module):
    """MrVI's generative model + amortised posterior (scvi-tools ``MRVAE``)."""

    def __init__(self, n_input, n_sample, n_batch, n_labels, n_latent=30, n_latent_u=10,
                encoder_n_hidden=128, encoder_n_layers=2, z_u_prior=True, z_u_prior_scale=0.0,
                u_prior_scale=0.0, u_prior_mixture=True, u_prior_mixture_k=20,
                learn_z_u_prior_scale=False, scale_observations=False, px_kwargs=None,
                qz_kwargs=None, qu_kwargs=None, n_obs_per_sample=None):
        super().__init__()
        self.n_input, self.n_sample, self.n_batch, self.n_labels = n_input, n_sample, n_batch, n_labels
        self.n_latent, self.n_latent_u = n_latent, n_latent_u
        self.z_u_prior, self.u_prior_mixture = z_u_prior, u_prior_mixture
        self.scale_observations = scale_observations
        if n_obs_per_sample is not None:
            self.register_buffer("n_obs_per_sample",
                                 torch.as_tensor(n_obs_per_sample, dtype=torch.float32),
                                 persistent=False)
        else:
            self.n_obs_per_sample = None

        pxk = {**DEFAULT_PX_KWARGS, **(px_kwargs or {})}
        qzk = {**DEFAULT_QZ_KWARGS, **(qz_kwargs or {})}
        quk = dict(qu_kwargs or {})

        is_iso = n_latent == n_latent_u
        self.px = DecoderZXAttention(n_latent, n_input, n_batch, **pxk)
        self.qz = EncoderUZ(n_latent, n_sample, n_latent_u=None if is_iso else n_latent_u, **qzk)
        self.qu = EncoderXU(n_input, n_latent if is_iso else n_latent_u, n_sample,
                            n_hidden=encoder_n_hidden, n_layers=encoder_n_layers, **quk)

        if learn_z_u_prior_scale:
            self.pz_scale = nn.Parameter(torch.zeros(n_latent))
        else:
            self.register_buffer("pz_scale", torch.full((n_latent,), float(z_u_prior_scale)))

        if u_prior_mixture:
            k = n_labels if n_labels > 1 else u_prior_mixture_k
            u_dim = n_latent_u if n_latent_u is not None else n_latent
            self.u_prior_logits = nn.Parameter(torch.zeros(k))
            self.u_prior_means = nn.Parameter(torch.randn(k, u_dim))
            self.u_prior_scales = nn.Parameter(torch.zeros(k, u_dim))
        else:
            self.register_buffer("u_prior_scale", torch.tensor(float(u_prior_scale)))

    # ------------------------------------------------------------------
    def inference(self, x, sample_index, mc_samples=None, cf_sample=None, use_mean=False):
        qu = self.qu(x, sample_index, training=self.training)
        if use_mean:
            u = qu.mean
        else:
            shape = (mc_samples,) if mc_samples is not None else ()
            u = qu.mean + torch.randn(shape + qu.mean.shape, device=qu.mean.device,
                                      dtype=qu.mean.dtype) * qu.scale
        sample_index_cf = sample_index if cf_sample is None else cf_sample
        z_base, eps_params = self.qz(u, sample_index_cf)
        qeps = None
        eps = eps_params
        if eps_params.shape[-1] == 2 * self.n_latent:
            loc, scale = eps_params[..., :self.n_latent], eps_params[..., self.n_latent:]
            qeps = Normal(loc, F.softplus(scale) + 1e-3)
            eps = qeps.mean if use_mean else qeps.mean + torch.randn_like(qeps.mean) * qeps.scale
        z = z_base + eps
        library = torch.log(x.sum(1, keepdim=True))
        return {"qu": qu, "qeps": qeps, "eps": eps, "u": u, "z": z, "z_base": z_base,
                "library": library}

    def generative(self, z, library, batch_index, label_index, transform_batch=None):
        if transform_batch is not None:
            batch_index = torch.full_like(batch_index, int(transform_batch))
        library_exp = torch.exp(library)
        px = self.px(z, batch_index, size_factor=library_exp)
        h = px.mean / library_exp
        if self.u_prior_mixture:
            offset = 0.0
            if self.n_labels >= 2:
                offset = 10.0 * F.one_hot(label_index.reshape(-1).to(torch.int64),
                                          self.n_labels).float()
            cats = Categorical(logits=self.u_prior_logits + offset)
            comp = Independent(Normal(self.u_prior_means, torch.exp(self.u_prior_scales)), 1)
            pu = MixtureSameFamily(cats, comp)
        else:
            pu = Normal(0, torch.exp(self.u_prior_scale))
        return {"px": px, "pu": pu, "h": h}

    def log_likelihood(self, b, px) -> torch.Tensor:
        mu, theta = px.mean.float(), px.theta.float()
        if "row" in b and mu.dim() == 2:
            return nb_rowsum(mu, theta, b["row"], b["col"], b["val"], flat=b.get("flat"))
        return log_nb_positive(b["x"], mu, theta).sum(-1)

    def forward(self, b, kl_weight: float = 1.0):
        inf = self.inference(b["x"], b["sample"])
        gen = self.generative(inf["z"], inf["library"], b["batch"], b.get("labels"))
        rec = -self.log_likelihood(b, gen["px"])
        if self.u_prior_mixture:
            kl_u = inf["qu"].log_prob(inf["u"]).sum(-1) - gen["pu"].log_prob(inf["u"])
        else:
            kl_u = kl_divergence(inf["qu"], gen["pu"]).sum(-1)
        kl_z = torch.zeros_like(kl_u)
        if self.z_u_prior:
            peps = Normal(0, torch.exp(self.pz_scale))
            kl_z = -peps.log_prob(inf["eps"]).sum(-1)
        kl_local = kl_u + kl_z
        loss_vec = rec + kl_weight * kl_local
        if self.scale_observations and self.n_obs_per_sample is not None:
            prefactors = self.n_obs_per_sample[b["sample"].reshape(-1).to(torch.int64)]
            loss_vec = loss_vec / prefactors.clamp(min=1.0)
        return {"loss": loss_vec.mean(), "reconstruction_loss": rec, "kl_local": kl_local,
                "kl_global": torch.zeros((), device=rec.device)}

    def compute_h_from_x_eps(self, x, sample_index, batch_index, extra_eps, cf_sample=None,
                             mc_samples=10):
        """Normalised expression ``h`` using a fixed ``extra_eps`` shift added to
        ``z_base`` (used by differential expression's log-fold-change estimate)."""
        inf = self.inference(x, sample_index, mc_samples=mc_samples, cf_sample=cf_sample,
                             use_mean=False)
        library = torch.full((x.shape[0], 1), 7.0, device=x.device, dtype=torch.float32)
        gen = self.generative(inf["z_base"] + extra_eps, library, batch_index,
                              torch.zeros(x.shape[0], 1, dtype=torch.long, device=x.device))
        return gen["h"]


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
@register_function(
    aliases=["MrVI", "MRVI", "sample-aware embedding", "多样本层级变分推断"],
    category="single",
    description="omicverse-native MrVI (no scvi-tools): hierarchical u/z VAE with a "
                "sample-conditioned attention encoder/decoder and a mixture-of-Gaussians "
                "prior, for multi-sample scRNA-seq with sample-level covariates. Exposes "
                "local sample representations, sample-sample distance matrices, "
                "differential abundance and multivariate differential expression.",
    examples=[
        "m = ov.vi.MRVI(adata, sample_key='donor', batch_key='batch')",
        "m.train()",
        "adata.obsm['X_mrvi_u'] = m.get_latent_representation()",
        "dists = m.get_local_sample_distances()",
        "de = m.differential_expression(sample_cov_keys=['disease'])",
    ],
    related=["single.mrvi", "vi.SCVI", "vi.SCANVI"],
)
class MRVI(BaseVIModel):
    """Multi-resolution Variational Inference (Boyeau et al., 2024).

    Arguments:
        adata: AnnData, AnnDataOOM or Dask-backed AnnData.
        sample_key: ``obs`` column with the biological sample identity (>= 2 samples);
            this is what MrVI's hierarchy and downstream analyses condition on.
        layer: Layer with raw counts (None = ``X``).
        batch_key: ``obs`` column with a *technical* batch (None = one batch); injected
            into the decoder's attention (distinct from ``sample_key``).
        labels_key: ``obs`` column with labels; with >= 2 labels the ``u`` mixture prior
            uses one component per label instead of ``u_prior_mixture_k``.
        genes: Training genes (mask / names / a ``var`` column).
        n_latent, n_latent_u: Dimensionality of ``z`` / ``u`` (defaults 30, 10, as
            scvi-tools 1.5.2).
        encoder_n_hidden, encoder_n_layers: ``EncoderXU`` size (defaults 128, 2).
        z_u_prior, z_u_prior_scale, learn_z_u_prior_scale: Gaussian prior on the
            ``z - z_base`` residual (defaults True, 0.0, False).
        u_prior_scale, u_prior_mixture, u_prior_mixture_k: Prior on ``u``: a mixture of
            ``u_prior_mixture_k`` Gaussians by default (True, 20) or a standard normal
            scaled by ``exp(u_prior_scale)``.
        scale_observations: Scale each cell's loss by ``1 / n_obs_of_its_sample``
            (default False).
        px_kwargs, qz_kwargs, qu_kwargs: Extra keyword arguments for the decoder / the
            two encoders (attention width, dropout, gradient stopping...).
        seed: Seed of the weight initialisation.

    Example::

        m = ov.vi.MRVI(adata, sample_key="donor", batch_key="batch")
        m.train()
        u = m.get_latent_representation()                 # sample-unaware state
        z = m.get_latent_representation(give_z=True)       # sample-aware state
        reps = m.get_local_sample_representation()         # (n_cells, n_sample, n_latent)
        dmat = m.get_local_sample_distances(groupby="cell_type")
        da = m.differential_abundance(sample_cov_keys=["disease"])
        de = m.differential_expression(sample_cov_keys=["disease"])
    """

    _module_cls = MRVAEModule

    def __init__(self, adata, *, sample_key, layer=None, batch_key=None, labels_key=None,
                genes=None, n_latent: int = 30, n_latent_u: int = 10,
                encoder_n_hidden: int = 128, encoder_n_layers: int = 2, z_u_prior: bool = True,
                z_u_prior_scale: float = 0.0, u_prior_scale: float = 0.0,
                u_prior_mixture: bool = True, u_prior_mixture_k: int = 20,
                learn_z_u_prior_scale: bool = False, scale_observations: bool = False,
                px_kwargs: dict | None = None, qz_kwargs: dict | None = None,
                qu_kwargs: dict | None = None, seed: int | None = 0):
        if sample_key is None:
            raise ValueError("MRVI needs sample_key= (the obs column with each cell's "
                             "biological sample identity)")
        if sample_key not in adata.obs.columns:
            raise KeyError(f"{sample_key!r} is not a column of adata.obs")
        sample_categories = _categories(adata.obs[sample_key])
        if len(sample_categories) < 2:
            raise ValueError(f"adata.obs[{sample_key!r}] must have at least 2 samples "
                             f"(got {len(sample_categories)})")
        super().__init__(adata, layer=layer, batch_key=batch_key, labels_key=labels_key,
                         genes=genes)
        self.registry["sample_key"] = sample_key
        self.registry["sample_categories"] = sample_categories
        kw = dict(n_latent=n_latent, n_latent_u=n_latent_u, encoder_n_hidden=encoder_n_hidden,
                  encoder_n_layers=encoder_n_layers, z_u_prior=z_u_prior,
                  z_u_prior_scale=z_u_prior_scale, u_prior_scale=u_prior_scale,
                  u_prior_mixture=u_prior_mixture, u_prior_mixture_k=u_prior_mixture_k,
                  learn_z_u_prior_scale=learn_z_u_prior_scale,
                  scale_observations=scale_observations, px_kwargs=px_kwargs,
                  qz_kwargs=qz_kwargs, qu_kwargs=qu_kwargs)
        self._build(seed=seed, **kw)

    # ------------------------------------------------------------------
    @property
    def n_sample(self):
        return len(self.registry["sample_categories"])

    @property
    def sample_order(self):
        return list(self.registry["sample_categories"])

    def _sample_codes(self, adata):
        r = self.registry
        return _codes(adata.obs[r["sample_key"]].values, r["sample_categories"], r["sample_key"])

    def _extra_obs(self, adata):
        return {"sample": self._sample_codes(adata)}

    def _n_obs_per_sample(self):
        counts = np.bincount(self._sample_codes(self.adata), minlength=self.n_sample)
        return torch.as_tensor(counts.astype(np.float32))

    def _build(self, seed=None, **kw):
        kw = {k: v for k, v in kw.items() if k != "seed"}
        self._module_kwargs = kw
        n_obs_per_sample = self._n_obs_per_sample() if kw.get("scale_observations") else None
        self._module_args = dict(n_input=self.n_vars, n_sample=self.n_sample,
                                 n_batch=self.n_batch, n_labels=self.n_labels,
                                 n_obs_per_sample=n_obs_per_sample, **kw)
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(seed)
            self.module = self._module_cls(**self._module_args)
        self.module.to(self._device)

    def train(self, *args, cuda_graph="auto", **kwargs):
        """As :meth:`BaseVIModel.train`, but ``cuda_graph`` defaults to ``False``.

        ``inference()``'s reparameterisation (``torch.randn`` during the CUDA-graph
        warm-up, before the capturable try/except) raises
        ``RuntimeError: Offset increment outside graph capture encountered
        unexpectedly`` and crashes (no eager fallback) when ``cuda_graph`` is enabled
        for this model, unlike ``torch.compile``'s own step-gated ``"auto"``. Pass
        ``cuda_graph=True`` explicitly to try it anyway.
        """
        if cuda_graph == "auto":
            cuda_graph = False
        return super().train(*args, cuda_graph=cuda_graph, **kwargs)

    # ------------------------------------------------------------------
    # latent representations
    # ------------------------------------------------------------------
    def get_latent_representation(self, adata=None, indices=None, give_z: bool = False,
                                  use_mean: bool = True, batch_size: int | None = None):
        """Posterior mean (``use_mean``) of ``u`` (sample-unaware, default) or ``z``
        (sample-aware, ``give_z=True``)."""
        mod = self.module

        def fn(b):
            inf = mod.inference(b["x"], b["sample"], use_mean=use_mean)
            return inf["z"] if give_z else inf["u"]

        return self._collect(fn, adata, indices, batch_size)

    def _cf_batches(self, adata=None, indices=None, batch_size=None):
        return self._batches(adata, indices, batch_size)

    def get_local_sample_representation(self, adata=None, indices=None, batch_size: int = 256,
                                        use_mean: bool = True, mc_samples: int = 10):
        """For every cell, the ``z`` it would have under every sample in turn:
        ``(n_cells, n_sample, n_latent)`` (``use_mean=False`` averages ``mc_samples``
        posterior draws instead of using the posterior mean)."""
        self._check_trained()
        mod = self.module
        src, it, rows, order, n = self._cf_batches(adata, indices, batch_size)
        n_sample = self.n_sample
        was = mod.training
        mod.eval()
        out = None
        with torch.no_grad():
            for b in it:
                x, sample_index = b["x"], b["sample"]
                reps = []
                for s in range(n_sample):
                    cf = torch.full_like(sample_index, s)
                    if use_mean:
                        z = mod.inference(x, sample_index, cf_sample=cf, use_mean=True)["z"]
                    else:
                        z = mod.inference(x, sample_index, cf_sample=cf, use_mean=False,
                                          mc_samples=mc_samples)["z"].mean(0)
                    reps.append(z)
                rep = torch.stack(reps, dim=1)
                if out is None:
                    out = torch.empty((n, n_sample, rep.shape[-1]), dtype=rep.dtype,
                                      device=rep.device)
                pos = b["idx"] if rows is None else torch.from_numpy(
                    np.searchsorted(rows, b["idx"].cpu().numpy())).to(rep.device)
                out[pos] = rep
        mod.train(was)
        res = out.cpu().numpy() if out is not None else np.zeros((0, n_sample, 0), np.float32)
        return res if order is None else res[order]

    def _local_baseline_dists(self, b, mc_samples: int = 125):
        """Mean / variance of the Euclidean distance between two independent posterior
        ``z`` draws of the *same* (observed) sample, per cell -- the baseline used to
        standardise ``normalize_distances=True``."""
        mod = self.module
        inf = mod.inference(b["x"], b["sample"], mc_samples=2 * mc_samples, use_mean=False)
        z = inf["z"]
        d = torch.sqrt(((z[:mc_samples] - z[mc_samples:]) ** 2).sum(-1)).T
        return d.mean(1), d.var(1)

    def get_local_sample_distances(self, adata=None, indices=None, batch_size: int = 256,
                                   use_mean: bool = True, groupby=None, keep_cell: bool = True,
                                   norm: str = "l2", normalize_distances: bool = False,
                                   mc_samples: int = 10):
        """Cell-specific ``(n_sample, n_sample)`` distance matrices between the
        counterfactual sample representations of :meth:`get_local_sample_representation`.

        Arguments:
            groupby: ``obs`` column(s); also returns the per-group average matrix
                (e.g. per cell type) as ``result['groups'][group_key][category]``.
            keep_cell: Keep the per-cell matrices (``result['cell']``,
                ``(n_cells, n_sample, n_sample)``).
            normalize_distances: Standardise by the per-cell within-sample distance
                baseline (forces ``use_mean=False``).
        Returns:
            dict with ``sample_order`` and, depending on the arguments, ``cell`` and
            ``groups``.
        """
        if normalize_distances and use_mean:
            use_mean = False
        if norm not in ("l2", "l1", "linf"):
            raise ValueError("norm must be 'l2', 'l1' or 'linf'")

        def dist_mat(rep):  # rep: (..., n_sample, n_latent) -> (..., n_sample, n_sample)
            d = rep[..., :, None, :] - rep[..., None, :, :]
            if norm == "l2":
                return torch.sqrt((d ** 2).sum(-1))
            if norm == "l1":
                return d.abs().sum(-1)
            return d.abs().max(-1).values

        self._check_trained()
        mod = self.module
        src, it, rows, order, n = self._cf_batches(adata, indices, batch_size)
        n_sample = self.n_sample
        was = mod.training
        mod.eval()
        cells, group_sums, group_counts = [], {}, {}
        ad = self.adata if adata is None else adata
        groupby = [groupby] if isinstance(groupby, str) else groupby
        with torch.no_grad():
            for b in it:
                x, sample_index = b["x"], b["sample"]
                if normalize_distances:
                    base_mean, base_var = self._local_baseline_dists(b, mc_samples=mc_samples)
                reps = []
                for s in range(n_sample):
                    cf = torch.full_like(sample_index, s)
                    msamp = mc_samples if not use_mean else None
                    z = mod.inference(x, sample_index, cf_sample=cf, use_mean=use_mean,
                                      mc_samples=msamp)["z"]
                    reps.append(z)
                rep = torch.stack(reps, dim=-2)  # (..., n_sample, n_latent) or (mc, B, n_sample, n_latent)
                dmat = dist_mat(rep)
                if dmat.ndim == 4:  # (mc, B, n_sample, n_sample)
                    if normalize_distances:
                        dmat = (dmat - base_mean.view(1, -1, 1, 1)) / (
                            base_var.view(1, -1, 1, 1).clamp(min=1e-8) ** 0.5)
                    dmat = dmat.mean(0)
                cells.append((b["idx"].cpu().numpy(), dmat.cpu().numpy()))
                if groupby:
                    idx_np = b["idx"].cpu().numpy()
                    for g in groupby:
                        col = np.asarray(ad.obs[g].values)[idx_np]
                        for cat in np.unique(col):
                            m = col == cat
                            s = dmat[torch.as_tensor(m, device=dmat.device)].sum(0).cpu().numpy()
                            group_sums.setdefault(g, {}).setdefault(cat, np.zeros((n_sample, n_sample)))
                            group_sums[g][cat] += s
                            group_counts.setdefault(g, {}).setdefault(cat, 0)
                            group_counts[g][cat] += int(m.sum())
        mod.train(was)
        result = {"sample_order": self.sample_order}
        if keep_cell:
            idx_all = np.concatenate([c[0] for c in cells])
            mat_all = np.concatenate([c[1] for c in cells], axis=0)
            full = np.empty((n,) + mat_all.shape[1:], dtype=mat_all.dtype)
            full[idx_all] = mat_all
            result["cell"] = full if order is None else full[order]
        if groupby:
            result["groups"] = {g: {cat: group_sums[g][cat] / max(group_counts[g][cat], 1)
                                    for cat in group_sums[g]} for g in group_sums}
        return result

    # ------------------------------------------------------------------
    # differential abundance
    # ------------------------------------------------------------------
    def get_aggregated_posterior(self, adata=None, sample=None, indices=None,
                                 batch_size: int = 256):
        """Aggregated posterior over ``u`` (a uniform mixture of every cell's ``qu``),
        optionally restricted to one sample -- ``torch.distributions.MixtureSameFamily``."""
        self._check_trained()
        ad = self.adata if adata is None else adata
        idx = np.arange(ad.n_obs) if indices is None else np.asarray(indices)
        if sample is not None:
            mask = np.asarray(ad.obs[self.registry["sample_key"]].values)[idx] == sample
            idx = idx[mask]
        mod = self.module
        src, it, rows, order, n = self._batches(ad, idx, batch_size)
        locs, scales = [], []
        was = mod.training
        mod.eval()
        with torch.no_grad():
            for b in it:
                qu = mod.qu(b["x"], b["sample"], training=False)
                locs.append(qu.mean)
                scales.append(qu.scale)
        mod.train(was)
        loc, scale = torch.cat(locs, 0), torch.cat(scales, 0)
        cats = Categorical(probs=torch.ones(loc.shape[0], device=loc.device) / loc.shape[0])
        return MixtureSameFamily(cats, Independent(Normal(loc, scale), 1))

    def differential_abundance(self, adata=None, sample_cov_keys=None, sample_subset=None,
                               compute_log_enrichment: bool = False,
                               omit_original_sample: bool = True, batch_size: int = 128):
        """Log p(sample | u) for every cell and every sample (how likely each cell's
        ``u`` is under each sample's aggregated posterior); with ``sample_cov_keys``,
        also aggregated over the samples sharing each covariate value (e.g. disease
        status), optionally as a log-enrichment vs. the rest.

        Returns a dict: ``log_probs`` ``(n_cells, n_sample)``, ``samples`` (names) and,
        per key in ``sample_cov_keys``, ``{key}_log_probs`` (and ``{key}_log_enrichs``).
        """
        from scipy.special import logsumexp

        ad = self.adata if adata is None else adata
        sample_key = self.registry["sample_key"]
        samples = np.asarray(ad.obs[sample_key].astype(str).values)
        unique_samples = list(pd.unique(samples))
        us = self.get_latent_representation(ad, use_mean=True, give_z=False,
                                            batch_size=batch_size)
        us_t = torch.as_tensor(us, dtype=torch.float32, device=self.device)
        n_splits = max(ad.n_obs // max(batch_size, 1), 1)
        log_probs = []
        for s in unique_samples:
            ap = self.get_aggregated_posterior(ad, sample=s, batch_size=batch_size)
            parts = [ap.log_prob(chunk).detach().cpu().numpy()
                    for chunk in torch.chunk(us_t, n_splits)]
            log_probs.append(np.concatenate(parts))
        log_probs = np.stack(log_probs, axis=1)  # (n_cells, n_sample)
        result = {"log_probs": log_probs, "samples": unique_samples}
        if not sample_cov_keys:
            return result
        sample_info = ad.obs[[sample_key] + list(sample_cov_keys)].drop_duplicates(sample_key)

        def aggregate(cols, omit_original):
            sel = np.isin(unique_samples, cols)
            sub = log_probs[:, sel]
            if omit_original:
                # Exact per-column match of each cell's own sample against the
                # *selected* samples, not membership-in-the-whole-group: `np.isin`
                # here returns one bool per cell (True iff the cell's own sample is
                # *anywhere* in `cols`), which `np.where` then broadcast across every
                # column of `sub` -- blanking a cell's entire row (not just the one
                # matching column) to -inf whenever its own sample belongs to the
                # aggregated group, making `aggregate()` return -inf for every cell
                # of its own covariate value. Elementwise `==` against the selected
                # sample names gives the intended (n_cells, n_selected) mask with
                # True only in the single column equal to the cell's own sample.
                one_hot = samples[:, None] == np.asarray(unique_samples)[sel][None, :]
                sub = np.where(one_hot, -np.inf, sub)
                denom = np.log(np.maximum((~one_hot).sum(1), 1))
            else:
                denom = np.log(max(sub.shape[1], 1))
            return logsumexp(sub, axis=1) - denom

        for key in sample_cov_keys:
            vals = sample_info.set_index(sample_key)[key]
            per_val = {}
            per_enrich = {}
            for v in pd.unique(vals.values):
                cov_samples = vals.index[vals == v].tolist()
                if sample_subset is not None:
                    cov_samples = [s for s in cov_samples if s in sample_subset]
                if not cov_samples:
                    continue
                per_val[v] = aggregate(cov_samples, omit_original_sample)
                if compute_log_enrichment:
                    rest = [s for s in unique_samples if s not in cov_samples]
                    if rest:
                        per_enrich[v] = per_val[v] - aggregate(rest, omit_original_sample)
            result[f"{key}_log_probs"] = per_val
            if compute_log_enrichment:
                result[f"{key}_log_enrichs"] = per_enrich
        return result

    # ------------------------------------------------------------------
    # differential expression
    # ------------------------------------------------------------------
    def _design_matrix(self, sample_cov_keys, normalize: bool = True):
        import pandas as pd
        sample_key = self.registry["sample_key"]
        info = self.adata.obs[[sample_key] + list(sample_cov_keys)].drop_duplicates(sample_key)
        info = info.set_index(sample_key).loc[self.sample_order]
        cols, names = [], []
        for key in sample_cov_keys:
            cov = info[key]
            if cov.dtype == object or str(cov.dtype) == "category":
                cov = pd.get_dummies(cov.astype("category"), drop_first=True)
                cols.append(cov.values.astype(np.float32))
                names += [f"{key}_{c}" for c in cov.columns]
            else:
                cols.append(cov.values.astype(np.float32)[:, None])
                names.append(key)
        X = np.concatenate(cols, axis=1)
        if normalize:
            lo, hi = X.min(0), X.max(0)
            X = (X - lo) / (1e-6 + hi - lo)
        return torch.as_tensor(X), np.asarray(names)

    def differential_expression(self, adata=None, sample_cov_keys=None, sample_subset=None,
                                batch_size: int = 128, normalize_design_matrix: bool = True,
                                mc_samples: int = 50, store_lfc: bool = False,
                                mc_samples_lfc: int = 15, eps_lfc: float = 1e-4,
                                lambd: float = 0.0):
        """Per-cell multivariate differential expression (Boyeau et al., 2024, section
        "Multivariate analyses"): for every cell, every counterfactual sample's
        ``eps = z - z_base`` shift is regressed on the samples' covariates (``Xmat``,
        one row per sample); the covariates' effect sizes (a chi-squared statistic),
        p-values (BH-adjusted) and regression coefficients (``beta``, in ``z`` space)
        are cell-specific.

        With ``store_lfc=True``, also decodes ``beta``'s shift through the decoder to
        report gene-level log2 fold-changes (``lfc``) -- the posterior mean over
        ``mc_samples_lfc`` draws of ``u``, using each cell's own batch (a simplification
        of the reference's n_batch-averaged, admissibility-filtered estimate: no
        cross-batch averaging, no outlier-sample filtering, no ``lfc_std`` / ``pde``).

        Returns a dict: ``beta`` ``(n_cells, n_covariate, n_latent)``, ``effect_size``
        and ``pvalue``/``padj`` ``(n_cells, n_covariate)``, ``covariates`` (names), and
        (if ``store_lfc``) ``lfc`` ``(n_cells, n_covariate, n_gene)``.
        """
        if not sample_cov_keys:
            raise ValueError("sample_cov_keys is required")
        from scipy.stats import false_discovery_control
        import scipy.linalg
        import torch.distributions as dist

        self._check_trained()
        mod = self.module
        ad = self.adata if adata is None else adata
        sample_mask = (np.isin(self.sample_order, sample_subset) if sample_subset is not None
                      else np.ones(self.n_sample, dtype=bool))
        kept = np.where(sample_mask)[0]
        Xmat, names = self._design_matrix(sample_cov_keys, normalize_design_matrix)
        Xmat = Xmat[sample_mask]
        n_cov = Xmat.shape[1]
        XtX = Xmat.T @ Xmat + lambd * torch.eye(n_cov)
        prefactor = torch.as_tensor(np.real(scipy.linalg.sqrtm(XtX.numpy())), dtype=torch.float32)
        Amat = (torch.linalg.pinv(XtX) @ Xmat.T).to(self.device)        # (n_cov, n_sample_kept)
        prefactor = prefactor.to(self.device)
        Xmat = Xmat.to(self.device)

        src, it, rows, order, n = self._cf_batches(ad, None, batch_size)
        was = mod.training
        mod.eval()
        betas, effects, pvals = [], [], []
        lfcs = [] if store_lfc else None
        n_genes = self.n_vars
        df = torch.tensor(float(len(kept)))
        with torch.no_grad():
            for b in it:
                x, sample_index, batch_index = b["x"], b["sample"], b["batch"]
                eps_list = []
                for s in kept:
                    cf = torch.full_like(sample_index, int(s))
                    eps_list.append(mod.inference(x, sample_index, cf_sample=cf,
                                                   mc_samples=mc_samples, use_mean=False)["eps"])
                eps = torch.stack(eps_list, dim=-2)  # (mc, B, n_sample_kept, n_latent)
                eps_std = eps.std(dim=-2, keepdim=True)
                eps_mean = eps.mean(dim=-2, keepdim=True)
                eps_n = (eps - eps_mean) / (1e-6 + eps_std)
                beta = torch.einsum("ks,mbsd->mbkd", Amat, eps_n)        # (mc, B, n_cov, n_latent)
                beta_norm = torch.einsum("mbkd,kl->mbld", beta, prefactor)
                ts = (beta_norm ** 2).mean(0).sum(-1)                    # (B, n_cov)
                pval = 1 - dist.Chi2(df).cdf(ts.detach().cpu())
                beta = beta * eps_std
                betas.append(beta.mean(0).cpu().numpy())
                effects.append(ts.cpu().numpy())
                pvals.append(pval.numpy())
                if store_lfc:
                    beta_mean = beta.mean(0)                              # (B, n_cov, n_latent)
                    gene_lfc = np.zeros((x.shape[0], n_cov, n_genes), dtype=np.float32)
                    h0 = mod.compute_h_from_x_eps(x, sample_index, batch_index,
                                                  extra_eps=torch.zeros_like(beta_mean[:, 0]),
                                                  mc_samples=mc_samples_lfc)
                    for k in range(n_cov):
                        h1 = mod.compute_h_from_x_eps(x, sample_index, batch_index,
                                                      extra_eps=beta_mean[:, k],
                                                      mc_samples=mc_samples_lfc)
                        lfc = torch.log2(h1 + eps_lfc) - torch.log2(h0 + eps_lfc)
                        gene_lfc[:, k] = lfc.mean(0).cpu().numpy()
                    lfcs.append(gene_lfc)
        mod.train(was)

        beta = np.concatenate(betas, axis=0)
        effect_size = np.concatenate(effects, axis=0)
        pvalue = np.concatenate(pvals, axis=0)
        if order is not None:
            beta, effect_size, pvalue = beta[order], effect_size[order], pvalue[order]
        padj = false_discovery_control(pvalue.flatten(), method="bh").reshape(pvalue.shape)
        result = {"beta": beta, "effect_size": effect_size, "pvalue": pvalue, "padj": padj,
                 "covariates": names, "cell_names": np.asarray(ad.obs_names)}
        if store_lfc:
            lfc = np.concatenate(lfcs, axis=0)
            result["lfc"] = lfc if order is None else lfc[order]
            result["genes"] = np.asarray(self.registry["var_names"])
        return result


__all__ = ["MRVI", "MRVAEModule", "DecoderZXAttention", "EncoderUZ", "EncoderXU"]
