"""DRVI (Disentangled Representation Variational Inference; Moinfar et al. 2023/2024):
a VAE whose decoder treats every latent dimension as an *independent* generator of
gene expression.

Native port of ``drvi-py``'s (``theislab/drvi``) ``DRVIModule`` / ``DecoderDRVI``
(``split_method='split_map'``, ``split_aggregation='logsumexp'``,
``decoder_reuse_weights='everywhere'`` — the paper's defaults): each latent
dimension ``z_i`` is mapped through its own small learned linear embedding
(:attr:`SplitDecoderDRVI.split_w`) into the shared decoder's input width, the
*same* decoder MLP is applied independently to every dimension's embedding
(so dimensions cannot share decoder capacity), and the ``n_latent`` resulting
per-gene logits are combined with log-sum-exp — the average of the per-dimension
contributions in probability space — before a softmax over genes and scaling by
the library size.  Because each dimension's contribution is computed and
combined independently of the others, the decoder is *additive*: ablating one
dimension removes exactly its own contribution to every gene, which is what
makes the per-dimension traversals in :func:`DRVI.differential_vars` meaningful.

Count likelihoods: ``'pnb'`` (DRVI's own log-mean-parameterised NB, the default),
``'nb'``, ``'poisson'`` (all three ``drvi-py``'s), plus ``'zinb'`` (not in
``drvi-py``, added here reusing ``ov.vi``'s existing zero-inflation head since the
spec asks for it; it is not part of the DRVI paper and is not covered by the
parity benchmark).

Reference: ``drvi-py`` ``drvi.scvi_tools_based.module._drvi.DRVIModule`` and
``drvi.scvi_tools_based.nn._base_components.DecoderDRVI``.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .._registry import register_function
from ._base import BaseVIModel
from ._distributions import (
    kl_normal_standard, log_nb_positive, log_pnb_positive, log_poisson, log_zinb_positive,
)
from ._nn import Encoder, FCLayers
from ._priors import make_prior


class SplitDecoderDRVI(nn.Module):
    """Additive per-latent-dimension decoder (``drvi-py``'s ``DecoderDRVI`` with
    ``split_method='split_map'``, ``n_split=n_latent``, ``reuse_weights='everywhere'``).

    Every one of the ``n_split`` splits of ``z`` (by default one split per latent
    dimension) is mapped by its own learned linear embedding
    (:attr:`split_w`, shape ``(n_split, split_size, n_latent)``) into a vector
    the width of the (shared) decoder's input, so the *same* decoder MLP can be
    applied to every split by folding the split axis into the batch axis.  The
    ``n_split`` resulting per-gene logits are aggregated with
    :attr:`split_aggregation` (``'logsumexp'`` by default: the dimensions
    combine additively in probability space) before the output heads.
    """

    def __init__(self, n_latent: int, n_output: int, n_cats_per_cov=(), n_hidden: int = 128,
                 n_layers: int = 2, dropout_rate: float = 0.0, n_split: int | None = None,
                 split_aggregation: str = "logsumexp", use_layer_norm: bool = True,
                 use_batch_norm: bool = False, activation_fn=nn.ELU, dispersion: str = "gene",
                 need_dropout: bool = False, inject_covariates: bool = True):
        super().__init__()
        n_split = n_latent if n_split in (None, -1) else int(n_split)
        if n_latent % n_split != 0:
            raise ValueError(f"n_latent={n_latent} must be a multiple of n_split={n_split}")
        self.n_split = n_split
        self.split_size = n_latent // n_split
        if split_aggregation not in ("logsumexp", "sum", "max"):
            raise ValueError("split_aggregation must be 'logsumexp', 'sum' or 'max'")
        self.split_aggregation = split_aggregation
        self.dispersion = dispersion
        # drvi-py `split_transformation_weight`, scaled the same way (1/sqrt(split_size))
        self.split_w = nn.Parameter(torch.randn(n_split, self.split_size, n_latent)
                                     / float(self.split_size) ** 0.5)
        cat_list = list(n_cats_per_cov)
        self.shared = FCLayers(n_latent, n_hidden, n_cat_list=cat_list, n_layers=n_layers,
                               n_hidden=n_hidden, dropout_rate=dropout_rate,
                               use_batch_norm=use_batch_norm, use_layer_norm=use_layer_norm,
                               activation_fn=activation_fn, inject_covariates=inject_covariates)
        self.mean_head = nn.Linear(n_hidden, n_output)
        if dispersion == "gene":
            self.px_r = nn.Parameter(torch.randn(n_output))
        elif dispersion == "gene-cell":
            self.r_head = nn.Linear(n_hidden, n_output)
        else:
            raise ValueError("dispersion must be 'gene' or 'gene-cell'")
        self.dropout_head = nn.Linear(n_hidden, n_output) if need_dropout else None

    def _aggregate(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(batch, n_split, n_output)`` -> ``(batch, n_output)``."""
        s = x.shape[1]
        if self.split_aggregation == "logsumexp":
            return torch.logsumexp(x, dim=1) - math.log(s)
        if self.split_aggregation == "sum":
            return x.sum(1) / s
        return x.amax(1)

    def forward(self, z: torch.Tensor, *cats):
        b, n_latent = z.shape
        s = self.n_split
        z_r = z.reshape(b, s, self.split_size)
        # drvi-py: einsum("bsd,sdn->bsn", z, split_transformation_weight)
        z_s = torch.einsum("bsd,sdn->bsn", z_r, self.split_w)
        h = z_s.reshape(b * s, n_latent)
        cats_r = tuple(c.reshape(-1, 1).repeat_interleave(s, dim=0) if c is not None else None
                       for c in cats)
        h = self.shared(h, *cats_r)
        h = h.reshape(b, s, -1)
        logit = self._aggregate(self.mean_head(h))
        if self.dispersion == "gene":
            r = self.px_r
        else:
            r = self._aggregate(self.r_head(h))
        dropout = self._aggregate(self.dropout_head(h)) if self.dropout_head is not None else None
        return logit, r, dropout


class DRVIModule(nn.Module):
    """DRVI's generative model and amortised posterior (``drvi-py``'s ``DRVIModule``).

    The encoder is a standard Gaussian VAE encoder (``ov.vi``'s shared
    :class:`~omicverse.vi._nn.Encoder`, architecturally identical to
    ``drvi-py``'s: two 128-wide hidden layers, ``var_eps=1e-4``, ELU,
    layer-norm, ``N(0, I)`` prior with an analytic KL).  The decoder is
    :class:`SplitDecoderDRVI`.  Library size is the raw per-cell count sum
    (not log-transformed, as in ``drvi-py`` — all its noise models normalise
    with a softmax over genes rather than a decoded scale)."""

    def __init__(self, n_input: int, n_batch: int = 0, n_hidden: int = 128, n_latent: int = 32,
                 n_layers: int = 2, n_cats_per_cov=None, dropout_rate: float = 0.0,
                 encoder_dropout_rate: float = 0.1, dispersion: str = "gene",
                 gene_likelihood: str = "pnb", n_split_latent: int | None = -1,
                 split_aggregation: str = "logsumexp", encode_covariates: bool = False,
                 deeply_inject_covariates: bool = False, prior: str = "normal"):
        super().__init__()
        if gene_likelihood not in ("pnb", "nb", "zinb", "poisson"):
            raise ValueError("gene_likelihood must be 'pnb', 'nb', 'zinb' or 'poisson'")
        self.n_input, self.n_batch, self.n_latent = n_input, n_batch, n_latent
        self.gene_likelihood = gene_likelihood
        self.dispersion = dispersion
        self.encode_covariates = encode_covariates
        cat_list = [n_batch] + list(n_cats_per_cov or [])
        enc_cats = cat_list if encode_covariates else None
        self.z_encoder = Encoder(n_input, n_latent, n_cat_list=enc_cats, n_layers=n_layers,
                                 n_hidden=n_hidden, dropout_rate=encoder_dropout_rate,
                                 distribution="normal", var_eps=1e-4,
                                 inject_covariates=deeply_inject_covariates,
                                 use_batch_norm=False, use_layer_norm=True, activation_fn=nn.ELU)
        self.decoder = SplitDecoderDRVI(n_latent, n_input, n_cats_per_cov=cat_list,
                                        n_hidden=n_hidden, n_layers=n_layers,
                                        dropout_rate=dropout_rate, n_split=n_split_latent,
                                        split_aggregation=split_aggregation,
                                        use_layer_norm=True, use_batch_norm=False,
                                        activation_fn=nn.ELU, dispersion=dispersion,
                                        need_dropout=gene_likelihood == "zinb",
                                        inject_covariates=deeply_inject_covariates)
        self.prior = make_prior(prior, n_latent)
        self.train_config = None

    @staticmethod
    def _cats(b):
        cat = b.get("cat")
        return () if cat is None else tuple(cat[:, i] for i in range(cat.shape[1]))

    def inference(self, x, batch_index, cat=(), sample=True):
        x_ = torch.log1p(x)  # drvi-py `preprocess_count_data` (library_normalization='none')
        enc_cats = (batch_index, *cat) if self.encode_covariates else ()
        qm, qv, z = self.z_encoder(x_, *enc_cats, sample=sample)
        library = x.sum(1, keepdim=True)
        return {"qz_m": qm, "qz_v": qv, "z": z, "library": library}

    def generative(self, z, library, batch_index, cat=()):
        logit, r, dropout = self.decoder(z, batch_index, *cat)
        log_scale = logit - torch.logsumexp(logit, dim=-1, keepdim=True)  # log softmax
        return {"log_scale": log_scale, "r": r, "dropout": dropout, "library": library}

    def log_likelihood(self, x, gen) -> torch.Tensor:
        """Per-cell log p(x | z) (summed over genes)."""
        log_scale, r = gen["log_scale"].float(), gen["r"].float()
        if self.gene_likelihood == "pnb":
            # drvi-py `LogNegativeBinomialNoiseModel`: log(theta) = 1 + r (no exp)
            log_mu = torch.log(gen["library"].clamp(min=1)) + log_scale
            log_theta = 1.0 + r
            return log_pnb_positive(x, log_mu, log_theta).sum(-1)
        mu = gen["library"] * log_scale.exp()
        theta = torch.exp(1.0 + r)  # drvi-py `NegativeBinomialNoiseModel`: theta = exp(1 + r)
        if self.gene_likelihood == "nb":
            return log_nb_positive(x, mu, theta).sum(-1)
        if self.gene_likelihood == "zinb":
            return log_zinb_positive(x, mu, theta, gen["dropout"].float()).sum(-1)
        return log_poisson(x, mu).sum(-1)  # poisson

    def forward(self, b, kl_weight: float = 1.0):
        cat = self._cats(b)
        inf = self.inference(b["x"], b["batch"], cat)
        gen = self.generative(inf["z"], inf["library"], b["batch"], cat)
        rec = -self.log_likelihood(b["x"], gen)
        kl_z = self.prior.kl(inf["qz_m"].float(), inf["qz_v"].float(), inf["z"].float())
        loss = torch.mean(rec + kl_weight * kl_z)
        return {"loss": loss, "reconstruction_loss": rec, "kl_local": kl_z,
                "kl_global": torch.zeros((), device=rec.device)}

    def latent_mean(self, b):
        return self.inference(b["x"], b["batch"], self._cats(b), sample=False)["qz_m"]

    def loss_terms(self, b, inf, gen, kl_weight):  # kept for symmetry with VAEModule
        rec = -self.log_likelihood(b["x"], gen)
        kl_z = self.prior.kl(inf["qz_m"].float(), inf["qz_v"].float(), inf["z"].float())
        loss = torch.mean(rec + kl_weight * kl_z)
        return {"loss": loss, "reconstruction_loss": rec, "kl_local": kl_z,
                "kl_global": torch.zeros((), device=rec.device)}


@register_function(
    aliases=["DRVI", "disentangled representation VI", "vi.DRVI"],
    category="single",
    description="omicverse-native DRVI (no drvi-py/scvi-tools): a VAE with an additive, "
                "per-latent-dimension decoder (log-sum-exp aggregation) that disentangles "
                "biological variation into separate latent dimensions; pnb/nb/zinb/poisson "
                "count likelihoods; trains on in-memory AnnData, AnnDataOOM and Dask-backed "
                "AnnData.",
    examples=[
        "m = ov.vi.DRVI(adata, layer='counts', batch_key='batch', n_latent=32)",
        "m.train()",
        "adata.obsm['X_drvi'] = m.get_latent_representation()",
        "effects, genes = m.differential_vars(dim=3, n_genes=20)",
    ],
    related=["vi.SCVI", "single.drvi"],
)
class DRVI(BaseVIModel):
    """DRVI: disentangled representation learning with an additive latent decoder.

    Arguments:
        adata, layer, batch_key, labels_key, categorical_covariate_keys,
        continuous_covariate_keys, genes: As :class:`~omicverse.vi.SCVI`.
        n_hidden, n_layers: Width / depth of the (shared) encoder and decoder MLPs
            (``drvi-py`` defaults: two 128-wide layers).
        n_latent: Number of latent dimensions (default 32, as the DRVI paper).
        n_split_latent: Number of independent decoder "views" of the latent code
            (``-1`` / ``None``: one per latent dimension — the paper's setting
            and the only one needed for disentanglement).
        split_aggregation: How the per-split decoder outputs are combined
            (``'logsumexp'`` default: additive in probability space; also
            ``'sum'``, ``'max'``).
        dispersion: ``'gene'`` or ``'gene-cell'`` NB/ZINB/pnb dispersion.
        gene_likelihood: ``'pnb'`` (default, ``drvi-py``'s own log-mean NB),
            ``'nb'``, ``'zinb'`` (not in ``drvi-py``) or ``'poisson'``.
        dropout_rate, encoder_dropout_rate: Decoder / encoder dropout
            (``drvi-py`` defaults 0.0 / 0.1).
        encode_covariates, deeply_inject_covariates: As ``drvi-py`` (both
            default ``False``).
        seed: Weight-initialisation seed.

    Example::

        m = ov.vi.DRVI(adata, layer="counts", batch_key="batch", n_latent=32)
        m.train()
        adata.obsm["X_drvi"] = m.get_latent_representation()
    """

    _module_cls = DRVIModule

    def __init__(self, adata, layer=None, batch_key=None, labels_key=None,
                 categorical_covariate_keys=None, continuous_covariate_keys=None, genes=None,
                 n_hidden: int = 128, n_layers: int = 2, n_latent: int = 32,
                 n_split_latent: int | None = -1, split_aggregation: str = "logsumexp",
                 dispersion: str = "gene", gene_likelihood: str = "pnb",
                 dropout_rate: float = 0.0, encoder_dropout_rate: float = 0.1,
                 encode_covariates: bool = False, deeply_inject_covariates: bool = False,
                 seed: int | None = 0, **module_kwargs):
        super().__init__(adata, layer=layer, batch_key=batch_key, labels_key=labels_key,
                         categorical_covariate_keys=categorical_covariate_keys,
                         continuous_covariate_keys=continuous_covariate_keys, genes=genes)
        kw = dict(n_hidden=n_hidden, n_layers=n_layers, n_latent=n_latent,
                  n_split_latent=n_split_latent, split_aggregation=split_aggregation,
                  dispersion=dispersion, gene_likelihood=gene_likelihood,
                  dropout_rate=dropout_rate, encoder_dropout_rate=encoder_dropout_rate,
                  encode_covariates=encode_covariates,
                  deeply_inject_covariates=deeply_inject_covariates, **module_kwargs)
        self._build(seed=seed, **kw)

    def _build(self, seed=None, **kw):
        kw = {k: v for k, v in kw.items() if k != "seed"}
        self._module_kwargs = kw
        r = self.registry
        self._module_args = dict(n_input=self.n_vars, n_batch=self.n_batch,
                                 n_cats_per_cov=[len(c) for c in r["cat_categories"]], **kw)
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(seed)
            self.module = self._module_cls(**self._module_args)
        self.module.to(self._device)

    def _init_params(self):
        return dict(self._module_kwargs)

    # ------------------------------------------------------------------
    def get_latent_representation(self, adata=None, indices=None, give_mean: bool = True,
                                  batch_size: int | None = None):
        """Posterior mean (``give_mean=True``) or a sample of ``z`` per cell."""
        mod = self.module

        def fn(b):
            inf = mod.inference(b["x"], b["batch"], mod._cats(b), sample=False)
            if give_mean:
                return inf["qz_m"]
            return inf["qz_m"] + inf["qz_v"].sqrt() * torch.randn_like(inf["qz_m"])

        return self._collect(fn, adata, indices, batch_size)

    def get_normalized_expression(self, adata=None, indices=None, library_size=1,
                                  batch_size: int | None = None, return_numpy: bool | None = None):
        """Denoised, library-size-normalised expression ``softmax(decoder logit) *
        library_size`` (the additive combination of every latent dimension's
        contribution)."""
        mod = self.module

        def fn(b):
            cat = mod._cats(b)
            inf = mod.inference(b["x"], b["batch"], cat, sample=False)
            gen = mod.generative(inf["qz_m"], inf["library"], b["batch"], cat)
            return gen["log_scale"].exp() * library_size

        out = self._collect(fn, adata, indices, batch_size)
        if return_numpy:
            return out
        import pandas as pd
        ad = self.adata if adata is None else adata
        obs_names = ad.obs_names if indices is None else ad.obs_names[np.asarray(indices)]
        return pd.DataFrame(out, index=obs_names, columns=self.registry["var_names"])

    # ------------------------------------------------------------------
    # interpretability: latent traversal / per-dimension gene effects
    # ------------------------------------------------------------------
    def latent_traverse(self, span: float = 5.0, n_samples: int = 200, dims=None,
                        library_size: float = 1e4, batch_size: int | None = None, seed: int = 0):
        """Per-dimension effect on (log) normalised expression: for each latent
        dimension ``i``, decode ``n_samples`` baselines with every other
        dimension drawn from ``N(0, I)`` noise and dimension ``i`` set to
        ``+/- span``, and return the mean ``log(effect@+span) - log(effect@-span)``
        across samples — the per-dimension analogue of ``drvi-py``'s
        ``make_traverse_adata`` (its systematic latent traversal decoded through
        the model), collapsed to the two extremes since the decoder is additive
        in each dimension (the effect of a dimension does not depend on where
        the others are set, up to the decoder's non-linearity).

        Arguments:
            span: Latent value of the "on" / "off" (``-span``) traversal points
                (``drvi-py`` traverses the observed min/max of each dimension;
                a fixed multiple of the standard-normal prior's scale is used
                here since no posterior statistics are required).
            n_samples: Random baselines averaged per dimension.
            dims: Dimensions to traverse (default: all ``n_latent``).
            library_size: Library size the effect is decoded at.
            seed: RNG seed for the baselines (reproducible traversal).

        Returns:
            ``(n_latent_or_len(dims), n_genes)`` array: natural-log fold change of
            the normalised expression between the ``+span`` and ``-span`` points
            of each dimension, averaged over ``n_samples`` random baselines.
        """
        self._check_trained()
        mod = self.module
        n_latent = mod.n_latent
        dims = np.arange(n_latent) if dims is None else np.asarray(dims)
        device = self.device
        g = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(n_samples, n_latent, generator=g).to(device)
        batch0 = torch.zeros(n_samples, dtype=torch.long, device=device)
        n_cat = len(self.registry["cat_categories"])
        cats0 = tuple(torch.zeros(n_samples, dtype=torch.long, device=device)
                     for _ in range(n_cat))
        lib = torch.full((n_samples, 1), float(library_size), device=device)
        was = mod.training
        mod.eval()
        out = np.empty((len(dims), self.n_vars), dtype=np.float32)
        with torch.no_grad():
            for k, d in enumerate(dims):
                lo, hi = noise.clone(), noise.clone()
                lo[:, d], hi[:, d] = -span, span
                gl = mod.generative(lo, lib, batch0, cats0)["log_scale"]
                gh = mod.generative(hi, lib, batch0, cats0)["log_scale"]
                out[k] = (gh - gl).mean(0).cpu().numpy()
        mod.train(was)
        return out

    def differential_vars(self, dim: int, n_genes: int = 20, **kwargs):
        """Top genes affected by latent dimension ``dim`` (``drvi-py``'s
        per-dimension "differential vars" of a latent traversal): the genes with
        the largest ``|effect|`` from :meth:`latent_traverse`, signed by the
        direction of the effect.

        Returns:
            ``(genes, effects)``: gene names (highest ``|effect|`` first) and
            their signed log-fold-change effect.
        """
        effect = self.latent_traverse(dims=[dim], **kwargs)[0]
        order = np.argsort(-np.abs(effect))[:n_genes]
        names = np.asarray(self.registry["var_names"])[order]
        return names, effect[order]

    def dimension_gene_effects(self, n_genes: int = 20, **kwargs):
        """:meth:`differential_vars` for every latent dimension.

        Returns:
            dict ``{dim: (genes, effects)}`` for ``dim in range(n_latent)``.
        """
        effects = self.latent_traverse(**kwargs)
        out = {}
        for d in range(effects.shape[0]):
            order = np.argsort(-np.abs(effects[d]))[:n_genes]
            out[d] = (np.asarray(self.registry["var_names"])[order], effects[d, order])
        return out


__all__ = ["DRVI", "DRVIModule", "SplitDecoderDRVI"]
