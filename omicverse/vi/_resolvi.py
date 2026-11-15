"""ResolVI (Ergen et al.): spatial expression denoising for imaging-based spatial
transcriptomics (Xenium / MERFISH / CosMx).

Native omicverse port of ``scvi.external.RESOLVI`` (scvi-tools 1.5.2, a pyro
model).  The model decomposes each cell's observed transcript counts into a
mixture of three Poisson-rate channels:

* ``true``      the cell's own (denoised) expression, decoded from a latent ``z``;
* ``diffusion`` transcripts leaking in from spatial neighbours, a per-cell,
  per-neighbour weighted combination of the neighbours' own decoded rates;
* ``background`` a per-batch ambient-RNA profile shared by every cell.

The three channel weights, the per-neighbour diffusion weights and the latent
``z`` are amortised by small encoder networks from the cell's own expression;
a handful of global scalars (how much of each channel to expect overall, the
spatial kernel width, per-gene dispersion, the per-batch background profile)
are point estimates trained jointly with the networks (the pyro reference's
``Delta``-guided parameters: this is exactly a MAP estimate of those globals,
so here they are plain ``nn.Parameter`` with an explicit Gamma / Dirichlet /
Exponential log-prior penalty added to the loss, rather than going through a
probabilistic-programming trace).  See ``docs/`` for the derivation; the
short version is in the class docstring below.

No pyro / jax / scvi-tools import: ``torch.distributions`` only.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .._registry import register_function
from ._base import BaseVIModel, _codes
from ._nn import Classifier, DecoderSCVI, Encoder, FCLayers
from ._scvi import _flags

_EPS = 1e-6


# ---------------------------------------------------------------------------
# spatial neighbours
# ---------------------------------------------------------------------------
def spatial_knn(coords: np.ndarray, batch_codes: np.ndarray | None, n_neighbors: int):
    """``index_neighbor`` / ``distance_neighbor`` (squared Euclidean), batch-aware
    (neighbours are only searched for within the same ``batch_codes`` group, as
    scvi-tools' ResolVI ``_prepare_data``)."""
    from sklearn.neighbors import NearestNeighbors
    n = coords.shape[0]
    index = np.zeros((n, n_neighbors), dtype=np.int64)
    dist = np.full((n, n_neighbors), 1e6, dtype=np.float32)
    groups = [np.arange(n)] if batch_codes is None else [
        np.flatnonzero(batch_codes == g) for g in np.unique(batch_codes)]
    for g in groups:
        if len(g) <= n_neighbors:
            raise ValueError("each spatial batch must have more cells than n_neighbors")
        nn_ = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords[g])
        d, i = nn_.kneighbors(coords[g])
        d, i = d[:, 1:], i[:, 1:]  # drop self
        index[g] = g[i]
        dist[g] = (d ** 2).astype(np.float32)
    return index, dist


# ---------------------------------------------------------------------------
# small building blocks not already in ._nn
# ---------------------------------------------------------------------------
class _DiffusionEncoder(nn.Module):
    """Amortised per-cell ``(true, diffusion, background)`` mixture weights
    from ``log1p(x)`` (scvi-tools ResolVI's ``diffusion_encoder``, reusing the
    shape of :class:`Encoder` but with a softmax output)."""

    def __init__(self, n_input, n_cat_list=None, n_layers=2, n_hidden=128, dropout_rate=0.0,
                 use_batch_norm=False, use_layer_norm=True, diffusion_eps=0.01):
        super().__init__()
        self.trunk = FCLayers(n_input, n_hidden, n_cat_list=n_cat_list, n_layers=n_layers,
                              n_hidden=n_hidden, dropout_rate=dropout_rate,
                              use_batch_norm=use_batch_norm, use_layer_norm=use_layer_norm)
        self.head = nn.Linear(n_hidden, 3)
        self.diffusion_eps = float(diffusion_eps)

    def forward(self, x_log, *cats):
        p = torch.softmax(self.head(self.trunk(x_log, *cats)), dim=-1)
        true, diffusion, background = p.split(1, dim=-1)
        diffusion = diffusion + self.diffusion_eps  # scvi-tools quirk: not renormalised
        return true, diffusion, background


def _softplus_inv(y: float) -> float:
    y = max(float(y), 1e-6)
    return float(np.log(np.expm1(y))) if y < 20 else y


class ResolVIModule(nn.Module):
    """The ResolVI generative model + amortised guide (no pyro trace: the
    model and the guide are evaluated together in :meth:`forward`, which
    returns the loss and its terms, as the other ``ov.vi`` modules).

    Likelihood (per cell, ``i``; neighbours ``j in N(i)``, genes summed)::

        z_i ~ q(z|x_i)                                 (Gaussian encoder)
        (true_i, diff_i, bg_i) = softmax(f(log1p(x_i))) (diffusion encoder)
        w_ij = softmax_j(diffusion_map[i] - sqrt(dist_ij/median_distance))
        rate_i = true_i * decode(z_i) + sum_j w_ij * diff_i * decode(z_j)
        bg_i = bg_i * exp(library_i) * background_profile[batch_i]
        mean_i = Gamma(theta, theta / (rate_i + bg_i + eps)).rsample()  [+ bg already
                 added for the Poisson draw below: mean_i = NB-compound(rate_i) + bg_i]
        x_i ~ Poisson(mean_i)

    ``z_j`` for a neighbour ``j`` is drawn from the *same* encoder applied to
    the neighbour's own raw counts, without gradient (scvi-tools: "best
    estimate of neighbouring cells"; frozen so the reconstruction loss of
    cell ``i`` does not push around the inference network through its
    neighbours, which collapsed the latent space for the pyro model too).
    """

    def __init__(self, n_input, n_obs, n_neighbors, n_batch=0, n_labels=0, n_hidden=32,
                 n_hidden_encoder=128, n_latent=10, n_layers=2, mixture_k=50,
                 dropout_rate=0.05, dispersion="gene", gene_likelihood="nb",
                 semisupervised=False, n_cats_per_cov=None, encode_covariates=False,
                 deeply_inject_covariates=True, use_batch_norm="none", use_layer_norm="both",
                 classifier_parameters=None, prior_true_amount=1.0, prior_diffusion_amount=0.3,
                 sparsity_diffusion=3.0, background_ratio=0.1, prior_proportions_rate=10.0,
                 median_distance=1.0, diffusion_eps=0.01,
                 downsample_counts_mean=None, downsample_counts_std=1.0, n_particles=5):
        super().__init__()
        self.n_particles = int(n_particles)
        if dispersion not in ("gene", "gene-batch"):
            raise ValueError("dispersion must be 'gene' or 'gene-batch'")
        if gene_likelihood not in ("nb", "poisson"):
            raise ValueError("gene_likelihood must be 'nb' or 'poisson'")
        self.n_input, self.n_batch, self.n_labels = n_input, n_batch, n_labels
        self.n_latent, self.n_neighbors, self.n_obs = n_latent, n_neighbors, n_obs
        self.dispersion, self.gene_likelihood = dispersion, gene_likelihood
        self.semisupervised = bool(semisupervised)
        self.encode_covariates = encode_covariates
        self.downsample_counts_mean = downsample_counts_mean
        self.downsample_counts_std = float(downsample_counts_std)
        self.diffusion_eps = float(diffusion_eps)
        self.register_buffer("eps", torch.tensor(_EPS))
        self.register_buffer("median_distance", torch.tensor(float(median_distance)))
        self.register_buffer("prior_proportions_rate", torch.tensor(float(prior_proportions_rate)))
        self.register_buffer("prior_amounts", torch.tensor(
            [float(prior_true_amount), float(prior_diffusion_amount),
             float(10 * background_ratio * prior_true_amount + 1e-3)]))

        bn_enc, bn_dec = _flags(use_batch_norm)
        ln_enc, ln_dec = _flags(use_layer_norm)
        cat_list = [n_batch] + list(n_cats_per_cov or [])
        enc_cats = cat_list if encode_covariates else None

        self.z_encoder = Encoder(n_input, n_latent, n_cat_list=enc_cats, n_layers=n_layers,
                                 n_hidden=n_hidden_encoder, dropout_rate=dropout_rate,
                                 inject_covariates=deeply_inject_covariates,
                                 use_batch_norm=bn_enc, use_layer_norm=ln_enc)
        self.diffusion_encoder = _DiffusionEncoder(n_input, n_cat_list=enc_cats, n_layers=n_layers,
                                                   n_hidden=n_hidden_encoder, dropout_rate=0.0,
                                                   use_batch_norm=False, use_layer_norm=True,
                                                   diffusion_eps=diffusion_eps)
        self.decoder = DecoderSCVI(n_latent, n_input, n_cat_list=cat_list, n_layers=n_layers,
                                   n_hidden=n_hidden, inject_covariates=deeply_inject_covariates,
                                   use_batch_norm=bn_dec, use_layer_norm=ln_dec,
                                   need_dropout=False, need_r=False)

        if self.semisupervised:
            cls = dict(n_layers=0, n_hidden=128, dropout_rate=0.0, use_batch_norm=False,
                      use_layer_norm=True, logits=True)
            cls.update(classifier_parameters or {})
            self.classifier = Classifier(n_latent, n_labels=n_labels, **cls)

        # -- global point estimates (pyro's Delta-guided parameters: trained by MAP,
        # the matching prior's log-density is added to the loss in `global_log_prior`) --
        init = _softplus_inv(0.01)
        if dispersion == "gene":
            self.px_r_inv_raw = nn.Parameter(torch.full((n_input,), init))
        else:
            self.px_r_inv_raw = nn.Parameter(torch.full((n_input, n_batch), init))
        self.sparsity_diffusion_raw = nn.Parameter(torch.tensor(_softplus_inv(20.0)))
        self.true_proportion_raw = nn.Parameter(torch.tensor(_softplus_inv(5.0)))
        self.diffusion_proportion_raw = nn.Parameter(torch.tensor(_softplus_inv(3.0)))
        self.background_proportion_raw = nn.Parameter(torch.tensor(_softplus_inv(0.5)))
        self.diffuse_scale_raw = nn.Parameter(torch.tensor(0.0))  # softplus(0)=ln2, ~1
        self.background_concentration = nn.Parameter(torch.zeros(max(n_batch, 1), n_input))
        self.per_neighbor_diffusion_map = nn.Parameter(torch.zeros(n_obs, n_neighbors))

        # -- mixture-of-Gaussians prior on z (learned jointly; label-boosted if semisupervised) --
        k = n_labels if self.semisupervised else mixture_k
        self.mixture_k = k
        self.u_prior_logits = nn.Parameter(torch.ones(k))
        if self.semisupervised:
            self.u_prior_means = nn.Parameter(torch.zeros(k, n_latent))
            self.u_prior_scales_raw = nn.Parameter(torch.full((k, n_latent), -1.0))
        else:
            self.u_prior_means = nn.Parameter(torch.randn(k, n_latent))
            self.u_prior_scales_raw = nn.Parameter(torch.full((k, n_latent), -1.0))

        self.full_counts = None  # set by RESOLVI._build: (n_obs, n_input) CPU buffer
        self.train_config = None

    # ------------------------------------------------------------------
    @staticmethod
    def _cats(b):
        cat = b.get("cat")
        return () if cat is None else tuple(cat[:, i] for i in range(cat.shape[1]))

    def _theta(self, batch_index):
        r_inv = F.softplus(self.px_r_inv_raw) + 1e-4
        if self.dispersion == "gene-batch":
            r_inv = F.embedding(batch_index.reshape(-1), r_inv.t())
        return 1.0 / r_inv

    def per_gene_background(self):
        return torch.softmax(self.background_concentration, dim=-1)

    def z_sample(self, x, batch_index, cat=(), sample=True):
        x_norm = torch.log1p(x / x.mean(-1, keepdim=True).clamp_min(1e-8))
        return self.z_encoder(x_norm, batch_index, *cat, sample=sample)

    def latent_mean(self, b):
        cat = self._cats(b)
        return self.z_sample(b["x"], b["batch"], cat, sample=False)[0]

    def mixture_prior(self, labels=None):
        """``(logits, means, scales)`` of the per-cell latent mixture prior;
        with semi-supervision the logits are boosted towards the cell's own
        label (unlabeled cells, code ``n_labels``, get the unboosted prior)."""
        means = self.u_prior_means
        scales = F.softplus(self.u_prior_scales_raw) + 1e-4
        logits = self.u_prior_logits
        if self.semisupervised and labels is not None:
            onehot = torch.where(labels.reshape(-1, 1) < self.n_labels,
                                 F.one_hot(labels.clamp(max=self.n_labels - 1), self.n_labels)
                                 .float(), torch.zeros(labels.shape[0], self.n_labels,
                                                       device=labels.device))
            logits = logits.unsqueeze(0) + 10.0 * onehot
        return logits, means, scales

    def kl_latent(self, qz_m, qz_v, z, labels=None):
        logits, means, scales = self.mixture_prior(labels)
        comp = torch.distributions.Independent(torch.distributions.Normal(means, scales), 1)
        if logits.dim() == 1:
            mix = torch.distributions.MixtureSameFamily(
                torch.distributions.Categorical(logits=logits), comp)
            log_p = mix.log_prob(z)
        else:  # per-cell boosted logits: broadcast means/scales to (B, K, L)
            b = logits.shape[0]
            comp_b = torch.distributions.Independent(
                torch.distributions.Normal(means.unsqueeze(0).expand(b, -1, -1),
                                           scales.unsqueeze(0).expand(b, -1, -1)), 1)
            mix = torch.distributions.MixtureSameFamily(
                torch.distributions.Categorical(logits=logits), comp_b)
            log_p = mix.log_prob(z)
        log_q = (-0.5 * ((z - qz_m) ** 2 / qz_v + torch.log(qz_v)
                        + float(np.log(2 * np.pi)))).sum(-1)
        return log_q - log_p

    def global_log_prior(self):
        """Sum of the log-densities of the global point estimates under their
        Gamma / Exponential / Dirichlet priors (the pyro reference's
        Delta-guide contribution to the ELBO); a scalar added to the loss."""
        rate = self.prior_proportions_rate
        sparsity = F.softplus(self.sparsity_diffusion_raw) + 1e-4
        g = torch.distributions.Gamma(rate, rate / sparsity).log_prob(sparsity)
        amounts = sparsity * self.prior_amounts
        true_p = F.softplus(self.true_proportion_raw) + 1e-4
        diff_p = F.softplus(self.diffusion_proportion_raw) + 1e-4
        back_p = F.softplus(self.background_proportion_raw) + 1e-4
        g = g + torch.distributions.Gamma(rate, rate / amounts[0]).log_prob(true_p)
        g = g + torch.distributions.Gamma(rate, rate / amounts[1]).log_prob(diff_p)
        g = g + torch.distributions.Gamma(rate, rate / amounts[2]).log_prob(back_p)
        scale = F.softplus(self.diffuse_scale_raw) + 1e-4
        g = g + torch.distributions.Exponential(1.0).log_prob(scale)
        pgb = self.per_gene_background()
        g = g + torch.distributions.Dirichlet(
            5.0 * torch.ones_like(pgb), validate_args=False).log_prob(pgb).sum()
        return g, true_p, diff_p, back_p, sparsity

    # ------------------------------------------------------------------
    def forward(self, b, kl_weight: float = 1.0):
        x = b["x"]
        batch_index = b["batch"]
        cat = self._cats(b)
        idx = b["idx"].reshape(-1)
        nb_idx, nb_dist = b["neighbor_idx"], b["neighbor_dist"]
        B, G = x.shape
        K = nb_idx.shape[1]
        library = torch.log(x.sum(1, keepdim=True).clamp_min(1.0))

        if self.training and self.downsample_counts_mean is not None:
            n = int(np.random.lognormal(self.downsample_counts_mean, self.downsample_counts_std)) + 10
            tot = x.sum(1, keepdim=True).clamp_min(1.0)
            x_ds = torch.distributions.Multinomial(
                total_count=n, probs=(x / tot).clamp_min(0)).sample()
        else:
            x_ds = x

        true_c, diff_c, back_c = self.diffusion_encoder(torch.log1p(x), batch_index, *cat)

        kernel = 30.0 * torch.exp(-(F.softplus(self.diffuse_scale_raw) * nb_dist
                                    / self.median_distance).clamp(max=20.0)) + 1e-3
        logits_n = (self.per_neighbor_diffusion_map[idx]
                   - torch.sqrt(nb_dist / self.median_distance).clamp(max=10.0))
        w = torch.softmax(logits_n, dim=-1)                       # (B, K)
        reg_neighbor = torch.distributions.Dirichlet(
            kernel, validate_args=False).log_prob(w)               # (B,)

        mix_point = torch.cat([true_c, diff_c, back_c], -1).clamp_min(1e-6)
        reg_mixture = torch.distributions.Dirichlet(
            torch.stack([F.softplus(self.true_proportion_raw) + 1e-4,
                        F.softplus(self.diffusion_proportion_raw) + 1e-4,
                        F.softplus(self.background_proportion_raw) + 1e-4]).expand(B, -1),
            validate_args=False,
        ).log_prob(mix_point)

        bidx = batch_index.reshape(-1) if self.n_batch else x.new_zeros(B, dtype=torch.long)
        background = back_c * torch.exp(library) * self.per_gene_background()[bidx]
        theta = self._theta(batch_index)
        labels = b.get("labels") if self.semisupervised else None

        # scvi-tools' pyro reference uses `Trace_ELBO(num_particles=5, vectorize_particles=True)`:
        # the Gamma-compound NB draw and the per-cell latent are each a single reparameterised
        # sample, so a 1-particle gradient of this mixture is high-variance enough to collapse
        # onto the (cheap, cell-independent) background channel; averaging a handful of
        # independent samples here reproduces that variance reduction.
        rec_acc = kl_z_acc = z_acc = None
        for _ in range(max(1, self.n_particles)):
            qz_m, qz_v, z = self.z_sample(x_ds, batch_index, cat, sample=True)
            px_scale, _, px_rate, _ = self.decoder(z, library, batch_index, *cat)

            with torch.no_grad():
                x_n = self.full_counts[nb_idx.reshape(-1)].reshape(B, K, G)
                xn_norm = torch.log1p(x_n / x_n.mean(-1, keepdim=True).clamp_min(1e-8))
                bi_n = batch_index.reshape(B, 1).expand(B, K).reshape(-1, 1)
                cat_n = tuple(c.reshape(B, 1).expand(B, K).reshape(-1) for c in cat)
                qzm_n, qzv_n, _ = self.z_encoder(xn_norm.reshape(B * K, G), bi_n, *cat_n,
                                                 sample=False)
                zn = qzm_n + qzv_n.sqrt() * torch.randn_like(qzm_n)
                lib_n = library.expand(B, K).reshape(-1, 1)
                _, _, px_rate_n, _ = self.decoder(zn, lib_n, bi_n, *cat_n)
                px_rate_n = px_rate_n.reshape(B, K, G)

            rate = (true_c * px_rate
                   + (w.unsqueeze(-1) * diff_c.unsqueeze(-1) * px_rate_n).sum(1)).clamp_min(1e-8)
            if self.gene_likelihood == "poisson":
                mean_nb = rate
            else:
                mean_nb = torch.distributions.Gamma(
                    theta.expand_as(rate), theta.expand_as(rate) / (rate + self.eps)).rsample()
            mean_poisson = (mean_nb + background).clamp_min(1e-9)
            log_lik = torch.distributions.Poisson(mean_poisson).log_prob(x).sum(-1)
            kl_z = self.kl_latent(qz_m, qz_v, z, labels)
            rec_acc = -log_lik if rec_acc is None else rec_acc - log_lik
            kl_z_acc = kl_z if kl_z_acc is None else kl_z_acc + kl_z
            z_acc = qz_m if z_acc is None else z_acc + qz_m  # for the classifier (posterior mean)
        p = max(1, self.n_particles)
        rec, kl_z, z = rec_acc / p, kl_z_acc / p * kl_weight, z_acc / p

        # `reg_neighbor` / `reg_mixture` are bare log-densities (log p(theta), no entropy term
        # since the guide is a point estimate / Delta): maximising them means SUBTRACTING them
        # from the loss we minimise, not adding them like a KL term (whose sign is already
        # log_q - log_p). Mixing the two signs up made the optimiser flee the prior's mode
        # instead of being pulled towards it, collapsing every cell onto background ~ 1.
        kl_local = kl_z - reg_neighbor - 5.0 * reg_mixture

        g_log_prior, *_ = self.global_log_prior()
        kl_global = -g_log_prior.expand(())

        loss_cls = x.new_zeros(())
        if self.semisupervised and labels is not None:
            observed = labels != self.n_labels
            if observed.any():
                logits = self.classifier(z[observed])
                loss_cls = F.cross_entropy(logits, labels[observed].long())

        loss = torch.mean(rec + kl_local) + kl_global + 50.0 * loss_cls
        out = {"loss": loss, "reconstruction_loss": rec, "kl_local": kl_local,
              "kl_global": kl_global.detach()}
        if self.semisupervised:
            out["classification_loss"] = loss_cls.detach()
        return out


# ---------------------------------------------------------------------------
@register_function(
    aliases=["ResolVI", "resolvi", "RESOLVI", "空间去噪", "spatial denoising", "vi.RESOLVI"],
    category="space",
    description="omicverse-native ResolVI (no scvi-tools): spatial expression denoising for "
                "imaging-based spatial transcriptomics (Xenium/MERFISH/CosMx); decomposes counts "
                "into true/diffusion/background Poisson-rate channels using a spatial kNN graph.",
    examples=[
        "m = ov.vi.RESOLVI(adata, layer='counts', spatial_key='spatial', batch_key='slide')",
        "m.train()",
        "adata.obsm['X_resolvi'] = m.get_latent_representation()",
        "denoised = m.get_normalized_expression()",
    ],
    related=["space.resolvi", "vi.SCVI"],
)
class RESOLVI(BaseVIModel):
    """ResolVI (Ergen et al.): denoise imaging-based spatial transcriptomics counts
    into cell-intrinsic, spatial-diffusion and background-ambient channels.

    Arguments:
        adata: AnnData, AnnDataOOM or Dask-backed AnnData (raw counts in ``layer``).
        spatial_key: ``obsm`` key with 2-D/3-D spatial coordinates (not read into
            the model's registry from ``adata`` again after construction, so moving
            or relabelling ``adata.obsm[spatial_key]`` afterwards has no effect).
        batch_key: ``obs`` column separating physical slides (neighbours are only
            searched for within the same slide) and giving the background profile.
        labels_key, unlabeled_category: Cell-type labels for the semi-supervised
            variant (classifier on ``z``, label-boosted latent mixture prior); as
            :class:`~omicverse.vi.SCANVI`. ``None`` (default): fully unsupervised.
        n_neighbors: Spatial kNN size (>= 6; empirical priors need a few genes'
            worth of neighbours).
        mixture_k: Components of the learned Gaussian-mixture prior on ``z``
            (unsupervised only; semi-supervised uses one component per label).
        n_hidden, n_hidden_encoder, n_latent, n_layers, dropout_rate: As
            scvi-tools 1.5.2 (defaults 32, 128, 10, 2, 0.05).
        dispersion: ``'gene'`` or ``'gene-batch'``.
        gene_likelihood: ``'nb'`` or ``'poisson'``.
        downsample_counts: Multinomial count down-sampling during training
            (scvi-tools' data-augmentation; mean/std of the log library size are
            estimated from the data).
        seed: Seed of the weight initialisation.

    Example::

        m = ov.vi.RESOLVI(adata, layer="counts", spatial_key="spatial", batch_key="slide")
        m.train()
        adata.obsm["X_resolvi"] = m.get_latent_representation()
        denoised = m.get_normalized_expression()          # true, denoised expression
        props = m.get_proportions()                       # per-cell true/diffusion/background
    """

    _module_cls = ResolVIModule

    def __init__(self, adata, layer=None, spatial_key="spatial", batch_key=None,
                 labels_key=None, unlabeled_category="Unknown", categorical_covariate_keys=None,
                 genes=None, n_neighbors: int = 10, n_hidden: int = 32,
                 n_hidden_encoder: int = 128, n_latent: int = 10, n_layers: int = 2,
                 mixture_k: int = 50, dropout_rate: float = 0.05, dispersion: str = "gene",
                 gene_likelihood: str = "nb", downsample_counts: bool = True,
                 prior_true_amount: float = 1.0, prior_diffusion_amount: float = 0.3,
                 sparsity_diffusion: float = 3.0, background_ratio: float | None = None,
                 median_distance: float | None = None, prior_proportions_rate: float = 10.0,
                 encode_covariates: bool = False, deeply_inject_covariates: bool = True,
                 use_batch_norm: str = "none", use_layer_norm: str = "both",
                 semisupervised: bool | None = None, seed: int | None = 0, **module_kwargs):
        if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int) or n_neighbors < 6:
            raise ValueError("n_neighbors must be an int >= 6")
        self.unlabeled_category_ = unlabeled_category
        super().__init__(adata, layer=layer, batch_key=batch_key, labels_key=labels_key,
                         categorical_covariate_keys=categorical_covariate_keys,
                         continuous_covariate_keys=None, genes=genes)
        if labels_key is not None:
            cats = [c for c in self.registry["label_categories"]
                    if str(c) != str(unlabeled_category)]
            self.registry["label_categories"] = cats
            self.registry["unlabeled_category"] = unlabeled_category
        self.semisupervised_ = bool(semisupervised) if semisupervised is not None \
            else labels_key is not None
        self.registry["spatial_key"] = spatial_key
        self.registry["n_neighbors"] = n_neighbors

        coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
        batch_codes = (_codes(adata.obs[batch_key].values, self.registry["batch_categories"],
                             batch_key) if batch_key else None)
        index_neighbor, distance_neighbor = spatial_knn(coords, batch_codes, n_neighbors)
        self.registry["index_neighbor"] = index_neighbor
        self.registry["distance_neighbor"] = distance_neighbor

        full_counts, dists = self._materialize_counts(adata, distance_neighbor)
        if background_ratio is None:
            n_small = max(full_counts.shape[1] // 50, 1)
            order = full_counts.sum(0).argsort()[:n_small]
            background_ratio = float(np.mean(
                full_counts[:, order].mean(1) / np.maximum(full_counts.mean(1), _EPS)))
        if median_distance is None:
            median_distance = float(np.median(np.partition(dists, 5, axis=1)[:, 5]))
        downsample_counts_mean = downsample_counts_std = None
        if downsample_counts:
            log_lib = np.log1p(full_counts.sum(1))
            downsample_counts_mean = float(np.median(log_lib))
            downsample_counts_std = float(np.std(log_lib)) or 1.0

        kw = dict(n_hidden=n_hidden, n_hidden_encoder=n_hidden_encoder, n_latent=n_latent,
                  n_layers=n_layers, mixture_k=mixture_k, dropout_rate=dropout_rate,
                  dispersion=dispersion, gene_likelihood=gene_likelihood,
                  semisupervised=self.semisupervised_, prior_true_amount=prior_true_amount,
                  prior_diffusion_amount=prior_diffusion_amount,
                  sparsity_diffusion=sparsity_diffusion, background_ratio=background_ratio,
                  prior_proportions_rate=prior_proportions_rate, median_distance=median_distance,
                  encode_covariates=encode_covariates,
                  deeply_inject_covariates=deeply_inject_covariates,
                  use_batch_norm=use_batch_norm, use_layer_norm=use_layer_norm,
                  downsample_counts_mean=downsample_counts_mean,
                  downsample_counts_std=downsample_counts_std or 1.0, **module_kwargs)
        self._build(seed=seed, full_counts=full_counts, **kw)

    def _materialize_counts(self, adata, distance_neighbor):
        """Read the full raw-count matrix once (CPU, dense float32): the lookup
        table used to gather neighbour expression at train/inference time.
        Bounded by ``n_obs * n_vars`` (imaging panels: 10s-1000s of genes), not
        ``n_obs * n_neighbors * n_vars``, and built from the model's own data
        source so it works for AnnDataOOM / Dask inputs too."""
        src = self._source(adata, device="cpu")
        out = np.empty((src.n_obs, self.n_vars), dtype=np.float32)
        for b in src.iterate(None, batch_size=8192, shuffle="none"):
            idx = b["idx"].numpy()
            out[idx] = b["x"].numpy()
        return out, distance_neighbor

    def _extra_obs(self, adata):
        if adata is not self.adata:
            raise ValueError("RESOLVI's spatial neighbour graph is fixed to the AnnData it was "
                             "constructed with; pass indices= instead of a different adata=")
        return {"neighbor_idx": self.registry["index_neighbor"].astype(np.int64),
               "neighbor_dist": self.registry["distance_neighbor"].astype(np.float32)}

    def _label_codes(self, adata):
        if not getattr(self, "semisupervised_", False) or not self.registry["labels_key"]:
            return np.zeros(adata.n_obs, np.int64)
        r = self.registry
        import pandas as pd
        vals = pd.Series(np.asarray(adata.obs[r["labels_key"]].values)).astype(object)
        unl = vals.astype(str) == str(r.get("unlabeled_category", self.unlabeled_category_))
        codes = np.full(len(vals), len(r["label_categories"]), dtype=np.int64)
        if (~unl).any():
            codes[~unl.values] = _codes(vals[~unl].values, r["label_categories"], r["labels_key"])
        return codes

    def _build(self, seed=None, full_counts=None, **kw):
        kw = {k: v for k, v in kw.items() if k != "seed"}
        self._module_kwargs = kw
        r = self.registry
        self._module_args = dict(
            n_input=self.n_vars, n_obs=self.adata.n_obs, n_neighbors=r["n_neighbors"],
            n_batch=self.n_batch, n_labels=len(r["label_categories"]), **kw)
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(seed)
            self.module = self._module_cls(**self._module_args)
        if full_counts is not None:
            self.module.full_counts = torch.from_numpy(full_counts)
        self.module.to(self._device)

    def to_device(self, device):
        super().to_device(device)
        if self.module is not None and self.module.full_counts is not None:
            self.module.full_counts = self.module.full_counts.to(self._device)
        return self

    def _init_params(self):
        return dict(self._module_kwargs)

    def save(self, dir_path, overwrite=False, save_anndata=False):
        path = super().save(dir_path, overwrite=overwrite, save_anndata=save_anndata)
        np.savez(f"{dir_path}/resolvi_neighbors.npz",
                index_neighbor=self.registry["index_neighbor"],
                distance_neighbor=self.registry["distance_neighbor"],
                full_counts=self.module.full_counts.numpy())
        return path

    @classmethod
    def load(cls, dir_path, adata=None, device="auto"):
        m = super().load(dir_path, adata=adata, device=device)
        z = np.load(f"{dir_path}/resolvi_neighbors.npz")
        m.registry["index_neighbor"] = z["index_neighbor"]
        m.registry["distance_neighbor"] = z["distance_neighbor"]
        m.module.full_counts = torch.from_numpy(z["full_counts"]).to(m._device)
        # `BaseVIModel.load()` builds a bare instance via `BaseVIModel.__init__` (bypassing
        # `RESOLVI.__init__`), so attributes it does not know about must be restored here.
        m.semisupervised_ = m.module.semisupervised
        m.unlabeled_category_ = m.registry.get("unlabeled_category", "Unknown")
        return m

    # ------------------------------------------------------------------
    def get_latent_representation(self, adata=None, indices=None, give_mean: bool = True,
                                  batch_size: int | None = None, return_dist: bool = False):
        """Posterior mean (``give_mean``) or a sample of ``z`` per cell."""
        mod = self.module

        def fn(b):
            qm, qv, z = mod.z_sample(b["x"], b["batch"], mod._cats(b), sample=not give_mean)
            return torch.cat([qm, qv], 1) if return_dist else (qm if give_mean else z)

        out = self._collect(fn, adata, indices, batch_size)
        if return_dist:
            k = out.shape[1] // 2
            return out[:, :k], out[:, k:]
        return out

    def get_proportions(self, adata=None, indices=None, batch_size: int | None = None):
        """Per-cell ``(true, diffusion, background)`` mixture weights (DataFrame)."""
        import pandas as pd
        mod = self.module

        def fn(b):
            t, d, g = mod.diffusion_encoder(torch.log1p(b["x"]), b["batch"], *mod._cats(b))
            return torch.cat([t, d, g], -1)

        out = self._collect(fn, adata, indices, batch_size)
        ad = self.adata if adata is None else adata
        idx = ad.obs_names if indices is None else ad.obs_names[np.asarray(indices)]
        return pd.DataFrame(out, index=idx, columns=["true", "diffusion", "background"])

    def get_normalized_expression(self, adata=None, indices=None, library_size=1,
                                  batch_size: int | None = None, n_samples: int = 1,
                                  return_numpy: bool = False):
        """Denoised, library-size normalised "true" expression (no background /
        diffusion contribution): ``px_scale * library_size`` from the decoder on
        the posterior mean of ``z`` (``library_size='latent'``: the model's own
        rate, before the true/diffusion/background split)."""
        mod = self.module

        def fn(b):
            qm, qv, _ = mod.z_sample(b["x"], b["batch"], mod._cats(b), sample=False)
            z = qm if n_samples == 1 else qm + qv.sqrt() * torch.randn(
                (n_samples,) + qm.shape, device=qm.device)
            library = torch.log(b["x"].sum(1, keepdim=True).clamp_min(1.0))
            px_scale, _, px_rate, _ = mod.decoder(z, library, b["batch"], *mod._cats(b))
            return px_rate if library_size == "latent" else px_scale * library_size

        out = self._collect(fn, adata, indices, batch_size)
        if return_numpy:
            return out
        import pandas as pd
        ad = self.adata if adata is None else adata
        idx = ad.obs_names if indices is None else ad.obs_names[np.asarray(indices)]
        return pd.DataFrame(out, index=idx, columns=self.registry["var_names"])

    def get_corrected_counts(self, adata=None, indices=None, batch_size: int | None = None,
                             return_numpy: bool = False):
        """``model_corrected`` of the reference: the observed-scale "true"
        reconstruction (``true=1``, ``diffusion=background=0``) — ``px_rate`` of
        the decoder at the cell's own library size, i.e. the denoised counts."""
        return self.get_normalized_expression(adata, indices, library_size="latent",
                                              batch_size=batch_size, return_numpy=return_numpy)

    def predict(self, adata=None, indices=None, soft: bool = False,
               batch_size: int | None = None):
        """Predicted cell-type labels (semi-supervised models only)."""
        if not self.semisupervised_:
            raise RuntimeError("predict() needs a semi-supervised model (labels_key=...)")
        import pandas as pd
        mod = self.module

        def fn(b):
            qm, _, _ = mod.z_sample(b["x"], b["batch"], mod._cats(b), sample=False)
            return torch.softmax(mod.classifier(qm), -1)

        probs = self._collect(fn, adata, indices, batch_size)
        cats = self.registry["label_categories"]
        if soft:
            ad = self.adata if adata is None else adata
            idx = ad.obs_names if indices is None else ad.obs_names[np.asarray(indices)]
            return pd.DataFrame(probs, index=idx, columns=cats)
        return np.asarray(cats, dtype=object)[probs.argmax(1)]

    def train(self, max_epochs=None, batch_size: int = 512, seed: int = 0, **kwargs):
        """Train (scvi-tools ResolVI defaults: batch size 512, 50 epochs, no
        held-out validation split — the model keeps a per-cell global parameter
        table, ``per_neighbor_diffusion_map``, so ``train_size`` is always 1.0)."""
        if max_epochs is None:
            max_epochs = 50
        kwargs.setdefault("plan_kwargs", {}).setdefault("lr", 3e-3)
        # the CUDA-graph / torch.compile fast paths strip "idx" from the minibatch dict
        # (other ov.vi models never need a cell's row index inside forward()); RESOLVI
        # indexes `per_neighbor_diffusion_map` by it, so both fast paths are disabled here.
        kwargs.setdefault("cuda_graph", False)
        kwargs.setdefault("compile", False)
        return super().train(max_epochs=max_epochs, batch_size=batch_size, seed=seed,
                             train_size=1.0, validation_size=0.0, **kwargs)


__all__ = ["RESOLVI", "ResolVIModule", "spatial_knn"]
