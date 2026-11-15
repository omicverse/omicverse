"""SysVI (Hrovatin et al., 2023): a cVAE for integrating multiple *systems*
(species, protocol, modality, organoid vs. tissue...) on already-normalized
expression (Gaussian likelihood), with an optional VampPrior and a latent
cycle-consistency loss that explicitly discourages the encoder from using
system identity as a shortcut.

Ported natively from ``scvi.external.SysVI`` (scvi-tools 1.5.2): same
architecture, defaults and loss (including its two numerical quirks, kept for
parity — the "variance" argument of ``GaussianNLLLoss`` is really the normal
*scale*, and the VampPrior mixture is computed per latent dimension rather
than jointly, see :class:`omicverse.vi._priors.VampPrior`). ``batch_key`` here
is the *system* covariate of the original paper (corrected via conditioning
**and** the cycle loss); ``categorical_covariate_keys`` /
``continuous_covariate_keys`` are weaker covariates used only to condition
the cVAE, not cycle-corrected.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .._registry import register_function
from ._base import BaseVIModel
from ._nn import FCLayers
from ._priors import VampPrior, make_prior


class _VarEncoder(nn.Module):
    """Strictly-positive variance head (``scvi.external.sysvi`` ``VarEncoder``):
    softplus with pre-activation clipping, ``'sample_feature'`` (a linear head
    on the shared hidden state) or ``'feature'`` (one constant per output,
    independent of the input)."""

    def __init__(self, n_input: int, n_output: int, mode: str = "feature"):
        super().__init__()
        if mode not in ("sample_feature", "feature"):
            raise ValueError("mode must be 'sample_feature' or 'feature'")
        self.mode = mode
        if mode == "sample_feature":
            self.encoder = nn.Linear(n_input, n_output)
        else:
            self.var_param = nn.Parameter(torch.zeros(1, n_output))
        self.activation = nn.Softplus()
        self._clip = math.log(torch.finfo(torch.get_default_dtype()).max) - 1e-4

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        v = self.encoder(h) if self.mode == "sample_feature" else \
            self.var_param.expand(h.shape[0], -1)
        v = torch.clamp(v, min=-self._clip, max=self._clip)
        v = self.activation(v)
        return torch.nan_to_num(v)


class GaussianBlock(nn.Module):
    """``FCLayers`` + a Gaussian (mean, variance) head; used as both the
    encoder and the decoder of :class:`SysVI` (``scvi.external.sysvi``
    ``EncoderDecoder``). Call signature matches
    :class:`omicverse.vi._nn.Encoder` (``(q_m, q_v, z)``), so it plugs
    directly into :class:`~omicverse.vi._priors.VampPrior`."""

    def __init__(self, n_input: int, n_output: int, n_cat_list, n_cont: int,
                 n_hidden: int = 256, n_layers: int = 2, dropout_rate: float = 0.1,
                 var_mode: str = "feature"):
        super().__init__()
        self.net = FCLayers(n_input, n_hidden, n_cat_list=n_cat_list, n_cont=n_cont,
                            n_layers=n_layers, n_hidden=n_hidden, dropout_rate=dropout_rate,
                            use_batch_norm=True, use_layer_norm=False)
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = _VarEncoder(n_hidden, n_output, mode=var_mode)

    def forward(self, x, *cats, cont=None, sample: bool = False):
        h = self.net(x, *cats, cont=cont)
        q_m = torch.nan_to_num(self.mean_encoder(h))
        q_v = self.var_encoder(h)
        if sample:
            z = q_m + (q_v.sqrt() + 1e-12) * torch.randn_like(q_m)
        else:
            z = q_m
        return q_m, q_v, z


class SysVAEModule(nn.Module):
    """The SysVI generative model: a conditional VAE with a Gaussian
    likelihood, an optional VampPrior and a latent cycle-consistency loss
    (``scvi.external.sysvi`` ``SysVAE``)."""

    def __init__(self, n_input: int, n_batch: int, n_continuous_cov: int = 0,
                 n_cats_per_cov=None, n_latent: int = 15, n_hidden: int = 256,
                 n_layers: int = 2, dropout_rate: float = 0.1,
                 out_var_mode: str = "feature", prior: str = "vamp",
                 n_prior_components: int = 5, trainable_priors: bool = True,
                 pseudoinput_data=None, z_distance_cycle_weight: float = 2.0,
                 reconstruction_weight: float = 1.0):
        super().__init__()
        if prior not in ("standard_normal", "vamp"):
            raise ValueError("prior must be 'standard_normal' or 'vamp'")
        self.n_batch = int(n_batch)
        self.n_latent = int(n_latent)
        self.z_distance_cycle_weight = float(z_distance_cycle_weight)
        self.reconstruction_weight = float(reconstruction_weight)
        n_cat_list = [self.n_batch] + list(n_cats_per_cov or [])
        self.encoder = GaussianBlock(n_input, n_latent, n_cat_list, n_continuous_cov,
                                     n_hidden=n_hidden, n_layers=n_layers,
                                     dropout_rate=dropout_rate, var_mode="sample_feature")
        self.decoder = GaussianBlock(n_latent, n_input, n_cat_list, n_continuous_cov,
                                     n_hidden=n_hidden, n_layers=n_layers,
                                     dropout_rate=dropout_rate, var_mode=out_var_mode)
        if prior == "standard_normal":
            self.prior = make_prior("normal", n_latent)
        else:
            if pseudoinput_data is None:
                raise ValueError("prior='vamp' needs pseudoinput_data")
            x, batch_index, cat_list, cont = pseudoinput_data
            self.prior = VampPrior(n_prior_components, self.encoder, x, n_cat_list,
                                   batch_index, cat_list, cont, trainable=trainable_priors)
        self.nll = nn.GaussianNLLLoss(reduction="none")

    # ------------------------------------------------------------------
    @staticmethod
    def _cats(b):
        cat = b.get("cat")
        return () if cat is None else tuple(cat[:, i] for i in range(cat.shape[1]))

    def inference(self, x, batch_index, cat=(), cont=None, sample: bool = True):
        q_m, q_v, z = self.encoder(x, batch_index, *cat, cont=cont, sample=sample)
        return {"qz_m": q_m, "qz_v": q_v, "z": z}

    def generative(self, z, batch_index, cat=(), cont=None):
        q_m, q_v, _ = self.decoder(z, batch_index, *cat, cont=cont, sample=False)
        # ``px_scale`` is a *scale* (std), fed verbatim as GaussianNLLLoss's "var"
        # argument below — the same (deliberate) quirk as scvi-tools' SysVI.
        return {"px_m": q_m, "px_scale": q_v.sqrt() + 1e-12}

    def random_select_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """A different batch (system) code per cell, drawn uniformly from the
        ``n_batch - 1`` systems other than its own."""
        oh = F.one_hot(batch.reshape(-1).long(), self.n_batch)
        avail = torch.nonzero(1 - oh, as_tuple=True)[1].view(batch.shape[0], -1)
        pick = avail[torch.arange(batch.shape[0], device=batch.device),
                     torch.randint(0, avail.shape[1], (batch.shape[0],), device=batch.device)]
        return pick

    @staticmethod
    def latent_cycle_consistency(qz_m: torch.Tensor, qz_m_cycle: torch.Tensor) -> torch.Tensor:
        """MSE between the two latent means, standardised jointly (else the
        model could cheat by shrinking the latent scale)."""
        z = torch.cat([qz_m, qz_m_cycle])
        mean = z.mean(0, keepdim=True)
        std = z.std(0, keepdim=True)
        a = (qz_m - mean) / std
        b = (qz_m_cycle - mean) / std
        return F.mse_loss(a, b, reduction="none").sum(-1)

    def forward(self, b, kl_weight: float = 1.0):
        cat = self._cats(b)
        inf = self.inference(b["x"], b["batch"], cat, b.get("cont"))
        gen = self.generative(inf["z"], b["batch"], cat, b.get("cont"))
        compute_cycle = self.z_distance_cycle_weight > 0
        if compute_cycle:
            cyc_batch = self.random_select_batch(b["batch"])
            # re-encode the model's own reconstruction, mislabelled under a
            # different system, as scvi-tools' SysVAE._get_inference_cycle_input does
            inf_c = self.inference(gen["px_m"], cyc_batch, cat, b.get("cont"), sample=False)
            cyc_loss = self.latent_cycle_consistency(inf["qz_m"], inf_c["qz_m"])
        else:
            cyc_loss = torch.zeros(b["x"].shape[0], device=b["x"].device, dtype=inf["z"].dtype)
        return self.loss_terms(b, inf, gen, cyc_loss, kl_weight)

    def latent_mean(self, b):
        """Posterior mean of z for a minibatch (used by sharded inference)."""
        return self.inference(b["x"], b["batch"], self._cats(b), b.get("cont"),
                              sample=False)["qz_m"]

    def loss_terms(self, b, inf, gen, cyc_loss, kl_weight):
        x = b["x"]
        rec = self.nll(gen["px_m"], x, gen["px_scale"]).sum(-1)
        kl_z = self.prior.kl(inf["qz_m"], inf["qz_v"], inf["z"])
        loss = torch.mean(rec * self.reconstruction_weight + kl_weight * kl_z
                          + self.z_distance_cycle_weight * cyc_loss)
        return {"loss": loss, "reconstruction_loss": rec, "kl_local": kl_z,
                "kl_global": torch.zeros((), device=rec.device), "cycle_loss": cyc_loss}


def _pseudoinput_data(model, adata, n_components: int, indices=None, seed: int = 0):
    """Expression and covariates of ``n_components`` real cells, read through
    the model's own data source (works on AnnData, AnnDataOOM and Dask alike),
    to initialise the VampPrior pseudo-inputs."""
    n = adata.n_obs
    if indices is not None:
        idx = np.asarray(indices)
        if idx.shape[0] != n_components:
            raise ValueError("pseudoinputs_data_indices must have n_prior_components entries")
    else:
        k = min(int(n_components), n)
        idx = np.random.RandomState(seed).choice(n, size=k, replace=False)
    _, it, rows, order, _ = model._batches(adata, indices=idx, batch_size=len(idx))
    b = next(iter(it))
    cat = [b["cat"][:, i] for i in range(b["cat"].shape[1])] if "cat" in b else []
    return b["x"].clone(), b["batch"].clone(), [c.clone() for c in cat], b.get("cont")


@register_function(
    aliases=["SysVI", "sysVI", "sysvi", "系统整合", "VampPrior", "cycle consistency"],
    category="single",
    description="omicverse-native SysVI (no scvi-tools): cVAE with VampPrior and latent "
                "cycle-consistency for integrating multiple systems (species, protocol, "
                "modality...) on normalized expression; trains on in-memory AnnData, "
                "AnnDataOOM (disk) and Dask-backed AnnData with block-shuffled streaming.",
    examples=[
        "m = ov.vi.SysVI(adata, layer='lognorm', batch_key='system')",
        "m.train()",
        "adata.obsm['X_sysVI'] = m.get_latent_representation()",
    ],
    related=["vi.SCVI", "single.sysvi", "single.batch_correction"],
)
class SysVI(BaseVIModel):
    """SysVI: a cVAE with an optional VampPrior and a latent cycle-consistency
    loss for integrating multiple *systems* (Hrovatin et al. 2023;
    omicverse-native re-implementation of ``scvi.external.SysVI`` 1.5.2 — no
    scvi-tools dependency; trains on in-memory AnnData, AnnDataOOM (disk) and
    Dask-backed AnnData with block-shuffled streaming, like the rest of
    ``ov.vi``).

    Arguments:
        adata: AnnData, AnnDataOOM (``ov.read(path, storage='disk')``) or
            Dask-backed AnnData (GPU mode). ``layer`` (or ``X``) must already
            be normalised (e.g. log1p); SysVI's likelihood is Gaussian, not a
            count model, and does **not** convert raw counts.
        layer: Layer with normalised expression (``None`` = ``X``).
        batch_key: ``obs`` column with the *system* (species, protocol,
            modality...) — the covariate corrected via conditioning **and**
            the latent cycle-consistency loss. Needs >= 2 categories.
        categorical_covariate_keys / continuous_covariate_keys: Weaker
            covariates (e.g. donor/sample) used only to condition the cVAE,
            not cycle-corrected.
        genes: Training genes (bool mask, names, or a ``var`` column).
        n_hidden, n_latent, n_layers, dropout_rate: Encoder / decoder width,
            depth, dropout (scvi-tools SysVI defaults: 256, 15, 2, 0.1).
        out_var_mode: Decoder variance head, ``'feature'`` (default, one
            variance per gene) or ``'sample_feature'``.
        prior: ``'vamp'`` (default; VampPrior with trainable pseudo-inputs) or
            ``'standard_normal'``.
        n_prior_components: VampPrior mixture components (default 5).
        trainable_priors: Pseudo-inputs are trained (default) or fixed.
        pseudoinputs_data_indices: Cell indices used to initialise the
            pseudo-inputs (default: a random sample without replacement).
        z_distance_cycle_weight: Weight of the latent cycle-consistency loss
            (default 2.0; scvi-tools found KL warm-up poor with this loss, so
            :meth:`train` disables it — ``n_epochs_kl_warmup=0``).
        reconstruction_weight: Weight of the reconstruction loss (default 1.0).
        seed: Seed of the weight initialisation (``None`` = current torch RNG).

    Example::

        m = ov.vi.SysVI(adata, layer="lognorm", batch_key="system")
        m.train()
        adata.obsm["X_sysVI"] = m.get_latent_representation()
    """

    _module_cls = SysVAEModule

    def __init__(self, adata, layer=None, batch_key=None, categorical_covariate_keys=None,
                 continuous_covariate_keys=None, genes=None, n_hidden: int = 256,
                 n_latent: int = 15, n_layers: int = 2, dropout_rate: float = 0.1,
                 out_var_mode: str = "feature", prior: str = "vamp",
                 n_prior_components: int = 5, trainable_priors: bool = True,
                 pseudoinputs_data_indices=None, z_distance_cycle_weight: float = 2.0,
                 reconstruction_weight: float = 1.0, seed: int | None = 0, **module_kwargs):
        if batch_key is None:
            raise ValueError("SysVI needs batch_key= (the 'system' covariate, >= 2 categories)")
        super().__init__(adata, layer=layer, batch_key=batch_key, labels_key=None,
                         categorical_covariate_keys=categorical_covariate_keys,
                         continuous_covariate_keys=continuous_covariate_keys, genes=genes)
        if self.n_batch < 2:
            raise ValueError(f"batch_key={batch_key!r} must have at least two categories "
                             "(the systems to integrate)")
        self._pending_pseudoinput_data = None
        if prior == "vamp":
            self._pending_pseudoinput_data = _pseudoinput_data(
                self, adata, n_prior_components, pseudoinputs_data_indices, seed or 0)
        kw = dict(n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers,
                  dropout_rate=dropout_rate, out_var_mode=out_var_mode, prior=prior,
                  n_prior_components=n_prior_components, trainable_priors=trainable_priors,
                  z_distance_cycle_weight=z_distance_cycle_weight,
                  reconstruction_weight=reconstruction_weight, **module_kwargs)
        self._build(seed=seed, **kw)

    def _check_counts(self, src):
        # SysVI's likelihood is Gaussian on already-normalised expression: the
        # base class's "looks like raw counts?" heuristic does not apply here.
        pass

    def _build(self, seed=None, **kw):
        kw = {k: v for k, v in kw.items() if k != "seed"}
        pseudoinput_data = getattr(self, "_pending_pseudoinput_data", None)
        self._pending_pseudoinput_data = None
        self._module_kwargs = kw
        r = self.registry
        n_cats_per_cov = [len(c) for c in r["cat_categories"]]
        n_cont = len(r["continuous_covariate_keys"])
        if kw.get("prior", "vamp") == "vamp" and pseudoinput_data is None:
            # SysVI.load(): shapes only, every value is overwritten by load_state_dict.
            k = kw.get("n_prior_components", 5)
            pseudoinput_data = (
                torch.zeros(k, self.n_vars),
                torch.zeros(k, dtype=torch.long),
                [torch.zeros(k, dtype=torch.long) for _ in n_cats_per_cov],
                torch.zeros(k, n_cont) if n_cont else None,
            )
        self._module_args = dict(n_input=self.n_vars, n_batch=self.n_batch,
                                 n_continuous_cov=n_cont, n_cats_per_cov=n_cats_per_cov,
                                 pseudoinput_data=pseudoinput_data, **kw)
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(seed)
            self.module = self._module_cls(**self._module_args)
        self.module.to(self._device)

    # ------------------------------------------------------------------
    def train(self, *args, plan_kwargs: dict | None = None, **kwargs):
        """Train the model. SysVI disables KL warm-up (``n_epochs_kl_warmup``
        / ``n_steps_kl_warmup`` reset to 0): scvi-tools found warm-up gave poor
        integration together with the cycle-consistency loss. ``cuda_graph``
        also defaults to ``False`` here (unlike the other ``ov.vi`` models):
        the encoder *and* decoder both apply dropout (two RNG-consuming calls
        per step, as in ``scvi.external.sysvi``'s own architecture), which the
        current CUDA-graph capture in :mod:`omicverse.vi._train` does not
        handle cleanly on all setups (a failed first capture can leave the
        stream stuck in "capturing", breaking the eager fallback too); pass
        ``cuda_graph=True`` explicitly to opt back in. See
        :meth:`omicverse.vi.BaseVIModel.train` for the other arguments."""
        plan_kwargs = dict(plan_kwargs or {})
        no_warmup = {"n_epochs_kl_warmup": 0, "n_steps_kl_warmup": 0}
        if any(plan_kwargs.get(k, v) != v for k, v in no_warmup.items()):
            warnings.warn("SysVI disables KL warm-up (poor integration together with the "
                          "cycle-consistency loss); n_epochs_kl_warmup / n_steps_kl_warmup "
                          "are reset to 0.", UserWarning, stacklevel=2)
        plan_kwargs.update(no_warmup)
        kwargs.setdefault("cuda_graph", False)
        return super().train(*args, plan_kwargs=plan_kwargs, **kwargs)

    def get_latent_representation(self, adata=None, indices=None, give_mean: bool = True,
                                  batch_size: int | None = None):
        """Posterior mean (``give_mean=True``) or a reparameterised sample of
        ``z`` per cell."""
        mod = self.module

        def fn(b):
            inf = mod.inference(b["x"], b["batch"], mod._cats(b), b.get("cont"), sample=False)
            if give_mean:
                return inf["qz_m"]
            return inf["qz_m"] + (inf["qz_v"].sqrt() + 1e-12) * torch.randn_like(inf["qz_m"])

        return self._collect(fn, adata, indices, batch_size)


__all__ = ["SysVI", "SysVAEModule", "GaussianBlock"]
