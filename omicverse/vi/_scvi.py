"""scVI (Lopez et al., 2018): VAE with a (ZI)NB likelihood for raw counts."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .._registry import register_function
from ._base import BaseVIModel
from ._distributions import (
    kl_normal_normal, log_nb_positive, log_normal, log_poisson, log_zinb_positive, nb_rowsum,
    zinb_rowsum,
)
from ._nn import DecoderSCVI, Encoder
from ._priors import make_prior


def _flags(where: str):
    return where in ("encoder", "both"), where in ("decoder", "both")


class VAEModule(nn.Module):
    """The scVI generative model and its amortised posterior (scvi-tools ``VAE``).

    ``forward(batch, kl_weight)`` returns the training loss and the per-cell
    reconstruction / KL terms; ``inference`` and ``generative`` expose the two
    halves for downstream methods."""

    def __init__(self, n_input: int, n_batch: int = 0, n_labels: int = 0, n_hidden: int = 128,
                 n_latent: int = 10, n_layers: int = 1, n_continuous_cov: int = 0,
                 n_cats_per_cov=None, dropout_rate: float = 0.1, dispersion: str = "gene",
                 log_variational: bool = True, gene_likelihood: str = "zinb",
                 latent_distribution: str = "normal", encode_covariates: bool = False,
                 deeply_inject_covariates: bool = True, use_batch_norm: str = "both",
                 use_layer_norm: str = "none", use_observed_lib_size: bool = True,
                 library_log_means=None, library_log_vars=None, var_activation=None,
                 prior: str = "normal", sparse_likelihood: bool = True):
        super().__init__()
        if dispersion not in ("gene", "gene-batch", "gene-label", "gene-cell"):
            raise ValueError("dispersion must be 'gene', 'gene-batch', 'gene-label' or 'gene-cell'")
        if gene_likelihood not in ("zinb", "nb", "poisson", "normal"):
            raise ValueError("gene_likelihood must be 'zinb', 'nb', 'poisson' or 'normal'")
        self.n_input, self.n_batch, self.n_labels = n_input, n_batch, n_labels
        self.n_latent = n_latent
        self.dispersion = dispersion
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_observed_lib_size = use_observed_lib_size
        self.n_continuous_cov = n_continuous_cov
        self.sparse_likelihood = sparse_likelihood
        if not use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError("use_observed_lib_size=False needs library_log_means / vars")
            self.register_buffer("library_log_means",
                                 torch.as_tensor(np.asarray(library_log_means), dtype=torch.float32))
            self.register_buffer("library_log_vars",
                                 torch.as_tensor(np.asarray(library_log_vars), dtype=torch.float32))
        if dispersion == "gene":
            self.px_r = nn.Parameter(torch.randn(n_input))
        elif dispersion == "gene-batch":
            self.px_r = nn.Parameter(torch.randn(n_input, n_batch))
        elif dispersion == "gene-label":
            self.px_r = nn.Parameter(torch.randn(n_input, n_labels))
        bn_enc, bn_dec = _flags(use_batch_norm)
        ln_enc, ln_dec = _flags(use_layer_norm)
        cat_list = [n_batch] + list(n_cats_per_cov or [])
        enc_cats = cat_list if encode_covariates else None
        n_in_enc = n_input + n_continuous_cov * encode_covariates
        self.z_encoder = Encoder(n_in_enc, n_latent, n_cat_list=enc_cats, n_layers=n_layers,
                                 n_hidden=n_hidden, dropout_rate=dropout_rate,
                                 distribution=latent_distribution,
                                 inject_covariates=deeply_inject_covariates,
                                 use_batch_norm=bn_enc, use_layer_norm=ln_enc,
                                 var_activation=var_activation)
        if not use_observed_lib_size:
            self.l_encoder = Encoder(n_in_enc, 1, n_layers=1, n_cat_list=enc_cats,
                                     n_hidden=n_hidden, dropout_rate=dropout_rate,
                                     inject_covariates=deeply_inject_covariates,
                                     use_batch_norm=bn_enc, use_layer_norm=ln_enc,
                                     var_activation=var_activation)
        self.decoder = DecoderSCVI(n_latent + n_continuous_cov, n_input, n_cat_list=cat_list,
                                   n_layers=n_layers, n_hidden=n_hidden,
                                   inject_covariates=deeply_inject_covariates,
                                   use_batch_norm=bn_dec, use_layer_norm=ln_dec,
                                   need_dropout=gene_likelihood == "zinb",
                                   need_r=dispersion == "gene-cell")
        self.prior = make_prior(prior, n_latent)
        self.train_config = None

    # ------------------------------------------------------------------
    @staticmethod
    def _cats(b):
        cat = b.get("cat")
        return () if cat is None else tuple(cat[:, i] for i in range(cat.shape[1]))

    def inference(self, x, batch_index, cat=(), cont=None, n_samples: int = 1, sample=True):
        library = torch.log(x.sum(1, keepdim=True)) if self.use_observed_lib_size else None
        x_ = torch.log1p(x) if self.log_variational else x
        if cont is not None and self.encode_covariates:
            x_ = torch.cat([x_, cont], -1)
        cats = (batch_index, *cat) if self.encode_covariates else ()
        qm, qv, z = self.z_encoder(x_, *cats, sample=sample)
        out = {"qz_m": qm, "qz_v": qv, "z": z}
        if not self.use_observed_lib_size:
            lm, lv, library = self.l_encoder(x_, *cats, sample=sample)
            out.update(ql_m=lm, ql_v=lv)
        if n_samples > 1:
            eps = torch.randn((n_samples,) + qm.shape, device=qm.device, dtype=qm.dtype)
            z = self.z_encoder.z_transformation(qm + qv.sqrt() * eps)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(n_samples, *library.shape)
            else:
                library = out["ql_m"] + out["ql_v"].sqrt() * torch.randn(
                    (n_samples,) + out["ql_m"].shape, device=qm.device)
            out["z"] = z
        out["library"] = library
        return out

    def generative(self, z, library, batch_index, cat=(), cont=None, y=None,
                   transform_batch=None):
        dec_in = z
        if cont is not None:
            c = cont.unsqueeze(0).expand(z.shape[0], -1, -1) if z.dim() == 3 else cont
            dec_in = torch.cat([z, c], -1)
        if transform_batch is not None:
            batch_index = torch.full_like(batch_index, int(transform_batch))
        px_scale, px_r, px_rate, px_dropout = self.decoder(dec_in, library, batch_index, *cat)
        if self.dispersion == "gene":
            px_r = self.px_r
        elif self.dispersion == "gene-batch":
            px_r = F.embedding(batch_index.reshape(-1), self.px_r.t())
        elif self.dispersion == "gene-label":
            px_r = F.embedding(y.reshape(-1), self.px_r.t())
        theta = torch.exp(px_r)
        return {"px_scale": px_scale, "px_rate": px_rate, "px_r": theta, "px_dropout": px_dropout}

    def log_likelihood(self, b, gen) -> torch.Tensor:
        """Per-cell log p(x | z) (summed over genes)."""
        x, mu, theta = b["x"], gen["px_rate"].float(), gen["px_r"].float()
        sparse = self.sparse_likelihood and "row" in b and mu.dim() == 2
        if self.gene_likelihood == "zinb":
            pi = gen["px_dropout"].float()
            if sparse:
                return zinb_rowsum(mu, theta, pi, b["row"], b["col"], b["val"],
                                   flat=b.get("flat"))
            return log_zinb_positive(x, mu, theta, pi).sum(-1)
        if self.gene_likelihood == "nb":
            if sparse:
                return nb_rowsum(mu, theta, b["row"], b["col"], b["val"], flat=b.get("flat"))
            return log_nb_positive(x, mu, theta).sum(-1)
        if self.gene_likelihood == "poisson":
            return log_poisson(x, mu).sum(-1)
        return log_normal(x, mu, theta).sum(-1)

    def forward(self, b, kl_weight: float = 1.0):
        cat = self._cats(b)
        inf = self.inference(b["x"], b["batch"], cat, b.get("cont"))
        gen = self.generative(inf["z"], inf["library"], b["batch"], cat, b.get("cont"),
                              b.get("labels"))
        return self.loss_terms(b, inf, gen, kl_weight)

    def latent_mean(self, b):
        """Posterior mean of z for a minibatch (used by sharded inference)."""
        return self.inference(b["x"], b["batch"], self._cats(b), b.get("cont"),
                              sample=False)["qz_m"]

    def loss_terms(self, b, inf, gen, kl_weight):
        rec = -self.log_likelihood(b, gen)
        kl_z = self.prior.kl(inf["qz_m"].float(), inf["qz_v"].float(), inf["z"].float())
        if not self.use_observed_lib_size:
            n = self.library_log_means.shape[1]
            oh = F.one_hot(b["batch"].reshape(-1), n).float()
            pm, pv = oh @ self.library_log_means.t(), oh @ self.library_log_vars.t()
            kl_l = kl_normal_normal(inf["ql_m"], inf["ql_v"], pm, pv).sum(1)
        else:
            kl_l = torch.zeros_like(kl_z)
        loss = torch.mean(rec + kl_weight * kl_z + kl_l)
        return {"loss": loss, "reconstruction_loss": rec, "kl_local": kl_z + kl_l,
                "kl_global": torch.zeros((), device=rec.device)}


def _library_size_priors(model, adata):
    """Per-batch mean / variance of log library size (scvi-tools ``_init_library_size``)."""
    src = model._source(adata, device="cpu")
    n_batch = model.n_batch
    s = np.zeros(n_batch)
    q = np.zeros(n_batch)
    c = np.zeros(n_batch)
    for b in src.iterate(None, batch_size=4096, shuffle="none"):
        lib = torch.log(b["x"].sum(1)).double().numpy()
        bi = b["batch"].numpy()
        np.add.at(s, bi, lib)
        np.add.at(q, bi, lib * lib)
        np.add.at(c, bi, 1)
    mean = s / np.maximum(c, 1)
    var = q / np.maximum(c, 1) - mean ** 2
    return mean.reshape(1, -1).astype(np.float32), var.reshape(1, -1).astype(np.float32)


@register_function(
    aliases=["scVI", "scvi", "SCVI", "单细胞变分推断", "variational autoencoder", "vi.SCVI"],
    category="single",
    description="omicverse-native scVI (no scvi-tools): VAE with ZINB/NB likelihood for batch "
                "integration and denoising; trains on in-memory AnnData, AnnDataOOM (disk) and "
                "Dask-backed AnnData with block-shuffled streaming.",
    examples=[
        "m = ov.vi.SCVI(adata, layer='counts', batch_key='batch', n_latent=30)",
        "m.train()",
        "adata.obsm['X_scVI'] = m.get_latent_representation()",
    ],
    related=["vi.SCANVI", "vi.TOTALVI", "single.batch_correction"],
)
class SCVI(BaseVIModel):
    """Single-cell Variational Inference (scvi-tools-compatible API, omicverse implementation).

    Arguments:
        adata: AnnData, AnnDataOOM (``ov.read(path, storage='disk')``) or
            Dask-backed AnnData (GPU mode).  Raw counts are read from ``layer``.
        layer: Layer with raw counts (None = ``X``).
        batch_key: ``obs`` column with the batch (None = one batch).
        labels_key: ``obs`` column with labels (used by ``dispersion='gene-label'``).
        categorical_covariate_keys / continuous_covariate_keys: Extra covariates
            (injected into the decoder; and the encoder with ``encode_covariates``).
        genes: Training genes: a boolean mask, gene names or a ``var`` column such
            as ``'highly_variable_features'`` (selected at read time for disk data).
        n_hidden, n_latent, n_layers, dropout_rate, dispersion, gene_likelihood,
        latent_distribution, use_batch_norm, use_layer_norm, encode_covariates,
        deeply_inject_covariates, use_observed_lib_size: As scvi-tools 1.5.2
            (defaults 128, 10, 1, 0.1, 'gene', 'zinb', 'normal', 'both', 'none',
            False, True, True).
        seed: Seed of the weight initialisation (None = current torch RNG).

    Example::

        m = ov.vi.SCVI(adata, layer="counts", batch_key="batch", n_latent=30)
        m.train()
        adata.obsm["X_scVI"] = m.get_latent_representation()
    """

    _module_cls = VAEModule

    def __init__(self, adata, layer=None, batch_key=None, labels_key=None,
                 categorical_covariate_keys=None, continuous_covariate_keys=None, genes=None,
                 n_hidden: int = 128, n_latent: int = 10, n_layers: int = 1,
                 dropout_rate: float = 0.1, dispersion: str = "gene",
                 gene_likelihood: str = "zinb", latent_distribution: str = "normal",
                 seed: int | None = 0, **module_kwargs):
        super().__init__(adata, layer=layer, batch_key=batch_key, labels_key=labels_key,
                         categorical_covariate_keys=categorical_covariate_keys,
                         continuous_covariate_keys=continuous_covariate_keys, genes=genes)
        kw = dict(n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers,
                  dropout_rate=dropout_rate, dispersion=dispersion,
                  gene_likelihood=gene_likelihood, latent_distribution=latent_distribution,
                  **module_kwargs)
        if not kw.get("use_observed_lib_size", True) and "library_log_means" not in kw:
            m, v = _library_size_priors(self, adata)
            kw.update(library_log_means=m, library_log_vars=v)
        self._build(seed=seed, **kw)

    def _build(self, seed=None, **kw):
        kw = {k: v for k, v in kw.items() if k != "seed"}
        self._module_kwargs = kw
        r = self.registry
        self._module_args = dict(
            n_input=self.n_vars, n_batch=self.n_batch, n_labels=self.n_labels,
            n_continuous_cov=len(r["continuous_covariate_keys"]),
            n_cats_per_cov=[len(c) for c in r["cat_categories"]], **kw)
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(seed)
            self.module = self._module_cls(**self._module_args)
        self.module.to(self._device)

    def _init_params(self):
        kw = dict(self._module_kwargs)
        for k in ("library_log_means", "library_log_vars"):
            if k in kw and kw[k] is not None:
                kw[k] = np.asarray(kw[k]).tolist()
        return kw

    # ------------------------------------------------------------------
    def get_latent_representation(self, adata=None, indices=None, give_mean: bool = True,
                                  mc_samples: int = 5000, batch_size: int | None = None,
                                  return_dist: bool = False):
        """Posterior mean (``give_mean``) or a sample of ``z`` per cell; with
        ``return_dist`` the posterior means and variances."""
        mod = self.module

        def fn(b):
            inf = mod.inference(b["x"], b["batch"], mod._cats(b), b.get("cont"), sample=False)
            qm, qv = inf["qz_m"], inf["qz_v"]
            if return_dist:
                return torch.cat([qm, qv], 1)
            if give_mean:
                if mod.latent_distribution == "ln":
                    eps = torch.randn((mc_samples,) + qm.shape, device=qm.device)
                    return torch.softmax(qm + qv.sqrt() * eps, -1).mean(0)
                return qm
            return mod.z_encoder.z_transformation(qm + qv.sqrt() * torch.randn_like(qm))

        out = self._collect(fn, adata, indices, batch_size)
        if return_dist:
            k = out.shape[1] // 2
            return out[:, :k], out[:, k:]
        return out

    def _transform_batch_codes(self, transform_batch):
        if transform_batch is None:
            return [None]
        tb = transform_batch if isinstance(transform_batch, (list, tuple)) else [transform_batch]
        cats = self.registry["batch_categories"]
        out = []
        for t in tb:
            if t in cats:
                out.append(cats.index(t))
            elif str(t) in [str(c) for c in cats]:
                out.append([str(c) for c in cats].index(str(t)))
            else:
                raise ValueError(f"transform_batch {t!r} is not a batch category")
        return out

    def get_normalized_expression(self, adata=None, indices=None, transform_batch=None,
                                  gene_list=None, library_size=1, n_samples: int = 1,
                                  batch_size: int | None = None, return_mean: bool = True,
                                  return_numpy: bool | None = None, as_generator: bool = False):
        """Denoised, library-size normalised expression ``px_scale * library_size``
        (``library_size='latent'`` gives the model's rate).  Like scvi-tools, ``z``
        is sampled from the posterior (``n_samples`` samples, averaged with
        ``return_mean``) and ``transform_batch`` averages over the given batches.

        For large data use ``as_generator=True``: yields ``(cell_indices, block)``
        in cell order, so the full matrix never has to fit in memory."""
        mod = self.module
        tbs = self._transform_batch_codes(transform_batch)
        gsel = None
        if gene_list is not None:
            names = self.registry["var_names"]
            pos = {g: i for i, g in enumerate(names)}
            gsel = torch.as_tensor([pos[g] for g in gene_list], dtype=torch.long)

        def fn(b):
            cat = mod._cats(b)
            acc = None
            for tb in tbs:
                inf = mod.inference(b["x"], b["batch"], cat, b.get("cont"), n_samples=n_samples)
                gen = mod.generative(inf["z"], inf["library"], b["batch"], cat, b.get("cont"),
                                     b.get("labels"), transform_batch=tb)
                v = gen["px_rate"] if library_size == "latent" else gen["px_scale"] * library_size
                acc = v if acc is None else acc + v
            v = acc / len(tbs)
            if gsel is not None:
                v = v[..., gsel.to(v.device)]
            if n_samples > 1:
                v = v.mean(0) if return_mean else v.permute(1, 0, 2)
            return v

        if as_generator:
            return self._generate(fn, adata, indices, batch_size)
        out = self._collect(fn, adata, indices, batch_size)
        if n_samples > 1 and not return_mean:
            out = out.transpose(1, 0, 2)
        if return_numpy or n_samples > 1 and not return_mean:
            return out
        import pandas as pd
        ad = self.adata if adata is None else adata
        names = gene_list if gene_list is not None else self.registry["var_names"]
        obs_names = ad.obs_names if indices is None else ad.obs_names[np.asarray(indices)]
        return pd.DataFrame(out, index=obs_names, columns=names)

    def _generate(self, fn, adata=None, indices=None, batch_size=None):
        self._check_trained()
        src, it, rows, order, n = self._batches(adata, indices, batch_size)
        self.module.eval()
        with torch.no_grad():
            for b in it:
                yield b["idx"].cpu().numpy(), fn(b).cpu().numpy()

    def posterior_predictive_sample(self, adata=None, indices=None, n_samples: int = 1,
                                    batch_size: int | None = None):
        """Counts drawn from p(x | z) with z from the posterior (dense, cells x genes)."""
        from ._distributions import sample_nb, sample_zinb
        mod = self.module

        def fn(b):
            cat = mod._cats(b)
            inf = mod.inference(b["x"], b["batch"], cat, b.get("cont"), n_samples=n_samples)
            gen = mod.generative(inf["z"], inf["library"], b["batch"], cat, b.get("cont"),
                                 b.get("labels"))
            mu, th = gen["px_rate"], gen["px_r"]
            if mod.gene_likelihood == "zinb":
                x = sample_zinb(mu, th, gen["px_dropout"])
            elif mod.gene_likelihood == "nb":
                x = sample_nb(mu, th)
            else:
                x = torch.poisson(mu)
            return x.permute(1, 2, 0) if n_samples > 1 else x

        return self._collect(fn, adata, indices, batch_size)


__all__ = ["SCVI", "VAEModule"]
