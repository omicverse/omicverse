"""totalVI (Gayoso et al., 2021): joint RNA + protein (CITE-seq) VAE."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from .._registry import register_function
from ._base import BaseVIModel, _max_epochs_heuristic
from ._distributions import (
    kl_normal_normal, kl_normal_standard, log_mixture_nb, log_nb_positive, log_zinb_positive,
    nb_rowsum, zinb_rowsum,
)
from ._nn import Classifier, FCLayers
from ._scvi import _flags

_LOG_2PI = float(np.log(2 * np.pi))


class EncoderTOTALVI(nn.Module):
    """``q(z, l | x, y)``: Gaussian latent and log-normal RNA library size."""

    def __init__(self, n_input, n_output, n_cat_list=None, n_layers=2, n_hidden=256,
                 dropout_rate=0.1, distribution="ln", use_batch_norm=True, use_layer_norm=False):
        super().__init__()
        kw = dict(n_cat_list=n_cat_list, n_hidden=n_hidden, dropout_rate=dropout_rate,
                  use_batch_norm=use_batch_norm, use_layer_norm=use_layer_norm)
        self.encoder = FCLayers(n_input, n_hidden, n_layers=n_layers, **kw)
        self.z_mean_encoder = nn.Linear(n_hidden, n_output)
        self.z_var_encoder = nn.Linear(n_hidden, n_output)
        self.l_gene_encoder = FCLayers(n_input, n_hidden, n_layers=1, **kw)
        self.l_gene_mean_encoder = nn.Linear(n_hidden, 1)
        self.l_gene_var_encoder = nn.Linear(n_hidden, 1)
        self.distribution = distribution

    def z_transformation(self, z):
        return torch.softmax(z, -1) if self.distribution == "ln" else z

    def forward(self, data, *cats, sample=True):
        q = self.encoder(data, *cats)
        qz_m = self.z_mean_encoder(q)
        qz_v = torch.exp(self.z_var_encoder(q)) + 1e-4
        ql = self.l_gene_encoder(data, *cats)
        ql_m = self.l_gene_mean_encoder(ql)
        ql_v = torch.exp(self.l_gene_var_encoder(ql)) + 1e-4
        if sample:
            untran_z = qz_m + qz_v.sqrt() * torch.randn_like(qz_m)
            log_l = ql_m + ql_v.sqrt() * torch.randn_like(ql_m)
        else:
            untran_z, log_l = qz_m, ql_m
        log_l = torch.clamp(log_l, max=15)
        return qz_m, qz_v, ql_m, ql_v, self.z_transformation(untran_z), torch.exp(log_l)


class DecoderTOTALVI(nn.Module):
    """``p(x, y | z)``: RNA NB / ZINB and protein background / foreground NB mixture."""

    def __init__(self, n_input, n_output_genes, n_output_proteins, n_cat_list=None, n_layers=1,
                 n_hidden=256, dropout_rate=0.0, use_batch_norm=True, use_layer_norm=False,
                 scale_activation="softmax", activation_function_bg="softplus"):
        super().__init__()
        self.bg_act = F.softplus if activation_function_bg == "softplus" else torch.exp
        self.scale_activation = scale_activation
        deep = dict(n_cat_list=n_cat_list, n_layers=n_layers, n_hidden=n_hidden,
                    dropout_rate=dropout_rate, use_batch_norm=use_batch_norm,
                    use_layer_norm=use_layer_norm)
        lin = dict(n_cat_list=n_cat_list, n_layers=1, use_activation=False, use_batch_norm=False,
                   use_layer_norm=False, dropout_rate=0)
        self.px_decoder = FCLayers(n_input, n_hidden, **deep)
        self.px_scale_decoder = FCLayers(n_hidden + n_input, n_output_genes, **lin)
        self.py_back_decoder = FCLayers(n_input, n_hidden, **deep)
        self.py_back_mean_log_alpha = FCLayers(n_hidden + n_input, n_output_proteins, **lin)
        self.py_back_mean_log_beta = FCLayers(n_hidden + n_input, n_output_proteins, **lin)
        self.py_fore_decoder = FCLayers(n_input, n_hidden, **deep)
        self.py_fore_scale_decoder = FCLayers(n_hidden + n_input, n_output_proteins,
                                              n_cat_list=n_cat_list, n_layers=1,
                                              use_activation=True, use_batch_norm=False,
                                              use_layer_norm=False, dropout_rate=0)
        self.sigmoid_decoder = FCLayers(n_input, n_hidden, **deep)
        self.px_dropout_decoder_gene = FCLayers(n_hidden + n_input, n_output_genes, **lin)
        self.py_background_decoder = FCLayers(n_hidden + n_input, n_output_proteins, **lin)

    def forward(self, z, library_gene, *cats, sample=True):
        px_, py_ = {}, {}
        px = self.px_decoder(z, *cats)
        u = self.px_scale_decoder(torch.cat([px, z], -1), *cats)
        px_["scale"] = torch.softmax(u, -1) if self.scale_activation == "softmax" else F.softplus(u)
        px_["rate"] = library_gene * px_["scale"]
        pb = torch.cat([self.py_back_decoder(z, *cats), z], -1)
        py_["back_alpha"] = self.py_back_mean_log_alpha(pb, *cats)
        py_["back_beta"] = self.bg_act(self.py_back_mean_log_beta(pb, *cats)) + 1e-8
        log_back = py_["back_alpha"] + (py_["back_beta"] * torch.randn_like(py_["back_beta"])
                                        if sample else 0.0)
        py_["rate_back"] = torch.exp(log_back)
        pf = torch.cat([self.py_fore_decoder(z, *cats), z], -1)
        py_["fore_scale"] = self.py_fore_scale_decoder(pf, *cats) + 1 + 1e-8
        py_["rate_fore"] = py_["rate_back"] * py_["fore_scale"]
        pm = torch.cat([self.sigmoid_decoder(z, *cats), z], -1)
        px_["dropout"] = self.px_dropout_decoder_gene(pm, *cats)
        py_["mixing"] = self.py_background_decoder(pm, *cats)
        mix = torch.sigmoid(py_["mixing"])
        py_["scale"] = F.normalize((1 - mix) * py_["rate_fore"], p=1, dim=-1)
        return px_, py_, log_back


class TOTALVAEModule(nn.Module):
    """scvi-tools ``TOTALVAE`` (panel = batch)."""

    def __init__(self, n_input_genes, n_input_proteins, n_batch=0, n_labels=0, n_hidden=256,
                 n_latent=20, n_layers_encoder=2, n_layers_decoder=1, n_continuous_cov=0,
                 n_cats_per_cov=None, dropout_rate_decoder=0.2, dropout_rate_encoder=0.2,
                 gene_dispersion="gene", protein_dispersion="protein", log_variational=True,
                 gene_likelihood="nb", latent_distribution="normal", protein_batch_mask=None,
                 encode_covariates=True, protein_background_prior_mean=None,
                 protein_background_prior_scale=None, use_observed_lib_size=True,
                 library_log_means=None, library_log_vars=None, use_batch_norm="both",
                 use_layer_norm="none", adversarial_classifier=False, pro_recons_weight=1.0,
                 sparse_likelihood=True):
        super().__init__()
        self.n_input_genes, self.n_input_proteins = n_input_genes, n_input_proteins
        self.n_batch, self.n_labels, self.n_latent = n_batch, n_labels, n_latent
        self.gene_dispersion, self.protein_dispersion = gene_dispersion, protein_dispersion
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_observed_lib_size = use_observed_lib_size
        self.pro_recons_weight = pro_recons_weight
        self.sparse_likelihood = sparse_likelihood
        self.n_continuous_cov = n_continuous_cov
        if protein_batch_mask is not None:
            self.register_buffer("protein_batch_mask",
                                 torch.as_tensor(np.asarray(protein_batch_mask), dtype=torch.float32))
        else:
            self.protein_batch_mask = None
        if not use_observed_lib_size:
            self.register_buffer("library_log_means",
                                 torch.as_tensor(np.asarray(library_log_means), dtype=torch.float32))
            self.register_buffer("library_log_vars",
                                 torch.as_tensor(np.asarray(library_log_vars), dtype=torch.float32))
        n_panel = n_batch
        if protein_background_prior_mean is None:
            if n_panel > 0:
                self.background_pro_alpha = nn.Parameter(3.0 * torch.ones(n_input_proteins, n_panel))
                self.background_pro_log_beta = nn.Parameter(torch.zeros(n_input_proteins, n_panel))
            else:
                self.background_pro_alpha = nn.Parameter(torch.ones(n_input_proteins))
                self.background_pro_log_beta = nn.Parameter(torch.zeros(n_input_proteins))
        else:
            mean = np.asarray(protein_background_prior_mean, dtype=np.float32)
            scale = np.asarray(protein_background_prior_scale, dtype=np.float32)
            if mean.ndim == 2 and mean.shape[1] == 1 and n_panel != 1:
                mean, scale = mean.ravel(), scale.ravel()
            self.background_pro_alpha = nn.Parameter(torch.from_numpy(mean))
            self.background_pro_log_beta = nn.Parameter(torch.log(torch.from_numpy(scale)))
        self.log_per_batch_efficiency = nn.Parameter(torch.zeros(n_input_proteins, n_batch))
        if gene_dispersion == "gene":
            self.px_r = nn.Parameter(torch.randn(n_input_genes))
        elif gene_dispersion == "gene-batch":
            self.px_r = nn.Parameter(torch.randn(n_input_genes, n_batch))
        elif gene_dispersion == "gene-label":
            self.px_r = nn.Parameter(torch.randn(n_input_genes, n_labels))
        else:
            raise ValueError("gene_dispersion must be 'gene', 'gene-batch' or 'gene-label'")
        if protein_dispersion == "protein":
            self.py_r = nn.Parameter(2 * torch.rand(n_input_proteins))
        elif protein_dispersion == "protein-batch":
            self.py_r = nn.Parameter(2 * torch.rand(n_input_proteins, n_batch))
        elif protein_dispersion == "protein-label":
            self.py_r = nn.Parameter(2 * torch.rand(n_input_proteins, n_labels))
        else:
            raise ValueError("protein_dispersion must be 'protein', 'protein-batch' or "
                             "'protein-label'")
        bn_enc, bn_dec = _flags(use_batch_norm)
        ln_enc, ln_dec = _flags(use_layer_norm)
        n_in = n_input_genes + n_input_proteins + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list(n_cats_per_cov or [])
        self.encoder = EncoderTOTALVI(n_in, n_latent, n_layers=n_layers_encoder,
                                      n_cat_list=cat_list if encode_covariates else None,
                                      n_hidden=n_hidden, dropout_rate=dropout_rate_encoder,
                                      distribution=latent_distribution, use_batch_norm=bn_enc,
                                      use_layer_norm=ln_enc)
        self.decoder = DecoderTOTALVI(n_latent + n_continuous_cov, n_input_genes,
                                      n_input_proteins, n_layers=n_layers_decoder,
                                      n_cat_list=cat_list, n_hidden=n_hidden,
                                      dropout_rate=dropout_rate_decoder, use_batch_norm=bn_dec,
                                      use_layer_norm=ln_dec)
        self.has_extra_step = bool(adversarial_classifier)
        if adversarial_classifier:
            self.adversarial_classifier = Classifier(n_latent, n_hidden=32, n_labels=n_batch,
                                                     n_layers=2, logits=True)
            self._adv_opt = None
        self.train_config = None

    # ------------------------------------------------------------------
    @staticmethod
    def _cats(b):
        cat = b.get("cat")
        return () if cat is None else tuple(cat[:, i] for i in range(cat.shape[1]))

    def optim_parameters(self):
        skip = {id(p) for p in self.adversarial_classifier.parameters()} \
            if self.has_extra_step else set()
        return [p for p in self.parameters() if id(p) not in skip]

    def _gather(self, param, idx):
        return F.embedding(idx.reshape(-1), param.t())

    def inference(self, x, y, batch_index, cat=(), cont=None, n_samples=1, sample=True):
        x_ = x / (1 + x.mean(1, keepdim=True))
        y_ = y / (1 + y.mean(1, keepdim=True))
        library_gene = x.sum(1, keepdim=True) if self.use_observed_lib_size else None
        if self.log_variational:
            x_, y_ = torch.log1p(x_), torch.log1p(y_)
        parts = [x_, y_] + ([cont] if cont is not None and self.encode_covariates else [])
        enc_in = torch.cat(parts, -1)
        cats = (batch_index, *cat) if self.encode_covariates else ()
        qz_m, qz_v, ql_m, ql_v, z, lib = self.encoder(enc_in, *cats, sample=sample)
        if not self.use_observed_lib_size:
            library_gene = lib
        if n_samples > 1:
            eps = torch.randn((n_samples,) + qz_m.shape, device=qz_m.device)
            z = self.encoder.z_transformation(qz_m + qz_v.sqrt() * eps)
            if self.use_observed_lib_size:
                library_gene = library_gene.unsqueeze(0).expand(n_samples, *library_gene.shape)
            else:
                library_gene = torch.exp(ql_m + ql_v.sqrt() * torch.randn(
                    (n_samples,) + ql_m.shape, device=ql_m.device))
        if self.n_batch > 0:
            prior_a = self._gather(self.background_pro_alpha, batch_index)
            prior_b = self._gather(torch.exp(self.background_pro_log_beta), batch_index)
        else:
            prior_a, prior_b = self.background_pro_alpha, torch.exp(self.background_pro_log_beta)
        return {"qz_m": qz_m, "qz_v": qz_v, "ql_m": ql_m, "ql_v": ql_v, "z": z,
                "library_gene": library_gene, "prior_back_alpha": prior_a,
                "prior_back_beta": prior_b}

    def generative(self, z, library_gene, batch_index, cat=(), cont=None, label=None,
                   transform_batch=None, sample=True):
        dec_in = z
        if cont is not None:
            c = cont.unsqueeze(0).expand(z.shape[0], -1, -1) if z.dim() == 3 else cont
            dec_in = torch.cat([z, c], -1)
        if transform_batch is not None:
            batch_index = torch.full_like(batch_index, int(transform_batch))
        eff = torch.exp(self._gather(self.log_per_batch_efficiency, batch_index))
        px_, py_, log_back = self.decoder(dec_in, library_gene, batch_index, *cat, sample=sample)
        if self.gene_dispersion == "gene":
            px_r = self.px_r
        elif self.gene_dispersion == "gene-batch":
            px_r = self._gather(self.px_r, batch_index)
        else:
            px_r = self._gather(self.px_r, label)
        if self.protein_dispersion == "protein":
            py_r = self.py_r
        elif self.protein_dispersion == "protein-batch":
            py_r = self._gather(self.py_r, batch_index)
        else:
            py_r = self._gather(self.py_r, label)
        px_["r"], py_["r"] = torch.exp(px_r), torch.exp(py_r)
        return {"px_": px_, "py_": py_, "per_batch_efficiency": eff, "log_pro_back_mean": log_back}

    def _rna_loglik(self, b, px_):
        mu, theta = px_["rate"].float(), px_["r"].float()
        sparse = self.sparse_likelihood and "row" in b and mu.dim() == 2
        if self.gene_likelihood == "zinb":
            pi = px_["dropout"].float()
            if sparse:
                return zinb_rowsum(mu, theta, pi, b["row"], b["col"], b["val"], flat=b.get("flat"))
            return log_zinb_positive(b["x"], mu, theta, pi).sum(-1)
        if sparse:
            return nb_rowsum(mu, theta, b["row"], b["col"], b["val"], flat=b.get("flat"))
        return log_nb_positive(b["x"], mu, theta).sum(-1)

    def forward(self, b, kl_weight=1.0):
        cat = self._cats(b)
        y = b["protein"]
        inf = self.inference(b["x"], y, b["batch"], cat, b.get("cont"))
        gen = self.generative(inf["z"], inf["library_gene"], b["batch"], cat, b.get("cont"),
                              b.get("labels"))
        px_, py_, eff = gen["px_"], gen["py_"], gen["per_batch_efficiency"]
        rec_gene = -self._rna_loglik(b, px_)
        lp = log_mixture_nb(y, eff * py_["rate_back"], eff * py_["rate_fore"], py_["r"], None,
                            py_["mixing"])
        mask = None
        if self.protein_batch_mask is not None:
            mask = self.protein_batch_mask[b["batch"].reshape(-1)]
            lp = lp * mask
        rec_pro = -lp.sum(-1)
        kl_z = kl_normal_standard(inf["qz_m"], inf["qz_v"])
        if not self.use_observed_lib_size:
            n = self.library_log_means.shape[1]
            oh = F.one_hot(b["batch"].reshape(-1), n).float()
            pm, pv = oh @ self.library_log_means.t(), oh @ self.library_log_vars.t()
            kl_l = kl_normal_normal(inf["ql_m"], inf["ql_v"], pm, pv).sum(1)
        else:
            kl_l = torch.zeros_like(kl_z)
        kl_back_full = kl_normal_normal(py_["back_alpha"], py_["back_beta"] ** 2,
                                        inf["prior_back_alpha"], inf["prior_back_beta"] ** 2)
        if mask is not None:
            kl_back_full = kl_back_full * mask
        log_eff = torch.log(eff)
        lkl_eff = log_eff + 0.5 * _LOG_2PI + 0.5 * log_eff ** 2      # -log LogNormal(0, 1)
        lkl_mix = 1e-3 * F.softplus(-py_["mixing"])                   # -1e-3 log Bern(1)
        kl_back = kl_back_full.sum(1) + lkl_eff.sum(1) + lkl_mix.sum(1)
        loss = torch.mean(rec_gene + kl_weight * self.pro_recons_weight * rec_pro
                          + kl_weight * kl_z + kl_l + kl_weight * kl_back)
        out = {"reconstruction_loss": rec_gene + rec_pro, "kl_local": kl_z + kl_l + kl_back,
               "kl_global": torch.zeros((), device=loss.device),
               "reconstruction_loss_gene": rec_gene, "reconstruction_loss_protein": rec_pro}
        if self.has_extra_step:
            kappa = 1.0 - float(kl_weight)
            if kappa > 0:
                loss = loss + kappa * self._adversarial_loss(inf["z"], b["batch"], False)
            out["_z"] = inf["z"]
        out["loss"] = loss
        return out

    def _adversarial_loss(self, z, batch_index, predict_true_class):
        n = self.n_batch
        logp = torch.log_softmax(self.adversarial_classifier(z), dim=1)
        oh = F.one_hot(batch_index.reshape(-1), n).float()
        target = oh if predict_true_class else (1 - oh) / max(n - 1, 1)
        return -(logp * target).sum(1).mean()

    def extra_step(self, out, b, kl_weight):
        """Update the adversarial batch classifier on the detached latent."""
        kappa = 1.0 - float(kl_weight)
        if kappa <= 0 or "_z" not in out:
            return
        if self._adv_opt is None:
            wd = self.train_config.weight_decay if self.train_config is not None else 1e-6
            self._adv_opt = torch.optim.Adam(self.adversarial_classifier.parameters(), lr=1e-3,
                                             eps=0.01, weight_decay=wd)
        self._adv_opt.zero_grad(set_to_none=True)
        (kappa * self._adversarial_loss(out["_z"].detach(), b["batch"], True)).backward()
        self._adv_opt.step()

    def latent_mean(self, b):
        inf = self.inference(b["x"], b["protein"], b["batch"], self._cats(b), b.get("cont"),
                             sample=False)
        return inf["qz_m"]


def _protein_priors(pro, batch, n_batch, mask=None, n_cells=100, seed=0):
    """Empirical protein background prior (scvi-tools ``_get_totalvi_protein_priors``):
    per batch, a 2-component GMM on log1p counts of 100 random cells; the lower
    component's mean / sd averaged over cells."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.mixture import GaussianMixture
    rs = np.random.RandomState(seed)
    mus, scales = [], []
    for b in range(n_batch):
        sel = batch == b
        if not sel.any():
            mus.append(0.0); scales.append(1.0)
            continue
        pe = pro[sel]
        if mask is not None:
            pe = pe[:, mask[b]]
            if pe.shape[1] < 5:
                mus.append(0.0); scales.append(0.05)
                continue
        cells = rs.choice(pe.shape[0], size=n_cells)
        gmm = GaussianMixture(n_components=2)
        m_, s_ = [], []
        for c in pe[cells]:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                try:
                    gmm.fit(np.log1p(c.reshape(-1, 1)))
                except (ConvergenceWarning, ValueError):
                    m_.append(0.0); s_.append(0.05)
                    continue
            means = gmm.means_.ravel()
            o = np.argsort(means)
            m_.append(means[o][0])
            s_.append(np.sqrt(gmm.covariances_[o].ravel()[0]))
        mus.append(float(np.mean(m_)))
        scales.append(float(np.sqrt(np.sum(np.square(s_)) / n_cells ** 2)))
    P = pro.shape[1]
    return (np.tile(np.asarray(mus, np.float32).reshape(1, -1), (P, 1)),
            np.tile(np.asarray(scales, np.float32).reshape(1, -1), (P, 1)))


@register_function(
    aliases=["totalVI", "totalvi", "TOTALVI", "CITE-seq", "RNA蛋白联合", "vi.TOTALVI"],
    category="single",
    description="omicverse-native totalVI (no scvi-tools): joint RNA + surface protein (CITE-seq) "
                "VAE for integration, denoised RNA / protein and protein background removal.",
    examples=[
        "m = ov.vi.TOTALVI(adata, layer='counts', batch_key='batch',"
        " protein_expression_obsm_key='protein_expression')",
        "m.train(); adata.obsm['X_totalVI'] = m.get_latent_representation()",
        "rna, pro = m.get_normalized_expression(n_samples=25)",
    ],
    related=["vi.SCVI", "single.batch_correction"],
)
class TOTALVI(BaseVIModel):
    """total Variational Inference for CITE-seq (scvi-tools-compatible API).

    Arguments:
        adata: AnnData / AnnDataOOM / Dask-backed AnnData with RNA counts in ``layer``
            and protein counts in ``obsm[protein_expression_obsm_key]`` (cells x
            proteins; DataFrame or array).
        protein_expression_obsm_key: ``obsm`` key of the protein counts.
        protein_names_uns_key: ``uns`` key with protein names (array input).
        n_latent, gene_dispersion, protein_dispersion, gene_likelihood,
        latent_distribution: As scvi-tools (20, 'gene', 'protein', 'nb', 'normal').
        empirical_protein_background_prior: GMM-based background prior
            initialisation (default: when more than 10 proteins).
        override_missing_proteins: Do not treat all-zero proteins of a batch as missing.
    """

    _module_cls = TOTALVAEModule

    def __init__(self, adata, layer=None, batch_key=None, protein_expression_obsm_key=None,
                 protein_names_uns_key=None, labels_key=None, categorical_covariate_keys=None,
                 continuous_covariate_keys=None, genes=None, n_latent=20,
                 gene_dispersion="gene", protein_dispersion="protein", gene_likelihood="nb",
                 latent_distribution="normal", empirical_protein_background_prior=None,
                 override_missing_proteins=False, seed=0, registry=None, **module_kwargs):
        if protein_expression_obsm_key is None and registry is None:
            raise ValueError("TOTALVI needs protein_expression_obsm_key=")
        super().__init__(adata, layer=layer, batch_key=batch_key, labels_key=labels_key,
                         categorical_covariate_keys=categorical_covariate_keys,
                         continuous_covariate_keys=continuous_covariate_keys, genes=genes,
                         registry=registry)
        if registry is None:
            pro = adata.obsm[protein_expression_obsm_key]
            if isinstance(pro, pd.DataFrame):
                names = [str(c) for c in pro.columns]
            elif protein_names_uns_key is not None:
                names = [str(c) for c in adata.uns[protein_names_uns_key]]
            else:
                names = [f"protein_{i}" for i in range(np.asarray(pro).shape[1])]
            self.registry["protein_expression_obsm_key"] = protein_expression_obsm_key
            self.registry["protein_names"] = names
        pro = self._protein(adata)
        batch = self._obs_arrays(adata)["batch"]
        # proteins with all-zero counts in a batch are missing there (scvi-tools)
        mask = np.stack([pro[batch == b].sum(0) > 0 if (batch == b).any()
                         else np.ones(pro.shape[1], bool) for b in range(self.n_batch)])
        missing = (~mask).any() and not override_missing_proteins
        if missing:
            warnings.warn("Some proteins have all 0 counts in some batches; they are treated as "
                          "missing there (override_missing_proteins=True to disable).",
                          UserWarning, stacklevel=2)
        emp = (empirical_protein_background_prior if empirical_protein_background_prior
               is not None else pro.shape[1] > 10)
        kw = dict(n_latent=n_latent, gene_dispersion=gene_dispersion,
                  protein_dispersion=protein_dispersion, gene_likelihood=gene_likelihood,
                  latent_distribution=latent_distribution, **module_kwargs)
        kw.setdefault("adversarial_classifier", bool(missing))
        if missing:
            kw.setdefault("protein_batch_mask", mask.astype(np.float32))
        if emp and "protein_background_prior_mean" not in kw:
            pm, ps = _protein_priors(pro, batch, self.n_batch, mask if missing else None,
                                     seed=seed or 0)
            kw.update(protein_background_prior_mean=pm, protein_background_prior_scale=ps)
        if not kw.get("use_observed_lib_size", True) and "library_log_means" not in kw:
            from ._scvi import _library_size_priors
            m, v = _library_size_priors(self, adata)
            kw.update(library_log_means=m, library_log_vars=v)
        self._build(seed=seed, **kw)

    def _protein(self, adata):
        pro = adata.obsm[self.registry["protein_expression_obsm_key"]]
        if isinstance(pro, pd.DataFrame):
            pro = pro[self.registry["protein_names"]].to_numpy() \
                if set(self.registry["protein_names"]) <= set(map(str, pro.columns)) else pro.to_numpy()
        return np.asarray(pro, dtype=np.float32)

    def _extra_obs(self, adata):
        return {"protein": self._protein(adata)}

    def _build(self, seed=None, **kw):
        kw = {k: v for k, v in kw.items() if k != "seed"}
        self._module_kwargs = kw
        r = self.registry
        self._module_args = dict(
            n_input_genes=self.n_vars, n_input_proteins=len(r["protein_names"]),
            n_batch=self.n_batch, n_labels=self.n_labels,
            n_continuous_cov=len(r["continuous_covariate_keys"]),
            n_cats_per_cov=[len(c) for c in r["cat_categories"]], **kw)
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(seed)
            self.module = self._module_cls(**self._module_args)
        self.module.to(self._device)

    def _init_params(self):
        out = {}
        for k, v in self._module_kwargs.items():
            out[k] = np.asarray(v).tolist() if isinstance(v, np.ndarray) else v
        return out

    @property
    def protein_names(self):
        return list(self.registry["protein_names"])

    def train(self, max_epochs=None, lr=4e-3, batch_size=256, early_stopping=True,
              reduce_lr_on_plateau=True, n_steps_kl_warmup=None, n_epochs_kl_warmup=None,
              **kwargs):
        """Train with scvi-tools' totalVI defaults: lr 4e-3, batch 256, early stopping,
        reduce-LR-on-plateau (validation every epoch), KL warm-up over
        ``0.75 * n_cells`` steps."""
        n = self.adata.n_obs
        if n_steps_kl_warmup is None and n_epochs_kl_warmup is None:
            n_steps_kl_warmup = int(0.75 * n)
        if max_epochs is None:
            max_epochs = _max_epochs_heuristic(n)
        if reduce_lr_on_plateau:
            kwargs.setdefault("check_val_every_n_epoch", 1)
        return super().train(max_epochs=max_epochs, batch_size=batch_size,
                             early_stopping=early_stopping, lr=lr,
                             reduce_lr_on_plateau=reduce_lr_on_plateau,
                             n_steps_kl_warmup=n_steps_kl_warmup,
                             n_epochs_kl_warmup=n_epochs_kl_warmup, **kwargs)

    def get_latent_representation(self, adata=None, indices=None, give_mean=True,
                                  batch_size=None):
        mod = self.module

        def fn(b):
            inf = mod.inference(b["x"], b["protein"], b["batch"], mod._cats(b), b.get("cont"),
                                sample=not give_mean)
            if give_mean and mod.latent_distribution == "ln":
                eps = torch.randn((500,) + inf["qz_m"].shape, device=inf["qz_m"].device)
                return torch.softmax(inf["qz_m"] + inf["qz_v"].sqrt() * eps, -1).mean(0)
            return inf["qz_m"] if give_mean else inf["z"]

        return self._collect(fn, adata, indices, batch_size)

    def get_latent_library_size(self, adata=None, indices=None, give_mean=True,
                                batch_size=None):
        mod = self.module

        def fn(b):
            inf = mod.inference(b["x"], b["protein"], b["batch"], mod._cats(b), b.get("cont"),
                                sample=not give_mean)
            if give_mean:
                return torch.exp(inf["ql_m"] + 0.5 * inf["ql_v"])
            return inf["library_gene"]

        return self._collect(fn, adata, indices, batch_size)

    def get_normalized_expression(self, adata=None, indices=None, transform_batch=None,
                                  gene_list=None, protein_list=None, library_size=1,
                                  n_samples=1, sample_protein_mixing=False, scale_protein=False,
                                  include_protein_background=False, batch_size=None,
                                  return_mean=True, return_numpy=None):
        """Denoised RNA (``px_scale * library_size``) and foreground protein
        expression; ``transform_batch`` averages over batches.  Returns
        ``(rna, protein)`` DataFrames (arrays with ``return_numpy``)."""
        mod = self.module
        cats = self.registry["batch_categories"]
        tbs = [None] if transform_batch is None else [
            cats.index(t) if t in cats else [str(c) for c in cats].index(str(t))
            for t in (transform_batch if isinstance(transform_batch, (list, tuple))
                      else [transform_batch])]
        G = self.n_vars
        gsel = None if gene_list is None else torch.as_tensor(
            [self.registry["var_names"].index(g) for g in gene_list])
        psel = None if protein_list is None else torch.as_tensor(
            [self.protein_names.index(p) for p in protein_list])

        def fn(b):
            cat = mod._cats(b)
            rna = pro = None
            for tb in tbs:
                inf = mod.inference(b["x"], b["protein"], b["batch"], cat, b.get("cont"),
                                    n_samples=n_samples)
                gen = mod.generative(inf["z"], inf["library_gene"], b["batch"], cat,
                                     b.get("cont"), b.get("labels"), transform_batch=tb)
                px_, py_ = gen["px_"], gen["py_"]
                r = px_["rate"] if library_size == "latent" else px_["scale"] * library_size
                mix = torch.sigmoid(py_["mixing"])
                if sample_protein_mixing:
                    mix = torch.bernoulli(mix)
                p = py_["rate_fore"] * (1 - mix)
                if include_protein_background:
                    p = p + py_["rate_back"] * mix
                if scale_protein:
                    p = F.normalize(p, p=1, dim=-1)
                rna = r if rna is None else rna + r
                pro = p if pro is None else pro + p
            rna, pro = rna / len(tbs), pro / len(tbs)
            if gsel is not None:
                rna = rna[..., gsel.to(rna.device)]
            if psel is not None:
                pro = pro[..., psel.to(pro.device)]
            out = torch.cat([rna, pro], -1)
            if n_samples > 1:
                out = out.mean(0) if return_mean else out.permute(1, 2, 0)
            return out

        out = self._collect(fn, adata, indices, batch_size)
        ng = G if gsel is None else len(gsel)
        rna, pro = out[:, :ng], out[:, ng:]
        if return_numpy or (n_samples > 1 and not return_mean):
            return rna, pro
        ad = self.adata if adata is None else adata
        idx = ad.obs_names if indices is None else ad.obs_names[np.asarray(indices)]
        gnames = gene_list if gene_list is not None else self.registry["var_names"]
        pnames = protein_list if protein_list is not None else self.protein_names
        return (pd.DataFrame(rna, index=idx, columns=gnames),
                pd.DataFrame(pro, index=idx, columns=pnames))

    def get_protein_foreground_probability(self, adata=None, indices=None, transform_batch=None,
                                           protein_list=None, n_samples=1, batch_size=None,
                                           return_mean=True, return_numpy=None):
        """Probability that each protein count comes from the foreground."""
        mod = self.module
        psel = None if protein_list is None else torch.as_tensor(
            [self.protein_names.index(p) for p in protein_list])

        def fn(b):
            cat = mod._cats(b)
            inf = mod.inference(b["x"], b["protein"], b["batch"], cat, b.get("cont"),
                                n_samples=n_samples)
            gen = mod.generative(inf["z"], inf["library_gene"], b["batch"], cat, b.get("cont"))
            v = 1 - torch.sigmoid(gen["py_"]["mixing"])
            if psel is not None:
                v = v[..., psel.to(v.device)]
            return v.mean(0) if n_samples > 1 else v

        out = self._collect(fn, adata, indices, batch_size)
        if return_numpy:
            return out
        ad = self.adata if adata is None else adata
        idx = ad.obs_names if indices is None else ad.obs_names[np.asarray(indices)]
        return pd.DataFrame(out, index=idx,
                            columns=protein_list if protein_list is not None else self.protein_names)


__all__ = ["TOTALVI", "TOTALVAEModule"]
