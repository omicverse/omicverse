"""scANVI (Xu et al., 2021): semi-supervised scVI with a label classifier."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from .._registry import register_function
from ._base import BaseVIModel, _codes, _max_epochs_heuristic
from ._data import split_train_val
from ._distributions import kl_normal_normal, kl_normal_standard
from ._nn import Classifier, Encoder, FCLayers
from ._scvi import SCVI, VAEModule, _flags


class DecoderZ(nn.Module):
    """``p(z1 | z2, y)`` (scvi-tools ``Decoder``): mean and variance of a Gaussian."""

    def __init__(self, n_input, n_output, n_cat_list=None, n_layers=1, n_hidden=128, **kwargs):
        super().__init__()
        self.decoder = FCLayers(n_input, n_hidden, n_cat_list=n_cat_list, n_layers=n_layers,
                                n_hidden=n_hidden, dropout_rate=0, **kwargs)
        self.mean_decoder = nn.Linear(n_hidden, n_output)
        self.var_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, x, *cats):
        p = self.decoder(x, *cats)
        return self.mean_decoder(p), torch.exp(self.var_decoder(p))


class SCANVAEModule(VAEModule):
    """scvi-tools ``SCANVAE``: the scVI VAE plus ``q(y | z1)``, ``q(z2 | z1, y)`` and
    ``p(z1 | z2, y)``.  The unsupervised loss marginalises the label of every
    cell with the classifier's probabilities; labelled cells (a second
    minibatch, ``batch['labelled']``) add ``classification_ratio`` x the
    cross-entropy of the classifier on their posterior mean."""

    def __init__(self, n_input, n_batch=0, n_labels=0, n_hidden=128, n_latent=10, n_layers=1,
                 dropout_rate=0.1, use_batch_norm="both", use_layer_norm="none",
                 linear_classifier=False, classifier_parameters=None, y_prior=None,
                 classification_ratio=50.0, **vae_kwargs):
        super().__init__(n_input, n_batch=n_batch, n_labels=n_labels, n_hidden=n_hidden,
                         n_latent=n_latent, n_layers=n_layers, dropout_rate=dropout_rate,
                         use_batch_norm=use_batch_norm, use_layer_norm=use_layer_norm,
                         **vae_kwargs)
        bn_enc, bn_dec = _flags(use_batch_norm)
        ln_enc, ln_dec = _flags(use_layer_norm)
        cls = {"n_layers": 0 if linear_classifier else n_layers,
               "n_hidden": 0 if linear_classifier else n_hidden,
               "dropout_rate": dropout_rate, "logits": True}
        cls.update(classifier_parameters or {})
        self.classifier = Classifier(n_latent, n_labels=n_labels, use_batch_norm=bn_enc,
                                     use_layer_norm=ln_enc, **cls)
        self.encoder_z2_z1 = Encoder(n_latent, n_latent, n_cat_list=[n_labels], n_layers=n_layers,
                                     n_hidden=n_hidden, dropout_rate=dropout_rate,
                                     use_batch_norm=bn_enc, use_layer_norm=ln_enc)
        self.decoder_z1_z2 = DecoderZ(n_latent, n_latent, n_cat_list=[n_labels],
                                      n_layers=n_layers, n_hidden=n_hidden,
                                      use_batch_norm=bn_dec, use_layer_norm=ln_dec)
        yp = (torch.as_tensor(np.asarray(y_prior), dtype=torch.float32).reshape(1, -1)
              if y_prior is not None else torch.ones(1, n_labels) / n_labels)
        self.y_prior = nn.Parameter(yp, requires_grad=False)
        self.classification_ratio = float(classification_ratio)

    def classify(self, x, batch_index, cat=(), cont=None, use_posterior_mean=True):
        """Logits of ``q(y | z1)``."""
        inf = self.inference(x, batch_index, cat, cont, sample=not use_posterior_mean)
        z = inf["qz_m"] if use_posterior_mean else inf["z"]
        return self.classifier(z)

    def forward(self, b, kl_weight=1.0):
        cat = self._cats(b)
        inf = self.inference(b["x"], b["batch"], cat, b.get("cont"))
        gen = self.generative(inf["z"], inf["library"], b["batch"], cat, b.get("cont"))
        rec = -self.log_likelihood(b, gen)
        z1, qm, qv = inf["z"], inf["qz_m"], inf["qz_v"]
        K, B = self.n_labels, z1.shape[0]
        ys = torch.arange(K, device=z1.device).repeat_interleave(B)       # (K*B,)
        z1s = z1.repeat(K, 1)                                             # (K*B, L)
        q2m, q2v, z2 = self.encoder_z2_z1(z1s, ys)
        p1m, p1v = self.decoder_z1_z2(z2, ys)
        kl_z2 = kl_normal_standard(q2m, q2v)                              # (K*B,)
        # -log N(z1s; p1m, p1v) per (label, cell)
        loss_z1_unweight = 0.5 * (((z1s - p1m) ** 2) / p1v + torch.log(p1v)
                                  + np.log(2 * np.pi)).sum(-1)
        # log q(z1 | x) at its own sample
        loss_z1_weight = (-0.5 * (((z1 - qm) ** 2) / qv + torch.log(qv)
                                  + np.log(2 * np.pi))).sum(-1)
        probs = F.softmax(self.classifier(z1), dim=-1)                    # (B, K)
        rec = rec + loss_z1_weight + (loss_z1_unweight.view(K, B).t() * probs).sum(-1)
        kl = (kl_z2.view(K, B).t() * probs).sum(-1)
        yp = self.y_prior.expand_as(probs)
        kl = kl + (probs * (torch.log(probs + 1e-30) - torch.log(yp))).sum(-1)
        if not self.use_observed_lib_size:
            n = self.library_log_means.shape[1]
            oh = F.one_hot(b["batch"].reshape(-1), n).float()
            pm, pv = oh @ self.library_log_means.t(), oh @ self.library_log_vars.t()
            kl = kl + kl_normal_normal(inf["ql_m"], inf["ql_v"], pm, pv).sum(1)
        loss = torch.mean(rec + kl * kl_weight)
        out = {"reconstruction_loss": rec, "kl_local": kl,
               "kl_global": torch.zeros((), device=rec.device)}
        lab = b.get("labelled")
        if lab is not None:
            logits = self.classify(lab["x"], lab["batch"], self._cats(lab), lab.get("cont"))
            ce = F.cross_entropy(logits, lab["labels"].reshape(-1).long())
            loss = loss + ce * self.classification_ratio
            out["classification_loss"] = ce
            out["_accuracy"] = (logits.argmax(-1) == lab["labels"].reshape(-1)).float().mean()
        out["loss"] = loss
        return out


class _AuxRows:
    """Labelled training / validation cells per epoch (picklable for DDP workers);
    with ``n_samples_per_label`` a per-epoch subsample of each class."""

    def __init__(self, train, val, labels, n_samples_per_label=None, seed=0):
        self.train, self.val = np.asarray(train), np.asarray(val)
        self.labels = np.asarray(labels)
        self.n = n_samples_per_label
        self.seed = seed

    def __call__(self, epoch, which):
        rows = self.train if which == "train" else self.val
        if self.n is None or which != "train" or not len(rows):
            return rows
        rs = np.random.RandomState((self.seed * 7919 + epoch) % (2**32))
        out = []
        for c in np.unique(self.labels[rows]):
            loc = rows[self.labels[rows] == c]
            out.append(loc if len(loc) <= self.n else rs.choice(loc, self.n, replace=False))
        return np.sort(np.concatenate(out))


@register_function(
    aliases=["scANVI", "scanvi", "SCANVI", "半监督注释", "label transfer", "vi.SCANVI"],
    category="single",
    description="omicverse-native scANVI (no scvi-tools): semi-supervised scVI for label "
                "transfer and integration; trains on AnnData, AnnDataOOM and Dask-backed data.",
    examples=[
        "m = ov.vi.SCANVI(adata, layer='counts', batch_key='batch', labels_key='cell_type',"
        " unlabeled_category='Unknown')",
        "m.train(); adata.obs['pred'] = m.predict()",
        "m = ov.vi.SCANVI.from_scvi_model(scvi_model, unlabeled_category='Unknown',"
        " labels_key='cell_type')",
    ],
    related=["vi.SCVI", "single.batch_correction"],
)
class SCANVI(SCVI):
    """Single-cell ANnotation using Variational Inference (scvi-tools-compatible API).

    Arguments:
        adata: AnnData, AnnDataOOM or Dask-backed AnnData (raw counts in ``layer``).
        labels_key: ``obs`` column with cell-type labels.
        unlabeled_category: Value of ``labels_key`` marking cells without a label
            (their labels are predicted).
        linear_classifier: Linear classifier instead of an MLP.
        classification_ratio: Weight of the classification loss (default 50, as
            scvi-tools).
        Other arguments: as :class:`SCVI` (defaults n_hidden 128, n_latent 10,
            n_layers 1, dropout 0.1, dispersion 'gene', gene_likelihood 'zinb').

    Example::

        m = ov.vi.SCANVI(adata, layer="counts", batch_key="batch",
                         labels_key="cell_type", unlabeled_category="Unknown")
        m.train()
        adata.obsm["X_scANVI"] = m.get_latent_representation()
        adata.obs["scANVI_pred"] = m.predict()
    """

    _module_cls = SCANVAEModule

    def __init__(self, adata, layer=None, batch_key=None, labels_key=None,
                 unlabeled_category="Unknown", categorical_covariate_keys=None,
                 continuous_covariate_keys=None, genes=None, n_hidden=128, n_latent=10,
                 n_layers=1, dropout_rate=0.1, dispersion="gene", gene_likelihood="zinb",
                 linear_classifier=False, seed=0, registry=None, **module_kwargs):
        if labels_key is None and registry is None:
            raise ValueError("SCANVI needs labels_key=")
        if dispersion == "gene-label":
            raise ValueError("dispersion='gene-label' is not supported by SCANVI")
        self.unlabeled_category_ = unlabeled_category
        BaseVIModel.__init__(self, adata, layer=layer, batch_key=batch_key,
                             labels_key=labels_key,
                             categorical_covariate_keys=categorical_covariate_keys,
                             continuous_covariate_keys=continuous_covariate_keys, genes=genes,
                             registry=registry)
        if registry is None:
            cats = [c for c in self.registry["label_categories"]
                    if str(c) != str(unlabeled_category)]
            self.registry["label_categories"] = cats
            self.registry["unlabeled_category"] = unlabeled_category
        self.unlabeled_category_ = self.registry.get("unlabeled_category", unlabeled_category)
        self.was_pretrained = False
        kw = dict(n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers,
                  dropout_rate=dropout_rate, dispersion=dispersion,
                  gene_likelihood=gene_likelihood, linear_classifier=linear_classifier,
                  **module_kwargs)
        if not kw.get("use_observed_lib_size", True) and "library_log_means" not in kw:
            from ._scvi import _library_size_priors
            m, v = _library_size_priors(self, adata)
            kw.update(library_log_means=m, library_log_vars=v)
        self._build(seed=seed, **kw)
        self.n_labels_ = self.n_labels

    # labels: real categories 0..K-1, unlabeled -> K (excluded from the classifier)
    def _label_codes(self, adata):
        r = self.registry
        vals = pd.Series(np.asarray(adata.obs[r["labels_key"]].values)).astype(object)
        unl = vals.astype(str) == str(self.registry.get(
            "unlabeled_category", getattr(self, "unlabeled_category_", "Unknown")))
        codes = np.full(len(vals), len(r["label_categories"]), dtype=np.int64)
        if (~unl).any():
            codes[~unl.values] = _codes(vals[~unl].values, r["label_categories"], r["labels_key"])
        return codes

    @property
    def labels_(self):
        return np.asarray(self.adata.obs[self.registry["labels_key"]].values)

    @classmethod
    def from_scvi_model(cls, scvi_model, unlabeled_category="Unknown", labels_key=None,
                        adata=None, **scanvi_kwargs):
        """Initialise from a trained :class:`SCVI` (same genes / batches / layer);
        its weights seed the shared encoder / decoder."""
        adata = scvi_model.adata if adata is None else adata
        r = scvi_model.registry
        labels_key = labels_key or r["labels_key"]
        if labels_key is None:
            raise ValueError("labels_key= is needed (the SCVI model has no labels)")
        init = {k: v for k, v in scvi_model._module_kwargs.items()
                if k not in ("library_log_means", "library_log_vars") or
                not scvi_model._module_kwargs.get("use_observed_lib_size", True)}
        init.update(scanvi_kwargs)
        genes = None if scvi_model._genes_for(adata) is None else scvi_model.registry["var_names"]
        m = cls(adata, layer=r["layer"], batch_key=r["batch_key"], labels_key=labels_key,
                unlabeled_category=unlabeled_category,
                categorical_covariate_keys=r["categorical_covariate_keys"] or None,
                continuous_covariate_keys=r["continuous_covariate_keys"] or None, genes=genes,
                **init)
        state = {k: v for k, v in scvi_model.module.state_dict().items()
                 if k in m.module.state_dict() and
                 m.module.state_dict()[k].shape == v.shape}
        m.module.load_state_dict(state, strict=False)
        m.was_pretrained = True
        return m

    def _configure_training(self, cfg):
        super()._configure_training(cfg)
        if "classification_ratio" in cfg.extra:
            self.module.classification_ratio = float(cfg.extra.pop("classification_ratio"))

    def train(self, max_epochs=None, n_samples_per_label=None, train_size=0.9,
              validation_size=None, shuffle_set_split=True, batch_size=128, seed=0, **kwargs):
        """Train (scvi-tools semi-supervised defaults: train_size 0.9 split separately
        within labelled and unlabelled cells, classification ratio 50; a pretrained
        model trains for ``min(10, max(2, round(heuristic / 3)))`` epochs)."""
        labels = self._label_codes(self.adata)
        K = len(self.registry["label_categories"])
        lab = np.flatnonzero(labels < K)
        unl = np.flatnonzero(labels == K)
        tr, va = [], []
        for idx in (lab, unl):
            if len(idx):
                t, v = split_train_val(len(idx), train_size, validation_size, batch_size, seed,
                                       shuffle_set_split)
                perm = idx[np.random.RandomState(seed).permutation(len(idx))] \
                    if shuffle_set_split else idx
                n_val = len(v)
                tr.append(np.sort(perm[n_val:n_val + len(t)]))
                va.append(np.sort(perm[:n_val]))
        train = np.sort(np.concatenate(tr)) if tr else np.zeros(0, np.int64)
        val = np.sort(np.concatenate(va)) if va else np.zeros(0, np.int64)
        lab_set = np.zeros(len(labels), bool)
        lab_set[lab] = True
        self._aux_rows = _AuxRows(train[lab_set[train]], val[lab_set[val]], labels,
                                  n_samples_per_label, seed)
        self._split_override = (train, val)
        if max_epochs is None:
            max_epochs = _max_epochs_heuristic(self.adata.n_obs)
            if getattr(self, "was_pretrained", False):
                max_epochs = int(min(10, max(2, round(max_epochs / 3.0))))
        return super().train(max_epochs=max_epochs, batch_size=batch_size, seed=seed,
                             train_size=train_size, validation_size=validation_size,
                             shuffle_set_split=shuffle_set_split, **kwargs)

    def predict(self, adata=None, indices=None, soft: bool = False,
                batch_size: int | None = None, use_posterior_mean: bool = True):
        """Predicted labels (``soft=True``: probabilities as a DataFrame)."""
        mod = self.module

        def fn(b):
            logits = mod.classify(b["x"], b["batch"], mod._cats(b), b.get("cont"),
                                  use_posterior_mean=use_posterior_mean)
            return torch.softmax(logits, -1)

        probs = self._collect(fn, adata, indices, batch_size)
        cats = self.registry["label_categories"]
        if soft:
            ad = self.adata if adata is None else adata
            idx = ad.obs_names if indices is None else ad.obs_names[np.asarray(indices)]
            return pd.DataFrame(probs, index=idx, columns=cats)
        return np.asarray(cats, dtype=object)[probs.argmax(1)]


__all__ = ["SCANVI", "SCANVAEModule"]
