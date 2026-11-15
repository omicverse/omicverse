"""Neural-network building blocks of the ``ov.vi`` models.

The layers have the same structure, defaults and parameter layout as their
scvi-tools counterparts (``FCLayers``, ``Encoder``, ``DecoderSCVI``,
``Classifier``): a linear layer acting on ``[x, continuous covariates,
one-hot categories]`` followed by batch-norm (momentum 0.01, eps 1e-3),
layer-norm (no affine), ReLU and dropout.  The one-hot part is never
materialised: ``W @ [x, onehot(c)]`` is computed as ``W_x @ x + W_c[:, c]``
(a gather), which is exact and avoids concatenating a ``(B, n_batch)``
matrix in every layer.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _norm_cat_list(n_cat_list):
    # categories with a single level carry no information (as scvi-tools)
    return [int(n) if n is not None and int(n) > 1 else 0 for n in (n_cat_list or [])]


class _InjectLinear(nn.Module):
    """Linear layer on ``[x, cont, onehot(cat_1), ...]`` with gathered one-hots."""

    def __init__(self, n_in: int, n_out: int, n_cont: int, n_cat_list: Sequence[int],
                 bias: bool = True):
        super().__init__()
        self.n_in, self.n_cont = int(n_in), int(n_cont)
        self.n_cat_list = list(n_cat_list)
        n_cov = self.n_cont + sum(self.n_cat_list)
        # one nn.Linear so the initialisation (fan_in = n_in + n_cov) and the
        # state-dict layout match scvi-tools exactly
        self.linear = nn.Linear(self.n_in + n_cov, n_out, bias=bias)
        offs, o = [], self.n_in + self.n_cont
        for n in self.n_cat_list:
            offs.append(o)
            o += n
        self._offsets = offs

    def forward(self, x, cats=(), cont=None):
        W, b = self.linear.weight, self.linear.bias
        if self.n_cont:
            x = torch.cat([x, cont.expand(*x.shape[:-1], cont.shape[-1]) if x.dim() == 3 else cont], -1)
        h = F.linear(x, W[:, :self.n_in + self.n_cont], b)
        for n, off, c in zip(self.n_cat_list, self._offsets, cats):
            if n:
                h = h + F.embedding(c.reshape(-1).long() + off, W.t())
        return h


class FCLayers(nn.Module):
    """Stack of fully connected layers with covariate injection (scvi-tools ``FCLayers``).

    Arguments:
        n_in, n_out: Input / output widths.
        n_cat_list: Number of categories of each categorical covariate (one-hot).
        n_cont: Number of continuous covariates.
        n_layers, n_hidden: Depth and hidden width.
        dropout_rate: Dropout after each layer.
        use_batch_norm, use_layer_norm, use_activation, bias: Layer options.
        inject_covariates: Covariates enter every layer (``True``) or the first only.
        activation_fn: Activation module class.
    """

    def __init__(self, n_in: int, n_out: int, n_cat_list: Iterable[int] | None = None,
                 n_cont: int = 0, n_layers: int = 1, n_hidden: int = 128,
                 dropout_rate: float = 0.1, use_batch_norm: bool = True,
                 use_layer_norm: bool = False, use_activation: bool = True, bias: bool = True,
                 inject_covariates: bool = True, activation_fn=nn.ReLU):
        super().__init__()
        self.n_cat_list = _norm_cat_list(n_cat_list)
        self.n_cont = int(n_cont)
        self.inject_covariates = inject_covariates
        dims = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self._inject = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            inject = i == 0 or inject_covariates
            self._inject.append(inject)
            self.layers.append(_InjectLinear(a, b, self.n_cont if inject else 0,
                                             self.n_cat_list if inject else [], bias=bias))
            post = []
            if use_batch_norm:
                post.append(nn.BatchNorm1d(b, momentum=0.01, eps=0.001))
            if use_layer_norm:
                post.append(nn.LayerNorm(b, elementwise_affine=False))
            if use_activation:
                post.append(activation_fn())
            if dropout_rate > 0:
                post.append(nn.Dropout(p=dropout_rate))
            self.norms.append(nn.Sequential(*post))

    def forward(self, x, *cats, cont=None):
        for lin, post, inj in zip(self.layers, self.norms, self._inject):
            x = lin(x, cats if inj else (), cont if inj else None)
            if x.dim() == 3:  # (n_samples, n_cells, width)
                if self.training:
                    x = torch.stack([post(s) for s in x])
                else:
                    s, n, w = x.shape
                    x = post(x.reshape(s * n, w)).reshape(s, n, w)
            else:
                x = post(x)
        return x


class Encoder(nn.Module):
    """Gaussian (or logistic-normal) encoder ``q(z | x)``.

    Returns ``(q_m, q_v, z)`` with ``q_v = exp(.) + var_eps`` and ``z`` a
    reparameterised sample (softmax-transformed for ``distribution='ln'``)."""

    def __init__(self, n_input: int, n_output: int, n_cat_list=None, n_layers: int = 1,
                 n_hidden: int = 128, dropout_rate: float = 0.1, distribution: str = "normal",
                 var_eps: float = 1e-4, var_activation=None, n_cont: int = 0, **kwargs):
        super().__init__()
        self.distribution = distribution
        self.var_eps = var_eps
        self.encoder = FCLayers(n_input, n_hidden, n_cat_list=n_cat_list, n_cont=n_cont,
                                n_layers=n_layers, n_hidden=n_hidden,
                                dropout_rate=dropout_rate, **kwargs)
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.var_activation = torch.exp if var_activation is None else var_activation

    def z_transformation(self, z):
        return torch.softmax(z, dim=-1) if self.distribution == "ln" else z

    def forward(self, x, *cats, cont=None, sample: bool = True):
        q = self.encoder(x, *cats, cont=cont)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        if not sample:
            return q_m, q_v, self.z_transformation(q_m)
        z = q_m + q_v.sqrt() * torch.randn_like(q_m)
        return q_m, q_v, self.z_transformation(z)


class DecoderSCVI(nn.Module):
    """``p(x | z, s)`` parameters: softmax ``px_scale``, ``px_rate = exp(library) * px_scale``,
    gene-cell dispersion and zero-inflation logits (only computed when needed)."""

    def __init__(self, n_input: int, n_output: int, n_cat_list=None, n_layers: int = 1,
                 n_hidden: int = 128, inject_covariates: bool = True,
                 use_batch_norm: bool = False, use_layer_norm: bool = False,
                 scale_activation: str = "softmax", need_dropout: bool = True,
                 need_r: bool = False, n_cont: int = 0):
        super().__init__()
        self.px_decoder = FCLayers(n_input, n_hidden, n_cat_list=n_cat_list, n_cont=n_cont,
                                   n_layers=n_layers, n_hidden=n_hidden, dropout_rate=0,
                                   inject_covariates=inject_covariates,
                                   use_batch_norm=use_batch_norm, use_layer_norm=use_layer_norm)
        self.scale_activation = scale_activation
        self.px_scale_decoder = nn.Linear(n_hidden, n_output)
        self.px_r_decoder = nn.Linear(n_hidden, n_output) if need_r else None
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output) if need_dropout else None

    def forward(self, z, library, *cats, cont=None):
        px = self.px_decoder(z, *cats, cont=cont)
        logits = self.px_scale_decoder(px)
        if self.scale_activation == "softmax":
            px_scale = torch.softmax(logits, dim=-1)
        else:
            px_scale = F.softplus(logits)
        px_dropout = self.px_dropout_decoder(px) if self.px_dropout_decoder is not None else None
        px_rate = torch.exp(library) * px_scale
        px_r = self.px_r_decoder(px) if self.px_r_decoder is not None else None
        return px_scale, px_r, px_rate, px_dropout


class Classifier(nn.Module):
    """MLP classifier (scvi-tools ``Classifier``): FCLayers + linear, softmax unless ``logits``."""

    def __init__(self, n_input: int, n_hidden: int = 128, n_labels: int = 5, n_layers: int = 1,
                 dropout_rate: float = 0.1, logits: bool = False, use_batch_norm: bool = True,
                 use_layer_norm: bool = False, activation_fn=nn.ReLU, **kwargs):
        super().__init__()
        self.logits = logits
        layers = []
        if n_hidden > 0 and n_layers > 0:
            layers.append(FCLayers(n_input, n_hidden, n_layers=n_layers, n_hidden=n_hidden,
                                   dropout_rate=dropout_rate, use_batch_norm=use_batch_norm,
                                   use_layer_norm=use_layer_norm, activation_fn=activation_fn,
                                   **kwargs))
        else:
            n_hidden = n_input
        layers.append(nn.Linear(n_hidden, n_labels))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        out = self.classifier(x)
        return out if self.logits else torch.softmax(out, dim=-1)


__all__ = ["FCLayers", "Encoder", "DecoderSCVI", "Classifier"]
