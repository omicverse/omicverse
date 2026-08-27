from typing import Optional

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule
from scvi import REGISTRY_KEYS
from torch.nn.functional import one_hot


class RegressionBackgroundDetectionTechPyroModel(PyroModule):
    r"""
    Given cell type annotation for each cell, the corresponding reference cell type signatures :math:`g_{f,g}`,
    which represent the average mRNA count of each gene `g` in each cell type `f={1, .., F}`,
    are estimated from sc/snRNA-seq data using Negative Binomial regression,
    which allows to robustly combine data across technologies and batches.

    This model combines batches, and treats data :math:`D` as Negative Binomial distributed,
    given mean :math:`\mu` and overdispersion :math:`\alpha`:

    .. math::
        D_{c,g} \sim \mathtt{NB}(alpha=\alpha_{g}, mu=\mu_{c,g})
    .. math::
        \mu_{c,g} = (\mu_{f,g} + s_{e,g}) * y_e * y_{t,g}

    Which is equivalent to:

    .. math::
        D_{c,g} \sim \mathtt{Poisson}(\mathtt{Gamma}(\alpha_{f,g}, \alpha_{f,g} / \mu_{c,g}))

    Here, :math:`\mu_{f,g}` denotes average mRNA count in each cell type :math:`f` for each gene :math:`g`;
    :math:`y_c` denotes normalisation for each experiment :math:`e` to account for  sequencing depth.
    :math:`y_{t,g}` denotes per gene :math:`g` detection efficiency normalisation for each technology :math:`t`.

    """

    def __init__(
        self,
        n_obs,
        n_vars,
        n_factors,
        n_batch,
        n_extra_categoricals=None,
        alpha_g_phi_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_alpha_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_mean_hyp_prior={
            "alpha": 1.0,
            "beta": 100.0,
        },
        detection_hyp_prior={"mean_alpha": 1.0, "mean_beta": 1.0},
        gene_tech_prior={"mean": 1, "alpha": 200},
        init_vals: Optional[dict] = None,
    ):
        """

        Parameters
        ----------
        n_obs
        n_vars
        n_factors
        n_batch
        n_extra_categoricals
        alpha_g_phi_hyp_prior
        gene_add_alpha_hyp_prior
        gene_add_mean_hyp_prior
        detection_hyp_prior
        gene_tech_prior
        """

        ############# Initialise parameters ################
        super().__init__()

        self.n_obs = n_obs
        self.n_vars = n_vars
        self.n_factors = n_factors
        self.n_batch = n_batch
        self.n_extra_categoricals = n_extra_categoricals

        self.alpha_g_phi_hyp_prior = alpha_g_phi_hyp_prior
        self.gene_add_alpha_hyp_prior = gene_add_alpha_hyp_prior
        self.gene_add_mean_hyp_prior = gene_add_mean_hyp_prior
        self.detection_hyp_prior = detection_hyp_prior
        self.gene_tech_prior = gene_tech_prior

        if (init_vals is not None) & (type(init_vals) is dict):
            self.np_init_vals = init_vals
            for k in init_vals.keys():
                self.register_buffer(f"init_val_{k}", torch.tensor(init_vals[k]))

        self.register_buffer(
            "detection_mean_hyp_prior_alpha",
            torch.tensor(self.detection_hyp_prior["mean_alpha"]),
        )
        self.register_buffer(
            "detection_mean_hyp_prior_beta",
            torch.tensor(self.detection_hyp_prior["mean_beta"]),
        )
        self.register_buffer(
            "gene_tech_prior_alpha",
            torch.tensor(self.gene_tech_prior["alpha"]),
        )
        self.register_buffer(
            "gene_tech_prior_beta",
            torch.tensor(self.gene_tech_prior["alpha"] / self.gene_tech_prior["mean"]),
        )

        self.register_buffer(
            "alpha_g_phi_hyp_prior_alpha",
            torch.tensor(self.alpha_g_phi_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "alpha_g_phi_hyp_prior_beta",
            torch.tensor(self.alpha_g_phi_hyp_prior["beta"]),
        )
        self.register_buffer(
            "gene_add_alpha_hyp_prior_alpha",
            torch.tensor(self.gene_add_alpha_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "gene_add_alpha_hyp_prior_beta",
            torch.tensor(self.gene_add_alpha_hyp_prior["beta"]),
        )
        self.register_buffer(
            "gene_add_mean_hyp_prior_alpha",
            torch.tensor(self.gene_add_mean_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "gene_add_mean_hyp_prior_beta",
            torch.tensor(self.gene_add_mean_hyp_prior["beta"]),
        )

        self.register_buffer("ones", torch.ones((1, 1)))
        self.register_buffer("eps", torch.tensor(1e-8))

    ############# Define the model ################
    @staticmethod
    def _get_fn_args_from_batch_no_cat(tensor_dict):
        x_data = tensor_dict[REGISTRY_KEYS.X_KEY]
        ind_x = tensor_dict["ind_x"].long().squeeze()
        batch_index = tensor_dict[REGISTRY_KEYS.BATCH_KEY]
        label_index = tensor_dict[REGISTRY_KEYS.LABELS_KEY]
        return (x_data, ind_x, batch_index, label_index, label_index), {}

    @staticmethod
    def _get_fn_args_from_batch_cat(tensor_dict):
        x_data = tensor_dict[REGISTRY_KEYS.X_KEY]
        ind_x = tensor_dict["ind_x"].long().squeeze()
        batch_index = tensor_dict[REGISTRY_KEYS.BATCH_KEY]
        label_index = tensor_dict[REGISTRY_KEYS.LABELS_KEY]
        extra_categoricals = tensor_dict[REGISTRY_KEYS.CAT_COVS_KEY]
        return (x_data, ind_x, batch_index, label_index, extra_categoricals), {}

    @property
    def _get_fn_args_from_batch(self):
        if self.n_extra_categoricals is not None:
            return self._get_fn_args_from_batch_cat
        else:
            return self._get_fn_args_from_batch_no_cat

    def create_plates(self, x_data, idx, batch_index, label_index, extra_categoricals):
        return pyro.plate("obs_plate", size=self.n_obs, dim=-2, subsample=idx)

    def list_obs_plate_vars(self):
        """Create a dictionary with the name of observation/minibatch plate,
        indexes of model args to provide to encoder,
        variable names that belong to the observation plate
        and the number of dimensions in non-plate axis of each variable"""

        return {
            "name": "obs_plate",
            "input": [],  # expression data + (optional) batch index
            "input_transform": [],  # how to transform input data before passing to NN
            "sites": {},
        }

    def forward(self, x_data, idx, batch_index, label_index, extra_categoricals):
        obs2sample = one_hot(batch_index.squeeze(-1), self.n_batch).float()
        obs2label = one_hot(label_index.squeeze(-1), self.n_factors).float()
        if self.n_extra_categoricals is not None:
            obs2extra_categoricals = torch.cat(
                [
                    one_hot(
                        extra_categoricals[:, i].view((extra_categoricals.shape[0], 1)).squeeze(-1),
                        n_cat,
                    ).float()
                    for i, n_cat in enumerate(self.n_extra_categoricals)
                ],
                dim=1,
            )

        obs_plate = self.create_plates(x_data, idx, batch_index, label_index, extra_categoricals)

        # =====================Per-cluster average mRNA count ======================= #
        # \mu_{f,g}
        per_cluster_mu_fg = pyro.sample(
            "per_cluster_mu_fg",
            dist.Gamma(self.ones, self.ones).expand([self.n_factors, self.n_vars]).to_event(2),
        )

        # =====================Gene-specific multiplicative component ======================= #
        # `y_{t, g}` per gene multiplicative effect that explains the difference
        # in sensitivity between genes in each technology or covariate effect
        if self.n_extra_categoricals is not None:
            detection_tech_gene_tg = pyro.sample(
                "detection_tech_gene_tg",
                dist.Gamma(
                    self.ones * self.gene_tech_prior_alpha,
                    self.ones * self.gene_tech_prior_beta,
                )
                .expand([np.sum(self.n_extra_categoricals), self.n_vars])
                .to_event(2),
            )

        # =====================Cell-specific detection efficiency ======================= #
        # y_c with hierarchical mean prior
        detection_mean_y_e = pyro.sample(
            "detection_mean_y_e",
            dist.Gamma(
                self.ones * self.detection_mean_hyp_prior_alpha,
                self.ones * self.detection_mean_hyp_prior_beta,
            )
            .expand([self.n_batch, 1])
            .to_event(2),
        )
        detection_y_c = obs2sample @ detection_mean_y_e  # (self.n_obs, 1)

        # =====================Gene-specific additive component ======================= #
        # s_{e,g} accounting for background, free-floating RNA
        s_g_gene_add_alpha_hyp = pyro.sample(
            "s_g_gene_add_alpha_hyp",
            dist.Gamma(self.ones * self.gene_add_alpha_hyp_prior_alpha, self.ones * self.gene_add_alpha_hyp_prior_beta),
        )
        s_g_gene_add_mean = pyro.sample(
            "s_g_gene_add_mean",
            dist.Gamma(
                self.gene_add_mean_hyp_prior_alpha,
                self.gene_add_mean_hyp_prior_beta,
            )
            .expand([self.n_batch, 1])
            .to_event(2),
        )  # (self.n_batch)
        s_g_gene_add_alpha_e_inv = pyro.sample(
            "s_g_gene_add_alpha_e_inv",
            dist.Exponential(s_g_gene_add_alpha_hyp).expand([self.n_batch, 1]).to_event(2),
        )  # (self.n_batch)
        s_g_gene_add_alpha_e = self.ones / s_g_gene_add_alpha_e_inv.pow(2)

        s_g_gene_add = pyro.sample(
            "s_g_gene_add",
            dist.Gamma(s_g_gene_add_alpha_e, s_g_gene_add_alpha_e / s_g_gene_add_mean)
            .expand([self.n_batch, self.n_vars])
            .to_event(2),
        )  # (self.n_batch, n_vars)

        # =====================Gene-specific overdispersion ======================= #
        alpha_g_phi_hyp = pyro.sample(
            "alpha_g_phi_hyp",
            dist.Gamma(self.ones * self.alpha_g_phi_hyp_prior_alpha, self.ones * self.alpha_g_phi_hyp_prior_beta),
        )
        alpha_g_inverse = pyro.sample(
            "alpha_g_inverse",
            dist.Exponential(alpha_g_phi_hyp).expand([1, self.n_vars]).to_event(2),
        )  # (self.n_batch or 1, self.n_vars)

        # =====================Expected expression ======================= #

        # overdispersion
        alpha = self.ones / alpha_g_inverse.pow(2)
        # biological expression
        mu = (
            obs2label @ per_cluster_mu_fg + obs2sample @ s_g_gene_add  # contaminating RNA
        ) * detection_y_c  # cell-specific normalisation
        if self.n_extra_categoricals is not None:
            # gene-specific normalisation for covatiates
            mu = mu * (obs2extra_categoricals @ detection_tech_gene_tg)
        # total_count, logits = _convert_mean_disp_to_counts_logits(
        #    mu, alpha, eps=self.eps
        # )

        # =====================DATA likelihood ======================= #
        # Likelihood (sampling distribution) of data_target & add overdispersion via NegativeBinomial
        with obs_plate:
            pyro.sample(
                "data_target",
                dist.GammaPoisson(concentration=alpha, rate=alpha / mu),
                # dist.NegativeBinomial(total_count=total_count, logits=logits),
                obs=x_data,
            )

    # =====================Other functions======================= #
    def compute_expected(self, samples, adata_manager, ind_x=None):
        r"""Compute expected expression of each gene in each cell. Useful for evaluating how well
        the model learned expression pattern of all genes in the data.

        Parameters
        ----------
        samples
            dictionary with values of the posterior
        adata
            registered anndata
        ind_x
            indices of cells to use (to reduce data size)
        """
        if ind_x is None:
            ind_x = np.arange(adata_manager.adata.n_obs).astype(int)
        else:
            ind_x = ind_x.astype(int)
        obs2sample = adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
        obs2sample = pd.get_dummies(obs2sample.flatten()).values[ind_x, :].astype("float32")
        obs2label = adata_manager.get_from_registry(REGISTRY_KEYS.LABELS_KEY)
        obs2label = pd.get_dummies(obs2label.flatten()).values[ind_x, :].astype("float32")
        if self.n_extra_categoricals is not None:
            extra_categoricals = adata_manager.get_from_registry(REGISTRY_KEYS.CAT_COVS_KEY)
            obs2extra_categoricals = np.concatenate(
                [
                    pd.get_dummies(extra_categoricals.iloc[ind_x, i]).astype("float32")
                    for i, n_cat in enumerate(self.n_extra_categoricals)
                ],
                axis=1,
            )

        alpha = 1 / np.power(samples["alpha_g_inverse"], 2)

        mu = (np.dot(obs2label, samples["per_cluster_mu_fg"]) + np.dot(obs2sample, samples["s_g_gene_add"])) * np.dot(
            obs2sample, samples["detection_mean_y_e"]
        )  # samples["detection_y_c"][ind_x, :]
        if self.n_extra_categoricals is not None:
            mu = mu * np.dot(obs2extra_categoricals, samples["detection_tech_gene_tg"])

        return {"mu": mu, "alpha": alpha}

    def compute_expected_subset(self, samples, adata_manager, fact_ind, cell_ind):
        r"""Compute expected expression of each gene in each cell that comes from
        a subset of factors (cell types) or cells.

        Useful for evaluating how well the model learned expression pattern of all genes in the data.

        Parameters
        ----------
        samples
            dictionary with values of the posterior
        adata
            registered anndata
        fact_ind
            indices of factors/cell types to use
        cell_ind
            indices of cells to use
        """
        obs2sample = adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
        obs2sample = pd.get_dummies(obs2sample.flatten())
        obs2label = adata_manager.get_from_registry(REGISTRY_KEYS.LABELS_KEY)
        obs2label = pd.get_dummies(obs2label.flatten())
        if self.n_extra_categoricals is not None:
            extra_categoricals = adata_manager.get_from_registry(REGISTRY_KEYS.CAT_COVS_KEY)
            obs2extra_categoricals = np.concatenate(
                [pd.get_dummies(extra_categoricals.iloc[:, i]) for i, n_cat in enumerate(self.n_extra_categoricals)],
                axis=1,
            )

        alpha = 1 / np.power(samples["alpha_g_inverse"], 2)

        mu = (
            np.dot(obs2label[cell_ind, fact_ind], samples["per_cluster_mu_fg"][fact_ind, :])
            + np.dot(obs2sample[cell_ind, :], samples["s_g_gene_add"])
        ) * np.dot(
            obs2sample, samples["detection_mean_y_e"]
        )  # samples["detection_y_c"]
        if self.n_extra_categoricals is not None:
            mu = mu * np.dot(obs2extra_categoricals[cell_ind, :], samples["detection_tech_gene_tg"])

        return {"mu": mu, "alpha": alpha}

    def normalise(self, samples, adata_manager, adata):
        r"""Normalise expression data by estimated technical variables.

        Parameters
        ----------
        samples
            dictionary with values of the posterior
        adata
            registered anndata

        """
        obs2sample = adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
        obs2sample = pd.get_dummies(obs2sample.flatten())
        if self.n_extra_categoricals is not None:
            extra_categoricals = adata_manager.get_from_registry(REGISTRY_KEYS.CAT_COVS_KEY)
            obs2extra_categoricals = np.concatenate(
                [pd.get_dummies(extra_categoricals.iloc[:, i]) for i, n_cat in enumerate(self.n_extra_categoricals)],
                axis=1,
            )
        # get counts matrix
        corrected = adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        # normalise per-sample scaling
        corrected = corrected / np.dot(obs2sample, samples["detection_mean_y_e"])
        # normalise per gene effects
        if self.n_extra_categoricals is not None:
            corrected = corrected / np.dot(obs2extra_categoricals, samples["detection_tech_gene_tg"])

        # remove additive sample effects
        corrected = corrected - np.dot(obs2sample, samples["s_g_gene_add"])

        # set minimum value to 0 for each gene (a hack to avoid negative values)
        corrected = corrected - corrected.min()

        return corrected
