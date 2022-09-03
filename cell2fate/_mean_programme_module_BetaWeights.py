from typing import Optional

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from pyro.distributions.transforms import ExpTransform
from pyro.nn import PyroModule
from scvi import REGISTRY_KEYS
from scvi.nn import one_hot
from torch.distributions import biject_to, transform_to

class _ExpPositive(type(constraints.positive)):
    def __init__(self):
        super().__init__(lower_bound=0.0)

exp_positive = _ExpPositive()

@biject_to.register(exp_positive)
@transform_to.register(exp_positive)
def _transform_to_positive(constraint):
    return ExpTransform()

class MeanProgrammePyroModel(PyroModule):
    """
    Two State programme model treats RNA count data :math:`D` as Poisson distributed,
    given transcription rate :math:`\mu_{c,g}` and a range of variables accounting for technical effects:

    .. math::
        D_{c,g} \sim \mathtt{NB}(\mu=\mu_{c,g}, \alpha_{a,g})
    .. math::
        \mu_{c,g} = ((\sum_f w_{c,f} g_{f,g}) + s_{e,g}) * y_c * y_{t,g}

    Here, :math:`\mu_{c,g}` denotes expected RNA count :math:`g` in each cell :math:`c`;
    :math:`\alpha_{a,g}` denotes per gene :math:`g` stochatic/unexplained overdispersion for each covariate :math:`a`;
    :math:`w_{c,f}` denotes cell loadings of each factor :math:`f` for each cell :math:`c`;
    :math:`g_{f,g}` denotes gene loadings of each factor :math:`f` for each gene :math:`g`;
    :math:`s_{e,g}` denotes additive background for each gene :math:`g` and for each experiment :math:`e`,
        to account for contaminating RNA;
    :math:`y_c` denotes normalisation for each cell :math:`c` with a prior mean for each experiment :math:`e`,
        to account for RNA detection sensitivity, sequencing depth;
    :math:`y_{t,g}` denotes per gene :math:`g` detection efficiency normalisation for each technology :math:`t`;
    """

    def __init__(
        self,
        n_obs,
        n_vars,
        n_factors,
        n_batch,
        n_extra_categoricals,
        n_var_categoricals,
        gene_bool: np.array,
        factor_prior={"rate": 1.0, "alpha": 1.0, "states_per_gene": 10.0},
        w_cf_prior = {"alpha": 0.25, "beta": 1.0},
        stochastic_v_ag_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_alpha_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_mean_hyp_prior={"alpha": 1.0, "beta": 100.0},
        detection_hyp_prior={"alpha": 20.0, "mean_alpha": 1.0, "mean_beta": 1.0},
        fixed_vals: Optional[dict] = None,
        init_vals: Optional[dict] = None
    ):
        """

        Parameters
        ----------
        n_obs
        n_vars
        n_factors
        n_batch
        gene_bool
        stochastic_v_ag_hyp_prior
        gene_add_alpha_hyp_prior
        gene_add_mean_hyp_prior
        detection_hyp_prior
        use_average_as_initial_value
        """

        ############# Initialise parameters ################
        super().__init__()

        self.n_obs = n_obs
        self.n_vars = n_vars
        self.n_factors = n_factors
        self.n_batch = n_batch
        self.n_extra_categoricals = n_extra_categoricals
        self.n_var_categoricals = n_var_categoricals
        self.w_cf_prior = w_cf_prior
        
        self.gene_bool = gene_bool.astype(int).flatten()
        self.gene_ind = np.where(gene_bool)[0]
        self.n_genes = len(self.gene_ind)
        self.register_buffer("gene_ind_tt", torch.tensor(self.gene_ind))

        self.region_ind = np.where(np.logical_not(gene_bool))[0]
        self.n_regions = self.n_vars - self.n_genes
        self.register_buffer("region_ind_tt", torch.tensor(self.region_ind))

        self.stochastic_v_ag_hyp_prior = stochastic_v_ag_hyp_prior
        self.gene_add_alpha_hyp_prior = gene_add_alpha_hyp_prior
        self.gene_add_mean_hyp_prior = gene_add_mean_hyp_prior

        self.detection_hyp_prior = detection_hyp_prior

        self.factor_prior = factor_prior

        # Initial values
        if (init_vals is not None) & (type(init_vals) is dict):
            self.np_init_vals = init_vals
            for k in init_vals.keys():
                self.register_buffer(f"init_val_{k}", torch.tensor(init_vals[k]))

        # Shared priors
        self.register_buffer(
            "detection_hyp_prior_alpha",
            torch.tensor(self.detection_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "detection_mean_hyp_prior_alpha",
            torch.tensor(self.detection_hyp_prior["mean_alpha"]),
        )
        self.register_buffer(
            "detection_mean_hyp_prior_beta",
            torch.tensor(self.detection_hyp_prior["mean_beta"]),
        )

        # per gene rate priors
        self.register_buffer(
            "factor_prior_alpha",
            torch.tensor(self.factor_prior["alpha"]),
        )
        self.register_buffer(
            "factor_prior_beta",
            torch.tensor(self.factor_prior["alpha"] / self.factor_prior["rate"]),
        )

        # RNA model priors
        self.register_buffer(
            "factor_states_per_gene",
            torch.tensor(self.factor_prior["states_per_gene"]),
        )

        self.register_buffer(
            "stochastic_v_ag_hyp_prior_alpha",
            torch.tensor(self.stochastic_v_ag_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "stochastic_v_ag_hyp_prior_beta",
            torch.tensor(self.stochastic_v_ag_hyp_prior["beta"]),
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
        
        self.register_buffer(
            "w_cf_alpha",
            torch.tensor(self.w_cf_prior["alpha"]),
        )
        self.register_buffer(
            "w_cf_beta",
            torch.tensor(self.w_cf_prior["beta"]),
        )

        self.register_buffer("ones", torch.ones((1, 1)))
        self.register_buffer("zeros", torch.zeros((1, 1)))
        self.register_buffer("n_factors_torch", torch.tensor(self.n_factors))
        self.register_buffer("ones_1_n_factors", torch.ones((1, self.n_factors)))
        self.register_buffer("eps", torch.tensor(1e-8))
        self.register_buffer("one", torch.tensor(1.))

    ############# Define the model ################
    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        x_data = tensor_dict['spliced']
        ind_x = tensor_dict["ind_x"].long().squeeze()
        batch_index = tensor_dict[REGISTRY_KEYS.BATCH_KEY]
        label_index = tensor_dict[REGISTRY_KEYS.LABELS_KEY]
        rna_index = tensor_dict["rna_index"].bool()
        extra_categoricals = tensor_dict[REGISTRY_KEYS.CAT_COVS_KEY]
        var_categoricals = tensor_dict["var_categoricals"].long()
        return (
            x_data,
            ind_x,
            batch_index,
            label_index,
            rna_index,
            extra_categoricals,
            var_categoricals,
        ), {}

    def create_plates(
        self,
        x_data,
        idx,
        batch_index,
        label_index,
        rna_index,
        extra_categoricals,
        var_categoricals,
    ):
        return pyro.plate("obs_plate", size=self.n_obs, dim=-2, subsample=idx)

    def list_obs_plate_vars(self):
        """Create a dictionary with the name of observation/minibatch plate,
        indexes of model args to provide to encoder,
        variable names that belong to the observation plate
        and the number of dimensions in non-plate axis of each variable"""

        return {
            "name": "obs_plate",
            "input": [0, 2, 4],  # expression data + (optional) batch index
            "input_transform": [
                torch.log1p,
                lambda x: x,
                lambda x: x,
            ],  # how to transform input data before passing to NN
            "input_normalisation": [
                False,
                False,
                False,
            ],  # whether to normalise input data before passing to NN
            "sites": {
                "detection_y_c": 1,
                "factors_per_cell": 1,
                "cell_factors_w_cf": self.n_factors,
            },
        }

    def forward(
        self,
        x_data,
        idx,
        batch_index,
        label_index,
        rna_index,
        extra_categoricals,
        var_categoricals,
    ):
        
        batch_size = len(idx)
        obs2sample = one_hot(batch_index, self.n_batch)
        obs2extra_categoricals = torch.cat(
            [
                one_hot(
                    extra_categoricals[:, i].view((extra_categoricals.shape[0], 1)),
                    n_cat,
                )
                for i, n_cat in enumerate(self.n_extra_categoricals)
            ],
            dim=1,
        )
        obs2var_categoricals = one_hot(var_categoricals, self.n_var_categoricals)
        obs_plate = self.create_plates(
            x_data,
            idx,
            batch_index,
            label_index,
            rna_index,
            extra_categoricals,
            var_categoricals,
        )

        def apply_plate_to_fixed(x, index):
            if x is not None:
                return x[index]
            else:
                return x

        # =====================Cell-specific programme activities ======================= #
        # programme per cell activities - w_{c, f1}
        with obs_plate as ind:
            cell_factors_w_cf = pyro.sample("cell_factors_w_cf", 
                dist.Beta(
                self.w_cf_alpha,
                self.w_cf_beta,
            ).expand([batch_size, self.n_factors]))
            
        # =====================Cell-specific detection efficiency ======================= #
        ### RNA model ###
        # y_c with hierarchical mean prior
        detection_mean_y_e = pyro.sample(
            "detection_mean_y_e",
            dist.Beta(
                self.ones * self.detection_mean_hyp_prior_alpha,
                self.ones * self.detection_mean_hyp_prior_beta,
            )
            .expand([self.n_batch, 1])
            .to_event(2),
        )
        detection_hyp_prior_alpha = pyro.deterministic(
            "detection_hyp_prior_alpha",
            self.detection_hyp_prior_alpha,
        )

        beta = detection_hyp_prior_alpha / (obs2sample @ detection_mean_y_e)
        with obs_plate, pyro.poutine.mask(mask=rna_index):
            detection_y_c = pyro.sample(
                "detection_y_c",
                dist.Gamma(detection_hyp_prior_alpha, beta),
            )  # (self.n_obs, 1)

        # g_{f,g}
        factor_level_g = pyro.sample(
            "factor_level_g",
            dist.Gamma(self.factor_prior_alpha, self.factor_prior_beta)
            .expand([1, self.n_genes])
            .to_event(2)
        )
        g_fg = pyro.sample(
            "g_fg",
            dist.Gamma(
                self.factor_states_per_gene / self.n_factors_torch,
                self.ones / factor_level_g,
            )
            .expand([self.n_factors, self.n_genes])
            .to_event(2)
        )

        # =====================Gene-specific additive component ======================= #
        # s_{e,g} accounting for background, free-floating RNA
        s_g_gene_add_alpha_hyp = pyro.sample(
            "s_g_gene_add_alpha_hyp",
            dist.Gamma(
                self.gene_add_alpha_hyp_prior_alpha,
                self.gene_add_alpha_hyp_prior_beta,
            )
            .expand([1, 1])
            .to_event(2),
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
            dist.Exponential(s_g_gene_add_alpha_hyp)
            .expand([self.n_batch, 1])
            .to_event(2),
        )  # (self.n_batch)
        s_g_gene_add_alpha_e = self.ones / s_g_gene_add_alpha_e_inv.pow(2)

        s_g_gene_add = pyro.sample(
            "s_g_gene_add",
            dist.Gamma(
                s_g_gene_add_alpha_e, s_g_gene_add_alpha_e / s_g_gene_add_mean
            )
            .expand([self.n_batch, self.n_genes])
            .to_event(2),
        )  # (self.n_batch, n_genes)

        # =====================Gene-specific stochastic/unexplained variance ======================= #
        stochastic_v_ag_hyp = pyro.sample(
            "stochastic_v_ag_hyp",
            dist.Gamma(
                self.stochastic_v_ag_hyp_prior_alpha,
                self.stochastic_v_ag_hyp_prior_beta,
            )
            .expand([self.n_var_categoricals, 1])
            .to_event(2),
        )
        stochastic_v_ag_inv = pyro.sample(
            "stochastic_v_ag_inv",
            dist.Exponential(stochastic_v_ag_hyp)
            .expand([self.n_var_categoricals, self.n_genes]).to_event(2),
        )  # (self.n_var_categoricals or 1, self.n_genes)
        stochastic_v_ag = obs2var_categoricals @ (
            self.ones / stochastic_v_ag_inv.pow(2)
        )

        # =====================Expected expression ======================= #
        # per cell biological expression
        mu_biol = cell_factors_w_cf @ g_fg
        # biological expression mediated by bursting
        mu = pyro.deterministic('mu',
            (mu_biol + obs2sample @ s_g_gene_add)  # contaminating RNA
            * detection_y_c
        )  # cell-specific normalisation only for now

        # =====================DATA likelihood ======================= #              
        # Likelihood
        with obs_plate, pyro.poutine.mask(mask=rna_index):
            pyro.sample(
                "data_rna",
                dist.GammaPoisson(
                    concentration=stochastic_v_ag, rate=stochastic_v_ag / mu
                ),
                obs=x_data[:, self.gene_ind_tt],)
