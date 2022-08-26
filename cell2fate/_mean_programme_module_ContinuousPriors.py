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
from pyro.distributions import RelaxedBernoulliStraightThrough
RelaxedBernoulliStraightThrough.mean = property(lambda self: self.probs)
from cell2fate.utils import G_a, G_b

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

    # training mode without observed data (just using priors)
    training_wo_data = False
    training_wo_observed = False
    training_wo_initial = False

    def __init__(
        self,
        n_obs,
        n_vars,
        n_batch,
        n_extra_categoricals,
        n_var_categoricals,
        gene_bool: np.array,
        n_factors = 30,
        n_factors_per_cell = 3,
        n_factors_per_gene = 1,
        factor_level_prior_alpha = 1.0,
        g_fg_prior_alpha = 1.,
        stochastic_v_ag_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_alpha_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_mean_hyp_prior={
            "alpha": 1.0,
            "beta": 100.0,
        },
        detection_hyp_prior={"alpha_experiments": 10.0, "alpha_cells": 100.0, "expected_probability": 0.15},
        gene_tech_prior={"mean": 1, "alpha": 1000},
        fixed_vals: Optional[dict] = None,
        init_vals: Optional[dict] = None,
        init_alpha=10.0,
        rna_model: bool = True,
        use_exp_positive: bool = False,
        p_f_prior = {"alpha": 1., "beta": 9.},
        p_g_prior = {"alpha": 1., "beta": 9.},
        w_cf_prior = {"alpha": 0.3, "beta": 0.7},
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
        gene_tech_prior
        use_average_as_initial_value
        """

        ############# Initialise parameters ################
        super().__init__()

        self.rna_model = rna_model

        self.n_obs = n_obs
        self.n_vars = n_vars
        self.n_factors = n_factors
        self.n_batch = n_batch
        self.n_extra_categoricals = n_extra_categoricals
        self.n_var_categoricals = n_var_categoricals
        self.use_exp_positive = use_exp_positive

        self.gene_bool = gene_bool.astype(int).flatten()
        self.gene_ind = np.where(gene_bool)[0]
        self.n_genes = len(self.gene_ind)

        self.region_ind = np.where(np.logical_not(gene_bool))[0]
        self.n_regions = self.n_vars - self.n_genes
        self.register_buffer("region_ind_tt", torch.tensor(self.region_ind))

        # RNA model priors
        self.stochastic_v_ag_hyp_prior = stochastic_v_ag_hyp_prior
        self.gene_add_alpha_hyp_prior = gene_add_alpha_hyp_prior
        self.gene_add_mean_hyp_prior = gene_add_mean_hyp_prior
        self.gene_tech_prior = gene_tech_prior
        self.g_fg_prior_alpha = g_fg_prior_alpha
        self.factor_level_prior_alpha = factor_level_prior_alpha
        self.detection_hyp_prior = detection_hyp_prior
        self.p_f_prior = p_f_prior
        self.p_g_prior = p_g_prior
        self.w_cf_prior = w_cf_prior

        # Fixed values (gene loadings or cell loadings)
        if (fixed_vals is not None) & (type(fixed_vals) is dict):
            self.np_fixed_vals = fixed_vals
            for k in fixed_vals.keys():
                self.register_buffer(f"fixed_val_{k}", torch.tensor(fixed_vals[k]))
            if "n_factors" in fixed_vals.keys():
                self.n_factors = fixed_vals["n_factors"]
            if "g_fg" in fixed_vals.keys():
                self.n_genes = fixed_vals["g_fg"].shape[1]

        # Initial values
        if (init_vals is not None) & (type(init_vals) is dict):
            self.np_init_vals = init_vals
            for k in init_vals.keys():
                self.register_buffer(f"init_val_{k}", torch.tensor(init_vals[k]))
            self.init_alpha = init_alpha
            self.register_buffer("init_alpha_tt", torch.tensor(self.init_alpha))

        # Shared priors
        self.register_buffer(
            "alpha_experiments",
            torch.tensor(self.detection_hyp_prior["alpha_experiments"]),
        )
        self.register_buffer(
            "alpha_cells",
            torch.tensor(self.detection_hyp_prior["alpha_cells"]),
        )
        self.register_buffer(
            "expected_probability",
            torch.tensor(self.detection_hyp_prior["expected_probability"]),
        )

        # per gene rate priors
        self.register_buffer(
            "factor_level_alpha",
            torch.tensor(self.factor_level_prior_alpha),
        )
        
        self.register_buffer(
            "n_factors_per_cell",
            torch.tensor(n_factors_per_cell),
        )
        
        self.register_buffer(
            "n_factors_per_gene",
            torch.tensor(n_factors_per_cell),
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
            "p_f_alpha",
            torch.tensor(self.p_f_prior["alpha"]),
        )
        self.register_buffer(
            "p_f_beta",
            torch.tensor(self.p_f_prior["beta"]),
        )
        
        self.register_buffer(
            "p_g_alpha",
            torch.tensor(self.p_g_prior["alpha"]),
        )
        self.register_buffer(
            "p_g_beta",
            torch.tensor(self.p_g_prior["beta"]),
        )
        
        self.register_buffer(
            "w_cf_alpha",
            torch.tensor(self.w_cf_prior["alpha"]),
        )
        self.register_buffer(
            "w_cf_beta",
            torch.tensor(self.w_cf_prior["beta"]),
        )
        
        self.register_buffer(
            "g_fg_alpha",
            torch.tensor(self.g_fg_prior_alpha),
        )
        
        self.register_buffer(
            "total_mRNA_mean",
            torch.tensor(1.6*10**5),
        )
        
        self.register_buffer(
            "total_mRNA_sd",
            torch.tensor(4*10**4),
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
        fraction_retained = tensor_dict["fraction_retained"]
        return (
            x_data,
            ind_x,
            batch_index,
            label_index,
            rna_index,
            extra_categoricals,
            var_categoricals,
            fraction_retained
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
        fraction_retained
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
        fraction_retained
    ):
        
        r = torch.mean(fraction_retained)
        
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
            fraction_retained
        )

        def apply_plate_to_fixed(x, index):
            if x is not None:
                return x[index]
            else:
                return x

        # =====================Cell-specific programme activities ======================= #
        
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
                self.ones * self.alpha_experiments,
                self.ones * self.alpha_experiments * (1 - self.expected_probability) / self.expected_probability,
            )
            .expand([self.n_batch, 1])
            .to_event(2),
        )
        alpha_y = (
            (obs2sample @ detection_mean_y_e)
            * self.ones
            * self.alpha_cells
        )
        beta_y = (
            (obs2sample @ (self.ones - detection_mean_y_e))
            * self.ones
            * self.alpha_cells
        )
        with obs_plate:
            detection_y_c = pyro.sample(
                "detection_y_c",
                dist.Beta(alpha_y, beta_y),
            )  # (self.n_obs, 1)

        # ===================== Gene programmes ======================== #
        # (set prior so that gene expression produces realistic total counts)
#         total_mRNA = pyro.sample('total_mRNA', 
#                                  dist.Gamma(
#                                  G_a(self.total_mRNA_mean, self.total_mRNA_sd),
#                                  G_b(self.total_mRNA_mean, self.total_mRNA_sd)))
                                      
#         self.mean_factor_level =  r * total_mRNA / (self.n_genes * self.n_factors  
#                                           * self.w_cf_alpha / (self.w_cf_alpha + self.w_cf_beta))
        
        self.mean_factor_level = self.one
        self.factor_level_beta = self.factor_level_alpha/self.mean_factor_level
        
        self.factor_level_g = pyro.sample(
            "factor_level_g",
            dist.Gamma(self.factor_level_alpha, self.factor_level_beta)
            .expand([1, self.n_genes])
            .to_event(2)
        )
        
        g_fg = pyro.sample(
            "g_fg",
            dist.Gamma(
                self.g_fg_alpha,
                self.g_fg_alpha/self.factor_level_g,
            )
            .expand([self.n_factors, self.n_genes])
            .to_event(2))
        
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

        # =====================DATA likelihood ======================= #
        # per cell biological expression
        mu_biol = cell_factors_w_cf @ g_fg
        # biological expression mediated by bursting
        mu = (
            (mu_biol + obs2sample @ s_g_gene_add)  # contaminating RNA
            * detection_y_c)  # cell specific normalisation                     
                          
        with obs_plate:
            pyro.sample(
                "data_rna",
                dist.GammaPoisson(
                    concentration=stochastic_v_ag, rate=stochastic_v_ag / mu),
                obs=x_data)
