from typing import Optional
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule
from scvi import REGISTRY_KEYS
import pandas as pd
from scvi.nn import one_hot
from cell2fate.utils import G_a, G_b, mu_mRNA_continousAlpha_globalTime_twoStates
from pyro.infer import config_enumerate
from pyro.ops.indexing import Vindex
from torch.distributions import constraints

class Cell2fate_DynamicalModel_module_TimeMixture(PyroModule):
    r"""
    - Models spliced and unspliced counts for each gene as a dynamical process in which transcriptional modules switch on
    at one point in time and increase the transcription rate by different values across genes and then optionally switches off
    to a transcription rate of 0. Splicing and degredation rates are constant for each gene. 
    - The underlying equations are similar to
    "Bergen et al. (2020), Generalizing RNA velocity to transient cell states through dynamical modeling"
    - In addition, the model includes negative binomial noise, batch effects and technical variables, similar to:
    "Kleshchevnikov et al. (2022), Cell2location maps fine-grained cell types in spatial transcriptomics".
    Although in the final version of this model technical variables will be modelled seperately for spliced and unspliced counts.
    """

    def __init__(
        self,
        n_obs,
        n_vars,
        n_batch,
        n_leiden1,
        n_leiden2,
        n_leiden3,
        n_leiden4,
        leiden1_cat,
        leiden2_cat,
        leiden3_cat,
        leiden4_cat,
        leiden1_to_leiden2,
        correlation_leiden1,
        cells_to_module,
        n_extra_categoricals=None,
        stochastic_v_ag_hyp_prior={"alpha": 6.0, "beta": 3.0},
        factor_prior={"rate": 1.0, "alpha": 1.0, "states_per_gene": 3.0},
        t_switch_alpha_prior = {"mean": 1000., "alpha": 1000.},
        splicing_rate_hyp_prior={"mean": 1.0, "alpha": 5.0,
                                "mean_hyp_alpha": 10., "alpha_hyp_alpha": 20.},
        degredation_rate_hyp_prior={"mean": 1.0, "alpha": 5.0,
                                "mean_hyp_alpha": 10., "alpha_hyp_alpha": 20.},
        activation_rate_hyp_prior={"mean_hyp_prior_mean": 1., "mean_hyp_prior_sd": 0.33,
                                    "sd_hyp_prior_mean": 0.33, "sd_hyp_prior_sd": 0.1},
        s_overdispersion_factor_hyp_prior={'alpha_mean': 100., 'beta_mean': 1.,
                                           'alpha_sd': 1., 'beta_sd': 0.1},
        detection_hyp_prior={"alpha": 10.0, "mean_alpha": 1.0, "mean_beta": 1.0},
        detection_i_prior={"mean": 1, "alpha": 100},
        detection_gi_prior={"mean": 1, "alpha": 200},
        gene_add_alpha_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_mean_hyp_prior={"alpha": 1.0, "beta": 100.0},
        Tmax_prior={"mean": 500., "sd": 200.},
        switch_time_sd = 0.1,
        init_vals: Optional[dict] = None
    ):
        
        """

        Parameters
        ----------
        n_obs
        n_vars
        n_batch
        n_extra_categoricals
        gene_add_alpha_hyp_prior
        gene_add_mean_hyp_prior
        detection_hyp_prior
        """

        ############# Initialise parameters ################
        super().__init__()
        self.n_obs = n_obs
        self.n_vars = n_vars
        self.n_batch = n_batch
        self.n_extra_categoricals = n_extra_categoricals
        self.factor_prior = factor_prior
        self.n_leiden1 = n_leiden1
        self.n_leiden2 = n_leiden2
        self.n_leiden3 = n_leiden3 
        self.n_leiden4 = n_leiden4
        self.stochastic_v_ag_hyp_prior = stochastic_v_ag_hyp_prior
        self.gene_add_alpha_hyp_prior = gene_add_alpha_hyp_prior
        self.gene_add_mean_hyp_prior = gene_add_mean_hyp_prior
        self.detection_hyp_prior = detection_hyp_prior
        self.splicing_rate_hyp_prior = splicing_rate_hyp_prior
        self.degredation_rate_hyp_prior = degredation_rate_hyp_prior
        self.s_overdispersion_factor_hyp_prior = s_overdispersion_factor_hyp_prior
        self.t_switch_alpha_prior = t_switch_alpha_prior
        self.detection_gi_prior = detection_gi_prior
        self.detection_i_prior = detection_i_prior

        if (init_vals is not None) & (type(init_vals) is dict):
            self.np_init_vals = init_vals
            for k in init_vals.keys():
                if k == 'I_cm':
                    self.register_buffer(f"init_val_{k}", torch.tensor(init_vals[k], dtype = torch.long))
                else:
                    self.register_buffer(f"init_val_{k}", torch.tensor(init_vals[k])) 
        
        self.register_buffer(
            "I_mc",
            cells_to_module,
        )         
        
        self.register_buffer(
            "correlation_leiden1",
            correlation_leiden1,
        ) 
        
        self.register_buffer(
            "leiden1_cat",
            torch.tensor(leiden1_cat),
        ) 
        
        self.register_buffer(
            "leiden2_cat",
            torch.tensor(leiden2_cat),
        ) 
            
        self.register_buffer(
            "leiden3_cat",
            torch.tensor(leiden3_cat),
        )
        
        self.register_buffer(
            "leiden4_cat",
            torch.tensor(leiden3_cat),
        )
        
        self.register_buffer(
            "n_leiden2_torch",
            torch.tensor(n_leiden2, dtype = torch.float32),
        )
        
        self.register_buffer(
            "n_leiden1_torch",
            torch.tensor(n_leiden1, dtype = torch.float32),
        )
        
        self.register_buffer(
            "s_overdispersion_factor_alpha_mean",
            torch.tensor(self.s_overdispersion_factor_hyp_prior["alpha_mean"]),
        )
        
        self.register_buffer(
            "s_overdispersion_factor_beta_mean",
            torch.tensor(self.s_overdispersion_factor_hyp_prior["beta_mean"]),
        )
        self.register_buffer(
            "s_overdispersion_factor_alpha_sd",
            torch.tensor(self.s_overdispersion_factor_hyp_prior["alpha_sd"]),
        )
        self.register_buffer(
            "s_overdispersion_factor_beta_sd",
            torch.tensor(self.s_overdispersion_factor_hyp_prior["beta_sd"]),
        )
        
        self.register_buffer(
            "detection_gi_prior_alpha",
            torch.tensor(self.detection_gi_prior["alpha"]),
        )
        self.register_buffer(
            "detection_gi_prior_beta",
            torch.tensor(self.detection_gi_prior["alpha"] / self.detection_gi_prior["mean"]),
        )
        
        self.register_buffer(
            "detection_i_prior_alpha",
            torch.tensor(self.detection_i_prior["alpha"]),
        )
        self.register_buffer(
            "detection_i_prior_beta",
            torch.tensor(self.detection_i_prior["alpha"] / self.detection_i_prior["mean"]),
        )
        
        self.register_buffer(
            "Tmax_mean",
            torch.tensor(Tmax_prior["mean"]),
        )
             
        self.register_buffer(
            "Tmax_sd",
            torch.tensor(Tmax_prior["sd"]),
        )
        
        self.register_buffer(
            "switch_time_sd",
            torch.tensor(switch_time_sd),
        )
        
        self.register_buffer(
            "t_mi_alpha_alpha",
            torch.tensor(t_switch_alpha_prior['alpha']),
        )
        
        self.register_buffer(
            "t_mi_alpha_mu",
            torch.tensor(t_switch_alpha_prior['alpha']),
        )
        
        self.t_switch_alpha_prior

        self.register_buffer(
            "detection_mean_hyp_prior_alpha",
            torch.tensor(self.detection_hyp_prior["mean_alpha"]),
        )
        self.register_buffer(
            "detection_mean_hyp_prior_beta",
            torch.tensor(self.detection_hyp_prior["mean_beta"]),
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
            "detection_hyp_prior_alpha",
            torch.tensor(self.detection_hyp_prior["alpha"]),
        )
        
        self.register_buffer("ones_n_batch_1", torch.ones((self.n_batch, 1)))
        
        self.register_buffer("ones", torch.ones((1, 1)))
        self.register_buffer("ones2", torch.ones((self.n_obs, self.n_vars)))
        self.register_buffer("eps", torch.tensor(1e-8))
        self.register_buffer("alpha_OFFg", torch.tensor(10**(-5)))
        self.register_buffer("one", torch.tensor(1.))
        self.register_buffer("zero", torch.tensor(0.))
        self.register_buffer("zero_point_one", torch.tensor(0.1))
        self.register_buffer("one_point_one", torch.tensor(1.1))
        self.register_buffer("one_point_two", torch.tensor(1.2))
        self.register_buffer("zeros", torch.zeros(self.n_obs, self.n_vars))
        self.register_buffer("ones_g", torch.ones((1,self.n_vars,1)))
        
        self.register_buffer("zeros_leiden1", torch.zeros(self.n_leiden1))
        
        # Register parameters for activation rate hyperprior:
        self.register_buffer(
            "activation_rate_mean_hyp_prior_mean",
            torch.tensor(activation_rate_hyp_prior["mean_hyp_prior_mean"]),
        )        
        self.register_buffer(
            "activation_rate_mean_hyp_prior_sd",
            torch.tensor(activation_rate_hyp_prior["mean_hyp_prior_sd"]),
        )
        self.register_buffer(
            "activation_rate_sd_hyp_prior_mean",
            torch.tensor(activation_rate_hyp_prior["sd_hyp_prior_mean"]),
        )
        self.register_buffer(
            "activation_rate_sd_hyp_prior_sd",
            torch.tensor(activation_rate_hyp_prior["sd_hyp_prior_sd"]),
        )     
        
        # Register parameters for splicing rate hyperprior:
        self.register_buffer(
            "splicing_rate_alpha_hyp_prior_mean",
            torch.tensor(self.splicing_rate_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "splicing_rate_mean_hyp_prior_mean",
            torch.tensor(self.splicing_rate_hyp_prior["mean"]),
        )
        self.register_buffer(
            "splicing_rate_alpha_hyp_prior_alpha",
            torch.tensor(self.splicing_rate_hyp_prior["alpha_hyp_alpha"]),
        )
        self.register_buffer(
            "splicing_rate_mean_hyp_prior_alpha",
            torch.tensor(self.splicing_rate_hyp_prior["mean_hyp_alpha"]),
        )
        
        # Register parameters for degredation rate hyperprior:
        self.register_buffer(
            "degredation_rate_alpha_hyp_prior_mean",
            torch.tensor(self.degredation_rate_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "degredation_rate_mean_hyp_prior_mean",
            torch.tensor(self.degredation_rate_hyp_prior["mean"]),
        )
        self.register_buffer(
            "degredation_rate_alpha_hyp_prior_alpha",
            torch.tensor(self.degredation_rate_hyp_prior["alpha_hyp_alpha"]),
        )
        self.register_buffer(
            "degredation_rate_mean_hyp_prior_alpha",
            torch.tensor(self.degredation_rate_hyp_prior["mean_hyp_alpha"]),
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
        
        self.register_buffer(
            "factor_states_per_gene",
            torch.tensor(self.factor_prior["states_per_gene"]),
        )
        
        self.register_buffer(
            "t_c_init",
            self.one*torch.ones((self.n_obs, 1, 1))/2.,
        ) 
        
        self.register_buffer(
            "probs_I_cm",
            torch.tensor(1.-10**(-10)),
        )
        
        self.register_buffer(
            "mixture_weights_prior",
            torch.ones(2),
        )
        
        self.register_buffer(
            "leiden1_to_leiden2",
            torch.tensor(leiden1_to_leiden2)
        )     
            
    ############# Define the model ################
    @staticmethod
    def _get_fn_args_from_batch_no_cat(tensor_dict):
        u_data = tensor_dict['unspliced']
        s_data = tensor_dict['spliced']
        ind_x = tensor_dict["ind_x"].long().squeeze()
        batch_index = tensor_dict[REGISTRY_KEYS.BATCH_KEY]
        return (u_data, s_data, ind_x, batch_index), {}

    @staticmethod
    def _get_fn_args_from_batch_cat(tensor_dict):
        u_data = tensor_dict['unspliced']
        s_data = tensor_dict['spliced']
        ind_x = tensor_dict["ind_x"].long().squeeze()
        batch_index = tensor_dict[REGISTRY_KEYS.BATCH_KEY]
        extra_categoricals = tensor_dict[REGISTRY_KEYS.CAT_COVS_KEY]
        return (u_data, s_data, ind_x, batch_index), {}

    @property
    def _get_fn_args_from_batch(self):
        if self.n_extra_categoricals is not None:
            return self._get_fn_args_from_batch_cat
        else:
            return self._get_fn_args_from_batch_no_cat

    def create_plates(self, u_data, s_data, idx, batch_index):
        return pyro.plate("obs_plate", size=self.n_obs, dim=-3, subsample=idx)

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
    
    def forward(self, u_data, s_data, idx, batch_index):
        
        batch_size = len(idx)
        obs2sample = one_hot(batch_index, self.n_batch)        
        obs_plate = self.create_plates(u_data, s_data, idx, batch_index)
        
        # ===================== Kinetic Rates ======================= #
        # Splicing rate:
        splicing_alpha = pyro.sample('splicing_alpha',
                              dist.Gamma(self.splicing_rate_alpha_hyp_prior_alpha,
                              self.splicing_rate_alpha_hyp_prior_alpha/self.splicing_rate_alpha_hyp_prior_mean))
        splicing_mean = pyro.sample('splicing_mean',
                              dist.Gamma(self.splicing_rate_mean_hyp_prior_alpha,
                              self.splicing_rate_mean_hyp_prior_alpha/self.splicing_rate_mean_hyp_prior_mean))
        beta_g = pyro.sample('beta_g', dist.Gamma(splicing_alpha, splicing_alpha/splicing_mean).expand([1,self.n_vars]).to_event(2))
        # Degredation rate:
        degredation_alpha = pyro.sample('degredation_alpha',
                              dist.Gamma(self.degredation_rate_alpha_hyp_prior_alpha,
                              self.degredation_rate_alpha_hyp_prior_alpha/self.degredation_rate_alpha_hyp_prior_mean))
        degredation_alpha = degredation_alpha + 0.001
        degredation_mean = pyro.sample('degredation_mean',
                              dist.Gamma(self.degredation_rate_mean_hyp_prior_alpha,
                              self.degredation_rate_mean_hyp_prior_alpha/self.degredation_rate_mean_hyp_prior_mean))
        gamma_g = pyro.sample('gamma_g', dist.Gamma(degredation_alpha, degredation_alpha/degredation_mean).expand([1,self.n_vars]).to_event(2))
        # Transcription rate contribution of each module:
        factor_level_g = pyro.sample(
            "factor_level_g",
            dist.Gamma(self.factor_prior_alpha, self.factor_prior_beta)
            .expand([1, self.n_vars])
            .to_event(2)
        )
        g_fg = pyro.sample( # (g_fg corresponds to module's spliced counts in steady state)
            "g_fg",
            dist.Gamma(
                self.factor_states_per_gene / self.n_leiden1,
                self.ones / factor_level_g,
            )
            .expand([self.n_leiden1, self.n_vars])
            .to_event(2)
        )
        A_mgON = pyro.deterministic('A_mgON', g_fg*gamma_g) # (transform from spliced counts to transcription rate)
        A_mgOFF = self.alpha_OFFg        
        # Activation and Deactivation rate:
        lam_mu = pyro.sample('lam_mu', dist.Gamma(G_a(self.activation_rate_mean_hyp_prior_mean, self.activation_rate_mean_hyp_prior_sd),
                                            G_b(self.activation_rate_mean_hyp_prior_mean, self.activation_rate_mean_hyp_prior_sd)))
        lam_sd = pyro.sample('lam_sd', dist.Gamma(G_a(self.activation_rate_sd_hyp_prior_mean, self.activation_rate_sd_hyp_prior_sd),
                                            G_b(self.activation_rate_sd_hyp_prior_mean, self.activation_rate_sd_hyp_prior_sd)))
        lam_m_mu = pyro.sample('lam_m_mu', dist.Gamma(G_a(lam_mu, lam_sd),
                                            G_b(lam_mu, lam_sd)).expand([self.n_leiden1, 1, 1]).to_event(3))
        lam_mi = pyro.sample('lam_mi', dist.Gamma(G_a(lam_m_mu, lam_m_mu*0.05),
                                       G_b(lam_m_mu, lam_m_mu*0.05)).expand([self.n_leiden1, 1, 2]).to_event(3))
        
        # =====================Time======================= #
        # Global time for each cell:      
        Tmax = pyro.deterministic('Tmax', self.n_leiden1_torch*10.0)
        sd_1 = pyro.sample('sd_1', dist.Exponential(1/(Tmax/6)))
        covariance_1 = sd_1*self.correlation_leiden1
#         T_level1 = pyro.sample('T_level1', dist.MultivariateNormal(self.zeros_leiden1, covariance_1))
        T_level1 = pyro.sample('T_level1', dist.Normal(Tmax, sd_1).expand([self.n_leiden1]).to_event(1))
        sd_2 = pyro.sample('sd_2', dist.Exponential(1/(sd_1/self.n_leiden2)))
        T_level2 = pyro.sample('T_level2', dist.Normal(self.zero, sd_2).expand([self.n_leiden2]).to_event(1))
        sd_3 = pyro.sample('sd_3', dist.Exponential(1/(sd_1/self.n_leiden3)))
        T_level3 = pyro.sample('T_level3', dist.Normal(self.zero, sd_3).expand([self.n_leiden3]).to_event(1))
        sd_4 = pyro.sample('sd_4', dist.Exponential(1/(sd_1/self.n_leiden4)))
        T_level4 = pyro.sample('T_level4', dist.Normal(self.zero, sd_4).expand([self.n_leiden4]).to_event(1))
        sd_5 = pyro.sample('sd_5', dist.Exponential(1/(sd_1/self.n_obs)))
        with obs_plate as indx:
            T_levelF = pyro.sample('T_levelF', dist.Normal(self.zero, sd_5))
            one_hot_1 = one_hot(self.leiden1_cat[indx].unsqueeze(-1), self.n_leiden1)
            one_hot_2 = one_hot(self.leiden2_cat[indx].unsqueeze(-1), self.n_leiden2)
            one_hot_3 = one_hot(self.leiden3_cat[indx].unsqueeze(-1), self.n_leiden3)
            one_hot_4 = one_hot(self.leiden4_cat[indx].unsqueeze(-1), self.n_leiden4)
            T_c = pyro.deterministic('T_c', (torch.einsum('ci,i->c', one_hot_1, T_level1).unsqueeze(-1).unsqueeze(-1)
                                             + torch.einsum('ci,i->c', one_hot_2, T_level2).unsqueeze(-1).unsqueeze(-1)
                                             + torch.einsum('ci,i->c', one_hot_3, T_level3).unsqueeze(-1).unsqueeze(-1) 
                                             + torch.einsum('ci,i->c', one_hot_4, T_level4).unsqueeze(-1).unsqueeze(-1) 
                                             + T_levelF))

        # Global switch on time for each module:        
        T_mON = pyro.deterministic('T_mON', T_level1.unsqueeze(0).unsqueeze(0)-sd_2)
        
        # Global switch off time for each module:
        t_mOFF = pyro.sample('t_mOFF', dist.Exponential(1/(4*sd_2)).expand([1, 1, self.n_leiden1]).to_event(2))
        T_mOFF = pyro.deterministic('T_mOFF', T_mON + t_mOFF)
        
        # =========== Mean expression according to RNAvelocity model ======================= #
        mu_total = torch.stack([self.zeros[idx,...], self.zeros[idx,...]], axis = -1)
#         for m in range(self.n_leiden2):
#             subset = self.I_mc[m,idx]
#             mu_total[subset,...] += mu_mRNA_continousAlpha_globalTime_twoStates(
#                 A_mgON[m,:], A_mgOFF, beta_g, gamma_g, lam_mi[m,...], T_c[subset,:,0],
#                 T_mON[:,:,m], T_mOFF[:,:,m], self.zeros[idx,...][subset,...])
        for m in range(self.n_leiden1):
                mu_total += mu_mRNA_continousAlpha_globalTime_twoStates(
                    A_mgON[m,:], A_mgOFF, beta_g, gamma_g, lam_mi[m,...], T_c[:,:,0],
                    T_mON[:,:,m], T_mOFF[:,:,m], self.zeros[idx,...])

        with obs_plate:
            mu_expression = pyro.deterministic('mu_expression', mu_total)
        
        # =============Detection efficiency of spliced and unspliced counts =============== #
        # Cell specific relative detection efficiency with hierarchical prior across batches:
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
        with obs_plate:
            detection_y_c = pyro.sample(
                "detection_y_c",
                dist.Gamma(detection_hyp_prior_alpha.unsqueeze(dim=-1), beta.unsqueeze(dim=-1)),
            )  # (self.n_obs, 1)        
        
        # Global relative detection efficiency between spliced and unspliced counts
        detection_y_i = pyro.sample(
            "detection_y_i",
            dist.Gamma(
                self.ones * self.detection_i_prior_alpha,
                self.ones * self.detection_i_prior_alpha,
            )
            .expand([1, 1, 2]).to_event(3)
        )
        
        # Gene specific relative detection efficiency between spliced and unspliced counts
        detection_y_gi = pyro.sample(
            "detection_y_gi",
            dist.Gamma(
                self.ones * self.detection_gi_prior_alpha,
                self.ones * self.detection_gi_prior_alpha,
            )
            .expand([1, self.n_vars, 2])
            .to_event(3),
        )
        
        # =======Gene-specific additive component (Ambient RNA/ "Soup") for spliced and unspliced counts ====== #
        # Independently sampled for spliced and unspliced counts:
        s_g_gene_add_alpha_hyp = pyro.sample(
            "s_g_gene_add_alpha_hyp",
            dist.Gamma(self.gene_add_alpha_hyp_prior_alpha, self.gene_add_alpha_hyp_prior_beta).expand([2]).to_event(1),
        )
        s_g_gene_add_mean = pyro.sample(
            "s_g_gene_add_mean",
            dist.Gamma(
                self.gene_add_mean_hyp_prior_alpha,
                self.gene_add_mean_hyp_prior_beta,
            )
            .expand([self.n_batch, 1, 2])
            .to_event(3),
        ) 
        s_g_gene_add_alpha_e_inv = pyro.sample(
            "s_g_gene_add_alpha_e_inv",
            dist.Exponential(s_g_gene_add_alpha_hyp).expand([self.n_batch, 1, 2]).to_event(3),
        )
        s_g_gene_add_alpha_e = self.ones / s_g_gene_add_alpha_e_inv.pow(2)
        s_g_gene_add = pyro.sample(
            "s_g_gene_add",
            dist.Gamma(s_g_gene_add_alpha_e, s_g_gene_add_alpha_e / s_g_gene_add_mean)
            .expand([self.n_batch, self.n_vars, 2])
            .to_event(3),
        )

        # =========Gene-specific overdispersion of spliced and unspliced counts ============== #
        # Overdispersion of unspliced counts:
        stochastic_v_ag_hyp = pyro.sample(
        "stochastic_v_ag_hyp",
        dist.Gamma(
            self.stochastic_v_ag_hyp_prior_alpha,
            self.stochastic_v_ag_hyp_prior_beta,
        ).expand([1, 2]).to_event(2))
        stochastic_v_ag_hyp = stochastic_v_ag_hyp + 0.001
        stochastic_v_ag_inv = pyro.sample(
            "stochastic_v_ag_inv",
            dist.Exponential(stochastic_v_ag_hyp)
            .expand([1, self.n_vars, 2]).to_event(3),
        ) 
        stochastic_v_ag = (self.ones / stochastic_v_ag_inv.pow(2))        

        # =====================Expected expression ======================= #
        # biological expression
        with obs_plate:
            mu = pyro.deterministic('mu', (mu_expression + torch.einsum('cbi,bgi->cgi', obs2sample.unsqueeze(dim=-1), s_g_gene_add)) * \
        detection_y_c * detection_y_i * detection_y_gi)
        
        # =====================DATA likelihood ======================= #
        # Likelihood (sampling distribution) of data_target & add overdispersion via NegativeBinomial
        with obs_plate:
            pyro.sample("data_target", dist.GammaPoisson(concentration= stochastic_v_ag,
                       rate= stochastic_v_ag / mu), obs=torch.stack([u_data, s_data], axis = 2))