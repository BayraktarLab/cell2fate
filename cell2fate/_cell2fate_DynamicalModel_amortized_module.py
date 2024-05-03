from typing import Optional
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
import torch
from scvi import REGISTRY_KEYS
import pandas as pd
from scvi.nn import one_hot
from cell2fate.utils import G_a, G_b, mu_mRNA_continousAlpha_globalTime_twoStates
from pyro.infer import config_enumerate
from pyro.ops.indexing import Vindex
from torch.distributions import constraints
from ._cell2fate_DynamicalModel_module import \
Cell2fate_DynamicalModel_module

class Cell2fate_DynamicalModel_amortized_module(Cell2fate_DynamicalModel_module):
    r"""
    Identical to DynamicalModel, but using amortization for cell-specific variables.
    """
    
    def create_plates(self, x_data, idx, batch_index):
        return pyro.plate("obs_plate", size=self.n_obs, dim=-3, subsample=idx)
    
    @staticmethod
    def _get_fn_args_from_batch_no_cat(tensor_dict):
        u_data = tensor_dict['unspliced']
        s_data = tensor_dict['spliced']
        x_data = torch.stack([u_data, s_data], axis = 2)
        ind_x = tensor_dict["ind_x"].long().squeeze()
        batch_index = tensor_dict[REGISTRY_KEYS.BATCH_KEY]
        return (x_data, ind_x, batch_index), {}

    @staticmethod
    def _get_fn_args_from_batch_cat(tensor_dict):
        u_data = tensor_dict['unspliced']
        s_data = tensor_dict['spliced']
        x_data = torch.stack([u_data, s_data], axis = 2)
        ind_x = tensor_dict["ind_x"].long().squeeze()
        batch_index = tensor_dict[REGISTRY_KEYS.BATCH_KEY]
        extra_categoricals = tensor_dict[REGISTRY_KEYS.CAT_COVS_KEY]
        return (x_data, ind_x, batch_index), {}
    
    def list_obs_plate_vars(self):
        """Create a dictionary with the name of observation/minibatch plate,
        indexes of model args to provide to encoder,
        variable names that belong to the observation plate
        and the number of dimensions in non-plate axis of each variable"""
        
        def gene_transform(x):
                return torch.log1p(x.flatten(start_dim = -2))
        def no_transform(x):
                return(x)
        
        return {
            "n_in": 2*self.n_vars, # size of encoder input layer
            "name": "obs_plate", # name of plate that is being amortized
            "input": [0,2], # position of input arguments in forward method
            "input_transform": [gene_transform, no_transform], # transformation to input
            "sites": {"t_c": (1,1), # dimensions left of plate dimension
                     "detection_y_c": (1,1)}}
    
    def forward(self, x_data, idx, batch_index):
        
        batch_size = len(idx)
        obs2sample = one_hot(batch_index, self.n_batch)        
        obs_plate = self.create_plates(x_data, idx, batch_index)
        
        # ===================== Kinetic Rates ======================= #
        # Splicing rate:
        splicing_alpha = pyro.sample('splicing_alpha',
                              dist.Gamma(self.splicing_rate_alpha_hyp_prior_alpha,
                              self.splicing_rate_alpha_hyp_prior_alpha/self.splicing_rate_alpha_hyp_prior_mean
                                        ).expand([1,1]).to_event(2))
        splicing_mean = pyro.sample('splicing_mean',
                              dist.Gamma(self.splicing_rate_mean_hyp_prior_alpha,
                              self.splicing_rate_mean_hyp_prior_alpha/self.splicing_rate_mean_hyp_prior_mean
                                   ).expand([1,1]).to_event(2))
        beta_g = pyro.sample('beta_g', dist.Gamma(splicing_alpha, splicing_alpha/splicing_mean).expand([1,self.n_vars]).to_event(2))
        # Degredation rate:
        degredation_alpha = pyro.sample('degredation_alpha',
                              dist.Gamma(self.degredation_rate_alpha_hyp_prior_alpha,
                              self.degredation_rate_alpha_hyp_prior_alpha/self.degredation_rate_alpha_hyp_prior_mean
                                        ).expand([1,1]).to_event(2))
        degredation_alpha = degredation_alpha + 0.001
        degredation_mean = pyro.sample('degredation_mean',
                              dist.Gamma(self.degredation_rate_mean_hyp_prior_alpha,
                              self.degredation_rate_mean_hyp_prior_alpha/self.degredation_rate_mean_hyp_prior_mean
                                        ).expand([1,1]).to_event(2))
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
                self.factor_states_per_gene / self.n_factors_torch,
                self.ones / factor_level_g,
            )
            .expand([self.n_modules, self.n_vars])
            .to_event(2)
        )
        A_mgON = pyro.deterministic('A_mgON', g_fg*gamma_g) # (transform from spliced counts to transcription rate)
        A_mgOFF = self.alpha_OFFg        
        # Activation and Deactivation rate:
        lam_mu = pyro.sample('lam_mu', dist.Gamma(G_a(self.activation_rate_mean_hyp_prior_mean, self.activation_rate_mean_hyp_prior_sd),
                                            G_b(self.activation_rate_mean_hyp_prior_mean, self.activation_rate_mean_hyp_prior_sd)
                                                 ).expand([1,1]).to_event(2))
        lam_sd = pyro.sample('lam_sd', dist.Gamma(G_a(self.activation_rate_sd_hyp_prior_mean, self.activation_rate_sd_hyp_prior_sd),
                                            G_b(self.activation_rate_sd_hyp_prior_mean, self.activation_rate_sd_hyp_prior_sd)
                                                 ).expand([1,1]).to_event(2))
        lam_m_mu = pyro.sample('lam_m_mu', dist.Gamma(G_a(lam_mu, lam_sd),
                                            G_b(lam_mu, lam_sd)).expand([self.n_modules, 1, 1]).to_event(3))
        lam_mi = pyro.sample('lam_mi', dist.Gamma(G_a(lam_m_mu, lam_m_mu*0.05),
                                            G_b(lam_m_mu, lam_m_mu*0.05)).expand([self.n_modules, 1, 2]).to_event(3))
        
        # =====================Time======================= #
        # Global time for each cell:
        T_max = pyro.sample('Tmax', dist.Gamma(G_a(self.Tmax_mean, self.Tmax_sd), G_b(self.Tmax_mean, self.Tmax_sd)
                                              ).expand([1,1]).to_event(2))
        t_c_loc = pyro.sample('t_c_loc', dist.Gamma(self.one, self.one/0.5).expand([1,1]).to_event(2))
        t_c_scale = pyro.sample('t_c_scale', dist.Gamma(self.one, self.one/0.25).expand([1,1]).to_event(2))
        with obs_plate:
            t_c = pyro.sample('t_c', dist.Normal(t_c_loc, t_c_scale).expand([batch_size, 1, 1]))
            T_c = pyro.deterministic('T_c', t_c*T_max)
        # Global switch on time for each gene:
#         t_mON = pyro.sample('t_mON', dist.Uniform(self.zero, self.one).expand([1, 1, self.n_modules]).to_event(2))
        t_delta = pyro.sample('t_delta', dist.Gamma(self.one*20, self.one * 20 *self.n_modules_torch).
                              expand([self.n_modules]).to_event(1))
        t_mON = torch.cumsum(torch.concat([self.zero.unsqueeze(0), t_delta[:-1]]), dim = 0).unsqueeze(0).unsqueeze(0)
        T_mON = pyro.deterministic('T_mON', T_max*t_mON)
        # Global switch off time for each gene:
        t_mOFF = pyro.sample('t_mOFF', dist.Exponential(self.n_modules_torch).expand([1, 1, self.n_modules]).to_event(2))
        T_mOFF = pyro.deterministic('T_mOFF', T_mON + T_max*t_mOFF)
        
        # =========== Mean expression according to RNAvelocity model ======================= #
        mu_total = torch.stack([self.zeros[idx,...], self.zeros[idx,...]], axis = -1)
        for m in range(self.n_modules):
            mu_total += mu_mRNA_continousAlpha_globalTime_twoStates(
                A_mgON[m,:], A_mgOFF, beta_g, gamma_g, lam_mi[m,...], T_c[:,:,0], T_mON[:,:,m], T_mOFF[:,:,m], self.zeros[idx,...])
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
                       rate= stochastic_v_ag / mu), obs=x_data)
    