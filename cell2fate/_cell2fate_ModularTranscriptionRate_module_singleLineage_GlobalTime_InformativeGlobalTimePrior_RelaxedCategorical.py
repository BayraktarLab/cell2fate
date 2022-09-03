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

from pyro.distributions import RelaxedOneHotCategorical
RelaxedOneHotCategorical.mean = property(lambda self: self.probs)

class Cell2fate_ModularTranscriptionRate_module_SingleLineage_GlobalTime(PyroModule):
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
        n_extra_categoricals=None,
        n_modules = 10,
        stochastic_v_ag_hyp_prior={"alpha": 9.0, "beta": 3.0},
        factor_prior={"rate": 1.0, "alpha": 1.0, "states_per_gene": 3.0},
        t_switch_alpha_prior = {"mean": 1000., "alpha": 1000.},
        splicing_rate_hyp_prior={"mean_hyp_prior_mean": 1.0, "mean_hyp_prior_sd": 0.4,
                                 "sd_hyp_prior_mean": 0.1, "sd_hyp_prior_sd": 0.05},
        degredation_rate_hyp_prior={"mean_hyp_prior_mean": 0.25, "mean_hyp_prior_sd": 0.1,
                                    "sd_hyp_prior_mean": 0.1, "sd_hyp_prior_sd": 0.05},
        activation_rate_hyp_prior={"mean_hyp_prior_mean": 2.0, "mean_hyp_prior_sd": 0.33,
                                    "sd_hyp_prior_mean": 0.33, "sd_hyp_prior_sd": 0.1},
        s_overdispersion_factor_hyp_prior={'alpha_mean': 100., 'beta_mean': 1.,
                                           'alpha_sd': 1., 'beta_sd': 0.1},
        detection_hyp_prior={"alpha": 20.0, "mean_alpha": 1.0, "mean_beta": 1.0},
        detection_i_prior={"mean": 1, "alpha": 100},
        detection_gi_prior={"mean": 1, "alpha": 200},
        gene_add_alpha_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_mean_hyp_prior={"alpha": 1.0, "beta": 100.0},
        Tmax_prior={"mean": 300., "sd": 100.},
        switch_time_sd = 0.1,
        alpha_dirichlet = 1.0, 
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
        self.n_modules = n_modules
        self.n_obs = n_obs
        self.n_vars = n_vars
        self.n_batch = n_batch
        self.n_extra_categoricals = n_extra_categoricals
        self.factor_prior = factor_prior
        
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
            "n_modules_torch",
            torch.tensor(n_modules, dtype = torch.float32),
        )
        
        self.register_buffer(
            "alpha_dirichlet",
            torch.ones(4) * alpha_dirichlet,
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
        self.register_buffer("ones_m", torch.ones(n_modules))
        
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
            "splicing_rate_mean_hyp_prior_mean",
            torch.tensor(self.splicing_rate_hyp_prior["mean_hyp_prior_mean"]),
        )        
        self.register_buffer(
            "splicing_rate_mean_hyp_prior_sd",
            torch.tensor(self.splicing_rate_hyp_prior["mean_hyp_prior_sd"]),
        )
        self.register_buffer(
            "splicing_rate_sd_hyp_prior_mean",
            torch.tensor(self.splicing_rate_hyp_prior["sd_hyp_prior_mean"]),
        )
        self.register_buffer(
            "splicing_rate_sd_hyp_prior_sd",
            torch.tensor(self.splicing_rate_hyp_prior["sd_hyp_prior_sd"]),
        )
        
        # Register parameters for degredation rate hyperprior:
        self.register_buffer(
            "degredation_rate_mean_hyp_prior_mean",
            torch.tensor(self.degredation_rate_hyp_prior["mean_hyp_prior_mean"]),
        )        
        self.register_buffer(
            "degredation_rate_mean_hyp_prior_sd",
            torch.tensor(self.degredation_rate_hyp_prior["mean_hyp_prior_sd"]),
        )
        self.register_buffer(
            "degredation_rate_sd_hyp_prior_mean",
            torch.tensor(self.degredation_rate_hyp_prior["sd_hyp_prior_mean"]),
        )
        self.register_buffer(
            "degredation_rate_sd_hyp_prior_sd",
            torch.tensor(self.degredation_rate_hyp_prior["sd_hyp_prior_sd"]),
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
        
        self.register_buffer("n_factors_torch", torch.tensor(self.n_modules))
        
        self.register_buffer(
            "factor_states_per_gene",
            torch.tensor(self.factor_prior["states_per_gene"]),
        )
        
        self.register_buffer(
            "t_cmi_init",
            self.one*torch.ones((self.n_obs, self.n_modules, 2))/(2.*self.one),
        )
            
        self.register_buffer(
            "t_mOFF_init",
            self.one*torch.ones((1, self.n_modules, 1))/(2.*self.one),
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
    
    @config_enumerate
    def forward(self, u_data, s_data, idx, batch_index):
        
        batch_size = len(idx)
        obs2sample = one_hot(batch_index, self.n_batch)        
        obs_plate = self.create_plates(u_data, s_data, idx, batch_index)
        
        # ===================== Kinetic Rates ======================= #
        # Splicing rate:
        beta_mu = pyro.sample('beta_mu',
                   dist.Gamma(G_a(self.splicing_rate_mean_hyp_prior_mean, self.splicing_rate_mean_hyp_prior_sd),
                              G_b(self.splicing_rate_mean_hyp_prior_mean, self.splicing_rate_mean_hyp_prior_sd)))
        beta_sd = pyro.sample('beta_sd',
                   dist.Gamma(G_a(self.splicing_rate_sd_hyp_prior_mean, self.splicing_rate_sd_hyp_prior_sd),
                              G_b(self.splicing_rate_sd_hyp_prior_mean, self.splicing_rate_sd_hyp_prior_sd)))
        beta_g = pyro.sample('beta_g', dist.Gamma(G_a(beta_mu, beta_sd), G_b(beta_mu, beta_sd)).expand([1,self.n_vars]).to_event(2))
        # Degredation rate:
        gamma_mu = pyro.sample('gamma_mu',
                   dist.Gamma(G_a(self.degredation_rate_mean_hyp_prior_mean, self.degredation_rate_mean_hyp_prior_sd),
                              G_b(self.degredation_rate_mean_hyp_prior_mean, self.degredation_rate_mean_hyp_prior_sd)))
        gamma_sd = pyro.sample('gamma_sd',
                   dist.Gamma(G_a(self.degredation_rate_sd_hyp_prior_mean, self.degredation_rate_sd_hyp_prior_sd),
                              G_b(self.degredation_rate_sd_hyp_prior_mean, self.degredation_rate_sd_hyp_prior_sd)))
        gamma_g = pyro.sample('gamma_g', dist.Gamma(G_a(gamma_mu, gamma_sd), G_b(gamma_mu, gamma_sd)).expand([1, self.n_vars]).to_event(2))
        # Transcription rate contribution of each module:
        factor_level_g = pyro.sample(
            "factor_level_g",
            dist.Gamma(self.factor_prior_alpha, self.factor_prior_beta)
            .expand([1, self.n_vars])
            .to_event(2)
        )
        factor_level_m = pyro.sample("factor_level_m",
            dist.Gamma(self.one*20., self.one*20.)
            .expand([self.n_modules,1])
            .to_event(2))
        g_fg = pyro.sample( # (g_fg corresponds to module's spliced counts in steady state)
            "g_fg",
            dist.Gamma(
                self.factor_states_per_gene / self.n_factors_torch,
                self.ones / factor_level_g,
            )
            .expand([self.n_modules, self.n_vars])
            .to_event(2)
        )
        A_mgON = pyro.deterministic('A_mgON', g_fg*gamma_g*factor_level_m) # (transform from ss spliced counts to transcription rate)
        A_mgOFF = self.alpha_OFFg        
        # Module activation and Deactivation rate:
        lam_mu = pyro.sample('lam_mu', dist.Gamma(G_a(self.activation_rate_mean_hyp_prior_mean, self.activation_rate_mean_hyp_prior_sd),
                                            G_b(self.activation_rate_mean_hyp_prior_mean, self.activation_rate_mean_hyp_prior_sd)))
        lam_sd = pyro.sample('lam_sd', dist.Gamma(G_a(self.activation_rate_sd_hyp_prior_mean, self.activation_rate_sd_hyp_prior_sd),
                                            G_b(self.activation_rate_sd_hyp_prior_mean, self.activation_rate_sd_hyp_prior_sd)))
        lam_m_mu = pyro.sample('lam_m_mu', dist.Gamma(G_a(lam_mu, lam_sd),
                                            G_b(lam_mu, lam_sd)).expand([self.n_modules, 1, 1]).to_event(3))
        lam_mi = pyro.sample('lam_mi', dist.Gamma(G_a(lam_m_mu, lam_m_mu*0.05),
                                            G_b(lam_m_mu, lam_m_mu*0.05)).expand([self.n_modules, 1, 2]).to_event(3))
        
        # =====================Time======================= #
#         Tmax = pyro.sample('Tmax', dist.Gamma(G_a(self.Tmax_mean, self.Tmax_sd), G_b(self.Tmax_mean, self.Tmax_sd)).expand([1,self.n_modules,2]).to_event(3))
        Tmax = pyro.deterministic('Tmax', self.one*25.)
#         with obs_plate:
#             t_cmi = pyro.sample('t_c', dist.Gamma(G_a(self.one/(2.*self.one), self.one/(4.*self.one)),
#                                                  G_b(self.one/(2.*self.one), self.one/(4.*self.one))).
#                               expand([self.n_obs, self.n_modules, 2]))
        t_cmi = pyro.param('t_cmi', self.t_cmi_init, constraint=constraints.interval(0.05, 0.95))
        T_cmi = pyro.deterministic('T_cmi', t_cmi*Tmax)
#         t_mOFF = pyro.sample('t_mOFF', dist.Exponential(self.one).expand([1, self.n_modules, 1]).to_event(3))
        t_mOFF = pyro.param('t_mOFF', self.t_mOFF_init, constraint=constraints.positive)
        T_mOFF = pyro.deterministic('T_mOFF', t_mOFF*Tmax)
        
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
        stochastic_v_ag_inv = pyro.sample(
            "stochastic_v_ag_inv",
            dist.Exponential(stochastic_v_ag_hyp)
            .expand([1, self.n_vars, 2]).to_event(3),
        ) 
        stochastic_v_ag = (self.ones / stochastic_v_ag_inv.pow(2))
        
        # ===================== State of each module in each cell ======= #
        p_m = pyro.sample('p_m', dist.Dirichlet(self.alpha_dirichlet))
        with obs_plate:
            I_cm = pyro.sample("I_cm",
                               RelaxedOneHotCategorical(probs = p_m, temperature = self.one/10**6)
                               .expand([self.n_obs, 1, self.n_modules]))
            
        # ==================== Biological Mean ======================= #
        mu_total = torch.stack([self.zeros, self.zeros], axis = -1)
        for m in range(self.n_modules):
            mu_total += I_cm[...,m,1].unsqueeze(-1) * torch.stack([(A_mgON[m,:]/beta_g).repeat([self.n_obs,1]),
                    (A_mgON[m,:]/gamma_g).repeat([self.n_obs,1])], axis = -1)
            mu_total += I_cm[...,m,2].unsqueeze(-1) * mu_mRNA_continuousAlpha(A_mgON[m,:], beta_g, gamma_g, T_cmi[:,m,0].unsqueeze(-1),
                                           self.zeros, self.zeros, A_mgON[m,:], lam_mi[m,0,0]) 
            initial_state = mu_mRNA_continuousAlpha(A_mgON[m,:], beta_g, gamma_g, T_mOFF[:,m,:],
                                                   self.zeros, self.zeros, A_mgON[m,:], lam_mi[m,0,0])
            initial_alpha = mu_alpha(A_mgON[m,:], A_mgOFF, T_mOFF[:,m,:], lam_mi[m,0,0])
            mu_total += I_cm[...,m,3].unsqueeze(-1) * mu_mRNA_continuousAlpha(A_mgOFF, beta_g, gamma_g, T_cmi[:,m,1].unsqueeze(-1),
                                                                 initial_state[:,:,0], initial_state[:,:,1],
                                                                 initial_alpha*(-self.one), lam_mi[m,0,1])
        mu_expression = pyro.deterministic('mu_expression', torch.clip(mu_total, min = 10**(-5)))
        
        # observed counts mean
        mu = pyro.deterministic('mu', (mu_expression + torch.einsum('cbi,bgi->cgi', \
                                obs2sample.unsqueeze(dim=-1), s_g_gene_add)) * detection_y_c * detection_y_i * detection_y_gi)
        
        # =====================DATA likelihood ======================= #
        # Likelihood
        with obs_plate:
            pyro.sample("data_target", dist.GammaPoisson(concentration= stochastic_v_ag,
                       rate= stochastic_v_ag / mu), obs=torch.stack([u_data, s_data], axis = 2))

def mu_alpha(alpha_new, alpha_old, tau, lam):
    '''Calculates transcription rate as a function of new target transcription rate,
    old transcription rate at changepoint, time since change point and rate of transcription rate change.'''
    return (alpha_new - alpha_old) * (1 - torch.exp(-lam*tau)) + alpha_old
            
def mu_mRNA_continuousAlpha(alpha, beta, gamma, tau, u0, s0, delta_alpha, lam):
    ''' Calculates expected value of spliced and unspliced counts as a function of rates, latent time, initial states,
    difference to transcription rate in previous state and rate of transcription rate change.'''
    
    mu_u = u0*torch.exp(-beta*tau) + (alpha/beta)* (1 - torch.exp(-beta*tau)) + delta_alpha/(beta-lam+10**(-5))*(torch.exp(-beta*tau) - torch.exp(-lam*tau))
    mu_s = (s0*torch.exp(-gamma*tau) + 
    alpha/gamma * (1 - torch.exp(-gamma*tau)) +
    (alpha - beta * u0)/(gamma - beta+10**(-5)) * (torch.exp(-gamma*tau) - torch.exp(-beta*tau)) +
    (delta_alpha*beta)/((beta - lam+10**(-5))*(gamma - beta+10**(-5))) * (torch.exp(-beta*tau) - torch.exp(-gamma*tau))-
    (delta_alpha*beta)/((beta - lam+10**(-5))*(gamma - lam+10**(-5))) * (torch.exp(-lam*tau) - torch.exp(-gamma*tau)))

    return torch.stack([mu_u, mu_s], axis = -1)            
            
def mu_mRNA_continousAlpha_globalTime_twoStates(alpha_ON, alpha_OFF, beta, gamma, lam_gi, T_c, T_gON, T_gOFF, Zeros):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    global latent time, initial states and global switch times between two states'''
    n_cells = T_c.shape[-2]
    n_genes = alpha_ON.shape[-1]
    tau = torch.clip(T_c - T_gON, min = 10**(-5))
    t0 = T_gOFF - T_gON
    # Transcription rate in each cell for each gene:
    boolean = (tau < t0).reshape(n_cells, 1)
    alpha_cg = alpha_ON*boolean + alpha_OFF*~boolean
    # Time since changepoint for each cell and gene:
    tau_cg = tau*boolean + (tau - t0)*~boolean
    # Initial condition for each cell and gene:
    lam_g = ~boolean*lam_gi[:,1] + boolean*lam_gi[:,0]
    initial_state = mu_mRNA_continuousAlpha_withPlates(alpha_ON, beta, gamma, t0,
                                                       Zeros, Zeros, alpha_ON-alpha_OFF, lam_gi[:,0])
    initial_alpha = mu_alpha(alpha_ON, alpha_OFF, t0, lam_gi[:,0])
    u0_g = 10**(-5) + ~boolean*initial_state[:,:,0]
    s0_g = 10**(-5) + ~boolean*initial_state[:,:,1]
    delta_alpha = ~boolean*initial_alpha*(-1) + boolean*alpha_ON*(1)
    alpha_0 = alpha_OFF + ~boolean*initial_alpha
    # Unspliced and spliced count abundance for each gene in each cell:
    mu_RNAvelocity = torch.clip(mu_mRNA_continuousAlpha_withPlates(alpha_cg, beta, gamma, tau_cg,
                                                         u0_g, s0_g, delta_alpha, lam_g), min = 10**(-5))
    return mu_RNAvelocity            
            
            