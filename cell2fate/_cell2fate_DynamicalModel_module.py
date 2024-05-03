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

class Cell2fate_DynamicalModel_module(PyroModule):
    r"""
    Models spliced and unspliced counts for each gene as a dynamical process in which transcriptional modules switch on at one point in time and increase the transcription rate by different values across genes and then optionally switches off to a transcription rate of 0. Splicing and degradation rates are constant for each gene. The model also includes negative binomial noise, batch effects, and technical variables.

    Parameters
    ----------
    n_obs
        Number of observations in the dataset (e.g., number of cells or samples).
    n_vars
        Number of variables or features in the dataset (e.g., number of genes).
    n_batch
        Number of batches or experimental conditions in the dataset.
    n_extra_categoricals
        Number of additional categorical variables beyond the primary variables of interest.
    gene_add_alpha_hyp_prior
        Hyperparameter prior for the gene additive parameter.
    gene_add_mean_hyp_prior
        Hyperparameter prior for the mean of the gene additive parameter distribution.
    detection_hyp_prior
        Hyperparameter prior for the detection process.
    """

    def __init__(
        self,
        n_obs,
        n_vars,
        n_batch,
        n_extra_categoricals=None,
        n_modules = 10,
        stochastic_v_ag_hyp_prior={"alpha": 6.0, "beta": 3.0},
        factor_prior={"rate": 1.0, "alpha": 1.0, "states_per_gene": 3.0},
        t_switch_alpha_prior = {"mean": 1000., "alpha": 1000.},
        splicing_rate_hyp_prior={"mean": 1.0, "alpha": 5.0,
                                "mean_hyp_alpha": 10., "alpha_hyp_alpha": 20.},
        degredation_rate_hyp_prior={"mean": 1.0, "alpha": 5.0,
                                "mean_hyp_alpha": 10., "alpha_hyp_alpha": 20.},
        activation_rate_hyp_prior={"mean_hyp_prior_mean": 2., "mean_hyp_prior_sd": 0.33,
                                    "sd_hyp_prior_mean": 0.33, "sd_hyp_prior_sd": 0.1},
        s_overdispersion_factor_hyp_prior={'alpha_mean': 100., 'beta_mean': 1.,
                                           'alpha_sd': 1., 'beta_sd': 0.1},
        detection_hyp_prior={"alpha": 10.0, "mean_alpha": 1.0, "mean_beta": 1.0},
        detection_i_prior={"mean": 1, "alpha": 100},
        detection_gi_prior={"mean": 1, "alpha": 200},
        gene_add_alpha_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_mean_hyp_prior={"alpha": 1.0, "beta": 100.0},
        Tmax_prior={"mean": 50., "sd": 50.},
        switch_time_sd = 0.1,
        init_vals: Optional[dict] = None
    ):
        


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
        
        self.register_buffer("n_factors_torch", torch.tensor(self.n_modules))
        
        self.register_buffer(
            "factor_states_per_gene",
            torch.tensor(self.factor_prior["states_per_gene"]),
        )
        
        self.register_buffer(
            "t_c_init",
            self.one*torch.ones((self.n_obs, 1, 1))/2.,
        )
        
        self.register_buffer(
            "t_mON_init",
            torch.ones((1, 1, self.n_modules))/2.,
        )  
        
        self.register_buffer(
            "t_mOFF_init",
            torch.zeros((1, 1, self.n_modules)) + 0.1,
        )      
        
        self.register_buffer(
            "probs_I_cm",
            torch.tensor(1.-10**(-10)),
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
        """
        Creates a Pyro plate for observations.

        Parameters
        ----------
        u_data
            Unspliced count data.
        s_data
            Spliced count data.
        idx
            Index tensor to subsample.
        batch_index 
            Index tensor indicating batch assignments.

        Returns
        -------
        Pyro.plate
            A Pyro plate representing the observations in the dataset.
        """
            
        return pyro.plate("obs_plate", size=self.n_obs, dim=-3, subsample=idx)

            
    def list_obs_plate_vars(self):
        """
        Creates a dictionary with the name of observation/minibatch plate, indexes of model args to provide to encoder, variable names that belong to the observation plate and the number of dimensions in non-plate axis of each variable.
        
        Returns
        -------
        Dict
            A dictionary containing the following keys:
            
            - **name:** Name of the observation plate.
            
            - **input:** List of indexes of model arguments to provide to the encoder.
            
            - **input_transform:** List of transformations to apply to input data before passing to the neural network.
            
            - **sites:** Dictionary containing information about variables that belong to the observation plate, including their names and the number of dimensions in the non-plate axis of each variable.

        """

        return {
            "name": "obs_plate",
            "input": [],  # expression data + (optional) batch index
            "input_transform": [],  # how to transform input data before passing to NN
            "sites": {},
        }
    
    def forward(self, u_data, s_data, idx, batch_index):
        
        """
        Forward pass of the :class:`~cell2fate.Cell2fate_DynamicalModel_module`.

        Parameters
        ----------
        u_data
            Unspliced count data.
        s_data
            Spliced count data.
        idx
            Index tensor to subsample.
        batch_index
            Index tensor indicating batch assignments.
        """
        
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
                                            G_b(self.activation_rate_mean_hyp_prior_mean, self.activation_rate_mean_hyp_prior_sd)))
        lam_sd = pyro.sample('lam_sd', dist.Gamma(G_a(self.activation_rate_sd_hyp_prior_mean, self.activation_rate_sd_hyp_prior_sd),
                                            G_b(self.activation_rate_sd_hyp_prior_mean, self.activation_rate_sd_hyp_prior_sd)))
        lam_m_mu = pyro.sample('lam_m_mu', dist.Gamma(G_a(lam_mu, lam_sd),
                                            G_b(lam_mu, lam_sd)).expand([self.n_modules, 1, 1]).to_event(3))
        lam_mi = pyro.sample('lam_mi', dist.Gamma(G_a(lam_m_mu, lam_m_mu*0.05),
                                            G_b(lam_m_mu, lam_m_mu*0.05)).expand([self.n_modules, 1, 2]).to_event(3))
        
        # =====================Time======================= #
        # Global time for each cell:
        T_max = pyro.sample('Tmax', dist.Gamma(G_a(self.Tmax_mean, self.Tmax_sd), G_b(self.Tmax_mean, self.Tmax_sd)))
        t_c_loc = pyro.sample('t_c_loc', dist.Gamma(self.one, self.one/0.5))
        t_c_scale = pyro.sample('t_c_scale', dist.Gamma(self.one, self.one/0.25))
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
                       rate= stochastic_v_ag / mu), obs=torch.stack([u_data, s_data], axis = 2))