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
from pyro.infer import config_enumerate

def mu_alpha(alpha_new, alpha_old, tau, lam):
    '''Calculates transcription rate as a function of new target transcription rate,
    old transcription rate at changepoint, time since change point and rate of exponential change process'''
    return (alpha_new - alpha_old) * (1 - torch.exp(-lam*tau)) + alpha_old

def mu_mRNA_continuousAlpha(alpha, beta, gamma, tau, u0, s0, delta_alpha, lam):
    ''' Calculates expected value of spliced and unspliced counts as a function of rates, latent time, initial states,
    difference to transcription rate in previous state and rate of exponential change process between states.'''
    
    print('u0', u0.shape)
    print('part1', (torch.exp(-beta*tau)).shape)
    print('part2', (alpha/beta).shape)
    print('part3', (1 - torch.exp(-beta*tau)).shape)
    
    mu_u = u0*torch.exp(-beta*tau) + (alpha/beta)* (1 - torch.exp(-beta*tau)) + delta_alpha/(beta-lam+10**(-5))*(torch.exp(-beta*tau) - torch.exp(-lam*tau))
    mu_s = (s0*torch.exp(-gamma*tau) + 
    alpha/gamma * (1 - torch.exp(-gamma*tau)) +
    (alpha - beta * u0)/(gamma - beta+10**(-5)) * (torch.exp(-gamma*tau) - torch.exp(-beta*tau)) +
    (delta_alpha*beta)/((beta - lam+10**(-5))*(gamma - beta+10**(-5))) * (torch.exp(-beta*tau) - torch.exp(-gamma*tau))-
    (delta_alpha*beta)/((beta - lam+10**(-5))*(gamma - lam+10**(-5))) * (torch.exp(-lam*tau) - torch.exp(-gamma*tau)))

    return torch.stack([mu_u, mu_s], axis = -1)

def mu_mRNA_continousAlpha_globalTime_twoStates_Enumerate(alpha_ON, alpha_OFF, beta, gamma, lam_gi, T_c, T_gON, T_gOFF, Zeros):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    global latent time, initial states and global switch times between two states'''
    n_cells = T_c.shape[-4]
    n_genes = alpha_ON.shape[-1]
    tau = torch.clip(T_c - T_gON, min = 10**(-5))
    t0 = T_gOFF - T_gON
    # Transcription rate in each cell for each gene:
    boolean = (tau < t0).reshape(n_cells, 1, 1, 1)
    alpha_cg = alpha_ON*boolean + alpha_OFF*~boolean
    # Time since changepoint for each cell and gene:
    tau_cg = tau*boolean + (tau - t0)*~boolean
    # Initial condition for each cell and gene:
    lam_g = ~boolean*lam_gi[:,1] + boolean*lam_gi[:,0]
    initial_state = mu_mRNA_continuousAlpha(alpha_ON, beta, gamma, t0,
                                            Zeros, Zeros, alpha_ON-alpha_OFF, lam_gi[:,0].unsqueeze(-1))
    initial_alpha = mu_alpha(alpha_ON, alpha_OFF, t0, lam_gi[:,0])
    print('initial_alpha', initial_alpha.shape)
    u0_g = 10**(-5) + ~boolean*initial_state[...,0]
    s0_g = 10**(-5) + ~boolean*initial_state[...,1]
    delta_alpha = ~boolean*initial_alpha*(-1) + boolean*alpha_ON*(1)
    alpha_0 = alpha_OFF + ~boolean*initial_alpha
    # Unspliced and spliced count variance for each gene in each cell:
    mu_RNAvelocity = torch.clip(mu_mRNA_continuousAlpha(alpha_cg, beta, gamma, tau_cg,
                                                         u0_g, s0_g, delta_alpha, lam_g), min = 10**(-5))
    return mu_RNAvelocity

class Cell2fate_DynamicalModel_PriorKnowledge_Sequential_module(PyroModule):
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
        clusters,
        is_tf,
        initial_clusters = [],
        terminal_clusters = [],
        n_extra_categoricals=None,
        n_modules = 10,
        stochastic_v_ag_hyp_prior={"alpha": 6.0, "beta": 3.0},
        factor_prior={"rate": 1.0, "alpha": 1.0, "states_per_gene": 3.0},
        effect_prior={'alpha': 1.0, 'beta': 1.0/10**(-6)},
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
        Tmax_prior={"mean": 50., "sd": 50.},
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
        self.n_modules = n_modules
        self.n_obs = n_obs
        self.n_vars = n_vars
        self.n_batch = n_batch
        self.n_tfs = torch.sum(torch.tensor(is_tf))
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
        self.is_tf = is_tf
        self.discrete_variables = ["J_" + str(m) + 'c' for m in range(n_modules)]

        if (init_vals is not None) & (type(init_vals) is dict):
            self.np_init_vals = init_vals
            for k in init_vals.keys():
                if k == 'I_cm':
                    self.register_buffer(f"init_val_{k}", torch.tensor(init_vals[k], dtype = torch.long))
                else:
                    self.register_buffer(f"init_val_{k}", torch.tensor(init_vals[k]))              
        
        t_c_mean = torch.ones((self.n_obs,1,1,1))*0.5
        t_c_mean[np.array([c in initial_clusters for c in clusters]),...] = 0.2
        t_c_mean[np.array([c in terminal_clusters for c in clusters]),...] = 0.8
        
        t_c_sd = torch.ones((self.n_obs,1,1,1))*0.25
        t_c_sd[np.array([c in initial_clusters for c in clusters]),...] = 0.1
        t_c_sd[np.array([c in terminal_clusters for c in clusters]),...] = 0.1
        
        self.register_buffer(
            "t_c_loc",
            torch.tensor(t_c_mean, dtype = torch.float32),
        )     
        
        self.register_buffer(
            "t_c_scale",
            torch.tensor(t_c_sd, dtype = torch.float32),
        )  
        
        self.register_buffer(
            "n_modules_torch",
            torch.tensor(n_modules, dtype = torch.float32),
        ) 
        
        self.register_buffer(
            "effect_prior_alpha",
            torch.tensor(effect_prior["alpha"]),
        )
        
        self.register_buffer(
            "effect_prior_beta",
            torch.tensor(effect_prior["beta"]),
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
        self.register_buffer("zeros", torch.zeros(self.n_obs, self.n_vars, self.n_obs, 1))
        self.register_buffer("ones_g", torch.ones((1,self.n_vars,1)))
        self.register_buffer("ones_m", torch.ones(n_modules))
        self.register_buffer("ones_m1", torch.ones(n_modules-1))
        self.register_buffer("ones_lam", torch.ones(self.n_modules, 1, 2, 1))
        self.register_buffer("zeros_gg", torch.zeros(self.n_vars, self.n_vars))
        self.register_buffer("zeros_g", torch.zeros(1, self.n_vars, 1, 1))
        
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
        return pyro.plate("obs_plate", size=self.n_obs, dim=-4, subsample=idx)

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
        splicing_alpha = pyro.sample('splicing_alpha',
                              dist.Gamma(self.splicing_rate_alpha_hyp_prior_alpha,
                              self.splicing_rate_alpha_hyp_prior_alpha/self.splicing_rate_alpha_hyp_prior_mean))
        splicing_mean = pyro.sample('splicing_mean',
                              dist.Gamma(self.splicing_rate_mean_hyp_prior_alpha,
                              self.splicing_rate_mean_hyp_prior_alpha/self.splicing_rate_mean_hyp_prior_mean))
        beta_g = pyro.sample('beta_g', dist.Gamma(splicing_alpha, splicing_alpha/splicing_mean).expand([1,self.n_vars,1,1]).to_event(4))
        # Degredation rate:
        degredation_alpha = pyro.sample('degredation_alpha',
                              dist.Gamma(self.degredation_rate_alpha_hyp_prior_alpha,
                              self.degredation_rate_alpha_hyp_prior_alpha/self.degredation_rate_alpha_hyp_prior_mean))
        degredation_alpha = degredation_alpha + 0.001
        degredation_mean = pyro.sample('degredation_mean',
                              dist.Gamma(self.degredation_rate_mean_hyp_prior_alpha,
                              self.degredation_rate_mean_hyp_prior_alpha/self.degredation_rate_mean_hyp_prior_mean))
        gamma_g = pyro.sample('gamma_g', dist.Gamma(degredation_alpha, degredation_alpha/degredation_mean).expand([1,self.n_vars,1,1]).to_event(4))
        
        # Transcription rate contribution of each module:
        factor_level_g = pyro.sample(
            "factor_level_g",
            dist.Gamma(self.factor_prior_alpha, self.factor_prior_beta)
            .expand([1, self.n_vars, 1, 1])
            .to_event(4)
        )
        # Sequential dependence between modules:
        # First module
        g_0g = pyro.sample( 
            "g_0g",
            dist.Gamma(
                self.factor_states_per_gene / self.n_factors_torch,
                self.ones / factor_level_g,
            )
            .expand([1, self.n_vars, 1, 1])
            .to_event(4)
        )
        # Linear dependence between modules:
        mean_effect_t = pyro.sample(
            "mean_effect_t",
            dist.Gamma(self.effect_prior_alpha, self.effect_prior_beta)
            .expand([1, self.n_tfs, 1, 1])
            .to_event(4)
        )
        X_gtj = pyro.sample( 
            "X_gtj",
            dist.Gamma(
                self.one,
                self.one/mean_effect_t,
            )
            .expand([self.n_vars, self.n_tfs, 2, 1])
            .to_event(4)
        )
        A_mgOFF = self.alpha_OFFg        
        # Activation and Deactivation rate:
        lam_mu = pyro.sample('lam_mu', dist.Gamma(G_a(self.activation_rate_mean_hyp_prior_mean, self.activation_rate_mean_hyp_prior_sd),
                                       G_b(self.activation_rate_mean_hyp_prior_mean, self.activation_rate_mean_hyp_prior_sd)))
        lam_mi = pyro.deterministic('lam_mi', lam_mu * self.ones_lam)
        
        # =====================Time======================= #
        # Global time for each cell:
        T_max = pyro.sample('Tmax', dist.Gamma(G_a(self.Tmax_mean, self.Tmax_sd), G_b(self.Tmax_mean, self.Tmax_sd)))
        with obs_plate as indx:
            t_c = pyro.sample('t_c', dist.Normal(self.t_c_loc[indx], self.t_c_scale[indx]).expand([batch_size, 1, 1, 1]))
            T_c = pyro.deterministic('T_c', t_c*T_max)
        
        # Global switch times for each module:
        t_delta = pyro.deterministic('t_delta', self.ones_m/self.n_modules_torch)
        t_mON = torch.cumsum(torch.concat([self.zero.unsqueeze(0), t_delta[:-1]]), dim = 0).unsqueeze(0).unsqueeze(0)
        T_mON = pyro.deterministic('T_mON', T_max*t_mON)
        # Global switch off time for each module:
        t_mOFF = torch.cumsum(t_delta, 0).unsqueeze(0).unsqueeze(0)
        T_mOFF = pyro.deterministic('T_mOFF', T_max*t_mOFF)
        
        # =========== Mean expression according to RNAvelocity model ======================= #
        mu_total = (torch.stack([self.zeros, self.zeros], axis = -1)[idx,:,idx,...]).repeat(1,1,batch_size,1)
        mu_total_T0_list = []
        m = 0
        with obs_plate:
            # Latent state of each cell:
            J = pyro.sample(
                "J_c",
                dist.Bernoulli(self.one/2).expand([batch_size, 1, 1, 1]),
                infer={"enumerate": "parallel"})
        X = X_gtj[...,torch.squeeze(J_c).long(),:].movedim(2,0).movedim(2,1)
        print('X', X.shape)
        mu_total_T0_list += [g_0g.repeat(batch_size,1,batch_size,1)] #[torch.einsum('itjk, gtck->gcki', g_0g[:, self.is_tf,...], X) + 1e-5]
        A_mgON_list = [g_0g.repeat(batch_size,1,batch_size,1)*gamma_g]
        g_fg_list = [g_0g.repeat(batch_size,1,batch_size,1)]
        # Global switch times for each module:
        T_mON_list  = [self.zero.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)]
        # Global switch off time for each module:
        T_mOFF_list = [T_max*t_delta[0].unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)]       
        
        # Using pyro.markov as a context manager:
        for m in pyro.markov(range(0, self.n_modules), history=1):
            
            print('M', m)
            # Global switch times for each module:
            T_mON_list  += [T_mON_list[-1] + T_max*t_delta[0].unsqueeze(0).unsqueeze(0).unsqueeze(0)]
            # Global switch off time for each module:
            T_mOFF_list += [T_mOFF_list[-1] + T_max*t_delta[0].unsqueeze(0).unsqueeze(0).unsqueeze(0)]
            
            # Transcription rate of each module:
            X = X_gtj[...,torch.squeeze(J_c).long(),:].movedim(2,0).movedim(2,1)
            print('X', X.shape)
            mean = torch.einsum('itjk, gtck->icgk', mu_total_T0_list[-1][:, self.is_tf,...], X) + 1e-5
            print('mean', mean.shape)
            g_fg_list += [pyro.deterministic("g_" + str(m+1) + 'g', mean)]
            A_mgON_list += [g_fg_list[-1] * gamma_g]
            
            # Counts of each cell
            A = A_mgON_list[-1]
            print('A_mg', A.shape)
            print('mu_total_T0_list', mu_total_T0_list[-1].shape)
            mu_total_T0_list = [mu_total_T0_list[-1] + mu_mRNA_continousAlpha_globalTime_twoStates_Enumerate(
                A, A_mgOFF, beta_g, gamma_g, lam_mi[0, ...],
                T_mON_list[-1], T_mON_list[-2], T_mOFF_list[- 1], self.zeros_g)[..., 1]]
            
            print('mu_total_T0_list', mu_total_T0_list[-1].shape)
            mu_test =  mu_mRNA_continousAlpha_globalTime_twoStates_Enumerate(
                A, A_mgOFF, beta_g, gamma_g, lam_mi[0, ...], T_c, T_mON_list[-1],
                T_mOFF_list[-1], self.zeros[idx,:,idx,:].unsqueeze(-1))
            print('mu_test', mu_test.squeeze(-3).squeeze(-2).shape)
            print('mu_total', mu_total.shape)
            mu_total += mu_test.squeeze(-3).squeeze(-2)
            
#         with obs_plate:
        mu_expression = pyro.deterministic('mu_expression', mu_total)            
        A_mgON = pyro.deterministic('A_mgON', torch.stack(A_mgON_list, axis = 0))
        g_fg = pyro.deterministic('g_fg', torch.stack(g_fg_list, axis = 0))
        
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
                dist.Gamma(detection_hyp_prior_alpha.unsqueeze(dim=-1),
                           beta.unsqueeze(dim=-1)),
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
        
        print('mu_expression', mu_expression.shape)
        print('tensor2', (torch.einsum('cbi,bgi->cgi', obs2sample.unsqueeze(dim=-1), s_g_gene_add)).shape)
        
        with obs_plate:
            mu = pyro.deterministic('mu', (mu_expression.movedim(-2,-1)))
#             + torch.einsum('cbi,bgi->cgi', obs2sample.unsqueeze(dim=-1), s_g_gene_add)) * \
#         detection_y_c * detection_y_i * detection_y_gi)
        
        print('mu', mu.shape)
    
        # =====================DATA likelihood ======================= #
        # Likelihood (sampling distribution) of data_target & add overdispersion via NegativeBinomial
        with obs_plate:
            pyro.sample("data_target", dist.GammaPoisson(concentration= self.one,
                       rate= self.one/ mu).to_event(4), obs=torch.stack([u_data, s_data], axis = 2).unsqueeze(-1))