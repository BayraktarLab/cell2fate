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
from cell2fate.utils import G_a, G_b, mu_mRNA_discreteModularAlpha_localTime_4States
from pyro.infer import config_enumerate
from pyro.ops.indexing import Vindex

from pyro.distributions import RelaxedOneHotCategoricalStraightThrough
RelaxedOneHotCategoricalStraightThrough.mean = property(lambda self: self.probs)

class DifferentiationModel_ModularTranscriptionRate_FixedModules_LocalTime_4States(PyroModule):
    r"""
    - Models spliced and unspliced counts for each gene as a dynamical process in which transcriptional modules switch on
    at one point in time and increase the transcription rate by different values across genes and then optionally switches off
    to a transcription rate of 0. Splicing and degredation rates are constant for each gene. 
    - The underlying equations are similar to
    "Bergen et al. (2020), Generalizing RNA velocity to transient cell states through dynamical modeling"
    The difference is that modules are turned on gradually, rather an in a step-wise fashion. In addition, time is cell-specific 
    and thus shared across all genes. Furthermore, multiple lineages can be inferred with this model,
    by assuming different module switch times for each lineage.
    - In addition, the model includes negative binomial noise, batch effects and technical variables, similar to:
    "Kleshchevnikov et al. (2022), Cell2location maps fine-grained cell types in spatial transcriptomics".
    Although in the final version of this model technical variables will be modelled seperately for spliced and unspliced counts.
    """

    def __init__(
        self,
        factors,
        n_obs,
        n_vars,
        n_batch,
        n_extra_categoricals=None,
        n_lineages = 4,
        n_transitions = 8,
        n_modules = 10,
        detection_alpha=20.0,
        alpha_dirichlet = 1.,
        alpha_g_phi_hyp_prior={"alpha": 1.0, "beta": 1.0},
        gene_add_alpha_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_mean_hyp_prior={
            "alpha": 1.0,
            "beta": 10.0,
        },
        factor_prior={
            "rate": 1.0,
            "alpha": 1.0,
            "states_per_gene": 10.0},
        detection_hyp_prior={"mean_alpha": 1.0, "mean_beta": 1.0},
        module_activation_rate_prior={"mean": 100, "sd": 10},
        splicing_rate_hyp_prior={"mean_hyp_prior_mean": 0.8, "mean_hyp_prior_sd": 0.4,
                                 "sd_hyp_prior_mean": 0.04, "sd_hyp_prior_sd": 0.02},
        degredation_rate_hyp_prior={"mean_hyp_prior_mean": 0.2, "mean_hyp_prior_sd": 0.1,
                                    "sd_hyp_prior_mean": 0.1, "sd_hyp_prior_sd": 0.05},
        s_overdispersion_factor_hyp_prior={'alpha_mean': 100., 'beta_mean': 1.,
                                           'alpha_sd': 1., 'beta_sd': 0.1},
        factor_level_prior = {'alpha': 1.1 , 'beta': 0.5},
        T_OFF_prior={"mean": 2, "sd": 1},
        Tmax_k_prior={"alpha": 1., "beta": 10.},
        gene_tech_prior={"mean": 1., "alpha": 200.},
        init_vals: Optional[dict] = None
    ):
        
        """

        Parameters
        ----------
        n_obs
        n_vars
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
        self.n_lineages = n_lineages
        self.n_modules = n_modules
        self.n_transitions = n_transitions
        self.n_obs = n_obs
        self.n_vars = n_vars
        self.n_batch = n_batch
        self.n_extra_categoricals = n_extra_categoricals
        self.factor_prior = factor_prior

        self.alpha_g_phi_hyp_prior = alpha_g_phi_hyp_prior
        self.gene_add_alpha_hyp_prior = gene_add_alpha_hyp_prior
        self.gene_add_mean_hyp_prior = gene_add_mean_hyp_prior
        self.detection_hyp_prior = detection_hyp_prior
        self.gene_tech_prior = gene_tech_prior
        self.module_activation_rate_prior = module_activation_rate_prior
        self.splicing_rate_hyp_prior = splicing_rate_hyp_prior
        self.degredation_rate_hyp_prior = degredation_rate_hyp_prior
        self.T_OFF_prior = T_OFF_prior
        detection_hyp_prior["alpha"] = detection_alpha
        self.s_overdispersion_factor_hyp_prior = s_overdispersion_factor_hyp_prior

        if (init_vals is not None) & (type(init_vals) is dict):
            self.np_init_vals = init_vals
            for k in init_vals.keys():
                self.register_buffer(f"init_val_{k}", torch.tensor(init_vals[k]))
                
        self.register_buffer(
            "factors",
            torch.tensor(factors),
        )
                
        self.register_buffer(
            "factor_level_alpha",
            torch.tensor(factor_level_prior["alpha"]),
        )
        self.register_buffer(
            "factor_level_beta",
            torch.tensor(factor_level_prior["beta"]),
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
            "Tmax_k_alpha",
            torch.tensor(Tmax_k_prior['alpha']),
        )
        self.register_buffer(
            "Tmax_k_beta",
            torch.tensor(Tmax_k_prior['beta']),
        )

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
        
        # Register parameters for module activation rate:
        self.register_buffer(
            "module_activation_rate_mean",
            torch.tensor(self.module_activation_rate_prior["mean"]),
        )        
        self.register_buffer(
            "module_activation_rate_sd",
            torch.tensor(self.module_activation_rate_prior["sd"]),
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
        
        # Register parameters for maximum time:
        self.register_buffer(
            "T_OFF_mean",
            torch.tensor(self.T_OFF_prior["mean"]),
        )        
        self.register_buffer(
            "T_OFF_sd",
            torch.tensor(self.T_OFF_prior["sd"]),
        )
        
        self.register_buffer(
            "alpha_dirichlet",
            torch.tensor(alpha_dirichlet*torch.ones((4))),
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
            "ps_categorical_probs",
                    torch.ones(self.n_modules)/(n_modules)
        )
        
        self.register_buffer(
            "I_ctm_initial",
                            torch.zeros((self.n_obs, self.n_transitions, self.n_modules))
        )
        
        self.register_buffer(
            "ps_binary_initial",
                                    torch.zeros((self.n_modules, self.n_modules))
        )
        
        self.register_buffer(
            "ps_initial", torch.diag_embed(torch.ones(n_modules -1), offset = 1)
        )
        
        self.register_buffer(
            "t_ctON_initial", torch.zeros((self.n_obs, self.n_transitions, 1))
        )
        
        self.register_buffer(
            "t_ctOFF_initial", torch.zeros((self.n_obs, self.n_transitions, 1))
        )
        
        self.register_buffer(
            "I_ctm_initial_probs", torch.ones(1, self.n_modules)/self.n_modules)
        
        self.register_buffer(
            "t_ctON_initial", torch.zeros(self.n_obs, 1, 1))
            
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
        factor_level_m = pyro.sample("factor_level_m",
            dist.Gamma(self.factor_level_alpha, self.factor_level_beta).expand([self.n_modules,1]).to_event(2))
        A_mgON = pyro.deterministic('A_mgON', factor_level_m*self.factors*(1/beta_g[0,:] + 1/gamma_g[0,:])**(-1))
        A_mgOFF = self.alpha_OFFg
        # Module activation rate:
        lam = pyro.sample('lam', dist.Gamma(G_a(self.module_activation_rate_mean, self.module_activation_rate_sd),
                                            G_b(self.module_activation_rate_mean, self.module_activation_rate_sd)))

        # =====================Time======================= #
        # State of each module in each cell:
        w_k = pyro.sample('w_k', dist.Dirichlet(self.alpha_dirichlet))
        I_cm = pyro.sample('I_cm',
                           RelaxedOneHotCategoricalStraightThrough(probs = w_k,
                                                                        temperature = self.one/10**3
                                                                       ).expand([self.n_obs, self.n_modules]).to_event(2))
        # Maximal Time in Induction State:
        T_OFF_hyper = pyro.sample('T_OFF_hyper', dist.Gamma(G_a(self.T_OFF_mean, self.T_OFF_sd), G_b(self.T_OFF_mean, self.T_OFF_sd)
                                               ).expand([1,1, 1]).to_event(3))
        T_mOFF = pyro.sample('T_mOFF', dist.Exponential(self.one/T_OFF_hyper).expand([1, self.n_modules, 1]).to_event(3))
        # Maximal Time in Repression State:
        T_MAX_hyper = pyro.sample('T_MAX_hyper', dist.Gamma(G_a(self.T_OFF_mean, self.T_OFF_sd), G_b(self.T_OFF_mean, self.T_OFF_sd)
                                               ).expand([1,1, 1]).to_event(3))
        T_mMAX = pyro.sample('T_mMAX', dist.Exponential(self.one/T_MAX_hyper).expand([1, self.n_modules, 1]).to_event(3))       
        # Time given induction or repression State:
        with obs_plate:
            T_cmON = pyro.sample('T_cmON', dist.Exponential(self.one/T_mOFF).expand([self.n_obs, self.n_modules, 1]))
#         with obs_plate:
#             t_cmON = pyro.sample('t_cmON', dist.Uniform(self.zero, self.one).expand([self.n_obs, self.n_modules, 1]))
#         T_cmON = pyro.deterministic('T_cmON', t_cmON*T_mOFF)
        with obs_plate:
            T_cmOFF = pyro.sample('T_cmOFF', dist.Exponential(self.one/T_mMAX).expand([self.n_obs, self.n_modules, 1]))
#         with obs_plate:
#             t_cmOFF = pyro.sample('t_cmOFF', dist.Uniform(self.zero, self.one).expand([self.n_obs, self.n_modules, 1]))
#         T_cmOFF = pyro.deterministic('T_cmOFF', t_cmOFF*T_mMAX)
        # =========== Mean expression according to RNAvelocity model ======================= #
        mu_expression = pyro.deterministic('mu_expression', mu_mRNA_discreteModularAlpha_localTime_4States(
            A_mgON, A_mgOFF, beta_g, gamma_g, T_mOFF, T_cmON, T_cmOFF, I_cm, lam, self.zeros))
        
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
        detection_hyp_prior_alpha = pyro.deterministic(
            "detection_hyp_prior_alpha",
            self.ones_n_batch_1 * self.detection_hyp_prior_alpha,
        )

        beta = (obs2sample @ detection_hyp_prior_alpha) / (obs2sample @ detection_mean_y_e)
        with obs_plate:
            detection_y_c = pyro.sample(
                "detection_y_c",
                dist.Gamma((obs2sample @ detection_hyp_prior_alpha).unsqueeze(dim=-1), beta.unsqueeze(dim=-1)).expand([len(idx), 1, 1]))  # (self.n_obs, 1)
        
        # =====================Gene-specific additive component ======================= #
        # s_{e,g} accounting for background, free-floating RNA
        s_g_gene_add_alpha_hyp = pyro.sample(
            "s_g_gene_add_alpha_hyp",
            dist.Gamma(self.gene_add_alpha_hyp_prior_alpha, self.gene_add_alpha_hyp_prior_beta),
        )
        s_g_gene_add_mean = pyro.sample(
            "s_g_gene_add_mean",
            dist.Gamma(
                self.gene_add_mean_hyp_prior_alpha,
                self.gene_add_mean_hyp_prior_beta,
            )
            .expand([self.n_batch, 1])
            .to_event(2),
        ) 
        s_g_gene_add_alpha_e_inv = pyro.sample(
            "s_g_gene_add_alpha_e_inv",
            dist.Exponential(s_g_gene_add_alpha_hyp).expand([self.n_batch, 1]).to_event(2),
        )
        s_g_gene_add_alpha_e = self.ones / s_g_gene_add_alpha_e_inv.pow(2)

        s_g_gene_add = pyro.sample(
            "s_g_gene_add",
            dist.Gamma(s_g_gene_add_alpha_e, s_g_gene_add_alpha_e / s_g_gene_add_mean)
            .expand([self.n_batch, self.n_vars])
            .to_event(2),
        )

        # =========Gene-specific overdispersion of spliced and unspliced counts ============== #
        # Overdispersion of unspliced counts:
        alpha_g_phi_hyp = pyro.sample(
            "alpha_g_phi_hyp",
            dist.Gamma(self.alpha_g_phi_hyp_prior_alpha, self.alpha_g_phi_hyp_prior_beta),
        )
        alpha_gu_inverse = pyro.sample(
            "alpha_gu_inverse",
            dist.Exponential(alpha_g_phi_hyp).expand([1, self.n_vars,1]).to_event(2),
        )
        # Overdispersion of spliced counts:
        s_overdispersion_factor_alpha = pyro.sample(
            "s_overdispersion_factor_alpha", 
            dist.Gamma(G_a(self.s_overdispersion_factor_alpha_mean, self.s_overdispersion_factor_alpha_sd),
                       G_b(self.s_overdispersion_factor_alpha_mean, self.s_overdispersion_factor_alpha_sd)))
        s_overdispersion_factor_beta = pyro.sample(
            "s_overdispersion_factor_beta",
            dist.Gamma(G_a(self.s_overdispersion_factor_beta_mean, self.s_overdispersion_factor_beta_sd),
                       G_b(self.s_overdispersion_factor_beta_mean, self.s_overdispersion_factor_beta_sd)))
        s_overdispersion_factor_g = pyro.sample("s_overdispersion_factor_g",
            dist.Beta(s_overdispersion_factor_alpha, s_overdispersion_factor_beta).expand([1, self.n_vars, 1]).to_event(3))
        alpha_gs_inverse = pyro.deterministic(
            "alpha_gs_inverse", alpha_gu_inverse * s_overdispersion_factor_g)

        # =====================Expected expression ======================= #
        # overdispersion
        alpha = pyro.deterministic('alpha', self.ones / torch.concat([alpha_gu_inverse, alpha_gs_inverse], axis = -1).pow(2))
        # biological expression
        mu = pyro.deterministic('mu', (mu_expression + (obs2sample @ s_g_gene_add).unsqueeze(dim=-1)  # contaminating RNA
        ) * detection_y_c)  # cell-specific normalisation
        
        # =====================DATA likelihood ======================= #
        # Likelihood (sampling distribution) of data_target & add overdispersion via NegativeBinomial
        with obs_plate:
            pyro.sample("data_target", dist.GammaPoisson(concentration= alpha,
                       rate= alpha / mu), obs=torch.stack([u_data, s_data], axis = 2))

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

        mu = samples["per_cluster_mu_fg"] + np.dot(obs2sample, samples["s_g_gene_add"]) * np.dot(
            obs2sample, samples["detection_mean_y_e"]
        )  # samples["detection_y_c"][ind_x, :]
        if self.n_extra_categoricals is not None:
            mu = mu * np.dot(obs2extra_categoricals, samples["detection_tech_gene_tg"])

        return {"mu": mu, "alpha": alpha}
