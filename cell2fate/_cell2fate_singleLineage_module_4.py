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
from cell2fate.utils import G_a, G_b, mu_mRNA_continousAlpha_globalTime_twoStates_OnePlate2

class DifferentiationModel_OneLineage_DiscreteTwoStateTranscriptionRate(PyroModule):
    r"""
    - Models spliced and unspliced counts for each gene as a dynamical process in which each gene can switch on
    at one point in time to a specific transcription rate and then optionally switches off again to a transcription rate of 0.
    Splicing and degredation rates are constant for each gene. 
    - The underlying equations are similar to
    "Bergen et al. (2020), Generalizing RNA velocity to transient cell states through dynamical modeling"
    The difference is that time is cell-specific and thus shared across all genes.
    - In addition, the model includes negative binomial noise, batch effects and technical variables, similar to:
    "Kleshchevnikov et al. (2022), Cell2location maps fine-grained cell types in spatial transcriptomics".
    Although here they are modelled for both spliced and unspliced counts.
    """

    def __init__(
        self,
        n_obs,
        n_vars,
        n_batch,
        n_dim = 5,
        n_extra_categoricals=None,
        detection_alpha=1,
        dirichlet_alpha = 0.1,
        alpha_g_phi_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_alpha_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_mean_hyp_prior={
            "alpha": 0.5,
            "beta": 100.0,
        },
        detection_hyp_prior={"mean_alpha": 1.0, "mean_beta": 1.0},
        transcription_rate_hyp_prior={"mean_hyp_prior_mean": 1.0, "mean_hyp_prior_sd": 0.5,
                                     "sd_hyp_prior_mean": 1.0, "sd_hyp_prior_sd": 0.5},
        splicing_rate_hyp_prior={"mean_hyp_prior_mean": 1.0, "mean_hyp_prior_sd": 0.5,
                                 "sd_hyp_prior_mean": 0.5, "sd_hyp_prior_sd": 0.25},
        degredation_rate_hyp_prior={"mean_hyp_prior_mean": 0.2, "mean_hyp_prior_sd": 0.1,
                                    "sd_hyp_prior_mean": 0.1, "sd_hyp_prior_sd": 0.05},
        activation_rate_hyp_prior={"mean_hyp_prior_mean": 0.5, "mean_hyp_prior_sd": 0.1,
                                    "sd_hyp_prior_mean": 0.05, "sd_hyp_prior_sd": 0.05},
        s_overdispersion_factor_hyp_prior={'alpha_mean': 10, 'beta_mean': 1,
                                           'alpha_sd': 1, 'beta_sd': 0.1},
        u_detection_factor_mean_cv = 0.1,
        u_detection_factor_g_cv = 0.05,
        u_ambient_factor_mean_cv = 0.1,
        u_ambient_factor_g_cv = 0.1, 
        Tmax_prior={"mean": 50, "sd": 50},
        gene_tech_prior={"mean": 1, "alpha": 200},
        init_vals: Optional[dict] = None,
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

        self.n_dim = n_dim
        self.n_obs = n_obs
        self.n_vars = n_vars
        self.n_batch = n_batch
        self.n_extra_categoricals = n_extra_categoricals

        self.alpha_g_phi_hyp_prior = alpha_g_phi_hyp_prior
        self.gene_add_alpha_hyp_prior = gene_add_alpha_hyp_prior
        self.gene_add_mean_hyp_prior = gene_add_mean_hyp_prior
        self.detection_hyp_prior = detection_hyp_prior
        self.gene_tech_prior = gene_tech_prior
        self.transcription_rate_hyp_prior = transcription_rate_hyp_prior
        self.splicing_rate_hyp_prior = splicing_rate_hyp_prior
        self.degredation_rate_hyp_prior = degredation_rate_hyp_prior
        self.Tmax_prior = Tmax_prior
        detection_hyp_prior["alpha"] = detection_alpha
        self.s_overdispersion_factor_hyp_prior = s_overdispersion_factor_hyp_prior

        if (init_vals is not None) & (type(init_vals) is dict):
            self.np_init_vals = init_vals
            for k in init_vals.keys():
                self.register_buffer(f"init_val_{k}", torch.tensor(init_vals[k]))
        
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
            "u_detection_factor_mean_cv",
            torch.tensor(u_detection_factor_mean_cv),
        )
        
        self.register_buffer(
            "dirichlet_alpha",
            torch.ones(n_dim)*dirichlet_alpha,
        )
        
        self.register_buffer(
            "u_detection_factor_g_cv",
            torch.tensor(u_detection_factor_g_cv),
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
        self.register_buffer("ones_g", torch.ones((1,self.n_vars,1)))
        self.register_buffer("eps", torch.tensor(1e-8))
        self.register_buffer("alpha_OFFg", torch.tensor(10**(-5)))
        self.register_buffer("one", torch.tensor(1.))
        self.register_buffer("zero", torch.tensor(0.))
        self.register_buffer("zero_point_one", torch.tensor(0.1))
        self.register_buffer("one_point_one", torch.tensor(1.1))
        self.register_buffer("one_point_two", torch.tensor(1.2))
        self.register_buffer("zeros", torch.zeros(self.n_obs,self.n_vars))
        
        # Register parameters for transcription rate hyperprior:
        self.register_buffer(
            "transcription_rate_mean_hyp_prior_mean",
            torch.tensor(self.transcription_rate_hyp_prior["mean_hyp_prior_mean"]),
        )        
        self.register_buffer(
            "transcription_rate_mean_hyp_prior_sd",
            torch.tensor(self.transcription_rate_hyp_prior["mean_hyp_prior_sd"]),
        )
        self.register_buffer(
            "transcription_rate_sd_hyp_prior_mean",
            torch.tensor(self.transcription_rate_hyp_prior["sd_hyp_prior_mean"]),
        )
        self.register_buffer(
            "transcription_rate_sd_hyp_prior_sd",
            torch.tensor(self.transcription_rate_hyp_prior["sd_hyp_prior_sd"]),
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
        
        # Register parameters for maximum time:
        self.register_buffer(
            "Tmax_mean",
            torch.tensor(self.Tmax_prior["mean"]),
        )        
        self.register_buffer(
            "Tmax_sd",
            torch.tensor(self.Tmax_prior["sd"]),
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
        
        obs2sample = one_hot(batch_index, self.n_batch)        
        obs_plate = self.create_plates(u_data, s_data, idx, batch_index)
        
        with obs_plate:
            k_c = pyro.sample('k_c', dist.Dirichlet(self.dirichlet_alpha))
        
        # ===================== Kinetic Rates ======================= #
        # Transcription rate:
        alpha_mu = pyro.sample('alpha_mu',
                   dist.Gamma(G_a(self.transcription_rate_mean_hyp_prior_mean, self.transcription_rate_mean_hyp_prior_sd),
                              G_b(self.transcription_rate_mean_hyp_prior_mean, self.transcription_rate_mean_hyp_prior_sd)))
        alpha_sd = pyro.sample('alpha_sd',
                   dist.Gamma(G_a(self.transcription_rate_sd_hyp_prior_mean, self.transcription_rate_sd_hyp_prior_sd),
                              G_b(self.transcription_rate_sd_hyp_prior_mean, self.transcription_rate_sd_hyp_prior_sd)))
        alpha_ONgk = pyro.sample('alpha_ONgk', dist.Gamma(G_a(alpha_mu, alpha_sd), G_b(alpha_mu, alpha_sd)).expand([1, self.n_vars, 1, self.n_dim]).to_event(4))
        alpha_ONg = pyro.deterministic('alpha_g', torch.einsum('cgik,cgik->cg', k_c, alpha_ONgk).unsqueeze(-1))
        
        alpha_OFFg = self.alpha_OFFg
        # Splicing rate:
        beta_mu = pyro.sample('beta_mu',
                   dist.Gamma(G_a(self.splicing_rate_mean_hyp_prior_mean, self.splicing_rate_mean_hyp_prior_sd),
                              G_b(self.splicing_rate_mean_hyp_prior_mean, self.splicing_rate_mean_hyp_prior_sd)))
        beta_sd = pyro.sample('beta_sd',
                   dist.Gamma(G_a(self.splicing_rate_sd_hyp_prior_mean, self.splicing_rate_sd_hyp_prior_sd),
                              G_b(self.splicing_rate_sd_hyp_prior_mean, self.splicing_rate_sd_hyp_prior_sd)))
        beta_g = pyro.sample('beta_g', dist.Gamma(G_a(beta_mu, beta_sd), G_b(beta_mu, beta_sd)).expand([1,self.n_vars,1]).to_event(3))
        # Degredation rate:
        gamma_mu = pyro.sample('gamma_mu',
                   dist.Gamma(G_a(self.degredation_rate_mean_hyp_prior_mean, self.degredation_rate_mean_hyp_prior_sd),
                              G_b(self.degredation_rate_mean_hyp_prior_mean, self.degredation_rate_mean_hyp_prior_sd)))
        gamma_sd = pyro.sample('gamma_sd',
                   dist.Gamma(G_a(self.degredation_rate_sd_hyp_prior_mean, self.degredation_rate_sd_hyp_prior_sd),
                              G_b(self.degredation_rate_sd_hyp_prior_mean, self.degredation_rate_sd_hyp_prior_sd)))
        gamma_g = pyro.sample('gamma_g', dist.Gamma(G_a(gamma_mu, gamma_sd), G_b(gamma_mu, gamma_sd)).expand([1, self.n_vars,1]).to_event(3))
        # Activation and Deactivation rate:
        lam_mu = pyro.sample('lam_mu', dist.Gamma(G_a(self.activation_rate_mean_hyp_prior_mean, self.activation_rate_mean_hyp_prior_sd),
                                            G_b(self.activation_rate_mean_hyp_prior_mean, self.activation_rate_mean_hyp_prior_sd)))
        lam_sd = pyro.sample('lam_sd', dist.Gamma(G_a(self.activation_rate_sd_hyp_prior_mean, self.activation_rate_sd_hyp_prior_sd),
                                            G_b(self.activation_rate_sd_hyp_prior_mean, self.activation_rate_sd_hyp_prior_sd)))
        lam_g_mu = pyro.sample('lam_g_mu', dist.Gamma(G_a(lam_mu, lam_sd),
                                            G_b(lam_mu, lam_sd)).expand([self.n_vars, 1]).to_event(2))
        lam_gi = pyro.sample('lam_gi', dist.Gamma(G_a(lam_g_mu, lam_g_mu*0.1),
                                            G_b(lam_g_mu, lam_g_mu*0.1)).expand([self.n_vars, 2]).to_event(2))

        # =====================Time======================= #
        # Global time for each cell:
        T_max = pyro.sample('T_max', dist.Gamma(G_a(self.Tmax_mean, self.Tmax_sd), G_b(self.Tmax_mean, self.Tmax_sd)))
        with obs_plate:
            t_c = pyro.sample('t_c', dist.Uniform(self.zero, self.one).expand([self.n_obs, 1, 1]))
        T_c = pyro.deterministic('T_c', t_c*T_max)
        # Global switch on time for each gene:
        t_gON = pyro.sample('t_gON', dist.Uniform(self.zero, self.one).expand([1, self.n_vars, 1]).to_event(2))
        T_gON = pyro.deterministic('T_gON', -T_max*self.zero_point_one + t_gON*T_max*self.one_point_two)
        # Global switch off time for each gene:
        t_gOFF = pyro.sample('t_gOFF', dist.Uniform(self.zero, self.one).expand([1, self.n_vars, 1]).to_event(2))
        T_gOFF = pyro.deterministic('T_gOFF', T_gON + t_gOFF*(T_max*self.one_point_one - T_gON))

        # =========== Mean expression based on dynamical equations ======================= #
        mu_RNAvelocity =  pyro.deterministic('mu_RNAvelocity', mu_mRNA_continousAlpha_globalTime_twoStates_OnePlate2(alpha_ONg[:,:,0], alpha_OFFg, beta_g[:,:,0], gamma_g[:,:,0], lam_gi, T_c[:,:,0], T_gON[:,:,0], T_gOFF[:,:,0], self.zeros))         
        
        # =============Detection efficiency of spliced and unspliced counts =============== #
        # Spliced counts cell specific relative detection efficiency with hierarchical prior across batches:
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
            detection_y_cs = pyro.sample(
                "detection_y_cs",
                dist.Gamma((obs2sample @ detection_hyp_prior_alpha).unsqueeze(dim=-1),
                           beta.unsqueeze(dim=-1)).expand([self.n_obs, 1, 1]))
        # Relative detection efficiency of unspliced counts is scaled by a common factor across all cells:
        u_detection_factor_mean = pyro.sample(
            "u_detection_factor_mean",
            dist.Gamma(G_a(1, self.u_detection_factor_mean_cv), G_b(1, self.u_detection_factor_mean_cv)))
        detection_y_cu = pyro.deterministic(
            "detection_y_cu", detection_y_cs * u_detection_factor_mean)
        # Relative detection efficiency of unspliced counts is variable across genes: 
        u_detection_factor_g = pyro.sample("u_detection_factor_g",
            dist.Gamma(G_a(u_detection_factor_mean, self.u_detection_factor_g_cv*u_detection_factor_mean),
            G_b(u_detection_factor_mean, self.u_detection_factor_g_cv*u_detection_factor_mean)).expand([1, self.n_vars, 1]).to_event(3))
        
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
        alpha = self.ones / torch.concat([alpha_gu_inverse, alpha_gs_inverse], axis = -1).pow(2)
        # observed expression includes technical factors:
        mu = (mu_RNAvelocity + torch.einsum('cbi,bgi->cgi', obs2sample.unsqueeze(dim=-1), s_g_gene_add)) * \
        (torch.concat([detection_y_cu, detection_y_cs], axis = -1)*torch.concat([u_detection_factor_g, self.ones_g], axis = -1))
        
        # =====================DATA likelihood ======================= #
        # Likelihood (sampling distribution) of data_target & add overdispersion via NegativeBinomial
        with obs_plate:
            pyro.sample("data_target", dist.GammaPoisson(concentration= alpha, rate= alpha / mu),
                        obs=torch.stack([u_data, s_data], axis = 2))

    # =====================Other functions======================= #
    def compute_expected(self, samples, adata_manager, ind_x=None):
        r"""Compute expected expression of each gene in each cell. Useful for evaluating how well
        the model learned expression pattern of all genes in the data.

        Parameters
        ----------
        samples
            dictionary with expectation (mean) values of the posterior
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

        # Example:

    #     alpha = 1 / np.power(samples["alpha_g_inverse"], 2)

    #     mu = samples["per_cluster_mu_fg"] + np.dot(obs2sample, samples["s_g_gene_add"]) * np.dot(
    #         obs2sample, samples["detection_mean_y_e"])

        # New:

        alpha = 1 / np.power(np.concatenate([samples['alpha_gu_inverse'], samples['alpha_gs_inverse']], axis = -1), 2)

        mu = (samples['mu_RNAvelocity'][ind_x,:,:] + np.einsum('cbi,bgi->cgi', np.expand_dims(obs2sample, -1), samples['s_g_gene_add'])) * \
        (np.concatenate([samples['detection_y_cu'][ind_x,:,:], samples['detection_y_cs'][ind_x,:,:]], axis = -1)*
         np.concatenate([samples['u_detection_factor_g'], np.ones((1,self.n_vars,1))], axis = -1))

        return {"mu": mu, "alpha": alpha}
