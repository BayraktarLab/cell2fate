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
from cell2fate.utils import G_a, G_b, mu_mRNA_discreteAlpha_globalTime_twoStates_OnePlate2

class DifferentiationModel_MultiLineage_DiscreteTwoStateTranscriptionRate(PyroModule):
    r"""
    - Models spliced and unspliced counts for each gene as a dynamical process in which each gene can switch on
    at one point in time to a specific transcription rate and then optionally switches off again to a transcription rate of 0.
    Splicing and degredation rates are constant for each gene. 
    - The underlying equations are similar to
    "Bergen et al. (2020), Generalizing RNA velocity to transient cell states through dynamical modeling"
    The difference is that time is cell-specific and thus shared across all genes. Furthermore, multiple lineages can be
    inferred with this model, by assuming different transcription rates and switch times in each lineage.
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
        detection_alpha=200.0,
        alpha_g_phi_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_alpha_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_mean_hyp_prior={
            "alpha": 1.0,
            "beta": 100.0,
        },
        detection_hyp_prior={"mean_alpha": 1.0, "mean_beta": 1.0},
        transcription_rate_hyp_prior={"mean_hyp_prior_mean": 1.0, "mean_hyp_prior_sd": 0.5,
                                     "sd_hyp_prior_mean": 1.0, "sd_hyp_prior_sd": 0.5},
        splicing_rate_hyp_prior={"mean_hyp_prior_mean": 0.05, "mean_hyp_prior_sd": 0.025,
                                 "sd_hyp_prior_mean": 0.025, "sd_hyp_prior_sd": 0.0125},
        degredation_rate_hyp_prior={"mean_hyp_prior_mean": 0.2, "mean_hyp_prior_sd": 0.1,
                                    "sd_hyp_prior_mean": 0.1, "sd_hyp_prior_sd": 0.05},
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
        
        self.register_buffer(
            "detection_hyp_prior_alpha",
            torch.tensor(self.detection_hyp_prior["alpha"]),
        )
        
        self.register_buffer("ones_n_batch_1", torch.ones((self.n_batch, 1)))
        
        self.register_buffer("ones", torch.ones((1, 1)))
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
        
        # ===================== Kinetic Rates ======================= #
        # Transcription rate:
        alpha_mu = pyro.sample('alpha_mu',
                   dist.Gamma(G_a(self.transcription_rate_mean_hyp_prior_mean, self.transcription_rate_mean_hyp_prior_sd),
                              G_b(self.transcription_rate_mean_hyp_prior_mean, self.transcription_rate_mean_hyp_prior_sd)))
        alpha_sd = pyro.sample('alpha_sd',
                   dist.Gamma(G_a(self.transcription_rate_sd_hyp_prior_mean, self.transcription_rate_sd_hyp_prior_sd),
                              G_b(self.transcription_rate_sd_hyp_prior_mean, self.transcription_rate_sd_hyp_prior_sd)))
        alpha_ONg = pyro.sample('alpha_g', dist.Gamma(alpha_mu, alpha_sd).expand([1, self.n_vars, 1]).to_event(3))
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

        # =========== Mean expression according to RNAvelocity model ======================= #
        mu_RNAvelocity =  pyro.deterministic('mu_RNAvelocity',
                          mu_mRNA_discreteAlpha_globalTime_twoStates_OnePlate2(alpha_ONg[:,:,0], alpha_OFFg, beta_g[:,:,0], gamma_g[:,:,0],
                                                                               T_c[:,:,0], T_gON[:,:,0], T_gOFF[:,:,0], self.zeros))
        
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
                dist.Gamma((obs2sample @ detection_hyp_prior_alpha).unsqueeze(dim=-1), beta.unsqueeze(dim=-1)).expand([self.n_obs, 1, 1]))  # (self.n_obs, 1)
        
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

        # =====================Gene-specific overdispersion ======================= #
        alpha_g_phi_hyp = pyro.sample(
            "alpha_g_phi_hyp",
            dist.Gamma(self.alpha_g_phi_hyp_prior_alpha, self.alpha_g_phi_hyp_prior_beta),
        )
        alpha_g_inverse = pyro.sample(
            "alpha_g_inverse",
            dist.Exponential(alpha_g_phi_hyp).expand([self.n_vars,1]).to_event(2),
        )

        # =====================Expected expression ======================= #
        # overdispersion
        alpha = self.ones / alpha_g_inverse.pow(2)
        # biological expression
        mu = (mu_RNAvelocity + (obs2sample @ s_g_gene_add).unsqueeze(dim=-1)  # contaminating RNA
        ) * detection_y_c  # cell-specific normalisation
        
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
