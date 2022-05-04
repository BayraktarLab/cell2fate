from typing import List, Optional
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule
import pandas as pd
from scvi import REGISTRY_KEYS
from pyro.infer import config_enumerate

class GaussianMixtureModel(PyroModule):
    r"""
    GaussianMixture model with 2 components.
    """

    def __init__(
        self,
        n_obs,
        n_vars,
        n_batch,
        n_extra_categoricals=None,
        n_components = 2,
        scale = 1.,
        loc_mean = 10.,
        loc_sd = 10.,
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
        self.discrete_variables = ['assignment']
        self.n_components = n_components

        if (init_vals is not None) & (type(init_vals) is dict):
            self.np_init_vals = init_vals
            for k in init_vals.keys():
                self.register_buffer(f"init_val_{k}", torch.tensor(init_vals[k]))
        
        self.register_buffer(
            "scale",
            torch.tensor(scale),
        )
        self.register_buffer(
            "loc_mean",
            torch.tensor(loc_mean),
        )               
        self.register_buffer(
            "loc_sd",
            torch.tensor(loc_sd),
        )
            
    ############# Define the model ################
    @staticmethod
    def _get_fn_args_from_batch_no_cat(tensor_dict):
        data = tensor_dict[REGISTRY_KEYS.X_KEY]
        ind_x = tensor_dict["ind_x"].long().squeeze()
        batch_index = tensor_dict[REGISTRY_KEYS.BATCH_KEY]
        return (data, ind_x, batch_index), {}

    @staticmethod
    def _get_fn_args_from_batch_cat(tensor_dict):
        data = tensor_dict[REGISTRY_KEYS.X_KEY]
        ind_x = tensor_dict["ind_x"].long().squeeze()
        batch_index = tensor_dict[REGISTRY_KEYS.BATCH_KEY]
        extra_categoricals = tensor_dict[REGISTRY_KEYS.CAT_COVS_KEY]
        return (data, ind_x, batch_index), {}

    @property
    def _get_fn_args_from_batch(self):
        if self.n_extra_categoricals is not None:
            return self._get_fn_args_from_batch_cat
        else:
            return self._get_fn_args_from_batch_no_cat

    def create_plates(self, data, idx, batch_index):
        return pyro.plate("obs_plate", size=self.n_obs, dim=-2, subsample=idx)

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
    def forward(self, data, idx, batch_index):
        
        obs_plate = self.create_plates(data, idx, batch_index)
        
        # Global variables.
        weights = pyro.sample('weights', dist.Dirichlet(torch.ones(self.n_components)))
        locs = pyro.sample('locs', dist.Normal(self.loc_mean, self.loc_sd).expand([self.n_components]).to_event(1))
        
        # Local variables.
        with obs_plate:
            print('weights.shape', weights.shape)
            print('locs.shape', locs.shape)
            assignment = pyro.sample('assignment', dist.Categorical(weights))
            print('assignment.shape,', assignment.shape)
            print('locs[assignment].shape', locs[assignment].shape)
            print('data.shape', data.shape)
            test = pyro.sample('obs', dist.Normal(locs[assignment], self.scale), obs=data)
            print('test.shape', test.shape)

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
