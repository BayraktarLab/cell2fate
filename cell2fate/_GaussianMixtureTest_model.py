from typing import List, Optional
from datetime import date
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from pyro import clear_param_store
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.model.base import BaseModelClass, PyroSampleMixin, PyroSviTrainMixin
from scvi.utils import setup_anndata_dsp

from cell2fate._pyro_base_GaussianMixture_module import GaussianMixtureBaseModule
from cell2location.models.base._pyro_mixin import PltExportMixin, QuantileMixin
from cell2fate._GaussianMixtureTest_module import GaussianMixtureModel

from scvi.train import PyroTrainingPlan
from pyro.infer import TraceEnum_ELBO, Trace_ELBO, infer_discrete
from pyro import poutine
import torch

class GaussianMixture(QuantileMixin, PyroSampleMixin, PyroSviTrainMixin, PltExportMixin, BaseModelClass):
    """
    GaussianMixture with two components. User-end model class.

    Parameters
    ----------
    adata
        single-cell AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains spliced and unspliced counts in adata.layers['spliced'], adata.layers['unspliced']
    """

    def __init__(
        self,
        adata: AnnData,
        model_class=None,
        **model_kwargs,
    ):
        # in case any other model was created before that shares the same parameter names.
        clear_param_store()

        super().__init__(adata)

        if model_class is None:
            model_class = GaussianMixtureModel
        
        model_kwargs["init_vals"] = {'locs': torch.tensor(np.array((10., 1.)), dtype = torch.float32)}
        
        self.module = GaussianMixtureBaseModule(
            model=model_class,
            n_obs=self.summary_stats["n_cells"],
            n_vars=self.summary_stats["n_vars"],
            n_batch=self.summary_stats["n_batch"],
            **model_kwargs,
        )
        self._model_summary_string = f'Gaussian Mixture model with the following params: \nn_batch: {self.summary_stats["n_batch"]} '
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        **kwargs,
    ):
        """
        %(summary)s.

        Parameters
        ----------
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        adata.obs["_indices"] = np.arange(adata.n_obs).astype("int64")
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs: int = 30000,
        batch_size: int = None,
        train_size: float = 1,
        lr: float = 0.002,
        num_particles: int = 1,
        scale_elbo: float = 1.0,
        **kwargs,
    ):
        """Train the model with useful defaults

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        train_size
            Size of training set in the range [0.0, 1.0].
        batch_size
            Minibatch size to use during training. If `None`, no minibatching occurs and all
            data is copied to device (e.g., GPU).
        lr
            Optimiser learning rate (default optimiser is :class:`~pyro.optim.ClippedAdam`).
            Specifying optimiser via plan_kwargs overrides this choice of lr.
        kwargs
            Other arguments to scvi.model.base.PyroSviTrainMixin().train() method
        """

        kwargs["max_epochs"] = max_epochs
        kwargs["batch_size"] = batch_size
        kwargs["train_size"] = train_size
        kwargs["lr"] = lr

        if "plan_kwargs" not in kwargs.keys():
            kwargs["plan_kwargs"] = dict()
        if getattr(self.module.model, "discrete_variables", None) and (len(self.module.model.discrete_variables) > 0):
            kwargs["plan_kwargs"]["loss_fn"] = TraceEnum_ELBO(num_particles=num_particles)
        else:
            kwargs["plan_kwargs"]["loss_fn"] = Trace_ELBO(num_particles=num_particles)
        if scale_elbo != 1.0:
            if scale_elbo is None:
                scale_elbo = 1.0 / (self.summary_stats["n_cells"] * self.summary_stats["n_genes"])
            kwargs["plan_kwargs"]["scale_elbo"] = scale_elbo

        super().train(**kwargs)      
        
    def _export2adata(self, samples):
        r"""
        Export key model variables and samples

        Parameters
        ----------
        samples
            dictionary with posterior mean, 5%/95% quantiles, SD, samples, generated by ``.sample_posterior()``

        Returns
        -------
            Updated dictionary with additional details is saved to ``adata.uns['mod']``.
        """
        # add factor filter and samples of all parameters to unstructured data
        results = {
            "model_name": str(self.module.__class__.__name__),
            "date": str(date.today()),
            "var_names": self.adata.var_names.tolist(),
            "obs_names": self.adata.obs_names.tolist(),
            "post_sample_means": samples["post_sample_means"],
            "post_sample_stds": samples["post_sample_stds"],
            "post_sample_q05": samples["post_sample_q05"],
            "post_sample_q95": samples["post_sample_q95"],
        }

        return results

    def export_posterior(
        self,
        adata,
        export_slot = 'mod',
        sample_kwargs: Optional[dict] = None):
        """
        Summarises posterior distribution and exports results to anndata object. 
        Parameters
        ----------
        adata
            anndata object where results should be saved
        """

        sample_kwargs = sample_kwargs if isinstance(sample_kwargs, dict) else dict()

        # generate samples from posterior distributions for all parameters
        # and compute mean, 5%/95% quantiles and standard deviation
        self.samples = self.sample_posterior(**sample_kwargs)

        # export posterior distribution summary for all parameters and
        # annotation (model, date, var, obs and cell type names) to anndata object
        adata.uns[export_slot] = self._export2adata(self.samples)
        
        # Predict components:
        data = torch.tensor(self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY))
        idx = torch.tensor(self.adata_manager.get_from_registry(REGISTRY_KEYS.INDICES_KEY))
        batch_key = torch.tensor(self.adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY))
        
        serving_model = infer_discrete(self.module.model, first_available_dim=-3)
        k_unknown = serving_model(data, idx, batch_key)  # takes the same args as model()

#         guide_trace = poutine.trace(self.module.guide).get_trace(data, idx, batch_key)  # record the globals
#         trained_model = poutine.replay(self.module.model, trace=guide_trace)  # replay the globals
#         def classifier(data, idx, batch_key, temperature=0):
#             inferred_model = infer_discrete(trained_model, temperature=temperature,
#                                             first_available_dim=-2)  # avoid conflict with data plate
#             trace = poutine.trace(inferred_model).get_trace(data, idx, batch_key)
#             return trace.nodes[self.module.discrete_parameters]["value"]
#         adata.obs['k_c'] = classifier(data, idx, batch_key)

        return k_unknown,data, idx, batch_key
#         return classifier(data, idx, batch_key)