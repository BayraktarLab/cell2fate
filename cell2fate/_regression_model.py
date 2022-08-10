from typing import List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from cell2fate._pyro_mixin import (
    AutoGuideMixinModule,
    PltExportMixin,
    PyroAggressiveTrainingPlan,
    QuantileMixin,
    init_to_value,
)
from pyro import clear_param_store
from scipy.sparse import csr_matrix
from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.dataloaders import DataSplitter, DeviceBackedDataSplitter
from scvi.model.base import BaseModelClass, PyroSampleMixin, PyroSviTrainMixin
from scvi.model.base._pyromixin import PyroJitGuideWarmup
from scvi.module.base import PyroBaseModuleClass
from scvi.train import TrainRunner
from scvi.utils import setup_anndata_dsp

def compute_cluster_summary(adata, labels, use_raw=True, layer=None, summary="mean"):
    """
    Compute average expression of each gene in each cluster

    Parameters
    ----------
    adata
        AnnData object of reference single-cell dataset
    labels
        Name of adata.obs column containing cluster labels
    use_raw
        Use raw slow in adata?
    layer
        use layer in adata? provide layer name

    Returns
    -------
    pd.DataFrame of cluster average expression of each gene

    """

    if layer is not None:
        x = adata.layers[layer]
        var_names = adata.var_names
    else:
        if not use_raw:
            x = adata.X
            var_names = adata.var_names
        else:
            if not adata.raw:
                raise ValueError(
                    "AnnData object has no raw data, change `use_raw=True, layer=None` or fix your object"
                )
            x = adata.raw.X
            var_names = adata.raw.var_names

    if sum(adata.obs.columns == labels) != 1:
        raise ValueError("cluster_col is absent in adata_ref.obs or not unique")

    all_clusters = np.unique(adata.obs[labels])
    summary_mat = np.zeros((1, x.shape[1]))

    for c in all_clusters:
        sparse_subset = csr_matrix(x[np.isin(adata.obs[labels], c), :])
        if summary == "mean":
            summ = sparse_subset.mean(0)
        elif summary == "sum":
            summ = sparse_subset.sum(0)
        elif summary == "var":
            sparse_subset_copy = sparse_subset.copy()
            sparse_subset_copy.data = np.power(sparse_subset_copy.data, 2)
            summ = sparse_subset_copy.mean(0) - np.power(sparse_subset.mean(0), 2)
            # summ = np.var(np.array(sparse_subset.toarray()), axis=0).reshape((1, x.shape[1]))
        summary_mat = np.concatenate((summary_mat, summ))
    summary_mat = summary_mat[1:, :].T
    summary_df = pd.DataFrame(data=summary_mat, index=var_names, columns=all_clusters)

    return summary_df


class RegressionBaseModule(PyroBaseModuleClass, AutoGuideMixinModule):
    def __init__(
        self,
        model,
        amortised: bool = False,
        encoder_mode: Literal["single", "multiple", "single-multiple"] = "single",
        encoder_kwargs: Optional[dict] = None,
        guide_kwargs: Optional[dict] = None,
        data_transform=None,
        create_autoguide_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Module class which defines AutoGuide given model. Supports multiple model architectures.

        Parameters
        ----------
        amortised
            boolean, use a Neural Network to approximate posterior distribution of location-specific (local) parameters?
        encoder_mode
            Use single encoder for all variables ("single"), one encoder per variable ("multiple")
            or a single encoder in the first step and multiple encoders in the second step ("single-multiple").
        encoder_kwargs
            arguments for Neural Network construction (scvi.nn.FCLayers)
        kwargs
            arguments for specific model class - e.g. number of genes, values of the prior distribution
        """
        super().__init__()
        self.hist = []

        self._model = model(**kwargs)
        self._amortised = amortised
        if create_autoguide_kwargs is None:
            create_autoguide_kwargs = dict()
        if encoder_kwargs is None:
            encoder_kwargs = dict()
        if guide_kwargs is None:
            guide_kwargs = dict()

        n_cat_list = [kwargs["n_batch"]]
        if "n_extra_categoricals" in kwargs.keys():
            n_cat_list = n_cat_list + kwargs["n_extra_categoricals"]

        encoder_instance = (
            encoder_kwargs["encoder_instance"]
            if "encoder_instance" in encoder_kwargs
            else None
        )

        self._guide = self._create_autoguide(
            model=self.model,
            amortised=self.is_amortised,
            encoder_kwargs=encoder_kwargs,
            data_transform=data_transform,
            encoder_mode=encoder_mode,
            init_loc_fn=self.init_to_value,
            n_cat_list=n_cat_list,
            encoder_instance=encoder_instance,
            guide_kwargs=guide_kwargs,
            **create_autoguide_kwargs,
        )

        self._get_fn_args_from_batch = self._model._get_fn_args_from_batch

    @property
    def model(self):
        return self._model

    @property
    def guide(self):
        return self._guide

    @property
    def is_amortised(self):
        return self._amortised

    @property
    def list_obs_plate_vars(self):
        return self.model.list_obs_plate_vars()

    def init_to_value(self, site):

        if getattr(self.model, "np_init_vals", None) is not None:
            init_vals = {
                k: getattr(self.model, f"init_val_{k}")
                for k in self.model.np_init_vals.keys()
            }
        else:
            init_vals = dict()
        return init_to_value(site=site, values=init_vals)
