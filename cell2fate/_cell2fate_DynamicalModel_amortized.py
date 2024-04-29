from typing import List, Optional
from datetime import date
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
import torch
from pyro import clear_param_store
from scvi.model._utils import parse_use_gpu_arg
from scvi.dataloaders import AnnDataLoader
from scvi.utils import track
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
import pyro.distributions as dist
import scanpy as sc
import contextlib
import io
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from numpy import inner
import scvelo as scv
from scvelo.plotting.velocity_embedding_grid import compute_velocity_on_grid
from ._velocity_embedding_stream import velocity_embedding_stream_modules
import scipy
import gseapy as gp
from cell2fate._pyro_base_cell2fate_module import Cell2FateBaseModule
from cell2fate._pyro_mixin import QuantileMixin
from ._cell2fate_DynamicalModel_amortized_module import \
Cell2fate_DynamicalModel_amortized_module
from ._cell2fate_DynamicalModel import \
Cell2fate_DynamicalModel
from cell2fate.utils import multiplot_from_generator
from cell2fate.utils import mu_mRNA_continousAlpha_globalTime_twoStates
import cell2fate as c2f

class Cell2fate_DynamicalModel_amortized(Cell2fate_DynamicalModel):
    """
    Cell2fate model. User-end model class. See Module class for description of the model.

    Parameters
    ----------
    adata
        single-cell AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains spliced and unspliced counts in adata.layers['spliced'], adata.layers['unspliced']
    **model_kwargs
        Keyword args for :class:`~scvi.external.LocationModelLinearDependentWMultiExperimentModel`

    Examples
    --------
    TODO add example
    >>>
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
            model_class = Cell2fate_DynamicalModel_amortized_module
            
        self.module = Cell2FateBaseModule(
            amortised = True,
            encoder_kwargs={
            "dropout_rate": 0.1,
            "n_hidden": {
                "single": 2*model_kwargs['n_modules'],
                "t_c": 5,
                "detection_y_c": 5,
            },
            "use_batch_norm": False,
            "use_layer_norm": True,
            "n_layers": 1,
            "activation_fn": torch.nn.ELU,
            },
            model=model_class,
            n_obs=self.summary_stats["n_cells"],
            n_vars=self.summary_stats["n_vars"],
            n_batch=self.summary_stats["n_batch"],
            **model_kwargs,
        )
        self._model_summary_string = f'Cell2fate Dynamical Model with the following params: \nn_batch: {self.summary_stats["n_batch"]} '
        self.init_params_ = self._get_init_params(locals())