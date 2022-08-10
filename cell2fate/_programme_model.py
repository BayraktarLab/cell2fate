import warnings
from inspect import signature
from typing import List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from anndata import AnnData
from cell2fate._pyro_mixin import (
    PltExportMixin,
    PyroAggressiveConvergence,
    PyroAggressiveTrainingPlan,
    QuantileMixin,
)
from pyro import clear_param_store
from pyro.infer import Trace_ELBO, TraceEnum_ELBO
from pyro.infer.autoguide import AutoHierarchicalNormalMessenger
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
)
from scvi.dataloaders import DataSplitter, DeviceBackedDataSplitter
from scvi.model.base import BaseModelClass, PyroSampleMixin, PyroSviTrainMixin
from scvi.model.base._pyromixin import PyroJitGuideWarmup
from scvi.train import PyroTrainingPlan, TrainRunner
from scvi.utils import setup_anndata_dsp
from scvi.nn import one_hot
from cell2fate._regression_model import RegressionBaseModule
from cell2fate._mean_programme_module import MeanProgrammePyroModel

class ProgrammeModel(
    QuantileMixin,
    PyroSampleMixin,
    PyroSviTrainMixin,
    PltExportMixin,
    BaseModelClass,
):
    """
    Regulatory programme model.

    User-end model class.

    Parameters
    ----------
    adata
        single-cell AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    use_gpu
        Use the GPU?
    **model_kwargs
        Keyword args for :class:`~scvi.external.LocationModelLinearDependentWMultiExperimentModel`
    variance_categories
        Categories expected to have differing stochastic/unexplained variance

    Examples
    --------
    TODO add example
    >>>
    """

    use_max_count_cluster_as_initial = False

    def __init__(
        self,
        adata: AnnData,
        model_class=None,
        n_factors: int = 300,
        use_moments_as_initial: bool = False,
        use_moments_as_prior: bool = False,
        gene_cluster_init: Optional[str] = None,
        variance_categories: Optional[str] = None,
        rna_index: Optional[str] = None,
        factor_names: Optional[list] = None,
        **model_kwargs,
    ):
        # in case any other model was created before that shares the same parameter names.
        clear_param_store()

        super().__init__(adata)

        self.mi_ = []
        self.minibatch_genes_ = False

        if model_class is None:
            model_class = MeanProgrammePyroModel

        # create factor names
        self.n_factors_ = n_factors
        if factor_names is None:
            self.factor_names_ = np.array([f"factor_{i}" for i in range(n_factors)])
        else:
            self.factor_names_ = factor_names

        if "create_autoguide_kwargs" not in model_kwargs.keys():
            model_kwargs["create_autoguide_kwargs"] = {
                "guide_class": AutoHierarchicalNormalMessenger,
            }
        elif "guide_class" not in model_kwargs["create_autoguide_kwargs"].keys():
            model_kwargs["create_autoguide_kwargs"][
                "guide_class"
            ] = AutoHierarchicalNormalMessenger

        # annotations for extra categorical covariates
        if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry:
            self.extra_categoricals_ = self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            )
            self.n_extra_categoricals_ = self.extra_categoricals_.n_cats_per_key
            model_kwargs["n_extra_categoricals"] = self.n_extra_categoricals_

            model_kwargs["n_obs"] = self.summary_stats["n_cells"]
            model_kwargs["n_vars"] = self.summary_stats["n_vars"]
            model_kwargs["n_batch"] = self.summary_stats["n_batch"]

        self.module = RegressionBaseModule(
            model=model_class,
            n_factors=self.n_factors_,
            **model_kwargs,
        )

        self._model_summary_string = f"factorization model with the following params: \nn_factors: {self.n_factors_}"
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        transpose: bool = False,
        layer: Optional[str] = None,
        spliced_key = 'spliced_raw',
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        rna_index: Optional[str] = None,
        variance_categories: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        gene_bool_key: Optional[str] = None,
        region_motif_coo_key: Optional[str] = None,
        promoter_motif_coo_key: Optional[str] = None,
        gene_region_coo_key: Optional[str] = None,
        **kwargs,
    ):
        """
        %(summary)s.
        Parameters
        ----------
        %(param_adata)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_layer)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        %(param_copy)s
        Returns
        -------
        %(returns)s
        """

        setup_method_args = cls._get_setup_method_args(**locals())

        if transpose:
            # transpose anndata for minibatching genes
            adata_nontransposed = adata.T.copy()
            # if index for cells that have RNA measurements does not exist assume all cells have RNA
            if rna_index is None:
                adata_nontransposed.obs["_rna_index"] = True
                rna_index = "_rna_index"

            # using annotations of genes and regions derive plate indices ('local_indices')
            gene_bool = adata.obs[gene_bool_key]
            gene_ind = np.where(gene_bool)[0]
            n_genes = len(gene_ind)
            region_ind = np.where(np.logical_not(gene_bool))[0]
            n_regions = adata.n_obs - n_genes
            local_index_series = pd.Series(index=np.arange(adata.n_obs))
            local_index_series.loc[gene_ind] = np.arange(n_genes)
            local_index_series.loc[region_ind] = np.arange(n_regions)
            adata.obs["local_indices"] = local_index_series.values.astype(int).flatten()

            # add index for each gene (provided to pyro plate for correct minibatching)
            adata.obs["_indices"] = np.arange(adata.n_obs).astype("int64")

            anndata_fields = [
                LayerField(spliced_key, 'spliced', is_count_data=True),
                NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
                NumericalObsField("gene_bool", gene_bool_key),
                NumericalObsField("local_indices", "local_indices"),
            ]
            if gene_region_coo_key is not None:
                anndata_fields.append(ObsmField("gene_region_coo", gene_region_coo_key))
            if region_motif_coo_key is not None:
                anndata_fields.append(
                    ObsmField("region_motif_coo", region_motif_coo_key)
                )
            if promoter_motif_coo_key is not None:
                anndata_fields.append(
                    ObsmField("promoter_motif_coo", promoter_motif_coo_key)
                )
            # generate cell-specific fixed input ##########
            # add index for each cell (provided to pyro plate for correct minibatching)
            adata_nontransposed.obs["_indices"] = np.arange(
                adata_nontransposed.n_obs
            ).astype("int64")

            obs_anndata_fields = [
                LayerField('spliced', spliced_key, is_count_data=True),
                CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
                CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
                CategoricalJointObsField(
                    REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
                ),
                NumericalJointObsField(
                    REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
                ),
                NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
                NumericalObsField("rna_index", rna_index),
            ]

            # annotations for covariates that affect stochastic variance
            if variance_categories is not None:
                obs_anndata_fields.append(
                    CategoricalObsField("var_categoricals", variance_categories)
                )
            obs_adata_manager = AnnDataManager(
                fields=obs_anndata_fields, setup_method_args=setup_method_args
            )
            obs_adata_manager.register_fields(adata_nontransposed, **kwargs)
            cell_specific_vals = {
                k: obs_adata_manager.get_from_registry(k)
                for k in obs_adata_manager.data_registry.keys()
                if k != REGISTRY_KEYS.X_KEY
            }
            cell_specific_state_registry = {
                k: obs_adata_manager.get_state_registry(k)
                for k in obs_adata_manager.data_registry.keys()
                if k != REGISTRY_KEYS.X_KEY
            }
            del obs_adata_manager, obs_anndata_fields, adata_nontransposed

            adata_manager = AnnDataManager(
                fields=anndata_fields, setup_method_args=setup_method_args
            )
            adata_manager.register_fields(adata, **kwargs)
            cls.register_manager(adata_manager)
            return cell_specific_vals, cell_specific_state_registry
        else:
            # add index for each cell (provided to pyro plate for correct minibatching)
            adata.obs["_indices"] = np.arange(adata.n_obs).astype("int64")

            # if index for cells that have RNA measurements does not exist assume all cells have RNA
            if rna_index is None:
                adata.obs["_rna_index"] = True
                rna_index = "_rna_index"

            anndata_fields = [
                LayerField('spliced', spliced_key, is_count_data=True),
                CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
                CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
                CategoricalJointObsField(
                    REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
                ),
                NumericalJointObsField(
                    REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
                ),
                NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
                NumericalObsField("rna_index", rna_index),
            ]

            # annotations for covariates that affect stochastic variance
            if variance_categories is not None:
                anndata_fields.append(
                    CategoricalObsField("var_categoricals", variance_categories)
                )

            adata_manager = AnnDataManager(
                fields=anndata_fields, setup_method_args=setup_method_args
            )
            adata_manager.register_fields(adata, **kwargs)
            cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs: int = 50000,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 1,
        validation_size: Optional[float] = None,
        batch_size: int = 3,
        early_stopping: bool = False,
        lr: float = 0.001,
        num_particles: int = 1,
        scale_elbo: float = 1.0,
        training_plan: PyroTrainingPlan = PyroTrainingPlan,
        plan_kwargs: Optional[dict] = None,
        dl_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """
        Train the model.
        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training. If `None`, no minibatching occurs and all
            data is copied to device (e.g., GPU).
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        lr
            Optimiser learning rate (default optimiser is :class:`~pyro.optim.ClippedAdam`).
            Specifying optimiser via plan_kwargs overrides this choice of lr.
        training_plan
            Training plan :class:`~scvi.train.PyroTrainingPlan`.
        plan_kwargs
            Keyword args for :class:`~scvi.train.PyroTrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """

        if self.is_trained_:
            # if the model is already trained don't initialise with initial values
            self.module.model.np_init_vals = None

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()
        if getattr(self.module.model, "discrete_variables", None) and (
            len(self.module.model.discrete_variables) > 0
        ):
            plan_kwargs["loss_fn"] = TraceEnum_ELBO(num_particles=num_particles)
        else:
            plan_kwargs["loss_fn"] = Trace_ELBO(num_particles=num_particles)
        if scale_elbo != 1.0:
            if scale_elbo is None:
                scale_elbo = 1.0 / self.summary_stats["n_cells"]
            plan_kwargs["scale_elbo"] = scale_elbo
        if lr is not None and "optim" not in plan_kwargs.keys():
            plan_kwargs.update({"optim_kwargs": {"lr": lr}})
        if max_epochs is None:
            n_obs = self.adata_manager.adata.n_obs
            max_epochs = np.min([round((20000 / n_obs) * 1000), 1000])

        if dl_kwargs is None:
            dl_kwargs = dict()

        if batch_size is None:
            # use data splitter which moves data to GPU once
            data_splitter = DeviceBackedDataSplitter(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                use_gpu=use_gpu,
                **dl_kwargs,
            )
        else:
            data_splitter = DataSplitter(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                use_gpu=use_gpu,
                **dl_kwargs,
            )
        training_plan = training_plan(pyro_module=self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )

        if "callbacks" not in trainer_kwargs.keys():
            trainer_kwargs["callbacks"] = []
        trainer_kwargs["callbacks"].append(PyroJitGuideWarmup())

        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        return runner()

    def train_aggressive(
        self,
        max_epochs: Optional[int] = 1000,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 1,
        validation_size: Optional[float] = None,
        batch_size: int = None,
        early_stopping: bool = False,
        lr: Optional[float] = None,
        scale_elbo: float = 1.0,
        plan_kwargs: Optional[dict] = None,
        aggressive_kwargs: Optional[dict] = None,
        dl_kwargs: Optional[dict] = None,
        ignore_warnings=False,
        **trainer_kwargs,
    ):
        """
        Train the model.
        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training. If `None`, no minibatching occurs and all
            data is copied to device (e.g., GPU).
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        lr
            Optimiser learning rate (default optimiser is :class:`~pyro.optim.ClippedAdam`).
            Specifying optimiser via plan_kwargs overrides this choice of lr.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        aggressive_kwargs
            Keyword args for :class:`~scvi.train.PyroAggressiveConvergence`.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        if max_epochs is None:
            n_obs = self.adata_manager.adata.n_obs
            max_epochs = np.min([round((20000 / n_obs) * 1000), 1000])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()
        aggressive_kwargs = (
            aggressive_kwargs if isinstance(aggressive_kwargs, dict) else dict()
        )
        if lr is not None and "optim" not in plan_kwargs.keys():
            plan_kwargs.update({"optim_kwargs": {"lr": lr}})
        if scale_elbo != 1.0:
            if scale_elbo is None:
                scale_elbo = 1.0 / self.summary_stats["n_cells"]
            plan_kwargs["scale_elbo"] = scale_elbo

        if dl_kwargs is None:
            dl_kwargs = dict()

        if batch_size is None:
            # use data splitter which moves data to GPU once
            data_splitter = DeviceBackedDataSplitter(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                use_gpu=use_gpu,
                **dl_kwargs,
            )
        else:
            data_splitter = DataSplitter(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                use_gpu=use_gpu,
                **dl_kwargs,
            )
        training_plan = PyroAggressiveTrainingPlan(
            pyro_module=self.module, **plan_kwargs
        )

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )

        if "callbacks" not in trainer_kwargs.keys():
            trainer_kwargs["callbacks"] = []
        trainer_kwargs["callbacks"].append(PyroJitGuideWarmup())
        trainer_kwargs["callbacks"].append(
            PyroAggressiveConvergence(**aggressive_kwargs)
        )

        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        with warnings.catch_warnings():
            if ignore_warnings:
                warnings.simplefilter("ignore")
            res = runner()
        self.mi_ = self.mi_ + training_plan.mi
        return res

    def hide_expose_model_part(
        self,
        name: str,
        vars_status: str,
    ):
        """
        Hide or expose part of the model to prevent training of desired parameters.

        Parameters
        ----------
        vars_status
            "hide" or "expose" parameters?
        name
            Name of the model part which needs to correspond to self.module.model attribute
            or property listing all guide and model variables that need to

        Returns
        -------

        """
        if vars_status not in ["hide", "expose"]:
            raise ValueError('vars_status must be one of "hide", "expose"')
        # get a list of variables that need to be frozen
        vars_list = getattr(self.module.model, name)
        # set tensor.requires_grad = False or true depending on input
        self.change_requires_grad_guide(vars_list, vars_status=vars_status)
        self.change_requires_grad_model(vars_list, vars_status=vars_status)

    def change_requires_grad_guide(self, vars_list, vars_status):

        for k, v in self.module.guide.named_parameters():
            k_in_vars = np.any([i in k for i in vars_list])
            # hide variables on the list if they are not hidden
            if k_in_vars and v.requires_grad and (vars_status == "hide"):
                v.requires_grad = False
            # expose variables on the list if they are hidden
            if k_in_vars and (not v.requires_grad) and (vars_status == "expose"):
                v.requires_grad = True

    def change_requires_grad_model(self, vars_list, vars_status):

        for k, v in self.module.model.named_parameters():
            k_in_vars = np.any([i in k for i in vars_list])
            # hide variables on the list if they are not hidden
            if k_in_vars and v.requires_grad and (vars_status == "hide"):
                v.requires_grad = False
            # expose variables on the list if they are hidden
            if k_in_vars and (not v.requires_grad) and (vars_status == "expose"):
                v.requires_grad = True

    def _compute_cluster_summary(self, key=REGISTRY_KEYS.LABELS_KEY, summary="mean"):
        """
        Compute average per cluster (key=REGISTRY_KEYS.LABELS_KEY) or per batch (key=REGISTRY_KEYS.BATCH_KEY).

        Returns
        -------
        pd.DataFrame with variables in rows and labels in columns
        """
        # find cell label column
        label_col = self.adata_manager.get_state_registry(key).original_key

        # find data slot
        x_dict = self.adata_manager.data_registry["X"]
        if x_dict["attr_name"] == "X":
            use_raw = False
        else:
            use_raw = True
        if x_dict["attr_name"] == "layers":
            layer = x_dict["attr_key"]
        else:
            layer = None

        # compute mean expression of each gene in each cluster/batch
        aver = compute_cluster_summary(
            self.adata_manager.adata,
            labels=label_col,
            use_raw=use_raw,
            layer=layer,
            summary=summary,
        )

        return aver

    def export_posterior(
        self,
        adata,
        sample_kwargs: Optional[dict] = None,
        export_slot: str = "mod",
        export_varm_variables: list = ["burst_frequency_fg", "burst_size_fg"],
        export_obsm_variables: list = [
            "frequency_cell_factors_w_cf1",
            "size_cell_factors_v_cf2",
        ],
        add_to_varm: list = ["means", "stds", "q05", "q95"],
        add_to_obsm: list = ["means", "stds", "q05", "q95"],
        factor_names_keys: Optional[dict] = None,
    ):
        """
        Summarise posterior distribution and export results to anndata object:
        1. adata.obsm: Selected variables as pd.DataFrames for each posterior distribution summary `add_to_varm`,
            posterior mean, sd, 5% and 95% quantiles (['means', 'stds', 'q05', 'q95']).
            If export to adata.varm fails with error, results are saved to adata.var instead.
        2. adata.uns: Posterior of all parameters, model name, date,
            cell type names ('factor_names'), obs and var names.

        Parameters
        ----------
        adata
            anndata object where results should be saved
        sample_kwargs
            arguments for self.sample_posterior (generating and summarising posterior samples), namely:
                num_samples - number of samples to use (Default = 1000).
                batch_size - data batch size (keep low enough to fit on GPU, default 2048).
                use_gpu - use gpu for generating samples?
        export_slot
            adata.uns slot where to export results
        export_varm_variables
            variable/site names to export in varm
        export_obsm_variables
            variable/site names to export in obsm
        add_to_varm
            posterior distribution summary to export in adata.varm (['means', 'stds', 'q05', 'q95']).
        add_to_obsm
            posterior distribution summary to export in adata.obsm (['means', 'stds', 'q05', 'q95']).
        factor_names_keys
            if multiple factor names are present in `model.factor_names_`
            - a dictionary defining the correspondence between `export_varm_variables`/`export_obsm_variables` and
            `model.factor_names_` must be provided
        Returns
        -------

        """

        sample_kwargs = sample_kwargs if isinstance(sample_kwargs, dict) else dict()
        factor_names_keys = (
            factor_names_keys if isinstance(factor_names_keys, dict) else dict()
        )

        # generate samples from posterior distributions for all parameters
        # and compute mean, 5%/95% quantiles and standard deviation
        self.samples = self.sample_posterior(**sample_kwargs)

        # export posterior distribution summary for all parameters and
        # annotation (model, date, var, obs and cell type names) to anndata object
        adata.uns[export_slot] = self._export2adata(self.samples)

        # export estimated gene loadings
        # data frames contain mean, 5%/95% quantiles and standard deviation, denoted by a prefix

        for var in export_varm_variables:
            for k in add_to_varm:
                if type(self.factor_names_) is dict:
                    factor_names_key = factor_names_keys[var]
                else:
                    factor_names_key = ""
                sample_df = self.sample2df_vars(
                    self.samples,
                    site_name=var,
                    summary_name=k,
                    name_prefix="",
                    factor_names_key=factor_names_key,
                )
                try:
                    adata.varm[f"{k}_{var}"] = sample_df.loc[adata.var.index, :]
                except ValueError:
                    # Catching weird error with obsm: `ValueError: value.index does not match parent’s axis 1 names`
                    adata.var[sample_df.columns] = sample_df.loc[adata.var.index, :]

        # add estimated per cell regulatory programme activities as dataframe to obsm in anndata
        # data frames contain mean, 5%/95% quantiles and standard deviation, denoted by a prefix
        for var in export_obsm_variables:
            for k in add_to_obsm:
                if type(self.factor_names_) is dict:
                    factor_names_key = factor_names_keys[var]
                else:
                    factor_names_key = ""
                sample_df = self.sample2df_obs(
                    self.samples,
                    site_name=var,
                    summary_name=k,
                    name_prefix="",
                    factor_names_key=factor_names_key,
                )
                try:
                    adata.obsm[f"{k}_{var}"] = sample_df.loc[adata.obs.index, :]
                except ValueError:
                    # Catching weird error with obsm: `ValueError: value.index does not match parent’s axis 1 names`
                    adata.obs[sample_df.columns] = sample_df.loc[adata.obs.index, :]

        return adata
    
    def posterior_mu_vs_data(self, adata):
        '''Plots a few UMAP showing fit between posterior and adata'''
        obs2sample = one_hot(torch.tensor(self.adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)), 
                             self.module.model.n_batch)
        obs2extra_categoricals = torch.cat(
            [
                one_hot(
                    torch.tensor(np.array(self.adata_manager.get_from_registry(REGISTRY_KEYS.CAT_COVS_KEY)))[:, i].view(
                        (torch.tensor(np.array(self.adata_manager.get_from_registry(REGISTRY_KEYS.CAT_COVS_KEY))).shape[0], 1)),
                    n_cat,
                )
                for i, n_cat in enumerate(self.module.model.n_extra_categoricals)
            ],
            dim=1,
        )

        mu_biol = torch.tensor(self.samples['post_sample_means']['cell_factors_w_cf']) @ \
        torch.tensor(self.samples['post_sample_means']['g_fg'])
        mu = (
            (mu_biol + obs2sample @ torch.tensor(self.samples['post_sample_means']['s_g_gene_add']))  # contaminating RNA
            * torch.tensor(self.samples['post_sample_means']['detection_y_c'])
            * (obs2extra_categoricals @ torch.tensor(self.samples['post_sample_means']['detection_tech_gene_tg']))
        )
        adata.obs['log10_MSE'] = torch.log10(torch.mean((mu - torch.tensor(self.adata_manager.get_from_registry('spliced').toarray()))**2, axis = 1))
        adata.obs['r^2'] = np.array([torch.corrcoef(torch.stack([mu[i,:], torch.tensor(self.adata_manager.get_from_registry('spliced').toarray())[i,:]], axis = 0))[0,1] for i in range(self.module.model.n_obs)])**2
        adata.obs['Total Counts log10 Abs. Diff.'] = torch.log10(torch.abs(torch.sum(mu, axis = 1) - torch.sum(torch.tensor(self.adata_manager.get_from_registry('spliced').toarray()), axis = 1)))
        adata.obs['Total Counts Diff. Sign'] = torch.sum(mu, axis = 1) - torch.sum(torch.tensor(self.adata_manager.get_from_registry('spliced').toarray()), axis = 1)
        adata.obs['Total Counts Diff. Sign'] = torch.sign(torch.sum(mu, axis = 1) - torch.sum(torch.tensor(self.adata_manager.get_from_registry('spliced').toarray())))
        sc.set_figure_params(figsize=(8, 5))
        sc.pl.umap(adata, color = ['clusters', 'log10_MSE', 'r^2', 'Total Counts log10 Abs. Diff.', 'Total Counts Diff. Sign'],
                   legend_loc = 'on data', size = 300, color_map = 'cividis', ncols= 3)
