import logging
from datetime import date
from functools import partial
from typing import Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pytorch_lightning as pl
import torch
from pyro import poutine
from pyro.infer.autoguide import AutoNormal, init_to_feasible, init_to_mean
from pytorch_lightning.callbacks import Callback
from scipy.sparse import issparse
from scvi import REGISTRY_KEYS
from scvi.dataloaders import AnnDataLoader
from scvi.model._utils import parse_use_gpu_arg
from scvi.module.base import PyroBaseModuleClass
from scvi.train import PyroTrainingPlan
from scvi.utils import track


from cell2fate.AutoAmortisedNormalMessenger import (
    AutoAmortisedHierarchicalNormalMessenger,
)

logger = logging.getLogger(__name__)
       
import sys
from pyro.infer.autoguide.utils import deep_getattr, deep_setattr, helpful_support_errors
from torch.distributions import biject_to, constraints
import pyro.distributions as dist
from pyro.poutine.util import site_is_subsample
from pyro.nn.module import PyroModule, PyroParam, pyro_method
from pyro.infer.autoguide import AutoHierarchicalNormalMessenger
from pyro.poutine.runtime import get_plates
from pyro.distributions.distribution import Distribution
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    Union,
    ValuesView,
)

from scvi.module.base import PyroBaseModuleClass
from scvi.train import PyroTrainingPlan
from typing import Optional, Union
import pyro

max_epochs = 4000
start_lr = 0.01
final_lr = 0.001
lrd = (final_lr/start_lr)**(1/max_epochs)
clipped_adam = pyro.optim.ClippedAdam({"lr": start_lr, "lrd": lrd,  "clip_norm": 10.0})

def init_to_value(site=None, values={}):
    '''
    Initializes the value of a site to a specified value.

    Parameters
    ----------
    site
        The site dictionary containing information about the site. If `None`, returns a partial function.
    values
        A dictionary containing the values to initialize sites with.

    Returns
    -------
    Function or any
        If `site` is None, returns a partial function with the `values` preset. Otherwise, returns the value specified for the site in `values`, or initializes to the mean using `init_to_mean` with `fallback` set to `init_to_feasible` if the site is not in `values`.
    '''

    
    if site is None:
        return partial(init_to_value, values=values)
    if site["name"] in values:
        return values[site["name"]]
    else:
        return init_to_mean(site, fallback=init_to_feasible)


def expand_zeros_along_dim(tensor, size, dim):
    '''
    Expands a tensor with zeros along a specified dimension.

    Parameters
    ----------
    tensor
        The input tensor.
    size
        The size to expand along the specified dimension.
    dim
        The dimension along which to expand the tensor.

    Returns
    -------
    Numpy.ndarray
        A new tensor with zeros expanded along the specified dimension.
    '''
    shape = np.array(tensor.shape)
    shape[dim] = size
    return np.zeros(shape)


def complete_tensor_along_dim(tensor, indices, dim, value, mode="put"):
    
    '''
    Completes a tensor along a specified dimension with given indices and values.

    Parameters
    ----------
    tensor
        The input tensor.
    indices
        The indices to complete along the specified dimension.
    dim
        The dimension along which to complete the tensor.
    value
        The values to insert into the tensor.
    mode
        The mode of completion. "put" for putting values, "take" for taking values. Default is "put".

    Returns
    -------
    Numpy.ndarray
        A new tensor with completed values along the specified dimension.
    '''
    shape = value.shape
    shape = np.ones(len(shape))
    shape[dim] = len(indices)
    shape = shape.astype(int)
    indices = indices.reshape(shape)
    if mode == "take":
        return np.take_along_axis(arr=tensor, indices=indices, axis=dim)
    np.put_along_axis(arr=tensor, indices=indices, values=value, axis=dim)
    return tensor


def _complete_full_tensors_using_plates(
    means_global, means, plate_dict, obs_plate_sites, plate_indices, plate_dim
):
    '''
    Completes full-sized tensors with minibatch values given minibatch indices.

    Parameters
    ----------
    means_global
        Dictionary containing global means.
    means
        Dictionary containing means.
    plate_dict
        Dictionary containing plate information.
    obs_plate_sites
        Dictionary containing observed plate sites.
    plate_indices
        Dictionary containing plate indices.
    plate_dim
        Dictionary containing plate dimensions.

    Returns
    -------
    Dict
        A dictionary with completed global means.
    '''
    
    # complete full sized tensors with minibatch values given minibatch indices
    for k in means_global.keys():
        # find which and how many plates contain this tensor
        plates = [
            plate for plate in plate_dict.keys() if k in obs_plate_sites[plate].keys()
        ]
        if len(plates) == 1:
            # if only one plate contains this tensor, complete it using the plate indices
            means_global[k] = complete_tensor_along_dim(
                means_global[k],
                plate_indices[plates[0]],
                plate_dim[plates[0]],
                means[k],
            )
        elif len(plates) == 2:
            # subset data to index for plate 0 and fill index for plate 1
            means_global_k = complete_tensor_along_dim(
                means_global[k],
                plate_indices[plates[0]],
                plate_dim[plates[0]],
                means[k],
                mode="take",
            )
            means_global_k = complete_tensor_along_dim(
                means_global_k,
                plate_indices[plates[1]],
                plate_dim[plates[1]],
                means[k],
            )
            # fill index for plate 0 in the full data
            means_global[k] = complete_tensor_along_dim(
                means_global[k],
                plate_indices[plates[0]],
                plate_dim[plates[0]],
                means_global_k,
            )
            # TODO add a test - observed variables should be identical if this code works correctly
            # This code works correctly but the test needs to be added eventually
            # np.allclose(
            #     samples['data_chromatin'].squeeze(-1).T,
            #     mod_reg.adata_manager.get_from_registry('X')[
            #         :, ~mod_reg.adata_manager.get_from_registry('gene_bool').ravel()
            #     ].toarray()
            # )
        else:
            NotImplementedError(
                f"Posterior sampling/mean/median/quantile not supported for variables with > 2 plates: {k} has {len(plates)}"
            )
    return means_global



class AutoGuideMixinModule:
    """
    This mixin class provides methods for:

    - initialising standard AutoNormal guides
    - initialising amortised guides (AutoNormalEncoder)
    - initialising amortised guides with special additional inputs

    """

    def _create_autoguide(
        self,
        model,
        amortised,
        encoder_kwargs,
        encoder_mode,
        init_loc_fn=init_to_mean(fallback=init_to_feasible),
        n_cat_list: list = [],
        encoder_instance=None,
        guide_class=AutoNormal,
        guide_kwargs: Optional[dict] = None,
    ):
        if guide_kwargs is None:
            guide_kwargs = dict()

        if not amortised:
            if getattr(model, "discrete_variables", None) is not None:
                model = poutine.block(model, hide=model.discrete_variables)
            if issubclass(guide_class, poutine.messenger.Messenger):
                # messenger guides don't need create_plates function
                _guide = guide_class(
                    model,
                    init_loc_fn=init_loc_fn,
                    **guide_kwargs,
                )
            else:
                _guide = guide_class(
                    model,
                    init_loc_fn=init_loc_fn,
                    **guide_kwargs,
                    create_plates=self.model.create_plates,
                )
        else:
            encoder_kwargs = encoder_kwargs if isinstance(encoder_kwargs, dict) else dict()
            n_hidden = encoder_kwargs["n_hidden"] if "n_hidden" in encoder_kwargs.keys() else 200
            amortised_vars = model.list_obs_plate_vars()
            if len(amortised_vars["input"]) >= 2:
                encoder_kwargs["n_cat_list"] = n_cat_list
            if "n_in" in amortised_vars.keys():
                n_in = amortised_vars["n_in"]
            else:
                n_in = model.n_vars
            if getattr(model, "discrete_variables", None) is not None:
                model = poutine.block(model, hide=model.discrete_variables)
            _guide = AutoAmortisedHierarchicalNormalMessenger(
                model,
                amortised_plate_sites=amortised_vars,
                n_in=n_in,
                n_hidden=n_hidden,
                encoder_kwargs=encoder_kwargs,
                encoder_mode=encoder_mode,
                encoder_instance=encoder_instance,
                init_loc_fn=init_loc_fn,
                **guide_kwargs,
            )
        return _guide

class QuantileMixin:
    """
    This mixin class provides methods for:

    - computing median and quantiles of the posterior distribution using both direct and amortised inference

    """

    def _optim_param(
        self,
        lr: float = 0.01,
        autoencoding_lr: float = None,
        clip_norm: float = 200,
        module_names: list = ["encoder", "hidden2locs", "hidden2scales"],
    ):
        # TODO implement custom training method that can use this function.
        # create function which fetches different lr for autoencoding guide
        def optim_param(module_name, param_name):
            # detect variables in autoencoding guide
            if autoencoding_lr is not None and np.any([n in module_name + "." + param_name for n in module_names]):
                return {
                    "lr": autoencoding_lr,
                    # limit the gradient step from becoming too large
                    "clip_norm": clip_norm,
                }
            else:
                return {
                    "lr": lr,
                    # limit the gradient step from becoming too large
                    "clip_norm": clip_norm,
                }

        return optim_param

    @torch.no_grad()
    
    def _get_obs_plate_sites_v2(
        self,
        args: list,
        kwargs: dict,
        plate_name: str = None,
        return_observed: bool = False,
        return_deterministic: bool = True,
    ):
        """
        Automatically guess which model sites belong to observation/minibatch plate.
        This function requires minibatch plate name specified in `self.module.list_obs_plate_vars["name"]`.
        Parameters
        ----------
        args
            Arguments to the model.
        kwargs
            Keyword arguments to the model.
        return_observed
            Record samples of observed variables.
        Returns
        -------
        Dictionary with keys corresponding to site names and values to plate dimension.
        """
        if plate_name is None:
            plate_name = self.module.list_obs_plate_vars["name"]

        def try_trace(args, kwargs):
            try:
                trace_ = poutine.trace(self.module.guide).get_trace(*args, **kwargs)
                trace_ = poutine.trace(
                    poutine.replay(self.module.model, trace_)
                ).get_trace(*args, **kwargs)
            except ValueError:
                # if sample is unsuccessful try again
                trace_ = try_trace(args, kwargs)
            return trace_

        trace = try_trace(args, kwargs)

        # find plate dimension
        obs_plate = {
            name: {
                fun.name: fun
                for fun in site["cond_indep_stack"]
                if (fun.name in plate_name) or (fun.name == plate_name)
            }
            for name, site in trace.nodes.items()
            if (
                (site["type"] == "sample")  # sample statement
                and (
                    (
                        (not site.get("is_observed", True)) or return_observed
                    )  # don't save observed unless requested
                    or (
                        site.get("infer", False).get("_deterministic", False)
                        and return_deterministic
                    )
                )  # unless it is deterministic
                and not isinstance(
                    site.get("fn", None), poutine.subsample_messenger._Subsample
                )  # don't save plates
            )
            if any(f.name == plate_name for f in site["cond_indep_stack"])
        }

        return obs_plate

    
    
    def _posterior_quantile_minibatch(
        self, q: float = 0.5, batch_size: int = 2048, use_gpu: bool = None, use_median: bool = False
    ):
        """
        Compute median of the posterior distribution of each parameter, separating local (minibatch) variable
        and global variables, which is necessary when performing amortised inference.

        Note for developers: requires model class method which lists observation/minibatch plate
        variables (self.module.model.list_obs_plate_vars()).

        Parameters
        ----------
        q
            quantile to compute
        batch_size
            number of observations per batch
        use_gpu
            Bool, use gpu?
        use_median
            Bool, when q=0.5 use median rather than quantile method of the guide

        Returns
        -------
        dictionary {variable_name: posterior quantile}

        """

        gpus, device = parse_use_gpu_arg(use_gpu)

        self.module.eval()

        train_dl = AnnDataLoader(self.adata_manager, shuffle=False, batch_size=batch_size)

        # sample local parameters
        i = 0
        for tensor_dict in train_dl:

            args, kwargs = self.module._get_fn_args_from_batch(tensor_dict)
            args = [a.to(device) for a in args]
            kwargs = {k: v.to(device) for k, v in kwargs.items()}
            self.to_device(device)

            if i == 0:
                # find plate sites
                obs_plate_sites = self._get_obs_plate_sites(args, kwargs, return_observed=True)
                if len(obs_plate_sites) == 0:
                    # if no local variables - don't sample
                    break
                # find plate dimension
                obs_plate_dim = list(obs_plate_sites.values())[0]
                if use_median and q == 0.5:
                    means = self.module.guide.median(*args, **kwargs)
                else:
                    means = self.module.guide.quantiles([q], *args, **kwargs)
                means = {k: means[k].cpu().numpy() for k in means.keys() if k in obs_plate_sites}

            else:
                if use_median and q == 0.5:
                    means_ = self.module.guide.median(*args, **kwargs)
                else:
                    means_ = self.module.guide.quantiles([q], *args, **kwargs)

                means_ = {k: means_[k].cpu().numpy() for k in means_.keys() if k in obs_plate_sites}
                means = {k: np.concatenate([means[k], means_[k]], axis=obs_plate_dim) for k in means.keys()}
            i += 1
        # sample global parameters
        tensor_dict = next(iter(train_dl))
        args, kwargs = self.module._get_fn_args_from_batch(tensor_dict)
        args = [a.to(device) for a in args]
        kwargs = {k: v.to(device) for k, v in kwargs.items()}
        self.to_device(device)

        if use_median and q == 0.5:
            global_means = self.module.guide.median(*args, **kwargs)
        else:
            global_means = self.module.guide.quantiles([q], *args, **kwargs)
        global_means = {k: global_means[k].cpu().numpy() for k in global_means.keys() if k not in obs_plate_sites}

        for k in global_means.keys():
            means[k] = global_means[k]

        self.module.to(device)

        return means

    @torch.no_grad()
    def _posterior_quantile(
        self, q: float = 0.5, batch_size: int = None, use_gpu: bool = None, use_median: bool = False
    ):
        """
        Compute median of the posterior distribution of each parameter pyro models trained without amortised inference.

        Parameters
        ----------
        q
            Quantile to compute
        use_gpu
            Bool, use gpu?
        use_median
            Bool, when q=0.5 use median rather than quantile method of the guide

        Returns
        -------
        dictionary {variable_name: posterior quantile}

        """

        self.module.eval()
        gpus, device = parse_use_gpu_arg(use_gpu)
        if batch_size is None:
            batch_size = self.adata_manager.adata.n_obs
        train_dl = AnnDataLoader(self.adata_manager, shuffle=False, batch_size=batch_size)
        # sample global parameters
        tensor_dict = next(iter(train_dl))
        args, kwargs = self.module._get_fn_args_from_batch(tensor_dict)
        args = [a.to(device) for a in args]
        kwargs = {k: v.to(device) for k, v in kwargs.items()}
        self.to_device(device)

        if use_median and q == 0.5:
            means = self.module.guide.median(*args, **kwargs)
        else:
            means = self.module.guide.quantiles([q], *args, **kwargs)
        means = {k: means[k].cpu().detach().numpy() for k in means.keys()}

        return means

    def posterior_quantile(
        self, q: float = 0.5, batch_size: int = 2048, use_gpu: bool = None, use_median: bool = False
    ):
        """
        Compute median of the posterior distribution of each parameter.

        Parameters
        ----------
        q
            Quantile to compute
        use_gpu
            Wheter or not use gpu.
        use_median
            When q=0.5 use median rather than quantile method of the guide

        Returns
        -------
        Dict
            A dictionary containing the posterior quantile for each parameter.

        """

        return self._posterior_quantile_minibatch_v2(
            q=q, batch_size=batch_size, use_gpu=use_gpu, use_median=use_median
        )
        
        
        
        
        

    @torch.no_grad()
    def _posterior_quantile_minibatch_v2(
        self,
        q: list = 0.5,
        batch_size: int = 128,
        gene_batch_size: int = 50,
        use_gpu: bool = None,
        use_median: bool = False,
        return_observed: bool = True,
        exclude_vars: list = [],
        data_loader_indices=None,
        show_progress: bool = True,
    ):

        """
        Compute median of the posterior distribution of each parameter, separating local (minibatch) variable
        and global variables, which is necessary when performing amortised inference.

        Note for developers: requires model class method which lists observation/minibatch plate
        variables (self.module.model.list_obs_plate_vars()).

        Parameters
        ----------
        q
            Quantile to compute.
        batch_size
            Number of observations per batch.
        use_gpu
            Wheter or not use gpu.
        use_median
            When q=0.5 use median rather than quantile method of the guide.

        Returns
        -------
        Dict
            A dictionary containing the posterior quantile for each parameter.

        """

        gpus, device = parse_use_gpu_arg(use_gpu)
        self.module.eval()

        train_dl = AnnDataLoader(self.adata_manager, shuffle=False, batch_size=batch_size)

        # sample local parameters
        i = 0
        for tensor_dict in track(
            train_dl,
            style="tqdm",
            description=f"Computing posterior quantile {q}, data batch: ",
            disable=not show_progress,
        ):
            args, kwargs = self.module._get_fn_args_from_batch(tensor_dict)
            args = [a.to(device) for a in args]
            kwargs = {k: v.to(device) for k, v in kwargs.items()}
            self.to_device(device)

            if i == 0:
                minibatch_plate_names = self.module.list_obs_plate_vars["name"]
            
                plates = self.module.model.create_plates(*args, **kwargs)
                if not isinstance(plates, list):
                    plates = [plates]
                # find plate indices & dim
                plate_dict = {
                    plate.name: plate
                    for plate in plates
                    if (
                        (plate.name in minibatch_plate_names)
                        or (plate.name == minibatch_plate_names)
                    )
                }
                
                plate_size = {name: plate.size for name, plate in plate_dict.items()}
                
                if data_loader_indices is not None:
                    # set total plate size to the number of indices in DL not total number of observations
                    # this option is not really used
                    plate_size = {
                        name: len(train_dl.indices)
                        for name, plate in plate_dict.items()
                        if plate.name == minibatch_plate_names
                    }
                plate_dim = {name: plate.dim for name, plate in plate_dict.items()}
                plate_indices = {
                    name: plate.indices.detach().cpu().numpy()
                    for name, plate in plate_dict.items()
                }
                # find plate sites
                obs_plate_sites = {
                    plate: self._get_obs_plate_sites_v2(
                        args, kwargs, plate_name=plate, return_observed=return_observed
                    )
                    for plate in plate_dict.keys()
                }
                if use_median and q == 0.5:
                    # use median rather than quantile method
                    def try_median(args, kwargs):
                        try:
                            means_ = self.module.guide.median(*args, **kwargs)
                        except ValueError:
                            # if sample is unsuccessful try again
                            means_ = try_median(args, kwargs)
                        return means_

                    means = try_median(args, kwargs)
                else:
                    
                    def try_quantiles(args, kwargs):
                        try:
                            means_ = self.module.guide.quantiles([q], *args, **kwargs)
                        except ValueError:
                            # if sample is unsuccessful try again
                            means_ = try_quantiles(args, kwargs)
                        return means_

                    means = try_quantiles(args, kwargs)
                    
                means = {
                    k: means[k].detach().cpu().numpy()
                    for k in means.keys()
                    if k not in exclude_vars
                }
                means_global = means.copy()
                
                for plate in plate_dict.keys():
                    # create full sized tensors according to plate size
                    means_global = {
                        k: (
                            expand_zeros_along_dim(
                                means_global[k], plate_size[plate], plate_dim[plate]
                            )
                            if k in obs_plate_sites[plate].keys()
                            else means_global[k]
                        )
                        for k in means_global.keys()
                    }
                # complete full sized tensors with minibatch values given minibatch indices
                means_global = _complete_full_tensors_using_plates(
                    means_global=means_global,
                    means=means,
                    plate_dict=plate_dict,
                    obs_plate_sites=obs_plate_sites,
                    plate_indices=plate_indices,
                    plate_dim=plate_dim,
                )
                if np.all([len(v) == 0 for v in obs_plate_sites.values()]):
                    # if no local variables - don't sample further - return results now
                    break
            else:
                if use_median and q == 0.5:

                    def try_median(args, kwargs):
                        try:
                            means_ = self.module.guide.median(*args, **kwargs)
                        except ValueError:
                            # if sample is unsuccessful try again
                            means_ = try_median(args, kwargs)
                        return means_

                    means = try_median(args, kwargs)
                else:

                    def try_quantiles(args, kwargs):
                        try:
                            means_ = self.module.guide.quantiles([q], *args, **kwargs)
                        except ValueError:
                            # if sample is unsuccessful try again
                            means_ = try_quantiles(args, kwargs)
                        return means_

                    means = try_quantiles(args, kwargs)
                means = {
                    k: means[k].detach().cpu().numpy()
                    for k in means.keys()
                    if k not in exclude_vars
                }
                # find plate indices & dim
                plates = self.module.model.create_plates(*args, **kwargs)
                if not isinstance(plates, list):
                    plates = [plates]
                plate_dict = {
                    plate.name: plate
                    for plate in plates
                    if (
                        (plate.name in minibatch_plate_names)
                        or (plate.name == minibatch_plate_names)
                    )
                }
               
                plate_indices = {
                    name: plate.indices.detach().cpu().numpy()
                    for name, plate in plate_dict.items()
                }
                # TODO - is this correct to call this function again? find plate sites
                obs_plate_sites = {
                    plate: self._get_obs_plate_sites_v2(
                        args, kwargs, plate_name=plate, return_observed=return_observed
                    )
                    for plate in plate_dict.keys()
                }
                # complete full sized tensors with minibatch values given minibatch indices
                means_global = _complete_full_tensors_using_plates(
                    means_global=means_global,
                    means=means,
                    plate_dict=plate_dict,
                    obs_plate_sites=obs_plate_sites,
                    plate_indices=plate_indices,
                    plate_dim=plate_dim,
                )
            i += 1

        self.module.to(device)

        return means_global



class PltExportMixin:
    r"""
    This mixing class provides methods for common plotting tasks and data export.
    """

    @staticmethod
    def plot_posterior_mu_vs_data(mu, data):
        """
        Plot expected value of the model (e.g. mean of NB distribution) vs observed data.

        Parameters
        ----------
        mu
            Expected value.
        data
            Data value.
        """

        plt.hist2d(
            np.log10(data.flatten() + 1),
            np.log10(mu.flatten() + 1),
            bins=50,
            norm=matplotlib.colors.LogNorm(),
        )
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("Data, log10")
        plt.ylabel("Posterior expected value, log10")
        plt.title("Reconstruction accuracy")
        plt.tight_layout()

    def plot_history(self, iter_start=0, iter_end=-1, ax=None):
        r"""Plot training history

        Parameters
        ----------
        iter_start
            Omit initial iterations from the plot.
        iter_end
            Omit last iterations from the plot.
        ax
            Matplotlib axis.

        """
        if ax is None:
            ax = plt
            ax.set_xlabel = plt.xlabel
            ax.set_ylabel = plt.ylabel
        if iter_end == -1:
            iter_end = len(self.history_["elbo_train"])

        ax.plot(
            self.history_["elbo_train"].index[iter_start:iter_end],
            np.array(self.history_["elbo_train"].values.flatten())[iter_start:iter_end],
            label="train",
        )
        ax.legend()
        ax.xlim(0, len(self.history_["elbo_train"]))
        ax.set_xlabel("Training epochs")
        ax.set_ylabel("-ELBO loss")
        plt.tight_layout()

    def _export2adata(self, samples):
        r"""
        Export key model variables and samples

        Parameters
        ----------
        samples
            dictionary with posterior mean, 5%/95% quantiles, SD, samples, generated by ``.sample_posterior()``.

        Returns
        -------
            Updated dictionary with additional details is saved to ``adata.uns['mod']``.
        """
        # add factor filter and samples of all parameters to unstructured data
        results = {
            "model_name": str(self.module.__class__.__name__),
            "date": str(date.today()),
            "factor_filter": list(getattr(self, "factor_filter", [])),
            "factor_names": list(self.factor_names_),
            "var_names": self.adata.var_names.tolist(),
            "obs_names": self.adata.obs_names.tolist(),
            "post_sample_means": samples["post_sample_means"],
            "post_sample_stds": samples["post_sample_stds"],
            "post_sample_q05": samples["post_sample_q05"],
            "post_sample_q95": samples["post_sample_q95"],
        }
        if type(self.factor_names_) is dict:
            results["factor_names"] = self.factor_names_

        return results

    def sample2df_obs(
        self,
        samples: dict,
        site_name: str = "w_sf",
        summary_name: str = "means",
        name_prefix: str = "cell_abundance",
        factor_names_key: str = "",
    ):
        """Export posterior distribution summary for observation-specific parameters
        (e.g. spatial cell abundance) as Pandas data frame
        (means, 5%/95% quantiles or sd of posterior distribution).

        Parameters
        ----------
        samples
            Dictionary with posterior mean, 5%/95% quantiles, SD, samples, generated by ``.sample_posterior()``.
        site_name
            Name of the model parameter to be exported.
        summary_name
            Posterior distribution summary to return ['means', 'stds', 'q05', 'q95'].
        name_prefix
            Prefix to add to column names (f'{summary_name}{name_prefix}_{site_name}_{self\.factor_names_}').

        Returns
        -------
        Pandas.DataFrame
            Pandas data frame corresponding to either means, 5%/95% quantiles or sd of the posterior distribution.

        """
        if type(self.factor_names_) is dict:
            factor_names_ = self.factor_names_[factor_names_key]
        else:
            factor_names_ = self.factor_names_

        return pd.DataFrame(
            samples[f"post_sample_{summary_name}"].get(site_name, None),
            index=self.adata.obs_names,
            columns=[f"{summary_name}{name_prefix}_{site_name}_{i}" for i in factor_names_],
        )

    def sample2df_vars(
        self,
        samples: dict,
        site_name: str = "gene_factors",
        summary_name: str = "means",
        name_prefix: str = "",
        factor_names_key: str = "",
    ):
        r"""Export posterior distribution summary for variable-specific parameters as Pandas data frame
        (means, 5%/95% quantiles or sd of posterior distribution).

        Parameters
        ----------
        samples
            Dictionary with posterior mean, 5%/95% quantiles, SD, samples, generated by ``.sample_posterior()``.
        site_name
            Name of the model parameter to be exported.
        summary_name
            Posterior distribution summary to return ('means', 'stds', 'q05', 'q95').
        name_prefix
            Prefix to add to column names (f'{summary_name}{name_prefix}_{site_name}_{self\.factor_names_}').

        Returns
        -------
        Pandas.DataFrame
            Pandas data frame corresponding to either means, 5%/95% quantiles or sd of the posterior distribution.

        """
        if type(self.factor_names_) is dict:
            factor_names_ = self.factor_names_[factor_names_key]
        else:
            factor_names_ = self.factor_names_

        return pd.DataFrame(
            samples[f"post_sample_{summary_name}"].get(site_name, None),
            columns=self.adata.var_names,
            index=[f"{summary_name}{name_prefix}_{site_name}_{i}" for i in factor_names_],
        ).T

    def plot_QC(self, summary_name: str = "means", use_n_obs: int = 1000):
        """
        Show quality control plots:

        .. note::
            Reconstruction accuracy to assess if there are any issues with model training.
            The plot should be roughly diagonal, strong deviations signal problems that need to be investigated.
            Plotting is slow because expected value of mRNA count needs to be computed from model parameters. Random
            observations are used to speed up computation.

        Parameters
        ----------
        summary_name
            Posterior distribution summary to use ('means', 'stds', 'q05', 'q95').

        """

        if getattr(self, "samples", False) is False:
            raise RuntimeError("self.samples is missing, please run self.export_posterior() first")
        if use_n_obs is not None:
            ind_x = np.random.choice(
                self.adata_manager.adata.n_obs, np.min((use_n_obs, self.adata.n_obs)), replace=False
            )
        else:
            ind_x = None

        self.expected_nb_param = self.module.model.compute_expected(
            self.samples[f"post_sample_{summary_name}"], self.adata_manager, ind_x=ind_x
        )
        x_data = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)[ind_x, :]
        if issparse(x_data):
            x_data = np.asarray(x_data.toarray())
        self.plot_posterior_mu_vs_data(self.expected_nb_param["mu"], x_data)


class PyroAggressiveConvergence(Callback):
    """
    A callback to compute/apply aggressive training convergence criteria for amortised inference.
    Motivated by this paper: https://arxiv.org/pdf/1901.05534.pdf.
    
    Parameters
    ----------
    dataloader
        An AnnDataLoader object containing the data loader. Default is `None`.
    patience
        The patience for early stopping. Default is 10.
    tolerance
        The tolerance for early stopping. Default is 1e-4.
    """

    def __init__(self, dataloader: AnnDataLoader = None, patience: int = 10, tolerance: float = 1e-4) -> None:
        super().__init__()
        self.dataloader = dataloader
        self.patience = patience
        self.tolerance = tolerance

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused: Optional = None
    ) -> None:
        """
        Compute aggressive training convergence criteria for amortised inference.
        """
        pyro_guide = pl_module.module.guide
        if self.dataloader is None:
            dl = trainer.datamodule.train_dataloader()
        else:
            dl = self.dataloader
        for tensors in dl:
            tens = {k: t.to(pl_module.device) for k, t in tensors.items()}
            args, kwargs = pl_module.module._get_fn_args_from_batch(tens)
            break
        mi_ = pyro_guide.mutual_information(*args, **kwargs)
        mi_ = np.array([v for v in mi_.values()]).sum()
        pl_module.log("MI", mi_, prog_bar=True)
        if len(pl_module.mi) > 1:
            if pl_module.mi[-1] >= (mi_ - self.tolerance):
                pl_module.n_epochs_patience += 1
        else:
            pl_module.n_epochs_patience = 0
        if pl_module.n_epochs_patience > self.patience:
            # stop aggressive training by setting epoch counter to max epochs
            # pl_module.aggressive_epochs_counter = pl_module.n_aggressive_epochs + 1
            logger.info('Stopped aggressive training after "{}" epochs'.format(pl_module.aggressive_epochs_counter))
        pl_module.mi.append(mi_)


class PyroAggressiveTrainingPlan1(PyroTrainingPlan):
    """
    Lightning module task to train Pyro scvi-tools modules.
    Parameters
    ----------
    pyro_module
        An instance of :class:`~scvi.module.base.PyroBaseModuleClass`. This object
        should have callable `model` and `guide` attributes or methods.
    loss_fn
        A Pyro loss. Should be a subclass of :class:`~pyro.infer.ELBO`.
        If `None`, defaults to :class:`~pyro.infer.Trace_ELBO`.
    optim
        A Pyro optimizer instance, e.g., :class:`~pyro.optim.Adam`. If `None`,
        defaults to :class:`pyro.optim.Adam` optimizer with a learning rate of `1e-3`.
    optim_kwargs
        Keyword arguments for **default** optimiser :class:`pyro.optim.Adam`.
    n_aggressive_epochs
        Number of epochs in aggressive optimisation of amortised variables.
    n_aggressive_steps
        Number of steps to spend optimising amortised variables before one step optimising global variables.
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
        Only activated when ``n_epochs_kl_warmup`` is set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from 0 to 1.
        Overrides ``n_steps_kl_warmup`` when both are not `None`.
    """

    def __init__(
        self,
        pyro_module: PyroBaseModuleClass,
        loss_fn: Optional[pyro.infer.ELBO] = None,
        optim: Optional[pyro.optim.PyroOptim] = None,
        optim_kwargs: Optional[dict] = None,
        n_aggressive_epochs: int = 1000,
        n_aggressive_steps: int = 20,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = 400,
        aggressive_vars: Union[list, None] = None,
        invert_aggressive_selection: bool = False,
    ):
        super().__init__(
            pyro_module=pyro_module,
            loss_fn=loss_fn,
            optim=optim,
            optim_kwargs=optim_kwargs,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
        )

        self.n_aggressive_epochs = n_aggressive_epochs
        self.n_aggressive_steps = n_aggressive_steps
        self.aggressive_steps_counter = 0
        self.aggressive_epochs_counter = 0
        self.mi = []
        self.n_epochs_patience = 0

        # in list not provided use amortised variables for aggressive training
        if aggressive_vars is None:
            aggressive_vars = list(self.module.list_obs_plate_vars["sites"].keys())
            aggressive_vars = aggressive_vars + [f"{i}_initial" for i in aggressive_vars]
            aggressive_vars = aggressive_vars + [f"{i}_unconstrained" for i in aggressive_vars]

        self.aggressive_vars = aggressive_vars
        self.invert_aggressive_selection = invert_aggressive_selection

        self.svi = pyro.infer.SVI(
            model=pyro_module.model,
            guide=pyro_module.guide,
            optim=self.optim,
            loss=self.loss_fn,
        )

    def change_requires_grad(self, aggressive_vars_status, non_aggressive_vars_status):
        
        """
        Change the ``requires_grad`` status of the parameters based on provided conditions.

        Parameters
        ----------
        aggressive_vars_status
            The status to set for aggressive variables. Options are "hide" or "expose".
        non_aggressive_vars_status
            The status to set for non-aggressive variables. Options are "hide" or "expose".
        """

        for k, v in self.module.guide.named_parameters():
            k_in_vars = np.any([i in k for i in self.aggressive_vars])
            # hide variables on the list if they are not hidden
            if k_in_vars and v.requires_grad and (aggressive_vars_status == "hide"):
                v.requires_grad = False
            # expose variables on the list if they are hidden
            if k_in_vars and (not v.requires_grad) and (aggressive_vars_status == "expose"):
                v.requires_grad = True

            # hide variables not on the list if they are not hidden
            if (not k_in_vars) and v.requires_grad and (non_aggressive_vars_status == "hide"):
                v.requires_grad = False
            # expose variables not on the list if they are hidden
            if (not k_in_vars) and (not v.requires_grad) and (non_aggressive_vars_status == "expose"):
                v.requires_grad = True

    def training_epoch_end(self, outputs):
        """
        Method called at the end of each training epoch.

        Parameters
        ----------
        outputs
            List of dictionaries containing the output of each training step.
        """

        self.aggressive_epochs_counter += 1

        elbo = 0
        n = 0
        for out in outputs:
            elbo += out["loss"]
            n += 1
        elbo /= n
        self.log("elbo_train", elbo, prog_bar=True)

    def training_step(self, batch, batch_idx):
        """
        Method called for each training batch.

        Parameters
        ----------
        batch
            The batch of data.
        batch_idx
            The index of the current batch.

        Returns
        -------
        dict
            A dictionary containing the loss value for the current batch.
        """
        args, kwargs = self.module._get_fn_args_from_batch(batch)
        # Set KL weight if necessary.
        # Note: if applied, ELBO loss in progress bar is the effective KL annealed loss, not the true ELBO.
        if self.use_kl_weight:
            kwargs.update({"kl_weight": self.kl_weight})

        if self.aggressive_epochs_counter < self.n_aggressive_epochs:
            if self.aggressive_steps_counter < self.n_aggressive_steps:
                self.aggressive_steps_counter += 1
                # Do parameter update exclusively for amortised variables
                if self.invert_aggressive_selection:
                    self.change_requires_grad(
                        aggressive_vars_status="hide",
                        non_aggressive_vars_status="expose",
                    )
                else:
                    self.change_requires_grad(
                        aggressive_vars_status="expose",
                        non_aggressive_vars_status="hide",
                    )
                loss = torch.Tensor([self.svi.step(*args, **kwargs)])
            else:
                self.aggressive_steps_counter = 0
                # Do parameter update exclusively for non-amortised variables
                if self.invert_aggressive_selection:
                    self.change_requires_grad(
                        aggressive_vars_status="expose",
                        non_aggressive_vars_status="hide",
                    )
                else:
                    self.change_requires_grad(
                        aggressive_vars_status="hide",
                        non_aggressive_vars_status="expose",
                    )
                loss = torch.Tensor([self.svi.step(*args, **kwargs)])
        else:
            # Do parameter update for both types of variables
            self.change_requires_grad(
                aggressive_vars_status="expose",
                non_aggressive_vars_status="expose",
            )
            loss = torch.Tensor([self.svi.step(*args, **kwargs)])

        return {"loss": loss}


class PyroAggressiveTrainingPlan(PyroAggressiveTrainingPlan1):
    """
    Lightning module task to train Pyro scvi-tools modules.
    Parameters
    ----------
    pyro_module
        An instance of :class:`~scvi.module.base.PyroBaseModuleClass`. This object
        should have callable `model` and `guide` attributes or methods.
    loss_fn
        A Pyro loss. Should be a subclass of :class:`~pyro.infer.ELBO`.
        If `None`, defaults to :class:`~pyro.infer.Trace_ELBO`.
    optim
        A Pyro optimizer instance, e.g., :class:`~pyro.optim.Adam`. If `None`,
        defaults to :class:`pyro.optim.Adam` optimizer with a learning rate of `1e-3`.
    optim_kwargs
        Keyword arguments for **default** optimiser :class:`pyro.optim.Adam`.
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
        Only activated when``n_epochs_kl_warmup``is set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from 0 to 1.
        Overrides ``n_steps_kl_warmup`` when both are not `None`.
    """

    def __init__(
        self,
        scale_elbo: Union[float, None] = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if scale_elbo != 1.0:
            self.svi = pyro.infer.SVI(
                model=poutine.scale(self.module.model, scale_elbo),
                guide=poutine.scale(self.module.guide, scale_elbo),
                optim=self.optim,
                loss=self.loss_fn,
            )
        else:
            self.svi = pyro.infer.SVI(
                model=self.module.model,
                guide=self.module.guide,
                optim=self.optim,
                loss=self.loss_fn,
            )

class MyAutoHierarchicalNormalMessenger(AutoHierarchicalNormalMessenger):
   

    @pyro_method
    def __call__(self, *args, **kwargs):
        # Since this guide creates parameters lazily, we need to avoid batching
        # those parameters by a particle plate, in case the first time this
        # guide is called is inside a particle plate. We assume all plates
        # outside the model are particle plates.
        self._outer_plates = tuple(f.name for f in get_plates())
            
        try: 
            if self._computing_quantiles==False:
                self._computing_quantiles=False
        except:
            self._computing_quantiles=False

        try:
            return self.call_new(*args, **kwargs)
        finally:
            del self._outer_plates


    def call_new(self, *args, **kwargs) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        """
        Draws posterior samples from the guide and replays the model against
        those samples.

        :returns: A dict mapping sample site name to sample value.
            This includes latent, deterministic, and observed values.
        :rtype: Dict
        """
        self.args_kwargs = args, kwargs
        try:
            with self:
                self.model(*args, **kwargs)
        finally:
            del self.args_kwargs
        model_trace, guide_trace = self.get_traces()

        if self._computing_quantiles:
            with poutine.block():
                model = poutine.condition(self.model, self.quantile_dict)
                trace = poutine.trace(model).get_trace(*args, **kwargs)
                samples = {
                 name: site["value"]
                 for name, site in trace.nodes.items()
                 if site["type"] == "sample"
                 if not site_is_subsample(site)
                 }

            #samples = self.quantile_dict
            #print(samples.keys())

            
            return samples

        else:
            samples = {
                name: site["value"]
                for name, site in model_trace.nodes.items()
                if site["type"] == "sample"
            }

        return samples

    
    def _pyro_sample(self, msg):
            

        if msg["is_observed"] or site_is_subsample(msg):
            return
        prior = msg["fn"]
        msg["infer"]["prior"] = prior
        posterior = self.get_posterior(msg["name"], prior)
        if isinstance(posterior, torch.Tensor):
            posterior = dist.Delta(posterior, event_dim=prior.event_dim)
        if posterior.batch_shape != prior.batch_shape:
            posterior = posterior.expand(prior.batch_shape)
        
        if self._computing_quantiles==True:
            quantiles = self.get_posterior_quantile(msg["name"], prior)
        msg["fn"] = posterior

    def get_posterior_quantile(
        self,
        name: str,
        prior: Distribution,
    ) -> Union[Distribution, torch.Tensor]:
        """
        Get the posterior quantile or median.

        Parameters
        ----------
        name
            The name of the parameter.
        prior
            The prior distribution.

        Returns
        -------
        Union[Distribution, torch.Tensor]
            The posterior quantile or median.
        """

        if self._computing_median:
            return self._get_posterior_median(name, prior)
        if self._computing_quantiles:
            return self._get_posterior_quantiles(name, prior)
        return self.get_posterior_quantile(name, prior)
    
    
    
    def quantiles(self, quantiles, *args, **kwargs):
        """
        Compute quantiles of the posterior distribution.

        Parameters
        ----------
        quantiles
            List of quantiles to compute.
        *args, **kwargs
            Additional arguments and keyword arguments to be passed to the underlying function.

        Returns
        -------
        List
            The result of the computation.
        """

        self._computing_quantiles = True
        self._quantile_values = quantiles
        try:
            return self(*args, **kwargs)
        finally:
            print(f"Sampled for quantile: {quantiles[0]}")

    @torch.no_grad()
    def _get_posterior_quantiles(self, name, prior):
        transform = biject_to(prior.support)
        if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
            loc, scale, weight = self._get_params(name, prior)
            loc = loc + transform.inv(prior.mean) * weight
        else:
            loc, scale = self._get_params(name, prior)

        site_quantiles = torch.tensor(self._quantile_values, dtype=loc.dtype, device=loc.device)
        site_quantiles_values = dist.Normal(loc, scale).icdf(site_quantiles)
        try:
            if self.quantile_dict=={}:
                self.quantile_dict={}
        except:
            self.quantile_dict={}            
        
        self.quantile_dict[name]=transform(site_quantiles_values)
        return transform(site_quantiles_values)


class PyroTrainingPlan_ClippedAdamDecayingRate(PyroTrainingPlan):
    """
    Lightning module task to train Pyro scvi-tools modules.
    Parameters
    ----------
    pyro_module
        An instance of :class:`~scvi.module.base.PyroBaseModuleClass`. This object should have callable `model` and `guide` attributes or methods.
    loss_fn
        A Pyro loss. Should be a subclass of :class:`~pyro.infer.ELBO`. If `None`, defaults to :class:`~pyro.infer.Trace_ELBO`.
    optim
        A Pyro optimizer instance, e.g., :class:`~pyro.optim.Adam`. If `None`, defaults to :class:`pyro.optim.Adam` optimizer with a learning rate of `1e-3`.
    optim_kwargs
        Keyword arguments for **default** optimiser :class:`pyro.optim.Adam`.
    n_aggressive_epochs
        Number of epochs in aggressive optimisation of amortised variables.
    n_aggressive_steps
        Number of steps to spend optimising amortised variables before one step optimising global variables.
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1. Only activated when `n_epochs_kl_warmup` is set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from 0 to 1. Overrides `n_steps_kl_warmup` when both are not `None`.
    """

    def __init__(
        self,
        pyro_module: PyroBaseModuleClass,
        loss_fn: Optional[pyro.infer.ELBO] = None,
        optim: Optional[pyro.optim.PyroOptim] = clipped_adam,
        optim_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            pyro_module=pyro_module,
            loss_fn=loss_fn,
            optim=optim,
            optim_kwargs=optim_kwargs
        )

        self.svi = pyro.infer.SVI(
            model=pyro_module.model,
            guide=pyro_module.guide,
            optim=self.optim,
            loss=self.loss_fn,
        )