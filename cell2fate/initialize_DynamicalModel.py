import numpy as np
import torch
import scipy

def initialize_DynamicalModel(adata, NMF_posterior_means, batch_size, cluster_order = None, 
                              beta_prior_mean = 1.0, percentile = 0.01, unspliced_key = 'unspliced',
                              spliced_key = 'spliced', activity_cutoff = 0.1, global_time = False):
    '''Gets initial values for Dynamical Model based on prior NMF run and steady state assumptions.
    Optionally initializes time, based on cluster_order, but this is mostly for testing purposes.'''
    n_cells = NMF_posterior_means["detection_y_c"].shape[0]
    n_genes = NMF_posterior_means["factor_level_g"].shape[1]
    n_modules = NMF_posterior_means["g_fg"].shape[0]
    # Steady State Initialization:
    if scipy.sparse.issparse(adata.layers[unspliced_key]):
        u_counts = adata.layers[unspliced_key].toarray()
    else:
        u_counts = adata.layers[unspliced_key]
    u_top = np.array([np.mean(u_counts[
        np.argsort(u_counts[:,g])[-int(np.ceil(n_cells*percentile)):],g])
     for g in range(n_genes)])
    del u_counts
    if scipy.sparse.issparse(adata.layers[spliced_key]):
        s_counts = adata.layers[spliced_key].toarray()
    else:
        s_counts = adata.layers[spliced_key]
    s_top = np.array([np.mean(s_counts[
        np.argsort(s_counts[:,g])[-int(np.ceil(n_cells*percentile)):],g])
     for g in range(n_genes)])
    del s_counts
    gamma_g = u_top/s_top * beta_prior_mean
    # Initialization based on NMF model:
    us_ratio = np.sum(adata.layers['unspliced'])/np.sum(adata.layers['spliced'])
    init_vals = {"gamma_g": torch.tensor(gamma_g).unsqueeze(0),
                 "factor_level_g": NMF_posterior_means["factor_level_g"],
                 "g_fg" : NMF_posterior_means["g_fg"],
                 "detection_mean_y_e": torch.tensor(NMF_posterior_means["detection_mean_y_e"]),
                 "s_g_gene_add_alpha_hyp" : torch.stack([torch.tensor(NMF_posterior_means["s_g_gene_add_alpha_hyp"]),
                                                         torch.tensor(NMF_posterior_means["s_g_gene_add_alpha_hyp"])], axis = -1),
                 "s_g_gene_add_mean" : torch.stack([us_ratio*torch.tensor(NMF_posterior_means["s_g_gene_add_mean"]),
                                                    torch.tensor(NMF_posterior_means["s_g_gene_add_mean"])], axis = -1),
                 "s_g_gene_add_alpha_e_inv" : torch.stack([torch.tensor(NMF_posterior_means["s_g_gene_add_alpha_e_inv"]),
                                                           torch.tensor(NMF_posterior_means["s_g_gene_add_alpha_e_inv"])], axis = -1),
                 'stochastic_v_ag_hyp': torch.tensor(NMF_posterior_means['stochastic_v_ag_hyp']).repeat([1, 2]),
                 'stochastic_v_ag_inv': torch.tensor(NMF_posterior_means['stochastic_v_ag_inv']).unsqueeze(-1).repeat([1,1,2])
                 }
    # Only initialize cell specific parameters if batch training is not used:
    if not batch_size:
        init_vals["detection_y_c"] =  torch.tensor(NMF_posterior_means["detection_y_c"]).unsqueeze(-1)
    # Optionally initialize time (mostly for testing purposes):
    if cluster_order:
        n_clusters = len(cluster_order)
        time_steps = 1/n_clusters
        t_c = torch.tensor(np.ones(n_cells), dtype = torch.float).unsqueeze(-1).unsqueeze(-1)*0.5
        t_c[adata.obs['clusters'] == cluster_order[0],:,:] = 0*time_steps + time_steps/2
        t_c[adata.obs['clusters'] == cluster_order[-1],:,:] = n_clusters*time_steps-time_steps/2
        if n_clusters > 2:
            for i in range(1,(n_clusters)-1):
                t_c[adata.obs['clusters'] == cluster_order[i],:,:] = i*time_steps + time_steps/2
        adata.obs['t_c_initial'] = np.array(t_c).flatten()
        init_vals['t_c'] = t_c
    # Initialize module activity based on NMF:
    if global_time:
        I_cm = np.zeros((n_cells, n_modules))
        I_cm[:,:] = int(0)
        for_plotting = []
        for m in range(n_modules):
            active = NMF_posterior_means['cell_factors_w_cf'][:,m] > activity_cutoff
            I_cm[active,m] =  int(1)
        init_vals['I_cm'] = I_cm
    return init_vals, adata, n_modules