from typing import Optional
from pyro.infer import SVI, Trace_ELBO
import torch
import numpy as np
import pyro
from pyro.infer import Predictive
import pandas as pd
import scanpy as sc
import random
import scvelo as scv
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from numpy import inner

def compute_velocity_graph_Bergen2020(adata, n_neighbours = None, full_posterior = True):
    """
    Computes a "velocity graph" similar to the method in:
    "Bergen et al. (2020), Generalizing RNA velocity to transient cell states through dynamical modeling"
    
    Parameters
    ----------
    adata
        anndata object with velocity information in adata.layers['velocity'] (expectation value) or 
        adata.uns['velocity_posterior'] (full posterior). Also normalized spliced counts in adata.layers['spliced_norm'].
    n_neighbours
        how many nearest neighbours to consider (all non nearest neighbours have edge weights set to 0)
        if not specified, 10% of the total number of cells is used.
    full_posterior
        whether to use full posterior to compute velocity graph (otherwise expectation value is used)  
    Returns
    -------
    velocity_graph
    """
    M = len(adata.obs_names)
    if not n_neighbours:
        n_neighbours = int(np.round(M*0.1, 0))
    sc.pp.neighbors(adata, n_neighbors = n_neighbours)
    adata.obsp['binary'] = adata.obsp['connectivities'] != 0
    distances = []
    velocities = []
    cosines = []
    transition_probabilities = []
    matrices = []
    if full_posterior:
        for i in range(M):
            distances += [adata.layers['spliced_norm'][adata.obsp['binary'].toarray()[i,:],:] - adata.layers['spliced_norm'][i,:].flatten()]
            velocities += [adata.uns['velocity_posterior'][:,i,:]]
            cosines += [inner(distances[i], velocities[i])/(norm(distances[i])*norm(velocities[i]))]
            transition_probabilities += [np.exp(2*cosines[i])]
            transition_probabilities[i] = transition_probabilities[i]/np.sum(transition_probabilities[i], axis = 0)
            matrices += [csr_matrix((np.mean(np.array(transition_probabilities[i]), axis = 1),
                    (np.repeat(i, len(transition_probabilities[i])), np.where(adata.obsp['binary'][i,:].toarray())[1])),
                                   shape=(M, M))]
    else:
        for i in range(M):
            distances += [adata.layers['spliced_norm'][adata.obsp['binary'].toarray()[i,:],:] - adata.layers['spliced_norm'][i,:].flatten()]
            velocities += [adata.layers['velocity'][i,:].reshape(1,len(adata.var_names))]
            cosines += [inner(distances[i], velocities[i])/(norm(distances[i])*norm(velocities[i]))]
            transition_probabilities += [np.exp(2*cosines[i])]
            transition_probabilities[i] = transition_probabilities[i]/np.sum(transition_probabilities[i], axis = 0)
            matrices += [csr_matrix((np.mean(np.array(transition_probabilities[i]), axis = 1),
                    (np.repeat(i, len(transition_probabilities[i])), np.where(adata.obsp['binary'][i,:].toarray())[1])),
                                   shape=(M, M))]
    return sum(matrices)

def plot_velocity_umap_Bergen2020(adata, use_full_posterior = True,
                                  plotting_kwargs: Optional[dict] = None):
    """
    Visualizes RNAvelocity with arrows on a UMAP, using the method introduced in 
    "Bergen et al. (2020), Generalizing RNA velocity to transient cell states through dynamical modeling"
    The method computes a "velocity graph" before plotting (see the referenced paper for details),
    unless such a graph is already available. The graph is based on the full velocity posterior distribution if available,
    otherwise it is based on the velocity expectation values.
    Velocity is expected in adata.layers['velocity'] or adata.uns['velocity_posterior'] (for the full posterior)
    and the graph is saved/expected in adata.layers['velocity_graph'].
    
    Parameters
    ----------
    adata
        anndata object with velocity information
    use_full_posterior
        use full posterior to compute velocity graph (if available)
    plotting_kwargs
        
    Returns
    -------
    UMAP plot with RNAvelocity arrows.
    """
    if 'velocity_graph' in adata.uns.keys():
        print('Using existing velocity graph')
        scv.pl.velocity_embedding_stream(adata, basis='umap', **plotting_kwargs)
    else:
        if use_full_posterior and 'velocity_posterior' in adata.uns.keys():
            print('Using full velocity posterior to calculate velocity graph')
            adata.uns['velocity_graph'] = compute_velocity_graph_Bergen2020(adata, full_posterior = True)          
        elif use_full_posterior and 'velocity_posterior' not in adata.uns.keys():
            print('Full velocity posterior not found, using expectation value to calculate velocity graph')
            adata.uns['velocity_graph'] = compute_velocity_graph_Bergen2020(adata, full_posterior = False)
        elif not use_full_posterior:
            print('Using velocity expectation value to calculate velocity graph.')
            adata.uns['velocity_graph'] = compute_velocity_graph_Bergen2020(adata, full_posterior = False)
        
        scv.pl.velocity_embedding_stream(adata, basis='umap', **plotting_kwargs)

def get_training_data(adata, remove_clusters = None, cells_per_cluster = 100,
                         cluster_column = 'clusters', min_shared_counts = 10, n_var_genes = 2000):
    """
    Reduces and anndata object to the most relevant cells and genes for understanding the differentiation trajectories
    in the data.
    
    Parameters
    ----------
    adata
        anndata
    remove_clusters
        names of clusters to be removed
    cells_per_cluster
        how many cells to keep per cluster. For Louvain clustering with resolution = 1, keeping more than 300 cells
        per cluster does not provide much extra information.
    cluster_column
        name of the column in adata.obs that contains cluster names
    min_shared_counts
        minimum number of spliced+unspliced counts across all cells for a gene to be retained
    n_var_genes
        number of top variable genes to retain
        
    Returns
    -------
    adata object reduced to the most informative cells and genes
    """
    random.seed(a=1)
    adata = adata[[c not in remove_clusters for c in adata.obs[cluster_column]], :]
    # Restrict samples per cell type:
    N = cells_per_cluster
    unique_celltypes = np.unique(adata.obs[cluster_column])
    index = []
    for i in range(len(unique_celltypes)):
        if adata.obs[cluster_column].value_counts()[unique_celltypes[i]] > N:
            subset = np.where(adata.obs[cluster_column] == unique_celltypes[i])[0]
            subset = random.sample(list(subset), N)
        else:
            subset = np.where(adata.obs[cluster_column] == unique_celltypes[i])[0]
        index += list(subset)
    adata = adata[index,:]
    print('Keeping at most ' + str(N) + ' cells per cluster')
    scv.pp.filter_genes(adata, min_shared_counts=min_shared_counts)
    sc.pp.normalize_total(adata, target_sum=1e4)
    scv.pp.filter_genes_dispersion(adata, n_top_genes=n_var_genes)
    return adata

def add_prior_knowledge(adata, cluster_column, initial_stages = None, initial_stages_lineage = None,
                           intermediate_stages = None, intermediate_stages_lineage = None,
                           terminal_stages = None, terminal_stages_lineage = None):
    
    """
    Adds prior knowledge about differentiation trajectories to the anndata object.
    
    Parameters
    ----------
    adata
        anndata
    cluster_column
        name of the column in adata.obs that contains cluster names
    initial_stages
        list of cluster names that are known to be initial stages of a differentiation trajectory
    initial_stages_lineage
        list of lineage names for initial_stage clusters, if not known provide 'Unknown'
    intermediate_stages
        list of cluster names that are known to be intermediate stages of a differentiation trajectory
    intermediate_stages_lineage
        list of lineage names for intermediate_stage clusters, if not known provide 'Unknown'
    terminal_stages
        list of cluster names that are known to be terminal stages of a differentiation trajectory
    terminal_stages_lineage
        list of lineage names for terminal_stage clusters, if not known provide 'Unknown' 
    Returns
    -------
    adata object with adata.obs['Stage'] containing differentiation stage of cell (either 'Unknown' or 'Initial', 'Intermediate', 'Terminal') \
    adata.obs['Lineage'] contains lineage of cell (either 'Unknown' or as named by user) \
    """
    
    adata.obs['Stage'] = 'Unknown'
    adata.obs['Lineage'] = 0
#     adata.obs['Prior Time'] = 0.5
    
    if initial_stages:
        subset = [c in initial_stages for c in adata.obs[cluster_column]]
        adata.obs['Stage'].loc[subset] = 'Initial'
#         adata.obs['Prior Time'].loc[subset] = 0.1
    if intermediate_stages:
        subset = [c in intermediate_stages for c in adata.obs[cluster_column]]
        adata.obs['Stage'].loc[subset] = 'Intermediate'
#         adata.obs['Prior Time'].loc[subset] = 0.5
    if terminal_stages:
        subset = [c in terminal_stages for c in adata.obs[cluster_column]]
        adata.obs['Stage'].loc[subset] = 'Terminal'
#         adata.obs['Prior Time'].loc[subset] = 0.9
    
    if initial_stages_lineage:
        for i in range(len(initial_stages)):
            adata.obs['Lineage'].loc[[c ==  initial_stages[i] for c in adata.obs[cluster_column]]] = initial_stages_lineage[i]
    if intermediate_stages_lineage:
        for i in range(len(intermediate_stages)):
            adata.obs['Lineage'].loc[[c ==  intermediate_stages[i] for c in adata.obs[cluster_column]]] = intermediate_stages_lineage[i]
    if terminal_stages_lineage:
        for i in range(len(terminal_stages)):
            adata.obs['Lineage'].loc[[c ==  terminal_stages[i] for c in adata.obs[cluster_column]]] = terminal_stages_lineage[i]
    
    return adata

def G_a(mu, sd):
    return mu**2/sd**2

def G_b(mu, sd):
    return mu/sd**2

def mu_alpha(alpha_new, alpha_old, tau, lam):
    '''Calculates transcription rate as a function of new target transcription rate,
    old transcription rate at changepoint, time since change point and rate of exponential change process'''
    return (alpha_new - alpha_old) * (1 - torch.exp(-lam*tau)) + alpha_old

def mu_mRNA_discreteAlpha(alpha, beta, gamma, tau, u0, s0):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    latent time and initial states'''
    
    mu_u = u0*torch.exp(-beta*tau) + (alpha/beta)* (1 - torch.exp(-beta*tau))
    mu_s = (s0*torch.exp(-gamma*tau) + 
    alpha/gamma * (1 - torch.exp(-gamma*tau)) +
    (alpha - beta * u0)/(gamma - beta + 10**(-5)) * (torch.exp(-gamma*tau) - torch.exp(-beta*tau)))

    return torch.stack([mu_u, mu_s], axis = 2)

def mu_mRNA_discreteAlpha_OnePlate(alpha, beta, gamma, tau, u0, s0):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    latent time and initial states'''
    
    mu_u = u0*torch.exp(-beta*tau) + (alpha/beta)* (1 - torch.exp(-beta*tau))
    mu_s = (s0*torch.exp(-gamma*tau) + 
    alpha/gamma * (1 - torch.exp(-gamma*tau)) +
    (alpha - beta * u0)/(gamma - beta + 10**(-5)) * (torch.exp(-gamma*tau) - torch.exp(-beta*tau)))

    return torch.stack([mu_u, mu_s], axis = 0)

def mu_mRNA_discreteAlpha_withPlates(alpha, beta, gamma, tau, u0, s0):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    latent time and initial states'''
    
    mu_u = u0*torch.exp(-beta*tau) + (alpha/beta)* (1 - torch.exp(-beta*tau))
    mu_s = (s0*torch.exp(-gamma*tau) + 
    alpha/gamma * (1 - torch.exp(-gamma*tau)) +
    (alpha - beta * u0)/(gamma - beta + 10**(-5)) * (torch.exp(-gamma*tau) - torch.exp(-beta*tau)))

    return torch.concat([mu_u, mu_s], axis = -1)

def mu_mRNA_continuousAlpha(alpha, beta, gamma, tau, u0, s0, delta_alpha, lam):
    ''' Calculates expected value of spliced and unspliced counts as a function of rates, latent time, initial states,
    difference to transcription rate in previous state and rate of exponential change process between states.'''
    
    mu_u = u0*torch.exp(-beta*tau) + (alpha/beta)* (1 - torch.exp(-beta*tau)) + delta_alpha/(beta-lam)*(torch.exp(-beta*tau) - torch.exp(-lam*tau))
    mu_s = (s0*torch.exp(-gamma*tau) + 
    alpha/gamma * (1 - torch.exp(-gamma*tau)) +
    (alpha - beta * u0)/(gamma - beta + 10**(-5)) * (torch.exp(-gamma*tau) - torch.exp(-beta*tau)) +
    (delta_alpha*beta)/((beta - lam + 10**(-5))*(gamma - beta + 10**(-5))) * (torch.exp(-beta*tau) - torch.exp(-gamma*tau))-
    (delta_alpha*beta)/((beta - lam + 10**(-5))*(gamma - lam + 10**(-5))) * (torch.exp(-lam*tau) - torch.exp(-gamma*tau)))

    return torch.stack([mu_u, mu_s], axis = 2)

def mu_mRNA_continuousAlpha_withPlates(alpha, beta, gamma, tau, u0, s0, delta_alpha, lam):
    ''' Calculates expected value of spliced and unspliced counts as a function of rates, latent time, initial states,
    difference to transcription rate in previous state and rate of exponential change process between states.'''
    
    mu_u = u0*torch.exp(-beta*tau) + (alpha/beta)* (1 - torch.exp(-beta*tau)) + delta_alpha/(beta-lam)*(torch.exp(-beta*tau) - torch.exp(-lam*tau))
    mu_s = (s0*torch.exp(-gamma*tau) + 
    alpha/gamma * (1 - torch.exp(-gamma*tau)) +
    (alpha - beta * u0)/(gamma - beta + 10**(-5)) * (torch.exp(-gamma*tau) - torch.exp(-beta*tau)) +
    (delta_alpha*beta)/((beta - lam + 10**(-5))*(gamma - beta + 10**(-5))) * (torch.exp(-beta*tau) - torch.exp(-gamma*tau))-
    (delta_alpha*beta)/((beta - lam + 10**(-5))*(gamma - lam + 10**(-5))) * (torch.exp(-lam*tau) - torch.exp(-gamma*tau)))

    return torch.stack([mu_u, mu_s], axis = -1)

def mu_mRNA_discreteAlpha_localTime_twoStates(alpha_ON, alpha_OFF, beta, gamma, tau, t0):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    local latent time, initial states and switch time between two states'''
    n_cells = tau.shape[0]
    n_genes = alpha_ON.shape[1]
    # Transcription rate in each cell for each gene:
    boolean = (tau < t0).reshape(n_cells, n_genes)
    alpha_cg = alpha_ON*boolean + alpha_OFF*~boolean
    # Time since changepoint for each cell and gene:
    tau_cg = tau*boolean + (tau - t0)*~boolean
    # Initial condition for each cell and gene:
    initial_state = mu_mRNA_discreteAlpha(alpha_ON, beta, gamma, t0, torch.zeros(n_cells,n_genes), torch.zeros(n_cells,n_genes))
    u0_g = ~boolean*initial_state[:,:,0]
    s0_g = ~boolean*initial_state[:,:,1]
    # Unspliced and spliced counts for each gene in each cell:
    return mu_mRNA_discreteAlpha(alpha_cg, beta, gamma, tau_cg, u0_g, s0_g)

def mu_mRNA_discreteAlpha_globalTime_twoStates(alpha_ON, alpha_OFF, beta, gamma, T_c, T_gON, T_gOFF):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    global latent time, initial states and global switch times between two states'''
    n_cells = T_c.shape[0]
    n_genes = alpha_ON.shape[1]
    tau = torch.clip(T_c - T_gON, min = 10**(-5))
    t0 = T_gOFF - T_gON
    # Transcription rate in each cell for each gene:
    boolean = (tau < t0).reshape(n_cells, n_genes)
    alpha_cg = alpha_ON*boolean + alpha_OFF*~boolean
    # Time since changepoint for each cell and gene:
    tau_cg = tau*boolean + (tau - t0)*~boolean
    # Initial condition for each cell and gene:
    initial_state = mu_mRNA_discreteAlpha(alpha_ON, beta, gamma, t0, torch.zeros(n_cells,n_genes), torch.zeros(n_cells,n_genes))
    u0_g = 10**(-5) + ~boolean*initial_state[:,:,0]
    s0_g = 10**(-5) + ~boolean*initial_state[:,:,1]
    # Unspliced and spliced count variance for each gene in each cell:
    return torch.clip(mu_mRNA_discreteAlpha(alpha_cg, beta, gamma, tau_cg, u0_g, s0_g), min = 10**(-5))

def mu_mRNA_discreteAlpha_globalTime_twoStates_OnePlate(alpha_ON, alpha_OFF, beta, gamma, T_c, T_gON, T_gOFF, Zeros):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    global latent time, initial states and global switch times between two states'''
    n_cells = T_c.shape[0]
    n_genes = alpha_ON.shape[1]
    tau = torch.clip(T_c - T_gON, min = 10**(-5))
    t0 = T_gOFF - T_gON
    # Transcription rate in each cell for each gene:
    boolean = (tau < t0).reshape(n_cells, n_genes)
    alpha_cg = alpha_ON*boolean + alpha_OFF*~boolean
    # Time since changepoint for each cell and gene:
    tau_cg = tau*boolean + (tau - t0)*~boolean
    # Initial condition for each cell and gene:
    initial_state = mu_mRNA_discreteAlpha_OnePlate(alpha_ON, beta, gamma, t0, Zeros, Zeros)
    u0_g = 10**(-5) + ~boolean*initial_state[0,:,:]
    s0_g = 10**(-5) + ~boolean*initial_state[1,:,:]
    # Unspliced and spliced count variance for each gene in each cell:
    return torch.clip(mu_mRNA_discreteAlpha_OnePlate(alpha_cg, beta, gamma, tau_cg, u0_g, s0_g), min = 10**(-5))

def var_mRNA_discreteAlpha_globalTime_twoStates(alpha_ON, alpha_OFF, beta, gamma, T_c, T_gON, T_gOFF):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    global latent time, initial states and global switch times between two states'''
    n_cells = T_c.shape[0]
    n_genes = alpha_ON.shape[1]
    tau = torch.clip(T_c - T_gON, min = 10**(-5))
    t0 = T_gOFF - T_gON
    # Transcription rate in each cell for each gene:
    boolean = (tau < t0).reshape(n_cells, n_genes)
    alpha_cg = alpha_ON*boolean + alpha_OFF*~boolean
    # Time since changepoint for each cell and gene:
    tau_cg = tau*boolean + (tau - t0)*~boolean
    # Initial condition for each cell and gene:
    initial_state = mu_mRNA_discreteAlpha(alpha_ON, beta, gamma, t0, torch.zeros(n_cells,n_genes), torch.zeros(n_cells,n_genes))
    u0_g = 10**(-5) + ~boolean*initial_state[:,:,0]
    s0_g = 10**(-5) + ~boolean*initial_state[:,:,1]
    # Unspliced and spliced count variance for each gene in each cell:
    p_u = torch.exp(-beta*tau_cg)
    lambda_u = alpha_cg/beta*(1-torch.exp(-beta*tau_cg))
    q_u = torch.where(beta == gamma, beta*tau_cg*torch.exp(-beta*tau_cg), beta/(gamma - beta) * (torch.exp(-beta*tau_cg) - torch.exp(-gamma*tau_cg)))
    lambda_s = torch.where(beta == gamma, alpha_cg/beta*(1-torch.exp(-beta*tau_cg)) - alpha_cg*tau_cg*torch.exp(-beta*tau_cg) ,alpha_cg/gamma * (1-torch.exp(-gamma*tau_cg)) + alpha_cg/(beta-gamma) * (torch.exp(-beta*tau_cg)-torch.exp(-gamma*tau_cg)))
    var_u = u0_g * p_u*(1-p_u) + lambda_u
    var_s = u0_g * q_u*(1-q_u) + s0_g*torch.exp(-gamma*tau_cg)*(1-torch.exp(-gamma*tau_cg)) + lambda_s
    return torch.stack([var_u, var_s], axis = 2)

def mu_mRNA_discreteAlpha_globalTime_twoStates_withPlates(alpha_ON, alpha_OFF, beta, gamma, T_c, T_gON, T_gOFF):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    global latent time, initial states and global switch times between two states'''
    n_cells = T_c.shape[0]
    n_genes = alpha_ON.shape[1]
    tau = torch.clip(T_c - T_gON, min = 10**(-5))
    t0 = T_gOFF - T_gON
    # Transcription rate in each cell for each gene:
    boolean = (tau < t0)
    alpha_cg = alpha_ON*boolean + alpha_OFF*~boolean
    # Time since changepoint for each cell and gene:
    tau_cg = tau*boolean + (tau - t0)*~boolean
    # Initial condition for each cell and gene:
    initial_state = mu_mRNA_discreteAlpha_withPlates(alpha_ON, beta, gamma, t0, torch.zeros(boolean.shape), torch.zeros(boolean.shape))
    u0_g = 10**(-3) + ~boolean*initial_state[...,0:1]
    s0_g = 10**(-3) + ~boolean*initial_state[...,1:2]
    # Unspliced and spliced counts for each gene in each cell:
    return torch.clip(mu_mRNA_discreteAlpha_withPlates(alpha_cg, beta, gamma, tau_cg, u0_g, s0_g), min = 10**(-3))

def mu_mRNA_continuousAlpha_globalTime_transcriptionalModules(alpha_mg, beta, gamma, T_c, T_mON, T_mOFF, T_max, lam):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    global latent time, initial states and global switch times of transcriptional modules.'''
    n_cells = T_c.shape[0]
    n_genes = alpha_mg.shape[1]
    n_modules = alpha_mg.shape[0]
    changepoints = torch.sort(torch.concat([T_mON, T_mOFF], axis = 1)).values
    delta_T = T_c - changepoints
    minimum = torch.min((delta_T + ((delta_T < 0) * T_max*100)), axis = 1)
    # Last change point in each cell:
    P_c = minimum.indices
    # Time since last transcription rate change point in each cell:
    tau_c = minimum.values
    # Target transcription rate at each changepoints:
    N_pm = (changepoints.T >= T_mON)*(changepoints.T < T_mOFF)*1.
    alpha_pg = torch.mm(N_pm, alpha_mg)
    # Time since last transcription rate change point at each changepoint:
    tau_p = torch.concat([torch.zeros(1,1), changepoints[:,1:] - changepoints[:,:(2*n_modules -1)]], axis = 1)
    # Actual transcription rate at each changepoint:
    A_pg = [torch.zeros((1, n_genes))] 
    for i in range(2*n_modules):
        A_pg += [mu_alpha(torch.concat([torch.zeros((1,n_genes)),alpha_pg], axis = 0)[i,:], A_pg[i], tau_p[0,i], lam)]
    A_pg = torch.stack(A_pg, axis = 1)[0,1:,:]
    # Expected unspliced and spliced counts at each changepoint:
    mu_p = [torch.stack([torch.zeros((1, n_genes)), torch.zeros((1, n_genes))], axis = 2)]
    for i in range(2*n_modules):
        mu_p += [mu_mRNA_continuousAlpha(torch.concat([torch.zeros((1,n_genes)),alpha_pg], axis = 0)[i,:], beta, gamma, tau_p[0,i], mu_p[i][:,:,0], mu_p[i][:,:,1],
                        torch.concat([torch.zeros((1,n_genes)),alpha_pg], axis = 0)[i,:] - torch.concat([torch.zeros((1,n_genes)),A_pg], axis = 0)[i,:], lam)]
    mu_p = torch.stack(mu_p, axis = 3)[:,:,:,1:]
    # Calculate expected unspliced and spliced counts for each gene in each cell:
    mu_cg = mu_mRNA_continuousAlpha(alpha_pg[P_c,:], beta, gamma,
                    tau_c.reshape((n_cells,1)), 
                    mu_p[:,:,0,P_c].T.reshape((n_cells,n_genes)),
                    mu_p[:,:,1,P_c].T.reshape((n_cells,n_genes)),
                    alpha_pg[P_c,:] - A_pg[P_c,:], lam)
    return torch.clip(mu_cg, min = 10**(-5))

def mu_mRNA_continuousAlpha_globalTime_transcriptionalModules_returnAlpha(alpha_mg, beta, gamma, T_c, T_mON, T_mOFF, T_max, lam):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    global latent time, initial states and global switch times of transcriptional modules.'''
    n_cells = T_c.shape[0]
    n_genes = alpha_mg.shape[1]
    n_modules = alpha_mg.shape[0]
    changepoints = torch.sort(torch.concat([T_mON, T_mOFF], axis = 1)).values
    delta_T = T_c - changepoints
    minimum = torch.min((delta_T + ((delta_T < 0) * T_max*100)), axis = 1)
    # Last change point in each cell:
    P_c = minimum.indices
    # Time since last transcription rate change point in each cell:
    tau_c = minimum.values
    # Target transcription rate at each changepoints:
    N_pm = (changepoints.T >= T_mON)*(changepoints.T < T_mOFF)*1.
    alpha_pg = torch.mm(N_pm, alpha_mg)
    # Time since last transcription rate change point at each changepoint:
    tau_p = torch.concat([torch.zeros(1,1), changepoints[:,1:] - changepoints[:,:(2*n_modules -1)]], axis = 1)
    # Actual transcription rate at each changepoint:
    A_pg = [torch.zeros((1, n_genes))] 
    for i in range(2*n_modules):
        A_pg += [mu_alpha(torch.concat([torch.zeros((1,n_genes)),alpha_pg], axis = 0)[i,:], A_pg[i], tau_p[0,i], lam)]
    A_pg = torch.stack(A_pg, axis = 1)[0,1:,:]
    # Expected unspliced and spliced counts at each changepoint:
    mu_p = [torch.stack([torch.zeros((1, n_genes)), torch.zeros((1, n_genes))], axis = 2)]
    for i in range(2*n_modules):
        mu_p += [mu_mRNA_continuousAlpha(torch.concat([torch.zeros((1,n_genes)),alpha_pg], axis = 0)[i,:], beta, gamma, tau_p[0,i], mu_p[i][:,:,0], mu_p[i][:,:,1],
                        torch.concat([torch.zeros((1,n_genes)),alpha_pg], axis = 0)[i,:] - torch.concat([torch.zeros((1,n_genes)),A_pg], axis = 0)[i,:], lam)]
    mu_p = torch.stack(mu_p, axis = 3)[:,:,:,1:]
    # Calculate expected unspliced and spliced counts for each gene in each cell:
    mu_cg = mu_mRNA_continuousAlpha(alpha_pg[P_c,:], beta, gamma,
                    tau_c.reshape((n_cells,1)), 
                    mu_p[:,:,0,P_c].T.reshape((n_cells,n_genes)),
                    mu_p[:,:,1,P_c].T.reshape((n_cells,n_genes)),
                    alpha_pg[P_c,:] - A_pg[P_c,:], lam)
    # Also calculate transcription rate:
    alpha_cg = mu_alpha(alpha_pg[P_c,:], A_pg[P_c,:], tau_c.reshape(n_cells,1), lam)
    return torch.clip(mu_cg, min = 10**(-5)), alpha_cg


def mu_mRNA_continuousAlpha_globalTime_transcriptionalModules_withPlates(alpha_mg, beta, gamma, T_c, T_mON, T_mOFF, T_max, lam):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    global latent time, initial states and global switch times of transcriptional modules.
    Assumes extra plate dimensions for cells.'''
    n_cells = T_c.shape[0]
    n_genes = alpha_mg.shape[1]
    n_modules = alpha_mg.shape[0]
    changepoints = torch.sort(torch.concat([T_mON, T_mOFF], axis = -1)).values
    delta_T = T_c - changepoints[0,0,0,0,:]
    minimum = torch.min((delta_T + ((delta_T < 0) * T_max*100)), axis = 1)
    # Last change point in each cell:
    P_c = minimum.indices
    # Time since last transcription rate change point in each cell:
    tau_c = minimum.values
    # Target transcription rate at each changepoints:
    N_pm = (changepoints.T >= T_mON)*(changepoints.T < T_mOFF)*1.
    alpha_pg = torch.mm(N_pm[:,0,0,0,:], alpha_mg)
    # Time since last transcription rate change point at each changepoint:
    tau_p = torch.concat([torch.zeros(1,1,1,1,1), changepoints[:,:,:,:,1:] - changepoints[:,:,:,:,:(2*n_modules -1)]], axis = -1)
    # Actual transcription rate at each changepoint:
    A_pg = [torch.zeros((1, n_genes))] 
    for i in range(2*n_modules):
        A_pg += [mu_alpha(torch.concat([torch.zeros((1,n_genes)),alpha_pg], axis = 0)[i,:], A_pg[i], tau_p[0,0,0,0,i], lam)]
    A_pg = torch.stack(A_pg, axis = 1)[0,1:,:]
    # Expected unspliced and spliced counts at each changepoint:
    mu_p = [torch.stack([torch.zeros((1, n_genes)), torch.zeros((1, n_genes))], axis = 2)]
    for i in range(2*n_modules):
        mu_p += [mu_mRNA_continuousAlpha_withPlates(torch.concat([torch.zeros((1,n_genes)),alpha_pg], axis = 0)[i,:], beta[0,0,0,0,:], gamma[0,0,0,0,:], tau_p[0,0,0,0,i], mu_p[i][:,:,0], mu_p[i][:,:,1],
                        torch.concat([torch.zeros((1,n_genes)),alpha_pg], axis = 0)[i,:] - torch.concat([torch.zeros((1,n_genes)),A_pg], axis = 0)[i,:], lam)]
    mu_p = torch.stack(mu_p, axis = 3)[:,:,:,1:]
    # Calculate expected unspliced and spliced counts for each gene in each cell:
    mu_cg = mu_mRNA_continuousAlpha_withPlates(alpha_pg[P_c,:], beta, gamma,
                    tau_c.reshape((n_cells,1)), 
                    mu_p[:,:,0,P_c].T.reshape((n_cells,n_genes)),
                    mu_p[:,:,1,P_c].T.reshape((n_cells,n_genes)),
                    alpha_pg[P_c,:] - A_pg[P_c,:], lam)
    # Also calculate transcription rate:
    alpha_cg = mu_alpha(alpha_pg[P_c,:], A_pg[P_c,:], tau_c.reshape(n_cells,1), lam)
    return torch.clip(mu_cg, min = 10**(-5)), alpha_cg

def mu_mRNA_discreteAlpha_withLineages(alpha, beta, gamma, tau, u0, s0):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    latent time and initial states'''
    
    mu_u = u0*torch.exp(-beta*tau) + (alpha/beta)* (1 - torch.exp(-beta*tau))
    mu_s = (s0*torch.exp(-gamma*tau) + 
    alpha/gamma * (1 - torch.exp(-gamma*tau)) +
    (alpha - beta * u0)/(gamma - beta + 10**(-5)) * (torch.exp(-gamma*tau) - torch.exp(-beta*tau)))

    return torch.concat([mu_u, mu_s], axis = -3)

def mu_mRNA_discreteAlpha_globalTime_twoStates_withLineages(alpha_ON, alpha_OFF, beta, gamma, T_c, T_gON, T_gOFF):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    global latent time, initial states and global switch times between two states'''
    n_cells = T_c.shape[-1]
    n_genes = alpha_ON.shape[-2]
    tau = torch.clip(T_c - T_gON, min = 10**(-5))
    t0 = T_gOFF - T_gON
    # Transcription rate in each cell for each gene:
    boolean = (tau < t0)
    alpha_cg = alpha_ON*boolean + alpha_OFF*~boolean
    # Time since changepoint for each cell and gene:
    tau_cg = tau*boolean + (tau - t0)*~boolean
    # Initial condition for each cell and gene:
    initial_state = mu_mRNA_discreteAlpha_withLineages(alpha_ON, beta, gamma, t0, torch.zeros(boolean.shape), torch.zeros(boolean.shape))
    u0_g = 10**(-3) + ~boolean*initial_state[...,0:1,:,:]
    s0_g = 10**(-3) + ~boolean*initial_state[...,1:2,:,:]
    # Unspliced and spliced counts for each gene in each cell:
    return torch.clip(mu_mRNA_discreteAlpha_withLineages(alpha_cg, beta, gamma, tau_cg, u0_g, s0_g), min = 10**(-3))