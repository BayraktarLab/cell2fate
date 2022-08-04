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
import matplotlib.pyplot as plt
from contextlib import contextmanager
import seaborn as sns
import os,sys

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def test_differentiation_time(mod, clusters, origin_cluster, target_clusters):
    with suppress_stdout():
        T_c_posterior = mod.sample_posterior(**{'return_sites' : ['T_c'], 'num_samples' : 10, 'return_samples' : 'True',
                                        'batch_size' : len(mod.samples['post_sample_means']['T_c'].flatten())})['posterior_samples']['T_c']
        for i in range(2):
            posterior = mod.sample_posterior(**{'return_sites' : ['T_c'], 'num_samples' : 10, 'return_samples' : 'True',
                                            'batch_size' : len(mod.samples['post_sample_means']['T_c'].flatten())})
            T_c_posterior = np.concatenate([T_c_posterior, posterior['posterior_samples']['T_c']])
    i = 0
    df_list = []
    for i in range(len(target_clusters)):
        df = pd.DataFrame(columns = ['T_c', 'cluster'])
        cluster = target_clusters[i]
        df['T_c'] = (T_c_posterior[:, clusters == cluster,...] - np.mean(T_c_posterior[:, clusters == origin_cluster,...])).flatten()
        df['cluster'] = cluster
        df_list += [df]
    df = pd.concat(df_list)
    
    Means = df.groupby('cluster')['T_c'].mean()

    p=sns.violinplot(x="cluster", y='T_c', data=df, palette="PRGn", inner = None)
    # p = sns.stripplot(x="cluster", y='T_c', data=df, color = 'black', alpha = 0.1, size = 10)
    sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'red', 'ls': '-', 'lw': 2},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=10,
                x="cluster",
                y='T_c',
                data=df,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=p)
    # plt.setp(ax.collections, alpha=0)
    plt.scatter(x=range(len(Means)), y=Means,c="red")
    plt.ylabel('Time (hours)')
    plt.xticks(rotation=90)
    plt.title('Posterior samples of differentiation time in each cell')

def removeOutliers(adata, min_total, max_total, min_ratio, max_ratio, plot = True, remove = True):
    """
    Removes outlier cells from adata object based on total counts or unspliced/spliced counts ratio.   
    Parameters
    ----------
    adata
        anndata
    min_total
        lower threshold for total counts
    max total
        upper threshold for total counts
    min_ratio
        lower threshold for ratio of unspliced/spliced counts
    max_ratio
        upper threshold for ratio of unspliced/spliced counts
    plot
        whether to show plot overviews of identified outliers in UMAP and violin plots
    remove
        whether to return an adata object with outliers removed        
    Returns
    -------
    Optionally adata object with outliers removed.
    """
    adata.obs['spliced_total'] = np.sum(adata.layers['spliced'], axis = 1)
    adata.obs['unspliced_total'] = np.sum(adata.layers['unspliced'], axis = 1)
    adata.obs['counts_total'] = adata.obs['spliced_total'] + adata.obs['unspliced_total']
    adata.obs['unspliced_spliced_ratio'] = adata.obs['unspliced_total']/adata.obs['spliced_total']
    subset = np.array([adata.obs['counts_total'].iloc[i] > min_total and
             adata.obs['counts_total'].iloc[i] < max_total and
             adata.obs['unspliced_spliced_ratio'].iloc[i] > min_ratio and
             adata.obs['unspliced_spliced_ratio'].iloc[i] < max_ratio for
             i in range(len(adata.obs['counts_total']))])
    subset_ratio = np.array([adata.obs['unspliced_spliced_ratio'].iloc[i] > min_ratio and
             adata.obs['unspliced_spliced_ratio'].iloc[i] < max_ratio for
             i in range(len(adata.obs['counts_total']))])
    subset_total = np.array([adata.obs['counts_total'].iloc[i] > min_total and
             adata.obs['counts_total'].iloc[i] < max_total for
             i in range(len(adata.obs['counts_total']))])
    adata.obs['Outliers'] = ~subset
    adata.obs['Counts Outliers'] = 'False'
    adata.obs['Counts Outliers'].loc[~subset_total] = 'True'
    adata.obs['Ratio Outliers'] = 'False'
    adata.obs['Ratio Outliers'].loc[~subset_ratio] = 'True'
    if plot:
        fig, ax = plt.subplots(2,5, figsize = (20, 7))
        sc.pl.umap(adata, color = ['clusters'], legend_loc = 'on data', size = 200, ax = ax[0,0], show = False)
        sc.pl.umap(adata, color = ['unspliced_spliced_ratio'], legend_loc = 'on data', size = 200, ax = ax[0,2], show = False)
        sc.pl.umap(adata, color = ['counts_total'], legend_loc = 'on data', size = 200, ax = ax[0,1], show = False)
        sc.pl.violin(adata, ['unspliced_spliced_ratio'], jitter=0.4, size = 3, ax = ax[0,4], show = False)
        sc.pl.violin(adata, ['counts_total'], jitter=0.4, size = 3, ax = ax[0,3], show = False)
        sc.pl.umap(adata, color = ['clusters'], legend_loc = 'on data', size = 200, ax = ax[1,0], show = False)
        sc.pl.umap(adata, color = ['Counts Outliers'], legend_loc = None, size = 200, ax = ax[1,1], show = False)
        sc.pl.umap(adata, color = ['Ratio Outliers'], legend_loc = None, size = 200, ax = ax[1,2], show = False)
        sc.pl.violin(adata, ['unspliced_spliced_ratio'], jitter=0.4, size = 3, ax = ax[1,4], groupby = 'Ratio Outliers', show = False)
        sc.pl.violin(adata, ['counts_total'], jitter=0.4, size = 3, groupby = 'Counts Outliers', ax = ax[1,3], show = False)
        plt.tight_layout()
    if not remove:
        print(str(np.sum(~subset_total)) + ' Cells would be removed as total counts outliers.')
        print(str(np.sum(~subset_ratio)) + ' Cells would be removed as ratio outliers.')
    else:
        print(str(np.sum(~subset_total)) + ' Cells removed as total counts outliers.')
        print(str(np.sum(~subset_ratio)) + ' Cells removed as ratio outliers.')
    if remove:
        return adata[subset,:]

def multiplot_from_generator(g, num_columns, figsize_for_one_row=None, row_size = 15):
    # Copied from: https://aaronmcdaid.com/blog.posts/multiplot_from_generator/
    # on 28/04/2022
    # Plots multiple plots side by side in a jupyter notebook
        next(g)
        # default to 15-inch rows, with square subplots
        if figsize_for_one_row is None:
            figsize_for_one_row = (row_size, row_size/num_columns)
        try:
            while True:
                # call plt.figure once per row
                plt.figure(figsize=figsize_for_one_row)
                for col in range(num_columns):
                    ax = plt.subplot(1, num_columns, col+1)
                    next(g)
        except StopIteration:
            pass

def compute_velocity_graph_Bergen2020(adata, n_neighbours = None, full_posterior = True, spliced_key = 'Ms'):
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
        n_neighbours = int(np.round(M*0.05, 0))
    scv.pp.neighbors(adata, n_neighbors = n_neighbours)
    adata.obsp['binary'] = adata.obsp['connectivities'] != 0
    distances = []
    velocities = []
    cosines = []
    transition_probabilities = []
    matrices = []
    if full_posterior:
        for i in range(M):
            distances += [adata.layers[spliced_key][adata.obsp['binary'].toarray()[i,:],:] - adata.layers[spliced_key][i,:].flatten()]
            velocities += [adata.uns['velocity_posterior'][:,i,:]]
            cosines += [inner(distances[i], velocities[i])/(norm(distances[i])*norm(velocities[i]))]
            transition_probabilities += [np.exp(2*cosines[i])]
            transition_probabilities[i] = transition_probabilities[i]/np.sum(transition_probabilities[i], axis = 0)
            matrices += [csr_matrix((np.mean(np.array(transition_probabilities[i]), axis = 1),
                    (np.repeat(i, len(transition_probabilities[i])), np.where(adata.obsp['binary'][i,:].toarray())[1])),
                                   shape=(M, M))]
    else:
        for i in range(M):
            distances += [adata.layers[spliced_key][adata.obsp['binary'].toarray()[i,:],:] - adata.layers[spliced_key][i,:].flatten()]
            velocities += [adata.layers['velocity'][i,:].reshape(1,len(adata.var_names))]
            cosines += [inner(distances[i], velocities[i])/(norm(distances[i])*norm(velocities[i]))]
            transition_probabilities += [np.exp(2*cosines[i])]
            transition_probabilities[i] = transition_probabilities[i]/np.sum(transition_probabilities[i], axis = 0)
            matrices += [csr_matrix((np.mean(np.array(transition_probabilities[i]), axis = 1),
                    (np.repeat(i, len(transition_probabilities[i])), np.where(adata.obsp['binary'][i,:].toarray())[1])),
                                   shape=(M, M))]
    return sum(matrices)

def plot_velocity_umap_Bergen2020(adata, use_full_posterior = True, n_neighbours = None,
                                  plotting_kwargs = None, save = False, spliced_key = 'Ms'):
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
            adata.uns['velocity_graph'] = compute_velocity_graph_Bergen2020(adata,
                                                                            full_posterior = True,
                                                                            n_neighbours = n_neighbours,
                                                                            spliced_key = spliced_key)          
        elif use_full_posterior and 'velocity_posterior' not in adata.uns.keys():
            print('Full velocity posterior not found, using expectation value to calculate velocity graph')
            adata.uns['velocity_graph'] = compute_velocity_graph_Bergen2020(adata,
                                                                            full_posterior = False,
                                                                            n_neighbours = n_neighbours,
                                                                            spliced_key = spliced_key)
        elif not use_full_posterior:
            print('Using velocity expectation value to calculate velocity graph.')
            adata.uns['velocity_graph'] = compute_velocity_graph_Bergen2020(adata,
                                                                            full_posterior = False,
                                                                            n_neighbours = n_neighbours,
                                                                            spliced_key = spliced_key)
        
        scv.pl.velocity_embedding_stream(adata, basis='umap', save = save, **plotting_kwargs)

def get_training_data(adata, remove_clusters = None, cells_per_cluster = 100000,
                         cluster_column = 'clusters', min_shared_counts = 20, n_var_genes = 2000,
                         n_pcs=30, n_neighbors=30):
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
    print('Keeping at most ' + str(N) + ' cells per cluster')
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
    print("Saving raw counts in adata.layers['spliced_raw'] and adata.layers['unspliced_raw']")
    adata.layers['spliced_raw'] = adata.layers['spliced']
    adata.layers['unspliced_raw'] = adata.layers['unspliced']
    scv.pp.filter_and_normalize(adata, min_shared_counts=min_shared_counts, n_top_genes=n_var_genes)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
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
    # Converts mean and sd for Gamma distribution into parameter
    return mu**2/sd**2

def G_b(mu, sd):
    # Converts mean and sd for Gamma distribution into beta parameter
    return mu/sd**2
#

####------------RNAvelocity model equations -------------------- ######

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

def mu_mRNA_discreteAlpha_OnePlate2(alpha, beta, gamma, tau, u0, s0):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    latent time and initial states'''
    
    mu_u = u0*torch.exp(-beta*tau) + (alpha/beta)* (1 - torch.exp(-beta*tau))
    mu_s = (s0*torch.exp(-gamma*tau) + 
    alpha/gamma * (1 - torch.exp(-gamma*tau)) +
    (alpha - beta * u0)/(gamma - beta + 10**(-5)) * (torch.exp(-gamma*tau) - torch.exp(-beta*tau)))

    return torch.stack([mu_u, mu_s], axis = 2)

def mu_mRNA_discreteAlpha_OnePlate3(alpha, beta, gamma, tau, u0, s0, u_detection_factor_g):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    latent time and initial states'''
    
    mu_u = u0*torch.exp(-beta*tau) + (alpha/beta)* (1 - torch.exp(-beta*tau))
    mu_s = (s0*torch.exp(-gamma*tau) + 
    alpha/gamma * (1 - torch.exp(-gamma*tau)) +
    (alpha - beta * u0)/(gamma - beta + 10**(-5)) * (torch.exp(-gamma*tau) - torch.exp(-beta*tau)))

    return torch.stack([mu_u, mu_s], axis = 2)

def mu_mRNA_discreteAlpha_OnePlate_MultiLineage(alpha, beta, gamma, tau, u0, s0):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    latent time and initial states'''
    
    mu_u = u0*torch.exp(-beta*tau) + (alpha/beta)* (1 - torch.exp(-beta*tau))
    mu_s = (s0*torch.exp(-gamma*tau) + 
    alpha/gamma * (1 - torch.exp(-gamma*tau)) +
    (alpha - beta * u0)/(gamma - beta + 10**(-5)) * (torch.exp(-gamma*tau) - torch.exp(-beta*tau)))

    return torch.stack([mu_u, mu_s], axis = -1)

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

def c(x, m = 10**(-100)):
    return torch.clip(x, min = m)

def mu_mRNA_continuousAlpha_withPlates(alpha, beta, gamma, tau, u0, s0, delta_alpha, lam):
    ''' Calculates expected value of spliced and unspliced counts as a function of rates, latent time, initial states,
    difference to transcription rate in previous state and rate of exponential change process between states.'''
    
    mu_u = u0*torch.exp(-beta*tau) + (alpha/beta)* (1 - torch.exp(-beta*tau)) + delta_alpha/(beta-lam+10**(-5))*(torch.exp(-beta*tau) - torch.exp(-lam*tau))
    mu_s = (s0*torch.exp(-gamma*tau) + 
    alpha/gamma * (1 - torch.exp(-gamma*tau)) +
    (alpha - beta * u0)/(gamma - beta+10**(-5)) * (torch.exp(-gamma*tau) - torch.exp(-beta*tau)) +
    (delta_alpha*beta)/((beta - lam+10**(-5))*(gamma - beta+10**(-5))) * (torch.exp(-beta*tau) - torch.exp(-gamma*tau))-
    (delta_alpha*beta)/((beta - lam+10**(-5))*(gamma - lam+10**(-5))) * (torch.exp(-lam*tau) - torch.exp(-gamma*tau)))

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

def mu_mRNA_continousAlpha_globalTime_twoStates_OnePlate(alpha_ON, alpha_OFF, beta, gamma, T_c, T_gON, T_gOFF, Zeros):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    global latent time, initial states and global switch times between two states'''
    n_cells = T_c.shape[-2]
    n_genes = alpha_ON.shape[-1]
    tau = torch.clip(T_c - T_gON, min = 10**(-5))
    t0 = T_gOFF - T_gON
    # Transcription rate in each cell for each gene:
    boolean = (tau < t0).reshape(n_cells, n_genes)
    alpha_cg = alpha_ON*boolean + alpha_OFF*~boolean
    # Time since changepoint for each cell and gene:
    tau_cg = tau*boolean + (tau - t0)*~boolean
    # Initial condition for each cell and gene:
    initial_state = mu_mRNA_discreteAlpha_OnePlate2(alpha_ON, beta, gamma, t0, Zeros, Zeros)
    u0_g = 10**(-5) + ~boolean*initial_state[:,:,0]
    s0_g = 10**(-5) + ~boolean*initial_state[:,:,1]
    # Unspliced and spliced count variance for each gene in each cell:
    return torch.clip(mu_mRNA_discreteAlpha_OnePlate2(alpha_cg, beta, gamma, tau_cg, u0_g, s0_g), min = 10**(-5))


def mu_mRNA_discreteAlpha_globalTime_twoStates_OnePlate2(alpha_ON, alpha_OFF, beta, gamma, T_c, T_gON, T_gOFF, Zeros):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    global latent time, initial states and global switch times between two states'''
    n_cells = T_c.shape[-2]
    n_genes = alpha_ON.shape[-1]
    tau = torch.clip(T_c - T_gON, min = 10**(-5))
    t0 = T_gOFF - T_gON
    # Transcription rate in each cell for each gene:
    boolean = (tau < t0).reshape(n_cells, n_genes)
    alpha_cg = alpha_ON*boolean + alpha_OFF*~boolean
    # Time since changepoint for each cell and gene:
    tau_cg = tau*boolean + (tau - t0)*~boolean
    # Initial condition for each cell and gene:
    initial_state = mu_mRNA_discreteAlpha_OnePlate2(alpha_ON, beta, gamma, t0, Zeros, Zeros)
    u0_g = 10**(-5) + ~boolean*initial_state[:,:,0]
    s0_g = 10**(-5) + ~boolean*initial_state[:,:,1]
    # Unspliced and spliced count variance for each gene in each cell:
    return torch.clip(mu_mRNA_discreteAlpha_OnePlate2(alpha_cg, beta, gamma, tau_cg, u0_g, s0_g), min = 10**(-5))

def mu_mRNA_discreteAlpha_globalTime_twoStates_OnePlate3(alpha_ON, alpha_OFF, beta, gamma,
                                                         T_c, T_gON, T_gOFF, Zeros, u_detection_factor_g):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    global latent time, initial states and global switch times between two states'''
    n_cells = T_c.shape[-2]
    n_genes = alpha_ON.shape[-1]
    tau = torch.clip(T_c - T_gON, min = 10**(-5))
    t0 = T_gOFF - T_gON
    # Transcription rate in each cell for each gene:
    boolean = (tau < t0).reshape(n_cells, n_genes)
    alpha_cg = alpha_ON*boolean + alpha_OFF*~boolean
    # Time since changepoint for each cell and gene:
    tau_cg = tau*boolean + (tau - t0)*~boolean
    # Initial condition for each cell and gene:
    initial_state = mu_mRNA_discreteAlpha_OnePlate2(alpha_ON, beta, gamma, t0, Zeros, Zeros, u_detection_factor_g)
    u0_g = 10**(-5) + ~boolean*initial_state[:,:,0]
    s0_g = 10**(-5) + ~boolean*initial_state[:,:,1]
    # Unspliced and spliced count variance for each gene in each cell:
    return torch.clip(mu_mRNA_discreteAlpha_OnePlate2(alpha_cg, beta, gamma, tau_cg, u0_g, s0_g, u_detection_factor_g), min = 10**(-5))

def mu_mRNA_discreteAlpha_globalTime_twoStates_OnePlate_MultiLineage(alpha_ON, alpha_OFF, beta, gamma, T_c, T_gON, T_gOFF, Zeros):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    global latent time, initial states and global switch times between two states'''
    n_cells = T_c.shape[-2]
    n_genes = alpha_ON.shape[-1]
    n_lineages = T_gON.shape[-3]
    tau = torch.clip(T_c - T_gON, min = 10**(-5))
    t0 = T_gOFF - T_gON
    # Transcription rate in each cell for each gene:
    boolean = (tau < t0).reshape(n_lineages, n_cells, n_genes)
    alpha_cg = alpha_ON*boolean + alpha_OFF*~boolean
    # Time since changepoint for each cell and gene:
    tau_cg = tau*boolean + (tau - t0)*~boolean
    # Initial condition for each cell and gene:
    initial_state = mu_mRNA_discreteAlpha_OnePlate_MultiLineage(alpha_ON, beta, gamma, t0, Zeros, Zeros)
    u0_g = 10**(-5) + ~boolean*initial_state[:,:,:,0]
    s0_g = 10**(-5) + ~boolean*initial_state[:,:,:,1]
    # Unspliced and spliced count variance for each gene in each cell:
    return torch.clip(mu_mRNA_discreteAlpha_OnePlate_MultiLineage(alpha_cg, beta, gamma, tau_cg, u0_g, s0_g), min = 10**(-5))

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

def mu_mRNA_continousAlpha_localTime_twoStates_OnePlate(alpha_ON, alpha_OFF, beta, gamma, lam_gi, T_cg, T_gOFF, Zeros):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    global latent time, initial states and global switch times between two states'''
    n_cells = T_cg.shape[-2]
    n_genes = alpha_ON.shape[-1]
    tau = T_cg
    t0 = T_gOFF
    # Transcription rate in each cell for each gene:
    boolean = (tau < t0).reshape(n_cells, n_genes)
    alpha_cg = alpha_ON*boolean + alpha_OFF*~boolean
    # Time since changepoint for each cell and gene:
    tau_cg = tau*boolean + (tau - t0)*~boolean
    # Initial condition for each cell and gene:
    lam_g = ~boolean*lam_gi[:,1] + boolean*lam_gi[:,0]
    initial_state = mu_mRNA_continuousAlpha_withPlates(alpha_ON, beta, gamma, t0,
                                                       Zeros, Zeros, alpha_ON-alpha_OFF, lam_gi[:,0])
    initial_alpha = mu_alpha(alpha_ON, alpha_OFF, t0, lam_gi[:,0])
    u0_g = 10**(-5) + ~boolean*initial_state[:,:,0]
    s0_g = 10**(-5) + ~boolean*initial_state[:,:,1]
    delta_alpha = ~boolean*initial_alpha*(-1) + boolean*alpha_ON*(1)
    alpha_0 = alpha_OFF + ~boolean*initial_alpha
    # Unspliced and spliced count variance for each gene in each cell:
    mu_RNAvelocity = torch.clip(mu_mRNA_continuousAlpha_withPlates(alpha_cg, beta, gamma, tau_cg,
                                                         u0_g, s0_g, delta_alpha, lam_g), min = 10**(-5))
    return mu_RNAvelocity

def mu_mRNA_continousAlpha_globalTime_twoStates_OnePlate2(alpha_ON, alpha_OFF, beta, gamma, lam_gi, T_c, T_gON, T_gOFF, Zeros):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    global latent time, initial states and global switch times between two states'''
    n_cells = T_c.shape[-2]
    n_genes = alpha_ON.shape[-1]
    tau = torch.clip(T_c - T_gON, min = 10**(-5))
    t0 = T_gOFF - T_gON
    # Transcription rate in each cell for each gene:
    boolean = (tau < t0).reshape(n_cells, n_genes)
    alpha_cg = alpha_ON*boolean + alpha_OFF*~boolean
    # Time since changepoint for each cell and gene:
    tau_cg = tau*boolean + (tau - t0)*~boolean
    # Initial condition for each cell and gene:
    lam_g = ~boolean*lam_gi[:,1] + boolean*lam_gi[:,0]
    initial_state = mu_mRNA_continuousAlpha_withPlates(alpha_ON, beta, gamma, t0,
                                                       Zeros, Zeros, alpha_ON-alpha_OFF, lam_gi[:,0])
    initial_alpha = mu_alpha(alpha_ON, alpha_OFF, t0, lam_gi[:,0])
    u0_g = 10**(-5) + ~boolean*initial_state[:,:,0]
    s0_g = 10**(-5) + ~boolean*initial_state[:,:,1]
    delta_alpha = ~boolean*initial_alpha*(-1) + boolean*alpha_ON*(1)
    alpha_0 = alpha_OFF + ~boolean*initial_alpha
    # Unspliced and spliced count variance for each gene in each cell:
    mu_RNAvelocity = torch.clip(mu_mRNA_continuousAlpha_withPlates(alpha_cg, beta, gamma, tau_cg,
                                                         u0_g, s0_g, delta_alpha, lam_g), min = 10**(-5))
#     mu_Alpha = mu_alpha(alpha_cg, alpha_0, tau_cg,lam_g)
    return mu_RNAvelocity

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

def mu_mRNA_discreteAlpha_withPlates(alpha, beta, gamma, tau, u0, s0):
    '''Calculates expected value of spliced and unspliced counts as a function of rates,
    latent time and initial states'''
    
    mu_u = u0*torch.exp(-beta*tau) + (alpha/beta)* (1 - torch.exp(-beta*tau))
    mu_s = (s0*torch.exp(-gamma*tau) + 
    alpha/gamma * (1 - torch.exp(-gamma*tau)) +
    (alpha - beta * u0)/(gamma - beta + 10**(-5)) * (torch.exp(-gamma*tau) - torch.exp(-beta*tau)))

    return torch.stack([mu_u, mu_s], axis = -1)

def mu_mRNA_discreteModularAlpha_localTime(
    A_mgON, A_mgOFF, beta_g, gamma_g, T_cm, T_mOFF, lam, Zeros):    
    '''Calculates expected value of spliced and unspliced counts summed over multiple modules.
    Needs rates, local latent time, module activation rate and steady-state transcription rate.'''
    n_cells = T_cm.shape[-3]
    n_modules = A_mgON.shape[-2]
    n_genes = A_mgON.shape[-1]
    
    # Time since switching on (clipped to positive values):
    tau_cm = T_cm
    # Switch off point:
    t0_m = T_mOFF
    # Module state in each cell and transition:
    boolean_cm = (tau_cm < t0_m).reshape(n_cells, n_modules)
    # Sum over modules activated in each transition to get
    # total unspliced + spliced counts in each cell:
    mu_cg = torch.stack([Zeros, Zeros], axis = -1)
    
    for m in range(n_modules):
        # Transcription rate in each cell for each gene:
        alpha_cg = A_mgON[m,:].unsqueeze(0) * boolean_cm[:,m].unsqueeze(-1) + \
                    A_mgOFF * ~boolean_cm[:,m].unsqueeze(-1)
        # Time since changepoint for each cell:
        tau_c = tau_cm[:,m,:]*boolean_cm[:,m].unsqueeze(-1) + \
                    (tau_cm[:,m,:] - t0_m[:,m,:])*~boolean_cm[:,m].unsqueeze(-1)
        # Initial condition for each cell and gene:
        initial_state = mu_mRNA_discreteAlpha_withPlates(A_mgON[m,:], beta_g, gamma_g,
                                                                     t0_m[:,m,:], Zeros, Zeros)
        
        u0_cg = A_mgOFF + ~boolean_cm[:,m].unsqueeze(-1)*initial_state[...,0]
        s0_cg = A_mgOFF + ~boolean_cm[:,m].unsqueeze(-1)*initial_state[...,1]
        mu_cmg = mu_mRNA_discreteAlpha_withPlates(alpha_cg, beta_g, gamma_g,
                                                              tau_c, u0_cg, s0_cg)
        
        mu_cg = mu_cg + mu_cmg
    return torch.clip(mu_cg, min = 10**(-5))

def mu_mRNA_discreteModularAlpha_localTime_4States(
    A_mgON, A_mgOFF, beta_g, gamma_g, T_mOFF, T_cmON, T_cmOFF, I_cm, lam, Zeros):    
    '''Calculates expected value of spliced and unspliced counts summed over multiple modules.
    Needs rates, local latent time, module activation rate and steady-state transcription rate.'''
    n_cells = T_cmON.shape[-3]
    n_modules = A_mgON.shape[-2]
    n_genes = A_mgON.shape[-1]
    # Sum over modules activated in each transition to get
    # total unspliced + spliced counts in each cell:
    mu_cg = torch.stack([Zeros, Zeros], axis = -1)    
    for m in range(n_modules):
        # (Also sum over possible activation states of modules given by I_cm)
        ## OFF STATE ###
        mu_cmg = torch.stack([Zeros, Zeros], axis = -1)
        ### ON STATE ###
        mu_cmg += I_cm[:,m, 1].unsqueeze(-1).unsqueeze(-1) * torch.stack([(A_mgON[m,:]/beta_g).repeat([n_cells,1]),
                                              (A_mgON[m,:]/gamma_g).repeat([n_cells,1])], axis = -1)
        ### Induction STATE ###
        mu_cmg += I_cm[:,m, 2].unsqueeze(-1).unsqueeze(-1) * mu_mRNA_discreteAlpha_withPlates(A_mgON[m,:], beta_g, gamma_g,
                                                      T_cmON[:,m,:], Zeros, Zeros)
        ### Repression STATE ###
        # Initial condition for each cell and gene:
        initial_state = mu_mRNA_discreteAlpha_withPlates(A_mgON[m,:], beta_g, gamma_g,
                                                                     T_mOFF[:,m,:], Zeros, Zeros)
        mu_cmg += I_cm[:,m, 3].unsqueeze(-1).unsqueeze(-1) * mu_mRNA_discreteAlpha_withPlates(A_mgOFF, beta_g, gamma_g,
                                                              T_cmOFF[:,m,:], initial_state[...,0], initial_state[...,0])
        
        mu_cg = mu_cg + mu_cmg
    return torch.clip(mu_cg, min = 10**(-5))