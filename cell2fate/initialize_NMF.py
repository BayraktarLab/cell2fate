import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib as mpl
import seaborn as sns
from scipy import sparse
from matplotlib import rcParams
import random

### Step 1.0 - select subset of data ###
def subset_cells(
    adata,
    cells_per_category=5000,
    stratify_category_key='sample',
):
    
    adata.obs['_cell_index'] = np.arange(adata.n_obs)
    subset_ind = list()
    
    for ct in adata.obs[stratify_category_key].unique():
        ind = adata.obs[stratify_category_key] == ct
        subset_ind_ = adata.obs['_cell_index'][ind]
        n_samples = np.min((len(subset_ind_), cells_per_category))
        subset_ind = subset_ind + list(np.random.choice(subset_ind_, size=n_samples, replace=False))
    
    return adata[subset_ind, :].copy()

### Step 2.0 - cluster genes using PCs ###
def rescale_distribution(dist, n_factors):

        dist = (
        dist 
        * (dist > np.quantile(dist, 0.01, axis=1).reshape((n_factors, 1))) 
        + 0.01 
        - np.quantile(dist, 0.01, axis=1).reshape((n_factors, 1)) 
        * (dist > np.quantile(dist, 0.01, axis=1).reshape((n_factors, 1)))
        )
        dist = (dist.T / dist.max(1)).T
        #dist = (dist.T - dist.min(1) + 0.001).T

        return dist
    
# max_min_sampling copied from https://github.com/dpeerlab/Palantir/blob/master/src/palantir/core.py#L121-L163
def max_min_sampling(data, n_waypoints):
    """Function for max min sampling of waypoints

    :param data: Data matrix along which to sample the waypoints,
                 usually diffusion components
    :param n_waypoints: Number of waypoints to sample
    :param num_jobs: Number of jobs for parallel processing
    :return: pandas Series reprenting the sampled waypoints
    """

    waypoint_set = list()
    no_iterations = int((n_waypoints) / data.shape[1])

    # Sample along each component
    N = data.shape[0]
    for ind in data.columns:
        # Data vector
        vec = np.ravel(data[ind])

        # Random initialzlation
        iter_set = random.sample(range(N), 1)

        # Distances along the component
        dists = np.zeros([N, no_iterations])
        dists[:, 0] = abs(vec - data[ind].values[iter_set])
        for k in range(1, no_iterations):
            # Minimum distances across the current set
            min_dists = dists[:, 0:k].min(axis=1)

            # Point with the maximum of the minimum distances is the new waypoint
            new_wp = np.where(min_dists == min_dists.max())[0][0]
            iter_set.append(new_wp)

            # Update distances
            dists[:, k] = abs(vec - data[ind].values[new_wp])

        # Update global set
        waypoint_set = waypoint_set + iter_set

    # Unique waypoints
    waypoints = data.index[waypoint_set].unique()

    return waypoints

def find_waypoint_gene_clusters(
    adata_neighbours,
    k='aver_norm',
    n_factors=300,
    margin_of_error=20,
    n_neighbors=15,
    labels_key=None, 
    label_filter=None,
    verbose=True,
    plot = False
):
    
    init_n_factors = n_factors
    
    if labels_key is not None:
        from cell2location.cluster_averages import compute_cluster_averages
        aver = compute_cluster_averages(adata_neighbours, labels_key, use_raw=False)
        if label_filter is not None:
            aver = aver.loc[:, label_filter]
        gene_rates = {'aver': aver}
        for k in ['aver']:
            gene_rates[k+'_norm'] = (
                gene_rates[k].T / (
                    gene_rates[k].sum(1) + np.random.gamma(1e+2, 1e-8, size=gene_rates[k].sum(1).shape)
                )
            ).T
    else:
        # Use PCs
        aver = pd.DataFrame(
            adata_neighbours.varm['PCs'],
            index=adata_neighbours.var_names,
            columns=[f'PC_{i+1}' for i in range(adata_neighbours.varm['PCs'].shape[1])],
        )
        gene_rates = {'aver': aver}
        for k in ['aver']:
            gene_rates[k+'_norm'] = (gene_rates[k].T / gene_rates[k].abs().max(1)).T

    

    # compute KNN for genes and cluster genes by bursting rates at meta-cells
    adata_neighbours_g = adata_neighbours[0:10,:].copy().T
    # adata_neighbours_g = adata_neighbours_g[adata_neighbours.uns['mod']['gene_names'], :]

    adata_neighbours_g.obsm[k] = gene_rates[k].values
    adata_neighbours_g.obs['total_counts'] = np.log10(np.array(adata_neighbours_g.X.mean(1)).flatten())

    sc.pp.neighbors(adata_neighbours_g, n_neighbors=n_neighbors, 
                        use_rep=k, metric='correlation')
    sc.tl.umap(adata_neighbours_g, min_dist = 0.1, spread = 2.5)

    X_pd = pd.DataFrame(
            adata_neighbours_g.obsm[k],
            columns=[
                f"{k}_{i}" for i in range(adata_neighbours_g.obsm[k].shape[1])
            ],
            index=adata_neighbours_g.obs_names,
    )

    waypoints = max_min_sampling(data=X_pd, n_waypoints=n_factors)
    total_steps = 0
    while (abs(len(waypoints) - n_factors) > margin_of_error) and (total_steps <= 10):
        if verbose:
            print(len(waypoints), init_n_factors)
        waypoints = max_min_sampling(data=X_pd, n_waypoints=np.round(init_n_factors))
        init_n_factors += 1.0 * (n_factors - len(waypoints))
        total_steps += 1
    n_factors = len(waypoints)

    # plot
    adata_neighbours_g.obs["is_waypoint"] = adata_neighbours_g.obs_names.isin(waypoints)
    adata_neighbours_g.obs["is_waypoint_size"] = np.array(
        [10 if x else 1 for x in adata_neighbours_g.obs["is_waypoint"]]
    )
    adata_neighbours_g.obs["is_waypoint"] = adata_neighbours_g.obs["is_waypoint"].astype("category")
    
    if plot:
        with mpl.rc_context({'figure.figsize': [6, 6]}):
            sns.scatterplot(adata_neighbours_g.obsm["X_umap"][:,0], 
                        adata_neighbours_g.obsm["X_umap"][:,1], 
                        hue=adata_neighbours_g.obs['is_waypoint'], 
                        s=adata_neighbours_g.obs['is_waypoint_size']);
            plt.show();
    
    return adata_neighbours_g, n_factors

def compute_mu_std(X):
        
    mu = np.array(X.mean(0))
    mu_sq = mu ** 2
    X = X.copy()
    X.data = X.data ** 2
    sq_mu = np.array(X.mean(0))
    std = np.sqrt(sq_mu - mu_sq) + 1e-8
    
    return mu, std

def compute_w_initial_waypoint(
    adata_neighbours,
    adata_neighbours_g,
    n_factors,
    k='aver_norm',
    scale=False, tech_category_key=None,
    use_x=True, layer=None,
    knn_smoothing=False,
    scale_max_value=10,
    plot = False,
    verbose = False
):
    
    if use_x and layer is None:
        X = adata_neighbours[:, adata_neighbours_g.obs_names].X.copy()
    elif layer is not None:
        X = adata_neighbours[:, adata_neighbours_g.obs_names].layers[layer].copy()
    
    waypoints = adata_neighbours_g.obs_names[adata_neighbours_g.obs["is_waypoint"]]
    
    # Scale with no HVG selection
    if scale:
        if tech_category_key is None:
#             print(X)
            mu, std = compute_mu_std(X)
            X = X.multiply(1 / std).minimum(scale_max_value)
        else:
            for tech in adata_subset.obs[tech_category_key].unique():
                mu, std = compute_mu_std(X[adata_subset.obs[tech_category_key] == tech, :])
                X[adata_subset.obs[tech_category_key] == tech, :] = (
                    X[adata_subset.obs[tech_category_key] == tech, :]
                    .multiply(1 / std).minimum(scale_max_value)
                )
    
    w_init_dict = dict()
    for i, k in enumerate([k]):
        # Compute sum gene expression across neighbours of each waypoint gene
        w_init = np.array(
            X.dot(
                adata_neighbours_g.obsp["connectivities"][adata_neighbours_g.obs_names.isin(waypoints), :].T
            ).toarray() 
            / adata_neighbours_g.obsp["connectivities"][adata_neighbours_g.obs_names.isin(waypoints), :].sum(1).T)
        # Apply KNN smoothing if necessary
        if knn_smoothing:
            w_init = np.array(
                adata_neighbours.obsp["connectivities"].dot(w_init)
                / adata_neighbours.obsp["connectivities"].sum(1))
        # Convert to dataframe with correct labels
        w_init = pd.DataFrame(
            w_init, 
            index=adata_neighbours.obs_names, 
            columns=[f'factor_{i}' for i in range(n_factors)]
        )
        # Rescale distribution, setting maximum to 1 and bottom 1% of values to 0
        w_init_dict[f'cell_factors_w_cf'] = rescale_distribution(w_init.T, n_factors=n_factors).T

    adata_neighbours.uns['mod_init'] = dict()
    adata_neighbours.uns['mod_init']['initial_values'] = {
        'w_init': w_init_dict,

    }
    if plot:
        for i, k in enumerate(['aver']):
            plt.hist(w_init_dict[f'cell_factors_w_cf'].values.flatten(), bins=500);
            plt.show();

    return adata_neighbours

### Step 1.1 - compute PCs by apply standard workflow with a few exceptions ##
def compute_pcs_knn_umap(
    adata_subset, 
    tech_category_key=None, plot_category_keys=list(), 
    scale_max_value=10, n_comps=100, n_neighbors=15,
    stratify_category_key = None, plot = True
):
    adata_subset.obs['total_counts'] = np.array(adata_subset.X.sum(1)).flatten()
    adata_subset.layers['counts'] = adata_subset.X.copy()
    # No normalisation by total count
    sc.pp.log1p(adata_subset)
    # Scale with no HVG selection
    if tech_category_key is None:
        sc.pp.scale(adata_subset, max_value=scale_max_value)
    else:
        for tech in adata_subset.obs[tech_category_key].unique():
            mu, std = compute_mu_std(adata_subset[adata_subset.obs[tech_category_key] == tech].X)
            adata_subset[adata_subset.obs[tech_category_key] == tech].X = (
                np.minimum((adata_subset[adata_subset.obs[tech_category_key] == tech].X - mu) / std, scale_max_value)
            )
    # A lot of PC dimensions
    sc.tl.pca(adata_subset, svd_solver='arpack', n_comps=n_comps, use_highly_variable=False)
    # Plot PCs to confirm that PC1 is indeed linked to total count
    # sc.pl.pca(adata_subset, color=['total_counts'],
    #           components=['1,2', '2,3', '4,5'],
    #           color_map = 'RdPu', ncols = 3, legend_loc='on data',
    #           legend_fontsize=10)
    if plot:
        plt.hist2d(adata_subset.obsm['X_pca'][:, 0].flatten(),
                   adata_subset.obs['total_counts'].values.flatten(),
                   bins=200,
                   norm=mpl.colors.LogNorm());
        plt.xlabel('PC 1');
        plt.ylabel('Total RNA count');
        plt.show()

    # Remove PC1
    adata_subset.obsm['X_pca'] = adata_subset.obsm['X_pca'][:, 1:]
    adata_subset.varm['PCs'] = adata_subset.varm['PCs'][:, 1:]

    # compute KNN and UMAP to see how well this represents the dataset
    sc.pp.neighbors(adata_subset, n_neighbors=n_neighbors)
    sc.tl.umap(adata_subset, min_dist = 0.2, spread = 0.8)
    
    if plot:
    # Plot UMAP
        sc.pl.umap(adata_subset, color=[stratify_category_key] + plot_category_keys,
                   color_map = 'RdPu', ncols = 3, #legend_loc='on data',
                   legend_fontsize=10)
    return adata_subset

from scipy.optimize import linear_sum_assignment
def align_plot_stability(fac1, fac2, name1, name2, align=True, return_aligned=False,
                         title=''):
        r"""Align columns between two np.ndarrays using scipy.optimize.linear_sum_assignment,
            then plot correlations between columns in fac1 and fac2, ordering fac2 according to alignment

        :param fac1: np.ndarray 1, factors in columns
        :param fac2: np.ndarray 2, factors in columns
        :param name1: axis x name
        :param name2: axis y name
        :param align: boolean, match columns in fac1 and fac2 using linear_sum_assignment?
        """

        corr12 = np.corrcoef(fac1, fac2, False)
        ind_top = np.arange(0, fac1.shape[1])
        ind_right = np.arange(0, fac2.shape[1]) + fac1.shape[1]
        corr12 = corr12[ind_top, :][:, ind_right]
        corr12[np.isnan(corr12)] = -1

        if align:
            img = corr12[:, linear_sum_assignment(2 - corr12)[1]]
        else:
            img = corr12

        plt.imshow(img)

        plt.title(f"{title}\n{name1} vs {name2}")
        plt.xlabel(name2)
        plt.ylabel(name1)

        plt.tight_layout()

        if return_aligned:
            return corr12, linear_sum_assignment(2 - corr12)[1]
        
def find_stable_waypoint_gene_clusters(
    adata_neighbours_,
    k='aver_norm',
    n_factors=300,
    n_neighbors=20,
    labels_key='cell_type',
    n_repeats=5,
    cluster_max_cutoff=0.2,
    margin_of_error=20,
    bootstrap_p=0.9,
    verbose=True,
):
    np.random.seed(1)
    adata_list = list()
    for i in range(n_repeats):
        from sklearn.model_selection import train_test_split
        ind_, _ = train_test_split(
            np.arange(adata_neighbours.n_obs), 
            test_size=1 - bootstrap_p, 
            shuffle=True, 
            stratify=adata_neighbours.obs[labels_key],
        )
        adata_neighbours_ = adata_neighbours[ind_, :].copy()

        adata_neighbours_g, _ = find_waypoint_gene_clusters(
            adata_neighbours=adata_neighbours_,
            k=k,
            n_factors=n_factors,
            margin_of_error=margin_of_error,
            n_neighbors=n_neighbors,
            labels_key=labels_key,
            verbose=True,
        )
        adata_list = adata_list + [adata_neighbours_g.copy()]

    cluster_max = np.zeros((adata_list[0].obs["is_waypoint"].values.astype(bool).sum()))
    for i in range(len(adata_list) - 1):
        clusters_0 = np.array(adata_list[0].obsp["connectivities"][
            adata_list[0].obs["is_waypoint"].values.astype(bool), :].T.toarray())
        clusters_1 = np.array(adata_list[i+1].obsp["connectivities"][
            adata_list[i+1].obs["is_waypoint"].values.astype(bool), :].T.toarray())
        corr01, assignment = align_plot_stability(
            fac1=clusters_0,  
            fac2=clusters_1, 
            name1='0', 
            name2=f'{i+1}', 
            title='Bootstrap step',
            align=True, return_aligned=True
        )
        if verbose:
            plt.show()
        else:
            plt.clear()
            plt.close()
        cluster_max = np.array([cluster_max, corr01.max(1)]).mean(0)

    plt.hist(cluster_max, bins=20);
    plt.show();

    waypoints = adata_list[0].obs_names[adata_list[0].obs["is_waypoint"].values.astype(bool)]
    waypoints = waypoints[cluster_max > cluster_max_cutoff]
    n_factors = len(waypoints)
    
     # plot
    adata_neighbours_g = adata_list[0].copy()
    adata_neighbours_g.obs["is_waypoint"] = adata_neighbours_g.obs_names.isin(waypoints)
    adata_neighbours_g.obs["is_waypoint_size"] = np.array(
        [10 if x else 1 for x in adata_neighbours_g.obs["is_waypoint"]]
    )
    adata_neighbours_g.obs["is_waypoint"] = adata_neighbours_g.obs["is_waypoint"].astype("category")

    with mpl.rc_context({'figure.figsize': [6, 6]}):
        sns.scatterplot(adata_neighbours_g.obsm["X_umap"][:,0], 
                    adata_neighbours_g.obsm["X_umap"][:,1], 
                    hue=adata_neighbours_g.obs['is_waypoint'], 
                    s=adata_neighbours_g.obs['is_waypoint_size']);
        plt.show();
    
    return adata_neighbours_g, n_factors

def initialize_NMF(adata, n_factors = 30, counts_layer_key = False, stratify_category_key='clusters',
    tech_category_key=None, plot_category_keys=['clusters'], plot = False, cells_per_category=10**10,
                  verbose = False):
    '''Gets initial values for the weights in non-negative matrix factorization.'''

    adata_neighbours = adata.copy()

    if counts_layer_key:
        adata.X = adata.layers[counts_layer_key]

    adata_neighbours.uns['mod'] = dict()

    adata_neighbours.uns['mod']['gene_names'] = np.array(adata.var.index)

    ### Step 1.0 - select subset of data ###
    print('Subsetting data to ' + str(cells_per_category) + ' cells per cluster ...')
    np.random.seed(1)
    adata_subset = subset_cells(
        adata_neighbours, cells_per_category=cells_per_category, 
        stratify_category_key=stratify_category_key,
    )
    adata_neighbours = adata_subset.copy()

    ### Step 1.1 - compute PCs by applying standard workflow with a few exceptions ##
    print('Compute PCs ...')
    adata_subset = compute_pcs_knn_umap(
        adata_subset, 
        tech_category_key=tech_category_key, plot_category_keys=plot_category_keys, 
        scale_max_value=10, n_comps=n_factors, n_neighbors=25,
        stratify_category_key = stratify_category_key, plot = plot
    )

    ### Step 2.0 - cluster genes using PCs ###
    print('Cluster genes using PCs ...')
    adata_subset_g, n_factors = find_waypoint_gene_clusters(
        adata_subset,
        k='aver_norm',
        n_factors=n_factors,
        margin_of_error=20,
        n_neighbors=20,
        labels_key=None, 
        label_filter=None,
        verbose=True,
        plot = plot
    )

    adata_subset.X = sparse.csr_matrix(adata_subset.X)
    adata_subset_g.X = sparse.csr_matrix(adata_subset_g.X)

    ### Step 3.0 - compute initial values of cell loadings ###
    print('Compute initial values for cell loadings ...')
    adata_subset = adata_subset[adata_neighbours.obs_names, :].copy()
    adata_subset = compute_w_initial_waypoint(
        adata_subset, adata_subset_g, n_factors,
        scale=True, tech_category_key=tech_category_key,
        use_x=True,
        knn_smoothing=True,
        plot = plot,
        verbose = verbose
    )
    adata_neighbours.uns['mod_init'] = adata_subset.uns['mod_init'].copy()

    adata_subset.obs = adata_subset.obs.copy()  # fragmentation warnings
    adata_subset.obs[adata_subset.uns['mod_init']['initial_values']['w_init']['cell_factors_w_cf'].columns] = (
        adata_subset.uns['mod_init']['initial_values']['w_init']['cell_factors_w_cf']
    )
    adata_subset.obs = adata_subset.obs.copy()  # fragmentation warnings
    adata_subset.obsm['X_umap'] = adata_neighbours.obsm['X_umap'].copy()
    if plot:
        rcParams['figure.figsize'] = [3, 3]
        sc.pl.umap(adata_subset, 
                   color=plot_category_keys
                   + list(adata_subset.uns['mod_init']['initial_values']['w_init']['cell_factors_w_cf'].columns),
                   color_map = 'RdPu', gene_symbols='SYMBOL', use_raw=False, vmax='p99.9')

    waypoints = adata_subset_g.obs_names[adata_subset_g.obs["is_waypoint"].values.astype(bool)]

    init_vals = adata_neighbours.uns['mod_init']['initial_values']['w_init']
    init_vals = {k: v.loc[adata_neighbours.obs_names, :].values.astype('float32') for k, v in init_vals.items()}
    n_factors = init_vals[list(init_vals.keys())[0]].shape[1]
    print('Done.')
    return init_vals, n_factors