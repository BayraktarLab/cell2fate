import torch
import numpy as np
import anndata as ad
import cell2fate as c2f
import scanpy as sc
import matplotlib.pyplot as plt
import scvelo as scv
import os
from cell2fate._pyro_mixin import MyAutoHierarchicalNormalMessenger
from cell2fate._pyro_mixin import PyroTrainingPlan_ClippedAdamDecayingRate, PyroAggressiveTrainingPlan1

def dummy_iterator(start, end, step):
    current = start
    while current <= end:
        yield current
        current += step

        
def create_synthetic_anndata(n_cells=100, n_genes=100, n_clusters=3, high_var_gene_fraction=0.5, high_var_lambda_multiplier=1000):
    """
    Generate a synthetic AnnData object with spliced and unspliced counts, clusters, cluster_colors, UMAP coordinates, 
    additional gene statistics in .var (including mean and log1p mean counts), and cell statistics in .obs (including total counts and log1p total counts).
    This version introduces higher variability for a subset of genes.

    Parameters:
    - n_cells: Number of cells in the dataset.
    - n_genes: Number of genes in the dataset.
    - n_clusters: Number of clusters to simulate in the dataset.
    - high_var_gene_fraction: Fraction of genes to be made highly variable.
    - high_var_lambda_multiplier: Multiplier to increase the lambda parameter for highly variable genes.

    Returns:
    - An AnnData object with synthetic data, including clusters, cluster_colors, UMAP coordinates, gene statistics, and cell statistics.
    """
    np.random.seed(42)  # For reproducibility
    num_high_var_genes = int(n_genes * high_var_gene_fraction)

    # Lambda values for spliced and unspliced counts
    lambda_spliced = 5
    lambda_unspliced = 2

    # Initialize counts
    spliced_counts = np.zeros((n_cells, n_genes))
    unspliced_counts = np.zeros((n_cells, n_genes))

    # Assign higher lambda for a subset of genes
    for i in range(n_genes):
        if i < num_high_var_genes:
            lam_spliced = lambda_spliced * high_var_lambda_multiplier
            lam_unspliced = lambda_unspliced * high_var_lambda_multiplier
        else:
            lam_spliced = lambda_spliced
            lam_unspliced = lambda_unspliced

        spliced_counts[:, i] = np.random.poisson(lam=lam_spliced, size=n_cells)
        unspliced_counts[:, i] = np.random.poisson(lam=lam_unspliced, size=n_cells)

    total_counts_per_cell = np.sum(spliced_counts + unspliced_counts, axis=1)
    
    # Log-transformed total counts per cell
    log1p_total_counts_per_cell = np.log1p(total_counts_per_cell)
    
    # Total and mean counts for each gene (summing and averaging across cells)
    total_counts = np.sum(spliced_counts + unspliced_counts, axis=0)
    mean_counts = total_counts / n_cells
    log1p_total_counts = np.log1p(total_counts)
    log1p_mean_counts = np.log1p(mean_counts)
    
    # Create AnnData objectfig, ax = plt.subplots(1,1, figsize = (6, 4))

    genes = [f'Gene{i}' for i in range(n_genes)]
    cells = [f'Cell{i}' for i in range(n_cells)]

    adata = ad.AnnData(X=spliced_counts, 
                       obs={'cell_ids': cells},
                       var={'gene_ids': genes})
    adata.layers['spliced'] = spliced_counts
    adata.layers['unspliced'] = unspliced_counts

    # Simulate cluster assignment
    clusters = np.random.choice(range(n_clusters), size=n_cells)
    adata.obs['clusters'] = clusters
    
    for cluster in adata.obs['clusters']:
        cluster = str(cluster)

    # Generate synthetic UMAP coordinates
    umap_coords = np.random.normal(loc=0, scale=1, size=(n_cells, 2))
    adata.obsm['X_umap'] = umap_coords
    adata.obs["umap"]="Test"
    
    return adata


def test_cell2fate():
    
    # Define the path to save the model
    save_path = "./cell2fate_model_test/"
    
    # check if GPU is available and set accelerator accordingly
    if torch.cuda.is_available():
        use_gpu = True
        accelerator = "gpu"
    else:
        use_gpu = False
        accelerator = "cpu"

    # Create synthetic AnnData object
    adata = create_synthetic_anndata()
    
    # test remove any clusters if needed
    clusters_to_remove = []

    # test prepare data for training
    adata_train = c2f.utils.get_training_data(adata, cells_per_cluster=30, cluster_column='clusters', remove_clusters=clusters_to_remove, min_shared_counts=5, n_var_genes=50)
    
    # test determine the number of modules
    n_modules = c2f.utils.get_max_modules(adata_train)
    
    # test setup AnnData object for Cell2fate
    c2f.Cell2fate_DynamicalModel.setup_anndata(adata_train, spliced_label='spliced', unspliced_label='unspliced')
        
    # test initialize Cell2fate model
    mod = c2f.Cell2fate_DynamicalModel(adata_train, n_modules=n_modules)
    
    # test view setup of AnnData object
    mod.view_anndata_setup()

    # test train the model with one epoch
    mod.train(max_epochs=5, accelerator=accelerator)
    
    # test train the model with one epoch using a specific batch size
    mod.train(max_epochs=5, batch_size=10, accelerator=accelerator)
    
    #test robust optimization
    mod = c2f.utils.robust_optimization(mod,save_path, use_gpu = use_gpu, max_epochs = [1,2])
    
    # test amortized (currently experimental just for coverage of the else statement.)
    adata_train = c2f.utils.get_training_data(adata, cells_per_cluster=30, cluster_column='clusters', remove_clusters=clusters_to_remove, min_shared_counts=5, n_var_genes=50)
    c2f.Cell2fate_DynamicalModel_amortized.setup_anndata(adata_train, spliced_label='spliced', unspliced_label='unspliced')
    mod_amortised = c2f.Cell2fate_DynamicalModel_amortized(adata_train, n_modules=n_modules)
    mod_amortised.train(max_epochs=5, accelerator=accelerator)

    # test view history
    mod.view_history()
    
    # test save/load
    mod.save(save_path, overwrite=True, save_anndata=True)
    mod = mod.load(save_path)
    
    # test export posterior samples
    adata_posterior = mod.export_posterior(adata_train, sample_kwargs={"num_samples": 20, "batch_size": None, "use_gpu": use_gpu, 'return_samples': True})
    
    # test batch export posterior sampling
    adata_posterior = mod.export_posterior(adata_train, sample_kwargs={"num_samples": 20, "batch_size": 20, "use_gpu": use_gpu, 'return_samples': True})
    
    # test compute module summary statistics
    adata_posterior = mod.compute_module_summary_statistics(adata_posterior)
            
    # test plot module summary statistics
    mod.plot_module_summary_statistics(adata_posterior, save="test.png")
    mod.plot_module_summary_statistics_2(adata_posterior, [0,1], [0,1], marker_genes=["1","2"], marker_TFs=["1","2"], save=None)
    mod.plot_module_summary_statistics_2(adata_posterior, [0,1], [0,1],  marker_genes=["1","2"], marker_TFs=["1","2"],save="test.png")
    
    # test compare module activation
    mod.compare_module_activation(adata_posterior, chosen_modules=[1, 2, 3], save=None)
        
    # test compute and plot module velocity
    mod.compute_and_plot_module_velocity(adata_posterior,save="test.png", delete=False)
    mod.compute_and_plot_module_velocity(adata_posterior,save=None, delete=True)
    
    # test compute and plot total velocity
    mod.compute_and_plot_total_velocity(adata_posterior, save="test.png", delete=True)
    adata_with_velocity=mod.compute_and_plot_total_velocity(adata_posterior, save=None, delete=False)
    
    # test compute and plot total velocity using scvelo
    mod.compute_and_plot_total_velocity_scvelo(adata_posterior, save="test.png", delete=True)
    mod.compute_and_plot_total_velocity_scvelo(adata_posterior, save=None, delete=False)

    # test visualize module trajectories
    chosen_module = 0
    mod.visualize_module_trajectories(adata_posterior, chosen_module)
    mod.visualize_module_trajectories(adata_posterior, chosen_module, plotting_kwargs = {"color": 'clusters', 'legend_fontsize': 10, 'legend_loc': 'on data', 'dpi': 300, 'cmap': 'Greys', "layer": ["spliced","unspliced"]})

    #test plot_velocity_umap_Bergen2020
    c2f.utils.plot_velocity_umap_Bergen2020(adata_posterior, plotting_kwargs={}, spliced_key="spliced")
    del adata_posterior.uns["velocity_graph"]
    c2f.utils.plot_velocity_umap_Bergen2020(adata_posterior, plotting_kwargs={}, spliced_key="spliced")
    c2f.utils.plot_velocity_umap_Bergen2020(adata_posterior, use_full_posterior = False, plotting_kwargs={}, spliced_key="spliced")
    
    #test other utils functions
    c2f.utils.suppress_stdout()
    c2f.utils.multiplot_from_generator(dummy_iterator(0, 100, 10),1)
    
    # test plot technical variables
    mod.plot_technical_variables(adata_posterior, save="test.png")
    
    # test get top features for each module
    tab, all_results = mod.get_module_top_features(adata_posterior, p_adj_cutoff=0.01, background=adata.var_names)
    tab, all_results = mod.get_module_top_features(adata_posterior, species="Human", p_adj_cutoff=0.01, background=adata.var_names)
        
    # test plot top features
    mod.plot_top_features(adata_posterior, tab, [0,1])
    mod.plot_top_features(adata_posterior, tab, [0,1], save="test.png")

    
    # test plot genes
    mod.plot_genes(adata_posterior, ["1","2"], ["1","2"]) 
    mod.plot_genes(adata_posterior, ["1","2"], ["1","2"], save = "test.png") 
    
    # test example module activation
    mod.example_module_activation(adata_posterior, 1, time_max = 10, save ="test.png")
    
    # test quantile training and sampling
    adata_train = c2f.utils.get_training_data(adata, cells_per_cluster=30, cluster_column='clusters', remove_clusters=clusters_to_remove, min_shared_counts=5, n_var_genes=50)
    c2f.Cell2fate_DynamicalModel.setup_anndata(adata_train, spliced_label='spliced', unspliced_label='unspliced')
    mod = c2f.Cell2fate_DynamicalModel(adata_train, guide_class=MyAutoHierarchicalNormalMessenger, n_modules=n_modules)
    mod.train(batch_size = 50, max_epochs = 5, **{'training_plan' : PyroTrainingPlan_ClippedAdamDecayingRate},
         early_stopping = True, early_stopping_min_delta = 10**(-4),
         early_stopping_monitor = 'elbo_train', early_stopping_patience = 10, accelerator=accelerator)
    adata_quantile_posterior=mod.export_posterior_quantiles(adata_train,batch_size=50,use_gpu=use_gpu)
    
    # test agressive training plan
    adata_train = c2f.utils.get_training_data(adata, cells_per_cluster=30, cluster_column='clusters', remove_clusters=clusters_to_remove, min_shared_counts=5, n_var_genes=50)
    c2f.Cell2fate_DynamicalModel.setup_anndata(adata_train, spliced_label='spliced', unspliced_label='unspliced')
    mod = c2f.Cell2fate_DynamicalModel(adata_train, n_modules=n_modules)
    mod.train(batch_size = 50, max_epochs = 5, **{'training_plan' : PyroAggressiveTrainingPlan1}, accelerator=accelerator)
    
    # test export of posterior quantiles from amortized model
    adata_posterior_amortized = mod_amortised.export_posterior_quantiles(adata_train, use_gpu = use_gpu)
    