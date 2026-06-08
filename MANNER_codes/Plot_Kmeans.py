##Function for creating Kmeans plots

import sys
import pandas as pd
import numpy as np
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import anndata as ad
from anndata import AnnData, read_h5ad
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse
from scipy.sparse import issparse, vstack, csr_matrix
## convert sc.pp.neighbor output to edge_index format for GCN
from scipy.sparse import coo_matrix
##For scaling gene expression for heatmaps
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches  # Import mpatches for creating legend entries
import matplotlib.lines as mlines

import pickle
import os

import time

import matplotlib.gridspec as gridspec
#################################################################################


def plot_kmeans(data, n_cluster, out_dir, plot_title, save_name_prefix, fig_size = (10,8), 
                kmeans_var="data_tmp.X", x_cord_key="array_row",y_cord_key="array_col",
                dot_size = 0.3, invert_y = False, dpi=1000, titles = None, show_plot=False):
    # data: input data with gene expression to compute k-means
    # n_clusters: number of k-means clusters to form
    # out_dir: folder to save the plot
    # plot_title: title of the plot
    # save_name_prefix: front part of the file name of the plot
    # kmeans_var: variable from data_tmp to do KMeans on; data_tmp.X OR data_tmp.obsm["X_gcn"]
    # If show_plot = False, only save cluster assignments CSV without plotting

    # Ensure the directory exists
    os.makedirs(out_dir, exist_ok=True)

    data_tmp = data.copy()

    # Check if .X is sparse, and convert to dense if needed
    if issparse(data_tmp.X):
        print("data_tmp.X is sparse. Converting to dense...")
        data_tmp.X = data_tmp.X.toarray()
    else:
        print("data_tmp.X is already dense.")
    
    if "log1p" not in data_tmp.uns:
        print("log1p not found in .uns — applying log1p transformation to adata.X...")
        sc.pp.log1p(data_tmp)
    else:
        print("log1p transformation already applied — skipping.")


    # Evaluate the string input for `kmeans_var`
    try:
        kmeans_data = eval(kmeans_var, {"data_tmp": data_tmp})  # Evaluates kmeans_var safely within context
    except Exception as e:
        print(f"Error: Unable to evaluate kmeans_var `{kmeans_var}`. Exception: {e}")
        return
    
    # Perform K-Means clustering
    print("Running KMeans clustering...")
    data_tmp.obs["kmeans_clusters"] = KMeans(n_clusters=n_cluster, random_state=42).fit_predict(kmeans_data)

    # Ensure cluster labels are sorted numerically
    clusters = data_tmp.obs['kmeans_clusters'].astype(int)  # Convert to integers
    sorted_unique_clusters = np.sort(np.unique(clusters))  # Sort the unique cluster labels

    # Create a mapping from old cluster labels to new ordered labels
    cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_unique_clusters)}

    # Apply the mapping to ensure sequential cluster labels (0,1,2,3,... instead of 1,11,3,...)
    data_tmp.obs['kmeans_clusters'] = clusters.map(cluster_mapping) 

    ##save the cluster assignments as a .csv file of 2 cols (spot_id, cluster_label)
    # Save cluster assignments as a CSV
    cluster_df = pd.DataFrame({
        "spot_id": data_tmp.obs_names,
        "kmeans_clusters": data_tmp.obs["kmeans_clusters"]
    })
    file_name = os.path.join(out_dir, f"{save_name_prefix}_KMeans_{n_cluster}_cluster_assignments.csv")
    cluster_df.to_csv(file_name, index=False)
    print(f"Cluster assignments saved to {file_name}")

    if not show_plot:
        # Skip plotting to save memory
        del data_tmp
        return
    
    # Extract spatial coordinates using the specified keys
    x_cord = data_tmp.obs[x_cord_key].values.astype(float)
    y_cord = data_tmp.obs[y_cord_key].values.astype(float)

    # Convert ordered clusters to categorical labels
    cluster_codes = data_tmp.obs['kmeans_clusters'].values
    unique_clusters = np.unique(cluster_codes)
    print(f"Number of K-means clusters: {len(unique_clusters)}")

    # Define two colormaps and combine them to support more than 20 colors
    cmap1 = plt.get_cmap('tab20')   # First 20 colors
    cmap2 = plt.get_cmap('tab20b')  # Another set of 20 colors

    # Generate color palette
    if len(unique_clusters) <= 20:
        palette = [cmap1(i/20) for i in range(len(unique_clusters))]
    else:
        # For clusters greater than 20, combine the two colormaps
        palette = ([cmap1(i/20) for i in range(20)] + 
                   [cmap2(i/(len(unique_clusters)-20)) for i in range(len(unique_clusters)-20)])

    # Create a figure with grid spec layout to reserve space for legend
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])  # 4:1 ratio for plot vs legend

    # --- Plot MAIN scatter plot (without legend) ---
    fig, ax = plt.subplots(figsize=fig_size)
    ax.scatter(x=x_cord, y=y_cord, c=[palette[i] for i in cluster_codes], s=dot_size, marker="o", edgecolors='none')
    ax.set_aspect('equal', adjustable='box')

    ## add plot title & axes titles if title is not None
    if titles:
        ax.set_xlabel('X Coordinate', fontsize=16)
        ax.set_ylabel('Y Coordinate', fontsize=16)
        ax.set_title(plot_title, fontsize=24)

    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=14)

    if invert_y:
        print("inverting y-axis...")
        ax.invert_yaxis()
    else:
        print("y is NOT INVERTED!!")

    file_path = os.path.join(out_dir, f"{save_name_prefix}_KMeans_{n_cluster}_cluster_plot.png")
    plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Main plot saved to {file_path}")

    # --- Plot LEGEND as separate figure ---
    legend_fig, legend_ax = plt.subplots(figsize=(2, len(unique_clusters) * 0.3))
    legend_ax.axis('off')

    handles = [mpatches.Patch(color=palette[i], label=str(unique_clusters[i])) for i in range(len(unique_clusters))]
    legend_ax.legend(handles=handles, title="K-Means Clusters", loc='center left', frameon=False)

    legend_path = os.path.join(out_dir, f"{save_name_prefix}_KMeans_{n_cluster}_legend.png")
    legend_fig.savefig(legend_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Legend saved to {legend_path}")

    del data_tmp
