
### Functions used for pre-processing and creating inputs for MANNER

############################################################################################################################################
############################################################################################################################################
import sys

from PIL import Image

import os

import pandas as pd
import numpy as np
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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
import matplotlib.patches as mpatches  # Import mpatches for creating legend entries

##for plotting clusters (Sicong codes)
import matplotlib.lines as mlines

## simulate gene expression from NegBin (not Normal, since gene exp counts cant be negative)
from scipy.stats import nbinom

##read in hipt output
import pickle

##for GCN
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__) 
print("Torch CUDA version:", torch.version.cuda)  
print("Is CUDA available:", torch.cuda.is_available())  # Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))  # Name of the first GPU
    print("CUDA Device Count:", torch.cuda.device_count())  # Number of GPUs

from torch import Tensor

import hnswlib

import importlib

import gc

import json
from scipy.io import mmread
import locale
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.feather as feather
##h5py to remove attributes from .h5ad to match DCA input format
import h5py

from scipy.ndimage import convolve
from sklearn.decomposition import PCA
from tqdm import tqdm  # Add this at the top if not already imported

import xml.etree.ElementTree as ET

################################################################################################
## Image processing functions
################################################################################################

## Functions to load image and mask file
def load_image(filename, verbose=True):
    img = Image.open(filename)
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # remove alpha channel
    if verbose:
        print(f'Image loaded from {filename}')
    return img

def load_mask(filename, verbose=True):
    mask = load_image(filename, verbose=verbose)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask.any(2)
    return mask

################################################################################################
## Function to create input data for GCN
################################################################################################
def create_GCN_random_batch(adata, hipt_data, batch_outdir, output_header, n_XY_knn=None, 
                     n_HIPT_knn=25, n_hipt_PC = 50, remove_low_genes=False,
                     batch_size = 0, verbose=True, apply_log1p=False):

    ## Create batch_outdir path if it doesnt exist
    os.makedirs(batch_outdir, exist_ok=True)

    ## Convert adata.X to float32
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.astype(np.float32)
    else:
        adata.X = adata.X.astype(np.float32)

    ## Preprocess adata : Remove bottom 25% in terms of mean expression
    ##MANNER's drop-out rate estimation step requires raw data, not log1p transformed data
    ##which violates NegBin assumption
    # Apply log(x + 1) transformation using sc.pp.log1p
    if apply_log1p:
        print("Applying log(x + 1) transformation to adata.X...")
        sc.pp.log1p(adata)
        print(adata)
    else:
        print("SKIPPING log(x + 1) transformation!!!")

    if remove_low_genes:
        print("Removing 25th percentile of genes based on on mean expression!")
        # Calculate the mean gene expression across all cells
        if issparse(adata.X):
            print("adata.X is sparse!")
            gene_means = np.array(adata.X.mean(axis=0)).flatten()  # Sparse-safe mean calculation
        else:
            print("adata.X is NOT sparse!")
            gene_means = np.mean(adata.X, axis=0)

        # Calculate the 25th percentile for gene expression
        q1_threshold = np.percentile(gene_means, 25)

        # Identify genes to keep (those above the 25th percentile)
        genes_to_keep = np.where(gene_means >= q1_threshold)[0]

        # Filter the AnnData object to retain only the selected genes
        adata_filtered = adata[:, genes_to_keep]
        adata = adata_filtered
        print("adata with lowest 25th percentile genes removed: ", adata)
    else:
        print("Using all genes in the dataset!!")

    ## Extract 50 PCs of HIPT data (for KNN graph construction)
    ## From HIPT embedding (576,y-dim, x-dim), reduce dimension using PCA to (50,y-dim,x-dim) for KNN identification
    # Step 1: Reshape hipt_data to (576, 1231*3023)
    hipt_data_reshaped = hipt_data.reshape(576, -1)  # Shape: (576, y-dim * x-dim)
    hipt_data_reshaped.shape 
    #shape should be (576, y-dim * x-dim)

    # Step 2: Apply PCA to reduce to 50 dimensions (default)
    pca = PCA(n_components=n_hipt_PC)
    hipt_data_PC_flat = pca.fit_transform(hipt_data_reshaped.T).T  # Transpose before and after PCA

    # Check the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # Compute the cumulative variance explained by the first 50 PCs
    cumulative_variance = np.sum(explained_variance_ratio)
    print(f"Variance explained by the first 50 PCs: {cumulative_variance * 100:.2f}%")

    # Step 3: Reshape the result back to (50, y-dim, x-dim)
    hipt_data_PC = hipt_data_PC_flat.reshape(50, hipt_data.shape[1], hipt_data.shape[2])

    print("Reduced hipt_data shape:", hipt_data_PC.shape)

    def construct_knn_graph_hnsw_adata(adata, embedding_key="hipt_embedding", k=25, space="cosine"):
            """
            Constructs a KNN graph using hnswlib from the specified embedding in adata.obsm.

            Parameters:
            - adata: AnnData object containing the embedding in obsm.
            - embedding_key: Key in adata.obsm where the embedding is stored.
            - k: Number of nearest neighbors to find.
            - space: Distance metric for HNSW ('l2', 'cosine', etc.).

            Returns:
            - edge_index: PyTorch tensor of shape (2, num_edges), representing the adjacency list of the KNN graph.
            """
            # Extract embedding from adata
            data = adata.obsm[embedding_key]

            # Ensure data is in float32
            data = data.astype(np.float32)
            num_samples, dim = data.shape

            # Initialize HNSW index
            p = hnswlib.Index(space=space, dim=dim)
            p.init_index(max_elements=num_samples, ef_construction=200, M=16)
            p.add_items(data)

            # Set ef parameter for accuracy
            p.set_ef(50)

            # Query for KNN
            indices, distances = p.knn_query(data, k=k)

            # Construct edge_index (PyTorch tensor)
            row_indices = np.repeat(np.arange(num_samples), k)
            col_indices = indices.flatten()
            edge_index = torch.tensor(np.vstack((row_indices, col_indices)), dtype=torch.long)

            return edge_index

    # adata = adata_raw.copy()
    # Calculate ranges of raw adata X (array_row) and Y (array_col)
    raw_x_range = (adata.obs['array_row'].min(), adata.obs['array_row'].max())
    raw_y_range = (adata.obs['array_col'].min(), adata.obs['array_col'].max())

    # Print the ranges
    print(f"raw data X range: {raw_x_range}")
    print(f"raw data Y range: {raw_y_range}") 

    ## Determine batch intervals (if `batch_size > 0`)
    if batch_size > 0:
        ##Creating Batches:
        ##Count total number of cells
        n_spots = adata.n_obs

        ## Randomly shuffle all indices
        np.random.seed(42)
        all_indices = np.random.permutation(n_spots)

        #split_size
        split_size = n_spots // batch_size

        # Precompute list of arrays of indices per batch
        batch_indices = []

        for i in range(batch_size):
            start = i * split_size
            end = (i + 1) * split_size if i < batch_size - 1 else n_spots  # last gets remainder
            batch_indices.append(all_indices[start:end])

        # Print batch sizes
        for i, idx in enumerate(batch_indices):
            print(f"Batch {i+1}: {len(idx)} spots")
    else:
        batch_indices = [0]  # Treat entire dataset as a single batch


    for i, idx in enumerate(batch_indices):
        subset_tmp = adata if batch_size == 0 else adata[idx].copy()
        
        # Store array_row and array_col in the .uns attribute
        subset_tmp.uns["array_coords"] = subset_tmp.obs[["array_row", "array_col"]].copy()

        # Drop from .obs
        subset_tmp.obs = subset_tmp.obs.drop(columns=["array_row", "array_col"])

        # Round 'hipt_x' and 'hipt_y' to integers (to match with HIPT coords)
        subset_tmp.obs['hipt_x'] = subset_tmp.obs['hipt_x'].round().astype(int)
        subset_tmp.obs['hipt_y'] = subset_tmp.obs['hipt_y'].round().astype(int)

        # use hipt_x and y to create a KNN adj matrix 
        # Backup the original spatial coordinates before overwriting
        # Assign hipt_x and hipt_y to a new obsm key
        subset_tmp.obsm['hipt_spatial'] = subset_tmp.obs[['hipt_x', 'hipt_y']].values

        # Use the new obsm key for KNN computation
        if n_XY_knn is not None:
            sc.pp.neighbors(subset_tmp, use_rep='hipt_spatial', n_neighbors=n_XY_knn, metric='euclidean', knn=True)  
            print("HIPT X and Y coordinate based KNN graph computed!")
        else:
            print("n_XY_knn is None, skipping KNN computation.")
         
    
        ##############################################################################################
        ## Create a HIPT_embedding subset based on the hipt_X and hipt_Y in adata
        ## For each spot in adata (where it be full data or 120k subset), find 50 hipt PCs by matching (hipt_x,hipt_y) to hipt embedding matrix, 
        ## which has shape(50PCs, y-coord, x-coord)
        # Extract "spot_ID", "hipt_x", and "hipt_y"
        hipt_match = pd.DataFrame({
            "spot_ID": subset_tmp.obs.index,  # Spot IDs from subset_tmp
            "hipt_x": subset_tmp.obs["hipt_x"].astype(int),  # Ensure x-coordinates are integer
            "hipt_y": subset_tmp.obs["hipt_y"].astype(int)   # Ensure y-coordinates are integer
        })

        # Ensure valid (hipt_x, hipt_y) coordinates
        max_x, max_y = hipt_data_PC.shape[2], hipt_data_PC.shape[1]
        valid_mask = (hipt_match["hipt_x"] < max_x) & (hipt_match["hipt_y"] < max_y)
        hipt_match = hipt_match[valid_mask]
        # print(hipt_match)

        # Get unique (hipt_x, hipt_y) pairs
        unique_coords = hipt_match[["hipt_x", "hipt_y"]].drop_duplicates()

        # Fetch PC values for unique coordinates
        def extract_pc(row):
            return hipt_data_PC[:, row["hipt_y"], row["hipt_x"]]

        unique_coords["hipt_PC"] = unique_coords.apply(extract_pc, axis=1)

        # Expand PC values into separate columns
        hipt_PC_cols = [f"PC{j+1}" for j in range(hipt_data_PC.shape[0])]
        pc_values_df = pd.DataFrame(unique_coords["hipt_PC"].to_list(), index=unique_coords.index, columns=hipt_PC_cols)

        # Merge back to original dataframe
        hipt_match = hipt_match.merge(pd.concat([unique_coords, pc_values_df], axis=1).drop(columns="hipt_PC"), on=["hipt_x", "hipt_y"], how="left")

        ## Add hipt_match to subset_tmp as a new layer "hipt_embedding"
        # Extract spot_IDs and 50 PCs from hipt_match
        hipt_pcs = hipt_match.set_index("spot_ID").loc[subset_tmp.obs.index, [f"PC{j}" for j in range(1, 51)]]

        # Convert to a NumPy array
        hipt_embedding_array = hipt_pcs.to_numpy()

        # Store it in subset_tmp.obsm (for embeddings)
        subset_tmp.obsm["hipt_embedding"] = hipt_embedding_array

        ## final check for subset_tmp.X to be float32
        # Check if sparse
        if scipy.sparse.issparse(subset_tmp.X):
            if subset_tmp.X.dtype != np.float32:
                subset_tmp.X = subset_tmp.X.astype(np.float32)
                print("Converted sparse matrix to float32.")
            else:
                print("Sparse matrix is already float32.")
        else:
            if subset_tmp.X.dtype != np.float32:
                subset_tmp.X = subset_tmp.X.astype(np.float32)
                print("Converted dense matrix to float32.")
            else:
                print("Dense matrix is already float32.")

        # Verify
        print("Added hipt_embedding with shape:", subset_tmp.obsm["hipt_embedding"].shape)

        ##Save subset_tmp with hipt_embedding 50PCs + (x,y)-based KNNs
        if n_XY_knn is not None:
            output_with_knn_filename = f"{output_header}{i}_with_hipt_50PCs_spatial_knn.h5ad"
        else:
            output_with_knn_filename = f"{output_header}{i}_with_hipt_50PCs_NO_spatial_knn.h5ad"

        subset_tmp.write_h5ad(batch_outdir + output_with_knn_filename, compression='gzip')
        print('output saved in directory: ', batch_outdir+output_with_knn_filename)
    

        ## construct KNN based on hipt-embedding for current subset_tmp
        # Construct edge_index from HIPT PCs
        edge_index_pcs = construct_knn_graph_hnsw_adata(adata=subset_tmp, embedding_key="hipt_embedding", 
                                                        k=n_HIPT_knn, space="cosine").numpy()  # Convert to NumPy
        print("HIPT-based Edge Index shape:", edge_index_pcs.shape)

        ## If we have (X,Y)-based KNN, concatenate with HIPT-based KNN
        if n_XY_knn is not None:
    
            # Fetch the adjacency matrix from ScanPy (spatial KNN graph)
            adj_matrix_xy = subset_tmp.obsp["connectivities"]

            # Convert to COO format
            coo_xy = coo_matrix(adj_matrix_xy)

            # Extract edge indices (spatial KNN)
            edge_index_xy = np.vstack((coo_xy.row, coo_xy.col))  # Shape: (2, num_edges)
            print("(x,y)-based Edge Index shape:", edge_index_xy.shape)
        
            # Concatenate both edge indices
            combined_edge_index = np.hstack((edge_index_xy, edge_index_pcs))  # Shape: (2, total_edges)
            print("Combined Edge Index shape:", combined_edge_index.shape)

            # Convert to PyTorch tensor
            combined_edge_index_tensor = torch.tensor(combined_edge_index, dtype=torch.long)

            # Save the PyTorch tensor to a file
            tensor_filename = f"{output_header}{i}_xy_cord_and_hipt_50pc_combined_KNN_{n_XY_knn}_{n_HIPT_knn}_edge_index.pt"
            tensor_filepath = os.path.join(batch_outdir, tensor_filename)
            torch.save(combined_edge_index_tensor, tensor_filepath)
        
        else:

            ## If we dont have (X,Y)-based KNN, we just save HIPT-based KNN
            combined_edge_index = edge_index_pcs  # Shape: (2, total_edges)
            print("HIPT-KNN only Edge Index shape:", combined_edge_index.shape)

            # Convert to PyTorch tensor
            combined_edge_index_tensor = torch.tensor(combined_edge_index, dtype=torch.long)

            # Save the PyTorch tensor to a file
            tensor_filename = f"{output_header}{i}_hipt_50pc_KNN_{n_HIPT_knn}_edge_index.pt"
            tensor_filepath = os.path.join(batch_outdir, tensor_filename)
            torch.save(combined_edge_index_tensor, tensor_filepath)

        # Save the combined COO matrix to a pickle file
        #coo_mtx_filename = f"{output_header}{i}_xy_cord_and_hipt_50pc_combined_edge_index.pickle"
        #with open(batch_outdir + coo_mtx_filename, "wb") as f:
        #    pickle.dump(combined_edge_index, f)

        #print('COO format concatenated edge indices saved in directory: ', batch_outdir + coo_mtx_filename)
    
        print('pytorch tensor of edge indices saved in directory: ', batch_outdir+tensor_filename)

        # Print results
        print("Final Edge Index shape:", combined_edge_index_tensor.shape)
    
        ##delete all tmp files to free space
        subset_tmp.obsm.clear()
        subset_tmp.uns.clear()
        subset_tmp.obs.drop(subset_tmp.obs.index, inplace=True)
        del subset_tmp
        gc.collect()
       