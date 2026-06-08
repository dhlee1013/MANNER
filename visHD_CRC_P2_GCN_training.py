## python file for Visium HD GCN training for MANNER
## File to test run GCN part for MANNER
## First prepare 2 input datasets
## Gene expression dataset 
## Edge_index representing a KNN adj matrix based on top 50PCs of HIPT image embedding

import sys

sys.path.append("/path/to/MANNER/MANNER_codes/")

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
from sklearn.metrics.pairwise import cosine_similarity 
import matplotlib.patches as mpatches  
import matplotlib.lines as mlines

##read in hipt output
import pickle

##for GCN
import os

import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)
print("Torch CUDA version:", torch.version.cuda)  # Shows the CUDA version PyTorch 
print("Is CUDA available:", torch.cuda.is_available())  # Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))  # Name of the first GPU
    print("CUDA Device Count:", torch.cuda.device_count())  # Number of GPUs

from torch import Tensor
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data, Batch, DataLoader
import torch.optim as optim

## to measure time it took for each step
import time

## import functions used for GCN training
from Train_GCN import create_gcn_input, MANNER_GCN

import subprocess

#####################################################################

# Step 1: Extract X matrix from adata
# If X is sparse, convert to dense matrix 
# Check if adata.X is sparse

############################################################################
##set number of PCs to use for GCN training
n_gene_PC=300

# Initialize an empty list to store Data objects
# we are using the full data without splitting them into batches

gcn_batches = []

for i in range(1):

    gcn_filename = f"visHD_crc_p2_masked_fulldata{i}_with_hipt_50PCs_NO_spatial_knn.h5ad"
    edge_index_filename = f"visHD_crc_p2_masked_fulldata{i}_hipt_50pc_KNN_25_edge_index.pt"

    data = create_gcn_input(
        gcn_preprocess_dir = "/path/to/MANNER/visHD_CRC_P2/MANNER_preprocessed/", 
        gcn_filename = gcn_filename, 
        edge_index_filename = edge_index_filename, 
        n_gene_PC=n_gene_PC,
        HVG=None,
        apply_log1p=True
        )
    
    ##Append the output for current batch to the list
    gcn_batches.append(data)

# Print confirmation
print(f"Successfully processed {len(gcn_batches)} batches.")

###############################################################################################
## Train GCN and extract GCN embedding
###############################################################################################
def setup_device():
    if torch.cuda.is_available():
        try:
            # Query GPU memory via nvidia-smi
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
                encoding="utf-8"
            )
            free_mem_list = [int(x) for x in result.strip().split("\n")]
            best_gpu = max(range(len(free_mem_list)), key=lambda i: free_mem_list[i])
            device = torch.device(f'cuda:{best_gpu}')
            
            print(f"[nvidia-smi] Using CUDA device: {torch.cuda.get_device_name(best_gpu)} "
                  f"(cuda:{best_gpu}) with {free_mem_list[best_gpu]/1024:.2f} GB free")
        
        except Exception as e:
            # Fallback to first available CUDA device if nvidia-smi fails
            device = torch.device("cuda:0")
            print(f"nvidia-smi failed ({e}), using CUDA device: {torch.cuda.get_device_name(0)} (cuda:0)")
    
    else:
        device = torch.device("cpu")
        print("CUDA device not available, using CPU")
    
    return device


# Device setup
device = setup_device()


# Initialize the GCN model
input_dim = gcn_batches[0].x.shape[1]  # Number of features (genes)
print(f"dimension of input gene expression features: {input_dim}")
hidden_dim = 128              # Hidden layer dimension 
output_dim = 64               # Target output dimension
model = MANNER_GCN(input_dim, hidden_dim, output_dim).to(device)

# Free up GPU memory before training
torch.cuda.empty_cache()

# Start training
start_time = time.time()
loss_history = model.fit(
    batches=gcn_batches, #batch-wise training with list of PyTorch Geometric Data objects
    lr=0.005, 
    max_epochs=5000, 
    update_interval=100, 
    weight_decay=5e-4, 
    tol=1e-7,
    early_stopping=True,
    device=device
    )
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

#################################################################################################################################
# Save MSE Loss History Plot
#################################################################################################################################
# Ensure the directory exists
MANNER_plotdir = "/path/to/MANNER/visHD_CRC_P2/MANNER_preprocessed/plot/"
os.makedirs(MANNER_plotdir, exist_ok=True)  # Create the directory if it doesn't exist

# Plot and save the loss history
plt.figure()
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Total MSE Loss')
plt.title(f'GCN Training Loss ({n_gene_PC} log1p genes PCs, HIPT-only KNN with k=25, learningRate=0.005)')
plt.savefig(os.path.join(MANNER_plotdir, f"fulldata_{n_gene_PC}_genes_PCs_hipt_only_knn_25_loss_lr_005_plot.png"))  # Save the plot as a .png file
plt.close()  # Close the plot to free resources
#################################################################################################################################

# Save the model's state_dict
gcn_embed_dir = "/path/to/MANNER/visHD_CRC_P2/MANNER_preprocessed/GCN_embedding/"
os.makedirs(gcn_embed_dir, exist_ok=True)

torch.save(model.state_dict(), gcn_embed_dir+f'{n_gene_PC}_gene_PCs_GCN_model.pth')

#################################################################################################################################
## Re-initialize model using the saved .pth file
gcn_embed_dir = "/path/to/MANNER/visHD_CRC_P2/MANNER_preprocessed/GCN_embedding/"
gcn_model_name = f'{n_gene_PC}_gene_PCs_GCN_model.pth'
model = MANNER_GCN(input_dim=n_gene_PC, hidden_dim=128, output_dim=64).to(device)
model.load_state_dict(torch.load(gcn_embed_dir + gcn_model_name))
model.eval()

#################################################################################################################################
# Generate GCN embeddings

start_time = time.time()
embeddings = model.predict(gcn_batches[0], device=device)
end_time = time.time()
print(f"Embedding extraction completed in {end_time - start_time:.2f} seconds.")

# Check output shape
print(f"GCN output shape: {embeddings.shape}")

###############################################################################################
# Save the final GCN embeddings:
# Ensure the output tensor is moved to the CPU before saving
MANNER_dir = "/path/to/MANNER/visHD_CRC_P2/MANNER_preprocessed/GCN_embedding/"
output_file = f"visHD_crc_p2_{n_gene_PC}_gene_PCs_GCN_output.pt"

# Use detach() to prevent computation graph retention
torch.save(embeddings.detach().cpu(), MANNER_dir + output_file)  
print(f"GCN output saved to {MANNER_dir + output_file}.")

## Check embedding quality by clustering analysis
gcn_embeddings = embeddings.detach().cpu().numpy()

## Load full data with all 18074 genes
gcn_preprocess_dir="/path/to/MANNER/visHD_CRC_P2/MANNER_preprocessed/"
gcn_filename="visHD_crc_p2_masked_fulldata0_with_hipt_50PCs_NO_spatial_knn.h5ad"
adata = sc.read_h5ad(gcn_preprocess_dir+gcn_filename)
print(adata)
print("adata.X.dtype before adding X_gcn: ",adata.X.dtype)

## Add GCN embedding to h5ad file
adata.obsm["X_gcn"] = gcn_embeddings

# Remove 'array_coords' from uns
del adata.uns["array_coords"]

# Confirm changes
print("Updated obs columns:", adata.obs.columns)
print("Removed 'array_coords' from uns:", "array_coords" not in adata.uns)

## remove unncessary variables
# Remove specific keys from .uns, .obsm, and .obsp
keys_to_remove_uns = ["neighbors", "spatial"]
keys_to_remove_obsm = ["hipt_spatial"]
keys_to_remove_obsp = list(adata.obsp.keys())  # Remove all obsp entries
keys_to_remove_var = ["feature_types", "genome"]  # Remove specific var columns

# Remove keys from .uns
for key in keys_to_remove_uns:
    if key in adata.uns:
        del adata.uns[key]

# Remove keys from .obsm
for key in keys_to_remove_obsm:
    if key in adata.obsm:
        del adata.obsm[key]

# Remove all keys from .obsp
adata.obsp.clear()

# Remove columns from .var
adata.var.drop(columns=keys_to_remove_var, inplace=True, errors="ignore")

# Convert hipt_x and hipt_y to more memory-efficient types
adata.obs["hipt_x"] = adata.obs["hipt_x"].astype(np.int32)  # Convert to int32 (4 bytes)
adata.obs["hipt_y"] = adata.obs["hipt_y"].astype(np.int32)  # Convert to int32 (4 bytes)


# Print updated structure
print(adata)
print(adata.obs.dtypes)
print(adata.X.dtype)

MANNER_prep_dir = "/path/to/MANNER/visHD_CRC_P2/MANNER_preprocessed/GCN_embedding/"
##Create directory
os.makedirs(MANNER_prep_dir, exist_ok=True)

##save the gene expression + GCN embedding data as .h5ad, which will be the input for MANNER denoising algorithm
output_filename = f"visHD_crc_p2_{n_gene_PC}_gene_PCs_GCN_embedding_cleaned.h5ad"
adata.write_h5ad(MANNER_prep_dir + output_filename, compression='gzip')