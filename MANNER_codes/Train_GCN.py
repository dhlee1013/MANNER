## Functions for training GCN
## First prepare 2 input datasets
## Gene expression dataset 
## Edge_index representing a KNN adj matrix based on top 50PCs of HIPT image embedding

import sys
import pandas as pd
import numpy as np
import scanpy as sc
#!pip3 install igraph
#!pip3 install leidenalg
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
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data, Batch, DataLoader
import torch.optim as optim
import scanpy as sc

import time

from tqdm import tqdm

########################################################################
# Step 1: Extract X matrix from adata
# If X is sparse, convert to dense matrix 
# Check if adata.X is sparse

def create_gcn_input(gcn_preprocess_dir, gcn_filename, edge_index_filename, n_gene_PC=300, HVG=None, apply_log1p=False):
    
    """
    Create a PyTorch Geometric `Data` object for GCN input from a preprocessed AnnData file.

    Parameters:
    - gcn_preprocess_dir (str): Path to directory containing the input files.
    - gcn_filename (str): Filename of the AnnData object (.h5ad).
    - edge_index_filename (str): Filename of the edge index (.pt).
    - n_gene_PC (int): Number of PCs or GLM-PCs to extract as features.
    - HVG (int or None): If provided, select this many highly variable genes before PCA.
    - apply_log1p (bool): Whether to apply log1p transformation before PCA (ignored if glm_pca=True).

    Returns:
    - data (torch_geometric.data.Data): PyG Data object with node features and edges.
    """

 # Load AnnData
    adata = read_h5ad(gcn_preprocess_dir + gcn_filename)
    print(f"Loaded AnnData: {adata}")

    if apply_log1p:
        print("🔹 Applying log1p transformation...")
        sc.pp.log1p(adata)
    else:
        print("⚠️ Skipping log1p transformation.")

    if HVG is not None:
        print(f"🔹 Selecting top {HVG} highly variable genes before PCA...")
        sc.pp.highly_variable_genes(adata, n_top_genes=HVG, subset=True)

    if n_gene_PC is not None:
        print(f"🔹 Running PCA to extract {n_gene_PC} PCs...")
        sc.tl.pca(adata, n_comps=n_gene_PC, zero_center=False)
        X_np = adata.obsm["X_pca"][:, :n_gene_PC]
        explained_var = np.sum(adata.uns["pca"]["variance_ratio"]) * 100
        print(f"🔸 Variance explained by {n_gene_PC} PCs: {explained_var:.2f}%")
    else:
        print("🔸 No PCA: using gene expression matrix as-is.")
        X_np = adata.X

    if issparse(X_np):
        print("🔹 Converting sparse matrix to dense...")
        X_np = X_np.toarray()

    # Convert features to PyTorch tensor
    X = torch.tensor(X_np, dtype=torch.float32)
    print("✅ GCN input X shape:", X.shape)


    # Step 2: load HIPT KNN edge index PyTorch Tensor file (from .pt file)
    ##load pytorch tensor file as "edge_index"
    edge_index = torch.load(gcn_preprocess_dir + edge_index_filename)

    # Print shape to verify
    print("Edge Index Shape:", edge_index.shape)

    # Step 3: Create a PyTorch Geometric Data object
    #fulldata = Data(x=x, edge_index=edge_index)
    data = Data(x=X, edge_index=edge_index)

    # Check the Data object
    print("Data object:")
    print(data)

    print('GCN input data properties')
    print('==============================================================')
    'edge_attr' in data #False
    print(data.num_nodes) # Number of spots in data
    print(data.num_edges) 
    print(data.num_node_features) #Number of genes in data
    print(data.has_isolated_nodes()) #TRUE: 1 isolated node
    print(data.has_self_loops()) #False: no self-loops
    print(data.is_directed()) #False: no directed edges

    return data

############################################################################

class MANNER_GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        # 2-Layer MLP Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(output_dim, hidden_dim),  # First layer (output_dim -> hidden_dim)
            torch.nn.ReLU(),                          # Activation function
            torch.nn.Linear(hidden_dim, input_dim)    # Second layer (hidden_dim -> input_dim)
        )

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        edge_index = edge_index.to(torch.long)  # Ensure edge_index is long type
        
        # Forward pass through GCN layers
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        
        # Decode the embedding back to input space
        reconstructed = self.decoder(x)
        return x, reconstructed

    def fit(self, batches, lr=0.001, max_epochs=5000, update_interval=10,
            weight_decay=5e-4, tol=1e-6, early_stopping=True, device='cuda'):
        """
        Train GCN, optimizing the total MSE loss.

        Parameters:
        - batches: List of PyTorch Geometric 'Data' object containing:
            - 'data.x' for gene PCs
            - 'data.edge_index' for edge_index
        - lr: Learning rate for Adam optimizer
        - max_epochs: Maximum number of training epochs
        - update_interval: Print loss every X epochs
        - weight_decay: L2 regularization for Adam
        - tol: Early stopping threshold
        - early_stopping: Enable early stopping
        - device: 'cuda' or 'cpu'

        Returns:
        - loss_history: List of total loss values over epochs
        """

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_history = []
        previous_loss = None

        # Initialize a counter for consecutive epochs with small loss change
        consecutive_tol_count = 0  
        patience = 10  # Number of consecutive epochs required for early stopping


        # Outer loop with epoch-level progress bar
        # for epoch in range(max_epochs):
        for epoch in tqdm(range(max_epochs), desc="Training Epochs"):
            total_loss = 0
            # Inner loop with batch-level progress bar
            batch_iterator = tqdm(batches, desc=f"Epoch {epoch}", leave=False)

            for batch_data in batch_iterator:
                batch_data = batch_data if hasattr(batch_data, 'to') else batch_data
                # Move entire Data object to device (more efficient than separate tensor transfers)
                batch_data = batch_data.to(device)

                # Forward pass through GCN
                embeddings, reconstructed = self(batch_data)

                # Compute loss
                loss = F.mse_loss(reconstructed, batch_data.x)  # batch_data.x contains node features
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate total loss
                total_loss += loss.item()

            # update tqdm bar with current loss
            batch_iterator.set_postfix(loss=loss.item())

            loss_history.append(total_loss)

            if early_stopping and previous_loss is not None:
                loss_diff = abs(previous_loss - total_loss)
                if loss_diff < tol:
                    consecutive_tol_count +=1 #increase count if condition is met

                if consecutive_tol_count >= patience:
                    print(f"Early stopping at epoch {epoch}. Loss change < tol for {patience} consecutive epochs.")
                    break
                
            else:
                consecutive_tol_count = 0  # Reset count if condition is not met

            previous_loss = total_loss

            if epoch % update_interval == 0:
                print(f"Epoch {epoch}: Total Loss = {total_loss:.4f}")

        print("Training completed.")
        return loss_history
    
    def predict(self, data=None, X=None, edge_index=None, edge_weight=None, device='cuda'):
        """
        Generate embeddings using either a PyG Data object or raw tensors.

        Parameters
        ----------
        data : torch_geometric.data.Data, optional
            If provided, this Data object should contain `x`, `edge_index`, and optionally `edge_weight`.
        X : np.ndarray or torch.Tensor, optional
            Node features (used if `data` is None)
        edge_index : np.ndarray or torch.Tensor, optional
            Edge indices (used if `data` is None)
        edge_weight : np.ndarray or torch.Tensor, optional
            Edge weights (used if `data` is None)
        device : str or torch.device
            'cuda' or 'cpu'

        Returns
        -------
        embeddings : torch.Tensor
            Node embeddings on CPU
        """

        self.eval()
        
        # Determine input
        if data is not None:
            # Move entire Data object to device
            data = data.to(device)
        else:
            if X is None or edge_index is None:
                raise ValueError("Either `data` or both `X` and `edge_index` must be provided.")
            # Convert numpy to torch if needed and move to device
            if not torch.is_tensor(X):
                X = torch.FloatTensor(X).to(device)
            else:
                X = X.to(device)
            if not torch.is_tensor(edge_index):
                edge_index = torch.LongTensor(edge_index).to(device)
            else:
                edge_index = edge_index.to(device)
            if edge_weight is not None:
                if not torch.is_tensor(edge_weight):
                    edge_weight = torch.FloatTensor(edge_weight).to(device)
                else:
                    edge_weight = edge_weight.to(device)
            data = Data(x=X, edge_index=edge_index, edge_weight=edge_weight)

        with torch.no_grad():
            embeddings, _ = self(data)

        return embeddings.cpu()
