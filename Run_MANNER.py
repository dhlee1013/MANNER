##Run MANNER denoising algorithm
##Inputs: 

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.chdir("/path/to/MANNER")

import sys
sys.path.append('/path/to/MANNER/MANNER_codes/')

import argparse
import time
import scanpy as sc

from Get_Disp import *
from Decoder_NBC import *
# from PE_NODE_Embedding import *
from Bootstrap_NB import *
from sklearn.decomposition import PCA
from pytorch_lightning.utilities import rank_zero_only
import subprocess
import anndata as ad


# Set matmul precision globally (must be done before any heavy torch ops)
# torch.matmul is a trade-off for precision
# default is full float32, the slowest, with max numerical accuracy
# high is mixed float32 and float16, faster, but less precise that default
# medium is mixed float32 and float16, the fastest, with least precision
torch.set_float32_matmul_precision('high')  # or 'medium'


##when running in 2+ gpu, need this so that only the first GPU (rank 0 gpu) writes the output files

@rank_zero_only
def save_adata(adata2, output_path, output_filename):
    adata2.write_h5ad(os.path.join(output_path, output_filename), compression='gzip')
    print("MANNER output has been saved!")


## select GPU with most free memory
def setup_device():
    if torch.cuda.is_available():
        try:
            # Query GPU memory via nvidia-smi
            result = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],encoding="utf-8")

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

@rank_zero_only
def count_gpus():
    return torch.cuda.device_count()

@rank_zero_only
def print_gpu_info():
    print(f"Number of GPUs available: {count_gpus()}")


def main(filename, output_path, output_filename):

    # Setup the device once (all processes call this, but only rank 0 prints)
    device = setup_device()

    
    def load_and_preprocess_data(filename):
        adata = sc.read_h5ad(filename)
        print(adata)

        # Check and convert adata.X to float32 if not already
        if adata.X.dtype != np.float32:
            print(f"Converting adata.X from {adata.X.dtype} to float32...")
            adata.X = adata.X.astype(np.float32)
        else:
            print("adata.X is already float32.")

        # Check if adata.X is a sparse matrix, and convert to dense matrix
        if issparse(adata.X):
            print("adata.X is a sparse matrix. Converting to dense matrix...")
            adata.X = adata.X.toarray()

            # Count NaN values in adata.X
            nan_count = np.isnan(adata.X).sum()
            # Print the count
            print(f"Total number of NaN values in dense adata.X: {nan_count}")
        else:
            print("adata.X is already a dense matrix.")
            # Count NaN values in adata.X
            nan_count = np.isnan(adata.X).sum()
            # Print the count
            print(f"Total number of NaN values in dense adata.X: {nan_count}")

        if issparse(adata.X):
            has_neg = (adata.X < 0).sum() > 0
        else:
            has_neg = np.any(adata.X < 0)

        print(f"Contains negative values in adata.X: {has_neg}")

        is_integer = np.all(np.equal(np.mod(adata.X, 1), 0))
        print(f"All values in dense adata.X are integers: {is_integer}")

        # Run without filtering (all the genes)
        adata_filtered = adata

        # Count NaN values in adata.X
        nan_count2 = np.isnan(adata_filtered.X).sum()
        # Print the count
        print(f"Total number of NaN values in dense adata_filtered.X: {nan_count2}")

        return adata_filtered
    
    # All ranks (gpus) load and preprocess data
    adata_filtered = load_and_preprocess_data(filename)

    # Run Decoder_NB on all GPUs
    denoising_start_time = time.time()

    adata2 = Decoder_NB_GCN(adata_filtered, device)

    denoising_end_time = time.time()
    print(f"Denoising step completed in {denoising_end_time - denoising_start_time:.2f} seconds.")
    print("Denoising completed! Starting post-processing...")

    # Only rank 0 logs and processes the final result
    @rank_zero_only
    def postprocess_and_save_adata(adata2, output_path, output_filename):
        # Print the denoised summary
        print("Denoised output adata2: ")
        print(adata2)
        X_dtype = adata2.X.dtype
        print(f"X matrix dtype: {X_dtype}")

        # Keep only the specified columns in adata2.var
        # List of desired columns
        columns_to_keep = ['gene_ids', 'genome', 'n_cells']

        # Check which columns exist in adata2.var
        existing_columns = [col for col in columns_to_keep if col in adata2.var.columns]

        # Keep only the existing columns
        adata2.var = adata2.var[existing_columns]

        # Print the selected columns
        print("Selected columns in adata2.var:", list(adata2.var.columns))

        # Keep only 'spatial' in adata2.uns if it exists, otherwise clear it
        adata2.uns = {'spatial': adata2.uns['spatial']} if 'spatial' in adata2.uns else {}

        # Print confirmation
        print("Updated adata2.uns (keeping only 'spatial' if it exists):")
        print(adata2.uns)

        # Create output directory and save the result
        os.makedirs(output_path, exist_ok=True)
        save_adata(adata2, output_path, output_filename)

    # Only rank 0 postprocesses and saves
    postprocess_and_save_adata(adata2, output_path, output_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MANNER on a specified h5ad file.')
    parser.add_argument('filename', type=str, help='Path to the input h5ad file')
    parser.add_argument('output_path', type=str, help='Path for the output h5ad file')
    parser.add_argument('output_filename', type=str, help='Filename for the output h5ad file')
    args = parser.parse_args()

    # Ensure that GPU info is printed only once (rank 0)
    print_gpu_info()

    full_start_time = time.time()
    main(args.filename, args.output_path, args.output_filename)
    full_end_time = time.time()
    print(f"Entire Denoising Process completed in {full_end_time - full_start_time:.2f} seconds.")
    

