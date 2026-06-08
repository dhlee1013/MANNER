'''
## This .py file is for creating input data for the GCN training using Visium HD Colorectal Cancer Patient 2 dataset
## Process visiumHD image file for HIPT image embedding extraction for GCN
'''


############################################################################################################################################
############################################################################################################################################
import sys
## location for pre-trained models "vit4k_xs_dino.pth" and "vit256_small_dino.pth" for istar
sys.path.append("/path/to/istar/checkpoints/")

## location for MANNER codes
sys.path.append("/path/to/MANNER/MANNER_codes/")

# --- Point to the HistoSweep source folder so imports resolve ---
sys.path.append('/path/to/HistoSweep-main') 

from MANNER_prep_utils import load_image, load_mask, create_GCN_random_batch

## PIL is under Pillow module
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import tifffile

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

##for creating binary adjancy matrix from image feature matrix
import hnswlib

import importlib
import gc
import json
from scipy.io import mmread
import locale
import h5py

from scipy.ndimage import convolve
from sklearn.decomposition import PCA
from tqdm import tqdm  # Add this at the top if not already imported

import xml.etree.ElementTree as ET
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.feather as feather

import subprocess

############################################################################################################################################
data_dir = '/path/to/MANNER/visHD_CRC_P2/'
data_dir0 = data_dir
out_dir = data_dir + 'img_processed/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

foldername = 'square_008um'
bin_size = 8
data_dir = data_dir + "binned_outputs/" + foldername + "/"
out_dir = out_dir + foldername + "/"

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

## split original btf image into two halves because the whole image is too large for .jpg
full_he = tifffile.imread(data_dir0 + "extras/he-raw.btf", level=0)
print("full image shape is", full_he.shape)
Image.fromarray(full_he[:,:40000,:], 'RGB').save(data_dir0 + 'extras/he-raw-left.jpg')
Image.fromarray(full_he[:,40000:,:], 'RGB').save(data_dir0 + 'extras/he-raw-right.jpg')
del full_he

##Load the raw HE image (right side, so x-coordinates of corresponding adata should all be shifted by -40000)
he = load_image(data_dir0 + 'extras/he-raw-right.jpg')
print("he-raw-right jpg's shape is", he.shape)

#######################################################
## preprocess the binned data to be in Visium format
tissue_positions = pd.read_parquet(data_dir+"/spatial/tissue_positions.parquet")
tissue_positions.set_index(tissue_positions.columns[0], inplace=True)
tissue_positions = tissue_positions.astype(np.float32)
tissue_positions.to_csv(data_dir+"/spatial/tissue_positions_list.csv")

adata = sc.read_visium(path=data_dir)
adata.obsm['spatial'] = adata.obsm['spatial'].astype(np.float32)
adata.var_names_make_unique()
adata.obs_names_make_unique()

## filter out out-of-image sequences
full_size = [he.shape[0], he.shape[1]+40000]
locs = adata.obsm['spatial']
idx = np.where((locs[:,0] >= 40000) & (locs[:, 0] <= he.shape[1]+40000) & (locs[:, 1] >= 0) & (locs[:, 1] <= he.shape[0]))[0]
adata2 = adata[idx]
adata2.obsm['spatial'][:,0] = adata2.obsm['spatial'][:,0] - 40000
spot_diameter_fullres = adata2.uns['spatial'][next(iter(adata2.uns['spatial']))]['scalefactors']['spot_diameter_fullres']
radius = spot_diameter_fullres*0.5 ## number of pixels per spot radius
locs = pd.DataFrame(adata2.obsm['spatial'].astype(int), columns=['x', 'y'], index=adata2.obs_names).astype(np.float32)

## Step 2: automatically cut the full H&E into the part with expressions
xl = int(np.ceil(max([0, locs.x.min()-2*radius])))
xr = int(np.floor(min([locs.x.max()+2*radius, he.shape[1]])))
yl = int(np.ceil(max([0, locs.y.min()-2*radius])))
yr = int(np.floor(min([locs.y.max()+2*radius, he.shape[0]])))
adata2 = adata2[(adata2.obsm['spatial'][:,0] < xr-2*radius) &
                (adata2.obsm['spatial'][:,1] < yr-2*radius) &
                (adata2.obsm['spatial'][:,0] > xl+2*radius) &
                (adata2.obsm['spatial'][:,1] > yl+2*radius)]
adata2.obsm['spatial'][:, 0] = adata2.obsm['spatial'][:, 0] - xl
adata2.obsm['spatial'][:, 1] = adata2.obsm['spatial'][:, 1] - yl
he_cut = he[yl:yr, xl:xr, :]

## save the filtered H&E image based on gene expression
## This he-raw.jpg is used for HistoSweep to create tissue mask
Image.fromarray(he_cut, 'RGB').save(out_dir+'he-raw.jpg')

#######################################################
## Use HistoSweep to create a mask 
#######################################################
## import HistoSweep files and functions
import utils as hs_utils
from saveParameters   import saveParams
from computeMetrics   import compute_metrics
from densityFiltering import compute_low_density_mask
from textureAnalysis  import run_texture_analysis
from ratioFiltering   import run_ratio_filtering
from generateMask     import generate_final_mask

# ===== USER-DEFINED INPUT PARAMETERS =====
## Use default parameters of Histosweep except for pixel_size_raw
HE_prefix           = '/path/to/MANNER/visHD_CRC_P2/img_processed/square_008um/'
output_directory    = "HistoSweep_Output"
need_scaling_flag   = True
need_preprocessing_flag = True
pixel_size_raw      = 0.2738262923862567
density_thresh      = 100
clean_background_flag = True
min_size            = 10
patch_size          = 16
pixel_size          = 0.5


if need_scaling_flag:
    subprocess.run([
        sys.executable, 'rescale.py',
        '--image',
        '--pixelSizeRaw', str(pixel_size_raw),
        '--pixelSize',    str(pixel_size),
        '--prefix',       HE_prefix
    ], check=True)

if need_preprocessing_flag:
    subprocess.run([
        sys.executable, 'preprocess.py',
        '--image',
        '--patchSize', str(patch_size),
        '--prefix',    HE_prefix
    ], check=True)

# ===== LOAD IMAGE =====
image = hs_utils.load_image(hs_utils.get_image_filename(HE_prefix + 'he'))
print(image.shape)

# ===== OUTPUT DIR =====
os.makedirs(f"{HE_prefix}{output_directory}", exist_ok=True)

# ===== PIPELINE =====
saveParams(HE_prefix, output_directory, need_scaling_flag, need_preprocessing_flag,
           pixel_size_raw, density_thresh, clean_background_flag, min_size, patch_size, pixel_size)

he_std_norm_image_, he_std_image_, z_v_norm_image_, z_v_image_, ratio_norm_, ratio_norm_image_ = \
    compute_metrics(image, patch_size=patch_size)

mask1_lowdensity = compute_low_density_mask(z_v_image_, he_std_image_, ratio_norm_, density_thresh=density_thresh)
print('Total selected for density filtering: ', mask1_lowdensity.sum())

mask1_lowdensity_update = run_texture_analysis(
    prefix=HE_prefix, image=image, tissue_mask=mask1_lowdensity,
    output_dir=output_directory, patch_size=patch_size, glcm_levels=64
)

mask2_lowratio, otsu_thresh = run_ratio_filtering(ratio_norm_, mask1_lowdensity_update)
print(mask2_lowratio.shape)

generate_final_mask(
    prefix=HE_prefix, he=image,
    mask1_updated=mask1_lowdensity_update, mask2=mask2_lowratio,
    output_dir=output_directory,
    clean_background=clean_background_flag,
    super_pixel_size=patch_size, minSize=min_size
)

##HistoSweep creates masks at 2 different scales:
# mask_small is super-pixel scale at HIPT feature extracted coordinates, while mask is he-raw coordinates


#######################################################
#######################################################

locs = pd.DataFrame(adata2.obsm['spatial'].astype(int), columns=['x', 'y'], index=adata2.obs_names).astype(np.float32)
adata2 = adata2[(locs.x > 2*radius) & (locs.x < he_cut.shape[1]-2*radius) & (locs.y > 2*radius) & (locs.y < he_cut.shape[0]-2*radius)]
locs = pd.DataFrame(adata2.obsm['spatial'].astype(int), columns=['x', 'y'], index=adata2.obs_names).astype(np.float32)

#######################################################
x = adata2.X.tocsr()
cnts_sparse = pd.DataFrame.sparse.from_spmatrix(x, columns=adata2.var_names, index=adata2.obs_names)
cnts_dense = cnts_sparse.apply(lambda col: col.sparse.to_dense(), axis=0)
table = pa.Table.from_pandas(cnts_dense)
pq.write_table(table, out_dir + 'cnts.parquet', compression='brotli')

locs.to_parquet(out_dir+'locs-raw.parquet', compression="brotli")
# locs.to_csv(out_dir + 'locs-raw.tsv', sep='\t')
with open(out_dir+"radius-raw.txt", 'w') as file:
    file.write(str(radius))
with open(out_dir+"pixel-size-raw.txt", 'w') as file:
    file.write(str(bin_size/2/radius))
with open(out_dir+"pixel-size.txt", 'w') as file:
    file.write(str(0.5))

#######################################################
## Run functions included in Zhang et al.'s iStar (https://github.com/daviddaiweizhang/istar)
##Run istar's rescale.py from ipython: input is image file "he-raw.jpg" and pixel-size-raw.txt, pixel-size.txt, output is "he-scaled.jpg"

## rescale the image file
prefix = "/path/to/MANNER/visHD_CRC_P2/img_processed/square_008um/"
command = f"""
source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate /path/to/conda_environment/py39istar && \
python /path/to/istar/rescale_modified.py {prefix} --image
"""

# Execute the command
subprocess.run(command, shell=True, executable="/bin/bash")

## rescale the spatial bin coordinates
prefix = "/path/to/MANNER/visHD_CRC_P2/img_processed/square_008um/"
command = f"""
source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate /path/to/conda_environment/py39istar && \
python /path/to/istar/rescale_modified.py {prefix} --locs
"""

# Execute the command
subprocess.run(command, shell=True, executable="/bin/bash")

##Run istar/pre-process.py: the input is image file named "he-scaled.jpg", output "he.jpg"
prefix = "/path/to/MANNER/visHD_CRC_P2/img_processed/square_008um/"
command = f"""
source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate /path/to/conda_environment/py39istar && \
python /path/to/istar/preprocess.py {prefix} --image
"""

# Execute the command
subprocess.run(command, shell=True, executable="/bin/bash")

prefix = "/path/to/MANNER/visHD_CRC_P2/img_processed/square_008um/"
command = f"""
source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate /path/to/conda_environment/py39istar && \
python /path/to/istar/extract_features.py {prefix}
"""

# Execute the command
subprocess.run(command, shell=True, executable="/bin/bash")


#######################################################
##hipt embedding output is .pickle file; read it into python and check data structure
# Path to your .pickle file
file_path = "/path/to/MANNER/visHD_CRC_P2/img_processed/square_008um/embeddings-hist.pickle"

# Open the file in read-binary mode and load it
with open(file_path, "rb") as file:
    hipt_data = pickle.load(file)

# Print or inspect the loaded data: .pickle is a dictionary object
print(type(hipt_data))
print(hipt_data.keys())

for key, value in hipt_data.items():
    print(f"Key: {key}, Type: {type(value)}")
    if isinstance(value, (list, dict, np.ndarray)):
        print(f"Length/Shape: {len(value) if hasattr(value, '__len__') else value.shape}")
    elif isinstance(value, (int, float, str)):
        print(f"Value: {value}")

## Output
#Key: cls, Type: <class 'list'>
#Length/Shape: 192
#Key: sub, Type: <class 'list'>
#Length/Shape: 384
#Key: rgb, Type: <class 'numpy.ndarray'>
#Length/Shape: 3


## check dimensions of each element

# For 'cls'
cls_array = np.array(hipt_data['cls'])
print(f"'cls' shape: {cls_array.shape}")

# For 'sub'
sub_array = np.array(hipt_data['sub'])
print(f"'sub' shape: {sub_array.shape}")

# For 'rgb'
print(f"'rgb' shape: {hipt_data['rgb'].shape}")

'''
For the HIPT output .pickle file

1. "cls" is coarse features of shape (192, 784, 848)
2. "sub" is fine features of shape (384, 784, 848)
3. "rgb" is rgb of shape (3, 784, 848)

'''

## Combine "cls" and "sub" from HIPT output to create an array of (192+384, 784, 848)
# Combine along the first axis
combined_hipt = np.concatenate([cls_array, sub_array], axis=0)  

#check shape (dimension)
print(combined_hipt.shape)
(576, 784, 848)

##save combined_hipt as a .pickle file for consistency
with open("/path/to/MANNER/visHD_CRC_P2/img_processed/square_008um/embeddings_hist_combined.pickle", 'wb') as f:
    pickle.dump(combined_hipt, f)


# Optional: Convert to a DataFrame (if meaningful for your analysis)
# Flatten each 2D (688, 832) feature into a single row
flattened_hipt = combined_hipt.reshape(combined_hipt.shape[0], -1)  # Shape: (192 + 384, 784*848)

# Create a DataFrame
df_hipt = pd.DataFrame(flattened_hipt)

# Inspect the resulting DataFrame
print(df_hipt.shape)  # (576, 664832 = 784*848)

#######################################################
## Read in HIPT rescaled locs file (HIPT rescale: x*(pixel-size-raw/pixel-size), which is near 0.5; 
## Need to divide coordinates by 16 again to match to HIPT image embedding coords)
hipt_locs_path = "/path/to/MANNER/visHD_CRC_P2/img_processed/square_008um/locs.parquet"

# Read the .parquet file into a DataFrame
hipt_locs = pd.read_parquet(hipt_locs_path)

# Display the first few rows
print(hipt_locs.head())
print(hipt_locs.shape)
# (344017, 2)

hipt_locs['x2'] = np.floor(hipt_locs['x'] / 16).astype(int)
hipt_locs['y2'] = np.floor(hipt_locs['y'] / 16).astype(int)

## Get range of X and Y
x_min, x_max = hipt_locs["x"].min(), hipt_locs["x"].max()
y_min, y_max = hipt_locs["y"].min(), hipt_locs["y"].max()
x2_min, x2_max = hipt_locs["x2"].min(), hipt_locs["x2"].max()
y2_min, y2_max = hipt_locs["y2"].min(), hipt_locs["y2"].max()

print(f"Range of x: {x_min} to {x_max}")
print(f"Range of y: {y_min} to {y_max}")
print(f"Range of x2: {x2_min} to {x2_max}")
print(f"Range of y2: {y2_min} to {y2_max}")
#Range of x: 324 to 13819
#Range of y: 3 to 12426
#Range of x2: 20 to 863
#Range of y2: 0 to 776

## add locs_hipt x and y to adata2.obs
# Add locs_hipt to adata2.obs with matching cell names

# Drop the old columns if they exist
adata2.obs = adata2.obs.drop(columns=['hipt_x', 'hipt_y'], errors='ignore')

# Add the new ones
adata2.obs = adata2.obs.join(
    hipt_locs[['x2', 'y2']].rename(columns={'x2': 'hipt_x', 'y2': 'hipt_y'})
)

# Display the first few rows of the modified AnnData object
print(adata2.obs.head())

##load mask created using HistoSweep (mask_small is super-pixel scale at HIPT feature extracted coordinates, while mask is he-raw coordinates)
he_mask= load_mask("/path/to/MANNER/visHD_CRC_P2/img_processed/square_008um/mask-small.png")
he_mask.shape
# (y, x) format
# (784, 848)


# Ensure hipt_x and hipt_y are integers and within bounds
x_coords = adata2.obs['hipt_x'].astype(int).values
y_coords = adata2.obs['hipt_y'].astype(int).values

# Initialize mask column with False by default
adata2.obs['mask'] = False

# Filter only valid coordinates within the he_mask shape 
valid_mask = (y_coords >= 0) & (y_coords < he_mask.shape[0]) & \
             (x_coords >= 0) & (x_coords < he_mask.shape[1])

# Apply mask values from he_mask
adata2.obs.loc[valid_mask, 'mask'] = he_mask[y_coords[valid_mask], x_coords[valid_mask]]

# Remove the 'in_adatatissue' column from obs
adata2.obs.drop(columns=["in_tissue"], inplace=True, errors="ignore")

# Convert array_row/col to integers
adata2.obs["array_row"] = adata2.obs["array_row"].astype('float32')
adata2.obs["array_col"] = adata2.obs["array_col"].astype('float32')

# Convert hipt_x/y to smaller ints
adata2.obs["hipt_x"] = adata2.obs["hipt_x"].astype('float32')
adata2.obs["hipt_y"] = adata2.obs["hipt_y"].astype('float32')

# Extract spatial coordinates
spatial = adata2.obsm['spatial']

# Add to obs as new columns
adata2.obs['raw_x'] = pd.Series(spatial[:, 0], index=adata2.obs.index).astype('float32')
adata2.obs['raw_y'] = pd.Series(spatial[:, 1], index=adata2.obs.index).astype('float32')

## remove adata2.obsm
adata2.obsm.clear()

## create masked subset
adata3 = adata2[adata2.obs['mask']].copy()


## Plot a marker gene heatmap to check proper application of the mask:
# Extract PIGR expression
PIGR_expr = (
    adata3[:, 'PIGR'].X.toarray().flatten()
    if hasattr(adata3[:, 'PIGR'].X, "toarray")
    else adata3[:, 'PIGR'].X.flatten()
)

# Create figure
fig, ax = plt.subplots()

# Plot heatmap using scatter
sc = ax.scatter(
    adata3.obs['hipt_x'],
    adata3.obs['hipt_y'],
    c=PIGR_expr,
    cmap='turbo',
    s=0.3, marker="o",
    edgecolor='none'
)

ax.set_title('PIGR Expression Heatmap')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.invert_yaxis()

# Add colorbar
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label('Expression level')

# Save the plot
save_path = '/path/to/MANNER/visHD_CRC_P2/img_processed/square_008um/masked_PIGR_heatmap.png'
plt.tight_layout()
plt.savefig(save_path, dpi=1000)
plt.close()

print(f"✅ Saved PIGR heatmap to: {save_path}")

# Save the preprocessed visium HD data with HIPT-scaled spatial information and Histosweep mask label
MANNER_dir = "/path/to/MANNER/visHD_CRC_P2/MANNER_preprocessed/"
#output_full_filename = "visHD_crc_p2_fulldata_with_hipt_spatial.h5ad"
output_masked_filename = "visHD_crc_p2_fulldata_masked_with_hipt_spatial.h5ad"
os.makedirs(MANNER_dir, exist_ok=True)

#adata2.write_h5ad(MANNER_dir + output_full_filename, compression='gzip')
adata3.write_h5ad(MANNER_dir + output_masked_filename, compression='gzip')

#######################################################
## Create a binary adjacency matrix based on histology features
#######################################################
## load gene expression dataset (with KNN info)
MANNER_dir = "/path/to/MANNER/visHD_CRC_P2/MANNER_preprocessed/"
gene_exp_filename = "visHD_crc_p2_fulldata_masked_with_hipt_spatial.h5ad"

## adata_raw is the full dataset of 540K spots without log1p transformation
adata_raw = read_h5ad(MANNER_dir + gene_exp_filename)
print(adata_raw)

##check mitochondrial genes ("MT-")
mt_genes = adata_raw.var_names[adata_raw.var_names.str.startswith('MT-')]
print(f"Number of mitochondrial genes: {len(mt_genes)}")
print(mt_genes.tolist())
'''
Number of mitochondrial genes: 11
['MT-ND1', 'MT-ND2', 'MT-CO2', 'MT-ATP6', 'MT-CO3', 'MT-ND3', 'MT-ND4L', 'MT-ND4', 'MT-ND5', 'MT-ND6', 'MT-CYB']

'''

# Keep only non-mitochondrial genes (remove 11 genes with "MT-")
adata_raw = adata_raw[:, ~adata_raw.var_names.str.startswith('MT-')].copy()

# If adata.X is dense
if not hasattr(adata_raw.X, "toarray"):
    zero_genes = np.sum(np.all(adata_raw.X == 0, axis=0))
else:
    # If adata.X is sparse
    zero_genes = np.sum(np.array((adata_raw.X == 0).sum(axis=0)).flatten() == adata_raw.n_obs)

print("Number of genes with 0 expression in all spots:", zero_genes)


## load HIPT data
hipt_path = "/path/to/MANNER/visHD_CRC_P2/img_processed/square_008um/embeddings_hist_combined.pickle"

# Open the file in read-binary mode and load it
with open(hipt_path, "rb") as file:
    hipt_data = pickle.load(file)


# Range of hipt_x and hipt_y in adata.obs
hipt_x_range = (adata_raw.obs['hipt_x'].min(), adata_raw.obs['hipt_x'].max())
hipt_y_range = (adata_raw.obs['hipt_y'].min(), adata_raw.obs['hipt_y'].max())
print(f"Range of hipt_x in adata.obs: {hipt_x_range}")
print(f"Range of hipt_y in adata.obs: {hipt_y_range}")

# Range of x and y in hipt_data (embedding, Y coord, X coord)
hipt_data_x_range = (0, hipt_data.shape[2] - 1)  # Assuming x corresponds to the 3rd dimension
hipt_data_y_range = (0, hipt_data.shape[1] - 1)  # Assuming y corresponds to the 2nd dimension
print(f"Range of x in hipt_data: {hipt_data_x_range}")
print(f"Range of y in hipt_data: {hipt_data_y_range}")

'''
Range of hipt_x in adata.obs: (20.0, 845.0)
Range of hipt_y in adata.obs: (1.0, 776.0)
Range of x in hipt_data: (0, 847)
Range of y in hipt_data: (0, 783)
'''

##################################################################################################################################
## create_GCN_random_batch function can split gene expression and adjacency matrix into batches
## To use the full data, set batch_size=0 (default)
##################################################################################################################################
create_GCN_random_batch(adata=adata_raw.copy(), hipt_data=hipt_data, 
                 n_hipt_PC = 50, remove_low_genes=False, batch_size = 0, 
                 n_XY_knn=None, n_HIPT_knn=25,
                 batch_outdir="/path/to/MANNER/visHD_CRC_P2/MANNER_preprocessed/", 
                 output_header="visHD_crc_p2_masked_fulldata", verbose=True, apply_log1p=False)
