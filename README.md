# MANNER: Morphology-Aware Neural Network for Expression Recovery
MANNER is a deep learning framework for spatial transcriptomics denoising guided by morphological features extracted from paired histology images.
**Manuscript:** [Link coming soon]

## Installation

### MANNER Installation
To install MANNER, open a terminal (Linux/macOS) or Anaconda Prompt (Windows), and run:
```bash
git clone https://github.com/your-repo/MANNER.git
cd MANNER
conda env create -f MANNER_environment.yaml
conda activate MANNER
```

> GPU acceleration is strongly recommended. CUDA 11.3 and a compatible NVIDIA driver (≥ 450.80) are required.

### External Dependencies (Histosweep and iStar)
MANNER relies on two external tools for histology image processing and feature extraction:
- [Histosweep](https://github.com/amesch441-o1/HistoSweep) to create tissue masks. Please follow the installation instructions in the HistoSweep repository.
  
- [iStar](https://github.com/daviddaiweizhang/istar/tree/master) to process and perform [HIPT](https://github.com/mahmoodlab/HIPT)-based image feature extraction.

Morphological feature extraction is performed via [iStar](https://github.com/daviddaiweizhang/istar/tree/master), which implements [HIPT](https://github.com/mahmoodlab/HIPT)-based patch-level image embeddings. iStar should be installed as a **separate conda environment** from MANNER.

> **Note:** MANNER relies on slightly modified versions of iStar's `rescale.py` and `utils.py`. These are provided under the `istar_mod/` folder in this repository and should be used in place of the originals.

---
## Tutorial

A complete end-to-end tutorial is provided using the publicly available [Visium HD Human Colorectal Cancer Patient 2 (Sample P2 CRC)](https://www.10xgenomics.com/platforms/visium/product-family/dataset-human-crc) dataset.

> All Visium HD analyses in the MANNER manuscript were performed on 8 µm binned gene expression data. MANNER is also compatible with 2 µm and 16 µm resolutions, provided that the paired histology image is scaled accordingly during preprocessing.

### Step 1 — Preprocessing

The preprocessing pipeline:
1. Generates a tissue mask from the paired H&E image (`.btf`) using HistoSweep
2. Extracts patch-level morphological features for each spatial bin using iStar/HIPT
3. Constructs a binary adjacency matrix from morphological features for use as GCN input

A fully annotated preprocessing script is provided at: visHD_CRC_P2_input_preprocessing.py

### Step 2 — GCN Training and Embedding Extraction

Using the preprocessed data as input, a Graph Convolutional Network (GCN) is trained and a 64-dimensional spatial embedding is extracted per bin.

A fully annotated training script is provided at: visHD_CRC_P2_GCN_training.py

### Step 3 — MANNER Denoising

Once the gene expression matrix and GCN embeddings are prepared, run MANNER denoising as follows:

```bash
python /path/to/MANNER/MANNER_code/Run_MANNER.py \
    /path/to/MANNER/visHD_CRC_P2/MANNER_preprocessed/GCN_embedding/visHD_crc_p2_300_gene_PCs_GCN_embedding_cleaned.h5ad \
    /path/to/MANNER/visHD_CRC_P2/MANNER_output/visHD_crc_p2_300_gene_PCs_MANNER_denoised.h5ad
```

**Arguments:**
| Position | Description |
|----------|-------------|
| 1 | Path to preprocessed `.h5ad` file containing raw gene expression count and GCN embeddings |
| 2 | Path for the denoised output `.h5ad` file |

---









