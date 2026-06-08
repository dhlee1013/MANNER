
from Get_Disp import *
from Decoder_NBC import *
# from PE_NODE_Embedding import *
# from sklearn.decomposition import PCA
from datetime import timedelta
import scanpy as sc
import os
import anndata as ad


def NormalizeMat(Xmat):
    Means = np.mean(Xmat, axis=0)
    Stds = np.std(Xmat, axis=0)
    NMat = (Xmat - Means) / Stds
    return NMat


## Decoder_NB_GCN takes in GCN embedding as X1
def Decoder_NB_GCN(adata, device, min_cells=10, embed_size=64, batch_size=64, n_hidden1=128, n_hidden2=256, detect_outliers=False,
               learning_rate=1e-4, max_epochs=100, min_delta=1e-4, patience=6):
    print("No normalization or log1p transformation will be performed!")
    print("Filter out zero expressed genes")
    ## remove genes that are expressed in less than min_cell number of cells
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"Number of genes remaining after filtering (min_cells={min_cells}): {adata.n_vars}")

    print("Detect the outlier genes: input must be RAW COUNTS!")
    adata2 = Detect_Genes(adata, nper = 1000, quantile = 0.95, permutated="gene", threshold_value=0.99)

    print("Prepare the input for de-noising")
    if detect_outliers:
        den_ids = np.where(adata2.var["is.DEN"] != 0)[0]
        den_adata = adata2[:, den_ids]
    else:
        den_ids = np.arange(adata2.shape[1])
        den_adata = adata2

    # Ensure data is on the correct device
    Ymat = den_adata.X  # X must be the count matrix
    # Move to GPU as integer
    Y = torch.tensor(Ymat, dtype=torch.int32).to(device)

    # Use GCN embedding as input
    print("Using GCN embedding as input")
    X1 = torch.tensor(adata2.obsm["X_gcn"], dtype=torch.float32, device=device)

    # Ensure rho is on the correct device
    rho = torch.tensor(adata2.uns["optimized_rho"], dtype=torch.float32, device=device)

    n_sample, n_gene = Y.shape
    Trdataset = TrDataset(X1=X1, Y=Y)
    nval = int(len(Y) * 0.2)
    ntrain = len(Y) - nval
    train_set, val_set = random_split(Trdataset, [ntrain, nval])

    Train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    Val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=min_delta, patience=patience, verbose=True, mode="min")

    # Check the number of available GPUs
    # num_gpus = torch.cuda.device_count()
    num_gpus = 1

    if num_gpus > 1:
        # Set the strategy based on the number of GPUs
        strategy = 'ddp'  

        # Initialize the Trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[early_stop_callback],
            accelerator='gpu',  # Use GPU
            devices=num_gpus,   # Use all detected GPUs
            strategy=strategy    # Use appropriate strategy
        )
    else:
        # Initialize the Trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[early_stop_callback],
            accelerator='gpu',  # Use GPU
            devices=num_gpus   # Use all detected GPUs
        )

    # Ensure the model is on the correct device (DDP will handle it across GPUs)
    plmodel = NBR_plnet(n_embed=embed_size, n_hidden1=n_hidden1, n_hidden2=n_hidden2, n_output=n_gene, rho=rho, learning_rate=learning_rate)
    trainer.fit(plmodel, train_dataloaders=Train_dataloader, val_dataloaders=Val_dataloader)

    print("Calculating denoised matrix")
    pred_con_gpu, pred_rate_gpu = plmodel(X1)

    ##################################################################################
    ## Feb 2025: Batch-wise ypreds, pred_rate, and pred_con calculation to reduce memory usage
    ##################################################################################
    ## Calculate ypreds in batches to reduce memory burden
    batch_size = 5000  # Adjust based on available memory
    ypreds_list = []
    pred_rate_list = []
    pred_con_list = []

    for i in range(0, Y.shape[0], batch_size):
        batch_pred_con = pred_con_gpu[i:i + batch_size]
        batch_pred_rate = pred_rate_gpu[i:i + batch_size]

        # The mean (mu) of the Gamma-Poisson distribution is alpha / beta (or con / rate)
        batch_ypreds = batch_pred_con / batch_pred_rate  # Compute ypreds


        # Move results to CPU and convert to NumPy
        ypreds_list.append(batch_ypreds.detach().cpu().numpy())
        pred_con_list.append(batch_pred_con.detach().cpu().numpy())
        pred_rate_list.append(batch_pred_rate.detach().cpu().numpy())


    # Reassemble full matrix in CPU memory
    ypreds = np.vstack(ypreds_list)
    pred_con = np.vstack(pred_con_list)
    pred_rate = np.vstack(pred_rate_list)

    # Free memory
    del pred_con_gpu, pred_rate_gpu
    del ypreds_list, pred_con_list, pred_rate_list, batch_pred_con, batch_pred_rate
    torch.cuda.empty_cache()
    ##################################################################################      
    print("Empirical bayes estimation")
    Y = Y.detach().cpu().numpy()
    ypost = (Y + pred_rate * ypreds) / (pred_rate + 1)
    ## save the denoised gene expression as float32 after rounding to 4 decimals
    adata2[:, den_ids] = np.round(ypost, 4).astype(np.float32)

    ## round to 3 decimals and save as float32
    ypreds = np.round(ypreds, 3).astype(np.float32)
    pred_con = np.round(pred_con, 3).astype(np.float32)
    pred_rate = np.round(pred_rate, 3).astype(np.float32)

    adata2.uns["predicted"] = ypreds
    adata2.uns["pred_con"] = pred_con
    adata2.uns["pred_rate"] = pred_rate

    return adata2