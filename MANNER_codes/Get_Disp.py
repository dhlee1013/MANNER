import scanpy as sc
from scipy.sparse import issparse
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


## Detect_Genes calculated the rho (dropout rate parameter)
## Detect_Genes need RAW COUNTS to meet NegBin Assumption
## DO NOT log(1+x) transform gene expression for Detect_Genes

def Detect_Genes(adata, nper = 1000, quantile = 0.95, permutated="gene", threshold_value=0.99):
    ## threshold_value is used to remove extreme outliers during bootstrapping
    # X must be the count matrix
    print("fit regression and calculate dropout rate")
    X = adata.X
    nspot, ngene = adata.shape
    if issparse(X):
        X = X.A
    mus = X.mean(axis=0)
    vars = X.var(axis=0)

    # fit the regression
    ys = vars - mus
    ################################################################################
    ## March 2025 addition to remove extreme outliers from linear regression for rho
    ## We mask out top 1% outliers ONLY for RHO ESTIMATION
    ## We still estimate dropout rates for all genes, without masking!!!
    # Compute threshold to remove top 1% outliers
    threshold = np.percentile(ys, threshold_value*100)

    # Create mask to exclude top 1% of ys values
    mask = ys < threshold  # shape: (ngene,)

    # Prepare filtered regression inputs
    mus2_masked = (mus**2)[mask]
    ys_masked = ys[mask]

    # Reshape for sklearn
    XX = mus2_masked.reshape(-1, 1)
    y = ys_masked.reshape(-1, 1)

    ##number of gene after filtering out outliers; this will be used for boostrapping
    n_gene_masked = XX.shape[0]
    ################################################################################

    lm_fit = LinearRegression(fit_intercept=False)
    lm_model = lm_fit.fit(X=XX, y=y)
    rho = 1/lm_model.coef_
    print(f"Observed Dropout Parameter (rho)= {rho}")
    observedDropouts = np.mean(X==0, axis=0)
    predDropouts = (rho/(mus+rho))**rho
    predDropouts = predDropouts.reshape(-1)

    print(f"NaNs in predDropouts: {np.isnan(predDropouts).sum()}")
    print(f"Positive infs in predDropouts: {np.isposinf(predDropouts).sum()}")
    print(f"Negative infs in predDropouts: {np.isneginf(predDropouts).sum()}")

    if np.any(predDropouts < 0):
        print("predDropouts contains negative values!")
    else:
        print("predDropouts has no negative values.")

    print("Start the bootstrap")
    preds = []

    i = 0  # valid iteration counter
    attempts = 0  # total attempts

    while i < nper:
        attempts += 1

        try:
            if permutated == "gene":
                #ids = np.random.choice(ngene, ngene)
                ids = np.random.choice(n_gene_masked,n_gene_masked)
                ## since we already masked out 1% outlier in XX and y, Xnew and ynew are already outlier-filtered
                Xnew = XX[ids]
                ynew = y[ids]

            else:
                ids = np.random.choice(nspot, nspot)
                Xmat = X[ids, :]
                New_mus = Xmat.mean(axis=0)
                New_vars = Xmat.var(axis=0)
                New_ys = New_vars - New_mus
                New_mus2 = New_mus**2
                Xnew = New_mus2.reshape(len(mus), 1)
                ynew = New_ys.reshape(len(mus), 1)

                ##March 2020: outlier masking set by threshold_value parameter
                threshold = np.percentile(ynew, threshold_value * 100)
                mask = ynew.flatten() < threshold
                Xnew = Xnew[mask]
                ynew = ynew[mask]

            lm_model_new = LinearRegression(fit_intercept=False).fit(X=Xnew, y=ynew)
            rho_new = 1 / lm_model_new.coef_

            # Check for bad rho
            if not np.isfinite(rho_new).all() or np.any(rho_new <= 0):
                print(f"Invalid rho in attempt {attempts}. Skipping...")
                continue

            # Predict dropout
            with np.errstate(divide='ignore', invalid='ignore'):
                predDropoutsNew = (rho_new / (mus + rho_new)) ** rho_new

            # Check prediction validity
            if not np.isfinite(predDropoutsNew).all():
                print(f"NaNs or infs in predDropoutsNew in attempt {attempts}. Skipping...")
                continue

            preds.append(predDropoutsNew.reshape(-1))
            i += 1  # Only increment if successful

        except Exception as e:
            print(f"Exception in bootstrap attempt {attempts}: {e}")
            continue

    print(f"Completed {nper} valid bootstraps in {attempts} attempts.")


    preds = pd.DataFrame(np.array(preds), columns=adata.var_names)
    quantile1 = 1 - quantile
    quantile2 = quantile
    df_quantile = preds.quantile([quantile1, 0.5, quantile2])
    lower_bound = df_quantile.iloc[0, :]
    middle_bound = df_quantile.iloc[1, :]
    upper_bound = df_quantile.iloc[2, :]
    adata.var["mu"] = mus
    adata.var["{:.2f}".format(quantile1)] = lower_bound
    adata.var["{:.2f}".format(0.5)] = middle_bound
    adata.var["{:.2f}".format(quantile2)] = upper_bound
    adata.var['Dropout'] = observedDropouts
    adata.var['FittedDropout'] = predDropouts

    from scipy.optimize import fmin_l_bfgs_b as optim
    
    ## April 2025: new opt_f to handle extreme rho values
    def opt_f(params, *args):
        rho_opt = np.clip(params[0], 1e-6, 1e3)  # Constrain rho to valid range
        middle_bound = np.array(args[0])
        mus = np.array(args[1])
    
        epsilon = 1e-6
        try:
            predicted = (rho_opt / (mus + rho_opt + epsilon)) ** rho_opt
            if not np.all(np.isfinite(predicted)):
                return np.inf  # Penalize invalid values
            cost_res = np.sum((middle_bound - predicted) ** 2)
        except Exception as e:
            print(f"Exception in opt_f: {e}")
            cost_res = np.inf  # Return large cost if something breaks
        return cost_res

    ## April 2025: Bound rho_opt_res to prevent errors
    #rho_opt_res = optim(opt_f, x0 = rho, args=(middle_bound,mus), approx_grad=1)
    rho_opt_res = optim(opt_f, x0=rho, args=(middle_bound, mus), bounds=[(1e-6, 1e3)], approx_grad=1)

    rho_opt = rho_opt_res[0]
    ix_impute = np.where(observedDropouts > upper_bound)[0]
    ix_downgrade = np.where(observedDropouts < lower_bound)[0]
    print(f"{len(ix_impute) + len(ix_downgrade)} genes need to be de-noised.")
    print(f"Among them, {len(ix_impute)} will be **imputed** (observed dropout > expected)")
    print(f"And {len(ix_downgrade)} will be **downgraded** (observed dropout < expected)")

    den_indicator = np.zeros(ngene)
    den_indicator[ix_impute] = 2
    den_indicator[ix_downgrade] = 1
    adata.var["is.DEN"] = den_indicator
    adata.uns["optimized_rho"] = rho_opt

    # Remove intermediate variables to save memory
    # del adata.var["mu"]
    del adata.var["{:.2f}".format(quantile1)]
    del adata.var["{:.2f}".format(0.5)]
    del adata.var["{:.2f}".format(quantile2)]

    return adata


if __name__=="__main__":
    adata = sc.read_h5ad("data/IDC.h5ad")
    res = Detect_Genes(adata, permutated="spot")
    res.write_h5ad("data/IDC_detectedd_by_spot.h5ad")
    print(res)