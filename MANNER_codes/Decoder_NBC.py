import torch
from pyro.distributions import GammaPoisson
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import pytorch_lightning as pl
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping ##for pytorch_lightning < 1.5
from pytorch_lightning.callbacks import EarlyStopping

#############################################################
#############################################################

## Negative log-likelihood for Gamma-Poisson mixture = NegBin distribution for gene expression count data
def gammapoisson_nll(concentration, rate, y):
    dist = GammaPoisson(concentration=concentration, rate=rate)
    loss = (-1) * dist.log_prob(y)
    return loss.mean()

#############################################################
##April 2025: updated code to change KL-divergence loss part
##Sept 2025: changed exp(con) and exp(rate) to softplus(con) and softplus(rate) for numerical stability

class NBR_plnet(pl.LightningModule):
    def __init__(self, n_embed, n_hidden1, n_hidden2, n_output, rho, learning_rate=1e-5):
        super(NBR_plnet, self).__init__()
        self.learning_rate = learning_rate
        self.hidden1 = torch.nn.Linear(n_embed, n_hidden1)  # hidden layer1
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)  # hidden layer2
        self.predict_con = torch.nn.Linear(n_hidden2, n_output)  # output layer for concentration parameter
        self.predict_rate = torch.nn.Linear(n_hidden2, n_output)  # output layer for rate parameter
        self.rho = rho.to(self.device)  # Move rho to the correct device when created

        ## small epsilon to prevent log(0) issues
        self.eps = 1e-6

    def forward(self, x1):
        # Move input to the same device as the model
        x1 = x1.to(self.device)
        # Ensure rho is also on the same device
        self.rho = self.rho.to(self.device)

        x = F.relu(self.hidden1(x1))
        x = F.relu(self.hidden2(x))

        con_raw = self.predict_con(x)
        rate_raw = self.predict_rate(x)

        ##use softplus(x)= ln(1+exp(x)) instead of just x for numerical stability
        cal_con = F.softplus(con_raw)+self.eps
        cal_rate = F.softplus(rate_raw)+self.eps

        # clamp to avoid extremely small or large values
        cal_con = torch.clamp(cal_con, min=self.eps, max=1e6)
        cal_rate = torch.clamp(cal_rate, min=self.eps, max=1e6)

        return cal_con, cal_rate

    def training_step(self, batch, batch_idx):
        x1, y = batch
        x1 = x1.to(self.device)  # Ensure input tensors are on the correct device
        y = y.to(self.device)

        cal_con, cal_rate = self.forward(x1)
        ## no need for exponentition since softplus ensures positive con and rate parameter for gamma-poisson nll
        loss1 = gammapoisson_nll(concentration=cal_con, rate=cal_rate, y=y)
        mu = torch.round(cal_con / cal_rate)

        ##dropout/KL computation
        binary_mat = (y == 0).type(torch.FloatTensor).to(self.device)  # Ensure binary_mat is on the correct device
        dropout = torch.mean(binary_mat, dim=0).to(self.device)  # Ensure dropout is on the correct device
        pred_dropout = torch.pow(self.rho / (torch.mean(mu, dim=0) + self.rho), self.rho).to(self.device)  # Ensure pred_dropout is on the correct device

        # April 2025: Clamp to avoid log(0)
        dropout = torch.clamp(dropout, min=self.eps, max=1.0 - self.eps)
        pred_dropout = torch.clamp(pred_dropout, min=self.eps, max=1.0 - self.eps)

        # April 2025: Updated KL divergence loss
        loss2 = F.kl_div(pred_dropout.log(), dropout, reduction='sum', log_target=False)

        loss = loss1 + loss2
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False)

        # --- PRINT INITIAL LOSS ---
        if batch_idx == 0 and self.current_epoch == 0:
            print(f"⚡ Initial training loss (first batch, epoch 0): {loss.item():.4f}")

        return loss

    def validation_step(self, batch, batch_idx):
        x1, y = batch
        x1 = x1.to(self.device)  # Ensure input tensors are on the correct device
        y = y.to(self.device)

        cal_con, cal_rate = self.forward(x1)

        loss1 = gammapoisson_nll(concentration=cal_con, rate=cal_rate, y=y)

        mu = torch.round(cal_con / cal_rate)

        binary_mat = (y == 0).type(torch.FloatTensor).to(self.device)  # Ensure binary_mat is on the correct device
        dropout = torch.mean(binary_mat, dim=0).to(self.device)  # Ensure dropout is on the correct device
        pred_dropout = torch.pow(self.rho / (torch.mean(mu, dim=0) + self.rho), self.rho).to(self.device)  # Ensure pred_dropout is on the correct device

        ##April 2025: clamp values to prevent log(0) in kullback-leibler KL = sum_g[dropout_g * log(dropout_g / pred_dropout_g)]
        dropout = torch.clamp(dropout, min=self.eps, max=1.0 - self.eps)
        pred_dropout = torch.clamp(pred_dropout, min=self.eps, max=1.0 - self.eps)

        ##April 2025: Kullback-Leibler loss for pred vs observed dropout rates
        loss2 = F.kl_div(pred_dropout.log(), dropout, reduction='sum',log_target=False)

        loss = loss1 + loss2
        
        # Debug info if loss is NaN or inf
        if not torch.isfinite(loss):
            print(f"\n⚠️ Non-finite loss in validation batch {batch_idx}")
            print(f"  loss1: {loss1.item()}, loss2: {loss2.item()}")
            print(f"  dropout (mean): {dropout.mean().item():.4f}")
            print(f"  pred_dropout (mean): {pred_dropout.mean().item():.4f}")
            print(f"  dropout values (first 10): {dropout[:10].cpu().numpy()}")
            print(f"  pred_dropout values (first 10): {pred_dropout[:10].cpu().detach().numpy()}")
            print(f"  mu stats → min: {mu.min().item()}, max: {mu.max().item()}, mean: {mu.float().mean().item():.4f}")
            return None

        # --- PRINT INITIAL VALIDATION LOSS ---
        if batch_idx == 0 and self.current_epoch == 0:
            print(f"⚡ Initial validation loss (first batch, epoch 0): {loss.item():.4f}")

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class TrDataset(torch.utils.data.Dataset):
    """ Information about the datasets """
    def __init__(self, X1, Y):
        self.x1 = X1
        self.y = Y

    def __getitem__(self, index):
        return self.x1[index], self.y[index]

    def __len__(self):
        return len(self.x1)


