#!/usr/bin/env python3

from netCDF4 import Dataset as NC
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from scipy.stats import dirichlet
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_eigenvectors(omegas, F, cutoff=0.999):
    print(f"Getting Eigenvectors with cutoff {cutoff}")
    F_mean = (F * omegas).sum(axis=0)
    F_bar = F - F_mean  # Eq. 28
    S = F_bar.T @ torch.diag(omegas.squeeze()) @ F_bar  # Eq. 27
    lamda, V = torch.eig(S, eigenvectors=True)  # Eq. 26
    lamda = lamda[:, 0].squeeze()

    cutoff_index = torch.sum(torch.cumsum(lamda / lamda.sum(), 0) < cutoff)
    lamda_truncated = lamda.detach()[:cutoff_index]
    V = V.detach()[:, :cutoff_index]
    V_hat = V @ torch.diag(torch.sqrt(lamda_truncated))
    # A slight departure from the paper: Vhat is the
    # eigenvectors scaled by the eigenvalue size.  This
    # has the effect of allowing the outputs of the neural
    # network to be O(1).  Otherwise, it doesn't make
    # any difference.
    return V_hat, F_bar, F_mean


class Emulator(nn.Module):
    def __init__(self, n_parameters, n_eigenvectors, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, V_hat, F_mean):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.l_1 = nn.Linear(n_parameters, n_hidden_1)
        self.norm_1 = nn.LayerNorm(n_hidden_1)
        self.dropout_1 = nn.Dropout(p=0.0)
        self.l_2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.norm_2 = nn.LayerNorm(n_hidden_2)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.l_3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.norm_3 = nn.LayerNorm(n_hidden_3)
        self.dropout_3 = nn.Dropout(p=0.5)
        self.l_4 = nn.Linear(n_hidden_3, n_hidden_4)
        self.norm_4 = nn.LayerNorm(n_hidden_3)
        self.dropout_4 = nn.Dropout(p=0.5)
        self.l_5 = nn.Linear(n_hidden_4, n_eigenvectors)

        self.V_hat = torch.nn.Parameter(V_hat, requires_grad=False)
        self.F_mean = torch.nn.Parameter(F_mean, requires_grad=False)

    def forward(self, x, add_mean=False):
        # Pass the input tensor through each of our operations

        a_1 = self.l_1(x)
        a_1 = self.norm_1(a_1)
        a_1 = self.dropout_1(a_1)
        z_1 = torch.relu(a_1)

        a_2 = self.l_2(z_1)
        a_2 = self.norm_2(a_2)
        a_2 = self.dropout_2(a_2)
        z_2 = torch.relu(a_2) + z_1

        a_3 = self.l_3(z_2)
        a_3 = self.norm_3(a_3)
        a_3 = self.dropout_3(a_3)
        z_3 = torch.relu(a_3) + z_2

        a_4 = self.l_4(z_3)
        a_4 = self.norm_3(a_4)
        a_4 = self.dropout_3(a_4)
        z_4 = torch.relu(a_4) + z_3

        z_5 = self.l_5(z_4)
        if add_mean:
            F_pred = z_5 @ self.V_hat.T + self.F_mean
        else:
            F_pred = z_5 @ self.V_hat.T

        return F_pred


def criterion_ae(F_pred, F_obs, omegas, area):
    instance_misfit = torch.sum(torch.abs((F_pred - F_obs)) ** 2 * area, axis=1)
    return torch.sum(instance_misfit * omegas.squeeze())


def train_surrogate(e, X_train, F_train, omegas, area, batch_size=128, epochs=3000, eta_0=0.01, k=1000.0):

    print(f"Training the surrogate model with batch size {batch_size} and {epochs} epochs")
    omegas_0 = torch.ones_like(omegas) / len(omegas)
    training_data = TensorDataset(X_train, F_train, omegas)

    train_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(e.parameters(), lr=eta_0, weight_decay=0.0)

    # Loop over the data
    for epoch in range(epochs):
        # Loop over each subset of data
        for param_group in optimizer.param_groups:
            param_group["lr"] = eta_0 * (10 ** (-epoch / k))

        for x, f, o in train_loader:
            e.train()
            # Zero out the optimizer's gradient buffer
            optimizer.zero_grad()

            f_pred = e(x)

            # Compute the loss
            loss = criterion_ae(f_pred, f, o, area)

            # Use backpropagation to compute the derivative of the loss with respect to the parameters
            loss.backward()

            # Use the derivative information to update the parameters
            optimizer.step()

        e.eval()
        F_train_pred = e(X_train)
        # Make a prediction based on the model
        loss_train = criterion_ae(F_train_pred, F_train, omegas, area)
        # Make a prediction based on the model
        loss_test = criterion_ae(F_train_pred, F_train, omegas_0, area)

        # Print the epoch, the training loss, and the test set accuracy.
        if epoch % 10 == 0:
            print(epoch, loss_train.item(), loss_test.item())


grid_res = 4000
nc0 = NC("calibration_samples/ltop_calibration_sample_0.nc")
m_gridpoints = len(nc0.variables["precipitation"][:].ravel())
nc0.close()

ensemble_file = "ltop_calibration_samples_100.csv"
samples = pd.read_csv(ensemble_file)
X = samples.values[:, 1::]
m_samples, n_parameters = X.shape

P = np.zeros((m_samples, m_gridpoints))
print("Reading calibration samples")
for idx in tqdm(range(m_samples)):
    m_file = f"calibration_samples/ltop_calibration_sample_{idx}.nc"
    nc = NC(m_file)
    p = nc.variables["precipitation"][:]
    P[idx, :] = p.ravel()
    nc.close()

P[P == 0] = 1e-5
P = P[:, ::4]
m_gridpoints_prunded = P.shape[1]
point_area = np.ones(m_gridpoints_prunded) * grid_res ** 2

X = torch.from_numpy(X)
F_lin = torch.from_numpy(P)
F = torch.log10(F_lin)
point_area = torch.from_numpy(point_area)

X = X.to(torch.float32)
F = F.to(torch.float32)

X = X.to(device)
F = F.to(device)

X_hat = torch.log10(X)

point_area.to(device)
normed_area = point_area / point_area.sum()

torch.manual_seed(0)
np.random.seed(0)

n_hidden_1 = 128
n_hidden_2 = 128
n_hidden_3 = 128
n_hidden_4 = 128

n_models = 1  # To reproduce the paper, this should be 50
for model_index in range(n_models):
    print(f"Running model {model_index+1}/{n_models}")
    omegas = torch.tensor(dirichlet.rvs(np.ones(m_samples)), dtype=torch.float, device=device).T

    V_hat, F_bar, F_mean = get_eigenvectors(omegas, F)
    n_eigenvectors = V_hat.shape[1]

    e = Emulator(n_parameters, n_eigenvectors, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, V_hat, F_mean)
    e.to(device)

    train_surrogate(e, X_hat, F_bar, omegas, normed_area)
    torch.save(e.state_dict(), f"emulator_ensemble/emulator_{model_index}.h5")
