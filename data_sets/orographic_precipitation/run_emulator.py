#!/usr/bin/env python3

from netCDF4 import Dataset as NC
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from scipy.stats import dirichlet
from sklearn.decomposition import PCA
from tqdm import tqdm
from time import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_eigenvectors(omegas, F, cutoff=0.999):
    start = time()
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
    end = time()
    print(f"It took {end - start} seconds!")
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


def draw_sample(mu, cov, eps=1e-10):
    L = torch.cholesky(cov + eps * torch.eye(cov.shape[0], device=device))
    return mu + L @ torch.randn(L.shape[0], device=device)


def get_proposal_likelihood(Y, mu, inverse_cov, log_det_cov):
    return -0.5 * log_det_cov - 0.5 * (Y - mu) @ inverse_cov @ (Y - mu)


def MALA_step(X, h, local_data=None):
    if local_data is not None:
        pass
    else:
        local_data = get_log_like_gradient_and_hessian(V, X, compute_hessian=True)

    log_pi, g, H, Hinv, log_det_Hinv = local_data

    X_ = draw_sample(X, 2 * h * Hinv).detach()
    X_.requires_grad = True

    log_pi_ = get_log_like_gradient_and_hessian(V, X_, compute_hessian=False)

    logq = get_proposal_likelihood(X_, X, H / (2 * h), log_det_Hinv)
    logq_ = get_proposal_likelihood(X, X_, H / (2 * h), log_det_Hinv)

    log_alpha = -log_pi_ + logq_ + log_pi - logq
    alpha = torch.exp(min(log_alpha, torch.tensor([0.0], device=device)))
    u = torch.rand(1, device=device)
    if u <= alpha and log_alpha != np.inf:
        X.data = X_.data
        local_data = get_log_like_gradient_and_hessian(V, X, compute_hessian=True)
        s = 1
    else:
        s = 0
    return X, local_data, s


def MALA(
    X,
    n_iters=10001,
    h=0.1,
    acc_target=0.25,
    k=0.01,
    beta=0.99,
    sample_path="./samples/",
    model_index=0,
    save_interval=1000,
    print_interval=50,
):
    print("***********************************************")
    print("***********************************************")
    print("Running Metropolis-Adjusted Langevin Algorithm for model index {0}".format(model_index))
    print("***********************************************")
    print("***********************************************")
    local_data = None
    vars = []
    acc = acc_target
    for i in range(n_iters):
        X, local_data, s = MALA_step(X, h, local_data=local_data)
        vars.append(X.detach())
        acc = beta * acc + (1 - beta) * s
        h = min(h * (1 + k * np.sign(acc - acc_target)), 1)
        if i % print_interval == 0:
            print("===============================================")
            print("sample: {0:d}, acc. rate: {1:4.2f}, log(P): {2:6.1f}".format(i, acc, local_data[0].item()))
            print(f"curr. m: {X.data.cpu().numpy()}")
            print("===============================================")

        if i % save_interval == 0:
            print("///////////////////////////////////////////////")
            print("Saving samples for model {0:03d}".format(model_index))
            print("///////////////////////////////////////////////")
            X_posterior = torch.stack(vars).cpu().numpy()
            np.save(open(sample_path + "X_posterior_model_{0:03d}.npy".format(model_index), "wb"), X_posterior)
    X_posterior = torch.stack(vars).cpu().numpy()
    return X_posterior


def find_MAP(X, n_iters=50, print_interval=10):
    print("***********************************************")
    print("***********************************************")
    print("Finding MAP point")
    print("***********************************************")
    print("***********************************************")
    # Line search distances
    alphas = np.logspace(-4, 0, 11)
    # Find MAP point
    for i in range(n_iters):
        log_pi, g, H, Hinv, log_det_Hinv = get_log_like_gradient_and_hessian(V, X, compute_hessian=True)
        p = Hinv @ -g
        alpha_index = np.nanargmin(
            [
                get_log_like_gradient_and_hessian(V, X + alpha * p, compute_hessian=False).detach().cpu().numpy()
                for alpha in alphas
            ]
        )
        mu = X + alphas[alpha_index] * p
        X.data = mu.data
        if i % print_interval == 0:
            print("===============================================")
            print(f"iter: {i}, ln(P): {log_pi}, curr. m: {X.data.cpu().numpy()}")
            print("===============================================")
    return X


def get_log_like_gradient_and_hessian(V, X, eps=1e-2, compute_hessian=False):
    log_pi = V(X)
    if compute_hessian:
        g = torch.autograd.grad(log_pi, X, retain_graph=True, create_graph=True)[0]
        H = torch.stack([torch.autograd.grad(e, X, retain_graph=True)[0] for e in g])
        lamda, Q = torch.eig(H, eigenvectors=True)
        lamda_prime = torch.sqrt(lamda[:, 0] ** 2 + eps)
        lamda_prime_inv = 1.0 / torch.sqrt(lamda[:, 0] ** 2 + eps)
        H = Q @ torch.diag(lamda_prime) @ Q.T
        Hinv = Q @ torch.diag(lamda_prime_inv) @ Q.T
        log_det_Hinv = torch.sum(torch.log(lamda_prime_inv))
        return log_pi, g, H, Hinv, log_det_Hinv
    else:
        return log_pi


def V(X):
    P_pred = 10 ** model(X, add_mean=True)

    r = P_pred - P_obs
    X_bar = (X - X_min) / (X_max - X_min)

    L1 = -0.5 * r @ Tau @ r
    L2 = torch.sum((alpha_b - 1) * torch.log(X_bar) + (beta_b - 1) * torch.log(1 - X_bar))

    return -(L1 + L2)


grid_res = 4000
nc0 = NC("calibration_samples/ltop_calibration_sample_0.nc")
m_gridpoints = len(nc0.variables["precipitation"][:].ravel())
_, my, mx = nc0.variables["precipitation"][:].shape
nc0.close()

ensemble_file = "ltop_calibration_samples_30.csv"
samples = pd.read_csv(ensemble_file)
X = samples.values[:, 1::]
m_samples, n_parameters = X.shape

P = np.zeros((m_samples, my, mx))
print("Reading calibration samples")
for idx in tqdm(range(m_samples)):
    m_file = f"calibration_samples/ltop_calibration_sample_{idx}.nc"
    nc = NC(m_file)
    p = nc.variables["precipitation"][0, :, :]
    P[idx, :] = p
    nc.close()

P[P == 0] = 1e-5
Precip = P[:, ::4, ::4].reshape(m_samples, -1)
m_gridpoints_prunded = Precip.shape[1]
point_area = np.ones(m_gridpoints_prunded) * grid_res ** 2

X = torch.from_numpy(X)
F_lin = torch.from_numpy(Precip)
F = torch.log10(F_lin)
point_area = torch.from_numpy(point_area)

X = X.to(torch.float32)
F = F.to(torch.float32)
point_area = point_area.to(torch.float32)

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

n_models = 3  # To reproduce the paper, this should be 50

# for model_index in range(n_models):
#     print(f"Running model {model_index+1}/{n_models}")
#     omegas = torch.tensor(dirichlet.rvs(np.ones(m_samples)), dtype=torch.float, device=device).T

#     V_hat, F_bar, F_mean = get_eigenvectors(omegas, F)
#     n_eigenvectors = V_hat.shape[1]

#     e = Emulator(n_parameters, n_eigenvectors, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, V_hat, F_mean)
#     e.to(device)

#     train_surrogate(e, X_hat, F_bar, omegas, normed_area)
#     torch.save(e.state_dict(), f"emulator_ensemble/emulator_{model_index}.h5")


nc = NC("../climate/MIROC-ESM-CHEM_Peninsula_4000m_clim_1995-2014.nc")
P_rcm = nc.variables["precipitation"][:]
P_obs = P_rcm[::4, ::4].ravel()
nc.close()

P_obs = torch.from_numpy(P_obs)
P_obs = P_obs.to(device)

models = []

for model_index in range(n_models):
    state_dict = torch.load(f"emulator_ensemble/emulator_{model_index}.h5")
    e = Emulator(
        state_dict["l_1.weight"].shape[1],
        state_dict["V_hat"].shape[1],
        n_hidden_1,
        n_hidden_2,
        n_hidden_3,
        n_hidden_4,
        state_dict["V_hat"],
        state_dict["F_mean"],
    )
    e.load_state_dict(state_dict)
    e.to(device)
    e.eval()
    models.append(e)

torch.manual_seed(0)
np.random.seed(0)


sigma2 = 1 ** 2

Sigma_obs = sigma2 * torch.eye(P_obs.shape[0], device=device)
Sigma = Sigma_obs

rho = 1.0 / (grid_res ** 2)
K = torch.diag(point_area * rho)
Tau = K @ torch.inverse(Sigma) @ K

from scipy.stats import beta

alpha_b = 3.0
beta_b = 3.0

X_min = X_hat.cpu().numpy().min(axis=0) - 1e-3
X_max = X_hat.cpu().numpy().max(axis=0) + 1e-3

X_prior = beta.rvs(alpha_b, beta_b, size=(10000, n_parameters)) * (X_max - X_min) + X_min

X_min = torch.tensor(X_min, dtype=torch.float32, device=device)
X_max = torch.tensor(X_max, dtype=torch.float32, device=device)

for j, model in enumerate(models):
    X = torch.tensor(
        X_prior[np.random.randint(X_prior.shape[0], size=5)].mean(axis=0),
        requires_grad=True,
        dtype=torch.float,
        device=device,
    )
    X = find_MAP(X)
    X_posterior = MALA(X, n_iters=10000, model_index=j, save_interval=1000, print_interval=100)
