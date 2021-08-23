import tqdm
from math import floor
import torch
import gpytorch

from scipy.io import loadmat
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader


def load_data():
    data = torch.Tensor(loadmat("data/elevators.mat")["data"])
    X = data[:, :-1]
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    y = data[:, -1]
    return X, y

X, y = load_data()

train_n = int(floor(0.8 * len(X)))
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()

test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()

train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class GPModel(ApproximateGP):
    def __init__(self, ind_pts):
        var_dist = CholeskyVariationalDistribution(ind_pts.size(0))
        var_strategy = VariationalStrategy(
            self, ind_pts, var_dist, 
            learn_inducing_locations=True
        )
        super(GPModel, self).__init__(var_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
ind_pts = train_x[:500, :]
model = GPModel(ind_pts=ind_pts)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.01)

mll = gpytorch.mlls.VariationalELBO(
    likelihood, model, 
    num_data=train_y.size(0)
)

num_epochs = 1

for i in range(num_epochs):

    for j, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(x_batch)
        loss = -mll(output, y_batch)
        loss.backward()
        optimizer.step()

        print(f"At Batch {j} of Epoch {i} Loss {loss}")

model.eval()
likelihood.eval()
means = torch.tensor([0.])
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        preds = model(x_batch)
        means = torch.cat([means, preds.mean.cpu()])
means = means[1:]
print('Test MAE: {}'.format(torch.mean(torch.abs(means - test_y.cpu()))))

