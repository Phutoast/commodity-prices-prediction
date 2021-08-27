import torch
import math
import matplotlib.pyplot as plt

import gpytorch

train_x = torch.linspace(0, 1, 100)

def generate_data(x):
    g_out = -torch.sin(8*math.pi*(x+1))/(2*x+1) - x**4
    h1 = torch.sin(3*x)
    h2 = 3*x
    f1 = torch.cos(g_out)**2 + h1
    f2 = torch.sin(10*x)*g_out**2 + h2
    return torch.stack([f1, f2], -1)

train_y = generate_data(train_x)

num_show_points = 50
show_points = torch.randperm(100)[:num_show_points]
print(show_points)

f1, f2 = train_y.T
x, f1, f2 = train_x.numpy(), f1.numpy(), f2.numpy()

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
model = MultitaskGPModel(train_x, train_y, likelihood)

model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.1)

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(50):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, 50, loss.item()))
    optimizer.step()
    break

model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 1, 51)
    predictions = likelihood(model(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

fig, axes = plt.subplots(nrows=2)
axes[0].plot(x, f1)
axes[0].plot(test_x.numpy(), mean[:, 0].numpy(), c='r')
axes[0].fill_between(test_x.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.2, color='r')
axes[0].scatter(x[show_points], f1[show_points], marker="x", s=40, c='k')
axes[0].grid()

axes[1].plot(x, f2)
axes[1].plot(test_x.numpy(), mean[:, 1].numpy(), c='r')
axes[1].fill_between(test_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.2, color='r')
axes[1].scatter(x[show_points], f2[show_points], marker="x", s=40, c='k')
axes[1].grid()

plt.show()

