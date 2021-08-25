import math
import torch
import gpytorch
from matplotlib import pyplot as plt

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

train_x1 = torch.rand(50)
train_x2 = torch.rand(50)

train_y1 = torch.sin(train_x1 * (2 * math.pi)) + torch.randn(train_x1.size()) * 0.2
train_y2 = torch.cos(train_x2 * (2 * math.pi)) + torch.randn(train_x2.size()) * 0.2

train_i_task1 = torch.full((train_x1.shape[0],1), dtype=torch.long, fill_value=0)
train_i_task2 = torch.full((train_x2.shape[0],1), dtype=torch.long, fill_value=1)

train_x = torch.cat([train_x1, train_x2])
train_i = torch.cat([train_i_task1, train_i_task2])
train_y = torch.cat([train_y1, train_y2])

class MultitaskSparseGPModel(ApproximateGP):
    def __init__(self, ind_pts):
        
        var_dist = CholeskyVariationalDistribution(ind_pts.size(0))
        var_strategy = VariationalStrategy(
            self, ind_pts, var_dist, 
            learn_inducing_locations=True
        )
        super(MultitaskSparseGPModel, self).__init__(var_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)

    def forward(self, x, all_ind):
        print(x)
        assert False
        mean_x = self.mean_module(x)

        covar_x = self.covar_module(x)
        covar_i = self.task_covar_module(all_ind)
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)


ind_pts = train_x[:20]
ind_pts = torch.zeros(20)
ind_index = torch.cat([torch.ones(10, 1)*0, torch.ones(10, 1)*1], axis=0)

model = MultitaskSparseGPModel(ind_pts=ind_pts)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.1)

mll = gpytorch.mlls.VariationalELBO(
    likelihood, model, 
    num_data=train_y.size(0)
)


for i in range(150):
    optimizer.zero_grad()
    output = model(train_x, all_ind=torch.cat([ind_index, train_i], axis=0))
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(i, loss)

model.eval()
likelihood.eval()

f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))

test_x = torch.linspace(0, 1, 51)
test_i_task1 = torch.full((test_x.shape[0],1), dtype=torch.long, fill_value=0)
test_i_task2 = torch.full((test_x.shape[0],1), dtype=torch.long, fill_value=1)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred_y1 = likelihood(model(test_x, all_ind=torch.cat([ind_index, test_i_task1], axis=0)))
    observed_pred_y2 = likelihood(model(test_x, all_ind=torch.cat([ind_index, test_i_task2], axis=0)))

# Define plotting function
def ax_plot(ax, train_y, train_x, rand_var, title):
    # Get lower and upper confidence bounds
    lower, upper = rand_var.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.detach().numpy(), train_y.detach().numpy(), 'k*')
    # Predictive mean as blue line
    ax.plot(test_x.detach().numpy(), rand_var.mean.detach().numpy(), 'b')
    print(rand_var.mean.detach().numpy().shape)
    # Shade in confidence
    ax.fill_between(test_x.detach().numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    ax.set_title(title)

# Plot both tasks
ax_plot(y1_ax, train_y1, train_x1, observed_pred_y1, 'Observed Values (Likelihood)')
ax_plot(y2_ax, train_y2, train_x2, observed_pred_y2, 'Observed Values (Likelihood)')
plt.show()