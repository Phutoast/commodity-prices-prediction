from scipy.io import loadmat
from math import floor

import torch
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL

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

class DeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(DeepGPHiddenLayer, self).__init__(
            variational_strategy, input_dims, output_dims
        )

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            num_sample = gpytorch.settings.num_likelihood_samples.value()
            expect = gpytorch.distributions.MultitaskMultivariateNormal

            if isinstance(x, expect):
                x = x.rsample()

            processed_inputs = [
                inp.rsample() if isinstance(inp, expect) else inp.unsqueeze(0).expand(num_sample, *inp.shape) 
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)
        
        return super().__call__(x, are_samples=bool(len(other_inputs)))



num_output_dims = 2

class DeepGP(DeepGP):
    def __init__(self, train_x_shape):
        hidden_layer = DeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_output_dims,
            mean_type="linear"
        )

        second_layer = DeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_output_dims,
            mean_type="linear"
        )

        last_layer = DeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims*2,
            output_dims=None,
            mean_type="constant"
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.second_layer = second_layer
        self.likelihood = GaussianLikelihood()
    
    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        hidden_rep2 = self.second_layer(inputs) 
        output = self.last_layer(hidden_rep1, hidden_rep2)
        return output
    
    def predict(self, test_loader):
        with torch.no_grad():
            mus, variances, lls = [], [], []
            for i, (x_batch, y_batch) in enumerate(test_loader):
                print(f"Running Pred: {i}/{len(test_loader)}")
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(self.likelihood.log_marginal(y_batch, self(x_batch)))
        
        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)

model = DeepGP(train_x.shape)
num_epochs = 1
num_samples = 3 

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.05)
# 0.05 for skip connection, 0.01 for normall

mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_x.shape[-2]))

for i in range(num_epochs):
    for j, (x_batch, y_batch) in enumerate(train_loader):
        with gpytorch.settings.num_likelihood_samples(num_samples):
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
        
        print(f"At Batch {j} of Epoch {i} Loss {loss}")

model.eval()
predictive_means, predictive_variances, test_lls = model.predict(test_loader)

rmse = torch.mean(torch.pow(predictive_means.mean(0) - test_y, 2)).sqrt()
print(f"RMSE: {rmse.item()}, NLL: {-test_lls.mean().item()}")
