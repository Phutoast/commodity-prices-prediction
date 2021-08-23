from scipy.io import loadmat
from math import floor

import gpytorch
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import MeanFieldVariationalDistribution, CholeskyVariationalDistribution
from gpytorch.models.deep_gps.dspp import DSPPLayer, DSPP
from gpytorch.mlls import DeepPredictiveLogLikelihood
import gpytorch.settings as settings

from utils.custom_dspp import CustomDSPPLayer

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

batch_size = 1024
num_inducing_pts = 300 
hidden_dim = 2
num_quadrature_sites = 8


class DSPPHHiddenLayer(CustomDSPPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=300, mean_type='constant', Q=8):
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

        super(DSPPHHiddenLayer, self).__init__(
            variational_strategy, input_dims, output_dims, Q
        )

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)

        self.covar_module = ScaleKernel(
            MaternKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def integrate(self, inputs, quad_sites):
        # Following the Source Code
        expect_type = gpytorch.distributions.MultitaskMultivariateNormal
        assert isinstance(inputs, expect_type)

        mus, sigmas = inputs.mean, inputs.variance.sqrt()
        qg = quad_sites.view([self.num_quad_sites] + [1] * (mus.dim() - 2) + [mus.size(-1)])
        sigmas = sigmas * qg
        return mus + sigmas 
    
    def __call__(self, x, *other_inputs, **kwargs):
            
        expect_type = gpytorch.distributions.MultitaskMultivariateNormal

        if len(other_inputs):

            each_sizes = [
                inp.mean.size(-1) if isinstance(inp, expect_type) else inp.size(-1)
                for inp in [x] + list(other_inputs)
            ]

            each_quad_sites = torch.split(self.quad_sites, each_sizes, dim=-1)

            if isinstance(x, expect_type):
                x = self.integrate(x, each_quad_sites[0])
            
            processed_inputs = [
                self.integrate(inp, each_quad_sites[i+1]) if isinstance(inp, expect_type) else inp.unsqueeze(0).expand(self.num_quad_sites, *inp.shape) 
                for i, inp in enumerate(other_inputs)
            ]
            
            x = torch.cat([x] + processed_inputs, dim=-1)

        # is_sample = isinstance(x, expect_type)
            
        # if isinstance(x, expect_type):
        #     x = self.integrate(x, self.quad_sites)

        return super().__call__(x, are_samples=bool(len(other_inputs)), **kwargs)

class TwoLayerDSPP(DSPP):
    def __init__(self, train_x_shape, Q=3):
        hidden_layer = DSPPHHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=hidden_dim,
            mean_type="linear",
            num_inducing=300,
            Q=Q
        )
        
        second_layer = DSPPHHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=3,
            mean_type="linear",
            num_inducing=300,
            Q=Q
        )

        last_layer = DSPPHHiddenLayer(
            input_dims=hidden_layer.output_dims + 3,
            output_dims=None,
            mean_type="constant",
            num_inducing=300,
            Q=Q
        )

        likelihood = GaussianLikelihood()

        super().__init__(Q)
        self.likelihood = likelihood
        self.last_layer = last_layer
        self.hidden_layer = hidden_layer
        self.second_layer = second_layer
    
    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        hidden_rep2 = self.second_layer(inputs) 
        output = self.last_layer(hidden_rep1, hidden_rep2)
        # output = self.last_layer(hidden_rep1)
        return output
    
    def predict(self, loader):
        with settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            mus, variances, lls = [], [], []
            for x_batch, y_batch in loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean.cpu())
                variances.append(preds.variance.cpu())

                base_batch_ll = self.likelihood.log_marginal(y_batch, self(x_batch))
                deep_batch_ll = self.quad_weights.unsqueeze(-1) + base_batch_ll
                batch_log_prob = deep_batch_ll.logsumexp(dim=0)
                lls.append(batch_log_prob.cpu())

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)

with gpytorch.settings.debug():
    model = TwoLayerDSPP(
        train_x.shape,
        Q=69
    )

    model.train()

    adam = torch.optim.Adam(
        [{'params': model.parameters()}], 
        lr=0.05, betas=(0.9, 0.999)
    )
    objective = DeepPredictiveLogLikelihood(
        model.likelihood, model, 
        num_data=train_n, beta=0.05
    )

    for i in range(1):
        for j, (x_batch, y_batch) in enumerate(train_loader):
            adam.zero_grad()
            output = model(x_batch)
            loss = -objective(output, y_batch)
            loss.backward()
            adam.step()
            print(f"At Batch {j} of Epoch {i} Loss {loss}")
        
model.eval()
means, vars, ll = model.predict(test_loader)
weights = model.quad_weights.unsqueeze(-1).exp().cpu()
rmse = ((weights * means).sum(0) - test_y.cpu()).pow(2.0).mean().sqrt().item()
ll = ll.mean().item()

print('RMSE: ', rmse, 'Test NLL: ', -ll)
