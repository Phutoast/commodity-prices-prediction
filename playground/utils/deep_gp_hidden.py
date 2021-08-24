import torch

import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer

from utils.utils import prepare_module

class DeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, is_linear=False):
        
        indc_points, batch_shape, var_dist = prepare_module(
            input_dims, output_dims, num_inducing, 
            CholeskyVariationalDistribution
        )
        
        variational_strategy = VariationalStrategy(
            self,
            indc_points,
            var_dist,
            learn_inducing_locations=True
        )

        super(DeepGPHiddenLayer, self).__init__(
            variational_strategy, input_dims, output_dims
        )

        if not is_linear:
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
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
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