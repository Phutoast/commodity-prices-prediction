import torch

import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps.dspp import DSPPLayer
from gpytorch import settings
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.models import ApproximateGP
from gpytorch.lazy import BlockDiagLazyTensor

from utils.utils import prepare_module

class CustomDSPPLayer(DSPPLayer):
    # Copy the main source code and change abit
    def __call__(self, inputs, are_samples=False, **kwargs):
        deterministic_inputs = not are_samples
        if isinstance(inputs, MultitaskMultivariateNormal):
            # This is for subsequent layers. We apply quadrature here
            # Mean, stdv are q x ... x n x t
            mus, sigmas = inputs.mean, inputs.variance.sqrt()
            qg = self.quad_sites.view([self.num_quad_sites] + [1] * (mus.dim() - 2) + [self.input_dims])
            sigmas = sigmas * qg
            inputs = mus + sigmas  # q^t x n x t
            deterministic_inputs = False

        if settings.debug.on():
            if not torch.is_tensor(inputs):
                raise ValueError(
                    "`inputs` should either be a MultitaskMultivariateNormal or a Tensor, got "
                    f"{inputs.__class__.__Name__}"
                )

            if inputs.size(-1) != self.input_dims:
                raise RuntimeError(
                    f"Input shape did not match self.input_dims. Got total feature dims [{inputs.size(-1)}],"
                    f" expected [{self.input_dims}]"
                )

        # Repeat the input for all possible outputs
        if self.output_dims is not None:
            inputs = inputs.unsqueeze(-3)
            inputs = inputs.expand(*inputs.shape[:-3], self.output_dims, *inputs.shape[-2:])

        # Now run samples through the GP
        output = ApproximateGP.__call__(self, inputs, **kwargs)

        # If this is the first layer (deterministic inputs), expand the output
        # This allows quadrature to be applied to future layers
        if deterministic_inputs:
            output = output.expand(torch.Size([self.num_quad_sites]) + output.batch_shape)

        if self.num_quad_sites > 0:
            if self.output_dims is not None and not isinstance(output, MultitaskMultivariateNormal):
                mean = output.loc.transpose(-1, -2)
                covar = BlockDiagLazyTensor(output.lazy_covariance_matrix, block_dim=-3)
                output = MultitaskMultivariateNormal(mean, covar, interleaved=False)
        else:
            output = output.loc.transpose(-1, -2)  # this layer provides noiseless kernel interpolation

        return output

class DSPPHHiddenLayer(CustomDSPPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, is_linear=False, num_quad_sites=8):
        
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

        super(DSPPHHiddenLayer, self).__init__(
            variational_strategy, input_dims, output_dims, num_quad_sites
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

        return super().__call__(x, are_samples=bool(len(other_inputs)), **kwargs)