import torch
from gpytorch.models.deep_gps.dspp import DSPPLayer

from gpytorch import settings
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.lazy import BlockDiagLazyTensor

from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.models import ApproximateGP

class CustomDSPPLayer(DSPPLayer):
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



