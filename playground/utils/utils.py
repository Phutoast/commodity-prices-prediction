import torch

def prepare_module(input_dims, output_dims, num_inducing, dist_type):
    if output_dims is None:
        indc_points = torch.randn(num_inducing, input_dims)
        batch_shape = torch.Size([])
    else:
        indc_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])

    var_dist = dist_type(
        num_inducing_points=num_inducing,
        batch_shape=batch_shape
    )

    return indc_points, batch_shape, var_dist




