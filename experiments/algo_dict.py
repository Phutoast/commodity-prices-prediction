from utils.data_structure import Hyperparameters
from models.ARIMA import ARIMAModel
from models.Mean import IIDDataModel
from models.GP import IndependentGP

from gpytorch import kernels
import torch

algorithms_dic = {
    # The ind span pred should be the same as len_out
    "ARIMA": [Hyperparameters(
        len_inp=0, 
        len_out=10, 
        is_date=False, 
        order=(10, 2, 5), 
    ), ARIMAModel],
    "Mean": [Hyperparameters(
        len_inp=0, 
        len_out=10, 
        is_date=False, 
        dist="Gaussian",
        is_verbose=False
    ), IIDDataModel],
    "Mean-Test": [Hyperparameters(
        len_inp=3, 
        len_out=2, 
        is_date=False, 
        dist="Gaussian",
        is_verbose=False
    ), IIDDataModel],
    "GP": [Hyperparameters(
        len_inp=10, 
        len_out=1, 
        lr=0.1,
        optim_iter=250,
        jitter=1e-4,
        is_time_only=False,
        is_date=True, 
        is_batch=False,
        kernel=kernels.ScaleKernel(kernels.RBFKernel())
    ), IndependentGP],
    "GP-Test": [Hyperparameters(
        len_inp=10, 
        len_out=1, 
        lr=0.1,
        optim_iter=100,
        jitter=1e-4,
        is_time_only=False,
        is_date=True, 
        is_batch=False,
        kernel=kernels.ScaleKernel(kernels.MaternKernel())
    ), IndependentGP],
}

# Possible Kernels for Normal GP
kernels.ScaleKernel(kernels.RBFKernel()) + kernels.ScaleKernel(kernels.PeriodicKernel(power=2))

# Possible Kernels for Batch

kernels.ScaleKernel(
    kernels.ScaleKernel(kernels.CosineKernel(batch_shape=torch.Size([1])), batch_shape=torch.Size([1]))+ 
    kernels.ScaleKernel(kernels.MaternKernel(batch_shape=torch.Size([1])), batch_shape=torch.Size([1])), batch_shape=torch.Size([1])
)
kernels.ScaleKernel(
    kernels.RBFKernel(batch_shape=torch.Size([2])),
    batch_shape=torch.Size([2])
)