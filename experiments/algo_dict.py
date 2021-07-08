from utils.data_structure import Hyperparameters
from models.ARIMA import ARIMAModel
from models.Mean import IIDDataModel
from models.GP import FeatureGP

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
    "ARIMA2": [Hyperparameters(
        len_inp=0, 
        len_out=20, 
        is_date=False, 
        order=(5, 4, 5), 
    ), ARIMAModel],
    "ARMA": [Hyperparameters(
        len_inp=0, 
        len_out=20, 
        is_date=False, 
        order=(5, 0, 5), 
    ), ARIMAModel],
    "Mean": [Hyperparameters(
        len_inp=0, 
        len_out=10, 
        is_date=False, 
        dist="Gaussian",
        is_verbose=False
    ), IIDDataModel],
    "GP": [Hyperparameters(
        len_inp=10, 
        len_out=1, 
        lr=0.1,
        optim_iter=500,
        jitter=1e-4,
        is_time_only=False,
        is_date=True, 
        kernel=kernels.ScaleKernel(
            kernels.RBFKernel(batch_shape=torch.Size([1])),
            batch_shape=torch.Size([1])
        )
    ), FeatureGP],
    "GP-2": [Hyperparameters(
        len_inp=10, 
        len_out=1, 
        lr=0.1,
        optim_iter=500,
        jitter=1e-4,
        is_time_only=True,
        is_date=True, 
        kernel=kernels.ScaleKernel(kernels.MaternKernel()) + kernels.ScaleKernel(kernels.PolynomialKernel(power=2))
    ), FeatureGP],
    "GP-Multi-Out": [Hyperparameters(
        len_inp=10, 
        len_out=2, 
        lr=0.1,
        optim_iter=250,
        jitter=1e-4,
        is_time_only=False,
        is_date=True, 
        kernel=kernels.ScaleKernel(
            kernels.RBFKernel(batch_shape=torch.Size([2])),
            batch_shape=torch.Size([2])
        )
    ), FeatureGP],
}
