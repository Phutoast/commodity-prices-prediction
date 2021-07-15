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
        optim_iter=100,
        is_time_only=False,
        is_date=False, 
        is_past_label=False,
        kernel="Composite_2"
    ), IndependentGP],
    "GP-Test": [Hyperparameters(
        len_inp=10, 
        len_out=1, 
        lr=0.1,
        optim_iter=100,
        is_time_only=False,
        is_date=False, 
        is_past_label=True,
        kernel="Composite_1"
    ), IndependentGP],
    "GP-Multi-Task": [Hyperparameters(
        len_inp=10, 
        len_out=1, 
        lr=0.1,
        optim_iter=100,
        is_time_only=False,
        is_date=False, 
        is_past_label=True,
        kernel="Composite_1"
    ), None],
    "GP-Special-GPU": [Hyperparameters(
        len_inp=10, 
        len_out=1, 
        lr=0.1,
        optim_iter=100,
        is_time_only=False,
        is_date=False, 
        is_past_label=True,
        kernel="Composite_1",
        is_gpu=True
    ), IndependentGP],
}

class_name = {
    "ARIMAModel": ARIMAModel,
    "IIDDataModel": IIDDataModel,
    "IndependentGP": IndependentGP,
}

kernel_name = {
    "RBF_Scale": kernels.ScaleKernel(kernels.RBFKernel()),
    "Matern_Scale": kernels.ScaleKernel(kernels.MaternKernel()),
    "Composite_1": kernels.ScaleKernel(kernels.RBFKernel()) + kernels.ScaleKernel(kernels.PeriodicKernel(power=2)),
    "Composite_2": kernels.ScaleKernel(kernels.RBFKernel()) + kernels.ScaleKernel(kernels.PolynomialKernel(power=2)),
    "Batch_1": lambda num_task: kernels.ScaleKernel(
        kernels.ScaleKernel(kernels.CosineKernel(batch_shape=torch.Size([num_task])), batch_shape=torch.Size([num_task]))+ 
        kernels.ScaleKernel(kernels.MaternKernel(batch_shape=torch.Size([num_task])), batch_shape=torch.Size([num_task])), batch_shape=torch.Size([num_task])
    ),
    "Batch_2": lambda num_task: kernels.ScaleKernel(
        kernels.RBFKernel(batch_shape=torch.Size([num_task])),
        batch_shape=torch.Size([num_task])
    )
}

