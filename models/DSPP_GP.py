import torch
import gpytorch
from models.Deep_GP import DeepGPMultiOut

from models.GP_model import create_deep_GP, create_dspp_config
from torch.utils.data import TensorDataset, DataLoader

from gpytorch.mlls import DeepPredictiveLogLikelihood

class DSPPMultiOut(DeepGPMultiOut):
    
    expect_using_first = True
    is_external_likelihood = False

    def __init__(self, list_train_data, list_config, using_first):
        super().__init__(list_train_data, list_config, using_first)
        self.name = "deep_sigma_gp"
        self.loss_class = lambda likelihood, model, num_data : DeepPredictiveLogLikelihood(
            likelihood, model, num_data, beta=0.5
        )
        self.create_funct = create_dspp_config 