import numpy as np

import torch
import gpytorch
from gpytorch.mlls import DeepPredictiveLogLikelihood

from models.DeepGP_graph_prop import DeepGPGraphPropagate
from models.GP_model import create_dspp_config

from torch.utils.data import TensorDataset, DataLoader

class DSPPGraphPropagate(DeepGPGraphPropagate):
    
    expect_using_first = False
    is_external_likelihood = False

    def __init__(self, list_train_data, list_config, using_first):
        super().__init__(list_train_data, list_config, using_first)
        self.name = "non_linear_graph_prop_deep_gp"
        self.loss_class = lambda likelihood, model, num_data : DeepPredictiveLogLikelihood(
            likelihood, model, num_data, beta=0.5
        )
        self.create_funct = create_dspp_config 
        self.underly_graph = torch.from_numpy(np.load(
            self.hyperparam["graph_path"]
        ))
        