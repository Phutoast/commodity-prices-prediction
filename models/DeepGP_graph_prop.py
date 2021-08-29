import numpy as np

import torch
import gpytorch
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO

from models.Nonlinear_MT_GP import NonlinearMultiTaskGP
from models.GP_model import create_non_linear_graph_gp, create_deep_gp_config

from torch.utils.data import TensorDataset, DataLoader

class DeepGPGraphPropagate(NonlinearMultiTaskGP):
    
    expect_using_first = False
    is_external_likelihood = False

    def __init__(self, list_train_data, list_config, using_first):
        super().__init__(list_train_data, list_config, using_first)
        self.name = "non_linear_graph_prop_deep_gp"
        self.loss_class = lambda likelihood, model, num_data : DeepApproximateMLL(
            VariationalELBO(likelihood, model, num_data)
        )
        self.create_funct = create_deep_gp_config
        self.underly_graph = torch.from_numpy(np.load(
            self.hyperparam["graph_path"]
        ))
        
    
    def build_training_model(self):
        self.model = create_non_linear_graph_gp(
            self.create_funct, self.feat_size, 
            self.num_task, self.hyperparam, 
            graph_structure=self.underly_graph,
            num_inducing=self.train_x.size(0)//(3*self.num_task),
            hidden_layer_size=8
        )

        return self.model