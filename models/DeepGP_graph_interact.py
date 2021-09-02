import numpy as np

import torch
import gpytorch
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO

from models.Nonlinear_MT_GP import NonlinearMultiTaskGP
from models.GP_model import create_non_linear_interact, create_deep_gp_config

class DeepGPGraphInteract(NonlinearMultiTaskGP):
    
    expect_using_first = False
    is_external_likelihood = False

    def __init__(self, list_train_data, list_config, using_first):
        super().__init__(list_train_data, list_config, using_first)
        self.name = "non_linear_graph_interact_deep_gp"
        self.loss_class = lambda likelihood, model, num_data : DeepApproximateMLL(
            VariationalELBO(likelihood, model, num_data)
        )
        self.create_funct = create_deep_gp_config
        
    
    def build_training_model(self):
        self.model = create_non_linear_interact(
            self.create_funct, self.feat_size, 
            self.num_task, self.hyperparam, 
            num_inducing=self.train_x.size(0)//(3*self.num_task),
            hidden_layer_size=8
        )

        return self.model
    
    def build_model_from_loaded(self, all_data, list_config, num_task):
        (state_dict, self.train_x, self.train_y, 
            self.mean_x, self.std_x, self.train_ind) = all_data
        
        self.feat_size = self.train_x.size(-1)
        
        self.model = create_non_linear_interact(
            self.create_funct, self.feat_size, 
            self.num_task, self.hyperparam, 
            num_inducing=self.train_x.size(0)//(3*self.num_task),
            hidden_layer_size=8
        )