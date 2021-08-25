import torch
import gpytorch
from models.Deep_GP import DeepGPMultiOut

from models.deep_layers import DSPPHHiddenLayer
from gpytorch.mlls import DeepPredictiveLogLikelihood
from gpytorch.models.deep_gps.dspp import DSPP

from models.GP_model import create_deep_GP
from torch.utils.data import TensorDataset, DataLoader

def dspp_ll(model, x_batch, y_batch):
    base_batch_ll = model.likelihood.log_marginal(y_batch, model(x_batch))
    deep_batch_ll = model.quad_weights.unsqueeze(-1) + base_batch_ll
    batch_log_prob = deep_batch_ll.logsumexp(dim=0)
    return batch_log_prob.cpu()

def create_dspp_config(num_inducing, num_quad_site):
    return {
        "class_type": DSPP,
        "hidden_layer_type": DSPPHHiddenLayer,
        "hidden_info": {"num_inducing": num_inducing, "num_quad_sites": num_quad_site},
        "class_init_info": {"num_quad_sites": num_quad_site},
        "log_likelihood_cal": dspp_ll,
    }

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