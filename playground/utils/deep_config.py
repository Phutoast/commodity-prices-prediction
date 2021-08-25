from utils.dspp_hidden import DSPPHHiddenLayer
from gpytorch.models.deep_gps.dspp import DSPP
from gpytorch.mlls import DeepPredictiveLogLikelihood

from gpytorch.models.deep_gps import DeepGP
from utils.deep_gp_hidden import DeepGPHiddenLayer
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO

def deep_gp_ll(model, x_batch, y_batch):
    return model.likelihood.log_marginal(y_batch, model(x_batch)).cpu()

def dspp_ll(model, x_batch, y_batch):
    base_batch_ll = model.likelihood.log_marginal(y_batch, model(x_batch))
    deep_batch_ll = model.quad_weights.unsqueeze(-1) + base_batch_ll
    batch_log_prob = deep_batch_ll.logsumexp(dim=0)
    return batch_log_prob.cpu()

def create_deep_gp_config(num_inducing):
    return {
        "class_type": DeepGP,
        "hidden_layer_type": DeepGPHiddenLayer,
        "objective": lambda likelihood, model, num_data : DeepApproximateMLL(
            VariationalELBO(likelihood, model, num_data)
        ),
        "hidden_info": {"num_inducing": num_inducing},
        "class_init_info": {},
        "log_likelihood_cal": deep_gp_ll,
    }

def create_dspp_config(num_inducing, num_quad_site):
    return {
        "class_type": DSPP,
        "hidden_layer_type": DSPPHHiddenLayer,
        "objective": lambda likelihood, model, num_data : DeepPredictiveLogLikelihood(
            likelihood, model, num_data, beta=0.5
        ),
        "hidden_info": {"num_inducing": num_inducing, "num_quad_sites": num_quad_site},
        "class_init_info": {"num_quad_sites": num_quad_site},
        "log_likelihood_cal": dspp_ll,
    }
