import numpy as np

import torch
import gpytorch
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO

from models.Nonlinear_MT_GP import NonlinearMultiTaskGP
from models.GP_model import create_dspp_config

from gpytorch.mlls import DeepPredictiveLogLikelihood

class NonlinearMultiTaskGSPP(NonlinearMultiTaskGP):

    expect_using_first = False
    is_external_likelihood = False

    def __init__(self, list_train_data, list_config, using_first):
        super().__init__(list_train_data, list_config, using_first)
        self.name = "non_linear_multi_task_gspp"
        self.loss_class = lambda likelihood, model, num_data : DeepPredictiveLogLikelihood(
            likelihood, model, num_data, beta=0.5
        )
        self.create_funct = create_dspp_config
    