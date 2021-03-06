import numpy as np
import gpytorch
import torch

from models.base_model import BaseModel
from models.GP_model import OneDimensionGP
from models.train_model import BaseTrainModel

from experiments import algo_dict
import copy

import matplotlib.pyplot as plt
import pandas as pd
from utils import data_visualization
 
class IndependentGP(BaseTrainModel):
    """
    Simple Gaussian Process Model that takes date 
        as inp and return the price prediction.
    """
    def __init__(self, train_data, model_hyperparam):
        super().__init__(train_data, model_hyperparam)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if self.hyperparam["is_gpu"]:
            self.likelihood = self.likelihood.cuda()
    
    def prepare_data(self):
        all_prices = self.pack_data(
            self.train_data, is_label=self.hyperparam["is_past_label"]
        )

        if self.hyperparam["is_time_only"]:
            self.train_x = torch.from_numpy(all_prices[:, 0]).float()
            self.train_y = torch.from_numpy(all_prices[:, -1]).float()
        else:
            self.train_x = torch.from_numpy(all_prices[:, :-1]).float()
            self.train_y = torch.from_numpy(all_prices[:, -1]).float()
        
        if self.hyperparam["is_gpu"]:
            self.train_x = self.train_x.cuda()
            self.train_y = self.train_y.cuda()
        
        self.train_x = self.normalize_data(self.train_x, is_train=True)
        return self.train_x, self.train_y
    
    def build_training_model(self):
        kernel = self.load_kernel(self.hyperparam["kernel"])
        self.model = OneDimensionGP(
            self.train_x, self.train_y, self.likelihood, kernel
        )
        return self.model
    
    def build_optimizer_loss(self):
        self.model.train()
        self.likelihood.train()

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
        ], lr=self.hyperparam["lr"])

        self.loss_obj = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model
        )

        return self.optimizer, self.loss_obj
    
    def after_training(self):
        self.model.eval()
        self.likelihood.eval()
    
    def predict_step_ahead(self, test_data, step_ahead, all_date, ci=0.9, is_sample=False):
        """
        Args: (See superclass)
        Returns: (See superclass)
        """ 
        self.model.eval()
        self.likelihood.eval()

        inp_test = self.pack_data(
            test_data, is_label=self.hyperparam["is_past_label"]
        )

        if self.hyperparam["is_time_only"]:
            inp_test = inp_test[:, 0]
        else:
            inp_test = inp_test[:, :-1]

        size_test_data = inp_test.shape[0]
        assert step_ahead <= size_test_data
        
        date_size = len(all_date)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.from_numpy(inp_test).float()
            if self.hyperparam["is_gpu"]:
                test_x = test_x.cuda()
            test_x = self.normalize_data(test_x, is_train=False)

            if not is_sample:
                pred = self.likelihood(self.model(test_x.float()))
                pred_mean = pred.mean.detach().cpu().numpy().tolist()
                lower, upper = pred.confidence_region()
                pred_lower = lower.detach().cpu().numpy().tolist()
                pred_upper = upper.detach().cpu().numpy().tolist()
                return pred_mean, pred_lower, pred_upper, all_date[date_size-step_ahead:]
            else:
                rv = self.model(test_x.float())
                rv = rv.sample(sample_shape=torch.Size([1000])).cpu().numpy()
                return rv, all_date[date_size-step_ahead:]
 
 
    def save(self, path):
        torch.save(self.model.state_dict(), path + ".pth")
        torch.save(self.train_x, path + "_x.pt")
        torch.save(self.train_y, path + "_y.pt")
        torch.save(self.mean_x, path + "_mean_x.pt")
        torch.save(self.std_x, path + "_std_x.pt")
    
    def load(self, path):
        
        all_ext = [".pth", "_x.pt", "_y.pt", "_mean_x.pt", "_std_x.pt"]
        all_data = []
        if self.hyperparam["is_gpu"]:
            all_data = [torch.load(path + ext, map_location="cuda:0") for ext in all_ext]
        else:
            all_data = [torch.load(path + ext, map_location=torch.device('cpu')) for ext in all_ext]
        
        (state_dict, self.train_x, self.train_y, 
            self.mean_x, self.std_x) = all_data

        self.model = OneDimensionGP(
            self.train_x, self.train_y, 
            self.likelihood, 
            self.load_kernel(self.hyperparam["kernel"])
        )

        if self.hyperparam["is_gpu"]:
            self.model.cuda()

        self.model.load_state_dict(state_dict)