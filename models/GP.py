import numpy as np
import gpytorch
import torch
from models.base_model import BaseModel
from gpytorch.constraints import Positive

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=6)
        # self.covar_module.initialize_from_data(train_x, train_y)
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel() + gpytorch.kernels.PeriodicKernel())
            
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class SimpleGaussianProcessModel(BaseModel):
    """
    Simple Gaussian Process Model that takes date 
        as inp and return the price prediction.
    """
    def __init__(self, train_data, model_hyperparam):
        super().__init__(train_data, model_hyperparam)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    def train(self):
        all_prices = self.pack_data(self.train_data)
        train_x = torch.from_numpy(all_prices[:, 0]).float()
        train_y = torch.from_numpy(all_prices[:, 1]).float()
        
        self.mean_x = torch.mean(train_x)
        self.std_x = torch.std(train_x)

        train_x = (train_x - self.mean_x)/self.std_x
        self.model = ExactGPModel(train_x, train_y, self.likelihood)

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(
            [{'params': self.model.parameters()}],
            lr=self.hyperparam["lr"]
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model
        )

        num_iter = self.hyperparam["optim_iter"]
        for i in range(num_iter):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            print(f"Loss {i}/{num_iter}", loss)
            loss.backward()
            optimizer.step()

        self.model.eval()
        self.likelihood.eval()
    
    def predict_step_head(self, test_data, step_ahead, ci=0.9):
        """
        Args: (See superclass)
        Returns: (See superclass)
        """

        inp_test = test_data.data_out
        size_test_data = len(inp_test)
        assert step_ahead <= size_test_data
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.from_numpy(inp_test["Date"].to_numpy("float64"))
            test_x = (test_x - self.mean_x)/self.std_x
            observed_pred = self.likelihood(self.model(test_x.float()))

            pred_mean = observed_pred.mean.numpy().tolist()
            lower, upper = observed_pred.confidence_region()
            pred_lower = lower.numpy().tolist()
            pred_upper = upper.numpy().tolist()
        
        return pred_mean, pred_lower, pred_upper, inp_test["Date"].to_list()
    
class FeatureGP(BaseModel):
    """
    Simple Gaussian Process Model that takes date 
        as inp and return the price prediction.
    """
    def __init__(self, train_data, model_hyperparam):
        super().__init__(train_data, model_hyperparam)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    def train(self):
        with gpytorch.settings.cholesky_jitter(self.hyperparam["jitter"]):
            all_prices = self.pack_data(self.train_data)

            train_x = torch.from_numpy(all_prices[:, :-1]).float()
            train_y = torch.from_numpy(all_prices[:, -1]).float()
            
            self.mean_x = torch.mean(train_x, axis=0)
            self.std_x = torch.std(train_x, axis=0)

            train_x = (train_x - self.mean_x)/self.std_x
            self.model = ExactGPModel(train_x, train_y, self.likelihood)

            self.model.train()
            self.likelihood.train()

            optimizer = torch.optim.Adam(
                [{'params': self.model.parameters()}],
                lr=self.hyperparam["lr"]
            )
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.likelihood, self.model
            )

            num_iter = self.hyperparam["optim_iter"]
            for i in range(num_iter):
                optimizer.zero_grad()
                output = self.model(train_x)
                loss = -mll(output, train_y)
                print(f"Loss {i}/{num_iter}", loss)
                loss.backward()
                optimizer.step()

            self.model.eval()
            self.likelihood.eval()
    
    def predict_step_head(self, test_data, step_ahead, ci=0.9):
        """
        Args: (See superclass)
        Returns: (See superclass)
        """

        with gpytorch.settings.cholesky_jitter(self.hyperparam["jitter"]):
            inp_test = self.pack_data(test_data)[:, :-1]
            size_test_data = len(inp_test)
            assert step_ahead <= size_test_data
            
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_x = torch.from_numpy(inp_test).float()
                test_x = (test_x - self.mean_x)/self.std_x
                observed_pred = self.likelihood(self.model(test_x.float()))

                pred_mean = observed_pred.mean.numpy().tolist()
                lower, upper = observed_pred.confidence_region()
                pred_lower = lower.numpy().tolist()
                pred_upper = upper.numpy().tolist()
            
            return pred_mean, pred_lower, pred_upper, inp_test[:, -1]
