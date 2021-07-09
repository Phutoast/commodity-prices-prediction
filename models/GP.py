import numpy as np
import gpytorch
import torch
from models.base_model import BaseModel
from models.base_GP import OneDimensionGP, BatchGP
 
class IndependentGP(BaseModel):
    """
    Simple Gaussian Process Model that takes date 
        as inp and return the price prediction.
    """
    def __init__(self, train_data, model_hyperparam):
        super().__init__(train_data, model_hyperparam)

        if self.hyperparam["is_batch"]:
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=self.hyperparam["len_out"]
            )
        else:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    def train(self):
        with gpytorch.settings.cholesky_jitter(self.hyperparam["jitter"]):
            all_prices = self.pack_data(self.train_data)

            self.index_out = -1 * self.hyperparam["len_out"]

            if self.hyperparam["is_time_only"]:
                self.train_x = torch.from_numpy(all_prices[:, 0]).float()
                self.train_y = torch.from_numpy(all_prices[:, self.index_out:]).float()
            else:
                self.train_x = torch.from_numpy(all_prices[:, :self.index_out]).float()
                self.train_y = torch.from_numpy(all_prices[:, self.index_out:]).float()
            
            if self.hyperparam["len_out"] == 1:
                self.train_y = self.train_y.flatten()
            
            self.mean_x = torch.mean(self.train_x, axis=0)
            self.std_x = torch.std(self.train_x, axis=0)

            self.train_x = (self.train_x - self.mean_x)/self.std_x
            if self.hyperparam["is_batch"]:
                self.model = BatchGP(
                    self.train_x, self.train_y, self.likelihood, self.hyperparam["kernel"], self.hyperparam["len_out"]
                )
            else:
                self.model = OneDimensionGP(
                    self.train_x, self.train_y, self.likelihood, self.hyperparam["kernel"], self.hyperparam["len_out"]
                ) 

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
                output = self.model(self.train_x)
                loss = -mll(output, self.train_y)

                if self.hyperparam["is_verbose"]:
                    if i%10 == 0:
                        print(f"Loss {i}/{num_iter}", loss)
                loss.backward()
                optimizer.step()

            self.model.eval()
            self.likelihood.eval()
    
    def predict_step_ahead(self, test_data, step_ahead, ci=0.9):
        """
        Args: (See superclass)
        Returns: (See superclass)
        """
        
        self.model.eval()
        self.likelihood.eval()

        with gpytorch.settings.cholesky_jitter(self.hyperparam["jitter"]):
            if self.hyperparam["is_time_only"]:
                inp_test = self.pack_data(test_data)[:, 0]
            else:
                inp_test = self.pack_data(test_data)[:, :self.index_out] 
                
            size_test_data = len(inp_test)
            assert step_ahead == size_test_data * self.hyperparam["len_out"]
            
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_x = torch.from_numpy(inp_test).float()
                test_x = (test_x - self.mean_x)/self.std_x
                observed_pred = self.likelihood(self.model(test_x.float()))

                flatten_result = lambda x: x.numpy().flatten().tolist()

                pred_mean = flatten_result(observed_pred.mean)
                lower, upper = observed_pred.confidence_region()
                pred_lower = flatten_result(lower)
                pred_upper = flatten_result(upper)
            
            return pred_mean, pred_lower, pred_upper
    
    def save(self, path):
        torch.save(self.model.state_dict(), path + ".pth")
        torch.save(self.train_x, path + "_x.pt")
        torch.save(self.train_y, path + "_y.pt")
        torch.save(self.mean_x, path + "_mean_x.pt")
        torch.save(self.std_x, path + "_std_x.pt")
    
    def load(self, path):
        state_dict = torch.load(path + ".pth")
        self.train_x = torch.load(path + "_x.pt")
        self.train_y = torch.load(path + "_y.pt")
        self.mean_x = torch.load(path + "_mean_x.pt")
        self.std_x = torch.load(path + "_std_x.pt")

        self.model = OneDimensionGP(
            self.train_x, self.train_y, 
            self.likelihood, 
            self.hyperparam["kernel"]
        )

        self.model.load_state_dict(state_dict)



