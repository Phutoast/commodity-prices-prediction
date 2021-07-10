import numpy as np
import pandas as pd
import json

import torch
import gpytorch
from models.train_model import BaseTrainMultiTask
from models.GP_model import MultioutputGP
from models.GP import IndependentGP

from experiments import algo_dict
from utils import others

class GPMultiTask(BaseTrainMultiTask):
    def __init__(self, list_train_data, list_config, using_first):
        super().__init__(list_train_data, list_config, using_first)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.num_task
        )
        self.list_config_json = {
            "list_config": list_config, "using_first": using_first
        }
    
    def prepare_data(self):
        all_data, self.train_y = self.pack_data_merge(
            self.train_data, self.hyperparam["is_past_label"]
        )
        self.train_y = torch.from_numpy(self.train_y).float()

        if self.hyperparam["is_time_only"]:
            self.train_x = torch.from_numpy(all_data[:, 0]).float()
        else:
            self.train_x = torch.from_numpy(all_data[:, :-1]).float()
        
        self.train_x = self.normalize_data(self.train_x, is_train=True)
        return self.train_x, self.train_y
 
    def build_training_model(self):
        kernel = algo_dict.kernel_name[self.hyperparam["kernel"]]
        self.model = MultioutputGP(
            self.train_x, self.train_y, self.likelihood, kernel, self.num_task
        )
        return self.model
    
    def build_optimizer_loss(self):
        self.model.train()
        self.likelihood.train()

        self.optimizer = torch.optim.Adam(
            [{'params': self.model.parameters()}],
            lr=self.hyperparam["lr"]
        )
        self.loss_obj = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model
        )

        return self.optimizer, self.loss_obj

    def after_training(self):
        self.model.eval()
        self.likelihood.eval()

    def predict_step_ahead(self, list_test_data, list_step_ahead, ci=0.9):
        """
        Predict multiple independent multi-model data

        Args:
            list_test_data: All testing for each models
            list_step_ahead: All number step a head 
                for each model
            list_all_date: All the date used along side of prediction
        
        Returns:
            list_prediction: List of all prediction for each models
        """
        self.model.eval()
        self.likelihood.eval()
        
        all_data, _ = self.pack_data_merge(
            list_test_data, self.hyperparam["is_past_label"]
        )
        if self.hyperparam["is_time_only"]:
            all_data = all_data[:, 0]
        else:
            all_data = all_data[:, :-1]
        assert all(step_ahead <= all_data.shape[0] for step_ahead in list_step_ahead)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.from_numpy(all_data).float()
            test_x = self.normalize_data(test_x, is_train=False)

            pred = self.likelihood(self.model(test_x.float()))
            pred_mean = pred.mean.numpy()
            lower, upper = pred.confidence_region()
            pred_lower = lower.numpy()
            pred_upper = upper.numpy()

        return pred_mean, pred_lower, pred_upper


    def save(self, base_path):
        others.create_folder(base_path)
        path = base_path + "/multi_GP/"
        others.create_folder(path)
        path += "model"

        torch.save(self.model.state_dict(), path + ".pth")
        torch.save(self.train_x, path + "_x.pt")
        torch.save(self.train_y, path + "_y.pt")
        torch.save(self.mean_x, path + "_mean_x.pt")
        torch.save(self.std_x, path + "_std_x.pt")
        
        with open(f"{base_path}/config.json", 'w', encoding="utf-8") as f:
            json.dump(
                self.list_config_json, f, ensure_ascii=False, indent=4
            )
    
    def load(self, base_path): 
        with open(f"{base_path}/config.json", 'r', encoding="utf-8") as f:
            data = json.load(f)
        
        list_config = data["list_config"]
        num_task = len(list_config)
        
        path = base_path + "/multi_GP/"
        path += "model"

        state_dict = torch.load(path + ".pth")
        self.train_x = torch.load(path + "_x.pt")
        self.train_y = torch.load(path + "_y.pt")
        self.mean_x = torch.load(path + "_mean_x.pt")
        self.std_x = torch.load(path + "_std_x.pt")
        
        self.model = MultioutputGP(
            self.train_x, self.train_y, self.likelihood, 
            algo_dict.kernel_name[list_config[0][0]["kernel"]], num_task
        )

        self.model.load_state_dict(state_dict)
 
    @classmethod
    def load_from_path(cls, path):
        with open(f"{path}/config.json", 'r', encoding="utf-8") as f:
            data = json.load(f)
        
        using_first = data["using_first"]
        list_config = data["list_config"]
        num_task = len(list_config)

        model = cls([[]]*num_task, list_config, using_first)
        model.load(path)
        return model
    
