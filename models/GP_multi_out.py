import numpy as np
import pandas as pd
import json

import torch
import gpytorch
from models.train_model import BaseTrainMultiTask
from models.GP_model import MultioutputGP
from models.GP import IndependentGP
from utils.data_structure import Hyperparameters

from experiments import algo_dict
from utils import others

import matplotlib.pyplot as plt
import pandas as pd
from utils import data_visualization

import copy

class GPMultiTaskMultiOut(BaseTrainMultiTask):

    expect_using_first = True

    def __init__(self, list_train_data, list_config, using_first):
        super().__init__(list_train_data, list_config, using_first)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.num_task
        )
        self.list_config_json = {
            "list_config": list_config, "using_first": using_first
        }
        assert using_first
        
        self.list_train_data = list_train_data
        self.list_config = list_config
        self.using_first = using_first
        
        if self.hyperparam["is_gpu"]:
            self.likelihood = self.likelihood.cuda()
    
    def prepare_data(self):
        all_data, self.train_y = self.pack_data_merge(
            self.train_data, self.hyperparam["is_past_label"], self.using_first
        )
        self.train_y = torch.from_numpy(self.train_y).float()

        if self.hyperparam["is_time_only"]:
            self.train_x = torch.from_numpy(all_data[:, 0]).float()
        else:
            self.train_x = torch.from_numpy(all_data[:, :-1]).float()

        if self.hyperparam["is_gpu"]:
            self.train_x = self.train_x.cuda()
            self.train_y = self.train_y.cuda()
        
        self.train_x = self.normalize_data(self.train_x, is_train=True)
        return self.train_x, self.train_y
 
    def build_training_model(self):
        kernel = self.load_kernel(self.hyperparam["kernel"])
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
    
    def merge_all_data(self, data_list, label_list):
        return data_list, label_list

    def predict_step_ahead(self, list_test_data, list_step_ahead, list_all_date, ci=0.9, is_sample=False):
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
        
        all_data_list, _ = self.pack_data_merge(
            list_test_data, self.hyperparam["is_past_label"], using_first=False
        )
        all_data = all_data_list[0]

        if self.hyperparam["is_time_only"]:
            all_data = all_data[:, 0]
        else:
            all_data = all_data[:, :-1]
        
        # Make sure that the first one is the biggest
        max_size = all_data_list[0].shape[0]
        assert max(d.shape[0] for d in all_data_list) == max_size
        assert all(step_ahead <= data.shape[0] for step_ahead, data in zip(list_step_ahead, all_data_list))

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_cg_iterations(2000):
            test_x = torch.from_numpy(all_data).float()
            if self.hyperparam["is_gpu"]:
                test_x = test_x.cuda()
            test_x = self.normalize_data(test_x, is_train=False)

            if not is_sample:
                pred = self.likelihood(self.model(test_x))
                pred_mean = pred.mean.detach().cpu().numpy()
                lower, upper = pred.confidence_region()
                pred_lower = lower.detach().cpu().numpy()
                pred_upper = upper.detach().cpu().numpy()
            else:
                rv = self.model(test_x)
                rv = rv.sample(sample_shape=torch.Size([1000])).cpu().numpy()

        if not is_sample:
            list_pred_mean = []        
            list_pred_lower = []
            list_pred_upper = []
            all_date = []

            for i in range(self.num_task):
                expect_size_data = all_data_list[i].shape[0]
                advance_index = max_size - expect_size_data
                list_pred_mean.append(pred_mean[advance_index:, i])
                list_pred_lower.append(pred_lower[advance_index:, i])
                list_pred_upper.append(pred_upper[advance_index:, i])
                all_date.append(list_all_date[i][advance_index:])

            return list_pred_mean, list_pred_lower, list_pred_upper, all_date
        else:
            list_sample = []
            all_date = []
            for i in range(self.num_task):
                expect_size_data = all_data_list[i].shape[0]
                advance_index = max_size - expect_size_data
                list_sample.append(rv[:, :, i][:, advance_index:])
                all_date.append(list_all_date[i][advance_index:])

            return list_sample, all_date

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
        
        others.dump_json(f"{base_path}/config.json", self.list_config_json)
    
    def load(self, base_path): 
        data = others.load_json(f"{base_path}/config.json")
        
        list_config = data["list_config"]
        num_task = len(list_config)
        
        path = base_path + "/multi_GP/"
        path += "model"
        
        all_ext = [".pth", "_x.pt", "_y.pt", "_mean_x.pt", "_std_x.pt"]
        all_data = []
        if self.hyperparam["is_gpu"]:
            all_data = [torch.load(path + ext, map_location="cuda:0") for ext in all_ext]
        else:
            all_data = [torch.load(path + ext, map_location=torch.device('cpu')) for ext in all_ext]
        
        (state_dict, self.train_x, self.train_y, 
            self.mean_x, self.std_x) = all_data
        
        self.model = MultioutputGP(
            self.train_x, self.train_y, self.likelihood, 
            self.load_kernel(list_config[0][0]["kernel"]), num_task
        )
        
        if self.hyperparam["is_gpu"]:
            self.model.cuda()

        self.model.load_state_dict(state_dict)
 
    @classmethod
    def load_from_path(cls, path):
        data = others.load_json(f"{path}/config.json")
        
        using_first = data["using_first"]
        list_config = data["list_config"]
        num_task = len(list_config)
        
        all_list_config = []
        for (dict_config, type_model) in list_config:
            all_list_config.append([Hyperparameters(**dict_config), type_model])

        model = cls([[]]*num_task, all_list_config, using_first)
        model.load(path)
        return model
    

