import numpy as np
import pandas as pd
import json

import torch
import gpytorch
from models.train_model import BaseTrainMultiTask
from models.GP_model import MultiTaskGPIndexModel

from experiments import algo_dict
from utils import others
from utils.data_structure import Hyperparameters

import copy

class GPMultiTaskIndex(BaseTrainMultiTask):
    def __init__(self, list_train_data, list_config, using_first):
        super().__init__(list_train_data, list_config, using_first)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.list_config_json = {
            "list_config": list_config, "using_first": using_first
        }

        if self.hyperparam["is_gpu"]:
            self.likelihood = self.likelihood.cuda()

        assert not using_first
    
    def merge_all_data(self, data_list, label_list):
        train_ind = [
            np.ones(shape=(data_list[i].shape[0], 1), dtype=np.float32) * i
            for i in range(self.num_task)
        ] 
        all_train_data = np.concatenate(data_list, axis=0)
        all_train_ind = np.concatenate(train_ind, axis=0)
        return (all_train_data, all_train_ind), np.concatenate(label_list, axis=0).flatten()
     
    def prepare_data(self):
        (all_data, train_ind), self.train_y = self.pack_data_merge(
            self.train_data, self.hyperparam["is_past_label"]
        )

        self.train_y = torch.from_numpy(self.train_y).float()

        if self.hyperparam["is_time_only"]:
            self.train_x = torch.from_numpy(all_data[:, 0]).float()
        else:
            self.train_x = torch.from_numpy(all_data[:, :-1]).float()
        
        self.train_ind = torch.from_numpy(train_ind).float()
        
        if self.hyperparam["is_gpu"]:
            self.train_x = self.train_x.cuda()
            self.train_y = self.train_y.cuda()
            self.train_ind = self.train_ind.cuda()
        
        self.train_x = self.normalize_data(self.train_x, is_train=True)
        return self.train_x, self.train_y
 
    def build_training_model(self):
        kernel = self.load_kernel(self.hyperparam["kernel"])
        self.model = MultiTaskGPIndexModel(
            (self.train_x, self.train_ind), self.train_y, self.likelihood, kernel, self.num_task
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
    
    def cal_train_loss(self):
        output = self.model(self.train_x, self.train_ind)
        loss = -self.loss_obj(output, self.train_y)
        return output, loss

    def after_training(self):
        self.model.eval()
        self.likelihood.eval()

    def predict_step_ahead(self, list_test_data, list_step_ahead, list_all_date, ci=0.9, is_sample=False):
        self.model.eval()
        self.likelihood.eval() 
        
        (all_data, test_ind), _ = self.pack_data_merge(
            list_test_data, self.hyperparam["is_past_label"]
        )
        if self.hyperparam["is_time_only"]:
            all_data = all_data[:, 0]
        else:
            all_data = all_data[:, :-1]
        assert all(step_ahead <= all_data.shape[0] for step_ahead in list_step_ahead)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.from_numpy(all_data).float()
            test_ind = torch.from_numpy(test_ind).float()
            if self.hyperparam["is_gpu"]:
                test_x = test_x.cuda()
                test_ind = test_ind.cuda()
            test_x = self.normalize_data(test_x, is_train=False)

            if not is_sample:
                pred = self.likelihood(self.model(test_x, test_ind))
                pred_mean = pred.mean.detach().cpu().numpy()
                lower, upper = pred.confidence_region()
                pred_lower = lower.detach().cpu().numpy()
                pred_upper = upper.detach().cpu().numpy()
                test_ind = test_ind.detach().cpu()
            else:
                rv = self.model(test_x, test_ind)
                rv = rv.sample(sample_shape=torch.Size([1000])).numpy()
        
        if not is_sample:
            list_mean, list_lower, list_upper = [], [], [] 
            for i in range(self.num_task):
                index_task = (test_ind == i).nonzero(as_tuple=True)[0]
                list_mean.append(np.reshape(pred_mean[index_task], (-1)))
                list_lower.append(np.reshape(pred_lower[index_task], (-1)))
                list_upper.append(np.reshape(pred_upper[index_task], (-1)))
            
            return list_mean, list_lower, list_upper, list_all_date
        else:
            list_sample = []
            for i in range(self.num_task):
                index_task = (test_ind == i).nonzero(as_tuple=True)[0]
                list_sample.append(rv[:, index_task])

            assert all(
                list_sample[i].shape[1] == len(list_all_date[i])
                for i in range(self.num_task)
            )
            return list(zip(list_sample, list_all_date))


    def save(self, base_path):
        others.create_folder(base_path)
        path = base_path + "/multi_index_GP/"
        others.create_folder(path)
        path += "model"

        torch.save(self.model.state_dict(), path + ".pth")
        torch.save(self.train_x, path + "_x.pt")
        torch.save(self.train_y, path + "_y.pt")
        torch.save(self.mean_x, path + "_mean_x.pt")
        torch.save(self.std_x, path + "_std_x.pt")
        torch.save(self.train_ind, path + "_train_ind.pt")
        
        with open(f"{base_path}/config.json", 'w', encoding="utf-8") as f:
            json.dump(
                self.list_config_json, f, ensure_ascii=False, indent=4
            )
    
    def load(self, base_path): 
        with open(f"{base_path}/config.json", 'r', encoding="utf-8") as f:
            data = json.load(f)
        
        list_config = data["list_config"]
        num_task = len(list_config)
        
        path = base_path + "/multi_index_GP/"
        path += "model"

        all_ext = [".pth", "_x.pt", "_y.pt", "_mean_x.pt", "_std_x.pt", "_train_ind.pt"]
        all_data = []
        if self.hyperparam["is_gpu"]:
            all_data = [torch.load(path + ext, map_location="cuda:0") for ext in all_ext]
        else:
            all_data = [torch.load(path + ext, map_location=torch.device("cpu")) for ext in all_ext]
        
        (state_dict, self.train_x, self.train_y, 
            self.mean_x, self.std_x, self.train_ind) = all_data

        self.model = MultiTaskGPIndexModel(
            (self.train_x, self.train_ind), self.train_y, self.likelihood, 
            self.load_kernel(list_config[0][0]["kernel"]), num_task
        )
        
        if self.hyperparam["is_gpu"]:
            self.model.cuda()

        self.model.load_state_dict(state_dict)
 
    @classmethod
    def load_from_path(cls, path):
        with open(f"{path}/config.json", 'r', encoding="utf-8") as f:
            data = json.load(f)
        
        using_first = data["using_first"]
        list_config = data["list_config"]
        num_task = len(list_config)

        all_list_config = []
        for (dict_config, type_model) in list_config:
            all_list_config.append([Hyperparameters(**dict_config), type_model])

        model = cls([[]]*num_task, all_list_config, using_first)
        model.load(path)
        return model
    

