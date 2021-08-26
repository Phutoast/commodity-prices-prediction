import numpy as np

import torch
import gpytorch
from models.GP_multi_index import GPMultiTaskIndex
from models.GP_model import MultitaskSparseGPIndex

from utils import others
from utils.data_structure import Hyperparameters
from torch.utils.data import TensorDataset, DataLoader

import math

class SparseGPIndex(GPMultiTaskIndex):
    
    expect_using_first = False

    def __init__(self, list_train_data, list_config, using_first):
        super().__init__(list_train_data, list_config, using_first)
        self.name = "sparse_multi_index_gp"
    
    def create_ind_points(self, train_x, num_task):
        total_points = train_x.size(0)//3
        each_task_points = math.ceil(total_points/num_task)
        total_points = each_task_points * num_task

        ind_index = torch.cat([
            torch.ones(each_task_points, 1)*i
            for i in range(num_task)
        ], axis=0)
        ind_points = torch.randn(*train_x[:total_points].size())
        return ind_index, ind_points
    
    def build_training_model(self):
        kernel = self.load_kernel(self.hyperparam["kernel"])

        self.ind_index, self.ind_points = self.create_ind_points(
            self.train_x, self.num_task
        )

        self.model = MultitaskSparseGPIndex(
            self.ind_points, kernel, self.num_task
        )
        return self.model
    
    def build_optimizer_loss(self):
        self.model.train()
        self.likelihood.train()

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=self.hyperparam["lr"])

        self.loss_obj = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model,
            num_data=self.train_y.size(0)
        )
 
        train_dataset = TensorDataset(
            torch.cat([self.train_x, self.train_ind], dim=1), self.train_y
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=True
        )

        return self.optimizer, self.loss_obj
    
    def training_loop(self, epoch, num_iter):
        # Training in batch instead

        num_batch = len(self.train_loader)
        for j, (xi_batch, y_batch) in enumerate(self.train_loader):

            x_batch = xi_batch[:, :-1]
            i_batch = xi_batch[:, -1].view(-1, 1)

            self.optimizer.zero_grad()
            output = self.model(
                x_batch, all_ind=torch.cat([self.ind_index, i_batch], axis=0)
            )
            loss = -self.loss_obj(output, y_batch)
            loss.backward()
            self.optimizer.step()
            
            if j%5 == 0 and self.hyperparam["is_verbose"]:
                print(f"Loss At Epoch {epoch}/{num_iter} At Batch {j}/{num_batch}", loss)
            
    def pred_all(self, all_data, test_ind, is_sample):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.from_numpy(all_data).float()
            test_ind = torch.from_numpy(test_ind).float()
            if self.hyperparam["is_gpu"]:
                test_x = test_x.cuda()
                test_ind = test_ind.cuda()
            test_x = self.normalize_data(test_x, is_train=False)

        rv = self.model(
            test_x, all_ind=torch.cat([self.ind_index, test_ind], axis=0)
        )
        if not is_sample:
            pred = self.likelihood(self.model(
                test_x, all_ind=torch.cat([self.ind_index, test_ind], axis=0)
            ))
            pred_mean = pred.mean.detach().cpu().numpy()
            lower, upper = pred.confidence_region()
            pred_lower = lower.detach().cpu().numpy()
            pred_upper = upper.detach().cpu().numpy()
            test_ind = test_ind.detach().cpu()
            return pred_mean, pred_lower, pred_upper, test_ind
        else:
            rv = self.model(
                test_x, all_ind=torch.cat([self.ind_index, test_ind], axis=0)
            )
            rv = rv.sample(sample_shape=torch.Size([1000])).cpu().numpy()
            return rv

    def build_model_from_loaded(self, all_data, list_config, num_task):
        (state_dict, self.train_x, self.train_y, 
            self.mean_x, self.std_x, self.train_ind) = all_data

        self.ind_index, self.ind_points = self.create_ind_points(
            self.train_x, self.num_task
        )

        self.model = MultitaskSparseGPIndex(
            self.ind_points, self.load_kernel(list_config[0][0]["kernel"]), 
            num_task
        )

        