import numpy as np

import torch
import gpytorch
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO

from models.Deep_Graph_GP import DeepGraphMultiOutputGP
from models.GP_model import create_non_linear_mtl, create_deep_gp_config

from torch.utils.data import TensorDataset, DataLoader

class NonlinearMultiTaskGP(DeepGraphMultiOutputGP):

    expect_using_first = False
    is_external_likelihood = False

    def __init__(self, list_train_data, list_config, using_first):
        super().__init__(list_train_data, list_config, using_first)
        self.name = "non_linear_multi_task_gp"
        self.loss_class = lambda likelihood, model, num_data : DeepApproximateMLL(
            VariationalELBO(likelihood, model, num_data)
        )
        self.create_funct = create_deep_gp_config
    
    def build_training_model(self):
        self.model = create_non_linear_mtl(
            self.create_funct, self.feat_size, 
            self.num_task, self.hyperparam, 
            num_inducing=self.train_x.size(0)//(3*self.num_task),
            hidden_layer_size=self.hyperparam["num_hidden_dim"]
        ) 

        return self.model
    
    def build_optimizer_loss(self):
        self.model.train()
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.hyperparam["lr"]
        )

        self.loss_obj = self.loss_class(
            self.model.likelihood, self.model, 
            num_data=self.train_y.size(0)
        )

        self.train_x = self.train_x.view(-1, self.num_task, self.feat_size) 
        
        train_dataset = TensorDataset(self.train_x, self.train_y)

        self.train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True
        )

        return self.optimizer, self.loss_obj
    
    def training_loop(self, epoch, num_iter):
        num_batch = len(self.train_loader)
        for j, (x_batch, y_batch) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.model(x_batch)
            loss = -self.loss_obj(output, y_batch.T.flatten())
            loss.backward()
            self.optimizer.step()

            if epoch%10 == 0 and j%10 == 0 and self.hyperparam["is_verbose"]:
                print(f"Loss At Epoch {epoch}/{num_iter} At Batch {j}/{num_batch}", loss)

            # break
 
    def after_training(self):
        self.model.eval()
    
    def pred_all(self, all_data, test_ind, is_sample):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.from_numpy(all_data).float()
            test_ind = torch.from_numpy(test_ind).float()
            if self.hyperparam["is_gpu"]:
                test_x = test_x.cuda()
                test_ind = test_ind.cuda()
            test_x = self.normalize_data(test_x, is_train=False)
            test_x = self.repackage_data(test_x, test_ind, self.feat_size)
                
            rv = self.model(test_x)
            rv = rv.sample(sample_shape=torch.Size([1000])).mean(1).cpu().numpy()

            if not is_sample:
                pred = self.model.likelihood(self.model(test_x))
                pred_mean = pred.mean.mean(0)
                pred_var = pred.variance.mean(0)
                
                lower = pred_mean - 2 * pred_var.sqrt()
                upper = pred_mean + 2 * pred_var.sqrt()

                pred_mean = pred_mean.detach().cpu().numpy()
                pred_lower = lower.detach().cpu().numpy()
                pred_upper = upper.detach().cpu().numpy()
                return pred_mean, pred_lower, pred_upper, test_ind.cpu()
            else:
                rv = self.model(test_x)
                rv = rv.sample(sample_shape=torch.Size([1000])).mean(1).cpu().numpy()
                return rv
    
    def build_model_from_loaded(self, all_data, list_config, num_task):
        (state_dict, self.train_x, self.train_y, 
            self.mean_x, self.std_x, self.train_ind) = all_data
        
        _, self.feat_size = self.train_x.size()

        assert self.feat_size%self.num_task == 0
        self.feat_size = self.feat_size//self.num_task

        self.model = create_non_linear_mtl(
            self.create_funct, self.feat_size, 
            self.num_task, self.hyperparam, 
            num_inducing=self.train_x.size(0)//(3*self.num_task),
            hidden_layer_size=self.hyperparam["num_hidden_dim"]
        ) 

        return self.model
