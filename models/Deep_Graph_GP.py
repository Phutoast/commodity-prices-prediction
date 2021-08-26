import torch
import numpy as np
import gpytorch
from gpytorch.likelihoods import MultitaskGaussianLikelihood

from models.GP_multi_index import GPMultiTaskIndex
from models.GP_model import DeepKernelMultioutputGP
from models.GP_model import TwoLayerGCN

from torch.utils.data import TensorDataset, DataLoader

class DeepGraphMultiOutputGP(GPMultiTaskIndex):
    
    expect_using_first = False

    def __init__(self, list_train_data, list_config, using_first):
        super().__init__(list_train_data, list_config, using_first)
        self.name = "deep_graph_gp"
        self.likelihood = MultitaskGaussianLikelihood(
                num_tasks=self.num_task
        )
    
    def repackage_data(self, data, data_ind, feat_size):  
        split_task = []
        for i in range(self.num_task):
            index_task = (data_ind == i).nonzero(as_tuple=True)[0]
            split_task.append(
                data[index_task].view(-1, 1, feat_size)
            )
        
        return torch.cat(split_task, axis=1)
    
    def reverse_package_out(self, data):
        data_T = torch.transpose(data, 0, 1)
        return torch.flatten(data_T, start_dim=0)

         
    def prepare_data(self):
        super().prepare_data()

        _, self.feat_size = self.train_x.size()

        self.train_x = self.repackage_data(self.train_x, self.train_ind, self.feat_size)

        self.train_y = torch.squeeze(self.repackage_data(
            self.train_y, self.train_ind, 1
        ))

        self.data_size = self.train_x.size(0)        

        underly_graph = torch.unsqueeze(torch.from_numpy(np.load(
            self.hyperparam["graph_path"]
        )), 0).float()
        self.num_node = underly_graph.shape[1]
        self.underly_graph = underly_graph
        # self.underly_graph = underly_graph.view(1, -1)

        assert self.num_node == self.num_task

        if self.hyperparam["is_gpu"]:
            self.underly_graph = self.underly_graph.cuda()
        
        return self.train_x.view(self.data_size, -1), self.train_y
    
    def build_training_model(self):
        assert len(set(
            self.train_ind.flatten().cpu().numpy().tolist()
        )) == self.num_node

        kernel = self.load_kernel(self.hyperparam["kernel"])
        self.graph_NN = TwoLayerGCN(
            num_feature=self.feat_size,
            hidden_channels=self.hyperparam["num_hidden_dim"],
            final_size=self.hyperparam["final_size"]
        )
        self.model = DeepKernelMultioutputGP(
            self.train_x, self.train_y, self.likelihood, 
            kernel, self.num_task, 
            self.feat_size, self.graph_NN, self.underly_graph
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
    
    def cal_train_loss(self):
        output = self.model(self.train_x)
        loss = -self.loss_obj(output, self.train_y)
        return output, loss
    
    def pred_all(self, all_data, test_ind, is_sample):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.from_numpy(all_data).float()
            test_ind = torch.from_numpy(test_ind).float()
            if self.hyperparam["is_gpu"]:
                test_x = test_x.cuda()
                test_ind = test_ind.cuda()

            test_x = self.normalize_data(test_x, is_train=False)
            test_x = self.repackage_data(test_x, test_ind, self.feat_size)
            test_x = test_x.view(test_x.size(0), -1)
        
        if not is_sample:
            pred = self.likelihood(self.model(test_x))
            pred_mean = self.reverse_package_out(pred.mean.detach().cpu())
            lower, upper = pred.confidence_region()
            pred_lower = self.reverse_package_out(lower.detach().cpu())
            pred_upper = self.reverse_package_out(upper.detach().cpu())
            return (
                pred_mean.numpy().flatten(), pred_lower.numpy().flatten(), 
                pred_upper.numpy().flatten(), test_ind
            )
        else:
            rv = self.model(test_x)
            rv = rv.sample(sample_shape=torch.Size([1000])).cpu()
            rv = torch.transpose(rv, 1, 2)
            rv = torch.flatten(rv, start_dim=1)
            return rv.numpy()

