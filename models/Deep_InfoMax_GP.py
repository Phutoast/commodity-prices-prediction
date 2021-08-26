import torch
import numpy as np
import gpytorch
from gpytorch.likelihoods import MultitaskGaussianLikelihood

from models.Deep_Graph_GP import DeepGraphMultiOutputGP
from models.GP_model import DeepKernelMultioutputGP
from models.GP_model import TwoLayerGCN

from torch.nn import Parameter
from torch_geometric.nn.inits import uniform
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

class DeepGraphInfoMaxMultiOutputGP(DeepGraphMultiOutputGP):
    
    expect_using_first = False

    def __init__(self, list_train_data, list_config, using_first):
        super().__init__(list_train_data, list_config, using_first)
        self.name = "deep_graph_infomax_gp"
        self.likelihood = MultitaskGaussianLikelihood(
                num_tasks=self.num_task
        )
    
    def pretrain_infomax(self, graph_NN):
        train_x = self.train_x.view(-1, self.num_task, self.feat_size) 

        discri = Discrimenator(self.hyperparam["final_size"])

        optim_infomax = torch.optim.Adam([
            {'params': discri.parameters()},
            {'params': graph_NN.parameters()},
        ], lr=0.001)

        graph_NN.train()
        discri.train()

        for epoch in range(2500):

            optim_infomax.zero_grad()
            corruptor = torch.randperm(train_x.size(1))
            not_corrupt = graph_NN(
                train_x, self.underly_graph, 
                is_summary=False
            )
            
            corrupt = graph_NN(
                train_x[:, corruptor, :], self.underly_graph, 
                is_summary=False
            )

            all_loss = torch.mean(
                -torch.log(discri(not_corrupt, not_corrupt) + 1e-15), 
                dim=1
            ) + torch.mean(
                -torch.log(1-discri(corrupt, not_corrupt) + 1e-15)
            )

            mean_loss = torch.mean(all_loss)

            if epoch%500 == 0 and self.hyperparam["is_verbose"]:
                print(f"At Epoch {epoch}. Loss {mean_loss}")
            
            mean_loss.backward()
            optim_infomax.step()
        
        graph_NN.eval()
        discri.eval()

    
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

        self.pretrain_infomax(self.graph_NN)

        self.model = DeepKernelMultioutputGP(
            self.train_x, self.train_y, self.likelihood, 
            kernel, self.num_task, 
            self.feat_size, self.graph_NN, self.underly_graph
        )
        return self.model

class Discrimenator(nn.Module):
    # Following from pytorch_geometric/DeepGraphInfomax
    def __init__(self, size_hidden):
        super(Discrimenator, self).__init__()
        self.weight = Parameter(torch.Tensor(size_hidden, size_hidden))
        self.sigmoid = nn.Sigmoid()

        uniform(size_hidden, self.weight)
    
    def forward(self, graph_latent, graph_latent_true):

        true_z = self.sigmoid(
            torch.mean(graph_latent_true, dim=1)
        ).unsqueeze(-1)

        return self.sigmoid(
            torch.matmul(graph_latent, torch.matmul(self.weight, true_z))
        ).squeeze()
