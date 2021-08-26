import numpy as np

import torch
import gpytorch
from models.Sparse_GP_index import SparseGPIndex
from models.GP_model import SparseGraphGP

from utils import others
from utils.data_structure import Hyperparameters
from torch.utils.data import TensorDataset, DataLoader

import math

class SparseMaternGraphGP(SparseGPIndex):
    
    expect_using_first = False
    
    def __init__(self, list_train_data, list_config, using_first):
        super().__init__(list_train_data, list_config, using_first)
        self.name = "matern_graph_gp"

        self.underly_graph = torch.from_numpy(np.load(
            self.hyperparam["graph_path"]
        ))
        eigen_val, eigen_vec = self.get_eigen_pairs(
            self.underly_graph
        )

        self.eigen_val = eigen_val.float()
        self.eigen_vec = eigen_vec.float()
    
    def get_eigen_pairs(self, adj_matrix):
        laplacian = torch.diag(
            torch.sum(adj_matrix, axis=1, keepdims=False)
        ) - adj_matrix

        eigen_val, eigen_vec = torch.linalg.eigh(laplacian)
        return eigen_val, eigen_vec
    
    def build_training_model(self):
        assert len(set(
            self.train_ind.flatten().cpu().numpy().tolist()
        )) == self.underly_graph.shape[0]
        
        kernel = self.load_kernel(self.hyperparam["kernel"])
        
        self.ind_index, self.ind_points = self.create_ind_points(
            self.train_x, self.num_task
        )

        self.model = SparseGraphGP(
            self.ind_points, kernel, (self.eigen_vec, self.eigen_val)
        )

        return self.model
    


