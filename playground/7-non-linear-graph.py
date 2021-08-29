import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
import math
from utils.deep_config import create_deep_gp_config, create_dspp_config
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal

from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

num_inducing = 128
hidden_layer_size = 2
num_quad_site = 3

# lr=0.1, decay=0.95, iter=500
curr_config = create_deep_gp_config(num_inducing)

# lr=0.1, decay=0.9, iter=300
# curr_config = create_dspp_config(num_inducing, num_quad_site)

# Will compare this with the 

class GPDeepGraphModel(curr_config["class_type"]):
    def __init__(self, inp_feat_size, num_task, 
        graph_structure, hidden_layer_size=3): 

        LayerClass = curr_config["hidden_layer_type"]

        self.inp_feat_size = inp_feat_size
        self.num_task = num_task

        common_feat_extractor = LayerClass(
            input_dims=inp_feat_size,
            output_dims=hidden_layer_size,
            is_linear=True,
            **curr_config["hidden_info"]
        )

        # Adding self-loop ?  
        graph_structure = graph_structure + torch.eye(self.num_task)
        graph_structure = torch.min(graph_structure, torch.ones_like(graph_structure))
            
        self.all_neigh = [
            torch.nonzero(
                graph_structure[i, :], as_tuple=True
            )[0].cpu().tolist()
            for i in range(self.num_task)
        ]

        ind_extractor = [
            LayerClass(
                input_dims=hidden_layer_size*len(self.all_neigh[task]),
                output_dims=hidden_layer_size,
                is_linear=True,
                **curr_config["hidden_info"]
            )
            for task in range(num_task)
        ]
        
        if is_cuda:
            ind_extractor = [
                ind_extractor[i].cuda()
                for i in range(num_task)
            ]

        readout = LayerClass(
            input_dims=hidden_layer_size*num_task,
            output_dims=None,
            is_linear=False,
            **curr_config["hidden_info"]
        )
        
        super().__init__(**curr_config["class_init_info"])
        
        self.num_task = num_task
        self.common_feat_extractor = common_feat_extractor
        self.ind_extractor = ind_extractor
        self.readout = readout
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs): 
        indv_feat = [
            self.common_feat_extractor(inputs[:, task_i, :])
            for task_i in range(self.num_task)
        ]
        
        indv_feat_agg = []
        for task_j in range(self.num_task):
            feat_agg = self.ind_extractor[task_j](*[
                indv_feat[i] for i in self.all_neigh[task_j]
            ])
            indv_feat_agg.append(feat_agg)
        
        final = self.readout(*indv_feat_agg)
        return final

class GPDeepInteractionModel(curr_config["class_type"]):
    def __init__(self, inp_feat_size, num_task, hidden_layer_size=3): 

        LayerClass = curr_config["hidden_layer_type"]

        self.inp_feat_size = inp_feat_size
        self.num_task = num_task
        
        ind_extractor = [
            LayerClass(
                input_dims=inp_feat_size,
                output_dims=hidden_layer_size,
                is_linear=True,
                **curr_config["hidden_info"]
            )
            for task in range(num_task)
        ]
        
        if is_cuda:
            ind_extractor = [
                ind_extractor[i].cuda()
                for i in range(num_task)
            ]

        relation = LayerClass(
            input_dims=hidden_layer_size*2,
            output_dims=hidden_layer_size,
            is_linear=True,
            **curr_config["hidden_info"]
        )
        
        aggregator = LayerClass(
            input_dims=hidden_layer_size*(self.num_task-1),
            output_dims=hidden_layer_size,
            is_linear=True,
            **curr_config["hidden_info"]
        )

        readout = LayerClass(
            input_dims=hidden_layer_size*self.num_task,
            output_dims=None,
            is_linear=False,
            **curr_config["hidden_info"]
        )
        
        super().__init__(**curr_config["class_init_info"])
        
        self.num_task = num_task
        self.ind_extractor = ind_extractor
        self.relation = relation
        self.aggregator = aggregator
        self.readout = readout
        self.likelihood = GaussianLikelihood()
    
    def forward(self, inputs): 
        indv_feat = [
            self.ind_extractor[task_i](inputs[:, task_i, :])
            for task_i in range(self.num_task)
        ]

        all_out = np.zeros((self.num_task, self.num_task)).tolist()

        # Calcualte pairwise
        for task_j in range(1, self.num_task):
            for task_i in range(task_j):
                all_out[task_j][task_i] = self.relation(
                    indv_feat[task_i], indv_feat[task_j]
                )

        final_aggr = []
        for task_i in range(self.num_task):
            used_representation = []
            for j in range(self.num_task):
                if task_i == j:
                    pass
                else:
                    pair = all_out[task_i][j]
                    if pair != 0:
                        used_representation.append(pair)
                    else:
                        used_representation.append(all_out[j][task_i])

            final_aggr.append(self.aggregator(*used_representation))

        return self.readout(*final_aggr)

is_cuda = torch.cuda.is_available()

all_data, all_graph, all_eng = [t.float() for t in map(torch.from_numpy, [
    np.load("data/loc_velo.npy"),
    np.load("data/graphs.npy"),
    np.load("data/eng.npy")
])]

inp_feat_size = all_data.size(2)
num_task = all_data.size(1)
baseline_model = GPDeepGraphModel(
    inp_feat_size, num_task, torch.eye(num_task),
    hidden_layer_size=3
)

graph_model = GPDeepGraphModel(
    inp_feat_size, num_task, all_graph[0, :, :],
    hidden_layer_size=3
)

relation_model = GPDeepInteractionModel(
    inp_feat_size, num_task
)

if is_cuda:
    all_data = all_data.cuda()
    all_graph = all_graph.cuda()
    all_eng = all_eng.cuda()
    baseline_model = baseline_model.cuda()
    graph_model = graph_model.cuda()
    relation_model = relation_model.cuda()

def training_model(curr_model):

    optimizer = torch.optim.Adam(
        [{'params': curr_model.parameters()}], 
        lr=0.1
    )

    scheduler = ExponentialLR(optimizer, gamma=0.9)

    objective = curr_config["objective"](
        curr_model.likelihood, curr_model, 
        num_data=all_data.size(0)
    )


    train_dataset = TensorDataset(all_data, all_eng)
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
        
    for i in range(500):
        for j, (batch_data, batch_eng) in enumerate(train_loader):
            optimizer.zero_grad()
            output = curr_model(batch_data)
            loss = -objective(output, batch_eng)
            loss.backward()
            optimizer.step()
        
        if i%25 == 0:
            print(f"At Epoch {i} batch {j} Loss {loss}")
            scheduler.step()

    test_all_data, test_all_graph, test_all_eng = [t.float() for t in map(torch.from_numpy, [
        np.load("data/test_loc_velo.npy"),
        np.load("data/test_graphs.npy"),
        np.load("data/test_eng.npy")
    ])]

    if is_cuda:
        test_all_data = test_all_data.cuda()
        test_all_graph = test_all_graph.cuda()
        test_all_eng = test_all_eng.cuda()

    preds = curr_model.likelihood(curr_model(test_all_data))
    error = torch.sqrt(torch.mean((preds.mean.mean(0) - test_all_eng)**2))
    print(f"Test Loss is {error}")

training_model(baseline_model)
print("-------------------------")
training_model(graph_model)
training_model(relation_model)
