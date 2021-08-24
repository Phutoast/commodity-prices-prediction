import torch

import gpytorch
from gpytorch.likelihoods import GaussianLikelihood

from scipy.io import loadmat
from math import floor
from torch.utils.data import TensorDataset, DataLoader

from utils.deep_config import create_deep_gp_config, create_dspp_config

num_inducing = 128
num_quad_site = 3
hidden_layer_size = 2

curr_config = create_deep_gp_config(num_inducing)

class TwoLayerDeep(curr_config["class_type"]):
    def __init__(self, train_x_shape): 
        LayerClass = curr_config["hidden_layer_type"]

        hidden_layer = LayerClass(
            input_dims=train_x_shape[-1],
            output_dims=hidden_layer_size,
            is_linear=True,
            **curr_config["hidden_info"]
        )

        second_layer = LayerClass(
            input_dims=train_x_shape[-1],
            output_dims=hidden_layer_size,
            is_linear=True,
            **curr_config["hidden_info"]
        )

        last_layer = LayerClass(
            input_dims=hidden_layer.output_dims*2,
            output_dims=None,
            is_linear=False,
            **curr_config["hidden_info"]
        )
        
        super().__init__(**curr_config["class_init_info"])
        
        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.second_layer = second_layer
        self.likelihood = GaussianLikelihood()
    
    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        hidden_rep2 = self.second_layer(inputs) 
        output = self.last_layer(hidden_rep1, hidden_rep2)
        return output

# ---------------------------------------------------

def load_data():
    data = torch.Tensor(loadmat("data/elevators.mat")["data"])
    X = data[:, :-1]
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    y = data[:, -1]
    return X, y

X, y = load_data()

train_n = int(floor(0.8 * len(X)))
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()

test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()

train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

# ---------------------------------------------------

model = TwoLayerDeep(train_x.shape)
adam = torch.optim.Adam(
    [{'params': model.parameters()}], 
    lr=0.05, betas=(0.9, 0.999)
)
objective = curr_config["objective"](
    model.likelihood, model, 
    num_data=train_n
)

for i in range(1):
    for j, (x_batch, y_batch) in enumerate(train_loader):
        adam.zero_grad()
        output = model(x_batch)
        loss = -objective(output, y_batch)
        loss.backward()
        adam.step()
        print(f"At Batch {j} of Epoch {i} Loss {loss}")


model.eval()

mus, var, lls = [], [], []
with torch.no_grad():
    for i, (x_batch, y_batch) in enumerate(test_loader):
        print(f"Here {i}/{len(test_loader)}")
        preds = model.likelihood(model(x_batch))
        mus.append(preds.mean)
        var.append(preds.variance)
        lls.append(curr_config["log_likelihood_cal"](model, x_batch, y_batch))

mus, var, lls = torch.cat(mus, dim=-1), torch.cat(var, dim=-1), torch.cat(lls, dim=-1)
rmse = torch.mean(torch.pow(mus.mean(0) - test_y, 2)).sqrt()
print(f"RMSE: {rmse.item()}, NLL: {-lls.mean().item()}")




