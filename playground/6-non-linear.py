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

num_inducing = 128
hidden_layer_size = 2
num_quad_site = 3

# lr=0.1, decay=0.95, iter=500
curr_config = create_deep_gp_config(num_inducing)

# lr=0.1, decay=0.9, iter=300
# curr_config = create_dspp_config(num_inducing, num_quad_site)

is_cuda = torch.cuda.is_available()

# ----------------------------------------------------------------------------

train_x = torch.linspace(0, 1, 100)

def generate_data(x):
    g_out = -torch.sin(8*math.pi*(x+1))/(2*x+1) - x**4
    h1 = torch.sin(3*x)
    h2 = 3*x
    f1 = torch.cos(g_out)**2 + h1
    f2 = torch.sin(10*x)*g_out**2 + h2
    return (
        torch.cat([f1, f2]), 
        torch.cat([torch.ones_like(f1)*0, torch.ones_like(f2)*1])
    )

train_y, train_ind = generate_data(train_x)
train_x = torch.cat([train_x, train_x])

num_show_points = 50
show_points = torch.randperm(100)[:num_show_points]

f1, f2 = train_y[:100], train_y[100:]
x, f1, f2 = train_x.numpy()[:100], f1.numpy(), f2.numpy()

num_task = 2

# ----------------------------------------------------------------------------

class NonLinearMultiTask(curr_config["class_type"]):
    def __init__(self, train_x_shape, num_task, hidden_layer_size=3): 
        LayerClass = curr_config["hidden_layer_type"]

        mean_hidden_layer = LayerClass(
            input_dims=train_x_shape[-1],
            output_dims=hidden_layer_size,
            is_linear=True,
            **curr_config["hidden_info"]
        )

        all_hidden_layer = [
            LayerClass(
                input_dims=train_x_shape[-1],
                output_dims=hidden_layer_size,
                is_linear=True,
                **curr_config["hidden_info"]
            )
            for _ in range(num_task)
        ]

        if is_cuda:
            all_hidden_layer = [
                all_hidden_layer[i].cuda()
                for i in range(num_task)
            ]

        self.num_task = num_task


        last_layer = LayerClass(
            input_dims=hidden_layer_size*2,
            output_dims=None,
            is_linear=False,
            **curr_config["hidden_info"]
        )
        
        super().__init__(**curr_config["class_init_info"])
        
        self.mean_hidden_layer = mean_hidden_layer
        self.all_hidden_layer = all_hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()
    
    def forward(self, inputs, all_ind):
        
        all_task_result = []

        for task_i in range(self.num_task):
            index_task_i = (all_ind == 0).nonzero().squeeze()
            inp_task = inputs[index_task_i, :]

            mean_out = self.mean_hidden_layer(inp_task)
            specific_out = self.all_hidden_layer[task_i](inp_task)

            print(mean_out)
            print(specific_out)

            all_task_result.append(
                self.last_layer(mean_out, specific_out)
            )
        
        # num_task = 3
        # Therefore for num_sample x b x b 
        # We expect a final size of num_sample x (b*num_task) x (b*num_task)
        
        # Test Batch of numbers num_task: 3, num_sample: 2
        # a = torch.stack([torch.ones(3, 3)*1, torch.ones(3, 3)*2])
        # b = torch.stack([torch.ones(3, 3)*3, torch.ones(3, 3)*4])
        # c = torch.stack([torch.ones(3, 3)*5, torch.ones(3, 3)*6])
        
        # mat_size = a.size(1)

        # expect_out = torch.stack([torch.block_diag(*[
        #         task[sample, :, :]
        #         for task in [a, b, c]  
        #     ])
        #     for sample in range(a.size(0))
        # ])


        # Assuming sample matrix size
        mat_size = all_task_result[0].covariance_matrix.size(1)
        
        # print(all_task_result[0].mean.size())
        print(all_task_result[0].covariance_matrix[0, :, :])
        print(all_task_result[0].covariance_matrix[0, :, :].size())
        assert False


        # Block Diagonal Across Batch
        block_sample_independent = torch.sum(torch.stack([
            F.pad(task_sample.covariance_matrix, (
                mat_size*i, mat_size*(self.num_task-1-i), mat_size*i, mat_size*(self.num_task-1-i)
            ))
            for task_sample, i in zip(all_task_result, range(num_task))
        ]), dim=0)

        return MultivariateNormal(
            torch.cat([all_task_result[i].mean for i in range(self.num_task)], dim=1),
            block_sample_independent
        )

# ----------------------------------------------------------------------------

batch_train_x = train_x.unsqueeze_(1)

model = NonLinearMultiTask(batch_train_x.size(), 2)
# model(batch_train_x, train_ind)
if is_cuda:
    batch_train_x = batch_train_x.cuda()
    train_ind = train_ind.cuda()
    train_y = train_y.cuda()
    model = model.cuda()

objective = curr_config["objective"](
    model.likelihood, model, 
    num_data=batch_train_x.size(0)
)

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.1)
scheduler = ExponentialLR(optimizer, gamma=0.95)

for i in range(500):
    optimizer.zero_grad()
    output = model(batch_train_x, train_ind)
    loss = -objective(output, train_y)
    loss.backward()
    optimizer.step()

    if i%25 == 0:
        print(f"At Epoch {i} Loss {loss}")
        # scheduler.step()


# ----------------------------------------------------------------------------

test_x = torch.unsqueeze(torch.cat([
    torch.linspace(0, 1, 50), 
    torch.linspace(0, 1, 50)], dim=0
), dim=1)
test_ind = torch.cat([torch.ones(50)*0, torch.ones(50)*1])

if is_cuda:
    test_x = test_x.cuda()
    test_ind = test_ind.cuda()

out_likelihood = model.likelihood(model(test_x, test_ind))
out_mean = out_likelihood.mean.mean(0)
out_var = out_likelihood.variance.mean(0)

lower = out_mean - 2 * out_var.sqrt()
upper = out_mean + 2 * out_var.sqrt()

# assert False


fig, axes = plt.subplots(nrows=2, figsize=(10, 10))
axes[0].plot(x, f1)
axes[0].plot(test_x[:50, 0].cpu().numpy(), out_mean[:50].cpu().detach().numpy(), c='r')
axes[0].fill_between(test_x[:50, 0].cpu().numpy(), lower[:50].cpu().detach().numpy(), upper[:50].cpu().detach().numpy(), alpha=0.2, color='r')
axes[0].scatter(x[show_points], f1[show_points], marker="x", s=40, c='k')
axes[0].grid()

axes[1].plot(x, f2)
axes[1].plot(test_x[50:, 0].cpu().numpy(), out_mean[50:].cpu().detach().numpy(), c='r')
axes[1].fill_between(test_x[:50, 0].cpu().numpy(), lower[50:].cpu().detach().numpy(), upper[50:].cpu().detach().numpy(), alpha=0.2, color='r')
axes[1].scatter(x[show_points], f2[show_points], marker="x", s=40, c='k')
axes[1].grid()

plt.show()



