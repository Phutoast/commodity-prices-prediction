import torch
import gpytorch
from gpytorch.likelihoods import MultitaskGaussianLikelihood
import math
import matplotlib.pyplot as plt

train_x = torch.linspace(0, 1, 100)

train_y = torch.stack([
    torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
    torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
    torch.sin(train_x * (2 * math.pi)) + 2 * torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
    -torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
], -1)

train_x = train_x.unsqueeze(-1)
num_tasks = train_y.size(-1)

from utils.deep_config import create_deep_gp_config, create_dspp_config

num_inducing = 128
num_quad_site = 3
hidden_layer_size = 2

# curr_config = create_dspp_config(num_inducing, num_quad_site)
curr_config = create_deep_gp_config(num_inducing)

class MultitaskDeepGP(curr_config["class_type"]):
    def __init__(self, train_x_shape):
        LayerClass = curr_config["hidden_layer_type"]

        hidden_layer = LayerClass(
            input_dims=train_x_shape[-1],
            output_dims=hidden_layer_size,
            is_linear=True,
            **curr_config["hidden_info"]
        )
        last_layer = LayerClass(
            input_dims=hidden_layer.output_dims,
            output_dims=num_tasks,
            is_linear=False,
            **curr_config["hidden_info"]
        )
        
        super().__init__(**curr_config["class_init_info"])

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output

model = MultitaskDeepGP(train_x.shape)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
objective = curr_config["objective"](
    model.likelihood, model, 
    num_data=train_y.size(0)
)

for i in range(10):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -objective(output, train_y)
    loss.backward()
    optimizer.step()

    if i%10 == 0:
        print(f"At Epoch {i} Loss {loss}")

model.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 1, 51).unsqueeze(-1)
    
    preds = model.likelihood(model(test_x)).to_data_independent_dist()

    mean, var = preds.mean.mean(0), preds.variance.mean(0)
    lower = mean - 2 * var.sqrt()
    upper = mean + 2 * var.sqrt()

fig, axs = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))
for task, ax in enumerate(axs):
    ax.plot(train_x.squeeze(-1).detach().numpy(), train_y[:, task].detach().numpy(), 'k*')
    ax.plot(test_x.squeeze(-1).numpy(), mean[:, task].numpy(), 'b')
    ax.fill_between(test_x.squeeze(-1).numpy(), lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    ax.set_title(f'Task {task + 1}')
fig.tight_layout()
plt.show()

