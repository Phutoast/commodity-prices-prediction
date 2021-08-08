import gpytorch
import torch

class OneDimensionGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(OneDimensionGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
    
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class BatchGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, num_out):
        super(BatchGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_out])
        )
        self.covar_module = kernel
    
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean, covar)
        )

class MultioutputGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, num_out):
        super(MultioutputGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_out
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            kernel, num_tasks=num_out, rank=min(2, num_out)
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class MultiTaskGPIndexModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, num_task):
        super(MultiTaskGPIndexModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_task, rank=2)
    
    def forward(self, x, i):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        covar_i = self.task_covar_module(i)
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)