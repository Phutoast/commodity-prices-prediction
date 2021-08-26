import torch
import gpytorch
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.constraints import Positive

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


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
        self.task_covar_module = gpytorch.kernels.IndexKernel(
            num_tasks=num_task, rank=min(2, num_task)
        )
    
    def forward(self, x, i):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        covar_i = self.task_covar_module(i)
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

class MultitaskSparseGPIndex(ApproximateGP):
    def __init__(self, ind_pts, kernel, num_task):
        
        var_dist = CholeskyVariationalDistribution(ind_pts.size(0))
        var_strategy = VariationalStrategy(
            self, ind_pts, var_dist, 
            learn_inducing_locations=True
        )
        super(MultitaskSparseGPIndex, self).__init__(var_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
        self.task_covar_module = gpytorch.kernels.IndexKernel(
            num_tasks=num_task, rank=min(2, num_task)
        )

    def forward(self, x, all_ind):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        covar_i = self.task_covar_module(all_ind)
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)    
        
def create_deep_GP(
        create_config_funct, train_x_shape, num_tasks, hyperparameters, 
        num_inducing=256, hidden_layer_size=32, num_quad_site=3
    ):

    curr_config = create_config_funct(num_inducing, num_quad_site)

    class TwoLayerMultitaskDeepGP(curr_config["class_type"]):
        def __init__(self, train_x_shape, num_tasks):
            LayerClass = curr_config["hidden_layer_type"]

            hidden_layer = LayerClass(
                input_dims=train_x_shape[-1],
                output_dims=hidden_layer_size,
                is_linear=True,
                kernel_type=hyperparameters["kernel"],
                **curr_config["hidden_info"]
            )
            last_layer = LayerClass(
                input_dims=hidden_layer.output_dims,
                output_dims=num_tasks,
                is_linear=False,
                kernel_type=hyperparameters["kernel"],
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
    
    return TwoLayerMultitaskDeepGP(train_x_shape, num_tasks)

class GraphKernel(gpytorch.kernels.Kernel):
    is_stationary = False
    
    def __init__(self, eigenpairs, nu=3, kappa=4, sigma_f=1, **kwargs):
        super().__init__(**kwargs)

        self.eigenvectors, self.eigenvalues = eigenpairs
        self.num_verticies = self.eigenvectors.size()[0]

        name_value = [("nu", nu), ("kappa", kappa), ("sigma_f", sigma_f)]

        for name, value in name_value:
            self.register_parameter(
                name="raw_"+name, 
                parameter=torch.nn.Parameter(
                    torch.ones(*self.batch_shape, 1, 1) * value
                )
            )
            self.register_constraint("raw_"+name, Positive())
    
    @property
    def nu(self):
        return self.raw_nu_constraint.transform(self.raw_nu)
    
    @nu.setter
    def nu(self, value):
        return self._set_nu(value)
    
    def _set_nu(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_nu)
        
        self.initialize(
            raw_nu=self.raw_nu_constraint.inverse_transform(value)
        )
    
    @property
    def kappa(self):
        return self.raw_kappa_constraint.transform(self.raw_kappa)
    
    @kappa.setter
    def kappa(self, value):
        return self._set_kappa(value)
    
    def _set_kappa(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_kappa)
        
        self.initialize(
            raw_kappa=self.raw_kappa_constraint.inverse_transform(value)
        )
    
    @property
    def sigma_f(self):
        return self.raw_sigma_f_constraint.transform(self.raw_sigma_f)
     
    @sigma_f.setter
    def sigma_f(self, value):
        return self._set_sigma_f(value)
    
    def _set_sigma_f(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_sigma_f)
        
        self.initialize(
            raw_sigma_f=self.raw_sigma_f_constraint.inverse_transform(value)
        )

    def forward(self, x1, x2, **params):
        x1, x2 = x1.long(), x2.long()

        S1 = torch.pow(self.eigenvalues + 2*self.nu/(self.kappa**2), -self.nu)
        S2 = S1 * (self.num_verticies*self.sigma_f)/torch.sum(S1)

        # Very simple No Batch At All
        U1 = self.eigenvectors[x1.flatten(), :]
        U2 = self.eigenvectors[x2.flatten(), :]

        return torch.matmul(U1 * S2, U2.T)

class SparseGraphGP(ApproximateGP):
    def __init__(self, ind_pts, kernel, eigen_pairs):

        var_dist = CholeskyVariationalDistribution(ind_pts.size(0))
        var_strategy = VariationalStrategy(
            self, ind_pts, var_dist, 
            learn_inducing_locations=True
        )
        super(SparseGraphGP, self).__init__(var_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
        self.task_covar_module = GraphKernel(
            eigen_pairs, nu=3/2, 
            kappa=5, sigma_f=1
        )

    def forward(self, x, all_ind):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        covar_i = self.task_covar_module(all_ind)
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)    
