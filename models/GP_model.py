import torch
import gpytorch
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.constraints import Positive

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

from models.deep_layers import DSPPHHiddenLayer, DeepGPHiddenLayer
from gpytorch.models.deep_gps import DeepGP

from models.deep_layers import DSPPHHiddenLayer
from gpytorch.models.deep_gps.dspp import DSPP
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood

from models.Conv_Graph_NN import CustomGCN
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

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
            kernel, num_tasks=num_out, rank=min(5, num_out)
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
            num_tasks=num_task, rank=min(5, num_task)
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
            num_tasks=num_task, rank=min(5, num_task)
        )

    def forward(self, x, all_ind):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        covar_i = self.task_covar_module(all_ind)
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)    

def deep_gp_ll(model, x_batch, y_batch):
    return model.likelihood.log_marginal(y_batch, model(x_batch)).cpu()

def create_deep_gp_config(num_inducing, num_quad_site):
    return {
        "class_type": DeepGP,
        "hidden_layer_type": DeepGPHiddenLayer,
        "hidden_info": {"num_inducing": num_inducing},
        "class_init_info": {},
        "log_likelihood_cal": deep_gp_ll,
    }

def dspp_ll(model, x_batch, y_batch):
    base_batch_ll = model.likelihood.log_marginal(y_batch, model(x_batch))
    deep_batch_ll = model.quad_weights.unsqueeze(-1) + base_batch_ll
    batch_log_prob = deep_batch_ll.logsumexp(dim=0)
    return batch_log_prob.cpu()

def create_dspp_config(num_inducing, num_quad_site):
    return {
        "class_type": DSPP,
        "hidden_layer_type": DSPPHHiddenLayer,
        "hidden_info": {"num_inducing": num_inducing, "num_quad_sites": num_quad_site},
        "class_init_info": {"num_quad_sites": num_quad_site},
        "log_likelihood_cal": dspp_ll,
    }


        
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

class TwoLayerGCN(torch.nn.Module):
    def __init__(self, num_feature, hidden_channels, final_size):
        super(TwoLayerGCN, self).__init__()
        self.conv1 = CustomGCN(num_feature, hidden_channels)
        self.out = CustomGCN(hidden_channels, final_size)
        self.leakyReLU = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, graph_batch, is_summary=True):
        x = self.conv1(x, graph_batch)
        x = self.leakyReLU(x)
        x = self.dropout(x)
        x = self.out(x, graph_batch)

        np.save("embedding.npy", x.detach().cpu().numpy())

        if is_summary:
            x = torch.mean(x, dim=1)

        return x
    
class DeepKernelMultioutputGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, 
        likelihood, kernel, num_task, 
        num_feature, hyperparam, graph_adj, is_freeze=False):

        super(DeepKernelMultioutputGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_task
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            kernel, num_tasks=num_task, rank=min(5, num_task)
        )

        self.num_task = num_task
        self.num_feature = num_feature
        self.graph_adj = graph_adj
        
        self.feature_extractor = TwoLayerGCN(
            num_feature=num_feature,
            hidden_channels=hyperparam["num_hidden_dim"],
            final_size=hyperparam["final_size"]
        )

        if hyperparam["is_gpu"]:
            self.feature_extractor = self.feature_extractor.cuda()
        
        self.is_freeze = is_freeze
    
    def forward(self, x):
        x = x.view(-1, self.num_task, self.num_feature) 
        x = self.feature_extractor(x, self.graph_adj)

        if self.is_freeze:
            x = x.detach()

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def create_ind_between_task(all_task_result, num_task):
    # Assuming sample matrix size
    mat_size = all_task_result[0].covariance_matrix.size(1)

    # Block Diagonal Across Batch
    block_sample_independent = torch.sum(torch.stack([
        F.pad(task_sample.covariance_matrix, (
            mat_size*i, mat_size*(num_task-1-i), mat_size*i, mat_size*(num_task-1-i)
        ))
        for task_sample, i in zip(all_task_result, range(num_task))
    ]), dim=0)

    return MultivariateNormal(
        torch.cat([all_task_result[i].mean for i in range(num_task)], dim=1),
        block_sample_independent
    )


def create_non_linear_mtl(
        create_config_funct, train_x_shape, num_tasks, hyperparameters, 
        num_inducing=256, hidden_layer_size=32, num_quad_site=3
    ):

    curr_config = create_config_funct(num_inducing, num_quad_site)
    
    class NonLinearMultiTask(curr_config["class_type"]):
        def __init__(self, inp_feat_size, num_task, hidden_layer_size=3): 
            LayerClass = curr_config["hidden_layer_type"]

            mean_hidden_layer = LayerClass(
                input_dims=inp_feat_size,
                output_dims=hidden_layer_size,
                is_linear=True,
                kernel_type=hyperparameters["kernel"],
                **curr_config["hidden_info"]
            )

            all_hidden_layer = [
                LayerClass(
                    input_dims=inp_feat_size,
                    output_dims=hidden_layer_size,
                    is_linear=True,
                    kernel_type=hyperparameters["kernel"],
                    **curr_config["hidden_info"]
                )
                for _ in range(num_task)
            ]

            if hyperparameters["is_gpu"]:
                all_hidden_layer = [
                    all_hidden_layer[i].cuda()
                    for i in range(num_task)
                ]

            all_hidden_layer = nn.ModuleList(all_hidden_layer)
            self.num_task = num_task


            last_layer = LayerClass(
                input_dims=hidden_layer_size*2,
                output_dims=None,
                is_linear=False,
                kernel_type=hyperparameters["kernel"],
                **curr_config["hidden_info"]
            )
            
            super().__init__(**curr_config["class_init_info"])
            
            self.mean_hidden_layer = mean_hidden_layer
            self.all_hidden_layer = all_hidden_layer
            self.last_layer = last_layer
            self.likelihood = GaussianLikelihood()
        
        def forward(self, inputs):

            all_task_result = []

            for task_i in range(self.num_task):

                inp_task = inputs[:, task_i, :]

                mean_out = self.mean_hidden_layer(inp_task)
                specific_out = self.all_hidden_layer[task_i](inp_task)

                all_task_result.append(
                    self.last_layer(mean_out, specific_out)
                )

            return create_ind_between_task(all_task_result, self.num_task)
    
    return NonLinearMultiTask(train_x_shape, num_tasks, hidden_layer_size)


def create_non_linear_graph_gp(
    create_config_funct, inp_feat_size, num_tasks, hyperparameters, graph_structure,
    num_inducing=256, hidden_layer_size=8, num_quad_site=3):

    curr_config = create_config_funct(num_inducing, num_quad_site)

    class GPDeepGraphModel(curr_config["class_type"]):
        def __init__(self, inp_feat_size, num_task, 
            graph_structure, hidden_layer_size=3): 

            LayerClass = curr_config["hidden_layer_type"]

            self.inp_feat_size = inp_feat_size
            self.num_task = num_task

            common_feat_extractor = LayerClass(
                input_dims=inp_feat_size,
                output_dims=hidden_layer_size,
                is_linear=False,
                kernel_type=hyperparameters["kernel"],
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
                    output_dims=None,
                    is_linear=False,
                    kernel_type=hyperparameters["kernel"],
                    **curr_config["hidden_info"]
                )
                for task in range(num_task)
            ]
            
            if hyperparameters["is_gpu"]:
                ind_extractor = [
                    ind_extractor[i].cuda()
                    for i in range(num_task)
                ]
            
            ind_extractor = nn.ModuleList(ind_extractor)
            
            super().__init__(**curr_config["class_init_info"])
            
            self.num_task = num_task
            self.common_feat_extractor = common_feat_extractor
            self.ind_extractor = ind_extractor
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
            
            return create_ind_between_task(indv_feat_agg, self.num_task)
    
    return GPDeepGraphModel(inp_feat_size, num_tasks, 
        graph_structure, hidden_layer_size)

def create_non_linear_interact(
    create_config_funct, inp_feat_size, num_tasks, hyperparameters,num_inducing=256, hidden_layer_size=8, num_quad_site=3
):

    curr_config = create_config_funct(num_inducing, num_quad_site)

    class GPDeepInteractionModel(curr_config["class_type"]):
        def __init__(self, inp_feat_size, num_task, hidden_layer_size=3): 

            LayerClass = curr_config["hidden_layer_type"]

            self.inp_feat_size = inp_feat_size
            self.num_task = num_task
            
            ind_extractor = [
                LayerClass(
                    input_dims=inp_feat_size,
                    output_dims=hidden_layer_size,
                    is_linear=False,
                    kernel_type=hyperparameters["kernel"],
                    **curr_config["hidden_info"]
                )
                for task in range(num_task)
            ]
            
            if hyperparameters["is_gpu"]:
                ind_extractor = [
                    ind_extractor[i].cuda()
                    for i in range(num_task)
                ]

            ind_extractor = nn.ModuleList(ind_extractor)

            relation = LayerClass(
                input_dims=hidden_layer_size*2,
                output_dims=hidden_layer_size,
                is_linear=False,
                kernel_type=hyperparameters["kernel"],
                **curr_config["hidden_info"]
            )
            
            aggregator = LayerClass(
                input_dims=hidden_layer_size*(self.num_task-1),
                output_dims=None,
                is_linear=False,
                kernel_type=hyperparameters["kernel"],
                **curr_config["hidden_info"]
            )

            super().__init__(**curr_config["class_init_info"])
            
            self.num_task = num_task
            self.ind_extractor = ind_extractor
            self.relation = relation
            self.aggregator = aggregator
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

            return create_ind_between_task(final_aggr, self.num_task)
    
    return GPDeepInteractionModel(
        inp_feat_size, num_tasks, hidden_layer_size
    )
            
