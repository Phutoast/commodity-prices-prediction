import torch
from torch_geometric.nn.inits import glorot, zeros, normal
from torch.nn import Parameter

class CustomGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, list_graph=None):
        super(CustomGCN, self).__init__()    

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))
        self.is_graph_batch = None

        # If we are caching the matrix
        if list_graph is not None:
            self.prop_matrix = self.cal_batch_prop(list_graph)
            self.is_graph_batch = len(list_graph) > 1
 
        self.is_cache_graph = list_graph is not None

        glorot(self.weight)
        zeros(self.bias)
    
    def cal_batch_prop(self, batch_adj_matrix):
        num_nodes = batch_adj_matrix.size(-1)
        num_batch = batch_adj_matrix.size(0)

        I = torch.eye(num_nodes).repeat(num_batch, 1, 1)
        if batch_adj_matrix.is_cuda:
            I = I.cuda()
        
        hat_A = batch_adj_matrix + I

        hat_D = torch.sum(hat_A, axis=1).pow(-0.5)
        hat_D[hat_D == float('inf')] = 0
        hat_D = torch.diag_embed(hat_D)

        return torch.bmm(hat_D, torch.bmm(hat_A, hat_D))
     
    def forward(self, input, graph=None):

        if input.dim() == 2:
            input = input.expand(1, -1, -1)

        if graph is not None:
            assert not self.is_cache_graph

            inp_batch_size = input.size(0)

            if graph.size(0) < inp_batch_size:
                assert graph.size(0) == 1
                graph = torch.tile(graph, (inp_batch_size, 1, 1))
                
            prop_matrix = self.cal_batch_prop(graph)
        else:
            prop_matrix = self.prop_matrix
        
        if prop_matrix.size(0) != input.size(0):
            propagate = torch.matmul(prop_matrix, input)
        else:
            propagate = torch.bmm(prop_matrix, input)
        
        out = torch.matmul(propagate, self.weight) + self.bias
        return out



