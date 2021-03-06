from utils.data_structure import Hyperparameters

from models.ARIMA import ARIMAModel
from models.Mean import IIDDataModel
from models.GP import IndependentGP

from models.ind_multi_model import IndependentMultiModel
from models.GP_multi_out import GPMultiTaskMultiOut
from models.GP_multi_index import GPMultiTaskIndex
from models.Deep_GP import DeepGPMultiOut
from models.DSPP_GP import DSPPMultiOut
from models.Sparse_GP_index import SparseGPIndex
from models.GP_Graph import SparseMaternGraphGP
from models.Deep_Graph_GP import DeepGraphMultiOutputGP
from models.Deep_InfoMax_GP import DeepGraphInfoMaxMultiOutputGP
from models.Nonlinear_MT_GP import NonlinearMultiTaskGP
from models.Nonlinear_MT_GSPP import NonlinearMultiTaskGSPP
from models.DeepGP_graph_prop import DeepGPGraphPropagate
from models.DeepGP_graph_interact import DeepGPGraphInteract
from models.DSPP_graph_interact import DSPPGraphInteract
from models.DSPP_graph_prop import DSPPGraphPropagate

from models.full_AR_model import FullARModel

import numpy as np
import copy

from gpytorch import kernels
import torch

kernel_name = {
    "RBF": kernels.ScaleKernel(kernels.RBFKernel()),
    "Matern": kernels.ScaleKernel(kernels.MaternKernel()),
    "Composite_1": kernels.ScaleKernel(kernels.RBFKernel()) + kernels.ScaleKernel(kernels.PeriodicKernel(power=2)),
    "Composite_2": kernels.ScaleKernel(kernels.MaternKernel()) + kernels.ScaleKernel(kernels.PeriodicKernel(power=2)),
    "Composite_3": kernels.ScaleKernel(kernels.RBFKernel()) + kernels.ScaleKernel(kernels.PolynomialKernel(power=2)),
    "Composite_4": kernels.ScaleKernel(kernels.MaternKernel()) + kernels.ScaleKernel(kernels.PolynomialKernel(power=2)),
}

# For independent Learner Only
class_name = {
    "ARIMAModel": ARIMAModel,
    "IIDDataModel": IIDDataModel,
    "IndependentGP": IndependentGP,
}
multi_task_algo = {
    "IndependentMultiModel": IndependentMultiModel,
    "GPMultiTaskMultiOut": GPMultiTaskMultiOut,
    "GPMultiTaskIndex": GPMultiTaskIndex,
    "DeepGPMultiOut": DeepGPMultiOut,
    "DSPPMultiOut": DSPPMultiOut, 
    "SparseGPIndex": SparseGPIndex,
    "SparseMaternGraphGP": SparseMaternGraphGP,
    "DeepGraphMultiOutputGP": DeepGraphMultiOutputGP,
    "DeepGraphInfoMaxMultiOutputGP": DeepGraphInfoMaxMultiOutputGP,
    "NonlinearMultiTaskGP": NonlinearMultiTaskGP,
    "NonlinearMultiTaskGSPP": NonlinearMultiTaskGSPP,
    "DeepGPGraphPropagate": DeepGPGraphPropagate,
    "DeepGPGraphInteract": DeepGPGraphInteract,
    "DSPPGraphInteract": DSPPGraphInteract,
    "DSPPGraphPropagate": DSPPGraphPropagate
}

class_name_to_display = {
    "ARIMAModel": "ARIMA",
    "IIDDataModel": "IID",
    "IndependentGP": "Independent GP",
    "IndependentMultiModel": "Independent Multi-Model",
    "GPMultiTaskMultiOut": "Multi-Task GP Output",
    "GPMultiTaskIndex": "Multi-Task GP Index",
    "DeepGPMultiOut": "2 Layer Deep GP",
    "DSPPMultiOut": "2 Layer DSPP",
    "SparseGPIndex": "Sparse Multi-Task GP Index",
    "SparseMaternGraphGP": "Sparse Matern Graph GP",
    "DeepGraphMultiOutputGP": "Deep Graph GP",
    "DeepGraphInfoMaxMultiOutputGP": "Deep Graph InfoMax GP",
    "NonlinearMultiTaskGP": "Non-Linear Multi-Task GP",
    "NonlinearMultiTaskGSPP": "Non-Linear Multi-Task DSPP",
    "DeepGPGraphPropagate": "Deep GP Graph Propagate",
    "DeepGPGraphInteract": "Deep GP Relation",
    "DSPPGraphInteract": "DSPP Graph Relation",
    "DSPPGraphPropagate": "DSPP Graph Propagate",
}

algo_is_using_first = {k: v.expect_using_first for k, v in multi_task_algo.items()}
using_first_algo = [k for k, v in algo_is_using_first.items() if v]
using_out_only = [k for k, v in class_name.items() if issubclass(v, FullARModel)]

class AlgoDict(object):

    def __init__(self):
        self.base_GP = Hyperparameters(
            len_inp=10, 
            len_out=1, 
            lr=0.1,
            optim_iter=100,
            is_time_only=False,
            is_date=False, 
            is_past_label=True,
            kernel=None,
            num_hidden_dim=32,
            final_size=16,
            graph_path="exp_result/graph_result/kendell_test_graph.npy"
        )

        self.base_ARIMA = Hyperparameters(
            len_inp=0, 
            len_out=1, 
            is_date=False, 
            order=(2, 0, 5), 
            is_full_pred=True
        )

        self.base_iid = Hyperparameters(
            len_inp=0, 
            len_out=1, 
            is_date=False, 
            dist="Gaussian",
            is_verbose=False,
            is_full_pred=True
        )
    
    def step_modi_GP(self, list_new_val):
        curr_setting = copy.deepcopy(self.base_GP)
        key_change = [
            ("kernel", str), ("optim_iter", int), 
            ("len_inp", int), ("lr", float), 
            ("graph_path", str)
        ]

        for (key, convert), new_val in zip(key_change, list_new_val):
            curr_setting[key] = convert(new_val)
        
        return curr_setting
    
    def cheat_modify(self, setting, desc):
        setting["len_out"] = int(desc[2])
        setting["is_full_pred"] = False

    def modify_return(self, return_val):
        return_val[0]["is_verbose"] = self.is_verbose
        return_val[0]["is_gpu"] = torch.cuda.is_available()
        return return_val

    def __getitem__(self, item):

        self.is_verbose = False
        if "v-" == item[:2]:
            self.is_verbose = True
            item = item[2:]

        desc = item.split("-")
        algo = desc[0].lower()

        if "gp" in algo:
            curr_setting = self.step_modi_GP(desc[1:])
            algo_multi_name = ["gp_multi_task"]
            if algo in algo_multi_name:
                return self.modify_return([curr_setting, None])
            else:
                return self.modify_return([curr_setting, IndependentGP])
        
        elif "arima" in algo:
            curr_setting = copy.deepcopy(self.base_ARIMA)
            if len(desc[1:]) == 0:
                return self.modify_return([curr_setting, ARIMAModel])

            curr_setting["order"] = tuple(int(o) for o in desc[1].split(","))
            
            if algo == "arima_cheat":
                self.cheat_modify(curr_setting, desc)
            
            return self.modify_return([curr_setting, ARIMAModel])
        
        elif "iid" in algo:
            curr_setting = copy.deepcopy(self.base_iid)
            if len(desc[1:]) == 0:
                return self.modify_return([curr_setting, IIDDataModel])
            
            curr_setting["dist"] = str(desc[1])
            if algo == "iid_cheat":
                self.cheat_modify(curr_setting, desc)
            
            return self.modify_return([curr_setting, IIDDataModel])
        else:
            raise ValueError("NO Algorithm Found !!!")

algorithms_dic = AlgoDict()

def encode_params(algo, is_verbose, is_test, **kwargs):
    text = "v-" + algo if is_verbose else algo


    if "-" in algo:
        raise ValueError("Shouldn't have - in the name")

    def encode_text(value):
        final = ""
        for v in value:
            if v is None:
                break
            final += f"-{v}"
        
        return final

    if "gp" in algo.lower():

        all_keys = ["kernel", "optim_iter", "len_inp", "lr", "graph_path"]
        assert all(k in all_keys for k in kwargs.keys())
        
        list_text = [kwargs["kernel"]]
        optim_iter = kwargs.get("optim_iter")
        if is_test:
            optim_iter = 1
        list_text.append(optim_iter)
        for modi in all_keys[2:]:
            list_text.append(kwargs.get(modi))
        
    elif "iid" in algo.lower():
        all_keys = ["dist", "len_out"]
        assert all(k in all_keys for k in kwargs.keys())

        list_text = [kwargs["dist"], kwargs.get("len_out")]

    elif "arima" in algo.lower():
        all_keys = ["order", "len_out"]
        assert all(k in all_keys for k in kwargs.keys())

        order_text = ",".join([str(o) for o in kwargs["order"]])
        list_text = [order_text, kwargs.get("len_out")]

    return text + encode_text(list_text)


# print(algorithms_dic["v-GP-Composite_1-1-5-0.01"])
# print(encode_params("gp", True, True, kernel="Composite_1", optim_iter="10"))
# print(algorithms_dic["v-GP_Multi_task-Composite_1-1-5-0.01"])
# print(algorithms_dic["ARIMA_Cheat-2,0,5-100"])
# print(encode_params("iid", True, True, dist="Gamma", len_out=))
# print(algorithms_dic["iid-Gaussian-10"])

# test = encode_params("arima_cheat", is_verbose=True, is_test=True, order=(0,1,3), len_out=10)
# print(algorithms_dic[test])
# assert False