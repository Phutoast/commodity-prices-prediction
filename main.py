import numpy as np
import random
import argparse
import torch

from examples.simple_example import example_plot_all_algo_lag, example_plot_walk_forward
from utils.others import create_folder

from models.ind_multi_model import IndependentMultiModel
from models.train_model import BaseTrainMultiTask
from models.GP_multi_out import GPMultiTaskMultiOut
from models.GP_multi_index import GPMultiTaskIndex

from utils.data_structure import DatasetTaskDesc
from experiments import algo_dict

import warnings
# warnings.filterwarnings("ignore")

np.random.seed(48)
random.seed(48)
torch.manual_seed(48)
torch.random.manual_seed(48)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-type", help="Type of ", type=str)
    args = parser.parse_args()

    test_type = args.test_type
    create_folder("save")
    
    exp_setting1 = {
        "task": {
            "sub_model": ["GP-Multi-Task", "GP-Multi-Task"],
            "dataset": [
                DatasetTaskDesc(
                    inp_metal_list=["aluminium"],
                    use_feature=["Date"],
                    use_feat_tran_lag=None,
                    out_feature="aluminium.Price",
                    out_feat_tran_lag=(22, 0, lambda x: x),
                ),
                DatasetTaskDesc(
                    inp_metal_list=["aluminium"],
                    use_feature=["Date"],
                    use_feat_tran_lag=None,
                    out_feature="aluminium.Price",
                    out_feat_tran_lag=(44, 0, lambda x: x),
                ),
            ],
            # Only used for plot all_algo_lag
            "len_pred_show": [100, 100]
        }, "algo": GPMultiTaskMultiOut, 
        "using_first": True
    }
    
    exp_setting2 = {
        "task": {
            "sub_model": ["GP-Test", "GP-Test"],
            "dataset": [
                DatasetTaskDesc(
                    inp_metal_list=["aluminium"],
                    use_feature=["Date"],
                    use_feat_tran_lag=None,
                    out_feature="aluminium.Price",
                    out_feat_tran_lag=(22, 0, lambda x: x),
                ),
                DatasetTaskDesc(
                    inp_metal_list=["aluminium"],
                    use_feature=["Date"],
                    use_feat_tran_lag=None,
                    out_feature="aluminium.Price",
                    out_feat_tran_lag=(44, 0, lambda x: x),
                ),
            ],
            # Only used for plot all_algo_lag
            "len_pred_show": [100, 100]
        }, "algo": IndependentMultiModel, 
        "using_first": True
    }
    
    exp_setting3 = {
        "task": {
            "sub_model": ["GP-Multi-Task", "GP-Multi-Task"],
            "dataset": [
                DatasetTaskDesc(
                    inp_metal_list=["aluminium"],
                    use_feature=["Date"],
                    use_feat_tran_lag=None,
                    out_feature="aluminium.Price",
                    out_feat_tran_lag=(22, 0, lambda x: x),
                ),
                DatasetTaskDesc(
                    inp_metal_list=["aluminium"],
                    use_feature=["Date"],
                    use_feat_tran_lag=None,
                    out_feature="aluminium.Price",
                    out_feat_tran_lag=(44, 0, lambda x: x),
                ),
            ],
            # Only used for plot all_algo_lag
            "len_pred_show": [100, 100]
        }, "algo": GPMultiTaskIndex, 
        "using_first": False
    }
    
    exp_setting4 = {
        "task": {
            "sub_model": ["Mean", "Mean"],
            "dataset": [
                DatasetTaskDesc(
                    inp_metal_list=["aluminium"],
                    use_feature=["Date"],
                    use_feat_tran_lag=None,
                    out_feature="aluminium.Price",
                    out_feat_tran_lag=(22, 0, lambda x: x),
                ),
                DatasetTaskDesc(
                    inp_metal_list=["aluminium"],
                    use_feature=["Date"],
                    use_feat_tran_lag=None,
                    out_feature="aluminium.Price",
                    out_feat_tran_lag=(44, 0, lambda x: x),
                ),
            ],
            # Only used for plot all_algo_lag
            "len_pred_show": [100, 100]
        }, "algo": IndependentMultiModel, 
        # There is a problem with the display when not using first of the data.....
        "using_first": False
    }

    if test_type == "f":
        example_plot_all_algo_lag(
            exp_setting1, is_save=True, is_load=False,
            load_path="GP-Multi"
            # load_path="07-14-21-19-38-29-GP-Multi"
        )
    elif test_type == "w":
        
        for (_, v) in algo_dict.algorithms_dic.items():
            v[0]["is_verbose"] = False
        
        # print("Multi-Task Out")
        # example_plot_walk_forward(exp_setting1, "Multi-GP-Out",
        #     is_save=True, is_load=False,
        #     load_path="Multi-GP"
        # )
        # print("Independent GP")
        # example_plot_walk_forward(exp_setting2, "Ind-GP",
        #     is_save=True, is_load=False,
        #     load_path="Multi-GP"
        # )
        # print("Multi-Task Index")
        # example_plot_walk_forward(exp_setting3, "Multi-GP-Index",
        #     is_save=True, is_load=False,
        #     load_path="Multi-GP"
        # )
        # print("Mean")
        # example_plot_walk_forward(exp_setting4, "Mean",
        #     is_save=True, is_load=False,
        #     load_path="Multi-GP"
        # )
        example_plot_walk_forward(
            exp_setting3, "Multi-GP-Index", is_save=False, is_load=True, is_show=False,
            # load_path="GP-Ind"
            load_path="07-16-21-22-46-23-Multi-GP-Index"
        )
        example_plot_walk_forward(
            exp_setting2, "Ind-GP", is_save=False, is_load=True, is_show=False,
            # load_path="GP-Ind"
            load_path="07-16-21-22-46-04-Ind-GP"
        )
        example_plot_walk_forward(
            exp_setting1, "Multi-GP-Out", is_save=False, is_load=True, is_show=False,
            # load_path="GP-Ind"
            load_path="07-16-21-22-45-34-Multi-GP-Out"
        )
    

if __name__ == '__main__':
    main()


