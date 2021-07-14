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

import warnings
warnings.filterwarnings("ignore")

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
                    inp_metal_list=["aluminium", "copper"],
                    use_feature=["Date", "copper.Price"],
                    use_feat_tran_lag=[None, (22, 1, lambda x: x)],
                    out_feature="aluminium.Price",
                    out_feat_tran_lag=(22, 1, lambda x: x),
                ),
                DatasetTaskDesc(
                    inp_metal_list=["aluminium", "copper"],
                    use_feature=["Date"],
                    use_feat_tran_lag=None,
                    out_feature="copper.Price",
                    out_feat_tran_lag=(22, 1, lambda x: x),
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
                    out_feat_tran_lag=(22, 1, lambda x: x),
                ),
                DatasetTaskDesc(
                    inp_metal_list=["copper"],
                    use_feature=["Date"],
                    use_feat_tran_lag=None,
                    out_feature="copper.Price",
                    out_feat_tran_lag=(22, 1, lambda x: x),
                ),
            ],
            # Only used for plot all_algo_lag
            "len_pred_show": [100, 100]
        }, "algo": IndependentMultiModel, 
        "using_first": False
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
                    out_feat_tran_lag=(22, 1, lambda x: x),
                ),
                DatasetTaskDesc(
                    inp_metal_list=["copper"],
                    use_feature=["Date"],
                    use_feat_tran_lag=None,
                    out_feature="copper.Price",
                    out_feat_tran_lag=(22, 1, lambda x: x),
                ),
            ],
            # Only used for plot all_algo_lag
            "len_pred_show": [100, 100]
        }, "algo": GPMultiTaskIndex, 
        "using_first": False
    }

    if test_type == "f":
        example_plot_all_algo_lag(
            exp_setting3, is_save=True, is_load=False,
            load_path="GP-Multi"
            # load_path="07-14-21-19-38-29-GP-Multi"
        )
    elif test_type == "w":
        example_plot_walk_forward(exp_setting1, "Multi-GP",
            is_save=True, is_load=False,
            load_path="Multi-GP"
        )
        example_plot_walk_forward(exp_setting2, "Independent-GP",
            is_save=True, is_load=False,
            load_path="Independent-GP"
        )


if __name__ == '__main__':
    main()


