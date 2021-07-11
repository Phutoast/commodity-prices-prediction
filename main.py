import numpy as np
import random
import argparse
import torch

from examples.simple_example import example_plot_all_algo_lag, example_plot_walk_forward
from utils.others import create_folder

from models.ind_multi_model import IndependentMultiModel
from models.train_model import BaseTrainMultiTask
from models.GP_multi_task import GPMultiTask

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
    
    exp_setting_2 = {
        "task": [
            ["Mean", "aluminium", 22, 1, 100],
            ["Mean", "copper", 22, 9, 100],
        ], "algo": IndependentMultiModel,
        "using_first": True
    }
    
    exp_setting = {
        "task": {
            "sub_model": ["GP-Multi-Task", "GP-Multi-Task"],
            "dataset": [
                DatasetTaskDesc(
                    inp_metal_list=["aluminium"],
                    use_feature=["Date"],
                    use_feat_tran_lag=None,
                    out_feature="aluminium.Price",
                    out_feat_tran_lag=(22, 9, lambda x: x),
                ),
                DatasetTaskDesc(
                    inp_metal_list=["aluminium"],
                    use_feature=["Date"],
                    use_feat_tran_lag=None,
                    out_feature="aluminium.Price",
                    out_feat_tran_lag=(22, 100, lambda x: x),
                ),
            ],
            # Only used for plot all_algo_lag
            "len_pred_show": [100, 100]
        }, "algo": GPMultiTask, 
        "using_first": True
    }

    if test_type == "f":
        example_plot_all_algo_lag(
            exp_setting, is_save=True, is_load=False,
            # load_path="GP-Multi"
        )
    elif test_type == "w":
        example_plot_walk_forward(exp_setting, "Multi-GP",
            is_save=True, is_load=False,
            load_path="Multi-GP"
        )
        # example_plot_walk_forward(exp_setting_2, "Independent-GP",
        #     is_save=True, is_load=False,
        #     load_path="Independent-GP"
        # )


if __name__ == '__main__':
    main()
