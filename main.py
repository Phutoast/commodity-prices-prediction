import numpy as np
import random
import argparse
import torch

from examples.simple_example import example_plot_all_algo_lag, example_plot_walk_forward
from utils.others import create_folder

from models.ind_multi_model import IndependentMultiModel
from models.GP_multi_out import GPMultiTaskMultiOut
from models.GP_multi_index import GPMultiTaskIndex

from utils.data_structure import DatasetTaskDesc, CompressMethod
from utils.data_visualization import plot_latex
from utils.data_preprocessing import GlobalModifier
from experiments import algo_dict

import warnings
# warnings.filterwarnings("ignore")

np.random.seed(48)
random.seed(48)
torch.manual_seed(48)
torch.random.manual_seed(48)

np.seterr(invalid='raise')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-type", help="Type of Testing (w, f)", type=str)
    parser.add_argument("--iter", help="Number of Training Iteration for GP", type=int)
    args = parser.parse_args()

    test_type = args.test_type
    num_train_iter = args.iter
    create_folder("save")

    length_dataset = 795

    dataset_test = [
        DatasetTaskDesc(
            inp_metal_list=["aluminium"],
            metal_modifier=[
                CompressMethod(3, "pca"),
            ],
            use_feature=["Date"] + [f"aluminium.Feature{i+1}" for i in range(3)],
            use_feat_tran_lag=None,
            out_feature="aluminium.Price",
            out_feat_tran_lag=(22, 0, "id"),
            len_dataset=795,
        ),
        DatasetTaskDesc(
            inp_metal_list=["aluminium"],
            use_feature=["Date"],
            metal_modifier=[
                CompressMethod(3, "pca")
            ],
            use_feat_tran_lag=None,
            out_feature="aluminium.Price",
            out_feat_tran_lag=(66, 0, "id"),
            len_dataset=795,
        )
    ]
    
    exp_setting2 = {
        "task": {
            "sub_model": ["GP-Test", "GP-Test"],
            "dataset": all_dataset,
            "len_pred_show": 130,
            "len_train_show": (275, 130)
        }, "algo": "IndependentMultiModel", 
        "using_first": True
    }
    
    exp_setting3 = {
        "task": {
            "sub_model": ["GP-Multi-Task", "GP-Multi-Task"],
            "dataset": all_dataset,
            "len_pred_show": 130,
            "len_train_show": (275, 130)
        }, "algo": "GPMultiTaskIndex", 
        "using_first": False
    }
    
    exp_setting4 = {
        "task": {
            "sub_model": ["Mean", "Mean"],
            "dataset": all_dataset,
            # Only used for plot all_algo_lag
            "len_pred_show": 130,
            "len_train_show": (275, 130)
        }, "algo": "IndependentMultiModel", 
        # There is a problem with the display when not using first of the data.....
        "using_first": False
    }
       
    for (k, v) in algo_dict.algorithms_dic.items():
        v[0]["is_verbose"] = True
        v[0]["optim_iter"] = num_train_iter    
    
    if test_type == "f":
        example_plot_all_algo_lag(
            exp_setting4, is_save=False, is_load=False,
            load_path="GP-Multi",
            # load_path="07-19-21-17-29-31-GP-Multi"
        )
    elif test_type == "w":
        print("Multi-Task Out")
        example_plot_walk_forward(None, "Multi-GP-Out",
            is_save=False, is_load=True, is_show=True,
            load_path="07-19-21-17-30-06-Multi-GP-Out", 
        )
        
        # print("Multi-Task Out")
        # example_plot_walk_forward(exp_setting1, "Multi-GP-Out",
        #     is_save=True, is_load=False, is_show=True,
        #     load_path="Multi-GP", size_train=280, size_test=190
        # )
        # print("Independent GP")
        # example_plot_walk_forward(exp_setting2, "Ind-GP",
        #     is_save=True, is_load=False, is_show=True,
        #     load_path="Independent-GP", size_train=290, size_test=190
        # )
        # print("Multi-Task Index")
        # example_plot_walk_forward(exp_setting3, "Multi-GP-Index",
        #     is_save=True, is_load=False, is_show=True,
        #     load_path="Index-GP", size_train=290, size_test=190
        # )
        # print("Mean")
        # example_plot_walk_forward(exp_setting4, "Mean",
        #     is_save=True, is_load=False, is_show=True,
        #     load_path="Mean", size_train=190, size_test=135
        # )

        # print("Multi-Task Out")
        # output1 = example_plot_walk_forward(exp_setting1, "Multi-GP-Out",
        #     is_save=False, is_load=True, is_show=False,
        #     load_path="07-18-21-21-54-55-Multi-GP-Out"
        # )
        # print("Independent GP")
        # output2 = example_plot_walk_forward(exp_setting2, "Ind-GP",
        #     is_save=False, is_load=True, is_show=False,
        #     load_path="07-18-21-21-55-18-Ind-GP"
        # )
        # print("Multi-Task Index")
        # output3 = example_plot_walk_forward(exp_setting3, "Multi-GP-Index",
        #     is_save=False, is_load=True, is_show=False,
        #     load_path="07-18-21-21-55-29-Multi-GP-Index"
        # )
        # print("Mean")
        # output4 = example_plot_walk_forward(exp_setting4, "Mean",
        #     is_save=False, is_load=True, is_show=False,
        #     load_path="07-18-21-21-56-16-Mean"
        # )

        # # --------------------------------------------------------

        # print("Multi-Task Out")
        # output5 = example_plot_walk_forward(exp_setting1, "Multi-GP-Out-Metal",
        #     is_save=False, is_load=True, is_show=False,
        #     load_path="07-19-21-08-24-23-Multi-GP-Out-Metal"
        # )
        # print("Independent GP")
        # output6 = example_plot_walk_forward(exp_setting2, "Ind-GP-Metal",
        #     is_save=False, is_load=True, is_show=False,
        #     load_path="07-19-21-08-24-59-Ind-GP-Metal"
        # )
        # print("Multi-Task Index")
        # output7 = example_plot_walk_forward(exp_setting3, "Multi-GP-Index-Metal",
        #     is_save=False, is_load=True, is_show=False,
        #     load_path="07-19-21-08-25-14-Multi-GP-Index-Metal"
        # )
        # print("Mean")
        # output8 = example_plot_walk_forward(exp_setting4, "Mean-Metal",
        #     is_save=False, is_load=True, is_show=False,
        #     load_path="07-19-21-08-26-45-Mean-Metal"
        # )
        
        # # --------------------------------------------------------

        plot_latex(
            names=[["Multi-Task Out", "Independent GP", "Multi-Task Index", "Mean"]]*2,
            results=[[output1, output2, output3, output4], [output5, output6, output7, output8]],
            multi_task_name=[["Date (22)", "Date (44)"], ["Aluminium", "Copper"]],
        )

    

if __name__ == '__main__':
    main()

