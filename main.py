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
from experiments import algo_dict, gen_experiment, list_dataset
from run_experiments import gen_task_list

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
    
    # exp_setting1 = gen_experiment.create_exp_setting(list_dataset.diff_time_pca_feature, "GPMultiTaskMultiOut")
    # exp_setting2 = gen_experiment.create_exp_setting(list_dataset.diff_time_pca_feature, "IndependentGP")
    # exp_setting3 = gen_experiment.create_exp_setting(list_dataset.diff_time_pca_feature, "GPMultiTaskIndex")
    # exp_setting4 = gen_experiment.create_exp_setting(list_dataset.diff_time, "IIDDataModel")

    # Getting the first one and the actual content
    exp_setting2 = gen_task_list(
        ["GPMultiTaskIndex"], "time", 
        {"copper": CompressMethod(0, "drop"), "aluminium": CompressMethod(0, "drop")}, 
        "copper"
    )[0][1]
    exp_setting2['using_first'] = True
     
    for (k, v) in algo_dict.algorithms_dic.items():
        v[0]["is_verbose"] = True
        v[0]["optim_iter"] = num_train_iter    
    
    if test_type == "f":
        example_plot_all_algo_lag(
            exp_setting2, is_save=False, is_load=False,
            load_path="GP-Multi",
            # load_path="07-19-21-17-29-31-GP-Multi"
        )
    elif test_type == "w":
        print("Mean")
        example_plot_walk_forward(exp_setting2, "Mean",
            is_save=True, is_load=False, is_show=True,
            load_path="Mean"
        )
        
if __name__ == '__main__':
    main()

