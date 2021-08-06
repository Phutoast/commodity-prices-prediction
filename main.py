import numpy as np
import random
import argparse
import torch

from examples.simple_example import example_plot_all_algo_lag, example_plot_walk_forward
from utils.others import create_folder, find_all_metal_names

from utils.data_structure import CompressMethod
from experiments import algo_dict, list_dataset
from run_experiments import gen_task_list
from utils.data_visualization import plot_hyperparam_search
from utils.data_preprocessing import GlobalModifier, load_metal_data, save_date_common
from utils import explore_data
from utils.data_structure import DatasetTaskDesc

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

    length_dataset = 794
    is_test = False
    multi_task_gp_config = algo_dict.encode_params(
        "gp_multi_task", is_verbose=True, 
        is_test=is_test, 
        kernel="Composite_1", 
        optim_iter=num_train_iter,
        len_inp=10
    )

    algo_config = {
        "IIDDataModel": algo_dict.encode_params(
            "iid", is_verbose=False, 
            is_test=is_test, dist="gaussian"
        ), 
        "GPMultiTaskMultiOut": multi_task_gp_config,
        "GPMultiTaskIndex": multi_task_gp_config,
        "IndependentGP": algo_dict.encode_params(
            "gp", is_verbose=False, 
            is_test=is_test, 
            kernel="Composite_1", 
            optim_iter=num_train_iter,
            len_inp=10
        ),
        "ARIMAModel": algo_dict.encode_params(
            "arima", is_verbose=False, 
            is_test=is_test, order=(2, 0, 5)
        ),
    }

    common_compress = CompressMethod(
        # 0, "id", info={"range_index": (0, 260)}
        3, "pca"
    )

    all_modifiers = {
        metal: common_compress
        for metal in find_all_metal_names("data")
    }

    # Getting the first one and the actual content
    # exp_setting2 = gen_task_list(
    #     ["GPMultiTaskIndex"], "time", all_modifiers, 
    #     "copper", algo_config, len_dataset=-1, len_train_show=(100, 32 + 20)
    # )
    # exp_setting2 = gen_task_list(
    #     ["ARIMAModel"], "metal", all_modifiers, 
    #     ["copper", "lldpe"], algo_config, len_dataset=-1, len_train_show=(100, 32 + 20)
    # )
    
    # name, exp_setting2 = exp_setting2[0]

    # Expected Interface
    # gen_task_list(
    #     ["GPMultiTaskIndex", "IndependentGP"], "metal", all_modifiers,
    #     [["copper", "lldpe"], ["carbon"]], algo_config, len_dataset=-1, len_train_show=(100, 32 + 20)
    # )
    
    # gen_task_list(
    #     ["GPMultiTaskIndex"], "time", all_modifiers,
    #     "copper", algo_config, len_dataset=-1, len_train_show=(100, 32 + 20)
    # )


    exp_setting2 = {
        "task": {
            "sub_model": [
                ['v-gp_multi_task-Composite_1', 'v-gp_multi_task-Composite_1'], ['arima-2,0,5']
            ],
            "dataset": [
                [
                    DatasetTaskDesc(
                        inp_metal_list=["copper"],
                        out_feature="copper.Price",
                        out_feat_tran_lag=(22, 0, 'id'),
                        is_drop_nan= False,
                        len_dataset= -1, 
                        metal_modifier=[
                            CompressMethod(compress_dim=3, method='pca', info={})
                        ],
                        use_feature=['Date', 'copper.Feature1', 'copper.Feature2', 'copper.Feature3'],
                        use_feat_tran_lag=[None, None, None, None]
                    ),
                    DatasetTaskDesc(
                        inp_metal_list=["lldpe"],
                        out_feature="lldpe.Price",
                        out_feat_tran_lag=(22, 0, 'id'),
                        is_drop_nan= False,
                        len_dataset= -1, 
                        metal_modifier=[
                            CompressMethod(compress_dim=3, method='pca', info={})
                        ],
                        use_feature=['Date', 'lldpe.Feature1', 'lldpe.Feature2', 'lldpe.Feature3'],
                        use_feat_tran_lag=[None, None, None, None]
                    ),
                ],
                [
                    DatasetTaskDesc(
                        inp_metal_list=["carbon"],
                        out_feature="carbon.Price",
                        out_feat_tran_lag=(22, 0, 'id'),
                        is_drop_nan= False,
                        len_dataset= -1, 
                        metal_modifier=[
                            CompressMethod(compress_dim=3, method='pca', info={})
                        ],
                        use_feature=['Date', 'carbon.Feature1', 'carbon.Feature2', 'carbon.Feature3'],
                        use_feat_tran_lag=[None, None, None, None]
                    ),
                ]
            ],
            "len_pred_show": 130,
            "len_train_show":(100, 52)
        },
        "algo": ["GPMultiTaskIndex", "IndependentMultiModel"],
        "using_first": [False, False]
    }

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
    # print("HERE")
    # save_date_common("raw_data", "data")

    # explore_data.plot_window_unrelated()
    # explore_data.plot_window_related()
    # explore_data.plot_correlation_all()
    # explore_data.plot_years_correlation("nickel", "copper")
    # explore_data.plot_years_correlation("natgas", "copper")
    # explore_data.plot_cf_and_acf()
    # explore_data.distance_between_time_series()
    # explore_data.clustering_dataset(num_cluster=4, is_verbose=True)
    # explore_data.clustering_dataset()

    main()

