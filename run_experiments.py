import numpy as np
import random
import torch
import os
import json
import argparse

from utils.others import create_folder, dump_json, load_json, find_sub_string, find_all_metal_names
from utils.data_visualization import plot_latex
from utils.data_structure import DatasetTaskDesc, CompressMethod
from utils.data_preprocessing import load_transform_data, parse_series_time, load_metal_data, parse_series_time
from experiments import algo_dict, gen_experiment, metal_desc

np.random.seed(48)
random.seed(48)
torch.manual_seed(48)
torch.random.manual_seed(48)

np.seterr(invalid='raise')

parser = argparse.ArgumentParser()
parser.add_argument("--test", help="Are we testing", dest='is_test', action='store_true')
parser.add_argument("--train", help="Are we training", dest='is_test', action='store_false')
parser.set_defaults(is_test=True)

parser.add_argument("--not-verbose", dest='is_verbose', action='store_false')
parser.add_argument("--verbose", help="Are we training", dest='is_test', action='store_true')
parser.set_defaults(is_verbose=False)

args = parser.parse_args()

is_test = args.is_test
is_verbose=args.is_verbose
    
multi_task_algo = ["GPMultiTaskMultiOut", "IndependentGP", "GPMultiTaskIndex"]
metric = ["MSE", "CRPS"]

def run_multi_task_gp(save_path, modifier, len_inp=10, len_dataset=794, len_train_show=(274, 130)):
    config = algo_dict.encode_params(
        "gp_multi_task", is_verbose=is_verbose, 
        is_test=is_test, 
        kernel="Composite_1", 
        optim_iter=50,
        len_inp=len_inp
    )
    multi_task_config = {
        "GPMultiTaskMultiOut": config,
        "IndependentGP": algo_dict.encode_params(
            "gp", is_verbose=is_verbose, 
            is_test=is_test, 
            kernel="Composite_1", 
            optim_iter=50,
            len_inp=len_inp
        ),
        "GPMultiTaskIndex": config,
    }
    task = gen_experiment.gen_task_cluster(
        multi_task_algo, "metal", modifier, 
        clus_metal_desc=[metal_desc.metal_names], 
        clus_time_desc=None,
        algo_config=multi_task_config,
        len_dataset=len_dataset, len_train_show=len_train_show
    )
    output = gen_experiment.run_experiments(task, save_path=save_path)
    dump_json(f"{save_path}all_data.json", output) 

    summary = {}

    for algo_name, result in output.items():
        data = {}
        for metric, r in result.items():
            mean_across_task = np.mean(np.array(r), axis=0)[0]
            data[metric] = mean_across_task
        summary[algo_name] = data 
    
    return summary

def run_hyperparam_search(): 
    num_inp = np.arange(2, 14, step=2)
    num_pca = np.arange(2, 8)

    all_results = {
        algo:{m: np.zeros((len(num_inp), len(num_pca))).tolist() for m in metric}
        for algo in multi_task_algo
    }
    all_metal_name = find_all_metal_names()
    
    for i, len_inp in enumerate(num_inp):
        for j, pca in enumerate(num_pca):
            pca_modifier = {
                metal: CompressMethod(int(pca), "pca", info={})
                for metal in all_metal_name
            }
            curr_result = run_multi_task_gp(
                f"save/hyper_search/run_pca_{pca}_feat_{len_inp}/",
                pca_modifier,
                len_inp=int(len_inp)
            )
            for algo in multi_task_algo:
                for met in metric:
                    all_results[algo][met][i][j] = curr_result[algo][met]
    
    dump_json("save/hyper_search/final_result.json", all_results) 
    return all_results 

def general_testing():
    all_metal_name = find_all_metal_names()
    no_modifier = {
        metal: CompressMethod(0, "drop")
        for metal in all_metal_name
    }
    pca_modifier = {
        metal: CompressMethod(3, "pca")
        for metal in all_metal_name
    }
    
    all_algo = ["GPMultiTaskMultiOut", "IndependentGP", "GPMultiTaskIndex", "IIDDataModel", "ARIMAModel"]
    display_name_to_algo = dict(zip(
        ["GPMultiTaskMultiOut", "IndependentGP", "GPMultiTaskIndex", "IIDDataModel", "ARIMAModel"], 
        ["Multi-Task Out", "Independent GP", "Multi-Task Index", "Mean", "ARIMA"],
    ))

    default_config = {
        "GPMultiTaskMultiOut": algo_dict.encode_params(
            "gp_multi_task", is_verbose=is_verbose, 
            is_test=is_test, 
            kernel="Composite_1", 
            optim_iter=50,
            len_inp=10
        ),
        "IndependentGP": algo_dict.encode_params(
            "gp", is_verbose=is_verbose, 
            is_test=is_test, 
            kernel="Composite_1", 
            optim_iter=50,
            len_inp=10
        ),
        "GPMultiTaskIndex": algo_dict.encode_params(
            "gp_multi_task", is_verbose=is_verbose, 
            is_test=is_test, 
            kernel="Composite_1", 
            optim_iter=50,
            len_inp=10
        ),
        "IIDDataModel": algo_dict.encode_params(
            "iid", is_verbose=is_verbose, 
            is_test=is_test, dist="gaussian"
        ),
        "ARIMAModel": algo_dict.encode_params(
            "arima", is_verbose=is_verbose, 
            is_test=is_test, order=(2, 0, 5)
        ),
    }

    def original_test():

        (time_al, time_cu, commodity, time_al_feat, time_cu_feat, commodity_feat) = [
            gen_experiment.gen_task_cluster(
                all_algo, type_task, modifier, 
                clus_metal_desc=metal_desc, 
                clus_time_desc=time_desc,
                algo_config=default_config
            )
            for type_task, modifier, metal_desc, time_desc in [
                ("time", no_modifier, "aluminium", [[22, 44, 66]]),
                ("time", no_modifier, "copper", [[22, 44, 66]]),
                ("metal", no_modifier, [["aluminium", "copper"]], None),
                ("time", pca_modifier, "aluminium", [[22, 44, 66]]),
                ("time", pca_modifier, "copper", [[22, 44, 66]]),
                ("metal", pca_modifier, [["aluminium", "copper"]], None),
            ]
        ]
        
        task_train = [time_al, commodity, time_al_feat, commodity_feat]
        task_names = ["Price", "Metal", "Price_Feat", "Metal_Feat"]

        def how_to_plot(super_task):
            plot_latex(
                names=[all_algo, all_algo],
                results=[super_task["Price"], super_task["Metal"]],
                multi_task_name=[["Date (22)", "Date (44)", "Date (66)"], ["Aluminium", "Copper"]],
                display_name_to_algo=display_name_to_algo
            )
            print()
            print()
            plot_latex(
                names=[all_algo, all_algo],
                results=[super_task["Price_Feat"], super_task["Metal_Feat"]],
                multi_task_name=[["Date (22)", "Date (44)", "Date (66)"], ["Aluminium", "Copper"]],
                display_name_to_algo=display_name_to_algo
            )

        return task_train, task_names, how_to_plot
    
    def test_all_metal():
        task = gen_experiment.gen_task_cluster(
            all_algo, "metal", pca_modifier, 
            clus_metal_desc=[metal_desc.metal_names], 
            clus_time_desc=None,
            algo_config=default_config
        )

        def how_to_plot(super_task):
            plot_latex(
                names=[all_algo],
                results=[super_task["All Meta"]],
                multi_task_name=[metal_desc.metal_names],
                display_name_to_algo=display_name_to_algo
            )

        return [task], ["All Metal"], how_to_plot

    task_train, task_names, how_to_plot = test_all_metal()

    super_task = {}
    for task, name in zip(task_train, task_names):
        all_out = gen_experiment.run_experiments(task)
        super_task.update({name: all_out})
    
    dump_json("save/all_data.json", super_task) 
    # super_task = load_json("save/all_data.json")

    how_to_plot(super_task)
    
def run_multi_task_range(all_results, start_ind, end_ind, save_name, compress_type="id"):
    common = CompressMethod(0, compress_type, info={"range_index": (start_ind, end_ind)})
    data_modi = {"copper": common, "aluminium": common} 
    curr_result = run_multi_task_gp(save_name, data_modi, len_dataset=-1, len_train_show=(100, 32 + 20))
        
    for algo in multi_task_algo:
        for met in metric:
            all_results[algo][met].append(curr_result[algo][met])

def run_years_prediction():
    _, data_al = load_transform_data("aluminium", 22)

    all_date = data_al["Date"].to_list()
    years = [str(2005 + i) for i in range(17)]
    
    all_results = {
        algo:{m: [] for m in metric}
        for algo in multi_task_algo
    }

    for i in range(len(years)-1):
        start_ind = find_sub_string(all_date, f"{years[i]}-05")
        end_ind = find_sub_string(all_date, f"{years[i+1]}-05")
        run_multi_task_range(all_results, start_ind, end_ind, f"save/test-mlt-gp/test-year-{years[i]}-{years[i+1]}/") 
    
    dump_json("save/test-mlt-gp/all_result.json", all_results)

def run_years_prediction_feature():
    data = load_metal_data("aluminium")
    data = data.dropna()

    all_date = data["Date"].to_list()
    years = [str(2018 + i) for i in range(4)]
    
    all_results = {
        algo:{m: [] for m in metric}
        for algo in multi_task_algo
    }

    for i in range(len(years)-1):
        start_ind = find_sub_string(all_date, f"{years[i]}-05")
        end_ind = find_sub_string(all_date, f"{years[i+1]}-05")
        run_multi_task_range(
            all_results, start_ind, 
            end_ind, 
            f"save/test-mlt-gp-feature/test-year-{years[i]}-{years[i+1]}-feature/", 
            compress_type="pca"
        ) 
    
    dump_json("save/test-mlt-gp-feature/all_result_feature.json", all_results)

def run_window_prediction():
    """
    This is going to be similar to run_years_prediction
    """
    _, data_al = load_transform_data("aluminium", 22)
        
    window_size = 260

    # 3 months
    skip_size = 66
    
    all_results = {
        algo:{m: [] for m in metric}
        for algo in multi_task_algo
    }

    total_len = (len(data_al)-window_size) // skip_size
    for i in range(total_len):
        start_ind = i*skip_size
        end_ind = start_ind+window_size
        run_multi_task_range(
            all_results, start_ind, 
            end_ind, f"save/test-mlt-gp-feat/test-window-3-mo-{i}/", 
            compress_type="pca"
        )
    
    dump_json("save/test-mlt-gp-feat/all_result_window.json", all_results)

def run_window_prediction_feature():
    data = load_metal_data("aluminium")
    data = data.dropna()
        
    window_size = 260

    # half months
    skip_size = 11
    
    all_results = {
        algo:{m: [] for m in metric}
        for algo in multi_task_algo
    }

    total_len = (len(data)-window_size) // skip_size
    for i in range(total_len):
        start_ind = i*skip_size
        end_ind = start_ind+window_size
        run_multi_task_range(
            all_results, start_ind, 
            end_ind, f"save/test-mlt-gp-feat/test-window-half-mo-{i}/", 
            compress_type="pca"
        ) 

    dump_json("save/test-mlt-gp-feat/all_result_window.json", all_results)


def main():
    create_folder("save")
    run_hyperparam_search()
    # run_multi_task_gp("save/test")
    # run_years_prediction()
    # run_window_prediction()
    # run_years_prediction_feature()
    # run_window_prediction_feature()
    # general_testing()


if __name__ == '__main__':
    main()

