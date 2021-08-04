import numpy as np
import random
import torch
import os
import json

from utils.others import create_folder, dump_json, load_json, find_sub_string, find_all_metal_names
from utils.data_visualization import plot_latex
from utils.data_structure import DatasetTaskDesc, CompressMethod
from utils.data_preprocessing import load_transform_data, parse_series_time, load_metal_data, parse_series_time
from experiments import algo_dict, gen_experiment, list_dataset

np.random.seed(48)
random.seed(48)
torch.manual_seed(48)
torch.random.manual_seed(48)

np.seterr(invalid='raise')
is_test = True
    
multi_task_algo = ["GPMultiTaskMultiOut", "IndependentGP", "GPMultiTaskIndex"]
metric = ["MSE", "CRPS"]

def gen_task_list(all_algo, type_task, modifier, metal_type, 
    algo_config, len_dataset=794, len_train_show=(274, 130)):

    def gen_list(dataset, algo_list):
        multi_task_dataset = []
        for algo in algo_list:
            if algo in algo_dict.using_out_only:

                get_type_using_out = lambda x: "drop" if x in ["pca", "drop"] else "id"
                cu_modi = modifier["copper"]
                al_modi = modifier["aluminium"]

                new_modi = {
                    "copper": CompressMethod(
                        0, get_type_using_out(cu_modi), 
                        cu_modi.info
                    ), 
                    "aluminium": CompressMethod(
                        0, get_type_using_out(al_modi), 
                        al_modi.info
                    )
                }
                if type_task == "time":
                    dataset = list_dataset.gen_datasets(
                        type_task, 
                        new_modi,
                        metal_type,
                        len_dataset=len_dataset
                    )
                else:
                    dataset = list_dataset.gen_datasets(
                        type_task, 
                        new_modi,
                        metal_type,
                        len_dataset=len_dataset
                    )[0]

            multi_task_dataset.append(
                (algo, gen_experiment.create_exp(dataset, algo, algo_config, len_train_show))
            )
        return multi_task_dataset

    if type_task == "time":
        dataset = list_dataset.gen_datasets("time", modifier, metal_type, len_dataset=len_dataset)
        time_multi_task = gen_list(dataset, all_algo)
        return time_multi_task

    elif type_task == "metal":
        commo, commo_first = list_dataset.gen_datasets("metal", modifier, metal_type, len_dataset=len_dataset)
        metal_multi_task = gen_list(
            commo_first, 
            list(filter(lambda x : x in algo_dict.using_first_algo, all_algo))
        )
        metal_multi_task += gen_list(
            commo, 
            list(filter(lambda x : not x in algo_dict.using_first_algo, all_algo))
        )
        return metal_multi_task
    else:
        raise ValueError("There are only 2 tasks for now, time and metal")

def run_multi_task_gp(save_path, modifier, len_inp=10, len_dataset=794, len_train_show=(274, 130)):
    config = algo_dict.encode_params(
        "gp_multi_task", is_verbose=False, 
        is_test=is_test, 
        kernel="Composite_1", 
        optim_iter=100,
        len_inp=len_inp
    )
    multi_task_config = {
        "GPMultiTaskMultiOut": config,
        "IndependentGP": algo_dict.encode_params(
            "gp", is_verbose=False, 
            is_test=is_test, 
            kernel="Composite_1", 
            optim_iter=100,
            len_inp=len_inp
        ),
        "GPMultiTaskIndex": config,
    }
    task = gen_task_list(
        multi_task_algo, "metal", 
        modifier, ["aluminium", "copper"], multi_task_config,
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
    num_feature = np.arange(2, 14, step=2)
    num_pca = np.arange(2, 8)

    all_results = {
        algo:{m: np.zeros((len(num_feature), len(num_pca))).tolist() for m in metric}
        for algo in multi_task_algo
    }
    
    for i, feature in enumerate(num_feature):
        for j, pca in enumerate(num_pca):
            pca_modifier = {
                "copper": CompressMethod(int(pca), "pca", info={}), 
                "aluminium": CompressMethod(int(pca), "pca", info={})
            }
            curr_result = run_multi_task_gp(
                f"save/hyper_search/run_pca_{pca}_feat_{feature}/",
                pca_modifier,
                len_inp=int(feature)
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
    all_algo = ["GPMultiTaskIndex", "IIDDataModel"]
    display_name_to_algo = dict(zip(
        ["GPMultiTaskMultiOut", "IndependentGP", "GPMultiTaskIndex", "IIDDataModel", "ARIMAModel"], 
        ["Multi-Task Out", "Independent GP", "Multi-Task Index", "Mean", "ARIMA"],
    ))


    defaul_config = {
        "GPMultiTaskMultiOut": algo_dict.encode_params(
            "gp_multi_task", is_verbose=False, 
            is_test=is_test, 
            kernel="Composite_1", 
            optim_iter=100,
            len_inp=10
        ),
        "IndependentGP": algo_dict.encode_params(
            "gp", is_verbose=False, 
            is_test=is_test, 
            kernel="Composite_1", 
            optim_iter=100,
            len_inp=10
        ),
        "GPMultiTaskIndex": algo_dict.encode_params(
            "gp_multi_task", is_verbose=False, 
            is_test=is_test, 
            kernel="Composite_1", 
            optim_iter=100,
            len_inp=10
        ),
        "IIDDataModel": algo_dict.encode_params(
            "iid", is_verbose=False, 
            is_test=is_test, dist="gaussian"
        ),
        "ARIMAModel": algo_dict.encode_params(
            "arima", is_verbose=False, 
            is_test=is_test, order=(2, 0, 5)
        ),
    }

    def original_test():
        time_al = gen_task_list(all_algo, "time", no_modifier, "aluminium", defaul_config)
        time_cu = gen_task_list(all_algo, "time", no_modifier, "copper", defaul_config) 
        commodity = gen_task_list(all_algo, "metal", no_modifier, ["aluminium", "copper"], defaul_config)
        
        time_al_feat = gen_task_list(all_algo, "time", pca_modifier, "aluminium", defaul_config)
        time_cu_feat = gen_task_list(all_algo, "time", pca_modifier, "copper", defaul_config)
        commodity_feat = gen_task_list(all_algo, "metal", pca_modifier, ["aluminium", "copper"], defaul_config)
        
        task_train = [time_al, commodity, time_al_feat, commodity_feat]
        task_names = ["Price", "Metal", "Price_Feat", "Metal_Feat"]
        return task_train, task_names

    task_train, task_names = original_test()

    super_task = {}
    for task, name in zip(task_train, task_names):
        all_out = gen_experiment.run_experiments(task)
        super_task.update({name: all_out})
    
    dump_json("save/all_data.json", super_task) 
    # super_task = load_json("save/all_data.json")
    
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
    # run_hyperparam_search()
    # run_multi_task_gp("save/test")
    # run_years_prediction()
    # run_window_prediction()
    # run_years_prediction_feature()
    # run_window_prediction_feature()
    general_testing()


if __name__ == '__main__':
    main()

