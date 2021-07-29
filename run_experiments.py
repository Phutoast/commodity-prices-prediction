import numpy as np
import random
import torch
import os
import json

from utils.others import create_folder, dump_json, load_json
from utils.data_visualization import plot_latex
from utils.data_structure import DatasetTaskDesc, CompressMethod
from experiments import algo_dict, gen_experiment, list_dataset

np.random.seed(48)
random.seed(48)
torch.manual_seed(48)
torch.random.manual_seed(48)

np.seterr(invalid='raise')
is_test = True

def gen_task_list(all_algo, type_task, modifier, metal_type, algo_config):

    def gen_list(dataset, algo_list):
        multi_task_dataset = []
        for algo in algo_list:
            if algo in algo_dict.using_out_only:
                if type_task == "time":
                    dataset = list_dataset.gen_datasets(
                        type_task, 
                        {"copper": CompressMethod(0, "drop"), "aluminium": CompressMethod(0, "drop")}, 
                        metal_type
                    )
                else:
                    dataset = list_dataset.gen_datasets(
                        type_task, 
                        {"copper": CompressMethod(0, "drop"), "aluminium": CompressMethod(0, "drop")}, 
                        metal_type
                    )[0]

            multi_task_dataset.append(
                (algo, gen_experiment.create_exp(dataset, algo, algo_config))
            )
        return multi_task_dataset

    if type_task == "time":
        dataset = list_dataset.gen_datasets("time", modifier, metal_type)
        time_multi_task = gen_list(dataset, all_algo)
        return time_multi_task

    elif type_task == "metal":
        commo, commo_first = list_dataset.gen_datasets("metal", modifier, metal_type=None)
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

def run_multi_task_gp(save_path, len_inp=10, pca_dim=3):
    multi_task_algo = ["GPMultiTaskMultiOut", "IndependentGP", "GPMultiTaskIndex"]
    pca_modifier = {
        "copper": CompressMethod(pca_dim, "pca"), 
        "aluminium": CompressMethod(pca_dim, "pca")
    }
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
            len_inp=10
        ),
        "GPMultiTaskIndex": config,
    }
    task = gen_task_list(multi_task_algo, "metal", pca_modifier, None, multi_task_config)
    output = gen_experiment.run_experiments(task, save_path=save_path)
    print(output)
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
    multi_task_algo = ["GPMultiTaskMultiOut", "IndependentGP", "GPMultiTaskIndex"]
    metric = ["MSE", "CRPS"]
    
    num_feature = np.arange(2, 14, step=2)
    num_pca = np.arange(2, 8)
    # num_feature = np.arange(2, 6, step=2)
    # num_pca = np.arange(2, 5)

    all_results = {
        algo:{m: np.zeros((len(num_feature), len(num_pca))).tolist() for m in metric}
        for algo in multi_task_algo
    }

    for i, feature in enumerate(num_feature):
        for j, pca in enumerate(num_pca):
            curr_result = run_multi_task_gp(
                f"save/hyper_search/run_pca_{pca}_feat_{feature}/",
                len_inp=int(feature), pca_dim=int(pca)
            )
            for algo in multi_task_algo:
                for met in metric:
                    all_results[algo][met][i][j] = curr_result[algo][met]
    
    dump_json("save/hyper_search/final_result.json", all_results) 
    return all_results

def plot_hyperparam_search(load_path):
    all_results = load_json(load_path)
    print(all_results)
    


def general_testing():
    no_modifier = {"copper": CompressMethod(0, "drop"), "aluminium": CompressMethod(0, "drop")}
    pca_modifier = {"copper": CompressMethod(3, "pca"), "aluminium": CompressMethod(3, "pca")}
    
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

    time_al = gen_task_list(all_algo, "time", no_modifier, "aluminium", defaul_config)
    time_cu = gen_task_list(all_algo, "time", no_modifier, "copper", defaul_config) 
    commodity = gen_task_list(all_algo, "metal", no_modifier, None, defaul_config)
    
    time_al_feat = gen_task_list(all_algo, "time", pca_modifier, "aluminium", defaul_config)
    time_cu_feat = gen_task_list(all_algo, "time", pca_modifier, "copper", defaul_config)
    commodity_feat = gen_task_list(all_algo, "metal", pca_modifier, None, defaul_config)
    
    task_train = [time_al, commodity, time_al_feat, commodity_feat]
    task_names = ["Price", "Metal", "Price_Feat", "Metal_Feat"]

    super_task = {}
    for task, name in zip(task_train, task_names):
        all_out = gen_experiment.run_experiments(task)
        super_task.update({name: all_out})
    
    # dump_json("save/all_data.json", super_task) 
    super_task = load_json("save/all_data.json")
    print(super_task)
    assert False
    
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


def main():
    create_folder("save")
    run_hyperparam_search()


if __name__ == '__main__':
    main()

