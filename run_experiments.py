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

def main():
    create_folder("save")
    
    no_modifier = {"copper": CompressMethod(0, "drop"), "aluminium": CompressMethod(0, "drop")}
    pca_modifier = {"copper": CompressMethod(3, "pca"), "aluminium": CompressMethod(3, "pca")}
    
    all_algo = ["GPMultiTaskMultiOut", "IndependentGP", "GPMultiTaskIndex", "IIDDataModel", "ARIMAModel"]
    all_algo = ["IIDDataModel"]
    # display_name_to_algo = dict(zip(
    #     ["Multi-Task Out", "Independent GP", "Multi-Task Index", "Mean", "ARIMA"], 
    #     ["GPMultiTaskMultiOut", "IndependentGP", "GPMultiTaskIndex", "IIDDataModel", "ARIMAModel"], 
    # ))
    display_name_to_algo = dict(zip(
        ["GPMultiTaskMultiOut", "IndependentGP", "GPMultiTaskIndex", "IIDDataModel", "ARIMAModel"], 
        ["Multi-Task Out", "Independent GP", "Multi-Task Index", "Mean", "ARIMA"],
    ))

    defaul_config = {
        "GPMultiTaskMultiOut": "v-GP_Multi_Task-Composite_1-100",
        "IndependentGP": "v-GP-Composite_1-100",
        "GPMultiTaskIndex": "v-GP_Multi_Task-Composite_1-100",
        "IIDDataModel": "iid",
        "ARIMAModel": "ARIMA"
    }
    
    defaul_config = {
        "GPMultiTaskMultiOut": "v-GP_Multi_Task-Composite_1-100",
        "IndependentGP": "v-GP-Composite_1-100",
        "GPMultiTaskIndex": "v-GP_Multi_Task-Composite_1-100",
        "IIDDataModel": "iid",
        "ARIMAModel": "ARIMA"
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
    
    dump_json("save/all_data.json", super_task) 
    
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


if __name__ == '__main__':
    main()

