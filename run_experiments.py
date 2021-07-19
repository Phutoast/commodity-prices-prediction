import numpy as np
import random
import torch
import os
import json

from utils.others import create_folder, dump_json, load_json
from utils.data_visualization import plot_latex
from experiments import algo_dict, gen_experiment, list_dataset
from utils.data_structure import DatasetTaskDesc, CompressMethod

np.random.seed(48)
random.seed(48)
torch.manual_seed(48)
torch.random.manual_seed(48)

np.seterr(invalid='raise')

def generated_train_list(type_task="price"):
    if type_task == "price":
        print("HERE")
    elif type_task == "commodity":
        print("HERE")

def main():
    create_folder("save")
    for (k, v) in algo_dict.algorithms_dic.items():
        v[0]["is_verbose"] = True
        v[0]["optim_iter"] = 1

    # Not work if aluminium: None
    out, out_is_first = list_dataset.generate_list_dataset("metal", modifier={
        "copper": CompressMethod(3, "pca"), "aluminium": CompressMethod(3, "pca")
    })

    exp1 = gen_experiment.create_exp_setting(out, "GPMultiTaskIndex")
    exp2 = gen_experiment.create_exp_setting(out_is_first, "GPMultiTaskMultiOut")
    gen_experiment.run_experiments(zip(["GPMultiTaskIndex", "GPMultiTaskMultiOut"], [exp1, exp2]))

    assert False

    all_algo = ["GPMultiTaskMultiOut", "IndependentGP", "GPMultiTaskIndex", "IIDDataModel", "ARIMAModel"]
    display_name_to_algp = dict(zip(
        ["Multi-Task Out", "Independent GP", "Multi-Task Index", "Mean", "ARIMA"], all_algo
    ))

    price_multi_task = [
        (algo, gen_experiment.create_exp_setting(list_dataset.diff_time, algo))
        for algo in all_algo
    ]

    commodity_multi_task = [
        ("GPMultiTaskMultiOut", gen_experiment.create_exp_setting(list_dataset.diff_matal_use_first, "GPMultiTaskMultiOut"))
    ] + [
        (algo, gen_experiment.create_exp_setting(list_dataset.diff_matal, algo))
        for algo in all_algo[1:]
    ]

    price_multi_task_feature = {
        (algo, gen_experiment.create_exp_setting(list_dataset.diff_time_pca_feature, algo))
        for algo in all_algo
    }
       
    for (k, v) in algo_dict.algorithms_dic.items():
        v[0]["is_verbose"] = True
        v[0]["optim_iter"] = 1
    
    super_task = {}
    all_out = gen_experiment.run_experiments(price_multi_task)
    super_task.update({"Price": all_out})
    
    all_out = gen_experiment.run_experiments(commodity_multi_task)
    super_task.update({"Metal": all_out})

    # all_save_path = list(filter(lambda x: not ".json" in x, sorted(os.listdir("save/"))))
    # all_save_path = zip(all_algo + all_algo, all_save_path)
    # gen_experiment.load_experiments(all_save_path)

    dump_json("save/all_data.json", super_task) 
    # super_task = load_json("save/all_data.json")

    plot_latex(
        names=[
            ["Multi-Task Out", "Independent GP", "Multi-Task Index", "Mean", "ARIMA"], 
            ["Multi-Task Out", "Independent GP", "Multi-Task Index", "Mean", "ARIMA"]
        ],
        results=[super_task["Price"], super_task["Metal"]],
        multi_task_name=[["Date (22)", "Date (44)", "Date (66)"], ["Aluminium", "Copper"]],
        display_name_to_algp=display_name_to_algp
    )
    

if __name__ == '__main__':
    main()

