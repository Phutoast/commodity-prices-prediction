import numpy as np
import random
import argparse
import torch

from examples.simple_example import example_plot_walk_forward
from utils.others import create_folder, dump_json, load_json

from utils.data_structure import DatasetTaskDesc, CompressMethod
from utils.data_visualization import plot_latex
from experiments import algo_dict, gen_experiment
import os

import warnings

np.random.seed(48)
random.seed(48)
torch.manual_seed(48)
torch.random.manual_seed(48)

np.seterr(invalid='raise')

def main():
    create_folder("save")

    length_dataset = 795

    dataset1 = [
        DatasetTaskDesc(
            inp_metal_list=["aluminium"],
            use_feature=["Date"],
            use_feat_tran_lag=None,
            out_feature="aluminium.Price",
            out_feat_tran_lag=(22, 0, "id"),
            len_dataset=length_dataset,
        ),
        DatasetTaskDesc(
            inp_metal_list=["aluminium"],
            use_feature=["Date"],
            use_feat_tran_lag=None,
            out_feature="aluminium.Price",
            out_feat_tran_lag=(44, 0, "id"),
            len_dataset=length_dataset,
        ),
        DatasetTaskDesc(
            inp_metal_list=["aluminium"],
            use_feature=["Date"],
            use_feat_tran_lag=None,
            out_feature="aluminium.Price",
            out_feat_tran_lag=(66, 0, "id"),
            len_dataset=length_dataset,
        ),
    ]
     
    dataset2 = [
        DatasetTaskDesc(
            inp_metal_list=["aluminium"],
            use_feature=["Date"],
            use_feat_tran_lag=None,
            out_feature="aluminium.Price",
            out_feat_tran_lag=(22, 0, "id"),
            len_dataset=length_dataset,
        ),
        DatasetTaskDesc(
            inp_metal_list=["copper"],
            use_feature=["Date"],
            use_feat_tran_lag=None,
            out_feature="copper.Price",
            out_feat_tran_lag=(22, 0, "id"),
            len_dataset=length_dataset,
        ),
    ]
    
    dataset3 = [
        DatasetTaskDesc(
            inp_metal_list=["aluminium", "copper"],
            use_feature=["Date", "copper.Price"],
            use_feat_tran_lag=[None, (22, 0, "id")],
            out_feature="aluminium.Price",
            out_feat_tran_lag=(22, 0, "id"),
            len_dataset=length_dataset,
        ),
        DatasetTaskDesc(
            inp_metal_list=["copper"],
            use_feature=["Date"],
            use_feat_tran_lag=None,
            out_feature="copper.Price",
            out_feat_tran_lag=(22, 0, "id"),
            len_dataset=length_dataset,
        ),
    ]

    all_algo = ["GPMultiTaskMultiOut", "IndependentGP", "GPMultiTaskIndex", "IIDDataModel", "ARIMAModel"]
    algo_to_display_name = dict(zip(
        all_algo, ["Multi-Task Out", "Independent GP", "Multi-Task Index", "Mean", "ARIMA"], 
    ))
    display_name_to_algp = dict(zip(
        ["Multi-Task Out", "Independent GP", "Multi-Task Index", "Mean", "ARIMA"], all_algo
    ))

    price_multi_task = [
        (algo, gen_experiment.create_exp_setting(dataset1, algo))
        for algo in all_algo
    ]

    commondity_multi_task = [
        ("GPMultiTaskMultiOut", gen_experiment.create_exp_setting(dataset3, "GPMultiTaskMultiOut"))
    ] + [
        (algo, gen_experiment.create_exp_setting(dataset2, algo))
        for algo in all_algo[1:]
    ]
       
    for (k, v) in algo_dict.algorithms_dic.items():
        v[0]["is_verbose"] = True
        v[0]["optim_iter"] = 1    
    
    # super_task = {}
    # all_out = gen_experiment.run_experiments(price_multi_task)
    # super_task.update({"Price": all_out})
    
    # all_out = gen_experiment.run_experiments(commondity_multi_task)
    # super_task.update({"Metal": all_out})


    # all_save_path = list(filter(lambda x: not ".json" in x, sorted(os.listdir("save/"))))
    # all_save_path = zip(all_algo + all_algo, all_save_path)
    # gen_experiment.load_experiments(all_save_path)

    # dump_json("save/all_data.json", super_task) 
    super_task = load_json("save/all_data.json")
    # print(super_task)

    plot_latex(
        names=[
            ["Multi-Task Out", "Independent GP", "Multi-Task Index", "Mean", "ARIMA"], 
            ["Independent GP", "Multi-Task Index", "Mean", "ARIMA"]
        ],
        results=[super_task["Price"], super_task["Metal"]],
        multi_task_name=[["Date (22)", "Date (44)", "Date (66)"], ["Aluminium", "Copper"]],
        display_name_to_algp=display_name_to_algp
    )
    

if __name__ == '__main__':
    main()

