import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import torch
import copy

from utils.data_preprocessing import load_transform_data, parse_series_time
from utils.data_structure import DisplayPrediction
from utils.data_visualization import visualize_time_series, visualize_walk_forward, show_result_fold, pack_result_data
from utils.others import create_folder, save_fold_data, load_fold_data, create_name, dump_json, load_json, create_legacy_exp_setting
from utils.data_structure import DatasetTaskDesc
from utils.data_preprocessing import load_dataset_from_desc
from experiments.metal_desc import metal_to_display_name

from experiments.algo_dict import algorithms_dic, multi_task_algo, class_name_to_display
from experiments.eval_methods import prepare_dataset, walk_forward
from models.cluster_multi_model import HardClusterMultiModel

class SkipLookUp(object):
    def __init__(self, skip, all_date):
        self.skip = skip
        self.all_date = all_date
    
    def __getitem__(self, date):
        if date not in self.all_date:
            raise KeyError("Date Not Found", date)
        
        return self.skip + date
    
    def __call__(self, date):
        return self[date]
    
    def reverse(self, date):
        if date < self.skip:
            raise ValueError("Can't Go Back to Negative Time")
        else:
            return date - self.skip

def get_data_example(dataset_desc): 
    features, log_prices = load_dataset_from_desc(dataset_desc)
    log_prices = log_prices[["Output"]]

    # This can be integrated later.....
    len_dataset = dataset_desc["len_dataset"]

    if len_dataset != -1:
        features = features.head(len_dataset)
        log_prices = log_prices.head(len_dataset)

    len_data = len(features)

    first_day = features["Date"].iloc[0]

    convert_obj = SkipLookUp(
        skip=dataset_desc["out_feat_tran_lag"][1],
        all_date=parse_series_time(
            features["Date"].to_list(), 
            first_day
        )[0]
    )
    return features, log_prices, first_day, len_data, convert_obj 

def create_task(len_inp, len_out, len_pred_show, dataset_desc):
    return_lag = dataset_desc["out_feat_tran_lag"][0]
    features, log_prices, first_day, len_data, convert_date = get_data_example(
        dataset_desc
    )

    splitted_data = prepare_dataset(
        features, first_day, log_prices, 
        len_inp=len_data-len_pred_show, len_out=len_pred_show, return_lag=0, 
        convert_date=True, offset=-1, is_show_progress=False, num_dataset=-1, is_padding=False
    )     
    features_train, log_prices_train, feature_test, log_prices_test = splitted_data[0]
    train_dataset = prepare_dataset(
        features_train, first_day, log_prices_train, 
        len_inp=len_inp, len_out=len_out, return_lag=return_lag, 
        convert_date=False, offset=1, is_show_progress=False, num_dataset=-1, is_padding=False
    )  
    pred_dataset = prepare_dataset(
        feature_test, first_day, log_prices_test, 
        len_inp=len_inp, len_out=len_out, return_lag=return_lag, 
        convert_date=False, offset=len_out, is_show_progress=False, num_dataset=-1, is_padding=False
    )
    return (features_train, log_prices_train, feature_test, log_prices_test), (train_dataset, pred_dataset), convert_date, first_day

def prepare_task(task, len_inp, return_lag, plot_gap):
    _, _, feature_test, log_prices_test = task[0]
    train_dataset, pred_dataset = task[1]
    convert_date = task[2]

    all_date_pred = list(itertools.chain.from_iterable(
        [point.data_out["Date"].map(convert_date).to_list() for point in pred_dataset]
    ))
    true_date = feature_test["Date"].map(convert_date).to_list()
    true_price = log_prices_test["Output"].to_list() 
    
    if plot_gap:
        missing_x = true_date[:len_inp+return_lag]
        missing_y = true_price[:len_inp+return_lag] 
        missing_data = (missing_x, missing_y)
    else:
        missing_data = ([], [])
    
    return all_date_pred, true_date, true_price, missing_data, convert_date

def gen_prepare_task(len_inp, len_out, 
    len_pred_show, plot_gap, dataset_desc):

    return_lag = dataset_desc["out_feat_tran_lag"][0]
    task = create_task(
        len_inp, len_out, len_pred_show, dataset_desc
    )
    helper = prepare_task(task, len_inp, return_lag, plot_gap)
    return (task, helper)

def get_display_data(all_exp_setting):
    all_task = all_exp_setting["task"]
    is_all_independent = all(len(sub_model) == 1 for sub_model in all_task["sub_model"])
    is_all_joint = len(all_task["sub_model"]) == 1

    clus_num = list(itertools.chain(*[
        [i+1]*len(task) 
        for i, task in enumerate(all_task["sub_model"])
    ]))

    output_name = [
        metal_to_display_name[desc["out_feature"].split(".")[0]]
        for desc in itertools.chain(*all_task["dataset"])
    ]
        
    method_name = []
    for i, sub_model in enumerate(itertools.chain(*all_task["sub_model"])):
        base_method = algorithms_dic[sub_model][1] 

        # Mean that it is not independent
        if base_method is None:
            name = all_exp_setting["algo"][clus_num[i]-1]
        else:
            name = base_method.__name__

        method_name.append(class_name_to_display[name])
    
    is_show_cluster = not (is_all_independent or is_all_joint)
    return clus_num, output_name, method_name, is_show_cluster

def loading_all_exp_setting(setting_path):
    all_exp_setting = load_json(setting_path)
    all_exp_setting["task"]["dataset"] = [
        [DatasetTaskDesc(**d) for d in clus_desc]
        for clus_desc in all_exp_setting["task"]["dataset"]
    ]
    return all_exp_setting
    
def example_plot_all_algo_lag(all_exp_setting, 
    plot_gap=True, is_save=True, is_load=False, load_path=None, save_path="save/"):

    def prepare_model_train(exp_setting):
        train_dataset_list = []
        algo_hyper_class_list = []

        pred_dataset_list = []
        len_pred_list = []
        date_pred_list = []

        true_pred_list = []
        full_feature_list = []
        log_prices_train_list = []
        missing_data_list = []

        first_day_list = []


        plot_all_algo = [
            exp_setting["task"]["sub_model"],
            exp_setting["task"]["dataset"],
        ]
        len_pred_show = exp_setting["task"]["len_pred_show"]
        num_task = len(plot_all_algo[0])
        assert all(num_task == len(a) for a in plot_all_algo)
        plot_all_algo_iter = zip(*plot_all_algo)
        
        for i, (algo_name, dataset) in enumerate(plot_all_algo_iter):
            # New modifier every-task
            hyperparam, algo_class = algorithms_dic[algo_name]
            len_inp = hyperparam["len_inp"]
            len_out = hyperparam["len_out"]

            # Adding info the hyperparam
            hyperparam["using_first"] = exp_setting["using_first"]

            task_helper = gen_prepare_task(
                len_inp=len_inp, 
                len_out=len_out,
                len_pred_show=len_pred_show, 
                plot_gap=plot_gap, 
                dataset_desc=dataset,
            )
            task, helper = task_helper

            features_train, log_prices_train, _, _ = task[0]
            train_dataset, pred_dataset = task[1]
            first_date = task[3]
            all_date_pred, true_date, true_price, missing_data, convert_date = helper 
            return_lag = dataset["out_feat_tran_lag"][0]

            # Used For Training
            train_dataset_list.append(train_dataset)
            algo_hyper_class_list.append((hyperparam, algo_class))
            first_day_list.append(first_date)
            
            # Used For Prediction
            pred_dataset_list.append(pred_dataset)
            
            if not exp_setting["using_first"]:
                len_pred_list.append(len(all_date_pred))
                date_pred_list.append(all_date_pred)
            else:
                if i == 0:
                    basis_time_step = [convert_date.reverse(d) for d in all_date_pred]
                
                    len_pred_list.append(len(all_date_pred))
                    date_pred_list.append(all_date_pred)
                else:
                    len_pred_list.append(len(all_date_pred))
                    all_date_pred = [convert_date(d) for d in basis_time_step]
                    
                    date_pred_list.append(all_date_pred)

            # Used For Display
            true_pred_list.append(DisplayPrediction(
                pack_result_data(true_price[len_inp+return_lag:], [], [], true_date[len_inp+return_lag:]),
                name="True Value", is_bridge=False
            ))
        
            full_feature = features_train.copy()
            full_feature["Date"] = full_feature["Date"].map(convert_date)

            full_feature_list.append(full_feature)
            log_prices_train_list.append(log_prices_train)
            missing_data_list.append(missing_data)

        construct_model = (
            train_dataset_list, algo_hyper_class_list, 
            exp_setting["using_first"], multi_task_algo[exp_setting["algo"]]
        )
        prediction = (
            pred_dataset_list, 
            len_pred_list, date_pred_list
        )
        plotting = (
            num_task, full_feature_list, log_prices_train_list, 
            true_pred_list, missing_data_list, first_day_list
        )

        return construct_model, prediction, plotting
    
 
    def build_model_pred(all_setting):
        model_inp, pred_inp, plotting_data = zip(*[
            prepare_model_train(exp_setting)
            for exp_setting in create_legacy_exp_setting(all_setting)
        ])

        model = HardClusterMultiModel(*zip(*model_inp))

        # Redo the zipping
        all_plot = [
            sum(i) if isinstance(i[0], int) else list(itertools.chain(*i))
            for i in zip(*plotting_data)
        ]

        return model, pred_inp, all_plot
    
    def train_model(all_exp_setting, is_save=False, is_train=True):
        model, pred_inp, useful_info = build_model_pred(all_exp_setting)
        if is_train:
            model.train()
        if is_save:
            base_name = create_name(save_path, load_path)
            model.save(base_name)
            dump_json(base_name + "/exp_setting.json", all_exp_setting)

        return model, pred_inp, useful_info
    
    if load_path is not None:
        if is_load:
            load_base = save_path + load_path
            all_exp_setting = loading_all_exp_setting(f"{load_base}/exp_setting.json")
            _, pred_inp, useful_info = train_model(
                all_exp_setting, is_train=False
            )
            model = HardClusterMultiModel.load_from_path(load_base)
        elif is_save:
            model, pred_inp, useful_info = train_model(all_exp_setting, is_save=True)
        else:
            model, pred_inp, useful_info = train_model(all_exp_setting)
    else:
        model, pred_inp, useful_info = train_model(all_exp_setting)
    
    (num_task, full_feature_list, log_prices_train_list, 
        true_pred_list, missing_data_list, first_day_list) = useful_info

    pred = model.predict(*zip(*pred_inp))
    pred = list(itertools.chain(*pred))

    fig, axes = plt.subplots(nrows=num_task, figsize=(15, 6))

    # Generating Names
    clus_num, output_name, method_name, is_show_cluster = get_display_data(all_exp_setting)  

    for i in range(num_task):
        model_pred = DisplayPrediction(
            pred[i], name=method_name[i], color="p", is_bridge=False
        )

        additional = f"on Cluster {clus_num[i]}" if is_show_cluster else ""

        fig, ax1 = visualize_time_series(
            (fig, axes[i]), 
            ((full_feature_list[i], log_prices_train_list[i]), [true_pred_list[i], model_pred]), 
            "k", missing_data_list[i], "o", first_day_list[i], 
            title=f"Prediction Results: {output_name[i]} " + additional
        )

    fig.tight_layout()        
    plt.show()
    
def example_plot_walk_forward(all_exp_setting, model_name, load_path, 
    is_save=False, is_load=True, is_show=True, save_path="save/"):
    
    def full_model_running(exp_setting):
        all_data = []
        return_lag = []
         
        plot_all_algo = [
            exp_setting["task"]["dataset"],
            exp_setting["task"]["sub_model"],
        ]
        num_task = len(plot_all_algo[0])
        assert all(num_task == len(a) for a in plot_all_algo)
        plot_all_algo_iter = zip(*plot_all_algo)
        
        for dataset_desc, algo_name in plot_all_algo_iter:
            features, log_prices, _, _, convert_date = get_data_example(dataset_desc) 
            all_data.append(
                (features, log_prices, convert_date, algorithms_dic[algo_name])
            )
            return_lag.append(dataset_desc["out_feat_tran_lag"][0])

        return all_data
    
    def running_experiment(all_exp_setting, is_train=True):
        task_all_data = [
            (full_model_running(exp_setting), exp_setting["task"])
            for exp_setting in create_legacy_exp_setting(all_exp_setting)
        ]

        list_all_data, clus_task_setting = zip(*task_all_data)
        
        if not is_train:
            return list_all_data
        
        size_train, size_test = all_exp_setting["task"]["len_train_show"]
        
        run_fold = lambda: walk_forward(
            list(list_all_data), list(clus_task_setting),
            clus_algo=[
                multi_task_algo[algo]
                for algo in all_exp_setting["algo"]
            ],
            size_train=size_train, size_test=size_test, 
            train_offset=1, 
            clus_using_first=all_exp_setting["using_first"],
            is_train_pad=True, 
            is_test_pad=False 
        )
        fold_result = run_fold()
        return list_all_data, fold_result
    

    if load_path is not None:
        if is_load:
            all_exp_setting = loading_all_exp_setting(save_path + load_path  + "/all_exp_setting.json")
            list_all_data = running_experiment(all_exp_setting, is_train=False)
            fold_result = load_fold_data(
                load_path, model_name, HardClusterMultiModel, 
                save_path=save_path
            )

        elif is_save:
            list_all_data, fold_result = running_experiment(all_exp_setting)
            base_folder = create_name(save_path, model_name)
            save_fold_data(fold_result, model_name, base_folder)
            dump_json(base_folder + "/all_exp_setting.json", all_exp_setting)
        else:
            list_all_data, fold_result = running_experiment(all_exp_setting)
 
    else:
        list_all_data, fold_result = running_experiment(all_exp_setting)
    
    display_data = get_display_data(all_exp_setting)  

    clus_num, output_name, method_name, is_show_cluster = display_data

    total_num_task = len(clus_num)
    flatten_all_data = itertools.chain(*list_all_data)

    if is_show:

        fig = plt.figure(figsize=(15, 5*len(clus_num)), constrained_layout=True)
        subfigs = fig.subfigures(len(clus_num), 1)

        for task_number, all_data in enumerate(flatten_all_data):
            axs = subfigs[task_number].subplots(2, 1, sharex=True, sharey=False)

            if is_show_cluster:
                added_str = f" of Cluster ({clus_num[task_number]})"
            else:
                added_str = ""

            test_features, test_log_prices, test_convert_date, _ = all_data
            visualize_walk_forward(
                subfigs[task_number], axs, test_features, 
                test_log_prices, fold_result[task_number], test_convert_date,
                lag_color="o", pred_color="b", below_err="r",
                title=f"Walk Forward Validation Loss Visualization" + added_str,
                true_value_name=output_name[task_number],
                method_name=method_name[task_number]
            )

        fig.savefig(f"img/walk_forward_task_{model_name}.pdf")
        plt.show()
    
    return show_result_fold(fold_result, all_exp_setting, other_details=display_data)
