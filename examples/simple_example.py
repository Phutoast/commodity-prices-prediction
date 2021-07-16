import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import torch

from utils.data_preprocessing import load_transform_data, parse_series_time
from utils.data_structure import DisplayPrediction
from utils.data_visualization import visualize_time_series, visualize_walk_forward, show_result_fold, pack_result_data
from utils.others import create_folder, save_fold_data, load_fold_data, create_name
from utils.data_structure import DatasetTaskDesc
from utils.data_preprocessing import load_dataset_from_desc

from experiments.algo_dict import algorithms_dic
from experiments.eval_methods import prepare_dataset, walk_forward
from experiments.calculation import PerformanceMetric

from models.ind_multi_model import IndependentMultiModel

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
    total_dataset = dataset_desc["len_dataset"]

    if total_dataset != -1:
        features = features.head(total_dataset)
        log_prices = log_prices.head(total_dataset)
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
    features, log_prices, first_day, len_data, convert_date = get_data_example(dataset_desc)
    splitted_data = prepare_dataset(
        features, first_day, log_prices, 
        len_inp=len_data-len_pred_show, len_out=len_pred_show, return_lag=0, 
        convert_date=True, offset=-1, is_show_progress=False, num_dataset=-1, is_padding=False
    )     
    features_train, log_prices_train, feature_test, log_prices_test = splitted_data[0]
    print(feature_test.head(25))
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
    return (features_train, log_prices_train, feature_test, log_prices_test), (train_dataset, pred_dataset), convert_date

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

def gen_prepare_task(
    len_inp, len_out, len_pred_show, plot_gap, dataset_desc):
    return_lag = dataset_desc["out_feat_tran_lag"][0]
    task = create_task(
        len_inp, len_out, len_pred_show, dataset_desc
    )
    helper = prepare_task(task, len_inp, return_lag, plot_gap)
    return (task, helper)

def example_plot_all_algo_lag(exp_setting, plot_gap=True, is_save=True, is_load=False, load_path=None):

    train_dataset_list = []
    algo_hyper_class_list = []

    pred_dataset_list = []
    len_pred_list = []
    date_pred_list = []

    true_pred_list = []
    full_feature_list = []
    log_prices_train_list = []
    missing_data_list = []

    plot_all_algo = [
        exp_setting["task"]["sub_model"],
        exp_setting["task"]["dataset"],
        exp_setting["task"]["len_pred_show"]
    ]
    num_task = len(plot_all_algo[0])
    assert all(num_task == len(a) for a in plot_all_algo)
    plot_all_algo_iter = zip(*plot_all_algo)
    
    for i, (algo_name, dataset, len_pred_show) in enumerate(plot_all_algo_iter):
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
            dataset_desc=dataset
        )
        task, helper = task_helper

        features_train, log_prices_train, _, _ = task[0]
        train_dataset, pred_dataset = task[1]
        all_date_pred, true_date, true_price, missing_data, convert_date = helper 
        return_lag = dataset["out_feat_tran_lag"][0]

        # Used For Training
        train_dataset_list.append(train_dataset)
        algo_hyper_class_list.append((hyperparam, algo_class))
        
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
    
    model = exp_setting["algo"](
        train_dataset_list, 
        algo_hyper_class_list,
        exp_setting["using_first"]
    )

    if load_path is not None:
        if is_load:
            # model.load(f"save/{load_path}")
            model = exp_setting["algo"].load_from_path(f"save/{load_path}")
        elif is_save:
            model.train()
            base_name = create_name("save/", load_path)
            model.save(base_name)
        else:
            model.train()
    else:
        model.train()

    pred = model.predict(
        pred_dataset_list, 
        len_pred_list, 
        date_pred_list, 
        ci=0.9
    )

    fig, axes = plt.subplots(nrows=num_task, figsize=(15, 6))
    for i in range(num_task):
        model_pred = DisplayPrediction(
            pred[i], name=exp_setting["task"]["sub_model"][i], color="p", is_bridge=False
        )

        fig, ax1 = visualize_time_series(
            (fig, axes[i]), 
            ((full_feature_list[i], log_prices_train_list[i]), [true_pred_list[i], model_pred]), 
            "k", missing_data_list[i], "o", title="Log Lag over Time"
        )

    fig.tight_layout()        
    plt.show()

def example_plot_walk_forward(exp_setting, model_name, load_path, is_save=False, is_load=True):

    all_data = []
    return_lag_list = []
    
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
        return_lag_list.append(dataset_desc["out_feat_tran_lag"][0])
 
    metric = PerformanceMetric()

    run_fold = lambda: walk_forward(
        all_data, exp_setting["task"],  
        exp_setting["algo"],
        metric.square_error, 
        size_train=300, size_test=200, 
        train_offset=1, 
        return_lag_list=return_lag_list, 
        convert_date=convert_date,
        using_first=exp_setting["using_first"],
        is_train_pad=True, 
        is_test_pad=False 
    )
    
    if load_path is not None:
        if is_load:
            fold_result = load_fold_data(
                load_path, model_name, exp_setting["algo"]
            )
        elif is_save:
            base_folder = create_name("save/", model_name)
            fold_result = run_fold()
            save_fold_data(fold_result, model_name, base_folder)
        else:
            fold_result = run_fold()
    else:
        fold_result = run_fold()

    for task_number in range(len(all_data)):
        test_features, test_log_prices, test_convert_date, _ = all_data[task_number]
        fig, ax = visualize_walk_forward(
            test_features, test_log_prices, fold_result[task_number], test_convert_date,
            lag_color="o", pred_color="b", below_err="r",
            title=f"Task {task_number+1}"
        )
        fig.savefig(f"img/walk_forward_task_{model_name}_task_{task_number}")
    
    show_result_fold(fold_result, exp_setting)
    # plt.show()

