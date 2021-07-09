import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from utils.data_preprocessing import load_transform_data, parse_series_time
from utils.data_structure import DisplayPrediction, pack_result_data
from utils.data_visualization import visualize_time_series, visualize_walk_forward, show_result_fold
from utils.others import create_folder, save_fold_data, load_fold_data, create_name

from experiments.algo_dict import algorithms_dic
from experiments.eval_methods import prepare_dataset, walk_forward
from experiments.calculation import PerformanceMetric

from models.ind_multi_model import IndependentMultiModel

def get_data_example(return_lag, skip):
    metal_type = "aluminium"
    total_dataset = 1000

    features, log_prices = load_transform_data(metal_type, return_lag, skip) 
    log_prices = log_prices[["Price"]]

    features = features.head(total_dataset)
    log_prices = log_prices.head(total_dataset)
    len_data = len(features)

    first_day = features["Date"].iloc[0]
    # features = features[["Date", "FeatureFamily.TECHNICAL"]]
    features = features[["Date"]]

    def pred_date_conversion():
        """
        The given a skip, the date of the prediction of the model has to 
            be added forward to match the real time
        """
        features_no_skip, _ = load_transform_data(metal_type, return_lag, 0) 
        features_no_skip = features_no_skip.head(total_dataset+skip).tail(total_dataset)

        inp_date_data, _ = parse_series_time(
            features["Date"].to_list(), first_day) 
        true_date, _ = parse_series_time(
            features_no_skip["Date"].to_list(), first_day) 
         
        return dict(zip(inp_date_data, true_date))
     
    return features, log_prices, first_day, len_data, pred_date_conversion()

def create_task(len_inp, len_out, return_lag, len_pred_show, skip):
    features, log_prices, first_day, len_data, convert_date = get_data_example(return_lag, skip)
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
    return (features_train, log_prices_train, feature_test, log_prices_test), (train_dataset, pred_dataset), convert_date

def prepare_task(task, len_inp, return_lag, skip, plot_gap):
    _, _, feature_test, log_prices_test = task[0]
    train_dataset, pred_dataset = task[1]
    convert_date = task[2]

    all_date_pred = list(itertools.chain.from_iterable(
        [point.data_out["Date"].map(convert_date).to_list() for point in pred_dataset]
    ))
    true_date = feature_test["Date"].map(convert_date).to_list()
    true_price = log_prices_test["Price"].to_list() 
    
    if plot_gap:
        missing_x = true_date[:len_inp+return_lag]
        missing_y = true_price[:len_inp+return_lag] 
        missing_data = (missing_x, missing_y)
    else:
        missing_data = ([], [])
    
    return all_date_pred, true_date, true_price, missing_data, convert_date

def gen_prepare_task(len_inp, len_out, return_lag, len_pred_show, skip, plot_gap):
    task = create_task(
        len_inp, len_out, 
        return_lag, len_pred_show, skip
    )
    helper = prepare_task(task, len_inp, return_lag, skip, plot_gap)
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
    
    for i, (algo_name, return_lag, skip, len_pred_show) in enumerate(exp_setting):
        hyperparam, algo_class = algorithms_dic[algo_name]
        len_inp = hyperparam["len_inp"]
        len_out = hyperparam["len_out"]

        task_helper = gen_prepare_task(
            len_inp=len_inp, 
            len_out=len_out,
            return_lag=return_lag, len_pred_show=len_pred_show, 
            skip=skip, plot_gap=plot_gap
        )
        task, helper = task_helper

        features_train, log_prices_train, _, _ = task[0]
        train_dataset, pred_dataset = task[1]
        all_date_pred, true_date, true_price, missing_data, convert_date = helper 

        # Used For Training
        train_dataset_list.append(train_dataset)
        algo_hyper_class_list.append((hyperparam, algo_class))
        
        # Used For Prediction
        pred_dataset_list.append(pred_dataset)
        len_pred_list.append(len(all_date_pred))
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
    
    model = IndependentMultiModel(
        train_dataset_list, 
        algo_hyper_class_list,
        num_model=len(exp_setting)
    )

    if load_path is not None:
        if is_load:
            model.load(f"save/{load_path}")
        elif is_save:
            model.train()
            model.save(load_path)
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

    fig, axes = plt.subplots(nrows=len(exp_setting), figsize=(15, 6))
    for i in range(len(exp_setting)):
        model_pred = DisplayPrediction(
            pred[i], name=algo_name, color="p", is_bridge=False
        )

        fig, ax1 = visualize_time_series(
            (fig, axes[i]), 
            ((full_feature_list[i], log_prices_train_list[i]), [true_pred_list[i], model_pred]), 
            "k", missing_data_list[i], "o", title="Log Lag over Time"
        )

    fig.tight_layout()        
    plt.show()

def example_plot_walk_forward(algo_name, base_name=None):
    hyperparam, algo_class = algorithms_dic[algo_name]

    return_lag = 22
    skip = 100
    len_inp = hyperparam["len_inp"]
    len_out = hyperparam["len_out"]
    features, log_prices, first_day, len_data, convert_date = get_data_example(return_lag, skip)
 
    metric = PerformanceMetric()

    if base_name is not None:
        fold_result = load_fold_data(base_name, algo_name)
        show_result_fold(fold_result, algo_name)
    else:
        fold_result = walk_forward(
            features, log_prices, algo_class, hyperparam, 
            metric.square_error, 
            size_train=300, size_test=200, 
            train_offset=1, 
            test_offset=len_out, 
            return_lag=return_lag, 
            convert_date=convert_date,
            is_train_pad=True, 
            is_test_pad=False 
        )
        save_fold_data(fold_result, algo_name)

    # Since we added more datapoint to not waste data
    fig, ax = visualize_walk_forward(
        features, log_prices, fold_result, convert_date,
        lag_color="o", pred_color="b", below_err="r"
    )
    fig.savefig("img/walk_forward.png")
    plt.show()
