import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from utils.data_preprocessing import load_transform_data, parse_series_time
from utils.data_structure import DisplayPrediction, pack_data
from utils.data_visualization import visualize_time_series, visualize_walk_forward, show_result_fold
from utils.others import create_folder, save_fold_data, load_fold_data

from experiments.algo_dict import algorithms_dic
from experiments.eval_methods import prepare_dataset, walk_forward
from experiments.calculation import PerformanceMetric

def get_data_example(return_lag, skip):
    metal_type = "aluminium"

    features, log_prices = load_transform_data(metal_type, return_lag, skip) 
    log_prices = log_prices[["Price"]]

    # We will have 1000 datapoints for training and testing
    features, log_prices = features.head(1000+skip), log_prices.head(1000+skip)
    len_data = len(features)

    first_day = features["Date"].iloc[0]
    # features = features[["Date", "FeatureFamily.TECHNICAL"]]
    features = features[["Date"]]
    return features, log_prices, first_day, len_data

def pred_date_conversion(return_lag, skip):
    """
    The given a skip, the date of the prediction of the model has to 
        be added forward to match the real time
    """
    no_skip_features, _, first_day, _ = get_data_example(return_lag, 0)
    no_skip_date, _ = parse_series_time(no_skip_features["Date"].to_list(), first_day)
    
    skip_features, _, first_day2, _ = get_data_example(return_lag, skip)
    skip_date, _ = parse_series_time(skip_features["Date"].to_list(), first_day2)
    skip_date = skip_date[skip:]

    return dict(zip(no_skip_date, skip_date))


def create_task(len_inp, len_out, return_lag, len_pred_show, skip):
    features, log_prices, first_day, len_data = get_data_example(return_lag, skip)
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
    return (features_train, log_prices_train, feature_test, log_prices_test), (train_dataset, pred_dataset)

def prepare_task(task, len_inp, return_lag, skip, plot_gap):
    _, _, feature_test, log_prices_test = task[0]
    _, pred_dataset = task[1]

    convert_date = pred_date_conversion(return_lag, skip)
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

def example_plot_all_algo_lag(algo_name, plot_gap=True, load_path=None, is_save=True, is_load=False):
    hyperparam, algo_class = algorithms_dic[algo_name]

    return_lag = 22
    skip = 4
    len_inp = hyperparam["len_inp"]
    len_out = hyperparam["len_out"]
    len_pred_show = 100

    task = create_task(len_inp, len_out, return_lag, len_pred_show, skip)
    helper = prepare_task(task, len_inp, return_lag, skip, plot_gap)

    features_train, log_prices_train, feature_test, log_prices_test = task[0]
    train_dataset, pred_dataset = task[1]
    all_date_pred, true_date, true_price, missing_data, convert_date = helper

    if load_path is not None and is_load:
        model = algo_class([], hyperparam)
        parent, file_name = load_path
        model.load(f"save/{parent}/{file_name}")
    else:
        model = algo_class(train_dataset, hyperparam)
        model.train()

    if load_path is not None and is_save:
        parent, file_name = load_path
        create_folder(f"save/{parent}")
        model.save(f"save/{parent}/{file_name}")
    
    pred = model.predict(pred_dataset, len(all_date_pred), all_date_pred, ci=0.9)
    model_pred = DisplayPrediction(pred, name=algo_name, color="p", is_bridge=False)

    true_pred = DisplayPrediction(
        pack_data(true_price[len_inp+return_lag:], [], [], true_date[len_inp+return_lag:]),
        name="True Value", is_bridge=False
    )

    full_feature = features_train.copy()
    full_feature["Date"] = full_feature["Date"].map(convert_date)

    visualize_time_series(
        ((full_feature, log_prices_train), [true_pred, model_pred]), 
        "k", missing_data, "o", title="Log Lag over Time"
    )
        
    plt.show()


def example_plot_walk_forward(algo_name, base_name=None):
    hyperparam, algo_class = algorithms_dic[algo_name]

    return_lag = 22
    skip = 10
    len_inp = hyperparam["len_inp"]
    len_out = hyperparam["len_out"]
    features, log_prices, first_day, len_data = get_data_example(return_lag, skip)
    convert_date = pred_date_conversion(return_lag, skip)
 
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
        features.iloc[:len(features)-skip], log_prices, fold_result, convert_date,
        lag_color="o", pred_color="b", below_err="r"
    )
    fig.savefig("img/walk_forward.png")
    plt.show()


