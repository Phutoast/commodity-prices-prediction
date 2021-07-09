import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from utils.data_preprocessing import load_transform_data
from utils.data_structure import DisplayPrediction, pack_data
from utils.data_visualization import visualize_time_series, visualize_walk_forward, show_result_fold
from utils.others import create_folder, save_fold_data, load_fold_data

from experiments.algo_dict import algorithms_dic
from experiments.eval_methods import prepare_dataset, walk_forward
from experiments.calculation import PerformanceMetric

def get_data_example(return_lag):
    metal_type = "aluminium"

    features, log_prices = load_transform_data(metal_type, return_lag=return_lag) 
    log_prices = log_prices[["Price"]]
    features, log_prices = features.head(1000), log_prices.head(1000)
    len_data = len(features)

    first_day = features["Date"].iloc[0]
    # features = features[["Date", "FeatureFamily.TECHNICAL"]]
    features = features[["Date"]]
    return features, log_prices, first_day, len_data

def example_plot_all_algo_lag(algo_name, plot_gap=True, load_path=None, is_save=True, is_load=False):
    hyperparam, algo_class = algorithms_dic[algo_name]

    return_lag = 0
    len_inp = hyperparam["len_inp"]
    len_out = hyperparam["len_out"]
    len_predict_show = 100
    features, log_prices, first_day, len_data = get_data_example(return_lag)
    splitted_data = prepare_dataset(
        features, first_day, log_prices, 
        len_inp=len_data-len_predict_show-return_lag, len_out=len_predict_show, return_lag=0, 
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
    
    all_date_pred = list(itertools.chain.from_iterable([point.data_out["Date"].to_list() for point in pred_dataset]))
    true_date = feature_test["Date"].to_list()
    true_price = log_prices_test["Price"].to_list() 
    
    if plot_gap:
        missing_x = true_date[:len_inp+return_lag]
        missing_y = true_price[:len_inp+return_lag] 
        missing_data = (missing_x, missing_y)
    else:
        missing_data = ([], [])
    
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

    ARIMA_pred = DisplayPrediction(pred, name=algo_name, color="p", is_bridge=False)

    true_pred = DisplayPrediction(
        pack_data(true_price[len_inp+return_lag:], [], [], true_date[len_inp+return_lag:]),
        name="True Value", is_bridge=False
    )
    visualize_time_series(
        ((features_train, log_prices_train), [true_pred, ARIMA_pred]), "k", missing_data, "o", title="Log Lag over Time")
        
    plt.show()


def example_plot_walk_forward(algo_name, base_name=None):
    hyperparam, algo_class = algorithms_dic[algo_name]

    return_lag = 22
    len_inp = hyperparam["len_inp"]
    len_out = hyperparam["len_out"]
    features, log_prices, first_day, len_data = get_data_example(return_lag)
        
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
            is_train_pad=True, 
            is_test_pad=False 
        )
        save_fold_data(fold_result, algo_name)

    fig, ax = visualize_walk_forward(
        features, log_prices, fold_result, 
        lag_color="o", pred_color="b", below_err="r"
    )
    fig.savefig("img/walk_forward.png")
    plt.show()
