import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.data_preprocessing import load_transform_data, prepare_dataset, find_missing_data
from utils.data_structure import DisplayPrediction, Hyperparameters, pack_data
from utils.data_visualization import visualize_time_series
from utils.data_structure import TrainingPoint

from models.ARIMA import ARIMAModel
from models.Mean import IIDDataModel
from models.GP import FeatureGP
import itertools

from gpytorch import kernels

def get_data_example(return_lag):
    metal_type = "aluminium"

    features, log_prices = load_transform_data(metal_type, return_lag=return_lag) 
    log_prices = log_prices[["Price"]]
    features, log_prices = features.head(1000), log_prices.head(1000)
    len_data = len(features)

    first_day = features["Date"].iloc[0]
    features = features[["Date"]]
    return features, log_prices, first_day, len_data

algorithms_dic = {
    # The ind span pred should be the same as len_out
    "ARIMA": [Hyperparameters(
        len_inp=0, 
        len_out=10, 
        order=(10, 2, 5), 
    ), ARIMAModel],
    "Mean": [Hyperparameters(
        len_inp=0, 
        len_out=10, 
        dist="Gaussian"
    ), IIDDataModel],
    "GP": [Hyperparameters(
        len_inp=5, 
        len_out=1, 
        lr=0.1,
        optim_iter=200,
        jitter=1e-4,
        is_time_only=True,
        kernel=kernels.ScaleKernel(kernels.MaternKernel())
    ), FeatureGP],
}

def example_plot_all_algo_lag():
    algo_name = "GP"
    hyperparam, algo_class = algorithms_dic[algo_name]

    return_lag = 0
    len_inp = hyperparam["len_inp"]
    len_out = hyperparam["len_out"]
    len_predict_show = 200
    true_features, true_log_prices, true_first_day, true_len_data = get_data_example(0)
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
    
    missing_x = true_date[:len_inp+return_lag]
    missing_y = true_price[:len_inp+return_lag] 
    missing_data = (missing_x, missing_y)
    
    model = algo_class(train_dataset, hyperparam)
    model.train()
    pred = model.predict(pred_dataset, len(all_date_pred), all_date_pred, ci=0.9)
    ARIMA_pred = DisplayPrediction(pred, name=algo_name, color="p")

    true_pred = DisplayPrediction(
        pack_data(true_price[len_inp+return_lag:], [], [], true_date[len_inp+return_lag:]),
        name="True Value", is_bridge=False
    )
    visualize_time_series(
        ((features_train, log_prices_train), [true_pred, ARIMA_pred]), "k", missing_data, "o", title="Log Lag over Time")
    plt.show()



