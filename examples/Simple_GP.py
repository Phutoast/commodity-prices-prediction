import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from utils.data_preprocessing import load_transform_data, walk_forward, prepare_dataset, find_missing_data
from utils.data_structure import DisplayPrediction, Hyperparameters, pack_data
from utils.data_visualization import visualize_time_series
from utils.data_structure import TrainingPoint
from utils.calculation import PerformanceMetric

from models.GP import SimpleGaussianProcessModel, FeatureGP

def simple_GP_plot():
    metal_type = "aluminium"

    # We add this to show that lagging works
    return_lag = 0
    len_inp = 1
    len_out = 1

    len_predict_show = 200

    features, log_prices = load_transform_data(metal_type, return_lag=return_lag) 
    log_prices = log_prices[["Price"]]
    features, log_prices = features.tail(1000), log_prices.tail(1000)
    len_data = len(features)

    first_day = features["Date"].iloc[0]
    features = features[["Date"]]

    splitted_data = prepare_dataset(
        features, first_day, log_prices, 
        len_inp=len_data-len_predict_show-return_lag, len_out=len_predict_show, return_lag=return_lag, 
        convert_date=True, offset=-1, is_show_progress=False, num_dataset=-1, is_padding=True
    ) 
    features_train, log_prices_train, feature_test, log_prices_test = splitted_data[0]

    training_dataset = prepare_dataset(
        features_train, first_day, log_prices_train, 
        len_inp=len_inp, len_out=0, return_lag=return_lag, 
        convert_date=False, offset=-1, is_show_progress=False, num_dataset=-1, is_padding=True
    ) 

    # We don't have any inputs now just prediction something !!
    show_dataset = prepare_dataset(
        feature_test, first_day, log_prices_test, 
        len_inp=0, len_out=len_predict_show, return_lag=0, 
        convert_date=False, offset=-1, is_show_progress=False, num_dataset=-1, is_padding=True
    )

    missing_x, missing_y = find_missing_data(features, log_prices, log_prices_train, log_prices_test, first_day, return_lag)
    missing_data = (missing_x, missing_y)

    ARIMA_hyperparam1 = Hyperparameters(
        len_inp=len_inp, 
        len_out=len_out, 
        lr=0.1,
        optim_iter=0,
    )
    model1 = SimpleGaussianProcessModel(training_dataset, ARIMA_hyperparam1)
    model1.train()

    # This is special for ARIMA as we may consider the lagging
    # But it doesn't care about the time step so, we simply shift it.
    # prediction_inp = training_dataset[1]
    # if return_lag > 0:
    #     prediction_inp = TrainingPoint(
    #         training_dataset[1].data_inp, 
    #         training_dataset[1].label_inp,
    #         training_dataset[1].data_out,
    #         missing_y
    #     )

    show_data = show_dataset[0]
    # missing_data = ([], [])

    pred = model1.predict(show_data, len_predict_show, ci=0.9)
    ARIMA_pred1 = DisplayPrediction(pred, name="GP Simple", color="p")

    true_pred = DisplayPrediction(
        pack_data(show_data.label_out["Price"].to_list(), [], [], show_data.data_out["Date"].to_list()),
        name="True Value", is_bridge=False
    )

    visualize_time_series(
        ((features_train, log_prices_train), [true_pred, ARIMA_pred1]), "k", missing_data, "o", title="Log Lag over Time")
    plt.show()

def feature_GP_plot():
    metal_type = "aluminium"

    # We add this to show that lagging works
    return_lag = 0
    len_inp = 5
    len_out = 1

    len_predict_show = 200

    features, log_prices = load_transform_data(metal_type, return_lag=return_lag) 
    # Calculating Not-None Data
    # print(sum(features.isna().sum(axis=1) == 0))
    features, log_prices = features.head(1000), log_prices.head(1000)
    log_prices = log_prices[["Price"]]
    len_data = len(features)
    # print(log_prices.head(10))

    first_day = features["Date"].iloc[0]
    features = features[["Date", "FeatureFamily.TECHNICAL"]]
    features = features[["Date"]]

    splitted_data = prepare_dataset(
        features, first_day, log_prices, 
        len_inp=len_data-len_predict_show-return_lag, len_out=len_predict_show, return_lag=return_lag, 
        convert_date=True, offset=-1, is_show_progress=False, num_dataset=-1, is_padding=True
    ) 
    features_train, log_prices_train, feature_test, log_prices_test = splitted_data[0]

    training_dataset = prepare_dataset(
        features_train, first_day, log_prices_train, 
        len_inp=len_inp, len_out=len_out, return_lag=return_lag, 
        convert_date=False, offset=1, is_show_progress=False, num_dataset=-1, is_padding=True
    ) 

    # We don't have any inputs now just prediction something !!
    show_dataset = prepare_dataset(
        feature_test, first_day, log_prices_test, 
        len_inp=len_inp, len_out=len_out, return_lag=0, 
        convert_date=False, offset=1, is_show_progress=False, num_dataset=-1, is_padding=True
    )
    
    full_show_dataset = prepare_dataset(
        feature_test, first_day, log_prices_test, 
        len_inp=0, len_out=len_predict_show, return_lag=0, 
        convert_date=False, offset=-1, is_show_progress=False, num_dataset=-1, is_padding=True
    )

    # print(show_dataset[0])
    # assert False

    missing_x, missing_y = find_missing_data(features, log_prices, log_prices_train, log_prices_test, first_day, return_lag)
    missing_data = (missing_x, missing_y)

    ARIMA_hyperparam1 = Hyperparameters(
        len_inp=len_inp, 
        len_out=len_out, 
        lr=0.1,
        optim_iter=100,
        jitter=1e-4
    )
    model1 = FeatureGP(training_dataset, ARIMA_hyperparam1)
    model1.train()

    # This is special for ARIMA as we may consider the lagging
    # But it doesn't care about the time step so, we simply shift it.
    # prediction_inp = training_dataset[1]
    # if return_lag > 0:
    #     prediction_inp = TrainingPoint(
    #         training_dataset[1].data_inp, 
    #         training_dataset[1].label_inp,
    #         training_dataset[1].data_out,
    #         missing_y
    #     )

    show_data = show_dataset
    # missing_data = ([], [])

    pred = model1.predict(show_data, len_predict_show-len_inp, ci=0.9)
    ARIMA_pred1 = DisplayPrediction(pred, name="GP Simple", color="p")

    true_pred = DisplayPrediction(
        pack_data(full_show_dataset[0].label_out["Price"].to_list(), [], [], full_show_dataset[0].data_out["Date"].to_list()),
        name="True Value", is_bridge=False
    )

    visualize_time_series(
        ((features_train, log_prices_train), [true_pred, ARIMA_pred1]), "k", missing_data, "o", title="Log Lag over Time")
    plt.show()
