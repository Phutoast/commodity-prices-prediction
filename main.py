import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import random

from utils.data_preprocessing import load_transform_data, walk_forward, prepare_dataset, find_missing_data
from utils.data_structure import DisplayPrediction, Hyperparameters, pack_data
from utils.data_visualization import visualize_time_series
from utils.calculation import PerformanceMetric
from models.ARIMA import ARIMAModel, BaseModel

from examples.ARIMA_plot import example_ARIMA_simple_predtion_plot

np.random.seed(48)
random.seed(48)

def main():
    # metal_type = "aluminium"
    # return_lag = 0

    # features, log_prices = load_transform_data(metal_type, return_lag=return_lag) 
    # first_day = features["Date"][0]

    # features = features[["Date"]]

    # training_dataset = prepare_dataset(
    #     features, first_day, log_prices, 
    #     len_inp=400, len_out=22, return_lag=return_lag, 
    #     convert_date=True, is_rand=False, offset=-1, 
    #     is_show_progress=True, num_dataset=2
    # ) 
    # X_train, y_train, X_test, y_test = training_dataset[1]
    # missing_x, missing_y = find_missing_data(features, log_prices, y_train, y_test, first_day, return_lag)
    # ARIMA_hyperparam1 = Hyperparameters(
    #     len_inp=400, 
    #     len_out=22, 
    #     order=(5, 2, 5), 
    #     ind_span_pred=10
    # )
    # model1 = ARIMAModel(training_dataset[1], ARIMA_hyperparam1)

    # metal_type = "aluminium"
    # return_lag = 0
    # len_inp = 3
    # len_out = 2

    # features, log_prices = load_transform_data(metal_type, return_lag=return_lag) 
    # first_day = features["Date"][0]
    
    # features = features[["Date"]]
    # features, log_prices = features.head(20), log_prices.head(20)

    # ARIMA_hyperparam1 = Hyperparameters(
    #     len_inp=len_inp, 
    #     len_out=len_out, 
    #     order=(5, 2, 5), 
    #     ind_span_pred=1
    # )

    # metric = PerformanceMetric()
    
    # # FOR ARIMA Experiment we will assume no offset at all
    # out_loss, _ = walk_forward(
    #     features, log_prices, ARIMAModel, ARIMA_hyperparam1, 
    #     metric.square_error, 
    #     size_train=11, size_test=5, train_offset=-1, test_offset=1, 
    #     test_step=2, return_lag=return_lag, is_train_pad=True, is_test_pad=True, 
    #     is_rand=False
    # )
    # print(out_loss)

    example_ARIMA_simple_predtion_plot()

if __name__ == '__main__':
    main()
