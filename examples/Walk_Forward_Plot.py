import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from utils.data_preprocessing import load_transform_data, walk_forward, prepare_dataset, find_missing_data
from utils.data_structure import DisplayPrediction, Hyperparameters, pack_data
from utils.data_visualization import visualize_walk_forward
from utils.data_structure import TrainingPoint
from utils.calculation import PerformanceMetric

from models.ARIMA import ARIMAModel

def example_ARIMA_walk_forward_plot():
    metal_type = "aluminium"
    return_lag = 0

    # What model normally test
    len_inp = 0
    
    # What model normally predict
    len_out = 10

    features, log_prices = load_transform_data(metal_type, return_lag=return_lag) 
    log_prices = log_prices[["Price"]]
    first_day = features["Date"][0]
    
    features = features[["Date"]]
    features, log_prices = features.head(1300), log_prices.head(1300)

    ARIMA_hyperparam1 = Hyperparameters(
        len_inp=len_inp, 
        len_out=len_out, 
        order=(5, 2, 5), 
        ind_span_pred=10
    )

    metric = PerformanceMetric()
    
    # FOR ARIMA Experiment we will assume no offset at all

    # Rules for Perfect Walkforward
    # (Size of features - size_train) should be divisible by size_test
    # (size_test - len_inp) should be divisible by 
    #   len_out + test_offset & test_step should be equal to len_out
    
    def save_data():
        out_loss, num_test, (model_result, cutting_index) = walk_forward(
            features, log_prices, ARIMAModel, ARIMA_hyperparam1, 
            metric.square_error, 
            size_train=300, size_test=200, train_offset=-1, test_offset=20, 
            test_step=20, return_lag=return_lag, is_train_pad=True, is_test_pad=False, 
            is_rand=False
        )

        model_result.to_csv("cache/test.csv")
        with open("cache/cutting_index.txt", 'wb') as f:
            pickle.dump(cutting_index, f)

        with open("cache/num_test.txt", 'wb') as f:
            pickle.dump(num_test, f)

        with open("cache/out_loss.txt", 'wb') as f:
            pickle.dump(out_loss, f)

    # save_data()

    with open("cache/cutting_index.txt", 'rb') as f:
        cutting_index = pickle.load( f)
    
    with open("cache/num_test.txt", 'rb') as f:
        num_test = pickle.load(f)
    
    with open("cache/out_loss.txt", 'rb') as f:
        out_loss = pickle.load(f)
    
    print("Cutting Index", cutting_index)
    print("Num Test", num_test)
    print("Out loss", out_loss)

    model_result = pd.read_csv("cache/test.csv", index_col=0) 
    visualize_walk_forward(features, log_prices, model_result, out_loss, cutting_index, num_test, first_day)