import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import random

from utils.data_preprocessing import load_transform_data, walk_forward, prepare_dataset
from utils.data_structure import Prediction, Hyperparameters
from utils.data_visualization import visualize_time_series
from utils.calculation import PerformanceMetric
from models.ARIMA import ARIMAModel, BaseModel

np.random.seed(48)
random.seed(48)

def main():
    metal_type = "aluminium"
    return_lag = 22

    features, log_prices = load_transform_data(metal_type, return_lag=return_lag)
    first_day = features["Date"][0]

    # Starting with the Most Rudimentory data
    features = features[["Date"]]
    # features = features.head(40)
    # log_prices = log_prices.head(40)

    prepare_dataset(features, first_day, log_prices, 5, len_out=2, return_lag=return_lag, convert_date=True, 
                        is_rand=False, offset=1, relative_time=False, is_show_progress=False)
    # test_hyper = Hyperparameters(len_inp=5, len_out=2, wtf=48, abc=3)
    # perf_metric = PerformanceMetric()
    # walk_forward(
    #     features, log_prices, 
    #     BaseModel, test_hyper, 
    #     perf_metric.dummy_loss, 
    #     size_train=13, size_test=8, 
    #     train_offset=2, test_offset=1, test_step=2
    # )


    # pd.plotting.autocorrelation_plot(log_prices)
    # price.plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
    # plt.show()
    
    # start = 500
    # cut = len(features)-100

    # x_train, y_train = features[start:cut], log_prices[start:cut]
    # x_test, y_test = features[cut:], log_prices[cut:]

    # test_model = ARIMAModel(x_train, y_train, (5, 2, 5))
    # y_pred_1 = test_model.predict(x_test, y_test, 100)
    # y_pred_2 = test_model.predict(x_test, y_test, 5)

    # vis_data = ((features, log_prices), [y_pred_1, y_pred_2])
    # visualize_time_series(vis_data, start, cut, ('k', ['r', 'b']), metal_type)

if __name__ == '__main__':
    main()
