import numpy as np
import pandas as pd

from utils.data_preprocessing import load_transform_data, prepare_dataset
from utils.data_structure import DisplayPrediction, Hyperparameters, pack_data
from utils.data_visualization import visualize_time_series
from models.ARIMA import ARIMAModel

def example_ARIMA_simple_plot():
    metal_type = "aluminium"
    return_lag = 0

    features, log_prices = load_transform_data(metal_type, return_lag=return_lag)
    
    first_day = features["Date"][0]
    features = features[["Date"]]

    training_dataset = prepare_dataset(
        features, first_day, log_prices, 
        len_inp=400, len_out=22, return_lag=return_lag, 
        convert_date=True, is_rand=False, offset=-1, 
        relative_time=False, is_show_progress=True, num_dataset=2
    ) 

    X_train, y_train, X_test, y_test = training_dataset[0]
    ARIMA_hyperparam1 = Hyperparameters(
        len_inp=1000, 
        len_out=22, 
        order=(5, 2, 5), 
        ind_span_pred=5
    )
    model1 = ARIMAModel(training_dataset[0], ARIMA_hyperparam1)
    model1.train()
    ARIMA_pred1 = DisplayPrediction(model1.predict(training_dataset[0], 22, ci=0.9), name="ARIMA", color="p")
    true_pred = DisplayPrediction(pack_data(y_test["Price"].to_list(), [], [], X_test["Date"].to_list()), name="True Value")

    visualize_time_series(
        ((X_train, y_train), [true_pred, ARIMA_pred1]), "k", metal_type)
