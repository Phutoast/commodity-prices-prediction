import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.data_preprocessing import load_transform_data, prepare_dataset, find_missing_data
from utils.data_structure import DisplayPrediction, Hyperparameters, pack_data
from utils.data_visualization import visualize_time_series
from utils.data_structure import TrainingPoint

from models.ARIMA import ARIMAModel

def example_ARIMA_simple_predtion_plot():
    """
    Showing the result of ARIMA model on simple prediction tasks. 
    This only works with return lag = 0
    """
    metal_type = "aluminium"

    # We add this to show that lagging works
    return_lag = 0
    len_inp = 400
    len_out = 22

    len_predict_show = 200

    features, log_prices = load_transform_data(metal_type, return_lag=return_lag) 
    len_data = len(features)

    first_day = features["Date"][0]
    features = features[["Date"]]

    cut_pt = len_data-len_predict_show

    splitted_data = prepare_dataset(
        features, first_day, log_prices, 
        len_inp=len_data-len_predict_show-return_lag, len_out=len_predict_show, return_lag=return_lag, 
        convert_date=True, is_rand=False, offset=-1, is_show_progress=False, num_dataset=-1, is_padding=True
    ) 
    features_train, log_prices_train, feature_test, log_prices_test = splitted_data[0]

    training_dataset = prepare_dataset(
        features_train, first_day, log_prices_train, 
        len_inp=len_inp, len_out=len_out, return_lag=return_lag, 
        convert_date=False, is_rand=False, offset=-1, is_show_progress=True, num_dataset=-1, is_padding=True
    ) 

    # We don't have any inputs now just prediction something !!
    show_dataset = prepare_dataset(
        feature_test, first_day, log_prices_test, 
        len_inp=0, len_out=len_predict_show, return_lag=0, 
        convert_date=False, is_rand=False, offset=-1, is_show_progress=False, num_dataset=-1, is_padding=True
    )

    missing_x, missing_y = find_missing_data(features, log_prices, log_prices_train, log_prices_test, first_day, return_lag)
    missing_data = (missing_x, missing_y)

    ARIMA_hyperparam1 = Hyperparameters(
        len_inp=len_inp, 
        len_out=len_out, 
        order=(10, 2, 5), 
        ind_span_pred=20
    )
    model1 = ARIMAModel(training_dataset, ARIMA_hyperparam1)
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
    ARIMA_pred1 = DisplayPrediction(pred, name="ARIMA", color="p")

    true_pred = DisplayPrediction(
        pack_data(show_data.label_out["Price"].to_list(), [], [], show_data.data_out["Date"].to_list()),
        name="True Value", is_bridge=True
    )

    visualize_time_series(
        ((features_train, log_prices_train), [true_pred, ARIMA_pred1]), "k", missing_data, "o", title="Log Lag over Time")
    plt.show()
