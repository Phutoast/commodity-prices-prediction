import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

from utils.data_preprocessing import data_to_date_label
from utils.data_structure import Prediction
from models.ARIMA import ARIMAModel

def load_metal_data(metal_type):
    """
    Loading the metal data (both feature and *raw* prices). 
    The files will be stores given the path: 
        data/{metal_type}/{metal_type}_features.xlsx
        data/{metal_type}/{metal_type}_raw_prices.xlsx

    Args:
        metal_type: type of the matal, we want to import
    Returns
        features: the feature of the related to the metal
        prices: the prices of metal without any preprocessing 
    """

    feature_path = f"data/{metal_type}/{metal_type}_features_true.csv"
    price_path = f"data/{metal_type}/{metal_type}_raw_prices.csv"

    feature = pd.read_csv(feature_path)

    # Rename column 
    columns = list(feature.columns)
    columns[0] = "Date"
    feature.columns = columns
    feature = feature.drop([0,1,2]).reset_index(drop=True)

    price = pd.read_csv(price_path)
    price.columns = ["Date", "Price"]

    # Adding Missing Dates
    diff_date = set(price['Date'])- set(feature['Date']) 
    original_len = len(feature)    
    for i, date in enumerate(diff_date):
        feature = feature.append(pd.Series(), ignore_index=True)
        feature["Date"][original_len+i] = date
    
    feature = feature.sort_values(by=["Date"]).reset_index(drop=True)
     
    return pd.merge(feature, price, on="Date")

def transform_full_data(full_data, is_drop_nan=False):
    """
    Given the concatenated data, we transform 
        and clean the data with splitting the feature and price.

    Args:
        full_data: Concatenated data 

    Returns:
        x: Feature over time. 
        y: (Log)-Price over time.
    """

    if is_drop_nan:
        full_data = full_data.dropna()
    
    full_data["Price"] = np.log(full_data["Price"])
    return full_data.iloc[:, :-1], full_data.iloc[:, -1]

def visualize_time_series(data, start, cut, colors, metal_type):
    """
    Ploting out the time series, given each time step label to be stored in the DataFrame

    Args:
        data: The time series data collected in tuple form ((x_train, y_train), (x_test, y_test, y_pred))
        colors: Tuple of correct color and prediction color for presenting.
        label: The label for each time series step (which will be plotted).
    """

    corr_color, pred_color = colors
    ((x, y), y_pred_list) = data

    x, _ = data_to_date_label(x)
    cut_point = x[cut]

    x_train, y_train = x[start:cut], y[start:cut]
    x_test, y_test = x[cut:], y[cut:]

    assert len(pred_color) == len(y_pred_list)

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(x_train, y_train, f'{corr_color}-')
    ax.plot(x_test, y_test, f'{corr_color}-')

    for color, y_pred in zip(pred_color, y_pred_list):
        mean_pred, upper_pred, lower_pred = y_pred
        ax.fill_between(x_test, upper_pred, lower_pred, color=color, alpha=0.3)
        ax.plot(x_test, mean_pred, f'{color}-')

    ax.axvline(cut_point, color='k', linestyle='--')
    ax.grid()

    ax.set_xlabel(f"Number of Days")
    ax.set_ylabel(f"Log of prices {metal_type}")
    ax.set_xlim(left=cut_point-100)

    plt.show()

def main():
    metal_type = "aluminium"

    data_all = load_metal_data(metal_type)
    date_price, log_price = transform_full_data(data_all)

    # pd.plotting.autocorrelation_plot(log_price)
    # price.plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
    # plt.show()
 
    time_step, label = data_to_date_label(date_price)
    
    start = 500
    cut = len(date_price)-100

    x_train, y_train = date_price[start:cut], log_price[start:cut]
    x_test, y_test = date_price[cut:], log_price[cut:]

    test_model = ARIMAModel(x_train, y_train, (5, 2, 5))
    y_pred_1 = test_model.predict(x_test, y_test, 100)
    y_pred_2 = test_model.predict(x_test, y_test, 20)

    vis_data = ((date_price, log_price), [y_pred_1, y_pred_2])
    visualize_time_series(vis_data, start, cut, ('k', ['r', 'b']), metal_type)

if __name__ == '__main__':
    main()
