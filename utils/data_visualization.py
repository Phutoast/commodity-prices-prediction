import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from collections import defaultdict

from utils.data_preprocessing import parse_series_time

color = {
    "o": "#ff7500",
    "p": "#5a08bf",
    "b": "#0b66b5",
    "k": "#1a1a1a",
    "g": "#20c406"
}
color = defaultdict(lambda:"#1a1a1a", color)

def visualize_time_series(data, inp_color, missing_data, lag_color,
    x_label="Number of Days", y_label="Log of Aluminium Price", title="Prices over time"):
    """
    Ploting out the time series, given each time step label to be stored in the DataFrame

    Args:
        data: Data that is useful for plotting the result, 
            as it has the structure of ((x_train, y_train), y_pred_list)
            Note that we doesn't include the "correct" value of x_train, y_train, 
            user will have to add them by themselves
        inp_color: Color of the  x_train, y_train plot.
        lag: Number of lags in the training data ()
        missing_data: The data that we have discarded due to calculating log-lagged return.
            The reason to not inlcude in the main data is because of interface 
            + make suring we won't get confused. 
            If there is missing data the bridge between training an testing will be ignored.
        x_label: Label of x-axis
        y_label: Label of y-axis
        title: Plot title 
    """
    ((x_train, y_train), y_pred_list) = data

    missing_x, missing_y = missing_data
    is_missing = len(missing_x) != 0

    convert_date = lambda x: x["Date"].to_list()
    convert_price = lambda x: x["Price"].to_list()

    x_train = convert_date(x_train)
    y_train = convert_price(y_train)

    x_missing = convert_date(missing_x)
    y_missing = convert_price(missing_y)
    
    cut_point = x_train[-1]

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(x_train, y_train, color=color[inp_color])
        

    for i, y_pred in enumerate(y_pred_list):
        data, plot_name, color_code, is_bridge = y_pred
        mean_pred, upper_pred, lower_pred, x_test = data["mean"], data["upper"], data["lower"], data["x"]
        
        if i == 0 and is_missing:
            ax.axvline(x_test[0], color=color[lag_color], linestyle='--', linewidth=0.5, dashes=(5, 0), alpha=0.2)
            ax.plot([x_missing[-1], x_test[0]], [y_missing[-1], mean_pred[0]], color[lag_color], linestyle="dashed")
            ax.axvspan(cut_point, x_test[0], color=color[lag_color], alpha=0.1)

        ax.fill_between(x_test, upper_pred, lower_pred, color=color[color_code], alpha=0.25)
        ax.plot(x_test, mean_pred, color[color_code], linewidth=1.5, label=plot_name)

        if is_bridge and (not is_missing): 
            ax.plot([x_train[-1], x_test[0]], [y_train[-1], mean_pred[0]], color[color_code], linewidth=1.5)

    if is_missing:
        ax.plot(x_missing, y_missing, color=color[lag_color], linestyle="dashed")
        ax.plot([x_train[-1], x_missing[0]], [y_train[-1], y_missing[0]], color[lag_color], linestyle="dashed")
        ax.axvline(cut_point, color=color[lag_color], linestyle='--', linewidth=0.5, dashes=(5, 0), alpha=0.2)
    else:
        ax.axvline(cut_point, color=color["k"], linestyle='--')

    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.grid()
    ax.legend()

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.set_xlim(left=cut_point-100)
