import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from utils.data_preprocessing import parse_series_time

color = {
    "o": "#ff7500",
    "p": "#5a08bf",
    "b": "#0b66b5",
    "k": "#1a1a1a",
    "g": "#20c406"
}

def visualize_time_series(data, inp_color, metal_type):
    """
    Ploting out the time series, given each time step label to be stored in the DataFrame

    Args:
        data: The time series data collected in tuple form ((x_train, y_train), [pred1, pred2, dots])
        colors: Tuple of correct color and prediction color for presenting.
        label: The label for each time series step (which will be plotted).
    """
    ((x_train, y_train), y_pred_list) = data

    convert_date = lambda x: x["Date"].to_list()
    convert_price = lambda x: x["Price"].to_list()

    x_train = convert_date(x_train)
    y_train = convert_price(y_train)

    cut_point = x_train[-1]

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(x_train, y_train, f'{inp_color}-')

    for y_pred in y_pred_list:
        data, plot_name, color_code, is_bridge = y_pred
        mean_pred, upper_pred, lower_pred, x_test = data["mean"], data["upper"], data["lower"], data["x"]

        ax.fill_between(x_test, upper_pred, lower_pred, color=color[color_code], alpha=0.25)
        ax.plot(x_test, mean_pred, color[color_code], linewidth=1.5, label=plot_name)

        if is_bridge:
            ax.plot([x_train[-1], x_test[0]], [y_train[-1], mean_pred[0]], color[color_code], linewidth=1.5)

    ax.axvline(cut_point, color='k', linestyle='--')
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.grid()
    ax.legend()

    ax.set_xlabel(f"Number of Days")
    ax.set_ylabel(f"Log of prices {metal_type}")
    ax.set_xlim(left=cut_point-100)

    plt.show()