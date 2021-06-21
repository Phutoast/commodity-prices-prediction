import matplotlib.pyplot as plt
from utils.data_preprocessing import parse_series_time

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

    x, _ = parse_series_time(x["Date"].to_list())
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