import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from collections import defaultdict

from utils.data_preprocessing import parse_series_time

color = {
    "o": "#ff7500",
    "p": "#5a08bf",
    "b": "#0062b8",
    "k": "#1a1a1a",
    "g": "#20c406",
    "grey": "#ebebeb"
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
    
    if is_missing:
        x_missing = convert_date(missing_x)
        y_missing = convert_price(missing_y)
    
    cut_point = x_train[-1]

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(x_train, y_train, color=color[inp_color])
        

    for i, y_pred in enumerate(y_pred_list):
        data, plot_name, color_code, is_bridge = y_pred
        mean_pred, x_test = data["mean"], data["x"]

        if i == 0 and is_missing:
            ax.axvline(x_test[0], color=color[lag_color], linestyle='--', linewidth=0.5, dashes=(5, 0), alpha=0.2)
            ax.plot([x_missing[-1], x_test[0]], [y_missing[-1], mean_pred[0]], color[lag_color], linestyle="dashed")
            ax.axvspan(cut_point, x_test[0], color=color[lag_color], alpha=0.1)

        plot_bound(ax, data, color[color_code], plot_name)

        if is_bridge and (not is_missing): 
            ax.plot([x_train[-1], x_test[0]], [y_train[-1], mean_pred[0]], color[color_code], linewidth=1.5)

    if is_missing:
        ax.plot(x_missing, y_missing, color=color[lag_color], linestyle="dashed")
        ax.plot([x_train[-1], x_missing[0]], [y_train[-1], y_missing[0]], color[lag_color], linestyle="dashed")
        ax.axvline(cut_point, color=color[lag_color], linestyle='--', linewidth=0.5, dashes=(5, 0), alpha=0.2)
    else:
        ax.axvline(cut_point, color=color["k"], linestyle='--')

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid()
    ax.legend()

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.set_xlim(left=cut_point-500)

def plot_bound(ax, data, color, plot_name):
    """
    Plotting with graph with uncertainty 

    Args:
        ax: Main plotting axis
        data: Packed data
        color: Color of the line
        plot_name: Name of the line
    """
    mean_pred, upper_pred, lower_pred, x_test = data["mean"], data["upper"], data["lower"], data["x"]
    ax.fill_between(x_test, upper_pred, lower_pred, color=color, alpha=0.3)
    ax.plot(x_test, mean_pred, color, linewidth=1.5, label=plot_name)


def visualize_walk_forward(full_data_x, full_data_y, model_result, out_loss, cutting_index, num_test, first_day):
    convert_date = lambda x: x["Date"].to_list()
    convert_price = lambda x: x["Price"].to_list()

    x, _ = parse_series_time(convert_date(full_data_x), first_day)
    y = convert_price(full_data_y)

    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True, sharey=False)
    fig.suptitle("Walk Forward Validation Loss Visualization")

    ind = 0
    for i in num_test:
        indexes = cutting_index[ind:ind+i]
        loss_cut = out_loss[ind:ind+i]
        first_x = indexes[0][0]
        last_x = indexes[-1][1]
        
        axs[0].axvline(first_x, color=color["grey"], linestyle='-')
        axs[0].axvline(last_x, color=color["grey"], linestyle='-')
        axs[0].axvspan(first_x, last_x, color=color["grey"], alpha=0.4)

        # We will have to search for it :(
        loc_first = model_result.index[model_result["x"] == first_x].tolist()[0]
        loc_last = model_result.index[model_result["x"] == last_x].tolist()[0] 
        plot_bound(axs[0], model_result.iloc[loc_first:loc_last+1, :], color["b"], "Output")
        
        axs[1].axvline(first_x, color=color["grey"], linestyle='-')
        axs[1].axvline(last_x, color=color["grey"], linestyle='-')
        axs[1].axvspan(first_x, last_x, color=color["grey"], alpha=0.4)

        for (i_start, j_start), loss in zip(indexes, loss_cut):
            loc_bar = (i_start + j_start)/2
            width = loc_bar - i_start
            axs[1].bar(loc_bar, loss, width, color=color['o'])

        ind += i

    axs[0].plot(x, y, color=color['k'])

    axs[0].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0].grid()
    axs[1].grid()
    axs[0].set_xlim(left=0)
    
    axs[1].set_xlabel("Time Step")
    axs[0].set_ylabel("Log Prices")

    axs[1].set_ylabel("Square Loss")

    plt.show()
