import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from collections import defaultdict
import pandas as pd

from utils.data_preprocessing import parse_series_time
from scipy.interpolate import make_interp_spline
from models.ind_multi_model import IndependentMultiModel

color = {
    "o": "#ff7500",
    "p": "#5a08bf",
    "b": "#0062b8",
    "k": "#1a1a1a",
    "g": "#20c406",
    "grey": "#ebebeb",
    "r": "#d6022a"
}
color = defaultdict(lambda:"#1a1a1a", color)

def visualize_time_series(fig_ax, data, inp_color, missing_data, lag_color,
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
    fig, ax = fig_ax
    ((x_train, y_train), y_pred_list) = data

    missing_x, missing_y = missing_data
    is_missing = len(missing_x) != 0

    convert_date = lambda x: x["Date"].to_list()
    convert_price = lambda x: x["Output"].to_list()

    x_train = convert_date(x_train)
    y_train = convert_price(y_train)
    
    cut_point = x_train[-1]
    ax.plot(x_train, y_train, color=color[inp_color])

    for i, y_pred in enumerate(y_pred_list):
        data, plot_name, color_code, is_bridge = y_pred
        mean_pred, x_test = data["mean"], data["x"]

        if i == 0 and is_missing:
            ax.axvline(x_test[0], color=color[lag_color], linestyle='--', linewidth=0.5, dashes=(5, 0), alpha=0.2)
            ax.plot([missing_x[-1], x_test[0]], [missing_y[-1], mean_pred[0]], color[lag_color], linestyle="dashed")
            ax.axvspan(cut_point, x_test[0], color=color[lag_color], alpha=0.1)

        plot_bound(ax, data, color[color_code], plot_name)

        if is_bridge and (not is_missing): 
            ax.plot([x_train[-1], x_test[0]], [y_train[-1], mean_pred[0]], color[color_code], linewidth=1.5)

    if is_missing:
        ax.plot(missing_x, missing_y, color=color[lag_color], linestyle="dashed")
        ax.plot([x_train[-1], missing_x[0]], [y_train[-1], missing_y[0]], color[lag_color], linestyle="dashed")
        ax.axvline(cut_point, color=color[lag_color], linestyle='--', linewidth=0.5, dashes=(5, 0), alpha=0.2)
    else:
        ax.axvline(cut_point, color=color["k"], linestyle='--')

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid()
    ax.legend()

    # ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.set_xlim(left=cut_point-500)
    return fig, ax

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

def plot_area(axs, x, y, miss, start_ind, end_ind, lag_color):
    missing_x, missing_y = miss
    axs[0].plot(
        missing_x, missing_y, 
        color=color[lag_color], 
        linestyle="dashed", alpha=0.6
    )
    axs[0].axvline(
        x[start_ind], color=color[lag_color], 
        linestyle='--', linewidth=1.5, dashes=(5, 0), 
        alpha=0.2
    )
    axs[0].axvline(
        x[end_ind], color=color[lag_color], 
        linestyle='--', linewidth=1.5, dashes=(5, 0), 
        alpha=0.2
    )
    axs[0].axvspan(x[start_ind], x[end_ind], color=color[lag_color], alpha=0.1)
    axs[0].plot(
        [x[start_ind], missing_x[0]], [y[start_ind], missing_y[0]], 
        color[lag_color], linestyle="dashed"
    )
    axs[0].plot(
        [missing_x[-1], x[end_ind]], [missing_y[-1], y[end_ind]], 
        color[lag_color], linestyle="dashed"
    )


def visualize_walk_forward(full_data_x, full_data_y, fold_result, convert_date_dict, lag_color="o", pred_color="p", below_err="g", title="Walk Forward Validation Loss Visualization"):

    convert_date = lambda x: x["Date"].to_list()
    convert_price = lambda x: x["Output"].to_list()
    first_day = full_data_x["Date"][0] 

    x, _ = parse_series_time(convert_date(full_data_x), first_day)
    x = list(map(lambda a: convert_date_dict[a], x))
    y = convert_price(full_data_y)
    get_first_day = lambda df: df["x"][0]

    _, (miss_x, miss_y), _, _ = fold_result[0]
    is_missing = len(miss_x) != 0

    if is_missing:
        day_plot = (
            0, 0, miss_x[0], x.index(miss_x[0])
        )

    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(nrows=2, hspace=0)
    axs = gs.subplots(sharex=True, sharey=False)
    fig.suptitle(title)

    if is_missing:
        axs[0].plot(
            x[day_plot[1]:day_plot[3]], 
            y[day_plot[1]:day_plot[3]], 
            color=color["k"], linestyle='-'
        )


    for i, (pred, missing_data, intv_loss, _) in enumerate(fold_result):
        first_day = pred["x"].iloc[0]
        first_index = x.index(first_day)
        last_day = pred["x"].iloc[-1]
        last_index = x.index(last_day)

        if not is_missing:
            if i == 0:
                axs[0].plot(
                    x[:first_index], 
                    y[:first_index], 
                    color=color["k"], linestyle='-'
                )
                axs[0].axvline(first_day, color=color["k"], linestyle='--')
            axs[0].axvline(last_day, color=color["k"], linestyle='--')
        
        axs[0].plot(
            pred["x"].to_list(), pred["true_mean"].to_list(),
            color=color["k"] 
        )

        if is_missing:
            missing_x, _ = missing_data
            start_area_index = day_plot[3]-1

            # Fixing incomplete testing (since we don't do padding)
            start_miss = x.index(missing_x[0])
            axs[0].plot(
                x[start_area_index:start_miss+1], y[start_area_index:start_miss+1],
                color=color["k"], linestyle='-'
            )

            plot_area(
                axs, x, y, missing_data, start_miss, 
                first_index, lag_color
            )

            axs[0].axvspan(
                first_day, last_day, 
                color="grey", alpha=0.1
            )

        plot_bound(axs[0], pred, color[pred_color], "Test")
        
        axs[1].axvline(first_day, color=color["grey"])
        axs[1].axvline(last_day, color=color["grey"])
        axs[1].axvspan(first_day, last_day, color=color["grey"], alpha=0.4) 

        axs[1].plot(pred["x"], pred["time_step_error"], 
            color=color[below_err], alpha=0.6)

        if is_missing:
            day_plot = (
                first_day, first_index, last_day, last_index
            )
        
        for i_start, j_start, loss in intv_loss:
            loc_bar = (i_start + j_start)/2
            width = loc_bar - i_start
            if width == 0:
                width = 1.0
            axs[1].bar(loc_bar, loss, width, color=color[below_err], alpha=0.6)
        

    axs[0].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0].grid()

    axs[0].set_xlim(left=x[0])
    axs[1].set_xlim(left=x[0])
    
    axs[1].set_xlabel("Time Step")
    axs[0].set_ylabel("Log Prices")
    axs[1].set_ylabel("Square Loss")

    return fig, axs

def show_result_fold(fold_results, exp_setting):
    """
    Printing out the Result of the fold data

    Args:
        fold_result: Result of walk_forward 
        exp_setting: Experiment Setting for Each Task
    """
    header = ["", "All Error", "Interval Error"]
    table = []

    all_mean_error = []
    all_std_error = []

    for i, fold_result in enumerate(fold_results):
        all_error_ind, all_error_intv = [], []
        for result in fold_result:
            all_error_ind += result.pred["time_step_error"].to_list() 
            all_error_intv += [loss for _, _, loss in result.interval_loss]
        
        task_setting = exp_setting["task"]
        task_prop = task_setting["dataset"][i]["out_feat_tran_lag"]
        metal = "+".join(task_setting["dataset"][i]["inp_metal_list"])

        mean_error = np.mean(all_error_ind)
        std_error = np.std(all_error_ind)/np.sqrt(len(all_error_ind))

        all_mean_error.append(mean_error)
        all_std_error.append(std_error)

        table.append([
            f"Task {i+1} (Metal={metal}, Lag={task_prop[0]}, Step ahead={task_prop[1]})", 
            # f"{round(np.median(all_error_ind), 7)}", 
            # f"{round(np.median(all_error_intv), 7)}"
            f"{round(mean_error, 7)} ± {round(std_error, 7)}", 
            f"{round(np.mean(all_error_intv), 7)} ± {round(np.std(all_error_intv)/np.sqrt(len(all_error_intv)), 7)}"
        ])

    print(tabulate(table, headers=header, tablefmt="grid"))
    return all_mean_error, all_std_error

def pack_result_data(mean, upper, lower, x):
    """
    Given the numpy/list data, pack the result into panda dataframe
        Ready for display/save 
    
    Args:
        mean: Mean prediction of the predictor
        upper: Upper Confidence Bound
        lower: Lower Confidence Bound
        x: x-axis that is used to display (should contains the data)
    
    Return:
        packed_data: Data ready to display
    """
    if len(upper) == 0 and len(lower) == 0:
        upper = mean
        lower = mean
    d = {"mean": mean, "upper": upper, "lower": lower, "x": x}
    return pd.DataFrame(data=d)
    

    