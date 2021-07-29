import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from collections import defaultdict
import pandas as pd
import math

from utils.data_preprocessing import parse_series_time
from scipy.interpolate import make_interp_spline
from models.ind_multi_model import IndependentMultiModel

from collections import OrderedDict

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
    ax.fill_between(x_test, upper_pred, lower_pred, color=color, alpha=0.2)
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


    for i, (pred, missing_data, _, loss_detail) in enumerate(fold_result):
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

        axs[1].plot(pred["x"], loss_detail["time_step_error"], 
            color=color[below_err], alpha=0.6)

        if is_missing:
            day_plot = (
                first_day, first_index, last_day, last_index
            )
        
        for i_start, j_start, loss in loss_detail["intv_loss"]:
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
    header = ["", "MSE", "CRPS", "Interval Error"]
    table = []

    all_time_step_loss = []
    all_crps_loss = []

    for i, fold_result in enumerate(fold_results):
        all_error_ind, all_error_intv = [], []
        all_error_crps = []
        for result in fold_result:
            loss_detail = result.loss_detail
            all_error_ind += loss_detail["time_step_error"]
            all_error_intv += [
                loss for _, _, loss in loss_detail["intv_loss"]
            ]
            all_error_crps.append(loss_detail["all_crps"])
        
        task_setting = exp_setting["task"]
        task_prop = task_setting["dataset"][i]["out_feat_tran_lag"]
        metal = task_setting["dataset"][i].gen_name()

        cal_mean_std = lambda x: (
            np.mean(x), np.std(x)/np.sqrt(len(x))
        )

        time_step_mean, time_step_std = cal_mean_std(all_error_ind)
        all_time_step_loss.append((time_step_mean, time_step_std))

        intv_mean, intv_std = cal_mean_std(all_error_intv)

        crps_mean, crps_std = cal_mean_std(all_error_crps)
        all_crps_loss.append((crps_mean, crps_std))

        num_round = 7

        table.append([
            f"Task {i+1} (Metal={metal}, Lag={task_prop[0]}, Step ahead={task_prop[1]})", 
            f"{time_step_mean:.{num_round}} ± {time_step_std:.{num_round}}", 
            f"{crps_mean:.{num_round}} ± {crps_std:.{num_round}}", 
            f"{intv_mean:.{num_round}} ± {intv_std:.{num_round}}"
        ])

    print(tabulate(table, headers=header, tablefmt="grid"))
    return all_time_step_loss, all_crps_loss

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
    
def plot_latex(names, results, multi_task_name, display_name_to_algo):

    for multi_task_no, mt_name in enumerate(multi_task_name):
        all_algo_name = names[multi_task_no]
        all_results = results[multi_task_no]

        df = OrderedDict({"Names": [display_name_to_algo[a] for a in all_algo_name]})
    
        num_algo = len(all_algo_name)
        num_task = len(all_results[list(all_results.keys())[0]]["MSE"])
        len_evals = -1

        for n_task in range(num_task):
            task_result = []
            for name in all_algo_name:
                total_eval = []
                eval_dict = all_results[name]
                for eval_method, eval_result in eval_dict.items():
                    evals_per_task = [
                        f"{mean:.5f} $\pm$ {std:.5f}"
                        for mean, std in eval_result
                    ]
                    total_eval.append(evals_per_task)
                
                len_evals = len(eval_dict) 
                task_result.append(list(zip(*total_eval))[n_task])
            
            df.update({f"{mt_name[n_task]}.Task {n_task+1}": task_result})
        
        if multi_task_no == 0:
            total_df = pd.DataFrame(df)
        else:
            total_df = total_df.merge(
                pd.DataFrame(df), how="left", on="Names"
            )
    
    def map_numpy(x):
        if isinstance(x, float):
            return np.array([np.inf for _ in range(len_evals)])
        else:
            return np.array([float(val.split(" ")[0]) for val in x])

    all_vals = total_df.to_numpy()[:, 1:]
    new_vals = np.zeros_like(all_vals).tolist()

    for i in range(all_vals.shape[0]):
        for j in range(all_vals.shape[1]):
            new_vals[i][j] = map_numpy(all_vals[i, j])
    
    new_vals = np.array(new_vals)
    min_values = np.argmin(new_vals, axis=0)

    for index_method, row in total_df.iterrows():
        row_name = row[0]
        eval_values = [[] for i in range(len_evals)]
        for task_num, contents in enumerate(row[1:]):
            for eval_num in range(len_evals):
                if isinstance(contents, float):
                    eval_values[eval_num].append("-")
                else:
                    if index_method == min_values[task_num, eval_num]:
                        bold_max = contents[eval_num].split(" ")
                        bold_max[0] = "\\textbf{" + bold_max[0] + "}" 
                        eval_values[eval_num].append(' '.join(bold_max))
                    else:
                        eval_values[eval_num].append(contents[eval_num])

        for i in range(len_evals):
            if i == 0:
                print(row_name, end='')
            print(" & ", end='')
            print(" & ".join(eval_values[i]) + "\\\\")
        print("\\addlinespace")

