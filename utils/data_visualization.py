import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator

from collections import defaultdict
import pandas as pd
import math

from utils import data_preprocessing
from utils import others
from scipy.interpolate import make_interp_spline

from experiments import algo_dict
from experiments.metal_desc import metal_to_display_name, cluster_type_to_display_name

from datetime import datetime  
from datetime import timedelta  

from collections import OrderedDict

import itertools
import functools
import os

from experiments.metal_desc import metal_to_display_name

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

def save_figure(save_path, is_bbox_inches=True):
    def actual_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)

            if isinstance(out, tuple):
                fig, ax = out
                if save_path is not None:
                    if is_bbox_inches:
                        fig.savefig(save_path, bbox_inches='tight')
                    else:
                        fig.savefig(save_path)
                return fig, ax
            
            return out
        return wrapper
    return actual_decorator
    
def plot_axis_date(ax, data_date, month_interval=3):
    # https://matplotlib.org/stable/gallery/text_labels_and_annotations/date.html
    fmt_half_year = mdates.MonthLocator(interval=month_interval)
    ax.xaxis.set_major_locator(fmt_half_year)

    # Minor ticks every month.
    fmt_month = mdates.MonthLocator()
    ax.xaxis.set_minor_locator(fmt_month)

    # Text in the x axis will be displayed in 'YYYY-mm' format.
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Round to nearest years.
    datemin = np.datetime64(data_date[0], 'Y')
    datemax = np.datetime64(data_date[-1], 'Y') + np.timedelta64(12, 'M')
    ax.set_xlim(datemin, datemax)

    # Format the coords message box, i.e. the numbers displayed as the cursor moves
    # across the axes within the interactive GUI.
    ax.format_xdata = mdates.DateFormatter('%Y-%m')
    ax.format_ydata = lambda x: f'${x:.2f}'  # Format the price.


def visualize_time_series(fig_ax, data, inp_color, missing_data, lag_color, first_date,
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
    ((x_train_raw, y_train_raw), y_pred_list) = data

    missing_x, missing_y = missing_data
    is_missing = len(missing_x) != 0

    first_date = datetime.strptime(first_date, '%Y-%m-%d')

    convert_date = lambda x: [
        np.datetime64((first_date + timedelta(days=d)).strftime('%Y-%m-%d'))
        for d in x
    ]
    convert_price = lambda x: x["Output"].to_list()

    x_train = convert_date(x_train_raw["Date"].to_list())
    y_train = convert_price(y_train_raw)
    
    cut_point = x_train[-1]
    ax.plot(x_train, y_train, color=color[inp_color])

    for i, y_pred in enumerate(y_pred_list):
        data, plot_name, color_code, is_bridge = y_pred
        mean_pred, x_test_raw = data["mean"], data["x"]
        x_test = convert_date(x_test_raw)

        if i == 0 and is_missing:
            missing_x = convert_date(missing_x)
            ax.axvline(x_test[0], color=color[lag_color], linestyle='--', linewidth=0.5, dashes=(5, 0), alpha=0.2)
            ax.plot([missing_x[-1], x_test[0]], [missing_y[-1], mean_pred[0]], color[lag_color], linestyle="dashed")
            ax.axvspan(cut_point, x_test[0], color=color[lag_color], alpha=0.1)

        plot_bound(ax, data, x_test, color[color_code], plot_name)

        if is_bridge and (not is_missing): 
            ax.plot([x_train[-1], x_test[0]], [y_train[-1], mean_pred[0]], color[color_code], linewidth=1.5)

    if is_missing:
        ax.plot(missing_x, missing_y, color=color[lag_color], linestyle="dashed")
        ax.plot([x_train[-1], missing_x[0]], [y_train[-1], missing_y[0]], color[lag_color], linestyle="dashed")
        ax.axvline(cut_point, color=color[lag_color], linestyle='--', linewidth=0.5, dashes=(5, 0), alpha=0.2)
    else:
        ax.axvline(cut_point, color=color["k"], linestyle='--')

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.legend()

    # ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # ax.set_xlim(left=cut_point-np.timedelta64(1, 'm'))
    plot_axis_date(ax, x_train + missing_x + x_test)
    ax.grid()
    return fig, ax

def plot_bound(ax, data, x_test, color, plot_name):
    """
    Plotting with graph with uncertainty 

    Args:
        ax: Main plotting axis
        data: Packed data
        color: Color of the line
        plot_name: Name of the line
    """
    mean_pred, upper_pred, lower_pred = data["mean"], data["upper"], data["lower"]
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


def visualize_walk_forward(fig, axs, full_data_x, full_data_y, 
    fold_result, convert_date_dict, lag_color="o", pred_color="p", below_err="g", title="Walk Forward Validation Loss Visualization", true_value_name="Commodity", method_name="Regressor"):

    convert_date = lambda x: x["Date"].to_list()
    convert_price = lambda x: x["Output"].to_list()
    first_day = full_data_x["Date"][0] 

    x, _ = data_preprocessing.parse_series_time(convert_date(full_data_x), first_day)
    x = list(map(lambda a: convert_date_dict[a], x))
    
    first_day_abs = datetime.strptime(first_day, '%Y-%m-%d')
    convert_date2 = lambda x: [
        np.datetime64((first_day_abs + timedelta(days=d)).strftime('%Y-%m-%d'))
        for d in x
    ]

    x = convert_date2(x)
    y = convert_price(full_data_y)

    _, (miss_x, miss_y), _, _ = fold_result[0]
    is_missing = len(miss_x) != 0

    if is_missing:
        miss_x = convert_date2(miss_x)
        day_plot = (
            0, 0, miss_x[0], x.index(miss_x[0])
        )

    # fig = plt.figure(figsize=(15, 5))
    # gs = fig.add_gridspec(nrows=2, hspace=0)
    # axs = gs.subplots(sharex=True, sharey=False)
    fig.suptitle(title)

    if is_missing:
        axs[0].plot(
            x[day_plot[1]:day_plot[3]], 
            y[day_plot[1]:day_plot[3]], 
            color=color["k"], linestyle='-',
            label=true_value_name
        )


    for i, (pred, missing_data, _, loss_detail) in enumerate(fold_result):
        pred_x = convert_date2(pred["x"])
        first_day = pred_x[0]
        first_index = x.index(first_day)
        last_day = pred_x[-1]
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
            pred_x, pred["true_mean"].to_list(),
            color=color["k"] 
        )

        if is_missing:
            missing_x_raw, _ = missing_data
            #  = convert_date2(missing_x)

            missing_x = []
            for a in missing_x_raw:
                day = first_day_abs + timedelta(days=a)
                day_str = day.strftime('%Y-%m-%d')
                missing_x.append(np.datetime64(day_str))
            
            missing_data = (missing_x, missing_data[1])

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

        plot_bound(
            axs[0], pred, pred_x, 
            color[pred_color], 
            method_name if i == 0 else None
        )
        
        axs[1].axvline(first_day, color=color["grey"])
        axs[1].axvline(last_day, color=color["grey"])
        axs[1].axvspan(first_day, last_day, color=color["grey"], alpha=0.4) 

        axs[1].plot(pred_x, loss_detail["time_step_error"], 
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

    axs[1].set_xlabel("Time Step")
    axs[0].set_ylabel("Log Prices")
    axs[1].set_ylabel("Square Loss")

    plot_axis_date(axs[0], x, month_interval=3)
    plot_axis_date(axs[1], x, month_interval=3)
    axs[0].grid()
    axs[0].legend()

    return fig, axs

def show_result_fold(fold_results, all_exp_setting, other_details):
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
    list_exp_setting = others.create_legacy_exp_setting(all_exp_setting)

    flatten_dataset = list(itertools.chain(*[
        exp_setting["task"]["dataset"]
        for exp_setting in list_exp_setting 
    ]))
    clus_num, output_name, method_name, is_show_cluster = other_details


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

        dataset = flatten_dataset[i]
        task_prop = dataset["out_feat_tran_lag"]
        metal = dataset.gen_name()

        cal_mean_std = lambda x: (
            np.mean(x), np.std(x)/np.sqrt(len(x))
        )

        time_step_mean, time_step_std = cal_mean_std(all_error_ind)
        all_time_step_loss.append((time_step_mean, time_step_std))

        intv_mean, intv_std = cal_mean_std(all_error_intv)

        crps_mean, crps_std = cal_mean_std(all_error_crps)
        all_crps_loss.append((crps_mean, crps_std))

        num_round = 7

        cluster_text = f"(Cluster {clus_num[i]})\n " if is_show_cluster else ""

        table.append([
            f"Task {i+1} " + cluster_text + f"  Algorithm: {method_name[i]}\n   Commodity: {output_name[i]}\n   Lag: {task_prop[0]}", 
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
                        f"{mean*100:.4f} $\pm$ {std*100:.4f}"
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
                        # bold_max[0] = bold_max[0] + "*"
                        eval_values[eval_num].append(' '.join(bold_max))
                    else:
                        eval_values[eval_num].append(contents[eval_num])

        for i in range(len_evals):
            if i == 0:
                print(row_name, end='')
            print(" & ", end='')
            print(" & ".join(eval_values[i]) + "\\\\")
        print("\\addlinespace")

def plot_heat_map(ax, matrix, row_name, column_name, xlabel="PCA", ylabel="Backward Time", round_acc=3):
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

    cmap = plt.cm.coolwarm
    cmap.set_bad(color=color["p"])
    im = ax.imshow(matrix, cmap="coolwarm")

    len_col = range(len(column_name))
    len_row = range(len(row_name))

    ax.set_xticks(len_col)
    ax.set_yticks(len_row)
    ax.set_xticklabels(column_name)
    ax.set_yticklabels(row_name)

    for i in len_row:
        for j in len_col:
            if np.isnan(matrix[i, j]):
                text = ax.text(
                    j, i, "n/a",
                    ha="center", va="center", color="w"
                )
            else:
                text = ax.text(
                    j, i, round(matrix[i, j], round_acc),
                    ha="center", va="center", color="w"
                )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

@save_figure("figure/hyperparam_search_gp.pdf", False)
def plot_hyperparam_search_gp():
    load_path = "exp_result/save_hyper/hyper_search_"
    algorithms = ["SparseGPIndex", "GPMultiTaskMultiOut", "GPMultiTaskIndex"]
    kernel_names = ["matern", "rbf"]
    kernel_display = ["Matern", "Radial Basis Function"]
    return plot_hyperparam_search(load_path, algorithms, kernel_names, kernel_display)


@save_figure("figure/hyperparam_search_gp_deep.pdf", False)
def plot_hyperparam_search_gp_deep():
    load_path = "exp_result/save_hyper_deep/hyper_search_"
    algorithms = ["DeepGPMultiOut", "NonlinearMultiTaskGP"]
    kernel_names = ["matern"]
    kernel_display = ["Matern"]
    return plot_hyperparam_search(load_path, algorithms, kernel_names, kernel_display)


def plot_hyperparam_search(load_path, algorithms, kernel_names, kernel_display):

    row_name = np.arange(2, 12, step=2)
    column_name = np.arange(2, 7)
    
    fig = plt.figure(figsize=(12, 4*len(kernel_names)), constrained_layout=True)

    subfigs = fig.subfigures(len(kernel_names), 1)
    if len(kernel_names) == 1:
        subfigs = [subfigs]

    all_kernel = []
    for i, kernel in enumerate(kernel_names):
        curr_fig = subfigs[i]
        axes = curr_fig.subplots(1, len(algorithms))

        curr_fig.suptitle(kernel_display[i], fontsize=20)
        # plt.text(x=0.5, y=0.98-i*0.5, s=kernel_display[i], fontsize=20, ha="center", transform=fig.transFigure)

        load_folder = load_path + kernel
        result = others.load_json(load_folder + f"/final_result_{kernel}.json")
        total_result = []

        for j, algo in enumerate(algorithms):
            result_crps = np.array(result[algo]["CRPS"])
            curr_ax = axes[j]
            plot_heat_map(curr_ax, result_crps, row_name, column_name)
            curr_ax.set_title(algo_dict.class_name_to_display[algo])
            total_result.append(result_crps)
        
        all_kernel.append(total_result)
    
    all_algo = []
    find_lowest_index = lambda arr: np.unravel_index(np.nanargmin(arr), arr.shape)
    find_highest_index = lambda arr: np.unravel_index(np.nanargmax(arr), arr.shape)

    for j, algo in enumerate(algorithms):
        algo_kernl = []
        for i, kernel in enumerate(kernel_names):
            algo_kernl.append(all_kernel[i][j])
        
        print("At Algorithm:", algo)
        matrix = np.stack(algo_kernl)

        matrix[np.isnan(matrix)] = -100

        worst_kernel, worst_row, worst_col = find_highest_index(matrix)
        print(f"Worst Kernel: {kernel_display[worst_kernel]} Worst Inp Len: {row_name[worst_row]} Worst PCA: {column_name[worst_col]} Worst CRPS {np.max(matrix)}") 

        matrix[matrix == -100] = 100
        kernel, row, col = find_lowest_index(matrix)
        print(f"Optimal Kernel: {kernel_display[kernel]} Optimal Inp Len: {row_name[row]} Optimal PCA: {column_name[col]} Best CRPS {np.min(matrix)}") 
        matrix[find_lowest_index(matrix)] = np.inf
        second_kernel, second_row, second_col = find_lowest_index(matrix)
        print(f"Second Optimal Kernel: {kernel_display[second_kernel]} Second Optimal Inp Len: {row_name[second_row]} Second Optimal PCA: {column_name[second_col]} Second Best CRPS {np.min(matrix)}")
        
        print("------------")

    # fig.tight_layout(rect=[0, 0.03, 1, 3])
    
    return fig, axes

@save_figure("figure/compare_cluster_best.pdf")
def plot_compare_cluster_best():
    return plot_compare_cluster("exp_result/range_best_hyperparam/compare_cluster_4.json")

@save_figure("figure/compare_cluster_worst.pdf")
def plot_compare_cluster_worst():
    return plot_compare_cluster("exp_result/range_worst_hyperparam/compare_cluster_4.json")


def plot_compare_cluster(data_path):
    result_cluster = others.load_json(data_path)


    keys = list(result_cluster.keys())
    multi_task_gp = sorted(list(result_cluster[keys[0]].keys()))

    # multi_task_gp = ["GPMultiTaskMultiOut", "GPMultiTaskIndex", "SparseGPIndex"]

    full_model_result = result_cluster["full_model"]

    del result_cluster["full_model"]
    num_cluster = len(result_cluster)
    
    fig, axes = plt.subplots(ncols=len(multi_task_gp), nrows=1, figsize=(4*len(multi_task_gp), 7), sharey=True)
    all_cluster_names = []

    # xlim_min = [-0.025, 0.15]
    # xlim = [0.025, 0.25]

    for j, (mtl_gp, ax) in enumerate(zip(multi_task_gp, axes)):

        diff_result = []
        for i, (test_name, result) in enumerate(result_cluster.items()):
            convert = {
                "-".join(k.split(" ")) : v
                for k, v in 
                cluster_type_to_display_name.items()
            }
            if j == 0:
                all_cluster_names.append(convert[test_name])
            
            if not result[mtl_gp] is None:
                result_mean, result_std = result[mtl_gp]
                full_mean, full_std = full_model_result[mtl_gp]

                diff = full_mean - result_mean
                data_err = np.sqrt(result_std**2+full_std**2)
                ax.scatter(
                    x=diff, y=i, 
                    color=color["r"] if diff < 0 else color["g"], 
                    s=40, zorder=6
                )
                ax.errorbar(
                    x=diff, y=i, xerr=data_err, 
                    color=color["k"], linewidth=1.5, 
                    zorder=3, capsize=5
                )

                diff_result.append(diff)

        if np.sign(np.max(diff_result)) != np.sign(np.min(diff_result)):
            ax.axvline(x=0.0, color=color["k"], linestyle="--")

        # ax.set_xlim(
        #     np.min(diff_result)-np.std(diff_result)/2.0, 
        #     np.max(diff_result)+np.std(diff_result)/2.0
        # )
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(zorder=0, linestyle='--')
        ax.set_title(algo_dict.class_name_to_display[mtl_gp])
        ax.set_xlabel("Improvement")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
 
        if j == 0:
            ax.set_yticklabels(all_cluster_names, rotation=0, fontsize=10)
            ax.set_ylim(-1, num_cluster)
            ax.set_yticks(np.arange(0, num_cluster))

    fig.tight_layout()
    # plt.show()
    return fig, axes

@save_figure("figure/compare_graph_best.pdf")
def plot_compare_graph_best(): 
    return plot_compare_graph("exp_result/graph_compare/bound_compare_graph_best.json")

@save_figure("figure/compare_graph_worst.pdf")
def plot_compare_graph_worst(): 
    return plot_compare_graph("exp_result/graph_compare/bound_compare_graph_worst.json")
    
def plot_compare_graph(path):
    result_cluster = others.load_json(path)

    suitable_baseline_compare = {
        "SparseMaternGraphGP": "SparseGPIndex",
        "DeepGraphMultiOutputGP":"DeepGPMultiOut",
        "DeepGraphInfoMaxMultiOutputGP": "DeepGPMultiOut"
    }

    graph_name_to_display = {
        "distance correlation": "Distance Correlation",
        "hsic": "HSIC",
        "kendell": "Kendel",
        "peason": "Peason",
        "spearman": "Spearman"
    }

    type_graph = list(result_cluster.keys())
    type_graph.remove("no_graph_model")
    type_graph = sorted(type_graph)
    num_graph = len(type_graph)

    baseline_result = result_cluster["no_graph_model"]
    del result_cluster["no_graph_model"]

    multi_task_gp = list(result_cluster[type_graph[0]].keys())
    num_mlt = len(multi_task_gp)

    fig, axes = plt.subplots(
        ncols=num_mlt, nrows=1, figsize=(5*num_mlt, 4), sharey=True
    )

    if num_mlt == 1:
        axes = [axes]
    
    
    for j, (mtl_gp, ax) in enumerate(zip(multi_task_gp, axes)):
        list_improvement = []
        for i, graph in enumerate(type_graph):
            baseline_mean, baseline_std = baseline_result[
                suitable_baseline_compare[mtl_gp]
            ]
            result_mean, result_std = result_cluster[graph][mtl_gp]

            diff = baseline_mean - result_mean

            list_improvement.append(diff)

            ax.scatter(
                x=diff, y=i, 
                color=color["r"] if diff < 0 else color["g"], 
                s=40, zorder=6
            )
            ax.errorbar(
                x=diff, y=i, 
                xerr=np.sqrt(baseline_std**2+result_std**2),
                color=color["k"], linewidth=1.5, 
                zorder=3, capsize=5,
            )
        
        least_imporv, max_improv = min(list_improvement), max(list_improvement) 
        
        if np.sign(np.max(list_improvement)) != np.sign(np.min(list_improvement)):
            ax.axvline(x=0.0, color=color["k"], linestyle="--")

        # ax.set_xlim(
        #     least_imporv-np.std(list_improvement)/2.0, 
        #     max_improv+np.std(list_improvement)/2.0
        # )
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(zorder=0)
        ax.set_title(algo_dict.class_name_to_display[mtl_gp])
        ax.set_xlabel("Improvement")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

        if j == 0:
            ax.set_yticklabels(
                [graph_name_to_display[g] for g in type_graph], 
                rotation=0, fontsize=10
            )
            ax.set_ylim(-1, num_graph)
            ax.set_yticks(np.arange(0, num_graph))
    
    return fig, axes

def plot_arma_hyper_search(main_path):
    get_all_npy = lambda path: sorted([f for f in os.listdir(path) if ".npy" in f])
    all_path = get_all_npy(main_path)

    batch_name = np.arange(2, 12, step=1)
    row_name = np.arange(2, 12, step=1)
    col_name = np.arange(2, 12, step=1)

    for path in all_path:
        # fig, ax = plt.subplots(figsize=(10, 10))
        data = np.load(main_path + "/" + path) * 0.00001
        p, q, d = np.unravel_index(data.argmin(), data.shape)
        print(f"For Metal {metal_to_display_name[path.split('.')[0]]}: Optimal Order: ({row_name[p]}, {batch_name[d]}, {col_name[q]})")

def cluster_label_to_dict(labels):
    num_cluster = len(set(labels))
    # print(labels)
    # assert sorted(list(set(labels))) == list(range(num_cluster))

    return {
        cluster : [i for i, l in enumerate(labels) if l == cluster]
        for cluster in range(num_cluster)
    }

def print_tables_side_by_side(tables, headers, titles, spacing=3):
    # Adapted from: https://gist.github.com/edisongustavo/d8116e9dc41a9a509a6f2b7c7d74f299
    string_tables_split = [tabulate(t, headers=h, tablefmt="grid").splitlines() for t, h in zip(tables, headers)]
    spacing_str = " " * spacing

    # Printing Titles 
    for t, string in zip(titles, string_tables_split):
        all_len = len(string[0]) + spacing
        print(t, end='')
        print(" " * (all_len - len(t)), end='')
    print()

    num_lines = max(map(len, string_tables_split))
    paddings = [max(map(len, s_lines)) for s_lines in string_tables_split]

    for i in range(num_lines):
        line_each_table = []
        for padding, table_lines in zip(paddings, string_tables_split):
            if len(table_lines) <= i:
                line_each_table.append(" " * (padding + spacing))
            else:
                line_table_string = table_lines[i]
                line_len = len(line_table_string)
                line_each_table.append(
                    line_table_string + (" " * (padding - line_len)) + spacing_str
                )

        final_line_string = "".join(line_each_table)
        print(final_line_string)

# @save_figure("figure/grid_all_deep_gp.pdf")
# def plot_grid_commodity_deep():
#     return plot_grid_commodity_all("exp_result/grid_result/grid_result_non_deep.json")

@save_figure("figure/grid_all.pdf")
def plot_grid_commodity_gp():
    # return plot_grid_commodity_all("exp_result/grid_result/grid_result_deep.json")
    return plot_grid_commodity_all("exp_result/grid_result/grid_result_all.json")

def plot_grid_commodity_all(load_path):
    load_details = others.load_json(load_path)
    all_algo = [algo for algo in list(load_details.keys()) if algo != "metal_names"]
    metal_names = others.find_all_metal_names()
    
    fig, axes = plt.subplots(ncols=4, nrows=2, figsize=(35, 15))

    for i, algo in enumerate(all_algo):
        curr_axes = axes.flatten()[i]
        result = np.array(load_details[algo], dtype=np.float32)
        result = result + result.T
        np.fill_diagonal(result, np.nan)
        plot_heat_map(
            curr_axes, result, [metal_to_display_name[n] for n in metal_names], 
            [metal_to_display_name[n] for n in metal_names],  xlabel="Commodities", 
            ylabel="Commodities", 
            round_acc=3
        )
        plt.setp(curr_axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        curr_axes.set_title(algo_dict.class_name_to_display[algo], fontsize=20)
     
    fig.delaxes(axes.flatten()[-1])
    fig.tight_layout()
    # plt.show()
    return fig, axes

@save_figure("figure/range_algo_best.pdf")
def plot_range_algo_best():
    return plot_range_algo("exp_result/range_best_hyperparam/compare_cluster_")

@save_figure("figure/range_algo_worst.pdf")
def plot_range_algo_worst():
    return plot_range_algo("exp_result/range_worst_hyperparam/compare_cluster_")

def plot_range_algo(base_path):

    range_cluster = range(2, 8)
    sample_path = base_path + "2.json"
    sample_data = others.load_json(sample_path)

    cluster_methods = sorted(list(sample_data.keys()))
    learning_methods = sorted(list(sample_data[cluster_methods[0]].keys()))
    cluster_methods.remove("full_model")

    # methods.remove("full_model")
    # print(learning_methods)
    # print(cluster_methods)

    num_row = len(cluster_methods)
    num_col = len(learning_methods)

    # num_row = 4
    # num_col = 2

    fig, axes = plt.subplots(
        nrows=num_row,
        ncols=num_col, 
        figsize=(25, 25),
        # figsize=(10, 10)
    )

    for row in range(num_row):
        for col in range(num_col):

            curr_axes = axes[row][col]

            algo_name = learning_methods[col]
            cluster_name = cluster_methods[row]

            all_data = []

            for num_cluster in range_cluster:
                cluster_path = base_path + f"{num_cluster}.json"
                data = others.load_json(cluster_path)

                full_mean, full_std = data["full_model"][algo_name]
                result_mean, result_std = data[cluster_name][algo_name]

                if result_mean is None:
                    pass
                else:
                    diff = full_mean - result_mean
                    real_std = np.sqrt(full_std**2 + result_std**2)
                    all_data.append([num_cluster, diff, real_std])

            all_data = np.array(all_data)
            x, y, err = all_data.T

            for data_x, data_y, data_err in zip(x, y, err):
                curr_color = color["r"] if data_y < 0 else color["g"]
                curr_axes.scatter(
                    x=data_x, y=data_y, 
                    color=curr_color, 
                    s=60, zorder=6
                )
 
                curr_axes.errorbar(
                    data_x, data_y, yerr=data_err,
                    zorder=3, color=color["gray"], 
                    linewidth=1.5, 
                    ecolor=color["k"], capsize=5
                )

            curr_axes.plot(x, y, zorder=3, color=color["gray"], linewidth=0.5)

            if np.sign(np.max(y)) != np.sign(np.min(y)):
                curr_axes.axhline(y=0, color=color["k"], linestyle="--")
            
            curr_axes.grid(zorder=0, linestyle='--')
            
            if row == 0:
                curr_axes.set_title(
                    algo_dict.class_name_to_display[algo_name], 
                    fontsize=16, y=1.1
                )
            
            if col == 0: 
                convert = {
                    "-".join(k.split(" ")) : v
                    for k, v in 
                    cluster_type_to_display_name.items()
                }
                curr_axes.set_ylabel(
                    convert[cluster_name], 
                    rotation=0, fontsize=16, labelpad=15
                )
                curr_axes.yaxis.set_label_coords(-0.5, 0.5)
            
            curr_axes.set_xlim(range_cluster[0]-0.2, range_cluster[-1]+0.2)
            curr_axes.set_xticks(range_cluster)
            curr_axes.set_xticklabels(range_cluster)
            curr_axes.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            
    fig.tight_layout()
    
    # plt.show()
    return fig, axes

def plot_table_cluster():
    from experiments.metal_desc import metal_to_display_name
    colors = [
        "#ff7500", "#5a08bf", "#0062b8", 
        "#1a1a1a", "#20c406", "#ebebeb",
        "#d6022a", "#009688", "#00e5ff", "#1a237e"
    ]

    def text_color(color, text):
        return "\\textcolor[HTML]{" + color + "}{\\textbf{" + text + "}}"

    display_name_metal = {
        "aluminium": "Al",
        "carbon": "CC",
        "copper": "Cu",
        "lldpe": "LLDPE",
        "natgas": "NG",
        "nickel": "Ni",
        "palladium": "Pd",
        "platinum": "Pt",
        "pvc": "PVC",
        "wheat": "WH",
    }

    display_name_cluster = {
        "peason": "Peason",
        "spearman": "Spearman",
        "kendell": "Kendell",
        "euclidean": "Euclidean",
        "dtw": "DTW",
        "soft-dtw divergence": "Soft-DTW Div.",
        "euclidean knn": "Euclidean KNN",
        "dtw knn": "DTW KNN",
        "softdtw knn": "Soft-DTW KNN",
        "kshape": "KShape",
        "expert": "Expert"
    }

    def plot_row_color(names, colors=None):
        row_texts = ""
        for i, n in enumerate(names):
            if colors is None:
                curr_text = "\\textbf{" + n + "}" 
            elif colors[i] is None:
                curr_text = n
            else:
                curr_text = text_color(colors[i], n)

            if i == len(names) - 1:
                curr_text = "\t" + curr_text + " \\\\"
            elif i == 0:
                curr_text = curr_text + " &"
            else:
                curr_text = "\t" + curr_text + " &"

            row_texts += curr_text

        return row_texts

    open("table_cluster.txt", 'w').close()

    with open("table_cluster.txt", 'a') as f:

        for cluster_num in [2, 3, 4, 5, 6, 7]:
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{lcccccccccc}\n")
            f.write("\\toprule\n")

            all_metal_name = others.find_all_metal_names()
            f.write(plot_row_color(
                ["Name"] + [display_name_metal[a] for a in all_metal_name]
            ) + "\n")
            f.write("\\midrule\n")
            
            all_cluster_path = f"exp_result/cluster_result/feat_data/cluster_{cluster_num}.json"

            cluster_data = others.load_json(all_cluster_path)
            for k, v in cluster_data.items():
                f.write(plot_row_color(
                    [display_name_cluster[k]] + [str(i) for i in v],
                    [None] + [colors[i][1:] for i in v]
                ) + "\n")

            # f.write("\\midrule\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\caption{some caption}\n")
            f.write("\\label{some label}\n")
            f.write("\\end{table}\n")

@save_figure("figure/embedding_graph.pdf")
def plot_embedding():
    embedding_noinfomax = np.load("exp_result/embedding/embedding_noinfomax.npy")
    embedding_infomax = np.load("exp_result/embedding/embedding_infomax.npy")
    embedding_start = np.load("exp_result/embedding/pre_embedding_infomax.npy")
    # embedding = embedding[:50]
    num_data = embedding_start.shape[0]
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    def plot_data(embedding, ax, i):
        embedding = np.reshape(embedding,(-1, embedding.shape[-1]) )
        labels = np.concatenate([
            np.arange(0, 10)
            for i in range(num_data)
        ])
        colors = ["#ff7500", "#5a08bf", "#0062b8", "#1a1a1a", "#20c406", "#ebebeb","#d6022a", "#009688", "#00e5ff", "#1a237e"]
        
        all_metal_name = others.find_all_metal_names()
        display_name_metal = {
            "aluminium": "Al",
            "carbon": "CC",
            "copper": "Cu",
            "lldpe": "LLDPE",
            "natgas": "NG",
            "nickel": "Ni",
            "palladium": "Pd",
            "platinum": "Pt",
            "pvc": "PVC",
            "wheat": "WH",
        }

        all_metal_name = [display_name_metal[a] for a in all_metal_name]

        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import umap
        import matplotlib
     
        # def plot_3D():
        #     pca = PCA(n_components=3)
        #     reduced_data = pca.fit_transform(embedding)

        #     fig, ax = plt.subplots(figsize=(10, 10),subplot_kw=dict(projection='3d'))
        #     x, y, z = reduced_data.T
        #     ax.scatter3D(
        #         x, y, z, c=labels, s=10.0, 
        #         cmap=matplotlib.colors.ListedColormap(colors)
        #     )
        
        # def plot_2D():

        pca = TSNE(n_components=2, perplexity=50)
        # pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(embedding).T
        # reduced_data = np.load("cache.npy")
        # np.save("cache.npy", reduced_data)

        import matplotlib.patches as mpatches
        recs = []
        for i in range(0,len(colors)):
            recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))
        ax.legend(recs, all_metal_name, loc=4, ncol=2)

        
        ax.scatter(
            reduced_data[0], 
            reduced_data[1], 
            c=labels, 
            cmap=matplotlib.colors.ListedColormap(colors),
            zorder=3
        )

        ax.grid(zorder=0)

    plot_data(embedding_start, axes[0], 0)
    plot_data(embedding_noinfomax, axes[1], 1)
    plot_data(embedding_infomax, axes[2], 2)

    fig.tight_layout()
    plt.show()

    return fig, axes

