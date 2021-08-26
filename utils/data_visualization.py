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

def save_figure(save_path):
    def actual_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)

            if isinstance(out, tuple):
                fig, ax = out
                if save_path is not None:
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
    datemax = np.datetime64(data_date[-1], 'Y') + np.timedelta64(6, 'M')
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

def plot_heat_map(ax, matrix, row_name, column_name, xlabel="PCA", ylabel="Backward Time", round_acc=3):
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    im = ax.imshow(matrix, cmap="coolwarm")

    len_col = range(len(column_name))
    len_row = range(len(row_name))

    ax.set_xticks(len_col)
    ax.set_yticks(len_row)
    ax.set_xticklabels(column_name)
    ax.set_yticklabels(row_name)

    for i in len_row:
        for j in len_col:
            text = ax.text(
                j, i, round(matrix[i, j], round_acc),
                ha="center", va="center", color="w"
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

def plot_hyperparam_search():

    def plot_individual(result, axes):
        row_name = np.arange(2, 12, step=2).tolist()
        column_name = np.arange(2, 7).tolist()

        for i, metric in enumerate(all_metric):
            for j, method in enumerate(all_methods):
                data = np.array(results[method][metric]) * 10
                curr_ax = axes[i, j] 
                curr_ax.set_title(f"{algo_dict.class_name_to_display[method]} {metric}") 
                plot_heat_map(curr_ax, data, row_name, column_name)

 
    fig = plt.figure(figsize=(15, 15), constrained_layout=True)
    subfigs = fig.subfigures(2, 2)
    
    list_load_path = [
        "exp_result/save_hyper/hyper_search_matern/final_result_matern.json",
        "exp_result/save_hyper/hyper_search_matern_periodic/final_result_matern_periodic.json",
        "exp_result/save_hyper/hyper_search_rbf/final_result_rbf.json",
        "exp_result/save_hyper/hyper_search_rbf_periodic/final_result_rbf_periodic.json"
    ]

    list_name = [
        "Matern", "Matern + Periodic", "RBF", "RBF + Periodic"
    ]
            
    for i, (load_path, name) in enumerate(zip(list_load_path, list_name)):    
        results = others.load_json(load_path)
        all_methods = list(results.keys())
        all_metric = list(results[all_methods[0]].keys())
        # all_metric = ["CRPS"]

        curr_subfig = subfigs.flatten()[i]
        axes = curr_subfig.subplots(ncols=len(all_methods), nrows=len(all_metric))

        if axes.ndim == 1:
            axes = np.expand_dims(axes, axis=0)

        plot_individual(results, axes)
        curr_subfig.suptitle(name, fontsize=25)
    
    fig.savefig("figure/hyperparam.pdf")

@save_figure("figure/compare_cluster.pdf")
def plot_compare_cluster():
    result_cluster = others.load_json("exp_result/cluster_compare/compare_cluster.json")

    multi_task_gp = list(algo_dict.multi_task_algo.keys())
    multi_task_gp.remove("IndependentMultiModel")

    full_model_result = result_cluster["full_model"]
    diff_result = {}

    del result_cluster["full_model"]
    num_cluster = len(result_cluster)
    
    fig, axes = plt.subplots(ncols=len(multi_task_gp), nrows=1, figsize=(10, 4), sharey=True)
    all_cluster_names = []

    xlim_min = [-0.025, 0.15]
    xlim = [0.025, 0.25]

    for j, (mtl_gp, ax) in enumerate(zip(multi_task_gp, axes)):
        for i, (test_name, result) in enumerate(result_cluster.items()):
            if j == 0:
                all_cluster_names.append(cluster_type_to_display_name[test_name])
            diff = full_model_result[mtl_gp] - result[mtl_gp]  
            ax.scatter(x=diff, y=i, color=color["r"] if diff < 0 else color["g"], s=40, zorder=3)
    
        ax.axvline(x=0.0, color=color["k"], linestyle="--")
        ax.set_xlim(xlim_min[j], xlim[j])
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(zorder=0)
        ax.set_title(algo_dict.class_name_to_display[mtl_gp])
        ax.set_xlabel("Improvement")
 
        if j == 0:
            ax.set_yticklabels(all_cluster_names, rotation=0, fontsize=10)
            ax.set_ylim(-1, num_cluster)
            ax.set_yticks(np.arange(0, num_cluster))

    fig.tight_layout()
    plt.show()
    return fig, axes
    
@save_figure("figure/compare_graph.pdf")
def plot_compare_graph():
    result_cluster = others.load_json(
        "exp_result/grid_graph/compare_graph.json"
    )

    suitable_baseline_compare = {
        "SparseMaternGraphGP": "SparseGPIndex"
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
    
    list_improvement = []
    
    for j, (mtl_gp, ax) in enumerate(zip(multi_task_gp, axes)):
        for i, graph in enumerate(type_graph):
            diff = baseline_result[
                suitable_baseline_compare[mtl_gp]
            ] - result_cluster[graph][mtl_gp]

            list_improvement.append(diff)

            ax.scatter(
                x=diff, y=i, 
                color=color["r"] if diff < 0 else color["g"], 
                s=40, zorder=3
            )
        
        least_imporv, max_improv = min(list_improvement), max(list_improvement) 
        
        ax.axvline(x=0.0, color=color["k"], linestyle="--")
        ax.set_xlim(least_imporv-0.02, max_improv+0.02)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(zorder=0)
        ax.set_title(algo_dict.class_name_to_display[mtl_gp])
        ax.set_xlabel("Improvement")

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

    row_name = np.arange(2, 12, step=1)
    col_name = np.arange(2, 12, step=1)

    for path in all_path:
        # fig, ax = plt.subplots(figsize=(10, 10))
        data = np.load(main_path + "/" + path) * 0.00001
        row, col = np.unravel_index(data.argmax(), data.shape)
        print(f"For Metal {metal_to_display_name[path.split('.')[0]]}: Optimal Order: ({row_name[row]}, 0, {col_name[col]})")

def cluster_label_to_dict(labels):
    num_cluster = len(set(labels))
    assert sorted(list(set(labels))) == list(range(num_cluster))

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

@save_figure("figure/grid_error.pdf")
def plot_grid_commodity(load_path):
    load_details = others.load_json(load_path)
    all_algo = [algo for algo in list(load_details.keys()) if algo != "metal_names"]
    metal_names = load_details["metal_names"]
    
    fig, axes = plt.subplots(ncols=len(all_algo), nrows=1, figsize=(16, 8))

    for i, algo in enumerate(all_algo):
        result = np.array(load_details[algo], dtype=np.float32)
        plot_heat_map(
            axes[i], result, metal_names, 
            metal_names, xlabel="Commodities", 
            ylabel="Commodities", 
            round_acc=3
        )
        plt.setp(axes[i].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        axes.set_title(algo_dict.class_name_to_display[algo])
     
    fig.tight_layout()
    plt.show()
    return fig, axes

def plot_arma_param_search():
    loading_path = "exp_result/hyper_param_arma"
    
