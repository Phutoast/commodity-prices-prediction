import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import torch
from collections import Counter
import matplotlib.ticker as ticker
import copy

from utils.data_preprocessing import load_transform_data, parse_series_time, load_metal_data, parse_series_time, cal_lag_return, GlobalModifier
from utils.data_structure import DatasetTaskDesc
from utils.data_visualization import plot_axis_date, plot_heat_map
from utils.others import find_sub_string, load_json, find_all_metal_names, create_folder, save_figure
from utils.data_structure import CompressMethod
from datetime import datetime

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.decomposition import PCA
import scipy.stats as stats
from tabulate import tabulate
import networkx as nx

# Ordered in a Group
metal_names = [
    "aluminium", "copper", "nickel", 
    "palladium", "platinum", "lldpe", 
    "pvc", "natgas", "carbon", "wheat"
]
metal_to_display_name = {
    "aluminium": "Aluminium",
    "carbon": "Carbon Credits",
    "copper": "Copper",
    "lldpe": "LLDPE",
    "natgas": "Natural Gas",
    "palladium": "Palladium",
    "platinum": "Platinum",
    "pvc": "PVC",
    "wheat": "Wheat",
    "nickel": "Nickel"
}

def plot_frequency_features():
    metal_names = ["aluminium", "copper"]
    metal_to_color = {"copper": "#D81B60", "aluminium": "#512DA8"}

    fig, axs = plt.subplots(ncols=2, figsize=(15, 5))

    for k, metal in enumerate(metal_names): 
        df = load_metal_data(metal)
        total_num_feature = df.shape[1] - 2
        num_total_data = df.shape[0]
        num_feature_left = total_num_feature-df.isnull().sum(axis=1).to_numpy()
        freq = Counter(num_feature_left)

        num_feat, count_freq = zip(*sorted(freq.most_common(), key=lambda x: x[0]))
        count_freq = np.array(count_freq)/num_total_data * 100
        print(count_freq)

        x_pos = list(range(max(num_feat)+1))
        count_freq_all = []
        j = 0
        for i in x_pos:
            if i not in num_feat:
                count_freq_all.append(0)
            else:
                count_freq_all.append(count_freq[j])
                j += 1 

        axs[k].bar(x_pos, count_freq_all, color=metal_to_color[metal], zorder=3)
        axs[k].set_xlabel("Number of Features")
        axs[k].set_ylabel("Percentage of Datapoint")
        axs[k].set_title(f"Percentage of number of Features in {metal}")
        axs[k].grid(zorder=0)
    
    fig.savefig(f"img/data_vis/freq_feat.pdf")

@save_figure("figure/PCA_multi_task.pdf")
def plot_feature_PCA():

    fig, axs = plt.subplots(nrows=2, ncols=5,figsize=(20, 10) ,subplot_kw=dict(projection='3d'))
    axs = axs.flatten()

    def plot_PCA_metal(ax, metal_name):
        data = load_metal_data(metal_name)
        data = data.dropna()
        price = np.log(data["Price"])
        
        data = data.loc[:, data.columns != "Price"]
        data = data.loc[:, data.columns != "Date"].to_numpy()

        pca = PCA(n_components=3)
        reduced_data = pca.fit_transform(data)

        num_data_show = 200

        x, y, z = reduced_data[:num_data_show, :].T
        ax.scatter3D(x, y, z, c=price[:num_data_show], s=10.0, cmap=plt.cm.coolwarm)
        ax.view_init(20, -120)
        ax.set_title(metal_to_display_name[metal_name])
    
    for i, metal in enumerate(metal_names):
        plot_PCA_metal(axs[i], metal)
    
    return fig, axs

@save_figure("figure/all_data.pdf")
def plot_all_data():
    all_output_data = [
        load_transform_data(metal, 22)[1] 
        for metal in metal_names
    ]
    x = all_output_data[0]["Date"]

    dates, text_dates = parse_series_time(x, x[0])
    text_dates = [np.datetime64(date) for date in text_dates]

    num_data = len(all_output_data)
    fig, axs = plt.subplots(nrows=num_data, figsize=(20, 20))

    for i, ax in enumerate(axs):
        ax.plot(text_dates, all_output_data[i]["Price"], color="k")
        plot_axis_date(ax, text_dates, month_interval=18)
        ax.set_title(metal_to_display_name[metal_names[i]])
        ax.grid()
    
    fig.tight_layout()
    
    return fig, axs

def stationary_test(is_tabulate=True):
    get_p_val = lambda metal : adfuller(load_transform_data(metal, 22)[1]["Price"])[1]
    stationary_result = [
        (metal_to_display_name[metal], get_p_val(metal), "✅" if get_p_val(metal) < 0.05 else "❌")
        for metal in metal_names
    ]

    if is_tabulate:
        print(tabulate(stationary_result, headers=["Commodity", "P-Value", "Is Stationary"], tablefmt="grid"))
    else:
        dataframe = {
            header: data 
            for header, data in zip(["Commodity", "P-Value", "Is Stationary"], zip(*stationary_result))
        }
        return pd.DataFrame(dataframe)

def stationary_test_features():
    data = load_metal_data(
        "copper", global_modifier=GlobalModifier(CompressMethod(3, "pca"))
    )
    # data = data.loc[:, data.columns != "Price"]
    data = data.loc[:, data.columns != "Date"]

    print(adfuller(data["FeatureFamily.Feature1"]))
    # get_p_val = lambda metal : adfuller(load_transform_data(metal, 22)[1]["Price"])
    # print(get_p_val("copper"))

    print(coint_johansen(data, 0, 1).cvt)
    # print(coint_johansen(data, 0, 0).cvt)
    
    print(coint_johansen(data, 0, 1).lr1)
    # print(coint_johansen(data, 0, 0).lr2)
    
def corr_test(list_a, list_b, test_name, all_test, is_verbose=False):
    corr_value = []
    p_value = []

    # P-value is low, so they are correlated....
    for n, test in zip(test_name, all_test):
        r, p = test(list_a, list_b)
        corr_value.append(r)
        p_value.append(p)
        if is_verbose:
            print(f"{n}: {r:.5f} with P-Value: {p:.5f}")        
    
    return corr_value, p_value

@save_figure("figure/all_correlation.pdf")
def plot_correlation_all():
    
    data = [
        load_transform_data(metal, 22)[1]["Price"]
        for metal in metal_names
    ]

    names = [metal_to_display_name[metal] for metal in metal_names]
    num_metals = len(metal_names) 
    
    test_name = ["Peason", "Spearman", "Kendell"]
    all_test = [stats.pearsonr, stats.spearmanr, stats.kendalltau]
    num_test = len(test_name)
    
    fig, axes = plt.subplots(ncols=num_test, nrows=2, figsize=(20, 10))
    
    for name, test, ax in zip(test_name, all_test, axes.T): 
        ax_top, ax_bot = ax
        correlation = np.zeros((num_metals, num_metals))
        is_correlated = np.zeros((num_metals, num_metals))

        for i, d1 in enumerate(data):
            for j, d2 in enumerate(data):
                test_result = test(d1, d2)
                correlation[i, j] = test_result[0]
                if i != j:
                    is_correlated[i, j] = test_result[1] < 0.05
                else:
                    is_correlated[i, j] = False
        
        ax_top.set_title(name) 
        plot_heat_map(ax_top, correlation, names, names, xlabel="Commodities", ylabel="Commodities", round_acc=2)
        plt.setp(ax_top.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        num_note_to_date = {
            i : names[i]
            for i in range(10)
        }

        G = nx.from_numpy_matrix(is_correlated) 
        options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
        pos = nx.spring_layout(G, k=0.5)
        nx.draw(
            G, with_labels=True, pos=pos,
            node_color='orange', node_size=1500, 
            edge_color='black', linewidths=1, 
            font_size=5, ax=ax_bot, labels=num_note_to_date
        )
    
    fig.tight_layout()
    return fig, ax

@save_figure("figure/p_value_window_unrelated.pdf")
def plot_window_unrelated():
    return plot_window("natgas", "copper")

@save_figure("figure/p_value_window_related.pdf")
def plot_window_related():
    return plot_window("nickel", "palladium")

def plot_window(metal1, metal2):
    _, data1 = load_transform_data(metal1, 22)
    _, data2 = load_transform_data(metal2, 22)

    title = f"Sliding Window ({metal_to_display_name[metal1]} vs {metal_to_display_name[metal2]})" 
    fig, axes = plot_correlation_window(data1, data2, title=title, skip_size=1)
    return fig, axes

def plot_correlation_window(data_al, data_cu, 
    window_size=260, skip_size=66, metric="CRPS", show_p_value=True, title="Sliding Window",
    load_json_path="save-hyperparam/test-mlt-gp-window/all_result_window.json"):
    
    all_test = [stats.pearsonr, stats.spearmanr, stats.kendalltau]
    test_name = ["Peason", "Spearman", "Kendell"]
    color_list = ["#ff7500", "#0062b8", "#d6022a"]
    color_list2 = ["#0074bf", "#f24a00", "#00db64"]

    total_len = (len(data_al)-window_size) // skip_size

    all_corr, all_p_val, all_date = [], [], []
    for i in range(total_len):
        start_ind = i*skip_size
        data_al_wind = data_al.iloc[start_ind:start_ind+window_size]
        data_cu_wind = data_cu.iloc[start_ind:start_ind+window_size]

        test_al, test_cu = data_al_wind["Price"], data_cu_wind["Price"]
        corr, p_val = corr_test(test_al, test_cu, test_name, all_test)
        all_corr.append(corr)
        all_p_val.append(p_val)

        start_day = datetime.strptime(data_al_wind.iloc[0]["Date"], '%Y-%m-%d')
        end_day = datetime.strptime(data_al_wind.iloc[-1]["Date"], '%Y-%m-%d')
        middle = start_day + (end_day - start_day)/2
        all_date.append(np.datetime64(middle.strftime('%Y-%m-%d')))
        
    fig, axes = plt.subplots(nrows=2, figsize=(15, 6), sharex=True)

    for i, (color, n, result, result2) in enumerate(zip(color_list, test_name, zip(*all_corr), zip(*all_p_val))):
        axes[0].plot(all_date, result, label=n, linestyle="-", color=color, zorder=3)
        plot_axis_date(axes[0], all_date, month_interval=18)
        if show_p_value:
            axes[1].plot(all_date, result2, label=n, linestyle="-", color=color_list2[i], zorder=3)
            plot_axis_date(axes[1], all_date, month_interval=18)
    
    if not show_p_value:
        results = load_json(load_json_path)
        all_methods = list(results.keys())
        all_metric = list(results[all_methods[0]].keys())
        num_windows = len(results[all_methods[0]][all_metric[0]])

        assert num_windows == total_len

        for i, method in enumerate(all_methods):
            axes[1].plot(
                all_date, 
                results[method][metric], 
                linestyle="-", zorder=3, color=color_list2[i],
                label=method
            )
            plot_axis_date(axes[1], all_date, month_interval=18)
        axes[1].set_ylabel(f"{metric} Over Windows")
    else:
        axes[0].axhline(0.0, color="#1a1a1a")
        axes[1].axhline(0.05, color="#1a1a1a")
        axes[1].set_ylabel("P-Value")
    
    axes[0].grid(zorder=0)
    axes[0].legend()
    axes[0].set_ylabel("Correlation")
    
    axes[1].grid(zorder=0)
    axes[1].legend()
    axes[1].set_xlabel("Middle Date")

    axes[0].set_title(title)
    return fig, axes

@save_figure("figure/p_value_year_unrelated.pdf")
def plot_year_unrelated():
    return plot_years_correlation("natgas", "copper")

@save_figure("figure/p_value_year_related.pdf")
def plot_year_related():
    return plot_years_correlation("nickel", "palladium")

def plot_years_correlation(metal1, metal2):
    _, data1 = load_transform_data(metal1, 22)
    _, data2 = load_transform_data(metal2, 22)

    title = f"Correlation Years ({metal_to_display_name[metal1]} vs {metal_to_display_name[metal2]})" 
    fig, axes = plot_correlation_year(data1, data2, title=title)
    return fig, axes

def plot_correlation_year(data_al, data_cu, 
    start_year=2009, num_year_forward=12, month="05", show_p_value=True, title="Correlation Years",
    load_json_path="save-hyperparam/test-mlt-gp/all_result.json"):
    
    test_name = ["Peason", "Spearman", "Kendell"]
    all_test = [stats.pearsonr, stats.spearmanr, stats.kendalltau]
    color_list = ["#ff7500", "#0062b8", "#d6022a"]
    color_list2 = ["#0074bf", "#f24a00", "#00db64"]

    all_corr, all_p_val = [], []
    years = [str(start_year + i) for i in range(num_year_forward+1)]
    all_date = data_al["Date"].to_list()
    for i in range(len(years)-1):
        start_ind = find_sub_string(all_date, f"{years[i]}-{month}")
        end_ind = find_sub_string(all_date, f"{years[i+1]}-{month}")

        al_data = data_al.iloc[start_ind:end_ind]
        cu_data = data_cu.iloc[start_ind:end_ind]

        corr, p_val = corr_test(
            al_data["Price"], cu_data["Price"], 
            test_name, all_test
        )
        all_corr.append(corr)
        all_p_val.append(p_val)
    
    def plot_graph(ax, value, y_label, title, color_list):
        for color, n, result in zip(color_list, test_name, zip(*value)):
            x_val = np.arange(len(result)*2, step=2)
            ax.plot(x_val+1, result, label=n, linestyle="--", color=color, zorder=3)
            ax.scatter(x_val+1, result, color=color, marker="s", zorder=3)
        
        ax.grid(zorder=0)
        ax.legend()
        ax.set_xticks(np.arange(len(years)*2, step=2))
        ax.set_xticklabels([f"{y}-{month}"for y in years])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))

        ax.set_ylabel(y_label)
        ax.set_title(title)
    
    def plot_bar(ax, type_eval="CRPS"):
        results = load_json(load_json_path)

        all_methods = list(results.keys())
        all_metric = list(results[all_methods[0]].keys())
        num_years = len(results[all_methods[0]][all_metric[0]])

        colors = ["#0074bf", "#f24a00", "#00db64"]

        loc_list = np.arange(0, num_years*2, step=2)
        width = 0.3
        for i, loc in enumerate(loc_list):        
            for j, method in enumerate(all_methods): 
                ax.bar(
                    x=loc+width*(j-1.5)+width/2 + 1.0, 
                    height=results[method][type_eval][i], 
                    width=width, 
                    label=None if i != 0 else method,
                    color=colors[j], 
                    edgecolor = "k",
                    zorder=3,
                ) 
        
        ax.grid(zorder=0) 
        ax.legend()
    
    fig, axes = plt.subplots(nrows=2, figsize=(15, 6), sharex=True)
    plot_graph(axes[0], all_corr, y_label="Correlation", title="Correlation and Years", color_list=color_list)

    if not show_p_value:
        plot_bar(axes[1])
    else:
        plot_graph(axes[1], all_p_val, y_label="P-Value", title="P-Value and Years", color_list=color_list2)

    axes[-1].set_xlabel("Years")
    axes[-1].set_ylabel("CRPS Error")
    axes[1].axhline(0.05, color="#1a1a1a")
    axes[0].set_title(title)

    return fig, axes


def main():
    # explore_data_overall()
    # check_data()
    pass


if __name__ == '__main__':
    main()