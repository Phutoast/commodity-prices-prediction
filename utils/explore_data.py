import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import torch
from collections import Counter
import matplotlib.ticker as ticker
import copy
from datetime import datetime

from utils.data_preprocessing import load_metal_data, parse_series_time, GlobalModifier, load_dataset_from_desc, get_data
from utils.data_structure import DatasetTaskDesc
from utils.data_visualization import plot_axis_date, plot_heat_map, cluster_label_to_dict, print_tables_side_by_side, save_figure
from utils.others import find_sub_string, load_json, find_all_metal_names, create_folder, dump_json
from utils.data_structure import CompressMethod
from experiments.gen_experiment import gen_datasets
from experiments.metal_desc import metal_to_display_name

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

from sklearn.decomposition import PCA
import scipy.stats as stats
from tabulate import tabulate
import networkx as nx
from sklearn.cluster import SpectralClustering
from tslearn.clustering import KernelKMeans, TimeSeriesKMeans, KShape
import tslearn
    
from hyppo.time_series import DcorrX
from kernel_test.hsic import wild_bootstrap_HSIC

import json

metal_names = find_all_metal_names()

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

        num_data_show = 300

        x, y, z = reduced_data[num_data_show:, :].T
        ax.scatter3D(x, y, z, c=price[num_data_show:], s=10.0, cmap=plt.cm.coolwarm)
        ax.view_init(20, -120)
        ax.set_title(metal_to_display_name[metal_name])
    
    for i, metal in enumerate(metal_names):
        plot_PCA_metal(axs[i], metal)
    
    return fig, axs

@save_figure("figure/all_data.pdf")
def plot_all_data():
    all_output_data = [
        get_data(metal, is_price_only=False, is_feat=False)
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
    get_p_val = lambda metal : adfuller(get_data(metal))[1]
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
    print(coint_johansen(data, 0, 1).cvt) 
    print(coint_johansen(data, 0, 1).lr1)
    
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

def correlation_over_dataset(test_name, all_test, test_corr):
    list_correlation, list_is_correlated, list_addi_info = [], [], []
    names = [metal_to_display_name[metal] for metal in metal_names]
    num_metals = len(metal_names) 
    
    data = [
        np.expand_dims(get_data(metal).to_numpy(), axis=1)
        for metal in metal_names
    ]

    for name, test in zip(test_name, all_test): 
        correlation = np.zeros((num_metals, num_metals))
        is_correlated = np.zeros((num_metals, num_metals))
        addi_info = np.zeros((num_metals, num_metals))

        for row in range(len(data)):
            for col in range(row):
                print(f"Running {row} and {col}")
                stat, is_corr, addi_compare = test_corr(test, data[row], data[col])
                correlation[row, col] = stat
                addi_info[row, col] = addi_compare
                is_correlated[row, col] = is_corr
        
        is_correlated = is_correlated + is_correlated.T
        correlation = correlation + correlation.T
        np.fill_diagonal(correlation, 1)

        list_correlation.append(correlation)
        list_is_correlated.append(is_correlated) 
        list_addi_info.append(addi_info)
 
    return list_correlation, list_is_correlated, list_addi_info

def show_test_stat_graph(names, test_name, list_correlation, list_is_correlated, size=1500, k=0.5):
    fig, axes = plt.subplots(ncols=len(test_name), nrows=2, figsize=(20, 10))
    for name, correlation, is_correlated, ax in zip(test_name, list_correlation, list_is_correlated, axes.T): 
        ax_top, ax_bot = ax
        ax_top.set_title(name) 
        plot_heat_map(ax_top, correlation, names, names, xlabel="Commodities", ylabel="Commodities", round_acc=2)
        plt.setp(ax_top.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        num_note_to_date = {
            i : names[i]
            for i in range(10)
        }

        np.save(f"exp_result/graph_result/{name.lower()}_test_graph.npy", is_correlated)

        G = nx.from_numpy_matrix(is_correlated) 
        pos = nx.spring_layout(G, k=k)
        # pos = nx.circular_layout(G)
        nx.draw(
            G, with_labels=True, pos=pos,
            node_color='orange', node_size=size, 
            edge_color='black', linewidths=1, 
            font_size=5, ax=ax_bot, labels=num_note_to_date
        )
    
    fig.tight_layout()
    return fig, axes

@save_figure("figure/all_correlation.pdf")
def plot_correlation_all():
    create_folder("exp_result/graph_result")
    names = [metal_to_display_name[metal] for metal in metal_names]
    num_metals = len(metal_names) 
    
    test_name = ["Peason", "Spearman", "Kendell"]
    all_test = [stats.pearsonr, stats.spearmanr, stats.kendalltau]
    num_test = len(test_name)

    def test_corr(test, d1, d2):
        test_result = test(d1.flatten(), d2.flatten())
        return test_result[0], test_result[1] < 0.05, -1
    
    list_correlation, list_is_correlated, _ = correlation_over_dataset(test_name, all_test, test_corr) 
    return show_test_stat_graph(names, test_name, list_correlation, list_is_correlated)

@save_figure("figure/hsic_graph.pdf")
def plot_graph_hsic():
    create_folder("exp_result/graph_result")
    names = [metal_to_display_name[metal] for metal in metal_names]
    num_metals = len(metal_names) 
    
    def run_hsic(X, Y):
        threshold, stat = wild_bootstrap_HSIC(X, Y, 0.05, 1000)
        return stat, threshold <= stat, -1
    
    def run_DcorrX(X, Y):
        stat, p_val, optim_lag = DcorrX(max_lag=0).test(X, Y, reps=1000, workers=-1)
        return stat, p_val < 0.05, optim_lag["opt_lag"]
    
    test_name = ["HSIC", "Distance Correlation"]
    all_test = [run_hsic, run_DcorrX]

    num_test = len(test_name)
    is_save = False

    if is_save:
        list_test_stat, list_is_correlated, list_addi_info = correlation_over_dataset(
            test_name, all_test, 
            lambda test, d1, d2: test(d1, d2)
        )

        for name, test_stat, is_correlated, addi_info in zip(test_name, list_test_stat, list_is_correlated, list_addi_info):
            np.save(f"exp_result/graph_result/{name.lower()}_test_graph.npy", is_correlated)
            np.save(f"exp_result/graph_result/{name.lower()}_test_stat.npy", test_stat)
            np.save(f"exp_result/graph_result/{name.lower()}_test_addi_info.npy", addi_info)
    else:

        list_test_stat = [
            np.load(f"exp_result/graph_result/{name.lower()}_test_stat.npy")
            for name in test_name
        ]
        
        list_is_correlated = [
            np.load(f"exp_result/graph_result/{name.lower()}_test_graph.npy")
            for name in test_name
        ]

    return show_test_stat_graph(names, test_name, list_test_stat, list_is_correlated, size=1500, k=5)

@save_figure("figure/p_value_window_unrelated.pdf")
def plot_window_unrelated():
    return plot_window("natgas", "copper")

@save_figure("figure/p_value_window_related.pdf")
def plot_window_related():
    return plot_window("nickel", "palladium")

def plot_window(metal1, metal2):
    data1 = get_data(metal1, is_price_only=False)
    data2 = get_data(metal2, is_price_only=False)

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
    data1 = get_data(metal1, is_feat=False, is_price_only=False)
    data2 = get_data(metal2, is_feat=False, is_price_only=False)

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

    if not show_p_value:
        axes[-1].set_ylabel("CRPS Error")
    else:
        axes[-1].set_ylabel("P-Value")

    axes[1].axhline(0.05, color="#1a1a1a")
    axes[0].set_title(title)
    axes[0].axhline(0.0, color="#1a1a1a")

    return fig, axes


class CacheSoftDTW(object):
    def __init__(self, data, len_data=10, memory=None):
        if memory is None:
            self.memory = np.ones((len_data, len_data)) * -1
        else:
            self.memory = memory
        self.data = data
    
    def __call__(self, ind1, ind2, gamma=0.1):
        
        if self.memory[ind1, ind2] == -1:
            result = tslearn.metrics.soft_dtw(
                self.data[ind1], self.data[ind2], gamma=gamma
            )
            self.memory[ind1, ind2] = float(result)
            self.memory[ind2, ind1] = float(result)
            return result
        else:
            return self.memory[ind1, ind2]
 
    def save_data(self, path):
        np.save(path, self.memory)

@save_figure("figure/distance_time_series.pdf")
def distance_between_time_series(all_data=True, is_show=True, pre_computed_data=None):
    data = [
        get_data(metal, is_feat=not all_data).to_numpy() 
        for metal in metal_names
    ]
    
    soft_dtw = CacheSoftDTW(data, len(data), pre_computed_data)

    def soft_dtw_divergence(ind1, ind2, gamma=0.1):
        part1 = soft_dtw(ind1, ind2, gamma=gamma)
        part2 = soft_dtw(ind1, ind1, gamma=gamma)
        part3 = soft_dtw(ind2, ind2, gamma=gamma)
        return part1 - 0.5*(part2 + part3)
         
    def euclidean(ind1, ind2):
        return np.sqrt(np.sum((data[ind1] - data[ind2])**2))
    
    def dtw(ind1, ind2):
        return tslearn.metrics.dtw(data[ind1], data[ind2])
    
    name = ["Euclidean", "DTW", "Soft-DTW Divergence"]
    all_test = [euclidean, dtw, soft_dtw_divergence]

    num_metals = len(data)
    
    all_distances = []
    for test in all_test: 
        distance = np.zeros((num_metals, num_metals))
        for i, d1 in enumerate(data):
            for j, d2 in enumerate(data):
                distance[i, j] = test(i, j)
        
        all_distances.append(distance)

    create_folder("result/distance")
    soft_dtw.save_data("result/distance/result.npy")

    if is_show: 
        fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
        names = [metal_to_display_name[metal] for metal in metal_names]

        for ax, matrix, n in zip(axes, all_distances, name):
            plot_heat_map(ax, matrix, names, names, xlabel="Commodities", ylabel="Commodities", round_acc=2)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        fig.tight_layout()
        plt.show()

        return fig, axes
    else:
        return all_distances


def clustering_dataset(is_side_by_side=True, num_cluster=4, is_verbose=True, use_all_data=False):

    if use_all_data:
        base_folder = "result/cluster_result/all_data"
    else:
        base_folder = "result/cluster_result/feat_data"

    create_folder(base_folder)
    result = {}

    def load_price_feature(metal):
        pca_modi = {metal: CompressMethod(3, "pca")}
        dataset_desc, _ = gen_datasets(type_task="metal", modifier=pca_modi, metal_type=[metal], len_dataset=-1)
        features, log_prices = load_dataset_from_desc(dataset_desc[0])
        features = features.loc[:, features.columns != "Date"].to_numpy()
        log_prices = log_prices.loc[:, log_prices.columns != "Date"].to_numpy()
        return np.expand_dims(np.concatenate([features, log_prices], axis=1), axis=0)

    def prepare_table(labels):
        cluster_to_index = cluster_label_to_dict(labels)
        max_csize = 0
        all_data = []
        for i in range(len(cluster_to_index)):
            cluster_metal_name = [
                metal_to_display_name[metal_names[name_i]]
                for name_i in cluster_to_index[i]
            ]

            curr_csize = len(cluster_to_index[i]) 
            max_csize = curr_csize if max_csize < curr_csize else max_csize
            all_data.append(cluster_metal_name)
         
        padded = [data + [None] * (max_csize - len(data)) for data in all_data]
        all_data = np.array(padded).T.tolist()

        return all_data, cluster_to_index
    
    def run_cluster(test_name, list_distance_matrix):
        all_tables, all_headers = [], []
        for name, distance_matrix in zip(test_name, list_distance_matrix):
            spectral_cluster = SpectralClustering(
                n_clusters=num_cluster, assign_labels="discretize", 
                random_state=48, affinity="precomputed"
            )
            cluster = spectral_cluster.fit(distance_matrix)
            labels = cluster.labels_

            all_data, cluster_to_index = prepare_table(labels) 
            all_tables.append(all_data)

            header = [f"Cluster {i+1}" for i in range(len(cluster_to_index))]
            all_headers.append(header)

            result.update({name.lower(): labels.tolist()})

            if not is_side_by_side and is_verbose:
                print(f"{name} Test")
                print(tabulate(all_data, headers=header, tablefmt="grid"))
                print()
                print()
        
        if is_side_by_side and is_verbose:
            print_tables_side_by_side(all_tables, all_headers, [f"{n} Test"for n in test_name], spacing=6)
    
    all_test = [stats.pearsonr, stats.spearmanr, stats.kendalltau]
    test_name = ["Peason", "Spearman", "Kendell"]
    list_correlation, _ = correlation_over_dataset(test_name, all_test)
    run_cluster(test_name, [np.abs(c) for c in list_correlation])

    precomputed_data = np.load("result/distance/result.npy")

    list_distance = distance_between_time_series(
        all_data=use_all_data, is_show=False, 
        pre_computed_data=precomputed_data
    )

    def rbf_affinity(matrix, delta=1.0):
        return np.exp(- matrix ** 2 / (2. * delta ** 2))

    normalize_score = [2, 1, 20]
    
    list_distance = [
        rbf_affinity(l/n) 
        for l, n in zip(list_distance, normalize_score)
    ]

    run_cluster(    
        ["Euclidean", "DTW", "Soft-DTW Divergence"], 
        list_distance
    )

    is_feat = not use_all_data
    data = [
        load_price_feature(metal)
        for metal in metal_names
    ]

    all_dataset = np.concatenate(data, axis=0)
    metric_names = ["euclidean", "dtw", "softdtw"]

    all_tables, all_headers = [], []
    for metric in metric_names:
        params = {"gamma": .05} if metric == "softdtw" else None

        kmean = TimeSeriesKMeans(
            n_clusters=num_cluster, metric=metric, max_iter=5,
            random_state=48, metric_params=params, 
            max_iter_barycenter=5, n_jobs=-1
        )
        labels = kmean.fit_predict(all_dataset)
        
        result.update({metric.lower() + " knn": labels.tolist()})
        
        all_data, cluster_to_index = prepare_table(labels) 
        all_tables.append(all_data)
        
        header = [f"Cluster {i+1}" for i in range(len(cluster_to_index))]
        all_headers.append(header)

        if not is_side_by_side and is_verbose:
            print(metric.capitalize())
            print(tabulate(all_data, headers=header, tablefmt="grid"))
            print()
            print()
    
    if is_side_by_side and is_verbose:
        print_tables_side_by_side(all_tables, all_headers, [n.capitalize() for n in metric_names], spacing=6)

    all_tables, all_headers = [], []
    gak_km = KShape(n_clusters=num_cluster, n_init=3, random_state=48)
    labels = gak_km.fit_predict(all_dataset)

    all_data, cluster_to_index = prepare_table(labels)
    all_tables.append(all_data)
    all_headers.append([f"Cluster {i+1}" for i in range(len(cluster_to_index))])
    
    if is_verbose:
        print_tables_side_by_side(all_tables, all_headers, ["KShape"], spacing=6)

    result.update({"kshape": labels.tolist()})

    if num_cluster == 4:
        expert_cluster = [0, 2, 0, 1, 2, 0, 0, 0, 1, 3]
    elif num_cluster == 5:
        expert_cluster = [0, 3, 0, 2, 3, 1, 1, 1, 2, 4]
    
    result.update({"expert": expert_cluster})

    for k, v in result.items():
        print(k, ":", v)

    dump_json(f"{base_folder}/cluster_{num_cluster}.json", result)

@save_figure("figure/acf_pacf_all.pdf")
def plot_cf_and_acf(metal=None):
    data = [
        get_data(metal).to_numpy() 
        for metal in metal_names
    ]

    if not metal is None:
        data = [get_data(metal).to_numpy()] 

    num_data = len(data)
    
    fig, axes = plt.subplots(ncols=2, nrows=num_data, figsize=(25, 15))

    for i in range(num_data):
        ax_left, ax_right = axes[i, :]
        sm.graphics.tsa.plot_acf(data[i], lags=100, ax=ax_left)
        sm.graphics.tsa.plot_pacf(data[i], lags=100, ax=ax_right)

        name = metal_to_display_name[metal_names[i]]

        ax_left.set_title(f"{name} ACF")
        ax_right.set_title(f"{name} PACF")
    
    fig.tight_layout()  
    
    return fig, axes

def main():
    # explore_data_overall()
    # check_data()
    # plot_correlation_all()
    pass


if __name__ == '__main__':
    main()