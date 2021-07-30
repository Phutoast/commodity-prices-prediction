import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import torch
from collections import Counter
import matplotlib.ticker as ticker

from utils.data_preprocessing import load_transform_data, parse_series_time, load_metal_data, parse_series_time
from utils.data_structure import DatasetTaskDesc
from utils.data_visualization import plot_axis_date
from datetime import datetime

from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA
import scipy.stats as stats

metal_to_color = {"copper": "#D81B60", "aluminium": "#512DA8"}

def plot_frequency_features():
    all_metals = ["aluminium", "copper"]

    fig, axs = plt.subplots(ncols=2, figsize=(15, 5))

    for k, metal in enumerate(all_metals): 
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

def check_data():
    metal = "aluminium"
    data = load_metal_data(metal)
    data = data.dropna()

    metal = "copper"
    data1 = load_metal_data(metal)
    data1 = data1.dropna()

    # Confirm that they are the same
    print(data["Date"].to_list() == data1["Date"].to_list())

def plot_feature_PCA_overtime():
    metal = "aluminium"
    data = load_metal_data(metal)
    data = data.dropna()
    
    data = data.loc[:, data.columns != "Price"]
    data = data.loc[:, data.columns != "Date"].to_numpy()

    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(data)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x, y, z = reduced_data[:100, :].T
    ax.scatter3D(x[0], y[0], z[0], c="#1a1a1a", s=20.0)
    ax.scatter3D(x, y, z, c="#1a1a1a", s=1.0)
    ax.plot(x,y,z, color="#5a08bf")
    plt.show()

def plot_multiple_graph(ax, list_of_tuple, color_list, 
    name_list, is_point_label=False, offset=0.0, linestyle="--"):

    for color, n, result in zip(color_list, name_list, zip(*list_of_tuple)):
        ax.plot(np.arange(len(result))+offset, result, label=n, linestyle=linestyle, color=color, zorder=3)
        if is_point_label:
            ax.scatter(np.arange(len(result))+offset, result, color=color, marker="s", zorder=3)

def explore_data_overall():

    def plot_all_data(x, all_data):
        dates, text_dates = parse_series_time(x, x[0])
        text_dates = [np.datetime64(date) for date in text_dates]

        num_data = len(all_data)
        fig, axs = plt.subplots(nrows=num_data, figsize=(15, 5))

        for i, ax in enumerate(axs):
            ax.plot(text_dates, all_data[i]["Price"], color="k")
            plot_axis_date(ax, text_dates, month_interval=18)
            ax.grid()
    
        plt.show()
    
    def stationary_test():
        # Augmented Dickey-Fuller
        print("Aluminium")
        result = adfuller(data_al["Price"])
        print("P-value:", result[1])
        print("---"*5)
        
        print("Copper")
        result = adfuller(data_cu["Price"])
        print("P-value:", result[1])
        print("---"*5)
    
    def corr_test(list_a, list_b, is_verbose=False):
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

    def find_sub_string(str_list, substr):
        for i, s in enumerate(str_list):
            if substr in s:
                return i
        
        assert False
    
    def plot_correlation_year():
        all_corr, all_p_val = [], []
        years = [str(2005 + i) for i in range(17)]
        all_date = data_al["Date"].to_list()
        for i in range(len(years)-1):
            start_ind = find_sub_string(all_date, f"{years[i]}-05")
            end_ind = find_sub_string(all_date, f"{years[i+1]}-05")

            al_data = data_al.iloc[start_ind:end_ind]
            cu_data = data_cu.iloc[start_ind:end_ind]

            corr, p_val = corr_test(al_data["Price"], cu_data["Price"])
            all_corr.append(corr)
            all_p_val.append(p_val)
        
        def plot_graph(ax, value, y_label, title):
            plot_multiple_graph(ax, value, color_list, test_name, is_point_label=True, offset=0.5)
            
            ax.grid(zorder=0)
            ax.legend()
            ax.set_xticks(np.arange(len(years)))
            ax.set_xticklabels(years)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))

            ax.set_ylabel(y_label)
            ax.set_title(title)
        
        fig, axes = plt.subplots(nrows=2, figsize=(15, 6), sharex=True)
        plot_graph(axes[0], all_corr, y_label="Correlation", title="Correlation and Years")
        plot_graph(axes[1], all_p_val, y_label="P-Value", title="P-Value and Years")
        axes[-1].set_xlabel("Years")
        
        plt.show()
    
    def plot_correlation_window():

        # This is about a year...
        window_size = 260
        all_corr, all_p_val, all_date = [], [], []
        for i in range(len(data_al)-window_size):
            data_al_wind = data_al.iloc[i:i+window_size]
            data_cu_wind = data_cu.iloc[i:i+window_size]

            test_al, test_cu = data_al_wind["Price"], data_cu_wind["Price"]
            corr, p_val = corr_test(test_al, test_cu)
            all_corr.append(corr)
            all_p_val.append(p_val)

            start_day = datetime.strptime(data_al_wind.iloc[0]["Date"], '%Y-%m-%d')
            end_day = datetime.strptime(data_al_wind.iloc[-1]["Date"], '%Y-%m-%d')
            middle = start_day + (end_day - start_day)/2
            all_date.append(np.datetime64(middle.strftime('%Y-%m-%d')))
            
        fig, axes = plt.subplots(nrows=2, figsize=(15, 6), sharex=True)

        for color, n, result, result2 in zip(color_list, test_name, zip(*all_corr), zip(*all_p_val)):
            axes[0].plot(all_date, result, label=n, linestyle="-", color=color, zorder=3)
            axes[1].plot(all_date, result2, label=n, linestyle="-", color=color, zorder=3)
            plot_axis_date(axes[0], all_date, month_interval=18)
            plot_axis_date(axes[1], all_date, month_interval=18)

        axes[0].grid(zorder=0)
        axes[0].legend()
        axes[0].set_ylabel("Correlation")
        
        axes[1].grid(zorder=0)
        axes[1].legend()
        axes[1].axhline(0.05, color="#1a1a1a")
        axes[1].set_ylabel("P-Value")
        axes[1].set_xlabel("Middle Date")

        axes[0].set_title("Sliding Window")

        plt.show()

    metal = "aluminium"
    _, data_al = load_transform_data(metal, 22)
    
    metal = "copper"
    _, data_cu = load_transform_data(metal, 22)

    all_test = [stats.pearsonr, stats.spearmanr, stats.kendalltau]
    test_name = ["Peason", "Spearman", "Kendell"]
    color_list = ["#ff7500", "#0062b8", "#d6022a"]

    plot_correlation_year()



def main():
    explore_data_overall()


if __name__ == '__main__':
    main()