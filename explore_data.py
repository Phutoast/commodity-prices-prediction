import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import torch
from collections import Counter

from utils.data_preprocessing import load_transform_data, parse_series_time, load_metal_data, parse_series_time
from utils.data_structure import DatasetTaskDesc
from utils.data_visualization import plot_axis_date
import datetime

from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA

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


    # identity_modifier = GlobalModifier((0, "id"))
    metal = "aluminium"
    _, data_al = load_transform_data(metal, 22)
    
    metal = "copper"
    _, data_cu = load_transform_data(metal, 22)

    print("Aluminium")
    result = adfuller(data_al["Price"])
    print("P-value:", result[1])
    print("---"*5)
    
    print("Copper")
    result = adfuller(data_cu["Price"])
    print("P-value:", result[1])
    print("---"*5)

    # Both P-values are less that 0.05, thus both are stationary.


def main():
    explore_data_overall()


if __name__ == '__main__':
    main()