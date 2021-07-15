import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import torch
from collections import Counter

from utils.data_preprocessing import load_transform_data, parse_series_time, load_metal_data
from utils.data_structure import DatasetTaskDesc

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

def main():
    plot_frequency_features()


if __name__ == '__main__':
    main()