import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import random

from utils.data_preprocessing import load_transform_data, walk_forward, prepare_dataset, find_missing_data
from utils.data_structure import DisplayPrediction, Hyperparameters, pack_data
from utils.data_visualization import visualize_time_series, visualize_walk_forward
from utils.calculation import PerformanceMetric
from models.GP import FeatureGP
import pickle

from examples.Walk_Forward_Plot import example_ARIMA_walk_forward_plot
from examples.simple_example import example_plot_all_algo_lag 

import warnings
warnings.filterwarnings("ignore")

np.random.seed(48)
random.seed(48)

def main():
    # example_ARIMA_walk_forward_plot()
    example_plot_all_algo_lag()

if __name__ == '__main__':
    main()
