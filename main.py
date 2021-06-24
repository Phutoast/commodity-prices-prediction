import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import random

from utils.data_preprocessing import load_transform_data, walk_forward, prepare_dataset, find_missing_data
from utils.data_structure import DisplayPrediction, Hyperparameters, pack_data
from utils.data_visualization import visualize_time_series, visualize_walk_forward
from utils.calculation import PerformanceMetric
from models.GP import SimpleGaussianProcessModel, FeatureGP
import pickle

from examples.ARIMA_plot import examples_Mean_simple_prediction_plot
from examples.Walk_Forward_Plot import example_ARIMA_walk_forward_plot
from examples.Simple_GP import simple_GP_plot, feature_GP_plot

import warnings
warnings.filterwarnings("ignore")

np.random.seed(48)
random.seed(48)

def main():
    # example_ARIMA_walk_forward_plot()
    # examples_Mean_simple_prediction_plot()
    # simple_GP_plot()
    # feature_GP_plot()

if __name__ == '__main__':
    main()
