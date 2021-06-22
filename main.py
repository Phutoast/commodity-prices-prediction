import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import random

from utils.data_preprocessing import load_transform_data, walk_forward, prepare_dataset, find_missing_data
from utils.data_structure import DisplayPrediction, Hyperparameters, pack_data
from utils.data_visualization import visualize_time_series
from utils.calculation import PerformanceMetric
from models.ARIMA import ARIMAModel, BaseModel

from examples.ARIMA_plot import example_ARIMA_simple_predtion_plot

np.random.seed(48)
random.seed(48)

def main():
    example_ARIMA_simple_predtion_plot()

if __name__ == '__main__':
    main()
