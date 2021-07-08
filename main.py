import numpy as np
import random

from examples.simple_example import example_plot_all_algo_lag, example_plot_walk_forward
from utils.others import create_folder

import warnings
warnings.filterwarnings("ignore")

np.random.seed(48)
random.seed(48)

def main():
    create_folder("save")
    # example_plot_all_algo_lag(
    #     "GP-Multi-Out", load_path=("gp-multi-out", "model"),
    #     is_save=True, is_load=False
    # )
    example_plot_all_algo_lag(
        "GP", load_path=("gp", "model"),
        is_save=True, is_load=False
    )
    # example_plot_walk_forward("GP-3")

if __name__ == '__main__':
    main()
