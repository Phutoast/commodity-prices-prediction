import numpy as np
import random
import argparse
import torch

from examples.simple_example import example_plot_all_algo_lag, example_plot_walk_forward
from utils.others import create_folder

import warnings
warnings.filterwarnings("ignore")

np.random.seed(48)
random.seed(48)
torch.manual_seed(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", help="Algorithm Name", type=str)
    parser.add_argument("--test-type", help="Type of ", type=str)
    args = parser.parse_args()

    algo = args.algo
    test_type = args.test_type
    create_folder("save")

    if test_type == "f":
        example_plot_all_algo_lag(
            algo, load_path=(algo.lower(), "model"),
            is_save=True, is_load=False
        )
    elif test_type == "w":
        example_plot_walk_forward(algo)

    # example_plot_all_algo_lag(
    #     "GP", load_path=("gp", "model"),
    #     is_save=True, is_load=False
    # )

if __name__ == '__main__':
    main()
