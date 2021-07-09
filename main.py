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
torch.manual_seed(48)
torch.random.manual_seed(48)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", help="Algorithm Name", type=str)
    parser.add_argument("--test-type", help="Type of ", type=str)
    args = parser.parse_args()

    algo = args.algo
    test_type = args.test_type
    create_folder("save")
    
    exp_setting = [
        ("GP-Test", 0, 4, 100),
        ("GP", 22, 5, 100),
    ]

    if test_type == "f":
        example_plot_all_algo_lag(
            exp_setting, is_save=False, is_load=True,
            load_path="07-09-21-19-17-16-test-multi-1"
        )
    elif test_type == "w":
        example_plot_walk_forward(algo)

if __name__ == '__main__':
    main()
