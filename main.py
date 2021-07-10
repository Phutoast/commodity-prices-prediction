import numpy as np
import random
import argparse
import torch

from examples.simple_example import example_plot_all_algo_lag, example_plot_walk_forward
from utils.others import create_folder
from models.ind_multi_model import IndependentMultiModel

import warnings
warnings.filterwarnings("ignore")

np.random.seed(48)
random.seed(48)
torch.manual_seed(48)
torch.random.manual_seed(48)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-type", help="Type of ", type=str)
    args = parser.parse_args()

    test_type = args.test_type
    create_folder("save")
    
    exp_setting = {
        "task": [
            ("GP-Test", 0, 4, 100),
            ("GP", 22, 5, 100),
        ], "algo": IndependentMultiModel
    }
    
    if test_type == "f":
        example_plot_all_algo_lag(
            exp_setting, is_save=True, is_load=False,
            load_path="multi-out-1"
            # load_path="07-10-21-12-15-06-multi-out-1"
        )
    elif test_type == "w":
        example_plot_walk_forward(exp_setting, "Walk-Forward",
            is_save=True, is_load=False,
            load_path="Walk-Forward"
        )


if __name__ == '__main__':
    main()
