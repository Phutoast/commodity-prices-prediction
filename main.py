import numpy as np
import random

from examples.simple_example import example_plot_all_algo_lag, example_plot_walk_forward

import warnings
warnings.filterwarnings("ignore")

np.random.seed(48)
random.seed(48)

def main():
    # example_plot_all_algo_lag()
    example_plot_walk_forward()

if __name__ == '__main__':
    main()
