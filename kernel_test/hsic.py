import numpy as np
import scipy.io
from scipy.signal import lfilter
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.arima_process import ArmaProcess

from utils.kernel import find_median_length, create_gram_matrix, cal_test_stat, permute_matrix
from utils.kernel_examples import generate_time_series

def bootstrap_series_2(length, num_path):
    ln = 20
    ar = np.exp(-1/ln)
    var = 1-np.exp(-2/ln)

    random_data = np.random.randn(length, num_path)
    w = np.sqrt(var) * random_data
    a = [1, -ar]

    # https://stackoverflow.com/questions/16936558/matlab-filter-not-compatible-with-python-lfilter
    return lfilter([1], a, w, axis=0)


# https://github.com/Phutoast/Kernel-Statistical-Test
# https://github.com/kacperChwialkowski/wildBootstrap
def wild_bootstrap_HSIC(data1, data2, alpha, num_bootstrap=300, length1=None, length2=None):
    len_data, _ = data1.shape
    Kc, _, L, H = cal_test_stat(data1, data2, length1, length2, is_dist_sqrt=True)
    Lc = np.dot(np.dot(H, L), H)
    test_stat_matrix = Kc*Lc
    process = bootstrap_series_2(len_data, num_bootstrap)
    test_stat = len_data * np.mean(test_stat_matrix.flatten())

    perm_stat = []
    for i in range(num_bootstrap):
        mean = np.mean(process[:, i])
        process_center = np.expand_dims(process[:, i]-mean, axis=1)
        matFix = np.dot(process_center, process_center.T)
        perm_stat.append(len_data * np.mean(matFix*test_stat_matrix))
    
    perm_stat = sorted(perm_stat)
    threshold = perm_stat[round((1-alpha)*num_bootstrap)]

    return threshold, test_stat

def is_dependent_wild_HSIC(X, Y, num_bootstrap=500):
    threshold, stat = wild_bootstrap_HSIC(X, Y, 0.05, num_bootstrap)
    return threshold <= stat


def main():
    X, Y, Z = generate_time_series(length=1000)

    threshold, stat = wild_bootstrap_HSIC(X, Y, 0.05, 300)
    if threshold > stat:
        print("Accept: Independent")
    else:
        print("Reject: Dependent")
    
    threshold, stat = wild_bootstrap_HSIC(X+Z, X+Z, 0.05, 300)
    if threshold > stat:
        print("Accept: Independent")
    else:
        print("Reject: Dependent")


if __name__ == '__main__':
    main()
