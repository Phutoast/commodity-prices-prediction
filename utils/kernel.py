import numpy as np

def find_distance(matrix1, matrix2):
    num_data1, _ = matrix1.shape
    num_data2, _ = matrix2.shape
    
    G = np.sum(matrix1*matrix1, axis=1, keepdims=True)
    H = np.sum(matrix2*matrix2, axis=1, keepdims=True)
    Q = np.tile(G, (1, num_data2))
    R = np.tile(H.T, (num_data1, 1))

    dists = Q + R - 2*np.dot(matrix1, matrix2.T)
    return dists


def create_gram_matrix(data1, data2, length):
    H = find_distance(data1, data2)
    return np.exp(-H/(2*length**2))

def find_median_length(data, is_dist_sqrt=False):
    dists = find_distance(data, data)
    dists = dists - np.tril(dists)
    if is_dist_sqrt:
        dists = np.sqrt(dists)
    return np.sqrt(0.5*np.median(dists[dists>0]))

def permute_matrix(matrix, index, num_data):
    R = np.tile(index, (1, num_data))
    Q = R.T
    perm_L = matrix[R, Q]
    return perm_L

def cal_test_stat(data1, data2, length1=None, length2=None, is_dist_sqrt=False):
    data_size, _ = data1.shape

    if length1 is None:
        rbf_len1 = find_median_length(data1, is_dist_sqrt)
    else:
        rbf_len1 = length1
    
    if length2 is None:
        rbf_len2 = find_median_length(data2, is_dist_sqrt)
    else:
        rbf_len2 = length2 

    K = create_gram_matrix(data1, data1, rbf_len1)
    L = create_gram_matrix(data2, data2, rbf_len2)

    H = np.eye(data_size) - 1/data_size * np.ones((data_size, data_size))
    Kc = np.dot(np.dot(H, K), H)
    test_stat = 1/(data_size**2) * np.sum(Kc.T*L)
    
    return Kc, test_stat, L, H

