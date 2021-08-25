import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess

def make_vec(x):
    return np.expand_dims(x, axis=1)

def generate_extinct_gaussian(extinction_rate, length=1200, phi=0.2):
    start_val = np.random.randn(2, 1)
    curr_val = start_val
    matrix = np.diag([phi, phi])

    results = []
    for _ in range(length):
        results.append(curr_val)

        accept = True
        while accept:
            U = np.random.rand()
            noise1, noise2 = np.random.randn(2)
            if U > extinction_rate or noise1**2 + noise2**2 > 1:
                curr_noise = np.array([[noise1], [noise2]])
                accept = False

        curr_val = np.dot(matrix, curr_val) + curr_noise

    X, Y = np.hstack(results)
    return make_vec(X), make_vec(Y)

def generate_non_linear(length=500):
    start_val = np.random.randn(2, 1)
    curr_val = start_val

    results = []
    for _ in range(length):
        noise1, noise2 = np.random.randn(2)

        results.append(curr_val)
        curr_val = np.array([
            [noise1*curr_val[1, 0]], 
            [noise2]
        ])
    
    X, Y = np.hstack(results)
    return make_vec(X), make_vec(Y)

def generate_time_series(length=500):
    # Example From https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima_process.ArmaProcess.html
    arparams = np.array([.75, -.25])
    maparams = np.array([.65, .35])
    ar = np.r_[1, -arparams]
    ma = np.r_[1, maparams]
    arma_process = ArmaProcess(ar, ma)

    X, Y, Z = [
        np.expand_dims(arma_process.generate_sample(length), axis=1)
        for i in range(3)
    ]

    return X, Y, Z

def generate_iid(size=200): 
    mean = np.zeros((10,))
    covariance = np.eye(10)

    A = np.random.choice([0, 1], size=(size, 10), p=[0.5, 0.5])
    A[A==0] = -1

    random_cov = np.random.multivariate_normal(mean, covariance, size=10)
    noise = np.random.multivariate_normal(
        mean, np.dot(random_cov, random_cov.T)
    )

    test_x = np.random.multivariate_normal(mean, covariance, size=size)
    test_y_ind = np.random.multivariate_normal(mean, covariance, size=size)
    test_y_not_ind = (np.sinh(test_x)*0.1 + noise*0.01)*A;

    return test_x, test_y_ind, test_y_not_ind
