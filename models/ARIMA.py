import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from models.full_AR_model import FullARModel
from statsmodels.tsa.arima_model import ARIMAResults

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from utils import data_visualization

class ARIMAModel(FullARModel):
    """
    Auto Regressive Integrated Moving Average Model wrapper. 

    Args:
        order: The order of the ARIMA model
    """
    def __init__(self, train_data, model_hyperparam):
        super().__init__(train_data, model_hyperparam)
        assert self.hyperparam["len_inp"] == 0
    
    def predict_fix_step(self, step_ahead, ci, is_sample):
        self.model = self.build_model()
        result = self.model.get_forecast(step_ahead)
        summary_frame = result.summary_frame()
        mean_pred = summary_frame["mean"].to_list()

        if not is_sample:
            upper_pred, lower_pred = result.conf_int(alpha=1-ci).T 
            return mean_pred, list(upper_pred), list(lower_pred)
        else:
            assert "norm" in result.dist.__class__.__name__

            rv = multivariate_normal(
                mean=mean_pred,
                cov=np.diag(summary_frame["mean_se"].to_numpy()**2)
            ).rvs(size=self.hyperparam["sample_size"])

            return rv

    
    def build_model(self):
        order = self.hyperparam["order"]
        return ARIMA(np.squeeze(self.all_data), order=order).fit()

    def save(self, path):
        self.model.save(path + ".pkl")
        np.save(path + "_all_data.npy", self.all_data)
    
    def load(self, path):
        self.model = ARIMAResults.load(path + ".pkl")
        self.all_data = np.load(path + "_all_data.npy")