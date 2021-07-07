from statsmodels.tsa.arima.model import ARIMA
import numpy as np

from models.full_AR_model import FullARModel

class ARIMAModel(FullARModel):
    """
    Auto Regressive Integrated Moving Average Model wrapper. 

    Args:
        order: The order of the ARIMA model
    """
    def __init__(self, train_data, model_hyperparam):
        super().__init__(train_data, model_hyperparam)
        assert self.hyperparam["len_inp"] == 0
    
    def predict_fix_step(self, step_ahead, ci):
        order = self.hyperparam["order"]
        model = ARIMA(np.squeeze(self.all_data), order=order).fit()
        result = model.get_forecast(step_ahead)
        upper_pred, lower_pred = result.conf_int(alpha=1-ci).T 
        mean_pred = result.summary_frame()["mean"].to_list()

        return mean_pred, list(upper_pred), list(lower_pred)