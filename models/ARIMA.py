import warnings
import math
from statsmodels.tsa.arima.model import ARIMA

from models.base_model import BaseModel
from utils.data_structure import pack_data

class ARIMAModel(BaseModel):
    """
    Auto Regressive Integrated Moving Average Model wrapper. 

    Args:
        order: The order of the ARIMA model
        pred_rollout: Prediction rollout of ARIMA model
        upper_rollout: Upperbound on the rollout
        lower_rollout: Lowerbound on the rollout

    """
    def __init__(self, train_data, model_hyperparam):

        # Although repeated, we want to keep the same interface
        super().__init__(train_data, model_hyperparam)
        self.initialize()
    
    def initialize(self):
        """
        Reset all the data for a new prediction.
        """
        self.pred_rollout, self.upper_rollout, self.lower_rollout = [], [], []
    
    def train(self):
        """
        No formal training in ARIMA model as we simply fit the data, 
        while when performing a prediction we will have to refit some of the data
        """
        warnings.warn("There is no formal training in ARIMA model")
    
    def predict_time_step(self, pred_span, ci):
        """
        Predict the ARIMA mdoel given the number of steps ahead, 
            and update the interal value

        Args:
            pred_span: Number of data points we want to predict ahead of time
        """

        order = self.hyperparam["order"]
        model = ARIMA(self.all_data, order=order).fit()
        result = model.get_forecast(pred_span)
        upper_pred, lower_pred = result.conf_int(alpha=1-ci).T 
        mean_pred = result.summary_frame()["mean"].to_list()

        self.upper_rollout += list(upper_pred)
        self.lower_rollout += list(lower_pred)
        self.pred_rollout += mean_pred
    
    def predict(self, test_data, pred_span, ci=0.9):
        """
        Predict the data for each time span until it covers al the testing time step, 
            for pred_span = 1, we retrain (using testing resutk) the model every step 
                (cost a lot of computations)
            for pred_span = length of test step, we didn't use any of testing result. 

        Args:
            x_pred: Data for performing a prediction
            y_pred: Correct Log-Price of prediction (if None the pred_span is length of test_step)
            pred_span: Number of prediction step before retrain the data with correct testing data.
            ci: confidence interval, in terms of percentage.
        
        Return:
            prediction: Tuple contains means, upper and lower bound of the prediction. 
        """
        self.initialize()
        self.all_data = test_data.label_inp["Price"].to_list()
        span_per_round = self.hyperparam["ind_span_pred"]

        num_iter = math.floor(pred_span/span_per_round)
        num_left = pred_span - span_per_round*num_iter

        y_pred = test_data.label_out["Price"].to_list()

        for i in range(num_iter):
            self.predict_time_step(span_per_round, ci)
            self.all_data += y_pred[i*span_per_round:(i+1)*span_per_round]

        if num_left > 0:
            self.predict_time_step(num_left, ci)
        
        return pack_data(self.pred_rollout, self.upper_rollout, self.lower_rollout, test_data.data_out["Date"].to_list())
