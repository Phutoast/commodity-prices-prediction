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
        all_data: All data collected during training (and added during testing for additional input)
    """
    def __init__(self, train_data, model_hyperparam):
        super().__init__(train_data, model_hyperparam)
        self.initialize()
    
    def initialize(self):
        """
        Reset all the data for a new prediction.
        """
        self.pred_rollout, self.upper_rollout, self.lower_rollout = [], [], []
    
    def train(self):
        """
        The training part for ARIMA is to construct appropriate train set 
            by augmenting all the data points into 1 data-list 
            (although we assume the offset to be -1, we make sure that everything works)
        """

        all_prices = self.collect_all_prices()
        self.all_data = all_prices[:, 1].tolist()
    
    
    def predict_time_step(self, step_ahead, ci):
        """
        Predict the ARIMA mdoel given the number of steps ahead, 
            and update the interal value

        Args:
            step_ahead: Number of data points we want to predict ahead of time
        """

        order = self.hyperparam["order"]
        model = ARIMA(self.curr_train, order=order).fit()
        result = model.get_forecast(step_ahead)
        upper_pred, lower_pred = result.conf_int(alpha=1-ci).T 
        mean_pred = result.summary_frame()["mean"].to_list()

        self.upper_rollout += list(upper_pred)
        self.lower_rollout += list(lower_pred)
        self.pred_rollout += mean_pred
    
    def predict(self, test_data, step_ahead, ci=0.9):
        """
        Predict the data for each time span until it covers al the testing time step, 
            for step_ahead = 1, we retrain (using testing resutk) the model every step 
                (cost a lot of computations)
            for step_ahead = length of test step, we didn't use any of testing result. 

        Args:
            x_pred: Data for performing a prediction
            y_pred: Correct Log-Price of prediction (if None the step_ahead is length of test_step)
            step_ahead: Number of prediction step before retrain the data with correct testing data.
            ci: confidence interval, in terms of percentage.
        
        Return:
            prediction: Tuple contains means, upper and lower bound of the prediction. 
        """
        self.initialize()
        # Note that ARIMA is clueless about a time step, so we can't do anything.
        self.curr_train = self.all_data + test_data.label_inp["Price"].to_list()
        span_per_round = self.hyperparam["ind_span_pred"]

        assert span_per_round <= step_ahead

        num_iter = math.floor(step_ahead/span_per_round)
        num_left = step_ahead - span_per_round*num_iter

        y_pred = test_data.label_out["Price"].to_list()

        # Cheating !!!
        for i in range(num_iter):
            print("Predicting...", i, "/", num_iter)
            self.predict_time_step(span_per_round, ci)
            self.curr_train += y_pred[i*span_per_round:(i+1)*span_per_round]

        if num_left > 0:
            self.predict_time_step(num_left, ci)
        
        return pack_data(self.pred_rollout, self.upper_rollout, self.lower_rollout, test_data.data_out["Date"].to_list())
