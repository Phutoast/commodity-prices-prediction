import warnings
import math
from statsmodels.tsa.arima.model import ARIMA

from models.base_model import BaseModel

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
        assert self.hyperparam["len_inp"] == 0
    
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

        all_prices = self.pack_data(self.train_data)
        # We will give it only to be date and prediction.
        self.all_data = all_prices[:, self.hyperparam["len_out"]].tolist()
        self.curr_train = self.all_data
    
    
    def predict_time_step(self, step_ahead, ci):
        """
        Predict the ARIMA mdoel given the number of steps ahead, 
            and update the interal value

        Args:
            step_ahead: Number of data points we want to predict ahead of time
            ci: Confidence Interval of the prediction
        """

        order = self.hyperparam["order"]
        model = ARIMA(self.curr_train, order=order).fit()
        result = model.get_forecast(step_ahead)
        upper_pred, lower_pred = result.conf_int(alpha=1-ci).T 
        mean_pred = result.summary_frame()["mean"].to_list()

        self.upper_rollout += list(upper_pred)
        self.lower_rollout += list(lower_pred)
        self.pred_rollout += mean_pred
    
    def predict_step_head(self, test_data, step_ahead, ci=0.9):
        """
        Predict the data for each time span until it covers al the testing time step, 
            for ind_span_pred = 1, we retrain (using testing result) the model every step 
                (cost a lot of computations)
            for ind_span_pred = length of test step, we didn't use any of testing result. 
        
        Args: (See superclass)
        Returns: (See superclass)
        """
        self.initialize()
        # Note that ARIMA is clueless about a time step, so we can't do anything.
        # Have to use old data because ARIMA doesn't have a sense of time :(
        self.curr_train += test_data.label_inp["Price"].to_list()
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
            print("HERE`")
            self.predict_time_step(num_left, ci)
        
        return self.pred_rollout, self.upper_rollout, self.lower_rollout, test_data.data_out["Date"].to_list()
