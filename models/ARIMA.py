import warnings
import math
from statsmodels.tsa.arima.model import ARIMA

from models.base_model import BaseModel
from utils.data_preprocessing import data_to_date_label
from utils.data_structure import Prediction

class ARIMAModel(BaseModel):
    """
    Auto Regressive Integrated Moving Average Model wrapper. 

    Args:
        x_train: Data from the training set
        y_train: Log-Price of training set
        order: The order of the ARIMA model
        pred_rollout: Prediction rollout of ARIMA model
        upper_rollout: Upperbound on the rollout
        lower_rollout: Lowerbound on the rollout
        all_data: All y_train data

    """
    def __init__(self, x_train, y_train, order):

        # Although repeated, we want to keep the same interface
        x_train, _ = data_to_date_label(x_train)
        super().__init__(x_train, y_train)
        self.order = order
        self.initialize()
    
    def initialize(self):
        """
        Reset all the data for a new prediction.
        """
        self.pred_rollout, self.upper_rollout, self.lower_rollout = [], [], []
        self.all_data = self.y_train.to_list()
    
    def train(self):
        """
        No formal training in ARIMA model as we simply fit the data, 
        while when performing a prediction we will have to refit some of the data
        """
        warnings.warn("There is no formal training in ARIMA model")
    
    def predict_time_step(self, pred_span):
        """
        Predict the ARIMA mdoel given the number of steps ahead, 
            and update the interal value

        Args:
            pred_span: Number of data points we want to predict ahead of time
        """
        model = ARIMA(self.all_data, order=self.order).fit()
        result = model.get_forecast(pred_span)
        upper_pred, lower_pred = result.conf_int(alpha=0.1).T 
        mean_pred = result.summary_frame()["mean"].to_list()

        self.upper_rollout += list(upper_pred)
        self.lower_rollout += list(lower_pred)
        self.pred_rollout += mean_pred
    
    def predict(self, x_pred, y_pred, pred_span, ci=90):
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
        x_pred, _ = data_to_date_label(x_pred)
        length_data = len(x_pred)

        assert len(y_pred) == length_data or y_pred is None

        if y_pred is None:
            pred_span = len(x_pred)
        else:
            assert length_data >= pred_span
            y_pred = y_pred.to_list()
        
        num_iter = math.floor(length_data/pred_span)
        num_left = length_data - pred_span*num_iter

        for i in range(num_iter):
            self.predict_time_step(pred_span)

            if y_pred is not None:
                self.all_data += y_pred[i*pred_span:(i+1)*pred_span]

        if num_left > 0:
            self.predict_time_step(num_left)
        
        return Prediction(self.pred_rollout, self.upper_rollout, self.lower_rollout)
