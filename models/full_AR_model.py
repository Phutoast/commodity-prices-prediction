import math
import numpy as np

from models.base_model import BaseModel

class FullARModel(BaseModel):
    """
    Full Auto-Regressive Model: A model that can output any size. However, len_out is what the model is number of trained/expect output to be. 

    Args:
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
    
    def get_all_data(self, train_data):
        """
        Getting all the train data and transform into the self.all_data 
        that contains a matrix of features N x d where d is the 

        Args:
            train_data: Training dataset in our defined dataset format.
        
        Return:
            all_data: Collection of data in numpy array
        """
        packed_data = self.pack_data(train_data)
        return packed_data[:, [self.hyperparam["len_out"]]]
    
    def get_batch_test_data(self, test_data):
        """
        Given a list of test data transform it to a list of matrix of features 
        that are ready to be appended into the all_data to perform the next prediction

        Args:
            test_data: Testing dataset in our defined dataset format. 
        
        Returns:
            list_packed_data: a list with (possible the same size as num_iter) containing len_out x d.
        """
        packed_data = self.pack_data(test_data)
        out = packed_data[:, self.hyperparam["len_out"]:]
        return list(np.expand_dims(out, -1))
    
    def train(self):
        """
        Symbolically create the all_data as to define the training process
        """
        self.all_data = self.get_all_data(self.train_data)
    
    def append_all_data(self, new_data):
        """
        Append the data to self.all_data
        """
        self.all_data = np.concatenate([self.all_data, new_data]) 
    
    
    def pred_fix_step(self, step_ahead, ci):
        """
        Predicting a Fixed Step given any specified step_ahead (Used only when performing actual prediction)

        In Autoregressive-RNN we can chooses to augment the the all_data with the real one our the output of that RNN (its up to us)

        Args:
            step_ahead: Number of data points we want to predict ahead of time
            ci: Confidence Interval of the prediction
        
        Returns:
            mean: Mean prediction of the model.
            upper: Upperbound of the prediction of the model.
            lower: Lowerbound of the prediction of the model.
        """
    
    def add_results(self, mean, upper, lower):
        """
        Collecting results from the prediction from pred_fix_step.

        Args:
            mean: Mean prediction of the model.
            upper: Upperbound of the prediction of the model.
            lower: Lowerbound of the prediction of the model.
        """
        self.pred_rollout += mean
        self.upper_rollout += upper
        self.lower_rollout += lower

    def predict_step_ahead(self, test_data, step_ahead, ci=0.9):
        self.initialize()
        span_per_round = self.hyperparam["len_out"]
        assert span_per_round <= step_ahead
        
        data = self.get_batch_test_data(test_data)

        num_iter = math.floor(step_ahead/span_per_round)
        num_left = step_ahead - span_per_round*num_iter

        for i in range(num_iter):
            print("Predicting...", i, "/", num_iter)
            mean, upper, lower = self.predict_fix_step(span_per_round, ci)
            self.add_results(mean, upper, lower)

            # Constantly adding the dta
            self.append_all_data(data[i])
            
        if num_left > 0:
            print("Predicting num rest", num_left)
            mean, upper, lower = self.predict_fix_step(num_left, ci)
            self.add_results(mean, upper, lower)
        
        return self.pred_rollout, self.upper_rollout, self.lower_rollout