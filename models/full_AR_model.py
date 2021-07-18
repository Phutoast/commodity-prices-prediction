import math
import numpy as np

from models.base_model import BaseModel

class FullARModel(BaseModel):
    """
    Full Auto-Regressive Model: A model that can output any size. However, len_out doesn't affect the training.

    Args:
        pred_rollout: Prediction rollout of ARIMA model
        upper_rollout: Upperbound on the rollout
        lower_rollout: Lowerbound on the rollout
        all_data: All data collected during training (and added during testing for additional input)
    """

    def __init__(self, train_data, model_hyperparam):
        super().__init__(train_data, model_hyperparam)
        self.initialize()
        if self.hyperparam["is_full_pred"]:
            assert self.hyperparam["len_inp"] + self.hyperparam["len_out"] == 1
    
    def initialize(self):
        """
        Reset all the data for a new prediction.
        """
        self.pred_rollout, self.upper_rollout, self.lower_rollout = [], [], []
        self.sample = []
    
    def get_all_data(self, train_data, include_miss=True):
        """
        Getting all the train data and transform into the self.all_data 
        that contains a matrix of features N x d where d is the feature dimension.

        Args:
            train_data: Training dataset in our defined dataset format.
        
        Return:
            all_data: Collection of data in numpy array
        """
        packed_data = self.pack_data(train_data, is_full_AR=True)

        # Getting all data, get the first training point, 
        # Get the last one that is the target

        total_num_data = self.hyperparam["len_inp"] + self.hyperparam["len_out"]
        assert packed_data.shape[1] == total_num_data

        all_data = packed_data[:, 0, :]

        # The missing data (occur because padding)
        if include_miss:
            all_data = np.concatenate(
                [all_data, packed_data[-1, range(1, total_num_data), :]]
            )

        if not self.hyperparam["is_date"]:
            all_data = all_data[:, 1:]

        return all_data 
    
    def train(self):
        """
        Symbolically create the all_data as to define the training process
        """
        # Only getting the value this is for Mean and ARIMA only

        self.all_data = self.get_all_data(
            self.train_data, include_miss=True
        ) 
        self.model = self.build_model()
    
    def get_batch_test_data(self, test_data):
        """
        Given a list of test data transform it to a list of matrix of features 
        that are ready to be appended into the all_data to perform the next prediction

        Args:
            test_data: Testing dataset in our defined dataset format. 
        
        Returns:
            list_packed_data: a list with (possible the same size as num_iter) containing len_out x d.
        """
        packed_data = self.pack_data(test_data, is_full_AR=True)
        
        if not self.hyperparam["is_date"]:
            packed_data = packed_data[:, :, 1:]

        return list(packed_data)
    
    def append_all_data(self, new_data):
        """
        Append the data to self.all_data
        """
        self.all_data = np.concatenate([self.all_data, new_data]) 
    
    
    def predict_fix_step(self, step_ahead, ci, is_sample):
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
        raise NotImplementedError()
    
    def add_sample_result(self, samples):
        print(samples.shape)
        self.sample.append(samples)
    
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
    
    def pred_and_add(self, step_ahead, ci, is_sample):
        if is_sample:
            samples = self.predict_fix_step(step_ahead, ci, is_sample)
            self.add_sample_result(samples)
        else:
            mean, upper, lower = self.predict_fix_step(step_ahead, ci, is_sample)
            self.add_results(mean, upper, lower)
            

    def predict_step_ahead(self, test_data, step_ahead, all_date, ci=0.9, is_sample=False):
        self.initialize()
        span_per_round = self.hyperparam["len_out"]
        assert span_per_round <= step_ahead

        if self.hyperparam["is_full_pred"]:
            self.pred_and_add(step_ahead, ci, is_sample)
            pred_len = step_ahead
        else:
            data = self.get_batch_test_data(test_data)

            num_iter = math.floor(step_ahead/span_per_round)
            num_left = step_ahead - span_per_round*num_iter

            for i in range(num_iter):
                if self.hyperparam["is_verbose"]:
                    print("Predicting...", i, "/", num_iter)
                
                self.pred_and_add(span_per_round, ci, is_sample)

                # Constantly adding the data 
                self.append_all_data(data[i])

            if num_left > 0:
                if self.hyperparam["is_verbose"]:
                    print("Predicting num rest", num_left)
                self.pred_and_add(num_left, ci, is_sample)
            pred_len = num_iter*span_per_round
         
        len_date = len(all_date)

        if not is_sample:
            return self.pred_rollout, self.upper_rollout, self.lower_rollout, all_date[len_date-pred_len:]
        else:
            return np.concatenate(self.sample, axis=1), all_date[len_date-pred_len:]