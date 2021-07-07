import numpy as np
import pandas as pd

from utils.data_structure import pack_data

class BaseModel(object):
    """
    Abstraction of training model. 

    Args:
        train_data: Data from the training set
        kwargs: model_hyperparam: model hyperparameter

    """
    def __init__(self, train_data, model_hyperparam):
        self.train_data = train_data
        self.hyperparam = model_hyperparam
    

    def train(self):
        """
        Training the data given the training parameters.
        """
        pass
     
    def pack_data(self, dataset, is_full_AR=False):
        """
        Given the training data, pack all the data 
            into the single numpy array where the last column is the target

        WARNING: It won't do any jobs on the transformation of data 
            (including selecting relevent feature) 
        
        Args:
            dataset: Training dataset.  
        
        Return:
            full_numpy: The full training data in numpy format.
        """

        def sort_data_points(data, label, len_data):
            assert len(data)%len_data == 0
            feature_num = len(data) // len_data

            list_data = []
            for i in range(len_data):
                point_data = np.concatenate((
                    data[i*feature_num:(i+1)*feature_num], 
                    [label[i]]
                ))
                list_data.append(np.expand_dims(point_data, axis=0))
            
            return np.concatenate(list_data)

        def datapoint_to_list(data_point):
            """
            For each data point in the data set, we turn 
                them into row of the full numpy array

            Args:
                data_point: Element of the dataset
            
            Return:
                row_data: Row of the full_numpy array
            """
            all_float = lambda df, order: df.apply(pd.to_numeric, errors='coerce').to_numpy().flatten(order)
            first = all_float(data_point.data_inp, "C")
            second = all_float(data_point.label_inp, "C")
            third = all_float(data_point.data_out, "C")
            forth = all_float(data_point.label_out, "F")
            if not is_full_AR:
                return np.concatenate([first, second, third, forth])
            else:
                full_data = []
                if len(first) != 0 or len(second) != 0:
                    data = sort_data_points(
                        first, second, self.hyperparam["len_inp"]
                    )
                    full_data.append(data)

                data = sort_data_points(
                    third, forth, self.hyperparam["len_out"]
                )
                full_data.append(data)
                return np.expand_dims(np.vstack(full_data), axis=0) 

        num_data = len(dataset)
        num_feature = len(datapoint_to_list(dataset[0]))

        all_prices = []
        for data in dataset:
            row_data = datapoint_to_list(data)
            all_prices.append(row_data)
        
        out_data = np.vstack(all_prices)
        return out_data
    
    def predict_step_ahead(self, test_data, step_ahead, ci=0.9):
        """
        Wrapping by predict method

        Args:
            test_data: Testing data given for testing
            step_ahead: Number of step a ahead we wany ot compute 
            ci: Confidence Interval set in ratio.

        Returns: 
            Tuple of list of mean, upperbound and lower bound, which will be used for paching data
        """
     
    def predict(self, test_data, step_ahead, all_date, ci=0.9):
        """
        Predict the data for each time span until it covers al the testing time step 
            (if auto-regressive, if not we resort to the use of )

        _Warning_ y_pred has to be used carefully, 
            so that there is no leak in the dataset
        
        Args:
            test_data: Testing data set given for testing (must be dataframe)
            step_ahead: Number of step a ahead we wany ot compute 
                if -1 then we use the same value as len_out in Hyperparameter
            ci: Confidence Interval set in ratio.
        
        Returns:
            prediction: Dataframe that contains means, upper, 
                lower bound of the prediction and the data necessary to the plotting
        """
        step_ahead = self.hyperparam["len_out"] if step_ahead == -1 else step_ahead
        pred_rollout, upper_rollout, lower_rollout = self.predict_step_ahead(test_data, step_ahead, ci=0.9)
        return pack_data(pred_rollout, upper_rollout, lower_rollout, all_date)
