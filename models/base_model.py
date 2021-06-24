import numpy as np
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
    
    def collect_all_prices(self):
        """
        Given the training data, gather all prices together with the time index.

        Return:
            data_collection: (num_data x 2) the full collection 
                of the dataset together with at tine (for now)
        """
        # It is clear that the last index of the dataset is the last index of the training set
        # Also did a testing for any weird behavior, it is consistence with training loop 
        #      as we have to go through all the training data anyways
        # Abit redundance but it works 
        # last_index = self.train_data[-1].label_out.index[-1]
        # first_index = self.train_data[0].label_inp.index[0]    

        collection = {}

        def add_data(df_data):
            for index, data in zip(df_data.index, df_data["Price"]):
                if index in collection:
                    assert collection[index] == data
                else:
                    collection[index] = data

        for data in self.train_data:
            _, train_inp, _, train_out = data
            add_data(train_inp)
            add_data(train_out)
        
        num_data = len(collection)
        data = np.zeros((num_data, 2))
        for i, (day, d) in enumerate(collection.items()):
            data[i, :] = [day, d]
        return data
    
    def predict_step_head(self, test_data, step_ahead, ci=0.9):
        """
        Wrapping by predict method

        Args:
            test_data: Testing data given for testing
            step_ahead: Number of step a ahead we wany ot compute 
            ci: Confidence Interval set in ratio.

        Returns: 
            Tuple of list of mean, upperbound and lower bound, which will be used for paching data
        """
     
    def predict(self, test_data, step_ahead=-1, ci=0.9):
        """
        Predict the data for each time span until it covers al the testing time step 
            (if auto-regressive, if not we resort to the use of )

        _Warning_ y_pred has to be used carefully, 
            so that there is no leak in the dataset
        
        Args:
            test_data: Testing data given for testing
            step_ahead: Number of step a ahead we wany ot compute 
                if -1 then we use the same value as len_out in Hyperparameter
            ci: Confidence Interval set in ratio.
        
        Returns:
            prediction: Dataframe that contains means, upper, 
                lower bound of the prediction and the data necessary to the plotting
        """
        step_ahead = self.hyperparam["len_out"] if step_ahead == -1 else step_ahead
        pred_rollout, upper_rollout, lower_rollout, dates = self.predict_step_head(test_data, step_ahead, ci=0.9)
        return pack_data(pred_rollout, upper_rollout, lower_rollout, dates)
