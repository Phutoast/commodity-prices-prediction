
class BaseModel(object):
    """
    Abstraction of training model. 

    Args:
        train_data: Data from the training set
        kwargs: model_hyperparam: model hyperparameter

    """
    def __init__(self, train_data, model_hyperparam):
        self.train_data = train_data
        self.model_hyperparam = model_hyperparam
    

    def train(self):
        """
        Training the data given the training parameters.
        """
        pass
    
    def predict(self, test_data, step_ahead=-1):
        """
        Predict the data given data given the current model

        _Warning_ y_pred has to be used carefully, 
            so that there is no leak in the dataset
        
        Args:
            test_data: Testing data given for testing
            step_ahead: Number of step a ahead we wany ot compute 
                if -1 then we use the same value as len_out in Hyperparameter
        
        Returns:
            pred: prediction of length step_ahead
        """
        pass
