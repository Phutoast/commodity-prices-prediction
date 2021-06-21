from collections import namedtuple
import warnings

# Storing Prediction Results for Plotting
Prediction = namedtuple("Prediction", ["y_pred", "y_up", "y_low"])

# Storing Trainig Data
TrainingPoint = namedtuple("TrainingSet", ["data_inp", "label_inp", "data_out", "label_out"])

# Storing Useful Hyperparameters
class Hyperparameters(dict):
    """
    Way to collect the hyperparameter of the model. 
        It collects the default field that are common to all models 
        and additional fields
    
    Args:
        len_inp: length of input (in terms of time step), -1 if auto-regressive
        len_out: length of output (in terms of time step), -1 if auto-regressive
        kwargs: other model specific hyperparameters (e.g hidden-layer size). 
    """
    def __init__(self, len_inp, len_out, **kwargs): 
        self["len_inp"] = len_inp
        self["len_out"] = len_out
        self.update(kwargs)
    
    def __repr__(self):
        return f"{type(self).__name__}({super().__repr__()})"


