from collections import namedtuple
import pandas as pd
import warnings

# Storing Prediction Results for Plotting
DisplayPrediction = namedtuple("DisplayPrediction", ["packed_data", "name", "color", "is_bridge"], defaults=(None, "k", True))

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

def pack_data(mean, upper, lower, x):
    """
    Given the numpy/list data, pack the result into panda dataframe
        Ready for display/save 
    
    Args:
        mean: Mean prediction of the predictor
        upper: Upper Confidence Bound
        lower: Lower Confidence Bound
        x: x-axis that is used to display (should contains the data)
    
    Return:
        packed_data: Data ready to display
    """
    if len(upper) == 0 and len(lower) == 0:
        upper = mean
        lower = mean
    d = {"mean": mean, "upper": upper, "lower": lower, "x": x}
    return pd.DataFrame(data=d)

