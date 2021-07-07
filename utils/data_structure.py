from collections import namedtuple
import pandas as pd
import warnings

# Storing Prediction Results for Plotting
DisplayPrediction = namedtuple("DisplayPrediction", ["packed_data", "name", "color", "is_bridge"], defaults=(None, "k", True))

FoldWalkForewardResult = namedtuple("FoldWalkForewardResult", ["pred", "missing_data", "interval_loss"])

# Storing Trainig Data
# We are interested in the model where
# ([data_inp], [label_inp]) -> Model (at state ready to do prediction)
# Model + [data_out] -> [pred_label]
# Loss=l([pred_label], [label_out])
# In normal regressor, the inp would be: Model([data_inp], [label_inp], [data_out]) -> [pred_label]
class TrainingPoint(namedtuple("TrainingSet", ["data_inp", "label_inp", "data_out", "label_out"])):
    def __eq__(self, other):
        cond1 = self.data_inp.equals(other.data_inp)
        cond2 = self.label_inp.equals(other.label_inp)
        cond3 = self.data_out.equals(other.data_out)
        cond4 = self.label_out.equals(other.label_out)

        return all([cond1, cond2, cond3, cond4])

# Storing Useful Hyperparameters
class Hyperparameters(dict):
    """
    Way to collect the hyperparameter of the model. 
        It collects the default field that are common to all models 
        and additional fields
    
    Args:
        len_inp: length of input (in terms of time step)
        len_out: length of output (in terms of time step)
        is_date: Include date into the training ?
        kwargs: other model specific hyperparameters (e.g hidden-layer size). 
    """
    def __init__(self, len_inp, len_out, is_date, **kwargs): 
        self["len_inp"] = len_inp
        self["len_out"] = len_out
        self["is_date"] = is_date
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

# class FeatureDataset(object):
#     """
#     Transforming TrainingPoint object 
#         into a list corresponding to user-specified label. 

#     Args:
#         data_feat: List of feature name that 
#             we want to include in the input of the model.
#         label_feat_data: List of feature name that 
#             we want to include in the label of the input of the model. 
#         label_feat_pred: List of feature name that 
#             we want to include as the predion of the model. 
#     """
#     def __init__(self, data_feat=None, label_feat=None):

#         self.data_feat = data_feat
#         self.label_feat = label_feat
    
#     def __call__(self, data_point):
#         """
#         Transform the data_point into a list based 
#             on the label passed in __init__ so that 
#             we can add them to numpy array. 
#             The data will be returned in the following format. 

#             [data_feat] + [label_feat] + [data_feat] + []
        
#         Args:
#             data_point: Data from dataset stored in TrainingPoint format.
        
#         Return:
#             data_list: Computed data format that is ready 
#                 to be stored in number array
#             total_length: Total length of the data_list
#         """