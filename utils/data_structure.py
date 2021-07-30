from collections import namedtuple
import pandas as pd
import torch
from utils import data_preprocessing

import warnings

# Storing Prediction Results for Plotting
DisplayPrediction = namedtuple("DisplayPrediction", ["packed_data", "name", "color", "is_bridge"], defaults=(None, "k", True))
FoldWalkForewardResult = namedtuple("FoldWalkForewardResult", ["pred", "missing_data", "model", "loss_detail"])

CompressMethod = namedtuple(
    "CompressMethod", ["compress_dim", "method", "info"], defaults=(None, None, {})
)

full_features_names = {
    "aluminium": ['Date', 'COT', 'COT.1', 'CTA', 'CTA.1', 'CURVE', 'CURVE.1', 'CURVE.2', 'CURRENCY', 'CURRENCY.1', 'CURRENCY.2', 'CURRENCY.3', 'CURRENCY.4', 'CURRENCY.5', 'CURRENCY.6', 'CURRENCY.7', 'CURRENCY.8', 'CURRENCY.9', 'FREIGHT', 'FREIGHT.1', 'FREIGHT.2', 'FREIGHT.3', 'FREIGHT.4', 'FREIGHT.5', 'FREIGHT.6', 'FREIGHT.7', 'FREIGHT.8', 'FREIGHT.9', 'INVENTORIES', 'INVENTORIES.1', 'INVENTORIES.2', 'INVENTORIES.3', 'INVENTORIES.4', 'INVENTORIES.5', 'SATELLITE', 'SATELLITE.1', 'SATELLITE.2', 'SATELLITE.3', 'SATELLITE.4', 'SATELLITE.5', 'SATELLITE.6', 'SATELLITE.7', 'SATELLITE.8', 'SATELLITE.9', 'SATELLITE.10', 'SEASONALITY', 'MACRO', 'MACRO.1', 'MACRO.2', 'MACRO.3', 'MACRO.4', 'MACRO.5', 'MACRO.6', 'MACRO.7', 'MACRO.8', 'TECHNICAL', 'Price'], 
    "copper": ['Date', 'COT', 'COT.1', 'COT.2', 'CTA', 'CTA.1', 'CURVE', 'CURVE.1', 'CURVE.2', 'CURRENCY', 'CURRENCY.1', 'CURRENCY.2', 'CURRENCY.3', 'CURRENCY.4', 'CURRENCY.5', 'FREIGHT', 'FREIGHT.1', 'FREIGHT.2', 'FREIGHT.3', 'FREIGHT.4', 'FREIGHT.5', 'FREIGHT.6', 'FREIGHT.7', 'FREIGHT.8', 'FREIGHT.9', 'INVENTORIES', 'INVENTORIES.1', 'INVENTORIES.2', 'INVENTORIES.3', 'INVENTORIES.4', 'INVENTORIES.5', 'SATELLITE', 'SATELLITE.1', 'SATELLITE.2', 'SEASONALITY', 'MACRO', 'MACRO.1', 'MACRO.2', 'MACRO.3', 'MACRO.4', 'MACRO.5', 'MACRO.6', 'MACRO.7', 'MACRO.8', 'TECHNICAL', 'Price']
}

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
    def __init__(self, len_inp, len_out, is_date, 
        is_verbose=True, is_gpu=False, sample_size=1000, **kwargs): 

        self["len_inp"] = len_inp
        self["len_out"] = len_out
        self["is_date"] = is_date
        self["is_verbose"] = is_verbose
        self["is_gpu"] = is_gpu
        self["sample_size"] = sample_size

        if not torch.cuda.is_available() and is_gpu:
            warnings.warn(UserWarning("No CUDA available all used for dogecoin mining, fallback to use CPU."))
            self["is_gpu"] = False

        self.update(kwargs)
    
    def __repr__(self):
        return f"{type(self).__name__}({super().__repr__()})"

class DatasetTaskDesc(dict):
    """
    Way to describe the construction of the multi-task dataset
    
    Args:
        inp_metal_list: The metal data we are going to use to perform the prediction.
        feature: The feature (from the inp_metal) we are going to used to perform the prediction.
        out_feature: The feature we want to predict
        kwargs: other model specific hyperparameters (e.g hidden-layer size). 
    """
    def __init__(self, inp_metal_list,
        use_feature, use_feat_tran_lag, out_feature, 
        out_feat_tran_lag, is_drop_nan=False, len_dataset=-1, 
        metal_modifier=None, name=None, **kwargs): 

        self["inp_metal_list"] = inp_metal_list
        self["out_feature"] = out_feature
        self["out_feat_tran_lag"] = out_feat_tran_lag
        self["is_drop_nan"] = is_drop_nan
        self["len_dataset"] = len_dataset
        self["metal_modifier"] = metal_modifier

        if all("Date" not in col_name for col_name in use_feature):
            raise ValueError("Date has to be included in use_feature (but can be removed later)")
        
        if out_feature in use_feature:
            warnings.warn(UserWarning("Duplication between the output column and Feature"))
        
        if isinstance(use_feat_tran_lag, list):
            if len(use_feat_tran_lag) != len(use_feature):
                raise ValueError("If defining use_feat_tran_lag to be a list, the length of it should be the same as use_feature")
        elif use_feat_tran_lag is None:
            use_feat_tran_lag = [None] * len(use_feature)
        else:
            raise TypeError("Wrong Type For use_feat_tran_lag")
        
        if metal_modifier is None:
            self["metal_modifier"] = [
                (0, "id")
                for _ in range(len(inp_metal_list))
            ]
        elif isinstance(metal_modifier, list):
            if len(metal_modifier) != len(inp_metal_list):
                raise ValueError("The modifier for the metal list should be the same length is the metal_list.")
            self["metal_modifier"] = [
                (0, "id") if modi is None else modi
                for modi in metal_modifier
            ]
        else:
            raise TypeError("Wrong Type For metal_modifier")
        
        self.name = name 
        
        self["use_feature"] = use_feature
        self["use_feat_tran_lag"] = use_feat_tran_lag
        self.update(kwargs)
    
    def __repr__(self):
        return f"{type(self).__name__}({super().__repr__()})"
    
    def gen_name(self):
        if self.name is None:
            metal_names = " ".join(map(lambda x: x.capitalize(), self["inp_metal_list"]))
            if all(modi is None for modi in self["metal_modifier"]):
                generated_name = metal_names
            else:
                modifier = "*".join(map(lambda x: f"{x[1].upper()}({x[0]})", self["metal_modifier"]))
                generated_name = metal_names + " + " + modifier
            
            return generated_name
        
        return self.name
