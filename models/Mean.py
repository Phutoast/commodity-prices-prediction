import numpy as np
import pickle
import scipy.stats as sps
from collections import namedtuple

from models.full_AR_model import FullARModel

import matplotlib.pyplot as plt
import pandas as pd
from utils import data_visualization
from scipy.stats import norm

GaussianParam = namedtuple("GaussianParam", ["mean", "std"])

class IIDDataModel(FullARModel):
    """
    Mean Baseline Model. 
    Simply calculate the moment matching
    """
    def __init__(self, train_data, model_hyperparam):
        super().__init__(train_data, model_hyperparam)
     
    def predict_fix_step(self, step_ahead, ci=0.9, is_sample=False):
        if self.hyperparam["dist"].lower() == "gaussian":
            self.model = self.build_model()
            mean, std = self.model
            dist = sps.norm(loc=mean, scale=std)
            
            left = (1 - ci)/2
            right = 1 - (1 - ci)/2

            total_mean = np.ones(step_ahead) * mean.mean()

            if not is_sample:
                upper = np.ones(step_ahead) * dist.ppf(right)
                lower = np.ones(step_ahead) * dist.ppf(left)
                return total_mean.tolist(), upper.tolist(), lower.tolist()
            else:
                upper = np.ones(step_ahead) * dist.ppf(right)
                lower = np.ones(step_ahead) * dist.ppf(left)
            
                rv = np.reshape(dist.rvs(size=self.hyperparam["sample_size"]), (-1, 1)) 
                rv = np.repeat(rv, step_ahead, axis=1)
                return rv
        else:
            assert False, "No Distribution Avaliable"
    
    def build_model(self):
        mean = np.mean(self.all_data)
        std = np.std(self.all_data)
        return GaussianParam(mean, std)

    def save(self, path):
        with open(path + ".pkl", "wb") as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        np.save(path + "_all_data.npy", self.all_data)
    
    def load(self, path):
        with open(path + ".pkl", "rb") as handle:
            self.model = pickle.load(handle)
        
        self.all_data = np.load(path + "_all_data.npy")
    