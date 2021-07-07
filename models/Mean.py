import numpy as np
import scipy.stats as sps

from models.full_AR_model import FullARModel

class IIDDataModel(FullARModel):
    """
    Mean Baseline Model. 
    Simply calculate the moment matching
    """
    def __init__(self, train_data, model_hyperparam):
        super().__init__(train_data, model_hyperparam)
     
    def predict_fix_step(self, step_ahead, ci=0.9):
        if self.hyperparam["dist"] == "Gaussian":
            mean = np.mean(self.all_data)
            std = np.std(self.all_data)
            dist = sps.norm(loc=mean, scale=std)
            
            left = (1 - ci)/2
            right = 1 - (1 - ci)/2

            total_mean = np.ones(step_ahead) * mean
            upper = np.ones(step_ahead) * dist.ppf(right)
            lower = np.ones(step_ahead) * dist.ppf(left)
            return total_mean.tolist(), upper.tolist(), lower.tolist()
        else:
            assert False, "No Distribution Avaliable"

        assert False
    