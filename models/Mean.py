import warnings
import math

from models.base_model import BaseModel
from utils.data_structure import pack_data

class MeanModel(BaseModel):
    """
    Mean Baseline Model. 
    Simply calculate the mean of the training set data and 
    """
    def __init__(self, train_data, model_hyperparam):
        super().__init__(train_data, model_hyperparam)
    
    def train(self):
        """
        No formal training in ARIMA model as we simply fit the data, 
        while when performing a prediction we will have to refit some of the data
        """
        print("train data")
        print(self.train_data)
        assert False