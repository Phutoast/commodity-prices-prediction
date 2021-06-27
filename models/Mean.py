import numpy as np
from models.base_model import BaseModel

class MeanModel(BaseModel):
    """
    Mean Baseline Model. 
    Simply calculate the mean of the training set data and 
    """
    def __init__(self, train_data, model_hyperparam):
        super().__init__(train_data, model_hyperparam)
    
    def train(self):
        all_prices = self.pack_data(self.train_data)
        print(all_prices)
        self.param = np.mean(all_prices[:, 1])
    
    def predict_step_head(self, test_data, step_ahead, ci=0.9):
        """
        Args: (See superclass)
        Returns: (See superclass)
        """
        
        return np.ones(step_ahead) * self.param, [], [], test_data.data_out["Date"].to_list()
    
