import numpy as np
import pandas as pd

from utils.others import create_name, create_folder
from models.base_model import BaseModel

class IndependentMultiModel(object):
    """
    Wrapper for multiple independent models

    Args:
        list_train_data: Data from the training set
            if it is not a list then we apply the same train_data to all models
        list_config: List of each model 
            and its hyperparameter (as seen in algo_dict), 
            if it is not a list then we apply the same hyperparameter to all models
        num_model: Number of models that we want to output.

    """
    def __init__(self, list_train_data, list_config, num_model):
        self.num_model = num_model
        assert all(
            len(a) == num_model 
            for a in [list_train_data, list_config]
        )
        self.models = [
            list_config[i][1](list_train_data[i], list_config[i][0])
            for i in range(num_model)
        ] 

    def train(self):
        """
        Training the data given the training parameters.
        """
        for i, m in enumerate(self.models):
            print(f"Training Model: {i}/{len(self.models)}")
            m.train()

    def predict(self, list_test_data, list_step_ahead, list_all_date, ci=0.9):
        """
        Predict multiple independent multi-model data

        Args:
            list_test_data: All testing for each models
            list_step_ahead: All number step a head 
                for each model
            list_all_date: All the date used along side of prediction
        
        Returns:
            list_prediction: List of all prediction for each models
        """
        assert all(
            len(a) == self.num_model 
            for a in [list_test_data, list_step_ahead, list_all_date]
        )

        return [
            self.models[i].predict(
                list_test_data[i], list_step_ahead[i], list_all_date[i], ci=ci
            )
            for i in range(self.num_model)
        ]
    
    def save(self, model_name):
        """
        Save the model to the path given

        Args:
            base_model: The name of the whole multimodel
        """
        base_folder = create_name("save/", model_name)
        create_folder(base_folder)

        for i in range(self.num_model):
            model_path = base_folder + f"model_{i}/"
            create_folder(model_path)
            self.models[i].save(model_path + "model")

    
    def load(self, path):
        """
        Load the model from the given path
        
        Args:
            path: Path where the model is saved (not including the extensions)
        """ 

        for i in range(self.num_model):
            model_path = path + f"/model_{i}/model"
            self.models[i].load(model_path)
    

