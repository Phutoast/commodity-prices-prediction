import numpy as np
import pandas as pd
import os
import json

from utils import others
from models.base_model import BaseModel
from experiments.algo_dict import class_name
from utils.data_preprocessing import replace_dataset

class IndependentMultiModel(object):
    """
    Wrapper for multiple independent models

    Args:
        list_train_data: Data from the training set
            if it is not a list then we apply the same train_data to all models
        list_config: List of each model 
            and its hyperparameter (as seen in algo_dict), 
            if it is not a list then we apply the same hyperparameter to all models
        using_first: Using the first data set inputs but with difference labels.

    """
    def __init__(self, list_train_data, list_config, using_first):
        assert len(list_train_data) == len(list_config) 
        self.num_model = len(list_config)
        self.using_first = using_first

        self.models = []  
        self.list_config_json = {"hyperparam": [], "model_class": [], "using_first": using_first}

        if self.using_first:
            list_train_data = replace_dataset(list_train_data)

        for i in range(self.num_model):
            self.models.append(
                list_config[i][1](list_train_data[i], list_config[i][0])
            )
            self.list_config_json["hyperparam"].append(
                dict(list_config[i][0])
            )
            self.list_config_json["model_class"].append(
                list_config[i][1].__name__
            )
        

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

        # self.num_model == 0 when we are loading a model
        if self.num_model != 0:
            assert all(
                len(a) == self.num_model 
                for a in [list_test_data, list_step_ahead, list_all_date]
            )
        else:
            assert all(
                len(a) == len(list_test_data)
                for a in [list_test_data, list_step_ahead, list_all_date]
            )
        
        if self.using_first:
            list_test_data = replace_dataset(list_test_data)

        return [
            # WE have to follow the first one, so we will ignore the actual dataset.
            self.models[i].predict(
                list_test_data[i], list_step_ahead[i], list_all_date[i], ci=ci
            )
            for i in range(self.num_model)
        ]
    
    def save(self, base_path):
        """
        Save the model to the path given

        Args:
            base_model: The name of the whole multimodel
        """
        others.create_folder(base_path)

        with open(f"{base_path}/config.json", 'w', encoding="utf-8") as f:
            json.dump(
                self.list_config_json, f, ensure_ascii=False, indent=4
            )

        for i in range(self.num_model):
            model_path = base_path + f"/model_{i}/"
            others.create_folder(model_path)
            self.models[i].save(model_path + "model")

    
    def load(self, path):
        """
        Load the model from the given path, 
            if we already know the underlying models
        
        Args:
            path: Path where the model is saved (not including the extensions)
        """ 

        list_sub_model = [f.path for f in os.scandir(path) if f.is_dir()]
        self.num_model = len(list_sub_model)

        for i, sub_path in enumerate(list_sub_model):
            model_path = f"{sub_path}/model"
            self.models[i].load(model_path)
    
    @classmethod
    def load_from_path(cls, path):
        """
        Create Model from the path without 
            knowing the underlying models

        Args:
            path: Path of the folder where the model is saved 
        """
        with open(f"{path}/config.json", 'r', encoding="utf-8") as f:
            data = json.load(f)
        
        num_model = len(data["hyperparam"])
        assert num_model == len(data["model_class"])

        # Constructing list config
        list_config = []
        for hyper, name in zip(data["hyperparam"], data["model_class"]):
            list_config.append((hyper, class_name[name]))
        
        model = cls(
            [[]] * num_model, 
            list_config, 
            data["using_first"]
        )
        model.load(path)
        
        return model



    

