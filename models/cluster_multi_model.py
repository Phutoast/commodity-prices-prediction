# Super-Abstraction of Multi-Task Learning Algorithm
# Sort the multi-task learning algorithmm into hard cluster and train them
# Very Similar to Ind multi Model

import numpy as np
import pandas as pd
import os
import json
import warnings

from utils import others
from experiments import algo_dict
from utils import data_preprocessing
from utils.data_structure import Hyperparameters

class HardClusterMultiModel(object):
    
    def __init__(self, cluster_train_data, 
    cluster_config, list_using_first, mtl_model):
        
        all_params = [
            mtl_model, cluster_train_data, cluster_config, list_using_first
        ]

        assert all(len(param) == len(all_params[0]) for param in all_params)

        self.all_mult_model = [
            model(clus_data, clus_config, clus_using_first)
            for model, clus_data, clus_config, clus_using_first in zip(*all_params)
        ]
        self.num_cluster = len(cluster_config)
        self.list_config_json = {
            "cluster_info": [len(train_data) for train_data in cluster_train_data],
            "mtl_model": [model.__name__ for model in mtl_model]
        }
    
    def train(self):
        for clus_i, m in enumerate(self.all_mult_model):
            print(f"Training Model at Cluster: {clus_i}/{len(self.all_mult_model)}")
            m.train()
    
    def predict(self, clus_test_data, clus_step_ahead, clus_all_date, ci=0.9, is_sample=False):
        # We will check only the number of cluster 
        #   as each task should be check by individual algorithm

        # If num_cluster == 0, then we are loading the model

        target_num_cluster = self.num_cluster if self.num_cluster != 0 else len(clus_test_data)

        assert all(
            len(a) == target_num_cluster
            for a in [clus_test_data, clus_step_ahead, clus_all_date]
        )

        return [
            self.all_mult_model[i].predict(
                clus_test_data[i], clus_step_ahead[i], clus_all_date[i],
                ci=0.9, is_sample=is_sample
            )
            for i in range(target_num_cluster)
        ]
    
    def save(self, base_path):
        others.create_folder(base_path)
        others.dump_json(f"{base_path}/config.json", self.list_config_json)

        for clus_ind in range(self.num_cluster):
            cluster_path = base_path + f"/cluster_{clus_ind}/"
            others.create_folder(cluster_path)
            self.all_mult_model[clus_ind].save(cluster_path + "mtl_model")
    
    def load(self, path):
        # list_sub_model = sorted([f.path for f in os.scandir(path) if f.is_dir()])
        # self.num_cluster = len(list_sub_model)
        
        # for i, sub_path in enumerate(list_sub_model):
        #     model_path = f"{sub_path}/mtl_model"
        #     self.all_mult_model[i].load(model_path)
        pass
    
    @classmethod
    def load_from_path(cls, path):
        data = others.load_json(f"{path}/config.json") 
        cluster_info = data["cluster_info"]
        mtl_model_name = data["mtl_model"]

        num_cluster = len(cluster_info)

        # Load each model in cluster first, and then construct a model..

        list_loaded_model = []

        list_train_data = []
        list_config = []
        list_using_first = []
        list_loaded_model_class = []

        for i, model_name in enumerate(mtl_model_name):
            model_path = f"{path}/cluster_{i}/mtl_model"

            model_class = algo_dict.multi_task_algo[model_name] 
            loaded = model_class.load_from_path(model_path)

            list_loaded_model.append(loaded) 
            list_train_data.append(loaded.list_train_data) 
            list_config.append(loaded.list_config) 
            list_using_first.append(loaded.using_first)
            list_loaded_model_class.append(model_class)


        model = cls(
            list_train_data, list_config, 
            list_using_first, list_loaded_model_class
        )

        model.all_mult_model = list_loaded_model

        return model

        
    

