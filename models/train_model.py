import numpy as np
import gpytorch
import torch

from models.base_model import BaseModel
from utils.data_structure import pack_result_data

class BaseTrainModel(BaseModel):
    """
    Simple Gaussian Process Model that takes date 
        as inp and return the price prediction.
    """
    def __init__(self, train_data, model_hyperparam):
        super().__init__(train_data, model_hyperparam)
    
    def prepare_data(self):
        """
        Getting all the training data
        """
        raise NotImplementedError()
    
    def build_training_model(self):
        """
        Building the model that will be used for training and prediction
        """
        raise NotImplementedError()
    
    def build_optimizer_loss(self):
        """
        Defining the optimizer and loss objective
        """
        raise NotImplementedError()

    def after_training(self):
        """
        What are we going to do after the training.
        """
        raise NotImplementedError()
    
    def normalize_data(self, data, is_train):
        if is_train:
            self.mean_x = torch.mean(data, axis=0)
            self.std_x = torch.std(data, axis=0)
        
        return (data - self.mean_x)/self.std_x
    
    def train(self):
        self.train_x, self.train_y = self.prepare_data()
        self.model = self.build_training_model()
        self.optimizer, self.loss_obj = self.build_optimizer_loss()

        num_iter = self.hyperparam["optim_iter"]
        for i in range(num_iter):
            self.optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -self.loss_obj(output, self.train_y)

            if self.hyperparam["is_verbose"]:
                if i%10 == 0:
                    print(f"Loss {i}/{num_iter}", loss)
            loss.backward()
            self.optimizer.step()
        
        self.after_training()
    
    def predict_step_ahead(self, test_data, step_ahead, ci=0.9):
        """
        Args: (See superclass)
        Returns: (See superclass)
        """
        raise NotImplementedError()

class BaseTrainMultiTask(BaseTrainModel):
    """
    We will have to handle IndependentMultiModel interface. 
        Using first is by default as we need to merge the data

    """

    def __init__(self, list_train_data, list_config, using_first):
        self.is_using_past_label = list_config[0][0]["is_past_label"]
        assert len(list_train_data) == len(list_config) 
        assert all(
            self.is_using_past_label == c[0]["is_past_label"] 
            for c in list_config
        )
        self.num_task = len(list_config)
        self.using_first = using_first

        # We don't care about the model as we will define here.
        hyperparam = list_config[0][0]
        super().__init__(list_train_data, hyperparam)
    
    def pack_data_merge(self, list_train_data, include_label):
        """
        In the case where the input is the same, 
            but have difference output, we will pack the data 
            together in one matrix by getting the input to be 
            the first one. This implies that the first dataset present, will be the "basis" of the data.
        """
        assert self.using_first

        data_list, label_list = [], []
        for train_data in list_train_data:
            packed_data = self.pack_data(
                train_data, 
                is_label=include_label
            )
            data_list.append(packed_data[:, :-1])
            label_list.append(packed_data[:, [-1]])
        
        return data_list[0], np.concatenate(label_list, axis=1)
    
    def predict(self, list_test_data, list_step_ahead, list_all_date, ci=0.9):
        all_mean, all_lower, all_upper = self.predict_step_ahead(
            list_test_data, list_step_ahead
        )

        pred_list = []
        for i in range(self.num_task):
            pred_list.append(pack_result_data(
                all_mean[:, i].tolist(), 
                all_lower[:, i].tolist(), 
                all_upper[:, i].tolist(),  
                list_all_date[i]
            ))

        return pred_list



        

    

