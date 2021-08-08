import numpy as np
import gpytorch
import torch

from models.base_model import BaseModel
from utils import data_visualization

from experiments import algo_dict
import copy
import warnings

class BaseTrainModel(BaseModel):
    """
    Simple Gaussian Process Model that takes date 
        as inp and return the price prediction.
    """
    def __init__(self, train_data, model_hyperparam):
        super().__init__(train_data, model_hyperparam)
        self.optim_iter = self.hyperparam["optim_iter"]
    
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
          
        if self.hyperparam["is_gpu"]:
            self.mean_x = self.mean_x.cuda()
            self.std_x = self.std_x.cuda()
        
        return (data - self.mean_x)/self.std_x
    
    def cal_train_loss(self):
        output = self.model(self.train_x)
        loss = -self.loss_obj(output, self.train_y)
        return output, loss

    
    def train(self):
        self.train_x, self.train_y = self.prepare_data()
        self.model = self.build_training_model()

        if self.hyperparam["is_gpu"]:
            self.model = self.model.cuda()

        self.optimizer, self.loss_obj = self.build_optimizer_loss()

        num_iter = self.optim_iter
        for i in range(num_iter):
            self.optimizer.zero_grad()
            output, loss = self.cal_train_loss()

            if self.hyperparam["is_verbose"]:
                if i%10 == 0:
                    print(f"Loss {i}/{num_iter}", loss)

            loss.backward()
            self.optimizer.step()
        
        self.after_training()
    
    def load_kernel(self, kernel_name):
        return copy.deepcopy(algo_dict.kernel_name[kernel_name])
    
    def predict_step_ahead(self, test_data, step_ahead, all_date, ci=0.9):
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

        if self.expect_using_first:
            assert using_first
        else:
            if using_first != self.expect_using_first:
                warnings.warn(UserWarning(f"To gain the best performance, we requires using_first to be {self.expect_using_first}"))
                raise ValueError(f"To gain the best performance, we requires using_first to be {self.expect_using_first}")

        # We don't care about the model as we will define here.
        hyperparam = list_config[0][0]
        super().__init__(list_train_data, hyperparam)
        self.optim_iter = self.num_task * self.hyperparam["optim_iter"]
    
    def merge_all_data(self, data_list, label_list):
        raise NotImplementedError()
    
    def pack_data_merge(self, list_train_data, include_label, using_first=False):
        """
        In the case where the input is the same, 
            but have difference output, we will pack the data 
            together in one matrix by getting the input to be 
            the first one. This implies that the first dataset present, will be the "basis" of the data.
        """
        data_list, label_list = [], []
        expected_data_len = np.inf
        for train_data in list_train_data:
            packed_data = self.pack_data(
                train_data, 
                is_label=include_label
            )

            dataset = packed_data[:, :-1]
            len_data = dataset.shape[0]

            if expected_data_len > len_data:
                expected_data_len = len_data

            data_list.append(dataset)
            label_list.append(packed_data[:, [-1]])

        if using_first: 
            # Using using_first have to reduce the number of dataset, 
            # this is seen as ineffective use of dataset aka. it is the model fault 
            # not able to use all the avaliable data
            
            label_list = [l[:expected_data_len] for l in label_list]
            data_list = [d[:expected_data_len] for d in data_list]
            return data_list[0], np.concatenate(label_list, axis=1)
        else:
            return self.merge_all_data(data_list, label_list)
    
    def predict(self, list_test_data, list_step_ahead, list_all_date, ci=0.9, is_sample=False):

        if not is_sample:
            all_mean, all_lower, all_upper, all_date = self.predict_step_ahead(
                list_test_data, list_step_ahead, list_all_date
            )

            pred_list = []
            for i in range(self.num_task):
                pred_list.append(data_visualization.pack_result_data(
                    all_mean[i].tolist(), 
                    all_lower[i].tolist(), 
                    all_upper[i].tolist(),  
                    all_date[i]
                ))

        else:
            all_sample, all_date = self.predict_step_ahead(
                list_test_data, list_step_ahead, list_all_date, is_sample=is_sample
            )
            pred_list = list(zip(all_sample, all_date))
        
        return pred_list

