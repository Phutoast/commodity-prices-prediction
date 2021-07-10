import pickle
import os

from pathlib import Path
from datetime import datetime
import pandas as pd

from experiments.algo_dict import algorithms_dic
from utils.data_structure import FoldWalkForewardResult
from models import ind_multi_model

def create_folder(path):
    """
    Creating Folder given the path that can be nested

    Args:
        path: The path to the folder. 
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def create_name(base_folder, name):
    """
    Create Base Folder Name using the Date Time

    Args:
        base_folder: Base Folder Name
        model_name: The sub-name that goes 
            together with date
    
    Returns:
        modified_base_folder: Appending a 
            new folder under the base folder
    """
    now = datetime.now()
    date_time = now.strftime("%m-%d-%y-%H-%M-%S") + f"-{name}/"
    base_folder += date_time
    return base_folder

def save_fold_data(all_fold_result, model_name, base_folder):
    """
    Given the result of the walk forward model, 
        we save it in the folder separated by each fold.
    
    Args:
        all_fold_result: Result from all forward model
        model_name: Name of the model
    """
    create_folder(base_folder)

    for task_num, fold_result in enumerate(all_fold_result):
        task_folder = base_folder + f"task_{task_num}/"
        create_folder(task_folder)
        for i, (pred, miss_data, intv_loss, model) in enumerate(fold_result):
            curr_folder = task_folder + f"fold_{i}/"
            create_folder(curr_folder)

            pred.to_csv(curr_folder + "pred.csv")
            with open(curr_folder + "miss_data.pkl", "wb") as handle:
                pickle.dump(miss_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
            with open(curr_folder + "intv_loss.pkl", "wb") as handle:
                pickle.dump(intv_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
            model_save_folder = curr_folder + model_name
            create_folder(model_save_folder)
            model.save(model_save_folder)

def load_fold_data(base_folder, model_name, model_class):
    """
    Load the fold data given the base_folder and model name
        The data, including the model, which can be used to generate a plot
    
    Args:
        base_folder: Name of the base folder
        model_name: Name of the model
    
    Returns:
        fold_result: Loaded data in the fold data format
    """
    base_folder = "save/" + base_folder

    task_list = []
    for task_folder in sorted(os.listdir(base_folder)):
        task_folder = base_folder + "/" + task_folder

        fold_result_list = []
        for fold_folder in sorted(os.listdir(task_folder)):
            curr_folder = task_folder + "/" + fold_folder + "/"
            pred = pd.read_csv(curr_folder + "pred.csv")
            with open(curr_folder + "miss_data.pkl", "rb") as handle:
                miss_data = pickle.load(handle)
            
            with open(curr_folder + "intv_loss.pkl", "rb") as handle:
                intv_loss = pickle.load(handle)
            
            model = model_class.load_from_path(
                curr_folder + model_name
            )
            result_fold = FoldWalkForewardResult(
                pred=pred, missing_data=miss_data, interval_loss=intv_loss, model=model
            )
            fold_result_list.append(result_fold)
        
        task_list.append(fold_result_list)
    
    return task_list