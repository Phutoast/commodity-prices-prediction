import pickle
import os

from pathlib import Path
from datetime import datetime
import pandas as pd

from experiments.algo_dict import algorithms_dic
from utils.data_structure import FoldWalkForewardResult

def create_folder(path):
    """
    Creating Folder given the path that can be nested

    Args:
        path: The path to the folder. 
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def save_fold_data(fold_result, model_name):
    """
    Given the result of the walk forward model, 
        we save it in the folder separated by each fold.
    
    Args:
        fold_result: Result from the walk forward model
        model_name: Name of the model
    """
    base_folder = "save/"
    now = datetime.now()
    date_time = now.strftime("%m-%d-%y-%H-%M-%S") + f"-{model_name}/"
    base_folder += date_time

    create_folder(base_folder)

    for i, (pred, miss_data, intv_loss, model) in enumerate(fold_result):
        curr_folder = base_folder + f"fold_{i}/"
        create_folder(curr_folder)

        pred.to_csv(curr_folder + "pred.csv")
        with open(curr_folder + "miss_data.pkl", "wb") as handle:
            pickle.dump(miss_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        with open(curr_folder + "intv_loss.pkl", "wb") as handle:
            pickle.dump(intv_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        create_folder(curr_folder + f"model_{model_name}")
        model.save(curr_folder + f"model_{model_name}/model")

def load_fold_data(base_folder, model_name):
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

    fold_folder_list = []
    for fold_folder in sorted(os.listdir(base_folder)):
        curr_folder = base_folder + "/" + fold_folder + "/"

        pred = pd.read_csv(curr_folder + "pred.csv")
        with open(curr_folder + "miss_data.pkl", "rb") as handle:
            miss_data = pickle.load(handle)
        
        with open(curr_folder + "intv_loss.pkl", "rb") as handle:
            intv_loss = pickle.load(handle)
        
        hyperparam, algo_class = algorithms_dic[model_name]
        model = algo_class([], hyperparam)
        model.load(f"{curr_folder}model_{model_name}/model")
        result_fold = FoldWalkForewardResult(
            pred=pred, missing_data=miss_data, interval_loss=intv_loss, model=model
        )
        fold_folder_list.append(result_fold)
    
    return fold_folder_list


