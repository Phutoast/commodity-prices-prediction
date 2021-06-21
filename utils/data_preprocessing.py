import datetime
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from utils.data_structure import TrainingPoint
import random

def parse_series_time(dates, first_day):
    """
    Given the time from panda dataframe, we turn it to time lengths and label

    Args:
        dates: Pandas series of the observed date. 
    
    Returns:
        time_step: Number of day from the first date. 
        label: Label used for displaying the data
    """

    parse_date = lambda d : datetime.datetime.strptime(d, '%Y-%m-%d')
    first_day = parse_date(first_day)
    time_step, label = [], []

    for d in dates:
        current_date = parse_date(d)
        time_step.append((current_date - first_day).days)
        label.append(current_date.strftime('%d/%m/%Y'))

    return time_step, label
    
def load_metal_data(metal_type):
    """
    Loading the metal data (both feature and *raw* prices). 
    The files will be stores given the path: 
        data/{metal_type}/{metal_type}_features.xlsx
        data/{metal_type}/{metal_type}_raw_prices.xlsx

    Args:
        metal_type: type of the matal, we want to import
    Returns
        features: the feature of the related to the metal
        prices: the prices of metal without any preprocessing 
    """

    feature_path = f"data/{metal_type}/{metal_type}_features_true.csv"
    price_path = f"data/{metal_type}/{metal_type}_raw_prices.csv"

    feature = pd.read_csv(feature_path)

    # Rename column 
    columns = list(feature.columns)
    columns[0] = "Date"
    feature.columns = columns
    feature = feature.drop([0,1,2]).reset_index(drop=True)

    price = pd.read_csv(price_path)
    price.columns = ["Date", "Price"]

    # Adding Missing Dates
    diff_date = set(price['Date'])- set(feature['Date']) 
    original_len = len(feature)    
    for i, date in enumerate(diff_date):
        feature = feature.append(pd.Series(), ignore_index=True)
        feature["Date"][original_len+i] = date
    
    feature = feature.sort_values(by=["Date"]).reset_index(drop=True)
     
    return pd.merge(feature, price, on="Date")

def transform_full_data(full_data, is_drop_nan=False):
    """
    Given the concatenated data, we transform 
        and clean the data with splitting the feature and price.

    Args:
        full_data: Concatenated data 
        is_drop_nan: Remover any row that contains NaN in it.

    Returns:
        x: Feature over time. 
        y: (log)-Price over time.
    """

    if is_drop_nan:
        full_data = full_data.dropna()
    
    full_data["Price"] = np.log(full_data["Price"])
    return full_data.iloc[:, :-1], full_data.iloc[:, -1]

def load_transform_data(metal_type, is_drop_nan=False):
    """
    Loading and Transform the data in one function 
        to get the dataset and label. We will assume log-price. 
    
    Args:
        metal_type: Type of metal (aka type of data)
        is_drop_nan: Remove any row that contains NaN in it.
    
    Returns:
        X: Feature over time. 
        y: (log)-Price over time.
    """
    data_all = load_metal_data(metal_type)
    X, y = transform_full_data(data_all, is_drop_nan=is_drop_nan)
    return X, y

def df_to_numpy(data, label):
    """
    Turning a DataFrame into numpy array.

    Args:
        data: Pandas DataFrame of the data
        label: Pandas DataFrame of the data
    
    Returns:
        data: numpy array for the data
        label: numpy array for label
    """
    data = data.apply(pd.to_numeric).to_numpy()
    label = label.to_numpy()
    return data, label

def prepare_dataset(X, first_day, y, len_inp, len_out=22, convert_date=True, 
                        is_rand=False, offset=1, 
                        relative_time=False, is_show_progress=False):
    """
    Creating Set for the prediction Chopping up the data (can be used for both training and testing):
            -------xxxx
            (offset)-------xxxx
                    (offset)-------xxxx
        where - indicates input-data
              x indicates output-data
    
    Args:
        X: Training input (not including prices)
        first_day: First Day of the dataset (for calculating a relative time)
        y: Training output
        len_input: Length of input data
        len_out: Length of output horizon we want to predict
        convert_date: Convert the date to a number or not.
        is_rand: If True, Shuffle the data-subset and then split training-testing
        offset: Number of data that got left-out from previous training data (If -1 then we consider the partion)
        relative_time: If True, each date feature will start at 0, else, we set date from the first datapoint
        is_show_progress: showing tqdm progress bar
    
    Return:
        train_set: Training set, including the input and out 
    """

    size_data = len(X)
    size_subset = len_inp+len_out
    assert size_subset <= size_data

    if offset == -1:
        offset = size_subset

    num_subset = math.floor((size_data-size_subset)/offset) + 1
    all_subset = []

    for index in tqdm(range(num_subset), disable=(not is_show_progress)):
        start_index = index*offset
        data = X[start_index:start_index+size_subset]
        label = y[start_index:start_index+size_subset]

        if convert_date:
            if not relative_time: 
                date_val, _ = parse_series_time(data["Date"].to_list(), first_day)
            else:
                date_val = np.arange(size_subset)

            data.loc[:, "Date"] = date_val

        data_inp, data_out = data[:len_inp], data[len_inp:]
        label_inp, label_out = label[:len_inp], label[len_inp:]

        all_subset.append(
            TrainingPoint(data_inp, label_inp, data_out, label_out)
        )
    
    if is_rand:
        random.shuffle(all_subset)

    return all_subset

def walk_forward(X, y, model, model_hyperparam, loss, size_train, 
            size_test, train_offset, test_offset, test_step, is_rand=False, relative_time=False):
    """
    Performing walk forward testing (k-fold like) of the models
        In terms of setting up training and testing data.
        This works similar to above where the offset is size_test.
    
    A------xxx
       B------xxx
           C------xxx
    
    size_train: Length of (------) -- Within training we can have offset calculating the tests
    size_test: Length of (xxx) -- Within testing we can have offset for calculating the tests

    Args:
        X: All data avaliable
        y: All label avaliable
        model: Training Model class
        model_hyperparam: Hyperparameter for the model 
            (will be used to construct a model object)
        loss: Loss for calculating performance.
        size_train: number of training size
        size_test: number of testing size
        train_offset: Offset during the training. 
        test_offset: Offset during the testing.
        test_step: number of testing ahead we want to test for
        is_rand: If true, shuffle the data-subset and then split training-testing
        relative_time: If true, each date feature will 
            start at 0, else, we set date from the first datapoint
    
    Return:
        perf: Performance of model given
    """

    len_inp, len_out = model_hyperparam["len_inp"], model_hyperparam["len_out"]
    size_subset = len_inp + len_out
    assert size_subset <= size_test and size_subset <= size_train

    first_day = X["Date"][0]
    fold_list = prepare_dataset(X, first_day, y, size_train, 
                len_out=size_test, convert_date=False, 
                is_rand=is_rand, offset=size_test, 
                relative_time=relative_time)
    
    loss_list, num_test_list = [], []
    for i, data in enumerate(fold_list):
        X_train, y_train, X_test, y_test = data

        train_dataset = prepare_dataset(X_train, first_day, y_train, len_inp, 
                            len_out=len_out, is_rand=is_rand, 
                            offset=train_offset, relative_time=relative_time)  

        model_fold = model(train_dataset, model_hyperparam)         
        model_fold.train()

        # Testing Model
        test_dataset = prepare_dataset(X_test, first_day, y_test, len_inp, 
                            len_out=test_step, is_rand=is_rand, offset=1)

        loss_total, num_test = 0, 0
        for j, data_test in enumerate(test_dataset):
            # We assume that during the prediction there is no change of state within the predictor
            model_pred = model_fold.predict(data_test, step_ahead=test_step)
            loss_total += loss(data_test.label_out, model_pred)
            num_test += 1
        
        loss_list.append(loss_total)
        num_test_list.append(num_test)

    return loss_list, num_test_list

