import datetime
import pandas as pd
import numpy as np

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

    feature_path = f"data/{metal_type}/{metal_type}_features.csv"
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
    return full_data.iloc[:, :-1], full_data.iloc[:, [0,-1]]

def load_transform_data(metal_type, return_lag, skip=0, is_drop_nan=False):
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
    y = cal_lag_return(y, return_lag)
    X, y = X[:len(X)-return_lag], y[:len(y)-return_lag]
    return X[:len(X)-skip], y[skip:]

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

def cal_lag_return(output, length_lag):
    """
    Calculate the lagged return for the data

    Args:
        output: Price that we wanted to calculate the lag. 
        length_lag: The length between difference returns that we want to calculate. 
            (If equal to zero, then return the original data)
    
    Returns:
        lag_return: Result of Log-Return over a length of lag
    """
    if length_lag != 0:
        length_data = len(output)
        first = output["Price"][:length_data-length_lag].to_numpy()
        second = output.tail(length_data-length_lag)["Price"].to_numpy()
        diff = np.pad(second-first, (0, length_lag), 'constant', constant_values=np.nan)
        lag_return = output.copy()
        lag_return["Price"] = diff
    else:
        lag_return = output

    return lag_return
