import datetime
import pandas as pd

def parse_series_time(dates):
    """
    Given the time from panda dataframe, we turn it to time lengths and label

    Args:
        dates: Pandas series of the observed date. 
    
    Returns:
        time_step: Number of day from the first date. 
        label: Label used for displaying the data
    """

    parse_date = lambda d : datetime.datetime.strptime(d, '%Y-%m-%d')
    first_date = parse_date(dates[0])
    time_step, label = [], []

    for d in dates:
        current_date = parse_date(d)
        time_step.append((current_date - first_date).days)
        label.append(current_date.strftime('%d/%m/%Y'))

    return time_step, label
    
def data_to_date_label(x_data):
    """
    Given the input dataframe, extract the date, 
        including the number of dats from the first date
        and labels.
    
    Args:
        x_data: Training input that includes date data. 
    
    Returns:
        time_step: Number of day from the first date. 
        label: Label used for displaying the data
    """
    
    time_step = x_data["Date"].to_list()
    time_step, label = parse_series_time(time_step)
    return time_step, label


