from datetime import datetime
import pandas as pd
import numpy as np
import warnings

from utils import data_structure
from sklearn.decomposition import PCA
import copy

column_transform = {
    "id": lambda x: x,
    "log": lambda x: np.log(x),
    "sin": lambda x: np.sin(x),
}

class GlobalModifier(object):
    """
    TODO: Cache this thing.....
    """
    def __init__(self, compress_method):
        self.compress_dim, self.method = compress_method

        if self.method.lower() == "id":
            self.is_drop = False
            self.is_id = True
            self.compress_dim = 0
        elif self.method.lower() == "drop":
            self.is_drop = True
            self.is_id = False
            self.compress_dim = 0
        elif self.method.lower() == "pca":
            self.is_drop = True
            self.is_id = False
        else:
            assert False

        self.base_name = "FeatureFamily."
    
    def extract_numpy(self, df):
        self.price = df["Price"]
        self.date = df["Date"]
        np_data = df.loc[:, df.columns != "Price"]
        np_data = np_data.loc[:, np_data.columns != "Date"].to_numpy(dtype=np.float32)
        return np_data
    
    def numpy_to_df(self, np_arr):
        df = {"Date": self.date}
        for i in range(self.compress_dim):
            df.update({f"{self.base_name}Feature{i+1}": np_arr[:, i]})
        df.update({"Price": self.price})
        return pd.DataFrame(df)
    
    def __call__(self, data):
        original_dim = len(data.columns)
        if original_dim == self.compress_dim:
            return data

        if self.is_id:
            return data

        if self.is_drop:
            data = data.dropna()

        if self.method.lower() == "pca": 
            np_data = self.extract_numpy(data)
            pca = PCA(n_components=self.compress_dim)
            reduced_data = pca.fit_transform(np_data)
            return self.numpy_to_df(reduced_data)
        elif self.method.lower() == "drop": 
            return data
        

identity_modifier = GlobalModifier((0, "id"))

def parse_series_time(dates, first_day):
    """
    Given the time from panda dataframe, we turn it to time lengths and label

    Args:
        dates: Pandas series of the observed date. 
    
    Returns:
        time_step: Number of day from the first date. 
        label: Label used for displaying the data
    """
    
    parse_date = lambda d : datetime.strptime(d, '%Y-%m-%d')
    first_day = parse_date(first_day)
    time_step, label = [], []

    for d in dates:
        current_date = parse_date(d)
        time_step.append((current_date - first_day).days)
        label.append(current_date.strftime('%d/%m/%Y'))

    return time_step, label
    
def load_metal_data(metal_type, global_modifier):
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
     
    # Will Train with Features for now
    return global_modifier(pd.merge(feature, price, on="Date"))

def transform_full_data(
        full_data, 
        feature_name="Price",
        trans_column=lambda x: np.log(x),
        use_only_last=False
    ):
    """
    Given the concatenated data, we transform 
        and clean the data with splitting the feature and price.

    Args:
        full_data: Concatenated data 

    Returns:
        x: Feature over time. 
        y: (log)-Price over time.
    """
    full_data[feature_name] = trans_column(full_data[feature_name])

    all_col = list(full_data.columns)
    index_feat = all_col.index(feature_name)
    if not use_only_last:
        rest_of_col = list(range(len(all_col)))
        rest_of_col.remove(index_feat)

        return full_data.iloc[:, rest_of_col], full_data.iloc[:, [0, index_feat]]
    else:
        return full_data.iloc[:, [0]], full_data.iloc[:, [0, index_feat]]

def load_transform_data(
        metal_type, return_lag, 
        skip=0,
        feature_name="Price",
        trans_column=lambda x: np.log(x),
        use_only_last=False, 
        global_modifier=identity_modifier
    ):
    """
    Loading and Transform the data in one function 
        to get the dataset and label. We will assume log-price. 
    
    Args:
        metal_type: Type of metal (aka type of data)
    
    Returns:
        X: Feature over time. 
        y: (log)-Price over time.
    """
    data_all = load_metal_data(metal_type, global_modifier)
    X, y = transform_full_data(
        data_all, feature_name=feature_name, trans_column=trans_column,
        use_only_last=use_only_last
    )
    y = cal_lag_return(y, return_lag, feature_name)
    if not use_only_last:
        X, y = X[:len(X)-return_lag], y[:len(y)-return_lag]
        return X[:len(X)-skip], y[skip:]
    else:
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

def cal_lag_return(output, length_lag, feature_name="Price"):
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
        first = output[feature_name][:length_data-length_lag].to_numpy()
        second = output.tail(length_data-length_lag)[feature_name].to_numpy()
        diff = np.pad(second-first, (0, length_lag), 'constant', constant_values=np.nan)
        lag_return = output.copy()
        lag_return[feature_name] = diff
    else:
        lag_return = output

    return lag_return

def replace_dataset(list_train_data):
    """
    When we call for the using_first in multi-task setting, 
        we will have to replace the dataset with the current 
        input data (but difference output).
    
    Args:
        list_train_data: A list of data for each task
    
    Return:
        all_train_data: Modified training data
    """
    all_train_data = [] 
    for i in range(len(list_train_data)):
        if i == 0:
            all_label_out = [
                (a, b, c) for a, b, c, _ in list_train_data[i]
            ] 
            current_train_data = list_train_data[i]
        else:
            current_train_data = [
                data_structure.TrainingPoint(a, b, c, d) for (_, _, _, d), (a, b, c)
                in zip(list_train_data[i], all_label_out)
            ] 
        
        all_train_data.append(current_train_data)
    
    return all_train_data

def load_dataset_from_desc(dataset_desc):

    # We assume under the same date 
    # (because skipping and return_log doesn't make sense)
    default_trans = (0, 0, "id")
    base_name = "FeatureFamily."

    def find_version(list_parse):
        if len(list_parse) != 1:
            assert len(list_parse) == 2
            feature_version = list_parse[-1]
            ver = f"-{feature_version}"
        else:
            ver = ""
        return ver


    def get_loc_feat(feat_name):
        if feat_name != "Date":
            parse_out_feat = feat_name.split(".")
            out_metal_ind = inp_metal_list.index(parse_out_feat[0])

            feature_parse = parse_out_feat[1].split("-")
            feature_name = feature_parse[0]

            ver = find_version(feature_parse)

            if feature_name == "Price":
                return (out_metal_ind, feature_name+ver)
            else:
                return (out_metal_ind, base_name+feature_name+ver)
        else:
            return (None, "Date")
    
    def get_final_feat_name(loc, name):

        name_list = name.split("-")
        name = name_list[0]

        ver = find_version(name_list)

        if name == "Date":
            return "Out_Date"
        
        if name == "Price":
            return inp_metal_list[loc] + ".Price" + ver
         
        return inp_metal_list[loc] + "." + name.split(".")[1] + ver
    
    def load_data_all(dataset_index, feature_name, return_trans):
        if return_trans is None:
            return_trans = default_trans
        
        full_feature_name = feature_name
        # A trick is that if there is no separators, then we doesn't change anything
        feature_name = feature_name.split("-")[0]

        return_lag, skip, transform = return_trans
        if feature_name == "Price":
            transform = "log"

        # We will use the Date of the first dataset here.
        if dataset_index is None:
            assert feature_name == "Date"
            dataset_index = 0
            transform = "id"

        date, column = load_transform_data(
            inp_metal_list[dataset_index], 
            return_lag=return_lag,
            skip=skip,
            feature_name=feature_name,
            trans_column=column_transform[transform],
            use_only_last=True,
            global_modifier=GlobalModifier(dataset_desc["metal_modifier"][dataset_index])
        )

        column.columns = ["Date", get_final_feat_name(
            dataset_index, full_feature_name
        )]

        return date, column
    
    inp_metal_list = dataset_desc["inp_metal_list"]
    use_feat = dataset_desc["use_feature"]
    use_feat_tran_lag = dataset_desc["use_feat_tran_lag"]
    out_feat = dataset_desc["out_feature"]

    list_loc_use_feat = [get_loc_feat(feat) for feat in use_feat]
    out_dataset_index, out_feature_name = get_loc_feat(out_feat)
    out_return_trans = dataset_desc["out_feat_tran_lag"]

    # Getting the Out Data
    out_date, out_column = load_data_all(
        out_dataset_index, 
        out_feature_name, 
        out_return_trans
    )

    # Finding the most common dates
    all_dates = set(out_date["Date"])
    all_inp_col = []

    for return_trans, (data_index, feat_name) in zip(use_feat_tran_lag, list_loc_use_feat):
        inp_date, inp_column = load_data_all(
            data_index, 
            feat_name, 
            return_trans
        )
        all_dates = all_dates & set(inp_date["Date"])
        all_inp_col.append((inp_date, inp_column))
    
    all_dates = sorted(
        list(all_dates), 
        key=lambda x: datetime.timestamp(datetime.strptime(x, '%Y-%m-%d'))
    )

    data_frame = {}

    def extract_data(date, col):
        feat_name = col.columns[1]
        all_indices_use = [
            date["Date"].to_list().index(d) 
            for d in all_dates
        ]
        data = col[feat_name].to_numpy()[all_indices_use]
        assert feat_name != "Date"

        if feat_name == "Out_Date":
            feat_name = "Date"
        
        return {feat_name:data}

    for date, col in all_inp_col: 
        data_frame.update(extract_data(date, col))
    
    input_col_name = list(data_frame.keys())
    
    data_frame.update({"Date-Out": data_frame["Date"]})

    output_feature = extract_data(out_date, out_column)
    list_feature_name = list(output_feature.keys())
    assert len(list_feature_name) == 1
    output_col_name = ["Date-Out", "Output"]
    output_feature = {"Output": output_feature[list_feature_name[0]]}
    data_frame.update(
        output_feature
    )
 
    all_data_frame = pd.DataFrame(data_frame)
    if dataset_desc["is_drop_nan"]:
        if all_data_frame["Output"].isnull().values.any():
            warnings.warn(UserWarning("There is a NaN in the output, we can still drop it but the time series may be irregular."))
        all_data_frame = all_data_frame.dropna()
    
    out_df = all_data_frame[output_col_name]
    out_columns = out_df.columns
    assert out_columns[0] == "Date-Out" and len(out_columns) == 2
    out_df.columns = ["Date", "Output"]

    return all_data_frame[input_col_name], out_df