import numpy as np
import math
from tqdm import tqdm
import itertools

from utils.data_structure import TrainingPoint, FoldWalkForewardResult
from utils.data_preprocessing import parse_series_time
from experiments.calculation import PerformanceMetric

def prepare_dataset(X, first_day, y, len_inp, 
            len_out=22, return_lag=22, is_padding=False, convert_date=True, offset=1, is_show_progress=False, num_dataset=-1):
    """
    Creating Set for the prediction Chopping up the data (can be used for both training and testing):
            -----+++xxxx
            (offset)-----+++xxxx
                    (offset)-----+++xxxx
                            (lack)-----+++xxxx
            ssssssssssssssssssssssssssssssssss
        where - indicates input-data
              x indicates output-data
              + indicates lags
    
    if padding is true then, we will find the offset (lack) that allow us to get the full data
    
    Args:
        X: Training input (not including prices)
        first_day: First Day of the dataset (for calculating a relative time)
        y: Training output
        len_input: Length of input data
        len_out: Length of output horizon we want to predict
        return_lag: Lagging of the return (used in lagged return)
        is_padding: Pad the dataset so that we cover all the data in the dataset.
        convert_date: Convert the date to a number or not.
        offset: Number of data that got left-out from previous training data (If -1 then we consider the partion)
        is_show_progress: showing tqdm progress bar
        num_dataset: getting first_numdataset dataset we want to output (-1 if we want all)
    
    Return:
        train_set: Training set, including the input and out 
    """
    
    size_data = len(X)
    size_subset = len_inp+len_out+return_lag
    assert size_subset <= size_data

    if offset == -1:
        offset = size_subset
    
    num_offset_apply = math.floor((size_data-size_subset)/offset)
    num_subset = num_offset_apply + 1 if num_dataset == -1 else num_dataset
    all_subset = []

    def split_data(start_index):
        
        """
        Method for splitting data into usable forms

        Args:
            start_index: index where the cut starts
        
        Return:
            data: tuple containing all the data
        """
        data = X[start_index:start_index+size_subset].copy()
        label = y[start_index:start_index+size_subset].copy()

        if convert_date:
            date_val, _ = parse_series_time(data["Date"].to_list(), first_day)

            # This is universal so there shouldn't be a problem ?
            data.loc[:, "Date"] = date_val

        data_inp, data_out = data[:len_inp], data[len_inp+return_lag:]
        label_inp, label_out = label[:len_inp], label[len_inp+return_lag:]
        return data_inp, label_inp, data_out, label_out

    for index in tqdm(range(num_subset), disable=(not is_show_progress)):
        start_index = index*offset
        data = split_data(start_index)

        all_subset.append(
            TrainingPoint(*data)
        )
    
    if is_padding and num_subset == num_offset_apply+1:
        # Have to check that whether padding is needed or not ?
        last_index = num_offset_apply*offset + size_subset
        if last_index != len(X):
            data = split_data(len(X)-size_subset)
            all_subset.append(
                TrainingPoint(*data)
            )
    
    return all_subset

def transpose_list(list_mat):
    result = []
    num_c = len(list_mat[0])
    num_r = len(list_mat)
    for i in range(num_c):
        temp = []
        for j in range(num_r):
            temp.append(list_mat[j][i]) 
        result.append(temp)
    
    return result
    

def walk_forward(all_data, task_setting, multi_model_class, size_train, 
            size_test, train_offset, return_lag_list, convert_date, using_first,
            is_train_pad=True, is_test_pad=False):
    """
    Performing walk forward testing (k-fold like) of the models
        In terms of setting up training and testing data.
        This works similar to above where the offset is size_test.
    
    A------xxx
       B------xxx
           C------xxx
    
    size_train: Length of (------) -- Within training we can have offset calculating the tests
    size_test: Length of (xxx) -- Within testing we can have offset for calculating the tests
    
    Within the "fold" we have, At training section:
    A------------------iiiiiii
     (train_offset)------------------iiiiiii
                    (train_offset)------------------iiiiiii
                                  (padding)------------------iiiiiii
    I----------------------training length-------------------------I

    At testing section:

    xxxxxxxxxxxxxxxxxxkkkkkkkk
    (test_offset)xxxxxxxxxxxxxxxxxxkkkkkkkk
                (padding)xxxxxxxxxxxxxxxxxxkkkkkkkk
    I-------------testing length------------------I
                                                
    
    len_inp (in model_hyperparam): Length of ------- && xxxxxxxx
    len_out (in model_hyperparam): Length of iiiiiii
    len_out: Length of kkkkkkkk

    Args:
        X: All data avaliable
        y: All label avaliable
        algo_class: Training Model class
        model_hyperparam: Hyperparameter for the model 
            (will be used to construct a model object)
        loss: Loss for calculating performance.
        size_train: number of training size
        size_test: number of testing size
        train_offset: Offset during the training. 
        test_offset: Offset during the testing.
        return_lag: Lagging of the return (used in lagged return)
        is_train_pad: Pad the dataset so that we cover all the data in the training set.
        is_test_pad: Pad the dataset so that we cover all the data in the testing set.
        intv_loss: Loss on interval if none then it is normal loss averaged
    
    Return:
        fold_result_list: Performance of model + Model
    """

    task_list_all = []
    for X, y, _, algo_class in all_data:
        model_hyperparam, model_class = algo_class

        len_inp = model_hyperparam["len_inp"]
        len_out = model_hyperparam["len_out"]
        size_subset = len_inp + len_out
        assert size_subset <= size_test and size_subset <= size_train

        first_day = X["Date"][0]
        fold = prepare_dataset(X, first_day, y, size_train, 
                    len_out=size_test, convert_date=True, 
                    offset=size_test, return_lag=0, 
                    is_padding=False) 
        task_list_all.append(fold)
    
    assert all(len(fold) == len(task_list_all[0]) for fold in task_list_all)

    fold_list_task = transpose_list(task_list_all)
    
    fold_result_list = []
    for i, all_task_data in enumerate(fold_list_task):
        print("At fold", i+1, "/", len(fold_list_task))

        train_dataset_list = []
        algo_hyper_class_list = []

        pred_dataset_list = []
        date_pred_list = []
        len_pred_list = []
        
        true_pred_list = []
        missing_data_list = []
        algo_prop_list = []

        for j, (X_train, y_train, X_test, y_test) in enumerate(all_task_data):
            _, _, convert_date, algo_class = all_data[j]
            model_hyperparam, model_class = algo_class
            model_hyperparam["using_first"] = using_first

            len_inp = model_hyperparam["len_inp"]
            len_out = model_hyperparam["len_out"]

            return_lag, skip, _ = task_setting["dataset"][j]["out_feat_tran_lag"]

            train_dataset = prepare_dataset(
                X_train, None, y_train, 
                len_inp=len_inp, len_out=len_out, return_lag=return_lag, 
                offset=train_offset, is_padding=is_train_pad, convert_date=False
            )

            train_dataset_list.append(train_dataset)
            algo_hyper_class_list.append((model_hyperparam, model_class))
        
            test_dataset = prepare_dataset(
                X_test, None, y_test, len_inp, 
                len_out=len_out, return_lag=return_lag, 
                offset=len_out,is_padding=is_test_pad, convert_date=False
            )
            all_date_pred = list(
                itertools.chain.from_iterable([point.data_out["Date"].map(convert_date).to_list() for point in test_dataset])
            )

            pred_dataset_list.append(test_dataset)

            if not using_first:
                len_pred_list.append(len(all_date_pred))
                date_pred_list.append(all_date_pred)
            else:
                if j == 0:
                    basis_time_step = [convert_date.reverse(d) for d in all_date_pred]
                
                    len_pred_list.append(len(all_date_pred))
                    date_pred_list.append(all_date_pred)
                else:
                    len_pred_list.append(len(all_date_pred))
                    all_date_pred = [convert_date(d) for d in basis_time_step]
                    
                    date_pred_list.append(all_date_pred)

            # date_pred_list.append(all_date_pred)
            # len_pred_list.append(len(all_date_pred))
        
            true_date = X_test["Date"].map(convert_date).to_list()
            true_price = y_test["Output"].to_list() 

            missing_x = true_date[:len_inp+return_lag]
            missing_y = true_price[:len_inp+return_lag] 
            missing_data = (missing_x, missing_y)

            true_pred_list.append((true_date, true_price))
            missing_data_list.append(missing_data)
            algo_prop_list.append((len_inp, len_out, return_lag))

        model = multi_model_class(
            train_dataset_list, 
            algo_hyper_class_list,
            using_first
        )
        model.train()
        all_task_pred = model.predict(
            pred_dataset_list, 
            len_pred_list, 
            date_pred_list, 
            ci=0.9
        )
        all_task_sample = model.predict(
            pred_dataset_list, 
            len_pred_list, 
            date_pred_list, 
            ci=0.9, is_sample=True
        )

        summary_iter = enumerate(zip(
            all_task_pred, all_task_sample, 
            true_pred_list, missing_data_list, algo_prop_list
        ))

        task_result_list = []        
        for k, (pred, task_sample, date_price, miss_data, algo_prop) in summary_iter:
            len_inp, len_out, return_lag = algo_prop
            true_date, true_price = date_price
            pred, loss_detail = cal_walk_forward_result(
                pred, true_date, true_price, 
                len_inp, len_out, 
                return_lag, task_sample
            )

            result_fold = FoldWalkForewardResult(
                pred=pred, missing_data=miss_data, model=model, loss_detail=loss_detail
            )
            task_result_list.append(result_fold)
    
        fold_result_list.append(task_result_list)

    return transpose_list(fold_result_list)


def cal_walk_forward_result(pred, true_date, 
    true_price, len_inp, len_out, return_lag, task_sample):

    true_date_pred = true_date[len_inp+return_lag:]
    true_data_pred = true_price[len_inp+return_lag:]
    
    metric = PerformanceMetric()
    loss_detail = {}
    mapper = dict(zip(true_date_pred, true_data_pred))
    
    pred["true_mean"] = pred["x"].map(mapper)

    loss_detail["time_step_error"] = metric.square_error(
        pred["true_mean"], pred["mean"]
    ).to_list()

    loss_detail["all_crps"] = metric.crps(
        task_sample[0], np.array(true_data_pred[:len(task_sample[1])])
    )

    # Sometimes the method uses the test data to do the prediction 
    # Retrain, such as in the case of ARIMA so it is better to consider the loss
    # It is useful to have multiple output error data too 

    assert len(pred)%len_out == 0

    interval_loss = []
    intv_loss = lambda x, y : np.median(metric.square_error(x, y))

    for i in range(len(pred)//len_out):
        parti = pred.iloc[i*len_out:(i+1)*len_out, :]
        interval_loss.append((
            parti["x"].iloc[0].item(), 
            parti["x"].iloc[-1].item(), 
            intv_loss(parti["true_mean"], parti["mean"]).item()
        ))
    
    loss_detail["intv_loss"] = interval_loss
    
    return pred, loss_detail

