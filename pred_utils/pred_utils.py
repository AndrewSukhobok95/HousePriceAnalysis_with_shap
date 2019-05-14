import numpy as np
import pandas as pd



def rmsle_exp(y_true_log1p, y_pred_log1p):
    y_true = np.expm1(y_true_log1p)
    y_pred = np.expm1(y_pred_log1p)
    return np.sqrt(np.mean(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))

def cross_val_split(train_df, n_folds, fold_size):
    index_list = []
    sorted_dates_str = sorted(train_df.timestamp_year_month.unique())
    for i in range(n_folds):
        # cur_fold_train_dates = sorted_dates_str[:-fold_size - i]
        # cur_fold_test_dates = sorted_dates_str[-fold_size-i:len(sorted_dates_str)-i]
        cur_fold_train_dates = sorted_dates_str[:-fold_size - n_folds + 1 + i]
        cur_fold_test_dates = sorted_dates_str[-fold_size - n_folds + 1 + i:len(sorted_dates_str) - n_folds + 1 + i]

        print(cur_fold_test_dates)

        cur_fold_train_bool_index = train_df.timestamp_year_month.isin(cur_fold_train_dates)
        cur_fold_test_bool_index = train_df.timestamp_year_month.isin(cur_fold_test_dates)

        cur_fold_train_index = cur_fold_train_bool_index.where(cur_fold_train_bool_index == True).dropna().index
        cur_fold_test_index = cur_fold_test_bool_index.where(cur_fold_test_bool_index == True).dropna().index
        index_list.append([cur_fold_train_index, cur_fold_test_index])
    return index_list
