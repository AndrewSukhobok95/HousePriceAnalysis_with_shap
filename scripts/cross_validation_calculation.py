import os
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
import pickle
import json
from datetime import datetime

from sklearn.model_selection import KFold

from pred_utils.pred_utils import rmsle_exp, cross_val_split
from data_prep.prepare_data import prepare_data

from configs.columns import essential_columns, macro_cols

train_df = pd.read_csv("./../data/train.csv", parse_dates=['timestamp'])
test_df = pd.read_csv("./../data/test.csv", parse_dates=['timestamp'])
macro_df = pd.read_csv("./../data/macro.csv", parse_dates=['timestamp'])

train_df_processed, test_df_processed, Y_log1p = prepare_data(train_df, test_df, macro_df)

with open("./../configs/good_features_06.txt", "r") as f:
    not_corr_other_columns = list(map(lambda x: x.replace("\n", ""), f.readlines()))


train_df_for_pred = train_df_processed[["timestamp_year_month"] + essential_columns + not_corr_other_columns + macro_cols]
test_df_for_pred = test_df_processed[["timestamp_year_month"] + essential_columns + not_corr_other_columns + macro_cols]

train_test_index_list = cross_val_split(train_df_for_pred, 5, 3)

X = train_df_for_pred.copy()
del X["timestamp_year_month"]

X_test = test_df_for_pred.copy()
del X_test["timestamp_year_month"]

model_name = "xgb_0.81_model_2019_05_06_15_19_5th"
model = pickle.load(open("./../trained_models/" + model_name + ".dat", "rb"))

result_metrics = []

for (train, val), i in zip(train_test_index_list, range(5)):
    pred_train = model.predict(X.iloc[train])
    pred_val = model.predict(X.iloc[val])

    train_rmsle_ts = rmsle_exp(Y_log1p.iloc[train], pred_train)
    test_rmsle_ts = rmsle_exp(Y_log1p.iloc[val], pred_val)

    result_metrics.append(["ts_split_{}".format(i), train_rmsle_ts, test_rmsle_ts])

kf = KFold(n_splits=5, random_state=2105, shuffle=True)
for train, val in kf.split(X):
    pred_train = model.predict(X.iloc[train])
    pred_val = model.predict(X.iloc[val])

    train_rmsle_kfold = rmsle_exp(Y_log1p.iloc[train], pred_train)
    test_rmsle_kfold = rmsle_exp(Y_log1p.iloc[val], pred_val)

    result_metrics.append(["kfold_split", train_rmsle_kfold, test_rmsle_kfold])

df_metrics = pd.DataFrame(result_metrics,
                          columns=["split_type", "train_rmsle", "test_rmsle"])

df_metrics.to_csv("./../metrics.csv")

print("done!")


