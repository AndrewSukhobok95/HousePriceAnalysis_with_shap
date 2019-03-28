import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
import pickle
import json
from datetime import datetime

from pred_utils.pred_utils import rmsle_exp, cross_val_split
from configs.columns import flats_param_columns, custom_flats_param_columns
from data_prep.prepare_data import prepare_data, USE_FEATURES

train_df = pd.read_csv("data/train.csv", parse_dates=['timestamp'])
test_df = pd.read_csv("data/test.csv", parse_dates=['timestamp'])
macro_df = pd.read_csv("data/macro.csv", parse_dates=['timestamp'])


train_df_processed, test_df_processed, Y_log1p = prepare_data(train_df, test_df, macro_df)


train_test_index_list = cross_val_split(train_df_processed, 5, 3)

X = train_df_processed.copy()
del X["timestamp_year_month"]

X_test = test_df_processed.copy()
del X_test["timestamp_year_month"]

model = XGBRegressor()

for (train, val), i in zip(train_test_index_list, range(5)):
    model.fit(X.iloc[train], Y_log1p.iloc[train])
    pred_train = model.predict(X.iloc[train])
    pred_val = model.predict(X.iloc[val])

    print("train RMSLE:", rmsle_exp(Y_log1p.iloc[train], pred_train))
    print("test RMSLE:", rmsle_exp(Y_log1p.iloc[val], pred_val))
    print("-----------------------------")

model.fit(X, Y_log1p)

model_comment = "without_macro"
now_time = datetime.now()
now_time_str = now_time.strftime("%Y_%m_%d_%H_%M")
model_name = "xgb_{}_model_{}_{}".format(xgb.__version__, now_time_str, model_comment)
pickle.dump(model, open("trained_models/" + model_name + ".dat", "wb"))

json_features_data = {'features_names': USE_FEATURES}
# jstr = json.dumps(json_features_data, indent=4)
with open("trained_models/features_json/" + model_name + '.json', 'w') as outfile:
    json.dump(json_features_data, outfile)

pred_test_log1p = model.predict(X_test)
pred_test = np.expm1(pred_test_log1p)

df_sub = pd.DataFrame({'id': test_df.id, 'price_doc': pred_test})

# df_sub.to_csv('sub.csv', index=False)


print()

import shap
shap.force_plot()