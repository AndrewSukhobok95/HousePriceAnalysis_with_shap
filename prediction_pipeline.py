import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn import model_selection, preprocessing
# import xgboost as xgb
from xgboost import XGBRegressor

from data_prep.corrections import correct_macro_df, correct_flats_info_df
from data_prep.custom_data_creating import add_dates_info, prepare_choosed_features, \
    manual_processing, create_custom_columns
from configs.columns import flats_param_columns, custom_flats_param_columns

# import matplotlib.pyplot as plt
# import seaborn as sns
# color = sns.color_palette()
# sns.set()
# import plotly.plotly as py
# import plotly.graph_objs as go
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# pd.options.mode.chained_assignment = None # default='warn'
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.max_rows', 500)


USE_FEATURES = flats_param_columns + custom_flats_param_columns

train_df = pd.read_csv("data/train.csv", parse_dates=['timestamp'])
test_df = pd.read_csv("data/test.csv", parse_dates=['timestamp'])

train_df = correct_flats_info_df(train_df)
test_df = correct_flats_info_df(test_df)

y_price = train_df.price_doc
del train_df["price_doc"]

train_df = create_custom_columns(train_df)
test_df = create_custom_columns(test_df)

macro_df = pd.read_csv("data/macro.csv", parse_dates=['timestamp'])
macro_df = correct_macro_df(macro_df=macro_df)
macro_df.columns = ["timestamp"] + ["macro_" + c for c in macro_df.columns if c!="timestamp"]

train_with_macro_df = pd.merge(train_df, macro_df, how='left', on='timestamp')
test_with_macro_df = pd.merge(test_df, macro_df, how='left', on='timestamp')

train_with_macro_df = add_dates_info(train_with_macro_df)
test_with_macro_df = add_dates_info(test_with_macro_df)




train_df_processed, test_df_processed = prepare_choosed_features(
    train_with_macro_df, USE_FEATURES, test_df=test_with_macro_df,
    dont_touch_cols=["build_year", "timestamp", "material", "max_floor", "timestamp_year_month"])

train_df_processed = manual_processing(train_df_processed)
test_df_processed = manual_processing(test_df_processed)


y_price.apply(np.log1p)


def cross_val_split(train_df, n_folds, fold_size):
    sorted_dates_str = sorted(train_df.timestamp_year_month.unique())

    for i in range(n_folds):
        cur_fold_dates = sorted_dates_str[-fold_size-i:len(sorted_dates_str)-i]







def rmsle_exp(y_true_log1p, y_pred_log1p):
    y_true = np.expm1(y_true_log1p)
    y_pred = np.expm1(y_pred_log1p)
    return np.sqrt(np.mean(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))


# X = train_df_processed.copy()
# del X["timestamp"]
# Y = y_price.apply(np.log)
#
# model = XGBRegressor()
#
# for (train, test), i in zip(cv.split(X, Y), range(5)):
#     model.fit(X.iloc[train], Y.iloc[train])
#     pred_train = model.predict(X.iloc[train])
#     pred_test = model.predict(X.iloc[test])



print()
