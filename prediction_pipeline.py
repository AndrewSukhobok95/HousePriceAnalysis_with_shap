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
from pred_utils.pred_utils import rmsle_exp, cross_val_split
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
Y_log1p = y_price.apply(np.log1p)

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
pred_test_log1p = model.predict(X_test)
pred_test = np.expm1(pred_test_log1p)

df_sub = pd.DataFrame({'id': test_df.id, 'price_doc': pred_test})

df_sub.to_csv('sub.csv', index=False)


print()
