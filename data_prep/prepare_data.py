import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
import pickle
import json
from datetime import datetime

from data_prep.corrections import correct_macro_df, correct_flats_info_df
from data_prep.custom_data_creating import add_dates_info, prepare_choosed_features, \
    manual_processing, create_custom_columns
from pred_utils.pred_utils import rmsle_exp, cross_val_split
from configs.columns import flats_param_columns, custom_flats_param_columns

USE_FEATURES = flats_param_columns + custom_flats_param_columns

def prepare_data(train_df, test_df, macro_df):
    train_df = correct_flats_info_df(train_df)
    test_df = correct_flats_info_df(test_df)

    y_price = train_df.price_doc
    del train_df["price_doc"]
    Y_log1p = y_price.apply(np.log1p)

    train_df = create_custom_columns(train_df)
    test_df = create_custom_columns(test_df)


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

    return train_df_processed, test_df_processed, Y_log1p

if __name__=="__main__":

    train_df = pd.read_csv("data/train.csv", parse_dates=['timestamp'])
    test_df = pd.read_csv("data/test.csv", parse_dates=['timestamp'])
    macro_df = pd.read_csv("data/macro.csv", parse_dates=['timestamp'])

    train_df_processed, test_df_processed, Y_log1p = prepare_data(train_df, test_df, macro_df)