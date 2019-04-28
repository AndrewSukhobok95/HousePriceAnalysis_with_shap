import os
import numpy as np
import pandas as pd

from data_prep.corrections import correct_macro_df, correct_flats_info_df
from data_prep.custom_data_creating import add_dates_info, prepare_choosed_features, create_custom_columns
from data_prep.utils import prepare_sub_area_dummy_dict

sub_area_dict = prepare_sub_area_dummy_dict()

def essential_columns_processing(df):
    df = df.copy()

    # full_sq
    df["full_sq"] = df["full_sq"].fillna(df["full_sq"].mode().values[0])

    # life_sq
    df["life_sq"] = df["life_sq"].fillna(df["life_sq"].mode().values[0])

    # kitch_sq
    df["kitch_sq"] = df["kitch_sq"].fillna(df["kitch_sq"].mode().values[0])

    # num_room
    df["num_room"] = df["num_room"].fillna(df["num_room"].mode().values[0])

    # build_year
    df.loc[df.build_year > 2030, "build_year"] = np.NaN
    df.loc[df.build_year < 1600, "build_year"] = np.NaN
    df["nobuild"] = df.build_year.isnull().astype(int)
    df["transaction_since_build"] = pd.to_datetime(df.timestamp).dt.year - df.build_year
    df.build_year = df.build_year.fillna(df.build_year.mode().values[0])
    df["transaction_since_build"] = df["transaction_since_build"].fillna(df.transaction_since_build.mode().values[0])

    # floor
    df["floor0"] = (df.floor == 0).astype(int)
    df["floor1"] = (df.floor == 1).astype(int)

    # max_floor
    df["nomax_floor"] = df.max_floor.isnull().astype(int)
    df["max_floor"] = df.max_floor.fillna(df.max_floor.mode().values[0])

    # product_type
    df["product_type_OwnerOccupier"] = (df.product_type == "OwnerOccupier").astype(int)

    # state
    df.loc[df.state > 4, "state"] = np.NaN
    df["state"] = df['state'].fillna(0)

    # material
    df["material1"] = (df.material == 1).astype(int)
    df["material2"] = (df.material == 2).astype(int)
    df["material3"] = (df.material == 3).astype(int)
    df["material4"] = (df.material == 4).astype(int)
    df["material5"] = (df.material == 5).astype(int)
    df["material6"] = (df.material == 6).astype(int)

    # sub_area
    df['sub_area_num_indicator'] = df['sub_area'].map(sub_area_dict)

    del df['product_type']
    del df['material']
    del df['sub_area']

    return_essential_columns = ["full_sq", "life_sq", "kitch_sq", "num_room", "build_year", "max_floor", "state",
                                "product_type_OwnerOccupier", "sub_area_num_indicator",
                                "nobuild", "transaction_since_build", "floor0", "floor1", "nomax_floor",
                                "material1", "material2", "material3", "material4", "material5", "material6"]

    return df, return_essential_columns



def prepare_data(train_df, test_df, macro_df):
    train_df = correct_flats_info_df(train_df)
    test_df = correct_flats_info_df(test_df)

    y_price = train_df.price_doc
    del train_df["price_doc"]
    Y_log1p = y_price.apply(np.log1p)

    train_df = essential_columns_processing(train_df)
    test_df = essential_columns_processing(test_df)






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
        dont_touch_cols=dont_touch_cols)

    return train_df_processed, test_df_processed, Y_log1p

if __name__=="__main__":

    train_df = pd.read_csv("../data/train.csv", parse_dates=['timestamp'])
    test_df = pd.read_csv("../data/test.csv", parse_dates=['timestamp'])
    macro_df = pd.read_csv("../data/macro.csv", parse_dates=['timestamp'])

    essential_columns_processing(train_df)

    # train_df_processed, test_df_processed, Y_log1p = prepare_data(train_df, test_df, macro_df)