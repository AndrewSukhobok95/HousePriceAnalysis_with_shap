import os
import numpy as np
import pandas as pd

from data_prep.corrections import correct_macro_df, correct_flats_info_df
from data_prep.custom_data_creating import add_dates_info, prepare_choosed_features, create_custom_columns
from data_prep.utils import prepare_sub_area_dummy_dict, prepare_dict_build_year_from_sub_area,\
    prepare_dict_max_floor_from_build_year, prepare_linear_model_for_life_sq,\
    prepare_linear_model_for_kitch_sq

from configs.columns import flats_choosed_columns, drop_columns, essential_columns

sub_area_dict = prepare_sub_area_dummy_dict()
build_year_from_sub_area_dict = prepare_dict_build_year_from_sub_area()
max_floor_from_build_year_dict = prepare_dict_max_floor_from_build_year()
reg_model_life_sq = prepare_linear_model_for_life_sq()
reg_model_kitch_sq = prepare_linear_model_for_kitch_sq()

def calc_full_sq_correction(x):
    if x < 0:
        return x/2
    else:
        return 0

def essential_columns_processing(df):
    df = df.copy()

    # full_sq
    df['full_sq'].ix[df['full_sq'] > 300] = 300
    df["full_sq"] = df["full_sq"].fillna(df["full_sq"].mode().values[0])

    # life_sq
    df.loc[df["life_sq"] > 500, "life_sq"] = np.NaN
    df.loc[df["life_sq"] > df["full_sq"], "life_sq"] = np.NaN
    life_sq_pred = reg_model_kitch_sq.predict(df[["full_sq"]])
    df["life_sq_pred"] = life_sq_pred
    df["life_sq"] = df["life_sq"].fillna(df["life_sq_pred"])

    # kitch_sq
    df.loc[df["kitch_sq"] > 250, "kitch_sq"] = np.NaN
    df.loc[df["kitch_sq"] > df["full_sq"], "kitch_sq"] = np.NaN
    kitch_sq_pred = reg_model_life_sq.predict(df[["full_sq"]])
    df["kitch_sq_pred"] = kitch_sq_pred
    df["kitch_sq"] = df["kitch_sq"].fillna(df["kitch_sq_pred"])

    # full_sq correction
    df["full_sq_diff"] = df["full_sq"] - df["kitch_sq"] - df["life_sq"]
    df["life_and_kitch_sq_correction"] = df["full_sq_diff"].apply(calc_full_sq_correction)
    df["life_sq"] = df["life_sq"] + df["life_and_kitch_sq_correction"]
    df["kitch_sq"] = df["kitch_sq"] + df["life_and_kitch_sq_correction"]

    # num_room
    df["num_room"] = df["num_room"].fillna(df["num_room"].mode().values[0])

    # build_year
    df.loc[df.build_year > 2030, "build_year"] = np.NaN
    df.loc[df.build_year < 1600, "build_year"] = np.NaN
    df["nobuild"] = df.build_year.isnull().astype(int)
    df["build_year"] = df["build_year"].fillna(df["sub_area"].map(build_year_from_sub_area_dict))

    # floor and max_floor
    # floor
    df["floor0"] = (df.floor == 0).astype(int)
    df["floor1"] = (df.floor == 1).astype(int)
    df.loc[df["floor"]>df["max_floor"], "floor"] = -100
    # max_floor
    df.loc[df["max_floor"] > 60, "max_floor"] = 60
    df["nomax_floor"] = df.max_floor.isnull().astype(int)
    df["max_floor"] = df["max_floor"].fillna(df["build_year"].map(max_floor_from_build_year_dict))
    # floor
    df["floor"] = df["floor"].fillna(df["max_floor"] // 2)
    df.loc[df["floor"] == -100, "floor"] = np.NaN
    df["floor"] = df["floor"].fillna(df["max_floor"])

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
    del df['life_sq_pred']
    del df['kitch_sq_pred']
    del df['full_sq_diff']
    del df['life_and_kitch_sq_correction']

    return df



def _prepare_train_test_dfs(train_df, test_df):
    train_df = correct_flats_info_df(train_df)
    test_df = correct_flats_info_df(test_df)

    # ulimit_price_doc = np.percentile(train_df.price_doc.values, 99)
    # llimit_price_doc = np.percentile(train_df.price_doc.values, 1)
    # train_df['price_doc'].ix[train_df['price_doc']>ulimit_price_doc] = ulimit_price_doc
    # train_df['price_doc'].ix[train_df['price_doc']<llimit_price_doc] = llimit_price_doc

    y_price = train_df.price_doc
    del train_df["price_doc"]
    Y_log1p = y_price.apply(np.log1p)

    train_df = essential_columns_processing(train_df)
    test_df = essential_columns_processing(test_df)

    train_df = create_custom_columns(train_df)
    test_df = create_custom_columns(test_df)

    train_df_processed, test_df_processed = prepare_choosed_features(
        train_df=train_df,
        test_df=test_df,
        cols=["timestamp"]+list(set(train_df.columns) - set(drop_columns)),
        dont_touch_cols=["timestamp"]
    )

    return train_df_processed, test_df_processed, Y_log1p


def prepare_data(train_df, test_df, macro_df):
    train_df_processed, test_df_processed, Y_log1p = _prepare_train_test_dfs(train_df, test_df)

    macro_df = correct_macro_df(macro_df=macro_df)
    macro_df.columns = ["timestamp"] + ["macro_" + c for c in macro_df.columns if c!="timestamp"]

    train_with_macro_df = pd.merge(train_df_processed, macro_df, how='left', on='timestamp')
    test_with_macro_df = pd.merge(test_df_processed, macro_df, how='left', on='timestamp')

    train_with_macro_df = add_dates_info(train_with_macro_df)
    test_with_macro_df = add_dates_info(test_with_macro_df)

    return train_with_macro_df, test_with_macro_df, Y_log1p



if __name__=="__main__":

    train_df = pd.read_csv("../data/train.csv", parse_dates=['timestamp'])
    test_df = pd.read_csv("../data/test.csv", parse_dates=['timestamp'])
    macro_df = pd.read_csv("../data/macro.csv", parse_dates=['timestamp'])

    # df, return_essential_columns = essential_columns_processing(train_df)

    train_df_processed, test_df_processed, Y_log1p = prepare_data(train_df, test_df, macro_df)

    print()
