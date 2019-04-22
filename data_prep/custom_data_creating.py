import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn import model_selection, preprocessing


def add_dates_info(df):
    def correct_str_month(date):
        str_month = str(date.month)
        if len(str_month) == 1:
            return "0" + str_month
        else:
            return str_month

    df = df.copy()
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    df["timestamp_year"] = df["timestamp_dt"].apply(lambda x: int(x.year))
    df["timestamp_year_month"] = df["timestamp_year"].astype(str) + df["timestamp_dt"].apply(correct_str_month)
    return df

def add_price_per_sq_meter(df):
    df = df.copy()
    df["price_per_sq_meter"] = df["price_doc"] / df["full_sq"]
    return df






def create_custom_columns(df):
    df = df.copy()
    df["build_count_before_1945"] = df["build_count_before_1920"] + df["build_count_1921-1945"]
    df["build_count_wood_slag"] = df['build_count_wood'] + df['build_count_slag']
    df['female_male_diff'] = df['female_f'] - df['male_f']
    df['young_female_male_diff'] = df['young_female'] - df['young_male']
    df["edu_culture_service_km"] = df["kindergarten_km"] + df["preschool_km"] + \
                                   df["school_km"] + df["university_km"] + df["additional_education_km"] + \
                                   df["museum_km"] + df["exhibition_km"] + df["theater_km"]
    df["energy_industry_km"] = df['incineration_km'] + df['nuclear_reactor_km'] + \
                               df['radiation_km'] + df['power_transmission_line_km'] + df['thermal_power_plant_km']
    df["fitness_service_km"] = df['fitness_km'] + df['swim_pool_km'] + \
                               df['ice_rink_km'] + df['stadium_km'] + df['basketball_km']
    df['school_preschool_raion'] = df['preschool_education_centers_raion'] + \
                                   df['school_education_centers_raion']
    return df[sorted(df.columns.tolist())]





def prepare_choosed_features(train_df, cols, test_df=None, dont_touch_cols=[]):
    def categorize_column(df, column_to_cat, lblencoder, drop_first_binary_feature=False):
        df = df.copy()
        cat_df = pd.DataFrame(lblencoder.transform(df[column_to_cat].astype(str)))
        cat_df_cols = [column_to_cat + "_" + c for c in lblencoder.classes_]
        if cat_df.shape[1] == 1:
            cat_df_cols = cat_df_cols[0]
            cat_df.columns = [cat_df_cols]
        else:
            cat_df.columns = cat_df_cols

        if drop_first_binary_feature:
            del cat_df[column_to_cat + "_" + lblencoder.classes_[0]]
            cat_df_cols = cat_df_cols[1:]
        del df[column_to_cat]
        return pd.concat([df, cat_df], axis=1), cat_df_cols

    new_cols = []
    for col in cols:
        # print("Processing column:", col)
        if col in dont_touch_cols:
            new_cols.append(col)
        else:
            if is_string_dtype(train_df[col]):
                lbl = preprocessing.LabelBinarizer()
                lbl.fit(train_df[col])
                train_df, cat_columns = categorize_column(df=train_df, column_to_cat=col, lblencoder=lbl)
                new_cols += [cat_columns] if type(cat_columns) == str else cat_columns
                if test_df is not None:
                    test_df, test_cat_columns = categorize_column(df=test_df, column_to_cat=col, lblencoder=lbl)
            else:
                new_cols.append(col)
                if train_df[col].isnull().any():
                    train_df[col] = train_df[col].fillna(train_df[col].median())
    if test_df is not None:
        return train_df[new_cols], test_df[new_cols]
    else:
        return train_df[new_cols]








