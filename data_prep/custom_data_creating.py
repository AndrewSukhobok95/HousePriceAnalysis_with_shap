import pandas as pd
import numpy as np

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






