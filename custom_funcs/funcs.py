import pandas as pd
import numpy as np

def manual_processing(df):
    df = df.copy()

    df["material1"] = (df.material == 1).astype(int)
    df["material2"] = (df.material == 2).astype(int)
    df["material3"] = (df.material == 3).astype(int)
    df["material4"] = (df.material == 4).astype(int)
    df["material5"] = (df.material == 5).astype(int)
    df["material6"] = (df.material == 6).astype(int)

    df.loc[df.build_year > 2030, "build_year"] = np.NaN
    df.loc[df.build_year < 1600, "build_year"] = np.NaN
    df["nobuild"] = df.build_year.isnull().astype(int)
    df["transaction_since_build"] = pd.to_datetime(df.timestamp).dt.year - df.build_year
    df.build_year = df.build_year.fillna(df.build_year.mode().values[0])
    df.transaction_since_build = df.transaction_since_build.fillna(df.transaction_since_build.mode())

    df["floor0"] = (df.floor == 0).astype(int)
    df["floor1"] = (df.floor == 1).astype(int)
    # df["floorhuge"] = (df.floor > 40).astype(int)
    # df["lnfloor"] = np.log(df.floor+1)

    df["nomax_floor"] = df.max_floor.isnull().astype(int)
    df.max_floor = df.max_floor.fillna(df.max_floor.median())
    # df["max0"] = (df.max_floor == 0).astype(int)
    # df["max1"] = (df.max_floor == 1).astype(int)
    # df["maxhuge"] = (df.max_floor > 80).astype(int)
    # df["lnmax"] = np.log(df.max_floor+1)

    del df["material"]
    del df["timestamp"]

    return df
