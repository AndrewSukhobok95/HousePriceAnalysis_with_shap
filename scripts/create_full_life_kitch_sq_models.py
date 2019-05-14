import os
import numpy as np
import pandas as pd
import pickle
import json

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()
sns.set()



if __name__=="__main__":

    train_df = pd.read_csv("../data/train.csv", parse_dates=['timestamp'])
    test_df = pd.read_csv("../data/test.csv", parse_dates=['timestamp'])

    train_df['full_sq'].ix[train_df['full_sq']>300] = 300

    train_df.loc[train_df.build_year > 2030, "build_year"] = np.NaN
    train_df.loc[train_df.build_year < 1600, "build_year"] = np.NaN

    ##### life_sq #####

    life_sq_df_plot_raw = train_df.loc[train_df["life_sq"]<=500, ["full_sq", "life_sq"]].dropna().copy()
    fig = plt.figure()
    plt.scatter(
        life_sq_df_plot_raw["full_sq"],
        life_sq_df_plot_raw["life_sq"],
        color='black'
    )
    plt.xlabel("full_sq")
    plt.ylabel("life_sq")
    # plt.show()x   
    fig.savefig('./../imgs/features_prep/raw_full_sq_vs_life_sq.png', dpi=fig.dpi)

    cond1_life_sq = train_df["life_sq"] < 0.9 * train_df["full_sq"]
    cond2_life_sq = train_df["life_sq"] > 10
    cond3_life_sq = train_df["life_sq"].notna()
    cond4_life_sq = train_df["full_sq"] < 300
    good_life_sq_index = train_df[cond1_life_sq & cond2_life_sq & cond3_life_sq & cond4_life_sq].index

    life_sq_df = train_df.loc[good_life_sq_index, ["full_sq", "life_sq"]].copy()

    x_life_sq = life_sq_df[["full_sq"]]
    y_life_sq = life_sq_df[["life_sq"]]

    reg_life_sq = LinearRegression()
    reg_life_sq.fit(x_life_sq, y_life_sq)
    y_life_sq_pred = reg_life_sq.predict(x_life_sq)


    fig = plt.figure()
    plt.scatter(
        life_sq_df_plot_raw["full_sq"],
        life_sq_df_plot_raw["life_sq"],
        color='gray'
    )
    plt.scatter(
        x_life_sq,
        y_life_sq,
        color='black'
    )
    plt.plot(
        x_life_sq,
        y_life_sq_pred,
        color='blue',
        linewidth=3
    )
    plt.xlabel("full_sq")
    plt.ylabel("life_sq")
    # plt.show()
    fig.savefig('./../imgs/features_prep/reg_full_sq_vs_life_sq.png', dpi=fig.dpi)

    ####################

    ##### kitch_sq #####

    kitch_sq_df_plot_raw = train_df.loc[train_df["kitch_sq"]<=250, ["full_sq", "kitch_sq"]].dropna().copy()
    fig = plt.figure()
    plt.scatter(
        kitch_sq_df_plot_raw["full_sq"],
        kitch_sq_df_plot_raw["kitch_sq"],
        color='black'
    )
    plt.xlabel("full_sq")
    plt.ylabel("kitch_sq")
    # plt.show()
    fig.savefig('./../imgs/features_prep/raw_full_sq_vs_kitch_sq.png', dpi=fig.dpi)

    cond1_kitch_sq = train_df["kitch_sq"] < 0.9 * train_df["full_sq"]
    cond2_kitch_sq = train_df["kitch_sq"] > 1
    cond3_kitch_sq = train_df["kitch_sq"].notna()
    cond4_kitch_sq = train_df["full_sq"] < 300
    good_kitch_sq_index = train_df[cond1_kitch_sq & cond2_kitch_sq & cond3_kitch_sq & cond4_kitch_sq].index

    kitch_sq_df = train_df.loc[good_kitch_sq_index, ["full_sq", "kitch_sq"]].copy()

    x_kitch_sq = kitch_sq_df[["full_sq"]]
    y_kitch_sq = kitch_sq_df[["kitch_sq"]]

    reg_kitch_sq = LinearRegression()
    reg_kitch_sq.fit(x_kitch_sq, y_kitch_sq)
    y_kitch_sq_pred = reg_kitch_sq.predict(x_kitch_sq)

    fig = plt.figure()
    plt.scatter(
        kitch_sq_df_plot_raw["full_sq"],
        kitch_sq_df_plot_raw["kitch_sq"],
        color='gray'
    )
    plt.scatter(
        x_kitch_sq,
        y_kitch_sq,
        color='black'
    )
    plt.plot(
        x_kitch_sq,
        y_kitch_sq_pred,
        color='blue',
        linewidth=3
    )
    plt.xlabel("full_sq")
    plt.ylabel("kitch_sq")
    # plt.show()
    fig.savefig('./../imgs/features_prep/reg_full_sq_vs_kitch_sq.png', dpi=fig.dpi)

    pickle.dump(reg_life_sq, open("./../trained_models/data_prep_models/" + "reg_life_sq" + ".dat", "wb"))
    pickle.dump(reg_kitch_sq, open("./../trained_models/data_prep_models/" + "reg_kitch_sq" + ".dat", "wb"))

    ####################

    ##### build_year #####

    build_year_sub_area_df_plot_raw = train_df.loc[
        train_df["build_year"]>1800, ["sub_area", "build_year"]].dropna().copy()

    fig = plt.figure(figsize=(50, 15))
    sns.boxplot(x="sub_area", y="build_year", data=build_year_sub_area_df_plot_raw)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.2)
    fig.savefig('./../imgs/features_prep/build_year_from_sub_area.png', dpi=fig.dpi)

    def get_mode_build_year(gdf):
        return int(gdf[["build_year"]].mode().values[0][0])

    build_year_from_sub_area_dict = build_year_sub_area_df_plot_raw.groupby(
        "sub_area").apply(get_mode_build_year).to_dict()

    with open("./../configs/build_year_from_sub_area_dict.json", "w") as write_file:
        json.dump(build_year_from_sub_area_dict, write_file)

    ####################

    ##### max_floor #####

    train_df["build_year"] = train_df["build_year"].fillna(train_df["sub_area"].map(build_year_from_sub_area_dict))

    max_floor_build_year_df_plot_raw = train_df.loc[train_df["max_floor"]<60, ["max_floor", "build_year"]].dropna().copy()
    max_floor_build_year_df_plot_raw["build_year"] = max_floor_build_year_df_plot_raw["build_year"].astype(int)

    fig = plt.figure(figsize=(50, 15))
    sns.boxplot(x="build_year", y="max_floor", data=max_floor_build_year_df_plot_raw)
    plt.xticks(rotation=45)
    # plt.subplots_adjust(bottom=0.2)
    fig.savefig('./../imgs/features_prep/max_floor_from_build_year.png', dpi=fig.dpi)

    def get_mode_max_floor(gdf):
        return int(gdf[["max_floor"]].mode().values[0][0])

    max_floor_from_build_year_dict = max_floor_build_year_df_plot_raw.groupby(
        "build_year").apply(get_mode_max_floor).to_dict()

    with open("./../configs/max_floor_from_build_year_dict.json", "w") as write_file:
        json.dump(max_floor_from_build_year_dict, write_file)


    print("done!")
