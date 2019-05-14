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

from data_prep.utils import prepare_sub_area_dummy_dict, prepare_dict_build_year_from_sub_area,\
    prepare_dict_max_floor_from_build_year, prepare_linear_model_for_life_sq,\
    prepare_linear_model_for_kitch_sq

if __name__=="__main__":

    train_df = pd.read_csv("../data/train.csv", parse_dates=['timestamp'])
    test_df = pd.read_csv("../data/test.csv", parse_dates=['timestamp'])

    train_df['full_sq'].ix[train_df['full_sq']>300] = 300

    train_df.loc[train_df.build_year > 2030, "build_year"] = np.NaN
    train_df.loc[train_df.build_year < 1600, "build_year"] = np.NaN


    ##### build_year #####

    build_year_sub_area_df_plot_raw = train_df.loc[
        train_df["build_year"]>1800, ["sub_area", "build_year"]].dropna().copy()

    fig = plt.figure(figsize=(50, 15), clear=True)
    sns.boxplot(x="sub_area", y="build_year", data=build_year_sub_area_df_plot_raw)
    plt.xticks(fontsize=14, rotation=90)
    plt.yticks(fontsize=18)

    plt.xlabel("sub_area", fontsize=18)
    plt.ylabel("build_year", fontsize=18)

    plt.subplots_adjust(bottom=0.4)
    # plt.show()
    fig.savefig('./../imgs/feature_prep_additional/build_year_from_sub_area_full.png', dpi=fig.dpi)

    ##################################

    build_year_sub_area_df_plot_part_raw = build_year_sub_area_df_plot_raw[
        build_year_sub_area_df_plot_raw["sub_area"].isin([
            'Hamovniki', 'Lianozovo', 'Poselenie Voskresenskoe',
            'Severnoe Butovo', 'Filevskij Park', 'Nekrasovka',
            'Juzhnoe Medvedkovo', 'Poselenie Pervomajskoe', 'Solncevo',
            'Ajeroport', 'Orehovo-Borisovo Juzhnoe', 'Nagornoe', "Mar'ino",
            'Strogino', 'Chertanovo Severnoe', 'Shhukino'
        ])
    ]

    fig = plt.figure(figsize=(20, 15), clear=True)
    sns.boxplot(x="sub_area", y="build_year", data=build_year_sub_area_df_plot_part_raw)
    plt.xticks(fontsize=14, rotation=90)
    plt.yticks(fontsize=18)

    plt.xlabel("sub_area", fontsize=18)
    plt.ylabel("build_year", fontsize=18)

    plt.subplots_adjust(bottom=0.4)
    # plt.show()
    fig.savefig('./../imgs/feature_prep_additional/build_year_from_sub_area_part.png', dpi=fig.dpi)




    ##### max_floor #####

    build_year_from_sub_area_dict = prepare_dict_build_year_from_sub_area()
    train_df["build_year"] = train_df["build_year"].fillna(train_df["sub_area"].map(build_year_from_sub_area_dict))

    max_floor_build_year_df_plot_raw = train_df.loc[train_df["max_floor"]<60, ["max_floor", "build_year"]].dropna().copy()
    max_floor_build_year_df_plot_raw["build_year"] = max_floor_build_year_df_plot_raw["build_year"].astype(int)

    fig = plt.figure(figsize=(50, 15), clear=True)
    sns.boxplot(x="build_year", y="max_floor", data=max_floor_build_year_df_plot_raw)
    plt.xticks(fontsize=18, rotation=90)
    plt.yticks(fontsize=18)

    plt.xlabel("build_year", fontsize=20)
    plt.ylabel("max_floor", fontsize=20)

    plt.subplots_adjust(bottom=0.2)
    # plt.show()
    fig.savefig('./../imgs/feature_prep_additional/max_floor_from_build_year_full.png', dpi=fig.dpi)



    print("done1")
