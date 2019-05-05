import os
import pandas as pd
import numpy as np
import tqdm

from configs.columns import flats_choosed_columns, essential_columns
from data_prep.prepare_data import _prepare_train_test_dfs

def get_f_index(f):
    if f not in flats_choosed_columns:
        return 0
    return flats_choosed_columns.index(f) + 1

def get_first_sort_column(row):
    f1_place = row["f1_place"]
    f2_place = row["f2_place"]
    if (f1_place==0) | (f2_place==0):
        return max(f1_place, f2_place)
    else:
        return min(f1_place, f2_place)

def get_second_sort_column(row):
    f1_place = row["f1_place"]
    f2_place = row["f2_place"]
    if (f1_place==0) | (f2_place==0):
        return min(f1_place, f2_place)
    else:
        return max(f1_place, f2_place)


if __name__=="__main__":

    CORR_THRESHOLD = 0.6

    train_df = pd.read_csv("./../data/train.csv", parse_dates=['timestamp'])
    test_df = pd.read_csv("./../data/test.csv", parse_dates=['timestamp'])

    train_df_processed, test_df_processed, Y_log1p = _prepare_train_test_dfs(train_df, test_df)

    train_df_processed_corr = train_df_processed.corr()
    print(train_df_processed_corr.shape)
    train_df_processed_corr = train_df_processed_corr.where(
        ~np.triu(np.ones(train_df_processed_corr.shape)).astype(np.bool)).stack().reset_index()
    train_df_processed_corr.columns = ["feature1", "feature2", "corr_value"]

    df_corr_thresh = train_df_processed_corr[train_df_processed_corr.corr_value.abs() > CORR_THRESHOLD]
    df_corr_thresh_abs = df_corr_thresh.copy()
    df_corr_thresh_abs["corr_value"] = train_df_processed_corr.corr_value.abs()

    df_corr_thresh_abs["f1_place"] = df_corr_thresh_abs["feature1"].apply(get_f_index)
    df_corr_thresh_abs["f2_place"] = df_corr_thresh_abs["feature2"].apply(get_f_index)

    df_corr_thresh_abs["sort_column_1"] = df_corr_thresh_abs.apply(get_first_sort_column, axis=1)
    df_corr_thresh_abs["sort_column_2"] = df_corr_thresh_abs.apply(get_second_sort_column, axis=1)

    df_corr_thresh_abs = df_corr_thresh_abs.sort_values(["sort_column_1", "sort_column_2"], ascending=True)

    drop_features_list = []
    for p in df_corr_thresh_abs[["feature1", "feature2"]].values:
        f1, f2 = p
        print(f1, f2)
        if (f1 in drop_features_list) | (f1 in drop_features_list):
            continue
        elif (f1 in essential_columns) & (f1 in essential_columns):
            continue
        else:
            if (f1 in essential_columns):
                drop_features_list.append(f2)
            elif (f2 in essential_columns):
                drop_features_list.append(f1)
            else:
                f1_num = flats_choosed_columns.index(f1)
                f2_num = flats_choosed_columns.index(f2)
                if (f1_num<f2_num):
                    if f2 not in drop_features_list:
                        drop_features_list.append(f2)
                else:
                    if f1 not in drop_features_list:
                        drop_features_list.append(f1)

    good_features = list(set(flats_choosed_columns) - set(drop_features_list))
    good_features_str = "\n".join(good_features)

    with open("./../configs/good_features_{}.txt".format(str(CORR_THRESHOLD).replace(".", "")), "w") as f:
        f.write(good_features_str)

    print("done!")
