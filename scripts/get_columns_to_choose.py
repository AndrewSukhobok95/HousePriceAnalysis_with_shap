import pandas as pd
import numpy as np

from data_prep.corrections import correct_macro_df, correct_flats_info_df
from data_prep.custom_data_creating import create_custom_columns, prepare_choosed_features
from data_prep.prepare_data import essential_columns_processing

from configs.columns import drop_columns, essential_columns

if __name__=="__main__":

    train_df = pd.read_csv("./../data/train.csv", parse_dates=['timestamp'])
    test_df = pd.read_csv("./../data/test.csv", parse_dates=['timestamp'])
    macro_df = pd.read_csv("./../data/macro.csv", parse_dates=['timestamp'])

    train_df = correct_flats_info_df(train_df)
    test_df = correct_flats_info_df(test_df)

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
        cols=list(set(train_df.columns) - set(drop_columns)))

    print(sorted(list(set(train_df_processed.columns) - set(essential_columns))))

    print()
