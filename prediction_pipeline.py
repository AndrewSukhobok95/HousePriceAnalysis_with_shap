import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn import model_selection, preprocessing
import xgboost as xgb

from data_prep.corrections import correct_macro_df, correct_flats_info_df
from data_prep.custom_data_creating import add_dates_info

# import matplotlib.pyplot as plt
# import seaborn as sns
# color = sns.color_palette()
# sns.set()
# import plotly.plotly as py
# import plotly.graph_objs as go
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# pd.options.mode.chained_assignment = None # default='warn'
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.max_rows', 500)


train_df = pd.read_csv("data/train.csv", parse_dates=['timestamp'])
test_df = pd.read_csv("data/test.csv", parse_dates=['timestamp'])

train_df = correct_flats_info_df(train_df)
test_df = correct_flats_info_df(test_df)

macro_df = pd.read_csv("data/macro.csv", parse_dates=['timestamp'])
macro_df = correct_macro_df(macro_df=macro_df)
macro_df.columns = ["timestamp"] + ["macro_" + c for c in macro_df.columns if c!="timestamp"]

train_with_macro_df = pd.merge(train_df, macro_df, how='left', on='timestamp')
test_with_macro_df = pd.merge(test_df, macro_df, how='left', on='timestamp')

train_with_macro_df = add_dates_info(train_with_macro_df)
test_with_macro_df = add_dates_info(test_with_macro_df)










