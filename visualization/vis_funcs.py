import numpy as np # linear algebra
import pandas as pd

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
color = sns.color_palette()
sns.set()

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#init_notebook_mode(connected=True)

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)



def get_corr_empty_info(df, cols, figsize=(8, 6)):
    def _calc_empty(x):
        return str(len(x.dropna())/len(x))
    
    cdf = df[sorted(cols)].sort_index()
    corr_df = cdf.corr().round(2)
    empty_info = pd.DataFrame(cdf[cols].apply(_calc_empty, axis=0), columns=["prct_non_empty"])
    empty_info = pd.merge(empty_info, pd.DataFrame(cdf.dtypes, columns=["col_type"]),
                          how="left", left_index=True, right_index=True)
    
    mask = np.zeros_like(corr_df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=figsize)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_df, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    
    print("##########################################################")
    print(empty_info)
    print("##########################################################")
    
    return corr_df, empty_info













