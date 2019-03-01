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



def get_corr_empty_info(df, cols):
    def _calc_empty(x):
        return str(len(x.dropna())/len(x))
    
    cdf = df[sorted(cols)].sort_index()
    corr_df = cdf.corr().round(2)
    empty_info = pd.DataFrame(cdf[cols].apply(_calc_empty, axis=0), columns=["prct_non_empty"])
    empty_info = pd.merge(empty_info, pd.DataFrame(cdf.dtypes, columns=["col_type"]),
                          how="left", left_index=True, right_index=True)
    
    mask = np.zeros_like(corr_df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(8, 6))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_df, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    
    print("##########################################################")
    print(empty_info)
    print("##########################################################")
    
    return corr_df, empty_info



def correct_macro_df(macro_df):
	def _replace_incorrect_str(x):
		if x=="#!":
			return np.nan
		return x.replace(",", ".") if pd.notnull(x) else x
	macro_df.child_on_acc_pre_school = macro_df.child_on_acc_pre_school.apply(_replace_incorrect_str).astype(float)
	macro_df.modern_education_share = macro_df.modern_education_share.apply(_replace_incorrect_str).astype(float)
	macro_df.old_education_build_share = macro_df.old_education_build_share.apply(_replace_incorrect_str).astype(float)
	return macro_df




def categorize_column(df, column_to_cat, lblencoder, drop_first_binary_feature=False):
    df = df.copy()
    cat_df = pd.DataFrame(lblencoder.transform(df[column_to_cat].astype(str)))
    cat_df_cols = [column_to_cat + "_" + c for c in lblencoder.classes_]
    if cat_df.shape[1]==1:
        cat_df_cols = cat_df_cols[0]
        cat_df.columns = [cat_df_cols]
    else:
        cat_df.columns = cat_df_cols
    
    if drop_first_binary_feature:
        del cat_df[column_to_cat + "_" + lbl.classes_[0]]
        cat_df_cols = cat_df_cols[1:]
    del df[column_to_cat]
    return pd.concat([df, cat_df], axis=1), cat_df_cols

def prepare_choosed_features(train_df, cols, dont_touch_cols=[]):
    new_cols = []
    for col in cols:
        print("Processing column:", col)
        if col in dont_touch_cols:
            new_cols.append(col)
        else:
            if is_string_dtype(train_df[col]):
                lbl = preprocessing.LabelBinarizer()
                lbl.fit(train_df[col])
                train_df, cat_columns = categorize_column(df=train_df, column_to_cat=col, lblencoder=lbl)
                new_cols += [cat_columns] if type(cat_columns)==str else cat_columns
            else:
                new_cols.append(col)
                if train_df[col].isnull().any():
                    train_df[col] = train_df[col].fillna(train_df[col].median())
    return train_df[new_cols]






