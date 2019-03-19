import pandas as pd
import numpy as np

def correct_macro_df(macro_df):
    def _replace_incorrect_str(x):
        if x=="#!":
            return np.nan
        return x.replace(",", ".") if pd.notnull(x) else x
    macro_df.child_on_acc_pre_school = macro_df.child_on_acc_pre_school.apply(_replace_incorrect_str).astype(float)
    macro_df.modern_education_share = macro_df.modern_education_share.apply(_replace_incorrect_str).astype(float)
    macro_df.old_education_build_share = macro_df.old_education_build_share.apply(_replace_incorrect_str).astype(float)
    return macro_df

def correct_flats_info_df(df):
    bad_full_sq_index = df[df.full_sq < 5].index
    df.loc[bad_full_sq_index, "full_sq"] = 10
    return df