import os
import pandas as pd

if __name__=="__main__":

    train_df = pd.read_csv("./../data/train.csv", parse_dates=['timestamp'])

    subarea_text = "#\n".join(pd.unique(train_df.sub_area).tolist())

    with open("../configs/subarea_dict.txt", "w") as f:
        f.write(subarea_text)

    print("done!")