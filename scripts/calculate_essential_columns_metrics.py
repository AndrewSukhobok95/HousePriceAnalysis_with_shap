import pandas as pd


if __name__=="__main__":

    train_df = pd.read_csv("./../data/train.csv", parse_dates=['timestamp'])

    full_life_sq_df  = train_df[["full_sq", "life_sq"]].dropna()

    full_life_sq_df["life_prct"] = full_life_sq_df.life_sq / full_life_sq_df.full_sq

    print("done!")

