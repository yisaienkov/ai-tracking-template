"""
Prepare.
"""
import os

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare():
    params = yaml.safe_load(open("params.yaml"))["prepare"]

    df = pd.read_csv("resources/iris.csv", index_col="Id")

    class_mapping = {k: i for i, k in enumerate(df["Species"].unique().tolist())}
    df["target"] = df["Species"].map(class_mapping)

    df_train, df_test = train_test_split(
        df, test_size=params["split"], random_state=params["seed"], stratify=df["target"]
    )

    os.makedirs("resources/out_prepare/", exist_ok=True)
    df_train.to_csv("resources/out_prepare/train.csv")
    df_test.to_csv("resources/out_prepare/test.csv")


if __name__ == "__main__":
    prepare()


