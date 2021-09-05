"""
Train.
"""
import os

import pandas as pd
from joblib import dump
import yaml
from sklearn.ensemble import RandomForestClassifier


def train():
    params = yaml.safe_load(open("params.yaml"))["train"]

    df_train = pd.read_csv("resources/out_prepare/train.csv")
    X_train = df_train[params["feature_columns"]]
    y_train = df_train["target"]

    model = RandomForestClassifier(**params["model"]["params"])

    model.fit(X_train, y_train)

    os.makedirs("resources/out_train/", exist_ok=True)
    dump(model, "resources/out_train/model.joblib") 


if __name__ == "__main__":
    train()