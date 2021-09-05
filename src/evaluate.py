import os

import yaml
import numpy as np
import pandas as pd
import joblib
import json
import sklearn.metrics as metrics


def evaluate():
    params = yaml.safe_load(open("params.yaml"))["train"]
    model = joblib.load("resources/out_train/model.joblib")

    df_train = pd.read_csv("resources/out_prepare/train.csv")
    X_train = df_train[params["feature_columns"]]
    y_train = df_train["target"]
    df_test = pd.read_csv("resources/out_prepare/test.csv")
    X_test = df_test[params["feature_columns"]]
    y_test= df_test["target"]

    y_train_pred_prob = model.predict_proba(X_train)
    y_train_pred = np.argmax(y_train_pred_prob, axis=1)
    y_test_pred_prob = model.predict_proba(X_test)
    y_test_pred = np.argmax(y_test_pred_prob, axis=1)

    train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
    train_f1 = metrics.f1_score(y_train, y_train_pred, average="micro")
    test_f1 = metrics.f1_score(y_test, y_test_pred, average="micro")

    os.makedirs("resources/out_evaluate/", exist_ok=True)
    with open("resources/out_evaluate/scores.json", "w") as fd:
        json.dump(
            {
                "train_accuracy": train_accuracy, 
                "test_accuracy": test_accuracy,
                "train_f1": train_f1,
                "test_f1": test_f1,
            }, 
            fd, 
            indent=4,
        )


if __name__ == "__main__":
    evaluate()