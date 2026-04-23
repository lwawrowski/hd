import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import json

def train_model(n_estimators):

    data = fetch_california_housing()

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test)

    mse = mean_squared_error(
        y_test,
        predictions
    )

    results = pd.DataFrame({
        "y_true": y_test,
        "y_pred": predictions
    })

    results.to_csv(
        f"predictions_{n_estimators}.csv",
        index=False
    )

    metrics = {
        "n_estimators": n_estimators,
        "mse": mse
    }

    with open(
        f"metrics_{n_estimators}.json",
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print("MSE:", mse)

if __name__ == "__main__":

    train_model(100)