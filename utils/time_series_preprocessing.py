from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


def parse_decimal(x: str) -> float:
    """
    Utility function to parse string representation of a decimal number into a float representation.

    Args:
        - x: string representation of a decimal number
    Returns:
        - y: float representation of a decimal number
    """
    y = float(x.replace(".", "").replace(",","."))
    return y


def sliding_window_from_dataframe(df: pd.DataFrame, column: str) -> np.ndarray:
    """
    Utility function to create a sliding window for a given column in the dataframe.
    
    Args:
        - df: Pandas dataframe
        - column: column name from the input dataframe

    Returns:
        - sw: sliding window of the input feature
    """
    X = np.array(df[column])
    X = np.insert(X, 0, np.nan)

    y = np.array(df[column])
    y = np.append(y, np.nan)

    sw = np.concatenate([X.reshape(-1,1), y.reshape(-1,1)], axis=1)

    return sw


def walk_forward_validation(model, data: np.ndarray, test_size: float) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Performs walk-forward validation on the time series window data.

    Args:
        - model: time series forecasting model
        - data: sliding window time series data
        - test_size: percentage of the data that should be used as a test subset

    Returns:
        - error: MAE between test (validation) data and predictions
        - y_test: target test (validation) values
        - predictions: predicted test (validation) values
    """
    predictions = list()
    train, test = train_test_split(data, test_size=test_size)
    history = [x for x in train]

    X_test, y_test = test[:, :-1], test[:, -1]
    X_history, y_history = np.array(history)[:, :-1], np.array(history)[:, -1]
    model.fit(X_history, y_history)
    
    for i in range(len(test)):
        y_pred = model.predict(X_test[i])
        predictions.append(y_pred)
        history.append(test[i])
        print(f"Expected = {y_test[i]}, Predicted = {y_pred[0]}")
    
    error = mean_absolute_error(test[:, -1], predictions)
    return error, y_test, predictions


def create_time_features(df: pd.DataFrame, indexing: str="Datetime") -> pd.DataFrame:
    """
    Creates time-related features and adds them to the DataFrame object as columns

    Args:
        - df: Pandas dataframe
        - indexing: Name of a column that sets index to extract time features
    
    Returns:
        - df_new: Pandas dataframe appended with time features
    """
    df_new = df.copy()

    df = df.set_index("Datetime")
    
    df_new["hour"] = df.index.hour
    df_new["dayofweek"] = df.index.dayofweek
    df_new["quarter"] = df.index.quarter
    df_new["month"] = df.index.month
    df_new["year"] = df.index.year
    df_new["dayofyear"] = df.index.dayofyear
    df_new["dayofmonth"] = df.index.day
    df_new["weekofyear"] = pd.Int64Index(df.index.isocalendar().week)
    
    return df_new