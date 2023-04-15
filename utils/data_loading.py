import pandas as pd
from pathlib import Path
from enum import Enum
from sklearn.preprocessing import MinMaxScaler
import numpy as np


DATA_DIR_PATH = Path(__file__).parents[1] / "data"


class DATASET(str, Enum):
    SUNSPOTS = "sunspots"
    ELECTRICITY = "electricity"
    MACKEY_GLASS = "mackey_glass"
    TEMPERATURE = "temperature"


DATA_PATH = {
    DATASET.SUNSPOTS: DATA_DIR_PATH / "Sunspots.csv",
    DATASET.ELECTRICITY: DATA_DIR_PATH / "electricity.csv",
    DATASET.MACKEY_GLASS: DATA_DIR_PATH / "mg.dat",
    DATASET.TEMPERATURE: DATA_DIR_PATH / "temperature.csv",
}


def load_sunspots(val_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH[DATASET.SUNSPOTS], index_col=0)
    df["ds"] = df.Date.astype("datetime64[m]")
    df["y"] = df["sunspots"]
    df = df[["ds", "y"]].sort_values(["ds"]).reset_index(drop=True)
    train_size = int(df.shape[0] * (1 - val_frac))
    return df.iloc[:train_size], df.iloc[train_size:]


def load_electricity(val_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH[DATASET.ELECTRICITY])
    df["ds"] = [pd.Timestamp(f"{d} {t}") for d, t in zip(df["Date"], df["Time"])]
    df["y"] = df["Consumption Amount (MWh)"].apply(
        lambda x: float(x.replace(".", "").replace(",", "."))
    )
    df = df[df["y"] > 10000]
    df = df[["ds", "y"]].sort_values(["ds"]).reset_index(drop=True)
    train_size = int(df.shape[0] * (1 - val_frac))
    return df.iloc[:train_size], df.iloc[train_size:]


def load_mackey_glass(val_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH[DATASET.MACKEY_GLASS], sep=" ", index_col=0, names=["y"])
    df["ds"] = pd.date_range(start="01-01-2000", periods=df.shape[0], freq="D")
    train_size = int(df.shape[0] * (1 - val_frac))
    return df.iloc[:train_size], df.iloc[train_size:]


def load_temperature(val_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH[DATASET.TEMPERATURE])
    df["ds"] = df.ds.astype("datetime64[m]")
    df = df[["ds", "y"]].sort_values(["ds"]).reset_index(drop=True)
    train_size = int(df.shape[0] * (1 - val_frac))
    return df.iloc[:train_size], df.iloc[train_size:]


class DataLoader:
    LAG = 1

    def __init__(self, dataset: DATASET, val_frac: float = 0.2):
        self.name = dataset
        self.val_frac = val_frac

        if self.name == DATASET.SUNSPOTS:
            self.df_train, self.df_val = load_sunspots(self.val_frac)
        elif self.name == DATASET.ELECTRICITY:
            self.df_train, self.df_val = load_electricity(self.val_frac)
        elif self.name == DATASET.MACKEY_GLASS:
            self.df_train, self.df_val = load_mackey_glass(self.val_frac)
        elif self.name == DATASET.TEMPERATURE:
            self.df_train, self.df_val = load_temperature(self.val_frac)
        else:
            raise ValueError(f"Unknown dataset: {self.name}")

        self._scaler: MinMaxScaler | None = None
        self._y_train_df: pd.DataFrame | None = None
        self._y_val_df: pd.DataFrame | None = None
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._y_val: np.ndarray | None = None

        self._X_train_unscaled: np.ndarray = (
            self.df_train["y"].iloc[: -self.LAG].values.reshape(-1, 1)
        )
        self._y_train_unscaled: np.ndarray = (
            self.df_train["y"].iloc[self.LAG :].values.reshape(-1, 1)
        )
        self._y_val_unscaled: np.ndarray = self.df_val["y"].values.reshape(-1, 1)

    @property
    def train_df(self) -> pd.DataFrame:
        return self.df_train

    @property
    def val_df(self) -> pd.DataFrame:
        return self.df_val

    @property
    def y_train_df(self) -> pd.DataFrame:
        if self._y_train_df is None:
            self._y_train_df = self.df_train.copy()
            self._y_train_df.y = self.scaler.transform(self.df_train.y.values.reshape(-1, 1))
        return self._y_train_df

    @property
    def y_val_df(self) -> pd.DataFrame:
        if self._y_val_df is None:
            self._y_val_df = self.df_val.copy()
            self._y_val_df.y = self.scaler.transform(self.df_val.y.values.reshape(-1, 1))
        return self._y_val_df

    @property
    def scaler(self) -> MinMaxScaler:
        if self._scaler is None:
            self._scaler = MinMaxScaler()
            self._scaler.fit(self._X_train_unscaled)
        return self._scaler

    @property
    def X_train(self) -> np.ndarray:
        if self._X_train is None:
            self._X_train = self.scaler.transform(self._X_train_unscaled)
        return self._X_train

    @property
    def y_train(self) -> np.ndarray:
        if self._y_train is None:
            self._y_train = self.scaler.transform(self._y_train_unscaled)
        return self._y_train

    @property
    def y_val(self) -> np.ndarray:
        if self._y_val is None:
            self._y_val = self.scaler.transform(self._y_val_unscaled)
        return self._y_val
