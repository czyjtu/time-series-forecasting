from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
import pandas as pd
import numpy as np
from auto_esn.esn.esn import DeepESN
from auto_esn.esn.reservoir.initialization import (
    CompositeInitializer,
    WeightInitializer,
)
import torch as th
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression


class BasePredictor(ABC):
    @abstractproperty
    def model(self):
        pass

    @abstractmethod
    def fit(self, dataset: pd.DataFrame):
        pass

    @abstractmethod
    def forecast(self, horizon: int) -> np.ndarray:
        pass

    @abstractmethod
    def load(self, path: str):
        pass


class ESNPredictor(BasePredictor):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        washout: int = 100,
        seed: int = 42,
    ):
        i = (
            CompositeInitializer()
            .with_seed(seed)
            .uniform()
            .sparse(density=0.1)
            .spectral_normalize()
            .scale(factor=1.0)
        )

        w = WeightInitializer(weight_hh_init=i)
        esn = DeepESN(
            initializer=w,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_dim=output_size,
            washout=washout,
        )
        self._esn = esn

        self.input_size = input_size
        self.output_size = output_size

        # to be set in fit method
        self.X_forecast: th.Tensor | None = None
        self.X_fit: th.Tensor | None = None
        self._need_reset = False

    @property
    def model(self) -> DeepESN:
        return self._esn

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_train = th.from_numpy(X)
        y_train = th.from_numpy(y)
        self._esn.fit(X_train, y_train)

        self.X_forecast = y_train[-1:]
        self.X_fit = X_train.clone()

    def forecast(self, horizon: int) -> np.ndarray:
        if self.X_forecast is None:
            raise ValueError("Must call fit before forecast")

        if self._need_reset:
            # ensure that .forecast is independent of previous calls
            self._reset()
            self._need_reset = False

        predictions = np.empty((horizon, self.output_size))
        for i in range(horizon):
            y_hat = self._esn(self.X_forecast)
            predictions[i] = y_hat.detach().numpy()
            self.X_forecast = y_hat

        self._need_reset = True
        return predictions

    def load(self, path: str):
        raise NotImplementedError

    def _reset(self):
        # DeepESN.reset_hidden() method is broken xd
        self._esn.initial_state = True
        self._esn.reservoir.esn_cell.reset_hidden()
        _ = self._esn(self.X_fit)


class SARIMAPredictor(BasePredictor):
    def __init__(
        self,
        order: tuple,  # (p, d, q)
        seasonal_order: tuple = (
            0,
            0,
            0,
            0,
        ),  # (P, D, Q, s), by default without seasonality
    ):
        self.model_params = {"order": order, "seasonal_order": seasonal_order}

        # to be set in fit method
        self._arima: ARIMA | None = None
        self.start_forecast: int | None = None

    @property
    def model(self) -> DeepESN:
        if self._arima is None:
            raise ValueError("Must call fit before model")
        return self._arima

    def fit(self, X: np.ndarray):
        self._arima = ARIMA(X, **self.model_params).fit()
        self.start_forecast = X.shape[0]

    def forecast(self, horizon: int) -> np.ndarray:
        if self.start_forecast is None:
            raise ValueError("Must call fit before forecast")

        predictions = self._arima.predict(
            start=self.start_forecast, end=self.start_forecast + horizon - 1
        )
        return predictions

    def load(self, path: str):
        raise NotImplementedError


class ProphetPredictor(BasePredictor):
    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10,
        seasonality_mode: str = "additive",
        daily_seasonality: bool = True,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True,
        custom_seasonality: int | None = None,  # in days
    ):
        self._prophet = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            seasonality_mode=seasonality_mode,
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
        )
        if custom_seasonality:
            self._prophet.add_seasonality(
                name=f"{custom_seasonality} days",
                period=custom_seasonality,
                fourier_order=10,
            )

        self.is_fitted: bool = False

    @property
    def model(self) -> DeepESN:
        if self._prophet is None:
            raise ValueError("Must call fit before model")
        return self._prophet

    def fit(self, X: pd.DataFrame):  # DataFrame must have `y` and `ds` columns
        self._prophet.fit(X)
        self.is_fitted = True

    def forecast(
        self, horizon: int | pd.DataFrame
    ) -> np.ndarray:  # horizon can be a DataFrame with `ds` column
        if not self.is_fitted:
            raise ValueError("Must call fit before forecast")

        if isinstance(horizon, int):
            future = self._prophet.make_future_dataframe(
                periods=horizon, include_history=False
            )
        else:
            future = horizon
        predictions = self._prophet.predict(future)
        return predictions.yhat.values

    def load(self, path: str):
        raise NotImplementedError


class XGBoostPredictor(BasePredictor):
    def __init__(
            self,
            n_estimators: int = 100,
            max_depth: int = 5,
            objective: str = "reg:squarederror",
            booster: str = "gbtree",
            include_hours: bool = True,
            lags: dict | None = None
    ):
        self._xgbr = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            objective=objective,
            booster=booster
        )

        # For features design
        self.include_hours = include_hours
        if lags:
            self.lags = lags
        
        self.is_fitted: bool = False

    @property
    def model(self) -> XGBRegressor:
        if self._xgbr is None:
            raise ValueError("Must call fit before model")
        return self._xgbr
    
    def fit(self, X: pd.DataFrame): # DataFrame must have `y` and `ds` columns
        self._create_features_map(X)
        X = self._add_time_features(X)
        X = self._add_lag_features(X)
        X = X.set_index("ds")
        self._xgbr.fit(X.drop("y", axis=1), X.y)
        self.is_fitted = True

    def forecast(self, horizon: int | pd.DataFrame) -> np.ndarray: # horizon can be a DataFrame with `ds` column
        if not self.is_fitted:
            raise ValueError("Must call fit before forecast")
        if isinstance(horizon, int):
            future = None
        else:
            future = horizon
        future = self._add_time_features(future)
        future = self._add_lag_features(future)
        future = future.set_index(future.columns[0])
        predictions = self._xgbr.predict(future)
        return predictions
    
    def load(self, path: str):
        raise NotImplementedError
        
    def _create_features_map(self, X):
        X_indexed = X.set_index("ds")
        self.features_map = X_indexed.y.to_dict()

    def _add_time_features(self, X):
        X_extended = X.copy()

        X = X.set_index("ds")
    
        if self.include_hours is True:
            X_extended["hour"] = X.index.hour
    
        X_extended["dayofweek"] = X.index.dayofweek
        X_extended["quarter"] = X.index.quarter
        X_extended["month"] = X.index.month
        X_extended["year"] = X.index.year
        X_extended["dayofyear"] = X.index.dayofyear
        X_extended["dayofmonth"] = X.index.day
        X_extended["weekofyear"] = pd.Int64Index(X.index.isocalendar().week)
    
        return X_extended
    
    def _add_lag_features(self, X):
        X_extended = X.copy()

        X = X.set_index("ds")

        for lag in self.lags.items():
            X_extended[lag[0]] = (X.index - pd.Timedelta(lag[1])).map(self.features_map)
        
        return X_extended


class LightGBMPredictor(BasePredictor):
    def __init__(
            self,
            n_estimators: int = 100,
            max_depth: int = 5,
            objective: str = "mse",
            boosting_type: str = "gbdt",
            include_hours: bool = True,
            lags: dict | None = None
    ):
        self._lgbmr = LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            objective=objective,
            boosting_type=boosting_type
        )

        # For features design
        self.include_hours = include_hours
        if lags:
            self.lags = lags
        
        self.is_fitted: bool = False

    @property
    def model(self) -> LGBMRegressor:
        if self._xgbr is None:
            raise ValueError("Must call fit before model")
        return self._lgbmr
    
    def fit(self, X: pd.DataFrame): # DataFrame must have `y` and `ds` columns
        self._create_features_map(X)
        X = self._add_time_features(X)
        X = self._add_lag_features(X)
        X = X.set_index("ds")
        self._lgbmr.fit(X.drop("y", axis=1), X.y)
        self.is_fitted = True

    def forecast(self, horizon: int | pd.DataFrame) -> np.ndarray: # horizon can be a DataFrame with `ds` column
        if not self.is_fitted:
            raise ValueError("Must call fit before forecast")
        if isinstance(horizon, int):
            future = None
        else:
            future = horizon
        future = self._add_time_features(future)
        future = self._add_lag_features(future)
        future = future.set_index(future.columns[0])
        predictions = self._lgbmr.predict(future)
        return predictions
    
    def load(self, path: str):
        raise NotImplementedError
        
    def _create_features_map(self, X):
        X_indexed = X.set_index("ds")
        self.features_map = X_indexed.y.to_dict()

    def _add_time_features(self, X):
        X_extended = X.copy()

        X = X.set_index("ds")
    
        if self.include_hours is True:
            X_extended["hour"] = X.index.hour
    
        X_extended["dayofweek"] = X.index.dayofweek
        X_extended["quarter"] = X.index.quarter
        X_extended["month"] = X.index.month
        X_extended["year"] = X.index.year
        X_extended["dayofyear"] = X.index.dayofyear
        X_extended["dayofmonth"] = X.index.day
        X_extended["weekofyear"] = pd.Int64Index(X.index.isocalendar().week)
    
        return X_extended
    
    def _add_lag_features(self, X):
        X_extended = X.copy()

        X = X.set_index("ds")

        for lag in self.lags.items():
            X_extended[lag[0]] = (X.index - pd.Timedelta(lag[1])).map(self.features_map)
        
        return X_extended


class TFTPredictor(BasePredictor):
    def __init__(
        self,
        epochs: int = 300,
        lr: float = 3e-3,
        batch_size: int = 32,
        n_samples: int = 10,
    ):
        self.input_chunk_length = 100
        self.forecast_horizon = 1
        self.n_samples = n_samples

        self._model = TFTModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.forecast_horizon,
            hidden_size=64,
            lstm_layers=1,
            num_attention_heads=4,
            dropout=0.1,
            batch_size=batch_size,
            n_epochs=epochs,
            add_relative_index=True,
            add_encoders=None,
            likelihood=QuantileRegression(),
            random_state=42,
            pl_trainer_kwargs={"accelerator": "cpu"},
            optimizer_kwargs={"lr": lr},
        )

        self._scaler: Scaler | None = None  # to be set in fit method

    @property
    def model(self) -> TFTModel:
        return self._model

    def fit(self, X: pd.DataFrame):
        self._scaler = Scaler()
        time_series = self._prepare_input_series(X)
        self._model.fit(time_series, verbose=True, num_loader_workers=0)

    def forecast(self, horizon: int) -> np.ndarray:
        predictions = self._model.predict(n=horizon, num_samples=self.n_samples)
        return np.array(predictions.values())

    def load(self, path: str) -> TFTPredictor:
        self._model = TFTModel.load(path)
        return self

    def _prepare_input_series(self, df: pd.DataFrame) -> TimeSeries:
        time_series = TimeSeries.from_dataframe(
            df, time_col="ds", value_cols="y", fill_missing_dates=True
        )
        time_series = self._scaler.fit_transform(time_series)
        time_series = MissingValuesFiller().transform(time_series)
        return time_series
