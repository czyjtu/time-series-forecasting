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
        order: tuple, # (p, d, q)
        seasonal_order: tuple = (0, 0, 0, 0), # (P, D, Q, s), by default without seasonality
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
        
        predictions = self._arima.predict(start=self.start_forecast, end=self.start_forecast + horizon - 1)
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
        custom_seasonality: int | None = None, # in days
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
                fourier_order=10)

        self.is_fitted: bool = False

    @property
    def model(self) -> DeepESN:
        if self._prophet is None:
            raise ValueError("Must call fit before model")
        return self._prophet

    def fit(self, X: pd.DataFrame): # DataFrame must have `y` and `ds` columns
        self._prophet.fit(X)
        self.is_fitted = True

    def forecast(self, horizon: int | pd.DataFrame) -> np.ndarray: # horizon can be a DataFrame with `ds` column
        if not self.is_fitted:
            raise ValueError("Must call fit before forecast")
        
        if isinstance(horizon, int):
            future = self._prophet.make_future_dataframe(periods=horizon, include_history=False)
        else:
            future = horizon
        predictions = self._prophet.predict(future)
        return predictions.yhat.values

    def load(self, path: str):
        raise NotImplementedError
