import click
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression
from darts.metrics.metrics import mae
import numpy as np
import torch as th
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.models import TFTModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning import Trainer
from dataclasses import dataclass, asdict
import os
from pathlib import Path
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
import matplotlib.pyplot as plt
from utils import data_loading as dl
from utils import predictors as pred


SCRATCH_DIR = Path(os.environ.get("SCRATCH_LOCAL", "."))


@dataclass
class HParams:
    lr: float
    batch_size: int
    hidden_size: int
    num_heads: int
    full_attention: bool


Trainer()


def score_tft(
    hparams: HParams,
    train: TimeSeries,
    val: TimeSeries,
    fut_covariates: TimeSeries,
    epochs: int,
    input_chunk_length: int,
    output_chunk_length: int,
    dataset_name: str,
):
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    likelihood = QuantileRegression(quantiles=quantiles)
    logger = WandbLogger(log_model=True, save_dir=SCRATCH_DIR)
    pl_kwargs = {
        "accelerator": "cpu",
        "logger": logger,
        "log_every_n_steps": 1,
    }

    model = TFTModel(
        force_reset=True,
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        hidden_size=hparams.hidden_size,
        lstm_layers=1,
        num_attention_heads=hparams.num_heads,
        dropout=0.1,
        batch_size=hparams.batch_size,
        n_epochs=epochs,
        add_relative_index=False,
        add_encoders=None,
        likelihood=likelihood,
        loss_fn=None,
        random_state=42,
        pl_trainer_kwargs=pl_kwargs,
        optimizer_kwargs={"lr": hparams.lr},
        full_attention=hparams.full_attention,
        save_checkpoints=True,
        work_dir=SCRATCH_DIR / "tft_sweep",
        model_name=f"tft_{dataset_name}_{hparams.lr}_{hparams.batch_size}_{hparams.hidden_size}_{hparams.num_heads}_{hparams.full_attention}",
    )
    model.fit(
        train,
        future_covariates=fut_covariates,
        epochs=epochs,
        val_future_covariates=fut_covariates,
        val_series=val,
        verbose=True,
        num_loader_workers=4,
    )
    horizon = len(val)
    pred_series = model.predict(n=horizon, num_samples=30)

    mae_score = mae(pred_series, val)

    # y_hat_test = pred_series.mean().values()
    # y_test = val.values()
    # y_train = train.values()[-input_chunk_length:]
    # xrange_real = np.arange(y_train.shape[0] + y_test.shape[0])
    # xrange_pred = np.arange(y_train.shape[0], y_train.shape[0] + y_test.shape[0])
    # wandb.log(
    #     {
    #         "val_pred": wandb.plot.line_series(
    #             xs=[xrange_real, xrange_pred],
    #             ys=[np.concatenate([y_train, y_test]), y_hat_test],
    #             keys=["true_series", "predicted"],
    #             title="predictions",
    #             xname="ts",
    #         )
    #     }
    # )

    return mae_score


def entrypoint(dataset):
    if dataset == "sunspots":
        data = dl.DataLoader(dl.DATASET.SUNSPOTS, use_test=False)
        input_len, output_len = 400, 133
    elif dataset == "electricity":
        data = dl.DataLoader(dl.DATASET.ELECTRICITY, use_test=False)
        input_len, output_len = 168, 24
    elif dataset == "mackey_glass":
        data = dl.DataLoader(dl.DATASET.MACKEY_GLASS, use_test=False)
        input_len, output_len = 100, 50
    elif dataset == "temperature":
        data = dl.DataLoader(dl.DATASET.TEMPERATURE, use_test=False)
        input_len, output_len = 90, 30
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    covariates_names = {
        dl.DATASET.ELECTRICITY: ["hour", "weekday", "month", "year", "day"],
        dl.DATASET.TEMPERATURE: ["year", "month", "day"],
        dl.DATASET.MACKEY_GLASS: ["year", "month", "day"],
        dl.DATASET.SUNSPOTS: ["year", "month"],
    }[data.name]
    train, val, covariates = pred.get_series_from_dataset(data, covariates_names)

    wandb.init(project="tft_sweep", entity="czyjtu")
    hparams = HParams(
        lr=wandb.config.lr,
        batch_size=wandb.config.batch_size,
        hidden_size=wandb.config.hidden_size,
        num_heads=wandb.config.num_heads,
        full_attention=wandb.config.full_attention,
    )

    score = score_tft(
        hparams,
        train,
        val,
        covariates,
        epochs=10,
        input_chunk_length=input_len,
        output_chunk_length=output_len,
        dataset_name=dataset,
    )
    wandb.log({"mae_score": score, "hparams": asdict(hparams)})


@click.command()
@click.option("--dataset", help="dataset name")
def main(dataset):
    sweep_configuration = {
        "name": f"tft_sweep_{dataset}",
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "mae_score"},
        "parameters": {
            "lr": {"values": [0.1, 0.01, 0.001]},
            "batch_size": {"values": [64, 128, 256]},
            "hidden_size": {"values": [16, 32, 64]},
            "num_heads": {"values": [2, 4, 6]},
            "full_attention": {"values": [True, False]},
        },
    }

    sweep_id = wandb.sweep(sweep_configuration, project="tft_sweep", entity="czyjtu")

    wandb.agent(sweep_id, lambda: entrypoint(dataset), count=15)


if __name__ == "__main__":
    main()
