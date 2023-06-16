import click
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression
from darts.metrics.metrics import rmse
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
import seaborn as sns

SCRATCH_DIR = Path(os.environ.get("SCRATCH_LOCAL", "."))
MODEL_DIR = Path("./models/sweep2")


@dataclass
class HParams:
    lr: float
    batch_size: int
    hidden_size: int
    num_heads: int
    full_attention: bool
    input_chunk_length: int
    output_chunk_length: int


def get_model(
    hparams: HParams,
    epochs: int,
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
        input_chunk_length=hparams.input_chunk_length,
        output_chunk_length=hparams.output_chunk_length,
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
    return model


def score_tft(
    hparams: HParams,
    train: TimeSeries,
    val: TimeSeries,
    fut_covariates: TimeSeries,
    epochs: int,
    dataset_name: str,
):
    model = get_model(
        hparams,
        epochs,
        dataset_name,
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
    pred_series = model.predict(n=horizon, num_samples=100)

    rmse_score = rmse(pred_series, val)
    # plot_predictions(
    #     train.values(),
    #     val.values(),
    #     pred_series.mean().values(),
    #     "predictions",
    #     f"rmse={rmse_score}",
    #     save=False,
    # )
    # wandb.log({"val_plot": plt})

    return rmse_score


def get_series_and_covariates(dataset, use_test=False):
    data = dl.DataLoader(dataset, use_test=use_test)

    covariates_names = {
        dl.DATASET.ELECTRICITY: ["hour", "weekday", "month", "year", "day"],
        dl.DATASET.TEMPERATURE: ["year", "month", "day"],
        dl.DATASET.MACKEY_GLASS: ["year", "month", "day"],
        dl.DATASET.SUNSPOTS: ["year", "month"],
    }[data.name]
    train, val, covariates = pred.get_series_from_dataset(data, covariates_names)

    return train, val, covariates


INPUTS_DIMENSIONS = {
    dl.DATASET.SUNSPOTS: (400, 133),
    dl.DATASET.ELECTRICITY: (168, 24),
    dl.DATASET.MACKEY_GLASS: (100, 50),
    dl.DATASET.TEMPERATURE: (90, 30),
}


def entrypoint(dataset):
    wandb.init(project="tft_sweep", entity="czyjtu")
    train, val, covariates = get_series_and_covariates(dataset, False)
    input_chunk_length, output_chunk_length = INPUTS_DIMENSIONS[dataset]
    print(
        "validation length",
        len(val),
        "input_chunk_length",
        wandb.config.input_chunk_length,
    )

    hparams = HParams(
        lr=wandb.config.lr,
        batch_size=wandb.config.batch_size,
        hidden_size=wandb.config.hidden_size,
        num_heads=wandb.config.num_heads,
        full_attention=wandb.config.full_attention,
        input_chunk_length=wandb.config.input_chunk_length,
        output_chunk_length=output_chunk_length,
    )

    score = score_tft(
        hparams,
        train,
        val,
        covariates,
        epochs=20,
        dataset_name=dataset,
    )
    wandb.log({"rmse_score": score, "hparams": asdict(hparams)})


def plot_predictions(y_train, y_test, y_hat_test, plot_name, title=None, save=True):
    xrange_train = np.arange(y_train.shape[0])
    xrange_test = np.arange(y_train.shape[0], y_train.shape[0] + y_test.shape[0])
    sns.lineplot(y=y_train, x=xrange_train, label="train")
    sns.lineplot(y=y_test, x=xrange_test, label="test")
    if y_hat_test is not None:
        sns.lineplot(y=y_hat_test, x=xrange_test, label="prediction")
    # plot vertical dotted line
    plt.axvline(x=y_train.shape[0], linestyle="--", color="black")
    if title:
        plt.title(title)
    plt.legend()
    if save:
        plt.savefig(f"{plot_name}.png")


@click.command()
@click.option("--dataset", help="dataset name")
@click.option("--sweep_id", required=False)
def main(dataset, sweep_id):
    dataset = {
        "s": dl.DATASET.SUNSPOTS,
        "t": dl.DATASET.TEMPERATURE,
        "e": dl.DATASET.ELECTRICITY,
        "m": dl.DATASET.MACKEY_GLASS,
    }[dataset]

    if sweep_id is None:
        input_chunk_length, output_chunk_length = INPUTS_DIMENSIONS[dataset]
        train, val, covariates = get_series_and_covariates(dataset, False)

        sweep_configuration = {
            "name": f"tft_sweep_{dataset}",
            "method": "bayes",
            "metric": {"goal": "minimize", "name": "rmse_score"},
            "parameters": {
                "lr": {"values": [0.1, 0.01, 0.001]},
                "batch_size": {"values": [64, 128, 256]},
                "hidden_size": {"values": [16, 32, 64]},
                "num_heads": {"values": [2, 4, 6]},
                "full_attention": {"values": [True, False]},
                "input_chunk_length": {
                    "values": [
                        int(output_chunk_length * i)
                        for i in [2, 2.5, 3, 3.5, 4, 4.5, 5]
                        if int(output_chunk_length * i) + output_chunk_length
                        <= len(val)
                    ]
                },
            },
        }

        sweep_id = wandb.sweep(
            sweep_configuration, project="tft_sweep", entity="czyjtu"
        )
        wandb.agent(sweep_id, lambda: entrypoint(dataset), count=40)
    else:
        # get best parameters from sweep, train model and save it
        api = wandb.Api()
        sweep = api.sweep(sweep_id)

        input_chunk_length, output_chunk_length = INPUTS_DIMENSIONS[dataset]
        # Get best run parameters
        best_run = sweep.best_run()
        best_parameters = best_run.config
        print(best_parameters)
        print(best_run.id)
        print(best_run.name)
        hparams = HParams(
            lr=best_parameters["lr"],
            batch_size=best_parameters["batch_size"],
            hidden_size=best_parameters["hidden_size"],
            num_heads=best_parameters["num_heads"],
            full_attention=best_parameters["full_attention"],
            input_chunk_length=best_parameters["input_chunk_length"],
            output_chunk_length=output_chunk_length,
        )
        train, val, covariates = get_series_and_covariates(dataset, True)
        print(
            input_chunk_length,
            output_chunk_length,
            len(train),
            len(val),
            len(covariates),
        )
        epochs = 20

        model = get_model(
            hparams,
            epochs,
            dataset.name,
        )

        model.fit(
            train,
            future_covariates=covariates,
            epochs=epochs,
            val_future_covariates=covariates,
            # val_series=val,
            verbose=True,
            num_loader_workers=4,
        )
        model.save(str(MODEL_DIR / f"{model.model_name}"))
        horizon = len(val)
        pred_series = model.predict(
            n=len(val),
            num_samples=100,
            trainer=Trainer(accelerator="cpu", precision="64"),
            series=train,
            future_covariates=covariates,
        )

        MAX_PLOT_HORIZON = 1000
        MAX_PLOT_TRAIN = 100
        rmse_score = rmse(val, pred_series)
        plot_predictions(
            train.values()[-MAX_PLOT_TRAIN:].flatten(),
            val.values()[:MAX_PLOT_HORIZON].flatten(),
            pred_series.mean().values()[:MAX_PLOT_HORIZON].flatten(),
            dataset.name,
            title=f"rmse={rmse_score}",
        )


if __name__ == "__main__":
    main()
