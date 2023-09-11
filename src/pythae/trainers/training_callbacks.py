"""Training Callbacks for training monitoring integrated in `pythae` (inspired from 
https://github.com/huggingface/transformers/blob/master/src/transformers/trainer_callback.py)"""

import importlib
import logging

import numpy as np
from tqdm.auto import tqdm

from ..models import BaseAEConfig
from .base_trainer.base_training_config import BaseTrainerConfig

logger = logging.getLogger(__name__)


def wandb_is_available():
    return importlib.util.find_spec("wandb") is not None


def mlflow_is_available():
    return importlib.util.find_spec("mlflow") is not None


def comet_is_available():
    return importlib.util.find_spec("comet_ml") is not None


def rename_logs(logs):
    train_prefix = "train_"
    eval_prefix = "eval_"

    clean_logs = {}

    for metric_name in logs.keys():
        if metric_name.startswith(train_prefix):
            clean_logs[metric_name.replace(train_prefix, "train/")] = logs[metric_name]

        if metric_name.startswith(eval_prefix):
            clean_logs[metric_name.replace(eval_prefix, "eval/")] = logs[metric_name]

    return clean_logs


class TrainingCallback:
    """
    Base class for creating training callbacks
    """

    def on_init_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """

    def on_train_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the beginning of training.
        """

    def on_train_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of training.
        """

    def on_epoch_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the beginning of an epoch.
        """

    def on_epoch_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of an epoch.
        """

    def on_train_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the beginning of a training step.
        """

    def on_train_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of a training step.
        """

    def on_eval_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the beginning of a evaluation step.
        """

    def on_eval_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of a evaluation step.
        """

    def on_evaluate(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called after an evaluation phase.
        """

    def on_prediction_step(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called after a prediction phase.
        """

    def on_save(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called after a checkpoint save.
        """

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        """
        Event called after logging the last logs.
        """

    def __repr__(self) -> str:
        return self.__class__.__name__


class CallbackHandler:
    """
    Class to handle list of Callback.
    """

    def __init__(self, callbacks, model):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                f"You are adding a {cb_class} to the callbacks but there one is already used."
                f" The current list of callbacks is\n: {self.callback_list}"
            )
        self.callbacks.append(cb)

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def on_init_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_init_end", training_config, **kwargs)

    def on_train_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_train_step_begin", training_config, **kwargs)

    def on_train_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_train_step_end", training_config, **kwargs)

    def on_eval_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_eval_step_begin", training_config, **kwargs)

    def on_eval_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_eval_step_end", training_config, **kwargs)

    def on_train_begin(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_train_begin", training_config, **kwargs)

    def on_train_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_train_end", training_config, **kwargs)

    def on_epoch_begin(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_epoch_begin", training_config, **kwargs)

    def on_epoch_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_epoch_end", training_config)

    def on_evaluate(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_evaluate", **kwargs)

    def on_save(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_save", training_config, **kwargs)

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        self.call_event("on_log", training_config, logs=logs, **kwargs)

    def on_prediction_step(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_prediction_step", training_config, **kwargs)

    def call_event(self, event, training_config, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                training_config,
                model=self.model,
                **kwargs,
            )


class MetricConsolePrinterCallback(TrainingCallback):
    """
    A :class:`TrainingCallback` printing the training logs in the console.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # make it print to the console.
        console = logging.StreamHandler()
        self.logger.addHandler(console)
        self.logger.setLevel(logging.INFO)

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):

        logger = kwargs.pop("logger", self.logger)
        rank = kwargs.pop("rank", -1)

        if logger is not None and (rank == -1 or rank == 0):
            epoch_train_loss = logs.get("train_epoch_loss", None)
            epoch_eval_loss = logs.get("eval_epoch_loss", None)

            logger.info(
                "--------------------------------------------------------------------------"
            )
            if epoch_train_loss is not None:
                logger.info(f"Train loss: {np.round(epoch_train_loss, 4)}")
            if epoch_eval_loss is not None:
                logger.info(f"Eval loss: {np.round(epoch_eval_loss, 4)}")
            logger.info(
                "--------------------------------------------------------------------------"
            )


class TrainHistoryCallback(MetricConsolePrinterCallback):
    def __init__(self):
        self.history = {"train_loss": [], "eval_loss": []}
        super().__init__()

    def on_train_begin(self, training_config: BaseTrainerConfig, **kwargs):
        self.history = {"train_loss": [], "eval_loss": []}

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        logger = kwargs.pop("logger", self.logger)

        if logger is not None:
            epoch_train_loss = logs.get("train_epoch_loss", None)
            epoch_eval_loss = logs.get("eval_epoch_loss", None)
            self.history["train_loss"].append(epoch_train_loss)
            self.history["eval_loss"].append(epoch_eval_loss)


class ProgressBarCallback(TrainingCallback):
    """
    A :class:`TrainingCallback` printing the training progress bar.
    """

    def __init__(self):
        self.train_progress_bar = None
        self.eval_progress_bar = None

    def on_train_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        epoch = kwargs.pop("epoch", None)
        train_loader = kwargs.pop("train_loader", None)
        rank = kwargs.pop("rank", -1)
        if train_loader is not None:
            if rank == 0 or rank == -1:
                self.train_progress_bar = tqdm(
                    total=len(train_loader),
                    unit="batch",
                    desc=f"Training of epoch {epoch}/{training_config.num_epochs}",
                )

    def on_eval_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        epoch = kwargs.pop("epoch", None)
        eval_loader = kwargs.pop("eval_loader", None)
        rank = kwargs.pop("rank", -1)
        if eval_loader is not None:
            if rank == 0 or rank == -1:
                self.eval_progress_bar = tqdm(
                    total=len(eval_loader),
                    unit="batch",
                    desc=f"Eval of epoch {epoch}/{training_config.num_epochs}",
                )

    def on_train_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        if self.train_progress_bar is not None:
            self.train_progress_bar.update(1)

    def on_eval_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        if self.eval_progress_bar is not None:
            self.eval_progress_bar.update(1)

    def on_epoch_end(self, training_config: BaseTrainerConfig, **kwags):
        if self.train_progress_bar is not None:
            self.train_progress_bar.close()

        if self.eval_progress_bar is not None:
            self.eval_progress_bar.close()


class WandbCallback(TrainingCallback):  # pragma: no cover
    """
    A :class:`TrainingCallback` integrating the experiment tracking tool
    `wandb` (https://wandb.ai/).

    It allows users to store their configs, monitor their trainings
    and compare runs through a graphic interface. To be able use this feature you will need:

        - a valid `wandb` account
        - the package `wandb` installed in your virtual env. If not you can install it with

        .. code-block::

            $ pip install wandb

        - to be logged in to your wandb account using

        .. code-block::

            $ wandb login
    """

    def __init__(self):
        if not wandb_is_available():
            raise ModuleNotFoundError(
                "`wandb` package must be installed. Run `pip install wandb`"
            )

        else:
            import wandb

            self._wandb = wandb

    def setup(
        self,
        training_config: BaseTrainerConfig,
        model_config: BaseAEConfig = None,
        project_name: str = "pythae_experiment",
        entity_name: str = None,
        **kwargs,
    ):
        """
        Setup the WandbCallback.

        args:
            training_config (BaseTrainerConfig): The training configuration used in the run.

            model_config (BaseAEConfig): The model configuration used in the run.

            project_name (str): The name of the wandb project to use.

            entity_name (str): The name of the wandb entity to use.
        """

        self.is_initialized = True

        training_config_dict = training_config.to_dict()

        self.run = self._wandb.init(project=project_name, entity=entity_name)

        if model_config is not None:
            model_config_dict = model_config.to_dict()

            self._wandb.config.update(
                {
                    "training_config": training_config_dict,
                    "model_config": model_config_dict,
                }
            )

        else:
            self._wandb.config.update({**training_config_dict})

        self._wandb.define_metric("train/global_step")
        self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

    def on_train_begin(self, training_config: BaseTrainerConfig, **kwargs):
        model_config = kwargs.pop("model_config", None)
        if not self.is_initialized:
            self.setup(training_config, model_config=model_config)

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        global_step = kwargs.pop("global_step", None)
        logs = rename_logs(logs)

        self._wandb.log({**logs, "train/global_step": global_step})

    def on_prediction_step(self, training_config: BaseTrainerConfig, **kwargs):
        kwargs.pop("global_step", None)

        column_names = ["images_id", "truth", "reconstruction", "normal_generation"]

        true_data = kwargs.pop("true_data", None)
        reconstructions = kwargs.pop("reconstructions", None)
        generations = kwargs.pop("generations", None)

        data_to_log = []

        if (
            true_data is not None
            and reconstructions is not None
            and generations is not None
        ):
            for i in range(len(true_data)):

                data_to_log.append(
                    [
                        f"img_{i}",
                        self._wandb.Image(
                            np.moveaxis(true_data[i].cpu().detach().numpy(), 0, -1)
                        ),
                        self._wandb.Image(
                            np.clip(
                                np.moveaxis(
                                    reconstructions[i].cpu().detach().numpy(), 0, -1
                                ),
                                0,
                                255.0,
                            )
                        ),
                        self._wandb.Image(
                            np.clip(
                                np.moveaxis(
                                    generations[i].cpu().detach().numpy(), 0, -1
                                ),
                                0,
                                255.0,
                            )
                        ),
                    ]
                )

            val_table = self._wandb.Table(data=data_to_log, columns=column_names)

            self._wandb.log({"my_val_table": val_table})

    def on_train_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.run.finish()


class MLFlowCallback(TrainingCallback):  # pragma: no cover
    """
    A :class:`TrainingCallback` integrating the experiment tracking tool
    `mlflow` (https://mlflow.org/).

    It allows users to store their configs, monitor their trainings
    and compare runs through a graphic interface. To be able use this feature you will need:

        - the package `mlfow` installed in your virtual env. If not you can install it with

        .. code-block::

            $ pip install mlflow
    """

    def __init__(self):
        if not mlflow_is_available():
            raise ModuleNotFoundError(
                "`mlflow` package must be installed. Run `pip install mlflow`"
            )

        else:
            import mlflow

            self._mlflow = mlflow

    def setup(
        self,
        training_config: BaseTrainerConfig,
        model_config: BaseAEConfig = None,
        run_name: str = None,
        **kwargs,
    ):
        """
        Setup the MLflowCallback.

        args:
            training_config (BaseTrainerConfig): The training configuration used in the run.

            model_config (BaseAEConfig): The model configuration used in the run.

            run_name (str): The name to apply to the current run.
        """
        self.is_initialized = True

        training_config_dict = training_config.to_dict()

        self._mlflow.start_run(run_name=run_name)

        logger.info(
            f"MLflow run started with run_id={self._mlflow.active_run().info.run_id}"
        )
        if model_config is not None:
            model_config_dict = model_config.to_dict()

            self._mlflow.log_params(
                {
                    **training_config_dict,
                    **model_config_dict,
                }
            )

        else:
            self._mlflow.log_params({**training_config_dict})

    def on_train_begin(self, training_config, **kwargs):
        model_config = kwargs.pop("model_config", None)
        if not self.is_initialized:
            self.setup(training_config, model_config=model_config)

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        global_step = kwargs.pop("global_step", None)

        logs = rename_logs(logs)
        metrics = {}
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                metrics[k] = v

        self._mlflow.log_metrics(metrics=metrics, step=global_step)

    def on_train_end(self, training_config: BaseTrainerConfig, **kwargs):
        self._mlflow.end_run()

    def __del__(self):
        # if the previous run is not terminated correctly, the fluent API will
        # not let you start a new run before the previous one is killed
        if (
            callable(getattr(self._mlflow, "active_run", None))
            and self._mlflow.active_run() is not None
        ):
            self._mlflow.end_run()


class CometCallback(TrainingCallback):  # pragma: no cover
    """
    A :class:`TrainingCallback` integrating the experiment tracking tool
    `comet_ml` (https://www.comet.com/site/).

    It allows users to store their configs, monitor
    their trainings and compare runs through a graphic interface. To be able use this feature
    you will need:

    - the package `comet_ml` installed in your virtual env. If not you can install it with

    .. code-block::

        $ pip install comet_ml
    """

    def __init__(self):
        if not comet_is_available():
            raise ModuleNotFoundError(
                "`comet_ml` package must be installed. Run `pip install comet_ml`"
            )

        else:
            import comet_ml

            self._comet_ml = comet_ml

    def setup(
        self,
        training_config: BaseTrainerConfig,
        model_config: BaseTrainerConfig = None,
        api_key: str = None,
        project_name: str = "pythae_experiment",
        workspace: str = None,
        offline_run: bool = False,
        offline_directory: str = "./",
        **kwargs,
    ):

        """
        Setup the CometCallback.

        args:
            training_config (BaseTraineronfig): The training configuration used in the run.

            model_config (BaseAEConfig): The model configuration used in the run.

            api_key (str): Your personal comet-ml `api_key`.

            project_name (str): The name of the wandb project to use.

            workspace (str): The name of your comet-ml workspace

            offline_run: (bool): Whether to run comet-ml in offline mode.

            offline_directory (str): The path to store the offline runs. They can to be
                synchronized then by running `comet upload`.
        """

        self.is_initialized = True

        training_config_dict = training_config.to_dict()

        if not offline_run:
            experiment = self._comet_ml.Experiment(
                api_key=api_key, project_name=project_name, workspace=workspace
            )
            experiment.log_other("Created from", "pythae")
        else:
            experiment = self._comet_ml.OfflineExperiment(
                api_key=api_key,
                project_name=project_name,
                workspace=workspace,
                offline_directory=offline_directory,
            )
            experiment.log_other("Created from", "pythae")

        experiment.log_parameters(training_config, prefix="training_config/")
        experiment.log_parameters(model_config, prefix="model_config/")

    def on_train_begin(self, training_config: BaseTrainerConfig, **kwargs):
        model_config = kwargs.pop("model_config", None)
        if not self.is_initialized:
            self.setup(training_config, model_config=model_config)

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        global_step = kwargs.pop("global_step", None)

        experiment = self._comet_ml.get_global_experiment()
        experiment.log_metrics(logs, step=global_step, epoch=global_step)

    def on_prediction_step(self, training_config: BaseTrainerConfig, **kwargs):
        global_step = kwargs.pop("global_step", None)

        column_names = ["images_id", "truth", "reconstruction", "normal_generation"]

        true_data = kwargs.pop("true_data", None)
        reconstructions = kwargs.pop("reconstructions", None)
        generations = kwargs.pop("generations", None)

        experiment = self._comet_ml.get_global_experiment()

        if (
            true_data is not None
            and reconstructions is not None
            and generations is not None
        ):
            for i in range(len(true_data)):

                experiment.log_image(
                    np.moveaxis(true_data[i].cpu().detach().numpy(), 0, -1),
                    name=f"{i}_truth",
                    step=global_step,
                )
                experiment.log_image(
                    np.clip(
                        np.moveaxis(reconstructions[i].cpu().detach().numpy(), 0, -1),
                        0,
                        255.0,
                    ),
                    name=f"{i}_reconstruction",
                    step=global_step,
                )
                experiment.log_image(
                    np.clip(
                        np.moveaxis(generations[i].cpu().detach().numpy(), 0, -1),
                        0,
                        255.0,
                    ),
                    name=f"{i}_normal_generation",
                    step=global_step,
                )

    def on_train_end(self, training_config: BaseTrainerConfig, **kwargs):
        experiment = self._comet_ml.config.get_global_experiment()
        experiment.end()
