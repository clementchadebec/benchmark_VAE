"""Training Callbacks for training monitoring (inspired from 
https://github.com/huggingface/transformers/blob/master/src/transformers/trainer_callback.py)"""

import importlib
import logging

import numpy as np
from tqdm import tqdm

from .base_trainer.base_training_config import BaseTrainerConfig

logger = logging.getLogger(__name__)


def wandb_is_available():
    return importlib.util.find_spec("wandb") is not None


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
    Base class for creating training callbacks"""

    def on_init_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
        pass

    def on_train_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the beginning of training.
        """
        pass

    def on_train_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of training.
        """
        pass

    def on_epoch_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        pass

    def on_epoch_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of an epoch.
        """
        pass

    def on_train_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the beginning of a training step.
        """
        pass

    def on_train_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of a training step.
        """
        pass

    def on_eval_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the beginning of a evaluation step.
        """
        pass

    def on_eval_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of a evaluation step.
        """
        pass

    def on_evaluate(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called after an evaluation phase.
        """
        pass

    def on_prediction_step(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called after a prediction phase.
        """
        pass

    def on_save(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called after a checkpoint save.
        """
        pass

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        """
        Event called after logging the last logs.
        """
        pass


class CallbackHandler:
    """
    Class to handle list of Callback
    """

    def __init__(self, callbacks, model, optimizer, scheduler):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

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
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                **kwargs,
            )


class MetricConsolePrinterCallback(TrainingCallback):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # make it print to the console.
        console = logging.StreamHandler()
        self.logger.addHandler(console)
        self.logger.setLevel(logging.INFO)

    def on_log(self, training_config, logs, **kwargs):
        logger = kwargs.pop("logger", self.logger)

        if logger is not None:
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


class ProgressBarCallback(TrainingCallback):
    def __init__(self):
        self.train_progress_bar = None
        self.eval_progress_bar = None

    def on_epoch_begin(self, training_config, **kwargs):
        epoch = kwargs.pop("epoch", None)
        train_loader = kwargs.pop("train_loader", None)
        eval_loader = kwargs.pop("eval_loader", None)

        if train_loader is not None:
            self.train_progress_bar = tqdm(
                total=len(train_loader),
                unit="batch",
                desc=f"Training of epoch {epoch}/{training_config.num_epochs}",
            )
            # self.train_progress_bar.set_description(
            #    f"Training of epoch {epoch}/{training_config.num_epochs}"
            # )

        if eval_loader is not None:
            self.eval_progress_bar = tqdm(
                total=len(eval_loader),
                unit="batch",
                desc=f"Eval of epoch {epoch}/{training_config.num_epochs}",
            )

    def on_train_step_end(self, training_config, **kwargs):
        if self.train_progress_bar is not None:
            self.train_progress_bar.update(1)

    def on_eval_step_end(self, training_config, **kwargs):
        if self.eval_progress_bar is not None:
            self.eval_progress_bar.update(1)

    def on_epoch_end(self, training_config, **kwags):
        if self.train_progress_bar is not None:
            self.train_progress_bar.close()

        if self.eval_progress_bar is not None:
            self.eval_progress_bar.close()


class WandbCallback(TrainingCallback):
    def __init__(self):
        if not wandb_is_available():
            raise ModuleNotFoundError(
                "`wandb` package must be installed. Run `pip install wandb`"
            )

        else:
            import wandb

            self._wandb = wandb

    def setup(self, training_config, **kwargs):
        self.is_initialized = True

        model_config = kwargs.pop("model_config", None)
        project_name = kwargs.pop("project_name", "pythae_benchmarking_vae")
        entity_name = kwargs.pop("entity_name", None)

        training_config_dict = training_config.to_dict()

        self._wandb.init(project=project_name, entity=entity_name)

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

    def on_train_begin(self, training_config, **kwargs):
        model_config = kwargs.pop("model_config", None)
        if not self.is_initialized:
            self.setup(training_config, model_config=model_config)

    def on_log(self, training_config, logs, **kwargs):
        global_step = kwargs.pop("global_step", None)
        logs = rename_logs(logs)

        self._wandb.log({**logs, "train/global_step": global_step})

    def on_prediction_step(self, training_config, **kwargs):
        global_step = kwargs.pop("global_step", None)

        column_names = ["images_id", "truth", "recontruction", "normal_generation"]

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
                            np.moveaxis(
                                reconstructions[i].cpu().detach().numpy(), 0, -1
                            )
                        ),
                        self._wandb.Image(
                            np.moveaxis(generations[i].cpu().detach().numpy(), 0, -1)
                        ),
                    ]
                )

            val_table = self._wandb.Table(data=data_to_log, columns=column_names)

            self._wandb.log({"my_val_table": val_table})
