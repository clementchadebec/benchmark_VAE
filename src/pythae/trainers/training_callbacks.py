"""Training Callbacks for training monitoring (inspired from 
https://github.com/huggingface/transformers/blob/master/src/transformers/trainer_callback.py)"""

from .base_trainer.base_training_config import BaseTrainingConfig
import importlib
import logging
from tqdm import tqdm

def wandb_is_available():
    return importlib.util.find_spec("wandb") is not None

class TrainingCallback:
    """
    Base class for creating training callbacks"""

    def on_init_end(self, training_config: BaseTrainingConfig, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
        pass

    def on_train_begin(self, training_config: BaseTrainingConfig, **kwargs):
        """
        Event called at the beginning of training.
        """
        pass

    def on_train_end(self, training_config: BaseTrainingConfig, **kwargs):
        """
        Event called at the end of training.
        """
        pass

    def on_epoch_begin(self, training_config: BaseTrainingConfig, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        pass

    def on_epoch_end(self, training_config: BaseTrainingConfig, **kwargs):
        """
        Event called at the end of an epoch.
        """
        pass

    def on_train_step_begin(self, training_config: BaseTrainingConfig, **kwargs):
        """
        Event called at the beginning of a training step.
        """
        pass

    def on_train_step_end(self, training_config: BaseTrainingConfig, **kwargs):
        """
        Event called at the end of a training step.
        """
        pass

    def on_eval_step_begin(self, training_config: BaseTrainingConfig, **kwargs):
        """
        Event called at the beginning of a evaluation step.
        """
        pass

    def on_eval_step_end(self, training_config: BaseTrainingConfig, **kwargs):
        """
        Event called at the end of a evaluation step.
        """
        pass

    def on_evaluate(self, training_config: BaseTrainingConfig, **kwargs):
        """
        Event called after an evaluation phase.
        """
        pass

    def on_save(self, training_config: BaseTrainingConfig, **kwargs):
        """
        Event called after a checkpoint save.
        """
        pass

    def on_log(self, training_config: BaseTrainingConfig, **kwargs):
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

    def on_init_end(self, training_config: BaseTrainingConfig, **kwargs):
        self.call_event("on_init_end", training_config, **kwargs)

    def on_train_step_begin(self, training_config: BaseTrainingConfig, **kwargs):
        self.call_event("on_train_step_begin", training_config, **kwargs)
    
    def on_train_step_end(self, training_config: BaseTrainingConfig, **kwargs):
        self.call_event("on_train_step_end", training_config, **kwargs)
    
    def on_eval_step_begin(self, training_config: BaseTrainingConfig, **kwargs):
        self.call_event("on_eval_step_begin", training_config, **kwargs)
    
    def on_eval_step_end(self, training_config: BaseTrainingConfig, **kwargs):
        self.call_event("on_eval_step_end", training_config, **kwargs)

    def on_train_begin(self, training_config: BaseTrainingConfig, **kwargs):
        self.call_event("on_train_begin", training_config,**kwargs)

    def on_train_end(self, training_config: BaseTrainingConfig, **kwargs):
        self.call_event("on_train_end", training_config, **kwargs)

    def on_epoch_begin(self, training_config: BaseTrainingConfig, **kwargs):
        self.call_event("on_epoch_begin", training_config,**kwargs)

    def on_epoch_end(self, training_config: BaseTrainingConfig, **kwargs):
        self.call_event("on_epoch_end", training_config)

    def on_evaluate(self, training_config: BaseTrainingConfig, **kwargs):
        control.should_evaluate = False
        self.call_event("on_evaluate", **kwargs)

    def on_save(self, training_config: BaseTrainingConfig, **kwargs):
        control.should_save = False
        self.call_event("on_save", training_config, **kwargs)

    def on_log(self, training_config: BaseTrainingConfig, logs, **kwargs):
        control.should_log = False
        self.call_event("on_log", training_config, **kwargs)

    def on_prediction_step(self, training_config: BaseTrainingConfig, **kwargs):
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

class ProgressBarCallback(TrainingCallback):

    def __init__(self):
        self.train_progress_bar = None
        self.eval_progress_bar = None

    def on_epoch_begin(self, training_config, **kwargs):
        epoch = kwargs.pop('epoch', None)
        train_loader = kwargs.pop('train_loader', None)
        eval_loader = kwargs.pop('eval_loader', None)

        if train_loader is not None:
            self.train_progress_bar = tqdm(
                total=len(train_loader),
                unit='batch',
                desc=f"Training of epoch {epoch}/{training_config.num_epochs}"
            )
            #self.train_progress_bar.set_description(
            #    f"Training of epoch {epoch}/{training_config.num_epochs}"
            #)
        
        if eval_loader is not None:
            self.eval_progress_bar = tqdm(
                total=len(eval_loader),
                unit='batch',
                desc=f"Eval of epoch {epoch}/{training_config.num_epochs}"
            )
        
    def on_train_step_end(self, training_config, **kwargs):
        if self.train_progress_bar is not None:
            batch_idx = kwargs.pop('batch_idx', None)
            self.train_progress_bar.update(batch_idx+1)

    def on_eval_step_end(self, training_config, **kwargs):
        if self.eval_progress_bar is not None:
            batch_idx = kwargs.pop('batch_idx', None)
            self.eval_progress_bar.update(batch_idx+1)
        
    def on_epoch_end(self, training_config, **kwags):
        if self.train_progress_bar is not None:
            self.train_progress_bar.close()
        
        if self.eval_progress_bar is not None:
            self.eval_progress_bar.close()


class WandbCallback(TrainingCallback):
    
    def __init__(self):
        if not wandb_is_available():
            raise ModuleNotFoundError('`wandb` package must be installed. Run `pip install wandb`')

        else:
            pass

