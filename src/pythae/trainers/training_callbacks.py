"""Training Callbacks for training monitoring (inspired from 
https://github.com/huggingface/transformers/blob/master/src/transformers/trainer_callback.py)"""

from .base_trainer.base_training_config import BaseTrainingConfig
import importlib

def wandb_is_available():
    return importlib.util.find_spec("wandb") is not None

class TrainingCallback:
    """
    Base class for creating training callbacks"""

    def on_init_end(self, args: BaseTrainingConfig, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
        pass

    def on_train_begin(self, args: BaseTrainingConfig, **kwargs):
        """
        Event called at the beginning of training.
        """
        pass

    def on_train_end(self, args: BaseTrainingConfig, **kwargs):
        """
        Event called at the end of training.
        """
        pass

    def on_epoch_begin(self, args: BaseTrainingConfig **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        pass

    def on_epoch_end(self, args: BaseTrainingConfig, **kwargs):
        """
        Event called at the end of an epoch.
        """
        pass

    def on_step_begin(self, args: TrainingArguments, **kwargs):
        """
        Event called at the beginning of a training step.
        """
        pass

    def on_step_end(self, args: TrainingArguments, **kwargs):
        """
        Event called at the end of a training step.
        """
        pass

    def on_evaluate(self, args: TrainingArguments, **kwargs):
        """
        Event called after an evaluation phase.
        """
        pass

    def on_save(self, args: TrainingArguments, **kwargs):
        """
        Event called after a checkpoint save.
        """
        pass

    def on_log(self, args: TrainingArguments, **kwargs):
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
            self.add_c
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = None
        self.eval_dataloader = None

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

    def on_init_end(self, args: TrainingArguments):
        return self.call_event("on_init_end", args, state, control)

    def on_train_begin(self, args: TrainingArguments):
        control.should_training_stop = False
        return self.call_event("on_train_begin", args, state, control)

    def on_train_end(self, args: TrainingArguments):
        return self.call_event("on_train_end", args, state, control)

    def on_epoch_begin(self, args: TrainingArguments):
        control.should_epoch_stop = False
        return self.call_event("on_epoch_begin", args, state, control)

    def on_epoch_end(self, args: TrainingArguments):
        return self.call_event("on_epoch_end", args, state, control)

    def on_step_begin(self, args: TrainingArguments):
        control.should_log = False
        control.should_evaluate = False
        control.should_save = False
        return self.call_event("on_step_begin", args, state, control)

    def on_step_end(self, args: TrainingArguments):
        return self.call_event("on_step_end", args, state, control)

    def on_evaluate(self, args: TrainingArguments, metrics):
        control.should_evaluate = False
        return self.call_event("on_evaluate", metrics=metrics)

    def on_save(self, args: TrainingArguments):
        control.should_save = False
        return self.call_event("on_save", args, state, control)

    def on_log(self, args: TrainingArguments, logs):
        control.should_log = False
        return self.call_event("on_log", args, state, control, logs=logs)

    def on_prediction_step(self, args: TrainingArguments):
        return self.call_event("on_prediction_step", args)

    def call_event(self, event, args, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                args,
                model=self.model,
                tokenizer=self.tokenizer,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )
            # A Callback can skip the return of `control` if it doesn't change it.
            if result is not None:
                control = result
        return control

class WandbCallback(TrainingCallback):
    
    def __init__(self):
        if not wandb_is_available():
            raise Modul
