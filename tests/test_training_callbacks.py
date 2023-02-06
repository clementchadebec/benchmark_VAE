import os
from collections import Counter

import pytest
import torch

from pythae.models import (
    AE,
    RAE_L2,
    VAEGAN,
    Adversarial_AE,
    Adversarial_AE_Config,
    AEConfig,
    RAE_L2_Config,
    VAEGANConfig,
)
from pythae.trainers import *
from pythae.trainers.training_callbacks import *

PATH = os.path.dirname(os.path.abspath(__file__))


class CustomCallback(TrainingCallback):
    def __init__(self):
        self.step_list = []
        pass

    def on_init_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
        self.step_list.append("on_init_end")

    def on_train_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the beginning of training.
        """
        self.step_list.append("on_train_begin")

    def on_train_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of training.
        """
        self.step_list.append("on_train_end")

    def on_epoch_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        self.step_list.append("on_epoch_begin")

    def on_epoch_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of an epoch.
        """
        self.step_list.append("on_epoch_end")

    def on_train_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the beginning of a training step.
        """
        self.step_list.append("on_train_step_begin")

    def on_train_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of a training step.
        """
        self.step_list.append("on_train_step_end")

    def on_eval_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the beginning of a evaluation step.
        """
        self.step_list.append("on_eval_step_begin")

    def on_eval_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of a evaluation step.
        """
        self.step_list.append("on_eval_step_end")

    def on_evaluate(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called after an evaluation phase.
        """
        self.step_list.append("on_evaluate")

    def on_prediction_step(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called after a prediction phase.
        """
        self.step_list.append("on_prediction_step")

    def on_save(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called after a checkpoint save.
        """
        self.step_list.append("on_save")

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        """
        Event called after logging the last logs.
        """
        self.step_list.append("on_log")


@pytest.fixture()
def init_callback():
    return ProgressBarCallback()


@pytest.fixture()
def dummy_handler(init_callback):
    return CallbackHandler(callbacks=[init_callback], model=None)


@pytest.fixture(params=[MetricConsolePrinterCallback(), CustomCallback()])
def callbacks(request):
    return request.param


class Test_CallbackHandler:
    def test_add_callback(self, dummy_handler, callbacks):
        dummy_handler.add_callback(callbacks)
        assert callbacks in dummy_handler.callbacks


class Test_TrainerCallbacks:
    @pytest.fixture
    def train_dataset(self):
        return torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample"))

    @pytest.fixture(params=[BaseTrainerConfig(num_epochs=3)])
    def training_configs(self, tmpdir, request):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        request.param.output_dir = dir_path
        return request.param

    @pytest.fixture(
        params=[
            AEConfig(input_dim=(1, 28, 28)),
        ]
    )
    def ae(self, request):
        return AE(model_config=request.param)

    def test_init_trainer(self, ae, train_dataset, callbacks):

        trainer = BaseTrainer(model=ae, train_dataset=train_dataset)

        trainer.prepare_training()

        assert callbacks not in trainer.callback_handler.callbacks

        assert ProgressBarCallback().__class__ in [
            cb.__class__ for cb in trainer.callback_handler.callbacks
        ]

        trainer = BaseTrainer(
            model=ae, callbacks=[callbacks], train_dataset=train_dataset
        )

        trainer.prepare_training()

        assert callbacks in trainer.callback_handler.callbacks

        assert ProgressBarCallback().__class__ in [
            cb.__class__ for cb in trainer.callback_handler.callbacks
        ]


class Test_TrainersCalls:
    @pytest.fixture
    def train_dataset(self):
        return torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample"))

    @pytest.fixture(
        params=[
            [
                AEConfig(input_dim=(1, 28, 28)),
                BaseTrainerConfig(num_epochs=2, steps_predict=1),
            ],
            [
                RAE_L2_Config(input_dim=(1, 28, 28)),
                CoupledOptimizerTrainerConfig(num_epochs=3, steps_predict=1),
            ],
            [
                Adversarial_AE_Config(input_dim=(1, 28, 28)),
                AdversarialTrainerConfig(num_epochs=3, steps_predict=2),
            ],
            [
                VAEGANConfig(input_dim=(1, 28, 28)),
                CoupledOptimizerAdversarialTrainerConfig(num_epochs=3, steps_predict=1),
            ],
        ]
    )
    def configs_and_models(self, tmpdir, request):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        request.param[1].output_dir = dir_path

        if request.param[0].name == "AEConfig":
            model = AE(request.param[0])
        elif request.param[0].name == "RAE_L2_Config":
            model = RAE_L2(request.param[0])
        elif request.param[0].name == "Adversarial_AE_Config":
            model = Adversarial_AE(request.param[0])
        else:
            model = VAEGAN(request.param[0])

        return [model, request.param[1]]

    @pytest.fixture()
    def dummy_callback(self):
        return CustomCallback()

    @pytest.fixture
    def trainer_cls(self, configs_and_models):
        if configs_and_models[1].name == "BaseTrainerConfig":
            return BaseTrainer
        elif configs_and_models[1].name == "CoupledOptimizerTrainerConfig":
            return CoupledOptimizerTrainer
        elif configs_and_models[1].name == "AdversarialTrainerConfig":
            return AdversarialTrainer
        else:
            return CoupledOptimizerAdversarialTrainer

    def test_trainer_callback_calls(
        self, trainer_cls, configs_and_models, train_dataset, dummy_callback
    ):

        trainer = trainer_cls(
            model=configs_and_models[0],
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            training_config=configs_and_models[1],
            callbacks=[dummy_callback],
        )

        trainer.prepare_training()

        assert "on_train_step_begin" not in dummy_callback.step_list
        assert "on_train_step_end" not in dummy_callback.step_list
        trainer.train_step(epoch=1)
        assert "on_train_step_begin" in dummy_callback.step_list
        assert "on_train_step_end" in dummy_callback.step_list

        assert "on_eval_step_begin" not in dummy_callback.step_list
        assert "on_eval_step_end" not in dummy_callback.step_list
        trainer.eval_step(epoch=1)
        assert "on_eval_step_begin" in dummy_callback.step_list
        assert "on_eval_step_end" in dummy_callback.step_list

        assert "on_save" not in dummy_callback.step_list
        trainer.save_model(
            configs_and_models[0],
            os.path.join(configs_and_models[1].output_dir, "final_model"),
        )
        assert "on_save" in dummy_callback.step_list

        assert "on_train_begin" not in dummy_callback.step_list
        assert "on_train_end" not in dummy_callback.step_list
        assert "on_epoch_begin" not in dummy_callback.step_list
        assert "on_epoch_end" not in dummy_callback.step_list
        assert "on_prediction_step" not in dummy_callback.step_list
        assert "on_log" not in dummy_callback.step_list
        trainer.train()
        assert "on_train_begin" in dummy_callback.step_list
        assert "on_train_end" in dummy_callback.step_list
        assert "on_epoch_begin" in dummy_callback.step_list
        assert "on_epoch_end" in dummy_callback.step_list
        assert "on_prediction_step" in dummy_callback.step_list
        assert "on_log" in dummy_callback.step_list

        count = Counter(dummy_callback.step_list)

        assert count["on_train_begin"] == 1
        assert count["on_train_end"] == 1
        assert (
            count["on_prediction_step"]
            == configs_and_models[1].num_epochs // configs_and_models[1].steps_predict
        )
        assert count["on_epoch_begin"] == configs_and_models[1].num_epochs
        assert count["on_epoch_end"] == configs_and_models[1].num_epochs
        assert count["on_log"] == configs_and_models[1].num_epochs

        assert count["on_train_step_begin"] == configs_and_models[1].num_epochs + 1
        assert count["on_train_step_end"] == configs_and_models[1].num_epochs + 1
        assert count["on_eval_step_begin"] == configs_and_models[1].num_epochs + 1
        assert count["on_eval_step_end"] == configs_and_models[1].num_epochs + 1
