import pytest
import torch
import os

from pythae.trainers.training_callbacks import *
from pythae.trainers import BaseTrainer, BaseTrainerConfig
from pythae.models import AE, AEConfig

PATH = os.path.dirname(os.path.abspath(__file__))


class CustomCallback(TrainingCallback):
    def __init__(self):
        pass


@pytest.fixture()
def init_callback():
    return ProgressBarCallback()


@pytest.fixture()
def dummy_handler(init_callback):
    return CallbackHandler(
        callbacks=[init_callback], model=None, optimizer=None, scheduler=None
    )


@pytest.fixture(params=[MetricConsolePrinterCallback(), CustomCallback()])
def callbacks(request):
    return request.param


class Test_Handler:
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

        assert callbacks not in trainer.callback_handler.callbacks

        assert ProgressBarCallback().__class__ in [
            cb.__class__ for cb in trainer.callback_handler.callbacks
        ]

        trainer = BaseTrainer(
            model=ae, callbacks=[callbacks], train_dataset=train_dataset
        )

        assert callbacks in trainer.callback_handler.callbacks

        assert ProgressBarCallback().__class__ in [
            cb.__class__ for cb in trainer.callback_handler.callbacks
        ]
