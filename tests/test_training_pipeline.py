import os
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop

from pyraug.models import RHVAE
from pyraug.models.rhvae import RHVAEConfig
from pyraug.pipelines.training import TrainingPipeline
from pyraug.trainers.training_config import TrainingConfig
from tests.data.rhvae.custom_architectures import (
    Decoder_MLP_Custom,
    Encoder_MLP_Custom,
    Metric_MLP_Custom,
)

PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(
    params=[
        [
            [
                100 * torch.rand(3, 20, 15, 30),  # train data
                torch.rand(3, 10, 25, 10),
                torch.rand(3, 10, 10, 30),
            ],
            (3, 3, 10, 10, 10),  # target shape
            [
                100
                * torch.rand(
                    3, 20, 15, 30
                ),  # eval data (should be compatible with target shape)
                torch.rand(3, 10, 25, 10),
                torch.rand(3, 10, 10, 30),
            ],
        ],
        [
            [
                100 * torch.rand(1, 20, 15, 30),
                torch.rand(1, 10, 25, 10),
                torch.rand(1, 10, 10, 30),
                10000 * torch.rand(1, 10, 10, 30),
                100 * torch.rand(1, 100, 30, 30),
            ],
            (5, 1, 10, 10, 10),
            np.random.randn(2, 1, 10, 10, 10),
        ],
        [torch.randn(4, 12, 10), (4, 12, 10), None],
        [
            np.random.randn(10, 2, 17, 28),
            (10, 2, 17, 28),
            [torch.rand(2, 17, 28), torch.rand(2, 17, 28)],
        ],
        [
            os.path.join(PATH, "data/loading/dummy_data_folder"),
            (5, 3, 12, 12),
            os.path.join(PATH, "data/loading/dummy_data_folder"),
        ],
    ]
)
def messy_data(request):
    return request.param


class Test_Pipeline:
    @pytest.fixture(
        params=[
            TrainingConfig(max_epochs=3),
            TrainingConfig(max_epochs=3, learning_rate=1e-8),
            TrainingConfig(max_epochs=3, batch_size=12, train_early_stopping=False),
            TrainingConfig(
                max_epochs=3,
                batch_size=12,
                train_early_stopping=False,
                eval_early_stopping=2,
            ),
        ]
    )
    def training_config(self, tmpdir, request):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        request.param.output_dir = dir_path
        return request.param

    @pytest.fixture
    def rhvae_config(self, messy_data):
        return RHVAEConfig(input_dim=np.prod(messy_data[1][1:]))

    @pytest.fixture
    def custom_encoder(self, rhvae_config):
        return Encoder_MLP_Custom(rhvae_config)

    @pytest.fixture
    def custom_decoder(self, rhvae_config):
        return Decoder_MLP_Custom(rhvae_config)

    @pytest.fixture
    def custom_metric(self, rhvae_config):
        return Metric_MLP_Custom(rhvae_config)

    @pytest.fixture(params=[torch.rand(1), torch.rand(1), torch.rand(1)])
    def rhvae_sample(
        self, rhvae_config, custom_encoder, custom_decoder, custom_metric, request
    ):
        # randomized

        alpha = request.param

        if alpha < 0.125:
            model = RHVAE(rhvae_config)

        elif 0.125 <= alpha < 0.25:
            model = RHVAE(rhvae_config, encoder=custom_encoder)

        elif 0.25 <= alpha < 0.375:
            model = RHVAE(rhvae_config, decoder=custom_decoder)

        elif 0.375 <= alpha < 0.5:
            model = RHVAE(rhvae_config, metric=custom_metric)

        elif 0.5 <= alpha < 0.625:
            model = RHVAE(rhvae_config, encoder=custom_encoder, decoder=custom_decoder)

        elif 0.625 <= alpha < 0:
            model = RHVAE(rhvae_config, encoder=custom_encoder, metric=custom_metric)

        elif 0.750 <= alpha < 0.875:
            model = RHVAE(rhvae_config, decoder=custom_decoder, metric=custom_metric)

        else:
            model = RHVAE(
                rhvae_config,
                encoder=custom_encoder,
                decoder=custom_decoder,
                metric=custom_metric,
            )

        return model

    @pytest.fixture(params=[None, Adagrad, Adam, Adadelta, SGD, RMSprop])
    def optimizer(self, request, rhvae_sample, training_config):

        if request.param is not None:
            optimizer = request.param(
                rhvae_sample.parameters(), lr=training_config.learning_rate
            )

        else:
            optimizer = None

        return optimizer

    def test_pipeline_tensor_data(
        self, messy_data, rhvae_sample, optimizer, training_config
    ):
        pipe = TrainingPipeline(
            model=rhvae_sample, optimizer=optimizer, training_config=training_config
        )

        start_model = deepcopy(rhvae_sample)

        pipe(train_data=messy_data[0], eval_data=messy_data[2])

        assert bool(all([c_data.min() == 0 for c_data in pipe.train_data])) and bool(
            all([c_data.max() == 1 for c_data in pipe.train_data])
        )

        assert not all(
            [
                torch.equal(
                    pipe.model.state_dict()[key].cpu(),
                    start_model.state_dict()[key].cpu(),
                )
                for key in start_model.state_dict().keys()
            ]
        )

    def test_saving(self, tmpdir, messy_data, rhvae_sample, optimizer, training_config):

        dir_path = training_config.output_dir

        pipe = TrainingPipeline(
            model=rhvae_sample, optimizer=optimizer, training_config=training_config
        )

        pipe(train_data=messy_data[0])

        model = deepcopy(pipe.model)

        training_dir = os.path.join(
            dir_path, f"training_{pipe.trainer._training_signature}"
        )
        assert training_dir

        final_dir = os.path.join(training_dir, f"final_model")
        assert os.path.isdir(final_dir)

        files_list = os.listdir(final_dir)

        assert set(["model.pt", "model_config.json", "training_config.json"]).issubset(
            set(files_list)
        )

        # check pickled custom decoder
        if not rhvae_sample.model_config.uses_default_decoder:
            assert "decoder.pkl" in files_list

        else:
            assert not "decoder.pkl" in files_list

        # check pickled custom encoder
        if not rhvae_sample.model_config.uses_default_encoder:
            assert "encoder.pkl" in files_list

        else:
            assert not "encoder.pkl" in files_list

        # check pickled custom metric
        if not rhvae_sample.model_config.uses_default_metric:
            assert "metric.pkl" in files_list

        else:
            assert not "metric.pkl" in files_list

        # check reload full model
        model_rec = RHVAE.load_from_folder(os.path.join(final_dir))

        assert all(
            [
                torch.equal(
                    model_rec.state_dict()[key].cpu(), model.state_dict()[key].cpu()
                )
                for key in model.state_dict().keys()
            ]
        )

        assert torch.equal(model_rec.M_tens.cpu(), model.M_tens.cpu())
        assert torch.equal(model_rec.centroids_tens.cpu(), model.centroids_tens.cpu())
        assert type(model_rec.encoder.cpu()) == type(model.encoder.cpu())
        assert type(model_rec.decoder.cpu()) == type(model.decoder.cpu())
        assert type(model_rec.metric.cpu()) == type(model.metric.cpu())


class Test_Logging:
    @pytest.fixture
    def train_data(self):
        return os.path.join(PATH, "data/loading/dummy_data_folder")

    @pytest.fixture
    def training_config(self, tmpdir):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        return TrainingConfig(output_dir=dir_path, max_epochs=2)

    @pytest.fixture
    def model_sample(self):
        return RHVAE(RHVAEConfig(input_dim=3 * 12 * 12))

    def test_create_log_file(self, tmpdir, model_sample, train_data, training_config):
        dir_log_path = os.path.join(tmpdir, "dummy_folder")

        pipe = TrainingPipeline(model=model_sample, training_config=training_config)

        pipe(train_data=train_data, log_output_dir=dir_log_path)

        assert os.path.isdir(dir_log_path)
        assert f"training_logs_{pipe.trainer._training_signature}.log" in os.listdir(
            dir_log_path
        )
