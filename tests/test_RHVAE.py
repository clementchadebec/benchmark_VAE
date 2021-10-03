import os
from copy import deepcopy

import dill
import pytest
import torch

from pythae.config import BaseConfig
from pythae.customexception import BadInheritanceError
from pythae.models import RHVAE
from pythae.models.nn.default_architectures import Decoder_MLP, Encoder_MLP, Metric_MLP
from pythae.models.rhvae.rhvae_config import RHVAEConfig
from tests.data.rhvae.custom_architectures import (
    Decoder_Conv,
    Encoder_Conv,
    Metric_Custom,
    NetBadInheritance,
)

PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(params=[RHVAEConfig(), RHVAEConfig(n_lf=1, temperature=10)])
def rhvae_configs_no_input_dim(request):
    return request.param


@pytest.fixture()
def rhvae_config_with_input_dim():
    return RHVAEConfig(
        input_dim=784,  # Simulates data loading (where the input shape is computed). This needed to
        # create dummy custom encoders and decoders
        latent_dim=10,
    )


@pytest.fixture
def demo_data():
    data = torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample"))[:]
    return (
        data
    )  # This is an extract of 3 data from MNIST (unnormalized) used to test custom architecture


@pytest.fixture
def custom_encoder(rhvae_config_with_input_dim):
    return Encoder_Conv(rhvae_config_with_input_dim)


@pytest.fixture
def custom_decoder(rhvae_config_with_input_dim):
    return Decoder_Conv(rhvae_config_with_input_dim)


@pytest.fixture
def custom_metric():
    return Metric_Custom()


class Test_Build_RHVAE:
    @pytest.fixture()
    def bad_net(self):
        return NetBadInheritance()

    def test_build_model(self, rhvae_config_with_input_dim):
        rhvae = RHVAE(rhvae_config_with_input_dim)

        assert all(
            [
                rhvae.n_lf == rhvae_config_with_input_dim.n_lf,
                rhvae.temperature == rhvae_config_with_input_dim.temperature,
            ]
        )

    def test_raises_bad_inheritance(self, rhvae_config_with_input_dim, bad_net):
        with pytest.raises(BadInheritanceError):
            rhvae = RHVAE(rhvae_config_with_input_dim, encoder=bad_net)

        with pytest.raises(BadInheritanceError):
            rhvae = RHVAE(rhvae_config_with_input_dim, decoder=bad_net)

        with pytest.raises(BadInheritanceError):
            rhvae = RHVAE(rhvae_config_with_input_dim, metric=bad_net)

    def test_raises_no_input_dim(
        self, rhvae_configs_no_input_dim, custom_encoder, custom_decoder, custom_metric
    ):
        with pytest.raises(AttributeError):
            rhvae = RHVAE(rhvae_configs_no_input_dim)

        with pytest.raises(AttributeError):
            rhvae = RHVAE(rhvae_configs_no_input_dim, encoder=custom_encoder)

        with pytest.raises(AttributeError):
            rhvae = RHVAE(rhvae_configs_no_input_dim, decoder=custom_decoder)

        with pytest.raises(AttributeError):
            rhvae = RHVAE(rhvae_configs_no_input_dim, metric=custom_metric)

        rhvae = RHVAE(
            rhvae_configs_no_input_dim,
            encoder=custom_encoder,
            decoder=custom_decoder,
            metric=custom_metric,
        )

    def test_build_custom_arch(
        self, rhvae_config_with_input_dim, custom_encoder, custom_decoder, custom_metric
    ):

        rhvae = RHVAE(
            rhvae_config_with_input_dim, encoder=custom_encoder, decoder=custom_decoder
        )

        assert rhvae.encoder == custom_encoder
        assert not rhvae.model_config.uses_default_encoder

        assert rhvae.decoder == custom_decoder
        assert not rhvae.model_config.uses_default_encoder

        assert rhvae.model_config.uses_default_metric

        rhvae = RHVAE(rhvae_config_with_input_dim, metric=custom_metric)

        assert rhvae.model_config.uses_default_encoder
        assert rhvae.model_config.uses_default_encoder

        assert rhvae.metric == custom_metric
        assert not rhvae.model_config.uses_default_metric


class Test_Model_Saving:
    def test_default_model_saving(self, tmpdir, rhvae_config_with_input_dim):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = RHVAE(rhvae_config_with_input_dim)

        # set random M_tens and centroids from testing
        model.M_tens = torch.randn(3, 10, 10)
        model.centroids_tens = torch.randn(3, 10, 10)

        model.state_dict()["encoder.layers.0.weight"][0] = 0

        model.save(dir_path=dir_path)

        assert set(os.listdir(dir_path)) == set(["model_config.json", "model.pt"])

        # reload model
        model_rec = RHVAE.load_from_folder(dir_path)

        # check configs are the same
        assert model_rec.model_config.__dict__ == model.model_config.__dict__

        assert all(
            [
                torch.equal(model_rec.state_dict()[key], model.state_dict()[key])
                for key in model.state_dict().keys()
            ]
        )

        assert torch.equal(model_rec.M_tens, model.M_tens)
        assert torch.equal(model_rec.centroids_tens, model.centroids_tens)

        assert callable(model_rec.G)
        assert callable(model_rec.G_inv)

    def test_custom_encoder_model_saving(
        self, tmpdir, rhvae_config_with_input_dim, custom_encoder
    ):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = RHVAE(rhvae_config_with_input_dim, encoder=custom_encoder)

        model.state_dict()["encoder.layers.0.weight"][0] = 0

        model.save(dir_path=dir_path)

        assert set(os.listdir(dir_path)) == set(
            ["model_config.json", "model.pt", "encoder.pkl"]
        )

        # reload model
        model_rec = RHVAE.load_from_folder(dir_path)

        # check configs are the same
        assert model_rec.model_config.__dict__ == model.model_config.__dict__

        assert all(
            [
                torch.equal(model_rec.state_dict()[key], model.state_dict()[key])
                for key in model.state_dict().keys()
            ]
        )

        assert torch.equal(model_rec.M_tens, model.M_tens)
        assert torch.equal(model_rec.centroids_tens, model.centroids_tens)

        assert callable(model_rec.G)
        assert callable(model_rec.G_inv)

    def test_custom_decoder_model_saving(
        self, tmpdir, rhvae_config_with_input_dim, custom_decoder
    ):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = RHVAE(rhvae_config_with_input_dim, decoder=custom_decoder)

        model.state_dict()["encoder.layers.0.weight"][0] = 0

        model.save(dir_path=dir_path)

        assert set(os.listdir(dir_path)) == set(
            ["model_config.json", "model.pt", "decoder.pkl"]
        )

        # reload model
        model_rec = RHVAE.load_from_folder(dir_path)

        # check configs are the same
        assert model_rec.model_config.__dict__ == model.model_config.__dict__

        assert all(
            [
                torch.equal(model_rec.state_dict()[key], model.state_dict()[key])
                for key in model.state_dict().keys()
            ]
        )

        assert torch.equal(model_rec.M_tens, model.M_tens)
        assert torch.equal(model_rec.centroids_tens, model.centroids_tens)

        assert callable(model_rec.G)
        assert callable(model_rec.G_inv)

    def test_custom_metric_model_saving(
        self, tmpdir, rhvae_config_with_input_dim, custom_metric
    ):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = RHVAE(rhvae_config_with_input_dim, metric=custom_metric)

        model.state_dict()["encoder.layers.0.weight"][0] = 0

        model.save(dir_path=dir_path)

        assert set(os.listdir(dir_path)) == set(
            ["model_config.json", "model.pt", "metric.pkl"]
        )

        # reload model
        model_rec = RHVAE.load_from_folder(dir_path)

        # check configs are the same
        assert model_rec.model_config.__dict__ == model.model_config.__dict__

        assert all(
            [
                torch.equal(model_rec.state_dict()[key], model.state_dict()[key])
                for key in model.state_dict().keys()
            ]
        )

        assert torch.equal(model_rec.M_tens, model.M_tens)
        assert torch.equal(model_rec.centroids_tens, model.centroids_tens)

        assert callable(model_rec.G)
        assert callable(model_rec.G_inv)

    def test_full_custom_model_saving(
        self,
        tmpdir,
        rhvae_config_with_input_dim,
        custom_encoder,
        custom_decoder,
        custom_metric,
    ):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = RHVAE(
            rhvae_config_with_input_dim,
            encoder=custom_encoder,
            decoder=custom_decoder,
            metric=custom_metric,
        )

        model.state_dict()["encoder.layers.0.weight"][0] = 0

        model.save(dir_path=dir_path)

        assert set(os.listdir(dir_path)) == set(
            [
                "model_config.json",
                "model.pt",
                "encoder.pkl",
                "decoder.pkl",
                "metric.pkl",
            ]
        )

        # reload model
        model_rec = RHVAE.load_from_folder(dir_path)

        # check configs are the same
        assert model_rec.model_config.__dict__ == model.model_config.__dict__

        assert all(
            [
                torch.equal(model_rec.state_dict()[key], model.state_dict()[key])
                for key in model.state_dict().keys()
            ]
        )

        assert torch.equal(model_rec.M_tens, model.M_tens)
        assert torch.equal(model_rec.centroids_tens, model.centroids_tens)

        assert callable(model_rec.G)
        assert callable(model_rec.G_inv)

    def test_raises_missing_files(
        self,
        tmpdir,
        rhvae_config_with_input_dim,
        custom_encoder,
        custom_decoder,
        custom_metric,
    ):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = RHVAE(
            rhvae_config_with_input_dim,
            encoder=custom_encoder,
            decoder=custom_decoder,
            metric=custom_metric,
        )

        model.state_dict()["encoder.layers.0.weight"][0] = 0

        model.save(dir_path=dir_path)

        os.remove(os.path.join(dir_path, "metric.pkl"))

        # check raises decoder.pkl is missing
        with pytest.raises(FileNotFoundError):
            model_rec = RHVAE.load_from_folder(dir_path)

        os.remove(os.path.join(dir_path, "decoder.pkl"))

        # check raises decoder.pkl is missing
        with pytest.raises(FileNotFoundError):
            model_rec = RHVAE.load_from_folder(dir_path)

        os.remove(os.path.join(dir_path, "encoder.pkl"))

        # check raises encoder.pkl is missing
        with pytest.raises(FileNotFoundError):
            model_rec = RHVAE.load_from_folder(dir_path)

        os.remove(os.path.join(dir_path, "model.pt"))

        # check raises encoder.pkl is missing
        with pytest.raises(FileNotFoundError):
            model_rec = RHVAE.load_from_folder(dir_path)

        os.remove(os.path.join(dir_path, "model_config.json"))

        # check raises encoder.pkl is missing
        with pytest.raises(FileNotFoundError):
            model_rec = RHVAE.load_from_folder(dir_path)


class Test_Model_forward:
    @pytest.fixture
    def rhvae(self, rhvae_config_with_input_dim, demo_data):
        rhvae_config_with_input_dim.input_dim = demo_data["data"][0].shape[-1]
        return RHVAE(rhvae_config_with_input_dim)

    def test_model_train_output(self, rhvae, demo_data):

        # rhvae_config_with_input_dim.input_dim = demo_data['data'][0].shape[-1]

        # rhvae = RHVAE(rhvae_config_with_input_dim)

        rhvae.train()

        out = rhvae(demo_data)
        assert set(
            [
                "loss",
                "recon_x",
                "z",
                "z0",
                "rho",
                "eps0",
                "gamma",
                "mu",
                "log_var",
                "G_inv",
                "G_log_det",
            ]
        ) == set(out.keys())

        rhvae.update()

    def test_model_output(self, rhvae, demo_data):

        # rhvae_config_with_input_dim.input_dim = demo_data['data'][0].shape[-1]

        rhvae.eval()

        out = rhvae(demo_data)
        assert set(
            [
                "loss",
                "recon_x",
                "z",
                "z0",
                "rho",
                "eps0",
                "gamma",
                "mu",
                "log_var",
                "G_inv",
                "G_log_det",
            ]
        ) == set(out.keys())


class Test_Load_RHVAE_Config_From_JSON:
    @pytest.fixture(
        params=[
            os.path.join(PATH, "data/rhvae/configs/model_config00.json"),
            os.path.join(PATH, "data/rhvae/configs/training_config00.json"),
            os.path.join(PATH, "data/rhvae/configs/generation_config00.json"),
        ]
    )
    def custom_config_path(self, request):
        return request.param

    @pytest.fixture
    def corrupted_config_path(self):
        return "corrupted_path"

    @pytest.fixture
    def not_json_config_path(self):
        return os.path.join(PATH, "data/rhvae/configs/not_json_file.md")

    @pytest.fixture(
        params=[
            [
                os.path.join(PATH, "data/rhvae/configs/model_config00.json"),
                RHVAEConfig(
                    latent_dim=11,
                    n_lf=2,
                    eps_lf=0.00001,
                    temperature=0.5,
                    regularization=0.1,
                    beta_zero=0.8,
                ),
            ],
            [
                os.path.join(PATH, "data/rhvae/configs/training_config00.json"),
                TrainingConfig(
                    batch_size=3,
                    max_epochs=2,
                    learning_rate=1e-5,
                    train_early_stopping=10,
                ),
            ],
            [
                os.path.join(PATH, "data/rhvae/configs/generation_config00.json"),
                RHVAESamplerConfig(
                    batch_size=3, mcmc_steps_nbr=3, n_lf=2, eps_lf=0.003
                ),
            ],
        ]
    )
    def custom_config_path_with_true_config(self, request):
        return request.param

    def test_load_custom_config(self, custom_config_path_with_true_config):

        config_path = custom_config_path_with_true_config[0]
        true_config = custom_config_path_with_true_config[1]

        if config_path == os.path.join(PATH, "data/rhvae/configs/model_config00.json"):
            parsed_config = RHVAEConfig.from_json_file(config_path)

        elif config_path == os.path.join(
            PATH, "data/rhvae/configs/training_config00.json"
        ):
            parsed_config = TrainingConfig.from_json_file(config_path)

        else:
            parsed_config = RHVAESamplerConfig.from_json_file(config_path)

        assert parsed_config == true_config

    def test_load_dict_from_json_config(self, custom_config_path):
        config_dict = BaseConfig._dict_from_json(custom_config_path)
        assert type(config_dict) == dict

    def test_raise_load_file_not_found(self, corrupted_config_path):
        with pytest.raises(FileNotFoundError):
            _ = BaseConfig._dict_from_json(corrupted_config_path)

    def test_raise_not_json_file(self, not_json_config_path):
        with pytest.raises(TypeError):
            _ = BaseConfig._dict_from_json(not_json_config_path)


class Test_Load_Config_From_Dict:
    @pytest.fixture(params=[{"latant_dim": 10}, {"batsh_size": 1}, {"mcmc_steps": 12}])
    def corrupted_keys_dict_config(self, request):
        return request.param

    def test_raise_type_error_corrupted_keys(self, corrupted_keys_dict_config):
        if set(corrupted_keys_dict_config.keys()).issubset(["latant_dim"]):
            with pytest.raises(TypeError):
                RHVAEConfig.from_dict(corrupted_keys_dict_config)

        elif set(corrupted_keys_dict_config.keys()).issubset(["batsh_size"]):
            with pytest.raises(TypeError):
                TrainingConfig.from_dict(corrupted_keys_dict_config)

        else:
            with pytest.raises(TypeError):
                RHVAESamplerConfig.from_dict(corrupted_keys_dict_config)

    @pytest.fixture(
        params=[
            {"latent_dim": "bad_type"},
            {"batch_size": "bad_type"},
            {"mcmc_steps_nbr": "bad_type"},
        ]
    )
    def corrupted_type_dict_config(self, request):
        return request.param

    def test_raise_type_error_corrupted_keys(self, corrupted_type_dict_config):

        if set(corrupted_type_dict_config.keys()).issubset(["latent_dim"]):
            with pytest.raises(ValidationError):
                RHVAEConfig.from_dict(corrupted_type_dict_config)

        elif set(corrupted_type_dict_config.keys()).issubset(["batch_size"]):
            with pytest.raises(ValidationError):
                TrainingConfig.from_dict(corrupted_type_dict_config)

        else:
            with pytest.raises(ValidationError):
                RHVAESamplerConfig.from_dict(corrupted_type_dict_config)
