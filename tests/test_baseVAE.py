import os

import pytest
import torch

from pyraug.customexception import BadInheritanceError
from pyraug.models import BaseVAE
from pyraug.models.base.base_config import BaseModelConfig, BaseSamplerConfig
from tests.data.rhvae.custom_architectures import (
    Decoder_Conv,
    Encoder_Conv,
    NetBadInheritance,
)


@pytest.fixture(params=[BaseModelConfig(), BaseModelConfig(latent_dim=5)])
def model_configs_no_input_dim(request):
    return request.param


@pytest.fixture(
    params=[
        BaseModelConfig(input_dim=784, latent_dim=10),
        BaseModelConfig(input_dim=100, latent_dim=5),
        BaseModelConfig(input_dim=1e4, latent_dim=5),
    ]
)
def model_config_with_input_dim(request):
    return request.param


@pytest.fixture
def custom_encoder(model_config_with_input_dim):
    return Encoder_Conv(model_config_with_input_dim)


@pytest.fixture
def custom_decoder(model_config_with_input_dim):
    return Decoder_Conv(model_config_with_input_dim)


class Test_Model_Building:
    @pytest.fixture()
    def bad_net(self):
        return NetBadInheritance()

    def test_build_model(self, model_config_with_input_dim):
        model = BaseVAE(model_config_with_input_dim)
        assert all(
            [
                model.input_dim == model_config_with_input_dim.input_dim,
                model.latent_dim == model_config_with_input_dim.latent_dim,
            ]
        )

    def test_raises_bad_inheritance(self, model_config_with_input_dim, bad_net):
        with pytest.raises(BadInheritanceError):
            model = BaseVAE(model_config_with_input_dim, encoder=bad_net)

        with pytest.raises(BadInheritanceError):
            model = BaseVAE(model_config_with_input_dim, decoder=bad_net)

    def test_raises_no_input_dim(
        self, model_configs_no_input_dim, custom_encoder, custom_decoder
    ):
        with pytest.raises(AttributeError):
            model = BaseVAE(model_configs_no_input_dim)

        with pytest.raises(AttributeError):
            model = BaseVAE(model_configs_no_input_dim, encoder=custom_encoder)

        with pytest.raises(AttributeError):
            model = BaseVAE(model_configs_no_input_dim, decoder=custom_decoder)

        model = BaseVAE(
            model_configs_no_input_dim, encoder=custom_encoder, decoder=custom_decoder
        )

    def test_build_custom_arch(
        self, model_config_with_input_dim, custom_encoder, custom_decoder
    ):

        model = BaseVAE(
            model_config_with_input_dim, encoder=custom_encoder, decoder=custom_decoder
        )

        assert model.encoder == custom_encoder
        assert not model.model_config.uses_default_encoder
        assert model.decoder == custom_decoder
        assert not model.model_config.uses_default_decoder

        model = BaseVAE(model_config_with_input_dim, encoder=custom_encoder)

        assert model.encoder == custom_encoder
        assert not model.model_config.uses_default_encoder
        assert model.model_config.uses_default_decoder

        model = BaseVAE(model_config_with_input_dim, decoder=custom_decoder)

        assert model.model_config.uses_default_encoder
        assert model.decoder == custom_decoder
        assert not model.model_config.uses_default_decoder


class Test_Model_Saving:
    def test_default_model_saving(self, tmpdir, model_config_with_input_dim):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = BaseVAE(model_config_with_input_dim)

        model.state_dict()["encoder.layers.0.weight"][0] = 0

        model.save(dir_path=dir_path)

        assert set(os.listdir(dir_path)) == set(["model_config.json", "model.pt"])

        # reload model
        model_rec = BaseVAE.load_from_folder(dir_path)

        # check configs are the same
        assert model_rec.model_config.__dict__ == model.model_config.__dict__

        assert all(
            [
                torch.equal(model_rec.state_dict()[key], model.state_dict()[key])
                for key in model.state_dict().keys()
            ]
        )

    def test_custom_encoder_model_saving(
        self, tmpdir, model_config_with_input_dim, custom_encoder
    ):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = BaseVAE(model_config_with_input_dim, encoder=custom_encoder)

        model.state_dict()["encoder.layers.0.weight"][0] = 0

        model.save(dir_path=dir_path)

        assert set(os.listdir(dir_path)) == set(
            ["model_config.json", "model.pt", "encoder.pkl"]
        )

        # reload model
        model_rec = BaseVAE.load_from_folder(dir_path)

        # check configs are the same
        assert model_rec.model_config.__dict__ == model.model_config.__dict__

        assert all(
            [
                torch.equal(model_rec.state_dict()[key], model.state_dict()[key])
                for key in model.state_dict().keys()
            ]
        )

    def test_custom_decoder_model_saving(
        self, tmpdir, model_config_with_input_dim, custom_decoder
    ):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = BaseVAE(model_config_with_input_dim, decoder=custom_decoder)

        model.state_dict()["encoder.layers.0.weight"][0] = 0

        model.save(dir_path=dir_path)

        assert set(os.listdir(dir_path)) == set(
            ["model_config.json", "model.pt", "decoder.pkl"]
        )

        # reload model
        model_rec = BaseVAE.load_from_folder(dir_path)

        # check configs are the same
        assert model_rec.model_config.__dict__ == model.model_config.__dict__

        assert all(
            [
                torch.equal(model_rec.state_dict()[key], model.state_dict()[key])
                for key in model.state_dict().keys()
            ]
        )

    def test_full_custom_model_saving(
        self, tmpdir, model_config_with_input_dim, custom_encoder, custom_decoder
    ):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = BaseVAE(
            model_config_with_input_dim, encoder=custom_encoder, decoder=custom_decoder
        )

        model.state_dict()["encoder.layers.0.weight"][0] = 0

        model.save(dir_path=dir_path)

        assert set(os.listdir(dir_path)) == set(
            ["model_config.json", "model.pt", "encoder.pkl", "decoder.pkl"]
        )

        # reload model
        model_rec = BaseVAE.load_from_folder(dir_path)

        # check configs are the same
        assert model_rec.model_config.__dict__ == model.model_config.__dict__

        assert all(
            [
                torch.equal(model_rec.state_dict()[key], model.state_dict()[key])
                for key in model.state_dict().keys()
            ]
        )

    def test_raises_missing_files(
        self, tmpdir, model_config_with_input_dim, custom_encoder, custom_decoder
    ):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = BaseVAE(
            model_config_with_input_dim, encoder=custom_encoder, decoder=custom_decoder
        )

        model.state_dict()["encoder.layers.0.weight"][0] = 0

        model.save(dir_path=dir_path)

        os.remove(os.path.join(dir_path, "decoder.pkl"))

        # check raises decoder.pkl is missing
        with pytest.raises(FileNotFoundError):
            model_rec = BaseVAE.load_from_folder(dir_path)

        os.remove(os.path.join(dir_path, "encoder.pkl"))

        # check raises encoder.pkl is missing
        with pytest.raises(FileNotFoundError):
            model_rec = BaseVAE.load_from_folder(dir_path)

        os.remove(os.path.join(dir_path, "model.pt"))

        # check raises encoder.pkl is missing
        with pytest.raises(FileNotFoundError):
            model_rec = BaseVAE.load_from_folder(dir_path)

        os.remove(os.path.join(dir_path, "model_config.json"))

        # check raises encoder.pkl is missing
        with pytest.raises(FileNotFoundError):
            model_rec = BaseVAE.load_from_folder(dir_path)
