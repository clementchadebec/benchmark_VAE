import os
from copy import deepcopy

import dill
import pytest
import torch

from pyraug.config import BaseConfig
from pyraug.customexception import BadInheritanceError
from pyraug.models import RHVAE
from pyraug.models.nn.default_architectures import Decoder_MLP, Encoder_MLP, Metric_MLP
from pyraug.models.rhvae.rhvae_config import RHVAEConfig
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


# class Test_Model_Saving:
#    def test_save_default_model(self, tmpdir, demo_data, custom_config_paths):
#        model = train_my_model(
#            demo_data,
#            path_to_model_config=custom_config_paths[0],
#            path_to_training_config=custom_config_paths[1],
#            output_model=True,
#        )
#
#        dir_path = os.path.join(tmpdir, "dummy_saving")
#
#        # save model
#        model.save(path_to_save_model=dir_path)
#
#        model_dict = {
#            "M": deepcopy(model.M_tens),
#            "centroids": deepcopy(model.centroids_tens),
#            "model_state_dict": deepcopy(model.state_dict()),
#        }
#
#        assert set(os.listdir(dir_path)) == set(
#            ["model_config.json", "model.pt"]
#        ), f"{os.listdir(dir_path)}"
#
#        rec_model_dict = torch.load(os.path.join(dir_path, "model.pt"))
#
#        ## check model_state_dict
#        assert torch.equal(model_dict["M"], rec_model_dict["M"])
#        assert torch.equal(model_dict["centroids"], rec_model_dict["centroids"])
#
#        assert (
#            sum(
#                [
#                    not torch.equal(
#                        rec_model_dict["model_state_dict"][key],
#                        model_dict["model_state_dict"][key],
#                    )
#                    for key in model_dict["model_state_dict"].keys()
#                ]
#            )
#            == 0
#        )
#
#        ## check model and training configs
#        parser = ConfigParserFromJSON()
#        rec_model_config = parser.parse_model(
#            os.path.join(dir_path, "model_config.json")
#        )
#
#        assert rec_model_config.__dict__ == model.model_config.__dict__
#
#    def test_save_default_model(
#        self, tmpdir, demo_data, custom_encoder, custom_decoder, custom_config_paths
#    ):
#        model = train_my_model(
#            demo_data,
#            path_to_model_config=custom_config_paths[0],
#            path_to_training_config=custom_config_paths[1],
#            output_model=True,
#            encoder=custom_encoder,
#            decoder=custom_decoder,
#        )
#
#        dir_path = os.path.join(tmpdir, "dummy_saving")
#
#        # save model
#        model.save(path_to_save_model=dir_path)
#
#        model_dict = {
#            "M": deepcopy(model.M_tens),
#            "centroids": deepcopy(model.centroids_tens),
#            "model_state_dict": deepcopy(model.state_dict()),
#        }
#
#        assert set(os.listdir(dir_path)) == set(
#            ["model_config.json", "model.pt", "decoder.pkl", "encoder.pkl"]
#        ), f"{os.listdir(dir_path)}"
#
#        rec_model_dict = torch.load(os.path.join(dir_path, "model.pt"))
#
#        ## check model_state_dict
#        assert torch.equal(model_dict["M"], rec_model_dict["M"])
#        assert torch.equal(model_dict["centroids"], rec_model_dict["centroids"])
#
#        assert (
#            sum(
#                [
#                    not torch.equal(
#                        rec_model_dict["model_state_dict"][key],
#                        model_dict["model_state_dict"][key],
#                    )
#                    for key in model_dict["model_state_dict"].keys()
#                ]
#            )
#            == 0
#        )
#
#        ## check model and training configs
#        parser = ConfigParserFromJSON()
#        rec_model_config = parser.parse_model(
#            os.path.join(dir_path, "model_config.json")
#        )
#
#        assert rec_model_config.__dict__ == model.model_config.__dict__
#
#        ## check custom encoder and decoder
#        with open(os.path.join(dir_path, "encoder.pkl"), "rb") as fp:
#            rec_encoder = dill.load(fp)
#        with open(os.path.join(dir_path, "decoder.pkl"), "rb") as fp:
#            rec_decoder = dill.load(fp)
#
#        assert type(rec_encoder) == type(model.encoder)
#        assert type(rec_decoder) == type(model.decoder)
