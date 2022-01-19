import pytest
import torch
import numpy as np

from pythae.models import AEConfig, VAEConfig, GMVAEConfig
from pythae.models.nn.benchmarks.mnist import *
from pythae.models.nn.benchmarks.celeba import *
from pythae.models.nn.benchmarks.cifar import *
from pythae.models.nn.default_architectures import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Test_MNIST_Benchmark:
    @pytest.fixture(
    params=[
        AEConfig(input_dim=(1, 28, 28), latent_dim=10),
        AEConfig(input_dim=(1, 28, 28), latent_dim=5),
    ]
    )
    def ae_mnist_config(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [1],
            None
        ]
    )
    def recon_layers_default(self, request):
        return request.param

    @pytest.fixture()
    def mnist_like_data(self):
        return torch.rand(3, 1, 28, 28).to(device)

    def test_ae_encoding_decoding_default(
        self, ae_mnist_config, mnist_like_data, recon_layers_default):
        encoder = Encoder_AE_MLP(ae_mnist_config).to(device)
        decoder = Decoder_AE_MLP(ae_mnist_config).to(device)

        embedding = encoder(mnist_like_data).embedding
    
        assert embedding.shape == (mnist_like_data.shape[0], ae_mnist_config.latent_dim)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == mnist_like_data.shape

        encoder_embed = encoder(mnist_like_data, output_layer_levels=recon_layers_default)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers_default)

        if recon_layers_default is None:
            assert 'embedding' in encoder_embed.keys()

        else:
            for lev in recon_layers_default:
                assert f'embedding_layer_{lev}' in encoder_embed.keys()

        if recon_layers_default is None:
            assert encoder_embed['embedding'].shape[1] == ae_mnist_config.latent_dim

        else:
            if 1 in recon_layers_default:
                assert encoder_embed[f'embedding_layer_1'].shape[1] == 512
                assert decoder_recon[f'reconstruction_layer_1'].shape[1] == 512

    def test_discriminator_default(self, ae_mnist_config, mnist_like_data, recon_layers_default):
        
        ae_mnist_config.discriminator_input_dim = (1, 28, 28)

        discriminator = Discriminator_MLP(ae_mnist_config).to(device)

        scores = discriminator(mnist_like_data, output_layer_levels=recon_layers_default)

        if recon_layers_default is None:
            assert 'embedding' in scores.keys()

        else:
            for lev in recon_layers_default:
                assert f'embedding_layer_{lev}' in scores.keys()

        if recon_layers_default is None:
            assert scores['embedding'].shape[1] == 1

        else:
            if 1 in recon_layers_default:
                assert scores[f'embedding_layer_1'].shape[1] == 256

            if 2 in recon_layers_default:
                assert scores[f'embedding_layer_2'].shape[1] == 1

    @pytest.fixture(
    params=[
            [3, 4],
            [np.random.randint(1, 5)],
            [1, 2, 4],
            None
        ]
    )
    def recon_layers_benchmark(self, request):
        return request.param

    def test_ae_encoding_decoding_benchmark(
        self, ae_mnist_config, mnist_like_data, recon_layers_benchmark):
        encoder = Encoder_AE_MNIST(ae_mnist_config).to(device)
        decoder = Decoder_AE_MNIST(ae_mnist_config).to(device)

        embedding = encoder(mnist_like_data).embedding
    
        assert embedding.shape == (mnist_like_data.shape[0], ae_mnist_config.latent_dim)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == mnist_like_data.shape

        encoder_embed = encoder(mnist_like_data, output_layer_levels=recon_layers_benchmark)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers_benchmark)

        if recon_layers_benchmark is None:
            assert 'embedding' in encoder_embed.keys()

        else:
            for lev in recon_layers_benchmark:
                assert f'embedding_layer_{lev}' in encoder_embed.keys()

        if recon_layers_benchmark is None:
            assert encoder_embed['embedding'].shape[1] == ae_mnist_config.latent_dim

        else:
            if 1 in recon_layers_benchmark:
                assert encoder_embed[f'embedding_layer_1'].shape[1] == 128
                assert decoder_recon[f'reconstruction_layer_1'].shape[1] == 1024

            if 2 in recon_layers_benchmark:
                assert encoder_embed[f'embedding_layer_2'].shape[1] == 256
                assert decoder_recon[f'reconstruction_layer_2'].shape[1] == 512

            if 3 in recon_layers_benchmark:
                assert encoder_embed[f'embedding_layer_3'].shape[1] == 512
                assert decoder_recon[f'reconstruction_layer_3'].shape[1] == 256

            if 4 in recon_layers_benchmark:
                assert encoder_embed[f'embedding_layer_4'].shape[1] == 1024
                assert decoder_recon[f'reconstruction_layer_4'].shape[1:] == ae_mnist_config.input_dim

    def test_discriminator_benchmark(
        self, ae_mnist_config, mnist_like_data, recon_layers_benchmark):
        
        ae_mnist_config.discriminator_input_dim = (1, 28, 28)

        discriminator = Discriminator_MNIST(ae_mnist_config).to(device)

        scores = discriminator(mnist_like_data, output_layer_levels=recon_layers_benchmark)

        if recon_layers_benchmark is None:
            assert 'embedding' in scores.keys()

        else:
            for lev in recon_layers_benchmark:
                assert f'embedding_layer_{lev}' in scores.keys()

        if recon_layers_benchmark is None:
            assert scores['embedding'].shape[1] == 1

        else:
            if 1 in recon_layers_benchmark:
                assert scores[f'embedding_layer_1'].shape[1] == 128

            if 2 in recon_layers_benchmark:
                assert scores[f'embedding_layer_2'].shape[1] == 256

            if 3 in recon_layers_benchmark:
                assert scores[f'embedding_layer_3'].shape[1] == 512

            if 4 in recon_layers_benchmark:
                assert scores[f'embedding_layer_4'].shape[1] == 1024

            if 5 in recon_layers_benchmark:
                assert scores[f'embedding_layer_5'].shape[1] == 1


    @pytest.fixture(
    params=[
        VAEConfig(input_dim=(1, 28, 28), latent_dim=10),
        VAEConfig(input_dim=(1, 28, 28), latent_dim=5),
    ]
    )
    def vae_mnist_config(self, request):
        return request.param

    def test_vae_encoding_decoding(self, vae_mnist_config, mnist_like_data, recon_layers_benchmark):
        encoder = Encoder_VAE_MNIST(vae_mnist_config).to(device)
        decoder = Decoder_AE_MNIST(vae_mnist_config).to(device)

        output = encoder(mnist_like_data)
        embedding, log_covariance = output.embedding, output.log_covariance
    
        assert embedding.shape == (mnist_like_data.shape[0], vae_mnist_config.latent_dim)
        assert log_covariance.shape == (mnist_like_data.shape[0], vae_mnist_config.latent_dim)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == mnist_like_data.shape

        encoder_embed = encoder(mnist_like_data, output_layer_levels=recon_layers_benchmark)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers_benchmark)

        if recon_layers_benchmark is None:
            assert 'embedding' in encoder_embed.keys()

        else:
            for lev in recon_layers_benchmark:
                assert f'embedding_layer_{lev}' in encoder_embed.keys()

        if recon_layers_benchmark is None:
            assert encoder_embed['embedding'].shape[1] == vae_mnist_config.latent_dim

        else:
            if 1 in recon_layers_benchmark:
                assert encoder_embed[f'embedding_layer_1'].shape[1] == 128
                assert decoder_recon[f'reconstruction_layer_1'].shape[1] == 1024

            if 2 in recon_layers_benchmark:
                assert encoder_embed[f'embedding_layer_2'].shape[1] == 256
                assert decoder_recon[f'reconstruction_layer_2'].shape[1] == 512

            if 3 in recon_layers_benchmark:
                assert encoder_embed[f'embedding_layer_3'].shape[1] == 512
                assert decoder_recon[f'reconstruction_layer_3'].shape[1] == 256

            if 4 in recon_layers_benchmark:
                assert encoder_embed[f'embedding_layer_4'].shape[1] == 1024
                assert decoder_recon[f'reconstruction_layer_4'].shape[1:] == vae_mnist_config.input_dim


    @pytest.fixture(
    params=[
        GMVAEConfig(input_dim=(1, 28, 28), latent_dim=10),
        GMVAEConfig(input_dim=(1, 28, 28), latent_dim=5, w_prior_latent_dim=7),
    ]
    )
    def gmvae_mnist_config(self, request):
        return request.param

    def test_gmvae_encoding_decoding_default(
        self, gmvae_mnist_config, mnist_like_data, recon_layers_default):
        encoder = Encoder_GMVAE_MLP(gmvae_mnist_config).to(device)

        encoder_output = encoder(mnist_like_data)
    
        embedding_z = encoder_output.embedding_z
        log_var_z = encoder_output.log_covariance_z
        embedding_w = encoder_output.embedding_w
        log_var_w = encoder_output.log_covariance_w
    
        assert embedding_z.shape == (mnist_like_data.shape[0], gmvae_mnist_config.latent_dim)
        assert log_var_z.shape == (mnist_like_data.shape[0], gmvae_mnist_config.latent_dim)
        assert embedding_w.shape == (mnist_like_data.shape[0], gmvae_mnist_config.w_prior_latent_dim)
        assert log_var_w.shape == (mnist_like_data.shape[0], gmvae_mnist_config.w_prior_latent_dim)
        
        encoder_embed = encoder(mnist_like_data, output_layer_levels=recon_layers_default)

        if recon_layers_default is None:
            assert set([
                'embedding_z', 'log_covariance_z', 'embedding_w', 'log_covariance_w'
                ]).issubset(set(encoder_embed.keys()))

        else:
            for lev in recon_layers_default:
                assert f'embedding_layer_{lev}' in encoder_embed.keys()

        if recon_layers_default is None:
            assert encoder_embed['embedding_z'].shape[1] == gmvae_mnist_config.latent_dim
            assert encoder_embed['log_covariance_z'].shape[1] == gmvae_mnist_config.latent_dim
            assert encoder_embed['embedding_w'].shape[1] == gmvae_mnist_config.w_prior_latent_dim
            assert encoder_embed['log_covariance_w'].shape[1] == gmvae_mnist_config.w_prior_latent_dim

        else:
            if 1 in recon_layers_default:
                assert encoder_embed[f'embedding_layer_1'].shape[1] == 512

    def test_gmvae_encoding_decoding_benchmark(self, gmvae_mnist_config, mnist_like_data, recon_layers_benchmark):
        encoder = Encoder_GMVAE_MNIST(gmvae_mnist_config).to(device)

        encoder_output = encoder(mnist_like_data)
    
        embedding_z = encoder_output.embedding_z
        log_var_z = encoder_output.log_covariance_z
        embedding_w = encoder_output.embedding_w
        log_var_w = encoder_output.log_covariance_w
    
        assert embedding_z.shape == (mnist_like_data.shape[0], gmvae_mnist_config.latent_dim)
        assert log_var_z.shape == (mnist_like_data.shape[0], gmvae_mnist_config.latent_dim)
        assert embedding_w.shape == (mnist_like_data.shape[0], gmvae_mnist_config.w_prior_latent_dim)
        assert log_var_w.shape == (mnist_like_data.shape[0], gmvae_mnist_config.w_prior_latent_dim)

        encoder_embed = encoder(mnist_like_data, output_layer_levels=recon_layers_benchmark)

        if recon_layers_benchmark is None:
            assert set([
                'embedding_z', 'log_covariance_z', 'embedding_w', 'log_covariance_w'
                ]).issubset(set(encoder_embed.keys()))

        else:
            for lev in recon_layers_benchmark:
                assert f'embedding_layer_{lev}' in encoder_embed.keys()

        if recon_layers_benchmark is None:
            assert encoder_embed['embedding_z'].shape[1] == gmvae_mnist_config.latent_dim
            assert encoder_embed['log_covariance_z'].shape[1] == gmvae_mnist_config.latent_dim
            assert encoder_embed['embedding_w'].shape[1] == gmvae_mnist_config.w_prior_latent_dim
            assert encoder_embed['log_covariance_w'].shape[1] == gmvae_mnist_config.w_prior_latent_dim

        else:
            if 1 in recon_layers_benchmark:
                assert encoder_embed[f'embedding_layer_1'].shape[1] == 128

            if 2 in recon_layers_benchmark:
                assert encoder_embed[f'embedding_layer_2'].shape[1] == 256

            if 3 in recon_layers_benchmark:
                assert encoder_embed[f'embedding_layer_3'].shape[1] == 512

            if 4 in recon_layers_benchmark:
                assert encoder_embed[f'embedding_layer_4'].shape[1] == 1024


class Test_CIFAR_Benchmark:
    @pytest.fixture(
    params=[
        AEConfig(input_dim=(3, 32, 32), latent_dim=10),
        AEConfig(input_dim=(3, 32, 32), latent_dim=5),
    ]
    )
    def ae_cifar_config(self, request):
        return request.param

    @pytest.fixture()
    def cifar_like_data(self):
        return torch.rand(3, 3, 32, 32).to(device)

    @pytest.fixture(
        params=[
            [3, 4],
            [np.random.randint(1, 5)],
            [1, 2, 4],
            None
        ]
    )
    def recon_layers(self, request):
        return request.param

    @pytest.fixture()
    def cifar_like_data(self):
        return torch.rand(3, 3, 32, 32).to(device)

    def test_ae_encoding_decoding(self, ae_cifar_config, cifar_like_data, recon_layers):
        encoder = Encoder_AE_CIFAR(ae_cifar_config).to(device)
        decoder = Decoder_AE_CIFAR(ae_cifar_config).to(device)

        embedding = encoder(cifar_like_data).embedding
    
        assert embedding.shape == (cifar_like_data.shape[0], ae_cifar_config.latent_dim)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == cifar_like_data.shape

        encoder_embed = encoder(cifar_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert 'embedding' in encoder_embed.keys()

        else:
            for lev in recon_layers:
                assert f'embedding_layer_{lev}' in encoder_embed.keys()

        if recon_layers is None:
            assert encoder_embed['embedding'].shape[1] == ae_cifar_config.latent_dim

        else:
            if 1 in recon_layers:
                assert encoder_embed[f'embedding_layer_1'].shape[1] == 128
                assert decoder_recon[f'reconstruction_layer_1'].shape[1] == 1024

            if 2 in recon_layers:
                assert encoder_embed[f'embedding_layer_2'].shape[1] == 256
                assert decoder_recon[f'reconstruction_layer_2'].shape[1] == 512

            if 3 in recon_layers:
                assert encoder_embed[f'embedding_layer_3'].shape[1] == 512
                assert decoder_recon[f'reconstruction_layer_3'].shape[1] == 256

            if 4 in recon_layers:
                assert encoder_embed[f'embedding_layer_4'].shape[1] == 1024
                assert decoder_recon[f'reconstruction_layer_4'].shape[1:] == ae_cifar_config.input_dim

            if 5 in recon_layers:
                assert encoder_embed[f'embedding_layer_5'].shape[1] == 1


    @pytest.fixture(
    params=[
        VAEConfig(input_dim=(3, 32, 32), latent_dim=10),
        VAEConfig(input_dim=(3, 32, 32), latent_dim=5),
    ]
    )
    def vae_cifar_config(self, request):
        return request.param

    def test_vae_encoding_decoding(self, vae_cifar_config, cifar_like_data, recon_layers):
        encoder = Encoder_VAE_CIFAR(vae_cifar_config).to(device)
        decoder = Decoder_AE_CIFAR(vae_cifar_config).to(device)

        output = encoder(cifar_like_data)
        embedding, log_covariance = output.embedding, output.log_covariance
    
        assert embedding.shape == (cifar_like_data.shape[0], vae_cifar_config.latent_dim)
        assert log_covariance.shape == (cifar_like_data.shape[0], vae_cifar_config.latent_dim)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == cifar_like_data.shape

        encoder_embed = encoder(cifar_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert 'embedding' in encoder_embed.keys()

        else:
            for lev in recon_layers:
                assert f'embedding_layer_{lev}' in encoder_embed.keys()

        if recon_layers is None:
            assert encoder_embed['embedding'].shape[1] == vae_cifar_config.latent_dim

        else:
            if 1 in recon_layers:
                assert encoder_embed[f'embedding_layer_1'].shape[1] == 128
                assert decoder_recon[f'reconstruction_layer_1'].shape[1] == 1024

            if 2 in recon_layers:
                assert encoder_embed[f'embedding_layer_2'].shape[1] == 256
                assert decoder_recon[f'reconstruction_layer_2'].shape[1] == 512

            if 3 in recon_layers:
                assert encoder_embed[f'embedding_layer_3'].shape[1] == 512
                assert decoder_recon[f'reconstruction_layer_3'].shape[1] == 256

            if 4 in recon_layers:
                assert encoder_embed[f'embedding_layer_4'].shape[1] == 1024
                assert decoder_recon[f'reconstruction_layer_4'].shape[1:] == vae_cifar_config.input_dim

    @pytest.fixture(
    params=[
        GMVAEConfig(input_dim=(3, 32, 32), latent_dim=10),
        GMVAEConfig(input_dim=(3, 32, 32), latent_dim=5, w_prior_latent_dim=7),
    ]
    )
    def gmvae_cifar_config(self, request):
        return request.param

    def test_gmvae_encoding_decoding_benchmark(self, gmvae_cifar_config, cifar_like_data, recon_layers):
        encoder = Encoder_GMVAE_CIFAR(gmvae_cifar_config).to(device)

        encoder_output = encoder(cifar_like_data)
    
        embedding_z = encoder_output.embedding_z
        log_var_z = encoder_output.log_covariance_z
        embedding_w = encoder_output.embedding_w
        log_var_w = encoder_output.log_covariance_w
    
        assert embedding_z.shape == (cifar_like_data.shape[0], gmvae_cifar_config.latent_dim)
        assert log_var_z.shape == (cifar_like_data.shape[0], gmvae_cifar_config.latent_dim)
        assert embedding_w.shape == (cifar_like_data.shape[0], gmvae_cifar_config.w_prior_latent_dim)
        assert log_var_w.shape == (cifar_like_data.shape[0], gmvae_cifar_config.w_prior_latent_dim)

        encoder_embed = encoder(cifar_like_data, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert set([
                'embedding_z', 'log_covariance_z', 'embedding_w', 'log_covariance_w'
                ]).issubset(set(encoder_embed.keys()))

        else:
            for lev in recon_layers:
                assert f'embedding_layer_{lev}' in encoder_embed.keys()

        if recon_layers is None:
            assert encoder_embed['embedding_z'].shape[1] == gmvae_cifar_config.latent_dim
            assert encoder_embed['log_covariance_z'].shape[1] == gmvae_cifar_config.latent_dim
            assert encoder_embed['embedding_w'].shape[1] == gmvae_cifar_config.w_prior_latent_dim
            assert encoder_embed['log_covariance_w'].shape[1] == gmvae_cifar_config.w_prior_latent_dim

        else:
            if 1 in recon_layers:
                assert encoder_embed[f'embedding_layer_1'].shape[1] == 128

            if 2 in recon_layers:
                assert encoder_embed[f'embedding_layer_2'].shape[1] == 256

            if 3 in recon_layers:
                assert encoder_embed[f'embedding_layer_3'].shape[1] == 512

            if 4 in recon_layers:
                assert encoder_embed[f'embedding_layer_4'].shape[1] == 1024

    


class Test_CELEBA_Benchmark:
    @pytest.fixture(
    params=[
        AEConfig(input_dim=(3, 64, 64), latent_dim=10),
        AEConfig(input_dim=(3, 64, 64), latent_dim=5),
    ]
    )
    def ae_celeba_config(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [3, 4],
            [np.random.randint(1, 5)],
            [1, 2, 4],
            None
        ]
    )
    def recon_layers(self, request):
        return request.param

    @pytest.fixture()
    def celeba_like_data(self):
        return torch.rand(3, 3, 64, 64).to(device)

    def test_ae_encoding_decoding(self, ae_celeba_config, celeba_like_data, recon_layers):
        encoder = Encoder_AE_CELEBA(ae_celeba_config).to(device)
        decoder = Decoder_AE_CELEBA(ae_celeba_config).to(device)

        embedding = encoder(celeba_like_data).embedding
    
        assert embedding.shape == (celeba_like_data.shape[0], ae_celeba_config.latent_dim)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == celeba_like_data.shape

        encoder_embed = encoder(celeba_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert 'embedding' in encoder_embed.keys()

        else:
            for lev in recon_layers:
                assert f'embedding_layer_{lev}' in encoder_embed.keys()

        if recon_layers is None:
            assert encoder_embed['embedding'].shape[1] == ae_celeba_config.latent_dim

        else:
            if 1 in recon_layers:
                assert encoder_embed[f'embedding_layer_1'].shape[1] == 128
                assert decoder_recon[f'reconstruction_layer_1'].shape[1] == 1024

            if 2 in recon_layers:
                assert encoder_embed[f'embedding_layer_2'].shape[1] == 256
                assert decoder_recon[f'reconstruction_layer_2'].shape[1] == 512

            if 3 in recon_layers:
                assert encoder_embed[f'embedding_layer_3'].shape[1] == 512
                assert decoder_recon[f'reconstruction_layer_3'].shape[1] == 256

            if 4 in recon_layers:
                assert encoder_embed[f'embedding_layer_4'].shape[1] == 1024
                assert decoder_recon[f'reconstruction_layer_4'].shape[1] == 128


    def test_discriminator(self, ae_celeba_config, celeba_like_data, recon_layers):
        
        ae_celeba_config.discriminator_input_dim = (3, 64, 64)

        discriminator = Discriminator_CELEBA(ae_celeba_config).to(device)

        scores = discriminator(celeba_like_data, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert 'embedding' in scores.keys()

        else:
            for lev in recon_layers:
                assert f'embedding_layer_{lev}' in scores.keys()

        if recon_layers is None:
            assert scores['embedding'].shape[1] == 1

        else:
            if 1 in recon_layers:
                assert scores[f'embedding_layer_1'].shape[1] == 128

            if 2 in recon_layers:
                assert scores[f'embedding_layer_2'].shape[1] == 256

            if 3 in recon_layers:
                assert scores[f'embedding_layer_3'].shape[1] == 512

            if 4 in recon_layers:
                assert scores[f'embedding_layer_4'].shape[1] == 1024

            if 5 in recon_layers:
                assert scores[f'embedding_layer_5'].shape[1] == 1


    @pytest.fixture(
    params=[
        VAEConfig(input_dim=(3, 64, 64), latent_dim=10),
        VAEConfig(input_dim=(3, 64, 64), latent_dim=5),
    ]
    )
    def vae_celeba_config(self, request):
        return request.param

    def test_vae_encoding_decoding(self, vae_celeba_config, celeba_like_data, recon_layers):
        encoder = Encoder_VAE_CELEBA(vae_celeba_config).to(device)
        decoder = Decoder_AE_CELEBA(vae_celeba_config).to(device)

        output = encoder(celeba_like_data)
        embedding, log_covariance = output.embedding, output.log_covariance
    
        assert embedding.shape == (celeba_like_data.shape[0], vae_celeba_config.latent_dim)
        assert log_covariance.shape == (celeba_like_data.shape[0], vae_celeba_config.latent_dim)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == celeba_like_data.shape

        encoder_embed = encoder(celeba_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert 'embedding' in encoder_embed.keys()

        else:
            for lev in recon_layers:
                assert f'embedding_layer_{lev}' in encoder_embed.keys()

        if recon_layers is None:
            assert encoder_embed['embedding'].shape[1] == vae_celeba_config.latent_dim

        else:
            if 1 in recon_layers:
                assert encoder_embed[f'embedding_layer_1'].shape[1] == 128
                assert decoder_recon[f'reconstruction_layer_1'].shape[1] == 1024

            if 2 in recon_layers:
                assert encoder_embed[f'embedding_layer_2'].shape[1] == 256
                assert decoder_recon[f'reconstruction_layer_2'].shape[1] == 512

            if 3 in recon_layers:
                assert encoder_embed[f'embedding_layer_3'].shape[1] == 512
                assert decoder_recon[f'reconstruction_layer_3'].shape[1] == 256

            if 4 in recon_layers:
                assert encoder_embed[f'embedding_layer_4'].shape[1] == 1024
                assert decoder_recon[f'reconstruction_layer_4'].shape[1] == 128
