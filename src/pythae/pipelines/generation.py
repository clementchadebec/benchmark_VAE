from pyraug.models import BaseVAE
from pyraug.models.base.base_sampler import BaseSampler
from pyraug.models.rhvae import RHVAESamplerConfig
from pyraug.models.rhvae.rhvae_sampler import RHVAESampler

from .base_pipeline import Pipeline


class GenerationPipeline(Pipeline):
    """
    This pipelines allows to generate new samples from a pre-trained model

    Parameters:
        model (BaseVAE): The model you want ot generate from
        sampler (BaseSampler): The sampler to use to sampler from the model

    .. warning::
        You must ensure that the sampler used handled the model provided

    .. tip::
        remember that a model can be easily reloaded from a folder using
        :class:`~pyraug.models.BaseVAE.load_from_folder`.

        Example for a :class:`~pyraug.models.RHVAE`

        .. code-block:: python

            >>> from pyraug.models import RHVAE
            >>> model_rec = RHVAE.load_from_folder('path/to_model_folder')
    """

    def __init__(self, model: BaseVAE, sampler: BaseSampler = None):

        self.model = model

        if sampler is None:
            sampler = RHVAESampler(model=model, sampler_config=RHVAESamplerConfig())

        self.sampler = sampler

    def __call__(self, samples_number):
        """Launch the data generation and save it in ``output_dir`` stated in the
        :class:`~pyraug.base.BaseSamplerConfig`. A folder ``generation_YYYY-MM-DD_hh-mm-ss`` is
        created and data is saved in ``.pt`` files in this created folder. If ``output_dir`` is
        None, data is saved in ``dummy_output_dir/generation_YYYY-MM-DD_hh-mm-ss``

        Args:

            samples_number (int): The number of samples to generate
            """
        self.sampler.sample(samples_number)
