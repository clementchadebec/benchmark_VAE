from ..models import BaseAE
from ..samplers import BaseSampler
from .base_pipeline import Pipeline


class GenerationPipeline(Pipeline):
    """
    This pipelines allows to generate new samples from a pre-trained model

    Parameters:
        model (BaseAE): The model you want ot generate from
        sampler (BaseSampler): The sampler to use to sampler from the model

    .. warning::
        You must ensure that the sampler used handled the model provided

    .. tip::
        remember that a model can be easily reloaded from a folder using
        :class:`~pythae.models.BaseAE.load_from_folder`.

        Example for a :class:`~pythae.models.RHVAE`

        .. code-block:: python

            >>> from pythae.models import RHVAE
            >>> model_rec = RHVAE.load_from_folder('path/to_model_folder')
    """

    def __init__(self, model: BaseAE, sampler: BaseSampler = None):

        self.model = model
        self.sampler = sampler

    def __call__(self, samples_number):
        """Launch the data generation and save it in ``output_dir`` stated in the
        :class:`~pythae.base.BaseSamplerConfig`. A folder ``generation_YYYY-MM-DD_hh-mm-ss`` is
        created and data is saved in ``.pt`` files in this created folder. If ``output_dir`` is
        None, data is saved in ``dummy_output_dir/generation_YYYY-MM-DD_hh-mm-ss``

        Args:

            samples_number (int): The number of samples to generate
            """
        self.sampler.sample(samples_number)
