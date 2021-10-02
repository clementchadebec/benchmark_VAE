from pydantic.dataclasses import dataclass

from pyraug.config import BaseConfig

@dataclass
class BaseSamplerConfig(BaseConfig):
    """
    This is the base configuration of a model sampler

    Parameters:
        samples_number (int): The number of samples to generate
        batch_size (int): The number of samples to generate in each batch
        samples_per_save (int): The number of samples to be saved together.
            By default, when generating, the generated data is saved in ``.pt`` format
            in several files. This specifies the number of samples to be saved in these
            files. Amend this argument if you deal with particularly large data. Default: 500.

        no_cuda (bool): Disable `cuda`. Default: False
    """

    output_dir: str = None
    batch_size: int = 50
    samples_per_save: int = 500
    no_cuda: bool = False
