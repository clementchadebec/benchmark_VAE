import random

import numpy as np
import torch


def set_seed(seed: int):
    """
    Functions setting the seed for reproducibility on ``random``, ``numpy``,
    and ``torch``

    Args:

        seed (int): The seed to be applied
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
