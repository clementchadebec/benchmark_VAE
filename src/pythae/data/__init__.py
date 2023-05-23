"""This module contrains the methods to load and preprocess the data.

.. note::

    As of now, only imaging modality is handled by pythae. In the near future other
    modalities should be added.
"""

from .datasets import BaseDataset

__all__ = ["BaseDataset"]
