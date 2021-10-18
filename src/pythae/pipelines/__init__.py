"""The Pipelines module is created to facilitate the use of the library. It provides ways to
perform end-to-end operation such as model training or generation. A typical Pipeline is composed by
several pythae's instances which are articulated together.

A :class:`__call__` function is defined and used to launch the Pipeline. """

from .training import TrainingPipeline

__all__ = ["TrainingPipeline"]
