import importlib
import io
import logging
from collections import OrderedDict
from typing import Any, Tuple

try:
    import pickle5 as pickle
except:
    import pickle
import torch

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

model_card_template = """---
language: en
tags:
- pythae
license: apache-2.0
---

### Downloading this model from the Hub
This model was trained with pythae. It can be downloaded or reloaded using the method `load_from_hf_hub`
```python
>>> from pythae.models import AutoModel
>>> model = AutoModel.load_from_hf_hub(hf_hub_path="your_hf_username/repo_name")
```
"""


def hf_hub_is_available():
    return importlib.util.find_spec("huggingface_hub") is not None


class ModelOutput(OrderedDict):
    """Base ModelOutput class fixing the output type from the models. This class is inspired from
    the ``ModelOutput`` class from hugginface transformers library"""

    def __getitem__(self, k):
        if isinstance(k, str):
            self_dict = {k: v for (k, v) in self.items()}
            return self_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)
