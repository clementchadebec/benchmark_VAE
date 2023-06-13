"""Utils class allowing to reload any :class:`pythae.models` automatically with the following 
lines of code.

.. code-block::

    >>> from pythae.models import AutoModel
    >>> model = AutoModel.load_from_folder(dir_path='path/to/my_model') 
"""

from .auto_config import AutoConfig
from .auto_model import AutoModel
