"""Utils class allowing to reload any :class:`pythae.models` automatically witht the following 
lines of code.

.. code-block::

    >>> from pythae.models import AutoModel
    >>> model = AutoModel.load_from_folder(dir_path='path/to/my_model') 
"""

from .auto_model import AutoModel