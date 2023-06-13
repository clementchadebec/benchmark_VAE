import os

import pytest

from pythae.models import AutoModel

PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture()
def corrupted_config():
    return os.path.join(PATH, "data", "corrupted_config")


def test_raises_file_not_found():

    with pytest.raises(FileNotFoundError):
        AutoModel.load_from_folder("wrong_file_dir")


def test_raises_name_error(corrupted_config):
    with pytest.raises(NameError):
        AutoModel.load_from_folder(corrupted_config)
