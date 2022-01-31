import pathlib

import pkg_resources
from setuptools import find_packages, setup

with pathlib.Path("requirements.txt").open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement in pkg_resources.parse_requirements(requirements_txt)
    ]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pythae",
    version="0.0.1",
    author="Clement Chadebec (HekA team INRIA)",
    author_email="clement.chadebec@inria.fr",
    description="Data Augmentation with VAE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/clementchadebec/pythae",
    project_urls={"Bug Tracker": "https://github.com/clementchadebec/pythae/issues"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.7.1",
        "torch==1.9.0",
        "dill>=0.3.3",
        "pydantic>=1.8.2",
        "dataclasses>=0.6",
        "tqdm>=4.62.3",
        "imageio",
        "sklearn",
        "typing-extensions",
    ],
    python_requires=">=3.6",
)
