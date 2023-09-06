from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pythae",
    version="0.1.2",
    author="Clement Chadebec (HekA team INRIA)",
    author_email="clement.chadebec@inria.fr",
    description="Unifying Generative Autoencoders in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/clementchadebec/benchmark_VAE",
    project_urls={
        "Bug Tracker": "https://github.com/clementchadebec/benchmark_VAE/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "cloudpickle>=2.1.0",
        "imageio",
        "numpy>=1.19",
        "pydantic>=2.0",
        "scikit-learn",
        "scipy>=1.7.1",
        "torch>=1.10.1",
        "tqdm",
        "typing_extensions",
        "dataclasses>=0.6",
    ],
    extras_require={':python_version == "3.7.*"': ["pickle5"]},
    python_requires=">=3.7",
)
