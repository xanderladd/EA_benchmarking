#!/usr/bin/env python3

import io
import os
import re

from setuptools import find_packages, setup
import setuptools.command.build_py


# Get version
def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


readme = open("README.md").read()
version = 0



# Run the setup
setup(
    name="benchmark_ea",
    version=version,
    description="package to bechmark the evolutionary algorithm for fitting neuronal models",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Alexander Ladd",
    url="https://github.com/xanderladd/EA_benchmarking",
    author_email="zladd@berkeley.edu",
    project_urls={
        "Documentation": "https://ea-benchmarking.readthedocs.io",
        "Source": "https://github.com/xanderladd/EA_benchmarking",
    },
    # packages=find_packages(exclude=["test", "test.*"]),
    python_requires=">=3.6",
    

    install_requires=[
        'bluepyopt',
        'mpi4py',
        'h5py',
        'scipy',
        'ipfx',
        'NEURON'
    ],
        

    test_suite="test",
    include_package_data=True
  
)