#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019  Oplatai.com (author: Ondrej Platek)
# Copyright 2019  Brno University of Technology (author: Karel Vesely)

# Licensed under the Apache License, Version 2.0 (the "License")

import setuptools

# Python 2 and 3: alternative 2
from io import open

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='kaldi_io',
    version='0.9.8',
    author='Karel Vesely',
    description='Glue code connecting Kaldi data and Python.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/vesis84/kaldi-io-for-python',
    install_requires=[ 'numpy>=1.15.3', ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
)
