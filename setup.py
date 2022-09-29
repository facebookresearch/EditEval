# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="editeval",
    version="0.1.0",
    description="EditEval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "spacy",
        "tensorflow",
        "bert-score",
        "numpy",
        "sacremoses",
        "transformers",
        "scipy",
        "datasets<=2.5.1",
        "cdifflib",
        "nltk",
        "rouge_score",
        "sacrebleu",
        "protobuf==3.20.0",
    ],
)
