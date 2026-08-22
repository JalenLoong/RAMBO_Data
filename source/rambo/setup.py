"""Installation script for the RAMBO Isaac Lab extension."""

from __future__ import annotations

import os

import toml
from setuptools import find_packages, setup


_EXTENSION_ROOT = os.path.dirname(os.path.realpath(__file__))
_METADATA = toml.load(os.path.join(_EXTENSION_ROOT, "config", "extension.toml"))


setup(
    name="rambo",
    version=_METADATA["package"]["version"],
    author=_METADATA["package"]["author"],
    maintainer=_METADATA["package"]["maintainer"],
    url=_METADATA["package"]["repository"],
    description=_METADATA["package"]["description"],
    keywords=_METADATA["package"]["keywords"],
    license="BSD-3-Clause",
    python_requires=">=3.11,<3.12",
    install_requires=["numpy<2", "gymnasium==1.2.0", "qpth==0.0.18"],
    packages=find_packages(include=["rambo", "rambo.*"]),
    package_data={"rambo": ["tasks/direct/*/agents/*.yaml"]},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
