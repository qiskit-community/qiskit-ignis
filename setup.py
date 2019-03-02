# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

from setuptools import setup, find_packages


requirements = [
    "numpy>=1.13",
    "qiskit-terra>=0.7.0",
    "scipy>=0.19,!=0.19.1",
]

def find_qiskit_ignis_packages():
    location = 'qiskit/ignis'
    prefix = 'qiskit.ignis'
    ignis_packages = find_packages(where=location, exclude=['test*'])
    pkg_list = list(
        map(lambda package_name: '{}.{}'.format(prefix, package_name),
            ignis_packages)
    )
    return pkg_list


setup(
    name="qiskit-ignis",
    version="0.1.0",
    description="Qiskit tools for quantum information science",
    url="https://github.com/Qiskit/qiskit-ignis",
    author="Qiskit Development Team",
    author_email="qiskit@qiskit.org",
    license="Apache 2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit sdk quantum",
    packages=find_qiskit_ignis_packages(),
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.5"
)
