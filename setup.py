# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os
import inspect
import setuptools
import sys


requirements = [
    "numpy>=1.13",
    "qiskit-terra>=0.13.0",
    "retworkx>=0.8.0",
    "scipy>=0.19,!=0.19.1",
    "setuptools>=40.1.0",
]


if not hasattr(setuptools,
               'find_namespace_packages') or not inspect.ismethod(
                    setuptools.find_namespace_packages):
    print("Your setuptools version:'{}' does not support PEP 420 "
          "(find_namespace_packages). Upgrade it to version >='40.1.0' and "
          "repeat install.".format(setuptools.__version__))
    sys.exit(1)


version_path = os.path.abspath(
    os.path.join(
        os.path.join(
            os.path.join(os.path.dirname(__file__), 'qiskit'), 'ignis'),
        'VERSION.txt'))
with open(version_path, 'r') as fd:
    version = fd.read().rstrip()

README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'README.md')
with open(README_PATH) as readme_file:
    README = readme_file.read()


setuptools.setup(
    name="qiskit-ignis",
    version=version,
    description="Qiskit tools for quantum information science",
    long_description=README,
    long_description_content_type='text/markdown',
    url="https://github.com/Qiskit/qiskit-ignis",
    author="Qiskit Development Team",
    author_email="hello@qiskit.org",
    license="Apache 2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit sdk quantum",
    packages=setuptools.find_namespace_packages(exclude=['test*']),
    extras_require={
        'visualization': ['matplotlib>=2.1'],
        'cvx': ['cvxpy>=1.0.15'],
        'iq': ["scikit-learn>=0.17"],
        'jit': ['numba'],
    },
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.6",
    zip_safe=False
)
