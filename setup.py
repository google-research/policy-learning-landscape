# IGNORE_COPYRIGHT: This is Google-owned code.
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Setup script for EagerPG.

This script will install eager_pg to make it accessible as a library externally.
"""

import codecs
from os import path
from setuptools import find_packages
from setuptools import setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file.
with codecs.open(path.join(here, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

install_requires = [
    'absl-py >= 0.2.2',
    'tensorflow',
    'gym >= 0.10.5'
]

description = (
    'eager_pg: Policy gradient components to make prototyping quick.')

setup(
    name='eager_pg',
    version='0.0.1',
    include_package_data=True,
    packages=find_packages(exclude=['docs']),  # Required
    install_requires=install_requires,
    description=description,
    long_description=long_description,
    author_email='opensource@google.com',
    classifiers=[  # Optional
        'Development Status :: 3 - Alpha'
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    license='Apache 2.0',
)
