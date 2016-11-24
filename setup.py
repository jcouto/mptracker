#!/usr/bin/env python
# Install script for mouselab pupil tracker
# Joao Couto - November 2016

import os
from setuptools import setup

longdescription = '''Mouse pupil tracker GUI and tools.'''

setup(
    name = 'mptracker',
    version = '0.1',
    author = 'Joao Couto',
    author_email = 'jpcouto@gmail.com',
    description = (' Mouse pupil tracker'),
    long_description = longdescription,
    license = 'GPL',
    packages = ['mptracker'],
    install_requires=[
          'mpi4py',
          'tifffile'
      ],
    entry_points = {
        'console_scripts': [
            'mptracker-gui = mptracker.gui:main',
            ]
        }
    )
