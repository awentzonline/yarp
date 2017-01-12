#!/usr/bin/env python
from distutils.core import setup


setup(
    name='yarp',
    version='0.0.1',
    description='Yet Another Reinforcement Learning Package',
    author='Adam Wentz',
    author_email='adam@adamwentz.com',
    url='https://github.com/awentzonline/yarp',
    packages=[
        'yarp',
    ],
    install_requires=[
        'keras',
        'numpy',
        'tqdm',
    ]
)
