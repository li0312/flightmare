'''
Author: Flightmare
Date: 2025-05-19 22:06:07 +0800
LastEditTime: 2025-05-27 13:29:54 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /flightmare/flightrl/setup.py
'''
import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

setup(
    name='lac_baselines',
    version='0.0.1',
    author='Yunlong Song',
    author_email='song@ifi.uzh.ch',
    description='Flightmare: A Quadrotor Simulator.',
    long_description='',
    install_requires=['gym==0.19.0', 'ruamel.yaml',
                      'numpy', 'stable_baselines3==1.4.0'],
    packages=['lac_baselines', 'lac_baselines.common', 'lac_baselines.envs'],
)
