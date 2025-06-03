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
    install_requires=['gymnasium==1.1.1', 'ruamel.yaml',
                      'numpy', 'stable_baselines3==2.6.0'],
    packages=['lac_baselines', 'lac_baselines.common', 'lac_baselines.envs'],
)
