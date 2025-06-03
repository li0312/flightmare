'''
Author: Lac_Creeper
Date: 2025-05-25 21:33:19 +0800
LastEditTime: 2025-05-25 21:42:56 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /flightmare/flightrl/lac_demos/track_target.py
'''
from ruamel.yaml import YAML, dump, RoundTripDumper

import os
import argparse
import seeding
import numpy as np
import torch as tr

from lac_baselines.envs import vec_env_wrapper as wrapper

from flightgym import TrackEnv_v1

def configure_random_seed(seed, env=None):
  if env is not None:
    env._np_random, env._np_random_seed = seeding.np_random(seed)