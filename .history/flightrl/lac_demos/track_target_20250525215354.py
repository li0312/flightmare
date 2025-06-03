'''
Author: Lac_Creeper
Date: 2025-05-25 21:33:19 +0800
LastEditTime: 2025-05-25 21:49:09 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /flightmare/flightrl/lac_demos/track_target.py
'''
from ruamel.yaml import YAML, dump, RoundTripDumper
from gymnasium.utils import seeding

import os
import argparse
import numpy as np
import torch as tr

from lac_baselines.envs import vec_env_wrapper as wrapper

from flightgym import TrackEnv_v1

def configure_random_seed(seed, env=None):
  if env is not None:
    env._np_random, env._np_random_seed = seeding.np_random(seed)
  np.random.seed(seed)
  tr.manual_seed(seed)
  tr.cuda.manual_seed(seed)
  # tr.backends.cudnn.deterministic = True  # 确保CUDA卷积运算结果确定
  # tr.backends.cudnn.benchmark = False     # 关闭自动优化（避免随机性）

def parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--train', type=int, default=1,
                      help="To train new model or simply test pre-trained model")
  parser.add_argument('--render', type=int, default=0,
                      help="Enable Unity Render")
  parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)),
                      help="Directory where to save the checkpoints and training metrics")
  parser.add_argument('--seed', type=int, default=0,
                      help="Random seed")
  parser.add_argument('-w', '--weight', type=str, default='./saved/track_env.zip',
                      help='trained weight path')
  return parser