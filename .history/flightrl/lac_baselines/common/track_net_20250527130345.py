'''
Author: Lac_Creeper
Date: 2025-05-27 12:03:16 +0800
LastEditTime: 2025-05-27 13:02:00 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /flightmare/flightrl/lac_baselines/common/track_net.py
'''
import gym.spaces
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym


class TrackCNN(BaseFeaturesExtractor):
  def __init__(self, observation_space: gym.spaces.Dict, features_dim = 128):
    super().__init__(observation_space, features_dim)

    self.net = nn.Sequential(
      nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(128*32, 256),
      nn.ReLU())
    
    self.fc1 = nn.Linear(in_features=256+3+3, out_features=features_dim)
    
    
  def forward(self, obs):
    obs_lidar = obs['lidar']
    obs_detect = obs['detect']
    obs_state = obs['state']

    x = self.net(obs_lidar)
    x = torch.cat(tensors=(x, obs_detect, obs_state), dim=-1)
    final_out = torch.relu(self.fc1(x))

    return final_out

# TEST
import numpy as np

obs = gym