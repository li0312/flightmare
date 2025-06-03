'''
Author: Lac_Creeper
Date: 2025-05-27 12:03:16 +0800
LastEditTime: 2025-05-27 12:06:33 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /flightmare/flightrl/lac_baselines/common/track_net.py
'''
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TrackCNN(BaseFeaturesExtractor):
  def __init__(self, observation_space, features_dim = 0):
    super().__init__(observation_space, features_dim)

    self.net = nn.Sequential(
      nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(128*32, features_dim),
      nn.ReLU())
    
    self.action_net = nn.Sequential(
        nn.Linear(256+3+3, 128),
        nn.ReLU(),
        nn.Linear(128, action_space.shape[0]))
    
    self.value_net = nn.Sequential(
        nn.Linear(256+3+3, 128),
        nn.ReLU(),
        nn.Linear(128, 1))