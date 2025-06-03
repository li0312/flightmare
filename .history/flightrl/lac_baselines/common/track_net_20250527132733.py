'''
Author: Lac_Creeper
Date: 2025-05-27 12:03:16 +0800
LastEditTime: 2025-05-27 13:25:41 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /flightmare/flightrl/lac_baselines/common/track_net.py
'''
import gym.spaces
import gym.spaces.box
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym


class TrackCNN(BaseFeaturesExtractor):
  def __init__(self, observation_space: gym.spaces.Dict, features_dim = 128):
    lidar_shape = observation_space['lidar'].shape  # (512, 3)
    detect_shape = observation_space['detect'].shape  # (3,)
    state_shape = observation_space['state'].shape  # (3,)
    super().__init__(observation_space, features_dim)

    self.net = nn.Sequential(
      nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(128*32, 256),
      nn.ReLU())
    
    self.fc = nn.Sequential(
      nn.Linear(in_features=256+3+3, out_features=features_dim),
      nn.ReLU())
    
    
    
  def forward(self, obs):
    obs_lidar = obs['lidar']
    obs_detect = obs['detect']
    obs_state = obs['state']

    lidar = self.net(obs_lidar)
    x = torch.cat(tensors=(lidar, obs_detect, obs_state), dim=1)
    final_out = self.fc(x)

    return final_out

# TEST
import numpy as np

obs_space = gym.spaces.Dict({
  'lidar': gym.spaces.Box(low=-0.5, high=0.5, shape=(3, 512), dtype=np.float32), 
  'detect': gym.spaces.Box(low=-0.5, high=0.5, shape=(3,), dtype=np.float32), 
  'state': gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
})

# obs = {
#     'lidar': torch.randn(1, 3, 512).cuda().float(),
#     'detect': torch.randn(1, 3).cuda().float(),
#     'state': torch.randn(1, 3).cuda().float()
# }
obs = {}
obs['lidar'] = torch.randn(1, 3, 512).cuda()
obs['detect'] = torch.randn(1, 3).cuda()
obs['state'] = torch.randn(1, 3).cuda()

track_net = TrackCNN(observation_space=obs_space, features_dim=128).cuda()
output_features = track_net.forward(obs)
