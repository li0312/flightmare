'''
Author: Lac_Creeper
Date: 2025-05-27 12:03:16 +0800
LastEditTime: 2025-06-11 14:36:03 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /flightmare/flightrl/lac_baselines/common/track_net.py
'''
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class TrackCNN(BaseFeaturesExtractor):
  def __init__(self, observation_space: gym.spaces.Dict, features_dim = 128):
    super().__init__(observation_space, features_dim)

    self.net = nn.Sequential(
      nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=1),
      nn.Tanh(),
      nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
      nn.Tanh(),
      nn.Flatten(),
      nn.Linear(128*32, 256),
      nn.Tanh())
    
    self.fc = nn.Sequential(
      nn.Linear(in_features=256+3+3, out_features=features_dim),
      nn.Tanh())
    
  def forward(self, obs):
    obs_lidar = obs['lidar']
    obs_detect = obs['detect']
    obs_state = obs['state']

    lidar = self.net(obs_lidar)
    x = torch.cat(tensors=(lidar, obs_detect, obs_state), dim=1)
    final_out = self.fc(x)
    # print("[Net]: test..")
    # print("[obs]: ", obs)
    # print("[output]", final_out)

    return final_out

# TEST
if __name__ == "__main__":
  import numpy as np
  import time


  time1 = time.time()
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
  print(time.time() - time1)
  print(output_features)
