'''
Author: Lac_Creeper
Date: 2025-05-26 18:34:13 +0800
LastEditTime: 2025-05-26 20:49:47 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /flightmare/flightrl/lac_baselines/common/custom_net.py
'''
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs


def log_normal_density(x, mean, log_std, std):
  var = std.pow(2)
  log_density = -(x - mean).poe(2) / (2*var) - 0.5 * math.log(2*math.pi) - log_std
  return log_density.sum(-1, keepdim=True)

class CustomNet(BaseFeaturesExtractor):
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
    
  def forward(self, observations: torch.Tensor) -> torch.Tensor:
    return self.net(observations)
  
class CustomPolicy(BasePolicy):
  def __init__(self,
      observation_space,
      action_space,
      lr_schedule,
      net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
      activation_fn: Type[nn.Module] = nn.ReLU,
      *args,
      **kwargs):
    super().__init__(
        observation_space,
        action_space,
        lr_schedule,
        *args, **kwargs)
    
    self.features_extractor = CustomNet(observation_space, features_dim=256)

    self.action_net = nn.Sequential(
        nn.Linear(256+3+3, 128),
        nn.ReLU(),
        nn.Linear(128, action_space.shape[0]))
    
    self.value_net = nn.Sequential(
        nn.Linear(256+3+3, 128),
        nn.ReLU(),
        nn.Linear(128, 1))
    
    self.logstd = nn.Parameter(torch.zeros(action_space.shape[0]))


  def forward(self, obs: PyTorchObs, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    features = self.features_extractor(obs["lidar"])
    detect = obs["detect"]
    state = obs["state"]

    actor_features = torch.cat([features, detect, state], dim=-1)
    critic_features = torch.cat([features.detach(), detect, state], dim=-1)

    mean = torch.tanh(self.action_net(actor_features))
    logstd = self.logstd.expand_as(mean)
    std = torch.exp(logstd)

    if deterministic:
      action = mean
    else:
      action = torch.normal(mean, std)
    
    logprob = log_normal_density(action, mean, logstd, std)

    value = self.value_net(critic_features)


  def evaluate_actions(self, obs: PyTorchObs, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = obs["lidar"]
    detect = obs["detect"]
    state = obs["state"]

    features = self.features_extractor(x)
    actor_features = torch.cat([features, detect, state], dim=-1)

    mean = torch.tanh(self.action_net(actor_features))
    logstd = self.logstd.expand_as(mean)
    std = torch.exp(logstd)

    logprob = log_normal_density(actions, mean, logstd, std)
    dist_entropy = 0.5 + 0.5*math.log(2*math.pi) + logstd
    dist_entropy = dist_entropy.sum(-1).mean()

    critic_features = torch.cat([features.detach(), detect, state], dim=-1)
    value = self.value_net(critic_features)

    return value, logprob, dist_entropy
  
register_policy("CustomPolicy", CustomPolicy)



