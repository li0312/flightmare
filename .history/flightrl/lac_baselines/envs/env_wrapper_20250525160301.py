'''
Author: Lac_Creeper
Date: 2025-05-25 15:18:40 +0800
LastEditTime: 2025-05-25 15:58:14 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /flightmare/flightrl/lac_baselines/envs/env_wrapper.py
'''
import gymnasium as gym
import numpy as np
import time


class EnvWrapper(gym.Env):
	def __init__(self, env):
		self.env = env
		self.env.init()
		self.num_obs = env.getObsDim()
		self.num_act = env.getActDim()

		self._observation_space = gym.spaces.Box(
			np.ones(self.num_obs) * -np.Inf,
			np.ones(self.num_obs) * np.Inf, 
			dtype=np.float32)
		# the actions are eventually constrained by the action space
		self._action_space = gym.spaces.Box(
			low=np.ones(self.num_act) * -1., 
			high=np.ones(self.num_act) * 1., 
			dtype=np.float32)
		self.observation = np.zeros(self.num_obs, 
			dtype=np.float32)
		self.reward = np.float32(0.0)
		self.done = False

		gym.Env.__init__(self)
		#
		self._max_episode_steps = 300

	def seed(self, seed=None):
		self.env.setSeed(seed)

	def step(self, action):
		self.reward = self.env.step(action, self.observation)
		terminal_reward = 0.0
		self.terminated, self.truncated = self.env.isTerminalState(terminal_reward)
		return self.observation.copy(), self.reward, \
			self.terminated, self.truncated, \
			[dict(reward_run=self.reward, reward_ctrl=0.0)]
	
	def reset(self, seed=None, options=None):
		self.reward = 0.0
		self.env.reset(self.observation, seed, options)
		info = {}
		return self.observation.copy(), info
	
	def obs(self):
		self.env.getObs(self.observation)
		return self.observation
	
	def close(self):
		return True
	
	def 