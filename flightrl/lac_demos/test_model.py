'''
Author: Lac_Creeper
Date: 2025-06-17 19:55:34 +0800
LastEditTime: 2025-06-17 20:49:04 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /flightmare/flightrl/lac_demos/test_model.py
'''
from stable_baselines3.ppo import PPO


def test_model(env, model: PPO):
    while True:
        obs, done, ep_len = env.reset(), False, 0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, infos = env.step(act)

            ep_len += 1

        