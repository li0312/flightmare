'''
Author: Lac_Creeper
Date: 2025-05-25 21:33:19 +0800
LastEditTime: 2025-05-25 21:58:01 +0800
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
import rpg_baselines.common.util as U

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

def main():
  args = parser().parse_args()
  cfg = YAML.load(open(os.environ["FLIGHTMARE_PATH"] +
                      "/flightlib/configs/vec_env.yaml", 'r'))
  if not args.train:
    cfg["env"]["num_envs"] = 1
    cfg["env"]["num_threads"] = 1

  if args.render:
    cfg["env"]["render"] = "yse"
  else:
    cfg["env"]["render"] = "no"

  env = wrapper.TrackEnvVec(TrackEnv_v1(dump(cfg, Dumper=RoundTripDumper), False))

  # set random seed
  configure_random_seed(args.seed, env=env)

  #
  if args.train:
    # save the configuration and other files
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/saved"
    saver = U.ConfigurationSaver(log_dir=log_dir)
    model = PPO2(
        tensorboard_log=saver.data_dir,
        policy=MlpPolicy,  # check activation function
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])], act_fun=tf.nn.relu),
        env=env,
        lam=0.95,
        gamma=0.99,  # lower 0.9 ~ 0.99
        # n_steps=math.floor(cfg['env']['max_time'] / cfg['env']['ctl_dt']),
        n_steps=250,
        ent_coef=0.00,
        learning_rate=3e-4,
        vf_coef=0.5,
        max_grad_norm=0.5,
        nminibatches=1,
        noptepochs=10,
        cliprange=0.2,
        verbose=1,
    )

      # tensorboard
      # Make sure that your chrome browser is already on.
      # TensorboardLauncher(saver.data_dir + '/PPO2_1')

      # PPO run
      # Originally the total timestep is 5 x 10^8
      # 10 zeros for nupdates to be 4000
      # 1000000000 is 2000 iterations and so
      # 2000000000 is 4000 iterations.
      logger.configure(folder=saver.data_dir)
      model.learn(
          total_timesteps=int(25000000),
          log_dir=saver.data_dir, logger=logger)
      model.save(saver.data_dir)

  # # Testing mode with a trained weight
  else:
      model = PPO2.load(args.weight)
      test_model(env, model, render=args.render)


if __name__ == "__main__":
  main()