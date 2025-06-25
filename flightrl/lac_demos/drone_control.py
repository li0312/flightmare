'''
Author: Lac_Creeper
Date: 2025-06-24 12:35:40 +0800
LastEditTime: 2025-06-24 13:42:14 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /flightmare/flightrl/lac_demos/drone_control.py
'''
from ruamel.yaml import YAML, dump, RoundTripDumper
from gymnasium.utils import seeding

from stable_baselines3.common import logger
from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

import os
import argparse
import numpy as np
import torch as th

from lac_baselines.envs import drone_control_wrapper as wrapper
import lac_baselines.common.util as U

from flightgym import QuadrotorEnv_v1

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


def test_model(env, model, render=False):
    #
    fig = plt.figure(figsize=(18, 12), tight_layout=True)
    gs = gridspec.GridSpec(5, 12)
    #
    ax_x = fig.add_subplot(gs[0, 0:4])
    ax_y = fig.add_subplot(gs[0, 4:8])
    ax_z = fig.add_subplot(gs[0, 8:12])
    #
    ax_dx = fig.add_subplot(gs[1, 0:4])
    ax_dy = fig.add_subplot(gs[1, 4:8])
    ax_dz = fig.add_subplot(gs[1, 8:12])
    #
    ax_euler_x = fig.add_subplot(gs[2, 0:4])
    ax_euler_y = fig.add_subplot(gs[2, 4:8])
    ax_euler_z = fig.add_subplot(gs[2, 8:12])
    #
    ax_euler_vx = fig.add_subplot(gs[3, 0:4])
    ax_euler_vy = fig.add_subplot(gs[3, 4:8])
    ax_euler_vz = fig.add_subplot(gs[3, 8:12])
    #
    ax_action0 = fig.add_subplot(gs[4, 0:3])
    ax_action1 = fig.add_subplot(gs[4, 3:6])
    ax_action2 = fig.add_subplot(gs[4, 6:9])
    ax_action3 = fig.add_subplot(gs[4, 9:12])

    max_ep_length = env.max_episode_steps
    num_rollouts = 5
    if render:
        env.connectUnity()
    for n_roll in range(num_rollouts):
        pos, euler, dpos, deuler = [], [], [], []
        actions = []
        obs, done, ep_len = env.reset(), False, 0
        while not (done or (ep_len >= max_ep_length)):
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, infos = env.step(act)
            #
            ep_len += 1
            #
            pos.append(obs[0, 0:3].tolist())
            dpos.append(obs[0, 6:9].tolist())
            euler.append(obs[0, 3:6].tolist())
            deuler.append(obs[0, 9:12].tolist())
            #
            actions.append(act[0, :].tolist())
        pos = np.asarray(pos)
        dpos = np.asarray(dpos)
        euler = np.asarray(euler)
        deuler = np.asarray(deuler)
        actions = np.asarray(actions)
        #
        t = np.arange(0, pos.shape[0])
        ax_x.step(t, pos[:, 0], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_y.step(t, pos[:, 1], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_z.step(t, pos[:, 2], color="C{0}".format(
            n_roll), label="pos [x, y, z] -- trail: {0}".format(n_roll))
        #
        ax_dx.step(t, dpos[:, 0], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_dy.step(t, dpos[:, 1], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_dz.step(t, dpos[:, 2], color="C{0}".format(
            n_roll), label="vel [x, y, z] -- trail: {0}".format(n_roll))
        #
        ax_euler_x.step(t, euler[:, -1], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_euler_y.step(t, euler[:, 0], color="C{0}".format(
            n_roll), label="trail :{0}".format(n_roll))
        ax_euler_z.step(t, euler[:, 1], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        #
        ax_euler_vx.step(t, deuler[:, -1], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_euler_vy.step(t, deuler[:, 0], color="C{0}".format(
            n_roll), label="trail :{0}".format(n_roll))
        ax_euler_vz.step(t, deuler[:, 1], color="C{0}".format(
            n_roll), label=r"$\theta$ [x, y, z] -- trail: {0}".format(n_roll))
        #
        ax_action0.step(t, actions[:, 0], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_action1.step(t, actions[:, 1], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_action2.step(t, actions[:, 2], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_action3.step(t, actions[:, 3], color="C{0}".format(
            n_roll), label="act [0, 1, 2, 3] -- trail: {0}".format(n_roll))
    #
    if render:
        env.disconnectUnity()
    ax_z.legend()
    ax_dz.legend()
    ax_euler_z.legend()
    ax_euler_vz.legend()
    ax_action3.legend()
    #
    plt.tight_layout()
    plt.show()


def configure_random_seed(seed, env=None):
  if env is not None:
    env._np_random, env._np_random_seed = seeding.np_random(seed)
  np.random.seed(seed)
#   th.manual_seed(seed)
#   th.cuda.manual_seed(seed)
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
  cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                      "/flightlib/configs/vec_env.yaml", 'r'))
  if not args.train:
    cfg["env"]["num_envs"] = 1
    cfg["env"]["num_threads"] = 1

  if args.render:
    cfg["env"]["render"] = "yes"
  else:
    cfg["env"]["render"] = "no"
  
  print(cfg["env"]["render"])
  env = wrapper.QuadEnvVec(QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False))


  # set random seed
  configure_random_seed(args.seed, env=env)

  #
  if args.train:
    # save the configuration and other files
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/control_saved"
    saver = U.ConfigurationSaver(log_dir=log_dir)
    model = PPO(
      tensorboard_log=saver.data_dir,
      policy="MlpPolicy",  # check activation function
      policy_kwargs=dict(
          net_arch=[dict(pi=[128, 128], vf=[128, 128])], activation_fn=th.nn.ReLU),
      env=env,
      gae_lambda=0.95,
      gamma=0.99,  # lower 0.9 ~ 0.99
      # n_steps=math.floor(cfg['env']['max_time'] / cfg['env']['ctl_dt']),
      n_steps=300,
      ent_coef=0.00,
      learning_rate=3e-4,
      vf_coef=0.5,
      max_grad_norm=0.5,
      batch_size=30000,
      n_epochs=10,
      clip_range=0.2,
      verbose=1,
    )
    # model = PPO.load(log_dir + "/model_11460000_steps")
    # model.set_parameters(log_dir + "/model_11460000_steps")
    

    # tensorboard
    # Make sure that your chrome browser is already on.
    # TensorboardLauncher(saver.data_dir + '/PPO2_1')

    # PPO run
    # Originally the total timestep is 5 x 10^8
    # 10 zeros for nupdates to be 4000
    # 1000000000 is 2000 iterations and so
    # 2000000000 is 4000 iterations.

    checkpoint_callback = CheckpointCallback(
      save_freq=1000000,
      save_path=log_dir + "/checkpoints/",
      name_prefix="model"
    )

    logger.configure(folder=saver.data_dir)
    model.learn(
        total_timesteps=int(25000000),
        callback=checkpoint_callback,
        tb_log_name=saver.data_dir + "/ppo_run",
        reset_num_timesteps=False)
    model.save(saver.data_dir)

  # # Testing mode with a trained weight
  else:
      model = PPO.load(args.weight)
      test_model(env, model, True)


if __name__ == "__main__":
  main()