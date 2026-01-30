'''
Author: Lac_Creeper
Date: 2025-06-24 14:23:27 +0800
LastEditTime: 2026-01-25 18:18:46 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /src/flightmare/flightrl/examples/track_control.py
'''
from ruamel.yaml import YAML, dump, RoundTripDumper
import warnings
warnings.filterwarnings("ignore", 
    category=FutureWarning,
    message="Passing \(type, 1\) or '1type' as a synonym of type is deprecated.*")
#
import os
import math
import time
import argparse
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#
from stable_baselines import logger

#
from stable_baselines.common.schedules import LinearSchedule
from rpg_baselines.common.policies import FeedForwardPolicy
from rpg_baselines.ppo.ppo2 import PPO2
from rpg_baselines.envs import vec_env_wrapper as wrapper
import rpg_baselines.common.util as U
#
from flightgym import TrackEnv_v1

import rospy


def test_model(env, model):
    ep_num = 0
    time_sum = 0
    time_count = 0
    while ep_num <= 0:
        obs, done, ep_len = env.reset(), False, 0
        print(obs.dtype)
        print(obs.shape)
        while not done:  # 双重检查
            start = time.perf_counter()
            act, _ = model.predict(obs, deterministic=True)
            end = time.perf_counter()
            time_sum += (end - start)*1000
            time_count += 1
            print(f"平均执行时间: {time_sum/time_count:.6f} ms")
            obs, rew, done, _ = env.step(act)  # 明确忽略infos避免未使用变量警告
            ep_len += 1
            if ep_len > 1590:
                break
        ep_num += 1


class TrackPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        # 父类初始化（必须调用）
        super(TrackPolicy, self).__init__(
            sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
            feature_extraction="mlp", **kwargs)

    def mlp_extractor(self, flattened_obs, **kwargs):
        """
        自定义特征提取逻辑(TensorFlow 实现）
        :param flattened_obs: 展平后的观测张量 (batch_size, flattened_dim)
        :return: (pi_features, vf_features) 用于 Actor/Critic 的特征
        """
        # 假设观测已展平为 [lidar(3*512) + detect(3) + state(3)]
        lidar_dim = 3 * 512
        detect_dim = 5
        state_dim = 3

        # 1. 拆分展平后的观测, shape=(batch, N)
        lidar_data = flattened_obs[:, :lidar_dim]
        detect_data = flattened_obs[:, lidar_dim:lidar_dim+detect_dim]
        state_data = flattened_obs[:, -state_dim:]
        # 2. 处理 Lidar 数据（1D CNN）
        lidar_data = tf.reshape(lidar_data, [-1, 3, 512])
        conv1 = tf.layers.conv1d(
            inputs=lidar_data,
            filters=32,
            kernel_size=5,
            strides=2,
            padding='same',
            activation=tf.nn.relu,
            name="lidar_conv1"
        )  # shape=(batch, 3, 32)
        conv2 = tf.layers.conv1d(
            inputs=conv1,
            filters=32,
            kernel_size=3,
            strides=2,
            padding='same',
            activation=tf.nn.relu,
            name="lidar_conv2"
        )  # shape=(batch, 3, 32)
        flattened = tf.layers.flatten(conv2)                     # shape=(batch, 3*32*128)
        lidar_features = tf.layers.dense(
            flattened,
            units=512,
            activation=tf.nn.relu,
            name="lidar_fc"
        )  # shape=(batch, 256)

        # 3. 合并所有特征
        combined = tf.concat([lidar_features, detect_data, state_data], axis=1)  # shape=(batch, 256+3+3)
        features = tf.layers.dense(
            combined,
            units=256,
            activation=tf.nn.relu,
            name="final_fc"
        )  # shape=(batch, 128)

        # 返回相同特征给 Actor 和 Critic（SB2 会在此后追加各自的 MLP）
        return features, features

def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

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
    parser.add_argument('-w', '--weight', type=str, default='./saved/quadrotor_env.zip',
                        help='trained weight path')
    return parser


def main():
    args = parser().parse_args()
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                           "/flightlib/configs/vec_env.yaml", 'r'))
    if not args.train:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1
        rospy.init_node("track_env_py", anonymous=True)

        
    if args.render:
        cfg["env"]["render"] = "yes"
    else:
        cfg["env"]["render"] = "no"

    env = wrapper.FlightEnvVec(TrackEnv_v1(
        dump(cfg, Dumper=RoundTripDumper), False))

    # set random seed
    configure_random_seed(args.seed, env=env)

    lr_schedule = LinearSchedule(3e7, 3e-5, 3e-4)

    #
    if args.train:
        # save the configuration and other files
        rsg_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = rsg_root + '/track_saved'
        saver = U.ConfigurationSaver(log_dir=log_dir)
        model = PPO2(
            tensorboard_log=saver.data_dir,
            policy=TrackPolicy,  # check activation function
            policy_kwargs=dict(
                net_arch=[dict(pi=[128, 128], vf=[128, 128])], act_fun=tf.nn.relu),
            env=env,
            lam=0.95,
            gamma=0.99,  # lower 0.9 ~ 0.99
            # n_steps=math.floor(cfg['env']['max_time'] / cfg['env']['ctl_dt']),
            n_steps=300,
            ent_coef=0.00,
            learning_rate=3e-4,
            vf_coef=0.5,
            max_grad_norm=0.5,
            nminibatches=1,
            noptepochs=10,
            cliprange=0.2,
            verbose=1,
        )
        # model = PPO2.load(log_dir + '/2026-01-14-00-55-13.zip', env=env, tensorboard_log=saver.data_dir)

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
            total_timesteps=int(60000000),
            log_dir=saver.data_dir, logger=logger)
        model.save(saver.data_dir)

    # # Testing mode with a trained weight
    else:
        model = PPO2.load(args.weight)
        test_model(env, model)


if __name__ == "__main__":
    main()
