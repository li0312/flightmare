'''
Author: Lac_Creeper
Date: 2026-03-06 19:02:48 +0800
LastEditTime: 2026-05-30 22:44:04 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /src/flightmare/flightrl/examples/trackAdv_control.py
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
from rpg_baselines.common.policies import LaserLstmPolicy, LaserMlpPolicy
from rpg_baselines.ppo.ppo2 import PPO2
from rpg_baselines.envs import vec_env_wrapper as wrapper
import rpg_baselines.common.util as U
#
from flightgym import TrackAdvEnv_v2

import rospy


def test_model(env, model):
    ep_num = 0
    time_sum = 0
    time_count = 0
    while ep_num <= 0:
        obs, done, ep_len = env.reset(), False, 0
        state = None
        mask = [True]
        print(obs.dtype)
        print(obs.shape)
        while not done:  # 双重检查
            start = time.perf_counter()
            act, next_state = model.predict(obs, state=state, mask=mask,
                                            deterministic=True)
            end = time.perf_counter()
            time_sum += (end - start)*1000
            time_count += 1
            print(f"平均执行时间: {time_sum/time_count:.6f} ms")
            obs, rew, done, _ = env.step(act)  # 明确忽略infos避免未使用变量警告
            done = done[0]  # 处理单环境的done
            state = next_state
            mask = [done]
            ep_len += 1
            if ep_len > 10000:
                break
        ep_num += 1


def my_schedule(initial_value=3e-4, final_value=3e-5):
    def schedule(progress):
        if progress >= 0.75:
            return initial_value
        elif progress <= 0.25:
            return final_value
        else:
            return final_value + (initial_value - final_value) * (progress - 0.25) / 0.5
    return schedule



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


best_mean_reward = -np.inf

def best_model_callback(_locals, _globals):
    global best_mean_reward
    
    self_obj = _locals['self']
    log_dir = _locals['log_dir']
    update = _locals['update']
    ep_info_buf = self_obj.ep_info_buf
    if update % 100 == 0 and len(ep_info_buf) > 0:
        mean_reward = np.mean([ep_info['r'] for ep_info in ep_info_buf])
        mean_reward = int(mean_reward)
        save_path = os.path.join(log_dir, f'model_{update}_{mean_reward}.zip')
        self_obj.save(save_path)

    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
        mean_reward = np.mean([ep_info['r'] for ep_info in ep_info_buf])
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            save_path = os.path.join(log_dir, 'best_model.zip')
            print(f"New best mean reward: {best_mean_reward:.2f} at update {update}, saving model to {save_path}")
            self_obj.save(save_path)
    return True  # 返回True继续训练，返回False可以中断训练

def main():
    args = parser().parse_args()
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                           "/flightlib/configs/vec_env.yaml", 'r'))
    if not args.train:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1
        rospy.init_node("trackAdv_env_py", anonymous=True)

        
    if args.render:
        cfg["env"]["render"] = "yes"
    else:
        cfg["env"]["render"] = "no"

    env = wrapper.FlightEnvVec(TrackAdvEnv_v2(
        dump(cfg, Dumper=RoundTripDumper), False))

    # set random seed
    configure_random_seed(args.seed, env=env)


    #
    if args.train:
        # save the configuration and other files
        rsg_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = rsg_root + '/trackNewAdv_saved'
        saver = U.ConfigurationSaver(log_dir=log_dir)
        # model = PPO2(
        #     tensorboard_log=saver.data_dir,
        #     policy=LaserLstmPolicy,  # check activation function
        #     env=env,
        #     lam=0.95,
        #     gamma=0.99,  # lower 0.9 ~ 0.99
        #     # n_steps=math.floor(cfg['env']['max_time'] / cfg['env']['ctl_dt']),
        #     n_steps=300,
        #     ent_coef=0.00,
        #     learning_rate=my_schedule(3e-4, 3e-5),
        #     vf_coef=0.5,
        #     max_grad_norm=0.5,
        #     nminibatches=1,
        #     noptepochs=10,
        #     cliprange=0.2,
        #     verbose=1,
        # )
        model = PPO2.load(log_dir + '/2026-04-09-05-31-37.zip', env=env, tensorboard_log=saver.data_dir)
        # print("Model loaded successfully.")

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
            log_dir=saver.data_dir, 
            logger=logger,
            callback=best_model_callback)
        model.save(saver.data_dir)

    # # Testing mode with a trained weight
    else:
        model = PPO2.load(args.weight)
        test_model(env, model)


if __name__ == "__main__":
    main()
