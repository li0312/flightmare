'''
Author: Lac_Creeper
Date: 2026-02-05 15:33:47 +0800
LastEditTime: 2026-02-27 02:58:34 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /flightmare/flightrl/examples/rl_control.py
'''
import yaml
from yaml.loader import SafeLoader

import warnings
warnings.filterwarnings("ignore", 
    category=FutureWarning,
    message="Passing \(type, 1\) or '1type' as a synonym of type is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
import sys
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

import tensorflow as tf
#
from stable_baselines import logger
from stable_baselines.common.schedules import LinearSchedule
from rpg_baselines.common.policies import FeedForwardPolicy
from rpg_baselines.ppo.ppo2 import PPO2
from rpg_baselines.envs import vec_env_wrapper as wrapper
import rpg_baselines.common.util as U

#
import rospy
import cv2
import tf
import tf2_ros
from geometry_msgs.msg import TransformStamped, TwistStamped, PoseStamped
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge


WINDOW_SIZE = 5
MODEL_NAME = "/home/lac/fm_test/src/flightmare/flightrl/examples/track_saved/2026-01-17-19-31-40.zip"


class TrackPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
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
    

class Esitmator:
    def __init__(self):
        self.B2C = R.from_euler('zyx', [np.pi/2.0, 0, np.pi/2]).as_matrix()
        self.K = np.array([[348.74, 0, 480],
                             [0, 348.74, 270],
                             [0, 0, 1]])
        self.K_inv = np.linalg.inv(self.K)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0, 
                        help="Drone id")
    return parser


class RLTrack():
    def __init__(self, model, ns):
        self.model = PPO2.load(model)

        #
        self.desire_bbox = [449, 158, 510, 362]
        self.desire_uvh = [480, 260, 210]
        self.desire_dist = 3

        self.scan_list = []
        self.odom_list = []
        self.detect = None
        self.vel = None

        # ROS Publisher
        self.cmd_pub = rospy.Publisher(ns + "/vel_ctrl", TwistStamped, queue_size=1)

        # ROS Subscriber
        rospy.Subscriber(ns + "/odom", Odometry, self.odomCB)
        rospy.Subscriber(ns + "/scan", LaserScan, self.scanCB)
        rospy.Subscriber(ns + "/targetBox", PoseStamped, self.detectCB)

    
    def run(self):
        if not self.has_init():
            return
        scan_flat = np.array(self.scan_list).reshape(-1)
        other_obs = np.array(self.detect + self.vel)
        obs = np.concatenate((scan_flat, other_obs))
        act, _ = self.model.predict(obs, deterministic=True)
        cmd_msg = TwistStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.twist.linear.x = act[0] * 2.5
        cmd_msg.twist.linear.y = act[1] * 2.0
        cmd_msg.twist.angular.z = act[2] * 0.5
        self.cmd_pub.publish(cmd_msg)


    def has_init(self):
        return self.scan_list and self.odom_list and self.detect
    

    def odomCB(self, msg: Odometry):
        if len(self.odom_list) < WINDOW_SIZE:
            self.odom_list.append(msg)
        else:
            self.odom_list = self.odom_list[1:] + [msg]
        
    def scanCB(self, msg: LaserScan):
        scan = msg.ranges
        scan_norm = [np.exp(-i) for i in scan]
        if not self.scan_list:
            self.scan_list = [scan_norm, scan_norm, scan_norm]
        else:
            self.scan_list = self.scan_list[1:] + [scan_norm]
    
    def detectCB(self, msg: PoseStamped):
        if not self.odom_list:
            return
        t_det = msg.header.stamp.to_sec()
        # for odom in self.odom_list:
        #     t_odom = odom.header.stamp.to_sec()
        #     print(f"id: {odom.header.seq} | t: {abs(t_odom - t_det):.2f}")

        odoms = sorted(self.odom_list, 
                      key=lambda x: abs(x.header.stamp.to_sec() - t_det))
        # for a in odoms:
        #     print(f"seq: {a.header.seq} | t: {abs(a.header.stamp.to_sec() - t_det):.2f}")

        odom = odoms[0]
        qw = odom.pose.pose.orientation.w
        qx = odom.pose.pose.orientation.x
        qy = odom.pose.pose.orientation.y
        qz = odom.pose.pose.orientation.z
        yaw = np.arctan2(2.0 * qw * qz + 2.0 * qx * qy,
                         qw * qw + qx * qx - qy * qy - qz * qz)
        vel = [odom.twist.twist.linear.x, 
               odom.twist.twist.linear.y,
               odom.twist.twist.angular.z]
        vx_body = np.cos(yaw) * vel[0] + np.sin(yaw) * vel[1]
        vy_body = -np.sin(yaw) * vel[0] + np.cos(yaw) * vel[1]
        self.vel = [vx_body, vy_body, vel[2]]
        umin = msg.pose.orientation.w
        umax = msg.pose.orientation.x
        vmin = msg.pose.orientation.y
        vmax = msg.pose.orientation.z
        bbox_obs = [((umin + umax) / 2.0 - 480.0) / 960.0,
                    ((vmin + vmax) / 2.0 - 260.0) / 540.0,
                    (vmax - vmin - 210.0) / 540.0]
        dirt_obs = [msg.pose.position.x, 
                    msg.pose.position.y]
        self.detect = bbox_obs + dirt_obs

        # publish RL command
        if not self.has_init():
            return
        scan_flat = np.array(self.scan_list).reshape(-1)
        other_obs = np.array(self.detect + self.vel)
        obs = np.concatenate((scan_flat, other_obs))
        act, _ = self.model.predict(obs, deterministic=True)
        cmd_msg = TwistStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.twist.linear.x = act[0] * 2.5
        cmd_msg.twist.linear.y = act[1] * 2.0
        cmd_msg.twist.angular.z = act[2] * 0.5
        self.cmd_pub.publish(cmd_msg)
        rospy.loginfo("CMD: [%.2f, %.2f, %.2f]"%(cmd_msg.twist.linear.x,
                                                 cmd_msg.twist.linear.y,
                                                 cmd_msg.twist.angular.z))
        
        





if __name__ == "__main__":
    args = parser().parse_args()
    name_space = "uav" + str(args.id)
    rospy.init_node(name_space + "_RL_control")
    track_control = RLTrack(MODEL_NAME, name_space)
    rospy.spin()