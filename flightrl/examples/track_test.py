'''
Author: Lac_Creeper
Date: 2026-01-07 15:55:18 +0800
LastEditTime: 2026-01-07 22:49:19 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /src/flightmare/flightrl/examples/track_test.py
'''
import yaml
from yaml.loader import SafeLoader

import warnings
warnings.filterwarnings("ignore", 
    category=FutureWarning,
    message="Passing \(type, 1\) or '1type' as a synonym of type is deprecated.*")

import os
import sys
import numpy as np

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
from geometry_msgs.msg import TransformStamped, TwistStamped
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

class TrackTest():
    def __init__(self) -> None:
        # Init
        # Parameter
        self.step = None
        self.laser_list = None

        self.scan_list = None
        # ROS Publisher
        self.cmd_pub = rospy.Publisher("vel_ctrl", TwistStamped, queue_size=1)
        # ROS Subscriber
        rospy.Subscriber("/hummingbird/ground_truth/odometry", Odometry, self.odomCB)
        rospy.Subscriber("/hummingbird/scan", LaserScan, self.scanCB)
        rospy.Subscriber("/hummingbird/target", Odometry, self.targetCB)

        # For Debug






    def enjoy(self, model: PPO2):
        done = False
        while self.step < 2000 and not done and not rospy.is_shutdown():
            obs, done = self._get_obs()
            act, _ = model.predict(obs, deterministic=True)
            self.do_action(act)
            self.step += 1


    def _setup_flight(self):
        while not self.scan_list and not self.drone_pose:
            rospy.sleep(0.1)
        self.step = 0

    def _get_obs(self):
        done = False


    def scanCB(self, msg: LaserScan):
        if self.scan_list is None:
            self.scan_list = [msg.ranges]*3
        else:
            self.scan_list = self.scan_list[1:] + [msg.ranges]

    def targetCB(self, msg: Odometry):
        self.target = msg

    def odomCB(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
