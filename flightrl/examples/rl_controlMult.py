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
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#
from stable_baselines import logger
from rpg_baselines.common.policies import LaserLstmPolicy
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
# MODEL_NAME = "/home/lac/fm_test/src/flightmare/flightrl/examples/trackNewMult_saved/2026-04-14-04-15-41/best_model.zip"
MODEL_NAME = "/home/lac/fm_test/src/flightmare/flightrl/examples/trackMult_saved/2026-03-16-13-29-10.zip"


USE_RNN = False
if (MODEL_NAME.find("Mult") >= 0):
    USE_RNN = True


    

class Estimator:
    def __init__(self):
        self.B2C = R.from_euler('zyx', [np.pi/2.0, 0, np.pi/2]).as_matrix()
        self.K = np.array([[348.74, 0, 480],
                             [0, 348.74, 270],
                             [0, 0, 1]])
        self.K_inv = np.linalg.inv(self.K)


    def esti_wpt(self, imu0_msg: Odometry, box0_msg: PoseStamped):
        ## 1. handle rosmsg
        imu0_p = np.array([imu0_msg.pose.pose.position.x,
                           imu0_msg.pose.pose.position.y,
                           imu0_msg.pose.pose.position.z])
        imu0_q = np.array([imu0_msg.pose.pose.orientation.x,
                           imu0_msg.pose.pose.orientation.y,
                           imu0_msg.pose.pose.orientation.z,
                           imu0_msg.pose.pose.orientation.w])
        
        B02W = R.from_quat(imu0_q).as_matrix()
        yaw = np.arctan2(2*imu0_q[3]*imu0_q[2] + 2*imu0_q[0]*imu0_q[1],
                         imu0_q[3]*imu0_q[3] + imu0_q[0]*imu0_q[0] - 
                         imu0_q[1]*imu0_q[1] - imu0_q[2]*imu0_q[2])
        B02W = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                         [np.sin(yaw), np.cos(yaw), 0.0,],
                         [0.0, 0.0, 1.0]])


        umin = box0_msg.pose.orientation.w
        umax = box0_msg.pose.orientation.x
        vmin = box0_msg.pose.orientation.y
        vmax = box0_msg.pose.orientation.z
        tpt0 = np.array([(umin + umax)/2.0, vmin])
        bpt0 = np.array([(umin + umax)/2.0, vmax])

        ## 2. 
        bpt0_3d = np.concatenate((bpt0, np.ones(1))).reshape(3, 1)
        wp0_temp = B02W @ (self.B2C.T @ (self.K_inv @ bpt0_3d))
        Kz = (0.0 - imu0_p[2]) / wp0_temp[2]
        wp0 = Kz * wp0_temp + imu0_p.reshape(3 ,1)

        return wp0
    
    def esti_by_proj(self, wp0, imu1_msg: Odometry):
        imu1_p = np.array([imu1_msg.pose.pose.position.x,
                           imu1_msg.pose.pose.position.y,
                           imu1_msg.pose.pose.position.z])
        imu1_q = np.array([imu1_msg.pose.pose.orientation.x,
                           imu1_msg.pose.pose.orientation.y,
                           imu1_msg.pose.pose.orientation.z,
                           imu1_msg.pose.pose.orientation.w])
        B12W = R.from_quat(imu1_q).as_matrix()
        pt1_hat_3d = self.K @ (self.B2C @ B12W.T @ (wp0 - imu1_p.reshape(3, 1)))
        pt1_hat = np.array([pt1_hat_3d[0] / pt1_hat_3d[2],
                            pt1_hat_3d[1] / pt1_hat_3d[2]]).reshape(2)
        return pt1_hat


    def esti_by_H1(self, imu1_msg: Odometry, imu0_msg: Odometry, 
                  box0_msg: PoseStamped):
        ## 1. rosmsg to parameters
        imu0_p = np.array([imu0_msg.pose.pose.position.x,
                           imu0_msg.pose.pose.position.y,
                           imu0_msg.pose.pose.position.z])
        imu0_q = np.array([imu0_msg.pose.pose.orientation.x,
                           imu0_msg.pose.pose.orientation.y,
                           imu0_msg.pose.pose.orientation.z,
                           imu0_msg.pose.pose.orientation.w])
        imu1_p = np.array([imu1_msg.pose.pose.position.x,
                           imu1_msg.pose.pose.position.y,
                           imu1_msg.pose.pose.position.z])
        imu1_q = np.array([imu1_msg.pose.pose.orientation.x,
                           imu1_msg.pose.pose.orientation.y,
                           imu1_msg.pose.pose.orientation.z,
                           imu1_msg.pose.pose.orientation.w])
        B02W = R.from_quat(imu0_q).as_matrix()
        B12W = R.from_quat(imu1_q).as_matrix()

        umin = box0_msg.pose.orientation.w
        umax = box0_msg.pose.orientation.x
        vmin = box0_msg.pose.orientation.y
        vmax = box0_msg.pose.orientation.z
        tpt0 = np.array([(umin + umax)/2.0, vmin])
        bpt0 = np.array([(umin + umax)/2.0, vmax])

        ## 2. calculate H
        R0 = self.B2C @ B02W.T
        R1 = self.B2C @ B12W.T
        t0 = R0 @ (-imu0_p.reshape(3, 1))
        t1 = R1 @ (-imu1_p.reshape(3, 1))

        n_w = np.array([0, 0, 1]).reshape(3, 1)
        d_w = 1e-1
        # d_w = 1.77
        n0 = R0 @ n_w
        d0 = d_w - n0.T @ t0
        if np.isclose(d0, 0):
            print("相机1位于平面上(d1=0),无法计算单应矩阵...")
            return None
        
        R_rel = R1 @ R0.T
        t_rel = t1 - R_rel @ t0
        H_p = self.K @ (R_rel - (t_rel @ n0.T) / d0) @ self.K_inv
        ## 3. calculate projected points
        tpt0_3d = np.concatenate((tpt0, np.ones(1))).reshape(3, 1)
        tpt1_hat_3d = (H_p @ tpt0_3d).reshape(3)
        tpt1_hat = np.array([tpt1_hat_3d[0] / tpt1_hat_3d[2],
                             tpt1_hat_3d[1] / tpt1_hat_3d[2]])
        bpt0_3d = np.concatenate((bpt0, np.ones(1))).reshape(3, 1)
        bpt1_hat_3d = (H_p @ bpt0_3d).reshape(3)
        bpt1_hat = np.array([bpt1_hat_3d[0] / bpt1_hat_3d[2],
                             bpt1_hat_3d[1] / bpt1_hat_3d[2]])
        return tpt1_hat, bpt1_hat
    


    def esti_by_H(self, imu1_msg: Odometry, imu0_msg: Odometry, 
                  box0_msg: PoseStamped):
        ## 1. rosmsg to parameters
        imu0_p = np.array([imu0_msg.pose.pose.position.x,
                           imu0_msg.pose.pose.position.y,
                           imu0_msg.pose.pose.position.z])
        imu0_q = np.array([imu0_msg.pose.pose.orientation.x,
                           imu0_msg.pose.pose.orientation.y,
                           imu0_msg.pose.pose.orientation.z,
                           imu0_msg.pose.pose.orientation.w])
        imu1_p = np.array([imu1_msg.pose.pose.position.x,
                           imu1_msg.pose.pose.position.y,
                           imu1_msg.pose.pose.position.z])
        imu1_q = np.array([imu1_msg.pose.pose.orientation.x,
                           imu1_msg.pose.pose.orientation.y,
                           imu1_msg.pose.pose.orientation.z,
                           imu1_msg.pose.pose.orientation.w])
        B02W = R.from_quat(imu0_q).as_matrix()
        B12W = R.from_quat(imu1_q).as_matrix()
        # yaw0 = np.arctan2(2*imu0_q[3]*imu0_q[2] + 2*imu0_q[0]*imu0_q[1],
        #                  imu0_q[3]*imu0_q[3] + imu0_q[0]*imu0_q[0] - 
        #                  imu0_q[1]*imu0_q[1] - imu0_q[2]*imu0_q[2])
        # B02W = np.array([[np.cos(yaw0), -np.sin(yaw0), 0.0],
        #                  [np.sin(yaw0), np.cos(yaw0), 0.0,],
        #                  [0.0, 0.0, 1.0]])
        # yaw1 = np.arctan2(2*imu1_q[3]*imu1_q[2] + 2*imu1_q[0]*imu1_q[1],
        #                  imu1_q[3]*imu1_q[3] + imu1_q[0]*imu1_q[0] - 
        #                  imu1_q[1]*imu1_q[1] - imu1_q[2]*imu1_q[2])
        # B12W = np.array([[np.cos(yaw1), -np.sin(yaw1), 0.0],
        #                  [np.sin(yaw1), np.cos(yaw1), 0.0,],
        #                  [0.0, 0.0, 1.0]])

        umin = box0_msg.pose.orientation.w
        umax = box0_msg.pose.orientation.x
        vmin = box0_msg.pose.orientation.y
        vmax = box0_msg.pose.orientation.z
        tpt0 = np.array([(umin + umax)/2.0, vmin])
        bpt0 = np.array([(umin + umax)/2.0, vmax])

        ## 2. calculate H
        R0 = self.B2C @ B02W.T
        R1 = self.B2C @ B12W.T
        t0 = R0 @ (-imu0_p.reshape(3, 1))
        t1 = R1 @ (-imu1_p.reshape(3, 1))

        n_w = np.array([0, 0, 1]).reshape(3, 1)
        d_w = 0.01
        # d_w = 1.77
        n0 = R0 @ n_w
        d0 = d_w - n0.T @ t0
        if np.isclose(d0, 0):
            print("相机1位于平面上(d1=0),无法计算单应矩阵...")
            return None
        P0 = R0 - (t0 @ n_w.T)/d_w
        P1 = R1 - (t1 @ n_w.T)/d_w
        H_p = self.K @ P1 @ np.linalg.inv(P0) @ self.K_inv
        ## 3. calculate projected points
        tpt0_3d = np.concatenate((tpt0, np.ones(1))).reshape(3, 1)
        tpt1_hat_3d = (H_p @ tpt0_3d).reshape(3)
        tpt1_hat = np.array([tpt1_hat_3d[0] / tpt1_hat_3d[2],
                             tpt1_hat_3d[1] / tpt1_hat_3d[2]])
        bpt0_3d = np.concatenate((bpt0, np.ones(1))).reshape(3, 1)
        bpt1_hat_3d = (H_p @ bpt0_3d).reshape(3)
        bpt1_hat = np.array([bpt1_hat_3d[0] / bpt1_hat_3d[2],
                             bpt1_hat_3d[1] / bpt1_hat_3d[2]])
        return tpt1_hat, bpt1_hat



def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0, 
                        help="Drone id")
    parser.add_argument('--use_esti', type=bool, default=0,
                        help="Use Estimator or not")
    # parser.add_argument('--use_rnn', type=bool, default=0,
    #                     help="Use RNN or not")
    return parser


class RLTrack():
    def __init__(self, model, ns, use_esti, use_rnn):
        self.model = PPO2.load(model)
        self.use_rnn = use_rnn

        # estimator
        other_ns = ""
        if ns == "uav0":
            other_ns = "uav1"
        elif ns == "uav1":
            other_ns = "uav0"
        self.use_esti = use_esti
        self.esti = Estimator()

        #
        self.desire_bbox = [449, 158, 510, 362]
        self.desire_uvh = [480, 260, 210]
        self.desire_dist = 3

        self.odom_list = []
        self.box_list = []
        self.scan_norm = None
        self.dirt = None
        self.vel = None
        self.box = None
        self.box_last = None
        self.odom_other_list = []
        self.box_other = None
        self.odom_other = None

        # For RNN
        self.mask = [True]
        self.state = None

        # ROS Publisher
        self.cmd_pub = rospy.Publisher(ns + "/vel_ctrl", TwistStamped, queue_size=1)

        # ROS Subscriber
        rospy.Subscriber(ns + "/odom", Odometry, self.odomCB)
        rospy.Subscriber(ns + "/scan", LaserScan, self.scanCB)
        rospy.Subscriber(ns + "/targetBox", PoseStamped, self.detectCB)
        rospy.Subscriber(other_ns + "/targetBox", PoseStamped, 
                         self.otherDetectCB)
        rospy.Subscriber(other_ns + "/odom", Odometry, self.otherOdomCB)

    
    def run(self):
        if not self.has_init():
            return
        scan_flat = np.array(self.scan_norm)
        detect_obs = np.array(self.box_list).reshape(-1)
        other_obs = np.array(self.dirt + self.vel)
        obs = np.concatenate((scan_flat, detect_obs))
        obs = np.concatenate((obs, other_obs))
        if self.use_rnn:
            act, next_state = self.model.predict(obs, state=self.state, 
                                                mask=self.mask, deterministic=True)
            self.state = next_state
            self.mask = [False]
        else:
            act, _ = self.model.predict(obs, deterministic=True)
        cmd_msg = TwistStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.twist.linear.x = act[0] * 3.5
        cmd_msg.twist.linear.y = act[1] * 3.5
        cmd_msg.twist.angular.z = act[2] * 0.5
        self.cmd_pub.publish(cmd_msg)


    def has_init(self):
        return self.scan_norm and self.odom_list and self.box_list
    

    def odomCB(self, msg: Odometry):
        if len(self.odom_list) < WINDOW_SIZE:
            self.odom_list.append(msg)
        else:
            self.odom_list = self.odom_list[1:] + [msg]
        
    def scanCB(self, msg: LaserScan):
        scan = msg.ranges
        scan_norm = [np.exp(-i) for i in scan]
        self.scan_norm = scan_norm
    
    def otherOdomCB(self, msg: Odometry):
        if len(self.odom_other_list) < WINDOW_SIZE:
            self.odom_other_list.append(msg)
        else:
            self.odom_other_list = self.odom_other_list[1:] + [msg]

    def otherDetectCB(self, msg: PoseStamped):
        if not self.odom_other_list:
            return
        t_det = msg.header.stamp.to_sec()
        odoms = sorted(self.odom_other_list,
                       key=lambda x: abs(x.header.stamp.to_sec() - t_det))
        self.odom_other = odoms[0]
        self.box_other = msg

    
    def detectCB(self, msg: PoseStamped):
        if not self.odom_list or not self.odom_other_list:
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
        ## get velocity observation 
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

        ## get detect observation
        umin = msg.pose.orientation.w
        umax = msg.pose.orientation.x
        vmin = msg.pose.orientation.y
        vmax = msg.pose.orientation.z
        tpt = np.array([(umin + umax)/2.0, vmin])
        bpt = np.array([(umin + umax)/2.0, vmax])
        if self.use_esti and self.odom_other_list:
            if abs(t_det - self.box_other.header.stamp.to_sec()) > 0.1:
                rospy.logwarn("Observation time difference between the two drones exceeds threshold..")
            elif self.box_other.pose.orientation.x < 0:
                rospy.logwarn("Other drone missed target..")
            else:
                esti_tpt, esti_bpt = self.esti.esti_by_H1(odom, self.odom_other, self.box_other)
                wpt0 = self.esti.esti_wpt(self.odom_other, self.box_other)
                wpt1 = self.esti.esti_wpt(odom, msg)
                pt1_hat = self.esti.esti_by_proj(wpt0, odom)
                if umin < 0:
                    rospy.logdebug("Drone did not observe target, using estimated values.")
                else:
                    print("diff_t: %.2f | [%.2f, %.2f]"%(
                        abs(t_det - self.box_other.header.stamp.to_sec()),
                        self.box_other.header.stamp.to_sec(),
                        self.odom_other.header.stamp.to_sec()
                    ))
                    print("world_point: [%.2f, %.2f, %.2f] | [%.2f, %.2f, %.2f]"%(
                        wpt0[0], wpt0[1], wpt0[1],
                        wpt1[0], wpt1[1], wpt1[2]
                    ))
                    print("proj_pt: [%.2f, %.2f] | [%.2f, %.2f] | %.2f"%(
                        pt1_hat[0], pt1_hat[1], bpt[0], bpt[1],
                        np.linalg.norm(pt1_hat - bpt)
                    ))
                    print("diff_tpt: %.2f | diff_bpt: %.2f"%(
                        np.linalg.norm(esti_tpt - tpt),
                        np.linalg.norm(esti_bpt - bpt)
                    ))
                    print("tpt: [%.2f, %.2f] | [%.2f, %.2f]"%(
                        tpt[0], tpt[1], esti_tpt[0], esti_tpt[1]
                    ))
                    print("bpt: [%.2f, %.2f] | [%.2f, %.2f]"%(
                        bpt[0], bpt[1], esti_bpt[0], esti_bpt[1]
                    ))
        else:
            pass

        
        bbox_obs = [((umin + umax) / 2.0 - 480.0) / 960.0,
                    ((vmin + vmax) / 2.0 - 260.0) / 540.0,
                    (vmax - vmin - 210.0) / 540.0,
                    msg.pose.position.x, 
                    msg.pose.position.y]
        if not self.box_list:
            self.box_list = [bbox_obs, bbox_obs, bbox_obs]
        else:
            self.box_list = self.box_list[1:] + [bbox_obs]
        dirt_obs = [msg.pose.position.x, 
                    msg.pose.position.y]
        self.dirt = dirt_obs

        # publish RL command
        if not self.has_init():
            return

        other_odom = self.odom_other
        oqw = other_odom.pose.pose.orientation.w
        oqx = other_odom.pose.pose.orientation.x
        oqy = other_odom.pose.pose.orientation.y
        oqz = other_odom.pose.pose.orientation.z
        other_yaw = np.arctan2(2.0 * oqw * oqz + 2.0 * oqx * oqy,
                         oqw * oqw + oqx * oqx - oqy * oqy - oqz * oqz)
        
        beta = other_yaw - yaw
        other_dirt = [np.cos(beta), np.sin(beta)]

        scan_flat = np.array(self.scan_norm)
        detect_obs = np.array(self.box_list).reshape(-1)
        other_obs = np.array(self.vel + other_dirt)
        obs = np.concatenate((scan_flat, detect_obs))
        obs = np.concatenate((obs, other_obs)).reshape(1, -1)
        if self.use_rnn:
            act, next_state = self.model.predict(obs, state=self.state, 
                                                mask=self.mask, deterministic=True)
            self.state = next_state
            self.mask = [False]
            act = act[0]
        else:
            act, _ = self.model.predict(obs, deterministic=True)
        cmd_msg = TwistStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.twist.linear.x = act[0] * 3.5
        cmd_msg.twist.linear.y = act[1] * 3.5
        cmd_msg.twist.angular.z = act[2] * 0.5
        self.cmd_pub.publish(cmd_msg)
        rospy.loginfo("CMD: [%.2f, %.2f, %.2f]"%(cmd_msg.twist.linear.x,
                                                 cmd_msg.twist.linear.y,
                                                 cmd_msg.twist.angular.z))
        
        





if __name__ == "__main__":
    args = parser().parse_args()
    name_space = "uav" + str(args.id)
    rospy.init_node(name_space + "_RL_control")
    track_control = RLTrack(MODEL_NAME, name_space, args.use_esti, USE_RNN)
    rospy.spin()