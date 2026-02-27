/***
 * @Author: Lac_Creeper
 * @Date: 2026-02-05 02:57:52 +0800
 * @LastEditTime: 2026-02-05 02:57:54 +0800
 * @LastEditors: Lac_Creeper
 * @Description:
 * @FilePath: /flightmare/flightros/src/multi_track/lac_env.cpp
 */
#include "flightros/multi_track/lac_env.hpp"

namespace flightros {

LacEnv::LacEnv(const ros::NodeHandle& nh, const ros::NodeHandle& pnh)
  : nh_(nh), pnh_(pnh), vel_param_(1.0), main_loop_freq_(50.0) {
  // load parameters
  if (!loadParams()) {
    ROS_WARN("[%s] Could not load all parameters.",
             pnh_.getNamespace().c_str());
  } else {
    ROS_INFO("[%s] Loaded all parameters.", pnh_.getNamespace().c_str());
  }
  // initialization parameter
  targetMaxV_ = 0.0;

  // initialization
  target_init_.setZero();
  target_xyY_ = target_init_;

  // initialize publisher
  map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("global_map", 1);
  targetGT_pub_ = nh_.advertise<nav_msgs::Odometry>("targetGT", 1);
  targetDetect_pub_ = nh_.advertise<nav_msgs::Odometry>("car_state", 1);
  debug_pub_ =
    nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("debug", 1);

  // initialize subscriber call backs
  sub_start_ =
    nh_.subscribe("/move_base_simple/goal", 1, &LacEnv::startCallback, this);

  // wait until the gazebo and unity are loaded
  ros::Duration(5.0).sleep();

  // Publish global map
  const auto& global_map = lidar_.getGlobalMap();
  sensor_msgs::PointCloud2 global_map_msg;
  pcl::toROSMsg(global_map, global_map_msg);
  global_map_msg.header.frame_id = "world";
  ROS_INFO("Map point size = %ld", global_map.size());
  for (int count = 0; count < 2; ++count) {
    map_pub_.publish(global_map_msg);
    ros::Duration(0.2).sleep();
  }

  timer_main_loop_ = nh_.createTimer(ros::Rate(main_loop_freq_),
                                     &LacEnv::mainLoopCallback, this);
  ROS_INFO("LacEnv init success..");
}

LacEnv::~LacEnv() {}


void LacEnv::startCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
  ROS_DEBUG("t0: %.2f", ros::Time::now().toSec());
  time_init_ = ros::Time::now().toSec();
  ROS_DEBUG("t1: %.2f", time_init_);
  ROS_DEBUG("t2: %.2f", ros::Time::now().toSec());
  flag_init_ = true;
}

void LacEnv::mainLoopCallback(const ros::TimerEvent& event) {
  // update target
  if (flag_init_) {
    Scalar t0 = static_cast<float>(ros::Time::now().toSec() - time_init_);
    if (t0 > 100) {
      ROS_DEBUG("t0: %.2f | vMax: %.2f", t0, targetMaxV_);
    }
    // ROS_DEBUG("t0: %.2f", t0);
    // Scalar traj_param = sqrt(vel_param_);
    Scalar traj_param = 1.0;
    Scalar omega = 1.41 * M_PI / 100.0 * vel_param_;
    target_xyY_(0) = target_init_(0) + 16.0 * traj_param * sin(omega * t0);
    target_xyY_(1) = target_init_(1) + 8.0 * traj_param * sin(2 * omega * t0);
    Scalar dx = 16.0 * traj_param * omega * cos(omega * t0);
    Scalar dy = 16.0 * traj_param * omega * cos(2.0 * omega * t0);
    target_xyY_(2) = atan2(dy, dx);
    Scalar speed_0 = sqrt(dx * dx + dy * dy);
    if (targetMaxV_ < speed_0) targetMaxV_ = speed_0;

    if (t0_last_ > 0) {
      Scalar speed_1 =
        (target_xyY_ - target_last_).head(2).norm() / (t0 - t0_last_);
      logger_.info(
        "target vel: %.2f | vel1: %.2f | Max vel: %.2f\nPose: [%.2f, %.2f, "
        "%.2f]",
        speed_0, speed_1, targetMaxV_, target_xyY_(0), target_xyY_(1),
        target_xyY_(2));
      // ROS_INFO("target vel: %.2f | vel1: %.2f | Max vel: %.2f", speed_0,
      //  speed_1, targetMaxV_);
    }
    t0_last_ = t0;
    target_last_ = target_xyY_;

    pubtargetGT();

  } else {
    pubtargetGT();
  }

  return;
}


void LacEnv::pubtargetGT() {
  nav_msgs::Odometry target;
  target.header.stamp = ros::Time::now();
  target.header.frame_id = "world";
  target.pose.pose.position.x = target_xyY_.x();
  target.pose.pose.position.y = target_xyY_.y();
  target.pose.pose.position.z = 0.0;
  target.pose.pose.orientation.w = std::cos(target_xyY_.z() / 2.0);
  target.pose.pose.orientation.x = 0.0;
  target.pose.pose.orientation.y = 0.0;
  target.pose.pose.orientation.z = std::sin(target_xyY_.z() / 2.0);
  target.twist.twist.linear.x = 0.0;
  target.twist.twist.linear.y = 0.0;
  target.twist.twist.angular.z = 0.0;
  targetGT_pub_.publish(target);
}
void LacEnv::pubtargetDetect() {
  nav_msgs::Odometry target;
  target.header.stamp = ros::Time::now();
  target.header.frame_id = "world";
  target.pose.pose.position.x = target_xyY_.x();
  target.pose.pose.position.y = target_xyY_.y();
  target.pose.pose.position.z = 0.0;

  target.pose.pose.orientation.w = std::cos(target_xyY_.z() / 2.0);
  target.pose.pose.orientation.x = 0.0;
  target.pose.pose.orientation.y = 0.0;
  target.pose.pose.orientation.z = std::sin(target_xyY_.z() / 2.0);
  target.twist.twist.linear.x = 0.0;
  target.twist.twist.linear.y = 0.0;
  target.twist.twist.angular.z = 0.0;
  targetDetect_pub_.publish(target);
}


bool LacEnv::loadParams(void) {
  // load parameters
  quadrotor_common::getParam("main_loop_freq", main_loop_freq_, pnh_);
  quadrotor_common::getParam("vel", vel_param_, pnh_);

  return true;
}

}  // namespace flightros