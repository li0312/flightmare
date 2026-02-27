/***
 * @Author: Lac_Creeper
 * @Date: 2026-02-05 03:16:20 +0800
 * @LastEditTime: 2026-02-05 03:16:21 +0800
 * @LastEditors: Lac_Creeper
 * @Description:
 * @FilePath: /flightmare/flightros/src/multi_track/lac_drone.cpp
 */
#include "flightros/multi_track/lac_drone.hpp"

namespace flightros {

LacDrone::LacDrone(const ros::NodeHandle& nh, const ros::NodeHandle& pnh)
  : nh_(nh),
    pnh_(pnh),
    droneID_(-1),
    droneNum_(-1),
    vel_param_(1.0),
    main_loop_freq_(50.0) {
  // load parameters
  if (!loadParams()) {
    ROS_WARN("[%s] Could not load all parameters.",
             pnh_.getNamespace().c_str());
  } else {
    ROS_INFO("[%s] Loaded all parameters.", pnh_.getNamespace().c_str());
  }

  // quad initialization
  quad_ptr_ = std::make_shared<Quadrotor>();
  quad_state_.setZero();
  Scalar yaw = 0.0;
  if (droneID_ < 0) {
    ROS_WARN("DroneID load failure!!");
  } else if (droneID_ == 0) {
    quad_state_.x[QS::POSX] = -3.0;
  } else if (droneID_ == 1) {
    quad_state_.x[QS::POSY] = -3.0;
    yaw = M_PI_2;
  }
  quad_state_.x[QS::POSZ] = 0.8;
  quad_state_.x[QS::ATTW] = std::cos(yaw / 2.0);
  quad_state_.x[QS::ATTZ] = std::sin(yaw / 2.0);
  quad_state_.qx /= quad_state_.qx.norm();
  quad_ptr_->reset(quad_state_);

  // target initialization
  target_init_.setZero();
  target_xyY_ = target_init_;


  // initialize publisher
  odom_pub_ = nh_.advertise<nav_msgs::Odometry>("odom", 1);
  cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("local_cloud", 1);
  scan_pub_ = nh_.advertise<sensor_msgs::LaserScan>("scan2", 1);
  scan2d_pub_ = nh_.advertise<sensor_msgs::LaserScan>("scan", 1);
  targetBox_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("targetBox", 1);
  // debug_pub_ =
  //   nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("debug", 1);


  // initialize subscriber call backs
  sub_cmd_ = nh_.subscribe("vel_ctrl", 1, &LacDrone::cmdCallback, this);
  // sub_odom_ = nh_.subscribe("other_odom", 1, &LacDrone::odomCallback, this);
  sub_targetGT_ = nh_.subscribe("targetGT", 1, &LacDrone::targetCallback, this);
  sub_start_ =
    nh_.subscribe("/move_base_simple/goal", 1, &LacDrone::startCallback, this);


  // wait until the gazebo is loaded
  ros::Duration(5.0).sleep();

  // lidar2D
  lidar2d_.loadMap("/home/lac/fm_test/my_logs/lac_map.log");

  // publish global map
  ScanParams scan_params = lidar_.getScanParams();
  scan_msg_.angle_min = scan_params.angle_min;
  scan_msg_.angle_max = scan_params.angle_max;
  scan_msg_.angle_increment = scan_params.angle_increment;
  scan_msg_.range_min = scan_params.range_min;
  scan_msg_.range_max = scan_params.range_max;
  scan_msg_.header.frame_id = "uav" + std::to_string(droneID_);

  timer_main_loop_ = nh_.createTimer(ros::Rate(main_loop_freq_),
                                     &LacDrone::mainLoopCallback, this);

  last_ctrl_t_ = ros::Time::now().toSec();
  cmd_.t = 0.0;
  cmd_.linear.setZero();
  cmd_.angular.setZero();
  ROS_INFO("LacDrone_%d init success..", droneID_);
}

LacDrone::~LacDrone() {}

void LacDrone::cmdCallback(const geometry_msgs::TwistStamped::ConstPtr& msg) {
  Scalar dt = (Scalar)(msg->header.stamp.toSec() - last_ctrl_t_);
  last_ctrl_t_ = msg->header.stamp.toSec();
  ROS_DEBUG("Dt0: %.2f | %.2f", (msg->header.stamp.toSec() - last_ctrl_t_), dt);

  if (dt > 0.1) return;
  // quad_ptr_->velocityControlBody(cmd_, dt);
  cmd_.t += dt;
  cmd_.linear.x() = msg->twist.linear.x;
  cmd_.linear.y() = msg->twist.linear.y;
  cmd_.angular.z() = msg->twist.angular.z;
  quad_ptr_->simpleVelControlBody(cmd_, dt);
  ROS_INFO("CMD: [%.2f, %.2f, %.2f] | %.2f", cmd_.linear.x(), cmd_.linear.y(),
           cmd_.angular.z(), dt);
  quad_ptr_->getState(&quad_state_);
}

void LacDrone::targetCallback(const nav_msgs::Odometry::ConstPtr& msg) {
  if (!has_target_) has_target_ = true;
  target_xyY_(0) = (Scalar)msg->pose.pose.position.x;
  target_xyY_(1) = (Scalar)msg->pose.pose.position.y;
  target_xyY_(2) = quat2euler(msg->pose.pose.orientation).z();
}

void LacDrone::startCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
  time_init_ = ros::Time::now().toSec();
  ROS_DEBUG("time: %.2f", time_init_);
  ROS_DEBUG("time1: %.2f", (Scalar)msg->header.stamp.toSec());
  flag_init_ = true;
}

void LacDrone::mainLoopCallback(const ros::TimerEvent& event) {
  timer_.tic();

  // Publish drone odometry
  pubOdom();

  // Lidar
  lidar_.renderLaserScanG(quad_state_);
  if (lidar_.isCollision()) {
    ROS_ERROR("Drone collision!!");
  }
  const auto& local_cloud = lidar_.getlocalCloud();
  sensor_msgs::PointCloud2 local_cloud_msg;
  pcl::toROSMsg(local_cloud, local_cloud_msg);
  local_cloud_msg.header.stamp = ros::Time::now();
  local_cloud_msg.header.frame_id = "uav" + std::to_string(droneID_);

  sensor_msgs::PointCloud2 transformed_cloud;
  pcl_ros::transformPointCloud("world", transform_.transform, local_cloud_msg,
                               transformed_cloud);
  transformed_cloud.header.stamp = ros::Time::now();
  transformed_cloud.header.frame_id = "world";
  cloud_pub_.publish(transformed_cloud);

  const auto& scan_data = lidar_.getScan();
  scan_msg_.ranges = scan_data;
  scan_msg_.header.stamp = local_cloud_msg.header.stamp;
  scan_pub_.publish(scan_msg_);

  // Lidar2D
  sensor_msgs::LaserScan scan2d;
  scan2d.header.stamp = local_cloud_msg.header.stamp;
  scan2d.header.frame_id = "uav" + std::to_string(droneID_);
  scan2d.angle_min = -M_PI;
  scan2d.angle_max = M_PI;
  scan2d.angle_increment = 2 * M_PI / (512 - 1);
  scan2d.time_increment = 0;
  scan2d.scan_time = 0.1;
  scan2d.range_min = 0.1;
  scan2d.range_max = 6.0;
  bool has_collision_2d =
    lidar2d_.simulateLidar(quad_state_, scan2d.ranges, 0.2, false);
  scan2d_pub_.publish(scan2d);


  if (!has_target_) return;
  detect_.updateTarget(target_xyY_);
  BBox result_bbox;
  bool is_valid = detect_.getBBoxG(quad_state_, result_bbox);
  if (is_valid) {
    Scalar yaw = quad_state_.euler_xyz().z();
    Scalar target_yaw = target_xyY_.z();
    Scalar theta = target_xyY_.z() - yaw;
    if (droneID_ == 1) {
      theta += M_PI_2;
    }
    geometry_msgs::PoseStamped box_msg;
    box_msg.header.stamp = local_cloud_msg.header.stamp;
    box_msg.pose.orientation.w = result_bbox.u_min;
    box_msg.pose.orientation.x = result_bbox.u_max;
    box_msg.pose.orientation.y = result_bbox.v_min;
    box_msg.pose.orientation.z = result_bbox.v_max;
    box_msg.pose.position.x = std::cos(theta);
    box_msg.pose.position.y = std::sin(theta);
    targetBox_pub_.publish(box_msg);
    Scalar dirt_theta = cos(yaw) * cos(target_yaw) + sin(yaw) * sin(target_yaw);
    ROS_INFO("[UAV_%d]: dirt_theta: %.3f", droneID_, dirt_theta);
    ROS_INFO("[UAV_%d]: theta: %.3f", droneID_, theta);
  }

  timer_.toc();
  logger_.debug("timer: %.3f.", timer_.mean());
}

void LacDrone::pubOdom() {
  quad_ptr_->getState(&quad_state_);
  nav_msgs::Odometry odom_msg;
  odom_msg.header.stamp = ros::Time::now();
  odom_msg.header.frame_id = "world";
  odom_msg.pose.pose.position.x = (double)quad_state_.x[QS::POSX];
  odom_msg.pose.pose.position.y = (double)quad_state_.x[QS::POSY];
  odom_msg.pose.pose.position.z = (double)quad_state_.x[QS::POSZ];
  odom_msg.pose.pose.orientation.w = (double)quad_state_.x[QS::ATTW];
  odom_msg.pose.pose.orientation.x = (double)quad_state_.x[QS::ATTX];
  odom_msg.pose.pose.orientation.y = (double)quad_state_.x[QS::ATTY];
  odom_msg.pose.pose.orientation.z = (double)quad_state_.x[QS::ATTZ];
  odom_msg.twist.twist.linear.x = (double)quad_state_.x[QS::VELX];
  odom_msg.twist.twist.linear.y = (double)quad_state_.x[QS::VELY];
  odom_msg.twist.twist.angular.z = (double)quad_state_.x[QS::OMEZ];
  odom_pub_.publish(odom_msg);

  geometry_msgs::TransformStamped transformStamped;
  transformStamped.header.stamp = odom_msg.header.stamp;
  transformStamped.header.frame_id = "world";
  transformStamped.child_frame_id = "uav" + std::to_string(droneID_);
  transformStamped.transform.translation.x = odom_msg.pose.pose.position.x;
  transformStamped.transform.translation.y = odom_msg.pose.pose.position.y;
  transformStamped.transform.translation.z = odom_msg.pose.pose.position.z;
  Scalar yaw = quad_state_.euler_xyz().z();
  transformStamped.transform.rotation.w = std::cos(yaw / 2.0);
  transformStamped.transform.rotation.x = 0.0;
  transformStamped.transform.rotation.y = 0.0;
  transformStamped.transform.rotation.z = std::sin(yaw / 2.0);
  br_world2laser_.sendTransform(transformStamped);
  transform_ = transformStamped;
}


Vector<3> LacDrone::quat2euler(const geometry_msgs::Quaternion& q) const {
  Quaternion quat(q.w, q.x, q.y, q.z);
  Vector<3> euler;
  euler.x() = std::atan2(2 * quat.w() * quat.x() + 2 * quat.y() * quat.z(),
                         quat.w() * quat.w() - quat.x() * quat.x() -
                           quat.y() * quat.y() + quat.z() * quat.z());
  euler.y() = -std::asin(2 * quat.x() * quat.z() - 2 * quat.w() * quat.y());
  euler.z() = std::atan2(2 * quat.w() * quat.z() + 2 * quat.x() * quat.y(),
                         quat.w() * quat.w() + quat.x() * quat.x() -
                           quat.y() * quat.y() - quat.z() * quat.z());
  return euler;
}


bool LacDrone::loadParams(void) {
  // load parameters
  quadrotor_common::getParam("main_loop_freq", main_loop_freq_, pnh_);
  quadrotor_common::getParam("drone_id", droneID_, pnh_);
  quadrotor_common::getParam("drone_num", droneNum_, pnh_);
  quadrotor_common::getParam("vel", vel_param_, pnh_);

  return true;
}

}  // namespace flightros