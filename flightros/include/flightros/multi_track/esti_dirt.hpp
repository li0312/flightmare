/***
 * @Author: Lac_Creeper
 * @Date: 2026-02-27 02:35:38 +0800
 * @LastEditTime: 2026-02-27 02:46:54 +0800
 * @LastEditors: Lac_Creeper
 * @Description:
 * @FilePath: /flightmare/flightros/include/flightros/multi_track/esti_dirt.hpp
 */
#pragma once

#include <eigen3/Eigen/Eigen>
#include <memory>
// ros
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>



class EstiDirt {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EstiDirt(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
  ~EstiDirt();

  // callbacks
  void mainLoopCallback(const ros::TimeEvent& event);
  void odom0Callback(const nav_msgs::Odometry::ConstPtr& msg);
  void odom1Callback(const nav_msgs::Odometry::ConstPtr& msg);
  void box0Callback(const geometry_msgs::PoseStamped::ConstPtr& msg);
  void box1Callback(const geometry_msgs::PoseStamped::ConstPtr& msg);

  Vector<3> quat2euler(const geometry_msgs::Quaternion& q) const;
  bool loadParams(void);

 private:
  // ros nodes
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  // publisher
  ros::Publisher esti0_pub;
  ros::Publisher esti1_pub;

  // subscriber
  ros::Subscriber sub_odom0_;
  ros::Subscriber sub_odom1_;
  ros::Subscriber sub_box0_;
  ros::Subscriber sub_box1_;

  // main loop timer
  ros::Timer timer_main_loop_;




}