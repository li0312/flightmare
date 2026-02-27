/***
 * @Author: Lac_Creeper
 * @Date: 2026-02-05 03:16:08 +0800
 * @LastEditTime: 2026-02-05 03:16:10 +0800
 * @LastEditors: Lac_Creeper
 * @Description:
 * @FilePath: /flightmare/flightros/include/flightros/multi_track/lac_drone.hpp
 */
#pragma once

#include <memory>

// ros
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

// rpg quadrotor
#include <autopilot/autopilot_helper.h>
#include <autopilot/autopilot_states.h>
#include <quadrotor_common/parameter_helper.h>
#include <quadrotor_msgs/AutopilotFeedback.h>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/objects/quadrotor.hpp"
#include "flightlib/sensors/detect.hpp"
#include "flightlib/sensors/lidar.hpp"
#include "flightlib/sensors/lidar2D.hpp"
#include "flightlib/sensors/rgb_camera.hpp"

using namespace flightlib;

namespace flightros {

class LacDrone {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  LacDrone(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
  ~LacDrone();

  // callbacks
  void mainLoopCallback(const ros::TimerEvent& event);
  void targetCallback(const nav_msgs::Odometry::ConstPtr& msg);
  // void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);
  void cmdCallback(const geometry_msgs::TwistStamped::ConstPtr& msg);
  void startCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
  void pubOdom();
  void pubtargetGT();
  void pubtargetDetect();

  Vector<3> quat2euler(const geometry_msgs::Quaternion& q) const;
  bool loadParams(void);

 private:
  // Drone ID
  int droneID_;
  int droneNum_;
  tf2_ros::TransformBroadcaster br_world2laser_;

  // ros nodes
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  // publisher
  ros::Publisher odom_pub_;
  ros::Publisher cloud_pub_;
  ros::Publisher scan_pub_;
  ros::Publisher scan2d_pub_;
  ros::Publisher targetBox_pub_;

  // subscriber
  ros::Subscriber sub_cmd_;
  ros::Subscriber sub_odom_;
  ros::Subscriber sub_targetGT_;
  ros::Subscriber sub_start_;
  ros::Subscriber sub_box_;

  geometry_msgs::TransformStamped transform_;

  // main loop timer
  ros::Timer timer_main_loop_;

  // Sensor
  bool has_target_{false};
  Lidar lidar_;
  Lidar2D lidar2d_{40, 30, 2 * M_PI, 512, 8.0, 0.25};
  DetectSim detect_;
  sensor_msgs::LaserScan scan_msg_;
  sensor_msgs::LaserScan scan2d_msg_;

  // Target
  Scalar t0_last_{-1.0};
  Scalar targetMaxV_;
  Scalar vel_param_;
  double time_init_{-1.0};
  bool flag_init_{false};
  Vector<3> target_xyY_;
  Vector<3> target_init_;
  Vector<3> target_last_;
  BBox detect_bbox_;

  // Debug
  Timer timer_{"PoseCB"};

  Logger logger_{"fast_wrapper",
                 std::string("/home/lac/fm_test/my_logs/fast.log")};

  // unity quadrotor
  std::shared_ptr<Quadrotor> quad_ptr_;
  QuadState quad_state_;
  Command cmd_;
  double last_ctrl_t_;


  // auxiliary variables
  Scalar main_loop_freq_{50.0};
};
}  // namespace flightros