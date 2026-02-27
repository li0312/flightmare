/***
 * @Author: Lac_Creeper
 * @Date: 2026-02-02 00:45:08 +0800
 * @LastEditTime: 2026-02-07 01:54:26 +0800
 * @LastEditors: Lac_Creeper
 * @Description:
 * @FilePath: /flightmare/flightros/include/flightros/multi_track/lac_env.hpp
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
#include "flightlib/sensors/rgb_camera.hpp"

using namespace flightlib;

namespace flightros {

class LacEnv {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  LacEnv(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
  ~LacEnv();

  // callbacks
  void mainLoopCallback(const ros::TimerEvent& event);
  void startCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
  void pubtargetGT();
  void pubtargetDetect();

  bool loadParams(void);

 private:
  // ros nodes
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  // publisher
  ros::Publisher map_pub_;
  ros::Publisher targetGT_pub_;
  ros::Publisher targetDetect_pub_;
  ros::Publisher debug_pub_;

  // subscriber
  ros::Subscriber sub_start_;

  // main loop timer
  ros::Timer timer_main_loop_;

  // Sensor
  Lidar lidar_;

  // Target
  Scalar t0_last_{-1.0};
  Scalar targetMaxV_;
  Scalar vel_param_;
  double time_init_{-1.0};
  bool flag_init_{false};
  Vector<3> target_xyY_;
  Vector<3> target_init_;
  Vector<3> target_last_;

  // Debug
  Timer timer_{"PoseCB"};

  Logger logger_{"lac_env",
                 std::string("/home/lac/fm_test/my_logs/lac_env.log")};

  // auxiliary variables
  Scalar main_loop_freq_{50.0};
};
}  // namespace flightros