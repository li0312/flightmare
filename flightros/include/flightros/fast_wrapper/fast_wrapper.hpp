/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-11-17 21:02:13 +0800
 * @LastEditTime: 2026-01-17 23:52:28 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /src/flightmare/flightros/include/flightros/fast_wrapper/fast_wrapper.hpp
 */

#pragma once

#include <memory>

// ros
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
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
#include "flightlib/sensors/rgb_camera.hpp"
#include "flightlib/sensors/detect.hpp"
#include "flightlib/sensors/lidar.hpp"

using namespace flightlib;

namespace flightros {

class FastWrapper {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FastWrapper(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
  ~FastWrapper();

  // callbacks
  void mainLoopCallback(const ros::TimerEvent& event);
  void poseCallback(const nav_msgs::Odometry::ConstPtr& msg);
  void startCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
  void pubtargetGT();
  void pubtargetDetect();

  bool setUnity(const bool render);
  bool connectUnity(void);
  bool loadParams(void);

 private:
  // ros nodes
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  // publisher
  ros::Publisher map_pub_;
  ros::Publisher scan_pub_;
  ros::Publisher cloud_pub_;
  ros::Publisher targetGT_pub_;
  ros::Publisher targetDetect_pub_;
  ros::Publisher debug_pub_;

  // subscriber
  ros::Subscriber sub_state_est_;
  ros::Subscriber sub_start_;

  geometry_msgs::TransformStamped transform_;

  // main loop timer
  ros::Timer timer_main_loop_;

  // Sensor
  bool has_odom_{false};
  Lidar lidar_;
  DetectSim detect_;
  sensor_msgs::LaserScan scan_msg_;

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
  
  Logger logger_{"fast_wrapper", std::string("/home/lac/fm_test/my_logs/fast.log")};

  // unity quadrotor
  std::shared_ptr<Quadrotor> quad_ptr_;
  std::shared_ptr<RGBCamera> rgb_camera_;
  QuadState quad_state_;

  // Flightmare(Unity3D)
  std::shared_ptr<UnityBridge> unity_bridge_ptr_;
  SceneID scene_id_{UnityScene::WAREHOUSE};
  bool unity_ready_{false};
  bool unity_render_{false};
  RenderMessage_t unity_output_;
  uint16_t receive_id_{0};

  // auxiliary variables
  Scalar main_loop_freq_{50.0};
};
}  // namespace flightros