/***
 * @Author: Lac_Creeper
 * @Date: 2025-05-21 20:25:57 +0800
 * @LastEditTime: 2025-06-09 15:42:41 +0800
 * @LastEditors: Lac_Creeper
 * @Description:
 * @FilePath:
 * /flightmare/flightlib/include/flightlib/envs/gimbal_env/gimbal_env.hpp
 */
#pragma once

#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <yaml-cpp/yaml.h>

#include <mutex>

#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/command.hpp"
#include "flightlib/common/logger.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/envs/env_base.hpp"
#include "flightlib/objects/quadrotor.hpp"
#include "flightlib/sensors/detect.hpp"
#include "flightlib/sensors/lidar.hpp"
#include "flightlib/sensors/lidar2D.hpp"

namespace flightlib {

namespace gtenv {

enum Ctl : int {
  // observations
  kObs = 0,
  //
  kLaser1 = 0,
  kNLaser1 = 512,
  kLaser2 = 512,
  kNLaser2 = 512,
  kLaser3 = 1024,
  kNLaser3 = 512,
  kDetect = 1536,
  kNDetect = 3,
  kDirt = 1539,
  kNDirt = 2,
  kState = 1541,
  kNState = 3,

  kNObs = 1544,

  // control actions
  kAct = 0,
  kNAct = 3,

};
};  // namespace gtenv
class GimbalEnv final : public EnvBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  GimbalEnv();
  GimbalEnv(const std::string &cfg_path);
  ~GimbalEnv();

  // - public OpenAI-gym-style functions
  bool reset(Ref<Vector<>> obs, const bool random = true) override;
  Scalar step(const Ref<Vector<>> act, Ref<Vector<>> obs) override;

  // - public set functions
  bool loadParam(const YAML::Node &cfg);

  // - public get functions
  bool getObs(Ref<Vector<>> obs) override;
  bool getAct(Ref<Vector<>> act) const;
  bool getAct(Command *const cmd) const;

  // - auxiliar functions
  bool isTerminalState(Scalar &reward) override;
  void addObjectsToUnity(std::shared_ptr<UnityBridge> bridge);

  friend std::ostream &operator<<(std::ostream &os,
                                  const GimbalEnv &gimbal_env);

  Vector<3> convVel();

  // For ROS visualization
  void visualizeObstacles(std::vector<std::shared_ptr<Obstacle>> &obstacles);
  void visualizeScan();
  void visualizeOdom();
  void visualizeGimbal();
  void visualizeTarget();
  Matrix<3, 3> yawToR(Scalar yaw);
  bool isReach(Scalar theta);

 private:
  // ROS
  std::unique_ptr<ros::NodeHandle> nh_;
  // ros::NodeHandle nh_;
  ros::Publisher map_pub_;
  ros::Publisher odom_pub_;
  ros::Publisher gimbalPose_pub_;
  ros::Publisher scan_pub_;
  ros::Publisher target_pub_;


  // quadrotor
  std::shared_ptr<Quadrotor> quadrotor_ptr_;
  QuadState quad_state_;
  Command cmd_;
  // Lidar lidar_;
  Lidar2D lidar_{-20, 20, -20, 20, 2 * M_PI, 512, 10.0, 0.4};
  DetectSim detect_;
  Logger logger_{"GimbalEnv"};
  Scalar gimbalY_, gimbalWz_;

  int step_num_;
  int reach_count_;

  // Define reward for training
  Scalar detect_coeff_, pos_coeff_, theta_coeff_, act_coeff_;
  int use_ros_;
  Vector<3> targetInitPose_;
  Scalar last_alpha_, last_dist_, last_theta_;
  bool has_init_reward_;

  // observations and actions (for RL)
  Vector<gtenv::kNObs> gimbal_obs_;
  Vector<gtenv::kNAct> gimbal_act_;

  // reward function design (for model-free RL)
  BBox detect_bbox_;
  BBox desired_bbox_;
  Vector<3> target_xyY_;
  Scalar desired_dist_;
  Scalar desired_dirt_;
  bool has_init_obs_;

  // action and observation normalization (for RL)
  Vector<gtenv::kNAct> act_std_;

  YAML::Node cfg_;
  Matrix<3, 2> world_box_;
};


}  // namespace flightlib
