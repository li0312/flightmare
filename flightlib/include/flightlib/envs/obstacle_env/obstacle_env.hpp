/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-06-16 12:28:39 +0800
 * @LastEditTime: 2025-06-25 13:52:56 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /flightmare/flightlib/include/flightlib/envs/obstacle_env/obstacle_env.hpp
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
#include "flightlib/sensors/lidar2D.hpp"

namespace flightlib {

namespace obstenv {

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
  kNDetect = 2,
  kState = 1538,
  kNState = 2,

  kNObs = 1540,

  // control actions
  kAct = 0,
  kNAct = 2,

};
};  // namespace ObstacleEnv
class ObstacleEnv final : public EnvBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ObstacleEnv();
  ObstacleEnv(const std::string &cfg_path);
  ~ObstacleEnv();

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

  friend std::ostream &operator<<(std::ostream &os, const ObstacleEnv &track_env);

  // For ROS visualization
  void visualizeObstacles(std::vector<std::shared_ptr<Obstacle>> &obstacles);
  void visualizeScan();
  void visualizeOdom();
  void visualizeTarget();
  Vector<2> convGoal();
  Vector<2> convVel();

 private:
  // ROS
  std::unique_ptr<ros::NodeHandle> nh_;
  // ros::NodeHandle nh_;
  ros::Publisher map_pub_;
  ros::Publisher odom_pub_;
  ros::Publisher scan_pub_;
  ros::Publisher target_pub_;


  // quadrotor
  std::shared_ptr<Quadrotor> quadrotor_ptr_;
  QuadState quad_state_;
  Command cmd_;
  // Lidar lidar_;
  Lidar2D lidar_{20, 20, 2 * M_PI, 512, 10.0, 0.4};
  Logger logger_{"ObstacleEnv"};

  int step_num_;
  bool has_init_obs_;
  int use_ros_;

  // Define reward for training
  Scalar last_dist_, last_alpha_;

  // observations and actions (for RL)
  Vector<obstenv::kNObs> obst_obs_;
  Vector<obstenv::kNAct> obst_act_;

  // reward function design (for model-free RL)
  Vector<2> target_xy_;

  // action and observation normalization (for RL)
  Vector<obstenv::kNAct> act_std_;

  YAML::Node cfg_;
  Matrix<3, 2> world_box_;
};


}  // namespace flightlib
