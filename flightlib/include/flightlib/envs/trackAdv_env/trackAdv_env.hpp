/***
 * @Author: Lac_Creeper
 * @Date: 2025-05-21 20:25:57 +0800
 * @LastEditTime: 2025-06-09 15:42:41 +0800
 * @LastEditors: Lac_Creeper
 * @Description:
 * @FilePath:
 * /flightmare/flightlib/include/flightlib/envs/track_env/track_env.hpp
 */
#pragma once

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
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
#include "flightlib/objects/target.hpp"
#include "flightlib/sensors/detect.hpp"
#include "flightlib/sensors/lidar.hpp"
#include "flightlib/sensors/lidar2D.hpp"

namespace flightlib {

namespace trackAdvenv {

enum Ctl : int {
  // observations
  kObs = 0,
  //
  kLaser = 0,
  kNLaser = 512,
  kDetect1 = 512,
  kNDetect1 = 3,
  kDirt1 = 515,
  kNDirt1 = 2,
  kDetect2 = 517,
  kNDetect2 = 3,
  kDirt2 = 520,
  kNDirt2 = 2,
  kDetect3 = 522,
  kNDetect3 = 3,
  kDirt3 = 525,
  kNDirt3 = 2,
  kState = 527,
  kNState = 3,

  kNObs = 530,

  // control actions
  kAct = 0,
  kNAct = 3,

};

// enum Ctl : int {
//   // observations
//   kObs = 0,
//   //
//   kLaser = 0,
//   kNLaser = 512,
//   kDetect1 = 512,
//   kNDetect1 = 3,
//   kDetect2 = 515,
//   kNDetect2 = 3,
//   kDetect3 = 518,
//   kNDetect3 = 3,
//   kDirt = 521,
//   kNDirt = 2,
//   kState = 523,
//   kNState = 3,

//   kNObs = 526,

//   // control actions
//   kAct = 0,
//   kNAct = 3,

// };
};  // namespace trackAdvenv
class TrackAdvEnv final : public EnvBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TrackAdvEnv();
  TrackAdvEnv(const std::string &cfg_path);
  ~TrackAdvEnv();

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

  friend std::ostream &operator<<(std::ostream &os, const TrackAdvEnv &track_env);

  Vector<3> convVel();

  // For ROS visualization
  void visualizeObstacles(std::vector<std::shared_ptr<Obstacle>>& obstacles);
  void visualizeScan();
  void visualizeOdom();
  void visualizeTarget();

 private:
  // ROS
  std::unique_ptr<ros::NodeHandle> nh_;
  // ros::NodeHandle nh_;
  ros::Publisher map_pub_;
  ros::Publisher odom_pub_;
  ros::Publisher scan_pub_;
  ros::Publisher target_pub_;
  ros::Publisher debug_pub_;

  Scalar target_maxV_;
  Scalar target_V_;
  Scalar tMaxV_;
  Target target_{Vector<3>{3.0f, 0.0f, 0.0f}, TrajectoryType::TRIANGLE,
                 1.0f};
  int target_traj_;
  int load_map_{0};
  int map_num_{0};


  // quadrotor
  std::shared_ptr<Quadrotor> quadrotor_ptr_;
  QuadState quad_state_;
  Command cmd_;
  // Lidar lidar_;
  Lidar2D lidar_{-18, 22, -18, 18, 2 * M_PI, 512, 5.0, 0.2};
  // Lidar2D lidar_{-20, 20, -15, 15, 2 * M_PI, 512, 5.0, 0.2};
  DetectSim detect_;
  Logger logger_{"TrackAdvEnv"};

  int step_num_;
  int reach_count_;

  // Define reward for training
  Scalar detect_coeff_, dist_coeff_, alpha_coeff_, pos_coeff_, theta_coeff_, act_coeff_;
  int use_ros_, fine_turn_, random_;
  Scalar traj_param_;
  Scalar randV_;
  int traj_type_;
  Vector<3> targetInitPose_;
  Scalar last_alpha_, last_dist_, last_theta_;
  bool has_init_reward_;

  // observations and actions (for RL)
  Vector<trackAdvenv::kNObs> track_obs_;
  Vector<trackAdvenv::kNAct> track_act_, last_act_;

  // reward function design (for model-free RL)
  BBox detect_bbox_;
  BBox desired_bbox_;
  Vector<3> target_xyY_;
  Scalar desired_dist_;
  Scalar desired_dirt_;
  bool has_init_obs_;

  // action and observation normalization (for RL)
  Vector<trackAdvenv::kNAct> act_std_;

  YAML::Node cfg_;
  Matrix<3, 2> world_box_;
};


}  // namespace flightlib
