/***
 * @Author: Lac_Creeper
 * @Date: 2026-02-01 17:21:15 +0800
 * @LastEditTime: 2026-02-01 17:21:16 +0800
 * @LastEditors: Lac_Creeper
 * @Description:
 * @FilePath:
 * /flightmare/flightlib/include/flightlib/envs/multiTrack_env/multiTrack_env.hpp
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
#include "flightlib/sensors/detect.hpp"
#include "flightlib/sensors/lidar.hpp"
#include "flightlib/sensors/lidar2D.hpp"

namespace flightlib {

namespace multitrack {

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
};  // namespace multitrack
class MultiTrack final : public EnvBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MultiTrack();
  MultiTrack(const std::string& cfg_path);
  ~MultiTrack();

  // - public OpenAI-gym-style functions
  bool reset(Ref<Vector<>> obs, const bool random = true) override;
  Scalar step(const Ref<Vector<>> act, Ref<Vector<>> obs) override;

  // - public set functions
  bool loadParam(const YAML::Node& cfg);

  // - public get functions
  bool getObs(Ref<Vector<>> obs) override;
  bool getAct(Ref<Vector<>> act) const;
  bool getAct(Command* const cmd) const;

  // - auxiliar functions
  bool isTerminalState(Scalar& reward) override;
  void addObjectsToUnity(std::shared_ptr<UnityBridge> bridge);

  friend std::ostream& operator<<(std::ostream& os,
                                  const MultiTrack& track_env);

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
  ros::Publisher scan_pub_;
  ros::Publisher odom_pub_;
  ros::Publisher odomPub0_;
  ros::Publisher odomPub1_;
  ros::Publisher odomPub2_;
  ros::Publisher scanPub0_;
  ros::Publisher scanPub1_;
  ros::Publisher scanPub2_;
  ros::Publisher target_pub_;
  ros::Publisher debug_pub_;

  Scalar target_maxV_;
  Scalar target_V_;
  Scalar tMaxV_;


  // quadrotor
  std::shared_ptr<Quadrotor> quadrotor_ptr_;
  std::shared_ptr<Quadrotor> drone0_;
  std::shared_ptr<Quadrotor> drone1_;
  std::shared_ptr<Quadrotor> drone2_;
  QuadState quadState0_;
  QuadState quadState1_;
  QuadState quadState2_;
  QuadState quad_state_;
  Command cmd_;
  Command cmd0_;
  Command cmd1_;
  Command cmd2_;
  // Lidar lidar_;
  Lidar2D lidar_{-20, 20, -15, 15, 2 * M_PI, 512, 8.0, 0.25};
  Lidar2D lidar0_{-20, 20, -15, 15, 2 * M_PI, 512, 8.0, 0.25};
  Lidar2D lidar1_{-20, 20, -15, 15, 2 * M_PI, 512, 8.0, 0.25};
  Lidar2D lidar2_{-20, 20, -15, 15, 2 * M_PI, 512, 8.0, 0.25};
  DetectSim detect_;
  DetectSim detect0_;
  DetectSim detect1_;
  DetectSim detect2_;
  Logger logger_{"multiTrack"};

  int step_num_;
  int reach_count_;

  // Define reward for training
  Scalar detect_coeff_, dist_coeff_, alpha_coeff_, pos_coeff_, theta_coeff_,
    act_coeff_;
  int use_ros_, fine_turn_, random_;
  Scalar traj_param_;
  Vector<3> targetInitPose_;
  Scalar last_alpha_, last_dist_, last_theta_;
  bool has_init_reward_;

  // observations and actions (for RL)
  Vector<multitrack::kNObs> track_obs_;
  Vector<multitrack::kNAct> track_act_, last_act_;

  // reward function design (for model-free RL)
  BBox detect_bbox_;
  BBox desired_bbox_;
  Vector<3> target_xyY_;
  Scalar desired_dist_;
  Scalar desired_dirt_;
  bool has_init_obs_;

  // action and observation normalization (for RL)
  Vector<multitrack::kNAct> act_std_;

  YAML::Node cfg_;
  Matrix<3, 2> world_box_;
};


}  // namespace flightlib
