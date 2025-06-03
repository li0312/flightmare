/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-05-23 15:11:53 +0800
 * @LastEditTime: 2025-05-23 15:26:38 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /flightmare/flightlib/src/envs/track_env/track_env.cpp
 */
#include "flightlib/envs/track_env/track_env.hpp"

namespace flightlib {

TrackEnv::TrackEnv(const std::string &cfg_path)
  : EnvBase(),
    detect_coeff_{0.0f},
    pos_coeff_{0.0f},
    theta_coeff_{0.0f},
    act_coeff_{0.0f} {
  // load configuration file
  YAML::Node cfg_ = YAML::LoadFile(cfg_path);

  quadrotor_ptr_ = std::make_shared<Quadrotor>();
  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quadrotor_ptr_->updateDynamics(dynamics);

  // define a world box
  world_box_ << -20, 20, -20, 20, 0, 20;
  if (!quadrotor_ptr_->setWorldBox(world_box_)) {
    logger_.error("Cannot set world box!!");
  }

  // define input and output dimension for the environment
  obs_dim_ = trackenv::kNObs;
  act_dim_ = trackenv::kNAct;

  // load parameters
  loadParam(cfg_);
}

TrackEnv::~TrackEnv() {}

bool TrackEnv::reset(Ref<Vector<>> obs, const bool random) {
  quad_state_.setZero();
  track_act_.setZero();



}


} // namespace flightlib
