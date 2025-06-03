/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-05-23 15:11:53 +0800
 * @LastEditTime: 2025-05-26 14:12:39 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /flightmare/flightlib/src/envs/track_env/track_env.cpp
 */
#include "flightlib/envs/track_env/track_env.hpp"

namespace flightlib {

TrackEnv::TrackEnv() 
  : TrackEnv(getenv("FLIGHTMARE_PATH") + 
             std::string("/flightlib/configs/track_env.yaml")) {}

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

  if (random) {
    // randomly reset the quadrotor state
    quad_state_.x(QS::POSX) = uniform_dist_(random_gen_);
    quad_state_.x(QS::POSY) = uniform_dist_(random_gen_);
    quad_state_.x(QS::POSZ) = 0.8f;
    quad_state_.x(QS::ATTW) = uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTX) = uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTY) = uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTZ) = uniform_dist_(random_gen_);
    quad_state_.qx /= quad_state_.qx.norm();
    // check collision
    // reset the target
  }
  quadrotor_ptr_->reset(quad_state_);

  cmd_.t = 0.0;
  cmd_.linear.setZero();
  cmd_.angular.setZero();

  // obtain observations
  has_init_obs = false;
  is_collision_ = false;
  getObs(obs);
  return true;
}

bool TrackEnv::getObs(Ref<Vector<>> obs) {
  quadrotor_ptr_->getState(&quad_state_);
  lidar_.renderLaserScan(quad_state_);
  const auto &scan_data = lidar_.getScan();
  Vector<trackenv::kNLaser1> scan = 
          Vector<trackenv::kNLaser1>::Map(scan_data.data(), scan_data.size());

  detect_.getBBox(quad_state_, detect_bbox_);

  
  if (!has_init_obs) {
    track_obs_.segment<trackenv::kNLaser1>(trackenv::kLaser1) = scan;
    track_obs_.segment<trackenv::kNLaser2>(trackenv::kLaser2) = scan;
    track_obs_.segment<trackenv::kNLaser3>(trackenv::kLaser3) = scan;
    track_obs_.segment<trackenv::kNDetect>(trackenv::kDetect) = 
  }

  track_obs_ << 
  
  return true;
}

Scalar TrackEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs) {
  track_act_ = act.cwiseProduct(act_std_);
  cmd_.t += sim_dt_;
  cmd_.linear.x() = act[0];
  cmd_.linear.y() = act[1];
  cmd_.angular.z() = act[2];
  
  // simulate quadrotor
  quadrotor_ptr_->velocityControlBody(cmd_, sim_dt_);

  // update observations
  getObs(obs);

  // ------------------- reward function design
  // - detection term
  Scalar detect_reward = detect_coeff_ * detect_bbox_.IoU(desired_bbox_);

  // - position term
  Scalar pos_reward = 
    pos_coeff_ * ((quad_state_.x.segment<2>(QS::POS) - 
                  target_xyY_.segment<2>(0)).squaredNorm() - desired_dist_);
  
  // - theta term
  Scalar theta_reward = 
    theta_coeff_;

  // - control action penalty
  Scalar act_reward = act_coeff_ * act.cast<Scalar>().norm();

  Scalar total_reward = 
    detect_reward + pos_reward + theta_reward + act_reward;

  // survival reward
  total_reward += 0.1f;

  return total_reward;
}

bool TrackEnv::isTerminalState(Scalar &reward) {
  if (!detect_bbox_.is_visual) {
    reward = -40;
    return true;
  }
  if (is_collision_) {
    reward = -80;
    return true;
  }
  reward = 0.0f;
  return false;
}

bool TrackEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["track_env"]) {
    sim_dt_ = cfg["track_env"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["track_env"]["max_t"].as<Scalar>();
  } else {
    return false;
  }

  if (cfg["rl"]) {
    // load RL related parameters
    detect_coeff_ = cfg["rl"]["detect_coeff"].as<Scalar>();
    pos_coeff_ = cfg["rl"]["pos_coeff"].as<Scalar>();
    theta_coeff_ = cfg["rl"]["theta_coeff"].as<Scalar>();
    act_coeff_ = cfg["rl"]["act_coeff"].as<Scalar>();
  } else {
    return false;
  }

  return true;
}

bool TrackEnv::getAct(Ref<Vector<>> act) const {
  if (cmd_.t >= 0.0 && track_act_.allFinite()) {
    act = track_act_;
    return true;
  }
  return false;
}

bool TrackEnv::getAct(Command *const cmd) const {
  if (!cmd_.valid()) return false;
  *cmd = cmd_;
  return true;
}

void TrackEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  bridge->addQuadrotor(quadrotor_ptr_);
}

std::ostream &operator<<(std::ostream &os, const TrackEnv &track_env) {
  os.precision(3);
  os << "Tracking Environment:\n"
     << "obs dim =            [" << track_env.obs_dim_ << "]\n"
     << "act dim =            [" << track_env.act_dim_ << "]\n"
     << "sim dt =             [" << track_env.sim_dt_ << "]\n"
     << "max_t =              [" << track_env.max_t_ << "]\n"
     << "act_std =            [" << track_env.act_std_.transpose() << std::endl;
  os.precision();
  return os;
}


} // namespace flightlib
