/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-05-21 20:25:57 +0800
 * @LastEditTime: 2025-05-31 10:38:04 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /flightmare/flightlib/include/flightlib/envs/track_env/track_env.hpp
 */
#pragma once

#include <yaml-cpp/yaml.h>

#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/command.hpp"
#include "flightlib/common/logger.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/envs/env_base.hpp"
#include "flightlib/objects/quadrotor.hpp"
#include "flightlib/sensors/detect.hpp"
#include "flightlib/sensors/lidar.hpp"

namespace flightlib {

namespace trackenv {

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
  kState = 1539,
  kNState = 3,
  
  kNObs = 1542,

  // control actions
  kAct = 0,
  kNAct = 3,

};
}; // namespace trackenv
class TrackEnv final : public EnvBase {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    TrackEnv();
    TrackEnv(const std::string &cfg_path);
    ~TrackEnv();

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
                                    const TrackEnv &track_env);
    
  private:
    // quadrotor
    std::shared_ptr<Quadrotor> quadrotor_ptr_;
    QuadState quad_state_;
    Command cmd_;
    Lidar lidar_;
    DetectSim detect_;
    Logger logger_{"TrackEnv"};

    int step_num;

    // Define reward for training
    Scalar detect_coeff_, pos_coeff_, theta_coeff_, act_coeff_;

    // observations and actions (for RL)
    Vector<trackenv::kNObs> track_obs_;
    Vector<trackenv::kNAct> track_act_;

    // reward function design (for model-free RL)
    BBox detect_bbox_;
    BBox desired_bbox_;
    Vector<3> target_xyY_;
    Scalar desired_dist_;
    Scalar desired_dirt_;
    bool has_init_obs_;

    // action and observation normalization (for RL)
    Vector<trackenv::kNAct> act_std_;

    YAML::Node cfg_;
    Matrix<3, 2> world_box_;

    

};

  
} // namespace flightlib

