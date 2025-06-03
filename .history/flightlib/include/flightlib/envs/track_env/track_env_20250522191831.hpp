/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-05-21 20:25:57 +0800
 * @LastEditTime: 2025-05-22 19:08:31 +0800
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

namespace flightlib {

namespace trackenv {

enum Ctl : int {
  // observations
  kObs = 0,
  //
  kDetect = 0,
  kNDetect = 3,
  kState = 3,
  kNState = 3,
  kLaser = 6,
  KNLaser = 1536,
  KNObs = 1542,
  // control actions
  kAct = 0,
  kNAct = 3,

};
};
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
    

}






  
} // namespace trackenv
  
} // namespace flightlib

