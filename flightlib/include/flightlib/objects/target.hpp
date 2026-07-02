/*** 
 * @Author: Lac_Creeper
 * @Date: 2026-03-04 10:08:02 +0800
 * @LastEditTime: 2026-03-23 17:12:08 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /src/flightmare/flightlib/include/flightlib/objects/target.hpp
 */
#pragma once

# include <random>

#include "flightlib/common/types.hpp"
#include "flightlib/common/logger.hpp"

namespace flightlib {

enum class TrajectoryType {
  ELLIPSE,
  TRIANGLE,
  FIGURE_EIGHT,
  RANDOM_WALK
};

enum class VelocityMode { CONSTANT, MAX_LIMIT };

class Target {
 public:
  Target();
  Target(const Vector<3>& init_pose, const TrajectoryType& traj_type,
         const Scalar& max_speed);
  ~Target();

  void setTrajectoryType(const TrajectoryType& traj_type) { traj_type_ = traj_type; };
  void setMaxSpeed(const Scalar& max_speed) { max_speed_ = max_speed; };


  void update(Scalar sim_time);
  Vector<3> getPose() const { return target_xyY_; };
  TrajectoryType getTrajectoryType() const { return traj_type_; };
  void reset(const Vector<3>& init_pose, const TrajectoryType& traj_type, const Scalar& max_speed) {
    target_init_ = init_pose;
    target_xyY_ = init_pose;
    target_last_ = init_pose;
    traj_type_ = traj_type;
    max_speed_ = max_speed;
    reset_flag_ = true;
  };


 private:
  Vector<3> target_init_;
  Vector<3> target_xyY_;
  Vector<3> target_last_; 
  TrajectoryType traj_type_;
  Scalar max_speed_;
  bool reset_flag_;

  Logger logger_{"Target"};
  std::random_device rd_;
  std::mt19937 rng_{rd_()};
  std::uniform_real_distribution<Scalar> uniform_dist_{-1.0, 1.0};
  std::uniform_real_distribution<Scalar> uniform_01_dist_{0.0, 1.0};
};


}  // namespace flightlib