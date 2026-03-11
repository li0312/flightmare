/*** 
 * @Author: Lac_Creeper
 * @Date: 2026-03-04 10:08:02 +0800
 * @LastEditTime: 2026-03-09 17:32:09 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /src/flightmare/flightlib/include/flightlib/objects/target.hpp
 */
#pragma once

# include <random>

#include "flightlib/common/types.hpp"

namespace flightlib {

enum class TrajectoryType {
  ELLIPSE,
  TRIANGLE,
  POLYNOMIAL,
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


  void update(Scalar sim_time, Scalar sim_dt);
  Vector<3> getPose() const;


 private:
  Vector<3> target_init_;
  Vector<3> target_xyY_;
  Vector<3> target_last_; 
  TrajectoryType traj_type_;
  Scalar max_speed_;
  Scalar last_time_;
  Scalar last_update_time_;
  Vector<3> random_dir_;

  std::random_device rd_;
  std::mt19937 rng_{rd_()};
  std::uniform_real_distribution<Scalar> uniform_dist_{-1.0, 1.0};
  
  



};


}  // namespace flightlib