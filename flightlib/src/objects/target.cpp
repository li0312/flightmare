/*** 
 * @Author: Lac_Creeper
 * @Date: 2026-03-04 10:08:14 +0800
 * @LastEditTime: 2026-03-10 15:16:04 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /src/flightmare/flightlib/src/objects/target.cpp
 */

#include "flightlib/objects/target.hpp"


namespace flightlib {

Target::Target() : Target(Vector<3>::Zero(), TrajectoryType::ELLIPSE, 1.0) {}

Target::Target(const Vector<3>& init_pose, const TrajectoryType& traj_type,
               const Scalar& max_speed)
  : target_init_(init_pose),
    target_xyY_(init_pose),
    target_last_(init_pose),
    traj_type_(traj_type),
    max_speed_(max_speed),
    last_time_(0.0f),
    last_update_time_(0.0f) {}

Target::~Target() {}

void Target::update(Scalar sim_time, Scalar sim_dt) {
  Vector<2> last_xy = target_xyY_.head<2>();

  switch (traj_type_) {
    case TrajectoryType::ELLIPSE: {
      Scalar a = 16.0, b = 8.0;
      Scalar omega = 0.0625 * max_speed_;
      target_xyY_.x() =
        target_init_.x() + 16.0 * cos(omega * sim_time);
      target_xyY_.y() =
        target_init_.y() + 8.0 * sin(omega * sim_time);
      Scalar dx_dt = -16.0 * omega * sin(omega * sim_time);
      Scalar dy_dt = 8.0 * omega * cos(omega * sim_time);
      target_xyY_.z() = atan2(dy_dt, dx_dt);
      break;
    }
    case TrajectoryType::TRIANGLE: {
      Scalar period = 12.0f / max_speed_;
      Scalar t = std::fmod(sim_time, period) / period;
      Vector<2> vertex[3] = {
        {0, 0}, {8, 6}, {8, -6} 
      };
      int seg = static_cast<int>(t * 3);
      Scalar seg_phase = t * 3 - seg;
      Vector<2> p =
        vertex[seg] + (vertex[(seg + 1) % 3] - vertex[seg]) * seg_phase;



    }
    case TrajectoryType::POLYNOMIAL: {
        Scalar T = 20.0f;
        Scalar t = std::fmod(sim_time, T) / T;
        target_xyY_.x() = target_init_.x() + 16.0 * t * t * (3 - 2 * t);
        target_xyY_.y() = target_init_.y() + 8.0 * t * t * (3 - 2 * t);
        Scalar dx_dt = 16.0 * (6 * t - 6 * t * t) / T;
        Scalar dy_dt = 8.0 * (6 * t - 6 * t * t) / T;
        target_xyY_.z() = atan2(dy_dt, dx_dt);
        break;


      
    }
    case TrajectoryType::FIGURE_EIGHT: {
      Scalar omega = 0.0441941679608 * max_speed_;
      target_xyY_.x() =
        target_init_.x() + 16.0 * sin(omega * sim_time);
      target_xyY_.y() =
        target_init_.y() + 8.0 * sin(2 * omega * sim_time);
      Scalar dx_dt = 16.0 * omega * cos(omega * sim_time);
      Scalar dy_dt = 16.0 * omega * cos(2.0 * omega * sim_time);
      target_xyY_.z() = atan2(dy_dt, dx_dt);
      break;
    }
  }



}
  









}