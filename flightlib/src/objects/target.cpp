/*** 
 * @Author: Lac_Creeper
 * @Date: 2026-03-04 10:08:14 +0800
 * @LastEditTime: 2026-04-10 05:58:13 +0800
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
    max_speed_(max_speed) {
  reset_flag_ = true;
}

Target::~Target() {}





void Target::update(Scalar sim_time) {
  Vector<2> last_xy = target_xyY_.head<2>();

  switch (traj_type_) {
    case TrajectoryType::ELLIPSE: {
      Scalar a = 16.0, b = 8.0;
      Scalar omega = 0.0625 * max_speed_;
      target_xyY_.x() = target_init_.x() + a * sin(omega * sim_time);
      target_xyY_.y() = target_init_.y() - b * cos(omega * sim_time) + b;
      Scalar dx_dt = a * omega * cos(omega * sim_time);
      Scalar dy_dt = b * omega * sin(omega * sim_time);
      target_xyY_.z() = atan2(dy_dt, dx_dt);
      break;
    }
    case TrajectoryType::TRIANGLE: {
      const Scalar R = 0.5;
      const Scalar theta = 2 * M_PI / 3.0;
      Vector<2> vertex[3] = {{-9, 0}, {9, 0}, {0, 9 * sqrt(3)}};
      
      Vector<2> edge[3];
      Scalar edge_len[3];
      for (int i = 0; i < 3; ++i) {
        edge[i] = vertex[(i + 1) % 3] - vertex[i];
        edge_len[i] = edge[i].norm();
        edge[i] /= edge_len[i];
      }
      const Scalar tangent_dist = R * tan(theta / 2.0);
      Scalar line_len = edge_len[0] - 2 * tangent_dist;

      Scalar total_len = 3 * line_len + 3 * R * theta;
      Scalar seg = std::fmod(sim_time * max_speed_ + line_len / 2, total_len);
      Vector<2> pos;
      Vector<2> vel;
      for (int i = 0; i < 3; ++i) {
        if (seg < line_len) {
          pos = vertex[i] + edge[i] * (tangent_dist + seg);
          vel = edge[i] * max_speed_;
          break;
        } else if (seg < line_len + R * theta) {
          Scalar angle = (seg - line_len) / R;
          pos = vertex[(i + 1) % 3] - edge[i] * tangent_dist +
                Vector<2>{-edge[i].y(), edge[i].x()} * R * (1 - cos(angle)) +
                edge[i] * R * sin(angle);
          vel = Vector<2>{-edge[i].y(), edge[i].x()} * max_speed_ * sin(angle) +
                edge[i] * max_speed_ * cos(angle);
          break;
        } else {
          seg -= line_len + R * theta;
        }
      }
      target_xyY_.x() = target_init_.x() + pos.x();
      target_xyY_.y() = target_init_.y() + pos.y();
      target_xyY_.z() = atan2(vel.y(), vel.x());
      break;
    }
    case TrajectoryType::FIGURE_EIGHT: {
      Scalar omega = 0.0441941679608 * max_speed_;
      // Scalar omega = 0.176776828093 * max_speed_;
      target_xyY_.x() =
        target_init_.x() + 16.0 * sin(omega * sim_time);
      target_xyY_.y() =
        target_init_.y() + 8.0 * sin(2 * omega * sim_time);
      Scalar dx_dt = 16.0 * omega * cos(omega * sim_time);
      Scalar dy_dt = 16.0 * omega * cos(2.0 * omega * sim_time);
      target_xyY_.z() = atan2(dy_dt, dx_dt);
      break;
    }
    case TrajectoryType::RANDOM_WALK: {
      const Scalar R = 1.0;
      const Scalar area_x = 15.0, area_y = 15.0;
      const Scalar border_margin = 2.0;
      const Scalar arc_turn_angle = M_PI / 3.0;
      const Scalar cooldown_time = 3.0;

      static Scalar walk_theta = 0.0;
      static Scalar last_turn_t = 0.0;
      static Scalar arc_progress = 0.0;
      static Scalar turn_angle = 0.0;
      static int turn_dir = 0; // -1 for left, 1 for right
      static Scalar next_allow_turn = 0.0;

      static Scalar prev_time = sim_time;

      if (reset_flag_) {
        walk_theta = 0.0;
        last_turn_t = 0.0;
        arc_progress = 0.0;
        turn_angle = 0.0;
        turn_dir = 0;  // -1 for left, 1 for right
        next_allow_turn = 0.0;
        prev_time = sim_time;

        reset_flag_ = false;
      }

      Scalar dt = sim_time - prev_time;
      if (dt < 1e-6) dt = 1e-3;
      prev_time = sim_time;

      Vector<2> pos = last_xy;
      Scalar min_dist_to_border =
        std::min({area_x - pos.x(), pos.x() + area_x, area_y - pos.y(),
                  pos.y() + area_y});
      bool force_turn = min_dist_to_border < border_margin;

      if (arc_progress < 1e-4) {
        if (force_turn) {
          Scalar max_dev = M_PI / 5;
          Scalar dev = uniform_dist_(rng_) * max_dev;
          Scalar new_angle = atan2(-last_xy.y(), -last_xy.x()) + dev;
          turn_angle = new_angle - walk_theta;
          arc_progress = 1e-4;
          last_turn_t = sim_time;
          next_allow_turn = sim_time + cooldown_time;
        } else {
          Scalar t_since_last = sim_time - last_turn_t;
          Scalar p_turn = 1 - exp(-t_since_last / 5.0);
          if (sim_time > next_allow_turn) {
            if (uniform_01_dist_(rng_) < p_turn) {
              turn_angle = uniform_dist_(rng_) * arc_turn_angle;
              arc_progress = 1e-4;
              last_turn_t = sim_time;
              next_allow_turn = sim_time + cooldown_time;
            } else {
              next_allow_turn += uniform_01_dist_(rng_) * 1.0;
            }
          } 
        }
      }
      if (arc_progress > 1e-5) {
        Scalar center_dir = walk_theta + turn_angle * arc_progress +
                            (turn_angle > 0 ? M_PI_2 : -M_PI_2);
        Vector<2> center =
          pos + Vector<2>{R * cos(center_dir), R * sin(center_dir)};
        Scalar arc_total = abs(turn_angle) * R;
        Scalar ds = max_speed_ * dt;
        arc_progress += ds / arc_total;
        Scalar heading = walk_theta + turn_angle * arc_progress;
        if (arc_progress > 1.0) {
          arc_progress = 0.0;
          walk_theta += turn_angle;
          heading = walk_theta;
        }
        Scalar angle1 = heading + (turn_angle > 0 ? -M_PI_2 : M_PI_2);
        pos = center + Vector<2>{R * cos(angle1), R * sin(angle1)};
        target_xyY_.z() = heading;
      } else {
        pos += Vector<2>{max_speed_ * dt * cos(walk_theta),
                         max_speed_ * dt * sin(walk_theta)};
        target_xyY_.z() = walk_theta;
      }
      target_xyY_.x() = pos.x();
      target_xyY_.y() = pos.y();
      
      break;
    }

  }
  // static Scalar last_time = sim_time;
  // static Scalar v_max = 0.0;
  // if (sim_time - last_time > 0.01) {
  //   Vector<2> vel = (target_xyY_.head<2>() - last_xy) / (sim_time - last_time);
  //   Scalar speed = vel.norm();
  //   v_max = std::max(v_max, speed);
  //   last_time = sim_time;
  //   logger_.debug("v: %.2f, vMax: %.2f, t: %.2f", speed, v_max, sim_time);
  // }
  
}
  









}