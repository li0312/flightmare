/*** 
 * @Author: Flightmare
 * @Date: 2024-11-27 17:32:59 +0800
 * @LastEditTime: 2025-05-15 12:55:36 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /src/flightmare/flightlib/include/flightlib/common/command.hpp
 */

#pragma once

#include <cmath>

#include "flightlib/common/types.hpp"

namespace flightlib {

struct Command {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Command();

  Command(const Scalar t, const Scalar thrust, const Vector<3>& omega);

  Command(const Scalar t, const Vector<4>& thrusts);

  Command(const Scalar t, const Vector<3>& linear, const Vector<3>& angular);

  bool valid() const;
  bool isSingleRotorThrusts() const;
  bool isRatesThrust() const;
  bool isVelocity() const;

  /// Custom stream operator for outputs.
  friend std::ostream& operator<<(std::ostream& os, const Command& command);

  /// Print timing information to console.
  void print() const;

  /// time in [s]
  Scalar t{NAN};

  /// Collective mass-normalized thrust in [m/s^2]
  Scalar collective_thrust{NAN};

  /// Bodyrates in [rad/s]
  Vector<3> omega{NAN, NAN, NAN};

  /// Single rotor thrusts in [N]
  Vector<4> thrusts{NAN, NAN, NAN, NAN};

  /// velocity command linear [m/s] and anglar [rad/s]
  Vector<3> linear{NAN, NAN, NAN};
  Vector<3> angular{NAN, NAN, NAN};
};

}  // namespace flightlib