/*** 
 * @Author: Flightmare
 * @Date: 2024-11-27 17:32:59 +0800
 * @LastEditTime: 2025-05-15 14:19:07 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /src/flightmare/flightlib/src/common/command.cpp
 */
#include "flightlib/common/command.hpp"


namespace flightlib {

Command::Command() {}

Command::Command(const Scalar t, const Scalar thrust, const Vector<3>& omega)
  : t(t), collective_thrust(thrust), omega(omega) {}

Command::Command(const Scalar t, const Vector<4>& thrusts)
  : t(t), thrusts(thrusts) {}

Command::Command(const Scalar t, const Vector<3>& linear, const Vector<3>& angular)
  : t(t), linear(linear), angular(angular) {}

bool Command::valid() const {
  int command_num = ((std::isfinite(collective_thrust) && omega.allFinite()) +
                     thrusts.allFinite()) +
                    (linear.allFinite() && angular.allFinite());
  return std::isfinite(t) && (command_num == 1);
}

bool Command::isSingleRotorThrusts() const {
  return std::isfinite(t) && thrusts.allFinite();
}

bool Command::isRatesThrust() const {
  return std::isfinite(t) && std::isfinite(collective_thrust) &&
         omega.allFinite();
}

bool Command::isVelocity() const {
  return std::isfinite(t) && linear.allFinite() && angular.allFinite();
}

std::ostream& operator<<(std::ostream& os, const Command& command) {
  if (!command.valid()) {
    os << "Command has no set yet." << std::endl;
    return os;
  }

  const std::streamsize prec = os.precision();
  os.precision(3);

  os << "Command in " << command.t << "s" << std::endl;

  if (command.isRatesThrust()) {
    os << "[collective_thrust]: " << command.collective_thrust << "\n"
       << "[omega]: [" << command.omega.x() << " " << command.omega.y() << " "
       << command.omega.z() << "]" << std::endl;
  } else if (command.isSingleRotorThrusts()) {
    os << "[thrusts]: [" << command.thrusts.x() << " " << command.thrusts.y()
       << " " << command.thrusts.z() << " " << command.thrusts.w() << "]"
       << std::endl;
  } else if (command.isVelocity()) {
    os << "[linear]: [" << command.linear.x() << " " << command.linear.y()
       << " " << command.linear.z() << "]\n"
       << "[angular]: [" << command.angular.x() << " " << command.angular.y()
       << " " << command.angular.z() << "]\n";
  }

  os.precision(prec);
  return os;
}

}  // namespace flightlib