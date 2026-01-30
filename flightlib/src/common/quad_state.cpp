/*** 
 * @Author: Flightmare
 * @Date: 2024-11-27 17:32:59 +0800
 * @LastEditTime: 2025-09-25 22:25:51 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /src/flightmare/flightlib/src/common/quad_state.cpp
 */
#include "flightlib/common/quad_state.hpp"

namespace flightlib {

QuadState::QuadState() {}

QuadState::QuadState(const Vector<IDX::SIZE>& x, const Scalar t) : x(x), t(t) {}

QuadState::QuadState(const QuadState& state) : x(state.x), t(state.t) {}

QuadState::~QuadState() {}

Quaternion QuadState::q() const {
  return Quaternion(x(ATTW), x(ATTX), x(ATTY), x(ATTZ));
}

Vector<3> QuadState::euler_xyz() const {
  // return q().toRotationMatrix().eulerAngles(0, 1, 2);
  Quaternion quat(x(ATTW), x(ATTX), x(ATTY), x(ATTZ));
  Vector<3> euler;
  euler.x() = std::atan2(2 * quat.w() * quat.x() + 2 * quat.y() * quat.z(),
                         quat.w() * quat.w() - quat.x() * quat.x() -
                           quat.y() * quat.y() + quat.z() * quat.z());
  euler.y() = -std::asin(2 * quat.x() * quat.z() - 2 * quat.w() * quat.y());
  euler.z() = std::atan2(2 * quat.w() * quat.z() + 2 * quat.x() * quat.y(),
                         quat.w() * quat.w() + quat.x() * quat.x() -
                           quat.y() * quat.y() - quat.z() * quat.z());
  return euler;
}

void QuadState::q(const Quaternion quaternion) {
  x(IDX::ATTW) = quaternion.w();
  x(IDX::ATTX) = quaternion.x();
  x(IDX::ATTY) = quaternion.y();
  x(IDX::ATTZ) = quaternion.z();
}

Matrix<3, 3> QuadState::R() const {
  return Quaternion(x(ATTW), x(ATTX), x(ATTY), x(ATTZ)).toRotationMatrix();
}

void QuadState::setZero() {
  t = 0.0;
  x.setZero();
  x(ATTW) = 1.0;
}

std::ostream& operator<<(std::ostream& os, const QuadState& state) {
  os.precision(3);
  os << "State at " << state.t << "s: \n"
     << "[POS]: [" << state.p.transpose() << "]\n"
     << "[ATT]: [" << state.euler_xyz().transpose() << "]\n"
     << "[VEL]: [" << state.v.transpose() << "]\n"
     << "[OMG]: [" << state.w.transpose() << "]\n"
     << "[ACC]: [" << state.a.transpose() << "]\n"
     << "[TAU]: [" << state.tau.transpose() << "]\n";
  os.precision();
  return os;
}

}  // namespace flightlib