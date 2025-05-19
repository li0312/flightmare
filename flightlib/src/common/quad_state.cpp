/*** 
 * @Author: Flightmare
 * @Date: 2024-11-27 17:32:59 +0800
 * @LastEditTime: 2025-05-15 20:28:22 +0800
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
  // Vector<3> euler_xyz = q().toRotationMatrix().eulerAngles(0, 1, 2);
  // return Vector<3>{euler_xyz(0), euler_xyz(1), euler_xyz(2)};
  return q().toRotationMatrix().eulerAngles(0, 1, 2);
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