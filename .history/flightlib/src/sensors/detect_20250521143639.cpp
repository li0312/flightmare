/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-05-20 21:38:23 +0800
 * @LastEditTime: 2025-05-21 14:28:31 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /flightmare/flightlib/src/sensors/detect.cpp
 */
#include "flightlib/sensors/detect.hpp"

namespace flightlib {

DetectSim::DetectSim()
  : width_(960),
    height_(540),
    fov_{69.0/180*M_PI} {

  Scalar focal_length = width_/(2*tan(fov_/2));
  K_ << focal_length, 0,            width_/2,
        0,            focal_length, height_/2,
        0,            0,            1;
  


}

DetectSim::~DetectSim() {}



bool DetectSim::setRelPose(const Ref<Vector<3>> t_BC,
                           const Ref<Matrix<3, 3>> R_BC) {
  if (!t_BC.allFinite() || !R_BC.allFinite()) {
    logger_.error(
      "The setting value for Camera Relative Pose Matrix is not valid, discard "
      "the setting.");
    return false;
  }
  t_BC_ = t_BC;
  R_BC_ = R_BC;
  T_BC_.block<3, 3>(0, 0) = R_BC;
  T_BC_.block<3, 1>(0, 3) = t_BC;
  T_BC_.row(3) << 0.0, 0.0, 0.0, 1.0;
  return true;
}

bool DetectSim::setWidth(const int width) {
  if (width <= 0.0) {
    logger_.warn(
      "The setting value for Image Width is not valid, discard the setting.");
    return false;
  }
  width_ = width;
  return true;
}

bool DetectSim::setHeight(const int height) {
  if (height <= 0.0) {
    logger_.warn(
      "The setting value for Image Height is not valid, discard the "
      "setting.");
    return false;
  }
  height_ = height;
  return true;
}

bool DetectSim::setFOV(const Scalar fov) {
  if (fov <= 0.0) {
    logger_.warn(
      "The setting value for Camera Field-of-View is not valid, discard the "
      "setting.");
    return false;
  }
  fov_ = fov;
  return true;
}

Matrix<4, 4> DetectSim::getRelPose(void) const { return T_BC_; }

int DetectSim::getWidth(void) const { return width_; }

int DetectSim::getHeight(void) const { return height_; }

Scalar DetectSim::getFOV(void) const { return fov_; }

} // namespace flightlib
