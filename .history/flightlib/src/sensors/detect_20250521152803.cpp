/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-05-20 21:38:23 +0800
 * @LastEditTime: 2025-05-21 15:27:12 +0800
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

BBox DetectSim::getBBox(const std::vector<Vector<3>>& world_poses) {
  std::vector<Vector2i> projected_points;

  for (const auto& pose : world_poses) {
    Vector<4> pose_world_homo(pose.x(), pose.y(), pose.z(), 1.0f);
    Vector<4> pose_camera_homo = T_BW_ * pose_world_homo;
    Scalar Z = 
  }


}

bool DetectSim::updatePose(const QuadState& state) {
  T_BW_.block<3, 3>(0, 0) = state.R();
  T_BW_.block<3, 1>(0, 3) = state.p;

  T_CW_ = T_CB_ * T_BW_;


}

bool DetectSim::setRelPose(const Ref<Vector<3>> t_CB,
                           const Ref<Matrix<3, 3>> R_CB) {
  if (!t_CB.allFinite() || !R_CB.allFinite()) {
    logger_.error(
      "The setting value for Camera Relative Pose Matrix is not valid, discard "
      "the setting.");
    return false;
  }
  t_CB_ = t_CB;
  R_CB_ = R_CB;
  T_CB_.block<3, 3>(0, 0) = R_CB;
  T_CB_.block<3, 1>(0, 3) = t_CB;
  T_CB_.row(3) << 0.0, 0.0, 0.0, 1.0;
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

Matrix<4, 4> DetectSim::getRelPose(void) const { return T_CB_; }

int DetectSim::getWidth(void) const { return width_; }

int DetectSim::getHeight(void) const { return height_; }

Scalar DetectSim::getFOV(void) const { return fov_; }

} // namespace flightlib
