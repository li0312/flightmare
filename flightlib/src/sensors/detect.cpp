/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-05-20 21:38:23 +0800
 * @LastEditTime: 2025-06-25 19:26:06 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /flightmare/flightlib/src/sensors/detect.cpp
 */
#include "flightlib/sensors/detect.hpp"

namespace flightlib {

DetectSim::DetectSim()
  : width_(960),
    height_(540),
    fov_{69.0/180.0*M_PI} {

  Scalar focal_length = width_/(2*tan(fov_/2));
  K_ << focal_length, 0,            width_/2,
        0,            focal_length, height_/2,
        0,            0,            1;
  
  Vector<3> target_xyY = {5.0f, 0.0f, 0.0f};
  updateTarget(target_xyY);

  t_BC_ = {0.1f, 0.0f, -0.03f};
  R_BC_ << 0.0f, -1.0f, 0.0f, 
           0.0f, 0.0f, -1.0f,
           1.0f, 0.0f, 0.0f;

  logger_.debug("Image size: %d", (height_));
  logger_.debug("Init success..");
  
}

DetectSim::~DetectSim() {}

bool DetectSim::getBBox(const QuadState& state, BBox& out_bbox) {
  std::vector<Vector2i> projected_points;

  R_WB_ = state.R();
  t_WB_ = state.p;

  for (const auto& pose : target_cube_) {
    Vector<3> p_body = R_WB_.transpose() * (pose - t_WB_);
    Vector<3> p_camera = R_BC_ * (p_body - t_BC_);
    Scalar Z = p_camera.z();

    if (Z <= 0.0f) continue;

    Vector<3> pixel_homo = K_ * p_camera;
    int u = (int)(pixel_homo.x() / pixel_homo.z());
    int v = (int)(pixel_homo.y() / pixel_homo.z());

    projected_points.emplace_back(u, v);
    }

  if (projected_points.empty()) {
    out_bbox = {-1, -1, -1, -1};
    return false;
  }
  int u_min = width_;
  int u_max = 0;
  int v_min = height_;
  int v_max = 0;
  for (const auto& uv : projected_points) {
    u_min = std::min(u_min, uv.x());
    u_max = std::max(u_max, uv.x());
    v_min = std::min(v_min, uv.y());
    v_max = std::max(v_max, uv.y());
  }
  u_min = std::clamp(u_min, 0, width_);
  u_max = std::clamp(u_max, 0, width_);
  v_min = std::clamp(v_min, 0, height_);
  v_max = std::clamp(v_max, 0, height_);

  bool is_valid = (u_max > u_min) && (v_max > (v_min + 20));
  if (is_valid) {
    out_bbox = {u_min, v_min, u_max, v_max};
  } else {
    out_bbox = {-1, -1, -1, -1};
  }
  // logger_.debug("bbox: %d, %d, %d, %d", out_bbox.u_min, out_bbox.u_max, out_bbox.v_min, out_bbox.v_max);

  return is_valid;

}

void DetectSim::updateTarget(const Ref<Vector<3>> target_xyY) {
  target_xyY_ = target_xyY;
  Scalar heading = target_xyY_.z();
  Matrix<3, 3> rot;
  rot << cos(heading), -sin(heading), 0,
         sin(heading), cos(heading), 0,
         0,            0,            1;
  
  target_cube_.clear();
  target_cube_.emplace_back(
    rot * Vector<3>{target_xyY_.x() - 0.1f, target_xyY_.y() - 0.25f, 0.0f});
  target_cube_.emplace_back(
    rot * Vector<3>{target_xyY_.x() - 0.1f, target_xyY_.y() + 0.25f, 0.0f});
  target_cube_.emplace_back(
    rot * Vector<3>{target_xyY_.x() + 0.1f, target_xyY_.y() - 0.25f, 0.0f});
  target_cube_.emplace_back(
    rot * Vector<3>{target_xyY_.x() + 0.1f, target_xyY_.y() + 0.25f, 0.0f});
  target_cube_.emplace_back(
    rot * Vector<3>{target_xyY_.x() - 0.1f, target_xyY_.y() - 0.25f, 1.7f});
  target_cube_.emplace_back(
    rot * Vector<3>{target_xyY_.x() - 0.1f, target_xyY_.y() + 0.25f, 1.7f});
  target_cube_.emplace_back(
    rot * Vector<3>{target_xyY_.x() + 0.1f, target_xyY_.y() - 0.25f, 1.7f});
  target_cube_.emplace_back(
    rot * Vector<3>{target_xyY_.x() + 0.1f, target_xyY_.y() + 0.25f, 1.7f});
}


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
