/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-05-20 21:10:17 +0800
 * @LastEditTime: 2025-05-26 15:41:06 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /flightmare/flightlib/include/flightlib/sensors/detect.hpp
 */
#pragma once

#include <algorithm>

#include "flightlib/common/logger.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/timer.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/sensors/sensor_base.hpp"

namespace flightlib {

struct BBox {
  int u_min, v_min;
  int u_max, v_max;
  // Scalar dirt_vec;

  Vector<3> obs() const {
    Vector<3> norm_cp;
    norm_cp.x() = ((u_min + u_max)/2.0 - 480.0)/960.0;
    norm_cp.y() = ((v_min + v_max)/2.0 - 260.0)/540.0;
    norm_cp.z() = (v_max - v_min - 201.0)/540.0;

    return norm_cp;
  }

  Scalar IoU(const BBox& other) const {
    Scalar inter_u_min = std::max(u_min, other.u_min);
    Scalar inter_v_min = std::max(v_min, other.v_min);
    Scalar inter_u_max = std::min(u_max, other.u_max);
    Scalar inter_v_max = std::min(v_max, other.v_max);

    Scalar inter_area = std::max(0.0f, inter_u_max - inter_u_min) * 
                        std::max(0.0f, inter_v_max - inter_v_min);
    Scalar area_this = (u_max - u_min) * (v_max - v_min);
    Scalar area_other = (other.u_max - other.u_min) * 
                        (other.v_max - other.v_min);
    Scalar union_area = area_this + area_other - inter_area;

    return (union_area > 0) ? (inter_area / union_area) : 0.0f;
  }

  bool is_valid() const {
    return (u_max > u_min) && (v_max > v_min);
  }

  friend std::ostream& operator<<(std::ostream& os, const BBox& bbox) {
    Scalar cu = (bbox.u_min + bbox.u_max) / 2.0;
    Scalar cv = (bbox.v_min + bbox.v_max) / 2.0;
    Scalar ch = bbox.v_max - bbox.v_min;
    os.precision(3);
    os << "BBox" << "\n"
       << "[TL]: [" << bbox.u_min << " " << bbox.v_min << "]\n"
       << "[BR]: [" << bbox.u_max << " " << bbox.v_max << "]\n"
       << "[CUVH: [" << cu << " " << cv << " " << ch << "]\n"
       << "[nCUVH] :[" << bbox.obs().transpose() << "]\n";
    os.precision();
    return os;
  }
};

class DetectSim : SensorBase {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DetectSim();
    ~DetectSim();



    // public set functions
    void updateTarget(const Ref<Vector<3>> target_xyY);
    bool setRelPose(const Ref<Vector<3>> B_r_BC, const Ref<Matrix<3, 3>> R_BC);
    bool setWidth(const int width);
    bool setHeight(const int height);
    bool setFOV(const Scalar fov);

    // public get functions
    bool getBBox(const QuadState& state, BBox& out_bbox);
    Matrix<4, 4> getRelPose() const;
    int getWidth() const;
    int getHeight() const;
    Scalar getFOV() const;
    bool getDetect();

  private:
    Logger logger_{"DetectSim"};
    Timer timer1_{"Timer", "Timer"};

    // camera parameters
    int width_;
    int height_;
    Scalar fov_;

    // Camera intrinsic parameters
    Matrix<3, 3> K_;
    // Camera extrinsic parameters
    // Matrix<4, 4> T_CW_;
    // Matrix<4, 4> T_BW_;
    Matrix<3, 3> R_WB_;
    Vector<3> t_WB_;


    // camera relative
    Vector<3> t_BC_;
    Matrix<3, 3> R_BC_;
    Matrix<4, 4> T_BC_;

    Vector<3> target_xyY_;
    std::vector<Vector<3>> target_cube_;
    
};

} // namespace flightlib
