/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-05-20 21:10:17 +0800
 * @LastEditTime: 2025-05-21 20:24:29 +0800
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
