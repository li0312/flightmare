/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-05-20 21:10:17 +0800
 * @LastEditTime: 2025-05-21 15:48:59 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /flightmare/flightlib/include/flightlib/sensors/detect.hpp
 */
#pragma once

#include "flightlib/common/logger.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/common/timer.hpp"
#include "flightlib/sensors/sensor_base.hpp"
#include "flightlib/common/quad_state.hpp"

namespace flightlib {

struct BBox {
  int u_min, v_min;
  int u_max, v_max;
  bool is_vaild;
};

class DetectSim : SensorBase {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DetectSim();
    ~DetectSim();



    // public set functions
    bool setRelPose(const Ref<Vector<3>> B_r_BC, const Ref<Matrix<3, 3>> R_BC);
    bool setWidth(const int width);
    bool setHeight(const int height);
    bool setFOV(const Scalar fov);

    // public get functions
    bool getBBox(const std::vector<Vector<3>>& world_poses,
                 const QuadState& state, BBox& out_bbox);
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
    Matrix<4, 4> T_CW_;
    Matrix<4, 4> T_BW_;


    // camera relative
    Vector<3> t_CB_;
    Matrix<3, 3> R_CB_;
    Matrix<4, 4> T_CB_;

    

}



} // namespace flightlib
