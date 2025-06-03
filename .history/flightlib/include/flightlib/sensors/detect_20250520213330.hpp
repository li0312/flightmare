/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-05-20 21:10:17 +0800
 * @LastEditTime: 2025-05-20 21:31:59 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /flightmare/flightlib/include/flightlib/sensors/detect.hpp
 */
#pragma once

#include "flightlib/common/logger.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/common/timer.hpp"
#include "flightlib/sensors/sensor_base.hpp"

namespace flightlib {

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
    Matrix<4, 4> getRelPose() const;
    int getWidth() const;
    int getHeight() const;
    Scalar getFOV() const;
    bool getDetect();

  private:
    Logger logger_{"DetectSim"};
    Timer timer1_{"Timer", "Printing"};





}



} // namespace flightlib
