/*** 
 * @Author: Lac_Creeper
 * @Date: 2026-03-04 09:48:09 +0800
 * @LastEditTime: 2026-03-06 09:06:23 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /src/flightmare/flightros/include/flightros/multi_track/dataset_generator.hpp
 */
#pragma once

#include <memory>

// rpg quadrotor
#include <autopilot/autopilot_helper.h>
#include <autopilot/autopilot_states.h>
#include <quadrotor_common/parameter_helper.h>
#include <quadrotor_msgs/AutopilotFeedback.h>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/objects/quadrotor.hpp"
#include "flightlib/sensors/detect.hpp"
#include "flightlib/sensors/lidar.hpp"
#include "flightlib/sensors/lidar2D.hpp"
#include "flightlib/sensors/rgb_camera.hpp"

using namespace flightlib;

namespace flightros {

class DatasetGenerator {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  DatasetGenerator();
  ~DatasetGenerator();

 private:
  // Drones


  // Target
  Vector<3> target_init_;
  Vector<3> target_xyY_;
  




}   // namespace flightros