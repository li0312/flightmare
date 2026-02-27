/***
 * @Author: Lac_Creeper
 * @Date: 2026-02-07 01:18:49 +0800
 * @LastEditTime: 2026-02-07 01:23:40 +0800
 * @LastEditors: Lac_Creeper
 * @Description:
 * @FilePath: /flightmare/flightros/src/multi_track/lac_env_node.cpp
 */
#include <ros/ros.h>

#include "flightros/multi_track/lac_env.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "lac_env");
  if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                     ros::console::levels::Debug)) {
    ros::console::notifyLoggerLevelsChanged();
  }
  flightros::LacEnv env(ros::NodeHandle(), ros::NodeHandle("~"));

  // spin the ros
  ros::spin();

  return 0;
}