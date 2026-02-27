/***
 * @Author: Lac_Creeper
 * @Date: 2026-02-07 01:24:15 +0800
 * @LastEditTime: 2026-02-07 01:24:16 +0800
 * @LastEditors: Lac_Creeper
 * @Description:
 * @FilePath: /flightmare/flightros/src/multi_track/lac_drone_node.cpp
 */
#include <ros/ros.h>

#include "flightros/multi_track/lac_drone.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "lac_drone");
  if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                     ros::console::levels::Debug)) {
    ros::console::notifyLoggerLevelsChanged();
  }
  flightros::LacDrone drone(ros::NodeHandle(), ros::NodeHandle("~"));

  // spin the ros
  ros::spin();

  return 0;
}