/*** 
 * @Author: Lac_Creeper
 * @Date: 2026-01-03 16:36:05 +0800
 * @LastEditTime: 2026-01-16 23:42:31 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /src/flightmare/flightros/src/fast_wrapper/fast_wrapper_node.cpp
 */
#include <ros/ros.h>

#include "flightros/fast_wrapper/fast_wrapper.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "fast_wrapper");
  if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                     ros::console::levels::Debug)) {
    ros::console::notifyLoggerLevelsChanged();
  }
  flightros::FastWrapper pilot(ros::NodeHandle(), ros::NodeHandle("~"));

  // spin the ros
  // ros::spin();
  ros::Rate rate(100);
  bool status = ros::ok();
  while (status) {
    ros::spinOnce();
    status = ros::ok();
    rate.sleep();
  }

  return 0;
}