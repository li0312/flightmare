/*** 
 * @Author: Lac_Creeper
 * @Date: 2026-01-25 18:27:13 +0800
 * @LastEditTime: 2026-01-25 18:37:52 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /src/flightmare/flightros/src/lac_test/map_pub.cpp
 */
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include "flightlib/sensors/lidar.hpp"

using namespace flightlib;

int main(int argc, char* argv[]) {
  // initialize ROS
  ros::init(argc, argv, "map_pub");
  ros::NodeHandle nh("");
  ros::NodeHandle pnh("~");
  ros::Rate rate(50.0);
  Lidar lidar_;

  // publisher
  ros::Publisher map_pub_ =
    nh.advertise<sensor_msgs::PointCloud2>("global_map", 1);


  const auto& global_map = lidar_.getGlobalMap();
  sensor_msgs::PointCloud2 global_map_msg;
  pcl::toROSMsg(global_map, global_map_msg);
  global_map_msg.header.frame_id = "world";
  ROS_INFO("Map point size = %ld", global_map.size());
  for (int count = 0; count < 10; ++count) {
    map_pub_.publish(global_map_msg);
    ros::Duration(0.2).sleep();
  }
  // while (ros::ok())
  // {
  //   ros::spinOnce();
  //   rate.sleep();
  // }
  
  ROS_INFO("Map published");
  return 0;
}