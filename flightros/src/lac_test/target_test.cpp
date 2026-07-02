/*** 
 * @Author: Lac_Creeper
 * @Date: 2026-03-12 17:07:57 +0800
 * @LastEditTime: 2026-05-05 21:00:28 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /src/flightmare/flightros/src/lac_test/target_test.cpp
 */
// ros
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/bridges/unity_message_types.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/timer.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/objects/quadrotor.hpp"
#include "flightlib/sensors/detect.hpp"
#include "flightlib/sensors/lidar2D.hpp"
#include "flightlib/sensors/rgb_camera.hpp"
#include "flightlib/objects/target.hpp"


using namespace flightlib;


int main(int argc, char** argv) {
  ros::init(argc, argv, "target_test");
  ros::NodeHandle nh;

  ros::Publisher target_pub = nh.advertise<nav_msgs::Odometry>("/target", 1);
  ros::Publisher map_pub =
    nh.advertise<visualization_msgs::MarkerArray>("/obstacles", 1);


  Lidar2D lidar_{-18, 18, -15, 15, 2 * M_PI, 512, 8.0, 0.25};
  lidar_.generateRandomMap(20, 20);
  // lidar_.loadMap("/home/lac/fm_test/my_logs/exp1/map2/random_map.log");
  std::vector<std::shared_ptr<Obstacle>> obstacles;
  obstacles = lidar_.getObstacles();
  visualization_msgs::MarkerArray markers;
  int id = 0;
  for (const auto& obs : obstacles) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.ns = "obstacles";
    marker.id = id++;
    marker.action = visualization_msgs::Marker::ADD;

    if (auto rect = dynamic_cast<Rectangle*>(obs.get())) {
      marker.type = visualization_msgs::Marker::CUBE;
      marker.pose.position.x = rect->center().x();
      marker.pose.position.y = rect->center().y();
      marker.pose.position.z = 1.0;

      tf::Quaternion q;
      q.setRPY(0, 0, rect->angle());
      marker.pose.orientation.x = q.x();
      marker.pose.orientation.y = q.y();
      marker.pose.orientation.z = q.z();
      marker.pose.orientation.w = q.w();

      marker.scale.x = rect->width();
      marker.scale.y = rect->height();
      marker.scale.z = 2.0;  // 小高度

      marker.color.r = 0.5;
      marker.color.g = 0.5;
      marker.color.b = 0.5;
      marker.color.a = 1.0;
    } else if (auto ellipse = dynamic_cast<Ellipse*>(obs.get())) {
      marker.type = visualization_msgs::Marker::CYLINDER;
      marker.pose.position.x = ellipse->center().x();
      marker.pose.position.y = ellipse->center().y();
      marker.pose.position.z = 1.0;

      tf::Quaternion q;
      q.setRPY(0, 0, ellipse->angle());
      marker.pose.orientation.x = q.x();
      marker.pose.orientation.y = q.y();
      marker.pose.orientation.z = q.z();
      marker.pose.orientation.w = q.w();

      marker.scale.x = ellipse->a() * 2;  // 直径
      marker.scale.y = ellipse->b() * 2;
      marker.scale.z = 2.0;  // 小高度

      marker.color.r = 0.5;
      marker.color.g = 0.5;
      marker.color.b = 0.5;
      marker.color.a = 1.0;
    }

    marker.lifetime = ros::Duration(0);
    markers.markers.push_back(marker);
  }
  for (int i = 0; i < 20; ++i) {
    map_pub.publish(markers);
    ros::Rate loop_rate(10);
    loop_rate.sleep();
  }

  Target target{Vector<3>{0.0f, 0.0f, 0.0f}, TrajectoryType::FIGURE_EIGHT, 1.0f};
  // target.setTrajectoryType(TrajectoryType::ELLIPSE);
  // target.setMaxSpeed(1.0f);

  ros::Rate loop_rate(20);
  double start_time = ros::Time::now().toSec();
  while (ros::ok()) {
    double curr_time = ros::Time::now().toSec();
    Scalar sim_time = curr_time - start_time;

    // if (sim_time > 30.0) {
    //   target.reset(Vector<3>{0.0f, 0.0f, 0.0f},
    //   TrajectoryType::RANDOM_WALK, 1.0f); start_time = curr_time; sim_time
    //   = curr_time - start_time;
    // }
    target.update(sim_time);
    Vector<3> target_pose = target.getPose();
    std::cout << "[Target Pose]: " << target_pose.transpose()
              << " | Sim Time: " << sim_time << std::endl;

    nav_msgs::Odometry target_msg;
    target_msg.header.stamp = ros::Time::now();
    target_msg.header.frame_id = "world";
    target_msg.pose.pose.position.x = target_pose.x();
    target_msg.pose.pose.position.y = target_pose.y();
    target_msg.pose.pose.position.z = 0.0;
    target_msg.pose.pose.orientation.w = std::cos(target_pose.z() / 2.0);
    target_msg.pose.pose.orientation.x = 0.0;
    target_msg.pose.pose.orientation.y = 0.0;
    target_msg.pose.pose.orientation.z = std::sin(target_pose.z() / 2.0);
    target_pub.publish(target_msg);

    loop_rate.sleep();
  }

  return 0;
}