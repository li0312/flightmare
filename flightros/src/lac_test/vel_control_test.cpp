/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-05-15 13:06:03 +0800
 * @LastEditTime: 2025-06-25 16:06:19 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /flightmare/flightros/src/lac_test/vel_control_test.cpp
 */

// ros
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
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
#include "flightlib/sensors/rgb_camera.hpp"
#include "flightlib/sensors/lidar.hpp"
#include "flightlib/sensors/detect.hpp"

#include "flightlib/envs/track_env/track_env.hpp"

using namespace flightlib;

void quadstateToROSMsg(const QuadState& state, nav_msgs::Odometry& msg) {
  msg.pose.pose.position.x = state.x[QS::POSX];
  msg.pose.pose.position.y = state.x[QS::POSY];
  msg.pose.pose.position.z = state.x[QS::POSZ];
  msg.pose.pose.orientation.w = state.x[QS::ATTW];
  msg.pose.pose.orientation.x = state.x[QS::ATTX];
  msg.pose.pose.orientation.y = state.x[QS::ATTY];
  msg.pose.pose.orientation.z = state.x[QS::ATTZ];

  msg.header.frame_id = "world";
  msg.header.stamp = ros::Time::now();

  static tf2_ros::TransformBroadcaster br_world_laser;
  geometry_msgs::TransformStamped transformStamped;
  transformStamped.header.stamp = msg.header.stamp;
  transformStamped.header.frame_id = "world";
  transformStamped.child_frame_id = "laser";
  transformStamped.transform.translation.x = msg.pose.pose.position.x;
  transformStamped.transform.translation.y = msg.pose.pose.position.y;
  transformStamped.transform.translation.z = msg.pose.pose.position.z;
  transformStamped.transform.rotation.w = msg.pose.pose.orientation.w;
  transformStamped.transform.rotation.x = msg.pose.pose.orientation.x;
  transformStamped.transform.rotation.y = msg.pose.pose.orientation.y;
  transformStamped.transform.rotation.z = msg.pose.pose.orientation.z;
  br_world_laser.sendTransform(transformStamped);

  return;
}

int main(int argc, char *argv[]) {
  // initialize ROS
  ros::init(argc, argv, "camera_example");
  ros::NodeHandle nh("");
  ros::NodeHandle pnh("~");
  ros::Rate(50.0);

  // publisher
  ros::Publisher odom_pub;
  ros::Publisher cloud_pub;
  ros::Publisher map_pub;
  ros::Publisher scan_pub;


  // subscriber

  // unity quadrotor
  YAML::Node cfg_ =
    YAML::LoadFile(getenv("FLIGHTMARE_PATH") +
                   std::string("/flightlib/configs/quadrotor_env.yaml"));

  std::shared_ptr<Quadrotor> quad_ptr = std::make_shared<Quadrotor>();
  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quad_ptr->updateDynamics(dynamics);
  // define quadsize scale (for unity visualization only)
  // Vector<3> quad_size(0.5, 0.5, 0.5);
  // quad_ptr->setSize(quad_size);
  QuadState quad_state;

  //
  // std::shared_ptr<RGBCamera> rgb_camera = std::make_shared<RGBCamera>();

  // Flightmare(Unity3D)
  std::shared_ptr<UnityBridge> unity_bridge_ptr = UnityBridge::getInstance();
  SceneID scene_id{UnityScene::WAREHOUSE};
  bool unity_ready{false};

  // initialize publishers
  odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 1);
  cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("local_pointcloud", 1);
  map_pub = nh.advertise<sensor_msgs::PointCloud2>("global_map", 1);
  scan_pub = nh.advertise<sensor_msgs::LaserScan>("scan", 1);


  // Flightmare
  // Vector<3> B_r_BC(0.0, 0.0, 0.3);
  // Matrix<3, 3> R_BC = Quaternion(1.0, 0.0, 0.0, 0.0).toRotationMatrix();
  // std::cout << R_BC << std::endl;
  // rgb_camera->setFOV(90);
  // rgb_camera->setWidth(640);
  // rgb_camera->setHeight(360);
  // rgb_camera->setRelPose(B_r_BC, R_BC);
  // rgb_camera->setPostProcesscing(
  //   std::vector<bool>{true, true, true});  // depth, segmentation, optical
  //   flow
  // quad_ptr->addRGBCamera(rgb_camera);

  // initialization
  quad_state.setZero();

  quad_state.x(QS::POSX) = 5;
  quad_state.x(QS::POSY) = 5;
  quad_state.x(QS::POSZ) = 0.8;
  // quad_state.x(QS::ATTW) = std::cos(5.0/180*M_PI);
  // quad_state.x(QS::ATTX) = 0.0;
  // quad_state.x(QS::ATTY) = 0.0;
  // quad_state.x(QS::ATTZ) = std::sin(5.0 / 180 * M_PI);
  quad_ptr->reset(quad_state);

  QuadState state4test;
  nav_msgs::Odometry odom_msg;
  sensor_msgs::PointCloud2 local_cloud_msg;
  sensor_msgs::LaserScan scan_msg;


  // connect unity
  unity_bridge_ptr->addQuadrotor(quad_ptr);
  unity_ready = unity_bridge_ptr->connectUnity(scene_id);

  Lidar lidar_test;
  DetectSim detect_test;

  bool rviz_visual = true;

  if (rviz_visual) {

    ScanParams scan_params = lidar_test.getScanParams();
    scan_msg.angle_min = scan_params.angle_min;
    scan_msg.angle_max = scan_params.angle_max;
    scan_msg.angle_increment = scan_params.angle_increment;
    scan_msg.range_min = scan_params.range_min;
    scan_msg.range_max = scan_params.range_max;
    scan_msg.header.frame_id = "laser";

    const auto& global_map = lidar_test.getGlobalMap();
    sensor_msgs::PointCloud2 global_map_msg;
    pcl::toROSMsg(global_map, global_map_msg);
    global_map_msg.header.frame_id = "world";
    ROS_INFO("Map point size = %ld", global_map.points.size());
    for (int count = 0; count < 2; ++count) {
      map_pub.publish(global_map_msg);
      ros::Duration(0.2).sleep();
    }
    quad_ptr->getState(&state4test);
    quadstateToROSMsg(state4test, odom_msg);
    odom_pub.publish(odom_msg);
  }

  FrameID frame_id = 0;
  Command cmd;
  cmd.t = 0;
  
  cmd.linear.setZero();
  cmd.angular.setZero();

  Logger logger("vel_test");
  
  ros::Time time_now = ros::Time::now();
  ros::Time time_last = ros::Time::now();

  Timer timer("Detect", "Printing");
  Timer timer_1("Lidar", "Printing");
  std::random_device rd_;
  std::mt19937 random_gen_{rd_()};
  std::uniform_real_distribution<Scalar> uniform_dist_{-1.0, 1.0};
  int step = 0;
  int epoch = 100;


  while (ros::ok() && unity_ready) {
    // quad_state.x[QS::POSZ] += 0.1;

    // quad_ptr->setState(quad_state);
    // Control by velocity control
    step += 1;
    int mod = (step % epoch);
    if (mod == 0) {
      quad_state.setZero();
      Scalar init_x = uniform_dist_(random_gen_) * 5.0f;
      Scalar init_y = uniform_dist_(random_gen_) * 5.0f;
      quad_state.x(QS::POSX) = init_x;
      quad_state.x(QS::POSY) = init_y;
      quad_state.x(QS::POSZ) = 0.8f;
      Scalar yaw = uniform_dist_(random_gen_) * M_PI;
      quad_state.x(QS::ATTW) = std::cos(yaw / 2.0);
      quad_state.x(QS::ATTX) = 0.0f;
      quad_state.x(QS::ATTY) = 0.0f;
      quad_state.x(QS::ATTZ) = std::sin(yaw / 2.0);
      quad_state.qx /= quad_state.qx.norm();
      quad_ptr->reset(quad_state);

      cmd.t = 0;
      cmd.linear.setZero();
      cmd.angular.setZero();
    }
    
    time_last = time_now;
    time_now = ros::Time::now();
    Scalar dt = (time_now - time_last).toSec();
    dt = 0.05;
    cmd.t += dt;
    cmd.linear.x() = uniform_dist_(random_gen_);
    cmd.angular.z() = uniform_dist_(random_gen_);
    // if (cmd.t < 15) {
    //   cmd.linear.x() = 2.0;
    //   cmd.angular.z() = 0.5;
    // } else if (cmd.t < 25) {
    //   cmd.linear.x() = 0.0;
    //   cmd.linear.y() = 1.0;
    //   cmd.angular.z() = 0.0;
    // } else if (cmd.t < 45){
    //   cmd.linear.x() = -2.0;
    //   cmd.linear.y() = 0.0;
    //   cmd.angular.z() = 0.0;
    // } else{
    //   cmd.linear.setZero();
    //   cmd.angular.setZero();
    // }
    // cmd.linear.x() = 2.0;
    // cmd.angular.z() = 0.5;
    // timer.tic();


    quad_ptr->velocityControlBody(cmd, dt);
    // timer.toc();
    // std::cout << timer << std::endl;
    // DEBUG:

    quad_ptr->getState(&state4test);
    logger.info("time: %.2f", state4test.t);

    timer.tic();
    BBox result_bbox;
    Vector<3> target_xyY = {6.0, 0.0, 0.0};
    detect_test.updateTarget(target_xyY);
    bool is_valid = detect_test.getBBox(state4test, result_bbox);
    // std::cout << result_bbox << std::endl;
    // logger.debug("is_valid: %d", is_valid);
    // logger.debug("bbox: %d, %d, %d, %d", result_bbox.u_min, result_bbox.u_max,
    //              result_bbox.v_min, result_bbox.v_max);

    timer.toc();

    timer_1.tic();

    // lidar_test.renderPointCloud(state4test);
    lidar_test.renderLaserScan(state4test);

    if (rviz_visual) {
      quadstateToROSMsg(state4test, odom_msg);
      odom_pub.publish(odom_msg);
      const auto& local_cloud = lidar_test.getlocalCloud();
      pcl::toROSMsg(local_cloud, local_cloud_msg);
      local_cloud_msg.header.stamp = odom_msg.header.stamp;
      local_cloud_msg.header.frame_id = "laser";
      cloud_pub.publish(local_cloud_msg);

      const auto& scan_data = lidar_test.getScan();
      scan_msg.ranges = scan_data;
      scan_msg.header.stamp = odom_msg.header.stamp;
      scan_pub.publish(scan_msg);
    }
    
    std::cout << state4test << std::endl;
    Vector<2> test_vector = Vector<2>{3.0, 4.0};
    Vector<2> test_vector_norm = test_vector.normalized();
    std::cout << "test vector:" << test_vector << std::endl;
    std::cout << "test norm:" << test_vector_norm << std::endl;
    Vector<3> euler_xyz = state4test.euler_xyz();
    Scalar yaw = euler_xyz.z();
    Scalar body_vel_x =
      cos(yaw) * state4test.v.x() + sin(yaw) * state4test.v.y();
    Scalar body_vel_y =
      -sin(yaw) * state4test.v.x() + cos(yaw) * state4test.v.y();
    // Vector<3> pose(state4test.p);
    // Matrix<3, 3> rot(state4test.q());
    // std::cout << rot << std::endl;
    // std::cout << state4test.R() << std::endl;
    // std::cout << pose << std::endl;
    // std::cout << pose[0] << std::endl;
    // std::cout << pose(0) << std::endl;
    logger.debug("dt: %.2f", dt);
    logger.debug("pose: %.2f, %.2f, %.2f", state4test.p.x(), state4test.p.y(),
                 state4test.p.z());
    logger.debug("body_v: %.5f, %.5f", body_vel_x, body_vel_y);
    logger.debug("world_v: %.2f, %.2f, %.2f", state4test.v.x(), 
                 state4test.v.y(), state4test.w.z());
    // std::cout << "[VEL]" << state4test.x(QS::VELX) << state4test.x(QS::VELY)
    //           << state4test.x(QS::VELZ) << state4test.x(QS::OMEZ) << std::endl;

    timer_1.toc();
    std::cout << timer_1 << std::endl;

    unity_bridge_ptr->getRender(frame_id);
    unity_bridge_ptr->handleOutput();

    // std::cout << timer << std::endl;
    // std::cout << Vector<2>::Zero() << std::endl;

    // cv::Mat img;

    // ros::Time timestamp = ros::Time::now();
    // rgb_camera->getRGBImage(img);
    // sensor_msgs::ImagePtr rgb_msg =
    //   cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
    // rgb_msg->header.stamp = timestamp;
    // rgb_pub.publish(rgb_msg);

    // rgb_camera->getDepthMap(img);
    // sensor_msgs::ImagePtr depth_msg =
    //   cv_bridge::CvImage(std_msgs::Header(), "32FC1", img).toImageMsg();
    // depth_msg->header.stamp = timestamp;
    // depth_pub.publish(depth_msg);

    // rgb_camera->getSegmentation(img);
    // sensor_msgs::ImagePtr segmentation_msg =
    //   cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
    // segmentation_msg->header.stamp = timestamp;
    // segmentation_pub.publish(segmentation_msg);

    // rgb_camera->getOpticalFlow(img);
    // sensor_msgs::ImagePtr opticflow_msg =
    //   cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
    // opticflow_msg->header.stamp = timestamp;
    // opticalflow_pub.publish(opticflow_msg);

    frame_id += 1;
  }

  return 0;
}
