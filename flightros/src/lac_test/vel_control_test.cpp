/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-05-15 13:06:03 +0800
 * @LastEditTime: 2025-05-18 22:08:54 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /src/flightmare/flightros/src/lac_test/vel_control_test.cpp
 */

// ros
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/bridges/unity_message_types.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/timer.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/objects/quadrotor.hpp"
#include "flightlib/sensors/rgb_camera.hpp"
#include "flightlib/sensors/lidar.hpp"

using namespace flightlib;

int main(int argc, char *argv[]) {
  // initialize ROS
  ros::init(argc, argv, "camera_example");
  ros::NodeHandle nh("");
  ros::NodeHandle pnh("~");
  ros::Rate(50.0);

  // publisher
  // image_transport::Publisher rgb_pub;
  // image_transport::Publisher depth_pub;
  // image_transport::Publisher segmentation_pub;
  // image_transport::Publisher opticalflow_pub;

  // sub

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
  // image_transport::ImageTransport it(pnh);
  // rgb_pub = it.advertise("/rgb", 1);
  // depth_pub = it.advertise("/depth", 1);
  // segmentation_pub = it.advertise("/segmentation", 1);
  // opticalflow_pub = it.advertise("/opticalflow", 1);

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
  quad_state.x(QS::POSZ) = 0.8;
  quad_ptr->reset(quad_state);

  // connect unity
  unity_bridge_ptr->addQuadrotor(quad_ptr);
  unity_ready = unity_bridge_ptr->connectUnity(scene_id);

  Lidar lidar_test;
  lidar_test.getViewer()->spinOnce(100);

  FrameID frame_id = 0;
  Command cmd;
  cmd.t = 0;
  
  cmd.linear.setZero();
  cmd.angular.setZero();

  Logger logger("vel_test");
  
  ros::Time time_now = ros::Time::now();
  ros::Time time_last = ros::Time::now();
  


  while (ros::ok() && unity_ready) {
    // quad_state.x[QS::POSZ] += 0.1;

    // quad_ptr->setState(quad_state);
    // Control by velocity control
    
    time_last = time_now;
    time_now = ros::Time::now();
    Scalar dt = (time_now - time_last).toSec();
    // dt = 0.01;
    cmd.t += dt;
    if (cmd.t < 15) {
      cmd.linear.x() = 2.0;
      cmd.angular.z() = 0.5;
    } else if (cmd.t < 25) {
      cmd.linear.x() = 0.0;
      cmd.linear.y() = 1.0;
      cmd.angular.z() = 0.0;
    } else if (cmd.t < 45){
      cmd.linear.x() = -2.0;
      cmd.linear.y() = 0.0;
      cmd.angular.z() = 0.0;
    } else{
      cmd.linear.setZero();
      cmd.angular.setZero();
    }
    // cmd.linear.x() = 2.0;
    // cmd.angular.z() = 0.5;
    // timer.tic();

    Timer timer("Timer", "Printing");
    Timer timer_1("Timer_1", "Printing");
    timer.tic();
    timer_1.tic();

    quad_ptr->velocityControl(cmd, dt);
    // timer.toc();
    // std::cout << timer << std::endl;
    // DEBUG:

    QuadState state4test;
    quad_ptr->getState(&state4test);
    lidar_test.renderPointCloud(state4test);
    lidar_test.getViewer()->spinOnce(20);

    std::cout << state4test << std::endl;
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
    // std::cout << "[VEL]" << state4test.x(QS::VELX) << state4test.x(QS::VELY)
    //           << state4test.x(QS::VELZ) << state4test.x(QS::OMEZ) << std::endl;

    timer_1.toc();
    std::cout << timer_1 << std::endl;

    unity_bridge_ptr->getRender(frame_id);
    unity_bridge_ptr->handleOutput();
    timer.toc();

    std::cout << timer << std::endl;

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
