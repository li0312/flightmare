#include "flightros/fast_wrapper/fast_wrapper.hpp"

namespace flightros {

FastWrapper::FastWrapper(const ros::NodeHandle& nh, const ros::NodeHandle& pnh)
  : nh_(nh),
    pnh_(pnh),
    scene_id_(UnityScene::INDUSTRIAL),
    unity_ready_(false),
    unity_render_(false),
    receive_id_(0),
    vel_param_(1.0),
    main_loop_freq_(50.0) {
  // load parameters
  if (!loadParams()) {
    ROS_WARN("[%s] Could not load all parameters.",
             pnh_.getNamespace().c_str());
  } else {
    ROS_INFO("[%s] Loaded all parameters.", pnh_.getNamespace().c_str());
  }

  // quad initialization
  quad_ptr_ = std::make_shared<Quadrotor>();

  // add mono camera
  // rgb_camera_ = std::make_shared<RGBCamera>();
  // Vector<3> B_r_BC(0.0, 0.0, 0.3);
  // Matrix<3, 3> R_BC = Quaternion(1.0, 0.0, 0.0, 0.0).toRotationMatrix();
  // std::cout << R_BC << std::endl;
  // rgb_camera_->setFOV(90);
  // rgb_camera_->setWidth(720);
  // rgb_camera_->setHeight(480);
  // rgb_camera_->setRelPose(B_r_BC, R_BC);
  // quad_ptr_->addRGBCamera(rgb_camera_);

  // initialization
  quad_state_.setZero();
  quad_ptr_->reset(quad_state_);
  target_init_.setZero();
  target_init_(0) = 3.0;
  target_xyY_ = target_init_;


  // initialize publisher
  map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("global_map", 1);
  cloud_pub_ =
    nh_.advertise<sensor_msgs::PointCloud2>("local_cloud", 1);
  scan_pub_ = nh_.advertise<sensor_msgs::LaserScan>("scan", 1);
  targetGT_pub_ = nh_.advertise<nav_msgs::Odometry>("targetGT", 1);
  targetDetect_pub_ = nh_.advertise<nav_msgs::Odometry>("car_state", 1);
  debug_pub_ =
    nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("debug", 1);


  // initialize subscriber call backs
  sub_state_est_ =
    nh_.subscribe("/visual_slam/odom", 1, &FastWrapper::poseCallback, this);
  sub_start_ = 
    nh_.subscribe("/move_base_simple/goal", 1, &FastWrapper::startCallback, this);

  
  // wait until the gazebo and unity are loaded
  ros::Duration(5.0).sleep();
  
  // Publish global map
  ScanParams scan_params = lidar_.getScanParams();
  scan_msg_.angle_min = scan_params.angle_min;
  scan_msg_.angle_max = scan_params.angle_max;
  scan_msg_.angle_increment = scan_params.angle_increment;
  scan_msg_.range_min = scan_params.range_min;
  scan_msg_.range_max = scan_params.range_max;
  scan_msg_.header.frame_id = "laser";

  const auto& global_map = lidar_.getGlobalMap();
  sensor_msgs::PointCloud2 global_map_msg;
  pcl::toROSMsg(global_map, global_map_msg);
  global_map_msg.header.frame_id = "world";
  ROS_INFO("Map point size = %ld", global_map.size());
  for (int count = 0; count < 2; ++count) {
    map_pub_.publish(global_map_msg);
    ros::Duration(0.2).sleep();
  }

  timer_main_loop_ = nh_.createTimer(ros::Rate(main_loop_freq_),
                                     &FastWrapper::mainLoopCallback, this);
  ROS_INFO("Fast wrapper init success..");

  // connect unity
  setUnity(unity_render_);
  connectUnity();
}

FastWrapper::~FastWrapper() {}

void FastWrapper::poseCallback(const nav_msgs::Odometry::ConstPtr& msg) {
  if (!has_odom_) has_odom_ = true;
  timer_.tic();
  quad_state_.x[QS::POSX] = (Scalar)msg->pose.pose.position.x;
  quad_state_.x[QS::POSY] = (Scalar)msg->pose.pose.position.y;
  quad_state_.x[QS::POSZ] = (Scalar)msg->pose.pose.position.z;
  quad_state_.x[QS::ATTW] = (Scalar)msg->pose.pose.orientation.w;
  quad_state_.x[QS::ATTX] = (Scalar)msg->pose.pose.orientation.x;
  quad_state_.x[QS::ATTY] = (Scalar)msg->pose.pose.orientation.y;
  quad_state_.x[QS::ATTZ] = (Scalar)msg->pose.pose.orientation.z;
  //
  quad_ptr_->setState(quad_state_);

  static tf2_ros::TransformBroadcaster br_world_laser;
  geometry_msgs::TransformStamped transformStamped;
  transformStamped.header.stamp = msg->header.stamp;
  transformStamped.header.stamp = ros::Time::now();
  transformStamped.header.frame_id = "world";
  transformStamped.child_frame_id = "laser";
  transformStamped.transform.translation.x = msg->pose.pose.position.x;
  transformStamped.transform.translation.y = msg->pose.pose.position.y;
  transformStamped.transform.translation.z = msg->pose.pose.position.z;
  transformStamped.transform.rotation.w = msg->pose.pose.orientation.w;
  transformStamped.transform.rotation.x = msg->pose.pose.orientation.x;
  transformStamped.transform.rotation.y = msg->pose.pose.orientation.y;
  transformStamped.transform.rotation.z = msg->pose.pose.orientation.z;
  br_world_laser.sendTransform(transformStamped);
  transform_ = transformStamped;


  timer_.toc();
  // logger_.debug("timer: %.3f.", timer_.mean());


  if (unity_render_ && unity_ready_) {
    unity_bridge_ptr_->getRender(0);
    unity_bridge_ptr_->handleOutput();

    if (quad_ptr_->getCollision()) {
      // collision happened
      ROS_INFO("COLLISION");
    }
  }
}

void FastWrapper::startCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
  ROS_DEBUG("time: %.2f", ros::Time::now().toSec());
  time_init_ = ros::Time::now().toSec();
  ROS_DEBUG("time: %.2f", time_init_);
  ROS_DEBUG("time: %.2f", ros::Time::now().toSec());
  flag_init_ = true;
}

void FastWrapper::mainLoopCallback(const ros::TimerEvent& event) {
  // empty
  // ROS_INFO("mainLoopCallback test %d..", flag_init_);
  // update target
  // if (time_init_ < 0.0) {
  //   time_init_ = ros::Time::now().toSec();
  // }
  if (flag_init_) {
    Scalar t0 = static_cast<float>(ros::Time::now().toSec() - time_init_);
    if (t0 > 200) {
      ROS_DEBUG("t0: %.2f | vMax: %.2f", t0, targetMaxV_);
    }
    // ROS_DEBUG("t0: %.2f", t0);
    // Scalar traj_param = sqrt(vel_param_);
    Scalar traj_param = 1.0;
    Scalar omega = 1.41 * M_PI / 100.0 * vel_param_;
    target_xyY_(0) = target_init_(0) + 16.0 * traj_param * sin(omega * t0);
    target_xyY_(1) = target_init_(1) + 8.0 * traj_param * sin(2 * omega * t0);
    Scalar dx = 16.0 * traj_param * omega * cos(omega * t0);
    Scalar dy = 16.0 * traj_param * omega * cos(2.0 * omega * t0);
    target_xyY_(2) = atan2(dy, dx);
    Scalar speed_0 = sqrt(dx * dx + dy * dy);
    if (targetMaxV_ < speed_0) targetMaxV_ = speed_0;

    if (t0_last_ > 0) {
      Scalar speed_1 =
        (target_xyY_ - target_last_).head(2).norm() / (t0 - t0_last_);
      logger_.info(
        "target vel: %.2f | vel1: %.2f | Max vel: %.2f\nPose: [%.2f, %.2f, "
        "%.2f]",
        speed_0, speed_1, targetMaxV_, target_xyY_(0), target_xyY_(1),
        target_xyY_(2));
      // ROS_INFO("target vel: %.2f | vel1: %.2f | Max vel: %.2f", speed_0,
      //  speed_1, targetMaxV_);
    }
    t0_last_ = t0;
    target_last_ = target_xyY_;

    pubtargetGT();
    detect_.updateTarget(target_xyY_);
    BBox result_bbox;
    bool is_valid = detect_.getBBox(quad_state_, result_bbox);
    if (is_valid) {
      pubtargetDetect();
    }
    if (!has_odom_) return;
    lidar_.renderLaserScan(quad_state_);
    if (lidar_.isCollision()) {
      ROS_ERROR("Drone collision.");
    }

    // For data analysis
    geometry_msgs::PoseWithCovarianceStamped debug_msg;
    debug_msg.header.stamp = ros::Time::now();
    debug_msg.header.frame_id = "txyY/dxyzY/cuv/tv/tvM/t0";
    debug_msg.pose.pose.position.x = target_xyY_(0);
    debug_msg.pose.pose.position.y = target_xyY_(1);
    debug_msg.pose.pose.position.z = target_xyY_(2);
    debug_msg.pose.pose.orientation.x = quad_state_.x[QS::POSX];
    debug_msg.pose.pose.orientation.y = quad_state_.x[QS::POSY];
    debug_msg.pose.pose.orientation.z = quad_state_.x[QS::POSZ];
    debug_msg.pose.pose.orientation.w = quad_state_.euler_xyz().z();
    std::fill(debug_msg.pose.covariance.begin(),
              debug_msg.pose.covariance.end(), 0.0);

    debug_msg.pose.covariance[0] = result_bbox.u_min;
    debug_msg.pose.covariance[1] = result_bbox.u_max;
    debug_msg.pose.covariance[2] = result_bbox.v_min;
    debug_msg.pose.covariance[3] = result_bbox.v_max;
    debug_msg.pose.covariance[4] = speed_0;
    debug_msg.pose.covariance[5] = targetMaxV_;
    debug_msg.pose.covariance[6] = t0;
    if (lidar_.isCollision()) {
      debug_msg.pose.covariance[7] = 1.0;
    }
    debug_pub_.publish(debug_msg);

  } else {
    pubtargetGT();
    detect_.updateTarget(target_xyY_);
    BBox result_bbox;
    bool is_valid = detect_.getBBox(quad_state_, result_bbox);
    if (is_valid) {
      pubtargetDetect();
    }
    if (!has_odom_) return;
    lidar_.renderLaserScan(quad_state_);
    if (lidar_.isCollision()) {
      ROS_ERROR("Drone collision.");
    }
  }
  
  
  
  const auto& local_cloud = lidar_.getlocalCloud();
  sensor_msgs::PointCloud2 local_cloud_msg;
  pcl::toROSMsg(local_cloud, local_cloud_msg);
  local_cloud_msg.header.stamp = ros::Time::now();
  local_cloud_msg.header.frame_id = "laser";

  sensor_msgs::PointCloud2 transformed_cloud;
  pcl_ros::transformPointCloud("world", transform_.transform,
                               local_cloud_msg, transformed_cloud);
  transformed_cloud.header.stamp = ros::Time::now();
  transformed_cloud.header.frame_id = "world";
  cloud_pub_.publish(transformed_cloud);

  const auto& scan_data = lidar_.getScan();
  scan_msg_.ranges = scan_data;
  scan_msg_.header.stamp = local_cloud_msg.header.stamp;
  scan_pub_.publish(scan_msg_);
  return;
}


void FastWrapper::pubtargetGT() {
  nav_msgs::Odometry target;
  target.header.stamp = ros::Time::now();
  target.header.frame_id = "world";
  target.pose.pose.position.x = target_xyY_.x();
  target.pose.pose.position.y = target_xyY_.y();
  target.pose.pose.position.z = 0.0;
  target.pose.pose.orientation.w = std::cos(target_xyY_.z() / 2.0);
  target.pose.pose.orientation.x = 0.0;
  target.pose.pose.orientation.y = 0.0;
  target.pose.pose.orientation.z = std::sin(target_xyY_.z() / 2.0);
  target.twist.twist.linear.x = 0.0;
  target.twist.twist.linear.y = 0.0;
  target.twist.twist.angular.z = 0.0;
  targetGT_pub_.publish(target);
}
void FastWrapper::pubtargetDetect() {
  nav_msgs::Odometry target;
  target.header.stamp = ros::Time::now();
  target.header.frame_id = "world";
  // target.pose.pose.position.x = target_xyY_.x() - 3.0*cos(target_xyY_.z());
  // target.pose.pose.position.y = target_xyY_.y() - 3.0*sin(target_xyY_.z());
  target.pose.pose.position.x = target_xyY_.x();
  target.pose.pose.position.y = target_xyY_.y();
  target.pose.pose.position.z = 0.0;

  target.pose.pose.orientation.w = std::cos(target_xyY_.z() / 2.0);
  target.pose.pose.orientation.x = 0.0;
  target.pose.pose.orientation.y = 0.0;
  target.pose.pose.orientation.z = std::sin(target_xyY_.z() / 2.0);
  target.twist.twist.linear.x = 0.0;
  target.twist.twist.linear.y = 0.0;
  target.twist.twist.angular.z = 0.0;
  targetDetect_pub_.publish(target);
}

bool FastWrapper::setUnity(const bool render) {
  unity_render_ = render;
  if (unity_render_ && unity_bridge_ptr_ == nullptr) {
    // create unity bridge
    unity_bridge_ptr_ = UnityBridge::getInstance();
    unity_bridge_ptr_->addQuadrotor(quad_ptr_);
    ROS_INFO("[%s] Unity Bridge is created.", pnh_.getNamespace().c_str());
  }
  return true;
}

bool FastWrapper::connectUnity() {
  if (!unity_render_ || unity_bridge_ptr_ == nullptr) return false;
  unity_ready_ = unity_bridge_ptr_->connectUnity(scene_id_);
  return unity_ready_;
}

bool FastWrapper::loadParams(void) {
  // load parameters
  quadrotor_common::getParam("main_loop_freq", main_loop_freq_, pnh_);
  quadrotor_common::getParam("unity_render", unity_render_, pnh_);
  quadrotor_common::getParam("vel", vel_param_, pnh_);

  return true;
}

}  // namespace flightros