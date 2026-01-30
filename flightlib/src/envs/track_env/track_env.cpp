/***
 * @Author: Lac_Creeper
 * @Date: 2025-05-23 15:11:53 +0800
 * @LastEditTime: 2025-06-09 15:48:29 +0800
 * @LastEditors: Lac_Creeper
 * @Description:
 * @FilePath: /flightmare/flightlib/src/envs/track_env/track_env.cpp
 */
#include "flightlib/envs/track_env/track_env.hpp"

namespace flightlib {

TrackEnv::TrackEnv()
  : TrackEnv(getenv("FLIGHTMARE_PATH") +
             std::string("/flightlib/configs/track_env.yaml")) {}

TrackEnv::TrackEnv(const std::string &cfg_path)
  : EnvBase(),
    detect_coeff_{0.0f},
    dist_coeff_{0.0f},
    alpha_coeff_{0.0f},
    pos_coeff_{0.0f},
    theta_coeff_{0.0f},
    act_coeff_{0.0f},
    fine_turn_{0},
    random_{0},
    target_maxV_{0.0f},
    target_V_{0.0f},
    tMaxV_{1.0f},
    use_ros_{0} {
  // load configuration file
  YAML::Node cfg_ = YAML::LoadFile(cfg_path);
  // load parameters
  loadParam(cfg_);
  logger_.info("use_ros: %d", use_ros_);

  if (use_ros_) {
    int argc = 0;
    char **argv = NULL;
    ros::init(argc, argv, "track_env_cpp");

    nh_ = std::make_unique<ros::NodeHandle>();

    map_pub_ = nh_->advertise<visualization_msgs::MarkerArray>("/obstacles", 1);
    scan_pub_ = nh_->advertise<sensor_msgs::LaserScan>("/scan", 1);
    odom_pub_ = nh_->advertise<nav_msgs::Odometry>("/odom", 1);
    target_pub_ = nh_->advertise<nav_msgs::Odometry>("/target", 1);
    debug_pub_ =
      nh_->advertise<geometry_msgs::PoseWithCovarianceStamped>("/debug", 1);
  }

  quadrotor_ptr_ = std::make_shared<Quadrotor>();
  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quadrotor_ptr_->updateDynamics(dynamics);

  // define a world box
  world_box_ << -50, 50, -50, 50, 0, 20;
  if (!quadrotor_ptr_->setWorldBox(world_box_)) {
    logger_.error("Cannot set world box!!");
  }

  // define input and output dimension for the environment
  obs_dim_ = trackenv::kNObs;
  act_dim_ = trackenv::kNAct;

  act_std_.setZero();
  act_std_.x() = 2.5f;
  act_std_.y() = 2.0f;
  act_std_.z() = 0.5f;
  // detect settings
  // desired_dist_ = 6.0f;   //For fov_69
  desired_dist_ = 3.0f;   //For fov_108
  // desired_bbox_ = {449, 158, 510, 362};  // For fov_69
  desired_bbox_ = {448, 154, 511, 365};  // For fov_108

  // lidar settings
  lidar_.loadMap("/home/lac/fm_test/my_logs/lac_map.log");
  // lidar_.generateRandomMap(30, 25);
  // lidar_.generateRandomMap(20, 20);

  if (use_ros_) {
    std::vector<std::shared_ptr<Obstacle>> obstacles;
    obstacles = lidar_.getObstacles();
    for (int i = 0; i < 20; ++i) {
      visualizeObstacles(obstacles);
      ros::Rate loop_rate(10);
      loop_rate.sleep();
    }
  }
  
  logger_.debug("Init success..");
}

TrackEnv::~TrackEnv() {}


Vector<3> TrackEnv::convVel() {
  Scalar yaw = quad_state_.euler_xyz().z();
  Vector<3> localVel;
  localVel.x() = cos(yaw) * quad_state_.v.x() + sin(yaw) * quad_state_.v.y();
  localVel.y() = -sin(yaw) * quad_state_.v.x() + cos(yaw) * quad_state_.v.y();
  // logger_.info("local_v: [%.2f, %.2f]", localVel.x(), local_vy);
  localVel.z() = quad_state_.x(QS::OMEZ);
  return localVel;
}

bool TrackEnv::reset(Ref<Vector<>> obs, const bool random) {
  step_num_ = 0;
  reach_count_ = 0;
  quad_state_.setZero();
  track_act_.setZero();
  last_act_.setZero();
  // traj_param_ = 1.75 + uniform_dist_(random_gen_) * 0.75;
  // traj_param_ = 2.5;
  traj_param_ = uniform_dist_(random_gen_) * 0.5 + 0.5;


  if (random_) {
    bool has_collision = true;
    bool has_visual = false;

    while (has_collision || (!has_visual)) {
      // randomly reset the quadrotor state
      Scalar init_x = uniform_dist_(random_gen_) * 3.0f;
      Scalar init_y = uniform_dist_(random_gen_) * 3.0f;
      Scalar tag_x = uniform_dist_(random_gen_) * 5.0f + 5.0 + desired_dist_;
      Scalar tag_y = uniform_dist_(random_gen_) * 2.0f;
      // Scalar tag_x = desired_dist_;
      // Scalar tag_y = 0.0f;
      quad_state_.x(QS::POSX) = init_x;
      quad_state_.x(QS::POSY) = init_y;
      quad_state_.x(QS::POSZ) = 0.8f;
      Scalar yaw = uniform_dist_(random_gen_) * M_PI;
      quad_state_.x(QS::ATTW) = std::cos(yaw / 2.0);
      quad_state_.x(QS::ATTX) = 0.0f;
      quad_state_.x(QS::ATTY) = 0.0f;
      quad_state_.x(QS::ATTZ) = std::sin(yaw / 2.0);
      quad_state_.qx /= quad_state_.qx.norm();
      // reset the target
      target_xyY_.x() = init_x + tag_x * std::cos(yaw) - tag_y * std::sin(yaw);
      target_xyY_.y() = init_y + tag_x * std::sin(yaw) + tag_y * std::cos(yaw);
      target_xyY_.z() = uniform_dist_(random_gen_) * M_PI;
      // target_xyY_.z() = 0.0;
      targetInitPose_ = target_xyY_;
      // target_xyY_.z() = uniform_dist_(random_gen_) * M_PI;
      // check collision
      std::vector<Scalar> temp_scan;
      has_collision = lidar_.simulateLidar(quad_state_, temp_scan, 1.0, false);
      // lidar_.renderPointCloud(quad_state_);
      // has_collision = lidar_.isCollision();
      // check target in FOV
      detect_.updateTarget(target_xyY_);
      detect_.getBBoxG(quad_state_, detect_bbox_);
      has_visual = detect_bbox_.is_valid();
    }
  }
  else {
    bool has_collision = true;
    bool has_visual = false;
    while (has_collision || (!has_visual)) {
      // Scalar init_x = uniform_dist_(random_gen_) * 2.0f;
      // Scalar init_y = uniform_dist_(random_gen_) * 2.0f;
      Scalar init_x = 0.0;
      Scalar init_y = 0.0;
      quad_state_.x(QS::POSX) = init_x;
      quad_state_.x(QS::POSY) = init_y;
      quad_state_.x(QS::POSZ) = 0.8f;
      // Scalar yaw = uniform_dist_(random_gen_) * M_PI;
      Scalar yaw = 0.0;
      quad_state_.x(QS::ATTW) = std::cos(yaw / 2.0);
      quad_state_.x(QS::ATTX) = 0.0f;
      quad_state_.x(QS::ATTY) = 0.0f;
      quad_state_.x(QS::ATTZ) = std::sin(yaw / 2.0);
      quad_state_.qx /= quad_state_.qx.norm();
      // Scalar tag_x = uniform_dist_(random_gen_) + desired_dist_;
      // Scalar tag_y = uniform_dist_(random_gen_) * 0.5f;
      Scalar tag_x = desired_dist_;
      Scalar tag_y = 0.0;
      target_xyY_.x() = init_x + tag_x * std::cos(yaw) - tag_y * std::sin(yaw);
      target_xyY_.y() = init_y + tag_x * std::sin(yaw) + tag_y * std::cos(yaw);
      target_xyY_.z() = yaw;
      targetInitPose_ = target_xyY_;
      std::vector<Scalar> temp_scan;
      has_collision = lidar_.simulateLidar(quad_state_, temp_scan, 1.0, false);
      // check target in FOV
      detect_.updateTarget(target_xyY_);
      detect_.getBBoxG(quad_state_, detect_bbox_);
      has_visual = detect_bbox_.is_valid();
    }
  }
  quadrotor_ptr_->reset(quad_state_);

  cmd_.t = 0.0;
  cmd_.linear.setZero();
  cmd_.angular.setZero();

  // obtain observations
  has_init_obs_ = false;
  has_init_reward_ = false;
  if (use_ros_) {
    visualizeTarget();
  }
  getObs(obs);
  // logger_.debug("reset success..");
  return true;
}

bool TrackEnv::getObs(Ref<Vector<>> obs) {
  // logger_.debug("getObs start..");

  quadrotor_ptr_->getState(&quad_state_);
  // lidar_.renderLaserScan(quad_state_, true);
  // const auto &scan_data = lidar_.getScan();
  std::vector<Scalar> scan_data;
  bool has_collision = lidar_.simulateLidar(quad_state_, scan_data, 0.2, true);

  Vector<trackenv::kNLaser1> scan =
    Vector<trackenv::kNLaser1>::Map(scan_data.data(), scan_data.size());

  detect_.updateTarget(target_xyY_);
  detect_.getBBoxG(quad_state_, detect_bbox_);
  // std::cout << detect_bbox_ << std::endl;

  Scalar yaw = quad_state_.euler_xyz().z();
  Scalar theta = target_xyY_.z() - yaw;

  if (!has_init_obs_) {
    track_obs_.segment<trackenv::kNLaser1>(trackenv::kLaser1) = scan;
    track_obs_.segment<trackenv::kNLaser2>(trackenv::kLaser2) = scan;
    track_obs_.segment<trackenv::kNLaser3>(trackenv::kLaser3) = scan;
    track_obs_.segment<trackenv::kNDetect>(trackenv::kDetect) =
      detect_bbox_.obs();
    track_obs_.segment<trackenv::kNDirt>(trackenv::kDirt) = 
      Vector<trackenv::kNDirt>{std::cos(theta), std::sin(theta)};
    track_obs_.segment<trackenv::kNState>(trackenv::kState) =
      Vector<trackenv::kNAct>::Zero();
    has_init_obs_ = true;
  } else {
    track_obs_.segment<trackenv::kNLaser1>(trackenv::kLaser1) =
      track_obs_.segment<trackenv::kNLaser2>(trackenv::kLaser2);
    track_obs_.segment<trackenv::kNLaser2>(trackenv::kLaser2) =
      track_obs_.segment<trackenv::kNLaser3>(trackenv::kLaser3);
    track_obs_.segment<trackenv::kNLaser3>(trackenv::kLaser3) = scan;
    track_obs_.segment<trackenv::kNDetect>(trackenv::kDetect) =
      detect_bbox_.obs();
    track_obs_.segment<trackenv::kNDirt>(trackenv::kDirt) =
      Vector<trackenv::kNDirt>{std::cos(theta), std::sin(theta)};
    track_obs_.segment<trackenv::kNState>(trackenv::kState) =
      convVel();
  }

  obs.segment<trackenv::kNObs>(trackenv::kObs) = track_obs_;

  // -DEBUG: 
  if (use_ros_) {
    visualizeScan();
    visualizeOdom();
    ros::Rate loop_rate(100);
    loop_rate.sleep();
  }

  return true;
}

Scalar TrackEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs) {
  step_num_ += 1;
  last_act_ = track_act_;
  Scalar k_v = 1.0;
  // Scalar k_v = 0.75;
  track_act_ = k_v * act.cwiseProduct(act_std_) + (1 - k_v) * last_act_;
  cmd_.t += sim_dt_;
  cmd_.linear.x() = track_act_[0];
  cmd_.linear.y() = track_act_[1];
  cmd_.angular.z() = track_act_[2];

  // simulate quadrotor
  quadrotor_ptr_->velocityControlBody(cmd_, sim_dt_);
  // quadrotor_ptr_->simpleVelControlBody(cmd_, sim_dt_);

  // target_xyY_.x() = targetInitPose_.x() + 6 * sin(2 * M_PI / 160 * cmd_.t);
  // target_xyY_.y() = targetInitPose_.y() + 6 * sin(2 * M_PI / 80 * cmd_.t);
  if (use_ros_) {
    /*>>>>>>>-8->>>>>>*/
    // Scalar traj_param = sqrt(tMaxV_);
    Scalar traj_param = 1.0;
    Scalar omega = 1.41 * M_PI / 100.0 * tMaxV_;
    Vector<3> last_xyY = target_xyY_;
    target_xyY_.x() =
      targetInitPose_.x() + 16.0 * traj_param * sin(omega * cmd_.t);
    target_xyY_.y() =
      targetInitPose_.y() + 8.0 * traj_param * sin(2 * omega * cmd_.t);
    Scalar dx_dt = 16.0 * traj_param * omega * cos(omega * cmd_.t);
    Scalar dy_dt = 16.0 * traj_param * omega * cos(2.0 * omega * cmd_.t);
    target_xyY_.z() = atan2(dy_dt, dx_dt);
    Scalar actual_speed = std::sqrt(dx_dt * dx_dt + dy_dt * dy_dt);
    if (target_maxV_ < actual_speed) {
      target_maxV_ = actual_speed;
    }
    target_V_ = actual_speed;
    Scalar speed = (target_xyY_ - last_xyY).head(2).norm() / sim_dt_;
    std::cout << "目标速度: " << 1.0 << ", 实际速度: " << actual_speed
              << ", Vel: " << speed
              << ", MaxV: " << target_maxV_
              << std::endl;
    /*<<<<<<<<<<<-8-<<<<<<<<*/

    /*>>>>>-Tra>>>>>>-*/
    // Scalar side_length = 6.0;
    // Scalar corner_radius = 0.5;
    // Scalar speed = 1.0;
    // Vector<3> vertex_A = targetInitPose_;
    // Vector<3> vertex_B = vertex_A;
    // vertex_B.x() += side_length;
    // Vector<3> vertex_C = vertex_A;
    // vertex_C.x() += side_length * 0.5;
    // vertex_C.y() += side_length * sqrt(3.0) * 0.5;

    // Vector<3> current_pos;
    // if (cmd_.t)


    /*<<<<<<<-Tra-<<<<<<<*/

    visualizeTarget();
  }
  // Fine_turn
  if (fine_turn_) {
    // Scalar traj_param = sqrt(traj_param_);
    Scalar traj_param = 1.0;
    Scalar omega = M_PI / 100.0 * traj_param_;
    Vector<3> last_xyY = target_xyY_;
    target_xyY_.x() =
      targetInitPose_.x() + 24.0 * traj_param * sin(omega * cmd_.t);
    target_xyY_.y() =
      targetInitPose_.y() + 12.0 * traj_param * sin(2 * omega * cmd_.t);
    Scalar dx_dt = 24.0 * traj_param * omega * cos(omega * cmd_.t);
    Scalar dy_dt = 24.0 * traj_param * omega * cos(2.0 * omega * cmd_.t);
    target_xyY_.z() = atan2(dy_dt, dx_dt);
  }

  // update observations
  getObs(obs);

  // ================= reward function design ===================
  // - detection term
  Scalar detect_reward = detect_coeff_ * detect_bbox_.IoU(desired_bbox_);
  // Vector<3> detect_norm = obs.segment<trackenv::kNDetect>(trackenv::kDetect);
  // Scalar detect_reward = 
  //     -0.1*abs(detect_norm.x()) - 0.1*abs(detect_norm.y()) - 0.1*abs(detect_norm.z());

  // - position term
  Vector<2> target2drone = 
      target_xyY_.segment<2>(0) - quad_state_.x.segment<2>(QS::POS);
  Scalar target_yaw = target_xyY_.z();
  Vector<2> vector_dir = target2drone.normalized();
  Scalar yaw = quad_state_.euler_xyz().z();
  if (!has_init_reward_) {
    last_alpha_ = cos(yaw) * vector_dir.x() + sin(yaw) * vector_dir.y();
    last_dist_ = abs(target2drone.norm() - desired_dist_);
    last_theta_ = cos(yaw) * cos(target_yaw) + sin(yaw) * sin(target_yaw);
    has_init_reward_ = true;
  }
  Scalar alpha = cos(yaw) * vector_dir.x() + sin(yaw) * vector_dir.y();
  Scalar dist = abs(target2drone.norm() - desired_dist_);

  Scalar pos_reward = (last_dist_ - dist) * dist_coeff_ + 
                      (alpha - last_alpha_) * alpha_coeff_;
  last_alpha_ = alpha;
  last_dist_ = dist;

  // - theta term
  /// TODO: add direction vector reward
  Scalar theta = cos(yaw) * cos(target_yaw) + sin(yaw) * sin(target_yaw);
  Scalar theta_reward = (theta - 1) * theta_coeff_;
  last_theta_ = theta;

  // - control action penalty
  Scalar act_norm = act.cast<Scalar>().norm();
  Scalar acc_punish = 0.0;
  Vector<3> vel_xyY = obs.segment<trackenv::kNState>(trackenv::kState);
  // logger_.warn("vel_xyY: [%.2f, %.2f, %.2f]", vel_xyY[0], vel_xyY[1],
  //              vel_xyY[2]);
  // logger_.warn("last_act_: [%.2f, %.2f, %.2f]", last_act_[0], last_act_[1],
  //              last_act_[2]);
  Vector<3> acc_xyY = (track_act_ - vel_xyY) / sim_dt_;
  // if (acc_xyY.head(2).norm() > 1.0 || acc_xyY.z() > 0.35) {
  //   acc_punish = -0.1;
  // }
  Scalar act_reward = -0.005 * act_norm -
                      0.005 * (track_act_ - vel_xyY).squaredNorm() -
                      0.002 * acc_xyY.squaredNorm();
  // if (act_norm < 2.0) {
  //   act_norm = 0.0;
  // }
  // Scalar act_reward = -0.0025 * act_norm - 0.005 * (track_act_ - last_act_).norm();

  // - reach reward
  Scalar reach_reward = 0.0;
  if ((detect_bbox_.IoU(desired_bbox_) > 0.8) && theta > 0.95) {
    reach_reward = 0.2;
    reach_count_ += 1;
    if (reach_count_ > 10 && (detect_bbox_.IoU(desired_bbox_) > 0.85) &&
        theta > 0.98) {
      reach_reward += 0.3;
    }
  } else {
    reach_count_ = 0;
  }

  Scalar total_reward =
    detect_reward + pos_reward + theta_reward + act_reward + reach_reward;
  // logger_.debug(
  //   "[reward]: \t DET\t POS\t THE\t ACT\t REA\n"
  //   "\t\t[VEL]: \t %.3f\t %.3f\t %.3f\t %.3f\t %.3f\n",
  //   detect_reward, pos_reward, theta_reward, act_reward, reach_reward);

  // survival reward
  total_reward += 0.1f;

  // DEBUG: Show the relative distance in real time.
  if (use_ros_) {
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
    debug_msg.pose.covariance[0] = detect_bbox_.u_min;
    debug_msg.pose.covariance[1] = detect_bbox_.u_max;
    debug_msg.pose.covariance[2] = detect_bbox_.v_min;
    debug_msg.pose.covariance[3] = detect_bbox_.v_max;
    debug_msg.pose.covariance[4] = target_V_;
    debug_msg.pose.covariance[5] = target_maxV_;
    debug_msg.pose.covariance[6] = cmd_.t;

    debug_pub_.publish(debug_msg);
    std::cout << "V_act: [" << track_act_[0] << ", " << track_act_[1] << "]\n"
              << "V_rel: [" << vel_xyY[0] << ", " << vel_xyY[1] << "]\n"
              << std::endl;
  }

  return total_reward;
}

bool TrackEnv::isTerminalState(Scalar &reward) {
  if (!detect_bbox_.is_valid()) {
    reward = -40;
    logger_.warn("target loss..%d", step_num_);
    return true;
  }
  if (lidar_.isCollision()) {
    reward = -50;
    logger_.warn("drone collision..%.d", step_num_);
    return true;
  }
  if (step_num_ >= 300 && reach_count_ > 20) {
    logger_.debug("time out..%d, %d", step_num_, reach_count_);
  }
  if ((abs(quad_state_.v.x()) > 3.5) || (abs(quad_state_.v.y()) > 3.5)) {
    logger_.fatal("control error..%d", step_num_);
  }
  reward = 0.0f;
  return false;
}

bool TrackEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["track_env"]) {
    sim_dt_ = cfg["track_env"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["track_env"]["max_t"].as<Scalar>();
    use_ros_ = cfg["track_env"]["use_ros"].as<int>();
    fine_turn_ = cfg["track_env"]["fine_turn"].as<int>();
    random_ = cfg["track_env"]["random"].as<int>();
    tMaxV_ = cfg["track_env"]["tMaxV"].as<Scalar>();
    logger_.info("cfg:use_ros: %d", cfg["track_env"]["use_ros"].as<int>());
  } else {
    return false;
  }

  if (cfg["rl"]) {
    // load RL related parameters
    detect_coeff_ = cfg["rl"]["detect_coeff"].as<Scalar>();
    dist_coeff_ = cfg["rl"]["dist_coeff"].as<Scalar>();
    alpha_coeff_ = cfg["rl"]["alpha_coeff"].as<Scalar>();
    pos_coeff_ = cfg["rl"]["pos_coeff"].as<Scalar>();
    theta_coeff_ = cfg["rl"]["theta_coeff"].as<Scalar>();
    act_coeff_ = cfg["rl"]["act_coeff"].as<Scalar>();
  } else {
    return false;
  }

  return true;
}

void TrackEnv::visualizeObstacles(
  std::vector<std::shared_ptr<Obstacle>> &obstacles) {
  visualization_msgs::MarkerArray markers;
  int id = 0;

  for (const auto &obs : obstacles) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.ns = "obstacles";
    marker.id = id++;
    marker.action = visualization_msgs::Marker::ADD;

    if (auto rect = dynamic_cast<Rectangle *>(obs.get())) {
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
    } else if (auto ellipse = dynamic_cast<Ellipse *>(obs.get())) {
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

  map_pub_.publish(markers);
}

void TrackEnv::visualizeScan() {
  // 发布TF (假设机器人在地图中心)
  static tf::TransformBroadcaster br;
  tf::Transform transform;
  transform.setOrigin(tf::Vector3(quad_state_.p.x(), quad_state_.p.y(), 0.0));
  // transform.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
  Vector<3> euler_xyz = quad_state_.euler_xyz();
  Scalar robot_yaw = euler_xyz.z();
  transform.setRotation(
    tf::Quaternion(0, 0, sin(robot_yaw / 2), cos(robot_yaw / 2)));
  // transform.setRotation(tf::Quaternion(0, 0, 0, 1));
  br.sendTransform(
    tf::StampedTransform(transform, ros::Time::now(), "world", "base_laser"));

  sensor_msgs::LaserScan scan;
  scan.header.stamp = ros::Time::now();
  scan.header.frame_id = "base_laser";
  scan.angle_min = -M_PI;
  scan.angle_max = M_PI;
  scan.angle_increment = 2 * M_PI / (512 - 1);
  scan.time_increment = 0;
  scan.scan_time = 0.1;
  scan.range_min = 0.1;
  scan.range_max = 6.0;
  bool is_collision =
    lidar_.simulateLidar(quad_state_, scan.ranges, 0.2, false);
  scan_pub_.publish(scan);
}

void TrackEnv::visualizeOdom() {
  nav_msgs::Odometry odom;
  odom.header.stamp = ros::Time::now();
  odom.header.frame_id = "world";
  odom.pose.pose.position.x = quad_state_.x[QS::POSX];
  odom.pose.pose.position.y = quad_state_.x[QS::POSY];
  odom.pose.pose.position.z = quad_state_.x[QS::POSZ];
  odom.pose.pose.orientation.w = quad_state_.x[QS::ATTW];
  odom.pose.pose.orientation.x = quad_state_.x[QS::ATTX];
  odom.pose.pose.orientation.y = quad_state_.x[QS::ATTY];
  odom.pose.pose.orientation.z = quad_state_.x[QS::ATTZ];
  // odom.twist.twist.linear.x = track_act_.x();
  // odom.twist.twist.linear.y = track_act_.y();
  odom.twist.twist.linear.x = quad_state_.x(QS::VELX);
  odom.twist.twist.linear.y = quad_state_.x(QS::VELY);
  odom.twist.twist.linear.z = 0.0;
  odom.twist.twist.angular.x = 0.0;
  odom.twist.twist.angular.y = 0.0;
  odom.twist.twist.angular.z = quad_state_.x(QS::OMEZ);
  odom_pub_.publish(odom);
}

void TrackEnv::visualizeTarget() {
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
  target_pub_.publish(target);
}

bool TrackEnv::getAct(Ref<Vector<>> act) const {
  if (cmd_.t >= 0.0 && track_act_.allFinite()) {
    act = track_act_;
    return true;
  }
  return false;
}

bool TrackEnv::getAct(Command *const cmd) const {
  if (!cmd_.valid()) return false;
  *cmd = cmd_;
  return true;
}

void TrackEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  bridge->addQuadrotor(quadrotor_ptr_);
}

std::ostream &operator<<(std::ostream &os, const TrackEnv &track_env) {
  os.precision(3);
  os << "Tracking Environment:\n"
     << "obs dim =            [" << track_env.obs_dim_ << "]\n"
     << "act dim =            [" << track_env.act_dim_ << "]\n"
     << "sim dt =             [" << track_env.sim_dt_ << "]\n"
     << "max_t =              [" << track_env.max_t_ << "]\n"
     << "act_std =            [" << track_env.act_std_.transpose() << std::endl;
  os.precision();
  return os;
}


}  // namespace flightlib
