/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-06-16 12:28:21 +0800
 * @LastEditTime: 2025-06-21 08:42:37 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /flightmare/flightlib/src/envs/obstacle_env/obstacle_env.cpp
 */
#include "flightlib/envs/obstacle_env/obstacle_env.hpp"

namespace flightlib {

ObstacleEnv::ObstacleEnv()
  : ObstacleEnv(getenv("FLIGHTMARE_PATH") +
             std::string("/flightlib/configs/obstacle_env.yaml")) {}

ObstacleEnv::ObstacleEnv(const std::string &cfg_path)
  : EnvBase() {
  // load configuration file
  YAML::Node cfg_ = YAML::LoadFile(cfg_path);

	if (use_ros_) {
    int argc = 0;
    char **argv = NULL;
    ros::init(argc, argv, "obstacle_env_cpp");

    nh_ = std::make_unique<ros::NodeHandle>();

    map_pub_ = nh_->advertise<visualization_msgs::MarkerArray>("/obstacles", 1);
    scan_pub_ = nh_->advertise<sensor_msgs::LaserScan>("/scan", 1);
    odom_pub_ = nh_->advertise<nav_msgs::Odometry>("/odom", 1);
    target_pub_ = nh_->advertise<visualization_msgs::Marker>("/target", 1);
  }

  quadrotor_ptr_ = std::make_shared<Quadrotor>();
  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quadrotor_ptr_->updateDynamics(dynamics);

  // define a world box
  world_box_ << -20, 20, -20, 20, 0, 20;
  if (!quadrotor_ptr_->setWorldBox(world_box_)) {
    logger_.error("Cannot set world box!!");
  }

  // define input and output dimension for the environment
  obs_dim_ = obstenv::kNObs;
  act_dim_ = obstenv::kNAct;

  act_std_.setZero();
  act_std_.x() = 1.0f;
  act_std_.y() = 1.0f;

  // lidar settings
  lidar_.generateRandomMap(10, 10);

  // load parameters
  loadParam(cfg_);

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

ObstacleEnv::~ObstacleEnv() {}


void ObstacleEnv::visualizeObstacles(
  std::vector<std::shared_ptr<Obstacle>> &obstacles) {
  visualization_msgs::MarkerArray markers;
  int id = 0;
  for (const auto &obs : obstacles) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.ns = "obstacles";
    marker.id = id++;
    marker.action = visualization_msgs::Marker::ADD;

    if (auto rect = dynamic_cast<Rectangle *>(obs.get())) {
      marker.type = visualization_msgs::Marker::CUBE;
      marker.pose.position.x = rect->center().x();
      marker.pose.position.y = rect->center().y();
      marker.pose.position.z = 0;

      tf::Quaternion q;
      q.setRPY(0, 0, rect->angle());
      marker.pose.orientation.x = q.x();
      marker.pose.orientation.y = q.y();
      marker.pose.orientation.z = q.z();
      marker.pose.orientation.w = q.w();

      marker.scale.x = rect->width();
      marker.scale.y = rect->height();
      marker.scale.z = 0.1;  // 小高度

      marker.color.r = 0.0;
      marker.color.g = 0.5;
      marker.color.b = 0.5;
      marker.color.a = 1.0;
    } else if (auto ellipse = dynamic_cast<Ellipse *>(obs.get())) {
      marker.type = visualization_msgs::Marker::CYLINDER;
      marker.pose.position.x = ellipse->center().x();
      marker.pose.position.y = ellipse->center().y();
      marker.pose.position.z = 0;

      tf::Quaternion q;
      q.setRPY(0, 0, ellipse->angle());
      marker.pose.orientation.x = q.x();
      marker.pose.orientation.y = q.y();
      marker.pose.orientation.z = q.z();
      marker.pose.orientation.w = q.w();

      marker.scale.x = ellipse->a() * 2;  // 直径
      marker.scale.y = ellipse->b() * 2;
      marker.scale.z = 0.1;  // 小高度

      marker.color.r = 0.5;
      marker.color.g = 0.0;
      marker.color.b = 0.5;
      marker.color.a = 1.0;
    }

    marker.lifetime = ros::Duration(0);
    markers.markers.push_back(marker);
  }

  map_pub_.publish(markers);
}

void ObstacleEnv::visualizeScan() {
  // 发布TF (假设机器人在地图中心)
  static tf::TransformBroadcaster br;
  tf::Transform transform;
  transform.setOrigin(tf::Vector3(quad_state_.p.x(), quad_state_.p.y(), 0.0));
  Vector<3> euler_xyz = quad_state_.euler_xyz();
  Scalar robot_yaw = euler_xyz.z();
  transform.setRotation(
    tf::Quaternion(0, 0, sin(robot_yaw / 2), cos(robot_yaw / 2)));
  br.sendTransform(
    tf::StampedTransform(transform, ros::Time::now(), "map", "base_laser"));

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
  bool is_collision = lidar_.simulateLidar(quad_state_, scan.ranges, false);
  scan_pub_.publish(scan);
}

void ObstacleEnv::visualizeOdom() {
  nav_msgs::Odometry odom;
  odom.header.stamp = ros::Time::now();
  odom.header.frame_id = "map";
  odom.pose.pose.position.x = quad_state_.x[QS::POSX];
  odom.pose.pose.position.y = quad_state_.x[QS::POSY];
  odom.pose.pose.position.z = quad_state_.x[QS::POSZ];
  odom.pose.pose.orientation.w = quad_state_.x[QS::ATTW];
  odom.pose.pose.orientation.x = quad_state_.x[QS::ATTX];
  odom.pose.pose.orientation.y = quad_state_.x[QS::ATTY];
  odom.pose.pose.orientation.z = quad_state_.x[QS::ATTZ];
  odom.twist.twist.linear.x = obst_act_.x();
  odom.twist.twist.linear.y = 0.0;
  odom.twist.twist.linear.z = 0.0;
  odom.twist.twist.angular.x = 0.0;
  odom.twist.twist.angular.y = 0.0;
  odom.twist.twist.angular.z = obst_act_.y();
  odom_pub_.publish(odom);
}

void ObstacleEnv::visualizeTarget() {
  visualization_msgs::Marker target;
  target.header.frame_id = "map";
  target.ns = "target";
  target.id = 0;
  target.type = visualization_msgs::Marker::SPHERE;
  target.action = visualization_msgs::Marker::ADD;
  target.scale.x = 1;
  target.scale.y = 1;
  target.scale.z = 1;
  target.pose.orientation.w = 1;
  target.color.a = 0.8;
  target.color.g = 1.0;
  target.pose.position.x = target_xy_.x();
  target.pose.position.y = target_xy_.y();
  target_pub_.publish(target);
}

Vector<2> ObstacleEnv::convGoal() {
	Scalar yaw = quad_state_.euler_xyz().z();
	Matrix<2, 2> inv_rot = Eigen::Rotation2Df(yaw).toRotationMatrix().transpose();
	Vector<2> quad_xy = {quad_state_.x(QS::POSX), quad_state_.x(QS::POSY)};
	Vector<2> localGoal = inv_rot * (target_xy_ - quad_xy);
  // logger_.info("yaw: %.2f, goal: [%.2f, %.2f], quad: [%.2f, %.2f]", yaw,
  //               target_xy_.x(), target_xy_.y(), quad_xy.x(), quad_xy.y());
  // logger_.info("conv: [%.2f, %.2f]", localGoal.x(), localGoal.y());
  return localGoal;
}
Vector<2> ObstacleEnv::convVel() {
  Scalar yaw = quad_state_.euler_xyz().z();
  Vector<2> localVel;
	localVel.x() = cos(yaw) * quad_state_.v.x() + sin(yaw) * quad_state_.v.y();
  Scalar local_vy =
    -sin(yaw) * quad_state_.v.x() + cos(yaw) * quad_state_.v.y();
  // logger_.info("local_v: [%.2f, %.2f]", localVel.x(), local_vy);
	localVel.y() = quad_state_.x(QS::OMEZ);
  return localVel;
}


bool ObstacleEnv::reset(Ref<Vector<>> obs, const bool random) {
  // logger_.debug("reset start..");

  step_num_ = 0;
  quad_state_.setZero();
  obst_act_.setZero();

  if (random) {
    bool has_collision = true;
    bool goal_collision = true;
		Scalar dist2goal = 0.0f;

    while (goal_collision) {
      target_xy_.x() = uniform_dist_(random_gen_) * 5.0f;
      target_xy_.y() = uniform_dist_(random_gen_) * 5.0f;
      // target_xyY_.z() = uniform_dist_(random_gen_) * M_PI;
			std::vector<Scalar> ranges;
      goal_collision =
        lidar_.simulateLidar(target_xy_, 0.0f, ranges, false);
    }
    
    while (has_collision || (dist2goal < 4.0) || (dist2goal > 6.0f)) {
      // randomly reset the quadrotor state
      Scalar init_x = uniform_dist_(random_gen_) * 5.0f;
      Scalar init_y = uniform_dist_(random_gen_) * 5.0f;
      quad_state_.x(QS::POSX) = init_x;
      quad_state_.x(QS::POSY) = init_y;
      quad_state_.x(QS::POSZ) = 0.8f;
      Scalar yaw = uniform_dist_(random_gen_) * M_PI;
      quad_state_.x(QS::ATTW) = std::cos(yaw / 2.0);
      quad_state_.x(QS::ATTX) = 0.0f;
      quad_state_.x(QS::ATTY) = 0.0f;
      quad_state_.x(QS::ATTZ) = std::sin(yaw / 2.0);
      quad_state_.qx /= quad_state_.qx.norm();
      // check distance
      Vector<2> quad_xy = {quad_state_.x(QS::POSX), quad_state_.x(QS::POSY)};
      dist2goal = (quad_xy - target_xy_).norm();
      
      // check collision
      std::vector<Scalar> temp_scan;
      has_collision = lidar_.simulateLidar(quad_state_, temp_scan, false);
      
    }
		last_dist_ = dist2goal;
    Vector<2> local_target = convGoal();
    last_alpha_ = local_target.x() / local_target.norm();
  }
  quadrotor_ptr_->reset(quad_state_);

  cmd_.t = 0.0;
  cmd_.linear.setZero();
  cmd_.angular.setZero();

  // obtain observations
  has_init_obs_ = false;
	if (use_ros_) {
    visualizeTarget();
  }
  getObs(obs);
  // logger_.debug("reset success..");
  return true;
}

bool ObstacleEnv::getObs(Ref<Vector<>> obs) {
  // logger_.debug("getObs start..");

  quadrotor_ptr_->getState(&quad_state_);
  std::vector<Scalar> scan_data;
  bool has_collision = lidar_.simulateLidar(quad_state_, scan_data, true);
  Vector<obstenv::kNLaser1> scan =
    Vector<obstenv::kNLaser1>::Map(scan_data.data(), scan_data.size());

  if (!has_init_obs_) {
    obst_obs_.segment<obstenv::kNLaser1>(obstenv::kLaser1) = scan;
    obst_obs_.segment<obstenv::kNLaser2>(obstenv::kLaser2) = scan;
    obst_obs_.segment<obstenv::kNLaser3>(obstenv::kLaser3) = scan;
    obst_obs_.segment<obstenv::kNDetect>(obstenv::kDetect) =
      convGoal();
    obst_obs_.segment<obstenv::kNState>(obstenv::kState) =
      Vector<obstenv::kNAct>::Zero();
    has_init_obs_ = true;
  } else {
    obst_obs_.segment<obstenv::kNLaser1>(obstenv::kLaser1) =
      obst_obs_.segment<obstenv::kNLaser2>(obstenv::kLaser2);
    obst_obs_.segment<obstenv::kNLaser2>(obstenv::kLaser2) =
      obst_obs_.segment<obstenv::kNLaser3>(obstenv::kLaser3);
    obst_obs_.segment<obstenv::kNLaser3>(obstenv::kLaser3) = scan;
    obst_obs_.segment<obstenv::kNDetect>(obstenv::kDetect) =
      convGoal();
    obst_obs_.segment<obstenv::kNState>(obstenv::kState) =
      convVel();
  }

  obs.segment<obstenv::kNObs>(obstenv::kObs) = obst_obs_;

  // -DEBUG:
	if (use_ros_) {
    visualizeScan();
    visualizeOdom();
    ros::Rate loop_rate(100);
    loop_rate.sleep();
  }

  // logger_.debug("getObs success..");
  return true;
}

Scalar ObstacleEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs) {
  // logger_.debug("step start..");
  step_num_ += 1;

  obst_act_ = act.cwiseProduct(act_std_);
  cmd_.t += sim_dt_;
  cmd_.linear.x() = (obst_act_[0] + 1.0) / 2.0;
  cmd_.angular.z() = obst_act_[1];

  // simulate quadrotor
  quadrotor_ptr_->velocityControlBody(cmd_, sim_dt_);

  // update observations
  getObs(obs);

  // ================= reward function design ===================
	// - distance term
	Vector<2> quad_xy = {quad_state_.x(QS::POSX), quad_state_.x(QS::POSY)};
	Scalar dist = (quad_xy - target_xy_).norm();
	Scalar dist_reward = (last_dist_ - dist) * 2.5;
	Vector<2> local_target = convGoal();
	Scalar alpha = local_target.x() / local_target.norm();
  Scalar alpha_reward = (alpha - 1.0) * 0.05;
	alpha_reward = 0.0;
  last_alpha_ = alpha;
	last_dist_ = dist;
	// - action term
	Scalar act_reward = 0;
	if (abs(quad_state_.x(QS::OMEZ)) > 0.6) {
    act_reward = -0.2 * (abs(quad_state_.x(QS::OMEZ)) - 0.6);
  }
	// logger_.debug("[DIST]: %.2f, [ALPHA]: %.2f", dist_reward, alpha_reward);

  Scalar total_reward = dist_reward + act_reward + alpha_reward;
  // logger_.debug(
  //   "[reward]: \t DET\t POS\t THE\t ACT\t REA\n"
  //   "\t[VEL]: \t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\n",
  //   detect_reward, pos_reward, theta_reward, act_reward, reach_reward);

  // survival reward
  // total_reward += 0.1f;

  return total_reward;
}

bool ObstacleEnv::isTerminalState(Scalar &reward) {
  Vector<2> quad_xy = {quad_state_.x(QS::POSX), quad_state_.x(QS::POSY)};
  Scalar dist = (quad_xy - target_xy_).norm();
  if (dist < 0.5) {
    reward = 20.0;
    logger_.debug("reach goal..%d", step_num_);
    return true;
  }
  if (lidar_.isCollision()) {
    reward = -20.0;
    logger_.warn("drone collision..%.d", step_num_);
    return true;
  }
  if (step_num_ >= 300) {
    reward = 0.0f;
    logger_.warn("time out..");
    return true;
  }
  if ((abs(quad_state_.v.x()) > 2.5) || (abs(quad_state_.v.y()) > 2.5)) {
    reward = -5.0f;
    logger_.error("control error..%d", step_num_);
    return true;
  }
  reward = 0.0f;
  return false;
}

bool ObstacleEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["obstacle_env"]) {
    sim_dt_ = cfg["obstacle_env"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["obstacle_env"]["max_t"].as<Scalar>();
    use_ros_ = cfg["obstacle_env"]["use_ros_"].as<int>();
  } else {
    return false;
  }

  return true;
}

bool ObstacleEnv::getAct(Ref<Vector<>> act) const {
  if (cmd_.t >= 0.0 && obst_act_.allFinite()) {
    act = obst_act_;
    return true;
  }
  return false;
}

bool ObstacleEnv::getAct(Command *const cmd) const {
  if (!cmd_.valid()) return false;
  *cmd = cmd_;
  return true;
}

void ObstacleEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  bridge->addQuadrotor(quadrotor_ptr_);
}

std::ostream &operator<<(std::ostream &os, const ObstacleEnv &obstacle_env) {
  os.precision(3);
  os << "Tracking Environment:\n"
     << "obs dim =            [" << obstacle_env.obs_dim_ << "]\n"
     << "act dim =            [" << obstacle_env.act_dim_ << "]\n"
     << "sim dt =             [" << obstacle_env.sim_dt_ << "]\n"
     << "max_t =              [" << obstacle_env.max_t_ << "]\n"
     << "act_std =            [" << obstacle_env.act_std_.transpose() << std::endl;
  os.precision();
  return os;
}


}  // namespace flightlib
