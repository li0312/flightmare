/***
 * @Author: Lac_Creeper
 * @Date: 2025-06-05 20:07:06 +0800
 * @LastEditTime: 2025-06-07 16:31:32 +0800
 * @LastEditors: Lac_Creeper
 * @Description:
 * @FilePath: /flightmare/flightros/src/lac_test/lidar2d_sim.cpp
 */
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>

#include <memory>
#include <thread>
#include <vector>

#include "flightlib/sensors/lidar2D.hpp"


using namespace flightlib;


class ObstacleLidarSim {
 public:
  ObstacleLidarSim() : nh_("~") {
    // 参数
    nh_.param("map_width", map_width_, (Scalar)20.0);
    nh_.param("map_height", map_height_, (Scalar)20.0);
    nh_.param("num_rectangles", num_rectangles_, 50);
    nh_.param("num_ellipses", num_ellipses_, 30);
    nh_.param("lidar_fov", lidar_fov_, (Scalar)(2 * M_PI));
    nh_.param("lidar_num_rays", lidar_num_rays_, 512);
    nh_.param("lidar_max_range", lidar_max_range_, (Scalar)10.0);
    nh_.param("robot_init_x", robotPos_.x(), (Scalar)0.0);
    nh_.param("robot_init_y", robotPos_.y(), (Scalar)0.0);
    nh_.param("robot_init_Y", robotYaw_, (Scalar)(-45.0 * M_PI / 180));

    step_ = 0;

    // 发布者
    scan_pub_ = nh_.advertise<sensor_msgs::LaserScan>("/scan", 1);
    obstacles_pub_ =
      nh_.advertise<visualization_msgs::MarkerArray>("/obstacles", 1);

    // 定时器
    timer_ =
      nh_.createTimer(ros::Duration(0.01), &ObstacleLidarSim::update, this);


    lidar2d_sim_.generateRandomMap(50, 50);
    obstacles_ = lidar2d_sim_.getObstacles();
    // // 初始化地图
    // for (int i = 0; i < 200; ++i) {
    //   map_timer_.tic();
    //   obstacles_.clear();
    //   initializeMap();
    //   map_timer_.toc();
    // }
    // std::cout << map_timer_ << std::endl;
  }


  void robotMove() {
    Scalar step = (Scalar)((step_ % 360) / 360);
    robotPos_.x() += 10 * cos(step * 2 * M_PI);
    robotPos_.y() += 10 * sin(step * 2 * M_PI);
    robotYaw_ = M_PI / 2 - atan2(robotPos_.y(), robotPos_.x());
    std::cout << "[POS]: " << robotPos_ << std::endl;
    std::cout << "[YAW]: " << robotYaw_ << std::endl;
  }

  void initializeMap(const int MAX_ATTEMPTS = 100) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Scalar> posX(-map_width_ / 2,
                                                map_width_ / 2);
    // std::uniform_real_distribution<Scalar> posX(0, map_width_);
    std::uniform_real_distribution<Scalar> posY(-map_height_ / 2,
                                                map_height_ / 2);
    // std::uniform_real_distribution<Scalar> posY(0, map_height_);
    std::uniform_real_distribution<Scalar> size(0.1, 1.0);
    std::uniform_real_distribution<Scalar> angle(0, M_PI);

    // 添加随机矩形
    for (int i = 0; i < num_rectangles_; ++i) {
      bool is_placed = false;
      for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
        Scalar cx = posX(gen);
        Scalar cy = posY(gen);
        Scalar width = size(gen);
        Scalar height = size(gen);
        Scalar yaw = angle(gen);
        auto rect =
          std::make_shared<Rectangle>(Vector<2>{cx, cy}, width, height, yaw);
        bool collision = false;
        for (const auto& existing : obstacles_) {
          if (rect->getBBox().intersects(existing->getBBox())) {
            collision = true;
            break;
          }
        }
        if (!collision) {
          obstacles_.push_back(rect);
          // treeBuilt_ = false;  // 需要重建树
          is_placed = true;
          break;
        }
      }
      if (!is_placed) std::cerr << "无法放置长方体 " << i << std::endl;
    }

    // 添加随机椭圆
    for (int i = 0; i < num_ellipses_; ++i) {
      bool is_placed = false;
      for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
        Scalar cx = posX(gen);
        Scalar cy = posY(gen);
        Scalar a = size(gen);
        Scalar b = size(gen);
        if (b > a) std::swap(a, b);  // 确保a是长轴
        Scalar yaw = angle(gen);
        auto ellipse = std::make_shared<Ellipse>(Vector<2>{cx, cy}, a, b, yaw);
        bool collision = false;
        for (const auto& existing : obstacles_) {
          if (ellipse->getBBox().intersects(existing->getBBox())) {
            collision = true;
            break;
          }
        }
        if (!collision) {
          obstacles_.push_back(ellipse);
          // treeBuilt_ = false;  // 需要重建树
          is_placed = true;
          break;
        }
      }
      if (!is_placed) std::cerr << "无法放置圆柱体 " << i << std::endl;
    }
  }

  void update(const ros::TimerEvent&) {
    // robotYaw_ += 0.002;

    step_ += 1;
    // if ((step_ % 4000) < 500) {
    //   robotPos_.x() += 0.01;
    // } else if ((step_ % 4000) < 1500) {
    //   robotPos_.y() += 0.01;
    // } else if ((step_ % 4000) < 2500) {
    //   robotPos_.x() -= 0.01;
    // } else if ((step_ % 4000) < 3500) {
    //   robotPos_.y() -= 0.01;
    // } else {
    //   robotPos_.x() += 0.01;
    // }

    // 发布TF (假设机器人在地图中心)
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(robotPos_.x(), robotPos_.y(), 0.0));
    // transform.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
    transform.setRotation(
      tf::Quaternion(0, 0, sin(robotYaw_ / 2), cos(robotYaw_ / 2)));
    // transform.setRotation(tf::Quaternion(0, 0, 0, 1));
    br.sendTransform(
      tf::StampedTransform(transform, ros::Time::now(), "map", "base_laser"));

    lidar_timer_.tic();
    // 模拟激光雷达数据
    // simulateLidar();

    sensor_msgs::LaserScan scan;
    scan.header.stamp = ros::Time::now();
    scan.header.frame_id = "base_laser";
    scan.angle_min = -lidar_fov_ / 2;
    scan.angle_max = lidar_fov_ / 2;
    scan.angle_increment = lidar_fov_ / (lidar_num_rays_ - 1);
    scan.time_increment = 0;
    scan.scan_time = 0.1;
    scan.range_min = 0.1;
    scan.range_max = lidar_max_range_ + 5.0;
    bool is_collision =
      lidar2d_sim_.simulateLidar(robotPos_, robotYaw_, scan.ranges);
    scan_pub_.publish(scan);
    lidar_timer_.toc();


    // 可视化障碍物
    vis_timer_.tic();
    visualizeObstacles();
    vis_timer_.toc();
    std::cout << lidar_timer_ << std::endl;
    std::cout << vis_timer_ << std::endl;
  }

  void simulateLidar() {
    sensor_msgs::LaserScan scan;
    scan.header.stamp = ros::Time::now();
    scan.header.frame_id = "base_laser";
    scan.angle_min = -lidar_fov_ / 2;
    scan.angle_max = lidar_fov_ / 2;
    scan.angle_increment = lidar_fov_ / (lidar_num_rays_ - 1);
    scan.time_increment = 0;
    scan.scan_time = 0.1;
    scan.range_min = 0.1;
    scan.range_max = lidar_max_range_ + 5.0;
    scan.ranges.resize(lidar_num_rays_, lidar_max_range_);

    /// ----------- Method 1 --------------
    // // 并行处理射线
    // std::vector<std::future<void>> futures;
    // futures.reserve(lidar_num_rays_);

    // for (int i = 0; i < lidar_num_rays_; ++i) {
    //   futures.push_back(std::async(std::launch::async, [&, i] {
    //     Scalar angle = robotYaw_ + scan.angle_min + i * scan.angle_increment;
    //     Vector<2> dir(std::cos(angle), std::sin(angle));

    //     Scalar t = lidar_max_range_;
    //     for (const auto& obs : obstacles_) {
    //     Scalar t_obs;
    //     if (obs->rayIntersect(robotPos_, dir, lidar_max_range_, t_obs)) {
    //       if (t_obs < t) t = t_obs;
    //     }
    //     }
    //     scan.ranges[i] = t;

    //   }));
    // }

    // // 等待所有射线完成
    // for (auto& f : futures) f.wait();

    /// ----------- Method 2 --------------
    for (int i = 0; i < lidar_num_rays_; ++i) {
      Scalar angle = robotYaw_ + scan.angle_min + i * scan.angle_increment;
      Vector<2> dir(std::cos(angle), std::sin(angle));

      Scalar t = lidar_max_range_;
      for (const auto& obs : obstacles_) {
        Scalar t_obs;
        if (obs->rayIntersect(robotPos_, dir, lidar_max_range_, t_obs)) {
          if (t_obs < t) t = t_obs;
        }
      }
      scan.ranges[i] = t;
    }

    /// ----------- Method 3 --------------
    // 根据CPU核心数确定线程数
    // const size_t num_threads = std::thread::hardware_concurrency();
    // std::cout << "[Thread]: " << num_threads << std::endl;
    // std::vector<std::thread> workers;

    // // 每个线程处理的射线数
    // const int rays_per_thread =
    //   (lidar_num_rays_ + num_threads - 1) / num_threads;

    // for (size_t t = 0; t < num_threads; ++t) {
    //   workers.emplace_back([&, t] {
    //     const int start = t * rays_per_thread;
    //     const int end = std::min(start + rays_per_thread, lidar_num_rays_);

    //     for (size_t i = start; i < end; ++i) {
    //       Scalar angle = robotYaw_ + scan.angle_min + i *
    //       scan.angle_increment; Vector<2> dir(std::cos(angle),
    //       std::sin(angle));

    //       Scalar t = lidar_max_range_;
    //       for (const auto& obs : obstacles_) {
    //         Scalar t_obs;
    //         if (obs->rayIntersect(robotPos_, dir, lidar_max_range_,
    //         t_obs)) {
    //           if (t_obs < t) t = t_obs;
    //         }
    //       }

    //       scan.ranges[i] = t;
    //     }
    //   });
    // }
    // for (auto& worker : workers) worker.join();

    scan_pub_.publish(scan);
  }

  void visualizeObstacles() {
    visualization_msgs::MarkerArray markers;
    int id = 0;
    Scalar angle_min = -lidar_fov_ / 2;
    Scalar angle_increment = lidar_fov_ / (lidar_num_rays_ - 1);
    visualization_msgs::Marker lidar_lines;
    lidar_lines.header.frame_id = "map";
    lidar_lines.header.stamp = ros::Time::now();
    lidar_lines.ns = "lidar_lines";
    lidar_lines.id = id++;
    lidar_lines.action = visualization_msgs::Marker::ADD;
    lidar_lines.type = visualization_msgs::Marker::LINE_LIST;
    lidar_lines.pose.orientation.w = 1.0;
    lidar_lines.scale.x = 0.02;
    lidar_lines.scale.y = 0.02;
    lidar_lines.scale.z = 0.02;
    lidar_lines.color.a = 0.8;
    lidar_lines.color.r = 0.0;
    lidar_lines.color.g = 0.0;
    lidar_lines.color.b = 0.0;
    for (int i = 0; i < lidar_num_rays_; ++i) {
      Scalar angle = robotYaw_ + angle_min + i * angle_increment;
      geometry_msgs::Point p_msg;
      p_msg.x = robotPos_.x();
      p_msg.y = robotPos_.y();
      p_msg.z = 0.0;
      lidar_lines.points.push_back(p_msg);
      p_msg.x = robotPos_.x() + lidar_max_range_ * cos(angle);
      p_msg.y = robotPos_.y() + lidar_max_range_ * sin(angle);
      p_msg.z = 0.0;
      lidar_lines.points.push_back(p_msg);
    }
    markers.markers.push_back(lidar_lines);

    for (const auto& obs : obstacles_) {
      visualization_msgs::Marker marker;
      marker.header.frame_id = "map";
      marker.header.stamp = ros::Time::now();
      marker.ns = "obstacles";
      marker.id = id++;
      marker.action = visualization_msgs::Marker::ADD;

      if (auto rect = dynamic_cast<Rectangle*>(obs.get())) {
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
      } else if (auto ellipse = dynamic_cast<Ellipse*>(obs.get())) {
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

      marker.lifetime = ros::Duration(0.5);
      markers.markers.push_back(marker);
    }

    obstacles_pub_.publish(markers);
  }

 private:
  ros::NodeHandle nh_;
  ros::Publisher scan_pub_;
  ros::Publisher obstacles_pub_;
  ros::Timer timer_;
  Timer lidar_timer_{"Lidar"};
  Timer vis_timer_{"Vis"};
  Timer map_timer_{"Map"};
  Logger logger_{"lidar2d_test"};

  Lidar2D lidar2d_sim_{40, 40, 2 * M_PI, 512, 5.0, 0.4};


  std::vector<std::shared_ptr<Obstacle>> obstacles_;

  // 参数
  int step_;
  Vector<2> robotPos_;
  Scalar robotYaw_;
  Scalar map_width_;
  Scalar map_height_;
  int num_rectangles_;
  int num_ellipses_;
  Scalar lidar_fov_;
  int lidar_num_rays_;
  Scalar lidar_max_range_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "obstacle_lidar_sim");
  ObstacleLidarSim sim;
  // while (!ros::isShuttingDown()) {
  //   sim.robotMove();
  //   ros::spinOnce();
  // }

  ros::spin();
  return 0;
}
