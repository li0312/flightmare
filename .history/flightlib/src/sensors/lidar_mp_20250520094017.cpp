/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-05-20 09:40:17 +0800
 * @LastEditTime: 2025-05-20 09:40:18 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /flightmare/flightlib/src/sensors/lidar copy.cpp
 */
/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-05-16 16:52:43 +0800
 * @LastEditTime: 2025-05-18 22:14:20 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /src/flightmare/flightlib/src/sensors/lidar.cpp
 */
#include "flightlib/sensors/lidar.hpp"

namespace flightlib {

Lidar::Lidar()
  : pc_resolution_(0.1),
    sensing_horizon_(8.0),
    hrz_laser_line_num_(512),
    vtc_laser_line_num_(16),
    hrz_laser_range_rad_(2 * M_PI),
    vtc_laser_range_rad_(30.0 / 180.0 * M_PI),
    scan_height_min_(0.0),
    scan_height_max_(1.0),
    scan_range_min_(0.1),
    scan_range_max_(5.0),
    collision_range_(0.4),
    is_visual_(true),
    is_check_collision_(true),
    use_resolution_filter_(true) {
  vtc_resolution_rad_ =
    vtc_laser_range_rad_ / (Scalar)(vtc_laser_line_num_ - 1);
  half_vtc_resolution_and_half_range_ =
    (vtc_laser_range_rad_ + vtc_resolution_rad_) / 2.0;
  hrz_resolution_rad_ = hrz_laser_range_rad_ / (Scalar)hrz_laser_line_num_;
  half_hrz_range_ = hrz_laser_range_rad_ / 2.0;

  idx_map_ =
    Eigen::MatrixXi::Constant(hrz_laser_line_num_, vtc_laser_line_num_, -1);
  dis_map_ =
    Eigen::MatrixXf::Constant(hrz_laser_line_num_, vtc_laser_line_num_, 9999.0);

  // cloud_all_map_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  // local_map_.reset(new pcl::PointCloud<pcl::PointXYZ>);

  if (is_visual_) {
    viewer_.reset(new pcl::visualization::PCLVisualizer("Lidar Viewwe"));
    viewer_->setBackgroundColor(0, 0, 0);
    viewer_->addCoordinateSystem(1.0);

    // viewer_->initCameraParameters();
    // viewer_->setCameraPosition(50, 50, 50, 0, 0, 0, 0, 0, 1);
  }

  has_global_map_ = false;
  lodaClobalPCFromPLY();
}

Lidar::~Lidar() {}

void Lidar::lodaClobalPCFromPLY() {
  if (has_global_map_) 
    return;

  logger_.info("Global Pointcloud received..");

  pcl::PointCloud<pcl::PointXYZ> cloud_input;
  if (pcl::io::loadPLYFile<pcl::PointXYZ>(
        "/home/lac/my_study/flightmare_pro/pointcloud_data/random_map_0.ply",
        cloud_input) == -1) {
    logger_.error("Couldn't read the PLY file.\n");
    return;
  }
  logger_.debug("Load PLY File success..");
  logger_.debug("Load PointCloud num: %d", cloud_input.points.size());

  voxel_sampler_.setLeafSize(pc_resolution_, pc_resolution_, pc_resolution_);
  voxel_sampler_.setInputCloud(cloud_input.makeShared());
  voxel_sampler_.filter(cloud_all_map_);

  kdtreeLocalMap_.setInputCloud(cloud_all_map_.makeShared());

  if (is_visual_) {
    if (!viewer_->updatePointCloud(cloud_all_map_.makeShared(), "global map")) {
      // pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> global_color(cloud_all_map_, "z");
      viewer_->addPointCloud<pcl::PointXYZ>(
          cloud_all_map_.makeShared(), "global map");
      viewer_->setPointCloudRenderingProperties(
          pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "global map");
    }
  }

  has_global_map_ = true;
}

bool Lidar::renderPointCloud(const QuadState& state) {
  if (!has_global_map_) {
    logger_.error("No global map received yet.");
    return false;
  }
  // Timer timer{"Timer", "Printing"};
  // Timer timer_1{"Timer_1", "Printing"};
  timer.tic();
  timer_1.tic();

  Matrix<3, 3> rot(state.q());
  Vector<3> laser_t(state.p);

  pointIdxRadiusSearch_.clear();
  pointRadiusSquaredDistance_.clear();
  pcl::PointXYZ searchPoint(state.p(0), state.p(1), state.p(2));
  kdtreeLocalMap_.radiusSearch(searchPoint, sensing_horizon_,
                               pointIdxRadiusSearch_,
                               pointRadiusSquaredDistance_);

  idx_map_.setConstant(-1);
  dis_map_.setConstant(9999.0);

  pcl::PointXYZ pt;
  for (size_t i = 0; i < pointIdxRadiusSearch_.size(); ++i) {
    pt = cloud_all_map_.points[pointIdxRadiusSearch_[i]];

    Vector2i idx;
    bool in_range = pt2LaserIdx(idx, Vector<3>(pt.x, pt.y, pt.z), laser_t, rot);
    if (!in_range) continue;
    Vector<3> pt_vec(pt.x - state.p(0), 
                     pt.y - state.p(1), 
                     pt.z - state.p(2));
    Scalar dis_curr_pt = pt_vec.norm();

    if (use_resolution_filter_) {
      Scalar vtc_rad =
        idx[1] * vtc_resolution_rad_ - vtc_laser_range_rad_ / 2.0;
      Scalar dis_to_z_axis = dis_curr_pt * cos(vtc_rad);
      Scalar mesh_len_hrz = dis_to_z_axis * hrz_resolution_rad_;
      Scalar mesh_len_vtc = dis_to_z_axis * vtc_resolution_rad_;
      int hrz_occ_grid_num = std::min((int)floor(pc_resolution_ / mesh_len_hrz),
                                      hrz_laser_line_num_);
      int vtc_occ_grid_num = std::min((int)floor(pc_resolution_ / mesh_len_vtc),
                                      vtc_laser_line_num_);
      int tmp1 = hrz_occ_grid_num, tmp2 = vtc_occ_grid_num;
      for (int d_hrz_idx = -tmp1; d_hrz_idx <= tmp1; ++d_hrz_idx)
        for (int d_vtc_idx = -tmp2; d_vtc_idx <= tmp2; ++d_vtc_idx) {
          int hrz_idx =
            (idx[0] + d_hrz_idx + hrz_laser_line_num_) % hrz_laser_line_num_;
          int vtc_idx = idx[1] + d_vtc_idx;
          if (vtc_idx >= vtc_laser_line_num_) continue;
          if (vtc_idx < 0) continue;
          if (dis_curr_pt < dis_map_(hrz_idx, vtc_idx)) {
            idx_map_(hrz_idx, vtc_idx) = i;
            dis_map_(hrz_idx, vtc_idx) = dis_curr_pt;
          }
        }
    } else {
      if (dis_curr_pt < dis_map_(idx[0], idx[1])) {
        idx_map_(idx[0], idx[1]) = i;
        dis_map_(idx[0], idx[1]) = dis_curr_pt;
      }
    }
  }

  local_map_.points.clear();
  local_vis_.points.clear();
  for (int x = 0; x < hrz_laser_line_num_; ++x)
    for (int y = 0; y < vtc_laser_line_num_; ++y) {
      /* use map cloud pts as laser pts
      if (idx_map_(x, y) == -1)
        continue;
      pt = cloud_all_map_.points[pointIdxRadiusSearch_[idx_map_(x, y)]];
      local_map_.points.push_back(pt);
      */

      // use laser line pts
      Vector<3> p;
      if (idx_map_(x, y) != -1) {
        idx2Pt(x, y, dis_map_(x, y), p);
        local_map_.points.emplace_back(p[0], p[1], p[2]);
        pt = cloud_all_map_.points[pointIdxRadiusSearch_[idx_map_(x, y)]];
        local_vis_.points.push_back(pt);
      }
    }

  local_map_.width = local_map_.points.size();
  local_map_.height = 1;
  local_map_.is_dense = true;

  local_vis_.width = local_vis_.points.size();
  local_vis_.height = 1;
  local_vis_.is_dense = true;

  logger_.debug("PointCloud num: %d", local_map_.width);

  if (is_check_collision_) {
    collision_pointIdxRadiusSearch_.clear();
    collision_pointRadiusSquaredDistance_.clear();
    if (kdtreeLocalMap_.radiusSearch(
          searchPoint, collision_range_, collision_pointIdxRadiusSearch_,
          collision_pointRadiusSquaredDistance_) > 0) {
      logger_.warn("ENVIRONMENT COLLISION DETECTED!!");
    }
  }

  timer.toc();
  Scalar timing_1 = timer.last();
  logger_.debug("timing_1: %.2f ms", 1000 * timing_1);

  if (is_visual_) {
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ>
      local_color(local_vis_.makeShared(), "z");
    if (!viewer_->updatePointCloud(local_vis_.makeShared(),
                                   local_color, "local pc")) {
      viewer_->addPointCloud<pcl::PointXYZ>(local_vis_.makeShared(),
                                            local_color, "local pc");
      viewer_->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "local pc");
    }
  }
  timer_1.toc();
  Scalar timing_2 = timer_1.last();
  logger_.debug("render+visual: %.2f ms", 1000 * timing_2);


  return true;
}

bool Lidar::renderLaserScan(const QuadState& state) {
  if (!renderPointCloud(state)) {
    logger_.error("No pointcloud!!");
    return false;
  }

  laserscan_.clear();
  uint32_t ranges_size = std::ceil(2.0 * half_hrz_range_ / hrz_resolution_rad_);
  laserscan_.assign(ranges_size, scan_range_max_);
  
  // Iterate through pointcloud
  for (const auto& point : local_map_) {
    if (std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z))
      continue;
    if (point.z > scan_height_max_ || point.z < scan_height_min_)
      continue;
    Scalar range = hypot(point.x, point.y);
    if (range < scan_range_min_ || range > scan_range_max_)
      continue;
    Scalar angle = atan2(point.y, point.x);
    // overwrite range at laserscan ray if new range is smaller
    int index = (angle + half_hrz_range_) / hrz_resolution_rad_;
    if (range < laserscan_[index]) {
      laserscan_[index] = range;
    }
  }

  return true;

}

pcl::visualization::PCLVisualizer::Ptr Lidar::getViewer() const {
  return viewer_;
}


} // namespace flightlib
