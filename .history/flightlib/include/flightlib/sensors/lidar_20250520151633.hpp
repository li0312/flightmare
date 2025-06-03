/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-05-16 15:13:22 +0800
 * @LastEditTime: 2025-05-20 15:16:14 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /flightmare/flightlib/include/flightlib/sensors/lidar.hpp
 */
#pragma once

#include <vector>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "flightlib/common/logger.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/common/timer.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/sensors/sensor_base.hpp"

namespace flightlib {

struct ScanParams {
  
  Scalar angle_min = -M_PI;
  Scalar angle_max = M_PI;
  Scalar angle_increment = 2*M_PI/512.0f;
  Scalar range_min = 0.1;
  Scalar range_max = 5.0;

};
  
class Lidar : SensorBase {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Lidar();
    ~Lidar();

    // public set functions
    bool setRelPose(const Ref<Vector<3>> B_r_BL, const Ref<Matrix<3, 3>> R_BL);

    // public get functions
    bool renderPointCloud(const QuadState& state);
    bool renderLaserScan(const QuadState &state);

    void lodaClobalPCFromPLY();

    const pcl::PointCloud<pcl::PointXYZ>::Ptr& getlocalCloud() const;
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& getGlobalMap() const;
    const std::vector<Scalar>& getScan() const;

    const ScanParams& getScanParams() const;

    /// some inline math functions
    /**
     * @brief
     * Given a line and a plane, calculating for the intersection and the
     * distance. line_dir is required to be a normalized vector.
     * @param intersection the intersecting point.
     * @param line_p A point in the line.
     * @param line_dir The line direction vector.
     * @param plane_p A point in the plane.
     * @param plane_normal The plane normal vector.
     * @return double
     * The distance between the query point and the intersection.
     * A negative value means the line direction vector points away from the
     * plane.
     */
    inline Scalar lineIntersectPlane(Vector<3> &intersection,
                                     const Vector<3> &line_p,
                                     const Vector<3> &line_dir,
                                     const Vector<3> &plane_p,
                                     const Vector<3> &plane_normal) {
      Scalar d =
        (plane_p - line_p).dot(plane_normal) / line_dir.dot(plane_normal);
      intersection = line_p + d * line_dir;
      return d;
    }
    /**
     * @brief
     * filter the points not in range
     * @param idx
     * @param pt
     * @param laser_t
     * @param laser_R must be normalized in each column
     * @return true
     * @return false
     */
    inline bool pt2LaserIdx(Vector2i &idx, const Vector<3> &pt,
                            const Vector<3> &laser_t,
                            const Matrix<3, 3> &laser_R) {
      Vector<3> inter_p;
      Scalar dis_pt_to_laser_plane = lineIntersectPlane(
        inter_p, pt, laser_R.col(2), laser_t, laser_R.col(2));
      Scalar dis_laser_to_inter_p = (inter_p - laser_t).norm();
      Scalar vtc_rad =
        std::atan2((pt - laser_t).dot(laser_R.col(2)), dis_laser_to_inter_p);
      if (std::fabs(vtc_rad) >= half_vtc_resolution_and_half_range_)
        return false;

      Scalar x_in_roll_pitch_plane = (inter_p - laser_t).dot(laser_R.col(0));
      Scalar y_in_roll_pitch_plane = (inter_p - laser_t).dot(laser_R.col(1));
      Scalar hrz_rad = std::atan2(y_in_roll_pitch_plane, x_in_roll_pitch_plane);
      if (std::fabs(hrz_rad) >= half_hrz_range_) return false;

      vtc_rad += half_vtc_resolution_and_half_range_;
      int vtc_idx = std::floor(vtc_rad / vtc_resolution_rad_);
      if (vtc_idx >= vtc_laser_line_num_) vtc_idx = 0;

      hrz_rad += M_PI + hrz_resolution_rad_ / 2.0;
      int hrz_idx = std::floor(hrz_rad / hrz_resolution_rad_);
      if (hrz_idx >= hrz_laser_line_num_) hrz_idx = 0;

      idx << hrz_idx, vtc_idx;
      return true;
    }

    /**
     * @brief
     *
     * @param x
     * @param y
     * @param dis
     * @param pt in laser coordinate.
     */
    inline void idx2Pt(int x, int y, Scalar dis, Vector<3> &pt) {
      Scalar vtc_rad = y * vtc_resolution_rad_ - vtc_laser_range_rad_ / 2.0;
      Scalar hrz_rad = x * hrz_resolution_rad_ - M_PI;
      pt[2] = sin(vtc_rad) * dis;  // z_in_laser_coor
      Scalar xy_square_in_laser_coor = cos(vtc_rad) * dis;
      pt[0] = cos(hrz_rad) * xy_square_in_laser_coor;  // x_in_laser_coor
      pt[1] = sin(hrz_rad) * xy_square_in_laser_coor;  // y_in_laser_coor
    }

   private:
    Logger logger_{"Lidar"};
    Timer timer{"Cloud", "Printing"};
    Timer timer_1{"Scan", "Printing"};

    bool is_check_collision_;
    bool use_resolution_filter_;

    // lidar parameters
    Scalar pc_resolution_, sensing_horizon_; // in meters
    int hrz_laser_line_num_, vtc_laser_line_num_;
    Scalar hrz_laser_range_rad_, vtc_laser_range_rad_;
    // laserscan parameters
    Scalar scan_height_min_, scan_height_max_;
    Scalar scan_range_min_, scan_range_max_;
    ScanParams scan_params_;
    // collision check parameters
    Scalar collision_range_;

    // REVIEW: PointT & PointT::Ptr which is better
    // pcl::PointCloud<pcl::PointXYZ> cloud_all_map_, local_map_;
    // pcl::PointCloud<pcl::PointXYZ> local_vis_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_all_map_, local_map_;
    
    pcl::VoxelGrid<pcl::PointXYZ> voxel_sampler_;
    std::vector<Scalar> laserscan_;

    Eigen::MatrixXi idx_map_;
    Eigen::MatrixXf dis_map_;

    pcl::search::KdTree<pcl::PointXYZ> kdtreeLocalMap_;
    std::vector<int> pointIdxRadiusSearch_;
    std::vector<Scalar> pointRadiusSquaredDistance_;

    // For collision check
    std::vector<int> collision_pointIdxRadiusSearch_;
    std::vector<Scalar> collision_pointRadiusSquaredDistance_;

    bool has_global_map_;
    Scalar half_vtc_resolution_and_half_range_;
    Scalar hrz_resolution_rad_, vtc_resolution_rad_;
    Scalar half_hrz_range_;
    
    // lidar relative
    Vector<3> B_r_BL;
    Matrix<4, 4> T_BL;

    




};


} // namespace flightlib
