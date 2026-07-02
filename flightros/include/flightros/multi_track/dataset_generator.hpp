/*** 
 * @Author: Lac_Creeper
 * @Date: 2026-03-04 09:48:09 +0800
 * @LastEditTime: 2026-04-06 04:41:56 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /src/flightmare/flightros/include/flightros/multi_track/dataset_generator.hpp
 */
#pragma once

#include <memory>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <vector>

// flightlib
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/objects/target.hpp"
#include "flightlib/sensors/detect.hpp"
#include "flightlib/sensors/lidar2D.hpp"
#include "flightlib/json/json.hpp"

using json = nlohmann::json;
using namespace flightlib;

namespace flightros {


struct SampleData {
  Vector<3> target_pose;
  Vector<3> uav_a_pose;
  Scalar uav_a_yaw;
  Vector<3> uav_b_pose;
  Scalar uav_b_yaw;

  Vector<3> pose_b_a_pos;
  Vector<4> pose_b_a_quat;
  BBox bbox_b;
  Vector<2> motion_b;
  Vector<3> pose_a_delta_pos;
  Vector<4> pose_a_delta_quat;
  BBox bbox_a_last;
  Vector<2> motion_a_last;

  BBox bbox_a;
  Vector<2> motion_a;

  int episode_id;
  int chunk_id;
  int frame_id;
  double timestamp;
  TrajectoryType traj_type;
  json to_json() const {
    json j;
    j["target_pose"] = {target_pose.x(), target_pose.y(), target_pose.z()};
    j["uav_a_pose"] = {uav_a_pose.x(), uav_a_pose.y(), uav_a_pose.z()};
    j["uav_a_yaw"] = uav_a_yaw;
    j["uav_b_pose"] = {uav_b_pose.x(), uav_b_pose.y(), uav_b_pose.z()};
    j["uav_b_yaw"] = uav_b_yaw;
                       
    j["pose_b_a"] = {pose_b_a_pos.x(),  pose_b_a_pos.y(),  pose_b_a_pos.z(),
                     pose_b_a_quat.w(), pose_b_a_quat.x(), pose_b_a_quat.y(),
                     pose_b_a_quat.z()};
    j["bbox_b"] = {bbox_b.u_min / 960.0 - 0.5, bbox_b.v_min / 540.0 - 0.5,
                   bbox_b.u_max / 960.0 - 0.5, bbox_b.v_max / 540.0 - 0.5};
    j["motion_b"] = {motion_b.x(), motion_b.y()};
    j["pose_a_delta"] = {pose_a_delta_pos.x(),  pose_a_delta_pos.y(),
                         pose_a_delta_pos.z(),  pose_a_delta_quat.w(),
                         pose_a_delta_quat.x(), pose_a_delta_quat.y(),
                         pose_a_delta_quat.z()};
    j["bbox_a_last"] = {
      bbox_a_last.u_min / 960.0 - 0.5, bbox_a_last.v_min / 540.0 - 0.5,
      bbox_a_last.u_max / 960.0 - 0.5, bbox_a_last.v_max / 540.0 - 0.5};
    j["motion_a_last"] = {motion_a_last.x(), motion_a_last.y()};
    j["bbox_a"] = {bbox_a.u_min / 960.0 - 0.5, bbox_a.v_min / 540.0 - 0.5,
                   bbox_a.u_max / 960.0 - 0.5, bbox_a.v_max / 540.0 - 0.5};
    j["motion_a"] = {motion_a.x(), motion_a.y()};
    j["episode_id"] = episode_id;
    j["chunk_id"] = chunk_id;
    j["frame_id"] = frame_id;
    j["timestamp"] = timestamp;
    j["traj_type"] = static_cast<int>(traj_type);
    return j;
  }

  // json to_json() const {
  //   json j;
  //   j["pose_b_a"] = {pose_b_a_pos.x(),  pose_b_a_pos.y(),  pose_b_a_pos.z(),
  //                    pose_b_a_quat.w(), pose_b_a_quat.x(), pose_b_a_quat.y(),
  //                    pose_b_a_quat.z()};
  //   j["bbox_b"] = {bbox_b.u_min / 960.0 - 0.5, bbox_b.u_max / 960.0 - 0.5,
  //                  bbox_b.v_min / 540.0 - 0.5, bbox_b.v_max / 540.0 - 0.5};
  //   j["motion_b"] = {motion_b.x(), motion_b.y()};
  //   j["pose_a_delta"] = {pose_a_delta_pos.x(),  pose_a_delta_pos.y(),
  //                        pose_a_delta_pos.z(),  pose_a_delta_quat.w(),
  //                        pose_a_delta_quat.x(), pose_a_delta_quat.y(),
  //                        pose_a_delta_quat.z()};
  //   j["bbox_a_last"] = {
  //     bbox_a_last.u_min / 960.0 - 0.5, bbox_a_last.u_max / 960.0 - 0.5,
  //     bbox_a_last.v_min / 540.0 - 0.5, bbox_a_last.v_max / 540.0 - 0.5};
  //   j["bbox_a"] = {bbox_a.u_min / 960.0 - 0.5, bbox_a.u_max / 960.0 - 0.5,
  //                  bbox_a.v_min / 540.0 - 0.5, bbox_a.v_max / 540.0 - 0.5};
  //   j["motion_a"] = {motion_a.x(), motion_a.y()};
  //   j["sequence_id"] = sequence_id;
  //   j["frame_id"] = frame_id;
  //   j["timestamp"] = timestamp;
  //   j["traj_type"] = static_cast<int>(traj_type);
  //   return j;
  // }
  // json to_json() const {
  //   json j;
  //   j["pose_b_a"] = {pose_b_a_pos.x(),  pose_b_a_pos.y(),  pose_b_a_pos.z(),
  //                    pose_b_a_quat.w()};
  //   j["bbox_b"] = {(bbox_b.u_min + bbox_b.u_max) / 2.0 / 960.0 - 0.5, 
  //                  (bbox_b.v_min + bbox_b.v_max) / 2.0 / 540.0 - 0.5, 
  //                  (bbox_b.v_max - bbox_b.v_min) / 540.0};
  //   j["motion_b"] = {motion_b.x(), motion_b.y()};
  //   j["pose_a_delta"] = {pose_a_delta_pos.x(),  pose_a_delta_pos.y(),
  //                        pose_a_delta_pos.z(),  pose_a_delta_quat.w()};
  //   j["bbox_a_last"] = {
  //     (bbox_a_last.u_min + bbox_a_last.u_max) / 2.0 / 960.0 - 0.5, 
  //     (bbox_a_last.v_min + bbox_a_last.v_max) / 2.0 / 540.0 - 0.5, 
  //     (bbox_a_last.v_max - bbox_a_last.v_min) / 540.0};
  //   j["bbox_a"] = {
  //     (bbox_a.u_min + bbox_a.u_max) / 2.0 / 960.0 - 0.5, 
  //     (bbox_a.v_min + bbox_a.v_max) / 2.0 / 540.0 - 0.5, 
  //     (bbox_a.v_max - bbox_a.v_min) / 540.0};
  //   j["motion_a"] = {motion_a.x(), motion_a.y()};
  //   j["sequence_id"] = sequence_id;
  //   j["frame_id"] = frame_id;
  //   j["timestamp"] = timestamp;
  //   j["traj_type"] = static_cast<int>(traj_type);
  //   return j;
  // }
};




class DatasetGenerator {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  DatasetGenerator();
  ~DatasetGenerator();

  void generate();
  void save_chunk(const std::vector<SampleData>& chunk,
                  const std::string& filename);
  void random_initial_states();
  void update_uavs();

  std::vector<SampleData> generate_sequence(int seq_id);

  TrajectoryType random_trajectory_type();

 private:
  DetectSim detect_;
  // Drones
  Vector<4> uavA_state_, uavB_state_;
  Vector<4> uavA_last_, uavB_last_;
  BBox bboxA_, bboxB_;
  BBox bboxA_l_, bboxB_l_;
  Vector<4> uavA_vel_, uavB_vel_;


  // Target
  Target target_;
  Vector<3> target_xyY_;
  Scalar sim_time_;
  const Scalar dt_ = 0.05f;

  std::ofstream train_list_, val_list_, test_list_;

  Logger logger_{"DataSet"};

  std::random_device rd_;
  std::mt19937 rng_{rd_()};
  std::uniform_real_distribution<Scalar> uniform_dist_{-1.0, 1.0};
  std::uniform_real_distribution<Scalar> uniform_01_dist_{0.0, 1.0};
};

}   // namespace flightros