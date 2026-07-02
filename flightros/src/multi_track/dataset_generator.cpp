/*** 
 * @Author: Lac_Creeper
 * @Date: 2026-03-04 09:48:32 +0800
 * @LastEditTime: 2026-04-06 03:01:42 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /src/flightmare/flightros/src/multi_track/dataset_generator.cpp
 */
#include "flightros/multi_track/dataset_generator.hpp"


using namespace flightlib;
namespace fs = boost::filesystem;
// namespace fs = std::filesystem;


namespace flightros {

DatasetGenerator::DatasetGenerator() {
  fs::create_directories(
    "/home/lac/fm_test/src/flightmare/flightrl/collaborative_detect/dataset3/"
    "samples");
  // fs::create_directories(
  //   "/home/lac/fm_test/src/flightmare/flightrl/collaborative_detect/dataset/"
  //   "sequences");

  train_list_.open(
    "/home/lac/fm_test/src/flightmare/flightrl/collaborative_detect/dataset3/"
    "train_files.txt");
  val_list_.open(
    "/home/lac/fm_test/src/flightmare/flightrl/collaborative_detect/dataset3/"
    "val_files.txt");
  test_list_.open(
    "/home/lac/fm_test/src/flightmare/flightrl/collaborative_detect/dataset3/"
    "test_files.txt");

  logger_.info("Dataset directory initialized..");
  

}

DatasetGenerator::~DatasetGenerator() {
  if (train_list_.is_open()) train_list_.close();
  if (val_list_.is_open()) val_list_.close();
  if (test_list_.is_open()) test_list_.close();
}


void DatasetGenerator::generate() {
  int num_episodes = 100;        // 总回合数 (例如100回合)
  int steps_per_episode = 2000;  // 每回合 2000 步
  int chunk_size = 40;           // 每 40 步保存一个文件

  int total_chunks_generated = 0;

  // 计算用于划分数据集的阈值 (按生成的 chunk 数量划分)
  int total_chunks_expected = num_episodes * (steps_per_episode / chunk_size);
  int train_split = total_chunks_expected * 0.8;
  int val_split = total_chunks_expected * 0.9;

  for (int ep_id = 0; ep_id < num_episodes; ++ep_id) {
    // 重置环境与状态
    random_initial_states();

    std::vector<SampleData> chunk_buffer;
    chunk_buffer.reserve(chunk_size);
    int chunk_id = 0;

    for (int step = 0; step < steps_per_episode; ++step) {
      sim_time_ = step * dt_;
      target_.update(sim_time_);
      target_xyY_ = target_.getPose();

      // 执行动力学更新与感知
      update_uavs();

      // 构造样本数据
      SampleData sample;
      sample.episode_id = ep_id;
      sample.chunk_id = chunk_id;
      sample.frame_id = step;
      sample.timestamp = sim_time_;
      sample.traj_type = target_.getTrajectoryType();
      sample.target_pose = target_xyY_;
      sample.uav_a_pose = uavA_state_.head<3>();
      sample.uav_a_yaw = uavA_state_.w();
      sample.uav_b_pose = uavB_state_.head<3>();
      sample.uav_b_yaw = uavB_state_.w();

      // -- 记录相对位姿 (保留您原有的坐标系转换逻辑) --
      sample.pose_b_a_pos = uavB_state_.head<3>() - uavA_state_.head<3>();
      sample.pose_b_a_quat = Vector<4>(
        0.0f, 0.0f, std::sin((uavB_state_.w() - uavA_state_.w()) / 2.0f),
        std::cos((uavB_state_.w() - uavA_state_.w()) / 2.0f));

      sample.pose_a_delta_pos = uavA_last_.head<3>() - uavA_state_.head<3>();
      sample.pose_a_delta_quat = Vector<4>(
        0.0f, 0.0f, std::sin((uavA_last_.w() - uavA_state_.w()) / 2.0f),
        std::cos((uavA_last_.w() - uavA_state_.w()) / 2.0f));

      sample.bbox_b = bboxB_;
      sample.bbox_a_last = bboxA_l_;
      sample.motion_b = Vector<2>{std::cos(target_xyY_.z() - uavB_state_.w()),
                                  std::sin(target_xyY_.z() - uavB_state_.w())};
      sample.motion_a_last = 
          Vector<2>{std::cos(target_xyY_.z() - uavA_last_.w()),
                    std::sin(target_xyY_.z() - uavA_last_.w())};

      sample.bbox_a = bboxA_;
      sample.motion_a = Vector<2>{std::cos(target_xyY_.z() - uavA_state_.w()),
                                  std::sin(target_xyY_.z() - uavA_state_.w())};

      // 将单帧加入缓冲区
      chunk_buffer.push_back(sample);

      // 当缓冲区达到 40 步时，保存文件
      if (chunk_buffer.size() == chunk_size) {
        std::ostringstream fname;
        fname
          << "/home/lac/fm_test/src/flightmare/flightrl/collaborative_detect/"
             "dataset3/samples/ep_"
          << std::setfill('0') << std::setw(4) << ep_id << "_chunk_"
          << std::setw(4) << chunk_id << ".json";

        save_chunk(chunk_buffer, fname.str());  // 执行落盘

        // 写入索引文件
        if (total_chunks_generated < train_split) {
          train_list_ << fname.str() << "\n";
        } else if (total_chunks_generated < val_split) {
          val_list_ << fname.str() << "\n";
        } else {
          test_list_ << fname.str() << "\n";
        }

        chunk_buffer.clear();  // 清空缓冲区
        chunk_id++;
        total_chunks_generated++;
      }
    }  // End of steps loop

    if ((ep_id + 1) % 10 == 0) {
      logger_.info("[PROGRESS] Generated %d/%d episodes...", ep_id + 1,
                   num_episodes);
    }
  }  // End of episodes loop

  logger_.info(
    "Dataset generation completed! Total chunks (40-steps) generated: %d",
    total_chunks_generated);
}


void DatasetGenerator::save_chunk(const std::vector<SampleData>& chunk,
                                  const std::string& filename) {
  json j_sequence = json::array();
  for (const auto& sample : chunk) {
    j_sequence.push_back(sample.to_json());
  }
  std::ofstream ofs(filename);
  if (ofs.is_open()) {
    ofs << j_sequence.dump();  // 节省空间不美化，若要调试可改 dump(2)
    ofs.close();
  } else {
    logger_.error("Failed to write %s", filename.c_str());
  }
}


void DatasetGenerator::random_initial_states() {
  sim_time_ = 0.0f;
  uavA_vel_.setZero();
  uavB_vel_.setZero();

  bool is_visualA = false, is_visualB = false;
  BBox bboxA, bboxB;

  Vector<3> target_init =
    Vector<3>{uniform_dist_(rng_), uniform_dist_(rng_), 0.0f};
  TrajectoryType traj_type = random_trajectory_type();
  target_.reset(target_init, traj_type, 1.0f);
  if (uniform_01_dist_(rng_) < 0.05f) {
    target_.setMaxSpeed(0.0f);
  }
  target_.update(sim_time_);
  target_xyY_ = target_.getPose();
  while (!is_visualA || !is_visualB) {
    // randomize target initial state

    

    Scalar init_distA = uniform_dist_(rng_) * 2.0f + 5.0f;
    Scalar init_yawA = uniform_dist_(rng_) * M_PI;
    uavA_state_ =
      Vector<4>{target_xyY_.x() - init_distA * std::cos(init_yawA),
                target_xyY_.y() - init_distA * std::sin(init_yawA), 0.8f,
                init_yawA + uniform_dist_(rng_) * M_PI * init_distA / 108};

    Scalar init_distB = uniform_dist_(rng_) * 2.0f + 5.0f;
    Scalar init_yawB = uniform_dist_(rng_) * M_PI;
    uavB_state_ =
      Vector<4>{target_xyY_.x() - init_distB * std::cos(init_yawB),
                target_xyY_.y() - init_distB * std::sin(init_yawB), 0.8f,
                init_yawB + uniform_dist_(rng_) * M_PI * init_distB / 108};

    // check visual
    detect_.updateTarget(target_xyY_);
    is_visualA = detect_.getBBox(uavA_state_, bboxA);
    is_visualB = detect_.getBBox(uavB_state_, bboxB);

  }
  bboxA_l_ = bboxA;
  bboxB_l_ = bboxB;
  bboxA_ = bboxA;
  bboxB_ = bboxB;
  
  uavA_last_ = uavA_state_;
  uavB_last_ = uavB_state_;

}
  

void DatasetGenerator::update_uavs() {
  Vector<4> uavA_temp = uavA_state_;
  Vector<4> uavB_temp = uavB_state_;
  Vector<2> uavA2target = target_xyY_.head<2>() - uavA_state_.head<2>();
  Vector<2> uavB2target = target_xyY_.head<2>() - uavB_state_.head<2>();

  Scalar dist_A = uavA2target.norm();
  Scalar dist_B = uavB2target.norm();
  Scalar desired_dist = 3.0f;
  Scalar kp = 0.5;

  if (dist_A < (desired_dist - 0.5f) || dist_A > 2*desired_dist) {
    Vector<2> dir_A = uavA2target.normalized();
    Scalar speed =
      std::max(-1.0f, std::min(1.0f, kp * (dist_A - desired_dist)));
    uavA_vel_.head<2>() = speed * dir_A;
  } else {
    Vector<2> dir_A = Vector<2>(uavA2target.y(), -uavA2target.x()).normalized();
    uavA_vel_.head<2>() = dir_A * 0.8f;
  }
  if (dist_B < (desired_dist - 0.5f) || dist_B > 2*desired_dist) {
    Vector<2> dir_B = uavB2target.normalized();
    Scalar speed =
      std::max(-1.0f, std::min(1.0f, kp * (dist_B - desired_dist)));
    uavB_vel_.head<2>() = speed * dir_B;
  } else {
    Vector<2> dir_B = Vector<2>(-uavB2target.y(), uavB2target.x()).normalized();
    uavB_vel_.head<2>() = dir_B * 0.8f;
  }
  uavA_temp.head<2>() += uavA_vel_.head<2>() * dt_;
  uavB_temp.head<2>() += uavB_vel_.head<2>() * dt_;

  Vector<2> dir_AT = uavA2target.normalized();
  Scalar betaA = std::cos(uavA_state_.w()) * dir_AT.y() -
                 std::sin(uavA_state_.w()) * dir_AT.x();
  Vector<2> dir_BT = uavB2target.normalized();
  Scalar betaB = std::cos(uavB_state_.w()) * dir_BT.y() -
                 std::sin(uavB_state_.w()) * dir_BT.x();

  bool is_visualA = false, is_visualB = false;
  BBox bboxA, bboxB;
  while (!is_visualA || !is_visualB) {
    if (std::abs(betaA) > 0.2f) {
      uavA_vel_.w() = kp * std::asin(betaA);
    } else {
      uavA_vel_.w() = uniform_dist_(rng_) * 0.3f;
    }
    if (std::abs(betaB) > 0.2f) {
      uavB_vel_.w() = kp * std::asin(betaB);
    } else {
      uavB_vel_.w() = uniform_dist_(rng_) * 0.3f;
    }
    uavA_temp.w() += uavA_vel_.w() * dt_;
    uavB_temp.w() += uavB_vel_.w() * dt_;

    // check visual
    detect_.updateTarget(target_xyY_);
    is_visualA = detect_.getBBox(uavA_temp, bboxA);
    is_visualB = detect_.getBBox(uavB_temp, bboxB);
    
  }
  bboxA_l_ = bboxA_;
  bboxB_l_ = bboxB_;
  bboxA_ = bboxA;
  bboxB_ = bboxB;



  uavA_last_ = uavA_state_;
  uavB_last_ = uavB_state_;

  uavA_state_ = uavA_temp;
  uavB_state_ = uavB_temp;

}


std::vector<SampleData> DatasetGenerator::generate_sequence(int seq_id) {
  std::vector<SampleData> samples;

  random_initial_states();
  int num_steps = 100;
  for (int step_id = 0; step_id < num_steps; ++step_id) {
    sim_time_ = step_id * dt_;
    target_.update(sim_time_);
    target_xyY_ = target_.getPose();
    update_uavs();
    
    SampleData sample;
    sample.pose_b_a_pos = uavB_state_.head<3>() - uavA_state_.head<3>();
    // sample.pose_b_a_quat = Vector<4>(
    //   0.0f, 0.0f, 0.0f, uavB_state_.w() - uavA_state_.w());
    sample.pose_b_a_quat = Vector<4>(
      0.0f, 0.0f, std::sin((uavB_state_.w() - uavA_state_.w()) / 2.0f),
      std::cos((uavB_state_.w() - uavA_state_.w()) / 2.0f));
    sample.bbox_b = bboxB_;
    sample.motion_b = Vector<2>{std::cos(target_xyY_.z() - uavB_state_.w()),
                                std::sin(target_xyY_.z() - uavB_state_.w())};
    sample.pose_a_delta_pos = uavA_last_.head<3>() - uavA_state_.head<3>();
    // sample.pose_a_delta_quat =
    //   Vector<4>(0.0f, 0.0f, 0.0f, uavA_last_.w() - uavA_state_.w());
    sample.pose_a_delta_quat =
      Vector<4>(0.0f, 0.0f, std::sin((uavA_last_.w() - uavA_state_.w()) / 2.0f),
                std::cos((uavA_last_.w() - uavA_state_.w()) / 2.0f));
    sample.bbox_a_last = bboxA_l_;
    sample.bbox_a = bboxA_;
    sample.motion_a = Vector<2>{std::cos(target_xyY_.z() - uavA_state_.w()),
                                std::sin(target_xyY_.z() - uavA_state_.w())};
    sample.episode_id = seq_id;
    sample.frame_id = step_id;
    sample.timestamp = sim_time_;
    sample.traj_type = target_.getTrajectoryType();
    samples.push_back(sample);
  }
  return samples;

}

TrajectoryType DatasetGenerator::random_trajectory_type() {
  int type = std::floor(uniform_01_dist_(rng_) * 4);
  return static_cast<TrajectoryType>(type);
}


} // namespace flightros




