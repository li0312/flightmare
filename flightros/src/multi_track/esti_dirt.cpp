/***
 * @Author: Lac_Creeper
 * @Date: 2026-02-27 02:35:53 +0800
 * @LastEditTime: 2026-02-27 02:35:55 +0800
 * @LastEditors: Lac_Creeper
 * @Description:
 * @FilePath: /flightmare/flightros/src/multi_track/esti_dirt.cpp
 */
#include "flightros/multi_track/esti_dirt.hpp"


EstiDirt::EstiDirt(const ros::NodeHandle& nh, const ros::NodeHandle& pnh)
  : nh_(nh), pnh_(pnh), main_loop_freq_(50) {}