/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-05-15 12:48:58 +0800
 * @LastEditTime: 2025-05-15 12:48:59 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /src/flightmare/flightlib/src/common/trajectory.cpp
 */
#include "flightlib/common/trajectory.hpp"

namespace flightlib 
{

Trajectory::Trajectory() :
	trajectory_type(TrajectoryType::UNDEFINED), points()
{
}

Trajectory::Trajectory(const flightlib::TrajectoryPoint& point) :
	trajectory_type(TrajectoryType::GENERAL), points()
{
	points.push_back(point);
}

Trajectory::~Trajectory()
{
}


} // namespace flightlib