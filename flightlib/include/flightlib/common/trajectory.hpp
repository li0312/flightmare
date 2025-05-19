/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-05-15 12:56:17 +0800
 * @LastEditTime: 2025-05-15 12:56:18 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /src/flightmare/flightlib/include/flightlib/common/trajectory.hpp
 */
#pragma once

#include <list>

#include "flightlib/common/trajectory_point.hpp"
#include "flightlib/common/types.hpp"

namespace flightlib 
{

struct Trajectory
{
	Trajectory();
	Trajectory(const flightlib::TrajectoryPoint& point);
	virtual ~Trajectory();

	enum class TrajectoryType
	{
		UNDEFINED, GENERAL, ACCELERATION, JERK, SNAP
	} trajectory_type;

	std::list<flightlib::TrajectoryPoint> points;
};


} // namespace flightlib