/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-05-15 12:48:49 +0800
 * @LastEditTime: 2025-06-16 19:46:45 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /flightmare/flightlib/src/common/trajectory_point.cpp
 */
#include "flightlib/common/trajectory_point.hpp"

namespace flightlib
{

TrajectoryPoint::TrajectoryPoint() :
    position(Vector<3>::Zero()), orientation(Quaternion::Identity()),
			velocity(Vector<3>::Zero()), acceleration(Vector<3>::Zero()), 
			jerk(Vector<3>::Zero()), snap(Vector<3>::Zero()), 
			bodyrates(Vector<3>::Zero()), angular_acceleration(Vector<3>::Zero()),
			angular_jerk(Vector<3>::Zero()), angular_snap(Vector<3>::Zero()),
			heading(0.0), heading_rate(0.0), 
			heading_acceleration(0.0)
{
}

TrajectoryPoint::~TrajectoryPoint()
{
}

} // namespace flightlib