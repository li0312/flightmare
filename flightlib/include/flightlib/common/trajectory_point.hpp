/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-05-15 12:56:14 +0800
 * @LastEditTime: 2025-06-16 19:47:36 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /flightmare/flightlib/include/flightlib/common/trajectory_point.hpp
 */
# pragma once

#include <eigen3/Eigen/Dense>
#include "flightlib/common/types.hpp"

namespace flightlib 
{

struct TrajectoryPoint
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    TrajectoryPoint();
    virtual ~TrajectoryPoint();

    // Pose
    Vector<3> position;
    Quaternion orientation;

    // Linear derivatives
    Vector<3> velocity;
    Vector<3> acceleration;
    Vector<3> jerk;
    Vector<3> snap;

    // Angular dervatives
    Vector<3> bodyrates;
    Vector<3> angular_acceleration;
    Vector<3> angular_jerk;
    Vector<3> angular_snap;

    // Heading angle with respect to world frame [rad]
    Scalar heading;
    Scalar heading_rate;
    Scalar heading_acceleration;
};

} // namespace flightlib