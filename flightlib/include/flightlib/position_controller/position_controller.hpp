#pragma once

#include "flightlib/common/command.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/trajectory.hpp"
#include "flightlib/common/trajectory_point.hpp"
#include <eigen3/Eigen/Dense>

#include "flightlib/position_controller/position_controller_params.hpp"

namespace flightlib
{

class PositionController {
 public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	PositionController();
	~PositionController();

  Command run(
      const QuadState& state_estimate,
		  const Trajectory& reference_trajectory,
		  const PositionControllerParams& config);

 private:
  Command computeNominalReferenceInputs(
		  const TrajectoryPoint& reference_state,
		  const Quaternion& attitude_estimate) const;

  Vector<3> computePIDErrorAcc(
		  const QuadState& state_estimate,
		  const TrajectoryPoint& reference_state,
		  const PositionControllerParams& config) const;
  
  Scalar computeDesiredCollectiveMassNormalizedThrust(
		  const Quaternion& attitude_estimate,
		  const Vector<3>& desired_acc,
		  const PositionControllerParams& config) const;

  Quaternion computeDesiredAttitude(
		  const Vector<3>& desired_acceleration, const Scalar reference_heading,
		  const Quaternion& attitude_estimate) const;

  Vector<3> computeRobustBodyXAxis(
		  const Vector<3>& x_B_prototype, const Vector<3>& x_C,
		  const Vector<3>& y_C,
		  const Quaternion& attitude_estimate) const;
  
  Vector<3> computeFeedBackControlBodyrates(
      const Quaternion& desired_attitude,
      const Quaternion& attitude_estimate,
      const PositionControllerParams& config) const;

  bool almostZero(const Scalar value) const;
  bool almostZeroThrust(const Scalar thrust_value) const;
  void limit(Scalar *val, const Scalar min, const Scalar max) const;

	// Constants
  static constexpr Scalar kMinNormalizedCollectiveThrust_ = 1.0;
  static constexpr Scalar kAlmostZeroValueThreshold_ = 0.001;
  static constexpr Scalar kAlmostZeroThrustThreshold_ = 0.01;

};

} // namespace flightlib