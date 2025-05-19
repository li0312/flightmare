#include "flightlib/position_controller/position_controller.hpp"

#include <iostream>

namespace flightlib {

PositionController::PositionController() {}

PositionController::~PositionController() {}

Command PositionController::run(
		const QuadState& state_estimate,
		const Trajectory& reference_trajectory,
		const PositionControllerParams& config) {
	Command command;
	
	TrajectoryPoint reference_state(
			reference_trajectory.points.front());
	
	Quaternion state_orientation(state_estimate.x(QS::ATTW),
															 state_estimate.x(QS::ATTX), 
															 state_estimate.x(QS::ATTY), 
															 state_estimate.x(QS::ATTZ));

	//Compute reference inputs
	Vector<3> drag_accelerations = Vector<3>::Zero();
	Command reference_inputs;
	if (config.perform_aerodynamics_compensation) {
		// Compute reference inputs that compensate for aerodynamic drag
		// computeAeroCompensatedReferenceInputs(reference_state, state_estimate,
		// 																			config, &reference_inputs,
		// 																			&drag_accelerations);
		std::cout << "ERROR" << std::endl;
	} else {
		//In this case we are not considering aerodynamic accelerations
		drag_accelerations = Vector<3>::Zero();

		// Compute reference inputs as feed forward terms
		reference_inputs = computeNominalReferenceInputs(
				reference_state, state_orientation);
	}

	// std::cout << reference_inputs << std::endl;


	// Compute desired control commands
	const Vector<3> pid_error_accelerations = 
			computePIDErrorAcc(state_estimate, reference_state, config);

	const Vector<3> desird_acceleration = pid_error_accelerations + 
																				reference_state.acceleration - 
																				GVEC - drag_accelerations;
	
	command.collective_thrust = computeDesiredCollectiveMassNormalizedThrust(
			state_orientation, desird_acceleration, config);
	if (config.perform_aerodynamics_compensation) {
		// This compensates for an acceleration component in thrust direction due 
		// to the square of the body-horizontal velocity.
		std::cout << "ERROR" << std::endl;

		command.collective_thrust -= 
				config.k_thrust_horz * (pow(state_estimate.x(QS::VELX), 2.0) + 
																pow(state_estimate.x(QS::VELY), 2.0));
	}

	const Quaternion desired_attitude = 
			computeDesiredAttitude(desird_acceleration, reference_state.heading,
															state_orientation);
	const Vector<3> feedback_bodyrates = computeFeedBackControlBodyrates(
			desired_attitude, state_orientation, config);
	
	if (config.use_rate_mode) {
		// Control mode [BODY_RATES] 
		command.omega = reference_inputs.omega + feedback_bodyrates;
	} else {
		// Control mode [ATTITUDE]
		// TODO: The control instructions are not defined in command.hpp 
		std::cout << "ERROR" << std::endl;

	}

	return command;
}

Command
PositionController::computeNominalReferenceInputs(
		const TrajectoryPoint& reference_state,
		const Quaternion& attitude_estimate) const {
	Command reference_command;

	const Quaternion q_heading =  Quaternion(
			Eigen::AngleAxisf(reference_state.heading, Vector<3>::UnitZ()));
	
	const Vector<3> x_C = q_heading * Vector<3>::UnitX();
	const Vector<3> y_C = q_heading * Vector<3>::UnitY();

	const Vector<3> des_acc = reference_state.acceleration - GVEC;

	// Reference attitude
	const Quaternion q_W_B = computeDesiredAttitude(
			des_acc, reference_state.heading, attitude_estimate);
	
	const Vector<3> x_B = q_W_B * Vector<3>::UnitX();
	const Vector<3> y_B = q_W_B * Vector<3>::UnitY();
	const Vector<3> z_B = q_W_B * Vector<3>::UnitZ();

	// REVIEW: orientation
	// reference_command.orientation = q_W_B;

	// Reference thrust
	reference_command.collective_thrust = des_acc.norm();

	// Reference body rates
	if (almostZeroThrust(reference_command.collective_thrust)) {
		reference_command.omega.x() = 0.0;
		reference_command.omega.y() = 0.0;
	} else {
		reference_command.omega.x() = -1.0 / 
																	reference_command.collective_thrust * 
																	y_B.dot(reference_state.jerk);
		reference_command.omega.y() = -1.0 / 
																	reference_command.collective_thrust * 
																	x_B.dot(reference_state.jerk);
	}

	if (almostZero((y_C.cross(z_B)).norm())) {
		reference_command.omega.z() = 0.0;
	} else {
		reference_command.omega.z() = 
				1.0 / (y_C.cross(z_B)).norm() * 
				(reference_state.heading_rate * x_C.dot(x_B) + 
				reference_command.omega.y() * y_C.dot(z_B));
	}

	// REVIEW: Reference angular accelerations

	return reference_command;
}

Vector<3> PositionController::computePIDErrorAcc(
		const QuadState& state_estimate,
		const TrajectoryPoint& reference_state,
		const PositionControllerParams& config) const {
	// Compute the desired accelerations due to control errors in world frame
  // with a PID controller
	Vector<3> acc_error;

	// x acceleration
	Scalar x_pos_error = 
			reference_state.position.x() - state_estimate.x(QS::POSX);
	limit(&x_pos_error, -config.pxy_error_max,
														config.pxy_error_max);
	
	Scalar x_vel_error = 
			reference_state.velocity.x() - state_estimate.x(QS::VELX);
	limit(&x_vel_error, -config.vxy_error_max,
														config.vxy_error_max);
	
	acc_error.x() = config.kpxy * x_pos_error + config.kdxy * x_vel_error;

	// y acceleration
	Scalar y_pos_error = 
			reference_state.position.y() - state_estimate.x(QS::POSY);
	limit(&y_pos_error, -config.pxy_error_max,
														config.pxy_error_max);
	
	Scalar y_vel_error = 
			reference_state.velocity.y() - state_estimate.x(QS::VELY);
	limit(&y_vel_error, -config.vxy_error_max,
														config.vxy_error_max);
	
	acc_error.y() = config.kpxy * y_pos_error + config.kdxy * y_vel_error;

	// z acceleration
	Scalar z_pos_error = 
			reference_state.position.z() - state_estimate.x(QS::POSZ);
	limit(&z_pos_error, -config.pz_error_max,
														config.pz_error_max);
	
	Scalar z_vel_error = 
			reference_state.velocity.z() - state_estimate.x(QS::VELZ);
	limit(&z_vel_error, -config.vz_error_max,
														config.vz_error_max);
	
	acc_error.z() = config.kpz * z_pos_error + config.kdz * z_vel_error;
	
	return acc_error;
}

Scalar PositionController::computeDesiredCollectiveMassNormalizedThrust(
		const Quaternion& attitude_estimate,
		const Vector<3>& desired_acc,
		const PositionControllerParams& config) const {
	const Vector<3> body_z_axis = 
			attitude_estimate * Vector<3>::UnitZ();
	
	Scalar normalized_thrust = desired_acc.dot(body_z_axis);
	if (normalized_thrust < kMinNormalizedCollectiveThrust_) {
		normalized_thrust = kMinNormalizedCollectiveThrust_;
	}
	return normalized_thrust;
}

Quaternion PositionController::computeDesiredAttitude(
		const Vector<3>& desired_acceleration, const Scalar reference_heading,
		const Quaternion& attitude_estimate) const {
	const Quaternion q_heading = Quaternion(
			Eigen::AngleAxisf(reference_heading, Vector<3>::UnitZ()));
	
	// Compute desired orientation
	const Vector<3> x_C = q_heading * Vector<3>::UnitX();
	const Vector<3> y_C = q_heading * Vector<3>::UnitY();

	Vector<3> z_B;
	if (almostZero(desired_acceleration.norm())) {
		// In case of free fall we keep the thrust direction to be the estimated one
    // This only works assuming that we are in this condition for a very short
    // time (otherwise attitude drifts)
		z_B = attitude_estimate * Vector<3>::UnitZ();
	} else {
		z_B = desired_acceleration.normalized();
	}

	const Vector<3> x_B_prototype = y_C.cross(z_B);
	const Vector<3> x_B = 
			computeRobustBodyXAxis(x_B_prototype, x_C, y_C, attitude_estimate);
	
	const Vector<3> y_B = (z_B.cross(x_B)).normalized();

	// From the compute desired body axes we can now compse a desired attitude 
	const Matrix<3, 3> R_W_B((Matrix<3, 3>() << x_B, y_B, z_B).finished());

	const Quaternion desired_attitude(R_W_B);

	return desired_attitude;
}

Vector<3> PositionController::computeRobustBodyXAxis(
		const Vector<3>& x_B_prototype, const Vector<3>& x_C,
		const Vector<3>& y_C,
		const Quaternion& attitude_estimate) const {
	Vector<3> x_B = x_B_prototype;

	if (almostZero(x_B.norm())) {
		// if cross(y_C, z_B) == 0, they are collinear => 
		// every x_B lies automatically in the x_C - z_C plane

		// Project estimated body x-axis into the x_C - z_C plane
		const Vector<3> x_B_estimate =
				attitude_estimate * Vector<3>::UnitX();
		const Vector<3> x_B_projected = 
				x_B_estimate - (x_B_estimate.dot(y_C)) * y_C;
		if (almostZero(x_B_projected.norm())) {
			// Not too much intelligent stuff we can do in this case but it should 
			// basically never occur
			x_B = x_C;
		} else {
			x_B = x_B_projected.normalized();
		}
	} else {
		x_B.normalized();
	}

	// if the quad is upside down, x_B will point in the "opposite" direction
  // of x_C => flip x_B (unfortunately also not the solution for our problems)
  //  if (x_B.dot(x_C) < 0.0)
  //  {
  //    x_B = -x_B;
  //  }

	return x_B;
}

Vector<3> PositionController::computeFeedBackControlBodyrates(
    const Quaternion& desired_attitude,
    const Quaternion& attitude_estimate,
    const PositionControllerParams& config) const {
  // Compute the error quaternion
  const Quaternion q_e = attitude_estimate.inverse() * desired_attitude;

  // Compute desired body rates from control error
  Vector<3> bodyrates;

  if (q_e.w() >= 0) {
    bodyrates.x() = 2.0 * config.krp * q_e.x();
    bodyrates.y() = 2.0 * config.krp * q_e.y();
    bodyrates.z() = 2.0 * config.kyaw * q_e.z();
  } else {
    bodyrates.x() = -2.0 * config.krp * q_e.x();
    bodyrates.y() = -2.0 * config.krp * q_e.y();
    bodyrates.z() = -2.0 * config.kyaw * q_e.z();
  }

  return bodyrates;
}

bool PositionController::almostZero(const Scalar value) const {
  return fabs(value) < kAlmostZeroValueThreshold_;
}

bool PositionController::almostZeroThrust(const Scalar thrust_value) const {
  return fabs(thrust_value) < kAlmostZeroThrustThreshold_;
}

void PositionController::limit(Scalar *val, const Scalar min, const Scalar max) const
{
  if (*val > max)
  {
    *val = max;
  }
  if (*val < min)
  {
    *val = min;
  }
}

} // namespace flightlib