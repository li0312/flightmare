/*** 
 * @Author: Lac_Creeper
 * @Date: 2025-05-15 13:01:09 +0800
 * @LastEditTime: 2025-05-15 13:41:19 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /src/flightmare/flightlib/include/flightlib/position_controller/position_controller_params.hpp
 */
#pragma once

#include "flightlib/common/types.hpp"

namespace flightlib
{

class PositionControllerParams {
	public:
		PositionControllerParams()
				: use_rate_mode(true), 
					kpxy(0.0),
					kdxy(0.0),
					kpz(0.0),
					kdz(0.0),
					krp(0.0),
					kyaw(0.0),
					pxy_error_max(0.0),
					vxy_error_max(0.0),
					pz_error_max(0.0),
					vz_error_max(0.0),
					yaw_error_max(0.0),
					perform_aerodynamics_compensation(false),
					k_drag_x(0.0),
					k_drag_y(0.0),
					k_drag_z(0.0),
					k_thrust_horz(0.0) {}

		~PositionControllerParams() {}

		void loadDefaultParams() {
			use_rate_mode = true;
			kpxy = 10.0;
			kdxy = 4.0;

			kpz = 15.0;
			kdz = 6.0;
			krp = 12.0;
			kyaw = 5.0;

			pxy_error_max = 0.6;
			vxy_error_max = 1.0;
			pz_error_max = 0.3;
			vz_error_max = 0.75;
			yaw_error_max = 0.7;

			perform_aerodynamics_compensation = false;

			k_drag_x = 0.0;
			k_drag_y = 0.0;
			k_drag_z = 0.0;
			k_thrust_horz = 0.0;
		}

		bool use_rate_mode;

		Scalar kpxy;  // [1/s^2]
		Scalar kdxy;  // [1/s]

		Scalar kpz;  // [1/s^2]
		Scalar kdz;  // [1/s]

		Scalar krp;   // [1/s]
		Scalar kyaw;  // [1/s]

		Scalar pxy_error_max;  // [m]
		Scalar vxy_error_max;  // [m/s]
		Scalar pz_error_max;   // [m]
		Scalar vz_error_max;   // [m/s]
		Scalar yaw_error_max;  // [rad]

		// Whether or not to compensate for aerodynamic effects
		bool perform_aerodynamics_compensation;
		Scalar k_drag_x;  // x-direction rotor drag coefficient
		Scalar k_drag_y;  // y-direction rotor drag coefficient
		Scalar k_drag_z;  // z-direction rotor drag coefficient
		// thrust correction coefficient due to body horizontal velocity
		Scalar k_thrust_horz;
};

} // namespace flightlib