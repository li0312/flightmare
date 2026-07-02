/***
 * @Author: Flightmare
 * @Date: 2025-05-19 22:37:29 +0800
 * @LastEditTime: 2025-07-01 16:35:50 +0800
 * @LastEditors: Lac_Creeper
 * @Description:
 * @FilePath: /flightmare/flightlib/src/wrapper/pybind_wrapper.cpp
 */

// pybind11
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// flightlib
#include "flightlib/envs/env_base.hpp"
#include "flightlib/envs/gimbal_env/gimbal_env.hpp"
#include "flightlib/envs/quadrotor_env/quadrotor_env.hpp"
#include "flightlib/envs/test_env.hpp"
#include "flightlib/envs/track_env/track_env.hpp"
#include "flightlib/envs/trackAdv_env/trackAdv_env.hpp"
#include "flightlib/envs/trackMult_env/trackMult_env.hpp"
#include "flightlib/envs/vec_env.hpp"

namespace py = pybind11;
using namespace flightlib;

PYBIND11_MODULE(flightgym, m) {
  py::class_<VecEnv<QuadrotorEnv>>(m, "QuadrotorEnv_v1")
    .def(py::init<>())
    .def(py::init<const std::string&>())
    .def(py::init<const std::string&, const bool>())
    .def("reset", &VecEnv<QuadrotorEnv>::reset)
    .def("step", &VecEnv<QuadrotorEnv>::step)
    .def("testStep", &VecEnv<QuadrotorEnv>::testStep)
    .def("setSeed", &VecEnv<QuadrotorEnv>::setSeed)
    .def("close", &VecEnv<QuadrotorEnv>::close)
    .def("isTerminalState", &VecEnv<QuadrotorEnv>::isTerminalState)
    .def("curriculumUpdate", &VecEnv<QuadrotorEnv>::curriculumUpdate)
    .def("connectUnity", &VecEnv<QuadrotorEnv>::connectUnity)
    .def("disconnectUnity", &VecEnv<QuadrotorEnv>::disconnectUnity)
    .def("getNumOfEnvs", &VecEnv<QuadrotorEnv>::getNumOfEnvs)
    .def("getObsDim", &VecEnv<QuadrotorEnv>::getObsDim)
    .def("getActDim", &VecEnv<QuadrotorEnv>::getActDim)
    .def("getExtraInfoNames", &VecEnv<QuadrotorEnv>::getExtraInfoNames)
    .def("__repr__", [](const VecEnv<QuadrotorEnv>& a) {
      return "RPG Drone Racing Environment";
    });

  // py::class_<TestEnv<QuadrotorEnv>>(m, "TestEnv_v0")
  //   .def(py::init<>())
  //   .def("reset", &TestEnv<QuadrotorEnv>::reset)
  //   .def("__repr__", [](const TestEnv<QuadrotorEnv>& a) { return "Test Env";
  //   });

  py::class_<VecEnv<TrackEnv>>(m, "TrackEnv_v1")
    .def(py::init<>())
    .def(py::init<const std::string&>())
    .def(py::init<const std::string&, const bool>())
    .def("reset", &VecEnv<TrackEnv>::reset)
    .def("step", &VecEnv<TrackEnv>::step)
    .def("testStep", &VecEnv<TrackEnv>::testStep)
    .def("setSeed", &VecEnv<TrackEnv>::setSeed)
    .def("close", &VecEnv<TrackEnv>::close)
    .def("isTerminalState", &VecEnv<TrackEnv>::isTerminalState)
    .def("curriculumUpdate", &VecEnv<TrackEnv>::curriculumUpdate)
    .def("connectUnity", &VecEnv<TrackEnv>::connectUnity)
    .def("disconnectUnity", &VecEnv<TrackEnv>::disconnectUnity)
    .def("getNumOfEnvs", &VecEnv<TrackEnv>::getNumOfEnvs)
    .def("getObsDim", &VecEnv<TrackEnv>::getObsDim)
    .def("getActDim", &VecEnv<TrackEnv>::getActDim)
    .def("getExtraInfoNames", &VecEnv<TrackEnv>::getExtraInfoNames)
    .def("__repr__", [](const VecEnv<TrackEnv>& a) {
      return "RPG Drone Tracking Environment";
    });

  py::class_<VecEnv<TrackAdvEnv>>(m, "TrackAdvEnv_v2")
    .def(py::init<>())
    .def(py::init<const std::string&>())
    .def(py::init<const std::string&, const bool>())
    .def("reset", &VecEnv<TrackAdvEnv>::reset)
    .def("step", &VecEnv<TrackAdvEnv>::step)
    .def("testStep", &VecEnv<TrackAdvEnv>::testStep)
    .def("setSeed", &VecEnv<TrackAdvEnv>::setSeed)
    .def("close", &VecEnv<TrackAdvEnv>::close)
    .def("isTerminalState", &VecEnv<TrackAdvEnv>::isTerminalState)
    .def("curriculumUpdate", &VecEnv<TrackAdvEnv>::curriculumUpdate)
    .def("connectUnity", &VecEnv<TrackAdvEnv>::connectUnity)
    .def("disconnectUnity", &VecEnv<TrackAdvEnv>::disconnectUnity)
    .def("getNumOfEnvs", &VecEnv<TrackAdvEnv>::getNumOfEnvs)
    .def("getObsDim", &VecEnv<TrackAdvEnv>::getObsDim)
    .def("getActDim", &VecEnv<TrackAdvEnv>::getActDim)
    .def("getExtraInfoNames", &VecEnv<TrackAdvEnv>::getExtraInfoNames)
    .def("__repr__", [](const VecEnv<TrackAdvEnv>& a) {
      return "RPG Drone Tracking Environment";
    });

  py::class_<VecEnv<TrackMultEnv>>(m, "TrackMultEnv_v0")
    .def(py::init<>())
    .def(py::init<const std::string&>())
    .def(py::init<const std::string&, const bool>())
    .def("reset", &VecEnv<TrackMultEnv>::reset)
    .def("step", &VecEnv<TrackMultEnv>::step)
    .def("testStep", &VecEnv<TrackMultEnv>::testStep)
    .def("setSeed", &VecEnv<TrackMultEnv>::setSeed)
    .def("close", &VecEnv<TrackMultEnv>::close)
    .def("isTerminalState", &VecEnv<TrackMultEnv>::isTerminalState)
    .def("curriculumUpdate", &VecEnv<TrackMultEnv>::curriculumUpdate)
    .def("connectUnity", &VecEnv<TrackMultEnv>::connectUnity)
    .def("disconnectUnity", &VecEnv<TrackMultEnv>::disconnectUnity)
    .def("getNumOfEnvs", &VecEnv<TrackMultEnv>::getNumOfEnvs)
    .def("getObsDim", &VecEnv<TrackMultEnv>::getObsDim)
    .def("getActDim", &VecEnv<TrackMultEnv>::getActDim)
    .def("getExtraInfoNames", &VecEnv<TrackMultEnv>::getExtraInfoNames)
    .def("__repr__", [](const VecEnv<TrackMultEnv>& a) {
      return "RPG Drone Tracking Environment";
    });

  py::class_<VecEnv<ObstacleEnv>>(m, "ObstEnv_v1")
    .def(py::init<>())
    .def(py::init<const std::string&>())
    .def(py::init<const std::string&, const bool>())
    .def("reset", &VecEnv<ObstacleEnv>::reset)
    .def("step", &VecEnv<ObstacleEnv>::step)
    .def("testStep", &VecEnv<ObstacleEnv>::testStep)
    .def("setSeed", &VecEnv<ObstacleEnv>::setSeed)
    .def("close", &VecEnv<ObstacleEnv>::close)
    .def("isTerminalState", &VecEnv<ObstacleEnv>::isTerminalState)
    .def("curriculumUpdate", &VecEnv<ObstacleEnv>::curriculumUpdate)
    .def("connectUnity", &VecEnv<ObstacleEnv>::connectUnity)
    .def("disconnectUnity", &VecEnv<ObstacleEnv>::disconnectUnity)
    .def("getNumOfEnvs", &VecEnv<ObstacleEnv>::getNumOfEnvs)
    .def("getObsDim", &VecEnv<ObstacleEnv>::getObsDim)
    .def("getActDim", &VecEnv<ObstacleEnv>::getActDim)
    .def("getExtraInfoNames", &VecEnv<ObstacleEnv>::getExtraInfoNames)
    .def("__repr__", [](const VecEnv<ObstacleEnv>& a) {
      return "RPG Drone Tracking Environment";
    });

  py::class_<VecEnv<GimbalEnv>>(m, "GimbalEnv_v1")
    .def(py::init<>())
    .def(py::init<const std::string&>())
    .def(py::init<const std::string&, const bool>())
    .def("reset", &VecEnv<GimbalEnv>::reset)
    .def("step", &VecEnv<GimbalEnv>::step)
    .def("testStep", &VecEnv<GimbalEnv>::testStep)
    .def("setSeed", &VecEnv<GimbalEnv>::setSeed)
    .def("close", &VecEnv<GimbalEnv>::close)
    .def("isTerminalState", &VecEnv<GimbalEnv>::isTerminalState)
    .def("curriculumUpdate", &VecEnv<GimbalEnv>::curriculumUpdate)
    .def("connectUnity", &VecEnv<GimbalEnv>::connectUnity)
    .def("disconnectUnity", &VecEnv<GimbalEnv>::disconnectUnity)
    .def("getNumOfEnvs", &VecEnv<GimbalEnv>::getNumOfEnvs)
    .def("getObsDim", &VecEnv<GimbalEnv>::getObsDim)
    .def("getActDim", &VecEnv<GimbalEnv>::getActDim)
    .def("getExtraInfoNames", &VecEnv<GimbalEnv>::getExtraInfoNames)
    .def("__repr__", [](const VecEnv<GimbalEnv>& a) {
      return "RPG Drone Tracking Environment";
    });
}