/*** 
 * @Author: Flightmare
 * @Date: 2025-05-19 22:37:29 +0800
 * @LastEditTime: 2025-05-23 19:42:00 +0800
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
#include "flightlib/envs/quadrotor_env/quadrotor_env.hpp"
#include "flightlib/envs/track_env/track_env.hpp"
#include "flightlib/envs/test_env.hpp"
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

  py::class_<TestEnv<QuadrotorEnv>>(m, "TestEnv_v0")
    .def(py::init<>())
    .def("reset", &TestEnv<QuadrotorEnv>::reset)
    .def("__repr__", [](const TestEnv<QuadrotorEnv>& a) { return "Test Env"; });


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

}