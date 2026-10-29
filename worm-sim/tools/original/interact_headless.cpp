// Headless stand-in for BAAIWorm's interact module + SimulationWindow (MC_C302 branch).
// Same function names and math as Metaworm/interact/interact.cpp; no renderer.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <iostream>
#include "Environment.h"
#include "fem/World.h"
#include "util/MathUtil.h"

namespace py = pybind11;
using F32 = py::array_t<float, py::array::c_style | py::array::forcecast>;

static Eigen::VectorXd toEigen(const F32 &a, int n) {
  Eigen::VectorXd v(n); auto r = a.unchecked<1>();
  for (int i = 0; i < n; i++) v[i] = r(i);
  return v;
}
static F32 toNp(const Eigen::VectorXd &v) {
  F32 a(v.size()); auto w = a.mutable_unchecked<1>();
  for (int i = 0; i < v.size(); i++) w(i) = (float)v[i];
  return a;
}

static std::vector<Environment *> mWorms;
static std::vector<Eigen::Vector3d> mInitPositions, mInitForwards, mFood;
static std::vector<Eigen::Vector4d> mInitRotations;

static int AddWorm(F32 initPos, F32 dir, F32 color) {
  int id = mWorms.size();
  auto worm = new Environment();
  worm->Reset();
  mWorms.push_back(worm);
  Eigen::Vector3d pos = toEigen(initPos, 3), forward = toEigen(dir, 3);
  mInitPositions.push_back(pos);
  mInitForwards.push_back(forward);
  Eigen::Quaterniond q;
  q.setFromTwoVectors(Eigen::Vector3d(0, 0, -1), forward);
  Eigen::AngleAxisd ax; ax = q;
  mInitRotations.push_back(Eigen::Vector4d(ax.angle() * 180.0 / M_PI, ax.axis()[0], ax.axis()[1], ax.axis()[2]));
  return id;
}

// one display tick of SimulationWindow::Timer after wormInEnv.step(): body steps
static void Tick() {
  for (auto worm : mWorms) {
    int ratio = worm->GetSimulationHz() / worm->GetControlHz();
    unsigned int cur_phase = worm->GetPhase();
    Eigen::VectorXd tmp = worm->GetStates();
    for (int i = 0; i < ratio; i++) worm->Step();
    worm->UpdateAverageVelocity();
    worm->UpdateHeadLocation();
    worm->SetPhase(cur_phase + 1);
  }
}

PYBIND11_MODULE(interact, m) {
  m.def("set_muscle_signals", [](F32 a, int id) {
    if (a.shape(0) != 96) { std::cout << "Activation levels size not match!" << std::endl; return; }
    mWorms[id]->GetCreature()->SetActivationLevelsDirectly(toEigen(a, 96));
  });
  m.def("get_head_location", [](int id) {
    const Eigen::VectorXd &x = mWorms[id]->GetSoftWorld()->GetPositions();
    Eigen::Vector3i idx = mWorms[id]->GetCreature()->GetLocalContourIndex();
    return toNp(x.block<3, 1>(3 * idx[0], 0));
  });
  m.def("get_tracking_location", [](int id) {
    auto env = mWorms[id];
    const Eigen::VectorXd &x = env->GetSoftWorld()->GetPositions();
    const Eigen::VectorXd &v = env->GetSoftWorld()->GetVelocities();
    const std::vector<Eigen::Vector3d> &ref = env->GetCreature()->GetSamplingReference();
    Eigen::Affine3d c = env->GetEvalLocalCoord();
    Eigen::Vector3d cv = env->GetCurrentCoordVelocity();
    Eigen::Matrix3d R = c.rotation(); Eigen::Vector3d t = c.translation();
    Eigen::AngleAxisd ref_coord(M_PI / 2, Eigen::Vector3d::UnitY());
    Eigen::Matrix3d rot = R * ref_coord;
    const std::vector<int> &si = env->GetCreature()->GetSamplingIndex();
    Eigen::VectorXd p(3 * si.size()), vv(3 * si.size());
    Eigen::Vector3d lcv = rot.transpose() * cv;
    for (size_t i = 0; i < si.size(); i++) {
      p.block<3, 1>(3 * i, 0) = rot.inverse() * (x.block<3, 1>(3 * si[i], 0) - t) - ref_coord.inverse() * ref[i];
      vv.block<3, 1>(3 * i, 0) = rot.transpose() * v.block<3, 1>(3 * si[i], 0) - lcv;
    }
    Eigen::VectorXd j(p.size() + vv.size()); j << p, vv;
    return toNp(j);
  });
  m.def("get_init_position", [](int id) { return toNp(mInitPositions[id]); });
  m.def("get_init_rotation", [](int id) { return toNp(mInitRotations[id]); });
  m.def("get_forward_direction", [](int id) {
    const Eigen::VectorXd &x = mWorms[id]->GetSoftWorld()->GetPositions();
    return toNp(mWorms[id]->GetCreature()->GetForwardVector(x));
  });
  m.def("get_dorsal_direction", [](int id) {
    const Eigen::VectorXd &x = mWorms[id]->GetSoftWorld()->GetPositions();
    Eigen::Vector3d f = mWorms[id]->GetCreature()->GetForwardVector(x), u = mWorms[id]->GetCreature()->GetUpVector(x);
    return toNp(f.cross(u));
  });
  m.def("add_worm", &AddWorm);
  m.def("add_food", [](F32 p) { mFood.push_back(toEigen(p, 3)); });
  m.def("save_worm", [](int id, std::string path) { mWorms[id]->SaveWormStates(path); });
  m.def("restore_worm", [](int id, std::string path) { mWorms[id]->RestoreWormStates(path); });
  m.def("worm_local_to_world", [](int id, F32 p) {
    auto &r = mInitRotations[id];
    Eigen::AngleAxisd ax(r[0] * M_PI / 180.0, Eigen::Vector3d(r.segment(1, 3)));
    return toNp(ax * toEigen(p, 3) + mInitPositions[id]);
  });
  m.def("world_to_worm_local", [](int id, F32 p) {
    auto &r = mInitRotations[id];
    Eigen::AngleAxisd ax(r[0] * M_PI / 180.0, Eigen::Vector3d(r.segment(1, 3)));
    return toNp(ax.inverse() * (toEigen(p, 3) - mInitPositions[id]));
  });
  m.def("set_fog_color", [](F32) {});
  m.def("set_floor_color", [](F32) {});
  m.def("set_fog_params", [](int, F32) {});
  m.def("set_neuron_voltages", [](F32) {});
  m.def("set_neuron_voltages_and_screen_shot", [](F32) {});
  // headless additions
  m.def("tick", &Tick);
  m.def("positions", [](int id) { return toNp(mWorms[id]->GetSoftWorld()->GetPositions()); });
  m.def("states", [](int id) { return toNp(mWorms[id]->GetStates()); });
}
