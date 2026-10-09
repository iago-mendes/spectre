// Distributed under the MIT License.
// See LICENSE.txt for details.

// DEBUG (umbrella zeta, research experiments/slow_root_accuracy): the slow
// magnetosonic speeds near the slow/entropy degeneracy, and a timing helper.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/Hydro/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace {
// The slow pair near the slow/entropy degeneracy (research
// experiments/slow_root_accuracy, umbrella zeta, 2026-10-08). Two radial faces
// of the Del Zanna jet's static ambient, recorded by the HLLEM face probe
// (a2_probe_t575 records 43196 and 309475): v = 0, B_r ~ 3e-5 and 9e-9. The
// expected slow speeds are the roots of the same quartic at 60 digits
// (analysis/rmhd_speed_oracle.py); v_n = 0, so they are +-mu.
tnsr::i<DataVector, 9> slow_test_speeds(
    const double rho, const double eps, const double h, const double w,
    const std::array<double, 3>& v, const std::array<double, 3>& b,
    const std::array<double, 3>& n,
    const grmhd::ValenciaDivClean::SlowMagnetosonicSpeedMethod method,
    const double adiabatic_index = 1.6666666666666667) {
  const EquationsOfState::IdealFluid<true> eos(adiabatic_index, 0.0);
  tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity{1_st, 0.0};
  tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field{1_st, 0.0};
  tnsr::i<DataVector, 3> unit_normal{1_st, 0.0};
  tnsr::ii<DataVector, 3, Frame::Inertial> flat_metric{1_st, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    spatial_velocity.get(i) = gsl::at(v, i);
    magnetic_field.get(i) = gsl::at(b, i);
    unit_normal.get(i) = gsl::at(n, i);
    flat_metric.get(i, i) = 1.0;
  }
  tnsr::i<DataVector, 9> speeds{1_st, 0.0};
  grmhd::ValenciaDivClean::characteristic_speeds_mhd(
      make_not_null(&speeds), spatial_velocity, magnetic_field,
      Scalar<DataVector>{DataVector{1_st, rho}},
      Scalar<DataVector>{DataVector{1_st, eps}},
      Scalar<DataVector>{DataVector{1_st, w}},
      Scalar<DataVector>{DataVector{1_st, h}}, flat_metric, unit_normal, eos,
      method);
  return speeds;
}

void test_slow_speeds_comoving() {
  using grmhd::ValenciaDivClean::MhdSpeed;
  using grmhd::ValenciaDivClean::SlowMagnetosonicSpeedMethod;
  struct Face {
    double rho, eps, h;
    std::array<double, 3> b;
    double exact_mu;      // 60 digits, rounded
    double prototype_mu;  // analysis/slow_root_accuracy.py comoving_hybrid
  };
  const std::array<Face, 2> faces{
      {{10.00000147497365,
        0.0014992494598037663,
        1.002498749099673,
        {{2.997194571166376e-05, 0.10007501873759235, 0.0}},
        7.480833985711296863266183e-06,
        7.480833985711296e-06},
       {10.0,
        0.001499999483693053,
        1.0024999991394885,
        {{-8.90989296669351e-09, 0.1000000516306809, 0.0}},
        2.224693536551416198672102e-09,
        2.224693536551416e-09}}};
  const std::array<double, 3> v{{0.0, 0.0, 0.0}};
  const std::array<double, 3> n{{-1.0, -0.0, -0.0}};
  for (const auto& face : faces) {
    CAPTURE(face.exact_mu);
    const auto comoving =
        slow_test_speeds(face.rho, face.eps, face.h, 1.0, v, face.b, n,
                         SlowMagnetosonicSpeedMethod::ReducedQuadraticComoving);
    const auto by_default = [&face, &v, &n]() {
      const EquationsOfState::IdealFluid<true> eos(1.6666666666666667, 0.0);
      tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity{1_st, 0.0};
      tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field{1_st, 0.0};
      tnsr::i<DataVector, 3> unit_normal{1_st, 0.0};
      tnsr::ii<DataVector, 3, Frame::Inertial> flat_metric{1_st, 0.0};
      for (size_t i = 0; i < 3; ++i) {
        spatial_velocity.get(i) = gsl::at(v, i);
        magnetic_field.get(i) = gsl::at(face.b, i);
        unit_normal.get(i) = gsl::at(n, i);
        flat_metric.get(i, i) = 1.0;
      }
      tnsr::i<DataVector, 9> speeds{1_st, 0.0};
      grmhd::ValenciaDivClean::characteristic_speeds_mhd(
          make_not_null(&speeds), spatial_velocity, magnetic_field,
          Scalar<DataVector>{DataVector{1_st, face.rho}},
          Scalar<DataVector>{DataVector{1_st, face.eps}},
          Scalar<DataVector>{DataVector{1_st, 1.0}},
          Scalar<DataVector>{DataVector{1_st, face.h}}, flat_metric,
          unit_normal, eos);
      return speeds;
    }();
    const auto reduced =
        slow_test_speeds(face.rho, face.eps, face.h, 1.0, v, face.b, n,
                         SlowMagnetosonicSpeedMethod::ReducedQuadratic);
    const double sm = get<MhdSpeed::SlowMagnetosonicMinus>(comoving)[0];
    const double sp = get<MhdSpeed::SlowMagnetosonicPlus>(comoving)[0];
    const double rm = get<MhdSpeed::SlowMagnetosonicMinus>(reduced)[0];
    const double rp = get<MhdSpeed::SlowMagnetosonicPlus>(reduced)[0];
    std::cout << std::setprecision(17) << "slow speeds, comoving: " << sm << " "
              << sp << "; reduced quadratic: " << rm << " " << rp
              << "; exact: -+" << face.exact_mu << "\n";
    CAPTURE(sm);
    CAPTURE(sp);
    CAPTURE(rm);
    CAPTURE(rp);
    // the default method is the comoving one
    CHECK(get<MhdSpeed::SlowMagnetosonicMinus>(by_default)[0] == sm);
    CHECK(get<MhdSpeed::SlowMagnetosonicPlus>(by_default)[0] == sp);
    // accurate relative to the distance from v_n = 0
    CHECK(std::abs(sm + face.exact_mu) <= 1.0e-6 * face.exact_mu);
    CHECK(std::abs(sp - face.exact_mu) <= 1.0e-6 * face.exact_mu);
    // the Python prototype (the research repo's comoving_hybrid)
    CHECK(std::abs(sm + face.prototype_mu) <= 1.0e-14 * face.exact_mu);
    CHECK(std::abs(sp - face.prototype_mu) <= 1.0e-14 * face.exact_mu);
    // ... and the reduced quadratic alone is not (the test discriminates)
    CHECK(std::max(std::abs(rm + face.exact_mu), std::abs(rp - face.exact_mu)) >
          1.0e-4 * face.exact_mu);
    // the fast pair, Alfven pair and entropy are untouched
    for (const size_t i : {0_st, 1_st, 2_st, 4_st, 6_st, 7_st, 8_st}) {
      CAPTURE(i);
      CHECK(comoving.get(i)[0] == reduced.get(i)[0]);
    }
  }
  // A Balsara-1 state (Gamma = 2, t = 0.4, normal z, B_n ~ 2e-7) where the
  // reduced quadratic returns slow_minus = slow_plus (1.2025e-7) and a Newton
  // iteration from that seed leaves the light cone (|lambda| up to 6.8e3)
  // unless it is guarded; run with FPEs trapping. 60-digit slow speeds from
  // analysis/rmhd_speed_oracle.py.
  {
    const auto speeds = slow_test_speeds(
        0.6953676820816324, 0.6997190337712661, 2.3994380675425324,
        1.0474884323238665,
        {{0.29653056001033645, -0.02617931983258507, 1.2840727654936489e-07}},
        {{0.500000427349988, 0.7161462537884182, 1.9845159331757503e-07}},
        {{0.0, 0.0, 1.0}},
        SlowMagnetosonicSpeedMethod::ReducedQuadraticComoving, 2.0);
    const double vn = 1.284072765493648850494078e-07;
    const double exact_minus = 1.322726028842970510357819e-08;
    const double exact_plus = 2.272776457004750699634155e-07;
    const double sm = get<MhdSpeed::SlowMagnetosonicMinus>(speeds)[0];
    const double sp = get<MhdSpeed::SlowMagnetosonicPlus>(speeds)[0];
    CAPTURE(sm);
    CAPTURE(sp);
    CHECK(std::abs(sm - exact_minus) <= 1.0e-6 * std::abs(exact_minus - vn));
    CHECK(std::abs(sp - exact_plus) <= 1.0e-6 * std::abs(exact_plus - vn));
  }
  // A subnormal B_n (3.5e-323 after scaling): a Balsara-1 transverse face at
  // rest, where binary 2e3b417cf trapped an FPE (overflow of the debug
  // interlacing slack). The slow pair is v_n exactly. FPEs trap here.
  {
    const auto speeds = slow_test_speeds(
        1.0, 1.0, 3.0, 1.0, {{0.0, 0.0, 0.0}},
        {{0.5, 1.0, 3.458459520888726e-323 * sqrt(3.0)}}, {{0.0, 0.0, 1.0}},
        SlowMagnetosonicSpeedMethod::ReducedQuadraticComoving, 2.0);
    CHECK(get<MhdSpeed::SlowMagnetosonicMinus>(speeds)[0] == 0.0);
    CHECK(get<MhdSpeed::SlowMagnetosonicPlus>(speeds)[0] == 0.0);
  }
  // B_n = 0: the slow pair is v_n exactly (at rest and moving)
  for (const auto& vel : {std::array<double, 3>{{0.0, 0.0, 0.0}},
                          std::array<double, 3>{{0.3, 0.2, 0.1}}}) {
    double v2 = 0.0;
    for (const double x : vel) {
      v2 += x * x;
    }
    const double w = 1.0 / sqrt(1.0 - v2);
    const double rho = 1.3;
    const double eps = 0.4;
    const double h = 1.0 + eps + (1.6666666666666667 - 1.0) * eps;
    const auto speeds = slow_test_speeds(
        rho, eps, h, w, vel, {{0.0, 0.7, -0.2}}, {{1.0, 0.0, 0.0}},
        SlowMagnetosonicSpeedMethod::ReducedQuadraticComoving);
    CAPTURE(vel[0]);
    CHECK(get<MhdSpeed::SlowMagnetosonicMinus>(speeds)[0] == vel[0]);
    CHECK(get<MhdSpeed::SlowMagnetosonicPlus>(speeds)[0] == vel[0]);
  }
}

// The slow left eigenvectors on a Del Zanna jet ambient face with B_r ~ 2.5e-6
// (a4_v2_probe_t11, a face of a late ambient blob; research
// experiments/delzanna_jet/ambient_blobs, L2), against the 60-digit values of
// the same formulas at the 60-digit speeds (analysis/
// magnetosonic_eigvec_highprec.py). There Z a^2 = 4e-12: a floor of Z a^2 at
// 1e-12 Z made h_1 (components D and tau) wrong.
void test_slow_left_eigenvector_small_bn() {
  using grmhd::ValenciaDivClean::MhdSpeed;
  const EquationsOfState::IdealFluid<true> eos(1.6666666666666667, 0.0);
  tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity{1_st, 0.0};
  tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field{1_st, 0.0};
  tnsr::i<DataVector, 3> unit_normal{1_st, 0.0};
  tnsr::ii<DataVector, 3, Frame::Inertial> flat_metric{1_st, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    flat_metric.get(i, i) = 1.0;
  }
  get<0>(magnetic_field) = -2.5453750302388786e-06;
  get<1>(magnetic_field) = 0.0999873353408945;
  get<0>(unit_normal) = -1.0;
  const Scalar<DataVector> rho{DataVector{1_st, 10.000007889440843}};
  const Scalar<DataVector> eps{DataVector{1_st, 0.0015001258924477047}};
  const Scalar<DataVector> w{DataVector{1_st, 1.0}};
  const Scalar<DataVector> h{DataVector{1_st, 1.002500209820746}};
  tnsr::i<DataVector, 9> speeds{1_st, 0.0};
  grmhd::ValenciaDivClean::characteristic_speeds_mhd(
      make_not_null(&speeds), spatial_velocity, magnetic_field, rho, eps, w, h,
      flat_metric, unit_normal, eos);
  // 60 digits: 6.3558977665308129955e-7
  CHECK(std::abs(get<MhdSpeed::SlowMagnetosonicPlus>(speeds)[0] -
                 6.3558977665308129955e-07) <= 1.0e-15 * 6.36e-07);
  tnsr::ij<DataVector, 9> modes{1_st, 0.0};
  tnsr::IJ<DataVector, 9> projectors{1_st, 0.0};
  grmhd::ValenciaDivClean::characteristic_eigenvectors_mhd(
      make_not_null(&modes), make_not_null(&projectors), speeds,
      spatial_velocity, magnetic_field, rho, eps, w, h, flat_metric,
      unit_normal, eos);
  // slow+ (wave 5); slow- has the components 0, 1, 8 with the other sign
  const std::array<double, 9> exact_plus{
      {-3.812249815123043e-7, 0.039942482736900092, 0.0, -2.538381851073254e-9,
       -0.13996875256624061, 0.0, 3.1e-61, 0.39986481376900771,
       0.0039937424173812001}};
  for (size_t comp = 0; comp < 9; ++comp) {
    CAPTURE(comp);
    CAPTURE(projectors.get(5, comp)[0]);
    CHECK(std::abs(projectors.get(5, comp)[0] - gsl::at(exact_plus, comp)) <=
          1.0e-9);
    const double sign = (comp == 0 or comp == 1 or comp == 8) ? -1.0 : 1.0;
    CAPTURE(projectors.get(3, comp)[0]);
    CHECK(std::abs(projectors.get(3, comp)[0] -
                   sign * gsl::at(exact_plus, comp)) <= 1.0e-9);
  }
}

// Timing of characteristic_speeds_mhd, ReducedQuadratic vs
// ReducedQuadraticComoving (no-op unless SPECTRE_SLOW_SPEED_TIMING is set):
// 20,000 points, half static jet-ambient states (rho ~ 10, p ~ 0.01, B_z = 0.1,
// small random B_r), half random admissible states; flat metric, normal -x.
// Prints the minimum over repetitions for each method.
void time_slow_speed_methods() {
  if (std::getenv("SPECTRE_SLOW_SPEED_TIMING") == nullptr) {
    return;
  }
  using grmhd::ValenciaDivClean::SlowMagnetosonicSpeedMethod;
  MAKE_GENERATOR(generator);
  namespace helper = TestHelpers::hydro;
  const auto nn_gen = make_not_null(&generator);
  constexpr size_t half = 10000;
  constexpr size_t num_points = 2 * half;
  const DataVector used_for_size(half);
  tnsr::ii<DataVector, 3, Frame::Inertial> flat_half{half, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    flat_half.get(i, i) = 1.0;
  }
  const auto rho_r = helper::random_density(nn_gen, used_for_size);
  const auto eps_r =
      helper::random_specific_internal_energy(nn_gen, used_for_size);
  const auto w_r = helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto v_r = helper::random_velocity(nn_gen, w_r, flat_half);
  const EquationsOfState::IdealFluid<true> eos(1.6666666666666667, 0.0);
  const auto p_r = eos.pressure_from_density_and_energy(rho_r, eps_r);
  const auto b_r = helper::random_magnetic_field(nn_gen, p_r, flat_half);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);

  Scalar<DataVector> rho{num_points};
  Scalar<DataVector> eps{num_points};
  Scalar<DataVector> w{num_points};
  Scalar<DataVector> h{num_points};
  tnsr::I<DataVector, 3, Frame::Inertial> v{num_points, 0.0};
  tnsr::I<DataVector, 3, Frame::Inertial> b{num_points, 0.0};
  tnsr::ii<DataVector, 3, Frame::Inertial> flat{num_points, 0.0};
  tnsr::i<DataVector, 3> n{num_points, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    flat.get(i, i) = 1.0;
  }
  get<0>(n) = -1.0;
  for (size_t p = 0; p < half; ++p) {
    // jet ambient
    get(rho)[p] = 10.0 * (1.0 + 1.0e-5 * unit(generator));
    get(eps)[p] = 0.0015 * (1.0 + 1.0e-3 * unit(generator));
    get(w)[p] = 1.0;
    get<0>(b)[p] = 3.0e-5 * unit(generator);
    get<1>(b)[p] = 0.1;
    // random
    const size_t q = half + p;
    get(rho)[q] = get(rho_r)[p];
    get(eps)[q] = get(eps_r)[p];
    get(w)[q] = get(w_r)[p];
    for (size_t i = 0; i < 3; ++i) {
      v.get(i)[q] = v_r.get(i)[p];
      b.get(i)[q] = b_r.get(i)[p];
    }
  }
  for (size_t p = 0; p < num_points; ++p) {
    get(h)[p] = 1.0 + get(eps)[p] + (1.6666666666666667 - 1.0) * get(eps)[p];
  }
  tnsr::i<DataVector, 9> speeds{num_points, 0.0};
  double best_reduced = std::numeric_limits<double>::infinity();
  double best_comoving = std::numeric_limits<double>::infinity();
  for (size_t rep = 0; rep < 40; ++rep) {
    for (const auto method :
         {SlowMagnetosonicSpeedMethod::ReducedQuadratic,
          SlowMagnetosonicSpeedMethod::ReducedQuadraticComoving}) {
      const auto start = std::chrono::steady_clock::now();
      grmhd::ValenciaDivClean::characteristic_speeds_mhd(
          make_not_null(&speeds), v, b, rho, eps, w, h, flat, n, eos, method);
      const double seconds = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
      double& best = method == SlowMagnetosonicSpeedMethod::ReducedQuadratic
                         ? best_reduced
                         : best_comoving;
      best = std::min(best, seconds);
    }
  }
  std::cout << std::setprecision(6) << "characteristic_speeds_mhd timing, "
            << num_points << " points, min of 40: ReducedQuadratic "
            << best_reduced * 1e9 / static_cast<double>(num_points)
            << " ns/point, ReducedQuadraticComoving "
            << best_comoving * 1e9 / static_cast<double>(num_points)
            << " ns/point, ratio " << best_comoving / best_reduced << "\n";
}
}  // namespace

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.SlowSpeedsComoving",
                  "[Unit][Evolution]") {
  test_slow_speeds_comoving();
  test_slow_left_eigenvector_small_bn();
  const ScopedFpeState disable_fpes(false);  // random states, as elsewhere
  time_slow_speed_methods();  // no-op unless SPECTRE_SLOW_SPEED_TIMING is set
}
