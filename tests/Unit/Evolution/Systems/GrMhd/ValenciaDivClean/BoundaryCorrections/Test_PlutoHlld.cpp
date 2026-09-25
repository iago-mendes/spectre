// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <limits>
#include <thread>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/PlutoHlld.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/pluto/pluto_hlld_shim.h"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"

namespace {
namespace helpers = TestHelpers::evolution::dg;

// The single most important test in this file: does the vendored PLUTO still
// produce the flux the ORIGINAL, unmodified PLUTO binary produces?
//
// These values come from PLUTO v4.4's own HLLD_Solver driven directly by a
// standalone C program (see meetings/2026-09-03/PROGRESS_taskA.md), on the ST1
// (Balsara-1) interface with gamma = 2:
//   left  (rho, p, v, B) = (1,     1,   0, (0.5,  1, 0))
//   right (rho, p, v, B) = (0.125, 0.1, 0, (0.5, -1, 0))
// If a future edit to the vendored sources changes the answer, this fails.
void test_matches_reference_pluto() {
  // PLUTO must be called with FP-exception trapping OFF. Its allocation and
  // root-finding paths legitimately raise FPE (here: SIGFPE inside Array2D via
  // MakeState), and the unit-test framework traps by default. PlutoHlld.cpp
  // does this for the production path; a direct call to the shim must do it
  // too, or the process dies inside PLUTO's memory setup.
  const ScopedFpeState fpe_off(false);
  const std::array<double, 8> vl{{1.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0}};
  const std::array<double, 8> vr{{0.125, 0.0, 0.0, 0.0, 0.5, -1.0, 0.0, 0.1}};
  const std::array<double, 8> expected{{
      2.4453400539364434e-01,   // RHO
      -9.6296630705276220e-01,  // MX1 (normal momentum, pressure carried apart)
      -3.6853136801338132e-01,  // MX2
      0.0,                      // MX3
      0.0,                      // BX1 (normal field: no flux)
      2.4936651533216450e-01,   // BX2
      0.0,                      // BX3
      4.9904982334815257e-01}};  // ENG = E - D (reduced energy)
  const double expected_press = 1.6250000000000000e+00;

  // Several batch sizes at once: 1 exercises the scalar path, 169 a realistic
  // face, 2500 forces the shim to chunk (its internal capacity is 1024) --
  // which is where an off-by-one in PLUTO's +1-offset right state showed up.
  for (const int npts : {1, 169, 2500}) {
    std::vector<double> batch_l(static_cast<size_t>(npts) * 8);
    std::vector<double> batch_r(static_cast<size_t>(npts) * 8);
    std::vector<double> flux(static_cast<size_t>(npts) * 8);
    std::vector<double> press(static_cast<size_t>(npts));
    for (int i = 0; i < npts; ++i) {
      for (size_t nv = 0; nv < 8; ++nv) {
        batch_l[static_cast<size_t>(i) * 8 + nv] = gsl::at(vl, nv);
        batch_r[static_cast<size_t>(i) * 8 + nv] = gsl::at(vr, nv);
      }
    }
    CHECK(pluto_hlld_flux(npts, batch_l.data(), batch_r.data(), 2.0,
                          flux.data(), press.data()) == 0);
    for (int i = 0; i < npts; ++i) {
      for (size_t nv = 0; nv < 8; ++nv) {
        CHECK(flux[static_cast<size_t>(i) * 8 + nv] ==
              approx(gsl::at(expected, nv)));
      }
      CHECK(press[static_cast<size_t>(i)] == approx(expected_press));
    }
  }
}

// Regression guard for the bug that made every A2 evolution wrong: PLUTO keeps
// lazily-allocated function statics (hll_speed.c's SL/SR scratch, mappers.c's
// enthalpy buffer, arrays.c's allocation registry, ...). SpECTRE runs Charm++
// in SMP mode, so unless every one of them is _Thread_local, worker threads
// race and the HLL wave speeds come out garbage. The symptom was NOT a crash:
// it was a plausible-looking, slightly smeared profile and ~19 MB of
// "RMHD_EnergySolve() failed" per run.
//
// The test must give each thread a DIFFERENT state. An earlier version handed
// every thread the same state, which cannot detect these races at all --
// racing threads simply write identical values. Verified to have real
// discriminating power: reverting just the hll_speed.c fix makes this report
// deviations up to 1e-1.
void test_thread_safety() {
  const ScopedFpeState fpe_off(false);
  constexpr int num_threads = 8;
  constexpr int npts = 169;
  const auto make_state = [](int seed, std::vector<double>* l,
                             std::vector<double>* r) {
    std::mt19937 gen(static_cast<unsigned>(seed) + 1u);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::array<double, 8> sl{}, sr{};
    sl[0] = 0.5 + dist(gen); sr[0] = 0.5 + dist(gen);
    for (size_t k = 1; k < 4; ++k) {
      sl[k] = -0.3 + 0.6 * dist(gen); sr[k] = -0.3 + 0.6 * dist(gen);
    }
    sl[4] = -1.0 + 2.0 * dist(gen); sr[4] = sl[4];   // normal B is continuous
    for (size_t k = 5; k < 7; ++k) {
      sl[k] = -1.0 + 2.0 * dist(gen); sr[k] = -1.0 + 2.0 * dist(gen);
    }
    sl[7] = 0.5 + dist(gen); sr[7] = 0.5 + dist(gen);
    l->resize(npts * 8); r->resize(npts * 8);
    for (int i = 0; i < npts; ++i) {
      for (size_t k = 0; k < 8; ++k) {
        (*l)[static_cast<size_t>(i) * 8 + k] = gsl::at(sl, k);
        (*r)[static_cast<size_t>(i) * 8 + k] = gsl::at(sr, k);
      }
    }
  };
  // single-threaded reference, one per distinct state
  std::vector<std::vector<double>> reference(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    std::vector<double> l, r, f(npts * 8), pr(npts);
    make_state(t, &l, &r);
    pluto_hlld_flux(npts, l.data(), r.data(), 5.0 / 3.0, f.data(), pr.data());
    reference[static_cast<size_t>(t)] = f;
  }
  std::vector<double> worst(num_threads, 0.0);
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([t, &reference, &worst, &make_state]() {
      std::vector<double> l, r, f(npts * 8), pr(npts);
      make_state(t, &l, &r);
      pluto_hlld_flux(npts, l.data(), r.data(), 5.0 / 3.0, f.data(), pr.data());
      double w = 0.0;
      for (size_t i = 0; i < f.size(); ++i) {
        const double ref = reference[static_cast<size_t>(t)][i];
        w = std::max(w, std::abs(f[i] - ref) / (1.0 + std::abs(ref)));
      }
      worst[static_cast<size_t>(t)] = w;
    });
  }
  for (auto& th : threads) { th.join(); }
  for (int t = 0; t < num_threads; ++t) {
    CHECK(worst[static_cast<size_t>(t)] == 0.0);
  }
}

// The interface state: PlutoHlld must take its fast-magnetosonic bounds from
// the two sides SEPARATELY, never from an average of them.
//
// This is the caller-side companion to
// Test_Characteristics.cpp's test_superluminal_interface_sound_speed. That one
// pins the guard inside characteristic_speeds_mhd -- what happens once a
// superluminal sound speed has already been manufactured. This one pins the
// construction, i.e. that it is never manufactured here in the first place.
//
// Until 2026-09-15 dg_boundary_terms averaged rho, eps and p INDEPENDENTLY
// across the face and built ONE eigensystem at that triple. The triple
// satisfies no equation of state: characteristic_speeds_mhd forms
//     c_s^2 = (chi + kappa p / rho^2) / h
// with chi and kappa at (rho_avg, eps_avg) but h from p_avg, and for an ideal
// fluid that tends to Gamma (Gamma - 1) as p_avg/rho_avg -> 0, which exceeds 1
// for every Gamma above the golden ratio (1+sqrt 5)/2 = 1.618. A superluminal
// c_s^2 pushes the magnetosonic quartic's extremal roots outside the light
// cone, the +/-1 Newton seeds then no longer separate the fast pair, and the
// deflation that follows is not a factorization -- the -7.8678 discriminant
// that aborted the production Del Zanna jet (job 74139).
//
// The two-sided (Davis) bound cannot do this, because each side IS a state.
// The test recovers the bounds the solver actually used, from the flux it
// returns, and requires them to be exactly the two-sided combination of each
// side's own fast speeds.
//
// Recovering the bounds. TildeYe is the one variable PlutoHlld carries with
// the fast bounds and never overwrites with PLUTO's fan, so its weak-form HLL
// flux
//     G = (l_max nf_int + l_min nf_ext + l_max l_min (u_ext - u_int))
//         / (l_max - l_min)
// is a direct probe. Three settings of (nf_int, nf_ext, u_ext - u_int) give
//     A = (1, 0, 0) -> l_max / d,   B = (0, 1, 0) -> l_min / d,
//     C = (0, 0, 1) -> l_max l_min / d,      d = l_max - l_min,
// hence d = C / (A B), l_max = A d, l_min = B d, exactly.
void test_interface_bounds_are_two_sided(
    const double adiabatic_index, const double rho_left,
    const double pressure_left, const double rho_right,
    const double pressure_right, const std::array<double, 3>& velocity,
    const std::array<double, 3>& b_field_left,
    const std::array<double, 3>& b_field_right, const std::string& what) {
  CAPTURE(what);
  CAPTURE(adiabatic_index);
  const auto eos =
      EquationsOfState::IdealFluid<true>{adiabatic_index}.promote_to_3d_eos();

  const double eps_left = pressure_left / ((adiabatic_index - 1.0) * rho_left);
  const double eps_right =
      pressure_right / ((adiabatic_index - 1.0) * rho_right);
  const double v_squared = velocity[0] * velocity[0] +
                           velocity[1] * velocity[1] +
                           velocity[2] * velocity[2];
  REQUIRE(v_squared < 1.0);
  const double lorentz = 1.0 / sqrt(1.0 - v_squared);

  // The premise. Averaging the three primitives independently -- what this
  // solver used to do -- puts the interface sound speed ABOVE the light speed,
  // while each side and the consistent state at (rho_avg, eps_avg) are both
  // comfortably subluminal. If a later change makes the premise false the test
  // would pass vacuously, so assert it -- and assert it through the EXACT
  // condition rather than through a rule of thumb.
  //
  // The exact condition. With T = p/rho, S = (T_L + T_R)/2,
  // k(Gamma) = (Gamma^2 - Gamma - 1)/(Gamma - 1), and
  //     C = (T_L - T_R)(rho_R - rho_L) / (2 (rho_L + rho_R))
  //       = S - p_avg/rho_avg,
  // the naive interface sound speed exceeds 1 exactly when
  //     M = S (eta - 1 + k) > 1,       eta = C/S in (-1, 1).
  // C is minus the normalised rho-T covariance, so M > 0 needs the RAREFIED
  // side to be the HOT one. Gamma (Gamma - 1) > 1, i.e. Gamma > the golden
  // ratio 1.618, is the degenerate case 1 - k < 1 and is necessary but nowhere
  // near sufficient. Two consequences worth having in a test rather than in a
  // document:
  //   * a high eps_avg is NOT the criterion. Balsara-3 has eps_avg = 750, 33x
  //     the jet's, and is perfectly safe, because rho_L = rho_R makes eta = 0
  //     and M = -417.
  //   * the Gamma = 2 tube suite has the WRONG SIGN of eta -- its dense side is
  //     also its hot side -- so its worst evolved face reaches M = 8.2e-4
  //     against the jet's 2.466. The Gamma = 2 case below is therefore built to
  //     give M > 1 deliberately (a 1000:1 density contrast at equal pressure),
  //     not by assuming a hot state suffices.
  const double rho_avg = 0.5 * (rho_left + rho_right);
  const double eps_avg = 0.5 * (eps_left + eps_right);
  const double p_avg = 0.5 * (pressure_left + pressure_right);
  const double temperature_left = pressure_left / rho_left;
  const double temperature_right = pressure_right / rho_right;
  const double mean_temperature =
      0.5 * (temperature_left + temperature_right);
  const double covariance = (temperature_left - temperature_right) *
                            (rho_right - rho_left) /
                            (2.0 * (rho_left + rho_right));
  // The two-point identity the condition rests on.
  CHECK(mean_temperature - p_avg / rho_avg == approx(covariance));
  const double k_gamma = (adiabatic_index * adiabatic_index - adiabatic_index -
                          1.0) / (adiabatic_index - 1.0);
  const double superluminality_margin =
      mean_temperature * (covariance / mean_temperature - 1.0 + k_gamma);
  CAPTURE(superluminality_margin);
  CHECK(superluminality_margin > 1.0);

  const double naive_sound_speed_squared =
      adiabatic_index * (adiabatic_index - 1.0) * eps_avg /
      (1.0 + eps_avg + p_avg / rho_avg);
  const double consistent_sound_speed_squared =
      adiabatic_index * (adiabatic_index - 1.0) * eps_avg /
      (1.0 + adiabatic_index * eps_avg);
  CAPTURE(naive_sound_speed_squared);
  CAPTURE(consistent_sound_speed_squared);
  // M > 1 and cs^2_naive > 1 are the same statement; check they agree, which
  // validates the instrument as well as the premise.
  CHECK(naive_sound_speed_squared > 1.0);
  CHECK(consistent_sound_speed_squared < 1.0);

  tnsr::ii<DataVector, 3, Frame::Inertial> flat_metric{1_st, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    flat_metric.get(i, i) = 1.0;
  }
  const Scalar<DataVector> lapse{DataVector{1_st, 1.0}};
  const tnsr::I<DataVector, 3, Frame::Inertial> shift{1_st, 0.0};
  const Scalar<DataVector> electron_fraction{DataVector{1_st, 0.1}};

  // Each side's own fast speeds, along the OUTWARD normal of that side: +x for
  // the interior, -x for the exterior (a neighbour always packages against its
  // own outward normal). This is the reference the solver has to reproduce.
  const auto own_fast_speeds = [&](const double rho, const double eps,
                                   const double pressure,
                                   const std::array<double, 3>& b_field,
                                   const double normal_sign) {
    const Scalar<DataVector> rest_mass_density{DataVector{1_st, rho}};
    const Scalar<DataVector> specific_internal_energy{DataVector{1_st, eps}};
    const Scalar<DataVector> pressure_scalar{DataVector{1_st, pressure}};
    const Scalar<DataVector> lorentz_factor{DataVector{1_st, lorentz}};
    const Scalar<DataVector> specific_enthalpy =
        hydro::relativistic_specific_enthalpy(
            rest_mass_density, specific_internal_energy, pressure_scalar);
    tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity{1_st, 0.0};
    tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field{1_st, 0.0};
    for (size_t i = 0; i < 3; ++i) {
      spatial_velocity.get(i) = gsl::at(velocity, i);
      magnetic_field.get(i) = gsl::at(b_field, i);
    }
    tnsr::i<DataVector, 3> unit_normal{1_st, 0.0};
    get<0>(unit_normal) = normal_sign;
    const auto speeds = grmhd::ValenciaDivClean::
        characteristic_speeds_approximate_mhd(
            rest_mass_density, electron_fraction, specific_internal_energy,
            specific_enthalpy, spatial_velocity, lorentz_factor,
            magnetic_field, lapse, shift, flat_metric, unit_normal, *eos);
    // Indices 1 and 7 are the ingoing and outgoing fast-magnetosonic speeds.
    return std::array<double, 2>{{speeds[7][0], speeds[1][0]}};
  };
  const auto own_int =
      own_fast_speeds(rho_left, eps_left, pressure_left, b_field_left, 1.0);
  const auto own_ext = own_fast_speeds(rho_right, eps_right, pressure_right,
                                       b_field_right, -1.0);
  const double expected_fast_max =
      std::max(0.0, std::max(own_int[0], -own_ext[1]));
  const double expected_fast_min =
      std::min(0.0, std::min(own_int[1], -own_ext[0]));
  CAPTURE(expected_fast_max);
  CAPTURE(expected_fast_min);
  // Each side is a real state, so its fast speeds are STRICTLY inside the
  // light cone. That is what separates the two constructions: with the
  // averaged state the sound speed is superluminal, the guard in
  // characteristic_speeds_mhd saturates it at exactly 1, and c_s^2 = 1 is an
  // exact factorization of the magnetosonic quartic -- lambda = +/-1 are then
  // roots identically -- so the replaced code returns bounds of exactly
  // (-1, +1) here. How wide the margin is depends on the case and is captured
  // rather than hard-coded: at Gamma = 5/3 the hot side's own fast speed is
  // ~0.59, but at Gamma = 2 a state hot enough to give M > 1 is necessarily
  // close to luminal on its hot side, because that equation of state's own
  // sound speed tends to Gamma - 1 = 1 as eps grows. The separation there is
  // a few percent, not a factor of two, and the exact comparison below is what
  // carries the test.
  REQUIRE(expected_fast_max > 0.0);
  REQUIRE(expected_fast_min < 0.0);
  REQUIRE(expected_fast_max < 1.0);
  REQUIRE(expected_fast_min > -1.0);
  CAPTURE(1.0 - expected_fast_max);
  CAPTURE(1.0 + expected_fast_min);

  const grmhd::ValenciaDivClean::BoundaryCorrections::PlutoHlld solver{1.0e-30,
                                                                      1.0e-8};

  // Package one side. `tilde_ye` and the TildeYe flux are the probe; every
  // other conserved variable is irrelevant here (PLUTO's fan overwrites
  // TildeD, TildeTau, TildeS and the tangential field, and TildePhi rides the
  // light-speed bounds), so they are set to benign values.
  struct Packaged {
    Scalar<DataVector> tilde_d{DataVector{1_st, 0.0}};
    Scalar<DataVector> tilde_ye{DataVector{1_st, 0.0}};
    Scalar<DataVector> tilde_tau{DataVector{1_st, 0.0}};
    tnsr::i<DataVector, 3, Frame::Inertial> tilde_s{1_st, 0.0};
    tnsr::I<DataVector, 3, Frame::Inertial> tilde_b{1_st, 0.0};
    Scalar<DataVector> tilde_phi{DataVector{1_st, 0.0}};
    Scalar<DataVector> nf_tilde_d{DataVector{1_st, 0.0}};
    Scalar<DataVector> nf_tilde_ye{DataVector{1_st, 0.0}};
    Scalar<DataVector> nf_tilde_tau{DataVector{1_st, 0.0}};
    tnsr::i<DataVector, 3, Frame::Inertial> nf_tilde_s{1_st, 0.0};
    tnsr::I<DataVector, 3, Frame::Inertial> nf_tilde_b{1_st, 0.0};
    Scalar<DataVector> nf_tilde_phi{DataVector{1_st, 0.0}};
    Scalar<DataVector> largest_out{DataVector{1_st, 0.0}};
    Scalar<DataVector> largest_in{DataVector{1_st, 0.0}};
    Scalar<DataVector> fast_out{DataVector{1_st, 0.0}};
    Scalar<DataVector> fast_in{DataVector{1_st, 0.0}};
    tnsr::i<DataVector, 3, Frame::Inertial> normal{1_st, 0.0};
    Scalar<DataVector> flatness{DataVector{1_st, 0.0}};
    Scalar<DataVector> rho{DataVector{1_st, 0.0}};
    tnsr::I<DataVector, 3, Frame::Inertial> velocity{1_st, 0.0};
    Scalar<DataVector> pressure{DataVector{1_st, 0.0}};
    Scalar<DataVector> lorentz_factor{DataVector{1_st, 0.0}};
    Scalar<DataVector> eps{DataVector{1_st, 0.0}};
  };
  const auto package = [&](const double rho, const double eps,
                           const double pressure,
                           const std::array<double, 3>& b_field,
                           const double normal_sign, const double tilde_ye_in,
                           const double nf_tilde_ye_in) {
    Packaged out{};
    const Scalar<DataVector> rest_mass_density{DataVector{1_st, rho}};
    const Scalar<DataVector> specific_internal_energy{DataVector{1_st, eps}};
    const Scalar<DataVector> pressure_scalar{DataVector{1_st, pressure}};
    const Scalar<DataVector> lorentz_factor{DataVector{1_st, lorentz}};
    const Scalar<DataVector> temperature =
        eos->temperature_from_density_and_energy(
            rest_mass_density, specific_internal_energy, electron_fraction);
    tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity{1_st, 0.0};
    tnsr::i<DataVector, 3, Frame::Inertial> velocity_one_form{1_st, 0.0};
    tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field{1_st, 0.0};
    for (size_t i = 0; i < 3; ++i) {
      spatial_velocity.get(i) = gsl::at(velocity, i);
      velocity_one_form.get(i) = gsl::at(velocity, i);
      magnetic_field.get(i) = gsl::at(b_field, i);
    }
    tnsr::i<DataVector, 3, Frame::Inertial> normal_covector{1_st, 0.0};
    tnsr::I<DataVector, 3, Frame::Inertial> normal_vector{1_st, 0.0};
    get<0>(normal_covector) = normal_sign;
    get<0>(normal_vector) = normal_sign;

    const Scalar<DataVector> tilde_d{DataVector{1_st, rho * lorentz}};
    const Scalar<DataVector> tilde_ye{DataVector{1_st, tilde_ye_in}};
    const Scalar<DataVector> tilde_tau{DataVector{1_st, pressure}};
    const tnsr::i<DataVector, 3, Frame::Inertial> tilde_s{1_st, 0.0};
    const Scalar<DataVector> tilde_phi{DataVector{1_st, 0.0}};
    const tnsr::I<DataVector, 3, Frame::Inertial> zero_flux{1_st, 0.0};
    tnsr::I<DataVector, 3, Frame::Inertial> flux_tilde_ye{1_st, 0.0};
    // n^i F_i must come out as the requested value; the normal is +/- x.
    get<0>(flux_tilde_ye) = normal_sign * nf_tilde_ye_in;
    const tnsr::Ij<DataVector, 3, Frame::Inertial> flux_tilde_s{1_st, 0.0};
    const tnsr::IJ<DataVector, 3, Frame::Inertial> flux_tilde_b{1_st, 0.0};

    solver.dg_package_data(
        make_not_null(&out.tilde_d), make_not_null(&out.tilde_ye),
        make_not_null(&out.tilde_tau), make_not_null(&out.tilde_s),
        make_not_null(&out.tilde_b), make_not_null(&out.tilde_phi),
        make_not_null(&out.nf_tilde_d), make_not_null(&out.nf_tilde_ye),
        make_not_null(&out.nf_tilde_tau), make_not_null(&out.nf_tilde_s),
        make_not_null(&out.nf_tilde_b), make_not_null(&out.nf_tilde_phi),
        make_not_null(&out.largest_out), make_not_null(&out.largest_in),
        make_not_null(&out.fast_out), make_not_null(&out.fast_in),
        make_not_null(&out.normal), make_not_null(&out.flatness),
        make_not_null(&out.rho), make_not_null(&out.velocity),
        make_not_null(&out.pressure), make_not_null(&out.lorentz_factor),
        make_not_null(&out.eps), tilde_d, tilde_ye, tilde_tau, tilde_s,
        magnetic_field, tilde_phi, zero_flux, flux_tilde_ye, zero_flux,
        flux_tilde_s, flux_tilde_b, zero_flux, lapse, shift,
        velocity_one_form, rest_mass_density, electron_fraction, temperature,
        spatial_velocity, specific_internal_energy, pressure_scalar,
        lorentz_factor, normal_covector, normal_vector, std::nullopt,
        std::nullopt, *eos);
    return out;
  };

  // The packaged per-side speeds are this side's own, so they are subluminal
  // by construction. Check that directly before recovering the combination.
  {
    const auto probe_int =
        package(rho_left, eps_left, pressure_left, b_field_left, 1.0, 0.0, 0.0);
    const auto probe_ext = package(rho_right, eps_right, pressure_right,
                                   b_field_right, -1.0, 0.0, 0.0);
    CHECK(get(probe_int.fast_out)[0] == approx(own_int[0]));
    CHECK(get(probe_int.fast_in)[0] == approx(own_int[1]));
    CHECK(get(probe_ext.fast_out)[0] == approx(own_ext[0]));
    CHECK(get(probe_ext.fast_in)[0] == approx(own_ext[1]));
    for (const double speed :
         {get(probe_int.fast_out)[0], get(probe_int.fast_in)[0],
          get(probe_ext.fast_out)[0], get(probe_ext.fast_in)[0]}) {
      CAPTURE(speed);
      CHECK(std::abs(speed) < 1.0);
    }
  }

  // Run the solver three times and read the bounds back out of TildeYe.
  const auto tilde_ye_correction = [&](const double tilde_ye_int,
                                       const double nf_int,
                                       const double tilde_ye_ext,
                                       const double nf_ext) {
    const auto interior = package(rho_left, eps_left, pressure_left,
                                  b_field_left, 1.0, tilde_ye_int, nf_int);
    const auto exterior = package(rho_right, eps_right, pressure_right,
                                  b_field_right, -1.0, tilde_ye_ext, nf_ext);
    Scalar<DataVector> g_d{DataVector{1_st, 0.0}};
    Scalar<DataVector> g_ye{DataVector{1_st, 0.0}};
    Scalar<DataVector> g_tau{DataVector{1_st, 0.0}};
    tnsr::i<DataVector, 3, Frame::Inertial> g_s{1_st, 0.0};
    tnsr::I<DataVector, 3, Frame::Inertial> g_b{1_st, 0.0};
    Scalar<DataVector> g_phi{DataVector{1_st, 0.0}};
    grmhd::ValenciaDivClean::BoundaryCorrections::PlutoHlld::dg_boundary_terms(
        make_not_null(&g_d), make_not_null(&g_ye), make_not_null(&g_tau),
        make_not_null(&g_s), make_not_null(&g_b), make_not_null(&g_phi),
        interior.tilde_d, interior.tilde_ye, interior.tilde_tau,
        interior.tilde_s, interior.tilde_b, interior.tilde_phi,
        interior.nf_tilde_d, interior.nf_tilde_ye, interior.nf_tilde_tau,
        interior.nf_tilde_s, interior.nf_tilde_b, interior.nf_tilde_phi,
        interior.largest_out, interior.largest_in, interior.fast_out,
        interior.fast_in, interior.normal, interior.flatness, interior.rho,
        interior.velocity, interior.pressure, interior.lorentz_factor,
        interior.eps, exterior.tilde_d, exterior.tilde_ye, exterior.tilde_tau,
        exterior.tilde_s, exterior.tilde_b, exterior.tilde_phi,
        exterior.nf_tilde_d, exterior.nf_tilde_ye, exterior.nf_tilde_tau,
        exterior.nf_tilde_s, exterior.nf_tilde_b, exterior.nf_tilde_phi,
        exterior.largest_out, exterior.largest_in, exterior.fast_out,
        exterior.fast_in, exterior.normal, exterior.flatness, exterior.rho,
        exterior.velocity, exterior.pressure, exterior.lorentz_factor,
        exterior.eps, dg::Formulation::WeakInertial);
    return get(g_ye)[0];
  };
  const double probe_a = tilde_ye_correction(0.0, 1.0, 0.0, 0.0);
  const double probe_b = tilde_ye_correction(0.0, 0.0, 0.0, 1.0);
  const double probe_c = tilde_ye_correction(0.0, 0.0, 1.0, 0.0);
  CAPTURE(probe_a);
  CAPTURE(probe_b);
  CAPTURE(probe_c);
  REQUIRE(std::abs(probe_a) > 1.0e-8);
  REQUIRE(std::abs(probe_b) > 1.0e-8);
  const double bound_difference = probe_c / (probe_a * probe_b);
  const double recovered_fast_max = probe_a * bound_difference;
  const double recovered_fast_min = probe_b * bound_difference;
  CAPTURE(recovered_fast_max);
  CAPTURE(recovered_fast_min);

  // THE assertion: the bounds the solver used are the two-sided (Davis)
  // combination of each side's own fast speeds, to round-off. With the
  // averaged interface state they are the light speeds instead, because the
  // averaged triple's sound speed is superluminal and saturates at c.
  CHECK(recovered_fast_max == approx(expected_fast_max));
  CHECK(recovered_fast_min == approx(expected_fast_min));
  // ... and in particular they are NOT the light-speed pair the averaged
  // construction collapses to.
  CHECK(recovered_fast_max < 1.0);
  CHECK(recovered_fast_min > -1.0);
}

void test_interface_state() {
  // 1. The Del Zanna jet's cocoon/ambient contact in miniature, at the jet's
  //    own Gamma = 5/3. These are the two states of
  //    Test_Characteristics.cpp's test_superluminal_interface_sound_speed:
  //    each admissible on its own, four real magnetosonic speeds inside the
  //    light cone, and an independently-averaged interface sound speed of
  //    1.0568.
  test_interface_bounds_are_two_sided(
      5.0 / 3.0, 0.01, 0.3, 5.0, 0.5, {{0.2, -0.9, -0.2}},
      {{-0.3, 0.1, 0.2}}, {{0.1, 0.3, -0.2}}, "Gamma = 5/3, jet cocoon rim");

  // 2. The shock-tube campaign's exposure, built deliberately rather than
  //    assumed. Every tube_ladder run is Gamma = 2 with AlwaysUseSubcells, so
  //    this code path executes on every FD face of every step, and the ceiling
  //    Gamma (Gamma - 1) = 2.0 is nearly twice the jet's 1.111. It has never
  //    fired, and the reason is NOT that the ceiling is out of reach: it is
  //    that the tubes' dense side is also their hot side, so eta < 0 and the
  //    worst evolved tube face reaches M = 8.2e-4 -- on the other side of
  //    zero, not merely short of the threshold. Invert that (1000:1 density
  //    contrast at equal pressure, so the rarefied side is the hot one) and
  //    Gamma = 2 gives M = 24.9, an order of magnitude past the jet's 2.466.
  //    The point of this case is that the protection is structural, not
  //    contingent on the tubes' contacts staying mild.
  //    (The Kelvin-Helmholtz ladder is a different story again: at its own
  //    Gamma = 4/3 the ceiling is 0.444 and nothing can cross 1, but its
  //    state has eta = 0.96 and would give M = 6.44 at Gamma = 5/3.)
  test_interface_bounds_are_two_sided(
      2.0, 0.01, 0.05, 10.0, 0.05, {{0.3, -0.4, 0.1}}, {{0.2, 0.4, -0.1}},
      {{0.2, -0.2, 0.3}}, "Gamma = 2, extreme shock-tube contact");
}

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.BoundaryCorrections.PlutoHlld",
                  "[Unit][GrMhd]") {
  PUPable_reg(grmhd::ValenciaDivClean::BoundaryCorrections::PlutoHlld);
  test_matches_reference_pluto();
  test_thread_safety();
  test_interface_state();

  MAKE_GENERATOR(gen);
  using system = grmhd::ValenciaDivClean::System;

  const tuples::TaggedTuple<
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>>
      ranges{std::array{0.3, 1.0}, std::array{0.01, 0.02}};
  const tuples::TaggedTuple<hydro::Tags::GrmhdEquationOfState> volume_data{
      EquationsOfState::IdealFluid<true>{4.0 / 3.0}.promote_to_3d_eos()};

  // Conservation: the numerical flux must be single valued across the
  // interface, i.e. what the interior side sees equals what the exterior sees.
  TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
      make_not_null(&gen),
      grmhd::ValenciaDivClean::BoundaryCorrections::PlutoHlld{1.0e-30, 1.0e-8},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);

  // Factory creation + round-trip through a base-class pointer.
  const auto pluto_hlld = TestHelpers::test_factory_creation<
      evolution::BoundaryCorrection,
      grmhd::ValenciaDivClean::BoundaryCorrections::PlutoHlld>(
      "PlutoHlld:\n"
      "  MagneticFieldMagnitudeForHydro: 1.0e-30\n"
      "  LightSpeedDensityCutoff: 1.0e-8\n");
  TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
      make_not_null(&gen),
      dynamic_cast<
          const grmhd::ValenciaDivClean::BoundaryCorrections::PlutoHlld&>(
          *pluto_hlld),
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);

  using bc = grmhd::ValenciaDivClean::BoundaryCorrections::PlutoHlld;
  CHECK_FALSE(bc{1.0e-30, 1.0e-8} != bc{1.0e-30, 1.0e-8});
  CHECK(bc{1.0e-30, 1.0e-8} != bc{2.0e-30, 1.0e-8});
  CHECK(bc{1.0e-30, 1.0e-8} != bc{1.0e-30, 2.0e-8});
}
}  // namespace
