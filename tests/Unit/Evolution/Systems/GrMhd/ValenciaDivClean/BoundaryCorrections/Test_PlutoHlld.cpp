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
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
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

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.BoundaryCorrections.PlutoHlld",
                  "[Unit][GrMhd]") {
  PUPable_reg(grmhd::ValenciaDivClean::BoundaryCorrections::PlutoHlld);
  test_matches_reference_pluto();
  test_thread_safety();

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
