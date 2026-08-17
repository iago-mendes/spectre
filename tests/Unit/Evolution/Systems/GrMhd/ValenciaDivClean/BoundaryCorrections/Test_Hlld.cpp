// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Parallel/Printf/Printf.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <string>

#include "DataStructures/TaggedTuple.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hlld.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/HlldImpl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

// The HLLD boundary correction reduces to the plain physical flux when the two
// sides agree and to the HLL flux when the fan collapses, and it is
// conservative (single-valued numerical flux at the interface). This test
// checks conservation, factory creation, serialization and option equality.
// The pointwise five-wave MUB2009 flux itself is validated bit-for-bit against
// independent reference implementations (PLUTO, E. Most's code) outside the
// unit tests, in runs-ai/mhd_marquina/hlld_xcheck/.
namespace {

// The CW test is a STATIONARY CONTACT: v is identical on both sides with
// v_n = 0, and p, v and B are all continuous -- only rho jumps. Every normal
// flux is then continuous across the interface (the advective pieces carry a
// factor v_n = 0, and what remains depends only on continuous quantities), so
// the exact solution is the initial data, unchanged, forever. HLLD inserts the
// contact explicitly and MUST reproduce that flux exactly.
//
// This drives the five-wave solver directly, with no simulation, so the debug
// loop is a fraction of a second. The primitive array layout matches Hlld.cpp's
// `build` lambda: {rho, eps, W*v_n, W*v_t1, W*v_t2, B_n, B_t1, B_t2, Ye}.

// CONSISTENCY: for identical left and right states the Riemann problem is
// trivial and ANY consistent solver must return exactly the physical flux
// F(u) -- no dissipation, no correction. Most of the RW domain is uniform, so a
// violation here would explain scatter across the whole test rather than just
// near the waves. Swept over a range of states including v_n != 0, which the
// stationary-contact test above does not reach.
//
// The user's principle for HLLD: in the worst case it should degrade to HLL,
// which is bounded. Anything far outside the data range is a bug by definition.

// "In the worst case HLLD should behave like HLL." Encode that: for ANY pair of
// states the numerical flux must lie within the envelope spanned by the two
// physical fluxes, widened by the largest dissipation the fan can supply,
// |lambda|_max * |Delta u|. HLL sits inside that envelope by construction, so
// anything outside it is a bug -- which is how the uniform-state degeneracy and
// the diverged root-find both showed up as wild scatter in the simulations.
void test_hlld_flux_stays_within_physical_bounds() {
  namespace hd = grmhd::ValenciaDivClean::BoundaryCorrections::hlld_detail;
  const ScopedFpeState fpe(false);
  const double adiabatic_index = 5.0 / 3.0;
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> rho_dist(0.05, 12.0);
  std::uniform_real_distribution<> p_dist(0.05, 60.0);
  std::uniform_real_distribution<> v_dist(-0.6, 0.6);
  std::uniform_real_distribution<> b_dist(-6.0, 6.0);
  size_t violations = 0;
  size_t hll_violations = 0;
  double worst_excess = 0.0;
  for (size_t trial = 0; trial < 4000; ++trial) {
    const double b_n = b_dist(gen);  // normal field is shared by both states
    const auto draw = [&]() {
      const double vx = v_dist(gen);
      const double vy = v_dist(gen);
      const double vz = v_dist(gen);
      const double v_sq = vx * vx + vy * vy + vz * vz;
      const double lorentz = 1.0 / sqrt(std::max(1.0 - v_sq, 1.0e-3));
      const double rho = rho_dist(gen);
      return std::array<double, 9>{rho,
                                   p_dist(gen) /
                                       ((adiabatic_index - 1.0) * rho),
                                   lorentz * vx,
                                   lorentz * vy,
                                   lorentz * vz,
                                   b_n,
                                   b_dist(gen),
                                   b_dist(gen),
                                   0.0};
    };
    const auto left = draw();
    const auto right = draw();
    hd::HLLDSolver<0> solver(left, right, adiabatic_index);
    const auto [flux, cons] = solver.solve(0.0);
    const double speed =
        std::max(std::fabs(solver.LL.lambda), std::fabs(solver.RR.lambda));
    for (size_t k = 0; k < hd::NUM; ++k) {
      const double f_lo = std::min(solver.LL.F[k], solver.RR.F[k]);
      const double f_hi = std::max(solver.LL.F[k], solver.RR.F[k]);
      const double dissipation =
          speed * std::fabs(solver.RR.U[k] - solver.LL.U[k]);
      const double scale =
          1.0 + std::fabs(f_lo) + std::fabs(f_hi) + dissipation;
      const double excess =
          std::max(flux[k] - (f_hi + dissipation),
                   (f_lo - dissipation) - flux[k]) /
          scale;
      const double f_hll = solver.hll.F[k];
      const double hll_excess =
          std::max(f_hll - (f_hi + dissipation), (f_lo - dissipation) - f_hll) /
          scale;
      if (hll_excess > 1.0e-8) {
        ++hll_violations;
      }
      if (not std::isfinite(flux[k]) or excess > 1.0e-8) {
        ++violations;
        worst_excess = std::max(worst_excess, excess);
        break;
      }
    }
  }
  Parallel::printf("  HLLD physical-bound violations: %zu / 4000 random state "
                   "pairs (worst relative excess %.3e); HLL control "
                   "violations: %zu\n",
                   violations, worst_excess, hll_violations);
  CHECK(violations == 0);
}

void test_hlld_is_consistent_for_uniform_states() {
  namespace hd = grmhd::ValenciaDivClean::BoundaryCorrections::hlld_detail;
  const ScopedFpeState fpe(false);
  size_t violations = 0;
  double worst = 0.0;
  std::array<double, 9> worst_state{};
  for (const double v_n : {0.0, 0.2, 0.4, -0.3, 0.7}) {
    for (const double v_t : {0.0, 0.3, 0.5}) {
      for (const double b_n : {0.5, 1.0, 2.4, 5.0}) {
        for (const double b_t : {0.0, 1.0, 1.6, 4.0}) {
          for (const double rho : {0.1, 1.0, 10.0}) {
            const double v_sq = v_n * v_n + 2.0 * v_t * v_t;
            if (v_sq >= 0.95) {
              continue;
            }
            const double lorentz = 1.0 / sqrt(1.0 - v_sq);
            const double pressure = 1.0;
            const double adiabatic_index = 5.0 / 3.0;
            const std::array<double, 9> state{
                rho,
                pressure / ((adiabatic_index - 1.0) * rho),
                lorentz * v_n,
                lorentz * v_t,
                lorentz * v_t,
                b_n,
                b_t,
                b_t,
                0.0};
            hd::HLLDSolver<0> solver(state, state, adiabatic_index);
            const auto [flux, cons] = solver.solve(0.0);
            for (size_t k = 0; k < hd::NUM; ++k) {
              const double deviation = std::abs(flux[k] - solver.LL.F[k]) /
                                       (1.0 + std::abs(solver.LL.F[k]));
              if (deviation > worst) {
                worst = deviation;
                worst_state = state;
              }
            }
            if (worst > 1.0e-8 and violations == 0) {
              ++violations;
            }
          }
        }
      }
    }
  }
  Parallel::printf(
      "  HLLD uniform-state consistency: worst relative deviation = %.3e%s\n",
      worst,
      worst < 1.0e-8
          ? ""
          : "   <-- INCONSISTENT (should be exactly the physical flux)");
  if (worst >= 1.0e-8) {
    Parallel::printf(
        "    worst state: rho=%g eps=%g Wv=(%g,%g,%g) B=(%g,%g,%g)\n",
                     worst_state[0], worst_state[1], worst_state[2],
                     worst_state[3], worst_state[4], worst_state[5],
                     worst_state[6], worst_state[7]);
  }
  CHECK(worst < 1.0e-8);
}

void test_hlld_reproduces_a_stationary_contact() {
  namespace hd = grmhd::ValenciaDivClean::BoundaryCorrections::hlld_detail;
  const double adiabatic_index = 5.0 / 3.0;
  const double v_t1 = 0.7;
  const double v_t2 = 0.2;
  const double lorentz = 1.0 / sqrt(1.0 - (v_t1 * v_t1 + v_t2 * v_t2));
  const double pressure = 1.0;

  for (const double field_scale : {0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0}) {
    const double b_n = 5.0 * field_scale;
    const double b_t1 = 1.0 * field_scale;
    const double b_t2 = 0.5 * field_scale;
    const auto primitives = [&](const double rho) {
      return std::array<double, 9>{rho,
                                   pressure / ((adiabatic_index - 1.0) * rho),
                                   0.0,
                                   lorentz * v_t1,
                                   lorentz * v_t2,
                                   b_n,
                                   b_t1,
                                   b_t2,
                                   0.0};
    };
    const auto left = primitives(10.0);
    const auto right = primitives(1.0);
    const double gamma = adiabatic_index;

    // The five-wave solver deliberately produces inf/nan for unphysical trial
    // states (its own masks discard them), so FP exceptions must be disabled
    // around it exactly as Hlld.cpp does in production.
    const ScopedFpeState fpe(false);
    hd::HLLDSolver<0> solver(left, right, gamma);
    const auto [flux, cons] = solver.solve(0.0);

    // For this state p, v and B are all continuous, so the two one-sided
    // physical fluxes agree and the exact answer is that common flux.
    double worst = 0.0;
    for (size_t k = 0; k < hd::NUM; ++k) {
      worst = std::max(worst, std::abs(flux[k] - solver.LL.F[k]) /
                                  (1.0 + std::abs(solver.LL.F[k])));
    }
    double dev_from_hll = 0.0;
    for (size_t k = 0; k < hd::NUM; ++k) {
      dev_from_hll =
          std::max(dev_from_hll, std::abs(flux[k] - solver.hll.F[k]));
    }

    // The exact total pressure is analytic here: p + b^2/2 with
    // b^2 = B^2/W^2 + (B.v)^2. Scan the residual the secant tries to zero
    // around that known root: if f is smooth with a sign change there, the
    // algebra is sound and the ROOT-FIND is at fault; if not, the residual is.
    const double b_dot_v = b_t1 * v_t1 + b_t2 * v_t2;
    const double b_squared =
        (b_n * b_n + b_t1 * b_t1 + b_t2 * b_t2) / (lorentz * lorentz) +
        b_dot_v * b_dot_v;
    const double exact_ptot = pressure + 0.5 * b_squared;

    CHECK(worst < 1.0e-8);
    if (worst > 1.0e-8) {
      hd::HLLDSolver<0> scan(left, right, gamma);
      for (const double frac : {0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0}) {
        const double trial = exact_ptot * frac;
        scan.rotL.update(scan.LL, trial);
        scan.rotR.update(scan.RR, trial);
        const double residual = scan.cd.update(scan.rotL, scan.rotR, trial);
          Parallel::printf("        p_tot=%11.4f (%.2fx)  f=%+.6e\n",
                         trial, frac, residual);
      }
    }
  }
}

namespace helpers = TestHelpers::evolution::dg;

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.BoundaryCorrections.Hlld",
                  "[Unit][GrMhd]") {
  PUPable_reg(grmhd::ValenciaDivClean::BoundaryCorrections::Hlld);
  test_hlld_is_consistent_for_uniform_states();
  test_hlld_flux_stays_within_physical_bounds();
  test_hlld_reproduces_a_stationary_contact();
  MAKE_GENERATOR(gen);

  using system = grmhd::ValenciaDivClean::System;

  const tuples::TaggedTuple<
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>>
      ranges{std::array{0.3, 1.0}, std::array{0.01, 0.02}};
  const tuples::TaggedTuple<hydro::Tags::GrmhdEquationOfState> volume_data{
      EquationsOfState::IdealFluid<true>{4.0 / 3.0}.promote_to_3d_eos()};

  // Conservation: the numerical flux is single valued across the interface.
  TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
      make_not_null(&gen),
      grmhd::ValenciaDivClean::BoundaryCorrections::Hlld{1.0e-30, 1.0e-8},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);

  // Factory creation + round-trip through a base-class pointer.
  const auto hlld = TestHelpers::test_factory_creation<
      evolution::BoundaryCorrection,
      grmhd::ValenciaDivClean::BoundaryCorrections::Hlld>(
      "Hlld:\n"
      "  MagneticFieldMagnitudeForHydro: 1.0e-30\n"
      "  LightSpeedDensityCutoff: 1.0e-8\n");
  TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
      make_not_null(&gen),
      dynamic_cast<const grmhd::ValenciaDivClean::BoundaryCorrections::Hlld&>(
          *hlld),
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);

  CHECK_FALSE(
      grmhd::ValenciaDivClean::BoundaryCorrections::Hlld{1.0e-30, 1.0e-8} !=
      grmhd::ValenciaDivClean::BoundaryCorrections::Hlld{1.0e-30, 1.0e-8});
  CHECK(grmhd::ValenciaDivClean::BoundaryCorrections::Hlld{1.0e-30, 1.0e-8} !=
        grmhd::ValenciaDivClean::BoundaryCorrections::Hlld{2.0e-30, 1.0e-8});
  CHECK(grmhd::ValenciaDivClean::BoundaryCorrections::Hlld{1.0e-30, 1.0e-8} !=
        grmhd::ValenciaDivClean::BoundaryCorrections::Hlld{1.0e-30, 2.0e-8});
}
}  // namespace
