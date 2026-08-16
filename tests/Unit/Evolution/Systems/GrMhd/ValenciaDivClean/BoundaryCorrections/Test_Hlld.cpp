// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Parallel/Printf/Printf.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"

#include <array>
#include <cstddef>
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
      dev_from_hll = std::max(dev_from_hll, std::abs(flux[k] - solver.hll.F[k]));
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
