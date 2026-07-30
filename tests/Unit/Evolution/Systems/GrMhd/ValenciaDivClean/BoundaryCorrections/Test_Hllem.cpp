// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/TaggedTuple.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hllem.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

// HLLEM is the HLL flux plus an eigenvector-based anti-diffusion that restores
// selected intermediate waves; in flat space it is active and in curved space
// it falls back to the (conservative) HLL flux, so this checks conservation,
// factory creation, serialization and option equality across wave-set choices.
// The anti-diffusion itself is exercised / compared against PLUTO on flat-space
// runs outside the unit tests.
namespace {
namespace helpers = TestHelpers::evolution::dg;
namespace bc = grmhd::ValenciaDivClean::BoundaryCorrections;

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.BoundaryCorrections.Hllem",
                  "[Unit][GrMhd]") {
  PUPable_reg(bc::Hllem);
  MAKE_GENERATOR(gen);

  using system = grmhd::ValenciaDivClean::System;

  const tuples::TaggedTuple<
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>>
      ranges{std::array{0.3, 1.0}, std::array{0.01, 0.02}};
  const tuples::TaggedTuple<hydro::Tags::GrmhdEquationOfState> volume_data{
      EquationsOfState::IdealFluid<true>{4.0 / 3.0}.promote_to_3d_eos()};

  for (const auto waves :
       {bc::HllemWaves::Contact, bc::HllemWaves::ContactSlow,
        bc::HllemWaves::ContactAlfven, bc::HllemWaves::All}) {
    TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
        make_not_null(&gen), bc::Hllem{waves, true, 1.0e-10, 1.0e-30, 1.0e-8},
        Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
        volume_data, ranges);
  }

  const auto hllem =
      TestHelpers::test_factory_creation<evolution::BoundaryCorrection,
                                         bc::Hllem>(
          "Hllem:\n"
          "  WavesToRestore: ContactSlow\n"
      "  UseComplementaryProjection: true\n"
          "  DegeneracyTolerance: 1.0e-10\n"
          "  MagneticFieldMagnitudeForHydro: 1.0e-30\n"
          "  LightSpeedDensityCutoff: 1.0e-8\n");
  TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
      make_not_null(&gen), dynamic_cast<const bc::Hllem&>(*hllem),
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);

  CHECK_FALSE(bc::Hllem{bc::HllemWaves::All, true, 1.0e-10, 1.0e-30, 1.0e-8} !=
              bc::Hllem{bc::HllemWaves::All, true, 1.0e-10, 1.0e-30, 1.0e-8});
  CHECK(bc::Hllem{bc::HllemWaves::All, true, 1.0e-10, 1.0e-30, 1.0e-8} !=
        bc::Hllem{bc::HllemWaves::ContactSlow, true, 1.0e-10, 1.0e-30, 1.0e-8});
  CHECK(bc::Hllem{bc::HllemWaves::All, true, 1.0e-10, 1.0e-30, 1.0e-8} !=
        bc::Hllem{bc::HllemWaves::All, true, 1.0e-9, 1.0e-30, 1.0e-8});
  CHECK(bc::Hllem{bc::HllemWaves::All, true, 1.0e-10, 1.0e-30, 1.0e-8} !=
        bc::Hllem{bc::HllemWaves::All, false, 1.0e-10, 1.0e-30, 1.0e-8});
  // per-wave (non-CPM) path is also conservative
  TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
      make_not_null(&gen),
      bc::Hllem{bc::HllemWaves::All, false, 1.0e-3, 1.0e-30, 1.0e-8},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);
}
}  // namespace
