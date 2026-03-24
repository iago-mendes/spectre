// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Marquina.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.BoundaryCorrections.Marquina",
                  "[Unit][GrMhd]") {
  PUPable_reg(grmhd::ValenciaDivClean::BoundaryCorrections::Marquina);
  MAKE_GENERATOR(gen);

  using system = grmhd::ValenciaDivClean::System;
  namespace helpers = TestHelpers::evolution::dg;

  const tuples::TaggedTuple<hydro::Tags::GrmhdEquationOfState> volume_data{
      EquationsOfState::IdealFluid<true>{1.5, 0.0}.promote_to_3d_eos()};

  const tuples::TaggedTuple<
      helpers::Tags::Range<hydro::Tags::RestMassDensity<DataVector>>,
      helpers::Tags::Range<hydro::Tags::SpecificInternalEnergy<DataVector>>,
      helpers::Tags::Range<
          hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>>,
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3, Frame::Inertial>>>
      ranges(std::array<double, 2>{{0.1, 1.0}},    // Density
             std::array<double, 2>{{0.1, 1.0}},    // Internal Energy
             std::array<double, 2>{{0.0, 0.5}},    // Velocity
             std::array<double, 2>{{0.5, 1.0}},    // Lapse
             std::array<double, 2>{{-0.1, 0.1}});  // Shift

  for (int i = 0; i < 1000; ++i)
    TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
        make_not_null(&gen),
        grmhd::ValenciaDivClean::BoundaryCorrections::Marquina{},
        Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
        volume_data, ranges, helpers::ZeroOnSmoothSolution::Yes, 1.0e-12, true);

  const auto marquina = TestHelpers::test_factory_creation<
      evolution::BoundaryCorrection,
      grmhd::ValenciaDivClean::BoundaryCorrections::Marquina>("Marquina:");

  CHECK_FALSE(grmhd::ValenciaDivClean::BoundaryCorrections::Marquina{} !=
              grmhd::ValenciaDivClean::BoundaryCorrections::Marquina{});
}
