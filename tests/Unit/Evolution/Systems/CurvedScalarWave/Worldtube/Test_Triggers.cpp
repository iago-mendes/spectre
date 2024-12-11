// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>

#include "Evolution/Systems/CurvedScalarWave/Worldtube/RadiusFunctions.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Triggers/InsideHorizon.hpp"
#include "Framework/TestHelpers.hpp"

namespace CurvedScalarWave::Worldtube {
namespace {
void test_inside_horizon() {
  const double exp = 1.5;
  const double rinf_large = 0.1;
  const double rb = 0.1;
  const double delta = 0.01;

  const std::array<double, 4> wt_radius_params_large{
      {exp, rinf_large, rb, delta}};
  // unused
  const tnsr::I<double, 3> vel{{0., 0., 0.}};

  // particle outside the horizon
  const tnsr::I<double, 3> pos_outside{{0., 2., 0.}};
  const std::array<tnsr::I<double, 3>, 2> pos_vel_outside{{pos_outside, vel}};
  const Triggers::InsideHorizon inside_horizon_trigger{};
  CHECK(inside_horizon_trigger(pos_vel_outside, wt_radius_params_large) ==
        false);

  // particle inside horizon but worldtube sticks out
  const tnsr::I<double, 3> pos_inside{{1.95, 0., 0.}};
  const std::array<tnsr::I<double, 3>, 2> pos_vel_inside{{pos_inside, vel}};
  CHECK(inside_horizon_trigger(pos_vel_inside, wt_radius_params_large) ==
        false);

  // shrink worldtube radius to fit inside horizons
  const double rinf_small = 0.01;
  const std::array<double, 4> wt_radius_params_small{
      {exp, rinf_small, rb, delta}};

  CHECK(inside_horizon_trigger(pos_vel_inside, wt_radius_params_small) == true);
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.CurvedScalarWave.Worldtube.Triggers",
                  "[Unit][Evolution]") {
  test_inside_horizon();
}
}  // namespace
}  // namespace CurvedScalarWave::Worldtube
