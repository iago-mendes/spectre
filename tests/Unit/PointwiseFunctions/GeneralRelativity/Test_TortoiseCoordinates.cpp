// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/TortoiseCoordinates.hpp"

namespace gr {

SPECTRE_TEST_CASE("Unit.GeneralRelativity.TortoiseCoordinates",
                  "[Unit][PointwiseFunctions]") {
  MAKE_GENERATOR(generator);
  {
    INFO("Random values");
    const pypp::SetupLocalPythonEnvironment local_python_env(
        "PointwiseFunctions/GeneralRelativity");
    pypp::check_with_random_values<3>(
        tortoise_radius_from_boyer_lindquist_minus_r_plus<DataVector>,
        "TortoiseCoordinates",
        "tortoise_radius_from_boyer_lindquist_minus_r_plus",
        {{{0.3, 10.0}, {0.3, 1.0}, {0.3, 1.0}}}, DataVector(5));
  }
  {
    INFO("Inverse");
    const size_t num_samples = 100;
    std::uniform_real_distribution<> dist_mass(0.1, 2.0);
    std::uniform_real_distribution<> dist_spin(0.0, 1.0);
    std::uniform_real_distribution<> dist_tortoise(-50.0, 100.0);
    const double mass = dist_mass(generator);
    const double dimensionless_spin = dist_spin(generator);
    const DataVector r_star = make_with_random_values<DataVector>(
                                  make_not_null(&generator),
                                  make_not_null(&dist_tortoise), num_samples) *
                              mass;
    CAPTURE(mass);
    CAPTURE(dimensionless_spin);
    CAPTURE(r_star);
    const auto r_minus_r_plus =
        boyer_lindquist_radius_minus_r_plus_from_tortoise(r_star, mass,
                                                          dimensionless_spin);
    CHECK_ITERABLE_APPROX(tortoise_radius_from_boyer_lindquist_minus_r_plus(
                              r_minus_r_plus, mass, dimensionless_spin),
                          r_star);
  }
  {
    INFO("Specific values");
    CHECK(boyer_lindquist_radius_minus_r_plus_from_tortoise(0.0, 1.0, 0.0) ==
          approx(0.55692908552214748));
  }
}

}  // namespace gr
