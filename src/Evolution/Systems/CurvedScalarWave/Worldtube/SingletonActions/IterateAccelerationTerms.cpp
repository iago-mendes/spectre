// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/IterateAccelerationTerms.hpp"

#include <cstddef>
#include <optional>
#include <tuple>

#include "Evolution/Systems/CurvedScalarWave/Worldtube/Inboxes.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SelfForce.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube {
void IterateAccelerationTerms::apply(
    gsl::not_null<Scalar<DataVector>*> acceleration_terms,
    const std::array<tnsr::I<double, Dim>, 2>& pos_vel,
    const tuples::TaggedTuple<
        gr::Tags::SpacetimeMetric<double, Dim>,
        gr::Tags::InverseSpacetimeMetric<double, Dim>,
        gr::Tags::SpacetimeChristoffelSecondKind<double, Dim>,
        gr::Tags::TraceSpacetimeChristoffelSecondKind<double, Dim>,
        Tags::TimeDilationFactor>& background,
    const tnsr::I<double, Dim, Frame::Inertial>& geodesic_acc,
    const Scalar<double>& psi_monopole, const Scalar<double>& dt_psi_monopole,
    const tnsr::i<double, Dim, Frame::Inertial>& psi_dipole, double charge,
    std::optional<double> mass, double time, std::optional<double> turn_on_time,
    std::optional<double> turn_on_interval) {
  const size_t data_size = 3;
  acceleration_terms->get() = DataVector(data_size, 0.);
  const auto& inverse_metric =
      get<gr::Tags::InverseSpacetimeMetric<double, Dim>>(background);
  const auto& dilation_factor = get<Tags::TimeDilationFactor>(background);
  const auto& vel = pos_vel.at(1);
  const auto self_force_acc =
      self_force_acceleration(dt_psi_monopole, psi_dipole, vel, charge,
                              mass.value(), inverse_metric, dilation_factor);
  for (size_t i = 0; i < Dim; ++i) {
    get(*acceleration_terms)[i] = geodesic_acc.get(i) + self_force_acc.get(i);
  }
}
}  // namespace CurvedScalarWave::Worldtube
